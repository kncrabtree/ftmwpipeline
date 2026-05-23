"""Residual-rescue pass: re-seed missed peaks from a window's |residual|.

The rescue treats the initial fit as **completely frozen**: it subtracts
the initial-fit model from the data ONCE to get the residual, detects
candidate peaks in that residual
(:func:`ftmwpipeline.fitting.residual_screening.find_residual_peaks`), then
fits them to the residual via :func:`conservative_fit` -- the same
add-one-peak / F-test / AIC / blend-aware machinery the original fit uses,
but applied to the *residual* spectrum. The initial fit's peaks are
*never* re-fit here; they're already accounted for by the subtraction.

A subsequent **joint refit** combining the initial and rescue peak sets
(with all parameters relaxed, plus a knockout sweep over the union) is a
separate, later step. This function only finds and fits the peaks the
initial fit missed. The clean separation makes the rescue's contribution
trivially attributable: ``rescue.fit.peaks`` is exactly the set the
rescue added, fit only to what the initial fit didn't explain.

Notes
-----
* Frozen contributors must be subtracted *before* calling this function --
  the rescue's residual is taken against ``current_fit`` (the free-peak
  conservative fit), so the frozen background must already be gone from
  ``complex_spectrum``.
* ``conservative_fit`` runs an internal knockout pass on the rescue's
  own fit; ``RescueOutcome.knockouts`` carries those results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

from .peak_model import ModelPeak, model_spectrum
from .residual_screening import (
    DEFAULT_COHERENCE_CLUSTER_FWHM,
    DEFAULT_COHERENCE_RATIO_THRESHOLD,
    ResidualPeakCandidate,
    filter_by_phase_coherence,
    find_residual_peaks,
)
from .validation import calculate_chi_squared_improvement, feature_fwhm
from .window_fit import (
    DEFAULT_MAX_PEAKS,
    DEFAULT_MIN_SEPARATION_FACTOR,
    DEFAULT_SIGNIFICANCE,
    AddStep,
    ConservativeFitResult,
    KnockoutResult,
    WindowFitResult,
    _seed_peak,
    conservative_fit,
    derive_window_fit_constraints,
    fit_window,
    knockout_test,
)


__all__ = [
    "DEFAULT_CLEANUP_SIGNIFICANCE",
    "DEFAULT_MERGE_SEPARATION_FACTOR",
    "DEFAULT_RESCUE_MAX_ROUNDS",
    "DEFAULT_RESCUE_SNR_THRESHOLD",
    "DEFAULT_RESCUE_PROMINENCE_THRESHOLD",
    "RescueOutcome",
    "attempt_residual_rescue",
    "merge_close_peaks_cleanup",
    "remove_and_refit_cleanup",
]


# A small number of rounds is enough in practice -- each round costs one
# residual screen + one fit_window call per candidate. Mutually-interfering
# pathologies (w198: ~5 lines) converge in 2-3 rounds.
DEFAULT_RESCUE_MAX_ROUNDS = 3
# The detector should nominate generously -- downstream F-test / AIC gating
# decides which candidates survive. 2.5*sigma_c (~1% false alarm under
# Rayleigh noise) catches borderline cases the conservative loop's seeded
# K=1 fit may have missed.
DEFAULT_RESCUE_SNR_THRESHOLD = 2.5
DEFAULT_RESCUE_PROMINENCE_THRESHOLD = 2.0
# Post-rescue cleanup runs a remove-and-refit knockout: each peak is
# dropped and the remaining peaks are jointly re-fit; if the resulting
# chi-squared is statistically indistinguishable from the K-peak fit, the
# peak is redundant. Unlike :func:`knockout_test` (which holds other peaks
# frozen and so misses duplicates -- each duplicate "carries its share"
# when its twin is frozen), this catches the iterative duplicate-fit
# pathology where a partially-captured candidate gets re-detected and
# re-added in a subsequent round.
DEFAULT_CLEANUP_SIGNIFICANCE = DEFAULT_SIGNIFICANCE
# Threshold (in FWHM units) below which adjacent peaks are merged in the
# post-rescue cleanup. Two peaks within 1 FWHM of each other typically
# cannot be physically resolved by the line-shape model -- the rescue's
# iterative duplicate-fit pathology produces tight clusters that split
# one true line into several sub-peaks. Merging at 1 FWHM is conservative
# (genuinely close pairs at the resolution limit may also collapse, but
# they were not resolvable to start with).
DEFAULT_MERGE_SEPARATION_FACTOR = 1.0


@dataclass(frozen=True)
class RescueOutcome:
    """Result of :func:`attempt_residual_rescue`.

    Attributes
    ----------
    fit : WindowFitResult
        The fit of the rescue-added peaks ON THE RESIDUAL spectrum. Its
        ``peaks`` are exactly the lines the rescue found (the initial
        fit's peaks are NOT included here -- they've been subtracted from
        the data to produce the residual the rescue operates on).
    audit : list of AddStep
        The conservative-loop audit trail for the rescue's fit (seed,
        blend-aware re-seeds, accept / reject, etc.). Identical semantics
        to :attr:`ConservativeFitResult.audit_trail`.
    candidates : list of ResidualPeakCandidate
        Detector candidates that survived the phase-coherence filter and
        were passed to :func:`conservative_fit`.
    rejected_by_coherence : list of ResidualPeakCandidate
        Detector candidates the phase-coherence filter rejected as
        phase-rotation artifacts (typically leakage from imperfect
        neighbour fits). They never reached the rescue's conservative
        loop.
    knockouts : list of KnockoutResult
        Knockout-test results from :func:`conservative_fit` 's final pass
        on the rescue fit.
    """

    fit: WindowFitResult
    audit: List[AddStep]
    candidates: List[ResidualPeakCandidate]
    rejected_by_coherence: List[ResidualPeakCandidate]
    knockouts: List[KnockoutResult]


def _merge_cluster(cluster: List[ModelPeak]) -> ModelPeak:
    """Collapse a cluster of close peaks into one via complex amplitude sum.

    The merged peak's amplitude is ``|sum_j A_j exp(i phi_j)|`` (proper
    complex superposition -- two peaks at the same offset with opposite
    phase would cancel, which is the right physics), and its phase is the
    angle of that complex sum. The merged offset is the amplitude-weighted
    mean of the cluster offsets.
    """
    if len(cluster) == 1:
        return cluster[0]
    complex_amps = np.array(
        [pk.amplitude * np.exp(1j * pk.phase) for pk in cluster],
        dtype=np.complex128,
    )
    total = complex_amps.sum()
    weights = np.array([pk.amplitude for pk in cluster], dtype=float)
    offsets = np.array([pk.offset_mhz for pk in cluster], dtype=float)
    total_weight = float(weights.sum())
    if total_weight > 0.0:
        merged_offset = float(np.average(offsets, weights=weights))
    else:
        merged_offset = float(np.mean(offsets))
    return ModelPeak(
        amplitude=float(abs(total)),
        offset_mhz=merged_offset,
        phase=float(np.angle(total)),
    )


def merge_close_peaks_cleanup(
    offset_grid_mhz: np.ndarray,
    complex_spectrum: np.ndarray,
    rms_noise: np.ndarray,
    fit: WindowFitResult,
    tau0_us: float,
    acquisition_us: float,
    *,
    fit_kwargs_inner: dict[str, Any],
    merge_separation_factor: float = DEFAULT_MERGE_SEPARATION_FACTOR,
    significance: float = DEFAULT_CLEANUP_SIGNIFICANCE,
) -> Tuple[WindowFitResult, int]:
    """Greedy F-test-gated merge of close peak pairs.

    Iteratively: find the closest pair in the current fit; if within
    ``merge_separation_factor * fwhm``, attempt to merge them and refit
    the resulting (K-1)-peak set. The merge is accepted only when the
    K-vs-(K-1) F-test does *not* reach significance -- i.e., when the
    merged model is statistically indistinguishable from the K-peak model,
    so the two peaks were not actually carrying independent information.
    Real distinct peaks that happen to sit close together survive: their
    merged fit would significantly worsen chi-squared and the F-test
    rejects the merge.

    Returns ``(updated_fit, n_merged)``. ``n_merged`` is the number of
    successful merges (each removes one peak from the set).
    """
    if fit.n_peaks < 2:
        return fit, 0
    u = np.asarray(offset_grid_mhz, dtype=float)
    z = np.asarray(complex_spectrum, dtype=np.complex128)
    sigma = np.asarray(rms_noise, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(u.size, float(sigma))
    order = np.argsort(u)
    u, z, sigma = u[order], z[order], sigma[order]

    fwhm = feature_fwhm(fit.tau_us, acquisition_us) if fit.tau_us > 0.0 else 0.0
    if fwhm <= 0.0:
        return fit, 0
    merge_threshold = merge_separation_factor * fwhm

    current = fit
    n_merged = 0
    while current.n_peaks >= 2:
        sorted_peaks = sorted(current.peaks, key=lambda p: p.offset_mhz)
        # Closest adjacent pair (after sorting, the minimum gap must be
        # between adjacent entries).
        min_dist = float("inf")
        merge_i = -1
        for i in range(len(sorted_peaks) - 1):
            d = sorted_peaks[i + 1].offset_mhz - sorted_peaks[i].offset_mhz
            if d < min_dist:
                min_dist = d
                merge_i = i
        if merge_i < 0 or min_dist >= merge_threshold:
            break
        merged_pair = _merge_cluster(
            [sorted_peaks[merge_i], sorted_peaks[merge_i + 1]]
        )
        merged_init = (
            sorted_peaks[:merge_i] + [merged_pair] + sorted_peaks[merge_i + 2:]
        )
        refit = fit_window(
            u, z, sigma, merged_init, tau0_us, acquisition_us,
            **fit_kwargs_inner,
        )
        if not refit.success:
            break
        # F-test: simpler=merged (K-1 peaks), complex=current (K peaks).
        # If pre-merge is significantly better, the merged peaks were
        # really distinct -- back out the merge and stop.
        p_value, _, _ = calculate_chi_squared_improvement(
            refit.chi_squared,
            current.chi_squared,
            3,
            current.n_data,
            current.n_params,
        )
        if p_value < significance:
            break
        current = refit
        n_merged += 1
    return current, n_merged


def remove_and_refit_cleanup(
    offset_grid_mhz: np.ndarray,
    complex_spectrum: np.ndarray,
    rms_noise: np.ndarray,
    fit: WindowFitResult,
    tau0_us: float,
    acquisition_us: float,
    *,
    fit_kwargs_inner: dict[str, Any],
    significance: float = DEFAULT_CLEANUP_SIGNIFICANCE,
) -> Tuple[WindowFitResult, int]:
    """Iteratively drop redundant peaks (K → K-1 refit, keep if improvement
    still significant); return ``(updated_fit, n_dropped)``.

    For each peak in the current fit, simulate "remove peak i, refit the
    remaining K-1". If the K-peak vs (K-1)-peak F-test fails to clear
    ``significance``, peak i is redundant -- the remaining peaks can absorb
    its contribution. Drop the worst offender (largest p-value) and repeat
    until every remaining peak is supported. This is the cleanup
    :func:`knockout_test` cannot do, since freezing other peaks during a
    knockout makes each duplicate "look supported" individually.

    The cleanup re-uses the same ``fit_kwargs_inner`` the rescue's trial
    fits used -- tau bounds, amp bounds, penalty weights -- so the refit
    converges on the same physical landscape (no surprise solutions that
    only show up because the constraints differ).
    """
    u = np.asarray(offset_grid_mhz, dtype=float)
    z = np.asarray(complex_spectrum, dtype=np.complex128)
    sigma = np.asarray(rms_noise, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(u.size, float(sigma))
    order = np.argsort(u)
    u, z, sigma = u[order], z[order], sigma[order]

    current = fit
    n_dropped = 0
    while current.n_peaks > 0:
        worst_p_value = -1.0
        worst_idx = -1
        worst_refit: Optional[WindowFitResult] = None
        for i in range(current.n_peaks):
            reduced_init = [pk for j, pk in enumerate(current.peaks) if j != i]
            if not reduced_init:
                # Compare K=1 fit to K=0 (null model): use the null chi-squared
                # directly rather than calling fit_window with an empty list.
                null_chi2 = float(
                    np.sum(np.abs(z / (sigma / np.sqrt(2.0))) ** 2)
                )
                # F-test convention: ``calculate_chi_squared_improvement``
                # takes ``(simpler, complex)`` chi-squared. ``current`` is the
                # complex (K-peak) model; the reduced model is the K=0 null.
                p_value, _, _ = calculate_chi_squared_improvement(
                    null_chi2,
                    current.chi_squared,
                    3,
                    current.n_data,
                    current.n_params,
                )
                if p_value > worst_p_value:
                    worst_p_value = p_value
                    worst_idx = i
                    worst_refit = None  # special: drop to empty fit
                continue
            refit = fit_window(
                u, z, sigma, reduced_init, tau0_us, acquisition_us,
                **fit_kwargs_inner,
            )
            if not refit.success:
                continue
            # Simpler (K-1) chi-squared first, then complex (K).
            p_value, _, _ = calculate_chi_squared_improvement(
                refit.chi_squared,
                current.chi_squared,
                3,
                current.n_data,
                current.n_params,
            )
            if p_value > worst_p_value:
                worst_p_value = p_value
                worst_idx = i
                worst_refit = refit
        if worst_idx < 0 or worst_p_value < significance:
            # Every remaining peak is supported (worst-case removal still
            # passes the F-test); stop.
            break
        n_dropped += 1
        if worst_refit is None:
            # Dropped the last peak -- produce an empty fit.
            current = fit_window(
                u, z, sigma, [], tau0_us, acquisition_us, **fit_kwargs_inner,
            )
        else:
            current = worst_refit
    return current, n_dropped


def attempt_residual_rescue(
    offset_grid_mhz: np.ndarray,
    complex_spectrum: np.ndarray,
    rms_noise: np.ndarray,
    current_fit: WindowFitResult,
    tau0_us: float,
    acquisition_us: float,
    *,
    snr_threshold: float = DEFAULT_RESCUE_SNR_THRESHOLD,
    prominence_threshold: float = DEFAULT_RESCUE_PROMINENCE_THRESHOLD,
    significance: float = DEFAULT_SIGNIFICANCE,
    min_separation_factor: float = DEFAULT_MIN_SEPARATION_FACTOR,
    max_peaks: int = DEFAULT_MAX_PEAKS,
    coherence_cluster_fwhm: float = DEFAULT_COHERENCE_CLUSTER_FWHM,
    coherence_ratio_threshold: float = DEFAULT_COHERENCE_RATIO_THRESHOLD,
    conservative_kwargs: Optional[dict[str, Any]] = None,
) -> RescueOutcome:
    """Find peaks the initial fit missed and fit them to its residual.

    The initial fit is treated as **completely frozen**. The function
    subtracts ``current_fit`` 's model from ``complex_spectrum`` ONCE,
    then runs :func:`conservative_fit` on the resulting residual. The
    rescue's returned ``fit`` contains *only* the lines the rescue added,
    fit purely to the residual -- the initial fit's peaks are NOT
    refitted, do NOT appear in ``RescueOutcome.fit``, and exist only as
    the implicit "background" that produced the residual.

    A subsequent **joint refit** of (initial + rescue) peaks with all
    parameters relaxed (plus a knockout sweep over the union) is a
    separate, later orchestration step -- not done here. This separation
    makes the rescue's contribution exactly attributable: the rescue's
    model evaluated on the window grid is what the rescue added to the
    initial model.

    Parameters
    ----------
    offset_grid_mhz
        Window's signed baseband-offset grid (any order; sorted internally
        by :func:`conservative_fit`).
    complex_spectrum
        Active-FT window data on the grid, with any frozen-contributor
        background already subtracted.
    rms_noise
        Per-bin complex noise RMS (scalar broadcast or per-bin array).
    current_fit
        The initial conservative-fit result for this window. Its peaks and
        ``tau_us`` define the residual the rescue operates on; they are
        otherwise untouched.
    tau0_us, acquisition_us
        Same values used by the initial fit. Forwarded to the rescue's
        :func:`conservative_fit` so the inner constraints match.
    snr_threshold, prominence_threshold
        Detector knobs (see :func:`find_residual_peaks`).
    significance, min_separation_factor, max_peaks
        Forwarded to :func:`conservative_fit`. The defaults here are
        deliberately permissive on the peak-count cap (``max_peaks``);
        residual screens often nominate many borderline candidates and
        the F-test should be the gate.
    conservative_kwargs
        Additional keyword arguments forwarded to :func:`conservative_fit`
        (e.g. ``tau_apodization_us``, ``max_decay_factor``,
        ``phase_penalty_lambda``). Pass the same options the initial fit
        received so the rescue's per-trial fits enforce the same physics.
    """
    u = np.asarray(offset_grid_mhz, dtype=float)
    z = np.asarray(complex_spectrum, dtype=np.complex128)
    sigma = np.asarray(rms_noise, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(u.size, float(sigma))

    # Compute the residual ONCE, from the (frozen) initial fit. Use the
    # initial fit's own peaks + tau here -- this is the actual model the
    # initial fit produced, regardless of whether its tau is physical.
    initial_model = model_spectrum(
        u, current_fit.peaks, current_fit.tau_us, acquisition_us
    )
    residual = z - initial_model

    # Tau policy for the rescue (basis and frozen-tau refit):
    # Default to the initial fit's tau (it's the LSQ-converged value for
    # the actual lines in this window). Override to the apodization tau
    # only when initial.tau is pegged at the lower bound -- that is the
    # signature of a broken initial fit (LSQ over-narrowed tau to absorb
    # unmodeled-peak residual, as in w198 with tau=1us at the bound),
    # and using that broken tau as the coherence basis would
    # under-project real residual peaks and reject them.
    ckwargs_in = conservative_kwargs or {}
    apodization_us = ckwargs_in.get("tau_apodization_us")
    max_decay_factor_in = ckwargs_in.get("max_decay_factor", 5.0)
    rescue_tau_us = float(current_fit.tau_us)
    if apodization_us and apodization_us > 0.0 and max_decay_factor_in > 0.0:
        tau_lower_bound = float(apodization_us) / float(max_decay_factor_in)
        # 5% tolerance for "at the lower bound"
        if current_fit.tau_us <= 1.05 * tau_lower_bound:
            rescue_tau_us = float(apodization_us)
    rescue_fwhm = (
        feature_fwhm(rescue_tau_us, acquisition_us)
        if rescue_tau_us > 0.0
        else 0.0
    )
    raw_candidates = find_residual_peaks(
        u, residual, sigma,
        snr_threshold=snr_threshold,
        prominence_threshold=prominence_threshold,
        fwhm_mhz=rescue_fwhm if rescue_fwhm > 0.0 else None,
    )
    # Phase-coherence filter: drop isolated candidates whose complex
    # projection onto a Lorentzian basis at their offset doesn't recover
    # the detected magnitude SNR -- those are phase-rotation artifacts
    # from imperfect neighbour fits, not real missed lines. Clustered
    # candidates (other candidate within ``coherence_cluster_fwhm * FWHM``)
    # pass through so the blend-aware seeder can handle real close pairs.
    if raw_candidates and rescue_fwhm > 0.0:
        candidates, rejected_by_coherence = filter_by_phase_coherence(
            raw_candidates, u, residual, sigma,
            rescue_tau_us, acquisition_us,
            fwhm_mhz=rescue_fwhm,
            cluster_threshold_fwhm=coherence_cluster_fwhm,
            coherence_ratio_threshold=coherence_ratio_threshold,
        )
    else:
        candidates = list(raw_candidates)
        rejected_by_coherence = []
    candidate_offsets = [c.frequency_mhz for c in candidates]

    # Run conservative_fit on the residual with tau FROZEN at the initial
    # fit's value. Rationale: rescue candidates are (putative) molecular
    # lines from the SAME experiment, so they share the same physical
    # line shape -- there is no information in the residual to re-fit
    # tau, and letting it float pegs at the apodization, distorts the
    # line shape, and breaks the LSQ for partial-capture candidates
    # (see w269: with tau floating, the seed converges to success=False
    # at amp~0; with tau frozen at initial.tau, the seed correctly
    # converges to amp~0 with success=True and the conservative loop
    # proceeds to F-test-reject the non-physical candidate).
    ckwargs: dict[str, Any] = dict(conservative_kwargs or {})
    # tau_apodization_us is irrelevant when tau is frozen; drop it.
    ckwargs.pop("tau_apodization_us", None)
    if not candidate_offsets:
        empty = conservative_fit(
            u, residual, sigma, [], rescue_tau_us, acquisition_us,
            fit_tau=False, **ckwargs,
        )
        return RescueOutcome(
            fit=empty.fit,
            audit=empty.audit_trail,
            candidates=candidates,
            rejected_by_coherence=rejected_by_coherence,
            knockouts=empty.knockouts,
        )

    rescue_result: ConservativeFitResult = conservative_fit(
        u, residual, sigma, candidate_offsets,
        rescue_tau_us, acquisition_us,
        fit_tau=False,
        significance=significance,
        min_separation_factor=min_separation_factor,
        max_peaks=max_peaks,
        **ckwargs,
    )
    return RescueOutcome(
        fit=rescue_result.fit,
        audit=rescue_result.audit_trail,
        candidates=candidates,
        rejected_by_coherence=rejected_by_coherence,
        knockouts=rescue_result.knockouts,
    )
