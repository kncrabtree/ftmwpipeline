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
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .peak_model import ModelPeak, model_spectrum
from .residual_screening import (
    ResidualPeakCandidate,
    find_residual_peaks,
)
from .validation import (
    DEFAULT_N_EFF_KIND,
    calculate_aicc,
    calculate_chi_squared_improvement,
    calculate_noise_weighted_chi2,
    effective_sample_size,
    feature_fwhm,
)
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
    "DEFAULT_N_EFF_KIND",
    "DEFAULT_RESCUE_MAX_ROUNDS",
    "DEFAULT_STRUCTURAL_MERGE_FACTOR",
    "DEFAULT_RESCUE_SNR_THRESHOLD",
    "DEFAULT_RESCUE_PROMINENCE_THRESHOLD",
    "ConsolidatedRescueOutcome",
    "RescueOutcome",
    "RescueRoundDiagnostics",
    "attempt_residual_rescue",
    "iterative_aicc_cleanup",
    "merge_close_peaks_cleanup",
    "remove_and_refit_cleanup",
    "rescue_and_consolidate",
]


# Per-window safety cap on rescue rounds. Each round costs one residual
# screen + one fit_window call per candidate. Calibrated on the 2638
# fixture (50-window survey, every 7th window): every window terminates
# naturally at "no candidates" by round 3; the extra two rounds of
# headroom are a safety net, not a working regime. Mutually-interfering
# pathologies (w198: ~5 lines) converge in 2-3 rounds.
DEFAULT_RESCUE_MAX_ROUNDS = 5
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
# Outer threshold (in FWHM units) above which adjacent peaks are not
# considered for merging. The inner tier (DEFAULT_STRUCTURAL_MERGE_FACTOR)
# merges sub-resolution pairs unconditionally; the AICc-gated outer tier
# fires for separations in [structural, outer] FWHM when AICc strictly
# prefers (K-1). Set equal to the structural threshold to disable the
# AICc-gated tier entirely -- pairs in [0.5, 1.0] FWHM are real close
# pairs whose collapse-or-keep call needs a different statistic than
# AICc-with-information-weighted-n_eff (the latter activates AICc out
# of the unidentifiable regime, where it merges real close pairs on
# chi-squared evidence alone without a phase-degeneracy penalty to
# distinguish them from duplicate-pair overfit).
DEFAULT_MERGE_SEPARATION_FACTOR = 0.5
# Inner / structural threshold: peaks closer than this fraction of the
# FWHM are merged unconditionally (no AICc test). They are physically
# unresolvable by a Lorentzian-only model and any "two-peak" fit at
# sub-resolution separations is a numerical artifact, not a real
# doublet. Catches the duplicate-pair overfit pathology (w148: 0.04
# MHz separation at FWHM ~0.1 MHz).
DEFAULT_STRUCTURAL_MERGE_FACTOR = 0.5
# Effective-sample-size weighting for the AICc-with-n_eff gates. The
# canonical definition lives in :mod:`ftmwpipeline.fitting.validation` so the
# merge cleanup (this module) and the knockout test
# (:mod:`ftmwpipeline.fitting.window_fit`) share a single source of truth;
# re-exported here for callers that imported the name from this module
# before the lift.


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
        Detector candidates the rescue passed to :func:`conservative_fit`.
    knockouts : list of KnockoutResult
        Knockout-test results from :func:`conservative_fit` 's final pass
        on the rescue fit.
    """

    fit: WindowFitResult
    audit: List[AddStep]
    candidates: List[ResidualPeakCandidate]
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
    structural_merge_factor: float = DEFAULT_STRUCTURAL_MERGE_FACTOR,
    significance: float = DEFAULT_CLEANUP_SIGNIFICANCE,
    n_eff_kind: str = DEFAULT_N_EFF_KIND,
) -> Tuple[WindowFitResult, int]:
    """Two-tier merge cleanup for close peak pairs.

    Iteratively finds the closest adjacent pair and decides whether to
    collapse it into a single peak. Two tiers, keyed on the pair's
    fractional FWHM separation:

    1. **Sub-resolution** (separation < ``structural_merge_factor *
       fwhm``): merge unconditionally. The Lorentzian-only model
       physically cannot distinguish these from a single peak, so any
       LSQ that converges to two separate peaks at this scale is a
       numerical artifact (the duplicate-pair-overfit pathology from
       w148/w269 in the 2638 fixture).
    2. **Above resolution** (``structural_merge_factor * fwhm`` <=
       separation < ``merge_separation_factor * fwhm``): merge only when
       AICc strictly prefers the (K-1)-peak model. ``n_eff`` (Kish on
       ``|model|^2`` by default) collapses to roughly K times the per-
       peak FWHM-in-bins, which on narrow features can put both AICc
       (K) and AICc(K-1) at ``+inf`` (model not identifiable). Tied AICc
       is read as "no evidence for merge" and preserves the K-peak fit
       -- this is the structural protection for real close pairs (w198
       outer shoulders at ~1 FWHM from inner peaks) the earlier
       AICc-only gate over-merged.

    The (K-1) refit locks tau at the current K-peak fit's value (tau is
    effectively a dataset-shared parameter; a single-window refit must
    not get the extra knob of broadening tau to absorb the merged
    peak's contribution). Matches the convention in
    :func:`~ftmwpipeline.fitting.window_fit.knockout_test` and
    :func:`iterative_aicc_cleanup`.

    ``significance`` is retained on the signature but is not used as a
    gate threshold.

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

    # Tau is effectively a dataset-shared parameter (transit time x natural
    # lifetime); a single-window (K-1) refit must not get the extra knob of
    # broadening tau to absorb the dropped peak's contribution. Lock tau at
    # the current K-fit value for every (K-1) trial -- matches the
    # convention used by ``knockout_test`` and ``iterative_aicc_cleanup``.
    refit_kwargs: dict[str, Any] = dict(fit_kwargs_inner)
    refit_kwargs["fit_tau"] = False

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
            u, z, sigma, merged_init, float(current.tau_us), acquisition_us,
            **refit_kwargs,
        )
        if not refit.success:
            break

        # Tier 1: sub-resolution -> merge unconditionally. The K-peak
        # fit at this scale is a numerical artifact; no statistical test
        # can disambiguate it from a single peak.
        if min_dist < structural_merge_factor * fwhm:
            current = refit
            n_merged += 1
            continue

        # Tier 2: above-resolution -> AICc-with-n_eff test, with
        # REJECT-on-tie. n_eff comes from the more-complex (K-peak)
        # model's magnitude (Kish); both AICc evaluations share it.
        # ``>=`` (rather than ``>``) makes the gate "merge only when
        # K-1 is strictly better"; ties (both AICc finite-equal or both
        # +inf because n_eff < k+1) preserve the K-peak fit, which is
        # the structural protection for real close pairs the AICc-only
        # gate over-merged.
        n_eff = effective_sample_size(
            current.fitted_spectrum, kind=n_eff_kind, sigma=sigma,
        )
        aicc_k = calculate_aicc(current.chi_squared, current.n_params, n_eff)
        aicc_km1 = calculate_aicc(refit.chi_squared, refit.n_params, n_eff)
        if aicc_km1 >= aicc_k:
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


def iterative_aicc_cleanup(
    offset_grid_mhz: np.ndarray,
    complex_spectrum: np.ndarray,
    rms_noise: np.ndarray,
    fit: WindowFitResult,
    acquisition_us: float,
    *,
    fit_kwargs_inner: dict[str, Any],
    n_eff_kind: str = DEFAULT_N_EFF_KIND,
) -> Tuple[WindowFitResult, int]:
    """Iteratively drop the worst AICc-with-n_eff offender until every
    remaining peak is supported.

    The per-peak :func:`knockout_test` produces accurate diagnostics in
    isolation (each peak's "supported" flag, ``aicc_delta``, ``n_eff``)
    but **cannot** be used non-iteratively to drop multiple peaks: when N
    peaks duplicate a single real feature, each individually fails its
    refit test (the surviving twins absorb), so an all-at-once drop kills
    the whole cluster including the underlying real signal (the w273
    failure mode: a real isolated peak gets rescue-assigned 2 candidates,
    both fail knockout, both are dropped, the feature is now unmodeled).

    The iterative form drops the *single* worst offender (lowest
    ``aicc_delta``), refits the surviving (K-1) peaks, then re-runs the
    per-peak test on the smaller set. The dropped peak's contribution is
    redistributed among the surviving peaks during the refit, so the
    remaining peaks' aicc_delta values reflect the *new* model state.
    When the cluster was N duplicates of one real peak, the first drop
    lets the surviving twin re-converge to full amplitude; the next
    iteration's test now compares "this one full-amplitude peak" vs the
    null, which is strongly supported for a real feature. The loop
    terminates with one surviving peak.

    Tau is locked at the K-fit's ``tau_us`` for every (K-1) refit (per
    Phase 2 spec). REJECT-on-tie: ``aicc_km1 >= aicc_k`` preserves the
    peak; only ``aicc_km1 < aicc_k`` drops it. The empty-set case (drop
    the last surviving peak) is decided by comparing AICc(K=1 fit) to
    AICc(K=0 null on data); the K=1 vs K=0 comparison uses the K-fit
    model's ``fitted_spectrum`` for ``n_eff`` (we have no model to base
    it on otherwise).

    Returns ``(cleaned_fit, n_dropped)``. ``cleaned_fit`` is the
    last refit (or the input ``fit`` if nothing was dropped); ``n_dropped``
    is the number of peaks the iterative loop removed.
    """
    if fit.n_peaks == 0:
        return fit, 0

    u = np.asarray(offset_grid_mhz, dtype=float)
    z = np.asarray(complex_spectrum, dtype=np.complex128)
    sigma = np.asarray(rms_noise, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(u.size, float(sigma))
    order = np.argsort(u)
    u, z, sigma = u[order], z[order], sigma[order]

    refit_kwargs: dict[str, Any] = dict(fit_kwargs_inner)
    refit_kwargs["fit_tau"] = False

    current = fit
    n_dropped = 0
    while current.n_peaks > 0:
        tau_locked = float(current.tau_us)
        n_eff = effective_sample_size(
            current.fitted_spectrum, kind=n_eff_kind, sigma=sigma,
        )
        aicc_k = calculate_aicc(current.chi_squared, current.n_params, n_eff)

        worst_aicc_km1 = float("inf")
        worst_idx = -1
        worst_refit: Optional[WindowFitResult] = None
        worst_is_null = False

        for i in range(current.n_peaks):
            kept = [pk for j, pk in enumerate(current.peaks) if j != i]
            if not kept:
                null_chi2 = calculate_noise_weighted_chi2(z, sigma)
                aicc_km1 = calculate_aicc(null_chi2, 0, n_eff)
                if aicc_km1 < worst_aicc_km1:
                    worst_aicc_km1 = aicc_km1
                    worst_idx = i
                    worst_refit = None
                    worst_is_null = True
                continue
            refit = fit_window(
                u, z, sigma, kept, tau_locked, acquisition_us,
                **refit_kwargs,
            )
            if not refit.success:
                continue
            aicc_km1 = calculate_aicc(refit.chi_squared, refit.n_params, n_eff)
            if aicc_km1 < worst_aicc_km1:
                worst_aicc_km1 = aicc_km1
                worst_idx = i
                worst_refit = refit
                worst_is_null = False

        # REJECT-on-tie: keep the peak unless AICc(K-1) is strictly
        # better than AICc(K). When both diverge to +inf (model not
        # identifiable for either K), ``inf < inf`` is False -> stop.
        if worst_idx < 0 or not (worst_aicc_km1 < aicc_k):
            break

        n_dropped += 1
        if worst_is_null:
            # Dropped the last peak; the cleaned fit is the null model.
            # Produce a zero-peak fit_window result so the loop's return
            # has the same type as every other branch.
            current = fit_window(
                u, z, sigma, [], tau_locked, acquisition_us,
                **refit_kwargs,
            )
            break
        assert worst_refit is not None
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
    conservative_kwargs: Optional[dict[str, Any]] = None,
    shape_error_epsilon: float = 0.0,
    excluded_offsets: Optional[Sequence[float]] = None,
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
    shape_error_epsilon : float, default 0.0
        Fractional lineshape-model error per unit parent amplitude.
        Inflates the screening-pipeline noise floor by
        ``epsilon * |current_model(f)|`` so candidates falling under
        existing strong peaks must exceed the expected irreducible
        Lorentzian-vs-true-shape residual to be considered. Calibrated
        per dataset (the chi^2_r ~ SNR^2 regression slope: see
        ``scratch/stage5-validation/diag_voigt_hypothesis.py``; ~0.0125
        for the 2638 fixture). ``0.0`` (default) preserves the canonical
        Stage 2 noise model and the pre-existing behaviour. The fitter
        (LSQ inside :func:`conservative_fit`) always uses the
        un-inflated sigma; inflation is a screening tool, not a fitting
        one.
    excluded_offsets : sequence of float, optional
        Additional offsets (MHz) the rescue must NOT re-propose
        candidates near. Combined with ``current_fit.peaks`` 's offsets
        into a single rejection list: any detector candidate whose
        frequency falls within +/-1 grid bin of either a *currently-
        fitted peak* or an *entry of this list* is dropped before
        going to :func:`conservative_fit`. The fitted-peak rejection
        prevents the screening pipeline from re-nominating the same
        line the initial fit already explains (residual structure
        under a fitted peak is shape-error / leakage, not a missed
        line). The explicit list is used by
        :func:`rescue_and_consolidate` to suppress candidates that a
        previous round nominated and the joint-refit +
        iterative-cleanup later rejected -- those candidates are not
        going to survive the next round's gate either, and
        re-detecting them clutters the audit trail. ``None`` (default)
        disables the explicit blacklist; the fitted-peak rejection
        always applies.
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
    # When a Stage 2b calibration is available, use ``tau_maj`` for every
    # window regardless of the initial fit's outcome -- the calibration is
    # the global physical tau and the rescue should anchor to it, not the
    # broken per-window LSQ tau. (Phase-3 step 7 of the tau-calibration
    # plan; closes the channel the cross-fixture-validation
    # "broken-initial-fit pathology" identified.) Legacy fallback: when no
    # ``tau_maj`` is plumbed, default to the initial fit's tau and override
    # to the apodization tau only when initial.tau is pegged at the lower
    # bound -- the original w198-style heuristic.
    ckwargs_in = conservative_kwargs or {}
    tau_maj_us = ckwargs_in.get("tau_maj_us")
    if tau_maj_us is not None and tau_maj_us > 0.0:
        rescue_tau_us = float(tau_maj_us)
    else:
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

    # Shape-error-aware sigma inflation for the screening pipeline.
    # When epsilon > 0, the per-bin noise floor seen by the detector +
    # phase-coherence filter grows by ``epsilon * |initial_model|`` --
    # representing the irreducible Lorentzian-vs-true-lineshape residual
    # that scales with parent amplitude (see the chi^2_r ~ SNR^2
    # regression in scratch/stage5-validation/diag_voigt_hypothesis.py).
    # Candidates sitting under existing strong peaks must clear this
    # inflated floor to be considered; far-from-peak candidates see the
    # canonical noise. conservative_fit further down still uses the raw
    # sigma -- the inflation gates which candidates enter the fit, not
    # how they fit.
    if shape_error_epsilon > 0.0:
        model_mag = np.abs(initial_model)
        sigma_screen = np.sqrt(sigma * sigma + (shape_error_epsilon * model_mag) ** 2)
    else:
        sigma_screen = sigma
    raw_candidates = find_residual_peaks(
        u, residual, sigma_screen,
        snr_threshold=snr_threshold,
        prominence_threshold=prominence_threshold,
        fwhm_mhz=rescue_fwhm if rescue_fwhm > 0.0 else None,
    )
    # find_residual_peaks gates on the *median* sigma_c (scipy's
    # find_peaks takes a scalar height by design here). With per-bin
    # sigma inflation that's much larger at a few peak-center bins than
    # at the median bin, the median-based gate is barely shifted -- the
    # per-bin filter has to run here as a post-step. Drop any candidate
    # whose magnitude fails the inflated-sigma SNR at its own bin.
    if shape_error_epsilon > 0.0 and raw_candidates:
        kept = []
        for c in raw_candidates:
            bin_sigma_c = sigma_screen[c.bin_index] / np.sqrt(2.0)
            if c.magnitude >= snr_threshold * bin_sigma_c:
                kept.append(c)
        raw_candidates = kept
    # Locality rejection: drop any candidate within +/-1 grid bin of a
    # currently-fitted peak OR an explicit blacklist entry. Fitted-peak
    # locality means the candidate is sitting under an existing peak --
    # the residual signal there is shape-error / leakage, not a missed
    # line -- and re-nominating it would only invite the joint-refit /
    # iterative-cleanup chain to drop it again. The explicit blacklist
    # is the across-rounds memory in :func:`rescue_and_consolidate`
    # (rescue peaks that previous rounds nominated and the cleanup
    # later rejected).
    rejection_offsets: List[float] = [pk.offset_mhz for pk in current_fit.peaks]
    if excluded_offsets:
        rejection_offsets.extend(float(x) for x in excluded_offsets)
    if rejection_offsets and raw_candidates and u.size >= 2:
        u_sorted = np.sort(u)
        df_mhz = float(np.min(np.diff(u_sorted)))
        if df_mhz > 0.0:
            raw_candidates = [
                c for c in raw_candidates
                if not any(
                    abs(c.frequency_mhz - x) <= df_mhz
                    for x in rejection_offsets
                )
            ]
    candidates = list(raw_candidates)
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
    # tau anchors are irrelevant when tau is frozen; drop them. The
    # conservative_fit inside the rescue uses ``rescue_tau_us`` directly
    # (held fixed via ``fit_tau=False``); the anchoring penalty and bounds
    # would simply waste residual elements.
    ckwargs.pop("tau_apodization_us", None)
    ckwargs.pop("tau_maj_us", None)
    ckwargs.pop("sigma_tau_us", None)
    if not candidate_offsets:
        empty = conservative_fit(
            u, residual, sigma, [], rescue_tau_us, acquisition_us,
            fit_tau=False, **ckwargs,
        )
        return RescueOutcome(
            fit=empty.fit,
            audit=empty.audit_trail,
            candidates=candidates,
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
        knockouts=rescue_result.knockouts,
    )


# ---------------------------------------------------------------------------
# Recursive rescue + joint refit + knockout consolidation (the "B-loop")
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RescueRoundDiagnostics:
    """Per-round bookkeeping for :func:`rescue_and_consolidate`.

    One record per rescue round actually executed. ``accepted`` is true when
    this round's joint refit (post-knockout) replaced the previous round's
    fit as the new ``current``. A round with ``accepted=False`` means the
    rescue produced peaks but the joint refit either failed to converge or
    its knockout sweep dropped everything -- in that case the previous
    round's fit is retained and the loop terminates.

    Attributes
    ----------
    round_idx : int
        Zero-based round counter.
    rescue : RescueOutcome
        The :func:`attempt_residual_rescue` result for this round.
    joint_fit : WindowFitResult or None
        The thawed-everything joint refit on the union of the previous
        round's peaks and the rescue's accepted peaks. ``None`` when the
        rescue accepted no peaks (the round terminates immediately).
    joint_knockouts : list of KnockoutResult
        Knockout-sweep results on ``joint_fit``.
    pruned_fit : WindowFitResult or None
        The joint refit after dropping any peaks the knockout sweep flagged
        as unsupported and refitting on the surviving subset. Equal to
        ``joint_fit`` (same identity) when nothing was pruned; ``None`` when
        pruning would have emptied the model.
    pruned_knockouts : list of KnockoutResult
        Knockout-sweep results on ``pruned_fit`` (the final consolidated
        knockouts for this round). Empty when ``pruned_fit is None``.
    n_initial_peaks : int
        Number of peaks the round inherited from the previous round (used
        as the origin marker: indices in ``joint_fit.peaks[:n_initial_peaks]``
        came from the previous round; the rest came from this round's
        rescue).
    n_rescue_added : int
        Number of peaks the rescue's conservative fit accepted.
    n_pruned_total : int
        Number of peaks the knockout sweep on ``joint_fit`` dropped.
    n_pruned_rescue_origin : int
        Of the pruned peaks, how many were rescue-origin (i.e., added this
        round). This is the **failsafe diagnostic**: a high count is the
        signal that the joint refit did not escape a pathological basin and
        is undoing the rescue's contribution. v1 logs it but does not act
        on it; a future fallback to A-mode for that window would be
        triggered here.
    n_merged : int
        Number of close-peak pairs the AICc-with-n_eff merge cleanup
        collapsed in the post-knockout fit. The merge addresses the
        duplicate-pair-overfit pathology that knockout cannot catch (each
        duplicate looks individually supported when its twin is frozen).
    chi2_before, chi2_after : float
        Noise-weighted chi-squared of the previous round's fit and of the
        consolidated (post-pruning) fit, both evaluated against the same
        ``complex_spectrum``. ``chi2_after`` equals ``chi2_before`` when
        the round was not accepted.
    tau_us_before, tau_us_after : float
        Shared decay constant before and after the round.
    accepted : bool
        Whether this round's contribution replaced the previous round's fit.
    reason : str
        Free-text note (which termination case, etc.).
    """

    round_idx: int
    rescue: RescueOutcome
    joint_fit: Optional[WindowFitResult]
    joint_knockouts: List[KnockoutResult]
    pruned_fit: Optional[WindowFitResult]
    pruned_knockouts: List[KnockoutResult]
    n_initial_peaks: int
    n_rescue_added: int
    n_pruned_total: int
    n_pruned_rescue_origin: int
    chi2_before: float
    chi2_after: float
    tau_us_before: float
    tau_us_after: float
    accepted: bool
    reason: str = ""
    n_merged: int = 0


@dataclass(frozen=True)
class ConsolidatedRescueOutcome:
    """Result of :func:`rescue_and_consolidate`.

    Attributes
    ----------
    fit : ConservativeFitResult
        The final consolidated fit, suitable for slotting into a
        :class:`WindowOutcome` 's ``fit`` field. Its ``audit_trail`` is the
        *initial* fit's audit (the rescue loop is a separate phase that
        does not extend the conservative loop's trail); its ``knockouts``
        reflect the final consolidated peak set. When the loop accepted
        nothing, this is the input ``initial_fit`` returned verbatim.
    initial_fit : ConservativeFitResult
        The input initial fit, preserved for diagnostics (e.g. comparing
        chi-squared before and after the rescue chain).
    rounds : list of RescueRoundDiagnostics
        Per-round diagnostics, in execution order. May be empty (when the
        initial fit's residual triggered no candidates on the first round).
    terminated_reason : str
        Why the loop stopped (``"no candidates"``, ``"joint failed"``,
        ``"all pruned"``, ``"max rounds reached"``).
    """

    fit: ConservativeFitResult
    initial_fit: ConservativeFitResult
    rounds: List[RescueRoundDiagnostics]
    terminated_reason: str


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def rescue_and_consolidate(
    offset_grid_mhz: np.ndarray,
    complex_spectrum: np.ndarray,
    rms_noise: np.ndarray,
    initial_fit: ConservativeFitResult,
    tau0_us: float,
    acquisition_us: float,
    *,
    max_rescue_rounds: int = DEFAULT_RESCUE_MAX_ROUNDS,
    snr_threshold: float = DEFAULT_RESCUE_SNR_THRESHOLD,
    prominence_threshold: float = DEFAULT_RESCUE_PROMINENCE_THRESHOLD,
    rescue_significance: float = DEFAULT_SIGNIFICANCE,
    knockout_significance: float = DEFAULT_SIGNIFICANCE,
    rescue_max_peaks: int = DEFAULT_MAX_PEAKS,
    conservative_kwargs: Optional[dict[str, Any]] = None,
    merge_separation_factor: float = DEFAULT_MERGE_SEPARATION_FACTOR,
    structural_merge_factor: float = DEFAULT_STRUCTURAL_MERGE_FACTOR,
    n_eff_kind: str = DEFAULT_N_EFF_KIND,
    shape_error_epsilon: float = 0.0,
) -> ConsolidatedRescueOutcome:
    """Iterate rescue + joint refit + merge + knockout consolidation
    (option B).

    The loop:

    1. Run :func:`attempt_residual_rescue` against the current fit's residual.
       If the rescue accepts 0 peaks, terminate (nothing to add).
    2. Joint-refit the union of the current fit's peaks and the rescue's
       accepted peaks against ``complex_spectrum`` with **every parameter
       thawed** (including tau). The starting tau is the rescue's tau
       (which already handled the apodization override when the previous
       fit's tau was pathological) -- this is the structural mitigation
       against w198-like cases where the previous fit's tau pegged at the
       lower bound. If the joint refit fails to converge, terminate
       (retain the previous round's fit).
    3. Run :func:`merge_close_peaks_cleanup` on the joint fit. This must
       happen BEFORE the knockout sweep because the refit-based knockout
       drops every member of a duplicate cluster (each twin's refit is
       absorbed by its siblings, so each individually scores as
       redundant). Merging first collapses sub-resolution clusters
       structurally (Tier 1) and strictly-AICc-preferred close pairs
       (Tier 2), leaving knockout with a clean K-peak set in which each
       entry is a physically distinct candidate.
    4. Run :func:`knockout_test` on the merged joint fit. If any peak is
       flagged unsupported (``aicc_delta < 0`` -- AICc strictly prefers
       the (K-1)-peak refit), drop those peaks and refit on the
       surviving subset. The pruned refit becomes the consolidated fit
       for this round; if pruning would empty the model, terminate
       (retain the previous round's fit).
    5. Loop back to step 1 with the consolidated fit as the new
       ``current``, until ``max_rescue_rounds`` is reached.

    Failsafe diagnostic
    -------------------
    Each :class:`RescueRoundDiagnostics` records
    ``n_pruned_rescue_origin`` -- the count of rescue-added peaks the
    knockout sweep dropped. A nonzero value indicates the joint refit may
    not have escaped a pathological basin (e.g., the previous fit's tau
    being still wrong even after thawing); a future A-mode fallback for
    that window would key off this signal. v1 logs only.

    Parameters
    ----------
    offset_grid_mhz, complex_spectrum, rms_noise
        Window grid, *frozen-background-subtracted* complex data, and
        per-bin complex noise RMS. The same arrays the initial fit
        consumed.
    initial_fit
        The conservative-fit result for this window's initial pass.
    tau0_us, acquisition_us
        Default / starting tau and active acquisition length. Used to
        derive joint-refit constraints (tau bounds, amplitude bounds,
        penalties) via :func:`derive_window_fit_constraints` -- the same
        constraints the initial fit used.
    max_rescue_rounds
        Cap on the rescue + joint-refit cycle. ``1`` reproduces a
        single-pass rescue with consolidation; the default ``3`` matches
        :data:`DEFAULT_RESCUE_MAX_ROUNDS`.
    snr_threshold, prominence_threshold, rescue_significance
        Forwarded to :func:`attempt_residual_rescue` each round.
    knockout_significance
        F-test p-value threshold for ``KnockoutResult.supported``. Peaks
        whose knockout p-value is *above* this threshold are dropped in
        the prune-and-refit step.
    rescue_max_peaks
        Cap on the rescue's own conservative loop -- forwarded as
        :func:`attempt_residual_rescue` 's ``max_peaks``. Defaults to
        :data:`DEFAULT_MAX_PEAKS`; the validation harness uses 32 so the
        rescue's per-round K is gated by statistics rather than an
        integer cap. The joint refit has no such cap (it just refits
        whatever the rescue handed it).
    conservative_kwargs
        Forwarded to :func:`attempt_residual_rescue` and used to derive
        the joint refit's constraints. Pass the same options the initial
        fit received (notably ``tau_apodization_us``, ``max_decay_factor``,
        and the penalty lambdas) so the rescue and the joint refit enforce
        the same physics as the original pass.
    """
    if max_rescue_rounds <= 0:
        return ConsolidatedRescueOutcome(
            fit=initial_fit,
            initial_fit=initial_fit,
            rounds=[],
            terminated_reason="max_rescue_rounds<=0",
        )

    u = np.asarray(offset_grid_mhz, dtype=float)
    z = np.asarray(complex_spectrum, dtype=np.complex128)
    sigma = np.asarray(rms_noise, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(u.size, float(sigma))

    # ``max_peaks`` is forwarded explicitly to attempt_residual_rescue;
    # strip it from the conservative-kwargs bag so the inner conservative_fit
    # call doesn't get the same kwarg twice.
    ckwargs_in = dict(conservative_kwargs or {})
    ckwargs_in.pop("max_peaks", None)

    # Derive the joint refit's constraints once. The bounds (tau, amp,
    # penalty references) are properties of the window's data, not of
    # whichever peak set we are currently fitting, so they stay constant
    # across rounds.
    constraints_kwargs = {
        k: ckwargs_in[k]
        for k in (
            "max_decay_factor",
            "min_separation_factor",
            "amp_max_headroom",
            "phase_penalty_lambda",
            "amp_penalty_lambda",
            "phase_penalty_cutoff_fwhm",
            "tau_penalty_lambda",
            "tau_penalty_n_sigma",
            "weak_window_snr_threshold",
            "tau_apodization_us",
            "tau_maj_us",
            "sigma_tau_us",
        )
        if k in ckwargs_in
    }
    constraints = derive_window_fit_constraints(
        z, sigma, tau0_us, acquisition_us, **constraints_kwargs
    )
    fit_kwargs_inner = constraints.fit_kwargs_inner

    current = initial_fit
    rounds: List[RescueRoundDiagnostics] = []
    terminated_reason = "max rounds reached"
    # Across-rounds rejection memory: any rescue-added peak that a
    # previous round's iterative cleanup dropped goes here. The next
    # round's :func:`attempt_residual_rescue` skips detector candidates
    # within +/-1 grid bin of any entry. Keeps the rescue from
    # re-proposing the same offsets every round (visible as audit-trail
    # clutter) and bounds the rescue's per-round work.
    rejected_offsets: List[float] = []

    for round_idx in range(max_rescue_rounds):
        n_initial = len(current.peaks)
        chi2_before = current.fit.chi_squared
        tau_before = float(current.fit.tau_us)

        rescue = attempt_residual_rescue(
            u, z, sigma, current.fit, tau0_us, acquisition_us,
            snr_threshold=snr_threshold,
            prominence_threshold=prominence_threshold,
            significance=rescue_significance,
            max_peaks=rescue_max_peaks,
            conservative_kwargs=ckwargs_in,
            shape_error_epsilon=shape_error_epsilon,
            excluded_offsets=rejected_offsets if rejected_offsets else None,
        )
        n_rescue_added = rescue.fit.n_peaks

        if n_rescue_added == 0:
            rounds.append(
                RescueRoundDiagnostics(
                    round_idx=round_idx,
                    rescue=rescue,
                    joint_fit=None,
                    joint_knockouts=[],
                    pruned_fit=None,
                    pruned_knockouts=[],
                    n_initial_peaks=n_initial,
                    n_rescue_added=0,
                    n_pruned_total=0,
                    n_pruned_rescue_origin=0,
                    chi2_before=chi2_before,
                    chi2_after=chi2_before,
                    tau_us_before=tau_before,
                    tau_us_after=tau_before,
                    accepted=False,
                    reason="no rescue candidates accepted",
                )
            )
            terminated_reason = "no candidates"
            break

        # Joint refit: union peaks. Tau policy depends on whether a
        # Stage 2b calibration is plumbed: when the inner-fit kwargs carry
        # ``tau_penalty_sigma_us`` (the bidirectional-prior signature), tau
        # is frozen at the calibration's ``tau_maj`` here so the joint refit
        # cannot re-introduce the tau-collapse channel the calibration is
        # meant to close. Without calibration, tau is free under the legacy
        # apodization-anchored bounds; start at the rescue's value (the
        # apodization-override structural mitigation for w198-like cases).
        union_init = list(current.peaks) + list(rescue.fit.peaks)
        calibrated = (
            fit_kwargs_inner.get("tau_penalty_sigma_us") is not None
            and float(fit_kwargs_inner.get("tau_penalty_sigma_us") or 0.0) > 0.0
        )
        if calibrated:
            joint_tau_start = float(fit_kwargs_inner["tau_penalty_reference"])
            joint_kwargs = dict(fit_kwargs_inner)
            joint_kwargs["fit_tau"] = False
        else:
            joint_tau_start = _clamp(
                float(rescue.fit.tau_us),
                constraints.tau_bounds[0],
                constraints.tau_bounds[1],
            )
            joint_kwargs = fit_kwargs_inner
        joint = fit_window(
            u, z, sigma, union_init, joint_tau_start, acquisition_us,
            **joint_kwargs,
        )
        if not joint.success:
            rounds.append(
                RescueRoundDiagnostics(
                    round_idx=round_idx,
                    rescue=rescue,
                    joint_fit=joint,
                    joint_knockouts=[],
                    pruned_fit=None,
                    pruned_knockouts=[],
                    n_initial_peaks=n_initial,
                    n_rescue_added=n_rescue_added,
                    n_pruned_total=0,
                    n_pruned_rescue_origin=0,
                    chi2_before=chi2_before,
                    chi2_after=chi2_before,
                    tau_us_before=tau_before,
                    tau_us_after=tau_before,
                    accepted=False,
                    reason="joint refit did not converge",
                )
            )
            terminated_reason = "joint failed"
            break

        # Merge-cleanup the joint refit BEFORE the knockout sweep. The
        # refit-based knockout is symmetric on duplicate clusters -- each
        # duplicate looks individually redundant because the surviving
        # twin(s) re-converge to absorb -- so running knockout first would
        # drop ALL N duplicates and lose the underlying real signal.
        # Collapsing sub-resolution duplicates (Tier 1) and AICc-strictly-
        # preferred close pairs (Tier 2) first leaves knockout with a
        # clean K-peak set where each entry represents a physically
        # distinct candidate to evaluate. Real close pairs at separations
        # where AICc is tied (e.g. w198 outer shoulders at ~1 FWHM) survive
        # the merge and are then evaluated peak-by-peak by knockout.
        merged_joint_fit, n_merged = merge_close_peaks_cleanup(
            u, z, sigma, joint,
            float(joint.tau_us), acquisition_us,
            fit_kwargs_inner=fit_kwargs_inner,
            merge_separation_factor=merge_separation_factor,
            structural_merge_factor=structural_merge_factor,
            n_eff_kind=n_eff_kind,
        )

        # Per-peak knockout produces persisted diagnostics (n_eff,
        # aicc_delta, supported) on the merged joint fit. This is the
        # diagnostic snapshot before iterative cleanup -- callers can
        # see what the per-peak gate said about the joint set.
        joint_knockouts = knockout_test(
            u, z, sigma, merged_joint_fit, acquisition_us,
            fit_kwargs_inner=fit_kwargs_inner,
            n_eff_kind=n_eff_kind,
            significance=knockout_significance,
        )
        # Iterative AICc cleanup: drops the worst offender, refits,
        # repeats. Non-iterative drops kill duplicate clusters wholesale
        # (every member individually fails when its twins absorb on
        # refit); the iterative loop redistributes the dropped peak's
        # contribution and re-evaluates, so a cluster of N duplicates
        # of one real feature converges to a single supported peak.
        pruned_fit_candidate, n_pruned_total = iterative_aicc_cleanup(
            u, z, sigma, merged_joint_fit, acquisition_us,
            fit_kwargs_inner=fit_kwargs_inner,
            n_eff_kind=n_eff_kind,
        )
        # Grid spacing for the +/-1-bin tolerance used in both the
        # rescue-survival check below and the across-rounds blacklist.
        u_sorted = np.sort(u)
        df_mhz = (
            float(np.min(np.diff(u_sorted))) if u_sorted.size >= 2 else 0.0
        )
        survival_tol = max(df_mhz, 1e-3)
        # Two-purpose bookkeeping:
        #
        # (1) Failsafe counter ``n_pruned_rescue_origin``: how many
        #     iterative-cleanup drops were rescue-origin. The diagnostic
        #     contract is ``n_pruned_rescue_origin <= n_pruned_total``
        #     (you cannot prune more than was dropped). Walk
        #     rescue.fit.peaks vs the two downstream snapshots
        #     (merged_joint_fit, pruned_fit_candidate); a rescue peak
        #     that survived merge but not iterative cleanup is a
        #     rescue-origin pruning. Match within +/-1 grid bin (LSQ
        #     shifts sub-bin; merge averages, which exceeds 1 bin when
        #     the pair was >2 bins apart, so the merge case is
        #     correctly classified).
        merged_offsets = [pk.offset_mhz for pk in merged_joint_fit.peaks]
        survivor_offsets = [pk.offset_mhz for pk in pruned_fit_candidate.peaks]
        n_pruned_rescue = 0
        for rescue_pk in rescue.fit.peaks:
            in_merged = any(
                abs(rescue_pk.offset_mhz - mo) <= survival_tol
                for mo in merged_offsets
            )
            in_pruned = any(
                abs(rescue_pk.offset_mhz - sp) <= survival_tol
                for sp in survivor_offsets
            )
            if in_merged and not in_pruned:
                n_pruned_rescue += 1
        # (2) Across-rounds blacklist: every detector candidate from
        #     this round whose frequency did NOT end up as a fitted
        #     peak in the consolidated set goes on the blacklist. The
        #     next round's detector will skip any frequency within
        #     +/-1 grid bin of these. Blacklisting the *detector*
        #     frequency (rather than the LSQ-refined offset) matters:
        #     the LSQ can pull a rescue-fit peak away from its
        #     detector position by a bin or more, so blacklisting the
        #     refined offset misses the detector's re-nomination of
        #     the original bin in the next round.
        if df_mhz > 0.0:
            for cand in rescue.candidates:
                freq = float(cand.frequency_mhz)
                became_peak = any(
                    abs(freq - sp) <= survival_tol for sp in survivor_offsets
                )
                if became_peak:
                    continue
                if not any(abs(freq - x) <= df_mhz for x in rejected_offsets):
                    rejected_offsets.append(freq)

        pruned_fit: Optional[WindowFitResult]
        pruned_knockouts: List[KnockoutResult]
        if pruned_fit_candidate.n_peaks == 0:
            rounds.append(
                RescueRoundDiagnostics(
                    round_idx=round_idx,
                    rescue=rescue,
                    joint_fit=joint,
                    joint_knockouts=joint_knockouts,
                    pruned_fit=None,
                    pruned_knockouts=[],
                    n_initial_peaks=n_initial,
                    n_rescue_added=n_rescue_added,
                    n_pruned_total=n_pruned_total,
                    n_pruned_rescue_origin=n_pruned_rescue,
                    chi2_before=chi2_before,
                    chi2_after=chi2_before,
                    tau_us_before=tau_before,
                    tau_us_after=tau_before,
                    accepted=False,
                    reason="iterative cleanup dropped every peak",
                    n_merged=n_merged,
                )
            )
            terminated_reason = "all pruned"
            break
        pruned_fit = pruned_fit_candidate
        # Final per-peak diagnostics on the consolidated set. These are
        # the ones persisted to KnockoutInfo on the surviving peaks.
        pruned_knockouts = knockout_test(
            u, z, sigma, pruned_fit, acquisition_us,
            fit_kwargs_inner=fit_kwargs_inner,
            n_eff_kind=n_eff_kind,
            significance=knockout_significance,
        )

        chi2_after = pruned_fit.chi_squared
        tau_after = float(pruned_fit.tau_us)
        reason_parts: List[str] = []
        if n_pruned_total > 0:
            reason_parts.append(f"knockout pruned {n_pruned_total} peak(s)")
        if n_merged > 0:
            reason_parts.append(f"merge collapsed {n_merged} pair(s)")
        if not reason_parts:
            reason = "joint refit consolidated rescue contribution"
        else:
            reason = "joint refit + " + " + ".join(reason_parts)
        rounds.append(
            RescueRoundDiagnostics(
                round_idx=round_idx,
                rescue=rescue,
                joint_fit=joint,
                joint_knockouts=joint_knockouts,
                pruned_fit=pruned_fit,
                pruned_knockouts=pruned_knockouts,
                n_initial_peaks=n_initial,
                n_rescue_added=n_rescue_added,
                n_pruned_total=n_pruned_total,
                n_pruned_rescue_origin=n_pruned_rescue,
                chi2_before=chi2_before,
                chi2_after=chi2_after,
                tau_us_before=tau_before,
                tau_us_after=tau_after,
                accepted=True,
                reason=reason,
                n_merged=n_merged,
            )
        )
        current = ConservativeFitResult(
            fit=pruned_fit,
            audit_trail=initial_fit.audit_trail,
            knockouts=pruned_knockouts,
        )
    else:
        terminated_reason = "max rounds reached"

    return ConsolidatedRescueOutcome(
        fit=current,
        initial_fit=initial_fit,
        rounds=rounds,
        terminated_reason=terminated_reason,
    )
