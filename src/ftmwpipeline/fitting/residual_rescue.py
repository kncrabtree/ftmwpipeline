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

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from . import validation
from .peak_model import ModelPeak, PeakShape, model_spectrum
from .residual_screening import (
    ResidualPeakCandidate,
    find_residual_peaks,
)
from .spur_detection import SpurMaskSpec
from .validation import (
    DEFAULT_N_EFF_KIND,
    calculate_aicc,
    calculate_chi_squared_improvement,
    calculate_noise_weighted_chi2,
    effective_sample_size,
    feature_fwhm,
    gate_aicc_pair,
)
from .window_fit import (
    DEFAULT_MAX_PEAKS,
    DEFAULT_MIN_PAIR_SEPARATION_RESOLUTION_FACTOR,
    DEFAULT_MIN_SEPARATION_FACTOR,
    DEFAULT_SIGNIFICANCE,
    AddStep,
    ConservativeFitResult,
    KnockoutResult,
    WindowFitResult,
    _effective_min_pair_separation,
    _seed_peak,
    conservative_fit,
    derive_window_fit_constraints,
    evaluate_baseline,
    fit_window,
    knockout_test,
)

__all__ = [
    "DEFAULT_CLEANUP_SIGNIFICANCE",
    "DEFAULT_MERGE_SEPARATION_FACTOR",
    "DEFAULT_N_EFF_KIND",
    "DEFAULT_OVERFIT_AMP_RATIO_BAND",
    "DEFAULT_OVERFIT_AMP_RATIO_THRESHOLD",
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
# Amplitude-ratio tiebreaker (GitHub issue #13). In the supra-resolution band
# out to ``DEFAULT_OVERFIT_AMP_RATIO_BAND`` active-FT resolution elements
# (``1/T_active``), a pair whose larger/smaller fitted-amplitude ratio clears
# ``DEFAULT_OVERFIT_AMP_RATIO_THRESHOLD`` is collapsed unconditionally: the weak
# member is a rescue-parked shape-error absorber beside a strong line, not a
# real doublet (AICc supports it, so the resolution floor and the Tier-2 gate
# both leave it). Balanced pairs (ratio below the threshold) in the same band
# are genuine close doublets and are preserved. Calibrated on the 2638 gaussian
# fixture (overfit absorbers 8-16:1, real close pairs <= ~4:1; band covers the
# w281/w143 pairs at ~1.1-1.3 elements while the closest real control pair sits
# at 1.08 elements with ratio ~1.2). Cross-fixture calibration debt. Set the
# threshold to 0 (or the band to 0) to disable the tier.
DEFAULT_OVERFIT_AMP_RATIO_BAND = 1.5
DEFAULT_OVERFIT_AMP_RATIO_THRESHOLD = 6.0
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
    min_pair_separation_resolution_factor: float = (
        DEFAULT_MIN_PAIR_SEPARATION_RESOLUTION_FACTOR
    ),
    overfit_amp_ratio_band: float = DEFAULT_OVERFIT_AMP_RATIO_BAND,
    overfit_amp_ratio_threshold: float = DEFAULT_OVERFIT_AMP_RATIO_THRESHOLD,
    significance: float = DEFAULT_CLEANUP_SIGNIFICANCE,
    n_eff_kind: str = DEFAULT_N_EFF_KIND,
    weighted_gate_chi2: Optional[bool] = None,
    spur_mask: Optional[SpurMaskSpec] = None,
    gate_budget_extra: Optional[np.ndarray] = None,
) -> Tuple[WindowFitResult, int]:
    """Multi-tier merge cleanup for close peak pairs.

    Iteratively finds the closest adjacent pair and decides whether to
    collapse it into a single peak. Three tiers, keyed on the pair's
    separation (and, for the third, its amplitude ratio):

    Both resolution tier thresholds carry a resolution-referenced floor: each
    is the larger of its FWHM-referenced factor and one active-FT resolution
    element (``min_pair_separation_resolution_factor / T_active``). On narrow
    features the per-window FWHM falls below the Fourier resolution limit, so a
    FWHM-only threshold leaves sub-resolution duplicate pairs uncollapsed
    (GitHub issue #13).

    1. **Sub-resolution** (separation < the structural threshold, ``max(
       structural_merge_factor * fwhm, min_pair_separation_resolution_factor /
       T_active)``): merge unconditionally. The single-shape model physically
       cannot distinguish these from a single peak, so any LSQ that converges
       to two separate peaks at this scale is a numerical artifact (the
       duplicate-pair-overfit pathology from w148/w269 in the 2638 fixture).
    2. **Above resolution** (structural threshold <=
       separation < the merge threshold ``max(merge_separation_factor * fwhm,
       min_pair_separation_resolution_factor / T_active)``): merge only when
       AICc strictly prefers the (K-1)-peak model. ``n_eff`` (the package
       default :data:`~ftmwpipeline.fitting.validation.DEFAULT_N_EFF_KIND`,
       ``perplexity_log1p_snr``) can still put both AICc(K) and AICc(K-1) at
       ``+inf`` on a feature too narrow to identify either model. Tied AICc
       is read as "no evidence for merge" and preserves the K-peak fit
       -- this is the structural protection for real close pairs (w198
       outer shoulders at ~1 FWHM from inner peaks) the earlier
       AICc-only gate over-merged.
    3. **Amplitude-ratio tier** (merge threshold <= separation <
       ``overfit_amp_ratio_band * (1 / T_active)``): merge unconditionally
       *only* when the pair's larger/smaller amplitude ratio clears
       ``overfit_amp_ratio_threshold``. This catches the rescue-parked
       shape-error absorber sitting just above the resolution floor next to a
       strong line (the w281 / w143 class on the 2638 gaussian fixture): AICc
       supports the extra peak (it absorbs real residual), so tiers 1-2 leave
       it, but a large amplitude ratio marks the weak member as an absorber
       rather than a real doublet. A balanced pair in this band is a genuine
       close doublet and is preserved. Calibration debt (GitHub issue #13).

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
    budget: Optional[np.ndarray] = None
    if gate_budget_extra is not None:
        budget = np.asarray(gate_budget_extra, dtype=float)[order]

    weighted = (
        validation.DEFAULT_WEIGHTED_GATE_CHI2
        if weighted_gate_chi2 is None
        else weighted_gate_chi2
    )
    if spur_mask is not None:
        keep = ~spur_mask.bin_mask(u)
        if not keep.any():
            keep = np.ones(u.size, dtype=bool)
    else:
        keep = np.ones(u.size, dtype=bool)
    sigma_keep = sigma[keep]
    budget_keep: Optional[np.ndarray] = None if budget is None else budget[keep]

    fwhm = (
        feature_fwhm(fit.tau_us, acquisition_us, shape=fit.shape)
        if fit.tau_us > 0.0
        else 0.0
    )
    if fwhm <= 0.0:
        return fit, 0
    # Both tier thresholds carry a resolution-referenced floor: the larger of
    # the FWHM-referenced factor and one active-FT resolution element
    # ``min_pair_separation_resolution_factor / T_active``. On narrow features
    # the per-window FWHM falls below the Fourier resolution limit, so a
    # FWHM-only threshold leaves sub-resolution duplicate pairs uncollapsed
    # (GitHub issue #13). The floor pulls the unconditional (Tier 1) band up to
    # one resolution element, where any two-peak solution is a numerical
    # artifact a single peak cannot be statistically distinguished from.
    merge_threshold = _effective_min_pair_separation(
        fwhm,
        acquisition_us,
        merge_separation_factor,
        min_pair_separation_resolution_factor,
    )
    structural_threshold = _effective_min_pair_separation(
        fwhm,
        acquisition_us,
        structural_merge_factor,
        min_pair_separation_resolution_factor,
    )
    # Amplitude-ratio tier (GitHub issue #13, w281/w143 class): a rescue can
    # park a small shape-error-absorber peak just *above* the resolution floor
    # next to a strong line (separation ~1.0-1.5 elements, amplitude ratio
    # >> 1). AICc supports it (it soaks real residual the single-shape model
    # leaves), so neither the structural tier nor the Tier-2 AICc gate collapse
    # it. In a band out to ``overfit_amp_ratio_band`` resolution elements, a
    # pair whose amplitude ratio clears ``overfit_amp_ratio_threshold`` is
    # collapsed unconditionally -- the weak member is an absorber, not a real
    # line. Balanced pairs (ratio below the threshold) in the same band are
    # genuine close doublets and are preserved. Cross-fixture calibration debt.
    amp_ratio_band = (
        overfit_amp_ratio_band / acquisition_us
        if acquisition_us > 0.0
        and overfit_amp_ratio_band > 0.0
        and overfit_amp_ratio_threshold > 0.0
        else 0.0
    )
    loop_band = max(merge_threshold, amp_ratio_band)

    # Tau is effectively a dataset-shared parameter (transit time x natural
    # lifetime); a single-window (K-1) refit must not get the extra knob of
    # broadening tau to absorb the dropped peak's contribution. Lock tau at
    # the current K-fit value for every (K-1) trial -- matches the
    # convention used by ``knockout_test`` and ``iterative_aicc_cleanup``.
    refit_kwargs: dict[str, Any] = dict(fit_kwargs_inner)
    refit_kwargs["fit_tau"] = False
    refit_kwargs.setdefault("shape", fit.shape)
    refit_kwargs["spur_mask"] = spur_mask

    current = fit
    n_merged = 0
    # Pairs the blend escape exempted from collapse this run: a kept pair
    # must not be re-selected as the closest pair forever (the loop would
    # spin), so it is keyed by its members' offsets and skipped. A merge of
    # a DIFFERENT pair refits every peak and shifts the kept pair's offsets
    # off its key -- it is then simply re-evaluated, which is correct (the
    # evidence may have changed).
    protected_pairs: set[Tuple[float, float]] = set()

    def _pair_key(lo: ModelPeak, hi: ModelPeak) -> Tuple[float, float]:
        return (round(lo.offset_mhz, 6), round(hi.offset_mhz, 6))

    while current.n_peaks >= 2:
        sorted_peaks = sorted(current.peaks, key=lambda p: p.offset_mhz)
        # Closest adjacent pair (after sorting, the minimum gap must be
        # between adjacent entries), skipping escape-protected pairs.
        min_dist = float("inf")
        merge_i = -1
        for i in range(len(sorted_peaks) - 1):
            if _pair_key(sorted_peaks[i], sorted_peaks[i + 1]) in protected_pairs:
                continue
            d = sorted_peaks[i + 1].offset_mhz - sorted_peaks[i].offset_mhz
            if d < min_dist:
                min_dist = d
                merge_i = i
        if merge_i < 0 or min_dist >= loop_band:
            break
        pair_lo = sorted_peaks[merge_i]
        pair_hi = sorted_peaks[merge_i + 1]
        amp_a = abs(pair_lo.amplitude)
        amp_b = abs(pair_hi.amplitude)
        amp_min = min(amp_a, amp_b)
        amp_ratio = max(amp_a, amp_b) / amp_min if amp_min > 0.0 else float("inf")
        # Lazy refit: only the closest pair is ever a merge candidate, and a
        # balanced pair in the amplitude-ratio band (>= merge_threshold,
        # ratio below the threshold) is a real doublet that is kept. Break
        # before paying the (K-1) ``fit_window`` refit in that case -- nothing
        # nearer remains, so the loop is done. (Without this, every real close
        # doublet out to ``overfit_amp_ratio_band`` would cost a wasted refit.)
        if min_dist >= merge_threshold and amp_ratio < overfit_amp_ratio_threshold:
            break
        if os.environ.get("FTMW_DEBUG_MERGE"):
            tier = (
                "T1"
                if min_dist < structural_threshold
                else ("T2" if min_dist < merge_threshold else "T3")
            )
            print(
                f"[merge] pair ({pair_lo.offset_mhz:+.4f},{pair_hi.offset_mhz:+.4f}) "
                f"dist={min_dist:.4f} {tier} amp_ratio={amp_ratio:.2f} "
                f"cancel={validation.pair_cancellation_fraction(pair_lo.amplitude, pair_lo.phase, pair_hi.amplitude, pair_hi.phase):.2f} "
                f"thr(st={structural_threshold:.4f},mg={merge_threshold:.4f},"
                f"band={amp_ratio_band:.4f})",
                flush=True,
            )
        merged_pair = _merge_cluster([sorted_peaks[merge_i], sorted_peaks[merge_i + 1]])
        merged_init = (
            sorted_peaks[:merge_i] + [merged_pair] + sorted_peaks[merge_i + 2 :]
        )
        refit = fit_window(
            u,
            z,
            sigma,
            merged_init,
            float(current.tau_us),
            acquisition_us,
            **refit_kwargs,
        )
        if not refit.success:
            break
        # Cleanup refits lock tau by design (a single-window K-1 refit must
        # not get the extra knob of broadening tau to absorb the merged
        # peak's contribution), but the merged fit's tau still originated
        # in the prior K-peak fit. Carry the originating ``tau_was_fit``
        # and ``tau_error`` forward so post-cleanup consumers can tell
        # frozen-by-gate from data-driven-but-cleaned and still see the
        # joint refit's tau uncertainty (the locked refit's covariance has
        # no tau slot, so it can't report one).
        refit.tau_was_fit = current.tau_was_fit
        refit.tau_error = current.tau_error

        # Tier 1: sub-resolution -> merge unconditionally, UNLESS the pair
        # earns the blend escape. The unconditional collapse targets the
        # cancelling near-duplicate artifact, but the complex-domain
        # evidence CAN distinguish a genuine sub-resolution blend from a
        # single peak (distinct member phases produce a profile one ``h_T``
        # cannot match): a pair whose collapse costs overwhelming raw
        # chi-squared and whose members are constructive is physical
        # structure, not the pathology (see
        # :data:`validation.DEFAULT_PAIR_CANCELLATION_MAX`).
        if min_dist < structural_threshold:
            if validation.blend_pair_escape(
                refit.chi_squared - current.chi_squared,
                max(current.n_params - refit.n_params, 1),
                pair_lo.amplitude,
                pair_lo.phase,
                pair_hi.amplitude,
                pair_hi.phase,
            ):
                protected_pairs.add(_pair_key(pair_lo, pair_hi))
                continue
            current = refit
            n_merged += 1
            continue

        # Tier 2: above-resolution -> AICc-with-n_eff test, with
        # REJECT-on-tie. n_eff is keyed on the more-complex (K-peak) model
        # via the default ``perplexity_log1p_snr`` kind; both AICc
        # evaluations share it.
        # ``>=`` (rather than ``>``) makes the gate "merge only when
        # K-1 is strictly better"; ties (both AICc finite-equal or both
        # +inf because n_eff < k+1) preserve the K-peak fit, which is
        # the structural protection for real close pairs the AICc-only
        # gate over-merged.
        if min_dist < merge_threshold:
            n_eff = effective_sample_size(
                current.fitted_spectrum,
                kind=n_eff_kind,
                sigma=sigma,
            )
            aicc_k, aicc_km1 = gate_aicc_pair(
                n_eff,
                more_n_params=current.n_params,
                less_n_params=refit.n_params,
                more_chi2_raw=current.chi_squared,
                less_chi2_raw=refit.chi_squared,
                weighted=weighted,
                more_residual=np.asarray(current.residual)[keep],
                less_residual=np.asarray(refit.residual)[keep],
                rms_noise=sigma_keep,
                weight_model=np.asarray(current.fitted_spectrum)[keep],
                n_eff_kind=n_eff_kind,
                ref_reduced_chi2=current.reduced_chi2,
                budget_extra=budget_keep,
            )
            if aicc_km1 >= aicc_k:
                break
            current = refit
            n_merged += 1
            continue

        # Tier 3: amplitude-ratio tiebreaker in the supra-resolution band
        # [merge_threshold, amp_ratio_band). AICc is deliberately NOT consulted
        # -- it prefers keeping the absorber. A high amplitude ratio is the
        # signal that the closer-but-resolvable weak peak is a shape-error
        # absorber beside a strong line; collapse it. A balanced pair here is a
        # real doublet -> stop (it is the closest pair, so nothing nearer
        # remains). The blend escape applies with the fidelity-floor-scaled
        # bar: an absorber's win is bounded by the lineshape floor it soaks
        # (``chi2_r ~ (kappa*SNR)^2``), so only evidence far beyond that
        # floor marks the weak member as a real unresolved line.
        if amp_ratio >= overfit_amp_ratio_threshold:
            floor = (
                float(current.reduced_chi2)
                if np.isfinite(current.reduced_chi2)
                else 1.0
            )
            if validation.blend_pair_escape(
                refit.chi_squared - current.chi_squared,
                max(current.n_params - refit.n_params, 1),
                pair_lo.amplitude,
                pair_lo.phase,
                pair_hi.amplitude,
                pair_hi.phase,
                evidence_floor=floor,
            ):
                protected_pairs.add(_pair_key(pair_lo, pair_hi))
                continue
            current = refit
            n_merged += 1
            continue
        break
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
    spur_mask: Optional[SpurMaskSpec] = None,
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
    refit_kwargs = dict(fit_kwargs_inner)
    refit_kwargs.setdefault("shape", fit.shape)
    refit_kwargs["spur_mask"] = spur_mask
    keep = (
        ~spur_mask.bin_mask(u) if spur_mask is not None else np.ones(u.size, dtype=bool)
    )
    if not keep.any():
        keep = np.ones(u.size, dtype=bool)
    while current.n_peaks > 0:
        worst_p_value = -1.0
        worst_idx = -1
        worst_refit: Optional[WindowFitResult] = None
        for i in range(current.n_peaks):
            reduced_init = [pk for j, pk in enumerate(current.peaks) if j != i]
            if not reduced_init:
                # Compare K=1 fit to K=0 (null model): use the null chi-squared
                # directly rather than calling fit_window with an empty list.
                # Spur bins are excluded to match the masked K-peak chi^2.
                null_chi2 = float(
                    np.sum(np.abs((z / (sigma / np.sqrt(2.0)))[keep]) ** 2)
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
                u,
                z,
                sigma,
                reduced_init,
                tau0_us,
                acquisition_us,
                **refit_kwargs,
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
                u,
                z,
                sigma,
                [],
                tau0_us,
                acquisition_us,
                **refit_kwargs,
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
    weighted_gate_chi2: Optional[bool] = None,
    spur_mask: Optional[SpurMaskSpec] = None,
    gate_budget_extra: Optional[np.ndarray] = None,
    gate_background: Optional[np.ndarray] = None,
    protected_offsets: Optional[Sequence[float]] = None,
    protected_tol_mhz: float = 0.0,
    initial_refits: Optional[Mapping[int, WindowFitResult]] = None,
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

    ``protected_offsets`` carries the offsets of peaks the rescue
    *inherited* (the round-start set: Stage-3-seeded conservative peaks and
    prior-round survivors). Their drop test runs without the
    ``gate_budget_extra`` skirt budget on **both** sides of the comparison:
    the budget exists to discount rescue-harvested skirt-fringe error, but
    on a deep-skirt window it is proportional to the dominant |background|
    and otherwise erases an inherited real line's entire evidence -- the
    budgeted score then actively prefers dropping every inherited peak
    (fewer parameters, no visible evidence lost) and the round consolidates
    to a single absorber. Rescue-origin peaks (anything not matching a
    protected offset within ``protected_tol_mhz``) keep the budgeted gate.

    ``gate_background`` (the window's complex frozen-contributor model on
    the input grid) feeds the line-evidence escape hatch: before a peak is
    selected as the drop candidate, its disputed evidence (the K-1 refit's
    residual on the peak's support) is matched-filter tested in raw
    currency beyond the span of the background's skirt-error modes and the
    surviving peaks' lineshape-error modes
    (:func:`~ftmwpipeline.fitting.validation.line_evidence_escape`). A peak
    whose evidence is line-like at >=10x the gate's evidence bar cannot be
    dropped this iteration -- the budget's blind band (a real line riding
    the skirt at comparable magnitude) and the sigma_eff self-blinding (a
    bright peak inflating its own bins' noise) are both overruled by
    overwhelming raw template evidence, while skirt fringes stay inside
    the nuisance span and remain droppable.

    initial_refits : Mapping[int, WindowFitResult], optional
        Pre-computed (K-1) refits keyed by peak index in the *caller's* grid
        order -- valid only for the first while-iteration (``current is
        fit``). Produced by passing ``refit_sink`` to :func:`knockout_test`
        on the same ``fit`` object. For each peak index ``i`` with a
        non-empty ``kept`` list: if ``initial_refits[i]`` exists it is used
        instead of calling ``fit_window``; missing keys fall back to calling
        ``fit_window`` as normal. Ignored from the second iteration on.

        Guard: if the input grid is not already ascending (the ``argsort``
        permutation is not the identity), ``initial_refits`` is discarded --
        the refits were produced on the caller's grid order, which the sort
        step would invalidate.
    """
    if fit.n_peaks == 0:
        return fit, 0

    u = np.asarray(offset_grid_mhz, dtype=float)
    z = np.asarray(complex_spectrum, dtype=np.complex128)
    sigma = np.asarray(rms_noise, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(u.size, float(sigma))
    order = np.argsort(u)
    # initial_refits are valid only when the input grid is already ascending
    # (identity permutation).  If a sort step is needed the refits were
    # produced on the caller's grid order and cannot be reused after reindex.
    _initial_refits: Optional[Mapping[int, WindowFitResult]] = initial_refits
    if _initial_refits is not None and not np.array_equal(order, np.arange(u.size)):
        _initial_refits = None
    u, z, sigma = u[order], z[order], sigma[order]
    budget: Optional[np.ndarray] = None
    if gate_budget_extra is not None:
        budget = np.asarray(gate_budget_extra, dtype=float)[order]
    background: Optional[np.ndarray] = None
    if gate_background is not None:
        background = np.asarray(gate_background, dtype=np.complex128)[order]

    refit_kwargs: dict[str, Any] = dict(fit_kwargs_inner)
    refit_kwargs["fit_tau"] = False
    refit_kwargs.setdefault("shape", fit.shape)
    refit_kwargs["spur_mask"] = spur_mask
    keep = (
        ~spur_mask.bin_mask(u) if spur_mask is not None else np.ones(u.size, dtype=bool)
    )
    if not keep.any():
        keep = np.ones(u.size, dtype=bool)

    weighted = (
        validation.DEFAULT_WEIGHTED_GATE_CHI2
        if weighted_gate_chi2 is None
        else weighted_gate_chi2
    )
    sigma_keep = sigma[keep]
    budget_keep: Optional[np.ndarray] = None if budget is None else budget[keep]

    protected = (
        np.asarray(list(protected_offsets), dtype=float)
        if protected_offsets is not None
        else None
    )

    def _is_protected(offset_mhz: float) -> bool:
        if protected is None or protected.size == 0:
            return False
        return bool(np.min(np.abs(protected - float(offset_mhz))) <= protected_tol_mhz)

    shape_coerced = PeakShape.coerce(fit.shape)

    # Per-iteration cache for per-peak nuisance column triplets (one entry per
    # established peak index j != i).  Keyed by peak index j; valid only within
    # one while-iteration where tau_locked and the peak set are fixed.  Cleared
    # at the top of each iteration.  Columns are cached on the FULL grid and
    # sliced per call: the support slice depends on the disputed peak's
    # template, so only full-grid columns are reusable across the iteration's
    # per-peak escape calls (model_spectrum is pointwise, so slicing a
    # full-grid column is byte-identical to a subgrid build).
    _pk_col_cache: dict[int, List[np.ndarray]] = {}

    def _escape_keeps(
        peak_idx: int,
        peak: ModelPeak,
        others: List[ModelPeak],
        other_indices: List[int],
        evidence_keep: np.ndarray,
        tau_locked: float,
        n_params_peak: int,
    ) -> Tuple[bool, float]:
        """Line-evidence escape for a would-be drop candidate (see docstring)."""
        if validation.DEFAULT_GATE_LINE_ESCAPE_LAMBDA is None:
            return False, 0.0
        template = model_spectrum(
            u, [peak], tau_locked, acquisition_us, shape=shape_coerced
        )
        tpl_keep = template[keep]
        sl = validation.line_escape_support_slice(tpl_keep)
        if sl is None:
            # Template is empty/zero or support too narrow: escape takes the
            # early-return path regardless of columns -- pass empty columns.
            return validation.line_evidence_escape(
                evidence_keep,
                sigma_keep,
                tpl_keep,
                [],
                n_params_peak=n_params_peak,
            )
        # Background columns: full-grid (gradient is not pointwise), then slice.
        bg_cols = validation.line_escape_background_columns(u, background)
        bg_sliced = [col[keep][sl] for col in bg_cols]
        # Per-peak columns: cached full-grid, sliced to this call's support.
        pk_sliced: List[np.ndarray] = []
        for j, other_pk in zip(other_indices, others):
            if j not in _pk_col_cache:
                _pk_col_cache[j] = validation.line_escape_peak_columns(
                    u,
                    [other_pk],
                    tau_locked,
                    acquisition_us,
                    shape=shape_coerced,
                )
            pk_sliced.extend(col[keep][sl] for col in _pk_col_cache[j])
        nuisance_sliced = bg_sliced + pk_sliced
        return validation.line_evidence_escape(
            evidence_keep[sl],
            sigma_keep[sl],
            tpl_keep[sl],
            nuisance_sliced,
            n_params_peak=n_params_peak,
        )

    current = fit
    n_dropped = 0
    # _iter_initial_refits holds the pre-computed refits for the current
    # iteration; cleared to None after the first iteration (refits are only
    # valid for the starting ``fit`` object's peak set and grid order).
    _iter_initial_refits: Optional[Mapping[int, WindowFitResult]] = _initial_refits
    while current.n_peaks > 0:
        _pk_col_cache.clear()
        tau_locked = float(current.tau_us)
        n_eff = effective_sample_size(
            current.fitted_spectrum,
            kind=n_eff_kind,
            sigma=sigma,
        )
        # Gate weights / residuals come from the more-complex K-peak model and
        # are shared across this iteration's per-peak K-1 comparisons (the
        # weighted and penalized branches of ``gate_aicc_pair`` consume them).
        cur_res_keep = np.asarray(current.residual)[keep]
        cur_model_keep = np.asarray(current.fitted_spectrum)[keep]

        def _aicc_k_for(budget_arg: Optional[np.ndarray]) -> float:
            aicc, _ = gate_aicc_pair(
                n_eff,
                more_n_params=current.n_params,
                less_n_params=current.n_params,
                more_chi2_raw=current.chi_squared,
                less_chi2_raw=current.chi_squared,
                weighted=weighted,
                more_residual=cur_res_keep,
                less_residual=cur_res_keep,
                rms_noise=sigma_keep,
                weight_model=cur_model_keep,
                n_eff_kind=n_eff_kind,
                ref_reduced_chi2=current.reduced_chi2,
                budget_extra=budget_arg,
            )
            return aicc

        # Both sides of a peak's drop comparison must share one currency:
        # budgeted for rescue-origin peaks, budget-free for protected
        # (inherited) ones -- so the reference score is computed per currency.
        aicc_k_budgeted = _aicc_k_for(budget_keep)
        aicc_k_raw = _aicc_k_for(None) if budget_keep is not None else aicc_k_budgeted

        worst_margin = 0.0
        worst_aicc_km1 = float("inf")
        worst_idx = -1
        worst_refit: Optional[WindowFitResult] = None
        worst_is_null = False

        for i in range(current.n_peaks):
            peak_protected = _is_protected(current.peaks[i].offset_mhz)
            budget_i = None if peak_protected else budget_keep
            aicc_k_i = aicc_k_raw if peak_protected else aicc_k_budgeted
            kept = [pk for j, pk in enumerate(current.peaks) if j != i]
            kept_indices = [j for j in range(current.n_peaks) if j != i]
            if not kept:
                # Zero model -> residual is the data itself.
                null_chi2 = calculate_noise_weighted_chi2(z[keep], sigma[keep])
                _, aicc_km1 = gate_aicc_pair(
                    n_eff,
                    more_n_params=current.n_params,
                    less_n_params=0,
                    more_chi2_raw=current.chi_squared,
                    less_chi2_raw=null_chi2,
                    weighted=weighted,
                    more_residual=cur_res_keep,
                    less_residual=z[keep],
                    rms_noise=sigma_keep,
                    weight_model=cur_model_keep,
                    n_eff_kind=n_eff_kind,
                    ref_reduced_chi2=current.reduced_chi2,
                    budget_extra=budget_i,
                )
                if aicc_km1 - aicc_k_i < worst_margin:
                    escaped, _ = _escape_keeps(
                        i,
                        current.peaks[i],
                        [],
                        [],
                        z[keep],
                        tau_locked,
                        max(current.n_params, 1),
                    )
                    if escaped:
                        continue
                    worst_margin = aicc_km1 - aicc_k_i
                    worst_aicc_km1 = aicc_km1
                    worst_idx = i
                    worst_refit = None
                    worst_is_null = True
                continue
            if _iter_initial_refits is not None and i in _iter_initial_refits:
                refit = _iter_initial_refits[i]
            else:
                refit = fit_window(
                    u,
                    z,
                    sigma,
                    kept,
                    tau_locked,
                    acquisition_us,
                    **refit_kwargs,
                )
            if not refit.success:
                continue
            # Cleanup refits lock tau by design; carry the originating
            # ``tau_was_fit`` and ``tau_error`` forward so the cleaned-up
            # fit advertises the original determination (not the locked-
            # cleanup-refit flag) and the joint refit's tau uncertainty
            # (the locked refit's covariance has no tau slot).
            refit.tau_was_fit = current.tau_was_fit
            refit.tau_error = current.tau_error
            _, aicc_km1 = gate_aicc_pair(
                n_eff,
                more_n_params=current.n_params,
                less_n_params=refit.n_params,
                more_chi2_raw=current.chi_squared,
                less_chi2_raw=refit.chi_squared,
                weighted=weighted,
                more_residual=cur_res_keep,
                less_residual=np.asarray(refit.residual)[keep],
                rms_noise=sigma_keep,
                weight_model=cur_model_keep,
                n_eff_kind=n_eff_kind,
                ref_reduced_chi2=current.reduced_chi2,
                budget_extra=budget_i,
            )
            margin_i = aicc_km1 - aicc_k_i
            escape_dchi2 = 0.0
            if margin_i < worst_margin:
                escaped, escape_dchi2 = _escape_keeps(
                    i,
                    current.peaks[i],
                    kept,
                    kept_indices,
                    np.asarray(refit.residual)[keep],
                    tau_locked,
                    max(current.n_params - refit.n_params, 1),
                )
                if escaped:
                    margin_i = float("inf")
            validation.debug_fringe_dump(
                "cleanup",
                u_keep=u[keep],
                less_res=np.asarray(refit.residual)[keep],
                more_res=cur_res_keep,
                sigma=sigma_keep,
                budget=budget_i,
                weight_model=cur_model_keep,
                template=model_spectrum(
                    u,
                    [current.peaks[i]],
                    tau_locked,
                    acquisition_us,
                    shape=PeakShape.coerce(current.shape),
                )[keep],
                cand_offset=current.peaks[i].offset_mhz,
                tau_us=tau_locked,
                acquisition_us=acquisition_us,
                raw_chi2_more=current.chi_squared,
                raw_chi2_less=refit.chi_squared,
                aicc_delta=aicc_km1 - aicc_k_i,
                escape_dchi2=escape_dchi2,
                protected=int(peak_protected),
            )
            if margin_i < worst_margin:
                worst_margin = margin_i
                worst_aicc_km1 = aicc_km1
                worst_idx = i
                worst_refit = refit
                worst_is_null = False

        # REJECT-on-tie: keep the peak unless AICc(K-1) is strictly better
        # than AICc(K) in that peak's own currency (``worst_margin`` starts
        # at 0, so only a strictly-negative margin selects a drop; the
        # both-+inf case reads as a tie and stops). Each currency's pair
        # shares its reference score, so budgeted and budget-free
        # comparisons never mix sides.
        if worst_idx < 0:
            break

        # After the first iteration the peak set changes; pre-computed refits
        # are no longer aligned with the new ``current`` object.
        _iter_initial_refits = None
        n_dropped += 1
        if worst_is_null:
            # Dropped the last peak; the cleaned fit is the null model.
            # Produce a zero-peak fit_window result so the loop's return
            # has the same type as every other branch.
            null_fit = fit_window(
                u,
                z,
                sigma,
                [],
                tau_locked,
                acquisition_us,
                **refit_kwargs,
            )
            null_fit.tau_was_fit = current.tau_was_fit
            null_fit.tau_error = current.tau_error
            current = null_fit
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
    excluded_offsets: Optional[Sequence[float]] = None,
    spur_mask: Optional[SpurMaskSpec] = None,
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
    excluded_offsets : sequence of float, optional
        Additional offsets (MHz) the rescue must NOT re-propose
        candidates near. Combined with ``current_fit.peaks`` 's offsets
        into a single rejection list: any detector candidate whose
        frequency falls within the locality radius of either a
        *currently-fitted peak* or an *entry of this list* is dropped
        before going to :func:`conservative_fit`. That radius is the
        larger of one persisted-grid bin and one active-FT resolution
        element (``min_pair_separation_resolution_factor / T_active``,
        read from ``conservative_kwargs``; GitHub issue #13). The
        fitted-peak rejection
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
    # initial fit's own peaks + tau + shape here -- this is the actual model
    # the initial fit produced, regardless of whether its tau is physical.
    # The fit's jointly-fit baseline term (when present) is part of that
    # model: without it the carried pedestal would re-read as residual and
    # the detector would nominate candidates on it.
    rescue_shape = current_fit.shape
    initial_model = model_spectrum(
        u,
        current_fit.peaks,
        current_fit.tau_us,
        acquisition_us,
        shape=rescue_shape,
    ) + evaluate_baseline(current_fit, u)
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
        feature_fwhm(rescue_tau_us, acquisition_us, shape=rescue_shape)
        if rescue_tau_us > 0.0
        else 0.0
    )

    raw_candidates = find_residual_peaks(
        u,
        residual,
        sigma,
        snr_threshold=snr_threshold,
        prominence_threshold=prominence_threshold,
        fwhm_mhz=rescue_fwhm if rescue_fwhm > 0.0 else None,
    )
    # Locality rejection: drop any candidate too close to a currently-fitted
    # peak OR an explicit blacklist entry. Fitted-peak locality means the
    # candidate is sitting under an existing peak -- the residual signal there
    # is shape-error / leakage, not a missed line -- and re-nominating it would
    # only invite the joint-refit / iterative-cleanup chain to drop it again.
    # The explicit blacklist is the across-rounds memory in
    # :func:`rescue_and_consolidate` (rescue peaks that previous rounds
    # nominated and the cleanup later rejected).
    #
    # The rejection radius for fitted peaks / blacklist entries is the larger
    # of one persisted-grid bin and one active-FT resolution element
    # ``min_pair_separation_resolution_factor / T_active`` (GitHub issue #13):
    # the persisted grid is oversampled (alpha-padded), so +/-1 of its bins is
    # well below the Fourier resolution limit and lets the rescue promote a
    # sub-resolution shape-error candidate as a duplicate line beside a strong
    # peak. A real missed line must sit at least ~1 resolution element away.
    k_res = float(
        ckwargs_in.get(
            "min_pair_separation_resolution_factor",
            DEFAULT_MIN_PAIR_SEPARATION_RESOLUTION_FACTOR,
        )
    )
    peak_reject_offsets: List[float] = [pk.offset_mhz for pk in current_fit.peaks]
    if excluded_offsets:
        peak_reject_offsets.extend(float(x) for x in excluded_offsets)
    # A spur bin has a large residual (no line shape fits it), so the
    # detector would nominate it. Reject candidates on a spur offset so the
    # rescue never seeds a peak there; the residual mask below is the
    # backstop (a peak on masked bins earns no chi^2 credit and is gated out).
    # Spurs are single-bin tones, so a grid-bin radius (not the resolution
    # element) is the right locality test for them.
    spur_reject_offsets: List[float] = (
        [float(o) for o in spur_mask.offsets_mhz] if spur_mask is not None else []
    )
    if (
        (peak_reject_offsets or spur_reject_offsets)
        and raw_candidates
        and (u.size >= 2)
    ):
        u_sorted = np.sort(u)
        df_mhz = float(np.min(np.diff(u_sorted)))
        if df_mhz > 0.0:
            resolution_mhz = (
                k_res / acquisition_us if acquisition_us > 0.0 and k_res > 0.0 else 0.0
            )
            peak_radius = max(df_mhz, resolution_mhz)
            raw_candidates = [
                c
                for c in raw_candidates
                if not any(
                    abs(c.frequency_mhz - x) <= peak_radius for x in peak_reject_offsets
                )
                and not any(
                    abs(c.frequency_mhz - x) <= df_mhz for x in spur_reject_offsets
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
    # The rescue passes ``significance`` / ``min_separation_factor`` /
    # ``max_peaks`` explicitly below as its own overrides, so drop any
    # values the caller's conservative_kwargs carried for these keys --
    # otherwise the ``**ckwargs`` spread would collide with the named
    # arguments on the conservative_fit call.
    ckwargs.pop("significance", None)
    ckwargs.pop("min_separation_factor", None)
    ckwargs.pop("max_peaks", None)
    ckwargs.setdefault("shape", rescue_shape)
    # The rescue's inner add-loop runs without the line-evidence escape:
    # its trial model holds only rescue peaks (the initial fit lives in the
    # subtracted residual), so the escape would lack the established-peak
    # nuisance context. Arbitration of the rescue's finds belongs to the
    # joint refit + iterative cleanup, where the escape has the full model.
    ckwargs["gate_line_escape"] = False
    if not candidate_offsets:
        empty = conservative_fit(
            u,
            residual,
            sigma,
            [],
            rescue_tau_us,
            acquisition_us,
            fit_tau=False,
            spur_mask=spur_mask,
            **ckwargs,
        )
        return RescueOutcome(
            fit=empty.fit,
            audit=empty.audit_trail,
            candidates=candidates,
            knockouts=empty.knockouts,
        )

    rescue_result: ConservativeFitResult = conservative_fit(
        u,
        residual,
        sigma,
        candidate_offsets,
        rescue_tau_us,
        acquisition_us,
        fit_tau=False,
        significance=significance,
        min_separation_factor=min_separation_factor,
        max_peaks=max_peaks,
        spur_mask=spur_mask,
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
    overfit_amp_ratio_band: float = DEFAULT_OVERFIT_AMP_RATIO_BAND,
    overfit_amp_ratio_threshold: float = DEFAULT_OVERFIT_AMP_RATIO_THRESHOLD,
    n_eff_kind: str = DEFAULT_N_EFF_KIND,
    shape: "PeakShape | str" = "lorentzian",
    spur_mask: Optional[SpurMaskSpec] = None,
    gate_budget_extra: Optional[np.ndarray] = None,
    gate_background: Optional[np.ndarray] = None,
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
        :data:`DEFAULT_MAX_PEAKS` (``0`` = no cap), so the rescue's per-round
        K is gated by the AICc statistics rather than an integer cap. The joint
        refit has no such cap either (it just refits whatever the rescue handed
        it).
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
    # Per-window sigma_eff budget (aligned with ``offset_grid_mhz``): ride the
    # conservative-kwargs bag into :func:`attempt_residual_rescue`'s inner
    # conservative_fit, and thread explicitly to the cleanup gates below.
    if gate_budget_extra is not None:
        ckwargs_in["gate_budget_extra"] = gate_budget_extra
    else:
        gate_budget_extra = ckwargs_in.get("gate_budget_extra")
    # Resolution-referenced minimum-pair-separation floor (GitHub issue #13).
    # ``attempt_residual_rescue`` reads it from ``ckwargs_in`` for its locality
    # rejection; the merge cleanup needs it as an explicit argument.
    merge_resolution_factor = float(
        ckwargs_in.get(
            "min_pair_separation_resolution_factor",
            DEFAULT_MIN_PAIR_SEPARATION_RESOLUTION_FACTOR,
        )
    )

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
            "fit_tau_min_snr",
            "tau_apodization_us",
            "tau_maj_us",
            "sigma_tau_us",
        )
        if k in ckwargs_in
    }
    shape_resolved = PeakShape.coerce(shape)
    constraints = derive_window_fit_constraints(
        z,
        sigma,
        tau0_us,
        acquisition_us,
        shape=shape_resolved,
        **constraints_kwargs,
    )
    fit_kwargs_inner = dict(constraints.fit_kwargs_inner)
    fit_kwargs_inner.setdefault("shape", shape_resolved)
    # A baseline the initial fit carries (the early conservative-phase
    # leakage-wing term) stays in the model across the whole consolidation
    # chain: the joint refit, the merge/knockout refits, and the iterative
    # cleanup all re-fit it jointly, so removing a peak never re-exposes
    # the pedestal as that peak's "evidence".
    if (
        initial_fit.fit.baseline_order is not None
        and initial_fit.fit.baseline_offset_scale
    ):
        fit_kwargs_inner["baseline_order"] = int(initial_fit.fit.baseline_order)
        fit_kwargs_inner["baseline_offset_scale"] = float(
            initial_fit.fit.baseline_offset_scale
        )

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
        # Inherited (round-start) peak offsets: the cleanup judges these in
        # budget-free currency (see ``iterative_aicc_cleanup``); only this
        # round's rescue-origin additions face the skirt budget.
        inherited_offsets = [float(pk.offset_mhz) for pk in current.fit.peaks]

        rescue = attempt_residual_rescue(
            u,
            z,
            sigma,
            current.fit,
            tau0_us,
            acquisition_us,
            snr_threshold=snr_threshold,
            prominence_threshold=prominence_threshold,
            significance=rescue_significance,
            max_peaks=rescue_max_peaks,
            conservative_kwargs=ckwargs_in,
            excluded_offsets=rejected_offsets if rejected_offsets else None,
            spur_mask=spur_mask,
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
        # Stage 2b calibration is plumbed: the calibration's bidirectional
        # Gaussian prior (``tau_penalty_sigma_us`` is the signature) stays
        # active in the joint fit -- the penalty soft-anchors tau near
        # ``tau_maj`` while letting data-driven drift narrow or broaden
        # the line shape when chi2 strongly prefers it. Without calibration,
        # tau is free under the legacy apodization-anchored bounds; start at
        # the rescue's value (the apodization-override structural mitigation
        # for w198-like cases).
        union_init = list(current.peaks) + list(rescue.fit.peaks)
        calibrated = (
            fit_kwargs_inner.get("tau_penalty_sigma_us") is not None
            and float(fit_kwargs_inner.get("tau_penalty_sigma_us") or 0.0) > 0.0
        )
        if calibrated:
            joint_tau_start = float(fit_kwargs_inner["tau_penalty_reference"])
            joint_kwargs = dict(fit_kwargs_inner)
            joint_kwargs["fit_tau"] = True
        else:
            joint_tau_start = _clamp(
                float(rescue.fit.tau_us),
                constraints.tau_bounds[0],
                constraints.tau_bounds[1],
            )
            joint_kwargs = fit_kwargs_inner
        joint = fit_window(
            u,
            z,
            sigma,
            union_init,
            joint_tau_start,
            acquisition_us,
            spur_mask=spur_mask,
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
            u,
            z,
            sigma,
            joint,
            float(joint.tau_us),
            acquisition_us,
            fit_kwargs_inner=fit_kwargs_inner,
            merge_separation_factor=merge_separation_factor,
            structural_merge_factor=structural_merge_factor,
            min_pair_separation_resolution_factor=merge_resolution_factor,
            overfit_amp_ratio_band=overfit_amp_ratio_band,
            overfit_amp_ratio_threshold=overfit_amp_ratio_threshold,
            n_eff_kind=n_eff_kind,
            spur_mask=spur_mask,
            gate_budget_extra=gate_budget_extra,
        )

        # Per-peak knockout produces persisted diagnostics (n_eff,
        # aicc_delta, supported) on the merged joint fit. This is the
        # diagnostic snapshot before iterative cleanup -- callers can
        # see what the per-peak gate said about the joint set.
        # The sink captures every (K-1) refit so iterative_aicc_cleanup can
        # reuse them in its first iteration, skipping the duplicate work.
        joint_refit_sink: dict[int, WindowFitResult] = {}
        joint_knockouts = knockout_test(
            u,
            z,
            sigma,
            merged_joint_fit,
            acquisition_us,
            fit_kwargs_inner=fit_kwargs_inner,
            n_eff_kind=n_eff_kind,
            significance=knockout_significance,
            spur_mask=spur_mask,
            gate_budget_extra=gate_budget_extra,
            refit_sink=joint_refit_sink,
        )
        # Iterative AICc cleanup: drops the worst offender, refits,
        # repeats. Non-iterative drops kill duplicate clusters wholesale
        # (every member individually fails when its twins absorb on
        # refit); the iterative loop redistributes the dropped peak's
        # contribution and re-evaluates, so a cluster of N duplicates
        # of one real feature converges to a single supported peak.
        # Grid spacing for the +/-1-bin tolerance used in the protected-peak
        # matching, the rescue-survival check below, and the blacklist.
        u_sorted = np.sort(u)
        df_mhz = float(np.min(np.diff(u_sorted))) if u_sorted.size >= 2 else 0.0
        survival_tol = max(df_mhz, 1e-3)
        pruned_fit_candidate, n_pruned_total = iterative_aicc_cleanup(
            u,
            z,
            sigma,
            merged_joint_fit,
            acquisition_us,
            fit_kwargs_inner=fit_kwargs_inner,
            n_eff_kind=n_eff_kind,
            spur_mask=spur_mask,
            gate_budget_extra=gate_budget_extra,
            gate_background=gate_background,
            protected_offsets=inherited_offsets,
            protected_tol_mhz=survival_tol,
            initial_refits=joint_refit_sink,
        )
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
                abs(rescue_pk.offset_mhz - mo) <= survival_tol for mo in merged_offsets
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
            u,
            z,
            sigma,
            pruned_fit,
            acquisition_us,
            fit_kwargs_inner=fit_kwargs_inner,
            n_eff_kind=n_eff_kind,
            significance=knockout_significance,
            spur_mask=spur_mask,
            gate_budget_extra=gate_budget_extra,
        )

        chi2_after = pruned_fit.chi_squared
        tau_after = float(pruned_fit.tau_us)
        # A round must EARN its install: the consolidated fit replaces the
        # round-start fit only on a strict raw chi-squared improvement. The
        # joint refit relaxes every parameter (including tau, whose penalty
        # anchor can drag it off the data-preferred value), and the cleanup
        # can prune the round back to nothing gained -- without this guard
        # such a round still installed a strictly worse fit. Rejected rounds
        # leave ``current`` untouched; their candidates are already
        # blacklisted above, so the loop converges rather than re-proposing.
        round_improves = chi2_after < chi2_before
        reason_parts: List[str] = []
        if n_pruned_total > 0:
            reason_parts.append(f"knockout pruned {n_pruned_total} peak(s)")
        if n_merged > 0:
            reason_parts.append(f"merge collapsed {n_merged} pair(s)")
        if not reason_parts:
            reason = "joint refit consolidated rescue contribution"
        else:
            reason = "joint refit + " + " + ".join(reason_parts)
        if not round_improves:
            reason += " (rejected: consolidated chi-squared did not improve)"
        rounds.append(
            RescueRoundDiagnostics(
                round_idx=round_idx,
                rescue=rescue,
                joint_fit=joint,
                joint_knockouts=joint_knockouts,
                pruned_fit=pruned_fit if round_improves else None,
                pruned_knockouts=pruned_knockouts if round_improves else [],
                n_initial_peaks=n_initial,
                n_rescue_added=n_rescue_added,
                n_pruned_total=n_pruned_total,
                n_pruned_rescue_origin=n_pruned_rescue,
                chi2_before=chi2_before,
                chi2_after=chi2_after if round_improves else chi2_before,
                tau_us_before=tau_before,
                tau_us_after=tau_after if round_improves else tau_before,
                accepted=round_improves,
                reason=reason,
                n_merged=n_merged,
            )
        )
        if not round_improves:
            continue
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
