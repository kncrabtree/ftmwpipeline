"""
Convert Stage 5 in-flight fit records to the persistent core data structures.

Stage 5's plan executor (:mod:`ftmwpipeline.fitting.plan_execution`) produces
:class:`WindowOutcome` / :class:`PlanFitOutcome` -- lightweight working
records carrying the raw least-squares arrays needed to drive the
fixed-contributor walk and the renegotiation loops. The persistent user-facing
types in :mod:`ftmwpipeline.core.data_structures`
(:class:`FittedPeak`, :class:`FittingResult`, :class:`SpectralWindow`,
:class:`SpectrumFit`, and the dataclass twins :class:`AuditStep` /
:class:`KnockoutInfo` / :class:`ThawInfo` / :class:`ReplanInfo`) carry the
spectroscopic parameters, the audit / renegotiation history, and the
plan-level diagnostics that downstream consumers (serialization,
visualization, hand-edit) need.

This module is the **pure conversion layer** that bridges the two -- arrays
in, dataclasses out, no algorithm changes and no file IO. It is called once
at the end of an :func:`~ftmwpipeline.fitting.plan_execution.execute_plan`
invocation; the in-flight records stay alive for any subsequent algorithm
work that needs them, and the persistent snapshot is what the serialization
layer consumes and the dual-interface wrappers expose.

Decoupling
----------
The persistent twins live in ``core/`` and the converter copies fields into
them rather than re-exporting the algorithm-side dataclasses. This matches
the Stage 4 precedent (:class:`FitWindow` / :class:`MergeRequest` live in
``core/`` even though Stage 4 algorithms produce/consume them): the
persistence layer is then independent of the algorithm module's evolution.

Window padding
--------------
The :class:`SpectralWindow` materialized per window here uses the active-FT
slice produced by :func:`materialize_window` -- i.e. the Stage 4 ``freq_range``
with no baseline-context margin. Each window carries exactly the active-FT
region it is fit over.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import List, Optional, Union

import numpy as np

from ftmwpipeline.core.data_structures import (
    AuditStep,
    DoubletAlternativeInfo,
    FittedPeak,
    FittingResult,
    FitWindow,
    KnockoutInfo,
    ReplanInfo,
    RescueCandidateInfo,
    RescueRoundInfo,
    Sideband,
    SpectralWindow,
    SpectrumFit,
    ThawInfo,
    WindowPlan,
)

from .doublet_alternative import DoubletAdjudication
from .peak_model import effective_tau_shape, molecular_frequency, sideband_sign
from .plan_execution import (
    PlanFitOutcome,
    ReplanEvent,
    RescueEvent,
    ThawEvent,
    WindowOutcome,
)
from .residual_screening import ResidualPeakCandidate
from .window_fit import AddStep, KnockoutResult

__all__ = [
    "build_covariance_param_labels",
    "sort_fitting_result_by_frequency",
    "window_outcome_to_spectral_window",
    "window_outcome_to_fitting_result",
    "plan_fit_outcome_to_spectrum_fit",
    "FittedLineView",
    "outcome_line_views",
]

SidebandLike = Union[Sideband, str]


# ---------------------------------------------------------------------------
# Shared fitted-line view (DRY: one decision surface for the cleanup)
# ---------------------------------------------------------------------------
import math
from dataclasses import dataclass


@dataclass
class FittedLineView:
    """The per-line fields the SNR-prune / VIF-collapse decisions reason about.

    One thin projection of a fitted line -- ``(frequency_mhz, offset_mhz,
    amplitude, amplitude_error, phase, snr, origin)`` plus its ``index`` in the
    source peak list -- computable from *either* a Stage-5
    :class:`~ftmwpipeline.fitting.plan_execution.WindowOutcome` (the in-walk
    cleanup) or a persisted
    :class:`~ftmwpipeline.core.data_structures.FittingResult` (the post-fit
    user-edit path). The prune/collapse decision code takes a list of these, so
    there is one decision path and no per-iteration ``FittingResult``
    construction inside the fit walk.

    ``offset_mhz`` is the signed-baseband offset the in-walk refit edits in;
    ``index`` is the line's position in the source list so an edit can target it.
    """

    frequency_mhz: float
    offset_mhz: float
    amplitude: float
    amplitude_error: Optional[float]
    phase: float
    snr: Optional[float]
    origin: str
    index: int

    def amplitude_vif(self) -> Optional[float]:
        """Diagonal amplitude variance-inflation factor ``(amp_err/amp)*snr``.

        Mirrors :func:`ftmwpipeline.fitting.validation.amplitude_vif` on the
        view's fields (the overfit discriminant). ``None`` when any input is
        missing / non-finite or the amplitude is zero."""
        amp = float(self.amplitude)
        amp_err = self.amplitude_error
        snr = self.snr
        if amp_err is None or snr is None:
            return None
        if not (math.isfinite(amp) and math.isfinite(amp_err) and math.isfinite(snr)):
            return None
        if abs(amp) <= 0.0:
            return None
        return abs(amp_err / amp) * float(snr)


def outcome_line_views(
    outcome: WindowOutcome,
    *,
    sideband: SidebandLike,
    acquisition_us: float,
) -> List[FittedLineView]:
    """Fitted-line views for a live :class:`WindowOutcome` (the in-walk path).

    Derives ``snr`` / ``amplitude_error`` by the same formulas
    :func:`window_outcome_to_fitting_result` uses, so a view computed here
    matches the line's persisted fields exactly. Every line carries
    ``origin="auto"`` -- the automatic fit has no user-origin peaks (those enter
    only on the post-fit edit path)."""
    inner = outcome.fit.fit
    center_mhz = _window_center(outcome)
    s = sideband_sign(sideband)
    rms_mean = float(np.mean(np.asarray(outcome.rms_noise, dtype=float)))
    tau_us = inner.tau_us
    tau_eff = (
        effective_tau_shape(inner.shape, tau_us, acquisition_us)
        if tau_us > 0
        else float("nan")
    )
    views: List[FittedLineView] = []
    for i, peak in enumerate(inner.peaks):
        peak_error = inner.peak_errors[i] if i < len(inner.peak_errors) else None
        if rms_mean > 0 and np.isfinite(tau_eff):
            snr: Optional[float] = float(0.5 * peak.amplitude * tau_eff / rms_mean)
        else:
            snr = None
        amp_err = (
            float(peak_error.amplitude)
            if peak_error is not None and np.isfinite(peak_error.amplitude)
            else None
        )
        views.append(
            FittedLineView(
                frequency_mhz=float(center_mhz + s * peak.offset_mhz),
                offset_mhz=float(peak.offset_mhz),
                amplitude=float(peak.amplitude),
                amplitude_error=amp_err,
                phase=float(peak.phase),
                snr=snr,
                origin="auto",
                index=i,
            )
        )
    return views


# ---------------------------------------------------------------------------
# Covariance label builder
# ---------------------------------------------------------------------------
def build_covariance_param_labels(
    n_peaks: int,
    fit_tau: bool,
    baseline_order: Optional[int],
) -> List[str]:
    """Return the ordered parameter labels for a per-window covariance matrix.

    The label ordering mirrors the NLS parameter vector assembled by
    :func:`~ftmwpipeline.fitting.window_fit.fit_window`:

    * ``amplitude_{i}``, ``offset_{i}``, ``phase_{i}``  for each peak *i* (0-based),
      peak-major (all three params for peak 0, then peak 1, …)
    * ``tau``  only when *fit_tau* is True
    * ``baseline_re_{k}``  for k in 0..baseline_order (real polynomial coefficients)
    * ``baseline_im_{k}``  for k in 0..baseline_order (imag polynomial coefficients)

    The baseline block is omitted when *baseline_order* is None or negative.

    Parameters
    ----------
    n_peaks:
        Number of fitted peaks (K).
    fit_tau:
        Whether tau was a free LSQ parameter in this window.
    baseline_order:
        Polynomial baseline order (non-negative) or None / negative when no
        baseline was fitted.
    """
    labels: List[str] = []
    for i in range(n_peaks):
        labels.append(f"amplitude_{i}")
        labels.append(f"offset_{i}")
        labels.append(f"phase_{i}")
    if fit_tau:
        labels.append("tau")
    if baseline_order is not None and baseline_order >= 0:
        n_base = baseline_order + 1
        for k in range(n_base):
            labels.append(f"baseline_re_{k}")
        for k in range(n_base):
            labels.append(f"baseline_im_{k}")
    return labels


def sort_fitting_result_by_frequency(result: FittingResult) -> None:
    """Reorder a window's fitted peaks (and covariance) by ascending frequency.

    The NLS assembles a window's peaks in seed order; sorting them by molecular
    frequency makes the persisted line list, the report and ``fit show`` tables,
    the numeric covariance matrix, and the correlation heatmap all read in one
    ascending order. Sorts ``fitted_peaks`` in place and applies the matching
    block permutation to ``covariance``: the peak-major ``(amplitude, offset,
    phase)`` triple for each peak moves as a unit while the shared ``tau`` and
    baseline coefficients keep their tail positions, so the positional
    ``covariance_param_labels`` stay valid and every peak keeps its own
    covariance block (``sqrt(diag)`` still matches that peak's stored errors).

    In place; idempotent (a no-op when the peaks are already ascending). If a
    covariance is present but its layout does not match the documented
    peak-major form, neither the peaks nor the covariance are reordered, so the
    two never fall out of correspondence.
    """
    peaks = result.fitted_peaks
    n = len(peaks)
    if n < 2:
        return
    order = sorted(range(n), key=lambda i: float(peaks[i].frequency_mhz))
    if order == list(range(n)):
        return

    cov = result.covariance
    labels = result.covariance_param_labels
    if cov is not None and labels is not None:
        arr = np.asarray(cov, dtype=float)
        layout_ok = (
            arr.ndim == 2
            and arr.shape[0] == arr.shape[1]
            and arr.shape[0] >= 3 * n
            and all(labels[3 * i] == f"amplitude_{i}" for i in range(n))
        )
        if not layout_ok:
            # Unexpected layout: leave both peaks and covariance untouched
            # rather than risk a mislabeled matrix.
            return
        perm = [3 * i + k for i in order for k in (0, 1, 2)]
        perm.extend(range(3 * n, arr.shape[0]))  # tau / baseline tail stays put
        result.covariance = arr[np.ix_(perm, perm)]

    result.fitted_peaks = [peaks[i] for i in order]


# ---------------------------------------------------------------------------
# Per-record converters (private)
# ---------------------------------------------------------------------------
def _convert_audit_step(step: AddStep) -> AuditStep:
    """Copy an algorithm-side :class:`AddStep` into the persistent twin."""
    return AuditStep(
        n_peaks_before=step.n_peaks_before,
        candidate_offset_mhz=step.candidate_offset_mhz,
        chi2_before=step.chi2_before,
        chi2_after=step.chi2_after,
        f_statistic=step.f_statistic,
        p_value=step.p_value,
        aic_before=step.aic_before,
        aic_after=step.aic_after,
        separation_ok=step.separation_ok,
        decision=step.decision,
        reason=step.reason,
        n_eff=step.n_eff,
        aicc_delta=step.aicc_delta,
    )


def _convert_thaw_event(event: ThawEvent) -> ThawInfo:
    """Copy an algorithm-side :class:`ThawEvent` into the persistent twin."""
    return ThawInfo(
        dependent_window_id=event.dependent_window_id,
        primary_window_id=event.primary_window_id,
        contributor_peak_index=event.contributor_peak_index,
        contributor_frequency_mhz=event.contributor_frequency_mhz,
        edge_side=event.edge_side,
        edge_coherence_before=event.edge_coherence_before,
        edge_coherence_after=event.edge_coherence_after,
        accepted=event.accepted,
        reason=event.reason,
    )


def _convert_replan_event(event: ReplanEvent) -> ReplanInfo:
    """Copy an algorithm-side :class:`ReplanEvent` into the persistent twin."""
    return ReplanInfo(
        triggering_window_id=event.triggering_window_id,
        partner_window_id=event.partner_window_id,
        surviving_window_id=event.surviving_window_id,
        edge_side=event.edge_side,
        edge_coherence_before=event.edge_coherence_before,
        revision_before=event.revision_before,
        revision_after=event.revision_after,
        accepted=event.accepted,
        reason=event.reason,
    )


def _convert_knockout(knockout: KnockoutResult) -> KnockoutInfo:
    """Copy a :class:`KnockoutResult` into the persistent :class:`KnockoutInfo`."""
    return KnockoutInfo(
        delta_chi2=knockout.delta_chi2,
        expected_delta_chi2=knockout.expected_delta_chi2,
        supported=knockout.supported,
        p_value=knockout.p_value,
        n_eff=knockout.n_eff,
        aicc_delta=knockout.aicc_delta,
    )


def _convert_rescue_candidate(
    candidate: ResidualPeakCandidate,
) -> RescueCandidateInfo:
    """Copy a :class:`ResidualPeakCandidate` into the persistent twin."""
    return RescueCandidateInfo(
        frequency_mhz=float(candidate.frequency_mhz),
        magnitude=float(candidate.magnitude),
        snr=float(candidate.snr),
    )


def _convert_doublet_adjudication(
    adj: DoubletAdjudication,
    center_mhz: float,
    s: float,
) -> DoubletAlternativeInfo:
    """Convert a fit-frame :class:`DoubletAdjudication` to molecular-frame persistent twin."""
    freq_a = float(center_mhz + s * adj.offset_a_mhz)
    freq_b = float(center_mhz + s * adj.offset_b_mhz)
    merged_freq = (
        float(center_mhz + s * adj.merged_offset_mhz)
        if not (adj.merged_offset_mhz != adj.merged_offset_mhz)  # NaN check
        else float("nan")
    )
    return DoubletAlternativeInfo(
        frequency_a_mhz=freq_a,
        frequency_b_mhz=freq_b,
        amplitude_a=float(adj.amplitude_a),
        amplitude_b=float(adj.amplitude_b),
        separation_res_elements=float(adj.separation_res_elements),
        amp_ratio=float(adj.amp_ratio),
        chi2r_production=float(adj.chi2r_production),
        chi2r_merged=float(adj.chi2r_merged),
        delta_chi2_raw=float(adj.delta_chi2_raw),
        delta_aicc=float(adj.delta_aicc),
        merged_frequency_mhz=merged_freq,
        merged_amplitude=float(adj.merged_amplitude),
        merged_phase=float(adj.merged_phase),
        merged_tau_us=float(adj.merged_tau_us),
        merged_success=bool(adj.merged_success),
        orth_evidence_delta_chi2=float(adj.orth_evidence_delta_chi2),
        orth_evidence_n_params=int(adj.orth_evidence_n_params),
        support_bins=int(adj.support_bins),
    )


def _convert_rescue_event(event: RescueEvent) -> RescueRoundInfo:
    """Copy a :class:`RescueEvent` into the persistent :class:`RescueRoundInfo`."""
    return RescueRoundInfo(
        window_id=int(event.window_id),
        round_idx=int(event.round_idx),
        n_initial_peaks=int(event.n_initial_peaks),
        n_rescue_added=int(event.n_rescue_added),
        n_pruned_total=int(event.n_pruned_by_knockout),
        n_pruned_rescue_origin=int(event.n_pruned_rescue_origin),
        n_merged=int(event.n_merged),
        chi2_before=float(event.chi2_before),
        chi2_after=float(event.chi2_after),
        tau_us_before=float(event.tau_us_before),
        tau_us_after=float(event.tau_us_after),
        accepted=bool(event.accepted),
        reason=str(event.reason),
        candidates=[_convert_rescue_candidate(c) for c in event.candidates],
    )


def _window_center(outcome: WindowOutcome) -> float:
    """Recover the molecular reference frequency from a window outcome.

    The plan executor stashes the midpoint of the fit window's freq_range as
    a private ``_center_mhz`` attribute on the outcome (see
    :func:`ftmwpipeline.fitting.plan_execution._window_center_mhz` for the
    contract); the converter reads it back here. Raises if absent so a
    misuse of the converter surfaces immediately.
    """
    center = getattr(outcome, "_center_mhz", None)
    if center is None:
        raise ValueError(
            "WindowOutcome is missing its molecular reference frequency "
            "(_center_mhz attribute). Only outcomes produced by "
            "execute_plan carry this; build a WindowOutcome through the "
            "plan executor or set _center_mhz before conversion."
        )
    return float(center)


def _match_free_peak_index(
    fitted_freq_mhz: float,
    free_peak_indices: Sequence[int],
    peak_frequencies_mhz: Sequence[float],
) -> int:
    """Nearest-frequency match from a fitted line to a Stage 3 peak index.

    The conservative add-one-peak loop reorders / re-seeds candidates and may
    introduce blend-aware K=2/K=3 lines without their own Stage 3 detections,
    so we cannot trust positional alignment between ``fit.peaks`` and
    ``free_peak_indices``. The robust contract is "nearest molecular
    frequency". Multiple fitted peaks can share a Stage 3 index (the
    blend-aware case).

    Falls back to the seed candidate index when ``free_peak_indices`` is
    empty (a degenerate but reachable case); the caller is responsible for
    handling that.
    """
    if not free_peak_indices:
        return -1
    distances = [
        (idx, abs(float(peak_frequencies_mhz[idx]) - fitted_freq_mhz))
        for idx in free_peak_indices
    ]
    best_idx, _ = min(distances, key=lambda item: item[1])
    return int(best_idx)


# ---------------------------------------------------------------------------
# Public converters
# ---------------------------------------------------------------------------
def window_outcome_to_spectral_window(
    outcome: WindowOutcome,
    fit_window: FitWindow,
    *,
    sideband: SidebandLike,
) -> SpectralWindow:
    """Build a :class:`SpectralWindow` from a Stage 5 :class:`WindowOutcome`.

    The window carries the molecular-frequency grid recovered from the
    outcome's signed-baseband offset grid (``f = f_c + s * delta``), the
    complex active-FT slice, and the original ``FitWindow.freq_range``.
    ``parent_ft`` is ``None`` -- the active-FT is not persisted as a
    :class:`ComplexFT` (it is regenerated on demand from the FID), so there
    is no parent to link.

    The slice spans the Stage 4 ``freq_range`` with no baseline-context margin:
    each window carries exactly the active-FT region it is fit over.
    """
    center_mhz = _window_center(outcome)
    freq_array = molecular_frequency(outcome.offset_grid_mhz, center_mhz, sideband)
    return SpectralWindow(
        parent_ft=None,
        freq_array=np.asarray(freq_array, dtype=float),
        complex_spectrum=np.asarray(outcome.complex_spectrum, dtype=np.complex128),
        freq_range=(
            float(fit_window.freq_range[0]),
            float(fit_window.freq_range[1]),
        ),
        window_id=fit_window.window_id,
    )


def window_outcome_to_fitting_result(
    outcome: WindowOutcome,
    fit_window: FitWindow,
    *,
    sideband: SidebandLike,
    peak_frequencies_mhz: Sequence[float],
    acquisition_us: float,
) -> FittingResult:
    """Persistent :class:`FittingResult` from a Stage 5 :class:`WindowOutcome`.

    Maps every fitted line back from the window's signed-baseband offset to
    the molecular frequency axis (``f = f_c + s * delta``), attaches the
    matching knockout entry, propagates the conservative add-one-peak audit
    trail, and copies the dependent-side thaw events touching this window.

    Parameters
    ----------
    outcome : WindowOutcome
        The plan executor's in-flight record for one window.
    fit_window : FitWindow
        The Stage 4 plan record for the same window.
    sideband : Sideband or str
        Pipeline sideband.
    peak_frequencies_mhz : sequence of float
        Frequencies of *all* Stage 3 peaks, indexed by ``free_peak_indices``
        (so a fitted line can be matched back to its Stage 3 index by
        nearest-frequency lookup).
    acquisition_us : float
        Active acquisition length ``T`` (microseconds), used for the
        effective-tau SNR estimate.
    """
    fit = outcome.fit  # ConservativeFitResult
    inner = fit.fit  # WindowFitResult

    center_mhz = _window_center(outcome)
    s = sideband_sign(sideband)

    # Build FittedPeak per converged line.
    rms_mean = float(np.mean(np.asarray(outcome.rms_noise, dtype=float)))
    tau_us = inner.tau_us
    tau_eff = (
        effective_tau_shape(inner.shape, tau_us, acquisition_us)
        if tau_us > 0
        else float("nan")
    )
    tau_error = inner.tau_error
    # d(1/tau)/d(tau) = -1/tau^2 -> decay_rate_error = tau_error / tau^2.
    decay_rate = 1.0 / tau_us if tau_us > 0 else None
    decay_rate_error = (
        float(tau_error) / (tau_us * tau_us)
        if (tau_error is not None and np.isfinite(tau_error) and tau_us > 0)
        else None
    )

    knockouts = list(fit.knockouts)
    fitted_peaks: list[FittedPeak] = []
    for i, peak in enumerate(inner.peaks):
        freq_mhz = float(center_mhz + s * peak.offset_mhz)
        peak_error = inner.peak_errors[i] if i < len(inner.peak_errors) else None
        peak_index = _match_free_peak_index(
            freq_mhz, fit_window.free_peak_indices, peak_frequencies_mhz
        )
        knockout = _convert_knockout(knockouts[i]) if i < len(knockouts) else None
        # SNR ~ on-resonance complex response / per-bin noise.
        snr: Optional[float]
        if rms_mean > 0 and np.isfinite(tau_eff):
            snr = float(0.5 * peak.amplitude * tau_eff / rms_mean)
        else:
            snr = None
        amp_err = (
            float(peak_error.amplitude)
            if peak_error is not None and np.isfinite(peak_error.amplitude)
            else None
        )
        freq_err = (
            float(peak_error.offset_mhz)
            if peak_error is not None and np.isfinite(peak_error.offset_mhz)
            else None
        )
        phase_err = (
            float(peak_error.phase)
            if peak_error is not None and np.isfinite(peak_error.phase)
            else None
        )
        chi_squared_for_peak = (
            float(knockout.delta_chi2) if knockout is not None else None
        )
        fitted_peaks.append(
            FittedPeak(
                peak_id=peak_index,
                frequency_mhz=freq_mhz,
                amplitude=float(peak.amplitude),
                decay_rate=decay_rate,
                phase=float(peak.phase),
                frequency_error=freq_err,
                amplitude_error=amp_err,
                decay_rate_error=decay_rate_error,
                phase_error=phase_err,
                snr=snr,
                chi_squared=chi_squared_for_peak,
                window_id=fit_window.window_id,
                knockout=knockout,
            )
        )

    shape_attr = inner.shape
    shape_str = shape_attr.value if hasattr(shape_attr, "value") else str(shape_attr)
    result = FittingResult(
        success=bool(inner.success),
        fitted_spectrum=np.asarray(outcome.full_fitted_spectrum, dtype=np.complex128),
        cost=float(inner.cost),
        iterations=len(fit.audit_trail),
        aic=float(inner.aic),
        reduced_chi2=float(inner.reduced_chi2),
        window=window_outcome_to_spectral_window(
            outcome, fit_window, sideband=sideband
        ),
        window_id=fit_window.window_id,
        shape=shape_str,
    )
    result.fitted_peaks = fitted_peaks

    # Per-window parameter covariance. Label it by the covariance's *actual*
    # dimension rather than trusting ``inner.fit_tau``: a cleanup / knockout
    # refit can lock tau (its covariance then has no tau row) while ``fit_tau``
    # stays True from the upstream tau-free determination. Try the tau-present
    # and tau-absent labelings and keep whichever matches, so a valid covariance
    # is always persisted.
    cov = inner.covariance
    if cov is not None:
        cov_arr = np.asarray(cov, dtype=float)
        if cov_arr.ndim == 2 and cov_arr.shape[0] == cov_arr.shape[1]:
            dim = cov_arr.shape[0]
            for tau_flag in (bool(inner.fit_tau), not bool(inner.fit_tau)):
                labels = build_covariance_param_labels(
                    n_peaks=len(inner.peaks),
                    fit_tau=tau_flag,
                    baseline_order=inner.baseline_order,
                )
                if len(labels) == dim:
                    result.covariance = cov_arr
                    result.covariance_param_labels = labels
                    break
            # else: genuine layout mismatch -- leave both None rather than
            # persist a mislabeled matrix.

    # Shared parameter: the per-window decay constant. ``fitted`` records
    # whether tau was determined by an LSQ that included it as a free
    # parameter (vs frozen-by-gate at ``tau0_us``). Reads
    # ``inner.tau_was_fit`` rather than ``inner.fit_tau`` because cleanup
    # refits (merge_close_peaks_cleanup / iterative_aicc_cleanup) and the
    # rescue's joint refit may overwrite ``fit_tau`` on the final
    # WindowFitResult even though the persisted ``tau_us`` came from a
    # tau-free fit upstream; ``tau_was_fit`` preserves the originating
    # determination through the cleanup chain. Disambiguates the two
    # ``error=None`` cases: frozen-by-gate (fitted=False, tau held at
    # tau0) vs free-but-singular-covariance (fitted=True, J^T J was
    # singular at the tau slot).
    tau_was_fit_attr = getattr(inner, "tau_was_fit", None)
    fitted_flag = (
        bool(tau_was_fit_attr) if tau_was_fit_attr is not None else bool(inner.fit_tau)
    )
    result.shared_parameters["tau_us"] = {
        "value": float(tau_us),
        "error": (
            float(tau_error)
            if tau_error is not None and np.isfinite(tau_error)
            else None
        ),
        "fitted": fitted_flag,
        "peak_ids": [p.peak_id for p in fitted_peaks],
    }

    # Fixed parameters: one entry per frozen ancestor line used in the fit. The
    # content is the ancestor's *fitted* (frequency, amplitude, phase) -- not a
    # Stage-3 contributor snapshot -- so the entries carry no Stage-3
    # ``peak_index`` (it is ``-1``); key by enumeration to keep the keys unique
    # (``_reconstruct_frozen_peaks`` reads every ``frozen_peak_*`` entry by value,
    # so the key index is immaterial on the round-trip).
    for i, frozen in enumerate(outcome.fixed_peaks):
        result.fixed_parameters[f"frozen_peak_{i}"] = {
            "peak_index": frozen.peak_index,
            "primary_window_id": frozen.primary_window_id,
            "frequency_mhz": frozen.frequency_mhz,
            "amplitude": frozen.model_peak.amplitude,
            "phase": frozen.model_peak.phase,
            "freeze_eligible": frozen.freeze_eligible,
        }

    result.residuals = np.asarray(outcome.full_residual, dtype=np.complex128)
    result.quality_metrics = {
        "edge_coherence_low": float(outcome.edge_coherence_low),
        "edge_coherence_high": float(outcome.edge_coherence_high),
        "n_fixed_contributors": float(len(outcome.fixed_peaks)),
    }
    # Leakage-wing baseline audit trail: whether the evidence trigger fired on
    # this window, the order, the triggering S_coh, and the fitted complex
    # coefficients. ``quality_metrics`` is ``Dict[str, float]``, so each
    # coefficient is recorded as a pair of scalar ``baseline_coeff{k}_re`` /
    # ``_im`` entries (``baseline_order`` says how many to expect). The inner
    # fit is the authority for the order / coefficients / scale: the early
    # (conservative-phase) baseline rides through the rescue chain's joint
    # refits, which re-fit its coefficients after the outcome's audit mirror
    # was stamped.
    if inner.baseline_order is not None and inner.baseline_coeffs is not None:
        coeffs = np.asarray(inner.baseline_coeffs, dtype=np.complex128)
        result.quality_metrics["baseline_applied"] = 1.0
        result.quality_metrics["baseline_order"] = float(inner.baseline_order)
        result.quality_metrics["baseline_edge_coherence"] = float(
            getattr(outcome, "baseline_edge_coherence", 0.0) or 0.0
        )
        result.quality_metrics["baseline_offset_scale"] = float(
            inner.baseline_offset_scale or 0.0
        )
        for k, c in enumerate(coeffs):
            result.quality_metrics[f"baseline_coeff{k}_re"] = float(c.real)
            result.quality_metrics[f"baseline_coeff{k}_im"] = float(c.imag)
    else:
        result.quality_metrics["baseline_applied"] = 0.0

    # Audit trail and per-window thaw events.
    result.audit_trail = [_convert_audit_step(s_) for s_ in fit.audit_trail]
    result.thaw_events = [_convert_thaw_event(e) for e in outcome.thaw_events]
    result.rescue_events = [_convert_rescue_event(e) for e in outcome.rescue_events]
    result.doublet_alternatives = [
        _convert_doublet_adjudication(adj, center_mhz, s)
        for adj in outcome.doublet_adjudications
    ]

    return result


def plan_fit_outcome_to_spectrum_fit(
    plan_outcome: PlanFitOutcome,
    plan: WindowPlan,
    *,
    sideband: SidebandLike,
    peak_frequencies_mhz: Sequence[float],
    peak_detection_passes: Optional[Sequence[str]] = None,
    acquisition_us: float,
    parameters: Optional[dict] = None,
    diagnostics: Optional[dict] = None,
) -> SpectrumFit:
    """Persistent :class:`SpectrumFit` from a Stage 5 :class:`PlanFitOutcome`.

    Walks ``plan_outcome.window_outcomes`` in ascending ``window_id`` order
    (the plan-level snapshot is order-agnostic so the persisted layout is
    stable), converts each to a :class:`FittingResult`, builds the merged
    global ``fitted_peaks`` list sorted by molecular frequency, and copies
    the plan-level thaw / replan histories and the final plan revision into
    the aggregate.

    Parameters
    ----------
    plan_outcome : PlanFitOutcome
        Plan executor output.
    plan : WindowPlan
        The plan that was executed. Used to look up each ``FitWindow`` for
        per-window conversion (so each :class:`FittingResult` carries the
        plan's ``freq_range`` and ``free_peak_indices`` context).
    sideband : Sideband or str
        Pipeline sideband.
    peak_frequencies_mhz : sequence of float
        Frequencies of *all* Stage 3 peaks.
    acquisition_us : float
        Active acquisition length ``T`` (microseconds).
    parameters : dict, optional
        Stage 5 parameters used to produce the fit
        (``tau0_us``, ``residual_edge_threshold``, etc.). Recorded verbatim
        on :attr:`SpectrumFit.parameters` so the persisted fit is
        self-describing. Defaults to an empty dict.
    diagnostics : dict, optional
        Plan-level diagnostics to attach.

    Raises
    ------
    KeyError
        If an outcome references a ``window_id`` that does not exist in
        ``plan.windows`` -- a sign that the plan and the outcome were
        produced against different plan revisions.
    """
    by_id = {w.window_id: w for w in plan.windows}

    window_fits: list[FittingResult] = []
    fitted_peaks: list[FittedPeak] = []
    for wid in sorted(plan_outcome.window_outcomes.keys()):
        outcome = plan_outcome.window_outcomes[wid]
        if wid not in by_id:
            raise KeyError(
                f"PlanFitOutcome window_id={wid} is not in plan.windows; the "
                f"plan and the outcome may be from different revisions"
            )
        fit_window = by_id[wid]
        fitting_result = window_outcome_to_fitting_result(
            outcome,
            fit_window,
            sideband=sideband,
            peak_frequencies_mhz=peak_frequencies_mhz,
            acquisition_us=acquisition_us,
        )
        window_fits.append(fitting_result)
        fitted_peaks.extend(fitting_result.fitted_peaks)

    fitted_peaks.sort(key=lambda p: p.frequency_mhz)

    return SpectrumFit(
        window_fits=window_fits,
        fitted_peaks=fitted_peaks,
        thaw_history=[_convert_thaw_event(e) for e in plan_outcome.thaw_history],
        replan_history=[_convert_replan_event(e) for e in plan_outcome.replan_history],
        rescue_history=[_convert_rescue_event(e) for e in plan_outcome.rescue_history],
        final_plan_revision=int(plan_outcome.final_plan_revision),
        parameters=dict(parameters) if parameters is not None else {},
        diagnostics=dict(diagnostics) if diagnostics is not None else {},
    )
