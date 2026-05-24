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
work that needs them, and the persistent snapshot is what gets serialized
(task 9) and exposed through the dual-interface wrappers (task 10).

Decoupling
----------
The persistent twins live in ``core/`` and the converter copies fields into
them rather than re-exporting the algorithm-side dataclasses. This matches
the Stage 4 precedent (:class:`FitWindow` / :class:`MergeRequest` live in
``core/`` even though Stage 4 algorithms produce/consume them): the
persistence layer is then independent of the algorithm module's evolution.

Window padding (D-6)
--------------------
The :class:`SpectralWindow` materialized per window here uses the bare
active-FT slice produced by :func:`materialize_window` -- i.e. the Stage 4
``freq_range`` with no baseline context margin. Window baseline padding per
D-6 is deferred (see ``dev-docs/planning/stage5-fitting.md`` "D8 open items").
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Union

import numpy as np

from ftmwpipeline.core.data_structures import (
    AuditStep,
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

from .peak_model import effective_tau, molecular_frequency, sideband_sign
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
    "window_outcome_to_spectral_window",
    "window_outcome_to_fitting_result",
    "plan_fit_outcome_to_spectrum_fit",
]

SidebandLike = Union[Sideband, str]


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

    The slice spans the bare Stage 4 ``freq_range``; D-6 baseline padding is
    deferred to a follow-up.
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
    tau_eff = effective_tau(tau_us, acquisition_us) if tau_us > 0 else float("nan")
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
    )
    result.fitted_peaks = fitted_peaks

    # Shared parameter: the per-window decay constant.
    result.shared_parameters["tau_us"] = {
        "value": float(tau_us),
        "error": (
            float(tau_error)
            if tau_error is not None and np.isfinite(tau_error)
            else None
        ),
        "peak_ids": [p.peak_id for p in fitted_peaks],
    }

    # Fixed parameters: one entry per frozen contributor used in the fit.
    for frozen in outcome.fixed_peaks:
        key = f"frozen_peak_{frozen.peak_index}"
        result.fixed_parameters[key] = {
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

    # Audit trail and per-window thaw events.
    result.audit_trail = [_convert_audit_step(s_) for s_ in fit.audit_trail]
    result.thaw_events = [_convert_thaw_event(e) for e in outcome.thaw_events]
    result.rescue_events = [
        _convert_rescue_event(e) for e in outcome.rescue_events
    ]

    return result


def plan_fit_outcome_to_spectrum_fit(
    plan_outcome: PlanFitOutcome,
    plan: WindowPlan,
    *,
    sideband: SidebandLike,
    peak_frequencies_mhz: Sequence[float],
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
        rescue_history=[
            _convert_rescue_event(e) for e in plan_outcome.rescue_history
        ],
        final_plan_revision=int(plan_outcome.final_plan_revision),
        parameters=dict(parameters) if parameters is not None else {},
        diagnostics=dict(diagnostics) if diagnostics is not None else {},
    )
