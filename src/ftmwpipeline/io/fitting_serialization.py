"""
Stage 5 fitting-result serialization to HDF5.

The fitted line list is scientific output and may be hand-edited (a curator
may re-tag, drop, or annotate fitted peaks before publication), so it is
**persisted, not recomputed** in a flat hand-editable layout with loud
validation -- mirroring the Stage 3 peak-list and Stage 4 window-plan
serializers.

Persist / recompute split (resolves O5-8 against
``dev-docs/SERIALIZATION_STRATEGY.md``):

* **Persisted:** the per-peak fitted parameters and uncertainties, the
  shared per-window tau, the frozen-contributor summaries used in the fit,
  the conservative-loop audit trail, the per-window thaw events, the
  plan-level thaw + structural-replan histories, the final plan revision,
  the Stage 5 parameters used, and plan-level diagnostics. These are the
  things a hand-edit can touch.
* **Recomputed on load (not persisted):** the per-window
  :class:`SpectralWindow` (the active-FT slice -- regenerable from the FID
  plus the canonical Stage 1 settings plus the fit window's freq_range),
  the fitted complex spectrum, and the complex residual. Re-evaluating
  ``model_spectrum`` from the persisted parameters reproduces them
  bit-for-bit, and they are large arrays -- the SERIALIZATION spec's
  lightweight-file invariant forbids storing them.

HDF5 layout (under the caller-provided group, e.g. ``/stage5_fitting``)::

    .attrs:
        n_windows, n_fitted_peaks, creation_time, stage_name,
        final_plan_revision,
        parameters       (JSON)  -- Stage 5 parameters used
        diagnostics      (JSON)  -- plan-level diagnostics
        thaw_history     (JSON)  -- plan-level chronological thaw events
        replan_history   (JSON)  -- plan-level structural replans
        rescue_history   (JSON)  -- plan-level chronological rescue rounds
    windows/
        window_0000/
            .attrs:
                window_id, success, cost, iterations, aic, reduced_chi2,
                tau_us, tau_error          (NaN if None),
                tau_fitted                  (i1, -1 if unknown from older file),
                edge_coherence_low, edge_coherence_high,
                fixed_parameters (JSON),    -- frozen-contributor summaries
                quality_metrics  (JSON),
                audit_trail      (JSON),    -- per-window AuditStep list
                thaw_events      (JSON),    -- per-window ThawInfo list
                rescue_events    (JSON)     -- per-window RescueRoundInfo list
            peaks/
                peak_id                          [i8]
                frequency_mhz                    [f8]
                amplitude                        [f8]
                phase                            [f8]
                decay_rate                       [f8]    (NaN -> None)
                frequency_error                  [f8]    (NaN -> None)
                amplitude_error                  [f8]
                phase_error                      [f8]
                decay_rate_error                 [f8]
                snr                              [f8]    (NaN -> None)
                chi_squared                      [f8]    (NaN -> None)
                window_id                        [i8]
                knockout_delta_chi2              [f8]    (NaN -> no knockout)
                knockout_expected_delta_chi2     [f8]
                knockout_supported               [i1]    (-1 -> no knockout)
                knockout_p_value                 [f8]    (NaN -> no knockout
                                                          or older file
                                                          predating this col)
                knockout_n_eff                   [f8]    (NaN -> no knockout
                                                          or older file
                                                          predating this col)
                knockout_aicc_delta              [f8]    (NaN -> no knockout
                                                          or older file
                                                          predating this col)
        window_0001/ ...

Round-trip contract: ``save`` -> hand-edit -> ``load`` returns the edited
fit. The merged global :attr:`SpectrumFit.fitted_peaks` list is rebuilt on
load (sorted by molecular frequency) from the per-window peaks; it is not
stored independently. A malformed group (missing required attr/dataset,
mismatched peak-column lengths, unknown audit-step decision) raises
:class:`ValueError` rather than silently dropping or guessing.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

from ..core.data_structures import (
    AuditStep,
    FittedPeak,
    FittingResult,
    KnockoutInfo,
    ReplanInfo,
    RescueCandidateInfo,
    RescueRoundInfo,
    SpectralWindow,
    SpectrumFit,
    ThawInfo,
)

__all__ = [
    "save_spectrum_fit_to_hdf5",
    "load_spectrum_fit_from_hdf5",
]


# --- peak column layout ----------------------------------------------------
_PEAK_COLUMNS = (
    "peak_id",
    "frequency_mhz",
    "amplitude",
    "phase",
    "decay_rate",
    "frequency_error",
    "amplitude_error",
    "phase_error",
    "decay_rate_error",
    "snr",
    "chi_squared",
    "window_id",
    "knockout_delta_chi2",
    "knockout_expected_delta_chi2",
    "knockout_supported",
)
# Columns added after the v1 schema was set. Older files won't have them;
# load tolerates missing entries by substituting NaN.
_OPTIONAL_PEAK_COLUMNS = (
    "knockout_p_value",
    "knockout_n_eff",
    "knockout_aicc_delta",
)

_VALID_AUDIT_DECISIONS = {
    "seed",
    "seed-blend",
    "accept",
    "promote",
    "tentative",
    "reject",
}
_VALID_EDGE_SIDES = {"low", "high"}


# ---------------------------------------------------------------------------
# JSON encoders for the persistent twins
# ---------------------------------------------------------------------------
def _audit_step_to_json(step: AuditStep) -> Dict[str, Any]:
    return {
        "n_peaks_before": int(step.n_peaks_before),
        "candidate_offset_mhz": float(step.candidate_offset_mhz),
        "chi2_before": float(step.chi2_before),
        "chi2_after": float(step.chi2_after),
        "f_statistic": float(step.f_statistic),
        "p_value": float(step.p_value),
        "aic_before": float(step.aic_before),
        "aic_after": float(step.aic_after),
        "separation_ok": bool(step.separation_ok),
        "decision": str(step.decision),
        "reason": str(step.reason),
        "n_eff": float(step.n_eff),
        "aicc_delta": float(step.aicc_delta),
    }


def _json_to_audit_step(blob: Dict[str, Any], where: str) -> AuditStep:
    try:
        decision = str(blob["decision"])
    except KeyError as exc:
        raise ValueError(f"{where} missing required field {exc.args[0]!r}") from exc
    if decision not in _VALID_AUDIT_DECISIONS:
        raise ValueError(
            f"{where} has unknown decision {decision!r}; "
            f"expected one of {sorted(_VALID_AUDIT_DECISIONS)}"
        )
    return AuditStep(
        n_peaks_before=int(blob["n_peaks_before"]),
        candidate_offset_mhz=float(blob["candidate_offset_mhz"]),
        chi2_before=float(blob["chi2_before"]),
        chi2_after=float(blob["chi2_after"]),
        f_statistic=float(blob["f_statistic"]),
        p_value=float(blob["p_value"]),
        aic_before=float(blob["aic_before"]),
        aic_after=float(blob["aic_after"]),
        separation_ok=bool(blob["separation_ok"]),
        decision=decision,
        reason=str(blob.get("reason", "")),
        # n_eff / aicc_delta are optional on older or hand-edited audit
        # blobs that omit the AICc-with-n_eff gate diagnostics.
        n_eff=float(blob.get("n_eff", float("nan"))),
        aicc_delta=float(blob.get("aicc_delta", float("nan"))),
    )


def _thaw_info_to_json(event: ThawInfo) -> Dict[str, Any]:
    return {
        "dependent_window_id": int(event.dependent_window_id),
        "primary_window_id": int(event.primary_window_id),
        "contributor_peak_index": int(event.contributor_peak_index),
        "contributor_frequency_mhz": float(event.contributor_frequency_mhz),
        "edge_side": str(event.edge_side),
        "edge_coherence_before": float(event.edge_coherence_before),
        "edge_coherence_after": float(event.edge_coherence_after),
        "accepted": bool(event.accepted),
        "reason": str(event.reason),
    }


def _json_to_thaw_info(blob: Dict[str, Any], where: str) -> ThawInfo:
    edge_side = str(blob.get("edge_side", ""))
    if edge_side not in _VALID_EDGE_SIDES:
        raise ValueError(
            f"{where} has invalid edge_side {edge_side!r}; "
            f"expected one of {sorted(_VALID_EDGE_SIDES)}"
        )
    return ThawInfo(
        dependent_window_id=int(blob["dependent_window_id"]),
        primary_window_id=int(blob["primary_window_id"]),
        contributor_peak_index=int(blob["contributor_peak_index"]),
        contributor_frequency_mhz=float(blob["contributor_frequency_mhz"]),
        edge_side=edge_side,
        edge_coherence_before=float(blob["edge_coherence_before"]),
        edge_coherence_after=float(blob["edge_coherence_after"]),
        accepted=bool(blob["accepted"]),
        reason=str(blob.get("reason", "")),
    )


def _replan_info_to_json(event: ReplanInfo) -> Dict[str, Any]:
    return {
        "triggering_window_id": int(event.triggering_window_id),
        "partner_window_id": int(event.partner_window_id),
        "surviving_window_id": int(event.surviving_window_id),
        "edge_side": str(event.edge_side),
        "edge_coherence_before": float(event.edge_coherence_before),
        "revision_before": int(event.revision_before),
        "revision_after": int(event.revision_after),
        "accepted": bool(event.accepted),
        "reason": str(event.reason),
    }


def _json_to_replan_info(blob: Dict[str, Any], where: str) -> ReplanInfo:
    edge_side = str(blob.get("edge_side", ""))
    if edge_side not in _VALID_EDGE_SIDES:
        raise ValueError(
            f"{where} has invalid edge_side {edge_side!r}; "
            f"expected one of {sorted(_VALID_EDGE_SIDES)}"
        )
    return ReplanInfo(
        triggering_window_id=int(blob["triggering_window_id"]),
        partner_window_id=int(blob["partner_window_id"]),
        surviving_window_id=int(blob["surviving_window_id"]),
        edge_side=edge_side,
        edge_coherence_before=float(blob["edge_coherence_before"]),
        revision_before=int(blob["revision_before"]),
        revision_after=int(blob["revision_after"]),
        accepted=bool(blob["accepted"]),
        reason=str(blob.get("reason", "")),
    )


def _rescue_candidate_to_json(c: RescueCandidateInfo) -> Dict[str, Any]:
    return {
        "frequency_mhz": float(c.frequency_mhz),
        "magnitude": float(c.magnitude),
        "snr": float(c.snr),
    }


def _json_to_rescue_candidate(
    blob: Dict[str, Any], where: str
) -> RescueCandidateInfo:
    try:
        return RescueCandidateInfo(
            frequency_mhz=float(blob["frequency_mhz"]),
            magnitude=float(blob["magnitude"]),
            snr=float(blob["snr"]),
        )
    except KeyError as exc:
        raise ValueError(f"{where} missing required field {exc.args[0]!r}") from exc


def _rescue_round_to_json(r: RescueRoundInfo) -> Dict[str, Any]:
    return {
        "window_id": int(r.window_id),
        "round_idx": int(r.round_idx),
        "n_initial_peaks": int(r.n_initial_peaks),
        "n_rescue_added": int(r.n_rescue_added),
        "n_pruned_total": int(r.n_pruned_total),
        "n_pruned_rescue_origin": int(r.n_pruned_rescue_origin),
        "n_merged": int(r.n_merged),
        "chi2_before": float(r.chi2_before),
        "chi2_after": float(r.chi2_after),
        "tau_us_before": float(r.tau_us_before),
        "tau_us_after": float(r.tau_us_after),
        "accepted": bool(r.accepted),
        "reason": str(r.reason),
        "candidates": [_rescue_candidate_to_json(c) for c in r.candidates],
    }


def _json_to_rescue_round(blob: Dict[str, Any], where: str) -> RescueRoundInfo:
    try:
        candidates_raw = blob.get("candidates", []) or []
        return RescueRoundInfo(
            window_id=int(blob["window_id"]),
            round_idx=int(blob["round_idx"]),
            n_initial_peaks=int(blob["n_initial_peaks"]),
            n_rescue_added=int(blob["n_rescue_added"]),
            n_pruned_total=int(blob["n_pruned_total"]),
            n_pruned_rescue_origin=int(blob["n_pruned_rescue_origin"]),
            n_merged=int(blob.get("n_merged", 0)),
            chi2_before=float(blob["chi2_before"]),
            chi2_after=float(blob["chi2_after"]),
            tau_us_before=float(blob["tau_us_before"]),
            tau_us_after=float(blob["tau_us_after"]),
            accepted=bool(blob["accepted"]),
            reason=str(blob.get("reason", "")),
            candidates=[
                _json_to_rescue_candidate(c, f"{where}.candidates[{i}]")
                for i, c in enumerate(candidates_raw)
            ],
        )
    except KeyError as exc:
        raise ValueError(f"{where} missing required field {exc.args[0]!r}") from exc


# ---------------------------------------------------------------------------
# Small attribute helpers
# ---------------------------------------------------------------------------
def _nan_if_none(value: Optional[float]) -> float:
    """``float(value)`` or NaN sentinel when ``value is None``."""
    if value is None:
        return float("nan")
    return float(value)


def _none_if_nan(value: float) -> Optional[float]:
    """Inverse of :func:`_nan_if_none`: NaN -> None, finite value otherwise."""
    f = float(value)
    return None if np.isnan(f) else f


def _peak_id_to_int(peak_id: Any) -> int:
    """Coerce the (Union[str, int]) ``peak_id`` to int for storage."""
    try:
        return int(peak_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"FittedPeak.peak_id must be int-coercible for HDF5 storage; "
            f"got {peak_id!r}"
        ) from exc


def _load_json_attr(h5_group: h5py.Group, name: str, default: Any) -> Any:
    raw = h5_group.attrs.get(name)
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(
            f"stage5_fitting attribute {name!r} on "
            f"{h5_group.name} is not valid JSON"
        ) from exc


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_spectrum_fit_to_hdf5(fit: SpectrumFit, h5_group: h5py.Group) -> None:
    """Write a :class:`SpectrumFit` to an HDF5 group.

    Parameters
    ----------
    fit : SpectrumFit
        The persistent fit aggregate. The merged global ``fitted_peaks``
        list is rebuilt on load from the per-window peaks; it is *not*
        stored independently here.
    h5_group : h5py.Group
        Destination group; any existing fit content is overwritten.
    """
    for key in list(h5_group.keys()):
        del h5_group[key]

    h5_group.attrs["n_windows"] = int(fit.n_windows)
    h5_group.attrs["n_fitted_peaks"] = int(fit.n_fitted_peaks)
    h5_group.attrs["creation_time"] = datetime.now().isoformat()
    h5_group.attrs["stage_name"] = "stage5_fitting"
    h5_group.attrs["final_plan_revision"] = int(fit.final_plan_revision)
    h5_group.attrs["parameters"] = json.dumps(fit.parameters, default=str)
    h5_group.attrs["diagnostics"] = json.dumps(fit.diagnostics, default=str)
    h5_group.attrs["thaw_history"] = json.dumps(
        [_thaw_info_to_json(e) for e in fit.thaw_history]
    )
    h5_group.attrs["replan_history"] = json.dumps(
        [_replan_info_to_json(e) for e in fit.replan_history]
    )
    h5_group.attrs["rescue_history"] = json.dumps(
        [_rescue_round_to_json(e) for e in fit.rescue_history]
    )

    windows_group = h5_group.create_group("windows")
    for window_fit in fit.window_fits:
        if window_fit.window_id is None:
            raise ValueError(
                "FittingResult.window_id is required for serialization "
                "(the active-FT slice cannot be reconstructed without it)"
            )
        wid = int(window_fit.window_id)
        wg = windows_group.create_group(f"window_{wid:04d}")
        _save_window_fit(window_fit, wg)


def _save_window_fit(window_fit: FittingResult, wg: h5py.Group) -> None:
    """Write one :class:`FittingResult` to its window subgroup."""
    assert window_fit.window_id is not None  # guarded by caller
    tau_entry = window_fit.shared_parameters.get("tau_us") or {}
    tau_us = float(tau_entry.get("value", float("nan")))
    tau_error = _nan_if_none(tau_entry.get("error"))
    tau_fitted_val = tau_entry.get("fitted")

    wg.attrs["window_id"] = int(window_fit.window_id)
    wg.attrs["success"] = bool(window_fit.success)
    wg.attrs["cost"] = float(window_fit.cost)
    wg.attrs["iterations"] = int(window_fit.iterations)
    wg.attrs["aic"] = float(window_fit.aic)
    wg.attrs["reduced_chi2"] = float(window_fit.reduced_chi2)
    wg.attrs["tau_us"] = tau_us
    wg.attrs["tau_error"] = tau_error
    # tau_fitted: 1 if tau was a free LSQ parameter, 0 if held at tau0_us,
    # -1 if unknown (only emitted by older files predating this flag).
    if tau_fitted_val is None:
        wg.attrs["tau_fitted"] = np.int8(-1)
    else:
        wg.attrs["tau_fitted"] = np.int8(1 if bool(tau_fitted_val) else 0)
    # Persist the window's molecular freq_range so visualization can
    # locate the window on the persisted spectrum without needing the
    # Stage 4 plan back. The complex spectrum slice itself stays
    # recomputable from the FID + active-FT.
    if window_fit.window is not None:
        wg.attrs["freq_min"] = float(window_fit.window.freq_range[0])
        wg.attrs["freq_max"] = float(window_fit.window.freq_range[1])
    else:
        wg.attrs["freq_min"] = float("nan")
        wg.attrs["freq_max"] = float("nan")
    wg.attrs["edge_coherence_low"] = float(
        window_fit.quality_metrics.get("edge_coherence_low", float("nan"))
    )
    wg.attrs["edge_coherence_high"] = float(
        window_fit.quality_metrics.get("edge_coherence_high", float("nan"))
    )
    wg.attrs["fixed_parameters"] = json.dumps(window_fit.fixed_parameters, default=str)
    wg.attrs["quality_metrics"] = json.dumps(window_fit.quality_metrics, default=str)
    wg.attrs["audit_trail"] = json.dumps(
        [_audit_step_to_json(s) for s in window_fit.audit_trail]
    )
    wg.attrs["thaw_events"] = json.dumps(
        [_thaw_info_to_json(e) for e in window_fit.thaw_events]
    )
    wg.attrs["rescue_events"] = json.dumps(
        [_rescue_round_to_json(e) for e in window_fit.rescue_events]
    )

    peaks_group = wg.create_group("peaks")
    _save_peak_columns(window_fit.fitted_peaks, peaks_group)


def _save_peak_columns(peaks: List[FittedPeak], peaks_group: h5py.Group) -> None:
    """Write a fitted-peak list as parallel arrays under ``peaks_group``."""
    n = len(peaks)
    columns: Dict[str, np.ndarray] = {
        "peak_id": np.empty(n, dtype="i8"),
        "frequency_mhz": np.empty(n, dtype="f8"),
        "amplitude": np.empty(n, dtype="f8"),
        "phase": np.empty(n, dtype="f8"),
        "decay_rate": np.empty(n, dtype="f8"),
        "frequency_error": np.empty(n, dtype="f8"),
        "amplitude_error": np.empty(n, dtype="f8"),
        "phase_error": np.empty(n, dtype="f8"),
        "decay_rate_error": np.empty(n, dtype="f8"),
        "snr": np.empty(n, dtype="f8"),
        "chi_squared": np.empty(n, dtype="f8"),
        "window_id": np.empty(n, dtype="i8"),
        "knockout_delta_chi2": np.empty(n, dtype="f8"),
        "knockout_expected_delta_chi2": np.empty(n, dtype="f8"),
        "knockout_supported": np.empty(n, dtype="i1"),
        "knockout_p_value": np.empty(n, dtype="f8"),
        "knockout_n_eff": np.empty(n, dtype="f8"),
        "knockout_aicc_delta": np.empty(n, dtype="f8"),
    }
    for i, p in enumerate(peaks):
        columns["peak_id"][i] = _peak_id_to_int(p.peak_id)
        columns["frequency_mhz"][i] = float(p.frequency_mhz)
        columns["amplitude"][i] = float(p.amplitude)
        columns["phase"][i] = _nan_if_none(p.phase)
        columns["decay_rate"][i] = _nan_if_none(p.decay_rate)
        columns["frequency_error"][i] = _nan_if_none(p.frequency_error)
        columns["amplitude_error"][i] = _nan_if_none(p.amplitude_error)
        columns["phase_error"][i] = _nan_if_none(p.phase_error)
        columns["decay_rate_error"][i] = _nan_if_none(p.decay_rate_error)
        columns["snr"][i] = _nan_if_none(p.snr)
        columns["chi_squared"][i] = _nan_if_none(p.chi_squared)
        columns["window_id"][i] = -1 if p.window_id is None else int(p.window_id)
        if p.knockout is None:
            columns["knockout_delta_chi2"][i] = float("nan")
            columns["knockout_expected_delta_chi2"][i] = float("nan")
            columns["knockout_supported"][i] = -1
            columns["knockout_p_value"][i] = float("nan")
            columns["knockout_n_eff"][i] = float("nan")
            columns["knockout_aicc_delta"][i] = float("nan")
        else:
            columns["knockout_delta_chi2"][i] = float(p.knockout.delta_chi2)
            columns["knockout_expected_delta_chi2"][i] = float(
                p.knockout.expected_delta_chi2
            )
            columns["knockout_supported"][i] = 1 if p.knockout.supported else 0
            columns["knockout_p_value"][i] = float(p.knockout.p_value)
            columns["knockout_n_eff"][i] = float(p.knockout.n_eff)
            columns["knockout_aicc_delta"][i] = float(p.knockout.aicc_delta)
    for name, data in columns.items():
        peaks_group.create_dataset(name, data=data)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_spectrum_fit_from_hdf5(h5_group: h5py.Group) -> SpectrumFit:
    """Load a :class:`SpectrumFit` from an HDF5 group, validating loudly.

    The merged global :attr:`SpectrumFit.fitted_peaks` list is rebuilt from
    the per-window peaks, sorted ascending by molecular frequency.

    Raises
    ------
    ValueError
        If the ``windows`` subgroup is missing, a window subgroup lacks a
        required attribute, the peak columns are missing or mismatched, or
        a JSON-encoded audit/thaw/replan entry has an invalid
        ``decision``/``edge_side`` label.
    """
    if "windows" not in h5_group:
        raise ValueError("stage5_fitting group missing required 'windows' subgroup")

    final_revision_attr = h5_group.attrs.get("final_plan_revision", 0)
    final_plan_revision = int(final_revision_attr)
    parameters = _load_json_attr(h5_group, "parameters", {})
    diagnostics = _load_json_attr(h5_group, "diagnostics", {})

    raw_thaw = _load_json_attr(h5_group, "thaw_history", [])
    thaw_history = [
        _json_to_thaw_info(blob, f"thaw_history[{i}]")
        for i, blob in enumerate(raw_thaw)
    ]
    raw_replan = _load_json_attr(h5_group, "replan_history", [])
    replan_history = [
        _json_to_replan_info(blob, f"replan_history[{i}]")
        for i, blob in enumerate(raw_replan)
    ]
    raw_rescue = _load_json_attr(h5_group, "rescue_history", [])
    rescue_history = [
        _json_to_rescue_round(blob, f"rescue_history[{i}]")
        for i, blob in enumerate(raw_rescue)
    ]

    windows_group = h5_group["windows"]
    window_fits: List[FittingResult] = []
    for name in sorted(windows_group.keys()):
        wg = windows_group[name]
        window_fits.append(_load_window_fit(wg, name))

    window_fits.sort(key=lambda wf: (wf.window_id if wf.window_id is not None else -1))

    # Rebuild the merged global peak list from per-window peaks.
    fitted_peaks: List[FittedPeak] = []
    for wf in window_fits:
        fitted_peaks.extend(wf.fitted_peaks)
    fitted_peaks.sort(key=lambda p: p.frequency_mhz)

    return SpectrumFit(
        window_fits=window_fits,
        fitted_peaks=fitted_peaks,
        thaw_history=thaw_history,
        replan_history=replan_history,
        rescue_history=rescue_history,
        final_plan_revision=final_plan_revision,
        parameters=parameters,
        diagnostics=diagnostics,
    )


def _load_window_fit(wg: h5py.Group, where: str) -> FittingResult:
    """Load one :class:`FittingResult` from a window subgroup."""
    required = (
        "window_id",
        "success",
        "cost",
        "iterations",
        "aic",
        "reduced_chi2",
        "tau_us",
    )
    for attr in required:
        if attr not in wg.attrs:
            raise ValueError(f"window {where!r} missing required attribute {attr!r}")
    if "peaks" not in wg:
        raise ValueError(f"window {where!r} missing required 'peaks' subgroup")

    # Reconstruct a lightweight SpectralWindow from the persisted freq_range
    # (the complex spectrum slice stays recomputable from the FID + active-FT;
    # what visualizations need from `window` is its freq_range).
    freq_min = float(wg.attrs.get("freq_min", float("nan")))
    freq_max = float(wg.attrs.get("freq_max", float("nan")))
    window_obj: Optional[SpectralWindow]
    if np.isnan(freq_min) or np.isnan(freq_max):
        window_obj = None
    else:
        window_obj = SpectralWindow(
            parent_ft=None,
            freq_array=np.array([], dtype=float),
            complex_spectrum=np.array([], dtype=np.complex128),
            freq_range=(freq_min, freq_max),
            window_id=int(wg.attrs["window_id"]),
        )

    result = FittingResult(
        success=bool(wg.attrs["success"]),
        fitted_spectrum=None,  # recomputed on demand
        cost=float(wg.attrs["cost"]),
        iterations=int(wg.attrs["iterations"]),
        aic=float(wg.attrs["aic"]),
        reduced_chi2=float(wg.attrs["reduced_chi2"]),
        window=window_obj,
        window_id=int(wg.attrs["window_id"]),
    )

    tau_us = float(wg.attrs["tau_us"])
    tau_error = _none_if_nan(float(wg.attrs.get("tau_error", float("nan"))))
    # tau_fitted: 1 -> True, 0 -> False, -1 or absent -> backward-compat
    # best-effort (finite tau_error implies tau was fit; otherwise unknown).
    tau_fitted: Optional[bool]
    if "tau_fitted" in wg.attrs:
        raw = int(wg.attrs["tau_fitted"])
        if raw == 1:
            tau_fitted = True
        elif raw == 0:
            tau_fitted = False
        else:
            tau_fitted = True if tau_error is not None else None
    else:
        tau_fitted = True if tau_error is not None else None
    fitted_peaks = _load_peak_columns(wg["peaks"], where=f"{where}/peaks")
    result.fitted_peaks = fitted_peaks
    result.shared_parameters["tau_us"] = {
        "value": tau_us,
        "error": tau_error,
        "fitted": tau_fitted,
        "peak_ids": [p.peak_id for p in fitted_peaks],
    }
    result.fixed_parameters = _load_json_attr(wg, "fixed_parameters", {})
    result.quality_metrics = _load_json_attr(wg, "quality_metrics", {})
    # Ensure edge-coherence scalar attrs make it back into quality_metrics
    # even if a hand-edit nuked the JSON attribute -- the scalar attrs are
    # canonical.
    if "edge_coherence_low" in wg.attrs:
        result.quality_metrics["edge_coherence_low"] = float(
            wg.attrs["edge_coherence_low"]
        )
    if "edge_coherence_high" in wg.attrs:
        result.quality_metrics["edge_coherence_high"] = float(
            wg.attrs["edge_coherence_high"]
        )

    raw_audit = _load_json_attr(wg, "audit_trail", [])
    result.audit_trail = [
        _json_to_audit_step(blob, f"{where}/audit_trail[{i}]")
        for i, blob in enumerate(raw_audit)
    ]
    raw_thaw = _load_json_attr(wg, "thaw_events", [])
    result.thaw_events = [
        _json_to_thaw_info(blob, f"{where}/thaw_events[{i}]")
        for i, blob in enumerate(raw_thaw)
    ]
    raw_rescue = _load_json_attr(wg, "rescue_events", [])
    result.rescue_events = [
        _json_to_rescue_round(blob, f"{where}/rescue_events[{i}]")
        for i, blob in enumerate(raw_rescue)
    ]
    return result


def _load_peak_columns(peaks_group: h5py.Group, *, where: str) -> List[FittedPeak]:
    """Load fitted peaks from parallel-array columns under ``peaks_group``."""
    missing = [c for c in _PEAK_COLUMNS if c not in peaks_group]
    if missing:
        raise ValueError(f"{where} missing required peak column(s): {missing}")
    cols = {c: peaks_group[c][:] for c in _PEAK_COLUMNS}
    # Optional columns: silently default to NaN when absent (older files).
    n_rows = len(cols["peak_id"])
    for c in _OPTIONAL_PEAK_COLUMNS:
        if c in peaks_group:
            cols[c] = peaks_group[c][:]
        else:
            cols[c] = np.full(n_rows, float("nan"), dtype="f8")
    lengths = {c: len(v) for c, v in cols.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{where} peak columns have mismatched lengths: {lengths}")
    n = next(iter(lengths.values()))
    peaks: List[FittedPeak] = []
    for i in range(n):
        ko_supported_raw = int(cols["knockout_supported"][i])
        ko_delta = float(cols["knockout_delta_chi2"][i])
        if ko_supported_raw < 0 or np.isnan(ko_delta):
            knockout: Optional[KnockoutInfo] = None
        else:
            knockout = KnockoutInfo(
                delta_chi2=ko_delta,
                expected_delta_chi2=float(cols["knockout_expected_delta_chi2"][i]),
                supported=bool(ko_supported_raw),
                p_value=float(cols["knockout_p_value"][i]),
                n_eff=float(cols["knockout_n_eff"][i]),
                aicc_delta=float(cols["knockout_aicc_delta"][i]),
            )
        wid_raw = int(cols["window_id"][i])
        peaks.append(
            FittedPeak(
                peak_id=int(cols["peak_id"][i]),
                frequency_mhz=float(cols["frequency_mhz"][i]),
                amplitude=float(cols["amplitude"][i]),
                phase=_none_if_nan(float(cols["phase"][i])),
                decay_rate=_none_if_nan(float(cols["decay_rate"][i])),
                frequency_error=_none_if_nan(float(cols["frequency_error"][i])),
                amplitude_error=_none_if_nan(float(cols["amplitude_error"][i])),
                phase_error=_none_if_nan(float(cols["phase_error"][i])),
                decay_rate_error=_none_if_nan(float(cols["decay_rate_error"][i])),
                snr=_none_if_nan(float(cols["snr"][i])),
                chi_squared=_none_if_nan(float(cols["chi_squared"][i])),
                window_id=None if wid_raw < 0 else wid_raw,
                knockout=knockout,
            )
        )
    return peaks
