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
  plus the persisted Stage 1 settings plus the fit window's freq_range),
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
            covariance                       [f8, shape (D,D)]
                                                          (omitted when JᵀJ singular)
            .attrs:
                covariance_param_labels (JSON)  -- ordered label list, one per row/col;
                                                   present iff covariance dataset is
            peaks/
                detection_index                  [i8]    (Stage 3 promoted-peak
                                                          index that seeded the
                                                          line; provenance, not
                                                          identity -- several
                                                          peaks in a blend share
                                                          one value. Older files
                                                          store this under the
                                                          column name
                                                          ``peak_id``, still
                                                          read.)
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
                origin                           [str]   ("auto" or "user";
                                                          absent in older files
                                                          -> default "auto")
                derivation                       [i8]    (Stage-6 decision id
                                                          that created/altered
                                                          the peak; -1 or an
                                                          absent column -> None,
                                                          "carried through
                                                          unchanged")
                peak_uid                         [i8]    (point-space identity,
                                                          stamped at birth;
                                                          -1 or an absent
                                                          column -> None)
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
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import h5py
import numpy as np

from ..core.data_structures import (
    AuditStep,
    DoubletAlternativeInfo,
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
from ._hdf5_helpers import (
    REQUIRED,
    ColumnSpec,
    build_columns,
    load_json_attr,
    nan_if_none,
    none_if_nan,
    read_attr_value,
    read_dataset_column,
    record_row,
    reset_group,
    resolve_column_selection,
    stack_columns,
    stamp_stage_header,
)

__all__ = [
    "save_spectrum_fit_to_hdf5",
    "update_spectrum_fit_windows_in_hdf5",
    "load_spectrum_fit_from_hdf5",
    "FIT_PEAK_COLUMN_SPECS",
    "FIT_WINDOW_COLUMN_SPECS",
    "FIT_AUDIT_COLUMN_SPECS",
    "FIT_DOUBLET_COLUMN_SPECS",
    "FIT_THAW_COLUMN_SPECS",
    "FIT_REPLAN_COLUMN_SPECS",
    "FIT_RESCUE_COLUMN_SPECS",
    "read_fit_peak_columns",
    "read_fit_window_columns",
    "read_fit_audit_columns",
    "read_fit_doublet_columns",
    "read_fit_thaw_columns",
    "read_fit_replan_columns",
    "read_fit_rescue_columns",
    "read_fit_scalars",
    "read_fit_parameters",
    "read_fit_peak_frequencies_by_window",
    "read_fit_peak_uids_by_window",
    "read_fit_peak_freqs_and_uids_by_window",
    "FitWindowCoverage",
    "read_fit_window_coverage",
]


# --- peak column layout ----------------------------------------------------
# ``detection_index`` is written under that name; a file written before this
# rename stores the identical column under ``peak_id``, and the loader
# accepts either (see ``_load_peak_columns``).
_PEAK_COLUMNS = (
    "detection_index",
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
# load tolerates missing entries by substituting NaN (numeric) or an empty
# byte-string sentinel (string columns).
_OPTIONAL_PEAK_COLUMNS = (
    "knockout_p_value",
    "knockout_n_eff",
    "knockout_aicc_delta",
)
# Optional string columns: absent in older files; load substitutes b"" (-> None)
# for clock_lattice (None when absent) and "auto" for origin (default provenance).
_OPTIONAL_PEAK_STR_COLUMNS = ("clock_lattice", "origin")

#: On-disk name of ``detection_index`` before the rename. A file written by
#: an older version stores the identical data under this name; every reader
#: below tries ``detection_index`` first and falls back to this.
_LEGACY_DETECTION_INDEX_COLUMN = "peak_id"

_VALID_AUDIT_DECISIONS = {
    "seed",
    "seed-blend",
    "accept",
    "promote",
    "tentative",
    "reject",
    "knockout-null",
    "spur-drop",
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


def _json_to_rescue_candidate(blob: Dict[str, Any], where: str) -> RescueCandidateInfo:
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


def _doublet_alternative_to_json(d: DoubletAlternativeInfo) -> Dict[str, Any]:
    return {
        "frequency_a_mhz": float(d.frequency_a_mhz),
        "frequency_b_mhz": float(d.frequency_b_mhz),
        "amplitude_a": float(d.amplitude_a),
        "amplitude_b": float(d.amplitude_b),
        "separation_res_elements": float(d.separation_res_elements),
        "amp_ratio": float(d.amp_ratio),
        "chi2r_production": float(d.chi2r_production),
        "chi2r_merged": float(d.chi2r_merged),
        "delta_chi2_raw": float(d.delta_chi2_raw),
        "delta_aicc": float(d.delta_aicc),
        "merged_frequency_mhz": float(d.merged_frequency_mhz),
        "merged_amplitude": float(d.merged_amplitude),
        "merged_phase": float(d.merged_phase),
        "merged_tau_us": float(d.merged_tau_us),
        "merged_success": bool(d.merged_success),
        "orth_evidence_delta_chi2": float(d.orth_evidence_delta_chi2),
        "orth_evidence_n_params": int(d.orth_evidence_n_params),
        "support_bins": int(d.support_bins),
    }


def _json_to_doublet_alternative(
    blob: Dict[str, Any], where: str
) -> DoubletAlternativeInfo:
    try:
        return DoubletAlternativeInfo(
            frequency_a_mhz=float(blob["frequency_a_mhz"]),
            frequency_b_mhz=float(blob["frequency_b_mhz"]),
            amplitude_a=float(blob["amplitude_a"]),
            amplitude_b=float(blob["amplitude_b"]),
            separation_res_elements=float(blob["separation_res_elements"]),
            amp_ratio=float(blob["amp_ratio"]),
            chi2r_production=float(blob["chi2r_production"]),
            chi2r_merged=float(blob.get("chi2r_merged", float("nan"))),
            delta_chi2_raw=float(blob.get("delta_chi2_raw", float("nan"))),
            delta_aicc=float(blob.get("delta_aicc", float("nan"))),
            merged_frequency_mhz=float(blob.get("merged_frequency_mhz", float("nan"))),
            merged_amplitude=float(blob.get("merged_amplitude", float("nan"))),
            merged_phase=float(blob.get("merged_phase", float("nan"))),
            merged_tau_us=float(blob.get("merged_tau_us", float("nan"))),
            merged_success=bool(blob.get("merged_success", False)),
            orth_evidence_delta_chi2=float(
                blob.get("orth_evidence_delta_chi2", float("nan"))
            ),
            orth_evidence_n_params=int(blob.get("orth_evidence_n_params", 3)),
            support_bins=int(blob.get("support_bins", 0)),
        )
    except KeyError as exc:
        raise ValueError(f"{where} missing required field {exc.args[0]!r}") from exc


# ---------------------------------------------------------------------------
# Small attribute helpers
# ---------------------------------------------------------------------------


def _detection_index_to_int(detection_index: Any) -> int:
    """Coerce the (Union[str, int]) ``detection_index`` to int for storage."""
    try:
        return int(detection_index)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"FittedPeak.detection_index must be int-coercible for HDF5 "
            f"storage; got {detection_index!r}"
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
    reset_group(h5_group)

    stamp_stage_header(
        h5_group,
        "stage5_fitting",
        n_windows=fit.n_windows,
        n_fitted_peaks=fit.n_fitted_peaks,
    )
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


def update_spectrum_fit_windows_in_hdf5(
    fit: SpectrumFit, h5_group: h5py.Group, window_ids: Iterable[int]
) -> None:
    """Rewrite only *window_ids*' subgroups of an already-persisted fit.

    The incremental counterpart to :func:`save_spectrum_fit_to_hdf5`, for the
    curation engine's single write point. A batch that edited one window used
    to delete ``/stage5_fitting`` and rewrite the whole thing, which costs the
    entire table (404 ms on a 251-window build) and -- because HDF5 does not
    reclaim a deleted group's space -- grew the file by ~919 kB on *every*
    curation write. Rewriting only what changed costs ~1.7 ms and ~6 kB.

    Everything group-level (the header counts, ``parameters``,
    ``diagnostics``, and the three history blobs) is rewritten
    unconditionally. They are small, and any of them can move without naming
    a window.

    The window-group **set** is reconciled against ``fit`` rather than taken
    from ``window_ids``: a group on disk whose window is no longer in the fit
    is deleted, and a window in the fit with no group on disk is written,
    named or not. Only the *contents* of an already-correct group are taken
    on the caller's word. So an under-reported id leaves a window holding its
    previous values -- never an orphaned group, and never a missing one.
    """
    stamp_stage_header(
        h5_group,
        "stage5_fitting",
        n_windows=fit.n_windows,
        n_fitted_peaks=fit.n_fitted_peaks,
    )
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

    if "windows" not in h5_group:
        h5_group.create_group("windows")
    windows_group = h5_group["windows"]

    expected: Dict[str, FittingResult] = {}
    for window_fit in fit.window_fits:
        if window_fit.window_id is None:
            raise ValueError(
                "FittingResult.window_id is required for serialization "
                "(the active-FT slice cannot be reconstructed without it)"
            )
        expected[f"window_{int(window_fit.window_id):04d}"] = window_fit

    for name in list(windows_group.keys()):
        if name not in expected:
            del windows_group[name]

    named = {f"window_{int(wid):04d}" for wid in window_ids}
    for name, window_fit in expected.items():
        if name not in named and name in windows_group:
            continue
        if name in windows_group:
            del windows_group[name]
        _save_window_fit(window_fit, windows_group.create_group(name))


def _save_window_fit(window_fit: FittingResult, wg: h5py.Group) -> None:
    """Write one :class:`FittingResult` to its window subgroup."""
    assert window_fit.window_id is not None  # guarded by caller
    tau_entry = window_fit.shared_parameters.get("tau_us") or {}
    tau_us = float(tau_entry.get("value", float("nan")))
    tau_error = nan_if_none(tau_entry.get("error"))
    tau_fitted_val = tau_entry.get("fitted")

    wg.attrs["window_id"] = int(window_fit.window_id)
    wg.attrs["success"] = bool(window_fit.success)
    wg.attrs["cost"] = float(window_fit.cost)
    wg.attrs["iterations"] = int(window_fit.iterations)
    wg.attrs["aic"] = float(window_fit.aic)
    wg.attrs["reduced_chi2"] = float(window_fit.reduced_chi2)
    wg.attrs["shape"] = str(getattr(window_fit, "shape", "lorentzian"))
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
    wg.attrs["doublet_alternatives"] = json.dumps(
        [
            _doublet_alternative_to_json(d)
            for d in getattr(window_fit, "doublet_alternatives", [])
        ]
    )

    # Per-window parameter covariance (omitted when singular / unavailable).
    cov = getattr(window_fit, "covariance", None)
    labels = getattr(window_fit, "covariance_param_labels", None)
    if cov is not None and labels is not None:
        wg.create_dataset("covariance", data=np.asarray(cov, dtype="f8"))
        wg.attrs["covariance_param_labels"] = json.dumps(list(labels))

    peaks_group = wg.create_group("peaks")
    _save_peak_columns(
        window_fit.fitted_peaks, peaks_group, window_id=int(window_fit.window_id)
    )


def _save_peak_columns(
    peaks: List[FittedPeak], peaks_group: h5py.Group, *, window_id: int
) -> None:
    """Write a fitted-peak list as parallel arrays under ``peaks_group``.

    ``window_id`` is the owning window group's id. A peak whose own
    ``window_id`` is ``None`` is stamped with it rather than with a ``-1``
    sentinel: the peak is stored *inside* that window, so the group already
    answers the grouping question, and a column that disagreed with the group
    would let a reader that trusts the column drop the row out of its window
    silently.
    """
    n = len(peaks)
    columns: Dict[str, np.ndarray] = {
        "detection_index": np.empty(n, dtype="i8"),
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
    # Variable-length UTF-8 string type for string columns.
    _vlen_str = h5py.string_dtype(encoding="utf-8")
    clock_lattice_col: np.ndarray = np.empty(n, dtype=object)
    origin_col: np.ndarray = np.empty(n, dtype=object)
    flat_decay_col: np.ndarray = np.empty(n, dtype="i1")
    derivation_col: np.ndarray = np.empty(n, dtype="i8")
    peak_uid_col: np.ndarray = np.empty(n, dtype="i8")
    for i, p in enumerate(peaks):
        columns["detection_index"][i] = _detection_index_to_int(p.detection_index)
        columns["frequency_mhz"][i] = float(p.frequency_mhz)
        columns["amplitude"][i] = float(p.amplitude)
        columns["phase"][i] = nan_if_none(p.phase)
        columns["decay_rate"][i] = nan_if_none(p.decay_rate)
        columns["frequency_error"][i] = nan_if_none(p.frequency_error)
        columns["amplitude_error"][i] = nan_if_none(p.amplitude_error)
        columns["phase_error"][i] = nan_if_none(p.phase_error)
        columns["decay_rate_error"][i] = nan_if_none(p.decay_rate_error)
        columns["snr"][i] = nan_if_none(p.snr)
        columns["chi_squared"][i] = nan_if_none(p.chi_squared)
        columns["window_id"][i] = window_id if p.window_id is None else int(p.window_id)
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
        # clock_lattice: empty string when absent (None), identity string when set.
        clock_lattice_col[i] = p.clock_lattice if p.clock_lattice is not None else ""
        # origin: always a non-empty string; default "auto" for every pipeline peak.
        origin_col[i] = p.origin
        # flat_decay: review hint, 0 for every peak unless the spur gate flagged it.
        flat_decay_col[i] = 1 if p.flat_decay else 0
        # derivation: Stage-6 decision id that created/altered the peak; -1
        # encodes None ("carried through the refit unchanged").
        derivation_col[i] = -1 if p.derivation is None else int(p.derivation)
        # peak_uid: point-space identity stamped at birth; -1 encodes None
        # (absent, or a fit produced before this field existed).
        peak_uid_col[i] = -1 if p.peak_uid is None else int(p.peak_uid)
    for name, data in columns.items():
        peaks_group.create_dataset(name, data=data)
    # String columns stored as variable-length UTF-8 datasets.
    peaks_group.create_dataset("clock_lattice", data=clock_lattice_col, dtype=_vlen_str)
    peaks_group.create_dataset("origin", data=origin_col, dtype=_vlen_str)
    peaks_group.create_dataset("flat_decay", data=flat_decay_col)
    peaks_group.create_dataset("derivation", data=derivation_col)
    peaks_group.create_dataset("peak_uid", data=peak_uid_col)


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
    parameters = load_json_attr(h5_group, "parameters", {}, label="stage5_fitting")
    diagnostics = load_json_attr(h5_group, "diagnostics", {}, label="stage5_fitting")

    raw_thaw = load_json_attr(h5_group, "thaw_history", [], label="stage5_fitting")
    thaw_history = [
        _json_to_thaw_info(blob, f"thaw_history[{i}]")
        for i, blob in enumerate(raw_thaw)
    ]
    raw_replan = load_json_attr(h5_group, "replan_history", [], label="stage5_fitting")
    replan_history = [
        _json_to_replan_info(blob, f"replan_history[{i}]")
        for i, blob in enumerate(raw_replan)
    ]
    raw_rescue = load_json_attr(h5_group, "rescue_history", [], label="stage5_fitting")
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

    # Files that pre-date the shape attribute were Lorentzian-only.
    shape_attr_raw = wg.attrs.get("shape", "lorentzian")
    if isinstance(shape_attr_raw, bytes):
        shape_attr_raw = shape_attr_raw.decode("utf-8")
    shape_str = str(shape_attr_raw)
    result = FittingResult(
        success=bool(wg.attrs["success"]),
        fitted_spectrum=None,  # recomputed on demand
        cost=float(wg.attrs["cost"]),
        iterations=int(wg.attrs["iterations"]),
        aic=float(wg.attrs["aic"]),
        reduced_chi2=float(wg.attrs["reduced_chi2"]),
        window=window_obj,
        window_id=int(wg.attrs["window_id"]),
        shape=shape_str,
    )

    tau_us = float(wg.attrs["tau_us"])
    tau_error = none_if_nan(float(wg.attrs.get("tau_error", float("nan"))))
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
    fitted_peaks = _load_peak_columns(
        wg["peaks"], where=f"{where}/peaks", window_id=int(wg.attrs["window_id"])
    )
    result.fitted_peaks = fitted_peaks
    result.shared_parameters["tau_us"] = {
        "value": tau_us,
        "error": tau_error,
        "fitted": tau_fitted,
        "detection_indices": [p.detection_index for p in fitted_peaks],
    }
    result.fixed_parameters = load_json_attr(
        wg, "fixed_parameters", {}, label="stage5_fitting"
    )
    result.quality_metrics = load_json_attr(
        wg, "quality_metrics", {}, label="stage5_fitting"
    )
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

    raw_audit = load_json_attr(wg, "audit_trail", [], label="stage5_fitting")
    result.audit_trail = [
        _json_to_audit_step(blob, f"{where}/audit_trail[{i}]")
        for i, blob in enumerate(raw_audit)
    ]
    raw_thaw = load_json_attr(wg, "thaw_events", [], label="stage5_fitting")
    result.thaw_events = [
        _json_to_thaw_info(blob, f"{where}/thaw_events[{i}]")
        for i, blob in enumerate(raw_thaw)
    ]
    raw_rescue = load_json_attr(wg, "rescue_events", [], label="stage5_fitting")
    result.rescue_events = [
        _json_to_rescue_round(blob, f"{where}/rescue_events[{i}]")
        for i, blob in enumerate(raw_rescue)
    ]
    # Tolerate missing attr (older files predating the doublet-alternative pass).
    raw_doublet = load_json_attr(wg, "doublet_alternatives", [], label="stage5_fitting")
    result.doublet_alternatives = [
        _json_to_doublet_alternative(blob, f"{where}/doublet_alternatives[{i}]")
        for i, blob in enumerate(raw_doublet)
    ]

    # Per-window parameter covariance: omitted in older files and when JᵀJ
    # was singular; both cases round-trip as None.
    if "covariance" in wg:
        cov_arr = np.asarray(wg["covariance"], dtype="f8")
        raw_labels = wg.attrs.get("covariance_param_labels")
        if raw_labels is None:
            raise ValueError(
                f"window {where!r} has 'covariance' dataset but is missing "
                "the 'covariance_param_labels' attribute"
            )
        try:
            labels_loaded: List[str] = json.loads(raw_labels)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(
                f"window {where!r} 'covariance_param_labels' is not valid JSON"
            ) from exc
        if cov_arr.ndim != 2 or cov_arr.shape[0] != cov_arr.shape[1]:
            raise ValueError(
                f"window {where!r} covariance matrix is not square: "
                f"shape {cov_arr.shape}"
            )
        if len(labels_loaded) != cov_arr.shape[0]:
            raise ValueError(
                f"window {where!r} covariance label count ({len(labels_loaded)}) "
                f"does not match matrix dimension ({cov_arr.shape[0]})"
            )
        n_amp_labels = sum(1 for lbl in labels_loaded if lbl.startswith("amplitude_"))
        n_fitted_peaks = len(result.fitted_peaks)
        if n_amp_labels != n_fitted_peaks:
            raise ValueError(
                f"window {where!r} covariance has {n_amp_labels} amplitude "
                f"label(s) but {n_fitted_peaks} fitted peak(s)"
            )
        result.covariance = cov_arr
        result.covariance_param_labels = labels_loaded

    return result


def _load_peak_columns(
    peaks_group: h5py.Group, *, where: str, window_id: Optional[int] = None
) -> List[FittedPeak]:
    """Load fitted peaks from parallel-array columns under ``peaks_group``.

    ``window_id`` is the owning window group's id, used to backfill a stored
    ``-1`` (written by versions before the writer stamped the group's id). The
    peak is inside that window whatever its own column says, so the group wins.

    ``detection_index`` is read under its current name, falling back to the
    pre-rename ``peak_id`` column when that is what the file has -- both are
    required (the row-count-defining anchor column), so exactly one of the two
    names must be present.
    """
    required = [c for c in _PEAK_COLUMNS if c != "detection_index"]
    missing = [c for c in required if c not in peaks_group]
    if missing:
        raise ValueError(f"{where} missing required peak column(s): {missing}")
    cols = {c: peaks_group[c][:] for c in required}
    if "detection_index" in peaks_group:
        cols["detection_index"] = peaks_group["detection_index"][:]
    elif _LEGACY_DETECTION_INDEX_COLUMN in peaks_group:
        cols["detection_index"] = peaks_group[_LEGACY_DETECTION_INDEX_COLUMN][:]
    else:
        raise ValueError(
            f"{where} missing required peak column(s): ['detection_index']"
        )
    # Optional numeric columns: silently default to NaN when absent (older files).
    n_rows = len(cols["detection_index"])
    for c in _OPTIONAL_PEAK_COLUMNS:
        if c in peaks_group:
            cols[c] = peaks_group[c][:]
        else:
            cols[c] = np.full(n_rows, float("nan"), dtype="f8")
    lengths = {c: len(v) for c, v in cols.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{where} peak columns have mismatched lengths: {lengths}")
    n = next(iter(lengths.values()))
    # Optional string column: absent in files written before clock-lattice annotation.
    # An empty string encodes a None (unannotated peak).
    if "clock_lattice" in peaks_group:
        raw_cl = peaks_group["clock_lattice"][:]
        clock_lattice_vals = [
            (v.decode("utf-8") if isinstance(v, bytes) else str(v)) or None
            for v in raw_cl
        ]
    else:
        clock_lattice_vals = [None] * n
    # Optional string column: absent in files written before Stage-6 provenance was
    # added.  An absent column or an empty string both default to "auto" so that
    # every pipeline-produced peak carries the correct provenance on load.
    if "origin" in peaks_group:
        raw_orig = peaks_group["origin"][:]
        origin_vals = [
            (v.decode("utf-8") if isinstance(v, bytes) else str(v)) or "auto"
            for v in raw_orig
        ]
    else:
        origin_vals = ["auto"] * n
    # Optional numeric column: absent in files written before the spur-review
    # flag; default False (no peak flagged) for back-compat.
    if "flat_decay" in peaks_group:
        flat_decay_vals = [bool(int(v)) for v in peaks_group["flat_decay"][:]]
    else:
        flat_decay_vals = [False] * n
    # Optional numeric column: absent in files written before the Stage-6
    # derivation tag. -1 (and an absent column) decode to None -- "carried
    # through unchanged", which is the correct reading for every peak in a file
    # that predates curation.
    if "derivation" in peaks_group:
        derivation_vals: List[Optional[int]] = [
            (None if int(v) < 0 else int(v)) for v in peaks_group["derivation"][:]
        ]
    else:
        derivation_vals = [None] * n
    # Optional numeric column: absent in files written before peak identity.
    # -1 (and an absent column) decode to None -- no identifier is the honest
    # value for a fit produced before this field existed.
    if "peak_uid" in peaks_group:
        peak_uid_vals: List[Optional[int]] = [
            (None if int(v) < 0 else int(v)) for v in peaks_group["peak_uid"][:]
        ]
    else:
        peak_uid_vals = [None] * n
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
        if wid_raw < 0 and window_id is not None:
            wid_raw = window_id
        peaks.append(
            FittedPeak(
                detection_index=int(cols["detection_index"][i]),
                frequency_mhz=float(cols["frequency_mhz"][i]),
                amplitude=float(cols["amplitude"][i]),
                phase=none_if_nan(float(cols["phase"][i])),
                decay_rate=none_if_nan(float(cols["decay_rate"][i])),
                frequency_error=none_if_nan(float(cols["frequency_error"][i])),
                amplitude_error=none_if_nan(float(cols["amplitude_error"][i])),
                phase_error=none_if_nan(float(cols["phase_error"][i])),
                decay_rate_error=none_if_nan(float(cols["decay_rate_error"][i])),
                snr=none_if_nan(float(cols["snr"][i])),
                chi_squared=none_if_nan(float(cols["chi_squared"][i])),
                window_id=None if wid_raw < 0 else wid_raw,
                knockout=knockout,
                clock_lattice=clock_lattice_vals[i],
                origin=origin_vals[i],
                flat_decay=flat_decay_vals[i],
                derivation=derivation_vals[i],
                peak_uid=peak_uid_vals[i],
            )
        )
    return peaks


# ---------------------------------------------------------------------------
# Read-only bulk column access (no SpectrumFit reconstruction)
# ---------------------------------------------------------------------------
#
# :func:`load_spectrum_fit_from_hdf5` rebuilds the whole persisted record --
# audit trails, thaw events, rescue rounds, doublet alternatives, covariance --
# because a curator editing the fit needs all of it. A consumer that wants a
# few columns per fitted peak does not, and the reconstruction is where the
# time goes (per-item h5py overhead paid thousands of times, plus a JSON parse
# per window). The readers below touch only the columns asked for.
#
# They are deliberately *raw*: the on-disk sentinels are preserved rather than
# translated to ``None`` the way the full loader does, because a bulk column is
# an array, not a list of objects. The sentinels are documented per column
# below and are exactly the ones the save side writes.

#: Per-fitted-peak read columns, in canonical order.
#:
#: Sentinels (as written by :func:`_save_peak_columns`): NaN encodes an absent
#: float (``phase``/``decay_rate``/the ``*_error`` columns/``snr``/
#: ``chi_squared``); ``derivation`` and ``peak_uid`` use ``-1`` for "none";
#: ``knockout_supported`` is tri-state (``1`` supported, ``0`` not, ``-1`` no
#: knockout was run); an empty ``clock_lattice`` means "off-lattice or no clock
#: declaration". ``shape`` is derived from the owning window's ``shape``
#: attribute -- the line shape is per window, and this column broadcasts it so a
#: consumer needs no join.
#:
#: ``detection_index`` is read under its current name, falling back to the
#: pre-rename ``peak_id`` column on a file written before this rename (see
#: :func:`read_fit_peak_columns`).
#:
#: ``window_id`` has **no** absent case: a peak is stored inside a window group,
#: so the group's id is always available and is backfilled over the ``-1`` that
#: older files wrote for a peak whose own ``window_id`` was ``None``. It is
#: always a real window id, and always the one the full loader groups the peak
#: under.
FIT_PEAK_COLUMN_SPECS: Dict[str, ColumnSpec] = {
    "detection_index": ("i8", REQUIRED),
    "window_id": ("i8", REQUIRED),
    "shape": ("str", "lorentzian"),  # derived from the window attr
    "frequency_mhz": ("f8", REQUIRED),
    "frequency_error": ("f8", REQUIRED),
    "amplitude": ("f8", REQUIRED),
    "amplitude_error": ("f8", REQUIRED),
    "phase": ("f8", REQUIRED),
    "phase_error": ("f8", REQUIRED),
    "decay_rate": ("f8", REQUIRED),
    "decay_rate_error": ("f8", REQUIRED),
    "snr": ("f8", REQUIRED),
    "chi_squared": ("f8", REQUIRED),
    "origin": ("str", "auto"),
    "clock_lattice": ("str", ""),
    "flat_decay": ("bool", False),
    "derivation": ("i8", -1),
    "peak_uid": ("i8", -1),
    "knockout_delta_chi2": ("f8", REQUIRED),
    "knockout_expected_delta_chi2": ("f8", REQUIRED),
    "knockout_supported": ("i1", REQUIRED),
    "knockout_p_value": ("f8", float("nan")),
    "knockout_n_eff": ("f8", float("nan")),
    "knockout_aicc_delta": ("f8", float("nan")),
}

#: ``shape`` is not a stored peak column; it is broadcast from the window.
_FIT_PEAK_DERIVED = ("shape",)

#: Per-fitted-window read columns, in canonical order. All are window-group
#: attributes except ``n_peaks``, which is the peak-column length.
#:
#: Sentinels: ``freq_min``/``freq_max`` are NaN when the window's freq_range was
#: not persisted; ``tau_error`` is NaN when tau was held fixed; ``tau_fitted``
#: is tri-state (``1`` free, ``0`` held, ``-1`` unknown -- an older file).
FIT_WINDOW_COLUMN_SPECS: Dict[str, ColumnSpec] = {
    "window_id": ("i8", REQUIRED),
    "freq_min": ("f8", float("nan")),
    "freq_max": ("f8", float("nan")),
    "shape": ("str", "lorentzian"),
    "n_peaks": ("i8", REQUIRED),  # derived from the peaks subgroup
    "success": ("bool", REQUIRED),
    "tau_us": ("f8", REQUIRED),
    "tau_error": ("f8", float("nan")),
    "tau_fitted": ("i1", -1),
    "cost": ("f8", REQUIRED),
    "iterations": ("i8", REQUIRED),
    "aic": ("f8", REQUIRED),
    "reduced_chi2": ("f8", REQUIRED),
    "edge_coherence_low": ("f8", float("nan")),
    "edge_coherence_high": ("f8", float("nan")),
}

_FIT_WINDOW_DERIVED = ("n_peaks",)


def _windows_group(h5_group: h5py.Group) -> h5py.Group:
    if "windows" not in h5_group:
        raise ValueError("stage5_fitting group missing required 'windows' subgroup")
    return h5_group["windows"]


def _peaks_subgroup(wg: h5py.Group, where: str) -> h5py.Group:
    try:
        return wg["peaks"]
    except KeyError:
        raise ValueError(f"{where} missing required 'peaks' subgroup") from None


def _peak_row_count(peaks_group: h5py.Group, where: str) -> int:
    """Row count of a peaks subgroup, from the required ``detection_index``
    column (or its pre-rename name, ``peak_id``, on an older file)."""
    try:
        dataset = peaks_group["detection_index"]
    except KeyError:
        try:
            dataset = peaks_group[_LEGACY_DETECTION_INDEX_COLUMN]
        except KeyError:
            raise ValueError(
                f"{where} missing required column 'detection_index'"
            ) from None
    return int(dataset.shape[0])


def read_fit_peak_columns(
    h5_group: h5py.Group,
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Read fitted-peak columns from a ``stage5_fitting`` group in bulk.

    Every window's ``peaks`` subgroup contributes its rows; each requested
    column is one whole-dataset read per window and nothing else in the window
    group (audit trail, thaw/rescue events, doublet alternatives, covariance)
    is touched.

    Rows are ordered by ascending molecular frequency, matching
    :attr:`SpectrumFit.fitted_peaks`, so a consumer can substitute this for the
    full loader row-for-row.

    Parameters
    ----------
    h5_group :
        The ``/stage5_fitting`` group.
    columns :
        Column names to read (see :data:`FIT_PEAK_COLUMN_SPECS`); ``None``
        reads all of them. Unknown names raise ``ValueError``.

    Returns
    -------
    dict
        ``{column_name: numpy array}``, all of equal length, in the requested
        order. See :data:`FIT_PEAK_COLUMN_SPECS` for the sentinel conventions.
    """
    requested = resolve_column_selection(
        columns, list(FIT_PEAK_COLUMN_SPECS), table="fit_peaks"
    )
    windows_group = _windows_group(h5_group)

    # frequency_mhz always read: it defines the row order.
    stored = [c for c in requested if c not in _FIT_PEAK_DERIVED]
    to_read = list(dict.fromkeys(["frequency_mhz", *stored]))
    chunks: Dict[str, List[np.ndarray]] = {
        c: [] for c in (*to_read, *_FIT_PEAK_DERIVED)
    }

    want_shape = "shape" in requested
    for name in sorted(windows_group.keys()):
        wg = windows_group[name]
        where = f"window {name!r} peaks"
        peaks_group = _peaks_subgroup(wg, f"window {name!r}")
        # The anchor column defines the window's row count; the rest must agree.
        n: Optional[int] = None
        for col in to_read:
            # detection_index: read under its current name, falling back to
            # the pre-rename `peak_id` dataset on an older file.
            dataset_name = col
            if col == "detection_index" and col not in peaks_group:
                dataset_name = _LEGACY_DETECTION_INDEX_COLUMN
            column = read_dataset_column(
                peaks_group,
                col,
                FIT_PEAK_COLUMN_SPECS[col],
                n,
                where=where,
                dataset=dataset_name,
            )
            if n is None:
                n = len(column)
            if col == "window_id":
                # The owning group is the authority on grouping, exactly as it
                # is for `shape`. Files written before the writer stamped the
                # group's id carry -1 for a peak whose own window_id was None;
                # backfilling here keeps the tap and the full loader agreeing
                # by construction, rather than silently ungrouping the row.
                wid_attr = read_attr_value(
                    wg,
                    "window_id",
                    FIT_WINDOW_COLUMN_SPECS["window_id"],
                    where=f"window {name!r}",
                )
                column = np.where(column < 0, np.int64(wid_attr), column)
            chunks[col].append(column)
        if want_shape:
            assert n is not None  # to_read always carries the anchor column
            shape_str = read_attr_value(
                wg, "shape", FIT_PEAK_COLUMN_SPECS["shape"], where=f"window {name!r}"
            )
            chunks["shape"].append(np.full(n, shape_str, dtype=object))

    read = stack_columns(
        chunks,
        FIT_PEAK_COLUMN_SPECS,
        [*to_read, *(c for c in _FIT_PEAK_DERIVED if c in requested)],
    )
    order = np.argsort(read["frequency_mhz"], kind="stable")
    return {c: read[c][order] for c in requested}


def read_fit_window_columns(
    h5_group: h5py.Group,
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Read per-window fit scalars from a ``stage5_fitting`` group in bulk.

    One attribute read per window per requested column; the JSON-encoded
    audit/thaw/rescue/doublet blobs on the window group are never parsed. Rows
    are ordered by ascending ``window_id``, matching
    :attr:`SpectrumFit.window_fits`.

    See :data:`FIT_WINDOW_COLUMN_SPECS` for the available columns and their
    sentinel conventions.
    """
    requested = resolve_column_selection(
        columns, list(FIT_WINDOW_COLUMN_SPECS), table="fit_windows"
    )
    windows_group = _windows_group(h5_group)

    # window_id always read: it defines the row order.
    attr_cols = [c for c in requested if c not in _FIT_WINDOW_DERIVED]
    to_read = list(dict.fromkeys(["window_id", *attr_cols]))
    rows: Dict[str, List[Any]] = {c: [] for c in (*to_read, *_FIT_WINDOW_DERIVED)}

    want_n_peaks = "n_peaks" in requested
    for name in sorted(windows_group.keys()):
        wg = windows_group[name]
        where = f"window {name!r}"
        for col in to_read:
            rows[col].append(
                read_attr_value(wg, col, FIT_WINDOW_COLUMN_SPECS[col], where=where)
            )
        if want_n_peaks:
            peaks_group = _peaks_subgroup(wg, where)
            rows["n_peaks"].append(_peak_row_count(peaks_group, f"{where} peaks"))

    keep = [*to_read, *(c for c in _FIT_WINDOW_DERIVED if c in requested)]
    built = build_columns(rows, FIT_WINDOW_COLUMN_SPECS, keep)
    order = np.argsort(built["window_id"], kind="stable")
    return {c: built[c][order] for c in requested}


def read_fit_scalars(h5_group: h5py.Group) -> Dict[str, Any]:
    """Read the cheap plan-level scalars from a ``stage5_fitting`` group.

    Reads only group attributes -- no window traversal at all. ``acquisition_us``
    is lifted out of the persisted Stage 5 ``parameters`` blob, which is where
    the fit records the active-FT acquisition length the resolution element
    ``1 / acquisition_us`` follows from.
    """
    parameters = load_json_attr(h5_group, "parameters", {}, label="stage5_fitting")
    acquisition = (
        parameters.get("acquisition_us") if isinstance(parameters, dict) else None
    )
    shape_attr = h5_group.attrs.get("shape", "lorentzian")
    if isinstance(shape_attr, bytes):
        shape_attr = shape_attr.decode("utf-8")
    creation = h5_group.attrs.get("creation_time", "unknown")
    if isinstance(creation, bytes):
        creation = creation.decode("utf-8")
    return {
        "n_windows": int(h5_group.attrs.get("n_windows", 0)),
        "n_fitted_peaks": int(h5_group.attrs.get("n_fitted_peaks", 0)),
        "shape": str(shape_attr),
        "final_plan_revision": int(h5_group.attrs.get("final_plan_revision", 0)),
        "acquisition_us": None if acquisition is None else float(acquisition),
        "creation_time": str(creation),
    }


def read_fit_parameters(h5_group: h5py.Group) -> Dict[str, Any]:
    """Cheap substitute for ``load_spectrum_fit_from_hdf5(h5_group).parameters``.

    Reads only the ``parameters`` JSON attribute on ``h5_group`` itself --
    the full loader's ``SpectrumFit.parameters`` is built from exactly this
    attribute and nothing else (see :func:`load_spectrum_fit_from_hdf5`), so
    this is the identical value without walking any window group. Returns
    ``{}`` when the attribute is absent, matching the full loader's default.
    """
    parameters: Dict[str, Any] = load_json_attr(
        h5_group, "parameters", {}, label="stage5_fitting"
    )
    return parameters


def read_fit_peak_frequencies_by_window(h5_group: h5py.Group) -> Dict[int, List[float]]:
    """Cheap substitute for grouping the full loader's fitted peaks by window.

    Equivalent to ``{wf.window_id: [p.frequency_mhz for p in wf.fitted_peaks]
    for wf in load_spectrum_fit_from_hdf5(h5_group).window_fits}``, but reads
    only each window's ``window_id`` attribute and its
    ``peaks/frequency_mhz`` dataset -- none of the other ~30 attrs/datasets
    the full loader pulls per window.

    Each window's list preserves its ``peaks`` subgroup row order -- NOT the
    full loader's separately (and globally) frequency-sorted
    ``SpectrumFit.fitted_peaks`` list. A caller that breaks a tie with
    ``min(fitted, key=...)`` depends on this per-window order to match the
    full loader's result exactly.
    """
    windows_group = _windows_group(h5_group)
    out: Dict[int, List[float]] = {}
    for name in sorted(windows_group.keys()):
        wg = windows_group[name]
        where = f"window {name!r}"
        wid = int(
            read_attr_value(
                wg, "window_id", FIT_WINDOW_COLUMN_SPECS["window_id"], where=where
            )
        )
        peaks_group = _peaks_subgroup(wg, where)
        freqs = read_dataset_column(
            peaks_group,
            "frequency_mhz",
            FIT_PEAK_COLUMN_SPECS["frequency_mhz"],
            None,
            where=f"{where} peaks",
        )
        out[wid] = [float(v) for v in freqs]
    return out


def read_fit_peak_uids_by_window(h5_group: h5py.Group) -> Dict[int, Set[int]]:
    """Cheap substitute for grouping the full loader's fitted peaks' uids by
    window.

    Equivalent to ``{wf.window_id: {p.peak_uid for p in wf.fitted_peaks if
    p.peak_uid is not None} for wf in
    load_spectrum_fit_from_hdf5(h5_group).window_fits}``, but reads only each
    window's ``window_id`` attribute and its ``peaks/peak_uid`` dataset.

    A window whose ``peaks`` subgroup has no ``peak_uid`` dataset (a file
    predating peak identity) maps to an empty set -- the same "contributes
    nothing to its window's set" result the full loader produces, since every
    row's ``peak_uid`` would decode to ``None`` (absent column, or a stored
    ``-1`` sentinel) and the full loader's set comprehension drops those.
    """
    windows_group = _windows_group(h5_group)
    out: Dict[int, Set[int]] = {}
    for name in sorted(windows_group.keys()):
        wg = windows_group[name]
        where = f"window {name!r}"
        wid = int(
            read_attr_value(
                wg, "window_id", FIT_WINDOW_COLUMN_SPECS["window_id"], where=where
            )
        )
        peaks_group = _peaks_subgroup(wg, where)
        if "peak_uid" in peaks_group:
            out[wid] = {int(v) for v in peaks_group["peak_uid"][:] if int(v) >= 0}
        else:
            out[wid] = set()
    return out


def read_fit_peak_freqs_and_uids_by_window(
    h5_group: h5py.Group,
) -> Tuple[Dict[int, List[float]], Dict[int, Set[int]]]:
    """Both per-window peak maps in ONE walk of the window groups.

    Returns exactly ``(read_fit_peak_frequencies_by_window(h5_group),
    read_fit_peak_uids_by_window(h5_group))`` -- same values, same per-window
    row order, same empty-set treatment of a file predating ``peak_uid`` --
    but resolving each window group, its ``window_id`` attribute and its
    ``peaks`` subgroup once instead of twice.

    For the caller that needs both (the curation advisory pass, when a batch
    carries a ``"uid:N"`` target). A caller wanting one of them should keep
    calling the single-column reader: this one always reads the
    ``frequency_mhz`` column, so it is not a free superset.
    """
    windows_group = _windows_group(h5_group)
    freqs_out: Dict[int, List[float]] = {}
    uids_out: Dict[int, Set[int]] = {}
    for name in sorted(windows_group.keys()):
        wg = windows_group[name]
        where = f"window {name!r}"
        wid = int(
            read_attr_value(
                wg, "window_id", FIT_WINDOW_COLUMN_SPECS["window_id"], where=where
            )
        )
        peaks_group = _peaks_subgroup(wg, where)
        freqs = read_dataset_column(
            peaks_group,
            "frequency_mhz",
            FIT_PEAK_COLUMN_SPECS["frequency_mhz"],
            None,
            where=f"{where} peaks",
        )
        freqs_out[wid] = [float(v) for v in freqs]
        if "peak_uid" in peaks_group:
            uids_out[wid] = {int(v) for v in peaks_group["peak_uid"][:] if int(v) >= 0}
        else:
            uids_out[wid] = set()
    return freqs_out, uids_out


class FitWindowCoverage(NamedTuple):
    """One window's cheap-resolution data, ordered ascending by ``window_id``
    to match :func:`load_spectrum_fit_from_hdf5`'s explicit sort of
    ``SpectrumFit.window_fits``.

    ``freq_range`` is ``None`` exactly when the full loader's
    ``FittingResult.window`` would be ``None`` (either bound NaN or absent --
    see ``_load_window_fit``). ``peak_uids`` is the set of non-``None``
    ``peak_uid`` values among the window's fitted peaks (empty on a file
    predating peak identity).
    """

    window_id: int
    freq_range: Optional[Tuple[float, float]]
    peak_uids: Set[int]


def read_fit_window_coverage(h5_group: h5py.Group) -> List[FitWindowCoverage]:
    """Cheap substitute for the per-window coverage data
    :func:`~ftmwpipeline._internal.stage5_impl._window_for_freq` and
    :func:`~ftmwpipeline._internal.stage6_impl._window_for_curation_token`
    resolve a curation target against.

    Equivalent to building, for each ``wf`` in
    ``load_spectrum_fit_from_hdf5(h5_group).window_fits``, the triple
    ``(wf.window_id, wf.window.freq_range if wf.window is not None else
    None, {p.peak_uid for p in wf.fitted_peaks if p.peak_uid is not None})``
    -- but reading only each window's ``window_id``, ``freq_min``,
    ``freq_max`` attributes and its ``peaks/peak_uid`` dataset. The result is
    sorted ascending by ``window_id``, the order those two resolvers iterate
    (first match wins), matching the full loader's own explicit sort.
    """
    windows_group = _windows_group(h5_group)
    rows: List[FitWindowCoverage] = []
    for name in sorted(windows_group.keys()):
        wg = windows_group[name]
        where = f"window {name!r}"
        wid = int(
            read_attr_value(
                wg, "window_id", FIT_WINDOW_COLUMN_SPECS["window_id"], where=where
            )
        )
        freq_min = float(
            read_attr_value(
                wg, "freq_min", FIT_WINDOW_COLUMN_SPECS["freq_min"], where=where
            )
        )
        freq_max = float(
            read_attr_value(
                wg, "freq_max", FIT_WINDOW_COLUMN_SPECS["freq_max"], where=where
            )
        )
        freq_range = (
            None if (np.isnan(freq_min) or np.isnan(freq_max)) else (freq_min, freq_max)
        )
        peaks_group = _peaks_subgroup(wg, where)
        if "peak_uid" in peaks_group:
            peak_uids = {int(v) for v in peaks_group["peak_uid"][:] if int(v) >= 0}
        else:
            peak_uids = set()
        rows.append(FitWindowCoverage(wid, freq_range, peak_uids))
    rows.sort(key=lambda r: r.window_id)
    return rows


# ---------------------------------------------------------------------------
# The event logs, as tables
# ---------------------------------------------------------------------------
#
# The fit's decision record -- the conservative add-loop audit, the doublet
# alternatives it weighed, the thaw and replan history, the rescue rounds -- is
# JSON-encoded rather than columnar, because it is written once and read as a
# narrative. These readers present it as flat tables anyway, for the consumer
# who wants the record in a spreadsheet or a dataframe.
#
# Unlike the peak and window readers, these are NOT cheap: reaching a JSON blob
# means parsing it, and there is no narrower path. The plan-level logs cost one
# parse each; the per-window ones (audit, doublets) cost one per window, which
# is a fraction of the full loader's work but not a constant. Said plainly here
# so nobody reads "read_*" as "free".

#: One row per audit step, across every window. The conservative add loop
#: records each candidate it tried and why it accepted, rejected or held it.
#: ``step_index`` is the position within its window's trail, so the narrative
#: order survives a re-sort.
FIT_AUDIT_COLUMN_SPECS: Dict[str, ColumnSpec] = {
    "window_id": ("i8", REQUIRED),
    "step_index": ("i8", REQUIRED),
    "decision": ("str", REQUIRED),
    "n_peaks_before": ("i8", REQUIRED),
    "candidate_offset_mhz": ("f8", REQUIRED),
    "chi2_before": ("f8", REQUIRED),
    "chi2_after": ("f8", REQUIRED),
    "f_statistic": ("f8", REQUIRED),
    "p_value": ("f8", REQUIRED),
    "aic_before": ("f8", REQUIRED),
    "aic_after": ("f8", REQUIRED),
    "n_eff": ("f8", float("nan")),
    "aicc_delta": ("f8", float("nan")),
    "separation_ok": ("bool", REQUIRED),
    "reason": ("str", ""),
}

#: One row per doublet alternative weighed, across every window: a pair the fit
#: could have merged into one line, with the evidence either way.
FIT_DOUBLET_COLUMN_SPECS: Dict[str, ColumnSpec] = {
    "window_id": ("i8", REQUIRED),
    "frequency_a_mhz": ("f8", REQUIRED),
    "frequency_b_mhz": ("f8", REQUIRED),
    "amplitude_a": ("f8", REQUIRED),
    "amplitude_b": ("f8", REQUIRED),
    "separation_res_elements": ("f8", REQUIRED),
    "amp_ratio": ("f8", REQUIRED),
    "chi2r_production": ("f8", REQUIRED),
    "chi2r_merged": ("f8", float("nan")),
    "delta_chi2_raw": ("f8", float("nan")),
    "delta_aicc": ("f8", float("nan")),
    "merged_frequency_mhz": ("f8", float("nan")),
    "merged_amplitude": ("f8", float("nan")),
    "merged_phase": ("f8", float("nan")),
    "merged_tau_us": ("f8", float("nan")),
    "merged_success": ("bool", False),
    "orth_evidence_delta_chi2": ("f8", float("nan")),
    "orth_evidence_n_params": ("i8", 3),
    "support_bins": ("i8", 0),
}

#: One row per thaw event: a frozen contributor released back to free because
#: holding it left the window's edge incoherent.
FIT_THAW_COLUMN_SPECS: Dict[str, ColumnSpec] = {
    "dependent_window_id": ("i8", REQUIRED),
    "primary_window_id": ("i8", REQUIRED),
    "contributor_peak_index": ("i8", REQUIRED),
    "contributor_frequency_mhz": ("f8", REQUIRED),
    "edge_side": ("str", REQUIRED),
    "edge_coherence_before": ("f8", REQUIRED),
    "edge_coherence_after": ("f8", REQUIRED),
    "accepted": ("bool", REQUIRED),
    "reason": ("str", ""),
}

#: One row per structural replan: a window boundary redrawn mid-fit because it
#: cut through a real feature.
FIT_REPLAN_COLUMN_SPECS: Dict[str, ColumnSpec] = {
    "triggering_window_id": ("i8", REQUIRED),
    "partner_window_id": ("i8", REQUIRED),
    "surviving_window_id": ("i8", REQUIRED),
    "edge_side": ("str", REQUIRED),
    "edge_coherence_before": ("f8", REQUIRED),
    "revision_before": ("i8", REQUIRED),
    "revision_after": ("i8", REQUIRED),
    "accepted": ("bool", REQUIRED),
    "reason": ("str", ""),
}

#: One row per rescue round: a re-search of a window's residual for peaks the
#: first pass missed. ``n_candidates`` counts the candidates the round examined;
#: their individual frequencies are nested in the record and stay with the full
#: loader.
FIT_RESCUE_COLUMN_SPECS: Dict[str, ColumnSpec] = {
    "window_id": ("i8", REQUIRED),
    "round_idx": ("i8", REQUIRED),
    "n_initial_peaks": ("i8", REQUIRED),
    "n_rescue_added": ("i8", REQUIRED),
    "n_pruned_total": ("i8", REQUIRED),
    "n_pruned_rescue_origin": ("i8", REQUIRED),
    "n_merged": ("i8", 0),
    "n_candidates": ("i8", 0),
    "chi2_before": ("f8", REQUIRED),
    "chi2_after": ("f8", REQUIRED),
    "tau_us_before": ("f8", REQUIRED),
    "tau_us_after": ("f8", REQUIRED),
    "accepted": ("bool", REQUIRED),
    "reason": ("str", ""),
}


def _read_plan_log(
    h5_group: h5py.Group,
    attr: str,
    specs: Dict[str, ColumnSpec],
    columns: Optional[Sequence[str]],
    *,
    table: str,
) -> Dict[str, np.ndarray]:
    """Read a plan-level JSON event log as a table. One parse, no traversal."""
    requested = resolve_column_selection(columns, list(specs), table=table)
    records = load_json_attr(h5_group, attr, [], label="stage5_fitting")
    rows: Dict[str, List[Any]] = {c: [] for c in requested}
    for i, record in enumerate(records):
        record_row(record, specs, requested, rows, where=f"{attr}[{i}]")
    return build_columns(rows, specs, requested)


def _read_window_log(
    h5_group: h5py.Group,
    attr: str,
    specs: Dict[str, ColumnSpec],
    columns: Optional[Sequence[str]],
    *,
    table: str,
    index_column: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Read a per-window JSON event log as one table, tagged by ``window_id``.

    *index_column*, when given, records each record's position within its
    window's log. Windows are visited in ascending id, so the table is grouped
    by window and ordered within it.
    """
    requested = resolve_column_selection(columns, list(specs), table=table)
    windows_group = _windows_group(h5_group)
    rows: Dict[str, List[Any]] = {c: [] for c in requested}

    for name in sorted(windows_group.keys()):
        wg = windows_group[name]
        window_id = read_attr_value(
            wg,
            "window_id",
            FIT_WINDOW_COLUMN_SPECS["window_id"],
            where=f"window {name!r}",
        )
        records = load_json_attr(wg, attr, [], label="stage5_fitting")
        for i, record in enumerate(records):
            extra: Dict[str, Any] = {"window_id": window_id}
            if index_column is not None:
                extra[index_column] = i
            record_row(
                record,
                specs,
                requested,
                rows,
                where=f"window {name!r} {attr}[{i}]",
                extra=extra,
            )

    built = build_columns(rows, specs, requested)
    return {c: built[c] for c in requested}


def read_fit_audit_columns(
    h5_group: h5py.Group,
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Read the conservative add-loop audit trail across every window.

    Costs one JSON parse per window -- see the note above these readers.
    See :data:`FIT_AUDIT_COLUMN_SPECS` for the available columns.
    """
    return _read_window_log(
        h5_group,
        "audit_trail",
        FIT_AUDIT_COLUMN_SPECS,
        columns,
        table="fit_audit",
        index_column="step_index",
    )


def read_fit_doublet_columns(
    h5_group: h5py.Group,
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Read the doublet alternatives weighed, across every window.

    Costs one JSON parse per window. Empty on files written before the
    doublet-alternative pass existed.
    See :data:`FIT_DOUBLET_COLUMN_SPECS` for the available columns.
    """
    return _read_window_log(
        h5_group,
        "doublet_alternatives",
        FIT_DOUBLET_COLUMN_SPECS,
        columns,
        table="fit_doublets",
    )


def read_fit_thaw_columns(
    h5_group: h5py.Group,
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Read the plan-level thaw history in chronological order.

    See :data:`FIT_THAW_COLUMN_SPECS` for the available columns.
    """
    return _read_plan_log(
        h5_group, "thaw_history", FIT_THAW_COLUMN_SPECS, columns, table="fit_thaw"
    )


def read_fit_replan_columns(
    h5_group: h5py.Group,
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Read the plan-level structural-replan history in chronological order.

    See :data:`FIT_REPLAN_COLUMN_SPECS` for the available columns.
    """
    return _read_plan_log(
        h5_group,
        "replan_history",
        FIT_REPLAN_COLUMN_SPECS,
        columns,
        table="fit_replans",
    )


def read_fit_rescue_columns(
    h5_group: h5py.Group,
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Read the plan-level rescue-round history in chronological order.

    See :data:`FIT_RESCUE_COLUMN_SPECS` for the available columns.
    """
    requested = resolve_column_selection(
        columns, list(FIT_RESCUE_COLUMN_SPECS), table="fit_rescues"
    )
    records = load_json_attr(h5_group, "rescue_history", [], label="stage5_fitting")
    rows: Dict[str, List[Any]] = {c: [] for c in requested}
    for i, record in enumerate(records):
        # n_candidates is a count of a nested list, not a stored field.
        counted = dict(record)
        counted["n_candidates"] = len(record.get("candidates") or ())
        record_row(
            counted,
            FIT_RESCUE_COLUMN_SPECS,
            requested,
            rows,
            where=f"rescue_history[{i}]",
        )
    return build_columns(rows, FIT_RESCUE_COLUMN_SPECS, requested)
