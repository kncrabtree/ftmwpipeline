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
    read_dataset_column,
    record_row,
    reset_group,
    resolve_column_selection,
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
    "LEGACY_FIT_LAYOUT_MESSAGE",
    "FitWindowCoverage",
    "read_fit_window_coverage",
]


# --- the flat table layout -------------------------------------------------
#
# One row per window in ``windows/``, one row per fitted peak in ``peaks/``,
# and a single ``covariance`` dataset holding every window's matrix
# end to end. A window addresses its own rows through
# ``peak_offset``/``peak_count`` and ``covariance_offset``/``covariance_dim``.
#
# Both tables are stored ascending by ``window_id``, and within a window the
# peak rows keep the order of its ``fitted_peaks`` list. Readers depend on
# both facts: the window order is what makes a coverage scan first-match-wins,
# and the row order is what a caller breaking a tie with ``min()`` matches
# against the full loader on.

#: Window-table columns and their on-disk dtypes. ``"str"`` means a
#: variable-length UTF-8 column.
_WINDOW_TABLE_COLUMNS: Dict[str, str] = {
    "window_id": "i8",
    "success": "i1",
    "cost": "f8",
    "iterations": "i8",
    "aic": "f8",
    "reduced_chi2": "f8",
    "tau_us": "f8",
    "tau_error": "f8",
    "tau_fitted": "i1",
    "freq_min": "f8",
    "freq_max": "f8",
    "edge_coherence_low": "f8",
    "edge_coherence_high": "f8",
    "peak_offset": "i8",
    "peak_count": "i8",
    "covariance_offset": "i8",
    "covariance_dim": "i8",
    "shape": "str",
    "fixed_parameters": "str",
    "quality_metrics": "str",
    "audit_trail": "str",
    "thaw_events": "str",
    "rescue_events": "str",
    "doublet_alternatives": "str",
    "covariance_labels": "str",
}

#: Peak columns a fit may legitimately lack, and the sentinel that stands in.
#: The *layout* is fixed, but a column can still be absent: ``peak_uid`` is
#: missing from a fit produced before peak identity existed, and the honest
#: value there is "no identifier", not a refusal to load.
_DEFAULTED_PEAK_COLUMNS: Dict[str, int] = {
    "flat_decay": 0,
    "derivation": -1,
    "peak_uid": -1,
}

#: The numeric half of the peak table (the two string columns,
#: ``clock_lattice`` and ``origin``, are built separately).
_PEAK_TABLE_NUMERIC: Dict[str, str] = {
    "detection_index": "i8",
    "frequency_mhz": "f8",
    "amplitude": "f8",
    "phase": "f8",
    "decay_rate": "f8",
    "frequency_error": "f8",
    "amplitude_error": "f8",
    "phase_error": "f8",
    "decay_rate_error": "f8",
    "snr": "f8",
    "chi_squared": "f8",
    "window_id": "i8",
    "knockout_delta_chi2": "f8",
    "knockout_expected_delta_chi2": "f8",
    "knockout_supported": "i1",
    "knockout_p_value": "f8",
    "knockout_n_eff": "f8",
    "knockout_aicc_delta": "f8",
    "flat_decay": "i1",
    "derivation": "i8",
    "peak_uid": "i8",
}


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
    """Write a :class:`SpectrumFit` to an HDF5 group, replacing its contents.

    Parameters
    ----------
    fit : SpectrumFit
        The persistent fit aggregate. The merged global ``fitted_peaks``
        list is rebuilt on load from the peak table; it is *not* stored
        independently here.
    h5_group : h5py.Group
        Destination group; any existing fit content is overwritten.
    """
    reset_group(h5_group)
    _write_fit_header(fit, h5_group)
    window_columns, peak_columns, covariance = _fit_tables(fit)
    _write_table(h5_group.require_group("windows"), window_columns)
    _write_table(h5_group.require_group("peaks"), peak_columns)
    _write_table(h5_group, {"covariance": covariance})


def update_spectrum_fit_windows_in_hdf5(
    fit: SpectrumFit, h5_group: h5py.Group, window_ids: Iterable[int]
) -> None:
    """Rewrite an already-persisted fit in place, without deleting anything.

    The incremental counterpart to :func:`save_spectrum_fit_to_hdf5` for the
    curation engine's one write point (S4). ``window_ids`` names the windows
    the batch changed; under the flat layout it is advisory only -- the two
    tables are contiguous, so a window whose peak count changed shifts every
    row after it, and rewriting the tables whole is both simpler and cheaper
    than splicing them. Writing ~30 columns of a few hundred rows costs
    single-digit milliseconds, against the 404 ms the per-window-group layout
    charged for the same edit.

    What matters here is *how* the rewrite happens: every dataset is chunked
    and resizable, and is resized and overwritten rather than deleted and
    recreated. HDF5 does not reclaim a deleted object's space, so the
    delete-and-recreate this replaced grew the file by the size of the whole
    fit on every curation write. In-place assignment does not.
    """
    del window_ids  # advisory under the flat layout; see the docstring
    _write_fit_header(fit, h5_group)
    window_columns, peak_columns, covariance = _fit_tables(fit)
    _write_table(h5_group.require_group("windows"), window_columns)
    _write_table(h5_group.require_group("peaks"), peak_columns)
    _write_table(h5_group, {"covariance": covariance})


def _write_fit_header(fit: SpectrumFit, h5_group: h5py.Group) -> None:
    """The group-level attrs: counts, parameters, diagnostics, histories."""
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


def _write_table(group: h5py.Group, columns: Dict[str, np.ndarray]) -> None:
    """Write *columns* as equal-length datasets, resizing in place if they
    already exist.

    Every dataset is created chunked with an unbounded ``maxshape`` so a
    later write can resize it rather than delete it -- see
    :func:`update_spectrum_fit_windows_in_hdf5` for why that matters.
    """
    for name, data in columns.items():
        if name in group:
            dataset = group[name]
            if dataset.shape[0] != len(data):
                dataset.resize((len(data),))
            if len(data):
                dataset[...] = data
            continue
        # An object array is a vlen-UTF-8 column. The dtype is passed
        # explicitly rather than inferred: h5py can infer it from a populated
        # object array but not from an empty one, and a fit with no windows
        # writes every string column empty.
        dtype = (
            h5py.string_dtype(encoding="utf-8")
            if getattr(data, "dtype", None) == object
            else None
        )
        group.create_dataset(
            name,
            data=data,
            dtype=dtype,
            maxshape=(None,),
            chunks=(max(len(data), 1),),
        )


def _fit_tables(
    fit: SpectrumFit,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
    """Flatten a :class:`SpectrumFit` into ``(windows, peaks, covariance)``.

    Windows come out ascending by ``window_id`` -- the order the loader and
    every column reader rely on. Peaks come out grouped by window in that
    same order, each window's rows keeping the order they have in its
    ``fitted_peaks`` list, which is what lets ``peak_offset``/``peak_count``
    address them and what preserves the per-window row order the column
    readers document.

    Each window's covariance matrix is flattened row-major into one shared
    1-D dataset, addressed by ``covariance_offset``/``covariance_dim``; a
    window without one contributes nothing and carries ``covariance_dim``
    of 0.
    """
    window_fits = sorted(
        fit.window_fits,
        key=lambda wf: (wf.window_id if wf.window_id is not None else -1),
    )
    for window_fit in window_fits:
        if window_fit.window_id is None:
            raise ValueError(
                "FittingResult.window_id is required for serialization "
                "(the active-FT slice cannot be reconstructed without it)"
            )

    rows: Dict[str, List[Any]] = {name: [] for name in _WINDOW_TABLE_COLUMNS}
    ordered_peaks: List[FittedPeak] = []
    peak_window_ids: List[int] = []
    covariance_blocks: List[np.ndarray] = []
    peak_offset = 0
    covariance_offset = 0

    for window_fit in window_fits:
        wid = int(window_fit.window_id)  # type: ignore[arg-type]
        tau_entry = window_fit.shared_parameters.get("tau_us") or {}
        tau_fitted_val = tau_entry.get("fitted")
        window = window_fit.window
        quality = window_fit.quality_metrics

        cov = getattr(window_fit, "covariance", None)
        labels = getattr(window_fit, "covariance_param_labels", None)
        if cov is not None and labels is not None:
            block = np.asarray(cov, dtype="f8")
            covariance_blocks.append(block.reshape(-1))
            cov_dim = int(block.shape[0])
            cov_labels = json.dumps(list(labels))
        else:
            cov_dim = 0
            cov_labels = ""

        rows["window_id"].append(wid)
        rows["success"].append(1 if bool(window_fit.success) else 0)
        rows["cost"].append(float(window_fit.cost))
        rows["iterations"].append(int(window_fit.iterations))
        rows["aic"].append(float(window_fit.aic))
        rows["reduced_chi2"].append(float(window_fit.reduced_chi2))
        rows["tau_us"].append(float(tau_entry.get("value", float("nan"))))
        rows["tau_error"].append(nan_if_none(tau_entry.get("error")))
        rows["tau_fitted"].append(
            -1 if tau_fitted_val is None else (1 if bool(tau_fitted_val) else 0)
        )
        # freq_range is persisted so visualization can place the window
        # without the Stage 4 plan; the spectrum slice stays recomputable.
        rows["freq_min"].append(
            float("nan") if window is None else float(window.freq_range[0])
        )
        rows["freq_max"].append(
            float("nan") if window is None else float(window.freq_range[1])
        )
        rows["edge_coherence_low"].append(
            float(quality.get("edge_coherence_low", float("nan")))
        )
        rows["edge_coherence_high"].append(
            float(quality.get("edge_coherence_high", float("nan")))
        )
        rows["peak_offset"].append(peak_offset)
        rows["peak_count"].append(len(window_fit.fitted_peaks))
        rows["covariance_offset"].append(covariance_offset)
        rows["covariance_dim"].append(cov_dim)
        rows["shape"].append(str(getattr(window_fit, "shape", "lorentzian")))
        rows["fixed_parameters"].append(
            json.dumps(window_fit.fixed_parameters, default=str)
        )
        rows["quality_metrics"].append(json.dumps(quality, default=str))
        rows["audit_trail"].append(
            json.dumps([_audit_step_to_json(s) for s in window_fit.audit_trail])
        )
        rows["thaw_events"].append(
            json.dumps([_thaw_info_to_json(e) for e in window_fit.thaw_events])
        )
        rows["rescue_events"].append(
            json.dumps([_rescue_round_to_json(e) for e in window_fit.rescue_events])
        )
        rows["doublet_alternatives"].append(
            json.dumps(
                [
                    _doublet_alternative_to_json(d)
                    for d in getattr(window_fit, "doublet_alternatives", [])
                ]
            )
        )
        rows["covariance_labels"].append(cov_labels)

        ordered_peaks.extend(window_fit.fitted_peaks)
        peak_window_ids.extend([wid] * len(window_fit.fitted_peaks))
        peak_offset += len(window_fit.fitted_peaks)
        covariance_offset += cov_dim * cov_dim

    window_columns = {
        name: _as_column(rows[name], dtype)
        for name, dtype in _WINDOW_TABLE_COLUMNS.items()
    }
    peak_columns = _peak_table(ordered_peaks, peak_window_ids)
    covariance = (
        np.concatenate(covariance_blocks)
        if covariance_blocks
        else np.empty(0, dtype="f8")
    )
    return window_columns, peak_columns, covariance


def _as_column(values: List[Any], dtype: str) -> np.ndarray:
    """One table column as a numpy array, vlen-UTF-8 for the string columns."""
    if dtype == "str":
        column = np.empty(len(values), dtype=object)
        for i, value in enumerate(values):
            column[i] = value
        return column
    return np.asarray(values, dtype=dtype)


def _peak_table(
    peaks: List[FittedPeak], window_ids: List[int]
) -> Dict[str, np.ndarray]:
    """The peak table's columns, one row per peak, in the given order.

    ``window_ids`` is the owning window's id per row. A peak whose own
    ``window_id`` is ``None`` is stamped with it rather than with a ``-1``
    sentinel: it belongs to that window whatever its own field says, and a
    column that disagreed would let a reader that trusts the column drop the
    row out of its window silently.
    """
    n = len(peaks)
    columns: Dict[str, np.ndarray] = {
        name: np.empty(n, dtype=dtype) for name, dtype in _PEAK_TABLE_NUMERIC.items()
    }
    clock_lattice_col: np.ndarray = np.empty(n, dtype=object)
    origin_col: np.ndarray = np.empty(n, dtype=object)

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
        columns["window_id"][i] = (
            window_ids[i] if p.window_id is None else int(p.window_id)
        )
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
        columns["flat_decay"][i] = 1 if bool(p.flat_decay) else 0
        columns["derivation"][i] = -1 if p.derivation is None else int(p.derivation)
        columns["peak_uid"][i] = -1 if p.peak_uid is None else int(p.peak_uid)
        clock_lattice_col[i] = p.clock_lattice or ""
        origin_col[i] = p.origin or "auto"

    columns["clock_lattice"] = clock_lattice_col
    columns["origin"] = origin_col
    return columns


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_spectrum_fit_from_hdf5(h5_group: h5py.Group) -> SpectrumFit:
    """Load a :class:`SpectrumFit` from an HDF5 group, validating loudly.

    Reads the two flat tables whole -- roughly thirty dataset reads,
    regardless of how many windows the fit has -- and slices each window's
    peaks out of the peak table by ``peak_offset``/``peak_count``. The
    merged global :attr:`SpectrumFit.fitted_peaks` list is rebuilt from those
    rows, sorted ascending by molecular frequency.

    Raises
    ------
    ValueError
        If the ``windows`` or ``peaks`` table is missing, a required column
        is absent, the columns have mismatched lengths, a window's peak or
        covariance slice does not lie inside the table, or a JSON-encoded
        audit/thaw/replan entry has an invalid ``decision``/``edge_side``
        label.
    """
    windows_group = _windows_group(h5_group)
    peaks_group = _peaks_table(h5_group)

    final_plan_revision = int(h5_group.attrs.get("final_plan_revision", 0))
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

    window_columns = _read_window_table(windows_group)
    peak_columns = _read_peak_table(peaks_group)
    covariance_flat = (
        np.asarray(h5_group["covariance"][:], dtype="f8")
        if "covariance" in h5_group
        else np.empty(0, dtype="f8")
    )
    n_peak_rows = len(peak_columns["detection_index"])

    window_fits: List[FittingResult] = []
    for row in range(len(window_columns["window_id"])):
        window_fits.append(
            _window_fit_from_row(
                window_columns,
                peak_columns,
                covariance_flat,
                row=row,
                n_peak_rows=n_peak_rows,
            )
        )

    window_fits.sort(key=lambda wf: (wf.window_id if wf.window_id is not None else -1))

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


def _decode(value: Any) -> str:
    """One vlen-UTF-8 cell as ``str``."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_window_table(windows_group: h5py.Group) -> Dict[str, np.ndarray]:
    """Every window column, validated for presence and equal length."""
    missing = [c for c in _WINDOW_TABLE_COLUMNS if c not in windows_group]
    if missing:
        raise ValueError(f"stage5_fitting windows table missing column(s): {missing}")
    columns = {c: windows_group[c][:] for c in _WINDOW_TABLE_COLUMNS}
    lengths = {c: len(v) for c, v in columns.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"stage5_fitting windows table columns have mismatched lengths: {lengths}"
        )
    return columns


def _read_peak_table(peaks_group: h5py.Group) -> Dict[str, np.ndarray]:
    """Every peak column, with the optional ones defaulted when absent.

    The layout is fixed, but individual columns are still allowed to be
    missing: ``peak_uid`` is absent from a fit produced before peak identity
    existed, and the honest value there is "no identifier", not a refusal.
    """
    required = [
        c
        for c in _PEAK_TABLE_NUMERIC
        if c not in _OPTIONAL_PEAK_COLUMNS and c not in _DEFAULTED_PEAK_COLUMNS
    ]
    missing = [c for c in required if c not in peaks_group]
    if missing:
        raise ValueError(f"stage5_fitting peaks table missing column(s): {missing}")

    columns: Dict[str, np.ndarray] = {c: peaks_group[c][:] for c in required}
    n_rows = len(columns["detection_index"])
    for name, fill in _DEFAULTED_PEAK_COLUMNS.items():
        if name in peaks_group:
            columns[name] = peaks_group[name][:]
        else:
            columns[name] = np.full(n_rows, fill, dtype=_PEAK_TABLE_NUMERIC[name])
    for name in _OPTIONAL_PEAK_COLUMNS:
        if name in peaks_group:
            columns[name] = peaks_group[name][:]
        else:
            columns[name] = np.full(n_rows, float("nan"), dtype="f8")

    lengths = {c: len(v) for c, v in columns.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"stage5_fitting peaks table columns have mismatched lengths: {lengths}"
        )

    if "clock_lattice" in peaks_group:
        columns["clock_lattice"] = peaks_group["clock_lattice"][:]
    else:
        columns["clock_lattice"] = np.array([""] * n_rows, dtype=object)
    if "origin" in peaks_group:
        columns["origin"] = peaks_group["origin"][:]
    else:
        columns["origin"] = np.array(["auto"] * n_rows, dtype=object)
    return columns


def _window_fit_from_row(
    window_columns: Dict[str, np.ndarray],
    peak_columns: Dict[str, np.ndarray],
    covariance_flat: np.ndarray,
    *,
    row: int,
    n_peak_rows: int,
) -> FittingResult:
    """Rebuild one :class:`FittingResult` from its row of the window table."""
    wid = int(window_columns["window_id"][row])
    where = f"window {wid}"

    freq_min = float(window_columns["freq_min"][row])
    freq_max = float(window_columns["freq_max"][row])
    window_obj: Optional[SpectralWindow]
    if np.isnan(freq_min) or np.isnan(freq_max):
        window_obj = None
    else:
        # A lightweight SpectralWindow: what visualization needs from
        # ``window`` is its freq_range, and the complex spectrum slice stays
        # recomputable from the FID + active FT.
        window_obj = SpectralWindow(
            parent_ft=None,
            freq_array=np.array([], dtype=float),
            complex_spectrum=np.array([], dtype=np.complex128),
            freq_range=(freq_min, freq_max),
            window_id=wid,
        )

    result = FittingResult(
        success=bool(window_columns["success"][row]),
        fitted_spectrum=None,  # recomputed on demand
        cost=float(window_columns["cost"][row]),
        iterations=int(window_columns["iterations"][row]),
        aic=float(window_columns["aic"][row]),
        reduced_chi2=float(window_columns["reduced_chi2"][row]),
        window=window_obj,
        window_id=wid,
        shape=_decode(window_columns["shape"][row]),
    )

    start = int(window_columns["peak_offset"][row])
    count = int(window_columns["peak_count"][row])
    if start < 0 or count < 0 or start + count > n_peak_rows:
        raise ValueError(
            f"{where} peak slice [{start}, {start + count}) does not lie inside "
            f"the {n_peak_rows}-row peaks table"
        )
    fitted_peaks = _peaks_from_rows(peak_columns, start, start + count, window_id=wid)
    result.fitted_peaks = fitted_peaks

    tau_error = none_if_nan(float(window_columns["tau_error"][row]))
    # tau_fitted: 1 -> True, 0 -> False, -1 -> unknown, inferred from whether
    # an error was estimated (a finite error implies tau was fit).
    raw_tau_fitted = int(window_columns["tau_fitted"][row])
    tau_fitted: Optional[bool]
    if raw_tau_fitted == 1:
        tau_fitted = True
    elif raw_tau_fitted == 0:
        tau_fitted = False
    else:
        tau_fitted = True if tau_error is not None else None
    result.shared_parameters["tau_us"] = {
        "value": float(window_columns["tau_us"][row]),
        "error": tau_error,
        "fitted": tau_fitted,
        "detection_indices": [p.detection_index for p in fitted_peaks],
    }

    result.fixed_parameters = _row_json(window_columns, "fixed_parameters", row, {})
    result.quality_metrics = _row_json(window_columns, "quality_metrics", row, {})
    # The scalar edge-coherence columns are canonical: they make it back into
    # quality_metrics even if a hand-edit nuked the JSON cell.
    result.quality_metrics["edge_coherence_low"] = float(
        window_columns["edge_coherence_low"][row]
    )
    result.quality_metrics["edge_coherence_high"] = float(
        window_columns["edge_coherence_high"][row]
    )

    result.audit_trail = [
        _json_to_audit_step(blob, f"{where}/audit_trail[{i}]")
        for i, blob in enumerate(_row_json(window_columns, "audit_trail", row, []))
    ]
    result.thaw_events = [
        _json_to_thaw_info(blob, f"{where}/thaw_events[{i}]")
        for i, blob in enumerate(_row_json(window_columns, "thaw_events", row, []))
    ]
    result.rescue_events = [
        _json_to_rescue_round(blob, f"{where}/rescue_events[{i}]")
        for i, blob in enumerate(_row_json(window_columns, "rescue_events", row, []))
    ]
    result.doublet_alternatives = [
        _json_to_doublet_alternative(blob, f"{where}/doublet_alternatives[{i}]")
        for i, blob in enumerate(
            _row_json(window_columns, "doublet_alternatives", row, [])
        )
    ]

    # Covariance: omitted when JtJ was singular, which round-trips as
    # covariance_dim == 0 -> None.
    dim = int(window_columns["covariance_dim"][row])
    if dim > 0:
        offset = int(window_columns["covariance_offset"][row])
        if offset < 0 or offset + dim * dim > len(covariance_flat):
            raise ValueError(
                f"{where} covariance slice [{offset}, {offset + dim * dim}) does "
                f"not lie inside the {len(covariance_flat)}-element covariance "
                f"dataset"
            )
        labels_raw = _decode(window_columns["covariance_labels"][row])
        try:
            labels_loaded: List[str] = json.loads(labels_raw)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"{where} 'covariance_labels' is not valid JSON") from exc
        if len(labels_loaded) != dim:
            raise ValueError(
                f"{where} covariance label count ({len(labels_loaded)}) does not "
                f"match matrix dimension ({dim})"
            )
        n_amp_labels = sum(1 for lbl in labels_loaded if lbl.startswith("amplitude_"))
        if n_amp_labels != len(fitted_peaks):
            raise ValueError(
                f"{where} covariance has {n_amp_labels} amplitude label(s) but "
                f"{len(fitted_peaks)} fitted peak(s)"
            )
        result.covariance = covariance_flat[offset : offset + dim * dim].reshape(
            dim, dim
        )
        result.covariance_param_labels = labels_loaded

    return result


def _row_json(columns: Dict[str, np.ndarray], name: str, row: int, default: Any) -> Any:
    """One JSON-encoded table cell, decoded; *default* when it is empty."""
    raw = _decode(columns[name][row])
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default


def _peaks_from_rows(
    columns: Dict[str, np.ndarray], start: int, stop: int, *, window_id: int
) -> List[FittedPeak]:
    """Rebuild ``[start, stop)`` of the peak table as :class:`FittedPeak`.

    ``window_id`` backfills a stored ``-1``: the peak lies in that window's
    slice whatever its own column says, so the slice wins.
    """
    peaks: List[FittedPeak] = []
    for i in range(start, stop):
        ko_supported_raw = int(columns["knockout_supported"][i])
        ko_delta = float(columns["knockout_delta_chi2"][i])
        if ko_supported_raw < 0 or np.isnan(ko_delta):
            knockout: Optional[KnockoutInfo] = None
        else:
            knockout = KnockoutInfo(
                delta_chi2=ko_delta,
                expected_delta_chi2=float(columns["knockout_expected_delta_chi2"][i]),
                supported=bool(ko_supported_raw),
                p_value=float(columns["knockout_p_value"][i]),
                n_eff=float(columns["knockout_n_eff"][i]),
                aicc_delta=float(columns["knockout_aicc_delta"][i]),
            )
        wid_raw = int(columns["window_id"][i])
        if wid_raw < 0:
            wid_raw = window_id
        derivation_raw = int(columns["derivation"][i])
        uid_raw = int(columns["peak_uid"][i])
        peaks.append(
            FittedPeak(
                detection_index=int(columns["detection_index"][i]),
                frequency_mhz=float(columns["frequency_mhz"][i]),
                amplitude=float(columns["amplitude"][i]),
                phase=none_if_nan(float(columns["phase"][i])),
                decay_rate=none_if_nan(float(columns["decay_rate"][i])),
                frequency_error=none_if_nan(float(columns["frequency_error"][i])),
                amplitude_error=none_if_nan(float(columns["amplitude_error"][i])),
                phase_error=none_if_nan(float(columns["phase_error"][i])),
                decay_rate_error=none_if_nan(float(columns["decay_rate_error"][i])),
                snr=none_if_nan(float(columns["snr"][i])),
                chi_squared=none_if_nan(float(columns["chi_squared"][i])),
                window_id=None if wid_raw < 0 else wid_raw,
                knockout=knockout,
                clock_lattice=_decode(columns["clock_lattice"][i]) or None,
                origin=_decode(columns["origin"][i]) or "auto",
                flat_decay=bool(int(columns["flat_decay"][i])),
                derivation=None if derivation_raw < 0 else derivation_raw,
                peak_uid=None if uid_raw < 0 else uid_raw,
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


#: What a reader is told when it opens a fit written in the pre-1.0 layout.
LEGACY_FIT_LAYOUT_MESSAGE = (
    "this file's Stage 5 fit uses the pre-1.0 per-window layout "
    "(stage5_fitting/windows/window_NNNN/), which this version cannot read. "
    "Re-run 'fit run' to rebuild the fit in the current layout. Any Stage 6 "
    "curation on it must be re-applied."
)


def _windows_group(h5_group: h5py.Group) -> h5py.Group:
    if "windows" not in h5_group:
        raise ValueError("stage5_fitting group missing required 'windows' table")
    windows = h5_group["windows"]
    if "window_id" not in windows:
        # The flat layout always has that column. Its absence is either a
        # corrupt table or -- far more likely, and worth saying out loud --
        # a file from before the layout changed, where `windows` held one
        # subgroup per window instead of one dataset per column. Guessing
        # wrong here costs nothing: both readings end in a refusal, and only
        # one of them tells the reader what to do about it.
        if any(isinstance(windows.get(name), h5py.Group) for name in windows):
            raise ValueError(LEGACY_FIT_LAYOUT_MESSAGE)
        raise ValueError("stage5_fitting windows table missing column 'window_id'")
    return windows


def _peaks_table(h5_group: h5py.Group) -> h5py.Group:
    if "peaks" not in h5_group:
        raise ValueError("stage5_fitting group missing required 'peaks' table")
    return h5_group["peaks"]


def _peak_anchor(peaks_group: h5py.Group) -> h5py.Dataset:
    """The peak table's row-count-defining column."""
    try:
        return peaks_group["detection_index"]
    except KeyError:
        raise ValueError(
            "stage5_fitting peaks table missing required column " "'detection_index'"
        ) from None


def read_fit_peak_columns(
    h5_group: h5py.Group,
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Read fitted-peak columns from a ``stage5_fitting`` group in bulk.

    One whole-dataset read per requested column against the flat peak table
    -- no per-window traversal, and nothing else in the fit (audit trails,
    thaw/rescue events, doublet alternatives, covariance) is touched.

    Rows are ordered by ascending molecular frequency, matching
    :attr:`SpectrumFit.fitted_peaks`, so a consumer can substitute this for
    the full loader row-for-row.

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
    peaks_group = _peaks_table(h5_group)
    n_rows = int(_peak_anchor(peaks_group).shape[0])

    # frequency_mhz always read: it defines the row order.
    stored = [c for c in requested if c not in _FIT_PEAK_DERIVED]
    to_read = list(dict.fromkeys(["frequency_mhz", *stored]))

    read: Dict[str, np.ndarray] = {}
    for col in to_read:
        read[col] = read_dataset_column(
            peaks_group,
            col,
            FIT_PEAK_COLUMN_SPECS[col],
            n_rows,
            where="stage5_fitting peaks",
        )

    if "window_id" in read or "shape" in requested:
        window_ids = np.asarray(_windows_group(h5_group)["window_id"][:], dtype="i8")
        offsets = np.asarray(_windows_group(h5_group)["peak_offset"][:], dtype="i8")
        counts = np.asarray(_windows_group(h5_group)["peak_count"][:], dtype="i8")
        # The row's slice is the authority on which window owns it, exactly as
        # the owning group used to be: a peak stored under a window belongs to
        # it whatever its own column says, and backfilling here keeps this tap
        # and the full loader agreeing rather than silently ungrouping a row.
        owner = np.full(n_rows, -1, dtype="i8")
        for wid, start_row, count in zip(window_ids, offsets, counts):
            owner[int(start_row) : int(start_row) + int(count)] = int(wid)
        if "window_id" in read:
            read["window_id"] = np.where(
                read["window_id"] < 0, owner, read["window_id"]
            )
        if "shape" in requested:
            shapes = _windows_group(h5_group)["shape"][:]
            per_row = np.empty(n_rows, dtype=object)
            for shape_value, start_row, count in zip(shapes, offsets, counts):
                per_row[int(start_row) : int(start_row) + int(count)] = _decode(
                    shape_value
                )
            read["shape"] = per_row

    order = np.argsort(read["frequency_mhz"], kind="stable")
    return {c: np.asarray(read[c])[order] for c in requested}


def read_fit_window_columns(
    h5_group: h5py.Group,
    columns: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Read per-window fit scalars from a ``stage5_fitting`` group in bulk.

    One whole-dataset read per requested column against the flat window
    table; the JSON-encoded audit/thaw/rescue/doublet cells are never parsed.
    Rows are ordered by ascending ``window_id``, matching
    :attr:`SpectrumFit.window_fits`.

    See :data:`FIT_WINDOW_COLUMN_SPECS` for the available columns and their
    sentinel conventions.
    """
    requested = resolve_column_selection(
        columns, list(FIT_WINDOW_COLUMN_SPECS), table="fit_windows"
    )
    windows_group = _windows_group(h5_group)
    n_rows = int(np.asarray(windows_group["window_id"]).shape[0])

    # window_id always read: it defines the row order.
    stored = [c for c in requested if c not in _FIT_WINDOW_DERIVED]
    to_read = list(dict.fromkeys(["window_id", *stored]))

    built: Dict[str, np.ndarray] = {}
    for col in to_read:
        built[col] = read_dataset_column(
            windows_group,
            col,
            FIT_WINDOW_COLUMN_SPECS[col],
            n_rows,
            where="stage5_fitting windows",
        )
    if "n_peaks" in requested:
        # A window's row count is the length of its slice of the peak table.
        built["n_peaks"] = np.asarray(windows_group["peak_count"][:], dtype="i8")

    order = np.argsort(built["window_id"], kind="stable")
    return {c: np.asarray(built[c])[order] for c in requested}


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


def _window_slices(h5_group: h5py.Group) -> List[Tuple[int, int, int]]:
    """``(window_id, start, stop)`` per window, ascending by ``window_id``.

    The join between the two tables, read once: three integer columns, no
    dataset of peak values touched. Ascending ``window_id`` is the order the
    coverage resolvers scan in (first match wins) and the order the full
    loader sorts ``window_fits`` into.
    """
    windows_group = _windows_group(h5_group)
    for column in ("window_id", "peak_offset", "peak_count"):
        if column not in windows_group:
            raise ValueError(f"stage5_fitting windows table missing column {column!r}")
    ids = np.asarray(windows_group["window_id"][:], dtype="i8")
    offsets = np.asarray(windows_group["peak_offset"][:], dtype="i8")
    counts = np.asarray(windows_group["peak_count"][:], dtype="i8")
    rows = [
        (int(wid), int(off), int(off) + int(count))
        for wid, off, count in zip(ids, offsets, counts)
    ]
    rows.sort(key=lambda r: r[0])
    return rows


def read_fit_peak_frequencies_by_window(h5_group: h5py.Group) -> Dict[int, List[float]]:
    """Cheap substitute for grouping the full loader's fitted peaks by window.

    Equivalent to ``{wf.window_id: [p.frequency_mhz for p in wf.fitted_peaks]
    for wf in load_spectrum_fit_from_hdf5(h5_group).window_fits}``, but
    reading the ``frequency_mhz`` column and the three join columns -- four
    datasets in total, whatever the window count -- and none of the ~20 other
    columns the full loader pulls.

    Each window's list preserves its slice's row order -- NOT the full
    loader's separately (and globally) frequency-sorted
    ``SpectrumFit.fitted_peaks`` list. A caller that breaks a tie with
    ``min(fitted, key=...)`` depends on this order to match the full loader's
    result exactly.
    """
    slices = _window_slices(h5_group)
    peaks_group = _peaks_table(h5_group)
    freqs = np.asarray(peaks_group["frequency_mhz"][:], dtype="f8")
    return {wid: [float(v) for v in freqs[start:stop]] for wid, start, stop in slices}


def _uid_column(h5_group: h5py.Group) -> Optional[np.ndarray]:
    """The ``peak_uid`` column, or ``None`` on a fit predating peak identity."""
    peaks_group = _peaks_table(h5_group)
    if "peak_uid" not in peaks_group:
        return None
    return np.asarray(peaks_group["peak_uid"][:], dtype="i8")


def read_fit_peak_uids_by_window(h5_group: h5py.Group) -> Dict[int, Set[int]]:
    """Cheap substitute for grouping the full loader's fitted peaks' uids by
    window.

    Equivalent to ``{wf.window_id: {p.peak_uid for p in wf.fitted_peaks if
    p.peak_uid is not None} for wf in
    load_spectrum_fit_from_hdf5(h5_group).window_fits}``, reading the
    ``peak_uid`` column and the three join columns.

    A fit predating peak identity has no ``peak_uid`` column and maps every
    window to an empty set -- the same "contributes nothing to its window's
    set" result the full loader produces, since every row's ``peak_uid``
    would decode to ``None`` and the full loader's set comprehension drops
    those.
    """
    slices = _window_slices(h5_group)
    uids = _uid_column(h5_group)
    if uids is None:
        return {wid: set() for wid, _start, _stop in slices}
    return {
        wid: {int(v) for v in uids[start:stop] if int(v) >= 0}
        for wid, start, stop in slices
    }


def read_fit_peak_freqs_and_uids_by_window(
    h5_group: h5py.Group,
) -> Tuple[Dict[int, List[float]], Dict[int, Set[int]]]:
    """Both per-window peak maps, reading the join columns once.

    Returns exactly ``(read_fit_peak_frequencies_by_window(h5_group),
    read_fit_peak_uids_by_window(h5_group))`` -- same values, same per-window
    row order, same empty-set treatment of a fit predating ``peak_uid``.

    For the caller that needs both (the curation advisory pass, when a batch
    carries a ``"uid:N"`` target). A caller wanting one of them should keep
    calling the single-column reader: this one always reads the
    ``frequency_mhz`` column, so it is not a free superset.
    """
    slices = _window_slices(h5_group)
    peaks_group = _peaks_table(h5_group)
    freqs = np.asarray(peaks_group["frequency_mhz"][:], dtype="f8")
    uids = _uid_column(h5_group)
    freqs_out: Dict[int, List[float]] = {}
    uids_out: Dict[int, Set[int]] = {}
    for wid, start, stop in slices:
        freqs_out[wid] = [float(v) for v in freqs[start:stop]]
        uids_out[wid] = (
            set() if uids is None else {int(v) for v in uids[start:stop] if int(v) >= 0}
        )
    return freqs_out, uids_out


class FitWindowCoverage(NamedTuple):
    """One window's cheap-resolution data, ordered ascending by ``window_id``
    to match :func:`load_spectrum_fit_from_hdf5`'s explicit sort of
    ``SpectrumFit.window_fits``.

    ``freq_range`` is ``None`` exactly when the full loader's
    ``FittingResult.window`` would be ``None`` (either bound NaN).
    ``peak_uids`` is the set of non-``None`` ``peak_uid`` values among the
    window's fitted peaks (empty on a fit predating peak identity).
    """

    window_id: int
    freq_range: Optional[Tuple[float, float]]
    peak_uids: Set[int]


def read_fit_window_coverage(h5_group: h5py.Group) -> List[FitWindowCoverage]:
    """Cheap substitute for the per-window coverage data
    :func:`~ftmwpipeline._internal.stage5_impl.window_covering_freq` resolves
    a curation target against.

    Equivalent to building, for each ``wf`` in
    ``load_spectrum_fit_from_hdf5(h5_group).window_fits``, the triple
    ``(wf.window_id, wf.window.freq_range if wf.window is not None else
    None, {p.peak_uid for p in wf.fitted_peaks if p.peak_uid is not None})``
    -- but reading five columns and nothing else. The result is sorted
    ascending by ``window_id``, the order those resolvers iterate in (first
    match wins), matching the full loader's own explicit sort.
    """
    windows_group = _windows_group(h5_group)
    for column in ("freq_min", "freq_max"):
        if column not in windows_group:
            raise ValueError(f"stage5_fitting windows table missing column {column!r}")
    ids = np.asarray(windows_group["window_id"][:], dtype="i8")
    freq_min = np.asarray(windows_group["freq_min"][:], dtype="f8")
    freq_max = np.asarray(windows_group["freq_max"][:], dtype="f8")
    bounds = {
        int(wid): (float(lo), float(hi)) for wid, lo, hi in zip(ids, freq_min, freq_max)
    }
    uids_by_window = read_fit_peak_uids_by_window(h5_group)

    rows: List[FitWindowCoverage] = []
    for wid, _start, _stop in _window_slices(h5_group):
        lo, hi = bounds[wid]
        freq_range = None if (np.isnan(lo) or np.isnan(hi)) else (lo, hi)
        rows.append(FitWindowCoverage(wid, freq_range, uids_by_window[wid]))
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
    if attr not in windows_group:
        raise ValueError(f"stage5_fitting windows table missing column {attr!r}")
    rows: Dict[str, List[Any]] = {c: [] for c in requested}

    ids = np.asarray(windows_group["window_id"][:], dtype="i8")
    cells = windows_group[attr][:]
    # Ascending window id, so the table stays grouped by window and ordered
    # within it -- the same order the per-window groups used to be visited in.
    for window_id, cell in sorted(zip(ids, cells), key=lambda pair: int(pair[0])):
        raw = _decode(cell)
        records = json.loads(raw) if raw else []
        for i, record in enumerate(records):
            extra: Dict[str, Any] = {"window_id": int(window_id)}
            if index_column is not None:
                extra[index_column] = i
            record_row(
                record,
                specs,
                requested,
                rows,
                where=f"window {int(window_id)} {attr}[{i}]",
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
