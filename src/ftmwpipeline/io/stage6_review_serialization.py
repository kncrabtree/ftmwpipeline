"""
Stage 6 review-state serialization to HDF5.

Persists :class:`~ftmwpipeline.core.data_structures.Stage6Review` in the
``stage6_review`` HDF5 group (created by :func:`review_run_impl`).

HDF5 layout (under the caller-supplied group)::

    .attrs:
        creation_time (ISO8601)
        n_windows     (int)
    window_statuses/ (JSON, per-window serialization)
        .attrs:
            data  (JSON string)
    decision_log/
        .attrs:
            data  (JSON string)
    final_products/ (present once consolidated; absent otherwise)
        .attrs:
            data  (JSON string)
    created_windows/ (absent in files predating Stage-6 window creation)
        .attrs:
            data  (JSON string)

The window-status data is a JSON list of dicts with keys
``window_id``, ``provenance``, ``attention_reasons``, ``invalidated``.
Each element of ``attention_reasons`` is a dict with keys
``kind``, ``detail``, ``severity``.

The decision log is a JSON list (a window carries entries once a `review` edit
records a decision against it). The final-products subgroup holds the
consolidated, frequency-calibrated line list `review run` builds. The
created-windows subgroup holds the Stage-6 overlay on the Stage 4 window plan
(new or widened windows a `create_window` decision installed) as a JSON list of
serialized ``FitWindow`` objects.

Reading a group that does not exist returns an empty :class:`Stage6Review`
(legacy-safe).
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List

import h5py

from ..core.data_structures import (
    AttentionReason,
    DecisionLogEntry,
    FinalPeak,
    FinalProducts,
    FitWindow,
    FixedContributor,
    Stage6Review,
    WindowReviewStatus,
)
from ._hdf5_helpers import opt_float

__all__ = [
    "save_stage6_review_to_hdf5",
    "load_stage6_review_from_hdf5",
    "load_stage6_review_from_file",
]


def _status_to_dict(status: WindowReviewStatus) -> Dict[str, Any]:
    return {
        "window_id": status.window_id,
        "provenance": status.provenance,
        "attention_reasons": [
            {
                "kind": r.kind,
                "detail": r.detail,
                "severity": r.severity,
                "locations": list(r.locations),
            }
            for r in status.attention_reasons
        ],
        "invalidated": status.invalidated,
    }


def _status_from_dict(d: Dict[str, Any]) -> WindowReviewStatus:
    reasons = [
        AttentionReason(
            kind=str(r["kind"]),
            detail=str(r["detail"]),
            severity=float(r["severity"]),
            locations=[float(x) for x in r.get("locations", [])],
        )
        for r in d.get("attention_reasons", [])
    ]
    return WindowReviewStatus(
        window_id=int(d["window_id"]),
        provenance=str(d.get("provenance", "auto")),
        attention_reasons=reasons,
        invalidated=bool(d.get("invalidated", False)),
    )


def _entry_to_dict(entry: DecisionLogEntry) -> Dict[str, Any]:
    return {
        "order_index": entry.order_index,
        "window_id": entry.window_id,
        "frequency_mhz": entry.frequency_mhz,
        "kind": entry.kind,
        "provenance": entry.provenance,
        "evidence": entry.evidence,
    }


def _entry_from_dict(d: Dict[str, Any]) -> DecisionLogEntry:
    return DecisionLogEntry(
        order_index=int(d["order_index"]),
        window_id=int(d["window_id"]),
        frequency_mhz=float(d["frequency_mhz"]),
        kind=str(d["kind"]),
        provenance=str(d.get("provenance", "user")),
        evidence=dict(d.get("evidence", {})),
    )


def _final_peak_to_dict(p: FinalPeak) -> Dict[str, Any]:
    return {
        "frequency_mhz": p.frequency_mhz,
        "frequency_raw_mhz": p.frequency_raw_mhz,
        "f_baseband_mhz": p.f_baseband_mhz,
        "sigma_f_khz": p.sigma_f_khz,
        "sigma_stat_khz": p.sigma_stat_khz,
        "sigma_eps_khz": p.sigma_eps_khz,
        "sigma_floor_khz": p.sigma_floor_khz,
        "amplitude": p.amplitude,
        "phase": p.phase,
        "snr": p.snr,
        "origin": p.origin,
        "window_id": p.window_id,
        "amplitude_error": p.amplitude_error,
        "phase_error": p.phase_error,
        "snr_error": p.snr_error,
        "clock_lattice": p.clock_lattice,
        "derivation": p.derivation,
        "peak_uid": p.peak_uid,
        "knockout_p_value": p.knockout_p_value,
        "knockout_supported": p.knockout_supported,
        "knockout_aicc_delta": p.knockout_aicc_delta,
    }


def _final_peak_from_dict(d: Dict[str, Any]) -> FinalPeak:
    return FinalPeak(
        frequency_mhz=float(d["frequency_mhz"]),
        frequency_raw_mhz=float(d["frequency_raw_mhz"]),
        f_baseband_mhz=float(d["f_baseband_mhz"]),
        sigma_f_khz=float(d["sigma_f_khz"]),
        sigma_stat_khz=float(d["sigma_stat_khz"]),
        sigma_eps_khz=float(d["sigma_eps_khz"]),
        sigma_floor_khz=float(d["sigma_floor_khz"]),
        amplitude=float(d["amplitude"]),
        phase=opt_float(d, "phase"),
        snr=opt_float(d, "snr"),
        origin=str(d.get("origin", "auto")),
        window_id=None if d.get("window_id") is None else int(d["window_id"]),
        amplitude_error=opt_float(d, "amplitude_error"),
        phase_error=opt_float(d, "phase_error"),
        snr_error=opt_float(d, "snr_error"),
        clock_lattice=(
            None if d.get("clock_lattice") is None else str(d["clock_lattice"])
        ),
        # Absent in tables written before the derivation tag; None reads as
        # "carried through unchanged", correct for a pre-curation table.
        derivation=(None if d.get("derivation") is None else int(d["derivation"])),
        # Absent in tables written before peak identity; None is the honest
        # value for a pre-existing table -- never backfilled.
        peak_uid=(None if d.get("peak_uid") is None else int(d["peak_uid"])),
        # Absent in tables written before the knockout statistics were carried
        # to the final table. None reads as "no knockout result", which is the
        # honest value for such a table -- and stays distinct from the nan the
        # test itself writes when its refit did not converge (nan survives the
        # JSON round-trip; json.dumps/loads handle it natively).
        knockout_p_value=opt_float(d, "knockout_p_value"),
        knockout_supported=(
            None
            if d.get("knockout_supported") is None
            else bool(d["knockout_supported"])
        ),
        knockout_aicc_delta=opt_float(d, "knockout_aicc_delta"),
    )


def _final_products_to_dict(fp: FinalProducts) -> Dict[str, Any]:
    return {
        "calibration_state": fp.calibration_state,
        "epsilon": fp.epsilon,
        "sigma_epsilon": fp.sigma_epsilon,
        "sigma_floor_khz": fp.sigma_floor_khz,
        "probe_freq_mhz": fp.probe_freq_mhz,
        "sideband": fp.sideband,
        "peaks": [_final_peak_to_dict(p) for p in fp.peaks],
    }


def _final_products_from_dict(d: Dict[str, Any]) -> FinalProducts:
    return FinalProducts(
        peaks=[_final_peak_from_dict(p) for p in d.get("peaks", [])],
        calibration_state=str(d.get("calibration_state", "rb_locked")),
        epsilon=float(d.get("epsilon", 0.0)),
        sigma_epsilon=float(d.get("sigma_epsilon", 0.0)),
        sigma_floor_khz=float(d.get("sigma_floor_khz", 0.0)),
        probe_freq_mhz=float(d.get("probe_freq_mhz", 0.0)),
        sideband=str(d.get("sideband", "upper")),
    )


def _fit_window_to_dict(w: FitWindow) -> Dict[str, Any]:
    return {
        "window_id": int(w.window_id),
        "freq_range": [float(w.freq_range[0]), float(w.freq_range[1])],
        "free_peak_indices": [int(i) for i in w.free_peak_indices],
        "batch": int(w.batch),
        "fixed_contributors": [
            {
                "peak_index": int(fc.peak_index),
                "primary_window_id": int(fc.primary_window_id),
                "frequency_mhz": float(fc.frequency_mhz),
                "freeze_eligible": bool(fc.freeze_eligible),
                "edge_free": bool(fc.edge_free),
            }
            for fc in w.fixed_contributors
        ],
        "diagnostics": w.diagnostics,
    }


def _fit_window_from_dict(d: Dict[str, Any]) -> FitWindow:
    fr = d["freq_range"]
    return FitWindow(
        window_id=int(d["window_id"]),
        freq_range=(float(fr[0]), float(fr[1])),
        free_peak_indices=[int(i) for i in d.get("free_peak_indices", [])],
        fixed_contributors=[
            FixedContributor(
                peak_index=int(fc["peak_index"]),
                primary_window_id=int(fc["primary_window_id"]),
                frequency_mhz=float(fc["frequency_mhz"]),
                freeze_eligible=bool(fc.get("freeze_eligible", True)),
                edge_free=bool(fc.get("edge_free", False)),
            )
            for fc in d.get("fixed_contributors", [])
        ],
        batch=int(d.get("batch", 0)),
        diagnostics=dict(d.get("diagnostics", {})),
    )


def save_stage6_review_to_hdf5(review: Stage6Review, group: h5py.Group) -> None:
    """Persist *review* into the HDF5 *group* (must already exist)."""
    group.attrs["creation_time"] = datetime.now().isoformat()
    group.attrs["n_windows"] = len(review.window_statuses)

    statuses_list = [_status_to_dict(s) for s in review.window_statuses.values()]
    ws_grp = group.require_group("window_statuses")
    ws_grp.attrs["data"] = json.dumps(statuses_list)

    log_list = [_entry_to_dict(e) for e in review.decision_log]
    dl_grp = group.require_group("decision_log")
    dl_grp.attrs["data"] = json.dumps(log_list)

    if review.final_products is not None:
        fp_grp = group.require_group("final_products")
        fp_grp.attrs["data"] = json.dumps(
            _final_products_to_dict(review.final_products)
        )

    cw_grp = group.require_group("created_windows")
    cw_grp.attrs["data"] = json.dumps(
        [_fit_window_to_dict(w) for w in review.created_windows], default=str
    )


def load_stage6_review_from_hdf5(group: h5py.Group) -> Stage6Review:
    """Load a :class:`Stage6Review` from *group*."""
    ws_grp = group.get("window_statuses")
    window_statuses: Dict[int, WindowReviewStatus] = {}
    if ws_grp is not None:
        raw = ws_grp.attrs.get("data", "[]")
        for d in json.loads(str(raw)):
            s = _status_from_dict(d)
            window_statuses[s.window_id] = s

    dl_grp = group.get("decision_log")
    decision_log: List[DecisionLogEntry] = []
    if dl_grp is not None:
        raw = dl_grp.attrs.get("data", "[]")
        for d in json.loads(str(raw)):
            decision_log.append(_entry_from_dict(d))

    fp_grp = group.get("final_products")
    final_products = None
    if fp_grp is not None:
        raw = fp_grp.attrs.get("data")
        if raw is not None:
            final_products = _final_products_from_dict(json.loads(str(raw)))

    # Absent in files written before Stage-6 window creation: an empty overlay
    # means "the effective plan is the Stage 4 plan", which is correct for them.
    cw_grp = group.get("created_windows")
    created_windows: List[FitWindow] = []
    if cw_grp is not None:
        raw = cw_grp.attrs.get("data", "[]")
        for d in json.loads(str(raw)):
            created_windows.append(_fit_window_from_dict(d))

    return Stage6Review(
        window_statuses=window_statuses,
        decision_log=decision_log,
        final_products=final_products,
        created_windows=created_windows,
    )


def load_stage6_review_from_file(file_path: str) -> Stage6Review:
    """Load :class:`Stage6Review` from a ``.ftmw`` file, or return empty.

    An unreadable / non-HDF5 file and a file without a ``stage6_review`` group
    both return an empty review (legacy-safe). A group that *is* present but
    fails to decode is a corrupt record, not an absent one: its error
    propagates rather than silently discarding the user's curation.
    """
    try:
        h5f = h5py.File(file_path, "r")
    except OSError:
        return Stage6Review()
    with h5f:
        if "stage6_review" not in h5f:
            return Stage6Review()
        return load_stage6_review_from_hdf5(h5f["stage6_review"])
