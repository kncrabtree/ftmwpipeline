"""
Stage 6 review-state serialization to HDF5.

Persists :class:`~ftmwpipeline.core.data_structures.Stage6Review` in the
``stage6_review`` HDF5 group (created by :func:`review_run_impl`).

HDF5 layout (under the caller-supplied group)::

    .attrs:
        creation_time (ISO8601)
        n_windows     (int)
    window_statuses/ (JSON, per-window serialisation)
        .attrs:
            data  (JSON string)
    decision_log/
        .attrs:
            data  (JSON string)

The window-status data is a JSON list of dicts with keys
``window_id``, ``provenance``, ``attention_reasons``, ``invalidated``.
Each element of ``attention_reasons`` is a dict with keys
``kind``, ``detail``, ``severity``.

The decision log is a JSON list (empty until Pass 2 verbs are invoked).

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
    Stage6Review,
    WindowReviewStatus,
)

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
            {"kind": r.kind, "detail": r.detail, "severity": r.severity}
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
    }


def _opt_float(d: Dict[str, Any], key: str) -> Any:
    v = d.get(key)
    return None if v is None else float(v)


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
        phase=_opt_float(d, "phase"),
        snr=_opt_float(d, "snr"),
        origin=str(d.get("origin", "auto")),
        window_id=None if d.get("window_id") is None else int(d["window_id"]),
        amplitude_error=_opt_float(d, "amplitude_error"),
        phase_error=_opt_float(d, "phase_error"),
        snr_error=_opt_float(d, "snr_error"),
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

    return Stage6Review(
        window_statuses=window_statuses,
        decision_log=decision_log,
        final_products=final_products,
    )


def load_stage6_review_from_file(file_path: str) -> Stage6Review:
    """Load :class:`Stage6Review` from a ``.ftmw`` file, or return empty."""
    try:
        with h5py.File(file_path, "r") as h5f:
            if "stage6_review" not in h5f:
                return Stage6Review()
            return load_stage6_review_from_hdf5(h5f["stage6_review"])
    except Exception:
        return Stage6Review()
