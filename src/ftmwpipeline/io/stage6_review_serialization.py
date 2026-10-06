"""
Stage 6 review-state serialization to HDF5.

Persists :class:`~ftmwpipeline.core.data_structures.Stage6Review` in the
``stage6_review`` HDF5 group (created by :func:`review_run_impl`).

HDF5 layout (under the caller-supplied group)::

    .attrs:
        creation_time (ISO8601)
        n_windows     (int)
        next_serial   (int; the serial the next decision takes -- a high-water
                       mark an undo never lowers; absent before serials, read
                       as 0)
        engine_version (int; the replay-engine version that wrote the review,
                       ``core.data_structures.ENGINE_VERSION``; absent on a
                       review a pre-engine build wrote)
        review_params (JSON object: ``bar``, ``attention_candidate_evidence``,
                       ``kappa``, ``noise_floor`` -- the attention-routing
                       parameters the statuses were computed under, recorded
                       by ``review run``; absent until a Stage 6 write records
                       them, read as ``None``)
    window_statuses/ (JSON, per-window serialization)
        .attrs:
            data  (JSON string)
    decision_log/
        .attrs:
            data  (JSON string)
    final_products/ (present once consolidated; absent otherwise)
        .attrs:
            data          (JSON string)
            peak_fields   (int; absent on tables that predate the per-line
                           fit fields -- see FINAL_PEAK_FIELDS_VERSION)
            calibration_clocks (JSON list of ClockSource dicts; the clock
                           declaration the table's calibration state was
                           derived from; absent on tables written before it
                           was recorded)
    created_windows/ (absent in files predating Stage-6 window creation)
        .attrs:
            data  (JSON string)

The window-status data is a JSON list of dicts with keys
``window_id``, ``provenance``, ``attention_reasons``, ``invalidated``.
Each element of ``attention_reasons`` is a dict with keys
``kind``, ``detail``, ``severity``, ``locations`` and ``evidence`` (a JSON
object; absent in records written before it existed, read as empty).

The decision log is a JSON list (a window carries entries once a `review` edit
records a decision against it). Each row carries its ``serial`` (absent on a
row a pre-engine build recorded, read as ``Absent.NOT_RUN``). The final-products subgroup holds the
consolidated, frequency-calibrated line list `review run` builds. Its per-line
fit fields (``decay_time_us``, ``decay_time_error_us``, ``shape``,
``fwhm_mhz``, ``detection_index``, ``fit_window_mhz``) can be
:class:`~ftmwpipeline.contract.Absent`; each is stored losslessly as its value
(``null`` when absent; ``fit_window_mhz`` as a two-element list) plus a sibling
``"<field>__status"`` integer using the columnar status codes (``0`` present,
``1`` not run, ``2`` undefined). A table written before those fields existed
carries no ``peak_fields`` stamp; :func:`final_products_predate_fit_fields`
detects it so a reader can rebuild the table in memory. The pre-contract fields
that can be absent (:data:`FINAL_PEAK_ABSENT_FIELDS`: the statistical and total
sigma, phase, SNR, the errors, ``window_id``, ``clock_lattice``,
``derivation``, ``peak_uid`` and the knockout fields) are stored the same way.
A record written before their status keys existed is decoded field by field
(:func:`_legacy_absent_field`), so it needs no rebuild. The
created-windows subgroup holds the Stage-6 overlay on the Stage 4 window plan
(new or widened windows a `create_window` decision installed) as a JSON list of
serialized ``FitWindow`` objects.

Reading a group that does not exist returns an empty :class:`Stage6Review`
(legacy-safe).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import h5py

from .._internal.absence_rules import (
    clock_lattice_or_absent,
    float_or_absent,
    int_or_absent,
    knockout_absence,
)
from .._internal.atomic import h5open
from ..core.absent import STATUS_PRESENT, Absent
from ..core.data_structures import (
    AttentionReason,
    DecisionLogEntry,
    FinalPeak,
    FinalProducts,
    FitWindow,
    FixedContributor,
    ReviewParams,
    Stage6Review,
    WindowReviewStatus,
)
from ..core.stage_fit_settings import ClockSource, coerce_clock_sources

__all__ = [
    "save_stage6_review_to_hdf5",
    "load_stage6_review_from_hdf5",
    "load_stage6_review_from_file",
    "read_created_window_bounds",
    "final_products_predate_fit_fields",
    "read_final_products_calibration_clocks",
    "fit_declares_clocks",
    "FINAL_PEAK_FIELDS_VERSION",
]

#: Attribute of the ``final_products`` subgroup recording the clock declaration
#: the table's calibration state was derived from.
_CALIBRATION_CLOCKS_ATTR = "calibration_clocks"

#: Version of the per-line field set a stored final-products table carries,
#: stamped as the ``peak_fields`` attribute of its subgroup. ``2`` added the
#: Stage 5 fit fields (:data:`FINAL_PEAK_FIT_FIELDS`); a table without the
#: stamp predates them.
FINAL_PEAK_FIELDS_VERSION: int = 2

#: The ``FinalPeak`` fields joined from the Stage 5 fit, each of which can be
#: ``Absent`` and is stored as a value plus a ``"<field>__status"`` code.
FINAL_PEAK_FIT_FIELDS: Tuple[str, ...] = (
    "decay_time_us",
    "decay_time_error_us",
    "shape",
    "fwhm_mhz",
    "detection_index",
    "fit_window_mhz",
)

#: The pre-contract ``FinalPeak`` fields that can be ``Absent``, each stored as
#: a value plus a ``"<field>__status"`` code like the fit fields. Paired with
#: the type a present value decodes to.
FINAL_PEAK_ABSENT_FIELDS: Tuple[Tuple[str, type], ...] = (
    ("sigma_f_khz", float),
    ("sigma_stat_khz", float),
    ("phase", float),
    ("snr", float),
    ("window_id", int),
    ("amplitude_error", float),
    ("phase_error", float),
    ("snr_error", float),
    ("clock_lattice", str),
    ("derivation", int),
    ("peak_uid", int),
    ("knockout_p_value", float),
    ("knockout_supported", bool),
    ("knockout_aicc_delta", float),
)

_STATUS_KEY_SUFFIX = "__status"


def _encode_absent(value: Any) -> Any:
    """Store an attention reason's evidence as JSON: an :class:`Absent` value
    of a key ``k`` becomes ``null`` with a sibling ``"k__status"`` code (the
    columnar codes), recursively through dicts and lists."""
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            if isinstance(v, Absent):
                out[k] = None
                out[f"{k}{_STATUS_KEY_SUFFIX}"] = v.status
            else:
                out[k] = _encode_absent(v)
        return out
    if isinstance(value, list):
        return [_encode_absent(v) for v in value]
    return value


def _decode_absent(value: Any) -> Any:
    """Inverse of :func:`_encode_absent`."""
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            if k.endswith(_STATUS_KEY_SUFFIX):
                continue
            code = value.get(f"{k}{_STATUS_KEY_SUFFIX}")
            out[k] = Absent.from_status(int(code)) if code else _decode_absent(v)
        return out
    if isinstance(value, list):
        return [_decode_absent(v) for v in value]
    return value


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
                "evidence": _encode_absent(dict(r.evidence)),
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
            evidence=_decode_absent(dict(r.get("evidence", {}))),
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
    out: Dict[str, Any] = {
        "order_index": entry.order_index,
        "window_id": entry.window_id,
        "frequency_mhz": entry.frequency_mhz,
        "kind": entry.kind,
        "provenance": entry.provenance,
        "evidence": entry.evidence,
    }
    if not isinstance(entry.serial, Absent):
        out["serial"] = int(entry.serial)
    return out


def _entry_from_dict(d: Dict[str, Any]) -> DecisionLogEntry:
    return DecisionLogEntry(
        order_index=int(d["order_index"]),
        window_id=int(d["window_id"]),
        frequency_mhz=float(d["frequency_mhz"]),
        kind=str(d["kind"]),
        provenance=str(d.get("provenance", "user")),
        evidence=dict(d.get("evidence", {})),
        serial=int(d["serial"]) if d.get("serial") is not None else Absent.NOT_RUN,
    )


def _final_peak_to_dict(p: FinalPeak) -> Dict[str, Any]:
    return {
        "frequency_mhz": p.frequency_mhz,
        "frequency_raw_mhz": p.frequency_raw_mhz,
        "f_baseband_mhz": p.f_baseband_mhz,
        "sigma_eps_khz": p.sigma_eps_khz,
        "sigma_floor_khz": p.sigma_floor_khz,
        "amplitude": p.amplitude,
        "origin": p.origin,
        **_status_fields_to_dict(p, [name for name, _ in FINAL_PEAK_ABSENT_FIELDS]),
        **_fit_fields_to_dict(p),
    }


def _status_fields_to_dict(p: FinalPeak, names: List[str]) -> Dict[str, Any]:
    """Fields of *p* that can be ``Absent`` as ``value`` + ``"<field>__status"``
    pairs (``null`` value when absent)."""
    out: Dict[str, Any] = {}
    for name in names:
        value = getattr(p, name)
        if isinstance(value, Absent):
            out[name] = None
            out[name + _STATUS_KEY_SUFFIX] = value.status
        else:
            out[name] = list(value) if isinstance(value, tuple) else value
            out[name + _STATUS_KEY_SUFFIX] = STATUS_PRESENT
    return out


def _fit_fields_to_dict(p: FinalPeak) -> Dict[str, Any]:
    """The per-line fit fields as ``value`` + ``"<field>__status"`` pairs."""
    return _status_fields_to_dict(p, list(FINAL_PEAK_FIT_FIELDS))


def _fit_fields_from_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Decode :func:`_fit_fields_to_dict`. A field the record does not carry
    (a table predating the fields) reads as ``Absent.UNDEFINED``; such a
    table is rebuilt by its reader (see
    :func:`final_products_predate_fit_fields`)."""
    out: Dict[str, Any] = {}
    for name in FINAL_PEAK_FIT_FIELDS:
        status = d.get(name + _STATUS_KEY_SUFFIX)
        value: Any = d.get(name)
        if status is None or (int(status) == STATUS_PRESENT and value is None):
            out[name] = Absent.UNDEFINED
        elif int(status) != STATUS_PRESENT:
            out[name] = Absent.from_status(int(status))
        elif name == "fit_window_mhz":
            lo, hi = value
            out[name] = (float(lo), float(hi))
        elif name == "shape":
            out[name] = str(value)
        elif name == "detection_index":
            out[name] = int(value)
        else:
            out[name] = float(value)
    return out


def _legacy_absent_field(
    name: str, d: Dict[str, Any], clocks_declared: Callable[[], bool]
) -> Any:
    """Decode field *name* of a record written before its status key existed.

    The record stored ``None`` (or a non-finite float) for absence; the rule is
    per field (``dev-docs/CONTRACT_STRATEGY.md`` §Missing values):

    - ``window_id``, ``derivation``, ``peak_uid``: ``None`` is ``NOT_RUN``.
    - ``phase``, ``snr`` and the errors: ``None`` or non-finite is
      ``UNDEFINED``.
    - ``clock_lattice``: ``None`` is ``NOT_RUN`` when the file's Stage 5 fit
      recorded no clock declaration, ``UNDEFINED`` (off-lattice) when it did.
    - knockout fields: no knockout record (``knockout_supported`` ``None``) is
      ``NOT_RUN`` for all three; otherwise a non-finite value is ``UNDEFINED``.
    - ``sigma_stat_khz``: ``0.0`` is the sentinel these records wrote for a
      line with no frequency error, so it is ``UNDEFINED`` (a covariance error
      is never exactly zero), and ``sigma_f_khz`` is then ``UNDEFINED`` too: a
      total missing its statistical term is not an honest uncertainty.
    """
    value = d.get(name)
    if name in ("window_id", "derivation", "peak_uid"):
        return int_or_absent(value, sentinel=None)
    if name == "clock_lattice":
        return clock_lattice_or_absent(value, declared=clocks_declared())
    if name.startswith("knockout_"):
        not_run = knockout_absence(d.get("knockout_supported"), None)
        if not_run is not None:
            return not_run
        if name == "knockout_supported":
            return bool(value)
        return float_or_absent(value)
    if name in ("sigma_stat_khz", "sigma_f_khz"):
        stat = d.get("sigma_stat_khz")
        if stat is None or not math.isfinite(float(stat)) or float(stat) == 0.0:
            return Absent.UNDEFINED
        return float_or_absent(value)
    return float_or_absent(value)


def _absent_fields_from_dict(
    d: Dict[str, Any], clocks_declared: Callable[[], bool]
) -> Dict[str, Any]:
    """Decode the :data:`FINAL_PEAK_ABSENT_FIELDS` of a stored record."""
    out: Dict[str, Any] = {}
    for name, kind in FINAL_PEAK_ABSENT_FIELDS:
        status = d.get(name + _STATUS_KEY_SUFFIX)
        value = d.get(name)
        if status is None:
            out[name] = _legacy_absent_field(name, d, clocks_declared)
        elif int(status) != STATUS_PRESENT:
            out[name] = Absent.from_status(int(status))
        elif value is None:
            out[name] = Absent.UNDEFINED
        else:
            out[name] = kind(value)
    return out


def _final_peak_from_dict(
    d: Dict[str, Any], clocks_declared: Callable[[], bool] = lambda: False
) -> FinalPeak:
    """Decode one stored ``FinalPeak``. *clocks_declared* is asked (once a
    record needs it) whether the file's Stage 5 fit recorded a clock
    declaration, to decode a pre-status ``clock_lattice``."""
    return FinalPeak(
        frequency_mhz=float(d["frequency_mhz"]),
        frequency_raw_mhz=float(d["frequency_raw_mhz"]),
        f_baseband_mhz=float(d["f_baseband_mhz"]),
        sigma_eps_khz=float(d["sigma_eps_khz"]),
        sigma_floor_khz=float(d["sigma_floor_khz"]),
        amplitude=float(d["amplitude"]),
        origin=str(d.get("origin", "auto")),
        **_absent_fields_from_dict(d, clocks_declared),
        **_fit_fields_from_dict(d),
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


def _final_products_from_dict(
    d: Dict[str, Any], clocks_declared: Callable[[], bool] = lambda: False
) -> FinalProducts:
    return FinalProducts(
        peaks=[_final_peak_from_dict(p, clocks_declared) for p in d.get("peaks", [])],
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


def save_stage6_review_to_hdf5(
    review: Stage6Review,
    group: h5py.Group,
    *,
    calibration_clocks: Optional[Tuple[ClockSource, ...]] = None,
) -> None:
    """Persist *review* into the HDF5 *group* (must already exist).

    ``calibration_clocks`` is the clock declaration the final-products table's
    calibration state was derived from; it is recorded beside the table when
    both are given (the Stage 6 writers always pass it with a table).
    """
    group.attrs["creation_time"] = datetime.now().isoformat()
    group.attrs["n_windows"] = len(review.window_statuses)
    group.attrs["next_serial"] = int(review.next_serial)
    if review.engine_version is not None:
        group.attrs["engine_version"] = int(review.engine_version)
    elif "engine_version" in group.attrs:
        del group.attrs["engine_version"]
    if review.review_params is not None:
        group.attrs["review_params"] = json.dumps(asdict(review.review_params))
    elif "review_params" in group.attrs:
        del group.attrs["review_params"]

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
        fp_grp.attrs["peak_fields"] = FINAL_PEAK_FIELDS_VERSION
        if calibration_clocks is not None:
            fp_grp.attrs[_CALIBRATION_CLOCKS_ATTR] = json.dumps(
                [c.to_dict() for c in calibration_clocks]
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
            declared: List[bool] = []

            def clocks_declared() -> bool:
                # Read once, and only when a pre-status record needs it.
                if not declared:
                    declared.append(fit_declares_clocks(group.file))
                return declared[0]

            final_products = _final_products_from_dict(
                json.loads(str(raw)), clocks_declared
            )

    # Absent in files written before Stage-6 window creation: an empty overlay
    # means "the effective plan is the Stage 4 plan", which is correct for them.
    cw_grp = group.get("created_windows")
    created_windows: List[FitWindow] = []
    if cw_grp is not None:
        raw = cw_grp.attrs.get("data", "[]")
        for d in json.loads(str(raw)):
            created_windows.append(_fit_window_from_dict(d))

    engine_version = group.attrs.get("engine_version")
    raw_params = group.attrs.get("review_params")
    review_params = (
        None
        if raw_params is None
        else ReviewParams(
            **{k: float(v) for k, v in json.loads(str(raw_params)).items()}
        )
    )
    return Stage6Review(
        window_statuses=window_statuses,
        decision_log=decision_log,
        final_products=final_products,
        created_windows=created_windows,
        next_serial=int(group.attrs.get("next_serial", 0)),
        engine_version=None if engine_version is None else int(engine_version),
        review_params=review_params,
    )


def read_created_window_bounds(
    group: Optional[h5py.Group],
) -> List[Tuple[int, float, float]]:
    """``(window_id, freq_min, freq_max)`` of each Stage-6-created window.

    Reads only the ``created_windows`` record of a ``stage6_review`` *group*
    (``None``, or a group without the record, yields an empty list -- files
    predating Stage-6 window creation). Bounds are ordered ``min, max`` whatever
    order they were stored in. A record that is present but undecodable raises,
    like :func:`load_stage6_review_from_hdf5`.
    """
    if group is None:
        return []
    cw_grp = group.get("created_windows")
    if cw_grp is None:
        return []
    raw = cw_grp.attrs.get("data", "[]")
    out: List[Tuple[int, float, float]] = []
    for d in json.loads(str(raw)):
        lo, hi = (float(v) for v in d["freq_range"])
        out.append((int(d["window_id"]), min(lo, hi), max(lo, hi)))
    return out


def read_final_products_calibration_clocks(
    group: Optional[h5py.Group],
) -> Optional[Tuple[ClockSource, ...]]:
    """The clock declaration the stored final-products table was derived under.

    *group* is a ``stage6_review`` group (or ``None``). ``None`` when no table
    is stored or the table was written before the declaration was recorded;
    an empty tuple when it was derived with no declaration. Never writes.
    """
    if group is None:
        return None
    fp_grp = group.get("final_products")
    if fp_grp is None or fp_grp.attrs.get("data") is None:
        return None
    raw = fp_grp.attrs.get(_CALIBRATION_CLOCKS_ATTR)
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return coerce_clock_sources(str(raw))


def fit_declares_clocks(h5f: h5py.File) -> bool:
    """Whether the file's Stage 5 fit recorded a non-empty clock declaration.

    Reads the persisted ``spur.clocks`` of the fit settings record -- the
    declaration the fit's clock-lattice annotation ran with. ``False`` when
    there is no record, no declaration, or one that does not decode. Never
    writes.
    """
    from .stage_fit_settings_serialization import STAGE_FIT_PATH

    spur = h5f.get(STAGE_FIT_PATH + "/spur")
    if spur is None:
        return False
    raw = spur.attrs.get("clocks")
    if raw is None:
        return False
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        clocks = coerce_clock_sources(str(raw))
    except ValueError:
        return False
    return bool(clocks)


def final_products_predate_fit_fields(group: Optional[h5py.Group]) -> bool:
    """Whether the stored final-products table predates the per-line fit fields.

    *group* is a ``stage6_review`` group (or ``None``). ``True`` only when a
    table is stored and its subgroup carries no ``peak_fields`` stamp (or an
    older one); ``False`` when no table is stored -- there is nothing to have
    predated anything. Reads one attribute; never writes.
    """
    if group is None:
        return False
    fp_grp = group.get("final_products")
    if fp_grp is None or fp_grp.attrs.get("data") is None:
        return False
    stamp = fp_grp.attrs.get("peak_fields")
    return stamp is None or int(stamp) < FINAL_PEAK_FIELDS_VERSION


def load_stage6_review_from_file(file_path: str) -> Stage6Review:
    """Load :class:`Stage6Review` from a ``.ftmw`` file, or return empty.

    An unreadable / non-HDF5 file and a file without a ``stage6_review`` group
    both return an empty review (legacy-safe). A group that *is* present but
    fails to decode is a corrupt record, not an absent one: its error
    propagates rather than silently discarding the user's curation.
    """
    try:
        h5f = h5open(file_path, "r")
    except OSError:
        return Stage6Review()
    with h5f:
        if "stage6_review" not in h5f:
            return Stage6Review()
        return load_stage6_review_from_hdf5(h5f["stage6_review"])
