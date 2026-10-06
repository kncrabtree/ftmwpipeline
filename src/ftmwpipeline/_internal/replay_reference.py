"""
The full-replay reference for Stage 6 curation, and its bitwise state digest.

:func:`replay_full` computes the curated state a decision log describes, from
scratch and in memory: the automatic-fit baseline, the file's static analysis
inputs and the log, nothing else. It persists nothing, keeps no cache and reads
no curated state (no ``/stage5_fitting`` curated fit, no ``/stage6_review``).
It is the computation the write path runs whenever a write refits
(:func:`~.stage6_impl._reference`: the replay of the log in one batch, the one
combined cascade, the statuses and the final-products build), called with
nothing persisted, so it is the oracle a write is checked against: after every
write, the persisted state's :func:`persisted_state_digest` must equal
:func:`state_digest` of :func:`replay_full` of the log the write left, under
the review parameters it recorded.

Because a refitting write *is* this computation, the check of such a write
pins what lies around it -- that the write reads no curated state, that it is
deterministic, and that the persist round-trips the state bit for bit -- not
the replay's semantics: a change to the replay or the cascade moves both
sides alike. The check is independent for a write that refits nothing (the
persisted fits kept, or the automatic fit restored by copy), and for any
write path that stops recomputing the whole log. The replay's semantics are
pinned by the scenario tests.

:func:`state_digest` covers what Stage 6 persists of a curated state: the fit
tables exactly as :func:`~ftmwpipeline.io.fitting_serialization.save_spectrum_fit_to_hdf5`
writes them (window and peak columns and the covariance, byte for byte, plus
the fit header's JSON records), and the review's window statuses, decision
log, final products and created-window overlay as their stored JSON. Write
stamps (``creation_time``) and bookkeeping attrs that are not part of the state
(the serial and window-id high-water marks, the engine version, the baseline's
lineage id) are left out.
:func:`persisted_state_digest` reads the same parts straight from a file, so the
two compare the bytes a write put on disk with the bytes a fresh write of the
reference would.

Not part of any public interface: a test and development oracle.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import h5py
import numpy as np

from ..core.data_structures import (
    DecisionLogEntry,
    ReviewParams,
    SpectrumFit,
    Stage6Review,
)
from ..io.fitting_serialization import (
    _fit_tables,
    _replan_info_to_json,
    _rescue_round_to_json,
    _thaw_info_to_json,
)
from ..io.stage6_review_serialization import (
    _entry_to_dict,
    _final_products_to_dict,
    _fit_window_to_dict,
    _status_to_dict,
)
from .atomic import h5open
from .fingerprint_impl import canonical_json
from .stage6_impl import (
    _FIT_EDIT_KINDS,
    DEFAULT_REVIEW_PARAMS,
    STAGE5_BASELINE_GROUP,
    _missing_baseline_error,
    _reference,
    refit_snap_tol_mhz_impl,
)

__all__ = [
    "CuratedState",
    "ReviewParams",
    "differing_parts",
    "persisted_state_digest",
    "replay_full",
    "state_digest",
]

#: The fit group's header records, in the order the writer stamps them.
_FIT_HEADER_ATTRS = (
    "n_windows",
    "n_fitted_peaks",
    "final_plan_revision",
    "parameters",
    "diagnostics",
    "thaw_history",
    "replan_history",
    "rescue_history",
)

#: The header records stored as integers.
_FIT_HEADER_INTS = ("n_windows", "n_fitted_peaks", "final_plan_revision")

#: The review records that make up the curated state, by subgroup name.
_REVIEW_RECORDS = (
    "window_statuses",
    "decision_log",
    "final_products",
    "created_windows",
)


@dataclass
class CuratedState:
    """A curated state held in memory: the window fits and the review."""

    spectrum_fit: SpectrumFit
    review: Stage6Review


def replay_full(
    file_path: Union[str, Path],
    log: Sequence[DecisionLogEntry],
    params: Optional[ReviewParams] = None,
) -> CuratedState:
    """The curated state *log* describes under *params*, from the automatic
    fit, in memory (:func:`~.stage6_impl._reference`).

    A log with fit-changing rows is replayed in one batch from the
    automatic-fit baseline: its rows applied as recorded, by peak identity and
    in log order (:func:`~.stage6_impl._resolve_replay_rows`, every refusal
    raised before any fit; one action per recorded user action), one combined
    cascade; a log without one is the automatic fit itself. Every window's
    status is then computed from scratch from the final fits, the log and
    *params* (:func:`~.stage6_impl._curated_statuses`), and the final products
    from the final fits. Reads the baseline and the file's static inputs
    only. *params* defaults to ``review run``'s defaults
    (:data:`~.stage6_impl.DEFAULT_REVIEW_PARAMS`); a write's check passes the
    parameters the file records. Every row is kept verbatim.

    A log with fit-changing rows needs the baseline; a file no write has
    curated (no baseline) has only an empty or accept-only log, whose state
    is ``/stage5_fitting``. Otherwise the file is corrupt
    (:class:`~ftmwpipeline.file_manager.PipelineCorruptionError`), as it is for
    every write.
    """
    path = str(file_path)
    log = list(log)
    with h5open(path, "r") as h5f:
        baseline = STAGE5_BASELINE_GROUP in h5f
    if any(e.kind in _FIT_EDIT_KINDS for e in log) and not baseline:
        raise _missing_baseline_error(path)
    curated = _reference(
        path,
        log,
        params or DEFAULT_REVIEW_PARAMS,
        recorded=len(log),
        shared=None,
        snap_tol_mhz=refit_snap_tol_mhz_impl(path),
    )
    return CuratedState(spectrum_fit=curated.spectrum_fit, review=curated.review)


# ---------------------------------------------------------------------------
# The bitwise digest
# ---------------------------------------------------------------------------


def _column_part(data: np.ndarray) -> Any:
    """One stored column, exactly: a string column as its values, any other
    as its dtype and the SHA-256 of its bytes."""
    if data.dtype == object:
        return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in data]
    arr = np.ascontiguousarray(data)
    return [arr.dtype.str, hashlib.sha256(arr.tobytes()).hexdigest()]


def _json_part(raw: Any) -> Any:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(str(raw))


def _digest(parts: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(parts)).hexdigest()


def _state_parts(state: CuratedState) -> Dict[str, Any]:
    fit = state.spectrum_fit
    window_columns, peak_columns, covariance = _fit_tables(fit)
    header: Dict[str, Any] = {
        "n_windows": int(fit.n_windows),
        "n_fitted_peaks": int(fit.n_fitted_peaks),
        "final_plan_revision": int(fit.final_plan_revision),
        "parameters": _json_part(json.dumps(fit.parameters, default=str)),
        "diagnostics": _json_part(json.dumps(fit.diagnostics, default=str)),
        "thaw_history": _json_part(
            json.dumps([_thaw_info_to_json(e) for e in fit.thaw_history])
        ),
        "replan_history": _json_part(
            json.dumps([_replan_info_to_json(e) for e in fit.replan_history])
        ),
        "rescue_history": _json_part(
            json.dumps([_rescue_round_to_json(e) for e in fit.rescue_history])
        ),
    }
    review = state.review
    return {
        "fit": {
            "header": header,
            "windows": {k: _column_part(v) for k, v in window_columns.items()},
            "peaks": {k: _column_part(v) for k, v in peak_columns.items()},
            "covariance": _column_part(covariance),
        },
        "review": {
            "window_statuses": _json_part(
                json.dumps(
                    [_status_to_dict(s) for s in review.window_statuses.values()]
                )
            ),
            "decision_log": _json_part(
                json.dumps([_entry_to_dict(e) for e in review.decision_log])
            ),
            "final_products": (
                None
                if review.final_products is None
                else _json_part(
                    json.dumps(_final_products_to_dict(review.final_products))
                )
            ),
            "created_windows": _json_part(
                json.dumps(
                    [_fit_window_to_dict(w) for w in review.created_windows],
                    default=str,
                )
            ),
        },
    }


def _read_columns(group: h5py.Group) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, ds in group.items():
        if h5py.check_string_dtype(ds.dtype) is not None:
            out[name] = _column_part(np.asarray(ds.asstr()[...], dtype=object))
        else:
            out[name] = _column_part(ds[...])
    return out


def _persisted_parts(path: str) -> Dict[str, Any]:
    with h5open(path, "r") as h5f:
        fit_grp = h5f["stage5_fitting"]
        header: Dict[str, Any] = {}
        for name in _FIT_HEADER_ATTRS:
            raw = fit_grp.attrs[name]
            header[name] = int(raw) if name in _FIT_HEADER_INTS else _json_part(raw)
        fit = {
            "header": header,
            "windows": _read_columns(fit_grp["windows"]),
            "peaks": _read_columns(fit_grp["peaks"]),
            "covariance": _column_part(fit_grp["covariance"][...]),
        }
        review_grp = h5f.get("stage6_review")
        review: Dict[str, Any] = {}
        for name in _REVIEW_RECORDS:
            sub = None if review_grp is None else review_grp.get(name)
            raw = None if sub is None else sub.attrs.get("data")
            review[name] = None if raw is None else _json_part(raw)
    # What the reader takes an absent record to mean.
    for name in ("window_statuses", "decision_log", "created_windows"):
        if review[name] is None:
            review[name] = []
    return {"fit": fit, "review": review}


def state_digest(state: CuratedState) -> str:
    """SHA-256 over *state* as Stage 6 writes it (see the module docstring for
    what it covers). Floats are compared exactly."""
    return _digest(_state_parts(state))


def persisted_state_digest(file_path: Union[str, Path]) -> str:
    """:func:`state_digest` of the curated state stored in *file_path*, read
    from the stored bytes (the fit tables and the review's JSON records)
    rather than from objects loaded from them."""
    return _digest(_persisted_parts(str(file_path)))


def differing_parts(state: CuratedState, file_path: Union[str, Path]) -> List[str]:
    """The parts (``"fit.peaks.frequency_mhz"``, ``"review.decision_log"``, ...)
    in which *state* differs from the state stored in *file_path*: a
    diagnostic for a failed digest comparison. Empty when the digests agree."""
    a, b = _state_parts(state), _persisted_parts(str(file_path))
    out: List[str] = []
    for top in ("fit", "review"):
        for key in sorted(set(a[top]) | set(b[top])):
            x, y = a[top].get(key), b[top].get(key)
            if isinstance(x, dict) and isinstance(y, dict):
                out.extend(
                    f"{top}.{key}.{k}"
                    for k in sorted(set(x) | set(y))
                    if canonical_json(x.get(k)) != canonical_json(y.get(k))
                )
            elif canonical_json(x) != canonical_json(y):
                out.append(f"{top}.{key}")
    return out
