"""
The full-replay reference for Stage 6 curation, and its bitwise state digest.

:func:`replay_full` computes the curated state a decision log describes, from
scratch and in memory: the automatic-fit baseline, the file's static analysis
inputs and the log, nothing else. It persists nothing, keeps no cache and reads
no curated state (no ``/stage5_fitting`` curated fit, no ``/stage6_review``).
It is built from the same functions the write path runs -- the shared fit
context, the batch appliers, the one combined cascade, the review derivation
and the final-products build -- in the order an undo replay runs them, so it is
the oracle a write is checked against: after a write, the persisted state's
:func:`persisted_state_digest` must equal :func:`state_digest` of
:func:`replay_full` of the log the write left.

:func:`state_digest` covers what Stage 6 persists of a curated state: the fit
tables exactly as :func:`~ftmwpipeline.io.fitting_serialization.save_spectrum_fit_to_hdf5`
writes them (window and peak columns and the covariance, byte for byte, plus
the fit header's JSON records), and the review's window statuses, decision
log, final products and created-window overlay as their stored JSON. Write
stamps (``creation_time``) and bookkeeping attrs that are not part of the state
(the serial high-water mark, the engine version, the baseline's lineage id) are
left out.
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
    Sideband,
    SpectrumFit,
    Stage6Review,
)
from ..file_manager import CurationConflictError
from ..fitting.validation import DEFAULT_CHI2R_NOISE_FLOOR, DEFAULT_SHAPE_ERROR_KAPPA
from ..io.fitting_serialization import (
    _fit_tables,
    _replan_info_to_json,
    _rescue_round_to_json,
    _thaw_info_to_json,
    load_spectrum_fit_from_hdf5,
)
from ..io.stage6_review_serialization import (
    _entry_to_dict,
    _final_products_to_dict,
    _fit_window_to_dict,
    _status_to_dict,
)
from .atomic import h5open
from .empty_window_attention import flagged_lineless_ids
from .fingerprint_impl import canonical_json
from .stage0_impl import load_fid_from_pipeline_impl
from .stage6_impl import (
    _FIT_EDIT_KINDS,
    DEFAULT_ATTENTION_CANDIDATE_EVIDENCE,
    DEFAULT_DISPLAY_BAR,
    STAGE5_BASELINE_GROUP,
    _apply_batch_segment,
    _auto_merged_window_freqs,
    _BatchChangeset,
    _BatchCtx,
    _build_shared_fit_ctx,
    _cascade_batch,
    _derive_batch_review,
    _final_products_for_fit,
    _overlay_created_windows,
    _replay_plan,
    _review_run_statuses,
    _seed_unresolved_spreads_from_diagnostics,
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


@dataclass(frozen=True)
class ReviewParams:
    """The ``review run`` attention-routing parameters a status computation
    uses (``bar``, ``attention_candidate_evidence``, ``kappa``,
    ``noise_floor``), at ``review run``'s defaults unless given."""

    bar: float = DEFAULT_DISPLAY_BAR
    attention_candidate_evidence: float = DEFAULT_ATTENTION_CANDIDATE_EVIDENCE
    kappa: float = DEFAULT_SHAPE_ERROR_KAPPA
    noise_floor: float = DEFAULT_CHI2R_NOISE_FLOOR


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
    """The curated state *log* describes, replayed in one batch from the
    automatic-fit baseline, in memory.

    The steps an undo runs after its restore, without the restore: the review
    ``review run`` would build on the baseline (statuses under *params*, final
    products), then the log's actions replayed in log order in one batch
    (:func:`~.stage6_impl._replay_plan`, one action per recorded user action),
    one combined cascade, and the review derived from the result with the
    log's rows. Reads the baseline and the file's static inputs only.

    A log with fit-changing rows needs the baseline; a log of bare accepts
    only replays onto ``/stage5_fitting``, which such a log never changed (it
    takes no baseline). Refused with ``baseline_unavailable`` otherwise.
    """
    path = str(file_path)
    log = list(log)
    params = params or ReviewParams()
    with h5open(path, "r") as h5f:
        baseline = STAGE5_BASELINE_GROUP in h5f
    if any(e.kind in _FIT_EDIT_KINDS for e in log) and not baseline:
        raise CurationConflictError(
            "baseline_unavailable",
            message="no automatic-fit baseline to replay the decision log from",
        )
    group = STAGE5_BASELINE_GROUP if baseline else "stage5_fitting"
    with h5open(path, "r") as h5f:
        spectrum_fit = load_spectrum_fit_from_hdf5(h5f[group])

    # The review a fresh ``review run`` builds on the baseline.
    fid = load_fid_from_pipeline_impl(path)
    statuses = _review_run_statuses(
        path,
        spectrum_fit,
        Stage6Review(),
        Sideband.coerce(fid.sideband),
        spur_centers_mhz=[
            float(v) for v in spectrum_fit.parameters.get("spur_centers_mhz", [])
        ],
        acquisition_us=float(spectrum_fit.parameters.get("acquisition_us", 0.0)),
        merged_window_freqs=_auto_merged_window_freqs(spectrum_fit),
        bar=params.bar,
        attention_candidate_evidence=params.attention_candidate_evidence,
        kappa=params.kappa,
        noise_floor=params.noise_floor,
        fit_group=group,
    )
    review = Stage6Review(
        window_statuses=statuses,
        final_products=_final_products_for_fit(path, spectrum_fit, fid),
    )

    plan = _replay_plan(log)
    snap_tol = refit_snap_tol_mhz_impl(path)
    shared = _build_shared_fit_ctx(path, fit_group=group)
    # A plan of bare accepts never loads the fit for a batch (it records one
    # accept at a time), so the load-time spread recovery does not run on it.
    if any(a.kind != "accept" or a.candidate is not None for a in plan):
        _seed_unresolved_spreads_from_diagnostics(spectrum_fit, snap_tol_mhz=snap_tol)
    fit_ids = {
        int(wf.window_id) for wf in spectrum_fit.window_fits if wf.window_id is not None
    }
    changeset = _BatchChangeset(
        spectrum_fit=spectrum_fit,
        created_windows=[],
        fit_window_map={
            w.window_id: w
            for w in _overlay_created_windows(shared.base_plan, []).windows
        },
        lineless_reviewable=frozenset(flagged_lineless_ids(review, fit_ids)),
        replayed=log,
    )
    # Never persisted: ``baseline_taken=False`` makes _finish_batch refuse it.
    ctx = _BatchCtx(shared=shared, changeset=changeset, baseline_taken=False)
    _apply_batch_segment(
        ctx, plan, snap_tol_mhz=snap_tol, action_indices={}, log_order=True
    )
    _cascade_batch(ctx, snap_tol_mhz=snap_tol)
    return CuratedState(
        spectrum_fit=ctx.changeset.spectrum_fit,
        review=_derive_batch_review(ctx, path, existing_review=review),
    )


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
