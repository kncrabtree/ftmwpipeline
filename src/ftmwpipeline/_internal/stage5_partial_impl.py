"""Stage 5 partial fits: write one on an interrupt, resume from one.

Normative spec: ``dev-docs/CONTRACT_STRATEGY.md`` §Events and cancellation →
§Stage 5 partial fits. The storage is
:mod:`ftmwpipeline.io.stage5_partial_serialization`; the walk side (what a
finished window is, carrying windows into a walk) is
:class:`~ftmwpipeline.fitting.plan_execution.WalkTracker` /
:class:`~ftmwpipeline.fitting.plan_execution.CarriedWindows`. This module is
the policy between them, called from
:func:`~ftmwpipeline._internal.stage5_impl.fit_peaks_impl`:

- :func:`build_partial_provenance` -- what a fit records about its inputs, and
  what a later fit compares before it resumes: the resolved Stage 5 settings
  (in the ``stage5_fit`` record's form), the values consumed from other stages
  (:class:`~ftmwpipeline.io.stage_fit_settings_serialization.Stage5Consumed`),
  ``ANALYSIS_EPOCH``, and the fit's derived context (the window plan and peak
  list identity, the gated spur catalog, the active-FT geometry, the decay
  seeds);
- :func:`decide_resume` -- resume, or start over and say why;
- :func:`write_partial_fit` -- the interrupt-time write.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import h5py
import numpy as np

from ..contract import FIT_RESTART_REASONS
from ..core.environment import ANALYSIS_EPOCH
from ..core.stage_fit_settings import StageFitSettings
from ..core.stage_fit_settings import to_attrs as stage_fit_to_attrs
from ..file_manager import invalidate_stages_in_file
from ..fitting.plan_execution import CarriedWindows, PartialWalk
from ..io.stage5_partial_serialization import (
    PartialCodecError,
    encode_stage5_partial,
    read_stage5_partial_provenance,
    read_stage5_partial_windows,
    stage5_partial_present,
    write_stage5_partial,
)
from ..io.stage_fit_settings_serialization import (
    Stage5Consumed,
    save_stage_fit_settings_to_h5,
)
from .atomic import h5open
from .events import StageScope

logger = logging.getLogger(__name__)

#: ``restart_reason`` values of a ``fit run`` summary (``None`` when the fit
#: resumed, or there was nothing to resume); the vocabulary is declared once,
#: as :data:`~ftmwpipeline.contract.FIT_RESTART_REASONS`.
RESTART_REQUESTED, SETTINGS_CHANGED, INCOMPLETE_PROVENANCE, THAW_REFIT = (
    FIT_RESTART_REASONS
)

#: The provenance keys a resume compares; any other difference is not one.
_COMPARED = ("analysis_epoch", "settings", "consumed", "context")


def _normalize(value: Any) -> Any:
    """A JSON-able, type-normalized form of ``value`` for comparison.

    Every number that is not a ``bool`` becomes a ``float`` (a knob given as
    ``3`` and read back from the file as ``3.0`` is the same value); sequences
    become lists, mappings dicts with string keys, enums their values,
    dataclasses their fields.
    """
    if value is None or isinstance(value, (bool, np.bool_)):
        return None if value is None else bool(value)
    if isinstance(value, enum.Enum):
        return _normalize(value.value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, (str, bytes)):
        return value.decode("utf-8") if isinstance(value, bytes) else value
    if isinstance(value, np.ndarray):
        return [_normalize(v) for v in value.tolist()]
    if isinstance(value, Mapping):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _normalize(getattr(value, f.name))
            for f in dataclasses.fields(value)
        }
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    return repr(value)


def _canonical(value: Any) -> str:
    return json.dumps(_normalize(value), sort_keys=True, allow_nan=True)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def plan_identity(plan: Any) -> Dict[str, Any]:
    """The fit-relevant identity of a Stage 4 window plan (its windows, their
    contributors, the dependency order and the planner's parameters)."""
    return {
        "plan_revision": int(plan.plan_revision),
        "n_windows": len(plan.windows),
        "digest": _digest(
            {
                "windows": [
                    {
                        "window_id": w.window_id,
                        "freq_range": w.freq_range,
                        "free_peak_indices": w.free_peak_indices,
                        "fixed_contributors": w.fixed_contributors,
                        "batch": w.batch,
                    }
                    for w in plan.windows
                ],
                "dependency_edges": plan.dependency_edges,
                "topological_order": plan.topological_order,
                "parameters": plan.parameters,
                "plan_revision": plan.plan_revision,
            }
        ),
    }


def build_partial_provenance(
    *,
    resolved: StageFitSettings,
    consumed: Stage5Consumed,
    context: Mapping[str, Any],
) -> Dict[str, str]:
    """What a fit compares before resuming: each entry canonical JSON.

    ``context`` is the fit's derived inputs beyond its settings and the values
    it consumed (the plan identity, the peak list, the gated spur catalog, the
    active-FT geometry, the decay seeds).
    """
    return {
        "analysis_epoch": _canonical(int(ANALYSIS_EPOCH)),
        "settings": _canonical(stage_fit_to_attrs(resolved)),
        "consumed": _canonical(consumed.to_attrs()),
        "context": _canonical(dict(context)),
    }


@dataclass
class ResumeDecision:
    """What :func:`decide_resume` found: the windows to carry (``None`` to fit
    every window) and, when starting over, why."""

    carried: Optional[CarriedWindows]
    restart_reason: Optional[str]


def decide_resume(
    file_path: str,
    provenance: Mapping[str, str],
    *,
    restart: bool,
    window_ids: List[int],
) -> ResumeDecision:
    """Resume from the file's partial fit, or say why the fit starts over.

    No partial fit: a fresh fit with no reason. ``restart``:
    ``restart_requested``. Missing or unreadable provenance, or windows that
    cannot be read back or are not in the plan: ``incomplete_provenance``. A
    different epoch, settings, consumed value or derived context:
    ``settings_changed``. A partial fit that saw an accepted thaw:
    ``thaw_refit`` (its windows may hold a primary the thaw changed). Never
    resumes on a guess.
    """
    with h5open(file_path, "r") as h5f:
        if not stage5_partial_present(h5f):
            return ResumeDecision(None, None)
        if restart:
            return _restart(RESTART_REQUESTED)
        try:
            return _decide_from_partial(h5f, provenance, window_ids)
        except Exception as exc:  # noqa: BLE001 - never resume on a guess
            # Backstop: whatever a malformed partial fit raises while it is
            # read, it is not resumed from.
            logger.warning("Stage 5 partial fit unreadable (%s); starting over", exc)
            return _restart(INCOMPLETE_PROVENANCE)


def _decide_from_partial(
    h5f: h5py.File, provenance: Mapping[str, str], window_ids: List[int]
) -> ResumeDecision:
    """:func:`decide_resume` once a partial fit is present and no restart was
    requested."""
    recorded = read_stage5_partial_provenance(h5f)
    if recorded is None or any(
        not isinstance(recorded.get(key), str) for key in _COMPARED
    ):
        return _restart(INCOMPLETE_PROVENANCE)
    walk = recorded.get("walk")
    if not isinstance(walk, dict) or not isinstance(walk.get("accepted_thaw"), bool):
        return _restart(INCOMPLETE_PROVENANCE)
    changed = [key for key in _COMPARED if recorded[key] != provenance[key]]
    if changed:
        logger.info(
            "Stage 5 partial fit not resumed: %s differ(s) from the " "partial fit's",
            ", ".join(changed),
        )
        return _restart(SETTINGS_CHANGED)
    if walk["accepted_thaw"]:
        return _restart(THAW_REFIT)
    try:
        carried = read_stage5_partial_windows(h5f)
    except PartialCodecError as exc:
        logger.warning("Stage 5 partial fit unreadable (%s); starting over", exc)
        return _restart(INCOMPLETE_PROVENANCE)
    known = set(window_ids)
    if not carried.order or any(wid not in known for wid in carried.order):
        return _restart(INCOMPLETE_PROVENANCE)
    return ResumeDecision(carried, None)


def _restart(reason: str) -> ResumeDecision:
    logger.info("Stage 5 partial fit discarded (%s); fitting every window", reason)
    return ResumeDecision(None, reason)


def write_partial_fit(
    file_path: str,
    partial: PartialWalk,
    provenance: Mapping[str, str],
    *,
    resolved: StageFitSettings,
    preset_name: Optional[str],
    events: Optional[StageScope],
) -> List[int]:
    """Write ``partial`` as the file's partial fit, inside the fit's transaction.

    Encodes first; when no window can be kept nothing is written (the previous
    fit stays). Otherwise one write: the previous fit and everything downstream
    of it are discarded (as are any earlier partial fit and the curation
    baseline), the partial fit is written, and the resolved settings are
    stamped as the persisted ``stage5_fit`` record without a ``consumed`` block
    (the stage is not complete), so a plain ``fit run`` resumes with them.
    Returns the window ids written, sorted.
    """
    encoded, refused = encode_stage5_partial(partial)
    if refused:
        logger.warning(
            "Stage 5 partial fit: %d finished window(s) hold a value that cannot "
            "be kept (%s); a resume refits them",
            len(refused),
            ", ".join(str(w) for w in refused),
        )
    if not encoded:
        return []
    record = dict(provenance)
    walk: Dict[str, Any] = {
        "phase": partial.phase,
        "accepted_thaw": bool(partial.accepted_thaw),
        "walk_mode": partial.walk_mode,
        "n_windows": int(partial.n_windows),
    }
    from .stage6_impl import STAGE5_BASELINE_GROUP

    with h5open(file_path, "a") as h5f:
        if "pipeline_stages" in h5f:
            invalidate_stages_in_file(
                h5f,
                ["stage5_fitting"],
                include_roots=True,
                reason="Stage 5 cancelled; the finished windows are kept as a "
                "partial fit",
                events=events,
            )
        if STAGE5_BASELINE_GROUP in h5f:
            del h5f[STAGE5_BASELINE_GROUP]
        written = write_stage5_partial(h5f, encoded, {**record, "walk": walk})
    save_stage_fit_settings_to_h5(file_path, resolved, preset_name=preset_name)
    logger.info(
        "Stage 5 interrupted: %d of %d window(s) kept as a partial fit",
        len(written),
        int(partial.n_windows),
    )
    return sorted(written)


__all__ = [
    "INCOMPLETE_PROVENANCE",
    "RESTART_REQUESTED",
    "SETTINGS_CHANGED",
    "THAW_REFIT",
    "ResumeDecision",
    "build_partial_provenance",
    "decide_resume",
    "plan_identity",
    "write_partial_fit",
]
