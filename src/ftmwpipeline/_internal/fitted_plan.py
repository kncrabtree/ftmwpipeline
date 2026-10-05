"""The fitted window plan: the windows a complete Stage 5 fit was made on.

Stage 5 fits the Stage 4 plan, but a structural replan can merge a window into
its touching neighbour mid-fit. The survivor keeps the lower id and the merged
range; the absorbed id is gone. From then on the fit's windows are not the
Stage 4 plan's, and everything that reads window geometry after a complete fit
-- ``window_status``, and every window Stage 6 resolves, edits, refits or mints
an id against -- has to read the fit's own plan (CONTRACT_STRATEGY §Window
status). This module is the one place that plan is assembled.

Three cases, by what the file holds:

* **No complete fit, or a fit whose plan was never revised**
  (``final_plan_revision == 0``): the fitted plan *is* the Stage 4 plan.
* **A revised fit with its plan stored** (``/stage5_fitting/fitted_plan``,
  written by the fit since this record existed): the Stage 4 plan with its
  windows, dependency edges and fit order replaced by the stored plan's. Its
  parameters and plan-level diagnostics stay the Stage 4 plan's.
* **A revised fit without a stored plan** (fitted before the record existed):
  the windows the fit was made on are not in the file. The Stage 4 plan is
  returned with the windows the merges touched marked unavailable, so Stage 6
  refuses to refit them (``curation_conflict``, reason
  :data:`FIT_PLAN_UNAVAILABLE`) rather than refit them on the wrong geometry.
  Their *ranges* are still known exactly -- a merge's range is the union of
  its members' -- so :func:`fitted_window_bounds` reports them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple, Union

import h5py

from ..core.data_structures import WindowPlan
from ..io._hdf5_helpers import load_json_attr
from ..io.window_serialization import (
    load_fitted_plan_from_hdf5,
    read_fitted_plan_bounds,
    read_window_plan_columns,
)
from ..preprocessing.window_planning import retired_window_ids
from .atomic import h5open

__all__ = [
    "FIT_PLAN_UNAVAILABLE",
    "FittedPlan",
    "fitted_plan_from_h5",
    "load_fitted_plan",
    "fitted_window_bounds",
]

#: ``curation_conflict`` reason: the edit would refit (or create a window
#: against) a window a structural merge changed, in a fit that predates the
#: stored fitted plan, so the geometry the fit was made on is not in the file.
FIT_PLAN_UNAVAILABLE = "fit_plan_unavailable"

_FIT_GROUP = "stage5_fitting"


@dataclass(frozen=True)
class FittedPlan:
    """The plan a file's fit was made on, as Stage 6 needs it.

    Attributes
    ----------
    plan :
        The fitted plan, without Stage 6 created windows (the caller overlays
        those).
    retired_window_ids :
        Ids a structural merge absorbed. They are not windows of ``plan`` and a
        new window must never take one.
    unavailable_window_ids :
        Windows whose fitted geometry the file does not hold (a revised fit
        without a stored plan): the merged windows and every window holding
        one of them as a fixed contributor. Empty otherwise.
    unavailable_spans_mhz :
        The merged frequency ranges of such a fit, ``(low, high)``; a window
        created inside one would overlap a window the fit has.
    """

    plan: WindowPlan
    retired_window_ids: FrozenSet[int] = frozenset()
    unavailable_window_ids: FrozenSet[int] = frozenset()
    unavailable_spans_mhz: Tuple[Tuple[float, float], ...] = ()


def _accepted_merges(fit_group: h5py.Group) -> List[List[int]]:
    """The window-id sets the fit's accepted merges joined, one per survivor.

    Built from the replan record: each accepted record joins its triggering
    and partner windows, and chained merges join into one set.
    """
    records: Sequence[Dict[str, Any]] = load_json_attr(
        fit_group, "replan_history", [], label=_FIT_GROUP
    )
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for rec in records:
        if not rec.get("accepted"):
            continue
        a = find(int(rec["triggering_window_id"]))
        b = find(int(rec["partner_window_id"]))
        if a != b:
            parent[max(a, b)] = min(a, b)
    groups: Dict[int, List[int]] = {}
    for wid in parent:
        groups.setdefault(find(wid), []).append(wid)
    return [sorted(ids) for _root, ids in sorted(groups.items())]


def _revision(fit_group: h5py.Group) -> int:
    return int(fit_group.attrs.get("final_plan_revision", 0))


def _merge_spans(
    groups: Sequence[Sequence[int]], bounds: Dict[int, Tuple[float, float]]
) -> Dict[int, Tuple[float, float]]:
    """Each merge's range, keyed by its survivor: the union of its members'."""
    spans: Dict[int, Tuple[float, float]] = {}
    for ids in groups:
        known = [bounds[w] for w in ids if w in bounds]
        if not known:
            continue
        spans[min(ids)] = (
            min(min(lo, hi) for lo, hi in known),
            max(max(lo, hi) for lo, hi in known),
        )
    return spans


def fitted_plan_from_h5(h5f: h5py.File, stage4_plan: WindowPlan) -> FittedPlan:
    """The fitted plan of the open file *h5f*, given its Stage 4 plan."""
    if _FIT_GROUP not in h5f:
        return FittedPlan(stage4_plan)
    fit_group = h5f[_FIT_GROUP]
    stored = load_fitted_plan_from_hdf5(fit_group)
    if stored is not None:
        plan = replace(
            stage4_plan,
            windows=stored.windows,
            dependency_edges=stored.dependency_edges,
            topological_order=stored.topological_order,
            plan_revision=stored.plan_revision,
        )
        return FittedPlan(plan, retired_window_ids=frozenset(retired_window_ids(plan)))
    if _revision(fit_group) <= 0:
        return FittedPlan(stage4_plan)

    # A revised fit from before the plan was stored.
    groups = _accepted_merges(fit_group)
    stage4_ids = {int(w.window_id) for w in stage4_plan.windows}
    if not groups:
        # Revised, yet no accepted merge on record: nothing says which windows
        # moved, so none of them can be trusted.
        return FittedPlan(
            stage4_plan,
            unavailable_window_ids=frozenset(stage4_ids),
            unavailable_spans_mhz=((float("-inf"), float("inf")),),
        )
    merged = {w for ids in groups for w in ids}
    affected = set(merged)
    for w in stage4_plan.windows:
        if any(int(fc.primary_window_id) in merged for fc in w.fixed_contributors):
            affected.add(int(w.window_id))
    bounds = {int(w.window_id): w.freq_range for w in stage4_plan.windows}
    spans = _merge_spans(groups, bounds)
    retired = merged - {min(ids) for ids in groups}
    return FittedPlan(
        stage4_plan,
        retired_window_ids=frozenset(retired),
        unavailable_window_ids=frozenset(affected),
        unavailable_spans_mhz=tuple(spans[k] for k in sorted(spans)),
    )


def load_fitted_plan(file_path: Union[str, Path]) -> FittedPlan:
    """The fitted plan of the file at *file_path* (raises before Stage 4)."""
    from .stage4_impl import load_windows_impl

    stage4_plan: WindowPlan = load_windows_impl(str(file_path))["plan"]
    with h5open(str(file_path), "r") as h5f:
        return fitted_plan_from_h5(h5f, stage4_plan)


def fitted_window_bounds(
    h5f: h5py.File,
) -> Optional[Tuple[Dict[int, Tuple[float, float]], Dict[int, List[int]]]]:
    """The fitted plan's window ranges and merges, without loading a plan.

    ``({window_id: (freq_min, freq_max)}, {survivor_id: [absorbed ids]})`` --
    the bounds as stored (``freq_min`` / ``freq_max`` columns), a few column
    reads. ``None`` before Stage 4. Before a complete fit, and for a fit whose
    plan was never revised, these are the Stage 4 plan's. For a revised fit
    from before the plan was stored, each merge's survivor takes the union of
    its members' Stage 4 ranges and the absorbed ids are dropped.
    """
    if "stage4_windows" not in h5f:
        return None
    if _FIT_GROUP in h5f:
        stored = read_fitted_plan_bounds(h5f[_FIT_GROUP])
        if stored is not None:
            return stored
    columns = read_window_plan_columns(
        h5f["stage4_windows"], ["window_id", "freq_min", "freq_max"]
    )
    bounds: Dict[int, Tuple[float, float]] = {
        int(i): (float(lo), float(hi))
        for i, lo, hi in zip(
            columns["window_id"], columns["freq_min"], columns["freq_max"]
        )
    }
    merged: Dict[int, List[int]] = {}
    if _FIT_GROUP in h5f and _revision(h5f[_FIT_GROUP]) > 0:
        groups = _accepted_merges(h5f[_FIT_GROUP])
        for survivor, span in _merge_spans(groups, bounds).items():
            ids = next(g for g in groups if min(g) == survivor)
            for w in ids:
                bounds.pop(w, None)
            bounds[survivor] = span
            merged[survivor] = [w for w in ids if w != survivor]
    return bounds, merged
