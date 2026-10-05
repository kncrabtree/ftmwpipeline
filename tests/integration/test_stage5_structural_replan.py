"""The structural replan on experiment 2638.

When a window edge carries a coherent residual with no contributor to thaw, Stage
5 asks to merge the window with its neighbour. The partner must *touch* the
window (at most one active-FT bin between them) and the merged window must fit the
plan's width cap, and a round applies a disjoint set of merges. Before that was
enforced the 2638 fit asked for far-apart merges, a window flagged on both edges
was named twice, Stage 4 rejected the round and no merge was ever applied
(``docs/source/changelog.rst``, ``ANALYSIS_EPOCH`` 5).

The merge needs the plan the cleanup golden fit uses (start detection with a
0.67 us guard margin, timebase and tau calibration): there the one pair that
touches, windows 100 and 101, merges in both line shapes. The tests fit windows
99..103 of that plan -- the same merge for a few seconds -- and check the whole
default-recipe plan (where the same flags are all refused) against the session's
whole-plan Gaussian fit.

The scripted and synthetic cases (selection, refusals, rounds, parallel ==
sequential history) are in ``tests/unit/fitting/test_plan_execution.py``.

Writes only to pytest temp dirs. Every test is ``slow``: building the plan and
fitting it take over ten seconds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import pytest

import ftmwpipeline.api as ftmw
import ftmwpipeline.fitting.plan_execution as plan_execution
from ftmwpipeline import Pipeline
from ftmwpipeline._internal.atomic import atomic_write
from ftmwpipeline._internal.stage4_impl import (
    load_windows_impl,
    save_window_plan_impl,
)
from ftmwpipeline.core.start_detection_settings import StartDetectionSettings
from tests.integration.test_stage5_resume import assert_same_fit

pytestmark = [pytest.mark.integration, pytest.mark.slow]

#: The one touching pair the golden plan's flagged edges ask to merge.
_PAIR = (100, 101)
#: Windows 99..103 of that plan: the pair, a flagged edge toward a far
#: neighbour (99), and two windows it does not interact with.
_WINDOWS = range(99, 104)


@pytest.fixture(scope="module")
def golden_subset(exp_2638_data_path, tmp_path_factory):
    """Stage 4 of the cleanup-golden recipe for 2638, cut to :data:`_WINDOWS`
    (read-only: tests copy it)."""
    fp = tmp_path_factory.mktemp("replan_golden") / "golden_subset.ftmw"
    pipe = Pipeline.create(filepath=fp, source=exp_2638_data_path, force=True)
    pipe.detect_start_time(settings=StartDetectionSettings(guard_margin_us=0.67))
    pipe.compute_ft(trim=(26500.0, 40000.0))
    pipe.calibrate_timebase(clocks=None)
    pipe.estimate_noise()
    pipe.calibrate_tau()
    pipe.detect_peaks()
    pipe.assign_windows()
    plan = load_windows_impl(str(fp))["plan"]
    keep = {w.window_id for w in plan.windows if w.window_id in _WINDOWS}
    assert set(_PAIR) <= keep, "the recipe no longer plans windows 100 and 101"
    plan.windows = [w for w in plan.windows if w.window_id in keep]
    plan.topological_order = [w for w in plan.topological_order if w in keep]
    plan.dependency_edges = [
        (a, b) for (a, b) in plan.dependency_edges if a in keep and b in keep
    ]
    with atomic_write(str(fp)):
        save_window_plan_impl(str(fp), plan)
    return fp


def _fit(src: Path, dest: Path, **kwargs: Any) -> Path:
    import shutil

    shutil.copy(src, dest)
    ftmw.fit_peaks(str(dest), **kwargs)
    return dest


@pytest.fixture(scope="module", params=["gaussian", "lorentzian"])
def fits(request, golden_subset, tmp_path_factory):
    """``(parallel, sequential, no replan)`` fits of the subset for one shape."""
    shape = request.param
    tmp = tmp_path_factory.mktemp(f"replan_{shape}")
    parallel = _fit(golden_subset, tmp / "par.ftmw", shape=shape)
    sequential = _fit(golden_subset, tmp / "seq.ftmw", shape=shape, jobs=1)
    with pytest.MonkeyPatch.context() as mp:
        # No round finds a trigger: the plan every window starts from.
        mp.setattr(plan_execution, "_dispatch_structural_round", lambda *a: [])
        off = _fit(golden_subset, tmp / "off.ftmw", shape=shape, jobs=1)
    return parallel, sequential, off


def _history(fp: Path) -> List[Tuple[Any, ...]]:
    return [
        (
            e.triggering_window_id,
            e.partner_window_id,
            e.surviving_window_id,
            e.edge_side,
            e.revision_before,
            e.revision_after,
            e.accepted,
            e.reason,
        )
        for e in ftmw.load_fit(fp).replan_history
    ]


def _windows(fit: Any) -> dict:
    return {w.window_id: w for w in fit.window_fits}


def _uids(fit: Any) -> List[int]:
    return sorted(p.peak_uid for w in fit.window_fits for p in w.fitted_peaks)


def test_the_touching_pair_merges_and_the_far_neighbour_is_refused(fits):
    parallel, _seq, _off = fits
    fit = ftmw.load_fit(parallel)
    merge, refused = (
        [e for e in fit.replan_history if e.accepted],
        [e for e in fit.replan_history if not e.accepted],
    )
    assert len(merge) == 1
    assert {merge[0].triggering_window_id, merge[0].partner_window_id} == set(_PAIR)
    assert merge[0].surviving_window_id == min(_PAIR)
    assert (merge[0].revision_before, merge[0].revision_after) == (0, 1)
    assert fit.final_plan_revision == 1
    # Window 100's low edge is flagged too, but 99 lies far below it: recorded as
    # not merged, with the plan revision untouched by it.
    assert refused, "expected the far neighbour's flag on record"
    for e in refused:
        assert (e.triggering_window_id, e.partner_window_id) == (100, 99)
        assert e.revision_after == e.revision_before
        assert e.reason.startswith("not merged: ")
        assert "no window touches it" in e.reason
    wins = _windows(fit)
    assert set(wins) == {99, 100, 102, 103}


def test_the_merged_window_covers_both_and_keeps_the_lines(fits):
    parallel, _seq, off = fits
    got, ref = _windows(ftmw.load_fit(parallel)), _windows(ftmw.load_fit(off))
    assert set(ref) == {99, 101, 102, 103}
    merged, old = got[100], ref[101]
    lo, hi = merged.window.freq_range
    old_lo, old_hi = old.window.freq_range
    assert lo < old_lo
    assert hi == pytest.approx(old_hi, abs=1e-6)
    assert len(merged.fitted_peaks) == len(old.fitted_peaks)
    # The other windows are the fit they were without the replan.
    for wid in (99, 102, 103):
        assert tuple(got[wid].window.freq_range) == tuple(ref[wid].window.freq_range)


def test_a_merge_keeps_every_peak_uid(fits):
    parallel, _seq, off = fits
    assert _uids(ftmw.load_fit(parallel)) == _uids(ftmw.load_fit(off))


def test_a_sequential_fit_records_the_same_replan_as_the_parallel_one(fits):
    parallel, sequential, _off = fits
    assert _history(sequential) == _history(parallel)
    assert_same_fit(sequential, parallel)


def test_the_default_recipe_plan_refuses_every_flag_and_records_it(
    stage5_gaussian_2638,
):
    """The tests' own whole 2638 plan (default start detection) has no touching
    flagged pair. Window 59 is flagged on both edges: before the fix that put
    window 59 in two merge requests and failed the round; now each flag is
    recorded as not merged and the fit is otherwise untouched."""
    fit = ftmw.load_fit(stage5_gaussian_2638)
    assert fit.replan_history, "expected flagged edges on the whole plan"
    assert fit.final_plan_revision == 0
    for e in fit.replan_history:
        assert not e.accepted
        assert e.revision_after == e.revision_before == 0
        assert e.reason.startswith("not merged: "), e.reason
        assert "no window touches it" in e.reason
    assert not [e for e in fit.replan_history if "replan failed" in e.reason]
