"""The structural replan on experiment 2638.

When a window edge carries a coherent residual with no contributor to thaw, Stage
5 asks to merge the window with its neighbour. The window's own fit must hold a
line (a window the cleanup emptied has no fitted line straddling its boundary),
the partner must *touch* the window (at most one active-FT bin between them),
the merged window must fit the plan's caps, and a round applies a disjoint set
of merges. Before the touching rule was enforced the 2638 fit asked for
far-apart merges, a window flagged on both edges was named twice, Stage 4
rejected the round and no merge was ever applied (``docs/source/changelog.rst``,
``ANALYSIS_EPOCH`` 5).

The plan the cleanup golden fit uses (start detection with a 0.67 us guard
margin, timebase and tau calibration) has one touching pair whose edge flags,
windows 100 and 101 -- but the flag is window 100's, and the cleanup empties
window 100's fit, so it is not merged, in either line shape. The tests fit
windows 99..103 of that plan: on their own, where every flag is recorded and
nothing merges, and with the dispatcher scripted to ask for the 100/101 merge
from window 101's edge (which holds lines), so the merge path runs end to end on
real data. They also check the whole default-recipe plan (where every flag is
refused for distance) against the session's whole-plan Gaussian fit.

The scripted and synthetic cases (selection, refusals, rounds, parallel ==
sequential history) are in ``tests/unit/fitting/test_plan_execution.py``.

Writes only to pytest temp dirs. Every test is ``slow``: building the plan and
fitting it take over ten seconds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Tuple

import numpy as np
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

#: The one touching pair among the golden plan's flagged edges.
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


def _merge_from_101() -> Callable[..., Any]:
    """``_dispatch_structural_round`` that asks, in the first round only, for
    the 100/101 merge from window 101's low edge (whose fit holds lines), as if
    that edge flagged. Selection, Stage 4's replan and the re-fit are the real
    ones."""
    real = plan_execution._dispatch_structural_round

    def dispatch(outcomes: Any, plan: Any, threshold: float) -> Any:
        if plan.plan_revision:
            return []
        return [
            p
            for p in real(outcomes, plan, -np.inf)
            if (p.window_id, p.partner_id) == (101, 100)
        ]

    return dispatch


@pytest.fixture(scope="module", params=["gaussian", "lorentzian"])
def fits(request, golden_subset, tmp_path_factory):
    """``(parallel, sequential, forced parallel, forced sequential)`` fits of the
    subset for one shape; the forced pair asks for the 100/101 merge."""
    shape = request.param
    tmp = tmp_path_factory.mktemp(f"replan_{shape}")
    parallel = _fit(golden_subset, tmp / "par.ftmw", shape=shape)
    sequential = _fit(golden_subset, tmp / "seq.ftmw", shape=shape, jobs=1)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(plan_execution, "_dispatch_structural_round", _merge_from_101())
        forced = _fit(golden_subset, tmp / "forced_par.ftmw", shape=shape)
        forced_seq = _fit(golden_subset, tmp / "forced_seq.ftmw", shape=shape, jobs=1)
    return parallel, sequential, forced, forced_seq


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


def test_an_emptied_window_s_flags_are_recorded_and_nothing_merges(fits):
    """Window 100 is flagged on both edges, toward 99 (far) and 101 (touching),
    but the cleanup emptied its fit: neither flag is evidence of a line across
    the boundary, so neither is merged and the plan stays as Stage 4 built
    it."""
    parallel, _seq, _forced, _forced_seq = fits
    fit = ftmw.load_fit(parallel)
    assert fit.final_plan_revision == 0
    assert fit.replan_history, "expected window 100's flags on record"
    assert not [e for e in fit.replan_history if e.accepted]
    assert {
        (e.triggering_window_id, e.partner_window_id) for e in fit.replan_history
    } == {
        (100, 99),
        (100, 101),
    }
    for e in fit.replan_history:
        assert e.revision_after == e.revision_before == 0
        assert e.reason.startswith("not merged: "), e.reason
        assert "fit holds no line" in e.reason
    assert set(_windows(fit)) == {99, 101, 102, 103}


def test_a_sequential_fit_records_the_same_replan_as_the_parallel_one(fits):
    parallel, sequential, _forced, _forced_seq = fits
    assert _history(sequential) == _history(parallel)
    assert_same_fit(sequential, parallel)


def test_a_forced_merge_is_recorded_and_bumps_the_revision(fits):
    _par, _seq, forced, _forced_seq = fits
    fit = ftmw.load_fit(forced)
    (merge,) = fit.replan_history
    assert merge.accepted
    assert (merge.triggering_window_id, merge.partner_window_id) == (101, 100)
    assert merge.surviving_window_id == min(_PAIR)
    assert (merge.revision_before, merge.revision_after) == (0, 1)
    assert fit.final_plan_revision == 1
    assert set(_windows(fit)) == {99, 100, 102, 103}


def test_the_merged_window_covers_both_and_keeps_the_lines(fits):
    parallel, _seq, forced, _forced_seq = fits
    got, ref = _windows(ftmw.load_fit(forced)), _windows(ftmw.load_fit(parallel))
    assert set(ref) == {99, 101, 102, 103}
    merged, old = got[100], ref[101]
    lo, hi = merged.window.freq_range
    old_lo, old_hi = old.window.freq_range
    assert lo < old_lo
    assert hi == pytest.approx(old_hi, abs=1e-6)
    assert len(merged.fitted_peaks) == len(old.fitted_peaks)
    # The other windows are the fit they were without the merge.
    for wid in (99, 102, 103):
        assert tuple(got[wid].window.freq_range) == tuple(ref[wid].window.freq_range)


def test_a_merge_keeps_every_peak_uid(fits):
    parallel, _seq, forced, _forced_seq = fits
    assert _uids(ftmw.load_fit(forced)) == _uids(ftmw.load_fit(parallel))


def test_a_sequential_forced_merge_equals_the_parallel_one(fits):
    _par, _seq, forced, forced_seq = fits
    assert _history(forced_seq) == _history(forced)
    assert_same_fit(forced_seq, forced)


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
