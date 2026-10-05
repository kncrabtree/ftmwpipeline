"""The fitted plan: the window plan a complete Stage 5 fit was made on.

``_internal.fitted_plan`` assembles it from ``/stage4_windows`` and, after a
structural merge, the plan the fit stored (``/stage5_fitting/fitted_plan``). A
merged fit from before that record existed keeps the Stage 4 plan, with the
windows the merges touched marked unavailable for Stage 6 refits. Files are
built with h5py in ``tmp_path`` only.
"""

from __future__ import annotations

import json
import math

import h5py
import pytest

from ftmwpipeline._internal.fitted_plan import (
    FIT_PLAN_UNAVAILABLE,
    fitted_plan_from_h5,
    fitted_window_bounds,
)
from ftmwpipeline.core.data_structures import (
    FitWindow,
    FixedContributor,
    WindowPlan,
)
from ftmwpipeline.io.window_serialization import (
    save_fitted_plan_to_hdf5,
    save_window_plan_to_hdf5,
)


def _contrib(primary: int) -> FixedContributor:
    return FixedContributor(peak_index=0, primary_window_id=primary, frequency_mhz=1.0)


def _stage4() -> WindowPlan:
    """Windows 0..3; window 3 holds a line of window 2 as a fixed contributor."""
    return WindowPlan(
        windows=[
            FitWindow(window_id=0, freq_range=(100.0, 110.0)),
            FitWindow(window_id=1, freq_range=(120.0, 130.0)),
            FitWindow(window_id=2, freq_range=(130.5, 140.0)),
            FitWindow(
                window_id=3, freq_range=(150.0, 160.0), fixed_contributors=[_contrib(2)]
            ),
        ],
        dependency_edges=[(3, 2)],
        topological_order=[0, 1, 2, 3],
        parameters={"min_freeze_snr": 7.0},
    )


def _record(trigger: int, partner: int, accepted: bool = True) -> dict:
    return {
        "triggering_window_id": trigger,
        "partner_window_id": partner,
        "surviving_window_id": min(trigger, partner),
        "edge_side": "low",
        "edge_coherence_before": 9.0,
        "revision_before": 0,
        "revision_after": 1 if accepted else 0,
        "accepted": accepted,
        "reason": "" if accepted else "not merged: test",
    }


def _file(tmp_path, *, revision=None, history=(), stored=None):
    path = tmp_path / "f.ftmw"
    with h5py.File(path, "w") as h5f:
        save_window_plan_to_hdf5(_stage4(), h5f.create_group("stage4_windows"))
        if revision is not None:
            fit = h5f.create_group("stage5_fitting")
            fit.attrs["final_plan_revision"] = revision
            fit.attrs["replan_history"] = json.dumps(list(history))
            if stored is not None:
                save_fitted_plan_to_hdf5(stored, fit)
    return path


def _merged() -> WindowPlan:
    """The plan after window 2 was folded into window 1."""
    return WindowPlan(
        windows=[
            FitWindow(window_id=0, freq_range=(100.0, 110.0)),
            FitWindow(
                window_id=1,
                freq_range=(120.0, 140.0),
                diagnostics={"merged_from": [1, 2]},
            ),
            FitWindow(
                window_id=3, freq_range=(150.0, 160.0), fixed_contributors=[_contrib(1)]
            ),
        ],
        dependency_edges=[(3, 1)],
        topological_order=[0, 1, 3],
        parameters={"min_freeze_snr": 99.0},
        plan_revision=1,
    )


def _open(path):
    return h5py.File(path, "r")


@pytest.mark.parametrize("revision", [None, 0])
def test_without_a_revised_fit_the_fitted_plan_is_stage4(tmp_path, revision):
    path = _file(tmp_path, revision=revision)
    with _open(path) as h5f:
        fitted = fitted_plan_from_h5(h5f, _stage4())
        bounds, merged = fitted_window_bounds(h5f)
    assert [w.window_id for w in fitted.plan.windows] == [0, 1, 2, 3]
    assert not fitted.retired_window_ids
    assert not fitted.unavailable_window_ids
    assert bounds[2] == (130.5, 140.0)
    assert merged == {}


def test_a_stored_plan_replaces_the_stage4_windows(tmp_path):
    path = _file(tmp_path, revision=1, history=[_record(2, 1)], stored=_merged())
    with _open(path) as h5f:
        fitted = fitted_plan_from_h5(h5f, _stage4())
        bounds, merged = fitted_window_bounds(h5f)
    plan = fitted.plan
    assert [w.window_id for w in plan.windows] == [0, 1, 3]
    assert plan.window(1).freq_range == (120.0, 140.0)
    assert plan.window(3).fixed_contributors[0].primary_window_id == 1
    assert plan.dependency_edges == [(3, 1)]
    assert plan.topological_order == [0, 1, 3]
    assert plan.plan_revision == 1
    # The Stage 4 settings stay the plan's settings.
    assert plan.parameters == {"min_freeze_snr": 7.0}
    assert fitted.retired_window_ids == {2}
    assert not fitted.unavailable_window_ids
    assert bounds == {0: (100.0, 110.0), 1: (120.0, 140.0), 3: (150.0, 160.0)}
    assert merged == {1: [2]}


def test_a_revised_fit_without_its_plan_marks_the_merge_unavailable(tmp_path):
    """Windows 1 and 2 merged; window 3 reads window 2's line. All three would
    be refit on geometry the fit was not made on."""
    path = _file(tmp_path, revision=1, history=[_record(0, 1, False), _record(2, 1)])
    with _open(path) as h5f:
        fitted = fitted_plan_from_h5(h5f, _stage4())
        bounds, merged = fitted_window_bounds(h5f)
    assert [w.window_id for w in fitted.plan.windows] == [0, 1, 2, 3]
    assert fitted.unavailable_window_ids == {1, 2, 3}
    assert fitted.unavailable_spans_mhz == ((120.0, 140.0),)
    assert fitted.retired_window_ids == {2}
    assert bounds == {0: (100.0, 110.0), 1: (120.0, 140.0), 3: (150.0, 160.0)}
    assert merged == {1: [2]}


def test_chained_merges_join_into_one_survivor(tmp_path):
    path = _file(tmp_path, revision=2, history=[_record(2, 1), _record(1, 0)])
    with _open(path) as h5f:
        fitted = fitted_plan_from_h5(h5f, _stage4())
        bounds, merged = fitted_window_bounds(h5f)
    assert fitted.unavailable_spans_mhz == ((100.0, 140.0),)
    assert fitted.retired_window_ids == {1, 2}
    assert bounds == {0: (100.0, 140.0), 3: (150.0, 160.0)}
    assert merged == {0: [1, 2]}


def test_a_revised_fit_with_no_merge_on_record_trusts_no_window(tmp_path):
    path = _file(tmp_path, revision=1, history=[_record(0, 1, False)])
    with _open(path) as h5f:
        fitted = fitted_plan_from_h5(h5f, _stage4())
    assert fitted.unavailable_window_ids == {0, 1, 2, 3}
    ((lo, hi),) = fitted.unavailable_spans_mhz
    assert math.isinf(lo) and math.isinf(hi)


def test_the_refusal_reason_is_a_stable_slug():
    assert FIT_PLAN_UNAVAILABLE == "fit_plan_unavailable"
