"""Events and cancellation (CONTRACT_STRATEGY §Events and cancellation).

The internal plumbing (``_internal/events.py``), the event types' wire form,
the cancel / callback errors, and the Stage 5 walk honouring a cancel -- in the
sequential walk and in the parallel walk, where a cancel must not wait for a
window still fitting.
"""

from __future__ import annotations

import logging
import multiprocessing
import pickle
import threading
import time

import pytest

from ftmwpipeline import (
    CallbackFailedError,
    CancelToken,
    Invalidated,
    OperationCancelledError,
    PipelineWarning,
    StageFinished,
    StageStarted,
    WindowProgress,
    to_jsonable,
)
from ftmwpipeline._internal.events import (
    detached_scope,
    emit_invalidated,
    operation_events,
)
from ftmwpipeline.contract import MANIFEST, WARNING_FIELDS, Absent, Stage
from ftmwpipeline.file_manager import BadSettingError
from ftmwpipeline.fitting import plan_execution as pe

pytestmark = [pytest.mark.unit]

_FORK = "fork" in multiprocessing.get_all_start_methods()


# ---- types and wire form ---------------------------------------------------


def test_event_wire_form_applies_the_absent_rule():
    ev = WindowProgress(
        "fit run", "fit", "initial", 1, 3, 7, Absent.UNDEFINED, float("nan"), 0.5, True
    )
    assert ev.stage is Stage.FIT and ev.chi2r is Absent.UNDEFINED
    assert to_jsonable(ev) == {
        "schema": "ftmw/window_progress@1",
        "operation": "fit run",
        "stage": "fit",
        "phase": "initial",
        "index": 1,
        "total": 3,
        "window_id": 7,
        "n_peaks": None,
        "n_peaks_absent": "undefined",
        "chi2r": None,
        "chi2r_absent": "undefined",
        "elapsed_s": 0.5,
        "dropped": True,
    }


def test_warning_flattens_its_code_fields_on_the_wire():
    w = PipelineWarning(
        "fit run",
        "fit",
        "slow_window",
        "slow",
        {"window_id": 1, "elapsed_s": 61.0, "threshold_s": 60.0},
    )
    assert to_jsonable(w) == {
        "schema": "ftmw/warning@1",
        "operation": "fit run",
        "stage": "fit",
        "code": "slow_window",
        "message": "slow",
        "window_id": 1,
        "elapsed_s": 61.0,
        "threshold_s": 60.0,
    }
    with pytest.raises(ValueError):
        PipelineWarning("x", None, "slow_window", "m", {"window_id": 1})
    with pytest.raises(ValueError):
        PipelineWarning("x", None, "no_such_code", "m", {})


def test_event_schemas_codes_and_vocabulary_are_declared():
    for cls in (StageStarted, StageFinished, WindowProgress, Invalidated):
        assert cls.__ftmw_schema__ in MANIFEST.schemas
    assert {"cancelled", "callback_failed"} <= set(MANIFEST.codes)
    assert set(MANIFEST.vocabularies["warning_code"]) == set(WARNING_FIELDS)


def test_threading_event_is_a_cancel_token():
    assert isinstance(threading.Event(), CancelToken)


def test_errors_round_trip_through_pickle():
    e = OperationCancelledError("fit", ["data", "ft"], [])
    back = pickle.loads(pickle.dumps(e))
    assert back.to_dict() == e.to_dict()
    assert e.to_dict()["stage"] == "fit"
    assert OperationCancelledError(None).to_dict()["stage"] is None
    c = CallbackFailedError("ftmw/warning@1")
    assert pickle.loads(pickle.dumps(c)).to_dict()["event_schema"] == "ftmw/warning@1"


# ---- OperationEvents -------------------------------------------------------


def test_stage_scope_order_and_completion():
    got = []
    ops = operation_events("fit run", got.append)
    with ops.stage(Stage.FIT, verb="fit run") as scope:
        emit_invalidated(scope, ["stage6_review"], reason="Stage stage5_fitting re-run")
        scope.finish({"n": 1})
    assert [type(e) for e in got] == [StageStarted, Invalidated, StageFinished]
    assert got[1].stages == (Stage.REVIEW,)
    assert got[2].summary == {"n": 1}
    assert ops.completed_stages == ["fit"]


def test_cancel_before_a_stage_is_between_stages():
    tok = threading.Event()
    tok.set()
    ops = operation_events("run", None, tok)
    with pytest.raises(OperationCancelledError) as info:
        with ops.stage("fit"):
            pass  # pragma: no cover - never entered
    assert info.value.stage is None


def test_cancel_inside_a_stage_names_it():
    tok = threading.Event()
    ops = operation_events("fit run", None, tok)
    with ops.stage("fit") as scope:
        tok.set()
        with pytest.raises(OperationCancelledError) as info:
            scope.check_cancel()
    assert info.value.stage == "fit"


def test_failing_callback_is_callback_failed_with_cause():
    def bad(event):
        raise ValueError("nope")

    ops = operation_events("fit run", bad)
    with pytest.raises(CallbackFailedError) as info:
        with ops.stage("fit"):
            pass  # pragma: no cover - StageStarted raises
    assert info.value.event_schema == "ftmw/stage_started@1"
    assert isinstance(info.value.__cause__, ValueError)


def test_bad_arguments_are_bad_setting():
    with pytest.raises(BadSettingError):
        operation_events("x", 5)  # type: ignore[arg-type]
    with pytest.raises(BadSettingError):
        operation_events("x", None, object())  # type: ignore[arg-type]


def test_nested_operation_reuses_the_outer_events():
    outer = operation_events("run")
    assert operation_events("fit run", outer) is outer


def test_invalidation_line_renders_without_a_listener(caplog):
    with caplog.at_level(logging.WARNING, logger="ftmwpipeline.file_manager"):
        emit_invalidated(None, ["stage6_review"], reason="Stage stage5_fitting re-run")
    assert [r.getMessage() for r in caplog.records] == [
        "Stage stage5_fitting re-run; invalidated stage(s) ['stage6_review'] -- "
        "re-run them to refresh."
    ]


# ---- the Stage 5 walk ------------------------------------------------------


def _plan():
    from tests.unit.fitting.test_plan_execution import TestParallelWalkEquivalence

    return TestParallelWalkEquivalence()._build_plan()


def _walk(workers, scope, monkeypatch):
    from tests.unit.fitting.test_plan_execution import (
        SIDEBAND,
        T_US,
        TAU_US,
        _make_active_ft,
    )

    plan, freqs, spec, noise, peak_freqs = _plan()
    monkeypatch.setattr(pe, "_FIT_WINDOW_WORKERS", workers)
    return pe.execute_plan(
        plan,
        _make_active_ft(freqs, spec),
        noise,
        peak_freqs,
        sideband=SIDEBAND,
        acquisition_us=T_US,
        tau0_us=TAU_US,
        events=scope,
    )


@pytest.mark.parametrize(
    "workers",
    [1, pytest.param(2, marks=pytest.mark.skipif(not _FORK, reason="needs fork"))],
)
def test_walk_reports_every_window_from_the_parent(workers, monkeypatch):
    got = []
    ops = operation_events("fit run", got.append)
    with ops.stage("fit") as scope:
        out = _walk(workers, scope, monkeypatch)
    progress = [e for e in got if isinstance(e, WindowProgress)]
    assert [e.index for e in progress] == [1, 2, 3]
    assert {e.window_id for e in progress} == {0, 1, 2}
    assert all(e.phase == "initial" and e.total == 3 for e in progress)
    assert set(out.window_outcomes) <= {0, 1, 2}


def test_sequential_walk_cancels_after_the_current_window(monkeypatch):
    tok = threading.Event()
    seen = []

    def cb(event):
        if isinstance(event, WindowProgress):
            seen.append(event.window_id)
            tok.set()

    ops = operation_events("fit run", cb, tok)
    with pytest.raises(OperationCancelledError) as info:
        with ops.stage("fit") as scope:
            _walk(1, scope, monkeypatch)
    assert len(seen) == 1
    assert info.value.stage == "fit" and info.value.completed_windows == []


@pytest.mark.skipif(not _FORK, reason="needs fork")
def test_parallel_cancel_does_not_wait_for_a_running_window(monkeypatch):
    original = pe._process_one_window

    def slow(win, **kwargs):
        if win.window_id == 1:
            time.sleep(60)  # terminated long before this returns
        return original(win, **kwargs)

    monkeypatch.setattr(pe, "_process_one_window", slow)
    tok = threading.Event()

    def cb(event):
        if isinstance(event, WindowProgress):
            tok.set()

    ops = operation_events("fit run", cb, tok)
    t0 = time.monotonic()
    with pytest.raises(OperationCancelledError):
        with ops.stage("fit") as scope:
            _walk(2, scope, monkeypatch)
    assert time.monotonic() - t0 < 30.0
    assert pe._WORKER_FIT_CTX is None
    deadline = time.monotonic() + 5.0
    while multiprocessing.active_children() and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not multiprocessing.active_children()


@pytest.mark.skipif(not _FORK, reason="needs fork")
def test_parallel_callback_failure_aborts_the_pool(monkeypatch):
    def bad(event):
        if isinstance(event, WindowProgress):
            raise RuntimeError("boom")

    ops = operation_events("fit run", bad)
    with pytest.raises(CallbackFailedError) as info:
        with ops.stage("fit") as scope:
            _walk(2, scope, monkeypatch)
    assert info.value.event_schema == "ftmw/window_progress@1"
    assert pe._WORKER_FIT_CTX is None


def test_detached_scope_renders_without_delivery(caplog):
    scope = detached_scope(Stage.FIT, verb="fit run")
    with caplog.at_level(logging.INFO, logger=pe.logger.name):
        pe._report_window(
            scope,
            pe._WindowReport(3, (1.0, 2.0), 2, 1.5, 0.1),
            phase="replan",
            index=1,
            total=4,
        )
    assert [r.getMessage() for r in caplog.records if r.name == pe.logger.name] == [
        "w3 [1.0-2.0 MHz]: 2 peaks, chi2r=1.5, 0.1s",
        "window 1/4",
    ]
