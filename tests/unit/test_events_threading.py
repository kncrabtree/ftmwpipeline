"""Events and cancellation threaded through every long operation.

CONTRACT_STRATEGY §Events and cancellation: every long operation takes
``events`` / ``cancel`` on all three interfaces, every long CLI verb takes
``--events``, and the plumbing the threading relies on (non-stage scopes, one
combined ``Invalidated``, a block that completes once begun, the
once-per-operation ``environment_drift`` check, the report pool's cancel).
"""

from __future__ import annotations

import argparse
import inspect
import json
import multiprocessing
import time
from pathlib import Path
from typing import Dict, List

import h5py
import pytest

import ftmwpipeline.api as api
from ftmwpipeline import (
    CallbackFailedError,
    Invalidated,
    OperationCancelledError,
    PipelineWarning,
    StageFinished,
    StageStarted,
)
from ftmwpipeline._internal.events import (
    STAGE_END_LINES,
    STAGE_START_LINES,
    WARNING_LINES,
    operation_events,
)
from ftmwpipeline._internal.report_html_impl import fork_map
from ftmwpipeline.cli.main import create_parser
from ftmwpipeline.contract import Stage
from ftmwpipeline.core.environment import capture_environment
from ftmwpipeline.pipeline import Pipeline

pytestmark = [pytest.mark.unit]

_FORK = "fork" in multiprocessing.get_all_start_methods()

#: The spec's long operations: (api function, Pipeline method).
LONG_OPERATIONS = [
    ("import_data", "create"),
    ("detect_start_time", "detect_start_time"),
    ("compute_ft", "compute_ft"),
    ("estimate_noise", "estimate_noise"),
    ("calibrate_tau", "calibrate_tau"),
    ("recommend_shape", "recommend_shape"),
    ("calibrate_timebase", "calibrate_timebase"),
    ("detect_peaks", "detect_peaks"),
    ("assign_windows", "assign_windows"),
    ("fit_peaks", "fit_peaks"),
    ("review_run", "review_run"),
    ("review_apply", "review_apply"),
    ("review_preview", "review_preview"),
    ("review_accept", "review_accept"),
    ("review_edit", "review_edit"),
    ("review_create", "review_create"),
    ("review_undo", "review_undo"),
    ("report_run", "report_run"),
    ("scan_run", "scan_run"),
    ("scan_all", "scan_all"),
    ("run_pipeline", "build"),
]

#: The long CLI verbs.
LONG_VERBS = [
    "data import",
    "start run",
    "ft run",
    "noise run",
    "tau run",
    "tau recommend",
    "timebase run",
    "peaks run",
    "windows run",
    "fit run",
    "review run",
    "review apply",
    "review preview",
    "review accept",
    "review edit",
    "review create",
    "review undo",
    "report run",
    "scan run",
    "scan all",
    "run",
]


@pytest.mark.parametrize("api_name,pipeline_name", LONG_OPERATIONS)
def test_every_long_operation_takes_events_and_cancel(api_name, pipeline_name):
    for fn in (getattr(api, api_name), getattr(Pipeline, pipeline_name)):
        params = inspect.signature(fn).parameters
        assert "events" in params and "cancel" in params, fn.__qualname__
        assert params["events"].default is None
        assert params["cancel"].default is None


def _verb_parsers() -> Dict[str, argparse.ArgumentParser]:
    found: Dict[str, argparse.ArgumentParser] = {}

    def walk(p: argparse.ArgumentParser, path: List[str]) -> None:
        for action in p._actions:
            if isinstance(action, argparse._SubParsersAction):
                for name, sub in action.choices.items():
                    walk(sub, path + [name])
                return
        found[" ".join(path)] = p

    walk(create_parser(), [])
    return found


@pytest.mark.parametrize("verb", LONG_VERBS)
def test_every_long_verb_takes_events(verb):
    parser = _verb_parsers()[verb]
    assert "--events" in parser._option_string_actions, verb


def test_stage_lines_are_keyed_by_existing_verbs():
    verbs = set(_verb_parsers())
    assert set(STAGE_START_LINES) <= verbs
    assert set(STAGE_END_LINES) <= verbs
    assert {"epoch_acknowledged", "slow_window", "walk_fallback"} <= set(WARNING_LINES)


# ---- plumbing ----------------------------------------------------------------


def test_a_step_that_is_not_a_stage_records_no_completion():
    seen: list = []
    ops = operation_events("start run", seen.append)
    with ops.stage(None, verb="start run") as scope:
        scope.finish({"x": 1})
    assert [type(e) for e in seen] == [StageStarted, StageFinished]
    assert all(e.stage is None for e in seen)
    assert ops.completed_stages == []


def test_finish_without_a_write_records_no_completion():
    ops = operation_events("review preview")
    with ops.stage(Stage.REVIEW) as scope:
        scope.finish({}, wrote=False)
    assert ops.completed_stages == []


def test_invalidations_in_several_steps_are_one_event(caplog):
    seen: list = []
    ops = operation_events("data import", seen.append)
    with ops.stage(Stage.DATA) as scope:
        with scope.collect_invalidations():
            scope.invalidated([Stage.FIT], reason="first")
            scope.invalidated([Stage.FT, Stage.FIT], reason="second")
        scope.finish({})
    inv = [e for e in seen if isinstance(e, Invalidated)]
    assert len(inv) == 1
    assert list(inv[0].stages) == [Stage.FT, Stage.FIT]
    # Each step logs its own line as it invalidates; only the event merges.
    lines = [r.getMessage() for r in caplog.records if "invalidated stage" in r.msg]
    assert len(lines) == 2
    assert lines[0].startswith("first; ") and lines[1].startswith("second; ")


def test_a_step_without_a_reason_logs_nothing_but_joins_the_event(caplog):
    seen: list = []
    ops = operation_events("data import", seen.append)
    with ops.stage(Stage.DATA) as scope:
        with scope.collect_invalidations():
            scope.invalidated([Stage.FIT], reason=None)
        scope.finish({})
    inv = [e for e in seen if isinstance(e, Invalidated)]
    assert len(inv) == 1 and list(inv[0].stages) == [Stage.FIT]
    assert not [r for r in caplog.records if "invalidated stage" in r.msg]


def test_collected_invalidation_is_delivered_when_a_later_step_raises():
    seen: list = []
    ops = operation_events("data import", seen.append)
    scope = ops.detached_scope(Stage.DATA)
    with pytest.raises(KeyError):
        with scope.collect_invalidations():
            scope.invalidated([Stage.FIT], reason="first")
            raise KeyError("later step")
    inv = [e for e in seen if isinstance(e, Invalidated)]
    assert len(inv) == 1 and list(inv[0].stages) == [Stage.FIT]


def test_a_raising_callback_does_not_mask_the_steps_exception():
    def cb(event):
        raise RuntimeError("listener broke")

    ops = operation_events("data import", cb)
    scope = ops.detached_scope(Stage.DATA)
    with pytest.raises(KeyError):
        with scope.collect_invalidations():
            scope.invalidated([Stage.FIT], reason="first")
            raise KeyError("later step")


def test_committing_defers_cancel_and_callback_failure():
    class Token:
        flag = False

        def is_set(self) -> bool:
            return self.flag

    token = Token()
    delivered: list = []

    def cb(event):
        delivered.append(event)
        raise RuntimeError("listener broke")

    ops = operation_events("review undo", cb, token)
    scope = ops.detached_scope(Stage.REVIEW)
    token.flag = True
    with pytest.raises(CallbackFailedError) as info:
        with scope.committing():
            scope.check_cancel()  # not honoured inside the block
            assert not scope.cancel_requested()
            scope.invalidated([Stage.FIT], reason="r")  # raises, held
            scope.invalidated([Stage.FIT], reason="r")  # not delivered
            reached = True
    assert reached
    assert len(delivered) == 1
    assert isinstance(info.value.__cause__, RuntimeError)
    with pytest.raises(OperationCancelledError):
        scope.check_cancel()  # honoured at the next check point


def _stamp_environments(path: Path, epoch_offset: int) -> None:
    rec = capture_environment().to_dict()
    rec["analysis_epoch"] = int(rec["analysis_epoch"]) + epoch_offset
    with h5py.File(path, "w") as f:
        g = f.create_group("pipeline_stages")
        g.attrs["stage_environments"] = json.dumps({"stage5_fitting": rec})


def test_environment_drift_is_reported_once_per_operation(tmp_path):
    fp = tmp_path / "drift.ftmw"
    _stamp_environments(fp, 1)
    seen: list = []
    ops = operation_events("fit run", seen.append)
    with ops.stage(Stage.FIT, file_path=fp):
        pass
    with ops.stage(Stage.REVIEW, file_path=fp):
        pass
    warnings = [e for e in seen if isinstance(e, PipelineWarning)]
    assert len(warnings) == 1
    assert warnings[0].code == "environment_drift"
    assert "analysis_epoch" in warnings[0].details["fields"]
    assert warnings[0].stage is Stage.FIT


def test_no_environment_drift_for_a_matching_file(tmp_path):
    fp = tmp_path / "same.ftmw"
    _stamp_environments(fp, 0)
    seen: list = []
    with operation_events("fit run", seen.append).stage(Stage.FIT, file_path=fp):
        pass
    assert not [e for e in seen if isinstance(e, PipelineWarning)]


# ---- the report's per-window pool --------------------------------------------


def _slow_item(i: int) -> int:
    time.sleep(0.05 if i == 0 else 30.0)
    return i


def test_fork_map_serial_check_runs_before_each_item():
    calls: list = []

    def check() -> None:
        calls.append(len(calls))
        if len(calls) == 3:
            raise OperationCancelledError(None, [], [])

    out: list = []
    with pytest.raises(OperationCancelledError):
        fork_map([1, 2, 3, 4], lambda x: out.append(x) or x, jobs=1, check=check)
    assert out == [1, 2]


@pytest.mark.skipif(not _FORK, reason="needs fork")
def test_fork_map_cancel_does_not_wait_for_a_running_item():
    done: list = []

    def check() -> None:
        if done:
            raise OperationCancelledError(None, [], [])

    t0 = time.monotonic()
    with pytest.raises(OperationCancelledError):
        fork_map(
            [0, 1, 2],
            _slow_item,
            jobs=2,
            override=2,
            progress=lambda i, n: done.append(i),
            check=check,
        )
    assert time.monotonic() - t0 < 10.0
