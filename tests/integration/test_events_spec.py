"""Events and cancellation on real stages (CONTRACT_STRATEGY §Events and
cancellation), checked from the spec.

Covers, on the 2638 baselines: the ordering rule of a stage's events,
``StageFinished.summary`` against the verb's ``run_result`` summary, a cancel
before a stage and in the middle of the Stage 5 walk (sequential and pooled)
leaving the file as it was and no worker behind, a failing callback, callbacks
on the calling thread, a cancel between ``run_pipeline`` stages and between scan
values, the per-window log lines, the CLI's ``--events`` and exit 130, and that
the functional API, ``Pipeline`` and the CLI emit the same event sequence.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
import re
import shutil
import signal
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import (
    CallbackFailedError,
    Invalidated,
    OperationCancelledError,
    ScanProgress,
    StageFinished,
    StageStarted,
    WindowProgress,
    to_jsonable,
)
from ftmwpipeline.cli import _events as cli_events
from ftmwpipeline.cli.main import main as cli_main
from ftmwpipeline.contract import Stage
from ftmwpipeline.pipeline import Pipeline
from tests._events_support import (
    Recorder,
    Token,
    assert_stage_order,
    cancel_on_nth,
    content_digest,
    wait_no_new_children,
    wire_sequence,
)

pytestmark = [pytest.mark.integration]

_FORK = "fork" in multiprocessing.get_all_start_methods()
_NEW_TRIM = (27000.0, 39000.0)

EVENT_SCHEMAS = {
    "ftmw/stage_started@1",
    "ftmw/stage_finished@1",
    "ftmw/window_progress@1",
    "ftmw/scan_progress@1",
    "ftmw/invalidated@1",
    "ftmw/warning@1",
}


def _copy(src: Path, tmp_path: Path, name: str = "work.ftmw") -> Path:
    dest = tmp_path / name
    shutil.copy(src, dest)
    return dest


def _states(path: Path) -> Dict[str, str]:
    return {row["stage"]: row["state"] for row in ftmw.status(path)["stages"]}


def _cli(argv: List[str], capsys) -> Tuple[int, str, str]:
    capsys.readouterr()
    rc = cli_main(argv)
    out, err = capsys.readouterr()
    return rc, out, err


def _event_lines(err: str) -> List[Dict[str, Any]]:
    """The stderr lines that are events (``--events`` writes one JSON object per
    line; other stderr lines, such as log records, are skipped)."""
    events: List[Dict[str, Any]] = []
    for line in err.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict) and obj.get("schema") in EVENT_SCHEMAS:
            events.append(obj)
    return events


def _last_json(err: str) -> Dict[str, Any]:
    lines = [ln for ln in err.splitlines() if ln.strip().startswith("{")]
    assert lines, f"no JSON line on stderr: {err!r}"
    return json.loads(lines[-1])


# One row per stage verb that is cheap enough to run here:
# (name, baseline fixture, CLI argv after the file, API call, operation, stage).
STAGE_CASES = {
    "noise": (
        "baseline_2638_stage1_raw",
        ["noise", "run"],
        [],
        lambda p, **k: ftmw.estimate_noise(p, **k),
        "noise run",
        "noise",
    ),
    "peaks": (
        "baseline_2638_stage2",
        ["peaks", "run"],
        [],
        lambda p, **k: ftmw.detect_peaks(p, **k),
        "peaks run",
        "peaks",
    ),
    "windows": (
        "baseline_2638_stage3",
        ["windows", "run"],
        [],
        lambda p, **k: ftmw.assign_windows(p, **k),
        "windows run",
        "windows",
    ),
    "fit": (
        "baseline_2638_stage4_small",
        ["fit", "run"],
        ["--jobs", "1"],
        lambda p, **k: ftmw.fit_peaks(p, jobs=1, **k),
        "fit run",
        "fit",
    ),
}


# ---- ordering ---------------------------------------------------------------------


def test_a_changed_input_orders_started_invalidated_finished(
    baseline_2638_stage2, tmp_path
):
    fp = _copy(baseline_2638_stage2, tmp_path)
    rec = Recorder()
    result = ftmw.compute_ft(fp, trim=_NEW_TRIM, events=rec)
    assert_stage_order(rec.events)
    inv = rec.of(Invalidated)
    assert len(inv) == 1
    # The event matches the result's field, in rerun_order.
    assert [s.value for s in inv[0].stages] == list(result.invalidated) == ["noise"]
    assert [type(e) for e in rec.events][-2:] == [Invalidated, StageFinished]
    assert all(e.operation == "ft run" and e.stage is Stage.FT for e in rec.events)


@pytest.mark.parametrize("case", sorted(STAGE_CASES))
def test_a_stage_run_emits_one_ordered_bracket(case, request, tmp_path):
    fixture, _argv, _extra, call, operation, stage = STAGE_CASES[case]
    fp = _copy(request.getfixturevalue(fixture), tmp_path)
    rec = Recorder()
    call(fp, events=rec)
    assert_stage_order(rec.events)
    assert all(e.operation == operation and e.stage == stage for e in rec.events)
    assert _states(fp)[stage] == "complete"
    if case == "fit":
        windows = rec.of(WindowProgress)
        assert windows, "the fit walk reports each window"
        assert windows[0].phase == "initial" and windows[0].round == 0
        assert all(w.phase in ("initial", "replan", "fallback") for w in windows)
        # Each pass (phase, round) counts its own windows from 1 to a total
        # fixed when it begins; initial and its fallback are round 0, a replan
        # round and its fallback are that round's number (from 1).
        passes: dict = {}
        for w in windows:
            passes.setdefault((w.phase, w.round), []).append(w)
        for ws in passes.values():
            assert [w.index for w in ws] == list(range(1, len(ws) + 1))
            assert {w.total for w in ws} == {len(ws)}
        replan_rounds = {rnd for phase, rnd in passes if phase == "replan"}
        for phase, rnd in passes:
            if phase == "initial":
                assert rnd == 0
            elif phase == "replan":
                assert rnd >= 1
            else:  # fallback: of the initial walk or of a replan round
                assert rnd == 0 or rnd in replan_rounds


@pytest.mark.parametrize("case", sorted(STAGE_CASES))
def test_stage_finished_summary_has_the_run_result_summary_keys(
    case, request, tmp_path, capsys
):
    """``--events --json``: the stage's StageFinished and the verb's
    ``ftmw/run_result@1`` come from one builder."""
    fixture, argv, extra, _call, operation, stage = STAGE_CASES[case]
    fp = _copy(request.getfixturevalue(fixture), tmp_path)
    rc, out, err = _cli([*argv, str(fp), *extra, "--events", "--json"], capsys)
    assert rc == 0, err
    run_result = json.loads(out)
    assert run_result["schema"] == "ftmw/run_result@1"
    events = _event_lines(err)
    finished = [e for e in events if e["schema"] == "ftmw/stage_finished@1"]
    assert len(finished) == 1
    fin = finished[0]
    assert fin["operation"] == run_result["verb"] == operation
    assert fin["stage"] == run_result["stage"] == stage
    assert set(fin["summary"]) == set(run_result["summary"])
    assert fin["elapsed_s"] >= 0.0
    # Every stderr event line is a complete schema-stamped object, started first,
    # finished last.
    assert events[0]["schema"] == "ftmw/stage_started@1"
    assert events[-1]["schema"] == "ftmw/stage_finished@1"


def _assert_summary_declared(operation: str, summary: Dict[str, Any]) -> None:
    """*summary* has every required key of *operation*'s declaration and no key
    the declaration does not name."""
    declared = ftmw.capabilities()["summary_keys"][operation]
    required, conditional = set(declared["required"]), set(declared["conditional"])
    assert required <= set(summary), required - set(summary)
    assert set(summary) <= required | conditional, set(summary) - required - conditional


def _finished_summary(rec: Recorder, operation: str) -> Dict[str, Any]:
    (fin,) = [e for e in rec.of(StageFinished) if e.operation == operation]
    return dict(to_jsonable(fin)["summary"])


@pytest.mark.parametrize("case", sorted(STAGE_CASES))
def test_stage_finished_summary_keys_are_declared(case, request, tmp_path):
    """``capabilities()["summary_keys"]`` covers the stage's emitted summary."""
    fixture, _argv, _extra, call, operation, _stage = STAGE_CASES[case]
    fp = _copy(request.getfixturevalue(fixture), tmp_path)
    rec = Recorder()
    call(str(fp), events=rec)
    _assert_summary_declared(operation, _finished_summary(rec, operation))


def test_ft_and_start_summary_keys_are_declared(baseline_2638_stage1_raw, tmp_path):
    fp = _copy(baseline_2638_stage1_raw, tmp_path)
    rec = Recorder()
    ftmw.compute_ft(str(fp), from_saved_params=True, events=rec)
    _assert_summary_declared("ft run", _finished_summary(rec, "ft run"))
    rec = Recorder()
    ftmw.detect_start_time(str(fp), stamp=False, events=rec)
    _assert_summary_declared("start run", _finished_summary(rec, "start run"))


def test_review_summary_keys_are_declared(baseline_2638_stage5_small, tmp_path):
    fp = _copy(baseline_2638_stage5_small, tmp_path)
    rec = Recorder()
    ftmw.review_run(str(fp), events=rec)
    _assert_summary_declared("review run", _finished_summary(rec, "review run"))
    window_id = int(ftmw.load_fit(str(fp)).window_fits[0].window_id)
    rec = Recorder()
    ftmw.review_accept(str(fp), window_id, events=rec)
    _assert_summary_declared("review accept", _finished_summary(rec, "review accept"))


# ---- cancel before a stage ---------------------------------------------------------------


@pytest.mark.parametrize("case", ["noise", "peaks", "fit"])
@pytest.mark.parametrize("via", ["api", "pipeline"])
def test_a_token_set_before_a_stage_cancels_it_and_leaves_the_file(
    case, via, request, tmp_path
):
    fixture, _argv, _extra, call, operation, stage = STAGE_CASES[case]
    fp = _copy(request.getfixturevalue(fixture), tmp_path)
    before = content_digest(fp)
    states_before = _states(fp)
    tok = Token()
    tok.set()
    rec = Recorder()
    with pytest.raises(OperationCancelledError) as info:
        if via == "api":
            call(fp, events=rec, cancel=tok)
        else:
            method = {
                "noise": "estimate_noise",
                "peaks": "detect_peaks",
                "fit": "fit_peaks",
            }[case]
            getattr(Pipeline.open(fp), method)(events=rec, cancel=tok)
    err = info.value
    assert err.code == "cancelled"
    assert err.completed_stages == [] and err.completed_windows == []
    # It fell before the stage began, so it was not an interrupted stage.
    assert err.stage is None
    assert not rec.of(StageFinished)
    assert content_digest(fp) == before
    assert _states(fp) == states_before


# ---- cancel in the Stage 5 walk -------------------------------------------------------------


def _fit_cancelled(fp: Path, *, jobs: int):
    tok = Token()
    rec = Recorder(cancel_on_nth(WindowProgress, 1, tok))
    before_children = set(multiprocessing.active_children())
    t0 = time.monotonic()
    with pytest.raises(OperationCancelledError) as info:
        ftmw.fit_peaks(fp, jobs=jobs, events=rec, cancel=tok)
    t_raised = time.monotonic()
    return info.value, tok, rec, t0, t_raised, before_children


@pytest.mark.parametrize(
    "jobs",
    [1, pytest.param(2, marks=pytest.mark.skipif(not _FORK, reason="needs fork"))],
)
def test_fit_cancelled_mid_walk_returns_promptly_and_keeps_a_partial_fit(
    jobs, baseline_2638_stage4_small, tmp_path
):
    fp = _copy(baseline_2638_stage4_small, tmp_path)
    before = content_digest(fp)
    err, tok, rec, _t0, t_raised, children = _fit_cancelled(fp, jobs=jobs)

    assert err.code == "cancelled"
    assert err.stage == "fit"
    # The windows that finished (each reported its WindowProgress) are kept as
    # a partial fit (Wave 5.2), and the error lists them.
    reported = sorted({e.window_id for e in rec.of(WindowProgress)})
    assert reported and err.completed_windows == reported
    assert err.completed_stages == []
    # Prompt: the parent polls every ~0.2 s and does not wait for a running
    # window, so the cancel reaches the caller well inside this bound.
    assert tok.set_at is not None and t_raised - tok.set_at < 10.0
    if jobs == 1:
        # The sequential walk honours a cancel after the current window.
        assert len(rec.of(WindowProgress)) == 1
    assert not rec.of(StageFinished)
    assert content_digest(fp) != before
    assert _states(fp)["fit"] == "partial"
    # No worker left behind.
    assert not wait_no_new_children(children)


@pytest.mark.parametrize(
    "jobs",
    [1, pytest.param(2, marks=pytest.mark.skipif(not _FORK, reason="needs fork"))],
)
def test_fit_callback_failure_aborts_like_a_cancel(
    jobs, baseline_2638_stage4_small, tmp_path
):
    fp = _copy(baseline_2638_stage4_small, tmp_path)
    before = content_digest(fp)
    children = set(multiprocessing.active_children())
    boom = RuntimeError("listener broke")

    def bad(event):
        if isinstance(event, WindowProgress):
            raise boom

    with pytest.raises(CallbackFailedError) as info:
        ftmw.fit_peaks(fp, jobs=jobs, events=bad)
    assert info.value.code == "callback_failed"
    assert info.value.event_schema == "ftmw/window_progress@1"
    assert info.value.__cause__ is boom
    # The window whose WindowProgress failed had finished: it is kept as a
    # partial fit, as after a cancel at that point.
    assert content_digest(fp) != before
    assert _states(fp)["fit"] == "partial"
    assert not wait_no_new_children(children)


def test_a_failing_callback_before_any_write_leaves_the_file(
    baseline_2638_stage1_raw, tmp_path
):
    fp = _copy(baseline_2638_stage1_raw, tmp_path)
    before = content_digest(fp)

    def bad(event):
        raise ValueError("no")

    with pytest.raises(CallbackFailedError) as info:
        ftmw.estimate_noise(fp, events=bad)
    assert info.value.event_schema == "ftmw/stage_started@1"
    assert isinstance(info.value.__cause__, ValueError)
    assert content_digest(fp) == before


# ---- callbacks run on the calling thread ---------------------------------------------------------


@pytest.mark.skipif(not _FORK, reason="needs fork")
def test_pooled_fit_delivers_every_event_on_the_calling_thread_and_process(
    baseline_2638_stage4_small, tmp_path
):
    fp = _copy(baseline_2638_stage4_small, tmp_path)
    rec = Recorder()
    ftmw.fit_peaks(fp, jobs=2, events=rec)
    assert rec.of(WindowProgress)
    assert set(rec.threads) == {threading.get_ident()}
    # Events from pool work are passed back to the parent first.
    assert set(rec.pids) == {os.getpid()}


def test_events_arrive_on_the_thread_that_made_the_call(
    baseline_2638_stage1_raw, tmp_path
):
    fp = _copy(baseline_2638_stage1_raw, tmp_path)
    rec = Recorder()
    caller: Dict[str, int] = {}
    errors: List[BaseException] = []

    def work() -> None:
        caller["ident"] = threading.get_ident()
        try:
            ftmw.estimate_noise(fp, events=rec)
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    t = threading.Thread(target=work)
    t.start()
    t.join()
    assert not errors, errors
    assert rec.events
    assert set(rec.threads) == {caller["ident"]}
    assert caller["ident"] != threading.get_ident()


# ---- run_pipeline -----------------------------------------------------------------------------------


@pytest.mark.parametrize("trigger", ["finished", "started"])
def test_run_pipeline_cancel_between_stages_keeps_the_completed_stages(
    trigger, exp_2638_data_path, tmp_path
):
    """A stage with no windows checks the token only before it starts: once the
    noise stage has begun it completes, and the cancel is honoured at the next
    check point, between stages."""
    out = tmp_path / "run.ftmw"
    tok = Token()

    def act(event):
        if event.stage is Stage.NOISE and isinstance(
            event, StageFinished if trigger == "finished" else StageStarted
        ):
            tok.set()

    rec = Recorder(act)
    with pytest.raises(OperationCancelledError) as info:
        ftmw.run_pipeline(
            exp_2638_data_path,
            out,
            trim=(26500.0, 40000.0),
            detect_start=False,
            calibrate=False,
            progress=False,
            events=rec,
            cancel=tok,
        )
    err = info.value
    assert err.stage is None, "the cancel fell between stages"
    assert err.completed_stages == ["data", "ft", "noise"]
    assert err.completed_windows == []
    # The completed stages stay as written; nothing later began.
    states = _states(out)
    assert [states[s] for s in ("data", "ft", "noise")] == ["complete"] * 3
    assert states["tau"] == "not_run" and states["peaks"] == "not_run"
    assert all(e.operation == "run" for e in rec.events)
    done = [Stage.DATA, Stage.FT, Stage.NOISE]
    assert [e.stage for e in rec.of(StageStarted)] == done
    assert [e.stage for e in rec.of(StageFinished)] == done


# ---- scans -------------------------------------------------------------------------------------------------

_KNOB = "stage2.window_mhz"
_GRID = [40.0, 80.0]


def test_scan_reports_each_value_and_cancels_between_values(
    baseline_2638_stage1_raw, tmp_path
):
    fp = baseline_2638_stage1_raw
    before = content_digest(fp)

    rec = Recorder()
    ftmw.scan_run(
        fp,
        _KNOB,
        grid=_GRID,
        output_dir=tmp_path / "full",
        make_plot=False,
        quiet=True,
        events=rec,
    )
    progress = rec.of(ScanProgress)
    assert [(p.index, p.total) for p in progress] == [(1, 2), (2, 2)]
    assert [p.value for p in progress] == _GRID
    assert {p.knob for p in progress} == {_KNOB}
    assert {p.operation for p in progress} == {"scan run"}

    tok = Token()
    rec = Recorder(cancel_on_nth(ScanProgress, 1, tok))
    with pytest.raises(OperationCancelledError) as info:
        ftmw.scan_run(
            fp,
            _KNOB,
            grid=_GRID,
            output_dir=tmp_path / "cancelled",
            make_plot=False,
            quiet=True,
            events=rec,
            cancel=tok,
        )
    assert len(rec.of(ScanProgress)) == 1  # stopped before the second value
    assert info.value.completed_windows == []
    assert content_digest(fp) == before  # the input is never mutated


def test_a_cancel_is_not_swallowed_by_scan_all_failure_isolation(
    baseline_2638_stage1_raw, tmp_path
):
    tok = Token()
    rec = Recorder(cancel_on_nth(ScanProgress, 1, tok))
    with pytest.raises(OperationCancelledError):
        ftmw.scan_all(
            baseline_2638_stage1_raw,
            "stage2",
            output_dir=tmp_path / "all",
            make_plot=False,
            quiet=True,
            events=rec,
            cancel=tok,
        )


# ---- log rendering ---------------------------------------------------------------------------------------------

_DETAIL = re.compile(
    r"^w(?P<wid>\d+) \[[-\d.]+-[-\d.]+ MHz\]: (?P<n>\d+) peaks, "
    r"chi2r=(?P<chi2r>\S+), (?P<elapsed>[\d.]+)s$"
)
_DROPPED = re.compile(
    r"^w(?P<wid>\d+) \[[-\d.]+-[-\d.]+ MHz\]: dropped \(cascaded to empty\)$"
)
_PROGRESS = re.compile(r"^window (?P<n>\d+)/(?P<total>\d+)$")


@pytest.mark.parametrize(
    "jobs",
    [1, pytest.param(2, marks=pytest.mark.skipif(not _FORK, reason="needs fork"))],
)
def test_per_window_lines_keep_their_text_and_are_logged_once_per_window(
    jobs, baseline_2638_stage4_small, tmp_path, caplog
):
    fp = _copy(baseline_2638_stage4_small, tmp_path)
    rec = Recorder()
    with caplog.at_level(logging.INFO, logger="ftmwpipeline.fitting.plan_execution"):
        ftmw.fit_peaks(fp, jobs=jobs, events=rec)
    progress = rec.of(WindowProgress)
    assert progress

    messages = [r.getMessage() for r in caplog.records]
    detail = [m for m in messages if _DETAIL.match(m) or _DROPPED.match(m)]
    counters = [m for m in messages if _PROGRESS.match(m)]
    # One detail (or dropped) line and one "window n/total" line per window the
    # events reported -- not zero (rendering lost) and not two (rendered both
    # in a worker and in the parent).
    assert len(detail) == len(progress)
    assert len(counters) == len(progress)

    def key(m: str) -> int:
        return int((_DETAIL.match(m) or _DROPPED.match(m)).group("wid"))  # type: ignore[union-attr]

    assert sorted(key(m) for m in detail) == sorted(e.window_id for e in progress)

    by_id: Dict[int, List[str]] = {}
    for m in detail:
        by_id.setdefault(key(m), []).append(m)
    for e in progress:
        line = by_id[e.window_id].pop(0)
        if e.dropped:
            assert _DROPPED.match(line), line
        else:
            m = _DETAIL.match(line)
            assert m, line
            assert int(m.group("n")) == e.n_peaks
            if isinstance(e.chi2r, float):
                assert m.group("chi2r") == f"{e.chi2r:.3g}"
            else:  # no finite value: the line has always printed nan
                assert m.group("chi2r") == "nan"
            assert m.group("elapsed") == f"{e.elapsed_s:.1f}"
    assert {(e.index, e.total) for e in progress} == {
        (int(m.group("n")), int(m.group("total")))
        for m in map(_PROGRESS.match, counters)
        if m
    }


# ---- CLI --------------------------------------------------------------------------------------------------------


def test_cli_events_are_json_lines_on_stderr_and_stdout_stays_clean(
    baseline_2638_stage1_raw, tmp_path, capsys
):
    fp = _copy(baseline_2638_stage1_raw, tmp_path)
    rc, out, err = _cli(["noise", "run", str(fp), "--events"], capsys)
    assert rc == 0, err
    events = _event_lines(err)
    assert [e["schema"] for e in events] == [
        "ftmw/stage_started@1",
        "ftmw/stage_finished@1",
    ]
    for e in events:
        assert e["operation"] == "noise run" and e["stage"] == "noise"
    assert "ftmw/stage_started@1" not in out  # events never go to stdout


def test_cli_without_events_writes_no_event_lines(
    baseline_2638_stage1_raw, tmp_path, capsys
):
    fp = _copy(baseline_2638_stage1_raw, tmp_path)
    rc, _out, err = _cli(["noise", "run", str(fp)], capsys)
    assert rc == 0
    assert _event_lines(err) == []


class _AlwaysSet:
    """A cancel token that is already set (what the first Ctrl-C leaves)."""

    def set(self) -> None:
        pass

    def is_set(self) -> bool:
        return True


def test_cli_cancel_exits_130_and_prints_the_error_dict(
    baseline_2638_stage1_raw, tmp_path, capsys, monkeypatch
):
    fp = _copy(baseline_2638_stage1_raw, tmp_path)
    before = content_digest(fp)
    monkeypatch.setattr(cli_events, "SignalCancelToken", _AlwaysSet)
    rc, out, err = _cli(["noise", "run", str(fp), "--json", "--events"], capsys)
    assert rc == 130
    assert out.strip() == ""  # no run_result for a cancelled verb
    error = _last_json(err)  # under --json the error is the last stderr line
    assert error["schema"] == "ftmw/error@1" and error["code"] == "cancelled"
    assert error["completed_stages"] == [] and error["completed_windows"] == []
    assert content_digest(fp) == before


def test_cli_cancel_in_text_mode_also_exits_130(
    baseline_2638_stage1_raw, tmp_path, capsys, monkeypatch
):
    fp = _copy(baseline_2638_stage1_raw, tmp_path)
    monkeypatch.setattr(cli_events, "SignalCancelToken", _AlwaysSet)
    rc, _out, err = _cli(["noise", "run", str(fp)], capsys)
    assert rc == 130
    assert "cancel" in err.lower()


def test_cli_fit_cancelled_mid_walk_exits_130_and_keeps_a_partial_fit(
    baseline_2638_stage4_small, tmp_path, capsys, monkeypatch
):
    """The first Ctrl-C, modelled as a token that the events writer sets on the
    first window: the verb stops at its next check point."""
    fp = _copy(baseline_2638_stage4_small, tmp_path)
    before = content_digest(fp)
    state = {"cancel": False}
    real_write = cli_events.write_event

    def write_and_cancel(event):
        real_write(event)
        if isinstance(event, WindowProgress):
            state["cancel"] = True

    class Flag:
        def set(self) -> None:
            state["cancel"] = True

        def is_set(self) -> bool:
            return state["cancel"]

    monkeypatch.setattr(cli_events, "write_event", write_and_cancel)
    monkeypatch.setattr(cli_events, "SignalCancelToken", Flag)
    rc, out, err = _cli(
        ["fit", "run", str(fp), "--jobs", "1", "--events", "--json"], capsys
    )
    assert rc == 130, err
    assert out.strip() == ""
    events = _event_lines(err)
    progress = [e for e in events if e["schema"] == "ftmw/window_progress@1"]
    assert len(progress) == 1
    assert not [e for e in events if e["schema"] == "ftmw/stage_finished@1"]
    error = _last_json(err)
    assert error["code"] == "cancelled" and error["stage"] == "fit"
    assert error["completed_windows"] == [progress[0]["window_id"]]
    assert content_digest(fp) != before
    assert _states(fp)["fit"] == "partial"


@pytest.mark.skipif(
    threading.current_thread() is not threading.main_thread(),
    reason="signal handlers need the main thread",
)
def test_first_interrupt_cancels_and_a_second_interrupts_at_once():
    token = cli_events.SignalCancelToken()
    previous = signal.getsignal(signal.SIGINT)
    with cli_events.first_interrupt_cancels(token, announce=False):
        assert not token.is_set()
        signal.raise_signal(signal.SIGINT)  # the first Ctrl-C
        assert token.is_set()
        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGINT)  # the second one
    assert signal.getsignal(signal.SIGINT) is previous


# ---- one event schema sequence on all three interfaces ----------------------------------------------------------


@pytest.mark.parametrize("case", ["noise", "fit"])
def test_api_pipeline_and_cli_emit_the_same_event_sequence(
    case, request, tmp_path, capsys
):
    fixture, argv, extra, call, operation, stage = STAGE_CASES[case]
    base = request.getfixturevalue(fixture)

    fa = _copy(base, tmp_path, "api.ftmw")
    rec_api = Recorder()
    call(fa, events=rec_api)

    fp = _copy(base, tmp_path, "pipe.ftmw")
    rec_pipe = Recorder()
    method = {"noise": "estimate_noise", "fit": "fit_peaks"}[case]
    kwargs: Dict[str, Any] = {"jobs": 1} if case == "fit" else {}
    getattr(Pipeline.open(fp), method)(events=rec_pipe, **kwargs)

    fc = _copy(base, tmp_path, "cli.ftmw")
    rc, _out, err = _cli([*argv, str(fc), *extra, "--events"], capsys)
    assert rc == 0, err
    cli_events_seen = _event_lines(err)

    api_seq = wire_sequence(rec_api.events)
    assert api_seq == wire_sequence(rec_pipe.events)
    assert api_seq == wire_sequence(cli_events_seen)
    assert api_seq[0][0] == "ftmw/stage_started@1"
    assert api_seq[-1][0] == "ftmw/stage_finished@1"
    assert all(row[1] == operation and row[2] == stage for row in api_seq)
    # The three wire forms agree key for key, not only in sequence.
    for e_api, e_cli in zip(rec_api.events, cli_events_seen):
        assert set(to_jsonable(e_api)) == set(e_cli)
