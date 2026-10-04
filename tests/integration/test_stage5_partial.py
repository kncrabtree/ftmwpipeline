"""Stage 5 partial fits: writing one, reading while partial, discarding one.

Normative spec: ``dev-docs/CONTRACT_STRATEGY.md`` §Events and cancellation →
§Stage 5 partial fits (Wave 5.2). A cancel (or a raising callback) during the
fit keeps the windows that finished, in the call's one atomic write:

- ``cancelled.completed_windows`` lists exactly the windows written;
- no window finished: nothing is written and any previous fit is kept;
- nothing is written *during* the walk;
- while partial, ``status`` says so and everything that reads the fit or the
  final products behaves as it does before Stage 5;
- anything that invalidates a complete fit discards a partial one (clocks and
  the timebase do not).

Resuming a partial fit (and the restart reasons) is in
``test_stage5_resume.py``. Writes only to pytest ``tmp_path``.
"""

from __future__ import annotations

import json
import multiprocessing
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import (
    CallbackFailedError,
    OperationCancelledError,
    StageDependencyError,
    StageFinished,
    StageStarted,
    WindowProgress,
)
from ftmwpipeline.cli import _events as cli_events
from ftmwpipeline.cli.main import main as cli_main
from ftmwpipeline.contract import FIT_RESTART_REASONS, MANIFEST, Absent
from ftmwpipeline.core.curation import CurationAction
from ftmwpipeline.core.stage_fit_settings import ClockSource
from ftmwpipeline.pipeline import Pipeline
from tests._events_support import (
    Recorder,
    Token,
    cancel_on_nth,
    content_digest,
    wait_no_new_children,
)

pytestmark = pytest.mark.integration

_FORK = "fork" in multiprocessing.get_all_start_methods()
JOBS = [
    1,
    pytest.param(2, marks=pytest.mark.skipif(not _FORK, reason="needs fork")),
]
_SUMMARY = ("resumed", "windows_carried", "restart_reason")


# ---- helpers ------------------------------------------------------------------------


def _copy(src: Path, tmp_path: Path, name: str = "work.ftmw") -> Path:
    dest = tmp_path / name
    shutil.copy(src, dest)
    return dest


def _states(path: Path) -> Dict[str, str]:
    return {r["stage"]: r["state"] for r in ftmw.status(path)["stages"]}


def _state(path: Path, stage: str = "fit") -> str:
    return _states(path)[stage]


def _has_partial(path: Path) -> bool:
    with h5py.File(path, "r") as h5f:
        return "stage5_partial" in h5f


def _kept_ids(path: Path) -> List[int]:
    """The window ids the file's partial fit holds (its ``window_ids``)."""
    with h5py.File(path, "r") as h5f:
        return sorted(int(w) for w in h5f["stage5_partial/window_ids"][()])


def _cancel_after(
    fp: Path, n: int = 1, *, jobs: int = 1, **kwargs: Any
) -> Tuple[OperationCancelledError, Recorder]:
    """Run ``fit_peaks`` and cancel from the callback on the *n*-th window."""
    tok = Token()
    rec = Recorder(cancel_on_nth(WindowProgress, n, tok))
    with pytest.raises(OperationCancelledError) as info:
        ftmw.fit_peaks(fp, jobs=jobs, events=rec, cancel=tok, **kwargs)
    return info.value, rec


def _reported(rec: Recorder) -> List[int]:
    return sorted({e.window_id for e in rec.of(WindowProgress)})


def _fit(fp: Path, **kwargs: Any) -> Tuple[Dict[str, Any], Recorder]:
    rec = Recorder()
    kwargs.setdefault("jobs", 1)
    ftmw.fit_peaks(fp, events=rec, **kwargs)
    (fin,) = rec.of(StageFinished)
    return dict(fin.summary), rec


@pytest.fixture(params=JOBS, ids=lambda j: f"jobs{j}")
def partial(request, baseline_2638_stage4_small, tmp_path):
    """A file whose fit was cancelled after its first finished window."""
    fp = _copy(baseline_2638_stage4_small, tmp_path)
    err, rec = _cancel_after(fp, 1, jobs=request.param)
    return fp, err, rec


# ---- writing a partial fit -----------------------------------------------------------


def test_a_cancel_lists_exactly_the_windows_it_wrote(partial):
    fp, err, rec = partial
    assert err.code == "cancelled" and err.stage == "fit"
    assert err.completed_stages == []
    assert not rec.of(StageFinished)
    # Every window that reported progress finished its whole pass, and only
    # those are kept: the error, the file and the events agree.
    assert err.completed_windows == _reported(rec) == _kept_ids(fp)
    assert err.completed_windows == sorted(err.completed_windows)
    assert len(err.completed_windows) >= 1
    assert err.to_dict()["completed_windows"] == err.completed_windows


def test_status_and_window_status_while_partial(partial):
    fp, err, _rec = partial
    status = ftmw.status(fp)
    states = {r["stage"]: r["state"] for r in status["stages"]}
    assert states["fit"] == "partial"
    assert states["review"] == "not_run"
    assert "fit" in status["runnable"] and "review" not in status["runnable"]
    # The earlier stages are untouched.
    assert states["windows"] == "complete"

    rows = {r.window_id: r for r in ftmw.window_status(fp)["windows"]}
    kept = set(err.completed_windows)
    assert kept < set(rows)
    for wid, row in rows.items():
        if wid in kept:
            assert isinstance(row.n_fitted_peaks, int)
            assert row.live == (row.n_fitted_peaks > 0)
        else:
            assert row.n_fitted_peaks is Absent.NOT_RUN
            assert row.live is Absent.NOT_RUN


def _calls() -> Dict[str, Callable[[Path], Any]]:
    add = [CurationAction(action="add", window_id=1, freq_mhz=30000.0)]
    return {
        "load_fit": lambda f: ftmw.load_fit(f),
        "review_run": lambda f: ftmw.review_run(f),
        "review_apply": lambda f: ftmw.review_apply(f, actions=add),
        "review_preview": lambda f: ftmw.review_preview(f, actions=add),
        "review_edit": lambda f: ftmw.review_edit(f, window_id=1, add=[30000.0]),
        "review_create": lambda f: ftmw.review_create(f, 30000.0),
        "review_accept_candidate": lambda f: ftmw.review_accept(
            f, 1, candidate_freq=30000.0
        ),
        "get_candidate_ledger": lambda f: ftmw.get_candidate_ledger(f),
        "window_model": lambda f: ftmw.window_model(f, 1),
        "spectrum_model": lambda f: ftmw.spectrum_model(f),
        "show_fit": lambda f: ftmw.show_fit(f, window_ids=[1]),
        "report_table": lambda f: ftmw.report_table(f),
    }


@pytest.mark.parametrize(
    "name",
    [
        "load_fit",
        "review_run",
        "review_apply",
        "review_preview",
        "review_edit",
        "review_create",
        "review_accept_candidate",
        "get_candidate_ledger",
        "window_model",
        "spectrum_model",
        "show_fit",
        "report_table",
    ],
)
def test_the_fit_and_review_accessors_refuse_while_partial(
    name, baseline_2638_stage4_small, tmp_path
):
    fp = _copy(baseline_2638_stage4_small, tmp_path)
    _cancel_after(fp)
    call = _calls()[name]
    with pytest.raises(StageDependencyError) as info:
        call(fp)
    assert info.value.code == "stage_not_run"
    assert _state(fp) == "partial"


def _outcome(call: Callable[[Path], Any], path: Path) -> Tuple[Any, ...]:
    try:
        result = call(path)
    except Exception as exc:  # noqa: BLE001 - the outcome is the comparison
        as_dict = exc.to_dict() if hasattr(exc, "to_dict") else {}
        return ("raises", type(exc).__name__, as_dict.get("code"))
    return ("ok", type(result).__name__, repr(result)[:200])


def test_a_partial_fit_reads_as_before_stage_5(baseline_2638_stage4_small, tmp_path):
    """Every accessor that reads the fit or the final products answers as it
    does on a file whose Stage 5 has never run (the same value, or the same
    refusal)."""
    never = _copy(baseline_2638_stage4_small, tmp_path, "never.ftmw")
    part = _copy(baseline_2638_stage4_small, tmp_path, "part.ftmw")
    _cancel_after(part)
    reads: Dict[str, Callable[[Path], Any]] = {
        **_calls(),
        "get_final_products": lambda f: ftmw.get_final_products(f),
        "get_review_status": lambda f: ftmw.get_review_status(f),
        "review_log": lambda f: ftmw.review_log(f),
        "fit_thresholds": lambda f: ftmw.fit_thresholds(f),
        "analysis_fingerprint": lambda f: ftmw.analysis_fingerprint(f)["digest"],
    }
    before = content_digest(part)
    for name, call in reads.items():
        assert _outcome(call, never) == _outcome(call, part), name
    # None of them wrote (a refusal leaves the partial fit as it was).
    assert content_digest(part) == before
    assert _state(part) == "partial"
    assert ftmw.get_final_products(part) is None


def test_a_partial_fit_discards_the_previous_fit_and_review(
    baseline_2638_stage4_small, tmp_path
):
    fp = _copy(baseline_2638_stage4_small, tmp_path)
    ftmw.fit_peaks(fp, jobs=1)
    ftmw.review_run(fp)
    assert _state(fp) == "complete" and _state(fp, "review") == "complete"
    assert ftmw.get_final_products(fp) is not None

    _cancel_after(fp)
    assert _state(fp) == "partial"
    assert _state(fp, "review") == "not_run"
    assert ftmw.get_final_products(fp) is None
    with pytest.raises(StageDependencyError) as info:
        ftmw.load_fit(fp)
    assert info.value.code == "stage_not_run"


def test_the_fingerprint_treats_a_partial_fit_as_not_run(
    baseline_2638_stage4_small, tmp_path
):
    from ftmwpipeline._internal.fingerprint_impl import canonical_fingerprint_inputs
    from ftmwpipeline.file_manager import IncompleteProvenanceError

    fp = _copy(baseline_2638_stage4_small, tmp_path)
    _cancel_after(fp)
    try:
        inputs = canonical_fingerprint_inputs(fp)
    except IncompleteProvenanceError as exc:  # a fixture without full provenance
        pytest.skip(f"fingerprint unavailable on the fixture: {exc}")
    assert inputs["fit"] is None
    assert inputs["fit_absent"] == "not_run"


# ---- nothing finished, nothing written -------------------------------------------------


@pytest.mark.parametrize("jobs", JOBS, ids=lambda j: f"jobs{j}")
@pytest.mark.parametrize("previous", [False, True], ids=["no_fit", "previous_fit"])
def test_a_cancel_before_any_window_finishes_writes_nothing(
    jobs, previous, baseline_2638_stage4_small, tmp_path
):
    fp = _copy(baseline_2638_stage4_small, tmp_path)
    if previous:
        ftmw.fit_peaks(fp, jobs=1)
    before = content_digest(fp)
    raw = fp.read_bytes()
    tok = Token()
    rec = Recorder(cancel_on_nth(StageStarted, 1, tok))
    children = set(multiprocessing.active_children())
    with pytest.raises(OperationCancelledError) as info:
        ftmw.fit_peaks(fp, jobs=jobs, events=rec, cancel=tok)
    assert info.value.completed_windows == []
    assert not rec.of(WindowProgress)
    assert content_digest(fp) == before
    assert fp.read_bytes() == raw  # not even replaced
    assert _state(fp) == ("complete" if previous else "not_run")
    assert not _has_partial(fp)
    if previous:
        assert ftmw.load_fit(fp).window_fits  # the previous fit is kept
    assert not wait_no_new_children(children)


def test_a_token_set_before_the_call_writes_nothing(
    baseline_2638_stage4_small, tmp_path
):
    fp = _copy(baseline_2638_stage4_small, tmp_path)
    raw = fp.read_bytes()
    tok = Token()
    tok.set()
    with pytest.raises(OperationCancelledError) as info:
        ftmw.fit_peaks(fp, jobs=1, cancel=tok)
    assert info.value.completed_windows == []
    assert fp.read_bytes() == raw


@pytest.mark.parametrize("jobs", JOBS, ids=lambda j: f"jobs{j}")
def test_nothing_is_written_during_the_walk(jobs, baseline_2638_stage4_small, tmp_path):
    """The file is the one the call started with at every event of the walk,
    up to the window the cancel interrupted: a process killed there leaves it
    as it was."""
    fp = _copy(baseline_2638_stage4_small, tmp_path)
    before = content_digest(fp)
    seen: List[str] = []
    tok = Token()
    cancel = cancel_on_nth(WindowProgress, 3, tok)

    def on_event(event: Any) -> None:
        seen.append(content_digest(fp))
        cancel(event)

    with pytest.raises(OperationCancelledError) as info:
        ftmw.fit_peaks(fp, jobs=jobs, events=on_event, cancel=tok)
    assert info.value.completed_windows
    assert seen and set(seen) == {before}
    assert content_digest(fp) != before  # ... and only the end of the call wrote


# ---- a raising callback writes the same ------------------------------------------------


@pytest.mark.parametrize("jobs", JOBS, ids=lambda j: f"jobs{j}")
def test_a_raising_callback_keeps_the_finished_windows_too(
    jobs, baseline_2638_stage4_small, tmp_path
):
    fp = _copy(baseline_2638_stage4_small, tmp_path)
    ftmw.fit_peaks(fp, jobs=1)  # superseded by the partial fit
    boom = ValueError("boom")
    seen: List[WindowProgress] = []

    def bad(event: Any) -> None:
        if isinstance(event, WindowProgress):
            seen.append(event)
            if len(seen) == 2:
                raise boom

    children = set(multiprocessing.active_children())
    with pytest.raises(CallbackFailedError) as info:
        ftmw.fit_peaks(fp, jobs=jobs, events=bad)
    assert info.value.code == "callback_failed"
    assert info.value.event_schema == "ftmw/window_progress@1"
    assert info.value.__cause__ is boom
    # What a cancel at that point leaves: every window that reported, the one
    # whose callback raised included.
    reported = sorted({e.window_id for e in seen})
    assert len(reported) == 2
    assert _kept_ids(fp) == reported
    assert _state(fp) == "partial" and _state(fp, "review") == "not_run"
    assert not wait_no_new_children(children)


def test_a_callback_raising_before_any_window_writes_nothing(
    baseline_2638_stage4_small, tmp_path
):
    fp = _copy(baseline_2638_stage4_small, tmp_path)
    before = fp.read_bytes()

    def bad(event: Any) -> None:
        if isinstance(event, StageStarted):
            raise ValueError("no")

    with pytest.raises(CallbackFailedError):
        ftmw.fit_peaks(fp, jobs=1, events=bad)
    assert fp.read_bytes() == before


# ---- discard -------------------------------------------------------------------------


def test_a_stage5_setting_change_discards_the_partial_fit(partial):
    fp, _err, _rec = partial
    result = ftmw.settings_set(fp, "stage5.conservative.max_peaks", 7)
    assert "fit" in result.invalidated
    assert _state(fp) == "not_run"
    assert not _has_partial(fp)
    assert all(
        r.n_fitted_peaks is Absent.NOT_RUN for r in ftmw.window_status(fp)["windows"]
    )


def test_unsetting_a_stage5_setting_discards_the_partial_fit(
    baseline_2638_stage4_small, tmp_path
):
    fp = _copy(baseline_2638_stage4_small, tmp_path)
    ftmw.settings_set(fp, "stage5.conservative.max_peaks", 7)
    _cancel_after(fp)
    assert _state(fp) == "partial"
    result = ftmw.settings_unset(fp, "stage5.conservative.max_peaks")
    assert "fit" in result.invalidated
    assert _state(fp) == "not_run" and not _has_partial(fp)


def test_an_upstream_rerun_discards_the_partial_fit(partial):
    fp, _err, _rec = partial
    result = ftmw.assign_windows(fp)
    assert "fit" in result.invalidated
    assert _state(fp) == "not_run"
    assert not _has_partial(fp)
    assert all(
        r.n_fitted_peaks is Absent.NOT_RUN for r in ftmw.window_status(fp)["windows"]
    )


def test_a_forced_reimport_discards_the_partial_fit(
    baseline_2638_stage4_small, exp_2638_data_path, tmp_path
):
    fp = _copy(baseline_2638_stage4_small, tmp_path)
    _cancel_after(fp)
    result = ftmw.import_data(fp, source=exp_2638_data_path, force=True)
    assert "fit" in result["invalidated"]
    assert _state(fp) == "not_run"
    assert not _has_partial(fp)


_CLOCKS = [
    ClockSource(freq_mhz=5760.0, locked=True, label="upconv"),
    ClockSource(freq_mhz=5120.0, locked=True, label="downconv"),
]


def test_clocks_leave_the_partial_fit_alone(baseline_2638_stage4_small, tmp_path):
    fp = _copy(baseline_2638_stage4_small, tmp_path)
    err, _ = _cancel_after(fp)
    ftmw.set_clock_sources(fp, _CLOCKS)
    assert _state(fp) == "partial"
    assert _kept_ids(fp) == err.completed_windows
    ftmw.clear_clock_sources(fp)
    assert _state(fp) == "partial"
    assert _kept_ids(fp) == err.completed_windows


def test_the_timebase_leaves_the_partial_fit_alone(
    baseline_2638_stage4_small, tmp_path
):
    fp = _copy(baseline_2638_stage4_small, tmp_path)
    err, _ = _cancel_after(fp)
    ftmw.calibrate_timebase(fp, clocks=_CLOCKS)
    assert _state(fp) == "partial"
    assert _state(fp, "timebase") == "complete"
    assert _kept_ids(fp) == err.completed_windows


# ---- every interface -----------------------------------------------------------------


def _cancel_via(via: str, fp: Path, capsys, monkeypatch, *, jobs: int = 1) -> List[int]:
    """Cancel a fit after its first window through ``via``; return the
    ``completed_windows`` the cancel reported."""
    if via == "api":
        err, _ = _cancel_after(fp, 1, jobs=jobs)
        return err.completed_windows
    if via == "pipeline":
        tok = Token()
        rec = Recorder(cancel_on_nth(WindowProgress, 1, tok))
        with pytest.raises(OperationCancelledError) as info:
            Pipeline.open(fp).fit_peaks(jobs=jobs, events=rec, cancel=tok)
        return info.value.completed_windows

    state = {"cancel": False}
    real_write = cli_events.write_event

    def write_and_cancel(event: Any) -> None:
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
    capsys.readouterr()
    rc = cli_main(["fit", "run", str(fp), "--jobs", str(jobs), "--events", "--json"])
    err_text = capsys.readouterr().err
    assert rc == 130, err_text
    error = json.loads(err_text.strip().splitlines()[-1])
    assert error["schema"] == "ftmw/error@1" and error["code"] == "cancelled"
    assert error["stage"] == "fit"
    return list(error["completed_windows"])


def test_api_pipeline_and_cli_cancel_alike(
    baseline_2638_stage4_small, tmp_path, capsys, monkeypatch
):
    kept: Dict[str, List[int]] = {}
    files: Dict[str, Path] = {}
    for via in ("api", "pipeline", "cli"):
        files[via] = _copy(baseline_2638_stage4_small, tmp_path, f"{via}.ftmw")
        kept[via] = _cancel_via(via, files[via], capsys, monkeypatch)
        assert _state(files[via]) == "partial"
        assert _kept_ids(files[via]) == kept[via]
    assert kept["api"] == kept["pipeline"] == kept["cli"]
    assert kept["api"]


def test_api_pipeline_and_cli_resume_alike(
    baseline_2638_stage4_small, tmp_path, capsys
):
    """A plain run after a cancel resumes, with the same summary fields, the
    same events and the same fit, on every interface."""
    summaries: Dict[str, Dict[str, Any]] = {}
    sequences: Dict[str, List[Tuple[Any, ...]]] = {}
    fits: Dict[str, Any] = {}
    n_kept = 0
    for via in ("api", "pipeline", "cli"):
        fp = _copy(baseline_2638_stage4_small, tmp_path, f"{via}.ftmw")
        err, _ = _cancel_after(fp)
        n_kept = len(err.completed_windows)
        rec = Recorder()
        if via == "api":
            ftmw.fit_peaks(fp, jobs=1, events=rec)
            summaries[via] = dict(rec.of(StageFinished)[0].summary)
        elif via == "pipeline":
            Pipeline.open(fp).fit_peaks(jobs=1, events=rec)
            summaries[via] = dict(rec.of(StageFinished)[0].summary)
        else:
            capsys.readouterr()
            assert cli_main(["fit", "run", str(fp), "--jobs", "1", "--json"]) == 0
            out = capsys.readouterr().out
            summaries[via] = json.loads(out)["summary"]
        if via != "cli":
            sequences[via] = [
                (e.phase, e.round, e.index, e.total, e.window_id)
                for e in rec.of(WindowProgress)
            ]
        fits[via] = [
            (
                w.window_id,
                tuple((p.peak_uid, p.origin, p.frequency_mhz) for p in w.fitted_peaks),
            )
            for w in sorted(ftmw.load_fit(fp).window_fits, key=lambda w: w.window_id)
        ]
        assert not _has_partial(fp)
    for via, summary in summaries.items():
        assert summary["resumed"] is True, via
        assert summary["windows_carried"] == n_kept, via
        assert summary["restart_reason"] is None, via
    assert set(summaries["api"]) == set(summaries["pipeline"]) == set(summaries["cli"])
    assert summaries["api"] == summaries["pipeline"]
    for key in summaries["api"]:
        assert summaries["cli"][key] == summaries["api"][key], key
    assert sequences["api"] == sequences["pipeline"]
    assert fits["api"] == fits["pipeline"] == fits["cli"]


def test_pipeline_and_cli_restart_agree(partial, tmp_path, capsys):
    fp, _err, _rec = partial
    other = _copy(fp, tmp_path, "other.ftmw")
    third = _copy(fp, tmp_path, "third.ftmw")

    rec = Recorder()
    Pipeline.open(fp).fit_peaks(jobs=1, restart=True, events=rec)
    (fin,) = rec.of(StageFinished)

    capsys.readouterr()
    rc = cli_main(["fit", "run", str(other), "--jobs", "1", "--restart", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    run_result = json.loads(out)
    assert set(run_result["summary"]) == set(fin.summary)
    for key in _SUMMARY:
        assert run_result["summary"][key] == fin.summary[key]
    assert run_result["summary"]["resumed"] is False
    assert run_result["summary"]["windows_carried"] == 0
    assert run_result["summary"]["restart_reason"] == "restart_requested"

    api_summary, _ = _fit(third, restart=True)
    assert api_summary == dict(fin.summary)


def test_the_cli_text_output_says_what_a_fit_did(
    baseline_2638_stage4_small, tmp_path, capsys
):
    fp = _copy(baseline_2638_stage4_small, tmp_path)
    _cancel_after(fp)
    capsys.readouterr()
    assert cli_main(["fit", "run", str(fp), "--jobs", "1"]) == 0
    assert "Resumed a partial fit" in capsys.readouterr().out

    fp2 = _copy(baseline_2638_stage4_small, tmp_path, "two.ftmw")
    _cancel_after(fp2)
    capsys.readouterr()
    assert cli_main(["fit", "run", str(fp2), "--jobs", "1", "--restart"]) == 0
    assert "restart_requested" in capsys.readouterr().out


# ---- the declared surface --------------------------------------------------------------


def test_the_summary_keys_and_vocabulary_are_declared():
    from ftmwpipeline._internal.stage5_impl import FIT_RUN_SUMMARY_KEYS

    assert set(_SUMMARY) <= set(FIT_RUN_SUMMARY_KEYS)
    assert set(MANIFEST.vocabularies["restart_reason"]) == set(FIT_RESTART_REASONS)
    assert set(FIT_RESTART_REASONS) == {
        "restart_requested",
        "settings_changed",
        "incomplete_provenance",
        "thaw_refit",
    }


def test_a_fresh_fit_summary_has_the_resume_keys(baseline_2638_stage4_small, tmp_path):
    from ftmwpipeline._internal.stage5_impl import FIT_RUN_SUMMARY_KEYS

    fp = _copy(baseline_2638_stage4_small, tmp_path)
    summary, _ = _fit(fp)
    assert set(summary) == set(FIT_RUN_SUMMARY_KEYS)
    assert summary["resumed"] is False
    assert summary["windows_carried"] == 0
    assert summary["restart_reason"] is None
