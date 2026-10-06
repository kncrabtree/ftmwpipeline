"""Events and cancellation on the ``ReviewSession`` verbs
(CONTRACT_STRATEGY §Events and cancellation).

A session verb is the same long operation as its ``Pipeline`` method: one
``review`` stage around the session's own transaction, so ``Invalidated`` and
``StageFinished`` reach the callback only once the write is durable, and a
cancelled call leaves the file -- and the session's staged preview, drift note
and fingerprint -- exactly as they were.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable, List, Tuple

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import (
    Invalidated,
    OperationCancelledError,
    StageFinished,
    WindowProgress,
)
from ftmwpipeline._internal import stage6_impl
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5
from ftmwpipeline.pipeline import Pipeline
from tests._events_support import Recorder, Token, assert_stage_order, content_digest

pytestmark = [pytest.mark.integration]


def _a_window_with_two_peaks(path: Path) -> Tuple[int, List[float]]:
    with h5py.File(path, "r") as f:
        sf = load_spectrum_fit_from_hdf5(f["stage5_fitting"])
    wf = next(w for w in sf.window_fits if len(w.fitted_peaks) >= 2)
    return int(wf.window_id), [float(p.frequency_mhz) for p in wf.fitted_peaks]


def _outcome(path: Path) -> Tuple[Any, ...]:
    """What an edit decides, comparable across two files (a whole-file digest
    is not: each write stamps its own time)."""
    with h5py.File(path, "r") as f:
        sf = load_spectrum_fit_from_hdf5(f["stage5_fitting"])
    stats = sorted(
        (int(wf.window_id), len(wf.fitted_peaks), round(float(wf.reduced_chi2), 9))
        for wf in sf.window_fits
        if wf.window_id is not None
    )
    return tuple(stats), len(ftmw.review_log(str(path)))


def _on_disk_at_commit(path: Path, seen: List[str]) -> Callable[[Any], None]:
    """An event action recording the file's content (read straight from disk,
    outside any transaction) when ``Invalidated`` / ``StageFinished`` arrive."""

    def act(event: Any) -> None:
        if isinstance(event, (Invalidated, StageFinished)):
            seen.append(content_digest(path))

    return act


def _shape(events: List[Any]) -> List[Tuple[Any, ...]]:
    """The comparable form of an event stream: everything but timings."""
    out: List[Tuple[Any, ...]] = []
    for e in events:
        if isinstance(e, WindowProgress):
            out.append(
                (type(e).__name__, e.operation, e.stage, e.phase, e.round)
                + (e.index, e.total, e.window_id, e.n_peaks, e.dropped)
            )
        elif isinstance(e, StageFinished):
            out.append((type(e).__name__, e.operation, e.stage, dict(e.summary)))
        elif isinstance(e, Invalidated):
            out.append((type(e).__name__, e.operation, e.stage, e.stages))
        else:
            out.append((type(e).__name__, e.operation, e.stage))
    return out


def test_session_edit_finishes_after_the_write_is_durable(stage5_small_file):
    path = stage5_small_file
    wid, freqs = _a_window_with_two_peaks(path)
    before = content_digest(path)
    at_commit: List[str] = []
    rec = Recorder(_on_disk_at_commit(path, at_commit))
    with Pipeline.open(str(path)).review_session() as session:
        result = session.review_edit(wid, remove=[freqs[0]], frame="raw", events=rec)
    assert_stage_order(rec.events)
    assert all(e.operation == "review edit" for e in rec.events)
    assert rec.of(WindowProgress)[0].window_id == wid
    assert rec.events[-1].summary == stage6_impl.review_edit_summary(
        result, (), [freqs[0]]
    )
    after = content_digest(path)
    assert after != before
    assert at_commit and all(d == after for d in at_commit)


def test_session_undo_finishes_after_the_write_is_durable(stage5_small_file):
    path = stage5_small_file
    wid, freqs = _a_window_with_two_peaks(path)
    ftmw.review_edit(str(path), wid, remove=[freqs[0]], frame="raw")
    serial = ftmw.review_log(str(path))[0].serial
    before = content_digest(path)
    at_commit: List[str] = []
    rec = Recorder(_on_disk_at_commit(path, at_commit))
    with Pipeline.open(str(path)).review_session() as session:
        result = session.review_undo([serial], events=rec)
    assert_stage_order(rec.events)
    assert all(e.operation == "review undo" for e in rec.events)
    assert rec.events[-1].summary == stage6_impl.review_undo_summary(result, False)
    assert ftmw.review_log(str(path)) == []
    after = content_digest(path)
    assert after != before
    assert at_commit and all(d == after for d in at_commit)


def test_cancel_mid_refit_leaves_file_and_session_unchanged(
    stage5_small_file, stage5_small_source, tmp_path
):
    path = stage5_small_file
    twin = tmp_path / "twin.ftmw"
    shutil.copy(stage5_small_source, twin)
    wid, freqs = _a_window_with_two_peaks(path)
    actions = [
        {"action": "remove", "window_id": wid, "freq_mhz": freqs[0], "frame": "raw"}
    ]
    before = content_digest(path)
    token = Token()

    def cancel_on_window(event: Any) -> None:
        if isinstance(event, WindowProgress):
            token.set()

    with Pipeline.open(str(path)).review_session() as session:
        session.review_preview(actions=actions)
        staged = session._staged
        fingerprint = session._fingerprint
        assert staged is not None
        with pytest.raises(OperationCancelledError) as info:
            session.review_edit(
                wid,
                remove=[freqs[1]],
                frame="raw",
                events=cancel_on_window,
                cancel=token,
            )
        assert info.value.stage == "review"
        assert token.is_set()
        assert content_digest(path) == before
        assert session._staged is staged
        assert session._fingerprint == fingerprint
        assert session._pending_base_changed is False

        # The session carries on: the staged preview is still the one an
        # apply persists, and a later edit lands as a sessionless one does.
        applied = session.review_apply(actions=actions)
        assert applied.base_changed is False
        session.review_edit(wid, remove=[freqs[1]], frame="raw")

    ftmw.review_apply(str(twin), actions=actions)
    ftmw.review_edit(str(twin), wid, remove=[freqs[1]], frame="raw")
    assert _outcome(path) == _outcome(twin)


def test_session_and_api_emit_the_same_events_for_one_edit(
    stage5_small_file, stage5_small_source, tmp_path
):
    twin = tmp_path / "twin.ftmw"
    shutil.copy(stage5_small_source, twin)
    wid, freqs = _a_window_with_two_peaks(stage5_small_file)
    api_events = Recorder()
    ftmw.review_edit(str(twin), wid, remove=[freqs[0]], frame="raw", events=api_events)
    session_events = Recorder()
    with Pipeline.open(str(stage5_small_file)).review_session() as session:
        session.review_edit(wid, remove=[freqs[0]], frame="raw", events=session_events)
    assert _shape(session_events.events) == _shape(api_events.events)
    assert _outcome(stage5_small_file) == _outcome(twin)
