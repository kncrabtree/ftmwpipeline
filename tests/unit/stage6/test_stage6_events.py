"""Stage 6 events and cancellation (CONTRACT_STRATEGY §Events and cancellation).

A Stage 6 call reports as the ``review`` stage: ``StageStarted``, a
``WindowProgress`` per window it re-fits, then ``StageFinished`` whose summary
is the verb's ``run_result`` summary. A curation batch is one unit: a cancel or
a failing callback discards all of it (the undo baseline the batch took is
removed again), and an undo honours a cancel only before its restore.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import (
    CallbackFailedError,
    OperationCancelledError,
    StageFinished,
    StageStarted,
    WindowProgress,
)
from ftmwpipeline._internal import stage6_impl
from ftmwpipeline.cli.main import main as cli_main
from ftmwpipeline.contract import Stage
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5

pytestmark = [pytest.mark.integration]


class _Token:
    def __init__(self) -> None:
        self.flag = False

    def set(self) -> None:
        self.flag = True

    def is_set(self) -> bool:
        return self.flag


def _content_digest(path: Path) -> str:
    """Every group, dataset and attribute -- not the bytes: a batch that took
    and then removed the undo baseline leaves the same content in a file
    whose free space differs."""
    h = hashlib.sha256()

    def add_attrs(obj: h5py.HLObject) -> None:
        for key in sorted(obj.attrs.keys()):
            value = obj.attrs[key]
            h.update(key.encode())
            h.update(
                value.tobytes()
                if isinstance(value, np.ndarray)
                else repr(value).encode()
            )

    def visit(name: str, obj: h5py.HLObject) -> None:
        h.update(name.encode())
        add_attrs(obj)
        if isinstance(obj, h5py.Dataset):
            data = obj[()]
            h.update(
                data.tobytes() if isinstance(data, np.ndarray) else repr(data).encode()
            )

    with h5py.File(path, "r") as f:
        add_attrs(f)
        f.visititems(visit)
    return h.hexdigest()


def _a_window_with_two_peaks(path: Path) -> Tuple[int, List[float]]:
    with h5py.File(path, "r") as f:
        sf = load_spectrum_fit_from_hdf5(f["stage5_fitting"])
    wf = next(w for w in sf.window_fits if len(w.fitted_peaks) >= 2)
    return int(wf.window_id), [float(p.frequency_mhz) for p in wf.fitted_peaks]


def _remove_action(path: Path) -> list:
    wid, freqs = _a_window_with_two_peaks(path)
    return [
        {"action": "remove", "window_id": wid, "freq_mhz": freqs[0], "frame": "raw"}
    ]


def _has_baseline(path: Path) -> bool:
    with h5py.File(path, "r") as f:
        return stage6_impl.STAGE5_BASELINE_GROUP in f


def test_review_edit_reports_the_review_stage(stage5_small_file):
    wid, freqs = _a_window_with_two_peaks(stage5_small_file)
    seen: list = []
    result = ftmw.review_edit(
        str(stage5_small_file), wid, remove=[freqs[0]], frame="raw", events=seen.append
    )
    assert isinstance(seen[0], StageStarted) and isinstance(seen[-1], StageFinished)
    assert all(e.stage is Stage.REVIEW for e in seen)
    windows = [e for e in seen if isinstance(e, WindowProgress)]
    assert [w.window_id for w in windows][:1] == [wid]
    assert seen[-1].summary == stage6_impl.review_edit_summary(result, (), [freqs[0]])


def test_cancelled_batch_leaves_the_file_as_it_was(stage5_small_file):
    before = _content_digest(stage5_small_file)
    assert not _has_baseline(stage5_small_file)
    token = _Token()

    def cancel_on_start(event: object) -> None:
        if isinstance(event, StageStarted):
            token.set()

    with pytest.raises(OperationCancelledError) as info:
        ftmw.review_apply(
            str(stage5_small_file),
            actions=_remove_action(stage5_small_file),
            events=cancel_on_start,
            cancel=token,
        )
    assert info.value.stage == "review"
    assert not _has_baseline(stage5_small_file)
    assert _content_digest(stage5_small_file) == before
    assert ftmw.review_log(str(stage5_small_file)) == []


def test_failing_callback_mid_batch_discards_it(stage5_small_file):
    before = _content_digest(stage5_small_file)

    def fail_on_window(event: object) -> None:
        if isinstance(event, WindowProgress):
            raise RuntimeError("listener broke")

    with pytest.raises(CallbackFailedError) as info:
        ftmw.review_apply(
            str(stage5_small_file),
            actions=_remove_action(stage5_small_file),
            events=fail_on_window,
        )
    assert info.value.event_schema == "ftmw/window_progress@1"
    assert isinstance(info.value.__cause__, RuntimeError)
    assert _content_digest(stage5_small_file) == before


def test_undo_honours_a_cancel_only_before_its_restore(stage5_small_file, monkeypatch):
    path = str(stage5_small_file)
    ftmw.review_apply(path, actions=_remove_action(stage5_small_file))
    applied = _content_digest(stage5_small_file)

    token = _Token()
    token.set()
    with pytest.raises(OperationCancelledError):
        ftmw.review_undo(path, [0], cancel=token)
    assert _content_digest(stage5_small_file) == applied

    # Once the restore has begun, a cancel no longer stops it.
    token = _Token()
    restore = stage6_impl._reset_to_baseline

    def restore_then_cancel(*args: object, **kwargs: object) -> None:
        restore(*args, **kwargs)  # type: ignore[arg-type]
        token.set()

    monkeypatch.setattr(stage6_impl, "_reset_to_baseline", restore_then_cancel)
    result = ftmw.review_undo(path, [0], cancel=token)
    assert result.removed and ftmw.review_log(path) == []


def test_preview_and_dry_run_record_no_completed_stage(stage5_small_file):
    path = str(stage5_small_file)
    before = _content_digest(stage5_small_file)
    seen: list = []
    ftmw.review_preview(
        path, actions=_remove_action(stage5_small_file), events=seen.append
    )
    assert isinstance(seen[-1], StageFinished)
    assert seen[-1].summary["n_actions"] == 1
    assert _content_digest(stage5_small_file) == before


def test_review_run_lines_render_once_from_events(stage5_small_file, caplog):
    seen: list = []
    with caplog.at_level(logging.INFO, logger="ftmwpipeline"):
        result = ftmw.review_run(str(stage5_small_file), events=seen.append)
    msgs = [r.getMessage() for r in caplog.records if r.name.endswith("stage6_impl")]
    assert sum(m.startswith("Stage 6 review: routing attention") for m in msgs) == 1
    assert sum(m.startswith("Saved Stage 6 review to") for m in msgs) == 1
    assert [type(e) for e in seen] == [StageStarted, StageFinished]
    assert seen[-1].summary["n_windows"] == result.n_windows


def test_cli_events_match_the_run_result_summary(stage5_small_file, capsys):
    rc = cli_main(["review", "run", str(stage5_small_file), "--events", "--json"])
    assert rc == 0
    out, err = capsys.readouterr()
    run_result = json.loads(out)
    events = [json.loads(line) for line in err.splitlines() if line.startswith("{")]
    assert [e["schema"] for e in events] == [
        "ftmw/stage_started@1",
        "ftmw/stage_finished@1",
    ]
    assert events[-1]["operation"] == "review run"
    assert set(events[-1]["summary"]) == set(run_result["summary"])
