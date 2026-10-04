"""A curation batch is one unit (CONTRACT_STRATEGY §Events and cancellation).

A cancel, or a callback that raises, in the middle of a batch that refits
several windows discards the whole batch: the file is as it was before the call,
the decision log is unchanged, and nothing is reported finished.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import (
    CallbackFailedError,
    OperationCancelledError,
    StageFinished,
    WindowProgress,
)
from ftmwpipeline.cli.main import main as cli_main
from ftmwpipeline.contract import Stage
from ftmwpipeline.file_manager import PipelineFileError
from ftmwpipeline.io.fitting_serialization import (
    load_spectrum_fit_from_hdf5,
    read_fit_peak_frequencies_by_window,
)
from ftmwpipeline.pipeline import Pipeline
from tests._events_support import (
    Recorder,
    Token,
    assert_stage_order,
    cancel_on_nth,
    content_digest,
)

pytestmark = [pytest.mark.integration]


def _remove_one_peak_in_each_of_several_windows(path: Path) -> List[Dict[str, Any]]:
    with h5py.File(path, "r") as f:
        fit = load_spectrum_fit_from_hdf5(f["stage5_fitting"])
    windows = sorted(
        (w for w in fit.window_fits if len(w.fitted_peaks) >= 1),
        key=lambda w: -len(w.fitted_peaks),
    )[:3]
    if len(windows) < 2:
        pytest.skip("the fixture has fewer than two fitted windows")
    return [
        {
            "action": "remove",
            "window_id": int(w.window_id),
            "freq_mhz": float(w.fitted_peaks[0].frequency_mhz),
            "frame": "raw",
        }
        for w in windows
    ]


@pytest.fixture
def batch(stage5_multi_file, stage5_multi_source, tmp_path):
    """(file under test, its batch of actions, the windows an uncancelled run
    of that batch refits)."""
    actions = _remove_one_peak_in_each_of_several_windows(stage5_multi_file)
    probe = tmp_path / "probe.ftmw"
    shutil.copy(stage5_multi_source, probe)
    rec = Recorder()
    ftmw.review_apply(str(probe), actions=actions, events=rec)
    assert_stage_order(rec.events)
    assert all(e.stage is Stage.REVIEW for e in rec.events)
    refit = rec.of(WindowProgress)
    if len(refit) < 2:
        pytest.skip("the batch refits fewer than two windows")
    return stage5_multi_file, actions, refit


def test_cancel_mid_batch_discards_the_whole_batch(batch):
    fp, actions, _refit = batch
    before = content_digest(fp)
    log_before = ftmw.review_log(str(fp))
    tok = Token()
    rec = Recorder(cancel_on_nth(WindowProgress, 1, tok))
    with pytest.raises(OperationCancelledError) as info:
        ftmw.review_apply(str(fp), actions=actions, events=rec, cancel=tok)
    err = info.value
    assert err.code == "cancelled" and err.stage == "review"
    assert err.completed_windows == []
    assert not rec.of(StageFinished)
    assert content_digest(fp) == before
    assert ftmw.review_log(str(fp)) == log_before


def test_failing_callback_mid_batch_discards_the_whole_batch(batch):
    fp, actions, _refit = batch
    before = content_digest(fp)
    boom = RuntimeError("listener broke")

    def bad(event):
        if isinstance(event, WindowProgress):
            raise boom

    with pytest.raises(CallbackFailedError) as info:
        ftmw.review_apply(str(fp), actions=actions, events=bad)
    assert info.value.event_schema == "ftmw/window_progress@1"
    assert info.value.__cause__ is boom
    assert content_digest(fp) == before


def test_a_token_set_before_the_batch_leaves_the_file(batch):
    fp, actions, _refit = batch
    before = content_digest(fp)
    tok = Token()
    tok.set()
    with pytest.raises(OperationCancelledError) as info:
        ftmw.review_apply(str(fp), actions=actions, cancel=tok)
    assert info.value.completed_stages == []
    assert content_digest(fp) == before


def test_the_batch_is_effective_when_not_cancelled(batch):
    # The guard against a vacuous test above: the same batch does change the file.
    fp, actions, _refit = batch
    before = content_digest(fp)
    ftmw.review_apply(str(fp), actions=actions)
    assert content_digest(fp) != before


# ---- crash safety: a discarded batch leaves the file's bytes (§Crash safety) --------

_BASELINE = "stage5_fitting_baseline"


def _bytes(path: Path) -> bytes:
    return path.read_bytes()


def _has_baseline(path: Path) -> bool:
    with h5py.File(path, "r") as f:
        return _BASELINE in f


def _copies(directory: Path) -> List[str]:
    return sorted(p.name for p in directory.iterdir() if ".ftmw-tmp." in p.name)


def test_a_cancelled_batch_leaves_the_bytes_and_no_baseline(batch):
    """The batch's baseline snapshot and its refits are made in the call's
    working copy, so a cancel takes the baseline back with everything else."""
    fp, actions, _refit = batch
    before = _bytes(fp)
    had_baseline = _has_baseline(fp)
    tok = Token()
    rec = Recorder(cancel_on_nth(WindowProgress, 1, tok))
    with pytest.raises(OperationCancelledError):
        ftmw.review_apply(str(fp), actions=actions, events=rec, cancel=tok)
    assert _bytes(fp) == before
    assert _has_baseline(fp) is had_baseline
    assert _copies(fp.parent) == []


def test_a_failing_callback_mid_batch_leaves_the_bytes_and_no_baseline(batch):
    fp, actions, _refit = batch
    before = _bytes(fp)
    had_baseline = _has_baseline(fp)

    def bad(event):
        if isinstance(event, WindowProgress):
            raise RuntimeError("listener broke")

    with pytest.raises(CallbackFailedError):
        ftmw.review_apply(str(fp), actions=actions, events=bad)
    assert _bytes(fp) == before
    assert _has_baseline(fp) is had_baseline
    assert _copies(fp.parent) == []


def test_a_refused_batch_leaves_the_bytes(stage5_multi_file):
    """A batch that fails validation or a row writes nothing at all."""
    fp = stage5_multi_file
    before = _bytes(fp)
    bad = [{"action": "remove", "window_id": 10**6, "freq_mhz": 1.0, "frame": "raw"}]
    with pytest.raises(PipelineFileError):
        ftmw.review_apply(str(fp), actions=bad)
    assert _bytes(fp) == before
    assert _copies(fp.parent) == []


def test_a_completed_batch_commits_with_its_baseline_and_no_copy(batch):
    fp, actions, _refit = batch
    seen: List[Dict[str, Any]] = []

    def probe(event):
        if isinstance(event, StageFinished):
            seen.append({"copies": _copies(fp.parent), "baseline": _has_baseline(fp)})

    ftmw.review_apply(str(fp), actions=actions, events=probe)
    # StageFinished is delivered after the replace: the copy is gone and the
    # file already carries the batch and its undo baseline.
    assert seen == [{"copies": [], "baseline": True}]
    assert _copies(fp.parent) == []
    assert len(ftmw.review_log(str(fp))) >= 1


@pytest.mark.parametrize("via", ["api", "pipeline", "cli"])
def test_review_apply_leaves_no_copy_and_agrees_on_every_interface(
    via, stage5_multi_source, stage5_multi_file, tmp_path
):
    actions = _remove_one_peak_in_each_of_several_windows(stage5_multi_file)
    ref = tmp_path / "ref.ftmw"
    shutil.copy(stage5_multi_source, ref)
    ftmw.review_apply(str(ref), actions=actions)

    fp = tmp_path / f"{via}.ftmw"
    shutil.copy(stage5_multi_source, fp)
    if via == "api":
        ftmw.review_apply(str(fp), actions=actions)
    elif via == "pipeline":
        Pipeline.open(fp).review_apply(actions=actions)
    else:
        spec = tmp_path / "actions.json"
        spec.write_text(json.dumps(actions))
        assert cli_main(["review", "apply", str(fp), "--actions", str(spec)]) == 0

    assert _copies(tmp_path) == []
    assert _has_baseline(fp)

    def peaks(path: Path) -> Dict[int, List[float]]:
        with h5py.File(path, "r") as f:
            by_w = read_fit_peak_frequencies_by_window(f["stage5_fitting"])
        return {int(w): sorted(fs) for w, fs in by_w.items()}

    got, want = peaks(fp), peaks(ref)
    assert got.keys() == want.keys()
    for w in want:
        assert got[w] == pytest.approx(want[w], abs=1e-6)
