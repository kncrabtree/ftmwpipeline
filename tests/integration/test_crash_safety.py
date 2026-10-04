"""Crash safety on real pipeline files (CONTRACT_STRATEGY §Crash safety).

Every call that writes a ``.ftmw`` writes it atomically: a temporary copy beside
the file, replaced onto it with one ``os.replace`` when the call ends. Checked
here on the 2638 baselines, through the functional API, ``Pipeline`` and the
CLI:

- after a writing call no working copy remains, and the result is correct;
- ``Invalidated`` and ``StageFinished`` arrive after the replace, and a callback
  that raises on either leaves the write in place;
- a cancel, or a callback that raises while the copy is being written, leaves
  the file byte-identical (not merely equal in content);
- reads leave the file untouched (same inode);
- ``SIGKILL`` at any point leaves the file byte-identical, the leftover copy is
  removed by the next write, and ``run_pipeline`` keeps the stages that
  finished;
- a file another process wrote meanwhile is ``write_conflict`` (exit 1) and the
  other write stands.

Writes only to pytest ``tmp_path``.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import (
    CallbackFailedError,
    Invalidated,
    OperationCancelledError,
    StageFinished,
    StageStarted,
    WindowProgress,
    WriteConflictError,
)
from ftmwpipeline._internal import atomic
from ftmwpipeline._internal.atomic import TMP_INFIX, atomic_write, h5open, tmp_copy_name
from ftmwpipeline.cli.main import main as cli_main
from ftmwpipeline.pipeline import Pipeline
from tests._events_support import Recorder, Token, cancel_on_nth

pytestmark = [pytest.mark.integration]

_POSIX = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals and pids")
_NEW_TRIM = (27000.0, 39000.0)


# ---- helpers ------------------------------------------------------------------


def _copy(src: Path, tmp_path: Path, name: str = "work.ftmw") -> Path:
    dest = tmp_path / name
    shutil.copy(src, dest)
    return dest


def _copies(directory: Path) -> List[str]:
    """The working-copy files (§Crash safety pattern) in *directory*."""
    return sorted(p.name for p in directory.iterdir() if TMP_INFIX in p.name)


def _bytes(path: Path) -> bytes:
    return path.read_bytes()


def _states(path: Path) -> Dict[str, str]:
    return {row["stage"]: row["state"] for row in ftmw.status(path)["stages"]}


def _cli(argv: List[str], capsys) -> int:
    capsys.readouterr()
    return cli_main(argv)


def _row(path: Path, knob: str):
    return {r.path: r for r in ftmw.settings_show(path)}[knob]


def _as_tuples(clocks):
    return [(c.freq_mhz, c.locked, c.label) for c in (clocks or ())]


def _native_ftmw(tmp_path: Path, name: str = "exp.ftmw") -> Path:
    """A minimal imported ``.ftmw`` (native-HDF5 source, no embedded clocks)."""
    src = tmp_path / f"{name}.src.h5"
    sig = np.cos(2 * np.pi * np.arange(1024) * 0.01)
    with h5py.File(src, "w") as f:
        f.attrs["ftmw_input_version"] = 1
        f.attrs["spacing_us"] = 0.02
        f.attrs["probe_freq_mhz"] = 40960.0
        f.create_dataset("fid", data=sig)
    out = tmp_path / name
    ftmw.import_data(str(out), source=str(src), format_name="ftmw-hdf5")
    return out


# ---- no copy remains, and the result is correct, on every interface ------------


@pytest.mark.parametrize("via", ["api", "pipeline", "cli"])
def test_noise_run_leaves_no_copy_and_completes(
    via, baseline_2638_stage1_raw, tmp_path, capsys
):
    fp = _copy(baseline_2638_stage1_raw, tmp_path)
    assert _states(fp)["noise"] == "not_run"
    if via == "api":
        ftmw.estimate_noise(fp)
    elif via == "pipeline":
        Pipeline.open(fp).estimate_noise()
    else:
        assert _cli(["noise", "run", str(fp)], capsys) == 0
    assert _states(fp)["noise"] == "complete"
    assert _copies(tmp_path) == []
    assert sorted(p.name for p in tmp_path.iterdir()) == [fp.name]


@pytest.mark.parametrize("via", ["api", "pipeline", "cli"])
def test_settings_set_and_unset_leave_no_copy(
    via, baseline_2638_stage2, tmp_path, capsys
):
    fp = _copy(baseline_2638_stage2, tmp_path)
    knob = "stage2.window_mhz"
    assert _states(fp)["noise"] == "complete"
    if via == "api":
        ftmw.settings_set(fp, knob, "111")
    elif via == "pipeline":
        Pipeline.open(fp).settings_set(knob, "111")
    else:
        assert _cli(["settings", "set", str(fp), knob, "111"], capsys) == 0
    row = _row(fp, knob)
    assert row.value == 111.0 and row.source == ".ftmw"
    # The setting moved the stage and everything built on it.
    assert _states(fp)["noise"] == "not_run"
    assert _copies(tmp_path) == []

    if via == "api":
        ftmw.settings_unset(fp, knob)
    elif via == "pipeline":
        Pipeline.open(fp).settings_unset(knob)
    else:
        assert _cli(["settings", "unset", str(fp), knob], capsys) == 0
    assert _row(fp, knob).source != ".ftmw"
    assert _copies(tmp_path) == []


@pytest.mark.parametrize("via", ["api", "pipeline", "cli"])
def test_clocks_verbs_leave_no_copy(via, tmp_path, capsys):
    fp = _native_ftmw(tmp_path)
    p = Pipeline.open(fp)
    if via == "api":
        ftmw.set_clock_sources(
            str(fp), [{"freq_mhz": 5760.0, "locked": True, "label": "synth"}]
        )
    elif via == "pipeline":
        p.set_clock_sources([{"freq_mhz": 5760.0, "locked": True, "label": "synth"}])
    else:
        assert _cli(["clocks", "set", str(fp), "5760:locked:synth"], capsys) == 0
    assert _as_tuples(ftmw.get_clock_sources(str(fp))) == [(5760.0, True, "synth")]
    assert _copies(tmp_path) == []

    # add, remove, clear: each is its own atomic write.
    if via == "api":
        ftmw.set_clock_sources(
            str(fp),
            [{"freq_mhz": 6250.0, "locked": False, "label": "dig"}],
            replace=False,
        )
    elif via == "pipeline":
        Pipeline.open(fp).set_clock_sources(
            [{"freq_mhz": 6250.0, "locked": False, "label": "dig"}], replace=False
        )
    else:
        assert _cli(["clocks", "add", str(fp), "6250:free:dig"], capsys) == 0
    assert _as_tuples(ftmw.get_clock_sources(str(fp))) == [
        (5760.0, True, "synth"),
        (6250.0, False, "dig"),
    ]
    assert _copies(tmp_path) == []

    if via == "api":
        ftmw.remove_clock_sources(str(fp), [6250.0])
    elif via == "pipeline":
        Pipeline.open(fp).remove_clock_sources([6250.0])
    else:
        assert _cli(["clocks", "remove", str(fp), "6250"], capsys) == 0
    assert _as_tuples(ftmw.get_clock_sources(str(fp))) == [(5760.0, True, "synth")]

    if via == "api":
        ftmw.clear_clock_sources(str(fp))
    elif via == "pipeline":
        Pipeline.open(fp).clear_clock_sources()
    else:
        assert _cli(["clocks", "clear", str(fp)], capsys) == 0
    assert ftmw.get_clock_sources(str(fp)) is None
    assert _copies(tmp_path) == []


def test_a_first_import_leaves_only_the_file(exp_2638_data_path, tmp_path):
    out = tmp_path / "fresh.ftmw"
    ftmw.import_data(out, source=exp_2638_data_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["fresh.ftmw"]
    assert _states(out)["data"] == "complete"


# ---- reads leave the file untouched ----------------------------------------------


def _identity(path: Path):
    st = os.stat(path)
    return (st.st_ino, st.st_mtime_ns, st.st_size)


def test_reads_and_previews_leave_the_file_untouched(baseline_2638_stage2, tmp_path):
    fp = _copy(baseline_2638_stage2, tmp_path)
    before = _identity(fp)
    digest = _bytes(fp)
    ftmw.status(fp)
    ftmw.settings_show(fp)
    ftmw.get_pipeline_info(fp)
    ftmw.compute_ft(fp, from_saved_params=True)
    assert _identity(fp) == before
    assert _bytes(fp) == digest
    assert _copies(tmp_path) == []


def test_importing_an_identical_source_again_leaves_the_file_untouched(
    exp_2638_data_path, tmp_path
):
    out = tmp_path / "again.ftmw"
    ftmw.import_data(out, source=exp_2638_data_path)
    before = _identity(out)
    ftmw.import_data(out, source=exp_2638_data_path)
    assert _identity(out) == before
    assert _copies(tmp_path) == []


# ---- StageFinished arrives after the replace --------------------------------------

_STAGE_CASES = {
    "ft": (
        "baseline_2638_stage2",
        lambda p, **k: ftmw.compute_ft(p, trim=_NEW_TRIM, **k),
    ),
    "noise": ("baseline_2638_stage1_raw", lambda p, **k: ftmw.estimate_noise(p, **k)),
    "peaks": ("baseline_2638_stage2", lambda p, **k: ftmw.detect_peaks(p, **k)),
    "windows": ("baseline_2638_stage3", lambda p, **k: ftmw.assign_windows(p, **k)),
    "fit": (
        "baseline_2638_stage4_small",
        lambda p, **k: ftmw.fit_peaks(p, jobs=1, **k),
    ),
}


@pytest.mark.parametrize("case", sorted(_STAGE_CASES))
def test_stage_finished_arrives_after_the_replace(case, request, tmp_path):
    fixture, call = _STAGE_CASES[case]
    fp = _copy(request.getfixturevalue(fixture), tmp_path)
    inode_before = os.stat(fp).st_ino
    seen: List[Dict[str, Any]] = []

    def probe(event):
        if isinstance(event, StageFinished):
            seen.append(
                {
                    "copies": _copies(tmp_path),
                    "state": _states(fp)[case],
                    "inode": os.stat(fp).st_ino,
                }
            )

    call(fp, events=probe)
    assert len(seen) == 1
    got = seen[0]
    # The copy is gone, the file is already the new one, and a reader inside
    # the callback sees the stage complete.
    assert got["copies"] == []
    assert got["inode"] != inode_before
    assert got["state"] == "complete"


def test_import_finishes_after_the_replace(exp_2638_data_path, tmp_path):
    out = tmp_path / "imp.ftmw"
    seen: List[Dict[str, Any]] = []

    def probe(event):
        if isinstance(event, StageFinished):
            seen.append(
                {
                    "exists": out.exists(),
                    "copies": _copies(tmp_path),
                    "state": _states(out)["data"] if out.exists() else None,
                }
            )

    ftmw.import_data(out, source=exp_2638_data_path, events=probe)
    assert seen == [{"exists": True, "copies": [], "state": "complete"}]


def test_a_callback_raising_on_stage_finished_leaves_the_write_in_place(
    baseline_2638_stage1_raw, tmp_path
):
    """The stage's write is done and replaced before ``StageFinished`` is
    delivered, so the listener's failure cannot undo it."""
    fp = _copy(baseline_2638_stage1_raw, tmp_path)
    before = _bytes(fp)
    boom = RuntimeError("listener broke")

    def bad(event):
        if isinstance(event, StageFinished):
            raise boom

    with pytest.raises(CallbackFailedError) as info:
        ftmw.estimate_noise(fp, events=bad)
    assert info.value.event_schema == "ftmw/stage_finished@1"
    assert info.value.__cause__ is boom
    assert _bytes(fp) != before
    assert _states(fp)["noise"] == "complete"
    assert _copies(tmp_path) == []


# ---- a cancel or a failure discards the copy: true bytes ----------------------------


def test_cancel_before_a_stage_leaves_the_bytes(baseline_2638_stage1_raw, tmp_path):
    fp = _copy(baseline_2638_stage1_raw, tmp_path)
    before = _bytes(fp)
    tok = Token()
    tok.set()
    with pytest.raises(OperationCancelledError):
        ftmw.estimate_noise(fp, cancel=tok)
    assert _bytes(fp) == before
    assert _copies(tmp_path) == []


@pytest.mark.parametrize("via", ["api", "pipeline"])
def test_cancel_in_the_fit_before_any_window_finishes_leaves_the_bytes(
    via, baseline_2638_stage4_small, tmp_path
):
    # Cancelled once the fit has begun but before a window finished: nothing
    # to keep as a partial fit, so nothing is written (a cancel after a window
    # finishes keeps a partial fit; see test_stage5_partial.py).
    fp = _copy(baseline_2638_stage4_small, tmp_path)
    before = _bytes(fp)
    inode = os.stat(fp).st_ino
    tok = Token()
    rec = Recorder(cancel_on_nth(StageStarted, 1, tok))
    with pytest.raises(OperationCancelledError) as info:
        if via == "api":
            ftmw.fit_peaks(fp, jobs=1, events=rec, cancel=tok)
        else:
            Pipeline.open(fp).fit_peaks(jobs=1, events=rec, cancel=tok)
    assert info.value.completed_windows == []
    assert _bytes(fp) == before
    assert os.stat(fp).st_ino == inode  # never replaced
    assert _copies(tmp_path) == []


def test_invalidated_arrives_after_the_replace(baseline_2638_stage2, tmp_path):
    """``ft run`` with a new trim invalidates the stages built on the old FT.
    ``Invalidated`` is delivered once the write is durable, just before
    ``StageFinished``: the copy is gone and a reader inside the callback sees
    the new FT and the invalidation."""
    fp = _copy(baseline_2638_stage2, tmp_path)
    inode_before = os.stat(fp).st_ino
    seen: List[Any] = []

    def probe(event):
        if isinstance(event, Invalidated):
            seen.append(
                {
                    "copies": _copies(tmp_path),
                    "inode": os.stat(fp).st_ino,
                    "states": _states(fp),
                }
            )
        seen.append(type(event))

    ftmw.compute_ft(fp, trim=_NEW_TRIM, events=probe)
    got = seen[seen.index(Invalidated) - 1]
    assert got["copies"] == []
    assert got["inode"] != inode_before
    assert got["states"]["ft"] == "complete"
    assert got["states"]["noise"] == "not_run"
    assert seen[-2:] == [Invalidated, StageFinished]


def test_a_callback_raising_on_invalidated_leaves_the_write_in_place(
    baseline_2638_stage2, tmp_path
):
    """The write is done and replaced before ``Invalidated`` is delivered, so a
    listener that raises on it fails the call with ``callback_failed`` but
    cannot undo the write; no ``StageFinished`` follows."""
    fp = _copy(baseline_2638_stage2, tmp_path)
    before = _bytes(fp)
    seen: List[type] = []

    def bad(event):
        seen.append(type(event))
        if isinstance(event, Invalidated):
            raise ValueError("no")

    with pytest.raises(CallbackFailedError) as info:
        ftmw.compute_ft(fp, trim=_NEW_TRIM, events=bad)
    assert info.value.event_schema == "ftmw/invalidated@1"
    assert isinstance(info.value.__cause__, ValueError)
    assert seen[-1] is Invalidated and StageFinished not in seen
    # The write stands: the new FT, and the stages built on the old one gone.
    assert _bytes(fp) != before
    states = _states(fp)
    assert states["ft"] == "complete" and states["noise"] == "not_run"
    assert _copies(tmp_path) == []


def test_a_callback_failing_mid_call_leaves_the_bytes(
    baseline_2638_stage4_small, tmp_path, monkeypatch
):
    """A listener that raises on an event delivered before the replace (here a
    warning the fit emits before any window finishes) fails the call inside
    its transaction: nothing of the call reaches the file, which is never
    replaced."""
    import ftmwpipeline.fitting.plan_execution as pe

    fp = _copy(baseline_2638_stage4_small, tmp_path)
    before = _bytes(fp)
    inode = os.stat(fp).st_ino
    original = pe._walk_windows_parallel

    def warn_first(plan, order, *, events, **kwargs):
        events.warn("walk_fallback", reason="test", n_windows=len(order))
        return original(plan, order, events=events, **kwargs)

    monkeypatch.setattr(pe, "_walk_windows_parallel", warn_first)

    def bad(event):
        if getattr(event, "code", None) == "walk_fallback":
            raise ValueError("no")

    with pytest.raises(CallbackFailedError) as info:
        ftmw.fit_peaks(fp, jobs=1, events=bad)
    assert info.value.event_schema == "ftmw/warning@1"
    assert _bytes(fp) == before
    assert os.stat(fp).st_ino == inode
    assert _copies(tmp_path) == []


def test_a_callback_failing_after_a_window_keeps_a_partial_fit_in_one_replace(
    baseline_2638_stage4_small, tmp_path
):
    """A listener that raises on a finished window's ``WindowProgress`` leaves
    what a cancel at that point leaves: the finished window as a partial fit,
    written in the call's one replace (no working copy left behind)."""
    fp = _copy(baseline_2638_stage4_small, tmp_path)
    before = _bytes(fp)

    def bad(event):
        if isinstance(event, WindowProgress):
            raise ValueError("no")

    with pytest.raises(CallbackFailedError) as info:
        ftmw.fit_peaks(fp, jobs=1, events=bad)
    assert info.value.event_schema == "ftmw/window_progress@1"
    assert _bytes(fp) != before
    assert _states(fp)["fit"] == "partial"
    assert _copies(tmp_path) == []


def test_a_failing_write_inside_a_stage_leaves_the_bytes(
    baseline_2638_stage2, tmp_path, monkeypatch
):
    """Any exception, not only a cancel, discards the call's writes."""
    fp = _copy(baseline_2638_stage2, tmp_path)
    before = _bytes(fp)
    import ftmwpipeline._internal.stage2_impl as s2

    def boom(*a, **k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(s2, "_update_stage_completion", boom)
    with pytest.raises(Exception, match="disk on fire"):
        ftmw.estimate_noise(fp)
    monkeypatch.undo()
    assert _bytes(fp) == before
    assert _copies(tmp_path) == []


# ---- write_conflict ----------------------------------------------------------------------


def test_another_process_writing_meanwhile_is_a_write_conflict(
    baseline_2638_stage2, tmp_path
):
    fp = _copy(baseline_2638_stage2, tmp_path)
    knob = "stage2.window_mhz"
    with pytest.raises(WriteConflictError) as info:
        with atomic_write(fp):
            with h5open(fp, "a") as f:
                f.attrs["mine"] = 1
            # Another process's `settings set`, a complete atomic write.
            r = subprocess.run(
                [sys.executable, "-m", "ftmwpipeline", "settings", "set", str(fp)]
                + [knob, "111"],
                capture_output=True,
                text=True,
            )
            assert r.returncode == 0, r.stderr
    assert info.value.code == "write_conflict"
    assert info.value.path == os.path.realpath(fp)
    # The other write stands; none of this call's changes landed.
    assert _row(fp, knob).value == 111.0
    with h5py.File(fp, "r") as f:
        assert "mine" not in f.attrs
    assert _copies(tmp_path) == []


def test_the_cli_reports_write_conflict_as_exit_1(
    baseline_2638_stage2, tmp_path, capsys, monkeypatch
):
    """A conflict is detected at the replace: the file's identity at commit is
    made to differ from the one recorded when the call began."""
    fp = _copy(baseline_2638_stage2, tmp_path)
    before = _bytes(fp)
    calls: Dict[str, int] = {}
    real = atomic._stat

    def moved(path):
        calls[path] = calls.get(path, 0) + 1
        return real(path) if calls[path] == 1 else (0, 0, 0)

    monkeypatch.setattr(atomic, "_stat", moved)
    capsys.readouterr()
    rc = cli_main(["settings", "set", str(fp), "stage2.window_mhz", "111", "--json"])
    err = capsys.readouterr().err
    assert rc == 1
    payload = json.loads(
        [ln for ln in err.splitlines() if ln.strip().startswith("{")][-1]
    )
    assert payload["schema"] == "ftmw/error@1" and payload["code"] == "write_conflict"
    assert payload["path"] == os.path.realpath(fp)
    monkeypatch.undo()
    assert _bytes(fp) == before
    assert _copies(tmp_path) == []


# ---- SIGKILL ---------------------------------------------------------------------------------

_CHILD_PRELUDE = textwrap.dedent("""
    import sys, time
    import ftmwpipeline.api as ftmw
    from ftmwpipeline import Invalidated, StageStarted
    from ftmwpipeline.contract import Stage

    def hang():
        print("READY", flush=True)
        time.sleep(600)
    """)

# Mid-write: ``ft run`` with a new trim has written its whole working copy
# (the new FT, the invalidation) and is about to replace the file with it.
_CHILD_FT = _CHILD_PRELUDE + textwrap.dedent("""
    from ftmwpipeline._internal import atomic

    def commit(txn):
        hang()

    atomic._commit = commit
    ftmw.compute_ft(sys.argv[1], trim=(27000.0, 39000.0))
    """)

# Between stages of a whole run: once the noise stage starts, the data and FT
# stages have finished (and replaced).
_CHILD_RUN = _CHILD_PRELUDE + textwrap.dedent("""
    def cb(event):
        if isinstance(event, StageStarted) and event.stage is Stage.NOISE:
            hang()

    ftmw.run_pipeline(
        sys.argv[1], sys.argv[2], trim=(26500.0, 40000.0), detect_start=False,
        calibrate=False, progress=False, events=cb,
    )
    """)


def _spawn(code: str, *args: str, log: Path) -> subprocess.Popen:
    """Run *code* in a child (its own process group); stdout goes to
    ``<log>.out`` and stderr to *log*."""
    return subprocess.Popen(
        [sys.executable, "-c", code, *args],
        stdout=open(log.with_suffix(".out"), "w"),
        stderr=open(log, "w"),
        start_new_session=True,
    )


def _wait_ready(proc: subprocess.Popen, log: Path, timeout: float = 300.0) -> None:
    out = log.with_suffix(".out")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if "READY" in out.read_text().split():
            return
        if proc.poll() is not None:
            break
        time.sleep(0.05)
    raise AssertionError(f"the child never reached its hook:\n{log.read_text()}")


def _kill_group(proc: subprocess.Popen) -> None:
    os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()
    assert proc.returncode == -signal.SIGKILL


@_POSIX
def test_sigkill_mid_write_leaves_the_file_byte_identical(
    baseline_2638_stage2, tmp_path
):
    fp = _copy(baseline_2638_stage2, tmp_path)
    before = _bytes(fp)
    log = tmp_path / "child.log"
    proc = _spawn(_CHILD_FT, str(fp), log=log)
    try:
        _wait_ready(proc, log)
        # The child is inside its write: the copy exists, named for its pid,
        # and the file has not been touched.
        leftover = tmp_copy_name(fp, pid=proc.pid)
        assert _copies(tmp_path) == [os.path.basename(leftover)]
        assert _bytes(fp) == before
    finally:
        _kill_group(proc)
    # SIGKILL leaves the file exactly as it was, and it still opens.
    assert _bytes(fp) == before
    assert _states(fp)["noise"] == "complete"
    assert ftmw.get_pipeline_info(fp)
    # The copy is left behind (nothing ran to remove it) ...
    assert _copies(tmp_path) == [os.path.basename(leftover)]
    assert (tmp_path / _copies(tmp_path)[0]).exists()
    assert socket.gethostname() in _copies(tmp_path)[0]
    # ... and the next write to the file removes it, being the same host's
    # and a dead pid's.
    ftmw.settings_set(fp, "stage2.window_mhz", "111")
    assert _copies(tmp_path) == []
    assert _row(fp, "stage2.window_mhz").value == 111.0


@_POSIX
@pytest.mark.slow
def test_a_killed_run_pipeline_keeps_the_stages_that_finished(
    exp_2638_data_path, tmp_path
):
    out = tmp_path / "killed.ftmw"
    log = tmp_path / "child.log"
    proc = _spawn(_CHILD_RUN, exp_2638_data_path, str(out), log=log)
    try:
        _wait_ready(proc, log)
    finally:
        _kill_group(proc)
    # Each stage is its own atomic write: data and FT survive the kill, the
    # stage that had begun left nothing behind, and the file opens.
    states = _states(out)
    assert states["data"] == "complete" and states["ft"] == "complete"
    assert states["noise"] == "not_run" and states["peaks"] == "not_run"
    # The next write removes whatever the dead writer left.
    ftmw.estimate_noise(out)
    assert _copies(tmp_path) == []
    assert _states(out)["noise"] == "complete"


@_POSIX
@pytest.mark.slow
def test_sigkill_during_a_real_fit_run_leaves_one_of_the_two_files(
    baseline_2638_stage4_small, tmp_path
):
    """Kill a real ``fit run`` as soon as its working copy appears. Whatever
    the timing -- before the copy, with it present, or just after the replace --
    the file is the one before the call or the one the call completed, and it
    opens."""
    fp = _copy(baseline_2638_stage4_small, tmp_path)
    before = _bytes(fp)
    proc = subprocess.Popen(
        [sys.executable, "-m", "ftmwpipeline", "fit", "run", str(fp), "--jobs", "1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    killed = False
    deadline = time.monotonic() + 600
    try:
        while time.monotonic() < deadline and proc.poll() is None:
            if _copies(tmp_path):
                _kill_group(proc)
                killed = True
                break
            time.sleep(0.002)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    assert proc.returncode is not None
    fit_state = _states(fp)["fit"]
    if _bytes(fp) == before:
        assert fit_state == "not_run"
    else:
        # The replace had landed: the call's completed file, not a mix.
        assert fit_state == "complete"
    if killed:
        # Whatever copy remains is the dead writer's; the next write sweeps it.
        ftmw.settings_set(fp, "stage2.window_mhz", "111")
        assert _copies(tmp_path) == []
