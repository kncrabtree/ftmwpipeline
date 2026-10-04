"""Atomic writes of a ``.ftmw`` file: ``atomic_write`` and ``h5open``.

Spec: ``dev-docs/CONTRACT_STRATEGY.md`` §Events and cancellation →
§Crash safety. These tests use a bare HDF5 file (the transaction does not care
what is in the file), so the promise is checked without a pipeline fixture:

- the working copy: its name, that writes go to it and reads in the
  transaction see them, and that the target is untouched until the commit;
- any failure (including a failed ``os.replace``) leaves the target
  byte-identical and no copy behind;
- a write-mode open outside a transaction raises (the bug guard);
- a nested transaction joins the outer one, a second thread waits;
- leftover copies of a killed writer are removed by the next write, but only
  the same host's dead pids';
- a file another process wrote meanwhile is ``write_conflict`` and the other
  write stands;
- the copy is compacted, so the space a write frees is reclaimed by the next.

Writes only to pytest ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import socket
import stat
import subprocess
import sys
import threading
from pathlib import Path
from typing import List

import h5py
import numpy as np
import pytest

from ftmwpipeline._internal.atomic import (
    TMP_INFIX,
    atomic_write,
    exists,
    h5open,
    in_transaction,
    remove_stale_copies,
    resolve,
    tmp_copy_name,
)
from ftmwpipeline.cli.contract_commands import exit_code_for
from ftmwpipeline.file_manager import PipelineFileError, WriteConflictError
from tests.unit._internal.test_compaction import _churn, _dump

pytestmark = [pytest.mark.unit]


# ---- helpers ----------------------------------------------------------------


def _make(path: Path, **attrs) -> Path:
    with h5py.File(path, "w") as f:
        f.attrs["format"] = "1.0"
        for key, value in attrs.items():
            f.attrs[key] = value
        f.create_dataset("data", data=np.arange(100, dtype="f8"))
    return path


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copies(directory: Path) -> List[str]:
    """Every working-copy file in *directory* (the pattern of §Crash safety)."""
    return sorted(p.name for p in directory.iterdir() if TMP_INFIX in p.name)


def _attr(path: Path, key: str):
    with h5py.File(path, "r") as f:
        return f.attrs.get(key)


def _dead_pid() -> int:
    """The pid of a process that has run and been reaped."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


@pytest.fixture
def target(tmp_path: Path) -> Path:
    return _make(tmp_path / "exp.ftmw")


# ---- the working copy -------------------------------------------------------


def test_copy_name_pattern(tmp_path: Path) -> None:
    t = tmp_path / "my.exp.ftmw"
    expected = f".my.exp.ftmw.ftmw-tmp.{socket.gethostname()}.{os.getpid()}"
    assert os.path.basename(tmp_copy_name(t)) == expected
    # The copy sits beside the target, so the replace stays on one filesystem.
    assert os.path.dirname(tmp_copy_name(t)) == os.path.realpath(tmp_path)
    other = os.path.basename(tmp_copy_name(t, pid=4242))
    assert other == f".my.exp.ftmw.ftmw-tmp.{socket.gethostname()}.4242"


def test_the_copy_in_a_transaction_has_the_documented_name(target: Path) -> None:
    with atomic_write(target):
        with h5open(target, "a") as f:
            f.attrs["x"] = 1
        assert _copies(target.parent) == [os.path.basename(tmp_copy_name(target))]


def test_writes_go_to_the_copy_and_the_target_is_unchanged_until_commit(
    target: Path,
) -> None:
    before = _digest(target)
    with atomic_write(target):
        with h5open(target, "a") as f:
            f.attrs["written"] = 7
        # The target is byte-identical while the call is in flight ...
        assert _digest(target) == before
        assert _attr(target, "written") is None
        # ... and a read in the transaction sees the write.
        with h5open(target, "r") as f:
            assert f.attrs["written"] == 7
        assert os.path.exists(resolve(target))
        assert resolve(target) == tmp_copy_name(target)
    assert _attr(target, "written") == 7
    assert _copies(target.parent) == []


def test_reads_before_the_first_write_go_to_the_file_and_make_no_copy(
    target: Path,
) -> None:
    with atomic_write(target):
        assert in_transaction(target)
        assert resolve(target) == os.fspath(target)
        with h5open(target, "r") as f:
            assert f.attrs["format"] == "1.0"
        assert _copies(target.parent) == []
    assert not in_transaction(target)


def test_a_transaction_that_writes_nothing_does_not_replace_the_file(
    target: Path,
) -> None:
    before = os.stat(target)
    with atomic_write(target):
        with h5open(target, "r") as f:
            f.attrs["format"]
    after = os.stat(target)
    assert (after.st_ino, after.st_mtime_ns, after.st_size) == (
        before.st_ino,
        before.st_mtime_ns,
        before.st_size,
    )
    assert _copies(target.parent) == []


def test_a_commit_replaces_the_file_with_one_new_inode(target: Path) -> None:
    inode = os.stat(target).st_ino
    reader = h5py.File(target, "r")  # a reader that opened the old version
    try:
        with atomic_write(target):
            with h5open(target, "a") as f:
                f.attrs["new"] = 1
        # The reader keeps the version it opened (§Crash safety).
        assert "new" not in reader.attrs
    finally:
        reader.close()
    assert os.stat(target).st_ino != inode
    assert _attr(target, "new") == 1


def test_a_transaction_can_create_the_file(tmp_path: Path) -> None:
    t = tmp_path / "new.ftmw"
    with atomic_write(t):
        assert not exists(t)
        with h5open(t, "w") as f:
            f.attrs["born"] = 1
        # Inside the transaction the working copy counts as the file ...
        assert exists(t)
        # ... on disk the file appears only at the commit.
        assert not t.exists()
    assert t.exists() and _attr(t, "born") == 1
    assert _copies(tmp_path) == []


def test_a_truncating_open_starts_the_copy_empty(target: Path) -> None:
    with atomic_write(target):
        with h5open(target, "w") as f:
            f.attrs["only"] = 1
    with h5py.File(target, "r") as f:
        assert dict(f.attrs) == {"only": 1}
        assert "data" not in f


def test_exclusive_create_refuses_an_existing_file(target: Path) -> None:
    before = _digest(target)
    for mode in ("w-", "x"):
        with pytest.raises(FileExistsError):
            with atomic_write(target):
                h5open(target, mode)
    assert _digest(target) == before
    assert _copies(target.parent) == []


def test_read_write_open_of_a_missing_file_raises(tmp_path: Path) -> None:
    t = tmp_path / "absent.ftmw"
    with pytest.raises(FileNotFoundError):
        with atomic_write(t):
            h5open(t, "r+")
    assert not t.exists() and _copies(tmp_path) == []


def test_a_symlinked_target_is_replaced_through_the_link(
    target: Path, tmp_path: Path
) -> None:
    link = tmp_path / "link.ftmw"
    link.symlink_to(target)
    with atomic_write(link):
        with h5open(link, "a") as f:
            f.attrs["via_link"] = 1
        # The copy is beside the real file.
        assert _copies(target.parent) == [os.path.basename(tmp_copy_name(target))]
    assert link.is_symlink()
    assert _attr(target, "via_link") == 1


# ---- failure leaves the file as it was --------------------------------------


def test_an_exception_leaves_the_target_byte_identical_and_no_copy(
    target: Path,
) -> None:
    before = _digest(target)
    boom = RuntimeError("boom")
    with pytest.raises(RuntimeError) as info:
        with atomic_write(target):
            with h5open(target, "a") as f:
                f.attrs["half"] = 1
                del f["data"]
            raise boom
    assert info.value is boom  # re-raised unchanged
    assert _digest(target) == before
    assert _copies(target.parent) == []
    assert not in_transaction(target)


def test_a_base_exception_also_discards_the_copy(target: Path) -> None:
    before = _digest(target)
    with pytest.raises(KeyboardInterrupt):
        with atomic_write(target):
            with h5open(target, "a") as f:
                f.attrs["half"] = 1
            raise KeyboardInterrupt
    assert _digest(target) == before
    assert _copies(target.parent) == []


def test_a_failed_replace_raises_and_leaves_the_file(
    target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = _digest(target)
    real = os.replace
    seen = []

    def refuse(src, dst, *a, **k):
        if os.fspath(dst) == os.path.realpath(target):
            seen.append(os.fspath(src))
            raise PermissionError(13, "the file is held open")
        return real(src, dst, *a, **k)

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(PermissionError):
        with atomic_write(target):
            with h5open(target, "a") as f:
                f.attrs["never"] = 1
    assert seen == [tmp_copy_name(target)]  # it did try to replace
    monkeypatch.undo()
    assert _digest(target) == before
    assert _copies(target.parent) == []
    # The failure leaves nothing registered: the next write works.
    with atomic_write(target):
        with h5open(target, "a") as f:
            f.attrs["after"] = 1
    assert _attr(target, "after") == 1


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_the_mode_of_the_file_survives_a_commit(target: Path) -> None:
    os.chmod(target, 0o640)
    with atomic_write(target):
        with h5open(target, "a") as f:
            f.attrs["x"] = 1
    assert stat.S_IMODE(os.stat(target).st_mode) == 0o640


# ---- the bug guard -----------------------------------------------------------


@pytest.mark.parametrize("mode", ["a", "r+", "w", "w-", "x"])
def test_a_write_mode_open_outside_a_transaction_raises(
    target: Path, mode: str
) -> None:
    before = _digest(target)
    with pytest.raises(RuntimeError, match="atomic_write"):
        h5open(target, mode)
    assert _digest(target) == before
    assert _copies(target.parent) == []


def test_the_guard_is_per_file(target: Path, tmp_path: Path) -> None:
    other = _make(tmp_path / "other.ftmw")
    with atomic_write(target):
        with pytest.raises(RuntimeError, match="atomic_write"):
            h5open(other, "a")
    assert _copies(tmp_path) == []


def test_reads_outside_a_transaction_open_the_file(target: Path) -> None:
    with h5open(target, "r") as f:
        assert f.attrs["format"] == "1.0"
    assert not in_transaction(target)


# ---- nesting and threads -----------------------------------------------------


def test_a_nested_transaction_joins_the_outer_one(
    target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replaces = []
    real = os.replace

    def spy(src, dst, *a, **k):
        replaces.append(os.fspath(dst))
        return real(src, dst, *a, **k)

    monkeypatch.setattr(os, "replace", spy)
    before = _digest(target)
    with atomic_write(target):
        with h5open(target, "a") as f:
            f.attrs["outer"] = 1
        with atomic_write(target):
            with h5open(target, "a") as f:
                f.attrs["inner"] = 2
            assert _copies(target.parent) == [os.path.basename(tmp_copy_name(target))]
        # The inner block ended, but nothing has landed yet.
        assert _digest(target) == before
        assert in_transaction(target)
    assert replaces == [os.path.realpath(target)]  # one replace for both
    with h5py.File(target, "r") as f:
        assert (f.attrs["outer"], f.attrs["inner"]) == (1, 2)


def test_a_nested_failure_discards_the_whole_call(target: Path) -> None:
    before = _digest(target)
    with pytest.raises(ValueError):
        with atomic_write(target):
            with h5open(target, "a") as f:
                f.attrs["outer"] = 1
            with atomic_write(target):
                raise ValueError("inner")
    assert _digest(target) == before
    assert _copies(target.parent) == []


def test_a_second_thread_waits_for_the_transaction_and_reads_the_file(
    target: Path,
) -> None:
    inside = threading.Event()
    release = threading.Event()
    other_entered = threading.Event()
    seen = {}

    def holder() -> None:
        with atomic_write(target):
            with h5open(target, "a") as f:
                f.attrs["holder"] = 1
            inside.set()
            assert release.wait(30)

    def waiter() -> None:
        # Not the owner: its reads see the file, not the owner's copy.
        with h5open(target, "r") as f:
            seen["holder_visible_midway"] = "holder" in f.attrs
        with atomic_write(target):
            other_entered.set()
            seen["holder_visible_after"] = _attr_inside(target, "holder")
            with h5open(target, "a") as f:
                f.attrs["waiter"] = 2

    t1 = threading.Thread(target=holder)
    t2 = threading.Thread(target=waiter)
    t1.start()
    assert inside.wait(30)
    t2.start()
    # The waiter is parked on the file's lock while the holder is in flight.
    assert not other_entered.wait(0.5)
    release.set()
    t1.join(30)
    t2.join(30)
    assert not t1.is_alive() and not t2.is_alive()
    assert seen == {"holder_visible_midway": False, "holder_visible_after": 1}
    # The second transaction started from the first one's result.
    with h5py.File(target, "r") as f:
        assert (f.attrs["holder"], f.attrs["waiter"]) == (1, 2)
    assert _copies(target.parent) == []


def _attr_inside(path: Path, key: str):
    with h5open(path, "r") as f:
        return f.attrs.get(key)


# ---- leftover copies ---------------------------------------------------------


def _fabricate(directory: Path, base: str, host: str, pid) -> Path:
    p = directory / f".{base}{TMP_INFIX}{host}.{pid}"
    p.write_bytes(b"left by a killed writer")
    return p


def test_a_write_removes_a_dead_pid_copy_from_this_host(target: Path) -> None:
    stale = _fabricate(target.parent, target.name, socket.gethostname(), _dead_pid())
    with atomic_write(target):
        with h5open(target, "a") as f:
            f.attrs["x"] = 1
    assert not stale.exists()
    assert _copies(target.parent) == []


def test_even_a_transaction_that_writes_nothing_cleans_up(target: Path) -> None:
    # "Before making its own copy, every write removes ..." -- the cleanup
    # precedes the copy, so it does not depend on this call writing.
    stale = _fabricate(target.parent, target.name, socket.gethostname(), _dead_pid())
    with atomic_write(target):
        pass
    assert not stale.exists()


def test_a_hostname_with_dots_is_parsed(
    target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(socket, "gethostname", lambda: "node.lab.example.edu")
    stale = _fabricate(target.parent, target.name, "node.lab.example.edu", _dead_pid())
    foreign = _fabricate(target.parent, target.name, "lab.example.edu", _dead_pid())
    remove_stale_copies(target)
    assert not stale.exists()
    assert foreign.exists()


def test_copies_from_other_hosts_are_never_touched(target: Path) -> None:
    foreign = _fabricate(target.parent, target.name, "some-other-host", _dead_pid())
    with atomic_write(target):
        with h5open(target, "a") as f:
            f.attrs["x"] = 1
    assert foreign.exists()


def test_a_copy_whose_pid_is_alive_is_left_alone(target: Path) -> None:
    sleeper = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdin=subprocess.DEVNULL,
    )
    try:
        host = socket.gethostname()
        live = _fabricate(target.parent, target.name, host, sleeper.pid)
        mine = _fabricate(target.parent, target.name, host, os.getpid())
        remove_stale_copies(target)
        assert live.exists()
        # This process's own pid is alive too: never treated as stale.
        assert mine.exists()
        # A write by this process does not remove another live pid's copy.
        with atomic_write(target):
            with h5open(target, "a") as f:
                f.attrs["x"] = 1
        assert live.exists()
    finally:
        sleeper.kill()
        sleeper.wait()


def test_only_copies_of_the_same_target_are_removed(
    target: Path, tmp_path: Path
) -> None:
    dead = _dead_pid()
    host = socket.gethostname()
    sibling = _fabricate(tmp_path, "other.ftmw", host, dead)
    prefix_only = _fabricate(tmp_path, "exp.ftmw.bak", host, dead)
    junk = [
        tmp_path / f".{target.name}{TMP_INFIX}{host}.notapid",
        tmp_path / f".{target.name}{TMP_INFIX}nohost",
        tmp_path / target.name.replace("exp", "exp2"),
    ]
    for j in junk:
        j.write_bytes(b"x")
    with atomic_write(target):
        pass
    assert sibling.exists() and prefix_only.exists()
    assert all(j.exists() for j in junk)


def test_a_missing_directory_entry_is_not_an_error(tmp_path: Path) -> None:
    remove_stale_copies(tmp_path / "no_such_dir" / "x.ftmw")  # no raise


@pytest.mark.parametrize(
    "pid_text",
    [
        "9" * 40,  # out of range for any pid
        "\u0661\u0662\u0663",  # non-ASCII digits (str.isdigit() is True)
        "\u00b2",  # a superscript digit: isdigit() but not int()-able
    ],
)
def test_an_unusable_pid_never_raises_and_is_left_alone(
    target: Path, pid_text: str
) -> None:
    odd = _fabricate(target.parent, target.name, socket.gethostname(), pid_text)
    remove_stale_copies(target)  # no raise
    with atomic_write(target):
        with h5open(target, "a") as f:
            f.attrs["x"] = 1
    assert odd.exists()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork")
def test_a_forked_child_never_deadlocks_on_a_lock_held_at_the_fork(
    target: Path,
) -> None:
    """A lock another parent thread holds at the fork is never released in the
    child; the child gets fresh locks, reads as its parent did, and its
    write-mode open still hits the forked-child guard."""
    from ftmwpipeline._internal import atomic

    held, release = threading.Event(), threading.Event()

    def hold() -> None:
        with atomic._registry_lock:
            held.set()
            release.wait(10)

    holder = threading.Thread(target=hold)
    holder.start()
    held.wait(10)
    try:
        with atomic_write(target):
            with h5open(target, "a") as f:
                f.attrs["x"] = 1
            r, w = os.pipe()
            pid = os.fork()
            if pid == 0:  # pragma: no cover - the child
                code = 1
                try:
                    with h5open(target, "r") as f:
                        sees = int(f.attrs.get("x", 0))
                    try:
                        h5open(target, "a")
                    except RuntimeError as exc:
                        code = 0 if (sees == 1 and "forked" in str(exc)) else 2
                finally:
                    os.write(w, bytes([code]))
                    os._exit(0)
            os.close(w)
            got = os.read(r, 1)
            os.close(r)
            os.waitpid(pid, 0)
    finally:
        release.set()
        holder.join()
    assert got == bytes([0])


# ---- write_conflict ----------------------------------------------------------


def _other_writer_replaces(target: Path, tmp_path: Path, marker: str) -> None:
    """Another process's atomic write: a new file replaces the target."""
    theirs = _make(tmp_path / "theirs.h5", other_writer=marker)
    os.replace(theirs, target)


def test_a_file_replaced_meanwhile_is_a_write_conflict(
    target: Path, tmp_path: Path
) -> None:
    with pytest.raises(WriteConflictError) as info:
        with atomic_write(target):
            with h5open(target, "a") as f:
                f.attrs["mine"] = 1
            _other_writer_replaces(target, tmp_path, "them")
    err = info.value
    assert err.code == "write_conflict"
    assert err.path == os.path.realpath(target)
    # The other write stands; none of this call's changes landed.
    assert _attr(target, "other_writer") == "them"
    assert _attr(target, "mine") is None
    assert _copies(tmp_path) == []
    assert not in_transaction(target)


def test_a_file_written_in_place_meanwhile_is_a_write_conflict(
    target: Path,
) -> None:
    with pytest.raises(WriteConflictError):
        with atomic_write(target):
            with h5open(target, "a") as f:
                f.attrs["mine"] = 1
            with h5py.File(target, "a") as f:  # another writer, in place
                f.attrs["theirs"] = "x" * 4096
    assert _attr(target, "theirs") == "x" * 4096
    assert _attr(target, "mine") is None
    assert _copies(target.parent) == []


def test_a_touched_mtime_is_a_write_conflict(target: Path) -> None:
    with pytest.raises(WriteConflictError):
        with atomic_write(target):
            with h5open(target, "a") as f:
                f.attrs["mine"] = 1
            st = os.stat(target)
            os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    assert _attr(target, "mine") is None
    assert _copies(target.parent) == []


def test_a_file_created_meanwhile_conflicts_with_a_create(tmp_path: Path) -> None:
    t = tmp_path / "new.ftmw"
    with pytest.raises(WriteConflictError):
        with atomic_write(t):
            with h5open(t, "w") as f:
                f.attrs["mine"] = 1
            _make(t, other_writer="them")  # another process created it first
    assert _attr(t, "other_writer") == "them"
    assert _attr(t, "mine") is None
    assert _copies(tmp_path) == []


def test_a_failing_body_is_not_masked_by_a_conflict(
    target: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="the real failure"):
        with atomic_write(target):
            with h5open(target, "a") as f:
                f.attrs["mine"] = 1
            _other_writer_replaces(target, tmp_path, "them")
            raise ValueError("the real failure")
    assert _attr(target, "other_writer") == "them"
    assert _copies(tmp_path) == []


def test_the_next_write_after_a_conflict_works(target: Path, tmp_path: Path) -> None:
    with pytest.raises(WriteConflictError):
        with atomic_write(target):
            with h5open(target, "a") as f:
                f.attrs["mine"] = 1
            _other_writer_replaces(target, tmp_path, "them")
    with atomic_write(target):
        with h5open(target, "a") as f:
            f.attrs["retry"] = 1
    with h5py.File(target, "r") as f:
        assert f.attrs["retry"] == 1 and f.attrs["other_writer"] == "them"


def test_a_working_copy_that_vanished_is_a_write_conflict(target: Path) -> None:
    before = _digest(target)
    with pytest.raises(WriteConflictError):
        with atomic_write(target):
            with h5open(target, "a") as f:
                f.attrs["x"] = 1
            os.remove(tmp_copy_name(target))  # another process removed it
    assert _digest(target) == before


@pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="POSIX permissions (root ignores them)",
)
def test_a_target_this_process_may_not_write_is_refused(target: Path) -> None:
    before = _digest(target)
    os.chmod(target, 0o444)
    try:
        with atomic_write(target):  # a transaction that only reads is fine
            with h5open(target, "r") as f:
                assert f.attrs["format"] == "1.0"
        with pytest.raises(PermissionError):
            with atomic_write(target):
                h5open(target, "a")
    finally:
        os.chmod(target, 0o644)
    assert _digest(target) == before
    assert _copies(target.parent) == []


def test_write_conflict_error_contract() -> None:
    err = WriteConflictError("/data/exp.ftmw")
    assert isinstance(err, PipelineFileError)
    assert err.code == "write_conflict" and err.path == "/data/exp.ftmw"
    d = err.to_dict()
    assert d["schema"] == "ftmw/error@1" and d["code"] == "write_conflict"
    assert d["path"] == "/data/exp.ftmw" and d["message"] == str(err)
    assert set(d) == {"schema", "code", "message", "path"}
    assert "/data/exp.ftmw" in str(err)
    back = pickle.loads(pickle.dumps(err))
    assert type(back) is WriteConflictError and back.to_dict() == d
    assert exit_code_for(err) == 1


# ---- compaction at copy-in ----------------------------------------------------


@pytest.fixture
def bloated(tmp_path: Path) -> Path:
    """A file with ~1 MB of dead space HDF5 never returns on its own (a
    rewritten big attribute, a re-created vlen dataset), plus live content."""
    p = tmp_path / "bloated.ftmw"
    with h5py.File(p, "w") as f:
        f.attrs["format"] = "1.0"
        f.create_dataset("fid", data=np.arange(100_000, dtype="f8"), chunks=(1000,))
    _churn(p, 10)
    return p


def test_the_copy_is_compacted_so_the_next_write_reclaims_dead_space(
    bloated: Path,
) -> None:
    before = os.path.getsize(bloated)
    content = _dump(bloated)
    with atomic_write(bloated):
        with h5open(bloated, "a") as f:
            f.attrs["small"] = 1
    after = os.path.getsize(bloated)
    assert after < before / 2
    expected = dict(content)
    expected["/attrs"] = {**content["/attrs"], "small": 1}
    assert _dump(bloated) == expected


def test_space_freed_by_one_write_is_reclaimed_by_the_next(tmp_path: Path) -> None:
    p = tmp_path / "freed.ftmw"
    with h5py.File(p, "w") as f:
        f.attrs["format"] = "1.0"
        f.create_dataset("big", data=np.zeros(1_000_000, dtype="f8"), chunks=(10_000,))
    big = os.path.getsize(p)
    assert big > 7_000_000
    with atomic_write(p):  # write 1 frees the big dataset
        with h5open(p, "a") as f:
            del f["big"]
    with atomic_write(p):  # write 2 starts from a compacted copy
        with h5open(p, "a") as f:
            f.attrs["next"] = 1
    assert os.path.getsize(p) < big / 10
    with h5py.File(p, "r") as f:
        assert "big" not in f and f.attrs["next"] == 1


def test_compaction_keeps_attributes_chunking_and_dtypes(tmp_path: Path) -> None:
    p = tmp_path / "rich.ftmw"
    with h5py.File(p, "w") as f:
        f.attrs["root"] = "r"
        g = f.create_group("g")
        g.attrs["json"] = '{"a": 1}'
        ds = g.create_dataset(
            "table",
            data=np.arange(2000, dtype="f4"),
            maxshape=(None,),
            chunks=(500,),
            compression="gzip",
        )
        ds.attrs["unit"] = "MHz"
        g.create_dataset(
            "names",
            data=np.array(["a", "bb", "ccc"], dtype=object),
            dtype=h5py.string_dtype(),
        )
    content = _dump(p)
    with atomic_write(p):
        with h5open(p, "a") as f:
            f.attrs["touched"] = 1
    expected = dict(content)
    expected["/attrs"] = {**content["/attrs"], "touched": 1}
    assert _dump(p) == expected
