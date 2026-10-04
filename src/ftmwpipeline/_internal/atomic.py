"""Atomic writes of pipeline (``.ftmw``) files -- the one place.

Normative spec: ``dev-docs/CONTRACT_STRATEGY.md`` §Events and cancellation →
§Crash safety.

Every public call that writes a pipeline file is exactly one transaction on
it, :func:`atomic_write`, opened at the call's ``_internal`` entry. Inside the
transaction every write lands in a *working copy* beside the file, named
``.<basename>.ftmw-tmp.<hostname>.<pid>``; when the call ends normally the
copy replaces the file with one :func:`os.replace`. A failure of any kind (a
cancel, a raising events callback, a bug, a ``SIGKILL``) leaves the file
exactly as it was before the call.

:func:`h5open` is *the* way package code opens a pipeline file. It resolves
the path through the transaction registry:

- inside a transaction on that file (on the thread that owns it), reads and
  writes go to the working copy once it exists; reads before the first write
  go to the file itself, whose content the copy will start from;
- a write-mode open with no transaction on that file raises
  :class:`RuntimeError`. That is a bug guard, not a contract error: it is how
  the package proves no write escapes a transaction.

The working copy is made lazily, at the first write-mode open, as a
*compacted* copy of the file (every root attribute and object copied into a
fresh file, which drops the dead space HDF5 never reclaims; see
:mod:`~ftmwpipeline._internal.compaction`). A truncating open (``"w"``) starts
the copy empty instead. A transaction that never opens for write makes no
copy and leaves the file (inode, mtime) untouched.

Concurrency:

- one in-process re-entrant lock per file serializes the writes of one process
  to one file; a nested :func:`atomic_write` on the same file and thread joins
  the outer transaction (it neither copies nor replaces);
- the file's ``(st_ino, st_size, st_mtime_ns)`` (or its absence) is recorded
  when the transaction begins; if it differs at commit (another process wrote
  the file meanwhile) the call raises
  :class:`~ftmwpipeline.file_manager.WriteConflictError` and the other write
  stands.

Before a transaction makes its working copy, every leftover copy of the same
file made on this host by a process that is no longer running is removed.
Copies from other hosts, and copies whose pid is alive, are never touched.

A forked child (a report or Stage 5 pool worker) gets fresh registry locks
(:func:`os.register_at_fork`), so a lock held at the fork never deadlocks it.
"""

from __future__ import annotations

import errno
import logging
import os
import shutil
import socket
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import h5py

logger = logging.getLogger(__name__)

PathLike = Union[str, "os.PathLike[str]"]

#: h5py open modes that write. ``"w"`` / ``"w-"`` / ``"x"`` also create.
WRITE_MODES = frozenset({"a", "r+", "w", "w-", "x"})
_TRUNCATING_MODES = frozenset({"w", "w-", "x"})

#: The infix of a working-copy (and compaction) file name.
TMP_INFIX = ".ftmw-tmp."

_Stat = Optional[Tuple[int, int, int]]


@dataclass
class _Transaction:
    """One active transaction on one target file."""

    target: str
    copy: str
    pid: int
    owner: int
    stat: _Stat
    materialized: bool = False
    #: A write-mode open of the working copy succeeded: from then on the copy
    #: holds this call's writes, and losing it is a conflict, not a no-op.
    opened: bool = False


_registry_lock = threading.Lock()
_target_locks: Dict[str, "threading.RLock"] = {}
_active: Dict[str, _Transaction] = {}


def _reinit_after_fork() -> None:
    """Give a forked child fresh locks.

    A fork copies a lock in whatever state it had, and the thread that held it
    does not exist in the child, so a lock held by any other parent thread at
    the fork would never be released there. The child (a report ``fork_map``
    worker, a Stage 5 pool worker) only reads; it keeps a snapshot of the
    active transactions -- so its reads resolve as its parent's did, and a
    write-mode open still hits the forked-child guard in :func:`h5open` --
    behind new, unheld locks.
    """
    global _registry_lock, _target_locks, _active
    _registry_lock = threading.Lock()
    _target_locks = {}
    _active = dict(_active)


if hasattr(os, "register_at_fork"):  # POSIX
    os.register_at_fork(after_in_child=_reinit_after_fork)


# ---------------------------------------------------------------------------
# Names and paths
# ---------------------------------------------------------------------------


def _key(path: PathLike) -> str:
    """The registry key of *path*: absolute, symlinks resolved (so the replace
    lands on the real file, beside it)."""
    return os.path.realpath(os.fspath(path))


def tmp_copy_name(target: PathLike, *, pid: Optional[int] = None) -> str:
    """The working-copy path of *target* for this host and *pid* (default:
    this process): ``<dir>/.<basename>.ftmw-tmp.<hostname>.<pid>``."""
    directory, base = os.path.split(_key(target))
    who = os.getpid() if pid is None else int(pid)
    return os.path.join(directory, f".{base}{TMP_INFIX}{socket.gethostname()}.{who}")


def _stat(path: str) -> _Stat:
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    return (st.st_ino, st.st_size, st.st_mtime_ns)


def _pid_alive(pid: int) -> bool:
    """Whether *pid* names a running process on this host (conservative:
    anything uncertain counts as alive, so its copy is left alone)."""
    if pid <= 0:
        return True
    if pid == os.getpid():
        return True
    if sys.platform == "win32":  # pragma: no cover - exercised on Windows only
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFO
        if not handle:
            # 87 (ERROR_INVALID_PARAMETER): no such process. Anything else
            # (access denied, ...) means it exists or we cannot tell.
            return bool(ctypes.get_last_error() != 87)
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return bool(code.value == 259)  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:  # EPERM: alive under another user; anything else: unsure
        return True
    return True


def remove_stale_copies(target: PathLike) -> None:
    """Remove the leftover working copies of *target* made on this host by a
    process that is no longer running (never another host's, never a live
    pid's). Errors are ignored: a copy that cannot be removed is left."""
    directory, base = os.path.split(_key(target))
    prefix = f".{base}{TMP_INFIX}"
    host = socket.gethostname()
    try:
        names = os.listdir(directory)
    except OSError:
        return
    for name in names:
        if not name.startswith(prefix):
            continue
        copy_host, sep, pid_text = name[len(prefix) :].rpartition(".")
        if (
            not sep
            or copy_host != host
            or not pid_text.isascii()
            or not pid_text.isdigit()
        ):
            continue
        try:
            alive = _pid_alive(int(pid_text))
        except Exception:  # an unparsable or out-of-range pid: never ours
            alive = True
        if alive:
            continue
        try:
            os.remove(os.path.join(directory, name))
            logger.debug("Removed a leftover working copy %s", name)
        except OSError:
            pass


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("Could not remove the working copy %s: %s", path, exc)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def _lock_for(key: str) -> "threading.RLock":
    with _registry_lock:
        lock = _target_locks.get(key)
        if lock is None:
            lock = threading.RLock()
            _target_locks[key] = lock
        return lock


def _current(path: PathLike) -> Optional[_Transaction]:
    """The active transaction on *path* that this thread owns, or ``None``."""
    key = _key(path)
    with _registry_lock:
        txn = _active.get(key)
    if txn is None or txn.owner != threading.get_ident():
        return None
    return txn


def in_transaction(path: PathLike) -> bool:
    """Whether this thread is inside a transaction on *path*."""
    return _current(path) is not None


def resolve(path: PathLike) -> str:
    """The path this thread's reads of *path* go to now: the working copy once
    a transaction on *path* has made it, else *path* itself.

    For the few places that look at the file other than through
    :func:`h5open` (an existence check, an ``os.stat`` fingerprint).
    """
    txn = _current(path)
    if txn is not None and txn.materialized:
        return txn.copy
    return os.fspath(path)


def exists(path: PathLike) -> bool:
    """Whether the pipeline file *path* exists as this thread sees it: inside a
    transaction that has created or copied it, the working copy counts."""
    return os.path.exists(resolve(path))


# ---------------------------------------------------------------------------
# Copying
# ---------------------------------------------------------------------------


def copy_compacted(src: str, dst: str) -> None:
    """Write *src* into a fresh *dst* without its dead space.

    Copies the root attributes and every root-level object (with everything
    beneath it -- ``h5py``'s object copy carries attributes, chunking,
    ``maxshape`` and dtypes across intact). Raises on failure (*dst* may be
    left partial; the caller removes it).
    """
    with h5py.File(src, "r") as s, h5py.File(dst, "w") as d:
        for key, value in s.attrs.items():
            d.attrs[key] = value
        for member in s:
            s.copy(member, d, name=member)


def _materialize(txn: _Transaction, *, truncate: bool) -> None:
    """Make *txn*'s working copy (once): compacted from the target, or -- for
    a truncating open, or a target that does not exist -- left for the open to
    create."""
    if txn.materialized:
        return
    if os.path.exists(txn.target) and not os.access(txn.target, os.W_OK):
        # Refused as an in-place write-mode open of the file would be: the
        # replace would otherwise land on a file this process may not write.
        raise PermissionError(
            errno.EACCES,
            "Permission denied: cannot write the pipeline file",
            txn.target,
        )
    if not truncate and os.path.exists(txn.target):
        try:
            copy_compacted(txn.target, txn.copy)
        except Exception as exc:
            logger.warning(
                "Could not compact %s into its working copy (%s); copying it as is",
                txn.target,
                exc,
            )
            _remove(txn.copy)
            shutil.copy2(txn.target, txn.copy)
    txn.materialized = True


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def h5open(path: PathLike, mode: str = "r", **kwargs: Any) -> h5py.File:
    """Open a pipeline file -- the one way package code does.

    Inside a transaction on *path* (this thread's), reads go to the working
    copy once it exists and every write-mode open goes to it (making it first).
    Outside one, a read opens *path* itself.

    Raises
    ------
    RuntimeError
        For a write-mode open (``"a"``, ``"r+"``, ``"w"``, ``"w-"``, ``"x"``)
        with no transaction on *path*: a bug guard -- every write must be inside
        :func:`atomic_write`.
    PermissionError
        For the first write-mode open of a transaction whose target exists and
        is not writable by this process (as an in-place open would be).
    """
    txn = _current(path)
    if mode not in WRITE_MODES:
        if txn is not None and txn.materialized:
            return h5py.File(txn.copy, mode, **kwargs)
        return h5py.File(os.fspath(path), mode, **kwargs)
    if txn is None:
        raise RuntimeError(
            f"write-mode open ({mode!r}) of {os.fspath(path)} outside an "
            "atomic_write transaction"
        )
    if txn.pid != os.getpid():
        raise RuntimeError(
            f"write-mode open ({mode!r}) of {os.fspath(path)} from a forked "
            "child of the process that owns its transaction"
        )
    truncating = mode in _TRUNCATING_MODES
    if mode in ("w-", "x") and (txn.materialized or os.path.exists(txn.target)):
        raise FileExistsError(f"Unable to create file {os.fspath(path)}: it exists")
    if mode == "r+" and not txn.materialized and not os.path.exists(txn.target):
        raise FileNotFoundError(f"No such pipeline file: {os.fspath(path)}")
    _materialize(txn, truncate=truncating)
    handle = h5py.File(txn.copy, "w" if truncating else mode, **kwargs)
    txn.opened = True
    return handle


@contextmanager
def atomic_write(path: PathLike) -> Iterator[None]:
    """One atomic write of the pipeline file *path* (§Crash safety).

    Every :func:`h5open` of *path* on this thread inside the block works on a
    working copy; a normal exit replaces *path* with it (if anything was
    written), any exception discards it and re-raises unchanged.

    Nested on the same file and thread, the inner block joins the outer
    transaction. Another thread's transaction on the same file waits for this
    one to end.

    Raises
    ------
    WriteConflictError
        At exit, when *path* changed on disk after the transaction began
        (another process wrote it); the working copy is discarded.
    OSError
        When the final replace fails; the file is unchanged.
    """
    key = _key(path)
    lock = _lock_for(key)
    with lock:
        outer = _current(key)
        if outer is not None:
            yield
            return
        remove_stale_copies(key)
        txn = _Transaction(
            target=key,
            copy=tmp_copy_name(key),
            pid=os.getpid(),
            owner=threading.get_ident(),
            stat=_stat(key),
        )
        with _registry_lock:
            _active[key] = txn
        try:
            try:
                yield
            except BaseException:
                if txn.pid == os.getpid():
                    _remove(txn.copy)
                raise
            if txn.pid == os.getpid():
                _commit(txn)
        finally:
            with _registry_lock:
                if _active.get(key) is txn:
                    del _active[key]


def _commit(txn: _Transaction) -> None:
    """Replace the target with the working copy (or do nothing when nothing
    was written).

    Raises
    ------
    WriteConflictError
        When the target changed on disk since the transaction began, or when
        the working copy this call wrote into has vanished (removed by another
        process): either way this call's writes cannot land as written.
    """
    from ..file_manager import WriteConflictError

    if not txn.materialized:
        _remove(txn.copy)
        return
    if not os.path.exists(txn.copy):
        if txn.opened:
            raise WriteConflictError(txn.target)
        return  # a truncating open that never created the copy: nothing written
    try:
        if _stat(txn.target) != txn.stat:
            raise WriteConflictError(txn.target)
        if txn.stat is not None:
            shutil.copymode(txn.target, txn.copy)
        _fsync(txn.copy)
        os.replace(txn.copy, txn.target)
    except BaseException:
        _remove(txn.copy)
        raise


def _fsync(path: str) -> None:
    """Flush *path*'s data to disk before it replaces the target, so the
    replace never publishes a file whose blocks are still only in memory
    (best effort: a platform or filesystem that refuses is not an error)."""
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def compact_now(path: PathLike) -> None:
    """Rewrite *path* compacted, atomically, as its own transaction (or do
    nothing inside a transaction on it: its copy is already compacted).
    Raises on failure; :func:`~ftmwpipeline._internal.compaction.compact_file`
    is the best-effort wrapper."""
    if in_transaction(path):
        return
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such file: {os.fspath(path)}")
    with atomic_write(path):
        txn = _current(path)
        assert txn is not None
        _materialize(txn, truncate=False)


__all__ = [
    "TMP_INFIX",
    "WRITE_MODES",
    "atomic_write",
    "compact_now",
    "copy_compacted",
    "exists",
    "h5open",
    "in_transaction",
    "remove_stale_copies",
    "resolve",
    "tmp_copy_name",
]
