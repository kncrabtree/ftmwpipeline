"""Reclaim dead space in a ``.ftmw`` file after a write that leaves some.

HDF5 never returns two kinds of space to a file, whatever file-space strategy
the file was created with: the object-header space a rewritten or deleted
*attribute* occupied, and the global-heap space a deleted or rewritten
*variable-length* dataset (every JSON / string column) occupied. Both are
exactly what this pipeline churns -- ``/stage6_review`` is JSON in attributes
and is deleted and recreated on every curation write, ``/stage5_fitting``
carries vlen columns that a refit rewrites and an undo's baseline restore
deletes and copies -- so a curated file grew by roughly a review group plus a
fit table per edit and never shrank (a measured 0.7 MB per single-window edit
and 1.2 MB per undo on a 6.5 MB build, until the file was three times its
content). Free-space tracking (``fs_strategy`` / ``fs_persist``) was measured
and does not help: the leaked space is not on any free list to begin with.

The remedy is the one ``h5repack`` uses -- copy every object into a fresh
file and swap it into place -- which costs tens of milliseconds on a file of
this size. Since every write is atomic (:mod:`~ftmwpipeline._internal.atomic`,
§Crash safety), that copy is the transaction's working copy: it is written
compacted when the transaction makes it, so the space a write frees is
reclaimed by the next write. There is deliberately no user-facing verb for
it: a file that is never bloated needs no command to un-bloat it.

:func:`compact_file` is a no-op inside a transaction on the file (its copy is
already compacted). Outside one it compacts the file as a transaction of its
own -- the same working-copy name, the same replace -- and
:func:`deferred_compaction` still collects such requests and runs each once
when its block exits. No pipeline call needs either any more; they remain for
callers that compact a file outside any write.
"""

from __future__ import annotations

import contextvars
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional, Union

from .atomic import compact_now, in_transaction

logger = logging.getLogger(__name__)

# The paths a deferral has collected so far, or ``None`` outside any deferral.
# A context variable rather than a module global so a deferral opened on one
# thread never captures another thread's writes.
_DEFERRED: contextvars.ContextVar[Optional[List[str]]] = contextvars.ContextVar(
    "ftmwpipeline_deferred_compaction", default=None
)


def compact_file(file_path: Union[Path, str]) -> None:
    """Rewrite *file_path* without its dead space, in place and atomically.

    Inside a transaction on the file (:func:`~.atomic.atomic_write`) this does
    nothing: the transaction's working copy was written compacted. Inside a
    :func:`deferred_compaction` block the path is recorded instead and
    compacted once when the block exits. Otherwise the file is copied compacted
    into its working copy (``.<basename>.ftmw-tmp.<hostname>.<pid>``), which
    then replaces it (:func:`~.atomic.compact_now`).

    Best-effort: a failure (a locked file on a platform that refuses to
    replace an open one, a full disk, a missing file) is logged as a warning
    and the original is left exactly as it was -- larger, never damaged. The
    working copy is removed on every path out.
    """
    if in_transaction(file_path):
        return
    path = str(file_path)
    deferred = _DEFERRED.get()
    if deferred is not None:
        if path not in deferred:
            deferred.append(path)
        return
    try:
        compact_now(path)
    except Exception as exc:  # never fail the write that preceded this
        logger.warning("Could not compact %s (left as written): %s", path, exc)


@contextmanager
def deferred_compaction() -> Iterator[None]:
    """Collect every :func:`compact_file` request made inside the block and
    run each distinct one once when the block exits (also on an exception:
    whatever was written before the failure is still worth packing).

    Nests: an inner block hands its paths to the outer one rather than
    compacting itself, so the outermost caller still decides when.
    """
    outer = _DEFERRED.get()
    if outer is not None:
        yield
        return
    collected: List[str] = []
    token = _DEFERRED.set(collected)
    try:
        yield
    finally:
        _DEFERRED.reset(token)
        for path in collected:
            if os.path.exists(path):
                compact_file(path)
