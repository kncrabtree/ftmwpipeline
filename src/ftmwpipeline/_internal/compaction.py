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
this size. :func:`compact_file` does that, atomically, and is called
unconditionally at the end of every stage run and every curation write, so a
``.ftmw`` on disk is always as small as its content. There is deliberately no
user-facing verb for it: a file that is never bloated needs no command to
un-bloat it. Callers that string several writes together
(:func:`~ftmwpipeline._internal.run_impl.run_pipeline_impl`, ``review undo``'s
restore-then-replay) wrap the sequence in :func:`deferred_compaction` so the
file is compacted once at the end instead of once per write.

Where the call sits (the complete list -- keep it current):

- :func:`~ftmwpipeline._internal.stage2_impl._update_stage_completion`, the
  completion stamp every stage impl from noise through review writes last,
  so one call covers Stages 2, 2b, 3, 4, 5, 6 and the timebase stage;
- the Stage 1 FT settings writer (:mod:`~ftmwpipeline._internal.stage1_impl`),
  which stamps its own completion rather than going through the shared
  helper;
- :func:`~ftmwpipeline._internal.stage6_impl._finish_batch`, the engine's one
  writer of ``/stage5_fitting``, and the two review-only writers of
  ``/stage6_review`` (:func:`~ftmwpipeline._internal.stage6_impl._record_decision`
  and :func:`~ftmwpipeline._internal.stage6_impl._record_bare_accept`);
- :func:`~ftmwpipeline._internal.stage6_impl.review_undo_impl` and the
  log-prefix path of :func:`~ftmwpipeline._internal.stage6_impl.apply_curation_impl`,
  whose restore-then-replay is deferred as one unit.

A fresh import writes a brand-new file with nothing to reclaim, so the
import stage does not call this.
"""

from __future__ import annotations

import contextvars
import logging
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional, Union

import h5py

logger = logging.getLogger(__name__)

# The paths a deferral has collected so far, or ``None`` outside any deferral.
# A context variable rather than a module global so a deferral opened on one
# thread never captures another thread's writes.
_DEFERRED: contextvars.ContextVar[Optional[List[str]]] = contextvars.ContextVar(
    "ftmwpipeline_deferred_compaction", default=None
)


def compact_file(file_path: Union[Path, str]) -> None:
    """Rewrite *file_path* without its dead space, in place and atomically.

    Copies the root attributes and every root-level object (with everything
    beneath it -- ``h5py``'s object copy carries attributes, chunking,
    ``maxshape`` and dtypes across intact) into a temporary file beside the
    original, then replaces the original with it. Readers that open the file
    afterwards see identical content; the bytes are simply packed.

    Inside a :func:`deferred_compaction` block the path is recorded instead
    and compacted once when the block exits.

    Best-effort: the write that preceded this call has already landed, so a
    failure here (a locked file on a platform that refuses to replace an open
    one, a full disk) is logged as a warning and the original is left exactly
    as it was -- larger, never damaged. The temporary file is removed on every
    path out.
    """
    path = str(file_path)
    deferred = _DEFERRED.get()
    if deferred is not None:
        if path not in deferred:
            deferred.append(path)
        return

    directory, name = os.path.split(os.path.abspath(path))
    tmp = os.path.join(directory, f".{name}.compact-{os.getpid()}.tmp")
    try:
        with h5py.File(path, "r") as src, h5py.File(tmp, "w") as dst:
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            for member in src:
                src.copy(member, dst, name=member)
        shutil.copymode(path, tmp)
        os.replace(tmp, path)
    except Exception as exc:  # never fail the write that preceded this
        logger.warning("Could not compact %s (left as written): %s", path, exc)
        try:
            os.remove(tmp)
        except OSError:
            pass


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
