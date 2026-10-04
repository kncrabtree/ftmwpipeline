"""Shared helpers for the Wave 5.1 events / cancellation tests.

These encode the promises of ``dev-docs/CONTRACT_STRATEGY.md`` §Events and
cancellation as small checks the test modules compose: a recording callback, a
cancel token, a structural digest of a ``.ftmw`` file (so "the file is exactly
as it was" is one equality), and the ordering rule of a stage's events.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import threading
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Set, Type

import h5py
import numpy as np

from ftmwpipeline import (
    Invalidated,
    StageFinished,
    StageStarted,
)


class Recorder:
    """An ``events`` callback that keeps every event, the thread and process
    each was delivered on, and optionally acts on one."""

    def __init__(self, on_event: Optional[Callable[[Any], None]] = None) -> None:
        import os

        self._getpid = os.getpid
        self.events: List[Any] = []
        self.threads: List[int] = []
        self.pids: List[int] = []
        self.on_event = on_event

    def __call__(self, event: Any) -> None:
        self.events.append(event)
        self.threads.append(threading.get_ident())
        self.pids.append(self._getpid())
        if self.on_event is not None:
            self.on_event(event)

    def of(self, kind: Type[Any]) -> List[Any]:
        return [e for e in self.events if isinstance(e, kind)]


class Token:
    """A minimal ``CancelToken`` (anything with ``is_set()`` qualifies) that
    remembers when it was set."""

    def __init__(self) -> None:
        self._flag = False
        self.set_at: Optional[float] = None

    def set(self) -> None:
        if not self._flag:
            self._flag = True
            self.set_at = time.monotonic()

    def is_set(self) -> bool:
        return self._flag


def cancel_on_nth(kind: Type[Any], n: int, token: Token) -> Callable[[Any], None]:
    """An event action: set *token* on the *n*-th (1-based) event of *kind*."""
    seen = {"n": 0}

    def act(event: Any) -> None:
        if isinstance(event, kind):
            seen["n"] += 1
            if seen["n"] == n:
                token.set()

    return act


def _value_bytes(value: Any) -> bytes:
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "O":
            return repr(value.tolist()).encode()
        return value.tobytes() + str(value.dtype).encode() + str(value.shape).encode()
    return repr(value).encode()


def content_digest(path: Path) -> str:
    """A digest of every group, dataset (shape, dtype, contents) and attribute.

    "Exactly as it was" is a statement about the file's content, not about its
    free space or byte layout, which a rewrite that restored every value may
    change.
    """
    h = hashlib.sha256()

    def add_attrs(obj: h5py.HLObject) -> None:
        for key in sorted(obj.attrs.keys()):
            h.update(b"@" + key.encode())
            h.update(_value_bytes(obj.attrs[key]))

    def visit(name: str, obj: h5py.HLObject) -> None:
        h.update(b"/" + name.encode())
        add_attrs(obj)
        if isinstance(obj, h5py.Dataset):
            h.update(str(obj.shape).encode() + str(obj.dtype).encode())
            h.update(_value_bytes(obj[()]))

    with h5py.File(path, "r") as f:
        add_attrs(f)
        f.visititems(visit)
    return h.hexdigest()


def listing(path: Path) -> List[str]:
    """Every group / dataset name in the file, sorted."""
    names: List[str] = []
    with h5py.File(path, "r") as f:
        f.visit(names.append)
    return sorted(names)


def completed_stages(path: Path) -> List[str]:
    """The storage keys recorded complete in the file."""
    with h5py.File(path, "r") as f:
        return sorted(json.loads(f["pipeline_stages"].attrs["completed_stages"]))


def assert_stage_order(events: List[Any]) -> None:
    """The ordering rule of one stage: ``StageStarted`` first, then the stage's
    other events, then ``Invalidated`` (if any), then ``StageFinished`` last."""
    assert events, "no events were delivered"
    assert isinstance(events[0], StageStarted), type(events[0]).__name__
    assert isinstance(events[-1], StageFinished), type(events[-1]).__name__
    assert sum(isinstance(e, StageStarted) for e in events) == 1
    assert sum(isinstance(e, StageFinished) for e in events) == 1
    inv = [i for i, e in enumerate(events) if isinstance(e, Invalidated)]
    assert len(inv) <= 1, "Invalidated is emitted once per call"
    last_other = max(
        (
            i
            for i, e in enumerate(events)
            if not isinstance(e, (StageStarted, Invalidated, StageFinished))
        ),
        default=0,
    )
    for i in inv:
        assert last_other < i < len(events) - 1, "Invalidated sits just before the end"


def wait_no_new_children(before: Set[Any], timeout: float = 10.0) -> Set[Any]:
    """The child processes started since *before* that are still alive after
    up to *timeout* seconds (empty once they are reaped)."""
    deadline = time.monotonic() + timeout
    while True:
        extra = set(multiprocessing.active_children()) - before
        if not extra or time.monotonic() > deadline:
            return extra
        time.sleep(0.1)


def wire_sequence(events: List[Any]) -> List[tuple]:
    """The deterministic part of an event sequence, for comparing interfaces:
    schema, operation and stage, plus a window's identity and position, and a
    stage's summary keys. Timings and values that depend on the file's path are
    left out."""
    from ftmwpipeline import to_jsonable

    seq: List[tuple] = []
    for e in events:
        d = e if isinstance(e, dict) else to_jsonable(e)
        row: tuple = (d["schema"], d["operation"], d["stage"])
        if d["schema"] == "ftmw/window_progress@1":
            row += (
                d["phase"],
                d["round"],
                d["index"],
                d["total"],
                d["window_id"],
                d["dropped"],
            )
        elif d["schema"] == "ftmw/stage_finished@1":
            row += (tuple(sorted(d["summary"])),)
        elif d["schema"] == "ftmw/invalidated@1":
            row += (tuple(d["stages"]),)
        seq.append(row)
    return seq
