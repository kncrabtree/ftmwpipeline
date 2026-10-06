"""
The Stage 6 replay engine's key store, ``/stage6_engine``.

The incremental write path (:func:`~ftmwpipeline._internal.stage6_impl._curate`)
keys every window's curated fit by what it was computed from, so a write can
keep the fits whose inputs it did not change. The keys live here, in a
top-level group of their own rather than under ``/stage6_review``: that group
is deleted and recreated on every write, and HDF5 does not reclaim deleted
space, so a store under it would be rewritten whole and grow the file each
time. This group is never deleted by a write; its key table is resized and
overwritten in place, as the fit tables are.

HDF5 layout::

    /stage6_engine
        .attrs:
            e_key         (str; the digest of the shared analysis context the
                           keys were computed under; absent when the group
                           holds no keys)
            create_chain  (JSON list; one entry per create row of the log, in
                           log order: ``digest`` (the chain digest through
                           that create), ``window`` (the serialized
                           ``FitWindow`` it installed), ``mode`` and
                           ``depends_on``)
        keys/             (one row per window the curated fit holds)
            window_id     (i8)
            kd            (str; the window's direct-phase key)
            kf            (str; its final-fit key)
            reached       (i1; whether a dirty strict ancestor reaches it)
            dirty         (i1; whether a decision edits it)

A file without the group, or with an empty key table, has no keys: the next
write that refits recomputes every window it curates. ``fit run`` drops the
group with the undo baseline, and an upstream invalidation drops it with
``/stage6_review``. Not part of any public interface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import h5py
import numpy as np

from .fitting_serialization import _write_table

__all__ = [
    "STAGE6_ENGINE_GROUP",
    "EngineWindowKey",
    "Stage6EngineState",
    "load_stage6_engine_state",
    "save_stage6_engine_state",
]

#: The engine's key store.
STAGE6_ENGINE_GROUP = "stage6_engine"


@dataclass(frozen=True)
class EngineWindowKey:
    """One window's keys: ``kd`` covers its direct phase, ``kf`` its final
    fit; ``reached`` and ``dirty`` are the trigger-rule flags they were
    computed with."""

    kd: str
    kf: str
    reached: bool
    dirty: bool


@dataclass
class Stage6EngineState:
    """What ``/stage6_engine`` holds: the context digest ``e_key`` the keys
    were computed under, the create chain, and each window's keys."""

    e_key: str
    create_chain: List[Dict[str, Any]] = field(default_factory=list)
    keys: Dict[int, EngineWindowKey] = field(default_factory=dict)


def save_stage6_engine_state(
    h5f: h5py.File, state: Optional[Stage6EngineState]
) -> None:
    """Write *state* to ``/stage6_engine``, resizing its datasets in place.
    ``None`` empties the store (no keys, no chain)."""
    group = h5f.require_group(STAGE6_ENGINE_GROUP)
    if state is None:
        for name in ("e_key", "create_chain"):
            if name in group.attrs:
                del group.attrs[name]
        keys: Dict[int, EngineWindowKey] = {}
    else:
        group.attrs["e_key"] = state.e_key
        group.attrs["create_chain"] = json.dumps(state.create_chain, default=str)
        keys = state.keys
    wids = sorted(keys)
    kd = np.empty(len(wids), dtype=object)
    kf = np.empty(len(wids), dtype=object)
    for i, w in enumerate(wids):
        kd[i] = keys[w].kd
        kf[i] = keys[w].kf
    _write_table(
        group.require_group("keys"),
        {
            "window_id": np.asarray(wids, dtype="i8"),
            "kd": kd,
            "kf": kf,
            "reached": np.asarray([keys[w].reached for w in wids], dtype="i1"),
            "dirty": np.asarray([keys[w].dirty for w in wids], dtype="i1"),
        },
    )


def load_stage6_engine_state(h5f: h5py.File) -> Optional[Stage6EngineState]:
    """The key store of *h5f*, or ``None`` when it holds no keys."""
    group = h5f.get(STAGE6_ENGINE_GROUP)
    if not isinstance(group, h5py.Group) or "e_key" not in group.attrs:
        return None
    raw_key = group.attrs["e_key"]
    e_key = raw_key.decode("utf-8") if isinstance(raw_key, bytes) else str(raw_key)
    raw_chain = group.attrs.get("create_chain", "[]")
    if isinstance(raw_chain, bytes):
        raw_chain = raw_chain.decode("utf-8")
    keys: Dict[int, EngineWindowKey] = {}
    table = group.get("keys")
    if isinstance(table, h5py.Group) and "window_id" in table:
        wids = table["window_id"][...]
        kd = table["kd"].asstr()[...]
        kf = table["kf"].asstr()[...]
        reached = table["reached"][...]
        dirty = table["dirty"][...]
        for i, w in enumerate(wids):
            keys[int(w)] = EngineWindowKey(
                kd=str(kd[i]),
                kf=str(kf[i]),
                reached=bool(reached[i]),
                dirty=bool(dirty[i]),
            )
    return Stage6EngineState(
        e_key=e_key, create_chain=list(json.loads(str(raw_chain))), keys=keys
    )
