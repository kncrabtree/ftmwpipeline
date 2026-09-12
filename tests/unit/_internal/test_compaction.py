"""Unit tests for ``_internal.compaction``: the repack itself, its deferral,
and its refusal to fail the write that preceded it.

These use a bare HDF5 file with the two kinds of churn HDF5 never reclaims on
its own (a rewritten attribute, a deleted-and-recreated vlen dataset); the
pipeline-level guarantees -- a curated ``.ftmw`` does not grow -- live in
``tests/unit/stage6/test_compaction.py``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import pytest

from ftmwpipeline._internal.compaction import compact_file, deferred_compaction

pytestmark = [pytest.mark.unit]


def _dump(path: Path) -> Dict[str, Any]:
    """Every object's attrs and every dataset's values, keyed by HDF5 path."""
    out: Dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        out["/attrs"] = {k: _norm(v) for k, v in f.attrs.items()}

        def visit(name: str, obj: Any) -> None:
            entry: Dict[str, Any] = {
                "attrs": {k: _norm(v) for k, v in obj.attrs.items()}
            }
            if isinstance(obj, h5py.Dataset):
                entry["dtype"] = str(obj.dtype)
                entry["shape"] = obj.shape
                entry["maxshape"] = obj.maxshape
                entry["chunks"] = obj.chunks
                entry["data"] = _norm(obj[()])
            out[name] = entry

        f.visititems(visit)
    return out


def _norm(v: Any) -> Any:
    """A comparable Python value: bytes decoded, arrays listed, NaN made
    equal to itself (a fit table carries NaN where a column does not apply)."""
    if isinstance(v, bytes):
        return v.decode("utf-8")
    if isinstance(v, np.ndarray):
        if v.dtype.kind == "O":
            return [
                x.decode("utf-8") if isinstance(x, bytes) else x for x in v.tolist()
            ]
        return [_norm(x) for x in v.tolist()]
    if isinstance(v, np.generic):
        return _norm(v.item())
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, float) and v != v:
        return "NaN"
    return v


def _churn(path: Path, rounds: int) -> None:
    """One open/close per round, like the pipeline: rewrite a big attribute
    and delete/recreate a vlen-string dataset."""
    for i in range(rounds):
        with h5py.File(path, "a") as f:
            f.attrs["blob"] = "x" * (100_000 + i)
            if "g" in f:
                del f["g"]
            g = f.create_group("g")
            g.create_dataset(
                "s",
                data=np.array([f"row{j}-{i}" for j in range(5000)], dtype=object),
                dtype=h5py.string_dtype(),
                maxshape=(None,),
                chunks=(5000,),
            )
            g.attrs["round"] = i


@pytest.fixture
def churned(tmp_path: Path) -> Path:
    p = tmp_path / "churn.h5"
    with h5py.File(p, "w") as f:
        f.attrs["format"] = "1.0"
        f.create_dataset("fid", data=np.arange(100_000, dtype="f8"), chunks=(1000,))
    _churn(p, 10)
    return p


def test_compact_reclaims_attribute_and_vlen_churn(churned: Path) -> None:
    before = os.path.getsize(churned)
    content = _dump(churned)
    compact_file(churned)
    after = os.path.getsize(churned)
    # Ten rounds of ~100 kB attribute plus a vlen table each: well over a
    # megabyte of dead space against under a megabyte of live content.
    assert after < before / 2
    assert _dump(churned) == content


def test_compact_is_idempotent(churned: Path) -> None:
    compact_file(churned)
    once = os.path.getsize(churned)
    compact_file(churned)
    assert os.path.getsize(churned) == once


def test_compact_preserves_mode_and_replaces_in_place(churned: Path) -> None:
    os.chmod(churned, 0o640)
    inode = os.stat(churned).st_ino
    compact_file(churned)
    assert oct(os.stat(churned).st_mode & 0o777) == oct(0o640)
    # A fresh inode (write-to-temp, then replace), and no temp file left over.
    assert os.stat(churned).st_ino != inode
    assert [p.name for p in churned.parent.iterdir()] == [churned.name]


def test_deferred_compaction_runs_once_on_exit(churned: Path, monkeypatch) -> None:
    import ftmwpipeline._internal.compaction as mod

    calls = []
    real = mod.compact_file

    def counting(path):
        calls.append(str(path))
        real(path)

    # Count only the *actual* repacks: patch what the deferral calls on exit.
    monkeypatch.setattr(mod, "compact_file", counting)
    big = os.path.getsize(churned)
    with deferred_compaction():
        mod.compact_file(churned)
        _churn(churned, 2)
        mod.compact_file(churned)
        mod.compact_file(str(churned))
        # Nothing has been repacked yet -- the file only grew.
        assert os.path.getsize(churned) > big
    assert os.path.getsize(churned) < big
    # Inside the block every call was a deferral; on exit exactly one repack.
    assert len(calls) == 4  # three deferred + one real on exit
    with h5py.File(churned, "r") as f:
        assert f["g"].attrs["round"] == 1


def test_deferred_compaction_nests_to_the_outermost_block(churned: Path) -> None:
    big = os.path.getsize(churned)
    with deferred_compaction():
        with deferred_compaction():
            compact_file(churned)
        # The inner block exited without repacking: the outer one owns it.
        assert os.path.getsize(churned) == big
    assert os.path.getsize(churned) < big


def test_deferred_compaction_still_compacts_after_an_exception(churned: Path) -> None:
    big = os.path.getsize(churned)
    with pytest.raises(RuntimeError):
        with deferred_compaction():
            compact_file(churned)
            raise RuntimeError("the write after this one failed")
    assert os.path.getsize(churned) < big


def test_compact_failure_warns_and_leaves_the_file(churned: Path, caplog) -> None:
    content = _dump(churned)
    missing = churned.parent / "not-there.h5"
    with caplog.at_level(logging.WARNING, logger="ftmwpipeline"):
        compact_file(missing)  # no raise
    assert any("Could not compact" in r.getMessage() for r in caplog.records)
    # And the original is untouched by a failure on another path.
    assert _dump(churned) == content
    assert not list(churned.parent.glob(".*.tmp"))
