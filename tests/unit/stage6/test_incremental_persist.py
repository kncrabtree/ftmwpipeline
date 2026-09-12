"""The incremental fit persist (S4): what it writes must equal what a full
rewrite would have written.

``_finish_batch`` used to delete ``/stage5_fitting`` and rewrite every window
group on every curation write. It now rewrites only the windows the batch
mutated (:func:`~ftmwpipeline.io.fitting_serialization.update_spectrum_fit_windows_in_hdf5`),
which is a **pure performance and file-size** change: the bytes that describe
the fit must be the same either way.

That equality is the whole safety argument, because the incremental writer
takes the batch's own ``mutated_wids`` on trust for the *contents* of an
existing window group. If a verb ever mutated a window without recording it,
the file would keep that window's previous values and nothing else in the
suite would notice -- so every curation verb is driven here twice, once
incrementally and once forced through the full writer, and the two files are
compared attribute by attribute and dataset by dataset.

``creation_time`` is excluded from the comparison: ``stamp_stage_header``
stamps ``now()`` on every write, so it differs between any two writes,
incremental or not.

Uses ``stage5_multi_file`` (the 12-window build) from ``conftest.py``.
Writes only to pytest ``tmp_path``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import pytest

from ftmwpipeline._internal.stage6_impl import apply_curation_impl
from ftmwpipeline.io import fitting_serialization as fitser
from ftmwpipeline.io.fitting_serialization import (
    load_spectrum_fit_from_hdf5,
    save_spectrum_fit_to_hdf5,
)

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize(value: Any) -> Any:
    """HDF5 attr/dataset value -> something two files can be compared on."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


def _snapshot(path: Path) -> Dict[str, Any]:
    """Every attribute and dataset under ``/stage5_fitting``, flattened."""
    out: Dict[str, Any] = {}

    def visit(name: str, obj: Any) -> None:
        for key, value in obj.attrs.items():
            if key == "creation_time":
                continue
            out[f"attr:{name}:{key}"] = _normalize(value)
        if isinstance(obj, h5py.Dataset):
            out[f"data:{name}"] = _normalize(obj[()])

    with h5py.File(str(path), "r") as h5f:
        group = h5f["stage5_fitting"]
        visit("", group)
        group.visititems(visit)
    return out


@pytest.fixture
def force_full_rewrite(monkeypatch):
    """Put the pre-S4 write back: ignore the named windows, rewrite all."""

    def full(fit, h5_group, window_ids):
        save_spectrum_fit_to_hdf5(fit, h5_group)

    monkeypatch.setattr(fitser, "update_spectrum_fit_windows_in_hdf5", full)


def _a_fitted_peak(path: Path) -> Tuple[int, float]:
    with h5py.File(str(path), "r") as h5f:
        fit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    for wf in fit.window_fits:
        if wf.window_id is not None and wf.fitted_peaks:
            return int(wf.window_id), float(wf.fitted_peaks[0].frequency_mhz)
    pytest.skip("No fitted peaks in the built subset")


def _clear_add_freq(path: Path, wid: int) -> float:
    """An in-window frequency far from every fitted peak in *wid*.

    Never the window center: Stage 4 builds a window around the detection
    that seeded it, so the center is a fitted peak's birth position and an
    add there collides with its identity at zero distance.
    """
    from ftmwpipeline._internal.stage4_impl import load_windows_impl

    plan = load_windows_impl(str(path))["plan"]
    window = next(w for w in plan.windows if int(w.window_id) == wid)
    lo, hi = sorted(float(v) for v in window.freq_range)
    with h5py.File(str(path), "r") as h5f:
        fit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    wf = next(w for w in fit.window_fits if int(w.window_id or -1) == wid)
    peaks = [float(p.frequency_mhz) for p in wf.fitted_peaks]
    if not peaks:
        return 0.5 * (lo + hi)
    grid = np.linspace(lo, hi, 512)[1:-1]
    dist = np.min(np.abs(grid[:, None] - np.asarray(peaks)[None, :]), axis=1)
    return float(grid[int(np.argmax(dist))])


def _write_curation(tmp_path: Path, name: str, text: str) -> str:
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def _apply_both_ways(
    stage5_multi_file: Path,
    tmp_path: Path,
    monkeypatch,
    curation_text: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run one curation twice -- incrementally, and through the full writer
    -- on identical copies, and return the two ``/stage5_fitting`` snapshots.
    """
    incremental = tmp_path / "incremental.ftmw"
    full = tmp_path / "full.ftmw"
    shutil.copy(stage5_multi_file, incremental)
    shutil.copy(stage5_multi_file, full)
    curation = _write_curation(tmp_path, "c.csv", curation_text)

    apply_curation_impl(incremental, curation)

    with monkeypatch.context() as m:
        m.setattr(
            fitser,
            "update_spectrum_fit_windows_in_hdf5",
            lambda fit, group, wids: save_spectrum_fit_to_hdf5(fit, group),
        )
        apply_curation_impl(full, curation)

    return _snapshot(incremental), _snapshot(full)


def _equal(a: Any, b: Any) -> bool:
    """Value equality that treats NaN as equal to NaN.

    Most of the fit's float columns are NaN wherever a quantity was not
    estimated (``tau_error`` on a window with tau held, every ``*_error``
    on a peak whose covariance was not recovered), and ``nan != nan``
    would report every one of those as a difference between two writes
    that in fact wrote the same bytes.
    """
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (np.isnan(a) and np.isnan(b))
    return bool(a == b)


def _assert_same(incremental: Dict[str, Any], full: Dict[str, Any]) -> None:
    assert set(incremental) == set(full), (
        f"only incremental: {sorted(set(incremental) - set(full))[:8]}; "
        f"only full: {sorted(set(full) - set(incremental))[:8]}"
    )
    differing = [k for k in sorted(incremental) if not _equal(incremental[k], full[k])]
    assert not differing, (
        f"{len(differing)} key(s) differ between an incremental and a full "
        f"write, first few: "
        + "; ".join(f"{k}: {incremental[k]!r} vs {full[k]!r}" for k in differing[:4])
    )


# ---------------------------------------------------------------------------
# Equivalence, per verb
# ---------------------------------------------------------------------------


def test_remove_persists_what_a_full_rewrite_would(
    stage5_multi_file, tmp_path, monkeypatch
):
    wid, freq = _a_fitted_peak(stage5_multi_file)
    incremental, full = _apply_both_ways(
        stage5_multi_file, tmp_path, monkeypatch, f"remove,{wid},{freq},\n"
    )
    _assert_same(incremental, full)


def test_add_persists_what_a_full_rewrite_would(
    stage5_multi_file, tmp_path, monkeypatch
):
    wid, _freq = _a_fitted_peak(stage5_multi_file)
    add_freq = _clear_add_freq(stage5_multi_file, wid)
    incremental, full = _apply_both_ways(
        stage5_multi_file, tmp_path, monkeypatch, f"add,{wid},{add_freq},\n"
    )
    _assert_same(incremental, full)


def test_create_persists_what_a_full_rewrite_would(
    stage5_multi_file, tmp_path, monkeypatch
):
    """The case where a window group has to APPEAR.

    A create is the one verb that adds a window to the fit rather than
    editing one, so it is the verb that would break if the incremental
    writer only ever rewrote groups that already existed.
    """
    anchor = _free_anchor(stage5_multi_file)
    incremental, full = _apply_both_ways(
        stage5_multi_file,
        tmp_path,
        monkeypatch,
        f"create,new,{anchor:.6f},\n",
    )
    _assert_same(incremental, full)
    assert any(k.startswith("data:windows/") for k in incremental)


def _free_anchor(path: Path) -> float:
    """A frequency clear of every fitted window, for a create to anchor on."""
    with h5py.File(str(path), "r") as h5f:
        fit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    highs = [
        max(wf.window.freq_range)
        for wf in fit.window_fits
        if wf.window is not None and wf.fitted_peaks
    ]
    if not highs:
        pytest.skip("fixture has no fitted windows")
    return max(highs) + 5.0


def test_curation_writes_land_at_content_size_whichever_writer_ran(
    stage5_multi_file, tmp_path, monkeypatch
):
    """The other half of the point used to be measured as file growth: the
    incremental writer leaked less than a full delete-and-rewrite (919 kB per
    write on the 2638 build). Every curation write now ends by compacting the
    file (``_internal.compaction``), so growth no longer distinguishes the two
    writers -- both land at the size of the content. This pins that: the same
    two edits through either writer give the same file size (within the
    noise of a repack), and the second edit adds nothing the first did not
    (the one-time undo baseline is the only real growth).
    """
    incremental = tmp_path / "incr.ftmw"
    full = tmp_path / "full.ftmw"
    shutil.copy(stage5_multi_file, incremental)
    shutil.copy(stage5_multi_file, full)

    wid, freq = _a_fitted_peak(incremental)
    add_freq = _clear_add_freq(incremental, wid)
    edits = [f"add,{wid},{add_freq},\n", f"remove,{wid},{freq},\n"]

    def run(target: Path) -> List[int]:
        sizes = []
        for i, text in enumerate(edits):
            apply_curation_impl(
                target, _write_curation(tmp_path, f"{target.stem}{i}.csv", text)
            )
            sizes.append(target.stat().st_size)
        return sizes

    incremental_sizes = run(incremental)
    with monkeypatch.context() as m:
        m.setattr(
            fitser,
            "update_spectrum_fit_windows_in_hdf5",
            lambda fit, group, wids: save_spectrum_fit_to_hdf5(fit, group),
        )
        full_sizes = run(full)

    assert incremental_sizes[-1] == pytest.approx(full_sizes[-1], rel=0.02), (
        f"incremental writer left {incremental_sizes[-1] / 1e3:.0f} kB, a full "
        f"rewrite {full_sizes[-1] / 1e3:.0f} kB -- compaction should erase the "
        f"difference"
    )
    # After the first edit took the undo baseline, a further edit changes the
    # content by a few peaks' worth, not by a table set.
    assert incremental_sizes[1] <= 1.02 * incremental_sizes[0], incremental_sizes


def _datasets(group: h5py.Group) -> List[h5py.Dataset]:
    found: List[h5py.Dataset] = []

    def visit(_name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            found.append(obj)

    group.visititems(visit)
    return found


def test_a_window_dropped_from_the_fit_loses_its_row(
    stage5_multi_file, tmp_path, monkeypatch
):
    """The tables are reconciled against the fit, not against the caller.

    ``update_spectrum_fit_windows_in_hdf5`` is called directly here with an
    EMPTY named set, which is the under-reporting case the trust model has to
    survive: a window absent from the fit must lose its row and its peaks,
    whether or not the caller named it.
    """
    target = tmp_path / "dropped.ftmw"
    shutil.copy(stage5_multi_file, target)

    with h5py.File(str(target), "r+") as h5f:
        fit = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
        live = [wf for wf in fit.window_fits if wf.window_id is not None]
        keep, dropped = live[:-1], live[-1]
        trimmed = type(fit)(**{**fit.__dict__, "window_fits": keep})
        fitser.update_spectrum_fit_windows_in_hdf5(trimmed, h5f["stage5_fitting"], [])

    with h5py.File(str(target), "r") as h5f:
        ids = [int(v) for v in h5f["stage5_fitting/windows/window_id"][:]]
        n_peak_rows = h5f["stage5_fitting/peaks/frequency_mhz"].shape[0]

    assert int(dropped.window_id) not in ids
    assert len(ids) == len(keep)
    assert n_peak_rows == sum(len(wf.fitted_peaks) for wf in keep)
