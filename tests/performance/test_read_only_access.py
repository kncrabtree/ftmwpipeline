"""The read-only tap reads columns, not records.

``load_fit`` reconstructs the whole persisted ``SpectrumFit`` -- audit trails,
thaw and rescue histories, doublet alternatives, covariance -- and the cost is
h5py's per-item Python overhead paid thousands of times, not the bytes moved. A
consumer that only wants a few columns per fitted peak should not pay it.

Following the suite's convention, this guards the win *deterministically*: it
counts h5py item accesses (group/dataset lookups and attribute reads) rather
than asserting a wall-clock speedup, which flakes with load. Wall-clock is
printed for the record but never gates.

The invariants are stated as *relationships*, never as a snapshot of what the
code currently costs: an absolute ceiling would have to be re-raised every time
a column or a metadata section is added, and would then be measuring nothing.
What is asserted here is:

* a narrow read never triggers a full deserialization at all;
* each additional column costs one read per window -- the read scales with what
  was asked for, not with how much the record holds;
* the metadata read is independent of the window count: the same file with four
  times the windows costs exactly the same;
* both narrow reads stay a large multiple cheaper than their full loader.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Callable, List, Tuple

import h5py
import pytest

import ftmwpipeline.api as ftmw
import ftmwpipeline.io.fitting_serialization as fserial

from .conftest import BuiltPipeline

pytestmark = [pytest.mark.performance, pytest.mark.slow]

#: Columns a bounds/line-list consumer actually keeps (the issue-42 use case).
NARROW_FIT_COLUMNS = ["frequency_mhz", "decay_rate", "shape"]
NARROW_WINDOW_COLUMNS = ["window_id", "freq_min", "freq_max"]

#: The narrow fit read must stay at least this much cheaper than ``load_fit``
#: (measured ~8x here, ~10x on a 262-window production file).
MIN_FIT_ACCESS_RATIO = 4.0

#: Ditto for the window plan against ``load_windows``. The margin is smaller
#: because a planned window carries far less than a fitted one, so the full
#: loader has less to skip (measured ~4.5x here, ~5.4x on a 262-window file).
MIN_PLAN_ACCESS_RATIO = 3.0


def _counting_h5py(monkeypatch) -> Callable[[], int]:
    """Count h5py item accesses; returns a reader for the running total."""
    counts = {"n": 0}
    targets: List[Tuple[type, str]] = [
        (h5py.Group, "__getitem__"),
        (h5py.Dataset, "__getitem__"),
        (h5py.AttributeManager, "__getitem__"),
    ]
    for cls, name in targets:
        original = getattr(cls, name)

        def wrapper(self, key, _original=original):
            counts["n"] += 1
            return _original(self, key)

        monkeypatch.setattr(cls, name, wrapper)

    def reset_and_read() -> int:
        value = counts["n"]
        counts["n"] = 0
        return value

    return reset_and_read


def _measure(label: str, fn: Callable[[], object], total: Callable[[], int]) -> int:
    """Run *fn*, print its wall-clock, and return its h5py access count."""
    total()  # reset
    started = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - started
    accesses = total()
    print(f"\n  {label:<34} {accesses:>7} accesses  {elapsed * 1e3:8.1f} ms")
    return accesses


def _multiply_windows(path: Path, factor: int) -> None:
    """Grow a file's window count *factor*-fold by cloning its window groups.

    Duplicating the persisted groups is the cheapest way to get two files that
    differ only in how many windows they hold, which is what lets a test assert
    that a read's cost does or does not follow the window count. The clones are
    byte-identical apart from their ids, so nothing else about the file changes.
    """
    with h5py.File(path, "a") as h5f:
        for stage in ("stage4_windows", "stage5_fitting"):
            group = h5f[f"{stage}/windows"]
            names = list(group)
            next_id = max(int(name.split("_")[1]) for name in names) + 1
            for _ in range(factor - 1):
                for name in names:
                    clone = f"window_{next_id:04d}"
                    group.copy(name, clone)
                    group[clone].attrs["window_id"] = next_id
                    next_id += 1


def test_narrow_read_costs_a_fraction_of_the_full_loader(
    built_pipeline: BuiltPipeline, monkeypatch
) -> None:
    path = str(built_pipeline.path)
    n_windows = built_pipeline.fit.n_windows
    assert n_windows > 1, "fixture must fit more than one window to be meaningful"

    total = _counting_h5py(monkeypatch)
    full = _measure("load_fit (whole record)", lambda: ftmw.load_fit(path), total)
    narrow = _measure(
        f"read fit_peaks ({len(NARROW_FIT_COLUMNS)} columns)",
        lambda: ftmw.read_table(path, "fit_peaks", NARROW_FIT_COLUMNS),
        total,
    )

    print(
        f"  -> {narrow / n_windows:.1f} accesses/window over {n_windows} windows; "
        f"{full / max(narrow, 1):.1f}x cheaper than the full loader"
    )
    assert full >= MIN_FIT_ACCESS_RATIO * narrow, (
        f"narrow read ({narrow} accesses) is no longer meaningfully cheaper than "
        f"load_fit ({full}); expected at least {MIN_FIT_ACCESS_RATIO}x"
    )


def test_read_cost_is_linear_in_the_columns_requested(
    built_pipeline: BuiltPipeline, monkeypatch
) -> None:
    """The read scales with what was asked for, not with the record's size.

    Measured as linearity rather than as a cost: the marginal price of a column
    must be the same whether it is the second one or the fourth. That is the
    actual claim -- ``cost = fixed + per_column x columns`` -- and it holds no
    matter how many h5py calls one column happens to take, so nothing here has
    to be revised when a column is added or the counting changes. A regression
    that grabbed the whole peaks subgroup, or parsed the window's JSON
    attributes, would put a constant term where the linear one belongs and the
    two marginal costs would diverge.
    """
    path = str(built_pipeline.path)
    n_windows = built_pipeline.fit.n_windows
    assert n_windows > 1, "fixture must fit more than one window to be meaningful"

    # frequency_mhz is the anchor the reader always reads, so a selection of it
    # alone is the fixed per-window overhead; the rest are marginal columns.
    selections = {
        1: ["frequency_mhz"],
        2: ["frequency_mhz", "amplitude"],
        4: ["frequency_mhz", "amplitude", "phase", "snr"],
    }
    total = _counting_h5py(monkeypatch)
    cost = {
        k: _measure(
            f"read fit_peaks ({k} column{'s' if k > 1 else ''})",
            lambda cols=columns: ftmw.read_table(path, "fit_peaks", cols),
            total,
        )
        for k, columns in selections.items()
    }

    first_step = cost[2] - cost[1]
    later_step = (cost[4] - cost[2]) / 2
    print(
        f"  -> marginal cost/column: {first_step} accesses (2nd) vs "
        f"{later_step:.1f} (3rd-4th), over {n_windows} windows"
    )

    assert first_step > 0, (
        "adding a column cost nothing; either selection is being ignored or the "
        "counter is not wired up"
    )
    assert later_step == pytest.approx(first_step, rel=0.05), (
        f"the marginal cost of a column is not constant: {first_step} accesses "
        f"for the second, {later_step:.1f} for each of the third and fourth. The "
        "read is picking up work that does not belong to the requested columns"
    )


def test_narrow_read_never_deserializes_the_record(
    built_pipeline: BuiltPipeline, monkeypatch
) -> None:
    """``_load_peak_columns`` fires once per window per *full* deserialization.

    The read path must never reach it -- that is the whole point of the surface.
    """
    calls = {"n": 0}
    original = fserial._load_peak_columns

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(fserial, "_load_peak_columns", counting)

    path = str(built_pipeline.path)
    ftmw.read_table(path, "fit_peaks", NARROW_FIT_COLUMNS)
    ftmw.read_table(path, "fit_windows")
    ftmw.read_metadata(path)
    assert calls["n"] == 0, (
        f"the read surface deserialized the fit record ({calls['n']} peak-column "
        "loads); it must read columns directly"
    )

    # Sanity: the counter does fire for the full loader, so a zero above means
    # "never called", not "counter not wired up".
    ftmw.load_fit(path)
    assert calls["n"] > 0


def test_window_bounds_read_costs_a_fraction_of_the_full_plan_loader(
    built_pipeline: BuiltPipeline, monkeypatch
) -> None:
    path = str(built_pipeline.path)
    n_windows = len(ftmw.read_table(path, "windows", ["window_id"])["window_id"])
    assert n_windows > 1

    total = _counting_h5py(monkeypatch)
    full = _measure("load_windows (whole plan)", lambda: ftmw.load_windows(path), total)
    narrow = _measure(
        "read windows (bounds only)",
        lambda: ftmw.read_table(path, "windows", NARROW_WINDOW_COLUMNS),
        total,
    )

    print(
        f"  -> {narrow / n_windows:.1f} accesses/window over {n_windows} windows; "
        f"{full / max(narrow, 1):.1f}x cheaper than the full loader"
    )
    assert full >= MIN_PLAN_ACCESS_RATIO * narrow, (
        f"bounds-only read ({narrow} accesses) is no longer meaningfully cheaper "
        f"than load_windows ({full}); expected at least {MIN_PLAN_ACCESS_RATIO}x"
    )


def test_metadata_read_does_not_scale_with_the_window_count(
    built_pipeline: BuiltPipeline, tmp_path, monkeypatch
) -> None:
    """``read_metadata`` reads group attributes only -- never a window walk.

    Asserted against a clone of the same file with four times the windows: the
    cost must be *identical*, not merely small. That is the property worth
    protecting (a regression that reached into the fit for ``acquisition_us``
    would make it O(N)), and unlike an access ceiling it needs no revision when
    a new scalar section is added.
    """
    small = built_pipeline.path
    large = tmp_path / "more_windows.ftmw"
    shutil.copy(small, large)
    _multiply_windows(large, factor=4)

    total = _counting_h5py(monkeypatch)
    small_meta = _measure(
        "read_metadata (1x windows)", lambda: ftmw.read_metadata(str(small)), total
    )
    large_meta = _measure(
        "read_metadata (4x windows)", lambda: ftmw.read_metadata(str(large)), total
    )
    small_table = _measure(
        "read fit_windows (1x windows)",
        lambda: ftmw.read_table(str(small), "fit_windows", ["window_id"]),
        total,
    )
    large_table = _measure(
        "read fit_windows (4x windows)",
        lambda: ftmw.read_table(str(large), "fit_windows", ["window_id"]),
        total,
    )

    # The counter must be sensitive to the window count, or "unchanged" below
    # would prove nothing.
    assert large_table > small_table, (
        "cloning the window groups did not make a per-window read more "
        "expensive; the differential this test rests on is not working"
    )
    assert large_meta == small_meta, (
        f"read_metadata cost {small_meta} accesses on the original file and "
        f"{large_meta} on a clone with 4x the windows; it must not walk them"
    )
    # It still has to produce the scalar the whole exercise is about.
    assert ftmw.read_metadata(str(small))["stage5.acquisition_us"] is not None


def test_column_selection_actually_reduces_the_read(
    built_pipeline: BuiltPipeline, monkeypatch
) -> None:
    """Asking for fewer columns must cost less -- selection is not cosmetic."""
    path = str(built_pipeline.path)
    total = _counting_h5py(monkeypatch)
    everything = _measure(
        "read fit_peaks (all columns)",
        lambda: ftmw.read_table(path, "fit_peaks"),
        total,
    )
    narrow = _measure(
        f"read fit_peaks ({len(NARROW_FIT_COLUMNS)} columns)",
        lambda: ftmw.read_table(path, "fit_peaks", NARROW_FIT_COLUMNS),
        total,
    )
    assert narrow < everything, (
        f"selecting {len(NARROW_FIT_COLUMNS)} columns cost {narrow} accesses vs "
        f"{everything} for all of them -- selection is not narrowing the read"
    )


def test_read_and_loader_agree_on_the_measured_file(
    built_pipeline: BuiltPipeline,
) -> None:
    """The cheap path is only a win if it is also correct on this fixture."""
    path = str(built_pipeline.path)
    columns = ftmw.read_table(path, "fit_peaks", ["frequency_mhz"])
    expected = [p.frequency_mhz for p in built_pipeline.fit.fitted_peaks]
    assert list(columns["frequency_mhz"]) == expected
