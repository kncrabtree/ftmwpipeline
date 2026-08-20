"""Stage 5 cross-window parallelism is correctness-neutral.

The Stage 5 fit levelizes the window dependency DAG and forks one worker per
window per level (see the ``jobs`` / ``FTMW_MAX_WORKERS`` knob). That pool is a
pure speedup: a parallel fit must produce a fit that is byte-identical to the
sequential one, window for window and peak for peak. This is the property the
project verifies when it changes the fit, so it is the property guarded here --
the test is deliberately *not* a wall-clock assertion, which would flake with
load and core count.

Marked ``performance`` (and ``slow``): it builds a real fit twice and is opt-in,
not part of the default suite. Run it with ``-m performance``.
"""

import shutil
from pathlib import Path

import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline.core.data_structures import SpectrumFit

pytestmark = [pytest.mark.performance, pytest.mark.slow]

EXAMPLE_DATA = Path("examples/blackchirp_data/2638")
ACTIVE_BAND = (26500, 40000)


def _assert_fits_identical(a: SpectrumFit, b: SpectrumFit) -> None:
    """Two fits of the same file carry byte-identical scientific content."""
    assert a.n_windows == b.n_windows
    assert a.n_fitted_peaks == b.n_fitted_peaks
    assert a.final_plan_revision == b.final_plan_revision

    by_a = {w.window_id: w for w in a.window_fits}
    by_b = {w.window_id: w for w in b.window_fits}
    assert set(by_a) == set(by_b)
    for wid in by_a:
        wa, wb = by_a[wid], by_b[wid]
        assert len(wa.fitted_peaks) == len(wb.fitted_peaks), (
            f"window {wid} peak count differs: "
            f"{len(wa.fitted_peaks)} vs {len(wb.fitted_peaks)}"
        )
        pa = sorted(wa.fitted_peaks, key=lambda p: p.frequency_mhz)
        pb = sorted(wb.fitted_peaks, key=lambda p: p.frequency_mhz)
        for x, y in zip(pa, pb):
            # The solver path is deterministic from the persisted inputs, so the
            # worker count must not perturb any fitted parameter at all.
            assert x.frequency_mhz == y.frequency_mhz
            assert x.amplitude == y.amplitude
            assert x.window_id == y.window_id
            assert x.detection_index == y.detection_index


@pytest.fixture(scope="module")
def stage4_multiwindow(tmp_path_factory):
    """Build a Stage-4 fixture with several dependency-free windows.

    Drives 2638 through window assignment, then trims the plan to the
    dependency-free prefix (the low-band-edge windows whose dependency edges
    stay inside the kept set). That leaves enough independent windows for the
    fork pool to run several concurrently in one DAG level -- which is exactly
    what the parallel path must get right -- while keeping the fit cheap enough
    to run twice.
    """
    if not EXAMPLE_DATA.exists():
        pytest.skip("Experiment 2638 data not available")

    from ftmwpipeline._internal.stage4_impl import (
        load_windows_impl,
        save_window_plan_impl,
    )

    tmp = tmp_path_factory.mktemp("perf_stage4")
    fp = tmp / "perf_2638_stage4.ftmw"

    ftmw.import_data(fp, source=str(EXAMPLE_DATA))
    ftmw.compute_ft(fp, trim=ACTIVE_BAND)
    ftmw.estimate_noise(fp)
    ftmw.calibrate_tau(fp)
    ftmw.detect_peaks(fp)
    ftmw.assign_windows(fp)

    plan = load_windows_impl(str(fp))["plan"]
    keep: list = []
    for wid in plan.topological_order:
        deps = [(a, b) for (a, b) in plan.dependency_edges if a == wid or b == wid]
        if all(a in keep or a == wid for (a, _) in deps) and all(
            b in keep or b == wid for (_, b) in deps
        ):
            keep.append(wid)
        if len(keep) >= 12:
            break
    keep_set = set(keep)
    plan.windows = [w for w in plan.windows if w.window_id in keep_set]
    plan.topological_order = [w for w in plan.topological_order if w in keep_set]
    plan.dependency_edges = [
        (a, b) for (a, b) in plan.dependency_edges if a in keep_set and b in keep_set
    ]
    save_window_plan_impl(str(fp), plan)
    return fp


def test_parallel_fit_matches_sequential(stage4_multiwindow, tmp_path):
    """fit_peaks(jobs=4) == fit_peaks(jobs=1), window for window."""
    seq_file = tmp_path / "seq.ftmw"
    par_file = tmp_path / "par.ftmw"
    shutil.copy(stage4_multiwindow, seq_file)
    shutil.copy(stage4_multiwindow, par_file)

    sequential = ftmw.fit_peaks(seq_file, jobs=1)
    parallel = ftmw.fit_peaks(par_file, jobs=4)

    # Guard against a fixture that exercises nothing: the comparison is only
    # meaningful if real lines were fitted across more than one window.
    assert sequential.n_fitted_peaks > 0
    assert sequential.n_windows > 1

    _assert_fits_identical(sequential, parallel)
