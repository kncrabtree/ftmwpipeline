"""Shared fixtures for the performance regression-guard suite.

The suite is opt-in (``-m performance``) and deliberately asserts *deterministic*
quantities — on-disk file size and operation counts that encode algorithmic
complexity — rather than wall-clock time, which flakes with load and core count.
Wall-clock is recorded for the record (``record_property`` / stdout) but never
gates. This mirrors ``test_parallel_fit.py``, which guards the parallel fit by
byte-identity rather than by a speedup assertion.

Building a real fit is expensive, so the full-pipeline fixture is session-scoped
and built once; the storage figures are captured *during* that single build.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline.core.data_structures import SpectrumFit

EXAMPLE_DATA = Path("examples/blackchirp_data/2638")
ACTIVE_BAND = (26500, 40000)


def _raw_source_bytes() -> int:
    """Total bytes of the raw Blackchirp source directory for 2638."""
    return sum(f.stat().st_size for f in EXAMPLE_DATA.rglob("*") if f.is_file())


@dataclass
class BuiltPipeline:
    """A fully built ``.ftmw`` plus the storage figures captured while building."""

    path: Path
    size_after_import: int
    size_after_stage2: int
    raw_source_bytes: int
    fit: SpectrumFit


@pytest.fixture(scope="session")
def example_data() -> Path:
    if not EXAMPLE_DATA.exists():
        pytest.skip("Experiment 2638 data not available")
    return EXAMPLE_DATA


@pytest.fixture(scope="session")
def imported_file(example_data: Path, tmp_path_factory) -> Path:
    """A cheap ``.ftmw`` that has only been imported (for the memory probe)."""
    tmp = tmp_path_factory.mktemp("perf_import")
    fp = tmp / "perf_2638_import.ftmw"
    ftmw.import_data(fp, source=str(example_data))
    return fp


@pytest.fixture(scope="session")
def built_pipeline(example_data: Path, tmp_path_factory) -> BuiltPipeline:
    """Drive 2638 through ``review_run`` on a trimmed, dependency-free plan.

    The window plan is trimmed to its dependency-free prefix (the same trick as
    ``test_parallel_fit``) so the fit stays cheap while leaving several real
    windows for the report to fold in. Storage figures are read off the file at
    the import and Stage-2 boundaries during this one build.
    """
    from ftmwpipeline._internal.stage4_impl import (
        load_windows_impl,
        save_window_plan_impl,
    )

    tmp = tmp_path_factory.mktemp("perf_full")
    fp = tmp / "perf_2638_full.ftmw"

    ftmw.import_data(fp, source=str(example_data))
    size_after_import = fp.stat().st_size

    ftmw.compute_ft(fp, trim=ACTIVE_BAND)
    ftmw.estimate_noise(fp)
    size_after_stage2 = fp.stat().st_size

    ftmw.calibrate_tau(fp)
    ftmw.detect_peaks(fp)
    ftmw.assign_windows(fp)

    plan = load_windows_impl(str(fp))["plan"]
    keep: List[int] = []
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

    fit = ftmw.fit_peaks(fp, jobs=1)
    ftmw.calibrate_timebase(fp)
    ftmw.review_run(fp)

    return BuiltPipeline(
        path=fp,
        size_after_import=size_after_import,
        size_after_stage2=size_after_stage2,
        raw_source_bytes=_raw_source_bytes(),
        fit=fit,
    )
