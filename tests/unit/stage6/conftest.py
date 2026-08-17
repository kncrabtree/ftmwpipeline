"""Shared Stage-6 fixtures.

Every Stage-6 test file used to define an identical ``scope="module"`` fixture
that ran the full 2638 build (import -> FT -> noise -> peaks -> windows -> trim to
the first 3 dependency-free windows -> ``fit_peaks``), i.e. the same expensive
fixture rebuilt once per file. These session-scoped builds run it **once** and
hand each test a fresh writable copy (function scope), so there is no cross-test
mutation risk and the build cost is paid a single time.

``test_report_full`` keeps its own local ``stage5_small_file`` (its build also
runs ``calibrate_tau``, which the report exercises); a file-local fixture
overrides this conftest one, so that module is unaffected.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.stage4_impl import load_windows_impl, save_window_plan_impl
from ftmwpipeline._internal.stage6_impl import review_run_impl

_DATA = Path("examples/blackchirp_data/2638")


@pytest.fixture(scope="session")
def exp_2638_data_path() -> str:
    if not _DATA.exists():
        pytest.skip("Experiment 2638 data not available")
    return str(_DATA)


def _build_stage5_small(dest: Path, data_path: str) -> None:
    """Import 2638, trim to the first 3 dependency-free windows, fit Stage 5."""
    ftmw.import_data(dest, source=data_path)
    ftmw.compute_ft(dest, trim=(26500, 40000))
    ftmw.estimate_noise(dest)
    ftmw.detect_peaks(dest)
    ftmw.assign_windows(dest)

    plan = load_windows_impl(str(dest))["plan"]
    candidates: List[int] = []
    for wid in plan.topological_order:
        deps = [(a, b) for (a, b) in plan.dependency_edges if a == wid or b == wid]
        if all(a in candidates or a == wid for (a, _) in deps) and all(
            b in candidates or b == wid for (_, b) in deps
        ):
            candidates.append(wid)
        if len(candidates) >= 3:
            break
    if not candidates:
        candidates = list(plan.topological_order[:3])
    keep = set(candidates)
    plan.windows = [w for w in plan.windows if w.window_id in keep]
    plan.topological_order = [w for w in plan.topological_order if w in keep]
    plan.dependency_edges = [
        (a, b) for (a, b) in plan.dependency_edges if a in keep and b in keep
    ]
    save_window_plan_impl(str(dest), plan)

    ftmw.fit_peaks(str(dest))


@pytest.fixture(scope="session")
def _stage5_small_built(exp_2638_data_path, tmp_path_factory) -> Path:
    """The shared post-fit (pre-review) small build -- read-only, built once."""
    fp = tmp_path_factory.mktemp("stage6_shared") / "stage5_small.ftmw"
    _build_stage5_small(fp, exp_2638_data_path)
    return fp


@pytest.fixture(scope="session")
def _stage5_reviewed_built(_stage5_small_built, tmp_path_factory) -> Path:
    """The shared post-review small build -- read-only, built once."""
    fp = tmp_path_factory.mktemp("stage6_shared_reviewed") / "stage5_reviewed.ftmw"
    shutil.copy(_stage5_small_built, fp)
    review_run_impl(str(fp))
    return fp


@pytest.fixture(scope="session")
def stage5_small_source(_stage5_small_built) -> Path:
    """The shared post-fit build, for a test that copies before it writes.

    Read-only: copy it, never open it for writing. ``stage5_small_file`` already
    hands out a private per-test copy, so a test that immediately copies *that*
    to its own name is paying for two copies of a ~7 MB file and reading neither.
    Copy from here instead -- the bytes are identical, and the discarded copy is
    the one that goes away.
    """
    return _stage5_small_built


@pytest.fixture(scope="session")
def stage5_reviewed_source(_stage5_reviewed_built) -> Path:
    """The shared post-review build. Read-only; see :func:`stage5_small_source`."""
    return _stage5_reviewed_built


@pytest.fixture
def stage5_small_file(_stage5_small_built, tmp_path) -> Path:
    """A fresh writable copy of the shared post-fit small fixture."""
    fp = tmp_path / "stage5_small.ftmw"
    shutil.copy(_stage5_small_built, fp)
    return fp


@pytest.fixture
def stage5_reviewed_file(_stage5_reviewed_built, tmp_path) -> Path:
    """A fresh writable copy of the shared post-review small fixture."""
    fp = tmp_path / "stage5_reviewed.ftmw"
    shutil.copy(_stage5_reviewed_built, fp)
    return fp


@pytest.fixture
def stage5_file(stage5_small_file) -> Path:
    """Name alias for the post-fit small fixture (used by test_curation)."""
    return stage5_small_file
