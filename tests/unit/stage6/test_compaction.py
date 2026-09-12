"""A ``.ftmw`` on disk stays as small as its content: every stage run and
every curation write compacts the file (``_internal.compaction``), so the
attribute and vlen churn Stage 6 inflicts -- a review group rewritten per
apply, a fit table restored per undo -- does not accumulate."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.compaction import compact_file
from ftmwpipeline._internal.stage6_impl import review_run_impl, review_undo_impl
from tests.unit._internal.test_compaction import _dump

pytestmark = [pytest.mark.integration]


def _a_peak(path: Path):
    import h5py

    from ftmwpipeline.io.fitting_serialization import (
        read_fit_peak_frequencies_by_window,
    )

    with h5py.File(path, "r") as f:
        by_w = read_fit_peak_frequencies_by_window(f["stage5_fitting"])
    wid = next((w for w, fs in by_w.items() if fs), None)
    if wid is None:
        pytest.skip("No fitted peaks in the built subset")
    return wid, by_w[wid][0]


def test_compacting_a_pipeline_file_preserves_its_content(stage5_small_file):
    content = _dump(stage5_small_file)
    compact_file(stage5_small_file)
    assert _dump(stage5_small_file) == content


def test_edit_undo_churn_does_not_grow_the_file(stage5_small_file):
    """Five edit/undo cycles; the size after each cycle stays put."""
    wid, freq = _a_peak(stage5_small_file)
    sizes = []
    for _ in range(5):
        ftmw.review_edit(str(stage5_small_file), wid, remove=[freq])
        last = ftmw.review_log(str(stage5_small_file))[-1].order_index
        review_undo_impl(stage5_small_file, [last])
        sizes.append(os.path.getsize(stage5_small_file))
    assert max(sizes) <= 1.02 * min(sizes), sizes


def test_stage_rerun_does_not_grow_the_file(stage5_small_file):
    review_run_impl(stage5_small_file)
    base = os.path.getsize(stage5_small_file)
    for _ in range(4):
        review_run_impl(stage5_small_file)
    assert os.path.getsize(stage5_small_file) <= 1.02 * base


def test_no_temp_file_is_left_beside_the_file(stage5_small_file, tmp_path):
    review_run_impl(stage5_small_file)
    assert sorted(p.name for p in tmp_path.iterdir()) == [stage5_small_file.name]
