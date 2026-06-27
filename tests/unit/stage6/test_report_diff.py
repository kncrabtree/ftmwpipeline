"""Tests for the post-curation before/after diff report (``report diff``)."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.report_diff_impl import report_diff_impl
from ftmwpipeline._internal.stage6_impl import refit_window_impl
from ftmwpipeline.pipeline import Pipeline


def _edit_weakest_peak(path: str) -> int:
    """Remove the weakest peak from the first window with >= 2 peaks (snapshots
    the automatic-fit baseline as a side effect). Returns the edited window id."""
    fit = ftmw.load_fit(path)
    wf = next((w for w in fit.window_fits if len(w.fitted_peaks) >= 2), None)
    if wf is None:
        pytest.skip("fixture has no multi-peak window to edit")
    wid = int(wf.window_id)
    freq = float(min(wf.fitted_peaks, key=lambda p: (p.snr or 0.0)).frequency_mhz)
    refit_window_impl(path, wid, remove=[freq])
    return wid


def test_no_baseline_writes_notice(stage5_reviewed_file, tmp_path):
    """With no curation edit (no baseline snapshot), a valid notice report is
    written rather than an error."""
    out = tmp_path / "out"
    path = report_diff_impl(str(stage5_reviewed_file), output_dir=str(out))

    p = Path(path)
    assert p.exists() and p.name.endswith("_diff.html")
    text = p.read_text()
    assert "<!DOCTYPE html>" in text
    assert "No curation" in text
    assert 'section class="win"' not in text  # no per-window sections


def test_edit_produces_before_after(stage5_reviewed_file, tmp_path):
    """After an edit, the diff report shows the changed window with side-by-side
    before/after panels."""
    wid = _edit_weakest_peak(str(stage5_reviewed_file))

    out = tmp_path / "out"
    path = report_diff_impl(str(stage5_reviewed_file), output_dir=str(out))
    text = Path(path).read_text()

    assert f"Window {wid}" in text
    assert "Before (automatic)" in text and "After (curated)" in text
    assert text.count("data:image/png;base64,") >= 2  # one panel per side
    assert text.count('section class="win"') >= 1


def test_diff_report_is_read_only(stage5_reviewed_file, tmp_path):
    """Rendering the diff report never mutates the input file."""
    _edit_weakest_peak(str(stage5_reviewed_file))
    before = hashlib.md5(Path(stage5_reviewed_file).read_bytes()).hexdigest()

    report_diff_impl(str(stage5_reviewed_file), output_dir=str(tmp_path / "out"))

    after = hashlib.md5(Path(stage5_reviewed_file).read_bytes()).hexdigest()
    assert before == after


def test_cross_interface_identical(stage5_reviewed_file, tmp_path):
    """The functional API, the Pipeline method, and the CLI verb produce a
    byte-identical diff report for the same edited file (dual-interface invariant)."""
    _edit_weakest_peak(str(stage5_reviewed_file))

    # Same stem in three dirs so the output filename and content match exactly.
    def _copy(name: str) -> Path:
        d = tmp_path / name
        d.mkdir()
        fp = d / "x.ftmw"
        shutil.copy(stage5_reviewed_file, fp)
        return fp

    api_fp, pipe_fp, cli_fp = _copy("api"), _copy("pipe"), _copy("cli")

    api_out = ftmw.report_diff(str(api_fp), output_dir=str(api_fp.parent))
    pipe_out = Pipeline.open(pipe_fp).report_diff(output_dir=str(pipe_fp.parent))

    result = subprocess.run(
        [
            "ftmwpipeline",
            "report",
            "diff",
            str(cli_fp),
            "--output-dir",
            str(cli_fp.parent),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    cli_out = cli_fp.parent / "x_diff.html"

    api_html = Path(api_out).read_text()
    pipe_html = Path(pipe_out).read_text()
    cli_html = cli_out.read_text()
    assert api_html == pipe_html
    assert api_html == cli_html
