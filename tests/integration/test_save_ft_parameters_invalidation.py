"""``save_ft_parameters`` reports the stages it invalidated.

``dev-docs/CONTRACT_STRATEGY.md`` §Status and settings: a call that changes
what the file's results were built on says which stages it invalidated, as
canonical stage names in re-run order, empty when nothing was. ``save_ft_
parameters`` has no ``events`` argument and no event scope, so the report is
its return value and the warning log line; ``visualize_ft(save_params=True)``
still returns its figure.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import List

import h5py
import matplotlib
import matplotlib.figure
import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.stage1_impl import save_ft_parameters_impl
from ftmwpipeline.contract import Stage
from ftmwpipeline.file_manager import rerun_order
from ftmwpipeline.pipeline import Pipeline

matplotlib.use("Agg")

pytestmark = [pytest.mark.integration]

_DT_US = 0.002
_N = 6325


def _completed(path: Path) -> List[str]:
    with h5py.File(path, "r") as h5f:
        return sorted(json.loads(h5f["pipeline_stages"].attrs["completed_stages"]))


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _build(directory: Path, *, through: str) -> Path:
    src = directory / "src.h5"
    t = np.arange(_N) * _DT_US
    rng = np.random.default_rng(0)
    fid = (
        np.cos(2 * np.pi * 37.0 * t) * np.exp(-t / 3.0)
        + np.cos(2 * np.pi * 91.0 * t) * np.exp(-t / 3.0)
        + 0.01 * rng.standard_normal(_N)
    )
    with h5py.File(src, "w") as f:
        f.attrs["ftmw_input_version"] = 1
        f.attrs["spacing_us"] = _DT_US
        f.attrs["probe_freq_mhz"] = 40960.0
        f.create_dataset("fid", data=fid)
    out = directory / f"{through}.ftmw"
    Pipeline.create(str(out), source=str(src), format_name="ftmw-hdf5")
    ftmw.compute_ft(out)
    if through in ("noise", "peaks"):
        ftmw.estimate_noise(out)
    if through == "peaks":
        ftmw.detect_peaks(out)
    return out


@pytest.fixture(scope="module")
def _templates(tmp_path_factory):
    root = tmp_path_factory.mktemp("save_ft")
    out = {}
    for through in ("ft", "peaks"):
        directory = root / through
        directory.mkdir()
        out[through] = _build(directory, through=through)
    return out


@pytest.fixture
def ft_only(_templates, tmp_path):
    dst = tmp_path / "ft.ftmw"
    shutil.copy(_templates["ft"], dst)
    return dst


@pytest.fixture
def peaks_file(_templates, tmp_path):
    dst = tmp_path / "peaks.ftmw"
    shutil.copy(_templates["peaks"], dst)
    return dst


# ---------------------------------------------------------------------------
# The return value
# ---------------------------------------------------------------------------


def test_an_unchanged_record_reports_an_empty_list(peaks_file):
    """Mutation: return ``None`` again, or report what was merely re-saved."""
    before = _completed(peaks_file)
    for parameters in ({}, {"start_us": 0.0}):
        result = ftmw.save_ft_parameters(peaks_file, parameters)
        assert result == [] and isinstance(result, list)
    assert _completed(peaks_file) == before


def test_a_changed_record_reports_the_downstream_stages_canonically(peaks_file):
    """Mutation: report storage keys (``stage2_noise_result``), or nothing."""
    assert _completed(peaks_file) == [
        "stage0_fid_data",
        "stage1_complex_ft",
        "stage2_noise_result",
        "stage3_peaks",
    ]
    result = ftmw.save_ft_parameters(peaks_file, {"start_us": 0.5})
    assert result == ["noise", "peaks"]
    assert set(result) <= {stage.value for stage in Stage}
    assert result == [name for name in rerun_order() if name in result]
    assert type(result) is list and all(type(name) is str for name in result)
    # What was reported is exactly what left the file.
    assert _completed(peaks_file) == ["stage0_fid_data", "stage1_complex_ft"]
    with h5py.File(peaks_file, "r") as h5f:
        assert "stage3_peaks" not in h5f and "stage2_noise_result" not in h5f


def test_a_second_identical_save_reports_nothing(peaks_file):
    assert ftmw.save_ft_parameters(peaks_file, {"start_us": 0.5}) == ["noise", "peaks"]
    assert ftmw.save_ft_parameters(peaks_file, {"start_us": 0.5}) == []


def test_a_change_with_nothing_downstream_reports_nothing(ft_only):
    """Only Stage 1 exists: the record changes, no stage was built on it."""
    assert ftmw.save_ft_parameters(ft_only, {"start_us": 0.5}) == []
    with h5py.File(ft_only, "r") as h5f:
        record = h5f["processing_parameters/ft_processing"].attrs["start_us"]
    assert record == 0.5


def test_after_a_partial_rebuild_only_the_stages_that_exist_are_reported(peaks_file):
    """Save once (noise and peaks go), rebuild noise alone, change the record
    again: only the noise result is there to invalidate."""
    ftmw.save_ft_parameters(peaks_file, {"start_us": 0.5})
    ftmw.estimate_noise(peaks_file)
    assert ftmw.save_ft_parameters(peaks_file, {"start_us": 0.25}) == ["noise"]


def test_the_warning_log_line_names_the_invalidation(peaks_file, caplog):
    """The report has no event scope: the warning is the human-facing half."""
    with caplog.at_level(logging.WARNING, logger="ftmwpipeline"):
        ftmw.save_ft_parameters(peaks_file, {"start_us": 0.5})
    assert any(
        record.levelno == logging.WARNING and "invalidated" in record.getMessage()
        for record in caplog.records
    )


def test_no_warning_when_nothing_was_invalidated(peaks_file, caplog):
    with caplog.at_level(logging.WARNING, logger="ftmwpipeline"):
        ftmw.save_ft_parameters(peaks_file, {"start_us": 0.0})
    assert not any("invalidated" in r.getMessage() for r in caplog.records)


def test_the_impl_returns_the_same_canonical_list(peaks_file):
    """The functional API is a thin wrapper over the impl's return value."""
    assert save_ft_parameters_impl(str(peaks_file), {"start_us": 0.5}) == [
        "noise",
        "peaks",
    ]


def test_a_missing_file_still_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ftmw.save_ft_parameters(tmp_path / "gone.ftmw", {"start_us": 0.5})
    assert not (tmp_path / "gone.ftmw").exists()


# ---------------------------------------------------------------------------
# visualize_ft(save_params=True) still returns the figure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("via", ["api", "pipeline"])
def test_visualize_ft_with_save_params_returns_the_figure_and_names_the_stages(
    peaks_file, via, caplog
):
    """Mutation: return the invalidated list from ``visualize_ft``."""
    kwargs = dict(
        start_us=0.5, save_params=True, interactive=False, show_fid_panels=False
    )
    with caplog.at_level(logging.INFO, logger="ftmwpipeline"):
        if via == "api":
            fig = ftmw.visualize_ft(peaks_file, **kwargs)
        else:
            fig = Pipeline.open(peaks_file).visualize_ft(**kwargs)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert _completed(peaks_file) == ["stage0_fid_data", "stage1_complex_ft"]
    saved = [
        r.getMessage()
        for r in caplog.records
        if "processing parameters" in r.getMessage()
    ]
    assert saved == ["Saved 1 processing parameters; invalidated noise, peaks"]


def test_visualize_ft_without_a_change_says_nothing_was_invalidated(peaks_file, caplog):
    with caplog.at_level(logging.INFO, logger="ftmwpipeline"):
        fig = ftmw.visualize_ft(
            peaks_file,
            start_us=0.0,
            save_params=True,
            interactive=False,
            show_fid_panels=False,
        )
    assert isinstance(fig, matplotlib.figure.Figure)
    saved = [
        r.getMessage()
        for r in caplog.records
        if "processing parameters" in r.getMessage()
    ]
    assert saved == ["Saved 1 processing parameters"]
    assert "stage3_peaks" in _completed(peaks_file)


def test_visualize_ft_without_save_params_does_not_touch_the_file(peaks_file):
    before = _md5(peaks_file)
    ftmw.visualize_ft(
        peaks_file, start_us=0.5, interactive=False, show_fid_panels=False
    )
    assert _md5(peaks_file) == before
