"""How a file's stage state is reported and kept honest.

* A missing predecessor or a missing stage result is a
  :class:`StageDependencyError` naming the stage it needs and the command that
  produces it, on every interface -- not a prose ``ValueError`` or a
  ``RuntimeError`` that flattened the type. It is still a ``ValueError``.
* Re-running Stage 2 with different settings invalidates every stage built on
  the old sigma; an identical re-run leaves them standing.
* ``completed_stages`` comes back in one fixed order (the re-run order), and
  names stages by their canonical names (``Stage`` values), never by storage
  key.
* The display amplitude unit follows the same Stage 1 resolution as the
  spectrum it labels, including before Stage 1 has run.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline.core.noise_settings import NoiseSettings
from ftmwpipeline.file_manager import (
    PipelineFileError,
    StageDependencyError,
    rerun_order,
)
from ftmwpipeline.pipeline import Pipeline

pytestmark = [pytest.mark.unit]

_DT_US = 0.002
_N = 6325


@pytest.fixture
def imported(tmp_path):
    """A Stage-0-only file from a small synthetic two-line FID."""
    src = tmp_path / "src.h5"
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
    out = tmp_path / "exp.ftmw"
    Pipeline.create(str(out), source=str(src), format_name="ftmw-hdf5")
    return out


@pytest.fixture
def peaks_file(imported):
    ftmw.compute_ft(imported)
    ftmw.estimate_noise(imported)
    ftmw.detect_peaks(imported)
    return imported


def _completed(path) -> list:
    return ftmw.get_pipeline_info(path)["completed_stages"]


# ---------------------------------------------------------------------------
# Typed dependency errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call, missing, command",
    [
        (lambda p: ftmw.estimate_noise(p), "stage1_complex_ft", "ft run"),
        (lambda p: Pipeline.open(p).estimate_noise(), "stage1_complex_ft", "ft run"),
        (lambda p: ftmw.load_peaks(p), "stage3_peaks", "peaks run"),
        (lambda p: ftmw.assign_windows(p), "stage3_peaks", "peaks run"),
        (lambda p: ftmw.load_fit(p), "stage5_fitting", "fit run"),
        (lambda p: ftmw.review_run(p), "stage5_fitting", "fit run"),
        (lambda p: ftmw.report_table(p), "stage6_review", "review run"),
    ],
)
def test_missing_stage_is_a_typed_error(imported, call, missing, command):
    with pytest.raises(StageDependencyError) as excinfo:
        call(imported)
    err = excinfo.value
    assert missing in err.missing_dependencies
    assert err.command == command
    assert str(err.filepath) == str(imported)
    # Still what these refusals raised before they were typed.
    assert isinstance(err, ValueError)
    assert isinstance(err, PipelineFileError)


def test_missing_stage_error_keeps_its_actionable_message(imported):
    with pytest.raises(StageDependencyError, match=r"Run compute_ft\(\)"):
        ftmw.estimate_noise(imported)


# ---------------------------------------------------------------------------
# Stage 2 re-run invalidation
# ---------------------------------------------------------------------------


def test_changed_noise_settings_invalidate_downstream(peaks_file):
    assert "peaks" in _completed(peaks_file)
    ftmw.estimate_noise(peaks_file, settings=NoiseSettings(window_mhz=37.0))
    assert "peaks" not in _completed(peaks_file)
    assert "noise" in _completed(peaks_file)
    with h5py.File(peaks_file, "r") as h5f:
        assert "stage3_peaks" not in h5f


def test_identical_noise_rerun_keeps_downstream(peaks_file):
    with h5py.File(peaks_file, "r") as h5f:
        created = h5f["stage3_peaks"].attrs["creation_time"]
    ftmw.estimate_noise(peaks_file)
    with h5py.File(peaks_file, "r") as h5f:
        assert h5f["stage3_peaks"].attrs["creation_time"] == created


# ---------------------------------------------------------------------------
# Deterministic stage listing
# ---------------------------------------------------------------------------


def test_completed_stages_come_back_canonical_and_in_rerun_order(peaks_file):
    """Mutation: name storage keys again, or sort alphabetically (which puts
    ``fit`` before ``ft`` and ``peaks`` before ``tau``)."""
    stages = _completed(peaks_file)
    assert stages == ["data", "ft", "noise", "peaks"]
    assert stages == [s for s in rerun_order() if s in stages]
    assert Pipeline.open(peaks_file).info()["completed_stages"] == stages


# ---------------------------------------------------------------------------
# Display unit before Stage 1
# ---------------------------------------------------------------------------


def test_display_unit_matches_the_resolved_settings_before_stage1(imported):
    """Before Stage 1 persists anything, the display FT's unit is the one the
    resolved settings would use -- the same answer ``settings show`` gives --
    not "unitless"."""
    rows = {
        r.path: r.value for r in ftmw.settings_show(imported, include_advanced=True)
    }
    units_power = rows["stage1.units_power"]
    assert units_power is not None
    display = ftmw.compute_display_ft(imported)
    assert display.metadata["amplitude_scale"] == pytest.approx(10.0**units_power)
    assert display.metadata["units_label"]


# ---------------------------------------------------------------------------
# Status calls raise for a file they cannot open
# ---------------------------------------------------------------------------


def test_info_raises_for_a_missing_file(tmp_path):
    missing = tmp_path / "absent.ftmw"
    with pytest.raises(FileNotFoundError):
        ftmw.get_pipeline_info(missing)
    with pytest.raises(FileNotFoundError):
        ftmw.list_available_stages(missing)


def test_info_raises_typed_for_a_file_that_is_not_a_pipeline(tmp_path):
    bogus = tmp_path / "bogus.ftmw"
    with h5py.File(bogus, "w") as h5f:
        h5f.attrs["unrelated"] = 1
    with pytest.raises(PipelineFileError):
        ftmw.get_pipeline_info(bogus)


def test_info_reports_a_readable_file(imported):
    info = ftmw.get_pipeline_info(imported)
    assert info["valid"] is True
    assert "error" not in info
    assert ftmw.list_available_stages(imported) == info["next_available_stages"]
