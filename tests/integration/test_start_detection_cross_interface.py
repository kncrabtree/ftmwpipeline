"""Cross-interface consistency for start-time detection (Pipeline / api / CLI).

All three surfaces must land on the same recommended ``start_us`` for the same
FID + same knobs, and the stamped value must flow into a subsequent
``compute_ft`` through the recommended-settings resolution chain.
"""

from __future__ import annotations

import shutil
import subprocess

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline.core.data_structures import FID
from ftmwpipeline.core.start_detection_settings import StartDetectionSettings
from ftmwpipeline.pipeline import Pipeline
from ftmwpipeline.preprocessing.start_detection import detect_start_time

# Coarse, band-restricted sweep so the three runs are quick but still resolve
# the chirp-end corner on the 2638 fixture.
_KW = dict(sweep_max_us=4.0, step_us=0.1, band=(26500.0, 40000.0))
_SETTINGS = StartDetectionSettings(
    sweep_max_us=4.0, step_us=0.1, band_min_mhz=26500.0, band_max_mhz=40000.0
)


def _run_cli(args) -> None:
    result = subprocess.run(
        ["ftmwpipeline"] + args,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"CLI {' '.join(args)} failed ({result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}"
        )


def _stamped_start_us(path) -> float:
    with h5py.File(path, "r") as h:
        return float(h["stage0_fid_data/recommended_processing"].attrs["start_us"])


@pytest.mark.integration
def test_start_detection_identical_across_interfaces(exp_2638_data_path, tmp_path):
    base = tmp_path / "base.ftmw"
    ftmw.import_data(str(base), source=exp_2638_data_path, force=True)

    p_copy = tmp_path / "pipeline.ftmw"
    f_copy = tmp_path / "functional.ftmw"
    c_copy = tmp_path / "cli.ftmw"
    for dst in (p_copy, f_copy, c_copy):
        shutil.copy(base, dst)

    res_pipeline = Pipeline.open(p_copy).detect_start_time(settings=_SETTINGS)
    res_functional = ftmw.detect_start_time(f_copy, settings=_SETTINGS)
    _run_cli(
        [
            "detect-start",
            str(c_copy),
            "--sweep-max-us",
            "4.0",
            "--step-us",
            "0.1",
            "--band",
            "26500",
            "40000",
        ]
    )

    # Identical recommended start across the in-process interfaces.
    assert res_pipeline.start_us == res_functional.start_us
    assert res_pipeline.chirp_end_us == res_functional.chirp_end_us

    # And the CLI stamped the same value (read back off the file).
    assert _stamped_start_us(c_copy) == pytest.approx(res_pipeline.start_us, abs=1e-9)
    assert _stamped_start_us(p_copy) == pytest.approx(res_pipeline.start_us, abs=1e-9)
    assert _stamped_start_us(f_copy) == pytest.approx(res_functional.start_us, abs=1e-9)

    # Sanity: a real chirp was found near the expected 2638 location.
    assert res_pipeline.chirp_detected
    assert res_pipeline.start_us == pytest.approx(2.35, abs=0.25)


@pytest.mark.integration
def test_chopped_fixture_has_no_chirp(exp_2638_data_path, tmp_path):
    """Excise the first 5 us (chirp + ringdown) from the real 2638 FID; the
    remaining pure molecular decay must read as no-chirp with a ~0 start."""
    path = tmp_path / "chop.ftmw"
    ftmw.import_data(str(path), source=exp_2638_data_path, force=True)
    fid = ftmw.load_fid(path)

    settings = StartDetectionSettings(
        step_us=0.05, band_min_mhz=26500.0, band_max_mhz=40000.0
    )
    full = detect_start_time(fid, settings=settings)

    n_chop = int(round(5.0e-6 / fid.spacing))
    chopped = FID(
        data=fid.data[n_chop:],
        spacing=fid.spacing,
        probe_freq_mhz=fid.probe_freq_mhz,
        sideband=fid.sideband,
        shots=fid.shots,
    )
    chop = detect_start_time(chopped, settings=settings)

    # The intact FID has an unmistakable chirp; the chopped one does not.
    assert full.chirp_detected
    assert not chop.chirp_detected
    assert chop.start_us == 0.0
    # The drop ratio cleanly separates the two cases around the gate.
    assert full.plateau / full.floor > settings.min_chirp_drop_ratio
    assert chop.plateau / chop.floor < settings.min_chirp_drop_ratio


@pytest.mark.integration
def test_stamped_start_us_flows_into_compute_ft(exp_2638_data_path, tmp_path):
    path = tmp_path / "inherit.ftmw"
    ftmw.import_data(str(path), source=exp_2638_data_path, force=True)

    res = ftmw.detect_start_time(path, settings=_SETTINGS)
    # compute_ft with NO explicit start_us must inherit the stamped recommendation.
    ftmw.compute_ft(path, zpf=2, trim=(26500.0, 40000.0))
    with h5py.File(path, "r") as h:
        canonical = float(h["processing_parameters/ft_processing"].attrs["start_us"])
    assert canonical == pytest.approx(res.start_us, abs=1e-9)
