"""Cross-interface consistency for the derived frequency-calibration read.

``api.frequency_calibration``, ``Pipeline.frequency_calibration``, and the CLI
``timebase state`` must report the same derived calibration for the same file,
in all three states -- an external tool that labels an axis from one of them
and a user checking the other must never disagree about whether the file's
frequencies are calibrated.
"""

import json
import subprocess
import sys
from dataclasses import asdict, replace

import h5py
import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import Pipeline
from ftmwpipeline.core.stage_fit_settings import ClockSource
from ftmwpipeline.core.stage_fit_settings import resolve as resolve_stage_fit_settings
from ftmwpipeline.fitting.timebase_calibration import TimebaseCalibrationResult
from ftmwpipeline.io.stage_fit_settings_serialization import (
    save_stage_fit_settings_to_h5,
)
from ftmwpipeline.io.timebase_serialization import (
    GROUP_PATH,
    save_timebase_calibration_to_hdf5,
)

PROBE_MHZ = 40960.0
EPS = 2.2e-6
SIGMA_EPS = 0.1e-6


def _make_native_ftmw(tmp_path):
    """Create a minimal imported .ftmw (native-HDF5 source, no embedded clocks)."""
    src = tmp_path / "src.h5"
    sig = np.cos(2 * np.pi * np.arange(1024) * 0.01)
    with h5py.File(src, "w") as f:
        f.attrs["ftmw_input_version"] = 1
        f.attrs["spacing_us"] = 0.02
        f.attrs["probe_freq_mhz"] = PROBE_MHZ
        f.create_dataset("fid", data=sig)
    out = tmp_path / "exp.ftmw"
    ftmw.import_data(str(out), source=str(src), format_name="ftmw-hdf5")
    return out


def _make_uncalibrated(path):
    """Declare an unlocked digitizer, with nothing measured against it."""
    resolved = resolve_stage_fit_settings(persisted=None)
    save_stage_fit_settings_to_h5(
        str(path),
        replace(
            resolved,
            spur=replace(
                resolved.spur,
                clocks=(
                    ClockSource(5120.0, locked=True),
                    ClockSource(6250.0, locked=False),
                ),
            ),
        ),
    )


def _make_self_calibrated(path):
    _make_uncalibrated(path)
    result = TimebaseCalibrationResult(
        epsilon=EPS,
        sigma_epsilon=SIGMA_EPS,
        n_used=5,
        n_detected=5,
        lattice_g_mhz=320.0,
        tone_reads=(),
        kappa_sys=0.0,
        snr_min=10.0,
        sample_dt_us=0.02,
        start_us=0.0,
        end_us=13.0,
        span_us=13.0,
        preconditions_passed=True,
    )
    with h5py.File(str(path), "a") as h5f:
        if GROUP_PATH in h5f:
            del h5f[GROUP_PATH]
        save_timebase_calibration_to_hdf5(result, h5f.create_group(GROUP_PATH))


def _cli_state(path, *args):
    proc = subprocess.run(
        [sys.executable, "-m", "ftmwpipeline", "timebase", "state", str(path), *args],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return proc


@pytest.mark.integration
class TestCalibrationStateCrossInterface:
    @pytest.mark.parametrize(
        "prepare,expected_state",
        [
            (lambda p: None, "rb_locked"),
            (_make_uncalibrated, "uncalibrated"),
            (_make_self_calibrated, "self_calibrated"),
        ],
    )
    def test_all_three_interfaces_agree(self, tmp_path, prepare, expected_state):
        path = _make_native_ftmw(tmp_path)
        prepare(path)

        from_api = ftmw.frequency_calibration(str(path))
        from_pipeline = Pipeline(str(path)).frequency_calibration()
        from_cli = json.loads(_cli_state(path, "--format", "json").stdout)

        assert from_api.state == expected_state
        assert from_pipeline == from_api
        assert from_cli == asdict(from_api)

    def test_text_output_names_the_state(self, tmp_path):
        path = _make_native_ftmw(tmp_path)
        _make_self_calibrated(path)
        out = _cli_state(path).stdout
        assert "self_calibrated" in out
        assert "+2.200" in out  # epsilon in ppm

    def test_cli_missing_file_is_a_user_error(self, tmp_path):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "ftmwpipeline",
                "timebase",
                "state",
                str(tmp_path / "nope.ftmw"),
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 1
        assert "data import" in (proc.stdout + proc.stderr)

    def test_no_interface_mutates_the_file(self, tmp_path):
        """All three are reads: none of them may touch the file."""
        path = _make_native_ftmw(tmp_path)
        _make_self_calibrated(path)
        before = path.read_bytes()

        ftmw.frequency_calibration(str(path))
        Pipeline(str(path)).frequency_calibration()
        _cli_state(path, "--format", "json")

        assert path.read_bytes() == before
