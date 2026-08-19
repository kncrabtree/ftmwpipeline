"""The public read of a file's derived frequency calibration.

``frequency_calibration_impl`` answers, for **any** file at **any** stage and
without mutating it, "what frame is this file's data in, and by how much" --
the question a display layer must answer before it may label an axis
"calibrated". The properties under test are the ones that make it safe to
build on:

- **Total.** Every missing input degrades to the documented default rather
  than raising, so a bare import (no FT, no fit, no Stage 6) still answers.
- **Derived, never stored.** It reads the clock declaration and the live
  timebase calibration, so it can never disagree with what a
  ``frame="calibrated"`` call will actually apply, and it is the single
  definition the final-products staleness stamp is built from.
- **Read-only.** It never writes to the file it reads.

The fixtures here are deliberately synthetic and cheap (a 1024-point native
HDF5 import): needing the full 2638 build to ask a pre-Stage-1 question would
undercut the very claim being tested.
"""

from __future__ import annotations

import shutil
from dataclasses import replace

import h5py
import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.stage6_impl import (
    _current_calibration_stamp,
    frequency_calibration_impl,
)
from ftmwpipeline.core.calibration import CalibrationStamp
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


@pytest.fixture
def imported(tmp_path):
    """A bare imported .ftmw: Stage 0 and nothing else."""
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


def _persist_unlocked_digitizer(path) -> None:
    """Persist Stage 5 fit settings declaring an unlocked digitizer clock."""
    resolved = resolve_stage_fit_settings(persisted=None)
    settings = replace(
        resolved,
        spur=replace(
            resolved.spur,
            clocks=(
                ClockSource(5120.0, locked=True),
                ClockSource(6250.0, locked=False),
            ),
        ),
    )
    save_stage_fit_settings_to_h5(str(path), settings)


def _stamp_timebase(path, *, epsilon=EPS, sigma_epsilon=SIGMA_EPS, passed=True) -> None:
    result = TimebaseCalibrationResult(
        epsilon=epsilon,
        sigma_epsilon=sigma_epsilon,
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
        preconditions_passed=passed,
    )
    with h5py.File(str(path), "a") as h5f:
        if GROUP_PATH in h5f:
            del h5f[GROUP_PATH]
        save_timebase_calibration_to_hdf5(result, h5f.create_group(GROUP_PATH))


# ---------------------------------------------------------------------------
# The three states
# ---------------------------------------------------------------------------


class TestDerivedState:
    def test_bare_import_is_rb_locked(self, imported):
        """Nothing declared: the "assume Rb-locked" default, on a file that has
        run no stage at all. epsilon is a null op, not an unknown."""
        stamp = frequency_calibration_impl(imported)
        assert isinstance(stamp, CalibrationStamp)
        assert stamp.state == "rb_locked"
        assert stamp.epsilon == 0.0
        assert stamp.sigma_epsilon == 0.0
        assert stamp.sigma_floor_khz == 0.0
        assert stamp.probe_freq_mhz == pytest.approx(PROBE_MHZ)
        assert stamp.sideband == "upper"

    def test_unlocked_digitizer_without_calibration_is_uncalibrated(self, imported):
        """An unlocked clock with nothing measured must NOT read rb_locked --
        that would claim an absolute axis the file cannot support."""
        _persist_unlocked_digitizer(imported)
        stamp = frequency_calibration_impl(imported)
        assert stamp.state == "uncalibrated"
        assert stamp.epsilon == 0.0
        assert stamp.sigma_epsilon == 0.0

    def test_unlocked_digitizer_with_calibration_is_self_calibrated(self, imported):
        _persist_unlocked_digitizer(imported)
        _stamp_timebase(imported)
        stamp = frequency_calibration_impl(imported)
        assert stamp.state == "self_calibrated"
        assert stamp.epsilon == pytest.approx(EPS)
        assert stamp.sigma_epsilon == pytest.approx(SIGMA_EPS)

    def test_failed_preconditions_is_uncalibrated(self, imported):
        """A measurement whose preconditions did not pass is not a calibration:
        the state degrades and the measured epsilon is not handed out."""
        _persist_unlocked_digitizer(imported)
        _stamp_timebase(imported, passed=False)
        stamp = frequency_calibration_impl(imported)
        assert stamp.state == "uncalibrated"
        assert stamp.epsilon == 0.0

    def test_locked_clocks_only_stay_rb_locked(self, imported):
        """Declaring only locked clocks is not a reason to doubt the axis."""
        resolved = resolve_stage_fit_settings(persisted=None)
        save_stage_fit_settings_to_h5(
            str(imported),
            replace(
                resolved,
                spur=replace(resolved.spur, clocks=(ClockSource(5120.0, locked=True),)),
            ),
        )
        assert frequency_calibration_impl(imported).state == "rb_locked"

    def test_recommended_declaration_counts_before_stage5(self, imported):
        """``clocks set`` writes the recommended layer, and ``timebase run``
        accepts that layer -- so the derivation must read it too. Reading only
        persisted Stage 5 settings would call this file rb_locked and silently
        drop the epsilon it actually measured."""
        ftmw.set_clock_sources(
            str(imported),
            [
                {"freq_mhz": 5120.0, "locked": True},
                {"freq_mhz": 6250.0, "locked": False},
            ],
        )
        assert frequency_calibration_impl(imported).state == "uncalibrated"

        _stamp_timebase(imported)
        stamp = frequency_calibration_impl(imported)
        assert stamp.state == "self_calibrated"
        assert stamp.epsilon == pytest.approx(EPS)


# ---------------------------------------------------------------------------
# The floor, and the properties the surface promises
# ---------------------------------------------------------------------------


class TestStampContents:
    def test_sigma_floor_is_the_declared_one(self, imported):
        """The read counterpart of ``set_sigma_floor``: the floor a budget will
        be built with, readable before any budget exists."""
        assert frequency_calibration_impl(imported).sigma_floor_khz == 0.0
        ftmw.set_sigma_floor(str(imported), 3.5)
        assert frequency_calibration_impl(imported).sigma_floor_khz == pytest.approx(
            3.5
        )

    def test_is_read_only(self, imported, tmp_path):
        """Byte-for-byte identical after the read: a display layer must be able
        to ask this of a file it has no business writing to."""
        _persist_unlocked_digitizer(imported)
        _stamp_timebase(imported)
        before = tmp_path / "before.ftmw"
        shutil.copy(imported, before)
        frequency_calibration_impl(imported)
        assert imported.read_bytes() == before.read_bytes()

    def test_frozen(self, imported):
        stamp = frequency_calibration_impl(imported)
        with pytest.raises(Exception):
            stamp.state = "self_calibrated"  # type: ignore[misc]

    def test_missing_file_raises_actionable_error(self, tmp_path):
        with pytest.raises(FileNotFoundError) as exc:
            frequency_calibration_impl(tmp_path / "nope.ftmw")
        assert "data import" in str(exc.value)

    def test_is_the_staleness_stamp(self, imported):
        """One definition, not two: the six-tuple the final-products staleness
        check compares against is this same read, flattened."""
        _persist_unlocked_digitizer(imported)
        _stamp_timebase(imported)
        ftmw.set_sigma_floor(str(imported), 1.25)
        stamp = frequency_calibration_impl(imported)
        assert _current_calibration_stamp(str(imported)) == (
            stamp.state,
            stamp.epsilon,
            stamp.sigma_epsilon,
            stamp.sigma_floor_khz,
            stamp.probe_freq_mhz,
            stamp.sideband,
        )

    def test_tracks_a_recalibration(self, imported):
        """Derived, never stored: a second timebase run is visible immediately,
        with no Stage 6 action to propagate it."""
        _persist_unlocked_digitizer(imported)
        _stamp_timebase(imported, epsilon=EPS)
        assert frequency_calibration_impl(imported).epsilon == pytest.approx(EPS)
        _stamp_timebase(imported, epsilon=3 * EPS)
        assert frequency_calibration_impl(imported).epsilon == pytest.approx(3 * EPS)


def test_settings_type_is_reachable() -> None:
    """The vocabulary is importable from the package top level, not only from a
    private module -- that is the point of publishing it."""
    import ftmwpipeline

    assert ftmwpipeline.CalibrationStamp is CalibrationStamp
    assert set(ftmwpipeline.CalibrationState.__args__) == {  # type: ignore[attr-defined]
        "rb_locked",
        "self_calibrated",
        "uncalibrated",
    }
