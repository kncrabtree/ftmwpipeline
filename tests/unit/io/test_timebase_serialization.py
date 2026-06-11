"""Round-trip tests for TimebaseCalibrationResult HDF5 serialization."""

from __future__ import annotations

import h5py

from ftmwpipeline.fitting.timebase_calibration import (
    TimebaseCalibrationResult,
    TimebaseToneRead,
)
from ftmwpipeline.io.timebase_serialization import (
    GROUP_PATH,
    SCHEMA_VERSION,
    load_timebase_calibration_from_hdf5,
    save_timebase_calibration_to_hdf5,
)


def _make_sample_result() -> TimebaseCalibrationResult:
    tones = (
        TimebaseToneRead(
            f_bb_mhz=320.0,
            k=1,
            df_mhz=-5.0e-3,
            sigma_mhz=6.4e-5,
            snr=12000.0,
            used=False,
            drift_control=False,
        ),
        TimebaseToneRead(
            f_bb_mhz=3200.0,
            k=10,
            df_mhz=6.4e-3,
            sigma_mhz=6.4e-4,
            snr=18000.0,
            used=True,
            drift_control=False,
        ),
        TimebaseToneRead(
            f_bb_mhz=6250.0,
            k=1,
            df_mhz=1.0e-4,
            sigma_mhz=1.3e-3,
            snr=9000.0,
            used=False,
            drift_control=True,
        ),
    )
    return TimebaseCalibrationResult(
        epsilon=2.006e-6,
        sigma_epsilon=1.2e-8,
        n_used=1,
        n_detected=2,
        lattice_g_mhz=320.0,
        tone_reads=tones,
        kappa_sys=0.2e-6,
        snr_min=8.0,
        sample_dt_us=2.0e-5,
        start_us=0.0,
        end_us=4.0,
        span_us=4.0,
        preconditions_passed=True,
        preconditions_notes=("ok",),
    )


def _assert_tone_equal(a: TimebaseToneRead, b: TimebaseToneRead) -> None:
    assert a.f_bb_mhz == b.f_bb_mhz
    assert a.k == b.k
    assert a.df_mhz == b.df_mhz
    assert a.sigma_mhz == b.sigma_mhz
    assert a.snr == b.snr
    assert a.used == b.used
    assert a.drift_control == b.drift_control


def test_round_trip(tmp_path):
    result = _make_sample_result()
    path = tmp_path / "tb.h5"
    with h5py.File(path, "w") as h5f:
        grp = h5f.create_group(GROUP_PATH)
        save_timebase_calibration_to_hdf5(result, grp)
        grp.attrs["creation_time"] = "2026-06-11T00:00:00"
        assert h5f[GROUP_PATH]["algorithm_info"].attrs["version"] == SCHEMA_VERSION

    with h5py.File(path, "r") as h5f:
        loaded = load_timebase_calibration_from_hdf5(h5f[GROUP_PATH])

    assert loaded.epsilon == result.epsilon
    assert loaded.sigma_epsilon == result.sigma_epsilon
    assert loaded.n_used == result.n_used
    assert loaded.n_detected == result.n_detected
    assert loaded.lattice_g_mhz == result.lattice_g_mhz
    assert loaded.kappa_sys == result.kappa_sys
    assert loaded.snr_min == result.snr_min
    assert loaded.sample_dt_us == result.sample_dt_us
    assert loaded.start_us == result.start_us
    assert loaded.end_us == result.end_us
    assert loaded.span_us == result.span_us
    assert loaded.preconditions_passed == result.preconditions_passed
    assert loaded.preconditions_notes == result.preconditions_notes
    assert len(loaded.tone_reads) == len(result.tone_reads)
    for a, b in zip(loaded.tone_reads, result.tone_reads):
        _assert_tone_equal(a, b)


def test_round_trip_empty_tones(tmp_path):
    result = TimebaseCalibrationResult(
        epsilon=0.0,
        sigma_epsilon=float("inf"),
        n_used=0,
        n_detected=0,
        lattice_g_mhz=0.0,
        tone_reads=tuple(),
        kappa_sys=0.2e-6,
        snr_min=8.0,
        sample_dt_us=2.0e-5,
        start_us=0.0,
        end_us=4.0,
        span_us=4.0,
        preconditions_passed=False,
        preconditions_notes=("no locked clocks declared",),
    )
    path = tmp_path / "tb_empty.h5"
    with h5py.File(path, "w") as h5f:
        grp = h5f.create_group(GROUP_PATH)
        save_timebase_calibration_to_hdf5(result, grp)
    with h5py.File(path, "r") as h5f:
        loaded = load_timebase_calibration_from_hdf5(h5f[GROUP_PATH])
    assert loaded.tone_reads == tuple()
    assert loaded.sigma_epsilon == float("inf")
    assert not loaded.preconditions_passed
    assert loaded.preconditions_notes == ("no locked clocks declared",)
