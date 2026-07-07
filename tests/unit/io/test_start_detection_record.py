"""Unit tests for the start-detection settings + outcome persistence layer.

Covers the HDF5 round-trip of :func:`write_recommended_start_detection` /
:func:`read_recommended_start_detection`: the record that lets a report replay
the exact Σ|FT|-vs-start sweep a ``start run`` invocation produced, instead of
re-running it with guessed default knobs.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from ftmwpipeline.core.start_detection_settings import StartDetectionSettings
from ftmwpipeline.io.stage_fit_settings_serialization import (
    read_recommended_start_detection,
    write_recommended_start_detection,
)
from ftmwpipeline.preprocessing.start_detection import StartDetectionResult


def _make_bare_stage0_ftmw(tmp_path: Path) -> str:
    p = tmp_path / "bare.ftmw"
    with h5py.File(p, "w") as h5f:
        h5f.create_group("stage0_fid_data")
    return str(p)


def _make_no_stage0_ftmw(tmp_path: Path) -> str:
    p = tmp_path / "no_stage0.ftmw"
    with h5py.File(p, "w") as h5f:
        h5f.create_group("other_group")
    return str(p)


def _result(**overrides: object) -> StartDetectionResult:
    import numpy as np

    defaults: dict = dict(
        start_us=2.27,
        chirp_end_us=1.6,
        chirp_detected=True,
        floor=100.0,
        plateau=1.0e6,
        band_mhz=None,
        starts_us=np.array([0.0, 1.0]),
        sum_magnitude=np.array([1.0e6, 100.0]),
    )
    defaults.update(overrides)
    return StartDetectionResult(**defaults)


class TestStartDetectionRecordRoundTrip:
    def test_absent_attr_returns_none(self, tmp_path: Path) -> None:
        p = _make_bare_stage0_ftmw(tmp_path)
        assert read_recommended_start_detection(p) is None

    def test_no_stage0_group_write_noop(self, tmp_path: Path) -> None:
        p = _make_no_stage0_ftmw(tmp_path)
        write_recommended_start_detection(p, StartDetectionSettings(), _result())
        assert read_recommended_start_detection(p) is None

    def test_round_trip_settings(self, tmp_path: Path) -> None:
        p = _make_bare_stage0_ftmw(tmp_path)
        settings = StartDetectionSettings(
            sweep_max_us=6.0,
            step_us=0.05,
            floor_factor=2.5,
            floor_tail_us=0.8,
            guard_margin_us=0.9,
            min_chirp_drop_ratio=8.0,
        )
        write_recommended_start_detection(p, settings, _result())
        record = read_recommended_start_detection(p)
        assert record is not None
        assert record.settings == settings

    def test_round_trip_band_override(self, tmp_path: Path) -> None:
        p = _make_bare_stage0_ftmw(tmp_path)
        settings = StartDetectionSettings(band_min_mhz=100.0, band_max_mhz=200.0)
        write_recommended_start_detection(p, settings, _result(band_mhz=(100.0, 200.0)))
        record = read_recommended_start_detection(p)
        assert record is not None
        assert record.settings.band_min_mhz == pytest.approx(100.0)
        assert record.settings.band_max_mhz == pytest.approx(200.0)
        assert record.band_mhz == pytest.approx((100.0, 200.0))

    def test_round_trip_no_band(self, tmp_path: Path) -> None:
        p = _make_bare_stage0_ftmw(tmp_path)
        write_recommended_start_detection(
            p, StartDetectionSettings(), _result(band_mhz=None)
        )
        record = read_recommended_start_detection(p)
        assert record is not None
        assert record.settings.band_min_mhz is None
        assert record.settings.band_max_mhz is None
        assert record.band_mhz is None

    def test_round_trip_diagnostic_fields(self, tmp_path: Path) -> None:
        p = _make_bare_stage0_ftmw(tmp_path)
        write_recommended_start_detection(
            p,
            StartDetectionSettings(),
            _result(chirp_end_us=1.7, chirp_detected=True, floor=42.0, plateau=4.2e5),
        )
        record = read_recommended_start_detection(p)
        assert record is not None
        assert record.chirp_end_us == pytest.approx(1.7)
        assert record.chirp_detected is True
        assert record.floor == pytest.approx(42.0)
        assert record.plateau == pytest.approx(4.2e5)

    def test_round_trip_no_chirp_found(self, tmp_path: Path) -> None:
        p = _make_bare_stage0_ftmw(tmp_path)
        write_recommended_start_detection(
            p, StartDetectionSettings(), _result(chirp_detected=False)
        )
        record = read_recommended_start_detection(p)
        assert record is not None
        assert record.chirp_detected is False

    def test_overwrite_replaces_previous(self, tmp_path: Path) -> None:
        p = _make_bare_stage0_ftmw(tmp_path)
        write_recommended_start_detection(
            p, StartDetectionSettings(), _result(chirp_end_us=1.0)
        )
        write_recommended_start_detection(
            p, StartDetectionSettings(), _result(chirp_end_us=2.5)
        )
        record = read_recommended_start_detection(p)
        assert record is not None
        assert record.chirp_end_us == pytest.approx(2.5)

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert (
            read_recommended_start_detection(str(tmp_path / "nonexistent.ftmw")) is None
        )
