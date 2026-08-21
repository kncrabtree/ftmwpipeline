"""Unit tests for the ChirpWindow declaration layer.

Covers:
1. ChirpWindow round-trip through write/read_recommended_chirp_window (HDF5
   helpers), including the ``__None__`` sentinel, absent attr, and missing
   Stage 0 group.
2. BlackChirpLoader._extract_chirp_window against the real 2638 example data.
3. KeysightMatLoader: chirp_start_us / chirp_end_us / start_margin_us params
   attach to fid.metadata and persist through .ftmw import.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from ftmwpipeline.core.data_structures import ChirpWindow
from ftmwpipeline.io.data_loaders.blackchirp import BlackChirpLoader
from ftmwpipeline.io.data_loaders.keysight_mat import KeysightMatLoader
from ftmwpipeline.io.stage_fit_settings_serialization import (
    read_recommended_chirp_window,
    write_recommended_chirp_window,
)

_DATA_2638 = Path("examples/blackchirp_data/2638")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bare_stage0_ftmw(tmp_path: Path) -> str:
    """An .ftmw file with an empty stage0_fid_data group."""
    p = tmp_path / "bare.ftmw"
    with h5py.File(p, "w") as h5f:
        h5f.create_group("stage0_fid_data")
    return str(p)


def _make_no_stage0_ftmw(tmp_path: Path) -> str:
    """An .ftmw file without a stage0_fid_data group."""
    p = tmp_path / "no_stage0.ftmw"
    with h5py.File(p, "w") as h5f:
        h5f.create_group("other_group")
    return str(p)


def _make_mat_file(
    tmp_path: Path,
    *,
    xinc: float = 1e-9,  # 1 GS/s — gives manageable sample counts
    yinc: float = 1.0,
    yorg: float = 0.0,
    channel: str = "Channel_1",
    pre_record_us: float = 2.0,
    frame_period_us: float = 5.0,
    n_frames: int = 3,
    tail_us: float = 1.0,
) -> Path:
    """Write a minimal synthetic Keysight MATLAB v7.3 .mat file.

    Sample count is derived from the layout so the loader can slice correctly.
    The default layout (pre=2 µs, 3×5 µs + 1 µs tail, 1 GS/s) uses
    2000 + 3*5000 + 1000 = 18000 samples.
    """
    n_samples = int(
        round((pre_record_us + n_frames * frame_period_us + tail_us) * 1e-6 / xinc)
    )
    rng = np.random.default_rng(7)
    data = rng.integers(-500, 500, size=n_samples, dtype=np.int16)
    mat_path = tmp_path / "test.mat"
    with h5py.File(mat_path, "w") as f:
        ch = f.create_group(channel)
        ch.create_dataset("Data", data=data.reshape(1, -1))
        ch.create_dataset("XInc", data=np.array([[xinc]]))
        ch.create_dataset("XOrg", data=np.array([[0.0]]))
        ch.create_dataset("YInc", data=np.array([[yinc]]))
        ch.create_dataset("YOrg", data=np.array([[yorg]]))
        frame_grp = f.create_group("Frame")
        model_arr = np.array([ord(c) for c in "TestScope\x00"], dtype=np.uint16)
        serial_arr = np.array([ord(c) for c in "SN00001\x00"], dtype=np.uint16)
        frame_grp.create_dataset("Model", data=model_arr.reshape(-1, 1))
        frame_grp.create_dataset("Serial", data=serial_arr.reshape(-1, 1))
    return mat_path


# ---------------------------------------------------------------------------
# 1. HDF5 round-trip tests
# ---------------------------------------------------------------------------


class TestChirpWindowHDF5RoundTrip:
    def test_absent_attr_returns_none(self, tmp_path: Path) -> None:
        p = _make_bare_stage0_ftmw(tmp_path)
        assert read_recommended_chirp_window(p) is None

    def test_write_none_reads_none(self, tmp_path: Path) -> None:
        p = _make_bare_stage0_ftmw(tmp_path)
        write_recommended_chirp_window(p, None)
        assert read_recommended_chirp_window(p) is None

    def test_no_stage0_group_write_noop(self, tmp_path: Path) -> None:
        p = _make_no_stage0_ftmw(tmp_path)
        cw = ChirpWindow(chirp_end_us=2.0, chirp_start_us=0.5)
        write_recommended_chirp_window(p, cw)
        assert read_recommended_chirp_window(p) is None

    def test_round_trip_all_fields(self, tmp_path: Path) -> None:
        p = _make_bare_stage0_ftmw(tmp_path)
        cw = ChirpWindow(chirp_end_us=1.6, chirp_start_us=0.6, start_margin_us=3.0)
        write_recommended_chirp_window(p, cw)
        result = read_recommended_chirp_window(p)
        assert result is not None
        assert result.chirp_end_us == pytest.approx(1.6)
        assert result.chirp_start_us == pytest.approx(0.6)
        assert result.start_margin_us == pytest.approx(3.0)

    def test_round_trip_no_start_us(self, tmp_path: Path) -> None:
        """chirp_start_us absent → reads back None."""
        p = _make_bare_stage0_ftmw(tmp_path)
        cw = ChirpWindow(chirp_end_us=2.68)
        write_recommended_chirp_window(p, cw)
        result = read_recommended_chirp_window(p)
        assert result is not None
        assert result.chirp_end_us == pytest.approx(2.68)
        assert result.chirp_start_us is None
        assert result.start_margin_us is None

    def test_round_trip_no_margin(self, tmp_path: Path) -> None:
        """start_margin_us absent → reads back None."""
        p = _make_bare_stage0_ftmw(tmp_path)
        cw = ChirpWindow(chirp_end_us=1.6, chirp_start_us=0.6)
        write_recommended_chirp_window(p, cw)
        result = read_recommended_chirp_window(p)
        assert result is not None
        assert result.start_margin_us is None

    def test_overwrite_replaces_previous(self, tmp_path: Path) -> None:
        p = _make_bare_stage0_ftmw(tmp_path)
        write_recommended_chirp_window(p, ChirpWindow(chirp_end_us=1.0))
        write_recommended_chirp_window(p, ChirpWindow(chirp_end_us=2.5))
        result = read_recommended_chirp_window(p)
        assert result is not None
        assert result.chirp_end_us == pytest.approx(2.5)

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert read_recommended_chirp_window(str(tmp_path / "nonexistent.ftmw")) is None


# ---------------------------------------------------------------------------
# 2. BlackChirpLoader chirp-window extraction against 2638 example data
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _DATA_2638.exists(),
    reason="examples/blackchirp_data/2638 not present",
)
class TestBlackchirpChirpWindowExtraction:
    """Extract timing from the real 2638 config files."""

    @pytest.fixture(scope="class")
    def extracted(self) -> dict:
        result = BlackChirpLoader._extract_chirp_window(_DATA_2638)
        assert result is not None, "_extract_chirp_window returned None for 2638"
        return result

    def test_chirp_end_us_correct(self, extracted: dict) -> None:
        # PreGate=0.5 µs + PreProtection=0.1 µs + DurationUs=1.0 µs = 1.6 µs
        assert extracted["chirp_end_us"] == pytest.approx(1.6)

    def test_chirp_start_us_correct(self, extracted: dict) -> None:
        # PreGate=0.5 + PreProtection=0.1 = 0.6 µs
        assert extracted["chirp_start_us"] == pytest.approx(0.6)

    def test_chirp_duration_sanity(self, extracted: dict) -> None:
        # Duration = end - start = 1.0 µs (matches chirps.csv DurationUs=1 for Chirp==0)
        duration = extracted["chirp_end_us"] - extracted["chirp_start_us"]
        assert duration == pytest.approx(1.0)

    def test_no_start_margin_us(self, extracted: dict) -> None:
        # Blackchirp does not supply a start margin; that is instrument-default.
        assert extracted.get("start_margin_us") is None


@pytest.mark.skipif(
    not _DATA_2638.exists(),
    reason="examples/blackchirp_data/2638 not present",
)
def test_blackchirp_load_fid_carries_chirp_window() -> None:
    """load_fid propagates the chirp_window into fid.metadata."""
    loader = BlackChirpLoader()
    fid = loader.load_fid(_DATA_2638)
    assert "chirp_window" in fid.metadata
    cw = fid.metadata["chirp_window"]
    assert isinstance(cw, dict)
    assert pytest.approx(cw["chirp_end_us"], abs=0.05) == 1.6
    assert pytest.approx(cw["chirp_start_us"], abs=0.05) == 0.6


# ---------------------------------------------------------------------------
# 3. KeysightMatLoader: chirp_window params attach and persist
# ---------------------------------------------------------------------------


def test_keysight_chirp_window_attaches_to_metadata(tmp_path: Path) -> None:
    """chirp_end_us / chirp_start_us / start_margin_us flow into fid.metadata."""
    mat = _make_mat_file(tmp_path)
    loader = KeysightMatLoader()
    fid = loader.load_fid(
        mat,
        pre_record_us=2.0,
        frame_period_us=5.0,
        n_frames=3,
        chirp_start_us=0.5,
        chirp_end_us=1.5,
        start_margin_us=0.5,
    )
    cw = fid.metadata.get("chirp_window")
    assert cw is not None
    assert cw["chirp_end_us"] == pytest.approx(1.5)
    assert cw["chirp_start_us"] == pytest.approx(0.5)
    assert cw["start_margin_us"] == pytest.approx(0.5)


def test_keysight_no_chirp_window_when_not_provided(tmp_path: Path) -> None:
    """No chirp_window in metadata when the params are omitted."""
    mat = _make_mat_file(tmp_path)
    loader = KeysightMatLoader()
    fid = loader.load_fid(
        mat,
        pre_record_us=2.0,
        frame_period_us=5.0,
        n_frames=3,
    )
    assert "chirp_window" not in fid.metadata


def test_keysight_chirp_window_persists_through_ftmw(tmp_path: Path) -> None:
    """chirp_window declared at import is readable via read_recommended_chirp_window."""
    import ftmwpipeline.api as ftmw

    mat = _make_mat_file(tmp_path)
    ftmw_path = str(tmp_path / "scope.ftmw")
    ftmw.import_data(
        ftmw_path,
        source=str(mat),
        pre_record_us=2.0,
        frame_period_us=5.0,
        n_frames=3,
        chirp_end_us=1.5,
        start_margin_us=0.5,
    )
    cw = read_recommended_chirp_window(ftmw_path)
    assert cw is not None
    assert cw.chirp_end_us == pytest.approx(1.5)
    assert cw.start_margin_us == pytest.approx(0.5)


def test_keysight_chirp_window_stamps_recommended_start(tmp_path: Path) -> None:
    """Import with chirp_end_us stamps recommended start_us = chirp_end + margin."""
    import ftmwpipeline.api as ftmw

    mat = _make_mat_file(tmp_path)
    ftmw_path = str(tmp_path / "scope.ftmw")
    ftmw.import_data(
        ftmw_path,
        source=str(mat),
        pre_record_us=2.0,
        frame_period_us=5.0,
        n_frames=3,
        chirp_end_us=1.5,
        start_margin_us=0.5,
    )
    with h5py.File(ftmw_path, "r") as h:
        stamped = float(h["stage0_fid_data/recommended_processing"].attrs["start_us"])
    # Expected: 1.5 + 0.5 = 2.0
    assert stamped == pytest.approx(2.0)


def test_keysight_chirp_window_default_margin(tmp_path: Path) -> None:
    """When start_margin_us is absent, the guard_margin_us default (0.67) is used."""
    import ftmwpipeline.api as ftmw
    from ftmwpipeline.core.start_detection_settings import StartDetectionSettings

    mat = _make_mat_file(tmp_path)
    ftmw_path = str(tmp_path / "scope_default.ftmw")
    ftmw.import_data(
        ftmw_path,
        source=str(mat),
        pre_record_us=2.0,
        frame_period_us=5.0,
        n_frames=3,
        chirp_end_us=1.5,
        # start_margin_us not given
    )
    with h5py.File(ftmw_path, "r") as h:
        stamped = float(h["stage0_fid_data/recommended_processing"].attrs["start_us"])
    expected = 1.5 + StartDetectionSettings().guard_margin_us
    assert stamped == pytest.approx(expected)


def test_keysight_only_chirp_end_no_start(tmp_path: Path) -> None:
    """chirp_start_us is optional; only chirp_end_us is required."""
    mat = _make_mat_file(tmp_path)
    loader = KeysightMatLoader()
    fid = loader.load_fid(
        mat,
        pre_record_us=2.0,
        frame_period_us=5.0,
        n_frames=3,
        chirp_end_us=1.5,
        # chirp_start_us not given
    )
    cw = fid.metadata.get("chirp_window")
    assert cw is not None
    assert cw["chirp_end_us"] == pytest.approx(1.5)
    assert cw.get("chirp_start_us") is None
