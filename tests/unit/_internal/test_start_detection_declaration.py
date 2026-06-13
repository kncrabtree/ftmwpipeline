"""Unit tests for declaration-aware start-time detection.

Verifies that ``detect_start_time_impl`` uses a declared chirp window when
present, fires the cross-check warning when detector and declaration disagree,
and behaves identically to the no-declaration path when the declaration is
absent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pytest

from ftmwpipeline.core.data_structures import ChirpWindow, FID, Sideband
from ftmwpipeline.core.start_detection_settings import StartDetectionSettings
from ftmwpipeline.file_manager import SourceMetadata, create_pipeline_file
from ftmwpipeline.io.stage_fit_settings_serialization import (
    read_recommended_chirp_window,
    write_recommended_chirp_window,
)
from ftmwpipeline._internal.start_detection_impl import detect_start_time_impl

# Coarse detection settings for speed.
_FAST = StartDetectionSettings(step_us=0.05, sweep_max_us=7.0)


# ---------------------------------------------------------------------------
# Synthetic FID + bare .ftmw pipeline helpers
# ---------------------------------------------------------------------------


def _make_fid(
    *,
    chirp_start_us: float = 0.5,
    chirp_dur_us: float = 1.0,
    duration_us: float = 20.0,
    dt_s: float = 4e-9,
    chirp_amp: float = 100.0,
    tone_amp: float = 5.0,
) -> FID:
    """Synthesize a simple chirp + decaying tone FID."""
    n = int(round(duration_us * 1e-6 / dt_s))
    t = np.arange(n) * dt_s
    t_us = t * 1e6
    tone = tone_amp * np.exp(-t_us / 10.0) * np.cos(2 * np.pi * 80e6 * t)
    in_chirp = (t_us >= chirp_start_us) & (t_us < chirp_start_us + chirp_dur_us)
    tc = t - chirp_start_us * 1e-6
    phase = 2 * np.pi * (20e6 * tc + 0.5 * (90e6 / (chirp_dur_us * 1e-6)) * tc**2)
    chirp_sig = np.zeros(n)
    chirp_sig[in_chirp] = chirp_amp * np.cos(phase[in_chirp])
    return FID(
        data=tone + chirp_sig,
        spacing=dt_s,
        probe_freq_mhz=1000.0,
        sideband=Sideband.LOWER,
    )


def _create_ftmw(tmp_path: Path, fid: FID, filename: str = "test.ftmw") -> str:
    """Persist a FID to a minimal .ftmw pipeline file."""
    p = str(tmp_path / filename)
    src = SourceMetadata(source_path=tmp_path, format_name="test")
    create_pipeline_file(p, fid, src, force=True)
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDeclarationDrivesRecommendation:
    """When a chirp window is declared the recommendation follows the declaration."""

    def test_declaration_used_flag_true(self, tmp_path: Path) -> None:
        fid = _make_fid(chirp_start_us=0.5, chirp_dur_us=1.0)
        p = _create_ftmw(tmp_path, fid)
        write_recommended_chirp_window(p, ChirpWindow(chirp_end_us=1.5))
        out = detect_start_time_impl(p, settings=_FAST, stamp=False)
        assert out["declaration_used"] is True

    def test_start_us_is_declared_end_plus_margin(self, tmp_path: Path) -> None:
        fid = _make_fid(chirp_start_us=0.5, chirp_dur_us=1.0)
        p = _create_ftmw(tmp_path, fid)
        declared_end = 1.5
        write_recommended_chirp_window(p, ChirpWindow(chirp_end_us=declared_end))
        out = detect_start_time_impl(p, settings=_FAST, stamp=False)
        expected = declared_end + _FAST.guard_margin_us
        assert out["start_us"] == pytest.approx(expected)
        assert out["start_detection"].start_us == pytest.approx(expected)

    def test_declared_margin_overrides_settings_margin(self, tmp_path: Path) -> None:
        fid = _make_fid(chirp_start_us=0.5, chirp_dur_us=1.0)
        p = _create_ftmw(tmp_path, fid)
        write_recommended_chirp_window(
            p,
            ChirpWindow(chirp_end_us=1.5, start_margin_us=2.0),
        )
        out = detect_start_time_impl(p, settings=_FAST, stamp=False)
        # Declared margin 2.0 wins over settings.guard_margin_us (0.67).
        assert out["start_us"] == pytest.approx(1.5 + 2.0)

    def test_settings_margin_used_when_declaration_has_none(
        self, tmp_path: Path
    ) -> None:
        fid = _make_fid()
        p = _create_ftmw(tmp_path, fid)
        write_recommended_chirp_window(p, ChirpWindow(chirp_end_us=1.5))
        settings = StartDetectionSettings(step_us=0.05, guard_margin_us=0.9)
        out = detect_start_time_impl(p, settings=settings, stamp=False)
        assert out["start_us"] == pytest.approx(1.5 + 0.9)

    def test_chirp_end_fields_in_result(self, tmp_path: Path) -> None:
        fid = _make_fid()
        p = _create_ftmw(tmp_path, fid)
        write_recommended_chirp_window(p, ChirpWindow(chirp_end_us=1.6))
        out = detect_start_time_impl(p, settings=_FAST, stamp=False)
        assert out["chirp_end_declared_us"] == pytest.approx(1.6)
        assert out["chirp_end_detected_us"] is not None  # sweep ran


class TestCrossCheckWarning:
    """The sweep detector runs as a cross-check; disagreement fires a WARNING."""

    def test_disagree_warning_fired(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Chirp spans [0.5, 1.5] us; detector sees chirp_end ≈ 1.5 us.
        # Declare a very different chirp_end to force the warning.
        fid = _make_fid(chirp_start_us=0.5, chirp_dur_us=1.0)
        p = _create_ftmw(tmp_path, fid)
        # Declare chirp_end 5 us away from the detector value.
        write_recommended_chirp_window(p, ChirpWindow(chirp_end_us=6.5))
        with caplog.at_level(logging.WARNING, logger="ftmwpipeline"):
            detect_start_time_impl(p, settings=_FAST, stamp=False)
        assert any(
            "declared" in r.message.lower() for r in caplog.records
        ), "Expected a WARNING mentioning 'declared' but none found"

    def test_agree_no_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Detector finds chirp_end ≈ 1.5 us; declare the same value.
        fid = _make_fid(chirp_start_us=0.5, chirp_dur_us=1.0)
        p = _create_ftmw(tmp_path, fid)
        write_recommended_chirp_window(p, ChirpWindow(chirp_end_us=1.5))
        with caplog.at_level(logging.WARNING, logger="ftmwpipeline"):
            detect_start_time_impl(p, settings=_FAST, stamp=False)
        declaration_warnings = [
            r for r in caplog.records if "declared" in r.message.lower()
        ]
        assert len(declaration_warnings) == 0


class TestStampBehavior:
    """stamp=True writes the declaration-derived start to the file."""

    def test_stamps_declaration_derived_start(self, tmp_path: Path) -> None:
        fid = _make_fid()
        p = _create_ftmw(tmp_path, fid)
        write_recommended_chirp_window(p, ChirpWindow(chirp_end_us=1.5))
        out = detect_start_time_impl(p, settings=_FAST, stamp=True)
        assert out["stamped"] is True
        with h5py.File(p, "r") as h:
            stamped = float(
                h["stage0_fid_data/recommended_processing"].attrs["start_us"]
            )
        assert stamped == pytest.approx(out["start_us"])

    def test_no_stamp_does_not_write(self, tmp_path: Path) -> None:
        fid = _make_fid()
        p = _create_ftmw(tmp_path, fid)
        write_recommended_chirp_window(p, ChirpWindow(chirp_end_us=1.5))
        out = detect_start_time_impl(p, settings=_FAST, stamp=False)
        assert out["stamped"] is False
        # The recommended layer should still have its original value (None / __None__).
        with h5py.File(p, "r") as h:
            raw = h["stage0_fid_data/recommended_processing"].attrs.get("start_us")
        # After create_pipeline_file with a FID with processing.start_us=None,
        # the recommended attr is the __None__ sentinel or None.
        assert raw is None or raw == "__None__"


class TestNoDeclarationPathUnchanged:
    """Without a declaration the behaviour is byte-identical to the original."""

    def test_no_declaration_flag_false(self, tmp_path: Path) -> None:
        fid = _make_fid()
        p = _create_ftmw(tmp_path, fid)
        out = detect_start_time_impl(p, settings=_FAST, stamp=False)
        assert out["declaration_used"] is False

    def test_no_declaration_result_matches_direct_detector(
        self, tmp_path: Path
    ) -> None:
        from ftmwpipeline.preprocessing.start_detection import detect_start_time

        fid = _make_fid()
        p = _create_ftmw(tmp_path, fid)
        out = detect_start_time_impl(p, settings=_FAST, stamp=False)
        # Direct detector on the same FID.
        direct = detect_start_time(fid, settings=_FAST)
        assert out["start_us"] == pytest.approx(direct.start_us)
        assert out["declaration_used"] is False

    def test_no_declaration_chirp_end_declared_none(self, tmp_path: Path) -> None:
        fid = _make_fid()
        p = _create_ftmw(tmp_path, fid)
        out = detect_start_time_impl(p, settings=_FAST, stamp=False)
        assert out["chirp_end_declared_us"] is None
