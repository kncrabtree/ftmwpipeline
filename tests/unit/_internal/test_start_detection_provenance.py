"""Unit tests for :func:`resolve_start_provenance`.

Covers the five ``StartProvenance.source`` classifications -- "declared",
"auto_detected", "no_chirp_found", "manual", and "none" -- resolved purely
from the persisted record (the resolved Stage 1 ``ft_processing`` layer, the
Stage 0 ``recommended_processing`` layer, a chirp-window declaration, and a
start-detection provenance record), matching what the Stage 0 report and its
figure read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.start_detection_impl import (
    detect_start_time_impl,
    resolve_start_provenance,
)
from ftmwpipeline.core.data_structures import FID, ChirpWindow, Sideband
from ftmwpipeline.core.start_detection_settings import StartDetectionSettings
from ftmwpipeline.file_manager import SourceMetadata, create_pipeline_file
from ftmwpipeline.io.stage_fit_settings_serialization import (
    write_recommended_chirp_window,
)

_FAST = StartDetectionSettings(step_us=0.05, sweep_max_us=7.0)


def _make_fid(
    *,
    chirp_start_us: float = 0.5,
    chirp_dur_us: float = 1.0,
    chirp_amp: float = 100.0,
    duration_us: float = 20.0,
    dt_s: float = 4e-9,
    tone_amp: float = 5.0,
) -> FID:
    """Synthesize a simple chirp + decaying tone FID (chirp_amp=0 -> no chirp)."""
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
    p = str(tmp_path / filename)
    src = SourceMetadata(source_path=tmp_path, format_name="test")
    create_pipeline_file(p, fid, src, force=True)
    return p


class TestNone:
    def test_nothing_set_anywhere(self, tmp_path: Path) -> None:
        p = _create_ftmw(tmp_path, _make_fid())
        ftmw.compute_ft(p)
        prov = resolve_start_provenance(p)
        assert prov.source == "none"
        assert prov.start_us is None
        assert prov.chirp_end_us is None


class TestDeclared:
    def test_declaration_governs(self, tmp_path: Path) -> None:
        p = _create_ftmw(tmp_path, _make_fid(chirp_start_us=0.5, chirp_dur_us=1.0))
        write_recommended_chirp_window(
            p, ChirpWindow(chirp_end_us=1.5, chirp_start_us=0.5, start_margin_us=0.8)
        )
        detect_start_time_impl(p, settings=_FAST, stamp=True)
        ftmw.compute_ft(p)
        prov = resolve_start_provenance(p)
        assert prov.source == "declared"
        assert prov.chirp_end_us == pytest.approx(1.5)
        assert prov.chirp_start_us == pytest.approx(0.5)
        assert prov.guard_margin_us == pytest.approx(0.8)
        assert prov.start_us == pytest.approx(1.5 + 0.8)
        # The sweep still ran as a cross-check, so its settings are on file.
        assert prov.detection_settings is not None

    def test_declared_without_start_run(self, tmp_path: Path) -> None:
        """No start_detection record at all -- declaration alone is enough."""
        p = _create_ftmw(tmp_path, _make_fid())
        write_recommended_chirp_window(p, ChirpWindow(chirp_end_us=1.6))
        # No detect_start_time_impl call: only the import-time-style declaration.
        from ftmwpipeline.file_manager import update_processing_parameters

        update_processing_parameters(
            p, {"start_us": 1.6 + StartDetectionSettings().guard_margin_us}
        )
        ftmw.compute_ft(p)
        prov = resolve_start_provenance(p)
        assert prov.source == "declared"
        assert prov.detection_settings is None


class TestAutoDetected:
    def test_sweep_finds_chirp(self, tmp_path: Path) -> None:
        p = _create_ftmw(
            tmp_path, _make_fid(chirp_start_us=0.5, chirp_dur_us=1.0, chirp_amp=100.0)
        )
        detect_start_time_impl(p, settings=_FAST, stamp=True)
        ftmw.compute_ft(p)
        prov = resolve_start_provenance(p)
        assert prov.source == "auto_detected"
        assert prov.chirp_detected is True
        assert prov.chirp_end_us is not None
        assert prov.detection_settings == _FAST


class TestNoChirpFound:
    def test_sweep_ran_but_found_nothing(self, tmp_path: Path) -> None:
        # No chirp signal at all -- the plateau/floor ratio never clears the
        # detection threshold.
        p = _create_ftmw(tmp_path, _make_fid(chirp_amp=0.0))
        out = detect_start_time_impl(p, settings=_FAST, stamp=True)
        assert out["start_detection"].chirp_detected is False
        ftmw.compute_ft(p)
        prov = resolve_start_provenance(p)
        assert prov.source == "no_chirp_found"
        assert prov.chirp_end_us is None
        assert prov.detection_settings is not None
        assert prov.chirp_detected is False


class TestManual:
    def test_stage1_override_wins(self, tmp_path: Path) -> None:
        p = _create_ftmw(tmp_path, _make_fid(chirp_start_us=0.5, chirp_dur_us=1.0))
        write_recommended_chirp_window(p, ChirpWindow(chirp_end_us=1.5))
        detect_start_time_impl(p, settings=_FAST, stamp=True)
        ftmw.compute_ft(p)  # inherit the Stage 0 recommendation into ft_processing
        recommended = resolve_start_provenance(p).start_us
        assert recommended is not None
        manual_value = recommended + 5.0
        ftmw.compute_ft(p, start_us=manual_value)
        prov = resolve_start_provenance(p)
        assert prov.source == "manual"
        assert prov.start_us == pytest.approx(manual_value)
        assert prov.recommended_start_us == pytest.approx(recommended)
        # Stage 0 context (declared chirp end) is still carried for reference.
        assert prov.chirp_end_us == pytest.approx(1.5)

    def test_manual_override_with_no_stage0_recommendation(
        self, tmp_path: Path
    ) -> None:
        p = _create_ftmw(tmp_path, _make_fid())
        ftmw.compute_ft(p, start_us=3.0)
        prov = resolve_start_provenance(p)
        assert prov.source == "manual"
        assert prov.start_us == pytest.approx(3.0)
        assert prov.recommended_start_us is None
