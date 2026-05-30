"""Unit tests for data-driven FID start-time detection.

Builds a synthetic FID with a broadband excitation chirp over a known window
followed by a slowly-decaying molecular tone, then checks that the detector
recovers the chirp end and the recommended start.
"""

from __future__ import annotations

import numpy as np
import pytest

from ftmwpipeline.core.data_structures import FID, Sideband
from ftmwpipeline.core.start_detection_settings import StartDetectionSettings
from ftmwpipeline.preprocessing.start_detection import detect_start_time

# Coarse sweep keeps the per-test FFT count small; the corner is still resolved.
_FAST = StartDetectionSettings(step_us=0.05)


def _make_fid(
    *,
    chirp_start_us: float = 1.0,
    chirp_dur_us: float = 1.0,
    duration_us: float = 20.0,
    dt_s: float = 4e-9,
    chirp_amp: float = 100.0,
    tone_amp: float = 5.0,
    tone_tau_us: float = 10.0,
    with_chirp: bool = True,
) -> FID:
    """Synthesize an FID: a strong broadband chirp + a decaying molecular tone."""
    n = int(round(duration_us * 1e-6 / dt_s))
    t = np.arange(n) * dt_s
    t_us = t * 1e6

    # Slowly-decaying molecular tone, present throughout (the post-chirp floor).
    tone = tone_amp * np.exp(-t_us / tone_tau_us) * np.cos(2 * np.pi * 80e6 * t)
    data = tone.copy()

    if with_chirp:
        in_chirp = (t_us >= chirp_start_us) & (t_us < chirp_start_us + chirp_dur_us)
        tc = t - chirp_start_us * 1e-6
        f0, f1 = 20e6, 110e6  # linear sweep, broadband
        rate = (f1 - f0) / (chirp_dur_us * 1e-6)
        phase = 2 * np.pi * (f0 * tc + 0.5 * rate * tc**2)
        data[in_chirp] += chirp_amp * np.cos(phase[in_chirp])

    return FID(data=data, spacing=dt_s, probe_freq_mhz=1000.0, sideband=Sideband.LOWER)


def test_detects_chirp_end_and_recommended_start():
    fid = _make_fid(chirp_start_us=1.0, chirp_dur_us=1.0)
    res = detect_start_time(fid, settings=_FAST)

    assert res.chirp_detected
    # Chirp spans [1, 2] us; the collapse lands at its trailing edge.
    assert 1.8 <= res.chirp_end_us <= 2.2
    # Primary recommendation is chirp_end + guard margin.
    assert res.start_us == pytest.approx(
        res.chirp_end_us + _FAST.guard_margin_us, abs=1e-6
    )
    # The chirp dominates: a large plateau/floor ratio.
    assert res.plateau / res.floor > 50


def test_chirp_end_tracks_chirp_duration():
    short = detect_start_time(_make_fid(chirp_dur_us=1.0), settings=_FAST)
    long = detect_start_time(_make_fid(chirp_dur_us=3.0), settings=_FAST)
    # A 2 us longer chirp pushes the corner ~2 us later.
    assert (long.chirp_end_us - short.chirp_end_us) == pytest.approx(2.0, abs=0.2)


def test_guard_margin_override():
    fid = _make_fid()
    res = detect_start_time(
        fid, settings=StartDetectionSettings(step_us=0.05, guard_margin_us=0.3)
    )
    assert res.start_us == pytest.approx(res.chirp_end_us + 0.3, abs=1e-6)


def test_no_chirp_is_flagged():
    fid = _make_fid(with_chirp=True, chirp_amp=0.0)  # tone only, no excitation
    res = detect_start_time(fid, settings=_FAST)
    assert not res.chirp_detected
    assert res.chirp_end_us == 0.0
    # Falls back to the bare guard margin.
    assert res.start_us == pytest.approx(_FAST.guard_margin_us, abs=1e-6)


def test_band_override_restricts_integration():
    fid = _make_fid()
    # Probe 1000 MHz, lower sideband -> molecular = 1000 - scope. The 80 MHz
    # tone sits at 920 MHz; restrict the band around it and detection still works.
    res = detect_start_time(fid, band=(880.0, 960.0), settings=_FAST)
    assert res.band_mhz == (880.0, 960.0)
    assert res.chirp_detected
    assert 1.8 <= res.chirp_end_us <= 2.2


def test_result_carries_sweep_arrays_for_viz():
    res = detect_start_time(_make_fid(), settings=_FAST)
    assert res.starts_us.shape == res.sum_magnitude.shape
    assert res.starts_us.size > 10
    assert np.all(np.isfinite(res.sum_magnitude))
