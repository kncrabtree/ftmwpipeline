"""Unit tests for the Stage 3 leakage-pedestal-subtracted SNR scoring.

In the Rician/leakage limit the magnitude spectrum sits on a coherent leakage
pedestal several sigma above the fluctuation floor. Scoring SNR as the raw
``|X| / sigma`` then floats *every* bin -- genuine line or pure pedestal noise --
above the promotion cutoff. :func:`_snap_to_active_grid` instead scores the
excess over the local pedestal, ``(|X| - pedestal) / sigma``; these tests pin
that behavior on a synthetic pedestal.
"""

import numpy as np
import pytest

from ftmwpipeline._internal.stage3_impl import _snap_to_active_grid
from ftmwpipeline.core.data_structures import ComplexFT, Peak


def _ft(freq, mag):
    spectrum = np.asarray(mag, dtype=float).astype(np.complex128)
    return ComplexFT.from_spectrum(spectrum, np.asarray(freq, dtype=float))


def test_pedestal_noise_not_promoted_real_line_is():
    # Two bins on a pedestal of 5 sigma: a real line towering above it, and a
    # pure-pedestal bin (no excess). sigma = 1.0 for easy arithmetic.
    freq = np.array([100.0, 101.0])
    sigma = np.array([1.0, 1.0])
    pedestal = np.array([5.0, 5.0])
    mag = np.array([5.3, 25.0])  # bin0: pedestal+0.3 (noise); bin1: real line
    snap_ft = _ft(freq, mag)

    internal = [
        Peak(frequency=100.0, intensity=5.3, snr=9.9, detection_pass="primary"),
        Peak(frequency=101.0, intensity=25.0, snr=30.0, detection_pass="primary"),
    ]
    out = _snap_to_active_grid(
        internal,
        snap_ft,
        sigma,
        pedestal,
        internal_min_snr=0.0,
        weak_medium_snr=10.0,
        medium_strong_snr=50.0,
        promotion_min_snr=3.0,
    )
    by_freq = {round(p.frequency): p for p in out}

    noise = by_freq[100]
    line = by_freq[101]
    # Excess SNR: (5.3-5)/1 = 0.3 -> not promoted; (25-5)/1 = 20 -> promoted.
    assert noise.snr == pytest.approx(0.3)
    assert noise.properties["promoted"] is False
    assert line.snr == pytest.approx(20.0)
    assert line.properties["promoted"] is True
    # Raw magnitude is preserved as the amplitude seed; pedestal is recorded.
    assert noise.intensity == 5.3
    assert noise.properties["leakage_pedestal"] == 5.0
    # Raw ratio (5.3) would have promoted the noise bin -- the bug being fixed.
    assert noise.intensity / noise.noise_std_local >= 3.0


def test_excess_clamped_at_zero():
    # A detection below its local pedestal scores SNR 0, never negative.
    freq = np.array([200.0])
    snap_ft = _ft(freq, np.array([4.0]))
    out = _snap_to_active_grid(
        [Peak(frequency=200.0, intensity=4.0, detection_pass="gap")],
        snap_ft,
        np.array([1.0]),
        np.array([6.0]),
        internal_min_snr=0.0,
        weak_medium_snr=10.0,
        medium_strong_snr=50.0,
        promotion_min_snr=3.0,
    )
    assert out[0].snr == 0.0
    assert out[0].properties["promoted"] is False


def test_no_pedestal_recovers_raw_snr():
    # With a zero pedestal the score is the plain |X|/sigma (clean-Rayleigh limit).
    freq = np.array([300.0, 301.0])
    snap_ft = _ft(freq, np.array([3.5, 12.0]))
    out = _snap_to_active_grid(
        [
            Peak(frequency=300.0, intensity=3.5, detection_pass="primary"),
            Peak(frequency=301.0, intensity=12.0, detection_pass="primary"),
        ],
        snap_ft,
        np.array([1.0, 1.0]),
        np.zeros(2),
        internal_min_snr=0.0,
        weak_medium_snr=10.0,
        medium_strong_snr=50.0,
        promotion_min_snr=3.0,
    )
    snrs = {round(p.frequency): p.snr for p in out}
    assert snrs[300] == 3.5
    assert snrs[301] == 12.0


def test_internal_floor_drops_subfloor_active_peaks():
    # A bump that clears the internal floor on its own grid but is pure pedestal
    # on the active grid (excess SNR 0.3) is dropped when the active-grid floor
    # is applied; one in the [floor, promotion) band is kept but not promoted;
    # a real line is kept and promoted.
    freq = np.array([100.0, 101.0, 102.0])
    snap_ft = _ft(freq, np.array([5.3, 7.5, 25.0]))  # +0.3, +2.5, +20 over pedestal
    pedestal = np.array([5.0, 5.0, 5.0])
    internal = [
        Peak(frequency=100.0, intensity=5.3, detection_pass="primary"),
        Peak(frequency=101.0, intensity=7.5, detection_pass="gap"),
        Peak(frequency=102.0, intensity=25.0, detection_pass="primary"),
    ]
    out = _snap_to_active_grid(
        internal,
        snap_ft,
        np.ones(3),
        pedestal,
        internal_min_snr=2.0,
        weak_medium_snr=10.0,
        medium_strong_snr=50.0,
        promotion_min_snr=3.0,
    )
    kept = {round(p.frequency): p for p in out}
    assert 100 not in kept  # excess 0.3 < internal floor 2.0 -> dropped
    assert 101 in kept  # excess 2.5 in [2.0, 3.0) -> kept, not promoted
    assert kept[101].properties["promoted"] is False
    assert 102 in kept  # excess 20 -> kept and promoted
    assert kept[102].properties["promoted"] is True
