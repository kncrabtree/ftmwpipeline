"""
Unit tests for the truncation-leakage tooling.

Covers the de-ramp / leakage-touched-interval helpers used by the Stage 3 gap
mask and Stage 4 window assignment.
"""

import numpy as np
import pytest

from ftmwpipeline.preprocessing.edge_coherence import (
    DEFAULT_EDGE_M,
    above_threshold_intervals,
    rolling_coherence,
)
from ftmwpipeline.preprocessing.leakage import (
    deramp_to_active_start,
    leakage_touched_intervals,
)


def _delayed_rfft_pair(n=4096, length=800, shift=500, dt=1e-9, seed=1):
    """rfft of a waveform active from index 0 vs the same shifted to ``shift``.

    A non-circular shift inside a long zero-padded array, so the shift theorem
    holds exactly: ``Xs[k] = exp(-2j pi k shift/n) X0[k]``.
    """
    rng = np.random.default_rng(seed)
    wave = rng.normal(size=length)
    x0 = np.zeros(n)
    x0[:length] = wave
    xs = np.zeros(n)
    xs[shift : shift + length] = wave
    freqs_mhz = np.fft.rfftfreq(n, dt) / 1e6
    return freqs_mhz, np.fft.rfft(x0), np.fft.rfft(xs), shift * dt * 1e6


class TestDerampToActiveStart:
    def test_inverts_known_sample_delay(self):
        """probe=0 -> f_bb=|f| -> the de-ramp is the exact rfft de-delay."""
        freqs, x0, xs, start_us = _delayed_rfft_pair()
        out = deramp_to_active_start(freqs, xs, probe_freq_mhz=0.0, start_us=start_us)
        assert np.allclose(out, x0)

    def test_sideband_framings_recover_line_up_to_global_phase(self):
        """|f-probe| makes the de-ramp correct for both sideband framings."""
        freqs_bb, x0, xs, start_us = _delayed_rfft_pair()
        probe = 41000.0
        for real_freqs in (probe - freqs_bb, probe + freqs_bb):
            out = deramp_to_active_start(real_freqs, xs, probe, start_us)
            k = int(np.argmax(np.abs(x0)))
            const = out[k] / x0[k]
            assert abs(abs(const) - 1.0) < 1e-6
            assert np.allclose(out, x0 * const)

    def test_identity_at_zero_start(self):
        freqs, _, xs, _ = _delayed_rfft_pair()
        out = deramp_to_active_start(freqs, xs, probe_freq_mhz=1234.0, start_us=0.0)
        assert np.allclose(out, xs)

    def test_preserves_magnitude(self):
        freqs, _, xs, start_us = _delayed_rfft_pair()
        out = deramp_to_active_start(
            freqs, xs, probe_freq_mhz=1234.0, start_us=start_us
        )
        assert np.allclose(np.abs(out), np.abs(xs))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal shape"):
            deramp_to_active_start(np.zeros(10), np.zeros(8, complex), 0.0, 1.0)

    def test_non_1d_raises(self):
        with pytest.raises(ValueError, match="1-dimensional"):
            deramp_to_active_start(
                np.zeros((4, 4)), np.zeros((4, 4), complex), 0.0, 1.0
            )

    def test_negative_start_raises(self):
        with pytest.raises(ValueError, match="start_us"):
            deramp_to_active_start(np.zeros(8), np.zeros(8, complex), 0.0, -1.0)


def _truncated_line_spectrum(with_line=True, seed=7):
    """rfft of a damped cosine truncated to an active region starting at t0!=0.

    Returns the real-grid frequency array (MHz, probe=0 framing), the complex
    spectrum, a constant per-bin complex noise RMS, and ``start_us``.
    """
    n = 32768
    dt = 1e-9  # 1 GS/s
    s, e = 6000, 22000  # active region [s, e)
    sig = np.zeros(n)
    if with_line:
        f0_hz = 80e6
        t = np.arange(n) * dt
        ta = t[s:e] - t[s]
        tau = (e - s) * dt / 2.0
        sig[s:e] = 5.0 * np.cos(2 * np.pi * f0_hz * t[s:e]) * np.exp(-ta / tau)
    noise_sigma = 1.0
    sig = sig + np.random.default_rng(seed).normal(0.0, noise_sigma, n)
    spectrum = np.fft.rfft(sig)
    freqs_mhz = np.fft.rfftfreq(n, dt) / 1e6
    # rfft of N white-noise samples: per-bin complex RMS = sigma_t * sqrt(N).
    rms = np.full(spectrum.shape, noise_sigma * np.sqrt(n))
    return freqs_mhz, spectrum, rms, s * dt * 1e6


class TestLeakageTouchedIntervals:
    def test_fires_over_truncated_line(self):
        freqs, spec, rms, start_us = _truncated_line_spectrum(with_line=True)
        intervals = leakage_touched_intervals(freqs, spec, rms, 0.0, start_us)
        assert intervals, "expected leakage-touched intervals around the line"
        f0_idx = int(np.argmin(np.abs(freqs - 80.0)))
        assert any(lo <= f0_idx <= hi for lo, hi in intervals)

    def test_clean_noise_minimal_coverage(self):
        freqs, spec, rms, start_us = _truncated_line_spectrum(with_line=False)
        intervals = leakage_touched_intervals(freqs, spec, rms, 0.0, start_us)
        covered = sum(hi - lo + 1 for lo, hi in intervals)
        assert covered < 0.03 * len(freqs)

    def test_deramp_beats_canonical(self):
        """The de-ramped leakage map covers far more than the non-de-ramped one."""
        freqs, spec, rms, start_us = _truncated_line_spectrum(with_line=True)
        dr = leakage_touched_intervals(freqs, spec, rms, 0.0, start_us)
        canon = above_threshold_intervals(rolling_coherence(spec, rms, DEFAULT_EDGE_M))
        cov_dr = sum(hi - lo + 1 for lo, hi in dr)
        cov_canon = sum(hi - lo + 1 for lo, hi in canon)
        assert cov_dr > 3 * cov_canon

    def test_intervals_sorted_and_valid(self):
        freqs, spec, rms, start_us = _truncated_line_spectrum(with_line=True)
        intervals = leakage_touched_intervals(freqs, spec, rms, 0.0, start_us)
        for lo, hi in intervals:
            assert 0 <= lo <= hi < len(freqs)
        los = [lo for lo, _ in intervals]
        assert los == sorted(los)
