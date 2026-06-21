"""
Unit tests for the truncation-leakage tooling.

Covers (a) the O1 reach estimator against the exact undamped-boxcar sinc
identity and a synthetic FFT, and (b) the de-ramp / leakage-touched-interval
helpers used by the Stage 3 gap mask and Stage 4 window assignment.
"""

import numpy as np
import pytest
import scipy.signal as spsig

from ftmwpipeline.preprocessing.edge_coherence import (
    DEFAULT_EDGE_M,
    above_threshold_intervals,
    rolling_coherence,
)
from ftmwpipeline.preprocessing.leakage import (
    deramp_to_active_start,
    estimate_leakage_reach,
    leakage_touched_intervals,
)


class TestUndampedClosedForm:
    def test_matches_boxcar_sinc_identity(self):
        """Undamped limit must equal peak_snr / (pi * T * min_snr)."""
        peak_snr, min_snr, T_us = 200.0, 5.0, 20.0
        T = T_us * 1e-6
        expected_mhz = peak_snr / (np.pi * T * min_snr) * 1e-6
        got = estimate_leakage_reach(
            peak_snr, acquisition_us=T_us, min_snr=min_snr, tau_us=None
        )
        assert got == pytest.approx(expected_mhz, rel=1e-12)


class TestSyntheticFFTValidation:
    def test_reach_predicts_last_sidelobe_above_floor(self):
        """Build a boxcar-truncated cosine; the predicted reach must match the
        offset of the furthest sidelobe peak that exceeds the gap-pass floor."""
        T_us = 20.0
        peak_snr = 200.0
        min_snr = 5.0

        dt = 1e-9  # 1 GS/s
        T = T_us * 1e-6
        n = int(round(T / dt))  # 20000 samples, hard truncation at T
        f0 = 5.0e6  # 5 MHz baseband tone
        t = np.arange(n) * dt
        sig = np.cos(2.0 * np.pi * f0 * t)

        n_pad = 2**20  # fine frequency grid
        spec = np.abs(np.fft.rfft(sig, n=n_pad))
        freqs = np.fft.rfftfreq(n_pad, dt)
        ratio = spec / spec.max()  # scale-independent (|S(Δf)|/|S(0)|)

        # Sidelobe peaks outside the main lobe (sinc zeros spaced 1/T).
        floor_frac = min_snr / peak_snr
        peak_idx = spsig.argrelmax(spec)[0]
        df = np.abs(freqs[peak_idx] - f0)
        outside = df > 5.0 / T
        above = ratio[peak_idx] >= floor_frac
        sel = outside & above
        assert sel.any(), "no qualifying sidelobe peaks found"
        measured_reach_mhz = df[sel].max() * 1e-6

        predicted = estimate_leakage_reach(
            peak_snr, acquisition_us=T_us, min_snr=min_snr, tau_us=None
        )
        # Envelope-level estimate vs discrete sidelobe peaks: ~20% is tight.
        assert predicted == pytest.approx(measured_reach_mhz, rel=0.2)


class TestScaling:
    def test_linear_in_peak_snr(self):
        a = estimate_leakage_reach(100.0, 10.0)
        b = estimate_leakage_reach(400.0, 10.0)
        assert b == pytest.approx(4.0 * a, rel=1e-12)

    def test_inverse_in_min_snr(self):
        a = estimate_leakage_reach(100.0, 10.0, min_snr=3.0)
        b = estimate_leakage_reach(100.0, 10.0, min_snr=6.0)
        assert b == pytest.approx(0.5 * a, rel=1e-12)

    def test_inverse_in_acquisition_for_undamped(self):
        a = estimate_leakage_reach(100.0, 10.0)
        b = estimate_leakage_reach(100.0, 20.0)
        assert b == pytest.approx(0.5 * a, rel=1e-12)


class TestDampedBehavior:
    def test_large_tau_approaches_undamped_limit(self):
        undamped = estimate_leakage_reach(100.0, 10.0, tau_us=None)
        big_tau = estimate_leakage_reach(100.0, 10.0, tau_us=1.0e6)
        assert big_tau == pytest.approx(undamped, rel=1e-3)

    def test_heavier_damping_widens_reach(self):
        """τ ≪ T (broad, short-coherence line) → wider conservative mask."""
        undamped = estimate_leakage_reach(100.0, 10.0, tau_us=None)
        tau_eq_T = estimate_leakage_reach(100.0, 10.0, tau_us=10.0)
        tau_short = estimate_leakage_reach(100.0, 10.0, tau_us=1.0)
        assert tau_eq_T > undamped
        assert tau_short > tau_eq_T

    def test_monotonic_decreasing_in_tau(self):
        taus = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 500.0]
        reaches = [estimate_leakage_reach(100.0, 10.0, tau_us=tu) for tu in taus]
        assert all(x > y for x, y in zip(reaches[:-1], reaches[1:]))


class TestVectorisation:
    def test_array_in_array_out(self):
        snrs = np.array([50.0, 100.0, 200.0])
        out = estimate_leakage_reach(snrs, 10.0, min_snr=3.0)
        assert isinstance(out, np.ndarray)
        assert out.shape == snrs.shape
        for s, r in zip(snrs, out):
            assert r == pytest.approx(
                estimate_leakage_reach(float(s), 10.0, min_snr=3.0)
            )

    def test_nonpositive_snr_yields_zero_reach(self):
        out = estimate_leakage_reach(np.array([-1.0, 0.0, 100.0]), 10.0)
        assert out[0] == 0.0
        assert out[1] == 0.0
        assert out[2] > 0.0
        assert estimate_leakage_reach(0.0, 10.0) == 0.0


class TestInputValidation:
    def test_nonpositive_acquisition_rejected(self):
        with pytest.raises(ValueError, match="acquisition_us"):
            estimate_leakage_reach(100.0, 0.0)

    def test_nonpositive_min_snr_rejected(self):
        with pytest.raises(ValueError, match="min_snr"):
            estimate_leakage_reach(100.0, 10.0, min_snr=0.0)

    def test_nonpositive_tau_rejected(self):
        with pytest.raises(ValueError, match="tau_us"):
            estimate_leakage_reach(100.0, 10.0, tau_us=-1.0)


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
