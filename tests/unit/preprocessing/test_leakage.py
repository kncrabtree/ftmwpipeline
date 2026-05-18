"""
Unit tests for the O1 truncation-leakage reach estimator.

Validates the closed form against (a) the exact undamped-boxcar sinc identity,
(b) a synthetic FFT of a truncated cosine, and checks scaling, the damped/τ
behaviour and limits, vectorisation, and input validation.
"""

import numpy as np
import pytest
import scipy.signal as spsig

from ftmwpipeline.preprocessing.leakage import estimate_leakage_reach


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


class TestDampedBehaviour:
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
