"""
Unit tests for the truncation-leakage de-ramp helper.

Covers :func:`~ftmwpipeline.preprocessing.leakage.deramp_to_active_start`, the
display-path primitive that references a full-record-rfft spectrum to the
active-region turn-on (see ``leakage.py``'s module docstring for how this
relates to the Stage 3 / Stage 4 leakage statistic, which is computed
directly on the active-FT slice and needs no de-ramp).
"""

import numpy as np
import pytest

from ftmwpipeline.preprocessing.leakage import deramp_to_active_start


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
