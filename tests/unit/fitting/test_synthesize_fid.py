"""Unit tests for the time-domain model-FID synthesizer.

The synthesizer is the time-domain twin of ``model_spectrum``; the windowed
fit view depends on the two agreeing. Two gates:

1. Boxcar round-trip: ``dt * rfft(synthesize_fid(...))`` reproduces
   ``model_spectrum`` at every line (image-limited tolerance).
2. Windowing physics: exponentially apodizing the synthesized FID is exactly a
   ``tau -> 1/(1/tau + 1/W)`` substitution -- a pure time-domain identity, so it
   must hold to numerical precision.
"""

import numpy as np
import pytest

from ftmwpipeline.core.peak_shape import PeakShape
from ftmwpipeline.fitting.peak_model import (
    ModelPeak,
    effective_tau,
    model_spectrum,
    synthesize_fid,
)


def _grid(n=4096, dt_us=0.002):
    t = np.arange(n) * dt_us
    T = n * dt_us
    f_bb = np.fft.rfftfreq(n, d=dt_us)
    return t, T, dt_us, f_bb


class TestBoxcarRoundTrip:
    def test_synth_rfft_matches_model_spectrum(self):
        t, T, dt, f_bb = _grid()
        tau = 3.0
        # Peaks placed on rfft bins (center = 0, so offset == f_bb).
        peaks_bb = [(1.0, f_bb[300], 0.4), (0.6, f_bb[900], -1.1)]
        fid = synthesize_fid(t, peaks_bb, tau)
        spec = dt * np.fft.rfft(fid)

        model = model_spectrum(
            f_bb,
            [ModelPeak(amplitude=a, offset_mhz=f, phase=p) for a, f, p in peaks_bb],
            tau,
            T,
        )
        scale = np.abs(model).max()
        # Image of a real signal + finite-dt discretization; in-band tol.
        assert np.abs(spec - model).max() / scale < 5e-3

    def test_on_line_amplitude_and_phase(self):
        t, T, dt, f_bb = _grid()
        tau, A, phi, k = 4.0, 2.5, 0.9, 512
        fid = synthesize_fid(t, [(A, f_bb[k], phi)], tau)
        spec = dt * np.fft.rfft(fid)
        # On-line value is 0.5 * A * e^{i phi} * h_T(0) = 0.5 * A * tau_eff * e^{i phi}.
        expected = 0.5 * A * effective_tau(tau, T) * np.exp(1j * phi)
        assert spec[k] == pytest.approx(expected, rel=2e-3)


class TestWindowingPhysics:
    def test_exp_apodization_is_tau_substitution(self):
        t, T, dt, f_bb = _grid()
        tau, W = 5.0, 8.0
        peaks_bb = [(1.0, f_bb[400], 0.2), (0.8, f_bb[1100], 2.0)]
        windowed = synthesize_fid(t, peaks_bb, tau) * np.exp(-t / W)
        tau_eff = 1.0 / (1.0 / tau + 1.0 / W)
        direct = synthesize_fid(t, peaks_bb, tau_eff)
        # exp(-t/tau)*exp(-t/W) == exp(-t/tau_eff): exact identity.
        assert np.allclose(windowed, direct, rtol=0, atol=1e-12)


class TestGaussianEnvelope:
    def test_gaussian_envelope_shape(self):
        t, T, dt, f_bb = _grid()
        tau_g = 6.0
        fid = synthesize_fid(
            t, [(1.0, f_bb[200], 0.0)], tau_g, shape=PeakShape.GAUSSIAN
        )
        # At t=0 envelope is 1 -> sample equals cos(0) = 1.
        assert fid[0] == pytest.approx(1.0, abs=1e-9)
        # Envelope is exp(-(t/tau_g)^2): the |signal| peaks decay accordingly.
        analytic_env = np.exp(-((t / tau_g) ** 2))
        assert np.all(np.abs(fid) <= analytic_env + 1e-9)


def test_zero_tau_raises():
    t, _, _, f_bb = _grid()
    with pytest.raises(ValueError, match="tau_us must be positive"):
        synthesize_fid(t, [(1.0, f_bb[10], 0.0)], 0.0)
