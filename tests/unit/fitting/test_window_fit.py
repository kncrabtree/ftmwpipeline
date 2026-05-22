"""
Unit tests for the Stage 5 per-window least-squares core.

Covers :mod:`ftmwpipeline.fitting.window_fit`: recovery of synthetic lines
(noiseless and noisy), shared-tau free vs fixed handling, the analytic
``model_jacobian`` against finite differences, and the parameter covariance --
both that the sigma/sqrt(2) weighting yields reduced chi-squared ~ 1 (D-8) and
that the reported uncertainties match the trial-to-trial scatter.
"""

import dataclasses

import numpy as np
import pytest

from ftmwpipeline.fitting.peak_model import ModelPeak, effective_tau, model_spectrum
from ftmwpipeline.fitting.window_fit import (
    ParameterErrors,
    WindowFitResult,
    fit_window,
    model_jacobian,
)

T_US = 12.65
TAU_US = 5.0
DF_MHZ = 0.0122
SEED = 20260522


def _offset_grid(half_width_mhz: float = 2.0, df_mhz: float = DF_MHZ) -> np.ndarray:
    """Symmetric baseband-offset grid at the FT bin spacing."""
    n = int(round(half_width_mhz / df_mhz))
    return np.arange(-n, n + 1) * df_mhz


def _amp_for_snr(snr: float, tau_us: float, sigma: float) -> float:
    """Amplitude whose on-line response (A/2)*tau_eff gives the requested SNR."""
    return 2.0 * snr * sigma / effective_tau(tau_us, T_US)


def _noise(m: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Complex Gaussian noise with per-bin RMS sigma (E[|n|^2] = sigma^2)."""
    s = sigma / np.sqrt(2.0)
    return rng.normal(0.0, s, m) + 1j * rng.normal(0.0, s, m)


# ---------------------------------------------------------------------------
# Noiseless recovery
# ---------------------------------------------------------------------------
class TestNoiselessRecovery:
    def test_single_line_tau_fixed(self):
        """A clean single line is recovered to machine precision."""
        u = _offset_grid(2.0)
        true = ModelPeak(amplitude=8.0, offset_mhz=0.31, phase=1.1)
        z = model_spectrum(u, [true], TAU_US, T_US)
        init = ModelPeak(amplitude=6.0, offset_mhz=0.20, phase=0.4)

        res = fit_window(u, z, 1.0, [init], TAU_US, T_US, fit_tau=False)

        assert res.success
        assert res.n_peaks == 1
        assert res.peaks[0].amplitude == pytest.approx(true.amplitude, rel=1e-6)
        assert res.peaks[0].offset_mhz == pytest.approx(true.offset_mhz, abs=1e-7)
        assert res.peaks[0].phase == pytest.approx(true.phase, abs=1e-6)
        assert res.chi_squared < 1e-12  # the model reproduces clean data exactly

    def test_two_lines_tau_fixed(self):
        """A clean two-line blend is recovered when fit jointly."""
        u = _offset_grid(2.5)
        true = [
            ModelPeak(amplitude=10.0, offset_mhz=-0.40, phase=0.3),
            ModelPeak(amplitude=4.0, offset_mhz=0.55, phase=2.7),
        ]
        z = model_spectrum(u, true, TAU_US, T_US)
        init = [
            ModelPeak(amplitude=8.0, offset_mhz=-0.30, phase=0.0),
            ModelPeak(amplitude=5.0, offset_mhz=0.45, phase=2.0),
        ]

        res = fit_window(u, z, 1.0, init, TAU_US, T_US, fit_tau=False)

        assert res.success
        for fitted, want in zip(res.peaks, true):
            assert fitted.amplitude == pytest.approx(want.amplitude, rel=1e-5)
            assert fitted.offset_mhz == pytest.approx(want.offset_mhz, abs=1e-6)

    def test_tau_free_recovers_decay(self):
        """With tau free, the shared decay constant is recovered."""
        u = _offset_grid(2.0)
        true = ModelPeak(amplitude=8.0, offset_mhz=0.10, phase=0.5)
        true_tau = 6.4
        z = model_spectrum(u, [true], true_tau, T_US)
        init = ModelPeak(amplitude=6.0, offset_mhz=0.05, phase=0.2)

        res = fit_window(u, z, 1.0, [init], 5.0, T_US, fit_tau=True)

        assert res.success
        assert res.fit_tau
        assert res.tau_us == pytest.approx(true_tau, rel=1e-5)
        assert res.tau_error is not None

    def test_fitted_spectrum_and_residual_are_consistent(self):
        """fitted_spectrum is the model of the fitted peaks; residual is z - it."""
        u = _offset_grid(1.5)
        true = ModelPeak(amplitude=5.0, offset_mhz=0.0, phase=0.0)
        z = model_spectrum(u, [true], TAU_US, T_US)
        res = fit_window(u, z, 1.0, [true], TAU_US, T_US, fit_tau=False)

        expected = model_spectrum(u, res.peaks, res.tau_us, T_US)
        assert np.allclose(res.fitted_spectrum, expected)
        assert np.allclose(res.residual, z - res.fitted_spectrum)


# ---------------------------------------------------------------------------
# tau free vs fixed
# ---------------------------------------------------------------------------
class TestTauHandling:
    def test_fixed_tau_is_held_and_has_no_error(self):
        """fit_tau=False holds tau at the input and reports no tau error."""
        u = _offset_grid(2.0)
        true = ModelPeak(amplitude=8.0, offset_mhz=0.2, phase=1.0)
        z = model_spectrum(u, [true], TAU_US, T_US)
        res = fit_window(u, z, 1.0, [true], TAU_US, T_US, fit_tau=False)

        assert not res.fit_tau
        assert res.tau_us == TAU_US
        assert res.tau_error is None
        assert res.n_params == 3  # amplitude, offset, phase -- tau is not free

    def test_free_tau_adds_a_parameter(self):
        u = _offset_grid(2.0)
        true = ModelPeak(amplitude=8.0, offset_mhz=0.2, phase=1.0)
        z = model_spectrum(u, [true], TAU_US, T_US)
        res = fit_window(u, z, 1.0, [true], TAU_US, T_US, fit_tau=True)

        assert res.fit_tau
        assert res.n_params == 4  # the shared tau is the 4th parameter

    def test_tau_is_bounded_by_decay_factor(self):
        """A free tau cannot escape [tau0 / k, tau0 * k]."""
        u = _offset_grid(2.0)
        # Synthesize with a tau far above the bound; the fit must clip to it.
        true = ModelPeak(amplitude=8.0, offset_mhz=0.0, phase=0.0)
        z = model_spectrum(u, [true], 40.0, T_US)
        res = fit_window(
            u, z, 1.0, [true], 5.0, T_US, fit_tau=True, max_decay_factor=3.0
        )
        assert res.tau_us <= 5.0 * 3.0 + 1e-9


# ---------------------------------------------------------------------------
# model_jacobian
# ---------------------------------------------------------------------------
def _finite_difference_jacobian(
    u: np.ndarray,
    peaks: list[ModelPeak],
    tau_us: float,
    include_tau: bool,
) -> np.ndarray:
    """Central finite-difference Jacobian of model_spectrum for cross-check."""
    eps = 1e-6
    columns: list[np.ndarray] = []
    for i in range(len(peaks)):
        for attr in ("amplitude", "offset_mhz", "phase"):
            plus = list(peaks)
            minus = list(peaks)
            plus[i] = dataclasses.replace(
                peaks[i], **{attr: getattr(peaks[i], attr) + eps}
            )
            minus[i] = dataclasses.replace(
                peaks[i], **{attr: getattr(peaks[i], attr) - eps}
            )
            columns.append(
                (
                    model_spectrum(u, plus, tau_us, T_US)
                    - model_spectrum(u, minus, tau_us, T_US)
                )
                / (2.0 * eps)
            )
    if include_tau:
        columns.append(
            (
                model_spectrum(u, peaks, tau_us + eps, T_US)
                - model_spectrum(u, peaks, tau_us - eps, T_US)
            )
            / (2.0 * eps)
        )
    return np.column_stack(columns)


class TestModelJacobian:
    @pytest.mark.parametrize("include_tau", [False, True])
    def test_matches_finite_difference(self, include_tau):
        """The analytic model Jacobian matches central finite differences."""
        u = _offset_grid(2.0)
        peaks = [
            ModelPeak(amplitude=7.0, offset_mhz=-0.3, phase=0.6),
            ModelPeak(amplitude=3.0, offset_mhz=0.5, phase=2.1),
        ]
        analytic = model_jacobian(u, peaks, TAU_US, T_US, include_tau=include_tau)
        fd = _finite_difference_jacobian(u, peaks, TAU_US, include_tau)
        rel_err = np.abs(analytic - fd).max() / np.abs(analytic).max()
        assert rel_err < 1e-6

    def test_shape(self):
        u = _offset_grid(1.0)
        peaks = [ModelPeak(1.0, 0.0, 0.0), ModelPeak(1.0, 0.1, 0.0)]
        assert model_jacobian(u, peaks, TAU_US, T_US).shape == (u.size, 6)
        assert model_jacobian(u, peaks, TAU_US, T_US, include_tau=True).shape == (
            u.size,
            7,
        )


# ---------------------------------------------------------------------------
# Noise weighting and covariance
# ---------------------------------------------------------------------------
class TestNoiseWeightingAndCovariance:
    def test_reduced_chi2_is_near_one(self):
        """sigma/sqrt(2) weighting puts reduced chi-squared at ~1 (D-8)."""
        rng = np.random.default_rng(SEED)
        sigma = 1.0
        u = _offset_grid(2.5)
        true = ModelPeak(_amp_for_snr(80.0, TAU_US, sigma), 0.15, 0.9)
        reduced = []
        for _ in range(12):
            z = model_spectrum(u, [true], TAU_US, T_US) + _noise(u.size, sigma, rng)
            res = fit_window(u, z, sigma, [true], TAU_US, T_US, fit_tau=True)
            reduced.append(res.reduced_chi2)
        # Weighting by sigma (not sigma/sqrt(2)) would centre this at ~0.5.
        assert 0.9 < float(np.mean(reduced)) < 1.1

    def test_covariance_matches_trial_scatter(self):
        """Reported offset uncertainty matches the trial-to-trial scatter."""
        rng = np.random.default_rng(SEED)
        sigma = 1.0
        u = _offset_grid(2.5)
        true = ModelPeak(_amp_for_snr(40.0, TAU_US, sigma), 0.22, 1.3)

        recovered, reported = [], []
        for _ in range(80):
            z = model_spectrum(u, [true], TAU_US, T_US) + _noise(u.size, sigma, rng)
            res = fit_window(u, z, sigma, [true], TAU_US, T_US, fit_tau=False)
            assert res.success and res.covariance is not None
            recovered.append(res.peaks[0].offset_mhz)
            reported.append(res.peak_errors[0].offset_mhz)

        scatter = float(np.std(recovered, ddof=1))
        mean_reported = float(np.mean(reported))
        # The covariance prediction and the empirical scatter agree.
        assert 0.7 < scatter / mean_reported < 1.4
        # ... and the fit is unbiased.
        assert abs(float(np.mean(recovered)) - true.offset_mhz) < 0.5 * scatter

    def test_higher_snr_gives_smaller_uncertainty(self):
        """A stronger line is pinned down better."""
        rng = np.random.default_rng(SEED)
        sigma, u = 1.0, _offset_grid(2.0)
        errors = {}
        for snr in (20.0, 200.0):
            true = ModelPeak(_amp_for_snr(snr, TAU_US, sigma), 0.0, 0.0)
            z = model_spectrum(u, [true], TAU_US, T_US) + _noise(u.size, sigma, rng)
            res = fit_window(u, z, sigma, [true], TAU_US, T_US, fit_tau=False)
            errors[snr] = res.peak_errors[0].offset_mhz
        assert errors[200.0] < errors[20.0]

    def test_scalar_and_array_noise_agree(self):
        """A scalar rms_noise broadcasts to the same fit as a constant array."""
        u = _offset_grid(1.5)
        true = ModelPeak(6.0, 0.1, 0.5)
        z = model_spectrum(u, [true], TAU_US, T_US)
        scalar = fit_window(u, z, 0.5, [true], TAU_US, T_US, fit_tau=False)
        array = fit_window(
            u, z, np.full(u.size, 0.5), [true], TAU_US, T_US, fit_tau=False
        )
        assert scalar.chi_squared == pytest.approx(array.chi_squared)
        assert scalar.peak_errors[0].offset_mhz == pytest.approx(
            array.peak_errors[0].offset_mhz
        )


# ---------------------------------------------------------------------------
# Result statistics and the null model
# ---------------------------------------------------------------------------
class TestResultStatistics:
    def test_empty_peaks_returns_null_model(self):
        """No initial peaks -> a failed result carrying the data's chi-squared."""
        u = _offset_grid(1.0)
        z = model_spectrum(u, [ModelPeak(4.0, 0.0, 0.0)], TAU_US, T_US)
        res = fit_window(u, z, 1.0, [], TAU_US, T_US)

        assert not res.success
        assert res.n_peaks == 0
        assert res.n_params == 0
        assert np.all(res.fitted_spectrum == 0.0)
        assert np.allclose(res.residual, z)
        # chi-squared is the unit-variance norm of the un-modelled data.
        sig_ri = 1.0 / np.sqrt(2.0)
        expected = float(np.sum(np.abs(z / sig_ri) ** 2))
        assert res.chi_squared == pytest.approx(expected)

    def test_reduced_chi2_and_aic_formulas(self):
        """The reduced chi-squared and AIC properties use the documented forms."""
        res = WindowFitResult(
            success=True,
            peaks=[ModelPeak(1.0, 0.0, 0.0)],
            peak_errors=[ParameterErrors(0.1, 0.1, 0.1)],
            tau_us=TAU_US,
            tau_error=None,
            fit_tau=False,
            cost=200.0,
            chi_squared=400.0,
            n_data=800,
            n_params=3,
            n_function_evals=10,
            fitted_spectrum=np.zeros(400, dtype=complex),
            residual=np.zeros(400, dtype=complex),
        )
        assert res.reduced_chi2 == pytest.approx(400.0 / (800 - 3))
        assert res.aic == pytest.approx(2 * 3 + 800 * np.log(400.0 / 800))

    def test_n_function_evals_is_recorded(self):
        u = _offset_grid(1.5)
        true = ModelPeak(6.0, 0.1, 0.5)
        z = model_spectrum(u, [true], TAU_US, T_US)
        res = fit_window(u, z, 1.0, [true], TAU_US, T_US, fit_tau=False)
        assert res.n_function_evals > 0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
class TestInputValidation:
    def _good(self):
        u = _offset_grid(1.0)
        z = model_spectrum(u, [ModelPeak(4.0, 0.0, 0.0)], TAU_US, T_US)
        return u, z

    def test_non_1d_grid_rejected(self):
        u, z = self._good()
        with pytest.raises(ValueError):
            fit_window(u.reshape(1, -1), z, 1.0, [], TAU_US, T_US)

    def test_length_mismatch_rejected(self):
        u, z = self._good()
        with pytest.raises(ValueError):
            fit_window(u, z[:-1], 1.0, [], TAU_US, T_US)

    def test_mis_shaped_noise_rejected(self):
        u, z = self._good()
        with pytest.raises(ValueError):
            fit_window(u, z, np.ones(u.size - 1), [], TAU_US, T_US)

    def test_non_positive_noise_rejected(self):
        u, z = self._good()
        with pytest.raises(ValueError):
            fit_window(u, z, 0.0, [], TAU_US, T_US)

    @pytest.mark.parametrize("tau0,acq", [(0.0, T_US), (-1.0, T_US), (TAU_US, 0.0)])
    def test_non_positive_tau_or_acquisition_rejected(self, tau0, acq):
        u, z = self._good()
        with pytest.raises(ValueError):
            fit_window(u, z, 1.0, [], tau0, acq)

    def test_degenerate_tau_bounds_rejected(self):
        u, z = self._good()
        peak = ModelPeak(4.0, 0.0, 0.0)
        with pytest.raises(ValueError):
            fit_window(u, z, 1.0, [peak], TAU_US, T_US, tau_bounds=(6.0, 6.0))

    def test_invalid_decay_factor_rejected(self):
        u, z = self._good()
        peak = ModelPeak(4.0, 0.0, 0.0)
        with pytest.raises(ValueError):
            fit_window(u, z, 1.0, [peak], TAU_US, T_US, max_decay_factor=1.0)
