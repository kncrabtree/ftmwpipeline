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
from ftmwpipeline.fitting.validation import feature_fwhm
from ftmwpipeline.fitting.window_fit import (
    DEFAULT_PHASE_PENALTY_CUTOFF_FWHM,
    ParameterErrors,
    WindowFitResult,
    _effective_min_pair_separation,
    _penalty_residuals_and_jacobian,
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
# Minimum-pair-separation floor (GitHub issue #13)
# ---------------------------------------------------------------------------
class TestEffectiveMinPairSeparation:
    """The resolution-referenced floor on the minimum allowed pair separation."""

    def test_fwhm_term_dominates_on_broad_features(self):
        # 0.5 * 0.30 = 0.15 MHz vs 1/T = 0.079 MHz -> FWHM term wins.
        sep = _effective_min_pair_separation(0.30, T_US, 0.5, 1.0)
        assert sep == pytest.approx(0.15)

    def test_resolution_term_dominates_on_narrow_features(self):
        # 0.5 * 0.116 = 0.058 MHz vs 1/T = 0.079 MHz -> resolution term wins.
        sep = _effective_min_pair_separation(0.116, T_US, 0.5, 1.0)
        assert sep == pytest.approx(1.0 / T_US)

    def test_resolution_factor_scales_the_floor(self):
        assert _effective_min_pair_separation(0.0, T_US, 0.5, 2.0) == pytest.approx(
            2.0 / T_US
        )

    def test_zero_resolution_factor_falls_back_to_fwhm(self):
        assert _effective_min_pair_separation(0.116, T_US, 0.5, 0.0) == pytest.approx(
            0.5 * 0.116
        )

    def test_nonpositive_acquisition_falls_back_to_fwhm(self):
        assert _effective_min_pair_separation(0.116, 0.0, 0.5, 1.0) == pytest.approx(
            0.5 * 0.116
        )


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


# ---------------------------------------------------------------------------
# Pair phase penalty (non-quadrature cos(phi_i - phi_j) form)
# ---------------------------------------------------------------------------
class TestPairPhasePenalty:
    """The pair penalty fires at both the in-phase degeneracy basin and the
    anti-phase cancellation basin, vanishes in quadrature, and the analytic
    Jacobian matches finite differences."""

    LAMBDA = 100.0
    FWHM = feature_fwhm(TAU_US, T_US)
    CUTOFF = DEFAULT_PHASE_PENALTY_CUTOFF_FWHM * FWHM

    def _packed(self, amp_phase_pairs):
        """Pack (amp, offset_mhz, phase) triples into the LSQ parameter
        vector layout."""
        params = []
        for amp, off, ph in amp_phase_pairs:
            params.extend([amp, off, ph])
        return np.array(params, dtype=float)

    def _residual(self, dphi: float, sep_mhz: float) -> float:
        params = self._packed([(1.0, 0.0, 0.0), (1.0, sep_mhz, dphi)])
        res, _ = _penalty_residuals_and_jacobian(
            params,
            k=2,
            tau0_us=TAU_US,
            fit_tau=False,
            phase_penalty_lambda=self.LAMBDA,
            amp_penalty_lambda=0.0,
            amp_floor=None,
            fwhm_mhz=self.FWHM,
            phase_penalty_cutoff_fwhm=DEFAULT_PHASE_PENALTY_CUTOFF_FWHM,
        )
        # One pair, no other penalties enabled -> one residual element.
        assert res.shape == (1,)
        return float(res[0])

    @pytest.mark.parametrize(
        "dphi,sign",
        [
            (0.0, +1.0),                # in-phase degeneracy
            (np.pi / 2.0, 0.0),          # quadrature
            (np.pi, -1.0),               # anti-phase cancellation
            (-np.pi / 2.0, 0.0),
        ],
    )
    def test_residual_at_zero_separation(self, dphi, sign):
        """At zero separation the residual is ``sqrt(lambda) * cos(dphi)``."""
        r = self._residual(dphi, sep_mhz=0.0)
        expected = np.sqrt(self.LAMBDA) * sign
        assert r == pytest.approx(expected, abs=1e-9)

    def test_residual_vanishes_beyond_cutoff(self):
        """The closeness weight is zero at or beyond ``cutoff``."""
        r_at = self._residual(0.0, sep_mhz=self.CUTOFF)
        r_beyond = self._residual(0.0, sep_mhz=2.0 * self.CUTOFF)
        assert r_at == pytest.approx(0.0, abs=1e-12)
        assert r_beyond == pytest.approx(0.0, abs=1e-12)

    def test_weight_is_linear_in_separation(self):
        """Half-way to the cutoff the residual is half of the zero-sep value."""
        r_zero = self._residual(0.0, sep_mhz=0.0)
        r_half = self._residual(0.0, sep_mhz=0.5 * self.CUTOFF)
        assert r_half == pytest.approx(0.5 * r_zero, abs=1e-9)

    def test_jacobian_matches_finite_difference(self):
        """The analytic penalty Jacobian matches central finite differences."""
        # Three peaks: one in-phase pair, one in-quadrature pair, one cancelling.
        params = self._packed(
            [
                (1.5, -0.3 * self.FWHM, 0.4),
                (1.2, +0.3 * self.FWHM, 0.4 + np.pi / 3.0),
                (0.8, +0.5 * self.FWHM, 0.4 + np.pi),
            ]
        )
        kwargs = dict(
            k=3,
            tau0_us=TAU_US,
            fit_tau=False,
            phase_penalty_lambda=self.LAMBDA,
            amp_penalty_lambda=0.0,
            amp_floor=None,
            fwhm_mhz=self.FWHM,
            phase_penalty_cutoff_fwhm=DEFAULT_PHASE_PENALTY_CUTOFF_FWHM,
        )
        _, jac = _penalty_residuals_and_jacobian(params, **kwargs)

        eps = 1e-6
        fd = np.zeros_like(jac)
        for idx in range(params.size):
            pp = params.copy(); pp[idx] += eps
            pm = params.copy(); pm[idx] -= eps
            rp, _ = _penalty_residuals_and_jacobian(pp, **kwargs)
            rm, _ = _penalty_residuals_and_jacobian(pm, **kwargs)
            fd[:, idx] = (rp - rm) / (2.0 * eps)

        # All three pairs have their full weight on this layout: separation
        # 0.6 / 0.8 / 0.2 FWHM, all < 2 FWHM cutoff.
        np.testing.assert_allclose(jac, fd, atol=5e-6)


# ---------------------------------------------------------------------------
# Bidirectional tau penalty (Gaussian prior centred on tau_maj, width sigma_tau)
# ---------------------------------------------------------------------------
class TestBidirectionalTauPenalty:
    """The Phase-3 tau-anchoring penalty pulls tau toward ``tau_maj`` from
    both sides at strength ``sqrt(lambda) / sigma_tau``. The legacy one-sided
    hinge form is preserved when ``tau_penalty_sigma_us`` is None.
    """

    LAMBDA = 500.0
    TAU_MAJ = 6.0
    SIGMA_TAU = 1.5

    def _packed_with_tau(self, tau_value: float) -> np.ndarray:
        # One peak (3 params) + tau (1 param) when fit_tau=True.
        return np.array([1.0, 0.0, 0.0, tau_value], dtype=float)

    def _penalty(self, tau_value: float, *, sigma_us=None):
        params = self._packed_with_tau(tau_value)
        res, _ = _penalty_residuals_and_jacobian(
            params,
            k=1,
            tau0_us=TAU_US,
            fit_tau=True,
            phase_penalty_lambda=0.0,
            amp_penalty_lambda=0.0,
            amp_floor=None,
            fwhm_mhz=None,
            phase_penalty_cutoff_fwhm=DEFAULT_PHASE_PENALTY_CUTOFF_FWHM,
            tau_penalty_lambda=self.LAMBDA,
            tau_penalty_reference=self.TAU_MAJ,
            tau_penalty_sigma_us=sigma_us,
        )
        # No other penalties enabled -> tau penalty is the only element.
        assert res.shape == (1,)
        return float(res[0])

    def test_bidirectional_residual_below_centre(self):
        # tau < tau_maj: residual = sqrt(L)*(tau - tau_maj)/sigma < 0.
        r = self._penalty(self.TAU_MAJ - self.SIGMA_TAU, sigma_us=self.SIGMA_TAU)
        expected = np.sqrt(self.LAMBDA) * (-1.0)
        assert r == pytest.approx(expected, abs=1e-9)

    def test_bidirectional_residual_above_centre(self):
        # tau > tau_maj: residual = sqrt(L)*(tau - tau_maj)/sigma > 0.
        r = self._penalty(self.TAU_MAJ + 0.5 * self.SIGMA_TAU, sigma_us=self.SIGMA_TAU)
        expected = np.sqrt(self.LAMBDA) * 0.5
        assert r == pytest.approx(expected, abs=1e-9)

    def test_bidirectional_vanishes_at_centre(self):
        r = self._penalty(self.TAU_MAJ, sigma_us=self.SIGMA_TAU)
        assert r == pytest.approx(0.0, abs=1e-12)

    def test_one_sided_hinge_above_centre(self):
        # sigma_us is None -> legacy one-sided behaviour: penalty is zero
        # when tau >= tau_ref.
        r = self._penalty(self.TAU_MAJ + 1.0, sigma_us=None)
        assert r == pytest.approx(0.0, abs=1e-12)

    def test_one_sided_hinge_below_centre(self):
        # sigma_us is None: positive residual proportional to
        # (tau_ref - tau) / tau_ref.
        tau_value = self.TAU_MAJ - 1.2
        r = self._penalty(tau_value, sigma_us=None)
        expected = np.sqrt(self.LAMBDA) * (self.TAU_MAJ - tau_value) / self.TAU_MAJ
        assert r == pytest.approx(expected, abs=1e-9)

    def test_bidirectional_jacobian_matches_finite_difference(self):
        params = self._packed_with_tau(self.TAU_MAJ + 0.4 * self.SIGMA_TAU)
        kwargs = dict(
            k=1,
            tau0_us=TAU_US,
            fit_tau=True,
            phase_penalty_lambda=0.0,
            amp_penalty_lambda=0.0,
            amp_floor=None,
            fwhm_mhz=None,
            phase_penalty_cutoff_fwhm=DEFAULT_PHASE_PENALTY_CUTOFF_FWHM,
            tau_penalty_lambda=self.LAMBDA,
            tau_penalty_reference=self.TAU_MAJ,
            tau_penalty_sigma_us=self.SIGMA_TAU,
        )
        _, jac = _penalty_residuals_and_jacobian(params, **kwargs)
        # Finite-difference check.
        eps = 1e-6
        fd = np.zeros_like(jac)
        for idx in range(params.size):
            pp = params.copy(); pp[idx] += eps
            pm = params.copy(); pm[idx] -= eps
            rp, _ = _penalty_residuals_and_jacobian(pp, **kwargs)
            rm, _ = _penalty_residuals_and_jacobian(pm, **kwargs)
            fd[:, idx] = (rp - rm) / (2.0 * eps)
        np.testing.assert_allclose(jac, fd, atol=1e-7)

    def test_one_sided_hinge_jacobian_matches_finite_difference(self):
        # Pick tau below the reference so the hinge is active.
        params = self._packed_with_tau(self.TAU_MAJ - 1.0)
        kwargs = dict(
            k=1,
            tau0_us=TAU_US,
            fit_tau=True,
            phase_penalty_lambda=0.0,
            amp_penalty_lambda=0.0,
            amp_floor=None,
            fwhm_mhz=None,
            phase_penalty_cutoff_fwhm=DEFAULT_PHASE_PENALTY_CUTOFF_FWHM,
            tau_penalty_lambda=self.LAMBDA,
            tau_penalty_reference=self.TAU_MAJ,
            tau_penalty_sigma_us=None,
        )
        _, jac = _penalty_residuals_and_jacobian(params, **kwargs)
        eps = 1e-6
        fd = np.zeros_like(jac)
        for idx in range(params.size):
            pp = params.copy(); pp[idx] += eps
            pm = params.copy(); pm[idx] -= eps
            rp, _ = _penalty_residuals_and_jacobian(pp, **kwargs)
            rm, _ = _penalty_residuals_and_jacobian(pm, **kwargs)
            fd[:, idx] = (rp - rm) / (2.0 * eps)
        np.testing.assert_allclose(jac, fd, atol=1e-7)


class TestDeriveWindowFitConstraintsCalibratedBounds:
    """When ``tau_maj_us`` + ``sigma_tau_us`` are supplied, ``derive_window_fit_constraints``
    builds a band of ``± N*sigma_tau`` around ``tau_maj`` intersected with
    ``[tau_maj/k, tau_maj*k]`` and switches the penalty to the bidirectional form.
    """

    def test_bounds_use_calibration(self):
        from ftmwpipeline.fitting.window_fit import derive_window_fit_constraints

        m = 41
        z = np.full(m, 0.5 + 0.0j)
        sigma = np.full(m, 0.01)
        c = derive_window_fit_constraints(
            z, sigma, tau0_us=3.0, acquisition_us=T_US,
            tau_maj_us=6.0, sigma_tau_us=0.5,
            tau_penalty_n_sigma=3.0, max_decay_factor=5.0,
        )
        # +- 3*0.5 = +- 1.5 around 6.0 -> (4.5, 7.5), well inside the
        # factor-5 cap (1.2, 30.0).
        assert c.tau_bounds == pytest.approx((4.5, 7.5))
        assert c.tau_penalty_reference == pytest.approx(6.0)
        assert c.tau_penalty_sigma_us == pytest.approx(0.5)
        assert c.fit_kwargs_inner["tau_penalty_sigma_us"] == pytest.approx(0.5)

    def test_legacy_apodization_path(self):
        from ftmwpipeline.fitting.window_fit import derive_window_fit_constraints

        m = 41
        z = np.full(m, 0.5 + 0.0j)
        sigma = np.full(m, 0.01)
        c = derive_window_fit_constraints(
            z, sigma, tau0_us=3.0, acquisition_us=T_US,
            tau_apodization_us=5.0, max_decay_factor=5.0,
        )
        # Upper bound = min(3*5, 5) = 5.
        assert c.tau_bounds[1] == pytest.approx(5.0)
        assert c.tau_penalty_reference == pytest.approx(5.0)
        # No sigma -> stays on the one-sided hinge form.
        assert c.tau_penalty_sigma_us is None


# ---------------------------------------------------------------------------
# Optional low-order complex baseline (leakage-wing nuisance term)
# ---------------------------------------------------------------------------
class TestComplexBaseline:
    """The ``baseline_order`` path on :func:`fit_window`."""

    def _wing(self, u, coeffs, u_s):
        """Evaluate B(u)=Σ_k (a_k+i b_k)(u/u_s)^k from a complex coeff vector."""
        x = u / u_s
        return sum(c * x**k for k, c in enumerate(coeffs))

    def test_baseline_absorbs_wing_spares_narrow_line(self):
        """A const complex wing under a narrow line is absorbed; the line is
        recovered and its sigma_A is only mildly inflated."""
        u = _offset_grid(2.0)
        u_s = float(np.max(np.abs(u)))
        line = ModelPeak(amplitude=6.0, offset_mhz=0.23, phase=0.7)
        true_coeffs = np.array([0.4 - 0.25j])  # const wing
        z = model_spectrum(u, [line], TAU_US, T_US) + self._wing(u, true_coeffs, u_s)
        sigma = np.full(u.size, 0.02)

        init = [ModelPeak(amplitude=5.0, offset_mhz=0.10, phase=0.0)]
        no_base = fit_window(u, z, sigma, init, TAU_US, T_US, fit_tau=False)
        with_base = fit_window(
            u, z, sigma, init, TAU_US, T_US, fit_tau=False, baseline_order=0
        )

        # Wing absorbed -> chi^2_r collapses toward 1.
        assert no_base.reduced_chi2 > 5.0
        assert with_base.reduced_chi2 < 1.5
        # Line recovered.
        assert with_base.peaks[0].amplitude == pytest.approx(6.0, abs=1e-3)
        assert with_base.peaks[0].offset_mhz == pytest.approx(0.23, abs=1e-3)
        # Coefficients recovered.
        assert with_base.baseline_order == 0
        assert with_base.baseline_coeffs == pytest.approx(true_coeffs, abs=1e-3)
        assert with_base.baseline_offset_scale == pytest.approx(u_s)
        # sigma_A from the joint covariance is finite and only mildly inflated.
        sa0 = no_base.peak_errors[0].amplitude
        sa1 = with_base.peak_errors[0].amplitude
        assert np.isfinite(sa1)
        assert 1.0 <= sa1 / sa0 < 1.3
        # n_params counts the 2 baseline coeffs.
        assert with_base.n_params == no_base.n_params + 2

    def test_clean_window_baseline_coeffs_near_zero(self):
        """On a clean window the baseline coefficients fit to ~0 and the line
        parameters are essentially unchanged."""
        u = _offset_grid(2.0)
        line = ModelPeak(amplitude=7.0, offset_mhz=-0.18, phase=-0.5)
        z = model_spectrum(u, [line], TAU_US, T_US)
        sigma = np.full(u.size, 0.02)
        init = [ModelPeak(amplitude=6.0, offset_mhz=0.0, phase=0.0)]

        with_base = fit_window(
            u, z, sigma, init, TAU_US, T_US, fit_tau=False, baseline_order=0
        )
        # Coeffs ~ 0 (a clean line carries no broad wing).
        assert np.max(np.abs(with_base.baseline_coeffs)) < 1e-3
        assert with_base.peaks[0].amplitude == pytest.approx(7.0, abs=1e-3)
        assert with_base.peaks[0].offset_mhz == pytest.approx(-0.18, abs=1e-3)

    def test_baseline_too_smooth_to_replace_a_narrow_line(self):
        """A low-order baseline alone cannot represent a narrow line: the
        best-fit baseline-only model leaves nearly all the line's energy in the
        residual. This is the load-bearing guardrail -- the baseline can soak
        up a broad wing but provably cannot mimic or absorb a real line."""
        from ftmwpipeline.fitting.window_fit import baseline_basis

        u = _offset_grid(2.0)
        u_s = float(np.max(np.abs(u)))
        line = ModelPeak(amplitude=8.0, offset_mhz=0.05, phase=0.3)
        z = model_spectrum(u, [line], TAU_US, T_US)

        # Least-squares fit of a linear complex baseline alone (design = the
        # basis columns; the complex coefficients are unconstrained).
        basis = baseline_basis(u, 1, u_s)  # (M, 2)
        coeffs, *_ = np.linalg.lstsq(basis, z, rcond=None)
        residual = z - basis @ coeffs
        captured = 1.0 - np.sum(np.abs(residual) ** 2) / np.sum(np.abs(z) ** 2)
        # The smooth baseline explains almost none of the narrow line's energy.
        assert captured < 0.2

    def test_baseline_jacobian_matches_finite_difference(self):
        """The analytic baseline Jacobian columns match central differences."""
        from ftmwpipeline.fitting.window_fit import baseline_basis

        u = _offset_grid(1.5)
        u_s = float(np.max(np.abs(u)))
        peaks = [ModelPeak(amplitude=5.0, offset_mhz=0.12, phase=0.4)]
        tau = TAU_US
        order = 1
        basis = baseline_basis(u, order, u_s)
        base_cols = np.concatenate([basis, 1j * basis], axis=1)
        analytic = np.concatenate(
            [model_jacobian(u, peaks, tau, T_US, include_tau=True), base_cols],
            axis=1,
        )

        def full_model(p):
            pk = ModelPeak(p[0], p[1], p[2])
            m = model_spectrum(u, [pk], p[3], T_US)
            a = p[4:4 + (order + 1)]
            b = p[4 + (order + 1):]
            return m + basis @ a + 1j * (basis @ b)

        p0 = np.array([5.0, 0.12, 0.4, tau, 0.3, -0.2, 0.05, 0.1])
        num = np.zeros_like(analytic)
        for j in range(p0.size):
            h = 1e-6 * max(abs(p0[j]), 1.0)
            pp = p0.copy(); pp[j] += h
            pm = p0.copy(); pm[j] -= h
            num[:, j] = (full_model(pp) - full_model(pm)) / (2 * h)
        assert np.max(np.abs(analytic - num)) < 1e-5

    def test_baseline_disabled_is_unchanged(self):
        """``baseline_order=None`` reproduces the no-baseline fit exactly."""
        u = _offset_grid(2.0)
        line = ModelPeak(amplitude=7.0, offset_mhz=0.2, phase=0.5)
        z = model_spectrum(u, [line], TAU_US, T_US)
        sigma = np.full(u.size, 0.02)
        init = [ModelPeak(6.0, 0.0, 0.0)]
        r = fit_window(u, z, sigma, init, TAU_US, T_US, fit_tau=False)
        assert r.baseline_order is None
        assert r.baseline_coeffs is None
        assert r.baseline_offset_scale is None
