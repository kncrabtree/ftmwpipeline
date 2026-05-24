"""
Unit tests for the Stage 5 statistical-test and linewidth-physics helpers.

Covers :mod:`ftmwpipeline.fitting.validation`: the apodization / finite-T
linewidths, the noise-weighted chi-squared and RMS residual, the AIC, the
nested-model F-test, and the peak-separation constraint.
"""

import numpy as np
import pytest

from ftmwpipeline.fitting.peak_model import ModelPeak, model_spectrum
from ftmwpipeline.fitting.validation import (
    calculate_aic,
    calculate_aicc,
    calculate_chi_squared_improvement,
    calculate_hwhm_from_apodization,
    calculate_noise_weighted_chi2,
    calculate_rms_residuals,
    effective_sample_size,
    feature_fwhm,
    passes_significance_test,
    validate_peak_separation,
)

T_US = 12.65
TAU_US = 5.0


# ---------------------------------------------------------------------------
# Linewidth physics
# ---------------------------------------------------------------------------
class TestApodizationHWHM:
    def test_absorption_base_value(self):
        """Absorption HWHM is 1 / (2 pi tau)."""
        hwhm = calculate_hwhm_from_apodization(
            5.0, "absorption", include_natural_broadening=False
        )
        assert hwhm == pytest.approx(1.0 / (2.0 * np.pi * 5.0), rel=1e-9)

    def test_magnitude_factor(self):
        """A magnitude spectrum widens the absorption HWHM by sqrt(3)."""
        absorption = calculate_hwhm_from_apodization(
            5.0, "absorption", include_natural_broadening=False
        )
        magnitude = calculate_hwhm_from_apodization(
            5.0, "magnitude", include_natural_broadening=False
        )
        assert magnitude / absorption == pytest.approx(np.sqrt(3.0))

    def test_natural_broadening_factor(self):
        """Natural broadening adds a sqrt(2) factor in quadrature."""
        without = calculate_hwhm_from_apodization(
            5.0, "magnitude", include_natural_broadening=False
        )
        with_nat = calculate_hwhm_from_apodization(
            5.0, "magnitude", include_natural_broadening=True
        )
        assert with_nat / without == pytest.approx(np.sqrt(2.0))

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            calculate_hwhm_from_apodization(0.0)
        with pytest.raises(ValueError):
            calculate_hwhm_from_apodization(5.0, "phase")


class TestFeatureFWHM:
    def test_2638_scale(self):
        """|h_T| FWHM at the 2638 scale is ~122 kHz (report section 1)."""
        fwhm = feature_fwhm(TAU_US, T_US)
        assert 0.10 < fwhm < 0.14

    def test_narrows_with_longer_tau(self):
        """A longer decay constant gives a narrower line."""
        assert feature_fwhm(10.0, T_US) < feature_fwhm(TAU_US, T_US)
        assert feature_fwhm(TAU_US, T_US) < feature_fwhm(2.0, T_US)

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            feature_fwhm(0.0, T_US)
        with pytest.raises(ValueError):
            feature_fwhm(TAU_US, -1.0)


# ---------------------------------------------------------------------------
# Residual statistics
# ---------------------------------------------------------------------------
class TestResidualStatistics:
    def test_rms_zero_model_is_data_rms(self):
        z = np.array([3.0 + 4.0j])
        # Stacked [3, 4] -> RMS = sqrt((9 + 16) / 2).
        assert calculate_rms_residuals(z) == pytest.approx(np.sqrt(12.5))

    def test_rms_with_model_is_residual_rms(self):
        z = np.array([3.0 + 4.0j, 1.0 - 2.0j])
        assert calculate_rms_residuals(z, z) == pytest.approx(0.0)

    def test_noise_weighted_chi2_zero_model(self):
        """The zero-model chi-squared is the sigma/sqrt(2)-weighted data norm."""
        z = np.array([2.0 + 1.0j, -1.0 + 3.0j])
        sigma = 0.5
        chi2 = calculate_noise_weighted_chi2(z, sigma)
        sig_ri = sigma / np.sqrt(2.0)
        expected = np.sum((z.real / sig_ri) ** 2) + np.sum((z.imag / sig_ri) ** 2)
        assert chi2 == pytest.approx(expected)

    def test_noise_weighted_chi2_with_model(self):
        z = np.array([2.0 + 1.0j, -1.0 + 3.0j])
        model = np.array([1.5 + 1.0j, -1.0 + 2.0j])
        chi2 = calculate_noise_weighted_chi2(z, 1.0, model)
        sig_ri = 1.0 / np.sqrt(2.0)
        r = z - model
        expected = np.sum((r.real / sig_ri) ** 2) + np.sum((r.imag / sig_ri) ** 2)
        assert chi2 == pytest.approx(expected)

    def test_noise_weighted_chi2_rejects_bad_sigma(self):
        with pytest.raises(ValueError):
            calculate_noise_weighted_chi2(np.array([1.0 + 0j]), 0.0)


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------
class TestAIC:
    def test_formula(self):
        aic = calculate_aic(400.0, 3, 800)
        assert aic == pytest.approx(2 * 3 + 800 * np.log(400.0 / 800))

    def test_degenerate_chi2_is_inf(self):
        assert calculate_aic(0.0, 3, 800) == float("inf")
        assert calculate_aic(-1.0, 3, 800) == float("inf")


class TestEffectiveSampleSize:
    def test_flat_spectrum_returns_n_data(self):
        spec = np.ones(200, dtype=np.complex128)
        assert effective_sample_size(spec) == pytest.approx(200.0)

    def test_all_zero_returns_n_data(self):
        spec = np.zeros(200, dtype=np.complex128)
        assert effective_sample_size(spec) == 200.0

    def test_delta_returns_one(self):
        spec = np.zeros(200, dtype=np.complex128)
        spec[100] = 1.0
        assert effective_sample_size(spec) == pytest.approx(1.0)

    def test_lorentzian_returns_roughly_fwhm_in_bins(self):
        """Kish on |model|^2 for a Lorentzian collapses to ~FWHM in bins."""
        u = np.linspace(-10.0, 10.0, 4001)
        # half-width 1 in u-units; spacing du = 20/4000 = 0.005 -> FWHM = 2/du = 400 bins
        f = 1.0 / (1.0 + u * u)
        n_eff = effective_sample_size(f.astype(np.complex128))
        fwhm_bins = 2.0 / (20.0 / 4000.0)
        # Kish(|f|^2) for unit-half-width Lorentzian is 2pi/3 in width units
        # i.e. 2pi/3 * (fwhm/2). Numerical check just bounds the result.
        assert 0.5 * fwhm_bins <= n_eff <= 2.0 * fwhm_bins

    def test_never_exceeds_n_data(self):
        rng = np.random.default_rng(0)
        spec = rng.normal(size=137) + 1j * rng.normal(size=137)
        assert effective_sample_size(spec) <= 137.0

    def test_kish_mag_returns_larger_than_kish_mag_sq(self):
        u = np.linspace(-5.0, 5.0, 1001)
        f = (1.0 / (1.0 + u * u)).astype(np.complex128)
        n_eff_sq = effective_sample_size(f, kind="kish_mag_sq")
        n_eff_mag = effective_sample_size(f, kind="kish_mag")
        assert n_eff_mag > n_eff_sq

    def test_hard_radius_counts_bins_above_threshold(self):
        spec = np.zeros(100, dtype=np.complex128)
        spec[40:50] = 1.0  # 10 bins at full height
        spec[20:30] = 0.05  # below 0.1 cutoff by default
        n_eff = effective_sample_size(spec, kind="hard_radius")
        assert n_eff == 10.0

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError):
            effective_sample_size(np.ones(10), kind="nonsense")


class TestAICc:
    def test_equals_aic_form_when_n_eff_equals_n_data(self):
        """When n_eff == n_data, AICc differs from AIC only by the
        small-sample correction term -- a known offset that vanishes as
        n_eff -> infinity."""
        chi2, k, n_eff = 400.0, 3, 800
        aic = calculate_aic(chi2, k, int(n_eff))
        aicc = calculate_aicc(chi2, k, n_eff)
        correction = 2 * k * (k + 1) / (n_eff - k - 1)
        assert aicc == pytest.approx(aic + correction)

    def test_full_formula(self):
        chi2, k, n_eff = 400.0, 3, 50.0
        expected = (
            2 * k
            + n_eff * np.log(chi2 / n_eff)
            + 2 * k * (k + 1) / (n_eff - k - 1)
        )
        assert calculate_aicc(chi2, k, n_eff) == pytest.approx(expected)

    def test_returns_inf_when_n_eff_at_or_below_k_plus_one(self):
        # Model not identifiable on this effective sample size.
        assert calculate_aicc(100.0, 4, n_eff=5.0) == float("inf")
        assert calculate_aicc(100.0, 4, n_eff=4.0) == float("inf")

    def test_returns_inf_on_degenerate_inputs(self):
        assert calculate_aicc(0.0, 3, n_eff=100.0) == float("inf")
        assert calculate_aicc(100.0, 3, n_eff=0.0) == float("inf")

    def test_smaller_n_eff_penalises_complex_models_more(self):
        """A K=2 vs K=1 comparison should swing toward K=1 as n_eff shrinks."""
        chi2_k1, k1 = 200.0, 4   # 1 peak + tau
        chi2_k2, k2 = 195.0, 7   # 2 peaks + tau, marginal chi^2 improvement
        # Large n_eff: marginal improvement may favour K=2.
        large = 200.0
        d_large = calculate_aicc(chi2_k2, k2, large) - calculate_aicc(
            chi2_k1, k1, large
        )
        # Small n_eff: penalty dominates -> K=1 strongly preferred or
        # K=2 unidentifiable (+inf).
        small = 15.0
        d_small = calculate_aicc(chi2_k2, k2, small) - calculate_aicc(
            chi2_k1, k1, small
        )
        assert d_small > d_large


class TestFTest:
    def test_no_improvement_returns_unity_p(self):
        p, f, diff = calculate_chi_squared_improvement(1000.0, 1000.0, 3, 800, 10)
        assert p == 1.0
        assert f == 0.0
        assert diff == 0.0

    def test_zero_dof_change_returns_unity_p(self):
        p, f, _ = calculate_chi_squared_improvement(1000.0, 800.0, 0, 800, 10)
        assert p == 1.0
        assert f == 0.0

    def test_real_improvement_is_significant(self):
        """A large chi-squared drop yields a tiny p-value and a positive F."""
        p, f, diff = calculate_chi_squared_improvement(1000.0, 800.0, 3, 800, 10)
        assert diff == pytest.approx(200.0)
        assert f > 0.0
        assert p < 0.01

    def test_passes_significance_test(self):
        assert passes_significance_test(1000.0, 800.0, 3, 800, 10)
        assert not passes_significance_test(1000.0, 1000.0, 3, 800, 10)

    def test_chi2_matches_fit_window(self):
        """noise_weighted_chi2 reproduces a WindowFitResult's chi_squared."""
        from ftmwpipeline.fitting.window_fit import fit_window

        u = np.linspace(-2.0, 2.0, 200)
        true = ModelPeak(5.0, 0.1, 0.7)
        rng = np.random.default_rng(7)
        noise = rng.normal(0, 1 / np.sqrt(2), u.size) + 1j * rng.normal(
            0, 1 / np.sqrt(2), u.size
        )
        z = model_spectrum(u, [true], TAU_US, T_US) + noise
        res = fit_window(u, z, 1.0, [true], TAU_US, T_US, fit_tau=False)
        recomputed = calculate_noise_weighted_chi2(z, 1.0, res.fitted_spectrum)
        assert recomputed == pytest.approx(res.chi_squared, rel=1e-9)


# ---------------------------------------------------------------------------
# Peak separation
# ---------------------------------------------------------------------------
class TestPeakSeparation:
    def test_well_separated_peaks_are_valid(self):
        ok, pairs = validate_peak_separation(np.array([0.0, 1.0, 2.0]), 0.1)
        assert ok
        assert pairs == []

    def test_close_pair_is_flagged(self):
        ok, pairs = validate_peak_separation(np.array([0.0, 0.05, 1.0]), 0.1)
        assert not ok
        assert pairs == [(0, 1)]

    def test_multiple_close_pairs(self):
        ok, pairs = validate_peak_separation(np.array([0.0, 0.02, 0.04]), 0.1)
        assert not ok
        assert set(pairs) == {(0, 1), (0, 2), (1, 2)}
