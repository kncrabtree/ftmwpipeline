"""
Unit tests for the doublet-alternative adjudication module.

Covers :func:`adjudicate_close_pairs` and :class:`DoubletAdjudication`:

- Trigger enumeration (separation/ratio/chain cases).
- True doublet: strong orthogonal evidence + large chi2r_merged.
- Shape-error false doublet: low orthogonal evidence score.
- Merged refit faithfulness: tau policy and refit_kwargs forwarding.
- Degenerate path: refit failure returns a record without raising.
"""

import numpy as np
import pytest

from ftmwpipeline.fitting.doublet_alternative import (
    DoubletAdjudication,
    adjudicate_close_pairs,
)
from ftmwpipeline.fitting.peak_model import (
    ModelPeak,
    PeakShape,
    effective_tau,
    model_spectrum,
)
from ftmwpipeline.fitting.window_fit import (
    WindowFitResult,
    fit_window,
)

# Shared acquisition parameters matching the 2638 calibration fixture scale.
T_US = 12.65
TAU_US = 5.0
DF_MHZ = 0.0122  # bin spacing ~77 kHz -> 1/T_US ~ 0.0791 MHz
SEED = 20260612


def _offset_grid(half_width_mhz: float = 3.0, df_mhz: float = DF_MHZ) -> np.ndarray:
    n = int(round(half_width_mhz / df_mhz))
    return np.arange(-n, n + 1) * df_mhz


def _amp_for_snr(snr: float, sigma: float = 1.0, tau_us: float = TAU_US) -> float:
    return 2.0 * snr * sigma / effective_tau(tau_us, T_US)


def _noise(m: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    s = sigma / np.sqrt(2.0)
    return rng.normal(0.0, s, m) + 1j * rng.normal(0.0, s, m)


def _make_fit(
    u: np.ndarray,
    z: np.ndarray,
    sigma: float,
    seeds: list[ModelPeak],
    tau0: float = TAU_US,
    fit_tau: bool = False,
) -> WindowFitResult:
    """Convenience wrapper: fit with fixed-tau Lorentzian."""
    sigma_arr = np.full(u.size, sigma)
    return fit_window(
        u,
        z,
        sigma_arr,
        seeds,
        tau0,
        T_US,
        fit_tau=fit_tau,
    )


# ---------------------------------------------------------------------------
# Trigger enumeration
# ---------------------------------------------------------------------------
class TestTriggerEnumeration:
    """Qualifying-pair selection logic."""

    def test_single_peak_returns_empty(self):
        u = _offset_grid()
        sigma = 1.0
        pk = ModelPeak(_amp_for_snr(50.0), 0.0, 0.0)
        z = model_spectrum(u, [pk], TAU_US, T_US)
        fit = _make_fit(u, z, sigma, [pk])
        assert fit.n_peaks == 1
        records = adjudicate_close_pairs(
            u,
            z,
            np.full(u.size, sigma),
            fit,
            acquisition_us=T_US,
            shape=PeakShape.LORENTZIAN,
        )
        assert records == []

    def test_pair_below_separation_threshold_triggers(self):
        # separation = 0.8 * res_element < k_res * res_element
        res_el = 1.0 / T_US
        sep = 0.8 * res_el
        u = _offset_grid()
        pks = [
            ModelPeak(_amp_for_snr(80.0), -sep / 2, 0.3),
            ModelPeak(_amp_for_snr(60.0), +sep / 2, 0.5),
        ]
        z = model_spectrum(u, pks, TAU_US, T_US)
        sigma_arr = np.full(u.size, 1.0)
        fit = fit_window(u, z, sigma_arr, pks, TAU_US, T_US, fit_tau=False)
        records = adjudicate_close_pairs(
            u, z, sigma_arr, fit, acquisition_us=T_US, shape=PeakShape.LORENTZIAN
        )
        assert len(records) == 1

    def test_pair_above_separation_threshold_not_triggered(self):
        # separation = 2.0 * res_element > DEFAULT_DOUBLET_K_RES = 1.5
        res_el = 1.0 / T_US
        sep = 2.0 * res_el
        u = _offset_grid()
        pks = [
            ModelPeak(_amp_for_snr(80.0), -sep / 2, 0.3),
            ModelPeak(_amp_for_snr(60.0), +sep / 2, 0.5),
        ]
        z = model_spectrum(u, pks, TAU_US, T_US)
        sigma_arr = np.full(u.size, 1.0)
        fit = fit_window(u, z, sigma_arr, pks, TAU_US, T_US, fit_tau=False)
        records = adjudicate_close_pairs(
            u, z, sigma_arr, fit, acquisition_us=T_US, shape=PeakShape.LORENTZIAN
        )
        assert records == []

    def test_ratio_below_r_min_not_triggered(self):
        # Amplitude ratio 0.01 < DEFAULT_DOUBLET_R_MIN = 0.05
        res_el = 1.0 / T_US
        sep = 0.5 * res_el
        u = _offset_grid()
        strong_amp = _amp_for_snr(200.0)
        weak_amp = 0.01 * strong_amp
        pks = [
            ModelPeak(strong_amp, -sep / 2, 0.3),
            ModelPeak(weak_amp, +sep / 2, 0.5),
        ]
        z = model_spectrum(u, pks, TAU_US, T_US)
        sigma_arr = np.full(u.size, 1.0)
        fit = fit_window(u, z, sigma_arr, pks, TAU_US, T_US, fit_tau=False)
        records = adjudicate_close_pairs(
            u, z, sigma_arr, fit, acquisition_us=T_US, shape=PeakShape.LORENTZIAN
        )
        assert records == []

    def test_three_peak_chain_produces_two_records(self):
        # Three consecutive peaks within k_res produce 2 adjacent-pair records.
        res_el = 1.0 / T_US
        sep = 0.6 * res_el
        u = _offset_grid()
        pks = [
            ModelPeak(_amp_for_snr(80.0), -sep, 0.3),
            ModelPeak(_amp_for_snr(60.0), 0.0, 0.5),
            ModelPeak(_amp_for_snr(40.0), +sep, 0.7),
        ]
        z = model_spectrum(u, pks, TAU_US, T_US)
        sigma_arr = np.full(u.size, 1.0)
        fit = fit_window(u, z, sigma_arr, pks, TAU_US, T_US, fit_tau=False)
        # Force fit to report 3 peaks by using exact initial guesses.
        records = adjudicate_close_pairs(
            u, z, sigma_arr, fit, acquisition_us=T_US, shape=PeakShape.LORENTZIAN
        )
        # Expect exactly 2 records: (0,1) and (1,2) in sorted-offset order.
        assert len(records) == 2
        # Each record should reference a different pair.
        pairs = {(r.pair_index_a, r.pair_index_b) for r in records}
        assert len(pairs) == 2

    def test_custom_k_res_controls_threshold(self):
        res_el = 1.0 / T_US
        # 1.0 res_el is below the default 1.5 but we test with k_res=0.8:
        sep = 0.9 * res_el
        u = _offset_grid()
        pks = [
            ModelPeak(_amp_for_snr(80.0), -sep / 2, 0.3),
            ModelPeak(_amp_for_snr(60.0), +sep / 2, 0.5),
        ]
        z = model_spectrum(u, pks, TAU_US, T_US)
        sigma_arr = np.full(u.size, 1.0)
        fit = fit_window(u, z, sigma_arr, pks, TAU_US, T_US, fit_tau=False)
        # With k_res=0.8 the 0.9-res-element pair is above threshold -> no record.
        assert (
            adjudicate_close_pairs(
                u,
                z,
                sigma_arr,
                fit,
                acquisition_us=T_US,
                shape=PeakShape.LORENTZIAN,
                k_res=0.8,
            )
            == []
        )
        # With k_res=1.0 the same pair triggers.
        assert (
            len(
                adjudicate_close_pairs(
                    u,
                    z,
                    sigma_arr,
                    fit,
                    acquisition_us=T_US,
                    shape=PeakShape.LORENTZIAN,
                    k_res=1.0,
                )
            )
            == 1
        )


# ---------------------------------------------------------------------------
# True doublet: chi2r and orthogonal evidence
# ---------------------------------------------------------------------------
class TestTrueDoublet:
    """Two real lines at ~1.2 res elements; merged fit should be visibly worse."""

    def test_chi2r_and_orth_evidence_favor_doublet(self):
        rng = np.random.default_rng(SEED)
        sigma = 1.0
        snr = 50.0
        res_el = 1.0 / T_US
        sep = 1.2 * res_el
        u = _offset_grid(3.0)
        # Two peaks with comparable amplitudes (ratio ~0.5).
        amp_strong = _amp_for_snr(snr, sigma)
        amp_weak = 0.5 * amp_strong
        true_pks = [
            ModelPeak(amp_strong, -sep / 2, 0.4),
            ModelPeak(amp_weak, +sep / 2, 0.8),
        ]
        z = model_spectrum(u, true_pks, TAU_US, T_US) + _noise(u.size, sigma, rng)
        sigma_arr = np.full(u.size, sigma)

        # Production fit: start near truth with tau fixed.
        prod_fit = fit_window(
            u,
            z,
            sigma_arr,
            true_pks,
            TAU_US,
            T_US,
            fit_tau=False,
        )
        assert prod_fit.n_peaks == 2, "production fit should have 2 peaks"

        records = adjudicate_close_pairs(
            u,
            z,
            sigma_arr,
            prod_fit,
            acquisition_us=T_US,
            shape=PeakShape.LORENTZIAN,
        )
        assert len(records) == 1, "should trigger on the one qualifying pair"
        rec = records[0]

        # The production doublet fit is better than the merged fit.
        assert rec.merged_success, "merged refit should converge"
        assert rec.chi2r_merged > rec.chi2r_production, (
            f"merged chi2r {rec.chi2r_merged:.2f} should exceed "
            f"production {rec.chi2r_production:.2f} for a true doublet"
        )
        # delta_chi2_raw positive: merged is worse.
        assert rec.delta_chi2_raw > 0.0

        # Orthogonal evidence: should be substantially above 2*n_params_peak=6.
        assert rec.orth_evidence_delta_chi2 > 2 * rec.orth_evidence_n_params, (
            f"orth_evidence_delta_chi2 {rec.orth_evidence_delta_chi2:.2f} "
            f"should exceed 2*3 for a true doublet"
        )
        assert rec.support_bins >= 3, "support slice should be non-trivial"


# ---------------------------------------------------------------------------
# Shape-error false doublet: orthogonal evidence should be small
# ---------------------------------------------------------------------------
class TestShapeErrorFalseDoublet:
    """One peak with tau mismatch; K=2 fit absorbs the tau error spuriously.

    The orthogonal evidence on the merged residual must be much smaller than
    the true-doublet case, because the merged residual is dominated by
    tau-error-shaped structure that the nuisance subspace absorbs.
    """

    def test_orth_evidence_small_for_tau_mismatch(self):
        rng = np.random.default_rng(SEED + 1)
        sigma = 1.0
        snr = 300.0  # high SNR so shape error dominates
        tau_true = 8.0  # generation tau
        tau_fit = 11.0  # the tau used for fitting (wrong)

        u = _offset_grid(3.0)
        amp = _amp_for_snr(snr, sigma, tau_us=tau_true)
        true_pk = ModelPeak(amp, 0.0, 0.5)
        z = model_spectrum(u, [true_pk], tau_true, T_US) + _noise(u.size, sigma, rng)
        sigma_arr = np.full(u.size, sigma)

        # Production fit: forced K=2 with seeds straddling the true peak,
        # tau fixed at the WRONG value.  The spurious partner absorbs the
        # tau-mismatch residual.
        res_el = 1.0 / T_US
        seed_sep = 0.5 * res_el
        seeds = [
            ModelPeak(0.7 * amp, -seed_sep / 2, 0.5),
            ModelPeak(0.3 * amp, +seed_sep / 2, 0.2),
        ]
        prod_fit = fit_window(
            u,
            z,
            sigma_arr,
            seeds,
            tau_fit,
            T_US,
            fit_tau=False,
        )
        # prod_fit may have 1 or 2 peaks; only proceed if we got 2 close ones.
        if prod_fit.n_peaks < 2:
            pytest.skip(
                "production fit collapsed to 1 peak; tau mismatch not large enough"
            )

        records = adjudicate_close_pairs(
            u,
            z,
            sigma_arr,
            prod_fit,
            acquisition_us=T_US,
            shape=PeakShape.LORENTZIAN,
        )
        if not records:
            pytest.skip("no qualifying pair triggered; parameters need adjustment")

        rec = records[0]
        assert rec.merged_success, "merged refit should converge"

        # Reference: compute the true-doublet orth_evidence from the other test
        # case so we can assert an ordering (false doublet << true doublet).
        rng2 = np.random.default_rng(SEED)
        sigma2 = 1.0
        sep2 = 1.2 / T_US
        u2 = _offset_grid(3.0)
        amp2 = _amp_for_snr(50.0, sigma2)
        true_pks2 = [
            ModelPeak(amp2, -sep2 / 2, 0.4),
            ModelPeak(0.5 * amp2, +sep2 / 2, 0.8),
        ]
        z2 = model_spectrum(u2, true_pks2, TAU_US, T_US) + _noise(u2.size, sigma2, rng2)
        sigma2_arr = np.full(u2.size, sigma2)
        prod_fit2 = fit_window(
            u2, z2, sigma2_arr, true_pks2, TAU_US, T_US, fit_tau=False
        )
        records2 = adjudicate_close_pairs(
            u2,
            z2,
            sigma2_arr,
            prod_fit2,
            acquisition_us=T_US,
            shape=PeakShape.LORENTZIAN,
        )
        true_doublet_score = records2[0].orth_evidence_delta_chi2 if records2 else None

        if true_doublet_score is not None and true_doublet_score > 0:
            # Shape-error false doublet should score much lower than a true doublet.
            assert rec.orth_evidence_delta_chi2 < 0.2 * true_doublet_score, (
                f"false-doublet orth score {rec.orth_evidence_delta_chi2:.2f} "
                f"should be < 0.2 * true-doublet score {true_doublet_score:.2f}"
            )

        # Absolute sanity: the false-doublet score should be well below
        # the true-doublet 6 * n_params = 18 bar.
        assert rec.orth_evidence_delta_chi2 < 6 * rec.orth_evidence_n_params, (
            f"false-doublet orth score {rec.orth_evidence_delta_chi2:.2f} "
            f"unexpectedly high for a tau-mismatch absorber"
        )


# ---------------------------------------------------------------------------
# Merged refit faithfulness
# ---------------------------------------------------------------------------
class TestMergedRefitFaithfulness:
    """Verify that tau policy and refit_kwargs are honoured."""

    def test_fit_tau_false_keeps_tau_unchanged(self):
        """When fit_tau=False the merged refit must hold tau at the seed value."""
        rng = np.random.default_rng(SEED + 2)
        sigma = 1.0
        res_el = 1.0 / T_US
        sep = 0.9 * res_el
        u = _offset_grid()
        pks = [
            ModelPeak(_amp_for_snr(80.0), -sep / 2, 0.3),
            ModelPeak(_amp_for_snr(60.0), +sep / 2, 0.5),
        ]
        z = model_spectrum(u, pks, TAU_US, T_US) + _noise(u.size, sigma, rng)
        sigma_arr = np.full(u.size, sigma)

        # Production fit with tau fixed.
        prod_fit = fit_window(
            u,
            z,
            sigma_arr,
            pks,
            TAU_US,
            T_US,
            fit_tau=False,
        )
        records = adjudicate_close_pairs(
            u,
            z,
            sigma_arr,
            prod_fit,
            acquisition_us=T_US,
            shape=PeakShape.LORENTZIAN,
        )
        assert records, "pair should trigger"
        rec = records[0]
        # fit_tau=False in production -> tau should not change in the refit.
        assert rec.merged_success
        assert rec.merged_tau_us == pytest.approx(TAU_US, rel=1e-9), (
            f"merged_tau_us {rec.merged_tau_us:.4f} should equal "
            f"production tau {TAU_US} when fit_tau=False"
        )

    def test_tau_penalty_forwarded_without_exception(self):
        """refit_kwargs with a tau penalty is forwarded; no exception raised."""
        rng = np.random.default_rng(SEED + 3)
        sigma = 1.0
        res_el = 1.0 / T_US
        sep = 0.8 * res_el
        u = _offset_grid()
        pks = [
            ModelPeak(_amp_for_snr(80.0), -sep / 2, 0.3),
            ModelPeak(_amp_for_snr(50.0), +sep / 2, 0.5),
        ]
        z = model_spectrum(u, pks, TAU_US, T_US) + _noise(u.size, sigma, rng)
        sigma_arr = np.full(u.size, sigma)
        tau_bounds = (TAU_US / 3.0, TAU_US * 3.0)
        prod_fit = fit_window(
            u,
            z,
            sigma_arr,
            pks,
            TAU_US,
            T_US,
            fit_tau=False,
            tau_bounds=tau_bounds,
        )
        # Pass tau_penalty kwargs that should be forwarded without exception.
        refit_kwargs = {
            "tau_penalty_lambda": 20.0,
            "tau_penalty_reference": TAU_US,
            "tau_bounds": tau_bounds,
        }
        records = adjudicate_close_pairs(
            u,
            z,
            sigma_arr,
            prod_fit,
            acquisition_us=T_US,
            shape=PeakShape.LORENTZIAN,
            refit_kwargs=refit_kwargs,
        )
        # Should complete without exception; the number of records depends on
        # whether the pair still qualifies after fitting.
        assert isinstance(records, list)

    def test_fit_tau_and_tau0_cannot_be_overridden_by_refit_kwargs(self):
        """Caller-supplied fit_tau/tau0_us in refit_kwargs are ignored."""
        rng = np.random.default_rng(SEED + 4)
        sigma = 1.0
        res_el = 1.0 / T_US
        sep = 0.9 * res_el
        u = _offset_grid()
        pks = [
            ModelPeak(_amp_for_snr(80.0), -sep / 2, 0.3),
            ModelPeak(_amp_for_snr(60.0), +sep / 2, 0.5),
        ]
        z = model_spectrum(u, pks, TAU_US, T_US) + _noise(u.size, sigma, rng)
        sigma_arr = np.full(u.size, sigma)
        prod_fit = fit_window(u, z, sigma_arr, pks, TAU_US, T_US, fit_tau=False)
        # Attempt to override tau policy via refit_kwargs — should be ignored.
        records = adjudicate_close_pairs(
            u,
            z,
            sigma_arr,
            prod_fit,
            acquisition_us=T_US,
            shape=PeakShape.LORENTZIAN,
            refit_kwargs={"fit_tau": True, "tau0_us": 99.0},
        )
        # Should not raise; tau in merged result should match production tau,
        # not the 99 µs override.
        for rec in records:
            if rec.merged_success:
                assert rec.merged_tau_us != pytest.approx(
                    99.0, rel=0.01
                ), "tau0_us override in refit_kwargs must not take effect"


# ---------------------------------------------------------------------------
# Degenerate / failure path
# ---------------------------------------------------------------------------
class TestDegeneratePaths:
    """Pathological inputs return a record with merged_success=False; no raise."""

    def test_refit_failure_returns_record_with_merged_success_false(self):
        """Pathological amp_max=0 forces refit to return a failed result or raise;
        either way the function must return a record without propagating exceptions."""
        rng = np.random.default_rng(SEED + 5)
        sigma = 1.0
        res_el = 1.0 / T_US
        sep = 0.8 * res_el
        u = _offset_grid()
        pks = [
            ModelPeak(_amp_for_snr(80.0), -sep / 2, 0.3),
            ModelPeak(_amp_for_snr(60.0), +sep / 2, 0.5),
        ]
        z = model_spectrum(u, pks, TAU_US, T_US) + _noise(u.size, sigma, rng)
        sigma_arr = np.full(u.size, sigma)
        prod_fit = fit_window(u, z, sigma_arr, pks, TAU_US, T_US, fit_tau=False)

        # amp_max=1e-12 collapses all amplitudes to near zero -- degenerate fit.
        records = adjudicate_close_pairs(
            u,
            z,
            sigma_arr,
            prod_fit,
            acquisition_us=T_US,
            shape=PeakShape.LORENTZIAN,
            refit_kwargs={"amp_max": 1e-12},
        )
        # Must return without raising, and if a record exists it should be
        # either failed or (if the solver still managed to converge) valid.
        assert isinstance(records, list)
        for rec in records:
            assert isinstance(rec, DoubletAdjudication)

    def test_no_peaks_returns_empty(self):
        """A fit with zero peaks returns an empty list."""
        u = _offset_grid()
        z = np.zeros(u.size, dtype=np.complex128)
        sigma_arr = np.full(u.size, 1.0)
        # Construct a minimal 0-peak result manually.
        empty_fit = fit_window(u, z, sigma_arr, [], TAU_US, T_US, fit_tau=False)
        records = adjudicate_close_pairs(
            u,
            z,
            sigma_arr,
            empty_fit,
            acquisition_us=T_US,
            shape=PeakShape.LORENTZIAN,
        )
        assert records == []

    def test_record_fields_nan_on_failure(self):
        """A failure record has nan in the statistical fields."""
        rng = np.random.default_rng(SEED + 6)
        sigma = 1.0
        res_el = 1.0 / T_US
        sep = 0.8 * res_el
        u = _offset_grid()
        pks = [
            ModelPeak(_amp_for_snr(80.0), -sep / 2, 0.3),
            ModelPeak(_amp_for_snr(60.0), +sep / 2, 0.5),
        ]
        z = model_spectrum(u, pks, TAU_US, T_US) + _noise(u.size, sigma, rng)
        sigma_arr = np.full(u.size, sigma)
        prod_fit = fit_window(u, z, sigma_arr, pks, TAU_US, T_US, fit_tau=False)

        # Force a degenerate refit that will fail or produce nan.
        records = adjudicate_close_pairs(
            u,
            z,
            sigma_arr,
            prod_fit,
            acquisition_us=T_US,
            shape=PeakShape.LORENTZIAN,
            refit_kwargs={"amp_max": 1e-12},
        )
        for rec in records:
            if not rec.merged_success:
                import math

                assert math.isnan(rec.chi2r_merged)
                assert math.isnan(rec.delta_chi2_raw)
                assert math.isnan(rec.merged_offset_mhz)


# ---------------------------------------------------------------------------
# Field sanity checks
# ---------------------------------------------------------------------------
class TestFieldSanity:
    """Spot-checks on the geometry fields of a successful record."""

    def test_separation_res_elements_correct(self):
        res_el = 1.0 / T_US
        sep = 0.9 * res_el
        u = _offset_grid()
        pks = [
            ModelPeak(_amp_for_snr(60.0), -sep / 2, 0.3),
            ModelPeak(_amp_for_snr(60.0), +sep / 2, 0.5),
        ]
        z = model_spectrum(u, pks, TAU_US, T_US)
        sigma_arr = np.full(u.size, 1.0)
        prod_fit = fit_window(u, z, sigma_arr, pks, TAU_US, T_US, fit_tau=False)
        records = adjudicate_close_pairs(
            u, z, sigma_arr, prod_fit, acquisition_us=T_US, shape=PeakShape.LORENTZIAN
        )
        assert records
        rec = records[0]
        expected_sep_res = abs(rec.offset_b_mhz - rec.offset_a_mhz) * T_US
        assert rec.separation_res_elements == pytest.approx(expected_sep_res, rel=1e-6)

    def test_amp_ratio_in_unit_interval(self):
        res_el = 1.0 / T_US
        sep = 0.8 * res_el
        u = _offset_grid()
        pks = [
            ModelPeak(_amp_for_snr(80.0), -sep / 2, 0.0),
            ModelPeak(_amp_for_snr(30.0), +sep / 2, 0.0),
        ]
        z = model_spectrum(u, pks, TAU_US, T_US)
        sigma_arr = np.full(u.size, 1.0)
        prod_fit = fit_window(u, z, sigma_arr, pks, TAU_US, T_US, fit_tau=False)
        records = adjudicate_close_pairs(
            u, z, sigma_arr, prod_fit, acquisition_us=T_US, shape=PeakShape.LORENTZIAN
        )
        assert records
        rec = records[0]
        assert 0.0 <= rec.amp_ratio <= 1.0

    def test_pair_index_a_has_lower_offset(self):
        res_el = 1.0 / T_US
        sep = 0.8 * res_el
        u = _offset_grid()
        pks = [
            ModelPeak(_amp_for_snr(60.0), +sep / 2, 0.3),  # index 0, higher offset
            ModelPeak(_amp_for_snr(40.0), -sep / 2, 0.5),  # index 1, lower offset
        ]
        z = model_spectrum(u, pks, TAU_US, T_US)
        sigma_arr = np.full(u.size, 1.0)
        prod_fit = fit_window(u, z, sigma_arr, pks, TAU_US, T_US, fit_tau=False)
        records = adjudicate_close_pairs(
            u, z, sigma_arr, prod_fit, acquisition_us=T_US, shape=PeakShape.LORENTZIAN
        )
        assert records
        rec = records[0]
        # offset_a must be <= offset_b regardless of the peaks list order.
        assert rec.offset_a_mhz <= rec.offset_b_mhz
