"""
Unit tests for the Stage 5 conservative add-one-peak loop.

Covers :mod:`ftmwpipeline.fitting.window_fit`'s :func:`conservative_fit`,
:func:`knockout_test`, and the blend-aware seeder: the F-test/AIC accept-reject
logic, recovery of a blended seed the plain sequential loop would miss, the
add-one-peak audit trail, and the per-line knockout validation.
"""

import numpy as np
import pytest

from ftmwpipeline.fitting.peak_model import ModelPeak, effective_tau, model_spectrum
from ftmwpipeline.fitting.validation import feature_fwhm
from ftmwpipeline.fitting.window_fit import (
    AddStep,
    conservative_fit,
    fit_window,
    knockout_test,
)

T_US = 12.65
TAU_US = 5.0
DF_MHZ = 0.0122
SEED = 20260523
FWHM = feature_fwhm(TAU_US, T_US)


def _offset_grid(half_width_mhz: float, df_mhz: float = DF_MHZ) -> np.ndarray:
    n = int(round(half_width_mhz / df_mhz))
    return np.arange(-n, n + 1) * df_mhz


def _amp_for_snr(snr: float, sigma: float = 1.0) -> float:
    return 2.0 * snr * sigma / effective_tau(TAU_US, T_US)


def _noise(m: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    s = sigma / np.sqrt(2.0)
    return rng.normal(0.0, s, m) + 1j * rng.normal(0.0, s, m)


def _window(peaks, half_width, sigma, rng):
    u = _offset_grid(half_width)
    z = model_spectrum(u, peaks, TAU_US, T_US) + _noise(u.size, sigma, rng)
    return u, z


# ---------------------------------------------------------------------------
# Accept / reject logic
# ---------------------------------------------------------------------------
class TestAcceptReject:
    def test_recovers_clean_multi_line_window(self):
        """Four well-separated lines are all accepted."""
        rng = np.random.default_rng(SEED)
        true = [
            ModelPeak(_amp_for_snr(150.0), -0.9, 0.3),
            ModelPeak(_amp_for_snr(60.0), -0.3, 2.1),
            ModelPeak(_amp_for_snr(100.0), 0.4, 5.0),
            ModelPeak(_amp_for_snr(40.0), 1.0, 1.2),
        ]
        u, z = _window(true, 2.0, 1.0, rng)
        candidates = [p.offset_mhz for p in true]

        res = conservative_fit(u, z, 1.0, candidates, TAU_US, T_US)

        assert res.success
        assert res.n_peaks == 4
        recovered = sorted(p.offset_mhz for p in res.peaks)
        for got, want in zip(recovered, sorted(p.offset_mhz for p in true)):
            assert abs(got - want) < 0.02

    def test_rejects_spurious_candidate(self):
        """A candidate at a noise-only location is not accepted."""
        rng = np.random.default_rng(SEED)
        true = [
            ModelPeak(_amp_for_snr(120.0), -0.5, 0.4),
            ModelPeak(_amp_for_snr(90.0), 0.6, 2.0),
        ]
        u, z = _window(true, 2.0, 1.0, rng)
        # The third candidate sits where there is no line.
        candidates = [-0.5, 0.6, 1.4]

        res = conservative_fit(u, z, 1.0, candidates, TAU_US, T_US)

        assert res.n_peaks == 2
        # The spurious candidate appears in the trail but was never accepted.
        assert any(
            abs(s.candidate_offset_mhz - 1.4) < 1e-9
            and s.decision in ("tentative", "reject")
            for s in res.audit_trail
        )

    def test_empty_candidate_list(self):
        """No candidates -> a graceful empty result."""
        rng = np.random.default_rng(SEED)
        u, z = _window([ModelPeak(_amp_for_snr(50.0), 0.0, 0.0)], 1.0, 1.0, rng)
        res = conservative_fit(u, z, 1.0, [], TAU_US, T_US)
        assert res.n_peaks == 0
        assert res.audit_trail == []
        assert res.knockouts == []

    def test_separation_constraint_drops_close_candidate(self):
        """A candidate closer than the FWHM-scaled minimum is rejected."""
        rng = np.random.default_rng(SEED)
        true = [ModelPeak(_amp_for_snr(150.0), 0.0, 0.5)]
        u, z = _window(true, 1.5, 1.0, rng)
        # A second candidate a hundredth of a FWHM from the real line.
        candidates = [0.0, 0.01 * FWHM]

        res = conservative_fit(
            u, z, 1.0, candidates, TAU_US, T_US, min_separation_factor=1.0
        )
        assert res.n_peaks == 1
        assert any(
            not s.separation_ok and s.decision == "reject" for s in res.audit_trail
        )


# ---------------------------------------------------------------------------
# The blend-aware seeder
# ---------------------------------------------------------------------------
class TestBlendAwareSeeder:
    def _blended_seed_window(self, rng):
        """A window whose single Stage-3 candidate is really a 2-line blend."""
        sep = 0.8 * FWHM
        true = [
            ModelPeak(_amp_for_snr(200.0), -sep / 2, 0.6),
            ModelPeak(_amp_for_snr(200.0), +sep / 2, 0.6 + np.pi),  # anti-phase
        ]
        u, z = _window(true, 1.3, 1.0, rng)
        return u, z, true

    def test_seeder_recovers_the_blended_pair(self):
        """An elevated single-cosine seed is re-seeded to K=2 and resolved."""
        rng = np.random.default_rng(SEED)
        u, z, true = self._blended_seed_window(rng)

        # One candidate only -- the blend centroid, as Stage 3 would detect it.
        res = conservative_fit(u, z, 1.0, [0.0], TAU_US, T_US)

        assert res.n_peaks == 2
        assert any(s.decision == "seed-blend" for s in res.audit_trail)
        recovered = sorted(p.offset_mhz for p in res.peaks)
        for got, want in zip(recovered, sorted(p.offset_mhz for p in true)):
            assert abs(got - want) < 0.02

    def test_disabled_seeder_returns_single_cosine(self):
        """With the seeder threshold lifted, the same blend fits as one line.

        This is the prototype's finding: the plain add-one-peak loop's
        sequential initialisation reports one line where the blend-aware
        seeder resolves two.
        """
        rng = np.random.default_rng(SEED)
        u, z, _ = self._blended_seed_window(rng)

        res = conservative_fit(
            u, z, 1.0, [0.0], TAU_US, T_US, seeder_rchi2_threshold=1.0e9
        )
        assert res.n_peaks == 1
        assert all(s.decision != "seed-blend" for s in res.audit_trail)


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------
class TestAuditTrail:
    def test_trail_records_each_decision(self):
        rng = np.random.default_rng(SEED)
        true = [
            ModelPeak(_amp_for_snr(150.0), -0.6, 0.3),
            ModelPeak(_amp_for_snr(80.0), 0.5, 2.0),
        ]
        u, z = _window(true, 2.0, 1.0, rng)
        res = conservative_fit(u, z, 1.0, [-0.6, 0.5], TAU_US, T_US)

        assert len(res.audit_trail) == res.n_iterations
        assert all(isinstance(s, AddStep) for s in res.audit_trail)
        # The first recorded decision is always the seed.
        assert res.audit_trail[0].decision == "seed"
        # An accepted candidate drops the chi-squared and the AIC.
        accepts = [s for s in res.audit_trail if s.decision == "accept"]
        assert accepts
        for step in accepts:
            assert step.chi2_after < step.chi2_before
            assert step.aic_after < step.aic_before
            assert step.p_value < 0.05


# ---------------------------------------------------------------------------
# Knockout validation
# ---------------------------------------------------------------------------
class TestKnockout:
    def test_real_lines_are_supported(self):
        """Every fitted line in a clean window survives the knockout test."""
        rng = np.random.default_rng(SEED)
        true = [
            ModelPeak(_amp_for_snr(150.0), -0.7, 0.3),
            ModelPeak(_amp_for_snr(90.0), 0.0, 2.1),
            ModelPeak(_amp_for_snr(120.0), 0.7, 5.0),
        ]
        u, z = _window(true, 2.0, 1.0, rng)
        res = conservative_fit(u, z, 1.0, [p.offset_mhz for p in true], TAU_US, T_US)

        assert res.n_peaks == 3
        assert len(res.knockouts) == 3
        for ko in res.knockouts:
            assert ko.supported
            # Removing a real line grows chi-squared by ~its own energy.
            assert ko.delta_chi2 > 0.0
            assert ko.delta_chi2 == pytest.approx(ko.expected_delta_chi2, rel=0.1)

    def test_knockout_test_on_a_direct_fit(self):
        """knockout_test works on a plain fit_window result."""
        rng = np.random.default_rng(SEED)
        true = [
            ModelPeak(_amp_for_snr(140.0), -0.4, 0.5),
            ModelPeak(_amp_for_snr(110.0), 0.5, 2.4),
        ]
        u, z = _window(true, 1.8, 1.0, rng)
        fit = fit_window(u, z, 1.0, true, TAU_US, T_US, fit_tau=True)

        knockouts = knockout_test(u, z, 1.0, fit, T_US)
        assert len(knockouts) == 2
        assert all(ko.supported for ko in knockouts)

    def test_no_peaks_gives_no_knockouts(self):
        rng = np.random.default_rng(SEED)
        u, z = _window([ModelPeak(_amp_for_snr(50.0), 0.0, 0.0)], 1.0, 1.0, rng)
        empty = fit_window(u, z, 1.0, [], TAU_US, T_US)
        assert knockout_test(u, z, 1.0, empty, T_US) == []
