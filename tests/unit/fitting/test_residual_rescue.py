"""
Unit tests for the residual-rescue B-loop
(:mod:`ftmwpipeline.fitting.residual_rescue`).

Covers :func:`rescue_and_consolidate`'s contract:

- A missed peak (real line whose offset the initial fit was never given) is
  detected on the residual, rescued, jointly refit with the initial peak,
  and survives knockout.
- A clean window (initial fit explains the data; residual is pure noise)
  terminates with zero accepted rounds and returns the initial fit
  unchanged.
- ``max_rescue_rounds=0`` is a hard short-circuit (returns initial verbatim,
  no rounds executed).
- The consolidated ``ConservativeFitResult`` preserves the initial fit's
  ``audit_trail`` (the rescue is a separate phase, not a continuation of
  the conservative add-one-peak loop).
- The per-round ``RescueRoundDiagnostics`` correctly tag rescue-origin
  pruning (the failsafe diagnostic for joint-refit pathology).
"""

import numpy as np
import pytest

from ftmwpipeline.fitting.peak_model import ModelPeak, effective_tau, model_spectrum
from ftmwpipeline.fitting.residual_rescue import (
    DEFAULT_RESCUE_MAX_ROUNDS,
    ConsolidatedRescueOutcome,
    attempt_residual_rescue,
    merge_close_peaks_cleanup,
    remove_and_refit_cleanup,
    rescue_and_consolidate,
)
from ftmwpipeline.fitting.validation import feature_fwhm
from ftmwpipeline.fitting.window_fit import (
    conservative_fit,
    derive_window_fit_constraints,
    fit_window,
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


class TestRescueAddsMissedPeak:
    """Initial fit handed only one of two real peaks; rescue recovers it."""

    def test_two_peak_window_with_one_candidate(self):
        rng = np.random.default_rng(SEED)
        # Two strong well-separated peaks. The initial fit is told about
        # only the first; the second creates a sharp residual feature the
        # rescue's detector should pick up.
        true = [
            ModelPeak(_amp_for_snr(200.0), -0.8, 0.5),
            ModelPeak(_amp_for_snr(150.0), +0.7, 1.4),
        ]
        u, z = _window(true, 1.6, 1.0, rng)
        sigma = np.full(u.size, 1.0)

        # Hold the seeder at K=1 (its residual re-seed would otherwise find
        # the withheld peak itself); the rescue path is what this exercises.
        initial = conservative_fit(u, z, sigma, [-0.8], TAU_US, T_US, seeder_max_k=1)
        assert initial.n_peaks == 1  # gets only the candidate we supplied
        assert initial.fit.reduced_chi2 > 2.0  # the missed peak makes chi^2 bad

        consolidated = rescue_and_consolidate(
            u,
            z,
            sigma,
            initial,
            TAU_US,
            T_US,
            max_rescue_rounds=DEFAULT_RESCUE_MAX_ROUNDS,
        )

        # The rescue may legitimately add more than the strict K=2 in
        # intermediate rounds (the joint refit's knockout sweep is what
        # prunes overshoots, not an a-priori cap). The contract we test is
        # weaker: both true peaks must be recovered within FWHM, and the
        # consolidated chi^2_r must be near 1.
        assert consolidated.fit.n_peaks >= 2
        got_offsets = [p.offset_mhz for p in consolidated.fit.peaks]
        for true_pk in true:
            assert any(abs(g - true_pk.offset_mhz) < 0.05 for g in got_offsets), (
                f"true peak at {true_pk.offset_mhz:+.3f} not recovered; "
                f"final peaks at {[round(g, 3) for g in got_offsets]}"
            )
        assert consolidated.fit.fit.reduced_chi2 < 1.5, (
            f"consolidated chi^2_r {consolidated.fit.fit.reduced_chi2:.3f} "
            f"should be near 1 once the missed peak is captured"
        )
        accepted = [r for r in consolidated.rounds if r.accepted]
        assert accepted, "expected at least one accepted rescue round"

    def test_preserves_initial_audit_trail(self):
        """The consolidated fit carries the initial fit's audit_trail
        verbatim; the rescue is a separate phase, not a loop continuation."""
        rng = np.random.default_rng(SEED)
        true = [
            ModelPeak(_amp_for_snr(180.0), -0.6, 0.3),
            ModelPeak(_amp_for_snr(140.0), +0.5, 2.1),
        ]
        u, z = _window(true, 1.4, 1.0, rng)
        sigma = np.full(u.size, 1.0)

        initial = conservative_fit(u, z, sigma, [-0.6], TAU_US, T_US)
        original_trail = list(initial.audit_trail)
        consolidated = rescue_and_consolidate(
            u,
            z,
            sigma,
            initial,
            TAU_US,
            T_US,
        )
        assert consolidated.fit.audit_trail is initial.audit_trail or (
            consolidated.fit.audit_trail == original_trail
        )


class TestRescueOnCleanWindow:
    """A correctly-fit window has noise-only residual; rescue accepts nothing."""

    def test_terminates_with_no_candidates(self):
        rng = np.random.default_rng(SEED)
        true = [
            ModelPeak(_amp_for_snr(150.0), -0.4, 0.7),
            ModelPeak(_amp_for_snr(120.0), +0.5, 2.4),
        ]
        u, z = _window(true, 1.3, 1.0, rng)
        sigma = np.full(u.size, 1.0)

        initial = conservative_fit(u, z, sigma, [-0.4, +0.5], TAU_US, T_US)
        assert initial.n_peaks == 2
        assert initial.fit.reduced_chi2 < 1.5

        consolidated = rescue_and_consolidate(
            u,
            z,
            sigma,
            initial,
            TAU_US,
            T_US,
        )
        # The chain terminates with no candidates (eventually): noise-only
        # residual on a final cleaner round.
        assert consolidated.terminated_reason == "no candidates"
        # Net peak count is unchanged. Any noise candidate the detector
        # nominates is dropped by the conservative-fit accept gate or the
        # iterative-cleanup sweep.
        assert consolidated.fit.n_peaks == 2
        got = sorted(p.offset_mhz for p in consolidated.fit.peaks)
        assert abs(got[0] - (-0.4)) < 0.05
        assert abs(got[1] - (+0.5)) < 0.05


class TestRescueShortCircuit:
    """``max_rescue_rounds=0`` short-circuits to a return of ``initial_fit``."""

    def test_max_rounds_zero_returns_initial(self):
        rng = np.random.default_rng(SEED)
        true = [
            ModelPeak(_amp_for_snr(180.0), -0.5, 0.3),
            ModelPeak(_amp_for_snr(140.0), +0.6, 2.2),
        ]
        u, z = _window(true, 1.5, 1.0, rng)
        sigma = np.full(u.size, 1.0)

        # Bad initial fit (missing peak), but max_rescue_rounds=0 means
        # no rescue runs and we get the initial back verbatim.
        initial = conservative_fit(u, z, sigma, [-0.5], TAU_US, T_US)
        consolidated = rescue_and_consolidate(
            u,
            z,
            sigma,
            initial,
            TAU_US,
            T_US,
            max_rescue_rounds=0,
        )
        assert consolidated.fit is initial
        assert consolidated.rounds == []
        assert "0" in consolidated.terminated_reason  # the no-op reason


class TestRescueDiagnosticsShape:
    """The ``RescueRoundDiagnostics`` records carry the failsafe counters."""

    def test_round_diagnostics_track_rescue_origin_pruning(self):
        rng = np.random.default_rng(SEED)
        true = [
            ModelPeak(_amp_for_snr(180.0), -0.6, 0.4),
            ModelPeak(_amp_for_snr(150.0), +0.7, 1.8),
        ]
        u, z = _window(true, 1.4, 1.0, rng)
        sigma = np.full(u.size, 1.0)

        initial = conservative_fit(u, z, sigma, [-0.6], TAU_US, T_US)
        consolidated = rescue_and_consolidate(
            u,
            z,
            sigma,
            initial,
            TAU_US,
            T_US,
        )
        assert isinstance(consolidated, ConsolidatedRescueOutcome)
        # Origin invariant: joint_fit.peaks = previous-round peaks +
        # this-round rescue peaks, in that order. The diagnostic records
        # the split so the failsafe (n_pruned_rescue_origin) can
        # attribute knockout pruning to the right origin.
        for diag in consolidated.rounds:
            if diag.joint_fit is None:
                continue
            assert (
                len(diag.joint_fit.peaks) == diag.n_initial_peaks + diag.n_rescue_added
            ), (
                f"origin invariant broken: round {diag.round_idx} has "
                f"{len(diag.joint_fit.peaks)} joint peaks but "
                f"n_initial={diag.n_initial_peaks} + n_rescue="
                f"{diag.n_rescue_added}"
            )
            # And pruned counts cannot exceed the joint peak count
            # (you can't prune more peaks than the joint fit contained).
            assert 0 <= diag.n_pruned_total <= len(diag.joint_fit.peaks)
            assert 0 <= diag.n_pruned_rescue_origin <= diag.n_pruned_total
            assert diag.n_pruned_rescue_origin <= diag.n_rescue_added


class TestMergeCleanupAICc:
    """AICc-with-n_eff merge gate behavior on the canonical duplicate-pair
    and real-close-pair regimes."""

    def _constraints_kwargs(self, u, z, sigma):
        c = derive_window_fit_constraints(z, sigma, TAU_US, T_US)
        return c.fit_kwargs_inner

    def test_duplicate_pair_collapses(self):
        """Two peaks at the same physical offset (split by half a bin) get
        merged: AICc(K-1) <= AICc(K) because n_eff ~ FWHM-in-bins makes the
        small-sample correction explode at K=2."""
        rng = np.random.default_rng(SEED + 1)
        true = ModelPeak(_amp_for_snr(120.0), 0.0, 0.5)
        u, z = _window([true], 0.8, 1.0, rng)
        sigma = np.full(u.size, 1.0)
        # Seed two peaks separated by half a bin around the true offset.
        duplicate_init = [
            ModelPeak(true.amplitude / 2, -0.5 * DF_MHZ, 0.5),
            ModelPeak(true.amplitude / 2, +0.5 * DF_MHZ, 0.5),
        ]
        fit_kwargs = self._constraints_kwargs(u, z, sigma)
        k2_fit = fit_window(
            u,
            z,
            sigma,
            duplicate_init,
            TAU_US,
            T_US,
            **fit_kwargs,
        )
        assert k2_fit.success and k2_fit.n_peaks == 2

        merged, n_merged = merge_close_peaks_cleanup(
            u,
            z,
            sigma,
            k2_fit,
            TAU_US,
            T_US,
            fit_kwargs_inner=fit_kwargs,
        )
        assert n_merged == 1
        assert merged.n_peaks == 1
        assert abs(merged.peaks[0].offset_mhz - 0.0) < 0.5 * FWHM

    def test_real_close_pair_survives(self):
        """Two genuinely distinct peaks ~0.7 FWHM apart should NOT merge:
        the K=2 model carries information the K=1 fit cannot reconstruct."""
        rng = np.random.default_rng(SEED + 2)
        sep = 0.7 * FWHM
        true = [
            ModelPeak(_amp_for_snr(220.0), -sep / 2, 0.5),
            ModelPeak(_amp_for_snr(220.0), +sep / 2, 2.0),  # different phase
        ]
        u, z = _window(true, 1.0, 1.0, rng)
        sigma = np.full(u.size, 1.0)
        fit_kwargs = self._constraints_kwargs(u, z, sigma)
        k2_fit = fit_window(
            u,
            z,
            sigma,
            true,
            TAU_US,
            T_US,
            **fit_kwargs,
        )
        assert k2_fit.success and k2_fit.n_peaks == 2

        merged, n_merged = merge_close_peaks_cleanup(
            u,
            z,
            sigma,
            k2_fit,
            TAU_US,
            T_US,
            fit_kwargs_inner=fit_kwargs,
        )
        assert n_merged == 0
        assert merged.n_peaks == 2

    def test_wide_pair_short_circuits(self):
        """Pairs farther apart than ``merge_separation_factor * FWHM``
        never reach the AICc gate."""
        rng = np.random.default_rng(SEED + 3)
        true = [
            ModelPeak(_amp_for_snr(150.0), -3.0 * FWHM, 0.5),
            ModelPeak(_amp_for_snr(150.0), +3.0 * FWHM, 1.5),
        ]
        u, z = _window(true, 8.0 * FWHM, 1.0, rng)
        sigma = np.full(u.size, 1.0)
        fit_kwargs = self._constraints_kwargs(u, z, sigma)
        k2_fit = fit_window(
            u,
            z,
            sigma,
            true,
            TAU_US,
            T_US,
            **fit_kwargs,
        )
        merged, n_merged = merge_close_peaks_cleanup(
            u,
            z,
            sigma,
            k2_fit,
            TAU_US,
            T_US,
            fit_kwargs_inner=fit_kwargs,
            merge_separation_factor=1.0,
        )
        assert n_merged == 0
        assert merged is k2_fit

    def test_above_resolution_real_pair_with_tied_aicc_survives(self):
        """A real pair at separation slightly above the structural cutoff
        but where AICc(K)=AICc(K-1)=+inf (model unidentifiable on the
        narrow n_eff) must NOT merge -- this is the w198-style
        outer-shoulder protection. REJECT-on-tie is the structural fix."""
        rng = np.random.default_rng(SEED + 5)
        # Separation = 0.7 FWHM: above 0.5 (no structural merge) and
        # below 1.0 (enters the AICc test). Two genuinely distinct peaks
        # with different phases at high SNR.
        sep = 0.7 * FWHM
        true = [
            ModelPeak(_amp_for_snr(250.0), -sep / 2, 0.4),
            ModelPeak(_amp_for_snr(250.0), +sep / 2, 2.3),
        ]
        u, z = _window(true, 0.8, 1.0, rng)
        sigma = np.full(u.size, 1.0)
        fit_kwargs = self._constraints_kwargs(u, z, sigma)
        k2_fit = fit_window(
            u,
            z,
            sigma,
            true,
            TAU_US,
            T_US,
            **fit_kwargs,
        )
        assert k2_fit.success and k2_fit.n_peaks == 2

        merged, n_merged = merge_close_peaks_cleanup(
            u,
            z,
            sigma,
            k2_fit,
            TAU_US,
            T_US,
            fit_kwargs_inner=fit_kwargs,
        )
        # Real pair must survive the gate even when AICc cannot
        # discriminate K=2 from K=1 on this narrow n_eff.
        assert n_merged == 0
        assert merged.n_peaks == 2

    def test_resolution_floor_merges_subresolution_pair(self, monkeypatch):
        """A pair separated by less than one active-FT resolution element
        (``1/T_active``) but more than ``structural_merge_factor * FWHM`` is
        merged unconditionally only because of the resolution-referenced floor
        (GitHub issue #13). With the floor disabled it survives the FWHM-only
        structural cutoff. The blend escape is disabled here: this test pins
        the floor mechanics, and the injected pair is real two-line structure
        the escape would (correctly) keep."""
        from ftmwpipeline.fitting import validation as _validation

        monkeypatch.setattr(_validation, "DEFAULT_PAIR_CANCELLATION_MAX", None)
        rng = np.random.default_rng(SEED + 11)
        # 0.5*FWHM = 0.0609 MHz < sep < 1/T_active = 0.0790 MHz: the band the
        # FWHM-only floor licenses but the resolution floor catches.
        sep = 0.070
        assert 0.5 * FWHM < sep < 1.0 / T_US
        true = [
            ModelPeak(_amp_for_snr(200.0), -sep / 2, 0.4),
            ModelPeak(_amp_for_snr(200.0), +sep / 2, 2.3),
        ]
        u, z = _window(true, 0.8, 1.0, rng)
        sigma = np.full(u.size, 1.0)
        fit_kwargs = self._constraints_kwargs(u, z, sigma)
        k2_fit = fit_window(u, z, sigma, true, TAU_US, T_US, **fit_kwargs)
        assert k2_fit.success and k2_fit.n_peaks == 2
        offs = sorted(p.offset_mhz for p in k2_fit.peaks)
        pair_sep = offs[1] - offs[0]
        assert 0.5 * FWHM < pair_sep < 1.0 / T_US

        # Floor disabled (k=0): FWHM-only structural cutoff leaves the pair.
        _, n_off = merge_close_peaks_cleanup(
            u,
            z,
            sigma,
            k2_fit,
            TAU_US,
            T_US,
            fit_kwargs_inner=fit_kwargs,
            min_pair_separation_resolution_factor=0.0,
        )
        assert n_off == 0

        # Floor on (k=1): the sub-resolution pair is collapsed.
        merged, n_on = merge_close_peaks_cleanup(
            u,
            z,
            sigma,
            k2_fit,
            TAU_US,
            T_US,
            fit_kwargs_inner=fit_kwargs,
            min_pair_separation_resolution_factor=1.0,
        )
        assert n_on == 1
        assert merged.n_peaks == 1

    def test_amp_ratio_tier_collapses_supraresolution_absorber(self, monkeypatch):
        """A pair just *above* the resolution floor (so tiers 1-2 leave it) with
        a large amplitude ratio is collapsed by the amplitude-ratio tier -- the
        weak member is a rescue-parked shape-error absorber, not a real doublet
        (GitHub issue #13, the w281/w143 class). A balanced pair in the same
        band survives. The blend escape is disabled here: this test pins the
        tier mechanics with a synthetic stand-in whose weak member is genuine
        injected structure the escape would (correctly) keep."""
        from ftmwpipeline.fitting import validation as _validation

        monkeypatch.setattr(_validation, "DEFAULT_PAIR_CANCELLATION_MAX", None)
        rng = np.random.default_rng(SEED + 21)
        sep = 0.10  # MHz ~ 1.27 resolution elements: in [1.0, 1.5] elem band.
        assert 1.0 / T_US < sep < 1.5 / T_US
        true = [
            ModelPeak(_amp_for_snr(150.0), -sep / 2, 0.4),
            ModelPeak(_amp_for_snr(15.0), +sep / 2, 0.4),  # 10:1 absorber
        ]
        u, z = _window(true, 0.8, 1.0, rng)
        sigma = np.full(u.size, 1.0)
        fit_kwargs = self._constraints_kwargs(u, z, sigma)
        k2_fit = fit_window(u, z, sigma, true, TAU_US, T_US, **fit_kwargs)
        assert k2_fit.success and k2_fit.n_peaks == 2

        # Tier disabled (threshold above the 10:1 ratio): supra-resolution pair
        # is preserved (tiers 1-2 do not reach it).
        _, n_off = merge_close_peaks_cleanup(
            u,
            z,
            sigma,
            k2_fit,
            TAU_US,
            T_US,
            fit_kwargs_inner=fit_kwargs,
            overfit_amp_ratio_threshold=100.0,
        )
        assert n_off == 0

        # Tier on (default threshold 6 < 10): the absorber is collapsed.
        merged, n_on = merge_close_peaks_cleanup(
            u,
            z,
            sigma,
            k2_fit,
            TAU_US,
            T_US,
            fit_kwargs_inner=fit_kwargs,
        )
        assert n_on == 1
        assert merged.n_peaks == 1

    def test_blend_escape_keeps_constructive_subresolution_pair(self):
        """A genuine sub-resolution two-line blend (balanced, constructive,
        collapse costs overwhelming raw chi-squared) survives the
        unconditional sub-resolution tier via the blend escape."""
        rng = np.random.default_rng(SEED + 31)
        sep = 0.070
        assert sep < 1.0 / T_US
        true = [
            ModelPeak(_amp_for_snr(200.0), -sep / 2, 0.4),
            ModelPeak(_amp_for_snr(200.0), +sep / 2, 0.9),
        ]
        u, z = _window(true, 0.8, 1.0, rng)
        sigma = np.full(u.size, 1.0)
        fit_kwargs = self._constraints_kwargs(u, z, sigma)
        k2_fit = fit_window(u, z, sigma, true, TAU_US, T_US, **fit_kwargs)
        assert k2_fit.success and k2_fit.n_peaks == 2
        merged, n_merged = merge_close_peaks_cleanup(
            u,
            z,
            sigma,
            k2_fit,
            TAU_US,
            T_US,
            fit_kwargs_inner=fit_kwargs,
            min_pair_separation_resolution_factor=1.0,
        )
        assert n_merged == 0
        assert merged.n_peaks == 2

    def test_blend_escape_decision_function(self):
        """The escape keeps only overwhelming-evidence constructive pairs:
        the cancelling pathology and weak-evidence pairs are rejected."""
        from ftmwpipeline.fitting.validation import (
            blend_pair_escape,
            pair_cancellation_fraction,
        )

        # Constructive pair, overwhelming evidence -> escape.
        assert blend_pair_escape(5000.0, 3, 1.0, 0.4, 0.9, 0.9)
        # Same pair, weak evidence (below 2*50*3) -> no escape.
        assert not blend_pair_escape(250.0, 3, 1.0, 0.4, 0.9, 0.9)
        # Near-anti-aligned (cancelling) pair -> no escape at any evidence.
        assert pair_cancellation_fraction(1.0, 0.0, 0.95, np.pi * 0.98) > 0.9
        assert not blend_pair_escape(1.0e6, 3, 1.0, 0.0, 0.95, np.pi * 0.98)
        # The fidelity-floor scaling raises the bar (the Tier-3 absorber
        # guard): evidence above the flat bar but below floor * bar fails.
        assert blend_pair_escape(5000.0, 3, 1.0, 0.4, 0.9, 0.9, evidence_floor=1.0)
        assert not blend_pair_escape(5000.0, 3, 1.0, 0.4, 0.9, 0.9, evidence_floor=50.0)

    def test_blend_escape_relative_evidence_lane(self):
        """A low-SNR constructive pair below the absolute bar escapes when
        its evidence is a large fraction of the feature's own (363 w87);
        dust pairs and small-fraction (shape-error-scale) pairs do not."""
        from ftmwpipeline.fitting.validation import blend_pair_escape

        # Below the absolute bar (2*50*3=300), no feature context -> reject.
        assert not blend_pair_escape(169.0, 3, 1.0, 0.4, 0.9, 0.9)
        # Same pair carrying 64% of the feature's evidence -> escape.
        assert blend_pair_escape(169.0, 3, 1.0, 0.4, 0.9, 0.9, feature_evidence=263.0)
        # High relative fraction but below the plain gate bar (2*5*3=30):
        # the dust guard rejects.
        assert not blend_pair_escape(20.0, 3, 1.0, 0.4, 0.9, 0.9, feature_evidence=40.0)
        # Shape-error scale (a few percent of a bright feature) -> reject.
        assert not blend_pair_escape(
            169.0, 3, 1.0, 0.4, 0.9, 0.9, feature_evidence=10000.0
        )
        # The cancellation veto still applies on the relative lane.
        assert not blend_pair_escape(
            169.0, 3, 1.0, 0.0, 0.95, np.pi * 0.98, feature_evidence=263.0
        )

    def test_amp_ratio_tier_preserves_balanced_supraresolution_pair(self):
        """A balanced (ratio ~1) pair in the amplitude-ratio band is a real
        close doublet and is NOT collapsed."""
        rng = np.random.default_rng(SEED + 22)
        sep = 0.10
        assert 1.0 / T_US < sep < 1.5 / T_US
        true = [
            ModelPeak(_amp_for_snr(120.0), -sep / 2, 0.3),
            ModelPeak(_amp_for_snr(120.0), +sep / 2, 2.4),  # 1:1, distinct phase
        ]
        u, z = _window(true, 0.8, 1.0, rng)
        sigma = np.full(u.size, 1.0)
        fit_kwargs = self._constraints_kwargs(u, z, sigma)
        k2_fit = fit_window(u, z, sigma, true, TAU_US, T_US, **fit_kwargs)
        assert k2_fit.success and k2_fit.n_peaks == 2
        merged, n_merged = merge_close_peaks_cleanup(
            u,
            z,
            sigma,
            k2_fit,
            TAU_US,
            T_US,
            fit_kwargs_inner=fit_kwargs,
        )
        assert n_merged == 0
        assert merged.n_peaks == 2

    def test_k1_input_returns_unchanged(self):
        """K<2 short-circuits to (fit, 0)."""
        rng = np.random.default_rng(SEED + 4)
        true = ModelPeak(_amp_for_snr(150.0), 0.0, 0.5)
        u, z = _window([true], 1.0, 1.0, rng)
        sigma = np.full(u.size, 1.0)
        fit_kwargs = self._constraints_kwargs(u, z, sigma)
        k1_fit = fit_window(
            u,
            z,
            sigma,
            [true],
            TAU_US,
            T_US,
            **fit_kwargs,
        )
        merged, n_merged = merge_close_peaks_cleanup(
            u,
            z,
            sigma,
            k1_fit,
            TAU_US,
            T_US,
            fit_kwargs_inner=fit_kwargs,
        )
        assert n_merged == 0
        assert merged is k1_fit

    def test_cleanup_preserves_tau_was_fit_and_tau_error_through_merge(self):
        """When a merge fires, the merged fit must carry the input's
        ``tau_was_fit`` AND ``tau_error`` forward. The cleanup refit
        internally uses ``fit_tau=False`` (a single-window K-1 refit
        must not broaden tau to absorb the merged peak), but the
        persisted ``tau_us`` came from the K-peak fit upstream -- its
        ``tau_was_fit`` and ``tau_error`` describe the actual tau
        determination and should not be clobbered by the locked-cleanup
        refit (whose covariance lacks a tau slot entirely).
        """
        rng = np.random.default_rng(SEED + 9)
        true = ModelPeak(_amp_for_snr(120.0), 0.0, 0.5)
        u, z = _window([true], 0.8, 1.0, rng)
        sigma = np.full(u.size, 1.0)
        duplicate_init = [
            ModelPeak(true.amplitude / 2, -0.5 * DF_MHZ, 0.5),
            ModelPeak(true.amplitude / 2, +0.5 * DF_MHZ, 0.5),
        ]
        fit_kwargs = self._constraints_kwargs(u, z, sigma)
        k2_fit = fit_window(
            u,
            z,
            sigma,
            duplicate_init,
            TAU_US,
            T_US,
            **fit_kwargs,
        )
        assert k2_fit.tau_was_fit is True
        assert k2_fit.tau_error is not None and np.isfinite(k2_fit.tau_error)
        merged, n_merged = merge_close_peaks_cleanup(
            u,
            z,
            sigma,
            k2_fit,
            TAU_US,
            T_US,
            fit_kwargs_inner=fit_kwargs,
        )
        assert n_merged == 1
        # The merged fit's mechanical fit_tau is False (refit locked tau),
        # but tau_was_fit + tau_error carry the originating K-fit's
        # answers forward.
        assert merged.fit_tau is False
        assert merged.tau_was_fit is True
        assert merged.tau_error == pytest.approx(k2_fit.tau_error)


# ---------------------------------------------------------------------------
# C1 immunity tests: user-peak protection (Tasks 1 & 2)
# ---------------------------------------------------------------------------


class TestMergeCleanupProtectedOffsets:
    """Protected peaks must never be collapsed by merge_close_peaks_cleanup."""

    def _constraints_kwargs(self, u, z, sigma):
        c = derive_window_fit_constraints(z, sigma, TAU_US, T_US)
        return c.fit_kwargs_inner

    def test_protected_peak_pair_not_collapsed(self):
        """Two sub-resolution duplicate peaks are normally collapsed (Tier 1).
        When one member is protected, the pair must be skipped and left intact.
        """
        rng = np.random.default_rng(SEED + 100)
        true = ModelPeak(_amp_for_snr(120.0), 0.0, 0.5)
        u, z = _window([true], 0.8, 1.0, rng)
        sigma = np.full(u.size, 1.0)
        # Force a K=2 duplicate pair at sub-resolution separation.
        duplicate_init = [
            ModelPeak(true.amplitude / 2, -0.5 * DF_MHZ, 0.5),
            ModelPeak(true.amplitude / 2, +0.5 * DF_MHZ, 0.5),
        ]
        fit_kwargs = self._constraints_kwargs(u, z, sigma)
        k2_fit = fit_window(u, z, sigma, duplicate_init, TAU_US, T_US, **fit_kwargs)
        assert k2_fit.n_peaks == 2

        # Without protection: the pair collapses (duplicate-overfit case).
        merged_no_prot, n_merged = merge_close_peaks_cleanup(
            u, z, sigma, k2_fit, TAU_US, T_US, fit_kwargs_inner=fit_kwargs
        )
        assert n_merged == 1

        # With one member protected: the pair must be preserved intact.
        protected = [k2_fit.peaks[0].offset_mhz]  # protect the first member
        merged_prot, n_merged_prot = merge_close_peaks_cleanup(
            u,
            z,
            sigma,
            k2_fit,
            TAU_US,
            T_US,
            fit_kwargs_inner=fit_kwargs,
            protected_offsets=protected,
            protected_tol_mhz=DF_MHZ,
        )
        assert n_merged_prot == 0, (
            "Protected pair should not be collapsed; n_merged=%d" % n_merged_prot
        )
        assert merged_prot.n_peaks == 2

    def test_unprotected_pair_still_collapses(self):
        """Without protected_offsets, the duplicate-pair collapse still fires."""
        rng = np.random.default_rng(SEED + 101)
        true = ModelPeak(_amp_for_snr(120.0), 0.0, 0.5)
        u, z = _window([true], 0.8, 1.0, rng)
        sigma = np.full(u.size, 1.0)
        duplicate_init = [
            ModelPeak(true.amplitude / 2, -0.5 * DF_MHZ, 0.5),
            ModelPeak(true.amplitude / 2, +0.5 * DF_MHZ, 0.5),
        ]
        fit_kwargs = self._constraints_kwargs(u, z, sigma)
        k2_fit = fit_window(u, z, sigma, duplicate_init, TAU_US, T_US, **fit_kwargs)
        # Protect a frequency far from either member: effectively no protection.
        merged, n_merged = merge_close_peaks_cleanup(
            u,
            z,
            sigma,
            k2_fit,
            TAU_US,
            T_US,
            fit_kwargs_inner=fit_kwargs,
            protected_offsets=[5.0],  # far away; no effect
            protected_tol_mhz=DF_MHZ,
        )
        assert n_merged == 1  # still collapses without protection


class TestRemoveAndRefitProtectedOffsets:
    """Protected peaks must never be dropped by remove_and_refit_cleanup."""

    def _constraints_kwargs(self, u, z, sigma):
        c = derive_window_fit_constraints(z, sigma, TAU_US, T_US)
        return c.fit_kwargs_inner

    def test_protected_peak_not_dropped(self):
        """When a peak is protected, it is excluded from the drop-candidate set.

        Build a noiseless K=2 fit where one peak is a very weak absorber
        (amplitude close to 0) so the F-test reliably flags it as redundant
        in the unprotected case.  With protection it must survive.
        """
        # Noiseless single-peak window -- gives a deterministic F-test result.
        true = [ModelPeak(_amp_for_snr(200.0, sigma=1.0), 0.0, 0.5)]
        u = _offset_grid(1.5)
        z = model_spectrum(u, true, TAU_US, T_US)  # noiseless
        sigma = np.full(u.size, 1.0)
        fit_kwargs = self._constraints_kwargs(u, z, sigma)

        # Ghost peak at 2× FWHM, initialised to near-zero amplitude.
        ghost_offset = 2.0 * FWHM
        epsilon_amp = _amp_for_snr(0.01, sigma=1.0)  # negligible amplitude
        k2_init = [
            ModelPeak(_amp_for_snr(200.0, sigma=1.0), 0.0, 0.5),
            ModelPeak(epsilon_amp, ghost_offset, 0.0),
        ]
        k2_fit = fit_window(u, z, sigma, k2_init, TAU_US, T_US, **fit_kwargs)
        assert k2_fit.n_peaks == 2

        # Unprotected: the ghost must be dropped (it is redundant by F-test).
        cleaned, n_dropped = remove_and_refit_cleanup(
            u, z, sigma, k2_fit, TAU_US, T_US, fit_kwargs_inner=fit_kwargs
        )
        assert n_dropped >= 1, (
            "Ghost peak should be flagged as redundant by the F-test; "
            "n_dropped=%d, ghost amp=%.4f, peaks=%s"
            % (
                n_dropped,
                k2_fit.peaks[1].amplitude,
                [(round(p.offset_mhz, 3), round(p.amplitude, 4)) for p in k2_fit.peaks],
            )
        )

        # Protected: the ghost must survive even though it is F-test-redundant.
        cleaned_prot, n_dropped_prot = remove_and_refit_cleanup(
            u,
            z,
            sigma,
            k2_fit,
            TAU_US,
            T_US,
            fit_kwargs_inner=fit_kwargs,
            protected_offsets=[k2_fit.peaks[1].offset_mhz],
            protected_tol_mhz=DF_MHZ,
        )
        assert n_dropped_prot < n_dropped, (
            "Protected peak should suppress at least one drop; "
            "n_dropped_prot=%d n_dropped=%d" % (n_dropped_prot, n_dropped)
        )


class TestRescueForbiddenOffsets:
    """Forbidden offsets must not be re-nominated by attempt_residual_rescue."""

    def test_forbidden_offset_not_proposed_by_rescue(self):
        """A detected residual candidate at a forbidden offset is dropped
        before being passed to conservative_fit.

        Strategy: build a noiseless two-peak window, fit only the first peak
        (leaving a deterministic residual at the second), then call
        attempt_residual_rescue with all offsets in the residual forbidden
        except a region far from any real signal.  The rescue's candidates
        detector finds the second peak; the forbidden filter drops it; the
        rescue accepts zero new peaks.
        """
        # Noiseless window: deterministic residual so the detector always
        # nominates exactly the true second peak's offset.
        true = [
            ModelPeak(_amp_for_snr(150.0, sigma=1.0), -0.8, 0.5),
            ModelPeak(_amp_for_snr(120.0, sigma=1.0), +0.7, 1.4),
        ]
        u = _offset_grid(2.0)
        z = model_spectrum(u, true, TAU_US, T_US)  # noiseless
        sigma = np.full(u.size, 1.0)

        # Initial fit: only the first peak (given its exact offset).
        initial = conservative_fit(
            u, z, sigma, [-0.8], TAU_US, T_US, seeder_max_k=1
        )
        # In noiseless data the fitter installs the seed unconditionally.

        # Rescue without forbidden: the detector nominates around +0.7.
        outcome_free = attempt_residual_rescue(
            u, z, sigma, initial.fit, TAU_US, T_US
        )
        # The second peak must be among the rescue's *candidates* (pre-fit).
        cand_offsets = [c.frequency_mhz for c in outcome_free.candidates]
        assert any(abs(o - true[1].offset_mhz) < 0.5 for o in cand_offsets), (
            "Rescue should detect a candidate near the second peak offset; "
            "candidates=%s" % [round(o, 3) for o in cand_offsets]
        )

        # Rescue with the second peak's offset forbidden: that specific
        # candidate must not appear in the forbidden outcome's candidate list.
        outcome_forbidden = attempt_residual_rescue(
            u,
            z,
            sigma,
            initial.fit,
            TAU_US,
            T_US,
            forbidden_offsets=[true[1].offset_mhz],
            forbidden_tol_mhz=0.5,  # half-MHz tolerance covers LSQ drift
        )
        cand_offsets_forbidden = [c.frequency_mhz for c in outcome_forbidden.candidates]
        assert not any(
            abs(o - true[1].offset_mhz) < 0.5 for o in cand_offsets_forbidden
        ), (
            "Forbidden candidate near +0.7 must not appear after filtering; "
            "candidates=%s" % [round(o, 3) for o in cand_offsets_forbidden]
        )

    def test_conservative_fit_forbidden_filters_add_loop(self):
        """conservative_fit drops forbidden candidates from the add-loop queue.

        Call conservative_fit with one candidate that is explicitly forbidden;
        the function must return an empty fit (no peaks accepted).
        """
        rng = np.random.default_rng(SEED + 301)
        true = [ModelPeak(_amp_for_snr(200.0), 0.0, 0.5)]
        u, z = _window(true, 1.2, 1.0, rng)
        sigma = np.full(u.size, 1.0)

        # Without forbidden: the candidate is accepted.
        result_free = conservative_fit(u, z, sigma, [0.0], TAU_US, T_US)
        assert result_free.n_peaks >= 1

        # With the candidate's offset forbidden: the add-loop queue is empty.
        result_forbidden = conservative_fit(
            u,
            z,
            sigma,
            [0.0],
            TAU_US,
            T_US,
            forbidden_offsets=[0.0],
            forbidden_tol_mhz=DF_MHZ,
        )
        assert result_forbidden.n_peaks == 0, (
            "Forbidden candidate should be dropped from the add-loop; "
            "got n_peaks=%d" % result_forbidden.n_peaks
        )
