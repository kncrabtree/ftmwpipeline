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
    merge_close_peaks_cleanup,
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

        initial = conservative_fit(u, z, sigma, [-0.8], TAU_US, T_US)
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

    def test_resolution_floor_merges_subresolution_pair(self):
        """A pair separated by less than one active-FT resolution element
        (``1/T_active``) but more than ``structural_merge_factor * FWHM`` is
        merged unconditionally only because of the resolution-referenced floor
        (GitHub issue #13). With the floor disabled it survives the FWHM-only
        structural cutoff."""
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

    def test_amp_ratio_tier_collapses_supraresolution_absorber(self):
        """A pair just *above* the resolution floor (so tiers 1-2 leave it) with
        a large amplitude ratio is collapsed by the amplitude-ratio tier -- the
        weak member is a rescue-parked shape-error absorber, not a real doublet
        (GitHub issue #13, the w281/w143 class). A balanced pair in the same
        band survives."""
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
