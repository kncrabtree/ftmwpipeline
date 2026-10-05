"""
Unit tests for Stage 5 plan-level execution (task 5).

Covers :mod:`ftmwpipeline.fitting.plan_execution`:

* fixed-contributor evaluation (frame remap, frozen-background subtraction),
* :func:`fit_window_with_fixed_contributors` (the free-peak fit-on-residual),
* the DAG/batch walk in :func:`execute_plan` (topological order, primaries
  fit before dependents, contributors pulled from primaries),
* residual edge-coherence + local thaw renegotiation (the contributor-pick
  heuristic, the joint co-fit improving a flagged edge -- the canonical
  coupled-pair case the 36350/36389 doublet stands in for).

All scenarios are synthetic so the ground truth is exact. The 2638 acquisition
scale (``T = 12.65 us``, ``tau = 5 us``, ``df ~ 12 kHz``, lower sideband at
``40960 MHz``) is used throughout so the FWHM, SNR, and bin spacing match what
the production code will see.
"""

from __future__ import annotations

import multiprocessing
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ftmwpipeline.core.data_structures import (
    FitWindow,
    FixedContributor,
    MergeRequest,
    Peak,
    PeakClassification,
    Sideband,
    WindowPlan,
)
from ftmwpipeline.fitting import plan_execution
from ftmwpipeline.fitting.active_ft import ActiveFTResult, PointMap
from ftmwpipeline.fitting.peak_model import (
    ModelPeak,
    effective_tau,
    h_T,
    model_spectrum,
    sideband_sign,
)
from ftmwpipeline.fitting.plan_execution import (
    DEFAULT_RESIDUAL_EDGE_THRESHOLD,
    REPLAN_REASON_PREFIXES,
    FrozenPeak,
    ReplanContext,
    RescueEvent,
    ThawEvent,
    WindowOutcome,
    _apply_structural_merges,
    _pair_of,
    _PendingMerge,
    _select_structural_merges,
    _window_center_mhz,
    attempt_thaw_round,
    evaluate_edge_free_contributors,
    execute_plan,
    fit_window_with_fixed_contributors,
    local_thaw_cofit,
    refit_outcome,
    residual_edge_coherence,
    select_contributor_to_thaw,
    subtract_frozen_background,
)
from ftmwpipeline.fitting.validation import feature_fwhm

if TYPE_CHECKING:
    from ftmwpipeline.fitting.residual_screening import ResidualPeakCandidate

needs_fork = pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(), reason="needs fork"
)

# --- 2638-scale acquisition --------------------------------------------------
T_US = 12.65
TAU_US = 5.0
DF_MHZ = 0.0122
PROBE_MHZ = 40960.0
SIDEBAND = Sideband.LOWER
START_US = 0.0  # de-ramp is identity for tests; spectrum is already in [0, T]
FWHM = feature_fwhm(TAU_US, T_US)
SEED = 20260524


# ---------------------------------------------------------------------------
# Synthetic spectrum builder
# ---------------------------------------------------------------------------
def _synth_spectrum(
    freq_array: np.ndarray,
    peaks_mhz_amp_phase: list[tuple[float, float, float]],
    tau_us: float = TAU_US,
    acquisition_us: float = T_US,
) -> np.ndarray:
    """Build a synthetic noise-free complex spectrum on a molecular-frequency grid.

    Each peak is ``(molecular_freq_mhz, amplitude, phase)``. Lower sideband is
    assumed (so ``f_bb = -(f - f_probe)``); ``h_T`` is evaluated on the signed
    baseband offset ``s*(f - f_j)`` from each line center.
    """
    s = -1.0  # lower sideband
    z = np.zeros(freq_array.shape, dtype=np.complex128)
    for f_j, amp, phase in peaks_mhz_amp_phase:
        du = s * (freq_array - f_j)
        z += 0.5 * amp * np.exp(1j * phase) * h_T(du, tau_us, acquisition_us)
    return z


def _amp_for_snr(snr: float, sigma: float = 1.0) -> float:
    """Amplitude needed for an on-resonance SNR of ``snr`` (relative to sigma)."""
    return 2.0 * snr * sigma / effective_tau(TAU_US, T_US)


def _complex_noise(n: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    s = sigma / np.sqrt(2.0)
    return rng.normal(0.0, s, n) + 1j * rng.normal(0.0, s, n)


def _make_active_ft(
    freq_array: np.ndarray, complex_spectrum: np.ndarray, *, alpha: float = 1.0
) -> ActiveFTResult:
    """Wrap a synthetic ``(freq, spectrum)`` pair as an :class:`ActiveFTResult`.

    The tests build their spectra on a uniform synthetic frequency grid in the
    natural ``h_T`` amplitude convention (``0.5 * A * exp(i*phi) * h_T(...)``),
    which is exactly the active-FT convention. ``alpha`` defaults to ``1.0``
    (no zero-padding context) since the tests do not exercise the persisted ->
    active noise rescale -- they pass ``rms_noise`` directly in active-FT units.
    """
    spec = np.asarray(complex_spectrum, dtype=np.complex128)
    n_active = spec.size
    n_raw = max(int(round(n_active / alpha)), n_active)
    return ActiveFTResult(
        freq_mhz=np.asarray(freq_array, dtype=float),
        complex_spectrum=spec,
        alpha=float(alpha),
        n_active=n_active,
        n_raw=n_raw,
    )


# ---------------------------------------------------------------------------
# subtract_frozen_background
# ---------------------------------------------------------------------------
class TestEvaluateEdgeFreeContributors:
    """The self-contained active-FT read for edge-free contributors -- a joint
    complex LSQ of the line template over the cluster's core bins (no primary
    fit). Recovers each line's (amplitude, phase) and remaps the offset into the
    dependent window's frame."""

    def test_joint_lsq_recovers_amplitude_phase_and_offset(self):
        freq = np.arange(36080.0, 36140.0, DF_MHZ)
        # A two-line cluster sharing one primary window; co-located so the joint
        # solve must de-contaminate (the whole point vs a single-bin phasor).
        lines = [(36100.0, 5.0, 0.3), (36100.6, 8.0, -0.7)]
        z = _synth_spectrum(freq, lines)
        active = _make_active_ft(freq, z)
        contributors = [
            FixedContributor(
                peak_index=1,
                primary_window_id=0,
                frequency_mhz=36100.0,
                edge_free=True,
            ),
            FixedContributor(
                peak_index=2,
                primary_window_id=0,
                frequency_mhz=36100.6,
                edge_free=True,
            ),
        ]
        dep_center = 36120.0
        frozen = evaluate_edge_free_contributors(
            contributors,
            active.freq_mhz,
            active.complex_spectrum,
            dependent_center_mhz=dep_center,
            sideband=SIDEBAND,
            tau_us=TAU_US,
            acquisition_us=T_US,
        )
        assert len(frozen) == 2
        by_pi = {f.peak_index: f for f in frozen}
        s = -1.0
        assert by_pi[1].model_peak.amplitude == pytest.approx(5.0, rel=1e-3)
        assert by_pi[2].model_peak.amplitude == pytest.approx(8.0, rel=1e-3)
        assert by_pi[1].model_peak.phase == pytest.approx(0.3, abs=1e-3)
        assert by_pi[2].model_peak.phase == pytest.approx(-0.7, abs=1e-3)
        assert by_pi[1].model_peak.offset_mhz == pytest.approx(
            s * (36100.0 - dep_center)
        )
        assert by_pi[1].edge_free is True
        assert by_pi[1].primary_window_id == 0

    def test_non_edge_free_contributors_ignored(self):
        freq = np.arange(36080.0, 36140.0, DF_MHZ)
        z = _synth_spectrum(freq, [(36100.0, 5.0, 0.3)])
        active = _make_active_ft(freq, z)
        contributors = [
            FixedContributor(
                peak_index=1,
                primary_window_id=0,
                frequency_mhz=36100.0,
                edge_free=False,
            ),
        ]
        frozen = evaluate_edge_free_contributors(
            contributors,
            active.freq_mhz,
            active.complex_spectrum,
            dependent_center_mhz=36120.0,
            sideband=SIDEBAND,
            tau_us=TAU_US,
            acquisition_us=T_US,
        )
        assert frozen == []

    def test_without_point_map_uid_stays_unset(self):
        """No PointMap supplied (the default) -- peak_uid stays None, matching
        every other seed constructor's ``point_map=None`` default."""
        freq = np.arange(36080.0, 36140.0, DF_MHZ)
        z = _synth_spectrum(freq, [(36100.0, 5.0, 0.3)])
        active = _make_active_ft(freq, z)
        contributors = [
            FixedContributor(
                peak_index=1,
                primary_window_id=0,
                frequency_mhz=36100.0,
                edge_free=True,
            ),
        ]
        frozen = evaluate_edge_free_contributors(
            contributors,
            active.freq_mhz,
            active.complex_spectrum,
            dependent_center_mhz=36120.0,
            sideband=SIDEBAND,
            tau_us=TAU_US,
            acquisition_us=T_US,
        )
        assert len(frozen) == 1
        assert frozen[0].model_peak.peak_uid is None

    def test_peak_uid_stamped_from_seed_matches_a_k1_seed_in_any_window(self):
        """A birth: the frozen model's peak_uid is stamped from the
        contributor's own seed frequency (``c.frequency_mhz``), and because
        point space is absolute the same number is what a K=1 seed at that
        Stage 3 frequency would get in ANY window's own frame -- not a
        lookup, a consequence of both being births at the same seed
        frequency (see scratch/edge-free-contributor-identity.md)."""
        from ftmwpipeline.fitting.active_ft import peak_uid_from_offset

        true_freq = 36100.0
        freq = np.arange(36080.0, 36140.0, DF_MHZ)
        z = _synth_spectrum(freq, [(true_freq, 5.0, 0.3)])
        active = _make_active_ft(freq, z)
        contributors = [
            FixedContributor(
                peak_index=1,
                primary_window_id=0,
                frequency_mhz=true_freq,
                edge_free=True,
            ),
        ]
        dep_center = 36120.0
        n_active, sample_dt_us = 1000, 0.05
        s = sideband_sign(SIDEBAND)
        point_map = PointMap.from_frame(
            dep_center, SIDEBAND, PROBE_MHZ, n_active, sample_dt_us
        )
        frozen = evaluate_edge_free_contributors(
            contributors,
            active.freq_mhz,
            active.complex_spectrum,
            dependent_center_mhz=dep_center,
            sideband=SIDEBAND,
            tau_us=TAU_US,
            acquisition_us=T_US,
            point_map=point_map,
        )
        assert len(frozen) == 1
        uid = frozen[0].model_peak.peak_uid
        assert uid is not None
        assert uid == point_map.stamp(s * (true_freq - dep_center))

        # Absoluteness: a K=1 seed at the same Stage 3 frequency, computed in
        # a completely different window's frame (arbitrary center, nothing to
        # do with dep_center), lands on the exact same identifier.
        other_center = 36042.0
        other_offset = s * (true_freq - other_center)
        other_uid = peak_uid_from_offset(
            other_offset, other_center, SIDEBAND, PROBE_MHZ, n_active, sample_dt_us
        )
        assert uid == other_uid

    def test_peak_uid_stamped_from_seed_not_from_vp_refined_position(self):
        """The VP frequency refinement (:data:`DEFAULT_EDGE_FREE_FREQ_REFINE`)
        moves ``line_freqs`` (plan_execution.py ~:833) and the returned
        ``FrozenPeak.frequency_mhz`` legitimately follows that refined value
        -- but the identifier must not. Deliberately mis-seed the
        contributor off the line's true position (within the VP trust
        region) so the read genuinely refines back toward the true center;
        pins that peak_uid is stamped from the (unrefined) seed and NOT
        from the refined read position, catching the exact re-derivation
        mistake the design forbids."""
        true_freq = 36100.0
        seed_freq = true_freq + 0.01  # deliberately off, within the VP bound
        freq = np.arange(36080.0, 36140.0, DF_MHZ)
        z = _synth_spectrum(freq, [(true_freq, 5.0, 0.3)])
        active = _make_active_ft(freq, z)
        contributors = [
            FixedContributor(
                peak_index=1,
                primary_window_id=0,
                frequency_mhz=seed_freq,
                edge_free=True,
            ),
        ]
        dep_center = 36120.0
        n_active, sample_dt_us = 1000, 0.05
        s = sideband_sign(SIDEBAND)
        point_map = PointMap.from_frame(
            dep_center, SIDEBAND, PROBE_MHZ, n_active, sample_dt_us
        )
        frozen = evaluate_edge_free_contributors(
            contributors,
            active.freq_mhz,
            active.complex_spectrum,
            dependent_center_mhz=dep_center,
            sideband=SIDEBAND,
            tau_us=TAU_US,
            acquisition_us=T_US,
            point_map=point_map,
        )
        assert len(frozen) == 1
        fp = frozen[0]

        # Confirm the VP refinement actually moved the read position -- the
        # returned frequency_mhz (f0, the refined value) should have pulled
        # back toward the true center, away from the deliberately-wrong seed.
        moved_mhz = abs(fp.frequency_mhz - seed_freq)
        assert moved_mhz > 0.002, (
            f"VP refinement only moved the read {moved_mhz:.5f} MHz -- too "
            "small to distinguish a stamp-from-seed from a "
            "stamp-from-refined-value regression"
        )
        assert fp.frequency_mhz == pytest.approx(true_freq, abs=1e-3)

        # The positive: stamped from the (unrefined) seed frequency.
        expected_from_seed = point_map.stamp(s * (seed_freq - dep_center))
        assert fp.model_peak.peak_uid == expected_from_seed

        # The negative that matters most: NOT the point-space id of the
        # VP-refined (fitted) position.
        uid_from_refined = point_map.stamp(s * (fp.frequency_mhz - dep_center))
        assert fp.model_peak.peak_uid != uid_from_refined


class TestSubtractFrozenBackground:
    def test_no_contributors_returns_data_unchanged(self):
        grid = np.linspace(-1.0, 1.0, 101)
        data = np.ones_like(grid, dtype=np.complex128) * (0.3 + 0.2j)
        bg, diff = subtract_frozen_background(grid, data, [], TAU_US, T_US)
        assert np.allclose(bg, 0.0)
        assert np.allclose(diff, data)

    def test_subtracts_known_model(self):
        """A frozen peak exactly recovers when subtracted from a synthetic line."""
        grid = np.linspace(-2.0, 2.0, 501)
        peak = ModelPeak(amplitude=5.0, offset_mhz=0.1, phase=0.5)
        line = model_spectrum(grid, [peak], TAU_US, T_US)
        frozen = FrozenPeak(0, 0, peak, frequency_mhz=0.0, freeze_eligible=True)
        bg, diff = subtract_frozen_background(grid, line, [frozen], TAU_US, T_US)
        assert np.allclose(bg, line)
        assert np.allclose(diff, 0.0)


# ---------------------------------------------------------------------------
# fit_window_with_fixed_contributors
# ---------------------------------------------------------------------------
class TestFitWindowWithFixedContributors:
    def test_recovers_free_peak_with_frozen_contributor(self):
        """Two-line scenario: one frozen, one free; free is recovered cleanly.

        Without the frozen contributor the free fit would be biased by the
        contributor's skirt; with it frozen, the free peak recovers to ~kHz.
        """
        rng = np.random.default_rng(SEED)
        sigma = 1.0
        # The frozen contributor lives off-grid (1.5 MHz to the left); its
        # skirt reaches into the window.
        contributor_peak = ModelPeak(
            amplitude=_amp_for_snr(200.0, sigma), offset_mhz=-1.5, phase=0.3
        )
        # The free line we want to recover.
        free_peak = ModelPeak(
            amplitude=_amp_for_snr(50.0, sigma), offset_mhz=0.0, phase=1.1
        )

        grid = np.arange(-0.6, 0.6 + DF_MHZ / 2, DF_MHZ)
        data = model_spectrum(
            grid, [contributor_peak, free_peak], TAU_US, T_US
        ) + _complex_noise(grid.size, sigma, rng)
        frozen = FrozenPeak(0, 0, contributor_peak, frequency_mhz=0.0)

        fit, background, full_fitted, full_residual = (
            fit_window_with_fixed_contributors(
                grid,
                data,
                sigma,
                [frozen],
                [0.0],  # one candidate, at the free line
                TAU_US,
                T_US,
            )
        )
        assert fit.fit.success
        assert fit.fit.n_peaks == 1
        assert abs(fit.fit.peaks[0].offset_mhz - free_peak.offset_mhz) < 0.005
        # The full fitted spectrum is free + background; residual is < 4 sigma.
        assert np.max(np.abs(full_residual)) < 4.0 * sigma

    def test_background_addition_reconstructs_full_model(self):
        """``full_fitted_spectrum == free_model + background`` by construction."""
        rng = np.random.default_rng(SEED + 1)
        sigma = 0.5
        contributor_peak = ModelPeak(_amp_for_snr(150.0, sigma), -1.2, 0.7)
        free_peak = ModelPeak(_amp_for_snr(70.0, sigma), 0.3, 2.1)
        grid = np.arange(-0.8, 0.8 + DF_MHZ / 2, DF_MHZ)
        data = model_spectrum(
            grid, [contributor_peak, free_peak], TAU_US, T_US
        ) + _complex_noise(grid.size, sigma, rng)
        frozen = FrozenPeak(0, 0, contributor_peak, frequency_mhz=0.0)

        fit, background, full_fitted, _ = fit_window_with_fixed_contributors(
            grid, data, sigma, [frozen], [0.3], TAU_US, T_US
        )
        free_only = model_spectrum(grid, fit.fit.peaks, fit.fit.tau_us, T_US)
        assert np.allclose(full_fitted, free_only + background)


# ---------------------------------------------------------------------------
# residual_edge_coherence
# ---------------------------------------------------------------------------
class TestResidualEdgeCoherence:
    def test_noise_only_residual_near_null_mean(self):
        """Pure-noise residual gives S_coh near the null mean ~0.886 on each edge."""
        rng = np.random.default_rng(SEED + 2)
        sigma = 1.0
        residual = _complex_noise(10_000, sigma, rng)
        # Average over many bands to suppress single-sample variance.
        statistics = []
        for offset in range(0, residual.size - 200, 100):
            low, high = residual_edge_coherence(
                residual[offset : offset + 200], sigma, band_m=64
            )
            statistics.extend([low, high])
        assert np.mean(statistics) == pytest.approx(np.sqrt(np.pi / 4.0), abs=0.05)

    def test_coherent_residual_flags_above_threshold(self):
        """A coherent leakage skirt on one edge exceeds the default threshold."""
        sigma = 1.0
        # Put a strong off-grid line whose skirt drives the high edge coherent.
        grid = np.arange(-1.0, 1.0 + DF_MHZ / 2, DF_MHZ)
        skirt_source = ModelPeak(
            amplitude=_amp_for_snr(500.0, sigma), offset_mhz=2.0, phase=0.0
        )
        residual = model_spectrum(grid, [skirt_source], TAU_US, T_US)
        low, high = residual_edge_coherence(residual, sigma, band_m=32)
        # The line is on the high-frequency side of the grid (offset > 0); the
        # high-edge band should flag, the low-edge band should not.
        assert high > DEFAULT_RESIDUAL_EDGE_THRESHOLD
        assert low < high

    def test_short_window_clamps_band_size(self):
        """A residual shorter than 2*band_m still produces two finite statistics."""
        sigma = 1.0
        residual = np.zeros(20, dtype=np.complex128)
        low, high = residual_edge_coherence(residual, sigma, band_m=64)
        assert np.isfinite(low)
        assert np.isfinite(high)


# ---------------------------------------------------------------------------
# select_contributor_to_thaw
# ---------------------------------------------------------------------------
class TestSelectContributorToThaw:
    def _window(self):
        return FitWindow(
            window_id=1,
            freq_range=(36100.0, 36110.0),
        )

    def _frozen(self, freq_mhz, *, freeze_eligible=True, peak_index=0):
        return FrozenPeak(
            peak_index=peak_index,
            primary_window_id=0,
            model_peak=ModelPeak(1.0, 0.0, 0.0),
            frequency_mhz=freq_mhz,
            freeze_eligible=freeze_eligible,
        )

    def test_picks_low_side_contributor(self):
        w = self._window()
        below = self._frozen(36099.0, peak_index=10)
        above = self._frozen(36111.0, peak_index=20)
        chosen = select_contributor_to_thaw(w, [below, above], "low")
        assert chosen is below

    def test_picks_high_side_contributor(self):
        w = self._window()
        below = self._frozen(36099.0, peak_index=10)
        above = self._frozen(36111.0, peak_index=20)
        chosen = select_contributor_to_thaw(w, [below, above], "high")
        assert chosen is above

    def test_prefers_freeze_ineligible(self):
        w = self._window()
        eligible = self._frozen(36099.5, freeze_eligible=True, peak_index=1)
        ineligible = self._frozen(36098.0, freeze_eligible=False, peak_index=2)
        chosen = select_contributor_to_thaw(w, [eligible, ineligible], "low")
        assert chosen is ineligible

    def test_returns_none_when_no_contributor_on_side(self):
        w = self._window()
        below = self._frozen(36099.0)
        chosen = select_contributor_to_thaw(w, [below], "high")
        assert chosen is None

    def test_rejects_bad_side_argument(self):
        w = self._window()
        with pytest.raises(ValueError, match="edge_side"):
            select_contributor_to_thaw(w, [], "left")


# ---------------------------------------------------------------------------
# execute_plan: DAG / batch order + happy-path fit
# ---------------------------------------------------------------------------
class TestExecutePlanHappyPath:
    def _build_two_window_plan(
        self, strong_freq: float, weak_freq: float, strong_snr: float, weak_snr: float
    ):
        """A canonical 2-window setup: A (strong) is B (weak)'s contributor."""
        sigma = 1.0
        # Spectrum spans both windows plus the strong-line skirt reach.
        freq_array = np.arange(strong_freq - 5.0, weak_freq + 5.0, DF_MHZ)
        strong_amp = _amp_for_snr(strong_snr, sigma)
        weak_amp = _amp_for_snr(weak_snr, sigma)
        # Phases are arbitrary; pick distinct values to catch phase mismaps.
        true_peaks = [
            (strong_freq, strong_amp, 0.3),
            (weak_freq, weak_amp, 2.1),
        ]
        spectrum = _synth_spectrum(freq_array, true_peaks)
        rng = np.random.default_rng(SEED + 3)
        spectrum = spectrum + _complex_noise(freq_array.size, sigma, rng)
        rms_noise = np.full(freq_array.size, sigma)

        # Window A: tight around the strong line.
        win_a = FitWindow(
            window_id=0,
            freq_range=(strong_freq - 0.6, strong_freq + 0.6),
            free_peak_indices=[0],
            fixed_contributors=[],
            batch=0,
        )
        # Window B: tight around the weak line; A is its frozen contributor.
        win_b = FitWindow(
            window_id=1,
            freq_range=(weak_freq - 0.6, weak_freq + 0.6),
            free_peak_indices=[1],
            fixed_contributors=[
                FixedContributor(
                    peak_index=0,
                    primary_window_id=0,
                    frequency_mhz=strong_freq,
                    freeze_eligible=True,
                )
            ],
            batch=1,
        )
        plan = WindowPlan(
            windows=[win_a, win_b],
            dependency_edges=[(1, 0)],
            topological_order=[0, 1],
        )
        peak_frequencies = [strong_freq, weak_freq]
        return plan, freq_array, spectrum, rms_noise, peak_frequencies, true_peaks

    def test_two_window_plan_recovers_both_lines(self):
        """The classic primary->dependent walk fits both lines cleanly."""
        strong_freq, weak_freq = 36100.0, 36110.0
        plan, freqs, spec, noise, peak_freqs, true_peaks = self._build_two_window_plan(
            strong_freq, weak_freq, strong_snr=300.0, weak_snr=50.0
        )

        outcome = execute_plan(
            plan,
            _make_active_ft(freqs, spec),
            noise,
            peak_freqs,
            sideband=SIDEBAND,
            acquisition_us=T_US,
            tau0_us=TAU_US,
        )

        # Both windows fit.
        assert set(outcome.window_outcomes.keys()) == {0, 1}
        w0 = outcome.window_outcomes[0]
        w1 = outcome.window_outcomes[1]
        assert w0.fit.success
        assert w1.fit.success
        assert w0.fit.n_peaks == 1
        assert w1.fit.n_peaks == 1

        # The dependent froze the ancestor window's actual fitted line as its
        # leakage background: keyed off the DAG edge (primary_window_id), content
        # read from window 0's fit -- no Stage-3 contributor record, so peak_index
        # is the no-link sentinel and the frozen frequency is the *fitted* one.
        assert len(w1.fixed_peaks) == 1
        assert w1.fixed_peaks[0].peak_index == -1
        assert w1.fixed_peaks[0].primary_window_id == 0
        assert w1.fixed_peaks[0].frequency_mhz == pytest.approx(strong_freq, abs=0.01)

        # The dependent's free fit recovers the weak line to ~kHz.
        s = -1.0
        # Frame: free peak is in dep's offset frame = s*(weak - dep_center).
        dep_center = 0.5 * (weak_freq - 0.6 + weak_freq + 0.6)
        expected_offset = s * (weak_freq - dep_center)
        got_offset = w1.fit.peaks[0].offset_mhz
        assert abs(got_offset - expected_offset) < 0.003  # 3 kHz

        # No thaw was triggered.
        assert outcome.thaw_history == []

    def test_frozen_background_is_one_per_ancestor_line_not_per_contributor(self):
        """§4: a dependent freezes the ancestor's *fitted* lines, deduping the
        Stage-3 contributor count. Two contributors that both reference the same
        single-line ancestor produce ONE frozen peak (no doubled skirt)."""
        strong_freq, weak_freq = 36100.0, 36110.0
        plan, freqs, spec, noise, peak_freqs, _ = self._build_two_window_plan(
            strong_freq, weak_freq, strong_snr=300.0, weak_snr=50.0
        )
        # Window 0 fits a single strong line, but give window 1 a SECOND
        # contributor that also points at window 0 (as if Stage 3 double-detected
        # the strong line). The pre-§4 path would freeze its skirt twice.
        win1 = plan.windows[1]
        win1.fixed_contributors.append(
            FixedContributor(
                peak_index=99,
                primary_window_id=0,
                frequency_mhz=strong_freq + 0.001,
                freeze_eligible=True,
            )
        )
        outcome = execute_plan(
            plan,
            _make_active_ft(freqs, spec),
            noise,
            peak_freqs,
            sideband=SIDEBAND,
            acquisition_us=T_US,
            tau0_us=TAU_US,
        )
        # Window 0 fit exactly one line -> the dependent freezes exactly one,
        # despite two contributors referencing window 0.
        assert outcome.window_outcomes[0].fit.n_peaks == 1
        assert len(outcome.window_outcomes[1].fixed_peaks) == 1
        assert outcome.window_outcomes[1].fixed_peaks[0].primary_window_id == 0

    def test_respects_topological_order(self):
        """A window referenced as a primary must be fit before its dependent."""
        # Build a 3-window plan with a chain 0 -> 1 -> 2 to exercise multiple
        # batches.
        strong_freq, mid_freq, weak_freq = 36100.0, 36115.0, 36130.0
        sigma = 1.0
        freq_array = np.arange(strong_freq - 5.0, weak_freq + 5.0, DF_MHZ)
        spectrum = _synth_spectrum(
            freq_array,
            [
                (strong_freq, _amp_for_snr(300.0, sigma), 0.4),
                (mid_freq, _amp_for_snr(150.0, sigma), 1.2),
                (weak_freq, _amp_for_snr(60.0, sigma), 2.7),
            ],
        )
        rng = np.random.default_rng(SEED + 5)
        spectrum = spectrum + _complex_noise(freq_array.size, sigma, rng)
        rms_noise = np.full(freq_array.size, sigma)

        windows = [
            FitWindow(
                window_id=0,
                freq_range=(strong_freq - 0.7, strong_freq + 0.7),
                free_peak_indices=[0],
                batch=0,
            ),
            FitWindow(
                window_id=1,
                freq_range=(mid_freq - 0.7, mid_freq + 0.7),
                free_peak_indices=[1],
                fixed_contributors=[
                    FixedContributor(0, 0, strong_freq, freeze_eligible=True)
                ],
                batch=1,
            ),
            FitWindow(
                window_id=2,
                freq_range=(weak_freq - 0.7, weak_freq + 0.7),
                free_peak_indices=[2],
                fixed_contributors=[
                    FixedContributor(1, 1, mid_freq, freeze_eligible=True),
                ],
                batch=2,
            ),
        ]
        plan = WindowPlan(
            windows=windows,
            dependency_edges=[(1, 0), (2, 1)],
            topological_order=[0, 1, 2],
        )
        peak_freqs = [strong_freq, mid_freq, weak_freq]

        outcome = execute_plan(
            plan,
            _make_active_ft(freq_array, spectrum),
            rms_noise,
            peak_freqs,
            sideband=SIDEBAND,
            acquisition_us=T_US,
            tau0_us=TAU_US,
        )
        # Every window converged with one free peak.
        assert all(out.fit.success for out in outcome.window_outcomes.values())
        for wid in (0, 1, 2):
            assert outcome.window_outcomes[wid].fit.n_peaks == 1
        # The chain dependents have their primaries' contributors.
        assert len(outcome.window_outcomes[1].fixed_peaks) == 1
        assert len(outcome.window_outcomes[2].fixed_peaks) == 1
        assert outcome.window_outcomes[1].fixed_peaks[0].primary_window_id == 0
        assert outcome.window_outcomes[2].fixed_peaks[0].primary_window_id == 1

    def test_no_fixed_contributors_path(self):
        """A plan with no contributors still runs (a degenerate-but-valid case)."""
        strong_freq = 36100.0
        sigma = 1.0
        freq_array = np.arange(strong_freq - 2.0, strong_freq + 2.0, DF_MHZ)
        spectrum = _synth_spectrum(
            freq_array, [(strong_freq, _amp_for_snr(200.0, sigma), 0.5)]
        )
        rng = np.random.default_rng(SEED + 6)
        spectrum = spectrum + _complex_noise(freq_array.size, sigma, rng)
        rms_noise = np.full(freq_array.size, sigma)
        win = FitWindow(
            window_id=0,
            freq_range=(strong_freq - 0.8, strong_freq + 0.8),
            free_peak_indices=[0],
            batch=0,
        )
        plan = WindowPlan(windows=[win], topological_order=[0])
        outcome = execute_plan(
            plan,
            _make_active_ft(freq_array, spectrum),
            rms_noise,
            [strong_freq],
            sideband=SIDEBAND,
            acquisition_us=T_US,
            tau0_us=TAU_US,
        )
        assert outcome.window_outcomes[0].fit.n_peaks == 1
        assert outcome.window_outcomes[0].fixed_peaks == []
        assert outcome.thaw_history == []


# ---------------------------------------------------------------------------
# Local thaw renegotiation: the coupled-pair case
# ---------------------------------------------------------------------------
class TestLocalThaw:
    def test_thaw_triggers_when_frozen_contributor_is_wrong(self):
        """A corrupted primary fit leaves a coherent dependent edge; thaw fixes it.

        Builds the truth-consistent spectrum and runs execute_plan to get clean
        outcomes; then mutates the primary's converged fit to mimic the
        canonical failure mode (a partially-resolved blended primary fit as one
        slightly-over-amped cosine -- the prototype's §5 ~1 kHz bias case),
        re-derives the dependent's frozen background and residual, verifies the
        residual edge facing the primary is now coherent, and calls
        :func:`attempt_thaw_round` to confirm the co-fit clears the flagged
        edge and the contributor is promoted to a free peak in the dependent.
        """
        rng = np.random.default_rng(SEED + 10)
        sigma = 1.0
        strong_freq = 36100.0
        # 4 MHz apart: A's skirt at B is ~A/(2*pi*4) ~ 4% of A's center, and
        # the weak line's skirt at A is similarly modest, so neither line
        # makes the other's window residual coherent under a clean fit.
        weak_freq = 36104.0
        strong_amp = _amp_for_snr(800.0, sigma)
        weak_amp = _amp_for_snr(100.0, sigma)
        freq_array = np.arange(strong_freq - 5.0, weak_freq + 5.0, DF_MHZ)
        spectrum = _synth_spectrum(
            freq_array,
            [(strong_freq, strong_amp, 0.3), (weak_freq, weak_amp, 1.7)],
        )
        spectrum = spectrum + _complex_noise(freq_array.size, sigma, rng)
        rms_noise = np.full(freq_array.size, sigma)

        # A in [36099.4, 36100.6], B in [36103.4, 36104.6] -- well-separated;
        # each window's residual under a clean fit is dominated by noise.
        win_a = FitWindow(
            window_id=0,
            freq_range=(strong_freq - 0.6, strong_freq + 0.6),
            free_peak_indices=[0],
            batch=0,
        )
        win_b = FitWindow(
            window_id=1,
            freq_range=(weak_freq - 0.6, weak_freq + 0.6),
            free_peak_indices=[1],
            fixed_contributors=[
                FixedContributor(0, 0, strong_freq, freeze_eligible=True)
            ],
            batch=1,
        )
        plan = WindowPlan(
            windows=[win_a, win_b],
            dependency_edges=[(1, 0)],
            topological_order=[0, 1],
        )

        # Clean run first to populate both outcomes.
        outcome = execute_plan(
            plan,
            _make_active_ft(freq_array, spectrum),
            rms_noise,
            [strong_freq, weak_freq],
            sideband=SIDEBAND,
            acquisition_us=T_US,
            tau0_us=TAU_US,
        )
        # On a clean spectrum the residual edges are quiet -- no spontaneous
        # thaw.
        assert outcome.thaw_history == []

        # Mutate the primary's converged fit: bump the strong line's fitted
        # amplitude 25% high. This stands in for the blended-primary case where
        # the conservative loop's sequential initialization collapses two close
        # cosines into one stronger-looking fit (§4 of the prototype report).
        primary = outcome.window_outcomes[0]
        original_amp = primary.fit.peaks[0].amplitude
        primary.fit.peaks[0].amplitude = original_amp * 1.25

        # Re-derive the dependent's frozen background, full model, residual,
        # and edge-coherence with the corrupted primary -- exactly what
        # execute_plan would have done if it had re-visited the dependent.
        dep = outcome.window_outcomes[1]
        dep_center_mhz = 0.5 * (weak_freq - 0.3 + weak_freq + 0.3)
        # Inline the single-peak remap that ``evaluate_fixed_contributor`` used
        # to do: win_a has exactly one fitted peak, so it is trivially the
        # "nearest match" to the contributor's persisted frequency.
        s = sideband_sign(SIDEBAND)
        primary_center_mhz = 0.5 * (win_a.freq_range[0] + win_a.freq_range[1])
        nearest_peak = primary.fit.peaks[0]
        fitted_freq_mhz = primary_center_mhz + s * nearest_peak.offset_mhz
        corrupted_frozen = FrozenPeak(
            peak_index=0,
            primary_window_id=0,
            model_peak=ModelPeak(
                amplitude=nearest_peak.amplitude,
                offset_mhz=s * (fitted_freq_mhz - dep_center_mhz),
                phase=nearest_peak.phase,
            ),
            frequency_mhz=fitted_freq_mhz,
            freeze_eligible=True,
        )
        dep.fixed_peaks = [corrupted_frozen]
        dep.background = model_spectrum(
            dep.offset_grid_mhz,
            [corrupted_frozen.model_peak],
            dep.fit.fit.tau_us,
            T_US,
        )
        dep.full_fitted_spectrum = dep.fit.fit.fitted_spectrum + dep.background
        dep.full_residual = dep.complex_spectrum - dep.full_fitted_spectrum
        low_before, high_before = residual_edge_coherence(
            dep.full_residual, dep.rms_noise
        )
        dep.edge_coherence_low = low_before
        dep.edge_coherence_high = high_before
        # The over-amped frozen skirt should flag at least one residual edge.
        assert max(low_before, high_before) > DEFAULT_RESIDUAL_EDGE_THRESHOLD, (
            f"expected the over-amped contributor to flag an edge, "
            f"got (low, high) = ({low_before:.2f}, {high_before:.2f})"
        )

        # Drive one round of the renegotiation handshake on the dependent.
        events = attempt_thaw_round(
            win_b,
            dep,
            outcomes=outcome.window_outcomes,
            sideband=SIDEBAND,
            acquisition_us=T_US,
            tau0_us=TAU_US,
        )

        assert events, "expected the flagged edge to produce a thaw event"
        accepted = [e for e in events if e.accepted]
        assert accepted, "expected the co-fit to clear the flagged edge"
        ev = accepted[0]
        assert ev.dependent_window_id == 1
        assert ev.primary_window_id == 0
        assert ev.contributor_peak_index == 0
        assert ev.edge_coherence_after < ev.edge_coherence_before
        # The dependent dropped the thawed contributor and gained it as a
        # free peak.
        contrib_ids = [fp.peak_index for fp in dep.fixed_peaks]
        assert 0 not in contrib_ids
        # Free-peak count grew by one (was 1, now 2: the weak line plus the
        # thawed strong line).
        assert dep.fit.fit.n_peaks == 2

    def test_accepted_thaw_installs_cofit_statistics(self):
        """An accepted thaw must install the fresh co-fit's own statistics
        (C1): ``peak_errors``, ``covariance``, ``chi_squared``, ``cost``,
        ``n_params``, ``n_data``, and ``tau_error`` all have to come from the
        joint co-fit, not survive as stale carry-over from the pre-thaw
        independent fit. Same corrupted-primary/coherent-edge scenario as
        ``test_thaw_triggers_when_frozen_contributor_is_wrong``, but this test
        asserts on the *installed statistics* rather than just the peak/edge
        bookkeeping.
        """
        rng = np.random.default_rng(SEED + 30)
        sigma = 1.0
        strong_freq = 36100.0
        weak_freq = 36104.0
        strong_amp = _amp_for_snr(800.0, sigma)
        weak_amp = _amp_for_snr(100.0, sigma)
        freq_array = np.arange(strong_freq - 5.0, weak_freq + 5.0, DF_MHZ)
        spectrum = _synth_spectrum(
            freq_array,
            [(strong_freq, strong_amp, 0.3), (weak_freq, weak_amp, 1.7)],
        )
        spectrum = spectrum + _complex_noise(freq_array.size, sigma, rng)
        rms_noise = np.full(freq_array.size, sigma)

        win_a = FitWindow(
            window_id=0,
            freq_range=(strong_freq - 0.6, strong_freq + 0.6),
            free_peak_indices=[0],
            batch=0,
        )
        win_b = FitWindow(
            window_id=1,
            freq_range=(weak_freq - 0.6, weak_freq + 0.6),
            free_peak_indices=[1],
            fixed_contributors=[
                FixedContributor(0, 0, strong_freq, freeze_eligible=True)
            ],
            batch=1,
        )
        plan = WindowPlan(
            windows=[win_a, win_b],
            dependency_edges=[(1, 0)],
            topological_order=[0, 1],
        )

        outcome = execute_plan(
            plan,
            _make_active_ft(freq_array, spectrum),
            rms_noise,
            [strong_freq, weak_freq],
            sideband=SIDEBAND,
            acquisition_us=T_US,
            tau0_us=TAU_US,
        )
        assert outcome.thaw_history == []

        # Corrupt the primary's fitted amplitude, exactly as in
        # test_thaw_triggers_when_frozen_contributor_is_wrong.
        primary = outcome.window_outcomes[0]
        original_amp = primary.fit.peaks[0].amplitude
        primary.fit.peaks[0].amplitude = original_amp * 1.25

        dep = outcome.window_outcomes[1]
        dep_center_mhz = 0.5 * (weak_freq - 0.3 + weak_freq + 0.3)
        s = sideband_sign(SIDEBAND)
        primary_center_mhz = 0.5 * (win_a.freq_range[0] + win_a.freq_range[1])
        nearest_peak = primary.fit.peaks[0]
        fitted_freq_mhz = primary_center_mhz + s * nearest_peak.offset_mhz
        corrupted_frozen = FrozenPeak(
            peak_index=0,
            primary_window_id=0,
            model_peak=ModelPeak(
                amplitude=nearest_peak.amplitude,
                offset_mhz=s * (fitted_freq_mhz - dep_center_mhz),
                phase=nearest_peak.phase,
            ),
            frequency_mhz=fitted_freq_mhz,
            freeze_eligible=True,
        )
        dep.fixed_peaks = [corrupted_frozen]
        dep.background = model_spectrum(
            dep.offset_grid_mhz,
            [corrupted_frozen.model_peak],
            dep.fit.fit.tau_us,
            T_US,
        )
        dep.full_fitted_spectrum = dep.fit.fit.fitted_spectrum + dep.background
        dep.full_residual = dep.complex_spectrum - dep.full_fitted_spectrum
        low_before, high_before = residual_edge_coherence(
            dep.full_residual, dep.rms_noise
        )
        dep.edge_coherence_low = low_before
        dep.edge_coherence_high = high_before
        assert max(low_before, high_before) > DEFAULT_RESIDUAL_EDGE_THRESHOLD

        # Snapshot the PRE-thaw (stale, independent-fit) statistics on both
        # windows before the renegotiation mutates them in place.
        stale_dep_chi2 = dep.fit.fit.chi_squared
        stale_dep_n_params = dep.fit.fit.n_params
        stale_dep_errors = [
            (e.amplitude, e.offset_mhz, e.phase) for e in dep.fit.fit.peak_errors
        ]
        stale_primary_chi2 = primary.fit.fit.chi_squared
        stale_primary_errors = [
            (e.amplitude, e.offset_mhz, e.phase) for e in primary.fit.fit.peak_errors
        ]
        assert len(stale_dep_errors) == 1  # only the weak line was free so far

        events = attempt_thaw_round(
            win_b,
            dep,
            outcomes=outcome.window_outcomes,
            sideband=SIDEBAND,
            acquisition_us=T_US,
            tau0_us=TAU_US,
        )
        accepted = [e for e in events if e.accepted]
        assert accepted, "expected the co-fit to clear the flagged edge"

        # -- Dependent window: gained the thawed line as a free peak (K=2).
        inner = dep.fit.fit
        assert inner.n_peaks == 2
        new_dep_errors = [
            (e.amplitude, e.offset_mhz, e.phase) for e in inner.peak_errors
        ]
        assert len(new_dep_errors) == 2
        assert all(np.isfinite(v) for e in new_dep_errors for v in e)
        # The weak line's own error must come from the fresh joint fit, not
        # the stale pre-thaw independent one.
        assert new_dep_errors[0] != stale_dep_errors[0]

        # n_params/n_data/chi_squared must reflect THIS window's own
        # two-peak, tau-free co-fit -- not the stale one-peak fit's.
        assert inner.n_params == 3 * 2 + 1  # 2 peaks * 3 + the shared tau
        assert inner.n_params != stale_dep_n_params
        assert inner.n_data == 2 * dep.offset_grid_mhz.size
        assert inner.chi_squared != pytest.approx(stale_dep_chi2)
        assert inner.cost == pytest.approx(0.5 * inner.chi_squared)
        dof = inner.n_data - inner.n_params
        assert dof > 0
        assert inner.reduced_chi2 == pytest.approx(inner.chi_squared / dof)
        # A believable co-fit on a correctly-modeled synthetic spectrum should
        # land near reduced_chi2 ~ 1, not blow up the way the stale-error
        # bug's uncorrected pre-thaw chi2 would suggest.
        assert inner.reduced_chi2 < 10.0

        assert inner.tau_error is not None
        assert np.isfinite(inner.tau_error)
        assert inner.covariance is not None
        assert inner.covariance.shape == (7, 7)
        # Per-line errors must match the covariance diagonal they were
        # derived from: peak-major (amplitude, offset, phase) blocks in
        # peak order, then the shared tau in the tail.
        diag = np.sqrt(np.diag(inner.covariance))
        assert inner.peak_errors[0].amplitude == pytest.approx(diag[0])
        assert inner.peak_errors[0].offset_mhz == pytest.approx(diag[1])
        assert inner.peak_errors[0].phase == pytest.approx(diag[2])
        assert inner.peak_errors[1].amplitude == pytest.approx(diag[3])
        assert inner.peak_errors[1].offset_mhz == pytest.approx(diag[4])
        assert inner.peak_errors[1].phase == pytest.approx(diag[5])

        # -- Primary window: same peak count (K=1) but a fresh joint refit,
        # so its stats must move too -- not stay pinned at the pre-thaw
        # (corrupted-amplitude) fit's values.
        p_inner = primary.fit.fit
        assert p_inner.n_peaks == 1
        assert p_inner.n_params == 3 * 1 + 1
        assert p_inner.tau_error is not None
        assert p_inner.covariance is not None
        assert p_inner.covariance.shape == (4, 4)
        new_primary_errors = [
            (e.amplitude, e.offset_mhz, e.phase) for e in p_inner.peak_errors
        ]
        assert new_primary_errors != stale_primary_errors
        assert p_inner.chi_squared != pytest.approx(stale_primary_chi2)

        # tau is one physically shared parameter: both windows must report
        # the identical joint tau and tau error, not independent values.
        assert dep.fit.fit.tau_us == pytest.approx(primary.fit.fit.tau_us)
        assert dep.fit.fit.tau_error == pytest.approx(primary.fit.fit.tau_error)

    def test_no_thaw_for_a_clean_fit(self):
        """When the primary fits correctly, no thaw event is generated."""
        rng = np.random.default_rng(SEED + 11)
        sigma = 1.0
        strong_freq, weak_freq = 36100.0, 36110.0
        freq_array = np.arange(strong_freq - 5.0, weak_freq + 5.0, DF_MHZ)
        spectrum = _synth_spectrum(
            freq_array,
            [
                (strong_freq, _amp_for_snr(300.0, sigma), 0.4),
                (weak_freq, _amp_for_snr(60.0, sigma), 1.9),
            ],
        )
        spectrum = spectrum + _complex_noise(freq_array.size, sigma, rng)
        rms_noise = np.full(freq_array.size, sigma)

        win_a = FitWindow(
            window_id=0,
            freq_range=(strong_freq - 0.7, strong_freq + 0.7),
            free_peak_indices=[0],
            batch=0,
        )
        win_b = FitWindow(
            window_id=1,
            freq_range=(weak_freq - 0.7, weak_freq + 0.7),
            free_peak_indices=[1],
            fixed_contributors=[FixedContributor(0, 0, strong_freq, True)],
            batch=1,
        )
        plan = WindowPlan(
            windows=[win_a, win_b],
            dependency_edges=[(1, 0)],
            topological_order=[0, 1],
        )

        outcome = execute_plan(
            plan,
            _make_active_ft(freq_array, spectrum),
            rms_noise,
            [strong_freq, weak_freq],
            sideband=SIDEBAND,
            acquisition_us=T_US,
            tau0_us=TAU_US,
        )
        assert outcome.thaw_history == []

    def test_thaw_rounds_bounded(self):
        """The thaw loop respects ``max_thaw_rounds``."""
        # A pathological set-up: the residual edge is always coherent because
        # we keep the over-amped contributor in the spectrum even after thaw.
        # We achieve that by simply not having a frozen contributor on the
        # right side -- the contributor that *would* have been thawed already
        # is gone, so `select_contributor_to_thaw` returns None on round 2.
        rng = np.random.default_rng(SEED + 12)
        sigma = 1.0
        strong_freq, weak_freq = 36100.0, 36110.0
        freq_array = np.arange(strong_freq - 5.0, weak_freq + 5.0, DF_MHZ)
        # An artefact at the low edge of B that cannot be cleared by any
        # available contributor: a coherent injection without a matching
        # fixed-contributor record.
        artefact = model_spectrum(
            -1.0 * (freq_array - (weak_freq - 1.5)),
            [ModelPeak(_amp_for_snr(300.0, sigma), 0.0, 0.0)],
            TAU_US,
            T_US,
        )
        spectrum = (
            _synth_spectrum(
                freq_array,
                [
                    (strong_freq, _amp_for_snr(200.0, sigma), 0.4),
                    (weak_freq, _amp_for_snr(60.0, sigma), 1.9),
                ],
            )
            + artefact
            + _complex_noise(freq_array.size, sigma, rng)
        )
        rms_noise = np.full(freq_array.size, sigma)
        win_a = FitWindow(
            window_id=0,
            freq_range=(strong_freq - 0.7, strong_freq + 0.7),
            free_peak_indices=[0],
            batch=0,
        )
        win_b = FitWindow(
            window_id=1,
            freq_range=(weak_freq - 0.7, weak_freq + 0.7),
            free_peak_indices=[1],
            # No fixed contributor on the low edge -- thaw cannot help.
            fixed_contributors=[],
            batch=1,
        )
        plan = WindowPlan(
            windows=[win_a, win_b],
            topological_order=[0, 1],
        )

        outcome = execute_plan(
            plan,
            _make_active_ft(freq_array, spectrum),
            rms_noise,
            [strong_freq, weak_freq],
            sideband=SIDEBAND,
            acquisition_us=T_US,
            tau0_us=TAU_US,
            max_thaw_rounds=2,
        )
        # The loop ran at most max_thaw_rounds rounds on window 1 and
        # recorded no-op thaw events (no contributor to blame).
        w1_events = outcome.window_outcomes[1].thaw_events
        # Each round yields up to two events (low and high edges); the loop
        # is bounded.
        assert all(not e.accepted for e in w1_events)
        # We attempted, but the total events come from at most max_thaw_rounds
        # passes through both edges.
        assert len(w1_events) <= 2 * 2  # 2 rounds * 2 edges max


# ---------------------------------------------------------------------------
# local_thaw_cofit (a more direct view)
# ---------------------------------------------------------------------------
class TestLocalThawCofit:
    def test_cofit_returns_combined_peaks(self):
        """Joint co-fit reconstructs primary + thawed + dependent free peaks."""
        rng = np.random.default_rng(SEED + 20)
        sigma = 1.0
        strong_freq, weak_freq = 36100.0, 36108.0
        strong_amp = _amp_for_snr(300.0, sigma)
        weak_amp = _amp_for_snr(80.0, sigma)
        freq_array = np.arange(strong_freq - 4.0, weak_freq + 4.0, DF_MHZ)
        spectrum = _synth_spectrum(
            freq_array,
            [(strong_freq, strong_amp, 0.2), (weak_freq, weak_amp, 1.3)],
        ) + _complex_noise(freq_array.size, sigma, rng)
        rms_noise = np.full(freq_array.size, sigma)

        # Manually build the two outcomes by running the plan executor first.
        win_a = FitWindow(
            window_id=0,
            freq_range=(strong_freq - 0.8, strong_freq + 0.8),
            free_peak_indices=[0],
            batch=0,
        )
        win_b = FitWindow(
            window_id=1,
            freq_range=(weak_freq - 0.8, weak_freq + 0.8),
            free_peak_indices=[1],
            fixed_contributors=[FixedContributor(0, 0, strong_freq, True)],
            batch=1,
        )
        plan = WindowPlan(windows=[win_a, win_b], topological_order=[0, 1])
        out = execute_plan(
            plan,
            _make_active_ft(freq_array, spectrum),
            rms_noise,
            [strong_freq, weak_freq],
            sideband=SIDEBAND,
            acquisition_us=T_US,
            tau0_us=TAU_US,
        )
        primary = out.window_outcomes[0]
        dependent = out.window_outcomes[1]

        joint, idx = local_thaw_cofit(
            dependent,
            primary,
            thawed=dependent.fixed_peaks[0],
            sideband=SIDEBAND,
            tau0_us=TAU_US,
            acquisition_us=T_US,
        )
        assert joint.success
        # The thawed line is *already* a free peak of the primary's fit, so the
        # joint peak list is primary peaks + dependent peaks (no extra slot).
        assert len(joint.peaks) == (primary.fit.n_peaks + dependent.fit.n_peaks)
        # The thawed index points to the primary peak nearest the contributor.
        assert idx.shape == (1,)
        assert 0 <= int(idx[0]) < primary.fit.n_peaks


# ---------------------------------------------------------------------------
# Structural renegotiation: execute_plan with a ReplanContext
# ---------------------------------------------------------------------------
# Stage 4 parameter defaults used when manually building a WindowPlan for the
# dispatcher tests; the values match the production defaults exposed in
# preprocessing.window_planning.
_STAGE4_PARAMS = {
    "edge_m": 16,
    "trim_m": 4,
    "edge_threshold": 1.5,
    "max_window_width_mhz": 40.0,
    "min_freeze_snr": 50.0,
    "min_window_half_width_mhz": 2.0,
    "acquisition_us": T_US,
    "tau_us": TAU_US,
    "start_us": START_US,
    "probe_freq_mhz": PROBE_MHZ,
}

#: Stage 4 parameters with the points cap off, so the MHz cap (40 MHz, far wider
#: than the fixtures) governs.
_WIDE_PARAMS = {**_STAGE4_PARAMS, "max_window_width_points": 0}


def _make_peak(
    frequency_mhz: float,
    snr: float,
    sigma: float,
    *,
    grid_index: int,
    classification: PeakClassification = PeakClassification.STRONG,
) -> Peak:
    """Promoted Peak in the shape Stage 4 expects."""
    return Peak(
        frequency=frequency_mhz,
        intensity=snr * sigma,
        index=grid_index,
        snr=snr,
        noise_std_local=sigma,
        classification=classification,
        promoted=True,
    )


class TestStructuralReplan:
    """``execute_plan`` with a :class:`ReplanContext` runs the structural
    renegotiation outer loop after the main fit + thaw passes. When a
    residual edge flags with no contributor on that side and a
    frequency-adjacent neighbor exists, a :class:`MergeRequest` is emitted
    and the plan is revised in place. Outcomes for the merged windows + their
    downstream dependents are dropped and refit on the revised plan.

    Existing tests pass ``replan_context=None`` (the default) and are
    unaffected; the synthetic fixtures here use a hand-built ``WindowPlan``
    so the dispatcher can be exercised without depending on Stage 4's
    contributor-attachment heuristics.
    """

    def _single_peak_setup(self, peak_freq: float, peak_snr: float, sigma: float):
        """Build a one-peak fixture: spectrum + peak list + Stage 5 inputs."""
        return self._lines_setup([(peak_freq, peak_snr)], sigma)

    @staticmethod
    def _lines_setup(lines, sigma: float):
        """Spectrum + peak list + Stage 5 inputs for ``lines = [(MHz, SNR)]``."""
        rng = np.random.default_rng(SEED + 30)
        freq_array = np.arange(36100.0, 36120.0 + DF_MHZ / 2, DF_MHZ)
        spectrum = _synth_spectrum(
            freq_array, [(f, _amp_for_snr(snr, sigma), 0.3) for f, snr in lines]
        )
        spectrum = spectrum + _complex_noise(freq_array.size, sigma, rng)
        rms = np.full(freq_array.size, sigma)
        peaks = [
            _make_peak(f, snr, sigma, grid_index=int(np.argmin(np.abs(freq_array - f))))
            for f, snr in lines
        ]
        return freq_array, spectrum, rms, peaks

    #: A strong line just inside window A's top edge, and window B's own line.
    _STRADDLE_LINES = [(36104.9, 300.0), (36112.0, 100.0)]

    def _straddle_run(self, b_low_mhz: float = 36105.0):
        """Window A ``[36100, 36105]`` holds the strong line 0.1 MHz below its
        top edge; window B ``[b_low_mhz, 36118]`` holds its own line at 36112
        MHz, and its low edge sees the strong line's skirt with no contributor
        to thaw. Returns ``(outcome, freq_array)``."""
        freq_array, spectrum, rms, peaks = self._lines_setup(
            self._STRADDLE_LINES, sigma=1.0
        )
        win_a = FitWindow(
            window_id=0,
            freq_range=(36100.0, 36105.0),
            free_peak_indices=[0],
            batch=0,
        )
        win_b = FitWindow(
            window_id=1,
            freq_range=(b_low_mhz, 36118.0),
            free_peak_indices=[1],
            batch=0,
        )
        plan = WindowPlan(
            windows=[win_a, win_b],
            topological_order=[0, 1],
            parameters=dict(_WIDE_PARAMS),
        )
        ctx = ReplanContext(
            peaks=peaks,
            active_freq_mhz=freq_array,
            active_complex_spectrum=spectrum,
            active_rms_noise=rms,
        )
        out = execute_plan(
            plan,
            _make_active_ft(freq_array, spectrum),
            rms,
            [f for f, _snr in self._STRADDLE_LINES],
            sideband=SIDEBAND,
            acquisition_us=T_US,
            tau0_us=TAU_US,
            replan_context=ctx,
        )
        return out, freq_array

    def test_no_replan_context_skips_structural_loop(self):
        """Without a ReplanContext, ``execute_plan`` behaves exactly as before:
        no structural events, final_plan_revision stays at 0."""
        freq_array, spectrum, rms, _peaks = self._single_peak_setup(
            36110.0, peak_snr=100.0, sigma=1.0
        )
        win = FitWindow(
            window_id=0,
            freq_range=(36108.0, 36112.0),
            free_peak_indices=[0],
            batch=0,
        )
        plan = WindowPlan(windows=[win], topological_order=[0])
        outcome = execute_plan(
            plan,
            _make_active_ft(freq_array, spectrum),
            rms,
            [36110.0],
            sideband=SIDEBAND,
            acquisition_us=T_US,
            tau0_us=TAU_US,
        )
        assert outcome.replan_history == []
        assert outcome.final_plan_revision == 0
        # With no replan the outcome carries the original plan unchanged, so
        # the caller's conversion/refit sees identical windows.
        assert outcome.final_plan is plan

    def test_merge_fires_when_feature_crosses_boundary(self):
        """A strong line sits just inside window A; window B holds a line of its
        own, and its low edge sees the strong line's leakage skirt. With no
        contributor attached to B's low side, the dispatcher emits a merge of A
        and B; replan produces one wide window; refit clears the
        previously-flagged edge.
        """
        outcome, _freqs = self._straddle_run()

        # A structural merge was applied.
        assert outcome.replan_history, "expected a structural-replan event"
        accepted = [e for e in outcome.replan_history if e.accepted]
        assert accepted, "expected the merge to be accepted"
        ev = accepted[0]
        assert (ev.triggering_window_id, ev.edge_side) == (1, "low")
        assert ev.surviving_window_id == 0
        assert ev.revision_after == ev.revision_before + 1
        assert outcome.final_plan_revision == 1

        # The revised plan travels back on the outcome so the caller persists
        # the survivor with the *union* freq_range. Without it the survivor
        # keeps window A's narrow range while its fit spans the merged span,
        # which renders fitted peaks outside the stored window (the w340 bug).
        assert outcome.final_plan is not None
        survivor = {w.window_id: w for w in outcome.final_plan.windows}
        assert set(survivor) == {0}
        assert survivor[0].freq_range == (36100.0, 36118.0)

        # Only one window left; it fits both lines; its edges are clean (the
        # feature is no longer cut by a boundary).
        assert set(outcome.window_outcomes.keys()) == {0}
        merged = outcome.window_outcomes[0]
        assert merged.fit.fit.success
        assert merged.fit.n_peaks == 2
        # The survivor's stored range contains its fitted lines (no escape).
        lo, hi = survivor[0].freq_range
        for f, _snr in self._STRADDLE_LINES:
            assert lo <= f <= hi
        # Edges below the dispatcher threshold (else another round would fire
        # if max_replan_rounds allowed).
        assert merged.edge_coherence_low <= DEFAULT_RESIDUAL_EDGE_THRESHOLD
        assert merged.edge_coherence_high <= DEFAULT_RESIDUAL_EDGE_THRESHOLD

    def test_a_flag_on_a_window_that_fits_no_line_is_not_merged(self):
        """Window B is empty: its low edge sees window A's line skirt and has no
        contributor to thaw, but B's fit holds no line, so no fitted line
        straddles the boundary. The flag is recorded as not merged and the plan
        is untouched."""
        sigma = 1.0
        peak_freq = 36104.9
        freq_array, spectrum, rms, peaks = self._single_peak_setup(
            peak_freq, peak_snr=300.0, sigma=sigma
        )
        win_a = FitWindow(
            window_id=0,
            freq_range=(36100.0, 36105.0),
            free_peak_indices=[0],
            batch=0,
        )
        win_b = FitWindow(
            window_id=1,
            freq_range=(36105.0, 36110.0),
            free_peak_indices=[],
            batch=0,
        )
        plan = WindowPlan(
            windows=[win_a, win_b],
            topological_order=[0, 1],
            parameters=dict(_STAGE4_PARAMS),
        )
        ctx = ReplanContext(
            peaks=peaks,
            active_freq_mhz=freq_array,
            active_complex_spectrum=spectrum,
            active_rms_noise=rms,
        )
        outcome = execute_plan(
            plan,
            _make_active_ft(freq_array, spectrum),
            rms,
            [peak_freq],
            sideband=SIDEBAND,
            acquisition_us=T_US,
            tau0_us=TAU_US,
            replan_context=ctx,
        )
        (ev,) = outcome.replan_history
        assert (ev.triggering_window_id, ev.edge_side) == (1, "low")
        assert ev.edge_coherence_before > DEFAULT_RESIDUAL_EDGE_THRESHOLD
        assert not ev.accepted
        assert ev.reason.startswith("not merged: ")
        assert "fit holds no line" in ev.reason
        assert ev.revision_after == ev.revision_before == 0
        assert outcome.final_plan_revision == 0
        assert outcome.final_plan is plan
        assert outcome.window_outcomes[1].fit.n_peaks == 0

    def test_flagged_edge_with_contributor_does_not_merge(self):
        """When a flagged edge already has a fixed contributor, thaw owns
        that edge -- the structural dispatcher must not also emit a merge.
        """
        sigma = 1.0
        peak_freq = 36104.9
        freq_array, spectrum, rms, peaks = self._single_peak_setup(
            peak_freq, peak_snr=300.0, sigma=sigma
        )
        win_a = FitWindow(
            window_id=0,
            freq_range=(36100.0, 36105.0),
            free_peak_indices=[0],
            batch=0,
        )
        # B claims the peak in A as a fixed contributor (the standard
        # leakage-attachment outcome), so thaw -- not merge -- would handle
        # any flagged B-low edge.
        win_b = FitWindow(
            window_id=1,
            freq_range=(36105.0, 36110.0),
            free_peak_indices=[],
            fixed_contributors=[
                FixedContributor(
                    peak_index=0,
                    primary_window_id=0,
                    frequency_mhz=peak_freq,
                    freeze_eligible=True,
                )
            ],
            batch=1,
        )
        plan = WindowPlan(
            windows=[win_a, win_b],
            dependency_edges=[(1, 0)],
            topological_order=[0, 1],
            parameters=dict(_STAGE4_PARAMS),
        )
        ctx = ReplanContext(
            peaks=peaks,
            active_freq_mhz=freq_array,
            active_complex_spectrum=spectrum,
            active_rms_noise=rms,
        )

        outcome = execute_plan(
            plan,
            _make_active_ft(freq_array, spectrum),
            rms,
            [peak_freq],
            sideband=SIDEBAND,
            acquisition_us=T_US,
            tau0_us=TAU_US,
            replan_context=ctx,
        )
        # No structural merge was emitted: the contributor on B's low side
        # made thaw the responsible primitive (thaw may or may not have
        # been triggered depending on residual coherence, but a merge is
        # ruled out by construction).
        assert outcome.replan_history == []

    def test_flagged_edge_with_no_adjacent_window_records_no_event(self):
        """A flagged edge at the plan's outer boundary has no neighbor to
        merge with -- the dispatcher silently drops the request and exits."""
        sigma = 1.0
        # Peak placed near the *low* edge of the only window so its skirt
        # leaks past that edge into a region the plan does not cover.
        peak_freq = 36100.2
        freq_array, spectrum, rms, peaks = self._single_peak_setup(
            peak_freq, peak_snr=300.0, sigma=sigma
        )
        win = FitWindow(
            window_id=0,
            freq_range=(36100.0, 36105.0),
            free_peak_indices=[0],
            batch=0,
        )
        plan = WindowPlan(
            windows=[win],
            topological_order=[0],
            parameters=dict(_STAGE4_PARAMS),
        )
        ctx = ReplanContext(
            peaks=peaks,
            active_freq_mhz=freq_array,
            active_complex_spectrum=spectrum,
            active_rms_noise=rms,
        )
        outcome = execute_plan(
            plan,
            _make_active_ft(freq_array, spectrum),
            rms,
            [peak_freq],
            sideband=SIDEBAND,
            acquisition_us=T_US,
            tau0_us=TAU_US,
            replan_context=ctx,
        )
        # No adjacent window -> no merge events recorded.
        assert outcome.replan_history == []

    def test_max_replan_rounds_zero_disables_loop(self):
        """``max_replan_rounds=0`` runs the initial walk only."""
        sigma = 1.0
        peak_freq = 36104.9
        freq_array, spectrum, rms, peaks = self._single_peak_setup(
            peak_freq, peak_snr=300.0, sigma=sigma
        )
        win_a = FitWindow(
            window_id=0,
            freq_range=(36100.0, 36105.0),
            free_peak_indices=[0],
            batch=0,
        )
        win_b = FitWindow(
            window_id=1,
            freq_range=(36105.0, 36110.0),
            free_peak_indices=[],
            batch=0,
        )
        plan = WindowPlan(
            windows=[win_a, win_b],
            topological_order=[0, 1],
            parameters=dict(_STAGE4_PARAMS),
        )
        ctx = ReplanContext(
            peaks=peaks,
            active_freq_mhz=freq_array,
            active_complex_spectrum=spectrum,
            active_rms_noise=rms,
            max_replan_rounds=0,
        )
        outcome = execute_plan(
            plan,
            _make_active_ft(freq_array, spectrum),
            rms,
            [peak_freq],
            sideband=SIDEBAND,
            acquisition_us=T_US,
            tau0_us=TAU_US,
            replan_context=ctx,
        )
        assert outcome.replan_history == []
        assert outcome.final_plan_revision == 0
        assert set(outcome.window_outcomes.keys()) == {0, 1}


# ---------------------------------------------------------------------------
# Structural replan: which merges a round applies
# ---------------------------------------------------------------------------
# Windows below are cut on a bin grid so "bins between two windows" and "bins
# between the outermost lines" -- what the structural merge is gated on -- are
# written down directly.
_BIN_FREQ = 36100.0 + np.arange(1640) * DF_MHZ


def _bin_plan(windows, params=None):
    """``(plan, peaks)`` from ``windows = [(id, lo_bin, hi_bin, [peak bins])]``."""
    peaks: list[Peak] = []
    fit_windows = []
    for wid, lo, hi, pbins in windows:
        idx = []
        for b in pbins:
            idx.append(len(peaks))
            peaks.append(_make_peak(float(_BIN_FREQ[b]), 100.0, 1.0, grid_index=b))
        fit_windows.append(
            FitWindow(
                window_id=wid,
                freq_range=(float(_BIN_FREQ[lo]), float(_BIN_FREQ[hi])),
                free_peak_indices=idx,
                diagnostics={"grid_span": [int(lo), int(hi)]},
            )
        )
    plan = WindowPlan(
        windows=fit_windows,
        topological_order=[w.window_id for w in fit_windows],
        parameters=dict(_STAGE4_PARAMS if params is None else params),
    )
    return plan, peaks


def _bin_ctx(peaks):
    n = _BIN_FREQ.size
    return ReplanContext(
        peaks=peaks,
        active_freq_mhz=_BIN_FREQ,
        active_complex_spectrum=np.zeros(n, dtype=complex),
        active_rms_noise=np.ones(n),
    )


def _trig(window_id, partner_id, side, coherence, n_lines=1):
    return _PendingMerge(
        window_id=window_id,
        partner_id=partner_id,
        edge_side=side,
        edge_coherence=coherence,
        reason=f"edge {coherence:g} on {side} of {window_id}",
        n_lines=n_lines,
    )


def _chain(ids=(4, 5, 6)):
    """Abutting windows ``X-1, X, X+1`` with one line each: 20 and 70 bins
    apart, inside the default 96-bin width cap."""
    x0, x1, x2 = ids
    return _bin_plan(
        [
            (x0, 0, 60, [50]),
            (x1, 61, 120, [70]),
            (x2, 121, 180, [140]),
        ]
    )


def _status(verdicts):
    return {(v.trigger.window_id, v.trigger.edge_side): v.status for v in verdicts}


class TestSelectStructuralMerges:
    """:func:`_select_structural_merges`: which of a round's triggers become
    merge requests -- a window that fits a line, a touching partner, a merged
    window inside the plan's caps, a disjoint set."""

    def test_a_window_flagged_on_both_sides_merges_once_and_defers_once(self):
        plan, peaks = _chain()
        pending = [
            _trig(5, 4, "low", 5.0),
            _trig(5, 6, "high", 3.0),
            _trig(4, 5, "high", 4.0),
            _trig(6, 5, "low", 2.0),
        ]
        requests, verdicts = _select_structural_merges(pending, plan, _bin_ctx(peaks))
        assert len(requests) == 1
        assert {requests[0].window_a_id, requests[0].window_b_id} == {4, 5}
        # The request carries the strongest trigger's reason.
        assert requests[0].reason == "edge 5 on low of 5"
        status = _status(verdicts)
        assert status == {
            (5, "low"): "apply",
            (4, "high"): "apply",
            (5, "high"): "deferred",
            (6, "low"): "deferred",
        }
        deferred = [v for v in verdicts if v.status == "deferred"]
        assert all(
            v.reason == "deferred: window 5 merges with window 4 this round"
            for v in deferred
        )

    def test_one_verdict_per_trigger_in_window_and_side_order(self):
        plan, peaks = _chain()
        pending = [_trig(6, 5, "low", 2.0), _trig(5, 6, "high", 3.0)]
        _requests, verdicts = _select_structural_merges(pending, plan, _bin_ctx(peaks))
        assert [(v.trigger.window_id, v.trigger.edge_side) for v in verdicts] == [
            (5, "high"),
            (6, "low"),
        ]

    def test_the_stronger_edge_takes_the_shared_window(self):
        plan, peaks = _chain()
        pending = [_trig(5, 4, "low", 3.0), _trig(5, 6, "high", 6.0)]
        requests, verdicts = _select_structural_merges(pending, plan, _bin_ctx(peaks))
        assert [{r.window_a_id, r.window_b_id} for r in requests] == [{5, 6}]
        assert _status(verdicts) == {(5, "low"): "deferred", (5, "high"): "apply"}

    def test_a_pair_s_strength_is_its_strongest_trigger(self):
        """Window 6's weak edge toward 5 does not count against the pair when
        window 5's edge toward 6 is the strongest of the round."""
        plan, peaks = _chain()
        pending = [
            _trig(5, 4, "low", 4.0),
            _trig(5, 6, "high", 7.0),
            _trig(6, 5, "low", 1.0),
        ]
        requests, _v = _select_structural_merges(pending, plan, _bin_ctx(peaks))
        assert [{r.window_a_id, r.window_b_id} for r in requests] == [{5, 6}]

    def test_equal_edges_tie_break_on_the_window_pair(self):
        plan, peaks = _chain()
        pending = [_trig(5, 6, "high", 3.0), _trig(5, 4, "low", 3.0)]
        requests, verdicts = _select_structural_merges(pending, plan, _bin_ctx(peaks))
        assert [{r.window_a_id, r.window_b_id} for r in requests] == [{4, 5}]
        assert _status(verdicts) == {(5, "low"): "apply", (5, "high"): "deferred"}

    def test_a_long_chain_pairs_off_into_a_disjoint_set(self):
        # Neighbouring lines are 101 bins apart: over the default 96-point cap,
        # so the MHz cap governs here.
        ids = tuple(range(8))
        spec = [(i, 101 * i, 101 * i + 100, [101 * i + 50]) for i in ids]
        plan, peaks = _bin_plan(spec, _WIDE_PARAMS)
        pending = []
        for i in ids:
            if i > 0:
                pending.append(_trig(i, i - 1, "low", 2.0))
            if i < ids[-1]:
                pending.append(_trig(i, i + 1, "high", 2.0))
        requests, verdicts = _select_structural_merges(pending, plan, _bin_ctx(peaks))
        used = [w for r in requests for w in (r.window_a_id, r.window_b_id)]
        assert len(used) == len(set(used)), "a window is in two requests"
        # Equal edges: the lowest pairs first, so 0/1, 2/3, 4/5, 6/7.
        assert sorted(frozenset((r.window_a_id, r.window_b_id)) for r in requests) == [
            frozenset(p) for p in [(0, 1), (2, 3), (4, 5), (6, 7)]
        ]
        # Every trigger got a verdict, and each deferral names a busy window.
        assert len(verdicts) == len(pending)
        for v in verdicts:
            assert v.status in ("apply", "deferred")
            if v.status == "deferred":
                assert "merges with window" in v.reason

    @pytest.mark.parametrize("gap, qualifies", [(0, True), (1, True), (2, False)])
    def test_a_partner_with_more_than_one_bin_between_is_not_merged(
        self, gap, qualifies
    ):
        plan, peaks = _bin_plan(
            [(0, 0, 100, [90]), (1, 101 + gap, 200 + gap, [110 + gap])]
        )
        requests, verdicts = _select_structural_merges(
            [_trig(0, 1, "high", 4.0)], plan, _bin_ctx(peaks)
        )
        (v,) = verdicts
        if qualifies:
            assert v.status == "apply"
            assert len(requests) == 1
        else:
            assert v.status == "not_merged"
            assert requests == []
            assert v.reason.startswith("not merged: ")
            assert "no window touches it" in v.reason
            assert f"{gap} active-FT bins lie between" in v.reason
            assert "window 1" in v.reason

    def test_a_distant_neighbour_is_not_merged_but_a_touching_one_still_merges(
        self,
    ):
        plan, peaks = _bin_plan(
            [
                (0, 0, 100, [90]),
                (1, 101, 200, [110]),
                (2, 600, 700, [650]),
            ]
        )
        pending = [_trig(1, 0, "low", 3.0), _trig(1, 2, "high", 9.0)]
        requests, verdicts = _select_structural_merges(pending, plan, _bin_ctx(peaks))
        # The far, stronger edge neither merges nor shadows the touching pair.
        assert [{r.window_a_id, r.window_b_id} for r in requests] == [{0, 1}]
        assert _status(verdicts) == {(1, "low"): "apply", (1, "high"): "not_merged"}

    def test_a_merge_over_the_points_cap_is_refused(self):
        params = {**_STAGE4_PARAMS, "max_window_width_points": 96}
        ok, peaks_ok = _bin_plan([(0, 0, 100, [10]), (1, 101, 200, [106])], params)
        over, peaks_over = _bin_plan([(0, 0, 100, [10]), (1, 101, 200, [107])], params)
        _r, v_ok = _select_structural_merges(
            [_trig(0, 1, "high", 4.0)], ok, _bin_ctx(peaks_ok)
        )
        reqs, v_over = _select_structural_merges(
            [_trig(0, 1, "high", 4.0)], over, _bin_ctx(peaks_over)
        )
        assert v_ok[0].status == "apply"
        assert v_over[0].status == "refused"
        assert reqs == []
        assert v_over[0].reason.startswith("refused: ")
        assert "97 active-FT bins" in v_over[0].reason
        assert "width cap of 96" in v_over[0].reason

    def test_a_merge_over_the_mhz_cap_is_refused_when_the_points_cap_is_off(self):
        # 0.6 MHz at 0.0122 MHz per bin: a cap of about 49 bins.
        params = {
            **_STAGE4_PARAMS,
            "max_window_width_points": 0,
            "max_window_width_mhz": 0.6,
        }
        ok, peaks_ok = _bin_plan([(0, 0, 100, [10]), (1, 101, 200, [58])], params)
        over, peaks_over = _bin_plan([(0, 0, 100, [10]), (1, 101, 200, [62])], params)
        _r, v_ok = _select_structural_merges(
            [_trig(0, 1, "high", 4.0)], ok, _bin_ctx(peaks_ok)
        )
        reqs, v_over = _select_structural_merges(
            [_trig(0, 1, "high", 4.0)], over, _bin_ctx(peaks_over)
        )
        assert v_ok[0].status == "apply"
        assert (v_over[0].status, reqs) == ("refused", [])
        assert "52 active-FT bins" in v_over[0].reason
        assert "width cap of 49." in v_over[0].reason

    def test_a_refused_pair_does_not_shadow_a_qualifying_one(self):
        """A refusal takes no part in the disjoint set: window 1's weaker edge
        toward its touching neighbour is not deferred behind a strong one that
        was refused for width."""
        plan, peaks = _bin_plan(
            [
                (0, 0, 100, [60]),
                (1, 101, 200, [110]),
                (2, 201, 400, [300]),
            ]
        )
        pending = [_trig(1, 2, "high", 9.0), _trig(1, 0, "low", 2.0)]
        requests, verdicts = _select_structural_merges(pending, plan, _bin_ctx(peaks))
        assert [{r.window_a_id, r.window_b_id} for r in requests] == [{0, 1}]
        assert _status(verdicts) == {(1, "low"): "apply", (1, "high"): "refused"}
        assert "width cap" in verdicts[1].reason

    def test_a_partner_that_is_not_in_the_plan_fails(self):
        plan, peaks = _chain()
        requests, verdicts = _select_structural_merges(
            [_trig(5, 99, "high", 4.0)], plan, _bin_ctx(peaks)
        )
        assert requests == []
        assert verdicts[0].status == "failed"
        assert verdicts[0].reason.startswith("failed: ")
        assert "unknown window 99" in verdicts[0].reason

    def test_a_window_whose_fit_holds_no_line_is_not_merged(self):
        """The cleanup emptied window 5: its strong edge toward 4 is no evidence
        of a straddling line. It is not merged, and it does not take window 5
        from window 6's merge."""
        plan, peaks = _chain()
        pending = [
            _trig(5, 4, "low", 9.0, n_lines=0),
            _trig(6, 5, "low", 2.0),
        ]
        requests, verdicts = _select_structural_merges(pending, plan, _bin_ctx(peaks))
        assert [{r.window_a_id, r.window_b_id} for r in requests] == [{5, 6}]
        assert _status(verdicts) == {(5, "low"): "not_merged", (6, "low"): "apply"}
        assert verdicts[0].reason == (
            "not merged: residual edge-coherence 9.00 on low side, but the "
            "window's fit holds no line, so no fitted line straddles its boundary "
            "with window 4"
        )

    @pytest.mark.parametrize(
        "max_peaks, status", [(0, "apply"), (3, "apply"), (2, "refused")]
    )
    def test_a_merge_over_the_peak_cap_is_refused(self, max_peaks, status):
        params = {**_STAGE4_PARAMS, "max_peaks_per_window": max_peaks}
        plan, peaks = _bin_plan([(0, 0, 100, [80]), (1, 101, 200, [110, 120])], params)
        reqs, (v,) = _select_structural_merges(
            [_trig(0, 1, "high", 4.0)], plan, _bin_ctx(peaks)
        )
        assert v.status == status
        if status == "refused":
            assert reqs == []
            assert v.reason == (
                "refused: merging with window 1 would hold 3 promoted lines, "
                "over the plan's max_peaks_per_window of 2"
            )
        else:
            assert len(reqs) == 1

    def test_a_pair_stage4_rejected_fails_and_frees_its_windows(self):
        """With 4/5 rejected earlier in the round, 5/6 is no longer deferred
        behind it: it is chosen, and no verdict defers to the failed merge."""
        plan, peaks = _chain()
        pending = [_trig(5, 4, "low", 5.0), _trig(5, 6, "high", 3.0)]
        ctx = _bin_ctx(peaks)
        _r, before = _select_structural_merges(pending, plan, ctx)
        assert _status(before) == {(5, "low"): "apply", (5, "high"): "deferred"}
        requests, verdicts = _select_structural_merges(
            pending, plan, ctx, {(4, 5): "failed: no"}
        )
        assert [{r.window_a_id, r.window_b_id} for r in requests] == [{5, 6}]
        assert _status(verdicts) == {(5, "low"): "failed", (5, "high"): "apply"}
        assert verdicts[0].reason == "failed: no"

    def test_every_reason_that_is_not_applied_starts_with_a_declared_prefix(self):
        plan, peaks = _bin_plan(
            [
                (0, 0, 60, [50]),
                (1, 61, 120, [70]),
                (2, 121, 180, [140]),
                (3, 181, 400, [300]),
                (4, 700, 800, [750]),
            ]
        )
        pending = [
            _trig(0, 1, "high", 2.0, n_lines=0),
            _trig(1, 2, "high", 5.0),
            _trig(2, 1, "low", 4.0),
            _trig(2, 3, "high", 3.0),
            _trig(3, 4, "high", 8.0),
            _trig(4, 9, "high", 1.0),
        ]
        _r, verdicts = _select_structural_merges(pending, plan, _bin_ctx(peaks))
        assert sorted({v.status for v in verdicts}) == [
            "apply",
            "failed",
            "not_merged",
            "refused",
        ]
        for v in verdicts:
            if v.status != "apply":
                assert v.reason.split(": ")[0] in REPLAN_REASON_PREFIXES, v.reason

    def test_the_choice_does_not_depend_on_the_order_triggers_arrive_in(self):
        plan, peaks = _bin_plan(
            [
                (0, 0, 60, [50]),
                (1, 61, 120, [70]),
                (2, 121, 180, [140]),
                (3, 181, 240, [200]),
                (4, 700, 800, [750]),
            ]
        )
        ctx = _bin_ctx(peaks)
        pending = [
            _trig(0, 1, "high", 2.0),
            _trig(1, 0, "low", 2.0),
            _trig(1, 2, "high", 2.0),
            _trig(2, 1, "low", 5.0),
            _trig(2, 3, "high", 2.0),
            _trig(3, 2, "low", 2.0),
            _trig(3, 4, "high", 8.0),
        ]
        expected = _select_structural_merges(pending, plan, ctx)
        requests, verdicts = expected
        # 1/2 is strongest; 0/1 and 2/3 wait behind it; 3/4 is a gap.
        assert [{r.window_a_id, r.window_b_id} for r in requests] == [{1, 2}]
        assert sorted(v.status for v in verdicts) == (
            ["apply"] * 2 + ["deferred"] * 4 + ["not_merged"]
        )
        rng = np.random.default_rng(SEED)
        for _ in range(12):
            shuffled = [pending[i] for i in rng.permutation(len(pending))]
            assert _select_structural_merges(shuffled, plan, ctx) == expected


class TestApplyStructuralMerges:
    """:func:`_apply_structural_merges`: the round's set goes to Stage 4 whole,
    and a rejection names exactly the requests Stage 4 refused."""

    @staticmethod
    def _fake_replan(bad_pairs):
        calls = []

        def fake(plan, requests, ctx):
            calls.append([(r.window_a_id, r.window_b_id) for r in requests])
            for r in requests:
                if _pair_of(r.window_a_id, r.window_b_id) in bad_pairs:
                    raise ValueError(f"cannot merge {r.window_a_id}")
            return WindowPlan(
                windows=list(plan.windows),
                parameters=dict(plan.parameters),
                plan_revision=plan.plan_revision + 1,
            )

        return fake, calls

    def _requests(self):
        return [MergeRequest(0, 1, "a"), MergeRequest(2, 3, "b"), MergeRequest(4, 5)]

    def test_all_requests_go_together_when_stage4_accepts_them(self, monkeypatch):
        fake, calls = self._fake_replan(set())
        monkeypatch.setattr(plan_execution, "_do_replan", fake)
        plan = WindowPlan()
        reqs = self._requests()
        new_plan, failed = _apply_structural_merges(plan, reqs, None)
        assert new_plan is not None and new_plan.plan_revision == 1
        assert failed == {}
        assert len(calls) == 1

    def test_a_rejected_request_is_named_and_no_plan_returned(self, monkeypatch):
        # The caller chooses the round's set again without it (a request that
        # was deferred behind it may now qualify), so no partial plan comes back.
        fake, calls = self._fake_replan({(2, 3)})
        monkeypatch.setattr(plan_execution, "_do_replan", fake)
        new_plan, failed = _apply_structural_merges(
            WindowPlan(), self._requests(), None
        )
        assert new_plan is None
        assert failed == {(2, 3): "failed: cannot merge 2"}
        # Tried cumulatively: the batch, then each request on top of the ones
        # that passed.
        assert calls[1:] == [[(0, 1)], [(0, 1), (2, 3)], [(0, 1), (4, 5)]]

    def test_every_request_rejected_names_each(self, monkeypatch):
        fake, _calls = self._fake_replan({(0, 1), (2, 3), (4, 5)})
        monkeypatch.setattr(plan_execution, "_do_replan", fake)
        new_plan, failed = _apply_structural_merges(
            WindowPlan(), self._requests(), None
        )
        assert new_plan is None
        assert sorted(failed) == [(0, 1), (2, 3), (4, 5)]
        assert all(v.startswith("failed: ") for v in failed.values())


# ---------------------------------------------------------------------------
# Structural replan through execute_plan: rounds, history, parallel == sequential
# ---------------------------------------------------------------------------
def _scripted_dispatch(by_revision, *, reverse=False):
    """A ``_dispatch_structural_round`` that returns the triggers listed for the
    plan's current revision (so a round's request is the test's, not the fit's
    residual statistics)."""

    def dispatch(outcomes, plan, threshold):
        out = list(by_revision.get(plan.plan_revision, []))
        return out[::-1] if reverse else out

    return dispatch


def _exec_fixture(windows):
    """``(plan, ctx, active_ft, rms, line frequencies)`` for ``windows = [(id,
    lo_bin, hi_bin, [line bin])]``: a noisy spectrum with one strong line at each
    listed bin."""
    plan, peaks = _bin_plan(windows, _WIDE_PARAMS)
    rng = np.random.default_rng(SEED + 31)
    sigma = 1.0
    lines = [(p.frequency, _amp_for_snr(100.0, sigma), 0.3) for p in peaks]
    spectrum = _synth_spectrum(_BIN_FREQ, lines)
    spectrum = spectrum + _complex_noise(_BIN_FREQ.size, sigma, rng)
    rms = np.full(_BIN_FREQ.size, sigma)
    ctx = ReplanContext(
        peaks=peaks,
        active_freq_mhz=_BIN_FREQ,
        active_complex_spectrum=spectrum,
        active_rms_noise=rms,
    )
    return (
        plan,
        ctx,
        _make_active_ft(_BIN_FREQ, spectrum),
        rms,
        [p.frequency for p in peaks],
    )


def _run_exec(fixture, **kwargs):
    plan, ctx, active_ft, rms, freqs = fixture
    return execute_plan(
        plan,
        active_ft,
        rms,
        freqs,
        sideband=SIDEBAND,
        acquisition_us=T_US,
        tau0_us=TAU_US,
        replan_context=ctx,
        **kwargs,
    )


def _summary(history):
    return [
        (
            e.triggering_window_id,
            e.partner_window_id,
            e.surviving_window_id,
            e.edge_side,
            e.revision_before,
            e.revision_after,
            e.accepted,
            e.reason.split(":")[0],
        )
        for e in history
    ]


_CHAIN_WINDOWS = [
    (0, 0, 400, [200]),
    (1, 401, 800, [600]),
    (2, 801, 1200, [1000]),
]


class TestStructuralReplanRounds:
    """The structural loop in :func:`execute_plan`, with the round's triggers
    scripted so the merge machinery (selection, Stage 4 replan, refit, history)
    is what is under test."""

    def _chain_script(self, **kw):
        return _scripted_dispatch(
            {
                0: [_trig(1, 0, "low", 5.0), _trig(1, 2, "high", 3.0)],
                1: [_trig(0, 2, "high", 2.0)],
            },
            **kw,
        )

    def test_a_chain_merges_one_pair_a_round_and_a_deferral_merges_next(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            plan_execution, "_dispatch_structural_round", self._chain_script()
        )
        out = _run_exec(_exec_fixture(_CHAIN_WINDOWS))
        first, deferred, second = out.replan_history
        assert (first.accepted, first.revision_before, first.revision_after) == (
            True,
            0,
            1,
        )
        assert first.surviving_window_id == 0
        # The shared window's other request waits, with the reason on record and
        # the plan revision untouched by it.
        assert deferred.accepted is False
        assert deferred.reason == "deferred: window 1 merges with window 0 this round"
        assert deferred.revision_after == deferred.revision_before == 0
        # The survivor's edge still flags in round 2: the deferred merge happens.
        assert (second.accepted, second.revision_before, second.revision_after) == (
            True,
            1,
            2,
        )
        assert out.final_plan_revision == 2
        assert [w.window_id for w in out.final_plan.windows] == [0]
        assert set(out.window_outcomes) == {0}
        assert out.window_outcomes[0].fit.n_peaks == 3

    def test_events_that_are_not_accepted_leave_the_revision_alone(self, monkeypatch):
        monkeypatch.setattr(
            plan_execution, "_dispatch_structural_round", self._chain_script()
        )
        out = _run_exec(_exec_fixture(_CHAIN_WINDOWS))
        for e in out.replan_history:
            if e.accepted:
                assert e.revision_after == e.revision_before + 1
            else:
                assert e.revision_after == e.revision_before

    def test_a_refusal_is_recorded_once_across_rounds(self, monkeypatch):
        """Windows 2 and 3 are 100+ bins apart: the trigger is refused in round
        1 and asked again in round 2 (windows 0 and 1 merged meanwhile) -- the
        unchanged refusal is not recorded twice."""
        windows = [
            (0, 0, 400, [200]),
            (1, 401, 800, [600]),
            (2, 900, 1150, [1000]),
            (3, 1300, 1600, [1450]),
        ]
        refused = _trig(2, 3, "high", 4.0)
        monkeypatch.setattr(
            plan_execution,
            "_dispatch_structural_round",
            _scripted_dispatch({0: [_trig(0, 1, "high", 5.0), refused], 1: [refused]}),
        )
        out = _run_exec(_exec_fixture(windows))
        merge, refusal = out.replan_history
        assert (merge.accepted, merge.surviving_window_id) == (True, 0)
        assert refusal.accepted is False
        assert "no window touches it" in refusal.reason
        assert refusal.revision_after == refusal.revision_before == 0
        assert out.final_plan_revision == 1
        assert sorted(w.window_id for w in out.final_plan.windows) == [0, 2, 3]

    def test_a_refusal_alone_stops_the_loop_and_changes_nothing(self, monkeypatch):
        windows = [(0, 0, 400, [200]), (1, 600, 1000, [800])]
        monkeypatch.setattr(
            plan_execution,
            "_dispatch_structural_round",
            _scripted_dispatch({0: [_trig(0, 1, "high", 4.0)]}),
        )
        fixture = _exec_fixture(windows)
        out = _run_exec(fixture)
        (ev,) = out.replan_history
        assert not ev.accepted
        assert out.final_plan_revision == 0
        assert out.final_plan is fixture[0]
        assert set(out.window_outcomes) == {0, 1}

    def test_a_request_stage4_rejects_is_dropped_and_the_round_goes_on(
        self, monkeypatch
    ):
        windows = [
            (0, 0, 400, [200]),
            (1, 401, 800, [600]),
            (2, 900, 1150, [1000]),
            (3, 1151, 1600, [1450]),
        ]
        real = plan_execution._do_replan

        def picky(plan, requests, ctx):
            for r in requests:
                if _pair_of(r.window_a_id, r.window_b_id) == (2, 3):
                    raise ValueError("synthetic stage 4 rejection")
            return real(plan, requests, ctx)

        monkeypatch.setattr(plan_execution, "_do_replan", picky)
        monkeypatch.setattr(
            plan_execution,
            "_dispatch_structural_round",
            _scripted_dispatch(
                {0: [_trig(0, 1, "high", 5.0), _trig(2, 3, "high", 4.0)]}
            ),
        )
        out = _run_exec(_exec_fixture(windows))
        good, bad = out.replan_history
        assert good.accepted and good.surviving_window_id == 0
        assert (good.revision_before, good.revision_after) == (0, 1)
        assert not bad.accepted
        assert bad.reason == "failed: synthetic stage 4 rejection"
        assert bad.revision_after == bad.revision_before
        assert out.final_plan_revision == 1
        assert sorted(w.window_id for w in out.final_plan.windows) == [0, 2, 3]
        assert set(out.window_outcomes) == {0, 2, 3}

    def test_every_request_rejected_records_each_and_stops(self, monkeypatch):
        windows = [(0, 0, 400, [200]), (1, 401, 800, [600])]

        def refuse(plan, requests, ctx):
            raise ValueError("synthetic stage 4 rejection")

        monkeypatch.setattr(plan_execution, "_do_replan", refuse)
        monkeypatch.setattr(
            plan_execution,
            "_dispatch_structural_round",
            _scripted_dispatch({0: [_trig(0, 1, "high", 5.0)]}),
        )
        out = _run_exec(_exec_fixture(windows))
        (ev,) = out.replan_history
        assert not ev.accepted
        assert ev.reason == "failed: synthetic stage 4 rejection"
        assert out.final_plan_revision == 0

    def test_a_merge_deferred_behind_a_rejected_one_goes_ahead_that_round(
        self, monkeypatch
    ):
        """Window 1's stronger edge asks for 0/1, so 1/2 waits behind it; Stage 4
        rejects 0/1, and 1/2 is merged in the same round instead of being
        recorded as deferred behind a merge that never happened."""
        real = plan_execution._do_replan

        def picky(plan, requests, ctx):
            for r in requests:
                if _pair_of(r.window_a_id, r.window_b_id) == (0, 1):
                    raise ValueError("synthetic stage 4 rejection")
            return real(plan, requests, ctx)

        monkeypatch.setattr(plan_execution, "_do_replan", picky)
        monkeypatch.setattr(
            plan_execution,
            "_dispatch_structural_round",
            _scripted_dispatch(
                {0: [_trig(1, 0, "low", 5.0), _trig(1, 2, "high", 3.0)]}
            ),
        )
        out = _run_exec(_exec_fixture(_CHAIN_WINDOWS))
        rejected, merged = out.replan_history
        assert (rejected.triggering_window_id, rejected.partner_window_id) == (1, 0)
        assert not rejected.accepted
        assert rejected.reason == "failed: synthetic stage 4 rejection"
        assert rejected.revision_after == rejected.revision_before == 0
        assert (merged.triggering_window_id, merged.partner_window_id) == (1, 2)
        assert merged.accepted
        assert (merged.revision_before, merged.revision_after) == (0, 1)
        assert not [e for e in out.replan_history if "deferred" in e.reason]
        assert out.final_plan_revision == 1
        assert sorted(w.window_id for w in out.final_plan.windows) == [0, 1]

    def test_a_stage4_rejection_is_recorded_once_across_rounds(self, monkeypatch):
        windows = [
            (0, 0, 400, [200]),
            (1, 401, 800, [600]),
            (2, 900, 1150, [1000]),
            (3, 1151, 1600, [1450]),
        ]
        real = plan_execution._do_replan

        def picky(plan, requests, ctx):
            for r in requests:
                if _pair_of(r.window_a_id, r.window_b_id) == (2, 3):
                    raise ValueError("synthetic stage 4 rejection")
            return real(plan, requests, ctx)

        bad = _trig(2, 3, "high", 4.0)
        monkeypatch.setattr(plan_execution, "_do_replan", picky)
        monkeypatch.setattr(
            plan_execution,
            "_dispatch_structural_round",
            _scripted_dispatch({0: [_trig(0, 1, "high", 5.0), bad], 1: [bad]}),
        )
        out = _run_exec(_exec_fixture(windows))
        assert _summary(out.replan_history) == [
            (0, 1, 0, "high", 0, 1, True, "edge 5 on high of 0"),
            (2, 3, 2, "high", 0, 0, False, "failed"),
        ]
        assert out.final_plan_revision == 1

    def test_a_merge_drops_the_records_of_the_windows_it_refits(self, monkeypatch):
        """Thaw, rescue and cleanup records of the merged windows (the survivor
        and the absorbed one) are dropped before the re-fit records its own;
        an untouched window keeps its records. Each walk is tagged with
        synthetic records so which walk wrote a record is visible."""
        real_walk = plan_execution._walk_windows_parallel

        def walk(plan, order, **kw):
            real_walk(plan, order, **kw)
            phase = kw["phase"]
            for wid in order:
                kw["thaw_history"].append(
                    ThawEvent(wid, -1, -1, 0.0, "low", 0.0, 0.0, False, phase)
                )
                kw["rescue_history"].append(
                    RescueEvent(
                        wid, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, False, phase
                    )
                )
                kw["cleanup_history"].append({"window_id": wid, "phase": phase})

        monkeypatch.setattr(plan_execution, "_walk_windows_parallel", walk)
        monkeypatch.setattr(
            plan_execution,
            "_dispatch_structural_round",
            _scripted_dispatch({0: [_trig(0, 1, "high", 5.0)]}),
        )
        out = _run_exec(_exec_fixture(_CHAIN_WINDOWS))
        assert out.final_plan_revision == 1
        expected = [(2, "initial"), (0, "replan")]
        thaws = [(e.dependent_window_id, e.reason) for e in out.thaw_history]
        rescues = [(e.window_id, e.reason) for e in out.rescue_history]
        cleanups = [(r["window_id"], r["phase"]) for r in out.cleanup_history]
        assert thaws == rescues == cleanups == expected

    @needs_fork
    def test_parallel_and_sequential_walks_record_the_same_history(self, monkeypatch):
        # The scripted triggers arrive in opposite orders (the order a parallel
        # walk's windows finish in is not fixed); the recorded history, merged
        # windows and refit results are the same.
        runs = []
        for jobs, reverse in ((1, False), (2, True)):
            monkeypatch.setattr(
                plan_execution,
                "_dispatch_structural_round",
                self._chain_script(reverse=reverse),
            )
            out = _run_exec(_exec_fixture(_CHAIN_WINDOWS), jobs=jobs)
            runs.append(out)
        seq, par = runs
        assert _summary(seq.replan_history) == _summary(par.replan_history)
        assert [e.reason for e in seq.replan_history] == [
            e.reason for e in par.replan_history
        ]
        assert seq.final_plan_revision == par.final_plan_revision == 2
        assert (
            seq.final_plan.windows[0].freq_range == par.final_plan.windows[0].freq_range
        )

    @pytest.mark.parametrize(
        "uncovered, merged", [(1, True), (7, False)], ids=["one-bin", "seven-bins"]
    )
    def test_a_real_flag_merges_across_a_bin_but_not_across_a_gap(
        self, uncovered, merged
    ):
        """No scripting: the fixture of
        ``TestStructuralReplan.test_merge_fires_when_feature_crosses_boundary``
        with window B's low edge moved up the grid so that ``uncovered`` bins lie
        between the two windows. B holds its own line, its low edge still sees
        the strong line's skirt, and it has no contributor to thaw. One
        uncovered bin leaves B touching A and the merge happens; seven leave
        spectrum neither window fits, so the flag is recorded as not merged and
        the plan is untouched."""
        grid = np.arange(36100.0, 36120.0 + DF_MHZ / 2, DF_MHZ)
        top = int(np.argmin(np.abs(grid - 36105.0)))
        out, freqs = TestStructuralReplan()._straddle_run(
            b_low_mhz=float(grid[top + uncovered + 1])
        )
        np.testing.assert_array_equal(freqs, grid)
        (ev,) = out.replan_history
        assert (ev.triggering_window_id, ev.edge_side) == (1, "low")
        if merged:
            assert ev.accepted
            assert out.final_plan_revision == 1
            assert [w.window_id for w in out.final_plan.windows] == [0]
        else:
            assert not ev.accepted
            assert ev.reason.startswith("not merged: ")
            assert f"{uncovered} active-FT bins lie between" in ev.reason
            assert ev.revision_after == ev.revision_before == 0
            assert out.final_plan_revision == 0
            assert set(out.window_outcomes) == {0, 1}


# ---------------------------------------------------------------------------
# Leakage-wing baseline trigger (_apply_baseline_to_outcome)
# ---------------------------------------------------------------------------
def _baseline_outcome(u, z, sigma, peaks, *, tau=TAU_US):
    """Minimal WindowOutcome (no frozen background) for baseline-trigger tests.

    The fit is the converged peaks-only model; the outcome's edge-coherence is
    measured on that residual so ``_apply_baseline_to_outcome`` reads a real
    trigger value.
    """
    from ftmwpipeline.fitting.plan_execution import residual_edge_coherence
    from ftmwpipeline.fitting.window_fit import (
        ConservativeFitResult,
        fit_window,
    )

    fit = fit_window(u, z, sigma, peaks, tau, T_US, fit_tau=False)
    bg = np.zeros(u.size, dtype=np.complex128)
    full_resid = z - fit.fitted_spectrum
    low, high = residual_edge_coherence(full_resid, sigma, band_m=16)
    inner = fit
    outcome = WindowOutcome(
        window_id=1,
        fit=ConservativeFitResult(inner, [], []),
        fixed_peaks=[],
        offset_grid_mhz=u,
        complex_spectrum=z,
        rms_noise=sigma,
        background=bg,
        full_fitted_spectrum=fit.fitted_spectrum,
        full_residual=full_resid,
        edge_coherence_low=low,
        edge_coherence_high=high,
    )
    outcome._center_mhz = PROBE_MHZ  # type: ignore[attr-defined]
    return outcome


def test_baseline_fires_on_coherent_wing():
    """A strong coherent wing residual (edge-coh > threshold) triggers the
    baseline; the refit absorbs it and records the audit fields."""
    from ftmwpipeline.fitting.peak_model import ModelPeak, model_spectrum
    from ftmwpipeline.fitting.plan_execution import _apply_baseline_to_outcome

    u = np.arange(-160, 161) * 0.0122
    u_s = float(np.max(np.abs(u)))
    line = ModelPeak(amplitude=6.0, offset_mhz=0.2, phase=0.5)
    x = u / u_s
    wing = (0.6 - 0.4j) * x**0  # const complex wing
    z = model_spectrum(u, [line], TAU_US, T_US) + wing
    sigma = np.full(u.size, 0.02)

    outcome = _baseline_outcome(u, z, sigma, [ModelPeak(5.0, 0.0, 0.0)])
    s_coh_before = max(outcome.edge_coherence_low, outcome.edge_coherence_high)
    assert s_coh_before > 3.5  # the wing makes the edge coherent

    fired = _apply_baseline_to_outcome(
        outcome,
        acquisition_us=T_US,
        residual_edge_m=16,
        baseline_order=0,
        baseline_edge_threshold=3.5,
        baseline_smooth_threshold=50.0,
        tau0_us=TAU_US,
        conservative_kwargs={},
    )
    assert fired is True
    assert outcome.baseline_applied is True
    assert outcome.baseline_order == 0
    assert outcome.baseline_edge_coherence == pytest.approx(s_coh_before)
    assert outcome.baseline_coeffs is not None
    # Wing absorbed -> residual edge-coherence drops back toward the null.
    assert max(outcome.edge_coherence_low, outcome.edge_coherence_high) < s_coh_before
    # Line still present and recovered.
    assert outcome.fit.fit.peaks[0].amplitude == pytest.approx(6.0, abs=2e-2)


def test_baseline_does_not_fire_below_threshold():
    """A clean window (edge-coh ~ null) does not trigger the baseline."""
    from ftmwpipeline.fitting.peak_model import ModelPeak
    from ftmwpipeline.fitting.plan_execution import _apply_baseline_to_outcome

    u = np.arange(-160, 161) * 0.0122
    line = ModelPeak(amplitude=6.0, offset_mhz=0.15, phase=0.3)
    from ftmwpipeline.fitting.peak_model import model_spectrum

    rng = np.random.default_rng(7)
    sigma = np.full(u.size, 0.02)
    noise = rng.normal(0, sigma / np.sqrt(2)) + 1j * rng.normal(0, sigma / np.sqrt(2))
    z = model_spectrum(u, [line], TAU_US, T_US) + noise

    outcome = _baseline_outcome(u, z, sigma, [ModelPeak(5.0, 0.0, 0.0)])
    assert max(outcome.edge_coherence_low, outcome.edge_coherence_high) < 3.5

    fired = _apply_baseline_to_outcome(
        outcome,
        acquisition_us=T_US,
        residual_edge_m=16,
        baseline_order=0,
        baseline_edge_threshold=3.5,
        baseline_smooth_threshold=50.0,
        tau0_us=TAU_US,
        conservative_kwargs={},
    )
    assert fired is False
    assert outcome.baseline_applied is False
    assert outcome.baseline_coeffs is None


# ---------------------------------------------------------------------------
# Cross-window parallel walk: levelization + scientific equivalence
# ---------------------------------------------------------------------------
from ftmwpipeline.fitting import plan_execution as _pe  # noqa: E402


class TestLevelize:
    """Antichain layering derives ordering from non-edge_free contributors."""

    def _win(self, wid, contributors=None):
        return FitWindow(
            window_id=wid,
            freq_range=(0.0, 1.0),
            free_peak_indices=[wid],
            fixed_contributors=contributors or [],
            batch=0,
        )

    def test_independent_windows_share_one_level(self):
        by_id = {w.window_id: w for w in (self._win(0), self._win(1), self._win(2))}
        levels = _pe._levelize([0, 1, 2], by_id)
        assert levels == [[0, 1, 2]]

    def test_chain_via_contributors_layers_in_order(self):
        w0 = self._win(0)
        w1 = self._win(1, [FixedContributor(0, 0, 0.5, freeze_eligible=True)])
        w2 = self._win(2, [FixedContributor(1, 1, 0.5, freeze_eligible=True)])
        by_id = {w.window_id: w for w in (w0, w1, w2)}
        levels = _pe._levelize([0, 1, 2], by_id)
        assert levels == [[0], [1], [2]]

    def test_edge_free_contributor_imposes_no_ordering(self):
        # window 1 depends on 0 only through an edge_free contributor -> no edge.
        w0 = self._win(0)
        w1 = self._win(1, [FixedContributor(0, 0, 0.5, True, edge_free=True)])
        by_id = {w.window_id: w for w in (w0, w1)}
        levels = _pe._levelize([0, 1], by_id)
        assert levels == [[0, 1]]

    def test_diamond_layers_correctly(self):
        # 0 -> {1, 2} -> 3 (3 depends on both 1 and 2).
        w0 = self._win(0)
        w1 = self._win(1, [FixedContributor(0, 0, 0.5, True)])
        w2 = self._win(2, [FixedContributor(0, 0, 0.5, True)])
        w3 = self._win(
            3,
            [
                FixedContributor(0, 1, 0.5, True),
                FixedContributor(0, 2, 0.5, True),
            ],
        )
        by_id = {w.window_id: w for w in (w0, w1, w2, w3)}
        levels = _pe._levelize([0, 1, 2, 3], by_id)
        assert levels == [[0], [1, 2], [3]]

    def test_dependency_edges_folded_in(self):
        # No contributors, but an explicit dependency edge (child=1, parent=0).
        by_id = {w.window_id: w for w in (self._win(0), self._win(1))}
        levels = _pe._levelize([0, 1], by_id, dependency_edges=[(1, 0)])
        assert levels == [[0], [1]]


class TestParallelWalkEquivalence:
    """The fork-per-level pool yields the same fit as the sequential walk."""

    def _build_plan(self):
        """Two independent strong lines (level 0, width 2) + a weak dependent."""
        sigma = 1.0
        f0, f1, f2 = 36100.0, 36200.0, 36105.0
        freq_array = np.arange(f0 - 5.0, f1 + 5.0, DF_MHZ)
        spectrum = _synth_spectrum(
            freq_array,
            [
                (f0, _amp_for_snr(300.0, sigma), 0.3),
                (f1, _amp_for_snr(280.0, sigma), 1.1),
                (f2, _amp_for_snr(60.0, sigma), 2.4),
            ],
        )
        rng = np.random.default_rng(SEED + 77)
        spectrum = spectrum + _complex_noise(freq_array.size, sigma, rng)
        rms_noise = np.full(freq_array.size, sigma)
        windows = [
            FitWindow(0, (f0 - 0.6, f0 + 0.6), free_peak_indices=[0], batch=0),
            FitWindow(1, (f1 - 0.6, f1 + 0.6), free_peak_indices=[1], batch=0),
            FitWindow(
                2,
                (f2 - 0.6, f2 + 0.6),
                free_peak_indices=[2],
                fixed_contributors=[FixedContributor(0, 0, f0, True)],
                batch=1,
            ),
        ]
        plan = WindowPlan(
            windows=windows,
            dependency_edges=[(2, 0)],
            topological_order=[0, 1, 2],
        )
        return plan, freq_array, spectrum, rms_noise, [f0, f1, f2]

    def _fit(self, workers, *, with_point_map: bool = False):
        plan, freqs, spec, noise, peak_freqs = self._build_plan()
        old = _pe._FIT_WINDOW_WORKERS
        _pe._FIT_WINDOW_WORKERS = workers
        extra: dict = {}
        if with_point_map:
            extra = dict(probe_freq_mhz=PROBE_MHZ, sample_dt_us=0.05)
        try:
            return execute_plan(
                plan,
                _make_active_ft(freqs, spec),
                noise,
                peak_freqs,
                sideband=SIDEBAND,
                acquisition_us=T_US,
                tau0_us=TAU_US,
                **extra,
            )
        finally:
            _pe._FIT_WINDOW_WORKERS = old

    @pytest.mark.skipif(
        "fork" not in __import__("multiprocessing").get_all_start_methods(),
        reason="requires fork start method",
    )
    def test_parallel_matches_sequential(self):
        seq = self._fit(1)  # in-process sequential reference
        par = self._fit(2)  # force the fork pool (level 0 has width 2)
        assert set(seq.window_outcomes) == set(par.window_outcomes) == {0, 1, 2}
        for wid in (0, 1, 2):
            so, po = seq.window_outcomes[wid], par.window_outcomes[wid]
            assert so.fit.n_peaks == po.fit.n_peaks
            sp = sorted(so.fit.peaks, key=lambda p: p.offset_mhz)
            pp = sorted(po.fit.peaks, key=lambda p: p.offset_mhz)
            for a, b in zip(sp, pp):
                assert a.offset_mhz == pytest.approx(b.offset_mhz, abs=1e-6)
                assert a.amplitude == pytest.approx(b.amplitude, rel=1e-6)
                assert a.phase == pytest.approx(b.phase, abs=1e-6)
        # The dependent (window 2) still saw window 0 as a frozen contributor.
        assert len(par.window_outcomes[2].fixed_peaks) == 1
        assert par.window_outcomes[2].fixed_peaks[0].primary_window_id == 0

    @pytest.mark.skipif(
        "fork" not in __import__("multiprocessing").get_all_start_methods(),
        reason="requires fork start method",
    )
    def test_parallel_and_sequential_agree_on_peak_uid(self):
        """The PointMap threaded via ``probe_freq_mhz``/``sample_dt_us`` must
        actually reach a worker process, not silently read back as None only
        under the parallel walk. Window 2 (batch 1, a dependent scheduled
        after window 0 via the DAG walk) is included deliberately -- it is
        the case that exercises the worker-dispatch boundary, not just the
        trivial width-1 level."""
        seq = self._fit(1, with_point_map=True)
        par = self._fit(2, with_point_map=True)
        assert set(seq.window_outcomes) == set(par.window_outcomes) == {0, 1, 2}
        for wid in (0, 1, 2):
            so, po = seq.window_outcomes[wid], par.window_outcomes[wid]
            assert so.fit.n_peaks == po.fit.n_peaks > 0
            sp = sorted(so.fit.peaks, key=lambda p: p.offset_mhz)
            pp = sorted(po.fit.peaks, key=lambda p: p.offset_mhz)
            for a, b in zip(sp, pp):
                assert a.peak_uid is not None
                assert b.peak_uid is not None
                assert a.peak_uid == b.peak_uid


# ---------------------------------------------------------------------------
# execute_plan's PointMap threading (P3, second landing). The seeding chain
# (window_fit._seed_peak, reached via conservative_fit) never sees
# center_mhz / sideband / probe_freq_mhz -- execute_plan builds a per-window
# PointMap from those (plus n_active / sample_dt_us) the moment a window's
# frame is known and rides it into every seed constructor via the
# conservative-kwargs bag.
# ---------------------------------------------------------------------------
class TestExecutePlanPeakUidStamping:
    def _single_line_plan(self):
        sigma = 1.0
        f0 = 36100.0
        freq_array = np.arange(f0 - 5.0, f0 + 5.0, DF_MHZ)
        spectrum = _synth_spectrum(freq_array, [(f0, _amp_for_snr(200.0, sigma), 0.4)])
        rng = np.random.default_rng(SEED + 41)
        spectrum = spectrum + _complex_noise(freq_array.size, sigma, rng)
        rms_noise = np.full(freq_array.size, sigma)
        win = FitWindow(0, (f0 - 1.0, f0 + 1.0), free_peak_indices=[0], batch=0)
        plan = WindowPlan(windows=[win], dependency_edges=[], topological_order=[0])
        active_ft = _make_active_ft(freq_array, spectrum)
        return plan, active_ft, rms_noise, f0

    def test_seeded_peak_gets_the_expected_uid(self):
        """The K=1 seed offset equals the Stage-3 candidate (a clean, strong,
        unblended line does not escalate), so the stamp is pinned exactly
        against PointMap.stamp of that known candidate offset -- not merely
        checked for presence."""
        sample_dt_us = 0.05
        plan, active_ft, rms_noise, f0 = self._single_line_plan()
        out = execute_plan(
            plan,
            active_ft,
            rms_noise,
            [f0],
            sideband=SIDEBAND,
            acquisition_us=T_US,
            tau0_us=TAU_US,
            probe_freq_mhz=PROBE_MHZ,
            sample_dt_us=sample_dt_us,
        )
        outcome = out.window_outcomes[0]
        assert outcome.fit.n_peaks == 1
        pk = outcome.fit.peaks[0]
        assert pk.peak_uid is not None

        center = _window_center_mhz(outcome)
        s = sideband_sign(SIDEBAND)
        candidate_offset = s * (f0 - center)
        point_map = PointMap.from_frame(
            center, SIDEBAND, PROBE_MHZ, active_ft.n_active, sample_dt_us
        )
        assert pk.peak_uid == point_map.stamp(candidate_offset)

    def test_without_probe_freq_mhz_or_sample_dt_us_peaks_stay_unstamped(self):
        """The default -- unset probe_freq_mhz/sample_dt_us -- leaves every
        peak unstamped, matching pre-landing behavior exactly."""
        plan, active_ft, rms_noise, f0 = self._single_line_plan()
        out = execute_plan(
            plan,
            active_ft,
            rms_noise,
            [f0],
            sideband=SIDEBAND,
            acquisition_us=T_US,
            tau0_us=TAU_US,
        )
        outcome = out.window_outcomes[0]
        assert outcome.fit.n_peaks == 1
        assert outcome.fit.peaks[0].peak_uid is None


# ---------------------------------------------------------------------------
# refit_outcome: live-outcome single-window refit primitive
# ---------------------------------------------------------------------------
class TestRefitOutcome:
    """The in-walk refit primitive: identity reproduction + edit semantics."""

    def _two_line_window_outcome(self, f0: float, f1: float) -> WindowOutcome:
        """Fit one window holding two well-separated lines; return its outcome.

        The single-window walk stashes the refit context on the outcome, which
        is exactly what :func:`refit_outcome` consumes.
        """
        sigma = 1.0
        center = 0.5 * (f0 + f1)
        freq_array = np.arange(center - 5.0, center + 5.0, DF_MHZ)
        a0 = _amp_for_snr(120.0, sigma)
        a1 = _amp_for_snr(80.0, sigma)
        spectrum = _synth_spectrum(freq_array, [(f0, a0, 0.3), (f1, a1, 1.7)])
        rng = np.random.default_rng(SEED + 7)
        spectrum = spectrum + _complex_noise(freq_array.size, sigma, rng)
        rms_noise = np.full(freq_array.size, sigma)
        win = FitWindow(
            window_id=0,
            freq_range=(center - 4.0, center + 4.0),
            free_peak_indices=[0, 1],
            fixed_contributors=[],
            batch=0,
        )
        plan = WindowPlan(windows=[win], dependency_edges=[], topological_order=[0])
        out = execute_plan(
            plan,
            _make_active_ft(freq_array, spectrum),
            rms_noise,
            [f0, f1],
            sideband=SIDEBAND,
            acquisition_us=T_US,
            tau0_us=TAU_US,
        )
        return out.window_outcomes[0]

    def test_identity_refit_reproduces_the_fit(self):
        """An identity refit (no edits) reproduces the converged peak set."""
        outcome = self._two_line_window_outcome(36100.0, 36106.0)
        before = sorted(outcome.fit.peaks, key=lambda p: p.offset_mhz)
        refit = refit_outcome(outcome)
        after = sorted(refit.fit.peaks, key=lambda p: p.offset_mhz)
        assert len(after) == len(before)
        for a, b in zip(after, before):
            assert a.offset_mhz == pytest.approx(b.offset_mhz, abs=2e-4)
            assert a.amplitude == pytest.approx(b.amplitude, rel=2e-3)

    def test_peak_uid_survives_refit_while_offset_moves(self):
        """P3 propagation: refit_outcome rebuilds ModelPeak from inner.peaks
        (plan_execution.py ~1507) -- it must carry each seed's existing
        peak_uid forward, and must NOT mint a fresh one from the refit's
        (moved) offset. Pins both halves: the carry, and the negative case
        that catches a re-derivation regression.

        The seed offsets are deliberately perturbed off their converged
        values before the refit (standing in for a warm-started refit whose
        seed is stale -- e.g. a neighbor's edit shifted this window's
        starting point): a plain identity refit reproduces the converged
        position almost exactly (see test_identity_refit_reproduces_the_fit's
        own 2e-4 MHz tolerance), which is too small a move to reliably
        distinguish "carried" from "re-derived" against rounding."""
        from ftmwpipeline.fitting.active_ft import peak_uid_from_offset

        outcome = self._two_line_window_outcome(36100.0, 36106.0)
        peaks = sorted(outcome.fit.peaks, key=lambda p: p.offset_mhz)
        assert len(peaks) == 2
        # Stand in for birth-time stamps from an earlier fit in this lineage
        # (arbitrary values a long way from anything the point-space formula
        # would produce for these offsets, so a re-derivation regression
        # cannot accidentally pass).
        peaks[0].peak_uid = 111111
        peaks[1].peak_uid = 222222
        # Perturb the seed well off its converged offset -- large enough
        # that the re-converged fit is a materially different position, not
        # a rounding-level nudge.
        perturbation_mhz = 0.05
        seed_offsets = {}
        for p in peaks:
            p.offset_mhz += perturbation_mhz
            seed_offsets[p.peak_uid] = p.offset_mhz
        center_mhz = _window_center_mhz(outcome)

        refit = refit_outcome(outcome)
        after = refit.fit.peaks
        assert {p.peak_uid for p in after} == {111111, 222222}

        n_active, sample_dt_us = 1000, 0.05
        for p in after:
            # The carry: the identifier is exactly the value stamped before
            # this refit, unchanged by it.
            assert p.peak_uid in (111111, 222222)
            # Confirm the refit actually moved this peak away from its
            # (perturbed) seed -- otherwise this case proves nothing.
            moved_mhz = abs(p.offset_mhz - seed_offsets[p.peak_uid])
            assert moved_mhz > 0.01, (
                f"refit only moved the peak {moved_mhz:.5f} MHz -- too small "
                "to distinguish carry from re-derivation"
            )
            # The negative case that matters most: the identifier is NOT the
            # point-space position recomputed from the fitted (moved) offset.
            recomputed_from_fit = peak_uid_from_offset(
                p.offset_mhz, center_mhz, SIDEBAND, PROBE_MHZ, n_active, sample_dt_us
            )
            assert p.peak_uid != recomputed_from_fit

    def test_remove_drops_the_targeted_peak(self):
        """A remove edit drops the nearest converged peak; the survivor holds."""
        outcome = self._two_line_window_outcome(36100.0, 36106.0)
        peaks = sorted(outcome.fit.peaks, key=lambda p: p.offset_mhz)
        assert len(peaks) == 2
        # Remove the lower-offset peak by its offset.
        drop_off = peaks[0].offset_mhz
        keep_off = peaks[1].offset_mhz
        refit = refit_outcome(outcome, remove_offsets=[drop_off])
        assert refit.fit.n_peaks == 1
        assert refit.fit.peaks[0].offset_mhz == pytest.approx(keep_off, abs=0.05)

    def test_requires_stashed_context(self):
        """refit_outcome needs a walk-produced outcome (refit context stashed)."""
        outcome = self._two_line_window_outcome(36100.0, 36106.0)
        # Strip the stashed context -> a clear error, not a silent wrong fit.
        delattr(outcome, "_ck_for_window")
        with pytest.raises(ValueError, match="refit context"):
            refit_outcome(outcome)


# ---------------------------------------------------------------------------
# F-1: _refresh_rescue_candidates
# ---------------------------------------------------------------------------
def test_refresh_rescue_candidates_destale_and_prune():
    """F-1: the persisted rescue ledger is re-measured on the final residual.

    A candidate that still has a real residual peak keeps its refreshed SNR; a
    candidate whose location is flat on the converged residual is dropped before
    it can reach the Stage 6 ledger.
    """
    from ftmwpipeline.fitting.plan_execution import (
        RescueEvent,
        _refresh_rescue_candidates,
    )
    from ftmwpipeline.fitting.residual_screening import ResidualPeakCandidate
    from ftmwpipeline.fitting.window_fit import (
        ConservativeFitResult,
        fit_window,
    )

    u = np.arange(-160, 161) * DF_MHZ
    sigma = np.full(u.size, 0.02)
    # The final residual carries one genuine leftover peak at offset +0.25 MHz
    # and nothing at -0.25 MHz. A localized bump (not a finite-T lineshape)
    # keeps the synthetic residual free of boxcar sidelobes elsewhere.
    real_off = 0.25
    full_resid = (0.3 * np.exp(-(((u - real_off) / 0.04) ** 2))).astype(np.complex128)
    # A converged single-line fit -- only its tau / shape are read by the refresh.
    fit = fit_window(
        u, full_resid, sigma, [ModelPeak(1.0, 0.0, 0.0)], TAU_US, T_US, fit_tau=False
    )
    outcome = WindowOutcome(
        window_id=1,
        fit=ConservativeFitResult(fit, [], []),
        fixed_peaks=[],
        offset_grid_mhz=u,
        complex_spectrum=full_resid,
        rms_noise=sigma,
        background=np.zeros(u.size, dtype=np.complex128),
        full_fitted_spectrum=np.zeros(u.size, dtype=np.complex128),
        full_residual=full_resid,
    )

    def _cand(off: float, snr: float) -> ResidualPeakCandidate:
        return ResidualPeakCandidate(
            bin_index=0,
            frequency_mhz=off,
            magnitude=snr * 0.02 / np.sqrt(2.0),
            snr=snr,
            prominence_sigma_c=snr,
            nearest_existing_detection_index=None,
            nearest_existing_freq_mhz=None,
            nearest_existing_separation_mhz=None,
            near_existing=False,
        )

    # The round records a stale candidate (-0.25, no final peak) and the real
    # one (+0.25) with an inflated rescue-time SNR of 40.
    outcome.rescue_events = [
        RescueEvent(
            window_id=1,
            round_idx=0,
            n_initial_peaks=1,
            n_candidates=2,
            n_rescue_added=0,
            n_pruned_by_knockout=0,
            n_pruned_rescue_origin=0,
            n_merged=0,
            chi2_before=1.0,
            chi2_after=1.0,
            tau_us_before=TAU_US,
            tau_us_after=TAU_US,
            accepted=False,
            reason="",
            candidates=[_cand(-0.25, 11.0), _cand(real_off, 40.0)],
        )
    ]

    _refresh_rescue_candidates(outcome, acquisition_us=T_US, conservative_kwargs={})

    cands = outcome.rescue_events[0].candidates
    assert len(cands) == 1  # the stale candidate is pruned
    kept = cands[0]
    assert kept.frequency_mhz == pytest.approx(real_off)
    # SNR refreshed to the measured final-residual value (the inflated 40 is gone).
    assert kept.snr != pytest.approx(40.0)
    assert kept.snr > 4.0  # a real peak well above the detector bar


# ---------------------------------------------------------------------------
# F-2: _add_from_convergence
# ---------------------------------------------------------------------------
def _one_line_outcome(f0: float, snr0: float = 120.0) -> WindowOutcome:
    """Fit a single-line spectrum and return the walk-produced outcome.

    The outcome carries a stashed refit context (from :func:`execute_plan`),
    which is required by :func:`_add_from_convergence`.  The window is
    centered on ``f0`` and the spectrum contains only one line at ``f0``
    (no companion), so ``outcome.fit.n_peaks == 1``.
    """
    sigma = 1.0
    center = f0
    freq_array = np.arange(center - 5.0, center + 5.0, DF_MHZ)
    a0 = _amp_for_snr(snr0, sigma)
    spectrum = _synth_spectrum(freq_array, [(f0, a0, 0.3)])
    rng = np.random.default_rng(SEED + 200)
    spectrum = spectrum + _complex_noise(freq_array.size, sigma, rng)
    rms_noise = np.full(freq_array.size, sigma)
    win = FitWindow(
        window_id=0,
        freq_range=(center - 4.0, center + 4.0),
        free_peak_indices=[0],
        fixed_contributors=[],
        batch=0,
    )
    plan = WindowPlan(windows=[win], dependency_edges=[], topological_order=[0])
    plan_out = execute_plan(
        plan,
        _make_active_ft(freq_array, spectrum),
        rms_noise,
        [f0],
        sideband=SIDEBAND,
        acquisition_us=T_US,
        tau0_us=TAU_US,
    )
    return plan_out.window_outcomes[0]


def _make_rescue_event(
    window_id: int,
    candidates: list,
) -> "RescueEvent":
    from ftmwpipeline.fitting.plan_execution import RescueEvent

    return RescueEvent(
        window_id=window_id,
        round_idx=0,
        n_initial_peaks=1,
        n_candidates=len(candidates),
        n_rescue_added=0,
        n_pruned_by_knockout=0,
        n_pruned_rescue_origin=0,
        n_merged=0,
        chi2_before=2.0,
        chi2_after=2.0,
        tau_us_before=TAU_US,
        tau_us_after=TAU_US,
        accepted=False,
        reason="",
        candidates=list(candidates),
    )


def _make_cand(off: float, snr: float, sigma: float = 1.0) -> "ResidualPeakCandidate":
    from ftmwpipeline.fitting.residual_screening import ResidualPeakCandidate

    return ResidualPeakCandidate(
        bin_index=0,
        frequency_mhz=off,
        magnitude=snr * sigma / np.sqrt(2.0),
        snr=snr,
        prominence_sigma_c=snr,
        nearest_existing_detection_index=None,
        nearest_existing_freq_mhz=None,
        nearest_existing_separation_mhz=None,
        near_existing=False,
    )


def test_add_from_convergence_recovers_companion():
    """F-2: a companion line missed by the seeder is recovered post-convergence.

    Fit a single-line spectrum (only f0).  Then inject the companion signal at
    f1 directly into the outcome's complex_spectrum and full_residual so the
    warm-started add has real data to fit.  Attach a high-SNR rescue candidate
    at f1's signed-baseband-offset coordinate.  With a no-op finalize_node,
    F-2 should accept the add and return a two-peak outcome.
    """
    from ftmwpipeline.fitting.plan_execution import (
        NodeCleanup,
        _add_from_convergence,
    )

    sigma = 1.0
    f0 = 36100.0
    f1 = 36101.5  # companion; 1.5 MHz from f0, well above min_sep

    outcome = _one_line_outcome(f0)
    assert outcome.fit.n_peaks == 1  # only f0 was seeded and fit

    grid = np.asarray(outcome.offset_grid_mhz, dtype=float)
    a1 = _amp_for_snr(35.0, sigma)
    # Build a companion spectrum on the same molecular-frequency array that
    # _one_line_outcome used (center - 5 to center + 5), then slice to the
    # window grid (center ± 4).  The companion is h_T-shaped at f1.
    center = f0
    freq_array_full = np.arange(center - 5.0, center + 5.0, DF_MHZ)
    companion_full = _synth_spectrum(freq_array_full, [(f1, a1, 1.1)])
    mask = (freq_array_full >= center - 4.0) & (freq_array_full <= center + 4.0)
    companion_in_window = companion_full[mask]
    assert companion_in_window.size == grid.size

    # Add the companion to both complex_spectrum and full_residual so they
    # stay consistent: full_residual = complex_spectrum - full_fitted_spectrum.
    outcome.complex_spectrum = (
        np.asarray(outcome.complex_spectrum, dtype=np.complex128) + companion_in_window
    )
    outcome.full_residual = (
        np.asarray(outcome.full_residual, dtype=np.complex128) + companion_in_window
    )

    # The rescue candidate's frequency_mhz must be in the signed baseband
    # offset frame (same as outcome.offset_grid_mhz).  For a lower sideband:
    # u = s * (f - f_center) = -(f1 - f0) = -1.5.
    s = -1.0  # lower sideband
    companion_offset = s * (f1 - center)  # -1.5

    outcome.rescue_events = [
        _make_rescue_event(0, [_make_cand(companion_offset, 30.0, sigma)])
    ]

    def noop_finalize(o: WindowOutcome) -> NodeCleanup:
        return NodeCleanup(outcome=o)

    result = _add_from_convergence(
        outcome,
        acquisition_us=T_US,
        conservative_kwargs={},
        snr_threshold=10.0,
        finalize_node=noop_finalize,
    )
    assert result.fit.n_peaks == 2  # companion was recovered


def test_add_from_convergence_stamps_the_added_peak_when_point_map_supplied():
    """F-2's warm-started add is another genuine seed birth
    (:func:`_add_from_convergence` calls ``_seed_peak`` directly, outside
    ``window_fit.py``) -- it must stamp too when a ``point_map`` rides in
    the ``conservative_kwargs`` bag, exactly like every other seed site."""
    from ftmwpipeline.fitting.plan_execution import (
        NodeCleanup,
        _add_from_convergence,
        _window_center_mhz,
    )

    sigma = 1.0
    f0 = 36100.0
    f1 = 36101.5

    outcome = _one_line_outcome(f0)
    assert outcome.fit.n_peaks == 1

    grid = np.asarray(outcome.offset_grid_mhz, dtype=float)
    a1 = _amp_for_snr(35.0, sigma)
    center = f0
    freq_array_full = np.arange(center - 5.0, center + 5.0, DF_MHZ)
    companion_full = _synth_spectrum(freq_array_full, [(f1, a1, 1.1)])
    mask = (freq_array_full >= center - 4.0) & (freq_array_full <= center + 4.0)
    companion_in_window = companion_full[mask]
    assert companion_in_window.size == grid.size

    outcome.complex_spectrum = (
        np.asarray(outcome.complex_spectrum, dtype=np.complex128) + companion_in_window
    )
    outcome.full_residual = (
        np.asarray(outcome.full_residual, dtype=np.complex128) + companion_in_window
    )

    s = -1.0  # lower sideband
    companion_offset = s * (f1 - center)

    outcome.rescue_events = [
        _make_rescue_event(0, [_make_cand(companion_offset, 30.0, sigma)])
    ]

    def noop_finalize(o: WindowOutcome) -> NodeCleanup:
        return NodeCleanup(outcome=o)

    window_center = _window_center_mhz(outcome)
    point_map = PointMap.from_frame(
        window_center, SIDEBAND, PROBE_MHZ, outcome.offset_grid_mhz.size * 100, 0.05
    )

    result = _add_from_convergence(
        outcome,
        acquisition_us=T_US,
        conservative_kwargs={"point_map": point_map},
        snr_threshold=10.0,
        finalize_node=noop_finalize,
    )
    assert result.fit.n_peaks == 2
    added = [p for p in result.fit.peaks if abs(p.offset_mhz - companion_offset) < 0.3]
    assert len(added) == 1
    assert added[0].peak_uid == point_map.stamp(companion_offset)


def test_add_from_convergence_rejects_collapse():
    """F-2: a candidate within min-separation of an existing peak is rejected.

    A candidate placed almost exactly on top of the already-fitted peak would
    collapse (the NLS drives the added line onto the existing one); the
    collapse guard must reject it and leave the outcome unchanged.
    """
    from ftmwpipeline.fitting.plan_execution import (
        NodeCleanup,
        _add_from_convergence,
    )

    sigma = 1.0
    f0 = 36100.0
    outcome = _one_line_outcome(f0)
    assert outcome.fit.n_peaks == 1

    # Place a candidate at the *same position* as the fitted peak.  This will
    # collapse (the two lines cannot be resolved) and must be rejected.
    existing_off = outcome.fit.peaks[0].offset_mhz

    outcome.rescue_events = [
        _make_rescue_event(0, [_make_cand(existing_off, 25.0, sigma)])
    ]

    def noop_finalize(o: WindowOutcome) -> NodeCleanup:
        return NodeCleanup(outcome=o)

    result = _add_from_convergence(
        outcome,
        acquisition_us=T_US,
        conservative_kwargs={},
        snr_threshold=10.0,
        finalize_node=noop_finalize,
    )
    # The collapse should have been caught; peak count unchanged.
    assert result.fit.n_peaks == 1


def test_add_from_convergence_skips_bright_line_sidelobe():
    """F-2: a candidate inside a bright line's shape-error shadow is not added.

    A strong (snr~3000) line casts a wide lineshape-error shadow
    (``reach = kappa * snr / evidence`` res). A residual candidate several
    resolution elements away -- well outside the collapse min-separation, so
    only the sidelobe pre-filter can stop it -- must be skipped before any
    refit, matching the Stage 6 ledger filter that drops the same sidelobe.
    """
    from ftmwpipeline.fitting.plan_execution import (
        NodeCleanup,
        _add_from_convergence,
    )

    sigma = 1.0
    f0 = 36100.0
    outcome = _one_line_outcome(f0, snr0=3000.0)
    assert outcome.fit.n_peaks == 1

    # Candidate ~5 resolution elements (0.4 MHz) from the bright line, evidence
    # 30: reach = 0.2 * 3000 / 30 = 20 res >> 5 res -> filtered as a sidelobe.
    s = -1.0  # lower sideband: u = -(f - f_center)
    sidelobe_offset = s * 0.4

    outcome.rescue_events = [
        _make_rescue_event(0, [_make_cand(sidelobe_offset, 30.0, sigma)])
    ]

    n_called = 0

    def counting_finalize(o: WindowOutcome) -> NodeCleanup:
        nonlocal n_called
        n_called += 1
        return NodeCleanup(outcome=o)

    result = _add_from_convergence(
        outcome,
        acquisition_us=T_US,
        conservative_kwargs={},
        snr_threshold=10.0,
        finalize_node=counting_finalize,
    )
    assert result.fit.n_peaks == 1  # sidelobe not installed
    assert n_called == 0  # skipped before any refit / finalize


def test_add_from_convergence_skips_below_threshold():
    """F-2: candidates below snr_threshold are not attempted."""
    from ftmwpipeline.fitting.plan_execution import (
        NodeCleanup,
        _add_from_convergence,
    )

    sigma = 1.0
    f0 = 36100.0
    f1 = 36101.5

    outcome = _one_line_outcome(f0)
    n_peaks_before = (
        outcome.fit.n_peaks
    )  # could be 1 or 2, we only test that F-2 is not called

    # Candidate SNR (5.0) is below the threshold (10.0): no add attempted.
    outcome.rescue_events = [_make_rescue_event(0, [_make_cand(f1, 5.0, sigma)])]

    n_called = 0

    def counting_finalize(o: WindowOutcome) -> NodeCleanup:
        nonlocal n_called
        n_called += 1
        return NodeCleanup(outcome=o)

    result = _add_from_convergence(
        outcome,
        acquisition_us=T_US,
        conservative_kwargs={},
        snr_threshold=10.0,
        finalize_node=counting_finalize,
    )
    assert result.fit.n_peaks == n_peaks_before  # unchanged
    assert n_called == 0  # finalize_node was never called


class TestReplanReanchorsMergedWindows:
    """A merge survivor keeps its id but not its range: after each accepted
    round its per-band tau anchor is recomputed from the merged range (the
    anchor a refit of the merged window resolves), the absorbed id's anchor
    drops out, and the caller's map is left as given."""

    def _spy_walks(self, monkeypatch):
        seen = []
        real = plan_execution._walk_windows_parallel

        def spy(plan, order, **kwargs):
            seen.append(
                (kwargs["phase"], list(order), dict(kwargs["window_tau_overrides"]))
            )
            return real(plan, order, **kwargs)

        monkeypatch.setattr(plan_execution, "_walk_windows_parallel", spy)
        return seen

    def test_each_round_reanchors_the_survivor_at_its_merged_range(self, monkeypatch):
        monkeypatch.setattr(
            plan_execution,
            "_dispatch_structural_round",
            _scripted_dispatch(
                {0: [_trig(1, 0, "low", 5.0)], 1: [_trig(0, 2, "high", 2.0)]}
            ),
        )
        seen = self._spy_walks(monkeypatch)
        ranges = []

        def anchor(freq_range):
            ranges.append(freq_range)
            return (TAU_US, 1.0 + len(ranges))

        fixture = _exec_fixture(_CHAIN_WINDOWS)
        fixture[1].tau_anchor_for_range = anchor
        given = {0: (TAU_US, 1.0), 1: (TAU_US, 1.0), 2: (TAU_US, 1.0)}
        out = _run_exec(fixture, window_tau_overrides=dict(given))

        assert out.final_plan_revision == 2
        (merged,) = out.final_plan.windows
        initial, round1, round2 = seen
        assert initial[0] == "initial" and initial[2] == given
        # Round 1 merges 0+1: 0 re-anchors at its merged range, 1 drops out.
        assert round1[0] == "replan"
        assert round1[2] == {0: (TAU_US, 2.0), 2: (TAU_US, 1.0)}
        # Round 2 merges 0+2: 0 re-anchors again, 2 drops out.
        assert round2[2] == {0: (TAU_US, 3.0)}
        assert ranges[-1] == merged.freq_range
        assert ranges[0][0] == merged.freq_range[0]
        assert ranges[0][1] < merged.freq_range[1]

    def test_a_survivor_no_band_holds_fits_on_the_band_wide_anchor(self, monkeypatch):
        monkeypatch.setattr(
            plan_execution,
            "_dispatch_structural_round",
            _scripted_dispatch({0: [_trig(0, 1, "high", 5.0)]}),
        )
        seen = self._spy_walks(monkeypatch)
        fixture = _exec_fixture(_CHAIN_WINDOWS[:2])
        fixture[1].tau_anchor_for_range = lambda freq_range: None
        given = {0: (TAU_US, 1.0), 1: (TAU_US, 1.0)}
        _run_exec(fixture, window_tau_overrides=dict(given))
        assert seen[-1][0] == "replan"
        assert seen[-1][2] == {}

    def test_without_a_resolver_the_overrides_stay_as_given(self, monkeypatch):
        monkeypatch.setattr(
            plan_execution,
            "_dispatch_structural_round",
            _scripted_dispatch({0: [_trig(0, 1, "high", 5.0)]}),
        )
        seen = self._spy_walks(monkeypatch)
        given = {0: (TAU_US, 1.0), 1: (TAU_US, 1.0)}
        overrides = dict(given)
        _run_exec(_exec_fixture(_CHAIN_WINDOWS[:2]), window_tau_overrides=overrides)
        assert seen[-1][2] == given
        assert overrides == given
