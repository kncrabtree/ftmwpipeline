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

import numpy as np
import pytest

from ftmwpipeline.core.data_structures import (
    FitWindow,
    FixedContributor,
    Sideband,
    WindowDifficulty,
    WindowPlan,
)
from ftmwpipeline.fitting.peak_model import (
    ModelPeak,
    effective_tau,
    h_T,
    model_spectrum,
)
from ftmwpipeline.fitting.plan_execution import (
    DEFAULT_RESIDUAL_EDGE_THRESHOLD,
    FrozenPeak,
    ThawEvent,
    WindowOutcome,
    attempt_thaw_round,
    evaluate_fixed_contributor,
    execute_plan,
    fit_window_with_fixed_contributors,
    local_thaw_cofit,
    residual_edge_coherence,
    select_contributor_to_thaw,
    subtract_frozen_background,
)
from ftmwpipeline.fitting.validation import feature_fwhm
from ftmwpipeline.fitting.window_fit import conservative_fit

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


# ---------------------------------------------------------------------------
# evaluate_fixed_contributor
# ---------------------------------------------------------------------------
class TestEvaluateFixedContributor:
    def test_frame_remap_into_dependent_window(self):
        """A FixedContributor's offset is remapped between the two window frames.

        Primary window is centered at 36100 MHz; the line lies at 36100.5 MHz.
        In the primary's frame ``delta_primary = s*(f - f_c_primary) = -0.5``
        (lower sideband). Dependent window is centered at 36120 MHz; the line's
        offset in the dependent frame should be ``s*(f - f_c_dep) = -(-19.5) =
        ... let s be -1 -> delta_dep = -1*(36100.5 - 36120) = 19.5``.
        """
        # A primary outcome with one fitted peak in the primary frame.
        primary_center = 36100.0
        line_freq = 36100.5
        s = -1.0
        primary_peak = ModelPeak(
            amplitude=1.0, offset_mhz=s * (line_freq - primary_center), phase=0.0
        )

        primary_outcome = _toy_outcome(
            window_id=0, center_mhz=primary_center, fit_peaks=[primary_peak]
        )
        contributor = FixedContributor(
            peak_index=7,
            primary_window_id=0,
            frequency_mhz=line_freq,
            freeze_eligible=True,
        )
        dep_center = 36120.0

        frozen = evaluate_fixed_contributor(
            contributor,
            primary_outcome,
            dependent_center_mhz=dep_center,
            sideband=SIDEBAND,
        )

        expected_offset = s * (line_freq - dep_center)
        assert frozen.model_peak.offset_mhz == pytest.approx(expected_offset)
        assert frozen.model_peak.amplitude == primary_peak.amplitude
        assert frozen.model_peak.phase == primary_peak.phase
        assert frozen.peak_index == 7
        assert frozen.primary_window_id == 0
        assert frozen.frequency_mhz == line_freq
        assert frozen.freeze_eligible is True

    def test_nearest_match_picks_correct_peak(self):
        """When the primary has multiple fitted lines, the nearest one wins."""
        primary_center = 36100.0
        s = -1.0
        # Two lines in the primary; the contributor is the one at 36099.7.
        line_a = ModelPeak(1.0, s * (36099.7 - primary_center), 0.1)
        line_b = ModelPeak(2.0, s * (36100.6 - primary_center), 0.4)
        primary_outcome = _toy_outcome(0, primary_center, [line_a, line_b])
        contributor = FixedContributor(
            peak_index=3, primary_window_id=0, frequency_mhz=36099.7
        )
        frozen = evaluate_fixed_contributor(
            contributor,
            primary_outcome,
            dependent_center_mhz=36130.0,
            sideband=SIDEBAND,
        )
        # The amplitude/phase reveal which primary peak was matched.
        assert frozen.model_peak.amplitude == line_a.amplitude
        assert frozen.model_peak.phase == line_a.phase

    def test_raises_when_primary_has_no_peaks(self):
        primary_outcome = _toy_outcome(0, 36100.0, [])
        contributor = FixedContributor(
            peak_index=1, primary_window_id=0, frequency_mhz=36100.5
        )
        with pytest.raises(ValueError, match="no fitted peaks"):
            evaluate_fixed_contributor(
                contributor,
                primary_outcome,
                dependent_center_mhz=36120.0,
                sideband=SIDEBAND,
            )


# ---------------------------------------------------------------------------
# subtract_frozen_background
# ---------------------------------------------------------------------------
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
            difficulty=WindowDifficulty.EASY,
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
            difficulty=WindowDifficulty.EASY,
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
            freqs,
            spec,
            noise,
            peak_freqs,
            probe_freq_mhz=PROBE_MHZ,
            sideband=SIDEBAND,
            start_us=START_US,
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

        # The dependent saw the strong line as a frozen contributor.
        assert len(w1.fixed_peaks) == 1
        assert w1.fixed_peaks[0].peak_index == 0
        assert w1.fixed_peaks[0].primary_window_id == 0

        # The dependent's free fit recovers the weak line to ~kHz.
        s = -1.0
        # Frame: free peak is in dep's offset frame = s*(weak - dep_center).
        dep_center = 0.5 * (weak_freq - 0.6 + weak_freq + 0.6)
        expected_offset = s * (weak_freq - dep_center)
        got_offset = w1.fit.peaks[0].offset_mhz
        assert abs(got_offset - expected_offset) < 0.003  # 3 kHz

        # No thaw was triggered.
        assert outcome.thaw_history == []

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
            freq_array,
            spectrum,
            rms_noise,
            peak_freqs,
            probe_freq_mhz=PROBE_MHZ,
            sideband=SIDEBAND,
            start_us=START_US,
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
            freq_array,
            spectrum,
            rms_noise,
            [strong_freq],
            probe_freq_mhz=PROBE_MHZ,
            sideband=SIDEBAND,
            start_us=START_US,
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
        # 4 MHz apart: A's skirt at B is ~A/(2*pi*4) ~ 4% of A's centre, and
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
            freq_array,
            spectrum,
            rms_noise,
            [strong_freq, weak_freq],
            probe_freq_mhz=PROBE_MHZ,
            sideband=SIDEBAND,
            start_us=START_US,
            acquisition_us=T_US,
            tau0_us=TAU_US,
        )
        # On a clean spectrum the residual edges are quiet -- no spontaneous
        # thaw.
        assert outcome.thaw_history == []

        # Mutate the primary's converged fit: bump the strong line's fitted
        # amplitude 25% high. This stands in for the blended-primary case where
        # the conservative loop's sequential initialisation collapses two close
        # cosines into one stronger-looking fit (§4 of the prototype report).
        primary = outcome.window_outcomes[0]
        original_amp = primary.fit.peaks[0].amplitude
        primary.fit.peaks[0].amplitude = original_amp * 1.25

        # Re-derive the dependent's frozen background, full model, residual,
        # and edge-coherence with the corrupted primary -- exactly what
        # execute_plan would have done if it had re-visited the dependent.
        dep = outcome.window_outcomes[1]
        dep_center_mhz = 0.5 * (weak_freq - 0.3 + weak_freq + 0.3)
        corrupted_frozen = evaluate_fixed_contributor(
            FixedContributor(0, 0, strong_freq, True),
            primary,
            dependent_center_mhz=dep_center_mhz,
            sideband=SIDEBAND,
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
            freq_array,
            spectrum,
            rms_noise,
            [strong_freq, weak_freq],
            probe_freq_mhz=PROBE_MHZ,
            sideband=SIDEBAND,
            start_us=START_US,
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
            freq_array,
            spectrum,
            rms_noise,
            [strong_freq, weak_freq],
            probe_freq_mhz=PROBE_MHZ,
            sideband=SIDEBAND,
            start_us=START_US,
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
            freq_array,
            spectrum,
            rms_noise,
            [strong_freq, weak_freq],
            probe_freq_mhz=PROBE_MHZ,
            sideband=SIDEBAND,
            start_us=START_US,
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
# Helpers used by the tests
# ---------------------------------------------------------------------------
def _toy_outcome(
    window_id: int, center_mhz: float, fit_peaks: list[ModelPeak]
) -> WindowOutcome:
    """Minimal WindowOutcome with the molecular-center attribute set.

    Used by the standalone evaluate_fixed_contributor tests where we want to
    skip the full plan executor and inject a pre-cooked primary outcome.
    """
    # An almost-empty WindowOutcome; we only need .fit.peaks and the center
    # for evaluate_fixed_contributor / the thaw helpers.
    grid = np.array([0.0])
    data = np.zeros(1, dtype=np.complex128)
    rms = np.array([1.0])
    bg = np.zeros(1, dtype=np.complex128)

    # We need a ConservativeFitResult shell with .peaks; the easiest is to use
    # conservative_fit on a trivial input -- but cheaper to build a fake fit
    # with just the peaks list set.
    from ftmwpipeline.fitting.window_fit import (
        ConservativeFitResult,
        WindowFitResult,
    )

    inner = WindowFitResult(
        success=True,
        peaks=list(fit_peaks),
        peak_errors=[],
        tau_us=TAU_US,
        tau_error=None,
        fit_tau=False,
        cost=0.0,
        chi_squared=0.0,
        n_data=2,
        n_params=3 * len(fit_peaks),
        n_function_evals=0,
        fitted_spectrum=np.zeros(1, dtype=np.complex128),
        residual=np.zeros(1, dtype=np.complex128),
    )
    fit_result = ConservativeFitResult(inner, [], [])
    outcome = WindowOutcome(
        window_id=window_id,
        fit=fit_result,
        fixed_peaks=[],
        offset_grid_mhz=grid,
        complex_spectrum=data,
        rms_noise=rms,
        background=bg,
        full_fitted_spectrum=np.zeros(1, dtype=np.complex128),
        full_residual=np.zeros(1, dtype=np.complex128),
    )
    outcome._center_mhz = center_mhz  # type: ignore[attr-defined]
    return outcome
