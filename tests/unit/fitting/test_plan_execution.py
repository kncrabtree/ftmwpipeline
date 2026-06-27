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
    Peak,
    PeakClassification,
    Sideband,
    WindowPlan,
)
from ftmwpipeline.fitting.active_ft import ActiveFTResult
from ftmwpipeline.fitting.peak_model import (
    ModelPeak,
    effective_tau,
    h_T,
    model_spectrum,
)
from ftmwpipeline.fitting.plan_execution import (
    DEFAULT_RESIDUAL_EDGE_THRESHOLD,
    FrozenPeak,
    ReplanContext,
    ReplanEvent,
    ThawEvent,
    WindowOutcome,
    attempt_thaw_round,
    evaluate_edge_free_contributors,
    evaluate_fixed_contributor,
    execute_plan,
    fit_window_with_fixed_contributors,
    local_thaw_cofit,
    refit_outcome,
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
    n_padded = max(int(round(n_active / alpha)), n_active)
    return ActiveFTResult(
        freq_mhz=np.asarray(freq_array, dtype=float),
        complex_spectrum=spec,
        alpha=float(alpha),
        n_active=n_active,
        n_padded=n_padded,
    )


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
        rng = np.random.default_rng(SEED + 30)
        freq_array = np.arange(36100.0, 36120.0 + DF_MHZ / 2, DF_MHZ)
        amp = _amp_for_snr(peak_snr, sigma)
        spectrum = _synth_spectrum(freq_array, [(peak_freq, amp, 0.3)])
        spectrum = spectrum + _complex_noise(freq_array.size, sigma, rng)
        rms = np.full(freq_array.size, sigma)
        gi = int(np.argmin(np.abs(freq_array - peak_freq)))
        peaks = [
            _make_peak(peak_freq, peak_snr, sigma, grid_index=gi),
        ]
        return freq_array, spectrum, rms, peaks

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
        """A peak sitting just inside window A; window B is empty but its low
        edge sees the peak's leakage skirt. With no contributor attached to
        B's low side, the dispatcher emits a merge of A and B; replan
        produces one wide window; refit clears the previously-flagged edge.
        """
        sigma = 1.0
        peak_freq = 36104.9
        freq_array, spectrum, rms, peaks = self._single_peak_setup(
            peak_freq, peak_snr=300.0, sigma=sigma
        )

        # Manual two-window plan: boundary at 36105.0, peak in window A.
        # Both windows have no fixed contributors -- the dispatcher will
        # blame the flagged edge on no contributor and merge.
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

        # A structural merge was applied.
        assert outcome.replan_history, "expected a structural-replan event"
        accepted = [e for e in outcome.replan_history if e.accepted]
        assert accepted, "expected the merge to be accepted"
        ev = accepted[0]
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
        assert survivor[0].freq_range == (36100.0, 36110.0)

        # Only one window left; it contains the peak as a free peak; its
        # edges are clean (the feature is no longer cut by a boundary).
        assert set(outcome.window_outcomes.keys()) == {0}
        merged = outcome.window_outcomes[0]
        assert merged.fit.fit.success
        assert merged.fit.n_peaks == 1
        # The survivor's stored range contains its fitted peak (no escape).
        lo, hi = survivor[0].freq_range
        assert lo <= peak_freq <= hi
        # Edges below the dispatcher threshold (else another round would fire
        # if max_replan_rounds allowed).
        assert merged.edge_coherence_low <= DEFAULT_RESIDUAL_EDGE_THRESHOLD
        assert merged.edge_coherence_high <= DEFAULT_RESIDUAL_EDGE_THRESHOLD

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
# Leakage-wing baseline trigger (_apply_baseline_to_outcome)
# ---------------------------------------------------------------------------
def _baseline_outcome(u, z, sigma, peaks, *, tau=TAU_US):
    """Minimal WindowOutcome (no frozen background) for baseline-trigger tests.

    The fit is the converged peaks-only model; the outcome's edge-coherence is
    measured on that residual so ``_apply_baseline_to_outcome`` reads a real
    trigger value.
    """
    from ftmwpipeline.fitting.peak_model import model_spectrum
    from ftmwpipeline.fitting.plan_execution import residual_edge_coherence
    from ftmwpipeline.fitting.window_fit import (
        ConservativeFitResult,
        WindowFitResult,
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

    def _fit(self, workers):
        plan, freqs, spec, noise, peak_freqs = self._build_plan()
        old = _pe._FIT_WINDOW_WORKERS
        _pe._FIT_WINDOW_WORKERS = workers
        try:
            return execute_plan(
                plan,
                _make_active_ft(freqs, spec),
                noise,
                peak_freqs,
                sideband=SIDEBAND,
                acquisition_us=T_US,
                tau0_us=TAU_US,
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
            nearest_existing_peak_id=None,
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
        nearest_existing_peak_id=None,
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
