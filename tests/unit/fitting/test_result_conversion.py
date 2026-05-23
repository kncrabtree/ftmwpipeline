"""
Unit tests for Stage 5 result-conversion wiring (task 8).

Covers :mod:`ftmwpipeline.fitting.result_conversion`:

* per-peak offset -> molecular-frequency mapping on both sidebands;
* audit trail and knockout attachment per :class:`FittingResult`;
* thaw events partition per-window vs at the plan level;
* merged global :class:`SpectrumFit.fitted_peaks` sorted by frequency, each
  tagged with its originating window id;
* propagation of ``final_plan_revision`` and parameters.

The synthetic-fixture builders from :mod:`test_plan_execution` are reused
so the in-flight :class:`PlanFitOutcome` we convert here is constructed the
same way the algorithm tests build one.
"""

from __future__ import annotations

import numpy as np
import pytest

from ftmwpipeline.core.data_structures import (
    AuditStep,
    FittedPeak,
    FittingResult,
    FitWindow,
    FixedContributor,
    KnockoutInfo,
    ReplanInfo,
    Sideband,
    SpectralWindow,
    SpectrumFit,
    ThawInfo,
    WindowDifficulty,
    WindowPlan,
)
from ftmwpipeline.fitting.peak_model import ModelPeak, molecular_frequency
from ftmwpipeline.fitting.plan_execution import (
    DEFAULT_RESIDUAL_EDGE_THRESHOLD,
    ReplanContext,
    evaluate_fixed_contributor,
    execute_plan,
    residual_edge_coherence,
    attempt_thaw_round,
)
from ftmwpipeline.fitting.result_conversion import (
    plan_fit_outcome_to_spectrum_fit,
    window_outcome_to_fitting_result,
    window_outcome_to_spectral_window,
)

# Reuse the synthetic builders from the plan-execution tests.
from tests.unit.fitting.test_plan_execution import (
    DF_MHZ,
    PROBE_MHZ,
    T_US,
    TAU_US,
    SEED,
    SIDEBAND,
    _STAGE4_PARAMS,
    _amp_for_snr,
    _complex_noise,
    _make_active_ft,
    _make_peak,
    _synth_spectrum,
)


# ---------------------------------------------------------------------------
# Two-window happy-path fixture (shared by several conversion tests)
# ---------------------------------------------------------------------------
def _two_window_plan_outcome(
    strong_freq: float = 36100.0,
    weak_freq: float = 36110.0,
    strong_snr: float = 300.0,
    weak_snr: float = 50.0,
    sigma: float = 1.0,
    sideband: Sideband = SIDEBAND,
):
    """Build a freshly-executed two-window plan + outcome.

    Returns the executed ``PlanFitOutcome`` plus the inputs needed by the
    converter (plan, peak_frequencies_mhz, acquisition_us, sideband). The
    spectrum is synthesized with the requested sideband sign so the converter
    can be exercised on both LSB and USB.
    """
    rng = np.random.default_rng(SEED + 100)
    freq_array = np.arange(strong_freq - 5.0, weak_freq + 5.0, DF_MHZ)
    s = -1.0 if sideband == Sideband.LOWER else 1.0
    z = np.zeros(freq_array.shape, dtype=np.complex128)
    from ftmwpipeline.fitting.peak_model import h_T

    for f_j, snr, phi in [
        (strong_freq, strong_snr, 0.3),
        (weak_freq, weak_snr, 1.7),
    ]:
        amp = _amp_for_snr(snr, sigma)
        du = s * (freq_array - f_j)
        z += 0.5 * amp * np.exp(1j * phi) * h_T(du, TAU_US, T_US)
    spectrum = z + _complex_noise(freq_array.size, sigma, rng)
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
        fixed_contributors=[FixedContributor(0, 0, strong_freq, freeze_eligible=True)],
        batch=1,
    )
    plan = WindowPlan(
        windows=[win_a, win_b],
        dependency_edges=[(1, 0)],
        topological_order=[0, 1],
    )
    peak_frequencies = [strong_freq, weak_freq]

    outcome = execute_plan(
        plan,
        _make_active_ft(freq_array, spectrum),
        rms_noise,
        peak_frequencies,
        sideband=sideband,
        acquisition_us=T_US,
        tau0_us=TAU_US,
    )
    return outcome, plan, peak_frequencies, freq_array, spectrum, rms_noise


# ---------------------------------------------------------------------------
# (a) Per-peak offset -> molecular frequency, both sidebands
# ---------------------------------------------------------------------------
class TestPerPeakFrequencyMapping:
    def test_lower_sideband_recovers_molecular_frequency(self):
        """LSB: fitted offset maps back to molecular frequency within ~kHz."""
        strong_freq, weak_freq = 36100.0, 36110.0
        plan_outcome, plan, peak_freqs, *_ = _two_window_plan_outcome(
            strong_freq, weak_freq, sideband=Sideband.LOWER
        )
        fit = plan_fit_outcome_to_spectrum_fit(
            plan_outcome,
            plan,
            sideband=Sideband.LOWER,
            peak_frequencies_mhz=peak_freqs,
            acquisition_us=T_US,
        )

        # One fitted peak per window; mapped back to molecular axis to
        # within a few kHz of truth.
        strong_fits = [p for p in fit.fitted_peaks if p.window_id == 0]
        weak_fits = [p for p in fit.fitted_peaks if p.window_id == 1]
        assert len(strong_fits) == 1
        assert len(weak_fits) == 1
        assert abs(strong_fits[0].frequency_mhz - strong_freq) < 0.005
        assert abs(weak_fits[0].frequency_mhz - weak_freq) < 0.01

    def test_upper_sideband_recovers_molecular_frequency(self):
        """USB sweep: the same fixture under USB recovers within ~kHz."""
        strong_freq, weak_freq = 36100.0, 36110.0
        plan_outcome, plan, peak_freqs, *_ = _two_window_plan_outcome(
            strong_freq, weak_freq, sideband=Sideband.UPPER
        )
        fit = plan_fit_outcome_to_spectrum_fit(
            plan_outcome,
            plan,
            sideband=Sideband.UPPER,
            peak_frequencies_mhz=peak_freqs,
            acquisition_us=T_US,
        )
        strong_fits = [p for p in fit.fitted_peaks if p.window_id == 0]
        weak_fits = [p for p in fit.fitted_peaks if p.window_id == 1]
        assert len(strong_fits) == 1
        assert len(weak_fits) == 1
        assert abs(strong_fits[0].frequency_mhz - strong_freq) < 0.005
        assert abs(weak_fits[0].frequency_mhz - weak_freq) < 0.01

    def test_peak_id_links_to_stage3_index(self):
        """``peak_id`` is the Stage 3 promoted-peak index, not a generic counter."""
        plan_outcome, plan, peak_freqs, *_ = _two_window_plan_outcome()
        fit = plan_fit_outcome_to_spectrum_fit(
            plan_outcome,
            plan,
            sideband=SIDEBAND,
            peak_frequencies_mhz=peak_freqs,
            acquisition_us=T_US,
        )
        # Window 0's free_peak_indices = [0]; window 1's = [1].
        for p in fit.fitted_peaks:
            if p.window_id == 0:
                assert p.peak_id == 0
            elif p.window_id == 1:
                assert p.peak_id == 1

    def test_decay_rate_and_error_propagated(self):
        """The persistent ``decay_rate = 1/tau`` and its error are derived
        from the algorithm's ``tau_us`` / ``tau_error`` (error propagation
        ``d(1/tau)/dtau = -1/tau^2``)."""
        plan_outcome, plan, peak_freqs, *_ = _two_window_plan_outcome()
        fit = plan_fit_outcome_to_spectrum_fit(
            plan_outcome,
            plan,
            sideband=SIDEBAND,
            peak_frequencies_mhz=peak_freqs,
            acquisition_us=T_US,
        )
        for window_fit in fit.window_fits:
            outcome_tau = window_fit.shared_parameters["tau_us"]["value"]
            for peak in window_fit.fitted_peaks:
                assert peak.decay_rate == pytest.approx(1.0 / outcome_tau)


# ---------------------------------------------------------------------------
# (b) Audit trail and knockouts on the right FittingResult
# ---------------------------------------------------------------------------
class TestAuditAndKnockoutAttachment:
    def test_audit_trail_matches_algorithm_record(self):
        """Every algorithm-side AddStep appears as a persistent AuditStep."""
        plan_outcome, plan, peak_freqs, *_ = _two_window_plan_outcome()
        fit = plan_fit_outcome_to_spectrum_fit(
            plan_outcome,
            plan,
            sideband=SIDEBAND,
            peak_frequencies_mhz=peak_freqs,
            acquisition_us=T_US,
        )
        for window_fit in fit.window_fits:
            wid = window_fit.window_id
            algo_steps = plan_outcome.window_outcomes[wid].fit.audit_trail
            assert len(window_fit.audit_trail) == len(algo_steps)
            for persisted, algo in zip(window_fit.audit_trail, algo_steps):
                assert isinstance(persisted, AuditStep)
                assert persisted.decision == algo.decision
                assert persisted.n_peaks_before == algo.n_peaks_before
                assert persisted.candidate_offset_mhz == algo.candidate_offset_mhz
                assert persisted.f_statistic == algo.f_statistic
                assert persisted.p_value == algo.p_value

    def test_knockout_attached_to_fitted_peak(self):
        """Each FittedPeak.knockout mirrors the algorithm's KnockoutResult.

        The algorithm-side ``knockouts`` list is in fitted-peak order; the
        converter walks the same order, so peak ``i`` carries the ``i``-th
        knockout entry.
        """
        plan_outcome, plan, peak_freqs, *_ = _two_window_plan_outcome()
        fit = plan_fit_outcome_to_spectrum_fit(
            plan_outcome,
            plan,
            sideband=SIDEBAND,
            peak_frequencies_mhz=peak_freqs,
            acquisition_us=T_US,
        )
        for window_fit in fit.window_fits:
            wid = window_fit.window_id
            algo_knockouts = plan_outcome.window_outcomes[wid].fit.knockouts
            assert len(window_fit.fitted_peaks) == len(algo_knockouts)
            for peak, algo_k in zip(window_fit.fitted_peaks, algo_knockouts):
                assert isinstance(peak.knockout, KnockoutInfo)
                assert peak.knockout.delta_chi2 == pytest.approx(algo_k.delta_chi2)
                assert peak.knockout.expected_delta_chi2 == pytest.approx(
                    algo_k.expected_delta_chi2
                )
                assert peak.knockout.supported == algo_k.supported

    def test_shared_parameter_carries_tau(self):
        """``shared_parameters['tau_us']`` carries the fitted tau and its error."""
        plan_outcome, plan, peak_freqs, *_ = _two_window_plan_outcome()
        fit = plan_fit_outcome_to_spectrum_fit(
            plan_outcome,
            plan,
            sideband=SIDEBAND,
            peak_frequencies_mhz=peak_freqs,
            acquisition_us=T_US,
        )
        for window_fit in fit.window_fits:
            entry = window_fit.shared_parameters["tau_us"]
            assert entry["value"] > 0
            # tau is free in both windows here, so the error is finite (or None
            # if the covariance was singular -- unlikely on these synthetics).
            assert entry["error"] is None or entry["error"] > 0

    def test_fixed_parameters_record_frozen_contributors(self):
        """A dependent window's frozen contributors appear in fixed_parameters."""
        plan_outcome, plan, peak_freqs, *_ = _two_window_plan_outcome()
        fit = plan_fit_outcome_to_spectrum_fit(
            plan_outcome,
            plan,
            sideband=SIDEBAND,
            peak_frequencies_mhz=peak_freqs,
            acquisition_us=T_US,
        )
        # Window 0 has no fixed contributors; window 1 has one.
        win0 = fit.window_fit(0)
        win1 = fit.window_fit(1)
        assert win0.fixed_parameters == {}
        assert "frozen_peak_0" in win1.fixed_parameters
        entry = win1.fixed_parameters["frozen_peak_0"]
        assert entry["peak_index"] == 0
        assert entry["primary_window_id"] == 0


# ---------------------------------------------------------------------------
# (c) Thaw events partition per-window vs at plan level
# ---------------------------------------------------------------------------
class TestThawEventPartition:
    def test_clean_outcome_has_no_thaw_events(self):
        """A clean fit produces no thaw events on either layer."""
        plan_outcome, plan, peak_freqs, *_ = _two_window_plan_outcome()
        fit = plan_fit_outcome_to_spectrum_fit(
            plan_outcome,
            plan,
            sideband=SIDEBAND,
            peak_frequencies_mhz=peak_freqs,
            acquisition_us=T_US,
        )
        assert fit.thaw_history == []
        for window_fit in fit.window_fits:
            assert window_fit.thaw_events == []

    def test_thaw_event_lands_on_dependent_window_and_plan(self):
        """A triggered thaw appears in plan-level history and on dependent
        window's FittingResult.thaw_events."""
        # Reuse the canonical coupled-pair fixture: clean fit, then mutate
        # the primary's amplitude so the dependent's frozen background is
        # wrong and the residual edge flags.
        rng = np.random.default_rng(SEED + 200)
        sigma = 1.0
        strong_freq, weak_freq = 36100.0, 36104.0
        strong_amp = _amp_for_snr(800.0, sigma)
        weak_amp = _amp_for_snr(100.0, sigma)
        freq_array = np.arange(strong_freq - 5.0, weak_freq + 5.0, DF_MHZ)
        spectrum = _synth_spectrum(
            freq_array,
            [(strong_freq, strong_amp, 0.3), (weak_freq, weak_amp, 1.7)],
        ) + _complex_noise(freq_array.size, sigma, rng)
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
            fixed_contributors=[FixedContributor(0, 0, strong_freq, True)],
            batch=1,
        )
        plan = WindowPlan(
            windows=[win_a, win_b],
            dependency_edges=[(1, 0)],
            topological_order=[0, 1],
        )
        plan_outcome = execute_plan(
            plan,
            _make_active_ft(freq_array, spectrum),
            rms_noise,
            [strong_freq, weak_freq],
            sideband=SIDEBAND,
            acquisition_us=T_US,
            tau0_us=TAU_US,
        )
        # Corrupt the primary's fit, recompute the dependent's frozen
        # background and residual, then drive one thaw round.
        primary = plan_outcome.window_outcomes[0]
        primary.fit.peaks[0].amplitude = primary.fit.peaks[0].amplitude * 1.25
        dep = plan_outcome.window_outcomes[1]
        dep_center = 0.5 * (weak_freq - 0.6 + weak_freq + 0.6)
        from ftmwpipeline.fitting.peak_model import model_spectrum

        corrupted_frozen = evaluate_fixed_contributor(
            FixedContributor(0, 0, strong_freq, True),
            primary,
            dependent_center_mhz=dep_center,
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
        low, high = residual_edge_coherence(dep.full_residual, dep.rms_noise)
        dep.edge_coherence_low = low
        dep.edge_coherence_high = high
        events = attempt_thaw_round(
            win_b,
            dep,
            outcomes=plan_outcome.window_outcomes,
            sideband=SIDEBAND,
            acquisition_us=T_US,
            tau0_us=TAU_US,
        )
        assert events, "expected at least one thaw event for the fixture"
        # Mirror the events into the plan-level history so the converter
        # sees them (execute_plan would do this if the loop were run inline).
        plan_outcome.thaw_history.extend(events)

        fit = plan_fit_outcome_to_spectrum_fit(
            plan_outcome,
            plan,
            sideband=SIDEBAND,
            peak_frequencies_mhz=[strong_freq, weak_freq],
            acquisition_us=T_US,
        )
        # Plan-level history matches the events we recorded.
        assert len(fit.thaw_history) == len(events)
        # Per-window FittingResult.thaw_events on the dependent mirrors them;
        # the primary's FittingResult.thaw_events stays empty (the primary
        # window does not store dependent-side thaws on its own outcome).
        win0_fit = fit.window_fit(0)
        win1_fit = fit.window_fit(1)
        assert win0_fit.thaw_events == []
        assert len(win1_fit.thaw_events) == len(events)
        for persisted, original in zip(win1_fit.thaw_events, events):
            assert isinstance(persisted, ThawInfo)
            assert persisted.dependent_window_id == original.dependent_window_id
            assert persisted.edge_side == original.edge_side


# ---------------------------------------------------------------------------
# (d) Merged global peak list sorted + tagged
# ---------------------------------------------------------------------------
class TestMergedGlobalPeakList:
    def test_global_list_sorted_by_frequency(self):
        plan_outcome, plan, peak_freqs, *_ = _two_window_plan_outcome(
            strong_freq=36100.0, weak_freq=36110.0
        )
        fit = plan_fit_outcome_to_spectrum_fit(
            plan_outcome,
            plan,
            sideband=SIDEBAND,
            peak_frequencies_mhz=peak_freqs,
            acquisition_us=T_US,
        )
        freqs = [p.frequency_mhz for p in fit.fitted_peaks]
        assert freqs == sorted(freqs)

    def test_each_peak_tagged_with_originating_window_id(self):
        plan_outcome, plan, peak_freqs, *_ = _two_window_plan_outcome()
        fit = plan_fit_outcome_to_spectrum_fit(
            plan_outcome,
            plan,
            sideband=SIDEBAND,
            peak_frequencies_mhz=peak_freqs,
            acquisition_us=T_US,
        )
        window_ids = {p.window_id for p in fit.fitted_peaks}
        # Every peak carries a window id, and only ids in the plan show up.
        assert None not in window_ids
        plan_ids = {w.window_id for w in plan.windows}
        assert window_ids <= plan_ids

    def test_per_window_peaks_match_merged_subset(self):
        """The merged list partitioned by window_id equals each FittingResult's
        ``fitted_peaks`` (after sorting by frequency)."""
        plan_outcome, plan, peak_freqs, *_ = _two_window_plan_outcome()
        fit = plan_fit_outcome_to_spectrum_fit(
            plan_outcome,
            plan,
            sideband=SIDEBAND,
            peak_frequencies_mhz=peak_freqs,
            acquisition_us=T_US,
        )
        for window_fit in fit.window_fits:
            wid = window_fit.window_id
            from_merged = sorted(
                (p for p in fit.fitted_peaks if p.window_id == wid),
                key=lambda p: p.frequency_mhz,
            )
            from_per_window = sorted(
                window_fit.fitted_peaks, key=lambda p: p.frequency_mhz
            )
            assert len(from_merged) == len(from_per_window)
            for a, b in zip(from_merged, from_per_window):
                assert a.frequency_mhz == b.frequency_mhz
                assert a.peak_id == b.peak_id


# ---------------------------------------------------------------------------
# (e) final_plan_revision propagation + parameters
# ---------------------------------------------------------------------------
class TestSpectrumFitMetadata:
    def test_final_plan_revision_propagates_from_outcome(self):
        """``SpectrumFit.final_plan_revision`` == ``PlanFitOutcome.final_plan_revision``."""
        plan_outcome, plan, peak_freqs, *_ = _two_window_plan_outcome()
        # No structural replan -> revision is 0.
        fit = plan_fit_outcome_to_spectrum_fit(
            plan_outcome,
            plan,
            sideband=SIDEBAND,
            peak_frequencies_mhz=peak_freqs,
            acquisition_us=T_US,
        )
        assert fit.final_plan_revision == plan_outcome.final_plan_revision == 0

    def test_final_plan_revision_after_structural_replan(self):
        """A structural merge bumps the revision; SpectrumFit carries it."""
        sigma = 1.0
        peak_freq = 36104.9
        rng = np.random.default_rng(SEED + 300)
        freq_array = np.arange(36100.0, 36120.0 + DF_MHZ / 2, DF_MHZ)
        amp = _amp_for_snr(300.0, sigma)
        spectrum = _synth_spectrum(freq_array, [(peak_freq, amp, 0.3)])
        spectrum = spectrum + _complex_noise(freq_array.size, sigma, rng)
        rms = np.full(freq_array.size, sigma)
        gi = int(np.argmin(np.abs(freq_array - peak_freq)))
        peaks = [_make_peak(peak_freq, 300.0, sigma, grid_index=gi)]
        win_a = FitWindow(
            window_id=0,
            freq_range=(36100.0, 36105.0),
            free_peak_indices=[0],
            difficulty=WindowDifficulty.EASY,
            batch=0,
        )
        win_b = FitWindow(
            window_id=1,
            freq_range=(36105.0, 36110.0),
            free_peak_indices=[],
            difficulty=WindowDifficulty.EASY,
            batch=0,
        )
        plan = WindowPlan(
            windows=[win_a, win_b],
            topological_order=[0, 1],
            parameters=dict(_STAGE4_PARAMS),
        )
        ctx = ReplanContext(
            peaks=peaks,
            persisted_freq_mhz=freq_array,
            persisted_complex_spectrum=spectrum,
            persisted_rms_noise=rms,
        )
        plan_outcome = execute_plan(
            plan,
            _make_active_ft(freq_array, spectrum),
            rms,
            [peak_freq],
            sideband=SIDEBAND,
            acquisition_us=T_US,
            tau0_us=TAU_US,
            replan_context=ctx,
        )
        assert plan_outcome.final_plan_revision == 1
        assert plan_outcome.replan_history, "expected the merge to fire"

        # The merged plan keeps only the surviving window id (0); the
        # converter walks the *outcome* keys, so we hand it a 1-window plan
        # describing the post-merge state.
        merged_window = FitWindow(
            window_id=0,
            freq_range=(36100.0, 36110.0),
            free_peak_indices=[0],
            difficulty=WindowDifficulty.EASY,
            batch=0,
        )
        merged_plan = WindowPlan(
            windows=[merged_window],
            topological_order=[0],
            plan_revision=plan_outcome.final_plan_revision,
        )
        fit = plan_fit_outcome_to_spectrum_fit(
            plan_outcome,
            merged_plan,
            sideband=SIDEBAND,
            peak_frequencies_mhz=[peak_freq],
            acquisition_us=T_US,
        )
        assert fit.final_plan_revision == 1
        # Replan history was converted into the persistent twin.
        assert len(fit.replan_history) == len(plan_outcome.replan_history)
        assert all(isinstance(e, ReplanInfo) for e in fit.replan_history)
        assert any(e.accepted for e in fit.replan_history)

    def test_parameters_and_diagnostics_propagate(self):
        plan_outcome, plan, peak_freqs, *_ = _two_window_plan_outcome()
        params = {"tau0_us": TAU_US, "residual_edge_threshold": 1.5}
        diagnostics = {"note": "synthetic test"}
        fit = plan_fit_outcome_to_spectrum_fit(
            plan_outcome,
            plan,
            sideband=SIDEBAND,
            peak_frequencies_mhz=peak_freqs,
            acquisition_us=T_US,
            parameters=params,
            diagnostics=diagnostics,
        )
        assert fit.parameters == params
        # The dict is copied, not aliased.
        assert fit.parameters is not params
        assert fit.diagnostics == diagnostics


# ---------------------------------------------------------------------------
# SpectralWindow materialization
# ---------------------------------------------------------------------------
class TestSpectralWindowMaterialization:
    def test_window_carries_molecular_freq_and_complex_spectrum(self):
        plan_outcome, plan, peak_freqs, *_ = _two_window_plan_outcome()
        outcome = plan_outcome.window_outcomes[0]
        win_a = plan.window(0)

        spectral_window = window_outcome_to_spectral_window(
            outcome, win_a, sideband=SIDEBAND
        )
        assert isinstance(spectral_window, SpectralWindow)
        # No parent ComplexFT: the active-FT is not persisted.
        assert spectral_window.parent_ft is None
        # Window carries the FitWindow's freq_range and window_id directly.
        assert spectral_window.freq_range == tuple(win_a.freq_range)
        assert spectral_window.window_id == win_a.window_id
        # The molecular grid is the inverse of baseband_offset on the
        # outcome's offset grid.
        center = 0.5 * (win_a.freq_range[0] + win_a.freq_range[1])
        expected = molecular_frequency(outcome.offset_grid_mhz, center, SIDEBAND)
        np.testing.assert_allclose(spectral_window.freq_array, expected)
        # Complex spectrum is the bin-aligned active-FT data the fit saw.
        np.testing.assert_allclose(
            spectral_window.complex_spectrum, outcome.complex_spectrum
        )

    def test_window_attached_to_fitting_result(self):
        plan_outcome, plan, peak_freqs, *_ = _two_window_plan_outcome()
        fit = plan_fit_outcome_to_spectrum_fit(
            plan_outcome,
            plan,
            sideband=SIDEBAND,
            peak_frequencies_mhz=peak_freqs,
            acquisition_us=T_US,
        )
        for window_fit in fit.window_fits:
            assert isinstance(window_fit.window, SpectralWindow)
            assert window_fit.window.window_id == window_fit.window_id


# ---------------------------------------------------------------------------
# Direct single-window conversion (no plan walk)
# ---------------------------------------------------------------------------
class TestSingleWindowConversion:
    def test_window_outcome_to_fitting_result_standalone(self):
        plan_outcome, plan, peak_freqs, *_ = _two_window_plan_outcome()
        outcome_0 = plan_outcome.window_outcomes[0]
        win_0 = plan.window(0)
        result = window_outcome_to_fitting_result(
            outcome_0,
            win_0,
            sideband=SIDEBAND,
            peak_frequencies_mhz=peak_freqs,
            acquisition_us=T_US,
        )
        assert isinstance(result, FittingResult)
        assert result.window_id == 0
        assert result.success
        assert result.n_peaks_fitted == 1
        assert isinstance(result.fitted_peaks[0], FittedPeak)

    def test_raises_when_center_missing(self):
        """A WindowOutcome without ``_center_mhz`` is rejected with a clear msg."""
        plan_outcome, plan, peak_freqs, *_ = _two_window_plan_outcome()
        outcome_0 = plan_outcome.window_outcomes[0]
        delattr(outcome_0, "_center_mhz")
        with pytest.raises(ValueError, match="molecular reference"):
            window_outcome_to_fitting_result(
                outcome_0,
                plan.window(0),
                sideband=SIDEBAND,
                peak_frequencies_mhz=peak_freqs,
                acquisition_us=T_US,
            )

    def test_unknown_window_id_in_outcome_raises(self):
        """An outcome keyed by a ``window_id`` not in the plan is a programmer
        error (mismatched plan revisions)."""
        plan_outcome, plan, peak_freqs, *_ = _two_window_plan_outcome()
        plan_outcome.window_outcomes[99] = plan_outcome.window_outcomes[0]
        with pytest.raises(KeyError, match="window_id=99"):
            plan_fit_outcome_to_spectrum_fit(
                plan_outcome,
                plan,
                sideband=SIDEBAND,
                peak_frequencies_mhz=peak_freqs,
                acquisition_us=T_US,
            )
