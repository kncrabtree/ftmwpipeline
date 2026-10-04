"""An undefined residual edge coherence is ``nan``, and every gate reads it as 0.

Spec (``dev-docs/CONTRACT_STRATEGY.md`` §Missing values, "Degenerate statistics
are UNDEFINED"): the edge coherence of an empty residual or of a band with no
positive noise is undefined, stored as ``nan``, and "gate decisions do not
change". Before, the statistic was ``0.0`` there, which no gate (all of the form
``S_coh > threshold``) ever flagged. These tests pin the statistic itself and
each gate that reads it: the thaw and structural-merge triggers, a thaw's
acceptance, the late-baseline trigger and its recorded ``baseline_edge_coherence``.
"""

from __future__ import annotations

import numpy as np
import pytest

from ftmwpipeline.core.data_structures import (
    FitWindow,
    FixedContributor,
    WindowPlan,
)
from ftmwpipeline.fitting import plan_execution
from ftmwpipeline.fitting.peak_model import ModelPeak, model_spectrum, sideband_sign
from ftmwpipeline.fitting.plan_execution import (
    DEFAULT_RESIDUAL_EDGE_THRESHOLD,
    FrozenPeak,
    WindowOutcome,
    _apply_baseline_to_outcome,
    attempt_thaw_round,
    edge_gate_value,
    execute_plan,
    residual_edge_coherence,
)
from ftmwpipeline.preprocessing.edge_coherence import (
    coherence_statistic,
    rolling_coherence,
)
from tests.unit.fitting.test_plan_execution import (
    DF_MHZ,
    SEED,
    SIDEBAND,
    T_US,
    TAU_US,
    _amp_for_snr,
    _baseline_outcome,
    _complex_noise,
    _make_active_ft,
    _synth_spectrum,
)

NAN = float("nan")


# ---------------------------------------------------------------------------
# The statistic and the gate view of it
# ---------------------------------------------------------------------------
class TestEdgeGateValue:
    def test_undefined_reads_as_zero(self):
        assert edge_gate_value(NAN) == 0.0

    @pytest.mark.parametrize("value", [0.0, 0.4, 1.0, DEFAULT_RESIDUAL_EDGE_THRESHOLD])
    def test_a_defined_edge_passes_through(self, value):
        assert edge_gate_value(value) == value

    def test_an_undefined_edge_never_flags(self):
        """The old stored 0.0 never exceeded a (positive) threshold; nor does nan."""
        assert not edge_gate_value(NAN) > DEFAULT_RESIDUAL_EDGE_THRESHOLD
        assert edge_gate_value(NAN) <= DEFAULT_RESIDUAL_EDGE_THRESHOLD


class TestUndefinedStatistic:
    def test_empty_residual_is_undefined_on_both_edges(self):
        low, high = residual_edge_coherence(np.array([], dtype=complex), 1.0)
        assert np.isnan(low) and np.isnan(high)

    def test_band_without_positive_noise_is_undefined(self):
        z = np.ones(40, dtype=complex)
        low, high = residual_edge_coherence(z, np.zeros(40), band_m=8)
        assert np.isnan(low) and np.isnan(high)

    def test_one_edge_can_be_undefined_beside_a_defined_one(self):
        z = np.ones(40, dtype=complex)
        sigma = np.ones(40)
        sigma[:8] = 0.0  # the low band has no positive noise
        low, high = residual_edge_coherence(z, sigma, band_m=8)
        assert np.isnan(low) and np.isfinite(high)

    def test_a_defined_statistic_is_unchanged(self):
        z = np.ones(40, dtype=complex)
        low, high = residual_edge_coherence(z, 1.0, band_m=16)
        assert low == pytest.approx(np.sqrt(16))
        assert high == pytest.approx(np.sqrt(16))

    def test_coherence_statistic_is_nan_for_degenerate_input(self):
        assert np.isnan(coherence_statistic(np.array([]), 1.0))
        assert np.isnan(coherence_statistic(np.array([1 + 1j]), 0.0))
        assert np.isnan(coherence_statistic(np.array([1 + 1j]), -1.0))

    def test_rolling_coherence_short_band_without_noise_still_scores_zero(self):
        """The rolling statistic's callers (Stage 3/4) keep ``0.0`` for a band
        with no noise: nothing to flag, as the vectorized path scores it."""
        z = np.ones(8, dtype=complex)
        out = rolling_coherence(z, np.zeros(8), band_m=64)
        assert out[4] == 0.0
        assert np.isnan(np.delete(out, 4)).all()

    def test_rolling_coherence_short_band_with_noise_is_the_statistic(self):
        z = np.ones(8, dtype=complex)
        out = rolling_coherence(z, np.ones(8), band_m=64)
        assert out[4] == pytest.approx(coherence_statistic(z, 1.0))

    def test_outcome_defaults_are_undefined_not_zero(self):
        fields = {f.name: f for f in WindowOutcome.__dataclass_fields__.values()}
        for name in ("edge_coherence_low", "edge_coherence_high"):
            assert np.isnan(fields[name].default)


# ---------------------------------------------------------------------------
# The late-baseline trigger and its recorded statistic
# ---------------------------------------------------------------------------
def _wing_outcome():
    """A window whose residual carries a coherent wing, so both edges flag."""
    u = np.arange(-160, 161) * 0.0122
    line = ModelPeak(amplitude=6.0, offset_mhz=0.2, phase=0.5)
    wing = 0.6 - 0.4j
    z = model_spectrum(u, [line], TAU_US, T_US) + wing
    sigma = np.full(u.size, 0.02)
    return _baseline_outcome(u, z, sigma, [ModelPeak(5.0, 0.0, 0.0)])


def _apply(outcome):
    # The smooth-residual trigger is set out of reach so only the edge fires it.
    return _apply_baseline_to_outcome(
        outcome,
        acquisition_us=T_US,
        residual_edge_m=16,
        baseline_order=0,
        baseline_edge_threshold=3.5,
        baseline_smooth_threshold=1e12,
        tau0_us=TAU_US,
        conservative_kwargs={},
    )


class TestLateBaselineTrigger:
    def test_one_undefined_edge_leaves_the_other_to_decide(self):
        outcome = _wing_outcome()
        high = outcome.edge_coherence_high
        assert high > 3.5
        outcome.edge_coherence_low = NAN
        assert _apply(outcome) is True
        assert outcome.baseline_applied is True
        # max(0.0, high): the recorded trigger value is the defined edge.
        assert outcome.baseline_edge_coherence == pytest.approx(high)

    def test_the_undefined_edge_is_not_the_one_that_fires(self):
        outcome = _wing_outcome()
        low = outcome.edge_coherence_low
        assert low > 3.5
        outcome.edge_coherence_high = NAN
        assert _apply(outcome) is True
        assert outcome.baseline_edge_coherence == pytest.approx(low)

    def test_both_edges_undefined_does_not_fire(self):
        outcome = _wing_outcome()
        outcome.edge_coherence_low = NAN
        outcome.edge_coherence_high = NAN
        assert _apply(outcome) is False
        assert outcome.baseline_applied is False
        assert outcome.baseline_coeffs is None

    def test_one_undefined_edge_below_threshold_on_the_other_does_not_fire(self):
        outcome = _wing_outcome()
        outcome.edge_coherence_low = NAN
        outcome.edge_coherence_high = 1.0
        assert _apply(outcome) is False

    def test_mirrored_baseline_stamp_reads_an_undefined_edge_as_zero(self):
        """``_mirror_outcome_baseline`` stamps ``max(low, high)`` with an
        undefined edge as 0.0 (finite, never nan)."""
        outcome = _wing_outcome()
        fired = _wing_outcome()
        assert _apply(fired) is True  # carries a real baseline fit
        outcome.fit = fired.fit
        outcome.edge_coherence_low = NAN
        outcome.edge_coherence_high = 2.5
        plan_execution._mirror_outcome_baseline(outcome)
        assert outcome.baseline_edge_coherence == 2.5
        outcome.edge_coherence_high = NAN
        plan_execution._mirror_outcome_baseline(outcome)
        assert outcome.baseline_edge_coherence == 0.0


# ---------------------------------------------------------------------------
# The thaw trigger and a thaw's acceptance
# ---------------------------------------------------------------------------
def _two_window_run():
    """A clean two-window plan, ready to corrupt for a thaw."""
    rng = np.random.default_rng(SEED + 10)
    sigma = 1.0
    strong_freq, weak_freq = 36100.0, 36104.0
    freq_array = np.arange(strong_freq - 5.0, weak_freq + 5.0, DF_MHZ)
    spectrum = _synth_spectrum(
        freq_array,
        [
            (strong_freq, _amp_for_snr(800.0, sigma), 0.3),
            (weak_freq, _amp_for_snr(100.0, sigma), 1.7),
        ],
    ) + _complex_noise(freq_array.size, sigma, rng)
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
        windows=[win_a, win_b], dependency_edges=[(1, 0)], topological_order=[0, 1]
    )
    out = execute_plan(
        plan,
        _make_active_ft(freq_array, spectrum),
        np.full(freq_array.size, sigma),
        [strong_freq, weak_freq],
        sideband=SIDEBAND,
        acquisition_us=T_US,
        tau0_us=TAU_US,
    )
    return out, win_a, win_b, strong_freq, weak_freq


def _corrupt_primary(out, win_a, win_b, strong_freq, weak_freq):
    """Bump the primary's amplitude 25% and re-derive the dependent's residual
    edges, as the established thaw test does (the over-amped frozen skirt flags
    the dependent's edge facing the primary)."""
    primary = out.window_outcomes[0]
    primary.fit.peaks[0].amplitude = primary.fit.peaks[0].amplitude * 1.25
    dep = out.window_outcomes[1]
    dep_center = 0.5 * (weak_freq - 0.3 + weak_freq + 0.3)
    s = sideband_sign(SIDEBAND)
    primary_center = 0.5 * (win_a.freq_range[0] + win_a.freq_range[1])
    peak = primary.fit.peaks[0]
    fitted_freq = primary_center + s * peak.offset_mhz
    frozen = FrozenPeak(
        peak_index=0,
        primary_window_id=0,
        model_peak=ModelPeak(
            amplitude=peak.amplitude,
            offset_mhz=s * (fitted_freq - dep_center),
            phase=peak.phase,
        ),
        frequency_mhz=fitted_freq,
        freeze_eligible=True,
    )
    dep.fixed_peaks = [frozen]
    dep.background = model_spectrum(
        dep.offset_grid_mhz, [frozen.model_peak], dep.fit.fit.tau_us, T_US
    )
    dep.full_fitted_spectrum = dep.fit.fit.fitted_spectrum + dep.background
    dep.full_residual = dep.complex_spectrum - dep.full_fitted_spectrum
    dep.edge_coherence_low, dep.edge_coherence_high = residual_edge_coherence(
        dep.full_residual, dep.rms_noise
    )
    assert max(dep.edge_coherence_low, dep.edge_coherence_high) > (
        DEFAULT_RESIDUAL_EDGE_THRESHOLD
    )
    return dep


def _round(out, win_b, dep):
    return attempt_thaw_round(
        win_b,
        dep,
        outcomes=out.window_outcomes,
        sideband=SIDEBAND,
        acquisition_us=T_US,
        tau0_us=TAU_US,
    )


class TestThawGates:
    def test_undefined_edges_trigger_no_thaw(self):
        out, _, win_b, *_ = _two_window_run()
        dep = out.window_outcomes[1]
        dep.edge_coherence_low = NAN
        dep.edge_coherence_high = NAN
        assert _round(out, win_b, dep) == []

    def test_an_undefined_edge_beside_a_flagged_one_still_thaws(self):
        out, win_a, win_b, strong, weak = _two_window_run()
        dep = _corrupt_primary(out, win_a, win_b, strong, weak)
        flagged = "low" if dep.edge_coherence_low >= dep.edge_coherence_high else "high"
        if flagged == "low":
            dep.edge_coherence_high = NAN
        else:
            dep.edge_coherence_low = NAN
        events = _round(out, win_b, dep)
        assert events and events[0].edge_side == flagged
        assert events[0].edge_coherence_before > DEFAULT_RESIDUAL_EDGE_THRESHOLD

    def test_thaw_accepts_when_the_edge_after_is_undefined(self, monkeypatch):
        """Acceptance is ``edge_after <= threshold`` with an undefined edge read
        as 0: the co-fit is accepted, and its ``edge_coherence_after`` is
        recorded as ``nan`` (the event has no status column)."""
        out, win_a, win_b, strong, weak = _two_window_run()
        dep = _corrupt_primary(out, win_a, win_b, strong, weak)
        monkeypatch.setattr(
            plan_execution, "residual_edge_coherence", lambda *a, **k: (NAN, NAN)
        )
        events = _round(out, win_b, dep)
        accepted = [e for e in events if e.accepted]
        assert accepted, [e.reason for e in events]
        assert np.isnan(accepted[0].edge_coherence_after)
        assert np.isfinite(accepted[0].edge_coherence_before)

    def test_thaw_rejects_when_the_edge_after_is_above_threshold(self, monkeypatch):
        """The same patch with a defined, flagged edge still rejects: the gate
        compares numbers as before."""
        out, win_a, win_b, strong, weak = _two_window_run()
        dep = _corrupt_primary(out, win_a, win_b, strong, weak)
        high = DEFAULT_RESIDUAL_EDGE_THRESHOLD + 1.0
        monkeypatch.setattr(
            plan_execution, "residual_edge_coherence", lambda *a, **k: (high, high)
        )
        events = _round(out, win_b, dep)
        assert events and not any(e.accepted for e in events)
        measured = [e for e in events if "did not lower" in e.reason]
        assert measured and all(e.edge_coherence_after == high for e in measured)
