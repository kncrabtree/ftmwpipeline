"""
Unit tests for the Phase-A peak-survival SNR-floor prune.

Tests the ``apply_snr_survival_prune`` helper directly, without a real fit.
"""

from __future__ import annotations

import math

import pytest

from ftmwpipeline._internal.stage5_impl import apply_snr_survival_prune
from ftmwpipeline.core.data_structures import (
    FittedPeak,
    FittingResult,
    SpectrumFit,
)


def _make_peak(
    freq: float,
    snr: float | None,
    origin: str = "auto",
    window_id: int = 1,
) -> FittedPeak:
    return FittedPeak(
        peak_id=0,
        frequency_mhz=freq,
        amplitude=1.0,
        snr=snr,
        window_id=window_id,
        origin=origin,
    )


def _make_window(
    window_id: int, peaks: list[FittedPeak]
) -> FittingResult:
    wf = FittingResult(window_id=window_id)
    wf.fitted_peaks = list(peaks)
    return wf


def _make_fit(windows: list[FittingResult]) -> SpectrumFit:
    all_peaks = sorted(
        [p for wf in windows for p in wf.fitted_peaks],
        key=lambda p: p.frequency_mhz,
    )
    fit = SpectrumFit(window_fits=list(windows), fitted_peaks=all_peaks)
    return fit


class TestApplySnrSurvivalPrune:
    FLOOR = 3.2

    def test_removes_sub_floor_auto_peaks(self):
        """Auto-origin peaks below the floor are removed."""
        w1 = _make_window(1, [
            _make_peak(1000.0, snr=10.0),  # keep
            _make_peak(1001.0, snr=2.0),   # drop
            _make_peak(1002.0, snr=3.2),   # keep (== floor)
        ])
        fit = _make_fit([w1])
        apply_snr_survival_prune(fit, self.FLOOR)
        assert len(fit.window_fits) == 1
        assert len(fit.window_fits[0].fitted_peaks) == 2
        freqs = [p.frequency_mhz for p in fit.window_fits[0].fitted_peaks]
        assert 1001.0 not in freqs

    def test_user_origin_immune_to_prune(self):
        """A user-origin peak below the floor is always kept."""
        w1 = _make_window(1, [
            _make_peak(1000.0, snr=1.0, origin="user"),  # keep (user)
            _make_peak(1001.0, snr=1.0, origin="auto"),  # drop (auto, sub-floor)
        ])
        fit = _make_fit([w1])
        apply_snr_survival_prune(fit, self.FLOOR)
        assert len(fit.window_fits) == 1
        remaining_freqs = [p.frequency_mhz for p in fit.window_fits[0].fitted_peaks]
        assert 1000.0 in remaining_freqs
        assert 1001.0 not in remaining_freqs

    def test_none_snr_kept_unconditionally(self):
        """Peaks with None snr are not prunable."""
        w1 = _make_window(1, [_make_peak(1000.0, snr=None)])
        fit = _make_fit([w1])
        apply_snr_survival_prune(fit, self.FLOOR)
        assert len(fit.window_fits) == 1
        assert len(fit.window_fits[0].fitted_peaks) == 1

    def test_nan_snr_kept_unconditionally(self):
        """Peaks with NaN snr are not prunable."""
        w1 = _make_window(1, [_make_peak(1000.0, snr=float("nan"))])
        fit = _make_fit([w1])
        apply_snr_survival_prune(fit, self.FLOOR)
        assert len(fit.window_fits) == 1
        assert len(fit.window_fits[0].fitted_peaks) == 1

    def test_all_dust_window_dropped(self):
        """A window whose every peak is sub-floor dust is removed from window_fits."""
        w1 = _make_window(1, [_make_peak(1000.0, snr=10.0)])  # keep window
        w2 = _make_window(2, [
            _make_peak(2000.0, snr=1.5),   # dust
            _make_peak(2001.0, snr=2.4),   # dust
        ])
        fit = _make_fit([w1, w2])
        apply_snr_survival_prune(fit, self.FLOOR)
        assert len(fit.window_fits) == 1
        assert fit.window_fits[0].window_id == 1

    def test_preexisting_empty_window_preserved(self):
        """A window that was already empty (K=0) before the prune is NOT
        dropped -- the survival prune only drops windows it empties itself
        (pre-existing empty windows are Stage 4/5 / window-construction signal)."""
        w_empty = _make_window(1, [])  # K=0 before any prune
        w2 = _make_window(2, [_make_peak(2000.0, snr=8.0, window_id=2)])
        fit = _make_fit([w_empty, w2])
        apply_snr_survival_prune(fit, self.FLOOR)
        ids = {wf.window_id for wf in fit.window_fits}
        assert ids == {1, 2}
        assert fit.diagnostics["peak_survival"]["dropped_window_ids"] == []

    def test_covariance_sliced_after_partial_prune(self):
        """A partial prune slices the window covariance to the survivors and
        rebuilds contiguous labels, preserving the per-peak invariant
        (amplitude-label count == fitted-peak count)."""
        import numpy as np

        from ftmwpipeline.fitting.result_conversion import (
            build_covariance_param_labels,
        )

        # Two peaks + a shared tau: labels A0,o0,p0,A1,o1,p1,tau -> 7x7.
        labels = build_covariance_param_labels(2, fit_tau=True, baseline_order=None)
        cov = np.diag(np.arange(1, len(labels) + 1).astype(float))
        wf = _make_window(1, [
            _make_peak(1000.0, snr=10.0, window_id=1),  # kept (idx 0)
            _make_peak(1001.0, snr=2.0, window_id=1),   # pruned (idx 1)
        ])
        wf.covariance = cov
        wf.covariance_param_labels = labels
        fit = _make_fit([wf])
        apply_snr_survival_prune(fit, self.FLOOR)
        out = fit.window_fits[0]
        assert len(out.fitted_peaks) == 1
        assert out.covariance is not None
        # Survivor params: A0,o0,p0 (idx 0,1,2) + tau (idx 6) -> 4x4.
        assert out.covariance.shape == (4, 4)
        assert out.covariance_param_labels == [
            "amplitude_0", "offset_0", "phase_0", "tau",
        ]
        # Diagonal entries preserved for the kept indices (1,2,3 for the peak,
        # 7 for tau, in the 1-based fill above).
        np.testing.assert_array_equal(
            np.diag(out.covariance), np.array([1.0, 2.0, 3.0, 7.0])
        )

    def test_fitted_peaks_rebuilt_sorted(self):
        """fit.fitted_peaks is rebuilt sorted by frequency after the prune."""
        w1 = _make_window(1, [
            _make_peak(3000.0, snr=10.0, window_id=1),
            _make_peak(1000.0, snr=2.0, window_id=1),  # drop
        ])
        w2 = _make_window(2, [
            _make_peak(2000.0, snr=8.0, window_id=2),
        ])
        fit = _make_fit([w1, w2])
        apply_snr_survival_prune(fit, self.FLOOR)
        freqs = [p.frequency_mhz for p in fit.fitted_peaks]
        assert freqs == sorted(freqs)
        assert 1000.0 not in freqs

    def test_diagnostics_populated(self):
        """diagnostics['peak_survival'] is populated with snr_floor, n_pruned, etc."""
        w1 = _make_window(1, [
            _make_peak(1000.0, snr=10.0),
            _make_peak(1001.0, snr=2.0),  # dust
        ])
        w_dust = _make_window(2, [_make_peak(2000.0, snr=1.0)])  # whole window dropped
        fit = _make_fit([w1, w_dust])
        apply_snr_survival_prune(fit, self.FLOOR)
        ps = fit.diagnostics["peak_survival"]
        assert ps["snr_floor"] == pytest.approx(self.FLOOR)
        assert ps["n_pruned"] == 2
        assert len(ps["pruned"]) == 2
        assert 2 in ps["dropped_window_ids"]

    def test_no_peaks_pruned_diagnostics_still_populated(self):
        """diagnostics is populated even when nothing is pruned."""
        w1 = _make_window(1, [_make_peak(1000.0, snr=10.0)])
        fit = _make_fit([w1])
        apply_snr_survival_prune(fit, self.FLOOR)
        ps = fit.diagnostics["peak_survival"]
        assert ps["n_pruned"] == 0
        assert ps["pruned"] == []
        assert ps["dropped_window_ids"] == []

    def test_pruned_records_carry_expected_fields(self):
        """Each pruned record has window_id, frequency_mhz, and snr."""
        w1 = _make_window(42, [_make_peak(1234.5, snr=1.5, window_id=42)])
        fit = _make_fit([w1])
        apply_snr_survival_prune(fit, self.FLOOR)
        assert fit.window_fits == []  # window dropped
        ps = fit.diagnostics["peak_survival"]
        assert len(ps["pruned"]) == 1
        rec = ps["pruned"][0]
        assert rec["window_id"] == 42
        assert rec["frequency_mhz"] == pytest.approx(1234.5)
        assert rec["snr"] == pytest.approx(1.5)


class TestSettingsWiring:
    """Verify the settings dataclass and hard defaults."""

    def test_hard_defaults_enabled_true(self):
        from ftmwpipeline.core.stage_fit_settings import resolve
        resolved = resolve()
        assert resolved.peak_survival.enabled is True

    def test_hard_default_snr_floor(self):
        from ftmwpipeline.core.stage_fit_settings import resolve
        resolved = resolve()
        assert resolved.peak_survival.snr_survival_floor == pytest.approx(3.2)

    def test_persisted_beats_default(self):
        """If a StageFitSettings with a different floor is passed as persisted,
        it outranks the hard default."""
        from ftmwpipeline.core.stage_fit_settings import (
            PeakSurvivalSubSettings,
            StageFitSettings,
            resolve,
        )
        persisted = StageFitSettings(
            peak_survival=PeakSurvivalSubSettings(snr_survival_floor=5.0)
        )
        resolved = resolve(persisted=persisted)
        assert resolved.peak_survival.snr_survival_floor == pytest.approx(5.0)

    def test_explicit_beats_persisted(self):
        from ftmwpipeline.core.stage_fit_settings import (
            PeakSurvivalSubSettings,
            StageFitSettings,
            resolve,
        )
        persisted = StageFitSettings(
            peak_survival=PeakSurvivalSubSettings(snr_survival_floor=5.0)
        )
        explicit = StageFitSettings(
            peak_survival=PeakSurvivalSubSettings(snr_survival_floor=7.0)
        )
        resolved = resolve(explicit=explicit, persisted=persisted)
        assert resolved.peak_survival.snr_survival_floor == pytest.approx(7.0)


class TestSettingsRoundTrip:
    """Verify the new sub-group round-trips through HDF5."""

    def test_hdf5_round_trip_peak_survival(self, tmp_path):
        import h5py
        from ftmwpipeline.core.stage_fit_settings import (
            PeakSurvivalSubSettings,
            StageFitSettings,
            resolve,
        )
        from ftmwpipeline.io.stage_fit_settings_serialization import (
            load_stage_fit_settings_from_h5,
            save_stage_fit_settings_to_h5,
        )

        p = tmp_path / "exp.ftmw"
        with h5py.File(p, "w") as h5f:
            h5f.create_group("placeholder")

        s = StageFitSettings(
            peak_survival=PeakSurvivalSubSettings(enabled=False, snr_survival_floor=5.0)
        )
        save_stage_fit_settings_to_h5(str(p), s)
        loaded = load_stage_fit_settings_from_h5(str(p))
        assert loaded is not None
        assert loaded.peak_survival.enabled == False  # noqa: E712 (np.False_ != is False)
        assert loaded.peak_survival.snr_survival_floor == pytest.approx(5.0)

    def test_resolved_hdf5_round_trip(self, tmp_path):
        """A fully resolved StageFitSettings round-trips the peak_survival sub-block."""
        import h5py
        from ftmwpipeline.core.stage_fit_settings import resolve
        from ftmwpipeline.io.stage_fit_settings_serialization import (
            load_stage_fit_settings_from_h5,
            save_stage_fit_settings_to_h5,
        )

        p = tmp_path / "exp.ftmw"
        with h5py.File(p, "w") as h5f:
            h5f.create_group("placeholder")

        resolved = resolve()
        save_stage_fit_settings_to_h5(str(p), resolved)
        loaded = load_stage_fit_settings_from_h5(str(p))
        assert loaded is not None
        assert loaded.peak_survival.enabled == True  # noqa: E712 (np.True_ != is True)
        assert loaded.peak_survival.snr_survival_floor == pytest.approx(3.2)
