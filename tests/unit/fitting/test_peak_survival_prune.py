"""
Unit tests for the Phase-A peak-survival SNR-floor prune.

Tests the ``apply_snr_survival_prune`` helper directly, without a real fit.
"""

from __future__ import annotations

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


def _make_window(window_id: int, peaks: list[FittedPeak]) -> FittingResult:
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


def _stub_refit(wf: FittingResult, dust_freqs: list[float]) -> FittingResult:
    """Stand-in for the production window-refit core.

    Returns a fresh :class:`FittingResult` for ``wf`` with the dust frequencies
    dropped from its peak list. It does no NLS (the orchestration under test
    does not depend on the converged parameters), and it clears the covariance
    to make the "the survivors come from a refit, not a slice" contract
    observable in the assertions.
    """
    kept = [p for p in wf.fitted_peaks if float(p.frequency_mhz) not in dust_freqs]
    new = FittingResult(window_id=wf.window_id)
    new.fitted_peaks = kept
    new.covariance = None
    new.covariance_param_labels = None
    return new


class TestApplySnrSurvivalPrune:
    FLOOR = 3.2

    def test_removes_sub_floor_auto_peaks(self):
        """Auto-origin peaks below the floor are removed."""
        w1 = _make_window(
            1,
            [
                _make_peak(1000.0, snr=10.0),  # keep
                _make_peak(1001.0, snr=2.0),  # drop
                _make_peak(1002.0, snr=3.2),  # keep (== floor)
            ],
        )
        fit = _make_fit([w1])
        apply_snr_survival_prune(fit, self.FLOOR, refit_window=_stub_refit)
        assert len(fit.window_fits) == 1
        assert len(fit.window_fits[0].fitted_peaks) == 2
        freqs = [p.frequency_mhz for p in fit.window_fits[0].fitted_peaks]
        assert 1001.0 not in freqs

    def test_user_origin_immune_to_prune(self):
        """A user-origin peak below the floor is always kept."""
        w1 = _make_window(
            1,
            [
                _make_peak(1000.0, snr=1.0, origin="user"),  # keep (user)
                _make_peak(1001.0, snr=1.0, origin="auto"),  # drop (auto, sub-floor)
            ],
        )
        fit = _make_fit([w1])
        apply_snr_survival_prune(fit, self.FLOOR, refit_window=_stub_refit)
        assert len(fit.window_fits) == 1
        remaining_freqs = [p.frequency_mhz for p in fit.window_fits[0].fitted_peaks]
        assert 1000.0 in remaining_freqs
        assert 1001.0 not in remaining_freqs

    def test_none_snr_kept_unconditionally(self):
        """Peaks with None snr are not prunable."""
        w1 = _make_window(1, [_make_peak(1000.0, snr=None)])
        fit = _make_fit([w1])
        apply_snr_survival_prune(fit, self.FLOOR, refit_window=_stub_refit)
        assert len(fit.window_fits) == 1
        assert len(fit.window_fits[0].fitted_peaks) == 1

    def test_nan_snr_kept_unconditionally(self):
        """Peaks with NaN snr are not prunable."""
        w1 = _make_window(1, [_make_peak(1000.0, snr=float("nan"))])
        fit = _make_fit([w1])
        apply_snr_survival_prune(fit, self.FLOOR, refit_window=_stub_refit)
        assert len(fit.window_fits) == 1
        assert len(fit.window_fits[0].fitted_peaks) == 1

    def test_all_dust_window_dropped(self):
        """A window whose every peak is sub-floor dust is removed from window_fits."""
        w1 = _make_window(1, [_make_peak(1000.0, snr=10.0)])  # keep window
        w2 = _make_window(
            2,
            [
                _make_peak(2000.0, snr=1.5),  # dust
                _make_peak(2001.0, snr=2.4),  # dust
            ],
        )
        fit = _make_fit([w1, w2])
        apply_snr_survival_prune(fit, self.FLOOR, refit_window=_stub_refit)
        assert len(fit.window_fits) == 1
        assert fit.window_fits[0].window_id == 1

    def test_preexisting_empty_window_preserved(self):
        """A window that was already empty (K=0) before the prune is NOT
        dropped -- the survival prune only drops windows it empties itself
        (pre-existing empty windows are Stage 4/5 / window-construction signal)."""
        w_empty = _make_window(1, [])  # K=0 before any prune
        w2 = _make_window(2, [_make_peak(2000.0, snr=8.0, window_id=2)])
        fit = _make_fit([w_empty, w2])
        apply_snr_survival_prune(fit, self.FLOOR, refit_window=_stub_refit)
        ids = {wf.window_id for wf in fit.window_fits}
        assert ids == {1, 2}
        assert fit.diagnostics["peak_survival"]["dropped_window_ids"] == []

    def test_partial_prune_refits_window(self):
        """A partial prune replaces the window with the refit's result and
        passes exactly the dust frequencies to the refit (it does not slice the
        stale joint covariance)."""
        import numpy as np

        captured: dict = {}

        def _capturing_refit(
            wf: FittingResult, dust_freqs: list[float]
        ) -> FittingResult:
            captured["window_id"] = wf.window_id
            captured["dust_freqs"] = list(dust_freqs)
            return _stub_refit(wf, dust_freqs)

        wf = _make_window(
            1,
            [
                _make_peak(1000.0, snr=10.0, window_id=1),  # kept
                _make_peak(1001.0, snr=2.0, window_id=1),  # dust
            ],
        )
        # A stale joint covariance from the pre-prune 2-peak fit.
        wf.covariance = np.eye(7)
        wf.covariance_param_labels = ["x"] * 7
        fit = _make_fit([wf])
        apply_snr_survival_prune(fit, self.FLOOR, refit_window=_capturing_refit)

        assert captured["window_id"] == 1
        assert captured["dust_freqs"] == [1001.0]
        out = fit.window_fits[0]
        assert len(out.fitted_peaks) == 1
        assert out.fitted_peaks[0].frequency_mhz == 1000.0
        # The result is the refit's window, not a slice of the stale matrix.
        assert out.covariance is None
        assert out.covariance_param_labels is None

    def test_no_dust_window_not_refitted(self):
        """A window with no sub-floor peaks is left untouched (no refit call)."""
        called = {"n": 0}

        def _counting_refit(
            wf: FittingResult, dust_freqs: list[float]
        ) -> FittingResult:
            called["n"] += 1
            return _stub_refit(wf, dust_freqs)

        wf = _make_window(
            1,
            [
                _make_peak(1000.0, snr=10.0, window_id=1),
                _make_peak(1001.0, snr=8.0, window_id=1),
            ],
        )
        fit = _make_fit([wf])
        apply_snr_survival_prune(fit, self.FLOOR, refit_window=_counting_refit)
        assert called["n"] == 0
        assert fit.window_fits[0] is wf  # same object, untouched

    def test_fitted_peaks_rebuilt_sorted(self):
        """fit.fitted_peaks is rebuilt sorted by frequency after the prune."""
        w1 = _make_window(
            1,
            [
                _make_peak(3000.0, snr=10.0, window_id=1),
                _make_peak(1000.0, snr=2.0, window_id=1),  # drop
            ],
        )
        w2 = _make_window(
            2,
            [
                _make_peak(2000.0, snr=8.0, window_id=2),
            ],
        )
        fit = _make_fit([w1, w2])
        apply_snr_survival_prune(fit, self.FLOOR, refit_window=_stub_refit)
        freqs = [p.frequency_mhz for p in fit.fitted_peaks]
        assert freqs == sorted(freqs)
        assert 1000.0 not in freqs

    def test_diagnostics_populated(self):
        """diagnostics['peak_survival'] is populated with snr_floor, n_pruned, etc."""
        w1 = _make_window(
            1,
            [
                _make_peak(1000.0, snr=10.0),
                _make_peak(1001.0, snr=2.0),  # dust
            ],
        )
        w_dust = _make_window(2, [_make_peak(2000.0, snr=1.0)])  # whole window dropped
        fit = _make_fit([w1, w_dust])
        apply_snr_survival_prune(fit, self.FLOOR, refit_window=_stub_refit)
        ps = fit.diagnostics["peak_survival"]
        assert ps["snr_floor"] == pytest.approx(self.FLOOR)
        assert ps["n_pruned"] == 2
        assert len(ps["pruned"]) == 2
        assert 2 in ps["dropped_window_ids"]

    def test_no_peaks_pruned_diagnostics_still_populated(self):
        """diagnostics is populated even when nothing is pruned."""
        w1 = _make_window(1, [_make_peak(1000.0, snr=10.0)])
        fit = _make_fit([w1])
        apply_snr_survival_prune(fit, self.FLOOR, refit_window=_stub_refit)
        ps = fit.diagnostics["peak_survival"]
        assert ps["n_pruned"] == 0
        assert ps["pruned"] == []
        assert ps["dropped_window_ids"] == []

    def test_pruned_records_carry_expected_fields(self):
        """Each pruned record has window_id, frequency_mhz, and snr."""
        w1 = _make_window(42, [_make_peak(1234.5, snr=1.5, window_id=42)])
        fit = _make_fit([w1])
        apply_snr_survival_prune(fit, self.FLOOR, refit_window=_stub_refit)
        assert fit.window_fits == []  # window dropped
        ps = fit.diagnostics["peak_survival"]
        assert len(ps["pruned"]) == 1
        rec = ps["pruned"][0]
        assert rec["window_id"] == 42
        assert rec["frequency_mhz"] == pytest.approx(1234.5)
        assert rec["snr"] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Phase B: degenerate-overfit VIF collapse
# ---------------------------------------------------------------------------


def _cpeak(
    freq: float,
    *,
    amplitude: float = 1.0,
    amplitude_error: float | None = None,
    snr: float | None = None,
    origin: str = "auto",
    window_id: int = 1,
) -> FittedPeak:
    return FittedPeak(
        peak_id=0,
        frequency_mhz=freq,
        amplitude=amplitude,
        amplitude_error=amplitude_error,
        snr=snr,
        window_id=window_id,
        origin=origin,
    )


def _window_with_range(
    window_id: int,
    peaks: list[FittedPeak],
    freq_range: tuple[float, float],
    doublet_alternatives: list | None = None,
) -> FittingResult:
    from types import SimpleNamespace

    wf = FittingResult(window_id=window_id)
    wf.fitted_peaks = list(peaks)
    # The collapse only reads ``wf.window.freq_range`` for the window centre.
    wf.window = SimpleNamespace(freq_range=freq_range)  # type: ignore[assignment]
    if doublet_alternatives is not None:
        wf.doublet_alternatives = doublet_alternatives
    return wf


def _stub_collapse(
    wf: FittingResult,
    remove_freqs: list[float],
    add_freqs: list[float],
    add_seeds: list,
) -> FittingResult:
    """Stand-in for the production collapse refit: drop the paired peaks and
    add one merged peak per ``add_freqs`` entry (origin auto)."""
    kept = [p for p in wf.fitted_peaks if float(p.frequency_mhz) not in remove_freqs]
    merged = [_cpeak(f, snr=50.0, amplitude=2.0) for f in add_freqs]
    new = FittingResult(window_id=wf.window_id)
    new.fitted_peaks = kept + merged
    return new


class TestAmplitudeVif:
    def test_basic_value(self):
        from ftmwpipeline._internal.stage5_impl import amplitude_vif

        p = _cpeak(1000.0, amplitude=2.0, amplitude_error=1.0, snr=10.0)
        assert amplitude_vif(p) == pytest.approx(5.0)  # (1/2)*10

    def test_none_when_error_missing(self):
        from ftmwpipeline._internal.stage5_impl import amplitude_vif

        assert amplitude_vif(_cpeak(1.0, amplitude=1.0, snr=10.0)) is None

    def test_none_when_snr_missing(self):
        from ftmwpipeline._internal.stage5_impl import amplitude_vif

        assert amplitude_vif(_cpeak(1.0, amplitude=1.0, amplitude_error=1.0)) is None

    def test_none_on_zero_amplitude(self):
        from ftmwpipeline._internal.stage5_impl import amplitude_vif

        p = _cpeak(1.0, amplitude=0.0, amplitude_error=1.0, snr=10.0)
        assert amplitude_vif(p) is None


class TestApplyVifCollapse:
    # Resolution element ~ 1/13 us => 0.0769 MHz; max_sep at 0.5 res = 0.0385.
    RES = 1.0 / 13.0
    THRESH = 100.0
    MAX_SEP_RES = 0.5

    def _run(self, fit, **kw):
        from ftmwpipeline._internal.stage5_impl import apply_vif_collapse
        from ftmwpipeline.core.data_structures import Sideband

        apply_vif_collapse(
            fit,
            vif_threshold=kw.get("vif_threshold", self.THRESH),
            max_sep_res=kw.get("max_sep_res", self.MAX_SEP_RES),
            res_element_mhz=self.RES,
            sideband=Sideband.UPPER,
            refit_collapse=kw.get("refit_collapse", _stub_collapse),
        )

    def test_degenerate_pair_collapses(self):
        # Two near-degenerate high-VIF lines 0.01 MHz apart (< 0.0385).
        pa = _cpeak(1000.00, amplitude=1.0, amplitude_error=10.0, snr=500.0)
        pb = _cpeak(1000.01, amplitude=1.0, amplitude_error=10.0, snr=500.0)
        wf = _window_with_range(1, [pa, pb], (999.9, 1000.1))
        fit = _make_fit([wf])
        self._run(fit)
        out = fit.window_fits[0]
        assert len(out.fitted_peaks) == 1  # collapsed to one line
        diag = fit.diagnostics["vif_collapse"]
        assert diag["n_collapsed_pairs"] == 1
        rec = diag["collapses"][0]
        assert rec["window_id"] == 1
        assert {rec["frequency_a_mhz"], rec["frequency_b_mhz"]} == {1000.00, 1000.01}

    def test_low_vif_pair_not_collapsed(self):
        # Same separation, but identifiable (VIF ~ 1).
        pa = _cpeak(1000.00, amplitude=1.0, amplitude_error=0.02, snr=50.0)
        pb = _cpeak(1000.01, amplitude=1.0, amplitude_error=0.02, snr=50.0)
        wf = _window_with_range(1, [pa, pb], (999.9, 1000.1))
        fit = _make_fit([wf])
        self._run(fit)
        assert len(fit.window_fits[0].fitted_peaks) == 2
        assert fit.diagnostics["vif_collapse"]["n_collapsed_pairs"] == 0

    def test_far_high_vif_pair_not_collapsed(self):
        # High VIF but 0.1 MHz apart (> 0.0385) -> a real doublet, kept.
        pa = _cpeak(1000.0, amplitude=1.0, amplitude_error=10.0, snr=500.0)
        pb = _cpeak(1000.1, amplitude=1.0, amplitude_error=10.0, snr=500.0)
        wf = _window_with_range(1, [pa, pb], (999.8, 1000.3))
        fit = _make_fit([wf])
        self._run(fit)
        assert len(fit.window_fits[0].fitted_peaks) == 2
        assert fit.diagnostics["vif_collapse"]["n_collapsed_pairs"] == 0

    def test_window_without_range_skipped(self):
        pa = _cpeak(1000.00, amplitude=1.0, amplitude_error=10.0, snr=500.0)
        pb = _cpeak(1000.01, amplitude=1.0, amplitude_error=10.0, snr=500.0)
        wf = _make_window(1, [pa, pb])  # no window/freq_range
        fit = _make_fit([wf])
        self._run(fit)
        assert len(fit.window_fits[0].fitted_peaks) == 2  # untouched

    def test_merged_seed_snaps_to_doublet_alternative(self):
        from ftmwpipeline.core.data_structures import DoubletAlternativeInfo

        da = DoubletAlternativeInfo(
            frequency_a_mhz=1000.00,
            frequency_b_mhz=1000.01,
            amplitude_a=1.0,
            amplitude_b=1.0,
            separation_res_elements=0.13,
            amp_ratio=1.0,
            chi2r_production=3.0,
            chi2r_merged=9.0,
            delta_chi2_raw=0.0,
            delta_aicc=0.0,
            merged_frequency_mhz=1000.007,  # distinct from the centroid 1000.005
            merged_amplitude=2.5,
            merged_phase=0.3,
            merged_tau_us=13.0,
            merged_success=True,
            orth_evidence_delta_chi2=0.0,
            orth_evidence_n_params=0,
            support_bins=8,
        )
        pa = _cpeak(1000.00, amplitude=1.0, amplitude_error=10.0, snr=500.0)
        pb = _cpeak(1000.01, amplitude=1.0, amplitude_error=10.0, snr=500.0)
        wf = _window_with_range(1, [pa, pb], (999.9, 1000.1), doublet_alternatives=[da])
        fit = _make_fit([wf])

        captured: dict = {}

        def _capturing(wf_, remove_freqs, add_freqs, add_seeds):
            captured["add_freqs"] = list(add_freqs)
            captured["seed_amp"] = add_seeds[0].amplitude
            captured["seed_phase"] = add_seeds[0].phase
            return _stub_collapse(wf_, remove_freqs, add_freqs, add_seeds)

        self._run(fit, refit_collapse=_capturing)
        assert captured["add_freqs"] == [pytest.approx(1000.007)]
        assert captured["seed_amp"] == pytest.approx(2.5)
        assert captured["seed_phase"] == pytest.approx(0.3)


class TestSettingsWiring:
    """Verify the settings dataclass and hard defaults."""

    def test_hard_defaults_enabled_true(self):
        from ftmwpipeline.core.stage_fit_settings import resolve

        resolved = resolve()
        assert resolved.peak_survival.enabled is True

    def test_hard_defaults_vif_collapse(self):
        from ftmwpipeline.core.stage_fit_settings import resolve

        ps = resolve().peak_survival
        assert ps.vif_collapse_threshold == pytest.approx(100.0)
        assert ps.collapse_max_separation_res == pytest.approx(0.5)
        assert ps.vif_attention_threshold == pytest.approx(4.0)

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
        assert (
            loaded.peak_survival.enabled == False
        )  # noqa: E712 (np.False_ != is False)
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
