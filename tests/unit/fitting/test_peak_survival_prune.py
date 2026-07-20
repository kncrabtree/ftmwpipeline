"""
Unit tests for the peak-survival cleanup decisions.

The SNR-floor prune and the degenerate-VIF collapse now run in the fit walk's
per-node tail (``build_finalize_node`` / ``_prune_outcome`` / ``_collapse_outcome``
in ``stage5_impl``), driven by ``refit_outcome`` on a ``WindowOutcome``; the old
global post-pass (``apply_snr_survival_prune`` / ``apply_vif_collapse``) is gone.
These tests pin the *decisions* those passes reason about -- the per-line dust
predicate (:func:`_is_survival_dust_view`) and the collapse-eligibility rank
(:func:`_collapse_rank`) -- which are pure functions over a
:class:`FittedLineView` shared by the in-walk cleanup and the post-fit user-edit
path. The orchestration around them (fixpoint iteration, window drop, diagnostics
assembly, the separation / footprint pair-selection guard) is covered by the
Stage-5 integration tests and the cross-fixture rebuilds, not re-stubbed here.
"""

from __future__ import annotations

import pytest

from ftmwpipeline._internal.stage5_impl import (
    _collapse_rank,
    _is_brightness_sidelobe,
    _is_survival_dust_view,
)
from ftmwpipeline.core.data_structures import (
    FittedPeak,
    FittingResult,
    SpectrumFit,
)
from ftmwpipeline.fitting.result_conversion import FittedLineView


def _view(
    *,
    snr: float | None = None,
    origin: str = "auto",
    amplitude: float = 1.0,
    amplitude_error: float | None = None,
    freq: float = 1000.0,
) -> FittedLineView:
    """A minimal :class:`FittedLineView` for the pure decision functions."""
    return FittedLineView(
        frequency_mhz=freq,
        offset_mhz=0.0,
        amplitude=amplitude,
        amplitude_error=amplitude_error,
        phase=0.0,
        snr=snr,
        origin=origin,
        index=0,
    )


def _make_fit(windows: list[FittingResult]) -> SpectrumFit:
    all_peaks = sorted(
        [p for wf in windows for p in wf.fitted_peaks],
        key=lambda p: p.frequency_mhz,
    )
    fit = SpectrumFit(window_fits=list(windows), fitted_peaks=all_peaks)
    return fit


class TestSurvivalDustView:
    """The per-line SNR-floor dust predicate (:func:`_is_survival_dust_view`)."""

    FLOOR = 3.2

    def test_sub_floor_auto_is_dust(self):
        assert _is_survival_dust_view(_view(snr=2.0), self.FLOOR) is True

    def test_at_floor_not_dust(self):
        # The floor is inclusive-keep: snr == floor survives.
        assert _is_survival_dust_view(_view(snr=3.2), self.FLOOR) is False

    def test_above_floor_not_dust(self):
        assert _is_survival_dust_view(_view(snr=10.0), self.FLOOR) is False

    def test_user_origin_immune(self):
        # A user-origin line below the floor is never dust.
        assert (
            _is_survival_dust_view(_view(snr=1.0, origin="user"), self.FLOOR) is False
        )

    def test_none_snr_not_dust(self):
        assert _is_survival_dust_view(_view(snr=None), self.FLOOR) is False

    def test_nan_snr_not_dust(self):
        assert _is_survival_dust_view(_view(snr=float("nan")), self.FLOOR) is False


# ---------------------------------------------------------------------------
# FittedPeak / window builders for the amplitude-VIF and window-cleanup tests.
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
    # The collapse only reads ``wf.window.freq_range`` for the window center.
    wf.window = SimpleNamespace(freq_range=freq_range)  # type: ignore[assignment]
    if doublet_alternatives is not None:
        wf.doublet_alternatives = doublet_alternatives
    return wf


class TestAmplitudeVif:
    def test_basic_value(self):
        from ftmwpipeline.fitting.validation import amplitude_vif

        p = _cpeak(1000.0, amplitude=2.0, amplitude_error=1.0, snr=10.0)
        assert amplitude_vif(p) == pytest.approx(5.0)  # (1/2)*10

    def test_none_when_error_missing(self):
        from ftmwpipeline.fitting.validation import amplitude_vif

        assert amplitude_vif(_cpeak(1.0, amplitude=1.0, snr=10.0)) is None

    def test_none_when_snr_missing(self):
        from ftmwpipeline.fitting.validation import amplitude_vif

        assert amplitude_vif(_cpeak(1.0, amplitude=1.0, amplitude_error=1.0)) is None

    def test_none_on_zero_amplitude(self):
        from ftmwpipeline.fitting.validation import amplitude_vif

        p = _cpeak(1.0, amplitude=0.0, amplitude_error=1.0, snr=10.0)
        assert amplitude_vif(p) is None


class TestCollapseRank:
    """Collapse-eligibility rank for one line (:func:`_collapse_rank`).

    The pure decision the VIF-collapse selects pairs from: a high amplitude VIF,
    or the singular-covariance degeneracy the VIF gate is blind to. The
    separation / footprint pair-selection guard around it lives in
    ``_collapse_outcome`` and is covered by the Stage-5 integration tests."""

    THRESH = 100.0

    def test_high_vif_eligible(self):
        # amp_err/amp * snr = 10/1 * 500 = 5000 > 100.
        v = _view(amplitude=1.0, amplitude_error=10.0, snr=500.0)
        assert _collapse_rank(v, self.THRESH) == pytest.approx(5000.0)

    def test_low_vif_not_eligible(self):
        # 0.02/1 * 50 = 1.0 < 100: identifiable, kept.
        v = _view(amplitude=1.0, amplitude_error=0.02, snr=50.0)
        assert _collapse_rank(v, self.THRESH) is None

    def test_singular_covariance_eligible_as_inf(self):
        # amplitude_error None -> VIF undefined; a finite, non-zero amplitude
        # with finite snr is the strongest degeneracy -> +inf (the gate misses it).
        v = _view(amplitude=1.0, amplitude_error=None, snr=500.0)
        assert _collapse_rank(v, self.THRESH) == float("inf")

    def test_dead_zero_amplitude_not_eligible(self):
        # Zero amplitude is a dead peak, not a degeneracy, even with singular cov.
        v = _view(amplitude=0.0, amplitude_error=None, snr=500.0)
        assert _collapse_rank(v, self.THRESH) is None

    def test_missing_snr_not_eligible(self):
        v = _view(amplitude=1.0, amplitude_error=None, snr=None)
        assert _collapse_rank(v, self.THRESH) is None

    def test_frac_unc_eligible_below_vif(self):
        # VIF = 0.2/1 * 50 = 10 < 100 (kept by the VIF gate alone), but the
        # fractional amplitude uncertainty 0.2 >= 0.15 -> eligible at its VIF.
        v = _view(amplitude=1.0, amplitude_error=0.2, snr=50.0)
        assert _collapse_rank(v, self.THRESH, 0.15) == pytest.approx(10.0)

    def test_frac_unc_below_bar_not_eligible(self):
        # frac 0.10 < 0.15 and VIF = 0.10*50 = 5 < 100: individually
        # constrained, kept (the resolvable-doublet protection).
        v = _view(amplitude=1.0, amplitude_error=0.10, snr=50.0)
        assert _collapse_rank(v, self.THRESH, 0.15) is None

    def test_frac_unc_default_is_noop(self):
        # Without the fractional bar (default +inf) a high-frac/low-VIF line is
        # not eligible -- preserves the VIF-only behavior for 2-arg callers.
        v = _view(amplitude=1.0, amplitude_error=0.5, snr=50.0)
        assert _collapse_rank(v, self.THRESH) is None


def _spur_set(centers_sources):
    """Build a minimal SpurSet (center_mhz, source) for cleanup tests."""
    from ftmwpipeline.fitting.spur_detection import GatedSpur, SpurSet

    spurs = tuple(
        GatedSpur(center_mhz=c, integer_mhz=int(round(c)), source=src)
        for c, src in centers_sources
    )
    return SpurSet(spurs=spurs, bin_spacing_mhz=1.0 / 13.0, mask_half_width_bins=2)


class TestApplyWindowCleanup:
    RES = 1.0 / 13.0

    def _run(self, fit, spur_set=None, drop_empty=True, drop_spur_only=True):
        from ftmwpipeline._internal.stage5_impl import apply_window_cleanup

        apply_window_cleanup(
            fit,
            spur_set=spur_set,
            res_element_mhz=self.RES,
            drop_empty=drop_empty,
            drop_spur_only=drop_spur_only,
        )

    def test_empty_window_dropped(self):
        w1 = _window_with_range(1, [_cpeak(1000.0, snr=10.0)], (999.0, 1001.0))
        w2 = _window_with_range(2, [], (2000.0, 2002.0))  # empty
        fit = _make_fit([w1, w2])
        self._run(fit)
        assert [w.window_id for w in fit.window_fits] == [1]
        assert fit.diagnostics["window_cleanup"]["n_empty_dropped"] == 1

    def test_empty_window_kept_when_disabled(self):
        w2 = _window_with_range(2, [], (2000.0, 2002.0))
        fit = _make_fit([w2])
        self._run(fit, drop_empty=False)
        assert len(fit.window_fits) == 1

    def test_spur_only_window_dropped(self):
        ss = _spur_set([(1000.02, "flat+saturated")])
        w1 = _window_with_range(1, [_cpeak(1000.0, snr=50.0)], (999.0, 1001.0))
        fit = _make_fit([w1])
        self._run(fit, spur_set=ss)
        assert fit.window_fits == []
        assert fit.diagnostics["window_cleanup"]["n_spur_only_dropped"] == 1

    def test_spur_only_ambiguous_source_kept(self):
        # 'narrow'/'drift' alone are not confidently instrumental -> keep.
        ss = _spur_set([(1000.02, "narrow")])
        w1 = _window_with_range(1, [_cpeak(1000.0, snr=50.0)], (999.0, 1001.0))
        fit = _make_fit([w1])
        self._run(fit, spur_set=ss)
        assert [w.window_id for w in fit.window_fits] == [1]

    def test_multi_line_window_on_spur_kept(self):
        # Two lines -> never dropped as spur-only (ambiguous).
        ss = _spur_set([(1000.02, "flat+saturated")])
        w1 = _window_with_range(
            1, [_cpeak(1000.0, snr=50.0), _cpeak(1000.5, snr=40.0)], (999.0, 1001.0)
        )
        fit = _make_fit([w1])
        self._run(fit, spur_set=ss)
        assert [w.window_id for w in fit.window_fits] == [1]

    def test_user_origin_spur_line_immune(self):
        ss = _spur_set([(1000.02, "flat+saturated")])
        w1 = _window_with_range(
            1, [_cpeak(1000.0, snr=50.0, origin="user")], (999.0, 1001.0)
        )
        fit = _make_fit([w1])
        self._run(fit, spur_set=ss)
        assert [w.window_id for w in fit.window_fits] == [1]

    def test_line_far_from_spur_kept(self):
        ss = _spur_set([(1000.5, "flat+saturated")])  # > 1 res from the line
        w1 = _window_with_range(1, [_cpeak(1000.0, snr=50.0)], (999.0, 1001.0))
        fit = _make_fit([w1])
        self._run(fit, spur_set=ss)
        assert [w.window_id for w in fit.window_fits] == [1]


class TestSettingsWiring:
    """Verify the settings dataclass and hard defaults."""

    def test_hard_defaults_enabled_true(self):
        from ftmwpipeline.core.stage_fit_settings import resolve

        resolved = resolve()
        assert resolved.peak_survival.enabled is True

    def test_hard_defaults_vif_collapse(self):
        from ftmwpipeline.core.stage_fit_settings import resolve

        ps = resolve().peak_survival
        assert ps.vif_collapse_threshold == pytest.approx(25.0)
        assert ps.collapse_frac_unc_threshold == pytest.approx(0.15)
        assert ps.collapse_frac_unc_max_separation_res == pytest.approx(0.5)
        assert ps.collapse_max_separation_res == pytest.approx(1.0)
        assert ps.sidelobe_prune_max_separation_res == pytest.approx(2.5)
        assert ps.degenerate_trial_frac == pytest.approx(0.5)
        assert ps.degenerate_trial_chi2r_rel_tol == pytest.approx(0.5)
        assert ps.drop_empty_windows is True
        assert ps.drop_spur_only_windows is True

    def test_hard_default_snr_floor(self):
        from ftmwpipeline.core.stage_fit_settings import resolve

        resolved = resolve()
        # The floor has no hard default: it tracks the Stage 3 promotion cutoff
        # via the factor (default 1.1). The absolute override stays unset.
        assert resolved.peak_survival.snr_survival_floor is None
        assert resolved.peak_survival.snr_survival_factor == pytest.approx(1.1)

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
        # The floor is an unset absolute override (None); the factor carries the
        # default that derives it from the Stage 3 promotion cutoff.
        assert loaded.peak_survival.snr_survival_floor is None
        assert loaded.peak_survival.snr_survival_factor == pytest.approx(1.1)


# ---------------------------------------------------------------------------
# Bright-neighbor sidelobe predicate (Type B), the pure decision function.
# The prune orchestration (faintest-first removal, refit-to-fixpoint) and the
# Type A degenerate merge-trial are covered by the Stage-5 integration rebuilds.
# ---------------------------------------------------------------------------


def test_brightness_sidelobe_predicate() -> None:
    res = 0.1  # MHz per resolution element
    cap = 2.5

    def sl(victim_snr, neighbor_snr, sep_res, **kw):
        return _is_brightness_sidelobe(
            _view(snr=victim_snr, freq=1000.0 + sep_res * res),
            _view(snr=neighbor_snr, freq=1000.0),
            max_sep_res=cap,
            res_element_mhz=res,
        )

    # A faint peak inside a much brighter neighbor's reach (0.2*100/10 = 2.0 res).
    assert sl(10.0, 100.0, 0.5) is True
    assert sl(10.0, 100.0, 1.9) is True
    # Past the brightness-scaled reach (2.0 res) -> not a sidelobe.
    assert sl(10.0, 100.0, 2.5) is False
    # Reach is capped: a very bright neighbor would reach far, but the cap holds.
    assert sl(5.0, 1000.0, 3.0) is False  # uncapped reach 40 res, capped to 2.5
    assert sl(5.0, 1000.0, 2.0) is True
    # Comparable brightness -> sub-kappa reach (0.2*100/80 = 0.25 res): a real
    # doublet (e.g. 360 w36 A/E at ~0.7 res) is NOT flagged.
    assert sl(80.0, 100.0, 0.6) is False
    # The neighbor must be brighter than the victim.
    assert sl(100.0, 10.0, 0.3) is False
    # Missing / non-finite SNR is never a sidelobe.
    assert sl(None, 100.0, 0.3) is False
    assert sl(10.0, None, 0.3) is False
    assert sl(float("nan"), 100.0, 0.3) is False
