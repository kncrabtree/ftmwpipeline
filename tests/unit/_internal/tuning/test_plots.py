"""Unit tests for the registered plot adapters, using duck-typed fake stage
results so no pipeline build is needed."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest

from ftmwpipeline._internal.tuning import get_knob
from ftmwpipeline._internal.tuning.engine import PlotContext, SweepRow
from ftmwpipeline._internal.tuning.registry import FtAtStart
from ftmwpipeline._internal.tuning.plots import (
    plot_ft_band_stack,
    plot_noise_sweep,
    plot_peak_detection,
    plot_shape_vote,
    plot_spectra_ladder,
    plot_start_detection,
    plot_tau_trend,
)


@pytest.fixture(autouse=True)
def _agg_backend():
    import matplotlib
    matplotlib.use("Agg")


@dataclass
class _FakeStart:
    start_us: float
    chirp_end_us: float
    starts_us: Any
    sum_magnitude: Any


@dataclass
class _FakeNoise:
    rms_noise: Any
    noise_mask: Any


@dataclass
class _FakeFT:
    freq_array: Any
    complex_spectrum: Any


@dataclass
class _FakeBand:
    label: str
    freq_lo_mhz: float
    freq_hi_mhz: float
    tau_maj_us: float


@dataclass
class _FakeTau:
    tau_maj_us: float
    sigma_tau_us: float
    n_contributors: int
    contributor_taus_us: Any = None
    contributor_freqs_mhz: Any = None
    contributor_snrs: Any = None
    band_majorities: tuple = ()
    tau_max_us: float = 65.0
    start_us: float = 0.0
    end_us: float = 15.0


@dataclass
class _FakeShape:
    recommended_shape: Optional[str]
    vote_rates: dict
    n_contributors: int


def _ctx() -> PlotContext:
    # Non-existent path: the FID panel falls back to hidden when it can't load.
    return PlotContext(ftmw_path=Path("/nonexistent/x.ftmw"))


def _close(fig):
    import matplotlib.pyplot as plt
    plt.close(fig)


def _ft_row(value, start_us, chirp_end_us=None, p50=0.5):
    freqs = np.linspace(26500.0, 40000.0, 300)
    ft = _FakeFT(freqs, np.abs(np.sin(freqs)) + 0.01)
    return SweepRow(value, {"p50": p50},
                    FtAtStart(ft=ft, start_us=start_us, chirp_end_us=chirp_end_us))


def test_plot_start_detection_returns_figure_without_fid():
    # detection knobs (sweep_max_us / min_chirp_drop_ratio) use this adapter
    starts = np.linspace(0.0, 7.5, 50)
    mag = np.exp(-starts)
    rows = [
        SweepRow(6.0, {"start_us": 2.18}, _FakeStart(2.18, 1.68, starts, mag)),
        SweepRow(9.0, {"start_us": 2.68}, _FakeStart(2.68, 1.68, starts, mag)),
    ]
    fig = plot_start_detection(get_knob("stage0.sweep_max_us"), rows, _ctx())
    assert fig is not None
    assert len(fig.axes) >= 1
    _close(fig)


def test_spectra_ladder_has_fid_panel_plus_one_per_value():
    spec = get_knob("stage1.start_us")
    rows = [_ft_row(1.5 + 0.1 * i, 1.5 + 0.1 * i) for i in range(3)]
    fig = plot_spectra_ladder(spec, rows, _ctx())
    # one FID panel + one spectrum panel per value
    assert len(fig.axes) == 1 + 3
    _close(fig)


def test_spectra_ladder_height_grows_with_values():
    spec = get_knob("stage1.start_us")
    f3 = plot_spectra_ladder(spec, [_ft_row(1.5 + 0.1 * i, 1.5 + 0.1 * i) for i in range(3)], _ctx())
    f6 = plot_spectra_ladder(spec, [_ft_row(1.5 + 0.1 * i, 1.5 + 0.1 * i) for i in range(6)], _ctx())
    assert f6.get_figheight() > f3.get_figheight()
    _close(f3)
    _close(f6)


def test_spectra_ladder_used_by_guard_with_chirp_end():
    # guard rows carry a chirp_end; the ladder must still render
    spec = get_knob("stage0.guard_margin_us")
    rows = [_ft_row(0.5, 2.18, chirp_end_us=1.68), _ft_row(1.0, 2.68, chirp_end_us=1.68)]
    fig = plot_spectra_ladder(spec, rows, _ctx())
    assert fig is not None
    _close(fig)


def test_plot_ft_band_stack_panel_per_value():
    # trim / end_us knobs use this adapter: one spectrum panel per value, no FID
    spec = get_knob("stage1.trim_min_mhz")
    rows = [_ft_row(26000.0 + 1000.0 * i, start_us=2.0) for i in range(3)]
    fig = plot_ft_band_stack(spec, rows, _ctx())
    assert len(fig.axes) == 3
    _close(fig)


def test_plot_noise_sweep_returns_figure():
    sigma = np.abs(np.sin(np.linspace(0, 6, 200))) + 0.01
    mask = np.ones_like(sigma, dtype=bool)
    rows = [
        SweepRow(40.0, {"median_sigma": 0.5, "noise_fraction": 0.98},
                 _FakeNoise(sigma, mask)),
        SweepRow(80.0, {"median_sigma": 0.52, "noise_fraction": 0.98},
                 _FakeNoise(sigma * 1.01, mask)),
    ]
    fig = plot_noise_sweep(get_knob("stage2.window_mhz"), rows, _ctx())
    assert fig is not None
    assert len(fig.axes) >= 2
    _close(fig)


def _fake_tau(tau, sigma, n, rng):
    bands = (
        _FakeBand("low", 26500.0, 31000.0, tau + 0.4),
        _FakeBand("mid", 31000.0, 35500.0, tau),
        _FakeBand("high", 35500.0, 40000.0, tau - 0.4),
    )
    return _FakeTau(
        tau, sigma, n,
        contributor_taus_us=np.abs(rng.normal(tau, sigma, n)),
        contributor_freqs_mhz=rng.uniform(26500.0, 40000.0, n),
        contributor_snrs=rng.uniform(5.0, 500.0, n),
        band_majorities=bands,
    )


def test_plot_tau_trend_numeric_returns_figure():
    rng = np.random.default_rng(0)
    rows = [
        SweepRow(8, {"tau_maj_us": 5.6, "sigma_tau_us": 1.4, "n_contributors": 3990},
                 _fake_tau(5.6, 1.4, 3990, rng)),
        SweepRow(12, {"tau_maj_us": 5.9, "sigma_tau_us": 1.4, "n_contributors": 5022},
                 _fake_tau(5.9, 1.4, 5022, rng)),
    ]
    fig = plot_tau_trend(get_knob("stage2b.stft.n_seg"), rows, _ctx())
    assert fig is not None
    # trend (+twin) + decay panel + tau-vs-freq panel (+colorbar) per value
    assert len(fig.axes) >= 6
    _close(fig)


def test_plot_shape_vote_returns_figure():
    rows = [
        SweepRow(0.05, {"recommended_shape": "gaussian", "exp": 0.3,
                        "gauss": 0.6, "voigt": 0.1, "n_contributors": 210},
                 _FakeShape("gaussian", {"exp": 0.3, "gauss": 0.6, "voigt": 0.1}, 210)),
        SweepRow(0.20, {"recommended_shape": "none", "exp": 0.45,
                        "gauss": 0.45, "voigt": 0.1, "n_contributors": 210},
                 _FakeShape(None, {"exp": 0.45, "gauss": 0.45, "voigt": 0.1}, 210)),
    ]
    fig = plot_shape_vote(
        get_knob("stage2b.recommendation.pure_margin_threshold"), rows, _ctx())
    assert fig is not None
    _close(fig)


def test_plot_tau_trend_none_for_nonnumeric():
    rows = [SweepRow("foo", {"tau_maj_us": 5.6}, _FakeTau(5.6, 1.4, 10))]
    assert plot_tau_trend(get_knob("stage2b.stft.n_seg"), rows, _ctx()) is None


def test_adapters_return_none_without_results():
    rows = [SweepRow(40.0, {"median_sigma": 0.5}, None)]
    ctx = _ctx()
    assert plot_noise_sweep(get_knob("stage2.window_mhz"), rows, ctx) is None
    assert plot_start_detection(get_knob("stage0.sweep_max_us"), rows, ctx) is None
    assert plot_spectra_ladder(get_knob("stage1.start_us"), rows, ctx) is None


@dataclass
class _FakeCls:
    value: str


@dataclass
class _FakePeak:
    frequency: float
    intensity: float
    snr: float
    properties: dict
    classification: Any = None


def _peak(f, inten, snr, promoted, dp="primary", band="weak"):
    return _FakePeak(f, inten, snr, {"promoted": promoted, "detection_pass": dp},
                     _FakeCls(band) if promoted else None)


def _peaks_result(peaks, min_snr=3.0):
    # Active FT spanning the 2638 band; a few bumps so the regions have lines.
    freqs = np.linspace(26500.0, 40000.0, 1400)
    mag = np.full(freqs.shape, 0.02)
    for p in peaks:
        mag[np.argmin(np.abs(freqs - p.frequency))] = p.intensity
    ft = _FakeFT(freqs, mag.astype(complex))
    rms = np.full(freqs.shape, 0.01)
    n_prom = sum(1 for p in peaks if p.properties["promoted"])
    n_prim = sum(1 for p in peaks if p.properties["detection_pass"] == "primary")
    return {
        "active_ft": ft, "active_rms": rms, "peaks": peaks,
        "promotion_min_snr": min_snr, "n_peaks": len(peaks),
        "n_promoted": n_prom, "n_primary": n_prim, "n_gap": len(peaks) - n_prim,
    }


def _peak_metrics(res):
    promoted = [p for p in res["peaks"] if p.properties["promoted"]]

    def _n(band):
        return sum(1 for p in promoted
                   if getattr(p.classification, "value", None) == band)

    return {
        "n_total": len(promoted), "n_strong": _n("strong"),
        "n_medium": _n("medium"), "n_weak": _n("weak"),
    }


def test_plot_peak_detection_returns_figure():
    # Two values whose promoted sets differ -> a region is auto-selected to zoom.
    r1 = _peaks_result([
        _peak(27000.0, 0.5, 30, True), _peak(27050.0, 0.3, 20, True, "gap"),
        _peak(35000.0, 0.4, 25, True), _peak(35080.0, 0.05, 2, False, "gap"),
    ], min_snr=2.0)
    r2 = _peaks_result([
        _peak(27000.0, 0.5, 30, True),
        _peak(35000.0, 0.4, 25, True), _peak(35080.0, 0.05, 2, False, "gap"),
    ], min_snr=5.0)
    rows = [SweepRow(2.0, _peak_metrics(r1), r1), SweepRow(5.0, _peak_metrics(r2), r2)]
    fig = plot_peak_detection(get_knob("stage3.promotion.min_snr"), rows, _ctx())
    assert fig is not None
    # trend (+twin) + at least one region column per value
    assert len(fig.axes) >= 3
    _close(fig)


def test_plot_peak_detection_handles_nonnumeric_value():
    # toggle knobs (run_gap_pass) sweep over bools; the categorical trend x must
    # not raise and the panels must still render.
    r1 = _peaks_result([_peak(27000.0, 0.5, 30, True)], min_snr=3.0)
    r2 = _peaks_result([_peak(27000.0, 0.5, 30, True),
                        _peak(27040.0, 0.2, 12, True, "gap")], min_snr=3.0)
    rows = [SweepRow(False, _peak_metrics(r1), r1),
            SweepRow(True, _peak_metrics(r2), r2)]
    fig = plot_peak_detection(get_knob("stage3.gap_pass.run_gap_pass"), rows, _ctx())
    assert fig is not None
    _close(fig)


def test_plot_peak_detection_none_without_results():
    rows = [SweepRow(3.0, {"n_peaks": 0}, None)]
    assert plot_peak_detection(get_knob("stage3.promotion.min_snr"), rows, _ctx()) is None


def test_peak_persistence_buckets_by_last_surviving_value():
    from ftmwpipeline._internal.tuning.plots import _peak_persistence

    # peak A promoted in both values -> last index 1; peak B only in value 0.
    r0 = _peaks_result([_peak(27000.0, 0.5, 30, True),
                        _peak(35000.0, 0.4, 8, True)], min_snr=2.0)
    r1 = _peaks_result([_peak(27000.05, 0.5, 30, True)], min_snr=5.0)
    rows = [SweepRow(2.0, _peak_metrics(r0), r0),
            SweepRow(5.0, _peak_metrics(r1), r1)]
    pers = {round(f): idx for (f, _inten, idx) in _peak_persistence(rows, 0.05)}
    assert pers[27000] == 1  # survives to the last value
    assert pers[35000] == 0  # drops out after the first value


def test_knob_plot_wiring():
    # the spectrum-impact knobs share the ladder; detection knobs show the curve
    assert get_knob("stage1.start_us").plot is plot_spectra_ladder
    assert get_knob("stage0.guard_margin_us").plot is plot_spectra_ladder
    assert get_knob("stage0.sweep_max_us").plot is plot_start_detection
    assert get_knob("stage0.min_chirp_drop_ratio").plot is plot_start_detection
    assert get_knob("stage2.window_mhz").plot is plot_noise_sweep
    # every Stage 3 knob renders the peak-detection spectrum-overlay view
    assert get_knob("stage3.promotion.min_snr").plot is plot_peak_detection
    assert get_knob("stage3.gap_pass.gap_leakage_floor_k").plot is plot_peak_detection


def test_detection_knobs_point_at_spectrum_knobs():
    assert "start_us" in (get_knob("stage0.sweep_max_us").see_also or "")
    # the guard knob is itself the spectrum view, so it carries no see_also
    assert get_knob("stage0.guard_margin_us").see_also is None
