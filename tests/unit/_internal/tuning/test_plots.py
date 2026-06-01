"""Unit tests for the registered plot adapters, using duck-typed fake stage
results so no pipeline build is needed."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ftmwpipeline._internal.tuning import get_knob
from ftmwpipeline._internal.tuning.engine import PlotContext, SweepRow
from ftmwpipeline._internal.tuning.plots import (
    plot_noise_sweep,
    plot_start_detection,
    plot_start_ladder,
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
class _FakeTau:
    tau_maj_us: float
    sigma_tau_us: float
    n_contributors: int


def _ctx() -> PlotContext:
    # A path that does not exist: the start-detection adapter falls back to the
    # detection panel when the FID cannot be loaded.
    return PlotContext(ftmw_path=Path("/nonexistent/x.ftmw"))


def _close(fig):
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_start_detection_returns_figure_without_fid():
    starts = np.linspace(0.0, 7.5, 50)
    mag = np.exp(-starts)
    rows = [
        SweepRow(0.5, {"start_us": 2.18}, _FakeStart(2.18, 1.68, starts, mag)),
        SweepRow(1.0, {"start_us": 2.68}, _FakeStart(2.68, 1.68, starts, mag)),
    ]
    fig = plot_start_detection(get_knob("start.guard_margin_us"), rows, _ctx())
    assert fig is not None
    assert len(fig.axes) >= 1
    _close(fig)


def test_plot_start_ladder_height_grows_with_values():
    freqs = np.linspace(26500.0, 40000.0, 300)
    spec = get_knob("stage1.start_us")

    def _rows(n):
        return [
            SweepRow(1.5 + 0.1 * i, {"p50": 1.0},
                     _FakeFT(freqs, np.abs(np.sin(freqs)) + 0.01))
            for i in range(n)
        ]

    fig3 = plot_start_ladder(spec, _rows(3), _ctx())
    fig6 = plot_start_ladder(spec, _rows(6), _ctx())
    assert len(fig3.axes) == 3
    assert len(fig6.axes) == 6
    # height grows with the number of values (fixed per-panel aspect)
    assert fig6.get_figheight() > fig3.get_figheight()
    _close(fig3)
    _close(fig6)


def test_plot_noise_sweep_returns_figure():
    sigma = np.abs(np.sin(np.linspace(0, 6, 200))) + 0.01
    mask = np.ones_like(sigma, dtype=bool)
    rows = [
        SweepRow(40.0, {"median_sigma": 0.5, "noise_fraction": 0.98},
                 _FakeNoise(sigma, mask)),
        SweepRow(80.0, {"median_sigma": 0.52, "noise_fraction": 0.98},
                 _FakeNoise(sigma * 1.01, mask)),
    ]
    fig = plot_noise_sweep(get_knob("stage2.scatter.window_mhz"), rows, _ctx())
    assert fig is not None
    assert len(fig.axes) >= 2  # σ(f) panel + metric-trend panel (+ twin)
    _close(fig)


def test_plot_tau_trend_numeric_returns_figure():
    rows = [
        SweepRow(8, {"tau_maj_us": 5.6, "sigma_tau_us": 1.4, "n_contributors": 3990},
                 _FakeTau(5.6, 1.4, 3990)),
        SweepRow(12, {"tau_maj_us": 5.9, "sigma_tau_us": 1.4, "n_contributors": 5022},
                 _FakeTau(5.9, 1.4, 5022)),
    ]
    fig = plot_tau_trend(get_knob("stage2b.stft.n_seg"), rows, _ctx())
    assert fig is not None
    _close(fig)


def test_plot_tau_trend_none_for_nonnumeric():
    rows = [SweepRow("foo", {"tau_maj_us": 5.6}, _FakeTau(5.6, 1.4, 10))]
    assert plot_tau_trend(get_knob("stage2b.stft.n_seg"), rows, _ctx()) is None


def test_2b_knob_plot_wiring():
    assert get_knob("stage2b.stft.n_seg").plot is plot_tau_trend
    assert get_knob("stage2b.stft.t_sigma").plot is plot_tau_trend
    assert get_knob("stage2b.polish.polish_snr_cap").plot is plot_tau_trend
    # the boolean knob is table-only
    assert get_knob("stage2b.polish.polish_noise_debias").plot is None


def test_adapters_return_none_without_results():
    rows = [SweepRow(40.0, {"median_sigma": 0.5}, None)]
    ctx = _ctx()
    assert plot_noise_sweep(get_knob("stage2.scatter.window_mhz"), rows, ctx) is None
    assert plot_start_detection(get_knob("start.guard_margin_us"), rows, ctx) is None
    assert plot_start_ladder(get_knob("stage1.start_us"), rows, ctx) is None


def test_registered_knobs_have_plot_adapters():
    assert get_knob("start.guard_margin_us").plot is plot_start_detection
    assert get_knob("stage1.start_us").plot is plot_start_ladder
    assert get_knob("stage2.scatter.window_mhz").plot is plot_noise_sweep


def test_start_knobs_have_see_also_pointer():
    assert "stage1.start_us" in (get_knob("start.guard_margin_us").see_also or "")
