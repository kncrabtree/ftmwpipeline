"""The Stage 5 detail figure's model overlay is the fit's model.

``_plot_per_window_detail`` draws through :mod:`ftmwpipeline.fitting.model_eval`,
so on a window fitted with a Gaussian line shape and a baseline the residual it
draws has the persisted chi-squared. (It used to draw a Lorentzian model with no
baseline, so the chi-squared of its residual disagreed with the fit's.)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import ftmwpipeline.api as ftmw  # noqa: E402
from ftmwpipeline._internal import model_impl  # noqa: E402
from ftmwpipeline.visualization import fit_visualization as fv  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_detail_overlay_chi2_equals_the_persisted_fit(
    stage5_gaussian_2638, monkeypatch
):
    path = str(stage5_gaussian_2638)
    ctx = model_impl._load_model_context(path, "window_model", "active")
    fit = ftmw.load_fit(path)

    drawn = []
    real = fv._to_persisted_frame

    def spy(model, scale, ramp):
        out = real(model, scale, ramp)
        drawn.append(np.asarray(out))
        return out

    monkeypatch.setattr(fv, "_to_persisted_frame", spy)

    candidates = []
    for wf in fit.window_fits:
        qa = wf.quality_metrics or {}
        if float(qa.get("baseline_applied", 0.0)) < 0.5:
            continue
        if wf.shape != "gaussian":
            continue
        p = model_impl._window_payload(ctx, int(wf.window_id), components=False)
        if p["excluded"].any():
            continue  # the plot does not mask spurs; compare unmasked windows
        candidates.append((wf, p))
    assert candidates, "the Gaussian fit has no unmasked baseline window"

    for wf, p in candidates[:3]:
        drawn.clear()
        fig = fv._plot_per_window_detail(
            ctx.grid.freq_mhz,
            ctx.grid.data,
            ctx.grid.sigma,
            fit,
            int(wf.window_id),
            ctx.sideband,
            ctx.acquisition_us,
            (8.0, 10.0),
            "overlay",
        )
        plt.close(fig)
        model = drawn[0]
        lo, hi = wf.window.freq_range
        sel = (ctx.grid.freq_mhz >= min(lo, hi)) & (ctx.grid.freq_mhz <= max(lo, hi))
        assert model.size == int(sel.sum()) == p["model"].size
        np.testing.assert_array_equal(model, p["model"])
        r = ctx.grid.data[sel] - model
        chi2 = float(np.sum(np.abs(r) ** 2 / (ctx.grid.sigma[sel] ** 2 / 2)))
        assert chi2 == pytest.approx(2.0 * wf.cost, rel=1e-9, abs=1e-9)
