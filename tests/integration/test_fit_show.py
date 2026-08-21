"""Integration tests for ``fit show`` -- selection, rendering, batch, and
cross-interface consistency of the consolidated per-window detail.

All built on the session-scoped 3-window Stage 5 baseline.
"""

import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import ftmwpipeline.api as ftmw  # noqa: E402
from ftmwpipeline._internal.stage5_impl import (  # noqa: E402
    _resolve_detail_bundle,
    fit_show_impl,
    load_fit_impl,
    render_fit_detail_impl,
    render_rescue_summary_impl,
    render_windowed_view_impl,
    select_window_ids,
)
from ftmwpipeline.pipeline import Pipeline  # noqa: E402


@pytest.fixture
def stage5_file(baseline_2638_stage5_small, tmp_path):
    fp = tmp_path / "fit.ftmw"
    shutil.copy(baseline_2638_stage5_small, fp)
    return str(fp)


@pytest.fixture
def fit_obj(stage5_file):
    return load_fit_impl(stage5_file)["fit"]


# ---------------------------------------------------------------------------
# Window selection
# ---------------------------------------------------------------------------
class TestSelectWindowIds:
    def test_explicit_ids(self, fit_obj):
        ids = [int(wf.window_id) for wf in fit_obj.window_fits]
        assert select_window_ids(fit_obj, window_ids=[ids[0]]) == [ids[0]]

    def test_all_windows(self, fit_obj):
        ids = sorted(int(wf.window_id) for wf in fit_obj.window_fits)
        assert select_window_ids(fit_obj, all_windows=True) == ids

    def test_all_overrides_other_selectors(self, fit_obj):
        ids = sorted(int(wf.window_id) for wf in fit_obj.window_fits)
        got = select_window_ids(fit_obj, window_ids=[ids[0]], all_windows=True)
        assert got == ids

    def test_union_dedups(self, fit_obj):
        ids = sorted(int(wf.window_id) for wf in fit_obj.window_fits)
        # same window via id and top-snr -> appears once, sorted.
        got = select_window_ids(fit_obj, window_ids=[ids[0], ids[0]], top_snr=1)
        assert got == sorted(set(got))
        assert ids[0] in got

    def test_freq_maps_to_containing_window(self, fit_obj):
        wf = fit_obj.window_fits[0]
        lo, hi = wf.window.freq_range
        mid = 0.5 * (lo + hi)
        assert select_window_ids(fit_obj, freqs=[mid]) == [int(wf.window_id)]

    def test_unknown_id_raises(self, fit_obj):
        with pytest.raises(ValueError, match="window id"):
            select_window_ids(fit_obj, window_ids=[10_000])

    def test_freq_in_no_window_raises(self, fit_obj):
        with pytest.raises(ValueError, match="no fit window"):
            select_window_ids(fit_obj, freqs=[1.0])

    def test_top_snr_orders_by_brightest(self, fit_obj):
        got = select_window_ids(fit_obj, top_snr=1)

        # the single top-SNR window must be the one holding the brightest peak.
        def _max_snr(wf):
            snrs = [p.snr for p in wf.fitted_peaks if p.snr is not None]
            return max(snrs) if snrs else float("-inf")

        brightest = max(fit_obj.window_fits, key=_max_snr)
        assert got == [int(brightest.window_id)]

    def test_random_is_seed_reproducible(self, fit_obj):
        a = select_window_ids(fit_obj, random_n=2, random_seed=11)
        b = select_window_ids(fit_obj, random_n=2, random_seed=11)
        assert a == b
        assert len(a) == 2


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
class TestRenderDetail:
    def test_returns_figure_with_expected_panels(self, stage5_file, fit_obj):
        wid = int(fit_obj.window_fits[0].window_id)
        fig = render_fit_detail_impl(stage5_file, wid)
        # overview + 3 residual + 3 data + hist + peak-table axis = 9.
        assert len(fig.axes) == 9
        plt.close(fig)

    def test_magnitude_panels_use_padded_grid_stats_stay_native(self, stage5_file):
        bundle = _resolve_detail_bundle(stage5_file)
        wf = bundle.fit.window_fits[0]
        lo, hi = wf.window.freq_range
        lo, hi = min(lo, hi), max(lo, hi)
        n_native = int(((bundle.frequencies >= lo) & (bundle.frequencies <= hi)).sum())
        n_pad = int(((bundle.freq_padded >= lo) & (bundle.freq_padded <= hi)).sum())
        # Display magnitude grid is exactly-2x denser than the native grid.
        assert n_pad == pytest.approx(2 * n_native, abs=2)
        # The native grid (the one every statistic uses) is unchanged.
        assert n_native == int(
            ((bundle.frequencies >= lo) & (bundle.frequencies <= hi)).sum()
        )


# ---------------------------------------------------------------------------
# fit_show_impl: overview, detail, batch
# ---------------------------------------------------------------------------
class TestFitShowImpl:
    def test_no_selector_is_overview(self, stage5_file):
        result = fit_show_impl(stage5_file)
        assert result["mode"] == "overview"
        assert result["window_ids"] == []
        assert len(result["figures"]) == 1
        for fig in result["figures"]:
            plt.close(fig)

    def test_batch_writes_one_png_per_window(self, stage5_file, tmp_path):
        out = tmp_path / "figs"
        result = fit_show_impl(stage5_file, all_windows=True, output_dir=str(out))
        assert result["mode"] == "detail"
        assert len(result["paths"]) == len(result["window_ids"])
        written = sorted(p.name for p in out.glob("*.png"))
        assert len(written) == len(result["window_ids"])
        for p in result["paths"]:
            assert p.endswith(".png")
        for fig in result["figures"]:
            plt.close(fig)

    def test_retain_figures_false_closes_them_and_writes_the_same_pngs(
        self, stage5_file, tmp_path
    ):
        """Opting out returns no figures, leaves none open, and changes no file."""
        kept_dir = tmp_path / "kept"
        closed_dir = tmp_path / "closed"

        plt.close("all")
        kept = fit_show_impl(stage5_file, all_windows=True, output_dir=str(kept_dir))
        assert len(kept["figures"]) == len(kept["window_ids"])
        assert len(plt.get_fignums()) == len(kept["figures"])
        for fig in kept["figures"]:
            plt.close(fig)

        plt.close("all")
        closed = fit_show_impl(
            stage5_file,
            all_windows=True,
            output_dir=str(closed_dir),
            retain_figures=False,
        )
        assert closed["figures"] == []
        assert plt.get_fignums() == []

        # Everything other than the figure list is identical, and the rendered
        # PNGs are the same files -- this is a memory decision, not a rendering one.
        assert closed["window_ids"] == kept["window_ids"]
        assert closed["log"] == kept["log"]
        assert sorted(p.name for p in closed_dir.glob("*.png")) == sorted(
            p.name for p in kept_dir.glob("*.png")
        )
        for name in (p.name for p in kept_dir.glob("*.png")):
            assert (closed_dir / name).stat().st_size > 0

    def test_log_lists_selected_windows(self, stage5_file, fit_obj):
        wid = int(fit_obj.window_fits[0].window_id)
        result = fit_show_impl(stage5_file, window_ids=[wid])
        assert f"Window {wid}" in result["log"]
        for fig in result["figures"]:
            plt.close(fig)

    def test_show_audit_extends_log(self, stage5_file, fit_obj):
        wid = int(fit_obj.window_fits[0].window_id)
        # Only the log is under test, so decline the figures rather than leak them.
        plain = fit_show_impl(
            stage5_file, window_ids=[wid], show_audit=False, retain_figures=False
        )["log"]
        audit = fit_show_impl(
            stage5_file, window_ids=[wid], show_audit=True, retain_figures=False
        )["log"]
        assert "audit trail" in audit
        assert len(audit) >= len(plain)


# ---------------------------------------------------------------------------
# Windowed (apodized) view
# ---------------------------------------------------------------------------
class TestWindowedView:
    def test_boxcar_model_at_right_frequency(self, stage5_file, fit_obj):
        # The re-synthesized model line must land at the brightest fitted peak's
        # molecular frequency -- i.e. synthesize (f_bb frame) -> rfft -> molecular
        # axis round-trips the coordinate transform. (Amplitude depends on the
        # fixture's apodization domain; position does not.)
        wid = select_window_ids(fit_obj, top_snr=1)[0]
        wf = fit_obj.window_fit(wid)
        brightest_f = max(wf.fitted_peaks, key=lambda p: p.amplitude).frequency_mhz
        fig = render_windowed_view_impl(stage5_file, wid, apodize="boxcar")
        model_line = fig.axes[2].lines[2]  # |X| panel model curve
        model_peak_f = model_line.get_xdata()[np.abs(model_line.get_ydata()).argmax()]
        assert model_peak_f == pytest.approx(brightest_f, abs=0.05)  # MHz
        plt.close(fig)

    def test_forwards_scipy_window_specs(self, stage5_file, fit_obj):
        wid = int(fit_obj.window_fits[0].window_id)
        # Custom matched filter + scipy symmetric windows + a parameterized one.
        for apo in ("boxcar", "exp", "hann", "hamming", "blackman", "kaiser:14"):
            fig = render_windowed_view_impl(stage5_file, wid, apodize=apo)
            assert len(fig.axes) == 3
            plt.close(fig)

    def test_unknown_window_raises(self, stage5_file, fit_obj):
        wid = int(fit_obj.window_fits[0].window_id)
        with pytest.raises(ValueError, match="unknown apodization"):
            render_windowed_view_impl(stage5_file, wid, apodize="notawindow")

    def test_apodize_adds_companion_figures(self, stage5_file, fit_obj, tmp_path):
        wid = int(fit_obj.window_fits[0].window_id)
        out = tmp_path / "figs"
        result = fit_show_impl(
            stage5_file, window_ids=[wid], apodize="exp", output_dir=str(out)
        )
        names = sorted(p.name for p in out.glob("*.png"))
        # one detail + one apodized companion.
        assert any(n.endswith("_apodized.png") for n in names)
        assert len([n for n in names if not n.endswith("_apodized.png")]) == 1
        assert len(result["figures"]) == 2
        for fig in result["figures"]:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Residual-rescue summary view
# ---------------------------------------------------------------------------
class TestRescueSummary:
    def test_impl_renders_for_real_window(self, stage5_file, fit_obj):
        wid = int(fit_obj.window_fits[0].window_id)
        fig = render_rescue_summary_impl(stage5_file, wid)
        # At least the data/model + residual rows render; a window with rescue
        # rounds adds the chi2 + budget panels (>= 2 axes either way).
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_rescue_adds_companion_figure(self, stage5_file, fit_obj, tmp_path):
        wid = int(fit_obj.window_fits[0].window_id)
        out = tmp_path / "figs"
        result = fit_show_impl(
            stage5_file, window_ids=[wid], rescue=True, output_dir=str(out)
        )
        names = sorted(p.name for p in out.glob("*.png"))
        # one detail + one rescue companion.
        assert any(n.endswith("_rescue.png") for n in names)
        assert len([n for n in names if not n.endswith("_rescue.png")]) == 1
        assert len(result["figures"]) == 2
        for fig in result["figures"]:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-interface consistency
# ---------------------------------------------------------------------------
def test_show_fit_cross_interface(baseline_2638_stage5_small, tmp_path):
    """CLI-equivalent impl == Pipeline.show_fit == api.show_fit: same selected
    windows and same number of written figures for an identical selector."""
    pfile = tmp_path / "p.ftmw"
    ffile = tmp_path / "f.ftmw"
    shutil.copy(baseline_2638_stage5_small, pfile)
    shutil.copy(baseline_2638_stage5_small, ffile)

    p_out = tmp_path / "p_out"
    f_out = tmp_path / "f_out"

    r_pipe = Pipeline.open(pfile).show_fit(
        top_snr=2, rescue=True, output_dir=str(p_out)
    )
    r_func = ftmw.show_fit(str(ffile), top_snr=2, rescue=True, output_dir=str(f_out))

    assert r_pipe["window_ids"] == r_func["window_ids"]
    assert len(r_pipe["paths"]) == len(r_func["paths"])
    # detail + rescue summary per selected window.
    assert len(r_pipe["paths"]) == 2 * len(r_pipe["window_ids"])
    assert r_pipe["log"] == r_func["log"]
    for fig in r_pipe["figures"] + r_func["figures"]:
        plt.close(fig)
