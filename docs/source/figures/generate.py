"""Regenerate the representative figures embedded in the pipeline-stage pages.

Builds a complete Stage 0-5 pipeline from the checked-in ``2638`` Blackchirp
fixture in a temporary directory and renders one figure per documented stage
into the committed ``docs/source/figures`` directory:

* ``stage0_start_detection.png`` -- the Σ|FT|-vs-start sweep that locates the
  chirp end and the recommended active-region start;
* ``stage1_canonical_ft.png`` -- the zero-padded, active-band DISPLAY FT
  (magnitude plus real/imaginary parts; same surface as the Stage 5 report /
  ``fit show`` magnitude panels) -- display-only, not the native FT fitting
  and noise are scored on;
* ``stage2_noise.png`` -- the per-bin scatter noise estimate overlaid on the
  active spectrum with the 3x/5x reference levels;
* ``stage2b_tau_distribution.png`` -- the Stage 2b decay-time distribution panel
  (contributor histogram with the majority-vote overlay, tau vs SNR and tau vs
  molecular frequency with the per-band levels, and the GMM bimodality fit);
* ``stage2b_tau_decay_examples.png`` -- the per-bin magnitude-vs-window decay
  for a strong line (with the exponential and Gaussian fits), a clock spur, and
  a noise bin;
* ``stage2b_tau_heatmap_zoom.png`` -- the STFT magnitude heatmap zoomed to a
  strong-line neighborhood with a clipped color range so the decays read.
* ``stage3_peaks.png`` -- the Stage 3 detection overlay (classified peaks on the
  active spectrum, marker-styled by pass) with the SNR-distribution curation
  panel and the promotion cutoff.
* ``stage4_windows.png`` -- the Stage 4 window plan over the active spectrum (the
  fit-window spans, free peaks, and fixed contributors) with the rolling
  edge-coherence statistic and its T_edge threshold in a lower panel. Built from
  the denser vinyl-cyanide (1512) fixture, not 2638: the sparse 2638 spectrum
  carries no material cross-window skirt, so it has no contributors to show.
* ``stage5_fitting.png`` -- the Stage 5 fit overview: the fitted model overlaid
  on the active spectrum with the windows shaded, and the magnitude residual
  against the persisted noise in a lower panel.
* ``stage5_fit_detail.png`` -- the per-window fit detail for a representative
  window (a resolved close blend beside a third line, with a masked clock spur):
  the real/imaginary/magnitude data with the model and residuals, the residual
  histogram, and the fitted-peak table.
* ``stage6_review.png`` -- the Stage 6 report's full-spectrum index overview: the
  finalized active spectrum with the review-flagged (attention) windows shaded.
* ``fit_curation_annotations.png`` -- two windows' report magnitude panels (rendered
  through the report's own per-window painter) overlaid with a curation cart: two
  merges over over-split close pairs and a missed-line add on one window, two splits
  on another (Concepts / fit curation).

The timebase self-calibration figures (``clock_timebase.png`` /
``clock_phase_ramp.png``) are built by the methods note's own harness,
``docs/source/methods/timebase_calibration/generate.py``, not here.

The scope-record acquisition-layout schematic on the scope-record-import page is a
hand-authored SVG (``figures/scope_acquisition_layout.svg``), not generated here:
it is a conceptual, data-free diagram, so it is committed as editable vector text
rather than a rendered raster.

Run as a script to (re)write the PNGs beside this file::

    python docs/source/figures/generate.py

The figures use the shared house style (:mod:`ftmwpipeline.visualization.report_style`)
and are rendered title-less; the page captions label them. A ``slow`` smoke
test (``tests/integration/test_stage_figures.py``) confirms the harness renders
without error.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

# Repo root: docs/source/figures/generate.py -> parents[3].
_ROOT = Path(__file__).resolve().parents[3]
_FIXTURE = _ROOT / "examples" / "blackchirp_data" / "2638"
# The Stage 4 figure uses a denser fixture (vinyl cyanide) so the fixed
# contributors and the dependency structure are visible: 2638 is sparse enough
# that the materiality gate carries no cross-window skirt (zero contributors),
# while 1512 keeps a representative set.
_VCN_FIXTURE = _ROOT / "examples" / "blackchirp_data" / "1512"
FIG_DIR = Path(__file__).resolve().parent

TRIM = (26500.0, 40000.0)
DPI = 130


def _save(fig: Any, name: str, **kwargs: Any) -> None:
    """Write ``fig`` to ``FIG_DIR/name`` and close it.

    The plot helpers hand the figure to their caller, so closing is this
    script's job. Twelve figures held open at once is harmless when the
    script is the whole process, but ``make_figures`` also runs inside the
    test suite, where the figures outlive the test and push the session past
    matplotlib's ``figure.max_open_warning``.
    """
    import matplotlib.pyplot as plt

    kwargs.setdefault("dpi", DPI)
    kwargs.setdefault("bbox_inches", "tight")
    fig.savefig(FIG_DIR / name, **kwargs)
    plt.close(fig)


def _build_pipeline(workdir: Path) -> str:
    """Import the 2638 fixture and run Stages 0-2b; return the ``.ftmw`` path."""
    import ftmwpipeline.api as ftmw

    path = str(workdir / "exp_2638.ftmw")
    ftmw.import_data(path, source=str(_FIXTURE))
    ftmw.detect_start_time(path, stamp=True)
    ftmw.compute_ft(path, trim=TRIM)
    ftmw.estimate_noise(path)
    # Default tau calibration: the auto-recommend shape vote runs and stamps the
    # recommended shape (Gaussian on 2638), building both twins, so Stage 5 fits
    # in the recommended shape -- the same fit a default run produces. (Forcing
    # the Lorentzian twin here would over-split the lines; see the Stage 5 note.)
    ftmw.calibrate_tau(path)
    ftmw.detect_peaks(path)
    ftmw.assign_windows(path)
    ftmw.fit_peaks(path)
    # Stage 6 review: build the curation layer (attention routing) and the
    # consolidated final-products table the report figure reads.
    ftmw.review_run(path)
    return path


def _build_through_windows(workdir: Path, fixture: Path, stem: str) -> str:
    """Import ``fixture`` and run Stages 0-4 (no fit); return the ``.ftmw`` path.

    Used for the Stage 4 window-plan figure, which only needs the plan, the
    active FT, and the noise -- not a completed fit.
    """
    import ftmwpipeline.api as ftmw

    path = str(workdir / f"{stem}.ftmw")
    ftmw.import_data(path, source=str(fixture))
    ftmw.detect_start_time(path, stamp=True)
    ftmw.compute_ft(path, trim=TRIM)
    ftmw.estimate_noise(path)
    ftmw.calibrate_tau(path)
    ftmw.detect_peaks(path)
    ftmw.assign_windows(path)
    return path


def make_figures() -> None:
    """Build the pipeline and write the stage figures."""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")

    import ftmwpipeline.api as ftmw
    from ftmwpipeline._internal.stage1_impl import visualize_ft_impl
    from ftmwpipeline._internal.stage2_impl import visualize_noise_impl
    from ftmwpipeline._internal.stage3_impl import visualize_peaks_impl
    from ftmwpipeline._internal.stage4_impl import visualize_windows_impl
    from ftmwpipeline._internal.stage5_impl import (
        load_fit_impl,
        render_fit_detail_impl,
        visualize_fit_impl,
    )
    from ftmwpipeline.visualization.report_style import apply_color_cycle
    from ftmwpipeline.visualization.start_detection_visualization import (
        plot_start_detection_from_file,
    )
    from ftmwpipeline.visualization.tau_calibration_visualization import (
        plot_stft_decay_examples_from_file,
        plot_tau_distribution_from_file,
        plot_tau_heatmap_from_file,
    )

    apply_color_cycle(matplotlib)

    with tempfile.TemporaryDirectory() as tmp:
        path = _build_pipeline(Path(tmp))

        fig0 = plot_start_detection_from_file(path, title="")
        _save(fig0, "stage0_start_detection.png")

        fig1 = visualize_ft_impl(
            path, title="", show_fid_panels=False, interactive=False
        )
        _save(fig1, "stage1_canonical_ft.png")

        fig2 = visualize_noise_impl(path, title="", interactive=False)
        _save(fig2, "stage2_noise.png")

        fig2b = plot_tau_distribution_from_file(path, shape="lorentzian", title="")
        _save(fig2b, "stage2b_tau_distribution.png")

        fig2c = plot_stft_decay_examples_from_file(path, shape="lorentzian", title="")
        _save(fig2c, "stage2b_tau_decay_examples.png")

        # Zoom the heatmap to the neighborhood of the strongest contributor and
        # clip the color range so the per-line decays read clearly.
        cal = ftmw.load_tau_calibration(path)
        f_center = float(
            cal.contributor_freqs_mhz[int(np.argmax(cal.contributor_snrs))]
        )
        fig2d = plot_tau_heatmap_from_file(
            path,
            shape="lorentzian",
            title="",
            freq_window=(f_center - 120.0, f_center + 120.0),
            clip_percentiles=(60.0, 99.9),
        )
        _save(fig2d, "stage2b_tau_heatmap_zoom.png")

        fig3 = visualize_peaks_impl(
            path, title="", interactive=False, show_snr_histogram=True
        )
        _save(fig3, "stage3_peaks.png")

        # Stage 4 window plan on the denser vinyl-cyanide fixture (see
        # ``_VCN_FIXTURE``), where the fixed contributors and dependency edges
        # are visible. Built in its own temp dir through Stage 4 only.
        with tempfile.TemporaryDirectory() as tmp4:
            wpath = _build_through_windows(Path(tmp4), _VCN_FIXTURE, "exp_1512")
            fig4 = visualize_windows_impl(wpath, title="", interactive=False)
            _save(fig4, "stage4_windows.png")

        fig5 = visualize_fit_impl(path, title="", interactive=False)
        _save(fig5, "stage5_fitting.png")

        # Per-window detail: the window covering ~36350 MHz -- a strong, resolved
        # doublet near 36350 (SNR ~300) beside a weak doublet near 36352 (SNR ~40),
        # all determined, so the qual determinacy score reads 4/4 across the window
        # (a weak line pinned as firmly as a strong one; determinacy is not
        # strength). Select it by frequency so the example survives window
        # renumbering; fall back to the highest-SNR multi-line window.
        fit5 = load_fit_impl(path)["fit"]
        detail_wid = None
        for wf in fit5.window_fits:
            if wf.window is None or not wf.fitted_peaks:
                continue
            lo, hi = wf.window.freq_range
            if min(lo, hi) <= 36350.0 <= max(lo, hi):
                detail_wid = wf.window_id
                break
        if detail_wid is None:
            cand = [
                (max((p.snr or 0.0) for p in wf.fitted_peaks), wf.window_id)
                for wf in fit5.window_fits
                if wf.window is not None and len(wf.fitted_peaks) >= 2
            ]
            detail_wid = max(cand)[1] if cand else fit5.window_fits[0].window_id
        fig5b = render_fit_detail_impl(path, detail_wid, title="")
        _save(fig5b, "stage5_fit_detail.png")

        # Stage 6: the report's full-spectrum index overview -- the finalized
        # spectrum with the review-flagged (attention) windows shaded. This is
        # the review surface over the consolidated line list.
        from ftmwpipeline._internal.report_html_impl import (
            _plot_index_overview,
            _window_range,
        )
        from ftmwpipeline._internal.stage5_impl import _resolve_detail_bundle
        from ftmwpipeline.io.stage6_review_serialization import (
            load_stage6_review_from_file,
        )

        bundle = _resolve_detail_bundle(path)
        review = load_stage6_review_from_file(path)
        win_fits = {
            int(wf.window_id): wf
            for wf in bundle.fit.window_fits
            if wf.window is not None and wf.window_id is not None
        }
        attention_ranges = [
            _window_range(win_fits[wid])
            for wid, st in sorted(review.window_statuses.items())
            if st.needs_attention and wid in win_fits
        ]
        fig6 = _plot_index_overview(
            bundle, attention_ranges, title="", figsize=(13.0, 3.2)
        )
        _save(fig6, "stage6_review.png")

        # Concepts / fit curation: two report magnitude panels overlaid with a
        # real curation cart (merges, an add, splits), in the report's color code.
        fig_cur = _plot_curation_annotations(bundle, np)
        _save(fig_cur, "fit_curation_annotations.png")


# A curation cart captured from the browser report on the 2638 example: on
# window 120 a weak line is split into two and a missed line is added between the
# pair; on window 265 a resolved close pair is merged back into one. Frequencies
# are the raw Stage 5 model values the edit verbs match on (the values the cart
# emits), so the figure annotates the real fitted peaks by frequency.
_CURATION_DEMO = (
    (120, "split", (31076.184844,)),
    (120, "add", (31074.2667,)),
    (265, "merge", (36147.043509, 36147.138722)),
)
_CURATION_INTO = 2  # the demo splits each line into two


def _plot_curation_annotations(bundle: Any, np: Any) -> Any:
    """The report's own window panels, annotated with a real curation cart.

    Paints the magnitude panel for each window named in :data:`_CURATION_DEMO`
    through the same renderer the HTML report uses
    (:func:`~ftmwpipeline.visualization.fit_detail.draw_component`, on the 2x
    zero-padded display grid with the fitted model and the absolute-frequency
    axis), then overlays the cart's queued edits exactly as the browser does: a
    vertical marker per frequency, colored by action. Window 120 carries a split
    of a weak line and a missed-line add between the pair; window 265 carries a
    merge over a resolved close pair. The markers are queued intentions, not
    applied edits.
    """
    import matplotlib.pyplot as plt

    from ftmwpipeline.visualization.fit_detail import (
        _stacked_pair,
        draw_component,
        prepare_window_panels,
    )
    from ftmwpipeline.visualization.report_style import (
        DOUBLE_DECKER,
        PINOT,
        POPPY,
        QUAD,
    )

    colors = {"remove": DOUBLE_DECKER, "add": QUAD, "merge": PINOT, "split": POPPY}

    # Resolve each cart group's panel by frequency, not window id: the ids are
    # incidental (and renumber when the fit changes), but the lines persist.
    def _window_for(freq: float) -> Any:
        for wf in bundle.fit.window_fits:
            if wf.window is None:
                continue
            lo, hi = wf.window.freq_range
            if min(lo, hi) <= freq <= max(lo, hi):
                return wf
        raise ValueError(f"no fit window contains {freq} MHz")

    panel_ids: list[int] = []
    for wid, _action, _freqs in _CURATION_DEMO:
        if wid not in panel_ids:
            panel_ids.append(wid)

    fig = plt.figure(figsize=(12.0, 4.8), constrained_layout=True)
    outer = fig.add_gridspec(1, len(panel_ids), wspace=0.16)
    spurs = (bundle.fit.diagnostics or {}).get("gated_spurs")

    for col, wid in enumerate(panel_ids):
        ops = [(a, fs) for (w, a, fs) in _CURATION_DEMO if w == wid]
        data = prepare_window_panels(
            _window_for(ops[0][1][0]),
            frequencies=bundle.frequencies,
            complex_spectrum=bundle.complex_spectrum,
            rms_noise=bundle.rms_noise,
            sideband=bundle.sideband,
            acquisition_us=bundle.acquisition_us,
            amplitude_scale=bundle.amplitude_scale,
            units_label=bundle.units_label,
            trim_mhz=bundle.trim_mhz,
            freq_padded=bundle.freq_padded,
            spec_padded=bundle.spec_padded,
            spurs=spurs,
        )
        ax_resid, ax_data = _stacked_pair(fig, outer[col])
        draw_component(ax_resid, ax_data, data, "mag")

        # Overlay the queued edits on the magnitude panel, as the report does.
        y_lo, y_hi = ax_data.get_ylim()
        ax_data.set_ylim(y_lo, y_hi * 1.16)
        for action, fs in ops:
            color = colors[action]
            for fc in fs:
                ax_data.plot(
                    [fc, fc],
                    [y_lo, y_hi],
                    color=color,
                    lw=1.6,
                    ls="--" if action == "remove" else "-",
                    zorder=6,
                )
                ax_data.plot([fc], [y_hi], marker="o", color=color, ms=6, zorder=7)
            text = f"split →{_CURATION_INTO}" if action == "split" else action
            ax_data.annotate(
                text,
                (0.5 * (fs[0] + fs[-1]), y_hi),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=color,
                zorder=7,
            )

    return fig


def main() -> None:
    make_figures()
    print(f"Wrote stage figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
