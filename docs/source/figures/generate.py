"""Regenerate the representative figures embedded in the pipeline-stage pages.

Builds a complete Stage 0-5 pipeline from the checked-in ``2638`` Blackchirp
fixture in a temporary directory and renders one figure per documented stage
into the committed ``docs/source/figures`` directory:

* ``stage0_start_detection.png`` -- the Σ|FT|-vs-start sweep that locates the
  chirp end and the recommended active-region start;
* ``stage1_canonical_ft.png`` -- the canonical unapodized FT over the active
  band (magnitude plus real/imaginary parts);
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
  edge-coherence statistic and its T_edge threshold in a lower panel.
* ``stage5_fitting.png`` -- the Stage 5 fit overview: the fitted model overlaid
  on the active spectrum with the windows shaded, and the magnitude residual
  against the canonical noise in a lower panel.
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
* ``clock_timebase.png`` -- the timebase self-calibration: each Rb-locked clock
  tone's measured frequency offset against its baseband frequency, with the
  shared-epsilon fit line and its 1-sigma band (Advanced / clock declaration).
* ``clock_phase_ramp.png`` -- how one tone's offset is measured: the residual
  phase ramp of the strongest demodulated clock tone and the coherent-sum scan
  whose peak locates the offset (Advanced / clock declaration).

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
FIG_DIR = Path(__file__).resolve().parent

TRIM = (26500.0, 40000.0)
DPI = 130


def _build_pipeline(workdir: Path) -> str:
    """Import the 2638 fixture and run Stages 0-2b; return the ``.ftmw`` path."""
    import ftmwpipeline.api as ftmw
    from ftmwpipeline.core.tau_calibration_settings import (
        RecommendationSubSettings,
        TauCalibrationSettings,
    )

    path = str(workdir / "exp_2638.ftmw")
    ftmw.import_data(path, source=str(_FIXTURE))
    ftmw.detect_start_time(path, stamp=True)
    ftmw.compute_ft(path, trim=TRIM)
    ftmw.estimate_noise(path)
    # Lorentzian tau calibration only; skip the auto-recommend pass (and its
    # cross-built Gaussian twin) so the figure build stays lean — the
    # distribution figure reads the exponential-twin contributor histogram.
    ftmw.calibrate_tau(
        path,
        settings=TauCalibrationSettings(
            recommendation=RecommendationSubSettings(auto_recommend=False)
        ),
    )
    ftmw.detect_peaks(path)
    ftmw.assign_windows(path)
    ftmw.fit_peaks(path)
    # Stage 6 review: build the curation layer (attention routing) and the
    # consolidated final-products table the report figure reads.
    ftmw.review_run(path)
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
        fig0.savefig(
            FIG_DIR / "stage0_start_detection.png", dpi=DPI, bbox_inches="tight"
        )

        fig1 = visualize_ft_impl(
            path, title="", show_fid_panels=False, interactive=False
        )
        fig1.savefig(FIG_DIR / "stage1_canonical_ft.png", dpi=DPI, bbox_inches="tight")

        fig2 = visualize_noise_impl(path, title="", interactive=False)
        fig2.savefig(FIG_DIR / "stage2_noise.png", dpi=DPI, bbox_inches="tight")

        fig2b = plot_tau_distribution_from_file(path, shape="lorentzian", title="")
        fig2b.savefig(
            FIG_DIR / "stage2b_tau_distribution.png", dpi=DPI, bbox_inches="tight"
        )

        fig2c = plot_stft_decay_examples_from_file(path, shape="lorentzian", title="")
        fig2c.savefig(
            FIG_DIR / "stage2b_tau_decay_examples.png", dpi=DPI, bbox_inches="tight"
        )

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
        fig2d.savefig(
            FIG_DIR / "stage2b_tau_heatmap_zoom.png", dpi=DPI, bbox_inches="tight"
        )

        fig3 = visualize_peaks_impl(
            path, title="", interactive=False, show_snr_histogram=True
        )
        fig3.savefig(FIG_DIR / "stage3_peaks.png", dpi=DPI, bbox_inches="tight")

        fig4 = visualize_windows_impl(path, title="", interactive=False)
        fig4.savefig(FIG_DIR / "stage4_windows.png", dpi=DPI, bbox_inches="tight")

        fig5 = visualize_fit_impl(path, title="", interactive=False)
        fig5.savefig(FIG_DIR / "stage5_fitting.png", dpi=DPI, bbox_inches="tight")

        # Per-window detail: the window covering ~29148 MHz -- a resolved close
        # blend beside a third line, with a masked clock spur. Select it by
        # frequency so the example survives any change in window numbering;
        # fall back to the highest-SNR multi-line window.
        fit5 = load_fit_impl(path)["fit"]
        detail_wid = None
        for wf in fit5.window_fits:
            if wf.window is None or not wf.fitted_peaks:
                continue
            lo, hi = wf.window.freq_range
            if min(lo, hi) <= 29148.0 <= max(lo, hi):
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
        fig5b.savefig(FIG_DIR / "stage5_fit_detail.png", dpi=DPI, bbox_inches="tight")

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
        fig6.savefig(FIG_DIR / "stage6_review.png", dpi=DPI, bbox_inches="tight")

        # Concepts / fit curation: two report magnitude panels overlaid with a
        # real curation cart (merges, an add, splits), in the report's color code.
        fig_cur = _plot_curation_annotations(bundle, np)
        fig_cur.savefig(
            FIG_DIR / "fit_curation_annotations.png", dpi=DPI, bbox_inches="tight"
        )

        # Advanced: the timebase self-calibration. Measure epsilon from the
        # Rb-locked clock spurs and plot each tone's measured offset against its
        # baseband frequency -- the kept tones fall on the shared-epsilon line.
        result = ftmw.calibrate_timebase(path)
        fig_tb = _plot_timebase(result, np)
        fig_tb.savefig(FIG_DIR / "clock_timebase.png", dpi=DPI, bbox_inches="tight")

        # Advanced: how one tone's offset is measured -- the residual phase ramp
        # of the demodulated tone and the coherent-sum scan that locates it.
        fig_pr = _plot_phase_ramp(path, result, np)
        fig_pr.savefig(FIG_DIR / "clock_phase_ramp.png", dpi=DPI, bbox_inches="tight")


# A curation cart captured from the browser report on the 2638 example: two
# over-split close pairs merged and a missed line added on window 61, and two
# weak lines split on window 80. Frequencies are the raw Stage 5 model values the
# edit verbs match on (the values the cart emits), so the figure annotates the
# real fitted peaks by frequency.
_CURATION_DEMO = (
    (61, "merge", (28802.036876, 28802.091386)),
    (61, "merge", (28802.843411, 28802.899680)),
    (61, "add", (28799.9725,)),
    (80, "split", (29351.363439,)),
    (80, "split", (29352.880164,)),
)
_CURATION_INTO = 2  # the demo splits each line into two


def _plot_curation_annotations(bundle: Any, np: Any) -> Any:
    """The report's own window panels, annotated with a real curation cart.

    Paints the magnitude panel for each window named in :data:`_CURATION_DEMO`
    through the same renderer the HTML report uses
    (:func:`~ftmwpipeline.visualization.fit_detail.draw_component`, on the 2x
    zero-padded display grid with the fitted model and the absolute-frequency
    axis), then overlays the cart's queued edits exactly as the browser does: a
    vertical marker per frequency, colored by action. Window 61 carries two
    merges over over-split close pairs and a missed-line add; window 80 carries
    two splits. The markers are queued intentions, not applied edits.
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


def _plot_timebase(result: Any, np: Any) -> Any:
    """Per-tone timebase self-calibration scatter with the fitted-epsilon line."""
    import matplotlib.pyplot as plt

    from ftmwpipeline.visualization.report_style import (
        AGGIE_BLUE,
        DOUBLE_DECKER,
        POPPY,
        apply_bare_style,
    )

    tones = list(result.tone_reads)
    kept = [t for t in tones if t.used]
    rej = [t for t in tones if not t.used]
    eps = result.epsilon
    sig = result.sigma_epsilon

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    # The fitted epsilon line (offset in kHz = epsilon * f_bb_MHz * 1e3) with its
    # 1-sigma band, drawn across the measured baseband range.
    f_max = max((t.f_bb_mhz for t in tones), default=1.0)
    xs = np.linspace(0.0, f_max * 1.02, 200)
    ax.fill_between(
        xs / 1e3,
        (eps - sig) * xs * 1e3,
        (eps + sig) * xs * 1e3,
        color=AGGIE_BLUE,
        alpha=0.15,
        linewidth=0,
    )
    ax.plot(
        xs / 1e3,
        eps * xs * 1e3,
        color=AGGIE_BLUE,
        linewidth=1.6,
        label=rf"$\varepsilon = {eps*1e6:+.2f} \pm {sig*1e6:.2f}$ ppm",
    )
    if rej:
        ax.scatter(
            [t.f_bb_mhz / 1e3 for t in rej],
            [t.df_mhz * 1e3 for t in rej],
            facecolors="none",
            edgecolors=POPPY,
            s=34,
            linewidths=1.2,
            label="rejected (off-line / inconsistent)",
        )
    if kept:
        ax.scatter(
            [t.f_bb_mhz / 1e3 for t in kept],
            [t.df_mhz * 1e3 for t in kept],
            color=DOUBLE_DECKER,
            s=38,
            label="kept (shared-$\\varepsilon$ fit)",
            zorder=3,
        )
    ax.axhline(0.0, color="#9aa3ad", linewidth=0.8, zorder=0)
    ax.set_xlabel("Baseband frequency (GHz)")
    ax.set_ylabel("Measured tone offset (kHz)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    apply_bare_style(ax)
    return fig


def _plot_phase_ramp(path: str, result: Any, np: Any) -> Any:
    """Demonstrate the phase-ramp measurement on the strongest clock tone.

    Reconstructs the calibration's active segment and demodulates it at the
    tone's nominal baseband frequency (exactly as ``calibrate_timebase_from_fid``
    does), then shows the residual rotation as (left) the block-averaged phase
    advancing linearly in time and (right) the coherent-sum scan whose peak is
    the measured offset.
    """
    import matplotlib.pyplot as plt

    from ftmwpipeline._internal.stage0_impl import load_fid_from_pipeline_impl
    from ftmwpipeline.fitting.timebase_calibration import _build_block_demod
    from ftmwpipeline.visualization.report_style import (
        AGGIE_BLUE,
        DOUBLE_DECKER,
        POPPY,
        apply_bare_style,
    )

    # The strongest tone kept in the fit makes the cleanest illustration.
    tone = max((t for t in result.tone_reads if t.used), key=lambda t: t.snr)
    fbb = float(tone.f_bb_mhz)

    # Rebuild the exact active segment and time vector the calibration used.
    fid = load_fid_from_pipeline_impl(path)
    x = np.asarray(fid.data, dtype=float)
    dt = float(result.sample_dt_us)
    i0 = max(int(result.start_us / dt), 0)
    i1 = min(int(result.end_us / dt), x.size)
    seg = x[i0:i1]
    t = (np.arange(i0, i1) - i0) * dt

    z = seg * np.exp(-2j * np.pi * fbb * t)

    # The coherent-sum scan uses the full block count (matching the estimator);
    # the offset is the grid frequency that maximizes the coherent amplitude.
    grid, _t_blocks, scan_matrix = _build_block_demod(t, 4096, 0.1, 0.0001)
    zb = np.array([c.mean() for c in np.array_split(z, 4096)])
    amp = np.abs(scan_matrix @ zb)
    df = float(grid[int(np.argmax(amp))])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)

    # Left: the residual phase ramp. The demodulated tone rotates at the offset,
    # so the slope of its phase in time is the offset (the same demodulation as
    # the scan; coarser blocks here only smooth the display). For a positive
    # offset df, the phase advances as +360*df*t degrees.
    n_disp = 160
    zc = np.array([c.mean() for c in np.array_split(z, n_disp)])
    tc = np.array([c.mean() for c in np.array_split(t, n_disp)])
    phase_deg = np.degrees(np.unwrap(np.angle(zc)))
    slope_line = phase_deg[0] + 360.0 * df * (tc - tc[0])
    axL.plot(
        tc,
        phase_deg,
        color=DOUBLE_DECKER,
        linewidth=1.4,
        label=f"residual phase ({fbb/1e3:.2f} GHz tone)",
    )
    axL.plot(
        tc,
        slope_line,
        color=AGGIE_BLUE,
        linewidth=1.2,
        linestyle="--",
        label=rf"slope $\Rightarrow \delta f = {df*1e3:+.1f}$ kHz",
    )
    axL.set_xlabel("Time (µs)")
    axL.set_ylabel("Demodulated phase (degrees)")
    axL.legend(loc="best", frameon=False, fontsize=9)
    apply_bare_style(axL)

    # Right: the coherent-sum scan. Its peak is the maximum-likelihood offset.
    axR.plot(grid * 1e3, amp / amp.max(), color=AGGIE_BLUE, linewidth=1.4)
    axR.axvline(0.0, color="#9aa3ad", linewidth=0.8, label="nominal frequency")
    axR.axvline(
        df * 1e3,
        color=POPPY,
        linewidth=1.2,
        linestyle="--",
        label=rf"peak $\Rightarrow \delta f = {df*1e3:+.1f}$ kHz",
    )
    axR.set_xlabel("Candidate offset from nominal (kHz)")
    axR.set_ylabel("Coherent-sum amplitude (normalized)")
    axR.legend(loc="upper left", frameon=False, fontsize=9)
    apply_bare_style(axR)
    return fig


def main() -> None:
    make_figures()
    print(f"Wrote stage figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
