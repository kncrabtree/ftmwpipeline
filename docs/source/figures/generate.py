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


def main() -> None:
    make_figures()
    print(f"Wrote stage figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
