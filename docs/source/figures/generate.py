"""Regenerate the representative figures embedded in the pipeline-stage pages.

Builds a complete Stage 0-2 pipeline from the checked-in ``2638`` Blackchirp
fixture in a temporary directory and renders one figure per documented early
stage into the committed ``docs/source/figures`` directory:

* ``stage0_start_detection.png`` -- the Σ|FT|-vs-start sweep that locates the
  chirp end and the recommended active-region start;
* ``stage1_canonical_ft.png`` -- the canonical unapodized FT over the active
  band (magnitude plus real/imaginary parts);
* ``stage2_noise.png`` -- the per-bin scatter noise estimate overlaid on the
  active spectrum with the 3x/5x reference levels;
* ``stage2b_tau_distribution.png`` -- the Stage 2b decay-time distribution panel
  (contributor histogram with the majority-vote overlay, tau vs SNR and tau vs
  molecular frequency, and the GMM bimodality fit).

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
    return path


def make_figures() -> None:
    """Build the pipeline and write the three stage figures."""
    import matplotlib

    matplotlib.use("Agg")

    from ftmwpipeline._internal.stage1_impl import visualize_ft_impl
    from ftmwpipeline._internal.stage2_impl import visualize_noise_impl
    from ftmwpipeline.visualization.report_style import apply_color_cycle
    from ftmwpipeline.visualization.start_detection_visualization import (
        plot_start_detection_from_file,
    )
    from ftmwpipeline.visualization.tau_calibration_visualization import (
        plot_tau_distribution_from_file,
    )

    apply_color_cycle(matplotlib)

    with tempfile.TemporaryDirectory() as tmp:
        path = _build_pipeline(Path(tmp))

        fig0 = plot_start_detection_from_file(path, title="")
        fig0.savefig(FIG_DIR / "stage0_start_detection.png", dpi=DPI, bbox_inches="tight")

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


def main() -> None:
    make_figures()
    print(f"Wrote stage figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
