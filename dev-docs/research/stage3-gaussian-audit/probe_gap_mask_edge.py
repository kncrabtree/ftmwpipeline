"""Probe 4: ``gap_pass.gap_mask_edge_threshold`` Gaussian retune.

Sweeps the de-ramped coherent-leakage cutoff above which gap-pass
detections are dropped as sidelobes. Gaussian envelope has a different
spectral leakage profile than Lorentzian (faster decay in frequency),
so the optimal cutoff may differ. Default = 8.0.

Run::

    conda run -n ftmwpipeline-dev python \\
        dev-docs/research/stage3-gaussian-audit/probe_gap_mask_edge.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from harness import (
    Stage3RunResult,
    plot_knob_sweep,
    prepare_fixture,
    run_detect_peaks,
    write_csv,
)

from ftmwpipeline.core.peak_detection_settings import PeakDetectionSettings

DATA = HERE / "data"
FIG = HERE / "figures"
DATA.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("probe-gap-mask-edge")

KNOB = "gap_pass.gap_mask_edge_threshold"
VALUES = (4.0, 6.0, 8.0, 10.0, 12.0)


def _settings_for(value: float) -> PeakDetectionSettings:
    s = PeakDetectionSettings()
    s.gap_pass.gap_mask_edge_threshold = float(value)
    return s


def main() -> None:
    lor_fp = prepare_fixture("lorentzian")
    gauss_fp = prepare_fixture("gaussian")
    rows: list[Stage3RunResult] = []
    for shape, fp in (("lorentzian", lor_fp), ("gaussian", gauss_fp)):
        for v in VALUES:
            r = run_detect_peaks(
                fp, shape=shape, settings=_settings_for(v),
                knob_label=KNOB, knob_value=v,
            )
            rows.append(r)
            logger.info(
                "  %s/%s=%.1f → n_peaks=%d, n_gap=%d, sidelobe-proxy=%d",
                shape, KNOB, v, r.n_peaks, r.n_gap,
                r.n_internal_high_user_low,
            )
    write_csv(rows, DATA / "probe_gap_mask_edge.csv")
    plot_knob_sweep(
        rows,
        x_label="gap_pass.gap_mask_edge_threshold",
        out_path=FIG / "probe_gap_mask_edge.png",
        title="Stage 3 `gap_pass.gap_mask_edge_threshold` sweep (2638 unapodized)",
    )


if __name__ == "__main__":
    main()
