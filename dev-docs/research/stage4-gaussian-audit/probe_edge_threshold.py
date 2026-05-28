"""Probe 2: ``coherence.edge_threshold`` Gaussian retune.

Sweeps the S_coh cutoff that flags leakage-touched regions for refinement.
Default = 8.0 (= √M at M=64 cache band width). The Gaussian envelope has
a different sidelobe profile than Lorentzian -- spectral leakage decays
faster -- so the optimal cutoff may shift on the Gaussian path.

Run::

    conda run -n ftmwpipeline-dev python \\
        dev-docs/research/stage4-gaussian-audit/probe_edge_threshold.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from harness import (
    Stage4RunResult,
    plot_knob_sweep,
    prepare_fixture,
    run_assign_windows,
    write_csv,
)

from ftmwpipeline.core.window_planning_settings import WindowPlanningSettings

DATA = HERE / "data"
FIG = HERE / "figures"
DATA.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("probe-edge-threshold")

KNOB = "coherence.edge_threshold"
VALUES = (4.0, 6.0, 8.0, 10.0, 12.0)


def _settings_for(value: float) -> WindowPlanningSettings:
    s = WindowPlanningSettings()
    s.coherence.edge_threshold = float(value)
    return s


def main() -> None:
    lor_fp = prepare_fixture("lorentzian")
    gauss_fp = prepare_fixture("gaussian")
    rows: list[Stage4RunResult] = []
    for shape, fp in (("lorentzian", lor_fp), ("gaussian", gauss_fp)):
        for v in VALUES:
            r = run_assign_windows(
                fp, shape=shape, settings=_settings_for(v),
                knob_label=KNOB, knob_value=v,
            )
            rows.append(r)
            logger.info(
                "  %s/%s=%.1f → n_windows=%d, n_hard=%d, n_fixed=%d",
                shape, KNOB, v, r.n_windows, r.n_hard, r.n_fixed_contributors,
            )
    write_csv(rows, DATA / "probe_edge_threshold.csv")
    plot_knob_sweep(
        rows,
        x_label="coherence.edge_threshold",
        out_path=FIG / "probe_edge_threshold.png",
        title="Stage 4 `coherence.edge_threshold` sweep (2638 unapodized)",
    )


if __name__ == "__main__":
    main()
