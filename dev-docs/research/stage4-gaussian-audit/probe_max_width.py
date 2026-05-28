"""Probe 3: ``clustering.max_window_width_mhz`` Gaussian retune.

Sweeps the width cap above which windows become HARD and gain a split
proposal. Default = 40.0 MHz. Question: does the 40 MHz cap line up
with the typical Gaussian linewidth distribution on 2638, or does a
Gaussian-narrower spectrum want a different cap?

Run::

    conda run -n ftmwpipeline-dev python \\
        dev-docs/research/stage4-gaussian-audit/probe_max_width.py
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
logger = logging.getLogger("probe-max-width")

KNOB = "clustering.max_window_width_mhz"
VALUES = (20.0, 30.0, 40.0, 60.0, 80.0)


def _settings_for(value: float) -> WindowPlanningSettings:
    s = WindowPlanningSettings()
    s.clustering.max_window_width_mhz = float(value)
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
                "  %s/%s=%.1f → n_windows=%d, n_hard=%d, "
                "width p95=%.2f, n_split_proposed=%d",
                shape, KNOB, v, r.n_windows, r.n_hard,
                r.width_p95_mhz, r.n_split_proposed,
            )
    write_csv(rows, DATA / "probe_max_width.csv")
    plot_knob_sweep(
        rows,
        x_label="clustering.max_window_width_mhz",
        out_path=FIG / "probe_max_width.png",
        title="Stage 4 `clustering.max_window_width_mhz` sweep (2638 unapodized)",
    )


if __name__ == "__main__":
    main()
