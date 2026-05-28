"""Probe 4: ``contributor.magnitude_attachment_threshold`` Gaussian retune.

Sweeps the tier-1 contributor-attachment threshold in σ_c units. A
strong promoted peak attaches to a window's ``fixed_contributors``
when its predicted mean |skirt| on that window's grid is at least
``threshold × sigma_c(w)``. Default = 0.1. Originally calibrated under
a Lorentzian skirt assumption; the Gaussian envelope has a different
skirt profile, so the threshold may shift.

Run::

    conda run -n ftmwpipeline-dev python \\
        dev-docs/research/stage4-gaussian-audit/probe_mag_attachment.py
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
logger = logging.getLogger("probe-mag-attachment")

KNOB = "contributor.magnitude_attachment_threshold"
VALUES = (0.05, 0.075, 0.1, 0.15, 0.2)


def _settings_for(value: float) -> WindowPlanningSettings:
    s = WindowPlanningSettings()
    s.contributor.magnitude_attachment_threshold = float(value)
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
                "  %s/%s=%.3f → n_windows=%d, n_fixed=%d, n_dep=%d",
                shape, KNOB, v, r.n_windows, r.n_fixed_contributors,
                r.n_dependencies,
            )
    write_csv(rows, DATA / "probe_mag_attachment.csv")
    plot_knob_sweep(
        rows,
        x_label="contributor.magnitude_attachment_threshold",
        out_path=FIG / "probe_mag_attachment.png",
        title="Stage 4 `contributor.magnitude_attachment_threshold` sweep (2638 unapodized)",
    )


if __name__ == "__main__":
    main()
