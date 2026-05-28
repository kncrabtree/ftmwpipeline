"""Probe 5: ``primary_pass.min_exclusion_mhz`` Gaussian retune.

Sweeps the half-width around each primary peak that is excluded from
the gap pass. Default = 0.0 (no exclusion floor; the gap pass relies
on the leakage mask alone). Gaussian lines decay faster than
Lorentzian skirts so a non-zero floor may not be necessary -- but
they're also narrower in frequency, so a too-small exclusion might
let the gap pass re-detect lines that the primary pass already found.

Run::

    conda run -n ftmwpipeline-dev python \\
        dev-docs/research/stage3-gaussian-audit/probe_min_exclusion.py
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
logger = logging.getLogger("probe-min-exclusion")

KNOB = "primary_pass.min_exclusion_mhz"
VALUES = (0.0, 0.05, 0.1, 0.2, 0.5)


def _settings_for(value: float) -> PeakDetectionSettings:
    s = PeakDetectionSettings()
    s.primary_pass.min_exclusion_mhz = float(value)
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
                "  %s/%s=%.2f → n_peaks=%d, n_gap=%d, n_primary=%d",
                shape, KNOB, v, r.n_peaks, r.n_gap, r.n_primary,
            )
    write_csv(rows, DATA / "probe_min_exclusion.csv")
    plot_knob_sweep(
        rows,
        x_label="primary_pass.min_exclusion_mhz",
        out_path=FIG / "probe_min_exclusion.png",
        title="Stage 3 `primary_pass.min_exclusion_mhz` sweep (2638 unapodized)",
    )


if __name__ == "__main__":
    main()
