"""Probe 3: ``promotion.internal_min_snr`` Gaussian retune.

Sweeps the internal zpf=1 detection floor across the Lorentzian and
Gaussian paths on the 2638 fixture. Default = 2.0; the floor at which
``detect_peaks`` actually fires on the apodization-smeared internal
grid. Pulling it lower recovers lines the user-grid re-measure would
otherwise miss; pulling it higher trims the candidate pool.

Run::

    conda run -n ftmwpipeline-dev python \\
        dev-docs/research/stage3-gaussian-audit/probe_internal_min_snr.py
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
logger = logging.getLogger("probe-internal-min-snr")

KNOB = "promotion.internal_min_snr"
VALUES = (1.5, 2.0, 2.5, 3.0)


def _settings_for(value: float) -> PeakDetectionSettings:
    s = PeakDetectionSettings()
    s.promotion.internal_min_snr = float(value)
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
                "  %s/%s=%.2f → n_peaks=%d, n_promoted=%d, sidelobe-proxy=%d",
                shape, KNOB, v, r.n_peaks, r.n_promoted,
                r.n_internal_high_user_low,
            )
    write_csv(rows, DATA / "probe_internal_min_snr.csv")
    plot_knob_sweep(
        rows,
        x_label="promotion.internal_min_snr",
        out_path=FIG / "probe_internal_min_snr.png",
        title="Stage 3 `promotion.internal_min_snr` sweep (2638 unapodized)",
    )


if __name__ == "__main__":
    main()
