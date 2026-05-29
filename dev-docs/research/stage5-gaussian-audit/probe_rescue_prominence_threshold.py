"""Probe Stage 5 ``rescue.prominence_threshold`` against 2638 chi^2.

The knob is the residual-peak prominence floor (in σ_c units; default
2.0) gating which |residual| local maxima are eligible for rescue
nomination. Independent of ``rescue.snr_threshold`` -- a candidate has
to clear *both* floors to be promoted, so each gate selects a different
slice of the rescue's candidate pool.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from harness import prepare_fixture, run_variant  # noqa: E402

KNOB = "rescue.prominence_threshold"
VALUES = [1.0, 1.5, 2.0, 3.0]  # default = 2.0

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("probe-rescue-prominence-threshold")


def main() -> None:
    lor_fp = prepare_fixture("lorentzian")
    gauss_fp = prepare_fixture("gaussian")
    for shape, fp in (("lorentzian", lor_fp), ("gaussian", gauss_fp)):
        for v in VALUES:
            run_variant(fp, shape=shape, knob=KNOB, knob_value=v)


if __name__ == "__main__":
    main()
