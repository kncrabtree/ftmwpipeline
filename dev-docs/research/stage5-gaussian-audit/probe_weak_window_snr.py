"""Probe Stage 5 ``conservative.weak_window_snr_threshold`` against 2638.

The knob is the in-window SNR floor below which the conservative loop
runs in *weak-window* mode (forces free-τ off + a tighter F-test gate;
default 10.0). Lowering it lets the free-τ path fire on more borderline
windows; raising it forces more windows into the conservative weak-mode
shell.

Grid is bracketing logarithmically. The probe re-uses the cached
``baseline__<shape>`` fits when the value hits the default.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from harness import prepare_fixture, run_variant  # noqa: E402

KNOB = "conservative.weak_window_snr_threshold"
VALUES = [5.0, 10.0, 15.0, 20.0]  # default = 10.0

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("probe-weak-window-snr")


def main() -> None:
    lor_fp = prepare_fixture("lorentzian")
    gauss_fp = prepare_fixture("gaussian")
    for shape, fp in (("lorentzian", lor_fp), ("gaussian", gauss_fp)):
        for v in VALUES:
            run_variant(fp, shape=shape, knob=KNOB, knob_value=v)


if __name__ == "__main__":
    main()
