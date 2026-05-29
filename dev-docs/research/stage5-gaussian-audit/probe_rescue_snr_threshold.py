"""Probe Stage 5 ``rescue.snr_threshold`` against 2638 fit_peaks chi^2.

The knob is the residual-rescue nomination SNR floor (default 2.5).
Lowering it widens the rescue's net (more residual candidates promoted
to free peaks); raising it tightens it.

Grid brackets the default; the lower edge (1.5) is the regime most
likely to surface *missed_peak* fixes, the upper edge (3.5) the regime
most likely to suppress noise-promoting *overfit* cases. The Step 5
cross-reference uses these endpoints when classifying flagged windows.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from harness import prepare_fixture, run_variant  # noqa: E402

KNOB = "rescue.snr_threshold"
VALUES = [1.5, 2.0, 2.5, 3.5]  # default = 2.5

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("probe-rescue-snr-threshold")


def main() -> None:
    lor_fp = prepare_fixture("lorentzian")
    gauss_fp = prepare_fixture("gaussian")
    for shape, fp in (("lorentzian", lor_fp), ("gaussian", gauss_fp)):
        for v in VALUES:
            run_variant(fp, shape=shape, knob=KNOB, knob_value=v)


if __name__ == "__main__":
    main()
