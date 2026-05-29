"""Probe Stage 5 ``thaw.residual_edge_threshold`` against 2638 chi^2.

The knob is the S_coh-based residual-edge threshold that triggers
local-thaw / structural-replan attempts (default 8.0). Lower → more
replans fired (potentially more *replan_needed* fixes); higher → fewer
replans (residual edge noise ignored).

The replan loop is gated by ``thaw.max_replan_rounds=2`` so the cost
per fit stays bounded; this probe just changes which residual edges
make the gate.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from harness import prepare_fixture, run_variant  # noqa: E402

KNOB = "thaw.residual_edge_threshold"
VALUES = [5.0, 8.0, 10.0, 12.0]  # default = 8.0

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("probe-residual-edge-threshold")


def main() -> None:
    lor_fp = prepare_fixture("lorentzian")
    gauss_fp = prepare_fixture("gaussian")
    for shape, fp in (("lorentzian", lor_fp), ("gaussian", gauss_fp)):
        for v in VALUES:
            run_variant(fp, shape=shape, knob=KNOB, knob_value=v)


if __name__ == "__main__":
    main()
