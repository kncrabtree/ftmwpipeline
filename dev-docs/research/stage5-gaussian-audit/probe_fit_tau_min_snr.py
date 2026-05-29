"""Probe Stage 5 ``tau.fit_tau_min_snr`` against the 2638 fit_peaks chi^2.

The knob is the per-window SNR floor above which ``fit_tau`` is allowed
to be True. Below the floor, the conservative loop holds tau at the
Stage 2b anchor (no per-window τ free parameter). Default = 50.0.

The grid brackets the default at log-ish spacing. The non-default values
make τ free on more (lower) / fewer (higher) windows; both directions
exercise the χ²ᵣ response to τ-mobility under shape/blend stress.

Outputs land in the shared per-window CSV / summary CSV; this script is
a thin caller into ``harness.run_variant``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from harness import prepare_fixture, run_variant  # noqa: E402

KNOB = "tau.fit_tau_min_snr"
VALUES = [25.0, 50.0, 75.0, 100.0]  # default = 50.0

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("probe-fit-tau-min-snr")


def main() -> None:
    lor_fp = prepare_fixture("lorentzian")
    gauss_fp = prepare_fixture("gaussian")
    for shape, fp in (("lorentzian", lor_fp), ("gaussian", gauss_fp)):
        for v in VALUES:
            run_variant(fp, shape=shape, knob=KNOB, knob_value=v)


if __name__ == "__main__":
    main()
