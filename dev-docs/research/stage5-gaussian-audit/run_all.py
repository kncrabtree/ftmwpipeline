"""Run all five Stage 5 knob probes in one go.

Wall-clock budget: 5 knobs × 4 values × 2 shapes = 40 variants, minus
the cached baselines (8 reused across knobs) → ~32 fits × ~2 min each
≈ 60-80 min on the dev workstation.

Resets the per-window + variants_summary CSVs at start, prepares both
shape fixtures (~4 min cached), then sweeps each probe in order. The
baseline (default-everywhere) fit per shape runs the first time any
probe asks for its hard-default value and is re-used by every other
probe via :func:`harness.run_variant`'s baseline-reuse path.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from harness import prepare_fixture, reset_csvs  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("run-all")


def main() -> None:
    t0 = time.perf_counter()
    reset_csvs()
    prepare_fixture("lorentzian")
    prepare_fixture("gaussian")

    # Import the probe modules in a fixed order. Each probe re-uses the
    # cached fixtures; only the per-variant fit_peaks fires here.
    import probe_fit_tau_min_snr  # noqa: F401
    import probe_weak_window_snr  # noqa: F401
    import probe_rescue_snr_threshold  # noqa: F401
    import probe_rescue_prominence_threshold  # noqa: F401
    import probe_residual_edge_threshold  # noqa: F401

    for label, mod in (
        ("fit_tau_min_snr", probe_fit_tau_min_snr),
        ("weak_window_snr_threshold", probe_weak_window_snr),
        ("rescue.snr_threshold", probe_rescue_snr_threshold),
        ("rescue.prominence_threshold", probe_rescue_prominence_threshold),
        ("thaw.residual_edge_threshold", probe_residual_edge_threshold),
    ):
        logger.info("=" * 72)
        logger.info("Probe: %s", label)
        logger.info("=" * 72)
        mod.main()

    dt = time.perf_counter() - t0
    logger.info("All probes complete in %.1f min", dt / 60.0)


if __name__ == "__main__":
    main()
