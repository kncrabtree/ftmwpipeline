"""Informational wall-clock for an on-demand ``compute_ft``.

Recorded, not asserted. Wall-clock flakes with load and core count, so the
suite's regression guards are deterministic (storage size, reload counts); this
test exists only to keep the ``compute_ft`` timing figure honest and citable per
``TESTING_STRATEGY.md`` (a timing claim is normative only if a test measures it).
The single bound here is a catastrophe tripwire -- it catches a hang, not a
performance regression.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pytest

import ftmwpipeline.api as ftmw

pytestmark = [pytest.mark.performance, pytest.mark.slow]

ACTIVE_BAND = (26500, 40000)
# Not a performance gate: compute_ft is ~0.04 s on the reference machine; this
# only fires if the call hangs.
HANG_TRIPWIRE_SECONDS = 30.0


def test_compute_ft_timing_recorded(imported_file: Path, record_property) -> None:
    start = perf_counter()
    ftmw.compute_ft(imported_file, trim=ACTIVE_BAND)
    elapsed = perf_counter() - start

    record_property("compute_ft_seconds", elapsed)
    print(f"\ncompute_ft wall-clock: {elapsed:.4f} s (informational)")

    assert elapsed < HANG_TRIPWIRE_SECONDS, (
        f"compute_ft took {elapsed:.2f} s -- a hang, not a slowdown "
        f"(tripwire {HANG_TRIPWIRE_SECONDS:.0f} s)"
    )
