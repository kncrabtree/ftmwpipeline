"""Repeated parameter exploration does not leak.

``SERIALIZATION_STRATEGY.md``: exploring FT parameters never requires
re-importing, and the lightweight-file design assumes recomputable arrays are
transient. So re-running a stage many times with different settings on one file
must not accumulate memory -- a leak would retain the (large) per-call FID/FT
arrays or HDF5 handles.

Measured with ``tracemalloc`` (Python-heap allocation, more deterministic than
RSS) across repeated ``compute_ft`` calls with alternating trim bands. After a
warmup that absorbs one-time import caches, additional iterations must not grow
the traced heap beyond a generous cap -- a per-iteration leak would scale with
the iteration count and blow past it.
"""

from __future__ import annotations

import gc
import tracemalloc
from pathlib import Path

import pytest

import ftmwpipeline.api as ftmw

pytestmark = [pytest.mark.performance, pytest.mark.slow]

# Alternating bands so each call genuinely recomputes the FT rather than hitting
# an identical-settings short-circuit.
TRIM_BANDS = [(26500, 40000), (27000, 39500)]
WARMUP_ITERS = 3
MEASURED_ITERS = 8
# Generous: a leak that retained even one per-call active-FT array would add
# megabytes per iteration and clear this many times over.
MAX_HEAP_GROWTH_BYTES = 8_000_000


def _explore(fp: Path, n: int) -> None:
    for i in range(n):
        ftmw.compute_ft(fp, trim=TRIM_BANDS[i % len(TRIM_BANDS)])


def test_repeated_ft_exploration_does_not_grow(imported_file: Path) -> None:
    tracemalloc.start()
    try:
        _explore(imported_file, WARMUP_ITERS)
        gc.collect()
        baseline, _ = tracemalloc.get_traced_memory()

        _explore(imported_file, MEASURED_ITERS)
        gc.collect()
        after, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    growth = after - baseline
    print(
        f"\nheap growth over {MEASURED_ITERS} compute_ft iterations: "
        f"{growth:,} bytes (baseline {baseline:,} -> {after:,})"
    )
    assert growth < MAX_HEAP_GROWTH_BYTES, (
        f"repeated parameter exploration grew the traced heap by {growth:,} "
        f"bytes over {MEASURED_ITERS} iterations (cap {MAX_HEAP_GROWTH_BYTES:,}) "
        "-- a per-call array or handle is being retained"
    )
