"""Storage-size benchmarks for the ``.ftmw`` lightweight-file invariant.

``SERIALIZATION_STRATEGY.md`` makes a storage-size figure normative only when a
benchmark measures it, and prohibits persisting ComplexFT or any large array
recomputable in interactive time. These tests pin the measured file size at the
import and Stage-2 boundaries (so the figures may be cited) and guard the
lightweight invariant directly: the file must stay well under its raw source,
which a ComplexFT-persist regression would not.

Reference figures measured on ``examples/blackchirp_data/2638`` (deterministic
for fixed input data). The tolerance band is wide enough to absorb a benign
serialization tweak (a new small attribute) yet far tighter than the multi-MB
jump that persisting a recomputable spectrum would cause.
"""

from __future__ import annotations

import pytest

from .conftest import BuiltPipeline

pytestmark = [pytest.mark.performance, pytest.mark.slow]

# Measured on 2638 (bytes). See the suite docstring; deterministic per input.
REFERENCE_SIZE_AFTER_IMPORT = 4_869_302
REFERENCE_SIZE_AFTER_STAGE2 = 5_922_802
SIZE_TOLERANCE = 0.25  # +/-25%: absorbs benign churn, trips on a persisted array


def _within(value: int, reference: int, frac: float) -> bool:
    return abs(value - reference) <= frac * reference


def test_size_after_import_is_lightweight(built_pipeline: BuiltPipeline) -> None:
    size = built_pipeline.size_after_import
    record = f"{size:,} bytes ({size / 1e6:.2f} MB)"
    print(f"\n.ftmw size after import: {record}")

    # The import stores the FID once; the file must stay well under its raw
    # multi-file source, the lightweight-file invariant.
    assert size < built_pipeline.raw_source_bytes, (
        f"imported file {record} is not smaller than its raw source "
        f"({built_pipeline.raw_source_bytes:,} bytes) -- something large was "
        "persisted"
    )
    assert _within(size, REFERENCE_SIZE_AFTER_IMPORT, SIZE_TOLERANCE), (
        f"size after import {record} drifted >{SIZE_TOLERANCE:.0%} from the "
        f"reference {REFERENCE_SIZE_AFTER_IMPORT:,} bytes"
    )


def test_size_after_stage2_stays_bounded(built_pipeline: BuiltPipeline) -> None:
    size = built_pipeline.size_after_stage2
    delta = size - built_pipeline.size_after_import
    record = f"{size:,} bytes ({size / 1e6:.2f} MB), Stage-2 delta {delta:,} bytes"
    print(f"\n.ftmw size after Stage 2: {record}")

    # Stage 2 persists its noise authority, not the (recomputable) ComplexFT;
    # the file stays under its raw source.
    assert size < built_pipeline.raw_source_bytes, (
        f"file after Stage 2 {record} is not smaller than its raw source "
        f"({built_pipeline.raw_source_bytes:,} bytes)"
    )
    assert _within(size, REFERENCE_SIZE_AFTER_STAGE2, SIZE_TOLERANCE), (
        f"size after Stage 2 {record} drifted >{SIZE_TOLERANCE:.0%} from the "
        f"reference {REFERENCE_SIZE_AFTER_STAGE2:,} bytes"
    )
