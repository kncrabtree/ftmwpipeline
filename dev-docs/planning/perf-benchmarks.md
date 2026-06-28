# Plan: Performance benchmarks

Status: **implemented**. The `tests/performance/` regression-guard suite is in
place (resolves ROADMAP divergence D5).

## Why this exists

`SERIALIZATION_STRATEGY.md` and `TESTING_STRATEGY.md` require that any
storage-size or timing figure stated as a requirement be backed by a
measurement, and that a storage or timing claim is normative only if a test
measures it. The suite makes those figures exist and guards against regressing
the optimization work that has since shipped (the profiling pass and its levers
are archived in `COMPLETED.md`).

## What it asserts, and what it does not

Wall-clock flakes with load and core count, so the suite's *guards* are
deterministic — on-disk file size and operation counts that encode algorithmic
complexity — while wall-clock is recorded for the record but never gates. This
follows the stance already set by `test_parallel_fit.py`, which guards the
parallel fit by byte-identity rather than by a speedup assertion.

The suite (all marked `performance` + `slow`, opt-in via `-m performance`),
built once per session off `examples/blackchirp_data/2638` on a trimmed,
dependency-free window plan:

1. **Storage size** (`test_storage_size.py`) — pins the `.ftmw` size after
   import and after Stage 2 against measured references (±25%), and asserts the
   file stays smaller than its raw multi-file source. Guards the lightweight-file
   invariant: a ComplexFT-persist (or any large recomputable array) regression
   trips it.
2. **Report reload complexity** (`test_report_reload_complexity.py`) — counts
   the per-window peak-column loader across one `report_run` and asserts the
   implied number of full fit deserializations stays a small constant (≤4;
   measured 2). Guards the profiling-pass win that removed the O(N²) per-window
   reload (which fired the loader ~N² times).
3. **Memory stability** (`test_memory_stability.py`) — `tracemalloc` heap growth
   across repeated `compute_ft` parameter exploration stays under a generous cap
   (measured: a few dozen bytes). Guards the "exploring parameters never leaks /
   never re-imports" invariant.
4. **`compute_ft` timing** (`test_compute_ft_timing.py`) — recorded
   (`record_property` + stdout), not asserted; the only bound is a hang
   tripwire, not a performance gate.
5. **Parallel fit byte-identity** (`test_parallel_fit.py`, pre-existing) — a
   parallel fit equals the sequential fit window-for-window.

## Reference figures (2638, deterministic per input)

- Size after import: 4,869,302 bytes (~4.87 MB); raw source ~15.9 MB.
- Size after Stage 2: 5,922,802 bytes (~5.92 MB).
- Report fit deserializations per render: 2.

## Not in scope

Optimization work. This suite is measurement and regression-guarding only;
tuning is tracked separately (the completed profiling/optimization plans in
`COMPLETED.md`).
