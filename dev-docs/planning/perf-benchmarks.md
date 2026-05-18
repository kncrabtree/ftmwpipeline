# Plan: Performance benchmarks

Status: **deferred** (ROADMAP divergence D5). Tracked, not started.

## Why this exists

`SERIALIZATION_STRATEGY.md` and `TESTING_STRATEGY.md` require that any
storage-size or timing figure stated as a requirement be backed by a
measurement. Today no such measurements exist and `tests/performance/` is
empty, so all such figures are currently non-normative.

## Scope when undertaken

A small `tests/performance/` suite (marked `performance`) measuring:

1. `.ftmw` file size after Stage 0 import and after Stage 2, on the
   `examples/blackchirp_data/2638` dataset.
2. Wall-clock time of an on-demand `compute_ft` for a representative FID.
3. Memory behavior of repeated parameter exploration (no growth).

These produce concrete numbers. Only then may the specs cite them as
requirements (with the benchmark as the reference).

## Not in scope

Optimization work. This plan is measurement only; tuning is separate and
follows from what the benchmarks show.
