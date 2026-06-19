# Performance profiling — measurement report

The measure-first deliverable of the
[`planning/performance-profiling.md`](../../planning/performance-profiling.md)
pass. Locates where `run`'s wall- and CPU-time go now that Stages 0–6 are
functionally complete, and feeds the ranked levers in
[`planning/performance-optimization.md`](../../planning/performance-optimization.md).

## Method

Harness `scratch/perf-profile/profile_run.py` mirrors `run_impl`'s stage sequence
(import → start → FT → noise → tau → peaks → windows → fit → timebase → review →
report) by calling the `Pipeline` methods directly, timing each stage's wall
(`perf_counter`) + CPU (`process_time`) + peak RSS, wrapping the whole run in
`cProfile`, and characterizing the window-dependency DAG (antichain levelization)
and replan history. Fixtures fresh-built from `examples/blackchirp_data/`. BLAS
pinned to 1 thread (`OPENBLAS/OMP/MKL/NUMEXPR_NUM_THREADS=1`) for clean CPU
attribution, with one unpinned 2638 run for the oversubscription delta. Runs are
sequential (isolated wall, no cross-fixture contention). Raw artifacts per fixture
in `scratch/perf-profile/<name>/` (`*_summary.json`, `*.prof`, `*_pstats.txt`).

The second-instrument fixture (succinimide/UXR) is deferred — the three Blackchirp
fixtures already span the moderate → dense-Gaussian → extreme-SNR-Lorentzian range
and isolate the cost structure; the dense ones ran `--no-report` (see Finding 1).

## Finding 1 — report generation dominates, via an O(N²) reload (not matplotlib)

2638 whole-pipeline `run`, pinned, total wall **3015 s (50 min)**:

| stage | wall (s) | % |
|---|--:|--:|
| **report** | **2730** | **90.5** |
| fit | 256 | 8.5 |
| start | 4.4 | |
| review | 4.3 | |
| windows | 2.9 | |
| peaks | 1.8 | |
| tau | 1.3 | |
| import | 1.1 | |
| noise | 0.4 | |
| ft | 0.04 | |

Peak RSS 1479 MB, almost all in the report. cProfile self-time is dominated by
**h5py deserialization**, not figure drawing: `fitting_serialization._load_peak_columns`
(the per-window-fit peak-column dictcomp) is called **77,560 ≈ 277²** times on a
277-window fit. Root cause: the report loop loads the detail `bundle` once but
calls `get_candidate_ledger_impl(path, wid)` per window, and that impl reloads the
**entire** `load_spectrum_fit_from_hdf5` (all window-fits) **and** the 750k-point
raw FID (`load_fid_from_pipeline_impl`) on every call → **O(N²) HDF5 reads +
O(N) raw-FID reloads**. A second waste: `plot_window_panels` builds a full-spectrum
*overview* figure per window that the report immediately discards (it uses the
shared interactive overview). Both are pure redundant work; fixes in the
optimization plan (1a, 1b). The dense fixtures ran `--no-report` because measuring
this O(N²) cost at 600–1200 windows would take hours and only re-confirm the bug —
the report is re-measured after 1a.

## Finding 2 — the fit DAG is wide, shallow, and *more* parallel as the fit gets costlier

| fixture | windows | dep-edges | levels | speedup ceiling | replans | fit wall |
|---|--:|--:|--:|--:|--:|--:|
| 2638 (moderate) | 375 | 73 | 3 | 125× | 0 | 256 s |
| 363 (dense Gaussian) | 486 | 62 | 2 | 243× | 0 | 779 s |
| 655 (extreme-SNR) | 1241 | **0** | **1** | **1241×** | 0 | **3496 s (58 min)** |

The window dependency graph widens and flattens as density/SNR rises, exactly as
the fit gets more expensive — so cross-window parallelism pays off most where it's
needed most, and the worst-case-cost fixture is the best case for parallelism.
**Zero replan events on all three.**

### Why 655 has zero edges despite the *most* leakage coupling

655 carries **8372 fixed contributors** (~6.7 per window; every window coupled —
the dense pedestal is fully present), but **all 8372 are `edge_free`**:

| fixture | fixed contributors | edge_free | edged |
|---|--:|--:|--:|
| 2638 | 395 | 265 | 130 |
| 363 | 1675 | 1556 | 119 |
| 655 | 8372 | 8372 | 0 |

An **edged** contributor reads the line's *converged fit* (amplitude, phase,
refined frequency) from its primary window (`_materialize_contributor`) — the
accurate leakage estimate — at the cost of a dependency edge (ordering). An
**edge_free** contributor re-derives those self-contained via an LSQ of the line
template over the shared active FT (`evaluate_edge_free_contributors`) — cruder
("global-crude"; using it everywhere regressed the bulk fit, 655 2.40→4.71) but
imposing no ordering. The cycle-breaker **prefers edges** and falls back to
edge_free only when an edge would close a dependency cycle (and only for the top-N
dominant orphans per window; the rest are dropped). At 655's density the leakage
graph is fully cyclic, so every edge is dropped and the dominant orphans become
edge_free, with the per-window leakage-wing baseline carrying the diffuse residual
pedestal. So 655 is fully parallel not because it is less coupled but because at
that density it is forced to trade leakage-subtraction *accuracy* (edged →
edge_free) for an acyclic plan — the issue-#3 design, 7-fixture-validated.

Implication for parallelism: antichain levelization preserves the edged accuracy
(a dependent still waits for its primary's real fit) and runs only the genuinely
independent windows concurrently, so **parallelism costs nothing in fit quality**.

## Finding 3 — BLAS oversubscription is net-negative

2638 fit, pinned vs unpinned:

| | wall (s) | cpu (s) | cpu/wall |
|---|--:|--:|--:|
| pinned (1 thread) | 256.5 | 256.4 | 1.00 |
| unpinned (default) | 261.0 | 337.3 | **1.29** |

Unpinned, the fit pulls ~1.3 cores but wall is *worse* (261 vs 256 s) — ~30 %
extra CPU for negative wall benefit, confirming the prior finding that the
per-window solve is below OpenBLAS's threading threshold. Policy: **pin BLAS to 1
and own parallelism at the window level.** Every other stage is `cpu/wall ≈ 1.0`
(single-threaded) under pinning.

## Conclusions → levers

1. **Report O(N²) reload** — the dominant cost; efficiency, low risk,
   byte-identical-HTML gate. (Plan 1a.)
2. **Report discarded overview** — trivial efficiency. (Plan 1b.)
3. **Report figure-render pool** — the wall lever on the post-1a/1b matplotlib
   floor. (Plan 1c.)
4. **Cross-window fit parallelism** — antichain levelization → process pool, BLAS
   pinned per worker, replan sequential; ceilings 125×/243×/1241×, leaf-heavy,
   0 replans; byte-identical-table gate. (Plan item 2.)
5. **BLAS policy** — pin to 1; never rely on BLAS threads for the fit.

Sub-20 s stages (import/ft/noise/tau/peaks/windows/review) are not worth
optimizing. Intra-window Stage 5 levers are already closed
([`planning/stage5-nls-performance.md`](../../planning/stage5-nls-performance.md)).
