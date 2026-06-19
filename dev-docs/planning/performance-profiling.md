# Plan: performance profiling pass

Status: **planning (scoping).** A measure-first pass to locate the pipeline's
real cost now that Stages 0–6 are functionally complete, producing a ranked lever
list that feeds two follow-ups: an optimization effort and the benchmark suite
([`perf-benchmarks.md`](perf-benchmarks.md)) that guards it. This document scopes
the *profiling*; it does not optimize.

## Why now

Core functionality is in place, so the shape of the work is stable enough to
profile honestly (the cost won't move out from under us mid-measurement). We have
named hot spots from development — **Stage 2b** (`calibrate_tau`), **Stage 5**
(the per-window fit), and **report image generation** — but only Stage 5 has been
profiled rigorously, and only at the *intra-window* level. The two things we have
never measured are (a) the **whole-pipeline** `run` attribution (where the wall
goes across all stages, not just inside one), and (b) the **cross-window**
parallelism opportunity. This pass fills both gaps before any optimization or
benchmark figure is committed.

## Two axes, kept separate

Profiling must report these independently, because a lever can help one and hurt
the other:

- **Efficiency** — total CPU-seconds / work done. Won by *algorithmic* change:
  trimming redundant recompute, caching reusable results, shrinking the
  per-iteration assembly. Helps every host equally.
- **Wall-time** — elapsed time to a finished `.ftmw`. Won by *parallelism* and
  *BLAS thread management*. Can *raise* total CPU (more cores busy) while cutting
  wall. Host- and core-count-dependent.

The headline metric for a user is wall-time; the headline metric for a shared
batch host is CPU-efficiency. We optimize for wall first, but never by spending
CPU we don't have to (oversubscription actively *hurts* both — see BLAS below).

## What we already know (do not re-litigate)

From [`stage5-nls-performance.md`](stage5-nls-performance.md), measured by
capture/replay on real 655/363 windows:

- Stage 5's per-window cost is **fixed per-`fit_window` overhead + O(M·K)
  line-shape assembly** (the `h_T` / Jacobian transcendental evals over the
  `(K, M)` grid), **not** the O(M·K²) solve (svd ≈ 6 %) and **not** the
  `fit_window` call count (the add-loop already self-compresses to ~8 calls/window
  at ~11 nfev each).
- The per-window Jacobians are **below OpenBLAS's threading threshold**, so the
  solve is single-threaded regardless of `OPENBLAS_NUM_THREADS`. The full-pipeline
  ~1300 % CPU comes from large-array ops *outside* the fit (the active-FT FFT,
  window materialization).
- Every **region-shrinking** lever (geometric-batch add-loop, frequency-window
  decomposition, Stage-4 small-window retune) failed on the **same** root cause:
  the AICc accept gate's `n_eff` is window-size-sensitive, so smaller/more windows
  systematically over-accept lines. Do not re-propose these as perf levers.
- The two shipped intra-window wins were `x_scale='jac'` (−17 % wall, fewer
  iterations of the fixed-cost assembly) and vectorizing the model/Jacobian over
  peaks (−8 %, byte-identical). The one remaining large *intra-window* lever named
  there is **local-support assembly** (evaluate each peak's `h_T` only over its
  support, O(M·K) → O(M·support)); it perturbs results (tail truncation) so it
  needs the metric gate.

The implication for this pass: the biggest *untried* lever is not inside a window
at all — it is **doing independent windows at the same time**, plus the same idea
for the two other embarrassingly-parallel hot spots (Stage 2b per-bin fits, report
per-window figures).

## Methodology

Profile the whole `run` and each hot spot in isolation, separating CPU from wall.

- **Fixtures (a deliberate spread, fresh-built per
  [[feedback-rebuild-fixtures-fresh]] discipline):** the dense Lorentzian
  worst-case (655), a dense Gaussian (363), a moderate fixture (2638), and the
  second-instrument fixture (succinimide / UXR) to catch grid-size effects (the
  out-of-band-floor lesson, D14). Report numbers per fixture; never average across
  the SNR/density span.
- **Tools, by question:**
  - *Where does the wall go across stages?* — a thin per-stage timer around the
    `run` sequence (import → FT → noise → τ → peaks → windows → fit → timebase →
    review → report), CPU-time and wall-time both, plus peak RSS. This is the
    missing whole-pipeline view.
  - *Where does CPU go inside a stage?* — `cProfile` + a caller/callee dump
    (`pstats`), as already done for Stage 5; attribute self-time to functions.
  - *Wall vs CPU divergence / native time / GIL?* — `py-spy` sampling
    (`record` + `dump`) on a live `run`, which sees native frames (NumPy, BLAS,
    matplotlib Agg) a Python profiler misses, and shows whether time is in Python,
    in BLAS, or blocked.
  - *Hot lines within a flagged function* — `line_profiler` on the few functions
    cProfile pins.
  - *Memory / leak under repeated exploration* — `tracemalloc` high-water marks
    (also the perf-benchmarks.md "no growth" requirement).
  - *BLAS oversubscription* — run each profile at `OMP_NUM_THREADS=1
    OPENBLAS_NUM_THREADS=1` vs unset, and watch the CPU% (a `>100 %` single-process
    run with no intended parallelism is the oversubscription tell). Reuse the
    existing `scratch/stage5-nls/` capture/replay harness for the fit.
- **Determinism:** fix any timestamp/seed inputs; the fit is already deterministic
  (byte-identical table on a fixture is the perf-acceptance bar
  [[stage5-perf-pass]]), so a perf change's correctness is checkable by the
  table fingerprint.
- **Artifacts land in gitignored `scratch/perf-profile/`** (never the working
  tree, per CLAUDE.md).

## Investigation areas (hypotheses to confirm or kill)

Ranked by expected upside × confidence. Each entry says what to measure and what
gate any eventual change must pass.

### A. Cross-window parallelism over the dependency DAG (the main wall lever)

`execute_plan` walks windows sequentially in `WindowPlan.topological_order`
(`fitting/plan_execution.py`). The order exists because a **`FixedContributor`**
needs its primary window fit first, and a **structural replan** can rewrite the
plan and force affected windows to refit. But the order is a *linearization of a
DAG*: windows with no dependency path between them are independent and could fit
concurrently.

- **The structure to exploit:** `plan.topological_order` + `plan.dependency_edges`
  already encode the DAG. Layer it (longest-path / antichain levelization): all
  windows at the same level have their dependencies satisfied by earlier levels
  and are mutually independent → fit a level concurrently, barrier between levels.
  This is exactly the "nodes at the same tree level run in parallel" structure.
- **Process pool, not threads:** the fit is CPU-bound Python+NumPy → the GIL caps
  thread parallelism; use `ProcessPoolExecutor`. Each window fit is largely
  self-contained (`fit_window` on its slice of the active FT + its materialized
  contributors), so the per-task payload is a window plan entry + the shared
  read-only active-FT/noise arrays (pass via shared memory / inherit on fork to
  avoid pickling the big arrays per task).
- **The two hard parts to measure, not assume:**
  1. **Replan interaction.** The structural-renegotiation loop runs *after* the
     initial walk and only on `replan_context`. Measure how often it fires per
     fixture and how many windows it touches — if rare and local (the
     ROADMAP/847-line evidence suggests it is), parallelize the initial walk and
     keep the replan rounds sequential (or re-parallelize only the affected
     antichain). Quantify the replan share of wall first.
  2. **BLAS oversubscription under a pool.** N worker processes × M BLAS threads
     each = N·M threads thrashing the cores. Per the prior finding the per-window
     solve is already single-threaded, so pinning `OPENBLAS_NUM_THREADS=1` in
     workers should be free correctness-wise and is *required* for the pool to
     scale. Measure speedup vs worker count at pinned vs unpinned BLAS.
- **Expected upside:** high — this is the only large untried wall lever, and a
  dense fixture has many independent windows (655 ≈ 10 min, mostly the sequential
  add-loop sweep). Linear-ish in cores up to the DAG's width, minus the replan
  tail and per-process overhead.
- **Gate:** **byte-identical fitted table** to the sequential run on all fixtures
  (the fit is deterministic and per-window independent given the order, so
  parallelizing an antichain must not change results). This is the cleanest
  possible correctness check.

### B. BLAS / thread oversubscription management (pipeline-wide)

Independent of the pool: the whole `run` shows ~1300 % CPU from the FFT and
array ops with no intended parallelism, which on a shared host is wasted
contention. Measure the wall cost of oversubscription directly (pinned vs unset
threads) and decide a default policy (pin BLAS to 1 and own the parallelism
explicitly at the window level, vs. leave BLAS free for the few large-array ops).
The interplay with area A is the real question — they must be decided together.

### C. Stage 2b parallelism + redundancy (`calibrate_tau`)

`calibrate_tau` runs a sliding-active-window STFT with a per-bin NLS shape fit —
the **~81k-NLS** floor that is the test-suite runtime driver
[[test-suite-runtime-budget]]. The per-bin fits are **embarrassingly parallel**
(no inter-bin dependency) → the same process-pool treatment as A, simpler (no DAG,
no replan). Also profile for redundant recompute across the exp/gauss/voigt twin
builds and the 3-way shape vote — the batched closed-form-log-seed + Gauss-Newton
solver is already ~150× faster than scipy, so confirm where the residual cost is
(assembly vs solve vs the STFT itself) before parallelizing.

- **Gate:** the recommended-shape vote and consumed τ_G must be unchanged across
  the 7 fixtures (the existing Stage 2b validation bar).

### D. Report image generation

The report renders per-window matplotlib panels (overview / re / im / mag / hist)
sequentially in the `_assemble_report_site` loop, then base64-embeds them. This is
the dominant report cost and is **per-window independent** → parallelize the figure
render (process pool; matplotlib Agg is picklable-output-friendly — render to PNG
bytes in workers, collect in the parent). Secondary levers to measure: figure DPI
/ panel count, the palette-quantization step, and whether the shared-overview
render dominates. The report is read-only over the record, so a parallel render
must produce byte-identical HTML (figures are deterministic at fixed DPI).

- **Gate:** byte-identical report HTML to the sequential render (the cross-interface
  report test already asserts byte-identity across interfaces — extend it to
  sequential-vs-parallel).

### E. Algorithmic efficiency (cross-cutting, lower confidence)

Profile-driven, only where cProfile/line_profiler points:

- **Caching reusable results:** contributor materializations (a primary's template
  evaluated into multiple dependents), per-window grids, repeated FFTs, the noise
  σ re-measurement paths (D14 trimmed-band estimate). Confirm each is actually
  recomputed before caching it.
- **Local-support assembly** (the one remaining intra-window Stage-5 lever): cap
  each peak's `h_T`/Jacobian evaluation to its support window. Perturbs results →
  metric-gated, not byte-identical; only pursue if A/C/D don't already buy enough.
- **Redundant-work trimming:** anything the whole-pipeline profile shows being done
  twice (e.g. an FT or noise estimate computed in a stage and again in a consumer).

## Deliverables and exit criteria

1. A **profiling report** in `dev-docs/research/performance-profiling/report.md`
   (reproducer + raw `pstats`/`py-spy` artifacts under `scratch/perf-profile/`):
   per-stage wall+CPU attribution on the fixture spread, the oversubscription
   measurement, and the replan-frequency / DAG-width characterization that decides
   whether A is worth it.
2. A **ranked lever list**: each lever with measured/estimated upside (wall and
   CPU separately), implementation risk, and the correctness gate it must pass.
3. That list feeds two separate efforts (out of scope here): the **optimization
   work** and the **benchmark suite** ([`perf-benchmarks.md`](perf-benchmarks.md))
   that becomes the regression guard once optimizations land.

## Non-goals

- **Doing the optimizations.** This pass measures and ranks; implementation is a
  separate, gated effort per lever.
- **The benchmark suite itself.** `perf-benchmarks.md` owns the committed
  `tests/performance/` numbers; this pass tells it what to measure.
- **Re-running the falsified intra-window levers** (batch add-loop, window
  decomposition, small-window retune) — closed in `stage5-nls-performance.md`.
