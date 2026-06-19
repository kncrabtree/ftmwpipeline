# Plan: performance optimization

Status: **in progress.** Report-efficiency fixes are scoped and ready (work item
1); Stage 5 cross-window parallelism is a tentative design pending the 655
worst-case DAG (work item 2). Driven by the measure-first
[`performance-profiling.md`](performance-profiling.md) pass; the benchmark suite
([`perf-benchmarks.md`](perf-benchmarks.md)) becomes the regression guard once
these land. Raw artifacts: `scratch/perf-profile/` (`*_summary.json`, `*.prof`,
`*_pstats.txt`, `FINDINGS.md`).

## What the profiling showed

Whole-pipeline `run`, BLAS pinned to 1 thread, per-stage wall:

| stage | 2638 (moderate) | 363 (dense Gaussian) | 655 (extreme-SNR) |
|---|--:|--:|--:|
| report | **2730 s (90.5 %)** | (skipped — see below) | (skipped) |
| fit | 256 s | 779 s | *pending* |
| all other stages | ~16 s total | ~20 s total | *pending* |
| **total (with report)** | **3015 s** | — | — |

Two surprises versus the plan's priors:

1. **Report generation dominates, ~10× the fit** on the moderate fixture — not
   Stage 5. cProfile pins the cost to **h5py deserialization, not matplotlib**:
   `fitting_serialization._load_peak_columns` runs **77,560 ≈ 277²** times on a
   277-window fit. The dense fixtures ran with `--no-report` on purpose — the
   report cost is an O(N²) reload bug, and re-confirming it at 600+ windows would
   cost hours; we re-measure the report after the fix.
2. **The window-dependency DAG is wide and shallow** — 2638: 375 windows, 3
   levels, **125× parallel-speedup ceiling**, 0 replans; 363: 486 windows, 2
   levels, **243×**, 0 replans. (655 pending — the realistic worst case for
   fixed-contributor depth.) `cpu/wall ≈ 1.0` everywhere under pinned BLAS,
   confirming each stage is single-threaded; the unpinned oversubscription delta
   is pending.

## Work item 1 — report efficiency (ready; do first)

Highest value, lowest risk, no parallelism. Every change is gated on
**byte-identical report HTML** to the current output.

### 1a. Eliminate the O(N²) per-window reload (the dominant win)

`_assemble_report_site`'s per-window loop calls `get_candidate_ledger_impl(path,
wid)` once per window (`report_html_impl.py`), and that impl reloads the **entire**
Stage 5 fit (`load_spectrum_fit_from_hdf5`, all window-fits) **and** the
750k-point raw FID (`load_fid_from_pipeline_impl`) on **every** call. The detail
`bundle` is already loaded once (`report_html_impl.py:2704`) and carries
`bundle.fit` + `bundle.sideband`; `acquisition_us` comes from
`spectrum_fit.parameters`.

- Add `*, spectrum_fit: SpectrumFit | None = None, sideband=None` to
  `get_candidate_ledger_impl`; when both are supplied, skip BOTH HDF5 loads and
  derive the ledger from them directly. (`render_fit_panels_impl(..., bundle=)`
  already follows this pattern — mirror it.)
- The report loop passes `spectrum_fit=bundle.fit, sideband=bundle.sideband`.
- Dual-interface-safe: the standalone `(path, wid)` call still self-loads, so the
  CLI / Pipeline / api ledger verbs are unchanged.
- **Test:** the bundle-passed ledger equals the self-loading ledger on a built
  fixture; **gate:** byte-identical report HTML (extend the existing report
  golden / cross-interface byte-identity check).

### 1b. Stop building the per-window overview that is discarded

`plot_window_panels` (`visualization/fit_detail.py:783`) builds
`figures["overview"]` — a full-spectrum `draw_overview` + model resynthesis over
the whole band — for **every** window, and the report immediately `plt.close()`s
it (`report_html_impl.py:2816`; it uses the *shared* interactive overview). ~N
full-spectrum figures built and thrown away.

- Add `include_overview: bool = True` to `plot_window_panels`; thread a
  `with_overview=False` through `render_fit_panels_impl` from the report caller.
- Output byte-identical by construction (the overview was discarded anyway).

### 1c. Parallelize per-window figure rendering (the wall lever, after 1a/1b)

Once 1a/1b remove the redundant work, the remaining report floor is matplotlib:
~4 real panels/window (re/im/mag/hist), the weakref/font-cache churn cProfile
showed, and the per-panel `h_T` model resynthesis. Per-window figure rendering is
**embarrassingly parallel** (each window's panels are independent).

- Render each window's panels in a `ProcessPoolExecutor` worker to PNG bytes
  (matplotlib Agg), collect bytes in the parent for the single-file embed.
- Pin BLAS per worker; cap workers to cores−2.
- **Gate:** byte-identical HTML (figures are deterministic at fixed DPI). Defer
  the on-plot-geometry capture (`_mag_axes_geometry`) correctly — it must run in
  the worker that drew the figure and travel back with the bytes.

Expected: 1a is the bulk of the 2730 s; 1b removes ~1 figure/window of resynthesis;
1c then cuts the matplotlib floor by ~core count. Re-profile the report after each.

## Work item 2 — Stage 5 cross-window parallelism (tentative; pending 655)

The fit is 8.5 %–? of wall (256 s on 2638, 779 s on 363, 655 pending) and is the
dominant cost once the report is fixed, especially on dense/extreme fixtures.
`execute_plan` (`fitting/plan_execution.py`) walks windows sequentially in
`WindowPlan.topological_order`; the order exists only because a `FixedContributor`
needs its primary window fit first and a structural replan can rewrite the plan.
That order is a **linearization of a DAG** — independent windows can fit
concurrently.

**Tentative design (to confirm against the 655 worst-case DAG):**

- **Antichain levelization.** Layer the DAG from `topological_order` +
  `dependency_edges` (the harness's `_levelize` already computes levels / widths /
  ceiling). Fit each level concurrently, barrier between levels: every window at
  level *k* has all its fixed-contributor primaries fit by levels `< k`.
- **Process pool, not threads** (CPU-bound Python+NumPy → GIL). Each task is one
  window's `conservative_fit` on its active-FT slice + its materialized
  contributors. Pass the large read-only arrays (active FT, noise) via
  fork-inheritance / shared memory, not per-task pickling; pass each window's
  fixed-contributor *materializations* (which depend on earlier-level fit results)
  explicitly into the task.
- **BLAS pinned to 1 per worker** — free per the profiling finding (the per-window
  solve is already single-threaded) and required so N workers don't each spawn M
  BLAS threads.
- **Replan stays sequential.** 0 replan events on 2638/363; if 655 agrees,
  parallelize the initial walk and run the (rare) structural-renegotiation rounds
  sequentially, re-parallelizing only the affected antichain. Quantify the replan
  share on 655 before committing.
- **Correctness gate: byte-identical fitted table** to the sequential run. An
  antichain has no inter-window dependency, so concurrent order cannot change
  results — the cleanest possible gate. Caveat: bit-reproducibility needs BLAS
  threads pinned identically in the sequential baseline and the parallel run
  (decide whether to pin globally in production or only assert under a pinned
  test).

**Open questions for the 655 data:**

- Does the extreme-SNR fixture have a **deeper/narrower** DAG (more bright-line
  fixed contributors → more edges)? That sets the realistic speedup ceiling. The
  user's hypothesis is that 655 limits parallelism more than 2638/363.
- How often does the **replan** fire on 655, and how many windows does it touch?
- Process-pool overhead vs per-window fit time — do small windows amortize the
  fork/IPC, or should tasks be chunked by level?

## Sequencing and gates

1. **Report 1a** (O(N²) reload) — biggest single win, lowest risk.
2. **Report 1b** (discarded overview) — trivial, byte-identical.
3. **Re-profile the report** (all fixtures, with report) to establish the new floor.
4. **Report 1c** (figure-render pool) — wall lever on the new floor.
5. **Stage 5 parallelism** — after the 655 DAG/replan data confirms the ceiling.
6. **Benchmark suite** ([`perf-benchmarks.md`](perf-benchmarks.md)) captures the
   before/after as the committed regression guard.

Every step gated on byte-identical output (HTML or fitted table); the fit is
deterministic, so the gate is exact, not statistical.

## Out of scope / already settled

- **Intra-window Stage 5 levers** (batch add-loop, window decomposition,
  small-window retune) — explored and closed in
  [`stage5-nls-performance.md`](stage5-nls-performance.md); the per-window cost is
  assembly-bound and the region-shrinking levers all over-add via the
  window-size-sensitive accept gate. The remaining intra-window lever
  (local-support assembly) is lower priority than cross-window parallelism.
- **Optimizing the sub-20 s stages** (import/ft/noise/tau/peaks/windows/review) —
  not worth it against report and fit.
