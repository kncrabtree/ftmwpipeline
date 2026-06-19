# Plan: performance optimization

Status: **ready for implementation handoff.** The profiling pass is complete
across the fixture spread (2638 / 363 / 655, pinned + an unpinned delta); both
work items are scoped with measured upside and correctness gates. Driven by the
measure-first [`performance-profiling.md`](performance-profiling.md) pass; the
benchmark suite ([`perf-benchmarks.md`](perf-benchmarks.md)) becomes the
regression guard once these land. Raw artifacts:
[`research/performance-profiling/report.md`](../research/performance-profiling/report.md)
and `scratch/perf-profile/` (`*_summary.json`, `*.prof`, `*_pstats.txt`).

**Implementation order for the fresh session:** 1a (O(N²) report reload) → 1b
(discarded overview) → re-profile report → 1c (figure-render pool) → 2
(cross-window fit parallelism). 1a is the single biggest win and the cleanest
change.

## What the profiling showed

Whole-pipeline `run`, BLAS pinned to 1 thread, per-stage wall (dense fixtures ran
`--no-report` — see below):

| stage | 2638 (moderate) | 363 (dense Gaussian) | 655 (extreme-SNR) |
|---|--:|--:|--:|
| report | **2730 s (90.5 %)** | (skipped) | (skipped) |
| fit | 256 s | 779 s | **3496 s (58 min)** |
| all other stages | ~16 s | ~20 s | ~46 s |

Three findings versus the plan's priors:

1. **Report generation dominates, ~10× the fit** on the moderate fixture — not
   Stage 5. cProfile pins the cost to **h5py deserialization, not matplotlib**:
   `fitting_serialization._load_peak_columns` runs **77,560 ≈ 277²** times on a
   277-window fit. The dense fixtures ran with `--no-report` on purpose — the
   report cost is an O(N²) reload bug, and re-confirming it at 600–1200 windows
   would cost hours; re-measure the report after 1a.
2. **The window-dependency DAG is wide, shallow, and gets *more* parallel as the
   fit gets more expensive** — the opposite of the worst-case worry:

   | fixture | windows | dep-edges | levels | speedup ceiling | replans | fit wall |
   |---|--:|--:|--:|--:|--:|--:|
   | 2638 | 375 | 73 | 3 | 125× | 0 | 256 s |
   | 363 | 486 | 62 | 2 | 243× | 0 | 779 s |
   | 655 | 1241 | **0** | **1** | **1241×** | 0 | 3496 s |

   **Why 655 has 0 edges despite the most leakage coupling** (the open question,
   resolved): 655 carries **8372 fixed contributors** (~6.7 per window, every
   window coupled — the dense pedestal is absolutely present), but **all 8372 are
   `edge_free`**. At that density the leakage graph is fully cyclic, so the
   cycle-breaker drops every dependency edge and converts the dominant orphans to
   `edge_free` contributors (2638: 265/395 edge_free, 130 edged; 363: 1556/1675,
   119 edged; 655: 8372/8372, 0 edged). An `edge_free` contributor is materialized
   *self-contained* at fit time — a joint LSQ of the line template over the
   **shared read-only active FT** at the window's τ
   (`evaluate_edge_free_contributors`), **not** from the primary window's fitted
   result — so it imposes **no fit ordering**. The diffuse residual pedestal is
   carried by the per-window leakage-wing baseline. So the heavy coupling is real
   but resolved into self-contained subtractions that fully parallelize; this is
   the intended issue-#3 design (7-fixture-validated), not an artifact. **The
   worst-case-cost fixture is the best case for parallelism.**
3. **BLAS oversubscription gives negative wall benefit.** 2638 fit unpinned:
   cpu/wall **1.29** (337 cpu / 261 wall) but wall *worse* than pinned (261 vs
   256 s) — ~30 % extra CPU for no wall gain. Confirms: pin BLAS to 1 and own
   parallelism at the window level.

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

## Work item 2 — Stage 5 cross-window parallelism (scoped; the dominant cost once 1a–1c land)

The fit is the dominant remaining cost once the report is fixed — and it scales
hard with density (256 s → 779 s → **3496 s**), exactly where the DAG is widest.
`execute_plan` (`fitting/plan_execution.py`) walks windows sequentially in
`WindowPlan.topological_order`; that order is a **linearization of a DAG** whose
only real constraints are the (few) **edged** fixed-contributor dependencies and
the (rare) structural replan. The profiling resolved the design unknowns:

- **Antichain levelization.** Layer the DAG from `topological_order` +
  `dependency_edges` (the harness's `_levelize` in `scratch/perf-profile/`
  computes levels / widths / ceiling). Fit each level concurrently, barrier
  between levels: every window at level *k* has its edged primaries fit by levels
  `< k`. Measured ceilings: 125× / 243× / **1241×**.
- **Per-window task inputs (clarified by the edge_free finding):**
  - The window's slice of the **shared, read-only active FT + noise** — pass via
    fork-inheritance / shared memory, never per-task pickling (they are ~200k–1.2M
    points).
  - **`edge_free` contributors need nothing from other tasks** — they are
    materialized self-contained from that same shared active FT at fit time
    (`evaluate_edge_free_contributors`). This is the overwhelming majority
    (655: 100 %, 363: 93 %, 2638: 67 % of contributors) and is why most windows
    are leaves with no inbound edge.
  - Only **edged** contributors (2638: 130, 363: 119, 655: 0) need the *fitted*
    (amplitude, phase, refined frequency) of their primary window — read from the
    primary's converged fit (`_materialize_contributor`), the accurate leakage
    estimate the edge exists to preserve (vs the cruder self-contained edge_free
    fallback). Supply those from the already-completed earlier level; the
    levelization guarantees availability, so **parallelism costs nothing in fit
    quality** — the dependent still waits for the primary's real fit, only the
    genuinely independent windows run concurrently.
- **Process pool, not threads** (CPU-bound Python+NumPy → GIL). `fork` on Linux
  inherits the read-only arrays cheaply; return each window's `WindowFitResult`.
- **BLAS pinned to 1 per worker** — free (the per-window solve is single-threaded)
  and required so N workers don't each spawn M BLAS threads (finding 3 above:
  unpinned is net-negative even single-process).
- **Replan stays sequential.** **0 replan events on all three fixtures** — so
  parallelize the initial walk and run the (rare) structural-renegotiation rounds
  sequentially, re-parallelizing only the affected antichain if one ever fires.
- **Correctness gate: byte-identical fitted table** to the sequential run. An
  antichain has no inter-window dependency, so concurrent order cannot change
  results. Bit-reproducibility needs BLAS threads pinned identically in the
  baseline and the parallel run — pin to 1 in both (the production default
  question is open: pin globally, or pin only inside the fit's pool).

**Remaining implementation question (not a blocker):** process-pool/fork overhead
vs per-window fit time — most windows are cheap, so chunk a level's windows across
workers (e.g. `chunksize` by chunked submit) rather than one task per window, to
amortize IPC. Tune against 363/655 where the win is largest.

## Sequencing and gates

1. **Report 1a** (O(N²) reload) — biggest single win, lowest risk.
2. **Report 1b** (discarded overview) — trivial, byte-identical.
3. **Re-profile the report** (all fixtures, with report) to establish the new floor.
4. **Report 1c** (figure-render pool) — wall lever on the new floor.
5. **Stage 5 parallelism** — the 655 DAG/replan data confirms the ceiling (1241×,
   0 replans, leaf-heavy); ready.
6. **Benchmark suite** ([`perf-benchmarks.md`](perf-benchmarks.md)) captures the
   before/after as the committed regression guard.

Every step gated on byte-identical output (HTML or fitted table); the fit is
deterministic, so the gate is exact, not statistical.

## Handoff to a fresh implementation session

Start with **1a** — it is the largest win (most of the 2730 s report), the
lowest-risk change, and unblocks the report re-profile that sizes everything else.

- **1a touch points:** `get_candidate_ledger_impl` (`_internal/stage6_impl.py`,
  ~line 510) — add `*, spectrum_fit=None, sideband=None`; when both are given,
  skip `load_spectrum_fit_from_hdf5` and `load_fid_from_pipeline_impl` and use them
  directly (`acquisition_us` is already in `spectrum_fit.parameters`). Caller:
  `_assemble_report_site` per-window loop (`_internal/report_html_impl.py:2850`) —
  pass `spectrum_fit=bundle.fit, sideband=bundle.sideband`. The bundle is loaded
  once at line 2704. Dual-interface-safe; the standalone `(path, wid)` ledger verb
  is unchanged.
- **1b touch points:** `plot_window_panels` (`visualization/fit_detail.py:783`) —
  add `include_overview: bool = True`, skip building `figures["overview"]` when
  False; `render_fit_panels_impl` (`_internal/stage5_impl.py:2452`) — thread a
  `with_overview=False` and pass it through; the report caller sets it False (it
  discards the overview at `report_html_impl.py:2816`).
- **Gates:** extend `tests/unit/stage6/test_report_full.py` — (i) a ledger
  equality test (bundle-passed == self-loading) and (ii) a byte-identical-HTML
  assertion across the change (build the report before/after on the small fixture,
  compare). For 1c and item 2, the gate is byte-identical HTML / fitted table.
- **Re-profile after 1a+1b:** rerun `scratch/perf-profile/profile_run.py 2638
  examples/blackchirp_data/2638 26500 40000` (BLAS pinned) to size the new report
  floor before building the 1c figure pool.
- **Reference data:** measured numbers + cProfile dumps live under
  `scratch/perf-profile/` and
  [`research/performance-profiling/report.md`](../research/performance-profiling/report.md);
  the harness is `scratch/perf-profile/profile_run.py`.

## Out of scope / already settled

- **Intra-window Stage 5 levers** (batch add-loop, window decomposition,
  small-window retune) — explored and closed in
  [`stage5-nls-performance.md`](stage5-nls-performance.md); the per-window cost is
  assembly-bound and the region-shrinking levers all over-add via the
  window-size-sensitive accept gate. The remaining intra-window lever
  (local-support assembly) is lower priority than cross-window parallelism.
- **Optimizing the sub-20 s stages** (import/ft/noise/tau/peaks/windows/review) —
  not worth it against report and fit.
