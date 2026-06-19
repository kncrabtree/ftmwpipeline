# Plan: performance optimization

Status: **Work item 1 (report) and Work item 2 (Stage 5 cross-window fit
parallelism) both complete and validated.** The report is near-optimal (clean
50.8 → 37.3 s, byte-identical). The fit now levelizes the window DAG into
antichains and fits each level across a fork-per-level process pool: clean 2638
fit **126.3 → 28.9 s (4.37×), byte-identical** to the sequential walk; 363 and
655 also byte-identical (max |Δfreq|=|Δamp|=|Δphase|=0 on all three). The
benchmark suite ([`perf-benchmarks.md`](perf-benchmarks.md)) becomes the
regression guard. Raw artifacts under `scratch/perf-profile/`
(`*_summary.json`, `*.prof`, `*_pstats.txt`) and the clean re-measurements +
equivalence harnesses under `scratch/cow-exp/`.

> **Profiled numbers below are cProfile-inflated — see the methodology caveat.**

> **⚠ Methodology caveat (discovered after 1c): `profile_run.py` numbers are
> cProfile-inflated and must NOT be read as production wall.** The harness wraps
> the whole run in `cProfile` and times each stage *while the profiler is active*.
> The forking render pool (1c) makes this acute: the workers **inherit the active
> profiler** across `fork`, so every matplotlib call in every worker ran
> instrumented (~5× slowdown). The clean (unprofiled) re-measurements on the
> reusable `scratch/perf-profile/2638/2638.ftmw` fixture:
> - **Per-window render, all 277 windows: 22.1 s wall, ideal/14 = 21.5 s → 97 %
>   parallel efficiency.** Not the "120 s / 40 % efficiency" the profile implied.
> - **Full report `report_run`, unprofiled: 50.8 s.** Not 279 s — cProfile inflated
>   it ~5.5×. (`top` confirmed: profiled = 42 % sys instrumentation thrash; clean =
>   90 % user / 5 % sys, compute-bound.)
> - The COW deep-copy/slice idea was tested (`scratch/cow-exp/exp.py`, shared-fork
>   vs per-window pickled slice) and **falsified** — identical wall, sys-CPU, and
>   minor faults. The ~3 M page faults are matplotlib RGBA raster-buffer
>   `mmap`/`munmap` churn (intrinsic to rendering), not COW on the shared fit graph
>   (~a few thousand faults, negligible). Data layout is not the lever.
>
> So the per-window figure pool (1c) is essentially done — 97 % efficient, ~22 s.
> The fit numbers below are also cProfile-inflated (less so — single process), so
> **re-measure the fit unprofiled before sizing item 2.**

**Profiled cumulative result (2638, BLAS pinned, byte-identical at every step) —
useful only for *relative* before/after, not absolute wall:**

| stage (PROFILED) | baseline | after 1a+1b | after 1c |
|---|--:|--:|--:|
| report | 2730 s | 1580 s | 279 s |
| fit | 256 s | 258 s | 257 s |
| total run | 3015 s | 1867 s | 565 s |

Clean unprofiled report after the full report-efficiency pass: **37.3 s** (1c
render pool → ~50.8 s; + in-memory no-disk assembly → ~47 s; + parallel methods
figures → 37.3 s). 1a+1b removed a genuine O(N²) per-window HDF5 reload; 1c
parallelized the matplotlib render; the assembly work removed the disk round-trip
and parallelized the methods figures. All byte-identical. Clean stage picture now:
**fit 119.5 s, report 37.3 s, other ~10 s** — the fit (Work item 2) is the lever.

**Report assembly: in-memory single-file path ✅ (no disk round-trip).** The
single-file production path no longer writes the O(N) per-window PNGs + HTML pages
to a scratch site and re-reads them in the collapse. `_assemble_report_site`
returns a `_ReportModel` (figure bytes + worker-downscaled hover thumbnails + page
HTML + css); `report_full_impl` passes `write_files=False` and
`_collapse_site_to_single_file` embeds straight from the model (only the O(1)
methods-page figures + overview stay on disk). The thumbnail LANCZOS downscale (the
bulk of the old collapse cost) moved into the render workers (parallelized).
**Byte-identical** to the disk path (gated by
`test_inmemory_collapse_byte_identical_to_disk`). Measured 2638: 51.4 → 47.3 s
(~4 s, ~8 %); scales with window count (≈18 s on 655's 1241 windows). The
multi-file site (`write_files=True`, default) is retained for the test/inspection
callers.

**Methods figures parallelized ✅.** `_methods_stage_figures` (the per-stage
diagnostic plots — 750k-point FID overview, noise/peaks/windows full-spectrum
visualizations) rendered its ~7 independent figures serially. Now they render
across the same forking pool as the window panels (thunks reached via a
fork-inherited global — they are local closures over the file path, not
picklable, so they are inherited not sent), each returning PNG bytes the parent
writes + builds HTML from in group order. Measured: `_methods_stage_figures`
18.0 → 8.2 s (2.2×, bounded by the FID overview, the long pole). Byte-identical,
covered by `test_report_figures_parallel_byte_identical_to_serial` (the
`_FIGURE_RENDER_WORKERS` toggle now gates both window and methods rendering).

**Clean report wall after this session's assembly work: 50.8 → 37.3 s (−27 %).**
1c (figure pool) + in-memory no-disk assembly + parallel methods figures, all
byte-identical. Remaining headroom: the ~8 s methods floor (the FID overview) and
the ~22 s window-render floor could *overlap* (submit methods thunks into the same
pool as the window panels so the 8 s hides under the 22 s) for ~another 8 s — a
larger restructure of the assemble flow, deferred. Against a 37 s report and a
119 s fit, the fit (Work item 2) is the next real lever.

**Implementation order:** 1a ✅ → 1b ✅ → re-profile ✅ → 1c (figure-render pool) ✅
→ **fix `profile_run.py` to disable cProfile around the report (or report
unprofiled wall) so future numbers are trustworthy** → **re-measure the fit
unprofiled** → 2 (cross-window fit parallelism) if the clean fit wall justifies it
(DAG ceiling 125× on 2638, 243× / 1241× on the dense fixtures).

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
   Stage 5. The pre-implementation read here was **wrong about the cause** and is
   corrected by the post-1a re-profile (see the Status block): cProfile flagged
   `fitting_serialization._load_peak_columns` running **77,560 ≈ 277²** times on a
   277-window fit, but that O(N²) reload was only ~42 % of the report wall, layered
   *on top of* a matplotlib floor that was always the dominant cost. Removing the
   reload (1a) + the discarded overview (1b) cut the report 2730 → 1580 s; the
   remaining 1580 s is ~100 % per-window figure rendering (1c's target). The dense
   fixtures ran with `--no-report` on purpose — re-confirming the floor at 600–1200
   windows would cost hours; size it from the 2638 re-profile and scale.
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

### 1a. Eliminate the O(N²) per-window reload ✅ (landed; −42 % of the report with 1b, *not* the dominant cost — see Status)

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

### 1b. Stop building the per-window overview that is discarded ✅ (landed)

`plot_window_panels` (`visualization/fit_detail.py:783`) builds
`figures["overview"]` — a full-spectrum `draw_overview` + model resynthesis over
the whole band — for **every** window, and the report immediately `plt.close()`s
it (`report_html_impl.py:2816`; it uses the *shared* interactive overview). ~N
full-spectrum figures built and thrown away.

- Add `include_overview: bool = True` to `plot_window_panels`; thread a
  `with_overview=False` through `render_fit_panels_impl` from the report caller.
  (The report loop *retains* its `panel == "overview"` discard guard so a
  caller that still hands back an overview is closed, not embedded — this is
  what makes the 1b change byte-identical and keeps the byte-identity test's
  forced-overview reconstruction faithful.)
- Output byte-identical by construction (the overview was discarded anyway).

### 1c. Parallelize per-window figure rendering ✅ (landed; report 1580 → 279 s, 5.7×)

The post-1a/1b report (1580 s on 2638) was ~100 % matplotlib —
`_save_figure_png`/`savefig` was **1239 s (78 %)** across **1402 per-window
figures** (re/im/mag/hist + correlation heatmap). Per-window figure rendering is
**embarrassingly parallel** (each window's panels are independent).

**As implemented** (`_internal/report_html_impl.py`):
- `_figure_png_bytes(fig, dpi, **kw)` factored out of `_save_figure_png` (the
  sink-agnostic encoder, so worker-returned bytes == file-written bytes).
- `_render_one_window_figures(wid, …)` renders one window's panels + correlation
  heatmap to PNG bytes and captures `_mag_axes_geometry` **in the worker, after
  savefig forces the constrained-layout pass**, returning it with the bytes.
- `_render_all_window_figures` drives a **forking** `ProcessPoolExecutor` (read-only
  `bundle` inherited via fork — never pickled per task; only the int `wid` goes
  out, PNG bytes come back), capped at `cpu_count − 2`, set via the fork-inherited
  module global `_WORKER_RENDER_CTX`. Serial fallback when `n < 2`, one worker, or
  no `fork`. `_FIGURE_RENDER_WORKERS` pins the count for the test.
- The report loop renders all windows up front, then the **parent** writes the
  bytes to the figure files and builds the pages (all figure IO stays serial in
  the parent).
- **Gate:** `test_report_figures_parallel_byte_identical_to_serial` — serial
  (`_FIGURE_RENDER_WORKERS=1`) vs pool render produce byte-identical figure PNGs
  *and* HTML.

**Measured (clean, unprofiled — see the Status methodology caveat):** all 277
windows render in **22.1 s at 97 % parallel efficiency** (ideal/14 = 21.5 s), and
the full unprofiled `report_run` is **50.8 s**. 1c is essentially optimal for the
per-window render. The profiled view (report 1580 → 279 s, 42 % sys, "40 %
efficiency") was a cProfile artifact: the forked workers inherited the active
profiler. The COW deep-copy/slice idea was tested and **falsified** — the page
faults are matplotlib raster-buffer `mmap` churn, not COW on the shared fit graph
(`scratch/cow-exp/`).

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
  inherits the read-only arrays cheaply; return each window's outcome. Fork
  **per level** so each level's workers inherit the outcomes of all earlier
  levels (set a module global before the pool forks, as 1c does).
- **BLAS pinned to 1 per worker** — free (the per-window solve is single-threaded)
  and required both to avoid N×M oversubscription *and* for bit-reproducibility
  (BLAS=1 fixes the reduction order; pin it in the sequential baseline too).

**Measured hazard analysis (persisted fits, `scratch/cow-exp/fit_hazards.py`) —
this CORRECTS the doc's earlier priors:**

| fixture | windows | edges | shared primaries | thaw accepted | replan attempts (accepted) |
|---|--:|--:|--:|--:|--:|
| 2638 | 375 | 73 | 9 (≤3 deps) | **0** | 0 |
| 363 | 486 | 62 | 0 | **0** | **10 (6)** |
| 655 | 1241 | **0** | 0 | **0** | **41 (0)** |

- **The real write hazard is the thaw co-fit, not the edge read.** `_perform_thaw`,
  on accept, calls `_install_cofit_outcome(primary_outcome, …)` — it **mutates the
  primary window's outcome in place**. Two same-level dependents that share a
  primary and both thaw-accept would race, and a worker's thaw-accept mutates only
  its *fork-private* copy of the primary (lost on return). **But thaw-accept = 0 on
  all three fixtures** (10/42/136 attempts, 0 accepted), so the common path has no
  cross-window write at all. Required: a **guard** — if any worker reports an
  accepted thaw, re-process the affected windows sequentially (or fall back).
- **Replan is NOT 0** (the doc's "0 replans on all three" was wrong — a cProfile-era
  claim). 363 accepts 6 merges (real plan revisions + re-walks); 655 makes 41 no-op
  attempts; 2638 none. So **replan must be a sequential tail**: parallelize the
  initial walk, then run the bounded structural-renegotiation rounds sequentially,
  re-parallelizing only the affected antichain. Not a blocker (the initial walk is
  the bulk), but real implementation surface.
- **655 has 0 edges → fully independent windows** = the cleanest, biggest win
  (1241 windows, the most expensive fit).

**Correctness gate: scientific equivalence, NOT byte-identity** (per the project
owner). Unlike the report (where figures are byte-identical), the fit gate is:
**structure identical** — same windows, same per-window peak counts, same accepted
merges/thaws — and **continuous parameters (freq/amp/phase/τ) agree with the
sequential run to ≪ the uncertainty scale** (ULP/roundoff, the same magnitude as
the already-validated thread-count variation). So fork + BLAS=1 numerical drift is
acceptable and the bit-reproducibility question is **not** make-or-break (it is a
strictly more reproducible regime than the thread-count sweep already accepted).
What must stay exact is *structure*: the thaw-accept guard and the sequential
replan tail are about structural correctness (a lost thaw-accept or skipped merge
moves a line by a real amount, not roundoff), not float bits.

**Staged implementation plan — all three stages DONE + validated:**
1. **Extract** the per-window body into a self-contained `_process_one_window`
   (`fitting/plan_execution.py`); the walk calls it in a loop. ✅ Verbatim block
   move, byte-identical to the pre-refactor 2638 fit.
2. **Levelize + parallelize the initial walk.** ✅ `_levelize` layers the DAG into
   antichains; `_walk_windows_parallel` forks a fresh pool per level (each level's
   workers inherit all earlier-level `outcomes` via fork, return their own
   outcome + thaw/rescue events). Dependencies are derived from each window's
   **non-`edge_free` `fixed_contributors`** (the authoritative read-ordering),
   unioned with `dependency_edges` (which the plan does not always populate — the
   original `dependency_edges`-only levelization was the first bug). `ex.map` with
   the default chunksize=1 gives dynamic load balancing. The thaw-accept guard
   re-fits a whole level sequentially if any worker reports an accepted thaw
   (never fires on the validated fixtures — thaw-accept = 0). BLAS pinned to 1 in
   the workers; the non-pool path (`_FIT_WINDOW_WORKERS == 1`, no `fork`, or a
   width-1 level) delegates to the original `_walk_windows_in_order` — the
   sequential reference.
3. **Validate.** ✅ Byte-identical (max |Δfreq|=|Δamp|=|Δphase|=0) on all three:
   2638 (4.37×), 363 (replan + 6 merges), 655 (0 edges). Because thaw-accept = 0
   everywhere, any valid topological linearization gives identical results, so the
   level-grouped walk matches the original order bit-for-bit — equivalence held
   even stronger than the roundoff gate required.

**Also parallelized: the replan tail.** The structural-renegotiation loop's
affected-window re-walk routes through `_walk_windows_parallel` too (the affected
set is its own antichain DAG; its already-fit primaries sit in `outcomes` and are
treated as satisfied). On 363 the sequential replan tail was **25.6 s (27 % of the
parallel-fit wall)** — the largest single serial component once the initial walk
parallelized.

**Where the remaining 363 wall goes (the "long tail").** Phase breakdown of the
94 s parallel fit: **initial walk 50.4 s** (level 0 = 424 windows @ 47.0 s, level
1 = 62 @ 3.4 s), **replan tail 25.6 s** (now parallelized), **~18 s** aggregation /
serialization outside `execute_plan`. The level-0 floor is **not** a single
monster window — measured per-window costs spread (summed 330.8 s, max single
20.7 s ≈ sum/14 = 23.6 s; top-10 windows = only 32 % of cost), so longest-job-first
scheduling would not help. Level 0 runs at ~50 % parallel efficiency (47 s vs the
23.6 s ideal floor) because dense per-window NLS is memory-bandwidth-bound — the
intra-window levers that would lower per-window cost are explored-and-closed in
[`stage5-nls-performance.md`](stage5-nls-performance.md).

## Sequencing and gates

1. **Report 1a** (O(N²) reload) — biggest single win, lowest risk.
2. **Report 1b** (discarded overview) — trivial, byte-identical.
3. **Re-profile the report** (all fixtures, with report) to establish the new floor.
4. **Report 1c** (figure-render pool) — wall lever on the new floor.
5. **Stage 5 parallelism** ✅ — levelize + fork-per-level pool + parallel replan
   tail; byte-identical on 2638/363/655; 2638 4.37×.
6. **Benchmark suite** ([`perf-benchmarks.md`](perf-benchmarks.md)) captures the
   before/after as the committed regression guard.

The report steps were gated on **byte-identical HTML**. The fit (Work item 2) was
gated on **scientific equivalence** — identical structure (windows / peak counts /
merges) with freq/amp/phase/τ within roundoff — and in fact came out
**byte-identical** on all three fixtures (thaw-accept = 0, so the levelization is a
valid topological linearization that reproduces the sequential result exactly).

## Implemented design (Work item 2)

The fit-parallelism surface lives in `fitting/plan_execution.py`:
- **`_levelize(order, by_id, dependency_edges)`** — antichain longest-path layering.
  Reads ordering from each window's non-`edge_free` `fixed_contributors` (the
  authoritative read-dependency) unioned with `dependency_edges`.
- **`_walk_windows_parallel(...)`** — the parallel walk. Forks a fresh
  `ProcessPoolExecutor("fork")` per level (workers inherit earlier-level
  `outcomes` + the shared `active_ft`/`noise` via fork; only the `(n_done, wid)`
  tuple is pickled out, the `WindowOutcome` + thaw/rescue events come back). The
  thaw-accept guard re-fits a whole level sequentially if any worker accepts a
  thaw. `_FIT_WINDOW_WORKERS` (None = auto cpu−2, 1 = sequential reference) pins
  the count; the non-pool path delegates to **`_walk_windows_in_order`** (the
  original sequential walk, kept as the reference).
- **`_fit_window_worker(task)`** — the pool entry point; pins BLAS to 1.
- **Per-phase timing** at INFO (`fit walk: … levels … workers`, `level k/n: … Ns`,
  `initial walk: Ns`, `replan tail: Ns`) makes the wall split visible per run.
- Both the initial walk and the replan loop's affected re-walk call
  `_walk_windows_parallel`.

**Equivalence gate harnesses (under `scratch/cow-exp/`, gitignored):**
- **`fit_equiv.py <name> <source> <lo> <hi>`** — the fast gate: builds stages 0–4
  **once** into a cached `cache_<name>_stage4.ftmw`, then copies it and runs ONLY
  `fit_peaks` for `workers=auto` vs `workers=1`, comparing the fitted table +
  reporting the speedup. (Stage 0–4, esp. `calibrate_tau`'s 81k-NLS, is the slow
  part — never redo it per fit.)
- `validate_parallel.py` (2638, full build twice) and
  `validate_parallel_fixture.py <name> … [ref.ftmw]` (build par, compare to a
  persisted sequential fixture — used for 655 to skip its ~20 min sequential fit).
- `diag_363_levels.py` / `probe_363_costs.py` — level structure + per-window cost
  distribution (how the long-tail finding was measured).

**BLAS note:** the workers set `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1`, but env set
after fork does not re-init an already-loaded OpenBLAS — production fit runs should
launch with BLAS pinned to 1 in the parent (the conda/CI harness already does).
The per-window solve is small and barely threads BLAS regardless (cpu/wall 1.29
unpinned), so oversubscription risk is low.

## Out of scope / already settled

- **Intra-window Stage 5 levers** (batch add-loop, window decomposition,
  small-window retune) — explored and closed in
  [`stage5-nls-performance.md`](stage5-nls-performance.md); the per-window cost is
  assembly-bound and the region-shrinking levers all over-add via the
  window-size-sensitive accept gate. The remaining intra-window lever
  (local-support assembly) is lower priority than cross-window parallelism.
- **Optimizing the sub-20 s stages** (import/ft/noise/tau/peaks/windows/review) —
  not worth it against report and fit.
