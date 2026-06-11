# Stage 4 — edge-free leakage-contributor subtraction (implementation overview)

Status: **implemented.** Resolves the deferred O5-10 Tier-2 item
([`stage5-fitting.md`](stage5-fitting.md)): a bright line just outside a window
leaks its finite-T skirt across that window, but the leakage was subtracted from
*no* window, so the in-window lines were under-fit / missed / mis-fit. The
Stage-4 plan ([`stage4-window-assignment.md`](stage4-window-assignment.md)) and
the Stage-5 leakage-wing baseline
([`stage5-leakage-wing-baseline.md`](stage5-leakage-wing-baseline.md)) remain
authoritative for everything this work does not change.

## The bug

Stage 4's Tier-1 magnitude attachment correctly detects the leak and attaches
the strong line as a `FixedContributor` of the affected window, but attaching
also creates a fit-ordering **dependency edge** (the dependent window must be fit
after the contributor's primary). On dense / high-dynamic-range spectra the
strong-line neighbourhood is densely cyclic, so the Step-7 cycle-breaker
(`_topological_batches` → `dropped_cyclic_dependencies`) dropped the edge **and
its `FixedContributor`** to keep the DAG acyclic — discarding the needed
leakage subtraction. Issue #3 showed this was the dominant worst-ε driver on
**all seven same-instrument fixtures**, severity scaling with dynamic range
(χ²ᵣ up to ~1900 beside an SNR-20685 line).

## What ships

Leakage *subtraction* was coupled to fit *ordering*. The fix decouples them: a
frozen contributor only needs a good estimate of the neighbour line's
parameters, which need not come from a strict fit-ordering predecessor.

### 1. Edge-free frozen contributor

`FixedContributor` gains an `edge_free: bool` discriminator
(`core/data_structures.py`; serialized as the additive, back-compatible
`fixed_edge_free` column in `io/window_serialization.py` — legacy plans default
it to `False`). An edge-free contributor carries **no dependency edge**: it is
excluded from the DAG, so the cycle-breaker has nothing to drop, and it is
skipped by the local-thaw handshake (a thaw needs the primary's converged fit,
which an edge-free contributor deliberately does not depend on).

### 2. Cycle-breaker recovery (`preprocessing/window_planning.py`)

When the Step-7 cycle-breaker drops a fit-ordering edge, instead of discarding
the orphaned contributors it converts the few **dominant** ones to `edge_free`
and keeps them; the rest are dropped as before. Targeting is capped at the top
`max_edge_free_neighbors` (default `DEFAULT_MAX_EDGE_FREE_NEIGHBORS = 3`)
primary windows per dependent window, ranked by aggregated predicted skirt (the
same analytic envelope the Tier-1 attachment uses), keeping *all* contributors
of each kept primary so an adjacent cluster (the 360 w288 triplet) stays whole.
The cap is what keeps the subtraction targeted on a dense ultra-high-SNR forest
(655 drops dozens of edges); converting all of them re-creates the
global-crude over-subtraction that was negative in Phase 1
([`../research/stage5-cross-fixture/report.md`](../research/stage5-cross-fixture/report.md)).
The count is recorded as `diagnostics["n_edge_free_contributors"]`.

### 3. Self-contained read at fit time (`fitting/plan_execution.py`)

`evaluate_edge_free_contributors` reads each edge-free line's frozen
`(amplitude, phase)` directly from the active FT via a **joint complex
least-squares of the finite-T line template** over the cluster's core bins
(contributors grouped by `primary_window_id`). Solving the co-located lines
together de-contaminates the leakage pedestal — the global *single-bin phasor*
read was negative on the dense 655 spectrum. The read uses the dependent
window's τ (`tau0_us`), the **same** decay the frozen skirt is later drawn with
by `subtract_frozen_background`: reading at a different τ than the skirt is drawn
at biases the amplitude (verified on the 360 w287 A/B —
`scratch/stage4-leakage-contributor/lsq_read_ab.py`), which is why the read is
done at fit time (where the per-band τ is known) rather than at plan time.
`(amplitude, phase)` are physical/frame-independent; the offset is remapped into
the dependent window's frame.

### 4. Evidence-triggered accept/reject

`_fit_one_window` fits the window **without** the edge-free skirt first (the
byte-stable path a healthy window keeps, its in-window leakage already covered by
the const leakage-wing baseline), then adopts the skirt only if it reduces the
noise-weighted residual sum of squares to at most
`DEFAULT_EDGE_FREE_ACCEPT_FRACTION = 0.95` of the no-skirt fit's. This keeps the
subtraction evidence-triggered: an orphaned bright neighbour's skirt is
subtracted where it helps, while on a window the skirt would harm the no-skirt
fit stands unchanged, so the edge-free skirt never fights the leakage-wing
baseline.

## Open questions — resolved

- **O1 (amplitude read).** Joint LSQ-of-template at the window's τ, performed at
  fit time. Reproduces the validated prototype (360 w287: production
  22.09 → 4.75 with K 1→2; the bare prototype A/B is 108.7 → 2.61).
- **O2 (double-counting the leakage-wing baseline).** Resolved by the
  accept/reject gate: the skirt only fires where it strictly improves the fit;
  baseline-handled healthy windows keep the no-skirt path.
- **O3 (neighbour cap).** Cap by distinct primary windows
  (`max_edge_free_neighbors`), not per-line, so clusters stay whole while a dense
  forest stays targeted.
- **O4 (subtract vs merge).** Edge-free subtraction is primary; the structural
  merge remains available (an edge-free contributor no longer blocks a merge
  dispatch, since it cannot be thawed).

## Validation (issue #3, all seven fixtures)

Each fixture's persisted issue-3 build was re-fit through Stage 4 + Stage 5 with
the change; only Stages 4–5 differ from the before-baseline. Gate: SNR-aware
(`fitting/validation.py`, F=3.0, κ=0.05).

| fixture | worst-ε target χ²ᵣ | bulk(<100) median | overall pass |
|---|---|---|---|
| 2638 w49 | 4.48 → **1.96** | 1.273 → 1.273 | 0.997 → **1.000** |
| 363 w80  | 211.8 → 211.8 (gate rejected) | 2.049 → 2.049 | 0.731 → 0.735 |
| 1231 w266 | 74.3 → **12.8** | 1.393 → 1.358 | 0.862 → 0.883 |
| 1512 w167 | 77.2 → **31.8** | 1.409 → 1.358 | 0.921 → 0.953 |
| 655 w37  | 256.2 → **165.2** | 1.213 → 1.207 | 0.819 → **0.860** |
| 1019 w56 | 1906.4 → **556.2** | 1.354 → 1.288 | 0.902 → **0.967** |
| 360 w287 | 22.1 → **4.75** (K 1→2) | 1.456 → 1.449 | 0.955 → 0.958 |

- **Healthy control no-regression.** 2638 w105 36.40 → 35.35 and w106
  23.19 → 20.88, both K=4 preserved.
- **Dense-forest no-regression.** Every fixture's bulk(<100) median is flat or
  improved and every overall SNR-aware pass rate is flat or improved — the
  global-crude over-subtraction (Phase 1, bulk 2.40 → 4.71) is avoided; the 655
  bulk pass rate *rises* 0.805 → 0.844.
- **Plan invariants.** Windows stay disjoint; the DAG stays acyclic; edge-free
  contributors survive the cycle-breaker out of `dependency_edges`
  (`test_window_planning.py::test_mutual_attachment_does_not_break_dag`).
- Full suite green (1245 passed).

Harness: `scratch/stage4-leakage-contributor/validate.py` (per-window A/B +
controls) over the issue-3 before-baselines.

## Residual / deferred

- **363 w80 unmoved.** The accept/reject gate rejects its skirt (no strict
  improvement). 363 is the densest Gaussian forest; w80 reads as a blend the
  per-line subtraction cannot cleanly separate rather than a single orphaned
  bright neighbour — the O4 window-merge fallback (reusing the #13
  sub-resolution machinery) is the lever there, not a frozen skirt. No
  regression; left for a dedicated blend-vs-subtract pass.
- **Splitter skirt-proximity guard — deferred.** The bounded-merge cap-split can
  still place a boundary inside a strong line's skirt (the 363 w349 sliver).
  Snapping the boundary to a leakage-clean interior point via the rolling S_coh
  statistic was tried and reverted: S_coh oscillates (has nodes), so its
  minimum-coherence bin can land *beside* the strong line, drifting a boundary
  into a bright cluster and regressing a healthy neighbour (2638 w105 went
  36 → 89 before revert). The existing `split_proposal` shares this limitation.
  A non-oscillating leakage-reach metric (e.g. the analytic envelope, not the
  coherent edge sum) is needed; the edge-free subtraction already covers the
  sliver's leakage regardless of where the boundary sits.

## Touch points

`core/data_structures.py` (`FixedContributor.edge_free`,
`FrozenPeak.edge_free`), `io/window_serialization.py` (`fixed_edge_free`
column), `preprocessing/window_planning.py` (cycle-breaker conversion +
`max_edge_free_neighbors` cap, threaded through `build_window_plan` / `replan` /
`_finalize_plan` and the persisted `parameters`), `fitting/plan_execution.py`
(`evaluate_edge_free_contributors`, the accept/reject gate in `_fit_one_window`,
thaw/merge-dispatch exclusion of edge-free contributors). No new user-facing
parameter; the dual interface inherits the behaviour change unchanged.
