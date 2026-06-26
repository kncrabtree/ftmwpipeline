# Stage 5 — fold per-window cleanup into the fit walk (design spec)

**Status:** spec (implementation deferred to a fresh session). Branch context:
`stage6-cascade-refit`. Companion: `dev-docs/planning/stage6-cascade-refit.md`
(the cascade gate that surfaced this), memory `stage6-cascade-gate-finding`.

## Problem

The Stage-5 fit is a dependency-DAG walk: each window's fit reads the converged
fits of its predecessors (frozen-contributor background). After the walk,
`fit_peaks_impl` runs a **global post-pass** that prunes sub-floor dust and
collapses degenerate sub-resolution pairs (`apply_snr_survival_prune` +
`apply_vif_collapse`), each fanned across a *second* fork pool.

Running the cleanup *after* the whole walk means **every dependent window was fit
against the pre-cleanup source.** When the post-pass then collapses or prunes a
source window, its dependents are never re-fit — their frozen-background snapshot
is now stale relative to the source's final line list. Concretely (655):

- `w918`'s hub is fit, then `w925`/`w917`/`w912` are fit against it, then the
  post-pass collapses `w918` — leaving `w925`'s frozen contributors a
  *pre-collapse* `w918` state (lines doubled, `w918`'s strong 37018.804 line
  missing). Those windows no longer reproduce on a refit (χ²ᵣ 12.9 → 765) and are
  internally inconsistent with their own source.

This is the `#3b` class from the cascade gate (≈47 windows on 655). It is **not**
a reason to build a cascade: it is an artifact of deferring a per-window
operation to a global phase. The cleanup is per-window — it prunes *this*
window's dust and collapses *this* window's pairs; its only external input is the
window's already-converged predecessors (the frozen background), which the walk
worker already holds. So the cleanup belongs at the **tail of each node's
convergence**, before the DAG releases that node's dependents. Then every
dependent reads the final clean source the first time; there is nothing to
cascade for the automatic fit.

### Why it isn't already there (history, not a blocker)

A parallelization accident, not a design constraint. The NLS walk was
parallelized first, so the cleanup fell out as a post-pass. The cleanup was then
parallelized *separately* (its own fork pool) instead of folding into the NLS
worker's tail. `plan_execution` does **not** import `stage6_impl`/`stage5_impl`,
so no import cycle forced the split; the per-window cleanup units already exist
(`_survival_prune_window`, `_collapse_one`), already fan per-window.

A second, quieter bug rides the same deferral: the post-pass **re-resolves fit
kwargs globally** and so re-anchored each refit's tau penalty at the band-wide
`tau_maj` instead of the window's per-band τ (the recurring `tau_maj`-vs-per-band
bug — patched separately via `resolve_window_tau_anchor`, see the
`stage6-cascade-gate-finding` memory and `cleanup-pass.md`). The in-walk cleanup
**inherits the node's exact `fw_kwargs`** (per-band τ anchor, baseline order,
spur mask, penalties) — it physically cannot re-resolve them wrong. Moving the
cleanup in-walk removes that whole bug class structurally.

## Design principle

**A node's convergence includes its cleanup.** The walk's native currency is
`WindowOutcome`; `FittingResult` is the persistence/API projection produced once
at the end (`plan_fit_outcome_to_spectrum_fit`). The cleanup should run in
outcome-space inside the worker, minimizing conversions, and store the *clean*
outcome so dependents and the final projection both see it.

## The decomposition

Today's refit primitive (`refit_window_core`, in `stage6_impl`) does three things
fused together: (1) **reconstruct** an outcome from a *persisted* `FittingResult`
(frozen background from the `fixed_parameters` snapshot, baseline replay, seeds
from `fitted_peaks`, per-band τ), (2) run the NLS, (3) **convert** back to a
`FittingResult`. For the in-walk path, (1) and (3) are exactly the wasteful
round-trip — the live outcome already holds the frozen background, and the result
should stay an outcome. Split it:

### 1. `refit_outcome(outcome, *, add, remove, freeze_inherited) -> WindowOutcome`

New, in `plan_execution` (beside `fit_seeds_window_outcome`). The **live-outcome**
refit: reuse the node's `offset_grid` / `z` / `rms` / `background` / `fixed_peaks`
from `outcome`, take the surviving seeds from `outcome.fit.peaks` (apply
`add`/`remove`), and re-run `fit_seeds_window_outcome` with **the node's own
`fw_kwargs`** (so τ anchor / baseline / spur mask / penalties are inherited, not
re-derived). Returns a `WindowOutcome`. No snapshot reconstruction, no
`FittingResult` conversion. This is the primitive the cleanup drives.

### 2. Outcome-level cleanup metrics (avoid the `FittingResult` round-trip)

The prune/collapse *decisions* read per-line `snr`, `amplitude_error`,
`frequency_mhz` — today via `FittingResult.fitted_peaks`. These are all derivable
from the outcome (`outcome.fit.fit`: `WindowFitResult` peaks + covariance, plus
`rms_noise`, τ, center). Provide light per-peak metric accessors on the outcome
(`snr`, `amplitude_vif`) so the cleanup classifies dust / selects degenerate
pairs **without** building a `FittingResult` each iteration. (`amplitude_vif`
already exists in `validation.py`; it needs `amp_err`/`amp`/`snr`, all on the
outcome.) The single `FittingResult` projection stays at end-of-walk.

**Decided (DRY): a shared fitted-line view.** Introduce one thin "fitted-line
view" — the per-line `(frequency_mhz, amplitude, amplitude_error, phase, snr)` the
cleanup reasons about — computable from *either* a `WindowOutcome` (in-walk) or a
`FittingResult` (post-fit user edit). The prune/collapse decision functions take
the view, so there is one decision code path serving both, no duplicated metric
logic, and no per-iteration `FittingResult` construction in the walk.

### 3. `finalize_node(outcome) -> outcome | DROPPED` (the per-window cleanup)

The prune→collapse sequence for one window, in outcome-space: classify dust →
`refit_outcome(remove=worst)` to a fixpoint; then select degenerate/singular pair
→ `refit_outcome(remove=pair, add=merged, freeze_inherited=...)` to a fixpoint;
iterate prune↔collapse as today (the existing fixpoint/`max_iterations` logic).
Returns the clean outcome, or a `DROPPED` sentinel when the window cascades to
empty (all dust). This is the existing `_survival_prune_window` + `_collapse_one`
logic, re-expressed on outcomes.

**Layering / injection.** `finalize_node` is built in `stage5_impl` (it owns the
floor, the VIF threshold, the per-band τ, the collapse footprint constants) and
**injected** into `execute_plan` / the walk as a callback
(`Callable[[WindowOutcome], WindowOutcome | DROPPED]`). The walk calls it; it does
not import the cleanup. `plan_execution` stays import-clean.

### 4. Evaluate the frozen background from the ancestor's *fit*, not Stage 3

This is the clean root fix (it subsumes the dedup patch below it in earlier
drafts). Today `_fit_one_window` builds one `FrozenPeak` **per `FixedContributor`**
— a Stage-3-derived record (`peak_index`, a Stage-3 `frequency_mhz`) — via
`evaluate_fixed_contributor`'s nearest-match into the primary's fit. That couples
contributor *content* to Stage 3, which is wrong: when a source is cleaned to
fewer lines than Stage 3 detected, several contributors nearest-match the **same**
cleaned source peak and its skirt is summed N times (the `w925` doubling), and a
stale Stage-3 frequency can match the wrong line.

**Stage 3 should determine the plan (which windows depend on which) and the gold
seeds — nothing about the frozen-background content.** The content is whatever the
**ancestor window actually fit.** So: for each ancestor (primary) window on a
dependent's dependency edges, freeze the ancestor's **current above-`min_freeze_snr`
fitted lines** that fall in the dependent's leakage range (the dependent's window
span expanded by the leakage margin Stage 4 already uses to establish the edge),
each at the ancestor's *fitted* freq/amp/phase. No `peak_index`, no per-Stage-3
nearest-match.

Consequences (all good, and why this is the right model):

- **No doubling, ever** — N ancestor lines → N `FrozenPeak`s, irrespective of how
  many Stage-3 detections there were.
- **Never stale** — the background is the ancestor's *current* fit by
  construction. In-walk that is the live cleaned outcome (`#3b` dissolves); a
  post-fit refit reads the ancestor's persisted fit in the `SpectrumFit`.
- **Simplifies the future user-edit cascade (`§§C/D`)** — "refresh a dependent
  after its source changes" becomes "re-evaluate from the source's current fit,"
  with no snapshot to keep in sync.
- **Retires the per-dependent frozen-background snapshot** (the
  `fixed_parameters` freq/amp/phase that `_reconstruct_frozen_peaks` reads): the
  dependency *edge* (`primary_window_id`, `edge_free`) is kept; the per-peak
  Stage-3 snapshot is no longer the source of truth for evaluation. Verify the
  serialization change against `_reconstruct_frozen_peaks`'s current consumers
  before removing fields (keep them as provenance if cheaper).

Implementation details to settle when the code takes shape: the exact leakage
inclusion criterion (reuse Stage 4's margin / the existing contributor span);
how `freeze_eligible` / thaw-eligibility (currently per-contributor) is derived
when contributors become per-ancestor-line; and `edge_free` contributors (read
self-contained from the active FT) which are already fit-independent and stay.

### 5. `refit_window_core` for post-fit user edits (module move)

After the cleanup moves in-walk, `refit_window_core`'s only remaining consumer is
the **post-fit** path (Stage 6 user edits: `refit_window_impl`, merge/split/
accept). Refactor it to wrap `refit_outcome`: reconstruct an outcome from the
persisted `FittingResult` (the snapshot/baseline/τ replay — genuinely needed
post-fit), call `refit_outcome`, convert to `FittingResult`. Have it return the
outcome too (or expose the outcome variant), so there is one refit core with one
projection step. **Move it to `stage5_impl`** (it is a fit primitive that belongs
with the fit driver); `stage6_impl` imports it downward (removing the current
`stage5 -> stage6` upward import for the cleanup callbacks). The per-band τ replay
already added to `refit_window_impl` stays.

## Walk integration

- **`_walk_windows_dag`** (default): after a worker produces `outcomes[wid]` and
  before decrementing successors' indegree, run `finalize_node`. On `DROPPED`,
  record the drop and treat dependents' contributors from it as absent (see edge
  cases). Run it **inside the worker** (`_fit_window_worker_dag` /
  `_process_one_window` tail) so the cleanup's refits ride the same fork and BLAS
  pin — one pool, not two.
- **`_walk_windows_in_order`** (sequential authority / thaw-accept fallback) and
  the **legacy level walk** (`FTMW_LEGACY_LEVEL_WALK`): same finalize-per-node
  tail, so all three paths stay identical (the existing byte-identity bar).
- **`fit_peaks_impl`**: delete the global `apply_snr_survival_prune` /
  `apply_vif_collapse` calls and the second fork pool. **Diagnostics (cleaner):**
  record each window's cleanup provenance (what it pruned / collapsed) **on its
  outcome**, where it belongs, and derive any aggregate (`peak_survival` /
  `vif_collapse` summaries) from those at end-of-walk rather than threading a
  global record list through the pool. Reports gain per-window cleanup provenance
  for free; the aggregate summary can keep its current shape if any consumer
  needs it.

## Edge cases (must be in the spec the implementation follows)

- **Cascade-to-empty source.** A window pruned to empty is `DROPPED`. Its
  dependents (fit later) must skip contributors that reference it — the same
  "no fitted peak to freeze" skip the spur-on-contributor path already uses. A
  dropped source carried only sub-floor dust, so dropping the leak is correct.
- **Thaw.** Thaw is a dependent-side op that mutates a primary in place; the DAG
  walk already falls back to the sequential authority if any thaw is accepted
  (0 on all fixtures). With in-walk cleanup the dependent thaws the *cleaned*
  primary — fine, but keep the same accepted-thaw → sequential-redo guard.
- **Rescue / cleanup order.** Per node: NLS → thaw → rescue → prune → collapse.
  Cleanup is the tail (after discovery), matching today's global order.
- **`freeze_inherited`.** The VIF-collapse sequential merge pins inherited peaks
  during the frozen intermediate refit; `refit_outcome` must support it (it is in
  `refit_window_core` today).
- **Doublet adjudications / baseline audit / covariance.** These live on the
  outcome already; ensure `finalize_node`'s refits carry them forward so the
  end-of-walk projection is unchanged in shape.
- **Determinism.** Outcomes are keyed by window id (order-independent); the
  per-node cleanup must be deterministic given the node + its predecessors. Keep
  the per-window fixpoint deterministic (it is — greedy highest-VIF, lowest-SNR).
- **User-edit refit semantics (unchanged scope).** Whether a *user* edit re-runs
  prune/collapse is a separate decision; this spec covers the automatic fit. The
  post-fit user-edit cascade (a user edits a source → its dependents) is still
  future `§§C/D` work — in-walk cleanup only removes the *automatic* fit's
  internal staleness, not user-driven propagation.

## What this fixes / does not fix

- **Fixes:** `#3b` (stale frozen-background snapshots in the automatic fit) — two
  ways, both by construction: in-walk cleanup means dependents read *cleaned*
  sources, and §4 means the frozen background is the ancestor's *current fit*,
  never a Stage-3 snapshot. Removes the global-kwargs re-resolution bug class.
  Likely makes the `w918`/`w900s` cluster reproduce. Retires the per-dependent
  frozen-background snapshot and the contributor doubling.
- **Eases (not in scope here):** the post-fit **user-edit** cascade (`§§C/D`)
  becomes "re-evaluate from the source's current fit" once §4 lands — no snapshot
  to refresh. Still future work, but much smaller.
- **Does not address:** `#3c` (giant-skirt degeneracy where even a correct
  frozen background is ill-conditioned — separate); the user-edit cascade itself
  (`§§C/D`); split/delete identity changes beyond the per-ancestor-line model.

## Validation & re-baseline

This changes the automatic fit (dependents fit against cleaned sources), so it
**re-baselines**. Plan:

1. Trimmed-band A/B first (`scratch/cascade/build_band.py` around the `w918` hub
   and a clean region) for fast iteration; confirm the hub cluster reproduces
   (`scratch/cascade/_repro_buckets.py`) and no clean-region regression.
2. Full 7-fixture rebuild; compare to `scratch/cascade/BASELINE.md` (lines,
   windows, χ²ᵣ med/p90, multiplet spread) and honest recall
   (`recall_honest.py`); the dense fixtures (655/363/1231) move most.
3. Reproducibility sweep (`_repro_buckets.py`) on 655: target the residual
   ≥50 kHz movers dropping to the `#3c` giant-skirt remainder only.
4. Cross-interface + determinism: the curated fit is byte-stable under re-walk;
   all three walk paths (dag / sequential / legacy level) identical.

## Open design questions (resolved — implementer settles the residual details)

- **§2 metrics — DECIDED:** one shared fitted-line view (DRY); both paths use it.
- **§ module home — implementer's call once the code takes shape.** `refit_outcome`
  (and likely `refit_window_core`) in a new `fitting/window_refit.py` that both
  `plan_execution` and `stage5_impl` import is the mild preference, but
  `plan_execution` is acceptable; no strong constraint.
- **§6 diagnostics — DECIDED: cleaner.** Per-window cleanup provenance on the
  outcome; derive any aggregate from it.
- **§4 contributor evaluation — DECIDED: evaluate from the ancestor's fit, not
  Stage 3.** Replaces the dedup patch entirely. Residual implementation details:
  the leakage inclusion criterion (reuse Stage 4's margin), `freeze_eligible` /
  thaw-eligibility when contributors become per-ancestor-line, and how far to take
  the snapshot retirement (vs keeping the fields as provenance). These settle when
  the code takes shape; verify against `_reconstruct_frozen_peaks` consumers.
