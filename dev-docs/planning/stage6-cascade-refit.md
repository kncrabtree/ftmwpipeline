# Stage 6 — contributor-edit cascade (window-level dependency resolution)

## Status

The cascade's prerequisites are all shipped and committed on
`stage6-cascade-refit`. **The cascade proper — propagating a Stage-6 contributor
edit into its dependent windows — is the remaining work.**

**The gate experiment is done and decisive (2026-06-26).** On the reproducible
baseline, with a faithful **window-level** refresh, a contributor edit propagates
**≪ σ_f** to dependents in every realistic class — 0/41 dependents moved ≥0.5σ at
the worst-case hub (655 `w1006`) for identity / insidious-satellite / structural
split, firing materially (5/41, 469 kHz) only on the drastic deletion of the
dominant strong line. The far-field skirt depends only on the source's total power
and centroid, invariant under split/merge. **The cascade is therefore a
correctness/honesty fix with rare practical bite — build the lightweight core, not
heavy per-hop machinery.** Full writeup: `scratch/cascade/EXPERIMENT_FINDINGS.md`
(headline block); harness `scratch/cascade/cascade_lab_wl.py`.

**The build is small** because the cascade folds into machinery that already
exists (see "The build" below). The model is **two states — automatic and revised
— with no multi-level undo history**: the revised fit is always re-derived from
the automatic baseline by replaying the decision log, and every mutation (an edit,
a second edit, an undo) re-derives from the automatic fit. This is exactly the
existing `review_undo_impl` path (`_restore_stage5_baseline` → rebuild review →
replay decisions); the cascade just widens what each replay recomputes. The only
genuinely new logic is the **window-level dependent refresh**.

"The decided design" (§§1–6) and "The build" below are the spec. Everything above
the horizontal rule is the prerequisite summary and original motivation.

## Implemented (prerequisites, committed)

The cascade kept bottoming out on fit robustness and refit reproducibility. Those
are resolved; the twist-by-twist investigation detail lives in git history and the
linked memories, not here.

- **Stage-4 curvature cycle-break (§B)** — `window_planning.py` Step 7 orients
  every leakage-dependency edge strong→weak into a tiered acyclic DAG and keeps a
  downward skirt edge-bearing only when *material* (`S_level` or curvature
  residual `S_resid`); legacy behind `FTMW_LEGACY_CYCLE_BREAK`. 7-fixture A/B was a
  net improvement (over-split removal, not a wash). Commit `580c3cd`.
- **Stage-5 seeder rework** — `conservative_fit` seeds all detected primaries up
  front (add-one gate over *gap* candidates only) and places them
  frozen-incremental against a robust line-masked baseline, fixing dense-cluster
  collapse; the final joint relax co-fits the real baseline (no contract change).
  Trusts the Stage-3 Blackman-Harris detections. Commit `7b366d4`.
- **Fit-walk performance** — per-window cleanup refits fan across one BLAS-pinned
  fork pool (`parallel_window_refit_map`); `_walk_windows_dag` releases each window
  as its own predecessors converge, replacing the fork-per-level barrier.
  Byte-identical; 655 wall 23:20 → 14:46. Commits `7b366d4`, `e5c0ba9`.
- **VIF-collapse consolidation** — degenerate sub-resolution overfits (including
  singular-covariance coincident pairs the diagonal VIF gate missed) collapse to
  one centroid line carrying a spread-inflated σ_f, iterated to a fixpoint, with a
  footprint guard against folding across a real gap and `vif_collapse_threshold`
  25 protecting resolvable doublets. Commits `eb433e6`, `7774385`, `a1b895b`.
- **Inline per-window cleanup (§§1–4 of [`stage5-inline-cleanup.md`](stage5-inline-cleanup.md))**
  — the SNR-prune + VIF-collapse run in the walk's per-node tail (the global
  post-pass is gone), and a dependent freezes the ancestor window's *current
  fitted lines* (keyed off the Stage-4 DAG edge), not Stage-3 contributor records.
  Dependents fit against clean, current sources by construction. Commits
  `2c0f1f8`, `45f3b83`.
- **Refit reproducibility (C0) + dead-code retirement (§5)** — a single-window
  refit was not a fixed point: `fit_window` cold-started the co-fit leakage-wing
  baseline at zero while warm-starting the peaks, so on giant-skirt windows
  untouched peaks slid up to ~1.2 MHz. Fixed by warm-starting the baseline from
  the persisted coefficients (`initial_baseline_coeffs`, in both
  `refit_window_core` and the in-walk `refit_outcome`); the `freeze_baseline`
  prototype is deleted, the dead global post-pass functions removed. 655
  reproducibility 21→14 movers (worst 1.2 MHz → 84 kHz); output-neutral on the
  automatic fit. Commit `bd51709`, ROADMAP **D16**.

**Why this unblocks the cascade.** A Stage-6 edit refit now reproduces the
originating fit (the "identity = no-op" invariant the gate experiment found
violated), and the in-walk dependency model already resolves a dependent from its
source's *current* fit. So the post-fit cascade is "re-evaluate a dependent from
its source's current fit and re-walk the closure" — not the snapshot-refresh
problem the original design wrestled with.

Measurement instruments and full context: `scratch/cascade/HANDOFF.md`,
`scratch/cascade/BASELINE.md` (the 7-fixture reference, `s4c` arm), and memories
`stage6-cascade-gate-finding`, `stage6-cascade-seeder-increments`,
`stage5-inline-cleanup-recall-regression`.

---

## The problem

A strong line fit freely in its own window contributes its finite-T leakage
**skirt** to neighboring windows as a frozen (non-re-fit) component — a
`FixedContributor`. When the strong line is **edited during Stage 6 curation**,
that edit does not reach the dependent windows. The dependent windows keep the
skirt they were given at Stage 5 fit time.

### What happens today, precisely

The frozen background a dependent window **D** subtracts is a **snapshot**, not
a live reference. At Stage 5 fit time each window persists its own
`fixed_parameters` dict — the contributor's `(frequency, amplitude, phase)`
frozen at the value found in the primary window **W**. Every later operation
(`_reconstruct_frozen_peaks` in `_internal/stage6_impl.py`) rebuilds D's frozen
background **from D's own persisted snapshot**, never from W's current fit.

There are two distinct "rebuild" paths and **neither** propagates an edit:

- **`review run` (decision replay).** Each decision is replayed window-locally
  (`refit_window_core` is explicitly "no cascade"). Editing contributor C in W
  replays onto W only; D is not in the decision log, is never touched, and keeps
  its **pre-edit** snapshot of C.
- **`fit run` (full Stage 5 re-fit).** Wipes Stage 5 and re-fits from the
  *data*. The dependency DAG fits W before D and D reads C live from W's
  converged fit — but that is the **auto** value of C, because the Stage 6 edit
  is not a Stage 5 input. After `fit run` the edit is gone until `review run`
  replays it, and replay again touches W only.

So a Stage 6 edit to a contributor **never reaches the dependent windows by any
path a user currently has.** The file is left internally inconsistent: C's own
line-list entry reflects the edit; every dependent window's *model of C*
reflects the pre-edit value. Because a line is only a contributor when it is
strong (`min_freeze_snr`) and its predicted skirt into D was large enough to
keep, the inconsistency is non-trivial by construction whenever a contributor
exists — the only open variable is how much the edit moved C.

### Documentation discrepancy to resolve

`docs/source/stage6_review.rst` ("Limitations") states: *"Stage 6 reports which
neighbors reference the window but does not re-fit them."* The attention-reason
machinery (`_compute_attention_reasons`, kinds `auto_merged_review`,
`worst_eps`, `overfit_vif`, `candidate_bearing`, `spur_adjacent`,
`edge_boundary`) and the report layer implement **no such neighbor-reference
reporting** — the staleness is silent today. This is a code-vs-doc divergence;
the cascade work resolves it (the doc text changes to describe the cascade).

## The decided design

### 1. Window-level dependency resolution (a baseline change, for the better)

Replace per-peak contributor resolution with **window-level** resolution. A
dependency edge points at a **source window**, not a specific peak. At fit time,
the dependent window includes the skirts of **all fitted peaks in the source
window above `min_freeze_snr`**.

This is the foundation that makes propagation well-defined, and it dissolves the
peak-identity problems:

- **Split** of a strong contributor into C1/C2 → both are above threshold →
  both skirts included automatically (no nearest-frequency ambiguity).
- **Deletion** of the contributor → no peaks above threshold in the source →
  the edge resolves to **zero skirt, harmlessly**.
- **Frequency / amplitude shift** → picked up automatically.
- A **newly strong** peak appearing in the source under a refit (e.g. a rescue
  add) → its skirt is now included, which is correct.

The semantics of *how* a peak changed no longer matter — the dependent always
subtracts "whatever the source window currently fits, above threshold."

This aligns with the existing data model rather than adding to it:
`WindowPlan.dependency_edges` is **already** `(window, depends_on_window)`. The
window-level edge exists; it is the per-peak `FixedContributor.peak_index` that
is the redundant, brittle layer that this change can retire.

**Cost / scope.** This changes the **baseline** contributor evaluation, not just
the cascade. Today's plan is more selective than "all above-threshold peaks": it
applies a predicted-skirt magnitude threshold and an edge-free top-3 cap
(`DEFAULT_MAX_EDGE_FREE_NEIGHBORS`). Switching to window-level resolution will
move baseline fits (possibly not byte-identical) and must be reconciled with the
edge-free / cycle-break path. **Treat it as a foundational Stage-5 change,
validated against the 7-fixture baseline first**; the cascade then rides on top.

**Edge-free contributors are unaffected in spirit.** Cycle-broken edges are
resolved `edge_free` — read from the active FT, not from a fitted neighbor — so
they are **cascade-immune by construction**, and that is correct: the data did
not change, so the skirt should not. Under window-level semantics the edge-free
read still uses the source window's above-threshold peak *frequencies* to know
where to read; amplitude/phase still come from the data.

### 2. Cascade is obligatory, logged, and reversible

The curation layer is a **single level** on top of the automatic baseline — not
iterative versions. State it precisely:

> curated fit = f(baseline, decision_log, dependency_plan), computed by one
> deterministic top-down window-plan walk.

Auto-cascade does not add a concept; it widens the set of windows that one
revision recomputes. When an edit changes a contributor, the walk recomputes the
**transitive closure of the edited windows' descendants** in the dependency DAG,
in topological order, seeded from the baseline everywhere except where edits or
updated upstream skirts intrude. This is `execute_plan` restricted to the
affected subgraph.

- **Obligatory**, because a partially-cascaded file *is* the inconsistent state
  we are eliminating; making it optional reintroduces the danger.
- **Logged as revisions** for every window the cascade actually re-fits (parent
  and dependents), so the change is never silent.
- **Reversible** via re-walk: persist only `(baseline, decision_log)`. The
  curated layer is fully derived; undo = drop the decision entry and re-walk from
  baseline. No per-window pre-cascade snapshots (they would be a redundant second
  source of truth that can drift). A full re-walk is acceptable (minutes);
  subgraph-limited re-walk is a later optimization not worth the bug risk now.

### 3. Composition of cascade refits with direct edits

The curation layer is no longer a set of independent window-local edits; the
cascade couples them. The deterministic rule:

> For each window in the affected closure, in topological order, fit it with
> *(updated upstream skirts)* ∧ *(its own direct user edits)*.

A single topological pass over the union of all directly-edited windows'
descendant closures resolves every ordering question — a window that is both a
cascade target and the subject of a direct edit (E2) is fit once, honoring E2
and the updated upstream skirt together. There is no "who wins" ambiguity
because there is no second pass. The implementation requirement is exactly that:
**one merged topo-ordered pass**, not cascade-refit and direct-edit as two
layered code paths.

### 4. Discovery in the cascade: NLS-only plus one bounded rescue round

`refit_window_core` is NLS-only (no conservative discovery, no rescue, no thaw,
no replan), which keeps the cascade deterministic and bounded. Consequence: a
pure-NLS cascade can shift/shrink a dependent's peaks but cannot *gain* one — so
it cannot recover a line in D that had been masked by the wrong skirt.

Decision: allow **a single residual-rescue round** in a cascade refit *when an
out-of-band contributor changed*. Rescue is deterministic (residual-max seed,
raw-χ² accept), so one round preserves determinism, and gating it on "a
contributor actually changed" keeps it from running everywhere.

**Guard:** rescue must **not re-add a line at a user-suppressed frequency.** User
removals already carry prune-immune provenance; this is the symmetric guard, or
an edit on W could resurrect, via a rescue in D, a peak the user explicitly
deleted in D. Rescue-added lines carry their own provenance and surface through
the peak-count flag (below).

### 5. Thawed-peak ownership folds into the same fix

`refit_window_core` holds **thawed** peaks (owned by a primary, co-fit into D)
frozen at their persisted values. A naive cascade would refresh the frozen
*skirt* but leave a *thawed* peak stale — the same inconsistency in the co-fit
channel. The topo-walk must refresh both channels from the re-fit primary. This
is not extra scope; it is the same defect (latent in the single-window model)
wearing a co-fit hat, fixed by the same mechanism.

### 6. Attention flags: surface only scientifically significant changes

The cascade report flags a dependent window for a second look — deliberately
**not** every re-fit window, since σ is typically ≪ bin size and small shifts
are invisible by eye. Two families:

- **Result changed.**
  - A peak's parameter moves by more than *k* × **max(σ_before, σ_after)** for
    that parameter (max, not a quadrature "mutual" of two highly-correlated
    fits — same data, slightly different background → ρ≈1, and quadrature
    inflates the bar in a way that is hard to reason about). "Did the reported
    number move by ≳ *k* × its stated precision" matches the downstream
    consumer's experience. *k* (2, 3, …) is **tunable after seeing the
    implementation** — pick what reads as scientifically significant.
  - A parameter's **uncertainty inflates** by more than ~2× (the fit became
    ill-conditioned; also catches a shift that only looks small because σ blew
    up).
  - **Peak-count change** in a dependent (under NLS-only this can only decrease —
    a collapse — except where the bounded rescue round adds one).
- **Fit quality changed.** Flag on χ²ᵣ change, **DOF-gated**: a 20% χ²ᵣ change is
  overwhelming on a high-DOF window and pure noise on a 4-DOF one (χ²ᵣ sampling
  spread ≈ √(2/DOF)). Use `|Δχ²ᵣ|/χ²ᵣ > max(20%, c·√(2/DOF))`, or equivalently a
  change in χ² significance. A χ²ᵣ *drop* can be legitimate (a better skirt gives
  a more precise determination), so this replaces a naive "σ deflation" rule on
  that side.

**Development aid:** expose the ability to **rank peaks / windows by
baseline-vs-curated difference** (the same metrics above). At least during
development this is how we find where the "scientifically significant" line
should be drawn before fixing thresholds.

### UX notes

- Even though the cascade is obligatory, an **edit-time hint** ("this line feeds
  windows [X, Y]; they will be re-fit") is cheap orientation, not a gate.
- **Obligatory bundling** means a user cannot keep an edit while rejecting its
  cascade consequence. If the cascade makes a dependent worse (flagged via
  χ²ᵣ), the recourse is to undo the original edit or to **directly curate the
  flagged dependent** — which is consistent with the model (the dependent then
  becomes a directly-curated window).

## The build

### Scope (measured) — wide-shallow, mostly local

`scratch/cascade/cascade_scope.py`, exact from each window's persisted
`fixed_parameters[*].primary_window_id` (identical to `_build_preds` restricted to
fitted windows):

| fixture | fitted | sources (edit cascades) | leaves (edit local) | worst blast |
|---|---|---|---|---|
| 2638 | 266 | 0 | 100% | 0 |
| 363 | 401 | 0 | 100% | 0 |
| 360 | 243 | 3 | 98.8% | 1 |
| 1231 | 207 | 4 | 98.1% | 2 |
| 1019 | 64 | 3 | 95.3% | 12 |
| 1512 | 86 | 5 | 94.2% | 3 |
| 655 | 716 | 15 | 97.9% | 42 |

94–100% of edits cascade to nothing; the big blasts are hubs (`w1006` feeds ~41
leaves in one antichain). On the rebuilt `s4c` arm the DAG is **maxhop=1** — every
closure re-fits in a single wave. Serial re-walk is trivially adequate.

### Everything is already in place except the dependent refresh

- **Refit unit** — `refit_window_core` (`stage6_impl.py`): NLS-only, reproducible
  (an identity refit is a fixed point), already drives every edit verb.
- **Dependency linkage** — each `FittingResult.fixed_parameters` carries
  `primary_window_id` per frozen contributor (and `peak_index=-1`: the in-walk path
  already writes window-level content, the ancestor's current above-threshold
  fitted lines). Reverse map (`succs`) + closure/topo are ~10 lines
  (`cascade_scope.py` prototypes them).
- **Two-state replay** — `review_undo_impl` (`stage6_impl.py:2989`) already does
  `_restore_stage5_baseline` → rebuild review (`review_run_impl`) → replay the
  decision log via `_execute_planned_action`. `apply_curation_impl` shares that
  replay engine. This *is* the "automatic + revised, re-derive from baseline on
  every mutation, no multi-level undo" model — so reversibility, determinism, and
  "undo all → exactly the automatic fit" come for free.

**The one new piece: the window-level dependent refresh.** A plain refit of a
dependent reconstructs its frozen background from its *own* persisted snapshot
(`_reconstruct_frozen_peaks`) → identity fixed point → **a no-op**. To propagate an
edit, each dependent must first rebuild its frozen background from its sources'
*current* fits — "all source peaks ≥ `min_freeze_snr`", the window-level semantics
`evaluate_ancestor_leakage` already uses in-walk. Prototyped and validated as
`refresh_frozen_wl` in `scratch/cascade/cascade_lab_wl.py` (~30 lines). This is the
foundation the decided design §1 calls for, but it costs **no Stage-5 re-baseline**:
it touches only the post-fit refresh, and the persisted format already stores
window-level content. It handles shift, add, remove, split, and delete uniformly
(remove → fewer source peaks above threshold; split → both children above
threshold), dissolving the nearest-match aliasing artifacts the per-peak refresh
hit.

### The build, in steps (one `_internal` change behind the dual-interface invariant)

- **B1 — reverse map + closure.** `succs: primary_wid -> {dependent_wid}` and the
  topo-ordered transitive closure of a set of edited windows, from
  `fixed_parameters` (lift `cascade_scope.py` into `_internal`). No new persisted
  structure.
- **B2 — window-level dependent refresh.** Productionize `refresh_frozen_wl`:
  rebuild a window's frozen contributors (and the thawed-peak channel, §5) from its
  source windows' current above-`min_freeze_snr` fitted lines. Reuses the
  `evaluate_ancestor_leakage` rule; replaces the per-peak snapshot read inside the
  cascade path.
- **B3 — `cascade_refit(edited_wids)`.** Topo-walk the merged closure of all
  `edited_wids`; for each window in order: refresh (B2) from current sources,
  identity-`refit_window_core`, splice back. A directly-edited window already
  carries its own edit in its peak set (applied in the replay phase), so the
  identity refit honors both the edit and the refreshed skirt in one fit (§3) — no
  second code path. The lone bounded rescue round (§4) only when an out-of-band
  contributor changed; it must not re-add a user-suppressed frequency.
- **B4 — fold into the replay.** Route every mutation through the existing
  restore-baseline → replay path, then run **one** `cascade_refit` over the union of
  all replayed edits' closures (§3). The interactive verbs append their decision and
  re-derive; undo drops the decision and re-derives. The decision log stays
  user-edits-only — the cascade is *derived*, never logged as separate undoable
  units, so "undo all → automatic" holds by construction.
- **B5 — surface (after correctness lands).** Attention flags (§6) comparing revised
  vs baseline, exposed via `review rank --by <diff-metric>`; the cascade section in
  `review show` / the HTML report; the edit-time "feeds windows [X,Y]" hint. Update
  the `docs/source/stage6_review.rst` "no re-fit" limitation and the ROADMAP D-row.

Validated against the 7-fixture `s4c` reference (`scratch/cascade/BASELINE.md`):
because the automatic fit's in-walk path already self-consistent-freezes sources,
**an un-edited file re-derives byte-identically** — the cascade only changes a file
once a user edits it. That is the primary regression gate.

## Experiment (the gate) — DONE, decisive

Ran on the reproducible `s4c` baseline with the window-level harness
(`scratch/cascade/cascade_lab_wl.py`). At the worst-case hub (655 `w1006`, 41
dependents) and 1512 ¹⁴N doublets (`w45`/`w192`), a contributor edit moved **0
dependents ≥0.5σ (worst 0 kHz)** for identity, insidious satellite add/remove, and
structural split; it fired materially (5/41, 469 kHz) only on the drastic deletion
of the dominant strong line. Far-field skirt = total power + centroid, invariant
under split/merge — so the cascade is a **correctness/honesty fix with rare
practical bite**. This reverses the pre-C0 finding (whose MHz propagation was the C0
baseline bug compounded with the per-peak nearest-match aliasing artifact, both now
resolved). Full writeup: `scratch/cascade/EXPERIMENT_FINDINGS.md` (headline block).

## Interface surface (to design during build)

All three interfaces must stay identical (the dual-interface invariant). The
cascade is internal to the edit verbs (`review edit/merge/split/accept` and
`review apply`) — it is not a new user verb; editing a contributor simply
re-fits more windows and records more revisions. New/changed surface to specify:

- The **ranking** development aid (likely `review rank --by <diff-metric>`,
  extending the existing `review rank`).
- The **cascade report** section (which windows were re-fit, which flagged, and
  why) in `review show` / the HTML report.
- Decision-log provenance for cascade-induced refits and rescue-added lines,
  distinct from direct user edits.

## Serialization

- Retire / deprecate `FixedContributor.peak_index` as the resolution key in
  favor of the window-level edge (keep the edge list; resolve dynamically).
- No per-window pre-cascade snapshots. Persisted truth stays `(baseline,
  decision_log)`; the curated fit is derived.
- Window-status / decision-log entries gain cascade provenance and the new
  attention-flag kinds.

## Test plan

- Cross-interface consistency tests for any verb whose behavior changed.
- A determinism test: the curated fit is byte-stable under re-walk (same
  baseline + decision log → identical result), including undo → re-apply.
- Composition test: a direct edit on a window that is also a cascade target is
  honored exactly once (single topo pass).
- Guard test: a cascade rescue round does not re-add a user-suppressed line.
- The window-level resolution change re-baselines the 7-fixture suite; the new
  baseline is the reference (document the deltas, expect small).

## Open questions (do not relitigate without a factual contradiction)

- Final flag thresholds (*k* for the shift, the χ²ᵣ gate constant) — tune after
  the experiment.
- Whether the predicted-skirt magnitude threshold and the edge-free top-3 cap
  survive unchanged under window-level resolution, or need re-tuning — settle
  from the 7-fixture re-baseline.
- Whether the single bounded rescue round is enough, or whether any structural
  edit needs more — settle from the experiment's structural-edit case.
