# Stage 6 — contributor-edit cascade (window-level dependency resolution)

## Status

The cascade's prerequisites are all shipped and committed on
`stage6-cascade-refit`. **The cascade proper — propagating a Stage-6 contributor
edit into its dependent windows (§§C/D) — is the remaining work.** It is
specified below in "The decided design" and "Cascade scope and the staged build
order"; **start there.** Everything above that is the implementation summary of
the prerequisites and the original motivation.

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

## Cascade scope (measured) and the staged build order

The decided design (sections 1–6 above) frames window-level resolution as an
obligatory foundation that must land and re-baseline before anything. Measuring
the actual blast radius on the seven fixtures changes the staging calculus.

**Measured scope** (`scratch/cascade/cascade_scope.py`, exact from each window's
persisted `fixed_parameters[*].primary_window_id`; confirmed identical to the
plan-derived `_build_preds` restricted to fitted windows):

| fixture | fitted | sources (edit cascades) | leaves (edit local) | DAG depth | worst blast |
|---|---|---|---|---|---|
| 2638 | 266 | 0 | 100% | 0 | 0 |
| 363 | 401 | 0 | 100% | 0 | 0 |
| 360 | 243 | 3 | 98.8% | 1 | 1 |
| 1231 | 207 | 4 | 98.1% | 1 | 2 |
| 1019 | 64 | 3 | 95.3% | 1 | 12 |
| 1512 | 86 | 5 | 94.2% | 1 | 3 |
| 655 | 716 | 15 | 97.9% | 4 | 42 |

The cascade is **wide-shallow, not deep**: 655's big blasts are hubs (`w1006`
feeds 42 dependents, `blast == direct` — those 42 are leaves), with only a small
`w1008–w1014` cluster (fan-in 4) needing multi-hop recursion. 94–100% of edits
cascade to nothing.

**What's already in place** (the cascade is mostly assembly):

- `refit_window_core` (`stage6_impl.py`) — the NLS-only per-window refit unit
  driving the edit verbs, now **reproducible** (the C0 baseline warm-start): an
  identity refit is a fixed point, so a cascade refit changes a dependent only
  because its source's skirt actually changed. (The in-walk cleanup drives the
  outcome-native twin `refit_outcome`; the cascade uses `refit_window_core`.)
- Each `FittingResult.fixed_parameters` holds its frozen contributors tagged with
  `primary_window_id` **and** a freq/amp/phase snapshot — both the exact
  dependency linkage and the stale state to refresh. `_reconstruct_frozen_peaks`
  reads it (today from the snapshot — the thing the cascade replaces; the in-walk
  path already reads the ancestor's current fit, per §4 of the inline-cleanup).
- Edit verbs (`merge_peaks_impl` / `split_peak_impl` / `review_accept_impl`, the
  add/remove refit) each append to `decision_log`, refit **one** window, persist.
  `apply_curation_impl` replays a batch one-window-at-a-time.
- `_walk_windows_dag` + `_build_preds` (`plan_execution.py`) — the dependency-
  ordered scheduler; the cascade re-walk reuses its indeg-gated loop.

**The build, in stages** (each a real `_internal` change behind the dual-interface
invariant, validated against the 7-fixture reference in `scratch/cascade/BASELINE.md`):

- **C1 — reverse dependency map.** `succs: primary_wid -> {dependent_wid}` from
  `fixed_parameters` (what `cascade_scope.py` already builds). No new persisted
  structure.
- **C2 — the cascade walk** `cascade_refit(edited_wids)`: closure = transitive
  descendants over `succs` ∪ the edited windows; topo-order the closure; for each
  window in order rebuild its frozen background **and** thawed-peak channel
  (section 5) from its primaries' *current* fits, apply its own direct edits,
  `refit_window_core`, splice back, record a revision. Reuse the
  `_walk_windows_dag` loop restricted to the closure (the closure is small — the
  42-blast is a single antichain that re-fits in one wave; serial is also fine).
- **C3 — hook into the edit verbs.** Each verb, after its direct edit, calls
  `cascade_refit({edited_wid})`. `apply_curation_impl` collects **all** directly-
  edited window ids and does **one merged** cascade walk over their combined
  closure (section 3: a window that is both a cascade target and directly edited
  is fit once, honoring both).
- **C4 — provenance / reversibility / flags.** Persisted truth stays
  `(baseline, decision_log)`; the curated fit is derived by replaying the log,
  each replay triggering its cascade; undo = drop entry + re-walk (section 2).
  Log entries gain cascade + rescue-add provenance; attention flags (section 6)
  surface via `review rank --by <diff-metric>`. Guard: the one bounded rescue
  round must not re-add a user-suppressed frequency (section 4).

**The staging refinement the scope unlocks.** Window-level resolution (section 1:
retire `peak_index`, resolve "all source peaks above `min_freeze_snr`") is needed
**only** for edits that change peak *identity* — split (1→2 contributors) and
delete (1→0), where `peak_index` resolution breaks. The exact linkage already
lives in `fixed_parameters` and refreshing a dependent is just re-reading its
primaries' current fits, so the cascade **core** (refresh + re-walk the closure)
is buildable on the *existing* per-window linkage with **no baseline change** for
the common **shift** edit class. So:

1. **Experiment first** (the gate below): on 655 hub `w1006` (42-blast) and a 1512
   ¹⁴N doublet, run an insidious (blend-satellite) and a structural (split) edit;
   measure Δfreq/σ, Δχ²ᵣ, peak-count decay per hop. Confirms the bounded,
   fast-decaying, mostly-single-hop hypothesis before any production wiring.
2. **C1–C4 on the existing linkage, shift-class edits** — no re-baseline, validated
   against the reference table. Most of the value, most of the safety.
3. **Window-level resolution (section 1) + split/delete cascades** — isolated,
   re-baselined, reconciled with the edge-free / cycle-break path — only where
   structurally required, **not** as an upfront gate on everything.

This inverts the decided design's "foundation first" into "cheap correct core
first, foundational change only where peak identity changes" — which the scope
(98% of edits cascade to nothing; the dependency graph is already exact and
minimal in `fixed_parameters`) makes safe.

## Experiment plan (gates the build)

Build a scratch harness that implements **window-level resolution** and a
cascade so we measure the magnitude/decay *and* validate the foundational
semantics before either touches Stage 5. There is no cascade entry point today,
and `_reconstruct_frozen_peaks` reads the stale snapshot, so the harness must
rebuild a dependent's frozen background from the source window's current
above-threshold peaks and re-fit.

Two fixtures, two purposes (build **fresh** per the rebuild-fresh rule):

- **655 → closure size / fan-out (the cost question).** Dense, high-SNR, many
  strong contributors with real dependents and a known leakage pedestal — the
  worst case for cascade radius. Pick the strongest line that is a contributor
  to ≥1 window (rank by predicted skirt into the dependent so an effect is
  visible) and measure how many hops the cascade reaches before every flag goes
  quiet. Confirms the "obligatory" cost is bounded (the 1–2-hop hypothesis).
- **1512 → scientifically-defensible structural edit (the magnitude question).**
  The ¹⁴N hyperfine doublets give a physically motivated split/merge on a strong
  line that is also plausibly a contributor — a *reasonable* curation decision,
  not a contrived one.

For each, run two edit classes on the chosen contributor's window W:

- **(a) Insidious / indirect.** Add or remove a *blend satellite* in W, leaving
  the strong line nominally alone, to see how much the contributor moves
  *indirectly* (joint-NLS correlation) and whether that trips a dependent.
- **(b) Structural.** Split the strong contributor — the upper bound.

Per hop record: Δfreq / σ_f, σ_after / σ_before, Δχ²ᵣ, peak-count change. Track
the decay across hops.

**Headline to watch for.** If even a structural split of a strong contributor
moves its top dependent by ≪ σ_f, the feature is a correctness/honesty fix with
rare practical bite (still worth doing — the file should never lie about its own
consistency) rather than a frequently-material recomputation. Either result is
useful and informs how much machinery is justified.

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
