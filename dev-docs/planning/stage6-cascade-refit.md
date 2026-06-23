# Stage 6 — contributor-edit cascade (window-level dependency resolution)

Status: **planning — gated on a prerequisite dependency-model investigation
(below).** This document records the design worked through for propagating a
Stage 6 curation edit from a contributor line into the windows that depend on
it, and the foundational change to how a window's frozen background is resolved
that makes the propagation well-defined. A characterization pass (see
"Prerequisite finding") surfaced that the dependency model is **inconsistent
across SNR** — the densest fixtures, which have the *most* contributor coupling,
end up with their entire dependency graph dropped and every contributor
demoted to edge-free, while moderate fixtures keep the edge-bearing machinery.
That must be understood and resolved before the cascade design (which presumes a
meaningful dependency graph to propagate along) is built. The normative
requirements remain in the `*_STRATEGY.md` specs; this plan is normative only
for the work it tracks.

Related: [`stage6-finalization.md`](stage6-finalization.md) (the read/edit
surface this builds on), [`stage6-peak-survival.md`](stage6-peak-survival.md)
and [`stage4-leakage-contributor-subtraction.md`](stage4-leakage-contributor-subtraction.md)
(the contributor / frozen-skirt machinery), and the documented limitation in
`docs/source/stage6_review.rst` ("No automatic cascade").

## Prerequisite finding: the dependency model is inconsistent across SNR

Characterizing the contributor / dependency structure of fresh and recent
fixtures (`scratch/cascade/characterize.py`) surfaced a latent inconsistency
that must be resolved before a cascade is meaningful. A dependent window only
goes stale on a *parameter* (amplitude/phase) edit when its contributor is
**edge-bearing** (reads the primary's *fitted* skirt). An **edge-free**
contributor reads (amp, phase) self-contained from the active FT (data), so it
is immune to a fit edit and self-consistent with the data by construction; a
**thaw** co-fit would hold a joint-optimum copy that does go stale.

Measured structure (windows / dependency_edges / edge-bearing / edge-free):

| Fixture | windows | edges | edge-bearing | edge-free |
|---|---|---|---|---|
| 360  | 331  | 114 | 234 | 1097 |
| 2638 | 375  | 73  | 130 | 265  |
| 363  | 486  | 62  | 119 | 1556 |
| 1231 | 305  | 63  | 115 | 798  |
| 1019 | 88   | 0   | 0   | 315  |
| 1512 | 267  | 0   | 0   | 962  |
| 655  | 1241 | 0   | 0   | 8372 |

The three **extreme-SNR** fixtures (655, 1512, 1019) — which by density have the
*most* contributor relationships — end up with **zero** surviving dependency
edges: every contributor is edge-free. On the fresh 655, the cycle-breaker
(`window_planning.py` Step 7) **dropped 9 523 cyclic dependency edges** and
converted the dominant orphaned contributors per window to edge-free (8 063
total), dropping the rest. So the densest spectra are fit as **independent
windows**, while the moderate fixtures (2638/360/363/1231) retain the
edge-bearing fit-ordering machinery. That is the inconsistency: the cases with
the heaviest coupling get the *least* sophisticated treatment.

**Thaw is dormant.** Thaw (the co-fit handshake intended to resolve coupling) is
gated to edge-bearing contributors only (`thawable = [fp for fp in fixed_peaks
if not fp.edge_free]`, `plan_execution.py:1373`), so once the cycle-breaker has
made everything edge-free there is nothing to thaw. On the fresh 655, all 132
thaw attempts reject with `reason='no frozen contributor on the flagged edge
side'`; thaw-accept is **0 on every fixture checked** (655, 2638, 360). Thaw may
be effectively dead code in production.

**Hypothesis (to test):** the cycle-breaker is too aggressive — dropping all
cyclic edges and demoting to edge-free pushes leakage-skirt modeling onto the
complex/leakage-wing **baseline**, which on the dense fixtures runs at order 4
in ~99% of windows and may be modeling leakage skirts that contributors should
carry. (The earlier visual impression that 655 was "deeply pathological" was a
display bug — the magnitude panels plotted a misaligned padded FT that erased
narrow lines — now fixed and committed. With a trustworthy display, 655 reads
healthy in bulk; the open question is the baseline-vs-contributor one, not a
broken fit.)

## Investigation plan

Done so far: fresh 655/1512/1019 built and reported; contributor structure
characterized (the SNR table above); the display bug found and fixed; the
faithful skirt A/B prototyped. The dependency-model question itself is **open**
and is the next session's work.

1. **Baseline-magnitude diagnostic (build this).** When a contributor skirt is
   modeled physically, the leakage-wing baseline should be a *small correction*,
   not a data-scale term. So compare each window's fitted baseline magnitude
   (and its slope) against a robust estimate of the average and slope of the
   real/imaginary data — a baseline whose magnitude is the same order as the data
   marks a **leakage-touched window that should be coupled** to a neighbor. Use
   this to map which windows the baseline is silently carrying (an investigation
   instrument, not necessarily a production flag).
2. **The contributor-model fix + cascade test on 1019 w79/w80** (the clean
   experiment). 1019 w79 has a strong peak over-split into a doublet (a merge in
   curation); w80 next to it has a weak feature riding w79's skirt, currently
   modeled by baseline alone (edge dropped → `ncon=0`). Fix the contributor model
   (window-level resolution) so w80 carries w79's fitted line(s) as contributors,
   then: merge w79's doublet and confirm the change *cascades* into w80's skirt
   and moves its fit. This is the decisive "does the dependency matter" test on a
   real, defensible curation decision.
3. **Decide the resolution**, consistent across SNR:
   - **Better cyclic-dependency resolution** than "drop all edges" — break the
     minimal feedback arc set, or resolve dense cycles with a real (accepted)
     thaw co-fit rather than edge-free demotion.
   - **Declare dependencies unnecessary** — if the baseline genuinely models the
     leakage adequately (the baseline-magnitude diagnostic says it is small),
     retire contributor edges uniformly so moderate fixtures match the dense ones.

Only once the dependency model is settled does the cascade design below apply:
the cascade propagates edits *along the dependency graph*, so its value and even
its existence depend on whether that graph should exist.

### Handoff state (scratch harnesses, reusable)

Built this session under `scratch/cascade/` (gitignored): `build_655.py
<name>` (fresh fixture build via `run_pipeline`), `characterize.py` (edge-free
vs edge-bearing contributor structure + closures), `pathology_scan.py`
(per-window baseline order + post-fit residual edge-coherence), `ab_skirt.py`
and `ab_faithful.py` (skirt-vs-baseline A/B; the *faithful* one restores a
dropped contributor into the persisted plan as edge-free and re-runs the
production fit so the evidence gate decides), `show_w24.py` (native Re/Im/|X|
decomposition). Fresh fixtures: `scratch/cascade/{655,1512,1019}.ftmw` and their
reports under `scratch/cascade/reports/`. Rebuild fresh per the rebuild-fixtures
rule if code changes (a stale fixture carries build-time settings).

### Deferred follow-up (diagnostic surfacing)

Independent of the resolution decision, the investigation showed these are
currently invisible to a reviewer and should be surfaced — at least the
**out-of-band contributor count** and the **leakage-wing baseline order** — in
both `fit show` and a prominent place in the HTML report. A window whose edge is
dominated by a neighbor's leakage wing modeled by an order-4 polynomial baseline
with zero contributors reads as "clean" today; the count + order would expose
that. Relatedly, the per-window panels should plot the native-bin values as
authoritative markers (not only an interpolated curve), so a reviewer reads the
measured spectrum directly. (Design thoughts pending; tracked here so it is not
lost.)

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
