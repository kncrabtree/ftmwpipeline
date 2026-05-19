# Plan: Stage 4 — Window assignment

Status: **planning** (not started). Scope is Stage 4 only — *classify and
propose*, do not fit (fitting is Stage 5). Registered in
[`../ROADMAP.md`](../ROADMAP.md).

Normative requirements remain in the `*_STRATEGY.md` specs; this document is
normative only for the Stage 4 work it tracks. It builds directly on the
finalized Stage 3 → Stage 4 contract in
[`stage3-peak-detection.md`](stage3-peak-detection.md).

## Objective

Turn the promoted Stage 3 peak list into a **fit plan**: a set of analysis
windows over the persisted user spectrum, each annotated with the peaks to fit
freely, the strong out-of-band lines whose leakage must be carried as a fixed
background, a fit dependency order, and a difficulty classification so Stage 5
can parallelize the easy windows and spend its budget on the hard ones. Stage 4
is purely structural: it makes no fits and changes no spectrum.

## Reuse map

Per the locked Stages 3–5 design, the refined `newfitting/` engine — which
held the *adaptive window selection* and *peak aggregation* — is permanently
lost. Stage 4 is therefore **recreate**, not port. The surviving
`~/github/bcfitting/src/bcfitting/ftmwfitting.py` contributes only philosophy
(predict extent from the strongest line, greedily expand while leakage is still
above baseline, sanity-check edges) and the analytic finite-T sinc-leakage
*model* (reused as the leakage-reach predictor, already ported in Stage 3 as
`preprocessing/leakage.py:estimate_leakage_reach`). No `newfitting` contract is
available to recreate against; the contract is defined here.

## The invariant (decided)

Two distinct notions, do not conflate them:

- **Fit window** — the contiguous frequency span whose points enter *that
  window's* least-squares residual. Fit windows are **disjoint and cover each
  spectrum point at most once**. This is the hard invariant (it guarantees no
  point's residual is double-counted and no peak is fit twice).
- **Contributor set** — the peaks whose model terms are *evaluated* over a
  window: **in-band/free** (the window's own peaks, free A/f/φ in Stage 5) and
  **out-of-band/fixed** (strong lines fit in *their* own window, parameters
  frozen, contributing only their leakage skirt). Contributor sets overlap
  across windows by design; that is how a strong line's leakage is represented
  everywhere without re-fitting it.

Guardrail: a fixed contributor is the strong line's **finite-T damped-cosine
term with frozen parameters**, evaluated in the dependent window's model
(consistent with the locked Stage 5 LS contract: residual = complex FFT(model)
vs complex FFT(window)). It is **not** analytic-sinc subtraction from the data
— the discarded `iterative_peak_subtraction` path is not revived.

Consequence: Stage 4's output is **not a flat partition**. It is an ordered set
of fit windows, each carrying `(free peaks, fixed out-of-band contributors)`
and a position in a fit **dependency DAG** (a window depends on the windows
that fit its fixed contributors). Stage 4 *classifies and proposes*; Stage 5
fits and **may revise** the separation — for strongly coupled regions the ideal
split may only be determinable at fit time.

## Algorithm (sketch — details are open research)

Inputs: the promoted peaks (`Peak.properties['promoted']`) on the persisted
user spectrum; the canonical Stage 2 noise (per-point RMS on that grid);
`estimate_leakage_reach` (SNR, acquisition `T`, assumed `τ`); and the
**complex** user spectrum (real and imaginary parts, not just magnitude — the
edge test below is a complex-domain coherence test and magnitude discards the
phase information it depends on).

1. **Propose extent.** For each strong line, predict its leakage reach from the
   analytic finite-T model; the proposed window spans the line ± reach.
2. **Validate/trim edges (complex-domain coherence test).** Leakage is
   phase-coherent; a leakage-free edge band is zero-mean white noise. Over an
   M-point edge band test `|Σ_edge complex| / (σ_local·√M)`: O(1) ⇒ clean edge
   (trim to here); ≫1 ⇒ coherent leakage still bleeding in (extend, and the
   window has an out-of-band contributor — feeds step 6). Model-predicted
   reach *proposes* the extent; this statistic *confirms/trims* it. A
   magnitude `k·RMS` floor is **not** sufficient (a sinc skirt can dip to
   near-noise in magnitude while still fully coherent). DC-offset-at-edges is
   the zero-distance degenerate case of this test. See O4-1.
3. **Strong clusters.** Strong lines whose reaches mutually overlap form a
   single **primary joint window** (they must be fit together first; none can
   be a fixed background for the others).
4. **Merge to fixpoint.** Overlapping proposed fit windows merge transitively,
   deterministically ordered (by frequency, then descending strength), until
   stable → disjoint fit windows covering each point ≤ 1.
5. **Assign in-band peaks, pruning leakage artifacts.** A promoted peak is a
   free peak of the unique fit window containing its frequency **only if it is
   not attributable to a contributor's leakage**. Stage 3 promotes a strong
   line's own sidelobes as peaks; once that line is a contributor (free
   in-band or fixed) its leakage explains them, so they must not also be fit
   as independent lines. Pruning uses the analytic leakage envelope of the
   window's strong contributor(s); the residual after the strong term is what
   defines genuine free peaks.
6. **Attach fixed contributors.** For each window, a freeze-eligible strong
   line in-band of a *different* window is attached as a fixed contributor
   (reference to its primary window) when its predicted leakage reaches here
   **and** the step-2 complex-edge test confirms coherent out-of-band
   structure. Analytic reach proposes the candidate; the complex-edge test
   confirms it. This adds a dependency edge.
7. **Classify difficulty (strong-line-driven, empirical).** Difficulty is
   *not* a promoted-peak-count threshold — that count is dominated by a strong
   line's leakage artifacts and is unreliable. A window is **hard** if it
   contains or is materially influenced by a strong line (has strong in-band
   peaks, or unresolved fixed contributors, or fails the complex-edge test),
   or exceeds the width cap, or sits in a comparably-strong coupled cluster
   with no dominant line to freeze. Everything else is **easy/independent**.
   The only count-like cap is *width*; "too many peaks" is replaced by
   "contains/near a strong line". For hard windows Stage 4 emits a *proposed*
   split (at a complex-edge-clean interior point, flagged as an approximation
   that knowingly cuts shared leakage) **and/or** a `needs_joint_treatment`
   marker. Stage 4 does not choose; it annotates.
8. **Emit the plan.** Topologically order the dependency DAG; independent
   windows form parallel batches. Each window carries: freq range, free peaks,
   fixed contributors (peak id + primary window id), difficulty class, batch
   id, and diagnostics (predicted vs trimmed extent, cap hits, split proposal).

## Data structures

The existing `core.data_structures.SpectralWindow` is close but insufficient:
it has `freq_range`/`peaks` but no free-vs-fixed contributor split, no
dependency edges, no difficulty class, no parallel-batch grouping. Stage 4
needs either an extended `SpectralWindow` or a new `WindowPlan` aggregate:

- per window: `window_id`, `freq_range`, `free_peaks` (refs into the Stage 3
  list), `fixed_contributors` (peak id + owning `window_id`), `difficulty`
  (`easy` | `hard`), `batch` (parallel group), `split_proposal`
  (optional), diagnostics.
- plan-level: the dependency DAG / topological order; the parallel batching.

`FittedPeak` already exists for Stage 5; fixed-contributor parameters become
known only after the contributor's primary window is fit (hence the ordering).

## Interface surface (dual-interface rule)

Logic in `_internal/stage4_impl.py`; thin identical wrappers:
`Pipeline.assign_windows()` / `api.assign_windows()` / CLI `assign-windows`,
plus `visualize-windows` and `load_windows()`. Consumes the **promoted** peaks
only. Parameters (documented defaults, configurable): `max_window_width_mhz`,
`max_free_peaks`, noise-floor `k`, freeze-eligibility threshold, assumed `τ`
for reach. Stage tracking: add `stage4_windows` to
`PipelineStageTracker.STAGE_DEPENDENCIES` (depends on `stage3_peaks`) and
`STAGE_DATA_PATHS`; it is then automatically invalidated by the existing
canonical-settings-change mechanism.

## Serialization

The window plan is curation/coordination substrate (like the peak list), not a
heavy derived array, so it is **persisted** under `/stage4_windows` (flat,
hand-editable, loud validation), not recomputed on demand — consistent with the
SERIALIZATION spec's treatment of peaks. Round-trip + hand-edit contract as in
peak serialization. Confirm against `SERIALIZATION_STRATEGY.md` (O4-6).

## Open research / questions

- **O4-1 Complex-edge baseline test (primary mechanism).** Statistic
  `|Σ_edge complex|/(σ_local·√M)`: calibrate the band width M and the
  threshold against synthetic clean-vs-leakage edges, then 2638. Sub-research:
  discriminate a *global* baseline/DC offset (flat across the band) from a
  *distance-dependent leakage envelope* (varies with proximity to the strong
  line) — both inflate the edge integral but their spatial signature differs.
  This subsumes the old `k·RMS`/DC-offset ideas.
- **O4-2 Freeze-eligibility + error-propagation guard.** Criterion for
  "strong/known enough to freeze". The empirical "does this window have a
  fixed contributor" question is answered by the O4-1 complex-edge test; what
  remains is "frozen background good enough vs must be thawed and re-fit"
  (partly Stage 5).
- **O4-3 Dense-cluster cap policy.** Width cap value (peak-count cap is
  dropped — Stage 3 promoted counts are leakage-artifact-dominated and
  unreliable; difficulty is strong-line-driven per algorithm step 7).
  Split-at-complex-edge-clean-point vs escalate-to-Stage-5 marker; how the
  approximation is logged. The leakage-artifact pruning of the free set
  (step 5) is itself research: how cleanly the analytic strong-line envelope
  removes its own detected sidelobes.
- **O4-4 Strong-cluster detection.** Robustly grouping mutually-reach-
  overlapping strong lines into one primary joint window and bounding its size.
- **O4-5 Determinism.** Merge order and tie-breaking must yield a reproducible
  partition.
- **O4-6 Persist vs recompute** the window plan (lean: persist).
- **O4-7 Stage 4 ↔ 5 renegotiation.** Stage 5 may find the proposed separation
  inadequate (coupling only visible at fit time). Define the handshake: Stage 5
  emits a re-plan request (merge/split) and Stage 4 exposes a re-plan entry
  point, vs Stage 4 over-provisioning hard windows up front.

## Downstream context (Stage 5, not in scope)

Stage 5 consumes the ordered plan: easy/independent windows fit in parallel;
hard windows get the conservative add-one-peak-with-significance loop and may
trigger joint/iterative treatment or thaw a fixed contributor. Fixed
contributors enter each window's model as frozen finite-T damped-cosine terms
(not data subtraction). The Stage 4↔5 renegotiation protocol (O4-7) is the main
cross-stage risk and is settled when the Stage 5 plan is written.

## Test plan

- Synthetic spectra (known A/f/φ/τ) for: isolated strong line; weak line on a
  strong line's skirt (must become a fixed contributor, not free); two strong
  lines with overlapping reach (one primary joint window); a dense comparably-
  strong cluster (width cap triggers a flagged split).
- Complex-edge statistic (O4-1): on synthetic edges with vs without an
  out-of-band coherent leakage tail, the statistic must separate the two
  (≈O(1) clean vs ≫1 with leakage) across SNR/τ; and a flat global DC offset
  must be distinguishable from a distance-dependent leakage envelope.
- Leakage-artifact pruning (step 5): a strong line's promoted sidelobes are
  excluded from the free set once it is a contributor; genuine nearby weak
  lines are retained.
- Invariant checks: fit windows disjoint and cover each point ≤ 1; every
  retained free peak in exactly one window's free set; dependency graph
  acyclic; topological order valid; batches independent.
- 2638 real data: sane window count; the strong doublets anchor windows; dense
  regions are flagged, not exploded; promoted-only consumption.
- Cross-interface identity (CLI/Pipeline/api); serialization round-trip +
  hand-edit; invalidation on Stage 1 canonical-settings change and on Stage 3
  re-detection.

## Task breakdown

1. [ ] Window-plan data structures (extend `SpectralWindow` or new
   `WindowPlan`) + unit tests.
2. [ ] Complex-edge coherence statistic + extent prediction/trim (O4-1) +
   unit tests (synthetic clean-vs-leakage; DC vs envelope).
3. [ ] Strong-cluster grouping + merge-to-fixpoint (O4-4, O4-5) + unit tests.
4. [ ] Fixed-contributor attachment (reach proposes, complex-edge confirms) +
   leakage-artifact pruning of the free set + dependency DAG +
   topological/batch ordering + unit tests.
5. [ ] Strong-line-driven difficulty classification + width-cap/split proposal
   (O4-3) + unit tests.
6. [ ] `io/window_serialization.py` + `stage4_windows` stage tracking +
   hand-edit round-trip tests; wire into the invalidation mechanism.
7. [ ] Wrappers (`Pipeline.assign_windows/visualize_windows/load_windows`,
   `api.*`, CLI `assign-windows`/`visualize-windows`) +
   `visualization/window_visualization.py`.
8. [ ] Cross-interface + 2638 real-data integration tests; resolve O4-1/3/4,
   record O4-7 handshake decision for the Stage 5 plan.
