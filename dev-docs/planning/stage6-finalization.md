# Stage 6 — user decisions, re-fits, and analysis finalization

Status: **implemented.** The `review run/show/edit/merge/split/accept` read/edit
surface, the candidate ledger, the anchored decision log, and the
consolidate/calibrate finalization layer (the persisted final-products table) are
all built and shipped; review-findings F1–F5 are resolved, and the curation
clients (reports + report-driven curation) build on this contract. The sections
below are the agreed contract this layer implements.

The CLI surface is a new stage object **`review`** (object-verb grammar). The
tracked stage / HDF5 group is **`stage6_review`** (requires `stage5_fitting`;
`timebase_calibration` is a *soft* input; report generation requires
`stage6_review`). Stage 6 has two roles: it captures *human decisions* about
the automatic model, and it *consolidates and calibrates* the final data
products that reports consume.

## Purpose

The pipeline through Stage 5 produces an automatic spectral model with full
audit records. Stage 6 is where a *person* enters: reviewing flagged
windows, adding/removing lines, adjudicating blends and sub-resolution
doublets, accepting or rejecting revived candidates — and where the
analysis is declared **final**. The finalized record is the sole input
contract for report generation (a separate, later feature).

## Established constraints (carried from prior arcs)

- **Prior-free pipeline boundary.** Stages 0–5 extract the best spectral
  model the data supports with no molecular-physics priors; Stage 6
  captures *human* decisions about that model. A user decision is recorded
  provenance, not a physics prior injected into the fit. Hybrid
  model-as-prior fitting belongs to the future UI / molecular-fitting
  layer (cf. `bcfitting`), with its independent-measurement questions.
- **Decisions bypass gates but carry provenance.** From the
  candidate-revival design
  ([`stage5-candidate-revival.md`](stage5-candidate-revival.md), absorbed
  here): user edits are window-scoped re-fits (`add F` / `remove F` /
  accept-merged-alternative) that bypass the automatic accept gates,
  flagged with a `user` origin, auditable, and kept separable from the
  automatic result (curated vs automatic views) in all validation tooling.
- **Reproducibility (Principle 4 / D11).** The `.ftmw` file remains
  self-contained: the decision log and the finalized state persist in the
  file, so a shared file reproduces the finalized analysis without any
  side channel. Re-running an upstream stage that invalidates the fit must
  interact deliberately with recorded decisions (replay where the
  decision's anchor survives, surface conflicts where it does not — not
  silent loss; the exact semantics are a design task).
- **Observation-only statistics stay observation-only.** The
  doublet-alternative pass and the audit records never change the fit;
  flips happen only here, as decisions.

## Inputs already persisted for this stage

- The Stage 5 fit with per-window audit trails, knockout/rescue/thaw
  histories, spur provenance, and clock-lattice annotations.
- Doublet-alternative records per close pair: both fits' χ²ᵣ (the ε pair
  and `doublet_not_required` computed at the validation layer), the
  orthogonal-evidence score, and the bookkeeping for the merged
  alternative — with the calibrated interpretation (ε > κ is a
  high-precision "doublet required"; `orth_frac` with its χ²₃ null and
  degradation markers) documented in
  [`stage5-doublet-alternative.md`](stage5-doublet-alternative.md).
- Considered-but-rejected candidates in the audit records (the
  candidate-revival ledger's source).

## Locked design

### A. Per-window status and per-peak origin (no global finalize lock)

There is **no global "finalize" action** that locks the record. Stage 6 is a
review/curation layer over the stage-5 fit carrying two **orthogonal** flags
per window, rendered as one combined label:

- **provenance** — `auto` (the fit produced it, a human has not touched it),
  `reviewed` (a human inspected it and accepted the automatic fit unchanged),
  or `user-edited` (a decision was applied);
- **attention** — none, or `needs-attention` with one or more justified
  reasons (worst-ε, ε>κ doublet pair, candidate-bearing, spur-adjacent,
  edge/boundary). Attention is **advisory** and never blocks report
  generation; a window may legitimately remain `auto · needs-attention`.

The two axes are independent: a window can be `user-edited · needs-attention`
(edited, but routing still flags a neighbouring spur) or `user-edited` with no
remaining flag. Each **peak** additionally carries an `origin` ∈ {`auto`,
`user`} that survives serialization and is shown in the window report so
user-added lines are visibly marked.

Report-readiness is **computed, not asserted**: the *only* hard bar to report
generation is a `user-edited` window whose decision has been **invalidated**
by re-running an upstream stage (see §E). `needs-attention` flags are reported
honestly but do not block.

### B. Attention routing (advisory)

One consolidated per-window "needs attention" surface — ranked and justified —
rather than per-feature lists. Inputs: SNR-aware worst-ε, ε>κ doublet pairs,
candidate-bearing windows, spur-adjacent fits, edge/boundary flags. A flag is
cleared either by an edit (the window becomes `user-edited`) or by an explicit
`review accept` (the window becomes `reviewed`) — distinguishing "a human
looked and accepted" from "nobody looked." Routing never forces resolution
(no rubber-stamping).

### C. Decision verb set and UX (`review` object)

Dual-interface per the architecture rule: CLI `review` verbs, `Pipeline.review_*`
methods, and a functional-API mirror, all delegating to one
`_internal/stage6_impl.py`.

- `review run` — build/refresh the curation layer (routing labels + candidate
  ledger). `review show` — attention-ranked window list with combined labels;
  `--window N` for the per-window detail (fitted peaks with `origin`, ledger
  candidates as hollow/grey ticks, raw fitted frequencies — see §F);
  `--candidates` for the machine-readable ledger.
- `review edit --window N --add F --remove F` — the user-directed window
  re-fit. `--add F` snaps to the nearest ledger candidate within tolerance
  (reviving its recorded seed) else seeds a fresh peak at F; `--remove F`
  snaps to the nearest fitted peak.
- `review merge --window N --peaks F1,F2[,…]` — collapse a set to one peak
  (amplitude Σ, SNR-weighted-mean frequency, τ from the dominant member).
  When the set matches a close pair for which the doublet-alternative pass
  already recorded a merged-single alternative, `merge` **snaps to that
  recorded seed** instead of reseeding from scratch.
- `review split --window N --peak F [--into K]` — replace one peak with K
  (default 2) straddling F by a fraction of a resolution element, amplitude
  divided.
- `review accept --window N` — the "looked, no change" dismissal (→
  `reviewed`); also accepts a revived candidate or the recorded doublet-merge
  alternative.

`merge`/`split` are physics-aware-reseed sugar over `add`+`remove`; accepting a
precomputed doublet alternative is the snap case of `merge`. All edits carry
`user` provenance.

**Gate semantics for user edits.** A user add bypasses the *accept* gate (the
user is the gate) but still faces the NLS honestly: if the optimizer drives
the peak to zero amplitude or collapses it onto a neighbour, the result
reports that rather than silently keeping a phantom. Knockout/χ² diagnostics
are computed and flagged `user` like any other peak. Subsequent automatic
passes (merge cleanup, AICc cleanup, rescue) must **not** prune a user-added
peak nor re-add a user-removed one — either would make the verb feel broken.

### D. Candidate ledger (pass 1)

Derived from the existing audit machinery (`audit_trail`, `rescue_events`):
add-loop `reject`/`tentative` steps, rescue-round `candidates`, blend-split
trial failures, separation rejects. Normalized to **one entry per distinct
molecular-MHz candidate** (deduped across rescue rounds and decision sites),
carrying frequency, best evidence seen (Δχ², f/p, or residual SNR), rejection
reason(s), decision site(s), and the recorded seed (offset/amplitude) for
revival. A **display bar** (strawman residual/rescue SNR ≥ 3, or gate evidence
within an order of magnitude of the bar) keeps gate-killed dust off the user;
the bar is a display threshold, tunable, orthogonal to the accept gates, and
calibrated so the four verified miss windows surface their candidates while a
quiet window surfaces none.

Because `audit_trail` and `rescue_events` are **already persisted** per-window
in the stage-5 group, the ledger is a pure function of persisted data and is
**derived on demand** in the `review` read path — `review show` renders it
without re-fitting and without touching the stage-5 fit logic or serialization.
A materialized cache (a persisted sibling of `diagnostics["gated_spurs"]`) is
an optional later optimization, not required for correctness.

### E. Decision log and replay (pass 2)

The decision log is an **ordered** list persisted in the `stage6_review` group
so the `.ftmw` reproduces the curated analysis with no side channel
(Principle 4 / D11). Each entry is **anchored** (originating window identity +
molecular frequency), with its kind (`add`/`remove`/`merge`/`split`/`accept`),
`user` provenance, and an evidence snapshot.

**Replay = re-apply + diff.** Re-running an upstream stage re-applies each
decision wherever its anchor still resolves, and surfaces a comparison
(χ²ᵣ with/without the decision, peak-count and frequency deltas) so the user
can decide whether to revisit it — decisions are never silently lost.
**Invalidated** = the anchor no longer resolves (the window is gone, or the
decision's frequency no longer lands in any window / has fallen out of band).
An invalidated `user-edited` window is the sole hard bar to report generation
(§A); the report, when generated, logs the Stage-6 decisions (old-vs-new fit
info) for honesty.

Automatic results stay separable from curated ones in all validation tooling
(curated-vs-automatic views): a curated fixture must never silently masquerade
as an automatic benchmark.

### F. Final data products (consolidation + calibration), persisted in-file

Stage 6 produces and **persists** the single canonical final-products table —
this is the "finalized record" reports consume. Per accepted peak: frequency,
the σ_f budget (statistical + ε-residual + lineshape floor), amplitude, phase,
and SNR. Computed from {stage-5 raw fitted peaks} × {frequency calibration} ×
{lineshape floor}.

**Calibration is a reported state, never a report gate.** Whether the
frequency axis needs self-calibration is a property of the instrument's clock
reference, declared alongside the clock tree: the `spur` clock declaration
carries a frequency-reference marker (sibling of `clocks` / `ClockSource.locked`),
defaulting to **assume Rb-locked / absolutely calibrated** when nothing says
otherwise. Three states, each stated plainly in the table and the report:

- **Rb-locked (declared, or the assumed default).** The axis is absolutely
  calibrated by the instrument's frequency standard; ε ≡ 0 and self-calibration
  is a null operation (the all-Rb-locked succinimide instrument measures
  ε = 0 ± 0.05 ppm). Frequencies are trusted as-is; the budget carries only the
  negligible clock-standard systematic. (succinimide is this case.)
- **Free-running, self-calibrated.** `timebase_calibration` ran; the measured
  ε is applied and its residual folded into the σ_f budget.
- **Free-running, not self-calibrated.** Frequencies are reported uncalibrated
  with a strong recommendation to self-calibrate.

**Self-cal topology constraint (documented limitation, not a TODO).** The
timebase self-calibration handles exactly one topology: all signal-chain
clocks Rb-locked, with the **digitizer/sampling clock the single free-running
source**. It builds the reference lattice from the locked fundamentals and
fits the digitizer's fractional scale error ε from how the locked spurs
appear to drift. The method therefore supports **at most one unlocked clock,
and it must be the digitizer.** The inverse — a locked digitizer with some
other free-running clock — is not handled and has no systematic remedy here;
a declaration outside the supported topology reports as free-running with
self-cal *unavailable* (frequencies as-is, caveated), rather than silently
producing a wrong ε. This restriction is stated plainly wherever the
frequency-reference marker is documented.

`timebase_calibration` is therefore a *soft* input — strongly recommended for a
declared free-running instrument (within the supported topology), a null op
for an Rb-locked one, and in no case a hard bar to report generation (the sole
hard bar remains an invalidated `user-edited` window, §A/§E). The report states
the calibration state and, for free-running instruments, whether self-cal was
performed.

There is **one** headline table. The raw (uncalibrated) fitted frequency lives
in the per-window fit *detail* (`review show --window N`) as a drill-down, not
as a peer table — the two representations sit at different altitudes, which
removes the two-table redundancy. Reports **render** this persisted table; they
do not recompute calibration. The table is subject to the §E replay/diff
honesty: re-running timebase or stage 5 recomputes it with a surfaced diff.

### G. Report grouping (presentation, not storage)

In the report, baseline coefficients and fixed-contributor information are
presented under the **window / stage-4** section, and the stage-5 section is
**peak** information. This is a *presentation* grouping in the report renderer:
internal storage is unchanged — the baseline coefficients and frozen-skirt
amplitudes are genuine stage-5 fit outputs (stage 4 owns window geometry,
difficulty class, and which contributors attach; stage 5 solves for their
values), so they stay in the stage-5 payload.

## Implementation passes

**Pass 1 — candidate ledger + user-directed edits.** §D ledger (derivation,
normalization/dedupe, display bar, persistence, `review show`/`--candidates`)
and §C edits (`review edit`/`merge`/`split`), the per-peak `origin` flag,
user-peak immunity in merge/cleanup/rescue, and provenance round-trip. This
half is self-contained and testable on the four known miss windows plus the
1019 overfit-remove direction.

**Pass 2 — finalization layer.** §A per-window status + §B attention routing,
`review accept`, §E decision log + replay/diff, §F finalized-products table
(including the clock-declaration frequency-reference marker and the
three-state calibration reporting), the `stage6_review` tracker wiring
(`requires stage5_fitting`; report generation requires `stage6_review`), and
the computed report-readiness check.

## Acceptance fixtures

- Candidate-revival misses (each verified present in the v8 audit record):
  1231 w50 (−1.737 MHz SNR-11 rescue candidate rejected every round),
  1231 w425 (two `tentative` adds awaiting a batch that never formed),
  1231 w426 (failed blend-split trial + separation rejects), 363 w76 (two
  final separation rejects after 10 accepts).
- The overfit-remove direction on a very-high-SNR fixture (e.g. 1019).
- The succinimide / 1512 doublet adjudications (`merge`/`split` + accept).
- The final-products / calibration path: the 1512 uncertainty-accuracy goal
  (shared with the reports acceptance test).
- Cross-interface consistency for the `review` verbs and accessors, and a
  curated end-to-end finalization of one fixture as the integration test.

## Scope notes and open questions

- **Neighbor staleness (carried from candidate-revival).** A re-fit window
  changes its own peaks only. If the window is a *contributor primary* for
  neighbours' frozen skirts, those neighbours' fits become stale. Pass 1
  documents the staleness and reports which neighbours reference the window;
  it does **not** cascade automatically. Cascading re-fit is a later phase.
- **Replan interaction.** The mid-fit structural replan can change window
  geometry between fits. The ledger and edit verbs address the *persisted*
  windows of the current fit; the §E replay handles geometry changes via the
  anchor-resolution / invalidation rule.
- **Serialization.** New per-window ledger dataset(s) and the `stage6_review`
  group (decision log + final-products table); old files without them open
  fine (empty ledger, no decisions, no curation layer).

## Out of scope

- Report generation itself (renders the finalized record; planned separately
  as the ROADMAP reports feature). Stage 6 defines the persisted contract; the
  reports feature owns presentation and the uncertainty-budget *composition*.
- Any molecular-model fitting or model-derived prior.
- Automatic cascading re-fit of neighbour windows (see Scope notes).
- GUI work — this stage defines the data products and verbs the future UI
  drives; the UI is a separate program.
