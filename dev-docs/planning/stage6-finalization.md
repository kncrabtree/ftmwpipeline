# Stage 6 — user decisions, re-fits, and analysis finalization

Status: **planned; work gated** on the open-issue sweep recorded in the
ROADMAP sequence (the fitted-table semantics must stop moving before a
finalization layer freezes them). This document collects the ideas and
constraints established across the Stage 5 arcs so the design starts from
them rather than rediscovering them.

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

## Design areas (to be developed in this document before implementation)

1. **Attention routing** — which windows merit user review, ranked and
   justified: SNR-aware worst-ε, ε>κ doublet pairs, candidate-bearing
   windows, spur-adjacent fits, edge/boundary flags. One consolidated
   per-window "needs attention" surface rather than per-feature lists.
2. **The decision verb set and UX** — `fit refit --window N --add F /
   --remove F`, accept/reject a doublet merge, accept a revived
   candidate; dual-interface (CLI + Pipeline/API) per the architecture
   rule. UX is the primary design consideration (carried over from the
   candidate-revival plan).
3. **The decision log** — schema, placement in the `.ftmw`
   (`stage6_finalization` group), ordering/replay semantics, interaction
   with upstream invalidation, and the curated-vs-automatic separation.
4. **Finalization semantics** — what "final" asserts (which decisions are
   resolved, which flags remain open), whether finalization locks the
   record, and what the stage tracker dependency graph looks like
   (`stage6` requires `stage5`; report generation requires `stage6`).
5. **Acceptance fixtures** — the candidate-revival misses (1231
   w50/w425/w426, 363 w76), the succinimide/1512 doublet adjudications,
   and a curated end-to-end finalization of one fixture as the
   integration test.

## Out of scope

- Report generation itself (consumes the finalized record; planned
  separately as the ROADMAP reports feature).
- Any molecular-model fitting or model-derived prior.
- GUI work — this stage defines the data products and verbs the future
  UI drives; the UI is a separate program.
