# Attention-metric refinements + the coupled Stage 5 fitting changes

Status: **fitting refinements (F-1/F-2/F-3) implemented; attention metrics
(A-1–A-4) and the report phase pending.** Reassessment of the Stage 6 attention
surface after the F1 rework, plus the Stage 5 fitting changes the reassessment
exposed as the real fix. Diagnosis was measured on the seven-fixture `s4c` set
(review run + `--windows attention` reports + targeted refits; harnesses under
`scratch/attention-reports/`, evidence summary in that directory's `FINDINGS.md`).

This document covers the **attention metrics** and the **fitting refinements**
they depend on. The coupled **report** work (vertical compaction, above-the-fold
window reorg, quick-clear controls + richer in-table attention detail, and the
post-curation diff report) is a later phase, scoped separately once these land.

## Why

`review run` over the seven fixtures flags 18–27% of windows on the dense ones.
Two reasons carry ~75% of that load — `overfit_vif` (165 instances) and
`candidate_bearing` (99) — and both are dominated by the **same** root cause:
bright-line lineshape mismodeling masquerading as a problem. The genuinely
actionable residue of each is ~20 windows. The reassessment goal (user): flags
should be high-precision and few; automate the clear calls, surface the rest on
demand.

Three measured facts drive the plan:

1. **`overfit_vif` conflates three populations.** `amplitude_vif =
   (amp_err/amp)·snr` equals `amp_err/σ_noise` (since `snr = amp/σ_noise`) — the
   amplitude uncertainty in noise-floor units, brightness-invariant. Splitting
   the 165 flagged windows on (weak-member fractional amplitude uncertainty) ×
   (SNR-aware lineshape gate) separates them cleanly: ~74% are spurious sub-res
   over-splits (weak member fracUnc ≳ 15%) that want silent merging; a small
   misfit corner (low fracUnc, fails the ε-gate) genuinely warrants a look (e.g.
   2638 w332); the rest are well-resolved doublets that are simply fine
   (over-production).

2. **`candidate_bearing` fires on a stale, contaminated signal.** Its
   `residual_snr` evidence is recorded at *rescue-round* time (a not-yet-converged
   residual). Re-measuring on the **final** post-cleanup residual (debug hook
   `FTMW_DEBUG_RESIDUAL_SNR`, all seven fixtures refit): of 35 isolated-strong
   candidates ~11 are stale (final SNR < 4, including every χ²ᵣ≈1 noise-grass
   case), and the persisting ~24 split into fit-deficient (χ²ᵣ ≥ 4 — a misfit,
   not a missed line) and a small **companion** set (χ²ᵣ ≈ 2, ~2 res from a
   moderate line) that human review confirmed are real missed lines. Separately,
   half of the *strong* candidates across the full ledger are lineshape sidelobes
   of bright lines that slip the shape-error filter's hard 2.0-res reach.

3. **The companion lines are rejected by a seed/collapse failure, not the gate.**
   Tracing 655 w156: the K=2 add attempt was rejected with `aicc_delta = -102.8`
   (AICc *strongly* preferred two peaks) and reason "peaks collapsed within min
   separation" — a mid-fit seeding failure, not a realness verdict. Seeded from
   the converged fit at the honest residual-peak location, the same gate would
   accept them.

## The fitting refinements (Stage 5) — implemented

F-1, F-2, and F-3 are implemented and validated against the seven-fixture `s4c`
set. They produce the honest candidate signals the attention layer reads.

**F-1 — refresh the candidate-ledger SNR on the final residual.**
`_refresh_rescue_candidates` in `fitting/plan_execution.py` re-runs
`find_residual_peaks` on the post-cleanup `outcome.full_residual` at the
`_process_one_window` tail, replaces each persisted rescue candidate's SNR /
magnitude with the matched final-residual value, and drops candidates with no
surviving residual peak. No schema change (the records are mutated in place by
reference, so both `rescue_events` and the plan-level `rescue_history` see it);
`derive_candidate_ledger` then reads an already-honest ledger. Line-list-neutral.

**F-2 — final add-from-convergence pass.** `_add_from_convergence` attempts one
warm-started add per surviving candidate above `final_add_snr_threshold` (new
`RescueSubSettings` knob, default `10`, `0` disables): it seeds the new line at
the residual peak via `refit_outcome(add_seeds=...)` and reads the AICc verdict
straight off the trial's `knockouts` (`fit_seeds_window_outcome` already runs the
knockout gate), accepting only when the added peak is `supported` and no pair
collapses within the min-separation guard. Key decision: the pass reuses F-3's
brightness-scaled sidelobe predicate as a **pre-filter**, so it never installs a
bright line's lineshape sidelobe (the AICc gate alone accepts them because the
core σ_eff budget does not cover the far skirt) — this keeps F-2 and F-3
consistent. Re-baseline: χ²ᵣ never worsens; 655 +19 genuine companions (χ²ᵣ p95
5.05→3.88, catalog recall 0.662→0.688 main), 1231 +3, 1512 +2, the
bright-line-dominated fixtures +0. Known caveat: the 1.5-res w156 doublet is *not*
recovered (the warm start still collapses it / the AICc gate rejects from
convergence) — accepted as the safe, over-production-free direction.

**F-3 — brightness-scaled shape-error reach.** The hard `2.0`-res /
`0.25`-fraction cap in `derive_candidate_ledger` is replaced by a single
`1/Δ`-decay rule: a candidate is a sidelobe when `sep_res ≤ κ·snr/evidence` for
some fitted peak (the finite-T boxcar sinc envelope, so the reach widens with the
neighbor's brightness). `SHAPE_ERROR_REACH_KAPPA = 0.2` lives in
`fitting/validation.py` (one calibration shared by F-3's ledger filter and F-2's
install pre-filter). Calibrated on the seven-fixture strong-candidate set: drops
83/88 measured sidelobes, keeps 0/35 genuine companions (which sit near
modest-SNR ≤ ~300 lines).

## The attention refinements (Stage 6)

`_compute_attention_reasons` in `_internal/stage6_impl.py`.

### A-1. Retire `overfit_vif` as a standalone flag; route its content

- The merge decision moves to the **weak-member fractional amplitude
  uncertainty** (the end-of-Stage-5 collapse already owns merging; extend its
  criterion so the fracUnc ≳ 15% sub-res pairs that currently survive past the
  1.0-res separation guard are merged). Merged windows keep the low-severity
  `auto_merged_review` advisory.
- The genuine misfits (low fracUnc, fails the SNR-aware ε-gate) surface through
  `worst_eps`, which already exists. Do **not** introduce a raw absolute χ²ᵣ bar:
  the lineshape model tolerates ~5% lineshape error, which is catastrophic in
  χ²ᵣ at high SNR by design, and the ε-gate already accounts for it.
- Net: `overfit_vif` no longer fires on its own; the well-resolved-doublet
  over-production disappears.

### A-2. `candidate_bearing` on the honest signal

With F-1/F-3 in place, the ledger is already de-staled and de-sidelobed. The flag
then:

- Fires only on isolated final-residual candidates that **persist** above bar and
  are **not** in the fit-deficient regime (those route to `worst_eps`/misfit, as
  the residual is real power the add could not model as a line). With F-2, the
  companion lines are *recovered automatically* and so leave the flag entirely —
  the flag's residue is the genuinely ambiguous remainder.
- Optional low-priority refinement: a residual-peak width / phase-coherence
  discriminant to demote single-bin spurs (e.g. 360 w172) to a lower-severity
  advisory — a real line spans several bins tracking the line shape.

### A-3. Demote `auto_merged_review` out of the default queue

The merge is the more-likely-correct call (~92% of the sub-res band is
over-splits); the advisory exists only so a user with catalog support can find
and re-split it. Keep it discoverable (e.g. `review rank --by`, or a report
section) but out of the default attention queue so it stops padding the count.

### A-4. B5 cascade surface (folded in from the cascade effort)

The contributor-edit cascade deferred its attention/report surface here: a
`review rank --by <diff-metric>` over revised-vs-baseline dependent diffs (peak
moved ≳k·σ, σ inflated, χ²ᵣ shifted, peak-count changed) and a cascade report
section. Land the ranking metric here; the report section rides the later report
phase.

## Interface surface (dual-interface invariant)

The fitting changes are internal to Stage 5 and inherited by all three interfaces
through `fit_peaks_impl`. The attention changes are internal to `review run` and
inherited likewise. New tunables (the fracUnc merge bar, the brightness-scaled
reach, the final-add bar) are promoted to first-class settings
(`PeakSurvivalSubSettings` / the candidate settings) with byte-identical defaults
where a default already exists, and exposed through the standard settings path —
no interface-specific logic. Any new `review rank --by` metric is added to
`RANK_METRICS` and the CLI `review rank` enumeration identically.

## Serialization

- F-1: no schema change — `RescueCandidateInfo.snr`/`.magnitude` already persist;
  the refresh mutates their values before persistence.
- F-2: no schema change — installed lines are ordinary `FittedPeak`s with `auto`
  provenance.
- New settings fields version through the existing settings serialization with
  defaults that reproduce current behavior unless explicitly set.

## Test plan

- Unit: F-1 stale-prune (a candidate whose final-residual SNR drops below bar is
  removed; one that persists is kept with the refreshed value); F-2 add-from-
  convergence (a known collapse-rejected companion — 655 w156 — installs and
  clears the gate; a true over-split still collapses and is rejected); F-3
  brightness-scaled reach (a bright-line sidelobe at 3–15 res is filtered, a real
  faint companion at the same separation from a faint line is not).
- Attention: A-1 (a fracUnc-high pair merges and flags `auto_merged_review`, a
  low-fracUnc misfit flags `worst_eps`, a clean resolved doublet flags nothing);
  A-2 (a stale/sidelobe candidate no longer flags; a persisting isolated one
  does).
- Cross-interface consistency tests for `review run` and `fit run` after the
  changes (the dual-interface gate).
- Seven-fixture re-baseline: line counts, χ²ᵣ distribution, honest recall
  (`scratch/cascade/recall_honest.py`), and the attention-flag counts per reason.
  Accept the recall/precision trade explicitly (F-2 is expected to *raise* recall
  on the companion lines).

## Open questions

- F-2 ordering vs the existing rescue rounds: a single post-cleanup pass, or fold
  the warm-started-from-convergence seed into the existing rescue loop's final
  round? Measure both against the bucket-C recovery count and wall time.
- The fracUnc merge bar (~15%) and the brightness-reach function are calibration
  numbers; fix them against the measured sets before promoting to settings.
- Whether `candidate_bearing` survives at all once F-1/F-2/F-3 land, or collapses
  to so few windows that it folds into `worst_eps` + the on-demand ledger.

## Deferred (later report phase)

Vertical compaction (hide covariance/correlation by default, tighter layout),
above-the-fold reorg of the per-window page, quick-clear controls + richer
in-table attention detail, and the **post-curation diff report** (side-by-side
plots and χ²ᵣ/peak-drift stats for windows a curation action touched, building on
the cascade's revised-vs-baseline diff). Scoped separately after the metrics land.

**On-plot attention annotations.** Mark the part of the window an attention reason
points at directly on the per-window plot — SVG overlays in the same vein as the
existing gated-spur annotations — so it is visually obvious *where* to look (the
flagged residual peak for a `candidate_bearing`, the non-identifiable pair for a
merge/misfit, the spur-adjacent line, the edge peak). This is best deferred until
the **final attention categories are settled** (so the annotation vocabulary
matches the reasons we keep), but it can land any time after that.
