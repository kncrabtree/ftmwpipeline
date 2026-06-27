# Attention-metric refinements + the coupled Stage 5 fitting changes

Status: **planned.** Reassessment of the Stage 6 attention surface after the F1
rework, plus the Stage 5 fitting changes the reassessment exposed as the real
fix. Diagnosis was measured on the seven-fixture `s4c` set (review run +
`--windows attention` reports + targeted refits; harnesses under
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

## The fitting refinements (Stage 5)

These produce the honest signals the attention layer then reads. They share one
hook: `_process_one_window` in `fitting/plan_execution.py`, immediately after the
per-node cleanup tail, where the converged `WindowOutcome` carries
`full_residual`, `rms_noise`, `offset_grid_mhz`, and the final peak set.

### F-1. Refresh the candidate-ledger SNR on the final residual; prune stale

After the cleanup tail, re-run `find_residual_peaks` on `outcome.full_residual`
(with `outcome.rms_noise`). For each persisted rescue candidate, replace its
recorded SNR/magnitude with the final-residual value matched by frequency, and
drop candidates that no longer clear the detector bar. The stale rescue SNR then
never leaves Stage 5; `derive_candidate_ledger` (Stage 6) reads an already-honest
ledger. Removes the stale bucket (~11/35 here, and the broader noise-grass tail).

- Data: mutate `RescueCandidateInfo.snr` / `.magnitude` in `outcome.rescue_events`
  (or rebuild the candidate list from the final residual and reconcile by
  frequency). Serialization already round-trips these fields.
- Bar: the residual detector's `snr_threshold` (default 2.5) for retention;
  the Stage 6 display/attention bars are unchanged and still apply downstream.

### F-2. Final add-from-convergence pass (recover the companions)

After F-1, for each surviving final-residual candidate above the display bar,
attempt **one** warm-started add through the existing conservative AICc gate:
warm-start the existing free peaks at their converged positions and seed the new
peak exactly at the residual-peak location. Keep the result only if it clears the
gate **and** does not collapse within the min-separation guard. The gate and the
collapse guard are unchanged, so a true over-split that collapses again is still
rejected — no new over-fit risk; the change is purely a better seed from a better
starting point.

- Reuses `attempt_residual_rescue` / the conservative add machinery, but seeded
  from the post-cleanup (post-baseline) state rather than mid-fit. The baseline
  must be carried as part of the held model during the add (the production rescue
  runs pre-baseline; this pass runs post-baseline, so the polynomial baseline
  joins the frozen background for the add's residual).
- Any installed line is re-baselined / re-cleaned consistently with the rest of
  the window before the node releases its dependents.
- Provenance: lines added here are `auto` (pipeline-installed), distinct from a
  Stage 6 user `review`/`refit --add`.

### F-3. Brightness-scaled shape-error reach in the candidate ledger

`derive_candidate_ledger`'s shape-error filter drops a candidate within
`SHAPE_ERROR_MAX_SEP_RES = 2.0` res of a fitted peak whose `snr·0.25` exceeds the
candidate evidence. Bright-line lineshape error extends far past 2.0 res (15–50
res for snr 10³–10⁵), so the cap lets ~half the strong candidates through as
sidelobes. Make the reach grow with the neighbor line's SNR (a wider lineshape-
error shadow for a brighter line) so a bright line's wings stop nominating
phantom candidates. Calibrate the reach against the measured sidelobe set
(`scratch/attention-reports/`, the class-C candidates).

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
