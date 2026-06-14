# Peak-survival pass — covariance/VIF overfit collapse + absolute SNR floor

Status: **design locked, evidence-backed; implementation phased.** Closes
review-findings F4 (degenerate-overfit identifiability) and F5 (absolute-weak
dust). Runs as an automatic pass at the **end of Stage 5** (`fit_peaks`), after
the conservative/rescue/cleanup loop produces the per-window fits, mutating the
persisted final-products table. Calibration evidence below was measured on the
persisted per-window covariance (the step-1 plumbing) on a fresh full 2638 fit
(`scratch/stage6-drive/exp_2638_review_fresh.ftmw`, 565 windows; probe scripts in
`scratch/stage6-cov/`).

## The discriminant (measured, not hypothesized)

The overfit signal is the **diagonal amplitude variance-inflation factor**

```
VIF = (amplitude_error / amplitude) * snr
```

≈ 1 for an identifiable line, ≫ 1 when a line is degenerate with a sub-resolution
neighbour (the *pair sum* is constrained, neither amplitude individually is). It
is a pure function of already-persisted per-peak fields — **no covariance matrix
needed** — so it is available even on the 11/431 non-empty windows whose
off-diagonal `JᵀJ` was singular.

**The off-diagonal correlation is NOT the discriminant** (correcting the
review-findings F4 hypothesis): measured amplitude correlation saturates near
`|1|` for *every* close pair regardless of keep/collapse — w217 B/D (collapse)
`+1.00`, w108 quartet (keep) `+0.93`, w419 blended doublet (keep) `−0.94`. Sign
is not diagnostic either. The persisted off-diagonal's value is **reports**
(honest correlated error bars) and **naming which pair trades** for a targeted
merge — not the keep/collapse decision.

Truth set (review-findings F4/F5), reproduced exactly from the persisted fit:

| window | verdict | sep (res-elem) | amp VIF | snr |
|---|---|---|---|---|
| w217 B/D | collapse | 0.17 | ~980 | 515–961 |
| w438 pair | collapse | 0.11 | ~7490 | 2333–2469 |
| w24 pair | collapse | 0.26 | 625–936 | 48–91 |
| w281 | marginal (attention) | 1.08 | 4.5 | 7–8 |
| w161 doublet | keep | — | 0.8–2.1 | 45–109 |
| w108 quartet | keep | 1.08 | 2.1–2.5 | 7–11 |
| w419 blended doublet | keep | 1.12 | 2.2–3.1 | 19–72 |
| w441 dust | drop (SNR) | — | 0.6–10.4 | 1.5–2.4 |

Keeps sit at VIF ≤ 3.1 regardless of SNR (the w108 guard: large *raw* amplitude
errors at low SNR, but SNR-normalized VIF stays ~2). Marginal w281 at 4.5.
Degenerate overfit explodes to hundreds–thousands.

## Two orthogonal automatic cuts + one attention range

Apply in order at end of Stage 5:

1. **F5 absolute SNR floor (prune).** Drop any fitted peak with post-fit
   `snr < snr_survival_floor` (default `3.2`, anchored to the Stage-3 detection
   threshold — a peak the detector would never promote should not survive the
   fit). Then drop windows left empty. File-wide: 32 peaks, 16 all-dust windows.
   Safe — every truth-set keep is SNR ≥ 7. Catches w441 entirely, including its
   VIF≈0.6 *absolute-weak* C/D that the identifiability cut misses (F4 and F5 are
   complementary). Record each prune in the per-window audit / plan diagnostics.

2. **F4 degenerate-overfit collapse.** For a close pair where
   `VIF > vif_collapse_threshold` (default `100`) **and** the two lines are within
   `collapse_max_separation_res` (default `0.5`) resolution elements, collapse the
   pair to one line. The 0.5 guard is the safety lever — two lines closer than
   half a resolution element are fundamentally unresolvable, so a split there is
   spurious; real doublets (w161 0.82, w108 1.08, w419 1.12) sit well above it.
   Every VIF>50 peak file-wide comes in a high-VIF *pair* (no isolated high-VIF
   lines), so the collapse target is structural. At the defaults this fires on 3
   clean pairs (w438/w217/w24); high-VIF pairs at sep 0.5–0.75 (w389 0.73, w309
   0.59, w317 0.63) route to attention instead.

   **Critical: the collapse overrides χ²/AICc unconditionally.** The
   doublet-alternative machinery already evaluated these pairs and the merged
   1-line χ²ᵣ is *much worse* (w438 8.6→165, w217 3.1→9.4, dAICc favours 2 lines)
   — which is exactly why the existing eps/AICc merge tier *kept* them. At high
   SNR χ²ᵣ is a lineshape-fidelity floor (D10), not a noise statistic: a
   near-degenerate second line absorbs lineshape mismodeling, so χ²/AICc *reward*
   the spurious split. VIF is not fooled. The collapse therefore fires on
   VIF+separation alone; the window χ²ᵣ *rising* on collapse is expected (it is
   removing shape-error absorption, not signal) and must **not** trigger a
   re-acceptance/rejection. Reuse the precomputed
   `DoubletAlternativeInfo.merged_{frequency,amplitude,phase,tau}` for the matching
   pair (fall back to a fresh merged refit if no alternative was recorded). Record
   the collapse with provenance in the audit and decision-relevant diagnostics.

3. **Attention range (feeds Stage 6, review-findings F1 — not this pass).** Peaks
   with `VIF ≥ 4` not auto-collapsed (w281, w217 A/C), normalized frequency error
   `> ~0.25 × separation`, or borderline post-floor SNR (3–5) become a
   high-precision overfit attention reason in `review run`. The VIF utility built
   here is the shared dependency; the attention wiring is roadmap Stage-6 step 4.

## Settings

New resolved knobs (resolution chain `explicit > persisted > preset > recommended
> default`), in a `StageFitSettings` sub-group (e.g. `peak_survival`):

- `enabled` (default `True`)
- `snr_survival_floor` (default `3.2`)
- `vif_collapse_threshold` (default `100.0`)
- `collapse_max_separation_res` (default `0.5`)
- `vif_attention_threshold` (default `4.0`) — surfaced for step 4; computed/stored,
  no auto-action.

Resolution element = `1 / T_active` (MHz), derived from the persisted canonical FT
settings (start/end/trim) — keep it a pure function of the file.

## Prerequisite refactor — one bare window-fit core

The survival pass must **refit** an affected window (drop the dust / collapse the
pair, then re-converge the surviving peaks), not merely filter the peak list: a
filter leaves the survivors' point estimates and the window χ²ᵣ/AIC computed with
the removed peaks still in the model, and (for the covariance) only a marginal
slice of the stale joint fit. A refit re-estimates the survivors free of the
removed peaks' influence and recomputes honest covariance + window statistics.

The refit must route through the **single bare NLS-to-outcome core**, not the
heavy `refit_window_impl` (which re-reads the file, replays the spur catalogue,
and records review decisions). Current state:

- `fit_window` (`fitting/window_fit.py`) is already the one peak-NLS primitive
  (explicit inputs → joint least-squares + covariance); every peak fit routes
  through it. Input prep is shared via `build_stage5_fit_context` (file → active
  FT + noise + sideband + spur set) and `materialize_window` (window → grid /
  data / noise / center).
- **Duplication:** `_fit_one_window` (`plan_execution.py`, the main loop) and
  `refit_window_impl` (`stage6_impl.py`) each build an identical `WindowOutcome`
  (full-fitted / residual / edge-coherence, then stash `_center_mhz`/`_spur_mask`)
  around `fit_window` + `knockout_test`. `refit_window_impl` mirrors
  `_fit_one_window`'s tail by hand.

**Refactor (byte-identical, its own step):** extract a bare core that takes a
materialized window + frozen background + a *given* seed set + fit kwargs and
returns a `FittingResult` via single `fit_window` + `knockout_test` →
`WindowOutcome` → `window_outcome_to_fitting_result` — no file I/O, no spur
re-derivation, no decision recording. Route `refit_window_impl`'s body, the
survival-pass refits, **and** `_fit_one_window`'s final-fit/outcome construction
through it (the conservative add-one-peak *search* stays as orchestration above
the core, since it is more than a single fit). Acceptance: the deterministic
7-fixture fitted tables stay **byte-identical** and the full suite is green —
the refactor changes structure, not output.

## Phasing

- **Step 0 — bare window-fit core extraction.** *Done* (committed: "Route window
  fitting and the review refit through one outcome core"). Extracted
  `build_window_outcome` + `fit_seeds_window_outcome` in `plan_execution.py`;
  routed `_fit_one_window` and `refit_window_impl` through them. Byte-identical
  (full suite green, zero baseline edits).
- **Phase A — SNR-floor prune** (closes F5). *Implemented (interim slice version,
  in tree).* The peak-set filter + empty-window-drop + settings + diagnostics are
  done (`apply_snr_survival_prune` in `stage5_impl.py`, gated on
  `peak_survival.{enabled,snr_survival_floor}`, default floor 3.2; drops **only**
  windows it empties, never pre-existing `K=0` windows). It currently **slices** a
  partially-pruned window's covariance to the survivors — superseded by R2's real
  refit. Validated on a fresh full 2638 fit: 39 dust peaks pruned (max removed SNR
  3.186 < 3.2), 23 prune-emptied windows dropped, 134 pre-existing empty windows
  preserved, every truth-set keep retained (w441 → 0; w161/w108/w419/w217
  unchanged). The settings group also reserves `vif_collapse_threshold` /
  `collapse_max_separation_res` / `vif_attention_threshold` for Phase B.
- **R2 — rework Phase A from slice to refit.** *Next.* The faithful "refit a
  window with an edited peak set" already exists as `refit_window_impl`'s body: it
  derives the per-window `fw_kwargs` from resolved settings, reconstructs the
  frozen contributors, and (post Step 0) calls `fit_seeds_window_outcome`. The
  survival pass needs the same without `refit_window_impl`'s file I/O,
  spur-catalogue *re-load*, and decision recording. So extract that in-memory
  **refit core** — `refit_window_core(fit_ctx, plan, window_id, seed_peaks, *,
  resolved, …) -> FittingResult` (materialize → reconstruct frozen → derive
  `fw_kwargs` → `fit_seeds_window_outcome` → `window_outcome_to_fitting_result`) —
  and route both `refit_window_impl` (keeping its file-load/spur-replay/decision
  shell) and the survival pass (which already holds `fit_ctx`/`plan`/`resolved`
  live in `fit_peaks_impl`) through it. The survival pass then, per window with
  sub-floor peaks: drop the window if all dust, else call `refit_window_core` with
  the surviving seeds; the refit recomputes parameters + covariance + χ²ᵣ honestly
  (no slice). Removing noise-level peaks never pushes a real survivor below the
  floor, so one pass suffices. Extracting `refit_window_core` is itself a
  byte-identical refactor of `refit_window_impl` (verify before the behavior
  change); then re-baseline the 7 fixtures and confirm recall holds (only
  dust-touched windows change).
- **Phase B (R3) — VIF collapse** (higher risk; reuses the merged-fit machinery).
  Build the shared VIF utility, the collapse decision (VIF>100 + sep<0.5 res,
  overriding χ²/AICc), and the merged-params reuse, on the `refit_window_core`
  from R2. Truth-set test: collapse w438/w217/w24; never touch
  w161/w108/w419/w281. Then cross-fixture re-validation.

## Validation

- Truth-set assertions on a fresh 2638 fit are the per-window acceptance test.
- Both cuts mutate the persisted tables → the 7-fixture references and the
  byte-identical fixture-equality tests change; **re-baseline deliberately** and
  confirm ground-truth recall (1512 / 655 / …) does not regress (the prune should
  only remove dust; the collapse only removes spurious sub-resolution splits).
  This is a human-judged step, not an automatic test update.
