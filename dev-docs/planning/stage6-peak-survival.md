# Peak-survival pass — covariance/VIF overfit collapse + absolute SNR floor

Status: **implemented and re-baselined.** Closes
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

2. **F4 degenerate-pair merge.** *(Thresholds below are the original design;
   superseded by the merge-policy update in §Settings — `vif_collapse_threshold`
   is now 4.0 and `collapse_max_separation_res` 1.0, the merge is the default for
   any degenerate pair, and merged windows are flagged `auto_merged_review`.)*
   For a close pair where
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

3. **Attention range (feeds Stage 6, review-findings F1 — *now wired*).** Peaks
   with `VIF ≥ vif_attention_threshold` (default 4.0) not auto-collapsed become
   the `overfit_vif` attention reason, and peaks in the tight borderline band
   just above the prune floor become the `low_snr` reason, both in `review run`.
   The VIF utility built here (`amplitude_vif`) is the shared dependency;
   `vif_attention_threshold` and `snr_survival_floor` are resolved from the file
   in `review_run_impl`. Implementation: Stage 6 sequence step 4 in
   `_internal/stage6_impl.py:_compute_attention_reasons` (see
   `stage6-review-findings.md` F1). Note: in practice the high-SNR w217 A/C pair
   clears the attention band because its degenerate B/D collapsed and the honest
   refit normalized A/C to VIF ~3.1–3.7; w281 (1.08 res, never collapsed) is the
   surviving marginal case. The literal "post-floor SNR 3–5" band was narrowed
   to within ~10% of the floor — SNR 3–5 is a normal weak-line range and the wide
   band flooded the queue. The normalized-frequency-error cut was not needed: the
   amplitude VIF already covers w281.

## Settings

New resolved knobs (resolution chain `explicit > persisted > preset > recommended
> default`), in a `StageFitSettings` sub-group (e.g. `peak_survival`):

- `enabled` (default `True`)
- `snr_survival_floor` (default `3.2`)
- `vif_collapse_threshold` (default `4.0` — see the merge-policy update below;
  was `100.0` in the original design)
- `collapse_max_separation_res` (default `1.0` — was `0.5`)
- `vif_attention_threshold` (default `4.0`) — the `overfit_vif` flag for residual
  high-VIF pairs above the merge separation bound.
- `drop_empty_windows` (default `True`) / `drop_spur_only_windows` (default `True`)
  — end-of-Stage-5 window cleanup (review-findings F1).

> **Merge-policy update (supersedes §2/§3 thresholds; calibrated against the
> 1512/655 catalog truth).** The collapse is the *default merge* for degenerate
> close pairs, not a rare extreme-overfit cut. Calibration showed the amplitude
> VIF does **not** separate real doublets from over-splits in the 0.5–1.0 res
> band — real high-SNR multiplets carry VIF as high as over-splits, and the band
> is ~92 % over-splits (48:4); no prior-free statistic (VIF, SNR, χ²/AICc, the
> orthogonal second-line evidence) separates them. Prior-free, a split is a
> high-bar claim, so the policy is **merge by default, split is opt-in**:
> threshold dropped to the identifiability floor (`vif_collapse_threshold` 4.0)
> and the separation bound widened to one full resolution element
> (`collapse_max_separation_res` 1.0). Every merged window is flagged
> `auto_merged_review` (severity 0.1) for `review split` overrule. `low_snr` was
> retired as a flag (rework details + the candidate-currency fix are in
> `stage6-review-findings.md` F1).
>
> **Catastrophic-merge veto (`merge_chi2_veto`, default 100).** Merging a pair
> at high SNR can leave a 1-line model fitting catastrophically badly (1512 w250
> reached χ²ᵣ 5275) — the data overwhelmingly demands two components. When the
> post-merge reduced χ² exceeds the veto, the merge is reverted: the split is
> kept (and flags `overfit_vif`) rather than ship a broken fit. Calibrated on
> the 1512/655 catalog truth — ordinary over-split merges leave χ²ᵣ ≤ ~90,
> real-doublet merges blow up (≥ ~300), and the veto sits in that gap; the
> SNR-normalized eps does not separate them (the D10 floor saturates it), so the
> veto is on raw χ²ᵣ. This is also the super-resolution boundary: low SNR can't
> resolve a split (merged χ²ᵣ stays low → merge), high SNR resolves it (merged
> χ²ᵣ blows up → keep split). Net trade: 1512 recall 0.443→0.417, 655
> 0.579→0.568, with the catastrophic χ²ᵣ tail removed (1512 p95 42→19, max
> 5275→88) — precision and conservative multiplicity without shipping broken
> fits, every merge or kept-split flagged for review.

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
- **R2 — rework Phase A from slice to refit.** *Done.* Extracted the in-memory
  **refit core**
  `refit_window_core(fit_ctx, fit_win, wf, *, resolved, shape_enum, tau_maj_us,
  sigma_tau_us, peak_frequencies_mhz, add, remove, …) -> FittingResult`
  (materialize → reconstruct frozen → derive `fw_kwargs` → replay baseline →
  `fit_seeds_window_outcome` → `window_outcome_to_fitting_result` → origin stamp →
  thaw re-insert) from `refit_window_impl`, which is now a thin file-bound shell
  (load, spur-catalogue replay, persistence, decision recording) over it. The
  extraction is **byte-identical** (identity refits reproduce the persisted fit to
  the digit on real multi-peak windows; the stage6 + cross-interface suites stay
  green). The survival pass routes partial-prune windows through the core via a
  `refit_window=(wf, dust_freqs) -> FittingResult` callable injected from
  `fit_peaks_impl` (which holds `fit_ctx` / `plan` / `resolved` live): per window,
  drop it if all dust, leave it untouched if no dust, else `refit_window_core(...,
  remove=dust_freqs)` so the survivors' parameters + covariance + χ²ᵣ are
  re-estimated honestly (no slice). One pass suffices (removing noise-level peaks
  does not push a real survivor below the floor). Validated on a fresh full 2638
  fit: 565→542 windows, 604→565 peaks (same dust the slice version found), **0**
  auto survivors below floor, **0** windows with a covariance/peak-count mismatch
  (the honest refit covariance replaces the slice). The `_slice_window_covariance`
  helper is removed. 7-fixture re-baseline done (see Validation).
- **Phase B (R3) — VIF collapse.** *Done.* Added the shared `amplitude_vif`
  utility, `apply_vif_collapse`, and
  the `vif_collapse_threshold` / `collapse_max_separation_res` /
  `vif_attention_threshold` knobs on `PeakSurvivalSubSettings` (defaults 100.0 /
  0.5 / 4.0; the attention threshold is reserved for the Stage 6 surface, no
  auto-action here). The collapse runs after the SNR prune in `fit_peaks_impl`:
  per window, each peak with `amplitude_vif > threshold` pairs greedily
  (highest-VIF first) with its nearest neighbour within
  `collapse_max_separation_res × (1/T_active)`; the window is refitted once
  through `refit_window_core` (via an injected `(wf, remove_freqs, add_freqs,
  add_seeds) -> FittingResult` callable) with the paired members removed and one
  merged seed added per pair, stamped `origin="auto"` (a new `add_origin` param on
  `refit_window_core`, default `"user"` keeps every existing caller unchanged).
  The merged seed reuses the recorded `DoubletAlternativeInfo.merged_{frequency,
  amplitude,phase}` when present, else the amplitude-weighted centroid. The
  collapse fires on VIF + separation alone and never vetoes on the rising χ²ᵣ.
  Truth-set validated on a fresh full 2638 fit: collapses **exactly** w24 (VIF
  936/625, sep 0.264 res), w217 (981/978, 0.174 res), w438 (7494/7490, 0.108 res),
  and touches none of w161/w108/w419/w281 (562 peaks = 565 post-prune − 3 pairs).
  Unit tests cover the VIF utility + collapse gate (degenerate collapses, low-VIF
  kept, far-pair kept, no-window-range skipped, doublet-alt snap). 7-fixture
  re-baseline done (see Validation).

## Validation

- Truth-set assertions on a fresh 2638 fit are the per-window acceptance test
  (prune removes the 39 dust / 23 all-dust windows with no survivor below the
  floor; collapse fires on exactly w24/w217/w438 and no real doublet).
- **7-fixture re-baseline done and accepted.** The full cross-fixture harness
  rebuilt all 7 fixtures with the pass on. Because the prior ship-audit-d14
  reference predates the intervening Stage-3/4 work, recall was isolated by an
  on-vs-off comparison on current code (same build, only `peak_survival.enabled`
  toggled): a clean precision/recall trade, accepted as the new baseline.
  - **1512** (lorentzian): removed 22 lines (21 pruned + 1 collapse); recall
    0.443 → 0.435 (one sub-3.2σ catalog match lost, 51 → 50).
  - **655** (lorentzian): removed 1247 lines (1234 pruned + 984 all-dust windows
    dropped + 13 collapses); recall 0.527 → 0.518 (three sub-3.2σ matches lost,
    173 → 170). Accuracy-over-precision improved on both.

  The lost matches are below the Stage-3 detection threshold the floor is
  anchored to (the detector would not promote them) or one member of a
  sub-½-resolution degenerate collapse, so the small recall dip buys a large
  precision gain. The scratch audit + isolation harnesses live under
  `scratch/stage6-survival/` (gitignored).
