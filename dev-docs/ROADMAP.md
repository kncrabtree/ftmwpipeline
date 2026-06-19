# Roadmap

Coordination and task-tracking for ftmwpipeline development. This document is
deliberately lean: it tracks *what is being worked on and what is decided*, and
points to the documents that hold the detail. It is not a changelog or a status
narrative.

- **Verified current state:** see [`../STATUS.md`](../STATUS.md) (regenerated
  from code, not from plans). Do not restate status here; link to it.
- **Normative specs:** the `*_STRATEGY.md` documents in this directory are
  specifications — timeless requirements, not progress reports.
- **Implementation planning:** per-feature plans live in
  [`planning/`](planning/) and are registered below. On completion they become
  implementation overviews that later seed user documentation.

## Vision

A dual-interface (CLI + Python), file-centric pipeline for FTMW spectroscopy.
Each experiment is one portable `.ftmw` file progressing through stages
`FID → ComplexFT → NoiseResult → Peaks → Windows → FittedPeaks`. All interfaces
share one implementation and must produce identical results.

## Priorities

Forward-looking only. Completed work — the closed gating sweep, predecessor
milestones, and finished planning documents — is archived in
[`COMPLETED.md`](COMPLETED.md); verified current state is in
[`../STATUS.md`](../STATUS.md). Two tracks:

1. **Complete Stage 6 — covariance, peak-survival metrics, window construction,
   attention** (in progress). The read/edit surface
   (`review run/show/edit/merge/split/accept`,
   [`planning/stage6-finalization.md`](planning/stage6-finalization.md),
   absorbing [`planning/stage5-candidate-revival.md`](planning/stage5-candidate-revival.md))
   is built; a human review pass surfaced the backlog in
   [`planning/stage6-review-findings.md`](planning/stage6-review-findings.md)
   (F1–F5). Land in this order, expecting iteration across the middle steps:
   1. **Persist the per-window parameter covariance** (F4 plumbing) — *done.*
      The complete second-order uncertainty of the fit (correlated error bars
      for reports, the structure the future graphical client and any
      molecular-fitting layer need), now serialized per window.
   2. **Variance–covariance peak-survival metrics** (F4 + F5) — *done and
      re-baselined*, per
      [`planning/stage6-peak-survival.md`](planning/stage6-peak-survival.md).
      An end-of-Stage-5 automatic pass: an absolute SNR floor (3.2) prunes dust
      (F5, partial windows refitted honestly through one shared in-memory
      window-refit core), and a diagonal **amplitude VIF** `(amp_err/amp)·snr`
      with a sub-half-resolution separation guard collapses degenerate overfit
      (F4, reusing the recorded doublet-alternative merged seed, overriding
      χ²/AICc). Calibration measured on the persisted covariance: the
      off-diagonal *correlation* is **not** the discriminant (it saturates near
      `|1|` for every close pair); the diagonal VIF is. Validated on the 2638
      truth set (collapses exactly w438/w217/w24; never w161/w108/w419/w281) and
      a 7-fixture re-baseline (on-vs-off recall isolation: a precision/recall
      trade — removes 22/1247 dust+degenerate lines on 1512/655 for −0.008/−0.009
      recall, all lost matches sub-3.2σ; accepted as the new baseline).
   3. **Address window construction** (F2 + F3) — *done*, overview in
      [`planning/stage4-window-assignment.md`](planning/stage4-window-assignment.md)
      §"Window margin and the content-bounded cap split". The cap split now
      bounds peak **content** (not the padded span), so a content-fitting
      cluster is never bisected at a sub-minimum interior gap; the window margin
      is a coherent points knob (`min_window_half_width_points`, default 32 =
      `trim_m`, decoupled from `edge_m`) used as the proto half-width and a
      post-construction trim that removes the empty pedestals and the
      `min_window_half_width_mhz` inertness. Cross-fixture re-fit: recall up on
      both ground-truth fixtures (1512 0.435→0.443, 655 0.518→0.521),
      `tier1_pass` up everywhere, χ²ᵣ bulk flat, 25–40% fewer/tighter/centred
      windows, no mega-windows.
   4. **Attention metrics** (F1) — *done*, a human-reviewed rework of the whole
      attention layer (details in
      [`planning/stage6-review-findings.md`](planning/stage6-review-findings.md)
      F1). Principle: high-precision flags only — automate the clear decisions,
      surface the rest on demand. Automated: drop empty (K=0) and spur-only
      windows (the latter closing #11 — single instrumental-spur line, e.g. the
      ADC image). Retired `doublet_eps_gt_kappa` (kept visible in
      `review show --window`), `low_snr`, and `worst_eps`-on-empty; fixed
      `candidate_bearing` (shape-error sidelobe filter + the `aicc_delta`
      evidence-currency bug). The decisive result, calibrated against the
      1512/655 catalog truth (frequency-within-uncertainty + amplitude-ratio
      vs catalog-intensity): **no prior-free statistic** (VIF, SNR, χ²/AICc, the
      orthogonal second-line evidence) separates a real sub-resolution doublet
      from an over-split in the 0.5–1.0 res band, which is ~92 % over-splits.
      So multiplicity is a high-bar prior-free claim: the end-of-Stage-5 pass
      **merges any degenerate close pair** (amplitude VIF ≥ 4 within 1.0 res,
      iterated) by default and flags `auto_merged_review` for opt-in
      `review split`, **with a catastrophic-merge veto** (`merge_chi2_veto`,
      default 100) that keeps the split when the merged 1-line model fits
      catastrophically (the data overwhelmingly demands two components). New
      7-fixture baseline accepted (`scratch/veto-check`): line counts down 5–61
      per fixture (over-splits removed), median χ²ᵣ flat, χ²ᵣ tail tamed by the
      veto (1512 p95 42→19), recall 1512 0.443→0.417 / 655 0.579→0.568 — a
      deliberate precision-for-conservative-multiplicity trade, every
      merge/kept-split flagged and re-splittable. (`vif_collapse_threshold` 4.0,
      `collapse_max_separation_res` 1.0 supersede the original 100/0.5 in
      [`planning/stage6-peak-survival.md`](planning/stage6-peak-survival.md).)
2. **Fork once Stage 6 stabilizes — report generation favored over the
   interactive shell.** The Stage 6 loose end is *done*: `review rank --by
   <metric>` ranks all windows worst-first by any persisted per-window statistic
   (`min-snr`, `max-vif`, `chi2r`, `candidate-evidence`, `edge-distance`,
   `spur-proximity`, `merged-chi2r`), the "surface on demand" half of the F1
   principle that replaced the retired flood flags — read-only, dual-interface
   (`Pipeline.rank_windows` / `api.rank_windows`).
   - **Reports from the `.ftmw` record** (core; **shipped**, planned in
     [`planning/stage6-reports.md`](planning/stage6-reports.md); follow-ups in
     [`planning/stage6-reports-followups.md`](planning/stage6-reports-followups.md)):
     the top-level **`report`** object (assemble-once / render-many over the
     persisted record, dual-interface) exposes two verbs: **`report run`** — the
     default deliverable, writing the Level-1 calibrated line table (CSV) **and**
     the self-contained single-file Level-3 HTML report with every window folded
     in, with flags to trim the output (`--level1-only`, `--no-table`,
     `--windows attention`, `--summary`) — and **`report table`**
     (Level-1 export only: CSV / JSON / LaTeX). The report is always one
     self-contained HTML file (the multi-file linked site was retired). The Level-2 standalone Markdown
     document was retired; its code-versioned methods prose (`_render_markdown`)
     is folded into the Level-3 report's methods page.
     **Catalog cross-reference + σ_f pull calibration shipped** (`--catalog`
     / `--catalog-nsigma` on every level; one shared reader — CSV with
     header-inferred units **and** Pickett/SPCAT `.cat` — and tolerance helper;
     opaque-label echo, never an assignment; the pull = (f_fit−f_cat)/σ_f
     distribution is the user-facing budget-calibration surface, not a gate).
     Report UX round also shipped: an interactive full-spectrum overview whose
     image doubles as the quick-nav — one shared overview image (rendered and
     embedded once) carries an SVG overlay of clickable per-window rects with a
     hover-zoom popup, attention tinting, and a "you are here" highlight on each
     window page — auto-merge (VIF-collapse) fit-history provenance,
     MathJax-rendered methods equations with interleaved distribution histograms,
     a responsive stylesheet, a single-file build (CSS inlined, figures
     base64-embedded as 256-colour palette PNGs) with a compact-mode toggle, and
     a spine-free / light-grid restyle of all spectral plots (report + `fit show`). The σ_f budget is diagnosed and settled: the
     "overly optimistic" σ_f was not a covariance/χ² bug but an uncorrected
     unlocked-digitizer-clock bias (ε ≈ 2 ppm, `Δf = ε·f_baseband`), recovered
     prior-free from the clock spurs (`calibrate_timebase`). **Design decision:
     report precision only** — ship `sqrt(σ_stat² + (σ_ε·f_baseband)² +
     σ_floor²)` with **σ_floor a user-settable accuracy floor, default 0**
     (persisted in-file, surfaced in the report), because the dominant unmodeled
     term — a per-acquisition flat absolute offset `δ_down` (~4–12 kHz) — is not
     independently determinable at the few-kHz level (the clock lattice can't pin
     it), so we do not bake it in.
   - **Report-driven curation**
     ([`planning/stage6-report-curation.md`](planning/stage6-report-curation.md))
     — proposed: turn the Level-3 HTML report into a *curation author*. Per-line
     Remove/Split/Merge and per-candidate Add controls accumulate edits into a
     client-side cart that exports a human-editable **curation file** (CSV),
     consumed by a new dual-interface **`review apply`** verb (with `--dry-run`)
     that delegates each row to the existing edit impls — byte-identical to the
     hand-typed verb sequence, add/remove coalesced to one refit per window. The
     CSV doubles as an edit *language*: **`review log`** dumps the persisted user
     edits as id'd CSV rows and **`review undo --id`** rolls them back via the
     existing decision-log replay. No new fit logic, no new persisted schema, no
     change to the conservative defaults. **Decision: retire the multi-file site**
     — the self-contained single HTML file is the only build (cart lives on one
     page); a read-only topnav toggle restores the clean document view. Phase-2
     polish: click-on-plot "add at this frequency" + on-plot SVG markers for
     queued edits, both via the data-axes bbox/limits stamped onto each figure at
     render time.
   - **Interactive CLI review shell**
     ([`planning/stage6-interactive-review.md`](planning/stage6-interactive-review.md))
     — **polish, not core**: the non-interactive verbs already cover the
     workflow and the planned C++/Qt client is the real graphical shell.
     Deferred unless prioritized.

Longer horizon, no order implied: the ultra-high-SNR **lineshape floor**
([`planning/stage5-cross-fixture-validation.md`](planning/stage5-cross-fixture-validation.md);
Phase-2 asym-τ / per-fixture shape arc; drift falsified as its cause), the
**covariance-based intra-window decomposition**
([`planning/intra-window-clustering.md`](planning/intra-window-clustering.md);
a natural consumer of the persisted covariance), benchmark/perf work
([`planning/perf-benchmarks.md`](planning/perf-benchmarks.md), deferred D5),
the **frequency-calibration / σ_f research report**
([`research/frequency-calibration-uncertainty/PLAN.md`](research/frequency-calibration-uncertainty/PLAN.md))
— consolidate the settled calibration-uncertainty analysis (ε self-cal, catalog
comparison, the SNR relationship, cross-experiment reproducibility, and the
pedestal-influence test that confirmed `δ_down` is not a pedestal artifact) into
one report backed by a self-contained reproducer; findings established, write-up
deferred (best timed after the pending third vinyl-cyanide fixture lands, which
turns the run-to-run comparison from one pair into a stable-vs-random `δ_down`
test), and **end-user documentation / release readiness** (the planning-doc
lifecycle's "seed docs" step; start opportunistically as features stabilize).

Resolved — **the SNR-survival prune now re-checks the floor after each refit.**
`apply_snr_survival_prune` (`_internal/stage5_impl.py`) iterates to a fixpoint:
it removes the single lowest-SNR sub-floor line, refits the survivors, and
re-classifies until none is sub-floor. Removing one line per pass is the
over-pruning-cascade guard — a borderline neighbour whose SNR was depressed only
by the dust recovers on the refit instead of being swept out. 7-fixture
re-baseline accepted: recall-neutral (1512 .417 / 655 .512 unchanged), window
counts identical on all seven, only the dense fixtures shed sub-floor survivors
(363 −3, 655 −6, every removed peak below 3.2σ), and the tail χ² improves
slightly (655 p95 7.22→6.97).

## Stage status

Authoritative detail in [`../STATUS.md`](../STATUS.md). Summary only:

| Stage | State | Plan |
|---|---|---|
| 0 Data import | Implemented | — |
| 1 FT processing | Implemented | — |
| 2 Noise estimation | Implemented | [`planning/stage2-noise-estimation.md`](planning/stage2-noise-estimation.md) |
| 3 Peak detection | Implemented | [`planning/stage3-peak-detection.md`](planning/stage3-peak-detection.md) |
| 4 Window assignment | Implemented | [`planning/stage4-window-assignment.md`](planning/stage4-window-assignment.md) |
| 5 Fitting | Implemented | [`planning/stage5-fitting.md`](planning/stage5-fitting.md) |
| 6 Review & finalization | In progress | [`planning/stage6-finalization.md`](planning/stage6-finalization.md) |

Stages 3–5 are partly **port-and-refine**, partly **recreate**. The earlier
reference `~/github/bcfitting/src/bcfitting/ftmwfitting.py` survives (detection
`locate_peaks`, the analytic sinc-leakage model, and the conservative
time-domain *orchestration* shell). The refined `newfitting/` engine —
`fit_time_domain_peaks`, adaptive window selection, peak aggregation — is
permanently lost and is recreated against the surviving shell's known
contract.

## Specifications

| Spec | Covers |
|---|---|
| [`API_STRATEGY.md`](API_STRATEGY.md) | Python API (Pipeline class + functional API), `.ftmw` file lifecycle, safe re-import |
| [`CLI_STRATEGY.md`](CLI_STRATEGY.md) | Command-line interface contract |
| [`SERIALIZATION_STRATEGY.md`](SERIALIZATION_STRATEGY.md) | `.ftmw` storage model and invariants |
| [`TESTING_STRATEGY.md`](TESTING_STRATEGY.md) | Test requirements across interfaces |

## Planning documents

Per-feature implementation plans. Lifecycle and conventions:
[`planning/README.md`](planning/README.md). Completed plans are archived in [`COMPLETED.md`](COMPLETED.md).

| Document | Status |
|---|---|
| [`planning/pipeline-run.md`](planning/pipeline-run.md) | **Design locked; implementing** — a top-level `run` bare verb / `api.run_pipeline` / `Pipeline.build` driving a raw source through every stage in sequence (import → FT → noise → τ → peaks → windows → fit → timebase → review, then optional `report run`). Orchestration only (calls the existing stage methods); fresh build by default; stops at `review_run` (`--report` also emits artifacts); `trim` the one required input; `calibrate_tau` always runs (shape/τ vote); start detection on by default; timebase on by default but non-fatal — auto-resolves the loader-extracted BlackChirp clocks, **warns + skips** when no clock declaration is resolvable, `--no-cal` to skip. Live per-stage progress (banner + a Stage-5 percentage bridged from the existing `plan_execution` per-window logs; TTY-aware). API exposes per-stage override dicts; the CLI forwards a `--preset`. Supersedes the stages-0–2 `workflows.process_experiment` |
| [`planning/stage6-finalization.md`](planning/stage6-finalization.md) | **Design locked; implementation in progress** — the `review run/show/edit/merge/split/accept` read/edit surface is built (Priorities track 1; remaining work in [`planning/stage6-review-findings.md`](planning/stage6-review-findings.md)). Stage 6 = a `review` stage object (`stage6_review` group, requires `stage5_fitting`; `timebase_calibration` soft input; reports require `stage6_review`) with two roles: capture *human decisions* about the automatic model and *consolidate/calibrate* the persisted final-products table. Locked: orthogonal per-window provenance (`auto`/`reviewed`/`user-edited`) + advisory attention flags, no global finalize lock (report-readiness computed; only an invalidated `user-edited` window blocks); `review` verbs `run`/`show`/`edit`/`merge`/`split`/`accept` (merge snaps to the doublet alternative; user edits bypass the accept gate but face the NLS, immune to auto-prune); candidate ledger; anchored decision log with re-apply+diff replay; one final-products table persisted in-file (raw freqs are per-window drill-down), calibration a reported STATE not a gate (frequency-reference marker on the clock declaration; default assume Rb-locked/absolutely calibrated; free-running → self-cal recommended; never blocks reports). Built in two passes (ledger+edits, then finalization layer). Absorbs the candidate-revival plan; the finalized record is the reports input contract |
| [`planning/stage6-reports.md`](planning/stage6-reports.md) | **Shipped — `report` built (`report run` = default L1 table + L3 single-file HTML; `report table` = L1-only export; L2 methods prose folded into the L3 methods page), plus the catalog cross-reference + σ_f pull-calibration surface and the report-UX round (interactive-overview navigation, auto-merge provenance, MathJax + interleaved histograms, single-file + palette-quantized figures + compact mode, responsive + spine-free restyle); follow-ups tracked in [`planning/stage6-reports-followups.md`](planning/stage6-reports-followups.md).** Reports from the `.ftmw` record: timebase-corrected frequencies + the σ_f uncertainty budget + ledgers/provenance/catalog-match summaries. **Step 1 landed:** ε applied + three-term σ_f budget consolidated into a persisted `FinalProducts` table (built by `review run`), σ_floor homed file-level (`/frequency_calibration`, default 0, `review run --sigma-floor`), calibration state derived (rb_locked / self_calibrated / uncalibrated), dual-interface + tests. **`report` feature designed (§C):** a top-level `report` object, assemble-once/render-many over the persisted record — `report table` (L1 CSV + JSON + LaTeX; spectroscopic `.lin`/SPFIT out of scope — the pipeline emits unassigned lines) and `report run` (the default: L1 table + the L3 HTML report reusing the `fit show` figure renderer, self-contained single file by default), with the L2 methods prose folded into the L3 methods page; catalog match an optional `--catalog` input. Diagnosis settled: the optimistic σ_f is not a covariance/χ² bug (the full joint `JᵀJ` is inverted; σ_f is honest precision) but an uncorrected unlocked-digitizer-clock bias `Δf = ε·f_baseband` (ε ≈ 2 ppm), recovered prior-free from the clock spurs (`calibrate_timebase`). **Design decision: report precision only.** Catalog-free run-to-run comparison (same molecule, two acquisitions) shows our random reproducibility ~2.7 kHz (below the catalog's own ~3 kHz model-precision accuracy) plus a per-acquisition flat absolute offset `δ_down` (~4–12 kHz) that is additive (ε can't touch it) and **not determinable prior-free** — the clock lattice can't pin it (upconv contamination + the candidate spurs sit on the δ-free downconv comb). So ship `sqrt(σ_stat² + (σ_ε·f_baseband)² + σ_floor²)` with **σ_floor a user-settable accuracy floor, default 0, persisted in-file and surfaced in the report** (the user's accuracy declaration; we don't assert a systematic we can't measure). σ_stat is already the CRB (ρ=0.96 vs linewidth/√SNR). Pull ≈ unit normal = user-calibration tool, not a shipped gate. Build: apply ε + persist the final-products table (incl. σ_floor) → promote the harness to the pull calibration surface → render |
| [`planning/stage6-peak-survival.md`](planning/stage6-peak-survival.md) | **Implemented and re-baselined** — end-of-Stage-5 automatic peak-survival pass closing review-findings F4 (covariance/VIF degenerate-overfit collapse) and F5 (absolute SNR-floor dust prune). Calibrated on the persisted per-window covariance: diagonal amplitude VIF `(amp_err/amp)·snr` is the discriminant (off-diagonal correlation is not), collapse gated by VIF>100 + sub-½-resolution separation and overrides χ²/AICc (high-SNR lineshape floor fools them). Phase A SNR prune (partial windows refitted through one shared in-memory window-refit core), then Phase B VIF collapse (reuses the doublet-alternative merged seed); built on the step-1 covariance persistence. **Merge-policy update (supersedes the original thresholds):** prior-free, a sub-resolution split is a high-bar claim and the ambiguous 0.5–1.0 res band is ~92 % over-splits (no statistic separates real doublets from over-splits — VIF as high for both), so the collapse is the *default merge* for any degenerate pair (`vif_collapse_threshold` 100→4, `collapse_max_separation_res` 0.5→1.0), with a **catastrophic-merge veto** (`merge_chi2_veto` 100) keeping the split when the merged fit is catastrophic, plus end-of-Stage-5 window cleanup (drop empty + instrumental-spur-only windows). Merges flag `auto_merged_review`, vetoed splits flag `overfit_vif`. New 7-fixture baseline accepted |
| [`planning/stage6-review-findings.md`](planning/stage6-review-findings.md) | **All findings (F1–F5) resolved** — F2/F3 (window construction) and F4/F5 (peak-survival) shipped; F1 (attention layer) reworked to high-precision flags + merge-by-default multiplicity with a catastrophic-merge veto (see Priorities track 1.4). Original triage backlog preserved below for provenance. **F1:** attention flagging is low-precision (~1-in-5 sampled windows needed an edit); `doublet_eps_gt_kappa` dominates (33/76 flagged) yet is the calibrated "doublet required" *observation* the fit usually got right, and no reason kind targets a weak/implausible fitted peak (the SNR≈2 overfit case) — candidate fixes: drop/stiffen the doublet attention trigger, add a low-evidence-peak reason. **F2:** window boundaries split inside sub-minimum gaps leaving features off-centre — windows 309/310 bisect a <1.1 MHz cluster at a 0.39 MHz Stage-3 gap (309 came out 3.77 MHz, below the 4 MHz `min_window_half_width_mhz` floor); the split arises because Stage 4 windows the sparse Stage-3 detections but Stage-5 rescue fills the gap, and the structural-replan merge net did not fire — candidate fixes: don't split a cluster that fits one min-size window / enforce the floor on split products (Stage 4), investigate the replan merge trigger (Stage 5), and possibly a window-level merge verb (Stage 6). **F3:** `clustering.min_window_half_width_mhz` (default 2.0 MHz) is inert — its one consumer `max()`-es it against `edge_m=64` bins so the bin count always wins (the MHz value would only bind above ~5 MHz), mirroring by accident the deliberate points-supersede-MHz on the max side; and no minimum *final* window width is enforced at all (the F2 enabler). Cleanup: retire the stale knob or re-express+enforce the minimum as a bin/points count. **F4:** the fit (co)variance is an unused overfit signal across **all** fitted parameters, not just amplitude. The variance-inflation factor `(amp_err/amp)×snr` is ~1 for an identifiable line and ≫1 for a non-identifiable one, but the signal can live in frequency or phase instead; each parameter needs its own normalization (amplitude vs value, frequency vs resolution element/separation, **phase vs π** — not vs the phase value). Truth set: w217 (high-SNR amplitude degeneracy — one line as two anticorrelated components at 0.17 res-elements, VIF~980 despite SNR 500-960), w281 (low-SNR pair where amplitude VIF is only ~4.5 but frequency error ~30% of the separation flags it; the phase *value* is not diagnostic — no empirical frequency–phase relationship on this instrument, so only a phase's error/correlation counts, never its value or inter-line difference), w161 (genuine identifiable doublet at 0.82 res-elements, VIF~2), and two guard cases that must NOT collapse: w108 (legitimate low-SNR quartet whose raw amplitude errors are large — 22-32%, bigger than 217's *good* peak — but VIF stays ~2.1-2.5, so a raw-error threshold would wrongly collapse it and only the SNR-normalized VIF/correlation keeps it) and w419 (true blended doublet with no visible dip — 1.79×FWHM apart, 3.8:1 amplitudes so the weak line is an unresolved shoulder; VIF ~2-3 = keep, and its χ²ᵣ 2.98 is the strong-line lineshape floor not overfit, so the covariance reads it correctly where a dip-heuristic or χ²ᵣ would not). The intended attention metric flags a window when any normalized parameter error or pairwise (anti)correlation exceeds a threshold (the high-precision overfit reason F1 lacks); legit cases sit at VIF~1-2.5 regardless of SNR, overfit explodes. Cheap diagonal flag needs no new persistence (`amplitude_error`/`frequency_error`/`phase_error`+`snr` already stored); the correlation part needs the off-diagonal covariance, computed at fit time but not serialized. **Agreed follow-up: persist the full per-window parameter covariance** (the complete second-order uncertainty of the fit — correlated error bars for the final-products table/reports, the structure the Qt frontend + molecular-fitting layer need; serialization-design task: a `stage5_fitting` per-window covariance dataset with documented parameter ordering, SERIALIZATION_STRATEGY update, tri-interface round-trip). Feeds F1's missing overfit attention reason. **F5:** fitted peaks below an absolute post-fit SNR floor (~3) should be auto-discarded — w441 is four pure-dust peaks (SNR 1.5-2.4) and file-wide 32/603 peaks are sub-3 across 16 all-dust windows. Independent of F4: F4 catches *degenerate* dust (441 A/B, VIF~9-10) but not *absolute-weak* dust (441 C/D, VIF~0.6 yet SNR~1.5); the SNR floor is the orthogonal cut. Safe — every truth-set keep is SNR≥7; anchor the floor to the Stage-3 detection threshold (~3.2). Dust reaches the model as rescue/split products whose post-fit SNR fell below promotion with nothing pruning them → needs a final post-fit cleanup pass (prune+drop emptied windows). Items graduate to their own doc on prioritization |
| [`planning/stage6-report-curation.md`](planning/stage6-report-curation.md) | In progress — report-driven curation: the Level-3 HTML report becomes a *curation author*. **Shipped:** the curation-file format + parser, **`review apply`** (`--dry-run`, add/remove coalescing to one refit per window, frequency-resolution warnings), and read-only **`review log`** — all dual-interface with unit + integration tests; plus the multi-file-site retirement. **Blocked/next: `review undo --id`** needs the automatic Stage 5 fit snapshotted + a replay engine (edits mutate `stage5_fitting` in place; no baseline exists), then it reuses `apply_curation_impl`. **Remaining:** the in-report controls/cart/export + phase-2 plot interaction. Per-line Remove/Split/Merge + per-candidate Add controls (data already on the page — every review verb keys on `(window_id, molecular MHz)`) accumulate edits into a client-side cart that exports a human-editable **curation file** (CSV; order-preserving, comment-friendly, hand-editable, the canonical interchange), consumed by a new dual-interface **`review apply <curation-file>`** verb. `apply` delegates each row 1:1 to the existing edit impls (`refit_window_impl`/`merge_peaks_impl`/`split_peak_impl`/`review_accept_impl`), coalescing a maximal run of same-window add/remove into one `review edit` refit (union resolved against the rendered model state; merge/split/accept are barriers), warns on frequency-match ambiguity (two fitted peaks within one resolution element), and offers `--dry-run` to preview the resolved plan without mutating. Central correctness lever = byte-identical `.ftmw` vs the hand-typed verb sequence. The CSV is also an edit *language*: **`review log`** lists persisted user edits as id'd CSV rows (a dump re-parses as a curation file) and **`review undo --id N`** rolls back one or more by replaying the decision log minus the named ids. Adds **no** fit logic, **no** new persisted schema (the curation file is external; edits persist as the verbs already do — decision log, `origin=user`, provenance flip), and does **not** change the conservative automatic defaults. **Resolved decisions:** retire the multi-file site (single self-contained HTML is the only build → in-memory cart, no cross-page persistence); export = download + `<textarea>` fallback; coalesce add/remove by default; log the edit not its source; CSV only (JSON deferred). Static `file://` artifact → controls only *emit* text; vanilla JS only (no Plotly/React); a read-only topnav toggle restores the clean document view. **Phase 2 polish:** click-on-plot "add peak at this frequency" + on-plot SVG markers for queued edits — since we own figure generation, stamp the magnitude panel's data-axes pixel bbox + frequency limits as `data-axes-*` attributes at render time (`plot_window_panels`) so a transparent overlay inverts a click-x to molecular MHz and positions the queued-edit glyphs (~30 lines vanilla JS; a plain freq input gets most of the value first). Third client of the same Stage 6 edit contract, complementary to the interactive shell (batch/async vs terminal-REPL) |
| [`planning/stage6-interactive-review.md`](planning/stage6-interactive-review.md) | Proposed — terminal-interactive review shell over the existing `review` verbs. A CLI-only REPL that keeps the *current window* as session state, renders it to an image **file** (no live matplotlib, no `ipywidgets` — both ruled out), and offers a single-key menu (add `F` / remove-by-display-letter / merge / split / accept / revive-candidate) whose actions delegate 1:1 to `refit_window_impl`/`merge_peaks_impl`/`split_peak_impl`/`review_accept_impl` — so an interactive `r B` is byte-identical to `review edit --remove <freq-of-B>`. Adds **no** fit logic and **no** persisted state; quitting always leaves a consistent file. Three entry selections: attention queue (severity-ranked), browse-all, or start-nearest-`--near FREQ`. The one new coupling is the display-letter→peak map, which must reuse `_peak_labels` so figure/table/menu agree and re-derive after each edit. Interim tool for the planned C++/Qt graphical shell (the two are clients of the same Stage 6 decision-log contract); file-based rendering keeps the CLI out of the graphics-shell role. Test lever: scripted-stdin run must produce a byte-identical file to the equivalent non-interactive verb sequence |
| [`planning/stage5-candidate-revival.md`](planning/stage5-candidate-revival.md) | Proposed — candidate ledger + user-directed window re-fit. The remaining cross-fixture misses are weak near-blend lines the gates *considered and rejected as marginal* (verified in the persisted audit record on 1231 w50/w425/w426 + 363 w76); lowering the automatic bars to capture them buys dust everywhere else, so the principled lane is human arbitration: surface rejected-but-plausible candidates (normalized/deduped, above a display bar) with the fit, and add a window-scoped `fit refit --window N --add F --remove F` verb (dual-interface) whose user edits bypass the accept gate but carry full provenance (`user` origin flag, audit `user-add`/`user-remove`, curated-vs-automatic separation in validation tooling). Also covers the overfit direction (user removes a peak on imperfect-lineshape ultra-high-SNR windows, e.g. 1019). UX is the primary design consideration |
| [`planning/stage5-cross-fixture-validation.md`](planning/stage5-cross-fixture-validation.md) | Planning — per-dataset shape-error ε calibration framework; cross-fixture acceptance metrics; covers the lineshape-deficit physics discovery from Phase 1 validation on 2638 |
| [`planning/intra-window-clustering.md`](planning/intra-window-clustering.md) | Stub — covariance-based intra-window decomposition; supplants Stage 5 `split`. Same frequency-coupling block structure the NLS-performance plan would exploit in-fit (block-diagonal `JᵀJ` ⇔ block-diagonal covariance) |
| [`planning/perf-benchmarks.md`](planning/perf-benchmarks.md) | Deferred (D5) |

## Code vs spec divergences

The specs state intended requirements. Where the code diverges, it is recorded
here and resolved deliberately (amend spec, or change code), never silently.

| # | Divergence | Resolution |
|---|---|---|
| D1 | CLI command names: code used `data-load`/`ft-process`/`ft-visualize`/`data-visualize`/`data-info` vs spec verb-object intent | **Resolved (code):** renamed to `import-data`, `compute-ft`, `visualize-ft`, `visualize-data`, `formats`. `estimate-noise`/`visualize-noise` already conformed. Stage 3+ commands follow the verb-object scheme |
| D2 | `Pipeline("x.ftmw")` smart constructor required by spec, not implemented | **Resolved (code):** `Pipeline(path)` opens if present, raises `FileNotFoundError` with guidance otherwise; `create()`/`open()` unchanged |
| D3 | Spec showed functional API as top-level `import ftmwpipeline as ftmw; ftmw.import_data(...)` | **Resolved (spec):** the canonical functional namespace is `ftmwpipeline.api` (`import ftmwpipeline.api as ftmw`); only `process_experiment`/`batch_process_experiments` are top-level. API_STRATEGY amended to match |
| D4 | CLI `info <file>` + machine-readable `--format json` not implemented | **Resolved (code):** `info` command added with `text`/`json` output |
| D5 | Performance/benchmark tests absent; storage-reduction figures unmeasured | **Deferred (tracked):** specs already made unmeasured figures non-normative; benchmark work tracked in [`planning/perf-benchmarks.md`](planning/perf-benchmarks.md). `tests/performance/` remains empty until then |
| D6 | `io/complex_ft_serialization.py` never invoked by the pipeline | **Resolved (code):** module and its tests removed; the serialization spec prohibits persisting ComplexFT, so it was dead by design |
| D7 | User-chosen FT processing settings (trim, zpf, expf, …) are not persisted as canonical state; later stages silently fall back to import-time *recommended* defaults instead of what the user chose. Stage 3 currently masks this with interim per-stage `trim`/`zpf` options | **Resolved (code + spec):** Stage 1 now persists user-chosen settings (incl. trim) as canonical; Stages 2–5 operate on that grid; changing canonical settings invalidates downstream results. Stage 3's interim `trim`/`zpf` ownership removed from `_internal/stage3_impl`, `Pipeline.detect_peaks`, `api.detect_peaks`, and `cli/peak_commands.py`. `SERIALIZATION_STRATEGY.md` and `API_STRATEGY.md` amended with normative canonical-settings text |
| D8 | Truncation-leakage handling is wrong in both Stage 3 and Stage 4. Stage 3's gap pass promotes a strong line's sinc sidelobes as weak lines — its leakage mask (`estimate_leakage_reach`) is 7–25× too narrow vs the real ±20+ MHz coherent skirt. Stage 4's edge statistic `S_coh` (a coherent windowed sum) cancels on the oscillating sinc skirt and reads noise-level over obvious leakage, so its leakage-touched map, fixed-contributor attachment, and difficulty classification are unreliable on real data | **Resolved (code + docs):** root cause is the full-record rfft phase ramp `exp(±i2πf·t₀)` (`t₀=start_us`) that makes truncation leakage oscillate so a coherent sum cancels on it. Fixed by one shared de-ramp to the active-region turn-on (`deramp_to_active_start` / `leakage_touched_intervals` in `preprocessing/leakage.py`) feeding the existing `S_coh` — no new statistic. Stage 4's edge-coherence calls and Stage 3's gap-pass mask both consume the de-ramped leakage-touched map; `estimate_leakage_reach` is demoted to an unused analytic proposal; `T_edge` calibrated to 8 for both stages. Both research reports (`research/peak-detection`, `research/complex-edge-coherence`) revised. Implementation overview in [`planning/leakage-detection-rework.md`](planning/leakage-detection-rework.md) |
| D9 | Stage 5 fits on the persisted Stage 1 spectrum, an rfft of the *whole* zero-padded record. Adjacent bins are correlated by a Dirichlet kernel (the FFT of the zero-padding indicator), so the effective number of independent samples in any band of `M` bins is `M·α` with `α = N_active/N_padded` (≈ 0.42 for 2638). The naive `N_dof = M − N_params` overcounts by `1/α`; reduced χ², F-test, AIC, and the conservative loop's accept thresholds (incl. the blend-aware seeder's `rchi2 > 1.5` trigger) are all biased optimistic on real data | **Resolved (code + spec):** Stage 5 fits on the **active-portion FT** — `_internal/stage5_impl.py` calls `compute_active_ft` and routes the resulting independent-bin spectrum into `execute_plan`. The σ/√2 weighting and per-bin noise close out the statistics. Stage 5 carries an explicit dependency on `stage0_fid_data` alongside `stage4_windows`. Plan text in [`planning/stage5-fitting.md`](planning/stage5-fitting.md) §"Spectral domain for the fit". **Amended (active-FT noise authority):** the noise reference moved from per-stage re-measurement to a single estimator and grid. Stage 2 now measures and persists its σ on the **canonical active FT** via the scatter estimator (`estimate_active_ft_noise` / `_internal/active_ft_support.py`), and Stages 3 (snap/score), 4 (window planning), and 5 (fit weighting + replan) all consume that active-FT σ — noise is zpf-invariant and varies only with active length + apodization, so measuring once in active-FT space needs no analytic transfer. The front-zeroed full-record persisted FT is **no longer** a scoring/detection/planning domain; it survives only as the Stage 0/1 start-time comparison view. The legacy `estimate_noise_adaptive` kernel is retired (minimal comparison reference at `dev-docs/research/noise-snr-scaling/legacy_adaptive.py`) |
| D10 | The Stage 5 cross-fixture Tier-1 acceptance gate (`planning/stage5-cross-fixture-validation.md`: per-window reduced χ² median ≤ 1.5, p95 ≤ 4, max ≤ 10) is **unachievable on any high-SNR fixture**. At extreme SNR the per-window reduced χ² is a model-fidelity floor, not a noise statistic: a sub-percent lineshape/τ deficit becomes hundreds of σ/bin under a SNR ~10⁴–10⁵ line, so χ²ᵣ tracks SNR² (655: ≥10⁴ → χ²ᵣ ~10⁵) even where the line is fit to part-in-10⁵ (freq recovered to 0.2 kHz). Window sizing, cross-window leakage, and τ were each falsified as levers (`research/stage5-cross-fixture/report.md`) | **Resolved (spec):** the Tier-1 gate is reformulated SNR-aware — pass iff `χ²ᵣ ≤ F + (κ·SNR_max)²` with the fractional deficit `ε = √(max(χ²ᵣ−F,0))/SNR_max` reported alongside; κ defaults to `0.05` (just above the measured ~1–3% vinyl-cyanide deficit) and the noise-regime allowance `F` defaults to `3.0` (budgets for the reduced-χ² sampling scatter of a good fit at low SNR, where the deficit term is negligible — without it the gate rejects healthy noise-dominated windows). It collapses to χ²ᵣ≤F in the noise-dominated regime and grows ∝SNR² in the deficit-dominated regime. Implemented as `snr_aware_chi2_pass`/`shape_error_fraction`/`DEFAULT_SHAPE_ERROR_KAPPA`/`DEFAULT_CHI2R_NOISE_FLOOR` in `fitting/validation.py`, consumed by the `validate-stage5-shape-error` command. The Tier-1/2/3 *skeleton* is unchanged; only the lever it calibrates changed. `planning/stage5-cross-fixture-validation.md` §"Tier 1" amended. **Consequence (theme T4):** because the SNR-aware gate handles the chi²ᵣ ~ SNR² shape-error floor at the acceptance layer, the earlier per-dataset `shape_error_epsilon` rescue-screening knob is redundant — it defaulted 0.0 in production (never plumbed through `stage5_impl`) and is removed from `attempt_residual_rescue` / `rescue_and_consolidate` |
| D11 | The Stages 2–5 settings `resolve()` orders the layers `explicit > preset(.yml) > persisted(.ftmw) > recommended > default`, so an instrument `.yml` preset **overrides** the value persisted in the `.ftmw`. This breaks the Principle-4 reproducibility paradigm: a shared `.ftmw` must reproduce identical output from the file alone, but a recipient's `.yml` (possibly tuned for a different instrument) would silently change the result. It also contradicts the Stage 1 canonical-settings order (`explicit > persisted > recommended`) already specified in `SERIALIZATION_STRATEGY.md` and `planning/processing-settings-persistence.md`; the preset layer was inserted above persisted only in the later Stages 2–5 backfill | **Resolved (code + spec).** `SERIALIZATION_STRATEGY.md` §"Settings resolution and reproducibility" pins the normative order `explicit > persisted(.ftmw) > preset(.yml) > recommended > default` (preset *seeds* unfixed fields, never overrides persisted ones). The layer tuple is flipped in every Stages 2–5 `resolve()` (`noise_settings`, `tau_calibration_settings`, `peak_detection_settings`, `window_planning_settings`, `stage_fit_settings`); the per-field merge and the explicit-over-persisted escape hatch are unchanged. Each stage's unit suite asserts persisted beats a conflicting preset (`test_persisted_beats_preset`); the settings-propagation and cross-interface suites are re-baselined. Stale precedence text reconciled in `planning/settings-backfill.md`, `planning/stage5-fit-settings.md`, `planning/companion-tuning-tools.md`. Was the prerequisite task of the `settings`/`scan` verbs, tracked in [`planning/tune-settings-verb.md`](planning/tune-settings-verb.md) |
| D13 | The initial vision exposed user apodization of the canonical FT (`expf_us` exponential apodization, `window_function`/`winf` FID window, `zpf` zero-pad factor) as Stage 1 settings. This is the wrong tool given the pipeline's statistical foundations: apodization trades resolution and biases the line shape, and zero-padding interpolates the spectrum bins and corrupts the Stage 2 scatter-noise authority and the Stage 5 χ² statistics. The robust per-window finite-`T` `h_T` fit (with the leakage-wing baseline) is the intended alternative | **Resolved (code + spec):** the three knobs are removed everywhere — `FTSettings`, `FID.preprocess` / `FIDProcessingParameters`, `compute_ft` / `compute_active_ft`, the data loaders + `ft_processing`/`recommended_processing` serialization, the CLI flags (now rejected by argparse), and the `settings`/`scan` knob surfaces. The canonical FT is unconditionally **unapodized, un-windowed, native-length**; only data selection (`start_us`/`end_us`/`trim`) and display (`units_power`/`rdc`) remain. Stage 5 τ₀ now defaults to the per-band Stage 2b `τ_maj` (else band-wide `τ_maj`, else `T_active/3`); Stage 3's `tau_basis_us` drops its `expf_us` fallback. Legacy `.ftmw` files carrying the retired keys open with a warning (`file_manager._warn_legacy_ft_apodization_keys`) and are recomputed unapodized. `API_STRATEGY.md` and `SERIALIZATION_STRATEGY.md` canonical-settings text amended; the 2638 `standard_ft_params` and all derived fixtures rebuilt unapodized with value-specific assertions re-baselined. Implementation plan in [`planning/ft-apodization-removal.md`](planning/ft-apodization-removal.md). Out of scope (kept): the `fit show --apodize` display-only windowed view and Stage 3's internal sub-bin position-finding zero-pad |
| D12 | The CLI is flat **verb-object** (`compute-ft`, `estimate-noise`, `visualize-ft`, …) with the tuning surface as a `tune <subcommand>` namespace — the one command that broke the verb-object contract. The grammar is being moved wholesale to **object-verb** (the object is the pipeline stage), which uniformly groups each stage's execute (`run`) + visualize (`show`) with `stageN` synonyms, and groups the cross-cutting tools under `settings` / `scan` meta-objects | **Meta-objects landed; stage-command migration pending.** `CLI_STRATEGY.md` specifies the object-verb grammar (`<stage> run\|show`, `data import`, `tau show --kind …`, `fit check`; meta-objects `settings {show,set,export}` / `scan {list,run,all}`; bare utilities `info`/`formats`/`validate`/`version`). The **meta-objects are implemented** (issue #28): `settings {show,set,export}` ship, and the legacy `tune` namespace is removed and replaced by `scan {list,run,all}` (Python wrappers `scan_list`/`scan_run`/`scan_all`). The **stage** commands are migrated to `<stage> run\|show` (issue #31, [`planning/cli-object-verb-migration.md`](planning/cli-object-verb-migration.md)): every stage object carries `run`/`show` and a `stageN` synonym; `data import <file> <src>`, `tau run --gaussian`, `tau show --kind …`, `fit check`; utilities stay bare. Pre-release hard cutover — the old flat commands are removed |
| D14 | The noise-authority spec (D9 amendment: Stage 5 fit weighting consumes the canonical active-FT σ, measured on the **trimmed** analysis band) was violated by `fit_peaks_impl`: it re-ran `estimate_active_ft_noise` on the **untrimmed** full active grid (0–f_Nyquist baseband). Out-of-band bins carry no chirp/receiver energy, so on instruments whose out-of-band floor sits well below the in-band floor the region-aware estimate is dragged down — on the succinimide UXR fixture (10.5 GHz analysis band on a 64 GHz grid, ~85% quiet bare-ADC floor) the fit's in-band σ came out **3.4× too small**, inflating every significance currency ×11–13: reported χ²ᵣ read 8–13 on windows whose honest χ²ᵣ ≈ 1, and the rescue/add lanes installed thousands of noise-grass peaks (10,866 fitted lines, 7,171 at apparent SNR 5–10 = true 1.5–3). Home fixtures masked the bug because their out-of-band floor ≈ in-band floor (16 GHz scope bandwidth; 13.5 of 25 GHz active). The replan context, `fit show`, and `fit check` paths already used the trimmed authority (`build_active_grid_with_noise`), so the renderer plotted honest noise under inflated fit statistics — the visible inconsistency that exposed the bug | **Resolved (code):** `fit_peaks_impl` re-measures the in-band σ on the trimmed band alone (same persisted scatter knobs; matches `build_active_grid_with_noise` by construction) and passes that to `execute_plan` as the fit weighting. The full-grid estimate is retained for the spur sweep only, whose nomination floor and out-of-band clock anchors were calibrated on the full grid. Home-fixture fitted tables shift under the honest σ → cross-fixture re-validation / reference re-baseline required |

New divergences are appended here as they arise.

Exact on-disk HDF5 group/attribute names are an implementation detail; the code
is the source of truth and the current layout is recorded in `STATUS.md`. The
serialization spec defines invariants, not literal field names.

## Conventions

- Keep this file short. Detail belongs in `STATUS.md`, the specs, or
  `planning/` docs.
- A new Stage gets a `planning/` doc *before* implementation; the doc is
  registered in the table above.
- Resolve divergences explicitly; update the relevant spec or code, then strike
  the row.
- No emojis, no dated "status" prose, no per-commit narrative.
- Completed planning docs and closed roadmap items move to `COMPLETED.md`; the roadmap stays forward-looking.
