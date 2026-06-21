# Handoff: the Stage 5 fitting-statistics methods note

Fresh-session brief for writing the **last** of the agreed methods-note set. The
methods section currently has four notes; this adds the fifth and finishes the
promotion workstream.

## Where this sits

Active workstream: promote `dev-docs/research/*` reports into
`docs/source/methods/` as timeless "why can I trust this" notes. Done and
committed:

- `methods/noise_snr_scaling` (pre-existing)
- `methods/matched_filter_detection` (4c3a11b)
- `methods/edge_coherence` (2126fbf)

Agreed remaining: **one merged Stage 5 note**. Then the methods-promotion
workstream is done and the docs effort returns to its prior NEXT (Reference
section: cli / api / changelog — see [[user-docs-effort]]).

Reports that are NOT promoted (decided): `stage5-tau-calibration` (archive —
`stage2b_tau.rst` already absorbed it), `frequency-calibration-uncertainty`
(defer — pending final fixtures; shipped decision already in stage6/clock pages),
and the various `*-gaussian-audit` / `peak-detection` / `coherence-screen` /
`stage4-poststage23` / `performance-profiling` reports (archive-only).

## What the Stage 5 note is

**Merge three reports into ONE note** (they are halves of one story):

- `dev-docs/research/stage5-fitting/report.md` — the finite-T model, the
  add-one-peak loop, sigma weighting, the blend study.
- `dev-docs/research/residual-rescue/report.md` — the evolution of the accept /
  knockout / merge gates from the prototype's F-test to the shipped
  **AICc-with-`n_eff`** criterion, plus the rescue chain.
- `dev-docs/research/stage5-cross-fixture/report.md` — **fold in** the
  SNR-aware chi^2 gate and the "2638 defaults generalize across the SNR decades"
  validation. Do NOT make this a standalone note.

### Unique content the note should add (everything else is already in the stage page)

`docs/source/stage5_fitting.rst` is already substantial and covers the "how" at
user altitude. Read it first and add ONLY the derivation / statistics / validation
residue it does not contain:

1. **The AICc-with-`n_eff` gate derivation** — the highest-value content. Why a
   full-`n_data` F-test is structurally biased toward the more-complex model (the
   ~hundreds of quiet bins dilute the denominator); the information-weighted
   effective sample size `n_eff` (`perplexity_log1p_snr`, weight `log(1+SNR)`)
   that counts only the bins the lines inform; the self-consistent AICc form;
   REJECT-on-tie. This is the gate that the stage page already (correctly, since
   commit 4c3a11b) describes — the note derives it.
2. **Blend recovery-vs-detectability** — the quantitative result that blends are
   always detectable and recoverable, but the sequential loop fails on
   *initialization*, which is why the blend-aware straddling K=2/K=3 seeder exists.
3. **The residual-rescue chain** — option-B joint refit (per round, against the
   original data, not residual recursion); decay-aware rescue tau; the two-tier
   merge (tier-1 unconditional sub-0.5-FWHM structural; tier-2 AICc-gated,
   currently disabled at `merge_separation_factor == structural_merge_factor ==
   0.5`); the removed phase-coherence projection test.
4. **The SNR-aware chi^2 gate + cross-fixture generalization** — `chi2r <= F +
   (kappa*SNR)^2`, the fractional deficit, the irreducible ~1% bulk fidelity
   floor at extreme SNR; that the 2638 defaults hold across ~3 decades of SNR.
   This is the `fit check` surface (`stage5_fitting.rst` ~lines 442-448).

### Staleness to handle (verified against current code)

- **F-test framing is superseded.** Both prototype reports describe the accept
  gate as "F-test (p<0.05) AND AIC decrease." Shipped is AICc-with-`n_eff`
  (`window_fit.py` ~2722-2753; the F-test survives only as a recorded diagnostic,
  ~1417). The stage page is already fixed; the note must present the F-test as
  historical, not current.
- **No per-window peak cap.** `DEFAULT_MAX_PEAKS = 0` (`window_fit.py`),
  `conservative.max_peaks` default 0 in `stage_fit_settings.py`. Drop the
  "patience" / peak-cap framing; the loop is candidate- and width-bounded.
- **Apodization is gone.** The reports quote `T=12.65us, tau_eff=5us` and
  attribute it to `expf_us=5` apodization — that knob no longer exists (canonical
  FT is unconditionally unapodized). State 2638's tau_maj~5us as a *measured*
  decay; the residual-rescue Section 7 chi2_r distributions were measured on the
  removed apodized fixture and won't reproduce — do not quote them.
- **Headline asym-tau result is dormant/falsified.** The cross-fixture report's
  asymmetric long-anchor tau penalty was falsified for the bulk by its own Phase
  2 and is dormant in code (`tau_anchor_us` / `tau_penalty_sigma_lo_factor` have
  no production caller). Drop it from the note.
- **All cross-fixture numbers are stale** (4+ re-baselines since). Regenerate or
  snapshot fresh; correct stale defaults if quoted (`fit_tau_min_snr` is 10, not
  50; `max_peaks` 0, not 8).
- **Post-report shipped methodology the reports never had** — acknowledge so the
  note isn't complete-but-stale: trimmed-active-grid sigma authority (D14),
  the survival-prune + VIF-collapse pass, and Stage 5's consumption of the
  shape-matching Stage 2b tau twin. These need not be re-derived, just named.

## The harness pattern (copy from the two notes just built)

`methods/matched_filter_detection/` and `methods/edge_coherence/` are the
templates. Each note is:

- `methods/<name>.rst` — `.. index::` block; timeless present-tense prose;
  derivation + validation; cross-link the stage page; a "Reproducing this note"
  section. Voice: do not explain the audience to itself; American English; no
  source-evolution markers ([[feedback-docs-writing-voice]]).
- `methods/<name>/generate.py` — drives the **shipped** functions where possible
  (so the note validates production code, not a prototype). Flags `--no-2638` /
  `--no-figures`; writes `figures/*.png` + `results.json`. Figures use the brand
  palette via `ftmwpipeline.visualization.report_style` (`apply_bare_style`,
  `apply_color_cycle`, `BRAND_CYCLE`, `AGGIE_BLUE`/`AGGIE_GOLD`, etc.); no titles
  (captions label them). Heatmap text-contrast rule: white if value > 0.5 on
  `aggie_blue_cmap`.
- `tests/unit/preprocessing/test_<name>_invariants.py` — fast, data-free; loads
  `generate.py` by `importlib` path; asserts the closed-form / synthetic claims.
- `tests/integration/test_<name>_report.py` — `slow`-marked drift guard;
  regenerate the cheap parts and check against committed `results.json`.
- Wire: add to the `index.rst` Methods toctree + the landing-page list; cross-link
  from `stage5_fitting.rst`.
- Build: `cd docs && conda run -n ftmwpipeline-dev sphinx-build -W -q -b html
  source build/html` must be warning-clean.
- Lint: `black --target-version py311`, `isort`, `mypy` on the harness/tests. The
  docs harness carries env-only `import-untyped` / `no-any-return` mypy notes
  (same as the noise/MF/edge harnesses) — acceptable; fix only genuine errors
  (var-annotated, etc.).

### Cross-fixture harness — the one real decision, already made

The 7-fixture roll-up is **lightweight + snapshot JSON** (chosen). A full Stage 5
fit on all 7 fixtures is minutes (655 ~236s), too heavy for a `slow` drift test.
So:

- Snapshot the 7-fixture roll-up (recall where available / chi2r distribution /
  pass rate) as a committed `results.json`, NOT regenerated in the slow test.
- The slow test regenerates only a cheap figure (e.g. the SNR-aware gate curve
  from cached per-window chi2r/SNR, or 1-2 light fixtures) and checks the
  snapshot's invariants.
- Reusable builder: `scratch/edge-fix-rebaseline/rebaseline.py` already builds all
  7 fixtures fresh through Stage 5 and emits per-fixture metrics (windows,
  contributors, peaks, chi2r median/p95/max, runtime). Adapt it (add recall vs
  catalog for the ground-truth fixtures if recall is wanted — 1512/363/360/655
  have catalogs per [[fixture-splitting-physics]]).

### Fixture-build gotchas

- **Rebuild fixtures FRESH** from `examples/blackchirp_data/<id>/`; never reuse an
  old `.ftmw` (persisted settings outrank code defaults, D11)
  ([[feedback-rebuild-fixtures-fresh]]). Recipe: `import_data(force=True)` ->
  `detect_start_time(band=(26500,40000), stamp=True)` -> `compute_ft(trim=...)`
  -> `estimate_noise` -> `calibrate_tau` -> `detect_peaks` -> `assign_windows`
  -> `fit_peaks`. Pass NO explicit shape (the Stage 2b vote stamps it).
- **BLAS is fixed** (commit 9266e1a, threadpoolctl) — fixture builds no longer
  thrash and need no shell `OMP_NUM_THREADS` workaround. Still run fixtures
  **sequentially** (the per-fixture fork pool already saturates cores; concurrent
  fixtures oversubscribe — [[fixture-baseline-parallelize]]).
- All commands via `conda run -n ftmwpipeline-dev`; `-o addopts=""` drops coverage.
- Direct all artifacts to gitignored `scratch/`.

## Process

Per the per-page gate ([[user-docs-effort]]): read the stage page + the three
reports + the code, then DISCUSS the note's framing/scope with the user BEFORE
writing (the matched-filter note used an AskUserQuestion for scope; the user
chose "+2638 validation"). For Stage 5 the open scope question is what the
cross-fixture figure should show and whether to compute recall (needs the catalog
readers).

Commit with the `commit` skill (timeless message). Update [[user-docs-effort]]
and add a memory entry when done.
