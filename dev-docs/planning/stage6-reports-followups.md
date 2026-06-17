# Stage 6 reports — follow-up items (handoff)

Handoff for a fresh session. The three report levels are built and polished:

- **L1 `report table`** — CSV / JSON / LaTeX of the persisted `FinalProducts`.
- **L2 `report summary`** — Markdown methods + results; an HTML twin
  (`methods.html`) is rendered inside the L3 site with distribution histograms.
- **L3 `report full`** — linked HTML site: `index.html` + `methods.html` +
  `windows/window_NNN.html` per fit window, `figures/`, `assets/style.css`.

All three are dual-interface (`Pipeline.report_*` / `api.report_*` /
`cli report …`) and read-only over the persisted record (never recompute the
fit). Code map:

- `src/ftmwpipeline/_internal/report_impl.py` — L1 + L2. `_SummaryModel`,
  `assemble_summary_model`, `_assemble_summary`, `_render_markdown`,
  formatters `_concise` (value(unc) BCE) / `_freq` / `_g` / `_scaled` /
  `_amplitude_unit`, `_percentiles` / `_percentile_table`.
- `src/ftmwpipeline/_internal/report_html_impl.py` — L3 + the HTML methods page.
  `report_full_impl` (driver), `_md_to_html`, `_summary_page`,
  `_summary_distribution_specs`, `_covariance_block`, `_audit_block`,
  `_param_symbol_html` / `_param_symbol_mathtext` / `_param_value`,
  `_window_peak_table`, `_index_*`. The single stylesheet is `_STYLESHEET`.
- `src/ftmwpipeline/visualization/fit_detail.py` — figures.
  `plot_window_panels`, `plot_correlation_heatmap`, `plot_summary_histograms`,
  `frequency_sorted_labels` (the one source of the peak display letters).
- `src/ftmwpipeline/core/data_structures.py` — `FinalPeak` / `FinalProducts`
  (calibrated line list + σ_f budget breakdown: `sigma_f_khz`,
  `sigma_stat_khz`, `sigma_eps_khz`, `sigma_floor_khz`, `f_baseband_mhz`).

Suggested order: **1 catalog cross-ref → 2 pull surface → 3 survival-prune
recheck → 4 smaller polish → 5 CSS polish (last).** 1 and 2 compound; 5 is
deliberately last so the structure is final before styling.

---

## 1. Catalog cross-reference (`--catalog`) — DONE

Shipped: `_internal/catalog_xref.py` (the shared reader + tolerance/match helper
+ pull tally), wired through impl/pipeline/api/cli at all three levels with
`--catalog PATH` / `--catalog-nsigma N` (default 3). Reader accepts
CSV/whitespace (header-keyword columns, uncertainty unit from the header) and
Pickett/SPCAT `.cat`. Proximity annotation only — opaque-label echo, never an
assignment, never alters the fit. Unit + cross-interface tests in
`tests/unit/stage6/test_catalog_xref.py` and the three report test modules.
Original plan retained below.

Optional `--catalog <file>` input on every report level. For each reported line,
flag the nearest catalog entry within a geometric tolerance and **echo its
opaque label** — proximity annotation only, **never an assignment** (the
pipeline emits UNASSIGNED lines; this is a cross-check, not a fit input).

- **Tolerance:** `|f_line − f_cat| ≤ N · sqrt(σ_f² + σ_cat²)` (default `N≈3`).
  `σ_f` is `FinalPeak.sigma_f_khz`; `σ_cat` from the catalog (0 if absent).
  Put the match test in **one shared helper** — the pull surface (item 2) reuses
  it.
- **Catalog input:** a format-light reader: CSV/whitespace (`frequency_mhz`,
  optional uncertainty with the unit read from the header — `*_mhz` / `*_khz` /
  `*_hz`, opaque label) **and** Pickett/SPCAT `.cat` fixed-width (the standard
  predicted-line catalog: frequency + MHz error + a species-tag/quantum-number
  opaque label). Pickett `.lin`/SPFIT *emission* (writing) stays out of scope.
- **Surfacing:** L1 — extra columns `catalog_label` / `catalog_freq_mhz` /
  `catalog_delta_khz` / `catalog_pull`. L2 — a match-rate line + a worst-pull
  table. L3 — a "catalog" column in the index final-line table and the
  per-window fitted-lines table; a match badge.
- **Acceptance:** dual-interface + cross-interface test; opaque-label echo never
  alters the fit; tolerance helper unit-tested; runs with and without `--catalog`.
- Calibration truth sets already exist for 1512 / 655 (frequency-within-
  uncertainty + amplitude-ratio vs catalog intensity); use them to sanity-check
  match rates.

## 2. Pull-calibration surface (Step 2) — DONE

Shipped with item 1: the catalog section (L2) and methods page (L3) report the
pull distribution (mean, sample std) with an honest / optimistic / conservative
verdict on σ_f and a systematic-offset note; L3 adds a σ_f pull histogram to the
distributions figure; L1 JSON carries `pull_mean` / `pull_std` in the catalog
metadata. Original plan retained below.

Promote the scratch validation harness's `ground_truth.pull` into a user-facing
surface. Pull `= (f_fit − f_cat) / σ_f` should be ≈ unit normal when σ_f is
honest — a **user calibration tool to validate the budget, not a shipped gate**
(see [[frequency-uncertainty-digitizer-clock]]: ship precision-only, σ_floor
user-owned default 0). Needs a catalog, so it sits on top of item 1 and shares
its tolerance/match helper.

- Report the pull distribution (mean, std, a histogram via
  `plot_summary_histograms`) and flag std ≫ 1 (σ_f optimistic) or ≪ 1
  (conservative). The harness logic in
  `dev-docs/research/stage5-cross-fixture/stage5_cross_fixture.py`
  (`ground_truth` / pull) is the reference to lift.

## 3. Survival-prune floor recheck (deferred Stage 5 fix)

`apply_snr_survival_prune` (`src/ftmwpipeline/_internal/stage5_impl.py`)
classifies sub-floor auto peaks as dust **once**, then refits a partially-pruned
window via `refit_window_core`; the refit can push a *surviving* peak below the
3.2 floor and nothing re-checks it (observed: 2638 window 4, survivor at SNR
3.169 after a neighbour at 3.19 is removed). Fix = **iterate to a fixpoint**
(re-classify after each refit), guarding against an over-pruning cascade.

- Changes Stage 5 output → needs the **7-fixture re-baseline**. Tooling and
  fresh keeper fixtures are ready: run
  `dev-docs/research/stage5-cross-fixture/stage5_cross_fixture.py`
  per-fixture in parallel (cap `OMP_NUM_THREADS`), then a `--reuse` rollup.
  Compare on-vs-off; this *will* shift the affected windows' lines (unlike the
  covariance/audit metadata fixes, which did not).
- See [[stage5-survival-prune-refit-recheck]]; flagged in ROADMAP.

## 4. Smaller polish follow-ons

- **Methods-page equations.** `$$…$$` blocks render as LaTeX *source* in
  `<pre class="equation">` (`_md_to_html`). To render math, add MathJax (or
  KaTeX) via a CDN `<script>` in `_page` / the methods head. Caveat: a CDN
  dependency is not offline-safe — gate it or vendor a minimal build if offline
  rendering matters.
- **Interleave histograms with their tables.** Today the methods page groups all
  histograms in a trailing "Distributions" section (one `plot_summary_histograms`
  figure). To place each histogram beside its percentile table, either render
  per-stat PNGs and inject them at each table in `_md_to_html`, or (cleaner)
  build the methods page from `_SummaryModel` directly instead of via md→html so
  tables and histograms are emitted together. The raw distributions are already
  on the model (`chi2r_values` / `eps_values` / `sigma_*_values` /
  `snr_values_promoted`).

## 5. CSS polish — LAST (after items 1–4)

Edit **`_STYLESHEET` only** (HTML structure is stable; the design lives in the
stylesheet by intent). Stable classes: `fit-panels` / `panel-grid` /
`panel-overview`, `cov-heatmap`, tables `variances` / `covariance` / `audit` /
`window-list` / `final-list` / `peak-list` / `ledger` / `decisions`,
`cov-legend` / `audit-legend`, `hist`, `pre.equation`, `summary`, `badge`, `nav`.

- **User request (explicit):** make better use of horizontal space with
  grid/flex layouts that **flow to a single column on smaller screens** (the
  same responsive pattern as the per-window `panel-grid`'s `@media (max-width:
  900px)` collapse). Candidates: index summary list + key numbers side by side;
  index window-list and final-line-list two-up on wide screens; per-window page
  pairing fitted-lines beside the covariance/variances; methods-page parameter
  tables in a responsive grid.
- Keep it dependency-free (hand-rolled CSS, no framework). Verify with the
  headless-Chrome screenshot workflow at wide and narrow widths
  (`google-chrome-stable --headless=new --window-size=W,H --screenshot=…`).

---

Demo site (regenerate after changes):
`ftmw.report_full(<file>, output_dir=…, windows='attention')`; a full
review-run 2638 fixture lives at `scratch/rebaseline-cov-audit/2638/exp_2638.ftmw`
(copy out, `review_run`, then `report_full`). See [[stage6-reports-step1]].
