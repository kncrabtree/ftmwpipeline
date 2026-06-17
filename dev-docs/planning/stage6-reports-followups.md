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

## 3. Survival-prune floor recheck (deferred Stage 5 fix) — DONE

`apply_snr_survival_prune` now iterates to a fixpoint: it removes the single
lowest-SNR sub-floor line, refits the survivors, and **re-classifies** until no
survivor is sub-floor. This closes the single-pass gap (a refit could push a
*surviving* peak below the 3.2 floor with nothing re-checking it) and the
one-line-per-pass removal is the over-pruning-cascade guard (a borderline
neighbour whose SNR was depressed by the dust recovers on the refit instead of
being swept out). New unit tests pin both behaviours.

7-fixture re-baseline (off-vs-on, `scratch/item3-rebaseline/`): window counts
identical on all 7; only the dense fixtures shed sub-floor survivors (363 −3,
655 −6, every removed peak < 3.2σ), recall unchanged on both ground-truth
fixtures (1512 0.417, 655 0.512), and tail χ² slightly improves (655 p95
7.22→6.97). Accepted.

- See [[stage5-survival-prune-refit-recheck]].

## 4. Smaller polish follow-ons — DONE

- **Methods-page equations.** Equations now render via MathJax v3 (tex-svg, CDN,
  `async`) loaded only on the methods page (`_MATHJAX_HEAD` via `_page`'s
  `head_extra`); `_md_to_html` emits `\[…\]` display math. Offline-safe: the raw
  TeX stays visible in the styled `.equation` block when the CDN is unreachable.
- **Interleave histograms with their tables.** The methods page renders one
  figure per distribution group (`_summary_distribution_groups`: SNR /
  fit-quality / σ_f budget / catalog pull) and `_inject_after_table` places each
  immediately after the percentile table it summarizes; any that cannot be
  placed fall back to a trailing Distributions section.

## 5. CSS polish — DONE (`_STYLESHEET` only)

Responsive horizontal-space pass, edits confined to `_STYLESHEET`:

- `ul.summary` is a responsive `auto-fit` grid (multi-column on wide screens,
  single column when narrow) — the index + per-window key-value lists.
- Long data tables (`window-list` / `final-list` / `peak-list` / `audit` /
  `ledger` / `covariance`) fill the column width; zebra rows + row hover +
  sticky `thead th` keep hundred-row tables readable.
- `@media (max-width: 700px)` tightens gutters and table type; the per-window
  `panel-grid` keeps its existing 900px collapse.

Verified with the headless-Chrome screenshot workflow at 1400 px and 480 px
(`scratch/catalog-demo/shots/`): the index, methods page (MathJax-rendered
equations + interleaved histograms), and per-window page all reflow to a single
column when narrow. Dependency-free apart from the optional MathJax CDN.

---

Demo site (regenerate after changes):
`ftmw.report_full(<file>, output_dir=…, windows='attention')`; a full
review-run 2638 fixture lives at `scratch/rebaseline-cov-audit/2638/exp_2638.ftmw`
(copy out, `review_run`, then `report_full`). See [[stage6-reports-step1]].
