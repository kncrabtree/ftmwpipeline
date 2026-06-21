# User documentation, code review, and repository cleanup

Umbrella plan for the combined effort that produces the ftmwpipeline user
documentation site, reviews the stage implementations that back each page, and
archives the planning material once its content has been folded into the docs.

This is a multi-session effort. Per-stage tracking documents live alongside this
README (`stageN-docs.md`), created as each stage is reached. This README holds
the process, the documentation-site map, and the cross-session progress
checklist.

## Deliverable

ReadTheDocs-style Sphinx/reStructuredText documentation under `docs/source/`,
written for spectroscopists. The audience wants to know how to use the software,
what algorithms it employs, and on what basis they can trust the results.
Coding-level implementation detail (efficiency, internal data structures, how
performance is achieved) belongs in developer documentation, not the user guide;
performance appears in the user guide only as the knobs a user controls.

## Style and conventions

The documentation mirrors Blackchirp's conventions
(`~/github/blackchirp/src/doc/AGENTS.md`):

- **Voice and tense.** Present-tense, impersonal ("the pipeline estimates the
  noise", never "we estimate" or "the pipeline will estimate").
- **No source-evolution markers.** No "Phase 2", "now uses", "previously",
  "recently added". Version-keyed information lives in the changelog, not the
  guide. Runtime-sequence language ("after the fit converges") is fine.
- **American English** throughout (`normalize`, `behavior`, `color`,
  `visualization`, `analyze`). Match UI/CLI labels exactly when quoting them.
- **Index block.** Every page opens with a `.. index::` block naming the
  user-facing terms it introduces.
- **Cross-references.** Sphinx `:doc:`/`:ref:` directives, not raw HTML anchors.
- **Theme.** `sphinx_rtd_theme`; a `docs/source/requirements.txt` (or equivalent
  doc-build dependency set) pins the build dependencies.

## Documentation-site map

Grouped into captioned `toctree` sections in `index.rst`, in roughly the order a
new user encounters them. (Page list is the target; individual pages are created
as their stage is reached.)

- **Getting Started** — `overview` (purpose, philosophy, the `.ftmw` file model,
  the three-interface rule, the stage pipeline at a glance) · `installation` ·
  `quickstart`.
- **Concepts** — `settings_and_presets` (salvage and refresh the existing page) ·
  `file_format` (the `.ftmw` HDF5 model, stage tracking, provenance).
- **Pipeline Stages** — one page each: `stage0_import` · `stage1_ft` ·
  `stage2_noise` · `stage2b_tau` · `stage3_peaks` · `stage4_windows` ·
  `stage5_fitting` · `stage6_review` (review / reports / finalization).
- **Advanced** — `clock_declaration` (instrument clock tree and timebase
  self-calibration) · `scope_record_import` (raw-scope-record loaders) ·
  `performance` (parallelism and the knobs that control it).
- **Methods & Validation** — self-contained, regenerable technical notes that
  justify algorithmic choices for the "why can I trust this" reader. Each note
  owns a subdirectory with its `generate.py` harness, `figures/`, and a
  `results.json` regression target; figures + numbers are regenerated from the
  checked-in example data, never hand-quoted, and a `slow` test guards the key
  results against drift. First note: `methods/noise_snr_scaling` (the
  leakage-pedestal failure of naive noise estimation at high signal-to-noise).
  This is where promoted `dev-docs/research/*` reports live in **timeless**
  form (no source-evolution framing; a naive running-median baseline replaces
  the "old vs new estimator" narrative).
- **Reference** — `cli` (CLI command reference) · `api/index` (autodoc Python
  API) · `changelog`.

The existing `docs/source/` scaffold is stale boilerplate (it references a
`process_experiment` entry point and placeholder modules that do not match the
shipped API); only `settings_and_presets.rst` reflects the current design. The
scaffold is rewritten, not extended.

## Per-stage process

Each stage page follows the same gated sequence. The code-revision discussion
gate is mandatory: documentation is not written against code that is about to
change.

1. **Read the planning record.** The stage's `dev-docs/planning/*.md` documents
   plus the ROADMAP / COMPLETED entries that reference it. Some are stale; note
   discrepancies against the code rather than trusting the prose.
2. **Review the code (thorough).** Read the `_internal/stageN_impl.py` plus the
   stage's supporting modules and the three interface wrappers. Surface code
   smells, inconsistent practices, TODO markers, and gaps between the stage's
   stated intent and its test coverage — and propose structural cleanups even
   when they are not strictly doc-blocking. **Stop and discuss any proposed code
   revisions with the user before writing documentation or changing code.**
3. **Mine the research reports.** The `dev-docs/research/*` reports justify the
   chosen approach and record alternatives considered. Extract what informs a
   technical reader; avoid excessive apologia.
4. **American-English scan.** Sweep the stage's user-facing output — CLI help
   strings, log and error messages, docstrings — for British spellings and
   correct them in code.
5. **Write the page.** Apply the style conventions above.

Code changes that result from step 2 are committed via the `commit` skill, and
project memory is updated whenever a code revision changes a documented behavior.

## Cross-cutting passes

- **Repository-level American-English sweep.** A repo-wide pass over all
  user-facing strings (beyond the per-stage scans), once stage pages are drafted.
- **Repository cleanup.** Stray run artifacts at the repo root
  (`*_enhanced_spectrum.png`, `output/`) are removed; the working tree is kept
  free of run output per the `scratch/` convention.
- **Archival.** Only after the user has reviewed the documentation: completed
  planning documents are archived (their content now living in the docs), and
  research reports are updated where still useful or removed where obsolete.

## Findings to resolve

Code-review observations surfaced while writing the docs, held for discussion
before any code change (per the per-stage gate).

- **Stale `README.md` status section.** The root `README.md` states Stages 3–5
  are "not yet implemented"; all stages ship. Update during the repository
  cleanup pass.

## Resolved during review

Code changes made while reviewing the docs, with user sign-off:

- **Stage 4 review: difficulty/split annotations removed, dead stub package
  deleted, the window plot recolored, British spellings swept.** The Stage 4
  `WindowDifficulty` (HARD/EASY) label, the `FitWindow.split_proposal`
  frequency, and the `needs_joint_treatment` flag were *computed but consumed by
  no downstream stage* — Stage 5's structural-replan handshake is driven by its
  own fit-time edge-coherence check emitting `MergeRequest`s, never by the
  Stage 4 hints, and nothing branches on the difficulty label (it reached only
  the `windows run` summary, the `windows show` shading, the `scan` tuning
  plots, and one line of the Stage 6 report). With sign-off all three were
  removed end to end: the data structures (`WindowDifficulty` enum + the three
  `FitWindow` fields + both `__repr__`s + the `core`/package exports), the
  window-planning step 6 (which now only records the per-window
  `edge_coherence_statistic`/`edge_coherence_fail` diagnostic — the difficulty
  classification, the content-vs-cap `too_wide` test, and the
  `split_proposal`/`needs_joint_treatment` assignment are gone, and the
  now-dead `edge_m`/`max_window_width_mhz` parameters were dropped from
  `_finalize_plan`), the HDF5 serialization (the `difficulty` required attr +
  its `_VALID_DIFFICULTY` validation, the `split_proposal` NaN column, and the
  `needs_joint_treatment` attr), the `stage4_impl`/`pipeline`/`api`/CLI
  summaries and docstrings (the `n_hard`/`n_easy` result keys and the
  hard/split/joint print lines), the `scan` surface (the `n_hard`/`n_easy`/
  `n_split` metric columns + the two Stage 4 tuning plots' HARD hatching/split
  dotting/legend), and the Stage 6 report (`n_hard_windows` + the "N flagged
  hard (joint treatment)" line). The dead `window_assignment` placeholder
  package (`assign_analysis_windows` / `optimize_window_boundaries` /
  `resolve_overlaps`, three `NotImplementedError` "Phase 5" stubs — the live
  stage is `preprocessing/window_planning.py`) was deleted along with the
  commented-out re-exports in the package `__init__` and the
  `test_package_imports` case. `window_visualization.plot_window_plan` was
  recolored to the UC Davis brand palette (uniform `AGGIE_GOLD` window spans now
  that difficulty is gone, `AGGIE_BLUE` free peaks, `POPPY` fixed-contributor
  squares, `PINOT` S_coh trace, `DOUBLE_DECKER` `T_edge` line) so the Stage 4
  doc figure will match the committed Stage 0/1/2/2b/3 set, and the
  `split_proposal` marker was dropped. Two stale docstrings were corrected (the
  `edge_coherence.above_threshold_intervals` "default 3.0" → 8.0; the
  `assign_windows_impl` "raises if settings= and preset= are both supplied"
  clause, left over from before composition was allowed) and British spellings
  swept from the Stage 4 modules. The `plan_revision` round-trip gap was left
  as-is (the field is never persisted with a non-zero value) and its docstring
  trimmed to say so. Validation: the Stage 4 unit/integration/serialization,
  the fitting plan-execution/result-conversion, the tuning-plot, the
  Stage 6 report, the scan cross-interface, and the package-import suites pass
  (290 tests across the touched suites); the touched files are
  black/isort/mypy-clean.
- **Stage 3 review: dead stub package removed, stale wrapper docstrings fixed,
  detection plot recolored, British spellings swept.** The whole
  `peak_detection` package (`basic_detection` / `hybrid_detection` /
  `classification`) was three never-wired `NotImplementedError` "Phase 4" stubs
  — the live detector is `preprocessing.peak_detection` — so it was deleted and
  the two references repointed: `workflows.validate_installation` now imports
  `core`/`preprocessing` only, and `test_package_imports` drops the stub case
  (the commented-out re-exports in the package `__init__` went too). Two thin
  wrapper docstrings (`Pipeline.detect_peaks`, the CLI `cmd_detect_peaks`) still
  described the retired design — an "unapodized gap pass masked within each
  strong line's analytic truncation-leakage reach" — and were corrected to the
  shipped matched-filter gap pass with the continuous leakage-aware floor scored
  on the canonical active FT (the `_internal` impl and kernel were already
  right; only the wrappers had drifted). `peak_visualization.plot_peak_detection`
  was recolored to the UC Davis brand palette (WEAK/MEDIUM/STRONG → QUAD / POPPY
  / DOUBLE_DECKER, the noise line and SNR histogram → AGGIE_BLUE, the promotion
  cutoff → DOUBLE_DECKER), so the new Stage 3 doc figure matches the committed
  Stage 0/1/2/2b set; the earlier "defer recoloring to Stage 6" note is
  superseded for this plot. The detection overlay now encodes promotion in the
  marker fill — promoted peaks filled, below-cutoff candidates open — so the
  unreachable `visualize_peaks_impl(promoted_only=)` switch (wired to no
  interface) was removed in favor of the always-on visual split. British
  spellings were swept from the Stage 3 modules. Validation: the Stage 3 unit +
  integration suites and the changed-file black/isort/mypy are green; the figure
  smoke test passes.
- **Stage 2b Gaussian/Lorentzian twin code paths fully unified.** The Gaussian
  τ_G calibration had been bolted on as a near-duplicate of the exponential
  twin and promoted to a first-class shape; the parallel code paths had bitten
  repeatedly. The duplication was collapsed into one shape-parameterized path:
  the engine extractors share a ``_finalize_tau_result`` core; the Gaussian
  impl module was deleted and folded into a single
  ``stage2b_impl.calibrate_tau_impl(shape=)`` (with ``save`` / ``load`` /
  ``present`` taking ``shape=`` and a normalized result key); and the public API
  reduced to ``calibrate_tau(shape=)`` / ``load_tau_calibration(shape=)`` on the
  functional API and ``Pipeline`` (the ``_G`` variants removed). The CLI keeps
  ``tau run --gaussian``, gains a ``tau recommend`` verb and ``--gaussian`` on
  ``tau show``. Three latent issues the dual path had masked were fixed in the
  same pass: the ``polish_top_n`` knob — wired through settings, scan, and
  provenance but never consumed — was removed; the ``--min-contributors``
  aggregation-to-Gaussian routing moved into the impl so every interface treats
  it identically; and the ``recommended_shape`` reset was made symmetric across
  shapes but gated on whether the call is a primary run, so the auto-built twin
  does not wipe the vote it acts on. The tau visualizations were restyled to the
  brand palette and routed by shape; British spellings and source-evolution
  markers were swept from the tau modules. Validation: a fresh-fixture golden of
  every persisted τ field is byte-identical before and after; the Stage 2b/3/5/6
  and cross-interface suites pass; the touched files are black/isort/mypy clean.
  New tests cover the precondition-failure outcome, the per-band majority
  helpers, and the min-contributors routing.
- **Complex-domain σ cross-check added to Stage 2 (guardrail, magnitude stays
  primary).** A reader question — is ``line_k=8`` too lax, leaving line skirts in
  the noise set and inflating σ? — opened a noise-estimator investigation. The
  self-mask clip is one-sided (``keep = resid < line_k·σ``), so an aggressive
  ``line_k`` (3–4) falsely excludes the upper tail of genuine noise (~0.8% of
  pure-noise bins at k=3, biasing σ ~3.5% low) while the broad lower-envelope
  median — not the mask — is the real defense against line contamination; ``8``
  is correctly past the knee, **unchanged**. The deeper finding: the magnitude
  estimator carries a small intrinsic *low* bias (it must undo the Rayleigh
  pedestal + Rician ``C(R)``), where an independent estimate from the symmetric,
  correction-free real/imaginary scatter does not. Shipped that as a **read-only
  guardrail**: :func:`estimate_noise_complex_scatter` plus a ``mag/complex`` σ
  ratio + divergence warning folded into ``bin_info`` and overlaid on
  ``noise show``. A **hybrid** (measure σ in the complex domain) was prototyped
  behind a ``sigma_source`` flag and **rejected**: on dense, high-SNR spectra the
  complex estimate is leakage-floor-limited (655 noise slope −0.375 vs the
  magnitude's −0.448, against the stationary 1/√N = −0.5), because oscillatory
  line leakage sits in the noise bins and a running median can't remove it (it
  removes the smooth magnitude pedestal, not Re/Im oscillation). The fixtures
  also show real ~5–9% signal drift across the acquisition (sign varies), so no
  clean stationary line-free truth exists in this data and the 1/√N slope was the
  drift-robust arbiter. **Decision (user):** keep the magnitude estimator
  primary, complex as the cross-check; the flag was reverted. Reproducible
  analysis in the gitignored ``scratch/line-k/``.
- **Plotly removed; matplotlib is the sole visualization backend.** The second
  rendering backend was ripped out across the stack: the ``_plot_*_plotly``
  paths and the plotly statistics-table helper in ``spectrum_visualization`` /
  ``noise_visualization``; the ``backend`` parameter and its pass-through on
  every visualization surface (``stage1``–``stage5`` impls, ``report_html_impl``
  callers, the five ``api`` / ``Pipeline`` ``visualize_*`` methods, and the
  ``ft`` / ``noise`` / ``peaks`` / ``windows`` CLI commands, including the
  ``noise show --backend`` option and the ``.html`` save path); the
  ``has_plotly`` package-info flag and ``version`` print; the ``pytest_plotly``
  marker; and the dependency declarations (the dead ``[viz]`` extra —
  plotly + the equally unused bokeh/seaborn — plus the conda env entries and the
  ``plotly.*`` mypy override). Docstrings that promised "matplotlib or plotly"
  now say matplotlib. ``installation.rst`` drops the ``[viz]`` extra.
- **Dead not-implemented placeholders removed.** The ``fit_diagnostics`` and
  ``summary_reports`` modules were whole-module ``NotImplementedError`` "Phase 8"
  stubs (never wired up); the ``plot_peaks`` / ``plot_windows`` placeholders in
  ``spectrum_visualization`` shadowed the real ``peak_visualization`` /
  ``window_visualization`` implementations; and ``visualization/__init__``
  exported a non-existent ``plot_spectrum``. All removed, and the package
  ``__init__`` trimmed to the spectrum + noise re-exports that actually exist.
- **Earlier-stage visualizations restyled to the house style.** Stage 0
  (``start_detection_visualization``), Stage 1 (``spectrum_visualization`` —
  real → Double Decker, imag → Gunrock, magnitude → Cabernet), and Stage 2
  (``noise_visualization``) now use the brand palette, ``apply_bare_style``, and
  the suppressible-title convention (``title=""`` for doc figures). The FT figure
  uses constrained layout to avoid the spanning-axes ``tight_layout`` warning.
- **Committed stage figures + regenerable harness.** ``docs/source/figures/``
  holds one figure per documented early stage, rebuilt by
  ``docs/source/figures/generate.py`` (a 2638 pipeline in a temp dir, render
  Stage 0/1/2), embedded with ``.. figure::`` + captions in the stage pages, and
  guarded by the ``slow`` smoke test ``tests/integration/test_stage_figures.py``.
- **No-code data-input path built (custom loaders + clock declarations).** The
  generic `csv` and `hdf5` loaders were non-functional stubs (`load_fid` raised
  "not yet implemented"), so there was no way to bring your own data without
  writing a `BaseLoader` subclass, and clock declarations
  (`ClockSource`: `freq_mhz`/`locked`/`label`) could only be injected by a
  loader or the ephemeral `run --clocks`. Designed (`data-input-format-spec.md`,
  signed off) and built: a native **`ftmw-hdf5`** input format (self-describing,
  embeds clocks), a real **`csv`** loader (with a `column` selector), a JSON/YAML
  **`--metadata` sidecar**, and a **`clocks`** declaration surface
  (CLI `show`/`set`/`add`/`remove`/`clear` + `api`/`Pipeline` methods) writing
  the recommended-clock-sources layer. The dead `hdf5` stub was removed
  (`ftmw-hdf5` replaces it). Shared resolver `io/input_metadata.py` enforces
  `explicit > sidecar > embedded > default`. Unit + cross-interface tests added.
  The user-facing Sphinx pages remain to be written (Stage 0 + a dedicated
  input-format reference page).
- **Rician C(R) correction table monotonized (correct physics).** The baked
  `_SCATTER_R_TAB` / `_SCATTER_C_TAB` in `preprocessing/noise_estimation.py` were
  a raw M=1e5 Monte-Carlo cloud, jagged at the Rayleigh end. The jaggedness is
  not sample noise — near the Rayleigh limit `R = scatter/pedestal` saturates
  (~0.565), so it is a poor lookup coordinate there and sorting-by-R folds
  smooth-in-θ points into apparent noise. The true `C(R)` is smooth and monotone
  increasing, so the table is now the **isotonic (monotone) fit** of a high-M
  (1e6) simulation on a clean ascending R grid (C: 1.000→1.499), produced by the
  single canonical `generate.build_cr_table` and pinned by the unit test. Effect
  on σ is <0.4% (median 0.09%), so the full-suite re-baseline was **clean —
  1893 passed, 2 skipped, zero value-specific edits**. The `slow` regen results
  shifted within tolerance (655 overestimate 5.87→5.88×).
- **Stage 2 noise diagnostics de-staled (retired adaptive-estimator leftovers).**
  The `noise run` summary and the `noise show` plot read `bin_info` keys
  (`n_bins`, `smoothing_window` / `smoothing_window_mhz`, `bin_edges`) that only
  the retired *adaptive* estimator emitted, so they always printed "unknown" and
  the bin-boundary overlay could never draw. Fixed the diagnostics to read the
  scatter estimator's real keys (`n_region_windows`, `n_line_bins`,
  `smoothing_mhz`, `region_aware`, `noise_fraction`); removed the dead
  `--show-bin-boundaries` flag and `bin_edges` drawing across the CLI, impl,
  `api`/`Pipeline`, and `noise_visualization` (the plotly dead "Adaptive Bins"
  panel now shows the σ/3σ/5σ levels); refreshed the stale `noise show` help
  ("adaptive binning"); and moved `stage2_impl`'s bottom-of-file `json`/
  `datetime` imports to the top (dropping an unused `open_pipeline_file`). σ
  values and persistence are untouched.
- **``rdc`` knob removed; DC removal is unconditional.** The canonical FT
  always subtracts the active-region mean before transforming (as the Stage 1
  page documents), so the ``rdc`` toggle carried no information. Removed from
  ``FIDProcessingParameters`` / ``FID.preprocess`` (DC removal now
  unconditional), ``FTSettings`` (field + ``_HARD_DEFAULTS`` + ``to_attrs`` /
  ``from_attrs``, which now ignores a legacy ``rdc`` attr like the retired
  apodization keys), the input-metadata resolver + sidecar key set, the ``csv``
  / ``ftmw-hdf5`` loaders, the Blackchirp loaders (the instrument ``FidRemoveDC``
  cell is no longer carried through), `fid_serialization`, the `data import
  --rdc/--no-rdc` flag, and the FID visualization panels. The internal
  diagnostic utilities (`utils/signal_processing`, `fitting/active_ft`) keep
  their ``rdc`` parameter (default True), parallel to the retained apodization
  knobs there. The 2638 fixture has ``FidRemoveDC;true``, so the canonical FT is
  byte-identical and no fixture re-baseline was needed; `input_formats.rst`
  drops the ``rdc`` rows. Tests updated across settings/loader/serialization/
  three-stage suites.
- **``probe_freq_mhz`` made optional (default 0 MHz).** The generic-loader
  metadata resolver required both ``spacing_us`` and ``probe_freq_mhz``; now only
  ``spacing_us`` is required and ``probe_freq_mhz`` defaults to ``0`` — a
  direct-sampling instrument whose baseband *is* the molecular frequency (with
  the default ``upper`` sideband, molecular = baseband). Changed in
  `io/input_metadata.py` (`_DEFAULTS` + the required-field check), with the
  `csv` / `ftmw-hdf5` loader docstrings, the `data import --probe_freq_mhz` help,
  and `input_formats.rst` updated to match; resolver tests cover the new default.
- **``--trim`` form unified on the colon form.** The whole-pipeline ``run``
  command parsed ``--trim LO HI`` (two space-separated floats) while the
  per-stage ``ft run`` parsed ``--trim MIN:MAX`` (colon-delimited). ``run`` now
  uses the colon form too (the shared `_parse_trim` from `core.settings`),
  matching `ft run`, the persisted/`settings` convention, and the README, so a
  single grammar holds across every surface. Docs example in `quickstart.rst`
  updated; the run-pipeline tests call `run_pipeline_impl` directly so no CLI
  test changed.
- **Stale `ft show` help text removed.** `ft show` (and its `--help` epilog)
  advertised saving "complete parameter sets (preprocessing + postprocessing) as
  defaults" and prompting "(y/N)"; the command never persists or prompts (it is
  visualization-only, and there is no longer any postprocessing). The docstring
  and epilog now state plainly that visualization never persists and point to
  `ft run` for storing settings.
- **Blackchirp naming corrected repo-wide.** The companion program is
  "Blackchirp", not "BlackChirp". A guarded replace (negative lookahead on the
  `BlackChirpLoader` class identifier) fixed all prose occurrences across
  `docs/source`, user-facing CLI help/docstrings, `README.md`, `STATUS.md`,
  `CLAUDE.md`, and the user-docs planning notes; the `blackchirp` format key and
  `BlackChirpLoader` class name are unchanged.
- **`report run --output-dir` made optional.** It was a holdover from the
  multi-page report site; the report is now two files, so an omitted
  `--output-dir` writes them to the current directory (no auto-created
  directory). Applied across impl + `Pipeline`/`api` + CLI, with a test.
- **`settings=` and `preset=` now compose.** The two were mutually exclusive on
  every interface; they populate different resolution layers (explicit vs
  preset), so combining them is well-defined and enables "preset recipe + a few
  explicit overrides." The `ValueError` guards were removed from the five stage
  impls and the five CLI commands; the mutual-exclusion tests became
  composition tests. The resolution order is unchanged: explicit > persisted >
  preset > recommended > default, so a preset still never overrides a persisted
  value (the D11 reproducibility guarantee). To keep the old "mutually
  exclusive" claim from creeping back, the change was reconciled across the
  records that survive this commit: ROADMAP D11 (amended), `SERIALIZATION_STRATEGY.md`
  (states the composition rule), and the planning docs `stage5-fit-settings.md`,
  `settings-backfill.md`, and `settings-surface-finalization.md`.

## Sphinx build

Stood up before content so reStructuredText and toctree errors surface
immediately: add the doc-build dependencies (`sphinx`, `sphinx_rtd_theme`, and
the autodoc/napoleon extensions already named in `conf.py`), repair the stale
`conf.py`/`index.rst`, and confirm a clean `make html` on the empty-but-valid
site. The build is exercised continuously as pages are added.

Always build into the default `docs/build/html` (e.g.
`sphinx-build -W -q -b html source build/html`, or `make html`, from `docs/`),
never a throwaway tmp location. `docs/build` is gitignored, so it does not
pollute the tracked tree, and it is where the user opens the rendered HTML to
review each stage page.

## Progress

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done.

- [x] Scope and register the effort (this plan).
- [x] Stand up the Sphinx build (deps, theme, repaired `conf.py`/`index.rst`,
  green `make html`). The `docs` extra is trimmed to `sphinx` +
  `sphinx-rtd-theme` (the only dependencies `conf.py` loads) and is the single
  source of truth for doc tooling; the runtime `environment.yml` stays minimal
  and the dev env installs it via `-e .[dev,docs]`. `docs/Makefile`,
  `docs/make.bat`, and a `.readthedocs.yaml` (`fail_on_warning: true`) are
  added; `index.rst` carries the full captioned-toctree site map; every page
  exists as a stub (with a real `.. index::` block and one-line scope) except
  `settings_and_presets.rst`, which is migrated intact. The build is
  warning-clean under `sphinx-build -W`.
- [x] Getting Started: `overview`, `installation`, `quickstart`. Written
  against the verified API/CLI surface (`README.md`, `API_STRATEGY.md`,
  `CLI_STRATEGY.md`, and the actual `api.py`/`pipeline.py`/`cli` source);
  documented forms smoke-checked against ``--help``.
- [x] Concepts: `settings_and_presets` refreshed during review (precedence
  order corrected to `explicit > persisted > preset > recommended > default`,
  the `settings=`/`preset=` composition documented, the stale per-knob-kwarg
  examples replaced); `file_format` written (the `.ftmw` HDF5 model, stage
  tracking + dependency graph, persisted-vs-recomputed contract, provenance +
  safe re-import, the `PipelineFileError` family), against the verified
  `file_manager.py` / `SERIALIZATION_STRATEGY.md` surface.
- [x] Stage 0 — import. `stage0_import.rst` written (import workflow, format
  detection, provenance + safe re-import, start-time detection), plus a new
  Concepts page `input_formats.rst` — the reference for the no-code input path
  (native `ftmw-hdf5` and `csv` formats, the `--metadata` sidecar, clock
  declarations, and writing a custom loader). The generic loaders, sidecar, and
  `clocks` surface they document were built first (see Resolved during review).
  Build is warning-clean under `sphinx-build -W`.
- [x] Stage 1 — FT. `stage1_ft.rst` written (the canonical unapodized /
  native-length transform and why, the active-region + DC-removal selection
  steps, the molecular-frequency / sideband mapping, the four parameters, the
  canonical-settings binding + downstream invalidation, and `ft show`). Code
  revisions resolved during review (see below): the `--trim` form unified on the
  colon form, the stale `ft show` persistence claims removed, and the Blackchirp
  naming corrected repo-wide. Build is warning-clean under `sphinx-build -W`.
- [x] Stage 2 — noise. `stage2_noise.rst` written (where σ is measured — the
  canonical active FT — and the complex-RMS convention; why a level estimator
  fails on high-SNR line-dense spectra via the leakage pedestal; the scatter
  estimator's four steps; the eight knobs; and reading `noise show`). Code
  revisions resolved during review (see below): the stale adaptive-estimator
  `bin_info` keys fixed, the dead `--show-bin-boundaries` overlay removed, the
  `noise show` help refreshed, and the bottom-of-file imports in `stage2_impl`
  cleaned up. Build warning-clean under `sphinx-build -W`.
- [x] Methods & Validation section stood up + first note. New captioned toctree
  section; `methods/noise_snr_scaling.rst` is the noise-estimation justification
  rewritten **timeless** (the leakage pedestal, the 1/√N discriminator, the
  cross-fixture overestimate, the weak-line floor, the Rician `C(R)` correction)
  with no source-evolution framing — the contrast is against a naive
  running-median baseline, not a "retired estimator." Self-contained +
  regenerable: `methods/noise_snr_scaling/generate.py` rebuilds `figures/*.png`
  and `results.json` from the checked-in fixtures (~15 s); two fast data-free
  unit invariants (the `C(R)` table matches the baked constants;
  synthetic 1/√N slope split) in
  `tests/unit/preprocessing/test_noise_snr_invariants.py`, and a `slow` regen
  guard `tests/integration/test_noise_snr_report.py` re-runs the harness and
  compares `results.json` within tolerance. `stage2_noise.rst` now links the
  note via `:doc:` instead of the `dev-docs/research` path. The
  `dev-docs/research/noise-snr-scaling/` original stays as internal provenance
  until the archival pass (it carries the retired-method history); the
  user-facing docs no longer depend on it.
- [x] Figure-style conventions + brand palette. `visualization/report_style.py`
  (the shared house style) gained the **UC Davis brand palette**: `AGGIE_BLUE` /
  `AGGIE_GOLD`, the named secondary colors (`DOUBLE_DECKER`, `GUNROCK`, `QUAD`,
  `POPPY`, `PINOT`, `ARBORETUM`, `REDBUD`, `MERLOT`, `REDWOOD`, `CABERNET`,
  `TAHOE`, `SUNFLOWER`), a contrast-ordered `BRAND_CYCLE`, `apply_color_cycle`,
  and `aggie_blue_cmap` / `aggie_gold_cmap` single-hue gradients (divergent /
  perceptually-uniform data still uses a purpose-built map). Convention: methods
  figures carry **no title** (the caption labels them) and use the bare
  spine-free style. The noise note's figures were restyled accordingly (data as
  scatter + the power-law fit as a line on the two scaling plots; enlarged MC
  cloud on the C(R) plot). Recoloring the existing pipeline report/CLI plots to
  the palette (e.g. real/imag/magnitude → Gunrock / Double Decker / Cabernet) is
  deferred to the Stage 6 report pages.
- [x] Stage 2b — tau calibration. `stage2b_tau.rst` written (the
  sliding-window STFT and why it is fit-free, per-bin classification, the
  signal-to-noise-weighted majority + polish + per-band majorities, the two
  shape variants and the 3-way line-shape vote, the knobs, reading the
  diagnostics, the acceptance pre-conditions, the horn-coupling frequency
  trend, and what Stages 3/5 consume). Three regenerable figures were added to
  `docs/source/figures/generate.py` (a lean Lorentzian-only calibration, no
  auto-recommend) and guarded by the `slow` smoke test: the decay-time
  distribution panel (with legible per-band boundary/level/label overlay on the
  τ-vs-frequency scatter, and both τ scatters capped at 1.5·T_active with
  off-screen bins flagged as open triangles); a per-bin decay-example figure (a
  strong line with the exponential and Gaussian fits, a saturated clock spur,
  and a noise bin); and a frequency-windowed, color-clipped heatmap zoom. The
  visualization gained `plot_stft_decay_examples`, a `freq_window` /
  `clip_percentiles` heatmap, and a `tau_cap_factor` on the distribution. The
  Gaussian/Lorentzian twin code paths were fully unified first (see Resolved
  during review). Build warning-clean under `sphinx-build -W`.
- [x] Stage 3 — peak detection. `stage3_peaks.rst` written (the two-pass design
  — Blackman-Harris primary for robust strong-line positions plus the shape-aware
  matched-filter gap pass for weak-line recovery; the concave-down Savitzky-Golay
  locator and why it pairs with the matched filter; the continuous leakage-aware
  detection floor that replaces a hard mask; scoring/snap-back onto the canonical
  active FT with sub-bin apex correction; SNR classification and the
  detection-vs-promotion split with store-all; the hand-edit Stage 3 → Stage 4
  boundary; the knobs). A regenerable `stage3_peaks.png` (the detection overlay +
  SNR-distribution curation panel) was added to
  `docs/source/figures/generate.py` (now builds through Stage 3) and guarded by
  the `slow` smoke test. Code revisions resolved during review (see below): the
  dead Phase-4 placeholder `peak_detection` package removed, two stale wrapper
  docstrings corrected, the peak-detection plot recolored to the brand palette,
  and British spellings swept. Build warning-clean under `sphinx-build -W`.
- [ ] Stage 4 — window assignment.
- [ ] Stage 5 — fitting.
- [ ] Stage 6 — review / reports / finalization.
- [ ] Advanced: `clock_declaration`, `scope_record_import`, `performance`.
- [ ] Reference: `cli`, `api/index`, `changelog`.
- [ ] Repo-wide American-English sweep.
- [ ] Repository cleanup (stray root artifacts).
- [ ] User review of the documentation.
- [ ] Archive completed planning docs; update/remove obsolete research reports.

## Handoff: next session is Stage 4 — window assignment

**State going in.** The Getting Started, Concepts, Stage 0/1/2/2b/3 pages, the
Methods & Validation section + first note, the brand style system, and the
committed early-stage figures (now including `stage3_peaks.png`) are done and
committed. The Stage 3 peak-detection page is written and its review-pass code
revisions landed first (see *Resolved during review*). The working tree is clean;
the Stage 3 suites (180 tests) and the figure smoke test pass, and `sphinx-build
-W` is warning-clean. Nothing is mid-flight.

**Last stage's trail (so nothing is re-litigated).** Stage 3's two passes are the
Blackman-Harris primary (robust strong-line positions) and the shape-aware
matched-filter gap pass (weak-line recovery, τ/shape from Stage 2b); both share
the concave-down Savitzky-Golay locator and a *continuous* leakage-aware floor
(`k·S_coh/√M·σ`, primary k=1 / gap k=3) — there is no hard leakage mask. Every
peak is scored on the canonical active FT (apex-snapped, de-duplicated); detection
runs at a fixed internal floor 2.0 while the user-facing `min_snr` (3.0) is the
*promotion* cutoff, and all detected peaks are stored. The dead `peak_detection`
stub package was removed; `Pipeline.detect_peaks` / CLI `cmd_detect_peaks`
docstrings were de-staled; `plot_peak_detection` is on the brand palette and now
encodes promotion in the marker fill (the unreachable `promoted_only` switch was
removed). Two follow-ups are logged under *Findings to resolve* for the Stage 4
review: window difficulty (HARD/EASY) is computed but unused downstream, and
`window_assignment` is the next dead Phase-5 stub package to remove (both kept
this session to stay scoped).

**The Stage 4 task — apply the per-stage process (this README, "Per-stage
process"), in order:**
1. *Read the planning record.* `dev-docs/planning/stage4-window-assignment.md`
   and `stage4-leakage-contributor-subtraction.md`, plus the ROADMAP/STATUS
   entries and the D8 leakage-touched-map work; note stale prose against the code.
2. *Review the code (thorough).* `_internal/stage4_impl.py`, the window-planning
   engine (`preprocessing/window_planning.py` and the leakage/edge-coherence
   modules it consumes), the serialization, and the three interface wrappers.
   Surface code smells, dead/`Phase`-era stubs (remove as encountered — standing
   approval; the `window_assignment` placeholder package is the known one), and
   test-coverage gaps. **Stop and discuss any proposed code revision with the
   user before writing docs or changing code.**
3. *Mine the research reports.* `dev-docs/research/complex-edge-coherence`,
   `stage4-*`, and the leakage-detection rework for the justification; extract
   what informs a technical reader.
4. *American-English scan.* Sweep the stage's CLI help / log / error strings /
   docstrings.
5. *Write `stage4_windows.rst`* (currently a stub) per the style conventions; add
   a Stage 4 figure by extending `docs/source/figures/generate.py` (it now builds
   through Stage 3) and guard it with the `slow` smoke test.

**Conventions.** Build docs into `docs/build/html` (gitignored) so the user can
review the rendered HTML; direct all run artifacts to `scratch/`; the noise
methods harness and `docs/source/figures/generate.py` are the reference patterns
for committed, regenerable figures. Run project commands via
`conda run -n ftmwpipeline-dev`; scope tests to the stage (per the test-budget
memory) and save one full run for the end.
