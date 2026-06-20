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
- [ ] Stage 2b — tau calibration.
- [ ] Stage 3 — peak detection.
- [ ] Stage 4 — window assignment.
- [ ] Stage 5 — fitting.
- [ ] Stage 6 — review / reports / finalization.
- [ ] Advanced: `clock_declaration`, `scope_record_import`, `performance`.
- [ ] Reference: `cli`, `api/index`, `changelog`.
- [ ] Repo-wide American-English sweep.
- [ ] Repository cleanup (stray root artifacts).
- [ ] User review of the documentation.
- [ ] Archive completed planning docs; update/remove obsolete research reports.

## Next session: viz restyle + stage figures + remove plotly

Stages 0–2 pages, the Methods & Validation section + first note, and the brand
style system are done. The next session has three linked tasks:

1. **Restyle the earlier-stage visualizations to the house style.** Bring the
   documented-stage plots into line with `visualization/report_style.py`: the
   brand palette (`AGGIE_BLUE` / `AGGIE_GOLD`, the named secondary colors,
   `BRAND_CYCLE`, `apply_color_cycle`, `aggie_blue_cmap` / `aggie_gold_cmap`),
   `apply_bare_style`, and the suppressible-title convention (`resolve_title`;
   pass `title=""` for doc figures, captions label them). The modules for the
   documented stages: `start_detection_visualization.py` (Stage 0),
   `spectrum_visualization.py` (Stage 1; the `ft show` real/imag/magnitude
   plot — the user's preferred mapping is real → Double Decker, imag → Gunrock,
   magnitude → Cabernet), and `noise_visualization.py` (Stage 2; already mostly
   styled — verify). Later-stage viz (tau, peaks, windows, fit) gets restyled as
   those stages are documented. **A wrong-start was reverted:** an unused
   `report_style` import was added to `start_detection_visualization.py` and
   rolled back — start fresh there.

2. **Remove plotly entirely** — code, docs, and dependencies. Plotly is a second
   rendering backend across the visualization modules (`spectrum_visualization`,
   `noise_visualization`, and others carry `_plot_*_plotly` paths + a `--backend
   plotly` CLI option and `backend=` kwargs). Rip it out: drop the plotly
   branches and the `backend` plumbing (matplotlib becomes the sole backend),
   remove plotly from the dependency lists (`pyproject.toml` /
   `environment*.yml`), and purge plotly mentions from docstrings, CLI help, and
   the docs (e.g. the `noise show --backend` text and the `stage2_noise.rst` /
   `noise_snr_scaling.rst` "plotly backend" lines). Run the suite after.

3. **Add figures to the stage pages.** With the matplotlib paths restyled and
   plotly gone, add a committed figure harness (mirroring
   `methods/noise_snr_scaling/generate.py`): build a 2638 pipeline in a temp dir
   (import → start → ft → noise) and render one representative figure per
   documented stage into a committed `docs/source/figures/` dir — Stage 0 the
   start-detection sweep, Stage 1 the canonical FT (magnitude + real/imag),
   Stage 2 the noise overlay — then embed each with a `.. figure::` + caption in
   `stage0_import.rst` / `stage1_ft.rst` / `stage2_noise.rst`. Add a `slow`
   smoke test that the harness renders without error. Build docs into
   `docs/build/html` and confirm warning-clean.

Reminder on conventions: build docs into `docs/build/html` (gitignored) so they
are reviewable; direct any run artifacts to `scratch/`; the noise methods harness
is the reference pattern for committed, regenerable figures.
