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
- [ ] Stage 2 — noise.
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

## Next: Stage 0 (fresh session)

Getting Started and Concepts are done; the stage pages begin here. Stage 0 is
the first to run the full per-stage gate (read planning record → thorough code
review, discuss revisions before writing → mine research → American-English
scan → write the page). A fresh session should:

- Create the per-stage tracking doc `stage0-docs.md` alongside this README.
- Read the planning record for import: `planning/scope-record-import.md`
  (raw-scope loaders), the start-detection material, and the ROADMAP /
  COMPLETED entries that touch import; flag anything stale against the code.
- Review the code surface: `file_manager.py` (create/open/validate, provenance),
  `io/data_loaders/` (the loader registry + `detect_format`, the `blackchirp` /
  `csv` / `hdf5` loaders), `_internal/stage0_impl.py`, the `start` detection
  path, and the `data import` CLI / `import_data` / `Pipeline.create` wrappers.
  Surface smells, TODOs, and intent-vs-coverage gaps; propose structural
  cleanups and **discuss any code revisions before writing**.
- Mine `research/` for any import/start-detection justifications worth
  surfacing.
- Write `stage0_import.rst` (currently a stub).

The page should cover both the Blackchirp path (the home instrument) and the
generic CSV/HDF5 + raw-scope-record paths; the instrument-specific scope-record
detail has its own :doc:`Advanced page <scope_record_import>`, so Stage 0 should
link to it rather than duplicate it.
