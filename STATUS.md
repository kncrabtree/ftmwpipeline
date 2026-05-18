# Project Status

**As of:** 2026-05-17
**Scope:** verified factual state of the codebase. Every claim below was
checked against source or a test run, not against the planning docs. For where
the project is *going*, see `dev-docs/` (roadmap); this file is only what *is*.

## Snapshot

- Version `0.1.0`, Python >= 3.9 (dev/CI on 3.11).
- **Tests: 184 passing, 0 failing.** Reproduce:
  `conda run -n ftmwpipeline-dev python -m pytest -q --no-header -o addopts=""`
  (the `-o addopts=""` is required: `pyproject.toml` hardwires `--cov` flags, so
  `-p no:cov` alone breaks argument parsing).
  - unit: 142 (`tests/unit`), integration: 42 (`tests/integration`),
    performance: 0 (`tests/performance` is an empty package).
- Dev environment is the conda env `ftmwpipeline-dev` (from
  `environment-dev.yml`, the superset). `environment.yml` is the lightweight
  runtime env. (These two files' roles were swapped on 2026-05-17 to match
  convention; README still describes them backwards — pending doc-sync.)

## Pipeline stages

The pipeline is `FID -> ComplexFT -> NoiseResult -> Peaks -> Windows ->
FittedPeaks`. Stages are tracked in the `.ftmw` file with declared
dependencies (`file_manager.py: PipelineStageTracker.STAGE_DEPENDENCIES`).

| Stage | Key | State |
|---|---|---|
| 0 Data import | `stage0_fid_data` | Implemented |
| 1 FT processing | `stage1_complex_ft` | Implemented |
| 2 Noise estimation | `stage2_noise_result` | Implemented |
| 3 Peak detection | — | **Not implemented** (stubs) |
| 4 Window assignment | — | **Not implemented** (stubs) |
| 5 Fitting | — | **Not implemented** (stubs) |

**Stages 0–2 (implemented):** data loading (BlackChirp / CSV / HDF5 via a
loader registry, format auto-detection), FT processing (preprocess → FFT →
`ComplexFT`, optional trim, parameter persistence), and adaptive
variance/skewness-based noise estimation (`NoiseResult`). Each is exposed
identically through all three interfaces with HDF5 (de)serialization and
diagnostic visualization.

**Stages 3–5 (not implemented):** every function in
`src/ftmwpipeline/peak_detection/`, `window_assignment/`, and `fitting/` is a
one-line `raise NotImplementedError`. The original `bcfitting` reference
implementation these were to be ported from is **permanently lost** (never
committed to GitHub, lost in a machine migration). Stages 3–5 are therefore a
**clean-room reimplementation**, not a port — there is no source to extract
from.

## Architecture (as built)

- **Three interfaces, one implementation.** CLI, the file-bound `Pipeline`
  class, and the stateless functional API (`ftmwpipeline.api`) are thin
  wrappers; all stage logic lives once in `src/ftmwpipeline/_internal/
  stage{0,1,2}_impl.py`. A cross-interface consistency test suite (18 tests)
  enforces identical results. Verified delegation: `cli/* → _internal`,
  `api.py → Pipeline → _internal`, `Pipeline → _internal`.
- **`.ftmw` files are lightweight.** Each experiment is one self-contained
  HDF5 file holding FID + parameters + provenance only (~100KB). `ComplexFT`
  is **recomputed on demand**, never persisted (this enables real parameter
  exploration). `NoiseResult` *is* persisted (group `stage2_noise_result`).
- **Verified `.ftmw` group layout:** `source_metadata` (attrs),
  `pipeline_stages` (attr `completed_stages`), `stage0_fid_data` (FID +
  `recommended_processing`), `processing_parameters/ft_processing` (Stage 1
  params — there is intentionally no `stage1_*` data group),
  `stage2_noise_result` (when noise estimation has run).
- **Interfaces — real names** (planning docs are wrong about several):
  - CLI subcommands: `data-load`, `data-visualize`, `data-info`,
    `ft-process`, `ft-visualize`, `estimate-noise`, `visualize-noise`,
    `validate`, `version`. (No `import-data`/`compute-ft`/`detect-peaks`.)
  - `Pipeline` is constructed via `Pipeline.create(...)` /
    `Pipeline.open(...)` only — there is **no** no-arg or single-path
    constructor.
  - Functional API is the `ftmwpipeline.api` namespace
    (`import ftmwpipeline.api as ftmw`). `process_experiment` /
    `batch_process_experiments` are also re-exported at top level.

## Foundations pass — changes made 2026-05-17

A "fix code before syncing docs" pass resolved 6 failing tests:

- **Stage-completion tracking made coherent.** `stage1_impl` wrote a
  non-canonical key (`stage1_ft_processing`) directly, bypassing the tracker;
  `Pipeline.info()` read a stale in-memory tracker; `validate_pipeline_file`
  assumed every stage has a same-named HDF5 group (false for lightweight
  Stage 1). All three corrected.
- **`validate_installation()`** no longer calls a bare `Pipeline()` (it was
  always reporting `pipeline_creation: False`, visible via
  `ftmwpipeline validate`).
- **`workflows.py`** reimplemented as thin `Pipeline` wrappers
  (`process_experiment`, `batch_process_experiments`; `quick_fit` removed),
  re-exported in `__init__.py`.
- Two tests asserting a never-built monolithic API, and one stale CLI
  success-string assertion, were corrected.

## Known issues / caveats

- **Lint/format debt (repo-wide):** the codebase was never run through its
  configured black / isort / mypy-strict. `black --check` would reformat
  essentially every file (including untouched ones). Deferred as a standalone
  normalization task.
- **`io/complex_ft_serialization.py` is dead code:** present and tested in
  isolation, but never called by the pipeline (ComplexFT is on-demand).
- **Storage-efficiency numbers** quoted in `dev-docs/SERIALIZATION_STRATEGY.md`
  ("~95% reduction", "12.8MB → 100KB") are design targets, not measured;
  `tests/performance/` is empty.
- **Test artifact pollution:** non-interactive `ft-visualize` writes PNGs to
  the working directory; test runs leave `*_enhanced_spectrum.png` in the repo
  root. Not git-ignored.
- Planning docs (`README.md`, `dev-docs/*`, `CLAUDE.md`) contain numerous
  stale/false claims (test counts, command names, `Pipeline()` usage, dead
  links, `bcfitting` port premise). Correcting these is the next phase.

## Not in scope of current state

Peak detection, window assignment, fitting, batch parallelism, performance
benchmarking, and PyPI packaging are all unimplemented.
