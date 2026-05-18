# Project Status

**As of:** 2026-05-17
**Scope:** verified factual state of the codebase. Every claim below was
checked against source or a test run, not against the planning docs. For where
the project is *going*, see `dev-docs/` (roadmap); this file is only what *is*.

## Snapshot

- Version `0.1.0`, Python >= 3.9 (dev/CI on 3.11).
- **Tests: 170 passing, 0 failing.** Reproduce:
  `conda run -n ftmwpipeline-dev python -m pytest -q --no-header -o addopts=""`
  (the `-o addopts=""` is required: `pyproject.toml` hardwires `--cov` flags, so
  `-p no:cov` alone breaks argument parsing).
  - unit: 126 (`tests/unit`), integration: 44 (`tests/integration`),
    performance: 0 (`tests/performance` is an empty package).
- Dev environment is the conda env `ftmwpipeline-dev` (from
  `environment-dev.yml`, the superset). `environment.yml` is the lightweight
  runtime env.

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
one-line `raise NotImplementedError`. Reuse is mixed: the earlier reference
`~/github/bcfitting/src/bcfitting/ftmwfitting.py` survives (detection
`locate_peaks`, the analytic sinc-leakage model, and the conservative
time-domain orchestration shell) and is the starting point. The refined
`newfitting/` engine — `fit_time_domain_peaks`, adaptive window selection,
peak aggregation — is permanently lost (machine migration, never committed)
and is recreated against the surviving shell's contract. Stage 3 has a
planning doc (`dev-docs/planning/stage3-peak-detection.md`); Stages 4–5 do
not yet.

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
- **Interfaces:**
  - CLI subcommands: `import-data`, `visualize-data`, `formats`,
    `compute-ft`, `visualize-ft`, `estimate-noise`, `visualize-noise`,
    `info`, `validate`, `version`. Stages 3–5 commands not yet added.
  - `Pipeline` is constructed via `Pipeline.create(...)`,
    `Pipeline.open(...)`, or the smart constructor `Pipeline(path)`
    (opens if present, else `FileNotFoundError` with guidance).
  - Functional API is the `ftmwpipeline.api` namespace
    (`import ftmwpipeline.api as ftmw`). `process_experiment` /
    `batch_process_experiments` and `Pipeline` are also exported at the
    package top level.

## Notable behaviors

- **Stage-completion tracking is coherent.** Stages record their canonical
  key (`stage0_fid_data`, `stage1_complex_ft`, `stage2_noise_result`) through
  one mechanism; `Pipeline.info()` reflects on-disk state; validation checks
  each stage's real persisted artifact (Stage 1 is lightweight — no group).
- **`workflows.py`** is thin `Pipeline` wrappers only
  (`process_experiment`, `batch_process_experiments`); no analysis logic.
- `ftmwpipeline validate` exercises a real installation smoke check.

## Known issues / caveats

- **Lint/format debt (repo-wide):** the codebase was never run through its
  configured black / isort / mypy-strict. `black --check` would reformat
  essentially every file (including untouched ones). Deferred as a standalone
  normalization task.
- **Performance/storage figures are unmeasured.** Any storage-size or timing
  claim is non-normative until a benchmark measures it; `tests/performance/`
  is empty. Tracked in `dev-docs/planning/perf-benchmarks.md` (ROADMAP D5).
- **Test artifact location:** non-interactive `visualize-ft` writes PNGs to
  the working directory; test runs leave `*_enhanced_spectrum.png` (now
  git-ignored). The underlying default-output-to-cwd behavior remains.

## Not in scope of current state

Peak detection, window assignment, fitting, batch parallelism, performance
benchmarking, and PyPI packaging are all unimplemented.
