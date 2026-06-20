# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`ftmwpipeline` is a Python package for FTMW (Fourier Transform Microwave) spectroscopy
signal processing and peak fitting. Implementation is **stage-based and incremental**;
Stages 0–2 (data import → FT → noise estimation) are implemented. Later stages (peak
detection, window assignment, fitting) are `NotImplementedError` stubs under
`src/ftmwpipeline/`. Reuse is mixed: the earlier reference
`~/github/bcfitting/src/bcfitting/ftmwfitting.py` survives (it has `locate_peaks`, the
sinc-leakage model, and the conservative time-domain *orchestration* shell) and is the
starting point for Stage 3+. The refined `newfitting/` engine
(`fit_time_domain_peaks`, adaptive window selection, peak aggregation) is permanently
lost and is recreated against the surviving shell's contract. See
`dev-docs/planning/` for per-stage plans before implementing any of Stages 3–5.

`STATUS.md` is the verified current state (regenerated from code). `dev-docs/ROADMAP.md`
is the coordination/task doc and holds the code-vs-spec divergence log. The
`dev-docs/*_STRATEGY.md` files are normative specs (timeless requirements), not status.

## Commands

The dev environment is the conda env `ftmwpipeline-dev` (from `environment-dev.yml`, the
superset with tooling + editable install). Run all project commands through it:

```bash
conda env create -f environment-dev.yml          # one-time
conda run -n ftmwpipeline-dev python -m pytest -q --no-header -o addopts=""   # full suite
conda run -n ftmwpipeline-dev python -m pytest -o addopts="" tests/unit/io/test_fid_serialization.py::test_name
conda run -n ftmwpipeline-dev python -m pytest -o addopts="" -m "not slow"    # markers: slow, integration, unit, performance
conda run -n ftmwpipeline-dev ftmwpipeline validate
```

`pyproject.toml` hardwires `--cov` flags into pytest `addopts`, so `-p no:cov` alone
breaks argument parsing — pass `-o addopts=""` to run without coverage. pytest-cov is
in the dev env, so plain `pytest` (with coverage) also works.

**Never pollute the working tree with run artifacts.** Direct any output-producing
command or test (plots, exports, scratch `.ftmw`/`.h5`, coverage HTML) to an *untracked*
location: the gitignored `scratch/` directory at the repo root, or system tmp. Tests must
write only to pytest `tmp_path` (the integration suite already does). Some CLI commands
(`ft show`/`noise show` non-interactive) default to writing into the current
directory — always pass an explicit `--output scratch/...` (or run from `scratch/`) so
nothing lands in tracked paths. `.gitignore` already covers `scratch/`, `output/`,
`cache/`, `*.h5`, `*_enhanced_spectrum.png`; this is the backstop, not the primary
defense — direct output deliberately rather than relying on ignore patterns.

mypy is configured strict (`disallow_untyped_defs`, etc.); new code in `src/` must be
fully type-annotated. Line length is 88. Note the repo was never run through its
configured black/isort/mypy — most files are not yet black-clean; "would reformat" on a
file you didn't make clean is pre-existing debt, not your regression. Keep *new* code
locally black/mypy-clean.

## Architecture: the dual-interface rule

This is the single most important thing to understand. There are **three user-facing
interfaces that must behave identically**, and they must not duplicate logic:

1. **CLI** — `src/ftmwpipeline/cli/*.py`, an **object-verb** grammar: stage objects
   `data`/`start`/`ft`/`noise`/`tau`/`peaks`/`windows`/`fit` (each `run`/`show`, with
   `stageN` synonyms; `data import` creates the file, `tau run --gaussian` /
   `tau show --kind`, `fit check`), cross-cutting meta-objects `settings`/`scan`, and
   bare utilities `formats`/`info`/`validate`/`version`. Entry point:
   `ftmwpipeline.cli:main`.
2. **Pipeline class** — `src/ftmwpipeline/pipeline.py`, file-bound OO interface
   (`Pipeline.create(...)` / `Pipeline.open(...)` / `Pipeline(path)` smart constructor,
   then `.compute_ft()`, `.estimate_noise()`, …).
3. **Functional API** — `src/ftmwpipeline/api.py`, stateless functions taking a `.ftmw`
   path as first arg (`import ftmwpipeline.api as ftmw`).

**All three are thin wrappers.** Real logic lives once in `src/ftmwpipeline/_internal/stage{0,1,2}_impl.py`
(plus `_internal/shared_utils.py`). The functional API and CLI generally delegate through
the `Pipeline` class, which delegates to `_internal`. When adding or changing stage
behavior, edit the `_internal` impl and let all three interfaces inherit it — never patch
one interface in isolation. There are integration tests dedicated to this
(`tests/integration/test_cross_interface_consistency.py`); run them after any stage change.

Note: the `dev-docs/*_STRATEGY.md` files are normative specs (intent); `STATUS.md` is
verified current state; `dev-docs/ROADMAP.md` holds the code-vs-spec divergence log.
The known divergences (D1–D6) are resolved — code and specs currently agree — but if
you find a new mismatch, log it in ROADMAP and resolve it deliberately (amend spec or
change code), never silently.

## Architecture: the `.ftmw` file model

Each experiment is one self-contained, portable HDF5 file with a `.ftmw` extension.
`src/ftmwpipeline/file_manager.py` owns this file format and provides
`create_pipeline_file` / `open_pipeline_file` / `validate_pipeline_file`, the
`SourceMetadata` provenance record, and `PipelineStageTracker`.

Stages are tracked by name with explicit dependencies (`PipelineStageTracker.STAGE_DEPENDENCIES`):

- `stage0_fid_data` — raw FID; created at import, no deps
- `stage1_complex_ft` — FT result; requires stage 0
- `stage2_noise_result` — noise estimate; requires stage 1
- `stage2b_tau_calibration` — data-driven τ_maj ± σ_τ from a sliding-active-window
  STFT on the raw FID; requires stages 0, 1, 2. Optional dependency of Stages 3+5:
  the gap-pass matched filter consumes `τ_maj` for its tau_basis_us; Stage 5 uses
  it as the anchor for the bidirectional Gaussian tau penalty and the rescue τ.
  Stage 3 and Stage 5 still run without it (Stage 3 falls back to a default
  `tau_basis_us`; Stage 5's τ₀ falls back to `T_active/3`).
  **Two shape twins.** There are two τ calibrations — the exponential/Lorentzian
  twin here and the Gaussian twin `stage2b_tau_G_calibration` (built by
  `calibrate_tau_G`). Stage 5 fits the *recommended* shape and consumes only the
  **matching** twin (gaussian fit → τ_G twin; lorentzian fit → exp twin); if the
  matching twin is absent it silently falls back to `T_active/3`. So `calibrate_tau`
  (and `calibrate_tau_G`) run the 3-way lineshape vote and, when it names the
  *other* shape, **also build that twin** (default-on, gated by `auto_recommend`),
  keeping Stage 2b self-consistent — a single `calibrate_tau` call yields whichever
  twin Stage 5 will need. The fallback being driven by a missing twin (not the
  `preconditions_passed` flag, which never blocks consumption) was a recurring trap.

Running a stage whose dependency is missing raises `StageDependencyError`. Other custom
exceptions (all subclass `PipelineFileError`): `PipelineExistsError` (create over a file
with a different source without `force=True`), `PipelineCorruptionError`. Re-importing the
same source is detected via `SourceMetadata` hashing and is safe (Jupyter re-run friendly);
importing a *different* source over an existing file is refused unless `force=True`.

Per-stage HDF5 (de)serialization lives in `src/ftmwpipeline/io/*_serialization.py`. Input
formats are pluggable via a loader registry: `src/ftmwpipeline/io/data_loaders/` registers
`blackchirp`, `csv`, `ftmw-hdf5` (the native self-describing HDF5 input format), and
`keysight-mat`; `detect_format()` auto-detects. To add a *new binary* format, subclass
`BaseLoader` and `register_loader(...)` in `data_loaders/__init__.py`. For *generic* data,
the no-code path is to shape it into `ftmw-hdf5` or a `csv` column (acquisition metadata
and clock declarations via a `--metadata` JSON/YAML sidecar or the `clocks` CLI); the
`csv`/`ftmw-hdf5` loaders share `io/input_metadata.py`, which resolves metadata by
`explicit > sidecar > embedded > default`. The full contract is
`dev-docs/planning/user-docs/data-input-format-spec.md`.

## Core data structures

`src/ftmwpipeline/core/data_structures.py` defines the domain types: `FTMWData`, `FID`
(raw time-domain, has `.ft(...)`), `ComplexFT` (frequency domain, has `.trim_to_range(...)`),
`FIDProcessingParameters`, `Sideband`, plus the not-yet-wired `Peak`/`FittedPeak`/
`SpectralWindow`/`FittingResult`. `NoiseResult` lives in
`preprocessing/noise_estimation.py`. The sole Stage 2 estimator is
`estimate_noise_scatter` — a high-pass, region-aware, Rician-corrected scatter
MAD with broad lower-envelope smoothing, immune to the leakage-pedestal σ
inflation on high-SNR line-dense spectra. It emits the per-bin complex-RMS σ_x
that every later stage consumes. See
`dev-docs/research/noise-snr-scaling/report.md`. Stage 2 measures and persists
this σ on the **canonical active FT** (`estimate_active_ft_noise` /
`_internal/active_ft_support.py`), not the full-record persisted spectrum — the
active FT is the single grid every later stage scores, plans, and fits on (see
the noise-authority work below). The legacy level-based `estimate_noise_adaptive`
estimator has been retired from the package; a minimal comparison reference
survives only beside its research report at
`dev-docs/research/noise-snr-scaling/legacy_adaptive.py`.

## Example data and reference parameters

`examples/blackchirp_data/2638/` is a real BlackChirp experiment checked in for tests and
manual runs. FID: 750k points, 15 µs, 40.96 GHz probe, lower sideband. The integration
tests' `standard_ft_params` for this experiment is just the frequency trim to the active
region **26500–40000 MHz**; the canonical FT itself is unapodized and native-length.

```python
import ftmwpipeline.api as ftmw
ftmw.import_data("exp_2638.ftmw", source="examples/blackchirp_data/2638/")
ft = ftmw.compute_ft("exp_2638.ftmw", trim=(26500, 40000))
noise = ftmw.estimate_noise("exp_2638.ftmw")
```

**The canonical FT is unconditionally unapodized, un-windowed, and native-length.**
Explicit user apodization of the canonical FT has been removed: the `expf_us`
(exponential apodization), `window_function` / `winf` (FID window), and `zpf` (zero-pad
factor) knobs no longer exist on `compute_ft` / `FTSettings` / the settings + scan
surfaces. Apodization trades resolution and biases the line shape; zero-padding
interpolates the spectrum bins and corrupts the Stage 2/5 noise and χ² statistics — the
robust per-window fit is the intended alternative. `compute_ft` accepts only data
selection (`start_us` / `end_us` / `trim`) plus display knobs (`units_power` / `rdc`).
Stage 3 peak detection still applies its *own* internal zero-padding for sub-bin
position finding (a throwaway grid, independent of the removed `zpf`). Legacy `.ftmw`
files carrying the retired keys open with a warning and are recomputed unapodized. Run
`ftmw.calibrate_tau(...)` to extract `τ_maj ± σ_τ` before peak detection: Stage 3's
gap-pass matched filter uses `τ_maj` for `tau_basis_us`, and Stage 5 anchors its
bidirectional Gaussian τ penalty on `τ_maj` (the per-window starting τ₀ defaults to the
band-local `τ_maj`, else the band-wide `τ_maj`, else `T_active/3`).

## When extending the pipeline (new stage)

Follow the established pattern, in order: add the algorithm/data structure → add
`_internal/stageN_impl.py` with dependency checking and HDF5 storage → add the stage name
+ deps to `PipelineStageTracker.STAGE_DEPENDENCIES` → add serialization in `io/` → expose
it identically through `pipeline.py`, `api.py`, and a `cli/*_commands.py` subcommand →
add unit tests *and* a cross-interface consistency test. Before starting a new stage,
create its planning doc in `dev-docs/planning/` and register it in
`dev-docs/ROADMAP.md` (see `dev-docs/planning/README.md` for the lifecycle). The
normative requirements for each piece are in the `dev-docs/*_STRATEGY.md` specs.
