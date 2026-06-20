# Project Status

**As of:** 2026-06-08
**Scope:** verified factual state of the codebase. Every claim below was
checked against source or a test run, not against the planning docs. For where
the project is *going*, see `dev-docs/` (roadmap); this file is only what *is*.

## Snapshot

- Version `0.1.0`, Python >= 3.9 (dev/CI on 3.11).
- **Tests: 1247 collected; 1193 pass with `-m "not slow"`, 0 failing** (the
  54 `slow` tests are not run on the inner loop; the full suite is 1245 passed).
  Reproduce:
  `conda run -n ftmwpipeline-dev python -m pytest -q --no-header -o addopts="" -m "not slow"`
  (the `-o addopts=""` is required: `pyproject.toml` hardwires `--cov` flags, so
  `-p no:cov` alone breaks argument parsing).
  - unit: 953 (`tests/unit`), integration: 294 (`tests/integration`),
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
| 3 Peak detection | `stage3_peaks` | Implemented |
| 4 Window assignment | `stage4_windows` | Implemented |
| 5 Fitting | `stage5_fitting` | Implemented |

**Stages 0–2 (implemented):** data loading (Blackchirp / CSV / HDF5 via a
loader registry, format auto-detection), FT processing (preprocess → FFT →
`ComplexFT`, optional trim, parameter persistence; the canonical FT is
unconditionally unapodized, un-windowed, and native-length — no
`expf_us`/`window_function`/`zpf` knobs), and per-bin noise estimation
(`NoiseResult`). The Stage 2 estimator is `estimate_noise_scatter`
— a high-pass, region-aware, Rician-corrected scatter MAD with broad
lower-envelope smoothing, immune to the leakage-pedestal σ inflation on
high-SNR line-dense spectra. It is measured and persisted on the canonical
**active FT** (the single grid every later stage scores, plans, and fits on);
the front-zeroed full-record FT is a Stage 0/1 comparison view only. (The legacy
`estimate_noise_adaptive` variance/skewness binning estimator has been retired
from the package; a minimal comparison reference survives at
`dev-docs/research/noise-snr-scaling/legacy_adaptive.py`.) Each stage is exposed
identically through all three interfaces with HDF5 (de)serialization and
diagnostic visualization.

**Stage 2b (implemented):** `stage2b_tau_calibration` extracts a data-driven
molecular decay constant `τ_maj ± σ_τ` from a sliding-active-window STFT on
the raw FID (between Stage 2 and Stage 3); a Gaussian twin
(`stage2b_tau_G_calibration`) and a 3-way L/G/V shape recommendation ship
alongside. The per-bin exp/gauss/voigt shape fits run on a batched
closed-form-log-seed + Gauss-Newton solver (`solver="scipy"` retained as the
equivalence oracle). Stage 3's gap pass and Stage 5 auto-detect and consume
the calibration.

**Stages 3–5 (implemented):** peak detection (two-pass: blackman-harris
primary + matched-filter gap pass, scored on the scatter noise behind a
continuous leakage-aware floor), window assignment (peak-clustering-driven
edges, `S_coh` de-ramped leakage map, plus edge-free leakage-contributor
subtraction — a bright neighbour's skirt orphaned by the cycle-breaker is kept
as an `edge_free` contributor and subtracted self-contained at fit time, gated
to fire only where it strictly improves the fit), and the Stage 5 fit
(active-portion FT, conservative add-one-peak loop whose AICc-with-`n_eff` accept
gate self-regulates K on width-bounded windows — the per-window peak cap is
removed so a dense cluster is one wide window with enough `n_eff` for the gate,
not several `n_eff`-starved fragments — blend-aware seeder, knockout test,
residual-rescue chain, local thaw +
structural replan, per-band τ routing, Lorentzian/Gaussian shapes,
evidence-triggered leakage-wing baseline (fires on a coherent edge wing or a
smooth in-band leakage pedestal, order-4 with a re-freed τ — the pedestal a
dense ultra-high-SNR spectrum would otherwise force the shared τ to collapse to
absorb), spur masking). Per-stage detail and provenance live in `dev-docs/ROADMAP.md` and
`dev-docs/planning/`. The originally-lost `newfitting/` engine was recreated
against the surviving `bcfitting` shell's contract.

## Architecture (as built)

- **Three interfaces, one implementation.** CLI, the file-bound `Pipeline`
  class, and the stateless functional API (`ftmwpipeline.api`) are thin
  wrappers; all stage logic lives once in `src/ftmwpipeline/_internal/
  stage{0,1,2,2b,2b_g,3,4,5}_impl.py`. A cross-interface consistency test
  suite (20 tests) enforces identical results. Verified delegation:
  `cli/* → _internal`, `api.py → Pipeline → _internal`, `Pipeline → _internal`.
- **`.ftmw` files are lightweight.** Each experiment is one self-contained
  HDF5 file holding FID + parameters + provenance only (~100KB). `ComplexFT`
  is **recomputed on demand**, never persisted (this enables real parameter
  exploration). `NoiseResult` *is* persisted (group `stage2_noise_result`).
- **Verified `.ftmw` group layout:** `source_metadata` (attrs),
  `pipeline_stages` (attr `completed_stages`), `stage0_fid_data` (FID +
  `recommended_processing`), `processing_parameters/ft_processing` (Stage 1
  params — there is intentionally no `stage1_*` data group),
  `stage2_noise_result`, `stage2b_tau_calibration` /
  `stage2b_tau_G_calibration`, `stage3_peaks`, `stage4_windows`,
  `stage5_fitting` (each present once its stage has run).
- **Interfaces:**
  - CLI (object-verb grammar): stage objects `data`/`stage0`
    (`import`/`show`), `start` (`run`/`show`), `ft`/`stage1`
    (`run`/`show`), `noise`/`stage2` (`run`/`show`), `tau`/`stage2b`
    (`run [--gaussian]` / `show --kind heatmap|distribution`),
    `peaks`/`stage3` (`run`/`show`), `windows`/`stage4` (`run`/`show`),
    `fit`/`stage5` (`run`/`show`/`check`); the `scan` meta-object
    (`scan list`/`run`/`all`), the `settings` meta-object
    (`settings show`/`set`/`export`); bare utilities `formats`, `info`,
    `validate`, `version`.
  - `Pipeline` is constructed via `Pipeline.create(...)`,
    `Pipeline.open(...)`, or the smart constructor `Pipeline(path)`
    (opens if present, else `FileNotFoundError` with guidance).
  - Functional API is the `ftmwpipeline.api` namespace
    (`import ftmwpipeline.api as ftmw`). `process_experiment` /
    `batch_process_experiments` and `Pipeline` are also exported at the
    package top level.

## Notable behaviors

- **Stage-completion tracking is coherent.** Stages record their canonical
  key (`stage0_fid_data` … `stage5_fitting`, with `stage2b_tau_calibration`
  between Stage 2 and Stage 3) through one mechanism with declared
  dependencies; `Pipeline.info()` reflects on-disk state; validation checks
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
- **Test artifact location:** non-interactive `ft show` writes PNGs to
  the working directory; test runs leave `*_enhanced_spectrum.png` (now
  git-ignored). The underlying default-output-to-cwd behavior remains.

## Not in scope of current state

Batch parallelism, performance benchmarking, and PyPI packaging are all
unimplemented.
