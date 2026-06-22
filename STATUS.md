# Project Status

**As of:** 2026-06-21
**Scope:** verified factual state of the codebase. Every claim below was
checked against source or a test run, not against the planning docs. For where
the project is *going*, see `dev-docs/` (roadmap); this file is only what *is*.

## Snapshot

- Version `0.1.0`, Python >= 3.9 (dev/CI on 3.11).
- **Tests: 1943 collected; 1865 run with `-m "not slow"`, 0 failing** (the
  78 `slow` tests are not run on the inner loop).
  Reproduce:
  `conda run -n ftmwpipeline-dev python -m pytest -q --no-header -o addopts="" -m "not slow"`
  (the `-o addopts=""` is required: `pyproject.toml` hardwires `--cov` flags, so
  `-p no:cov` alone breaks argument parsing).
  - unit: 1607 (`tests/unit`), integration: 336 (`tests/integration`),
    performance: 0 (`tests/performance` is an empty package).
- Dev environment is the conda env `ftmwpipeline-dev` (from
  `environment-dev.yml`, the superset). `environment.yml` is the lightweight
  runtime env.

## Pipeline stages

The pipeline is `FID -> ComplexFT -> NoiseResult -> Peaks -> Windows ->
FittedPeaks -> final products`. Stages are tracked in the `.ftmw` file with
declared dependencies (`file_manager.py: PipelineStageTracker.STAGE_DEPENDENCIES`).

| Stage | Key | State |
|---|---|---|
| 0 Data import | `stage0_fid_data` | Implemented |
| 1 FT processing | `stage1_complex_ft` | Implemented |
| 2 Noise estimation | `stage2_noise_result` | Implemented |
| 2b τ calibration | `stage2b_tau_calibration` / `stage2b_tau_G_calibration` | Implemented |
| 3 Peak detection | `stage3_peaks` | Implemented |
| 4 Window assignment | `stage4_windows` | Implemented |
| 5 Fitting | `stage5_fitting` | Implemented |
| 6 Review / reports | `stage6_review` | Implemented |
| Timebase calibration | `timebase_calibration` | Implemented |

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
subtraction — a bright neighbor's skirt orphaned by the cycle-breaker is kept
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

**Stage 6 (implemented):** `stage6_review` is the read/edit + final-products
surface over the Stage 5 fit. The `review` verbs rank and inspect windows by
attention metric and apply user edits (peak survival merges/splits, candidate
revival) that replay the persisted Stage 5 spur catalog rather than re-deriving
it; the `report` verbs emit the calibrated final-products table (`report table`:
CSV / JSON / LaTeX) and the Level-1-table + Level-3-HTML deliverable
(`report run`), with optional catalog cross-referencing. The reported frequency
uncertainty is a precision budget `sqrt(σ_stat² + (σ_ε·baseband)² + σ_floor²)`
with a user-settable `σ_floor` (default 0).

**Timebase calibration (implemented):** `timebase_calibration` (a Stage-0
dependant, run via the `timebase` verbs or auto-populated) declares the
instrument clock tree and gates the spur lattice / digitizer-clock frequency
correction; the declared clocks resolve from a sidecar (e.g. Blackchirp
`clocks.csv`) or the `clocks` CLI.

## Architecture (as built)

- **Three interfaces, one implementation.** CLI, the file-bound `Pipeline`
  class, and the stateless functional API (`ftmwpipeline.api`) are thin
  wrappers; all stage logic lives once in `src/ftmwpipeline/_internal/
  stage{0,1,2,2b,3,4,5,6}_impl.py` (plus `run_impl.py` for the
  whole-experiment driver and `stage5_validation_impl.py`). A cross-interface
  consistency test suite (33 tests) enforces identical results. Verified
  delegation: `cli/* → _internal`, `api.py → Pipeline → _internal`,
  `Pipeline → _internal`.
- **`.ftmw` files are lightweight.** Each experiment is one self-contained
  HDF5 file holding FID + parameters + provenance only (~100KB). `ComplexFT`
  is **recomputed on demand**, never persisted (this enables real parameter
  exploration). `NoiseResult` *is* persisted (group `stage2_noise_result`).
- **Verified `.ftmw` group layout:** `source_metadata` (attrs),
  `pipeline_stages` (attr `completed_stages`), `stage0_fid_data` (FID +
  `recommended_processing`), `processing_parameters/ft_processing` (Stage 1
  params — there is intentionally no `stage1_*` data group),
  `stage2_noise_result`, `stage2b_tau_calibration` /
  `stage2b_tau_G_calibration`, `timebase_calibration`, `stage3_peaks`,
  `stage4_windows`, `stage5_fitting`, `stage6_review` (each present once its
  stage has run). Resolved per-stage knobs persist separately under
  `processing_parameters/stage{2_noise,2b_tau,3_peaks,4_windows,5_fit}`.
- **Interfaces:**
  - CLI (object-verb grammar): stage objects `data`/`stage0`
    (`import`/`show`), `start` (`run`/`show`), `ft`/`stage1`
    (`run`/`show`), `noise`/`stage2` (`run`/`show`), `tau`/`stage2b`
    (`run [--gaussian]` / `show --kind heatmap|distribution` / `recommend`),
    `peaks`/`stage3` (`run`/`show`), `windows`/`stage4` (`run`/`show`),
    `fit`/`stage5` (`run`/`show`/`check`), `timebase`
    (`run`/`show`), `review` (`run`/`rank`/`show` + the edit verbs
    `apply`/`accept`/`edit`/`merge`/`split`/`undo`/`log`); the `report`
    object (`table`/`run`), the `scan` meta-object (`scan list`/`run`/`all`),
    the `settings` meta-object (`settings show`/`set`/`export`), the `clocks`
    declaration utility, and the bare `run` whole-experiment driver; bare
    utilities `formats`, `info`, `validate`, `version`.
  - `Pipeline` is constructed via `Pipeline.create(...)`,
    `Pipeline.open(...)`, or the smart constructor `Pipeline(path)`
    (opens if present, else `FileNotFoundError` with guidance).
  - Functional API is the `ftmwpipeline.api` namespace
    (`import ftmwpipeline.api as ftmw`). `Pipeline` and the `api` module are
    exported at the package top level; the whole-experiment workflow is
    `api.run_pipeline` / `Pipeline.build`.

## Notable behaviors

- **Stage-completion tracking is coherent.** Stages record their canonical
  key (`stage0_fid_data` … `stage5_fitting`, with `stage2b_tau_calibration`
  between Stage 2 and Stage 3) through one mechanism with declared
  dependencies; `Pipeline.info()` reflects on-disk state; validation checks
  each stage's real persisted artifact (Stage 1 is lightweight — no group).
- **`workflows.py`** holds only the `validate_installation` smoke check; the
  whole-experiment workflow lives in `api.run_pipeline` / `Pipeline.build`.
- `ftmwpipeline validate` exercises a real installation smoke check.

## Known issues / caveats

- **Lint/format:** `src/` is `mypy --strict` clean (132 source files) and
  `black` / `isort` clean. The historical repo-wide formatting debt has been
  normalized; keep new code clean (line length 88).
- **Performance/storage figures are unmeasured.** Any storage-size or timing
  claim is non-normative until a benchmark measures it; `tests/performance/`
  is empty. Tracked in `dev-docs/planning/perf-benchmarks.md` (ROADMAP D5).
- **Test artifact location:** non-interactive `ft show` writes PNGs to
  the working directory; test runs leave `*_enhanced_spectrum.png` (now
  git-ignored). The underlying default-output-to-cwd behavior remains.

## Not in scope of current state

Batch parallelism, performance benchmarking, and PyPI packaging are all
unimplemented.
