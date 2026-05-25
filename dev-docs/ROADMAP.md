# Roadmap

Coordination and task-tracking for ftmwpipeline development. This document is
deliberately lean: it tracks *what is being worked on and what is decided*, and
points to the documents that hold the detail. It is not a changelog or a status
narrative.

- **Verified current state:** see [`../STATUS.md`](../STATUS.md) (regenerated
  from code, not from plans). Do not restate status here; link to it.
- **Normative specs:** the `*_STRATEGY.md` documents in this directory are
  specifications — timeless requirements, not progress reports.
- **Implementation planning:** per-feature plans live in
  [`planning/`](planning/) and are registered below. On completion they become
  implementation overviews that later seed user documentation.

## Vision

A dual-interface (CLI + Python), file-centric pipeline for FTMW spectroscopy.
Each experiment is one portable `.ftmw` file progressing through stages
`FID → ComplexFT → NoiseResult → Peaks → Windows → FittedPeaks`. All interfaces
share one implementation and must produce identical results.

## Stage status

Authoritative detail in [`../STATUS.md`](../STATUS.md). Summary only:

| Stage | State | Plan |
|---|---|---|
| 0 Data import | Implemented | — |
| 1 FT processing | Implemented | — |
| 2 Noise estimation | Implemented | [`planning/stage2-noise-estimation.md`](planning/stage2-noise-estimation.md) |
| 3 Peak detection | Implemented | [`planning/stage3-peak-detection.md`](planning/stage3-peak-detection.md) |
| 4 Window assignment | Implemented | [`planning/stage4-window-assignment.md`](planning/stage4-window-assignment.md) |
| 5 Fitting | Planning | [`planning/stage5-fitting.md`](planning/stage5-fitting.md) |

Stages 3–5 are partly **port-and-refine**, partly **recreate**. The earlier
reference `~/github/bcfitting/src/bcfitting/ftmwfitting.py` survives (detection
`locate_peaks`, the analytic sinc-leakage model, and the conservative
time-domain *orchestration* shell). The refined `newfitting/` engine —
`fit_time_domain_peaks`, adaptive window selection, peak aggregation — is
permanently lost and is recreated against the surviving shell's known
contract.

## Specifications

| Spec | Covers |
|---|---|
| [`API_STRATEGY.md`](API_STRATEGY.md) | Python API (Pipeline class + functional API), `.ftmw` file lifecycle, safe re-import |
| [`CLI_STRATEGY.md`](CLI_STRATEGY.md) | Command-line interface contract |
| [`SERIALIZATION_STRATEGY.md`](SERIALIZATION_STRATEGY.md) | `.ftmw` storage model and invariants |
| [`TESTING_STRATEGY.md`](TESTING_STRATEGY.md) | Test requirements across interfaces |

## Planning documents

Per-feature implementation plans. Lifecycle and conventions:
[`planning/README.md`](planning/README.md).

| Document | Status |
|---|---|
| [`planning/stage2-noise-estimation.md`](planning/stage2-noise-estimation.md) | Implementation summary — MAD/median subdivision + moving-median σ + Lorentzian-skirt exclusion. Algorithmic-choice provenance in [`research/noise-grid-invariance/report.md`](research/noise-grid-invariance/report.md) |
| [`planning/stage3-peak-detection.md`](planning/stage3-peak-detection.md) | Implemented (finalized; detection/promotion split + provenance) |
| [`planning/stage4-window-assignment.md`](planning/stage4-window-assignment.md) | Implemented (finalized) |
| [`planning/stage5-fitting.md`](planning/stage5-fitting.md) | Planning (step-1 plan; implementation not started) |
| [`planning/stage5-residual-rescue.md`](planning/stage5-residual-rescue.md) | Implementation summary — rescue chain + phase-coherence screening + AICc-with-n_eff at all three hypothesis-test sites. Algorithmic-choice provenance in [`research/residual-rescue/report.md`](research/residual-rescue/report.md). |
| [`planning/stage5-cross-fixture-validation.md`](planning/stage5-cross-fixture-validation.md) | Planning — per-dataset shape-error ε calibration framework; cross-fixture acceptance metrics; covers the lineshape-deficit physics discovery from Phase 1 validation on 2638 |
| [`planning/intra-window-clustering.md`](planning/intra-window-clustering.md) | Stub — covariance-based intra-window decomposition; supplants Stage 5 `split` |
| [`planning/leakage-detection-rework.md`](planning/leakage-detection-rework.md) | Resolved (D8) — implementation overview |
| [`planning/processing-settings-persistence.md`](planning/processing-settings-persistence.md) | Resolved (D7) |
| [`planning/perf-benchmarks.md`](planning/perf-benchmarks.md) | Deferred (D5) |

## Code vs spec divergences

The specs state intended requirements. Where the code diverges, it is recorded
here and resolved deliberately (amend spec, or change code), never silently.

| # | Divergence | Resolution |
|---|---|---|
| D1 | CLI command names: code used `data-load`/`ft-process`/`ft-visualize`/`data-visualize`/`data-info` vs spec verb-object intent | **Resolved (code):** renamed to `import-data`, `compute-ft`, `visualize-ft`, `visualize-data`, `formats`. `estimate-noise`/`visualize-noise` already conformed. Stage 3+ commands follow the verb-object scheme |
| D2 | `Pipeline("x.ftmw")` smart constructor required by spec, not implemented | **Resolved (code):** `Pipeline(path)` opens if present, raises `FileNotFoundError` with guidance otherwise; `create()`/`open()` unchanged |
| D3 | Spec showed functional API as top-level `import ftmwpipeline as ftmw; ftmw.import_data(...)` | **Resolved (spec):** the canonical functional namespace is `ftmwpipeline.api` (`import ftmwpipeline.api as ftmw`); only `process_experiment`/`batch_process_experiments` are top-level. API_STRATEGY amended to match |
| D4 | CLI `info <file>` + machine-readable `--format json` not implemented | **Resolved (code):** `info` command added with `text`/`json` output |
| D5 | Performance/benchmark tests absent; storage-reduction figures unmeasured | **Deferred (tracked):** specs already made unmeasured figures non-normative; benchmark work tracked in [`planning/perf-benchmarks.md`](planning/perf-benchmarks.md). `tests/performance/` remains empty until then |
| D6 | `io/complex_ft_serialization.py` never invoked by the pipeline | **Resolved (code):** module and its tests removed; the serialization spec prohibits persisting ComplexFT, so it was dead by design |
| D7 | User-chosen FT processing settings (trim, zpf, expf, …) are not persisted as canonical state; later stages silently fall back to import-time *recommended* defaults instead of what the user chose. Stage 3 currently masks this with interim per-stage `trim`/`zpf` options | **Resolved (code + spec):** Stage 1 now persists user-chosen settings (incl. trim) as canonical; Stages 2–5 operate on that grid; changing canonical settings invalidates downstream results. Stage 3's interim `trim`/`zpf` ownership removed from `_internal/stage3_impl`, `Pipeline.detect_peaks`, `api.detect_peaks`, and `cli/peak_commands.py`. `SERIALIZATION_STRATEGY.md` and `API_STRATEGY.md` amended with normative canonical-settings text |
| D8 | Truncation-leakage handling is wrong in both Stage 3 and Stage 4. Stage 3's gap pass promotes a strong line's sinc sidelobes as weak lines — its leakage mask (`estimate_leakage_reach`) is 7–25× too narrow vs the real ±20+ MHz coherent skirt. Stage 4's edge statistic `S_coh` (a coherent windowed sum) cancels on the oscillating sinc skirt and reads noise-level over obvious leakage, so its leakage-touched map, fixed-contributor attachment, and difficulty classification are unreliable on real data | **Resolved (code + docs):** root cause is the full-record rfft phase ramp `exp(±i2πf·t₀)` (`t₀=start_us`) that makes truncation leakage oscillate so a coherent sum cancels on it. Fixed by one shared de-ramp to the active-region turn-on (`deramp_to_active_start` / `leakage_touched_intervals` in `preprocessing/leakage.py`) feeding the existing `S_coh` — no new statistic. Stage 4's edge-coherence calls and Stage 3's gap-pass mask both consume the de-ramped leakage-touched map; `estimate_leakage_reach` is demoted to an unused analytic proposal; `T_edge` calibrated to 8 for both stages. Both research reports (`research/peak-detection`, `research/complex-edge-coherence`) revised. Implementation overview in [`planning/leakage-detection-rework.md`](planning/leakage-detection-rework.md) |
| D9 | Stage 5 fits on the persisted Stage 1 spectrum, an rfft of the *whole* zero-padded record. Adjacent bins are correlated by a Dirichlet kernel (the FFT of the zero-padding indicator), so the effective number of independent samples in any band of `M` bins is `M·α` with `α = N_active/N_padded` (≈ 0.42 for 2638). The naive `N_dof = M − N_params` overcounts by `1/α`; reduced χ², F-test, AIC, and the conservative loop's accept thresholds (incl. the blend-aware seeder's `rchi2 > 1.5` trigger) are all biased optimistic on real data. Task 5's just-committed `fitting/plan_execution.py` carries this bias | **In progress (code; spec already amended):** Stage 5 fits on the **active-portion FT** — the rfft of just the `[t₀, t₀+T]` FID samples with the canonical Stage 1 apodization, so bins are independent (α = 1) and statistics work as written. The persisted FT remains the spectrum of record for display, Stage 3 peak detection, and Stage 4 window planning; the active-FT is internal to Stage 5 and computed on demand. The σ/√2 D-8 weighting still applies bin-by-bin. Per-bin noise on the active-FT is measured fresh by running the existing Stage 2 `estimate_noise_adaptive` directly on the active-FT magnitude spectrum (an earlier "rescale `σ_persisted/√α`" formulation was dropped because the `/√α` relation only holds under unitary FFT normalization, which the production persisted FT does not use; measuring on the same spectrum the fit sees sidesteps that fragility). Stage 5 plan amended ([`planning/stage5-fitting.md`](planning/stage5-fitting.md) §"Spectral domain for the fit"); implementation steps tracked there as task 6. Stage 5 gains an explicit dependency on `stage0_fid_data` (the FID drives the active-FT) alongside `stage4_windows` |

New divergences are appended here as they arise.

Exact on-disk HDF5 group/attribute names are an implementation detail; the code
is the source of truth and the current layout is recorded in `STATUS.md`. The
serialization spec defines invariants, not literal field names.

## Conventions

- Keep this file short. Detail belongs in `STATUS.md`, the specs, or
  `planning/` docs.
- A new Stage gets a `planning/` doc *before* implementation; the doc is
  registered in the table above.
- Resolve divergences explicitly; update the relevant spec or code, then strike
  the row.
- No emojis, no dated "status" prose, no per-commit narrative.
