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
| 2 Noise estimation | Implemented | — |
| 3 Peak detection | Implemented (finalized: detection/promotion split, provenance) | [`planning/stage3-peak-detection.md`](planning/stage3-peak-detection.md) |
| 4 Window assignment | Planning | [`planning/stage4-window-assignment.md`](planning/stage4-window-assignment.md) |
| 5 Fitting | Not started | `planning/stage5-fitting.md` (TBD) |

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
| [`planning/stage3-peak-detection.md`](planning/stage3-peak-detection.md) | Implemented (finalized; detection/promotion split + provenance) |
| [`planning/stage4-window-assignment.md`](planning/stage4-window-assignment.md) | Planning — **current task** |
| [`planning/processing-settings-persistence.md`](planning/processing-settings-persistence.md) | Resolved (D7) |
| [`planning/perf-benchmarks.md`](planning/perf-benchmarks.md) | Deferred (D5) |
| [`planning/noise-estimation-followups.md`](planning/noise-estimation-followups.md) | Open — heuristic audit, deferred bolstering |

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
