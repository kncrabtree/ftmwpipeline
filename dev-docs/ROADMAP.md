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
| 3 Peak detection | Not started | `planning/stage3-peak-detection.md` (TBD) |
| 4 Window assignment | Not started | `planning/stage4-window-assignment.md` (TBD) |
| 5 Fitting | Not started | `planning/stage5-fitting.md` (TBD) |

Stages 3–5 are a **clean-room reimplementation**. The original `bcfitting`
reference code is permanently lost; there is no source to port from.

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
| _(none yet — Stage 3 plan to be created before Stage 3 work begins)_ | — |

## Code vs spec divergences

The specs state intended requirements. Where the current code diverges, it is
recorded here as a pending decision. Nothing is silently changed in either the
spec or the code; each row is resolved deliberately (amend spec, or change code,
or accept as intentional).

| # | Spec requirement | Current code | Status |
|---|---|---|---|
| D1 | CLI command names use verb-object intent: `import-data`, `compute-ft`, `visualize-ft`, `estimate-noise`, `detect-peaks`, `assign-windows`, `fit-peaks`, `info`, `formats` (CLI_STRATEGY) | `data-load`, `ft-process`, `ft-visualize`, `estimate-noise`, `visualize-noise`, `data-visualize`, `data-info`, `validate`, `version`. No `info`, `formats`, `export`, or JSON output mode | Open — decide canonical naming before adding Stage 3+ commands |
| D2 | Convenience constructor `Pipeline("x.ftmw")` opens-if-exists (API_STRATEGY) | Only `Pipeline.create()` / `Pipeline.open()`; bare/single-arg constructor unsupported | Open — implement smart constructor or drop from spec |
| D3 | Functional API usable as `import ftmwpipeline as ftmw; ftmw.import_data(...)` (API_STRATEGY) | Functions live under `ftmwpipeline.api`; only `process_experiment`/`batch_process_experiments` are top-level | Open — decide top-level export surface |
| D4 | CLI provides `info <file>` and machine-readable `--format json` (CLI_STRATEGY) | Not implemented | Open — needed by some tests' helper code; schedule with D1 |
| D5 | Performance/benchmark tests exist; storage-reduction figures are validated (TESTING/SERIALIZATION) | `tests/performance/` is empty; storage figures are unmeasured targets | Open — add benchmarks or remove the numeric claims |
| D6 | `io/complex_ft_serialization.py` is part of the storage layer | Present and unit-tested but never invoked by the pipeline (ComplexFT is on-demand) | Open — remove dead module or document as reserved |

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
