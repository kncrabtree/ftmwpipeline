# Stage-command migration to the object-verb CLI grammar

The CLI **stage commands** use the **object-verb** grammar pinned by
[`../CLI_STRATEGY.md`](../CLI_STRATEGY.md), having moved off the earlier flat
verb-object form (`compute-ft`, `estimate-noise`, …). Charter: GitHub issue #31,
resolving the second half of divergence **D12** in
[`../ROADMAP.md`](../ROADMAP.md). The cross-cutting **meta-objects** (`settings`,
`scan`) are the first half (issue #28).

This was a CLI-surface refactor only: the analysis logic, the `_internal` impls,
and the Pipeline / functional-API method names are unchanged (those are governed
by [`../API_STRATEGY.md`](../API_STRATEGY.md), not the CLI grammar). Only the
shape of the argument parser and the command strings users (and tests) type
changed.

## Grammar (the contract)

```
ftmwpipeline <object> <verb> <file.ftmw> [options]
```

Every stage object accepts `run` (execute) and `show` (visualize) and carries a
`stageN` synonym interchangeable with its name (`ft run` == `stage1 run`).

| Stage | Object (synonym) | Execute | Visualize | Replaces |
|---|---|---|---|---|
| 0 Data import | `data` (`stage0`) | `import <src>` | `show` | `import-data`, `visualize-data` |
| Start detection | `start` | `run` | `show` | `detect-start`, `visualize-start-detection` |
| 1 FT | `ft` (`stage1`) | `run` | `show` | `compute-ft`, `visualize-ft` |
| 2 Noise | `noise` (`stage2`) | `run` | `show` | `estimate-noise`, `visualize-noise` |
| 2b Tau | `tau` (`stage2b`) | `run [--gaussian]` | `show --kind heatmap\|distribution` | `calibrate-tau`, `calibrate-tau-G`, `visualize-tau-heatmap`, `visualize-tau-distribution` |
| 3 Peaks | `peaks` (`stage3`) | `run` | `show` | `detect-peaks`, `visualize-peaks` |
| 4 Windows | `windows` (`stage4`) | `run` | `show` | `assign-windows`, `visualize-windows` |
| 5 Fit | `fit` (`stage5`) | `run` | `show` + `check` | `fit-peaks`, `visualize-fit`, `validate-stage5-shape-error` |

Decisions baked in (from the spec / issue):

- Execute is uniform `run`, except stage 0 *creates* a file, so its verb is
  `import` with the **source as a positional** (`data import <file> <source>`;
  the old `--source` flag is dropped).
- `start` has no `stageN` synonym (pre-FT step feeding stage 1, not a numbered
  stage).
- Output variants are `--kind`, not new commands (`tau show --kind ...`,
  default `heatmap`).
- `tau run --gaussian` selects the pure-Gaussian τ_G twin.
- Stage-specific extra verbs stay on the object: `fit check` is the SNR-aware
  shape-error report (was `validate-stage5-shape-error`).
- Utilities stay **bare** (no object): `info`, `formats`, `validate`, `version`.
  Note `formats` was registered alongside the data commands but is a bare
  utility; it stays bare.

This is a **pre-release hard cutover** — the old flat commands are removed, no
deprecated aliases (matching the `tune` → `scan` cutover in #28).

## Implementation

- A shared helper `cli/utils.py::add_stage_object(subparsers, name, *, synonym,
  help, description)` builds the object parser (with the `stageN` alias) and its
  `run`/`show` verb subparsers, defaulting to print-help-and-exit-1 when invoked
  with no verb. Each stage's `register_*` / `add_*` function now attaches its
  argument blocks to the verb subparsers instead of to top-level flat parsers.
- The `cmd_*` handlers are unchanged: argument `dest` names are preserved, so the
  handlers read the same `args.*` attributes. The two `tau` consolidations are
  thin dispatchers (`run` routes on `--gaussian`; `show` routes on `--kind`)
  over the existing four handlers; the `run` parser carries the union of the
  exp/Gaussian knobs and `show` carries the shared plot knobs plus `--kind`.
- `data import`'s `source` is a positional; `cmd_data_load` already reads
  `args.source`.

## Test + doc reconciliation

- The integration `_run_cli` helpers and every test that invokes a CLI command
  string use the object-verb form (subprocess arg lists).
- The cross-interface consistency tests' CLI arms use the object-verb form.
- End-user docs / examples (`CLAUDE.md`, README, `docs/source`, the main parser
  epilog) invoke the object-verb commands.

## Scope boundaries

This refactor deliberately left untouched:

- The `settings` / `scan` meta-objects (#28).
- The Pipeline / functional-API method names (API grammar, not CLI).
- Any analysis default (#3).
