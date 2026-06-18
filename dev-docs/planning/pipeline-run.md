# Pipeline `run` — end-to-end orchestration

A single command that drives a raw source through every stage in sequence
(import → FT → noise → τ → peaks → windows → fit → timebase → review), with live
progress, so a user gets a finalized `.ftmw` (and optionally its report) from one
invocation instead of eight. It is **orchestration only** — it adds no analysis;
each stage's logic stays in its own `_internal/stage*_impl`, and `run` calls the
existing `Pipeline` stage methods in order.

## Sequence

```
[detect_start_time(stamp)]    # on by default; compute_ft inherits the stamped start
compute_ft(trim)              # the one required experiment-specific input
estimate_noise()
calibrate_tau()               # stamps recommended_shape + builds the matching τ twin
detect_peaks()
assign_windows()
fit_peaks()                   # no explicit shape — consumes the Stage 2b vote
[calibrate_timebase(clocks)]  # on by default; auto-resolves clocks; warn + skip if none
review_run(sigma_floor)       # → FinalProducts (what reports read)
[report run]                  # only with report=True / --report
```

## Decisions

- **Stops at `review_run`** (the file is finalized and report-ready); `--report`
  also emits the L1 table + L3 single-file report.
- **`trim` is required.** There is no active-band auto-detector and the FT trim
  is not part of the Stage 2–5 settings preset layer, so it is an explicit
  argument. Everything else uses the cross-fixture-validated defaults.
- **`calibrate_tau` always runs** — it stamps the recommended shape and the τ
  anchor `fit_peaks` consumes; skipping it silently degrades the fit (T_active/3
  + a default shape). No explicit `shape` is passed to the fit.
- **Start detection on by default** (`--no-start-detect` to skip); it stamps
  `start_us`, which `compute_ft` inherits.
- **Timebase calibration on by default, non-fatal.** Clocks resolve as explicit
  `clocks`/`--clocks` > persisted Stage-5 `spur.clocks` > the
  `recommended_clock_sources` the loader auto-extracts at import (BlackChirp
  `clocks.csv` + `header.csv`: synthesizer fundamentals locked, the digitizer
  unlocked). When no clock declaration is resolvable the stage is **skipped with
  a warning** (frequencies stay precision-only / `uncalibrated`) rather than
  failing the run; `--no-cal` skips it deliberately with no warning.
- **Fresh build by default** (`force=True` re-import): persisted settings outrank
  code defaults, so a clean rebuild avoids stale build-time knobs.
- **Stop at the first failing stage**, returning which stage failed and the
  error; completed stages are reported.

## Progress output

Required: the user must see processing status without the full INFO log firehose.

- The orchestrator prints a per-stage banner to stderr — `[i/N] <stage> …` then
  `✓ <stage> (Xs)` (or `✗` on failure).
- One log handler is installed for the run. It renders the per-window Stage-5
  records (`plan_execution` logs `window n/total …` at INFO) as a percentage bar,
  passes WARNING+ records through on their own line, and drops the rest — so the
  user sees stage progress and real Stage-5 completion percentage, not every
  message. TTY-aware: a `\r`-updated bar on a terminal, periodic plain lines
  under `conda run` / pipes.

## Interface (dual-interface rule)

- **`_internal/run_impl.py::run_pipeline_impl(...)`** — the single orchestration
  + progress implementation. Returns a structured result
  (`pipeline_file`, `status`, `completed_stages`, `failed_stage`, `error`,
  `timebase`, `report`, `elapsed_s`).
- **`Pipeline.build(source, *, trim, …)`** classmethod → delegates to it.
- **`api.run_pipeline(source, …)`** — per-stage override dicts (`ft_params`,
  `noise_params`, `tau_params`, `peak_params`, `window_params`, `fit_params`,
  `review_params`), the settings cascade applying underneath.
- **CLI `ftmwpipeline run <source> --trim LO HI`** (bare verb) — forwards a
  `--preset` to the stages that accept it rather than exposing per-stage flags;
  `--output`, `--sigma-floor`, `--report` / `--report-dir`, `--force`,
  `--no-start-detect`, `--no-cal`, `--clocks`.
- Supersedes the stages-0–2-only `workflows.process_experiment` (left in place).

## Test plan

- Unit: the `_StageProgress` reporter (banner formatting, per-window record →
  percentage parsing, TTY vs non-TTY rendering); stop-at-first-failure returns
  the failing stage; timebase failure warns + skips without aborting.
- Integration (2638 example): a full `run` produces a finalized file with
  `stage6_review` / FinalProducts; `--report` emits the artifacts; cross-interface
  (api == Pipeline == impl) parity on `completed_stages`.
