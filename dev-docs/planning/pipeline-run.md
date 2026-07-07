# Pipeline `run` — end-to-end orchestration

Status: **implemented.** The bare `run` CLI verb, `api.run_pipeline`, and
`Pipeline.build` are shipped across all three interfaces. The sections below are
the overview of what was built.

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
  `review_params`) plus `start_detection_params`, the settings cascade applying
  underneath.
- **CLI `ftmwpipeline run <source> --trim LO HI`** (bare verb) — forwards a
  `--preset` to the stages that accept it; also exposes every stage's individual
  knobs as namespaced flags (`--ft.start-us`, `--fit.tau.max-decay-factor`, …
  see "Namespaced per-knob passthrough" below); `--output`, `--sigma-floor`,
  `--report` / `--report-dir`, `--force`, `--no-start-detect`, `--no-cal`,
  `--clocks`.
- Supersedes the stages-0–2-only `workflows.process_experiment` (left in place).

## Namespaced per-knob passthrough

Beyond the always-on flags above, `run` also exposes every stage's individual
knobs as **namespaced flags**, generated from the same settings dataclass each
stage's own `*_commands.py` subcommand uses — `run` never carries a separate
copy of a stage's CLI surface that could drift from it:

```
ftmwpipeline run raw.dat --trim 8000:18000 \
  --start.guard-margin-us 1.0 \
  --ft.start-us 2.5 \
  --fit.tau.max-decay-factor 0.9
```

- **Mechanism**: `cli/_argspec.py`'s `add_settings_args` / `settings_from_namespace`
  gained an optional `prefix` (e.g. `"ft"`, `"fit"`) that namespaces every
  generated flag/dest under `"{prefix}."` (`--ft.start-us`, dest `"ft.start_us"`;
  a sub-block field becomes `--fit.tau.max-decay-factor`, dest
  `"fit.tau.max_decay_factor"`). The flag body is always derived deterministically
  from the field path (ignoring any hand-picked `flag=` override), so two stages
  can never collide once namespaced onto the shared `run` parser. `prefix=None`
  (the default) reproduces the exact pre-existing per-stage subcommand behavior
  byte-for-byte.
- **Stage → settings class map** (mirrors each stage's own subcommand):
  `ft` → `FTSettings` (`trim` excluded from the generated namespaced flags;
  `--trim` is the canonical top-level flag, with `--ft.trim` accepted as an
  alias of it on the same argument — keeping the exclusion is what lets the
  alias register without an argparse duplicate-option clash),
  `noise` → `NoiseSettings`, `tau` → `TauCalibrationSettings`,
  `peaks` → `PeakDetectionSettings`, `windows` → `WindowPlanningSettings`,
  `fit` → `StageFitSettings`.
- **Stage 0 (`--start.*`)**: `StartDetectionSettings` is a flat frozen dataclass
  with concrete hard defaults (not the `Optional`-everywhere resolution-chain
  pattern), so it carries no `knob_field` metadata for `_argspec` to walk. A
  dedicated `add_start_detection_args` / `start_settings_from_namespace` pair in
  `cli/_argspec.py` builds one `--[prefix.]<field>` float flag per dataclass
  field (from a shared help map) and reconstructs the sparse overrides from only
  the user-specified fields. Both the standalone `start run` / `start show`
  subcommands (prefix `None`) and `run` (prefix `"start"`) call this single
  generator, so the two flag lists cannot drift; `start` additionally keeps a
  `--band MIN MAX` convenience pair folded onto `band_min_mhz` / `band_max_mhz`.
- **Routing**: each namespaced group reconstructs a sparse settings instance
  and, only when at least one flag in that namespace was given
  (`instance.is_empty()` is `False`), passes `{"settings": instance}` as that
  stage's `*_params` override dict — unaffected when no namespaced flag is
  used. **FT is special-cased**: `Pipeline.compute_ft` takes `start_us` /
  `end_us` / `units_power` as flat kwargs (no `settings=` parameter), so `run`
  passes `FTSettings(...).overrides()` (a flat dict) as `ft_params` instead of
  a `{"settings": ...}` wrapper; `run_pipeline_impl` already `setdefault`s
  `trim` onto that same dict.
- **Skipped stages**: `timebase` / `review` / `report` have no
  knob-metadata-bearing settings dataclass wired to a CLI subcommand, so they
  get no namespaced group (a candidate follow-up if/when they grow one).

## Test plan

- Unit: the `_StageProgress` reporter (banner formatting, per-window record →
  percentage parsing, TTY vs non-TTY rendering); stop-at-first-failure returns
  the failing stage; timebase failure warns + skips without aborting.
- Integration (2638 example): a full `run` produces a finalized file with
  `stage6_review` / FinalProducts; `--report` emits the artifacts; cross-interface
  (api == Pipeline == impl) parity on `completed_stages`.
