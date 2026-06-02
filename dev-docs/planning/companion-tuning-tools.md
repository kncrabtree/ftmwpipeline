# Companion tuning & visualization tools

Plan for productionizing the accumulated parameter-scan and stage-visualization
tooling into a first-class, user-facing surface so a user can understand,
visualize, and optimize pipeline parameters for **their own instrument**, then
capture the result as a reusable instrument preset. Charter: GitHub issue #27.

This is the design + categorization document. Implementation is sequenced in
§Sequencing; nothing here re-derives a specific default (that is the per-knob
audit work tracked by [`instrument-tunable-knobs.md`](instrument-tunable-knobs.md)
and issue #3). This doc owns the *tooling surface*.

## Problem

Development produced a large body of throwaway scripts that scan/tune knobs and
visualize stage outputs — exactly what a user needs for a new spectrometer, but
today undiscoverable, untested, and mostly 2638-shaped. They live in three
places:

- **Tracked** `dev-docs/research/*/`: per-stage `probe_<knob>.py` knob sweeps,
  `harness.py` fixture builders, `prototype.py` algorithm studies, and
  `stage5-cross-fixture/stage5_cross_fixture.py`.
- **Tracked** `scripts/development/`: the generalized
  `stage5-validation/generate_validation.py` (per-window fit visualizer that
  already reads shape/probe/trim/units from any `.ftmw`) and the
  `stage3-coherence-study/` survey scripts.
- **Gitignored** `scratch/` (~142 `.py` across ~50 dirs): the throwaway versions
  whose *intent* is worth preserving (noise_viz, mad-calibration, noise_research,
  issue1_start_time, stage2b-bias, stage2b-polish-validation, stage3_benchmark,
  stage4-audit, the stage5-validation/cross-fixture families).

The goal is a single, discoverable, fixture-agnostic surface that subsumes their
intent, follows the repo's dual-interface rule, and feeds the existing settings
resolver + preset system.

## Inventory and parameter -> tool map

The backbone is the Y-rated (instrument-sensitive) knob set in
[`instrument-tunable-knobs.md`](instrument-tunable-knobs.md). Each Y knob is
cross-referenced below against the existing tool that scans/visualizes it. A
**tracked probe** is a `dev-docs/research/.../probe_<knob>.py` sweep; **scratch**
is a gitignored throwaway; **none** means no tool exists yet. This map is what
the companion surface must cover.

### Pre-Stage 1 — start detection (`StartDetectionSettings`)

| knob | existing tool | kind |
|---|---|---|
| guard_margin_us *(headline)* | `scratch/issue1_start_time/` | scratch |
| sweep_max_us | `scratch/issue1_start_time/` | scratch |
| min_chirp_drop_ratio | `scratch/issue1_start_time/` | scratch |

### Stage 2 — noise σ(f) (`NoiseSettings` adaptive + scatter)

| knob | existing tool | kind |
|---|---|---|
| smoothing.smoothing_window_mhz | `scratch/noise_viz/`, `mad-calibration/` | scratch |
| skirt_exclusion.{strong_peak_snr, skirt_exclusion_k, max_skirt_exclusion_mhz} | `scratch/noise_research/` | scratch |
| scatter window_mhz / pedestal_mhz / smoothing_mhz / line_k / smoothing_percentile | `scratch/noise_research/`, `noise_viz/`; `research/noise-snr-scaling/prototype.py` | scratch + prototype |

The `noise_viz/` family is already multi-fixture (`FIXTURES = [...]`) — the
strongest scratch visualizer to lift.

### Stage 2b — τ calibration (`TauCalibrationSettings`)

| knob | existing tool | kind |
|---|---|---|
| stft.n_seg | `scratch/stage2b-bias/` | scratch |
| stft.t_sigma | `scratch/stage2b-bias/`, `tau_noise_coupling/` | scratch |
| polish.polish_snr_cap | `research/stage5-tau-calibration/polish_snr_cap_validation.py`; `scratch/stage2b-polish-validation/` | prototype + scratch |
| polish.polish_noise_debias | `research/stage5-tau-calibration/polish_validation.py` | prototype |
| gaussian.snr_min, recommendation.snr_min | (rides the same builds; no dedicated sweep) | none |

### Stage 3 — peak detection (`PeakDetectionSettings`)

| knob | existing tool | kind |
|---|---|---|
| promotion.min_snr | `research/stage3-gaussian-audit/probe_min_snr.py` | tracked probe |
| promotion.internal_min_snr | `research/stage3-gaussian-audit/probe_internal_min_snr.py` | tracked probe |
| promotion.{weak_medium_snr, medium_strong_snr} | `research/stage3-snr-corner/o2_o4_validation.py`; `scratch/stage3_benchmark/` | tracked + scratch |
| primary_pass.min_exclusion_mhz | `research/stage3-gaussian-audit/probe_min_exclusion.py` | tracked probe |
| gap_pass.gap_mask_edge_threshold | `research/stage3-gaussian-audit/probe_gap_mask_edge.py` | tracked probe |

### Stage 4 — window assignment (`WindowPlanningSettings`)

| knob | existing tool | kind |
|---|---|---|
| coherence.edge_threshold | `research/stage4-gaussian-audit/probe_edge_threshold.py` | tracked probe |
| clustering.max_window_width_mhz | `research/stage4-gaussian-audit/probe_max_width.py` | tracked probe |
| contributor.magnitude_attachment_threshold | `research/stage4-gaussian-audit/probe_mag_attachment.py` | tracked probe |
| leakage.tau_us | `research/stage4-gaussian-audit/probe_leakage_tau.py` (+`_p5` Stage 5 χ²ᵣ follow-up) | tracked probe |
| contributor.min_freeze_snr | (none) | none |

### Stage 5 — fitting (`StageFitSettings`)

| knob | existing tool | kind |
|---|---|---|
| tau.fit_tau_min_snr | `research/stage5-gaussian-audit/probe_fit_tau_min_snr.py` | tracked probe |
| conservative.weak_window_snr_threshold | `research/stage5-gaussian-audit/probe_weak_window_snr.py` | tracked probe |
| rescue.snr_threshold | `research/stage5-gaussian-audit/probe_rescue_snr_threshold.py` | tracked probe |
| rescue.prominence_threshold | `research/stage5-gaussian-audit/probe_rescue_prominence_threshold.py` | tracked probe |
| thaw.residual_edge_threshold | `research/stage5-gaussian-audit/probe_residual_edge_threshold.py` | tracked probe |
| baseline.edge_threshold | `scratch/issue3_audit/audit_knobs.py`; `research/stage5-cross-fixture/` | scratch + harness |
| spur.* | `scratch/spur-validation/`, `spur-saturated-validation/` | scratch |

**Coverage summary.** Of the Y-rated knobs: Stages 3–5 have ~13 tracked
`probe_<knob>.py` sweeps with explicit grids; start-detection, Stage 2, and
Stage 2b were tuned in `scratch/` only; a handful (Stage 2b
`gaussian/recommendation.snr_min`, Stage 4 `contributor.min_freeze_snr`) have no
dedicated tool. The companion surface must (a) lift the tracked probes verbatim,
(b) preserve the intent of the scratch tuners, and (c) fill the gaps.

### Reusable harness assets

These three are templates, not throwaways — the companion engine reuses them
rather than rewriting:

- **`research/<stage>-gaussian-audit/harness.py`** — build a fixture `.ftmw`
  through the stage under study once, cache it, then run thin per-variant
  sweeps. Exports `prepare_fixture()`, `run_variant()`, `plot_knob_sweep()`,
  result dataclasses, `write_csv()`. This *is* the sweep loop the engine
  formalizes.
- **`research/stage5-cross-fixture/stage5_cross_fixture.py`** — end-to-end
  canonical pipeline (Stages 0–5) over a fixture list with `--reuse` to reload
  persisted results; emits tiered health metrics. The cross-instrument
  validation template.
- **`scripts/development/stage5-validation/generate_validation.py`** — already
  fixture-agnostic per-window fit visualizer (`--fixture/--random-sample/--worst/--near/--list`,
  reads shape/probe/trim/units from the file). The visualize-half template.

## Existing surface the design builds on

- **CLI** already has a per-stage `visualize-*` command for every data-producing
  stage: `visualize-data`, `visualize-start-detection`, `visualize-ft`,
  `visualize-noise`, `visualize-tau-*`, `visualize-peaks`, `visualize-windows`,
  `visualize-fit`. The **scan/tune half is what is missing.**
- **Dual-interface rule.** Every capability exists identically in CLI
  (`cli/*_commands.py`) + `Pipeline` (`pipeline.py`) + functional `api.py`, all
  thin wrappers over `_internal/stageN_impl.py`. New commands obey this.
- **Settings system.** Each stage has a settings dataclass
  (`core/{noise,peak_detection,window_planning,stage_fit,tau_calibration,start_detection}_settings.py`)
  composed of sub-blocks, with `resolve()` (four-layer: explicit > preset >
  persisted > recommended > hard default), `from_yaml_dict()`, `to_yaml_dict()`,
  and `load_preset()`. Presets wrap each stage under a top-level
  `stage2:`/`stage2b:`/`stage3:`/`stage5:` block
  (`presets/instrument_bc_2638.yaml`). `to_yaml_dict()` already serializes
  chosen values — preset emission is wiring, not new physics.

The one thing that does **not** exist: a central registry mapping a dotted knob
path (e.g. `stage3.promotion.min_snr`) to its stage, its settings sub-field, a
default sweep grid, a metric extractor, and an optional plot. That registry is
the spine of this work.

## Design: a unified `tune` namespace over a knob registry

Decision (from issue #27 deliverable 3, confirmed with the user): a single
`tune` command **namespace** whose engine is **knob-registry driven** so any
dotted settings path is sweepable by name, with **table-by-default output and
optional per-knob plot adapters**. This combines the generality of a generic
sweep engine with the reality that each knob's diagnostic output differs:
the engine always produces a table; a knob with a registered plot adapter also
gets its plot; a knob without one falls back to table-only. No per-knob
top-level commands proliferate.

### The knob registry

A new module (`_internal/tuning/registry.py`) enumerates every tunable knob as a
`KnobSpec`:

```
KnobSpec:
  path: str               # dotted: "stage3.promotion.min_snr"
  stage: str              # which stage the sweep re-runs
  settings_field: ...     # how to set it on the stage's settings dataclass
  default_grid: Sequence  # the probe's sweep values (lifted from probe_<knob>.py)
  metric: Callable        # (built .ftmw) -> scalar/row the table reports
  plot: Optional[Callable]  # (sweep results) -> figure; None => table-only
  inst_sensitivity: str   # Y / N / maybe (mirrors the knob registry)
  help: str               # one-line meaning (mirrors instrument-tunable-knobs.md)
```

The registry is seeded directly from the tracked `probe_<knob>.py` grids and
the `instrument-tunable-knobs.md` table, so it stays the single source of truth
for "what is tunable and how." Setting a dotted path on the right settings
dataclass reuses the existing sub-block mechanics (`from_yaml_dict`-style field
addressing); no new settings plumbing.

### The sweep engine

`_internal/tuning/engine.py` is knob-agnostic:

1. Build the fixture once through the stage *upstream* of the knob (reusing the
   `<stage>-gaussian-audit/harness.py` build + cache pattern), with a `--reuse`
   path to reload an already-built `.ftmw` (the `stage5_cross_fixture.py`
   pattern).
2. For each grid value: set the knob via its `KnobSpec`, re-run only the
   affected stage(s), collect the `metric`. A **progress indicator** (header
   naming the knob + grid, then an in-place per-value line) is emitted to
   stderr on every surface unless `quiet=True` (CLI `-q/--quiet`); a custom
   `progress(done, total, value)` callback can replace it.
3. Emit a table (always) to stdout + a CSV under a designated **output
   directory** (default: cwd; development work points it at a `scratch/`
   subdir). If `plot` is registered, render the figure to the same dir.
4. **Recommend** a setting from the sweep when the `KnobSpec` carries a
   recommender (best grid value under the knob's metric, with the comparison
   direction declared per knob). Best-effort — a knob may omit it. Regardless
   of whether a recommendation is produced, the engine always prints
   **how to apply** the chosen value: persist it onto the experiment's `.ftmw`
   (the persisted settings layer) or write/update a target instrument `.yml`.
   This is the seam to the deferred preset-emit UX (§Open decisions 1).

Per-knob output specificity is handled entirely by the `metric`/`plot`/recommender
fields, so the engine never grows knob-specific branches. Interactive rendering
is a presentation concern handled at the CLI layer (below), not the engine.

### CLI surface (sketch — subject to refinement during build)

```
ftmwpipeline tune list                       # enumerate knobs (path, stage, Y/N, default grid)
ftmwpipeline tune scan  <file> --knob stage3.promotion.min_snr \
    [--grid 2,3,4,5] [--reuse] [--output-dir DIR] [--interactive] [-q]
ftmwpipeline tune show  <file> --knob ...    # render the knob's plot/table for the persisted value
```

`tune scan` defaults the grid from the `KnobSpec` when `--grid` is omitted.
`--output-dir` defaults to cwd (development work passes a `scratch/` subdir);
the command never writes the tree implicitly. `--interactive` steers plotting
to an interactive backend (e.g. QtAgg) when the environment supports it, falling
back to file output otherwise.

### Dual-interface obligations

Per the rule, the sweep engine is implemented once in `_internal/tuning/` and
exposed through all three surfaces: CLI `cli/tune_commands.py`,
`Pipeline.tune_scan(...)` / `Pipeline.tune_list()`, and `api.tune_scan(...)` /
`api.tune_list()`. A cross-interface consistency test accompanies it (the
registry + engine make this cheap: same engine, three thin wrappers).

`--interactive` is the one **CLI-only** affordance: the Pipeline/api surfaces run
inside user-controlled scripts where the caller owns figure handling (they get
the result object / figure back and display it themselves), so an interactive
backend belongs only to the terminal entry point. The progress indicator, by
contrast, is available on **all** surfaces (on by default, `quiet=True` to
suppress) since scripts benefit from it too. All other behavior — output
directory, grid, reuse, recommendation — is identical across the three surfaces.

## Preset emission — deferred

Issue #27 deliverable 4 is the "tune -> write instrument preset YAML" end goal.
**Deferred by decision:** the UX is best chosen after the
`tune` surface is usable, to avoid committing to non-intuitive caching/merge
behavior before the workflow is felt. The enabling infrastructure already exists
(`to_yaml_dict()` per stage + the `stageN:`-block preset format), so this is a
fast follow once the surface lands. Captured as an open decision below.

## Sequencing

Stable stages first (issue #27 deliverable 5, confirmed): the spine
(registry + engine + dual-interface scaffolding + table/plot-adapter contract)
is proven on a low-churn stage, then fanned out.

1. **Spine on a stable stage.** Build `registry.py` + `engine.py` +
   `cli/tune_commands.py` + Pipeline/api wrappers + cross-interface test, with
   **Stage 2 noise** and **start detection** as the first registered knobs
   (these were scratch-only and are algorithmically stable). Lift `noise_viz/`
   and `issue1_start_time/` intent into their plot adapters.
2. **Stage 2b τ calibration.** Register `n_seg`, `t_sigma`, `polish_snr_cap`,
   `polish_noise_debias`; lift `stage2b-bias/` and `stage2b-polish-validation/`.
3. **Stage 3 -> 4 -> 5.** Heaviest; register the ~13 tracked `probe_<knob>.py`
   grids/metrics/plots verbatim into the registry (they already encode grid +
   metric + plot). Reuse the `<stage>-gaussian-audit/harness.py` builders.
4. **Preset emission** (resolve the deferred decision) once the surface is felt.
5. **Gap-fill** the no-tool knobs (Stage 2b `*.snr_min`, Stage 4
   `min_freeze_snr`) as registry entries.

## Implementation status

The surface lives in `src/ftmwpipeline/_internal/tuning/` (`registry.py`,
`engine.py`, `plots.py`), exposed through `cli/tune_commands.py`
(`tune list` / `tune scan`), `Pipeline.tune_scan` / `Pipeline.tune_list`, and
`api.tune_scan` / `api.tune_list`. Tests: `tests/unit/_internal/tuning/` and
`tests/integration/test_tune_cross_interface.py`.

Built (sequencing steps 1–2):

- **Engine + registry.** Dotted-path `KnobSpec` (run / metric / optional plot
  adapter / optional recommender / `see_also`); per sweep the engine always
  emits a table + CSV, renders the plot when an adapter is registered (else
  table-only), produces a best-effort recommendation, and prints how-to-apply
  text. Flags: `--output-dir` (default cwd), `--reuse`, `--interactive`
  (CLI-only), and a progress indicator on every surface (`quiet` / `-q` to
  suppress). Plot adapters receive a `PlotContext` (working `.ftmw`) for source
  data such as the FID.
- **Registered knobs:**
  - Spectrum-vs-start — `stage1.start_us` and `start.guard_margin_us` share one
    spectra ladder: a top FID panel marking the window starts (+ chirp end for
    guard), over linear active-band |FT| panels with a shared y scaled to the
    percentile floor, so the chirp/ringdown residue and its collapse across
    starts are visible; percentile-floor metric. Guard only offsets the start
    past the detected chirp end, so it detects once per sweep and varies the
    offset (a spectrum-impact knob, not a detection knob).
  - Detection knobs — `start.sweep_max_us`, `start.min_chirp_drop_ratio` move
    the chirp end, so they keep the Σ|FT|-vs-start detection-curve plot and
    `see_also`-point at the spectrum knobs.
  - Stage 2 noise — `stage2.scatter.{window_mhz,pedestal_mhz,smoothing_mhz}`,
    `stage2.smoothing.smoothing_window_mhz`: σ(f) overlay + metric trend.
    Every Stage 2 sweep drives the estimator through a `NoiseSettings`
    bundle (`settings=`), not the deprecated per-knob kwargs — this
    required backfilling a `scatter` sub-block into `NoiseSettings` (the
    scatter estimator, the default, previously had no settings route); see
    `settings-backfill.md` shims #15/#16.
  - Stage 2b tau — `stage2b.stft.{n_seg,t_sigma}`,
    `stage2b.polish.{polish_snr_cap,polish_noise_debias}`: τ_maj ± σ_τ trend
    with contributor count (the boolean `polish_noise_debias` is table-only).

Remaining:

- **Stage 3 → 4 → 5 knobs** (step 3): lift the ~13 tracked `probe_<knob>.py`
  grids/metrics/plots into registry entries, reusing the
  `<stage>-gaussian-audit/harness.py` builders. Stage 5 sweeps re-run
  `fit_peaks` per value — keep grids tight, lean on `--reuse`, and test against
  the small dependency-free-windows fixture.
- **Preset emission** (step 4, deferred — see §Open decisions 1).
- **Gap-fill** the no-tool knobs (step 5).
- **Manual validation:** every knob beyond `start.guard_margin_us` still needs a
  drive-through on real data to confirm its metric/plot before it is relied on.

## Test plan

- Unit: registry well-formed (every `KnobSpec.path` resolves to a real settings
  field; grids non-empty; metric callable); engine sweep on a tiny fixture
  returns a row per grid value; table-only fallback when `plot is None`.
- Cross-interface consistency: `tune scan`/`tune list` identical across CLI,
  Pipeline, api (new `tests/integration` case). `--interactive` is CLI-only and
  excluded from the parity assertion.
- Output directory is honored (writes land in the passed dir, default cwd) and
  nothing writes the tree implicitly; assert with `tmp_path`.
- Recommendation: a knob with a recommender returns the expected best grid
  value; the "how to apply" instructions are always emitted regardless.
- Keep the suite within the runtime budget — sweeps in tests use a trimmed
  small fixture (the `baseline_2638_stage4_small` dependency-free-windows
  pattern), not full builds.

## Out of scope

- Re-deriving any specific default (per-knob audit; issue #3 /
  `instrument-tunable-knobs.md`).
- Cleaning up / consolidating the `scratch/` duplication clusters (e.g. the
  ~12 `stage5-validation*` dirs) — orthogonal housekeeping.

## Open decisions

1. **Preset-emit UX** (deferred above): standalone `emit-preset` vs
   `--save-preset` on `tune scan` vs both; and the merge/caching semantics when
   building a preset incrementally across sessions. Recommendation: engine stage
   prints recommended setting and shows user how to persist if desired: either
   local to the `.ftmw` file or to a target `.yml` file. Likely best to separate
   from the raw sweep/visualization code.
2. **Plot-adapter depth at first landing**: ship table-only for all knobs first
   and add plot adapters incrementally, or port each knob's plot as it is
   registered. Leaning table-first to get the surface usable fast.
3. **`tune show` scope**: whether a persisted-value visualizer is part of the
   spine or folds into the existing `visualize-*` commands.
