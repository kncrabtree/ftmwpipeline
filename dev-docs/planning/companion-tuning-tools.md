# Companion tuning & visualization tools

Plan for productionizing the accumulated parameter-scan and stage-visualization
tooling into a first-class, user-facing surface so a user can understand,
visualize, and optimize pipeline parameters for **their own instrument**, then
capture the result as a reusable instrument preset. Charter: GitHub issue #27.

> **CLI naming note.** This document designs the surface under a `tune`
> command namespace (`tune list` / `tune scan` / `tune scan-all`). That surface
> shipped and was then renamed to the object-verb **`scan`** meta-object
> (`scan list` / `scan run` / `scan all`) in issue #28, with the Python wrappers
> renamed `scan_list` / `scan_run` / `scan_all`; the `tune` command no longer
> exists. The preset-emit half deferred here (deliverable 4) shipped as the
> `settings set` / `settings export` change-grammar. Read the `tune *` command
> names below as their `scan *` equivalents; see
> [`tune-settings-verb.md`](tune-settings-verb.md).

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

### Stage 2 — noise σ(f) (`NoiseSettings` scatter)

| knob | existing tool | kind |
|---|---|---|
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
   This is the seam to #28's value-set grammar (§Preset emission).

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

## Preset emission — folds into #28

Issue #27 deliverable 4 is the "tune -> write instrument preset YAML" end goal.
**Resolved:** preset/value persistence is primarily issue **#28's** scope, not a
separate `tune`-side emit path. #28 builds the grammar for *viewing and setting*
resolved knob values (the `tune settings` verb — resolved per-knob value +
provenance across `.ftmw` / `.yml` / default). Once that set-grammar exists, the
`tune scan` text output gains a closing line per knob that tells the user how to
persist their chosen value — into either a target `.yml` preset or the `.ftmw`
file itself — reusing #28's set path rather than inventing a `tune`-local
emitter. This keeps the sweep/visualization code free of caching/merge UX and
puts a single value-persistence surface in #28. The enabling infrastructure
already exists (`to_yaml_dict()` per stage + the `stageN:`-block preset format),
so this is wiring on top of #28. Design now lives in
[`tune-settings-verb.md`](tune-settings-verb.md). Note the precedence model it
settled: the persisted `.ftmw` value **outranks** the `.yml` preset
(reproducibility paradigm — [`../SERIALIZATION_STRATEGY.md`](../SERIALIZATION_STRATEGY.md)
§"Settings resolution and reproducibility" / ROADMAP D11), correcting the current
resolver; a preset therefore *seeds* unfixed fields rather than overriding the
file.

## Sequencing

Stable stages first (issue #27 deliverable 5, confirmed): the spine
(registry + engine + dual-interface scaffolding + table/plot-adapter contract)
is proven on a low-churn stage, then fanned out.

1. **Spine on a stable stage.** *(done)* `registry.py` + `engine.py` +
   `cli/tune_commands.py` + Pipeline/api wrappers + cross-interface test, with
   Stage 2 noise and start detection as the first registered knobs.
2. **Stage 2b τ calibration.** *(done)* — and extended well past the initial
   four knobs: the full τ surface (exp τ, Gaussian τ_G, multi-band, shape vote)
   is registered (see §Implementation status).
3. **Stages 0, 1, 2 full coverage + surface ergonomics.** *(done)* Every
   instrument-relevant knob for Stages 0–2 is registered and tiered; the
   listing, batch mode, and spectrum-impact plots landed here.
4. **Stage 3 -> 4 -> 5.** Heaviest; register the tracked `probe_<knob>.py`
   grids/metrics/plots into the registry (they already encode grid + metric +
   plot). Reuse the `<stage>-gaussian-audit/harness.py` builders. Follow the
   conventions in §Lessons for the fan-out. **Stages 3, 4, and all of Stage 5
   done** — fit-quality + rescue / spur / thaw families (see §Implementation
   status). Issue #28 is next.
5. **Issue #28** — the resolved-settings view/set grammar (`tune settings`).
   Value persistence folds in here (see §Preset emission): once the set path
   exists, `tune scan` appends per-knob instructions for writing the chosen value
   to a `.yml` preset or the `.ftmw` file. No separate `tune`-side emitter.
6. **Gap-fill** any remaining no-tool knobs as registry entries.

## Implementation status

The surface lives in `src/ftmwpipeline/_internal/tuning/` (`registry.py`,
`engine.py`, `plots.py`), exposed through `cli/tune_commands.py`
(`tune list` / `tune scan` / `tune scan-all`), `Pipeline.tune_{list,scan,scan_batch}`,
and `api.tune_{list,scan,scan_batch}`. Tests: `tests/unit/_internal/tuning/` and
`tests/integration/test_tune_cross_interface.py`.

**Engine + registry.** Dotted-path `KnobSpec` (run / metric / optional plot
adapter / optional recommender / `see_also` / `tier`); per sweep the engine
always emits a table + CSV, renders the plot when an adapter is registered (else
table-only), produces a best-effort recommendation, and prints how-to-apply
text. `run_scan_batch` sweeps a list of knobs, isolating per-knob failures into
`BatchItem`s. Flags: `--output-dir` (default cwd; the engine writes working
copies under a `.tune_work/` subdir there — gitignored), `--reuse`,
`--interactive` (CLI-only), progress on every surface (`quiet` / `-q`). Plot
adapters receive a `PlotContext` (working `.ftmw`) for source data such as the
FID/spectrum.

**Zoom controls (region-based plots).** The Stage 3 and Stage 4 plots auto-select
their per-region zoom panels by divergence; the user can override that on every
surface through the `PlotContext`: `--zoom LO-HI,LO-HI` (`zoom_regions=`) pins
explicit MHz windows verbatim, or `--n-zoom` / `--zoom-width` (`n_zoom=` /
`zoom_width_mhz=`) tune how many regions the auto-selector picks and how wide
each is. Knobs whose plots have no zoom panels ignore these. The controls are
plumbed identically through CLI / Pipeline / api into `run_scan` /
`run_scan_batch`, and a single `--zoom` set applies to every knob in a batch
(regions are knob-independent frequency windows).

**Surface ergonomics.**
- **Tier + sub-block grouping.** Each `KnobSpec` carries `tier`
  (`primary` / `advanced`). `tune list` shows primary knobs by default (a short
  curated entry point), `--all` reveals advanced; a positional path-prefix
  selector (`tune list stage2b.gaussian`) filters; `list_knobs(selector,
  include_advanced=)` is the shared data filter. The listing is a single
  prefix-elided table (`cli/tune_commands.py::_elide_path`) — repeated dotted
  prefixes are blanked/padded, a blank line separates stages.
- **Batch mode.** `tune scan-all <file> [selector] [--all]` sweeps every matched
  knob on its default grid; a knob whose required stage is absent is reported as
  a failed `BatchItem` and the batch continues.

**Knob coverage — Stages 0, 1, 2, 2b complete (tiered).**
- **Stage 0 (start detection, `stage0.*`).** Primary: `guard_margin_us`,
  `sweep_max_us`, `min_chirp_drop_ratio`. Advanced: `step_us`, `floor_factor`,
  `floor_tail_us`.
  `guard_margin_us` shares the spectra ladder (spectrum-impact); the rest use the
  Σ|FT|-vs-start detection-curve plot. `band_min_mhz`/`band_max_mhz` are *not*
  swept (the detector ignores them unless both are set → no meaningful solo
  sweep; reach via `settings=`/`preset=`).
- **Stage 1 (FT, `stage1.*`).** Primary: `start_us` (start ladder),
  `trim_min_mhz`, `trim_max_mhz`, `end_us` (a no-FID-panel band-stack plot —
  the spectrum *is* the impact). `zpf` / `expf_us` / `window_function` are
  deliberately excluded (the canonical analysis is a raw, unapodized FT; they
  corrupt the Stage 2/5 noise + fit statistics). `units_power` is excluded as a
  sweep (degenerate rescale) and deferred to the resolved-settings verb (#28).
  Trim default grids are MHz-absolute and 2638-shaped — override with `--grid`.
- **Stage 2 (noise, `NoiseSettings`).** The scatter estimator is the sole
  Stage 2 method (the legacy adaptive estimator was retired as a user-facing
  method). The knobs are flat on `NoiseSettings` (no sub-block — Stage 2 has one
  estimator), so the paths are `stage2.<field>`. Primary:
  `stage2.{window_mhz,pedestal_mhz,smoothing_mhz}`. Advanced: the rest
  (`line_k,n_iter,region_aware,smoothing_percentile,convolve_mhz`). Every
  Stage 2 sweep drives through a `NoiseSettings` bundle —
  the scatter estimator was backfilled into `NoiseSettings` for this
  (`settings-backfill.md` shims #15/#16). Plot: σ-trend + a full-width
  σ(f)-over-spectrum overlay zoomed to the noise band.
- **Stage 2b (τ, `TauCalibrationSettings`).** Full surface across all sub-blocks:
  `stft`, `polish`, `aggregation`, `band` (multi-band majorities), `gaussian`
  (Gaussian τ_G via `calibrate_tau_G`), `recommendation` (exp-vs-gauss shape vote
  via `recommend_shape`). `_run_tau` routes by sub-block to the right
  orchestrator; all three return `TauCalibrationResult`/`ShapeRecommendation`.
  Plots: τ trend + per-value contributor decay-cloud and τ-vs-frequency panels;
  a vote-bar plot for the shape knobs. Tuple-valued fields (`band_edges_mhz`,
  `tau_G_seeds`) and workflow toggles (`auto_recommend`) are not swept.

**Knob coverage — Stage 3 complete (tiered).** Every `PeakDetectionSettings`
field is registered as `stage3.<sub_block>.<field>` (23 knobs), all driving
`detect_peaks_impl` through a one-field settings bundle (`_run_peaks`) and
sharing one metric (`n_peaks / n_promoted / n_primary / n_gap / snr_p95`) and one
plot (`plot_peak_detection`). `requires="stage2_noise_result"` — Stage 2b is
optional (its presence shape-matches and τ-anchors the gap matched filter; absent,
the gap pass falls back to the user apodization).
- **Primary** (the Y-rated detection-shaping knobs): `promotion.min_snr`,
  `promotion.internal_min_snr`, `primary_pass.min_exclusion_mhz`,
  `primary_pass.primary_leakage_floor_k`, `gap_pass.gap_leakage_floor_k`.
- **Advanced**: the classification edges (`promotion.weak_medium_snr` /
  `medium_strong_snr`), the Savitzky-Golay block, primary apodization/zpf, the
  primary pass's own apodized-domain σ (`primary_pass.noise_*`, mirroring the
  Stage 2 scatter knobs — the second, leakage-suppressed noise level), and the
  gap-pass structural toggles (`run_gap_pass`, `gap_active_zpf`).
- **Plot:** the active FT is invariant across the sweep, so the figure stacks
  (1) a scalar count/SNR trend, (2) a full-width band-wide **survival panel** —
  the log spectrum drawn once with every promoted peak coloured by the *last*
  swept value it survives (plasma ramp: early-drop → survives-throughout), with
  the zoom regions shaded via `axvspan`, and (3) per-value × per-region zoom
  detail: each row one swept value, each column one auto-selected ~100 MHz region
  (ranked by how much the peak set *diverges* across values, richest-region
  fallback), log-scaled with the noise floor pinned near the axis bottom,
  **promoted peaks only** marked solid by pass (primary = blue, gap = green) and
  the per-bin `min_snr·σ` threshold dashed over each. Dropped/below-cutoff
  markers are deliberately *not* drawn (the threshold + survival panel carry that
  story without clutter). This is the "which lines does this value find and
  promote, and how deep into the sweep do they survive" readout — the
  spectrum-impact convention applied to Stage 3.
- The replaced `gap_pass.gap_mask_edge_threshold` (hard `S_coh` mask) is *not*
  registered; its continuous-floor successor `gap_pass.gap_leakage_floor_k` is.

**Knob coverage — Stage 4 complete (tiered).** Every `WindowPlanningSettings`
field is registered as `stage4.<sub_block>.<field>` (9 knobs), all driving
`assign_windows_impl` through a one-field settings bundle (`_run_windows`) and
sharing one metric (the plan shape: `n_windows / n_hard / n_easy / n_free /
n_fixed / n_dep / n_split` + the `width_p50/p95/max` distribution) and one plot
(`plot_window_planning`). `requires="stage3_peaks"`.
- **Primary** (the Y-rated partition-shaping knobs, grids lifted from the
  `stage4-gaussian-audit` probes): `coherence.edge_threshold`,
  `clustering.max_window_width_mhz`,
  `contributor.magnitude_attachment_threshold`, and `contributor.min_freeze_snr`
  (the no-tool gap, filled with a sensible grid).
- **Advanced**: `leakage.tau_us` — Y-rated but demoted, since the boxcar default
  (its `None` grid value) only widens windows and the split proposals absorb
  that downstream; it is a single band-wide scalar (Stage 2b τ is *not* auto-fed
  into Stage 4 — the resolver's recommended layer is reserved-but-unwired), so it
  is low-leverage and its fate (keep / wire the Stage 2b feed / remove in favour
  of other controls) is deferred to the cross-fixture audit (issue #6). Plus the
  coherence band scales (`coherence.edge_m` / `trim_m`) and the isolated-peak /
  per-window caps (`clustering.min_window_half_width_mhz` /
  `max_peaks_per_window`).
- **Plot:** the active FT is invariant across the sweep, so the figure stacks
  (1) a plan-count trend (`n_windows` / `n_hard` / `n_fixed` / `n_split` vs the
  swept value), (2) a full-width **boundary-shift overlay** — the band drawn
  once with *every* swept value's window boundaries overlaid as vertical lines
  coloured by value (HARD spans hatched, split proposals dotted), the zoom
  regions `axvspan`-shaded — so a glance shows how the partition walks as the
  knob changes, and (3) per-value × per-region zoom detail: each row one swept
  value, each column one auto-selected ~150 MHz region (ranked by how much the
  *partition* — window-edge and HARD counts — diverges across values, richest-
  region fallback), log-scaled with the noise floor near the axis bottom, window
  spans shaded by difficulty (easy = green, hard = red) with boundaries and
  split proposals, free peaks (filled) vs fixed contributors (open square), and
  the driving `S_coh` coherence statistic with its `T_edge` threshold on a twin
  axis. This is the "where does this value move the boundaries, and what does
  the statistic that set them look like" readout — the spectrum-impact
  convention applied to Stage 4. *(On a dense fixture like 2638 — ~300–500
  windows — the band-wide overlay reads as a forest of edges; it is most legible
  on sparser instruments. The zoom rows carry the per-region detail regardless.)*

**Knob coverage — Stage 5 fit-quality family done (tiered).** Stage 5 has
*heterogeneous* knobs (rescue adds peaks, spur masks bins, thaw moves
boundaries, τ frees linewidth), so unlike Stages 3–4 it gets **knob-family
plots** rather than one shared plot. The first family — *fit quality*
(`tau` / `conservative` / `penalties` / `seeder` / `baseline`, 24 knobs, 3
primary: `tau.fit_tau_min_snr`, `conservative.weak_window_snr_threshold`,
`baseline.edge_threshold`) — is registered as `stage5.<sub_block>.<field>`, all
driving `fit_peaks_impl` through a one-field bundle (`_run_fit`),
`requires="stage4_windows"`.
- **Metric (`_metric_fit`):** the honest quality lens is the SNR-normalised
  shape-error fraction **ε** (`eps_p50/p95`) and the SNR-aware fail count
  (`n_fail`), reusing the shipped `fitting.validation`
  (`shape_error_fraction` / `snr_aware_chi2_pass`, κ=0.05, F=3.0) so the tuning
  surface and the Stage 5 health report agree — *not* raw χ²ᵣ, which rides an
  SNR² floor and is kept only as a de-emphasised secondary (`chi2r_p50/p95`).
  The category columns `n_peaks` / `n_free_tau` / `sigma_f_khz` track what a fit
  knob structurally moves.
- **Plot (`plot_fit_quality`):** (1) an ε-percentile + fail/peak trend; (2) the
  headline **ε-vs-SNR scatter** coloured by swept value with the pass boundary
  drawn as the flat line `ε = κ` (the gate `χ²ᵣ ≤ F+(κ·SNR)²` *is* `ε ≤ κ`); (3)
  an ε-vs-frequency strip showing where on the band the knob moved the misfit.
- **Window selection (the Stage 5 cost-control, `_internal/tuning/fit_support.py`):**
  a fit sweep would re-fit every window per value (~300 on 2638), so a knob's
  `prepare` hook reduces the plan *once* on the working copy to a representative
  subset — the `fit_top_snr` (3) brightest + a seeded `fit_sample` (20) sample +
  the windows nearest each `fit_freqs` value — closed over joint-fit dependency
  components (out-of-subset fixed contributors stay frozen). For SNR-threshold
  knobs (`select_hint="snr_threshold"`) the sample straddles the grid's SNR range
  so the knob is guaranteed to bite instead of looking inert from a coverage gap.
  Exposed on every surface as `--fit-top-snr` / `--fit-sample` / `--fit-freqs` /
  `--fit-sample-seed` / `--fit-all`. *(Finding: several Stage 5 knobs read flat
  on 2638 — first-try defaults / low leverage; and `fit_tau_min_snr` was orphaned
  until wired — see the §Lessons note.)*

**Knob coverage — Stage 5 rescue / spur / thaw families done (tiered).** The
three renegotiation families complete Stage 5: 18 knobs registered as
`stage5.{rescue,spur,thaw}.<field>`, reusing the fit-quality family's `_run_fit`
runner and the window-reduction `prepare` hook, but each with its own metric +
provenance plot (the families *do* distinct things, so a shared plot would lie).
Primaries are the Y-rated gates: `rescue.{snr_threshold, prominence_threshold}`,
`spur.{integer_tol_mhz, narrowness_ratio, mask_half_width_bins}`,
`thaw.residual_edge_threshold`; the round-caps, cleanup/merge factors, and
detector floors are advanced. The plots read the *persisted renegotiation
histories* (`SpectrumFit.rescue_history` / `thaw_history` / `replan_history` and
the band-level spur catalogue in `parameters`), so no fitting-code change was
needed — but they are bounded by what is persisted. The band-overlay backdrop
(`_band_spectrum`) is the **persisted canonical FT** (loaded via `ctx`), *not*
the Stage 5 `active_ft`: the active FT is an rfft of the truncated FID and so
spans the full 0→Nyquist RF band (the trim is applied only downstream to
windows/peaks), whereas the persisted FT is the truncated-FID *and* trimmed-band
analysis spectrum — what the overlays should show. Markers are absolute MHz, so
they register on it either way. The families are:
- **Rescue (`_metric_rescue` / `plot_rescue`):** per-round counts (added /
  rescue-origin-pruned / merged), χ² before→after, and the detector candidates
  (offset + SNR) are persisted; a per-peak "came from rescue" origin flag is
  *not*, so the view is aggregate. Three co-equal panels (user-chosen): (1) a
  count + median-χ²-drop trend vs value (watch `n_pruned_rescue`, the failsafe —
  lines rescue added that a later refit undid); (2) a band-wide where-rescue-fires
  raster, one row per value, over the spectrum; (3) the candidate-SNR-vs-gate
  strip (per-value gate line in colour when sweeping `snr_threshold`). Columns:
  `n_added / n_pruned_rescue / n_merged / n_win / n_rounds / chi2_drop_pct /
  eps_p50 / n_peaks`.
- **Spur (`_metric_spur` / `plot_spur`):** the gated catalogue
  (`spur_centers_mhz` / `spur_sources` / `spur_mask_half_width_bins`) is computed
  on the *full active FT*, so spur counts are immune to the plan reduction. Plot
  (user-chosen spectrum overlay): (1) a count-by-source trend (total / narrow /
  saturated); (2) `|FT|` drawn once with each gated integer-MHz spur as a vertical
  marker, ±mask half-width shaded, coloured by *how many* values gate it (a
  robustness ramp, not a last-surviving ramp: spur gating is not monotonic in one
  direction across the spur knobs — looser `narrowness_ratio` adds spurs while a
  higher `snr_threshold` removes them — so a directional survival colour would
  collapse to one shade). Columns: `n_spurs / n_narrow / n_saturated /
  mask_hw_bins / eps_p50 / n_peaks`.
- **Thaw (`_metric_thaw` / `plot_thaw`):** literal original-vs-final window
  boundaries are *not* persisted — only the edge-coherence handshake events — so
  the plan's "`plot_thaw` = boundary moves" is realised as the **coherence
  handshake** (user-confirmed). On 2638 thaw accepts 0/12 (replan 0/7), so the
  view must read at zero accepts. Three panels: (1) thaw/replan attempt+accept
  counts + plan-revision trend; (2) a contested-edge raster (○ thaw at the
  contributor freq, △ replan at the surviving-window centre; filled = accepted);
  (3) the before→after edge-S_coh scatter with the trigger threshold drawn —
  diagonal points are edges the handshake left unchanged. Columns: `n_thaw /
  n_thaw_acc / n_replan / n_replan_acc / rev / coh_flag_p95 / coh_red_p50 /
  eps_p50`.

Tests: `_metric_rescue/spur/thaw` reducers + the three adapters in
`tests/unit/_internal/tuning/{test_registry,test_plots}.py` (fake duck-typed
results), plus the real-fit smoke on the 2638 stage-4 fixture.

Remaining:

- **Issue #28 — resolved-settings view/set grammar** (`tune settings`): a view of
  resolved per-knob values + provenance (`.ftmw`/`.yml`/default), reusing the
  registry walk + selector + tiering, plus the set path that writes a chosen value
  to a `.yml` preset or the `.ftmw` file. Value persistence (the #27 deliverable-4
  "tune -> preset" goal) folds in here: once the set path exists, `tune scan`
  appends per-knob persistence instructions (see §Preset emission). This closes
  out the #27 surface.
- **Manual validation:** the Stage 0/1/2/2b/3/4 knobs and the full Stage 5
  surface (fit-quality + rescue / spur / thaw) await a user drive-through on real
  data to confirm each metric/plot before they are relied on. Known low-leverage
  spots on 2638: many fit knobs read flat (first-try defaults), and thaw accepts
  ~0 — so the thaw plot is validated for "reads correctly at zero accepts", not
  for showing accepted handshakes.

## Lessons / conventions for the Stage 3→5 fan-out

Patterns proven on Stages 0–2b that the Stage 3→5 registration should follow:

- **Expose everything, but tier it.** Register every instrument-relevant knob;
  mark the few headline ones `primary` and the rest `advanced`. The default
  `tune list` stays a short starting point; `--all` reaches the long tail. This
  resolved the "don't drown the user" tension.
- **Drive stages through their `settings=` bundle, never per-knob kwargs.** The
  kwargs are deprecation-bound (`settings-backfill.md`). `_run_*` helpers build a
  one-field settings instance and pass `settings=`. If a stage's estimator has no
  settings representation, backfill one first (as was done for scatter).
- **The spectrum impact is what users care about.** Where a knob changes the
  spectrum/fit, the plot should show that directly (band-stack, σ(f)-over-spectrum
  overlay, per-window fit overlays for Stage 5) — not only a scalar-metric trend.
- **Per-value diagnostic panels** (the τ decay-cloud / τ-vs-frequency pattern)
  generalise: for Stage 4/5, consider per-window fit + residual panels per grid
  value, gridspec-stacked, degrading gracefully when data is absent.
- **Don't expose knobs that break the canonical analysis or are display-only.**
  `zpf`/`expf_us`/`window_function` (raw-FT invariant) and `units_power` (scale)
  were excluded deliberately; apply the same judgement to Stage 3–5 internals.
- **Skip un-sweepable fields.** Tuple-valued knobs and toggles with no solo
  effect (`band_edges_mhz`, `tau_G_seeds`, `auto_recommend`, `band_min/max_mhz`)
  are reachable via `settings=`/`preset=` but are not registered as scalar sweeps.
- **Batch + grouping make review tractable.** `tune scan-all stage3` will sweep a
  whole stage in one pass; lean on it (and `--reuse`) for the heavy Stage 5 grids.
- **A flat sweep can mean a dead knob, not a robust default.** The Stage 5
  fit-quality sweep surfaced `fit_tau_min_snr` reading completely flat — the knob
  was orphaned (resolved + documented but consumed nowhere; the live free-τ gate
  was `weak_window_snr_threshold`). The tuning surface is a cheap audit for this:
  when a Y-rated knob is invariant across its whole grid (and the knob-aware
  straddle confirms its regime is covered), suspect the wiring before trusting
  the default. (Fixed: `fit_tau_min_snr` is now the τ-specific floor composed
  with the weak-window floor.)

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

1. **Preset-emit UX** — *resolved* (see §Preset emission): value persistence
   folds into issue **#28**. #28 owns the view/set grammar; once its set path
   lands, `tune scan` appends per-knob instructions for persisting the chosen
   value into a `.yml` preset or the `.ftmw` file. No standalone `tune`-side
   `emit-preset` / `--save-preset` path, and no caching/merge logic in the sweep
   code.
2. **Plot-adapter depth at first landing**: ship table-only for all knobs first
   and add plot adapters incrementally, or port each knob's plot as it is
   registered. Leaning table-first to get the surface usable fast.
3. **`tune show` scope**: whether a persisted-value visualizer is part of the
   spine or folds into the existing `visualize-*` commands.
