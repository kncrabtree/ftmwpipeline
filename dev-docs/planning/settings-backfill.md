# Plan: per-stage settings backfill across the pipeline

Status: **Stage 2b and Stage 2 shipped.** Stages 3 and 4 are queued behind them.

## Stage 2b state

The Stage 2b `TauCalibrationSettings` plumbing is live across all three
user-facing surfaces. The full non-slow test suite (810 tests) is green
under the new wiring, including the existing
`tests/integration/test_cross_interface_consistency.py` and
`tests/integration/test_stage5_settings_propagation.py` suites.

Components landed:

- **`src/ftmwpipeline/core/tau_calibration_settings.py`** — top-level
  `TauCalibrationSettings` plus six sub-dataclasses (`StftSubSettings`,
  `PolishSubSettings`, `AggregationSubSettings`, `BandSubSettings`,
  `GaussianSubSettings`, `RecommendationSubSettings`). `_HARD_DEFAULTS`
  mirrors every `DEFAULT_*` in `fitting/tau_calibration.py`. `resolve()`
  walks the four-layer chain; `to_attrs`/`from_attrs` round-trip the
  nested dict; `to_yaml`/`from_yaml`/`load_preset` cover YAML
  interchange. The Stage 2b `load_preset` reads the `stage2b:` block
  from packaged presets (returns empty when no such block is present
  so a Stage-5-only preset loads cleanly).
- **`src/ftmwpipeline/io/tau_calibration_settings_serialization.py`** —
  HDF5 persistence at `processing_parameters/stage2b_tau` with one
  subgroup per sub-dataclass; `creation_time` and `preset_name` audit
  attrs; `save_/load_/present` helpers; tuple fields (`tau_G_seeds`,
  `band_edges_mhz`, `band_labels`) round-trip via 1-D attr arrays.
- **`src/ftmwpipeline/_internal/tau_settings_resolution.py`** — shared
  resolver glue (`_required_int` / `_required_float` / `_required_bool`
  coercers, `resolve_with_preset_and_persisted` walker) used by all
  three Stage 2b orchestrators.
- **`_internal/stage2b_impl.calibrate_tau_impl`,
  `_internal/stage2b_g_impl.calibrate_tau_G_impl`,
  `_internal/shape_recommendation_impl.recommend_shape_impl`** — gain
  `settings: Optional[TauCalibrationSettings]` and `preset:
  Optional[str]` kwargs (mutually exclusive, matching Stage 5). Each
  builds an explicit `TauCalibrationSettings` from its legacy per-knob
  kwargs, walks `resolve(...)`, lifts the resolved fields into the
  kernel call, and persists the resolved settings via
  `save_tau_calibration_settings_to_h5` so a follow-up no-kwargs call
  inherits the recipe. The pure-exp twin and the Gaussian twin share
  one persisted settings record (Stage 2b is one stage, one settings
  block, three consumers).
- **`Pipeline.calibrate_tau` / `Pipeline.calibrate_tau_G` /
  `Pipeline.recommend_shape`** and the matching `ftmwpipeline.api`
  functions — gain `settings=` / `preset=` kwargs. The `compute_band_majorities`
  parameter dropped from `bool = True` to `Optional[bool] = None` on
  every layer so the resolver drives it; observable no-kwargs behaviour
  unchanged (hard default matches the previous `True`). `Pipeline.calibrate_tau_G`
  / `api.calibrate_tau_G` gain `tau_G_seeds` (previously hidden from
  the public surface).
- **CLI** — `calibrate-tau` and `calibrate-tau-G` each gain
  `--preset NAME_OR_PATH`.
- **Packaged presets** — `instrument_bc_2638.yaml`,
  `gaussian_default.yaml`, `lorentzian_legacy.yaml` rewritten with the
  new `stage5:` wrapper. `core.stage_fit_settings.load_preset` accepts
  both `stage5:` (current) and `fit:` (legacy) spellings; a preset
  carrying both wrappers raises `ValueError`. The Stage 5 loader also
  strips sibling stage blocks (`stage2b:`, `stage2:`, `stage3:`,
  `stage4:`) before parsing so a stage-spanning preset doesn't trip
  the unknown-key gate.

Test coverage shipped:

- `tests/unit/core/test_tau_calibration_settings.py` (24 tests).
- `tests/unit/io/test_tau_calibration_settings_serialization.py` (8 tests).
- `tests/unit/io/test_preset_loading.py` extended (20 tests total) for
  the `fit:` ↔ `stage5:` rename, sibling-block coexistence, the
  Stage 2b preset loader, and the "both wrappers" error.
- `tests/integration/test_stage2b_settings_propagation.py` (50 tests)
  — every routed `TauCalibrationSettings` field reaches the matching
  kernel (`extract_tau_majority` / `extract_tau_G_majority` /
  `compute_shape_recommendation`); the mutual-exclusion rule on
  `settings=` / `preset=` fires on all three impls; a no-kwargs
  follow-up inherits the persisted `n_seg`.

This document grew out of the "Backfill to other stages" follow-up
bullet on [`stage5-fit-settings.md`](stage5-fit-settings.md) and is the
coordination doc for the project. The Stage 5 work is the architectural
template: every backfilled stage reuses the same dataclass shape,
resolution chain, HDF5 persistence convention, and YAML preset
interchange. Where a stage diverges from the template it is documented
here.

## Architectural template (carry-over from Stage 5)

The shape every backfilled stage inherits:

- **Dataclass.** `core/<stage>_settings.py` defines a top-level
  `XxxSettings` dataclass holding `Optional` sub-dataclasses. Every
  field defaults to `None` so the resolution chain can compose. One
  sub-dataclass per HDF5 subgroup / YAML block; sub-blocks are the
  inspection unit.
- **`_HARD_DEFAULTS`.** Nested dict mirroring every `DEFAULT_*`
  constant in the stage's fitting / preprocessing kernel module.
  The kernel module's constants stay the readable canonical source;
  `_HARD_DEFAULTS` tracks them.
- **Resolution chain.** `resolve(explicit, preset, persisted,
  recommended)` walks four layers left-to-right (per field), then
  falls back to `_HARD_DEFAULTS`. Stages whose *recommended* layer
  has no upstream feeder (Stage 2b is the canonical case — it
  *produces* the Stage 5 recommendation; nothing upstream produces
  a Stage 2b recommendation) keep the slot in the signature and
  pass `recommended=None`; the uniform shape is preferred over a
  smaller-but-asymmetric signature so a future cross-stage
  recommender can land without API churn.
- **HDF5 persistence.** `io/<stage>_settings_serialization.py`
  writes to `processing_parameters/<stage>/` (one HDF5 subgroup
  per sub-dataclass; `creation_time` and `preset_name` audit
  attrs; `__None__` sentinel for unset fields).
- **YAML interchange.** `to_yaml` / `from_yaml` / `load_preset`.
  Packaged presets live under `src/ftmwpipeline/presets/` and use
  per-stage top-level blocks (`stage2:`, `stage2b:`, `stage5:`).
  See § "Preset YAML wrapper migration".
- **Three-surface parity.** CLI subcommand, `Pipeline` method,
  and `ftmwpipeline.api` function each gain `settings=` and
  `preset=` kwargs (mutually exclusive — passing both raises
  `ValueError`, matching Stage 5). The existing per-knob kwargs
  stay on signatures and win per field; their hard-coded defaults
  drop to `None` so the resolver picks them up from a preset or
  the persisted layer.
- **Tests.** Per-stage unit suite covering the dataclass /
  resolver / round-trip + an integration "settings propagation"
  test that intercepts the kernel call and asserts every routed
  field reaches the kwargs bag.

## Preset YAML wrapper migration

The first packaged preset (`instrument_bc_2638.yaml`) ships with a
top-level `fit:` block wrapping Stage 5 settings, plus a comment in
the Stage 5 plan that it "leaves room for a future stage-spanning
`ft:` block". The Stage 2b backfill is the moment that future
arrives, so the convention is migrating to **per-stage top-level
blocks** (`stage2b:`, `stage5:`). New shape:

```yaml
stage2b:
  stft:
    n_seg: 10
    polish_snr_cap: 9.0
  gaussian:
    snr_min: 20.0
stage5:
  shape: gaussian
  tau:
    tau_penalty_lambda: 50
```

**Backwards-compat shim (logged in § "Back-compat shims" below).**
`load_preset` accepts both `fit:` (legacy) and `stage5:` (new)
spellings for the Stage 5 block. The `fit:` spelling stays silent
through this project; a `DeprecationWarning` is added in a later
release cycle (tracked as a follow-up here). Existing user presets
and research / dev scripts that already write the `fit:` wrapper
continue to load without modification.

## Stage ordering

| Stage | Module name                       | Status              |
|------:|-----------------------------------|---------------------|
| 2b    | `TauCalibrationSettings`          | Shipped             |
| 2     | `NoiseSettings`                   | Shipped             |
| 3     | `PeakDetectionSettings`           | Queued              |
| 4     | `WindowPlanningSettings`          | Queued              |

Ordering rationale: Stage 2b first because it is the shape-
recommendation home (its persisted attr is what the Stage 5
resolver's *recommended* layer reads). Stage 2 next because the
noise estimator's `DEFAULT_SMOOTHING_MHZ` is the canonical
instrument-tunable knob and the calibration's per-bin σ feeds
Stage 2b's `sigma_x_full_override`. Stages 3 and 4 last, in
pipeline order.

## Stage 2 state

The Stage 2 `NoiseSettings` plumbing is live across all three user-facing
surfaces. The full non-slow test suite (861 tests, +51 from the Stage 2b
landing) is green under the new wiring, including the cross-interface
and Stage 5 / Stage 2b propagation suites.

Components landed:

- **`src/ftmwpipeline/core/noise_settings.py`** — top-level
  `NoiseSettings` plus four sub-dataclasses (`BinningSubSettings`,
  `SkewnessSubSettings`, `SmoothingSubSettings`,
  `SkirtExclusionSubSettings`). `_HARD_DEFAULTS` mirrors every
  function-signature default + module-level constant in
  `preprocessing/noise_estimation.py`. `resolve()` walks the four-layer
  chain; `to_attrs` / `from_attrs` / YAML helpers + `load_preset`
  reading the `stage2:` block from packaged presets.
- **`src/ftmwpipeline/io/noise_settings_serialization.py`** — HDF5
  persistence at `processing_parameters/stage2_noise` with one
  subgroup per sub-dataclass and the standard `__None__` sentinel.
  Intentionally distinct from the existing `/stage2_noise_result`
  group (settings under `processing_parameters/`, results at the root
  — same pattern Stage 5 uses).
- **`preprocessing/noise_estimation.estimate_noise_adaptive`** — the
  kernel signature gains five new keyword-only kwargs threading the
  formerly module-level constants
  (`subdivision_threshold`, `abs_min_bin_size`, `strong_peak_snr`,
  `skirt_exclusion_k`, `max_skirt_exclusion_mhz`). The module-level
  `SUBDIVISION_THRESHOLD = 0.08` / etc. remain the readable canonical
  source the dataclass mirrors; they are the kernel's parameter
  defaults so old callers see identical behaviour.
  `_compute_mad_based_bins` gains `subdivision_threshold`;
  `_exclude_strong_line_skirts` gains the three skirt-exclusion
  kwargs.
- **`_internal/stage2_impl.compute_noise_estimation_impl`** — gains
  `settings: Optional[NoiseSettings]` and `preset: Optional[str]` kwargs
  (mutually exclusive, matching Stages 5 and 2b). Builds an explicit
  `NoiseSettings` from the four legacy per-knob kwargs, walks
  `resolve(...)`, lifts the resolved fields into the kernel call,
  and persists the resolved settings via
  `save_noise_settings_to_h5`. The legacy `from_saved_params=True`
  flag is preserved (reads the legacy `processing_parameters/noise_estimation`
  block, ignores explicit / settings / preset / persisted-stage2_noise
  layers); passing it alongside `settings=` / `preset=` raises
  `ValueError`.
- **`Pipeline.estimate_noise` / `api.estimate_noise`** — gain
  `settings=` / `preset=` kwargs.
- **CLI** — `estimate-noise` gains `--preset NAME_OR_PATH`.

Test coverage shipped:

- `tests/unit/core/test_noise_settings.py` (19 tests).
- `tests/unit/io/test_noise_settings_serialization.py` (8 tests).
- `tests/unit/io/test_preset_loading.py` extended (28 tests total) for
  the `stage2:` block path and sibling-block coexistence.
- `tests/integration/test_stage2_settings_propagation.py` (14 tests)
  — every routed `NoiseSettings` field reaches `estimate_noise_adaptive`;
  `settings=` + `preset=` mutual exclusion; `from_saved_params=True`
  + `settings=` mutual exclusion; a no-kwargs follow-up inherits the
  persisted `smoothing_window_mhz`; `from_saved_params=True` still
  reads the legacy block (back-compat preserved).

## Stage 2b design reference

The sub-block layout, resolver shape, and persistence/YAML conventions
documented here apply to the shipped Stage 2b implementation. They
double as the reference contract for Stages 3 and 4, which inherit the
same architectural template.

### Surface area

Three user-facing kernels, summarized from
[`stage2b-tau-calibration.md`](stage2b-tau-calibration.md):

* `calibrate_tau` — pure-exp τ calibration. Kwargs: `n_seg`,
  `t_sigma`, `tau_max_us`, `rss_gate_factor`,
  `relative_gate_fraction`, `spur_cluster_multiplier`,
  `min_contributors`, `sigma_tau_fraction_max`,
  `bimodality_dominant_fraction`, `polish`, `polish_n_iter`,
  `polish_top_n`, `polish_snr_cap`, `polish_noise_debias`,
  `sigma_x_full`, `compute_band_majorities_flag`,
  `band_edges_mhz`, `band_labels`, `min_contributors_per_band`.
* `calibrate_tau_G` — Gaussian-shape τ_G calibration. Every kwarg
  above plus `snr_min`, `tau_G_bound_lo`, `tau_G_bound_hi`,
  `tau_G_seeds`, `delta_chi2r_min`, `tau_G_upper_fraction`,
  `min_contributors` (smaller default).
* `recommend_shape` — 3-way AICc L/G/V vote. Kwargs: `n_seg`,
  `t_sigma`, `tau_max_us`, `rss_gate_factor`, `snr_min`,
  `tau_bound_lo`, `tau_bound_hi`, `tau_G_seeds`,
  `pure_margin_threshold`, `sigma_time`.

### Sub-block layout

`TauCalibrationSettings` carries six sub-dataclasses, chosen so the
HDF5 subgroups inspect cleanly and the YAML blocks compose:

* **`stft`** — sliding-active-window STFT knobs (shared by all
  three consumers). Fields: `n_seg`, `t_sigma`, `tau_max_us`,
  `tau_max_factor`, `rss_gate_factor`, `relative_gate_fraction`,
  `sigma_x_full`, `sigma_time`.
* **`polish`** — pure-exp polish step (used by `calibrate_tau`
  only; `calibrate_tau_G` polishes internally via its own NLS
  multi-start, and `recommend_shape` operates on the classifier
  output without polish). Fields: `polish`, `polish_n_iter`,
  `polish_top_n`, `polish_snr_cap`, `polish_noise_debias`.
* **`aggregation`** — majority-vote + acceptance pre-conditions
  shared by both τ twins. Fields: `min_contributors`,
  `sigma_tau_fraction_max`, `bimodality_dominant_fraction`,
  `sigma_tau_floor_us`, `spur_cluster_multiplier`.
* **`band`** — per-band majority routing (both twins). Fields:
  `compute_band_majorities`, `band_edges_mhz`, `band_labels`,
  `min_contributors_per_band`.
* **`gaussian`** — `calibrate_tau_G`-only knobs. Fields:
  `snr_min`, `tau_G_bound_lo`, `tau_G_bound_hi`, `tau_G_seeds`,
  `delta_chi2r_min`, `tau_G_upper_fraction`, `min_contributors`
  (Gaussian-twin pre-condition; **distinct from `aggregation.min_contributors`**
  — kept on its own sub-block to avoid the name collision the
  next-session prompt flagged).
* **`recommendation`** — `recommend_shape`-only knobs. Fields:
  `snr_min`, `tau_bound_lo`, `tau_bound_hi`, `tau_G_seeds`,
  `pure_margin_threshold`. Overlaps with `gaussian` in three
  fields (snr_min / bounds / seeds) by design: the two consumers
  are conceptually independent (the recommendation hook may run
  with a wider or narrower contributor pool than the production
  τ_G calibration), so each carries its own block.

### Resolver

Four layers, matching Stage 5:

```
explicit kwarg > preset / settings > persisted > recommended > hard default
```

The `recommended` slot is reserved but empty for Stage 2b — Stage 2b
is the originator of recommendations, not a consumer. A future
Stage-2 → Stage-2b feeder (SNR-driven `snr_min` from the per-bin
`rms_noise` distribution is the obvious candidate) can land
without API churn.

`preset` and `settings` populate the same layer; passing both
raises `ValueError`, matching the Stage 5 contract.

### HDF5 persistence

```
processing_parameters/
  stage2b_tau/                  (TauCalibrationSettings; nested subgroups)
    @creation_time
    @preset_name                (optional)
    stft/  polish/  aggregation/  band/  gaussian/  recommendation/
```

`STAGE2B_TAU_SETTINGS_PATH = "processing_parameters/stage2b_tau"`.
The Lorentzian and Gaussian twins share one persisted settings
record — they are alternative outputs of the same algorithm under
different shape selections, and the shape itself is not a
`TauCalibrationSettings` field (the shape selector lives on
Stage 5's `StageFitSettings`).

`tau_calibration_settings_present(file_path)` mirrors
`stage_fit_settings_present` as the resolver's persisted-layer
probe.

### Preset YAML

Per § "Preset YAML wrapper migration", packaged presets gain a
top-level `stage2b:` block alongside the renamed `stage5:` block.
The `fit:` → `stage5:` rename is the back-compat shim landing in
this session.

```yaml
stage2b:
  stft:
    n_seg: 10
  polish:
    polish_snr_cap: 9.0
  gaussian:
    snr_min: 20.0
    delta_chi2r_min: 1.0
stage5:
  shape: gaussian
  tau:
    tau_penalty_lambda: 50
```

### Cross-interface parity

* `cli/tau_commands.py`: `calibrate-tau`, `calibrate-tau-G`, and
  (when added) `recommend-shape` each gain `--preset NAME_OR_PATH`.
  Existing per-knob flags lose their argparse hard defaults so the
  resolver picks them up from the preset.
* `Pipeline.calibrate_tau` / `Pipeline.calibrate_tau_G` /
  `Pipeline.recommend_shape`: gain
  `settings: Optional[TauCalibrationSettings] = None` and
  `preset: Optional[str] = None`.
* `api.calibrate_tau` / `api.calibrate_tau_G` / `api.recommend_shape`:
  same.

### Coverage

* `tests/unit/core/test_tau_calibration_settings.py` — empty
  dataclass; resolution-chain precedence per-layer; sub-dataclass
  independence; attrs round-trip; YAML I/O; unknown-key rejection.
* `tests/unit/io/test_tau_calibration_settings_serialization.py`
  — HDF5 round-trip; sparse settings; audit attrs; overwrite;
  subgroup layout.
* `tests/unit/io/test_preset_loading.py` — extended for the
  `stage2b:` block and the `fit:` ↔ `stage5:` legacy-alias path.
* `tests/integration/test_stage2b_settings_propagation.py` —
  intercepts the kernel call and asserts every routed field
  reaches the inner-function kwargs bag.
* `tests/integration/test_cross_interface_consistency.py` and
  `tests/integration/test_stage5_settings_propagation.py` stay
  green.

## Back-compat shims (migration target for research / dev scripts)

This section catalogues every backwards-compatibility shim
introduced (or preserved) by the backfill project. When the
project completes, research and development scripts that depend on
the older interface need migration; the shims also flag candidates
for `DeprecationWarning` in a later release cycle. **Append-only;
do not silently delete entries.**

| # | Shim | Where | Reason | Migration path |
|--:|------|-------|--------|----------------|
| 1 | `load_preset` accepts both `fit:` (legacy) and `stage5:` (new) spellings for the Stage 5 wrapper block | `core/stage_fit_settings.py::load_preset` | The Stage 2b backfill renames the preset wrapper from the original `fit:` to per-stage `stage2b:` / `stage5:` blocks. Pre-existing user presets and research scripts (notably `scratch/gaussian-retune/`, `dev-docs/research/stage5-tau-calibration/`) write `fit:` and must still load. | Rewrite preset YAML to use `stage5:` instead of `fit:`. After the next release cycle adds `DeprecationWarning`, the warning's stacktrace surfaces the call site. |
| 2 | Legacy per-knob kwargs stay on `Pipeline.calibrate_tau` / `Pipeline.calibrate_tau_G` / `Pipeline.recommend_shape` and on the corresponding `api` / impl signatures | `_internal/stage2b_impl.py`, `_internal/stage2b_g_impl.py`, `_internal/shape_recommendation_impl.py`, `pipeline.py`, `api.py` | Match the Stage 5 migration policy: existing call-sites that pass individual kwargs (e.g. `polish_snr_cap=9.0`) keep working; they bundle into an explicit `TauCalibrationSettings` inside the impl and route through the resolver. | Move kwarg payload to a `TauCalibrationSettings(...)` instance or to a YAML preset. `DeprecationWarning` follow-up tracked below. |
| 3 | `DEFAULT_*` constants stay live in `fitting/tau_calibration.py` | `fitting/tau_calibration.py` | The constants are still imported by the kernel functions as their parameter defaults; once every consumer reads from a resolved `TauCalibrationSettings`, the constants become docstring-only. Same status as the Stage 5 `DEFAULT_*` family. | Delete one release after the `DeprecationWarning` for the legacy per-knob kwargs lands. |
| 4 | `compute_band_majorities` dropped from `bool = True` to `Optional[bool] = None` on every Stage 2b layer | `_internal/stage2b_impl.py`, `_internal/stage2b_g_impl.py`, `pipeline.py`, `api.py` | The Stage 5 pattern: a `None` default lets the resolver pick the value up from a preset or persisted layer. The hard default in `band.compute_band_majorities` is `True`, so observable no-kwargs behaviour is unchanged. **Landed in this project.** | None — the migration is internal; user-visible defaults are preserved. |
| 5 | CLI argparse defaults for Stage 2b stay at `None` (already the case) | `cli/tau_commands.py` | Existing per-knob flags already used `type=int/float` with no explicit `default=`, so argparse defaults to `None`. The resolver picks them up unchanged; only `--preset` was added. **Landed in this project.** | None — user-visible defaults are preserved. |
| 6 | Legacy `from_saved_params=True` on `estimate_noise` reads `processing_parameters/noise_estimation` (a separate block from the new `processing_parameters/stage2_noise` the resolver writes) | `_internal/stage2_impl.compute_noise_estimation_impl`, `_internal/stage2_impl.save_noise_parameters_impl`, `cli/noise_commands.py`, `Pipeline.estimate_noise`, `api.estimate_noise` | The cross-interface consistency suite drives a `visualize_noise(save_params=True)` → `estimate_noise(from_saved_params=True)` round-trip via the legacy block. The new resolver-persisted block is the canonical record; the legacy block stays for that contract. `from_saved_params=True` is incompatible with `settings=` / `preset=` (raises `ValueError`). | Drop `from_saved_params=True`; rely on the resolver's persisted layer (a no-kwargs follow-up call now inherits the persisted `stage2_noise` block automatically). |
| 7 | Legacy per-knob kwargs (`skew_target`, `min_bin_fraction`, `smoothing_window_mhz`, `min_noise_fraction`) stay on `estimate_noise` signatures | `_internal/stage2_impl.py`, `pipeline.py`, `api.py`, `cli/noise_commands.py` | Match the Stage 5 / Stage 2b migration policy: existing call-sites that pass individual kwargs keep working; they bundle into an explicit `NoiseSettings` inside the impl. Note the public surface exposes only the four user-tunable knobs that were historically there; the new instrument-tunable knobs (`subdivision_threshold`, `abs_min_bin_size`, `inc`, the three skirt-exclusion knobs) flow through `settings=` / `preset=` only. | Move kwarg payload to a `NoiseSettings(...)` instance or to a YAML preset. |
| 8 | Module-level constants (`SUBDIVISION_THRESHOLD`, `ABS_MIN_BIN_SIZE`, `STRONG_PEAK_SNR`, `SKIRT_EXCLUSION_K`, `MAX_SKIRT_EXCLUSION_MHZ`, `DEFAULT_SMOOTHING_MHZ`) stay live in `preprocessing/noise_estimation.py` | `preprocessing/noise_estimation.py` | They remain the kernel's parameter defaults and the readable canonical source the `NoiseSettings._HARD_DEFAULTS` table mirrors. Once every consumer reads from a resolved `NoiseSettings`, they become docstring-only. Research scripts and docs that name these constants (`dev-docs/research/noise-grid-invariance/report.md`, `dev-docs/research/noise-heuristic-audit/report.md`) continue to work — the constants still exist. | Delete one release after the `DeprecationWarning` for the legacy per-knob kwargs lands. |
| 9 | `estimate_noise_adaptive` kernel signature gains five new keyword-only kwargs (`subdivision_threshold`, `abs_min_bin_size`, `strong_peak_snr`, `skirt_exclusion_k`, `max_skirt_exclusion_mhz`) | `preprocessing/noise_estimation.py::estimate_noise_adaptive`, `_compute_mad_based_bins`, `_exclude_strong_line_skirts` | The new kwargs default to the module-level constants (e.g. `subdivision_threshold=SUBDIVISION_THRESHOLD`), so old callers see identical behaviour. The instrument-tunable constants now flow through `NoiseSettings` end-to-end. **Landed in this project.** | None for old callers; new callers can pass per-knob kwargs or build a `NoiseSettings`. |

## Follow-ups (not part of this project's session work)

- **`DeprecationWarning` on legacy per-knob kwargs and the `fit:`
  preset wrapper.** Lands on the next release cycle per the
  migration plan in [`stage5-fit-settings.md`](stage5-fit-settings.md)
  § Migration notes. Single warning per call-site, not per kwarg,
  to keep the noise floor down.
- **Stage 2 settings (`NoiseSettings`).** Instrument-tunable knobs
  centred on `DEFAULT_SMOOTHING_MHZ` plus the MAD-binning
  parameters. The Stage 2 noise estimator is the canonical
  instrument-tunable surface (see
  [`memory: noise-estimator-mad-shipped`](../../../.claude/projects/-home-kncrabtree-github-ftmwpipeline/memory/noise-estimator-mad-shipped.md)).
- **Stage 3 settings (`PeakDetectionSettings`).** SG window /
  order, matched-filter knobs, the τ-aware gap-pass parameters
  that consume Stage 2b's `τ_maj`.
- **Stage 4 settings (`WindowPlanningSettings`).** Clustering
  edges and the minimum-separation factors.
- **Cross-fixture validation of the shape-aware classifier.** The
  classifier landed against 2638 only; a clean-Lorentzian fixture
  is the generalisation check.
- **Productionising the 3-way recommendation as an auto-run step**
  inside `calibrate_tau` / `calibrate_tau_G`. With the
  shape-aware-classifier landing the per-call cost is fixed
  (`shape='best_of_three'` ~50 s on 2638 vs the ~10 s
  `shape='gaussian'`), so this is a settings-layer decision: should
  `calibrate_tau_G(..., auto_recommend=True)` (a new knob in
  `RecommendationSubSettings`) trigger the verdict for free?
- **Stage 3 / Stage 5-rescue τ consumers** still consume the
  pure-exp `τ_maj` even when the Stage 5 shape is Gaussian. The
  classifier work resolved the upstream; the consumer-side
  question (should the rescue τ be shape-conditioned?) is open.
- **`preset_git_hash` audit attr.** Per
  [`stage5-fit-settings.md`](stage5-fit-settings.md) §
  Follow-ups; deferred until a portable preset-content-hashing
  implementation is justified.

## Provenance

- Architectural template: [`stage5-fit-settings.md`](stage5-fit-settings.md)
  (`StageFitSettings`, four-layer resolver, HDF5 persistence,
  packaged presets, `--preset` CLI).
- Stage 2b knob surface and operating points:
  [`stage2b-tau-calibration.md`](stage2b-tau-calibration.md) +
  [`../research/stage5-tau-calibration/report.md`](../research/stage5-tau-calibration/report.md).
- Original settings-architecture proposal:
  [`../../scratch/settings-architecture-proposal.md`](../../scratch/settings-architecture-proposal.md).
