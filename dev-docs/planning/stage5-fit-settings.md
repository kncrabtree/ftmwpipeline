# Plan: Stage 5 fit settings and preset architecture

Status: **Implemented**. Cross-interface tests on the 2638 fixture green
under the new plumbing; new unit tests cover the dataclass, the
resolution chain, HDF5 round-trip, YAML I/O, and preset loading. End-user
walkthrough at [`docs/source/settings_and_presets.rst`](../../docs/source/settings_and_presets.rst).

## What's in the codebase

- **`core/peak_shape.py`** — `PeakShape` enum (`LORENTZIAN`, `GAUSSIAN`)
  with a `coerce()` classmethod. Relocated from `fitting/peak_model.py`
  so settings dataclasses in `core/` can reference it without a
  `core → fitting` cycle. `fitting/peak_model.py` re-exports it for
  back-compat.
- **`core/stage_fit_settings.py`** — the `StageFitSettings` dataclass
  plus six sub-dataclasses (`TauSubSettings`, `SeederSubSettings`,
  `ConservativeSubSettings`, `PenaltySubSettings`, `RescueSubSettings`,
  `ThawSubSettings`) and a `ShapeSpec` discriminator wrapping
  `PeakShape`. `resolve()` walks four layers (explicit > preset >
  persisted > recommended) and falls back to a nested `_HARD_DEFAULTS`
  dict that mirrors each `DEFAULT_*` constant in
  `fitting/{window_fit,residual_rescue,plan_execution,validation}.py`.
  `to_attrs`/`from_attrs` round-trip the nested dict (None ↔
  `__None__` sentinel); `to_yaml`/`from_yaml`/`load_preset` cover the
  YAML interchange.
- **`io/stage_fit_settings_serialization.py`** — HDF5 persistence at
  `processing_parameters/stage5_fit/` with one subgroup per
  sub-dataclass and the `__None__` sentinel convention shared with
  `io.fid_serialization`. `creation_time` and `preset_name` ride along
  as top-level audit attrs.
  `read_stage2b_recommended_shape` / `write_stage2b_recommended_shape`
  carry the Stage 2b → Stage 5 shape-recommendation contract (currently
  written as `__None__` until a discriminator lands).
- **`_internal/stage5_impl.py`** — `fit_peaks_impl` builds an explicit
  `StageFitSettings` from its legacy kwargs, calls
  `resolve(explicit=kwargs_built, preset=settings_or_loaded_preset,
  persisted=load_stage_fit_settings_from_h5(file_path), recommended=
  ShapeSpec(...))`, reads every downstream parameter off the resolved
  instance, forwards them into the `conservative_kwargs` and
  `rescue_kwargs` dicts that drive `execute_plan` /
  `rescue_and_consolidate`, and stamps the resolved settings back onto
  the file after the fit. Existing kwargs stay on the signature and
  win per field; `per_band_tau` and `shape` lose their hard-coded
  `True` / `"lorentzian"` defaults so the resolver can pick them up
  from a preset or persisted layer (the hard defaults produce
  identical values). Every field with a concrete consumer in
  `fitting/{window_fit,residual_rescue}.py` is forwarded: the
  `tau`, `seeder`, `conservative`, `penalties`, `rescue.cleanup_*`,
  and rescue separation knobs reach the LSQ via this driver, not via
  the inner functions' `DEFAULT_*` fallbacks. The
  `tests/integration/test_stage5_settings_propagation.py` suite
  intercepts the planner call and asserts each routed field reaches
  the kwargs bag — a "phantom field" smoke test that would catch a
  silent drop on any future refactor.
- **`_internal/stage2b_impl.py`** — `save_tau_calibration_impl` stamps
  `recommended_shape = __None__` on `stage2b_tau_calibration/.attrs/`
  as the contract for a future L/G discriminator.
- **`api.fit_peaks` / `Pipeline.fit_peaks`** — gain `settings:
  Optional[StageFitSettings]` and `preset: Optional[str]` kwargs.
  Mutually exclusive (passing both raises `ValueError`); both populate
  the preset layer of `resolve()`.
- **`cli/fitting_commands.py`** — `fit-peaks` gains `--preset
  NAME_OR_PATH`. `--shape` and `--no-per-band-tau` drop to argparse
  `default=None` so the resolver picks them up from a preset; observable
  no-flag behaviour stays the same.
- **`presets/`** — three packaged YAML files: `gaussian_default`
  (clean Gaussian baseline), `lorentzian_legacy` (historical default,
  named for A/B), `instrument_bc_2638` (metadata-only today: every
  field the 2638 calibration would have pinned — shape, per_band_tau,
  τ-prior strength — is now the package-wide hard default, and shape
  is additionally stamped by Stage 2b's auto-recommend pass; the
  preset is kept as a stable name for future 2638-specific knobs).
  Top-level `fit:` wrapper leaves room for a future stage-spanning
  `ft:` block.

## Persistence layout

```
processing_parameters/
  ft_processing/          (FTSettings; flat attrs)
  stage5_fit/             (StageFitSettings; nested subgroups)
    @creation_time
    @preset_name          (optional; bare name or path)
    shape/
      @kind               ("lorentzian" | "gaussian")
    tau/
      @max_decay_factor
      @tau_penalty_lambda
      @per_band_tau
      ...
    seeder/  conservative/  penalties/  rescue/  thaw/  ...
```

Each sub-block is its own HDF5 group so `h5dump -p` can inspect one
block in isolation. Unset fields encode as the `__None__` sentinel
string; this matches Stage 1's flat attrs convention except for the
nesting.

## Precedence chain

Per field, in `resolve()`:

```
explicit kwarg > preset / settings > persisted > recommended > hard default
```

`preset` and `settings` populate the same layer — they're alternative
surfaces (YAML by name/path vs Python dataclass) and are enforced
mutually exclusive in `fit_peaks_impl`. The recommended layer reads
from `stage2b_tau_calibration/.attrs/recommended_shape`; the persisted
layer reads from `processing_parameters/stage5_fit`. A no-arg
`fit_peaks` call on a fresh `.ftmw` resolves to the documented hard
defaults; a no-arg call on a file that's already been fit inherits the
prior fit's settings.

## Coverage

- `tests/unit/core/test_stage_fit_settings.py` (32 tests) — empty
  dataclass, ShapeSpec coercion, resolution-chain precedence
  per-layer, sub-dataclass independence, attrs round-trip,
  YAML I/O, unknown-key rejection.
- `tests/unit/io/test_stage_fit_settings_serialization.py` (12 tests)
  — HDF5 round-trip, sparse settings, Gaussian shape, audit attrs,
  overwrite, subgroup layout. Stage 2b recommended-shape stamp and
  read-back.
- `tests/unit/io/test_preset_loading.py` (11 tests) — packaged
  preset resolution, path resolution, error messages, `fit:` wrapper
  unwrap, explicit-kwarg-beats-preset precedence.
- `tests/integration/test_cross_interface_consistency.py` (existing,
  19 tests) — CLI / Pipeline / api produce identical fits with
  `frequency_mhz` matched at `abs=1e-6`. Green under the new
  plumbing.
- `tests/integration/test_stage5_fitting.py` (existing, 8 tests) —
  green; confirms persistence wiring doesn't perturb the fit.

## Migration notes

The legacy per-knob kwargs on `fit_peaks` still work; they bundle into
an explicit `StageFitSettings` inside `fit_peaks_impl`. No
`DeprecationWarning` is emitted in this work cycle; the warning lands
in the next release cycle. The `DEFAULT_*` constants in
`fitting/{window_fit,residual_rescue,plan_execution,validation}.py` and
`_internal/stage5_impl.py` stay in place as the readable canonical
source the resolver's `_HARD_DEFAULTS` mirrors; they can be deleted
once one release has passed.

The `ftmwpipeline.config` placeholder package is gone; nothing imported
it, the canonical settings dataclasses live in `core/`.

## What this unblocks

- **Stage 5 Gaussian retuning sweeps.** The motivating workstream:
  vary one or two knobs from `instrument_bc_2638.yaml` per sweep
  variant, get each variant's full resolved settings persisted into
  the experiment file. The §2 ("Gaussian acceptance retuning")
  candidates from `scratch/settings-architecture-proposal.md` —
  τ-penalty λ sweep, `max_decay_factor` sweep, F-test threshold
  audit — become preset YAML diffs.
- **Per-instrument calibration.** Each lab/instrument ships its own
  preset YAML alongside fixture data; `--preset path/to/lab.yaml`
  reproduces the recipe.
- **3-way L / G / V hypothesis test.** Voigt support lands as a
  `PeakShape.VOIGT` enum value + a `VoigtParams` sub-dataclass on
  `ShapeSpec`; no API churn elsewhere. Gated on longer-T fixture data
  that can discriminate τ_L from τ_G (current 2638 cannot).

## Follow-ups (not part of this work)

- **Per-band τ₀ for fixed-τ windows. Resolved.**
  ``fitting/plan_execution.py:_walk_windows_in_order`` now derives a
  per-window ``tau0_us`` from ``window_tau_overrides[wid][0]``
  alongside the prior-anchor override, so fixed-τ windows
  (``fit_tau=False``) seed at their band's τ rather than the band-wide
  value. On 2638 the persisted fixed-τ ``tau_us`` now reads
  8.39 / 6.75 / 6.24 μs across low / mid / high (was 6.955 across
  every band).
  ``tests/integration/test_stage5_settings_propagation.py::``
  ``test_per_band_tau_routes_tau0_per_window`` asserts the kwarg
  matches band-local ``τ_maj`` on the captured ``_fit_one_window`` call.
- **Stage 2b τ_G reassessment for Gaussian. Resolved.**
  The per-band-anchor / window-fit gap on 2638 was an estimator
  mismatch, not a contributor-exclusion filter bug: the Voigt fit
  recovers ``τ_G`` as the pure-Gaussian component *after* the
  Lorentzian decay is absorbed into a separate ``τ_L``, while the
  Stage 5 ``shape='gaussian'`` window fit fits a pure-Gaussian model to
  the full envelope. Swapping the per-bin estimator in
  ``extract_tau_G_majority`` from Voigt to pure-Gauss lands the per-
  band anchors within 5–8 % of the λ=0 strong-window-fit medians on
  every band (8.39 vs 8.77; 6.75 vs 6.92; 6.24 vs 5.77). The Voigt
  helpers stay in the module for the future 3-way L/G/V comparator.
  ``instrument_bc_2638.yaml`` keeps ``λ=50`` against the new anchors;
  the runaway-suppression cliff (95 free-τ windows runaway at λ=0 →
  5 at λ=50, ~98.6 %) is unchanged and higher λ degrades the bulk χ²ᵣ
  tail in mid/high bands.
- **Stage 2b shape discriminator -- 3-way hook landed (productionised,
  shape-aware classifier in place).**
  ``fitting/tau_calibration.compute_shape_recommendation`` returns a
  :class:`ShapeRecommendation` (per-bin AICc(exp / gauss / voigt)
  vote, SNR-weighted; the dominant pure shape wins when its margin
  over the other pure shape clears
  ``DEFAULT_SHAPE_RECOMMENDATION_PURE_MARGIN``, otherwise no
  recommendation). End-to-end orchestrator
  ``_internal/shape_recommendation_impl.recommend_shape_impl``;
  user-facing surfaces ``api.recommend_shape`` and
  ``Pipeline.recommend_shape``. Persistence:
  ``write_stage2b_recommended_shape`` now stamps both
  ``stage2b_tau_calibration`` and ``stage2b_tau_G_calibration``
  group attrs when present; ``read_stage2b_recommended_shape``
  falls back from the Lorentzian twin to the Gaussian twin so the
  Stage 5 resolver's *recommended* layer fires regardless of which
  τ calibration ran. The hook runs the STFT classifier in
  ``shape='best_of_three'`` mode so on-line bins enter the cls=3
  contributor pool no matter which candidate model fits them. On
  2638 the verdict reads ``exp 11.6 % / gauss 63.5 % / voigt
  24.9 %`` (n=835 contributors, median ΔAICc(gauss−exp) = −14) →
  recommendation ``"gaussian"`` with the pure-shape margin at
  52 % well clear of the 10 % threshold. The Stage 5 resolver
  picks it up via the integration test
  ``test_recommend_shape_persists_and_feeds_resolver``.
- **Backfill to other stages.** `TauCalibrationSettings`,
  `NoiseSettings`, `PeakDetectionSettings`, `WindowPlanningSettings`
  follow the same pattern. Order: Stage 2b first (shape recommendation
  is its feeder into Stage 5), then Stage 2, Stage 3, Stage 4.
- **`preset_git_hash` audit attr.** Optional reproducibility attr in
  the persisted record (proposal §4e). Deferred because running `git`
  from inside the library is fragile in sandboxed envs / editable
  installs; a robust implementation would hash the preset YAML bytes
  (`sha256` of the file content) rather than the repo's git state.
- **CLI auto-generation for inner knobs.** The current CLI exposes a
  hand-coded subset of knobs (`--max-decay-factor`,
  `--max-residual-rescue-rounds`, …). Walking the sub-dataclass fields
  via metadata (analogous to `FTSettings.cli_field` /
  `cli/_argspec.py`) would surface the rest; not needed for the preset
  workflow that motivated this work, so deferred.
- **`DeprecationWarning` on legacy per-knob kwargs.** Land on the next
  release cycle per the migration plan in
  `scratch/settings-architecture-proposal.md` § D4.

## Provenance

The design recommendations (D1–D7) and step breakdown live in
[`../../scratch/settings-architecture-proposal.md`](../../scratch/settings-architecture-proposal.md);
that document is the authoritative record of why each decision was
made. Implemented over four commits on the `main` branch:

- `379b12f` — dataclasses + tests; PeakShape relocated; dead
  `default_settings.py` deleted.
- `a655fba` — kwargs route through `StageFitSettings`; cross-interface
  bit-identity confirmed.
- `3c665f5` — HDF5 persistence + persisted/recommended layers in
  `resolve()` + Stage 2b stub.
- `4379ff7` — packaged preset YAMLs + `load_preset()` + CLI
  `--preset`.
- `d03a380` — `config/` placeholder package deleted; user-facing
  doc shipped.
