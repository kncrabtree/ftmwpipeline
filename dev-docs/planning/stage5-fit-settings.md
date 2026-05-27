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
  named for A/B), `instrument_bc_2638` (Gaussian + per-band τ routing
  + retuned `tau_penalty_lambda: 50` for the BlackChirp 2638 fixture;
  see the preset YAML's docstring for the sweep evidence that picked
  λ=50).
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

- **Per-band τ₀ for fixed-τ windows.** ``tau0_us`` in
  ``_internal/stage5_impl.py:594-603`` is set once from the band-wide
  ``tau_maj_us``. Per-band routing currently overrides only the prior
  *anchor* (``window_tau_overrides[wid] = (tau_maj_band, sigma_band)``);
  it does not override ``tau0_us`` per window, so weak windows with
  ``fit_tau=False`` (SNR < ``fit_tau_min_snr``) get pinned at the
  band-wide value rather than their band's anchor. Empirically confirmed
  on 2638: every fixed-τ window across all three bands sits at
  τ=8.51874 μs (the band-wide majority), independent of band. Fix is to
  add a per-window ``tau0_us`` override into the per-band routing
  pass. Deferred until Stage 2b's τ_G calibration itself is reassessed
  for Gaussian (see next bullet).
- **Stage 2b τ_G reassessment for Gaussian.** Strong-window medians
  (max-peak-SNR ≥ 20, λ=0) on 2638 land at τ ≈ 6.79 μs -- *below* every
  per-band anchor (low 9.23, mid 8.80, high 7.58). Two candidate
  explanations: the τ_G calibration's deliberate exclusion of the
  strongest STFT contributors may be biasing the estimator high; or
  STFT-contributor τ_G is an intrinsically different estimator from
  window-fit τ_G. Worth re-running the contributor filter at the
  Stage 2b level. Every pre-Stage-5 stage was originally tuned for the
  Lorentzian path; Stage 2b has had the most Gaussian work but is not
  necessarily optimized.
- **Stage 2b shape discriminator.** Compute a recommendation
  (`"lorentzian"` / `"gaussian"`) by comparing the persisted
  Lorentzian and Gaussian τ calibrations; write to
  `stage2b_tau_calibration/.attrs/recommended_shape`. The
  Stage 5 resolver already wires the *recommended* layer; only the
  computation is missing.
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
