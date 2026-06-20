# Plan: settings-surface finalization — remove legacy kwargs, unify the knob declaration

Status: **Implemented.** Endgame of the settings backfill
([`settings-backfill.md`](settings-backfill.md)): retire the legacy per-knob
keyword arguments now that the resolver, persistence, presets, and the
`settings`/`scan` surface are all live, and collapse the per-knob declaration
onto a single source — the settings-dataclass field.

This is the last API-surface task before user documentation and codebase
cleanup. It is deliberately sequenced before docs: documenting an API that
offers two ways to pass every knob (loose kwargs *and* `settings=`/`preset=`)
is wasted effort and confuses the reader. Collapse to one way first.

Registered in [`../ROADMAP.md`](../ROADMAP.md). Normative requirements remain
in the `*_STRATEGY.md` specs; this document is normative only for the work it
tracks.

## Implementation status

| Stage | Field metadata | Kwargs removed | CLI generated | Registry from field |
|------:|:--------------:|:--------------:|:-------------:|:-------------------:|
| 1 (template) | yes (`cli_field`) | n/a (already clean) | yes | n/a |
| 2 | **done** | **done** | **done** | **done** |
| 2b | **done** | **done** | **done** | **done** |
| 3 | **done** | **done** | **done** | **done** |
| 4 | **done** | **done** | **done** | **done** |
| 5 | **done** | **done** | **done** | **done** |
| 0 (outlier) | n/a | **done** | n/a | n/a |

Stage 0 was the lightest: its impl (`detect_start_time_impl`) and CLI
(`start_commands.py`, which already builds a `StartDetectionSettings` from its
flags) were settings-only already. Only the `api`/`Pipeline` wrappers carried
the four per-knob kwargs (`sweep_max_us` / `step_us` / `guard_margin_us` /
`floor_factor`); those were removed, leaving `band` (a convenience tuple folded
onto the bundle's band fields), `stamp`, and `settings=`. `StartDetectionSettings`
stays a flat, frozen, concrete-default bundle — it is *not* on the `None`-sentinel
resolver/preset model (no persisted-settings layer; the stage stamps a
recommended `start_us`, not a settings record), so it has no `knob_field`
metadata, no generated flags, and no settings-vs-persisted precedence. The CLI
keeps its hand-rolled flags.

Stage 5 kept three explicit args rather than folding them into a generated
flag: `shape` (the lineshape selector / Stage 2b twin chooser) and the
`tau_maj_override_us` / `sigma_tau_override_us` pair (a cross-stage τ override,
not a fit knob). They overlay onto a deep copy of the passed `settings=`
explicit layer without mutating the caller's object, so `fit_peaks(preset=…,
shape="gaussian")` and `fit_peaks(settings=…, shape=…)` both compose correctly.
The `rescue.max_rounds` / `rescue.snr_threshold` fields carry explicit `flag=`
to preserve their historical `--max-residual-rescue-rounds` /
`--rescue-snr-threshold` spellings; `rescue.prominence_threshold` had no flag and
stays settings-only. The fit is deterministic, so the refit is pure plumbing —
the `fit show`/cross-interface equivalence suite (CLI == Pipeline == api) is the
acceptance guard.

**Deprecation machinery retired.** With the last warning-emitting stage
finalized, `_internal/deprecation.py` (`warn_legacy_kwargs` / `warn_legacy_flag`)
and `tests/integration/test_deprecation_warnings.py` are deleted — nothing calls
them. Shims #2/#7/#10/#13/#15/#16 are resolved as removed. (Stage 0 never emitted
the warning, so its outstanding removal is independent of this deletion.)

The Stage 2 pass established the reusable machinery the remaining stages inherit:

- `core/knob_metadata.py` — `knob_field(...)` (the stage-agnostic declaration)
  plus `knob_meta` / `iter_knob_fields` / `field_knob_meta` accessors.
  `core/settings.py::cli_field` now delegates to it, so `FTSettings` and every
  later stage share one metadata schema (`{"knob": {...}, "cli": {...}}`).
- `cli/_argspec.py` — `add_settings_args` / `settings_from_namespace` are now
  nested-aware (descend one sub-block level) and emit a flag only for fields
  tagged `cli=True`. Flat fields keep their bare argparse `dest` (so the FT
  command's `args.trim` still works); sub-block fields are namespaced
  (`"subblock.field"`) to avoid leaf-name collisions.
- `registry.py` — a stage's `KnobSpec` descriptors (`help` / `tier` /
  `inst_sensitivity` / `default_grid`) are read from the field via
  `field_knob_meta`; the registry keeps only behavior (`run` / `metric` /
  `plot`) and the structural `path` / `stage` / `requires`. A unit test asserts
  the registry echoes the field (single source, no drift).

Stage 2b validated the nested path and the sub-block collision rule the plan
flagged: `min_contributors` lives in both `aggregation` and `gaussian`, and the
historical `tau run` exposed a single `--min-contributors` flag whose target
depended on the `--gaussian` twin switch. Resolution: tag only
`aggregation.min_contributors` `cli=True` (the shared flag); the
`cmd_calibrate_tau_G` handler routes that value into `gaussian.min_contributors`
before calling the impl. The generated flag set then matches the retired
`tau run` surface exactly (13 flags). The `scan` registry's `see_also` pointers
stay literals on the `KnobSpec` (navigational, not a field descriptor); only
help/tier/inst-sensitivity/grid move to the field.

One realized refinement vs the plan's "no flag rename": the tri-state
`region_aware` flag is generated with `BooleanOptionalAction`, so it gains
`--region-aware` *alongside* the legacy `--no-region-aware` (a superset, not a
rename). The deprecation machinery is untouched except that Stage 2's cases were
removed from `test_deprecation_warnings.py` (the api-chain proof moved to
`detect_peaks`); the file and `_internal/deprecation.py` are deleted when the
last stage is finalized.

## Motivation and the "still a good idea?" verdict

The backfill kept the legacy per-knob kwargs alive as a back-compat shim
(16 shims catalogued in [`settings-backfill.md`](settings-backfill.md)
§"Back-compat shims") so existing call sites would not break when the resolver
landed. Every impl already fires `warn_legacy_kwargs(...)`
(`_internal/deprecation.py`), and `tests/integration/test_deprecation_warnings.py`
is the canary enumerating the per-stage legacy lists. The staging Follow-up #1
described is in place; this task is its endgame.

**Verdict: still a good idea, and now is the right time.** Reasons:

- The removal loses nothing functionally. The `settings`/`scan` registry plus
  `settings set <stage>.<knob> <value>` plus YAML presets plus
  `settings=Dataclass(...)` are a complete superset of every legacy kwarg —
  every deleted parameter remains reachable.
- This is a pre-1.0, single-consumer research package; the only callers of the
  legacy form are in-repo (tests, a couple of research scripts, the CLI), so
  the breaking change is cheap and fully under our control.
- Two parallel ways to pass knobs is exactly the ambiguity user docs should not
  have to explain.

The one real caveat is CLI ergonomics: the per-knob flags (`--min-snr`, etc.)
are genuinely convenient and must be **preserved**, not deleted — but routed
through a settings object instead of loose kwargs. That is the Stage 1 pattern
(`core/settings.py::cli_field` + `cli/_argspec.py`) generalized to every stage,
which is also the "data-driven declaration" half of this task.

## Scope decisions (taken)

1. **Removal reaches the impl layer (full removal).** Strip the legacy per-knob
   kwargs from `api.py`, `pipeline.py`, *and* the `_internal/stage*_impl.py`
   signatures. The CLI binds to the impls, so this is the variant that forces
   the CLI rework — which is wanted, because the CLI flags become data-driven in
   the same pass. Only `settings=`/`preset=` (and genuine non-knob args) remain
   on every stage entry point.
2. **Unify the knob declaration onto the dataclass field.** The settings-dataclass
   field's metadata becomes the single source for a knob's *declaration*
   (help text, CLI flag/type, sweep grid, tier, instrument-sensitivity). The
   CLI flag, the `scan` registry's descriptive columns, and the resolver default
   all derive from it. See §"Unified field metadata" for the boundary: the
   `scan` engine's *behavioral* closures stay in `registry.py`.

## Current state (what exists, what changes)

Settings dataclasses (the field is the structural source of truth already):

| Stage | Dataclass | Shape | CLI today |
|------:|-----------|-------|-----------|
| 0 | `StartDetectionSettings` | flat, **frozen, concrete defaults** (outlier) | hand-rolled flags → impl kwargs |
| 1 | `FTSettings` | flat, `None`-sentinel, **`cli_field` metadata** | **generated** (`_argspec.py`) |
| 2 | `NoiseSettings` | flat, `None`-sentinel | hand-rolled flags → impl kwargs |
| 2b | `TauCalibrationSettings` | nested (6 sub-blocks) | hand-rolled flags → impl kwargs |
| 3 | `PeakDetectionSettings` | nested (4 sub-blocks) | hand-rolled flags → impl kwargs |
| 4 | `WindowPlanningSettings` | nested (4 sub-blocks) | hand-rolled flags → impl kwargs |
| 5 | `StageFitSettings` | nested (10 sub-blocks) | hand-rolled flags → impl kwargs |

Only Stage 1 fulfills the data-driven ideal today. Every other stage declares
each knob in up to three places: the dataclass field (structure), the
hand-rolled argparse flag + hand-built kwargs dict in `cli/*_commands.py`, and a
`registry.py` `KnobSpec` (for `scan`/`settings`). This task collapses the first
two and de-duplicates the descriptive half of the third.

### Legacy kwargs to remove (the deletion targets)

All default `None` (except Stage 0's outlier shape), all map to dataclass fields:

- `detect_start_time` (Stage 0): `sweep_max_us`, `step_us`, `guard_margin_us`,
  `floor_factor`. Keep `band` (runtime band override) and `stamp` (side-effect
  control) and `settings=`. **No `preset=`** — Stage 0 is outside the preset
  system. Note: `detect_start_time` may not currently fire `warn_legacy_kwargs`;
  confirm and treat consistently.
- `estimate_noise` (Stage 2): `window_mhz`, `pedestal_mhz`, `line_k`, `n_iter`,
  `region_aware`, `smoothing_mhz`, `smoothing_percentile`, `convolve_mhz`. Keep
  `from_saved_params` (legacy block reader — separate shim #6, decide
  independently) and `settings=`/`preset=`.
- `calibrate_tau` (2b): `n_seg`, `t_sigma`, `tau_max_us`, `rss_gate_factor`,
  `sigma_time`, `min_contributors`, `sigma_tau_fraction_max`,
  `bimodality_dominant_fraction`, `compute_band_majorities`,
  `min_contributors_per_band`.
- `calibrate_tau_G` (2b): the `calibrate_tau` set plus `snr_min`,
  `tau_G_bound_lo`, `tau_G_bound_hi`, `tau_G_seeds`, `delta_chi2r_min`,
  `tau_G_upper_fraction`.
- `recommend_shape` (2b): `n_seg`, `t_sigma`, `tau_max_us`, `rss_gate_factor`,
  `sigma_time`, `snr_min`, `tau_bound_lo`, `tau_bound_hi`, `tau_G_seeds`,
  `pure_margin_threshold`.
- `detect_peaks` (3): `min_snr`, `weak_medium_snr`, `medium_strong_snr`,
  `sg_window`, `sg_order`, `primary_window`, `min_exclusion_mhz`, `run_gap_pass`.
- `assign_windows` (4): `edge_m`, `trim_m`, `edge_threshold`,
  `max_window_width_mhz`, `min_freeze_snr`, `min_window_half_width_mhz`,
  `magnitude_attachment_threshold`, `tau_us`, `max_peaks_per_window`,
  `max_window_width_points`, `min_window_half_width_points`.
- `fit_peaks` (5): `tau0_us`, `fit_tau`, `max_decay_factor`,
  `residual_edge_threshold`, `residual_edge_m`, `max_thaw_rounds`,
  `max_replan_rounds`, `max_residual_rescue_rounds`, `rescue_snr_threshold`,
  `rescue_prominence_threshold`, `per_band_tau`, plus the three special cases
  below.

`compute_ft` (Stage 1) is already clean — no legacy kwargs, `FTSettings`-driven.

### `fit_peaks` special cases (decide explicitly)

- `shape` — a real `StageFitSettings.shape` field but also the most-used
  one-shot convenience (`fit_peaks(shape="gaussian")` appears in tests and
  research). **Recommendation: keep `shape=` as a first-class arg** (it selects
  the lineshape and drives twin selection; it is not an instrument knob). It can
  still populate `settings.shape` internally.
- `tau_maj_override_us` / `sigma_tau_override_us` — a paired atomic override of
  the Stage 2b τ calibration, not plain `StageFitSettings` fields. They are an
  A/B-testing escape hatch. **Recommendation: keep the pair as explicit args**
  (they cross a stage boundary; folding them into `StageFitSettings` would
  misrepresent them as fit knobs). Document as advanced.

## Resolution-layer semantics of `settings=` vs `preset=`

Removing the per-knob kwargs forced a decision the backfill had deferred. The
resolver order is `explicit > persisted(.ftmw) > preset(.yml) > recommended >
default` (D11). Originally the per-knob kwargs filled the **explicit** layer
(override the file) and `settings=` filled the **preset** layer (seed; the file
wins). With the kwargs gone, a `settings=` left at the preset layer would lose
to anything persisted — and because the first run of a stage persists *all*
resolved fields, that meant `noise run --window-mhz 120` (the CLI builds a
`settings=` bundle) was silently ignored on an already-run file.

**Decision: a passed `settings=` bundle is the explicit override** (fills the
explicit layer, outranks persisted), matching the retired per-knob kwargs and
the documented D11 order; **`preset=` (a `.yml`) stays at the preset layer**
(seeds only unfixed fields; a persisted `.ftmw` outranks it, preserving the
shared-file reproducibility contract). `settings=` and `preset=` populate
different layers and may be combined in one call (explicit wins per field, the
preset seeds the rest). Each migrated impl routes `explicit=settings`, `preset=load(preset)`;
a per-stage propagation test asserts `settings=` beats a persisted value, and
the existing tests still assert `preset=`/persisted-inherit precedence. Applied
to Stages 2, 2b, and 3.

## Design: unified field metadata

Generalize `core/settings.py::cli_field` into a stage-agnostic `knob_field`
declared once per knob, carrying both the CLI and the `scan` descriptive
metadata. The field default stays `None` (the unset sentinel); concrete values
come only from the resolver, never the dataclass default.

```python
def knob_field(
    *,
    help: str,                       # one-line physical meaning (CLI + scan)
    cli: bool = False,               # opt-in: gets a generated CLI flag
    flag: str | None = None,         # explicit long flag (collision/legacy name)
    argtype: Callable | None = None, # argparse type= (coercion in metadata)
    metavar: str | None = None,
    is_flag: bool = False,           # tri-state bool -> BooleanOptionalAction
    tier: str = "advanced",          # "primary" | "advanced" (scan list view)
    inst_sensitivity: str = "N",     # "Y" | "N" | "maybe" (scan/audit)
    grid: tuple | None = None,       # scan default sweep values
) -> Any: ...
```

Two boundaries make this tractable:

1. **CLI flags are opt-in per field (`cli=True`).** Today's CLI exposes a
   curated subset of knobs per stage — the historically-public ones — not every
   field. Keep that: only fields tagged `cli=True` get a generated flag. The
   rest stay reachable via `settings set` / preset / `settings=`. This also
   sidesteps the nested-name collision problem (e.g. `min_contributors` lives in
   both the `aggregation` and `gaussian` sub-blocks of
   `TauCalibrationSettings`): a collision is only a problem if both want a flag,
   and the disambiguating `flag=` is available when it is.

2. **The `scan` engine keeps its behavior; the registry reads descriptions from
   the field.** A `KnobSpec` mixes static description (help, tier,
   `inst_sensitivity`, `default_grid`, `requires`, `see_also`) with per-knob
   *behavior* (`run`/`metric`/`plot`/`recommend`/`prepare`/`select_hint`
   closures). The closures genuinely belong in `registry.py` and stay there. The
   static description moves to the field; `registry.py` builds each `KnobSpec`
   by looking up the field metadata for the dotted path and supplying only the
   behavioral closures. Net: help text / tier / grid are declared once (on the
   field) and the registry stops restating them.

### Data-driven CLI generation (nested-aware)

`cli/_argspec.py` today handles a *flat* dataclass (`FTSettings`). Generalize:

- `add_settings_args(parser, cls)` walks `cls` recursively: for a top-level
  flat dataclass, iterate fields; for a nested settings dataclass, descend one
  level into each sub-block (max nesting is two: `stageX.subblock.field`). Emit a
  flag only for fields with `cli=True` metadata.
- `settings_from_namespace(args, cls)` reconstructs the (possibly nested)
  settings instance: read each `cli=True` field off the namespace, place it in
  the right sub-block, leave everything else at the `None` sentinel so the
  resolver fills it. Flag→field mapping carries the sub-block path so
  reconstruction is unambiguous.

Each `cli/*_commands.py` then drops its hand-built `params` dict and instead
calls `settings = settings_from_namespace(args, <StageSettings>)` →
`<stage>_impl(file_path=..., settings=settings, preset=args.preset)`. The
argparse flag surface is registered by `add_settings_args` rather than spelled
out per command.

## Work breakdown

Per stage (0, 2, 2b, 3, 4, 5 — Stage 1 is the template, already done):

1. **Field metadata.** Convert the dataclass fields to `knob_field(...)`,
   tagging the historically-public knobs `cli=True` (preserve current flag
   names via `flag=`), and porting help/tier/`inst_sensitivity`/grid from the
   matching `registry.py` `KnobSpec`. (Stage 0's frozen/concrete-default shape:
   decide whether to keep it frozen — it is outside the resolver — and expose
   only the four historically-public flags; it likely keeps its own small path.)
2. **Impl signature.** Delete the legacy per-knob params from
   `stage*_impl.py`; keep `settings`/`preset` (and genuine non-knob args). Remove
   the now-empty `warn_legacy_kwargs(...)` call and the legacy→settings bundling
   block. The resolver entry stays.
3. **`pipeline.py` + `api.py`.** Delete the mirrored legacy params and the
   pass-through; keep `settings`/`preset` and the documented non-knob args.
   Update docstrings (drop the per-knob parameter lists, point at
   `settings`/`preset` and `settings show <stage>`).
4. **CLI.** Replace the hand-rolled flag block + `params` dict with
   `add_settings_args` / `settings_from_namespace`. Verify the generated flag
   names match the retired hand-rolled ones (no user-visible flag rename) or
   record any deliberate rename.
5. **`registry.py`.** Repoint each `KnobSpec`'s descriptive fields at the
   field metadata (single source); keep the behavioral closures.

Cross-cutting:

- **Deprecation machinery.** Once every impl stops calling it,
  `_internal/deprecation.py` (`warn_legacy_kwargs` / `warn_legacy_flag`) and
  `tests/integration/test_deprecation_warnings.py` are dead. Delete both (the
  warnings tested a surface that no longer exists). Resolve back-compat shims
  #2/#7/#10/#13/#15 (legacy kwargs) as **removed**; decide #6
  (`from_saved_params`) and #1 (`fit:` preset alias) independently — both are
  separate from the per-knob kwargs and can outlive this task.
- **Constants.** Shims #3/#8/#11/#14 (the `DEFAULT_*` module constants) stay —
  they remain the kernel parameter defaults and the readable source the
  `_HARD_DEFAULTS` tables mirror. No change.
- **`workflows.py:90`.** `pipe.estimate_noise(**(noise_params or {}))` forwards
  an opaque dict; audit/migrate its contract to `settings=`.
- **In-repo callers.** ~8 test call sites (dominated by
  `detect_peaks(min_snr=...)` and `fit_peaks(shape=...)`); migrate to
  `settings=`/`preset=` (or keep `shape=` per the decision above). Research
  scripts are already migrated except a couple of `shape=` uses (fine if `shape`
  stays). `dev-docs/research/` Category-B kernel-direct scripts are out of scope
  (they call kernels, not the public API).
- **Strategy docs.** Amend `API_STRATEGY.md` and `CLI_STRATEGY.md` to state that
  per-stage knobs are passed *only* via `settings=`/`preset=` (Python) and
  generated flags / `settings set` (CLI); the field metadata is the single knob
  declaration. Note in `settings-backfill.md` §"Back-compat shims" which shims
  are now retired.

## Test plan

- **Settings propagation suites** (`test_stage{2,2b,3,4,5}_settings_propagation.py`)
  stay green: every routed field still reaches its kernel — now via
  `settings=` only. Drop the assertions that drove the legacy-kwarg path.
- **Cross-interface consistency** (`test_cross_interface_consistency.py`): CLI /
  Pipeline / api still produce identical results, now that the CLI builds a
  settings object instead of passing kwargs.
- **New CLI-generation tests**: `add_settings_args` emits the expected flags for
  a nested dataclass (only `cli=True` fields); `settings_from_namespace`
  round-trips a parsed namespace into the correct nested settings; a `--flag`
  with no value leaves the field at the `None` sentinel (resolver fills it).
- **Flag-parity guard**: assert the generated flag set per stage equals the
  retired hand-rolled set (catch accidental renames).
- **Registry/field consistency**: a test asserting every `KnobSpec` dotted path
  resolves to a real dataclass field and pulls its help/tier from the field
  (no drift between registry and field).
- Delete `test_deprecation_warnings.py`.
- Full non-slow suite green; mypy-clean on new code (the metadata helper is
  fully typed).

## Migration (in-repo only — no external users)

- Python: `f(path, min_snr=4.0)` → `f(path, settings=PeakDetectionSettings(
  promotion=PromotionSubSettings(min_snr=4.0)))` or
  `f(path, preset="name")`. `shape=` and the τ-override pair survive as args.
- CLI: unchanged for the user (the same `--min-snr` flag, now generated).
  Knobs without a flag: `settings set <stage>.<path> <value>` then run, or a
  preset.

## Risks / ordering

- **Order:** do Stage 2 first (flat, `None`-sentinel — the simplest non-template
  case) to validate the `knob_field` + nested-aware `_argspec` generalization,
  then the nested stages 2b/3/4/5, then Stage 0 (frozen outlier — decide its
  treatment last). One stage per commit, each gated on its propagation +
  cross-interface suite.
- **Flag collisions** in nested dataclasses are handled by `cli=True` being
  opt-in plus explicit `flag=`; verify none of the curated public knobs collide
  before tagging.
- **Stage 0 outlier**: `StartDetectionSettings` is frozen with concrete
  defaults and no `preset`. It does not fit the `None`-sentinel resolver shape;
  keep its surface small (four flags) and do not force it into the preset
  system as part of this task.
- **Byte-identical gate** where applicable: the fit (Stage 5) is deterministic,
  so a removed-kwarg refactor must leave the 7-fixture fit tables byte-identical
  (same acceptance bar as the perf work).

## Provenance

- Backfill coordination + shim catalogue:
  [`settings-backfill.md`](settings-backfill.md).
- Stage 1 data-driven CLI template: `core/settings.py::cli_field`,
  `cli/_argspec.py`, [`processing-settings-persistence.md`](processing-settings-persistence.md).
- `settings`/`scan` surface and the knob registry:
  [`tune-settings-verb.md`](tune-settings-verb.md),
  [`companion-tuning-tools.md`](companion-tuning-tools.md),
  [`instrument-tunable-knobs.md`](instrument-tunable-knobs.md).
- Resolution order (D11): persisted(.ftmw) > preset(.yml), pinned in
  `SERIALIZATION_STRATEGY.md`.
