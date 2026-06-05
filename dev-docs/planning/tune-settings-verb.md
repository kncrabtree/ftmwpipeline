# `tune settings`: resolved per-knob values and their provenance

Plan for a settings-inspection verb that answers the question `tune list` does
not: **what value is actually in effect for this experiment right now, and which
layer supplied it.** Charter: GitHub issue #28. Companion to the tuning surface
in [`companion-tuning-tools.md`](companion-tuning-tools.md) (issue #27), which
this completes — it is the resolved-settings half of that surface and the home
for the "how do I persist a chosen value" grammar deferred out of #27.

This is the design + categorization document; nothing here re-derives a specific
default (that is the per-knob audit, issue #3).

## Problem

`tune list` enumerates *what is tunable* (registry: paths, tiers, default sweep
grids). It is silent on *what is resolved* for a given `.ftmw`. A user driving
the tooling on their own instrument needs the second view to decide. The
canonical case: **"will my analysis use the data-detected `start_us`, or
`chirp_end + guard_margin`?"** Today nothing shows the resolved value and the
layer that produced it.

Proposed: `tune settings <file.ftmw> [selector] [--all]`, grammar parallel to
`tune list`, printing **per setting: the resolved value and its provenance
layer**, plus the hard default for reference, and a footer describing how to
change a value at the `.ftmw` or `.yml` level.

## Resolution model (settled) — and the precedence correction it requires

The verb must report the *true* resolver precedence, and that precedence is now
pinned by spec. Per
[`SERIALIZATION_STRATEGY.md`](../SERIALIZATION_STRATEGY.md) §"Settings resolution
and reproducibility":

```
explicit override  >  persisted (.ftmw)  >  preset (.yml)  >  recommended  >  hard default
```

**This is a change.** The current per-stage `resolve()` in `core/*_settings.py`
orders the layers `explicit > preset > persisted > recommended > default` — i.e.
the `.yml` preset **overrides** the persisted `.ftmw` value, the opposite of the
above. That ordering breaks the Principle-4 reproducibility paradigm: a `.ftmw`
must be a self-contained, shareable artifact that reproduces identically from the
file alone, so a `.yml` a recipient happens to have (possibly tuned for a
different instrument) must never silently override a setting the file persists.
The corrected order makes the preset a *seed* for fields the file has not fixed,
never an override of fields it has — and restores consistency with the Stage 1
canonical-settings order (`explicit > persisted > recommended`) that
[`processing-settings-persistence.md`](processing-settings-persistence.md) and
SERIALIZATION_STRATEGY already specify (the preset layer was inserted *above*
persisted only later, in the Stages 2–5 backfill).

Logged as **divergence D11** in [`../ROADMAP.md`](../ROADMAP.md). Resolving it is
a **prerequisite task of this verb** (a verb that faithfully reports an
unreproducible precedence would just document the bug):

- Flip the layer order in every stage `resolve()` (`noise_settings`,
  `tau_calibration_settings`, `peak_detection_settings`,
  `window_planning_settings`, `stage_fit_settings`) to put `persisted` above
  `preset`. The per-field merge is unchanged; only the layer tuple order moves.
- Re-baseline / re-run the settings-propagation and cross-interface integration
  tests; add a regression test that a persisted value wins over a conflicting
  preset for at least one field per stage.
- Reconcile the now-stale precedence statements in
  [`settings-backfill.md`](settings-backfill.md),
  [`stage5-fit-settings.md`](stage5-fit-settings.md), and the matching ROADMAP
  table descriptions (they read `explicit > preset > persisted > …`).

## Source of truth: the settings dataclasses, not the knob registry

`tune list` walks the **knob registry** (`_internal/tuning/registry.py`), which
intentionally omits fields that do not sweep meaningfully in isolation:
`stage1.units_power` (a display/storage rescale — degenerate to sweep),
`stage1.{zpf, expf_us, window_function}` (excluded because the canonical
analysis is a raw, unapodized FT), and `stage0.{band_min_mhz, band_max_mhz}`
(inert unless both set). These are exactly the resolved settings a user needs to
*see and change* — `units_power` is the worked example in the issue thread.

So the verb's source of truth is the **settings dataclasses themselves**
(`dataclasses.fields()` over each stage settings class and its sub-blocks), not
`list_knobs()`. The registry is consulted only to *enrich* a row that happens to
correspond to a registered knob (help text, tier) so the listing can still split
primary/advanced and show the one-line help. Rows with no registered knob still
appear; they carry help from the dataclass field metadata/docstring where
available and are tiered as advanced by default.

This makes the field-enumeration walk the spine of the verb. It must cover the
same stages the resolver covers — Stage 1 FT settings (incl. the four unswept
fields above), Stage 2/2b/3/4/5 settings dataclasses — plus the start-detection
special case below.

## Per-field resolution and provenance

For each field, compute the resolved value by running the (corrected) layer
chain, and record **which layer won**:

- **explicit** — not applicable in the verb context (no per-invocation kwargs);
  always absent. Documented so the column legend is complete.
- **persisted (`.ftmw`)** — load via the stage's
  `io/*_settings_serialization.py::load_*_settings_from_h5`. Returns `None` when
  the stage has not been run / nothing persisted; then the field falls through.
- **preset (`.yml:<name>`)** — see *Preset discovery* below.
- **recommended** — the Stage 2b shape recommendation for Stage 5's `shape`;
  import-time recommended FT params for Stage 1; otherwise `None`.
- **default** — the hard default constant.

Output columns (mirroring `tune list`'s prefix-elided, stage→sub-block table via
`cli/tune_commands.py::_elide_path`):

| column | content |
|---|---|
| knob | dotted settings path, prefix-elided |
| value | resolved value |
| source | winning layer: `.ftmw` / `.yml:<name>` / `recommended` / `default` |
| default | the hard default, for reference |

Filtered by the same path-prefix `selector` (`stage2b`, `stage2b.gaussian`) and
the same primary/advanced tiering (`--all`).

### Start-detection special case (the headline `start_us`)

Start detection (`core/start_detection_settings.py`) is **not** in the resolver
chain — it is a pre-Stage-1 frozen dataclass whose *output* (a recommended
`start_us`) is what matters. The resolved `start_us` lives in the Stage 1 FT
settings (`processing_parameters/ft_processing`). The verb must therefore source
`start_us`'s value from the FT settings and label its provenance from there:
persisted (a user-stamped/detected value in the file) vs recommended (the
`chirp_end + guard_margin` detector output) vs default. This row is the issue's
worked example and should read correctly end-to-end.

## Changing a setting: present the grammar

Alongside the resolved view, show *how to change* a value at either level
(footer/legend, not per-row clutter):

- **`.ftmw` (persisted layer):** re-run the stage with the value set (the stage
  persists it), or a direct persist path. The persisted loaders already exist
  (`io/*_settings_serialization.py::{load,save}_*_settings_*`).
- **`.yml` (preset layer):** write/update the instrument preset under its
  `stageN:` block. `to_yaml_dict()` per settings dataclass already serialises
  chosen values; `load_preset(name_or_path)` resolves bare names against
  `ftmwpipeline/presets/*.yaml` or a path.

This is the home for the #27 deliverable-4 "tune → preset" persistence grammar
(deferred there, folded here per
[`companion-tuning-tools.md`](companion-tuning-tools.md) §"Preset emission").

## Preset discovery — which `.yml` is in effect

For the verb to report `.yml:<name>` provenance it must know which preset
applies. Today there is **no** notion of a preset bound to a file:
`load_preset(name_or_path)` always takes an explicit bare name or path; a stage
run with `--preset` stamps a `preset_name` audit attribute but nothing rebinds
it. Design (smallest correct surface first):

1. **v1 — explicit `--preset <name|path>`.** The verb resolves `.yml` provenance
   only for the preset the user names on the command line; with no `--preset`,
   the preset layer is empty and provenance is `.ftmw` / `recommended` /
   `default` only. Stateless, unambiguous, matches the current model.
2. **Enhancement — surface the stamped `preset_name`.** When a stage persisted a
   `preset_name` audit attr, show it informationally ("last run under preset X")
   even though, under the corrected precedence, persisted values already outrank
   it. This keeps provenance honest without inventing a file-bound active preset.

A file-local "active preset" pointer or a config default is explicitly **out of
scope** here (and largely moot once persisted outranks preset). Note the
interaction: because `.ftmw` now wins, a named preset changes the resolved view
*only* for fields the file has not persisted — which is the correct, reproducible
behavior and worth stating in the verb's help.

## Dual-interface obligations

Per the repo rule, implement once in `_internal` and expose identically on all
three surfaces:

- **Shared core** — a new `_internal/tuning/settings_inspection.py` (or an
  addition to the tuning package) exposing something like
  `resolve_settings_view(file_path, selector=None, *, include_advanced=False,
  preset=None) -> tuple[SettingRow, ...]`, where each `SettingRow` carries
  `path, value, source_layer, hard_default` plus the registry enrichment
  (tier/help). Returning **structured rows** (not printed text) lets Pipeline /
  api hand back data while the CLI formats the table.
- **CLI** — `cli/tune_commands.py::cmd_tune_settings`, reusing `_elide_path` and
  the `cmd_tune_list` layout; new `tune settings` subcommand.
- **Pipeline** — `Pipeline.tune_settings(selector=..., include_advanced=...,
  preset=...)` returning the rows.
- **api** — `tune_settings(file_path, ...)` delegating to `Pipeline`.

## Test plan

- Unit: the field-enumeration walk covers every settings-dataclass field
  including the four unswept Stage 1 fields and `units_power`; provenance is
  correctly attributed for each layer (construct fixtures with a persisted value,
  a preset value, and neither, and assert the winning layer + value).
- Unit: the precedence-flip regression — a persisted `.ftmw` value beats a
  conflicting `--preset` value, per stage.
- Integration: cross-interface consistency — CLI / Pipeline / api return
  identical rows for the 2638 fixture at a representative post-Stage-N state.
- Integration: the `start_us` headline row reads correctly for a file with a
  detected start vs a defaulted one.

## Sequencing

1. **Precedence flip (D11).** Reorder the layers in every stage `resolve()`;
   re-baseline tests; add the persisted-beats-preset regression. Land first — it
   is a spec-conformance fix independent of the verb and unblocks honest
   provenance.
2. **Field-enumeration + resolution core** in `_internal/tuning` returning
   structured rows with provenance.
3. **Presentation + dual-interface** (`tune settings` across CLI / Pipeline /
   api), reusing the `tune list` layout, selector, and tiering.
4. **Change-grammar footer** and the explicit `--preset` provenance path.
5. **Doc reconciliation** of the stale precedence statements (see D11 task list).

## Open items

- Whether `tune settings` subsumes the deferred `tune show` name from #27 (likely
  yes — one resolved-settings verb).
- Help-text source for non-registered fields (dataclass field metadata vs a
  small hand-authored map) — decide during the field-enumeration build.
- The `preset_name` audit-attr surfacing (enhancement 2) can ship after v1.

## Non-goals

- Re-deriving any default (issue #3 / the per-knob audit).
- A preset discovery/registration mechanism beyond explicit `--preset` (a
  file-bound active preset is out of scope and moot under the corrected
  precedence).
- The broader preset-emit UX beyond the change-grammar footer (tracked with #27).
