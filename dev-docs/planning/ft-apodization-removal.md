# FT apodization removal (expf / winf / zpf)

**Status: Plan (execution deferred).** Drafted for later execution; no code
changed yet.

## Decision

Explicit user apodization of the canonical FT is deprecated and to be removed.
The early-stage processing knobs `expf_us` (exponential apodization),
`window_function` / `winf` (FID window), and `zpf` (zero-pad factor) are removed
from the user-facing settings surface and the canonical FT path. This is a
deliberate change from the initial vision: the statistical foundations of the
pipeline (per-bin scatter noise authority, the finite-T `h_T` line-shape fit,
the leakage-wing baseline) make user apodization the wrong tool — apodization
trades resolution and biases the line shape, and zero-padding interpolates bins
and corrupts the Stage 2/5 noise and χ² statistics. The robust per-window fit is
the intended alternative.

The canonical FT becomes unconditionally **unapodized, un-windowed, un-padded**
(`zpf = 0`, `expf_us = None`, `window_function = None`). Fixtures built with the
legacy `standard_ft_params` (`zpf=2`, `expf_us=5.0`) are outdated and rebuilt.

### What is removed vs. retained

Removed (apodization / interpolation of the canonical record):
- `expf_us` — exponential apodization time constant.
- `window_function` / `winf` — FID window function.
- `zpf` — zero-pad factor (canonical FT runs at native length).

Retained (not apodization; statistically sound):
- `start_us` / `end_us` — define the active region (data selection, not weighting).
- `rdc` — DC/mean removal (a baseline correction, not a window).
- `units_power` — display amplitude scaling.
- `trim` — analysis-band restriction (data selection).

Explicitly **out of scope** (these stay):
- `fit show --apodize` windowed *display* view — a diagnostic comparison
  against conventional symmetric-window results; it is display-only and never
  touches the canonical record. See
  [`fit-show / windowed view`](../../src/ftmwpipeline/visualization/fit_detail.py).
- Stage 3's *internal* zero-padding for sub-bin peak-position finding — a
  position-only refinement on a throwaway grid, not the canonical FT. Verify at
  execution that its zero-pad path is independent of the removed `zpf` setting.

## Cross-stage couplings to resolve first

These are the non-mechanical parts; resolve each deliberately.

1. **Stage 5 τ₀ default is tied to `expf_us`.** `fit_peaks_impl` defaults the
   per-window starting decay `tau0_us` to the Stage 1 `expf_us` when set, else
   `T_active / 3`. Removing `expf_us` means τ₀ always falls back to
   `T_active / 3` (or, preferably, the Stage 2b `τ_maj` when present). Decide the
   replacement default and confirm it does not regress the fit. (Stage 2b τ
   anchoring already supersedes the apodization anchor on calibrated runs.)
2. **`compute_active_ft(expf_us=...)`.** Stage 5 fits on
   `compute_active_ft(expf_us=expf_us)`. With `expf_us` gone the active FT is
   unapodized — which already matches the canonical noise grid
   (`build_active_grid_with_noise`, `expf=None`). This *resolves* the
   render-domain undershoot noted in the fit-show work (fitted τ becomes the
   intrinsic τ; the `fit show` model then sits on the data). Drop the `expf_us`
   parameter from `compute_active_ft` or pin it to `None`.
3. **Persisted legacy `.ftmw` files.** Existing files carry `expf_us` / `winf` /
   `zpf` in `ft_processing`. Decide open-time behavior: ignore-with-warning
   (recompute the canonical FT unapodized) vs. hard error. Recommended:
   ignore-with-warning and recompute, so old files remain openable but are
   re-derived on the canonical grid.

## Surface to change

Re-enumerate precisely at execution start (`grep -rn "expf_us\|window_function\|
winf\|zpf"`); the categories:

- **Settings core** — `core/settings.py` `FTSettings`: drop the three fields,
  their `cli_field` flags (`--expf_us`, `--zpf`, window flag), hard defaults
  (`zpf` default), `to_preprocess_kwargs`, and HDF5 (de)serialization keys.
- **FT compute / preprocess** — `_internal/stage1_impl.py` (`compute_ft`),
  `core/data_structures.py` (`FID.preprocess` / `FIDProcessingParameters`),
  `utils/signal_processing.py`, `_internal/shared_utils.py`: stop applying the
  window / exp decay / zero-pad; run native-length unapodized.
- **Active FT** — `fitting/active_ft.py`, `_internal/active_ft_support.py`,
  `_internal/stage5_impl.py`: remove the `expf_us` parameter / pin to `None`;
  resolve the τ₀ default (coupling 1).
- **Resolution chain & persistence** — `file_manager.py`,
  `io/fid_serialization.py`, `io/peak_serialization.py`: legacy-key handling
  (coupling 3).
- **Knob / scan surface** — `_internal/tuning/registry.py`,
  `settings_inspection.py`, `settings_mutation.py`: remove the three knobs from
  the `scan` surface and `settings show/set`.
- **CLI** — `cli/ft_commands.py` (`ft run` flags), `cli/settings_commands.py`,
  `cli/utils.py`, `cli/main.py`: drop the flags; emit a clear error if a removed
  flag is passed (don't silently ignore).
- **Presets** — `src/ftmwpipeline/presets/`: strip any preset fields setting the
  three knobs.
- **Docs / examples** — `CLAUDE.md` and `README` 2638 example (currently
  `zpf=2`, `expf_us=5.0`), `STATUS.md`, the relevant `*_STRATEGY.md` (record the
  vision change), and the D7 settings-resolution notes in `ROADMAP.md`.

## Fixture & test cascade (the main cost)

Rebuilding the canonical FT unapodized shifts the noise σ, peak list, window
plan, and fit results for every fixture — a wide assertion cascade.

- **`standard_ft_params`** (`tests/integration/conftest.py`) and every baseline
  built from it (`baseline_2638_stage1..5`, the `_small` and trio variants)
  move to `zpf=0`, `expf_us=None`, `window_function=None`. The frequency trim
  (`26500–40000`) is retained.
- **Re-baseline value-specific assertions.** Tests that hardcode apodized-regime
  numbers (noise σ magnitudes, peak/window counts, χ²ᵣ, the SNR-corner
  thresholds, the 655/1512 validation numbers) must be re-derived against the
  unapodized fixtures. Audit `test_stage2_*`, `test_stage3_*`, `test_stage4_*`,
  `test_stage5_*`, the cross-fixture validation suite, and the start-time tests.
- **Cross-interface tests** are structural (identity across CLI/Pipeline/API)
  and should survive unchanged once fixtures rebuild.
- **`fit show` tests** — the windowed-view `test_boxcar_model_at_right_frequency`
  asserts position only; on unapodized fixtures the amplitude will also match,
  so the assertion can be tightened back to model≈data peak height if desired.
- Watch the suite runtime budget (full suite ~8–9 min; unapodized `zpf=0` FTs
  are smaller, so fits may speed up — re-measure).

## Approach

1. Resolve the three couplings (τ₀ default, active-FT `expf_us`, legacy-file
   handling) as explicit decisions.
2. Remove the knobs from `FTSettings` + the FT/preprocess path; force the
   canonical FT unapodized/native-length.
3. Strip the knobs from the scan surface, CLI flags, presets, and serialization
   (with legacy-key tolerance per coupling 3).
4. Rebuild the test fixtures unapodized; re-baseline the value-specific
   assertions stage by stage.
5. Update `CLAUDE.md` / `README` / `STATUS.md` / strategy specs and the
   ROADMAP D-log (record the vision change and resolution).
6. Run the full suite; confirm cross-interface identity and re-measure runtime.

## Open questions

- τ₀ default after `expf_us` removal: `T_active/3`, or require Stage 2b `τ_maj`?
- Legacy `.ftmw` on open: ignore-with-warning + recompute (recommended) vs. error?
- Do any presets or the `instrument_bc_2638` preset rely on these knobs in a way
  that needs a migration note?
- Should removed CLI flags hard-error (recommended) or warn-and-ignore for one
  release?
