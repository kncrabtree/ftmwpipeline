# Early timebase calibration + eps-aware spur detection (C3)

**Status: planned, deferred to a clean boundary.** This document scopes two
coupled changes that came out of the stage-implementation review follow-up **C3**
(`stage-impl-review-followups.md`): re-positioning the scope-timebase
self-calibration to run early (right after Stage 1) and, on top of that, making
Stage 5 spur detection eps-aware and bin-width-correct. Part 2 depends on Part 1
(Stage 5 must be able to read the measured clock scale error `eps` before it runs
its spur sweep), so both land together.

Maintainer decision (2026-07-21): running timebase after Stage 1 is acceptable
(pure Stage 0 is not required — see "Why after Stage 1, not Stage 0"), and the
re-positioning is deferred until a clean boundary rather than done inside the
review-follow-up pass.

## Motivation

C3 (review #14): Stage 5 spur detection is neither bin-width-aware nor
eps-aware.

- `fitting/spur_detection.py:97` `DEFAULT_INTEGER_TOL_MHZ = 0.04` is a fixed MHz
  tolerance, ~half a bin only at the reference `T_active ≈ 13 µs`. The nearest-bin
  match (`detect_active_ft_spurs`, ~:478) uses it directly instead of computing
  the match window from the actual active-FT bin spacing `1/T_active`.
- The match window is not widened by the digitizer clock scale error `eps`. A
  scale error displaces every measured tone by `eps·f_bb` (reference instrument:
  `eps ≈ 2.13 ppm` → 32–77 kHz across the baseband range), right at or beyond the
  0.04 MHz edge, so a genuine clock spur can fall outside its match window and go
  undetected — leaving a CW tone in the residual to inflate a window's χ².
- `detect_active_ft_spurs` ignores each `LatticePoint.window_mhz` in lattice
  mode, using the scalar tolerance instead of the per-point window the lattice
  already computed.

`eps` is exactly what `fitting/timebase_calibration.py` measures (joint weighted
fit over the Rb-locked clock lattice, parts-in-10⁷ formal precision). The blocker
is ordering: today the timebase stage runs **after** Stage 5, so its `eps` is not
available to the spur sweep.

## The ordering artifact

The pipeline order (`_internal/run_impl.py:111-119`) is
`import → [start detection] → FT → noise → calibrate tau → peaks → windows → fit
→ [timebase] → review → [report]`. Timebase sits after `fit` purely as an
artifact of how the feature evolved; nothing between FT and fitting produces
anything it consumes, and nothing downstream except the final **report** consumes
its output (`STAGE_DEPENDENCIES` has no edge into `timebase_calibration`;
`report_impl` reads it).

Timebase's real data dependency is **Stage 0 (raw FID) + the Stage 1 active-region
bounds** (`start_us`/`end_us`) + a clock declaration:

- `_internal/timebase_impl.py:152-172` reads `start_us`/`end_us` from the
  persisted Stage 1 FT settings and the raw FID from Stage 0, then calls
  `calibrate_timebase_from_fid`.
- The clock declaration resolves explicit > persisted `spur.clocks` >
  import-time recommended (`read_recommended_clock_sources`, available at Stage 0).

### Dependency under-declaration (fix in Part 1)

`file_manager.py:219` registers `"timebase_calibration": ["stage0_fid_data"]`,
but `_read_persisted_ft_settings` (`timebase_impl.py:56-65`) **raises**
`StageDependencyError(["stage1_complex_ft"])` when the Stage 1 settings are
absent. So the impl already hard-requires Stage 1; the registered dependency
under-declares it. Part 1 makes the declaration honest:
`["stage0_fid_data", "stage1_complex_ft"]`.

### Why after Stage 1, not Stage 0

The active-region bounds are canonically a **Stage 1** product (`start_us` from
start detection, `end_us`/`trim` from the FT settings the user fixed). Stage 0
persists only a *recommended* `start_us` (`stage0_impl.py:93-98`, chirp_end +
margin) and no canonical `end_us`. Running timebase at pure Stage 0 would force it
onto recommended (not user-canonical) bounds, shifting the active region the
`eps` fit is measured on — a semantic change to the measurement for no benefit,
since Stage 1 already runs long before Stage 5. Running right after Stage 1 keeps
the `eps` measurement **byte-identical** (same `start_us`/`end_us`) while making
`eps` available to every later stage.

## Part 1 — Re-position timebase to run after Stage 1

Byte-preserving for the `eps` measurement (same inputs); changes only *when* it
runs.

1. **Dependency declaration** (`file_manager.py:219`): `timebase_calibration`
   → `["stage0_fid_data", "stage1_complex_ft"]`. Update the adjacent comment
   (drop "read opportunistically" — it is required).
2. **Orchestration order** (`_internal/run_impl.py`): move `"timebase"` in the
   banner list (:111-119) and its call block (~:163+) to run immediately after
   the `FT` block and before `noise`. Preserve the non-fatal behavior (missing
   clock declaration → skip with warning, not a run failure) and the
   `calibrate=False` skip. The `timebase` result field semantics
   (`"calibrated"`/`"skipped"`/`"not_requested"`) are unchanged.
3. **Consumers**: the report (`report_impl`) already reads the persisted result
   regardless of order; no change. Confirm no test asserts the *position* of
   timebase in `completed_stages` (only membership).
4. **Cross-interface**: the standalone `calibrate_timebase` entry points
   (CLI/Pipeline/API) are unchanged — they already work off Stage 0 + Stage 1 and
   can be called at any point once those exist.

### Part 1 validation

- The 2638 golden fit snapshot is **unaffected** (timebase is not in the fit and
  Stage 5 does not yet read `eps`) — expect byte-identical.
- Timebase's own unit/fixture tests must still pass with the new position.
- A `run_pipeline` order test asserting timebase runs before noise.

## Part 2 — Eps-aware, bin-width-correct spur detection (C3 proper)

Depends on Part 1. Three sub-parts; all byte-sensitive; land and validate
together.

### 2a. Bin-width-unit match tolerance

Replace the fixed `DEFAULT_INTEGER_TOL_MHZ = 0.04` nearest-bin match in
`detect_active_ft_spurs` with a tolerance derived from the active-FT bin spacing
`Δf = 1/T_active` (the `SpurSet` already carries `bin_spacing_mhz`). Keep a knob,
but express its default as a bin fraction (e.g. `0.5·Δf`) so it is correct at any
active length rather than only at `T_active ≈ 13 µs`.

### 2b. Honor `LatticePoint.window_mhz`

In lattice mode, use each `LatticePoint.window_mhz` for its own nearest-bin match
window instead of the scalar `integer_tol_mhz`. `build_clock_lattice` already
computes per-point windows; `detect_active_ft_spurs` currently discards them.

### 2c. Eps-awareness

With Part 1 done, Stage 5 can read the persisted `timebase_calibration.epsilon`
(recommended-but-not-required, mirroring the Stage 2b policy: use it when present,
degrade gracefully when absent). Widen each tone's match window by the
scale-error displacement `eps·f_bb` (optionally `+ N·sigma_epsilon`) so a spur
displaced by the clock error still matches its lattice/integer anchor. Share the
lattice-construction machinery with `timebase_calibration.py` rather than
duplicating it (`build_clock_lattice` / `ClockLattice`).

Add a Stage 5 → `timebase_calibration` recommended edge (not a hard dependency)
and, in `run_pipeline`, ensure timebase precedes fitting (guaranteed by Part 1).

### Design decisions to settle before implementation

- **eps sign/direction**: `f_measured = f_true·(1+eps)`; the match must widen in
  the direction the measured tone moves. Confirm against
  `timebase_calibration.py:4-6` and the spur baseband convention.
- **Knob surface**: whether the widening is `eps·f_bb` alone or
  `eps·f_bb + N·sigma_epsilon`, and the default `N`.
- **Absent-eps degradation**: when no clock declaration exists (timebase
  skipped), fall back to the bin-width tolerance from 2a with no eps term —
  byte-identical to the pre-eps behavior for files without a clock declaration.

### Part 2 validation gap (needs maintainer fixtures)

The 2638 golden **exercises** the clock-lattice path (`examples/blackchirp_data/
2638/clocks.csv` is present), so it gates byte-identity for 2a/2b/2c on 2638. But
it **cannot validate the eps-awareness benefit** — that needs a fixture where a
real clock spur is displaced past the current 0.04 MHz edge by `eps`. The spur
thresholds were calibrated across 7 fixtures (`spur_detection.py` module
docstring); re-validate 2a/2b/2c against that full set, not 2638 alone, before
recording the golden. Any 2638 byte change is a *deliberate* behavior change and
must be justified as an improvement (e.g. a correctly-sized match window), not
accepted blindly.

## Test plan (both parts)

- Part 1: dependency-declaration unit test; `run_pipeline` order test; timebase
  fixture tests re-run in the new position; 2638 golden byte-identical.
- Part 2: unit tests for the bin-width tolerance and per-point window; an
  eps-widening unit test with a synthetic displaced tone; cross-interface
  consistency test for the Stage 5 `eps` read; multi-fixture spur re-validation +
  golden re-baseline (a deliberate, reviewed change).

## Provenance

Grew out of C3 in
[`stage-impl-review-followups.md`](stage-impl-review-followups.md). The
timebase-ordering finding and the Stage-0-vs-Stage-1 assessment are from the
2026-07-21 session; the maintainer chose the after-Stage-1 placement and deferred
implementation to a clean boundary.
