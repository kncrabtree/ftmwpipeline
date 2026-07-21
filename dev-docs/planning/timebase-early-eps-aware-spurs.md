# Early timebase calibration + eps-aware spur detection (C3)

**Status: implemented.** This document is the implementation overview for
follow-up **C3** (`stage-impl-review-followups.md`): the scope-timebase
self-calibration was re-positioned to run early (right after Stage 1) and, on top
of that, Stage 5 spur detection was made bin-width-correct and eps-aware. The two
parts landed as separate commits (Part 1 is byte-neutral; Part 2 is the C3
feature). Maintainer decision (2026-07-21): running timebase after Stage 1 is
acceptable (pure Stage 0 is not required — see "Why after Stage 1, not Stage 0").

## Motivation

C3 (review #14): Stage 5 spur detection was neither bin-width-aware nor
eps-aware.

- `fitting/spur_detection.py` `DEFAULT_INTEGER_TOL_MHZ = 0.04` was a fixed MHz
  tolerance, ~half a bin only at the reference `T_active ≈ 13 µs`. The
  nearest-bin match used it directly instead of computing the window from the
  actual active-FT bin spacing `Δf = 1/T_active`.
- The match window was not corrected for the digitizer clock scale error `eps`. A
  scale error displaces every measured tone by `eps·f_bb`, so a genuine clock
  spur can fall outside its match window and go undetected — leaving a CW tone in
  the residual to inflate a window's χ².
- `detect_active_ft_spurs` ignored each `LatticePoint.window_mhz` in lattice
  mode, using the scalar tolerance instead of the per-point window the lattice
  already computed.

`eps` is exactly what `fitting/timebase_calibration.py` measures (joint weighted
fit over the Rb-locked clock lattice, parts-in-10⁷ formal precision). The blocker
was ordering: the timebase stage used to run **after** Stage 5, so its `eps` was
not available to the spur sweep. Part 1 removes that blocker.

## Part 1 — timebase runs after Stage 1 (done)

Byte-preserving for the `eps` measurement (same inputs); changes only *when* it
runs.

1. **Dependency declaration** (`file_manager.py`): `timebase_calibration` is now
   `["stage0_fid_data", "stage1_complex_ft"]`. The impl already hard-required
   Stage 1 (`timebase_impl.py` raises `StageDependencyError(["stage1_complex_ft"])`
   without the persisted FT settings); the declaration was under-declaring it.
2. **Orchestration order** (`_internal/run_impl.py`): `timebase` runs immediately
   after the `FT` block and before `noise`. The non-fatal behavior (missing clock
   declaration → skip with a warning, not a run failure) and the `calibrate=False`
   skip are unchanged; the `timebase` result field
   (`"calibrated"`/`"skipped"`/`"not_requested"`) semantics are unchanged.
3. **Consumers**: the report reads the persisted result regardless of order. No
   `STAGE_DEPENDENCIES` edge points into `timebase_calibration`.

### Why after Stage 1, not Stage 0

The active-region bounds are canonically a **Stage 1** product (`start_us` from
start detection, `end_us`/`trim` from the FT settings the user fixed). Stage 0
persists only a *recommended* `start_us` and no canonical `end_us`. Running
timebase at pure Stage 0 would force it onto recommended (not user-canonical)
bounds, shifting the active region the `eps` fit is measured on. Running right
after Stage 1 keeps the `eps` measurement **byte-identical** (same
`start_us`/`end_us`) while making `eps` available to every later stage.

## Part 2 — eps-aware, bin-width-correct spur detection (done)

Three sub-parts, all in `fitting/spur_detection.py` / `clock_lattice.py` and wired
in `_internal/stage5_impl.py`.

### 2a. Bin-width match tolerance

`detect_active_ft_spurs` derives the nearest-bin match window from the active-FT
bin spacing `Δf = median(|diff(freqs)|)`. The tolerance is
`max(integer_tol_mhz, bin_fraction·Δf)` with `bin_fraction = 0.5` (half a bin):
the existing `integer_tol_mhz` acts as an absolute floor, and the bin-width term
makes the window correct at any active length. On long/reference records
(`0.5·Δf ≤ 0.04`) this is byte-neutral; on shorter records it widens the window
that the fixed 0.04 MHz left too tight.

### 2b. Per-point `LatticePoint.window_mhz`

In lattice mode each sweep entry uses `max(point.window_mhz, bin_fraction·Δf)`
instead of the scalar tolerance, so each point's own window (which the lattice
already computed) governs its match. Observable only on irregular/gapped grids
(on a dense uniform grid the bin-width floor dominates).

### 2c. Eps-awareness — a search-anchor **shift**, not a window widening

This is the sub-part where the original design note ("widen each tone's match
window by `eps·f_bb`") was **corrected during implementation**. `eps` is read
recommended-but-not-required in `stage5_impl.py` (only under `spur_cfg.clocks`,
so no-declaration files stay byte-identical) from the persisted
`timebase_calibration`, and passed to `build_spur_set`.

Why a shift and not a widening: `detect_active_ft_spurs` picks the single
active-FT bin nearest each lattice point's **predicted** frequency (`argmin`),
then tests narrowness/SNR at that bin. A scale error moves the measured tone to
`f·(1+eps)`; once that displacement exceeds ~half a bin the tone lands in a
*different* bin, and the predicted bin the detector inspects is empty. Widening
the match *tolerance* does not help — `argmin` still returns the empty prediction
bin. This was verified empirically on a realistic dense grid: window-widening
detected nothing, a directional shift detected the tone.

The fix (in `build_spur_set`, when `eps` is present): shift each lattice point's
search frequency onto the measured position,
`freq_mhz += sideband_sign · eps · f_bb` (with `f_bb =
ClockLattice.baseband_mhz(freq_mhz)`), and widen its window only by the eps
*uncertainty* term `N·|σ_eps|·f_bb` (default `N = 3`; the deterministic
displacement is absorbed by the shift). Absent `eps`, points are untouched —
byte-identical to the pre-eps path. `detect_drift_spurs` inherits the shifted
points; its wide windowed search already tolerated the small displacement.

A recommended (not hard) Stage 5 → `timebase_calibration` edge is documented in
the `stage5_fitting` comment in `file_manager.py` (mirroring the Stage 2b
policy): Stage 5 must still run without a clock declaration.

## Validation

- **Part 1**: 2638 golden byte-identical; timebase unit/fixture tests pass in the
  new position; a `run_pipeline` order test asserts `timebase` precedes `noise`.
- **Part 2 unit tests** (`tests/unit/fitting/`): the bin-width tolerance admits a
  short-record tone the fixed 0.04 window missed; the per-point window governs a
  lattice match on a gapped grid; **eps relocation** gates a spur displaced into a
  *different* bin on a plain dense grid (the case that proves the benefit — and
  that a window widening would fail); `ClockLattice.baseband_mhz` round-trips.
  Cross-interface consistency for the eps read (`tests/integration/`).
- **2638 golden**: byte-identical — 2638's active length and eps keep every tone
  in its predicted bin, so the change is a no-op there (byte-identity ≠ benefit,
  as expected).
- **7-fixture cross-fixture check**: fixtures 363/2638/655 unchanged;
  360/1231/1512/1019 each gain 1–2 genuine on-lattice clock spurs (`640×N (bb)`)
  or near-integer tones the old fixed window missed. Line counts drop only by the
  newly-masked tones; χ²ᵣ medians hold or improve. No regression. (Run via a
  focused before/after comparator on the shipped `run_pipeline` path — the tracked
  `dev-docs/research/stage5-cross-fixture` harness is bitrotted against the current
  API and was not used. Validation ran at parallel workers after an unrelated
  fork-safety crash in the Stage 5 pool was fixed — see the parallel-fit fix
  commit.)

## Provenance

Grew out of C3 in
[`stage-impl-review-followups.md`](stage-impl-review-followups.md). The
timebase-ordering finding, the Stage-0-vs-Stage-1 assessment, and the eps
shift-vs-widen correction are from the 2026-07-21 session.
