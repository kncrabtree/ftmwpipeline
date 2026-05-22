# Plan: leakage detection rework — Stages 3–4 (D8 resolution)

Status: **in progress — tasks 1–5 done; resume at task 6.** Registered in
[`../ROADMAP.md`](../ROADMAP.md) as divergence **D8**.

Normative requirements remain in the `*_STRATEGY.md` specs; this document is
normative only for the D8 rework it tracks. It supersedes the earlier handoff
that recommended building a new matched-filter detector — validation showed the
fix is much smaller.

**Next session — start here.** The de-ramp helpers, both stage wirings, both
threshold calibrations, and the 2638 integration verification have landed
(tasks 1–5 below are checked off). Resume at **task 6** — the research report
revisions. Task 7 (striking D8, rewriting this doc) follows.

## Summary

Both Stage 3's gap-pass leakage mask and Stage 4's `S_coh` edge statistic
mishandle finite-acquisition truncation leakage on real data. The single root
cause is that the persisted spectrum is a **full-record rfft** in which the
active signal starts at `t₀ = start_us ≠ 0`, so a strong line's coherent
leakage skirt carries a phase ramp `exp(±i2πf·t₀)` and *oscillates*. A coherent
sum (`S_coh`) cancels on it; an unmasked gap pass detects its sidelobes as weak
lines.

The fix is one shared transform: **de-ramp the complex spectrum to the
active-region turn-on** before any coherence analysis. This collapses the
oscillating skirt to a smooth non-oscillating `1/Δf` envelope that `S_coh`
detects correctly. No new statistic is needed — `edge_coherence.py` is unchanged
and is simply fed de-ramped input. The same de-ramped `S_coh` leakage-extent map
serves as Stage 3's gap-pass mask.

## Diagnosis

Validated on the 2638 fixture (`expf_us=5.0, winf=None` — exponential filter,
no window function; the spectrum carries near-full boxcar truncation sidelobes).

**Problem 1 — Stage 3 gap pass promotes sidelobes as weak lines.** The two-pass
design is sound and stays: the windowed primary pass identifies strong peaks
(and is itself sidelobe-clean — verified byte-identical to a manual
`blackmanharris` FT); the unwindowed gap pass recovers genuine weak features.
The bug is the gap-pass *mask*: it is masked within
`preprocessing/leakage.py:estimate_leakage_reach` of each strong line
(≈1.4–3.0 MHz), but the real coherent skirt rings out to ±20–48 MHz. The mask is
**7–25× too narrow**, so the gap pass fires on sidelobes and promotes them.

**Problem 2 — Stage 4 `S_coh` is blind to oscillating leakage.**
`S_coh = |Σ z_k| / (σ√M)` is a coherent sum over M consecutive bins. Real
truncation leakage on the persisted spectrum oscillates (see *Root cause*); over
an M=64 band the sum cancels and `S_coh` sits at/near the noise null over
obvious coherent leakage. Everything in Stage 4 that consumes `S_coh` —
leakage-touched map, strong-cluster grouping, fixed-contributor attachment,
`edge_coherence_fail` difficulty — is therefore unreliable on real data.

## Root cause

`FID.preprocess` zeroes the FID outside `[start_us, end_us]` but keeps the full
record length, and `compute_fft` rfft's the whole (zero-padded) array. The FFT
time origin is the digitizer `t=0`; the active signal occupies `[t₀, t₁]` with
`t₀ = start_us` (2.35 µs on 2638). A truncated signal over `[t₀, t₁]` has rfft
response with **two oscillating edge terms and no non-oscillating component** —
unlike a signal truncated to `[0, T]`, whose response keeps a non-oscillating
`1/(i2πΔf)` term that a coherent sum preserves.

The complex-edge-coherence research report calibrated `S_coh` on a simulator
(`finite_T_response`) that builds every line over `[0, T]`, i.e. silently
assumes `t₀ = 0`. Its synthetic verification therefore passed, and its 2638
"verification" (report §5) ran `S_coh` on the real `t₀≠0` spectrum without the
de-ramp and is **invalid** — see *Research report addendum*.

Measuring the skirt phase slope on 2638 recovers the dominant edge at
`t_edge = −2.350 µs`, exactly `−start_us` — confirmation that the ramp is
precisely the active-region turn-on, and that the turn-on edge dominates (the
turn-off edge is tapered to ~8% by `expf` and is a small oscillating residual).

## The fix: de-ramp to the active-region turn-on

De-ramp factor, derived from the numpy rfft convention (a signal starting at
sample index `s` carries `X[k] = e^{−i2πks/N}·X′[k]`; recover `X′` by
multiplying by `e^{+i2πks/N} = e^{+i2π·f_bb·t₀}`, `f_bb` the baseband
frequency, `t₀ = s·dt = start_us`):

    z_referenced = z · exp(+i·2π · f_bb · t₀)

`f_bb` is the **baseband** frequency. Real (sideband-converted) frequency
relates to it by `f = f_probe ∓ f_bb` (lower / upper sideband), and a single
experiment is single-sideband, so

    f_bb = |f_real − f_probe|

makes the form sideband-proof: it reduces to `exp(−i2πf·t₀)` for lower-sideband
data and `exp(+i2πf·t₀)` for upper-sideband, automatically. The global constant
phase `exp(∓i2π·f_probe·t₀)` is irrelevant — it factors out of `|Σz|`.

`f_probe` and `sideband` live on the `FID`, which both stage impls already load;
`t₀ = start_us` is in the persisted processing parameters.

**Where it lives.** A shared helper in `preprocessing/leakage.py` (already the
cross-stage leakage module):

- `deramp_to_active_start(freq_mhz, complex_spectrum, probe_freq_mhz, sideband,
  start_us) -> np.ndarray` — the transform above.
- `leakage_touched_intervals(freq_mhz, complex_spectrum, rms_noise, probe,
  sideband, start_us, band_m=…, threshold=…) -> list[(lo, hi)]` — convenience
  that de-ramps, calls `edge_coherence.rolling_coherence`, and returns
  `above_threshold_intervals`. Both stages call this.

**Design decision.** The de-ramp is an *internal transform owned by the
leakage-detection code*, applied to the complex spectrum just before coherence
analysis. The persisted canonical spectrum is **not** changed (it would alter
the persisted artifact's phase and widen the blast radius for no benefit; the
magnitude spectrum is phase-invariant, so Stage 2 noise and Stage 3 detection
are unaffected either way). `edge_coherence.py`'s pure statistic functions are
unchanged.

## Validation evidence

Measured on 2638 (`scratch/deramp.py`, `deramp_global.py`, `investigate.py`):

- **Null preserved.** IID complex-Gaussian noise: `S_coh` mean 0.894 → 0.865
  after de-ramp, 0% false positives at `T_edge=3`. The de-ramp is a
  unit-magnitude phase multiply and provably cannot inflate noise.
- **Strong-line skirts (2–12 MHz band), damped 2638:** canonical `S_coh`
  median ~1.8–2.3 (erratic, 20–36% > 3) → de-ramped ~21–27 (**100% > 3**),
  6/6 strong lines.
- **Undamped (boxcar, `expf` off) 2638:** the worst case — both truncation
  edges leak hard — canonical ~1.1–1.5 (~null) → de-ramped ~12–18 (**100% >
  3**). The coherent sum isolates the turn-on edge regardless of the turn-off
  residual.
- **Stage 3 gap mask:** `estimate_leakage_reach` gives ±1.4–3.0 MHz; the
  de-ramped `S_coh` contiguous-above-threshold extent is ±20–48 MHz.

## Stage 4 changes

- `_internal/stage4_impl.py` / `preprocessing/window_planning.py`: feed the
  **de-ramped** spectrum to all edge-coherence calls. The de-ramp is applied
  once where the complex spectrum is obtained (the FID is already loaded there).
- `edge_coherence.py`: unchanged (pure statistic).
- Algorithm step 1 (propose extent): the de-ramped `S_coh` leakage map
  supersedes `estimate_leakage_reach` for both proposal and trimming.
- Re-run the Stage 4 consumers (leakage-touched map, strong-cluster grouping,
  fixed-contributor attachment, difficulty) against the de-ramped statistic;
  they are designed correctly and only need the corrected input.

## Stage 3 changes

- Gap-pass mask: replace the `estimate_leakage_reach`-based mask with the
  de-ramped `S_coh` above-threshold intervals (`leakage_touched_intervals`),
  computed on the canonical unapodized spectrum the gap pass already uses. The
  gap pass runs only in the genuinely leakage-free intervals.
- Threshold: the gap pass detects at ~2σ; the mask must cover where a
  sidelobe's *lobe peak* would clear that floor. Calibration on 2638 (task 4)
  put the genuine-weak-line / sidelobe valley in the de-ramped `S_coh`
  distribution at ~6–8 and **locked `T_edge = 8`** (`= √M`) — *not* the `2√M`
  the initial proposal guessed. That guess conflated the band-averaged `S_coh`
  with the sidelobe lobe peak, which rides ~2× above it; the lobe peak clears
  2σ already at `S_coh ≈ √M`. Stage 3 and Stage 4 share the threshold 8;
  `leakage_touched_intervals`'s default is used.
- The windowed primary pass is unchanged (already sidelobe-clean).
- Cross-check retained: an unwindowed candidate inside a leakage-touched
  interval that has no counterpart in the windowed primary spectrum is a
  sidelobe, not a submerged weak line.

## `estimate_leakage_reach` disposition

Demoted. The de-ramped `S_coh` map is the real leakage extent for both stages;
the analytic `1/Δf` reach under-predicts (it ignores the cumulative skirt of
multiple strong lines) and is no longer the masking/extent authority. Remove its
use from the Stage 3 mask and Stage 4 extent proposal. Final deletion of
`preprocessing/leakage.py:estimate_leakage_reach` is deferred to Stage 5
scoping (the finite-T reach formula may still seed the Stage 5 `τ` prior); until
then it stays, unused by Stages 3–4.

## Window-extent handling + threshold (calibrated)

Calibrated on 2638 (1366 promoted peaks, 133 strong; window count is ~350
regardless of threshold — it is set by peak clustering, not by `T_edge`).
Locked decisions:

- **`M = 64`, `T_edge = 8` for Stage 4.** `T_edge/√M` is the per-bin leakage
  amplitude (in σ) at the detection boundary, so `T_edge = √M = 8` flags
  coherent leakage that is at least noise-level per bin (~11% of 2638
  leakage-touched; per-strong-line touched run ~±20 MHz). `T_edge = 3` (the
  research report's value) flags sub-noise 0.38σ leakage and reads ~57% of the
  spectrum touched — operationally over-sensitive.
- **Thresholds.** `leakage_touched_intervals` takes a `threshold` argument.
  Both stages use 8: Stage 4 because `T_edge = √M` flags ≥1σ-per-bin coherent
  leakage; Stage 3 (task 4 calibration) because the genuine/sidelobe valley in
  the 2638 de-ramped `S_coh` distribution sits at ~6–8. The initial `≈16`
  guess for the Stage 3 mask was overturned — see *Stage 3 changes*.
- **Window extent is *not* the touched run.** A single strong line's touched
  run is ~80–100 MHz wide (its leakage is detectable ±40–50 MHz out); a window
  that wide for one line is wrong. Window extents stay tight — a peak's core
  plus `min_window_half_width_mhz` — and the touched map instead drives
  strong-cluster grouping, fixed-contributor attachment, and difficulty.
  `window_planning.py` step 2 drops the `estimate_leakage_reach` extent
  proposal; `estimate_leakage_reach` becomes unused by Stage 4.
- **The 36350/36389 doublet** (SNR 186 + 55, 39 MHz apart) is *not* forced into
  one joint window. At `T_edge = 8` it decouples — 36389 becomes its own
  window. This is **provisional**: if Stage 5 fits the pair poorly, revisit —
  re-couple via a fixed-contributor edge, or lower `T_edge` for that region so
  the overlap is recovered naturally. The doublet-coupling question genuinely
  belongs to Stage 5.

The locked values update the `edge_coherence.py` default and
`stage4-window-assignment.md`. Stage 4 is not trustworthy until task 3 lands.

## Research report revisions

Both research reports enshrine conclusions the de-ramp overturns; each needs a
substantive revision, not an addendum.

- **`dev-docs/research/peak-detection/report.md` (Stage 3).** §5 concludes the
  closed-form `estimate_leakage_reach` mask is "the only candidate above the
  noise floor" and that no phase-based sidelobe discriminator exists. The
  de-ramped `S_coh` leakage map is exactly such a phase-coherence
  discriminator, and D8 measures the reach mask 7–25× too narrow on real data.
  Revise §5 and the reach-mask conclusion (§6–§7): the gap-pass mask is now the
  de-ramped leakage-touched map; the closed-form reach is demoted to a
  proposal. The cost analysis and the primary-apodization calibration (§3, §6)
  stand.
- **`dev-docs/research/complex-edge-coherence/report.md` (Stage 4).** Redo the
  affected parts. `prototype.py:finite_T_response` builds `[0,T]` lines — add a
  turn-on offset `t₀` and rerun the synthetic sweep so the calibration covers
  the realistic `t₀≠0` case. Regenerate the 2638 figures (05/06) from the
  de-ramped statistic. Rewrite §4 (verification) and §5 (2638). The §3 null
  derivation and calibration stand — the statistic was never wrong, only its
  input.

## Out of scope / tracked separately

- **Window baseline padding.** Stage 4 windows end at their outermost peaks'
  extents with no noise-only margin; Stage 5 fitting needs a configurable
  per-window pad. Independent of the de-ramp; fold into Stage 4 finalization or
  the Stage 5 plan.
- **34154 MHz line anomaly.** This SNR-172 promoted peak de-ramps to only
  ~3.1 (damped) / ~1.7 (boxcar), unlike the other SNR~170–240 lines (~21–27 /
  ~12–18). Possibly a blend or a mis-scored detection. Investigate during
  implementation; not a de-ramp blocker.

## What is committed as WIP

The full Stage 4 implementation is committed (suite green):
`WindowDifficulty`/`FixedContributor`/`FitWindow`/`WindowPlan`,
`preprocessing/edge_coherence.py`, `preprocessing/window_planning.py`,
`_internal/stage4_impl.py`, `io/window_serialization.py`, the three interface
wrappers, `visualization/window_visualization.py`, and its tests.

**Trust boundary:** the Stage 4 data structures, serialization, stage
tracking/invalidation, and interface plumbing are sound and reusable.
`edge_coherence.py` is also sound — the statistic was never wrong, only its
input. The provisional parts are the *coherence input* (needs the de-ramp) and
the `T_edge`/`M` operating point (needs re-calibration).

## Task breakdown

1. [x] `deramp_to_active_start` + `leakage_touched_intervals` in
   `preprocessing/leakage.py` + unit tests (sideband sign both ways; null
   invariance; a synthetic `t₀≠0` line's skirt collapses to non-oscillating).
2. [x] Stage 4: de-ramp the spectrum feeding `edge_coherence` in
   `window_planning.py` / `stage4_impl.py` (and `window_visualization.py`).
   Stage 4 unit tests updated for de-ramped input.
3. [x] Stage 4 calibration: `DEFAULT_EDGE_THRESHOLD` 3 → 8;
   `window_planning.py` step 2 drops the `estimate_leakage_reach` extent
   proposal (uniform tight extents); 2638 plan verified (328 windows, max
   width ~31 MHz, no mega-windows); Stage 4 unit tests and
   `stage4-window-assignment.md` updated.
4. [x] **Stage 3.** Gap-pass mask replaced with `leakage_touched_intervals`
   (`stage3_impl.py` builds the de-ramped leakage map on the unapodized gap
   spectrum; `detect_peaks` masks the gap pass with it). `estimate_leakage_reach`
   and the `tau_us` parameter are retired from Stage 3 — `detect_peaks` and all
   three interfaces (CLI/Pipeline/functional API). Gap-mask threshold
   calibrated and **locked at `T_edge = 8`** on 2638 (`scratch/gap_mask_calibrate.py`
   — the de-ramped `S_coh` valley; the `≈16` proposal was overturned). On 2638
   the gap pass collapsed 2355 → 1576 promotions (779 strong-line sidelobes no
   longer promoted). Stage 3 unit tests updated.
5. [x] 2638 integration verified (`scratch/verify_task5.py`): 5 of the 6
   strongest lines have skirt `S_coh` 12–18 with 91–100 % of bins above
   `T = 8` (the 6th is the pre-tracked 34154 anomaly); only 5.8 % of the
   spectrum is leakage-touched; Stage 4 plan is sane (339 windows, max width
   30.3 MHz, no mega-windows). Added integration regression test
   `test_gap_pass_does_not_promote_strong_line_sidelobes`. Full integration
   suite (97) green, incl. cross-interface consistency. Also corrected the
   stale `edge_threshold` default (3.0 → 8.0) in the Stage 4 `assign_windows`
   docstrings (`api.py`, `pipeline.py`) — a task-3 leftover.
6. [ ] **Resume here.** Research report revisions — `peak-detection/report.md` §5 + reach-mask
   conclusion (Stage 3); `complex-edge-coherence/` redo the synthetic sweep with
   a `t₀` parameter and regenerate the 2638 figures (Stage 4). See *Research
   report revisions*.
7. [ ] Strike D8 from `../ROADMAP.md`; update the Stage 3/4 plan-doc status
   rows; rewrite this document as an implementation overview.

## Test plan

- **Unit:** `deramp_to_active_start` — lower/upper sideband sign; de-ramp of
  IID noise leaves `S_coh` distribution unchanged; a synthetic line truncated
  to `[t₀, t₁]` with `t₀≠0` has an oscillating skirt that the de-ramp collapses
  to a non-oscillating envelope. `leakage_touched_intervals` against a known
  synthetic leakage map.
- **Stage 3:** gap pass masked by the de-ramped map promotes no sidelobes;
  genuine submerged weak lines in leakage-free gaps are still recovered.
- **Stage 4:** edge-coherence consumers produce trustworthy leakage-touched
  regions on de-ramped 2638; invariants (disjoint fit windows, acyclic DAG)
  still hold. (The 36350/36389 pair decouples at `T_edge = 8` by design — see
  *Window-extent handling + threshold*.)
- **Cross-interface:** identical results from CLI / Pipeline / functional API
  after the rework.
- **Real data:** 2638 damped and boxcar — the strong-line and null evidence
  above as regression checks.

## Reproducing the evidence

```bash
# fixture through Stage 3 (canonical settings)
conda run -n ftmwpipeline-dev python -c "
import ftmwpipeline.api as ftmw
ftmw.import_data('scratch/exp_2638.ftmw', source='examples/blackchirp_data/2638/')
ftmw.compute_ft('scratch/exp_2638.ftmw', zpf=1, expf_us=5.0, trim=(26500, 40000))
ftmw.estimate_noise('scratch/exp_2638.ftmw')
ftmw.detect_peaks('scratch/exp_2638.ftmw')"
```

The de-ramp diagnostics are ad-hoc scripts under the gitignored `scratch/`
(`deramp.py` — single-line phase-slope + S_coh recovery; `deramp_global.py` —
global + multi-region; `investigate.py` — null check, damped/boxcar table,
Stage 3 mask comparison; `calibrate.py` — the `T_edge` sweep that fixed
`T_edge = 8` and is the starting point for the task-4 gap-mask threshold). The
boxcar (undamped) spectrum is built in-script via
`FID.preprocess(expf_us=None)` because the API's `expf_us=None` resolves to the
recommended default rather than "no filter".
