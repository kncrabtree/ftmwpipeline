# Leakage detection — Stages 3–4 (D8 resolution)

Status: **resolved.** Tracked in [`../ROADMAP.md`](../ROADMAP.md) as divergence
**D8**. This document is the implementation overview — what the rework changed
and how the result works. Normative requirements remain in the
`*_STRATEGY.md` specs; the Stage 3/4 plan docs
([`stage3-peak-detection.md`](stage3-peak-detection.md),
[`stage4-window-assignment.md`](stage4-window-assignment.md)) describe the
stages as they now stand.

## What D8 was

Both Stage 3's gap-pass leakage mask and Stage 4's `S_coh` edge statistic
mishandled finite-acquisition truncation leakage on real data:

- **Stage 3** — the unwindowed gap pass promoted strong lines' sinc sidelobes
  as weak lines. Its mask (`estimate_leakage_reach`) covered only ±1.4–3 MHz
  of each strong line; the real coherent skirt rings out to ±20–48 MHz, so the
  mask was 7–25× too narrow.
- **Stage 4** — `S_coh = |Σz|/(σ√M)`, a coherent sum over M consecutive bins,
  sat at the noise null over obvious coherent leakage, so every Stage 4
  consumer of it (leakage-touched map, strong-cluster grouping,
  fixed-contributor attachment, `edge_coherence_fail` difficulty) was
  unreliable on real data.

## Root cause

`FID.preprocess` zeroes the FID outside `[start_us, end_us]` but keeps the full
record length; `compute_fft` rfft's the whole zero-padded array. The FFT time
origin is the digitizer `t = 0`, but the active signal occupies `[t₀, t₁]`
with `t₀ = start_us ≠ 0` (2.35 µs on the 2638 fixture). By the rfft shift
identity, a signal starting at sample `s` carries a phase ramp
`exp(−i2π·f_bb·t₀)` on every bin (`f_bb` the baseband frequency). That ramp
makes a strong line's coherent leakage skirt **oscillate**: over an M-bin band
it winds through several full turns, so a coherent sum cancels on genuine
leakage. `S_coh` reads noise-level, and the analytic `1/Δf` reach
under-predicts.

The research reports' synthetic calibrations had silently assumed `t₀ = 0`
(lines built over `[0, T]`), so the bug never showed up in synthetic
verification — only on real data.

## The fix: de-ramp to the active-region turn-on

One shared transform, applied to the complex spectrum just before any
coherence analysis: multiply by `exp(+i2π·f_bb·t₀)` with the baseband
frequency `f_bb = |f − f_probe|`. The absolute value makes it sideband-proof —
it reduces to the correct sign for lower- and upper-sideband data
automatically. This collapses the oscillating skirt back to a smooth,
non-oscillating `1/Δf` envelope that `S_coh` detects correctly. **No new
statistic** — `edge_coherence.py` is unchanged and simply fed de-ramped input.

The persisted canonical spectrum is **not** modified: the de-ramp is an
internal transform owned by the leakage-detection code. The magnitude spectrum
is phase-invariant, so Stage 2 noise and Stage 3 magnitude detection are
unaffected either way.

### Where it lives

`preprocessing/leakage.py`:

- `deramp_to_active_start(freq_mhz, complex_spectrum, probe_freq_mhz,
  start_us)` — the phase-multiply transform.
- `leakage_touched_intervals(freq_mhz, complex_spectrum, rms_noise,
  probe_freq_mhz, start_us, band_m, threshold)` — de-ramps, runs the rolling
  complex-edge coherence, and returns the contiguous above-threshold index
  runs (the **leakage-touched map**). Both stages call it.

## Stage 3 — gap-pass mask

`_internal/stage3_impl.py` builds the de-ramped leakage-touched map on the
unapodized gap spectrum and passes its index runs to
`preprocessing/peak_detection.py:detect_peaks`, which skips gap-pass
candidates that fall inside a touched interval. The windowed primary pass is
unchanged (it is already sidelobe-clean). `estimate_leakage_reach` and the
`tau_us` parameter were retired from Stage 3 — `detect_peaks` and all three
interfaces (CLI / Pipeline / functional API).

Gap-mask threshold: **`T_edge = 8`** (`GAP_MASK_EDGE_THRESHOLD` in
`stage3_impl.py`). Calibrated on 2638: the de-ramped `S_coh` over the gap-pass
promotions is bimodal, with the genuine-weak-line / sidelobe valley at `S_coh
≈ 6–8`. `T_edge = √M` is the level at which a sidelobe's lobe peak clears the
gap pass's ~2σ detection floor. On 2638 the gap pass collapsed 2355 → 1576
promotions — 779 strong-line sidelobes no longer promoted.

## Stage 4 — edge statistic

`preprocessing/window_planning.py` and `_internal/stage4_impl.py` feed the
de-ramped spectrum to all edge-coherence calls; `edge_coherence.py` is
unchanged. `DEFAULT_EDGE_THRESHOLD` is **8** (`= √M` at `M = 64` — flags
coherent leakage of at least ~1σ per bin). Window *extents* are tight
peak-clustering extents (a peak's core plus `min_window_half_width_mhz`,
overlapping extents merged); they are **not** the leakage-touched run. The
de-ramped leakage-touched map instead drives strong-cluster grouping,
fixed-contributor attachment, and difficulty classification —
`window_planning.py` step 2 no longer uses `estimate_leakage_reach` to propose
extents.

## `estimate_leakage_reach` disposition

Demoted, not deleted. The de-ramped `S_coh` map is the leakage-extent
authority for both stages; the analytic `1/Δf` reach under-predicts the
cumulative skirt of multiple strong lines. `estimate_leakage_reach` stays in
`leakage.py`, unused by Stages 3–4 — the finite-T reach formula may still seed
a Stage 5 `τ` prior; final deletion is deferred to Stage 5 scoping.

## Verification (2638 fixture)

- **Null preserved.** The de-ramp is a unit-modulus phase multiply; IID
  complex-Gaussian noise leaves `S_coh` at the ~0.886 null.
- **Strong-line skirts.** Five of the six strongest lines show de-ramped
  `S_coh` 12–18 in the ±2–12 MHz skirt band, 91–100 % above `T_edge = 8`.
- **Leakage is localized.** ~11 % of the spectrum is de-ramped
  leakage-touched, vs 0.8 % for the raw (un-de-ramped) statistic.
- **Stage 4 plan stays sane.** 339 windows, max width ~30 MHz, no
  mega-windows.
- **Regression test.**
  `tests/integration/test_stage3_peak_detection.py::test_gap_pass_does_not_promote_strong_line_sidelobes`.

## Open items handed to Stage 5

- **Window baseline padding.** Stage 4 windows end at their outermost peaks'
  extents with no noise-only margin; Stage 5 fitting will want a configurable
  per-window pad.
- **The 36350/36389 doublet** (SNR 186 + 55, 39 MHz apart) decouples at
  `T_edge = 8` — the de-ramped statistic dips below threshold between the two
  lines. Each is carried into the other's window as a fixed contributor;
  whether to re-couple the pair for fitting is a Stage 5 question.
- **34154 MHz anomaly.** This SNR-172 promoted peak de-ramps to only `S_coh ≈
  3` (damped) / `≈ 1.7` (boxcar), unlike the other SNR ~170–240 lines (~12–27).
  Possibly a blend or a mis-scored detection — revisit in Stage 5.

## Reproducing

The de-ramp diagnostics are ad-hoc scripts under the gitignored `scratch/`:
`gap_mask_calibrate.py` (the Stage 3 `T_edge` sweep), `verify_task5.py` (2638
integration through Stage 4). The research prototypes regenerate the report
figures: `dev-docs/research/peak-detection/prototype.py` and
`dev-docs/research/complex-edge-coherence/prototype.py` — the latter takes a
turn-on offset `t₀` so its synthetic sweep covers the realistic `t₀ ≠ 0` case.
