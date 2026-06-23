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
  start_us)` — the phase-multiply transform. Both stages run the rolling
  complex-edge coherence (`edge_coherence.rolling_coherence`) on its output:
  Stage 4's `window_planning` to build the **leakage-touched map**
  (`above_threshold_intervals`), Stage 3's `_leakage_floor_amp` to raise the
  detection floor (see *Stage 3* below).
- `leakage_touched_intervals(...)` — a convenience wrapper bundling the de-ramp
  + rolling coherence + above-threshold runs; retained in `leakage.py`, though
  the live callers now compute those steps inline.

## Stage 3 — leakage-aware detection floor

The de-ramped statistic feeds Stage 3's **continuous leakage-aware detection
floor**, not a hard mask. Both passes raise their detection floor by
`k · (S_coh / √M) · σ` on the de-ramped spectrum (`rolling_coherence` inside
`_internal/stage3_impl.py:_leakage_floor_amp`): a strong line's coherent skirt
lifts the floor where it rings out, so the matched-filter gap pass no longer
promotes those sidelobes as weak lines, while the strong/cluster lines that
*generate* the coherence are preserved (a hard `S_coh` cutoff would delete
them). This continuous floor superseded D8's original hard gap-mask threshold
(the retired `GAP_MASK_EDGE_THRESHOLD`); the de-ramp that makes the statistic
honest is the durable D8 contribution, shared with Stage 4. Full mechanism +
the two per-pass `k` (`primary_leakage_floor_k = 1`, `gap_leakage_floor_k = 3`)
in [`stage3-peak-detection.md`](stage3-peak-detection.md) § "Leakage-aware
detection floor". `estimate_leakage_reach` and the `tau_us` parameter were
retired from `detect_peaks` and all three interfaces (CLI / Pipeline /
functional API).

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

Deleted. The de-ramped `S_coh` map is the leakage-extent authority for both
stages; the analytic `1/Δf` reach under-predicted the cumulative skirt of
multiple strong lines and never found a Stage 5 use. The finite-T leakage
envelope it was built on survives where Stage 4 still needs it, inlined in
`window_planning._leakage_envelope_fraction` (the weaker-detection-under-skirt
test); the standalone reach estimator and its tests are removed.

## Verification (2638 fixture)

- **Null preserved.** The de-ramp is a unit-modulus phase multiply; IID
  complex-Gaussian noise leaves `S_coh` at the ~0.886 null.
- **Strong-line skirts.** Five of the six strongest lines show de-ramped
  `S_coh` 12–18 in the ±2–12 MHz skirt band, 91–100 % above `T_edge = 8`.
- **Leakage is localized.** The de-ramped statistic concentrates leakage where
  it physically rings out — ~11–18 % of the 2638 spectrum reads
  leakage-touched, vs 0.8 % for the raw (un-de-ramped) statistic, which cancels
  on the oscillating skirt. The exact touched fraction depends on the noise
  estimator (a σ that absorbs skirt power into its floor hides that power from
  `S_coh`); the qualitative localisation is robust across estimators, and the
  current authority is the scatter σ measured on the active FT.
- **Stage 4 plan stays sane.** Under the original Stage 2 σ: 339
  windows, max width ~30 MHz, no mega-windows. Under the post-Stage-2/3-
  rework: 391 windows, max width 65.67 MHz at the 36350/36389
  recoupled cluster (`needs_joint_treatment`), all other widths in the
  4–30 MHz range. See
  [`../research/stage4-poststage23-audit/report.md`](../research/stage4-poststage23-audit/report.md).
- **Regression test.**
  `tests/integration/test_stage3_peak_detection.py::test_gap_pass_does_not_promote_strong_line_sidelobes`.

## Open items handed to Stage 5

- **Window baseline padding.** Stage 4 windows end at their outermost peaks'
  extents with no noise-only margin; Stage 5 fitting will want a configurable
  per-window pad.
- **The 36350/36389 doublet** (SNR 186 + 55, 39 MHz apart) **decoupled**
  under the original Stage 2 σ — the de-ramped statistic dipped below
  threshold between the two lines, and each was carried into the
  other's window as a fixed contributor. Under the post-rework Stage 2
  σ the inter-line skirt is no longer absorbed into σ, so S_coh stays
  above threshold throughout and strong-cluster grouping merges the two
  lines into one primary joint window. Stage 5 honours
  `needs_joint_treatment` via AICc-gated conservative fitting and chose
  9 of the 33 candidate peaks on that window with chi²_r = 3.48; the
  worst-case chi²_r over the whole spectrum simultaneously dropped from
  4183 to 125 at an unrelated window. The "whether to re-couple"
  question is answered by the noise estimator; the open follow-up is
  whether 9 fits is the right model order.
- **34154 MHz anomaly.** This SNR-172 promoted peak de-ramps to only `S_coh ≈
  3` (damped) / `≈ 1.7` (boxcar), unlike the other SNR ~170–240 lines (~12–27).
  Possibly a blend or a mis-scored detection — revisit in Stage 5.

## Reproducing

The de-ramp derivation and the complex-edge calibration regenerate from the
tracked research prototypes: `dev-docs/research/peak-detection/prototype.py` and
`dev-docs/research/complex-edge-coherence/prototype.py` — the latter takes a
turn-on offset `t₀` so its synthetic sweep covers the realistic `t₀ ≠ 0` case.
The integration result is pinned by `tests/integration/`
`test_stage3_peak_detection.py::test_gap_pass_does_not_promote_strong_line_sidelobes`.
