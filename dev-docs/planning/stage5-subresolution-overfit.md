# Stage 5 sub-resolution overfit discriminant

Tracks the fix for GitHub issue #13: residual rescue (and, by extension, the
blend-aware seeder and the merge cleanup) can add a duplicate line co-located
with a strong peak that is statistically *supported* — it absorbs the residual
the single-shape model leaves behind — yet is not a real line. The pair sits
**below the Fourier resolution limit** of the active FT, so no finite-T line
shape can distinguish it from a single feature.

## The discriminating invariant

The right scale for "too close to be two real lines" is the active-FT
resolution element

```
1/T_active = 1 / acquisition_us   (MHz)
```

where `T_active` is the *unpadded* active-FT acquisition length (the persisted
spectrum's alpha-padded grid is an oversampled interpolation grid and is **not**
the resolution). Two frequencies closer than `1/T_active` are fundamentally
unresolvable, so a genuine doublet must be at least ~1 element apart.

On the 2638 gaussian / `rescue_prominence=1.5` fixture
(`dev-docs/fixtures/2638-gaussian-rescue1p5/`, `1/T_active ≈ 79.05 kHz`) the
closest-pair separation in resolution-element units cleanly separates the
hand-labelled overfits from the good tight blends:

- **Overfits** (collapse): w012, w022, w040, w069, w071, w091, w143, w281 —
  closest pair 0.80–1.07 elements, amplitude ratio 4–10:1.
- **Good tight blends** (preserve): w020, w075, w159, w189, w218, w303 —
  closest pair 1.08–1.73 elements, amplitude ratio 1–4:1.

The FWHM-normalized version of the same separations overlaps (overfits
0.53–0.67 FWHM, controls 0.63–1.05 FWHM), because FWHM depends on the
per-window decay `tau` and can fall *below* the resolution limit on narrow
features. Hence the gate must reference `1/T_active`, not FWHM.

## Root cause (the loophole)

The seeder / merge / rescue minimum pair separation was
`min_pair_separation_factor · FWHM = 0.5 · FWHM`. For the flagged windows
`0.5 · FWHM ≈ 58 kHz ≈ 0.73` resolution elements — *below* the Fourier limit —
so the FWHM-only floor licensed sub-resolution pairs. Every overfit sat in the
gap between `0.5 · FWHM` and `1 · (1/T_active)`.

## The fix

A resolution-referenced floor on the minimum allowed pair separation:

```
min_pair_separation = max(min_pair_separation_factor · FWHM,
                          k · (1/T_active))
```

with `k = min_pair_separation_resolution_factor` (new knob, default `1.0`).

One knob drives the floor at three sites, "no two modeled lines closer than
`k` resolution elements unless the data statistically force it":

1. **Residual rescue locality rejection** (`attempt_residual_rescue`, the
   cheapest source). A detector candidate within the floor of an existing
   fitted peak is dropped before it ever reaches the rescue's inner fit — the
   residual that close to an established line is shape-error / leakage, not a
   missed line. (Previously the rejection radius was one *persisted-grid* bin,
   which is far below the resolution element.) Spur-offset rejection keeps the
   grid-bin radius — spurs are single-bin tones.
2. **Blend-aware seeder collapse check** (`_blend_aware_seed`). A K≥2
   escalation whose fitted peaks land within the floor of each other is
   rejected.
3. **Merge cleanup structural tier** (`merge_close_peaks_cleanup`). Both tier
   thresholds carry the floor, so a sub-resolution pair that slips through is
   collapsed unconditionally (Tier 1) rather than handed to the AICc gate,
   which — correctly, given a mis-specified single-shape line — prefers keeping
   the extra peak.

Shared helper: `window_fit._effective_min_pair_separation`.

### Amplitude-ratio tier (the supra-resolution absorber)

Some overfit absorbers sit *just above* the resolution floor (separation ~1.0–1.5
elements), so the floor does not reach them and AICc supports them (they soak
real residual). These are rescue-parked weak peaks beside a strong line with a
large amplitude ratio (8–16:1 on 2638). A third merge tier in
`merge_close_peaks_cleanup` handles them: in a band out to
`overfit_amp_ratio_band` resolution elements (default `1.5`), a pair whose
larger/smaller amplitude ratio clears `overfit_amp_ratio_threshold` (default
`6.0`) is collapsed unconditionally; a balanced pair (ratio below the threshold)
in the same band is a genuine close doublet and is preserved. Both knobs live in
`RescueSubSettings` (the merge knobs' home). This catches w281 (1.26 elements,
15.6:1, no baseline) and w143 (a rescue-added absorber the leakage-wing baseline
refit later drifts sub-resolution) at rescue time, before either reaches the
final fit.

## Validation (2638 gaussian, rescue_prominence=1.5)

`scratch/issue13_validate.py` rebuilds 2638 through Stage 4 (unapodized FT +
Stage 2b τ calibration) and re-fits the full feature on (defaults) vs off
(resolution floor `k=0` **and** the amplitude-ratio tier disabled — i.e.
pre-issue-#13 main), comparing the fitted peak count inside each fixture overfit
/ control freq band on **current code** (spur masking + leakage-wing baseline
default-on, both of which postdate the fixture's labelling — re-mapped by
`freq_range_mhz` per the fixture README).

Result: **8/8 study overfits collapsed, 6/6 controls unchanged**, total fitted
peaks 622 → 604. Every hand-labelled overfit loses exactly its flagged
absorber(s) (w069 9→8, the rest down to their intended 1–2 lines); every
hand-labelled control is peak-count-identical. The resolution floor collapses
the sub-resolution pairs; the amplitude-ratio tier collapses the
supra-resolution absorbers (w281 at 1.26 elements / 15.6:1; w143's rescue-added
absorber, caught at rescue time before the baseline refit drifts it).

The ~10 further peaks removed beyond the 8 study windows (622 → 604) are
sub-resolution / high-amp-ratio pairs elsewhere in the 380-window spectrum, the
same class — not individually re-labelled, but the designated control set (the
regression guard) is preserved byte-count-identical, and the discriminant is
principled. Cross-fixture confirmation is calibration debt.

## Follow-ups / calibration debt

- **Cross-fixture `k` and amp-ratio calibration.** `k = 1.0`,
  `overfit_amp_ratio_band = 1.5`, `overfit_amp_ratio_threshold = 6.0` are
  first-cut values calibrated on 2638 only (on 2638 the overfit absorbers run
  8–16:1 and the closest real control pair sits at 1.08 elements with ratio
  ~1.2, so the band/threshold separate the classes with comfortable margin).
- **Baseline-refit resolution floor (latent).** The leakage-wing baseline refit
  (`plan_execution._apply_baseline_to_outcome`, sequenced last) re-optimizes
  established lines with no separation constraint and no subsequent merge, so it
  *can* drift a pair sub-resolution (this is what happened to w143 before the
  amp-ratio tier caught the absorber upstream at rescue time). Applying the same
  resolution-floored merge inside the baseline refit would close this for cases
  the upstream tiers miss; deferred (needs care with the baseline term in the
  (K−1) merge refit).

These belong with the broader Stage 5 ε / lineshape calibration work tracked in
[`stage5-cross-fixture-validation.md`](stage5-cross-fixture-validation.md)
(issue #13 levers #1/#3 and #4). Per the Stage 5 penalty-tuning debt, re-tune
when applying to other instruments.

## Interface surface

Three new knobs, all riding the standard four-layer resolver, persisted in the
canonical `stage5_fit` HDF5 subgroups + YAML preset interchange like every other
Stage 5 knob, with no new CLI flag (these sub-knobs are preset/settings-driven):

- `ConservativeSubSettings.min_pair_separation_resolution_factor` (hard default
  `1.0`) → `conservative_kwargs` → `conservative_fit` /
  `attempt_residual_rescue` / `merge_close_peaks_cleanup`.
- `RescueSubSettings.overfit_amp_ratio_band` (`1.5`) and
  `overfit_amp_ratio_threshold` (`6.0`) → `rescue_kwargs` →
  `rescue_and_consolidate` → `merge_close_peaks_cleanup`.

A propagation test asserts each of the three reaches the planner in the correct
kwargs bag (`conservative` vs `rescue`).
