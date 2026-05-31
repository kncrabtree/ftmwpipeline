# Stage 3 detection threshold: the SNR corner, and why leakage — not noise — sets it

Two coupled questions drove this study, both unlocked by the switch to the
honest **scatter** Stage 2 noise estimator (which removed the leakage-pedestal σ
inflation that previously compressed SNR under lines):

1. Should the fixed Stage 3 SNR thresholds (internal floor 2.0, promotion 3.0,
   calibrated on 2638 alone) be replaced by a data-driven **corner** found from
   the peak-count-vs-threshold curve, and is that corner the same across the
   ~3-orders-of-magnitude SNR span of the seven fixtures, or must detection
   become SNR-adaptive?
2. Stage 3 scored its internal candidate detections on the *adaptive* estimator
   even though the persisted Stage 2 is now scatter. Should the internal calls
   become scatter too?

The answers turned out to be linked through a single physical fact: in these
line-dense FTMW spectra the "noise" a low detection threshold runs into is not
thermal noise — it is **coherent truncation-leakage structure**, and that is
what both questions are really about.

## 1. The corner is global, not SNR-adaptive

For each fixture, on the canonical production grid (raw `zpf=0` FT, scatter
σ_x), sweep a per-bin SNR threshold `s` and count local maxima of `|X|` whose
`SNR = |X|/σ_x` exceeds `s`. The count `N(s)` falls steeply through a noise/
structure regime, then flattens onto a real-line plateau. The transition — the
**corner** — was located by a Kneedle knee on `log N(s)`.

| fixture | max SNR | knee s\* |
|---|---|---|
| 363 | 2.2e2 | 3.20 |
| 2638 | 4.1e2 | 3.25 |
| 360 | 1.5e3 | 3.40 |
| 1231 | 1.9e3 | 3.20 |
| 1512 | 6.6e3 | 3.15 |
| 1019 | 2.2e4 | 3.05 |
| 655 | 3.0e4 | 3.35 |

The knee sits at **3.0–3.4 across all seven fixtures** spanning ~135× in
max SNR, with no systematic slide (the lowest- and highest-SNR fixtures, 363 and
655, give 3.20 and 3.35). It is stable under the sweep range as well (knee on
`s∈[1,8]` vs `[1.5,7]` moves <0.15). **One global threshold serves all SNR
regimes**; the brief's escalation trigger (regime-specific detection) is *not*
tripped at the threshold level. The current promotion cutoff of 3.0 sits right
at the cross-fixture knee — it was a good value, and is now a principled one
rather than a single-fixture guess. The internal floor 2.0 sits just below the
knee, appropriate as an aggressive candidate floor that the promotion cutoff
then filters.

Driver: `scratch/stage3_benchmark/analyze.py`, `robust.py`.

## 2. The Rayleigh model-crossover method is invalid here

The brief proposed a second corner locator: model the noise false-positive count
as `N_eff·exp(-s²)` (Rayleigh magnitude) and solve for the `s` that meets a
false-positive budget. This **fails on real data**, and the failure is the
finding.

Fitting the line-free exceedance to `N_eff·exp(-b·s²)` gives `b = 0.13–0.38`
across the fixtures — nowhere near the Rayleigh value `b = 1`, and the tail-only
fit (`s ≥ 3`) drives `b` *lower* (≈0.10–0.16), not toward 1. A synthetic
white-complex-Gaussian control run through the identical pipeline recovers
`b ≈ 0.95` (and `maxima/bin = 0.334`, matching the real data), so the method is
sound — **the real-data tail is genuinely ~6× heavier than Rayleigh in s².**

The line-free bins are not thermal: their high-SNR maxima are coherent
truncation-leakage / sidelobe ripple from the surrounding forest of strong
lines (the scatter estimator high-passes the smooth pedestal, but bin-scale
sidelobe ripple survives the high-pass). The scatter σ_x correctly sets the
*median* noise floor (validated independently by the cross-fixture √N collapse
in the noise-snr-scaling work) but does **not** describe the exceedance tail.
Consequently the Rayleigh crossover puts `s*` at 5.5–8 — it would throw away
real lines to suppress false positives that are not Poisson noise but
deterministic leakage. **Use the empirical knee, not the analytic crossover.**

Driver: `scratch/stage3_benchmark/diag_rayleigh.py`, `control_white.py`.

## 3. The estimator mismatch is real but must not be "fixed" by a blind swap

Stage 3 scored internal candidates (primary + gap grids) on `estimate_noise_adaptive`
while the persisted Stage 2 is scatter. Swapping those internal calls to scatter
*alone* increases promoted peaks by +14–25% on the fixtures (2638 754→862, 1512
488→608, 655 2027→2522). Those extra promotions are **100% primary-pass**,
**92–99% inside coherent-leakage regions** (`S_coh`-touched), and sit in the
broader wing/cluster territory rather than hugging line cores.

The reason is structural: the primary pass has **no leakage mask** (only the gap
pass does). Today it is held back from re-detecting strong-line skirts purely by
*accident* — the adaptive estimator runs σ high in pedestal regions, so the
candidate floor `2·σ_adaptive` is incidentally strict there. Make the internal
noise honest (scatter) and that accidental protection vanishes; the primary pass
floods with skirt detections.

So consistency (honest internal noise) is correct, but only if the primary pass
gains real leakage protection at the same time. A blind one-line swap degrades
the promoted list.

Drivers: `scratch/stage3_benchmark/estimator_swap.py`, `swap_leakage.py`.

## 4. The hard gap-mask threshold 8 was orphaned — and is now retired

(Resolved in §5: the hard `S_coh` cutoff is replaced by the continuous
leakage-aware floor on both passes. This section is the diagnosis that motivated
it.)

The gap-pass leakage mask drops detections where the de-ramped coherent-edge
statistic `S_coh = |Σz|/(σ√M)` exceeds 8 (= √M at M=64, i.e. coherent leakage
≥1σ/bin). That value was set by D8 from a *bimodal* `S_coh` distribution over
gap promotions with a genuine-line/sidelobe valley at 6–8 — but that calibration
was on the old apodized `zpf=2` grid and old noise estimator.

On the current production grid (raw FT + scatter), the bimodality is **gone**:
over 8,239 gap candidates on 2638 the `S_coh` distribution is unimodal-rising
(median 11.9), no valley. The band is broadly warm (full-band median `S_coh`
3.41 vs the null 0.89), and **85% of surviving gap-pass promotions sit in the
3<S_coh<8 wing zone** — the gap pass operates inside the wings. Threshold 8 is
no longer justified by its original argument.

However, the threshold is a **modest performance lever**: sweeping the gap mask
8→3 on 2638 moves promoted 754→705, windows 310→293, total window width −6%,
with median and max window width unchanged (the 70 MHz monster window is a dense
strong-line cluster, untouched by the wing peaks). Tightening it is a cleanup,
not the fix for wide windows / fit loops.

Driver: `scratch/stage3_benchmark/scoh_study.py`, `scoh_bimodal.py`,
`threshold_sweep_s34.py`.

## 5. Architecture: one continuous leakage-aware floor, both passes

A literal `S_coh` mask is **destructive**: on 2638, 100% of strong lines
(snr>50) and 95% of medium lines sit at `S_coh>8` — *they* are what generate the
coherence — so a hard mask would delete 482 real promotions including every
strong line. Location cannot discriminate a real cluster line from skirt ripple;
the discriminant is **amplitude**.

The shipped mechanism is therefore a continuous, leakage-aware detection floor,
applied to **both** passes (replacing the primary's lack of a mask *and* the
gap pass's former hard `S_coh` cutoff):

```
thresh(bin) = min_snr·σ(bin) + k · (S_coh(bin)/√M) · σ(bin)
```

The added term `(S_coh/√M)·σ` is the local coherent-leakage amplitude L, scaled
by `k`. A genuine line towers over it (kept); a strong line's skirt ripple *is*
≈L, so it does not clear `k·L` for `k ≥ ~2` (rejected). The strong/medium
exemption is automatic — no SNR gate — because their amplitude dwarfs `k·L`.
M = 64. Paired with switching the two internal noise calls to scatter, this
makes honest internal noise safe.

**The two passes need different `k` because they run on opposite-leakage
spectra.** The primary is Blackman-Harris apodized, which annihilates the
truncation leakage: on 1512 its `S_coh` is ~0.2 across the band (below the
noise-only null of 0.89), spiking only at the rare cluster cores (max ~890). The
window *is* the primary's leakage suppression; the floor is a surgical core
correction. The gap pass is the matched filter — matched to the line shape for
weak-line sensitivity, so it retains the full leakage: `S_coh` ~4–15 typical,
strong wings into the thousands (max ~8200). There the floor carries all the
suppression.

**Calibrating the two `k` (primary = 1, gap = 3).** A cross-fixture sweep
`k ∈ {0..4}` shows catalog recall on the ground-truth fixtures (1512, 655) is
flat across the range — the floor removes false detections, not real lines.

- **Primary `k = 1`**, set by direct visual validation on the sparse high-SNR
  fixture 1019 (its few very strong lines flood the floorless pass: k=0 promotes
  381, k=1 → 93). At k=1 every promoted peak is a real line and the skirt flood
  is gone; k=2 begins clipping real cluster lines (whose primary `S_coh` is
  modest, since the window cleaned their neighbourhood).
- **Gap `k = 3`**, set by direct visual validation on 1512. At gap k=1 (the
  primary's value) the floor sits at the wing level and the gap pass floods —
  541 gap promotions, 4% on catalog, gap `S_coh` median **231**. Raising k lifts
  the floor above the wing: k=2 → 37 (38% catalog), k=3 → 24, k=4 → 18 (61%
  catalog). k=3 is the level at which the wings are excluded and the survivors
  are genuine (catalogued, or clean-region finds at `S_coh < 3`).

Replacing the gap pass's hard cutoff with this floor is a net gain over the
orphaned threshold 8: the hard mask blanket-dropped everything in high-`S_coh`
regions, killing real weak lines sitting on a strong wing; the continuous floor
keeps those that tower above the local leakage (1512 recall 0.455 → ~0.50).

Drivers: `scratch/stage3_benchmark/primary_mask_risk.py`, `k_sweep.py`,
`gap_k_sweep.py`, `plot_1019_k.py`, `plot_1512_gap.py`.

## 6. 2638 validation: no downstream regression

Stage 3→4→5 A/B on 2638 (production raw-`zpf=0` grid), new path (scatter
internal + leakage floor k=2) vs old (adaptive internal, no floor):

| path | promoted | windows | maxW | fitted peaks | χ²ᵣ p50 | p90 | max | n>2 |
|---|---|---|---|---|---|---|---|---|
| NEW (k=1, shipped) | 793 | 311 | 70 | 568 | 1.30 | 2.45 | 81 | 52 |
| NEW (k=2) | 738 | 312 | 70 | 570 | 1.32 | 2.44 | 81 | 54 |
| OLD (adaptive, no floor) | 754 | 310 | 70 | 564 | 1.31 | 2.46 | 81 | 53 |

The new path is statistically indistinguishable from the old on every metric:
internal noise is now honest and consistent with persisted Stage 2, the primary
pass is protected by a real mechanism rather than by adaptive's incidental
pedestal inflation, and Stage 5 fit quality is unchanged. The ~570 fitted peaks
(vs the ~625–799 legacy-grid figures) is a grid effect shared by both paths —
the raw `zpf=0` re-baseline, not a regression.

Driver: `scratch/stage3_benchmark/assess_s5.py`.

## 7. O4: gap-pass detection sits on a flat `zpf ∈ {1,2}` plateau; the `FWHM_bins≥3` proxy points off it

The matched-filter gap pass zero-pads the active region by `_GAP_ACTIVE_ZPF = 2`
so the Lorentzian FWHM lands at ≈ 3 bins (SavGol's operating range), and
`_grid_aware_sg_window` picks an odd `sg_window` covering ~4 FWHM (floor 5).
Both were calibrated on 2638 *at the Stage-1 apodization* `expf_us = 5.0`. The
production path now feeds the matched filter the **Stage 2b data-driven `τ_maj`**
instead, which spans **3.1–9.3 µs** across the seven fixtures. Since the gap pass
FFTs the *active region only*, its native bin spacing is `Δf = 1/T_active`, so

```
FWHM_bins = 2^zpf · T_active / (π · τ),
```

and a fixed `zpf` cannot hold `FWHM_bins` constant when `τ` (and `T_active`)
vary. Measured on the real production grid (instrumented through the genuine
Stage 3 run at the shipped `zpf=2`):

| fixture | shape | τ_maj (µs) | gap Δf (kHz) | FWHM_bins @zpf2 | sg_window | covers FWHM |
|---|---|---|---|---|---|---|
| 363 | gaussian | 9.31 | 23.5 | 1.45 | 7 | 4.8 |
| 360 | gaussian | 7.34 | 23.5 | 1.85 | 7 | 3.8 |
| 1512 | lorentzian | 8.34 | 21.5 | 1.78 | 7 | 3.9 |
| 2638 | gaussian | 5.96 | 19.8 | 2.70 | 11 | 4.1 |
| 1019 | lorentzian | 4.78 | 23.5 | 2.84 | 11 | 3.9 |
| 655 | lorentzian | 3.14 | 25.1 | 4.04 | 17 | 4.2 |
| 1231 | lorentzian | 3.30 | 23.5 | 4.10 | 17 | 4.1 |

`FWHM_bins` falls **below the 3-bin calibration target on 5 of 7 fixtures** (the
wide-τ ones), exactly as the brief anticipated. The proxy says "raise `zpf` to
restore ≥ 3 bins." **The proxy is wrong, and the *direction* it points is the
trap.**

The honest test is the actual quality metric — catalog recall — swept across
`zpf ∈ {0,1,2,3}` (not just up). Two fixtures carry ground truth: 1512 (vinyl
cyanide) and 655 (same molecule, ~120× SNR), both scored against the 328-line
union catalog at 50 kHz tolerance. Recall (`promoted` / `gap` promotions /
`VC-recall`):

| fixture | zpf=0 | zpf=1 | zpf=2 (prod) | zpf=3 | proxy picks |
|---|---|---|---|---|---|
| 1512 | 489 / 16 / **0.372** | 554 / 25 / 0.460 | 563 / 24 / **0.466** | 556 / 6 / 0.433 | zpf 3 |
| 655 | 2306 / 62 / **0.299** | 2351 / 23 / **0.375** | 2316 / 4 / 0.348 | 2309 / 5 / 0.372 | zpf 2 |

Three things fall out, and they reframe the whole question:

1. **`zpf=0` is robustly worst** (recall 0.372 / 0.299, the floor on both). The
   active-region FT is *intrinsically coarse* — at `zpf=0` the line is **under
   one bin wide** (`FWHM_bins = T_active/πτ` = 0.44 on 1512, 1.01 on 655), and a
   2nd-derivative detector cannot resolve a sub-bin peak. So the zero-padding is
   not adding information (it can't) — it is the *necessary interpolation* that
   lifts an under-sampled active-FT line up to where the discrete derivative
   operator can see it. This is why `zpf` cannot simply be turned down to 0.

2. **`zpf=1` and `zpf=2` are statistically indistinguishable — a flat plateau.**
   The two truth fixtures disagree on which is marginally best (1512 favours
   `zpf=2` by 0.006; 655 favours `zpf=1` by 0.027), and the sign even flips with
   the catalog subset scored (on the v=0-only 115-line list, `zpf=1` beats
   `zpf=2` on 1512, 0.548 vs 0.522). Differences of ≤ 0.03 recall sit inside the
   noise of *which* borderline peaks each grid happens to promote. One doubling
   of the active FT already reaches the plateau; the second is a free lateral
   move on it.

3. **`zpf=3` is the start of the falloff** — recall flat-to-down and gap
   promotions collapse (1512: 24 → 6; the dense fixtures fall toward zero gap
   promotions). Zero-padding further drives `_grid_aware_sg_window` to a wider
   window (1512: 7 → 15), and the wider SavGol 2nd-derivative over-smooths the
   weak-line curvature the gap pass exists to catch. **The `FWHM_bins ≥ 3` proxy
   points here — past the plateau, into the over-smoothed falloff.**

So the detection-quality-vs-`zpf` curve is a broad plateau over `{1,2}` that
falls off on *both* sides: under-sampling below, over-smoothing above. The
`sg_window` rule (`K = 4`) is independently sound — it stays **odd, ≥ 5,
covering 3.8–4.8 FWHM on all seven fixtures** at the production grid.

**The result is shape-robust** (i.e. the exp/Lorentzian matched filter is not a
liability on the Gaussian-vote fixtures). `FWHM_bins` tracks τ *magnitude*, not
shape: the three Gaussian-recommendation fixtures (363/360/2638) do not cluster
apart — 1512 is *lorentzian* with τ = 8.3 µs and sits in the same low-`FWHM_bins`
regime as the wide-τ Gaussian ones, while the narrow-τ lorentzians (655/1231)
sit highest. This is expected: the gap pass only *finds positions* — every
amplitude/SNR/classification is re-measured on the unapodized user spectrum in
snap-back — so a filter-shape mismatch costs only a little weak-line sensitivity
(modest, per the Stage-2b gaussian audit) and never biases a reported value.
Matching the *filter shape* to the molecular line is a Stage-5 fitting concern,
already handled there; the detector needs only the data-driven time constant,
which it gets from `τ_maj` (or the Gaussian twin `tau_G_maj` when present).

**Verdict: keep `_GAP_ACTIVE_ZPF = 2` and the `K = 4` SavGol rule unchanged.**
`zpf=2` is the safe incumbent on the plateau; `zpf=1` is its equal and ~0.2 s
cheaper per `detect_peaks` (0.69 vs 0.89 s on 2638; 0.65 vs 0.92 s on 655 —
~10–29 % of an *already sub-second* stage), but that is not a clear enough win to
flip the default: the saving is immaterial end-to-end (Stage 3 runs after Stage
2b's ~minute-scale `calibrate_tau` NLS — the gap-FFT is nowhere near the
bottleneck), and the change is not free. At `zpf=1` gap promotions rise
materially (2638 30 → 85, 1231 16 → 49, 655 4 → 23) because the leakage-floor
`S_coh` is computed over `M` bins whose *physical* width tracks the grid — so the
gap floor `k = 3`, visually calibrated at `zpf=2` to suppress exactly that
skirt-flood, moves off its operating point and would need re-deriving, alongside
re-running the §6 2638 Stage 3→4→5 no-regression check and re-baselining the
pinned tests, all for a recall wash. The actionable finding is the **boundary**:
do *not* let the `FWHM_bins ≥ 3` heuristic push the grid to `zpf ≥ 3`, where it
demonstrably costs recall. No code change; O4 closed. (The adaptive-`zpf` rule
`max(2, ⌈log₂(3π·τ/T_active)⌉)` is derived in the recipe only to show it lands
in the falloff — it is *not* adopted. A deliberate `zpf=1` perf+simplicity flip
remains a possible follow-up if done with the `k`/downstream/test re-validation
above.)

Recipe: `o2_o4_validation.py` (this directory; sweeps `zpf ∈ {0,1,2,3}` and
scores 1512 + 655 against the catalog union) → `data/o2_o4_validation.json`.

## 8. O2: the fixed `10 / 50` SNR tiers generalise across the full SNR span

The detected-peak SNR is `|X|/σ_x` against the scatter σ; the question is
whether the WEAK<10≤MEDIUM<50≤STRONG boundaries separate the population across
the ~3-orders-of-magnitude span, or collapse (655 → all-strong, 363 → all-weak).
Promoted-peak distribution per fixture:

| fixture | max SNR | n prom | p50 | p90 | weak% | medium% | strong% |
|---|---|---|---|---|---|---|---|
| 363 | 222 | 1879 | 8.0 | 53 | 56 | 33 | 11 |
| 2638 | 414 | 776 | 5.5 | 82 | 63 | 18 | 19 |
| 360 | 1509 | 1058 | 7.5 | 90 | 60 | 24 | 16 |
| 1231 | 1894 | 926 | 7.2 | 66 | 61 | 27 | 12 |
| 1512 | 6635 | 563 | 4.4 | 33 | 75 | 17 | 8 |
| 1019 | 21675 | 90 | 26.4 | 345 | 27 | 41 | 32 |
| 655 | 29889 | 2316 | 9.9 | 65 | 50 | 36 | 13 |

**No fixture collapses.** All three tiers stay populated across the entire span:
even 655 (max SNR ~30000) is 50% weak and only 13% strong; even 363 (max 222) is
11% strong. The reason is structural — the distribution is **anchored at the
bottom** by the fixed detection floor (`min_snr = 3.0`, §1) and has a heavy upper
tail whose *length* scales with the brightest line but whose *bulk* stays in the
3–15 band on every fixture (p50 = 4.4–9.9 on six of seven). A bright spectrum
adds lines at the top of the tail; it does not shift the population up. The lone
high-p50 fixture (1019, p50 = 26) is the genuinely **sparse** high-SNR case (90
promotions, mostly real strong lines) — and even it keeps all three tiers
populated rather than collapsing.

This is exactly why **percentile / per-fixture-adaptive boundaries would be
worse**: SNR is already a normalised physical quantity, and "this line is bright
enough to generate truncation leakage and dominate its fit window" is an
*absolute* property of amplitude-over-noise, independent of how many other lines
share the spectrum. A percentile-50 cut would brand half of 1019's real strong
lines "weak" and would move the boundary to ~33 on 1512 vs ~10 on 363 with no
physical meaning. Fixed absolute tiers are the correct design.

**Stage-4 tie-in.** Only the **STRONG** boundary (`t2 = 50`) has a downstream
consumer: `window_planning.py` sets `is_strong = (classification == STRONG)` and
flags a window `HARD` when it contains/abuts a strong line
(`window_planning.py:335,592,610`). The WEAK/MEDIUM boundary (`t1 = 10`) drives
nothing — it is cosmetic labelling for the curation overlay. Across all seven
fixtures the strong tier is populated and non-degenerate (28–304 lines), so the
`t2 = 50` HARD-window trigger fires sensibly in every regime: it neither tags
everything HARD (655: 13% strong) nor starves the trigger (363: 11% strong).

**Verdict: keep `DEFAULT_WEAK_MEDIUM_SNR = 10` and
`DEFAULT_MEDIUM_STRONG_SNR = 50`.** They generalise; the provisional defaults are
now empirically signed off. O2 closed. (The only latent nuance — `t1 = 10` has no
operational consumer — is noted should a future Stage-4/curation feature want to
use the medium tier.)

Recipe: `o2_o4_validation.py` → `data/o2_o4_validation.json`.

## Open items

- ~~Productionize the floor coefficients through `PeakDetectionSettings`.~~
  *(Done — `primary_pass.primary_leakage_floor_k` and
  `gap_pass.gap_leakage_floor_k`.)*
- ~~Re-derive the gap-mask threshold; consider unifying the gap pass onto the
  continuous floor.~~ *(Done — the hard `S_coh` cutoff is retired; the gap pass
  uses the continuous floor at `gap k = 3`; see §5.)*
- ~~Re-baseline the pinned Stage 3/4/5 regression tests onto the production
  grid.~~ *(Done — `baseline_2638_stage2` migrated to the production grid.)*
- ~~Validate `k` on the other fixtures.~~ *(Done — cross-fixture k sweeps +
  1019/1512 visual ground truth fix primary `k = 1`, gap `k = 3`; see §5.)*
- ~~O4 (#10): cross-instrument validation of `_GAP_ACTIVE_ZPF` and the K=4
  SavGol rule.~~ *(Done — §7. Detection sits on a flat `zpf ∈ {1,2}` recall
  plateau (catalog-scored on 1512 + 655); `zpf=0` under-samples, `zpf≥3`
  over-smooths. Fixed `zpf=2` kept (incumbent on-plateau); the `FWHM_bins≥3`
  proxy is falsified — it points into the `zpf≥3` falloff. No code change.)*
- ~~O2 (#10): empirical sign-off of the WEAK/MEDIUM/STRONG SNR tiers.~~
  *(Done — §8. `10 / 50` generalise across the 3-orders-of-magnitude SNR span;
  no code change.)*
- The §1–6 benchmark drivers and `.ftmw` artifacts still live under untracked
  `scratch/` (#14). The O4/O2 close-out (§7–8) is reproduced by the tracked,
  self-building `o2_o4_validation.py` in this directory; the earlier
  corner/leakage-floor drivers remain to be migrated.
