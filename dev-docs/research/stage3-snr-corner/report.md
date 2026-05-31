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
- O4 (#10): cross-instrument validation of `_GAP_ACTIVE_ZPF` and the K=4 SavGol
  rule remains open (separate from the leakage-floor coefficients).
- The benchmark drivers and `.ftmw` artifacts referenced here live under
  untracked `scratch/` (#14): replace with tracked fixtures / regeneration
  recipes.
