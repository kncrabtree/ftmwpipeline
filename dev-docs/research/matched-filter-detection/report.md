# Matched-filter primary peak detection

A research report on whether replacing the peak-detection stage's
primary pass — a Savitzky-Golay second-derivative locator on a
Blackman-Harris-apodized spectrum — with a Lorentzian matched filter
(an exponentially-apodized FFT with per-bin SNR thresholding) followed
by the σ-weighted projection screen of
[the screen study](../stage3-coherence-screen/report.md) is a net win.
The companion script that regenerates every figure is
[`prototype.py`](prototype.py); the matched-filter reference
implementation lives there too (research scaffolding, not production
code).

The hypothesis (from the screen study's wiring proposal): a
matched-filter primary at `τ_basis ≈ 2-3 × τ_eff` + the screen should
recover at least the production primary pass's true-positive rate
while either (a) finding additional real lines or (b) producing a
cleaner candidate set, and should permit dropping the gap pass +
leakage mask + Blackman-Harris apodization.

**The verdict on the prompt's exact hypothesis is negative**, but a
hybrid design — exponential matched-filter apodization paired with the
production Savitzky-Golay concavity locator — uncovered while
investigating the failure mode dominates the production two-pass on
synthetic data at *every* SNR/FWHM cell tested and is a clean win in
principle. The bottleneck on real data is not the algorithm but the
grid: the matched filter on the active region only (no zero-padding,
per design) gives 0.079 MHz/bin on 2638, while the production primary
runs on a zpf=1-padded grid (0.033 MHz/bin) and recovers weak peaks
the active-FT cannot resolve.

The detailed walk-through is below. The prompt's pure
matched-filter-plus-projection-screen design fails because:

1. The matched filter's per-bin SNR threshold accepts every bin of a
   strong line's main-lobe + skirt above 4 σ, where production's
   Savitzky-Golay concavity test naturally rejects the monotonically-
   decaying tail bins.
2. The screen rates those tail bins as Lorentzian-coherent (correctly
   — they *are* tails of real Lorentzians), so it cannot remove them.
3. The auto-τ procedure proposed by the screen study converges to a
   value (1.15 µs) substantially smaller than the Stage 5 fit-determined
   τ_eff (≈ 3 µs); the auto-τ surface is not stable enough for
   production use without a fit-quality calibration loop.

The hybrid investigation (§§7-9, new) found:

- On synthetic data, **hybrid recall matches pure matched filter
  (≥ 0.97 across SNR ≥ 3, FWHM/bin ∈ [1, 5]) while halving the FP
  count** (concavity rejection kills the Lorentzian skirts that
  blow up pure-MF candidate counts).
- At high SNR (50 to 10⁴, where real strong lines actually live), the
  **hybrid has fewer FPs than the production two-pass** at the same
  recall — the exp apodization suppresses the sinc sidelobes that the
  unapodized gap pass picks up as FPs, *and* SavGol concavity rejects
  the matched-filter skirts.
- On 2638, the hybrid (sg_window=11) recovers fewer fit peaks than
  pure MF (79 % vs 89 %) because SavGol's window is grossly mis-tuned
  for the coarse active-FT grid (FWHM ≈ 2 active-FT bins, sg_window
  spans 5.5 FWHM). Reducing to sg_window=5 lifts the hybrid to 91 %
  recall, slightly *above* pure MF.
- The production primary remains the cleanest detector at high SNR
  (~30 FPs across all SNR cells tested) because BH apodization
  obliterates all but the strongest local maxima, and the SavGol
  window=11 is correctly tuned for the zpf=1 user grid where FWHM
  ≈ 5-8 bins.

§9 sketches the wiring proposal that *does* make sense given the new
findings: keep the production primary unchanged, replace the
unapodized gap pass's FFT with the matched-filter exp-apodized FFT
(keep the SavGol locator, keep the leakage mask). The MF gap pass
trades sinc-sidelobe FPs (production's gap-pass headache) for
Lorentzian-skirt FPs that the leakage mask + concavity test already
handle. The change is local, additive, and synthetic-validated.

## 1. Theory: σ-weighted Lorentzian projection ≡ apodized FFT

A σ-weighted complex projection of the active-portion FT `z(f)` onto a
finite-T Lorentzian basis at frequency `f_c` is

```
T(f_c) = ⟨h_T(·-f_c; τ_basis, T_active), z⟩_σ
       = Σ_k conj(h_T(f_k - f_c)) · z(f_k) / σ_c(f_k)²    /    ⟨h_T, h_T⟩_σ
```

with `σ_c = σ / √2` (per-component Rayleigh scale). The unweighted
correlation in the frequency domain is, by Parseval's theorem on a
finite-T record,

```
⟨h_T(·-f_c; τ_basis, T_active), z⟩
    = ∫_0^T exp(-t/τ_basis) · FID(t) · exp(-i 2π f_c t) dt
```

That is *the apodized FT of the FID at f_c*, with apodization weight
`exp(-t/τ_basis)`. Computing this at every `f_c` is one FFT (O(N log N)),
not N projections (which would be O(N²)). The σ-weighting becomes a
per-bin division `|X(f)| / σ_c(f)` rather than an inner-product
re-weighting. Under Gaussian time-domain noise the matched filter is
the Neyman-Pearson optimal detector for a damped-sinusoid signal of
known decay `τ`; the optimum is `τ_basis = τ_truth`.

Under that detector, per-bin noise on `|X|` follows a Rayleigh
distribution with scale `σ_c` (the per-real-or-imaginary-component
variance), and per-bin signal-plus-noise follows a Rice distribution.
The textbook detection threshold `|X|/σ_c ≥ 4` corresponds to a
Rayleigh tail probability `≈ 3.35 × 10⁻⁴`; with ~10⁵ bins in a 2638-
sized active-FT the expected pure-noise false-positive count is `~30`.

The current Stage 3 primary pass is structurally similar but uses
Blackman-Harris instead of exponential apodization and runs the
Savitzky-Golay second-derivative locator (`locate_peaks`) instead of
a simple SNR threshold. The Sav-Gol locator's concavity test
(`d²/df² < 0` AND local min of the second derivative) accepts only
*concave-down* features and rejects monotonically-decreasing
shoulders. That extra structural requirement turns out to matter on
real data (§5).

The matched-filter reference implementation
`matched_filter_detect(fid_samples, sample_dt_us, *, start_us, end_us,
tau_basis_us, ...)` takes the active region `[start_us, end_us]` only
— no zero-padding (the rfft length is `N_active = (end_us - start_us)
/ sample_dt_us`) — apodizes by `exp(-(t - start_us)/τ_basis)` so
`t = 0` aligns to the active-region turn-on, FFTs once, divides by
per-bin σ, and returns local maxima of the SNR statistic above
`detection_snr`. `min_separation_bins` is forwarded to
`scipy.signal.find_peaks(distance=...)` so adjacent main-lobe bins of a
single line are not both reported.

## 2. Smoke test

`figures/01_smoke_test.png` runs both detectors at SNR = 4,
FWHM/bin = 1.34 on a single simulator realisation (15 weak Lorentzians,
matched-τ basis). The matched filter recovers 15 / 15 truths with 5
"false positives" (candidates outside the truth-match tolerance of
1 bin) — all 5 sit within 2-4 bins of a real line and are noise
excursions riding the line's local shoulder. The production-like
detector (BH window + Sav-Gol locator at `min_snr=2`) recovers 4 / 15
truths with 7 FPs. The asymmetry is meaningful: the Blackman-Harris
window's main-lobe gain on a Lorentzian of FWHM ≈ 1.34 active-FT bins
suppresses on-line signal by ~5×, dropping most lines below the 2σ
detection floor on the BH-apodized spectrum.

## 3. Synthetic phase-space sweep

`figures/02_recall_fp_comparison_clean.png` (no strong lines) and
`figures/02_recall_fp_comparison_strong.png` (4 strong lines per
realisation at SNR = 50) sweep
`SNR ∈ {2, 3, 4, 5, 7, 10, 15, 25} × FWHM/bin ∈ {0.8, 1.0, 1.34, 2,
3, 5}`, 4 simulator trials per cell. Three detectors are compared:

1. **Matched filter** at `τ_basis = 2 × τ_truth`, `detection_snr = 4`,
   `min_separation_bins = max(2, round(fwhm_bins))`.
2. **Production-like primary** (BH window + Sav-Gol locator + apex
   snap at `min_snr = 2`).
3. **Production two-pass** (the production primary plus an unapodized
   Sav-Gol locator union, mirroring the gap pass with no leakage
   mask).

Pre-screen weak-line recall (no strong lines, fraction of injected
truths with ≥ 1 candidate within `max(1, fwhm_bins/2)` bins):

| SNR \ FWHM/bin | 0.8 | 1.0 | 1.34 | 2.0 | 3.0 | 5.0 |
|----------------|-----|-----|------|-----|-----|-----|
| **Matched filter** | | | | | | |
| 2.0  | 0.24 | 0.32 | 0.56 | 0.82 | 0.94 | 0.93 |
| 3.0  | 0.84 | 0.91 | 0.97 | 1.00 | 1.00 | 0.95 |
| 4.0  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 |
| 5.0  | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.96 |
| 10.0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 |
| **Production primary (BH+SavGol)** | | | | | | |
| 2.0  | 0.09 | 0.08 | 0.06 | 0.02 | 0.04 | 0.04 |
| 3.0  | 0.17 | 0.16 | 0.09 | 0.04 | 0.04 | 0.04 |
| 4.0  | 0.49 | 0.37 | 0.19 | 0.08 | 0.05 | 0.04 |
| 5.0  | 0.66 | 0.59 | 0.39 | 0.10 | 0.05 | 0.04 |
| 10.0 | 0.99 | 0.99 | 0.94 | 0.61 | 0.13 | 0.04 |
| **Production two-pass (primary + gap union)** | | | | | | |
| 2.0  | 0.20 | 0.23 | 0.26 | 0.32 | 0.44 | 0.44 |
| 3.0  | 0.41 | 0.47 | 0.54 | 0.59 | 0.78 | 0.80 |
| 4.0  | 0.72 | 0.72 | 0.76 | 0.89 | 0.95 | 0.94 |
| 5.0  | 0.89 | 0.92 | 0.94 | 0.95 | 0.98 | 0.99 |
| 10.0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Three findings:

- The matched filter reaches recall ≥ 0.97 across `SNR ≥ 4,
  FWHM/bin ∈ [0.8, 5]`. At SNR = 3, FWHM/bin ∈ [0.8, 5] it still
  delivers ≥ 0.84. Below SNR = 3 the matched filter degrades but only
  for FWHM/bin ≤ 1 (where the line is barely resolved).
- **The production primary pass alone is essentially nonfunctional in
  the regime real FTMW data sits in** (FWHM/bin ≈ 1-2). At SNR = 4,
  FWHM/bin = 1.34 the primary recovers only 19 % of injected lines —
  the BH window suppresses the on-line signal energy of a damped line
  below the 2σ detection floor. This is consistent with the
  [peak-detection report](../peak-detection/report.md) §6's finding
  that the primary pass shifts work to the gap pass on damped narrow
  lines.
- The two-pass production design (primary + gap union) recovers the
  lost recall at SNR ≥ 5 but still misses ≥ 25 % of lines at SNR = 4
  in the narrow-FWHM regime. Matched filter beats it by a clear
  margin at SNR ∈ [3, 5].

`figures/02_recall_fp_comparison_clean.png` also shows the pre-screen
**false-positive** counts: ~40 FPs across the grid for the production
primary (essentially constant noise-level FPs, independent of signal),
~70 FPs for the two-pass, and a *growing* FP count for the matched
filter (1-94 at SNR = 2; 250-786 at SNR = 25). The matched filter's
growing FP count is not noise excursions — it is real lines whose
Lorentzian skirts clear 4σ for many bins each, with each bin reported
as a separate candidate. The Sav-Gol concavity test would suppress
those (the skirt is monotonic, not a local minimum of the second
derivative); `find_peaks` with only a height threshold cannot.

### Post-screen behaviour

Applying the screen at `τ_basis = 2 × τ_truth`, `ratio ≥ 0.6` (the
operating point the screen study recommends) to each detector's
candidate set:

| | Matched filter | Production primary | Production two-pass |
|---|---|---|---|
| Recall at SNR=5, FWHM=1.34 (pre) | 1.00 | 0.39 | 0.94 |
| Recall at SNR=5, FWHM=1.34 (post)| 1.00 | n/a (screen rejects)| 0.93 |
| Recall at SNR=5, FWHM=3.0 (pre)  | 0.99 | 0.05 | 0.98 |
| Recall at SNR=5, FWHM=3.0 (post) | 0.99 | n/a | **0.13** |
| Total FPs at SNR=5, FWHM=2.0 (pre)| 108 | 45 | 78 |
| Total FPs at SNR=5, FWHM=2.0 (post)| 108| 45 | 4  |

Two surprises:

- **The screen barely touches the matched filter's FPs.** Pre-screen
  and post-screen FP counts are nearly identical (108 vs 108 in the
  highlighted cell). The "FPs" are off-line bins of real Lorentzians,
  and the screen rates them as Lorentzian-coherent (because they
  are — the local phase profile is the line's own).
- **The screen catastrophically loses two-pass recall at FWHM ≥ 2.**
  At FWHM = 3 the two-pass post-screen recall drops from 0.98 to 0.13
  at SNR = 5. The screen's Lorentzian basis at `τ_basis = 2 ·
  τ_truth` is narrower than the molecular linewidth, and the
  BH-apodized line shape is *non-Lorentzian* (BH convolves the
  Lorentzian with a wide non-Lorentzian instrument function), so the
  σ-weighted projection of the basis onto BH-apodized data drops the
  ratio below 0.6. The screen and Blackman-Harris are not compatible.

The screen AUC heatmaps (`figures/04_screen_auc_by_detector_*.png`)
make the same point quantitatively: the screen AUC on matched-filter
candidates rises from ≈ 0.6 at SNR = 4 to ≈ 0.99 at SNR ≥ 10 (the
screen *does* separate Lorentzian-coherent from noise once SNR is
high enough that off-line-tail and pure-noise FPs separate). On BH+
SavGol candidates the AUC is *higher* at low SNR (the surviving
candidates are mostly true positives because BH already filtered
heavily), but irrelevant — the pre-screen recall is already 0.05.

`figures/03_post_screen_comparison_*.png` show the recall-vs-FP
trade-off across the full grid. The matched filter is the only
detector that combines high recall with reasonable FP behaviour at
SNR ∈ [3, 10], FWHM ∈ [1, 2]. Outside that band, all detectors
suffer.

### Synthetic verdict

The matched filter primary is a meaningful improvement over the
production primary in the regime real FTMW data sits in
(`FWHM/bin ≈ 1-2`, `SNR ≈ 3-10`). The screen does not rescue
production candidates and does not clean matched-filter candidates of
the off-line-tail FPs. The architecture worth testing on real data
is matched-filter primary *with the production gap pass kept as a
backstop for wider lines* and *without* the screen — that is what §4
checks.

## 4. 2638 application

### 4.1 Auto-τ_basis calibration

The screen study's wiring proposal calls for a per-experiment τ_basis
calibrated from the brightest detected candidates' linewidths. The
prototype's `auto_calibrate_tau_basis`:

1. Builds the active-FT at the production-canonical apodization
   `expf_us = 5.0 µs` (the Stage 1 setting).
2. Finds the top-10 brightest candidates by per-bin SNR (`> 10 σ`).
3. Walks outward from each candidate's apex to the half-max bin on
   the magnitude spectrum.
4. Takes the median FWHM_mhz, converts to `τ_eff_observed =
   1 / (π · FWHM_mhz)`, and sets `τ_basis = 2 · τ_eff_observed`.

On 2638 this returns `τ_eff_observed ≈ 1.15 µs`,
`τ_basis ≈ 2.30 µs`. The prompt-stated strong-line fit-determined
`τ_eff` is ≈ 3 µs; the auto-τ procedure underestimates by ~2.6×. The
discrepancy comes from two effects:

- The half-max walk is measured on the *apodized* spectrum
  (apod = `exp(-t/5)`). The apodized line shape has effective
  `τ_apod = 1/(1/τ_truth + 1/5)` so observed FWHM > true molecular
  FWHM. Deconvolving naively gives `τ_truth ≈ 1.5 µs` — still off.
- Discrete-bin half-max walks at FWHM ≈ 2-3 active-FT bins are
  quantised and noise-biased; the median over 10 brightest lines does
  not converge to the fit-determined τ.

The fix that *would* work is a proper Lorentzian fit on the brightest
line, but that is itself a one-line Stage 5 mini-fit — at which point
the per-experiment calibration cost rivals the cost of running the
full pipeline once at a default τ_basis. The auto-τ procedure as
specified is not robust enough for production wiring.

### 4.2 Detection and screen results

At `τ_basis = 2.30 µs` (auto-τ), `detection_snr = 4`,
`min_separation_bins = 3`:

| Metric | Matched filter | Production (promoted) |
|--------|----------------|-----------------------|
| Total candidates (inside [26500, 40000] MHz) | 3,449 | 709 |
| Fit peaks recovered (out of 648) | 577 (89.0 %) | 621 (95.8 %) |
| Candidates × per fit peak | 5.3 | 1.1 |

The matched filter finds 5× more candidates than production but
recovers fewer Stage 5 fitted peaks. The diff:
- `both` (MF ∩ PL): 581
- `mf_only`: 2,868
- `pl_only`: 116

Most of the 2,868 mf-only candidates are not in the Stage 5 fit — i.e.
they were not curated as real lines.

The screen at ratio ≥ 0.6 keeps 3,441 / 3,449 = 99.8 % of matched-filter
candidates: it removes essentially nothing. `figures/06_2638_ratio_histogram.png`
shows the underlying distribution: matched-filter candidates on 2638
have ratios in [0.6, 1.5] regardless of whether they are within 2 bins of
a fit peak. The two populations (near-fit-peak / not-near-fit-peak)
substantially overlap. The screen's calibration on synthetic
analytic-σ data does not transfer to real spectra.

A sweep of `tau_basis ∈ {1.5, 2.30, 3.0, 6.0, 9.0} µs` and
`detection_snr ∈ {2, 3, ..., 15} σ`
(`figures/07_2638_threshold_sweep.png`) explores the parameter space:

- Fit recall peaks at `τ_basis = 6.0 µs` (the manual `2 × 3 µs`
  setting): 574 / 648 = 88.6 % at `detection_snr = 4`. Auto-τ at
  `τ_basis = 2.30 µs` recovers 560 / 648 (86.4 %), and `τ_basis = 3.0,
  9.0 µs` give comparable numbers. The matched filter's recall is
  remarkably insensitive to `τ_basis` over a factor of ~6.
- Raising `detection_snr` does *not* let the matched filter match
  production's fit recall (95.8 %) at lower candidate counts.
  At `detection_snr = 4` and `τ_basis = 6 µs` the MF candidate count
  is 5× production with 88.6 % recall; at `detection_snr = 10` it is
  ~equal to production (665 vs 709) with only 51.7 % fit recall — the
  candidate set is comparable in *size* but covers different peaks.
- There is no `(τ_basis, detection_snr)` operating point where the
  matched filter dominates production on both axes (more recall AND
  fewer candidates).

### 4.3 Why the matched filter under-recovers fit peaks on 2638

The matched filter is provably the optimal *bin-level* detector under
Gaussian noise — yet it recovers fewer Stage 5 fitted peaks than
production. Three reasons:

1. **Grid coarseness.** The matched filter runs on the active-FT
   (no zero padding: bin = 1/T_active = 0.0791 MHz). Production runs
   on the user grid at `zpf = 2` (bin = 0.0202 MHz). Weak peaks that
   sit between active-FT bins are missed by the matched filter's
   one-bin-per-candidate sampling, while production's finer grid
   resolves them. The screen study found `FWHM/bin ≈ 1-2` is the
   optimum; on 2638 the active-FT puts molecular FWHM ≈ 0.17 MHz at
   ≈ 2.1 active-FT bins, but the user grid puts the same FWHM at
   ≈ 8.5 user-grid bins. Production's "wasted" oversampling is
   purchasing localisation that matters for weak peaks.
2. **Stage 5 fitted peaks are not all above 4σ on the active-FT.**
   The Stage 5 curation accepts peaks down to user-grid SNR ≥ 3 (the
   promotion cutoff), and the per-bin SNR on the coarser active-FT
   grid is *not the same* as on the user grid — apodization, bin
   spacing, and noise estimation all differ. A user-grid SNR = 3
   peak can be at SNR ≤ 3 on the active-FT (where the coarser bin
   absorbs more of the line's energy *and* more noise variance), so
   `detection_snr = 4` drops it.
3. **The Stage 5 fit is not ground truth.** It is an outside
   reference: peaks the production pipeline accepted *and* the fit
   converged on. Some persistent fit peaks may not be physically real
   (curation accepts borderline lines), and some matched-filter
   candidates may be real lines the production pipeline missed. The
   ~7 % gap is the union of "MF missed a real line" and "MF found a
   real line not in the persisted fit", and the metrics here cannot
   tell them apart.

`figures/05_2638_overlay.png` shows the spatial distribution of
matched-filter candidates, production candidates, and fit peaks across
the 26500-40000 MHz band. The matched-filter excess is concentrated
near strong lines (Lorentzian skirts clearing 4σ for many bins);
production peaks are sparser and align cleanly with fit peaks.

## 5. Why the matched filter over-produces near strong lines

The fundamental issue is the per-bin SNR threshold. On a strong line
at SNR = 40, FWHM = 2 active-FT bins, the Lorentzian magnitude at bin
offset `Δ` is `40 · (FWHM/2)² / ((FWHM/2)² + Δ²) = 40 / (1 + Δ²)`
(normalising bin = 1). This clears 4σ for `|Δ| ≤ 3` bins, i.e. ±3
bins of the line centre. With `min_separation_bins = 3` the matched
filter reports up to 2 candidates per strong line (centre and one ~3
bins out); the bins between are suppressed by `find_peaks(distance=3)`
but the candidates 3 bins out are *kept* because they meet the
distance constraint and clear 4σ. Stronger lines (SNR = 100, 200)
have above-4σ extent of ±5 bins, producing 3-4 candidates each.

The Sav-Gol concavity test rejects these candidates because the line's
skirt is monotonically decreasing (the second derivative does not have
a local minimum there). This is the structural difference between
"local maximum above threshold" (matched filter) and "local minimum
of the second derivative below zero AND magnitude above threshold"
(Sav-Gol). The latter requires the candidate to be *concave-down* at
the noise scale, which is exactly the property that distinguishes a
genuine peak from a peak's own skirt.

The screen study reasoned that the σ-weighted projection ratio would
also distinguish these, but the projection at a skirt-bin candidate
sits on a real Lorentzian (the strong line's own shape) and projects
coherently. The screen ratio at a skirt-bin candidate is ≈ 1 — not
distinguishable from an on-line candidate.

## 6. Architectural questions

### 6.1 Does the gap pass still make sense?

The gap pass exists in the production design because the BH-apodized
primary pass smears weak lines below the 2σ floor (§3). At
`τ_basis = 2 × τ_eff` the matched filter places weak lines at-or-near
optimal SNR, so in principle the gap pass is unnecessary in a
matched-filter design.

Two observations qualify this:

- The matched-filter's weak-line recall on synthetic data is ≥ 0.95 at
  FWHM/bin ∈ [1, 5] across SNR ≥ 3, dominating both production primary
  and two-pass union. Architecturally the gap pass *would* be
  redundant *if* the matched filter were viable.
- On 2638, the matched filter recovers fewer fit peaks than the
  production two-pass design. The gap pass is doing work the matched
  filter cannot match on real data (the explanation in §4.3:
  finer-grid localisation, lower effective threshold). Until the
  matched filter dominates production on real data, the gap pass
  stays.

### 6.2 Does the leakage mask still apply?

The leakage mask (`leakage_touched_intervals` on the de-ramped
spectrum) excludes regions where a strong line's truncation skirt is
detectable, so the gap pass cannot re-detect skirt bins as weak
lines. The matched filter's per-bin SNR threshold *also* picks up
those skirt bins (§5), so the leakage mask remains relevant. The
screen does not substitute for the leakage mask — it cannot
distinguish a line's own tail from a separate line on the tail.

### 6.3 The matched-filter + Sav-Gol concavity hybrid

The diagnoses in §5 + §6.1-2 suggest the right next experiment: pair
the matched-filter exponential apodization (best per-bin SNR for a
damped sinusoid) with the production Sav-Gol concavity locator
(rejects monotonic skirts). The hybrid combines the strengths of
both: matched-filter SNR optimality on weak lines and structural
suppression of strong-line skirt bins. The prompt asked for
matched-filter + screen — this section asks the question the prompt
didn't, because the failure mode points here.

The hybrid is `matched_filter_concavity_detect` in `prototype.py`:
compute the active-FT at `expf_us = τ_basis`, then run `locate_peaks`
(the production locator) on its magnitude. No screen. No zero
padding. Apodization referenced to `start_us`.

## 7. Hybrid synthetic results

Pre-screen weak-line recall at `τ_basis = 2 × τ_truth`,
`min_snr = 2`, `sg_window = 11`, `sg_order = 3`:

| SNR \ FWHM/bin | 0.8 | 1.0 | 1.34 | 2.0 | 3.0 | 5.0 |
|----------------|-----|-----|------|-----|-----|-----|
| 2.0  | 0.32 | 0.37 | 0.62 | 0.84 | 0.93 | 0.93 |
| 3.0  | 0.56 | 0.82 | 0.97 | 0.99 | 0.98 | 0.95 |
| 4.0  | 0.87 | 0.97 | 1.00 | 1.00 | 0.99 | 0.95 |
| 5.0  | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.95 |
| 10.0 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 0.95 |

For comparison: pure MF is essentially identical (within ±0.05) at
SNR ≥ 3; production two-pass is 0.54 at the (SNR=3, FWHM=1.34) cell
where the hybrid hits 0.97. **The hybrid matches pure MF's recall
without paying its FP cost.**

Pre-screen total FPs at SNR=10, FWHM=1.34:

| Detector | Pre-screen FPs |
|----------|----------------|
| Pure matched filter (per-bin SNR ≥ 4) | 146 |
| Hybrid (MF apod + SavGol concavity, min_snr=2) | 74 |
| Production primary (BH + SavGol)            | 39 |
| Production two-pass (primary + gap)         | 71 |

The hybrid halves the matched filter's FP count. The remaining factor
of 2 vs production primary comes from the matched filter's wider
above-threshold extent compared to BH (the BH window suppresses
non-mainlobe magnitude much more aggressively than `exp(-t/τ_basis)`).

`figures/02_recall_fp_comparison_clean.png` shows the full grid for
all five detectors side-by-side. The hybrid is the only detector that
simultaneously has near-perfect recall AND fewer than 200 FPs at every
SNR/FWHM cell tested.

### 7.1 High-SNR scaling

Real FTMW spectra carry strong lines at SNR up to 10⁴. A Lorentzian
of FWHM = 2 bins and on-line SNR = `S` has above-4σ extent of
`Δ_max ≈ √(S/4 - 1)` bins per side — at S = 10⁴ that is 50 bins.
The pure matched filter's per-bin SNR threshold accepts all those
bins as candidates; the Sav-Gol concavity test should reject them
because they are concave-up (positive second derivative everywhere on
the skirt). `figures/09_high_snr_fp_scaling.png` tests whether the
expected rejection holds.

Total FPs at FWHM/bin = 1.34, sweeping injected SNR up to 10⁴:

| SNR    | Pure MF | Hybrid | Production primary | Production two-pass |
|--------|---------|--------|--------------------|---------------------|
| 50     | 723     | 261    | 28                 | 172                 |
| 100    | 1061    | 313    | 29                 | 270                 |
| 300    | 1033    | 294    | 28                 | 365                 |
| 1000   | 722     | 223    | 28                 | 297                 |
| 3000   | 416     | 162    | 28                 | 230                 |
| 10000  | 163     | 78     | 27                 | 156                 |

(Pure MF FPs *decrease* past SNR ≈ 300 because `find_peaks(distance=
max(2, FWHM_bins))` collapses the densely-spaced skirt-bin candidates
to fewer survivors per line — but they are still skirt bins, not real
finds.)

Three observations:

1. **The production primary stays at ~28 FPs across every SNR**
   tested. BH apodization smooths the spectrum so aggressively that
   only the on-line bin of each line stays above the magnitude
   threshold. This is the BH window's intended behaviour and its
   real selling point at the architectural level: a constant noise
   FP floor independent of how bright the strong lines are.
2. **The hybrid has fewer FPs than the production two-pass at every
   SNR ≥ 50**. The production two-pass picks up sinc sidelobes from
   the boxcar-truncated strong lines (consistent with the
   [peak-detection report](../peak-detection/report.md) §3's 211
   sidelobes-as-FPs row); the matched-filter exp apodization
   suppresses sidelobes so the hybrid sees far fewer of them.
3. **The hybrid recall stays at 1.0 across the whole high-SNR grid**
   (except 0.96 at FWHM = 5 — the wide-FWHM regime where every
   detector loses a small fraction to its truth-match tolerance).

The hybrid is *not* better than the production primary on high-SNR
synthetic data — production has the constant ~28-FP floor — but it is
strictly better than the production *two-pass*, which is what the
hybrid would replace. At low SNR (§7's table) production primary
collapses to recall ≤ 0.2 in the regime real data sits in, while the
hybrid stays at recall ≥ 0.95.

## 8. Hybrid on 2638

`figures/08_2638_hybrid_comparison.png` compares four configurations
on 2638 at `τ_basis = 6 µs`:

| Configuration | n_candidates (trim band) | Fit peaks recovered (of 648) |
|---|---|---|
| Production primary (Stage 3 user-grid output) | 709  | 648 / 648 (100 %) |
| Pure matched filter (per-bin SNR ≥ 4)          | 3,449 | 577 / 648 (89.0 %) |
| Hybrid (MF + SavGol, sg_window = 11)           | 2,969 | 513 / 648 (79.2 %) |
| Production + MF gap (union)                    | 3,162 | 648 / 648 (100 %) |

The hybrid under-recovers more than pure MF on 2638. Drilling in:
the production primary runs `locate_peaks(sg_window=11)` on the
*Stage 3 internal zpf=1 grid* (0.033 MHz/bin), where the molecular
FWHM ≈ 0.16 MHz ≈ 5 bins. The hybrid runs the same locator on the
*active-FT* (0.079 MHz/bin), where the apodized FWHM is ≈ 2 bins,
and `sg_window = 11` spans 5.5 FWHM — grossly mis-tuned for the
finer scale of the matched filter's lineshape.

Sweeping the SavGol window on the hybrid at 2638:

| sg_window | n_candidates | Fit-peak recovery |
|-----------|--------------|-------------------|
| 5         | 6,682        | 590 / 648 (91.0 %) |
| 7         | 4,673        | 554 / 648 (85.5 %) |
| 9         | 3,634        | 528 / 648 (81.5 %) |
| 11        | 2,969        | 513 / 648 (79.2 %) |
| 15        | 2,156        | 476 / 648 (73.5 %) |

A tighter SavGol window does recover narrower lines, at the cost of
more FPs. At sg_window = 5 the hybrid recall (91 %) beats pure MF
(89 %) — but at the cost of *2.3× the candidates* of pure MF. The
hybrid does not match production's 100 % fit-peak recovery at any
sg_window because the active-FT grid is fundamentally coarser than
production's zpf=1 user grid: peaks that fall between active-FT bins
are missed by any detector running on that grid, including the
hybrid.

**The 2638 gap between the hybrid and production is grid coarseness,
not algorithm.** The matched filter on the active region only (no
zero padding, per design) has bin spacing 1/T_active = 0.079 MHz.
Production runs at 0.033 MHz/bin on the Stage 3 internal zpf=1 grid,
2.4× finer. Weak fit peaks in the gaps between active-FT bins simply
aren't visible to the matched filter at the grid resolution it
operates on.

The "Production + MF gap union" row recovers 100 % of fit peaks (it
includes the production output, which by construction recovers all
fit peaks the production pipeline accepted) at 4× the production
candidate count. That is the local Pareto improvement available
without grid changes: take production's user-grid candidates as the
authoritative weak-line set, and add MF candidates for any *new*
peaks the production might have missed. The cost is the FP inflation
the MF brings.

## 9. Verdict (revised)

The prompt's pure matched-filter + projection-screen hypothesis fails:

- **TP recovery (≥ 95 % of production)**: not met. Pure MF recovers
  89 %; the screen kept 99.8 % of MF candidates (no discrimination
  on real data).
- **Architectural simplification (drop BH + SavGol + gap + mask)**:
  not realised. Skirt-bin FPs remain.

The hybrid (matched-filter apodization + Sav-Gol concavity locator,
no screen) is a different story:

- **Synthetic data**: dominates the production two-pass at every
  SNR/FWHM cell tested, including high SNR (50 to 10⁴). Recall
  ≥ 0.95 at SNR ≥ 3, FWHM ∈ [0.8, 5]; FP count ≤ 200 across the
  grid, fewer than production two-pass at high SNR.
- **2638**: under-recovers fit peaks (79-91 % depending on sg_window)
  because the active-FT grid is coarser than production's zpf=1 grid.
  Algorithm-equivalent if grid-matched (the hybrid at sg_window=5
  beats pure MF), but the grid is the limit, not the algorithm.

The cleanest architectural win available without a grid change is
**matched-filter exp-apodization as the gap pass** (§10 wiring).
That trades the production two-pass's sinc-sidelobe FPs (~70-365
across the grid, growing with SNR) for the matched-filter's
Lorentzian-skirt FPs at high SNR (~78-313, *not* growing with SNR
past a peak). Net: cleaner candidate set at high SNR, equal recall at
low SNR. The production primary stays unchanged; the leakage mask
stays unchanged; the screen is not adopted.

## 10. Wiring proposal: matched-filter gap pass

A drop-in replacement for the unapodized gap pass:

1. **Primary pass unchanged.** Continue `_spectrum_from_fid(...,
   expf_us=None, window_function="blackmanharris")` for the strong-
   line list. Its purpose (clean strong-line list to seed the leakage
   mask) is unaffected by gap-pass changes.
2. **Replace the gap-pass spectrum**: instead of `_spectrum_from_fid(
   ..., expf_us=None, window_function=None)` (unapodized, full FID
   zpf=1), build the matched-filter active-FT via `compute_active_ft(
   ..., expf_us=τ_basis, ...)`. `τ_basis` is set per experiment from
   the Stage 1 `expf_us` (e.g. `τ_basis = expf_us` matches the user-
   chosen apodization; `τ_basis = 2 × expf_us` is slightly narrower).
   The exact choice is a calibration question that the validation
   harness should decide — *but* the algorithm tolerates τ_basis
   over a factor of 6 on 2638 with < 5 % recall variation, so the
   sensitivity is low.
3. **Gap detection locator unchanged.** Run `locate_peaks(sg_window=
   max(5, fwhm_active_bins · 3), sg_order=3, thresh=min_snr · σ)` on
   the matched-filter active-FT magnitude. The sg_window must scale
   with grid coarseness (§8 finding) — production's `sg_window = 11`
   is correct for the zpf=1 user grid; for the active-FT it should be
   ~5-7.
4. **Leakage mask unchanged.** The de-ramped coherence map still
   gates the gap pass. The matched-filter apodization suppresses the
   sinc skirts the mask used to exclude (now a defense-in-depth
   layer), but Lorentzian skirts past the apodized line still warrant
   the mask.
5. **No screen.** Drop the projection-screen post-filter. The 2638
   ratio distribution (§4.2) shows no discrimination on real data.
6. **Snap-back unchanged.** Existing snap-back from the
   gap-detection grid to the user grid handles the bin-coarseness
   delta between detection and reporting.

Why the gap pass is the right place and not the primary:

- The primary's *strong*-line list is what the leakage mask depends
  on. Replacing it with a matched-filter primary risks introducing
  skirt-bin candidates into the strong-line list, polluting the
  mask. Keep the strong window for the strong-line list.
- The gap pass's job is *weak*-line recovery. The matched filter is
  exactly the right tool for that job (the smoke test + §7 tables
  make this clear).

Estimated impact: the gap-pass change is local to
`_spectrum_from_fid` and the gap-pass-specific `locate_peaks` call in
`detect_peaks_impl`. Validation harness on 2638 should catch any
regression in Stage 5 outcomes; if recall doesn't budge but
candidate quality improves at higher-SNR fixtures, ship it.

## 11. Open questions

- **Robust per-experiment τ calibration.** The half-max-walk auto-τ
  is biased and noisy; a fit-based estimate adds Stage-5-like cost to
  Stage 3. A closed-form estimator (e.g., from the second moment of
  the active-FT magnitude around the brightest candidate) might
  thread the needle. Less pressing now that the gap-pass τ is just
  the Stage 1 `expf_us`.
- **Why does the screen ratio distribution collapse on real data?**
  The screen study saw ratios spread over [0, 2+] on synthetic
  spectra with analytic σ. On 2638 ratios cluster in [0.6, 1.5]. The
  Stage 2 adaptive noise estimator on the active-FT (~10⁵ bins, not
  the user-grid ~10⁶) may behave differently than on the persisted
  spectrum, biasing σ and so the ratio numerator. A noise-estimator
  audit on the active-FT grid is the obvious follow-up — the user
  has flagged this as the next study.
- **Adaptive sg_window for the gap-pass locator.** The §8 finding —
  sg_window must match the apodized FWHM in bins — is a calibration
  question with one obvious answer (set sg_window from τ_basis and
  T_active) but no empirical sweep against fixtures other than 2638.
  Cross-instrument validation needed before shipping.
- **High-SNR FP behaviour on real spectra.** §7.1's synthetic
  high-SNR scaling shows the hybrid beating production two-pass past
  SNR ≈ 50. 2638's strongest lines are at SNR ~ 10³-10⁴; the wiring
  change should improve 2638 candidate quality near those lines but
  the 4σ noise FP rate may also shift. Worth measuring.
- **Cross-instrument generality.** 2638 is one fixture. The matched
  filter's synthetic advantage holds at FWHM/bin ∈ [1, 5], SNR ≥ 3;
  fixtures outside that band might give a different verdict.

## 12. What this study is *not* covering

- **Phase-only sub-threshold detection.** Out of scope per the
  prompt.
- **Replacing the screen with a different statistic.** Out of scope.
- **Matched-filter primary pass** (as opposed to gap pass). §§7-9
  established this loses to production on 2638 because of grid
  coarseness; the gap-pass variant in §10 is the right wiring.

## 13. Reproducibility

```bash
# from repository root, with the project conda env
conda run -n ftmwpipeline-dev python \
    dev-docs/research/matched-filter-detection/prototype.py
```

Runtime: ~30 seconds end-to-end on a contemporary laptop CPU. Writes
nine PNGs to `figures/` and three `.npz` intermediates to `data/`.
Re-running overwrites both. The 2638 sections need
`scratch/stage5-validation/exp_2638.ftmw`; the synthetic sections are
self-contained.

Random seeds for the simulator are fixed inside `prototype.py`
(seed `20260524` + trial index). The screen study's simulator
(`dev-docs/research/stage3-coherence-screen/prototype.py::simulate_active_ft`)
is loaded via `importlib`, so any change there propagates here.

## 14. Related artefacts

- The matched-filter reference implementation:
  [`prototype.py::matched_filter_detect`](prototype.py).
- The σ-weighted projection screen:
  [`src/ftmwpipeline/preprocessing/coherence_screen.py`](../../../src/ftmwpipeline/preprocessing/coherence_screen.py)
  + [its research report](../stage3-coherence-screen/report.md).
- The production peak-detection design:
  [its research report](../peak-detection/report.md).
- The active-FT builder shared with Stage 5:
  [`src/ftmwpipeline/fitting/active_ft.py`](../../../src/ftmwpipeline/fitting/active_ft.py).
- The Stage 3 orchestrator:
  [`src/ftmwpipeline/_internal/stage3_impl.py`](../../../src/ftmwpipeline/_internal/stage3_impl.py).
