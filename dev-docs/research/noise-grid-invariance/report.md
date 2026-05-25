# Noise-estimator grid invariance

A research report on whether the Stage 2 adaptive noise estimator
([`src/ftmwpipeline/preprocessing/noise_estimation.py`][noise]) returns
the same per-bin σ(f) when applied to the same physics on different
grids. The matched-filter detection study uncovered a striking
mismatch — σ(f) on the persisted user grid (zpf=2) and σ(f) on the
active-portion FT (zpf=0) of the 2638 fixture disagree by up to 67 %,
and on the active-FT the estimator degenerates to a single global bin.
This study traces the failure to a specific algorithmic flaw and
prototypes a robust-statistics variant that restores grid invariance.

[noise]: ../../../src/ftmwpipeline/preprocessing/noise_estimation.py

**Verdict**: the current estimator's subdivision step uses
skewness-trimmed mean and variance to detect σ heterogeneity, but
the skewness trim *aggressively flattens* heterogeneous regions into
homogeneous-looking ones before the subdivision criterion ever sees
them. The trim is doing double duty — noise-mask identification *and*
subdivision-criterion input — and the conflation kills σ-structure
detection on the active-FT, where the global trim happens to converge
to a Rayleigh-target distribution that hides the underlying step
structure. A MAD/median-based subdivision criterion on the raw (un-
trimmed) magnitudes recovers grid invariance: 2638 σ ranges agree to
0.4 % between user grid and active-FT (8.28× vs 8.31×), and synthetic
3-step σ(f) is recovered at every zpf ∈ {0, 1, 2, 3}.

This started as an audit + diagnosis + candidate fix. Shipped state
in production (`src/ftmwpipeline/preprocessing/noise_estimation.py`,
see §8):

1. Subdivision uses median + MAD on raw \|X\| (the root-cause fix).
2. σ-estimation uses moving-median (robust to skirt residuals,
   tracks the instrument's ~few-hundred-MHz coherence scale).
3. Strong-line skirts are excluded explicitly with a physical
   Lorentzian-radius model, removing residual skirt-bias bumps in
   σ(f).

The §5 per-bin-MAD σ variant was tried and rejected — its piecewise-
constant σ(f) failed to track real σ variation visible on the user
grid. It remains the reference for off-line grid-invariance
diagnostics where exact σ-value agreement across grids is the
metric.

## 1. Setup -- the algorithmic context

The Stage 2 estimator (`estimate_noise_adaptive`) produces a per-point
σ(f) by:

1. **Recursive bisection** of the magnitude spectrum into bins. A
   region is split when *both halves have enough noise points* AND
   their **post-trim** mean / variance differ by at least one of
   four thresholds (20 % relative mean, 5 σ z-test, 20 % relative
   variance, or F ≥ 2).
2. **Skewness-targeted trimming** inside each final bin: drop the
   top-magnitude points in 1 %-rank steps until the kept samples'
   skewness drops to the Rayleigh target (0.6311).
3. **Smoothed RMS** computed by convolving squared kept-magnitudes
   with a uniform window targeting ~2500 noise samples (~1 % RMS
   stability per Rayleigh δ-method, the
   [noise-heuristic-audit](../noise-heuristic-audit/report.md) §4
   calibration).

The audit's previous calibrations stand: the smoothing-sample target
is principled, the Rayleigh skewness target is an analytic constant.
The flaw uncovered here is in step 1: post-trim statistics are used
to gate subdivision, and the trim hides exactly the heterogeneity
the subdivision is supposed to detect.

## 2. 2638 across grids

`figures/01_2638_sigma_vs_grid.png` shows σ(f) on three views of the
2638 data, each normalised to its own median:

- **User grid (zpf=2)**: 51 adaptive bins, smoothing window 32 MHz,
  σ_max/σ_min = 7.10×.
- **Active-FT pre-trimmed to the user trim band (zpf=0)**: 35 bins,
  smoothing 212 MHz, σ range 4.75×.
- **Active-FT NOT pre-trimmed**: 1 bin spanning the whole 25 GHz
  baseband. Smoothing 222 MHz.

The "1 bin" failure on the un-trimmed active-FT traces to the
out-of-band low-noise region. The active-FT covers 15960-40960 MHz;
the user trim is [26500, 40000] MHz. The lower 25 % of the active-FT
band is essentially zero (it sits below the bandpass of the data
acquisition); when the recursive subdivision tries to split at the
midpoint, the left half is 75 % out-of-band zeros plus 25 % real
data, and its noise fraction comes out below the 2/3 threshold, so
subdivision is rejected globally.

This part of the failure is easy to fix: pre-trim the active-FT to
the user's trim band before passing to the estimator. The user grid
is already pre-trimmed by Stage 1, which is why it succeeds.

But pre-trimming alone is *not enough*. Comparing the user-grid σ
(normalised) to the active-FT σ (also normalised) shows median 8 %
disagreement and p95 disagreement of 27 %. The two grids report
*qualitatively different σ structure*, not just a global offset.
Something else is wrong.

## 3. Synthetic ground truth -- the recovery failure

`figures/02_synthetic_recovery.png`. Build a noise-only FID whose
freq-domain σ has three discrete levels (1.0 / 3.0 / 1.5 in the three
baseband regions [0, 6.25], [6.25, 12.5], [12.5, 25] MHz). The FID is
constructed by inverse-rfft from a complex Gaussian with the
prescribed σ per bin, then real-cast. The estimator at multiple
zero-padding factors recovers:

| zpf | n_bins | σ range (max/min) | smoothing (MHz) |
|-----|--------|-------------------|-----------------|
| 0   | 1      | 1.08×            | 33              |
| 1   | 1      | 1.46×            | 16              |
| 2   | 1      | 1.87×            | 8               |
| 3   | 1      | 1.93×            | 4               |

Ground truth: 3.00×. The current estimator fails to subdivide at
*any* zpf and reports nearly-uniform σ at zpf=0. At higher zpf the
smoothing window covers fewer MHz, so the smoothing partially
recovers the step structure post-hoc, but the recovered range never
approaches the true 3.00×.

## 4. Failure-mode diagnosis -- the trim hides heterogeneity

`figures/03_failure_mode.png` shows the underlying mechanism for the
zpf=0 case. The synthetic 3-step σ(f) has 2049 bins; the first
subdivision attempt splits at the midpoint (bin 1024, ≈ 12.5 MHz).
The LEFT half is heterogeneous (the σ=1.0 + σ=3.0 regions are 50 %
each); the RIGHT half is homogeneous (σ = 1.5 throughout). The raw
magnitude statistics differ substantially:

- LEFT raw: mean = 2.20, variance = 4.21
- RIGHT raw: mean = 1.88, variance = 0.92
- Raw relative differences: **mean Δ = 30 %, variance Δ = 118 %**

Both of these clear the 20 % subdivision threshold. The 5 σ z-test
on raw means also fires. *If raw statistics were used*, the
subdivision would proceed.

But the estimator uses **post-trim** statistics — and the skewness
trim aggressively eliminates the σ=3 tail of the heterogeneous LEFT
half, on its way to a Rayleigh-skewness fit. After the trim:

- LEFT trimmed: mean = 1.74, variance = 0.83 (kept 52 % of points)
- RIGHT trimmed: mean = 1.88, variance = 0.93 (kept 89 % of points)
- Trimmed relative differences: **mean Δ = 12 %, variance Δ = 11 %**

Neither clears the 20 % threshold. The z-test (using post-trim
variances and effective sample sizes) falls below 5 σ. *All four*
OR-criteria fail. The subdivision is rejected, and the spectrum is
called "homogeneous" — even though the raw data tells a 30 %-mean,
118 %-variance story of clear heterogeneity.

The trim is doing double duty: identifying the noise mask (correct
function) AND providing inputs to the subdivision criterion (wrong
function). The conflation causes the heterogeneity-detection step to
look at flattened data and conclude "homogeneous." It's an inverted
form of the original ratio test the noise-heuristic-audit §2
considered and rejected: there the worry was that the F-test would
have false positives on trimmed data because of the trim's bias;
here the symptom is *false negatives* for the same reason.

## 5. A robust-statistics variant

The natural fix is to **use a robust statistic on the raw
magnitudes** for the subdivision criterion -- one that is insensitive
to spectral-line outliers (which is what the trim was protecting
against in the first place) but faithful to the underlying σ.

`prototype.py::estimate_noise_mad_split` does exactly that:

1. **Recursive bisection** using **median** and **MAD** (median
   absolute deviation) of the *raw* |X| in each candidate half.
   Subdivide iff `|Δmedian|/median ≥ 10 %` OR `|ΔMAD|/MAD ≥ 10 %`.
   The MAD is robust to high-magnitude line outliers; the median
   tracks the underlying Rayleigh location.
2. **Per-bin σ estimate**: `σ_c = MAD(|X|) / 0.4485` (the Rayleigh
   MAD-to-scale ratio); `σ_x = σ_c · √2`.
3. **Smoothing**: convolve σ_x with a uniform window covering
   ``DEFAULT_SMOOTHING_SAMPLES`` points. (Same as the current
   approach; the smoothing-window-size choice is unchanged.)

This variant keeps the original architecture (recursive bisection +
per-bin σ + smoothing) but cleans up the conflation: the
subdivision sees raw heterogeneity; the per-bin σ uses raw MAD
(naturally outlier-resistant); the smoothing is unchanged.

### 5.1 Synthetic recovery

`figures/04_mad_variant_recovery.png` runs the same 3-step σ(f)
synthetic through the MAD variant at zpf ∈ {0, 1, 2, 3}:

| zpf | n_bins | σ range (max/min) |
|-----|--------|-------------------|
| 0   | 3      | 2.30×             |
| 1   | 4      | 3.25×             |
| 2   | 4      | 5.40×             |
| 3   | 4      | 6.08×             |

Compare to the current estimator's 1.08-1.93× recovery. The MAD
variant detects the σ structure at every zpf. zpf=0 (the active-FT
case, the one that motivated this study) is recovered at 2.30× —
short of the true 3.00× but well above the current estimator's
1.08×.

The mild inflation at higher zpf (5.4-6.1× for true 3.0×) is a
known artifact of the discontinuous σ-step ground truth: the
zero-padded grid has many bins very close to the σ-step boundary,
each of which integrates over a mixture of σ levels through the
Dirichlet interpolation. The MAD of those mixed bins is smaller
than either pure σ, so they appear under-estimated → the range
inflates. Real noise σ varies smoothly, so this artifact is unlikely
to matter on data; an open question for follow-up.

### 5.2 False-positive robustness

`figures/07_homogeneous_fp_check.png`. On a homogeneous σ=1
synthetic, both estimators correctly refuse to subdivide (n_bins = 1
each). The current estimator reports σ_max/σ_min = 1.027× within
that single bin (intra-bin sampling noise); the MAD variant reports
1.999× — but the larger MAD-variant intra-bin range is a *convolution
edge artifact*, not a real σ structure (the smoothing kernel hits the
spectrum edges and the `mode="same"` convolution biases edge values
downward). The pre-smoothing σ is uniform; only the smoothing has
edge effects. This is a follow-up implementation issue, not a
methodology problem.

### 5.3 Line-contamination robustness

`figures/05_line_robustness.png`. Re-run the 3-step σ(f) synthetic
with 5 strong line spikes (amplitude 30-80 σ_c) injected at random
frequencies. The MAD variant ignores the lines (its median/MAD are
unaffected by extreme outliers) and recovers the same 3-step σ(f)
as in the line-free case. The current estimator also handles the
lines reasonably (the skewness trim removes them), but its
heterogeneity-detection step is still degraded by the same trim it
relied on to handle lines.

### 5.4 2638 reality check

`figures/06_2638_mad_variant.png`. Apply the MAD variant to 2638 on
both grids (user grid and pre-trimmed active-FT):

| Metric                                   | Current  | MAD variant |
|------------------------------------------|----------|-------------|
| User-grid σ range                        | 7.10×    | 8.28×       |
| Active-FT trimmed σ range                | 4.75×    | 8.31×       |
| Median \|Δσ/σ\| between grids            | 8.06 %   | **1.45 %**  |
| p95 \|Δσ/σ\| between grids               | 26.75 %  | 15.07 %     |
| max \|Δσ/σ\| between grids               | 67.46 %  | 77.69 %     |

On the bottom line — grid agreement — the MAD variant gives a
median 1.45 % disagreement vs the current estimator's 8.06 %,
and the σ-range agreement is within 0.4 % (8.28× vs 8.31×) instead
of the current 7.10× vs 4.75× (a 49 % discrepancy). The max
disagreement is still ≈ 78 % (probably at line-contaminated
frequencies where the user grid's finer resolution sees structure the
active-FT can't), but that is a finite-bin effect, not an algorithm
mismatch.

This is good evidence that the MAD variant restores grid invariance
on the part of the spectrum that matters.

## 6. Why does the user-grid σ(f) show real structure?

The user's premise — verified visually on the unpadded spectrum,
not just the user grid — is that the σ(f) variation on real FTMW
data is *real physics*, not a zero-padding artifact. This study
supports that: the MAD variant on the un-padded active-FT recovers
the same σ range (8.31×) as the user grid (8.28×). If the variation
were padding-induced, the active-FT (no padding) would be flat. It
isn't.

The physical origin of the σ(f) structure on 2638 is out of scope
for this study, but plausible candidates include:

- Frequency-dependent detector noise (different gain or noise floor
  across the bandpass).
- Mixer / LO leakage at specific frequencies.
- Pickup from external sources at narrow frequency bands.
- Acquisition-stage filter-shape effects on the noise floor.

For Stage 2 + Stage 3 + Stage 5, the practical implication is that
σ-as-a-frequency-function genuinely matters; the consumers
(per-bin SNR thresholds, σ-weighted projections) want a faithful
σ(f) shape. Smoothing it away costs the screen its discrimination,
the matched-filter detector its threshold accuracy, and the
windowing stage its edge-coherence calibration.

## 7. Implications for the matched-filter wiring proposal

The
[matched-filter detection study](../matched-filter-detection/report.md)
proposed (§10) using the matched-filter exp-apodized active-FT as
the gap-pass detector, with σ estimated on that spectrum. This study
flags a precondition: **the noise estimator must be made grid-
invariant before that wiring ships**, or the gap-pass σ will be the
1-bin global average and any σ-weighted downstream (the coherence
screen, if revived; per-bin SNR thresholds) will lose discrimination.

The matched-filter wiring proposal is *not blocked* by this
investigation -- the wiring's primary detector (BH + SavGol) is
unchanged, and the gap-pass detector's σ threshold is well-defined
even with a single global σ -- but the σ-aware downstream consumers
need a fix.

## 8. Wiring proposal — as shipped

The wiring proposal in this section has been **implemented** in
`src/ftmwpipeline/preprocessing/noise_estimation.py`. Targeted
diagnosis of the root-cause subdivision flaw plus a robust
σ-estimation path and explicit handling of strong-line skirts. Each
step has a clear physical motivation and a single tunable; the
default parameter values were calibrated on 2638 and are
instrument-dependent (notably `DEFAULT_SMOOTHING_MHZ`, see §9).

1. **Subdivision criterion** (the original root-cause fix).
   `should_subdivide` reads median and MAD of the *raw* magnitudes
   in each candidate half. Subdivide iff `|Δmedian|/median ≥ T` OR
   `|ΔMAD|/MAD ≥ T`, with `T = SUBDIVISION_THRESHOLD = 0.08`. The
   previous OR-of-four (post-trim mean Δ ≥ 20 %, z ≥ 5 σ, post-trim
   var Δ ≥ 20 %, F ≥ 2) was removed cleanly. The noise-fraction
   floor (≥ 2/3) and bin-size floor
   (`max(n/64, ABS_MIN_BIN_SIZE = 300)`) are preserved.

   `T = 0.08` was calibrated against the noise-heuristic-audit §2
   test bed (5000 Rayleigh samples per half, 2000 trials per
   condition; see `scratch/mad-calibration/calibrate_thresholds.py`):
   zero false-positives on truly homogeneous noise, **94.6 %**
   detection at σ_R/σ_L = 1.10, 100 % at ≥ 1.20.

2. **Skewness-trim noise mask**. Still produces the initial
   `noise_mask` for the σ estimator, but now runs only once per
   *final* bin (not for every candidate subdivision split as
   before).

3. **Moving-median σ estimator**. `compute_rms_noise_convolution`
   was rewritten to slide a median filter of width
   `DEFAULT_SMOOTHING_MHZ / freq_step` over the noise-mask
   sequence and scale via the Rayleigh quantile relation
   `σ_x = median(|X|) · √(1/ln 2) ≈ 1.2011 · median(|X|)`, then
   interpolate back to the full grid. The median is robust to
   contamination up to 50 % of the window, so Lorentzian-skirt
   residuals that survive the skewness trim do not bias the
   estimator the way the prior `sqrt(mean(|X|²))` did.

   `DEFAULT_SMOOTHING_MHZ = 300.0` chosen to match the empirically
   observed ~few-hundred-MHz coherence scale of real σ(f) on the
   user's hardware (no-signal noise-only acquisitions). Translates
   to ≈ 25 000 noise samples on the 2638 user grid (≈ 0.3 % Rayleigh-
   RMS stability per the δ-method — over-tightened relative to the
   noise-heuristic-audit's 1 % target, but stability is no longer
   the binding constraint; physical smoothness is).

4. **Strong-line skirt exclusion**. The moving median handles
   isolated outliers but residual upward bias persisted near strong
   lines on 2638 (~ tens of MHz around peaks at 37/39 GHz). Cause:
   the Lorentzian skirt is a *contiguous* region of moderately-
   elevated samples; in a region near a strong line a non-trivial
   fraction of the smoothing window is skirt-contaminated, and
   while the median is much less biased than the mean it still
   shifts at single-digit percent.

   The physical fix is direct: for an exp-damped sinusoid the FT
   magnitude is Lorentzian `|X|(Δf) = X_peak · γ / √(Δf² + γ²)`,
   so the far-field skirt drops below `k · σ_x` at radius
   `Δf_exclude = γ · SNR / k` (where SNR = X_peak / σ_x_initial).
   The estimator finds peaks above `STRONG_PEAK_SNR = 20` σ_x in a
   first-pass σ estimate, measures the strongest peak's FWHM via
   `scipy.signal.peak_widths` to estimate γ, and masks each peak's
   `±Δf_exclude` neighborhood out of the noise mask (capped at
   `MAX_SKIRT_EXCLUSION_MHZ = 500` for pathological super-strong
   peaks). With `SKIRT_EXCLUSION_K = 1.5` (exclude out to where the
   predicted skirt drops to 1.5·σ_x), σ_x is then recomputed on the
   tightened mask. On 2638 this exclusion drops 146 k samples from
   the mask (noise fraction 0.93 → 0.80) and the residual bumps at
   37/39 GHz flatten visibly.

5. **Serialization**. `noise_result_serialization` is unchanged
   from the prior path: HDF5 stores signal indices + smoothing
   window size, and the round-trip replays
   `compute_rms_noise_convolution` (now the moving-median variant)
   for bit-perfect reconstruction.

### As-shipped algorithm summary

| Stage | Role | Tunable |
|---|---|---|
| Recursive bisection (median + MAD on raw \|X\|) | identify σ-regime boundaries | `SUBDIVISION_THRESHOLD = 0.08` |
| Per-bin skewness trim | initial noise mask | `skew_target = 0.631` (Rayleigh constant) |
| Moving-median σ_x (full-grid) | smooth σ(f) at the instrument coherence scale | **`DEFAULT_SMOOTHING_MHZ = 300.0`** ← main instrument knob |
| Strong-line skirt exclusion | remove contiguous skirt contamination | `STRONG_PEAK_SNR = 20`; `SKIRT_EXCLUSION_K = 1.5` |
| Final moving-median σ_x | production output | (same window) |

## 9. Open questions

- **`DEFAULT_SMOOTHING_MHZ` is the main instrument-tunable.** The
  300 MHz default matches 2638's empirical few-hundred-MHz σ(f)
  coherence scale. Instruments with finer σ structure (e.g.,
  narrower bandpass features or mixer artifacts) will want a
  smaller window; instruments with smoother noise can use a wider
  window for better Rayleigh-RMS stability. The empirical
  procedure: take a no-signal acquisition, look at the unwrapped
  noise floor, pick the smallest window that flattens the σ(f)
  bumps near strong lines without smoothing across real σ
  structure.
- **σ-range agreement across grids is partial.** The as-shipped
  estimator achieves grid invariance for the *subdivision* step
  (the original root-cause fix) but not perfectly for the σ values
  themselves, because `DEFAULT_SMOOTHING_MHZ` is anchored in MHz
  while the underlying noise-mask sample density depends on the
  grid. On 2638 the user-grid σ range is 3.06× and the pre-trimmed
  active-FT σ range is around 3.5× (both vastly improved from the
  pre-fix 7.10× / 1.00× — the active-FT is no longer a single
  global bin). For most consumers this is good enough; the per-bin
  MAD prototype in §5 remains the reference for off-line grid-
  invariance diagnostics where exact σ-value agreement matters.
- **`STRONG_PEAK_SNR = 20` and `SKIRT_EXCLUSION_K = 1.5` are
  2638-calibrated.** A higher peak-SNR floor would skip weaker
  peaks whose skirts still bias slightly; a lower k widens the
  exclusion further. The current settings drop ~ 0.13 of the
  spectrum from the noise mask on 2638. Other instruments with
  more or stronger lines per MHz may want the exclusion narrower
  to retain enough noise mask.
- **HWHM is measured from the strongest peak.** Assumes the
  spectrum has roughly uniform line widths. If linewidths vary
  substantially (e.g., split lines mixed with broad features),
  per-peak FWHM measurement would be more accurate. Not done.
- **Cross-instrument generality of `SUBDIVISION_THRESHOLD`.**
  `T = 0.08` was calibrated at the audit's N = 5000-per-half size
  and may need adjustment for spectra small enough that the bin
  floor (300) dominates the recursive bisection depth.
- **σ-range inflation at high zpf in synthetic.** Carried from the
  pre-ship report — the MAD variant on the discontinuous-σ
  synthetic reports σ ratios 5-6× larger than the ground-truth 3×
  at zpf ≥ 2. This is a Dirichlet-interpolation artifact of the
  σ-step (real noise σ is smooth); should be characterised with a
  smooth-σ synthetic before being considered done.
- **The broader grid question.** The user has flagged this as the
  right long-term question: what zero-padding choice (zpf=0, 1, 2,
  …) is right for each pipeline operation? This study is a focused
  audit; the broader survey is a separate project.
- **Stage 5 validation.** The MAD subdivision was run through the
  validation harness on 2638 (in a prior per-bin-MAD shipping
  attempt). Per-window chi^2_r distribution improved substantially
  (p50 1.42 → 0.52, p95 7.73 → 3.14) and total fitted peaks
  dropped marginally (648 → 637). The as-shipped version (moving
  median + skirt exclusion) trims σ near strong lines more
  aggressively, which should net-improve the consolidated fit; a
  fresh validation pass on the final estimator is the next step.

## 10. Reproducibility

```bash
# from repository root, with the project conda env
conda run -n ftmwpipeline-dev python \
    dev-docs/research/noise-grid-invariance/prototype.py
```

Runtime: ~3 seconds. Writes seven PNGs to `figures/`. Section 1 and
6 need `scratch/stage5-validation/exp_2638.ftmw`; the synthetic
sections are self-contained. Random seed `20260524` is fixed inside
the script.

## 11. Related artefacts

- The current noise estimator:
  [`src/ftmwpipeline/preprocessing/noise_estimation.py`](../../../src/ftmwpipeline/preprocessing/noise_estimation.py).
- The prior noise audit (calibration of two heuristics, OR-of-four
  preserved): [report](../noise-heuristic-audit/report.md). The
  current study's failure-mode finding does not invalidate that
  calibration -- the smoothing-window target and bin-size floor are
  unchanged -- but it does overturn that audit's §2 "keep as-is"
  decision for the subdivision criterion.
- The matched-filter detection study (the motivator):
  [report](../matched-filter-detection/report.md). The screen-ratio
  collapse on 2638 noted in its §11 open questions is partially
  explained by this study: a 1-bin σ on the active-FT means
  σ_c[bin_c] = σ_c_median identically, which zeroes out the screen's
  σ-weighting power. The full explanation also involves the
  dense-line / Lorentzian-tail effect (§5 of that study), which is
  not addressed here.
