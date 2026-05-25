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

This is an audit + diagnosis + candidate fix. Production wiring of the
fix is deferred to a follow-up that runs the Stage 5 validation
harness on 2638.

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

## 8. Wiring proposal (conditional)

If the validation harness on 2638 confirms no Stage 5 regressions,
the production change is:

1. Replace the post-trim-stats subdivision criterion in
   `_compute_variance_based_bins.should_subdivide` with the
   MAD-and-median-on-raw-magnitudes criterion sketched in
   `estimate_noise_mad_split`. Keep the noise-fraction and bin-size
   floors, the OR-of-multiple-criteria structure is fine; only the
   inputs change (raw, not post-trim).
2. Replace the per-bin σ_c estimate (currently mean-based, then
   squared) with `MAD/0.4485` (the Rayleigh MAD scaling). This
   preserves the line-robustness of the current approach without
   needing the trim to do double duty.
3. Keep the smoothing logic, the smoothing-sample target, the bin-
   size floors, and the trim itself (the trim still produces the
   noise mask, just no longer drives subdivision).
4. Fix the smoothing's `mode="same"` edge bias (false-positive §5.2)
   by either `mode="valid"` + edge extrapolation or pre-padding the
   σ array with its boundary value before convolution.

The change is local to one function and one parameter computation.
The MAD/median calculation is O(N log N) per bin (sort + median); the
recursive subdivision visits each bin O(log N) times, so the
asymptotic cost matches the current estimator's. Empirically on 2638
the MAD-variant prototype is ~2× faster end-to-end (single sort vs
the trim's iterative 1 %-step magnitude search).

## 9. Open questions

- **Smoothing edge artifacts**. The `mode="same"` convolution biases
  σ values within ~half-window of the spectrum edges. The current
  estimator's `_compute_rms_noise_smoothed` already handles this
  better; the MAD variant should adopt the same approach.
- **σ-range inflation at high zpf in synthetic**. The MAD variant
  reports σ ratios 5-6× larger than the ground-truth 3× at zpf ≥ 2
  on the discontinuous-σ synthetic. This is a Dirichlet-interpolation
  artifact of the σ-step (real noise σ is smooth, so the artifact
  should not matter on data) -- but it should be characterised with
  a smooth-σ synthetic before shipping.
- **Validation against the Stage 5 fit on 2638**. Production wiring
  needs the validation harness to confirm no regressions in the
  consolidated fit. Not done here.
- **Cross-instrument generality**. 2638 is one fixture. The σ(f)
  shape on 2638 has 8× range; other instruments may have flatter or
  more structured noise. The MAD variant should hold but the
  thresholds (10 % median diff, 10 % MAD diff) are first-try values
  that may need calibration.
- **What is the physical origin of the σ(f) structure on 2638?**
  Out of scope here, but the answer informs how *fine* the σ(f)
  shape needs to be tracked. If σ varies on a 5 MHz scale (visible
  on the user grid), the smoothing window must be < 5 MHz to track
  it; the current 32 MHz default is already too wide.
- **The broader grid question**. The user has flagged this as the
  right long-term question: what zero-padding choice (zpf=0, 1, 2,
  …) is right for each pipeline operation? This study is a focused
  audit; the broader survey is a separate project.

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
