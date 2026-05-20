# Plan: Noise-estimation follow-ups

Status: **open**. Scope is the noise-estimation stage in
`src/ftmwpipeline/preprocessing/noise_estimation.py`. The algorithm has
been audited; it is **functionally correct** and visually produces sensible
per-point RMS noise on the 2638 fixture, and a separate perf change
vectorising its inner skewness-trim loop will land alongside this document.
Registered in [`../ROADMAP.md`](../ROADMAP.md).

This document collects the **load-bearing heuristics whose constants are not
justified anywhere**. None of them are wrong; several are pragmatic choices
that work. The point of recording them is to make it cheap to come back
later — when the noise stage gets its own strategy spec or its own research
report — and replace each with a principled criterion.

## Algorithmic context

The noise stage produces a per-point RMS noise estimate `rms_noise[k]` by:

1. **Recursive bisection** of the magnitude spectrum into bins. A region
   is split when its two halves are statistically distinguishable in mean
   or variance, *and* both halves still have enough non-line points.
2. **Skewness-targeted trimming** inside each final bin. The magnitudes
   of a noise-only region in this experiment class are Rayleigh-distributed
   (the magnitude of complex-Gaussian post-FT noise), and Rayleigh
   skewness is the analytic constant ≈ 0.6311. The bin is trimmed from
   the top in 1% rank steps until the kept points' skewness drops below
   that target; the kept points are the noise mask for that bin.
3. **Smoothed RMS** computed by convolving the squared kept-magnitudes
   with a uniform window and square-rooting, then interpolated back to
   the full grid.

The Rayleigh skewness target is principled. The convolution-based RMS
estimator is fine. The heuristics below are the rest.

## Heuristics worth revisiting

### 1. Bin-split decision is an OR of four thresholds

`_compute_variance_based_bins.should_subdivide` decides to subdivide a
region only if **both halves have enough noise points** *and* the two
halves are "significantly different" in mean or variance, where
"significant" is **any** of:

- Means differ by ≥ 20% (relative).
- Means differ by ≥ 5σ on a pooled-SE z-test.
- Variances differ by ≥ 20% (relative).
- Variance ratio F ≥ 2.

Why it works: any one of those firing is enough, so the split is biased
toward over-subdividing when noise really does vary across the spectrum
— which on real spectra is the right error direction (under-subdivision
would merge noisy and quiet regions, biasing both estimates).

Why it deserves attention: four OR'd thresholds with four magic numbers
(20%, 5σ, 20%, 2.0) is hard to reason about. The pooled-SE z-test and
the F-test both compute formal p-values, but the thresholds are not
calibrated against any false-split-rate target. The "20% relative"
clauses are dimensionless heuristics layered on top of the formal tests,
present apparently to *override* the formal tests when they would say
"no significant difference" but the relative gap is still large.

Open: is there a single calibrated criterion — say, a Welch's t-test on
means combined with an F-test at a chosen false-positive rate — that
produces equivalent or better bins on representative real data?
Calibration would need a small synthetic + 2638 sweep counting splits,
final-bin sizes, and noise-fraction statistics.

### 2. Smoothing window default = 2 × average bin width

The default `smoothing_window_mhz`, when not supplied, is computed as
`2 × (freq_range / n_bins)`. It is converted to a point count for the
moving-average; the implementation also clamps the point count to at
least 10.

Why it works: 2× the typical bin width gives a smoothing window wider
than any individual bin, which both removes per-bin discontinuities in
the RMS estimate and supplies enough samples for the Rayleigh-RMS
estimator to be stable.

Why it deserves attention: the choice is dimensional but not derived
from any stability target. The right principled choice is sample-count
driven — pick `bl_bin` (in noise points) so the Rayleigh-RMS estimator
is stable to a chosen tolerance:

  Var(RMS_estimator) ≈ σ²·(4 − π) / (4·N_noise)

so for ±1% stability at 1σ, `N_noise ≳ 2000`. Converting that to a
point count in spectrum units requires the local noise fraction, which
the algorithm already computes per bin. The current default ignores
that information.

Open: parametrise the smoothing window by a stability target and the
local noise density, not the bin width.

### 3. The 1% skewness-search step

`_filter_by_skewness_cached` iterates `cutoff` from 0.0 in steps of
`inc = 0.01` (1% trim per iteration) and stops at the first cutoff
where the kept-data skewness drops below the Rayleigh target. The 1%
quantisation is arbitrary.

Why it works: a 1% step is finer than the noise-fraction guard
(`min_noise_fraction = 2/3`) needs, so the chosen cutoff rarely
matters at finer resolution.

Why it deserves attention: the perf-vectorised replacement landing
alongside this document computes the kept-data skewness analytically
for every possible per-point cutoff in O(N) using cumulative moments —
so a finer granularity is available essentially for free. The 1% step
remains for behavior parity with the prior implementation; whether
finer (e.g., 0.1% or per-point) granularity changes the final RMS in a
useful way is an open question.

Open: empirically test whether per-point cutoffs give a tighter or
visibly different noise mask, and if so adopt the finer granularity by
default.

### 4. Fallback rules in `_filter_by_skewness_cached`

Two fallbacks engage when the skewness target is not reachable inside
0–90% trimming:

- The "ran past `cutoff = 0.9`" branch keeps only the bottom 10% of
  the bin's magnitudes.
- The terminal "fallback to at least some points" branch keeps the
  first `max(1, N/10)` points by *position*, not by magnitude.

The latter is almost certainly never reached in practice (the loop
exits on either the `skew < target` branch or the `cutoff >= 0.9`
branch first), but its content is suspicious: it keeps the first
1/10 of the bin by *position*, which would systematically bias the
mask toward one end of the bin if it ever fired.

Open: confirm the position-based fallback is unreachable, remove the
dead branch if so, and replace the 10% magnitude floor with something
better-justified (e.g., the lowest `N_noise_target` points sufficient
for stable RMS as in §2 above).

### 5. `min_bin_size = max(N/64, 100)`

Bin sizes are bounded below by `1/64` of the spectrum length **or**
100 points, whichever is larger. The 1/64 is a default
(`min_bin_fraction`) and is configurable; 100 is hard-coded.

Why it works: very small bins have unstable Rayleigh-skewness
estimates, so capping below ~100 points is sensible.

Why it deserves attention: 100 is undocumented in the function
signature. The number should be derived from the stability of the
sample skewness estimator at sample size N — the standard
result is `Var(skewness) ≈ 6N(N−1)/((N−2)(N+1)(N+3))`, which is
~6/N for large N. At N = 100 the skewness estimator has σ ≈ 0.25,
i.e. comparable to the gap between Rayleigh skewness (0.631) and
half-Gaussian skewness (~0.995) — borderline. At N = 200 it's σ ≈
0.17. The current minimum is on the edge of usefulness.

Open: lift the minimum to a sample-skewness-stability-driven default
(probably ~200) and document.

## Out of scope

- The convolution-based RMS estimator (`compute_rms_noise_convolution`)
  and its edge padding: small numerical detail; behavior verified.
- The Rayleigh skewness target: analytic, no calibration needed.
- The `min_noise_fraction = 2/3` lower bound: this one is a deliberate
  curation choice (force every bin to contain mostly noise points, so
  the mask is dominated by genuine noise rather than line skirts), and
  while it could be calibrated, it serves its purpose.

## When this gets picked up

Likely when (a) noise estimation gets its own `NOISE_ESTIMATION_STRATEGY.md`
that needs to justify each choice, or (b) a future processing-class
experiment exposes one of these heuristics as actively wrong (e.g., a
spectrum where the smoothing window or the bin-split criterion
mis-bins a real region). The perf change has already converted the
inner loop to a cheap calculation, so re-running the algorithm under
different choices is fast — no infrastructure barrier remains.

Cross-reference: the perf calibration that drove the inner-loop
rewrite is documented in the same commit that introduces this file.
A future research report on the noise stage would absorb this
follow-up document and the calibration measurements together.
