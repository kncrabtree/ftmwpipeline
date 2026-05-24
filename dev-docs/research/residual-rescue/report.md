# The residual-rescue chain: catching peaks the initial fit missed

A research report on the residual-rescue subsystem of the FTMW
processing pipeline's per-window fitting stage. The conservative
add-one-peak loop that does the initial per-window fit only seeds
peaks the detection stage detected; any real line that detection
missed (or that the knockout test dropped during the conservative
fit) leaves an above-noise residual the initial fit cannot explain.
The rescue is a second pass — and a chain of passes — that detects
peaks in that residual, screens them with a sliding phase-coherence
test, folds them into a joint refit, and prunes the result against
a per-site AICc-with-`n_eff` test. This report explains why each
piece works the way it does, what the alternatives were, and what
the 2638 fixture told us at each step.

The implementation overview lives in
[`../../planning/stage5-residual-rescue.md`](../../planning/stage5-residual-rescue.md);
this report is the algorithmic-choice provenance.

## 1. The starting problem

The detection stage runs on the magnitude spectrum, with no notion of
phase coherence. Real molecular lines come back as candidates; so do
contributor skirts, noise excursions, and (occasionally) artifacts of
the detector's own filtering. The fitting stage trusts that list,
fits the lines it claims to be there, and stops.

When the list is *under-complete* — typical for windows containing a
hyperfine multiplet, a partially-resolved doublet, or a feature
whose strongest peak masked the others in the detector's neighbour-
suppression — the conservative loop converges to a model that
absorbs the missing peaks into the modelled ones. The signature is
visual: a window whose residual is far above the noise floor with
**phase-coherent** structure under the model peaks. The initial fit
does not know it is wrong; chi² is what it is, the F-test signature
says K is what it is, the noise floor is somewhere down there.

w198 on the 2638 fixture is the canonical case. The detector
delivered 3 candidate offsets; the initial fit pegs K=2 with tau at
the apodization lower bound (1 µs vs the dataset consensus of ~3
µs), reduced chi² 627. Visually obvious that there are 5+ real
lines in the window; structurally invisible to the conservative
loop without a second pass.

## 2. The B-loop chain

The rescue strictly separates the initial fit from the rescue pass:

1. Compute `residual = data − model(initial.peaks, initial.tau)`.
   Done once; the initial fit is never touched again.
2. Detect candidate peaks in the residual.
3. Drop candidates that fail the sliding phase-coherence check
   (§3).
4. Run a second conservative fit on the **residual itself**, with
   tau frozen at the rescue tau (§2.1). The result's peaks are
   exactly the lines the rescue added; the initial fit's peaks are
   not in this returned fit.

The chain — `rescue_and_consolidate` — iterates:

1. Initial fit → rescue₁.
2. **Joint refit** of `initial.peaks + rescue₁.peaks` against the
   original data, with all parameters thawed and the tau initial
   value pulled from the rescue.
3. Knockout sweep over the joint fit; merge cleanup; iterative
   AICc cleanup. The surviving subset is the consolidated fit.
4. The consolidated fit becomes the "current" for round 2; repeat
   against the new residual. Terminates when the rescue accepts no
   new peaks, the joint refit fails to converge, knockout would
   empty the model, or `max_rescue_rounds` is reached.

### 2.1 Why option B (joint refit per round) over option A (residual-only recursion)

The alternative we considered was option A: recurse the rescue on
the shrinking residual without a joint refit. Cheaper per round
(one conservative_fit, no joint LSQ); the rescue's peak set is
naturally additive and clean.

The structural argument against option A is that each round operates
against a model that is *more wrong* than the consolidated state
would be. The initial fit's wrong tau and wrong amplitudes — the
same wrongness that produced the residual in the first place —
propagate into round 2's "residual". A residual computed against a
broken initial fit contains structure that real lines in the data
do not produce. The rescue can chase that structure, accept noise
candidates, and have no mechanism to ever revisit them.

Option B's joint refit makes each rescue round's "what is left"
question well-defined: the residual is measured against a
properly-jointly-fit baseline that already incorporates every line
the chain has accepted so far. The cost is the joint refit per
round (~K² LSQ scaling, dominated by the conservative_fit cost) and
a small constant for the knockout sweep.

### 2.2 The apodization-override-aware tau

Real molecular lines in one experiment share a tau ≈ the applied
apodization (the canonical Stage 1 `expf_us`). The rescue must use
the right line-shape width — the phase-coherence basis depends on
it (§3) and the joint refit's warm start depends on it.

The shipped rule:

- Default to `initial.tau_us` — it's the LSQ-converged value for
  the real lines this window contains.
- **Override** to `tau_apodization_us` when `initial.tau_us` is
  within 5% of the lower bound
  (`tau_apodization_us / max_decay_factor`). That's the signature
  of a broken initial fit: LSQ over-narrowed tau to absorb
  unmodelled-peak residual. Using that broken tau as the coherence
  basis under-projects real peaks. The apodization is the right
  physical default.

In the 2638 validation set this fires on w198 (initial tau pegged
at 1 µs; apodization is 5 µs). The override propagates 5 µs into
the joint refit's warm start, which then converges to 2.3 µs on
round 0 — well above the 1 µs peg, escaping the pathological
basin in one step. The remaining gap to the dataset consensus of
~3 µs is the open work covered by the cross-fixture validation
doc's dataset-wide tau calibration section.

### 2.3 The failsafe diagnostic

Each round records `RescueRoundDiagnostics`; the
`n_pruned_rescue_origin` counter is the **failsafe diagnostic** —
a nonzero value means the knockout sweep dropped a peak the
rescue had just added. That is the signal that the joint refit
may not be escaping a pathological basin; the rescue's
contribution is being undone immediately. The current code logs
this without acting on it. A future fallback to option A on
windows that trip this signal repeatedly would key off it.

## 3. The phase-coherence projection

Magnitude alone cannot tell a real Lorentzian peak from a phase-
rotation artifact (a coherent-residual feature whose phase rotates
across a few bins, producing a magnitude bump that no single
Lorentzian can fit). The projection test asks the right question:
*is this magnitude peak the projection of a Lorentzian?*

For an isolated candidate at offset `f₀`, build the unit-amplitude
Lorentzian basis `basis(f) = h_T(f − f₀, τ, T)` and compute the
sigma-weighted complex projection

```
A_complex = Σ_f w_f · conj(basis(f)) · residual(f)
          ─────────────────────────────────────────
                  Σ_f w_f · |basis(f)|²
```

with `w_f = 1 / σ_c(f)²`. This is the closed-form solution for an
amplitude+phase-only fit with offset and tau frozen — exactly what
a Stage 5 `fit_window` call would converge to if only those two
parameters were free.

The "coherent SNR" at the peak is `|A_complex| · |basis(f₀)| / σ_c`.
Compared to the detected magnitude SNR `|residual(f₀)| / σ_c`:

- **Real Lorentzian peak in clean isolation**: ratio ≈ 1.
- **Real Lorentzian peak sitting in a neighbour's skirt**: ratio
  partially suppressed by leakage from the neighbour's (slightly
  mis-fit) Lorentzian tail. Contamination scales with the
  neighbour's Lorentzian magnitude at the candidate's offset —
  ~50% at 1 FWHM separation, ~6% at 5 FWHM.
- **Phase-rotation artifact**: ratio ≪ 1; the complex projection
  cancels across the rapidly-rotating phase.

### 3.1 Why the sliding threshold

The original design used a uniform ratio threshold (0.5). It did
the wrong thing on both ends: too strict on candidates near
fitted neighbours (legitimate peaks rejected because their
projection is inevitably contaminated), and arguably too loose on
far-isolated candidates (a 50σ candidate with 25σ coherent
projection passing despite half its magnitude being incoherent).

The shipped scheme ramps the threshold by neighbour proximity in
three bands:

- **Δ < cluster_threshold_fwhm × FWHM** (default 1.0 FWHM):
  defer entirely. A sub-cluster candidate is either a real blend
  the blend-aware seeder should handle, or a phase artifact the
  basis cannot disambiguate from a blend.
- **cluster_threshold_fwhm ≤ Δ < isolated_threshold_fwhm × FWHM**
  (default 1.0–5.0 FWHM): linear ramp from `close_threshold`
  (default 0.2) at the cluster boundary up to `isolated_threshold`
  (default 0.8) at the isolated boundary. The lenient close end
  expects projection contamination from the neighbour; the strict
  far end has no such excuse.
- **Δ ≥ isolated_threshold_fwhm × FWHM**: full
  `isolated_threshold`.

A more principled functional form (interpolating by the Lorentzian
skirt magnitude `|basis(Δ)|²` directly, since that tracks the
contamination level exactly) is an obvious follow-up; the linear
ramp is easier to tune and reason about as the first version.

### 3.2 Neighbour distance includes fitted peaks

The proximity check uses `min(distance to nearest other candidate,
distance to nearest peak in current_fit)`. Including the fitted
peaks is the structural fix for the w198-style case where a real
residual peak sits ~1.1 FWHM from a freshly-fit line — that
candidate was previously held to the uniform threshold,
projection-contaminated by the fitted neighbour, and rejected.

### 3.3 Shape-error-aware sigma inflation

A separate piece of the screening pipeline, necessary because the
merge gate alone left the rescue stuck in limit cycles on strong-
line windows (rescue keeps re-detecting irreducible shape residual
as candidate peaks; merge collapses them; next round re-detects
them; …).

The fix builds a position-dependent effective noise floor

```
σ_eff(f) = √( σ_c² + (ε · |current_model(f)|)² )
```

threaded as the `shape_error_epsilon` parameter (default 0.0 =
behaviour-preserving). The rescue's detector + phase-coherence
filter see the inflated sigma; the LSQ inside `conservative_fit`
keeps the canonical sigma — inflation is a screening tool, not a
fitting one. A per-bin post-filter is required because
`find_residual_peaks` uses the *median* sigma for scipy's
`find_peaks` height threshold, so a few-bin local inflation doesn't
shift the global gate; the post-filter checks each detected
candidate against `snr_threshold · σ_eff[bin] / √2` and drops those
that fail at the inflated floor.

ε is a **per-dataset constant** measured from the chi²_r vs SNR²
regression on the post-rescue fits. The 2638 fixture's calibrated
value is 0.05 (5% per-bin residual at the line center). The
*per-bin* ε is roughly 4× the *chi²_r-aggregated* ε because chi²_r
sums over many bins and divides by full-window dof; the per-bin
value is what the sigma inflation needs. This factor will likely
differ between instruments; the cross-fixture validation doc
covers the calibration protocol.

## 4. The duplicate-pair problem

While building the per-window detail / audit visualisations,
inspecting w148 made the central pathology concrete. The strong
doublet that should be 2 real lines was being fit as **4 peaks,
arranged as two pairs of two**, with each pair's members separated
by *less than the FT point spacing* and converging to roughly
equal amplitudes:

| peak | freq (MHz)       | amplitude (µV)  |
|---|---|---|
| A | 31848.5948(25)   | 4.38(22)        |
| C | 31848.5551(22)   | 4.641(187)      |
| B | 31849.7032(24)   | 4.113(135)      |
| D | 31849.65473(185) | 4.743(143)      |

A/C separation: 0.040 MHz. B/D separation: 0.049 MHz. Point
spacing on this fixture is ~0.05 MHz. With tau ≈ 4 µs, FWHM ≈
0.08 MHz — both pairs are within 1 FWHM and at the resolution
limit.

The chain manufactured duplicate peaks at the same physical line.
The per-peak knockout test (which compares the K-peak model to a
freeze-others "remove peak i" model) showed `p_knockout < 1e-15`
for every duplicate — each pair member was *individually*
supported even though the pair was physically redundant. Knockout
cannot ask the merge question (it sees a single peak's
contribution against the rest, not a pair's joint contribution
vs a single merged line).

The merge cleanup is supposed to catch this case — greedily
merging close adjacent pairs when the (K-1)-peak refit is
statistically indistinguishable from the K-peak fit — but the
F-test gate it used was systematically biased toward "really
distinct" verdicts, even on the physically-impossible w148
duplicates.

### 4.1 The F-test bias

All three Stage 5 hypothesis tests at the time shared the same
F-statistic shape:

```
F = (Δχ² / Δdof) / (χ²_K / (n_data − n_params_K))
```

All three used the full window `n_data` (~100–300 bins) in the
denominator. But the *informative* bins for distinguishing a K-peak
model from a (K±1)-peak model live within ~1 FWHM of the peak in
question — a handful of bins. The 200-bin denominator is mostly
noise far from the feature, which inflates the F-statistic for
marginal improvements and structurally biases every gate toward
*accepting the more-complex model*:

- **Conservative loop**: weak peaks pass the accept gate because
  adding three parameters buys ~6 in δχ² (~2.4σ of evidence) —
  but against a 200-bin denominator the p-value clears
  significance.
- **Knockout**: every duplicate-pair peak in w148 shows
  `p_KO < 1e-15` because removing a duplicate leaves a half-fit
  line — a large δχ² against a huge denominator. Individually
  supported even when physically redundant.
- **Merge**: w148's A/C and B/D pairs did not merge — same
  cause, evaluated head-on.

The user's framing: *fitting a narrow feature with 2 independent
peaks should bear a high burden of statistical proof*; *we have
artificially low p-values for weak peaks, likely because there are
so many points in a window*. The fix needs to operationalise both:
the burden grows because the denominator shrinks to the
informative bins only.

### 4.2 The AICc-with-`n_eff` rule

The shipped solution replaces the raw `n_data` with an **effective
sample size** `n_eff` that weights each bin by some function of
local model magnitude (Fisher-information-density flavour) and
feeds it into the small-sample-corrected AIC criterion:

```
AICc = 2k + n_eff · log(χ² / n_eff) + 2k(k+1) / (n_eff − k − 1)
```

Substituting `n_eff` everywhere — log-likelihood term and
correction — keeps the formula self-consistent: smaller `n_eff`
simultaneously rescales how much χ² improvements count for AND
grows the small-sample correction. Hybridising (`n_data` in the
log term, `n_eff` in the correction) is not a derivable AICc form.

The gate decision: accept the more-complex model iff its AICc is
strictly less than the simpler model's, on the **same `n_eff`**
(computed once from the more-complex model's magnitude basis, the
one with strictly more information about which bins discriminate
the comparison). REJECT-on-tie matches the conservative direction
on weak evidence: when both AICc values are tied (either both
finite-equal or both `+∞`), preserve the simpler model.

Three call sites: merge cleanup (K-vs-(K-1)), knockout test
(K-vs-(K-1)), and the conservative add-one-peak loop's accept gate
plus the blend-aware seeder's K=2/K=3 escalation
(K-vs-(K+1)). The same rule at all three sites.

The F-test stays as a familiar diagnostic statistic, recorded on
the persisted audit trail (`AddStep.p_value`, `KnockoutInfo.p_value`)
alongside the new gate fields (`n_eff`, `aicc_delta`).

### 4.3 The merge two-tier structure

A pure AICc-with-`n_eff` test at the merge site has a structural
blind spot. On narrow features where `n_eff < k + 1` for *both* K
and K-1, AICc returns `+∞` for both → tied — and on real close
pairs, ties at `+∞` are the typical regime, not a degenerate
edge case. The shipped form is a two-tier gate:

- **Tier 1 — sub-resolution structural merge.** Pairs at
  separation < `structural_merge_factor · FWHM` (default 0.5 FWHM)
  merge unconditionally; no AICc test. The Lorentzian-only model
  physically cannot distinguish these from a single peak, so any
  LSQ convergence at this scale is a numerical artifact. Catches
  the w148 / w269 duplicate-pair pathology.
- **Tier 2 — above-resolution AICc-with-`n_eff` gate.** Pairs at
  `structural_merge_factor · FWHM ≤ separation < merge_separation_factor · FWHM`
  merge only when AICc(K-1) is strictly less than AICc(K). Ties
  preserve the K-peak fit.

Tier 2 is currently disabled by setting
`merge_separation_factor` equal to `structural_merge_factor` (both
0.5). The reason is in §6.

## 5. The `n_eff` weighting question

The first cut used `kish_mag` (Kish on `|model(f)|`, i.e.
`n_eff = (Σ|model|)² / Σ|model|²`). It worked at the merge and
knockout sites — the K-vs-(K-1) AICc divergence at small `n_eff`
falls through to "preserve K", which is the conservative direction
on weak evidence.

It failed at the conservative-loop K-vs-(K+1) site.

### 5.1 The over-rejection failure mode

The conservative loop's gate evaluates `AICc(K+1) < AICc(K)?` to
decide whether to add a peak. With `n_eff` computed from the K+1
trial's magnitude basis, `n_eff` is typically a few times the
per-peak FWHM-in-bins — 5–8 on the 2638 fixture. The K+1 model has
`3(K+1) + 1` parameters; AICc requires `n_eff > k + 1`. For K=1 →
K=2, the K+1 model has 7 parameters and needs `n_eff > 8`. With
`kish_mag` returning 5–8 on a 2-peak window, the threshold is
brushed right at the edge.

The asymmetry that bites: the simpler K model has fewer
parameters (4 for K=1) and needs `n_eff > 5`. When `n_eff` sits
between 5 and 8 — as it often does on real 2-peak windows —
**AICc(K+1) diverges to `+∞` while AICc(K) is finite**. The gate
asks "+∞ < finite?" → False → reject the addition, **even when
the chi-squared drop is overwhelming** (p_F values of 10⁻¹² were
observed).

The validation surfaced this on the 2638 fixture's clean controls.
With `kish_mag` n_eff:

- w63 (K=3 clean control): K=1 → K=2 trial has chi² drop 348 →
  260, p_F = 7.8 × 10⁻¹², n_eff = 5.20 (AICc(K=2) = `+∞`) →
  REJECT. Real second peak locked out.
- w64 (K=2 clean control): same story. K=1 → K=2 rejected
  despite p_F = 10⁻¹⁰.
- w198 (K=2 doublet): rescue's joint refit cycles produce K=5
  configurations whose K+1 trials at K→K+1 all return `+∞` for
  AICc, locking the chain at K=1.

The K-vs-(K-1) gates at merge / knockout do not have this failure
mode: the *simpler* (K-1) model diverges first as `n_eff` drops,
so the typical mixed case is "AICc(K-1) = `+∞`, AICc(K) finite" —
the gate rejects the merge / preserves the peak, which is the
desired conservative direction. The structural bias **helps**
those sites and **hurts** the conservative-loop site.

### 5.2 The information-weighted alternative

The user's intuition during the diagnostic session: counting bins
where the model is "significantly above noise" (~30–40 bins for a
strong line in a 60-bin window) should dominate the gate. The
shipped `kish_mag` was returning ~5 — far below visual
expectation.

Both `kish_mag` and `kish_mag_sq` weight bins by the model
magnitude (or its square), which concentrates `n_eff` at the peak
centre. The skirt — where two K-peak vs (K+1)-peak models
genuinely differ — is down-weighted.

The fix is a different weight function: per-bin **information**,
`w_f = log(1 + |model(f)| / σ(f))`. The argument is information-
theoretic: `log(1 + SNR)` is approximately the Shannon information
of a "signal present" detection at that SNR (`SNR / ln 2` for
small SNR, `log(SNR)` for large SNR — the log-Bayes-factor of
signal vs noise). A bin where the model is just above noise
contributes ~`log 2` of information; a bin at the peak centre
contributes log of the peak SNR.

Aggregating with the **perplexity** of the normalised
distribution rather than the Kish formula:

```
p_f = w_f / Σ w_f
n_eff = exp(−Σ p_f log p_f) = exp(H(p))
```

(Kish and perplexity are both effective-N measures — Kish is the
L² inverse-participation ratio `1 / Σ p²`; perplexity is the
entropic form. They differ in tail sensitivity: perplexity weights
tails more because the log compresses dynamic range.)

For our use case the weight function dominates over the
aggregator. Empirically on the 2638 diagnostic windows:

| window | K | peak SNR | kish_mag_sq | kish_mag | kish_log1p_snr | perplexity_log1p_snr |
|---|---|---|---|---|---|---|
| w63 | 3 | 5.7 | 8.2 | 31.1 | 54.4 | **71.1** |
| w64 | 2 | 3.0 | 5.5 | 25.6 | 41.7 | **63.9** |
| w148 | 1 | 139 | 3.9 | 17.2 | 55.9 | **60.0** |
| w198 | 2 | 159 | 5.6 | 12.2 | 38.8 | **45.5** |
| w269 | 5 | 579 | 7.9 | 27.6 | 83.4 | **86.6** |
| w271 | 2 | 157 | 8.2 | 30.2 | 77.9 | **82.4** |

(These are upper bounds — the trajectory `n_eff` at intermediate
K is smaller because the trial model isn't fully built up yet.)

Both log-space aggregators give nearly identical numbers (kish and
perplexity converge once log compresses dynamic range), and both
match the user's "30–40 significant bins" intuition. The shipped
kind is `perplexity_log1p_snr` because the entropic framing makes
the interpretation rigorous (`n_eff` is literally exp of the
differential entropy of the bin distribution); `kish_log1p_snr`
would have worked nearly as well.

### 5.3 The per-call-site default split

The conservative-loop gate's `n_eff_kind` defaults to
`perplexity_log1p_snr` (the `DEFAULT_CONSERVATIVE_N_EFF_KIND`
constant). The K-vs-(K-1) gates' default stays at `kish_mag_sq`
(`DEFAULT_N_EFF_KIND`) — the magnitude-concentrated weight
structurally protects against dropping real peaks at small `n_eff`
via the AICc-divergence-on-the-simpler-side mechanism.

`conservative_fit` carries both parameters:

- `n_eff_kind` (default `DEFAULT_CONSERVATIVE_N_EFF_KIND`): the
  conservative-loop accept gate and the blend-aware seeder's
  escalation gate.
- `knockout_n_eff_kind` (default `DEFAULT_N_EFF_KIND`): the
  final `knockout_test` sweep inside `conservative_fit`.

### 5.4 Information-weighted `n_eff` at the merge/knockout sites

The single-kind experiment ran `perplexity_log1p_snr` at all three
gates simultaneously. The survey distribution on the 2638 fixture
**improved** (median 1.31 vs 1.34 with the split, p95 3.35 vs
4.05). The downside was on w198: the rescue's joint refit produces
K=5 configurations with adjacent pairs at 0.55 and 0.33 FWHM
separations. With `kish_mag` at merge, both AICc(K) and AICc(K-1)
are `+∞` on these narrow pairs and tier-2 ties → preserve K → no
merge. With `perplexity_log1p_snr` at merge, both AICc values are
finite, tier-2 actually decides, and chi-squared evidence alone
says "merge" for some of these pairs.

The user's reading: AICc-on-finite-`n_eff` is making decisions on
real close pairs without a phase-degeneracy penalty backing it
up. The kish-merge degeneracy at small `n_eff` was an accidental
form of protection. The structural fix is the phase-degeneracy
penalty (planned as O5-11 in the fitting plan); until that lands,
the empirical no-op of disabling tier-2 entirely is the safest
posture (§6).

## 6. The merge tier-2 disable

`DEFAULT_MERGE_SEPARATION_FACTOR = 0.5`, equal to
`DEFAULT_STRUCTURAL_MERGE_FACTOR`. The two thresholds bracket the
tier-2 region; setting them equal disables tier-2 entirely. Pairs
in [0.5, 1.0] FWHM are not considered for merging at all;
knockout and `iterative_aicc_cleanup` handle peaks in that range
when they are duplicates and leave them alone when chi-squared
evidence supports the K-peak model.

The empirical case: on the 2638 50-window survey, the change is a
no-op. Total merges drop from 6 to 4; knockout-pruning rises from
22 to 24; the chi²_r distribution is unchanged (median 1.31, p95
3.35, max 6.68). Per-target windows including the clean controls
are bit-identical.

The structural case: tier-2's AICc-with-`n_eff` test, with the
information-weighted weight, makes decisions on real close pairs
*on chi-squared evidence alone*. Without a phase-degeneracy
penalty distinguishing real close pairs from duplicate-pair LSQ
artifacts, those decisions over-collapse real outer shoulders in
the 0.5–1.0 FWHM band (w198 is the demonstration case). Tier 2
becomes safe to re-enable once the phase-degeneracy penalty is in
place; the disable is structural conservatism, not a tuned default.

## 7. Validation on the 2638 fixture

The 50-window every-7th sample (50 windows surveyed out of 347)
is the standing distribution check. The targeted windows are
w63/w64 (clean controls), w148 (duplicate-pair pathology), w198
(apod-override + multi-line doublet), w269 (seed-duplicate K=5),
w271 (decoupled doublet), and w16/w104/w127/w337 (borderline
K=1-vs-K=2).

The distribution evolved across the AICc-gate work:

| | survey median | p75 | p95 | max |
|---|---|---|---|---|
| Pre-AICc work (the merge gate not yet wired) | ~1.4 | 2.3 | 7.7 | (~65, outlier-dominated) |
| Merge gate AICc + shape-error inflation | 1.06 | 1.65 | 3.23 | 5.65 |
| + Knockout gate AICc + iterative cleanup | 1.37 | 2.03 | 3.90 | 6.68 |
| + Conservative-loop gate AICc (kish at merge/knockout) | 1.34 | 2.02 | 4.05 | 6.68 |
| + Information-weighted `n_eff` at all three gates | 1.31 | 2.00 | 3.35 | 6.68 |
| + Tier-2 merge disabled | 1.31 | 2.00 | 3.35 | 6.68 |

The median sits ~30% above 1.0 — the residual elevation in HARD
windows is dominated by the lineshape-model deficit
(symmetric-Lorentzian fit to a slightly-asymmetric Doppler profile)
the shape-error ε calibration tracks but does not eliminate. The
cross-fixture validation doc covers the calibration in detail.

### 7.1 Target-window outcomes

After all the AICc work landed and the harness settled on
`perplexity_log1p_snr` at all three gates with tier-2 disabled:

| window | K_init → K_final | χ²_r init → final | reading |
|---|---|---|---|
| w63 (clean control, K=3) | 3→3 | 0.81→0.81 | preserved |
| w64 (clean control, K=2) | 2→2 | 0.59→0.59 | preserved |
| w148 (duplicate-pair) | 1→2 | 715→6.22 | duplicate-pair overfit eliminated; chi²_r at the shape-error floor for SNR=145 |
| w198 (apod-override) | 2→3 | 627→148 | the open follow-up — see §8 |
| w269 (seed-duplicate K=5) | 4→4 | 18.65→18.55 | initial-seeding duplicate rejected at the conservative-loop gate (no longer requires the rescue's iterative cleanup) |
| w271 (decoupled doublet) | 2→2 | 9.45→9.40 | stable |
| w16/w104/w127/w337 (borderline) | 1→1 | (low → low) | iterative cleanup rejects rescue's borderline second peak each round |

w269 is the user-named target outcome: K dropped from 5 to 4 on
the initial fit, eliminating the seed-duplicate pathology that
the rescue's iterative cleanup had been carrying.

## 8. The w198 limit case

w198 is the one observed limit case. With the conservative-loop
gate replaced by the AICc-with-`n_eff` test, the rescue chain
converges to K=3 at χ²_r=148; an earlier configuration of the
chain (F-test conservative-loop gate, magnitude-concentrated `n_eff`
at merge/knockout) happened to reach K=7 at χ²_r=2.92 on this
window by an LSQ-luck route that depended on the F-test's
acceptance bias at small-`n_eff` and the magnitude-concentrated
gate's tie-driven protection of close pairs — neither of which
is principled, but both of which happened to land in the right
basin for this window. The initial fit lands
at K=2 correctly. The rescue chain converges to K=3 at χ²_r=148.
The K=7 outcome is in a different LSQ basin altogether.

The diagnosis from the per-round trace:

1. Initial fit: K=2 with **tau pegged at 1 µs** (apodization
   lower bound). The K=2 model cannot represent 5–7 real lines,
   so LSQ narrows tau to broaden each modelled peak to absorb
   the unmodelled-peak residual.
2. Round 0 joint refit pulls tau to 2.24 µs (escapes the peg),
   places K=5. Adjacent pair spacings in FWHM units: 0.72, 0.55,
   0.33, 1.09. The 0.33-FWHM pair is sub-resolution → tier-1
   merge → K=4. A second tier-1 merge follows after refit → K=3.
3. Subsequent rounds: rescue keeps finding the missing
   structure, joint refit produces K=4 or K=5, tier-1 merges
   collapse, steady state K=3.

A control run with tau locked at 3 µs (the dataset consensus)
end-to-end gives K=4 at χ²_r=80. Still short of K=7, but closer
— the joint refit at the right tau places peaks at FWHM
separations the merge step does not collapse, and the K=4 fit
captures more of the real structure.

The remaining gap is the joint refit's *re-thawing* of tau. Even
when the initial fit holds tau fixed, `rescue_and_consolidate`'s
joint refit step inside each round fits tau freely from its warm
start (5 µs from the apod-override) and converges to ~2.9 µs.
That's a different basin from the 3.0 µs that produces the K=7
fit.

The structural fix is **dataset-wide tau majority-vote freeze**:
compute the consensus tau from strong-line windows, identify
windows whose initial fit lands far from consensus, and re-fit
those windows with `fit_tau=False` end-to-end — initial fit,
rescue conservative loop, joint refit, knockout, merge. The cross-
fixture validation doc carries the implementation sketch and the
per-window record-keeping.

w198's lesson for the rescue itself: the rescue chain's
capabilities are bounded by the LSQ basins of the joint refit;
fixing tau is upstream work that lifts that bound.

## 9. Outstanding open questions

Most of the open work has homes outside this document:

- **Phase-degeneracy penalty** (O5-11 in the fitting plan): the
  LSQ-side complement to the AICc gates, which would also let us
  re-enable the merge tier-2 with confidence.
- **Dataset-wide tau calibration** (cross-fixture validation
  doc): resolves the w198 limit case.
- **Borderline real-vs-noise on w16/w104/w127/w337-class windows**
  (cross-fixture Tier-3 ground-truth check): the single-fixture
  data cannot answer it.
- **Phase-coherence projection as a stage-3 quality filter**
  (peak-detection plan, future enhancement): the same primitive
  applied earlier in the pipeline.

What remains rescue-specific:

- **Sliding-coherence parameter calibration.** The shipped anchor
  pair (0.2, 0.8) was chosen on the original 15-window sample
  and is not empirically calibrated against a wider set. Likely
  follow-ups: replace the linear ramp with the Lorentzian-skirt
  functional form (`threshold(Δ) = high − (high − low) · |basis(Δ, τ, T)|²`,
  which tracks the contamination level exactly); sweep
  `(close, isolated)` against the regime where clean windows
  acquire spurious peaks.
- **Round-cap calibration.** A w132 5-round experiment showed
  the chain settling at round 3; `DEFAULT_RESCUE_MAX_ROUNDS=3`
  may be too low for some windows. Probably 5–7 once the rescue
  becomes a non-zero default.
- **Default flip from `max_residual_rescue_rounds=0` to the
  calibrated round-cap.** The rescue is a structural part of the
  fit, not an opt-in tweak; the 0 default is transitional.
- **Per-window cost monitoring.** Every rescue round adds one
  `conservative_fit` + one joint refit + a knockout / merge /
  iterative-cleanup sweep. For the production pipeline (~400
  windows) the B-loop may multiply Stage 5 wall-time by a small
  constant; worth measuring once the rescue is on by default.
- **Audit-trail persistence.** The consolidated
  `ConservativeFitResult` inherits the initial fit's
  `audit_trail`; the rescue rounds and joint refits emit
  `RescueRoundDiagnostics` but those stay live-only (off
  `SpectrumFit`). Once the rescue is on by default, persist the
  per-window `RescueEvent` list into `SpectrumFit` so the
  on-disk fit is reconstruction-complete.

## Reproducibility

Investigation was carried out with the scratch scripts under
`scratch/stage5-validation/` against `scratch/stage5-validation/exp_2638.ftmw`.
Key scripts (each rebuilds the per-window state via the harness's
`_run_window_rescue` reconstruction so the numbers compare
directly to the production fit):

- `diag_phase1_merge_gate.py` — sweeps the merge gate's parameters
  on the 50-window survey and target windows.
- `diag_phase3_conservative_gate.py` — survey + targets with the
  per-call-site default split.
- `diag_perplexity_merge_knockout.py` — sweeps `n_eff_kind` at
  the merge / knockout sites against the survey baseline.
- `diag_phase3_neff_kinds.py` — empirical comparison of `n_eff`
  kinds (Kish, perplexity, hard-threshold) on the diagnostic
  windows.
- `diag_w198_chain.py` — per-round trace of w198 under default
  vs tau-locked-at-3-µs configurations.
- `diag_phase3_calibration.py` — per-step audit-trail dump for
  the diagnostic windows.

The full validation harness is
`scripts/development/stage5-validation/generate_validation.py`. It
emits per-window `detail.png`, `audit-trail.png`,
`detail-rr<n>.png` trajectory snapshots, and the `report.md` /
`report-rr.md` rollups under `scratch/stage5-validation/window_NNN/`.

The 2638 fixture itself (FTMW data plus the post-Stage-4 plan and
fit) lives at `scratch/stage5-validation/exp_2638.ftmw`. The
fixture is BlackChirp data with 632k active samples, 12.65 µs
acquisition, 5 µs apodization, and ~347 windows after Stage 4
planning.
