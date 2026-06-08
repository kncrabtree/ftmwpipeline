# Plan: Stage 2b — data-driven τ calibration via sliding-active-window STFT

Status: **Complete.** The method research (synthetic validation + the 2638
application), the production wiring (`stage2b_tau_calibration`, between Stage 2
noise and Stage 3 peak detection), and the LSQ cross-comparison have all
shipped.

- **Regression validation.** Carried by the committed integration suite,
  re-baselined to the scatter-noise production grid (the per-stage pinned-value
  tests for Stages 2b/3/4/5 plus the cross-interface suite).
- **LSQ cross-comparison.** The "why STFT, not LSQ histogram?"
  close-out lives in [`../research/stage5-tau-calibration/report.md`](../research/stage5-tau-calibration/report.md)
  § "LSQ cross-validation". Its § "Noise-reference robustness" extends it to
  the production scatter-noise model: the unbiased LSQ per-band τ (prior off,
  independent of the STFT noise reference) tracks the FID-tail STFT and not
  the lower scatter σ, settling that τ extraction keeps the FID-tail floor.
  Reproduced by [`../research/stage5-tau-calibration/lsq_noise_reference.py`](../research/stage5-tau-calibration/lsq_noise_reference.py).

The shipped stage is named `stage2b_tau_calibration` (a non-
disruptive prefix between `stage2_noise_result` and `stage3_peaks`);
a future broader rename pass may reshuffle the numbering for overall
consistency.

**Shape-fit solver (shipped beyond the original plan).** The per-bin
exp/gauss/voigt fits behind the bad-fit gate and the 3-way shape
recommendation no longer run a per-bin scipy `least_squares` multistart
loop. They run a **batched closed-form solver**: those magnitude decays
are linear in log space with polynomial regressors in segment-time
(`log|S| = logC - a/τ_L - (a/τ_G)²`), so a weighted log-linear solve seeds
all three with no trust region, then a few clipped Gauss-Newton steps refine
toward the linear-RSS optimum the AICc verdict uses (gauss/voigt multistart
over the τ-seed grid keeping per-bin best-RSS; exp is convex in log space →
single start). `stft_calibration(shape_solver="scipy")` keeps the per-bin
loop as the equivalence oracle. ~150× faster on the shape pass; across the
seven fixtures the `recommended_shape` is unchanged and the Gaussian-twin
`τ_G` matches the scipy reference on every fixture where it is consumed.

**Noise reference for τ extraction — settled (keep the FID-tail).** The
STFT classifier's noise floor is the FID-tail `σ_t`
(`estimate_sigma_time_from_tail`), *not* the Stage 2 `NoiseResult`. The tail
overestimates the true spectral noise (~3× above the scatter σ on 2638)
because the active-region tail still carries decaying signal — but that
inflation is *beneficial*: it acts as a stricter effective above-threshold
gate that keeps only well-determined on-line bins. Substituting the lower
(more physically accurate) Stage 2 scatter σ admits weak,
log-linear-high-biased bins in sparse bands and pulls the per-band majority
away from the independent LSQ-fit-and-histogram reference — on 2638 it
inverts the real frequency-dependent τ trend in the high band. The
`extract_tau_majority` `sigma_x_full` override exists for forensic
comparison only; the production path leaves it `None`.

Research artefacts: the consolidated
[`../research/stage5-tau-calibration/report.md`](../research/stage5-tau-calibration/report.md)
covers the method, synthetic acceptance (7 cases + pathological
corners), the 2638 application (`τ_maj ≈ 5.96 ± 1.59 µs` under the
production defaults), the LSQ-fit-and-histogram cross-validation, and
the polish design (`polish=True, polish_snr_cap=9`) that lands per-band
SNR-weighted majority τ within ±3.2 % of the LSQ per-band reference on
2638. The 2638 fixture shows a real frequency-dependent τ trend
attributed to W-band horn-coupling geometry.

Supersedes the "Dataset-wide tau calibration" §357-381 and
"Broken-initial-fit pathology and the majority-vote-freeze proposal"
§383-422 sections of
[`stage5-cross-fixture-validation.md`](stage5-cross-fixture-validation.md);
both will be reduced to a back-reference once this doc lands. Builds
on the w198 / tau-collapse findings in
[`../research/residual-rescue/report.md`](../research/residual-rescue/report.md)
§8.

## Shape-awareness and per-band routing (shipped beyond the original plan)

The original plan calibrated a single Lorentzian (exponential-decay) `τ_maj`.
Since then the stage grew a shape-aware twin and per-band routing, all shipped
and tracked in the line-shape planning docs — recorded here so this doc is not
read as exponential-only:

- **Gaussian τ_G twin (`stage2b_tau_G_calibration`).** A sibling stage runs a
  pure-Gaussian per-bin fit and extracts `tau_G_maj` (`extract_tau_G_majority`
  in `fitting/tau_calibration.py`), the Gaussian-envelope counterpart of
  `τ_maj`. Commits 6b793cf, a22ea0c. Detail in
  [`stage5-gaussian-shape.md`](stage5-gaussian-shape.md) and the Part-B
  provenance in [`stage5-voigt-deficit.md`](stage5-voigt-deficit.md).
- **Per-band τ routing.** `compute_band_majorities` produces per-band
  SNR-weighted majorities; Stage 5 routes each window to its band's τ
  (`per_band_tau` default True). Commits 83fb3b6, 6d6005f.
- **3-way shape recommendation.** `compute_shape_recommendation` returns a
  `ShapeRecommendation` (per-bin AICc vote over exp / gauss / voigt);
  `auto_recommend` (default True) stamps the verdict onto the file, and
  Stage 3 / Stage 5 read it (`recommended_shape`) to pick the τ basis and the
  fit shape. Commits 4518faa, a46da3a, 4abf454. Surface and resolver in
  [`stage5-fit-settings.md`](stage5-fit-settings.md).
- **Settings resolver.** The calibration knobs are resolved through
  `core/tau_calibration_settings.py` (`TauCalibrationSettings`) on the
  four-layer chain (commit 506df38; see
  [`settings-backfill.md`](settings-backfill.md)).

## Objective

Stage 2b replaces the former upfront exponential apodization (the
since-retired `expf_us`, 5 µs on the 2638 fixture) with a **data-driven,
fit-free τ calibration** that runs after Stages 0-2 and produces a single
global majority-vote `τ_maj ± σ_τ`. Subsequent Stages 3-5 consume `τ_maj` as:

- the gap-pass matched filter's `tau_basis_us` (Stage 3),
- the per-window LSQ seed for `τ0_us` and the centre of a
  bidirectional Gaussian-prior penalty (Stage 5),
- the rescue τ for `rescue_and_consolidate` and the locked value
  for the weak-window `fit_tau = False` path.

The calibration extracts τ via a **sliding-active-window STFT**:
zero-pad the full-length record and rotate which `T_w`-long
sub-interval contains the active samples, then read the magnitude
at every frequency bin across the resulting STFT frames. A real
molecular line at frequency `f₀` decays as `exp(-a/τ_mol)` vs. the
window start `a`; a clock spur stays constant; noise bins fail the
fit-quality gate. Per-bin exponential fits give a τ histogram of
thousands of bins, robust to single-window pathologies (blends,
shape error, fixed-contributor coupling) that complicate the
LSQ-fit-and-histogram alternative.

Stage 2b removed a regularizer (apodization) and replaced it
with a self-calibrated one (`τ_maj`). The hypothesis it rests on is that
molecular τ is a global property of the experimental geometry
(beam transit time, horn coupling, collisional / Doppler broadening
from the supersonic expansion), so it should be well-determined
from the STFT of any fixture with at least a handful of
above-threshold lines.

## Why this was a serious change, not a tweak

Apodization had done several things that removing it had to either replace or
validate independently. Inventory of removal consequences (now realized):

| consumer | how apodization was used | realized effect |
|---|---|---|
| Stage 1 persisted FT (`FID.preprocess`) | `expf_us` multiplied into FID before rfft; the apodization decay rate adds to the molecular decay rate and widens the *effective* magnitude FWHM the user sees | gone — peaks become **narrower** in magnitude (FWHM = 1/π·τ_mol vs 1/π·τ_obs, with τ_mol > τ_obs so FWHM_mol < FWHM_obs), sidelobes longer (no exponential suppression of the sinc) |
| Stage 2 σ estimator (`estimate_noise_scatter`) | scatter MAD — no explicit apodization parameter | the scatter estimator high-passes the magnitude and is immune to the leakage pedestal, so it re-converges without apodization (it superseded the adaptive MAD/median estimator entirely, which is retired) |
| Stage 3 primary pass | runs its OWN apodization (`DEFAULT_PRIMARY_WINDOW="blackmanharris"`) — independent of user's `expf_us` | unaffected structurally |
| Stage 3 gap-pass MF (`_mf_gap_spectrum`) | matched-filter uses `tau_basis_us = base_pp.expf_us or 5.0`; FWHM-in-bins calibrated against this τ; `_GAP_ACTIVE_ZPF = 2` keeps FWHM ≥ ~3 bins | switches to `tau_basis_us = τ_maj` (available after Stages 0-2 → STFT calibration → Stage 3); no fallback-to-5.0 path needed |
| Stage 4 S_coh (`DEFAULT_EDGE_THRESHOLD = 8`) | runs on Stage 1 persisted FT with Stage 2 σ in denominator; calibrated against the apodized spectrum's sidelobe-suppressed shape | **touched fraction will increase further** (already 11 % → 17.7 % across the recent Stage 2/3 rework); T_edge=8 calibration may need re-validation; strong-cluster grouping could over-merge (more long mega-windows like w302) |
| Stage 5 active-FT (`compute_active_ft`) | applies same `expf_us` as Stage 1; fit τ is τ_obs = combined | applies no apodization; fit τ is τ_mol |
| Stage 5 `tau_bounds` (`derive_window_fit_constraints`) | `(τ0/k, min(τ0·k, τ_apod))` — `τ_apod` is the **hard upper bound** on the fit τ (since τ_obs ≤ τ_apod for any τ_mol > 0); also clamps how *narrow* the fitted line can be | becomes `(τ_maj/k, τ_maj·k)` centred on `τ_maj` (or `(τ_maj − N·σ_τ, τ_maj + N·σ_τ)` — see §Bidirectional penalty) |
| Stage 5 `tau_penalty_reference` | one-sided penalty `sqrt(λ)·max(0, (τ_ref − τ)/τ_ref)`. Fires when LSQ drives τ below τ_ref. Mechanism: when too few peaks are modelled, LSQ buys χ² by driving τ DOWN (BROADENING modelled lines so they absorb the missing peaks' residual); the penalty pulls τ back UP toward τ_apod (= narrower, physical lineshape) | replaced by bidirectional Gaussian-prior penalty around `τ_maj`. Two failure modes are now possible — τ-collapse (DOWN, over-broaden to absorb residual) and τ-runaway (UP, over-narrow toward the clock-spur basin) — and only `τ_maj` anchors the physical-decay basin |
| Residual-rescue tau policy (`residual_rescue.py:644-660`) | reads `tau_apodization_us` from `conservative_kwargs`, uses it as the rescue τ when the initial fit's τ is pegged at the lower bound (signature of a τ-collapse / over-broadened fit); otherwise uses the initial fit's τ as-is | unconditional use of `τ_maj` as the rescue τ. The lower-bound-only check is insufficient post-removal — τ can be collapsed to a value above its bound but still well below `τ_maj` (e.g. w140's 2.96 µs against an implied τ_maj ≈ 7-8 µs), and the rescue would inherit it |

The proposal therefore touches every Stage 1-5 calibration point.
The validation plan below is structured to catch regressions stage
by stage rather than measure success only at Stage 5 chi².

## Prior work this consolidates

- **Cross-fixture-validation §Dataset-wide tau calibration** (lines
  357-381): post-hoc consensus τ from strong-line windows. Did not
  propose removing apodization; kept the consensus as a per-fixture
  invariant.
- **Cross-fixture-validation §Broken-initial-fit / majority-vote-freeze**
  (lines 383-422) + **residual-rescue/report.md §8**: identified
  the rescue's joint-refit thawing τ as the channel that defeats
  the freeze on windows like w198.

This plan **supersedes** the first (`τ_maj` replaces apodization
rather than supplementing it) and **implements** the second's
intent (rescue uses `τ_maj` as its anchor), via a different
extraction method.

## Definitions

- **τ_mol** — the underlying molecular decay constant. Set
  primarily by **experimental geometry** (beam transit time
  through the cavity, horn coupling, collisional / Doppler
  broadening from the supersonic expansion), so it is expected
  to be a **global property of the experiment** rather than a
  per-species value. Velocity slip in the expansion (heavy
  species lagging the buffer gas), pressure / collisional
  broadening on a subset of lines, and instrumental clock spurs
  (CW LO/mixer leakage with effectively infinite τ) are the
  canonical reasons τ extracted at fit time could split into
  clusters.
- **τ_apod** — the user-applied exponential apodization time
  constant (the former `expf_us`, default 5 µs on 2638). Removed
  when explicit apodization was retired.
- **τ_obs** — what Stage 5 fits returned with
  apodization on (historical): the combined decay
  `τ_obs = τ_mol · τ_apod / (τ_mol + τ_apod)`. Bounded above
  by τ_apod (the combined decay rate is the sum of rates, so the
  combined τ is the harmonic-mean-ish reciprocal). On 2638's
  τ_obs ≈ 3 µs with τ_apod = 5 µs, implied τ_mol ≈ 7.5 µs.
- **τ_maj** — the proposal's data-driven majority-vote estimate of
  τ_mol from the per-bin STFT decay fits. `σ_τ` is the
  histogram's robust spread (IQR / 1.349 or the fitted Gaussian
  width).
- **fit τ** — the per-window LSQ-fitted value of the shared decay
  parameter inside `window_fit.py`. Equal to τ_mol (was τ_obs while
  apodization was applied).

## The STFT calibration method

### What the math says

For a single damped cosine `s(t) = A · exp(-t/τ_mol) · cos(2πf₀t + φ)`
extracted on `[a, a + T_w]` and zero-padded to the full-length
record before rfft, the on-line FT magnitude is

```
|S(a, f₀)| = (A · τ_mol / 2) · exp(-a/τ_mol) · (1 - exp(-T_w/τ_mol))
           = constant · exp(-a/τ_mol)
```

so sliding `a ∈ [0, T_full - T_w]` traces out a pure exponential
decay vs `a` whose rate is `1/τ_mol` directly. Single-parameter
fit, no LSQ ambiguity.

| feature | |S(a, f)| vs a |
|---|---|
| real molecular line | `exp(-a/τ_mol)` exponential |
| clock spur (τ = ∞) | constant |
| noise bin | random walk, no clean exponential |
| strong-line skirt at f₀+Δf | `exp(-a/τ_mol)` × sinc(Δf·T_w·π) — same rate as parent |

Skirts inherit their parent line's decay rate, so they reinforce
the strong-line τ cluster in the histogram rather than producing a
separate "skirt cluster". That's convenient — multi-counting the
same τ is benign.

Overlapping skirts of multiple strong lines can decohere across
frames (phase factors `exp(2πi·f·a)` differ between lines), so
bins midway between two strong sources may show faster-than-`1/τ_mol`
apparent decay or non-monotonic structure. The per-bin
goodness-of-fit gate drops these.

### FFT grid policy

**Zero-fill outside the active sub-window, do not truncate the
record length.** Zero-fill preserves the full-record bin spacing
`Δf = 1 / T_full`, so all STFT frames share the same frequency
grid and per-bin time series are 1:1 comparable. Truncating to
`T_w` would double the bin width when `T_w = T_full / 2` and break
the comparison.

### Frame schedule

- **Active window length `T_w`** — calibration default
  `T_w = T_full / N_seg` with `N_seg = 10`. Each frame contains
  `T_full / 10` active samples (the rest zeroed).
- **Frame stride** — non-overlapping by default
  (`stride = T_w`, giving `N_seg = 10` frames). Optional overlap
  factor for finer fits at the cost of correlation between
  frames; not strictly needed when N_seg is modest.
- **Frame midpoints** — fit τ against the frame *centre* time
  `a_c = a + T_w/2`. Equivalent up to a constant offset for the
  fit; documenting the convention explicitly.

`N_seg` is a knob. Larger `N_seg` (more frames) → better τ fit
per bin but lower per-frame SNR (less signal in each active
window). 2638's expected τ_mol ≈ 7.5 µs and T_full = 12.65 µs put
`exp(-T_full / τ_maj) ≈ 0.18` — the line is at ~18 % of its peak by
the end of the FID, plenty of dynamic range for an exponential
fit even at modest N_seg. The synthetic study (§Workflow phase 1)
sweeps N_seg to pick the operating point.

### Per-bin fit and classification

For each frequency bin index `k`, the STFT magnitudes
`|S_n(k)| ≡ |S(a_n, f_k)|` form an N_seg-point time series. Two
candidate models:

- **Exponential**:  `|S_n| = B + C · exp(-a_n / τ_k)`
- **Constant**:  `|S_n| = D`

Both have analytic / cheap fit kernels (the exponential is a 3-par
NLS; on noise-poor bins a robust 1-par log-linear regression on
`log|S_n|` is a faster precursor). Classification per bin:

- **Discard** if the bin's mean magnitude is below the noise
  threshold `T_σ · σ_x(k)` (see §Above-threshold bin selection).
  Most bins are dropped here.
- **Spur** if the AICc-based model choice prefers the constant
  model OR the exponential fit returns `τ_k ≥ 0.95 · τ_max`
  (saturating the upper τ_bound).
- **Bad fit** if the exponential's residual sum of squares
  exceeds an N_seg-aware threshold (e.g. > 5 × per-frame noise);
  marks dense / contaminated bins (overlapping skirts, multi-line
  blends) that shouldn't enter the histogram.
- **Contributor** otherwise. The τ_k value enters the histogram
  weighted by the bin's on-line SNR (optional refinement — gives
  strong, low-noise bins more weight than borderline ones).

### Above-threshold bin selection

The STFT calibration runs after Stages 0-2, so the canonical Stage 2
σ on the full-record FT is available. Each STFT frame's noise
floor is **lower** than the full-record FT's by a factor that
depends on the active fraction and the apodization in use during
calibration (none, by hypothesis). For non-overlapping frames of
length `T_w = T_full / N_seg`:

```
σ_frame(k) ≈ σ_x(k) · sqrt(T_w / T_full) = σ_x(k) / sqrt(N_seg)
```

(noise in a bin is the rfft of a `T_w`-long noise segment, whose
RMS scales with `sqrt(T_w)`). The threshold per bin for
"contributor" classification is `T_σ · max_n |S_n(k)|` against
the *mean* (over n) of `σ_frame(k)` — i.e. require the strongest
frame's magnitude to clear the per-frame noise by `T_σ` (default
`T_σ = 5`). This is a tighter gate than the canonical
"|X| ≥ T_σ · σ_x" applied to the full-record FT because we're
asking it to hold on the smallest STFT frame.

### Clock-spur cross-checks (the simple version)

τ saturation already classifies most spurs as "spur". A complementary
visualisation:

- **Split-acquisition magnitude ratio.** For a flagged spur
  candidate, compute first-half vs second-half FT magnitudes
  (single rfft pair, no fit). The ratio
  `R = |S_{[0, T/2]}| / |S_{[T/2, T]}|` is ≈ `exp(T / (2 · τ_maj))`
  for a real line (≈ 2.3 on 2638 at τ_maj ≈ 7.5 µs) and ≈ 1 for a
  spur. The STFT-derived per-bin τ_k already encodes this
  information; the half-window ratio is a fast independent
  sanity-check.

A peak's catalogue lookup is the ground-truth filter when
available. The calibration module should accept an optional
`spur_frequencies_mhz` parameter; the filter then drops any bin
within ±1 MHz of a listed harmonic regardless of the in-band
classification.

### Distribution analysis

The "contributor" τ_k values form the calibration histogram. The
user's stated hypothesis is no τ-vs-SNR correlation (above the
threshold) and no τ-vs-frequency correlation. **Test rather than
assume**:

- **τ vs SNR.** Pearson correlation on the contributor sample.
  Reject the null (correlated) at p < 0.05 → investigate. A real
  positive correlation would suggest amplitude-dependent shape
  effects (the cos²θ Voigt-deficit physics from
  [`stage5-cross-fixture-validation.md`](stage5-cross-fixture-validation.md)
  pulling apparent τ down on the strongest lines).
- **τ vs frequency.** Same Pearson test, plus a horn-band partition
  (low / mid / high thirds of the trim range) with median
  comparison. Deviation from the global median across thirds
  beyond ±1·σ_τ flags systematic frequency dependence.
- **Multimodality.** Fit 1-component vs 2-component Gaussian
  mixture by AIC. Single τ_mol globally is the physical
  expectation; bimodal histograms call for investigation. Likely
  causes, in rough order of likelihood on a 2638-class fixture:
  1. **Residual clock spurs that escaped the saturation filter**
     (e.g. a spur driven slightly off the bound by a nearby real
     line). Check the split-acquisition ratio on the high-τ
     cluster before treating it as physical.
  2. **Velocity slip.** Different species at different terminal
     velocities → different transit times → different τ_mol.
     A real bimodal histogram with the two clusters separated by
     10-30 % is the signature. Per-species τ assignment via
     peak-frequency-catalogue lookup is the principled fix;
     larger structural change, deferred to a follow-up plan.
  3. **Multiple decay processes within one species** (pressure
     / collisional broadening on a subset). Rare; flag for
     manual investigation.
  4. **Shape-error spread** (cos²θ Voigt deficit). Tighter SNR
     cut suppresses this.

  Default policy when bimodality is detected: take the dominant
  cluster (highest weight in the GMM fit), accept residual
  mis-fit on the minor cluster, **flag the multimodality in the
  persisted `tau_calibration` diagnostics** so a human can audit
  and override.

Outputs: `τ_maj` (median or GMM-mode-centre), `σ_τ` (robust
spread), the contributor count, multimodality flag, the
spur-bin list, and per-region (frequency-third) statistics.

### Pre-conditions for accepting `τ_maj`

- ≥ 200 contributor bins after all filters. STFT gives orders of
  magnitude more bins than the LSQ-fit alternative; this floor is
  meant to catch pathological-low cases (mostly-empty spectra),
  not to be binding on real data.
- 2-component AIC is not strongly preferred (ΔAIC < 4 relative
  to 1-component) OR the dominant cluster carries ≥ 70 % of the
  weight.
- `σ_τ / τ_maj < 0.20`. The STFT method's higher contributor
  count justifies a tighter spread threshold than the LSQ-fit
  alternative's 0.30.

If any pre-condition fails, fall back to a **conservative
default**: `τ0_us = acquisition_us / 3` and disable the τ-anchoring
penalty for that fixture. This must be tested as a real path,
not a fail-safe in name only.

## Bidirectional τ penalty (LSQ-side complement)

The current `tau_penalty_lambda` term in
`window_fit._penalty_residuals_and_jacobian`:

    sqrt(λ) * max(0, (tau_ref - tau) / tau_ref)

is **one-sided** (fires only when `tau < tau_ref`). It exists to
resist the τ-collapse failure mode: when the initial fit's K is
too small, LSQ buys χ² by driving τ DOWN (= broadening each
modelled line — recall larger τ ↔ narrower FWHM 1/(π·τ)) so the
broadened lines absorb the residual from the missing peaks. The
penalty pulls τ back UP toward τ_apod (= narrower, physical
lineshape).

With apodization removed and `τ_apod` no longer the natural
anchor, two failure modes are possible:

- **τ-collapse (τ below τ_maj)**: same mechanism as today, the
  driver is unmodelled peaks broadening the modelled ones to
  absorb their residual.
- **τ-runaway (τ above τ_maj)**: with the upper bound now
  `τ0·k` rather than `τ_apod`, LSQ can drive τ UP into a
  narrow-peak basin. This is the basin clock spurs live in; an
  unmodelled spur in or near a window can pull the shared τ
  toward it.

Both motivate a bidirectional Gaussian-prior penalty centred on
`τ_maj`:

    sqrt(λ) * (tau - tau_maj) / sigma_tau

with `sigma_tau = σ_τ` from the calibration pass (floored at e.g.
0.5 µs if the histogram is unusually tight). Strength `λ` is the
main tuning lever. The shipped `DEFAULT_TAU_PENALTY_LAMBDA = 50` sits
at the validated knee of the 2638 sweep — see
[`instrument-tunable-knobs.md`](instrument-tunable-knobs.md) § "the
three former 2638 overrides are now defaults" for the sweep evidence.

For weak windows (the existing `weak_window_snr_threshold = 10`
gate), `fit_tau = False` and τ is locked at `τ_maj` exactly.

The penalty's Jacobian is a single non-zero entry at the τ column.
The prior one-sided code path can be deleted once the new path is
verified against the existing behaviour with `τ_apod` substituted
as the centre.

## Re-seeding the residual rescue

`residual_rescue.py:644-660` currently chooses the rescue τ from
the initial fit's τ, with an override only when that τ is at the
LSQ lower bound. This is the channel that defeats anchoring: with
apodization removed and τ-collapse possible above the bound
(w140's 2.96 µs against an implied τ_maj ≈ 7-8 µs is well above
any bound), the rescue inherits the broken value.

Replacement: the rescue τ is `τ_maj` for every window, regardless
of the initial fit's outcome. The rescue's `find_residual_peaks`
matched filter uses `rescue_fwhm = feature_fwhm(τ_maj, T)` — a
narrower FWHM (since τ_maj > current τ_obs) means a more sensitive
detector for the same line shape. The `conservative_fit` on the
residual runs with `fit_tau = False` and `tau = τ_maj`, which
matches the existing "frozen τ" rescue practice.

The joint refit's current τ-thaw behaviour must be **disabled**
when calibrated τ is available. This is the specific channel the
`stage5-cross-fixture-validation.md` §"Broken-initial-fit
pathology" investigation identified.

## Implementation

The calibration is named **stage 2b** (`stage2b_tau_calibration` in the HDF5 /
dependency tracker) because it sits between the canonical Stage 2 noise estimate
and Stage 3 peak detection. The synthetic method validation, the 2638
application, and the LSQ cross-comparison are written up in
[`../research/stage5-tau-calibration/report.md`](../research/stage5-tau-calibration/report.md);
the production wiring is:

- **Calibration module.** `fitting/tau_calibration.py` exposes
  `extract_tau_majority(fid, sample_dt_us, *, start_us, end_us, probe_freq_mhz,
  sideband, trim_lo_mhz, trim_hi_mhz, sigma_time=None, n_seg=10, t_sigma=5.0, …)
  → TauCalibrationResult`. The dataclass carries `τ_maj`, `σ_τ`, the contributor
  `(bin, τ, SNR, freq)` arrays, the GMM bimodality block, the per-band-third
  median summary, the clustered spur catalogue, the knobs used, and a
  `preconditions_passed` boolean + per-condition notes. Default knobs:
  `N_seg = 10`, `T_σ = 5`, `tau_max = 5·T_full`, `rss_gate_factor = 5`, hybrid
  absolute+relative bad-fit gate (`relative_gate_fraction = 0.05`), SNR-weighted
  majority on, GMM bimodality threshold `ΔAICc > 2`. The acceptance
  pre-conditions (≥ 200 contributors, no strong bimodality unless the dominant
  cluster ≥ 70 %, `σ_τ / τ_maj < 0.20`) are evaluated and stored; failing them
  logs a warning but does not block consumers (Stage 5 uses `τ_maj` even on the
  marginal 2638 calibration). The internal `sliding_stft`, `stft_calibration`,
  `majority_tau`, `gmm_bimodality`, and `group_spur_bins` are the canonical
  implementations.
- **Persistence.** `io/tau_calibration_serialization.py` round-trips the result
  to `/stage2b_tau_calibration`. The full STFT magnitude grid is *not* persisted
  (~50 MB on 2638); the heatmap view recomputes it on demand from the FID +
  persisted knobs. Per-cluster bin lists use a CSR-style flat-with-offsets
  layout; schema versioning at `algorithm_info/version`.
- **Bidirectional τ penalty.** `_penalty_residuals_and_jacobian` in
  `window_fit.py` takes a `tau_penalty_sigma_us`; with
  `tau_penalty_reference = τ_maj` the penalty residual is
  `sqrt(λ) · (τ − τ_ref) / σ_τ` — a bidirectional Gaussian prior firing
  symmetrically against τ-collapse and τ-runaway. With `tau_penalty_sigma_us =
  None` (no calibration) the legacy one-sided hinge is preserved.
- **Bounds.** `derive_window_fit_constraints` accepts `tau_maj_us` /
  `sigma_tau_us`; when both are positive the bounds become
  `(max(τ_maj − N·σ_τ, τ_maj/k), min(τ_maj + N·σ_τ, τ_maj·k))` with
  `N = DEFAULT_TAU_PENALTY_N_SIGMA = 5` and `k = max_decay_factor`.
- **Stage 3 wiring.** `_internal/stage3_impl.py` auto-detects the Stage 2b group
  and passes `τ_maj` as `tau_basis_us` to `_mf_gap_spectrum`, falling back to a
  default 5 µs basis when absent.
- **Stage 5 wiring.** `_internal/stage5_impl.py` reads the persisted calibration
  (warning-logs marginal pre-conditions), defaults `tau0_us` to `τ_maj` when
  unset, and forwards `tau_maj_us` / `sigma_tau_us` through `conservative_kwargs`
  so `derive_window_fit_constraints` and the penalty pick it up.
- **Rescue wiring.** `residual_rescue.py` consults
  `conservative_kwargs["tau_maj_us"]`: when present, the rescue τ is `τ_maj`
  unconditionally and the joint refit freezes `fit_tau = False` at `τ_maj`,
  closing the τ-thaw channel the cross-fixture-validation §"Broken-initial-fit
  pathology" investigation flagged.
- **Stage tracker / dependencies.** `stage2b_tau_calibration` sits between
  `stage2_noise_result` and `stage3_peaks` in
  `PipelineStageTracker.STAGE_DEPENDENCIES` (requires Stages 0, 1, 2). It is
  *recommended* (not enforced) on Stages 3 and 5, which auto-detect its presence
  at runtime, so the legacy single-stage path still runs without it.
- **CLI / Pipeline / functional-API surface.** Three thin wrappers per the
  dual-interface rule: CLI `tau run` (`cli/tau_commands.py`),
  `Pipeline.calibrate_tau`, and `ftmwpipeline.api.calibrate_tau`. Visualisation
  is `tau show --kind heatmap|distribution`, in
  `visualization/tau_calibration_visualization.py`. (The Gaussian twin runs via
  `tau run --gaussian`.)
- **Spur-cluster grouping.** Adjacent spur-classified bins within `n_seg`
  full-record bins of each other collapse via `group_spur_bins` into one
  `SpurCluster` whose representative is the largest-mean-magnitude bin. On 2638
  the 649 raw spur-classified bins collapse to ~135 clusters — within the
  expected n_seg-sinc-skirt envelope per CW source. This catalogue feeds Stage
  5 spur masking (see [`stage5-spur-masking.md`](stage5-spur-masking.md)).

Stage 2–5 regression validation is carried by the committed integration suite
(re-baselined to the scatter-noise production grid); the per-stage pinned-value
tests for Stages 2b/3/4/5 and the cross-interface suite stand in for the earlier
ad-hoc validation runs.
## Polish step on the contributor histogram

`extract_tau_majority` exposes three polish knobs that shape how the
log-linear weighted-regression bias is handled:

- **`polish` (default `True`)** — one Gauss-Newton step on `|S_n| =
  C · exp(-a/τ)` per contributor bin before the SNR-weighted
  majority. Closes the synthetic +3-5 % log-linear bias documented in
  [`report.md`](../research/stage5-tau-calibration/report.md)
  § "Synthetic validation → Case 1" to ≤ ±2.3 % across the (T_full,
  τ) grid (sub-1 % on T_full ≥ 30 µs).
- **`polish_snr_cap` (default `DEFAULT_POLISH_SNR_CAP = 9.0`)** —
  restricts the polish to contributors whose per-bin SNR is **below**
  the cap; high-SNR contributors retain the log-linear seed. Lands
  per-band SNR-weighted majority τ on 2638 within ±3.2 % of the LSQ
  per-band reference (vs ±8-10 % under polish=False or
  polish=True/no-cap). Pass `polish_snr_cap=None` to disable.
- **`polish_noise_debias` (default `False`)** — replace `|S_n|` with
  the Rician-unbiased magnitude `sqrt(|S_n|² − 2σ²)`. Sub-percent
  closure on single-isolated-line synthetics but over-corrects on
  multi-line spectra (inter-line skirt interference isn't
  Rician-Gaussian). Opt-in forensic knob.

The full motivation, sweep tables, per-band bias-flip pattern, and
why an SNR cap was preferred over loosening `relative_gate_fraction`
live in
[`report.md`](../research/stage5-tau-calibration/report.md)
§ "Polish design". The production-default per-band majority on 2638
under `polish=True, polish_snr_cap=9` is 7.62 / 6.16 / 5.29 µs across
low / mid / high arithmetic thirds.

## Outstanding open questions

- **Multi-fixture confidence.** Is `τ_maj` stable across fixtures
  from the same instrument? Across instruments? Cross-fixture
  validation (see
  [`stage5-cross-fixture-validation.md`](stage5-cross-fixture-validation.md))
  is the broader programme this fits into.
- **Per-species τ.** If multimodality is detected on any fixture,
  does the pipeline support per-cluster τ assignment? Substantial
  structural change (every `FixedContributor` / `FreePeak` needs
  a τ pointer) — defer to a follow-up plan, gated on at least one
  fixture demanding it.
- **Apodization as a user-facing display knob.** Even with
  apodization off for the fit, a user might want apodization
  applied to the **displayed** spectrum for visual comparison
  with older results. Decoupling Stage 1 storage
  (apodization-free) from Stage 1 display (user-chosen
  apodization) is a reasonable feature, out of scope for this
  plan.
- **`max_decay_factor` value.** Currently 5. With apodization
  gone, this is the only multiplicative bound on τ. 2638's
  expected τ_maj ≈ 7-8 µs gives an upper bound of 35-40 µs at
  factor 5 — reasonable. May tighten to 3 once `τ_maj` is known
  empirically; with the `σ_τ`-based bound (see *Implementation* → Bounds)
  this becomes less load-bearing.
- **Calibration-override knobs.** `--tau-maj-override` /
  `--sigma-tau-override` ship on `fit run` (CLI / `Pipeline.fit_peaks`
  / `ftmwpipeline.api.fit_peaks`). The atomic pair beats any persisted
  Stage 2b calibration for that fit; supplying only one of the pair
  raises. Useful for A/B-ing a hand-tuned tau anchor against the
  persisted value, or for forcing a calibrated tau on fixtures where
  Stage 2b has not yet been run.
- **Frequency-bucketed τ.** 2638 shows real τ-vs-frequency
  dependence (low-third 7.34 µs → high-third 6.06 µs, attributed
  to W-band horn-coupling geometry). The global-τ_maj assumption
  is approximate at ~15-20 % on 2638. If validation regressions
  push past tolerance, per-band τ (low/mid/high, or a smooth 1/f
  model) is the natural next refinement; Stage 5 would consume a
  per-band τ vector instead of a scalar.
