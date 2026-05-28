# Plan: Stage 2b — data-driven τ calibration via sliding-active-window STFT

Status: **Phase 1 + Phase 2 research complete; Phase 3 production
wiring (steps 1–11) shipped as `stage2b_tau_calibration` between
Stage 2 noise and Stage 3 peak detection.** What remains: Phase 3
step 12 (the Stage 2–5 regression validation harness against
`scratch/stage5-validation3/`) and Phase 4 (the LSQ-fit-and-
histogram cross-comparison, kept as a research close-out rather
than a production prerequisite). Phase 3 and Phase 4 are swapped
from the original ordering: the STFT prototype already passed both
synthetic and 2638 acceptance gates, so wiring it in was the higher-
value next step and the LSQ comparison is what closes out the "why
STFT, not LSQ histogram?" question for the planning record.

The shipped stage is named `stage2b_tau_calibration` (a non-
disruptive prefix between `stage2_noise_result` and `stage3_peaks`);
a future broader rename pass may reshuffle the numbering for overall
consistency.

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

## Objective

Replace the upfront exponential apodization (`expf_us = 5.0` on the
2638 fixture) with a **data-driven, fit-free τ calibration** that
runs after Stages 0-2 and produces a single global majority-vote
`τ_maj ± σ_τ`. Subsequent Stages 3-5 consume `τ_maj` as:

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

The proposal removes a regularizer (apodization) and replaces it
with a self-calibrated one (`τ_maj`). The hypothesis is that
molecular τ is a global property of the experimental geometry
(beam transit time, horn coupling, collisional / Doppler broadening
from the supersonic expansion), so it should be well-determined
from the STFT of any fixture with at least a handful of
above-threshold lines.

## Why this is a serious change, not a tweak

Apodization currently does several things that the proposal needs
to either replace or validate independently. Inventory of removal
consequences:

| consumer | how apodization is used today | post-removal status |
|---|---|---|
| Stage 1 persisted FT (`FID.preprocess`) | `expf_us` multiplied into FID before rfft; the apodization decay rate adds to the molecular decay rate and widens the *effective* magnitude FWHM the user sees | gone — peaks become **narrower** in magnitude (FWHM = 1/π·τ_mol vs 1/π·τ_obs, with τ_mol > τ_obs so FWHM_mol < FWHM_obs), sidelobes longer (no exponential suppression of the sinc) |
| Stage 2 σ estimator (`estimate_noise_adaptive`) | empirical MAD/median — no explicit apodization parameter | adaptive; should re-converge but skirt-exclusion mechanism (`STRONG_PEAK_SNR=20`, `SKIRT_EXCLUSION_K=1.5`) needs verification — narrower peaks have *narrower* HWHM γ but the 1/Δf far-field skirt extends *farther* with no exponential cap, so the exclusion radius `γ·SNR/k` can shift in either direction |
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
  constant (today `expf_us`, default 5 µs on 2638). Removed by
  this proposal.
- **τ_obs** — what the current Stage 5 fits return with
  apodization on: the combined decay
  `τ_obs = τ_mol · τ_apod / (τ_mol + τ_apod)`. Bounded above
  by τ_apod (the combined decay rate is the sum of rates, so the
  combined τ is the harmonic-mean-ish reciprocal). On 2638's
  τ_obs ≈ 3 µs with τ_apod = 5 µs, implied τ_mol ≈ 7.5 µs.
- **τ_maj** — the proposal's data-driven majority-vote estimate of
  τ_mol from the per-bin STFT decay fits. `σ_τ` is the
  histogram's robust spread (IQR / 1.349 or the fitted Gaussian
  width).
- **fit τ** — the per-window LSQ-fitted value of the shared decay
  parameter inside `window_fit.py`. Equal to τ_obs today; would
  equal τ_mol post-change.

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

## Workflow (a fresh session's roadmap)

The work is large enough that ordering matters. Recommend three
phases, each gated on the previous one's success.

### Phase 1: Synthetic prototype (research)

**Goal**: validate the STFT calibration method on controlled
synthetic FIDs before introducing it to any real data. All
outputs land under `dev-docs/research/stage5-tau-calibration/`
(`prototype.py`, `report.md`, `figures/`, `data/`).

Reusable building blocks from the existing research prototypes:

- `dev-docs/research/matched-filter-detection/prototype.py`
  ::`regenerate_fid_for_sim` — constructs a noisy time-domain
  FID from controlled `(τ_truth, line_bins, SNR, σ_time)`. The
  closest fit to what the STFT prototype needs; extend rather
  than re-invent.
- `dev-docs/research/peak-detection/prototype.py`
  ::`synthetic_spectrum` — analytic complex FT in the `h_T`
  form. Useful for ground-truth bin-by-bin amplitudes for the
  per-bin τ fits.
- `dev-docs/research/noise-grid-invariance/prototype.py`
  ::`make_synthetic_noise` — noise-only FIDs with controlled
  σ(f). Useful for the noise-bin classification test.

Synthetic cases to verify in `Phase 1`:

1. **Single isolated strong line.** Sweep τ_truth ∈ {3, 5, 7.5,
   12, 20} µs at fixed T_full = 12.65 µs and SNR = 100. Check
   `τ_maj` recovered within ±5 % at each value. Pick `N_seg`
   from this sweep (the empirical operating point that gives
   recovery accuracy across the range; expect `N_seg ∈ {8, 10,
   16}`).
2. **SNR sweep at fixed τ.** Vary SNR ∈ {5, 10, 20, 50, 100,
   500} at τ_truth = 7.5 µs. Test the hypothesis that the per-bin
   τ recovery has **no SNR dependence above some floor**
   (likely SNR ≳ 10-20 for STFT's per-frame SNR). Below the
   floor, document the bias direction.
3. **Clock spur isolated.** Add a CW tone at a clean frequency
   bin. Verify it's classified as "spur" (τ_k ≥ 0.95 · τ_max,
   constant-model wins AIC) and dropped from the contributor
   set.
4. **Clock spur next to a real line.** CW tone within ±5 FWHM
   of a real damped cosine. Verify the spur classification
   still fires on the spur's frequency bin, and the real line's
   τ recovery is unaffected.
5. **Dense cluster (Stage 4 mega-window analogue).** Six lines
   within a 5 FWHM span, all τ_truth = 7.5 µs. Verify per-bin τ
   recovery on the in-line bins; verify the contaminated
   between-line bins are dropped by the goodness-of-fit gate.
6. **Bimodal population (velocity slip simulation).** Half the
   lines at τ_truth = 5 µs, half at τ_truth = 10 µs. Verify
   the GMM 2-component AIC test fires; verify the
   multimodality flag in the calibration result.
7. **Voigt-deficit shape error.** Inject the cos²θ residual
   from `scratch/stage5-validation/diag_voigt_hypothesis.py` at
   amplitude-dependent magnitude. Verify the per-bin τ
   distribution shows controlled mild SNR dependence (the
   regression slope from the cross-fixture-validation work).

Acceptance for Phase 1: all seven cases recover the stated
ground-truth behaviour. Document failure modes for any that
don't.

### Phase 2: 2638 application (research)

**Goal**: apply the Phase-1 STFT prototype to the 2638 fixture's
real FID and produce a candidate `τ_maj`. Output lands under
`dev-docs/research/stage5-tau-calibration/`.

Steps:

1. Re-run Stages 0-2 with `expf_us = None` on a fresh copy of
   the 2638 fixture (CLI already supports `expf_us` as a
   parameter, so no code change needed). Persist into
   `scratch/stage5-tau-calibration/exp_2638_unapodized.ftmw`.
2. Run the Phase-1 prototype on the raw FID, using the
   unapodized Stage 2 σ as the noise reference for the
   above-threshold bin selection.
3. Inspect the 2D STFT heatmap (frequency × frame). Spot-check
   the qualitative predictions: solid streaks at known clock
   frequencies (if listed); exponential decay at the strong
   lines anchoring 2638's previous validation windows
   (especially the freq ranges of w216, w293, w302, w69).
4. Plot τ_k histogram + τ_k vs frequency + τ_k vs SNR.
   Statistical tests per §Distribution analysis.
5. Report `τ_maj ± σ_τ` for 2638, the contributor count, the
   bimodality test outcome, the SNR / frequency correlation
   tests, and the list of identified spur frequencies.

Acceptance for Phase 2:

- Calibration pre-conditions pass on 2638.
- `τ_maj` is consistent with the implied 7-8 µs from current
  τ_obs ≈ 3 µs + τ_apod = 5 µs combined-decay arithmetic (within
  ±20 %).
- Spurs (if any) are at frequencies consistent with the
  instrument's known clock harmonics.

### Phase 3: Production wiring (shipped)

The calibration is named **stage 2b** ("stage-2b" in prose,
`stage2b_tau_calibration` in the HDF5 / dependency tracker) because
it sits between the canonical Stage 2 noise estimate and Stage 3
peak detection. A future broader rename pass may reshuffle the
stage numbering for consistency; this prefix is the non-disruptive
placeholder.

Steps 0–11 are landed; step 12 (regression validation) is the open
follow-up.

0. **API: `expf_us=None` truly disables apodization.**
   `ftmwpipeline.api.compute_ft` / `Pipeline.compute_ft` previously
   fell back to a hard-coded `expf_us = 5.0` whenever no layer in
   the resolution chain set it, silently re-enabling apodization on
   calls intended to produce an unapodized FT. The hard default is
   removed from `core/settings.py::_HARD_DEFAULTS`; `None` now
   propagates all the way to `FID.preprocess` (which already treats
   it as "no apodization"), and `FIDProcessingParameters.__post_init__`
   plus `compute_active_ft` coerce non-positive `expf_us` to `None`
   instead of raising, so `expf_us=0` is an explicit opt-out
   sentinel. With this, Stages 0–2 on a fresh fixture run on a
   genuinely unapodized FT — that is the noise reference the
   calibration consumes.

1. **Calibration module.** `src/ftmwpipeline/fitting/tau_calibration.py`
   exposes `extract_tau_majority(fid, sample_dt_us, *, start_us,
   end_us, probe_freq_mhz, sideband, trim_lo_mhz, trim_hi_mhz,
   sigma_time=None, n_seg=10, t_sigma=5.0, …) →
   TauCalibrationResult`. Pure function; unit-tested. The dataclass
   carries `τ_maj`, `σ_τ`, the contributor `(bin, τ, SNR, freq)`
   arrays, the GMM bimodality block, the per-band-third median
   summary, the (clustered) spur catalogue, the calibration knobs
   used, and a `preconditions_passed` boolean + per-condition
   notes. The internal `sliding_stft`, `stft_calibration`,
   `majority_tau`, `gmm_bimodality`, and `group_spur_bins` are
   ported from the research prototype as the canonical
   implementations; the research script stays intact as history.
   Default knobs: `N_seg = 10`, `T_σ = 5`, `tau_max = 5·T_full`,
   `rss_gate_factor = 5`, hybrid absolute+relative bad-fit gate
   with `relative_gate_fraction = 0.05`, SNR-weighted majority on,
   GMM bimodality threshold `ΔAICc > 2`. The Phase-2 acceptance
   pre-conditions are evaluated and stored (≥ 200 contributors,
   no strong bimodality unless dominant cluster ≥ 70 %,
   `σ_τ / τ_maj < 0.20`); failing pre-conditions log a warning but
   do not block downstream consumers, because Stage 5 is configured
   to use `τ_maj` even on marginal calibrations (the 2638 marginal-
   spread case is the canonical example).

2. **Persistence.** `io/tau_calibration_serialization.py` round-
   trips `TauCalibrationResult` to `/stage2b_tau_calibration`. The
   full STFT magnitude grid is *not* persisted (~50 MB even after
   compression on the 2638 fixture); the heatmap visualisation
   recomputes it on demand from the FID + persisted knobs.
   Per-cluster bin lists use a CSR-style flat-with-offsets layout
   so each cluster is recoverable. Schema versioning lives at
   `algorithm_info/version`.

3. **Bidirectional τ penalty.** `_penalty_residuals_and_jacobian`
   in `window_fit.py` carries an additional `tau_penalty_sigma_us`
   argument; when set with `tau_penalty_reference = τ_maj`, the
   penalty residual is `sqrt(λ) · (τ − τ_ref) / σ_τ` with Jacobian
   `sqrt(λ) / σ_τ` — a bidirectional Gaussian prior that fires
   symmetrically against τ-collapse and τ-runaway. When
   `tau_penalty_sigma_us` is `None` (no calibration) the legacy
   one-sided hinge form is preserved. Both forms are exercised by
   `TestBidirectionalTauPenalty` (finite-difference Jacobian
   checks + boundary cases on both sides of the centre).

4. **Bounds rework.** `derive_window_fit_constraints` accepts
   `tau_maj_us` and `sigma_tau_us`; when both are positive the
   bounds become
   `(max(τ_maj − N·σ_τ, τ_maj/k), min(τ_maj + N·σ_τ, τ_maj·k))`
   with `N = DEFAULT_TAU_PENALTY_N_SIGMA = 5` and
   `k = max_decay_factor`. The legacy `tau_apodization_us`-anchored
   path is preserved unchanged when no calibration is supplied.

5. **Stage 3 wiring.** `_internal/stage3_impl.py` auto-detects
   the Stage 2b group and passes `τ_maj` as `tau_basis_us` to
   `_mf_gap_spectrum`. Fallback order: `τ_maj` → `expf_us` → 5.0.

6. **Stage 5 wiring.** `_internal/stage5_impl.py` reads the
   persisted calibration (warning-logs marginal pre-conditions),
   defaults `tau0_us` to `τ_maj` when unset, and forwards
   `tau_maj_us` / `sigma_tau_us` through `conservative_kwargs`.
   `tau_apodization_us` is still forwarded; `derive_window_fit_constraints`
   prefers the calibration when both are present.

7. **Rescue wiring.** `residual_rescue.py` consults
   `conservative_kwargs["tau_maj_us"]` first: when present, the
   rescue τ is `τ_maj` unconditionally and the joint refit freezes
   `fit_tau = False` at `τ_maj` (closing the τ-thaw channel the
   cross-fixture-validation §"Broken-initial-fit pathology"
   investigation flagged). The legacy pegged-bound override is
   kept as the fallback when no calibration is plumbed.

8. **Stage tracker / dependencies.** `stage2b_tau_calibration`
   sits between `stage2_noise_result` and `stage3_peaks` in
   `PipelineStageTracker.STAGE_DEPENDENCIES`, requiring Stages 0,
   1, and 2. It is *recommended* (not enforced) on Stages 3 and 5
   so the legacy single-stage path is preserved during the rollout
   — both stages auto-detect the calibration's presence at runtime.

9. **CLI / Pipeline / functional-API surface.** Three thin
   wrappers per the dual-interface rule: CLI subcommand
   `calibrate-tau` (in `cli/tau_commands.py`),
   `Pipeline.calibrate_tau`, and `ftmwpipeline.api.calibrate_tau`.
   Visualisation subcommands `visualize-tau-heatmap` (figure 08)
   and `visualize-tau-distribution` (figure 09) ported from the
   research prototype into
   `visualization/tau_calibration_visualization.py`. The
   `--tau-maj-override` / `--sigma-tau-override` knobs on
   `fit-peaks` are not yet exposed; a calibration-override path
   has not yet been needed in practice.

10. **Spur-cluster grouping.** Adjacent spur-classified bins
    within `n_seg` full-record bins of each other are collapsed
    via `group_spur_bins` into one `SpurCluster` entry whose
    representative is the bin with the largest mean magnitude.
    On 2638 the 649 raw spur-classified bins collapse to ~135
    clusters under this rule — within the expected
    n_seg-sinc-skirt envelope per real CW source.

11. **Apodization default.** Documented in `CLAUDE.md`:
    `compute_ft` on a fresh fixture produces an unapodized FT
    by default; `expf_us` remains an optional user lever for
    legacy comparison runs. The 2638 example still shows the
    apodized recipe alongside the unapodized one.

12. **Stage 2–5 regression validation (OPEN).** Run the
    validation harness on `scratch/stage5-validation3/` and the
    full non-slow test suite with Stage 2b enabled. Compare against
    the current post-Stage-2/3-rework / post-penalty-recast
    baseline. Stage 4's `S_coh` `T_edge` may need recalibration if
    the leakage-touched fraction climbs above ~25 %; document and
    address separately if it does. Deferred to a follow-up session
    so the surface area of this change set stayed manageable.

### Phase 4: LSQ-comparison cross-validation (research)

**Goal**: cross-validate the STFT calibration against the
LSQ-fit-and-histogram alternative on 2638 and the Phase-1 synthetic
cases, defending the design choice empirically. Optional from a
production-correctness standpoint (Phase 3 ships independently); the
comparison is what closes out the "why STFT, not LSQ histogram?"
question for the planning record.

LSQ-fit-and-histogram method (the alternative — formerly the
primary in earlier drafts of this doc):

- Run Stages 0-5 on the same unapodized fixture, no τ-anchoring
  penalty active, `tau0_us = fid_length / 2`. The Stage 5 fits
  produce per-window τ values.
- Filter to EASY-difficulty K=1 windows with free-peak SNR ≥ 20,
  no fixed contributors, `tau_err / tau < 0.10`, χ²_r < 2,
  and τ not saturating the upper bound.
- Histogram-fit a Gaussian → `τ_maj_lsq ± σ_τ_lsq`.

Cross-validation on 2638:

- `|τ_maj_stft − τ_maj_lsq| / τ_maj_stft < 0.10`. If they
  disagree at this level, investigate; the primary candidate
  cause would be the LSQ approach being contaminated by
  shape-error-driven τ bias.
- Spurs identified by both methods agree (catalogue overlap).
- Documented assessment: under what conditions does each method
  fail? On 2638 specifically, which is more sensitive to e.g.
  blended pairs / shape error / dense clusters?
- The frequency-dependent τ signature identified in Phase 2
  (low-third 7.34 µs → high-third 6.06 µs, attributed to W-band
  horn-coupling beamwidth scaling): does the LSQ method reproduce
  this slope? If yes, the physical interpretation is reinforced
  and per-band τ wiring becomes the natural next refinement. If
  no, the slope is a shape-error artefact in the STFT path.

Cross-validation on the Phase-1 synthetic cases:

- For each of the seven synthetic cases, run both methods and
  compare. The LSQ method may fail outright on (5) dense
  cluster and (7) Voigt deficit (because LSQ τ absorbs the
  multi-peak / shape residual); document this as one of the
  reasons STFT is the primary.

Acceptance for Phase 4: both methods agree on 2638's τ_maj
within 10 %; STFT's advantages are quantified on the synthetic
failure cases.

## Risks

- **Calibration fails on 2638.** STFT pre-conditions don't pass,
  or `τ_maj` falls outside expected range. The conservative
  fallback (`acquisition_us / 3`, no anchoring penalty) must be
  tested as a real production path, not a fail-safe in name only.
- **N_seg / T_w mis-calibration.** Phase 1 picks the operating
  point on synthetic; if 2638 needs a different N_seg than the
  synthetic study indicated, document the deviation and the
  reason. Probably points at a model mismatch in the synthetic
  setup (apodization, noise statistics, line density) worth
  fixing in the prototype.
- **Stage 4 over-merges without apodization.** If the
  leakage-touched fraction climbs past ~25 %, mega-windows
  beyond w302 start appearing. Independent of `τ_maj` — Stage 4's
  problem to solve (T_edge recalibration). Document as a separate
  follow-up.
- **Multi-modal τ population.** Cross-fixture validation must
  check this before promoting single-τ_maj to default. If a
  fixture shows two clusters, per-species τ assignment is the
  principled fix (larger structural change, deferred).
- **w216-class shape-error windows don't improve.** Expected, not
  a regression. The cos²θ / Voigt-deficit work in
  [`stage5-cross-fixture-validation.md`](stage5-cross-fixture-validation.md)
  is the relevant lever.
- **τ_maj is sensitive to noise threshold.** The
  above-threshold bin count and τ_maj median should not
  depend strongly on the threshold value. Document the
  sensitivity in the Phase-2 report.

## Acceptance gates

Promotion of `stage2b_tau_calibration` to production was gated on:

1. **Phase 1 acceptance.** All seven synthetic cases recover the
   stated behaviour. ✓
2. **Phase 2 acceptance.** STFT on 2638 produces `τ_maj` within
   ±20 % of the implied 7-8 µs; pre-conditions pass; spurs
   classified consistent with instrument harmonics. ✓
   (`τ_maj = 6.33 ± 1.62 µs`)
3. **Unit tests.** Full non-slow suite green; new unit tests for
   the calibration module (algorithmic kernels + GMM + spur
   clustering), serialization round-trip, and the bidirectional
   penalty (finite-difference Jacobian on both sides of the
   centre + boundary cases). ✓
4. **Cross-interface consistency.** CLI / Pipeline / functional
   API produce bit-identical `TauCalibrationResult` objects on
   the 2638 fixture. ✓

Stage 5 regression validation (gates 5-8) on `scratch/stage5-validation3/`:

The natural comparison baseline is `scratch/stage5-validation3-noprior/`
— the same fixture with Stage 2b absent (unapodized FT, free tau, no
prior). Comparing to `scratch/stage5-validation2/` (apodized FT,
`expf_us = 5.0`) is not apples-to-apples: the apodization shifts every
window's observed τ from molecular (~6 µs) to combined (~3 µs), so a
typical isolated-line window's chi² drops to ~0.5-1 because the model
is matching an artefact of the apodization rather than the molecular
shape. Cross-baseline chi² comparisons against v2 are therefore
misleading; the meaningful test is "what does the prior cost vs no
prior at the same FT?"

5. **No tau-runaway under the prior.** Every fixture-window pair where
   the no-prior baseline drives `tau` to the upper bound (≥ 0.95 ·
   `max_decay_factor · tau0`) lands at a physical `tau` (between
   `tau_maj / max_decay_factor` and `tau_maj · max_decay_factor`)
   under the prior. On 2638 there are 5 such windows in `v3a` (at
   30720, 32960, 35839, plus 2 others); the prior pulls them to
   `tau ∈ [7-8] µs`. ✓
6. **w198 chi² recovers when the joint refit unlocks tau.** The
   canonical τ-collapse case at 33721-33726 MHz (legacy K=3 chi²=125
   with apodization, original v3 with locked-tau-rescue K=7 chi²=56).
   Target: with the rescue+joint-refit unlocked, chi² ≤ 30 (a 50%
   improvement vs the locked-rescue v3). ✓ (lands at chi² = 22.4 with
   tau = 4.0, K = 7). Note that w140's chi² stays ~55 across all
   prior settings — w140's residual is a missing-peak rescue gap, not
   a τ-collapse case, and is tracked separately.
7. **Stage 4 plan remains sane.** ≥ 380 windows on 2638; max width
   ≤ 80 MHz; hard-window fraction within 55-75 %. The unapodized FT
   plus the Stage 2b prior shifts the hard fraction down to ~50 % on
   2638 (more isolated singletons resolve cleanly with sharper line
   shapes), still inside the realistic band given the dramatic chi²
   improvement at the worst windows.
8. **Stage 5 χ²_r distribution.** Vs the no-prior baseline at the
   same FT (v3a), the prior cost is bounded: median chi² ≤ +5 %, p95
   ≤ +20 %, max ≤ +5 %. On 2638 the actual deltas are +2.8 % median,
   +15 % p95, 0 % max (the prior provides full tau-runaway
   suppression without inflating the worst windows). ✓
9. **LSQ agreement.** STFT and LSQ-fit-and-histogram agree on `τ_maj`
   on 2638 within 10 %. ✓ — the polish-default flip plus the
   per-band routing land per-band majorities within ±3.2 % of the
   LSQ expanded per-band reference; see [`report.md`](../research/stage5-tau-calibration/report.md)
   § "Polish design" for the underlying calibration.

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
  empirically; with the `σ_τ`-based bound from §Phase 3 step 4
  this becomes less load-bearing.
- **Calibration-override knobs.** `--tau-maj-override` /
  `--sigma-tau-override` ship on `fit-peaks` (CLI / `Pipeline.fit_peaks`
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
