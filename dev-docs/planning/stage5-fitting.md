# Plan: Stage 5 — Per-window fitting

Status: **implemented** — implementation overview per
[`README.md`](README.md): the model, the algorithm, the interface surface, the
serialization, and the test plan, as built. Registered in
[`../ROADMAP.md`](../ROADMAP.md). Three line-shape/nuisance extensions built on
top of this core are tracked in their own documents — the Gaussian envelope
([`stage5-gaussian-shape.md`](stage5-gaussian-shape.md)), clock/LO spur masking
([`stage5-spur-masking.md`](stage5-spur-masking.md)), and the leakage-wing
baseline ([`stage5-leakage-wing-baseline.md`](stage5-leakage-wing-baseline.md)).
The Stage 5 *Open questions* below are triaged (issue #8): O5-3/O5-4/O5-6
resolved (cross-instrument), O5-9 deferred (no evidence of the failure mode),
O5-10 Tier-2 deferred.

Normative requirements remain in the `*_STRATEGY.md` specs; this document is
normative only for the Stage 5 work it tracks. It builds directly on the
finalized Stage 3 → Stage 4 contract
([`stage3-peak-detection.md`](stage3-peak-detection.md),
[`stage4-window-assignment.md`](stage4-window-assignment.md)) and consumes the
`WindowPlan` Stage 4 produces. The D8 leakage rework
([`leakage-detection-rework.md`](leakage-detection-rework.md)) is fully
resolved; its three "open items handed to Stage 5" are addressed below.

## Objective

Turn the Stage 4 `WindowPlan` into a fitted line list. For each fit window,
recover the spectroscopic parameters — frequency, amplitude, phase, and a
shared decay constant — of the lines it contains, by least-squares against the
**complex** FT, modelling each line with the exact finite-acquisition response
so that truncation leakage is reproduced rather than fought. Carry strong
out-of-band lines as frozen contributors; fit windows in the plan's dependency
order; renegotiate with Stage 4 when a fit reveals coupling the plan did not.

The project goal is to fit *unwindowed* (boxcar-truncated, full-resolution)
spectra: no apodization is applied to suppress leakage, so resolution is
preserved and **barely-resolved blended lines stay resolvable**. The fitting
model must therefore reproduce leakage exactly, and the algorithm's hardest job
is strong, partially-resolved blends — see *The blending problem* below.

## Spectral domain for the fit: the active-portion FT (normative)

The persisted Stage 1 spectrum is an rfft of the *whole* native-length record:
the active region `[start_us, end_us]` is shorter than the 15 µs record (on 2638
~632k active samples front-zeroed into a 750k-point FFT). That representation is
right for **display, peak detection, and window planning** — sub-bin centroid
interpolation helps the detector, and the edge-coherence statistic is calibrated
on the full grid. But it is wrong for **statistical fitting**: front-zeroing the
record outside the active region correlates adjacent bins by a Dirichlet kernel
(the FFT of the active-region indicator), so the effective number of independent
samples in any band of `M` bins is

    M_eff = M · α,   α = N_active / N_record   (≈ 0.84 for 2638),

and the naive `N_dof = M − N_params` overcounts by `1/α`. That bias
propagates into reduced χ², the F-test, and AIC — all in the direction of
optimism — and silently miscalibrates the conservative loop's accept/reject
decisions and the blend-aware seeder's `rchi2 > 1.5` trigger.

**Stage 5 fits in the active-portion FT frame** to dissolve the problem at
source. The fit-time spectrum is the rfft of just the active-region samples
of the persisted FID — unapodized, like the canonical Stage 1 FT:

    active_ft[k] = rfft( fid[t0:t0+T] )[k]

This representation has

- **independent bins** (no Dirichlet correlation; α = 1 by construction),
- **no phase ramp** (it is already in the `[0, T]` form `h_T` models),
- **the same molecular frequency axis** (bin spacing = `1/T_active` instead
  of `1/T_padded`, but the same continuous-frequency interpretation),
- **the same line shape** (`h_T(Δf; τ_apod)` on a coarser grid — ~10 bins
  per FWHM at the 2638 scale, well-sampled for a 3–4-parameter line model).

Consequences for the rest of the plan:

- **Statistics are honest by construction.** `reduced_chi2` ≈ 1 for a good
  fit on real data; the F-test and AIC work as written;
  `seeder_rchi2_threshold = 1.5` recalibrates against truth, not against
  `1/α ≈ 1.2`.
- **The de-ramp is a no-op.** `to_baseband_frame` no longer needs the
  `deramp_to_active_start` step; the active-FT is the `[0, T]` form
  natively. The de-ramp helper stays in `preprocessing/leakage.py` for
  Stage 4's edge-coherence work, but Stage 5 stops calling it.
- **The σ/√2 D-8 noise weighting still applies.** It splits the per-bin
  complex variance into Re/Im halves; it is independent of the bin-
  correlation question.
- **Per-bin noise is the active-FT scatter authority.** Stage 5 weights its
  fit by the scatter estimator
  (`preprocessing.noise_estimation.estimate_active_ft_noise`, the sole Stage 2
  estimator) measured on the active-FT magnitude spectrum it fits — the same
  estimator and grid Stage 2 persists. No conversion factor, no `σ / √α`
  rescale: the noise estimate comes from the same spectrum the fit sees, so any
  FFT-normalization choices cancel by construction. (Earlier drafts derived
  `σ_active = σ_persisted / √α` from a full-record Stage 2 result; this proved
  fragile because the formula only holds under unitary FFT normalization, and
  the persisted full-record FT uses `/N_orig × 10⁶`. The whole pipeline now
  measures noise once in active-FT space — Stage 2 measures and persists σ on
  the canonical active FT, and Stages 3/4/5 consume it — so the mismatch never
  arises. See `planning/stage2-noise-authority.md`.)
- **The active-FT is internal to Stage 5.** It is computed on demand from
  the persisted FID (`stage0_fid_data`) and the persisted Stage 1 settings
  (`start_us`/`end_us`/`trim`); it is not persisted in the `.ftmw` file (small,
  fast, and trivially regeneratable). Stage 5 therefore gains an explicit
  dependency on Stage 0 alongside Stage 4.
- **Display still uses the persisted FT.** Visualization overlays the
  fitted model on the high-resolution persisted grid by re-evaluating
  `model_spectrum` at those frequencies; the model is grid-agnostic.

An earlier persisted-FT fit path is preserved in the algorithm module as a
fall-back-debugging surface but is no longer the production fit frame. The
active-portion FT contract is registered in
[`../ROADMAP.md`](../ROADMAP.md) as divergence **D9** (resolved).

## Reuse map

Per the locked Stages 3–5 design, the refined `newfitting/` engine
(`fit_time_domain_peaks`, adaptive window selection, peak aggregation) is
**permanently lost** and is recreated against the surviving shell's contract.
What survives in `~/github/bcfitting/src/bcfitting/ftmwfitting.py` and is
ported/cleaned:

- `fit_weak_window_conservative_time_domain` — the **orchestration shell**: a
  conservative add-one-peak loop, strongest-peak-first, accept a candidate only
  on an F-test *and* AIC improvement, with a peak-separation constraint and a
  shared decay rate. Its statistical-test helpers (`calculate_aic`,
  `calculate_chi_squared_improvement`, `passes_significance_test`,
  `validate_peak_separation`, `calculate_noise_weighted_chi2`,
  `calculate_rms_residuals`) port directly.
- The contract of the lost `fit_time_domain_peaks`, inferred from its call
  sites: `(complex window, frequencies, shared decay, bounds) → result with`
  `success, fitted complex spectrum, per-peak {frequency, amplitude, decay},`
  `cost, AIC, reduced χ², iterations`. Stage 5 recreates this as the
  per-window least-squares core.
- `calculate_hwhm_from_apodization` — the apodization→linewidth physics, used
  for the τ prior and the separation constraint.

What is **not** revived: `iterative_peak_subtraction` and
`generate_sinc_leakage_pattern` (frequency-domain analytic-sinc subtraction
from the data). The discarded design subtracted leakage from the data; the
locked design *models* it. Incremental peak **addition** is used, not
subtraction (the prior effort assessed both; addition won — see *The
conservative add-one-peak loop*).

Existing stub package: `src/ftmwpipeline/fitting/` already holds placeholder
modules (`time_domain.py`, `conservative.py`, `validation.py`) raising
`NotImplementedError`. Stage 5 replaces these with the real implementation
(module layout below).

## The model

### Finite-acquisition line shape

A single molecular line is an exponentially damped cosine, excited at the
active-region turn-on and observed over the finite acquisition `[t₀, t₀+T]`
(`t₀ = start_us`, `T` = active acquisition length, the
`_active_acquisition_us` helper). Its complex-FT response near the line, in
**baseband (scope) frequency** offset `Δf` from line centre, is the
finite-T envelope derived in
[`../research/complex-edge-coherence/report.md`](../research/complex-edge-coherence/report.md) §2:

$$
X(f) \approx \tfrac12 A\, e^{i\varphi}\; h_T(\Delta f;\tau),
\qquad
h_T(\Delta f;\tau) = \frac{1 - \exp[-(1/\tau + i2\pi\Delta f)\,T]}{1/\tau + i2\pi\Delta f}.
$$

`h_T` is the exact complex FFT of a finite-T damped cosine: it reproduces the
$1/|\Delta f|$ truncation-leakage skirt *and* its phase coherence. A window's
model is a sum of such terms. Because the model carries leakage exactly,
subtracting a fitted line removes its skirt correctly — that is the whole
reason the fit is done in the complex-FT domain rather than by analytic-sinc
data subtraction.

### Effective time axis and model generation

The data to fit is a slice of the **active-portion FT** — the rfft of just the
`[t₀, t₀+T]` FID samples, unapodized like the canonical Stage 1 FT. By
construction this slice is the FFT of a damped cosine observed over the
finite acquisition `[0, T]` (no surrounding zeros, no zero-padded
interpolation), so the model is `h_T` evaluated directly on the slice's
frequency grid. Adjacent bins are statistically independent — see *Spectral
domain for the fit* — so the conventional `N_dof = M − N_params` is exact.

In practice that transform has the closed form `h_T(Δf;τ)` above, so the model
spectrum is `h_T` evaluated directly on the window's frequency grid — no
numerical FFT needed, and (unlike a literal decimated FFT) no aliasing of
**out-of-band fixed contributors**, whose skirts must be evaluated at large
`Δf`. The literal numerical FFT of the effective-time model is retained only
as a unit-test cross-check of the closed form (O5-1).

### Demodulation and the sideband mapping (derive and unit-test first)

This is the riskiest piece; it must be derived explicitly and unit-tested on
synthetic signals of known `A/f/φ/τ` on **both sidebands** before any
real-data fitting (the Stage 3 doc mandates exactly this).

The persisted spectrum is on a **molecular** frequency grid `f`; the line
physically sits at **baseband** frequency `f_bb` in the FID. The sideband sign
`s` connects them:

$$
f_{bb} = s\,(f - f_{probe}), \qquad s = -1 \text{ (lower sideband)},\; s = +1 \text{ (upper)}.
$$

2638 is lower sideband: baseband DC ($f_{bb}=0$) maps to $f = f_{probe} =
40960$ MHz, and increasing FID frequency maps to *decreasing* molecular
frequency — the persisted axis is descending.

**Demodulation** (the D-1 reparameterization): a window is fit about a
reference molecular frequency `f_c`. Every peak `j` is fit by its *signed
baseband offset* `δⱼ = s·(fⱼ − f_c)` — a small (~MHz) signed number — not its
absolute ~36000 MHz frequency. The window's grid is converted to the same
offset coordinate `u = s·(f − f_c)`. The model spectrum is

$$
\text{model}(u) = \sum_j \tfrac12 A_j e^{i\varphi_j}\, h_T(u - \delta_j;\tau)
                 + \sum_c \tfrac12 A_c e^{i\varphi_c}\, h_T(u - \delta_c;\tau_c),
$$

free peaks `j` plus frozen fixed contributors `c`. Recovered frequencies map
back as `fⱼ = f_c + s·δⱼ`.

The sideband sign is load-bearing: `h_T(−Δf) = conj(h_T(Δf))` exactly (the
denominator `1/τ ± i2πΔf` are conjugates, likewise the numerator). Getting `s`
wrong conjugates every leakage skirt — the imaginary part flips, and fitted
frequencies and phases come out biased while the magnitude residual can still
look plausible. The synthetic both-sideband unit tests exist to catch exactly
this.

**The turn-on ramp.** The persisted spectrum carries the D8 phase ramp
`exp(±i2π f_bb t₀)` because it is the rfft of the *whole* native-length record.
The **active-portion FT** Stage 5 actually fits on (see *Spectral domain for
the fit*) is the rfft of just the `[t₀, t₀+T]` samples, which is in the
`[0, T]` form natively — there is no phase ramp to remove. The
`preprocessing/leakage.py:deramp_to_active_start` helper remains in place for
Stage 4's edge-coherence work on the persisted spectrum; Stage 5's
`to_baseband_frame` no longer calls it. (Earlier drafts of this plan, and the
task-5 implementation that was committed against the persisted FT, did apply
the de-ramp — D-2. The active-FT contract supersedes that path and the call
is dropped.)

**Window function.** `h_T` models boxcar truncation × exponential decay only.
The canonical FT is unconditionally unapodized and un-windowed (no
`window_function` / `expf_us` knobs exist), so the model never has to carry an
apodization kernel (D-3).

**Near-DC mirror term.** The rfft of a real FID is conjugate-symmetric; near
`+f_bb` the `−f_bb` mirror contributes `½A e^{−iφ} h_T(u + 2f_{bb,c} + …)`.
It is negligible unless a window sits within a few `T⁻¹` of baseband DC
(2638's closest window is ~960 MHz away — negligible). Stage 5 includes the
mirror term only for windows below a documented `f_bb` cutoff (D-4).

## The fit

### Least-squares residual

Per the locked contract, the residual is in the **complex-FT domain**: model
`h_T` evaluated on the active-portion window grid versus the active-portion
complex window data, identical point counts. Real and imaginary parts are
stacked into one real residual vector; the least-squares objective is
noise-weighted.

The active-portion FT bin noise is the active-FT scatter authority σ measured
directly on the active-FT magnitude spectrum (`estimate_active_ft_noise`; see
*Spectral domain for the fit* and `planning/stage2-noise-authority.md`) — the
same estimator and grid Stage 2 persists, with no conversion factor and no
`σ / √α` rescale. The active-portion bins are independent,
so summing their squared residuals gives a sum that is exactly χ²-distributed
with `M − N_params` degrees of freedom. Real and imaginary parts of σ
each carry variance σ² / 2, so the stacked Re/Im residual is weighted
by **σ/√2** for every element to be unit-variance — then reduced χ²
≈ 1 and the F-test is calibrated (D-8; the prototype confirmed weighting by
σ alone leaves reduced χ² ≈ 0.5 and doubles the F-statistic). The bcfitting
`calculate_noise_weighted_chi2` applied a 1.53 magnitude→complex factor for
the same reason; with a genuine complex per-bin σ the correct factor is just
√2 and nothing else.

The solver is `scipy.optimize.least_squares` with an **analytic Jacobian** of
`h_T` w.r.t. `(A, δ, φ)` per peak and the shared `τ`. The prototype derived
and verified that Jacobian (agreement with finite differences to ~3×10⁻¹⁰), so
production uses it from the start — no finite-difference phase. The Jacobian
also yields the parameter **covariance** → real frequency/amplitude/phase
uncertainties — an improvement over the surviving reference, which left
`freq_err = 0` for want of a covariance matrix.

### τ handling

`τ` is **shared per window** (one decay constant for all lines in a window) —
chirped-pulse FTMW lines in one acquisition see the same instrumental decay and
similar pressure broadening, so a shared `τ` lets weak lines borrow the
constraint from a strong one. Phase is **not** shared (next section).

- **Default / prior.** `τ_default` from the Stage 2b calibration: the per-band
  `τ_maj` for band-routed windows, else the band-wide `τ_maj`, else `T/3` when
  no Stage 2b calibration is present. Always applied up front as the starting
  value.
- **Bounds.** `[τ_default / k, τ_default · k]`, `k = max_decay_factor ≈ 5`
  (configurable; from the bcfitting reference).
- **Free vs fixed (O5-4).** `τ` is a free shared parameter when the window has
  an amplitude anchor strong enough to constrain it; for a window whose
  strongest line is below a configurable SNR threshold, `τ` is **held fixed**
  at `τ_default` (the bcfitting `fixed_decay_rate` path). A window of only weak
  lines cannot constrain `τ` and must not try.

### Independent phases — the blending caveat

Per-peak phase `φ` is **fully free and independent**. The oscillators are
coherently driven by a chirped pulse whose phase ramps rapidly with frequency;
there is **no** inter-peak phase relationship — and crucially, *no guarantee
that even blended, barely-resolved lines share a phase*. The model must never
couple phases, not even as a soft prior, and the blended-line test cases must
include adjacent peaks with deliberately unrelated phases.

### The conservative add-one-peak loop

Ported from `fit_weak_window_conservative_time_domain`:

1. Seed with the strongest promoted peak in the window; fit it.
2. Repeatedly: pick the strongest peak in the current residual among the
   window's remaining Stage 4 `free_peak_indices` candidates; trial-fit the
   model with it added; accept it only if the χ² improvement passes an
   **F-test** (`p < significance_threshold`, default 0.05) **and** the **AIC**
   decreases.
3. Stop when no candidate is accepted, the candidate list is exhausted, or a
   width/`max_peaks` cap is hit. A candidate that violates the peak-separation
   constraint (`min_separation = factor · FWHM(τ)`) is dropped.

**Addition, not subtraction** — the prior effort assessed both incremental
addition and incremental subtraction; addition won.

**Blend-aware seeding (the prototype's main algorithmic finding).** The plain
add-one-peak loop fits one cosine, lets it drift to a blend's centroid, then
adds the next from that drifted state — and for a tight near-equal in-phase
blend that sequential path lands in a degenerate basin and the loop reports
one line. The prototype
([`../research/stage5-fitting/report.md`](../research/stage5-fitting/report.md)
§4) showed the blend is *not* the problem: a blend is always statistically
detectable (a single cosine leaves an elevated reduced χ²) and, fit jointly
with a proper K=2 initialisation, is recovered to ~1 kHz down to 0.5 FWHM. The
failure is purely **initialisation**. So the loop carries a **blend-aware
seeder**: when a single-cosine fit at a seed leaves an elevated reduced χ²,
retry K=2 (then K=3) initialised at *two/three positions straddling the
feature*, not at the drifted centroid plus one candidate. A **patience**
parameter (O5-5) — tolerate a small number of consecutive rejections before
stopping — is kept as cheap insurance, but the prototype found it marginal:
the matched-filter F-test is decisive (a real line's integrated leakage energy
makes it overwhelmingly significant), so there is little
individually-insignificant / jointly-significant middle ground for patience to
exploit. The seeder, not patience, is the real fix for under-resolved blends.

As implemented in task 4 the blend-aware seeder runs on the **seed** — the
strongest line, the documented failure mode where the strongest detection is
itself a blend. A blend that surfaces *mid-loop* — a candidate added later
whose own single-cosine fit leaves an elevated residual — is currently left to
the plain add-one-peak path. Extending the seeder to re-seed any candidate
whose addition leaves an elevated local reduced χ² is a possible refinement;
it is **O5-9**, to be assessed on the real-data blended fixtures in task 10.

**Audit trail.** Every iteration records `{peak tested, F-statistic, p-value,
AIC before/after, separation check, decision, reason}`. This decision log is
persisted so the conservative loop's behaviour can be validated and curated
(it is the analogue of the Stage 3 detection provenance).

### Knockout validation

After a window's fit converges, a per-peak **knockout test**: remove one
fitted line from the model, hold all other parameters frozen, and check that
(a) the residual grows by the expected amount and (b) the residual reacquires
the shape of the absent line at its frequency. A line that can be knocked out
without the residual responding as predicted was not genuinely supported by
the data — a flag for curation. The knockout result is stored per fitted peak.

## Fixed contributors and DAG-driven execution

A Stage 4 `FixedContributor` names a strong line fit as a free peak in its
`primary_window_id`. Stage 5 fits windows in `WindowPlan.topological_order`;
`batch` groups mutually-independent windows for parallel execution. When a
window has fixed contributors, each contributor's `FittedPeak` is read from its
already-fitted primary window and its `h_T` term — evaluated at this window's
(out-of-band, far-skirt) frequencies with **frozen** parameters — is added to
the model and varied not at all. Because `h_T` is the exact leakage shape, a
well-fit strong line's frozen term reproduces its skirt here exactly.

The fidelity of this depends entirely on the strong line being fit *well* in
its primary window — see *The blending problem*. `freeze_eligible` (Stage 4's
`min_freeze_snr`, default 50) gates it: a contributor below the SNR cutoff is
flagged for the thaw handshake rather than trusted frozen.

## The blending problem (primary prototype investigation)

These spectra routinely contain peaks separated by less than, or comparable to,
the linewidth — barely resolved, down to a separation ≈ ½ the feature FWHM.
Apodization would resolve the ambiguity by sacrificing resolution; the whole
point of fitting the unwindowed spectrum is to **preserve maximal resolution**
and recover blended lines.

The risk this poses to the architecture: if a strong line is itself a
**partially-resolved blend**, fitting it as a single damped cosine
mis-estimates its skirt — and that wrong skirt is then frozen and carried into
every dependent window as a fixed contributor. **Compromised skirt estimates
break the fixed-contributor model.** The conservative loop is supposed to catch
the blend (a 1-peak fit leaves structured residual; a 2-peak fit passes the
F-test), but near ½-FWHM separation with independent phases this is exactly
where the add-one-peak significance test is weakest.

Blended lines are also usually **unequal in intensity**. The canonical case is
nitrogen quadrupole hyperfine structure — in a typical limit a 3:5:1 intensity
triplet that can blend into one feature. A weaker hyperfine component sitting
on the skirt of a stronger one is harder to detect and harder to recover
unbiased than the equal-intensity case, so intensity ratio is a third sweep
axis alongside separation and phase.

This must be investigated carefully in the prototype stage (task 1) before the
production algorithm is committed:

- How well are blended lines recovered as a function of separation (sweep down
  to ≈ ½ FWHM), phase difference (independent, swept), and **intensity ratio**
  (including the 3:5:1-type unequal triplet)?
- At what separation does the single-cosine fit of an unrecognised blend
  corrupt the frozen skirt enough to bias a dependent window's weak lines?
- Does the add-one-peak F-test reliably detect the blend — especially a weak
  component on a strong one's flank — at these separations, or is the patience
  parameter (O5-5) load-bearing here?

The user will supply additional real-data fixtures targeting blends; the
prototype works synthetic blends (known `A/f/φ/τ`, independent phase) first.

## D8 open items

- **Window baseline padding (D-6).** Stage 4 windows end exactly at their
  outermost peaks with no noise-only margin, and the Stage 4 invariant counts
  each spectrum point's residual once. Stage 5 reads a configurable
  `window_pad_mhz` of context on each side, used for model evaluation, the τ
  and baseline constraint, and display — but the **least-squares residual sum
  runs strictly over the Stage 4 `freq_range`**, so padding points are
  read-only context and the disjoint-coverage invariant is preserved. *Fallback
  (O5-3):* if fits perform poorly with context-only padding, escalate to true
  Stage 4 window widening plus an aggregation step; flagged, not adopted now.
- **The 36350/36389 doublet.** Decoupled at `T_edge = 8`; each is the other's
  fixed contributor. Stage 5 fits them independently first, then the
  residual-coherence check (below) decides whether to thaw-and-co-fit. This is
  the canonical renegotiation test case.
- **The 34154 MHz anomaly.** An SNR-172 promoted peak with anomalously low
  de-ramped `S_coh` (~3 vs ~12–27 for comparable lines) — possibly a blend or
  a mis-scored detection. The conservative loop diagnoses it naturally (a blend
  shows up as a significant second peak); it is carried as a named real-data
  test case and surfaced in diagnostics, with no special-case code.

## Renegotiation handshake with Stage 4

Stage 4 *classifies and proposes*; coupling visible only at fit time is
resolved here. After each window fit, a **residual edge-coherence check** runs
`S_coh` (the existing `preprocessing/edge_coherence` code) on the *fit
residual* at the window edges: coherent residual above threshold means the
boundary cut a real feature or a frozen contributor's skirt was carried badly.

Stage 5 then emits a typed re-plan request:

- **`thaw(window, contributor)`** — handled **locally** inside Stage 5: unfreeze
  the contributor (promote it to a free peak) and co-fit the two windows
  together. This covers `freeze_eligible = False` contributors and the
  36350/36389 doublet without a round-trip. Cheap and expected to be the common
  case.
- **`merge(window_a, window_b)`** — structural, changes window boundaries:
  routed through a new Stage 4 `replan(plan, requests) → WindowPlan` entry
  point that produces a revised plan with a bumped revision counter; Stage 5
  re-fits the affected batches. Triggered when a window edge cuts a real
  feature (the flagged residual edge has no fixed contributor to blame and
  the adjacent window's facing edge is also coherent).

**`split` is deliberately not implemented.** Stage 4 already proposes
windows tight enough that splitting structurally would be the wrong remedy
in practice; the real need it would address — "this window contains
multiple features that don't all need to be fit jointly" — is better
served by *intra-window* parameter clustering driven by the post-fit
covariance matrix (the joint fit's off-diagonal blocks reveal which peaks
are independent, so the model decomposes into a sum of independent
sub-fits with shared `τ` at no orchestration cost). Captured as its own
planning effort in
[`intra-window-clustering.md`](intra-window-clustering.md); not part of
task 7.

Renegotiation rounds are bounded (default ≤ 2) for guaranteed termination. The
full renegotiation history is recorded in the Stage 5 output (D-5).

## Data structures

`core/data_structures.py` already defines `FittedPeak`, `FittingResult`, and
the data-bearing `SpectralWindow`; Stage 5 wires them via
[`fitting/result_conversion.py`](../../src/ftmwpipeline/fitting/result_conversion.py)
and adds the plan-level `SpectrumFit` aggregate:

- **`FittedPeak`** — per line: `peak_id` (link back to the Stage 3 peak index),
  `frequency_mhz`, `amplitude`, `phase`, `decay_rate` (= 1/τ), their
  uncertainties from the covariance, `snr`, `chi_squared`; `extra_parameters`
  carries the knockout-test result and the originating window id.
- **`FittingResult`** — per window: `success`, `aic`, `reduced_chi2`,
  `iterations`, `fitted_peaks`, `shared_parameters` = {τ and its error},
  `fixed_parameters` = the frozen contributors used, `residuals`,
  `quality_metrics`. Extended to carry the per-iteration **add-one-peak audit
  trail** and the edge-coherence renegotiation outcome.
- **New aggregate** (parallels `WindowPlan`): a `SpectrumFit` dataclass holding
  the per-window `FittingResult`s, the merged global fitted-peak list, the
  renegotiation history, the parameters used, and plan-level diagnostics.

`SpectralWindow` is the natural data-bearing window object the fitter operates
on (it carries `freq_array`/`complex_spectrum`/`peaks`); Stage 5 materializes
one per fit window (padded per D-6) from the active-portion FT and the plan.

## Pipeline integration

- **Stage key / dependencies.** Add `stage5_fitting` to
  `PipelineStageTracker.STAGE_DEPENDENCIES` with dependencies on **both**
  `stage0_fid_data` (the raw FID — the active-portion FT is computed from it
  on demand) **and** `stage4_windows` (the plan), plus a `STAGE_DATA_PATHS`
  entry (`stage5_fitting`). The Stage 1 canonical settings (`start_us`,
  `end_us`, `trim`) parameterize the active-FT, so changes to canonical Stage 1
  settings invalidate Stage 5 through the existing
  canonical-settings mechanism; Stage 3 re-detection and Stage 4 re-planning
  also propagate via `invalidate_downstream_stages`. The active-portion FT
  itself is *not* persisted: it is computed on demand from the FID and is
  trivially regenerable.
- **Serialization.** Persist a `/stage5_fitting` group. The fitted parameters
  are scientific output and may be hand-edited (like the peak list and the
  window plan), so they are **persisted, not recomputed**: flat, hand-editable
  layout, loud validation, round-trip + hand-edit contract. Heavy recomputable
  arrays (the fitted complex spectrum, the residual) are **not** persisted —
  consistent with the SERIALIZATION spec and the D6 precedent for ComplexFT;
  they are regenerated from the stored parameters on demand. The add-one-peak
  audit trail and renegotiation history are persisted as JSON. Confirm the
  persist/recompute split against `SERIALIZATION_STRATEGY.md` (O5-8).
  Implementation: `io/fitting_serialization.py`, mirroring
  `io/window_serialization.py`.

## Interface surface (dual-interface rule)

Logic in `_internal/stage5_impl.py` (orchestration) and the `fitting/` package
(algorithm); thin identical wrappers across all three interfaces:

- `Pipeline.fit_peaks()` / `api.fit_peaks()` / CLI `fit run`.
- `Pipeline.visualize_fit()` / `api.visualize_fit()` / CLI `fit show`,
  and `load_fit()` / `api.load_fit()`. The SNR-aware shape-error report is
  `fit check`.

Cross-interface consistency tests are mandatory
(`tests/integration/test_stage5_fitting.py`).

Algorithm module layout under `src/ftmwpipeline/fitting/` (replacing the
existing `NotImplementedError` stubs):

- `peak_model.py` — `h_T`, its Jacobian, the demodulation / sideband mapping,
  the analytic-vs-numerical-FFT cross-check.
- `window_fit.py` — the per-window least-squares core (the recreated
  `fit_time_domain_peaks` contract) and the conservative add-one-peak loop with
  the F-test/AIC/separation/patience logic and the knockout test.
- `validation.py` — physics constraints, the statistical-test helpers ported
  from the bcfitting shell.

## Visualization

Stage 5 needs clear per-window diagnostic figures. Based on the prior effort's
figures, the recommended panels are:

- real and imaginary parts of the **complex FT, model-on-data**, with their
  residuals;
- the magnitude spectrum with its residual;
- the time-domain model (for intuition);
- a rendering of the **add-one-peak audit trail** — the decision flow, so the
  conservative loop's accept/reject choices can be validated at a glance.

The prior effort's "time-domain residual against the spectral IFFT" panel is
**dropped** as unnecessary. `visualization/fit_visualization.py`, wired through
`fit show`.

## Test plan

Synthetic ground truth **before** any real-data fitting (mandated by the Stage
3 doc):

- **Sideband mapping.** Isolated lines of known `A/f/φ/τ` synthesized on both
  the lower and upper sideband, carried through the real Stage 0–4 pipeline,
  fit; parameters recovered to tolerance. This is the test that guards the
  `s`-sign / demodulation derivation.
- **Model unit tests.** `h_T` analytic form vs a numerical FFT of the
  effective-time damped cosine; the analytic Jacobian vs finite-difference; the
  `deramp_to_active_start` round-trip.
- **Blending** (the central investigation). Blended lines swept in separation
  down to ≈ ½ FWHM, with independent (swept) phases and **unequal intensities**
  — including a 3:5:1-type triplet (nitrogen quadrupole hyperfine): blend
  recovery accuracy; the separation at which an unrecognised blend's frozen
  skirt biases a dependent window; and whether the **blend-aware seeder**
  recovers the line count where the plain add-one-peak loop's sequential
  initialisation does not (the prototype's key finding).
- **Fixed contributors.** A strong line and, in a *separate* window, a weak
  line on its skirt; the strong line is fit, frozen, and carried — the weak
  line's parameters must come back unbiased. Repeated with the strong line a
  partially-resolved blend (the failure mode).
- **τ handling.** A weak-only window falls back to fixed `τ_default`; a window
  with a strong anchor fits `τ` free and recovers it.
- **Conservative loop.** The F-test/AIC accept/reject logic; the blend-aware
  seeder retrying K=2/K=3 where a single-cosine fit leaves an elevated reduced
  χ²; the knockout test grows the residual as predicted.
- **Renegotiation.** A synthetic coupled pair triggers the residual
  edge-coherence check → `thaw` → co-fit; a structural case exercises the Stage
  4 `replan` entry point.

Real data — the 2638 fixture:

- Work an initially **representative subset** of the ~328-window plan (a
  handful spanning easy/hard, isolated/blended/fixed-contributor, plus the
  36350/36389 doublet and the 34154 anomaly) — the full plan is the final
  integration check, not the development loop.
- Sane fitted line list; the doublet exercises renegotiation; the 34154
  anomaly is diagnosed (blend or mis-score) by the conservative loop.

Cross-cutting: cross-interface identity (CLI / Pipeline / api); serialization
round-trip + hand-edit; `stage5_fitting` invalidation on Stage 1
canonical-settings change, Stage 3 re-detection, and Stage 4 re-planning.

## Open questions

- **O5-1 — model realization. RESOLVED (prototype).** Analytic `h_T` on the
  window grid is the model — confirmed against a literal numerical FFT of a
  synthesized FID to a relative error of 7×10⁻³ (entirely the numerical FFT's
  finite-grid error; the closed form is exact). Used uniformly for free peaks
  and fixed-contributor skirts. See
  [`../research/stage5-fitting/report.md`](../research/stage5-fitting/report.md) §2.
- **O5-2 — blending and skirt fidelity. INVESTIGATED (prototype); one item
  open.** A blend fit jointly with the correct line count is recovered to
  ~1 kHz down to 0.5 FWHM across phase and intensity ratio, and is always
  statistically detectable. The failure mode is the add-one-peak loop's
  *sequential initialisation* — addressed by the blend-aware seeder (see *The
  conservative add-one-peak loop*). An unrecognised blended fixed contributor
  biases dependent windows by ~1 kHz. Open: validating the seeder + residual
  edge-coherence flag on real blended fixtures.
- **O5-3 — padding fallback. RESOLVED (not triggered, issue #8).** The fallback
  (escalate context-only padding (D-6) → Stage 4 window widening + aggregation)
  was conditioned on fits performing poorly with context-only padding. They do
  not: context-only padding holds at χ²ᵣ median ~1.2 across all 7 same-instrument
  fixtures *and* the cross-instrument succinimide fixture. Reopen only if a
  future fixture degrades.
- **O5-4 — τ free-vs-fixed threshold. RESOLVED (issue #8).** Shipped as the dual
  floor `max(fit_tau_min_snr, weak_window_snr_threshold)` (both default 10) in
  `window_fit`. Calibrated on 2638 and **cross-instrument audited** (issue #6):
  on succinimide, raising `fit_tau_min_snr` 10→100 cuts free-τ windows with χ²ᵣ
  unchanged, so the default 10 stands.
- **O5-5 — patience parameter. ASSESSED (prototype): marginal.** Kept as
  cheap insurance (default 1) but the matched-filter F-test is decisive, so
  patience rarely changes an outcome; the blend-aware seeder is the real fix
  for under-resolved blends. Tunable left as a parameter.
- **O5-6 — thaw protocol details. RESOLVED (issue #8).** The thaw +
  structural-replan machinery shipped with `residual_edge_threshold`=8,
  `max_thaw_rounds`=`max_replan_rounds`=2, and was cross-instrument audited
  (issue #6): thaw is near-dormant (acceptance ~0) on both 2638-family and
  succinimide — a sane trigger surface, not a mistune.
- **O5-7 — parameter uncertainties. RESOLVED (prototype).** The analytic
  Jacobian of `h_T` is verified (finite-difference agreement ~3×10⁻¹⁰); use it
  from the start, with its covariance for the parameter uncertainties. No
  finite-difference phase.
- **O5-8 — persist vs recompute. RESOLVED (task 9).** Implementation
  confirmed the split against `SERIALIZATION_STRATEGY.md`: per-peak
  fitted parameters, shared τ, frozen-contributor summaries, audit
  trail, thaw / replan histories, plan revision, Stage 5 parameters,
  and diagnostics are persisted; the per-window `SpectralWindow` (the
  active-FT slice), the fitted complex spectrum, and the complex
  residual are recomputed on load.
- **O5-9 — mid-loop blend-aware seeding. DEFERRED (no evidence of the failure
  mode, issue #8).** Task 4's blend-aware seeder runs on the seed only. Whether
  the loop also needs to re-seed a *mid-loop* candidate whose own single-cosine
  fit leaves an elevated local reduced χ² (a blend that is not the strongest
  line) remains open as a refinement — but no current fixture exhibits it
  (eps_p50=0 / no shape-error windows across the 7 same-instrument fixtures and
  succinimide). Reopen if a blended fixture shows a mid-loop candidate the
  add-one-peak path misses.
- **O5-10 — untreated fixed-contributor skirt leakage.** A window may sit
  downstream of one or more strong lines whose Lorentzian skirts carry a
  small but signed amount of power into the window. The complex residual
  then shows a structured offset (e.g. mean(Im) systematically below
  zero), and the |residual| histogram drifts away from the Rayleigh
  reference. The symptom is concrete on the clean K=1 window w337.

  **Diagnostic findings (2638 fixture).** The bias is **pervasive**, not
  a w337-only issue:

  | metric | initial | after cumulative-skirt subtract |
  |---|---|---|
  | median \|mean(Im)\|/σ_c | 0.31 | 0.17 |
  | p95 \|mean(Im)\|/σ_c | 1.09 | 0.62 |
  | windows with \|init Im\| > 0.5σ | 31% | — |
  | windows where subtract reduces \|Im\| | — | 69% |
  | windows reaching \|after Im\| < 0.1σ | — | 31% |

  Roughly half of all 347 windows have measurable Im-bias from external
  skirts. w337's visibility is incidental — it's a clean K=1 EASY window
  where the signature shows through unobscured; HARD windows hide the
  same effect under in-window fit structure.

  The bias source is the **coherent sum of many far-line skirts**, not
  any single nearby strong contributor. w337 has zero `fixed_contributors`
  attached, yet ~650 fitted peaks across the whole spectrum sum into
  -0.60σ Im + 0.42σ Re of cumulative skirt on its grid. The DC-leakage
  hypothesis (suggested by w337's 1.35 GHz proximity to probe) does not
  hold — the residual bias does not track DC proximity on the
  w337..w346 ladder, and the active-FT's mean-removal (rdc) suppresses any
  constant DC offset before the FFT.

  **Root cause.** Stage 4's contributor-attachment rule attaches a
  strong line `s` as a `FixedContributor` of window `w` only when `w`
  overlaps a rolling-coherence-touched region containing `s`. The
  rolling-coherence statistic identifies *runs of coherent leakage above
  threshold*; weak-but-coherent skirt power from lines 20+ MHz out of
  band does not produce a touched region that reaches `w`, so those
  contributors are never attached. The cumulative tail of those un-
  attached contributors is the bias.

  **Fix (v1, single-tier — magnitude-based attachment).** Replace the
  touched-region gate in `_finalize_plan` step 4 with an analytic
  skirt-magnitude rule: for every (strong promoted peak `s`, candidate
  window `w`) pair, predict `s`'s mean |skirt| on `w`'s grid via

  ```
  predicted_mean_skirt ≈ (s.intensity / tau_eff) × |h_T(f_s - f_w_center, τ, T)|
  ```

  and attach `s` as a `FixedContributor` of `w` when
  `predicted_mean_skirt ≥ T1_threshold × σ_c(w)` (default
  `T1_threshold = 0.1`). The dependency-edges / primary-window /
  thaw machinery is unchanged; only the attachment criterion changes.

  Per the contributor-count survey, this gives manageable per-window
  lists:

  | threshold | median | p75 | p95 | max |
  |-----------|--------|-----|-----|-----|
  | 0.05 σ_c  | 6      | 10  | 15  | 19  |
  | **0.1 σ_c**  | **2**  | **5** | **10** | **16** |
  | 0.2 σ_c   | 1      | 3   | 6   | 13  |

  At 0.1 σ_c the median attached count is 2, p95 is 10, max is 16. The
  coherent-bias capture is good: top-5 contributors (by predicted
  skirt magnitude) capture ≥80% of the Im bias in 64% of windows and
  ≥50% in 88%. The median count to reach 90% of |coherent Im bias|
  is 3.

  **Fix (v2, future — Tier-2 cumulative-tail background).** For the
  ~12% of windows where the long tail still carries measurable bias
  after Tier-1 attachment, add a per-window "external bias spectrum":
  one complex array computed as the sum of `model_spectrum`
  evaluations over every promoted peak NOT in Tier-1, subtracted from
  the window data before the per-window fit runs. Two implementation
  options:

  - **Plan-time, Stage 3 intensities only.** Compute the background
    in `_finalize_plan` from each contributor's `intensity` (peak FT
    magnitude) and an assumed-zero phase (the Lorentzian magnitude
    contribution is phase-invariant when integrated over a wide
    window; the *coherent* contribution depends on phase, which is
    unknown until fit time). This captures the magnitude budget but
    not the sign — only suitable as a chi²-correction, not as a
    bias-subtraction.
  - **Stage 5 multi-pass.** After Pass 1 (with Tier-1 contributors
    only), compute Tier-2 per window from all OTHER windows' fitted
    amplitudes + phases. Refit each window with Tier-1 + Tier-2
    subtracted. Iterate to convergence (2 passes typically suffice).

  v2 is deferred until v1 validation shows where the residual long-
  tail bias actually lives. Likely Tier-2 is unnecessary for the 2638
  fixture; instruments with denser line forests may need it.

  **Sequencing note:** this item is the prerequisite for the
  generalised effective-DoF / AICc-with-`n_eff` work covered in
  [`stage5-residual-rescue.md`](stage5-residual-rescue.md) and
  [`../research/residual-rescue/report.md`](../research/residual-rescue/report.md).
  Contributor-skirt leakage contaminates the per-window χ² and noise-
  floor baseline that every Stage 5 hypothesis test calibrates
  against; tuning effective-DoF thresholds against a contaminated
  baseline means re-tuning afterward. Land O5-10 (Tier-1) first.

  **v1 status: landed.** Tier-1 magnitude-based attachment is wired
  through `_finalize_plan` (`src/ftmwpipeline/preprocessing/window_planning.py`),
  the plan parameter is persisted in `WindowPlan.parameters`, and the
  cycle-breaker now also prunes orphaned `FixedContributor` entries
  whose dependency edge was dropped (two strong lines whose skirts
  mutually exceed threshold form a 2-cycle that needs both edges
  dropped; without the prune, execute_plan tripped on "un-fit primary"
  at fit time). 25 unit tests pass (3 new for magnitude-attach + 1 new
  for cycle-prune) and the full integration suite (103 tests) passes.

  2638 fixture validation: 71/347 windows (20.5%) now have at least one
  fixed_contributor (was effectively zero outside the small subset
  Stage 4's touched-region rule caught), 123 contributors attached
  total. 68 dependency edges were dropped as cyclic (adjacent-window
  strong-line pairs whose skirts mutually exceed threshold). Spot
  checks:

  | window | before chi²_r | after chi²_r | FC attached |
  |--------|---------------|---------------|-------------|
  | w337   | 1.503         | **0.985**     | 2 (from w336 doublet) |
  | w194   | (high)        | **0.916**     | 2 (from w193) |
  | w270   | (high)        | **21.9**      | 1 (from w269) |
  | w281   | (high)        | **4.18**      | 2 (from w282) |
  | w200   | 4182.9        | 4182.9        | 0 — unresolved-doublet misfit, unrelated to O5-10 |
  | w148   | (high)        | 715           | 0 — rescue-overfit pathology, unrelated to O5-10 |

  Per-window chi²_r p50/p75/p95 = 1.43 / 2.33 / 7.73 after Tier-1.
  Residual elevation above 1.0 in HARD windows is now dominated by
  rescue-overfit / unresolved-doublet pathologies (the agenda for the
  rescue-AICc / phase-degeneracy work that follows this fix), not by
  external-skirt leakage.

  **Tier-2 disposition.** The residual *coherent* bias Tier-2 would target
  is absorbed per-window by the evidence-triggered leakage-wing baseline
  ([`stage5-leakage-wing-baseline.md`](stage5-leakage-wing-baseline.md)),
  which prices a neighbouring strong line's mismodeled skirt as a low-order
  complex nuisance term inside the joint fit (so the per-line covariance pays
  the cost). Tier-2 cumulative-tail subtraction stays deferred; the wing
  baseline covers the practical need on 2638.

- **O5-11 — phase-degeneracy penalty for the conservative-fit
  residual.** The pair penalty in
  `window_fit._penalty_residuals_and_jacobian` is
  `sqrt(λ) · w(Δsep) · cos(φᵢ - φⱼ)` — fires at both Δφ=0 (in-phase
  degeneracy: two peaks at the same offset summing to a single
  feature's amplitude — the w148 A/C and B/D pairs before the AICc
  merge gate, and the LSQ-basin difference between w198's K=4 and
  K=7 outcomes) and Δφ=π (cancellation: destructive interference
  between two inflated amplitudes), zero only at Δφ=π/2
  (quadrature, the configuration where two close peaks carry
  independent information). User framing: "Our best bet for fitting
  blended features would likely occur when their phases are in
  quadrature." The `weight = max(0, 1 - sep / cutoff)` closeness
  factor (default cutoff 2 FWHM) keeps the penalty active only for
  pairs near the resolution limit.

  This is the LSQ-side complement to the AICc-gate work in
  [`stage5-residual-rescue.md`](stage5-residual-rescue.md): the
  AICc gates catch duplicate-pair overfit after the fact; the
  phase-degeneracy penalty prevents the optimiser from landing in
  the duplicate basin in the first place. With this penalty in
  place, re-enabling the AICc-gated outer tier of
  `merge_close_peaks_cleanup` (currently disabled by setting
  `DEFAULT_MERGE_SEPARATION_FACTOR = 0.5`, equal to the structural
  threshold) becomes safe — tier 2 can decide between merging and
  keeping real close pairs on chi-squared evidence with the
  penalty providing the LSQ-side distinguishing signal.

  **Status: landed.** Single quadrature-only `cos(φᵢ - φⱼ)` form
  replaces the prior anti-phase-only `sin((φᵢ - φⱼ)/2)` term;
  `DEFAULT_PHASE_PENALTY_LAMBDA = 100` and the 2-FWHM cutoff carry
  over unchanged. The analytic Jacobian is regression-tested
  against finite differences in
  `tests/unit/fitting/test_window_fit.py::TestPairPhasePenalty`.
  On 2638 the change produced +2 accepted fitted peaks, dropped
  global χ²_r p95 from 5.99 to 5.14, and improved targeted windows
  (w293: 3.00 → 2.74) without regressing the heavy-tail cases —
  w140/w216 do not move because their high χ² is driven by
  shape-mismatch / tau-collapse pathologies, not phase degeneracy
  (w216's three peaks land at non-degenerate Δφ; w140 collapses
  tau to ≈3µs against its ~5µs τ upper bound).

## Implementation map

The stage is built across the `fitting/` package (algorithm) and
`_internal/stage5_impl.py` (orchestration):

- **`fitting/peak_model.py`** — `h_T`, its analytic Jacobian, and the
  demodulation / sideband mapping (`to_baseband_offset`; a wrong sign is a
  200–400 kHz silent bias), with the literal numerical FFT retained as a
  unit-test cross-check of the closed form.
- **`fitting/active_ft.py`** — `compute_active_ft(...) -> ActiveFTResult`
  (`freq_mhz`, `complex_spectrum`, `alpha`, `n_active`, `n_padded`): the
  unapodized `rfft(fid[active])` in the `[0, T_active]` frame (no phase ramp).
  Stage 5 fits on this active-portion FT so bins are independent and reduced
  χ² / F-test / AIC are calibrated as written (D9).
- **`fitting/window_fit.py`** — the per-window least-squares core (the recreated
  `fit_time_domain_peaks` contract) with shared/fixed τ and parameter
  covariance, plus the conservative add-one-peak loop (F-test + AIC + separation
  + patience, the blend-aware K=2/K=3 seeder, the audit trail, the knockout
  test). The fit is noise-weighted by σ/√2 on the active-FT scatter σ.
- **`fitting/plan_execution.py`** — fixed-contributor materialization (the
  frozen skirt is subtracted before the fit so the core stays free-peak-only),
  the DAG walk over `WindowPlan.topological_order`, the residual-edge-coherence
  `thaw` renegotiation with joint-frame local co-fit, and `replan(plan,
  requests)`: **merge-only** (`MergeRequest` unions two adjacent windows and
  drops now-internal contributors; the `SplitRequest` primitive is deferred to
  [`intra-window-clustering.md`](intra-window-clustering.md)), tracked by a
  `plan_revision` counter.
- **`fitting/validation.py`** — physics constraints and the statistical-test
  helpers ported from the bcfitting shell.
- **`fitting/result_conversion.py`** — pure converters into the persistent twins
  (`AuditStep`, `KnockoutInfo`, `ThawInfo`, `ReplanInfo` in `core/`).
  `SpectrumFit` aggregates the per-window `FittingResult`s plus the merged
  global fitted-peak list, sorted by molecular frequency with each peak tagged
  by its originating window id.
- **`io/fitting_serialization.py`** — a flat hand-editable layout sibling to the
  peak/window serializers (resolves O5-8 against `SERIALIZATION_STRATEGY.md`):
  the per-peak parameters + uncertainties, shared τ, frozen-contributor
  summaries, audit trail, thaw + replan histories, final plan revision, and the
  Stage 5 parameters used are **persisted**; the active-FT slice, fitted
  spectrum, and complex residual are **recomputed on load** (the
  lightweight-file invariant). The merged global peak list is rebuilt from the
  per-window peaks on load, so a hand-edit propagates.
- **Stage tracking** — `stage5_fitting` depends on both `stage0_fid_data` and
  `stage4_windows`, so the existing `invalidate_downstream_stages` path drops
  Stage 5 results whenever Stage 0 or Stage 4 re-runs.
- **Interfaces** — `Pipeline.fit_peaks/visualize_fit/load_fit`, `api.*`, CLI
  `fit run` / `fit show` / `fit check`, over `visualization/fit_visualization.py`
  (overview mode overlays the fitted model on the persisted high-res FT;
  per-window detail mode via `--window-id` shows real/imaginary model-on-data
  with residuals, the magnitude residual, a synthesised time-domain envelope,
  and the add-one-peak audit trail).

Window baseline padding (D-6) is deferred — the bare Stage 4 `freq_range` slice
is what the `SpectralWindow` carries. The research prototype (`h_T` / Jacobian
verification, the sideband-sign bias, blend recovery, σ/√2 weighting, the
knockout test; resolved O5-1, O5-7) is archived at
[`../research/stage5-fitting/`](../research/stage5-fitting/report.md).
