# Plan: Stage 5 — Per-window fitting

Status: **planning (not started).** This is a step-1 planning document per
[`README.md`](README.md): it describes the approach, the model, the algorithm,
the interface surface, the serialization, and the test plan before any
implementation. Registered in [`../ROADMAP.md`](../ROADMAP.md).

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

The persisted Stage 1 spectrum is an rfft of the *whole* zero-padded record
(`zpf=2`, `N_active ≪ N_padded` for 2638: ~632k active samples in a 1.5M-bin
FFT). That representation is right for **display, peak detection, and window
planning** — sub-bin centroid interpolation helps the detector, and the
edge-coherence statistic is calibrated on the high-resolution grid. But it is
wrong for **statistical fitting**: adjacent bins are correlated by a Dirichlet
kernel (the FFT of the zero-padding indicator), so the effective number of
independent samples in any band of `M` bins is

    M_eff = M · α,   α = N_active / N_padded   (≈ 0.42 for 2638),

and the naive `N_dof = M − N_params` overcounts by `1/α`. That bias
propagates into reduced χ², the F-test, and AIC — all in the direction of
optimism — and silently miscalibrates the conservative loop's accept/reject
decisions and the blend-aware seeder's `rchi2 > 1.5` trigger.

**Stage 5 fits in the active-portion FT frame** to dissolve the problem at
source. The fit-time spectrum is the rfft of just the active-region samples
of the persisted FID, with the same `expf_us` apodization as the canonical
Stage 1 settings:

    active_ft[k] = rfft( fid[t0:t0+T] · exp(-(t − t0)/τ_apod) )[k]

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
  `1/α ≈ 2.4`.
- **The de-ramp is a no-op.** `to_baseband_frame` no longer needs the
  `deramp_to_active_start` step; the active-FT is the `[0, T]` form
  natively. The de-ramp helper stays in `preprocessing/leakage.py` for
  Stage 4's edge-coherence work, but Stage 5 stops calling it.
- **The σ/√2 D-8 noise weighting still applies.** It splits the per-bin
  complex variance into Re/Im halves; it is independent of the bin-
  correlation question.
- **Per-bin noise comes from Stage 2 via a derived rescale**, not a fresh
  estimation pass. The active-FT bin variance is `1/α` that of the
  persisted FT (same time-domain noise, fewer FT-length samples), so

      σ_active(f) = σ_persisted(f_nearest) / √α

  interpolated onto the active-FT grid. No second adaptive noise pass; one
  canonical `rms_noise` source.
- **The active-FT is internal to Stage 5.** It is computed on demand from
  the persisted FID (`stage0_fid_data`) and the canonical Stage 1
  apodization settings; it is not persisted in the `.ftmw` file (small,
  fast, and trivially regeneratable). Stage 5 therefore gains an explicit
  dependency on Stage 0 alongside Stage 4.
- **Display still uses the persisted FT.** Visualization overlays the
  fitted model on the high-resolution persisted grid by re-evaluating
  `model_spectrum` at those frequencies; the model is grid-agnostic.

The persisted-FT path that the task-5 implementation took is preserved in
the algorithm module as a fall-back-debugging surface but is no longer the
production fit frame; the migration is task 6 (see *Task breakdown*). The
active-portion FT contract is registered in
[`../ROADMAP.md`](../ROADMAP.md) as divergence **D9**.

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
`[t₀, t₀+T]` FID samples, with the canonical Stage 1 apodization. By
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
`exp(±i2π f_bb t₀)` because it is the rfft of the *whole* zero-padded record.
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
The pipeline's goal is unwindowed fitting and 2638's canonical settings have
`winf=None`; Stage 5 asserts `winf is None` and refuses with a clear error
otherwise rather than carrying an apodization kernel through the model (D-3).

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

The active-portion FT bin noise is derived from the canonical Stage 2
per-bin complex RMS by

    σ_active(f) = σ_persisted(f_nearest) / √α

(see *Spectral domain for the fit*). The active-portion bins are independent,
so summing their squared residuals gives a sum that is exactly χ²-distributed
with `M − N_params` degrees of freedom. Real and imaginary parts of σ_active
each carry variance σ_active² / 2, so the stacked Re/Im residual is weighted
by **σ_active/√2** for every element to be unit-variance — then reduced χ²
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
chirped-pulse FTMW lines in one acquisition see the same apodization and
similar pressure broadening, so a shared `τ` lets weak lines borrow the
constraint from a strong one. Phase is **not** shared (next section).

- **Default / prior.** `τ_default` from the apodization: when the canonical
  Stage 1 `expf_us` is set the exponential filter dominates, so
  `τ_default ≈ expf_us` (≈ 5 µs on 2638, ≈ T/2.5); otherwise `τ_default ≈ T/3`.
  Always applied up front as the starting value.
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
- **`merge(window_a, window_b)`** / **`split(window, freq)`** — structural,
  changes window boundaries: routed through a new Stage 4 `replan(plan,
  requests) → WindowPlan` entry point that updates the persisted plan in place
  and bumps a plan-revision counter; Stage 5 re-fits the affected batches.

Renegotiation rounds are bounded (default ≤ 2) for guaranteed termination. The
full renegotiation history is recorded in the Stage 5 output (D-5).

## Data structures

`core/data_structures.py` already defines `FittedPeak`, `FittingResult`, and
the data-bearing `SpectralWindow` — defined but not wired. Stage 5 wires them
and adds a plan-level aggregate:

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
  `end_us`, `expf_us`, `winf`, `zpf`) parameterize the active-FT, so changes
  to canonical Stage 1 settings invalidate Stage 5 through the existing
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

- `Pipeline.fit_windows()` / `api.fit_windows()` / CLI `fit-windows`.
- `Pipeline.visualize_fit()` / CLI `visualize-fit`, and `load_fit()`.

`fit-windows` / `visualize-fit` follow the verb-object CLI scheme (D1); the
names are reserved in `CLI_STRATEGY.md` alongside the other Stage 3+ commands.
Cross-interface consistency tests are mandatory
(`test_cross_interface_consistency.py`).

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
`visualize-fit`.

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
- **O5-3 — padding fallback.** Whether context-only padding (D-6) suffices, or
  Stage 4 window widening + aggregation is required. Decided empirically once
  fits run.
- **O5-4 — τ free-vs-fixed threshold.** The strongest-line SNR below which a
  window holds `τ` fixed at `τ_default`. Calibrated on 2638.
- **O5-5 — patience parameter. ASSESSED (prototype): marginal.** Kept as
  cheap insurance (default 1) but the matched-filter F-test is decisive, so
  patience rarely changes an outcome; the blend-aware seeder is the real fix
  for under-resolved blends. Tunable left as a parameter.
- **O5-6 — thaw protocol details.** The exact trigger thresholds for the
  residual edge-coherence check and the bound on renegotiation rounds.
- **O5-7 — parameter uncertainties. RESOLVED (prototype).** The analytic
  Jacobian of `h_T` is verified (finite-difference agreement ~3×10⁻¹⁰); use it
  from the start, with its covariance for the parameter uncertainties. No
  finite-difference phase.
- **O5-8 — persist vs recompute.** Confirm the parameters-persisted /
  arrays-recomputed split against `SERIALIZATION_STRATEGY.md` during
  implementation.
- **O5-9 — mid-loop blend-aware seeding.** Task 4's blend-aware seeder runs on
  the seed only. Whether the loop also needs to re-seed a *mid-loop* candidate
  whose own single-cosine fit leaves an elevated local reduced χ² (a blend that
  is not the strongest line) is open — to be assessed on the real-data blended
  fixtures in task 10, alongside the open part of O5-2.

## Task breakdown

1. [x] **Research prototype** — `h_T` and its Jacobian verified; the
   demodulation/sideband mapping derived (a wrong sign is a 200–400 kHz silent
   bias); blending investigated (joint recovery ~1 kHz to 0.5 FWHM; the loop's
   sequential initialisation, not detectability, is the failure → blend-aware
   seeder); fixed-contributor mis-fit bias ~1 kHz; σ/√2 noise weighting and the
   knockout test established. Archived in
   [`../research/stage5-fitting/`](../research/stage5-fitting/report.md)
   (`prototype.py` + `report.md` + figures). Resolved O5-1, O5-7; informed
   O5-2, O5-5, D-8.
2. [x] `fitting/peak_model.py` — `h_T`, Jacobian, demod/sideband mapping,
   de-ramp integration + unit tests (both sidebands).
3. [x] `fitting/window_fit.py` — per-window least-squares core (the recreated
   `fit_time_domain_peaks` contract), shared/fixed τ, parameter covariance +
   unit tests.
4. [x] Conservative add-one-peak loop — F-test + AIC + separation + patience +
   the **blend-aware seeder** (retry K=2/K=3 on elevated single-cosine χ²),
   the audit trail, the knockout test; `fitting/validation.py` helpers ported
   from the bcfitting shell + unit tests. Mid-loop re-seeding deferred as O5-9.
5. [x] Fixed-contributor evaluation + DAG/batch execution order + local `thaw`
   renegotiation + unit tests. Algorithm landed in
   [`fitting/plan_execution.py`](../../src/ftmwpipeline/fitting/plan_execution.py)
   (FrozenPeak materialization with primary's refined frequency, frozen-skirt
   subtraction-before-fit so `conservative_fit` stays free-peak-only, DAG walk
   over `WindowPlan.topological_order`, residual edge-coherence trigger, and
   the joint-frame local co-fit that promotes the thawed contributor to a free
   peak in both windows on accept). 22 unit tests including a coupled-pair
   thaw integration. `scratch/stage5_thaw_demo.py` shows the CLEAN vs COUPLED
   contrast end-to-end. **Implemented against the persisted Stage 1 FT;
   superseded by the active-portion FT contract in task 6 below (D9).** The
   algorithm survives the rewiring almost intact; only the input frame
   changes.
6. [ ] **Active-portion FT migration (D9).** Switch Stage 5 from fitting on
   the persisted zero-padded FT to fitting on the active-portion FT, so
   bins are independent and reduced χ², F-test, AIC are calibrated as
   written. Subtasks:
   1. `fitting/active_ft.py` (new algorithm module):
      `compute_active_ft(fid, sample_dt_us, *, start_us, end_us, expf_us,
      probe_freq_mhz, sideband) -> ActiveFTResult` with
      `(freq_mhz, complex_spectrum, alpha, n_active, n_padded)`. The FFT is
      `rfft(fid[active] * exp(-(t - t0)/τ_apod))` so the bin grid is
      `[0, T_active]`-natural — no phase ramp.
   2. `derive_active_noise(persisted_rms_noise, persisted_freq_mhz,
      active_freq_mhz, alpha) -> np.ndarray`: interpolate persisted Stage 2
      σ onto the active-FT grid, rescale by `1/√α`. One canonical noise
      source; no second adaptive pass.
   3. Rewire `plan_execution._materialize_window` to slice the active-FT
      result instead of the persisted FT. Drop the
      `deramp_to_active_start` call from `to_baseband_frame` (the helper
      stays in `preprocessing/leakage.py` for Stage 4). Rename
      `to_baseband_frame -> to_baseband_offset` for clarity.
   4. `execute_plan` takes an `ActiveFTResult` and active-grid `rms_noise`
      instead of the persisted FT + per-bin noise. The
      `_internal/stage5_impl.py` orchestrator (task 9) computes the
      active-FT once per Stage 5 invocation from `stage0_fid_data` +
      canonical Stage 1 settings.
   5. Tests: `tests/unit/fitting/test_active_ft.py` (synthetic damped
      cosines, recovery of A/f/φ/τ; verify α and per-bin σ_active relation
      to within ~5% on noise-only data; verify absence of phase ramp by
      comparing bin phases at line center). Update
      `tests/unit/fitting/test_plan_execution.py` for the new
      `execute_plan` signature.
   6. Update `STATUS.md` (when wired) and confirm no α-correction is
      smuggled in; the helpers in `validation.py` stay as-written.
7. [ ] Stage 4 `replan` entry point (`merge`/`split`) + the residual
   edge-coherence renegotiation handshake + unit tests. (Implemented
   against the active-FT frame established in task 6.)
8. [ ] Data-structure wiring — `FittedPeak`/`FittingResult`/`SpectralWindow` +
   the new `SpectrumFit` aggregate + unit tests.
9. [ ] `io/fitting_serialization.py` + `stage5_fitting` stage tracking and
   dependencies (depends on `stage0_fid_data` AND `stage4_windows`) +
   invalidation wiring + hand-edit round-trip tests.
10. [ ] Wrappers (`Pipeline.fit_windows/visualize_fit/load_fit`, `api.*`, CLI
    `fit-windows`/`visualize-fit`) + `visualization/fit_visualization.py`.
    Visualization overlays the fitted model on the persisted (high-res) FT
    by re-evaluating `model_spectrum` at the persisted frequencies.
11. [ ] Cross-interface + 2638 representative-subset integration tests; the
    doublet and 34154 cases; then the full-plan integration check.
