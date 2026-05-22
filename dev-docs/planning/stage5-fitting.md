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

Conceptually the model is the FFT of a sum of damped cosines on an *effective
time axis*: the data to fit is a small-bandwidth slice of the persisted
spectrum (M contiguous bins), and the inverse-DFT of an M-bin slice is an
M-point time series at a decimated sample spacing — short, because the window
bandwidth is small — spanning the full record duration, with the active signal
occupying the sub-interval `[t₀, t₀+T]`. The model is a sum of damped cosines
built on that axis and transformed back.

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
`exp(±i2π f_bb t₀)`. `h_T` as written is the `[0,T]` form. Stage 5 de-ramps
the window data once with the existing
`preprocessing/leakage.py:deramp_to_active_start` (sideband-proof via
`|f − f_probe|`) and fits the `[0,T]` `h_T`. The de-ramp is an internal
transform; the persisted spectrum is untouched. (Equivalently the ramp could be
carried in the model; de-ramping the data reuses Stage 4's transform and keeps
`h_T` in its simplest form — D-2.)

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
`h_T` evaluated on the window grid versus the de-ramped complex window data,
identical point counts. Real and imaginary parts are stacked into one real
residual vector; the least-squares objective is noise-weighted by the
canonical Stage 2 per-point RMS.

The Stage 2 `rms_noise` array is already a **per-bin complex RMS** (it is used
raw as σ in the Stage 4 `S_coh` statistic). Stage 5 weights by it directly —
the bcfitting `calculate_noise_weighted_chi2` applied a 1.53 magnitude→complex
conversion factor that is **not** ported (D-8); that factor existed only
because the older noise stage reported a magnitude standard deviation.

The solver is `scipy.optimize.least_squares` with an analytic Jacobian of
`h_T` w.r.t. `(A, δ, φ)` per peak and the shared `τ` (finite-difference
fallback acceptable initially; analytic Jacobian validated against
finite-difference in unit tests). The Jacobian also yields the parameter
**covariance** → real frequency/amplitude/phase uncertainties — an improvement
over the surviving reference, which left `freq_err = 0` for want of a
covariance matrix.

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
addition and incremental subtraction; addition won. The retained caveat is an
**early-modelling / underfitting failure**: an under-fit N-peak model can
leave a residual where peak N+1 alone shows no significant improvement, yet
N+1 *and* N+2 together do — a strict "stop at first non-improvement" loop
misses them. Mitigation: a **patience** parameter (O5-5) — continue trying
additions for a small number of steps past a non-improving candidate before
terminating, and re-evaluate the run as a whole.

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
one per fit window (padded per D-6) from the persisted spectrum and the plan.

## Pipeline integration

- **Stage key / dependencies.** Add `stage5_fitting` to
  `PipelineStageTracker.STAGE_DEPENDENCIES` (depends on `stage4_windows`) and a
  `STAGE_DATA_PATHS` entry (`stage5_fitting`). It is then automatically
  invalidated by the existing canonical-settings-change mechanism and by Stage
  3 re-detection / Stage 4 re-planning via `invalidate_downstream_stages`.
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
  skirt biases a dependent window; whether the add-one-peak F-test (with/without
  patience) detects the blend, especially a weak component on a strong flank.
- **Fixed contributors.** A strong line and, in a *separate* window, a weak
  line on its skirt; the strong line is fit, frozen, and carried — the weak
  line's parameters must come back unbiased. Repeated with the strong line a
  partially-resolved blend (the failure mode).
- **τ handling.** A weak-only window falls back to fixed `τ_default`; a window
  with a strong anchor fits `τ` free and recovers it.
- **Conservative loop.** The F-test/AIC accept/reject logic; the underfitting
  case where N+1 alone is insignificant but N+1 and N+2 are (patience); the
  knockout test grows the residual as predicted.
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

- **O5-1 — model realization.** Analytic `h_T` on the window grid (primary,
  needed for non-aliased fixed-contributor skirts) vs a literal numerical FFT
  of the effective-time model (cross-check, and the only route if `winf`
  support were ever needed). Confirm the closed form against the numerical FFT
  in the prototype.
- **O5-2 — blending and skirt fidelity.** The central prototype investigation
  (see *The blending problem*): blend recovery vs separation, phase, and
  intensity ratio, and the separation at which a mis-fit blend corrupts the
  fixed-contributor model.
- **O5-3 — padding fallback.** Whether context-only padding (D-6) suffices, or
  Stage 4 window widening + aggregation is required. Decided empirically once
  fits run.
- **O5-4 — τ free-vs-fixed threshold.** The strongest-line SNR below which a
  window holds `τ` fixed at `τ_default`. Calibrated on 2638.
- **O5-5 — patience parameter.** How many non-improving steps the add-one-peak
  loop tolerates before terminating, to survive the underfitting failure
  without admitting noise peaks.
- **O5-6 — thaw protocol details.** The exact trigger thresholds for the
  residual edge-coherence check and the bound on renegotiation rounds.
- **O5-7 — parameter uncertainties.** Analytic Jacobian → covariance from the
  start, or finite-difference initially with the analytic Jacobian as a
  follow-up.
- **O5-8 — persist vs recompute.** Confirm the parameters-persisted /
  arrays-recomputed split against `SERIALIZATION_STRATEGY.md` during
  implementation.

## Task breakdown

1. [ ] **Research prototype** — derive and unit-test the demodulation /
   sideband mapping; verify `h_T` (analytic vs numerical FFT) and its Jacobian;
   investigate **blending** (recovery vs separation/phase, fixed-contributor
   skirt corruption, add-one-peak detection) and the addition-vs-patience
   behaviour. Archive under `dev-docs/research/stage5-fitting/` with a report,
   following the Stage 3/4 prototype pattern. Resolves O5-1, O5-2, informs
   O5-5.
2. [ ] `fitting/peak_model.py` — `h_T`, Jacobian, demod/sideband mapping,
   de-ramp integration + unit tests (both sidebands).
3. [ ] `fitting/window_fit.py` — per-window least-squares core (the recreated
   `fit_time_domain_peaks` contract), shared/fixed τ, parameter covariance +
   unit tests.
4. [ ] Conservative add-one-peak loop — F-test + AIC + separation + patience,
   the audit trail, the knockout test; `fitting/validation.py` helpers ported
   from the bcfitting shell + unit tests.
5. [ ] Fixed-contributor evaluation + DAG/batch execution order + local `thaw`
   renegotiation + unit tests.
6. [ ] Stage 4 `replan` entry point (`merge`/`split`) + the residual
   edge-coherence renegotiation handshake + unit tests.
7. [ ] Data-structure wiring — `FittedPeak`/`FittingResult`/`SpectralWindow` +
   the new `SpectrumFit` aggregate + unit tests.
8. [ ] `io/fitting_serialization.py` + `stage5_fitting` stage tracking and
   dependencies + invalidation wiring + hand-edit round-trip tests.
9. [ ] Wrappers (`Pipeline.fit_windows/visualize_fit/load_fit`, `api.*`, CLI
   `fit-windows`/`visualize-fit`) + `visualization/fit_visualization.py`.
10. [ ] Cross-interface + 2638 representative-subset integration tests; the
    doublet and 34154 cases; then the full-plan integration check.
