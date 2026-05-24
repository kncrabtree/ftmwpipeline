# Stage 5 — Residual rescue and phase-coherence screening

Status: **implementation overview** for the production code that landed
in commits `3d403b7` (library functions) and the orchestration / B-loop /
sliding-coherence work that followed. The rescue pass is now wired into
the per-window orchestrator and the `fit_peaks_impl` public surface; the
gating knob is `max_residual_rescue_rounds` (integer, default `0` =
disabled, intended to become non-zero once validated at scale — the
rescue is a structural part of the fit, not an opt-in tweak). The
15-window 2638 validation set lives at `scratch/stage5-validation/`;
the harness emits, per window, `detail.png` (consolidated final fit),
`audit-trail.png` (rescue audit trail), `detail-rr<n>.png` (one per
consolidated round, trajectory snapshots), and the `report.md` /
`report-rr.md` text rollups.

**Immediate next-session sequencing** for the open work is captured in
[Open question 2 → "Suggested sequencing for the next session"](#suggested-sequencing-for-the-next-session):
contributor-skirt leakage ([`stage5-fitting.md`](stage5-fitting.md) O5-10)
→ effective-DoF / AICc generalization → phase-degeneracy penalty.
"Next steps" further down is the longer-horizon rescue-completion plan
(broader validation, calibration sweeps, default flip); both tracks
coexist.

Normative requirements remain in the `*_STRATEGY.md` specs; the parent
plan is [`stage5-fitting.md`](stage5-fitting.md). This document is
normative only for the residual-rescue subsystem.

## What residual rescue is

The Stage 5 conservative loop only ever seeds peaks Stage 3 detected.
Any real line Stage 3 missed (or that knockout dropped during the
conservative fit) leaves an above-noise residual that the initial fit
cannot explain. The rescue is a second pass — and now a *chain* of
passes — that detects peaks in that residual and folds them into the
fit.

The single-round rescue is strictly separated from the initial fit:

1. Compute `residual = data − model(initial.peaks, initial.tau)`. Done
   once; the initial fit is never touched again.
2. Detect candidate peaks in the residual
   (`fitting.residual_screening.find_residual_peaks`).
3. Drop candidates that fail the sliding phase-coherence check
   (`fitting.residual_screening.filter_by_phase_coherence`).
4. Run a second `conservative_fit` on the **residual itself**, with tau
   frozen at the rescue tau (see below). The result's peaks are exactly
   the lines the rescue added; the initial fit's peaks are not present
   in this returned fit.

The rescue chain (`fitting.residual_rescue.rescue_and_consolidate`,
"option B") iterates this:

1. Initial fit → rescue₁ (as above).
2. Joint refit of `initial.peaks + rescue₁.peaks` against the original
   data, with all parameters thawed and the tau initial value pulled
   from the rescue (apodization-override-aware — the structural fix for
   w198-like cases where the initial fit's tau is pegged at the lower
   bound).
3. Knockout sweep over the joint fit; any peak flagged unsupported is
   dropped and the surviving subset is refit.
4. The consolidated fit becomes the "current" for round 2; repeat
   against the new residual. Terminates when the rescue accepts no new
   peaks, the joint refit fails to converge, knockout would empty the
   model, or `max_rescue_rounds` is reached (default 3).

Each round records a `RescueRoundDiagnostics`; the
`n_pruned_rescue_origin` counter is the **failsafe diagnostic** — a
nonzero value means the knockout sweep dropped a peak the rescue had
just added, which is the signal that the joint refit may not be
escaping a pathological basin. v1 logs only; an A-mode fallback for
windows that consistently trip this signal would key off it.

## Public surface

`fitting/residual_screening.py`
- `find_residual_peaks(...)` — `scipy.signal.find_peaks` with a
  sigma-relative height + prominence cut on `|residual|`. Threshold knobs:
  `snr_threshold` (default 2.5σ_c), `prominence_threshold` (2.0σ_c).
- `filter_by_phase_coherence(candidates, ..., fitted_peak_offsets=...)`
  — projects each candidate's residual onto a unit-amplitude Lorentzian
  basis and rejects candidates whose coherent SNR ratio falls below a
  **sliding threshold** keyed on the candidate's distance to the nearest
  *other candidate or already-fitted peak*. Three bands (see "Phase-
  coherence projection" below): cluster floor (defer), linear ramp from
  `close_threshold` (default 0.2) to `isolated_threshold` (default 0.8)
  across the cluster→isolated FWHM band, full isolated threshold above.

`fitting/residual_rescue.py`
- `attempt_residual_rescue(...)` → `RescueOutcome` — one-round rescue;
  the rescue's `conservative_fit` runs with `fit_tau=False` so the
  rescue's peak set shares one frozen tau (the apodization-override-aware
  rescue tau).
- `rescue_and_consolidate(...)` → `ConsolidatedRescueOutcome` — the
  B-loop chain (rescue → joint refit → knockout → repeat). The
  consolidated `ConservativeFitResult` is what the orchestrator slots
  into `WindowOutcome.fit`; the `RescueRoundDiagnostics` list carries
  per-round bookkeeping including the failsafe pruning counts.
- The joint refit's constraints are derived once via
  `derive_window_fit_constraints` so the per-round LSQ sees the same
  tau / amplitude / penalty bounds as the initial fit; only the starting
  tau is updated (from the rescue) to feed the joint refit a sensible
  warm start.

`fitting/plan_execution.py`
- `_apply_rescue_to_outcome` runs the B-loop on each window's
  *post-thaw* outcome (so any contributor that thaw promoted to a free
  peak is already part of the model when the rescue measures the
  residual). Emits `RescueEvent` records aggregated into
  `PlanFitOutcome.rescue_history`.
- `execute_plan(..., max_residual_rescue_rounds=N, rescue_kwargs={...})`
  threads the gating knob and the tuning bag through to
  `_walk_windows_in_order`.

`fitting/window_fit.py` (earlier refactor)
- `derive_window_fit_constraints(...)` → `WindowFitConstraints` extracts
  the tau / amp / penalty derivation that used to live inline in
  `conservative_fit`. Both `conservative_fit` and the rescue's joint
  refit use it so they enforce identical constraints.
- `DEFAULT_MAX_NFEV` bumped 400 → 2000. The original 400 was tripping
  the `not fit.success` early-exit in `_blend_aware_seed` / the main
  loop on partial-capture LSQ converges (chi² stops moving but scipy
  flags `success=False` for hitting the cap). 2000 covers every case in
  the 2638 fixture; the cap is a runaway-safety, not a convergence
  criterion.

## Rescue tau policy

Real molecular lines in one experiment share a tau ≈ the applied
apodization (the canonical Stage 1 `expf_us`). The rescue must use the
right line-shape width or the phase-coherence basis under-projects real
peaks and the filter rejects them.

The rule:
- Default to `initial.tau_us` — it's the LSQ-converged value for the
  real lines this window contains.
- **Override** to `tau_apodization_us` when `initial.tau_us` is within
  5% of the lower bound (`tau_apodization_us / max_decay_factor`). That's
  the signature of a broken initial fit: LSQ over-narrowed tau to absorb
  unmodelled-peak residual. Using that broken tau as the coherence basis
  under-projects real peaks. The apodization is the right physical
  default.

In the 2638 validation set this fires on w198 (initial tau pegged at
1µs; apodization is 5µs).

## Phase-coherence projection

For an isolated candidate at offset `f_0`, build the unit-amplitude
Lorentzian basis `basis(f) = h_T(f − f_0, tau, T)` and compute the
sigma-weighted complex projection:

```
A_complex = Σ_f w_f · conj(basis(f)) · residual(f) / Σ_f w_f · |basis(f)|²
```

with `w_f = 1 / σ_c(f)²`. This is the closed-form solution for an
amp+phase-only fit with offset and tau frozen — exactly what
`fit_window` would converge to if only those two parameters were free.

The "coherent SNR" at the peak is `|A_complex| · |basis(f_0)| / σ_c`.
Compare it to the detected magnitude SNR (`|residual(f_0)| / σ_c`):

- **Real Lorentzian peak in clean isolation**: ratio ≈ 1 (the bin
  magnitude is dominated by a coherent line the basis captures).
- **Real Lorentzian peak sitting in a neighbour's skirt**: ratio
  partially suppressed by leakage from the neighbour's (slightly
  mis-fit) Lorentzian tail. The contamination scales with the
  neighbour's Lorentzian magnitude at the candidate's offset — ~50%
  at 1 FWHM separation, ~6% at 5 FWHM.
- **Phase-rotation artifact**: ratio ≪ 1 (the complex projection
  cancels across the rapidly-rotating phase).

### Sliding-threshold scheme

A uniform ratio threshold (the previous design, default 0.5) does the
wrong thing on both ends: too strict on candidates near fitted
neighbours (legitimate peaks get rejected because their projection is
inevitably contaminated), and arguably too loose on far-isolated
candidates (a 50σ candidate with 25σ coherent projection passes
despite half its magnitude being incoherent). The current scheme
ramps the threshold by neighbour proximity in three bands:

- **Δ < `cluster_threshold_fwhm` × FWHM** (default 1.0 FWHM): defer
  entirely. A sub-cluster candidate is either a real blend the
  blend-aware seeder should handle, or a phase artifact the basis
  cannot disambiguate from a blend.
- **`cluster_threshold_fwhm` ≤ Δ < `isolated_threshold_fwhm` × FWHM**
  (default 1.0–5.0 FWHM): linear ramp from `close_threshold`
  (default 0.2) at the cluster boundary up to `isolated_threshold`
  (default 0.8) at the isolated boundary. The lenient close end
  expects projection contamination from the neighbour; the strict far
  end has no such excuse.
- **Δ ≥ `isolated_threshold_fwhm` × FWHM**: full
  `isolated_threshold`.

A more principled functional form (interpolating by the Lorentzian
skirt magnitude `|basis(Δ)|²` directly, since that tracks the
contamination level exactly) is an obvious follow-up; the linear ramp
is easier to tune and reason about for v1.

### Neighbour distance includes fitted peaks

The proximity check uses `min(distance to nearest other candidate,
distance to nearest peak in current_fit)`. Including the fitted peaks
is the structural fix for the w198 case where a real residual peak
sat ~1.1 FWHM from a freshly-fit line — that candidate was previously
held to the 0.5 uniform threshold, projection-contaminated by the
fitted neighbour, and rejected.

## Validation results (2638 fixture, 15-window sample)

`scratch/stage5-validation/generate_validation.py` reproduces the
initial fit per window, runs the consolidated B-loop, and emits one
`detail-rr<n>.png` per round (showing the cumulative consolidated fit
at the end of round *n*) plus a single `report-rr.md` rollup with per-
round bookkeeping (rescue candidates, coherence-rejections, joint-refit
status, knockout pruning broken out by rescue origin).

### Post-B-loop + sliding-threshold (current behaviour)

| Window | Initial → final K | Initial chi²_r → final chi²_r | Notes |
|---|---|---|---|
| w63, w64 | 3→3, 2→2 | 0.81, 0.58 (unchanged) | clean controls — rescue terminates round 0 with no candidates. |
| w16 | 1→2 | (low → low) | borderline second peak rescued. |
| w104 | 1→2 | (low → 0.75) | Stage 3 candidate appears spurious; rescue replaces it. |
| w127 | 1→2 | (low → 0.55) | borderline second peak rescued. |
| w148 | 1→5 | 715 → **0.95** | missed 179σ peak captured round 0; the +0.5970 candidate (previously rejected by the 0.5 uniform threshold as "doublet leakage") is now accepted by the sliding threshold and survives knockout. **Subsequent inspection (see "Visual evidence on w148" below) shows the K=5 fit contains two sub-spacing in-phase duplicate pairs — the chi²_r=0.95 is achieved partly through overfit, not solely through correct line capture.** |
| w198 | 2→8 | 628 → **2.3** | apodization-override propagates a sensible warm-start tau into the joint refit (1 µs → 2.2 µs); the previously-rejected −0.482 candidate now passes coherence. |
| w269 | 5→10 | 17.4 → **1.6** | the +1.4558 candidate (previously rejected as "phantom") is now accepted and strongly supported by knockout. |
| w68, w132, w209, w215, w260, w271 | typically K +5–8 | typically chi²_r 4–30 → ~0.7–2.5 | most multi-round cases now converge near the noise floor. |
| w337 | 1→2 | (low → 1.3) | borderline second peak rescued. |

### Caveat: real peaks or overfitting?

Two of the cases the previous (uniform 0.5) doc cited as
coherence-rejection successes — w148 +0.5970 and w269 +1.4558 — are
now accepted and survive knockout. Two possible readings:

- **They were always real**, and the strict uniform threshold was
  over-rejecting them because their projections sat in fitted
  neighbours' skirts (consistent with the design rationale for the
  sliding scheme). The chi²_r → ~1 across the multi-round cases
  supports this — the previous threshold left systematic residual
  structure that the rescue is now explaining.
- **The looser close-end threshold (0.2) plus a 32-peak per-window
  `rescue_max_peaks` cap is overfitting**, and the knockout sweep is
  not strict enough to catch all overshoots. The failsafe diagnostic
  (`n_pruned_rescue_origin`) fires on w148 round 1, w269 round 1, and
  a few others — knockout *is* catching some — but doesn't tell us
  whether what survived was real.

Until a wider validation rules this out, treat the dramatic per-window
K increases as **preliminary**, not confirmed. Recommended diagnostics
before bumping `max_residual_rescue_rounds` to a non-zero default:

- Run the harness on the full 2638 fixture (~400 windows) and tabulate
  how often `n_pruned_rescue_origin > 0` (the knockout safety net
  firing).
- Compare the consolidated peak list against the experiment's
  blackchirp-era assignments, if any survive.
- If accessible, run on another FTMW dataset where the expected peak
  list is independently known.
- Sensitivity-sweep the close/isolated anchor pair (0.2/0.8 → 0.3/0.8,
  0.2/0.7, etc.) and look for the regime where w64/w63 acquire spurious
  peaks. That's the empirical floor for what counts as "real" in this
  signal.

The clean controls (w63, w64) staying at 0 rounds is at least a weak
guard against runaway acceptance — but a single clean-window pair is
not statistical evidence of overfitting safety.

**Update (post-visualization pass):** the overfitting reading is now
the load-bearing one. See "Visual evidence on w148: sub-spacing
duplicate-peak overfit" under open question 2 below — the doublet
that should be 2 lines is being fit as 4 sub-spacing in-phase peaks,
and the per-peak knockout test (now persisted as
`KnockoutInfo.p_value`) shows <1e-15 for every duplicate because
each pair member is individually supported even though the pair is
physically redundant. Knockout cannot ask the merge question; merge
cleanup is supposed to but isn't catching this case. Candidate
algorithmic fixes — generalised effective-DoF / AICc with `n_eff`,
and a phase-degeneracy penalty complementing the existing
cancellation penalty — are sketched in that subsection.

## Open questions

### 1. Phase-coherence projection as a general primitive (Stage 3?)

The projection-coherence test is general — it answers "is this magnitude
peak the projection of a Lorentzian, or just incoherent magnitude?".
Stage 3 currently detects peaks on the magnitude spectrum and would
benefit from the same test:

- Distinguish real lines from baseline / contributor systematics
  (w337's case — see [`stage5-fitting.md`](stage5-fitting.md) O5-10
  for the underlying contributor-skirt leakage; phase-coherence here
  would *flag* the signature, the closing fix is upstream).
- Filter Stage 3 candidates whose phase doesn't support a Lorentzian
  interpretation before promotion.
- Provide a per-peak coherence score in the persisted detection list as
  a quality marker.

Worth assessing — not immediately, but as a Stage 3 enhancement. Possible
caveat: Stage 3 operates on the *persisted* spectrum (full record with
the de-ramp phase), so the basis Lorentzian needs the same phase frame.

### 2. Are the newly-accepted peaks real, or is the rescue overfitting?

The sliding coherence threshold pushed several windows from chi²_r in
the 4–50 range down to ~1. Two of the previously-rejected candidates
the original planning doc cited as coherence-rejection successes
(w148 +0.5970, w269 +1.4558) are now accepted and survive the
knockout sweep. Two readings remain on the table; the harness's
15-window sample does not distinguish them:

- **The previous uniform-0.5 threshold was over-rejecting real peaks
  that sat in fitted neighbours' skirts.** The chi²_r → ~1 behaviour
  is consistent with finally fitting them. The fitted-peak-aware
  proximity check + sliding threshold are doing what they were
  designed for.
- **The looser close anchor (0.2) plus the 32-peak per-window
  `rescue_max_peaks` cap is overfitting**, and the knockout sweep is
  not strict enough to catch all overshoots. The failsafe diagnostic
  (`n_pruned_rescue_origin`) does fire on several windows (w148 r1,
  w269 r1, w132 every round), so the safety net is engaged — but it
  doesn't tell us whether what *survived* the knockout is real.

The clearest concrete evidence for the overfitting concern is **w132**:
the loop terminates on `max rounds reached` with chi²_r = 2.5
(not the ~1 noise floor that most others hit), and the failsafe fires
in every round (1, 1, 2 rescue-origin pruned). Three possible reads:

1. **Out of rounds**: more rounds would let the chain settle, the
   pruning is healthy because borderline candidates need joint-refit
   evaluation, and chi²_r would converge toward 1 given another 2–5
   rounds.
2. **Overshoots survive**: rescue is nominating ~3σ candidates, the
   knockout catches the worst, but the survivors are noise that
   locally improved chi² without being physical lines.
3. **Genuinely un-modellable residual structure**: a cluster of
   unresolved hyperfine lines, Voigt broadening, instrumental
   artifact — what the previous version of this question was about.
   The sliding threshold accepts more candidates, leaving less
   coherence-rejected material for direct inspection, but the
   underlying physical signal didn't change.

**Quick experiment for w132 with `max_residual_rescue_rounds=5`**
(`scratch/stage5-validation/_w132_extended.py`): K 15→18,
chi²_r 2.49→1.67. The failsafe fired in rounds 0–2 (1, 1, 2
rescue-origin pruned) and **stopped firing in rounds 3–4** (0, 0
pruned, both pure additive). The chain genuinely settled by round 3;
the prior 3-round cap was too low for this window. Strong support for
reading (1) — and a signal that the `DEFAULT_RESCUE_MAX_ROUNDS=3`
default may need to bump higher (5? 7?) before the rescue can be
turned on by default. Remaining 1.67 − 1.0 gap is open ground for (2)
vs (3); needs separate investigation.

Diagnostics to settle this in a fresh session:

- Run the harness on the full 2638 fixture (~400 windows) and
  tabulate failsafe-firing rate and final chi²_r distribution.
- Sensitivity-sweep the close anchor (0.2 → 0.3 → 0.4) and look for
  the regime where w64/w63 (clean controls) acquire spurious peaks —
  that's the empirical floor for what counts as "real" in this signal.
- For the persistent-residual-at-max-rounds cases (w132), plot the
  surviving residual's local complex spectrum and check whether it
  has Voigt-style wings, hyperfine substructure, or just noise.
- Where blackchirp-era line assignments survive, compare the
  consolidated peak list against them as ground truth.

#### Visual evidence on w148: sub-spacing duplicate-peak overfit

While building the per-window detail / audit visualisations (the new
`detail.png` and `audit-trail.png` harness artifacts), inspecting w148
made the overfitting concrete: the strong doublet that should be 2
real lines is being fit as **4 peaks, arranged as two pairs of two**,
with each pair's members separated by *less than the FT point spacing*
and converging to roughly equal amplitudes. The two pair members are
not independently resolvable — they are the rescue / joint-refit
chain manufacturing duplicate peaks at the same physical line.

| consolidated peak | freq (MHz)       | amplitude (µV)  |
|---|---|---|
| A | 31848.5948(25)   | 4.38(22)        |
| C | 31848.5551(22)   | 4.641(187)      |
| B | 31849.7032(24)   | 4.113(135)      |
| D | 31849.65473(185) | 4.743(143)      |

A/C separation: 0.040 MHz. B/D separation: 0.049 MHz. Point spacing on
this fixture is ~0.05 MHz. With tau ≈ 4 µs, FWHM ≈ 0.08 MHz — both
pairs are within 1 FWHM and at the resolution limit.

The merge-clean-up step (`merge_close_peaks_cleanup` in
`fitting/residual_rescue.py`, F-test-gated at
`DEFAULT_MERGE_SEPARATION_FACTOR = 1.0 FWHM`) is *supposed* to catch
this — it greedily merges close adjacent pairs when the merged
(K-1)-peak fit is statistically indistinguishable from the K-peak fit.
Either (a) the merge cleanup is not running in this code path
(`rescue_and_consolidate` may not be calling it on the final
consolidated fit), or (b) it is running but the F-test is rejecting
the merge — the duplicate pair locally improves chi-squared enough
that the F-test sees them as "really distinct" even though physically
they cannot be.

**Defer to a later session** — visualization pass needs to land first
so the diagnostic is visible. When picking this up:

1. Confirm whether `merge_close_peaks_cleanup` is called in the
   consolidated path (grep for call sites). If not, that's the wiring
   gap.
2. If it is being called, instrument it to log which merges were
   considered and which the F-test rejected. The expectation is that
   w148's A/C and B/D pairs are being considered and rejected.
3. The merge F-test compares chi-squared of the K-peak fit to a
   refitted (K-1)-peak fit. If duplicate peaks at sub-spacing
   separations split the line's signal between them, the K-peak fit's
   chi-squared can be marginally lower in a way that's statistically
   "significant" by F-test but physically meaningless — both peaks are
   fitting the same noise realization of the same physical line. The
   fix may be a structural separation cutoff (merge unconditionally
   when separation < point spacing or < 0.5 FWHM), not just an
   F-test-gated merge.

This is the clearest concrete instance of the overfitting concern the
parent open question is about. It's a Stage-5-algorithm fix, not a
visualization fix; the visualization just made it visible.

#### Candidate algorithmic fixes (item 1 LANDED; item 2 still deferred)

Two ideas the user surfaced while looking at w148 — both targeting
the same underlying problem from different angles. Item 1 (effective-
DoF / AICc) shipped in the Phase 1 series; the actual implementation
differs from the original proposal in a few important ways (see the
"Phase 1 implementation status" subsection below). Item 2 (phase-
degeneracy penalty) is still deferred.

**1. Generalised effective-DoF across all Stage 5 hypothesis tests.**
The Stage 5 fit currently runs three F-test-style gates, all sharing
the same statistic shape:

```
F = (Δχ² / Δdof) / (χ²_K / (n_data − n_params_K))
```

The three sites:

- **Conservative add-one-peak loop** (`window_fit.py`) — `accept`,
  `promote`, and `tentative` decisions gate on
  `p_value < significance AND trial.aic < current.aic`. K-vs-(K+1)
  comparison.
- **Knockout test** (`knockout_test` in `window_fit.py`) — per-peak
  K-vs-(K-1) comparison; sets the `supported` flag and the persisted
  `KnockoutInfo.p_value` (the column added in this work).
- **Merge cleanup** (`merge_close_peaks_cleanup` in
  `residual_rescue.py`) — pair-merge K-vs-(K-1) comparison; greedy
  F-test-gated cleanup of close adjacent pairs.

All three use the full window `n_data` (~100–300 bins) in the
denominator. But the *informative* bins for distinguishing a K-peak
model from a (K±1)-peak model live within ~1 FWHM of the peak in
question — a handful of bins. The 200-bin denominator is mostly noise
far from the feature, which inflates the F-statistic for marginal
improvements and structurally biases every gate toward
**accepting the more-complex model**:

- Conservative loop: weak peaks pass the accept gate because adding
  three parameters buys ~6 in δχ² (only ~2.4σ of evidence) — but
  against a 200-bin denominator the p-value clears significance.
  The AIC sibling-gate's `2k` penalty is not strong enough to backstop
  this. Symptom: artificially low p-values for weak peaks, suspected
  on inspection of the 2638 fixture's weak-peak windows.
- Knockout: every duplicate-pair peak in w148 shows `p_KO < 1e-15`
  because removing a duplicate leaves a half-fit line, which is a
  large δχ² against a huge denominator. The peak looks individually
  supported even though physically it's redundant.
- Merge: w148's A/C and B/D duplicate pairs aren't merging — same
  cause, evaluated head-on (most explicit symptom because the
  question itself is local).

Proposal: replace the raw `n_data` with an **effective sample size**
`n_eff` that weights each bin by local model magnitude (Fisher-
information-density flavour). Apply uniformly at all three test
sites. Weighting options:

- `w_f = |model(f)|²` plus the Kish formula `ν_eff = (Σ w_f)² / Σ w_f²`.
  Soft, smoothly down-weights bins far from any feature.
- `w_f = 1` only for `|model(f)| > c·max|model|` (hard radius,
  threshold-sensitive but simpler).
- Restrict the F-test to a "local window" of ±N·FWHM around the
  candidate (hard form of the same idea — chi-squared restricted to
  informative bins; loses the global noise floor estimate).

**Canonical form: AICc with `n_eff`.** AICc (small-sample-corrected
AIC) penalty grows as `2k(k+1)/(n - k - 1)`. If we feed it `n_eff`
instead of raw `n_data`, AICc handles the over-acceptance problem at
all three sites from one rule: the conservative-loop accept gate,
the knockout-test "supported" decision, and the merge gate all key
off the same effective-sample-size rule. This is the cleaner long-
term form than per-site F-test patching — F-test stays as a
diagnostic statistic; AICc with `n_eff` becomes the decision rule.

User's framing: "fitting a narrow feature with 2 independent peaks
should bear a high burden of statistical proof"; "we have
artificially low p-values for weak peaks; likely just because there
are so many points in a window." The effective-DoF formulation
operationalises both: the burden grows because the denominator
shrinks to the informative bins only.

Caveats to think through:

- The F-distribution assumes Gaussian residuals with the unit-
  variance noise model. Weighted residuals change the distribution;
  a strict derivation would need the right reference distribution
  (probably still F under reasonable assumptions, but worth
  checking). AICc dodges this — it's a likelihood criterion, not a
  distributional one.
- Weak isolated peaks that *should* be in the model (w16, w104, w127,
  w337-borderlines) must not drop out under the new rule. The same
  validation set that catches duplicate-pair overfit needs to also
  catch under-rejection of real-but-weak peaks. Sensitivity sweep
  the weighting choice (|model| vs |model|² vs hard window) against
  both regimes.
- The persisted `KnockoutInfo.p_value` semantics change if we move
  to AICc-with-`n_eff`. Either keep the raw F-test p as today and
  add an `aicc_delta` field, or repurpose the p_value field to the
  effective-DoF F-test result. Decide before persisting any new
  values to avoid mixed-semantics across fixture vintages.

**2. Phase-degeneracy penalty (sketch).** The existing pair penalty
(`window_fit.py:572-607`) is `sqrt(λ) · w(Δsep) · sin((φᵢ - φⱼ)/2)` —
zero for in-phase pairs (Δφ=0) and maximal for anti-phase pairs (Δφ=π).
It catches the cancellation pathology (a pair that fits noise by
producing destructive interference between two large amplitudes) but
explicitly does **not** penalise the *degeneracy* pathology (two
in-phase peaks at the same offset with similar amplitude — the case in
w148's A/C and B/D pairs).

User's framing: "Our best bet for fitting blended features would
likely occur when their phases are in quadrature." Quadrature (Δφ = π/2)
is the only configuration where two close peaks carry independent
information; both Δφ=0 (degenerate / co-aligned) and Δφ=π (cancelling)
are pathological.

The complementary penalty for the degeneracy pathology is
`cos((φᵢ - φⱼ)/2)` — 1 at Δφ=0 (max penalty) and 0 at Δφ=π (no
penalty). It mirrors the existing one. Two ways to wire this:

- **Separate penalty term** — add a `phase_degeneracy_penalty_lambda`
  parameter and emit a second penalty residual per pair with
  `sqrt(λ_deg) · w(Δsep) · cos((φᵢ - φⱼ)/2)`. Independent tuning;
  keeps the current cancellation penalty untouched.
- **Single non-quadrature penalty** — replace both halves with one
  term that fires at *both* Δφ=0 and Δφ=π, zero only at Δφ=π/2:
  candidates are `cos(φᵢ - φⱼ)` (peaks at both 0 and π) or
  `|cos(φᵢ - φⱼ)|`. One knob; cleaner conceptually but loses the
  ability to tune cancellation-vs-degeneracy independently if their
  failure modes turn out to need different λ.

Both should keep the existing `weight = max(0, 1 - sep/cutoff)`
closeness factor so the penalty only fires for pairs near the
resolution limit. The cutoff might need to be smaller for the
degeneracy half (e.g. 1 FWHM instead of the current 2 FWHM) since
the degeneracy pathology is specifically a sub-FWHM problem.

#### Phase 1 implementation status (item 1 landed)

Item 1 shipped against the **merge gate only** so far. The other two
sites (knockout `supported` flag and conservative-loop accept gate)
still use the raw-`n_data` F-test/AIC; they remain on the roadmap as
Phase 2 and Phase 3 respectively. The merge-gate implementation also
diverged from the original proposal in two structurally important
ways:

**(a) AICc formula uses `n_eff` uniformly, not just in the correction.**
The Burnham-Anderson AICc is `2k + n·log(chi²/n) + 2k(k+1)/(n - k - 1)`;
the original sketch only swapped `n_eff` into the correction term and
left `n_data` in the log-likelihood. That hybrid doesn't correspond to
any clean statistical derivation. The shipped form substitutes `n_eff`
for `n` everywhere — see `validation.calculate_aicc(chi2, n_params,
n_eff)`. The log-term scaling makes marginal chi² improvements count
for less when `n_eff` is small, which is the same conservatism the
small-sample correction provides, applied through a second channel.

**(b) Two-tier merge gate, not single AICc test.** Pure AICc-with-
`n_eff` has a structural blind spot: for K-peak fits on narrow features
where `n_eff < k + 1` for *both* K and (K-1), AICc returns `+inf` for
both → tied → any tie-break behaviour either over-merges (`inf > inf
== False` accepts the merge, eating real peaks like w198's outer
shoulders) or under-merges (rejects, leaving duplicate-pair overfit).
The fix is a two-tier merge gate (`merge_close_peaks_cleanup` in
`residual_rescue.py`):

- **Tier 1 — sub-resolution structural merge.** Pairs at separation <
  `structural_merge_factor * fwhm` (default 0.5 FWHM) merge
  unconditionally; no AICc test. The Lorentzian-only model physically
  cannot distinguish these from a single peak, so any K-peak LSQ
  convergence at this scale is a numerical artifact. Catches the w148
  / w269 duplicate-pair overfit pathology.
- **Tier 2 — above-resolution AICc-with-`n_eff` gate, REJECT-on-tie.**
  Pairs at `structural_merge_factor * fwhm` ≤ separation <
  `merge_separation_factor * fwhm` (0.5 to 1.0 FWHM by default) merge
  only when AICc(K-1) is **strictly less than** AICc(K). Ties (both
  finite-equal or both `+inf`) preserve the K-peak fit. Protects real
  close pairs like w198's outer shoulders (~1 FWHM from inner peaks)
  where AICc cannot discriminate.

The weighting choice for `n_eff` is `kish_mag` (Kish formula on
`|model(f)|`, not `|model(f)|²`); a sweep on the 2638 fixture showed
the `|model|²` weighting produced `n_eff` so small that even sub-
resolution duplicate pairs sometimes were unidentifiable and the gate
would either over- or under-merge unpredictably depending on tie-break
choice. `kish_mag` keeps `n_eff` large enough for Tier 2 AICc to be
informative across the 2638 fit's K range. `n_eff_kind` is a parameter
on the merge function; the production default is set in
`DEFAULT_N_EFF_KIND` and is dataset-relevant (per-instrument re-
calibration may be needed).

**(c) Cleanup wiring gap closed.** The pre-Phase-1 codebase had
`merge_close_peaks_cleanup` defined as a library function but no
caller in `rescue_and_consolidate` — confirmed via grep. The Phase 1
work added the call after each round's knockout sweep (with a
`knockout_test` re-run on the merged fit so persisted `supported`
flags match the post-merge peak set). `RescueRoundDiagnostics` gained
`n_merged` so the per-round audit shows where the gate fired.

**(d) Shape-error-aware sigma inflation for the rescue's screening
pipeline.** A separate Phase 1b change to `attempt_residual_rescue`,
necessary because the merge gate alone left the rescue stuck in limit
cycles on strong-line windows (rescue keeps re-detecting irreducible
shape residual as candidate peaks; merge collapses them; next round
re-detects them; ...). The fix builds a position-dependent effective
noise floor

```
σ_eff(f) = √(σ_c² + (ε · |current_model(f)|)²)
```

threaded as the new `shape_error_epsilon` parameter (default 0.0 =
behaviour-preserving). The rescue's detector + phase-coherence filter
see the inflated sigma; the LSQ inside `conservative_fit` keeps the
canonical sigma — inflation is a screening tool, not a fitting one. A
per-bin post-filter is required because `find_residual_peaks` uses
the *median* sigma for scipy's `find_peaks` height threshold, so a
few-bin local inflation doesn't shift the global gate; the post-
filter checks each detected candidate against
`snr_threshold * sigma_eff[bin_index] / sqrt(2)` and drops those that
fail at the inflated floor.

`ε` is a **per-dataset constant** measured from the chi²_r vs SNR²
regression on the post-rescue fits (see "Lineshape model deficit and
per-dataset calibration" in
[`stage5-cross-fixture-validation.md`](stage5-cross-fixture-validation.md)).
The 2638 fixture's calibrated value is 0.05 (5% per-bin residual at
the line center). Note that the *per-bin* `ε` is roughly 4× the
*chi²_r-aggregated* `ε` from the regression because chi²_r sums over
many bins and divides by full-window dof; the per-bin value is what
the sigma inflation needs. This factor will likely differ between
instruments.

**(e) Persistence semantics.** The original proposal asked whether to
keep raw F-test `p_value` or replace with the n_eff-corrected form.
Decision: persist all three diagnostics during development (raw F-test
p, n_eff-corrected p, and `aicc_delta`), slim to just (raw p +
`aicc_delta` + `n_eff`) once Phase 2 / 3 settle. **NOT YET WIRED** —
the merge gate is transient state (not persisted), so this only
becomes load-bearing for the Phase 2 (knockout) and Phase 3
(conservative-loop) work. No fixture-vintage hazard yet.

#### Validation results after Phase 1 (2638 fixture, kish_mag, ε=0.05)

Survey distribution (50 windows, every 7th):

| metric | baseline (pre-Phase 1) | Phase 1+1b+1c |
|---|---|---|
| chi²_r median | 1.43 | 1.06 |
| chi²_r p75 | 2.33 | 1.65 |
| chi²_r p95 | 7.73 | 3.23 |
| chi²_r max | (~65; outlier-dominated) | 5.65 |

Target windows:

| window | K_init → K_final | chi²_r init → final | note |
|---|---|---|---|
| w148 (duplicate-pair) | 1 → 2 | 715 → 6.27 | Duplicate-pair overfit eliminated (sub-resolution merge fires). chi²_r at shape-error floor for SNR=145. |
| w198 (apod-override + shoulders) | 2 → 7 | 627 → 2.92 | Real outer shoulders preserved by Tier-2 REJECT-on-tie. |
| w269 (duplicate-pair) | 5 → 4 | 17.4 → 18.65 | Duplicates from initial seeding collapsed; chi²_r at shape-error floor for SNR=474. Remaining initial-seeding duplicate likely needs Phase 3. |
| w271 (decoupled doublet) | 2 → 2 | 1578 → 9.45 | Initial-seeding A/C duplicate not yet resolved — Phase 3 issue. |
| w16, w104, w127, w337 (borderline) | 1 → 2 | (low → low) | Unchanged — weak-peak rescues preserved. |
| w63, w64 (clean controls) | unchanged | unchanged | No regressions. |

#### Suggested sequencing for the next session

Items 1 and 2 below are LANDED; the remaining sequencing focuses on
the still-deferred per-site AICc generalisation and the phase-
degeneracy penalty. The duplicate-pair overfit and contributor-skirt
leakage interaction noted in earlier versions of this section turned
out to be less coupled in practice than expected — Phase 1 work
proceeded cleanly without first re-doing the O5-10 leakage fix.

1. **Fix the contributor-skirt leakage** (`stage5-fitting.md` O5-10).
   LANDED in commit `456fec2` (drives Stage 4 contributor attachment
   from analytic skirt magnitude). Provides the clean baseline the
   AICc work calibrates against.

2. **AICc-with-`n_eff` at the merge gate** (this section, #1, merge
   site). LANDED as Phase 1+1b+1c — see "Phase 1 implementation
   status" above. Shipped form: two-tier gate (sub-resolution
   structural + above-resolution AICc-with-REJECT-on-tie), shape-
   error-aware sigma inflation for the rescue's screening pipeline,
   wiring gap closed in `rescue_and_consolidate`. The per-dataset
   `ε` calibration is the major generalisation lever — see
   [`stage5-cross-fixture-validation.md`](stage5-cross-fixture-validation.md).

3. **AICc-with-`n_eff` at the knockout test** (this section, #1,
   knockout site). NEXT. Per-peak K-vs-(K-1) comparison; flip
   `KnockoutResult.supported` from F-test `p_value < significance` to
   `aicc_delta < 0`. Schema additions (`n_eff`, `aicc_delta`) to
   `KnockoutInfo` with NaN defaults for backwards compat. Validation:
   w148/w269 duplicate-pair members go `supported=False`;
   w16/w104/w127/w337 stay supported.

4. **AICc-with-`n_eff` at the conservative-loop accept gate** (this
   section, #1, conservative-loop site). After Phase 3 lands. Replace
   the dual `p_value < significance AND trial.aic < current.aic` gate
   with the single AICc-with-`n_eff` test in both `_blend_aware_seed`
   K=2/K=3 escalation and the main loop. Largest cascading effect on
   K across all windows; expected to address the initial-seeding
   duplicate-pair pathology in w269/w271 (the post-Phase-1 remaining
   issue that's NOT a rescue problem).

5. **Add the phase-degeneracy penalty** (this section, #2). Once
   (3)+(4) are calibrated, this is the LSQ-side defence-in-depth:
   prevents the optimiser from landing in the duplicate basin in the
   first place. Cheap to prototype; calibrating its λ against the
   new gates is cleaner than against the current gates.

6. **Cross-fixture validation** (see
   [`stage5-cross-fixture-validation.md`](stage5-cross-fixture-validation.md)).
   Run alongside (3)–(5) as second / third fixtures become available.
   The per-dataset `ε` calibration is the key generalisation tool;
   if it works on a second fixture (especially from a different
   instrument), the gates and penalties don't need per-dataset tuning
   — only `ε` does.

Expected behaviour change for (3)+(4)+(5): borderline acceptance
moves toward the empirical truth. The risk is over-correction
(rejecting real-but-weak peaks); the validation set surfaces that as
a regression.

### 3. Total-model integration strategy (RESOLVED — option B)

Implemented as option B (`rescue_and_consolidate`): joint refit +
knockout between every rescue round. The structural argument was that
each round should operate against a properly-jointly-fit baseline
rather than an ever-thinner residual; the cost difference vs option A
(recursive then unroll) was small relative to the conservative-fit
calls already dominating Stage 5 wall time.

The two specific mitigations against the user's pathology concern
(joint refit failing to escape a broken-tau basin and dropping the
rescue's contribution back out) landed as:

- **Warm-start tau from the rescue, not the previous fit.** The rescue
  itself runs its conservative loop with the apodization-override-aware
  tau when it detects the previous fit's tau is pegged. That same value
  is then handed to the joint refit's LSQ as its starting tau. In w198
  this propagates 5 µs (apodization) into a joint refit that converges
  to 2.3 µs — escaping the 1 µs lower-bound peg in one step.
- **Failsafe diagnostic on knockout pruning.** Each
  `RescueRoundDiagnostics` tracks how many of the round's accepted
  rescue peaks the knockout sweep dropped (`n_pruned_rescue_origin`).
  A nonzero count means the joint refit may not be escaping a
  pathological basin and is undoing the rescue's contribution. v1 logs
  only; if validation finds a window class where this fires
  repeatedly, an A-mode fallback can be added for that window without
  reshaping the whole loop.

### 4. Sliding-coherence parameter calibration

The current anchor pair (0.2 at the cluster floor, 0.8 at the isolated
ceiling, ramping linearly from 1 to 5 FWHM) was chosen on a single
15-window sample and is not empirically calibrated. The
"validation results" caveat above lists the diagnostics needed before
treating these as production defaults. Likely follow-ups:

- Replace the linear ramp with the Lorentzian-skirt-magnitude form
  `threshold(Δ) = high − (high − low) · |basis(Δ, τ, T)|²`, which
  tracks the contamination level exactly.
- Calibrate `(close, isolated)` against the regime where clean
  windows acquire spurious peaks.
- Investigate whether the close anchor should depend on the *neighbour's*
  SNR — a 200σ neighbour leaves far more skirt energy in the
  candidate's basis than a 10σ neighbour, even at the same separation.

## Next steps

The **immediate next-session sequencing** is described under Open
question 2 → "Suggested sequencing for the next session": fix
contributor-skirt leakage first (`stage5-fitting.md` O5-10), then
generalise effective-DoF / AICc with `n_eff` across the three
hypothesis-test sites, then add the phase-degeneracy penalty.
That sequencing replaces what was, in earlier versions of this
doc, the "calibrate sliding-coherence first" framing — the duplicate-
pair overfit is now the binding constraint, not the sliding-threshold
tune.

The items below are the longer-horizon rescue-completion plan; they
remain valid but assume the immediate sequencing has landed first.

In order of dependency:

1. **Broader validation** (next-most-important): the 15-window sample's
   chi²_r → ~1 behaviour was the prompt for the immediate sequencing
   above. Once that lands, confirm against (a) the full 2638 fixture,
   (b) other FTMW datasets the user has access to, (c) blackchirp-era
   line-assignment ground truth where available. See "real peaks or
   overfitting?" above.
2. **Sliding-coherence calibration** — replace the linear ramp with the
   Lorentzian-skirt-magnitude functional form, and sweep the
   `(close, isolated)` anchors to find the regime where clean windows
   acquire spurious peaks. See "Sliding-coherence parameter
   calibration" open question.
3. **Round-cap calibration** — the w132 5-round experiment suggests
   `DEFAULT_RESCUE_MAX_ROUNDS=3` is too low for some windows; pick a
   default that's high enough for typical convergence (probably 5–7)
   without burning compute on windows that already converged at round 1.
4. **Flip the `max_residual_rescue_rounds` default** from 0 to the
   calibrated round-cap once (1)–(3) give a clean read. The rescue is
   a structural part of the fit, not an opt-in tweak — the current 0
   default is transitional.
5. **Per-window cost monitoring** — every rescue round adds one
   conservative_fit + one joint refit + a knockout sweep. For the
   production pipeline (~400 windows) the B-loop may multiply Stage 5
   wall-time by a small constant. Worth measuring on a full fixture
   once (4) lands.
6. **Independent revisit of the projection-coherence idea for Stage 3**
   — separate planning doc once Stage 5 rescue is settled.
7. **Audit-trail richness** — the consolidated `ConservativeFitResult`
   currently inherits the initial fit's `audit_trail`; the rescue
   rounds and joint refits emit `RescueRoundDiagnostics` but those
   stay live-only (off `SpectrumFit`). If the rescue chain becomes the
   default, persist the per-window `RescueEvent` list into
   `SpectrumFit` so the on-disk fit is reconstruction-complete.

### Validation-harness augmentations (delivered)

The harness (`scratch/stage5-validation/generate_validation.py`) is
the primary inspection surface for the rescue chain. Three
augmentations originally planned for a clean session have landed:

- **Per-window context view** — Figure 1 (`detail.png`) now opens
  with a full active-FT magnitude overview row, the current window's
  `freq_range` highlighted by an `axvspan` + edge vlines. Truncated
  to the persisted trim range and amplitude-scaled to the persisted
  units convention (e.g. µV).
- **vline markers at fitted-peak positions** — drawn on each
  residual and data+model panel of Figure 1. Letter labels (A, B,
  C, …) staggered cluster-aware so sub-FWHM clusters stay legible.
- **Final audit-trail figure** — `audit-trail.png` (Figure 2). Top
  half: window magnitude spectrum with consolidated model overlay.
  Bottom half: per-round audit panel laid out **bottom-to-top**
  (initial fit at the bottom, final consolidated peaks at the top
  with dotted vlines reaching up to the spectrum). Each round band
  shows a candidates row (coherence-accepted / coherence-rejected /
  rescue-fit-kept markers) and a merge row (knockout-pruned with
  red X; red border for rescue-origin pruning — the failsafe
  diagnostic). Figure height grows with chain length; small chains
  leave trailing blank space rather than stretching rows.

Per-peak provenance and the spectroscopic uncertainty formatting on
the peak listing are documented in "Loose threads / future harness
work" below — the heuristic attribution surfaces edge cases.

**Promotion to the main visualization code** is the next logical
step (`visualization/fit_visualization.py`,
`Pipeline.visualize_fit(window_id=..., rounds=True)`). Two plumbing
options:
- Persist `RescueRoundDiagnostics` to `SpectrumFit` so the viz reads
  from disk (schema change).
- Re-run the rescue on demand from the persisted state (lighter,
  ~1s per window).

Re-run-on-demand is cheaper for an initial promotion; persistence
makes sense once the chain is the default. The harness layout is now
stable enough to port; main blocker is the duplicate-pair overfit
work (Open question 2 → "Suggested sequencing") which may motivate
schema changes that should land before promotion to avoid double-
migrations.

## How to assess the validation artifacts

```
conda run -n ftmwpipeline-dev python scratch/stage5-validation/generate_validation.py
```

writes (per window, under `scratch/stage5-validation/window_NNN/`):

- `detail.png` — **consolidated final fit** (Figure 1, landscape
  letter). Full-spectrum overview + current-window axvspan; 3-column
  residual row (Re/Im/|z|) with vlines at fitted peak frequencies and
  cluster-aware letter labels; 3-column data+model row; bottom row of
  |residual| histogram vs Rayleigh + peak-listing axes with PDG-style
  spectroscopic uncertainties and the per-peak knockout p-value
  alongside the rescue-round origin tag. Amplitudes scaled by
  `10**units_power` (e.g. µV); overview truncated to the persisted
  trim range.
- `audit-trail.png` — **rescue audit trail** (Figure 2, portrait
  letter, grows taller with the chain length). Top: window magnitude
  spectrum with consolidated model overlay. Bottom: audit panel laid
  out **bottom-to-top** chronologically — initial fit at bottom, then
  each rescue round's candidates row (coherence-accepted as green
  triangles, coherence-rejected as red X, rescue-fit kept as open
  green circles) and merge row (knockout-pruned with red X, red
  border for rescue-origin failsafe firings), then the consolidated
  final peaks at top with dotted vlines reaching up to the spectrum.
- `detail-rr<n>.png` — **trajectory snapshots** (one per B-loop
  round). Uses the older 4×2 plot_spectrum_fit layout (no amplitude
  scaling, no peak listing); shows the consolidated state at the end
  of round *n* against the full window data. Useful for the
  monotonic-residual-shrink check across the chain. Not promoted to
  the new layout — its role is trajectory inspection, not final
  answer.
- `report.md` — text rollup of the per-window plan, fit statistics,
  fitted peaks, audit trail, thaw events, residual peak candidates
  surfaced by the detector. Refers to the **initial** fit (pre-rescue);
  the rescue chain is covered by `report-rr.md`.
- `report-rr.md` — per-round rollup of the rescue chain: candidate
  list, coherence-rejections, the joint refit's K and chi²_r, and
  the knockout pruning broken out by origin. Tags each round as
  ACCEPTED or REJECTED with the termination reason.

Read sequentially: `detail.png` (the answer) → `audit-trail.png`
(how we got here) → `detail-rr0.png` … `detail-rrN.png` (the
intermediate states if the audit needs forensic context). The residual
panels in the rr-trajectory should shrink monotonically toward the
noise floor; any remaining structure is what's still unexplained at
that round.

## Loose threads / future harness work

A grab-bag of items surfaced during the visualization pass; none are
blockers, but they belong here so they survive session boundaries.

### Per-peak provenance attribution is a heuristic

`_peak_provenance` in `generate_validation.py` attributes each
consolidated peak to a "source" (`init` / `r0` / `r1` / …) by closest
match in offset to that source's added-peak list. When the joint
refit reshuffles peaks across rounds, this is not exact — the w148
example shows two consolidated peaks (+0.5253 and +0.5650 offsets)
both attributing to `init` because the only initial seed was at
+0.5360, even though one of them physically descended from r0's
accept at +0.5970. Workable for first read; if a window's audit
gets a confusing attribution, this is why.

A more principled attribution would track peak identity through each
joint refit (e.g., by index permutation derived from the refit's
peak order). Worth doing only if the heuristic confuses real-world
reads.

### `detail-rr<n>.png` trajectory layout is unchanged

The per-round trajectory PNGs still go through
`fit_visualization.plot_spectrum_fit` (the older 4×2 layout, no
amplitude scaling, no peak listing). The new layout from `detail.png`
was deliberately *not* propagated to the trajectory artifacts — their
role is "here's the consolidated state at the end of round N" for
forensic comparison, not "here's the final answer." If the trajectory
PNGs end up being used heavily for inspection, lifting them to the
new layout (with a "round N" header and possibly the per-round audit
slice annotated) is a natural follow-up.

### Coherence-rejection X markers in Figure 2 are unexercised

The audit-trail figure has marker code for coherence-rejected
candidates (red X on the candidates row of each round). The 15-window
2638 sample has zero coherence rejections, so the markers have never
actually rendered. The code is in place; it just hasn't been
visually validated. A window that produces coherence rejections (or
a synthetic test fixture) would close this gap.

### Top-level `overview.png` doesn't use `DisplayStyle`

The spectrum-wide overview emitted by main() still goes through
`visualization.fit_visualization.plot_spectrum_fit` with
active-FT-native amplitudes — no trim, no units scaling. Per-window
artifacts use the new `DisplayStyle` (`detail.png`, `audit-trail.png`
respect trim + units), but the top-level overview is unchanged.
Either thread the style into `plot_spectrum_fit` (library change) or
emit a parallel harness-level overview (scratch-only). Library change
is the right long-term move, defer until the harness layout
stabilises and gets promoted.

### Dead code from the model-overlay drop

After removing the model overlay from Figure 1's full-spectrum
context row (the data+model overlap was unreadable at the figure
scale), `_full_spectrum_model` and the `other_window_fits` parameter
on `_plot_consolidated_detail` are no longer called. Left in place
in case a future iteration wants a different reduced overlay
(e.g., data−model residual at the full-spectrum scale, or a
contributor-only model to visualise leakage). If after another pass
they're still unused, remove them.

### Promotion to `Pipeline.visualize_fit(rounds=True)` (cross-ref)

The harness is the iteration surface. When the layouts and
provenance heuristics settle, the figures move into
`visualization/fit_visualization.py` with a `rounds=True` flag on
the per-window detail call (or a separate `visualize_audit_trail`
entry point). Plumbing options for the audit data (re-run-on-demand
vs persist `RescueRoundDiagnostics`) noted in §"Validation-harness
augmentations" above; re-run is the lighter starting move.
