# Stage 5 — Residual rescue and phase-coherence screening

**Implementation summary** for the residual-rescue subsystem of
the Stage 5 fitting pipeline. The algorithmic-choice provenance
lives in
[`../research/residual-rescue/report.md`](../research/residual-rescue/report.md);
this document is the operator's reference for what the system does,
how it is wired, and what knobs it exposes.

The parent plan is [`stage5-fitting.md`](stage5-fitting.md); the
cross-fixture calibration protocol is
[`stage5-cross-fixture-validation.md`](stage5-cross-fixture-validation.md);
the normative specs are the `*_STRATEGY.md` documents.

## What residual rescue is

The conservative add-one-peak loop that does the initial per-window
fit only seeds peaks Stage 3 detected. Any real line Stage 3 missed
— or that the knockout test dropped during the conservative fit —
leaves an above-noise residual the initial fit cannot explain. The
rescue is a second pass — and a chain of passes — that detects
peaks in that residual and folds them into the fit.

A single rescue round is strictly separated from the initial fit:

1. Compute `residual = data − model(initial.peaks, initial.tau)`.
   Done once; the initial fit is never touched again.
2. Detect candidate peaks in the residual
   (`fitting.residual_screening.find_residual_peaks`).
3. Drop candidates that fail the sliding phase-coherence check
   (`fitting.residual_screening.filter_by_phase_coherence`).
4. Run a second `conservative_fit` on the **residual itself**, with
   tau frozen at the rescue tau. The result's peaks are exactly
   the lines the rescue added; the initial fit's peaks are not in
   this returned fit.

The chain (`fitting.residual_rescue.rescue_and_consolidate`)
iterates this:

1. Initial fit → rescue₁.
2. **Joint refit** of `initial.peaks + rescue₁.peaks` against the
   original data, with all parameters thawed and the tau initial
   value pulled from the rescue (apodization-override-aware — see
   below).
3. Merge cleanup, knockout sweep, iterative AICc cleanup. Final
   knockout pass on the consolidated fit produces the persisted
   diagnostics.
4. The consolidated fit becomes the "current" for round 2; repeat
   against the new residual. Terminates when the rescue accepts
   no new peaks, the joint refit fails to converge, knockout
   would empty the model, or `max_rescue_rounds` is reached
   (default 3).

The gating knob on the public surface is
`max_residual_rescue_rounds` (integer, default `0` = disabled,
intended to become non-zero once cross-fixture-validated — the
rescue is a structural part of the fit, not an opt-in tweak).

## Public surface

`fitting/residual_screening.py`

- `find_residual_peaks(...)` — `scipy.signal.find_peaks` with a
  sigma-relative height + prominence cut on `|residual|`.
  Thresholds: `snr_threshold` (default 2.5σ_c),
  `prominence_threshold` (2.0σ_c).
- `filter_by_phase_coherence(candidates, ..., fitted_peak_offsets=...)`
  — projects each candidate's residual onto a unit-amplitude
  Lorentzian basis and rejects candidates whose coherent SNR
  ratio falls below a sliding threshold keyed on neighbour
  distance. Three bands; see "Phase-coherence projection" below.

`fitting/residual_rescue.py`

- `attempt_residual_rescue(...)` → `RescueOutcome` — one-round
  rescue. The rescue's `conservative_fit` runs with
  `fit_tau=False` so the rescue's peak set shares one frozen tau
  (the apodization-override-aware rescue tau).
- `rescue_and_consolidate(...)` → `ConsolidatedRescueOutcome` —
  the chain (rescue → joint refit → merge cleanup → knockout →
  iterative cleanup → repeat). The consolidated
  `ConservativeFitResult` is what the orchestrator slots into
  `WindowOutcome.fit`; the `RescueRoundDiagnostics` list carries
  per-round bookkeeping including the failsafe pruning counts.
- `merge_close_peaks_cleanup(...)` — two-tier merge. Tier 1
  (sub-resolution, separation < `structural_merge_factor · FWHM`,
  default 0.5 FWHM) merges unconditionally. Tier 2
  (`structural_merge_factor ≤ Δ < merge_separation_factor · FWHM`)
  runs the AICc-with-`n_eff` test. Tier 2 is **disabled by
  default** (`DEFAULT_MERGE_SEPARATION_FACTOR = 0.5 =
  DEFAULT_STRUCTURAL_MERGE_FACTOR`); see "Open follow-ups" for
  the dependency.
- `iterative_aicc_cleanup(...)` — iteratively drops the worst
  AICc-with-`n_eff` offender until every remaining peak is
  supported. Non-iterative drop kills duplicate clusters
  wholesale (each duplicate looks individually supported when
  its twin is frozen); the iterative form drops one at a time
  with a tau-locked (K-1) refit between drops.

`fitting/plan_execution.py`

- `_apply_rescue_to_outcome` runs the chain on each window's
  *post-thaw* outcome (so any contributor that thaw promoted to
  a free peak is part of the model when the rescue measures the
  residual). Emits `RescueEvent` records aggregated into
  `PlanFitOutcome.rescue_history`.
- `execute_plan(..., max_residual_rescue_rounds=N, rescue_kwargs={...})`
  threads the gating knob and the tuning bag through to
  `_walk_windows_in_order`.

`fitting/window_fit.py`

- `derive_window_fit_constraints(...)` → `WindowFitConstraints`
  extracts the tau / amp / penalty derivation. Both
  `conservative_fit` and the rescue's joint refit use it so they
  enforce identical constraints.
- `knockout_test(...)` and `conservative_fit(...)` accept
  `n_eff_kind` and (for `conservative_fit`)
  `knockout_n_eff_kind` parameters; see §"AICc-with-`n_eff`
  gates" below.

## Rescue tau policy

Real molecular lines in one experiment share a tau ≈ the applied
apodization (the canonical Stage 1 `expf_us`). The rescue must
use the right line-shape width or the phase-coherence basis
under-projects real peaks and the filter rejects them.

The rule:

- Default to `initial.tau_us` — it's the LSQ-converged value for
  the real lines this window contains.
- **Override** to `tau_apodization_us` when `initial.tau_us` is
  within 5% of the lower bound (`tau_apodization_us /
  max_decay_factor`). That's the signature of a broken initial
  fit: LSQ over-narrowed tau to absorb unmodelled-peak residual.
  Using that broken tau as the coherence basis under-projects
  real peaks. The apodization is the right physical default.

The rescue tau is also handed to the joint refit's LSQ as its
starting tau (warm start for the apodization-override case).

## Phase-coherence projection

For an isolated candidate at offset `f₀`, build the unit-amplitude
Lorentzian basis `basis(f) = h_T(f − f₀, τ, T)` and compute the
sigma-weighted complex projection:

```
A_complex = Σ_f w_f · conj(basis(f)) · residual(f) / Σ_f w_f · |basis(f)|²
```

with `w_f = 1 / σ_c(f)²`. This is the closed-form solution for an
amplitude+phase-only fit with offset and tau frozen.

The coherent SNR at the peak is `|A_complex| · |basis(f₀)| / σ_c`.
Compared to the detected magnitude SNR
`|residual(f₀)| / σ_c`:

- Real Lorentzian peak in clean isolation: ratio ≈ 1.
- Real Lorentzian peak in a neighbour's skirt: ratio partially
  suppressed by leakage from the neighbour's Lorentzian tail
  (~50% at 1 FWHM separation, ~6% at 5 FWHM).
- Phase-rotation artifact: ratio ≪ 1.

### Sliding-threshold scheme

The shipped scheme ramps the threshold by neighbour proximity in
three bands:

- **Δ < `cluster_threshold_fwhm` × FWHM** (default 1.0 FWHM):
  defer entirely. A sub-cluster candidate is either a real blend
  the blend-aware seeder should handle, or a phase artifact the
  basis cannot disambiguate from a blend.
- **`cluster_threshold_fwhm` ≤ Δ < `isolated_threshold_fwhm` × FWHM**
  (default 1.0–5.0 FWHM): linear ramp from `close_threshold`
  (default 0.2) at the cluster boundary up to `isolated_threshold`
  (default 0.8) at the isolated boundary.
- **Δ ≥ `isolated_threshold_fwhm` × FWHM**: full
  `isolated_threshold`.

The proximity check uses `min(distance to nearest other
candidate, distance to nearest peak in current_fit)`. Including
fitted peaks is the structural fix for real residual peaks
sitting near freshly-fit lines.

### Shape-error sigma inflation

The rescue's screening pipeline sees an inflated sigma:

```
σ_eff(f) = √( σ_c² + (ε · |current_model(f)|)² )
```

threaded as the `shape_error_epsilon` parameter (default 0.0 =
behaviour-preserving). The rescue's detector + phase-coherence
filter see the inflated sigma; the LSQ inside `conservative_fit`
keeps the canonical sigma — inflation is a screening tool, not a
fitting one. A per-bin post-filter is required because
`find_residual_peaks` uses the median sigma for scipy's
`find_peaks` height threshold.

ε is a **per-dataset constant**. The 2638 fixture's calibrated
value is 0.05 (5% per-bin residual at the line center). The
per-fixture calibration protocol is in
[`stage5-cross-fixture-validation.md`](stage5-cross-fixture-validation.md).

## AICc-with-`n_eff` gates

The three Stage 5 hypothesis tests (merge, knockout, conservative-
loop accept) all gate on the AICc criterion evaluated at an
effective sample size `n_eff` that weights each bin by some
function of the local model magnitude. Substituting `n_eff` into
the small-sample correction term `2k(k+1) / (n_eff − k − 1)`
makes the burden of statistical proof scale with the informative-
bin count rather than the full window. REJECT-on-tie at every
site: the simpler model is preserved when AICc cannot
discriminate.

Two `n_eff` weighting kinds live in `validation.py`:

- **`kish_mag_sq`** / **`kish_mag`**: Kish formula on `|model|²`
  or `|model|`. Concentrated near the peak centre; collapses to
  roughly the per-peak FWHM-in-bins on Lorentzian peaks. The
  `kish_mag_sq` form is `DEFAULT_N_EFF_KIND`.
- **`perplexity_log1p_snr`**: perplexity (`exp(H(p))`) of the
  normalised distribution `p_f ∝ log(1 + |model|/σ)`. The
  `log(1 + SNR)` weight is approximately the per-bin Shannon
  information of a signal-vs-noise detection. On the 2638
  fixture this returns ~30–80 bins on multi-peak windows. This
  is `DEFAULT_CONSERVATIVE_N_EFF_KIND`.

Per-call-site defaults:

| site | direction | `n_eff_kind` | rationale |
|---|---|---|---|
| Conservative-loop accept gate (main loop + `_blend_aware_seed`) | K-vs-(K+1) | `perplexity_log1p_snr` | The K+1 model is the magnitude basis. Magnitude-concentrated weights collapse `n_eff` below the AICc identifiability threshold for K+1 → +∞ on the more-complex side → REJECT real escalations. Information-weighted `n_eff` keeps the gate in the AICc-identifiable regime. |
| `knockout_test` (called inside `conservative_fit`) | K-vs-(K-1) | `kish_mag_sq` (via `knockout_n_eff_kind`) | The K-1 model is the simpler side. AICc divergence at small `n_eff` falls through to "preserve K", which is the desired conservative direction. |
| `knockout_test` / `merge_close_peaks_cleanup` / `iterative_aicc_cleanup` (called from `rescue_and_consolidate`) | K-vs-(K-1) | `n_eff_kind` from the rescue kwargs | The validation harness pins `perplexity_log1p_snr` here; production sites pass it through `rescue_kwargs`. |

The `effective_sample_size(..., kind=..., sigma=...)` API exposes
all three kinds; `sigma` is required for the SNR-weighted kind
and ignored by the magnitude-only kinds.

## Tau locking in (K-1) refits

The merge gate's (K-1) refit, the knockout test's (K-1) refit, and
the iterative AICc cleanup's (K-1) refit all run with
`fit_tau=False, tau0_us=current.tau_us`. Tau is effectively a
dataset-shared parameter (transit time × natural lifetime — a
property of the experiment, not the individual peak); a single-
window (K-1) refit must not get the extra knob of broadening tau
to absorb the dropped peak's contribution.

The joint refit inside `rescue_and_consolidate` is the one
exception — it fits tau freely from the rescue's warm start, so
it can escape the broken-initial-fit basin (see "Rescue tau
policy"). Locking the joint refit's tau is on the cross-fixture
follow-up track as the dataset-wide tau majority-vote-freeze
proposal.

## Validation harness

`scripts/development/stage5-validation/generate_validation.py`
reproduces the initial fit per window, runs the consolidated
chain, and emits per-window artifacts under
`scratch/stage5-validation/window_NNN/`:

- `detail.png` — consolidated final fit. Full-spectrum overview
  + current-window axvspan; 3-column residual row (Re/Im/|z|)
  with vlines at fitted peak frequencies and cluster-aware
  letter labels; 3-column data+model row; |residual| histogram
  vs Rayleigh + peak listing with PDG-style spectroscopic
  uncertainties and the per-peak knockout p-value.
- `audit-trail.png` — rescue audit trail. Top: window magnitude
  spectrum with consolidated model overlay. Bottom: per-round
  audit panel laid out bottom-to-top chronologically with
  candidates row (coherence-accepted as green triangles,
  coherence-rejected as red X, rescue-fit kept as open green
  circles) and merge row (knockout-pruned with red X, red border
  for rescue-origin failsafe firings).
- `detail-rr<n>.png` — trajectory snapshots (one per chain
  round). Useful for the monotonic-residual-shrink check.
- `report.md` — text rollup of the per-window plan, initial fit
  statistics, fitted peaks, audit trail, thaw events.
- `report-rr.md` — per-round rollup of the rescue chain:
  candidate list, coherence-rejections, joint-refit K and
  chi²_r, knockout pruning broken out by origin.

Read sequentially: `detail.png` (the answer) → `audit-trail.png`
(how we got here) → `detail-rr0.png` … `detail-rrN.png` (the
intermediate states if the audit needs forensic context).

## Known limitations

- **w198 on 2638**: the rescue chain converges to K=3 at
  χ²_r=148 instead of the K=7 at χ²_r=2.92 a tau-locked
  configuration reaches. Root cause is the joint refit's tau
  thaw: even when the initial fit holds tau fixed, the joint
  refit fits tau freely and converges to a different (~2.4 µs)
  basin from the dataset consensus (~3 µs). Resolution is the
  dataset-wide tau majority-vote-freeze proposal in
  [`stage5-cross-fixture-validation.md`](stage5-cross-fixture-validation.md).
- **Borderline real-vs-noise on w16/w104/w127/w337-class
  windows**: the rescue's iterative cleanup rejects these
  borderline second peaks each round. Whether they are real
  weak lines or noise cannot be determined from the 2638
  fixture alone; the cross-fixture validation Tier-3 ground-
  truth check is the discriminator.
- **Merge tier-2 disabled.** `DEFAULT_MERGE_SEPARATION_FACTOR =
  0.5 = DEFAULT_STRUCTURAL_MERGE_FACTOR` so pairs in [0.5, 1.0]
  FWHM are not considered for merging. Tier 2 becomes safe to
  re-enable once the phase-degeneracy penalty
  (`stage5-fitting.md` O5-11) provides the LSQ-side signal to
  distinguish real close pairs from duplicate-pair LSQ
  artifacts. Empirically a no-op on the 2638 fixture: 6 → 4
  total merges, knockout-pruning 22 → 24, chi²_r distribution
  unchanged.

## Rescue-specific open follow-ups

External follow-ups (phase-degeneracy penalty, dataset-wide tau
calibration, borderline-real ground truth, Stage 3
projection-coherence) live in the planning docs that own each
topic. Rescue-specific follow-ups:

- **Sliding-coherence parameter calibration.** The shipped
  anchor pair (0.2 at the cluster floor, 0.8 at the isolated
  ceiling, ramping linearly from 1 to 5 FWHM) was chosen on the
  original 15-window sample and is not empirically calibrated
  against a wider set. Likely follow-ups: replace the linear
  ramp with the Lorentzian-skirt-magnitude functional form
  `threshold(Δ) = high − (high − low) · |basis(Δ, τ, T)|²`
  (tracks contamination level exactly); sweep `(close,
  isolated)` against the regime where clean windows acquire
  spurious peaks.
- **Round-cap calibration.** A 5-round w132 experiment showed
  the chain settling at round 3; `DEFAULT_RESCUE_MAX_ROUNDS=3`
  may be too low for some windows. Probably 5–7 once the
  rescue becomes a non-zero default.
- **Default flip from `max_residual_rescue_rounds=0` to the
  calibrated round-cap.** The rescue is a structural part of
  the fit, not an opt-in tweak; the 0 default is transitional.
  Gates on the cross-fixture validation work.
- **Per-window cost monitoring.** Every rescue round adds one
  `conservative_fit` + one joint refit + a knockout / merge /
  iterative-cleanup sweep. For the production pipeline (~400
  windows) the chain may multiply Stage 5 wall-time by a small
  constant; worth measuring once the rescue is on by default.
- **Audit-trail persistence.** The consolidated
  `ConservativeFitResult` inherits the initial fit's
  `audit_trail`; the rescue rounds and joint refits emit
  `RescueRoundDiagnostics` but those stay live-only (off
  `SpectrumFit`). Once the rescue is on by default, persist
  the per-window `RescueEvent` list into `SpectrumFit` so the
  on-disk fit is reconstruction-complete.
- **Promotion to `Pipeline.visualize_fit(rounds=True)`.** The
  harness layout for `detail.png` / `audit-trail.png` is stable
  enough to port into `visualization/fit_visualization.py`.
  Plumbing options for the audit data (re-run-on-demand vs
  persist `RescueRoundDiagnostics`) — re-run is the lighter
  starting move; persistence makes sense once the rescue is on
  by default.

## Validation-harness loose threads

Visualization-pass items that survived session boundaries; none
are blockers.

- **Per-peak provenance attribution is a heuristic.**
  `_peak_provenance` in `generate_validation.py` attributes each
  consolidated peak to a "source" (`init` / `r0` / `r1` / …) by
  closest offset to that source's added-peak list. When the
  joint refit reshuffles peaks across rounds, this is not exact;
  workable for first read. A more principled attribution would
  track peak identity through each joint refit by index
  permutation.
- **`detail-rr<n>.png` trajectory layout.** Still uses the older
  4×2 `plot_spectrum_fit` layout (no amplitude scaling, no peak
  listing). Deliberately not propagated to the trajectory
  artifacts — their role is "consolidated state at the end of
  round N" for forensic comparison, not "final answer." If the
  trajectory PNGs end up being used heavily, lifting them to
  the new layout is a natural follow-up.
- **Coherence-rejection X markers in `audit-trail.png` are
  unexercised.** The figure has marker code for coherence-
  rejected candidates (red X on the candidates row of each
  round). The 15-window 2638 sample has zero coherence
  rejections, so the markers have never rendered. A window that
  produces coherence rejections (or a synthetic test fixture)
  would close this gap.
- **Top-level `overview.png` doesn't use `DisplayStyle`.** The
  spectrum-wide overview still goes through
  `fit_visualization.plot_spectrum_fit` with active-FT-native
  amplitudes — no trim, no units scaling. Either thread the
  style into `plot_spectrum_fit` (library change) or emit a
  parallel harness-level overview (scratch-only).
- **Dead code from the model-overlay drop.**
  `_full_spectrum_model` and the `other_window_fits` parameter
  on `_plot_consolidated_detail` are no longer called after the
  full-spectrum context row dropped the model overlay (data+model
  overlap was unreadable at the figure scale). Left in place in
  case a future iteration wants a different reduced overlay; if
  still unused after another pass, remove them.
