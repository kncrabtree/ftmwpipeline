# Stage 5 — Residual rescue

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
   (`fitting.residual_screening.find_residual_peaks`) with the
   shape-error-aware sigma inflation gating which detections
   reach the fitter (see "Shape-error sigma inflation" below).
3. Run a second `conservative_fit` on the **residual itself**, with
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
`max_residual_rescue_rounds` (integer, defaults to
`DEFAULT_RESCUE_MAX_ROUNDS` = 5; pass `0` to disable the rescue
pass entirely as an escape hatch for diagnostic re-fits). The
cap is calibrated to be a safety net rather than the working
regime — on the 2638 fixture every window terminates naturally
at "no candidates" by round 3.

## Public surface

`fitting/residual_screening.py`

- `find_residual_peaks(...)` — `scipy.signal.find_peaks` with a
  sigma-relative height + prominence cut on `|residual|`.
  Thresholds: `snr_threshold` (default 2.5σ_c),
  `prominence_threshold` (2.0σ_c). All detector output flows to
  the rescue's `conservative_fit`; the downstream AICc accept
  gate + iterative cleanup are the false-positive control.

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
- `knockout_test(...)` and `conservative_fit(...)` accept an
  `n_eff_kind` parameter; see §"AICc-with-`n_eff` gates" below.

## Rescue tau policy

Real molecular lines in one experiment share a tau ≈ the calibrated
molecular decay (the Stage 2b `τ_maj`; the canonical FT is unapodized).
The rescue uses the initial fit's tau as the frozen line-shape width
for its `conservative_fit`, with a structural override for broken
initial fits.

The rule:

- Default to `initial.tau_us` — it's the LSQ-converged value for
  the real lines this window contains.
- **Override** to `tau_apodization_us` when `initial.tau_us` is
  within 5% of the lower bound (`tau_apodization_us /
  max_decay_factor`). That's the signature of a broken initial
  fit: LSQ over-narrowed tau to absorb unmodelled-peak residual.
  Using that broken tau as the rescue basis under-projects real
  peaks. The apodization is the right physical default.

The rescue tau is also handed to the joint refit's LSQ as its
starting tau (warm start for the apodization-override case).

### Shape-error sigma inflation (retired)

An earlier prototype inflated the rescue detector's per-bin sigma by a
parent-amplitude-proportional term — `σ_eff(f) = √(σ_c² + (ε ·
|current_model(f)|)²)`, threaded as a `shape_error_epsilon` screening
parameter — so candidates sitting under bright peaks had to clear the
expected Lorentzian-vs-true-shape residual to enter the fit. ε was a
**per-dataset constant** requiring per-fixture calibration (the chi²ᵣ ~
SNR² regression slope).

It is **retired**. The same chi²ᵣ ~ SNR² shape-error floor is now handled
dataset-agnostically and at the right layer by the SNR-aware acceptance
gate (`snr_aware_chi2_pass` / `shape_error_fraction` in `validation.py`,
ROADMAP D10): a window's reduced chi-squared is allowed to grow as
`F + (κ·SNR_max)²`, so a bright line fit to its lineshape-fidelity limit
passes the gate rather than spawning spurious sub-resolution rescue
candidates. The screening-sigma inflation knob is therefore redundant —
it defaulted to 0.0 in production (never plumbed through `stage5_impl`),
and removing it drops a per-dataset calibration burden. See
[`stage5-cross-fixture-validation.md`](stage5-cross-fixture-validation.md)
theme T4.

## AICc-with-`n_eff` gates

The Stage 5 hypothesis tests (conservative-loop accept,
blend-aware K=2/K=3 escalation, knockout, merge, iterative
cleanup) all gate on the AICc criterion evaluated at an effective
sample size `n_eff` that weights each bin by the local
information content of the model. Substituting `n_eff` into the
small-sample correction term `2k(k+1) / (n_eff − k − 1)` makes
the burden of statistical proof scale with the informative-bin
count rather than the full window. REJECT-on-tie at every site:
the simpler model is preserved when AICc cannot discriminate.

The single shared weighting kind, exposed as `DEFAULT_N_EFF_KIND`
in `validation.py`, is **`perplexity_log1p_snr`**: the perplexity
`exp(H(p))` of the normalised distribution `p_f ∝ log(1 +
|model|/σ)`. The `log(1 + SNR)` weight is approximately the
per-bin Shannon information of a signal-vs-noise detection. On
the 2638 fixture this returns ~30–80 bins on multi-peak windows.
Empirically the same kind works at every gate — the alternative
considered (a magnitude-concentrated Kish weight at the
K-vs-(K-1) sites) gave a marginally worse survey distribution
and added a per-site rationale to maintain without supporting
evidence; one default removes that maintenance burden.

The `effective_sample_size(..., kind=..., sigma=...)` API still
exposes the three magnitude-only kinds (`kish_mag_sq`,
`kish_mag`, `hard_radius`) for direct callers and diagnostics;
`sigma` is required for `perplexity_log1p_snr` and ignored by the
magnitude-only kinds.

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
policy"). When a Stage 2b calibration is plumbed
(`tau_penalty_sigma_us` set), the joint refit's free tau is also
regularized by the bidirectional Gaussian prior centred on
`tau_maj`: tau drifts freely when data strongly prefers it (e.g.
w198 at 33724 MHz on 2638, where data lands tau ≈ 4 µs against a
band-mid anchor of 6.16 µs and the prior contributes ~18 chi² of
penalty) but is pulled back from runaway / collapse when data
support is weak (the low-SNR isolated lines that would otherwise
peg the upper bound at 5 · tau0).

## Validation harness

`scripts/development/stage5-validation/generate_validation.py`
reproduces the initial fit per window, runs the consolidated
chain, and emits per-window artifacts to its (untracked) output
directory:

- `detail.png` — consolidated final fit. Full-spectrum overview
  + current-window axvspan; 3-column residual row (Re/Im/|z|)
  with vlines at fitted peak frequencies and cluster-aware
  letter labels; 3-column data+model row; |residual| histogram
  vs Rayleigh + peak listing with PDG-style spectroscopic
  uncertainties and the per-peak knockout p-value.
- `audit-trail.png` — rescue audit trail. Top: window magnitude
  spectrum with consolidated model overlay. Bottom: per-round
  audit panel laid out bottom-to-top chronologically with
  candidates row (detector candidates as green triangles,
  rescue-fit kept as open green circles) and merge row
  (knockout-pruned with red X, red border for rescue-origin
  failsafe firings).
- `detail-rr<n>.png` — trajectory snapshots (one per chain
  round). Useful for the monotonic-residual-shrink check.
- `report.md` — text rollup of the per-window plan, initial fit
  statistics, fitted peaks, audit trail, thaw events.
- `report-rr.md` — per-round rollup of the rescue chain:
  candidate list, joint-refit K and chi²_r, knockout pruning
  broken out by origin.

Read sequentially: `detail.png` (the answer) → `audit-trail.png`
(how we got here) → `detail-rr0.png` … `detail-rrN.png` (the
intermediate states if the audit needs forensic context).

## Known limitations

- **w198 on 2638 (resolved with Stage 2b + joint-refit unlock)**:
  the rescue chain on the unapodized FT with Stage 2b enabled
  lands at K=7, tau=4.0 µs, χ²_r=22 (vs the legacy apodized run's
  K=3, tau=2.4 µs, χ²_r=148). The unapodized FT exposes the dense
  cluster's 4 hidden peaks for the rescue to find; the
  bidirectional Gaussian prior keeps the joint refit's tau in
  the physical band without freezing it at `tau_maj`.
- **Borderline real-vs-noise on w16/w104/w127/w337-class
  windows**: the rescue's iterative cleanup rejects these
  borderline second peaks each round. Whether they are real
  weak lines or noise cannot be determined from the 2638
  fixture alone; the cross-fixture validation Tier-3 ground-
  truth check is the discriminator.
- **Merge tier-2 disabled.** `DEFAULT_MERGE_SEPARATION_FACTOR =
  0.5 = DEFAULT_STRUCTURAL_MERGE_FACTOR` so pairs in [0.5, 1.0]
  FWHM are not considered for merging. Tier 2 was to become safe to
  re-enable once the phase-degeneracy penalty
  (`stage5-fitting.md` O5-11) provides the LSQ-side signal to
  distinguish real close pairs from duplicate-pair LSQ
  artifacts. That penalty has since landed (O5-11 status: landed,
  `DEFAULT_PHASE_PENALTY_LAMBDA = 100`), so re-enabling tier 2 is now a
  deliberate decision rather than a blocked item — it stays disabled
  pending the cross-fixture Tier-3 evidence to justify the change.
  Empirically a no-op on the 2638 fixture: 6 → 4 total merges,
  knockout-pruning 22 → 24, chi²_r distribution unchanged.

## Rescue-specific open follow-ups

External follow-ups (phase-degeneracy penalty, dataset-wide tau
calibration, borderline-real ground truth) live in the planning
docs that own each topic. Rescue-specific follow-ups:

- **Per-window cost monitoring.** Every rescue round adds one
  `conservative_fit` + one joint refit + a knockout / merge /
  iterative-cleanup sweep. For the production pipeline (~400
  windows) the chain multiplies Stage 5 wall-time by a small
  constant; worth measuring now that the rescue is on by default.
- **Promotion to `Pipeline.visualize_fit(rounds=True)`.** The
  harness layout for `detail.png` / `audit-trail.png` is stable
  enough to port into `visualization/fit_visualization.py`. With
  `rescue_history` and per-window `rescue_events` on disk, the
  audit-trail figure can be rendered from the persisted fit
  without re-running the rescue chain; the harness's
  `_run_window_rescue` re-run path stays useful for forensic
  dives into the intermediate `WindowFitResult`s the persistence
  layer deliberately skips.

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
