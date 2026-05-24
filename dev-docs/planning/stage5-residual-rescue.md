# Stage 5 — Residual rescue and phase-coherence screening

Status: **implementation overview** for the production code that landed
in commits `3d403b7` (library functions) and the orchestration / B-loop /
sliding-coherence work that followed. The rescue pass is now wired into
the per-window orchestrator and the `fit_peaks_impl` public surface; the
gating knob is `max_residual_rescue_rounds` (integer, default `0` =
disabled, intended to become non-zero once validated at scale — the
rescue is a structural part of the fit, not an opt-in tweak). The
15-window 2638 validation set lives at `scratch/stage5-validation/`;
the harness emits one `detail-rr<n>.png` per consolidated round plus a
single rollup `report-rr.md`.

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
| w148 | 1→5 | 715 → **0.95** | missed 179σ peak captured round 0; the +0.5970 candidate (previously rejected by the 0.5 uniform threshold as "doublet leakage") is now accepted by the sliding threshold and survives knockout. |
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

## Open questions

### 1. Phase-coherence projection as a general primitive (Stage 3?)

The projection-coherence test is general — it answers "is this magnitude
peak the projection of a Lorentzian, or just incoherent magnitude?".
Stage 3 currently detects peaks on the magnitude spectrum and would
benefit from the same test:

- Distinguish real lines from baseline / contributor systematics
  (w337's case — see ROADMAP).
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

In order of dependency:

1. **Broader validation** (next-most-important): the 15-window sample's
   chi²_r → ~1 behaviour is encouraging but suggests possible
   overfitting; confirm against (a) the full 2638 fixture, (b) other
   FTMW datasets the user has access to, (c) blackchirp-era
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

### Validation-harness augmentations (clean-session work)

The harness (`scratch/stage5-validation/generate_validation.py`) has
become the primary inspection surface for the rescue chain. Three
augmentations are worth picking up in a clean session:

- **Per-window context view**: a small `context.png` per window
  showing the full active-FT spectrum with the current window's
  `freq_range` highlighted (axvspan). Reuses the existing
  overview-plot machinery; the addition is the per-window highlight.
- **vline markers at fitted-peak positions** on every detail figure
  (`detail.png` and each `detail-rr<n>.png`). Makes the peak count
  visually obvious at a glance and helps spot blended peaks the
  continuous-line overlay smears across.
- **Final audit-trail figure** (separate from the spectrum
  detail) showing the complete decision history with rounds laid out
  cleanly. The structure to aim for (per user discussion):
    - the existing conservative-loop audit format for the initial
      round, followed by
    - a visual break,
    - one block per residual round with that round's rescue audit
      (candidate proposals, F-test/AIC outcomes), then a "merge step"
      panel showing knockout survival vs rejection between the
      previous round's peak set and this round's.
    - More vertical space than the current detail.png layout allows
      — likely a dedicated figure rather than one panel.

A clean session is the right place to design these — the user has
sketches of both versions to discuss. Once the audit-trail figure is
stable, it's a natural candidate for **promotion to the main
visualization code** (`visualization/fit_visualization.py`,
`Pipeline.visualize_fit(window_id=..., rounds=True)`) so the rescue
chain becomes user-inspectable without re-running the harness. Two
plumbing options for that promotion:
- Persist `RescueRoundDiagnostics` to `SpectrumFit` so the viz reads
  from disk (schema change).
- Re-run the rescue on demand from the persisted state (lighter, ~1s
  per window).

Re-run-on-demand is cheaper for an initial promotion; persistence
makes sense once the chain is the default.

## How to assess the validation artifacts

```
conda run -n ftmwpipeline-dev python scratch/stage5-validation/generate_validation.py
```

writes `scratch/stage5-validation/window_NNN/detail.png` plus
`detail-rr<n>.png` (one per consolidated B-loop round) and a single
`report-rr.md` rollup. All figures use the same 4×2 layout:

- `detail.png` — initial-fit assessment: data = active-FT, model
  overlay = initial fit, residual panel = data − initial model.
  Orange triangles on the `|residual|` panel mark detector candidates.
- `detail-rr<n>.png` — consolidated state at the end of round *n*:
  data = full active-FT (same as `detail.png`), model overlay = the
  consolidated peaks at this point in the chain (initial + accepted
  rescue contributions through round *n*, jointly refit and
  knockout-pruned), residual panel = data − consolidated model.

Read sequentially: `detail.png` → `detail-rr0.png` → … The residual
panel should shrink monotonically toward the noise floor; any
remaining structure is what's still unexplained at that round.

`report-rr.md` has one section per round with the rescue's candidate
list, coherence-rejections, the joint refit's K and chi²_r, and the
knockout pruning broken out by origin (the failsafe diagnostic). It
also tags each round as ACCEPTED or REJECTED with the termination
reason.
