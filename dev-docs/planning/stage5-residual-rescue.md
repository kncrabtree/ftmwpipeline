# Stage 5 — Residual rescue and phase-coherence screening

Status: **implementation overview** for the production code that landed in
commit `3d403b7`. The rescue pass and the phase-coherence filter exist as
library functions; integration into the per-window orchestrator
(`_fit_one_window`) and the `fit_peaks_impl` public surface is **not yet
done** (next-steps section). The 15-window 2638 validation set lives at
`scratch/stage5-validation/`; the rescue mode is generated alongside the
existing `detail.png` / `report.md` as `detail-rr.png` / `report-rr.md`.

Normative requirements remain in the `*_STRATEGY.md` specs; the parent
plan is [`stage5-fitting.md`](stage5-fitting.md). This document is
normative only for the residual-rescue subsystem.

## What residual rescue is

The Stage 5 conservative loop only ever seeds peaks Stage 3 detected. Any
real line Stage 3 missed (or that knockout dropped during the conservative
fit) leaves an above-noise residual that the initial fit cannot explain.
The rescue is a second pass: detect peaks in that residual and try to add
them.

The rescue is **strictly separated** from the initial fit:

1. Compute `residual = data − model(initial.peaks, initial.tau)`. Done
   once; the initial fit is never touched again.
2. Detect candidate peaks in the residual
   (`fitting.residual_screening.find_residual_peaks`).
3. Drop candidates that fail a phase-coherence check
   (`fitting.residual_screening.filter_by_phase_coherence`).
4. Run a second `conservative_fit` on the **residual itself**, with tau
   frozen at the rescue tau (see below). The result's peaks are exactly
   the lines the rescue added; the initial fit's peaks are not present
   here.

A subsequent **joint refit** combining `initial.peaks + rescue.peaks`
with all parameters relaxed (plus a knockout sweep over the union) is
the *next step* — not done by this function. See "Total-model integration
strategy" below.

## Public surface

`fitting/residual_screening.py`
- `find_residual_peaks(...)` — `scipy.signal.find_peaks` with a
  sigma-relative height + prominence cut on `|residual|`. Threshold knobs:
  `snr_threshold` (default 2.5σ_c), `prominence_threshold` (2.0σ_c).
- `filter_by_phase_coherence(candidates, ...)` — projects each candidate's
  residual onto a unit-amplitude Lorentzian basis at the candidate's
  offset; rejects isolated candidates whose coherent SNR is below
  `coherence_ratio_threshold` (default 0.5) of the detected magnitude SNR.
  Clustered candidates (others within `coherence_cluster_fwhm * FWHM`,
  default 1.0) pass through so the blend-aware seeder handles legitimate
  close pairs.

`fitting/residual_rescue.py`
- `attempt_residual_rescue(...)` → `RescueOutcome` with `fit`, `audit`,
  `candidates`, `rejected_by_coherence`, `knockouts`.
- The rescue's `conservative_fit` is called with `fit_tau=False` so the
  rescue's peak set shares one frozen tau.

`fitting/window_fit.py` (refactor)
- `derive_window_fit_constraints(...)` → `WindowFitConstraints` extracts
  the tau / amp / penalty derivation that used to live inline in
  `conservative_fit`. Both `conservative_fit` and the rescue use it so
  they enforce identical constraints. No behavioural change to the
  original conservative fit.
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

- **Real Lorentzian peak**: ratio ≈ 1 (the bin magnitude is dominated
  by a coherent line that the basis captures).
- **Phase-rotation artifact** (e.g., w269's +1.4558 candidate, 27.6σ
  magnitude but caused by leakage of a slightly-mis-fit neighbour):
  ratio ≪ 1 (the complex projection cancels across the rapidly-rotating
  phase).
- **Edge-effect / numerical noise**: ratio low.

Default threshold 0.5 (require coherent SNR ≥ half the detected SNR).

**Clustered candidates are not tested.** A real close pair contaminates
each other's projection: peak B's leakage skirt biases the A-projection
and vice-versa. The cluster check (`coherence_cluster_fwhm * FWHM`,
default 1.0) defers clusters to the blend-aware seeder for handling.

## Validation results (2638 fixture, 15-window sample)

`scratch/stage5-validation/generate_validation.py` reproduces the initial
fit per window, runs the rescue, builds a synthetic single-window
`SpectrumFit` with only the rescue peaks, and overlays it on the
residual via the existing `plot_spectrum_fit`. Outputs land in
`window_NNN/detail-rr.png` and `window_NNN/report-rr.md` alongside the
original `detail.png` / `report.md`.

Behaviour on the sample:

| Window | Notable outcome |
|---|---|
| w64, w63 | Clean residual, 0 candidates, no rescue. No regression. |
| w16 | One borderline 2nd peak rescued (1/1 accept). |
| w104 | Stage 3 candidate was likely spurious; rescue adds 2–3 real peaks. |
| w127 | Borderline second peak rescued. |
| w148 | Missed peak (179σ) rescued; doublet leakage at +0.5970 (17σ) phase-coherence rejected as fit imperfection. |
| w198 | Initial fit broken (tau=1µs). Apodization-tau override engages, real peaks pass coherence, rescue adds 4. |
| w269 | Phantom +1.4558 (27σ but phase-incoherent) correctly rejected by coherence filter. |
| w68, w260, w271 | Mix of accepts and phase-coherence rejections (mostly edge effects and small artifacts). |

The phase-coherence filter was the key insight: without it, the rescue
fit phantom peaks at phase-rotation artifacts (w269 grew a 5-peak
cluster modelling one phase-rotation pattern). With it, the rescue
nominates only candidates that pass an explicit Lorentzian-coherence
test before fitting.

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

### 2. Physical interpretation of the rejected (non-Lorentzian) signals

The phase-coherence filter rejects candidates that don't fit a Lorentzian
basis. In this 2638 fixture, those mostly correspond to known artifacts:
phase-rotation patterns from slightly-mis-fit neighbours (w269), doublet
leakage (w148), and window-edge effects (w68). But the residual could
also carry **physical** non-Lorentzian signal:

- **Doppler (Gaussian) broadening** convolves the Lorentzian into a
  Voigt profile. If the Doppler width is non-negligible, a pure
  Lorentzian basis would miss the Gaussian wings, leaving a structured
  residual.
- **Hyperfine / fine-structure splitting** beneath Stage 3's resolution
  would appear as a phase pattern across multiple closely-spaced
  unresolved lines.
- **Instrumental effects** — power broadening, magnetic-field
  inhomogeneity, mixing-product artifacts.
- **Just numerical noise** — most rejections in clean windows fall
  here.

A diagnostic pass for a fresh session: collect all coherence-rejected
candidates across the full 2638 fixture, plot their residual neighbourhoods,
and look for systematic structure. If a substantial fraction sit on top of
fitted peak centres (not random offsets), that points to physical
Voigt-Lorentzian mismatch rather than fit imperfection.

### 3. Total-model integration strategy

The rescue gives us `initial.peaks` (initial fit) and `rescue.peaks`
(fit on residual) with two separate tau values. Two approaches to
producing a single unified fit:

**A. Recursive residual-rescue with unrolling refits (user's lean).**
- Round 1: initial fit → residual → rescue₁ (frozen tau, residual-only).
- Round 2: rescue₁ residual → rescue₂.
- ... up to a small max depth (2–3).
- Unroll: at each level *up the chain*, do a thawed joint refit
  combining all peaks accumulated through that level.

**B. Full joint refit between rescue rounds.**
- Round 1: initial fit → residual → rescue₁.
- Joint refit `initial.peaks + rescue₁.peaks` with all params relaxed
  → "merged" fit. Knockout sweep over the union.
- Residual of merged fit → rescue₂.
- Joint refit `merged.peaks + rescue₂.peaks` → next merged.
- ... up to a small max depth.

Trade-offs:

| Aspect | A: recursive then unroll | B: refit between rounds |
|---|---|---|
| Cost per round | low (single conservative_fit on residual) | high (joint refit grows with K) |
| Tau drift across rounds | accumulates — each rescue uses frozen tau from its level | each refit re-converges tau globally |
| Sensitivity to broken initial tau | every level inherits the broken tau (apodization-override mitigates) | first joint refit can rescue tau using newly added peaks |
| Phantom propagation | phantoms accepted at round N feed round N+1's residual | each joint refit + knockout sweep can prune phantoms before the next residual screen |
| w198-like case | requires many levels; tau policy is critical | joint refit after round 1 likely fixes tau immediately, fewer rounds needed |

The user noted a preference for **A (recursive with max depth)** but
asked for critical assessment. My read: **B is structurally safer** —
the joint refit absorbs the rescue's contribution into the global model
before the next residual screen, so each round operates against a
fresh, properly-jointly-fit baseline rather than against an
ever-thinner residual that may not be physical. The cost difference
matters less than tau-stability and phantom-pruning.

A hybrid is possible: do A for a "rescue burst" of low max-depth (say
2), then a single B-style joint refit + knockout at the end. This keeps
cheap residual fits cheap while still reconciling globally before
returning the result.

The choice should be revisited once we have a wider validation set —
the current 2638 sample doesn't expose a case where rescue₂ would
matter (w198 is the candidate, but its first rescue already drops chi²_r
from 628 to 187 with 4 peaks; a second rescue would be tested against
the merged-fit residual, not the rescue-only residual).

## Next steps

In order of dependency:

1. **Production wiring of single-pass rescue** — `_fit_one_window` in
   `fitting/plan_execution.py` calls `attempt_residual_rescue` after the
   initial fit; plumb `enable_residual_rescue` and the detector/coherence
   knobs through `execute_plan` and `_internal/stage5_impl.fit_peaks_impl`
   with default-off behaviour so existing tests don't shift. Tracked as
   ROADMAP D-equivalent task; pending tasks #38 / #39 in the scratch
   session note these.
2. **Choose the integration strategy (A/B/hybrid)** and implement the
   joint refit + knockout sweep over the (initial + rescue) union. The
   joint refit step is what turns the rescue from a diagnostic into a
   production fit improvement.
3. **Decide on max recursion depth** and the per-round termination
   criterion (no new accepted candidates? residual chi²_r threshold?
   no further AIC improvement after joint refit?).
4. **Per-window cost monitoring** — every rescue round adds one full
   `conservative_fit` call. For the production pipeline (~400 windows)
   this may double or triple Stage 5 wall-time. Worth measuring on a
   full fixture once production-wired.
5. **Independent revisit of the projection-coherence idea for Stage 3**
   — separate planning doc once Stage 5 rescue is settled.

## How to assess the validation artifacts

```
conda run -n ftmwpipeline-dev python scratch/stage5-validation/generate_validation.py
```

writes `scratch/stage5-validation/window_NNN/detail.png` and
`detail-rr.png`, plus matching reports. The two PNGs use the same 4×2
layout:

- `detail.png` — original initial-fit assessment: data = active-FT,
  model overlay = initial fit, residual panel = data − initial model.
  Orange triangles on `|residual|` panel mark candidates the residual
  detector finds.
- `detail-rr.png` — rescue assessment: data = residual from initial fit,
  model overlay = rescue peaks (only), residual panel = residual − rescue
  model.

Side-by-side reading: detail.png's residual structure should match
detail-rr.png's data. Detail-rr's residual panel should be at noise
level if the rescue captures everything; remaining structure is what's
still unexplained.

Reports include the conservative-fit audit trail (rescue's own, not the
initial fit's), accepted candidates, and the phase-coherence rejections
with their detected SNRs — useful for spotting cases where the
coherence filter mis-classifies a real peak (none seen so far on the
2638 sample with the current 0.5 ratio threshold).
