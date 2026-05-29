# Stage 5 leakage-wing baseline term

Add an optional, evidence-triggered low-order **complex baseline** to a Stage 5
window's fit to absorb the coherent residual left by a neighbouring strong
line's mismodeled leakage wing. The goal is **reliable per-line frequency and
intensity extraction with fair uncertainties**, not a globally physically-
consistent model: the baseline is an explicit nuisance term, deployed only
where a coherent wing residual is detected, fit jointly with the free peaks so
its flexibility is honestly priced into the line uncertainties.

## Motivation

A strong line's finite-T leakage skirt reaches into neighbouring windows. The
single-exponential / single-Gaussian skirt is the wrong shape in the far wing
(the strong line is often itself a blend or non-single-shape — e.g. w223 is an
unresolved doublet at χ²ᵣ=202), so the modeled skirt leaves a **coherent,
systematic** residual (on 2638: a positive-imaginary wing that persists across
several windows). That residual inflates χ²ᵣ in the dependent windows and, worse
for the extraction goal, **biases the weak lines that sit on top of it** — a
weak line absorbs the under-modeled wing and reads high.

This was reached by elimination (see "Design history" below): a
freeze-eligibility guard cannot reach it (the contributor is well-determined;
its χ²ᵣ lives in its own core), and a physically-consistent skirt relaxation has
no slack (a strong line's parameters are pinned precisely because it is strong).
The pragmatic, goal-aligned fix is to model the wing as a smooth nuisance
baseline where it is observable.

## Design

### Mechanism

A per-window optional complex baseline `B(u) = Σ_{k=0..p} (a_k + i b_k)(u/u_s)^k`
added to the window model (free peaks + rigid frozen contributors), fit jointly.

- **Order `p`:** `const` (p=0, two real params) captures the bulk; `linear`
  (p=1) squeezes a little more on the strongest cases; `quad` (p=2) overfits and
  is not used. Default `const`, `linear` allowed. Low order is the load-bearing
  guardrail: a baseline this smooth **cannot represent a ~1/(πτ) ≈ 0.05 MHz line
  over a multi-MHz window**, so it provably cannot absorb or mimic a real narrow
  line — it can only soak up a broad wing.
- **Fair uncertainty:** line σ_f / σ_A come from the **joint (peaks + baseline)
  covariance** `inv(JᵀJ)` on the unit-variance residual, so the added degrees of
  freedom are priced into the reported line errors. Measured on 2638: σ_A
  inflation ≈ ×1.00–1.03 on triggered windows (the broad baseline is nearly
  orthogonal to a narrow line's amplitude); σ_f ≤ ×1.4 where the baseline adds
  flexibility — the honest price.
- **Intensity de-biasing:** removing the under-modeled wing drops the affected
  weak lines' amplitudes by ~1σ_A (2638: w226 −23 %, w367 −30 %) toward their
  truer values — an improvement in the extracted intensities, not a loss.

### Trigger (empirically settled)

Enable the baseline only where a **coherent wing residual** is present, measured
by the existing `residual_edge_coherence` statistic (`S_coh`, the same detector
the thaw step uses) **above a dedicated threshold ≈ 3.5** — well below the thaw
default of 8.0, because the two serve different remedies. Bake-off on 2638 (391
windows, 35 "beneficial" = const-baseline χ²ᵣ drop ≥ 0.3 at acceptable σ):

| trigger | fires | precision | recall | harmful fires |
|---|---:|---:|---:|---:|
| edge-coh > 3.5 | 70 | 0.44 | 0.89 | 0 |
| edge-coh > 4.0 | 53 | 0.49 | 0.74 | 0 |
| edge-coh > 8.0 (thaw default) | 13 | 0.62 | 0.23 | 0 |
| fixed-contributor & χ²ᵣ > 2.0 | 17 | 0.47 | 0.23 | 0 |

Why edge-coherence beats the coarse "fixed contributor present + χ²ᵣ floor" rule:
the biggest wins (w152 +6.96, w150 +5.31) have **no fixed contributor** — the
leakage comes from a strong neighbour that was never formally carried as a
contributor. The coarse rule is structurally blind to those; edge-coherence sees
the residual directly. Edge-coherence also stays near its null (~0.9) on clean
and on narrow/low-SNR windows (w20 = 0.99, w159 = 1.86, w189 = 1.15), so a
threshold ≥ 3.5 produces **zero harmful fires** — it never adds a baseline to a
window where it would only inflate σ (the w189 case: ×1.16 σ_A for no χ²ᵣ gain).
Low precision at 3.5 is harmless: firing on a window that gains only +0.2 χ²ᵣ
costs ≈ ×1.0 on σ. The threshold is 2638-tuned → instrument-tunable calibration
debt (see [[stage5-penalty-tuning-debt]]); cross-fixture calibration is open.

### Relationship to existing machinery

- **Thaw** (edge-coh > 8.0 → promote a contributor to a free peak via joint
  co-fit): for a *missing real line* at the edge. The baseline is for a *wrong
  skirt shape*. Thaw should be attempted first (model a real line physically);
  the baseline then mops up the residual skirt shape. They cannot double-count a
  narrow feature (the baseline is too smooth to represent one).
- **Residual rescue** (adds residual peaks): removing the coherent wing means
  rescue no longer spuriously nominates a peak on the wing — a clean synergy.

## Design history (why not the alternatives)

- **Freeze-eligibility guard (original O4-2 framing).** Re-elevate a contributor
  whose primary fit is poor. Falsified on 2638: w223's contributor is *tightly*
  pinned (σ_A/A ≈ 0.001) despite χ²ᵣ=202 — χ²ᵣ and per-peak σ are anti-correlated
  (high-χ²ᵣ windows are strong lines with tight σ), so the AND criterion is
  inert and σ is defeated by shape-mismatch overconfidence. Architecturally the
  guard reaches only the ~11-unit downstream shadow, never w223's own 202 (a core
  shape problem).
- **Bounded skirt relaxation.** Give the frozen contributor a local effective τ +
  complex scale in the dependent window. Falsified: a *sub-uncertainty* (leashed)
  relaxation recovers only ~1.4 χ²ᵣ units; the relaxation that actually cleans
  the wing is large (scale → 2–3×, dφ → ~1 rad) and would blow up the dominant
  window's χ²ᵣ by 100× — i.e. it is a different line, not a consistent skirt. The
  slack the mechanism assumed does not exist for a strong line.

Both negative results, plus the validated baseline numbers, live in
`research/stage5-gaussian-audit/report.md` and the probes
`scratch/probe_skirt_relaxation.py` / `scratch/probe_baseline_trigger.py`.

## Implementation surface

- **`fitting/window_fit.py`** — extend `fit_window` (and the
  fixed-contributor variant) to accept an optional baseline order; append the
  complex-baseline columns to the model + Jacobian; the covariance already in
  place then yields joint line+baseline uncertainties for free. `u_s`
  (offset scale) for conditioning.
- **`fitting/plan_execution.py`** — compute `residual_edge_coherence` on the
  initial per-window residual (already available on the `WindowOutcome`); when
  `max(low, high) > baseline_edge_threshold`, refit the window with the baseline
  enabled. Sequence after thaw, before/independent of rescue. Record the baseline
  decision + fitted coefficients on the outcome.
- **`core/stage_fit_settings.py`** — new `baseline` sub-block: `enabled`,
  `order` (`const`/`linear`), `edge_threshold` (default ≈ 3.5). Register in
  `instrument-tunable-knobs.md`. Default-on once validated; ship default-off if
  the first integrated run shows any sentinel regression (transitional, cf.
  [[stage5-rescue-default-intent]]).
- **Persistence** — record per-window baseline order + coefficients + the
  triggering `S_coh` under the `stage5_fit` parameter group (audit trail).
- **Dual-interface** — internal to the fit, controlled by the settings block, so
  it flows through `pipeline.py` / `api.py` / CLI via existing plumbing. Add a
  cross-interface settings-propagation test.

## Tests

- **Unit:** a synthetic window with a smooth complex wing + a narrow line →
  baseline absorbs the wing, the line's frequency/amplitude are recovered, and
  the line σ from the joint covariance is finite and only mildly inflated; a
  clean window → baseline coefficients ≈ 0, line params unchanged; a narrow
  low-SNR window → edge-coherence below threshold so the baseline never fires.
- **Integration (2638):** the w223 chain (w224–w227) + w309/w367 and the
  broader 35-window beneficial set drop toward χ²ᵣ ≈ 1.5; the regression
  sentinels (w020, w075, w159, w189, w303) are untouched (edge-coh below
  threshold → no baseline); no real line removed; weak-line σ stays fair.
- **Cross-interface consistency.**

## Validation / acceptance

Diff per-window χ²ᵣ + per-line σ against
`scratch/stage5-validation-rescue_prominence_threshold__1p5__gaussian/`:

- The 35 beneficial windows' χ²ᵣ drop (chain windows to ≈ 1.5; w152/w150 the
  largest absolute gains); no window with edge-coh below threshold is perturbed.
- Line σ_A inflation ≤ ~5 % on triggered windows; σ_f fairly priced; report the
  joint-covariance σ as the line uncertainty.
- Zero real-line removals; weak-line intensities shift ≤ ~1σ (de-biasing, not
  absorption); the w189-type windows are never triggered.

## Open questions

- **Edge threshold cross-fixture calibration** (3.5 is 2638-tuned;
  [[stage5-penalty-tuning-debt]]). A second fixture is the real confidence
  ceiling.
- **Order selection per window** — fixed `const` vs. an AICc choice between
  `const`/`linear` per window (linear helps the strongest few; AICc would prevent
  needless params elsewhere).
- **Thaw/baseline sequencing** at edge-coh > 8 (both eligible): confirm
  thaw-then-baseline does not double-count and that an accepted thaw lowers
  edge-coh below the baseline threshold (so the baseline then no-ops).
- **Frozen-contributor interaction** — whether to keep the rigid frozen skirt
  *and* add the baseline (current prototype) or let the baseline subsume small
  contributors.
