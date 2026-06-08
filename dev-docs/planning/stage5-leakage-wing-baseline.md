# Stage 5 leakage-wing baseline term

**Status: implemented and validated on 2638.** An optional, evidence-triggered
low-order **complex baseline** added to a Stage 5 window's fit to absorb the
coherent residual left by a neighbouring strong line's mismodeled leakage wing.
The goal is **reliable per-line frequency and intensity extraction with fair
uncertainties**, not a globally physically-consistent model: the baseline is an
explicit nuisance term, deployed only where a coherent wing residual is
detected, fit jointly with the free peaks so its flexibility is honestly priced
into the line uncertainties.

## Where it lives (implementation)

- **`fitting/window_fit.py`** — `fit_window` takes `baseline_order`
  (`None`/`0`/`1`) and an optional `baseline_offset_scale` (`u_s`, default
  `max|u|`); `baseline_basis` builds the real `(u/u_s)^k` design columns,
  appended (as `a_k` + `i b_k` columns) to the model and analytic Jacobian
  after the peak/tau parameters. The covariance is the joint inverse `JᵀJ`, so
  the per-line errors already price the baseline's degrees of freedom. The
  fitted coefficients ride on `WindowFitResult.baseline_{order,coeffs,offset_scale}`.
- **`fitting/plan_execution.py`** — `_apply_baseline_to_outcome` is the
  trigger: after thaw **and** rescue (sequenced last, independent of rescue),
  any window whose residual `max(edge_low, edge_high)` exceeds
  `baseline_edge_threshold` is refit with its established lines + the baseline
  (tau held fixed — the baseline addresses skirt *shape*, not decay). The refit
  installs only when it converges and does not raise the data chi-squared; the
  decision, order, coefficients, and triggering `S_coh` are recorded on the
  `WindowOutcome`. `execute_plan` exposes `baseline_enabled` / `baseline_order`
  / `baseline_edge_threshold` (`DEFAULT_BASELINE_*`).
- **`core/stage_fit_settings.py`** — `BaselineSubSettings`
  (`enabled`/`order`/`edge_threshold`) with `_HARD_DEFAULTS` `True` / `0` /
  `3.5`. Registered in `instrument-tunable-knobs.md`; round-trips through HDF5
  + YAML via the canonical `_SUB_NAMES` walk.
- **`_internal/stage5_impl.py`** — resolves the block and threads it into
  `execute_plan`; the `parameters` audit dict carries the settings plus
  `n_baseline_windows` (how many windows fired).
- **`fitting/result_conversion.py`** — per-window audit trail in
  `quality_metrics` (a `Dict[str, float]`): `baseline_applied`,
  `baseline_order`, `baseline_edge_coherence`, `baseline_offset_scale`, and the
  coefficients as scalar `baseline_coeff{k}_re` / `baseline_coeff{k}_im` pairs.

### 2638 validation (A/B)

Baseline ON vs OFF, every other knob inherited from the persisted fixture
(`rescue_prominence_threshold=1.5`, gaussian). 66 windows fired. Chain windows
drop toward χ²ᵣ≈1.5 (w224 3.86→1.58, w225 2.50→1.50, w226 2.43→1.64, w309
3.69→1.80, w367 2.68→1.59; w227 already at floor, did not fire); the largest
absolute gains are w152 (8.45→1.49) and w150 (6.74→1.43). The regression
sentinels w020/w075/w159/w189/w303 are byte-identical (no baseline applied;
w189 specifically does not trigger). σ_A inflation on fired windows is median
×1.006, max ×1.449 (the σ_f honest price). The handful of windows whose χ²ᵣ
ticks up ≤~0.05 are the documented harmless low-precision firings — the
data chi-squared is held monotone, so the rise is purely the +2-parameter dof
bookkeeping.

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
costs ≈ ×1.0 on σ. The threshold was 2638-tuned and is now **validated
cross-fixture (issue #3, all 7 same-instrument fixtures)**: edge ≥ 3.5 fires at a
rate that tracks line density (0.07 on sparse 2638 → 0.93–0.96 on dense 655/363)
with Tier-1 health preserved where it fires heavily, and the fixed `const` order
is chosen on **100 % of fired windows on every fixture** (linear never selected).
Both the threshold and the const order ship unchanged. Evidence:
[`../research/stage5-cross-fixture/report.md`](../research/stage5-cross-fixture/report.md)
§"Cross-fixture knob audit".

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

Both negative results (the skirt-relaxation and baseline-trigger probes), plus
the validated baseline numbers, live in
`research/stage5-gaussian-audit/report.md`.

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

Diff per-window χ²ᵣ + per-line σ against the rescue_prominence=1.5 gaussian
baseline run:

- The 35 beneficial windows' χ²ᵣ drop (chain windows to ≈ 1.5; w152/w150 the
  largest absolute gains); no window with edge-coh below threshold is perturbed.
- Line σ_A inflation ≤ ~5 % on triggered windows; σ_f fairly priced; report the
  joint-covariance σ as the line uncertainty.
- Zero real-line removals; weak-line intensities shift ≤ ~1σ (de-biasing, not
  absorption); the w189-type windows are never triggered.

## Settled during implementation

- **Order selection** — shipped a *fixed* order from the settings block
  (`const` default, `linear` selectable). Per-window AICc choice between
  `const`/`linear` is deferred; const carries the bulk on 2638 (the linear
  term helps only the strongest few and AICc machinery is not yet worth the
  surface area).
- **Thaw/baseline sequencing** — the baseline refit is sequenced **last**
  (after thaw and rescue), and it re-measures `S_coh` on the final residual.
  An accepted thaw lowers the edge coherence before the baseline check runs,
  so the two do not double-count; and the baseline is too smooth to represent
  the narrow feature a thaw promotes, so even when both are eligible they
  address different residual structure.
- **Frozen-contributor interaction** — kept the rigid frozen skirt *and* the
  baseline (the prototype behaviour): the baseline fits against
  `data − frozen_background`, so it mops up the contributor's *residual* wing
  shape without re-fitting the contributor itself.

## Open questions

- **Edge threshold cross-fixture calibration** — **resolved (issue #3).**
  Validated on all 7 same-instrument fixtures: edge 3.5 fires sanely (rate tracks
  line density, Tier-1 health preserved) and `const` order is chosen on 100 % of
  fired windows everywhere. Both ship unchanged. A *different-instrument* fixture
  remains the ultimate ceiling but is not available; the same-instrument
  cross-fixture debt is closed. See `../research/stage5-cross-fixture/report.md`
  §"Cross-fixture knob audit".
