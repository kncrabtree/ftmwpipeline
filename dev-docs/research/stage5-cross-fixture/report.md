# Stage 5 cross-fixture validation — preliminary findings

Status: **preliminary** (issues #2/#3/#4). This captures a session's worth of
cross-fixture work so it can be picked up in detail in a fresh session. The
headline new result is an **asymmetric long-anchor τ penalty** that resolves the
partially-resolved-hyperfine blends on the strong fixtures; it is prototyped and
window-level-validated but **not yet wired into production** (settings /
orchestrator / dual-interface / tests remain). Read alongside the reconciled
plan in [`../../planning/stage5-cross-fixture-validation.md`](../../planning/stage5-cross-fixture-validation.md)
and the (already-shipped) STFT τ work in
[`../stage5-tau-calibration/report.md`](../stage5-tau-calibration/report.md).

Fixtures: `examples/blackchirp_data/{2638,655,1019,360,363,1512,1231}`. Canonical
build per fixture: import → `detect_start_time` → `compute_ft(zpf=0, expf=None,
trim=(26500,40000))` → `estimate_noise` (scatter) → `calibrate_tau`. Tracked
recipe: [`stage5_cross_fixture.py`](stage5_cross_fixture.py) (T1 harness). The
asymmetric-τ prototype driver lives in `scratch/stage5_cross_fixture/`
(`asym_tau_ab.py`, not committed per the scratch-driver policy).

## T1 — characterization harness (done; backbone for #2/#3/#4)

`stage5_cross_fixture.py` builds + fits all seven fixtures and emits, per
fixture: Tier-1 χ²ᵣ health (median/p95/max), Tier-2 gate firing (merge rate,
`n_pruned_rescue_origin`, limit-cycle rounds), and the τ comparison (STFT vs
`calibrate_tau_G` vs a model-free FID-RMS τ), plus a roll-up JSON. Validated:
2638 reproduces the reference χ²ᵣ median **1.31**.

Tier-1: medians are healthy (1.1–1.4) on most fixtures with **heavy
catastrophic tails** (a few stuck windows per fixture); 363 is pathological
(median 28) and 655 elevated (median 2.04). Tier-2: `n_pruned_rescue_origin`
fires heavily (243/268/169 on 2638/655/1512) and thaw essentially never accepts
— both flagged for a later pass. **These tails are now substantially explained
by (a) wrong line-shape and (b) swallowed hyperfine blends — see below.**

## Line shape is per-fixture (and was a confound)

`recommend_shape` is decisive and per-fixture: **2638 is gaussian** (68% gauss
vote, 55% margin), **655 is lorentzian** (vinyl cyanide, 88% exp). Fitting the
wrong shape inflates χ²ᵣ via the classic Lorentzian-core/Gaussian-wing residual
(visible as a positive core residual on 2638 w233 under a Lorentzian fit). On
2638, simply using the gaussian shape drops w233 χ²ᵣ 56→14 and w272 22→13 —
independent of any τ change. **Any Stage 5 validation must fit each fixture in
its recommended shape**; the τ anchor must come from the matching calibration
(`calibrate_tau_G` band majorities for the gaussian path, `calibrate_tau` for
the lorentzian path).

## T2 — τ calibration

### Band-dependent τ is real (reconfirms the shipped 2638 result)

`calibrate_tau` `band_majorities` show τ falling monotonically with RF frequency
on every high-SNR fixture (2638 7.6→5.3, 1019 6.9→4.75, 1231 4.1→3.2, 655
3.7→3.1 µs low→high band); weak fixtures can't resolve it. This is the
horn-coupling τ∝1/f profile already characterized and shipped on 2638
(per-band routing, `per_band_tau=True`); the cross-fixture data confirms it
generalizes. The Gaussian τ_G majorities show the same gradient (2638 8.3→6.2).

### Model-free τ anchor and estimator biases

A model-free decay can be read off the raw FID as a sliding-window RMS, but only
after (1) truncating to the active region (drop the chirp) and (2) **subtracting
the DC offset** — without DC removal the constant offset pins the RMS floor and
collapses the dynamic range (2638 went dyn-range 0.4→12.7 once subtracted).
Trust it only on strong, single-dominant-line, single-exponential fixtures
(R²≥0.85, dyn-range≥4): 1019 τ=4.04 (R²1.0), 1231 3.41, 655 4.18. It fails (and
is correctly flagged) on beating doublets (2638 R²0.17) and weak spectra.

Against that anchor: `calibrate_tau_G` reads **high** everywhere (7.07/5.79 vs
true ~4 on 1019/655); the global STFT `τ_maj` is accurate on clean lines but
reads ~25% **low** on the dense, high-RF-weighted 655 (3.14 vs ~4.18) because it
is a band-averaged summary. Per-line narrowband demodulation is the wrong
model-free measure on vinyl cyanide — hyperfine splitting makes the demod
envelope beat and read a spuriously short τ.

### The blend-swallowing problem and the asymmetric long-anchor penalty (NEW)

On the very strong lines (655 especially), partially-resolved hyperfine biases
the STFT τ short → the Stage 5 model starts **too broad** → two blended
components get swallowed into one broad feature, ill-conditioned, never
separated. The fix (long ↔ narrow: a longer τ is a *narrower* line): **start
narrow, broaden cheaply, narrow expensively.** Concretely:

- Seed the NLS τ at a long (narrow) value `tm·F` — but only on windows that
  actually fit τ (strong); weak windows (frozen τ) keep the calibrated seed
  `tm` so they don't freeze too long.
- Anchor the prior at the calibrated `tm` (per-band) with an **asymmetric
  Gaussian penalty**: soft σ below the anchor (broadening cheap → fit relaxes
  onto the true τ and keeps blends resolvable) and stiff σ = σ_τ above (narrowing
  expensive → no tau-runaway / clean-feature overfit).

Implemented backward-compatibly in `src/ftmwpipeline/fitting/window_fit.py`
(`_penalty_residuals_and_jacobian`, `fit_window`,
`derive_window_fit_constraints`, `conservative_fit`, `_blend_aware_seed`) via a
new `tau_anchor_us` and `tau_penalty_sigma_lo_factor` (defaults None / 1.0
reproduce today's symmetric prior). **295/295 fitting unit tests pass.** Window-
level prototype defaults used so far: F=2.0 (F=3 overshoots), soft σ = 4·σ_τ,
anchor = per-band `tm`, narrow-seed gated to in-window SNR ≥ 10.

Prototype A/B (per-window `conservative_fit`, each fixture in its recommended
shape, vs the symmetric prior):

| fixture (shape) | window | OLD K/χ²ᵣ | NEW K/χ²ᵣ | note |
|---|---|---|---|---|
| 655 (lorentzian) | w29 | 3 / 44865 | **8 / 327** | hyperfine cluster resolved (137×) |
| 655 | w262 | 2 / 540233 | **8 / 38007** | ultra-strong line (SNR 57k) |
| 655 | w259 | 2 / 10057 | 8 / 7727 | |
| 2638 (gaussian) | w222 | 2 / 78 | **6 / 22** | under-counted wide window resolved |
| 2638 | w163 | 2 / 33 | 2 / 33 | no extra component in gaussian* |

Clean and weak windows are **unchanged** on both fixtures (τ stable, no spurious
peaks, no τ-runaway). *w163 resolved to K=3 under a *Lorentzian* fit (44→3.8) —
that third component was the Lorentzian compensating for its wing mismatch; in
the correct gaussian shape it stays K=2. Lesson: validate in the correct shape.

**Window-quality caveat (Stage 4).** Several of these high-χ²ᵣ windows are
*over-wide* — 655 w29 spans ±130 MHz and 2638 w222 ±24 MHz, with strong lines
jammed at the edges and little in between. That is a Stage 4 window-detection
problem (lumping spatially-distinct strong lines into one window), not a Stage 5
τ issue, and it confounds the per-window χ²ᵣ A/B (the narrow-start partly wins by
adding the edge lines a broad window forced into one fit). **Window detection
likely needs retuning** (tighter, feature-centred windows) before the Stage 5
per-window numbers are taken as final — flagged for a fresh session alongside the
D8 window review.

**Read:** the asymmetric narrow-start delivers its motivating win on the 655
Lorentzian hyperfine blends (orders-of-magnitude χ²ᵣ drops by resolving the
swallowed components) and is harmless on clean/weak windows. On 2638 the
dominant χ²ᵣ lever is *shape* (gaussian); the narrow-start adds genuine value on
under-counted windows. A residual tier remains on the very brightest cores
(χ²ᵣ still elevated after shape + narrow-start) — likely further hyperfine
substructure, residual shape deficit, or the K=8 `max_peaks` cap.

## What's validated vs. what's next (fresh session)

Validated (window level): the asymmetric long-anchor mechanism, backward-
compatible and unit-test-clean; its blend-resolution benefit on 655, its
harmlessness on clean/weak windows, the F≈2 sweet spot, and the per-fixture
shape requirement.

Next:
1. **Production plumbing**: `StageFitSettings.tau` gains `tau_long_factor` +
   `tau_penalty_sigma_lo_factor` (defaults 1.0); `fit_peaks_impl` seeds τ at
   `tm·F` gated to τ-fitting windows and forwards the factors; thread through
   `residual_rescue` (incl. the joint-refit warm-start) and the dual interface
   (`api`/`pipeline`/`cli`) + a cross-interface test.
2. **Full `fit_peaks` A/B** (not just per-window `conservative_fit`) on 655 +
   2638 with the correct shape, to confirm the rescue/merge/knockout chain
   behaves and the Tier-1/2 distributions improve; sweep F × soft for the
   production defaults.
3. The **residual tier** on the brightest cores (655 w262/w29 still χ²ᵣ ≫ 1):
   more components past the K=8 cap, or genuine substructure?
4. Carry the corrected shape choice into the T1 harness (it currently fits the
   default Lorentzian everywhere) and re-baseline Tier-1/2.
5. **Retune Stage 4 window detection** (D8 review): the high-χ²ᵣ windows are
   over-wide with edge-jammed strong lines and empty middles; tighter feature-
   centred windows are needed before the Stage 5 per-window χ²ᵣ numbers are
   final. This partly confounds the asymmetric-τ A/B above.

Out of scope (still fixture-blocked): #5 (3-way L/G/V) and #6's calibration half.
