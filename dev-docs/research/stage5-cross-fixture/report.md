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

## Phase 1 — Stage 4 window geometry under massive-SNR lines (NEW)

Driver: `scratch/stage5_cross_fixture/phase1_window_geometry.py`,
`phase1_inspect_offenders.py`, `bounded_merge.py`, `phase1_resegment_ab.py`
(geometry + offender population + the bounded-merge A/B).

### The "over-wide window" framing was wrong; the windows are GHz mega-windows

The high-χ²ᵣ 655 windows are not ±130 MHz over-wide-with-empty-middles. They are
**GHz-scale mega-windows packed with hundreds of genuine lines**: w262 spans 1429
MHz with 587 detected peaks (occupancy 0.99), w259 1190 MHz/438, w39 890 MHz/286,
w29 298 MHz/109. The detections are all `detection_pass='primary'`, all promoted,
133/40/57/12 of them STRONG, and spread across the whole band (w262 median
|f−f_bright| = 393 MHz; only 6/587 within 5 MHz of the bright line) — i.e. **real
vinyl-cyanide forest lines, not a bright line's leakage skirt**. Stage 5 then fits
only K=3–4 (the brightest doublets) under the K=8 `max_peaks` cap → χ²ᵣ 46030 on
w262. By contrast 2638 (SNR ≤ ~400) tops out at a 70 MHz window; window width and
detected-count scale with the brightest line's SNR.

### Root cause: the Step-3 strong-cluster force-merge

`build_window_plan` Step 3 force-merges every STRONG line sharing one
leakage-touched interval into a `(min, max)` grid span. In a dense, ultra-high-SNR
spectrum the rolling-coherence `touched` run covers the whole band (the SNR-10⁴–10⁵
skirts + the line forest never let `S_coh` drop below `T_edge`), so all 133 strong
lines collapse into one 1429 MHz window. The width-cap remedy is advisory only:
over-cap windows get a single `split_proposal` (or `needs_joint_treatment=True`
with none), and neither flag is consumed by Stage 5. This is exactly the
"strong line's touched run → mega-windows" risk the D8 review flagged (80–100 MHz
on 2638; GHz on 655) — the Stage-5 evidence the review deferred for.

### The bounded-merge fix: necessary, and it fixes the geometry

`bounded_merge.build_window_plan_bounded` (a verbatim copy with only Step 3
changed) caps the strong-cluster merge at `max_window_width_mhz` and adds an
*applied* post-merge split — recursively at the largest internal peak gap until
each window holds ≤ `max_peaks` (8) and ≤ the width cap, splitting at the gap
midpoint so windows stay disjoint and cross-window coupling is carried by the
Step-4/5 fixed contributors. Geometry (655): 278→633 windows, max width
1429→39.5 MHz, peaks/window max 587→8, 18→0 over-capacity windows; the fit now
models **1675 vs 707** lines. 2638 tightens similarly (70→29.7 MHz max) with no
loss of fittability. Bounding Step-3 alone is *insufficient* (655 is a genuine
dense forest, median line spacing 1.6 MHz, so the per-peak ±2 MHz proto-spans
re-chain) — the applied peak-count + width cap split is the load-bearing piece.

### But χ²ᵣ is a model-fidelity-vs-SNR floor, not a windowing metric

Counterintuitively the bounded plan makes Tier-1 χ²ᵣ *worse* (655 max
46030→1.02e6, p95 12→373) even though it fits far more lines and every window is
now sane. The reason: **per-window χ²ᵣ scales with the brightest in-window SNR**,
independent of windowing —

| max in-window SNR | windows | χ²ᵣ median |
|---|---|---|
| <100 | 568 | 2.9 |
| 100–1k | 55 | 70 |
| 1k–10k | 4 | 2364 |
| ≥10k | 6 | 208951 |

(Spearman χ²ᵣ vs maxSNR = +0.59.) The worst window (w420, maxSNR 16k) fits
successfully — resolves the line into a hyperfine triplet with 0.02 % amplitude
precision — yet χ²ᵣ = 1.02e6, because at SNR 10⁴–10⁵ the (correct, un-inflated
scatter) noise is so small that a sub-percent lineshape/τ deficit is hundreds of σ
per bin. Its fitted τ = 2.63 µs is pinned low (the STFT band-averaged bias; true
~4 µs), so the model is too broad → systematic core residual — the asymmetric-τ
target, but it can only lower the floor, not remove it (the analytic lineshape is
itself imperfect at part-in-10⁵). The mega-window's *lower* χ²ᵣ was an artifact of
diluting one bright line's residual across 587-line dof under an under-fit model.

**Consequences.**
1. The bounded merge is the right Stage 4 fix and is *required* to fit dense
   spectra at all. **PRODUCTIONIZED**: bounded Step-3 merge + applied
   peak-count/width cap-split live in `window_planning.build_window_plan`, gated
   by a new `clustering.max_peaks_per_window` setting (default 8, tracking Stage 5
   `conservative.max_peaks`), threaded through `assign_windows`
   (api/pipeline/cli) + serialization + unit/cross-interface tests. All 7
   fixtures re-plan clean (no window >40 MHz, none >8 peaks). The **2638 39 MHz
   doublet regression check** was run and resolved: splitting raises the bright
   doublet's window χ²ᵣ 21.7→199, but this was *measured* (not argued) to be pure
   χ²ᵣ dilution — the old 70 MHz window bundled ~180 noise-only bins that diluted
   the bright doublet's core residual; the new 7.3 MHz window excludes them,
   concentrating the *same* residual (95 % of the χ² is the 0.133 MHz SNR-400
   doublet core; subtracting the cycle-dropped 36389 contributor changes χ²ᵣ by
   0.01 %; all 29 promoted peaks in the region stay covered, no gaps). The lines
   are recovered identically (freq errors 0.2 kHz) — the fits are equivalent
   under the SNR-aware metric. (Earlier I claimed "no regression" from a leakage
   estimate and was wrong on the raw χ²ᵣ; the refit corrected it.)
2. The Tier-1 χ²ᵣ gate (median≤1.5, p95≤4, max≤10) is **unachievable at extreme
   SNR** and must be reformulated SNR-aware (judge on the maxSNR<100 bulk, or on
   fractional residual, or with the brightest cores excluded/capped). This is the
   same accuracy floor as the 655 σ_f and noise-vs-SNR findings.
3. The bulk (568/633 windows, maxSNR<100) fits at χ²ᵣ median 2.9 — that is where
   the asym-τ + correct-shape tuning is meaningful and the Phase-2 A/B belongs;
   the bright-core windows are floor-limited, so χ²ᵣ deltas there are not the
   right success signal.
4. The MAX_PEAKS=8 split may be *too* tight: the bulk median rose 2.04→2.9, partly
   from boundary windows now leaning on imperfect fixed-contributor leakage from
   bright neighbors (the D8 "does the leakage model carry the skirt" question). A
   window size between mega and ≤8-peak likely trades fit-capacity against
   contributor error — a tuning knob to sweep.

### SNR-aware health metric

The raw Tier-1 χ²ᵣ gate is meaningless at extreme SNR (see above). Two regimes:
low-SNR windows are noise-dominated (χ²ᵣ≈1 is the right target, the model deficit
is invisible below noise); high-SNR windows are deficit-dominated (χ²ᵣ=(SNR·ε)²).
The unifying gate is **χ²ᵣ ≤ 1 + (κ·SNR)²** (κ = tolerated fractional model
deficit), which collapses to χ²ᵣ≈1 at low SNR and grows ∝SNR² at high SNR. The
deficit-regime quantity is the fractional core residual **ε = √(max(χ²ᵣ−1,0))/SNR**;
on 655 it *decreases* with SNR (bright isolated lines fit cleanest, ~1.5 %; the
moderate/dense-forest windows carry larger relative residual ~3–4 %), so the model
work belongs on the dense bulk, not the bright cores.

### Window-tightness sweep (655): looser is better, but a floor at ~2.4 remains

Sweeping the coupled cap C (= Stage-4 per-window peak cap = Stage-5 fit
`max_peaks`) — width cap fixed at 40 MHz, so this varies window size ~13→40 MHz:

| C | windows | lines fit | bulk χ²ᵣ med | bulk ε | all-win p95 | gate pass |
|---|---|---|---|---|---|---|
| 8 | 633 | 1675 | 2.85 | 14.5 % | 373 | 0.142 |
| 16 | 492 | 1492 | 2.47 | 13.2 % | 280 | 0.175 |
| 24 | 467 | 1435 | 2.40 | 12.4 % | 234 | 0.188 |

Looser windows are monotonically better on every metric with no conditioning
blowup; the gain plateaus (all configs are pinned at the 40 MHz width cap). But the
**bulk χ²ᵣ floor sticks at ~2.4 regardless of window size** — window sizing alone
cannot reach χ²ᵣ≈1.

### The bulk floor is unmodeled cross-window leakage: the cycle-breaker drops ALL fixed contributors on 655

Root cause of the ~2.4 floor, found by tracing the zero-contributor anomaly: 655
gets **0 fixed contributors** from Stage 4 (2638 gets 50), even though it has the
brightest lines (SNR 30k) and strongest skirts. The magnitude attachment *does*
fire — it creates **2136** FixedContributors (skirts up to ~0.1, vs a ~8e-4 σ
threshold) — but each creates a window→primary dependency edge, and on a dense
strong-line forest the dependency graph is **densely cyclic** (windows mutually
attach each other). The Step-7 cycle-breaker drops cyclic edges *and their
FixedContributors* (`window_planning.py` ~648–651): on 655 **all 984 edges are
cyclic → all dropped → 0 survive**; on sparse 2638 only 92/122 drop, 50 survive.
So **no bright-line leakage is subtracted from any neighbor window on 655** — the
dominant driver of the bulk floor (and a contributor to the bright-core χ²ᵣ).

This is the deferred O5-10 "cumulative-tail subtraction" item, acceptable at
SNR~10³ (2638, weak skirts, most contributors survive) but catastrophic at
SNR~10⁴–10⁵ dense. The fix decouples leakage *subtraction* from fit *ordering*: a
detected strong line's skirt can be subtracted as a frozen, pre-computed additive
background (from its freq/intensity/τ) without requiring its window to fit first —
removing the dependency edge and thus the cycle. **This is likely a larger lever on
the 655 bulk χ²ᵣ than either window sizing or the asym-τ penalty**, and should be
prototyped before the Phase-2 asym-τ A/B (which targets the residual lineshape/τ
deficit that remains *after* leakage is subtracted).

### Frozen-from-detection skirt prototype (`phase1_frozen_skirt_ab.py`)

Reads each strong line's complex core straight from the active FT (amplitude =
2|z_core|/τ_eff, phase = arg(z_core)) instead of from its primary window's fit, so
the frozen skirt needs no fit-ordering dependency (no edge, no cycle). Per-window
A/B on the bounded-C24 655 windows, subtracting the would-be contributors'
(predicted skirt ≥ 0.1σ_c) frozen skirts before a bare `conservative_fit`:

- **Mechanism validated**: large wins where a bright neighbour dominates — w65
  χ²ᵣ 10556→342 (31×), w346 28912→12433, w425 38820→19782. The cycle-breaker drop
  is a real, recoverable loss.
- **Median gain modest** (bulk χ²ᵣ 10.67→8.79, −18 %) and **confounded**: (1) this
  bare `conservative_fit` has no leakage-wing baseline, so its OLD bulk (10.67) is
  ~4× the real `fit_peaks` bulk (2.40) — the shipped edge-coherence baseline
  already absorbs much in-window leakage, so the marginal value *on top of it* is
  untested here; (2) the 0.1σ threshold saturates at SNR 30k (all 304 strong lines
  attach to every window), and summing 304 single-bin-read skirts injects
  read-noise. The bright-core window (w381, SNR 30k) barely moves (1.22e6→1.22e6) —
  its floor is SNR² model-fidelity, not neighbour leakage.

**Decisive in-pipeline test (run; NEGATIVE).** `phase1_frozen_inpipeline.py`
pre-subtracts every strong line's skirt (phasor from its core) from the active-FT
spectrum *except* its home window — the center-independent global form
`phasor_j·h_T(s·(f−f_j);τ_j)` — then runs the real `fit_peaks` (baseline + rescue
intact) on the corrected FT via a `compute_active_ft` monkeypatch. Result: the bulk
χ²ᵣ gets **worse**, 2.40→4.71 (bulk frac≤2 0.38→0.06, p95 234→347). Two reasons,
both decisive:

1. **The shipped leakage-wing baseline already handles bulk leakage.** The bare
   A/B's apparent win was relative to a baseline-free `conservative_fit` (OLD bulk
   10.67); against the real pipeline (OLD bulk 2.40) crude subtraction only hurts.
2. **Single-bin phasor reads are pedestal-contaminated on a dense spectrum** — each
   strong line's core bin also carries ~300 other lines' summed skirts, so every
   `phasor_j` is over-estimated → over-subtraction → 304 accumulated residuals
   inject more error than they remove.

**Reframe:** the bulk χ²ᵣ ~2.4 floor is **not primarily recoverable cross-window
leakage** (the baseline already covers it) — it is the genuine lineshape/τ model
deficit. So the cycle-breaker contributor drop, though real, is *masked* by the
baseline; a frozen-contributor fix is the wrong lever in this global crude form.
Redirect the bulk-floor work to **Phase 2 (asym-τ + per-fixture shape)**. A frozen
contributor is only worth revisiting in a *targeted, robust* form (a few genuinely
adjacent bright neighbours, LSQ amplitude reads to avoid pedestal contamination) for
the handful of bright-neighbour windows (e.g. the bare A/B's w65, 31× win) — not as
a global pre-subtraction.

## Phase 2 — asymmetric-τ penalty on the bounded bulk windows (NEGATIVE)

Drivers: `phase2_asym_tau_bulk.py` (bare conservative_fit A/B over all bounded
windows) and `phase2_asym_tau_inpipeline.py` (asym-τ injected into the real
`fit_peaks` via a `conservative_fit` monkeypatch, baseline + rescue intact). The
asym-τ premise was: STFT τ biased low → bulk models too broad → seed τ long
(`tm·F`) with a soft below-anchor σ so it relaxes onto the true (~4 µs) value.

**Result on the maxSNR<100 bulk: null.** Bare A/B: bulk χ²ᵣ 10.67→9.23 (F=3, the
no-baseline level), but **τ does not relax** — median 3.12→3.09, and only 63/416
windows moved longer despite a tm·3≈9 µs seed. In-pipeline (the decisive test):
bulk χ²ᵣ **2.40→2.405**, bulk frac≤2 0.38→0.387, **bulk τ median 3.069→3.068**.
The bulk data genuinely prefers τ≈3.07 per-band; the long seed is pulled straight
back down. The asym-τ penalty's headline wins (655 w29 44865→327, w262 540233→38007)
were **entirely the mega-window windowing artifact** — once the bounded windows
isolate the bulk, asym-τ does nothing to it. (It may retain niche value on the
~15 % of windows with a genuine swallowed blend, but it is not a bulk-floor lever.)

### Synthesis — the bulk χ²ᵣ ~2.4 floor is irreducible model fidelity

Three independent levers are now ruled out for the bulk floor: **window sizing**
(sweep plateaus at 2.4), **cross-window leakage** (the shipped baseline already
covers it; crude frozen subtraction makes it worse), and **τ** (no relaxation, no
gain, in-pipeline confirmed). The bulk χ²ᵣ ≈ 2.4 at moderate SNR corresponds to a
fractional core residual ε ≈ 3 % — the honest fidelity of the Lorentzian model +
low-order baseline against real vinyl-cyanide lines (plus the ~10 kHz accuracy
floor). It is **not a defect to chase to χ²ᵣ→1**; the correct response is the
SNR-aware / fractional-residual metric (κ ≈ 3 %), under which the bulk is healthy.
The one shippable win from this arc is the **bounded-merge Stage 4 fix**, which is
required to fit dense spectra at all; the asym-τ production plumbing is **not**
warranted on this evidence.

## Issue #2 — metric reconciled, CLI landed, Tier 3 run

The SNR-aware metric is now production. The Tier-1 gate is reformulated (ROADMAP
D10) as **`χ²ᵣ ≤ F + (κ·SNR_max)²`** with the fractional deficit
**`ε = √(max(χ²ᵣ−F,0))/SNR_max`** reported, binned by `SNR_max`. The primitives
(`snr_aware_chi2_pass`, `shape_error_fraction`, `DEFAULT_SHAPE_ERROR_KAPPA=0.05`,
`DEFAULT_CHI2R_NOISE_FLOOR=3.0`) live in `fitting/validation.py` and are consumed
by the dual-interface read-only command **`validate-stage5-shape-error`**
(`_internal/stage5_validation_impl.py` → api/pipeline/cli + a cross-interface
test). The two regimes:

- **Noise-dominated** (low SNR): the deficit term vanishes, the gate is
  `χ²ᵣ ≤ F`. `F=3` budgets for the reduced-χ² sampling scatter of a *good* fit
  (mean ~1, variance ~2/dof) — `F=1` would reject healthy noise-dominated windows
  for normal upward fluctuation. With it, ε reads 0 there (no measurable deficit).
- **Deficit-dominated** (high SNR): `(κ·SNR)²` governs; bright cores fit to
  part-in-10⁵ pass (χ²ᵣ up to ~10³–10⁴) instead of failing for being bright.

The earlier claim that the dense bulk "is healthy at κ≈3%" was incomplete: the
bulk's elevated χ²ᵣ at *low* SNR is noise-regime scatter, not a fractional
deficit (ε is meaningless there) — it is the noise floor `F`, not κ, that admits
it. The true fractional deficit, measured where it is measurable (SNR≥100), is
**~0.8–1.3%** on 655 — *better* than the ~3% the bulk-floor synthesis above
inferred from low-SNR χ²ᵣ.

Cross-fixture, post-bounded-merge, each fixture in its recommended shape
(κ=0.05/F=3): **1512** (lorentzian, low-SNR) overall pass 0.93, bulk χ²ᵣ median
1.17; **655** (lorentzian, dense extreme-SNR) overall pass 0.69, bulk χ²ᵣ median
2.71 (the genuine fidelity floor), SNR≥1k bins pass 1.0. Tier 3: the **frequency-
accuracy floor** is ~9 kHz on the sparse 1512 VC list (detrended; raw 16 kHz, a
2.3 kHz/GHz clock drift + a −84 kHz offset) and the reported LSQ σ_f is honest as
precision but ~49× overconfident as absolute accuracy — the precision-vs-accuracy
gap. 655's dense-union match is recall-oriented but mismatch-noisy; 1512 is the
clean accuracy read. Per-fixture detail in `dev-docs/fixtures/{1512,655}.md`.

**Open follow-ups** (not #2): a tighter unambiguous-line matcher to isolate 655's
own accuracy floor; T4 (re-test whether `shape_error_epsilon` still earns its keep
under the SNR-aware metric — the evidence suggests it retires); and per-fixture
κ/F if a non-vinyl-cyanide instrument lands.
