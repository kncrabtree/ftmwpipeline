# Stage 5 Gaussian-path parameter optimization audit -- progress

Step 1 (parameter sweep) and Step 2 (leader selection) report.
Step 4 (per-window classification) is the user's out-of-session
task. Step 5 (cross-reference) updates this report once the
classifications land.

## Step 2 -- per-(knob, shape) leaders

Selection heuristic: lowest p95 chi2r subject to median chi2r
<= baseline_median * 1.05.

| knob | shape | leader value | median chi2r (Delta) | p95 chi2r (Delta) | n>5 | n>10 |
|---|---|---|---|---|---:|---:|
| `conservative.weak_window_snr_threshold` | gaussian | 5.0 | 1.385 (+0.005) | 6.819 (+0.000) | 29 | 12 |
| `conservative.weak_window_snr_threshold` | lorentzian | 5.0 | 1.363 (+0.000) | 5.723 (+0.000) | 23 | 13 |
| `rescue.prominence_threshold` | gaussian | 1.5 | 1.380 (+0.000) | 6.181 (-0.638) | 28 | 12 |
| `rescue.prominence_threshold` | lorentzian | 1.5 | 1.369 (+0.006) | 5.631 (-0.092) | 23 | 13 |
| `rescue.snr_threshold` | gaussian | 1.5 | 1.380 (+0.000) | 6.819 (+0.000) | 29 | 12 |
| `rescue.snr_threshold` | lorentzian | 3.5 | 1.360 (-0.003) | 5.635 (-0.088) | 23 | 12 |
| `tau.fit_tau_min_snr` | gaussian | 25.0 | 1.380 (+0.000) | 6.819 (+0.000) | 29 | 12 |
| `tau.fit_tau_min_snr` | lorentzian | 25.0 | 1.363 (+0.000) | 5.723 (+0.000) | 23 | 13 |
| `thaw.residual_edge_threshold` | gaussian | 5.0 | 1.380 (+0.000) | 6.819 (+0.000) | 29 | 12 |
| `thaw.residual_edge_threshold` | lorentzian | 5.0 | 1.363 (+0.000) | 5.723 (+0.000) | 23 | 13 |

## Step 3 -- leading variant for per-window walkthrough

**Chosen:** `rescue_prominence_threshold__1p5__gaussian`

- knob: `rescue.prominence_threshold`
- value: `1.5`
- shape: `gaussian`
- chi2r median / p95 / max: 1.380 / 6.181 / 202.025
- n_windows > 5: 28; > 10: 12
- rationale: Largest p95 chi2r improvement: 0.638 (rescue.prominence_threshold=1.5, shape=gaussian).

Fixture: `scratch/stage5-gaussian-audit/runs/rescue_prominence_threshold__1p5__gaussian.ftmw`

Per-window artifacts emitted by
`scripts/development/stage5-validation/generate_validation.py` with `--all-windows --variant-id <vid>`.

## Step 4 -- user classification (out-of-session)

Open the emitted
`scratch/stage5-validation-rescue_prominence_threshold__1p5__gaussian/windows.toml`
and fill `classification = "..."` only for windows that look
wrong. Leave the rest empty (the convention keeps the typing
budget small on the full 386-391 windows).

## Step 5 -- cross-reference and failure-mode analysis

Classifications landed in
`scratch/stage5-validation-rescue_prominence_threshold__1p5__gaussian/windows.toml`.
This section reads them against the Step 1 sweep.

### Classification tally

391 windows: 287 clean (73 %), 104 flagged. Ranked by total excess
chi2r (`sum(chi2r - 1)` over the bucket -- a proxy for where the chi2
mass concentrates):

| failure mode | n | median chi2r | max | sum excess chi2r |
|---|---:|---:|---:|---:|
| `shape_error` | 30 | 3.4 | 93.0 | **240.9** |
| `other` | 12 | 3.0 | 202.0 | 233.3 (w223 alone = 202) |
| `missed_peak` | 29 | 1.6 | 63.3 | 119.2 |
| `spur` | 10 | 3.3 | 58.5 | **105.5** |
| `baseline_offset` | 6 | 2.9 | 8.4 | 16.3 |
| `overfit` | 11 | 1.4 | 6.7 | 15.1 |
| `contributor_error` | 2 | 3.2 | 3.7 | 4.4 |
| `replan_needed` | 3 | 1.9 | 2.6 | 2.7 |

Top single offenders: w223 (202.0, doublet phase/shape/missing), w308
(93.0, tightly blended strong doublet), w151 (63.3, large missed peak +
strong doublet shape), w287 (58.5, spur), w144 (43.3, missed peak in a
9-line window), w368 (40.3, shape).

### Key finding: the high-chi2r tail is structural, not knob-tunable

The Step 2 leader table shows median chi2r flat across every knob and
only `rescue.prominence_threshold = 1.5` moving p95 at all (-0.638 on
gaussian). No knob shifts the n>5 (28) or n>10 (12) counts materially.
This is consistent with the Stage 3 tau-feeder and Stage 4 `leakage.tau_us`
audits, which reached the same conclusion from the other side: the
windows with chi2r > 5 carry an identical contributor set across knob
variants. **The remaining chi2 budget lives in the line-shape model, in
unmasked clock spurs, and in frozen-contributor coupling -- none of which
the five Y-rated knobs reach.** The defaults stand; the next iteration is
structural work, not a re-sweep.

### Why strong lines dominate chi2 (the `shape_error` bucket)

chi2 weights every bin by `1 / sigma^2`, but line-shape model error
scales with amplitude `A`, not with `sigma`. A strong line at
`A / sigma ~ 100` with a 10 % shape mismatch produces ~10-sigma per-bin
residuals across its core, i.e. ~100 chi2 units per bin. So a visually
excellent fit of a strong line reads as chi2r = 8-90 (w236, w308, w318,
w368, w360) while the residual is only ~1-5 % of `|X|`. This is the
single largest bucket (241 chi2 units) and it is intrinsic to the
weighting -- it cannot be tuned away.

The mechanism is architectural. In `fitting/window_fit.py`, tau is one
shared parameter for the entire window (`peak_model.ModelPeak` omits tau;
`model_spectrum` takes a single `tau_us`), pinned to `tau_maj +/- 5
sigma_tau` and bidirectionally penalized toward `tau_maj`
(`derive_window_fit_constraints`, lambda = 50). The strongest lines
genuinely carry a different effective decay than the window majority
(user note on w236: "tau higher than tau_G_maj"), and one penalized
shared tau cannot represent that.

### Recommended structural work (priority order)

**1. Spur detector + chi2/residual mask (cheapest, ~106 chi2 units,
near-zero regression risk).** There is no spur handling in Stage 5 today;
"spur" in `src/` refers only to Stage 2b's `spur_cluster_multiplier`,
which masks the tau-calibration STFT, never the fit. Clock spurs flow
into the fitter, and a single-bin delta cannot be represented by any
finite-T line shape, so they detonate chi2 (w287 = 58, w193 = 26,
w126 = 13). The user notes give a reliable fingerprint: **exact
integer-MHz center, a single non-zero bin, energy in only one quadrature
(real-only or imag-only).** A pre-fit classifier should (a) exclude spur
bins from peak nomination and (b) mask them out of the chi2/residual sum
-- the w245 note ("excluded from the fit *and* residual/chi2
calculation") is the load-bearing part: windows where the spur is
correctly not fitted still carry it in chi2. w074 / w389 show some spurs
already drop out by prominence luck; a deliberate detector makes it
reliable. Operationalizes the `spur` classification value added to the
TOML convention.

**2. chi2r-triggered, residual-localized shape escalation for strong
lines (attacks the 241-unit `shape_error` bucket).** The strongest lines
are the only ones with enough SNR to discriminate line shape cleanly, so
escalate *selectively* rather than globally:

   - **Trigger:** flag a window whose `chi2r` exceeds a threshold.
   - **Localize:** within a flagged window, find where `|X|` (or
     `|residual|`) is large and check whether it sits near an
     already-fitted line.
   - **Gate:** escalate only if that line's SNR clears a threshold high
     enough that the extra shape freedom is identifiable (tie to
     `tau.fit_tau_min_snr`, default 50).
   - **Escalate to one of:**
     - *per-peak tau* -- the line carries its own decay constant (still
       inside the calibrated band, still penalized toward `tau_maj`),
       while weak lines keep sharing the window tau. Smaller change:
       reuses the entire existing model; tau moves from window-level to
       optionally per-peak in `_pack` / `_unpack` / `model_jacobian`.
     - *Voigt shape* -- a Gaussian-anchored `tau_G` plus a small free
       Lorentzian component. `PeakShape` is only LORENTZIAN / GAUSSIAN
       today; several worst windows are strong lines neither pure shape
       fits. Larger lift (new `h_T` + Jacobian, reusing the existing
       `wofz` path), but directly captures the "~5-10 % on strong lines"
       residual.

   Worth prototyping per-peak tau first (smaller blast radius), then
   testing Voigt on the windows per-peak tau does not resolve. The
   escalation is the right shape because escalating globally would
   destabilize the weak-window fits that currently work.

**3. Freeze-eligibility guard for frozen contributors (fixes w223
chain + `contributor_error`).** w223 (chi2r = 202, the worst window) is a
poorly-fit strong/blended window whose contributor is then frozen and
propagated into w224-w227, each inheriting "poor treatment of window 223
baseline." A contributor is frozen on the strength of its *position*, not
the *quality* of its fit. A guard -- do not freeze a contributor whose
own post-fit chi2r or parameter uncertainty is bad; re-elevate it to free
in the dependent window -- stops one bad window from poisoning a
four-window chain. Same mechanism behind w309 and w367
(`contributor_error`). This is the O4-2 / ROADMAP thread #6 + #11 gap.

### Linkage: `overfit` is mostly a symptom of `shape_error`

The 11 `overfit` windows carry only 15 chi2 units and median chi2r 1.4.
A strong line's shape residual is large relative to noise, so the
AICc-with-`n_eff` gate (`window_fit.py`) correctly sees a significant
chi2 improvement when it adds a 10:1-ratio peak to absorb that residual
(w012, w022, w040, w091, w281). The extra peak is fitting shape mismatch,
not noise. Fixing the shape model should retire most of these without a
dedicated amplitude-ratio guard -- and a guard would risk killing real
weak blend members (sentinels below). Re-check the `overfit` count
*after* the shape change rather than adding a guard pre-emptively.

### What not to chase

`missed_peak` is 119 chi2 units but median chi2r only 1.6, and in nearly
every flagged case the missing line is weak and arguably should not be
forced (user assessment). The sweep confirms lowering
`rescue.snr_threshold` / `prominence_threshold` buys almost nothing in
aggregate. Leave the rescue defaults; route the handful of "definitely
one more fittable line" cases (w144, w151, w295) through the manual-seed
escape hatch (sketched in the next-session prompt, separately tracked)
rather than lowering a global floor that re-admits noise everywhere.

### Regression sentinels

windows.toml marks these as good fits to monitor when shape / tau / spur
changes land -- they guard against trading strong-line accuracy for lost
weak-blend members: w020, w075, w159, w189, w303 (and w218, w360 as
"large chi2r but excellent fit" cases that a shape change should
*improve*, not perturb).

### Acceptance for this analysis step

- Classifications cross-referenced against the Step 1/2 sweep: confirmed
  the high-chi2r tail is structural, not knob-reachable.
- Three structural work items identified and prioritized (spur mask;
  chi2r-triggered shape escalation; freeze-eligibility guard); each gets
  its own planning entry before code moves.
- This validation run is the baseline; future structural changes diff
  per-window chi2r against
  `scratch/stage5-validation-rescue_prominence_threshold__1p5__gaussian/`.

## Shape-escalation diagnostic -- per-peak tau or Voigt? (item 2 fork)

`probe_shape_escalation.py` settles the escalation-target fork for
structural work item 2 *before* any production code moves. For each of
the worst strong-line `shape_error` windows (w236, w308, w318, w368)
plus w360 as a "large chi2r but excellent fit" control, it takes the
shipped converged free-peak set as the starting model and runs a
fixed-K joint refit under three model forms, holding the
frozen-contributor background fixed (subtracted once at the persisted
shared tau) so all three variants see identical data and starts -- only
the free-peak model form differs:

* `baseline` -- shared-tau single shape (reproduces the shipped floor),
* `per_peak_tau` -- the dominant (highest-SNR) free line carries its own
  tau (calibrated band, bidirectional-Gaussian penalty toward `tau_maj`),
* `voigt` -- the dominant line refit as a finite-T Voigt (shared `tau_G`
  Gaussian core + its own free Lorentzian `tau_L`), reusing the `wofz`
  path; the prototype `h_T_voigt` is checked to reduce to `h_T_gaussian`
  as `tau_L -> inf`.

The shared / per-peak tau carry the same calibrated bounds + tau penalty
production uses; the phase/amp penalties are dropped (the shipped peaks
are well-separated and sensibly-amplituded, so those penalties are
near-zero and irrelevant to the residual floor under test). Output:
`data/shape_escalation_per_window.csv`.

### Results

The baseline refit reproduces the shipped per-window chi2r (w308
93.0 -> 93.3, w368 40.3 -> 40.5), validating the harness.

| window | dom SNR | baseline chi2r / rel% | per_peak_tau chi2r / rel% | voigt chi2r / rel% | voigt tau_L |
|---|---:|---|---|---|---:|
| w236 | 183 | 12.22 / 7.29 | **10.30 / 7.65** | 12.34 / 7.46 | 500 (railed) |
| w308 | 401 | 93.28 / 16.23 | **54.88 / 14.39** | 55.56 / 14.37 | 4.8 |
| w318 | 248 | 18.10 / 8.02 | 18.03 / 7.65 | 18.26 / 8.07 | 500 (railed) |
| w368 | 353 | 40.50 / 10.98 | **35.05 / 9.30** | 40.80 / 11.00 | 500 (railed) |
| w360 (ctrl) | 58 | 3.91 / 5.21 | 3.93 / 4.72 | 3.93 / 5.39 | 500 (railed) |

(`rel%` = max `|residual| / |X|` within 3 FWHM of the dominant line.)

### Verdict: per-peak tau, not Voigt

**The residual is not Voigt-shaped.** In 4 of 5 windows the Voigt
optimizer drove `tau_L` to its no-wing upper bound (500 us = pure
Gaussian), i.e. it *declined* the Lorentzian-wing freedom. Only the
tightly-blended w308 pulled a real wing (`tau_L = 4.8 us`), and there it
merely *tied* `per_peak_tau` (chi2r 55.6 vs 54.9) -- the Voigt freedom
reproduces what per-peak tau does more cheaply. Voigt never beats
per_peak_tau in any probed window.

**Per-peak tau is the lever the residual responds to**, where it
responds at all: per_peak_tau is the best or tied-best on both chi2r and
relative residual in every window, and -- importantly -- it *improves*
the w360 control's relative residual (5.21 -> 4.72 %) rather than
perturbing it, while Voigt slightly perturbs it (5.39 %). So *if* item 2
escalates, the target would be the **smaller per-peak-tau refactor** (tau
moves from window-level to optionally per-peak in
`_pack`/`_unpack`/`model_jacobian`), **not** a new Voigt `PeakShape`. The
decision on whether to escalate at all is below.

**But per-peak tau does not reach the noise floor either.** It knocks
~5-40 % off chi2r and leaves a 7-14 % relative-residual wall on the
worst windows. The wall does not respond to Voigt, so it is *not*
single-line functional-form mismatch -- on the blended windows (w308,
w318) it is residual blend / contributor structure. w318 in particular
moves under *no* lever (all three ~18.0), so it is not a
shape-escalation target at all; its chi2r belongs to another bucket
(blend / contributor). This sharpens the item-2 escalation gate: trigger
on a high-chi2r *isolated* strong line (SNR-gated, tie to
`tau.fit_tau_min_snr`), give it its own tau, and do **not** expect
escalation to rescue tightly-blended windows -- those route to the
blend / freeze-guard work (items 3 / O4-2), not shape escalation.

### How the tau values move under per-peak tau

Calibrated anchor `tau_maj = 6.41 us` (Gaussian `tau_G_maj`) on every
window. In the per-peak model the weak lines share `tau_w` and the
dominant carries `tau_d`; the baseline is forced onto one shared tau:

| window | dom SNR | baseline tau | tau_w (weak) | tau_d (dom) | tau_d - tau_w | delta chi2r |
|---|---:|---:|---:|---:|---:|---:|
| w236 | 183 | 7.01 | 6.66 | 7.48 | +0.82 | -1.9 |
| w308 | 401 | 5.90 | 8.41 | 4.78 | -3.63 | -38.4 |
| w318 | 248 | 5.80 | 5.79 | 5.94 | +0.15 | -0.07 |
| w368 | 353 | 5.06 | 4.52 | 5.54 | +1.02 | -5.4 |
| w360 (ctrl) | 58 | 6.09 | 6.11 | 6.27 | +0.16 | +0.02 |

The split tracks the chi2r movement exactly: where per-peak tau helped
(w236, w368, w308) the two tau's separated; where it didn't (w318, the
w360 control) they stayed together (< 0.16 us apart) and chi2r barely
moved -- the optimizer self-selects. The shared baseline tau is a forced
compromise dragged toward the strong line: in w308 freeing the dominant
lets the weak lines relax *up* to 8.4 us (narrower) while the dominant
drops to 4.78 us (broader), straddling the baseline 5.90 -- and the weak
tau moves as much as the dominant, so the baseline was mis-fitting the
whole window to accommodate the strong line, not just the strong line
itself. w236 confirms the user's "tau higher than tau_G_maj" note
(`tau_d = 7.48` vs majority 6.41).

Caveat reinforcing the isolated-line gate: in the blended w308 the
dominant pulled *down* (broader) -- per-peak tau partly broadened the
strong line to soak up its unresolved partner (the over-broadening basin
the lambda=50 tau penalty guards against; it was active and still
allowed 4.78). On a blend, per-peak tau can mis-attribute blend
structure to the strong line's width rather than fix a real decay
mismatch -- another reason the escalation gate should fire on *isolated*
strong lines.

### Decision: park shape escalation (item 2)

The fork resolves to per-peak tau over Voigt *if* escalating -- but the
case for escalating at all is weak on this fixture, so item 2 is parked:

- The chi2r gains are small outside the confounded blend (w236 -1.9,
  w368 -5.4), and the relative-residual wall barely moves.
- Fitting two nearby lines with different tau is not physically
  justified without an independent argument for why their decay
  constants differ; at this SNR, with no ground truth, we can't make
  one. The known frequency-dependence of tau is *already* captured
  band-wise by the Stage 2b STFT calibration (tau decreasing with
  frequency across its bands); a per-line tau is a finer effect this
  fixture cannot cleanly support.
- The tau penalty (`lambda = 50`) may be slightly tight -- it can hold
  tau short of its preferred value -- but `lambda = 0` overfits, and
  re-tuning lambda on a single fixture trades one un-grounded knob for
  another.
- The residual per-peak tau leaves is mostly blend / contributor
  structure (w308, w318), which routes to items 1 and 3, not shape
  escalation.

Revisit with a higher-SNR fixture, where the residual can be decomposed
and a power / pressure series can ground the physics (radiation damping,
saturation, self-absorption all predict tau *down* on strong lines; the
isolated-line tau *up* seen here is unexplained by those and by blends).
For now the finding is documented and the shipped shared-tau model
stands -- the pipeline's purpose, frequencies and intensities, is well
served by it. This was a throwaway diagnostic; the per-peak-tau plumbing
and `h_T_voigt` prototype live only in `probe_shape_escalation.py` and do
not move into production.

## Spur-detection prototype (item 1)

`probe_spur_detector.py` validates structural work item 1 -- detect clock
/ LO spurs and mask them from the fit -- before any production wiring.

### Fingerprint (verified, not assumed)

Two robust discriminators, confirmed on the active-FT:

* **exact integer-MHz center** -- every classified spur sits within a
  fraction of a bin of an integer MHz.
* **sub-resolution narrowness** -- a persistent CW tone is
  transform-limited by the full boxcar (first null ~1/T ~ one bin), so
  its peak bin is 10-25x its neighbours, whereas a real finite-T line has
  a coherent leakage skirt where adjacent bins are comparable.

The "energy in only one quadrature" criterion from the earlier item-1
note is **false** -- both probed spurs show comparable Re/Im (the spur's
phase relative to t0 is arbitrary), so it is dropped.

### Frequency-domain detector: catches the strong spurs, zero false positives

Integer-MHz + narrowness on the active-FT |X| flags 8 spur bins (29440,
30720, 32960, 34560, 35840, 39040, plus weaker 28460/39820). The
narrowness gate **spared all 354 real molecular lines** that sit near an
integer MHz, including the SNR-414 line at 38861 (ratio 1.10) -- real
lines are broad, spurs are not. Misses: split-bin spurs (39830/39930,
where the integer falls ~38 kHz *between* bins so energy splits across
two comparable bins) and near-noise spurs (39810).

### Remediation must be a cluster mask, not a single bin

The original "single non-zero bin" framing under-recovers badly. χ²ᵣ vs
mask half-width on the affected windows, and the total sum(chi2r)
reduction:

| mask half-width (bins) | sum(chi2r) recovered |
|---|---:|
| +-0 (single bin) | 22.9 |
| +-1 | 88.1 |
| +-2 | 98.0 |
| +-3 | 102.8 |

A strong CW tone is a full-window sinc whose skirt sits ~8 sigma above
noise for +-2-3 bins (the neighbours look small only because sigma is
tiny). A +-2-3 bin cluster mask recovers essentially the whole ~106
excess-chi2r spur bucket; a single bin recovers a fifth of it. w245
floors at ~5 even fully masked -- it is a spur *plus* real lines
(classified `other`), correctly not over-recovered.

### The temporal-persistence detector already exists in Stage 2b

Stage 2b's tau-calibration STFT classifies every frequency bin by
fitting exponential-vs-constant across the STFT frames (AICc); a bin is
labelled **spur** when the constant (persistent) model wins or tau
saturates at tau_max (CW tone, tau -> inf). It groups adjacent
sinc-skirt bins into a `SpurCluster` (`center_freq_mhz`, `bin_indices`)
and **persists the catalogue** at `/stage2b_*/spur_clusters`. This is the
temporal-persistence test in a more rigorous form than a first/last
ratio, and the cluster `bin_indices` give a *data-driven* mask extent.

**But the persisted catalogue cannot be used directly for masking.** It
carries 135 clusters and the constant/saturation criterion conflates
true CW spurs with **long-tau strong real lines** -- e.g. 33421.1 MHz
(real, SNR 187) and 38744.2 MHz (real, SNR 121) are in it. Masking those
would delete signal.

### Synthesis: a joint gate

The two detectors are complementary, neither a superset (the STFT misses
29440/28460/39820 that the frequency test catches; the frequency test
misses the split-bin spurs the STFT catches). A true CW spur is
persistent (STFT) **and** at integer-MHz **and** sub-resolution-narrow;
a long-tau real line is persistent but neither integer-MHz nor narrow. So
the design promotes the existing Stage 2b catalogue and consumes it
through a **joint integer-MHz ∧ persistence gate** -- integer-MHz is the
guard against the long-tau-line false positive, persistence rescues the
split-bin case. Remediation is a cluster mask over the gated spur's bins,
used for both peak-nomination exclusion and the χ²/residual sum;
spur-only windows can be dropped pre-Stage-5. The residual mask is still
required for spurs that share a window with real lines (w245, w287).

Design and task breakdown:
[`../../planning/stage5-spur-masking.md`](../../planning/stage5-spur-masking.md).
