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

> **Resolution: this item was redesigned, not implemented as stated.** Two
> prototype rounds falsified the framing on 2638 (probes
> `scratch/probe_freeze_w223.py`, `probe_skirt_relaxation.py`,
> `probe_baseline_trigger.py`):
> - **Freeze guard inert.** w223's contributor is *tightly* pinned
>   (σ_A/A ≈ 0.001) despite χ²ᵣ=202; fixture-wide, χ²ᵣ and per-peak σ are
>   anti-correlated (high-χ²ᵣ windows are strong lines with tight σ), so a
>   "bad χ²ᵣ AND loose σ" gate never fires and σ is defeated by shape-mismatch
>   overconfidence. The guard also cannot touch w223's own 202 (a core/doublet
>   shape problem) — only the ~11-unit downstream shadow.
> - **Skirt relaxation has no slack.** A *sub-uncertainty* local τ/scale
>   relaxation of the frozen skirt recovers only ~1.4 χ²ᵣ units; the relaxation
>   that actually cleans the wing would inflate the dominant window's χ²ᵣ ~100×
>   (it is a different line).
>
> The downstream residual is a *coherent leakage wing* (systematic positive-Im,
> persisting across windows) from a strong line whose single-shape skirt is
> mismodeled. The adopted fix (per-line-extraction goal, not global
> consistency) is an evidence-triggered **low-order complex baseline** nuisance
> term, triggered by `residual_edge_coherence > ~3.5` (recall 0.89, zero
> harmful fires; 35 beneficial windows fixture-wide — the coarse
> fixed-contributor rule misses the biggest wins w152/w150 which carry no
> contributor). Line σ from the joint covariance keeps the uncertainties fair
> (σ_A inflation ≈ ×1.01). See
> [`../../planning/stage5-leakage-wing-baseline.md`](../../planning/stage5-leakage-wing-baseline.md).

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

The baseline refit is qualitatively consistent with the shipped chi2r
(same ordering, same dominant-window structure). The absolute values
differ slightly from the shipped fit because the harness measures noise
via `estimate_noise_scatter` (the current Stage 2 default) while the
fixture was built against `estimate_noise_adaptive`; shipped values for
reference: w308 = 93.0, w368 = 40.3.

| window | dom SNR | baseline chi2r / rel% | per_peak_tau chi2r / rel% | voigt chi2r / rel% | voigt tau_L |
|---|---:|---|---|---|---:|
| w236 | 183 | 10.65 / 7.39 | **9.00 / 7.63** | 10.76 / 7.56 | 500 (railed) |
| w308 | 401 | 69.77 / 16.24 | **41.12 / 14.33** | 41.74 / 14.37 | 4.9 |
| w318 | 248 | 12.96 / 8.00 | 12.91 / 7.63 | 13.07 / 8.06 | 500 (railed) |
| w368 | 353 | 33.68 / 11.00 | **29.17 / 9.31** | 33.94 / 11.02 | 500 (railed) |
| w360 (ctrl) | 58 | 3.28 / 5.16 | 3.29 / 4.73 | 3.29 / 5.33 | 500 (railed) |

(`rel%` = max `|residual| / |X|` within 3 FWHM of the dominant line.)

### Verdict: per-peak tau, not Voigt

**The residual is not Voigt-shaped.** In 4 of 5 windows the Voigt
optimizer drove `tau_L` to its no-wing upper bound (500 us = pure
Gaussian), i.e. it *declined* the Lorentzian-wing freedom. Only the
tightly-blended w308 pulled a real wing (`tau_L = 4.8 us`), and there it
merely *tied* `per_peak_tau` (chi2r 41.7 vs 41.1) -- the Voigt freedom
reproduces what per-peak tau does more cheaply. Voigt never beats
per_peak_tau in any probed window.

**Per-peak tau is the lever the residual responds to**, where it
responds at all: per_peak_tau is the best or tied-best on both chi2r and
relative residual in every window, and -- importantly -- it *improves*
the w360 control's relative residual (5.16 -> 4.73 %) rather than
perturbing it, while Voigt slightly perturbs it (5.33 %). So *if* item 2
escalates, the target would be the **smaller per-peak-tau refactor** (tau
moves from window-level to optionally per-peak in
`_pack`/`_unpack`/`model_jacobian`), **not** a new Voigt `PeakShape`. The
decision on whether to escalate at all is below.

**But per-peak tau does not reach the noise floor either.** It knocks
~10-40 % off chi2r and leaves a 7-14 % relative-residual wall on the
worst windows. The wall does not respond to Voigt, so it is *not*
single-line functional-form mismatch -- on the blended windows (w308,
w318) it is residual blend / contributor structure. w318 in particular
moves under *no* lever (all three ~13.0), so it is not a
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
| w236 | 183 | 7.19 | 6.65 | 7.46 | +0.81 | -1.7 |
| w308 | 401 | 5.89 | 8.35 | 4.79 | -3.56 | -28.6 |
| w318 | 248 | 5.86 | 5.79 | 5.94 | +0.15 | -0.05 |
| w368 | 353 | 5.06 | 4.53 | 5.54 | +1.01 | -4.5 |
| w360 (ctrl) | 58 | 6.13 | 6.12 | 6.28 | +0.16 | +0.01 |

The split tracks the chi2r movement exactly: where per-peak tau helped
(w236, w368, w308) the two tau's separated; where it didn't (w318, the
w360 control) they stayed together (< 0.16 us apart) and chi2r barely
moved -- the optimizer self-selects. The shared baseline tau is a forced
compromise dragged toward the strong line: in w308 freeing the dominant
lets the weak lines relax *up* to 8.35 us (narrower) while the dominant
drops to 4.79 us (broader), straddling the baseline 5.89 -- and the weak
tau moves as much as the dominant, so the baseline was mis-fitting the
whole window to accommodate the strong line, not just the strong line
itself. w236 confirms the user's "tau higher than tau_G_maj" note
(`tau_d = 7.46` vs majority 6.41).

Caveat reinforcing the isolated-line gate: in the blended w308 the
dominant pulled *down* (broader) -- per-peak tau partly broadened the
strong line to soak up its unresolved partner (the over-broadening basin
the lambda=50 tau penalty guards against; it was active and still
allowed 4.79). On a blend, per-peak tau can mis-attribute blend
structure to the strong line's width rather than fix a real decay
mismatch -- another reason the escalation gate should fire on *isolated*
strong lines.

### Decision: park shape escalation (item 2)

The fork resolves to per-peak tau over Voigt *if* escalating -- but the
case for escalating at all is weak on this fixture, so item 2 is parked:

- The chi2r gains are small outside the confounded blend (w236 -1.7,
  w368 -4.5), and the relative-residual wall barely moves.
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

**The classifier's spur test is two branches.** A bin is `spur` when the
exp fit's tau saturates at tau_max (`spur_by_tau`, slope <= 0 = no decay
detected) **or** the 1-param constant model beats the 2-param exp by
AICc > 2 (`spur_by_aicc`). On 2638 the frame-by-frame evolution shows
these catch *different* populations:

* **True CW spurs are flat** (frame magnitudes ~0.93-1.00 across all 10
  frames) -> tau rails to tau_max -> `spur_by_tau`. Clean and reliable.
  No real line trips this: real tau <= 9 us << tau_max = 100 us, so a
  genuine line always shows decay (38861: 1.00 -> 0.54 -> 0.12, tau 3.7,
  exp wins, not a spur).
* **`spur_by_aicc` also fires on erratic, non-exponential bins** -- e.g.
  33421.1 MHz (frames 0.15 1.00 0.53 0.43 0.60 0.06 ...) and 38744.2 MHz
  (0.24 1.00 0.18 0.53 ...): strong, non-monotonic, peaking in frame 2,
  consistent with a damped beat (unresolved blend / interference).
  Neither flat nor decaying -- the exp fits badly so the constant wins by
  a thin margin (delta-AICc ~6). These are **not** spurs (and not long-tau
  lines).

**So the catalogue cannot be masked directly** -- its `cls == 1` set
mixes true flat spurs with these erratic beat/blend bins. Two things
guard against masking them: they are **non-integer-MHz** (33421.1,
38744.2 are off by > 1 bin), which the integer gate rejects, and the
**persistence half of the gate should key on flatness/saturation**
(`spur_by_tau` / a slope-equivalence test), not the raw `cls == 1`. A
positive flatness criterion catches the flat spurs without firing on the
erratic bins (which have large frame-to-frame spread) or clean decays.

### Synthesis: a joint gate

The two detectors are complementary in principle (the STFT misses
29440/28460/39820 that the frequency test catches -- 29440 is flat +
saturated but `discard`, STFT SNR 4.2 < t_sigma 5; in principle a split-bin
spur the frequency test misses could rail τ to saturation and be caught by
the STFT, though on 2638 the split-bin spurs 39830/39930 are flagged by
neither — see the production check below). A true CW spur is **flat in
time** (STFT saturation) **and** at integer-MHz **and** sub-resolution
narrow; the erratic beat/blend false positives are neither flat nor
integer-MHz. So the design promotes the existing Stage 2b catalogue and
consumes it through a **joint integer-MHz ∧ flatness gate** -- using the
`spur_by_tau` / slope-equivalence (flatness) signal rather than the raw
`cls == 1` set, with integer-MHz as the backstop against the erratic
non-integer bins. Remediation is a cluster mask over the gated spur's
bins, used for both peak-nomination exclusion and the χ²/residual sum;
spur-only windows can be dropped pre-Stage-5. The residual mask is still
required for spurs that share a window with real lines (w245, w287).

Design and task breakdown:
[`../../planning/stage5-spur-masking.md`](../../planning/stage5-spur-masking.md).

### Flatness-exposure measurement settles the persistence half

The production-wiring step measured, on the 2638 fixture, whether the
persisted `cls == 1` catalogue can be consumed directly under the integer
gate, or whether the flatness (`spur_by_tau`) signal must be exposed.
The integer gate alone does **not** clean the catalogue:

| set | count |
|---|---:|
| `cls == 1` in-trim bins | 649 |
| `cls == 1` ∧ integer-MHz | 59 |
| ... of those, `spur_by_tau` (flat) | 6 |
| ... of those, `spur_by_aicc`-only (erratic, **non-flat**) | 53 |
| in-trim `spur_by_tau` (flat) bins, any freq | 70 (6 integer + 64 skirts) |

So **53 of the 59 integer-MHz `cls == 1` bins are erratic `spur_by_aicc`
false positives that the integer gate does not reject** -- e.g. 38744.03
MHz at SNR 260 (frames `0.07 1.00 0.09 0.54 ...`, τ 3.4), a strong
*molecular* line near an integer MHz. Masking those would be the exact
real-line false-positive the acceptance gate forbids. The 6 integer-MHz
`spur_by_tau` bins are all genuine clock harmonics (30720, 32960,
35840 + skirts, 39040), dead-flat across the 10 frames (τ railed to
τ_max).

**Decision: expose the flatness signal.** `integer-MHz ∧ cls == 1` is
*not* clean; the persistence half of the gate keys on `spur_by_tau`
(saturation), persisted as a per-cluster `saturated` flag on
`SpurCluster`. The frequency-domain narrowness detector remains the
zero-false-positive primary (it independently catches 34560/29440 that
are flat-but-not-saturated or below `t_sigma`). The flat catalogue's role
is **corroborative, not additive** on 2638: its 4 saturated clusters
(30720/32960/35840/39040) are a strict *subset* of the narrow detections
(production check below), so the saturated half upgrades those four to
`narrow+saturated` provenance but gates no spur the narrowness test
missed. The split-bin spurs 39830/39930 are caught by **neither** detector
here — their split energy fails the narrowness ratio and is too weak/erratic
in the STFT to rail τ to saturation, so they never enter the `spur_by_tau`
set (consistent with the flatness table above, which enumerates exactly the
6 saturated bins and excludes them). The mechanism — a saturated catalogue
*can* gate a split-bin spur the narrowness test misses — holds on synthetic
input (`test_gate_union_of_narrow_and_saturated`); it just does not fire on
this fixture. The joint integer-MHz ∧ (narrow ∨ saturated) gate stands as a
safe union: on 2638 the union equals the narrow set.

### Production wiring & validation (2638, Gaussian)

The masking ships in `fitting/spur_detection.py` (detector + joint gate +
`SpurSet`), consumed by `fit_window` (per-bin residual/Jacobian/χ² mask,
`n_data` reduced) and `plan_execution.execute_plan` (per-window mask +
candidate-nomination exclusion + spur-contributor drop), built once in
`stage5_impl` from the active-FT + the persisted Stage 2b `saturated`
catalogue (auto-detected). Controlled by the `StageFitSettings.spur`
sub-block (default on, ±2-bin mask, integer-MHz band = the Stage 1 trim).

Re-running Stage 5 on the audit fixture with masking **off** vs **on**
(same Stage 3 peaks + Stage 4 plan), the frequency-domain gate flagged
exactly the prototype's 8 in-band integer-MHz narrow spurs (28460, 29440,
30720, 32960, 34560, 35840, 39040, 39820). Per-window χ²ᵣ (off → on):

| window | off | on | Δχ²ᵣ | npeaks |
|---|---:|---:|---:|---|
| w287 | 58.50 | 1.44 | 57.1 | 1→1 |
| w193 | 26.12 | 1.56 | 24.6 | 1→0 |
| w126 | 13.59 | 1.14 | 12.4 | 1→0 |
| w245 (spur + real) | 13.83 | 4.08 | 9.8 | 4→3 |
| w88 | 5.79 | 0.98 | 4.8 | 1→0 |
| w372 | 4.80 | 1.99 | 2.8 | 1→0 |

**sum Δχ²ᵣ over the classified spur windows = 111.5**, at or above the
prototype's ±2-bin figure (~98–103). w245 keeps its 3 real lines and
floors at χ²ᵣ ≈ 4.1 (the real-line "other" content, correctly not
over-recovered — it dropped only the spurious peak the fitter had placed
on the spur). Globally 632 → 625 fitted peaks: the 7 removed peaks were
all spurious peaks sitting on gated narrow integer-MHz spurs (spur-only
windows w88/126/193/372/386 plus w62, an unclassified spur the gate
caught); **no real molecular line was removed** (the broad integer-MHz
lines lack the narrowness signature and are never gated). The fixture's
persisted Stage 2b catalogue predates the `saturated` flag, so this run
exercised the frequency-domain detector alone.

#### Saturated-catalogue path exercised end-to-end (2638, Gaussian)

The flat-catalogue half was then run in production. On a scratch copy,
`calibrate_tau` + `calibrate_tau_G` were re-run (populating the per-cluster
`saturated` flag), Stages 3–5 re-run, and the spur-on fit compared against
both spur-off and the frequency-domain-only run
(`scratch/validate_spur_saturated.py`). Findings:

- **`saturated` populates as the flatness table predicts.** The persisted
  `/stage2b_tau_G_calibration/spur_clusters/saturated` carries 4 saturated
  clusters — 30720, 32960, 35840 (+ skirts grouped into the cluster), 39040
  — exactly the 6 `spur_by_tau` bins of the flatness-exposure measurement.
  The Lorentzian twin's catalogue agrees.
- **The gate activates but adds nothing on this fixture.** The gated set is
  the *same 8 spurs* as the frequency-domain-only run (28460, 29440, 30720,
  32960, 34560, 35840, 39040, 39820); the saturated half upgrades 4 of them
  to `source="narrow+saturated"` and leaves 28460/29440/34560/39820 as
  `narrow` (flat-but-not-saturated or sub-`t_sigma`). No `saturated`-only
  spur appears: the saturated set is a strict subset of the narrow set.
- **Split-bin spurs 39830/39930 do not appear** in either catalogue (no
  cluster within 0.6 MHz) and are gated by neither detector — correcting the
  earlier expectation that the flat catalogue would rescue them (detail in
  the next subsection).
- **Zero regression.** χ²ᵣ recovery is identical (sum Δχ²ᵣ = 111.5 vs
  spur-off, same as the frequency-domain-only run); window-id sets match
  across all three arms; no non-spur window loses a peak vs the
  frequency-domain run; w245 keeps its 3 real lines; total fitted peaks
  625 = 625.

So the saturated path is verified correct and safe end-to-end; on 2638 it
is redundant with the narrowness detector rather than additive. It earns
its place as a backstop for instruments/fixtures where a split-bin clock
harmonic *does* rail τ to saturation while failing the narrowness ratio —
not demonstrated on this dataset.

#### Why 39830 / 39930 fall through both detectors (and what happens to them)

Diagnosed bin-by-bin (`scratch/diag_39830_39930.py`). Both misses are for
honest threshold reasons, and the two thresholds fail on the *same* feature
from opposite directions — a **weak tone whose energy splits across two
active-FT bins straddling the integer**.

*Frequency-domain (narrowness) miss — a half-bin grid-aliasing effect.* The
active-FT bin spacing is 79.05 kHz, so 10 MHz = 126.5 bins: consecutive
×10-MHz clock harmonics drift half a bin against the FFT grid. 39820 lands a
bin 1.8 kHz off the integer (all energy in one bin → ratio 0.22, SNR 6.1 →
**gated**); its neighbours 39810/39830/39930 land ~38 kHz off (half a bin),
so the tone splits across two bins. The detector keys on the bin nearest the
integer and tests `max(neighbour)/peak ≤ 0.30`, but the split puts the
*larger* lobe in the off-integer neighbour — ratio 1.41 (39830), 1.02
(39930), 1.14 (39810), all ≫ 0.30. A split-bin spur reads as a *broad*
feature to the narrowness test; 39930 additionally drops its nearest-integer
bin to SNR 4.2 < 5. Whether a clock harmonic gates is thus partly an accident
of grid alignment.

*STFT (saturated) miss — sub-threshold per-frame SNR.* Every bin near
39810/39830/39930 classifies `cls = 0` (discard), **not** because it looks
like a decaying line — several are genuinely flat (τ railed to τ_max,
`saturated=True`, e.g. 39930.04 and all of 39810) — but because the per-frame
SNR is 1.8–4.2, below `t_sigma = 5`. The 10-frame STFT puts ⅒ the energy in
each frame (larger `sigma_frame`), so `is_spur = (SNR ≥ t_sigma) ∧
(saturated ∨ aicc)` fails its first clause. The flatness is real but never
recorded as a `SpurCluster` (clusters are built only from `cls == 1` bins).
So the full-record FT sees enough SNR but the split kills narrowness; the
STFT sees the flatness but not enough per-frame SNR. The split-bin spurs sit
exactly in the gap.

*What the pipeline does with them.* They are **not** masked: Stage 3 detects
them as `WEAK` peaks (39830.00 @ SNR 7.6, 39930.00 @ SNR 4.7, 39810.00 @ SNR
6.6) and Stage 5 fits them as ordinary lines (w387, w390, w385 respectively).
Because they are *weak* tones, a finite-T line shape fits a 2-bin feature
acceptably — those windows floor at χ²ᵣ ≈ 1.3–1.8 (cf. the strong harmonic
w287 at 58.5 pre-mask), so they were never in the ~106-unit spur bucket and
missing them costs **~0 χ²ᵣ**. The cost is **line-list pollution**: three
spurious weak lines (39810/39830/39930) enter the catalogue as if molecular.
See the follow-up below.
