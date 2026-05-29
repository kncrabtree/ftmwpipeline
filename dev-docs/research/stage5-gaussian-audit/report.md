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
