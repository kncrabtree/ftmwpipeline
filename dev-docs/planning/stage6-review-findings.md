# Stage 6 review findings — triage backlog

Status: **review pass complete; sequenced, implementation not started.** A list
of issues surfaced while a human reviewed fitted `.ftmw` files through the
Stage 6 `review` surface. Each entry is an observation with enough evidence to
act on. The agreed order of attack lives in the ROADMAP Priorities (Stage 6
completion): **(1)** persist the per-window parameter covariance (F4 plumbing),
**(2)** devise variance–covariance peak-survival metrics and iterate (F4 + F5),
**(3)** address window construction (F2 + F3), **(4)** reconsider attention
metrics (F1) — middle steps expected to iterate. Once an item is scheduled it
graduates to its own planning doc (or a section of the relevant stage doc) and
is struck from here.

Reference fixture for every entry below: the rebuilt
`scratch/stage6-drive/exp_2638_review.ftmw` (2638, fit through Stage 5,
`review run` done — 565 windows, 76 flagged for attention).

## F1 — Attention flagging is low-precision

**Observation.** A sample of flagged windows showed roughly one in five needed
any edit. The attention queue is 76 of 565 windows; ranked by each window's
top reason:

| count | top reason | nature |
|---|---|---|
| 33 | `doublet_eps_gt_kappa` | observation, usually *correct as fit* |
| 22 | `candidate_bearing` | a revivable candidate above the attention bar |
| 10 | `edge_boundary` | genuinely actionable (see F2) |
| 9 | `spur_adjacent` | line abutting a masked spur |
| 2 | `worst_eps` | the globally worst doublet ε |

**Reads.**

- `doublet_eps_gt_kappa` dominates (43%) and is the weakest *attention*
  justification: ε > κ is the calibrated "doublet required" verdict — the
  automatic fit is probably right to keep the pair — so flagging every such
  window asks the human to re-confirm decisions the pipeline already made. Most
  of the "needed no change" windows are this kind. Candidate fix: stop treating
  `doublet_eps_gt_kappa` as an attention trigger by default (keep it visible in
  `review show --window N`), or gate it far harder than the current
  `attention_candidate_evidence`.
- No reason kind targets a **weak / implausible fitted peak**. The one sampled
  window that *did* need an edit carried an SNR ≈ 2 fitted peak (a likely
  overfit — rescue dust or a split product); the attention machinery has no
  detector for that. Candidate fix: add a low-SNR / low-evidence fitted-peak
  attention reason.
- Not every reason is too loose: `edge_boundary` correctly flagged the F2
  window. The precision problem is specific to the doublet kind, not the whole
  routing.

Routing + reason kinds are defined in
[`stage6-finalization.md`](stage6-finalization.md) and the doublet calibration
in [`stage5-doublet-alternative.md`](stage5-doublet-alternative.md).

## F2 — Window boundaries split inside sub-minimum gaps, leaving features off-centre

**Observation.** Windows 309 `[33720.094, 33723.865]` and 310
`[33723.943, 33728.028]` split a single physical cluster across their shared
boundary at ~33723.9 MHz. Window 309 holds four lines jammed against its high
edge (`d_hi` 0.02–0.32 MHz) with the rest of the window empty; window 310 is
the mirror — five lines (incl. an SNR ≈ 170 line) against its low edge. The
real feature is ~9 fitted lines spanning 33723.5–33724.6 MHz (**< 1.1 MHz
total**).

**How it arises (ordering artifact).**

1. Stage 3 under-detected the cluster — four peaks only, largest gap **0.39
   MHz** at 33723.9 (`33723.47 → 33723.71 → [0.39] → 33724.10 → 33724.41`).
2. Stage 4 placed a boundary in that gap. On the four detections alone, 0.39
   MHz reads as a cluster separation.
3. Stage 5 residual-rescue then found the missing lines, *filling* the gap — so
   the boundary now bisects a contiguous forest, features piled on both inner
   edges.
4. Neither window froze the other's strong lines as leakage contributors (0
   each), so each half fits on the other half's unmodeled leakage skirt — 310
   sits at χ²ᵣ 3.01, 309 shows edge residual structure.

**Normative points (from review).**

- The minimum window size is ~4 MHz (`min_window_half_width_mhz = 2.0`, a
  resolution-element-based floor). A **0.39 MHz gap is not an appropriate split
  boundary** — it is far below the minimum window width.
- A cluster spanning **< 1 MHz fits inside one minimum-size window** and should
  **not** be split. Splitting is appropriate only for a genuinely dense forest
  that exceeds the window size budget (e.g. fixture 363), not for a sub-minimum
  Stage-3 gap.
- A window should not be constructed with its main feature(s) **off-centre**
  unless it is truly mid-forest. An edge-piled main feature is the symptom of a
  bad split. Window 309 came out 3.77 MHz wide — below the nominal 4 MHz
  minimum — but that minimum is **not actually enforced** (see F3); there is no
  floor on final window width, which is part of why the split products are
  sub-minimum and off-centre.

**Candidate fixes (to prioritize later), likely a combination:**

- **Window selection (Stage 4):** do not split when the cluster fits within one
  minimum-size (~4 MHz) window; enforce the `min_window_half_width_mhz` floor on
  *split products*, not only on isolated-line windows; prefer boundaries in gaps
  that are large relative to the window minimum and that leave features centred.
- **Structural replan / merge (Stage 5):** the replan net that merges straddled
  windows (it accepted replans elsewhere on this file) did not fire here —
  investigate why post-rescue lines piled on both sides of a shared edge did not
  trigger a merge. This is the data-driven fix: let the post-fit line set, not
  the sparse Stage-3 detections, decide window extent.
- **Review-layer gap:** there is no window-level merge verb (`review merge` is
  peak-level only), so a reviewer who spots this cannot fix it without re-running
  Stage 4/5. A "merge windows + joint refit" capability may be warranted.

Window planning is [`stage4-window-assignment.md`](stage4-window-assignment.md);
leakage-contributor freezing is
[`stage4-leakage-contributor-subtraction.md`](stage4-leakage-contributor-subtraction.md).

## F3 — `min_window_half_width_mhz` is inert; no enforced minimum window width

**Observation.** The Stage 4 setting `clustering.min_window_half_width_mhz`
(default 2.0 MHz, documented as "the minimum half-width of a window built around
an isolated weak line") is read at exactly one site —
`preprocessing/window_planning.py:475`:

```python
half_idx = max(int(round(min_window_half_width_mhz / step_mhz)), edge_m)
```

On the reference 2638 grid (`step_mhz = 0.0786`) the default 2.0 MHz is **25
bins**, and it is `max()`-ed against `edge_m` (default **64 bins**), so `edge_m`
always wins: the per-peak proto-window half-extent is set by `edge_m`, and the
MHz value never binds. It would only take effect if set above
`edge_m · step ≈ 5.03 MHz`. At the shipped defaults `min_window_half_width_mhz`
is **dead weight** — a stale MHz knob superseded by a bin-count parameter.

**Two consequences.**

- The parameter is effectively a no-op. The min/extent side has the same
  MHz→points story the *max* side already adopted deliberately
  (`max_window_width_points` supersedes `max_window_width_mhz`,
  `window_planning.py:493-499`) — but on the min side it happened by accident
  (overridden by `edge_m`) rather than by design, and the MHz knob was left in
  place.
- More importantly, nothing enforces a **minimum final window width**.
  `min_window_half_width_mhz` only (would) set the *proto*-extent before the
  trim/split steps; those steps produce sub-minimum windows (309 at 3.77 MHz)
  with no floor. The "~4 MHz minimum" is nominal, not a constraint the planner
  honours — which is the F2 enabler.

**Candidate cleanup (to prioritize later).** Decide the intended contract:
either retire `min_window_half_width_mhz` as superseded by `edge_m`, or
re-express the minimum as a bin/points count consistent with the rest of the
planner and **enforce it on split products** (the F2 fix). The two findings
share this resolution.

## F4 — Use the fit covariance + SNR as an overfit / non-identifiability signal

**Observation.** The per-window fit's parameter (co)variance carries an overfit
signal that nothing currently consumes. The diagnostic is the **variance-
inflation factor** of a fitted amplitude: its reported uncertainty divided by
what its SNR alone would imply,

```
VIF ≈ (amplitude_error / amplitude) × snr
```

≈ 1 when the line is independently identifiable, and ≫ 1 when it is degenerate
with a neighbour (anticorrelated, so the *pair sum* is constrained but neither
amplitude individually is). Two windows on the reference fixture are a clean
truth pair:

| window | line | sep (res. elem.) | err/amp | snr | VIF | reading |
|---|---|---|---|---|---|---|
| 217 (overfit) | B 31328.0932 | 0.17 | 102% | 961 | ~980 | one line split into two |
| 217 (overfit) | D 31328.0795 | 0.17 | 190% | 515 | ~980 | anticorrelated with B |
| 217 | A 31328.2546 | — | 3.1% | 491 | ~15 | mild (blend shape error) |
| 217 | C 31328.3214 | — | 24% | 49 | ~12 | absorbs A's shape error |
| 161 (real doublet) | A 30384.1110 | — | 2.8% | 73 | ~2.0 | identifiable |
| 161 (real doublet) | B 30384.0462 | 0.82 | 4.4% | 45 | ~2.0 | identifiable |
| 161 | C 30383.8075 | — | 0.7% | 109 | ~0.8 | identifiable |

Window 217's B/D are 13.7 kHz apart (0.17 of a 78.6 kHz resolution element) with
amplitude uncertainties *larger than the amplitudes* despite SNR 500–960 — the
fit modelled one physical line as two anticorrelated components. Window 161 is a
genuine doublet at comparable closeness (0.82 elements) whose lines are each well
determined (VIF ~1–2). The VIF separates the overfit from the real doublet by
~500×, where separation alone does not.

**Why the existing machinery misses it.** The current overfit defences gate on
*separation* (the #13 resolution floor), *amplitude ratio* (the Tier-3 merge),
and *AICc/n_eff*. Window 217's B/D survived because their amplitude ratio (~1.9)
reads as a balanced "real doublet" the merge tier keeps — yet they sit at 0.17
resolution elements and are non-identifiable. The covariance is a *direct*
identifiability measure that catches exactly this class. It also feeds F1's gap:
a "non-identifiable peak (VIF ≫ 1)" attention reason would be high-precision and
target the actionable overfit case the queue currently has no reason for.

**Plumbing — persist the per-window parameter covariance (agreed).** The cheap
VIF needs no new persistence (`amplitude_error` + `snr` are already stored), but
the *off-diagonal* covariance — which names B↔D as the trading pair for a
targeted merge rather than just flagging both — is computed at fit time
(`WindowFitResult.covariance`) and then **discarded**; only the diagonal
survives into `FittedPeak.amplitude_error`. Persisting the full per-window
parameter covariance is the agreed direction: it is the complete second-order
uncertainty description of the fit, valuable well beyond overfit detection —
honest correlated error bars and uncertainty propagation for the final-products
table and reports, the anticorrelation structure the future Qt frontend and any
downstream molecular-fitting layer need, and reproducibility of the reported
uncertainties without re-running the fit. Discarding it loses information the
file cannot reconstruct.

This is a serialization-design task (a follow-up, not done here): add a
per-window covariance dataset to the `stage5_fitting` group with an explicit,
documented parameter ordering (per peak `amplitude, offset, phase`, then the
shared `tau`, then the baseline coefficients), update
[`SERIALIZATION_STRATEGY.md`](../SERIALIZATION_STRATEGY.md), and round-trip it
through all three interfaces. A cheap derived statistic (e.g. max pairwise
|correlation| among amplitudes) can ride alongside for the attention flag.

**Generalize beyond amplitude — all parameters, and a threshold-based attention
metric (window 281).** The amplitude VIF is one face of a broader signal: the
fit estimates frequency and phase uncertainties (and their cross-correlations)
too, and the non-identifiability can live in any of them. Window 281
`[33291.58, 33301.63]` (χ²ᵣ 0.72) is the complementary case to 217 — a *low-SNR*
pair, not a high-SNR degeneracy:

| pk | freq (MHz) | f_err | err/amp | phase | ph_err/π | snr | amp VIF |
|---|---|---|---|---|---|---|---|
| A | 33296.6715 | 23.2 kHz | 56% | 2.298 | 14% | 8.0 | ~4.5 |
| B | 33296.7562 | 26.0 kHz | 62% | 0.673 | 16% | 7.1 | ~4.4 |

The pair is 84.7 kHz apart (1.08 resolution elements). The amplitude VIF (~4.5)
is only mildly elevated — an amplitude-only rule would pass it — but the
*frequency* errors are ~30% of the very separation that justifies resolving two
lines. The identifiability problem here lives in frequency (and amplitude), not
where 217's did. χ²ᵣ < 1 reinforces that the model is over-flexible for this weak
data. The phase *value* is **not** diagnostic here: the two lines' phases differ
by ~1.6 rad, but on this instrument (phase-ramped excitation pulse, then mixing
and amplification) there is no empirical frequency–phase relationship — genuine
doublets routinely have unrelated phases — so a phase *difference* between peaks
carries no realness signal. Only a phase's *uncertainty* (and its correlations
with other parameters) is informative, not its value.

Each parameter needs its **own** normalization for a "relative error" to mean
anything: amplitude against its value (or the SNR-implied scale, the VIF);
frequency against the resolution element / line separation; **phase error
against π** (its bounded range — *not* against the phase value, which has no
meaningful zero, blows up near 0, and is not comparable across lines anyway). The
intended metric is then per-window and parameter-agnostic: flag a window for
attention when **any** fitted parameter's appropriately-normalized *error*, or
any pairwise **(anti)correlation** among parameters, exceeds a threshold —
amplitude VIF catches 217, the frequency error/correlation catches 281. Phase
enters only through its error magnitude and its correlations, never through the
phase value. This is the high-precision overfit attention reason F1 is missing,
and it subsumes the amplitude-only VIF above.

**Guard case — a legitimate low-SNR multiplet the metric must NOT collapse
(window 108).** Window 108 `[28873.90, 28877.20]` (χ²ᵣ 1.19) is two genuine
doublets — A/B at 84.5 kHz and C/D at 86.0 kHz (each ~1.08 resolution elements,
the same closeness as 281), the pairs split by 906 kHz:

| pk | f_err | err/amp | snr | amp VIF |
|---|---|---|---|---|
| A 28875.0217 | 13.7 kHz | 32.0% | 7.2 | 2.3 |
| B 28875.1062 | 8.9 kHz | 22.7% | 10.9 | 2.5 |
| C 28876.0122 | 9.0 kHz | 22.3% | 10.0 | 2.2 |
| D 28876.0982 | 12.8 kHz | 29.5% | 7.1 | 2.1 |

This is the critical guard: its **raw** relative amplitude errors are *large*
(22–32%, larger than 217's well-determined peak A at 3.1%) purely because SNR is
7–11 — yet the **VIF is a flat ~2.1–2.5**. A metric thresholded on raw error
would wrongly collapse this real quartet; the **SNR-normalized** VIF correctly
passes it (the errors are what low SNR predicts, not degeneracy). This is exactly
why the normalization is mandatory. The off-diagonal confirms it once persisted:
108's close-pair amplitudes are independently determined (near-zero correlation),
where 217's B/D are ~−1 — the correlation magnitude is the clean separator.

So the legitimate cases sit at VIF ~1–2.5 *regardless of SNR* (108 low-SNR, 161
mid-SNR), 281 is marginal (~4.5), and 217's overfit explodes (~980).

**Second guard — a true blended doublet with no visible dip (window 419).**
Window 419 `[36132.04, 36134.63]` is two real lines 87.9 kHz apart
(1.79 × FWHM = 49.2 kHz) with a 3.8:1 amplitude ratio, so the weak line (A,
SNR 19) rides on the wing of the dominant line (B, SNR 72) as a shoulder —
**no resolved dip**, only an asymmetric/over-wide composite; the line width is
the sole visual cue. The fit resolves both and the **VIF stays modest
(A 2.2, B 3.1)** → keep. Two lessons: (a) the covariance carries the "two real
lines" verdict the eye cannot read off a missing dip, so a dip/separation
heuristic is not enough; (b) this window's χ²ᵣ is **2.98** yet the fit is
correct — that is the strong-line lineshape-fidelity floor (B at SNR 72), not a
defect, which is another reason χ²ᵣ is a poor attention signal and the
VIF/correlation is the better one (cf. F4 vs the SNR² χ²ᵣ floor).

**Candidate uses (to prioritize later).** The parameter-agnostic VIF/correlation
attention metric (cheap diagonal part available now; the correlation part rides
on the persisted covariance below); and a covariance-driven merge that collapses
a degenerate pair back to one line — distinct from the separation/ratio
heuristics, calibrated on the truth set: 217 (high-SNR amplitude degeneracy,
collapse) / 281 (marginal low-SNR pair) / 161 (real doublet, keep) / **108 (real
low-SNR quartet, must keep — the large-raw-error guard)** / **419 (true blended
doublet with no dip, must keep — the width-not-dip guard; χ²ᵣ 2.98 from the
strong-line floor, not overfit)**. Related: the
resolution-floor + amplitude-ratio overfit discriminant
([`stage5-subresolution-overfit.md`](stage5-subresolution-overfit.md)) and the
doublet-alternative evidence
([`stage5-doublet-alternative.md`](stage5-doublet-alternative.md)).

## F5 — Discard fitted peaks below an absolute SNR floor (~3)

**Observation.** Fitted peaks far below the Stage 3 detection threshold survive
into the final model as dust. Window 441 `[36361.02, 36363.54]` (χ²ᵣ 1.45) is
four such peaks, all SNR 1.5–2.4 — pure noise, no real line:

| pk | freq (MHz) | amp | snr | err/amp | VIF |
|---|---|---|---|---|---|
| A | 36361.0564 | 2.65e-7 | 2.11 | 416% | 8.8 |
| B | 36361.1264 | 2.97e-7 | 2.37 | 438% | 10.4 |
| C | 36361.7891 | 1.90e-7 | 1.51 | 41% | 0.6 |
| D | 36362.8167 | 2.01e-7 | 1.60 | 38% | 0.6 |

File-wide on the reference fixture: **32 of 603 fitted peaks are SNR < 3, and 16
windows are entirely sub-3** (pure-dust windows that hold no real signal and
should drop out).

**Independent of F4 — both cuts are needed.** F4's covariance/VIF catches
*degenerate* dust (441's A/B: VIF ~9–10 from their mutual anticorrelation), but
it does **not** catch *absolute-weak* dust: 441's C/D have VIF ~0.6 — they pass
the identifiability test — yet they are SNR-1.5 noise. A hard SNR floor is the
orthogonal cut that removes peaks the data simply does not support regardless of
their correlation structure. F4 (relative/identifiability) and F5 (absolute
evidence) are complementary, not substitutes.

**The rule.** Discard any fitted peak below an absolute post-fit SNR floor
(~3), and drop windows left empty. The floor is safe against the keep-set: every
legitimate line in the F4 truth set is SNR ≥ 7 (108's weakest is 7.1), so a
threshold at ~3 clears dust with wide margin. Anchor it to the **Stage 3
detection threshold** (~3.2, [`stage3-snr-corner-benchmark.md`](stage3-snr-corner-benchmark.md)):
a peak the detector would never have promoted should not survive the fit either.

**Why dust reaches the final model.** Stage 3 promotes at ~3.2, so these
sub-3 peaks are not primary detections — they are residual-rescue additions or
split products whose post-fit SNR fell below the promotion bar, with nothing
pruning them afterward. The fix is a final post-fit cleanup pass (prune <
floor, then drop emptied windows), distinct from the rescue/split accept gates
that admitted them.

**Relationship to F1.** With this auto-prune in place most dust never reaches
the user, so F1's missing "weak fitted peak" attention reason only needs to
surface the *borderline* band just above the floor — the hard floor handles the
clear cases, the attention reason handles the judgment calls.
