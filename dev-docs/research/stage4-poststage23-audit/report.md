# Stage 4 audit after the Stage 2/3 rework

Stage 2 (MAD subdivision + moving-median σ, commit `3618a56`) and Stage 3
(matched-filter exp-apodized active-FT gap pass, commit `5ab542d`) landed back
to back. The downstream stages were not re-tuned in either commit. This report
audits Stage 4 (window assignment) against the new Stage 2/3 surface on the
2638 fixture, decides whether its calibrated constants still hold, and
recommends actions.

## Method

Full-pipeline rebuild from raw blackchirp data → import → detect_start_time →
FT (trim=(26500, 40000), canonical unapodized) → noise → peaks → windows → fit.
Rebuild the fixture with [`build_fixture.py`](build_fixture.py). The audit's
noise comparison (old vs scatter estimator), edge-coherence calibration check,
outlier-window inspection, and doublet-recoupling probe were one-off diagnostic
drivers; their findings are recorded below.

## Headline numbers (2638)

| metric | pre-rework | post-rework |
|---|---|---|
| promoted peaks | 709 | 883 (+25 %) |
| windows | 347 | 391 (+13 %) |
| hard windows | 218 (63 %) | 254 (65 %) |
| windows with ≥1 fixed contributor | 71 (20.5 %) | 82 (21.0 %) |
| total fixed contributors | 123 | 151 (+23 %) |
| FC per window (median / p95 / max) | 2 / 10 / 16 | 0 / 2 / 7 |
| max window width | ≈ 31 MHz | 65.67 MHz |
| leakage-touched fraction | ≈ 11 % | 17.7 % |
| fitted peaks (Stage 5) | 723 / 648 (varying baseline) | 799 |
| chi²_r p50 / max | 1.30 / 4183 | 1.24 / 125 |

Full non-slow suite: **548 / 548 passing.**

## What the new Stage 2 σ does to Stage 4

Stage 4 has three σ-coupled levers; the new estimator shifts σ in a
spatially non-uniform way (`scratch/stage4-audit/compare_noise.py`):

| distance to nearest strong peak | n points | ratio (new/old) median |
|---|---|---|
| ≤ 2 MHz (in skirt) | 37 444 | 0.753 |
| 2–10 MHz | 106 272 | 0.761 |
| 10–40 MHz | 281 212 | 0.941 |
| > 40 MHz (far) | 707 534 | 0.998 |

The new estimator removes the skirt-bias inflation, so σ drops ≈ 25 % within
the Lorentzian-skirt zone of strong lines and is essentially unchanged in
noise-only regions. Per-window σ_c ratio: median 0.98, p05 0.63, p95 1.11.

### Lever 1 — `DEFAULT_MAGNITUDE_ATTACHMENT_THRESHOLD = 0.1 σ_c` (Tier-1)

The contributor-attachment threshold is `0.1 × σ_c(w)`. With σ_c dropping
≈ 25 % in skirt regions the effective threshold drops by the same factor
there, so naïvely we expect *more* attachments. The post-rework data show
the opposite pattern at the window-population level:

- pre-rework calibration (`stage5-fitting.md` §O5-10): median 2, p75 5, p95 10, max 16
- post-rework observed: median 0, p95 2, max 7

Total FC went up modestly (123 → 151, +23 %), matching the window count
growth (347 → 391, +13 %) plus a small per-window increment. The
per-window distribution shifted *downward* because the larger window
count spreads strong-line skirts across more candidate dependent windows,
so each window sees fewer geometrically-relevant strong primaries. The
σ-driven threshold loosening is real but second-order to the structural
re-partition.

**Verdict:** Tier-1 attachment is within the calibrated band and producing
*fewer* contributors per window than before — the opposite of the
over-attach risk. No change to `DEFAULT_MAGNITUDE_ATTACHMENT_THRESHOLD`.

### Lever 2 — `DEFAULT_EDGE_THRESHOLD = 8.0` (de-ramped S_coh on the user FT)

S_coh = |Σz| / (σ · √M) — σ in the denominator. With σ lower in skirt
regions the statistic reads higher there, so the leakage-touched map
extends further from each strong line. Observed on 2638:

- leakage-touched fraction: 11 % → 17.7 %
- touched intervals: 1777
- S_coh distribution (n = 1.13 M finite samples): median 4.18, mean 8.74,
  max 1312.8
- far-from-strong (> 60 MHz): median 3.54, p99 16.7, p99.9 70.4, 5.8 %
  above T = 8 (these "far" samples still include weak/medium-line skirts)
- skirt (≤ 20 MHz): median 8.23, p95 100.6, p99 244.4, 51 % above T = 8

The histogram is monotonic; there is no global bimodal valley to point
at, but the original calibration argument (`leakage-detection-rework.md`:
T_edge = √M is the per-bin leakage level at which a sidelobe's lobe peak
clears the gap-pass detection floor) is unchanged by the σ shift —
T_edge is locked to the statistic, not to the σ. The Stage-4-internal
consequences of the inflated touched-map are unambiguously sane:

- hard-window fraction barely shifts (63 % → 65 %),
- easy-window chi² distribution is excellent (p50 = 0.95, max = 2.45,
  91 % under 1.5),
- the strong-line skirt boundary used in step 6 difficulty classification
  remains the right discriminator.

The one visible structural change is the 36350 / 36389 doublet — see
**§Doublet recoupling** below.

**Verdict:** T_edge = 8 still partitions cleanly. No change to
`DEFAULT_EDGE_THRESHOLD`.

### Lever 3 — `DEFAULT_MIN_FREEZE_SNR = 50.0`

SNR-based, varies inversely with σ. Where σ drops 25 % the SNR rises ≈ 33 %,
so strong lines near other strong lines get *more* readily marked
`freeze_eligible`. This is the correct direction (a strong line in a less-
inflated noise floor is more confidently characterised, so freezing it is
safer). No observed pathology; no change recommended.

## Doublet recoupling (36350 / 36389)

The single structural plan change worth flagging. Pre-rework, the two
SNR > 50 strong lines decoupled into adjacent ~7 MHz windows because the
de-ramped S_coh dipped below T_edge = 8 between them (σ was inflated by
each line's own skirt). With the cleaner Stage 2 σ, S_coh stays above
threshold throughout the 36350–36389 stretch, so strong-cluster grouping
merges them into one **primary joint window**:

- post-rework w302: [36334.36, 36400.03] MHz, **65.67 MHz wide**, 33 free
  peaks (6 strong + 4 medium + 23 weak), hits `max_window_width_mhz = 40`,
  flagged `needs_joint_treatment = True` (no clean interior split point
  found).

Stage 5 honours `needs_joint_treatment` via its AICc-gated conservative
loop: 9 of the 33 candidate peaks were retained, chi²_r = 3.48, no replan
requested. Worst-case chi² over the whole spectrum dropped from 4183 to
125, and that 125 is now at w216 ([33721.73, 33726.39], 3 free peaks,
1 FC) — *not* at w302. The recoupling is consistent with the more accurate
noise estimate and appears to be the better fit-time arrangement.

This was an explicit open item in
`dev-docs/planning/leakage-detection-rework.md` §"Open items handed to
Stage 5" — "whether to re-couple the pair for fitting is a Stage 5
question" — and the new σ has answered it: the two halves now co-fit by
default. Stage 5 then chose to fit only 9 of 33 peaks, so the open
question becomes whether 9 peaks is the right model order; that is a
Stage 5 concern.

The two leakage-detection-rework.md notes that mention the decoupling
("decouples at T_edge = 8", "Each is carried into the other's window as
a fixed contributor") are now historical and need to be revised. See
**§Recommended doc updates**.

## Item-by-item findings (from the task prompt)

**1. Peak-count sensitivity (Stage 3 promotes ~25 % more peaks).** Window
count grew proportionally (+13 %); per-window free-peak count median = 2,
p95 = 6 — same shape as before. The newly-promoted gap-pass peaks
distribute 42 → hard windows / 61 → easy windows (out of 103 new
promotions). No fragmentation; no mass migration of weak detections into
strong-cluster windows. **No regression.**

**2. Tier-1 magnitude-attachment thresholds vs new σ.** Re-checked Lever 1
above. Per-window FC distribution shifted *downward* (max 16 → 7), so
the risk is under-attach, not over-attach. Within the calibrated 0.1 σ_c
operating band on the pre-rework noise (the user's calibration
table — `stage5-fitting.md` §O5-10) and producing FCs whose chi²
improvement is unambiguous. **No regression.**

**3. Edge-coherence threshold (DEFAULT_EDGE_THRESHOLD = 8 on the user FT).**
Re-checked Lever 2 above. Touched-region coverage increased 11 % → 17.7 %
(correct direction — σ no longer suppresses the test in skirts), but
hard-window fraction barely changed and easy-window chi² is excellent.
The one downstream consequence (36350 / 36389 recoupling) is documented
in §Doublet recoupling. **No regression.**

(The user's prompt also flagged the Stage 3-internal
`GAP_MASK_EDGE_THRESHOLD = 8` on the MF active-FT grid. That threshold
lives on a different statistic on a different grid; auditing it belongs
to Stage 3, not Stage 4. The Stage 3 rework's own §10 calibration
locked T = 8 on the new MF grid against its own σ; no Stage 4-side
audit can or should re-examine it.)

**4. Other Stage 4 constants baked in σ-magnitude or peak-count
assumptions.** Inventory from `preprocessing/window_planning.py`:

| constant | σ / peak-count dependency | post-rework behaviour |
|---|---|---|
| `DEFAULT_MAX_WINDOW_WIDTH_MHZ = 40.0` | none (geometric) | width cap still hits the 36350/36389 cluster and the 28800 cluster; correctly fires `needs_joint_treatment` |
| `DEFAULT_MIN_WINDOW_HALF_WIDTH_MHZ = 2.0` | none (geometric) | unchanged |
| `DEFAULT_EDGE_M = 64`, `DEFAULT_TRIM_M = 32` | none (bin counts) | unchanged |
| Step-5 leakage-artifact pruning (`pk.intensity < env`) | none (magnitude ratio) | unchanged |
| Step-6 edge-coherence fail | T_edge (lever 2) | covered above |

**No further recalibrations required.**

## Recommended actions

1. **No Stage 4 code changes.** All calibrated constants
   (`DEFAULT_EDGE_THRESHOLD`, `DEFAULT_MAGNITUDE_ATTACHMENT_THRESHOLD`,
   `DEFAULT_MIN_FREEZE_SNR`, `DEFAULT_MAX_WINDOW_WIDTH_MHZ`,
   `DEFAULT_MIN_WINDOW_HALF_WIDTH_MHZ`, edge/trim band widths) stay at
   their current values.
2. **Doc updates** to record the 36350 / 36389 recoupling and the new
   leakage-touched fraction; see §Recommended doc updates below.
3. **Spotlight w302 and the worst-chi² windows in Stage 5 follow-up.**
   The recoupling and the AICc-pruned joint fit are working, but
   chi²_r = 3.48 on the 33-peak cluster is the kind of number that warrants
   a per-window dive once Stage 5 stabilizes other knobs. Not a Stage 4
   concern.

## Recommended doc updates

- `dev-docs/planning/leakage-detection-rework.md` §"Verification (2638
  fixture)" line "Stage 4 plan stays sane. 339 windows, max width ~30
  MHz, no mega-windows." → revise to the post-Stage-2/3-rework numbers
  (391 windows, max width 65.67 MHz, one joint-required cluster at
  36350/36389).
- `dev-docs/planning/leakage-detection-rework.md` §"Open items handed
  to Stage 5" entry "The 36350/36389 doublet decouples at T_edge = 8 …" →
  mark resolved (the doublet now recouples under the new Stage 2 σ;
  Stage 5 joint-fits with AICc pruning).
- `dev-docs/planning/stage4-window-assignment.md` step 3 note "(D8): at
  the recalibrated T_edge = 8, 2638's 36350/36389 pair … does *not* stay
  above threshold throughout and so decouples into two windows." → update
  to reflect that under the post-Stage-2-rework σ the pair *does* stay
  above threshold and merges into a single primary joint window.
- `dev-docs/planning/stage4-window-assignment.md` §"Test plan" 2638 row
  "~328 windows, max width ~31 MHz" → 391 windows, max width 65.67 MHz.

These are status notes (changed observed behaviour on the 2638
fixture), not normative-spec changes. The algorithm, parameters, and
thresholds stay locked.

## Artefacts

[`build_fixture.py`](build_fixture.py) regenerates the fixture (into the
gitignored `scratch/` tree). The numbers above came from one-off diagnostic
drivers run against that fixture — a noise comparison (old vs scatter σ), an
edge-coherence / T_edge calibration check, an outlier-window inspection (top-N
by free / FC / width), and a doublet-recoupling probe (w302 anatomy, worst-χ²
windows, per-difficulty χ² split). Their findings are recorded in the sections
above.
