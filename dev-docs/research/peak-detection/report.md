# The peak-detection algorithm: cost, correctness, and the sidelobe problem

A research report on the detector used by the peak-detection stage of
the FTMW processing pipeline — the smoothed second-derivative locator
(`locate_peaks`) and the two-pass driver (`detect_peaks`) around it.
The investigation profiles the detector, characterises why it
over-produces on unapodized (boxcar-truncated) spectra, tests whether
that over-production can be cheaply suppressed, and verifies a concrete
fix on the 2638 fixture. Code that regenerates every figure here is in
`prototype.py`; reproducibility details at the end.

## 1. Algorithmic context

The peak-detection stage produces an ordered list of classified peaks
that (a) seed the downstream analysis windows and (b) ensure no line
above the SNR floor is missed. It is deliberately allowed to be coarse:
the windowing and fitting stages resolve close lines that detection
smears together. Two pieces do the work.

**`locate_peaks` — the locator.** A Savitzky-Golay filter smooths the
magnitude spectrum and returns its 1st and 2nd derivatives. Peaks are
the local minima of the negative-clipped 2nd derivative — points that
are both sharply concave-down and locally extremal. An optional
per-point threshold (`min_snr · σ`) drops sub-threshold detections, and
a split/merge heuristic collapses a pair that straddles one strong
feature. It is a textbook concavity detector; the concavity test is
what rejects noise shoulders on the side of a strong line.

**`detect_peaks` — the two-pass driver.** Real FTMW spectra are
*unapodized* downstream (full resolution, boxcar truncation), and a
boxcar-truncated strong line carries sinc-shaped truncation sidelobes
whose envelope decays only as `1/Δf`. Running the locator directly on
that spectrum detects those sidelobes as lines. The driver's answer:

1. **Primary pass** — run the locator on an *apodized* spectrum, where
   apodization suppresses the sidelobes, for a robust strong-line list.
2. **Gap pass** — run the locator on the *unapodized* spectrum to
   recover weak lines apodization smeared away, but drop any detection
   inside a **leakage-touched region**: a stretch where a strong line's
   coherent truncation skirt is still measurable. That region map is the
   de-ramped complex-edge coherence statistic of [the windowing
   report](../complex-edge-coherence/report.md)
   (`leakage_touched_intervals` in `preprocessing/leakage.py`).

This report asks three questions of that design: is it *fast enough*,
is it *correct*, and can the residual false positives — the user-facing
complaint, and a pain point for the sibling BlackChirp project, which
runs a single-pass detector on unapodized spectra — be cheaply
suppressed.

> **D8 update.** §5 below was written before the de-ramp result. As
> first investigated, the gap-pass mask was the closed-form leakage
> *reach* (`estimate_leakage_reach`), and §5 concluded that was the best
> available suppressor and that no cheap phase-based discriminator
> existed. The D8 rework overturned the conclusion, not the
> investigation: de-ramping the complex spectrum to the active-region
> turn-on restores a rolling-band phase-coherence statistic that maps
> the leakage-touched regions directly — and measured the closed-form
> reach 7–25× too narrow on real data. §5 and the §6–§8 reach-mask
> references are revised accordingly; the §2–§4 cost analysis and the
> §6 primary-apodization calibration are unaffected and stand.

## 2. Cost profile

`locate_peaks` is linear in the spectrum length and cheap. On white
noise from 10 K to 1 M points (`prototype.py` §1):

| N         | total    | Sav-Gol (d1+d2) | argrelmin |
|-----------|----------|-----------------|-----------|
| 10 000    | 0.9 ms   | 0.4 ms          | 0.2 ms    |
| 100 000   | 7.2 ms   | 3.0 ms          | 1.4 ms    |
| 500 000   | 46.9 ms  | 18.3 ms         | 18.1 ms   |
| 1 000 000 | 94.1 ms  | 35.5 ms         | 35.9 ms   |

![locate_peaks cost breakdown](figures/01_locate_peaks_cost.png)

Cost is split roughly evenly between the Savitzky-Golay convolution and
`argrelmin`, with the threshold/merge tail making up the rest. There is
no superlinear term and no pathological case.

**The detector is not the stage's bottleneck.** On the 2638 fixture
(N ≈ 566 K), the two `locate_peaks` calls cost ~113 ms combined, but the
stage spends far more recomputing its inputs (`prototype.py` §2):

![Stage 3 wall-clock breakdown on 2638](figures/02_pipeline_breakdown.png)

| step                | wall time |
|---------------------|-----------|
| primary FT recompute | 94 ms    |
| gap FT recompute     | 76 ms    |
| primary noise        | 240 ms   |
| gap noise            | 251 ms   |
| primary `locate_peaks` | 67 ms  |
| gap `locate_peaks`   | 46 ms    |

The two FFT recomputes (~170 ms) and the two full noise re-estimations
(~491 ms) dominate; the peak finder is under 15 % of the measured work.
The noise cost is dominated by `estimate_noise_scatter`
(the high-pass, region-aware MAD estimator), which is substantially
more expensive than a level-based estimator but also far more
accurate on line-dense high-SNR spectra. If the stage's latency ever
needs to come down, the redundant input recomputes are the target —
not the detector. Two cheap detector-side options exist but are not
pressing: the 1st-derivative Sav-Gol convolution is used only by the
merge heuristic's sign test and could be elided when the merge does not
fire, and `argrelmin` with `order = window//2` rescans a wide
neighbourhood that a single-pass comparison would not. Neither is worth
doing until the recompute cost is addressed first.

**Decision: leave the detector's performance alone.** It is linear,
predictable, and a minority of the stage cost. Recorded here so a
future optimisation pass starts at the FFT/noise recomputes.

## 3. Why unapodized spectra over-produce

The synthetic test bed builds a baseband spectrum from a known line
list (30 lines, peak SNR log-spaced 5–500, minimum separation 3 MHz,
T = 15 µs acquisition) by summing truncated cosines in the time domain,
applying an apodization, and forward-FFT'ing. The pipeline's canonical
FT is unconditionally unapodized; the apodizations here are
research-only choices applied via the common `apodize_fid` helper
(`ftmwpipeline.utils.signal_processing`) to characterise the
trade-off. Complex Gaussian noise is added in the time domain before
apodization. The locator runs at a 3σ floor; a detection within 0.5 MHz
of a truth line is a true positive (`prototype.py` §3).

| apodization          | true positives | false positives |
|----------------------|----------------|-----------------|
| boxcar (unapodized)  | 29             | **324**         |
| exponential 5 µs     | 26.5           | **401.5**       |
| Hann                 | 30             | 16.5            |
| Blackman-Harris      | 29.5           | **1.5**         |
| Kaiser β = 8.6       | 30             | **2**           |

![True/false positives across apodizations](figures/03_apodization_tp_fp.png)

The result is confirmed and quantified: a boxcar spectrum yields
roughly **200× the false positives** of a Blackman-Harris or Kaiser one,
for the same true-positive recovery. The strong windows pay for that
with a small handful of false negatives (the weakest lines, broadened
below the floor) — an acceptable trade for a *position-finding* pass
whose misses are backfilled by the gap pass.

The critical row is **exponential 5 µs: ~400 false positives** — worse
than boxcar in this synthetic (more FPs with fewer TPs), because the
mild exponential does not suppress sidelobes and slightly broadens
lines near the detection floor. This row matters because it is the
apodization the research baseline §6 benchmarks against; the shipped
primary pass uses Blackman-Harris instead.

## 4. Anatomy of the false positives

Every unapodized false positive is a sinc sidelobe. Taking the boxcar
case and measuring each false positive's distance to, and height
relative to, its nearest truth line (`prototype.py` §4):

![Anatomy of unapodized false positives](figures/04_fp_anatomy.png)

- **99.1 %** of false positives (320 of 323) lie within the closed-form
  leakage reach of some truth line.
- Median distance to the nearest truth line is **0.43 MHz**; 95th
  percentile **1.4 MHz**.
- Median height is **2.3 %** of the parent line's peak; 95th percentile
  45 %.
- The scatter tracks the analytic `1/(πΔf·T)` boxcar sidelobe envelope.

The false positives are not noise excursions and not a detector defect.
They are real, coherent spectral features — the truncation sidelobes
the finite acquisition genuinely puts there. The locator is correctly
reporting concave-down maxima above the noise floor; those maxima
simply are not independent lines.

## 5. Why no cheap local test suppresses them

The decisive negative result of this investigation: **a sinc sidelobe
and a genuine weak line are locally indistinguishable.** Both are
concave-down maxima of width ≈ 1/T; their heights overlap (a sidelobe
of a strong line is far brighter than a weak real line). Nothing
measurable in a neighbourhood of one candidate separates the two
classes. Four single-spectrum suppression heuristics were swept on the
boxcar synthetic, plus an apodized-companion veto (`prototype.py` §5):

| candidate                       | best operating point      | TP kept / 29 | FP kept / 323 |
|----------------------------------|---------------------------|--------------|---------------|
| prominence ≥ α·σ                 | α = 2σ                    | 14           | 141           |
| stronger-neighbour height ratio  | α = 0.05                  | 17           | 109           |
| phase anti-coherence vs neighbour| thr = −0.95               | 23           | 288           |
| closed-form reach mask (self)    | min_snr = 5               | 23           | 173           |
| apodized-amplitude veto (BH)     | k = 1.5σ                  | 19           | 110           |

![Suppression candidates, TP-retained vs FP-retained](figures/05_suppression_roc.png)

Every curve hugs the diagonal — removing false positives costs true
positives at nearly 1:1. Three results are worth recording so they are
not re-attempted:

- **Prominence fails** because a sinc sidelobe sits between two sinc
  zeros, so its prominence *is* essentially its full height; meanwhile
  a genuine weak line riding the skirt of a strong one has *low*
  prominence. The test removes real lines preferentially.
- **Phase gives no cheap *local* separator — but a rolling-band one
  exists.** The truncation phase factor `exp(−iπΔf·T)` winds at the
  *same rate* at a real line's own centre and at a distant line's
  sidelobe, so no phase-gradient or anti-coherence cut on a *single
  candidate's neighbourhood* holds — the "phase anti-coherence" row
  above confirms it (288 of 323 FPs kept). What the sweep did not test
  is a *rolling-band* phase statistic. The windowing stage's complex-edge
  coherence statistic integrates an oriented M-point band; on the
  full-record rfft of real data the leakage signal carries a turn-on
  phase ramp and oscillates, so a coherent sum cancels on it — which is
  why it looked inapplicable here. De-ramping the spectrum to the
  acquisition turn-on (D8) removes that oscillation, and the de-ramped
  rolling statistic then flags leakage-touched *regions* from the
  acquisition geometry alone (`start_us`, `probe_freq`) — no strong-line
  list required. It is a cheap phase-based discriminator; it was simply
  absent from this local-test sweep.
- **The apodized veto fails on close-in sidelobes.** A sidelobe within
  ~1 MHz of its parent hides under the *broadened* apodized main lobe
  of that parent, so the apodized magnitude there is high and the veto
  keeps it; meanwhile a genuine weak line, broadened by apodization,
  can drop below the veto floor. The veto alone is no better than the
  diagonal.

Among the swept *local* heuristics the closed-form reach mask is the
only candidate above the diagonal, and only mildly so — which is why
the design as first shipped masked the gap pass with it. **D8
superseded it.** Measured against the de-ramped rolling statistic on
the real 2638 spectrum, the closed-form reach under-predicts the true
coherent skirt by 7–25× (±1.4–3 MHz predicted vs ±20–48 MHz measured):
it models one isolated line and ignores the cumulative skirt of many
strong lines. The gap-pass mask is now the de-ramped leakage-touched
map; the closed-form reach is demoted to a cheap initial *proposal*
(see [the windowing report](../complex-edge-coherence/report.md) and
`dev-docs/planning/leakage-detection-rework.md`).

What does *not* change is the structural limit this section found: a
genuine weak line sitting inside a strong line's skirt cannot be told
from a sidelobe *at that location*, by any test, region-level or local.
The de-ramped statistic identifies the leakage-touched *region*; it
does not classify individual candidates within it. The gap pass
therefore *skips the region entirely* rather than running there and
trying to classify — accepting that a weak line buried in a strong
skirt is recovered by the windowed primary pass and the downstream
fit, not by the gap pass. That policy, not a per-candidate
discriminator, is the resolution.

**This is why the two-pass design exists, and the investigation
vindicates it.** The apodized primary pass is an *independent
measurement channel* in which the strong line's skirt is gone, so a
weak line sitting inside that skirt becomes visible on its own merits.
The gap pass then runs the boxcar locator only *outside* the de-ramped
leakage-touched regions — the region map supplying the provenance no
local test could synthesise. No bolt-on local suppressor improves on
this; the two-pass architecture *is* the fix, and it should be kept.

## 6. The primary pass is not clean — and the fix

The two-pass design is sound, but the research baseline — a 5 µs
exponential primary — is a poor configuration. The exponential is a
mild filter that leaves substantial sinc sidelobes (§3: ~400 false
positives). The "clean" pass is not clean on that choice: it detects
its own sidelobes as lines, which pollutes the returned peak list — and
a sidelobe mis-promoted into the strong-line list misleads the windowing
stage that consumes it.

Verified directly on 2638 by recomputing the primary spectrum under
four apodizations via `apodize_fid` and counting detections, with the
self-consistent reach mask flagging sidelobe-suspects (`prototype.py`
§7):

![2638 primary-pass detections vs apodization](figures/07_2638_primary_apodization.png)

| primary apodization         | detections | reach-flagged sidelobe-suspects |
|-----------------------------|------------|---------------------------------|
| exponential 5 µs (baseline) | 7856       | 764 (9.7 %)                     |
| Blackman-Harris             | 5871       | 104 (1.8 %)                     |
| Blackman                    | 6324       | 124 (2.0 %)                     |
| Hann                        | 6765       | 142 (2.1 %)                     |

Switching the primary pass to Blackman-Harris cuts its sidelobe-suspect
fraction by ~5× on the real 2638 spectrum and reduces raw detections by
~25 %. Those removed detections are overwhelmingly sidelobes the mild
exponential fails to suppress.

**The primary pass should apodize with a strong window (Blackman-Harris
is the natural default; Kaiser or Blackman are equivalent).** The
primary pass's sole job is robust strong-line *position* finding —
amplitude, SNR, and the de-ramped leakage-touched map are all measured
downstream on the unapodized canonical spectrum, so the primary
apodization has no effect on any reported quantity except *which
positions* are found. For that job the most sidelobe-suppressing window
available is unambiguously correct. The pipeline's shipped Stage 3
uses Blackman-Harris internally, applying it via `apodize_fid`
independent of the canonical (unapodized) FT — this investigation is
the calibration that motivated that choice.

The false negatives a strong window introduces (§3: Blackman-Harris
missed ~0.5 of 30 synthetic lines) are not a concern: those are the
weakest lines, and recovering weak lines is precisely the gap pass's
job. A strong primary window shifts work to the gap pass by design.

## 7. A single-pass recipe for BlackChirp

The sibling BlackChirp project runs a single-pass detector on
unapodized spectra and has no apodized companion to lean on. Section 5
shows it cannot be fixed by a *local* test — but the de-ramped
rolling-coherence region mask is a single-pass hardening, and a cheap
one:

1. Detect all concave-down maxima above the SNR floor (the existing
   `locate_peaks`).
2. De-ramp the complex spectrum to the acquisition turn-on — multiply
   by `exp(+i2π·f_bb·t₀)`, with `f_bb` the baseband frequency and `t₀`
   the active-region start. This needs only the acquisition geometry,
   no line list.
3. Roll the complex-edge coherence statistic across the de-ramped
   spectrum and threshold it into leakage-touched regions
   (`leakage_touched_intervals`).
4. Drop any detection that falls inside a leakage-touched region.

This needs no second FFT — it reuses the complex spectrum BlackChirp
already has — and is `O(N)` in the rolling sum. It is the same mask the
two-pass gap pass now uses, and on real data it covers the true skirt,
which the older closed-form reach mask under-predicted by 7–25× (§5).
It is still imperfect in the §5 sense — a genuine weak line inside a
masked region is dropped along with the sidelobes — but for a
*window-seeding* detector that is the acceptable direction.

BlackChirp should also be offered the full two-pass option: if it can
afford one extra apodized FFT, the apodized-primary + region-masked-gap
architecture of this pipeline removes essentially all sidelobe false
positives (§3: Blackman-Harris primary → ~2 on 30 synthetic lines) and
additionally recovers weak lines that the single-pass mask drops.

Both building blocks — `deramp_to_active_start` /
`leakage_touched_intervals` and the older closed-form
`estimate_leakage_reach` — are factored out in
`preprocessing/leakage.py`, so either recipe ports without dragging the
rest of the pipeline along.

## 8. Where the residual pipeline false positives come from

With a clean (strong-window) primary pass, the pipeline's remaining
false positives are bounded. The gap pass skips every de-ramped
leakage-touched region, and a strong line is by definition caught by
the primary pass (apodization cannot push a *strong* line below the
floor), so its skirt is masked at the source. The sidelobes that
survive into the final list are therefore those falling in a
*sub-threshold dip* of the de-ramped coherence map — between lobes of a
medium line whose band-averaged leakage does not clear the gap mask.
The §5 ambiguity now resolves the other way: a weak real line buried in
a strong skirt is *lost* by the gap pass, masked along with the
sidelobes, and recovered — if at all — by the windowed primary pass and
the downstream fit.

On 2638 the unapodized gap grid carries 6152 raw locator detections
(at a 2σ floor; earlier measurements at a different floor gave 4569).
The de-ramped leakage-touched mask (`T_edge = 8`, calibrated in D8)
removes the strong-line skirts: the gap pass promotes 1576 peaks, down
from 2355 under the old reach mask — 779 strong-line sidelobes no
longer reach the final list, and the per-known-strong-line skirt count
falls to 0–1 within ±8 MHz. The improvement is twofold: a clean primary
pass (§6) means the *strong-line list* — the part of the output the
windowing stage trusts most — is no longer ~10 % sidelobes, and the
de-ramped gap mask means the *gap* additions are no longer dominated by
strong-line skirt ripple.

## 9. Caveats and known edges

- **Synthetic noise convention.** The synthetic adds time-domain noise
  before apodization, so apodized cases have slightly lower per-bin
  noise (the window reduces `Σw²`); the noise estimator absorbs this
  and the SNR scale shifts marginally between apodizations. The
  true-positive/false-positive *counts* are robust to this; precise
  SNR-at-detection values are not directly comparable across the §3
  rows.
- **Undamped lines.** The synthetic uses undamped (boxcar-truncated)
  cosines — the leakiest case. Real damped lines have a true Lorentzian
  `1/Δf²` far wing, so their sidelobes are weaker and the false-positive
  counts in §3 are an upper bound. The qualitative finding (no local
  test works; the two-pass design is necessary) does not depend on it.
- **The 0.5 MHz match tolerance** in §3 is generous relative to the
  boxcar peak shift; tightening it moves a few true positives into the
  false-positive column but does not change any decision.
- **`locate_peaks` window/order** were held at the pipeline defaults
  (`window = 11`, `order = 3`) throughout. A sweep of those parameters
  is out of scope here — they affect the smoothing scale, not the
  sidelobe problem, which is a property of the spectrum, not the
  filter.

## 10. Reproducing this report

The script `prototype.py` regenerates every figure under `figures/`
and the empirical numbers cited above. From the repository root:

```bash
conda run -n ftmwpipeline-dev python \
    dev-docs/research/peak-detection/prototype.py
```

Requirements: the project conda environment `ftmwpipeline-dev`
(matplotlib, numpy, scipy, the installed `ftmwpipeline` package), and
the 2638 fixture at `scratch/exp_2638.ftmw`. Sections 1 and 3–5 are
synthetic and need no fixture; sections 2, 6 and 7 use it and are
skipped with a printed notice if it is absent. Recreate the fixture
with:

```python
import ftmwpipeline.api as ftmw
ftmw.import_data("scratch/exp_2638.ftmw", source="examples/blackchirp_data/2638", force=True)
ftmw.detect_start_time("scratch/exp_2638.ftmw", band=(26500,40000), stamp=True)
ftmw.compute_ft("scratch/exp_2638.ftmw", trim=(26500,40000))
ftmw.estimate_noise("scratch/exp_2638.ftmw")
```

Random seeds for the synthetic sweeps are fixed inside `prototype.py`
(line list `20260601`, per-trial noise `20260700 + trial`), so the
synthetic figures are stable across runs given the same numpy/scipy
versions. The 2638 sections depend on the live noise-estimation and
FT code, so numerical values may shift slightly with future changes;
the qualitative findings (§5's negative result, §6's recommendation)
are not expected to change unless the underlying physics does.
