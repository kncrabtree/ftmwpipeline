# Handoff: leakage detection in Stages 3–4

Status: **open** — diagnosis complete, rework not started. Registered in
[`../ROADMAP.md`](../ROADMAP.md) (divergence **D8**).

This document hands off a problem found while validating the freshly-landed
Stage 4 implementation against the 2638 fixture. **Stage 4 is committed as
WIP** (all 349 tests green), but the investigation showed that both Stage 3's
gap pass and Stage 4's edge statistic mishandle finite-acquisition truncation
leakage, so the Stage 4 window plans are not yet trustworthy on real data. The
findings and the agreed path forward are recorded here so the rework can be
picked up cleanly.

## The core finding (with evidence)

The persisted 2638 spectrum is `expf_us=5.0, winf=None` — an exponential
filter, no window function. The exponential does **not** taper the abrupt
signal turn-on at `t = 2.35 µs` (decay factor 1.0 there) and only reaches ~8%
at `t = 15 µs`, so the spectrum carries near-full boxcar truncation sidelobes.

Three checks, all on 2638 (region 28795–28835 MHz, around the SNR≈212 line at
28817):

1. **Window-function test.** Re-FFT the FID with a real window
   (`blackmanharris`/`hann`) applied to the active region: the window
   `[28806.9, 28829.6]` collapses from ~30 promoted peaks to **4**, and the
   neighbouring window from 7 to **2**. The extra detections are truncation
   sidelobes, not lines.
2. **Complex FT.** Between the strong lines the real and imaginary parts do
   **not** return to zero — they ring with a clean, coherent sinc oscillation
   spanning the full ±12 MHz. That region is coherent leakage, not noise.
3. **S_coh.** The Stage 4 edge statistic reads ≈1.3 (noise-null level) right
   across that obviously-coherent ringing.

## Problem 1 — Stage 3 gap pass promotes sidelobes as weak lines

The two-pass design is sound and **stays**: the windowed primary pass
identifies strong peaks (and which peaks are skirt-contaminated); the
**unwindowed** gap pass recovers genuine weak features that windowing
submerges. Running the gap pass on unwindowed data is correct.

The bug is the **leakage exclusion / gap identification**. The gap pass is
masked within `preprocessing/leakage.py:estimate_leakage_reach` of each strong
line — ≈1.8 MHz for the SNR-212 line — but the real sidelobe skirt rings
coherently out to ±12+ MHz, and at any point the skirt is the *cumulative*
sum over all strong lines. The mask is ~7–10× too narrow, so the gap pass
fires on sidelobes from ~2–12 MHz out and promotes them as weak lines.

The windowed primary pass itself is **correct** — verified: the window
function is applied to the active region `[2.35, 15.0] µs` (not the whole
record), and the Stage-3 primary spectrum is byte-identical to a manual
`blackmanharris` FT and is sidelobe-clean. The fault is entirely the gap pass's
notion of where a "gap" is.

## Problem 2 — Stage 4 S_coh is the wrong statistic

`S_coh = |Σ z_k| / (σ√M)` is a *coherent sum* over M consecutive bins. Real
truncation leakage is an oscillating sinc skirt whose sign flips every ~0.3
MHz; an M=64 band spans ~2–3 oscillation periods, so the sum largely cancels
and S_coh sits at the noise null over genuine leakage. It was calibrated on
smoothly-phased synthetic leakage, which real boxcar/`expf` leakage is not.
Lowering `T_edge` does not help — the statistic simply does not respond to
oscillating leakage.

Everything in Stage 4 that consumes S_coh is therefore unreliable on real
data: leakage-touched regions, strong-cluster grouping, fixed-contributor
attachment, and the `edge_coherence_fail` difficulty criterion.

## Design constraints (confirmed with the project owner)

- **The end goal is time-domain fitting of *unwindowed* data.** A window
  function as an end state is explicitly **not** wanted — automated fitting of
  *windowed* data is already a solved/easy problem (the surviving `bcfitting`
  code reaches >95%). Exponential filtering is acceptable; an end-state window
  function is not.
- Truncation leakage is therefore *inherent* and *expected*. Stage 5's
  finite-T damped-cosine model reproduces it exactly (the `1 − e^{…}` turn-on
  term). The job of Stages 3–4 is **not to remove leakage** but to (a) avoid
  detecting it as independent lines and (b) tell which windows are
  leakage-coupled so Stage 5 carries the right contributors.
- The windowed spectrum is a *tool* used internally (identify strong /
  skirt-contaminated peaks); it is never the fitting target.

## Recommended path forward

One new capability serves both problems: an **oscillation-aware leakage
detector** that recognises the coherent sinc *ringing pattern* over a
frequency range (e.g. a matched filter against the finite-T sinc skirt, or an
autocorrelation/period-detection statistic on the complex FT), instead of a
flat windowed sum. `cft_real_imag` shows exactly the signature to match.

- **Stage 4:** replace `S_coh` (`preprocessing/edge_coherence.py`) with that
  oscillation detector. Its above-threshold extent is the true
  leakage-touched map; feed that to strong-cluster grouping,
  fixed-contributor attachment, and difficulty classification.
- **Stage 3:** use the same leakage-extent map to define the gap pass's
  "gaps" — the regions genuinely free of any strong line's coherent skirt.
  Run the unwindowed detector only there. Cross-checking unwindowed
  candidates against the windowed spectrum helps separate submerged weak
  lines (real, in a gap) from skirt artifacts. Recalibrate or retire
  `estimate_leakage_reach` for masking accordingly — the per-line `1/Δf`
  envelope under-predicts and ignores the cumulative skirt.
- **Separately (still valid):** Stage 4 windows have no baseline padding —
  they end at their outermost peaks' extents. Stage 5 fitting needs a
  configurable noise-only margin per window.

## What is committed as WIP

The full Stage 4 implementation: `WindowDifficulty`/`FixedContributor`/
`FitWindow`/`WindowPlan`, `preprocessing/edge_coherence.py`,
`preprocessing/window_planning.py`, `_internal/stage4_impl.py`,
`io/window_serialization.py`, the three interface wrappers,
`visualization/window_visualization.py`, and 37 tests (suite 344→ green).
**Trust boundary:** `edge_coherence.py` and every Stage 4 step that consumes it
are provisional pending this rework; the data structures, serialization,
stage-tracking/invalidation, and interface plumbing are sound and reusable.

## Reproducing the evidence

```bash
# build/refresh the fixture through Stage 3 (current algorithm)
conda run -n ftmwpipeline-dev python -c "
import ftmwpipeline.api as ftmw
ftmw.import_data('scratch/exp_2638.ftmw', source='examples/blackchirp_data/2638/')
ftmw.compute_ft('scratch/exp_2638.ftmw', zpf=2, expf_us=5.0, trim=(26500, 40000))
ftmw.estimate_noise('scratch/exp_2638.ftmw')
ftmw.detect_peaks('scratch/exp_2638.ftmw')
ftmw.assign_windows('scratch/exp_2638.ftmw')"
```

The window-function comparison and complex-FT plots are produced by ad-hoc
scripts under the gitignored `scratch/` (`wf.py`, `cft.py`); regenerate them
by FFT-ing the FID with `window_function="blackmanharris"` vs the canonical
`expf_us=5.0, winf=None` and plotting the 28795–28835 MHz region.
