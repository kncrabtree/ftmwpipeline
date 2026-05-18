# Plan: Stage 3 — Peak detection

Status: **planning** (not started). Scope is Stage 3 only. Stages 4 (window
definition) and 5 (fitting) are kept separate — see *Downstream context*.

Normative requirements remain in the `*_STRATEGY.md` specs; this document is
normative only for the Stage 3 work it tracks.

## Objective

The project goal is to fit *unwindowed* spectra (full resolution, but with
boxcar truncation leakage). Peak detection's job is not to be the final peak
list — it is to (a) seed the analysis windows used downstream and (b) ensure no
weak line above the SNR threshold is missed. Detection is deliberately allowed
to be coarse where windowing smears close lines together; Stage 4/5 resolve
those.

## Reuse map (corrected)

The earlier reference implementation **survives** at
`~/github/bcfitting/src/bcfitting/ftmwfitting.py` (external to this repo;
reference only, not a dependency). Relevant to Stage 3:

- `locate_peaks` (~L555–651): Savitzky-Golay smoothed second-derivative
  detector with an optional per-point threshold (`min_snr * sd`) and a
  strong-peak split/merge heuristic (merges a double-detection straddling one
  strong feature). **Port and clean this.**
- Its `PeakResult` return shape (freqs, intensities, indices).

What is lost (the refined `newfitting/` engine) is downstream of Stage 3 and
does not block it.

## Algorithm

Inputs: the FID + Stage 1 processing parameters (ComplexFT is recomputed
on demand) and the Stage 2 `NoiseResult` (per-point noise `sd`).

Two passes:

1. **Primary (windowed/apodized).** Compute the magnitude spectrum *with*
   apodization (clean, leakage-suppressed) and run the ported `locate_peaks`
   with `thresh = min_snr * sd`. Close lines smearing together is acceptable.
   This yields the robust coarse peak list.
2. **Gap pass (unwindowed).** In the spectral regions *not* covered by a
   primary detection, recompute the magnitude spectrum *without* apodization
   (full resolution) and detect again at the same SNR threshold to recover weak
   lines the apodization suppressed. This pass must be **masked by strong-peak
   leakage reach** so a strong line's sinc sidelobes are not detected as weak
   lines (see Open question O1).

Each detected peak is **classified by SNR only** into
`PeakClassification.{WEAK, MEDIUM, STRONG}` via two configurable thresholds
(`weak < t1 ≤ medium < t2 ≤ strong`), with `min_snr` as the detection floor.

The apodization used for pass 1 is a Stage 3 parameter, independent of the
unwindowed spectrum the downstream fit uses; default to the Stage 1
`expf_us`-equivalent.

## Data structures

Reuse the existing `core.data_structures.Peak`
(`frequency, intensity, index, snr, noise_std_local, classification`) and
`PeakClassification`. No new types expected; extend `Peak.properties` if a
detection-provenance field (which pass found it) is useful.

Stage output: an ordered list of classified `Peak`s.

## Pipeline integration

- **Stage key / dependencies:** add `stage3_peaks` to
  `PipelineStageTracker.STAGE_DEPENDENCIES` (depends on `stage1_complex_ft` and
  `stage2_noise_result`) and a `STAGE_DATA_PATHS` entry.
- **Serialization:** persist a `stage3_peaks` group — parallel arrays
  (frequency, intensity, index, snr, noise_std_local, classification) plus the
  detection parameters and provenance. Peaks are scientific output and may be
  hand-edited (below), so they are persisted (not recomputed on demand).
- **Manual-edit boundary:** because Stages 3 and 4 are intentionally separate
  (swappable detectors; human curation before window assignment), the persisted
  peak list must be straightforward to load, edit, and re-save between
  Stage 3 and Stage 4. Document this round-trip explicitly; a malformed/edited
  list must fail loudly, not silently.
- **Interfaces (all three, shared `_internal` impl):**
  `Pipeline.detect_peaks(...)` / `ftmwpipeline.api.detect_peaks(file, ...)` /
  CLI `detect-peaks`; visualization via `Pipeline.visualize_peaks(...)` and CLI
  `visualize-peaks` (names reserved in `CLI_STRATEGY.md`), overlaying classified
  peaks on the spectrum.

## Test plan

- **Unit:** ported `locate_peaks` against synthetic spectra with known peaks
  (including the strong-peak split/merge case); SNR classification bin edges;
  gap-pass logic; leakage masking (O1).
- **Serialization:** `stage3_peaks` round-trip; hand-edited-list round-trip.
- **Cross-interface consistency:** identical classified peak lists from CLI,
  Pipeline class, and functional API on experiment 2638 (mandatory per
  `TESTING_STRATEGY.md`).
- **Real data:** 2638, sanity vs. known lines; gap pass demonstrably recovers a
  weak line the windowed pass misses.

## Open questions / research

- **O1 — leakage-reach masking for the gap pass.** Need an analytic estimate of
  how far a strong peak's truncation sidelobes stay above noise
  (≈ function of its SNR, acquisition T, and an assumed/shared τ), to mask the
  unwindowed gap pass. This is the same estimate Stage 4 needs for window
  extents; design it once, here, and share it.
- **O2 — classification thresholds.** `t1`, `t2` defaults require empirical
  tuning on 2638; ship configurable with documented provisional defaults.
- **O3 — is the gap pass always needed?** If the windowed pass already clears
  the SNR threshold for weak lines in practice, the gap pass may be optional.
  Decide from O2/real-data evidence; keep it switchable.

## Downstream context (Stages 4–5, not in scope here)

Recorded so the design discussion is not lost; each gets its own planning doc
before its implementation.

- **Stage 4 — window definition (static).** Predict initial window extent from
  the strongest in-window line; greedily expand when another strong line
  appears before leakage decays; sanity-check edges by magnitude. *Research:*
  test for DC offsets at the edges of the complex FT as the true
  "reached baseline" criterion (worked out conceptually, never implemented).
  Uses the O1 leakage-reach estimate.
- **Stage 5 — per-window fit.** Demodulate the complex FT window to DC and
  **decimate to bandwidth** → an effective, much shorter time axis with an
  explicit (direction-sensitive) mapping back to real frequency; sideband sign
  matters (2638 is lower-sideband: DC ↔ 40960 MHz, increasing FID frequency →
  decreasing FT frequency). Model: sum of finite-T damped cosines; per-peak
  free **A, f, φ** (φ fully independent — chirp + amplifier phase, no inter-peak
  pattern); **τ shared** per window, default from apodization (≈ T/2 or T/3,
  user-controllable, bounded). **Least-squares residual is the complex FFT of
  the model vs. the complex FFT of the window** (identical point counts);
  time-domain residuals are display-only. Rationale: LS of oscillating
  time-domain functions is ill-conditioned, whereas the complex spectrum of a
  finite-T damped cosine is smooth and well-localized — and the finite-T model
  reproduces leakage exactly, so subtracting a fitted strong peak removes its
  leakage correctly. Conservative add-one-peak-with-significance-test loop,
  separation constraints, and validation port from
  `fit_weak_window_conservative_time_domain`; the lost `fit_time_domain_peaks`
  engine is recreated against that known contract. The discarded
  `iterative_peak_subtraction` (frequency-domain analytic sinc subtraction) is
  *not* revived. The demodulate→effective-time→frequency mapping and sideband
  direction must be derived in the Stage 5 doc and unit-tested against
  synthetic signals (known A, f, φ, τ on both sidebands) before any real-data
  fitting.

## Task breakdown

1. Port and unit-test `locate_peaks` (+ `PeakResult` shape) into `_internal`.
2. Design and unit-test the O1 leakage-reach estimator (shared with Stage 4).
3. Two-pass driver (windowed primary + masked unwindowed gap pass) + SNR
   classification.
4. `stage3_peaks` serialization + stage tracking + hand-edit round-trip.
5. Three interface wrappers + `detect-peaks`/`visualize-peaks` + visualization.
6. Cross-interface and real-data tests; tune O2 defaults; decide O3.
