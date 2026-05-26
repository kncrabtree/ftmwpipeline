# Gaussian-shape Stage 5 comparison harness

Drives the production `fit_peaks(shape='lorentzian')` vs
`fit_peaks(shape='gaussian')` head-to-head on a single `.ftmw` fixture
and emits the artefacts the planning-doc acceptance gate evaluates.

See [`planning/stage5-gaussian-shape.md`](../../planning/stage5-gaussian-shape.md)
for the design motivation and acceptance criteria. The Gaussian path is
the production response to the Voigt-deficit finding in
[`planning/stage5-voigt-deficit.md`](../../planning/stage5-voigt-deficit.md)
that the joint `(τ_L, τ_G)` LSQ on 2638 degenerates to Gaussian-dominant.

## Usage

From the repository root:

```bash
conda run -n ftmwpipeline-dev python \
    dev-docs/research/gaussian-shape/compare_shapes.py \
    <fixture.ftmw> [--work-dir scratch/gaussian-shape-compare]
```

The fixture must have Stages 0-4 completed. The script:

1. Copies the fixture into per-shape working files (Lorentzian / Gaussian)
   under `--work-dir` (default `scratch/gaussian-shape-compare/`).
2. Ensures the Gaussian copy has `/stage2b_tau_G_calibration` — runs
   `calibrate_tau_G(...)` if absent.
3. Runs `fit_peaks(shape='lorentzian')` on the Lorentzian copy.
4. Runs `fit_peaks(shape='gaussian')` on the Gaussian copy.
5. Aligns the two `SpectrumFit` aggregates by `window_id` (warning on
   mismatch from divergent replan revisions) and matches per-peak pairs
   by nearest-frequency within ±0.5 MHz.

## Outputs

Under `dev-docs/research/gaussian-shape/`:

- `data/<stem>_summary.json` — aggregate stats: median/p95 χ²ᵣ per shape,
  median ΔAIC, peak-count totals, matched-peak frequency-residual /
  amplitude-ratio stats, plus per-window records for the four Part A
  shape-error windows (w141, w213, w310, w355) flagged with
  `gaussian_preferred` (ΔAIC > 5).
- `data/<stem>_per_window.csv` — one row per shared window (id, freq
  range, peak counts, χ²ᵣ, AIC for each shape, ΔAIC).
- `data/<stem>_per_peak.csv` — matched peak pairs (window, frequencies,
  amplitudes, SNRs for each shape).
- `figures/<stem>_panel.png` — 2-panel comparison figure: per-window
  χ²ᵣ scatter (log-log) with Part A windows ringed in red, per-peak
  frequency residual vs Lorentzian SNR.

## Acceptance criteria

From the planning doc (validated on the 2638 unapodized fixture first;
second fixture for cross-fixture generalisation):

- Median Gaussian χ²ᵣ < median Lorentzian χ²ᵣ.
- 95th-percentile Gaussian χ²ᵣ < 5 (vs ~50+ Lorentzian tail).
- Shared-line frequency agreement within the per-peak error bar (no
  shape-induced centre bias).
- ΔAIC > 5 in favour of Gaussian on each of w141, w213, w310, w355
  (the Part A shape-error windows).
