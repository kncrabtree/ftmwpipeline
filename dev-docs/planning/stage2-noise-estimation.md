# Plan: Stage 2 — Noise estimation

Status: **implemented**. Implementation overview only — the algorithm is
landed in `src/ftmwpipeline/preprocessing/noise_estimation.py` and
exposed identically through the CLI (`estimate-noise`,
`visualize-noise`), `Pipeline.estimate_noise`, and `ftmw.estimate_noise`.

Normative requirements remain in `dev-docs/NOISE_STRATEGY.md`; this
document is the implementation summary.

## Objective

Produce a per-frequency RMS noise estimate `σ_x(f) = σ_c · √2` on the
persisted FT magnitude spectrum. Two non-trivial properties are
required by the downstream consumers (Stage 3 SNR thresholding, Stage 5
chi²/weighting):

1. **Spectral resolution of real σ structure.** Real σ(f) on FTMW
   instruments varies on a ~few-hundred-MHz scale (bandpass shape,
   mixer/LO artifacts, frequency-dependent amplifier noise). The
   estimator must track that variation so downstream per-bin SNR
   thresholds are calibrated locally.
2. **Robustness to spectral lines.** Real spectra contain strong lines
   whose Lorentzian skirts extend tens of MHz on either side of the
   peak. Skirts contaminate any local noise statistic that includes
   them; a naïve moving-mean or moving-RMS gets biased high in line-
   dense regions and produces spurious "bumps" in σ(f) that follow the
   lines, not the underlying noise physics.

The shipped algorithm meets both: a subdivision-based bin layout
identifies σ-regimes, a moving-median (robust statistic) tracks σ(f)
continuously at an instrument-tunable smoothing scale, and an
explicit physical model excludes the contiguous Lorentzian-skirt
neighborhoods of strong lines.

## Algorithm

`estimate_noise_adaptive(frequencies, magnitudes)` in
[`src/ftmwpipeline/preprocessing/noise_estimation.py`](../../src/ftmwpipeline/preprocessing/noise_estimation.py).

Five stages, executed in order.

### 1. Recursive bin subdivision (median + MAD on raw |X|)

The magnitude spectrum is recursively bisected. A region is split iff
both halves still satisfy the bin-size floor AND
```
|Δmedian(|X|)| / median(|X|) ≥ SUBDIVISION_THRESHOLD  OR
|ΔMAD(|X|)|    / MAD(|X|)    ≥ SUBDIVISION_THRESHOLD
```
on the **raw** magnitudes of the two halves. Median and MAD are
robust to spectral-line outliers up to 50 % bin contamination — the
prior `post-trim mean / variance / z-test / F-test` OR-of-four
criterion was confounded by the skewness trim's pre-pass and could
not detect real σ heterogeneity (see
[`research/noise-grid-invariance/report.md`](../research/noise-grid-invariance/report.md)
§4 for the diagnosis).

Constants: `SUBDIVISION_THRESHOLD = 0.08` (calibrated for 0 % false-
positives on homogeneous noise, 94.6 % detection at σ_R/σ_L = 1.10);
`ABS_MIN_BIN_SIZE = 300` (the relative floor `n_points / 64`
dominates on real spectra).

### 2. Per-bin skewness-trim noise mask

Inside each final bin, the magnitudes are trimmed from the high-
magnitude tail in 1 %-rank steps until the kept distribution's sample
skewness drops below the Rayleigh value (`skew_target = 0.6311`). The
kept indices form the initial `noise_mask`. Same trim as the prior
estimator; the change is that it now runs **only on final bins**, not
at every candidate subdivision split (small perf win).

### 3. Moving-median σ_x (first pass)

`compute_rms_noise_convolution(frequencies, magnitudes, noise_mask, bl_bin)`
slides a median filter of width `bl_bin` along the noise-mask
sequence, scales each median via the Rayleigh quantile relation
`σ_x = median(|X|) · √(1 / ln 2) ≈ 1.2011 · median(|X|)`, and
interpolates back to the full frequency grid via `numpy.interp`. The
function name is historical — it used to compute moving-RMS —
preserved for backward compatibility of the public signature.

The median is robust to contamination up to 50 % of the local window.
For a 300 MHz window on the 2638 user grid (~25 000 noise samples), a
single line's skirt contaminates at most a few percent of the
window's samples, well below the breakdown point.

`bl_bin = DEFAULT_SMOOTHING_MHZ / freq_step` (default 300 MHz). See
*Tunables* below.

### 4. Strong-line skirt exclusion

The moving median handles isolated outliers but residual upward bias
persists near strong lines because the skirt is a *contiguous* region
of moderately-elevated samples that individually pass the skewness
trim. The physical fix:

The FT magnitude of an exp-damped sinusoid is Lorentzian,
`|X|(Δf) = X_peak · γ / √(Δf² + γ²)` (γ = HWHM). Far from the line
centre this reduces to `|X|(Δf) ≈ X_peak · γ / |Δf|`, so the skirt
drops below `k · σ_x` at
```
Δf_exclude = γ · (X_peak / σ_x) / k = γ · SNR / k
```

`_exclude_strong_line_skirts`:

1. Find local maxima in `|X|` with `|X| / σ_x_initial > STRONG_PEAK_SNR`.
2. Measure HWHM `γ` from the strongest peak's full-width-half-max via
   `scipy.signal.peak_widths(rel_height=0.5)`. Assumes roughly
   uniform line widths across the spectrum — adequate for typical
   FTMW data; per-peak FWHM would be more accurate but is not done.
3. For each strong peak, mask out the `±Δf_exclude` neighborhood
   from the noise mask, capped at `MAX_SKIRT_EXCLUSION_MHZ` against
   pathological super-strong peaks.

Constants: `STRONG_PEAK_SNR = 20.0`, `SKIRT_EXCLUSION_K = 1.5` (skirt
drops to 1.5·σ_x at the exclusion radius — aggressive enough to
flatten residual bias bumps without sacrificing too much noise mask),
`MAX_SKIRT_EXCLUSION_MHZ = 500.0`.

### 5. Final moving-median σ_x

Recompute `compute_rms_noise_convolution` on the skirt-refined noise
mask. This is the production `rms_noise` returned in the
`NoiseResult`.

## Tunables and instrument dependence

The default constants are calibrated on the 2638 fixture (BlackChirp,
13.5 GHz active bandwidth, expf=5 µs apodization,
linewidth ≈ 0.4 MHz FWHM). The main instrument-tunable parameter is:

- **`DEFAULT_SMOOTHING_MHZ = 300.0`** — the moving-median window width
  in physical MHz. Anchored in MHz so the σ_x estimate has the same
  spectral resolution across different zero-padding choices. Choose
  to match the instrument's actual σ(f) coherence scale: take a no-
  signal acquisition, view the unwrapped noise floor, and pick the
  smallest window that flattens σ_x bumps near strong lines without
  smoothing across real σ structure. Wider windows give better
  Rayleigh-RMS stability but track less σ(f) detail; narrower
  windows track finer features but leave more skirt-leakage bumps.

Secondary tunables (calibrated against 2638; defaults usually
correct but worth sanity-checking on a new instrument):

| Constant | Role | 2638 default |
|---|---|---|
| `SUBDIVISION_THRESHOLD` | median/MAD subdivision sensitivity | 0.08 |
| `ABS_MIN_BIN_SIZE` | smallest allowed bin (samples) | 300 |
| `STRONG_PEAK_SNR` | SNR floor for skirt-exclusion candidacy | 20.0 |
| `SKIRT_EXCLUSION_K` | skirt amplitude / σ_x at the exclusion radius | 1.5 |
| `MAX_SKIRT_EXCLUSION_MHZ` | per-peak exclusion cap | 500.0 |

The `RAYLEIGH_MAD_TO_SC = 0.4485` and the `1.2011 ≈ √(1/ln 2)` factor
are analytic constants of the Rayleigh distribution; do not retune.

## Data structures

`NoiseResult` (dataclass in `noise_estimation.py`):

```python
rms_noise:   np.ndarray   # σ_x per frequency point
noise_mask:  np.ndarray   # boolean: True where the bin counts as noise
bin_info:    dict         # diagnostics (see below)
```

`bin_info` keys produced by the as-shipped estimator:

- `bin_edges` — array of subdivision indices
- `n_bins` — number of final bins
- `algorithm = "mad_median_subdivision"`
- `noise_fraction` — fraction of points in `noise_mask`
- `smoothing_window_mhz`, `smoothing_window_points`
- `skirt_excluded` — count of points removed by the skirt-exclusion
  refinement
- `skirt_line_hwhm_mhz` — γ measured from the strongest peak

## Pipeline integration

Cross-interface (`pipeline.py`, `api.py`, `cli/noise_commands.py`):

- `Pipeline.estimate_noise(...)`
- `ftmwpipeline.api.estimate_noise(path, ...)`
- `ftmwpipeline estimate-noise <path> [...]`
- `ftmwpipeline visualize-noise <path>` (interactive matplotlib or
  saved PNG/HTML)

All three converge on `_internal/stage2_impl.compute_noise_estimation_impl`
which calls `estimate_noise_adaptive` and persists via
`save_noise_result_to_hdf5`.

Stage tracker: `stage2_noise_result`, depends on `stage1_complex_ft`.

## Serialization

`src/ftmwpipeline/io/noise_result_serialization.py`. The `.ftmw` file
stores:

- `signal_indices` (positions where `noise_mask == False`, ~60 %
  storage reduction vs storing the full boolean)
- `smoothing_params/smoothing_window_points` + `bin_edges` (under
  `bin_info/`)
- `bin_info/*` attrs (the dict above)
- 8th-order polynomial fit of `rms_noise` as a fallback (~104 bytes)

Round-trip path: `_reconstruct_rms_via_convolution` replays the same
`compute_rms_noise_convolution` (moving-median) on the deserialized
noise mask and window size. Bit-perfect by construction. The
polynomial fallback handles older `.ftmw` files missing the
`smoothing_params` group.

## Provenance

Original audit and calibration:
[`research/noise-heuristic-audit/`](../research/noise-heuristic-audit/report.md)
(2026 prior work) — set `skew_target = 0.6311`, `ABS_MIN_BIN_SIZE =
300`, the `DEFAULT_SMOOTHING_SAMPLES = 2500` predecessor of the
current MHz-based default.

Grid-invariance audit and the MAD-subdivision fix:
[`research/noise-grid-invariance/report.md`](../research/noise-grid-invariance/report.md).
§4 diagnoses the prior estimator's failure on the 2638 active-FT
(degenerated to 1 bin); §5 prototypes the MAD subdivision criterion;
§8 documents the as-shipped algorithm including the moving-median σ
and Lorentzian-skirt exclusion that landed in the implementation
phase. The §9 open questions point to what may need re-tuning on
other instruments.
