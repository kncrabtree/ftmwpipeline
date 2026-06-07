# Plan: Stage 2 — Noise estimation

Status: **implemented**. Implementation overview only — the algorithm is
landed in `src/ftmwpipeline/preprocessing/noise_estimation.py` and
exposed identically through the CLI (`estimate-noise`,
`visualize-noise`), `Pipeline.estimate_noise`, and `ftmw.estimate_noise`.

Normative requirements remain in `dev-docs/NOISE_STRATEGY.md`; this
document is the implementation summary.

## Objective

Produce a per-frequency RMS noise estimate `σ_x(f) = σ_c · √2` that the
downstream consumers (Stage 3 SNR thresholding, Stage 4 window planning,
Stage 5 chi²/weighting) divide by. Two non-trivial properties are required:

1. **Spectral resolution of real σ structure.** Real σ(f) on FTMW
   instruments varies on a ~few-hundred-MHz scale (bandpass shape,
   mixer/LO artifacts, frequency-dependent amplifier noise). The
   estimator must track that variation so downstream per-bin SNR
   thresholds are calibrated locally.
2. **Robustness to spectral lines and leakage.** Real high-SNR spectra
   contain strong lines whose far-wings sum into a smooth leakage
   *pedestal* that fills every quiet bin. A level-based estimator measures
   that pedestal, not the random noise, and over-reports σ by up to ~6× at
   SNR ~10⁵–10⁶ — a pure SNR-scaling failure (the pedestal is constant in
   shot count N while the noise averages down as 1/√N).

The shipped estimator (`estimate_noise_scatter`) meets both: it high-passes
the magnitude to separate the smooth pedestal from the white per-bin scatter,
and a region-aware Rician correction recovers the underlying complex σ.

## Where it is measured (the grid)

Stage 2 measures and persists σ on the **canonical active FT** — the
`dt_us·rfft` of just the `[start_us, end_us]` active region, trimmed to the
analysis band — not the front-zeroed full-record persisted FT. The active FT
is the single grid every later stage scores, plans, and fits on; the
full-record FT is a Stage 0/1 start-time comparison view only. The active FT is
built by `_internal/active_ft_support.build_trimmed_active_ft`, and the
σ-on-it by `build_active_grid_with_noise` (Stages 3/4/5 consume the same
builder). See [`stage2-noise-authority.md`](stage2-noise-authority.md).

## Algorithm

`estimate_noise_scatter(frequencies, magnitudes, **knobs)` in
[`src/ftmwpipeline/preprocessing/noise_estimation.py`](../../src/ftmwpipeline/preprocessing/noise_estimation.py)
(the function docstring carries the full step-by-step; summary here).

1. **Pedestal.** Estimate the smooth leakage pedestal as a broad running median
   of `|X|` (width `pedestal_mhz`). Line bins are interpolated over before each
   median pass so strong-line power does not pull the pedestal up near lines.
2. **High-pass + self-mask.** The residual `|X| − pedestal` is white where there
   is only noise. Flag bins whose residual exceeds `line_k` robust-σ as lines and
   iterate (`n_iter` passes) to refine the self-mask — Stage 2 precedes Stage 3,
   so there is no peak list to lean on.
3. **Region-aware scatter → σ.** In each sliding `window_mhz` region take the MAD
   of the residual over the surviving non-line bins (the scatter) and convert it
   to the underlying complex σ. With `region_aware` the conversion uses the
   Rician lookup `C(R)` (`R = scatter / pedestal`, the dimensionless regime
   indicator, monotone from ~1.0 under strong lines to ~1.47 in quiet Rayleigh
   regions); otherwise a fixed mid-regime factor.
4. **Smooth.** Interpolate σ across region centres and line positions, then smooth
   in two passes: a broad moving percentile (`smoothing_mhz`, median by default —
   a lower-envelope that rides the floor through line-dense bands) followed by a
   Gaussian convolution (`convolve_mhz`) that removes the median's staircase
   without re-inflating under lines.

The output is the per-bin complex-RMS `σ_x = σ_c · √2`. Rank-filter windows
(pedestal median, smoothing percentile) are clamped to the data length so a
coarse grid — where a physical-MHz width spans more bins than exist — does not
drive an O(N·window) blow-up.

## Tunables and instrument dependence

The knobs are the flat `NoiseSettings` fields (`stage2.<field>`), resolved on
the standard four-layer chain. Defaults are calibrated on the 2638 fixture.

| Field | Role | default |
|---|---|---|
| `window_mhz` | per-region scatter-MAD window width | 80.0 |
| `pedestal_mhz` | running-median width isolating the leakage pedestal | 20.0 |
| `line_k` | robust-σ multiple above which a residual bin is a line | 8.0 |
| `n_iter` | self-mask refinement iterations | 3 |
| `region_aware` | use the Rician `C(R)` lookup vs a fixed factor | True |
| `smoothing_mhz` | broad moving-percentile σ smoothing width (0 = off) | 800.0 |
| `smoothing_percentile` | 50 = median (unbiased); lower = lower-envelope | 50.0 |
| `convolve_mhz` | Gaussian σ of the step-removing 2nd pass (0 = off) | 200.0 |

The `C(R)` table and the `σ_x = σ_c·√2` quadrature factor are analytic; do not
retune. Instrument-family guidance is in
[`instrument-tunable-knobs.md`](instrument-tunable-knobs.md).

## Data structures

`NoiseResult` (dataclass in `noise_estimation.py`):

```python
rms_noise:   np.ndarray   # σ_x per frequency point (on the active-FT grid)
noise_mask:  np.ndarray   # boolean: True where the bin counts as noise
bin_info:    dict         # diagnostics
```

`bin_info` (from `_scatter_bin_info`) carries `algorithm =
"scatter_highpass_region_aware"`, the resolved knob values, `noise_fraction`,
and the line/region counts.

## Pipeline integration

Cross-interface (`pipeline.py`, `api.py`, `cli/noise_commands.py`):

- `Pipeline.estimate_noise(...)`
- `ftmwpipeline.api.estimate_noise(path, ...)`
- `ftmwpipeline estimate-noise <path> [...]`
- `ftmwpipeline visualize-noise <path>` (overlays σ on the active FT)

All three converge on `_internal/stage2_impl.compute_noise_estimation_impl`,
which builds the trimmed active FT, runs `estimate_active_ft_noise` (the
sort→`estimate_noise_scatter`→un-sort wrapper) with the resolved knobs, and
persists via `save_noise_result_to_hdf5`. Knobs resolve through
`core/noise_settings.py` (`NoiseSettings`) on the four-layer chain (see
[`settings-backfill.md`](settings-backfill.md)).

Stage tracker: `stage2_noise_result`, depends on `stage1_complex_ft`.

## Serialization

`src/ftmwpipeline/io/noise_result_serialization.py`. The `.ftmw` file stores:

- `signal_indices` (positions where `noise_mask == False` — compact vs the full
  boolean)
- `rms_noise_full` (the σ array, verbatim, gzip-compressed)
- `bin_info/*` (the diagnostics dict) and `algorithm_info` (`method =
  "verbatim_sigma"`)

Load rebuilds the canonical active FT for the grid, reconstructs the mask from
`signal_indices`, and reads `rms_noise_full` back element-for-element — bit-exact
by construction. (The earlier convolution/polynomial reconstruction paths existed
only for the retired adaptive estimator and were removed with it.)

## Provenance

The scatter estimator's derivation, the 1/√N validation, and the region-aware
`C(R)` calibration against frame-difference truth are documented in
[`research/noise-snr-scaling/report.md`](../research/noise-snr-scaling/report.md)
(prototype at `research/noise-snr-scaling/prototype.py`). The retired
level-based "adaptive" estimator it replaced — MAD/median subdivision +
skewness-trim noise mask + moving-median σ + Lorentzian-skirt exclusion — is
preserved as a minimal comparison reference at
[`research/noise-snr-scaling/legacy_adaptive.py`](../research/noise-snr-scaling/legacy_adaptive.py).
Why that level/adaptive family was retired (the heuristic-tuning and
grid-invariance audits, and the SNR-scaling failure that overtook both) is in
[`research/noise-snr-scaling/report.md`](../research/noise-snr-scaling/report.md) §3.1.
