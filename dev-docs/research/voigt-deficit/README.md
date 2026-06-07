# Voigt-deficit research

Investigates whether the high residual chi-squared on Stage 5 shape-error windows
(and on strong STFT contributor bins) is explained by a Voigt line shape — a
convolution of Lorentzian (`tau_L`) and Gaussian (`tau_G`) decays — rather than
the pure-exponential model the production pipeline uses.

Two independent analyses, each using the same 2638 fixture and unapodized active FT.

## Input fixture

Both scripts consume a Stage-5-fitted pipeline file:

```
scratch/stage5-tau-calibration-lsq/exp_2638_unapodized.ftmw
```

This is a gitignored scratch artifact. If it is missing, rebuild it from the
repository root:

```python
import ftmwpipeline.api as ftmw
from ftmwpipeline.core.stage_fit_settings import StageFitSettings
from ftmwpipeline.core.tau_calibration_settings import TauCalibrationSettings

fpath = "scratch/stage5-tau-calibration-lsq/exp_2638_unapodized.ftmw"
ftmw.import_data(fpath, source="examples/blackchirp_data/2638/", force=True)
ftmw.compute_ft(fpath, trim=(26500, 40000))
ftmw.estimate_noise(fpath)
tau_s = TauCalibrationSettings()
tau_s.band.compute_band_majorities = True
ftmw.calibrate_tau(fpath, settings=tau_s)
ftmw.detect_peaks(fpath)
ftmw.assign_windows(fpath)
fit_s = StageFitSettings()
fit_s.tau.tau0_us = 6.325   # T_active / 2
ftmw.fit_peaks(fpath, settings=fit_s)
```

The above recipe is also documented in the header of
`dev-docs/research/stage5-tau-calibration/lsq_comparison.py`.

## Part A — per-window joint (tau_L, tau_G) LSQ

`part_a_perwindow.py` selects Stage 5 shape-error candidate windows
(persisted `reduced_chi2 > 10`, `tau_fitted == 1`, `n_peaks >= 2`) and
re-fits each window's active-FT slice with:

1. A **baseline** pure-exponential model (tau free, no Stage 2b prior).
2. A **Voigt** model with shared `(tau_L, tau_G)` per window, multi-start on
   `tau_G`.

The comparison isolates the Voigt shape benefit from the prior anchoring: the
same per-peak seeds are used for both, and the baseline tau is free.

The active FT used here is **unapodized** — consistent with what Stage 5 fits
on. Apodizing before the fit would convolve in the window function's spectral
response and bias the (tau_L, tau_G) recovery, so it is intentionally omitted.

Run from the repository root:

```bash
conda run -n ftmwpipeline-dev python \
    dev-docs/research/voigt-deficit/part_a_perwindow.py
```

Outputs:

- `data/part_a_perwindow_summary.json` — per-window stats and global summary
- `data/part_a_perwindow_detail.csv` — per-window records
- `figures/03_voigt_vs_exp_chi2_perwindow.png` — chi2_r scatter (baseline vs Voigt)
- `figures/04_perwindow_tauG.png` — recovered (tau_L, tau_G) per window

## Part B — per-bin Voigt fits on STFT time series

`part_b_perbin.py` works independently of Part A. It tests whether the per-bin
`|S_n(a)|` time evolution across Stage 2b STFT frames is consistent with a
Voigt envelope vs pure-exponential decay.

Two bin pools:

- **Pool 1 (bad-fit, hi-SNR)**: STFT classification 2, SNR > 100. These are
  the original planning-doc target. On 2638 the hypothesis is contradicted —
  bad-fit bins are dominated by line-blend interference (non-monotonic
  `|S_n(a)|`) and no monotonic Voigt fits them either.
- **Pool 2 (contributor)**: STFT classification 3, SNR > 20. Voigt beats
  pure-exp by 2–5x on the strongest bins. The per-band `tau_G` calibration
  artifact produced here feeds Part A as a soft prior.

Run from the repository root:

```bash
conda run -n ftmwpipeline-dev python \
    dev-docs/research/voigt-deficit/part_b_perbin.py
```

Outputs:

- `data/part_b_perbin_summary.json` — per-band stats and global diagnostics for
  both pools
- `data/tau_G_band_majorities.json` — per-band `(tau_G, sigma_tau_G, n)` from
  the calibration-eligible contributor subset (Part A prior input)
- `data/part_b_perbin_detail.csv` — per-bin records (both pools)
- `figures/01_voigt_vs_exp_chi2.png` — chi2_r scatter, one panel per pool
- `figures/02_tauG_vs_freq.png` — recovered `tau_G` vs frequency on the
  contributor pool, with per-band medians overlaid
