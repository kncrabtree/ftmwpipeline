# Research reports

Standalone write-ups that document and justify load-bearing algorithmic
choices in the FTMW pipeline. Each report lives in its own subdirectory
with a self-contained reproducibility script (`prototype.py`) and the
figures it generates (`figures/`).

These are **archived findings**, not specifications. They explain *why*
the pipeline does what it does — the physics, the calibration, the
empirical sanity checks — at a level of detail that the normative
spec docs (`../*_STRATEGY.md`) and per-stage plans (`../planning/`)
deliberately omit. New reports are added when a load-bearing
algorithmic decision is made and want to be explained for future
readers.

## Available reports

- [Complex-edge coherence statistic](complex-edge-coherence/report.md) —
  how the windowing stage decides where one analysis window ends and
  the next begins. Derives the phase-coherent edge test, calibrates
  the null distribution and threshold, and verifies on synthetic
  ground truth and the 2638 fixture.
- [Peak-detection cost, correctness, and the sidelobe problem](peak-detection/report.md) —
  profiles the smoothed second-derivative detector, shows the
  unapodized false positives are sinc sidelobes that no cheap local
  test can suppress, vindicates the two-pass design, and verifies on
  2638 that the primary pass should apodize with a strong window.
- [The projection-coherence screen for peak detection](stage3-coherence-screen/report.md) —
  simulator-driven study of the σ-weighted Lorentzian-projection
  statistic as a low-SNR-vs-noise discriminator. Verdict: the screen
  works (matched-τ AUC ≥ 0.9 across the FWHM/bin ∈ [1, 2], SNR ≥ 2
  band, kills ~80 % of noise candidates at TPR 95 %), with a clear
  bin-vs-linewidth optimum at FWHM/bin ≈ 1–2 that translates to an
  acquisition-design principle (``T_active ≈ 3–5 · τ_eff``).
- [Matched-filter primary peak detection](matched-filter-detection/report.md) —
  tests whether replacing the production primary pass (BH window +
  Sav-Gol locator) with a Lorentzian matched filter (exp-apodized FFT
  + per-bin SNR threshold) followed by the projection screen is a
  net win. Verdict: the prompt's pure MF+screen design fails on 2638
  (screen has no discrimination on real data), but a hybrid (MF
  apodization + Sav-Gol concavity locator, no screen) dominates the
  production two-pass on synthetic data at every SNR/FWHM cell, and
  applied as the *gap pass* (replacing the unapodized FFT with the
  matched-filter active-FT, keeping the SavGol locator and leakage
  mask) is a clean local win. Recommended wiring: matched-filter gap
  pass, primary pass unchanged.
- [Noise estimation at extreme SNR (the leakage pedestal)](noise-snr-scaling/report.md) —
  the Stage 2 level-based estimator over-reports σ by up to ~6× on
  high-SNR, line-dense spectra because it measures the smooth leakage
  pedestal (summed far-wings of strong lines in the raw boxcar FT),
  not the noise. The error is a pure SNR-scaling failure: the pedestal
  is constant in shot count N while true noise falls as 1/√N. Validated
  a high-pass (scatter-based) replacement against line-free
  frame-difference noise and a cross-fixture 1/√N test (7 fixtures, two
  samples, SNR 10³–10⁶): the old estimator's error grows 1.4× → 6× with
  SNR and its slope goes from −0.47 to −0.02 (pedestal-pinned), while the
  scatter estimator stays within ~15 % of truth and tracks 1/√N. The 1/√N
  slope is the recommended acceptance invariant; a blank-FID series is
  the recommended definitive reference.

## Conventions for new reports

- One topic per subdirectory. Pick a timeless directory name (the
  algorithm, the question), not the stage number or the planning-doc
  open-question label.
- The report (`report.md`) is the only durable narrative artifact.
  Write timelessly: refer to pipeline stages by concept ("the
  windowing stage", "the noise-estimation stage"), not by number.
- Reproducibility script (`prototype.py`) regenerates every figure
  in `figures/` and reads any inputs from the conventional fixture
  locations (e.g. `scratch/exp_2638.ftmw`). Document missing-input
  fallbacks at the top of the script.
- Large intermediate artifacts (sweep `.npz` blobs > a few MB) stay
  out of git — the script is enough to regenerate them on demand.
  Only the figures the report references are committed.
