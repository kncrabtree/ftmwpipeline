# Stage 2 noise authority + Stage 3 gap-pass matched filter

Implementation summary for the "Stage 2 is the single noise authority / no user
grid" work and its final open item, the Stage 3 gap-pass matched-filter
reformulation. Both have shipped.

## Noise authority + no-user-grid (shipped)

- **The active FT is the single grid.** Stage 2 measures + persists σ on the
  canonical unapodized active FT (trimmed to the analysis band); Stages 3
  (snap/score), 4 (window planning), and 5 (fit weighting + structural replan +
  fit overlay) all consume it. The front-zeroed full-record FT is a Stage 0/1
  start-time comparison view only — never a scoring/detection/planning/fit
  domain. There is no "user grid".
- **Shared builders** live in `_internal/active_ft_support.py`:
  `build_trimmed_active_ft`, `build_active_grid_with_noise` (active FT + its
  scatter σ on the same grid), and `compute_canonical_active_ft` /
  `estimate_canonical_active_ft_noise`. The pure estimator wrapper is
  `preprocessing/noise_estimation.py::estimate_active_ft_noise`.
- **Stage 3 snap** is `_snap_to_active_grid`: detections snap onto the active
  FT and are scored against the active-FT σ.
- **Retired:** `estimate_noise_adaptive` and the full-record canonical-noise
  loader `_load_canonical_noise`. `estimate_noise_scatter` is the sole
  estimator; a minimal level-based reference lives at
  `../research/noise-snr-scaling/legacy_adaptive.py`. D9 in `ROADMAP.md` is
  amended.

## Stage 3 gap-pass matched filter (shipped)

The gap pass's detection spectrum was always exponentially apodized, even on
Gaussian instruments, and its per-bin σ was a third independent scatter estimate
on that spectrum. Both are resolved in `_internal/stage3_impl.py`:

- **Shape-aware matched window.** `_mf_gap_spectrum` now applies the line
  shape's matched window — `exp(-t/τ)` (Lorentzian) or `exp(-(t/τ)²)` (Gaussian),
  selected by the Stage 2b `recommended_shape` the same way Stage 5 reads it.
  This is the exact time-domain matched filter for the line shape; on the
  Gaussian 2638 fixture it lifts gap-pass weak-line recovery ~57% with all
  strong lines preserved (their apexes shift by ~1 active-grid bin, the
  exponential filter's position bias being corrected) and no false-positive
  inflation. The 655/1512 Lorentzian path is byte-for-byte the old exponential
  window (strict equivalence).

- **Analytic noise propagation (two noise sources, not three).** The gap σ is no
  longer re-estimated. The matched filter is a linear transform of the active
  region, so under white noise its per-bin σ is the active-FT authority σ scaled
  by the window gain `√(Σw²/N)` — propagated onto the gap grid by
  `_propagate_active_sigma_to_grid`. Verified on synthetic white noise (on the
  real 2638 grids) to reproduce a direct scatter estimate within 1–4%; the
  ~5–14% divergence seen on dense real spectra is leakage contamination of the
  *direct* estimate, which the propagated floor correctly excludes (coherent
  leakage is handled by the separate leakage-aware floor by design). The gap σ
  is detection-threshold-only — final SNR comes from the active-grid snap-back.

### Key decisions

- **Time-domain build, not the originally-proposed frequency-domain kernel
  convolution.** A bit-exact frequency-domain matched filter is just the
  time-domain operation routed through an extra FFT pair (no architectural win),
  so any frequency-domain version is a truncated-kernel *approximation* needing
  noise-color revalidation. The two axes were decoupled: keep the exact
  time-domain window (made shape-aware) and get the single-noise-source goal via
  analytic propagation, which is a property of the matched-filter operation
  independent of build domain. Result: exact filter + two noise sources, no
  approximation.

- **SavGol window left unchanged (problem-3 premise falsified).** The plan
  proposed re-sizing the SavGol window to the wider post-MF feature FWHM.
  Empirically this regressed weak-line recovery on *both* 655 and 2638:
  `detect_peaks` derives the apex-snap radius from `gap_sg_window`, so enlarging
  it over-merges nearby peaks. The existing `_SG_FWHM_COVERAGE=4.0` was already
  calibrated against the realised matched-filter gap grid (`sg_window≈13`), so
  the validated coverage is retained as-is.

- **Docstring contract reconciled.** `detect_peaks` and the `stage3_impl` module
  docstring no longer describe the gap spectrum as "unapodized"; it is the
  shape-aware matched filter, with `detect_peaks` itself making no assumption
  about how either spectrum was built.

## Primary pass: single-source collapse rejected (deferred sibling)

The deferred sibling — collapsing the **primary** pass onto the single active-FT
authority too — was investigated and **rejected**. Both routes regress:

- **Move the primary to the active-region frame** (clean window-gain
  propagation): the Blackman-Harris spectrum's `S_coh` runs ~7× higher in the
  active frame than in the full-record-de-ramped frame, so
  `PRIMARY_LEAKAGE_FLOOR_K=1.0` over-suppresses and ~halves the primary list.
- **Keep the full-record BH spectrum, propagate its noise** (convention factor
  `10^units_power·√(Σw²)/(N_total·dt·√N_active)`): matches a direct scatter
  estimate within a few % on white noise, but on dense real data the propagated
  σ is +26% on 655 (+4% on 2638) and loses ~41% of promoted lines.

**Why:** the BH primary spectrum genuinely has a *lower, cleaner* noise floor —
BH apodization suppresses the truncation leakage that inflates the boxcar
active-FT authority σ on a line-dense spectrum. Propagating the authority σ
over-estimates the primary's true floor. The primary's leakage-suppressed
spectrum must have its noise measured **on that spectrum**.

**The primary therefore keeps a dedicated noise floor — but it needs a proper
home.** Today that floor is an inline `estimate_noise_scatter` on the
full-record BH spectrum inside `stage3_impl` (a stopgap, not part of the
noise-authority model). The follow-up is to **calibrate and persist a second
Stage 2 noise level** for the apodized / leakage-suppressed domain — the same
canonical scatter machinery as the active-FT authority, but on the primary's
spectrum — so the primary consumes a first-class Stage 2 quantity rather than an
ad-hoc estimate. **This is a prerequisite for assessing any Stage 3 knobs** (the
tune surface needs a stable, calibrated noise definition to score against).

## Out of scope / unchanged

- **Stage 2b keeps its own FID-tail σ reference** for τ extraction (see
  `stage2b-tau-calibration.md`); "Stage 2 is the noise authority" applies to the
  frequency-domain σ(f) consumers (Stages 3/4/5).
