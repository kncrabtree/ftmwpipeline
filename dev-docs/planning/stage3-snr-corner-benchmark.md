# Stage 3 cross-fixture benchmark + the SNR "corner" detection threshold

Next-session brief. Two coupled goals:

1. **Benchmark Stage 3 peak detection across all seven fixtures** now that the
   **scatter** noise estimator is the default (replacing adaptive). The noise
   floor feeding every SNR is now honest (no leakage-pedestal inflation), so
   every SNR-dependent Stage 3 decision must be re-assessed.
2. **Replace the fixed SNR threshold with a data-driven "corner"** found from the
   peak-count-vs-threshold curve against a Rayleigh-noise model — and decide
   whether one threshold serves all SNR regimes or whether detection must become
   SNR-adaptive.

The larger frame is **cross-fixture algorithm performance**: the seven fixtures
span ~3 orders of magnitude in SNR (363 ≈ 6×10², 2638 ≈ 7×10², 360 ≈ 3×10³,
1231 ≈ 2×10³, 1019 ≈ 5×10⁴, 1512 ≈ 1×10³, 655 ≈ 1×10⁵). If detection/aggregation
behaves *dramatically* differently across regimes, that is the signal to explore
new approaches rather than re-tune one pipeline.

## Where Stage 3 stands today

- `src/ftmwpipeline/preprocessing/peak_detection.py` — `detect_peaks(...)`,
  `locate_peaks(...)`, `classify_by_snr(...)`. SNR is `|X| / rms_noise` (the
  per-bin scatter σ_x now).
- Thresholds are **fixed constants**, calibrated on 2638 only:
  - `DEFAULT_MIN_SNR = 3.0` — user-facing promotion cutoff (which peaks go to Stage 4).
  - `DEFAULT_INTERNAL_MIN_SNR = 2.0` — internal detection floor. Its docstring is
    **already a corner observation**: "detecting at 3.0 loses ~190 peaks that clear
    3.0 on the user grid; ~2.0 recovers them and then *plateaus* — below 2.0 is
    almost pure noise, no extra survivors." That plateau-then-noise-wall *is* the
    corner; today it is frozen at one value from one fixture.
  - `DEFAULT_WEAK_MEDIUM_SNR = 10`, `DEFAULT_MEDIUM_STRONG_SNR = 50` — classification.
- Stage 3 internals (gap-pass matched filter, zpf for position-finding) are in
  `_internal/stage3_impl.py`; see `dev-docs/planning/stage3-peak-detection.md`
  and memory `stage3-mf-gap-pass-shipped`.
- **Estimator mismatch to resolve.** `stage3_impl.py` calls
  `estimate_noise_adaptive` **directly** for its internal gap/active-grid
  detection noise (lines ~600/603/771), independent of the persisted Stage 2
  result — which is now scatter. So today Stage 3 scores detection on *adaptive*
  noise (on zpf'd active/gap grids) while downstream SNR/promotion uses the
  persisted *scatter* σ. First task of the benchmark: decide whether those
  internal calls should also become scatter (likely yes, for consistency), and
  measure the effect. Note scatter's broad-median smoothing assumes a
  raw-grid-scale noise floor; on the narrow zpf'd active/gap windows the
  smoothing widths may need rescaling (they are MHz-based, so they adapt, but
  validate).

## The corner method (the idea to formalize)

Sweep a per-bin SNR threshold `s` (SNR ≡ `|X| / σ_x`) and count detected local
maxima `N(s)`. Decompose:

- **Real peaks** form a *plateau*: once `s` drops below a real line's SNR it is
  detected and stays detected, so `N_real(s)` rises then saturates at the true
  line count as `s → 0`.
- **Noise false positives explode exponentially.** For magnitude noise (Rayleigh,
  scale σ_c = σ_x/√2), `P(|X| > s·σ_x) = exp(-s²)`. The expected number of noise
  bins over threshold is `≈ N_eff · exp(-s²)`; for *local maxima* it is that times
  a maxima-density factor (peaks are rarer than crossings, and the FT is
  oversampled/correlated — especially with any zpf — so use an **effective**
  independent-bin count, not the raw bin count).

`N(s) = N_real(s) + N_noiseFP(s)`. The **corner** is where the gentle real-peak
plateau gives way to the steep noise exponential — i.e. where `N_noiseFP(s)`
starts to dominate `dN/ds`. Set the detection floor at (or just above) the corner.

Two ways to locate it, do both and compare:
- **Model crossover**: solve for `s*` where `N_eff·exp(-s²)` equals a chosen
  false-positive budget (e.g. ≤ 1 expected FP over the band, or a fixed fraction
  of `N_real`). Needs `N_eff` (calibrate from the line-free regions: count noise
  maxima vs `s` and fit `exp(-s²)` to recover the maxima-density prefactor — this
  is a clean self-consistency check on the scatter σ too).
- **Empirical knee**: Kneedle / max-curvature on `log N(s)` vs `s`. The repo
  already uses a Kneedle elbow in start-detection
  (`preprocessing/start_detection.py`) — reuse the pattern.

The scatter σ_x makes this honest: with the old pedestal-inflated σ, SNR was
compressed under lines and the corner washed out. Validate the Rayleigh model
directly on the **line-free** bins per fixture (the noise-FP count vs `s` must
follow `N_eff·exp(-s²)` — if it doesn't, the σ or the independence assumption is
off, which is itself a finding).

## Concrete plan for the session

1. **Harness**: for each of the 7 fixtures build the canonical pipeline —
   `import_data → detect_start_time(band=(26500,40000)) → compute_ft(zpf=0,
   expf_us=None, trim=(26500,40000)) → estimate_noise()` (now scatter by
   default) → Stage 3. **Reuse `scratch/noise_viz/build_compare.py` as the
   import/start/FT/noise scaffold** — it already does this correctly for all
   seven (the start-detection step is mandatory; skipping it fabricates a false
   pedestal — see memory `scatter-noise-estimator-shipped`).
2. **Corner curves**: per fixture, sweep `s ∈ [1, 8]`, plot `N(s)` (observed)
   with the `N_eff·exp(-s²)` noise model overlaid; mark the model-crossover and
   Kneedle corners. Calibrate `N_eff` on line-free bins.
3. **Cross-regime map**: tabulate corner `s*`, plateau width, and true/false
   counts vs each fixture's SNR. Question: is `s*` roughly constant (→ one global
   threshold, replace the frozen 2.0/3.0 with the principled value) or does it
   slide with SNR (→ make detection SNR-adaptive)?
4. **Regression**: confirm 2638 Stage 3/4/5 against the current frozen baselines
   (those tests are pinned to `method="adaptive"` for now — see below). Decide
   per-fixture ground-truth checks: 1512 (vinyl cyanide, 115-line `.cat`) and 655
   (high-SNR VyCN) have catalog truth (memory `issue1-1512-fitting-test`,
   `655-vycn-validation`) — use them to score detection precision/recall vs `s`.
5. **Escalation trigger**: if the corner or precision/recall differs *dramatically*
   between the low-SNR (363/2638) and high-SNR (655/1019) regimes, that is the
   cue to prototype regime-specific detection rather than one-size-fits-all.

## State handed over

- Scatter is the **default** noise estimator (api/Pipeline/CLI `method="scatter"`);
  adaptive remains available via `method="adaptive"`.
- The existing Stage 3/4/5 **regression baselines are pinned to
  `method="adaptive"` AND still build on the legacy `standard_ft_params` (zpf=2,
  expf_us=5.0) FT** (frozen reference) so the suite stays green. Re-deriving them
  against the production grid — **scatter noise on a raw zpf=0 FT** — is the first
  re-baselining task of this benchmark, done deliberately (it changes peak/window/
  χ² expectations across the Stage 3/4/5 integration tests, and
  `test_stage3_user_grid`'s zpf snap-back test needs rethinking once the user grid
  is zpf=0). The scatter Stage 2 tests already run on a zpf=0
  (`baseline_2638_stage1_raw`) fixture — copy that pattern.
- **Known pre-existing red:** `test_stage3_peak_detection.py::test_gap_pass_recovers_a_weak_line`
  fails on this branch independent of the noise work (confirmed by stash-revert).
  The honest-noise floor may well change whether the gap pass recovers that weak
  line — fold it into the benchmark rather than patching the assertion blind.
- Per-shot noise context (cross-fixture σ·√N, the 2023 front-end step, 655's
  0.25 V range) is in memory `scatter-noise-estimator-shipped`; drivers under the
  gitignored `scratch/noise_viz/`.
