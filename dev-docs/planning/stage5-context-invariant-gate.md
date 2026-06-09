# Plan: Stage 5 context-invariant accept gate (information-weighted χ²)

Status: **proposed.** No code written. This is the structural fix that the
performance work ([`stage5-nls-performance.md`](stage5-nls-performance.md))
identified as the blocker for every region-shrinking speedup lever, and it is a
correctness improvement in its own right: the Stage 5 accept gate is sensitive
to **window size**, which it should not be — whether a line is real is a local
question, independent of how many empty bins surround it.

## The problem

The conservative add-one-peak accept gate (and the blend-escalation, knockout,
and merge gates) decides K vs K±1 by comparing
`AICc = 2k + n_eff·log(χ²/n_eff) + 2k(k+1)/(n_eff − k − 1)`
(`fitting/validation.py::calculate_aicc`), with
`n_eff = effective_sample_size(model, kind="perplexity_log1p_snr")` — the
perplexity `exp(H(p))` of the normalised per-bin information weights
`w_f = log1p(|model_f| / σ_f)`.

`n_eff` is (correctly) ~window-size-invariant: widening a window adds bins with
`w_f ≈ 0`, which barely move the perplexity. **But `χ²` is the raw
noise-weighted chi-squared over all `M` bins**, and for a unit-variance fit
`χ² ≈ n_data = 2M`. So the log-likelihood term is `n_eff·log(2M/n_eff)`, and the
accept benefit of adding a real peak is

```
ΔAICc_likelihood ≈ n_eff·log(χ²_K / χ²_{K+1}) ≈ n_eff·(Δχ²/χ²_K) ≈ n_eff·Δχ²/(2M)
```

A real peak's `Δχ²` (its energy in noise units) is **local** — independent of
`M`. So the accept benefit scales as **`1/M`** while the `2k` cost is fixed:
shrink the window and the same peak clears the bar more easily. That is the
entire window-size sensitivity, and it is a units mismatch — `n_eff` counts the
*informative* bins, but `χ²` is diluted by the `M − n_eff` **noise** bins, each
adding ~1 to `χ²` while carrying zero information about whether the peak is real.

### Evidence (this session)

The over-add that killed three separate speedup levers is all this one cause,
arriving by different routes:
- **Lever 0** (batched add-loop): over-added 103→116 on a heavy subset.
- **Lever 2a** (in-`conservative_fit` window decomposition): over-added 67→87 on
  363; metric held only because the downstream prune cleaned it up.
- **Stage-4 small-window retune** (`max_window_width_mhz` 40→8): −34 % wall on a
  moderate-density Lorentzian band with the metric holding, but **+67 % lines on
  a dense band and +87 % lines (2× slower) on 363**, because more/smaller
  windows systematically over-accept. Scaling the baseline order with point count
  did **not** fix it (order-4 was helping, not over-fitting), ruling out the
  baseline and confirming the gate.

See [`stage5-nls-performance.md`](stage5-nls-performance.md) "Measured outcomes"
for the full data and the band-fit samples.

## The fix: weight χ² with the same information weights as n_eff

Use an **information-weighted chi-squared** in the AICc, weighted by the *same*
`w_f = log1p(|model_f| / σ_f)` that defines `n_eff`:

```
χ²_w = n_eff · ( Σ_f w_f · r_f² ) / ( Σ_f w_f )        # = n_eff · weighted_mean(r²)
```

where `r_f² = |residual_f / σ_f|²` is the per-bin unit-variance squared residual
(real+imag, the same the raw χ² sums). For a good fit `r_f²` averages to ~1
everywhere, so `weighted_mean(r²) ≈ 1` and `χ²_w ≈ n_eff` — i.e.
`χ²_w / n_eff ≈ 1`, **independent of `M`**. Then:

- **Adding noise bins** (a wider window) adds `w_f ≈ 0` entries that move neither
  `χ²_w` nor `n_eff` → **AICc is window-size-invariant**. A peak's acceptance
  depends on its *local* evidence (how much it reduces the residual in
  high-information bins), not on how many empty bins surround it.
- **A real peak** reduces `r²` in its high-`w` bins → large `χ²_w` drop →
  accepted. **A spurious peak** reduces `r²` only in low-`w` noise bins →
  negligible `χ²_w` change → rejected. The gate weights evidence by information
  on *both* sides of the ledger, closing the loop perplexity opened.

## Implementation sketch

This is a small, central change with wide reach. Do it behind a flag so the
before/after is a clean A/B.

1. **`fitting/validation.py`:** add `noise_weighted_chi2_weighted(z, sigma,
   model, weights)` (or extend `calculate_noise_weighted_chi2`) computing
   `χ²_w = n_eff · Σ w·r² / Σ w`. Factor the `w_f = log1p(|model|/σ)` weight out
   of `effective_sample_size` so the gate computes `n_eff` and `χ²_w` from the
   *same* weight vector in one place. (`effective_sample_size` already builds
   `w`; expose it.)
2. **The four gates** (all in `window_fit.py` / `residual_rescue.py`): wherever a
   gate currently calls `calculate_aicc(fit.chi_squared, k, n_eff)`, pass `χ²_w`
   instead — computed from the **more-complex model's** weights so both sides of
   the comparison share one weight vector, exactly as `n_eff` already is shared:
   - `conservative_fit` add gate (and `_blend_aware_seed` K=2/3 escalation),
   - `knockout_test`,
   - `merge_close_peaks_cleanup` Tier-2,
   - `iterative_aicc_cleanup`.
   The shared-weights discipline is the analogue of the existing "n_eff keyed on
   the K+1 / K-fit model" rule — keep it.
3. **Keep reporting χ² raw.** `WindowFitResult.chi_squared` / `reduced_chi2`, the
   persisted `quality_metrics`, the SNR-aware pass metric
   (`snr_aware_chi2_pass`), and the diagnostic F-test stay on the raw χ² — they
   are calibrated and externally meaningful. Only the *internal AICc gates*
   switch to `χ²_w`. Be deliberate that gate-χ² and reported-χ² now differ.

## Validation plan

1. **It MUST change the baseline** (unlike a perf-only change) — that is the
   point. Re-run the cross-fixture SNR-aware metric (`stripped_metric.py` over
   2638/655/363/360/1231/1512/1019, persisted `stage5_fit` stripped) and require
   the pass rate to **hold or improve**, the bulk χ²ᵣ to stay sane, and the 2638
   control to stay ≈1.0. Diff the peak sets vs the shipped baseline to see what
   moved and confirm the changes are sensible (marginal lines, not strong ones).
2. **Window-invariance test** — the direct check: fit the same band at
   `max_window_width_mhz` = 40 and 8 (the `band_fit.py` sampler) and confirm the
   **line counts now match** (the over-add is gone). That is the gate working.
3. **Full suite** (`pytest`) — the gate is load-bearing; many fitting tests
   assert peak counts / acceptance and will need their expectations re-derived if
   the gate genuinely improved. Treat changed assertions as evidence to inspect,
   not noise to silence.

## Open questions to settle before/while building

- **Weight steepness.** `log1p(SNR)` is gentle; a near-threshold real peak has
  modest `w`, so its bins get modest weight — does that under-weight exactly the
  marginal peaks the gate decides on? A/B `log1p(SNR)` vs `|model|²` (the Fisher
  density, `kish_mag_sq`) for the weighted χ². This reconnects to **issue #9**
  (whether the K-vs-(K−1) merge/knockout sweeps want a magnitude-concentrated
  kind) — a context-invariant χ² may make that choice moot or may sharpen it.
- **Support-restricted variant.** Instead of the soft `n_eff·weighted_mean(r²)`,
  compute χ²_w only over the `hard_radius` informative support. More obviously
  local, but threshold-sensitive; the soft form is smoother across iterations.
  Decide which after the first A/B.
- **The `n_eff − k − 1 ≤ 0` divergence** still fires structurally; with an
  invariant `n_eff` it becomes purely a property of the feature, which is the
  intended behavior (a truly under-resolved blend hits `+inf` regardless of
  window).

## If this succeeds: revisit what failed this session

A window-invariant gate is the **enabler** the region-shrinking levers needed.
Once it holds the cross-fixture metric, revisit, in order of expected payoff:

1. **Stage-4 small-window retune** — the cleanest win (−34 % on the moderate
   band). With the gate fixed, the dense/Gaussian over-add should vanish. Bring
   back `max_window_width_mhz` ~8–12 + `magnitude_attachment_threshold` ~1.0
   (lean on the order-4 baseline for the now-flat far pedestal); validate the
   contributor relaxation separately. A **points-based** cap likely generalises
   better than MHz across fixtures. Consider gap-aware splitting only if a
   residual dense-blend issue remains.
2. **Lever 2a** (window decomposition inside `conservative_fit`) — its
   `window_fit_2a_decompose.patch` is preserved; the amplitude/shape-aware
   coupling model is correct physics. With an invariant gate the per-cluster
   null-gate stops over-/under-accepting.
3. **Lever 0** (batched add-loop) — least likely to pay (the loop is already
   ~8 solves/window) but its over-add was also gate-driven; re-evaluate only if
   1–2 leave the add-loop the bottleneck.

## Reusable harnesses (all in `scratch/stage5-nls/`)

- `capture_windows.py <fid> <n> <min_k>` + `replay_bench.py` — capture real
  `conservative_fit` calls, replay fast (the 97 % path) with peak-set
  fingerprint; `diff_dumps.py` for tolerance peak-set diffs.
- `band_fit.py <fid> <flo> <fhi> <cap> <thresh>` — fit only a frequency band
  (monkeypatched plan-load), the window-invariance + small-window sampler.
- `stripped_metric.py` / `strip_decomp.py` — full-fixture cross-fixture pass
  metric (strip persisted `stage5_fit` first — persisted-shadow gotcha).
- `profile_fit.py` — cProfile the replay (the assembly-cost breakdown).
- `char_stage4.py` / `char_thresh.py` / `check_contrib.py` — Stage-4 window /
  contributor characterisation.
- Full fits are 15–25 min each — **characterise and band-sample; do not sweep
  full fits.**
