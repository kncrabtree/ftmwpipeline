# Stage 3 cross-fixture benchmark + the SNR "corner" detection threshold

This benchmark re-assessed every SNR-dependent Stage 3 decision once the
**scatter** noise estimator became the default (an honest noise floor, no
leakage-pedestal inflation), and replaced the fixed-threshold reasoning with a
cross-fixture-validated detection "corner." It ran across all seven fixtures,
which span ~3 orders of magnitude in max SNR (363 ≈ 6×10², 2638 ≈ 7×10²,
360 ≈ 3×10³, 1231 ≈ 2×10³, 1512 ≈ 1×10³, 1019 ≈ 5×10⁴, 655 ≈ 1×10⁵). Findings
and the shipped architecture are written up in
[`../research/stage3-snr-corner/report.md`](../research/stage3-snr-corner/report.md).
Headlines:

- **The corner is global, not SNR-adaptive.** The empirical knee on `N(s)` sits
  at 3.0–3.4 across all seven fixtures (135× max-SNR span); the frozen
  promotion 3.0 is now a principled value. The Rayleigh model-crossover locator
  is invalid here — the line-free exceedance is ~6× heavier-tailed than Rayleigh
  (coherent leakage, not thermal; verified against a white-noise control).
- **The estimator mismatch was resolved, but not by a blind swap.** Stage 3's
  internal noise is now the honest `scatter` estimator (consistency with
  persisted Stage 2), made safe by a **continuous leakage-aware floor on both
  passes** (`min_snr·σ + k·(S_coh/√M)·σ`). A hard `S_coh` mask was ruled out —
  it would delete every strong line (they generate the coherence). The passes
  run on opposite-leakage spectra (Blackman-Harris primary annihilates leakage;
  matched-filter gap retains it), so they take separate, visually-calibrated
  coefficients: `primary_leakage_floor_k = 1.0` (1019) and
  `gap_leakage_floor_k = 3.0` (1512). The latter **retires the orphaned hard
  gap-mask threshold 8**, and keeps real weak lines on strong wings the hard
  mask over-killed. 2638 Stage 5 A/B shows no downstream regression.
- **The Stage 3/4/5 integration baselines were re-derived onto the production
  grid** (raw `zpf=0` + scatter): `baseline_2638_stage2` repointed to
  `baseline_2638_stage1_raw` + scatter.

The benchmark method — the corner decomposition `N(s) = N_real(s) +
N_noiseFP(s)`, the model-crossover vs Kneedle corner locators, the Rayleigh
white-noise control, the per-fixture cross-regime map, and the catalog-scored
precision/recall on the 1512 and 655 vinyl-cyanide fixtures — is documented in
full in the research report. All of the originally-open threads have since
closed:

- **Issue #10 O4** (cross-instrument validation of `_GAP_ACTIVE_ZPF` and the
  K=4 SavGol rule) signed off on the `zpf ∈ {1, 2}` plateau; `_GAP_ACTIVE_ZPF`
  stays at 2 (see [`stage3-peak-detection.md`](stage3-peak-detection.md) O4).
- The Stage 5 χ²-noise estimator (former divergence D9) unified onto the
  scatter authority when the adaptive estimator was retired repo-wide.
- The report's scratch driver references were replaced by tracked fixture-build
  scripts (issue #14).

The shipped Stage 3 detection architecture — two-pass detection on the active
FT, the continuous leakage-aware floor, and the frozen but now-principled
`min_snr` corner — is described in
[`stage3-peak-detection.md`](stage3-peak-detection.md).
