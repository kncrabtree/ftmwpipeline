# Noise estimation at extreme SNR: the leakage-pedestal failure

The Stage 2 noise estimator over-reports σ by up to ~6× on high-SNR,
line-dense spectra. It measures the *level* of the magnitude spectrum, which on
such spectra is dominated by the smooth deterministic **leakage pedestal** —
the summed far-wings of strong lines in the raw (boxcar) FT — not the random
noise. The error is invisible at the SNR the estimator was calibrated against
(~10³) and grows with SNR, reaching ~6× at ~10⁵–10⁶. This report establishes the
mechanism, validates a high-pass (scatter-based) replacement against two
independent references, and characterises its one residual imperfection.

## 1. Why this surfaced

On the high-SNR pure-vinyl-cyanide fixture **655** (2.13 M shots, same
spectrometer as 2638), every Stage 5 window came back with reduced χ² ≈ 0.17 —
far below 1. A χ²_r ≪ 1 means the per-bin σ fed to the fit is too large: the
model residuals sit well inside the claimed noise band. The artifacts confirmed
it by eye — the noise envelope drawn on every figure rode well above the visible
baseline scatter.

Because every downstream SNR-dependent decision (peak-detection thresholds,
skirt exclusion, window-edge tests, the Stage 5 χ² statistics) inherits the
Stage 2 σ, an inflated noise floor corrupts the whole pipeline. This had to be
understood before any of the SNR-sensitive parameters — all calibrated at
max SNR ~10³ — could be assessed against a spectrum now at ~10⁶.

## 2. The mechanism: a leakage pedestal, not noise

The canonical FT is deliberately **raw and un-apodized** (boxcar), to keep
per-bin noise statistics independent (see `../noise-grid-invariance/`). A boxcar
on a strong line produces a slowly-decaying far-field: the magnitude skirt of an
exp-damped sinusoid falls as `|X| ≈ X_peak·γ/|Δf|`, and the boxcar's own sinc
sidelobes ring across the whole record. At SNR 10³ these wings are a small
fraction of the noise and average out of the noise mask. At SNR 10⁶ they are
enormous, and the **sum** of the far-wings of hundreds of strong lines forms a
smooth pedestal that fills every "quiet" bin.

The current estimator characterises noise from the **level** of |X| (a
median/MAD over a noise mask, refined by a Lorentzian-skirt exclusion). On 655
that level is the pedestal. Measured directly on the persisted 655 FT, the
quiet-region median |X| runs 1.6–5.2× the true noise and tracks proximity to the
strong lines — it is the pedestal, not the noise. The skirt exclusion cannot
remove it: the radius is capped at 500 MHz while the strongest single line's
predicted skirt is ~770–3800 MHz, and the pedestal is the *sum* of many lines
spanning the whole band — excluding ±500 MHz around each would erase the
spectrum and still leave the summed pedestal underneath.

## 3. Why it is a pure SNR-scaling failure

The two contributions behave oppositely with shot count `N`:

- **True noise** averages down as `σ ∝ 1/√N`.
- **The pedestal ∝ signal amplitude**, and the signal is the *coherent average* —
  identical regardless of `N`. The pedestal is **constant in N**.

So a level-based estimator reads `≈ √(σ(N)² + pedestal²)`: at low N the noise
dominates and the estimate tracks 1/√N; at high N the pedestal dominates and the
estimate **plateaus**. This is exactly the observed behaviour and is the cleanest
possible diagnostic (§5.2).

## 4. The high-pass (scatter) estimator

The fix follows directly from §2: the noise is the **white, bin-uncorrelated**
part of the spectrum; the pedestal is the **smooth** part. Separating them is a
high-pass filter along the frequency axis.

Two high-passes were tried:

- **lag-1 difference of Re/Im** — fails. Adjacent-bin differencing is a high-pass
  with a known √2 noise factor, but the boxcar sidelobe ripple in Re/Im
  oscillates at ~1.3-bin period, so it *survives* the high-pass and inflates the
  estimate (3.9× over on 655, worst near strong lines).
- **|X| minus a broad running median** — works. The pedestal is smooth in
  *magnitude* even where Re/Im oscillate (the ripple is in phase, the magnitude
  envelope is smooth). The MAD of the high-passed |X| recovers the noise.

The estimator (self-contained — see `prototype.py`) runs **before** Stage 3, so
it cannot use a peak list; it self-masks by flagging sharp excursions above the
local robust scale as lines, iterating, and taking the windowed MAD of the
high-passed residual over the survivors:

```
σ(f) = k_corr · 1.4826 · MAD( |X| − medfilt_pedestal(|X|) )   over non-line bins
```

`k_corr` maps the magnitude-scatter to the underlying complex-Gaussian σ.
Because `|X|` is Rician, the correct value depends on the local regime; §4.1
makes it region-aware.

### 4.1 The region-aware (Rician) correction

`|X|` is Rician, so the magnitude-scatter relates to the underlying σ by a factor
that runs from **1.0** under strong lines (pedestal ≫ σ, Rician → Gaussian) to
**1.47** in quiet Rayleigh regions (pedestal → 0). A single fixed `k_corr` is
therefore wrong in a *regime-dependent* way — and wrong by ~20 % at exactly the
high-pedestal bins under lines, where χ²_r is computed.

The fix needs no frames. The dimensionless **ratio `R = scatter / pedestal`** is a
monotone function of the regime alone, so a single 1-D lookup `C(R) = σ/scatter`
recovers the regime-correct factor from one spectrum (both `scatter` and
`pedestal` are already computed per window). `C(R)` is built once by simulating
the Rician magnitude (`prototype.py`, `_build_CR`); production bakes it as
constants.

Validated against the line-free frame-difference truth across the five
multi-frame fixtures (per-region median σ_est/truth, figure
`fig5_region_aware.png`):

| | 2638 | 360 | 1019 | 655 | all (p90/p10 spread) |
|---|---|---|---|---|---|
| fixed `k_corr` = 1.20 | 0.86 | 0.95 | 0.86 | **1.15** | 1.70 |
| region-aware `C(R)` | 1.03 | 1.03 | 0.99 | **0.97** | 1.42 |

The fixed factor's bias is regime-dependent — most visibly +15 % on the high-SNR
655, where the whole pedestal problem lives — and no constant can fix both the
Rayleigh and under-line ends at once. `C(R)` centers the clean fixtures near 1.0
and tightens the overall band. It corrects the *level*, not the within-region
spread (which is dominated by frame-difference truth noise on the small-batch
fixtures and near-line pedestal residual) and does not touch the 1/√N slope (§6,
the weak-line floor). The 2-frame fixture 363 (noisiest truth) is the one
exception and is discounted.

## 5. Validation

### 5.1 Frame-difference cross-check — and why it is *not* an oracle

Blackchirp saves progressive backup frames (cumulative averages over disjoint
shot batches). Because the FT is linear, the disjoint-batch average between two
backups can be reconstructed, and differencing two such batches **cancels the
coherent signal** — leaving, in principle, signal-free noise (figure
`fig4_framediff_pedestal.png`).

It is not a clean oracle, however: the free-running digitizer clock drifts
*within* the acquisition (1–2 ppm), so a strong line sits at slightly different
frequencies in the early vs late batch, and `d(line)/df · δf` survives the
difference as a **residual peak**. The difference is therefore clean only in
**line-free regions** (no signal to imperfectly cancel, and the pedestal cancels
too). After masking lines and sigma-clipping the residual spikes, the line-free
frame-difference noise and the scatter estimator agree to ~10 % globally on 655
(0.0092 vs 0.0094) — two methods with no shared assumptions. Neither measures the
noise *under* the lines; that relies on the physical fact that noise is a smooth
spectrometer property and can be interpolated across line positions (which the
estimator does). The only reference that would measure noise under the lines
directly is a **blank-FID series** (increasing shots, no sample) — recommended as
a future fixture and the definitive validation.

### 5.2 The 1/√N test — the decisive discriminator

Applying *each* estimator to *every* backup frame separates noise from pedestal
unambiguously (§3): the scatter estimator must fall as 1/√N (slope −0.5 in
log–log), while a pedestal-pinned estimator must flatten. Figures
`fig0_sqrtN_655.png` (655 alone) and `fig1_sqrtN_all.png` (all multi-frame
fixtures):

- **Old estimator, 655: slope −0.02** — dead flat. It does not average down,
  because it is measuring the constant pedestal.
- **Scatter estimator, 655: slope −0.38** — falls with N (the residual −0.12
  from −0.5 is the contamination floor, §6).
- On the lower-SNR fixtures (1019, 2638, 360, 363) the **old** estimator recovers
  ~1/√N (slopes −0.41…−0.49) — it only breaks at extreme SNR, where the pedestal
  finally dominates. This is the SNR-scaling failure made visible.

### 5.3 Cross-fixture: the overestimate scales with SNR

Seven fixtures (two samples, same spectrometer), `data/cross_fixture_results.json`,
figure `fig2_overestimate_vs_snr.png`. Overestimate = old σ / frame-difference
truth (○) or / scatter (×, where no multi-frame data):

| fixture | max N | max SNR | old / truth | scatter / truth | old slope | scatter slope |
|---|---|---|---|---|---|---|
| 655  | 2.13 M | ~1.2e5 | **6.1×** | 1.14× | **−0.02** | −0.38 |
| 1019 | 1.59 M | ~5e4   | 1.5× | 0.84× | −0.42 | −0.45 |
| 2638 | 0.50 M | ~700   | 1.4× | 0.86× | −0.49 | −0.46 |
| 360  | 0.78 M | ~3e3   | 1.8× | 0.99× | −0.46 | −0.47 |
| 363  | 0.53 M | ~600   | 1.7× | (proxy) | −0.41 | −0.38† |
| 1512 | 17 860 | ~1e3   | 1.8× | (proxy) | — | — |
| 1231 | 74 740 | ~2e3   | 1.7× | (proxy) | — | — |

†2-frame fixture, single slope estimate. The old estimator's error grows from
~1.4–1.8× at low SNR to 6× at 655; the scatter estimator stays within ~15 % of
the frame-difference truth across the whole 10³–10⁶ SNR range. (Note even at low
SNR the old estimator over-reads ~1.4–1.8× vs the frame-difference truth — the
pedestal is never exactly zero — but the slope confirms it is dominated by noise
there.)

### 5.4 Downstream impact

Carrying the corrected (scatter) σ into the persisted 655 Stage 5 fit raises the
predicted median χ²_r from **0.17 to 3.0** (88 % of windows now > 1). The
inflated noise had been *masking* a real, spectrum-wide model deficit (the
Lorentzian-τ lineshape residual, documented separately): once the noise is
honest, χ²_r ≈ 3 correctly reflects that the fits sit ~2× above the true noise
floor. Fixing the noise is the prerequisite for assessing everything downstream.

## 6. The residual: a constant contamination floor (not Rician)

The scatter estimator's 1/√N slope is −0.38 on 655, short of −0.5. Modelling
`scatter(N)² = c/N + f²` fits the six 655 frames cleanly (figure
`fig3_floor_decomp.png`): a straight line in `1/N` with a **positive intercept**
`f ≈ 0.007`. The `c/N` term is the averaging-down noise (its slope is −0.500 by
construction); `f` is a **constant floor** — weak undetected lines that pepper a
spectrum this dense, plus pedestal residual, sitting in the bins counted as
line-free. It does not average down, so it flattens the slope at high N.

This rules out the magnitude (Rician) factor as the cause: it is a multiplicative
regime correction and cannot remove an additive floor — so neither the fixed
`k_corr` nor the region-aware `C(R)` (§4.1) changes the slope. Stricter line
masking shrinks the floor
(0.0070 → 0.0055) and steepens the slope (−0.38 → −0.41) — a partial easy win —
but cannot remove it: on a pure-VyCN spectrum at 10⁶ SNR there are weak lines
essentially everywhere. The principled removals are (a) the N-scaling fit itself
when frames are present (the `c/N + f²` decomposition *is* a floor-free noise
estimate), or (b) the blank-FID series. The single-spectrum estimate is ~14 %
high at max N on 655 — acceptable versus the 6× it replaces, and far better on
every less-dense fixture.

## 7. Caveats and known edges

- **Magnitude-regime (Rician) factor** is handled region-aware via `C(R)` (§4.1).
  The residual per-region spread that remains is dominated by other effects
  (frame-difference truth noise on small-batch fixtures, near-line pedestal
  residual), not the factor.
- **Frame-difference is line-free-only** (§5.1); not a substitute for a blank.
- **Self-mask vs Stage-3 peaks.** Self-masking flags ~4 % of 655 bins as lines;
  a Stage-3 peak list would mask more and shrink the floor, but Stage 2 precedes
  Stage 3 and must stand alone.
- **Single-frame fixtures (1512, 1231)** cannot be 1/√N-validated; the estimator
  still applies and the overestimate (~1.7×) is consistent with the low-SNR group.

## 8. Reproducing this report

All fixtures are committed under `examples/blackchirp_data/<id>/` with their
backup FID frames retained (plain blobs). Frame counts: 655 (6), 1019 (5), 2638
(3), 360 (3), 363 (2), 1512/1231 (1, single-frame). See `../../fixtures/README.md`.
- `prototype.py` — the self-contained scatter estimator (`estimate_sigma`).
- Cross-fixture driver, 1/√N test, floor decomposition, and figure scripts are in
  `scratch/noise_research/` (`analyze_all.py`, `goal3_slope.py`, `make_figures.py`)
  and `scratch/655_validation/` (`oracle_check.py`, `check_sqrtN.py`,
  `diag_noise_frames.py`). Results: `data/cross_fixture_results.json`.

The 1/√N slope is a clean, sample-independent acceptance invariant and should
become a unit test for any production estimator.

## 9. Implementation plan

Validated; the path to production:

1. **Land the estimator in `preprocessing/noise_estimation.py`** as
   `estimate_noise_scatter` (the `prototype.py` core, region-aware `C(R)` per
   §4.1 — bake the `(R_TAB, C_TAB)` arrays as module constants rather than
   simulating at import), returning the existing `NoiseResult` (per-bin σ, line
   mask, diagnostics) so it is a drop-in for the current `estimate_noise_adaptive`.
   Stage 2 precedes Stage 3, so it must remain self-masking — no peak-list
   dependency.
2. **Wire through `_internal/stage2_impl.py`** only; the dual interface
   (`Pipeline.estimate_noise` / `api.estimate_noise` / `estimate-noise` CLI) and
   the persisted `stage2_noise_result` inherit it unchanged. Run the
   cross-interface consistency tests.
3. **Acceptance tests.** (a) A synthetic 1/√N test: white complex noise + a few
   strong lines, FT at increasing N, assert the estimator's slope ≈ −0.5 and that
   it is pedestal-independent (varying line amplitude must not change σ̂) — this
   is the regression guard the old estimator would fail. (b) No-regression on the
   2638 fixture: Stage 3/4/5 outputs must hold (the calibration fixture; the
   region-aware estimator reads ~1.0× the frame-difference truth there vs the old
   1.4×, so verify the fit quality and gate firing do not shift materially).
4. **Optional frame-aware mode.** When backup frames are present, expose the
   `c/N + f²` floor-corrected estimate (§6) as a higher-accuracy option and the
   in-pipeline analogue of the blank-FID measurement.
5. **Then revisit the SNR-sensitive downstream knobs.** With honest σ flowing
   through, re-assess Stage 2b/3/4 thresholds (all calibrated at max SNR ~10³)
   against the now-correct ~10⁶ SNRs — the separate SNR-aware-parameters pass.

The blank-FID series (§5.1) is the recommended companion acquisition-protocol
change: a few no-sample FIDs at increasing shot counts give an
experiment-independent, under-the-lines noise reference for any instrument.
