# STFT τ-calibration: design, validation, and application

The Stage 2b tau calibration extracts a molecular-decay constant
`τ_maj ± σ_τ` from the raw FID without running any LSQ fit. The
calibration drives Stage 5's per-window tau prior (a bidirectional
Gaussian penalty anchored on `τ_maj`) and Stage 3's gap-pass matched
filter; the normative spec lives in
[`dev-docs/planning/stage2b-tau-calibration.md`](../../planning/stage2b-tau-calibration.md).

This report covers the underlying method, its synthetic acceptance,
the 2638 application, an LSQ-fit-and-histogram cross-validation, and
the polish-step design decisions that fix the contributor-level
log-linear bias.

Reproduce headline analyses (all scripts assume the
`ftmwpipeline-dev` conda env):

```bash
# synthetic validation (Cases 1-7 + pathological corners)
conda run -n ftmwpipeline-dev python \
    dev-docs/research/stage5-tau-calibration/prototype.py

# 2638 application + cached STFT classification figures
conda run -n ftmwpipeline-dev python -c "
import ftmwpipeline.api as ftmw
f = 'scratch/exp_2638.ftmw'
ftmw.import_data(f, source='examples/blackchirp_data/2638/', force=True)
ftmw.compute_ft(f, zpf=2, expf_us=None, trim=(26500, 40000))
ftmw.estimate_noise(f)
print(ftmw.calibrate_tau(f))
"

# LSQ cross-validation on the unapodized fixture
conda run -n ftmwpipeline-dev python -c "
import ftmwpipeline.api as ftmw
f = 'scratch/stage5-tau-calibration-lsq/exp_2638_unapodized.ftmw'
ftmw.import_data(f, source='examples/blackchirp_data/2638/', force=True)
ftmw.compute_ft(f, zpf=2, expf_us=None, trim=(26500, 40000))
ftmw.estimate_noise(f)
ftmw.detect_peaks(f)
ftmw.assign_windows(f)
ftmw.fit_peaks(f, tau0_us=(15.0-2.35)/2.0)
"
conda run -n ftmwpipeline-dev python \
    dev-docs/research/stage5-tau-calibration/lsq_comparison.py

# polish_snr_cap × relative_gate_fraction sweep (production-default calibration)
conda run -n ftmwpipeline-dev python \
    dev-docs/research/stage5-tau-calibration/polish_snr_cap_validation.py
```

Cached numerics live under [`data/`](data/); figures under
[`figures/`](figures/).

---

## Method

For an active FID of length `N` and sample spacing `dt`, the
calibration runs in three phases:

1. **Sliding-active-window STFT.** Split the active interval into
   `n_seg = 10` non-overlapping frames of length `T_w = T_full /
   n_seg`. For each frame, zero everything outside the frame and
   `rfft` the full-record-length signal. This preserves the full-
   record bin spacing `Δf = 1 / T_full` across all frames, so per-bin
   time-series are 1:1 comparable.

2. **Per-bin fit and classification.** At each frequency bin, fit
   `|S_n(a)| = C · exp(-a/τ)` across the `n_seg` frame midpoints `a`
   via weighted log-linear regression (`log|S_n| = log C - a/τ`,
   weights `w = |S_n|²`). Compute residual sum of squares `rss_exp`
   in linear `|S_n|` space and contrast against the constant-model
   `rss_const`. Classify each bin:
   - **discard** if `max_n |S_n| < T_σ · σ_frame` (per-frame SNR
     below the gate floor `T_σ = 5`).
   - **spur** if AICc prefers the constant model by ≥ 2 OR τ saturates
     at `0.95 · τ_max` (CW tone: no decay).
   - **bad-fit** if `rss_exp > 5 · n_seg · max(σ_frame², (0.05·m̄)²)`
     where `m̄` is the per-bin mean magnitude. The relative branch is
     necessary for high-SNR clean fits not to over-classify; the
     log-linear weighted regression does not minimise linear-space
     RSS, so its prediction error scales with signal level, not noise.
   - **contributor** otherwise. These enter the τ histogram.

3. **Aggregation.**
   - **SNR-weighted majority τ.** Weighted median of contributor τ
     by `snr_per_bin = max_n |S_n| / σ_frame`. `σ_τ = (weighted IQR)
     / 1.349`. The SNR weighting collapses the distribution onto
     on-line bins; the unweighted median is biased ~10 % high by
     near-threshold skirts whose log-linear fits pull positive
     (slope ≥ 0 → τ → ∞ floor).
   - **Per-band majorities.** When `compute_band_majorities=True`,
     compute the same SNR-weighted majority on contributor subsets
     inside each band (default: arithmetic thirds of `[trim_lo,
     trim_hi]`). Bands with fewer than `min_contributors_per_band`
     fall back to the band-wide majority. Stage 5's
     `Pipeline.fit_peaks(per_band_tau=True)` consumes these.
   - **GMM bimodality check.** Hand-rolled EM 2-component vs
     1-component on the contributor τ histogram; `ΔAICc > 2` flags
     two-cluster preference.
   - **Spur clustering.** Group adjacent spur-classified bins
     within `n_seg` full-record bins (= 1 STFT bin) into a single
     `SpurCluster` representing one CW tone.

The default operating points:

| knob | default | basis |
|---|---|---|
| `N_seg` | 10 | Case 1 across (T_full, τ_truth) grid. |
| `T_σ` | 5 | Case 2 SNR floor; passes contributors at SNR ≥ 5. |
| `tau_max_us` | 5 · T_full | High enough to distinguish CW spurs from longest physical τ. Case 3. |
| `rss_gate_factor` | 5 (relative + absolute) | Hybrid `max(5·n·σ², 5·n·(0.05·m̄)²)`. Without the relative term, high-SNR clean fits over-classify as bad-fit. |
| `relative_gate_fraction` | 0.05 | The 5 % of mean-magnitude factor in the relative branch. |
| SNR-weighting | on | Case 1 unweighted-median bias removed; Case 5 dense cluster collapses to truth. |
| GMM bimodality threshold | ΔAICc > 2 | Detects Case 6 synthetic; > 4 is too strict at realistic contributor counts (~50-150). |
| polish | on | Closes +3-5 % log-linear bias documented in Case 1; see § Polish design. |
| `polish_snr_cap` | 9.0 | Polish only contributors below this per-bin SNR. § Polish design. |
| spur-skirt exclusion radius | ± n_seg full-record bins | Case 4 close-spur contamination mitigation. |

---

## Synthetic validation

Seven synthetic cases probe the recovery floor, SNR sensitivity, spur
robustness, blend behaviour, multi-cluster detection, and shape-error
response. Full cached numerics in
[`data/phase1_summary.json`](data/phase1_summary.json) and
[`data/case1_tau_sweep.npz`](data/case1_tau_sweep.npz); reproduce
all seven with `prototype.py`.

### Case 1 — Single-line τ × T_full sweep

`τ_truth ∈ {3, 5, 7.5, 12, 20} µs × T_full ∈ {5, 10, 12.65, 20, 30,
40} µs`, SNR = 100, 8 trials per cell. Median SNR-weighted-majority τ
error (%) at N_seg = 10:

| T_full \ τ | 3 µs | 5 µs | 7.5 µs | 12 µs | 20 µs |
|---|---|---|---|---|---|
| 5 µs   | +1.7 | -1.5 | -3.2 | -6.3 | **-13.8** |
| 10 µs  | +2.3 | +1.5 | +1.5 | +1.5 | +1.3 |
| 12.65 µs | +4.2 | +3.1 | +3.0 | +2.9 | +3.0 |
| 20 µs  | +3.7 | +2.0 | +1.3 | +1.1 | +1.4 |
| 30 µs  | +4.8 | +2.8 | +1.2 | +0.8 | +0.9 |
| 40 µs  | +5.3 | +3.8 | +1.4 | +1.0 | +0.8 |

The `T_full = 5 µs, τ = 20 µs` cell is the long-τ-at-short-T
pathological corner (signal hasn't decayed appreciably); see § Pathological
corners. Outside that corner, every cell is within ±5.3 % at N_seg = 10.

A persistent +3-5 % positive bias appears at intermediate T_full
(≈ 12-20 µs): an artefact of the SNR² weighting in the log-linear
regression giving early frames disproportionate influence when the
signal is still ≥ exp(-T_full/τ) at the latest frame. Recovery shifts
back toward zero as T_full / τ grows. The polish step (§ Polish design)
addresses this. Figure:
[`figures/01_tau_sweep.png`](figures/01_tau_sweep.png).

### Case 2 — SNR sweep

τ = 7.5 µs, T_full = 12.65 µs, N_seg = 10. Median majority error vs
SNR:

| SNR | maj err | online err | fit_rate | mean contributors |
|---|---|---|---|---|
| 5 | -7.5 % | -7.1 % | 0.05 | 0.1 |
| 10 | +28.7 % | +32.5 % | 0.95 | 7.5 |
| 20 | +8.5 % | +8.4 % | 1.00 | 13.8 |
| 50 | +1.1 % | +0.0 % | 1.00 | 29.4 |
| 100 | +0.4 % | -0.5 % | 1.00 | 56.6 |
| 500 | -0.2 % | -0.4 % | 1.00 | 201 |
| 1000 | -0.2 % | -0.2 % | 1.00 | 166 |
| 10000 | -0.0 % | -0.0 % | 1.00 | 130 |

SNR floor for sub-1 % recovery: ≈ 50. For ±10 %: ≈ 20. Below SNR ≈ 10,
the on-line bin is rarely classified as a contributor; the τ_maj is
driven by skirts of a barely-detected line and is unreliable. Above
SNR ≈ 50, the fit is asymptotically unbiased and the contributor count
saturates at ~100-200 bins. Figure:
[`figures/02_snr_sweep.png`](figures/02_snr_sweep.png).

### Case 3 — Isolated clock spur

CW tone at SNR = 100 with one anchor line. The spur bin classifies as
spur (τ saturates at 5·T_full); the line bin classifies as contributor
(τ_line = 8.66 µs vs truth 7.5 µs, single-trial natural variance).
The spur produces 49 spur-classified neighbour bins — its
STFT-rectangular-window sinc-skirt cluster, also CW-shaped so
correctly not contributing to the τ histogram. Spur-cluster grouping
collapses these 49 bins into one `SpurCluster` entry. Figure:
[`figures/03_isolated_spur.png`](figures/03_isolated_spur.png).

### Case 4 — Spur near real line

Line at SNR = 100, spur at SNR = 100, two placements:

| label | offset (full bins) | offset (STFT bins) | spur cls | line cls | τ_line | τ_spur |
|---|---|---|---|---|---|---|
| far | 60 | 6.0 | spur | contrib | 7.72 µs | 63.0 µs |
| close | 5 | 0.5 | spur | spur | 20.05 µs | 19.97 µs |

The "close" case fails because the spur's STFT-skirt envelope (width
1 / T_w = n_seg / T_full ≡ 1 STFT bin ≡ 10 full-record bins at
n_seg = 10) extends into the line bin. The line bin's time series
becomes near-constant ≈ spur magnitude with a small decaying
component on top; AICc favours the constant model and the line is
mis-classified as a spur.

This is a real limitation of the STFT method. The mitigation is the
spur-skirt exclusion radius: drop any candidate contributor within
± n_seg full-record bins of a detected spur from the τ histogram.
Lines actually inside the exclusion zone are lost but were never
going to fit cleanly.

### Case 5 — Dense cluster

Six τ = 7.5 µs lines spaced 10 full-record bins (= 1 STFT bin) apart,
SNR = 100. All 6 on-line bins classify as contributors with τ in
[7.54, 7.76] µs (-0.5 % to +3.4 %); between-line bins also classify
as contributors with τ in [7.53, 12.7] µs (one outlier from
overlapping skirt decoherence). SNR-weighted τ_maj = 7.61 µs (+1.5 %).
The bad-fit gate doesn't aggressively drop between-line bins, but the
SNR weighting collapses majority onto the on-line peaks.

### Case 6 — Bimodal population

T_full = 20 µs (extended so the τ = 10 cluster decays across the FID).
24 lines split 50/50 between τ = 5 and τ = 10 µs, SNR = 200. GMM
2-component fit converges to μ_a = 5.05 ± 0.57, μ_b = 10.09 ± 2.63,
π_a = 0.42. ΔAIC = +53.3 (strongly bimodal). The `ΔAIC > 2` threshold
catches this cleanly. Figure:
[`figures/06_bimodal_population.png`](figures/06_bimodal_population.png).

### Case 7 — Voigt-deficit shape error

20 lines (SNR 10-1000), τ_L = 7.5 µs, Gaussian co-decay
exp(-(t/τ_G)²) injected. Three τ_G settings:

| τ_G | τ_maj (SNR-weighted) | on-line median τ | n contribs |
|---|---|---|---|
| ∞ (pure Lorentzian) | 7.69 µs | 7.54 µs | 119 |
| 30 µs (mild) | 7.04 µs | 7.61 µs | 109 |
| 12 µs (severe) | 5.12 µs | 4.69 µs | 121 |

The recovered single-exponential τ is the best fit to a Voigt
envelope, biased toward shorter τ as the Gaussian co-decay
strengthens. The shape-error signature is detectable per-fixture
(STFT τ_maj vs LSQ τ_maj on the same data) but not separable from
the natural shape distribution at fit time. Figure:
[`figures/07_voigt_deficit.png`](figures/07_voigt_deficit.png).

### Pathological corners

**Long τ at short T_full** (`τ_truth = 20 µs, T_full = 5 µs`): signal
decays by only `1 - exp(-T/τ) = 22 %` across the FID; per-bin |S_n|
vs `a` is nearly flat and the exponential fit is ill-conditioned.
Median error -7.6 %, mean contributors 17 (well below the 200-bin
acceptance pre-condition). Downstream mitigation: apply a gentle
known apodization `τ_apod = K · T_full` to force measurable decay,
then subtract the known rate (`1/τ_mol = 1/τ_obs - 1/τ_apod`).

**Short τ at long T_full** (`τ_truth = 3 µs, T_full = 40 µs`):
signal is in the first ~15 µs; later frames are noise-only. Late
noise-only frames pull the slope flatter (apparent τ slightly longer)
— consistent +7 % bias. Downstream mitigation: reduce `end_us` to
focus the FID on the signal-bearing region.

---

## 2638 application

The 2638 fixture is a real BlackChirp single-species recording (W-band,
40.96 GHz probe, lower sideband, 750 000 samples at 50 GS/s,
`expf_us=None`). Headline output of `extract_tau_majority` with all
defaults on the unapodized active region:

| quantity | value | notes |
|---|---|---|
| FID active region | 632 500 samples, T_full = 12.65 µs | start_us=2.35, end_us=15.00, sample_dt = 20 ps |
| N_seg | 10 | T_w = 1.265 µs per frame |
| Empirical σ_t | 4.16e-5 (FID units) | tail of active region |
| σ_x_full | 4.68e-7 | analytic σ_t · dt · √(N/2) |
| σ_frame | 1.48e-7 | σ_x_full / √N_seg |
| Contributors in trim (26500-40000 MHz) | 4417 bins | floor was 200 |
| Spur-classified bins | 649 | collapses to ~50-100 clusters |
| **τ_maj (SNR-weighted, production default)** | **~5.96 µs ± 1.59** | polish=True, polish_snr_cap=9 |
| σ_τ / τ_maj | 0.27 | exceeds the 0.20 pre-condition → marginal flag |
| Pearson r(freq, τ) | -0.32 | real systematic — see § Frequency dependence |
| Per-arithmetic-third majority τ (low / mid / high) | 7.62 / 6.16 / 5.29 µs | within ±3.2 % of LSQ reference |
| GMM ΔAIC | +754.7 | strongly bimodal |
| GMM components | μ_a = 6.42 ± 1.31 (π_a = 0.82), μ_b = 9.08 ± 2.51 (π_b = 0.18) | dominant cluster carries 82 % |

The `σ_τ / τ_maj = 0.27` exceeds the 0.20 pre-condition, so the
`preconditions_passed` flag is False; a per-band calibration would
tighten this. The 82 % dominant-cluster weight clears the
bimodality-policy threshold so downstream consumers still trust
`τ_maj`. Figures:
[`figures/08_2638_stft_heatmap.png`](figures/08_2638_stft_heatmap.png)
(2D STFT magnitude),
[`figures/09_2638_distribution_analysis.png`](figures/09_2638_distribution_analysis.png)
(τ histogram, τ vs SNR, τ vs frequency, GMM overlay).

### Frequency dependence — horn-coupling geometry

Per-third median τ across the trim band decreases monotonically:

- 26.5-31.0 GHz (low):  7.62 µs (n=931)
- 31.0-35.5 GHz (mid):  6.16 µs (n=2050)
- 35.5-40.0 GHz (high): 5.29 µs (n=2897)

A ~2.3 µs spread across the trim range (30 % of the band-wide
τ_maj) is not noise scatter — the per-third sampling error is well
under 0.1 µs at these contributor counts. The mechanism is
**horn-coupling geometry**, not molecular: 2638 is a single species,
so velocity slip is ruled out. The W-band transmit/receive horns
advertise a fixed gain (~25 dBi) so the half-power beamwidth scales
as ~1/f for a fixed aperture. A smaller probe volume at higher
frequency means molecules in the supersonic beam (~constant terminal
velocity) spend less time in the coherent interaction region. Shorter
transit time → shorter τ_mol. Slope direction (high f → low τ) and
approximate magnitude (~30 % across the W-band span) are both
consistent with this geometric story. The horn-coupling interpretation
predicts the same qualitative trend on every fixture taken with the
same horns — a testable cross-fixture validation hypothesis.

The per-band τ majorities are persisted (`band_majorities` field on
`TauCalibrationResult`) and consumed by Stage 5 via
`Pipeline.fit_peaks(per_band_tau=True)`. The band-wide `τ_maj` remains
the fallback under `per_band_tau=False`.

### Bimodality interpretation

The GMM 2-component fit prefers (ΔAIC = +754) μ_a = 6.42 µs (π_a =
82 %) and μ_b = 9.08 µs (π_b = 18 %). The minor 9 µs cluster is
consistent with the GMM absorbing low-band high-τ contributors —
looking at the per-third medians, the low-band sits between μ_a and
μ_b but closer to μ_b. Real τ-multimodality from species mixing is
ruled out (single-species fixture). Downstream consumers should still
treat the dominant-cluster majority as `τ_maj`; the multimodality
flag is a human-auditor signal.

### Spur catalogue

649 trim-band bins classify as spurs. After grouping adjacent
spur-classified bins (cluster gap ≤ n_seg full-record bins ≡ 1 STFT
bin), the list collapses to ~135 distinct spur clusters — a
plausible count for the instrument's clock-leakage harmonics + LO
artefacts. Each cluster carries center frequency, peak-bin index, and
bin count. Sample (sorted by frequency):

```
26613.75, 26613.83, 26613.91, 26614.00          # 4 nearby
27317.71 ... 27320.08 (~15 bins, one cluster)
```

---

## LSQ cross-validation

An independent τ measurement is obtained by running Stages 0-5 on the
same unapodized fixture with `tau0_us = T_active/2 = 6.325 µs` and
no tau anchor (auto-disabled because Stage 2b is absent under
`expf_us=None`), then aggregating the per-window LSQ-fit `τ_us`
values. The script is
[`lsq_comparison.py`](lsq_comparison.py); cached numerics
[`data/lsq_comparison.json`](data/lsq_comparison.json); figures
[`figures/12_lsq_histogram.png`](figures/12_lsq_histogram.png) and
[`figures/13_lsq_freq_third.png`](figures/13_lsq_freq_third.png).

### Two LSQ populations

Stage 5 reports `σ_τ = NaN` (coerced to `None` on persist) in two
distinct cases:

1. **`fit_tau = False`** — Stage 5's window-level SNR proxy
   `max|X| / median(σ) < weak_window_snr_threshold = 10` forced
   fit_tau false (240 of 382 windows on 2638). τ stays at `tau0_us`;
   no LSQ τ information.
2. **`fit_tau = True` with singular covariance** — fit ran, covariance
   computation failed at the τ slot because `J^T J`'s diagonal was
   non-positive (123 windows on 2638). The τ value lives in
   `shared_parameters["tau_us"]["value"]`; only `σ_τ` is missing.
   Typically tight blends where the τ column of the Jacobian becomes
   degenerate with amplitude/phase columns at the optimum.

The analysis distinguishes them by recomputing the window-level
`snr_proxy` (`_compute_snr_proxy_per_window` in
`lsq_comparison.py`). Two LSQ populations are defined:

- **Strict** — finite σ_τ that passes `σ_τ/τ < 0.10` AND the
  per-window quality cuts (K ≥ 1, no fixed contributors, max free-peak
  SNR ≥ 10, χ²_r < 3, τ not saturating bounds). **N = 15.**
- **Expanded** — strict plus singular-covariance windows that pass
  the per-window quality cuts (σ_τ/τ gate dropped because σ_τ is
  undefined). **N = 90 (= 15 finite + 2 finite-but-fail + 73 singular).**

Singular-cov windows are biased neither toward weak nor toward bad
fits — they're cases where the optimizer succeeded but landed in a
τ-amplitude-degenerate basin. They carry real τ information; including
them is the right thing for a population-level central-tendency
comparison. The strict gate empties to zero if the planning doc's
original `EASY K=1, SNR ≥ 20, σ_τ/τ < 0.10, χ²_r < 2` is applied
verbatim, because EASY K=1 windows are overwhelmingly weak isolated
singletons whose τ is frozen by the snr_proxy gate.

### Headline LSQ numbers

| population | N | Gaussian µ ± σ (µs) | median (µs) | IQR / 1.349 |
|---|---|---|---|---|
| Strict | 15 | 6.78 ± 1.85 | 6.38 | 1.93 |
| **Expanded** | **90** | **6.41 ± 1.64** | **6.26** | **1.71** |

Per-arithmetic-third LSQ medians:

| third (GHz) | LSQ strict (N) | **LSQ expanded (N)** | STFT polish=False (N) | STFT polish=True / no cap (N) |
|---|---|---|---|---|
| 26.5 - 31.0 | 8.80 (6) | **7.87 (27)** | 7.76 (792) | 7.20 (792) |
| 31.0 - 35.5 | 6.13 (4) | **6.27 (33)** | 6.73 (1459) | 5.78 (1459) |
| 35.5 - 40.0 | 5.38 (5) | **5.16 (30)** | 6.11 (2166) | 5.07 (2166) |

The expanded LSQ medians are the production-reference per-band τ
targets. Strict and expanded medians/means agree within their
respective spreads, confirming the strict subset is a clean
subsampling of the same distribution (singular-cov windows don't
introduce a systematic shift).

### Per-band bias-flip pattern (motivating the polish design)

Without the SNR cap, polish=True applies a roughly constant downward
shift of 0.5-1.0 µs to every contributor τ. That shift is too large
in the low band (low-band τ is already long; polish pushes past the
LSQ value), about right in the mid band (lands between polish=False
over-estimate and polish=True under-estimate), and right in the high
band (correctly corrects the polish=False over-estimate). The
band-wide majority lands on polish=False because the two errors
approximately cancel.

The frequency trend is real horn-coupling physics: both methods
agree on monotonic decrease across the band. The expanded LSQ slope
(7.87 → 6.27 → 5.16, range 2.71 µs) sits between the STFT polish=False
slope (7.76 → 6.73 → 6.11, range 1.65 µs) and the STFT polish=True
slope (7.20 → 5.78 → 5.07, range 2.13 µs); qualitative direction and
magnitude are robust, not an STFT shape-error artefact.

### τ-runaway and the bidirectional penalty

Two of the 19 finite-σ_τ windows ran τ to the upper bound (31.625 µs)
under the no-anchor LSQ run (wid 86, 190). Production runs with Stage
2b enabled and the bidirectional Gaussian-prior penalty active
constrain these back to the τ_maj basin — confirming the calibrated
penalty's value beyond the cross-fixture-validation cases that
originally motivated it.

### LSQ filter funnel

Cumulative window counts on the realised gate (unapodized 2638):

| stage | strict | expanded | note |
|---|---|---|---|
| Total windows | 382 | 382 | |
| `fit_tau = True` ran | 142 | 142 | snr_proxy ≥ 10 cleared the weak-window gate |
| └─ tau_error finite | 19 | 19 | covariance non-singular at τ slot |
| └─ tau_error NaN (singular cov) | 123 | 123 | fit converged, σ_τ undefined |
| ∧ no fixed contributors | 17 | 138 | |
| ∧ max free-peak SNR ≥ 10 | 17 | 138 | non-binding (already enforced by snr_proxy ≥ 10) |
| ∧ χ²_r < 3 | 16 | 95 | drops 1 finite-σ + 43 singular |
| ∧ τ not saturating bounds | 16 | 93 | wid 86, 190 saturate (both finite-σ) |
| ∧ σ_τ/τ < 0.10 (strict only) | **15** | (skip) | wid 52 fails (rel = 0.155) |
| singular-cov passes (expanded only) | (n/a) | **73** | new contribution |
| **Final** | **15** | **90** | |

The histogram (figure 12) shows the expanded sample as a single broad
mode centered near 6.3 µs with a long right tail to ~10 µs (low-band
windows). The frequency-third scatter (figure 13) overlays per-third
medians for LSQ strict / LSQ expanded / STFT polish=False / STFT
polish=True so the bias-flip pattern is visually unambiguous.

### Caveats

- Strict sample is small (N=15) and skewed toward the low band. The
  median is the robust descriptor; the Gaussian fit is illustrative.
- Per-third LSQ N (27 / 33 / 30) is modest vs ~1000-2000 STFT
  contributors per third. The systematic direction of the bias-flip is
  robust; the exact magnitudes carry sampling uncertainty.
- The LSQ filter selects "cleaner" windows; windows with severe
  leakage from out-of-window strong lines (carried-frozen-contributor
  cases) are excluded. The STFT averages over all in-band contributors
  including those windows' bins.

---

## Polish design

The log-linear regression in step 2 of § Method carries a +3-5 %
positive bias at intermediate `T_full / τ` ratios (Case 1).
`extract_tau_majority` exposes three knobs that shape the response.

### NLS polish (`polish=True`, default on)

One Gauss-Newton step on `|S_n| = C · exp(-a/τ)` per contributor bin
before the SNR-weighted majority. Vectorised across the contributor
set. Polished Case 1 (numbers: median majority τ error % over 8
trials, N_seg = 10, SNR = 100):

| T_full \ τ (µs) | 3.0 | 5.0 | 7.5 | 12.0 | 20.0 |
|---|---|---|---|---|---|
| 5.00  | -1.4 | -3.0 | -4.7 | -7.7 | -14.8 |
| 10.00 | +0.5 | +0.8 | +0.6 | +0.9 | +0.9 |
| 12.65 | +1.9 | +1.9 | +2.1 | +2.2 | +2.3 |
| 20.00 | +0.7 | +0.5 | +0.4 | +0.4 | +0.9 |
| 30.00 | -0.4 | -0.2 | -0.5 | -0.3 | -0.0 |
| 40.00 | -0.2 | -0.1 | +0.1 | -0.4 | -0.1 |

Closes the intermediate-T_full bias to ≤ 2.3 % everywhere. The
long-τ-at-short-T pathological corner (T_full = 5) is unchanged
because the polish can't manufacture decay that isn't in the data.
The residual +2 % at T_full = 12.65 traces to a noise-floor
contribution: late STFT frames at this intermediate `T_full / τ`
ratio sit at signal ~ noise, where Rician statistics tilt apparent τ
upward. Figures:
[`figures/10_polish_case1.png`](figures/10_polish_case1.png),
[`figures/11_polish_2638shape.png`](figures/11_polish_2638shape.png);
cached numerics
[`data/polish_case1.npz`](data/polish_case1.npz) and
[`data/polish_2638shape.json`](data/polish_2638shape.json). Script:
[`polish_validation.py`](polish_validation.py).

### Per-bin SNR cap (`polish_snr_cap`, default `DEFAULT_POLISH_SNR_CAP = 9.0`)

Restricts the polish to contributors whose per-bin SNR is **below**
the cap; high-SNR contributors retain the unpolished log-linear seed.
The +3-5 % log-linear bias concentrates at modest SNR — at high
per-bin SNR the log-linear regression is already near-unbiased, so
the same Gauss-Newton step over-corrects there. Without a cap, the
polish biases the per-band SNR-weighted majority low on 2638 (low
-8.3 %, mid -8.5 %, high -8.1 % vs LSQ); the cap removes the upper
tail of the contributor SNR distribution where the polish hurts more
than it helps.

Per-band SNR-weighted majority τ on 2638 (the quantity Stage 5 routes
on via `per_band_tau=True`) at the three polish configurations:

| config | low maj | mid maj | high maj | worst \|Δ\| |
|---|---|---|---|---|
| polish=False | 7.71 (-2.0 %) | 6.44 (+2.7 %) | 5.66 (+9.7 %) | 9.7 % |
| polish=True, cap=None | 7.22 (-8.3 %) | 5.73 (-8.5 %) | 4.74 (-8.1 %) | 8.5 % |
| **polish=True, cap=9.0 (default)** | **7.62 (-3.2 %)** | **6.16 (-1.8 %)** | **5.29 (+2.4 %)** | **3.2 %** |

The default cap = 9 was selected from a 2-axis sweep over `polish_snr_cap
∈ {None, 30, 20, 15, 12, 11, 10, 9, 8} × relative_gate_fraction ∈
{0.05, 0.10, 0.15, 0.20, 0.30, 0.50}`; the figure
[`figures/14_polish_snr_cap.png`](figures/14_polish_snr_cap.png)
shows the worst-case per-band |Δ| heatmap, with the production
operating point starred. Cap values in [8, 12] all pass the 5 %
acceptance gate; 9 is the worst-case minimum. Cached numerics:
[`data/polish_snr_cap.json`](data/polish_snr_cap.json). Script:
[`polish_snr_cap_validation.py`](polish_snr_cap_validation.py).

The band-wide τ_maj at the default cap (5.96 µs) sits between the two
polish endpoints by design; it's a secondary metric when per-band
routing is active. Callers using band-wide τ_maj only should be aware
of the -4.8 % shift from the LSQ band-wide reference (6.26 µs).

**Why the cap range is 5-30, not 100-200.** The original hypothesis
was that polish over-corrects at SNR ≫ 100 where the log-linear seed
is already unbiased. Empirically on 2638, the contributor SNR
distribution maxes at ~82 (median ~8); strong on-line bins (per-frame
SNR 240-360) are classified as `bad-fit` by `stft_calibration`,
*not* as contributors. Their `rss_exp` exceeds the relative gate
because real molecular lines aren't pure single-exponentials — line
shape, Doppler, saturation inflate the per-bin residual above the
5 %-of-mean budget. The bad-fit gate already does the high-SNR
exclusion the hypothesis worried about; the polish sees only
intermediate-SNR bins, and the cap removes the upper tail of *that*
distribution.

**Why an SNR cap and not a wider `relative_gate_fraction`.** Loosening
the bad-fit gate from 0.05 to 0.20 brings the strong on-line bins
into the contributor set and buys an extra ~0.5 % on the worst-case
per-band majority (down to 2.6 %). But the bad-fit gate is a global
STFT classifier — its behaviour affects every fixture's spur
classification, every synthetic test case, and the n_contributors
acceptance pre-conditions. The single-knob `polish_snr_cap` change
captures most of the win without that global classifier disturbance.

**The right acceptance metric is per-band SNR-weighted majority τ,
not per-third median τ.** Stage 5 routes each window to its band's
`BandMajority.tau_maj_us`, which uses the SNR-weighted quantile.
The per-third median treats every contributor equally, so it answers
a subtly different question. At `polish_snr_cap=10` the per-third
median worst-case is 5.5 % but the per-band majority worst-case is
4.0 % — the wider acceptance region under the majority metric is what
made cap=9 feasible at the shipped `relative_gate_fraction=0.05`.

### Rician noise debias (`polish_noise_debias=True`, opt-in)

Replaces `|S_n|` with the Rician-unbiased magnitude `sqrt(max(0,
|S_n|² - 2σ²))` inside the polish step. Theoretically correct for
Gaussian complex noise on a single isolated line — closes the
single-line Case-1 grid to sub-1 % everywhere. But on multi-line
spectra the per-bin "noise" includes inter-line skirt interference
that the Rician model doesn't capture, and the debiasing over-corrects:

| variant on 2638 | τ_maj (µs) | err vs LSQ band-wide (6.26) |
|---|---|---|
| polish=False | 6.328 | +1 % |
| polish=True, cap=None | 5.512 | -12 % |
| polish=True, cap=9 (default) | 5.96 | -5 % |
| polish=True + noise_debias | 4.37 | -30 % |

A 2638-shape multi-line synthetic with controlled `τ(f)` (7.5 → 6 µs
across the trim band) and `SNR(f)` (1× → 3×) lands polish=True +
debias at -4.9 % below truth, vs +2.7 % for polish=False and -1.1 %
for polish=True without debias. The debias remains an opt-in forensic
knob for single-isolated-line work.

---

## Production wiring summary

`extract_tau_majority` is the entry point; see the
[planning doc § "Polish step"](../../planning/stage2b-tau-calibration.md)
for the normative spec and the per-band routing details. The
production defaults emerge from this report's findings:

- `polish=True, polish_snr_cap=9` — per-band majority τ within ±3.2 %
  of the LSQ reference on 2638.
- `compute_band_majorities=True` opt-in on `Pipeline.calibrate_tau`
  populates the `BandMajority` tuple; Stage 5 consumes per-band τs
  via `Pipeline.fit_peaks(per_band_tau=True)`.
- The bidirectional Gaussian-prior τ penalty in Stage 5 uses
  `(τ_maj, σ_τ)` (or per-band when routed) as the anchor;
  `σ_τ` floors at 0.5 µs to prevent over-confident priors.
- `preconditions_passed = False` on 2638 (σ_τ/τ_maj = 0.27 > 0.20);
  the dominant-cluster weight clears the multimodality policy so
  downstream consumers still trust `τ_maj`. A per-band calibration
  tightens the σ_τ/τ_maj fraction directly because each band's
  spread is narrower than the band-wide.

---

## Known limitations

- **Cross-fixture confidence.** All numbers in this report come from
  a single fixture (2638). The polish_snr_cap=9 default was
  calibrated on 2638's contributor SNR distribution; another
  instrument with a substantially different per-bin SNR ceiling may
  need a different cap. Tracked under the broader
  cross-fixture-validation programme.
- **Per-species τ.** If a real multi-species fixture shows GMM
  multimodality with the minor cluster representing a distinct
  species (not just a band-of-the-same-species artefact as on 2638),
  Stage 5 would need to consume a per-cluster τ. Substantial
  structural change (every `FixedContributor` / `FreePeak` needs a τ
  pointer); deferred until at least one fixture demands it.
- **The bad-fit gate excludes strong on-line bins.** The strongest
  bins (per-frame SNR 240-360 on 2638) classify as bad-fit because
  real lines aren't pure single-exponentials. Their τ is recoverable
  with a more elaborate fit (e.g. fitting a Voigt-decay) but at the
  cost of the calibration's "single-parameter, no LSQ" property.
  Acceptable as long as the contributor population is large and
  representative — which it is on 2638 (4417 contributors across the
  trim band).
- **`σ_τ / τ_maj < 0.20` pre-condition.** 2638 fails this band-wide
  (0.27) because of the real frequency-dependent τ structure; the
  pre-condition was tuned against synthetic single-τ fixtures. The
  per-band majorities have narrower spreads and pass the pre-condition
  within each band.
