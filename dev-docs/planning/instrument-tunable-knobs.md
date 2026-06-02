# Instrument-tunable knob defaults

Cross-stage reference of every settings field's current hard default, physical
meaning, and instrument-sensitivity rating. The purpose is to decide which
defaults are correct for the 2638 (BlackChirp) reference instrument and which
should be revisited when a new instrument preset is added.

This document is a **reference**, not a planning doc tracking active work. The
parent migration project is [`settings-backfill.md`](settings-backfill.md)
Follow-up #2; that's where status updates against this table land.

## How to read this

Each table has six columns:

| column | meaning |
|--------|---------|
| **field** | `sub_block.field_name` as it appears on the dataclass / in YAML / in HDF5. |
| **default** | The hard default from `_HARD_DEFAULTS` in the matching `core/*_settings.py` module. |
| **source** | Where the kernel's `DEFAULT_*` constant lives (the readable canonical source the dataclass mirrors). |
| **meaning** | One-line physical description of what the value controls. |
| **inst-sens** | Instrument-sensitivity rating: **Y** depends on hardware (sample rate, T_acquire, probe band, noise floor, line shape); **N** is pure algorithmic conditioning (convergence tolerance, recursion cap, structural search limit); **maybe** depends on chemistry/sample under test, or could be calibrated per instrument but is not strictly required. |
| **2638** | Value from [`instrument_bc_2638.yaml`](../../src/ftmwpipeline/presets/instrument_bc_2638.yaml), or `—` if unset (i.e. inherits the hard default). |

The Y / N / maybe ratings are best-effort calls based on what the knob
controls; the final per-instrument decision is the user's. Treat **maybe**
as "needs experimental confirmation either way" rather than "probably N."

## 2638 preset coverage

`instrument_bc_2638.yaml` carries **no Stage 5 overrides**. Every field the
preset used to set (shape, per_band_tau, tau_penalty_lambda) is now a
package-wide hard default or, in the case of `shape.kind`, stamped onto
the file automatically by Stage 2b's auto-recommendation pass. The
preset is kept on disk as a stable name workflows can pin for future
2638-specific knobs — but currently it is metadata-only.

Stages 2, 2b, 3, and 4 inherit the package hard defaults on 2638 —
including several knobs rated **Y** below. That gap is the headline
driver for the per-instrument calibration audit (see *Open follow-ups*
at the end): any knob marked Y with `2638 = —` is a candidate for
per-instrument calibration on this fixture, even though 2638 has been
the primary calibration target throughout development.

### Resolved: the three former 2638 overrides are now defaults

* `shape.kind` — Stage 2b's `auto_recommend` pass (now the default in
  `RecommendationSubSettings`) computes the 3-way L/G/V verdict
  inside `calibrate_tau` / `calibrate_tau_G` and stamps the
  ``recommended_shape`` attr; the Stage 5 resolver's *recommended*
  layer picks it up automatically. The package hard default
  (LORENTZIAN) remains as the fallback when no Stage 2b calibration
  is present.
* `tau.per_band_tau` — was already the package hard default
  (`True`); the preset override was redundant and is dropped.
* `tau.tau_penalty_lambda` — the 2638 sweep cliff evidence
  (λ=0 → λ=50 closes compliance from 28 % to 92 % at a 5 % bulk-χ²ᵣ
  cost; higher λ degrades the tail) is a generic regularization-
  strength statement, not a hardware-specific calibration. The
  package hard default is now `50` (was `500`); the matching
  preset override is dropped.

## Stage 2 — `NoiseSettings` (scatter estimator)

Source: [`preprocessing/noise_estimation.py`](../../src/ftmwpipeline/preprocessing/noise_estimation.py).
Planning: [`stage2-noise-estimation.md`](stage2-noise-estimation.md).

The **scatter estimator is the sole Stage 2 method**; its knobs are the flat
`NoiseSettings` table further below (`stage2.<field>`, no sub-block). The legacy
adaptive estimator
(`estimate_noise_adaptive`) was retired as a user-facing method — its kernel
survives only as an internal helper (Stage 3 display fallback, Stage 5
per-window active-FT noise), tracked for removal in
[`stage2-noise-authority.md`](stage2-noise-authority.md). The adaptive knob
table below is retained for historical reference only; these fields no longer
appear on any settings dataclass, CLI flag, or `tune` knob.

### Adaptive estimator (retired internal helper) — `estimate_noise_adaptive`

| field | default | source | meaning | inst-sens | 2638 |
|---|---|---|---|---|---|
| binning.subdivision_threshold | 0.08 | `SUBDIVISION_THRESHOLD` | Relative magnitude/MAD threshold for recursive spectrum subdivision (Rayleigh sample robustness). | N | — |
| binning.abs_min_bin_size | 300 | `ABS_MIN_BIN_SIZE` | Minimum frequency-grid points per bin (noise-estimation stability floor). | N | — |
| binning.min_bin_fraction | 1/64 | function default | Minimum bin size as fraction of total grid length. | N | — |
| binning.min_noise_fraction | 2/3 | function default | Minimum fraction of trimmed-noise samples per bin half (skewness gate). | N | — |
| skewness.skew_target | 0.631 | function default | Target sample skewness for Rayleigh-like noise trimming (Rayleigh = 0.631). | N | — |
| skewness.inc | 0.01 | function default | Rank-step granularity for trimmed-sample skewness scan. | N | — |
| smoothing.smoothing_window_mhz | 300.0 | `DEFAULT_SMOOTHING_MHZ` | Moving-window size for per-point noise-estimate interpolation (MHz). | **Y** | — |
| skirt_exclusion.strong_peak_snr | 20.0 | `STRONG_PEAK_SNR` | SNR threshold above which strong lines trigger explicit Lorentzian-skirt masking. | **Y** | — |
| skirt_exclusion.skirt_exclusion_k | 1.5 | `SKIRT_EXCLUSION_K` | Lorentzian skirt radius in units of (HWHM × SNR / k) for exclusion. | **Y** | — |
| skirt_exclusion.max_skirt_exclusion_mhz | 500.0 | `MAX_SKIRT_EXCLUSION_MHZ` | Maximum per-line skirt-exclusion radius (caps pathologically strong peaks). | **Y** | — |

### Stage 2 (scatter estimator) — `estimate_noise_scatter`

Source: [`preprocessing/noise_estimation.py`](../../src/ftmwpipeline/preprocessing/noise_estimation.py).
Research: [`noise-snr-scaling/report.md`](../research/noise-snr-scaling/report.md).

The scatter (high-pass), region-aware estimator is the canonical and only
Stage 2 method: pedestal-immune on high-SNR, line-dense spectra. Its knobs sit
directly on `NoiseSettings` (a flat dataclass — Stage 2 has one estimator) and
resolve through the same four-layer chain as Stages 2b/5. They are
instrument-family-dependent because they encode the physical scale over which
σ(f) and the leakage pedestal vary, plus a line-detection threshold.

| field | default | source | meaning | inst-sens | 2638 |
|---|---|---|---|---|---|
| window_mhz | 80.0 | `SCATTER_WINDOW_MHZ` | Width of the per-region scatter-MAD window (MHz); the scale over which σ(f) is treated as constant. Same role as the adaptive `smoothing_window_mhz`. | **Y** | — |
| pedestal_mhz | 20.0 | `SCATTER_PEDESTAL_MHZ` | Running-median width (MHz) of the high-pass that isolates the smooth leakage pedestal from the white noise. Must be broader than the noise correlation length yet narrower than the pedestal's own curvature (set by line density + FT settings). | **Y** | — |
| line_k | 8.0 | `SCATTER_LINE_K` | Robust-σ multiple of the high-passed residual above which a bin is self-masked as a line. A detection threshold — depends on the sample's SNR and line density. | maybe | — |
| n_iter | 3 | `SCATTER_N_ITER` | Self-mask refinement iterations (interpolate masked lines → re-estimate pedestal). Algorithmic convergence, not hardware. | N | — |
| smoothing_mhz | 800.0 | `SCATTER_SMOOTHING_MHZ` | Width (MHz) of the broad moving-percentile σ smoothing — a lower-envelope median that rides the noise floor through line-dense bands. Must be wide enough to span the instrument's worst line clusters yet not erase real (slow) σ(f) structure. `0` disables. | **Y** | — |
| smoothing_percentile | 50.0 | `SCATTER_SMOOTHING_PERCENTILE` | Percentile of the smoothing filter. 50 = median (unbiased on clean spectrum, robust to ≤50 % per-window line contamination); lower = more aggressive floor de-inflation under wide dense bands at the cost of a clean-region low bias. Depends on sample line density. | maybe | — |
| convolve_mhz | 200.0 | `SCATTER_CONVOLVE_MHZ` | Gaussian σ (MHz) of the second smoothing pass that removes the median's staircase steps. Acts on the de-inflated median output so it cannot re-inflate under lines. Mostly algorithmic (cosmetic smoothness); keep well below `smoothing_mhz` so it does not broaden real σ(f) structure. `0` disables. | N | — |

`region_aware` (default `True`) selects the Rician `C(R)` lookup over a fixed
mid-regime factor; it is an algorithmic correctness switch, not an
instrument-calibrated knob (leave it on).

## Start detection (pre-Stage 1) — `StartDetectionSettings`

Source: [`preprocessing/start_detection.py`](../../src/ftmwpipeline/preprocessing/start_detection.py).
Flat frozen dataclass (no resolution chain); the only persisted output is the
recommended `start_us` stamped into the Stage 0 `recommended_processing` layer.

| field | default | source | meaning | inst-sens | 2638 |
|---|---|---|---|---|---|
| sweep_max_us | 7.5 | dataclass default | Upper bound of the start-time sweep (capped to FID duration). Must clear the chirp end + the floor-estimate tail. | **Y** | — |
| step_us | 0.02 | dataclass default | Sweep step; resolution of the chirp-end corner. | maybe | — |
| zpf | 0 | dataclass default | Zero-padding for the per-start FT; 0 is correct (integrated magnitude needs no interpolation). | N | — |
| floor_factor | 3.0 | dataclass default | Chirp-end = first start where Σ\|FT\| < factor × deep-tail floor. | N | — |
| floor_tail_us | 1.0 | dataclass default | Width of the deep-tail window for the robust floor estimate. | maybe | — |
| guard_margin_us | 0.67 | dataclass default | Margin added past the chirp end for switch-bounce ringdown settling. **The primary recommendation is chirp_end + this.** Tuned on 2638-family (chirp_dur+1.35 targets ⇒ chirp_end+0.67). | **Y** | — |
| min_chirp_drop_ratio | 10.0 | dataclass default | Min plateau/floor ratio for a chirp collapse to be considered present. | **Y** | — |

`guard_margin_us` is the headline instrument-specific knob: it is the
switch-bounce ringdown length, which varies with the instrument's RF
switch/protection hardware. Re-tune per instrument (the chirp-end corner itself
is hardware-robust; only the post-chirp margin moves).

## Stage 2b — `TauCalibrationSettings`

Source: [`fitting/tau_calibration.py`](../../src/ftmwpipeline/fitting/tau_calibration.py).
Planning: [`stage2b-tau-calibration.md`](stage2b-tau-calibration.md).

| field | default | source | meaning | inst-sens | 2638 |
|---|---|---|---|---|---|
| stft.n_seg | 10 | `DEFAULT_N_SEG` | Number of non-overlapping STFT frames (window = T_full / n_seg). | **Y** | — |
| stft.t_sigma | 5.0 | `DEFAULT_T_SIGMA` | Above-threshold SNR gate for per-frame signal detection (contributor floor). | **Y** | — |
| stft.tau_max_factor | 5.0 | `DEFAULT_TAU_MAX_FACTOR` | Upper clip on τ as multiple of full-record acquisition time (spur candidate). | N | — |
| stft.rss_gate_factor | 5.0 | `DEFAULT_RSS_GATE_FACTOR` | Hybrid-gate strength: bad-fit RSS threshold relative to per-frame noise. | N | — |
| stft.relative_gate_fraction | 0.05 | `DEFAULT_RELATIVE_GATE_FRACTION` | Relative branch of RSS gate (RSS > factor × (fraction × mean_mag)²). | N | — |
| polish.polish | True | function default | Enable Gauss-Newton polish on log-linear exponential seed. | N | — |
| polish.polish_n_iter | 1 | function default | Gauss-Newton iterations per bin (removes +3-5 % log-linear bias). | N | — |
| polish.polish_snr_cap | 9.0 | `DEFAULT_POLISH_SNR_CAP` | SNR above which polish is skipped (already near-unbiased; avoid over-correction). | **Y** | — |
| polish.polish_noise_debias | False | function default | Apply Rician-unbiased magnitude on high-SNR frames (removes residual +1-2 % bias). | **Y** | — |
| aggregation.min_contributors | 200 | `DEFAULT_MIN_CONTRIBUTORS` | Minimum contributor bins for validity of the calibration result. | N | — |
| aggregation.sigma_tau_fraction_max | 0.20 | `DEFAULT_SIGMA_TAU_FRACTION_MAX` | Maximum relative uncertainty (σ_τ / τ_maj) for acceptance. | N | — |
| aggregation.bimodality_dominant_fraction | 0.70 | `DEFAULT_BIMODALITY_DOMINANT_FRACTION` | Minimum dominant-cluster weight when two-component mixture preferred. | N | — |
| aggregation.sigma_tau_floor_us | 0.5 | `DEFAULT_SIGMA_TAU_FLOOR_US` | Minimum per-band σ_τ (prevents over-confident penalties on tight clusters). | maybe | — |
| aggregation.spur_cluster_multiplier | 1.0 | `DEFAULT_SPUR_CLUSTER_MULTIPLIER` | STFT spur-bin clustering gap in units of per-segment frequency bins. | N | — |
| band.compute_band_majorities | True | function default | Compute per-band SNR-weighted τ majorities for Stage 5 per-band routing. | maybe | — |
| band.min_contributors_per_band | 50 | function default | Minimum contributors per frequency band (fallback to band-wide if not met). | N | — |
| gaussian.snr_min | 20.0 | `DEFAULT_TAU_G_SNR_MIN` | SNR minimum for pure-Gaussian model eligibility. | **Y** | — |
| gaussian.tau_G_bound_lo | 0.5 | `DEFAULT_TAU_G_BOUND_LO` | Lower bound on Gaussian envelope decay constant τ_G (µs). | N | — |
| gaussian.tau_G_bound_hi | 100.0 | `DEFAULT_TAU_G_BOUND_HI` | Upper bound on Gaussian envelope decay constant τ_G (µs). | N | — |
| gaussian.tau_G_seeds | (100, 50, 20, 10, 5, 3) | `DEFAULT_TAU_G_SEEDS` | Initial-guess grid for Gaussian NLS multistart optimization. | N | — |
| gaussian.delta_chi2r_min | 1.0 | `DEFAULT_TAU_G_DELTA_CHI2R_MIN` | Minimum χ²ᵣ difference (exp − gauss) for Gaussian eligibility. | N | — |
| gaussian.tau_G_upper_fraction | 0.7 | `DEFAULT_TAU_G_UPPER_FRACTION` | Maximum on τ_G as fraction of calibration acquisition T. | N | — |
| gaussian.min_contributors | 50 | `DEFAULT_TAU_G_MIN_CONTRIBUTORS` | Minimum Gaussian-eligible bins for pure-Gaussian calibration path. | N | — |
| recommendation.snr_min | 20.0 | `DEFAULT_TAU_G_SNR_MIN` *(reused)* | SNR minimum for shape-recommendation voting. | **Y** | — |
| recommendation.tau_bound_lo | 0.5 | `DEFAULT_TAU_G_BOUND_LO` *(reused)* | Lower τ bound for recommendation eligibility (µs). | N | — |
| recommendation.tau_bound_hi | 100.0 | `DEFAULT_TAU_G_BOUND_HI` *(reused)* | Upper τ bound for recommendation eligibility (µs). | N | — |
| recommendation.pure_margin_threshold | 0.10 | `DEFAULT_SHAPE_RECOMMENDATION_PURE_MARGIN` | Minimum vote margin for one pure shape (L or G) to dominate Voigt. | N | — |
| recommendation.auto_recommend | True | dataclass-only | Auto-run `compute_shape_recommendation` after `calibrate_tau` / `calibrate_tau_G`. | N | — |

## Stage 3 — `PeakDetectionSettings`

Source: [`preprocessing/peak_detection.py`](../../src/ftmwpipeline/preprocessing/peak_detection.py) and [`_internal/stage3_impl.py`](../../src/ftmwpipeline/_internal/stage3_impl.py).
Planning: [`stage3-peak-detection.md`](stage3-peak-detection.md).

| field | default | source | meaning | inst-sens | 2638 |
|---|---|---|---|---|---|
| promotion.min_snr | 3.0 | `DEFAULT_MIN_SNR` | Detection floor: SNR cutoff for Stage 4 promotion (user-grid measurement). | **Y** | — |
| promotion.internal_min_snr | 2.0 | `DEFAULT_INTERNAL_MIN_SNR` | Internal detection floor on native zpf=1 grid (recovers lines apodization smears). | **Y** | — |
| promotion.weak_medium_snr | 10.0 | `DEFAULT_WEAK_MEDIUM_SNR` | Weak/medium SNR boundary for classification (provisional, O2). | **Y** | — |
| promotion.medium_strong_snr | 50.0 | `DEFAULT_MEDIUM_STRONG_SNR` | Medium/strong SNR boundary for classification (provisional, O2). | **Y** | — |
| savgol.sg_window | 11 | function default | Savitzky-Golay filter window size (bins, must be odd). | N | — |
| savgol.sg_order | 3 | function default | Savitzky-Golay polynomial order. | N | — |
| savgol.sg_fwhm_coverage | 4.0 | `_SG_FWHM_COVERAGE` | Window-size target in units of line FWHM (sg_window auto-derived at runtime). | N | — |
| savgol.sg_min_window | 5 | `_SG_MIN_WINDOW` | Minimum Savitzky-Golay window size (polynomial stability floor). | N | — |
| primary_pass.primary_window | "blackmanharris" | `DEFAULT_PRIMARY_WINDOW` | Apodization function for primary-pass position finding (sidelobe suppression). | N | — |
| primary_pass.min_exclusion_mhz | 0.0 | function default | Minimum half-width exclusion around each primary peak for gap pass (MHz). | **Y** | — |
| primary_pass.detection_zpf | 1 | `_DETECTION_ZPF` | Zero-padding factor for primary-pass spectrum computation. | N | — |
| gap_pass.run_gap_pass | True | function default | Enable second pass to recover weak lines primary-pass apodization suppressed. | N | — |
| gap_pass.gap_active_zpf | 2 | `_GAP_ACTIVE_ZPF` | Zero-padding factor for matched-filter active-region FFT. | N | — |
| gap_pass.gap_mask_edge_threshold | 8.0 | `GAP_MASK_EDGE_THRESHOLD` | Coherent-leakage threshold for masking truncation sidelobes in gap pass. | **Y** | — |

## Stage 4 — `WindowPlanningSettings`

Source: [`preprocessing/window_planning.py`](../../src/ftmwpipeline/preprocessing/window_planning.py) and [`preprocessing/edge_coherence.py`](../../src/ftmwpipeline/preprocessing/edge_coherence.py).
Planning: [`stage4-window-assignment.md`](stage4-window-assignment.md).

| field | default | source | meaning | inst-sens | 2638 |
|---|---|---|---|---|---|
| coherence.edge_m | 64 | `DEFAULT_EDGE_M` | Band width for rolling complex-edge coherence statistic (cache-sized, null-tightest). | N | — |
| coherence.trim_m | 32 | `DEFAULT_TRIM_M` | Band width for coherence refinement after leakage-region flag (finer spatial scale). | N | — |
| coherence.edge_threshold | 8.0 | `DEFAULT_EDGE_THRESHOLD` | S_coh threshold for leakage-touched-region detection (T_edge = √M at M=64). | **Y** | — |
| clustering.max_window_width_mhz | 40.0 | `DEFAULT_MAX_WINDOW_WIDTH_MHZ` | Window-width cap; windows exceeding this are HARD and get split proposals (MHz). | **Y** | — |
| clustering.min_window_half_width_mhz | 2.0 | `DEFAULT_MIN_WINDOW_HALF_WIDTH_MHZ` | Minimum half-width of isolated-peak proposed windows (MHz). | maybe | — |
| contributor.min_freeze_snr | 50.0 | `DEFAULT_MIN_FREEZE_SNR` | SNR floor for fixed-contributor freeze-eligibility (O4-2); below = thaw candidate. | **Y** | — |
| contributor.magnitude_attachment_threshold | 0.1 | `DEFAULT_MAGNITUDE_ATTACHMENT_THRESHOLD` | Tier-1 contributor attachment: predicted mean-skirt threshold in σ_c units. | **Y** | — |
| leakage.tau_us | None | — | Decay constant for analytic leakage-skirt envelope (None = boxcar / undamped limit). | **Y** | — |

## Stage 5 — `StageFitSettings`

Source: [`fitting/`](../../src/ftmwpipeline/fitting/) (`plan_execution.py`, `peak_model.py`, …).
Planning: [`stage5-fit-settings.md`](stage5-fit-settings.md), [`stage5-fitting.md`](stage5-fitting.md).

| field | default | source | meaning | inst-sens | 2638 |
|---|---|---|---|---|---|
| shape.kind | LORENTZIAN | `PeakShape.LORENTZIAN` *(literal default)* | Line-shape model selector: LORENTZIAN or GAUSSIAN envelope. Stage 2b's `auto_recommend` stamps the verdict onto the file. | maybe | — |
| tau.max_decay_factor | 5.0 | `DEFAULT_MAX_DECAY_FACTOR` | Tau bounds multiplier: τ ∈ [τ₀ / k, τ₀ × k] (O5-4 hard cap). | N | — |
| tau.fit_tau_min_snr | 50.0 | dataclass-only | SNR threshold above which τ becomes a free parameter (fixed below). | **Y** | — |
| tau.tau_penalty_lambda | 50.0 | `DEFAULT_TAU_PENALTY_LAMBDA` | Strength of bidirectional Gaussian prior on τ. | N | — |
| tau.tau_penalty_n_sigma | 5.0 | `DEFAULT_TAU_PENALTY_N_SIGMA` | τ-bound half-width in units of σ_τ from Stage 2b calibration. | N | — |
| tau.per_band_tau | True | function default | Route τ to per-band majorities (True) or band-wide (False). | maybe | — |
| seeder.seeder_rchi2 | 1.5 | `DEFAULT_SEEDER_RCHI2` | χ²ᵣ threshold: triggers K=2/K=3 blend-aware re-seed on single-peak fit. | N | — |
| seeder.seeder_straddle_factor | 1.0 | `DEFAULT_SEEDER_STRADDLE_FACTOR` | Re-seed offset grid spacing in units of line FWHM (blend resolution). | N | — |
| seeder.seeder_max_k | 3 | `DEFAULT_SEEDER_MAX_K` | Maximum escalation depth (K_initial=1 → K_max on blend detection). | N | — |
| conservative.significance | 0.05 | `DEFAULT_SIGNIFICANCE` | F-test significance level for add-one-peak acceptance (α). | N | — |
| conservative.max_peaks | 8 | `DEFAULT_MAX_PEAKS` | Hard cap on final peak count per window. | N | — |
| conservative.patience | 1 | `DEFAULT_PATIENCE` | Consecutive-rejection patience: drop loop after this many fails. | N | — |
| conservative.min_separation_factor | 1.0 | `DEFAULT_MIN_SEPARATION_FACTOR` | Minimum peak-to-peak separation in units of FWHM (unresolvable below). | N | — |
| conservative.min_pair_separation_factor | 0.5 | `DEFAULT_MIN_PAIR_SEPARATION_FACTOR` | Sanity-check floor on post-escalation peak pairs (reject if below), in FWHM units. | N | — |
| conservative.min_pair_separation_resolution_factor | 1.0 | `DEFAULT_MIN_PAIR_SEPARATION_RESOLUTION_FACTOR` | Resolution-referenced floor on the minimum pair separation, in active-FT elements `1/T_active`; effective floor is `max(min_pair_separation_factor·FWHM, this·(1/T_active))`. Gates sub-resolution duplicate overfits (issue #13). | N | cross-fixture `k` + amp-ratio tiebreaker debt |
| conservative.n_eff_kind | "perplexity_log1p_snr" | `DEFAULT_N_EFF_KIND` | Effective-sample-size weighting (perplexity_log1p_snr vs kish_mag_sq). | N | — |
| conservative.weak_window_snr_threshold | 10.0 | `DEFAULT_WEAK_WINDOW_SNR_THRESHOLD` | In-window SNR floor for free-τ eligibility (hold τ fixed below). | **Y** | — |
| conservative.max_nfev | 2000 | `DEFAULT_MAX_NFEV` | Solver evaluation cap (prevents runaway on ill-conditioned problems). | N | — |
| penalties.phase_penalty_lambda | 100.0 | `DEFAULT_PHASE_PENALTY_LAMBDA` | Soft phase-difference penalty strength (prevents in-/anti-phase degeneracy). | N | — |
| penalties.phase_penalty_cutoff_fwhm | 2.0 | `DEFAULT_PHASE_PENALTY_CUTOFF_FWHM` | Phase-penalty range: weak at this spacing, zero in quadrature. | N | — |
| penalties.amp_penalty_lambda | 10.0 | `DEFAULT_AMP_PENALTY_LAMBDA` | Soft amplitude-floor penalty strength (pushes noise-level peaks toward 0). | N | — |
| penalties.amp_max_headroom | 3.0 | `DEFAULT_AMP_MAX_HEADROOM` | Hard amplitude ceiling as multiple of (2 × max_data / τ_eff_min). | N | — |
| rescue.max_rounds | 5 | `DEFAULT_RESCUE_MAX_ROUNDS` | Maximum residual-rescue iterations per window (safety cap). | N | — |
| rescue.snr_threshold | 2.5 | `DEFAULT_RESCUE_SNR_THRESHOLD` | Residual-peak detection floor (nominates generously, F-test gates). | **Y** | — |
| rescue.prominence_threshold | 2.0 | `DEFAULT_RESCUE_PROMINENCE_THRESHOLD` | Residual-peak prominence threshold for candidate nomination. | **Y** | — |
| rescue.cleanup_significance | 0.05 | `DEFAULT_CLEANUP_SIGNIFICANCE` | F-test significance for remove-and-refit post-rescue cleanup. | N | — |
| rescue.merge_separation_factor | 0.5 | `DEFAULT_MERGE_SEPARATION_FACTOR` | AICc-gated merge threshold above-resolution (FWHM units). | N | — |
| rescue.structural_merge_factor | 0.5 | `DEFAULT_STRUCTURAL_MERGE_FACTOR` | Sub-resolution merge floor: pairs closer than this FWHM collapse unconditionally. | N | — |
| rescue.overfit_amp_ratio_band | 1.5 | `DEFAULT_OVERFIT_AMP_RATIO_BAND` | Upper bound (in `1/T_active` resolution elements) of the amplitude-ratio merge tier that collapses supra-resolution shape-error absorbers (issue #13). | N | cross-fixture calibration debt |
| rescue.overfit_amp_ratio_threshold | 6.0 | `DEFAULT_OVERFIT_AMP_RATIO_THRESHOLD` | Larger/smaller amplitude ratio above which a pair in the amp-ratio band is collapsed as an absorber (issue #13). Set 0 to disable. | N | cross-fixture calibration debt |
| thaw.max_thaw_rounds | 2 | `DEFAULT_MAX_THAW_ROUNDS` | Maximum iterations of local-thaw (re-fit on frozen fixed contributors). | N | — |
| thaw.max_replan_rounds | 2 | `DEFAULT_MAX_REPLAN_ROUNDS` | Maximum iterations of structural-replan (window boundary moves). | N | — |
| thaw.residual_edge_threshold | 8.0 | `DEFAULT_RESIDUAL_EDGE_THRESHOLD` | S_coh threshold for residual-edge-coherence boundary violation (replan trigger). | **Y** | — |
| thaw.residual_edge_m | 32 | `DEFAULT_RESIDUAL_EDGE_M` | Band width for residual edge-coherence detection. | N | — |
| spur.enabled | True | `_HARD_DEFAULTS["spur"]` | Master switch for clock/LO-spur detection + masking. | N | — |
| spur.integer_tol_mhz | 0.04 | `DEFAULT_INTEGER_TOL_MHZ` | Max distance (MHz) from an integer MHz for the spur gate's hard integer requirement (~½ active-FT bin). | **Y** | — |
| spur.narrowness_ratio | 0.30 | `DEFAULT_NARROWNESS_RATIO` | max(neighbour)/peak below which an integer-MHz bin is sub-resolution narrow (CW tone vs real line with a skirt). | **Y** | — |
| spur.snr_threshold | 5.0 | `DEFAULT_SNR_THRESHOLD` | Peak-bin / σ_c floor for the frequency-domain spur detector. | N | — |
| spur.mask_half_width_bins | 2 | `DEFAULT_MASK_HALF_WIDTH_BINS` | Residual-mask half-width (active-FT bins) around a spur; ±2 recovers ~the full bucket, ±3 the last of the strongest spur. | **Y** | — |
| spur.use_stft_catalogue | True | `_HARD_DEFAULTS["spur"]` | Consume the persisted Stage 2b flat-spur (`saturated`) catalogue as the gate's persistence half; False = frequency-domain detector only. | N | — |
| baseline.enabled | True | `_HARD_DEFAULTS["baseline"]` | Master switch for the evidence-triggered leakage-wing complex-baseline nuisance term. | N | — |
| baseline.order | 0 | `DEFAULT_BASELINE_ORDER` | Baseline polynomial order p (0 = const, 1 = linear; quad overfits). Low order is the guardrail — too smooth to mimic a narrow line. | maybe | — |
| baseline.edge_threshold | 3.5 | `DEFAULT_BASELINE_EDGE_THRESHOLD` | S_coh threshold (max of the two residual edges) gating the baseline refit; a dedicated threshold well below thaw's 8.0. | **Y** | — |

## High-priority instrument-tunable knobs

Filtered down to the **Y**-rated knobs across all five stages — these are
the candidates for per-instrument calibration when adding a new
`instrument_*` preset. Knobs already overridden in `instrument_bc_2638`
are noted; everything else is currently riding the hard default on the 2638
fixture.

| stage | field | hard default | 2638 |
|---|---|---|---|
| 2 | smoothing.smoothing_window_mhz | 300.0 | — |
| 2 | skirt_exclusion.strong_peak_snr | 20.0 | — |
| 2 | skirt_exclusion.skirt_exclusion_k | 1.5 | — |
| 2 | skirt_exclusion.max_skirt_exclusion_mhz | 500.0 | — |
| 2b | stft.n_seg | 10 | — |
| 2b | stft.t_sigma | 5.0 | — |
| 2b | polish.polish_snr_cap | 9.0 | — |
| 2b | polish.polish_noise_debias | False | — |
| 2b | gaussian.snr_min | 20.0 | — |
| 2b | recommendation.snr_min | 20.0 | — |
| 3 | promotion.min_snr | 3.0 | — |
| 3 | promotion.internal_min_snr | 2.0 | — |
| 3 | promotion.weak_medium_snr | 10.0 | — |
| 3 | promotion.medium_strong_snr | 50.0 | — |
| 3 | primary_pass.min_exclusion_mhz | 0.0 | — |
| 3 | gap_pass.gap_mask_edge_threshold | 8.0 | — |
| 4 | coherence.edge_threshold | 8.0 | — |
| 4 | clustering.max_window_width_mhz | 40.0 | — |
| 4 | contributor.min_freeze_snr | 50.0 | — |
| 4 | contributor.magnitude_attachment_threshold | 0.1 | — |
| 4 | leakage.tau_us | None | — |
| 5 | tau.fit_tau_min_snr | 50.0 | — |
| 5 | conservative.weak_window_snr_threshold | 10.0 | — |
| 5 | rescue.snr_threshold | 2.5 | — |
| 5 | rescue.prominence_threshold | 2.0 | — |
| 5 | thaw.residual_edge_threshold | 8.0 | — |
| 5 | baseline.edge_threshold | 3.5 | — |

## Validation status (2638)

The Y-rated knobs validated so far against the 2638 fixture:

| stage | knob | shape-sensitivity on 2638 | verdict |
|---|---|---|---|
| 3 | gap-pass `tau_basis_us` source | **shape-sensitive** | shipped: shape-aware feeder routes to `τ_G_maj` when `recommended_shape='gaussian'` (`stage3_impl`). |
| 3 | `promotion.min_snr` | shape-invariant | keep default 3.0; both shapes agree to ≤ 3.5 % across 2.0–5.0. |
| 3 | `promotion.internal_min_snr` | shape-invariant | keep default 2.0; both shapes share the same 2.0 knee. |
| 3 | `gap_pass.gap_mask_edge_threshold` | shape-invariant | keep default 8.0; monotonic response on both paths. |
| 3 | `primary_pass.min_exclusion_mhz` | shape-invariant | keep default 0.0; both shapes lose ~21 % of gap detections at excl=0.5. |
| 4 | `leakage.tau_us` (boxcar vs Stage 2b τ) | **shape-invariant; boxcar wins** | keep default `None` (boxcar). Window boundaries are byte-identical across τ variants (set by `min_window_half_width_mhz` + clustering, not by reach); aggregate Stage 5 χ²ᵣ is also unchanged (≤ 0.02 median, ≤ 0.18 p95). The 23-29 worst-χ²ᵣ windows get the *identical* contributor set on every variant, so τ-feed cannot remediate them. |
| 4 | `coherence.edge_threshold` | shape-invariant | keep default 8.0; sits at hard-count plateau knee on both shapes. |
| 4 | `clustering.max_window_width_mhz` | shape-invariant | keep default 40.0; cap is effectively dormant on 2638 (only 1 outlier window approaches it). |
| 4 | `contributor.magnitude_attachment_threshold` | shape-invariant | keep default 0.10; sits mid-slope between 0.05 cliff and 0.20 step on both shapes. |

Audit reports:
[`dev-docs/research/stage3-gaussian-audit/README.md`](../research/stage3-gaussian-audit/README.md),
[`dev-docs/research/stage4-gaussian-audit/README.md`](../research/stage4-gaussian-audit/README.md).
The Stage 5 high-Y-rated knobs (`tau.fit_tau_min_snr`,
`conservative.weak_window_snr_threshold`, `rescue.snr_threshold`,
`rescue.prominence_threshold`, `thaw.residual_edge_threshold`) were
**audited cross-fixture (all 7 same-instrument fixtures, issue #3,
validate-and-document) and ship unchanged** — each behaves sanely
across the SNR span (1512 lowest → 655 extreme):

| stage | knob | default | cross-fixture verdict |
|---|---|---|---|
| 5 | `tau.fit_tau_min_snr` | 50.0 | keep. Tau-free rate rises monotonically with SNR (1512 26 % → 655 93 %); no bright window ever wrongly held. |
| 5 | `conservative.weak_window_snr_threshold` | 10.0 | keep. Weak-window regime scales with SNR (0.84 → 0.39), never degenerate. |
| 5 | `rescue.snr_threshold` | 2.5 | keep. Rescue does bounded, meaningful work everywhere (accept 0.4–0.7). |
| 5 | `rescue.prominence_threshold` | 2.0 | keep. Same; no pathological all-/no-fire. |
| 5 | `thaw.residual_edge_threshold` | 8.0 | keep. Sane trigger surface; thaw acceptance ~0 cross-fixture (near-dormant, as on 2638) — a "does thaw earn its keep" follow-up, not a threshold mistune. |

Evidence: [`dev-docs/research/stage5-cross-fixture/report.md`](../research/stage5-cross-fixture/report.md)
§"Cross-fixture knob audit"; driver `scratch/issue3_audit/audit_knobs.py`. The
Stage 2 scatter / Stage 2b STFT+classifier knobs ride on the same builds and
produce the sane per-fixture inputs that audit depends on; no per-fixture retune
indicated.

## Open follow-ups against this table

1. **Decide the per-instrument calibration set.** Walk the Y rows above
   and split them into (a) values that are correct on 2638 and should
   become the documented "good for any BlackChirp 750k-FID / 15-µs-T_full
   instrument" defaults, and (b) values that need experimental calibration
   on 2638 before any other instrument is brought up. Promote the (a) set
   to the package hard defaults and route the (b) set through
   `instrument_bc_2638.yaml`.
2. **Re-rate the `maybe` rows.** Six rows currently sit on `maybe`
   (`aggregation.sigma_tau_floor_us`, `band.compute_band_majorities`,
   `clustering.min_window_half_width_mhz`, `shape.kind`,
   `tau.per_band_tau`, `baseline.order`). Each needs a one-off study to
   confirm whether it is genuinely instrument-sensitive or just convention.
3. **Sanity-check the N rows.** A handful (`polish.polish_n_iter`,
   `seeder.seeder_max_k`, `conservative.patience`, …) are rated N
   purely on the "pure algorithmic conditioning" heuristic. Sceptical
   reviewers should spot-check those against their own intuition; any
   that turn out to be hardware-coupled should move to Y in a follow-up
   revision.
