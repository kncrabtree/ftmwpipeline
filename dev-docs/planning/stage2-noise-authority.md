# Stage 3 gap-pass matched-filter kernel reformulation

Handoff for the one remaining piece of the "Stage 2 is the single noise
authority / no user grid" work: reformulating the Stage 3 **gap-pass** matched
filter so it stops exp-apodizing the FID in the time domain and instead applies
a **shape-aware frequency-domain template convolution on the unapodized active
FT**. This fixes an active shape-mismatch bug on the Gaussian 2638 fixture,
reconciles a stale `detect_peaks` contract, and removes the last
independently-apodized FT/noise context.

**Status:** the noise-authority + no-user-grid migration **shipped** (see
"Context" below); this kernel reformulation is the only open item. It is
*independent of and unblocked by* the shipped work — "which grid we snap/score
on" was deliberately separated from "what template the matched filter uses".

## Context — what already shipped (do not redo)

Commits `634f77c → 0e58cb2` on `issue-27-companion-tuning`:

- **The active FT is the single grid.** Stage 2 measures + persists σ on the
  canonical unapodized active FT (trimmed to the analysis band); Stages 3
  (snap/score), 4 (window planning), and 5 (fit weighting + structural replan +
  fit overlay) all consume it. The front-zeroed full-record FT is a Stage 0/1
  start-time comparison view only — never a scoring/detection/planning/fit
  domain. There is no "user grid".
- **Shared builders** live in `_internal/active_ft_support.py`:
  `build_trimmed_active_ft(file_path, trim_range)` (the active FT as a
  `ComplexFT`), `build_active_grid_with_noise(...)` (that FT + its scatter σ on
  the same grid, using the persisted Stage 2 knobs), and
  `compute_canonical_active_ft` / `estimate_canonical_active_ft_noise`. The pure
  estimator wrapper is `preprocessing/noise_estimation.py::estimate_active_ft_noise`.
- **Stage 3 snap** is `_snap_to_active_grid` (replaced `_snap_to_user_grid`):
  detections snap onto the active FT, scored against the active-FT σ.
- **Retired:** `estimate_noise_adaptive` and the full-record canonical-noise
  loader `_load_canonical_noise` are deleted. `estimate_noise_scatter` is the
  sole estimator; a minimal level-based reference lives at
  `../research/noise-snr-scaling/legacy_adaptive.py`. The scatter estimator
  clamps its rank-filter windows to the data length. D9 in `ROADMAP.md` is
  amended.

**What did NOT change (the subject of this doc):** Stage 3's two *detection*
spectra are still built and noise-measured inline, unchanged:

| detection pass | spectrum (current) | inline noise |
|---|---|---|
| primary | Blackman-Harris-apodized full-record rfft (`_spectrum_from_fid`) | `estimate_noise_scatter` on it (`stage3_impl.py:660`) |
| gap | **exp-τ-apodized active-region rfft** (`_mf_gap_spectrum`, `_GAP_ACTIVE_ZPF=2`) | `estimate_noise_scatter` on it (`stage3_impl.py:663`) |

Detection finds peak *positions* on these spectra; the positions then snap onto
the active grid for scoring. So the gap pass's exp-τ apodization is purely a
detection-time choice — it never touches the reported σ/SNR — but it is the
wrong matched filter on a Gaussian instrument (below).

## The task — reformulate the gap pass

### How it works today (verified, current code)

`stage3_impl.py::_mf_gap_spectrum` (line ~190) exp-apodizes the
`[start_us, end_us]` active FID at `tau_basis_us`, zero-pads by `_GAP_ACTIVE_ZPF`
(=2), rffts, and the detector runs SavGol-derivative apex finding on it.
`exp(-t/τ)·FID → rfft` **is** the time-domain matched filter for an exponential
(Lorentzian) line — one operation (the only frequency-domain step is generic
SavGol; there is no second τ-width kernel). `tau_basis_us` is resolved
shape-aware (`stage3_impl.py:611-626`): Stage 2b `tau_G_maj` when
`recommended_shape == "gaussian"`, else Stage 2b `tau_maj`, else Stage 1
`expf_us`, else 5.0 µs.

### The three problems

1. **Shape-mismatch (the bug).** The *filter* is always a hard exponential
   (`_mf_gap_spectrum:227` `active *= exp(-t_rel/tau_basis_us)`), even when
   `recommended_shape` is Gaussian — only the *time constant* is shape-aware.
   On a Gaussian-decay instrument the exponential matched filter is mismatched
   (suboptimal SNR, slight position bias). 2638 is Gaussian, so the gap pass
   runs the wrong matched filter on our primary fixture today.
2. **Contract drift.** `preprocessing/peak_detection.py::detect_peaks` still
   documents the gap pass as operating on the **unapodized** full-resolution
   spectrum "to recover weak lines the apodization smeared away." The exp-τ
   apodization was layered into `stage3_impl` during the matched-filter work and
   the library contract was never reconciled.
3. **SavGol window sized to the pre-MF FWHM.** `gap_sg_window` is sized from
   `line_fwhm = 1/(π·tau_basis_us)` (`stage3_impl.py:647-649`); matched filtering
   at the same τ roughly *doubles* the linewidth, so the smoother is under-sized
   for the feature it actually sees.

### The fix

Implement the matched filter as a **shape-aware frequency-domain template
convolution on the unapodized active FT**, instead of time-domain
exp-apodization:

- By the convolution theorem, `FFT(x · w_τ) ≡ FFT(x) ⊛ W_τ`, where `W_τ` is the
  spectral kernel = FT of the time-domain window. For `w_τ = exp(-t/τ)` (t ≥ 0),
  `W_τ` is a Lorentzian of HWHM `1/τ`; for a Gaussian window
  `exp(-t²/2τ_G²)`, `W` is a Gaussian. So the time-domain apodized MF equals
  convolving the *unapodized* active FT with the line shape's spectral kernel.
- **Shape selection** comes from Stage 2b's `recommended_shape`
  (LORENTZIAN → Lorentzian kernel from `τ_maj`; GAUSSIAN → Gaussian kernel from
  `τ_G`), the same selector Stage 5 reads. This fixes problem 1.
- The unapodized active FT is now a shared artifact: build it with
  `build_trimmed_active_ft(file_path, trim_range)` and convolve in frequency,
  rather than re-extracting + apodizing the FID. This removes the gap pass's
  independent FID re-processing.
- Size SavGol to the **post-kernel** FWHM (≈ 2× the pre-MF line FWHM), fixing
  problem 3, and reconcile the `detect_peaks` docstring, fixing problem 2.

**Noise of the reformulated gap spectrum.** The MF spectrum is a transform of
the unapodized active FT, so its per-bin σ is either (a) the unapodized active-FT
σ propagated through the normalized kernel `k` (white noise → per-bin variance
≈ `σ_unapod²·Σ|k|²` where σ is locally flat), or (b) `estimate_active_ft_noise`
re-run on the convolved spectrum (robust to a non-flat σ(f); preferred, and
parallels how the gap inline noise is measured today). Confirm the scatter
estimator recovers the correct white-noise floor on a windowed/convolved active
FT (σ scales by `√(Σw²/N)`) — synthetic white-noise FID + a check on 2638/655
(this is the one empirical prerequisite that was never run).

**Equivalence makes this low-risk on Lorentzian fixtures** — the freq-domain
Lorentzian MF reproduces the current detector's statistic. The win is on
Gaussian (correct kernel) and architecture (one base FT, no separate exp-τ FID
re-processing).

### Validation matrix (gate before keeping it)

- **1512 / 655 (Lorentzian):** weak-line recovery rate unchanged vs the current
  exp-apodized gap pass (equivalence check).
- **2638 (Gaussian):** the Gaussian kernel recovers ≥ the same lines, with the
  position/SNR of known lines at least as good, **without** inflating false
  positives.
- **Strong-line sidelobe suppression and the continuous leakage-aware floor**
  still behave (they were tuned against the apodized spectrum's statistics —
  re-confirm on a strong-line window, e.g. 2638 w245-class).
- The integration suite's frozen Stage 3/4 peak references will shift; regenerate
  them deliberately and confirm the fixtures above before accepting.

### Deferred sibling question (after the gap pass settles)

The **primary** pass still builds a Blackman-Harris-apodized full-record rfft
(`_spectrum_from_fid`) purely for leakage suppression to surface skirt
candidates. The same "matched filter in frequency space" thinking *might* let
the primary also live on the unapodized active FT (kernel = leakage-suppression
filter), collapsing that context too. Hold this until the gap pass is settled.

## Call sites (current line numbers)

- `src/ftmwpipeline/_internal/stage3_impl.py:190` — `_mf_gap_spectrum`, the
  exp-apodize-then-rfft gap-FT builder → reformulate to a freq-domain kernel
  convolution on `build_trimmed_active_ft(...)`.
- `src/ftmwpipeline/_internal/stage3_impl.py:85` — `_GAP_ACTIVE_ZPF` (gap-FT
  zero-padding) and `:611-626` (shape-aware `tau_basis_us` resolution; reuse for
  kernel shape + width).
- `src/ftmwpipeline/_internal/stage3_impl.py:647-649` — `line_fwhm_mhz` /
  `_grid_aware_sg_window` → size to the post-kernel FWHM.
- `src/ftmwpipeline/_internal/stage3_impl.py:663` — `gap_noise` (inline scatter
  on the gap FT) → measure on the convolved spectrum (or propagate σ_unapod).
- `src/ftmwpipeline/preprocessing/peak_detection.py::detect_peaks` — update the
  gap-pass contract docstring; size SavGol to the post-kernel FWHM.
- `src/ftmwpipeline/_internal/active_ft_support.py` — `build_trimmed_active_ft`
  is the unapodized active-FT source to convolve.

## Out of scope / unchanged

- **Stage 2b keeps its own FID-tail σ reference** (not the Stage 2 estimator) for
  τ extraction — see `stage2b-tau-calibration.md`. "Stage 2 is the noise
  authority" applies to the frequency-domain σ(f) consumers (Stages 3/4/5), not
  the time-domain τ gate.
- The **scatter estimator internals** are unchanged; this work is about the gap
  pass's detection spectrum, not the estimator.
- The **active-grid snap/score/plan/fit migration** is shipped (see Context) —
  do not revisit it.
