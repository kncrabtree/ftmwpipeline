# Stage 2 as the single noise authority

Planning doc for making Stage 2's σ(f) the one noise authority every later stage
consumes, reformulating the Stage 3 gap-pass matched filter so it stops spawning
a distinct apodized FT, and deleting the legacy `estimate_noise_adaptive` kernel
once nothing calls it.

**Status:** planning only — no code written yet. This doc is written to be
self-contained for a fresh session: §"Verified facts" and §"Verify first" are
the empirical state; §"Work sequence" is the order of operations; §"Call-site
map" is where the edits land.

## Principle

Stage 2 exists to produce one authoritative noise level as a function of
frequency. If a later stage re-measures noise locally, Stage 2 is not doing its
job and the two numbers can disagree. The local re-estimation is a residue of
the abandoned conceit that a user might pick arbitrary window/zpf/apodization per
stage. With the canonical FT fixed (raw, unapodized — D7 in
`processing-settings-persistence.md`), that justification is gone.

## The key physics (settles the design)

Noise on an FT is **invariant to zero-padding** and **varies only with (active
window length, apodization)**:

- **zpf is irrelevant to the per-bin noise level.** Zero-padding interpolates
  and correlates bins (the Dirichlet kernel), but each rfft bin is a sum over
  the *same* `N_active` real samples, so the per-bin variance is `N_active·σ_t²`
  regardless of the padded length. (Adjacent-bin *correlation* changes; the σ
  you divide by for SNR does not.) So consuming the same σ across zpf levels is
  correct.
- **Active length matters.** Per-bin variance scales with `N_active`, so a
  different `[start_us, end_us]` window changes σ by `√N_active`.
- **Apodization matters.** A window `w[n]` scales the per-bin noise variance by
  `Σw[n]²`; an apodized spectrum has a genuinely different noise floor.

**Consequence — the design.** Do all noise work in **active-FT space at the
canonical active length**, measured once, in Stage 2's domain; stages consume.
This dissolves the earlier "derive an analytic full-FT→active-FT transfer `g`"
idea (which an earlier draft of this doc proposed): there is nothing to transfer
if everything is *measured* in the same active-FT space — and *measuring* (not
deriving) is exactly what D9 wanted (noise measured on the spectrum you fit so
FFT-normalization cancels). We keep "measure where you fit"; we just measure
**once** and share it, instead of each stage re-measuring.

**Settled principle (active-FT truncation is the canonical domain).** Once
`start_us` is fixed, *all real processing truncates to the active FT* — the
slice `[start_us:end_us]` of length `n_active`, `dt_us·rfft`, unpadded
(`compute_active_ft`'s domain). The full-length **front-zeroed** record that
`FID.preprocess` produces (it keeps `N_total`, zeros `[:start_idx]`/`[end_idx:]`,
then pads) and `compute_fft` normalizes by `/N_total` is a **Stage 0/1 display
affordance only**: it holds the frequency grid fixed so a user can compare FTs
at different `start_us`. It is *not* the substrate for noise, detection, or
fitting. So Stage 2 measures its authoritative σ on the active FT, and the
persisted Stage 1 `complex_ft` stays full-length purely for the comparison view.
This is what makes fact A (below) a non-issue: there is no full-record→active
rescale because Stage 2 never measures on the full record.

**Corollary — there is no "user grid" (decided).** The persisted full-record
spectrum is *not* a supported reporting/snap/scoring domain. Stage 3's current
`_snap_to_user_grid` (it snaps detections to `user_ft.freq_array`, re-measures
intensity on `user_ft.magnitude_spectrum`, and scores against the full-record
Stage 2 σ) is **replaced** by an active-grid snap, not preserved. The single
grid for σ, snap-back, and reported peak frequency/SNR is the active FT.
Sequencing note: this couples the persisted-σ re-home and the Stage 3 snap
migration into **one** landing (changing the persisted σ grid breaks the
full-record snap, so both move together). To keep the fine detected positions
through the coarse `1/T_active` active grid, Stage 3 snaps onto a **zero-padded**
active FT (fine grid, active-region content, `dt_us·rfft`) and reads σ from the
unpadded active-FT authority interpolated onto it (σ is per-bin zpf-invariant).
The primary/gap detection-*spectrum construction* is unchanged here — that, and
the gap-pass freq-domain template, are the deferred **kernel** work; "which grid
we snap/score on" is separable from "what template the matched filter uses".
Frozen Stage 3 peak references regenerate as part of this landing's revalidation.

## Noise contexts in play (the thing to collapse)

Today there are three distinct active-FT noise contexts, because two detectors
apodize:

| context | window | current consumer | current code |
|---|---|---|---|
| unapodized active FT | boxcar | Stage 5 fit weighting | re-measured by `estimate_noise_adaptive` |
| BH-apodized active FT | Blackman-Harris | Stage 3 **primary** detection | re-measured by `estimate_noise_scatter` |
| exp-τ apodized active FT | `exp(-t/τ_basis)` | Stage 3 **gap** detection (matched filter) | re-measured by `estimate_noise_scatter` |

Goal end-state: Stage 2 (its module) emits the **unapodized active-FT σ** as the
base; the apodized contexts become deterministic transforms of it (or the same
estimator re-applied to the windowed spectrum), and the gap-pass reformulation
(below) removes the exp-τ FT as an *independent* entity entirely.

## Work item 1 — reformulate the Stage 3 gap-pass matched filter

This is first because it has an **active correctness bug** and because settling
it reduces the contexts the noise authority must serve.

**Current behavior (verified).** The gap pass exp-apodizes the active-region FID
at `τ_basis` (= Stage 2b `τ_maj`), zero-pads (`_GAP_ACTIVE_ZPF=2`), rffts, then
runs SavGol-derivative apex finding. `exp(-t/τ)·FID → rfft` **is** the
time-domain matched filter for an exponential (Lorentzian) line — one operation,
not a double filter (the only frequency-domain step is generic SavGol; there is
no second τ-width kernel). It is mathematically sound *for Lorentzian lines*.

**The three problems.**
1. **Shape-mismatch (the bug).** The kernel is hard-exponential. On a
   Gaussian-decay instrument the matched filter is *mismatched* — suboptimal SNR
   and a slight position bias. 2638 is Gaussian (Stage 2b `recommended_shape`),
   so the gap pass runs the wrong matched filter on our primary fixture today.
2. **Contract drift.** `preprocessing/peak_detection.py::detect_peaks` still
   documents the gap pass as the **unapodized** full-resolution spectrum "to
   recover weak lines the apodization smeared away." The exp-τ apodization was
   layered into `stage3_impl` during the matched-filter-detection work and the
   library contract was never reconciled.
3. **SavGol window sized to the pre-MF FWHM.** `gap_sg_window` is sized from
   `line_fwhm = 1/(π·τ_basis)`; matched filtering at the same τ roughly *doubles*
   the linewidth, so the smoother is under-sized for the feature it sees.

**The fix.** Implement the matched filter as a **shape-aware frequency-domain
template convolution on the unapodized active FT**, instead of time-domain
exp-apodization:

- By the convolution theorem, `FFT(x · w_τ) ≡ FFT(x) ⊛ W_τ`, where `W_τ` is the
  spectral kernel = FT of the window. For `w_τ = exp(-t/τ)` (t ≥ 0), `W_τ` is a
  Lorentzian of HWHM `1/τ`; for a Gaussian window `exp(-t²/2τ_G²)`, `W` is a
  Gaussian. So the time-domain apodized MF equals convolving the *unapodized*
  active FT with the line shape's spectral kernel.
- **Shape selection** comes from Stage 2b's `recommended_shape`
  (LORENTZIAN → Lorentzian kernel from `τ_maj`; GAUSSIAN → Gaussian kernel from
  `τ_G`), the same selector Stage 5 already reads. This fixes problem 1.
- Because the MF spectrum is now a transform of the canonical unapodized active
  FT, its noise derives from the unapodized active-FT σ (white noise through a
  normalized kernel `k` has per-bin variance ≈ `σ_unapod² · Σ|k|²` where σ is
  locally flat) — or scatter-measured on the convolved spectrum (robust to a
  non-flat σ(f); preferred). Either way the exp-τ FT stops being an independent
  FID re-processing with its own noise. This fixes problems 2 and 3 (size SavGol
  to the post-kernel FWHM).

**Equivalence makes this low-risk on Lorentzian fixtures** (the freq-domain
Lorentzian MF reproduces the current detector's statistic); the win is on
Gaussian (correct kernel) and architecture (one base FT).

**Validation matrix (gate before keeping it):**
- 1512 / 655 (Lorentzian): weak-line recovery rate unchanged vs the current
  exp-apodized gap pass (equivalence check).
- 2638 (Gaussian): Gaussian kernel recovers ≥ the same lines, with the
  position/SNR of known lines at least as good, **without** inflating false
  positives.
- Strong-line **sidelobe suppression** and the continuous leakage-aware floor
  still behave (they were tuned against the apodized spectrum's statistics —
  re-confirm on a strong-line window, e.g. 2638 w245-class).

**Deferred sibling question.** The **primary** pass uses Blackman-Harris
apodization purely for leakage suppression to surface skirt candidates. The same
"matched filter in frequency space" thinking *might* let the primary also live on
the unapodized active FT (kernel = leakage-suppression filter), collapsing
context 2 as well. Hold this until the gap pass is settled — the gap pass is the
one with the active bug.

## Work item 2 — the noise authority + Stage 5/3 consumption

Once the gap pass is reformulated, there is one base noise to produce.

1. **Noise authority surface.** One function in the Stage 2 module,
   `estimate_active_ft_noise(active_region, window=...)` (name TBD), returns the
   per-bin scatter σ for a given apodization. Stage 2 calls it with the boxcar
   window and **persists the unapodized active-FT σ** as the canonical array.
   This re-homes Stage 2's *measurement input*: today `estimate_noise_impl`
   (`stage2_impl.py:237-239`) runs the scatter estimator on the persisted
   full-record `complex_ft`; instead it builds the unapodized active FT on
   demand from `stage0_fid_data` + the canonical `start_us/end_us`
   (`compute_active_ft`, boxcar) and measures there. Per fact A the persisted
   `complex_ft` is the front-zeroed full record, so this is a domain change, not
   a re-sort. Knock-on sites that must move onto the active-FT grid with it:
   `_load_canonical_noise` (currently reconstructs σ on `user_ft.freq_array`,
   the full-record grid) and the `visualize-noise` overlay.
2. **Stage 5 consumes it.** Replace the `estimate_noise_adaptive` call in
   `stage5_impl` with the persisted unapodized active-FT σ, regridded onto the
   active-FT bin order (the code already re-sorts/unsorts). With Stage 2 now
   measuring in active-FT space at the same `start_us/end_us`, the persisted
   grid and the fit's active-FT grid coincide — no transfer, no scaling, at most
   a regrid for bin-order alignment.
3. **Stage 3 consumes it.** Primary/gap detection pull from the authority
   instead of the two inline `estimate_noise_scatter` calls; the display fallback
   drops its `estimate_noise_adaptive` call (Stage 3 already depends on Stage 2 —
   make a missing-Stage-2 a loud error, or draw the diagnostic without a noise
   overlay). Under the settled principle, Stage 3 detection is itself on the
   active FT (its internal zpf does the position-finding interpolation, and σ is
   per-bin invariant to zpf, so the active-FT σ is consumed across the padded
   detection grid by interpolation). The snap-back reclassification, currently
   against the full-record σ via `_load_canonical_noise`, moves onto the
   active-FT grid with everything else.
4. **Delete `estimate_noise_adaptive`** and its private helpers (`_mad`,
   `_compute_mad_based_bins`, `_build_noise_mask`, `_exclude_strong_line_skirts`,
   `_filter_by_skewness_cached`, `_bin_stats_from`, …), the module-level adaptive
   constants, and the `preprocessing/__init__` export, once nothing imports it.

**Dependency wrinkle.** `τ_basis`/`τ_G` are **Stage 2b** outputs (2b runs after
2), so the apodized/MF noise cannot be produced eagerly at Stage 2 time. The
authority function lives in the Stage 2 module but is invoked for the
shape-aware kernel once τ is known (at/after Stage 2b, or lazily by Stage 3,
derived from the persisted unapodized σ + the kernel). Stage 2 only persists the
unapodized array eagerly.

## Verified facts (this session — code as it stands)

- `fitting/active_ft.py::compute_active_ft` extracts `[start_us, end_us]`,
  applies `exp(-(t-t0)/expf_us)` (t0 = start_us), removes DC (`rdc`), and rffts
  **just the active region**; `n_padded = n_active·2^zpf_active`. Amplitude
  convention `dt_us·rfft(active·apod)`; canonical `zpf_active=0` → unpadded,
  `α=1`, independent bins.
- `_internal/stage5_impl.py` (~534–564): builds `active_ft` via
  `compute_active_ft(..., expf_us=expf_us)` where `expf_us` comes from
  `base_pp.expf_us` (canonical None → unapodized), then
  `active_noise = estimate_noise_adaptive(sorted_freq, sorted_mag)` on the
  active-FT **magnitude**; result unsorted back to `active_rms`. It *also* loads
  the canonical Stage 2 σ via `_load_canonical_noise(file_path, user_ft)`
  (~203, ~1103) but does not use it for the fit.
- `_internal/stage3_impl.py`: detection builds `primary_ft` (Blackman-Harris
  apodized) and `gap_ft` (exp-τ matched filter, `_GAP_ACTIVE_ZPF=2`); computes
  `primary_noise`/`gap_noise = estimate_noise_scatter(...)` on each (~668–673)
  behind continuous leakage-aware floors; `detect_peaks` scores on these; the
  snap-back reclassifies against the canonical Stage 2 σ (`user_rms` via
  `_load_canonical_noise`, ~580). The `estimate_noise_adaptive` call (~841–850)
  is a **display-only fallback** when Stage 2 is absent.
- `preprocessing/peak_detection.py::detect_peaks`: primary scored on the
  apodized spectrum, gap on the (contract: unapodized) spectrum; the only
  frequency-domain operation is SavGol d1/d2 apex finding — **no τ kernel**, so
  the current pipeline is not literally double-matched-filtered.

## Verify first (these size the job — read-mostly + one tiny harness)

- **(A) Slice vs front-zero. — VERIFIED: front-zero.** `FID.preprocess`
  (`core/data_structures.py:328-368`) keeps the full `N_total` array, zeros
  `[:start_idx]` and `[end_idx:]`, then zero-pads to `n_padded`; `compute_fft`
  (`:146-159`) rffts that full-length record and normalizes by
  `/= original_length` (= `N_total`). So the persisted Stage 1 FT (and the
  Stage 2 σ measured on it) lives on the **full-record** grid (fine `1/T_padded`
  bins) in a `/N_total` amplitude convention — a *different* domain from the
  active FT (`dt_us·rfft` of the `n_active` slice, coarse `1/T_active` bins).
  **This does not gate Work item 2 by a rescale: per the settled principle
  above, Stage 2's authoritative σ is re-homed to active-FT space, so the
  full-record→active transfer never arises.** The front-zeroed full-length FT
  remains only for the Stage 0/1 start-time comparison view.
- **(B) Scatter-on-windowed.** Confirm `estimate_noise_scatter` on a
  BH/exp-apodized active FT recovers the correct white-noise floor
  (σ scales by `√(Σw²/N)`). Synthetic white-noise FID + a check on 2638/655.
  Justifies treating the apodized contexts as the same estimator on the windowed
  spectrum.
- **(C) Canonical apodization off. — VERIFIED by code.**
  `_build_active_ft_inputs` (`stage5_impl.py:183`) takes
  `expf_us = base_pp.expf_us` from the persisted Stage 1 processing params,
  which is `None` canonically (the unapodized FT), so the Stage 5 active-FT
  context is genuinely unapodized and equals Stage 2's re-homed domain. (A live
  run-through is still worth a one-liner before the deletes land, but the code
  path is unambiguous.)

## Work sequence

0. Verify facts A/B/C (`scratch/` harness; read-mostly).
1. **Gap-pass MF reformulation** (Work item 1) + its validation matrix.
2. Stand up the noise authority; Stage 2 persists the unapodized active-FT σ
   (or confirm the existing persisted Stage 2 σ already *is* it, per fact A).
3. Stage 5 consumes it; remove its `estimate_noise_adaptive` call.
4. Stage 3 primary/gap consume from the authority; drop the display-fallback
   adaptive call.
5. Delete `estimate_noise_adaptive` + helpers + constants + export.
6. End-to-end revalidation (below); amend the D9 entry in `ROADMAP.md`.

## Call-site map (file:line as of this session)

- `src/ftmwpipeline/_internal/stage5_impl.py:560` — `estimate_noise_adaptive`
  on the active-FT magnitude → replace with consumed σ.
- `src/ftmwpipeline/_internal/stage5_impl.py:203, 1103` — `_load_canonical_noise`
  already loads Stage 2 σ (the array to consume).
- `src/ftmwpipeline/_internal/stage3_impl.py:668-673` — `primary_noise` /
  `gap_noise` via `estimate_noise_scatter` → consume from the authority.
- `src/ftmwpipeline/_internal/stage3_impl.py` (~195-225, `_GAP_ACTIVE_ZPF`,
  the exp-apodize gap-FT builder) → reformulate to a freq-domain kernel.
- `src/ftmwpipeline/_internal/stage3_impl.py` (~841-850) — display-fallback
  `estimate_noise_adaptive` → drop.
- `src/ftmwpipeline/preprocessing/peak_detection.py` — `detect_peaks`: update the
  gap-pass contract docstring; size SavGol to the post-kernel FWHM.
- `src/ftmwpipeline/fitting/active_ft.py` — `compute_active_ft`, the active-FT
  source for the authority.
- `src/ftmwpipeline/preprocessing/noise_estimation.py` —
  `estimate_noise_adaptive` + helpers + constants to delete; keep
  `estimate_noise_scatter`.

## Revalidation (gate before the deletes land)

This moves tuned numbers, so it gates on fixtures, not just unit tests:
- **Stage 5 χ²ᵣ + uncertainties** on 2638 (gaussian), 655 (high-SNR lorentzian),
  1512 (vinyl cyanide ground-truth freq + uncertainty). Per-line frequency/
  uncertainty accuracy on the unambiguous 1512 lines is the primary bar; χ²ᵣ
  distributions should not regress materially.
- **Stage 3 peak detection** parity on the same fixtures (the integration
  suite's frozen references already run on the scatter Stage 2 σ) plus the gap
  reformulation's own matrix above.
- A/B the consumed-σ vs the currently-re-measured σ per window on 2638/655;
  investigate any window where they diverge before trusting the consumption.

## Relationship to D9

D9 resolved the Stage 5 spectral domain by fitting on the **active FT** with
noise *re-measured* there. This work keeps the active-FT fit domain (D9's core
decision stands) and keeps *measuring* (not deriving), but measures **once** in
Stage 2's domain and shares it. If the gap reformulation also lands, the
exp-apodized gap FT is replaced by a freq-domain transform of the unapodized
active FT. When this ships, amend the D9 entry in `ROADMAP.md` to note the noise
reference moved from per-stage re-measurement to a shared Stage 2 active-FT σ.

## Out of scope / unchanged

- **Stage 2b keeps its own FID-tail σ reference** (not the Stage 2 estimator)
  for τ extraction — see `stage2b-tau-calibration.md` and its noise-reference
  close-out. "Stage 2 is the noise authority" applies to the frequency-domain
  σ(f) consumers (Stages 3/4/5), not the time-domain τ gate.
- The **scatter estimator itself** is unchanged; this work is about *who runs it
  and on which spectrum*, not its internals.
