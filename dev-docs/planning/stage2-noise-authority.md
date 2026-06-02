# Stage 2 as the single noise authority

Planning doc for retiring the last in-pipeline re-estimation of noise so that
every stage downstream of Stage 2 consumes the Stage 2 per-bin σ(f) directly,
and the legacy `estimate_noise_adaptive` kernel can be deleted.

## Principle

Stage 2 exists to produce one authoritative noise level as a function of
frequency — the per-bin complex-RMS σ_x array on the canonical FT. If a later
stage re-measures noise locally, Stage 2 is not doing its job and the two
numbers can disagree. The current local re-estimation is a residue of the
abandoned design conceit that a user might pick arbitrary window functions /
zero-padding / apodization per stage, which made a single propagated σ awkward.
With the canonical FT now fixed (raw, unapodized — see the D7 settings in
`processing-settings-persistence.md`), that justification is gone.

## Current state (what re-estimates, and why)

`estimate_noise_adaptive` is no longer a user-facing Stage 2 method (the scatter
estimator is the sole method; see `stage2-noise-estimation.md`). Its kernel
survives only because two sites still call it:

1. **Stage 5 — `_internal/stage5_impl.py`** (the substantive one). Per fit it
   calls `estimate_noise_adaptive` on the **active-FT magnitude spectrum** to
   get `active_rms`, the per-bin noise the window fits weight against. This is
   the D9 resolution: the noise is measured on the very spectrum the fit sees so
   "any FFT normalization choices cancel by construction." Stage 5 already
   *loads* the canonical Stage 2 σ(f) (`_load_canonical_noise`) but does not use
   it for the fit.
2. **Stage 3 — `_internal/stage3_impl.py`** (display fallback only). The primary
   Stage 3 path already consumes Stage 2 via `_load_canonical_noise`; the
   adaptive call is reached only when canonical Stage 2 noise is missing, to
   draw a peak-detection diagnostic. Low-stakes.

## The hard part: full-FT → active-FT noise transfer

The active FT used in Stage 5 is a re-FFT of the FID's active window
(`start_us`→`end_us`), so it has a different length and frequency grid than the
canonical full FT, and Stage 5 may apply an exponential apodization (`expf_us`)
to it. To consume the Stage 2 σ(f) there, we need the analytic map

    σ_activeFT(f) = g · σ_stage2(f_nearest)

where `g` accounts for:
- **FFT length / normalization** difference between the persisted full FT and
  the on-demand active FT (`N_active`, `N_padded`, the rfft normalization);
- **time-window length** — white-noise variance per bin scales with the number
  of summed samples, so a shorter active window changes the per-bin σ;
- **apodization**, if any — an `expf_us` taper colors the noise; the per-bin
  variance is scaled by the taper window's energy (∑w²), analytically tractable
  but it must be carried, not re-measured;
- **grid interpolation** — mapping σ_stage2 (full-FT grid) onto the active-FT
  bin centers.

The D9 workaround sidesteps all of this by re-measuring on the active FT. The
deliverable here is to derive `g` in closed form (it is a deterministic function
of the FFT parameters, not data-dependent), validate it against the
re-measured σ on several fixtures, then replace the call.

## Plan

1. **Derive + unit-test the noise transfer.** Implement `g` as a pure function
   of `(N_active, N_padded, sample_dt, expf_us, normalization)`. Unit test:
   on synthetic white-noise FIDs, the propagated σ matches a Monte-Carlo
   active-FT σ to within sampling error, with and without apodization.
2. **Stage 5: consume Stage 2 σ.** Replace the `estimate_noise_adaptive` call
   in `stage5_impl` with: load canonical σ_x (already done), interpolate onto
   the active-FT grid, apply `g`. Keep the result element-aligned with
   `active_ft.complex_spectrum` (the current code already re-sorts/unsorts).
3. **Stage 3: drop the fallback.** Make the missing-Stage-2 branch a loud error
   (Stage 3 already depends on Stage 2) instead of re-estimating, or draw the
   diagnostic without a noise overlay.
4. **Delete `estimate_noise_adaptive`** and its private helpers
   (`_mad`, `_compute_mad_based_bins`, `_build_noise_mask`,
   `_exclude_strong_line_skirts`, `_filter_by_skewness_cached`, …) and the
   module-level adaptive constants once nothing imports them. Drop the
   `preprocessing/__init__` export.
5. **Revalidate** (below) before the delete lands.

## Revalidation

This moves tuned numbers, so it gates on fixture validation, not just unit
tests:
- **Stage 5 χ²ᵣ + uncertainties** on 2638 (gaussian), 655 (high-SNR
  lorentzian), 1512 (vinyl cyanide ground-truth freq + uncertainty). The
  per-line frequency/uncertainty accuracy on the unambiguous 1512 lines is the
  primary acceptance bar; χ²ᵣ distributions should not regress materially.
- **Stage 3 peak detection** parity on the same fixtures (the integration
  suite's frozen references — these already run on the scatter Stage 2 σ).
- A/B the propagated-σ vs re-measured-σ per-window noise on 2638/655 directly
  (they should agree to the transfer's modeling error); investigate any window
  where they diverge before trusting the analytic `g`.

## Relationship to D9

D9 was resolved by fitting on the active FT with noise *re-measured* there. This
work keeps the active-FT fit domain (D9's core decision stands) but replaces the
re-measurement with a propagated Stage 2 σ. If it lands, the D9 divergence entry
in `ROADMAP.md` should be amended to note the noise reference changed from
re-measured to propagated.

## Open questions

- Does Stage 5 still apply `expf_us` to the active FT in the canonical
  (unapodized) path? If apodization is effectively always off now, `g` loses its
  hardest term and this simplifies considerably — confirm against the resolution
  chain before deriving the general form.
- Stage 2b deliberately keeps its **own** FID-tail σ reference (not the Stage 2
  estimator) for τ extraction — see `stage2b-tau-calibration.md` and the
  noise-reference close-out. That decision is independent of this work and
  stays; "Stage 2 is the noise authority" applies to the frequency-domain σ(f)
  consumers (Stages 3/4/5), not the time-domain τ gate.
