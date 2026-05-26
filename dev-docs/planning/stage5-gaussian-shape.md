# Stage 5 Gaussian-shape extension

Add a Gaussian time-domain envelope as an alternative line shape for Stage 5
fits, selectable per `fit_peaks(...)` call. The existing pure-exponential
(Lorentzian-frequency-domain) path is preserved unchanged as the default;
the Gaussian path is wired alongside it through the same window-fit
machinery via a shape-selector abstraction.

## Motivation

The Voigt-deficit research on 2638 (see
[`stage5-voigt-deficit.md`](stage5-voigt-deficit.md)) showed that the
per-window joint `(τ_L, τ_G)` LSQ has `τ_L` saturating at the upper bound
on the windows where Voigt beats pure-exp, while `τ_G` lands in a tight
6-9 µs band. The per-bin STFT analysis (Part B) confirmed the same on
408 strong contributor bins: median Voigt χ²ᵣ = 0.80 vs pure-exp 3.77,
with `τ_L → ∞` and the Gaussian factor carrying the entire envelope. On
this fixture (and presumably on other free-jet supersonic-beam setups
with similar geometry), the molecular line shape is empirically
**Gaussian-dominant**, not Lorentzian.

The pure-exp model approximates a Gaussian envelope by shortening `τ_L`,
producing `τ_eff ≈ 5 µs` that absorbs the Gaussian decay. This is a
**lossy** representation: the recovered `τ_eff` doesn't track the
physical mechanism, the per-window fits leave residual χ²ᵣ > 1 on
otherwise clean lines, and the per-band τ trend (1/f from horn coupling)
is only partly captured.

The pragmatic fix is a switch from the pure-exp envelope to a pure-
Gaussian envelope:

    s_i(t) = A_i · exp(-(t/τ_G)²) · cos(2π f_i t + φ_i)

One shared `τ_G` per window in place of `τ`; same parameter count, same
fit machinery. The frequency-domain analog is a finite-window Gaussian
(no closed form on `[0, T]`, but tractable via complex erf / Faddeeva).

The full Voigt `exp(-t/τ_L)·exp(-(t/τ_G)²)` from
[`stage5-voigt-deficit.md`](stage5-voigt-deficit.md) is the natural
unification, but its `(τ_L, τ_G)` identifiability concern (poor LSQ
conditioning when one dominates) and double-parameter cost don't
justify it for instruments where one component clearly dominates. The
shape-selector framework defined here is designed to extend naturally
to `voigt` as a third option later.

## Implementation surface

### Core model layer (`peak_model.py`)

Add a `PeakShape` enum (`LORENTZIAN`, `GAUSSIAN`) and a closed-form
Gaussian `h_T`:

    h_T_gaussian(Δf; τ_G, T) = ∫_0^T exp(-(t/τ_G)² - i 2π Δf t) dt

Closed form via the complex error function:

    = (τ_G √π / 2) · exp(β²) · [erf(T/τ_G + β) - erf(β)]
      with β = i π τ_G Δf

`scipy.special.erf` does not accept complex arguments, so the
implementation uses the Faddeeva function `wofz`:

    erf(z) = 1 - exp(-z²) · wofz(i·z)

Analytic Jacobian for `dh/d(Δf)` and `dh/d(τ_G)` is derivable in closed
form; the existing Lorentzian `h_T_jacobian` is the template.

A shape-dispatch helper:

    def h_T(shape, df_mhz, tau_us, T_us)
    def h_T_jacobian(shape, df_mhz, tau_us, T_us)

routes to the Lorentzian or Gaussian implementation. Existing call sites
(`model_spectrum`, `model_jacobian`) gain a `shape` parameter, defaulting
to `LORENTZIAN` for backward compatibility.

### Fit machinery (`window_fit.py`, `plan_execution.py`)

`fit_window`, `conservative_fit`, `residual_rescue`, and the per-window
LSQ residual / Jacobian functions all gain a `shape` parameter that is
threaded through to the model evaluation. The parameter packing
(`_pack`, `_unpack`) and bounds are shape-independent (both shapes have
1 shared decay constant per window); the only change inside the LSQ is
which `h_T`/`h_T_jacobian` is called.

The bidirectional Gaussian-prior τ penalty (currently anchored on
`τ_maj` from Stage 2b) generalizes trivially: for `shape=GAUSSIAN`,
anchor on `τ_G_maj` from the new Stage 2b twin (see below).

The per-window AICc, knockout, and residual-rescue logic are
shape-agnostic at the χ² level — they just need the shape-appropriate
model for evaluation.

### Stage 2b twin (`stage2bg_impl.py`, `fitting/tau_calibration.py`)

A companion Stage 2b path that calibrates `τ_G_maj` + per-band
`band_majorities_G` instead of pure-exp τ. Algorithm: lift Part B's
per-bin Voigt fit (`_fit_voigt_multistart` on contributor bins only,
filter to converged + finite `τ_G < cap` + `Δχ²ᵣ ≥ 1`) into
`fitting/tau_calibration.py` as `extract_tau_G_majority`, mirroring
`extract_tau_majority` in shape and signature.

New stage name: `stage2b_tau_G_calibration`, with dependencies
`[stage2_noise_result]` (same as `stage2b_tau_calibration`). The two
calibrations are independent — a user running `fit_peaks(shape=...)`
needs the matching calibration completed first.

HDF5 path: `/stage2b_tau_G_calibration`, schema mirrors
`/stage2b_tau_calibration` (scalars, contributors, band_majorities,
algorithm_info) with `τ_G` replacing `τ` in the field names.

### Stage 5 wiring

`fit_peaks(file_path, shape='lorentzian'|'gaussian', ...)`. When
`shape='gaussian'`:

- Verifies `stage2b_tau_G_calibration` is present (or runs no-prior
  fits if absent, with a warning).
- Reads `(τ_G_maj, σ_τ_G)` and `band_majorities_G` from the persisted
  calibration.
- Sets up the Stage 5 LSQ with `shape=GAUSSIAN` propagated through.
- Persists `/stage5_fitting` with a new `shape` attribute on the root
  group (default `'lorentzian'` for back-compat with existing files).
- Per-window: `tau_us` field stores `τ_G` when `shape='gaussian'`
  (or rename the field to a shape-neutral `tau_shape_us`? — TBD).

Cross-interface parity: CLI subcommand gains `--shape`, `Pipeline.fit_peaks`
gains `shape=...`, `api.fit_peaks` gains `shape=...`. All three
delegate identically.

### Persistence and back-compat

- The default for `fit_peaks(shape=...)` is `'lorentzian'` — existing
  scripts and tests keep behaving exactly as before.
- `/stage5_fitting` files created without a `shape` attribute are
  assumed `'lorentzian'`.
- The new Stage 2b twin uses a distinct HDF5 path, so it doesn't
  conflict with the existing pure-exp calibration.

### Tests

- Unit: `h_T_gaussian` value at `Δf=0` (= `τ_G √π/2 · erf(T/τ_G)`),
  Jacobian vs finite-difference (relative error < 1e-8), parity with
  `h_T_lorentzian` in the appropriate limit.
- Integration: tiny synthetic with known Gaussian envelope, fit
  recovers `τ_G` within noise.
- Cross-interface: CLI / Pipeline / api Gaussian fit produces
  identical persisted result.
- Existing Lorentzian tests run unchanged — shape selector must not
  break the default path.

## Validation

Research script in
[`research/gaussian-shape/`](../research/gaussian-shape/) runs both
shapes on the same fixture (initially 2638; later a user-supplied
clean-lines fixture with internal-rotation-doublet content for
generalization). Per-window comparison output: `χ²ᵣ`, AICc, peak
frequencies, peak amplitudes, peak count. Aggregated as a JSON +
2-panel figure (per-window χ² + per-peak frequency/amplitude scatter)
per fixture.

Acceptance for the Gaussian path on 2638:
- Median Stage 5 χ²ᵣ on `fit_peaks(shape='gaussian')` < median for
  `shape='lorentzian'` (the per-bin Voigt result predicts ~5× drop).
- 95th percentile χ²ᵣ < 5 (vs the current ~50+ tail).
- Recovered peak frequencies agree with the Lorentzian fit to within
  the per-peak frequency error bar (no shape-induced line-center bias).
- Recovered peak amplitudes within 20 % of the Lorentzian fit on
  isolated lines (blends may legitimately differ).
- AICc on shape-error windows prefers Gaussian by Δ > 5 on the
  windows Part A identified (w141, w213, w310, w355).

Cross-fixture: the second fixture validates that the Gaussian model
generalizes beyond 2638's specific shape. If the second fixture's lines
fit Lorentzian better, that's evidence the shape is instrument-
specific (per-fixture choice) rather than universal.

## Implementation progress

Landed this session (`commits 699db4c`, `2c01b30`):

- **Planning + ROADMAP** entry registered.
- **`PeakShape` enum** (`LORENTZIAN`, `GAUSSIAN`) with `coerce(...)` for
  string/enum normalisation at API boundaries.
- **`h_T_gaussian`** closed form via Faddeeva-stabilised expression
  (avoids `exp(β²)` overflow at large `|Δf|·τ_G`); verified to machine
  precision against `scipy.integrate.quad`.
- **`h_T_gaussian_jacobian`** analytic `(d/dΔf, d/dτ_G)` in the same
  stable form; matches central finite difference to relative error
  `< 1e-6`.
- **`effective_tau_gaussian`** centre value.
- **Shape dispatchers**: `h_T_shape`, `h_T_shape_jacobian`,
  `effective_tau_shape`.
- **Shape threaded through Stage 5 fitter**: `model_spectrum`,
  `model_jacobian`, `feature_fwhm`, `derive_window_fit_constraints`,
  `fit_window`, `conservative_fit`, `knockout_test`, `_seed_peak`,
  `_blend_aware_seed`, the rescue chain
  (`merge_close_peaks_cleanup`, `remove_and_refit_cleanup`,
  `iterative_aicc_cleanup`, `attempt_residual_rescue`,
  `rescue_and_consolidate`), and `plan_execution`
  (`subtract_frozen_background`, `fit_window_with_fixed_contributors`,
  `local_thaw_cofit`, `_frozen_subset_model`, `_perform_thaw`,
  `attempt_thaw_round`, `_install_cofit_outcome`,
  `_apply_rescue_to_outcome`, `execute_plan`) all accept a `shape`
  parameter defaulting to `LORENTZIAN`. `WindowFitResult` gains a
  `shape` field; downstream consumers (rescue, knockout, cleanup,
  result_conversion's `effective_tau` lookup) read `inner.shape` so
  Gaussian-fitted windows stay coherent through later evaluation.
- **632 / 632 non-slow tests pass**; the Lorentzian default propagates
  with no behaviour change at existing call sites.

Remaining work (next session):

- **Stage 2b twin**: add `extract_tau_G_majority` in
  `fitting/tau_calibration.py` (lift Part B's per-bin Voigt fit;
  filter to converged + finite `τ_G < cap` + `Δχ²ᵣ ≥ 1` on the
  contributor pool); add `_internal/stage2b_g_impl.py` mirroring
  `stage2b_impl.py`; new stage name `stage2b_tau_G_calibration` with
  the same dependencies as `stage2b_tau_calibration`. Reuse the
  existing `TauCalibrationResult` struct + serialisation under a
  different HDF5 path.
- **Stage 5 wiring**: `fit_peaks_impl(shape='lorentzian'|'gaussian')`
  reads `stage2b_tau_G_calibration` when `shape='gaussian'`. Pass
  `shape` into `execute_plan`. Persist a `shape` attribute on
  `/stage5_fitting` (default `'lorentzian'` for back-compat with
  existing files).
- **HDF5 persistence**: `fitting_serialization._save_window_fit` writes
  the per-window `shape` attribute; `_load_window_fit` reads it back
  into the `FittingResult` (default `'lorentzian'`).
- **CLI/Pipeline/api parity**: `--shape` flag on `fit-peaks`;
  `Pipeline.fit_peaks(shape=...)`; `api.fit_peaks(shape=...)`.
  `Pipeline.calibrate_tau_G(...)` / `api.calibrate_tau_G(...)` /
  `ftmwpipeline calibrate-tau-G` CLI.
- **Comparison script** in
  `dev-docs/research/gaussian-shape/compare_shapes.py`: run both
  shapes on the same fixture; emit per-window `χ²ᵣ`, AICc, peak
  frequencies, peak amplitudes, peak count as JSON + a 2-panel
  figure.
- **Tests**: unit tests for `extract_tau_G_majority`, integration test
  exercising `fit_peaks(shape='gaussian')` on a tiny synthetic,
  cross-interface consistency test for the Gaussian path.

## Open questions

- **Field naming.** `WindowFitResult.tau_us` currently stores the
  pure-exp τ. Under Gaussian shape it stores `τ_G`. Either keep
  `tau_us` as a shape-conditioned field (consumer reads `shape` attr
  to interpret), or rename to a shape-neutral name. The first is less
  intrusive but more error-prone; the second is cleaner but requires
  a schema migration. Start with shape-conditioned `tau_us`, defer
  rename to a follow-up.
- **Tau-error and the parameter-error covariance** are computed the
  same way for either shape; no schema change needed there.
- **Knockout / residual-rescue** are shape-agnostic at the χ² level;
  no code changes beyond passing `shape` through.
- **Stage 3 gap-pass and Stage 5 rescue τ** currently consume `τ_maj`
  from the original Stage 2b. When `shape='gaussian'` is selected at
  Stage 5, those upstream consumers don't change — they still use the
  pure-exp `τ_maj`. (Their role is peak detection / candidate
  generation, where the Lorentzian τ is an acceptable proxy.) Only
  the Stage 5 window-fit consumes the Gaussian-twin calibration.

## Out of scope

- Full Voigt `(τ_L, τ_G)` joint fit in production. Deferred; covered
  separately in [`stage5-voigt-deficit.md`](stage5-voigt-deficit.md).
- A shape-selector for Stage 3 (peak detection) — the Lorentzian
  approximation is fine for detection; only the fit needs Gaussian.
- Renaming `WindowFitResult.tau_us` field — incremental, follow-up.
- Mechanistic explanation of why the line shape is Gaussian on 2638
  (see [`stage5-voigt-deficit.md`](stage5-voigt-deficit.md) discussion;
  the geometry doesn't cleanly support either pure Doppler or pure
  spatial-windowing as the source). The shape selector ships
  regardless; mechanism understanding is a separate question.