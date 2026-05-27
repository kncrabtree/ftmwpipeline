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
`band_majorities_G` instead of pure-exp τ. Algorithm: per-bin pure-exp
NLS polish as the seed, then a multi-start pure-Gaussian fit
`|S| = C exp(-(a/τ_G)²)` on the contributor bins, filtered to
converged + finite `τ_G < cap` + `Δχ²ᵣ(exp − gauss) ≥ 1`. The
estimator matches the Stage 5 `shape='gaussian'` window-fit envelope —
a Voigt decomposition's `τ_G` would describe the Gaussian component
*after* the Lorentzian decay is absorbed into a separate `τ_L`, which
over-estimates the envelope-equivalent τ that the window fit recovers.
The function `extract_tau_G_majority` mirrors `extract_tau_majority`
in shape and signature.

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

Landed (`commits 699db4c`, `2c01b30`, plus the Stage 2b twin + wiring
described below):

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
- **Stage 2b twin**: `extract_tau_G_majority` in
  `fitting/tau_calibration.py` (per-bin pure-exp NLS polish + multi-
  start pure-Gaussian fit `|S| = C exp(-(a/τ_G)²)`; eligibility gate
  `converged ∧ τ_G < 0.7 · τ_G_max ∧ Δχ²ᵣ(exp − gauss) ≥ 1`);
  per-band SNR-weighted majority via the existing
  `compute_band_majorities` machinery. The Voigt helpers
  (`_voigt_residuals`, `_fit_voigt_nls_multistart`) stay in the module
  for the future 3-way L/G/V shape-recommendation comparator but no
  longer drive the production calibration -- their `τ_G` is the
  pure-Gaussian component after the Lorentzian decay is absorbed into
  a separate `τ_L`, which over-estimates the envelope-equivalent τ
  that the Stage 5 pure-Gaussian window fit recovers.
  `_internal/stage2b_g_impl.py` drives it (`calibrate_tau_G_impl`,
  `load_tau_G_calibration_impl`, `tau_G_calibration_present`); new
  stage name `stage2b_tau_G_calibration` registered in
  `PipelineStageTracker.STAGE_DEPENDENCIES` (deps:
  `[stage0_fid_data, stage1_complex_ft, stage2_noise_result]`,
  identical to the pure-exp twin). HDF5 group
  `/stage2b_tau_G_calibration` reuses the existing
  `tau_calibration_serialization` writer/reader (a `shape='gaussian'`
  attr on the root group disambiguates from the pure-exp twin).
- **Stage 5 wiring**: `fit_peaks_impl(shape='lorentzian'|'gaussian')`
  routes to the matching Stage 2b twin based on the caller's
  `shape`; missing τ_G calibration logs a warning and runs without a
  prior. `shape` is threaded into `execute_plan(...)` and recorded
  in the persisted `SpectrumFit.parameters['shape']`. The
  `/stage5_fitting` root group carries a new `shape` attr; per-
  window subgroups carry it too (default `'lorentzian'` for files
  pre-dating this attribute).
- **HDF5 persistence**: `FittingResult` gained a `shape: str` field
  (default `'lorentzian'`); `result_conversion` copies it from
  `WindowFitResult.shape`; `_save_window_fit` writes the attr;
  `_load_window_fit` reads it back with a `'lorentzian'` fallback for
  legacy files.
- **CLI/Pipeline/api parity**: `--shape {lorentzian,gaussian}` flag on
  `fit-peaks`; `Pipeline.fit_peaks(shape=...)`;
  `api.fit_peaks(shape=...)`. New `ftmwpipeline calibrate-tau-G` CLI
  with the same knob coverage as `calibrate-tau`;
  `Pipeline.calibrate_tau_G(...)` / `Pipeline.load_tau_G_calibration()`;
  `api.calibrate_tau_G(...)` / `api.load_tau_G_calibration(...)`.
- **Tests**: unit
  `tests/unit/fitting/test_tau_calibration.py::TestExtractTauGMajority`
  exercises the eligibility filter + multi-start pure-Gauss + majority
  on a synthetic Gaussian-envelope FID (τ_G recovery within ±20 %);
  integration suite
  `tests/integration/test_stage5_fitting.py` gains
  `test_calibrate_tau_G_cross_interface`,
  `test_fit_peaks_gaussian_cross_interface`, and
  `test_fit_peaks_gaussian_persists_and_loads_shape`. 635 / 635 non-
  slow tests pass.
- **Comparison script**:
  `dev-docs/research/gaussian-shape/compare_shapes.py` runs both
  shapes on the same fixture, dumps per-window χ²ᵣ / AICc / peak-
  count CSV and JSON aggregates, matched per-peak frequency/amp CSV,
  and a 2-panel comparison figure (per-window χ²ᵣ scatter with
  Part A shape-error windows highlighted, per-peak frequency
  residual vs SNR).

## First validation pass on 2638 unapodized

> Numbers here predate the Stage 2b τ_G estimator swap from Voigt to
> pure-Gaussian. The τ_G_maj quoted below is the Voigt-anchored value;
> on the same fixture the current pure-Gaussian estimator gives
> `τ_G_maj = 6.96 µs` band-wide (8.39 / 6.75 / 6.24 per band). Re-run
> `compare_shapes.py` against the new anchors before re-deriving
> per-shape acceptance numbers; the qualitative observations
> (Gaussian wins on Part-A shape-error windows, not everywhere) are
> expected to hold but should be re-verified.

The first run of `compare_shapes.py` on `exp_2638_unapodized.ftmw` (Stage
2b τ_G calibration: `τ_G_maj = 8.52 µs`, `σ_τ_G = 1.84 µs`, 382 windows
band-routed) returned:

- **Global aggregate**: median χ²ᵣ Lorentzian = 1.40, Gaussian = 1.41
  (essentially tied; Gaussian < Lorentzian on 182 / 382 windows). p95
  χ²ᵣ: Lorentzian = 6.36, Gaussian = 7.87. Median ΔAIC = −0.34. So
  globally, Gaussian does **not** beat Lorentzian on this fixture by
  the planning-doc bar (median χ²ᵣ drop predicted ~5×; p95 < 5).
- **Part A shape-error windows**: three of the four predicted "Gaussian
  wins" cases land emphatically:
  - w141: χ²ᵣ 20.07 → 8.88, ΔAIC = **+218.7** (Gaussian preferred)
  - w213: χ²ᵣ 18.80 → 8.48, ΔAIC = **+96.2**
  - w310: χ²ᵣ 38.61 → 18.04, ΔAIC = **+258.5**
  - w355: χ²ᵣ 2.31 → 6.11, ΔAIC = **−118.8** (Lorentzian preferred)
  So the Gaussian path *does* materially reduce χ²ᵣ on the windows the
  Voigt-deficit research nominated, but it isn't a universal win —
  w355 was already a good Lorentzian fit and Gaussian makes it worse.
- **Peak count / amplitude**: Gaussian fits 625 peaks vs Lorentzian's
  713, with 613 matched on shared frequencies. Frequency agreement on
  shared lines is excellent (median residual 0.4 kHz, RMS 55 kHz).
  Gaussian-fitted amplitudes are systematically ~75 % of the
  Lorentzian amplitudes on shared lines (different normalisation of
  the time-domain envelope; expected).

Interpretation: the Voigt-deficit Part A and Part B research found that
the per-window joint `(τ_L, τ_G)` LSQ degenerated to Gaussian-dominant
on the shape-error windows, and that's exactly where the Gaussian
production path now wins. The non-universal global win means the line
shape on 2638 is *not* a clean pure-Gaussian everywhere — some windows
(at least w355) sit closer to the Lorentzian limit. A future per-window
shape-selector (the planning doc's "out of scope" item) would let
Stage 5 pick the best shape per window rather than committing globally;
right now the operator picks one shape per fit and accepts the
trade-off.

The infrastructure works; the acceptance bar that needs revisiting is
the planning-doc claim that Gaussian wins everywhere. Refine the bar
in light of this finding, and/or commit to the per-window selector,
before the second fixture's validation pass.

Remaining work (next session):

- **Reconcile the acceptance bar**: the planning-doc prediction "median
  χ²ᵣ < Lorentzian, p95 < 5" did not hold globally on 2638. Decide
  whether (a) the bar was over-optimistic (relax it to "Gaussian wins
  on Part-A-style shape-error windows by ΔAIC > 5", which it does on
  3/4), (b) the production path needs a per-window shape selector to
  combine both shapes' strengths, or (c) something in the Gaussian
  fit machinery is suboptimal (e.g. the tau bound, the prior
  strength, or the rescue's behaviour with Gaussian-fitted contributors).
- **Cross-fixture validation**: the user-supplied second fixture
  (clean lines + internal-rotation doublets) drives the
  generalisation check. Compare the per-window χ²ᵣ pattern: does the
  second fixture show a different lorentzian-vs-gaussian split than
  2638?
- **w355 deep-dive**: w355 lost ΔAIC = −118 under Gaussian. Look at
  the window's residual under both shapes and the per-window τ to
  understand whether this is a true Lorentzian-dominant window
  (rotational-cooling regime) or a fit-machinery artefact.

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
- **Shape-aware STFT classifier (next-session headline; will shift
  every downstream τ_G calibration value on 2638).**
  `stft_calibration` labels bins via a pure-exp bad-fit gate:
  ``rss_exp > rss_gate_factor · n_seg · (relative_gate_fraction ·
  mean(|S|))²``. On Gaussian-envelope data this gate fires on the
  strong on-line bins themselves (their per-frame magnitudes don't
  follow pure-exp), so they land in `cls=2` and `extract_tau_G_majority`
  + the 3-way shape-recommendation hook see only sidelobe `cls=3`
  bins. The exp-only gate was set up before the Voigt-deficit work
  established that 2638's lines are Gaussian-dominant; the
  "rejected on-line bins must be line blends" rationale that was
  used to defend the gate has been superseded by the finding that
  the on-line bins fail the gate because the *model* is wrong, not
  the data. The pure-Gauss τ_G calibration that landed in this
  cycle matches the window-fit τ_G within 5-8 % per band on 2638
  because sidelobes near a Gauss-envelope line carry the same Gauss
  envelope (scaled by a sinc factor in amplitude) -- a principled
  rather than coincidental match -- but the production stance
  shouldn't rely on "benevolent sidelobe behaviour from an
  objectively incorrect classifier". The next-session fix is to
  parameterise `stft_calibration` by shape so its bad-fit gate uses
  the shape-matched residual (pure-exp for ``shape='lorentzian'``,
  pure-Gauss for ``shape='gaussian'``, or "best-of-three" for the
  3-way shape-recommendation hook); both τ calibrations and the
  recommendation hook then read shape-correct contributor pools.
  Downstream consequences are accepted: the per-band τ_G anchors
  on 2638 will shift (likely smaller, matching the on-line per-bin
  pure-Gauss τ_G ≈ 6 µs from the synthetic test more closely), the
  `instrument_bc_2638.yaml` λ choice will need re-validation, and
  the 3-way verdict's vote rates will shift toward gauss as the
  on-line bins enter the pool with their stronger AICc preference.
  See ``scratch/next-session-prompt.md`` for the full plan. The
  current ``compute_shape_recommendation(include_bad_fit_bins=True)``
  kwarg is plumbed for diagnostic experiments only and becomes
  obsolete once the classifier becomes shape-aware.

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