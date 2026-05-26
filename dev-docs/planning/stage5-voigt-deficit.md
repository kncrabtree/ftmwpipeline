# Stage 5 Voigt-deficit prototype

Research plan for extending the Stage 5 line shape from pure exponential
decay to a Voigt envelope (Lorentzian × Gaussian) on the small set of
windows where the single-exponential model leaves significant residual.

## Motivation

The Stage 5 model currently fits each line as a damped cosine
`s(t) = A · exp(-t/τ) · cos(2π f₀ t + φ)`. On 2638 there are ~6 windows
where this single-exponential shape can't capture the data: w198
(33721-33726 MHz) is the canonical case, with `τ` data-preferring
2.75 µs against a molecular `τ_maj ≈ 6.16 µs` and a `chi²_r = 15`
unprior-fit vs `chi²_r = 56` prior-anchored. The other shape-error
windows on 2638 (33724, 33840, 33659, 33421, 35839, 38862) show the
same signature.

Physical picture: τ in the current model is the transit-time-dominated
T₂ of the FID. A separate Gaussian contribution arises from the cos²θ
velocity-projection distribution of the supersonic beam through the
horn's probe volume (and from thermal broadening in static-cell
environments). Mathematically, an ensemble of molecules each emitting
at slightly different `f₀` with a (symmetric) Gaussian distribution of
width `σ_f` produces:

```
s(t) = ∫ g(f₀; f̄, σ_f) · A · exp(-t/τ_L) · cos(2π f₀ t + φ) df₀
     = A · exp(-t/τ_L) · exp(-(t/τ_G)²) · cos(2π f̄ t + φ)
```

with `τ_G = √2 / (2π σ_f)`. So the time-domain form is closed: an
exponential × Gaussian product. The frequency-domain Voigt profile
(convolution of Lorentzian × Gaussian) has no closed form — it is
evaluated via the Faddeeva function `scipy.special.wofz` or
approximated by pseudo-Voigt.

The single-exponential model absorbs the Voigt shape by shortening τ
to broaden the line; the unconstrained "tau-collapse" pattern in
shape-error windows is the LSQ doing this trade-off. The
[Stage 2b Voigt-deficit synthetic case](../research/stage5-tau-calibration/report.md)
already quantifies this: `τ_L = 7.5, τ_G = 12 µs` truth recovers
`τ_maj = 5.12 µs (-32 %)` under the pure-exponential fit.

## Two-part prototype

Run both parts in parallel; Part B confirms the hypothesis cleanly,
Part A tests whether the Stage 5 LSQ can carry the extra parameter.

### Part A — per-window (τ_L, τ_G) LSQ on shape-error windows

Build a research script that takes a Stage 4 plan + the persisted
active FT, picks the shape-error candidates from 2638 (w198 + the 5-6
others by frequency match), and re-fits each with the Voigt model.

Per-window output:
- `(τ_L, τ_G)` and their covariance from the LSQ
- Per-peak `(f, A, φ)` parameters
- `chi²_r` improvement vs the single-exponential fit
- Conditioning diagnostic: smallest eigenvalue of `Jᵀ J`, condition
  number, identifiability of `(τ_L, τ_G)`

Inner LSQ implementation:
- Start with pseudo-Voigt (Thompson-Cox-Hastings: `η · L + (1-η) · G`
  with `η` from L/G width ratio). ~99 % accuracy, ~2× exp evaluation
  cost. Analytic Jacobian rows for `τ_L` and `τ_G` are derivable.
- Fall back to `scipy.special.wofz` if pseudo-Voigt approximation
  proves insufficient at the chi² floor of these windows.
- Bound `τ_G` in a wide physical range (say [0.5, 100] µs).
- Initialize `τ_G = ∞` (pure Lorentzian) so the seed is the current
  best fit; LSQ shortens `τ_G` only if data supports it.

Acceptance:
- w198 chi² drops from 22 (unlocked-rescue + prior) to ≤ 10.
- At least 4 of the 6 shape-error windows show chi² reduction > 50 %.
- LSQ is well-conditioned at recovered `(τ_L, τ_G)`: condition number
  < 100, smallest eigenvalue > some sensible threshold.

If conditioning fails:
- Add a soft Gaussian prior on `τ_G` (analogous to the current
  bidirectional `τ_L` prior) sourced from Part B's calibration.

### Part A outcome on 2638

Implemented as
[`research/voigt-deficit/part_a_perwindow.py`](../research/voigt-deficit/part_a_perwindow.py).
Candidate set: 9 per-band-production windows with `χ²ᵣ > 10` AND
`tau_was_fit=1` AND `n_peaks ≥ 2` (single-peak high-`χ²ᵣ` windows are
prior-penalty-driven, not shape-error). The pure-exp baseline is a
free-τ re-fit on the same active-FT subset (no prior) so the
comparison isolates the Voigt shape benefit from prior anchoring.

Per-window results (sorted by baseline `χ²ᵣ`):

| wid  | freq (MHz)       | K  | baseline | voigt | τ_L  | τ_G   | factor |
|------|------------------|----|----------|-------|------|-------|--------|
| w300 | 36340-36355      | 4  | 76.62    | 78.97 | 5.04 | 100.0 | 1.0×   |
| w360 | 38857-38867      | 2  | 69.56    | 40.37 | 95.2 | 5.15  | 1.7×   |
| w148 | 31322-31331      | 3  | 54.96    | 55.45 | 7.50 | 18.4  | 1.0×   |
| w310 | 36806-36819      | 3  | 38.72    | **17.54** | 100  | 6.09  | **2.2×** |
| w141 | 31208-31219      | 11 | 20.05    | **8.71**  | 38.6 | 7.73  | **2.3×** |
| w218 | 33837-33844      | 3  | 16.11    | 17.07 | 5.55 | 17.2  | 0.9×   |
| w239 | 34558-34570      | 5  | 15.64    | 13.58 | 14.3 | 8.16  | 1.2×   |
| w355 | 38717-38722      | 2  | 15.45    | **5.32**  | 100  | 5.73  | **2.9×** |
| w213 | 33722-33726      | 8  | 13.79    | **6.39**  | 100  | 5.88  | **2.2×** |

- 4 of 9 (44 %) clear the > 50 % reduction gate: w141, w213, w310,
  w355. Plus w360 at 42 % (close). The four that succeed are the
  monotonic-envelope shape-error cases.
- 4 of 9 (w300, w148, w218, w239) show ≤ 13 % reduction — these are
  multi-line blends where neither pure-exp nor Voigt captures the
  line-to-line interference inside the window. Same population as
  Part B's bad-fit pool (negative finding there).
- w198 acceptance criterion: planning expected "w198 χ²ᵣ drops 22 →
  ≤ 10". Per the current per-band fit w198 itself sits at χ²ᵣ ≈ 1.87,
  but the analogous high-χ² window in the same region is w213
  (33722-33726 MHz). Voigt drops w213 from 13.79 → 6.39 (≤ 10 ✓).

**Identifiability and conditioning**: median LSQ condition number
~5e17 — the LSQ is poorly conditioned because `τ_L` saturates at the
upper bound (100 µs) on every window where Voigt helps. This is the
`τ_L`/`τ_G` identifiability concern the design noted: when `τ_G < τ_L`
the Gaussian dominates the decay envelope, so `τ_L` is loosely
constrained and the LSQ pushes it to the bound. Recovered `τ_G` on
the four winners (5.15-7.73 µs) tracks the Part B per-band
calibration (low 9.24, mid 8.68, high 7.56) within ~30 % — consistent
within band-to-band scatter and the broad `τ_G_seed` multi-start
grid.

**Architectural implications**:
- The simple "swap pure-exp envelope for Voigt envelope" extension
  (option A in the architectural-decision section) works for the
  shape-error subset that's a single Voigt-shaped line per window,
  but does not address multi-line blends.
- Production deployment will want the Part B `τ_G` prior anchoring
  to break the `τ_L`/`τ_G` identifiability degeneracy. With the
  prior, `τ_L` should land near a sensible Lorentzian and `τ_G`
  near the band-calibrated value — instead of `τ_L=100` pinning.
- Multi-line blends (w148, w218, w300) are out of scope for a single
  Voigt envelope; they need a different solution (joint blend fit
  with per-component shape parameters, or a baseline-correction
  approach for the inter-line interference).

### Part B — per-bin Voigt fit on currently-bad-fit STFT bins

The Stage 2b STFT calibration's bad-fit gate
(`rss_exp > 5 · n_seg · (0.05 · m̄)²`) currently excludes ~50-100
strong on-line bins per fixture (per-frame SNR 240-360 on 2638)
because their `rss_exp` exceeds the relative gate. The hypothesis is
**these bins are bad-fit precisely because they're Voigt-shaped**: the
per-bin `|S_n|` time evolution isn't pure exponential.

Test: fit those bins to
`|S_n(a)| = C · exp(-a/τ_L) · exp(-(a/τ_G)²)` (the per-bin
Voigt-shape time evolution at the population level). With per-frame
SNR > 100, the 2-parameter fit should be well-conditioned. Output per
bin: `(τ_L, τ_G, chi²_voigt)`. Compare against the bin's pure-exp
chi² (currently in the bad-fit classification).

Acceptance:
- Voigt model's per-bin chi² closer to noise floor than pure-exp on
  the same bins (e.g., median Voigt chi²_r < 2× the median noise-only
  expectation, vs pure-exp's 10-20×).
- `τ_G` per-bin extracted in a tight band per frequency third
  (consistent with horn-coupling geometry: shorter `τ_G` at higher f).
- Per-band median `τ_G` recoverable as a τ_G calibration prior for
  Part A.

If Part B succeeds, we have:
- Direct evidence the Voigt shape is the right physical model
- A per-band `τ_G(f)` calibration from independent strong-line bins
- An anchor for Part A's per-window `τ_G` fits

### Part B outcome on 2638

Implemented as
[`research/voigt-deficit/part_b_perbin.py`](../research/voigt-deficit/part_b_perbin.py).
The bad-fit-hi-SNR-bin hypothesis is **contradicted on 2638**: the 99
bad-fit bins above SNR 100 have non-monotonic `|S_n(a)|` traces
dominated by line-blend interference (two close lines beating inside
one bin's resolution), and no monotonic Voigt fits them either —
Voigt χ²ᵣ median = 887 vs pure-exp 1033, only a 1.16× factor at
χ²ᵣ ≫ 100. The script kept the bad-fit pool as a reported "negative
finding" and pivoted to the **strong-contributor pool** (cls=3,
SNR > 20, 408 bins in trim on 2638):

- Per-bin Voigt χ²ᵣ median = **0.80** (vs pure-exp **3.77** —
  4.7× improvement, at the noise floor).
- 94 % of contributors see Voigt improve over pure-exp; 311/408
  (76 %) clear the calibration gate (`Δχ²ᵣ ≥ 1` and
  `τ_G < 70 µs`).
- Per-band `τ_G` median + robust σ (IQR / 1.349) on the
  calibration-eligible subset:
  - low  (26500-31000 MHz, N=66):  τ_G = 9.24 µs ± 1.15 µs (12 % rel)
  - mid  (31000-35500 MHz, N=106): τ_G = 8.68 µs ± 1.75 µs (20 % rel)
  - high (35500-40000 MHz, N=139): τ_G = 7.56 µs ± 1.48 µs (20 % rel)
- Monotonic 1/f trend across the band — consistent with horn-coupling
  geometry (higher f → wider velocity-projection Δf → shorter τ_G).
- Per-band median `τ_L` runs long (17-40 µs) — when `τ_G < τ_L` the
  Gaussian dominates the decay envelope and the data weakly
  constrains `τ_L`. Implication for Part A: the existing
  Stage 2b/Stage 5 `τ_maj ≈ 5-7 µs` is an *effective* decay of a
  Gaussian-dominated Voigt, **not** the Lorentzian `τ_L`; Part A's
  joint `(τ_L, τ_G)` LSQ should NOT use `τ_maj` as a `τ_L` prior.

Calibration artefact for Part A:
[`data/tau_G_band_majorities.json`](../research/voigt-deficit/data/tau_G_band_majorities.json).

## Architectural decision (post-prototype)

Decide based on Part A + Part B evidence:

**Per-window free `τ_G`** (least intrusive): Stage 5 adds `τ_G` as a
per-window shared parameter alongside `τ_L`. Anchored by Part B's
per-band calibration (soft prior). No changes to Stage 2b's existing
bad-fit gate (the strong bins stay classified as bad-fit for the
`τ_L` calibration purpose; they're a separate input to a new `τ_G`
calibration).

**Per-fixture / per-band `τ_G`** (more efficient): Stage 5 reads
`τ_G(f_window_center)` from Part B's calibration as a fixed
parameter; only `τ_L` is fit per window. Fewer LSQ params, fits
faster, but loses per-window flexibility for non-horn-coupling shape
variation.

**Hybrid** (`τ_G` fit per window with a strong calibrated prior):
similar to current `τ_L` policy with bidirectional Gaussian prior.
Allows data-driven drift but biases toward calibration consensus.

## Asymmetric extension (deferred)

If the supersonic beam has an angle offset from the horn axis (a
known possibility on 2638's geometry), the cos²θ velocity-projection
distribution becomes skewed and the line shape is asymmetric. The
symmetric Voigt won't fully capture it; residuals at canonical
shape-error windows would still show asymmetric structure.

Defer until symmetric Voigt prototype lands. Approaches if extension
is needed:
- Skewed Voigt (extra asymmetry parameter; no closed form for the
  asymmetric profile)
- Stacked Voigt doublet (two Voigts with shared `τ_L`, `τ_G`,
  offset by ±δf; effectively a 2-component model for the asymmetric
  population)

## Implementation surface (post-prototype)

If the prototype lands:

- New shared parameter `τ_G_us` in `WindowFitResult.shared_parameters`
  (alongside `τ_us`).
- Per-window Voigt model in `peak_model.py` (or a new
  `voigt_peak_model.py`) — pseudo-Voigt as the default; wofz as a
  forensic/sanity-check option.
- Jacobian extension in `_penalty_residuals_and_jacobian` for `τ_G`
  derivatives.
- Per-bin Voigt extraction in
  `fitting/tau_calibration.py` (optional Part B output as a new
  `band_majorities` companion field `τ_G_band_majorities`).
- Stage 5 reads `τ_G_band_majorities` when present; falls back to
  `τ_G = ∞` (pure Lorentzian) when absent.
- Persistence: extend the HDF5 schema for `τ_G`, `σ_τ_G` on per-window
  fit results and per-band calibration.
- Cross-interface: CLI / Pipeline / functional API parity.

## Risks

- **τ_L / τ_G identifiability.** Both broaden the line; the LSQ may
  have a near-flat valley along a `(τ_L, τ_G)` direction. Mitigation:
  strong `τ_G` prior from Part B (turns the joint fit into an
  effectively single-parameter fit on `τ_L`).
- **Performance.** Pseudo-Voigt evaluation is ~2× exp; over hundreds
  of windows × tens of LSQ iterations, this is real wall-time. Profile
  early.
- **Stage 2b refresh implication.** Stage 2b's per-bin pure-exp fit
  on contributor bins (the *not*-bad-fit bins) extracts an effective τ
  that combines `τ_L` and `τ_G`. The current `τ_maj` is therefore
  effectively `τ_eff`, not `τ_L`. Two options for downstream
  consumption:
  - Keep `τ_maj` as `τ_eff` and use it directly (Stage 5 doesn't need
    `τ_L` directly when also fitting `τ_G` jointly).
  - Back-solve `τ_L` from `τ_maj` and Part B's `τ_G` per band:
    `1/τ_eff² = 1/τ_L² + (something) · 1/τ_G²`. Requires care with
    the actual time-domain product's effective-decay-time definition.
- **Cross-fixture validity.** The `τ_G(f)` calibration on 2638 is
  tied to that instrument's horn geometry. Other fixtures may have
  different `τ_G(f)` curves; the calibration is per-fixture.
- **Overfitting on weak windows.** Adding `τ_G` doubles the shape
  parameters. On windows with low SNR or few peaks, the data may not
  support the extra freedom. Mitigation: keep `τ_G` fixed (at
  band-calibrated value) on windows below a SNR threshold; only
  free-fit it on strong windows.

## Out of scope

- Multi-component Voigt (e.g., separate transit-time + thermal
  Gaussians for the same line). Single `τ_G` per window/band suffices.
- Changing Stage 3 peak detection. Voigt model only affects the
  Stage 5 LSQ shape, not the peak-finding upstream.
- Other line-shape models (Doppler-shifted, Faraday-rotated, …).
  Voigt is the standard FT-MW shape for the relevant physics.

## Success criteria

- **Part A**: w198 chi² ≤ 10 (currently 22). At least 4 of 6
  shape-error windows on 2638 show > 50 % chi² reduction.
- **Part B**: per-bin Voigt fits land chi² near the noise floor on
  the strong bins, with `τ_G` consistent across bins within each
  frequency band.
- **Combined (post-implementation)**: Stage 5 chi² distribution on
  stage5-validation3 sees p95 drop > 20 %, p99 > 30 %, max > 30 %.
- **Cross-fixture**: at least 1 additional fixture from a different
  instrument geometry validates the model (or motivates a fixture-
  specific `τ_G(f)` calibration).
