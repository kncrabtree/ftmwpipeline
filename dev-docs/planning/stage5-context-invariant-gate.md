# Plan: Stage 5 context-invariant accept gate (information-weighted χ²)

> **Status update — prototyped, not shipped.** The information-weighted χ² of
> this plan was **falsified** in validation; the penalized raw-χ² reformulation
> survives, and its high-SNR-vs-dense tension is **resolved** by two per-bin
> sigma_eff budgets (V4 local lineshape + V5 frozen-skirt): the band anchors
> hold simultaneously (363 cap=8 dense recovery K=6, 2638 control 20 lines,
> 1512 bright-band over-add 109→13 vs legacy 11, with the 10× slowdown gone).
> All variants are behind flags in `fitting/validation.py` with **defaults left
> at legacy** (`DEFAULT_GATE_PENALTY_LAMBDA=None`,
> `DEFAULT_GATE_FLOOR_SCALING=False`, `DEFAULT_WEIGHTED_GATE_CHI2=False`,
> `DEFAULT_GATE_SIGMA_EFF_KAPPA=None`, `DEFAULT_GATE_SIGMA_EFF_KAPPA_SKIRT=None`);
> 309 fitting tests green. The **Stage-4 small-window retune (lever 1) is
> validated**: cap=8 MHz (≈96 points; `max_window_width_points` productionized,
> default off) plus two flag-gated companion fixes
> (`plan_execution.DEFAULT_EDGE_FREE_FREQ_REFINE`,
> `validation.DEFAULT_ENFORCE_SEED_KNOCKOUT`) holds or beats legacy on all
> seven fixtures and makes the dense 363/655 full fits tractable (~5 min vs
> killed). See **Measured outcomes** and **Stage-4 small-window retune** at the
> bottom, and the handoff `scratch/stage5-gate/HANDOFF.md`.

Status: **proposed.** No code written. This is the structural fix that the
performance work ([`stage5-nls-performance.md`](stage5-nls-performance.md))
identified as the blocker for every region-shrinking speedup lever, and it is a
correctness improvement in its own right: the Stage 5 accept gate is sensitive
to **window size**, which it should not be — whether a line is real is a local
question, independent of how many empty bins surround it.

## The problem

The conservative add-one-peak accept gate (and the blend-escalation, knockout,
and merge gates) decides K vs K±1 by comparing
`AICc = 2k + n_eff·log(χ²/n_eff) + 2k(k+1)/(n_eff − k − 1)`
(`fitting/validation.py::calculate_aicc`), with
`n_eff = effective_sample_size(model, kind="perplexity_log1p_snr")` — the
perplexity `exp(H(p))` of the normalised per-bin information weights
`w_f = log1p(|model_f| / σ_f)`.

`n_eff` is (correctly) ~window-size-invariant: widening a window adds bins with
`w_f ≈ 0`, which barely move the perplexity. **But `χ²` is the raw
noise-weighted chi-squared over all `M` bins**, and for a unit-variance fit
`χ² ≈ n_data = 2M`. So the log-likelihood term is `n_eff·log(2M/n_eff)`, and the
accept benefit of adding a real peak is

```
ΔAICc_likelihood ≈ n_eff·log(χ²_K / χ²_{K+1}) ≈ n_eff·(Δχ²/χ²_K) ≈ n_eff·Δχ²/(2M)
```

A real peak's `Δχ²` (its energy in noise units) is **local** — independent of
`M`. So the accept benefit scales as **`1/M`** while the `2k` cost is fixed:
shrink the window and the same peak clears the bar more easily. That is the
entire window-size sensitivity, and it is a units mismatch — `n_eff` counts the
*informative* bins, but `χ²` is diluted by the `M − n_eff` **noise** bins, each
adding ~1 to `χ²` while carrying zero information about whether the peak is real.

### Evidence (this session)

The over-add that killed three separate speedup levers is all this one cause,
arriving by different routes:
- **Lever 0** (batched add-loop): over-added 103→116 on a heavy subset.
- **Lever 2a** (in-`conservative_fit` window decomposition): over-added 67→87 on
  363; metric held only because the downstream prune cleaned it up.
- **Stage-4 small-window retune** (`max_window_width_mhz` 40→8): −34 % wall on a
  moderate-density Lorentzian band with the metric holding, but **+67 % lines on
  a dense band and +87 % lines (2× slower) on 363**, because more/smaller
  windows systematically over-accept. Scaling the baseline order with point count
  did **not** fix it (order-4 was helping, not over-fitting), ruling out the
  baseline and confirming the gate.

See [`stage5-nls-performance.md`](stage5-nls-performance.md) "Measured outcomes"
for the full data and the band-fit samples.

## The fix: weight χ² with the same information weights as n_eff

Use an **information-weighted chi-squared** in the AICc, weighted by the *same*
`w_f = log1p(|model_f| / σ_f)` that defines `n_eff`:

```
χ²_w = n_eff · ( Σ_f w_f · r_f² ) / ( Σ_f w_f )        # = n_eff · weighted_mean(r²)
```

where `r_f² = |residual_f / σ_f|²` is the per-bin unit-variance squared residual
(real+imag, the same the raw χ² sums). For a good fit `r_f²` averages to ~1
everywhere, so `weighted_mean(r²) ≈ 1` and `χ²_w ≈ n_eff` — i.e.
`χ²_w / n_eff ≈ 1`, **independent of `M`**. Then:

- **Adding noise bins** (a wider window) adds `w_f ≈ 0` entries that move neither
  `χ²_w` nor `n_eff` → **AICc is window-size-invariant**. A peak's acceptance
  depends on its *local* evidence (how much it reduces the residual in
  high-information bins), not on how many empty bins surround it.
- **A real peak** reduces `r²` in its high-`w` bins → large `χ²_w` drop →
  accepted. **A spurious peak** reduces `r²` only in low-`w` noise bins →
  negligible `χ²_w` change → rejected. The gate weights evidence by information
  on *both* sides of the ledger, closing the loop perplexity opened.

## Implementation sketch

This is a small, central change with wide reach. Do it behind a flag so the
before/after is a clean A/B.

1. **`fitting/validation.py`:** add `noise_weighted_chi2_weighted(z, sigma,
   model, weights)` (or extend `calculate_noise_weighted_chi2`) computing
   `χ²_w = n_eff · Σ w·r² / Σ w`. Factor the `w_f = log1p(|model|/σ)` weight out
   of `effective_sample_size` so the gate computes `n_eff` and `χ²_w` from the
   *same* weight vector in one place. (`effective_sample_size` already builds
   `w`; expose it.)
2. **The four gates** (all in `window_fit.py` / `residual_rescue.py`): wherever a
   gate currently calls `calculate_aicc(fit.chi_squared, k, n_eff)`, pass `χ²_w`
   instead — computed from the **more-complex model's** weights so both sides of
   the comparison share one weight vector, exactly as `n_eff` already is shared:
   - `conservative_fit` add gate (and `_blend_aware_seed` K=2/3 escalation),
   - `knockout_test`,
   - `merge_close_peaks_cleanup` Tier-2,
   - `iterative_aicc_cleanup`.
   The shared-weights discipline is the analogue of the existing "n_eff keyed on
   the K+1 / K-fit model" rule — keep it.
3. **Keep reporting χ² raw.** `WindowFitResult.chi_squared` / `reduced_chi2`, the
   persisted `quality_metrics`, the SNR-aware pass metric
   (`snr_aware_chi2_pass`), and the diagnostic F-test stay on the raw χ² — they
   are calibrated and externally meaningful. Only the *internal AICc gates*
   switch to `χ²_w`. Be deliberate that gate-χ² and reported-χ² now differ.

## Validation plan

1. **It MUST change the baseline** (unlike a perf-only change) — that is the
   point. Re-run the cross-fixture SNR-aware metric (`stripped_metric.py` over
   2638/655/363/360/1231/1512/1019, persisted `stage5_fit` stripped) and require
   the pass rate to **hold or improve**, the bulk χ²ᵣ to stay sane, and the 2638
   control to stay ≈1.0. Diff the peak sets vs the shipped baseline to see what
   moved and confirm the changes are sensible (marginal lines, not strong ones).
2. **Window-invariance test** — the direct check: fit the same band at
   `max_window_width_mhz` = 40 and 8 (the `band_fit.py` sampler) and confirm the
   **line counts now match** (the over-add is gone). That is the gate working.
3. **Full suite** (`pytest`) — the gate is load-bearing; many fitting tests
   assert peak counts / acceptance and will need their expectations re-derived if
   the gate genuinely improved. Treat changed assertions as evidence to inspect,
   not noise to silence.

## Open questions to settle before/while building

- **Weight steepness.** `log1p(SNR)` is gentle; a near-threshold real peak has
  modest `w`, so its bins get modest weight — does that under-weight exactly the
  marginal peaks the gate decides on? A/B `log1p(SNR)` vs `|model|²` (the Fisher
  density, `kish_mag_sq`) for the weighted χ². This reconnects to **issue #9**
  (whether the K-vs-(K−1) merge/knockout sweeps want a magnitude-concentrated
  kind) — a context-invariant χ² may make that choice moot or may sharpen it.
- **Support-restricted variant.** Instead of the soft `n_eff·weighted_mean(r²)`,
  compute χ²_w only over the `hard_radius` informative support. More obviously
  local, but threshold-sensitive; the soft form is smoother across iterations.
  Decide which after the first A/B.
- **The `n_eff − k − 1 ≤ 0` divergence** still fires structurally; with an
  invariant `n_eff` it becomes purely a property of the feature, which is the
  intended behavior (a truly under-resolved blend hits `+inf` regardless of
  window).

## If this succeeds: revisit what failed this session

A window-invariant gate is the **enabler** the region-shrinking levers needed.
Once it holds the cross-fixture metric, revisit, in order of expected payoff:

1. **Stage-4 small-window retune** — the cleanest win (−34 % on the moderate
   band). With the gate fixed, the dense/Gaussian over-add should vanish. Bring
   back `max_window_width_mhz` ~8–12 + `magnitude_attachment_threshold` ~1.0
   (lean on the order-4 baseline for the now-flat far pedestal); validate the
   contributor relaxation separately. A **points-based** cap likely generalises
   better than MHz across fixtures. Consider gap-aware splitting only if a
   residual dense-blend issue remains.
2. **Lever 2a** (window decomposition inside `conservative_fit`) — its
   `window_fit_2a_decompose.patch` is preserved; the amplitude/shape-aware
   coupling model is correct physics. With an invariant gate the per-cluster
   null-gate stops over-/under-accepting.
3. **Lever 0** (batched add-loop) — least likely to pay (the loop is already
   ~8 solves/window) but its over-add was also gate-driven; re-evaluate only if
   1–2 leave the add-loop the bottleneck.

## Reusable harnesses (all in `scratch/stage5-nls/`)

- `capture_windows.py <fid> <n> <min_k>` + `replay_bench.py` — capture real
  `conservative_fit` calls, replay fast (the 97 % path) with peak-set
  fingerprint; `diff_dumps.py` for tolerance peak-set diffs.
- `band_fit.py <fid> <flo> <fhi> <cap> <thresh>` — fit only a frequency band
  (monkeypatched plan-load), the window-invariance + small-window sampler.
- `stripped_metric.py` / `strip_decomp.py` — full-fixture cross-fixture pass
  metric (strip persisted `stage5_fit` first — persisted-shadow gotcha).
- `profile_fit.py` — cProfile the replay (the assembly-cost breakdown).
- `char_stage4.py` / `char_thresh.py` / `check_contrib.py` — Stage-4 window /
  contributor characterisation.
- Full fits are 15–25 min each — **characterise and band-sample; do not sweep
  full fits.**

## Measured outcomes

The gate switch is centralised in `validation.gate_aicc_pair(...)`; the
`weighted_gate_chi2` bool and the `ref_reduced_chi2` value are threaded through
all five Stage-5 gates. Three variants were prototyped behind flags.

### V1 — information-weighted χ² (this plan's proposal): **falsified**
`χ²_w = n_eff·Σw·r²/Σw` fed to the AICc roughly **doubled the 2638 control line
count** (20→41 at fixed `cap=def`) and exploded on dense windows. Diagnosis: the
plan blamed a `1/M` *over*-accept on small windows, but the existing gate uses
`n_eff` (not `n_data`), so its benefit `≈ Δχ²·n_eff/(2M)` is already
*conservative* — the `n_eff/(2M) < 1` factor is its (window-size-dependent)
evidence bar. Replacing the `~2M` denominator with `χ²_w ≈ n_eff` removes that
shrinkage entirely → far too permissive. `DEFAULT_WEIGHTED_GATE_CHI2 = False`.

### V2 — penalized raw-χ² `score = χ²_raw + 2λk` (accept iff `Δχ² > 2λΔk`)
Window-size-invariant **by construction**: the two compared models share the
window's bins, so `Δχ²` is the local likelihood-ratio statistic — no `n_eff`, no
`M`. `λ` is the single knob (`DEFAULT_GATE_PENALTY_LAMBDA`).
- **λ=5 preserves production AND fixes the dense under-fit.** 2638 `cap=def`
  reproduces legacy exactly (20 lines, bulk χ²ᵣ 1.54); 655/363 `cap=def`
  ≈legacy. The win is at `cap=8`: the legacy `n_eff` AICc collapses to **K=1
  χ²ᵣ=52** on the dense 363 Q-branch sub-window (the `n_eff ≤ k+1` /
  small-`n_eff` pathology), and λ=5 recovers **K=6** (λ=2 → K=8) — so the same λ
  that holds 2638 unblocks the small-window approach. NB the 27 MHz Q-branch is
  **one window** at `cap=def` now (legacy K=13, χ²ᵣ=24); the narrow ~1 MHz w197
  it used to split into is gone with the peak cap.
- **But V2 fails the `cap=def` cross-fixture safety check at high SNR.** Pass
  rate and bulk χ²ᵣ *hold* on all seven fixtures, but the **line count balloons**
  (the aggregate metrics are blind to it): 2638 556→592, 1231 593→720, 360
  626→788, **1512 235→589 (2.5×, 10× slower)**. The extras are lineshape-absorber
  peaks chasing the χ²ᵣ ~ SNR² lineshape-fidelity floor; the absolute `Δχ²>2λΔk`
  bar (=30 at λ=5) is trivially cleared when per-bin residuals are tens–hundreds
  of σ. Legacy's *fractional* benefit `n_eff·Δχ²/χ²_K` (with `χ²_K ~ 2M·χ²ᵣ`)
  carried the SNR scaling that suppressed this; the absolute bar dropped it.

### V3 — χ²ᵣ-floor scaling `Δχ² > 2λΔk·max(1, χ²ᵣ_ref)`: **fails oppositely**
Restoring an SNR-scaled bar via the more-complex model's reduced χ² fixes the
high-SNR over-add (a high `χ²ᵣ_ref` lifts the bar and rejects absorbers) but
**re-breaks the 363 recovery** (cap=8 λ5 → K=1 χ²ᵣ=52). **Crux insight:** a dense
under-fit window and a high-SNR isolated line *both* carry a high reduced
chi-squared — the first because lines are **missing** (reducible: wants a *low*
bar), the second because of the irreducible lineshape floor (wants a *high*
bar). Reduced χ² cannot tell them apart, so any bar that is a function of χ²ᵣ
picks one failure or the other. `DEFAULT_GATE_FLOOR_SCALING = False` (the
`ref_reduced_chi2` plumbing stays for experiments).

### V4 — sigma_eff gate χ² (per-bin lineshape-fidelity budget): **works on
bright cores, insufficient alone**
The reducible-vs-irreducible discriminator V3 lacked is **location**: lineshape-
floor residuals sit *under bright model bins*; missing-line residuals sit where
the model is small. The sigma_eff variant computes the penalized gate's χ²
against the inflated per-bin noise ``σ_eff² = σ² + (κ·|model|)²``
(`validation.sigma_eff_chi2`, κ = `DEFAULT_GATE_SIGMA_EFF_KAPPA`, natural value
= the SNR-aware pass metric's κ=0.05; the weights model is the more-complex
model, shared by both sides). The blend-aware seeder's escalation trigger uses
the same σ_eff reduced χ² (a bright line at its floor reads ~1 and stops
escalating; a genuine blend still fires). Measured: kills the bright-core
absorber over-add (1512 w155 monster window: λ5's K=17 → K=9 vs legacy 8),
preserves the 363 cap=8 K=6 recovery and the 2638 control — but barely moves
the bright-band total (109→97), because the dominant over-add is **not under
the local model** (below).

### The dominant 1512 over-add: frozen-skirt fringe error, harvested by the
rescue (measured mechanism)
On the bright band the audit attributes only ~30 of λ5's +150 lines to the
conservative loop; the flood comes from the **residual rescue** in the
*skirt-side windows* (w153/w154, the 17–34 MHz wings of the SNR~7000 line at
37905). The frozen edge-free contributor skirt is subtracted there (the
SSR-fraction arbitration adopts it under both gates — the "arbitration
flipped" hypothesis was **falsified** by direct instrumentation,
`FTMW_DEBUG_SKIRT=1`), but the subtraction leaves a coherent fringe field of
~20–40 % of the subtracted amplitude (χ²ᵣ≈17 over the window; the frozen
(amp, freq, phase) is read at the line and `h_T` is evaluated with the
*dependent* window's τ, so the far-wing extrapolation error is large). The
fringes carry Δχ²~90–270 per feature — genuinely significant against any
honest noise model and **zero matches in the vinyl-cyanide catalogs** — so a
window-invariant local gate correctly-by-its-lights accepts them; legacy
rejected them only by the accident of its 1/M dilution (the same accident
that under-fits 363). Fitting K=51 noise-shaped peaks to a χ²ᵣ(K=1)=0.91
window is also the 10× slowdown.

### V5 — frozen-skirt sigma_eff budget: **fixes the flood; all anchors hold**
The fidelity budget must include the bright structure that was *subtracted*
from the window data (invisible to ``|model|``):
``σ_eff² = σ² + (κ·|model|)² + (κ_skirt·|frozen background|)²`` with
`DEFAULT_GATE_SIGMA_EFF_KAPPA_SKIRT` (κ_skirt≈0.4, the measured fringe/skirt
ratio). The plan executor computes the per-window budget from the subtracted
background and threads it through every gate (add loop, seeder + trigger,
knockout, merge Tier-2, iterative cleanup) and into the rescue. This is a
fixed fidelity constant × a *model* amplitude — not a residual-derived local
noise (the Stage 2 σ stays the only noise authority). Band anchors (λ5 + κ=0.05
+ κ_skirt=0.4): 1512 bright band **109→13** (legacy 11; w154 51→1 at 40× less
wall time), **363 cap=8 K=6 survives** (its dense window carries an edge-free
contributor, so the budget was active and harmless), **2638 control 20**
unchanged. Cross-fixture full fits (vs the issue3 rollup): **pass rate improves
on every completed fixture** — 1512 281 lines / pass 1.000 / bulk 1.34 (legacy
266 / 0.921 / 1.41; λ5 alone 589 and 10× slower), 360 778 / 0.992, 1231
677 / 0.986, 1019 94 / 1.000, 2638 580 / 1.000. κ_skirt sweep: the 1512 fringe
flood needs ≥0.4 (sk0.2→61, sk0.3→23, sk0.4→13) and the 363 dense recovery is
safe through 0.6 — 0.4 is the knee of a wide safe band. Caveats: 363/655 full
fits at cap=def are **prohibitively slow** with the uncapped gate (dense
mega-windows resolve to large K; the add loop and knockout/cleanup sweeps are
O(K²) in NLS solves — the cap=8 small-window retune is the necessary
companion); 360's +22 % line count (with improved pass) is consistent with
real dense-window recoveries but unaudited.

### Side change, measured null: parameter-aware skirt arbitration
The edge-free skirt SSR arbitration (`plan_execution.py`) is k-blind; under a
permissive gate the no-skirt fit can buy SSR with peaks and out-score the
skirt. Under the penalized gate the arbitration now scores both sides in gate
currency (``SSR + 2λk``; the frozen skirt costs zero free parameters); legacy
keeps the calibrated raw-SSR fraction rule. Measured **no behavior change** on
the 1512 band (the skirt was adopted under both gates); kept because it prices
complexity consistently with the gate it serves.

New harnesses for this work live in `scratch/stage5-gate/`: `viz_windows.py`
(per-window cross-gate overlay; gate tokens `legacy|lamN|lamNse[K]sk[S]`),
`strip_metric_gate.py` (gate-parameterised cross-fixture metric; env
`FTMW_GATE_LAMBDA` / `FTMW_GATE_SIGMA_EFF_KAPPA` / `FTMW_GATE_SKIRT_KAPPA`,
plus `FTMW_CAP_MHZ` / `FTMW_ATTACH_THR` / `FTMW_EF_REFINE` / `FTMW_SEED_KO`
for the Stage-4 retune stack), `audit_decisions.py` (per-gate audit-decision
attribution over a band). See `scratch/stage5-gate/HANDOFF.md`.

## Stage-4 small-window retune (lever 1): measured outcomes

The retune at `max_window_width_mhz=8` (≈ 80–100 active-FT points across the
seven fixtures; bin widths 79–100 kHz) under `lam5se0.05sk0.4` exposed two
pre-existing defects that the big-window plan had masked, each fixed behind
its own flag (defaults legacy):

### Edge-free read under-recovery (fixed: `DEFAULT_EDGE_FREE_FREQ_REFINE`)
At cap=8 the 1512 bright band re-flooded (gate 34 vs legacy 23; w285 K=10 on
a smooth pedestal). Instrumentation showed every skirt-side window adopts the
skirt under both gates, but the frozen background predicts only ~1/3 of the
actual pedestal (w285: `med|bg|` 4.7e-6 vs `med|data|` 1.30e-5), so the
κ_skirt budget — keyed to the *predicted* |bg| — was ~1.1σ against a 5.4σ
coherent leftover. The wing physics is sound (the monster's *converged* model
evaluated on w285's grid predicts 1.39e-5) and τ is a non-issue (read τ 8.34
vs fitted 9.15 µs, a ~5% core-height effect): the joint-LSQ read's **fixed
Stage-3 frequencies are sub-bin wrong** (40–60 kHz), which mis-phases the
sharp core template at SNR ~7000 and biases the linear amplitude read low
(core residual 27%; at converged frequencies 4.7% and the wing lands +8%
of truth). The fix is a bounded variable-projection frequency refinement in
`evaluate_edge_free_contributors` (frequencies free within ~1.5 bins, capped
at 0.45× the intra-group separation; amplitudes/phases stay linear). With it
the band total returns to legacy parity (24 vs 23) and all skirt windows
predict their pedestals to ~10%.

### Ungated K=1 seeds (fixed: `DEFAULT_ENFORCE_SEED_KNOCKOUT`)
The remaining cap=8 surplus was one SNR≈2 "dust" line per skirt-side window,
accepted by BOTH gates. Mechanism: the K=1 seed never faces the accept gate
(`_blend_aware_seed` installs it unconditionally), `knockout_test` correctly
marks it unsupported (raw Δχ² 11–22 vs the λ5 bar of 30) — but the verdict
is only a diagnostic: the sweeps that act on knockouts run *inside an
accepted rescue round*, so a quiet window keeps its lone unsupported peak.
More windows ⇒ more ungated seeds; cap=def had the same hole (its "SNR-1
slip-through") at lower exposure. The fix enforces the K=1-vs-null verdict at
`conservative_fit` exit — **in raw penalized currency (no σ_eff/skirt
budget)**: the budget rightly discounts incremental adds chasing subtraction
error, but a window's single dominant feature is routinely a real line riding
the same pedestal, and budget-currency enforcement deleted real lines
wholesale (655 pass 0.924→0.635 measured, reverted). Catalog ground truth on
655: the 241 lines raw-currency enforcement removes have **1 catalog match
(0%) and median SNR 1.3** — noise dust; the kept set matches at 10×. A new
audit decision `"knockout-null"` records the removal.

### Cross-fixture metric, full stack (`GATE=lam5sesk04`, cap=8 MHz, both fixes)
vs legacy (issue3 rollup) and the cap=def gate column:

| fixture | legacy lines/pass/bulk | lam5sesk04 cap=def | cap=8 + fixes |
|---|---|---|---|
| 1512 | 266 / 0.921 / 1.41 | 281 / 1.000 / 1.34 (88 s) | 175 / 0.951 / 1.40 (86 s) |
| 360  | 636 / 0.955 / 1.46 | 778 / 0.992 / 1.31 | 747 / 0.973 / 1.33 (191 s) |
| 1231 | 623 / 0.862 / 1.39 | 677 / 0.986 / 1.26 | 639 / 0.947 / 1.39 (125 s) |
| 1019 | 65 / 0.902 / 1.35 | 94 / 1.000 / 1.28 | 82 / 1.000 / 1.24 (45 s) |
| 2638 | 568 / 0.997 / 1.27 | 580 / 1.000 / 1.26 | 561 / 1.000 / 1.33 (162 s) |
| 363  | 0.731 (legacy) | **killed >105 CPU-min** | **1868 / 0.973 / 1.51 (304 s)** |
| 655  | 0.819 (legacy) | **prohibitively slow** | **1560 / 0.841 / 1.18 (335 s)** |

Every fixture beats legacy; the two dense fixtures that were *unfittable*
under the uncapped gate complete in ~5 min. Where the cap=8 column trails the
cap=def gate column (1512/1231 pass, line counts), the cap=def numbers
benefited from the ungated-seed hole — their extra lines are the same
sub-bar class the 655 catalog A/B measured at 0% truth. Band-anchor checks
all hold at cap=8 (363 K=6, 2638 control, 1512 bright band 11–13 lines with
all 10 catalog lines kept — cleaner than cap=def's 13-with-3-junk). The
contributor relaxation `magnitude_attachment_threshold=1.0` holds all three
anchors but was not adopted (the cap=8 wall times don't need it; revisit
with planning-cost profiling if Stage 4 becomes the bottleneck).

### Points-based cap (productionized, default off)
`max_window_width_points` (settings `clustering.max_window_width_points`,
default 0 = defer to MHz) supersedes the MHz cap when positive — the
portable form, since the gates reason over bins while the active-FT bin
width varies with acquisition length (79–100 kHz across these fixtures;
8 MHz ≡ 80–101 pts). Verified to reproduce the MHz partition exactly
(`points = mhz/step`, 1512: 364 windows identical) and threaded through
settings/api/pipeline/CLI/replan/tuning-registry. A ship-time default near
**96 points** reproduces the validated cap≈8 MHz on this instrument family.
