# Stage 5 doublet alternative fit — sub-resolution pair adjudication

Status: **implemented** (settings-gated observation-only pass, default on:
`fitting/doublet_alternative.py` + executor hook + persistence + `fit check`
section; see "Calibration results" below for the measured interpretation of
the two statistics). Driven by the cross-instrument validation arc: strong
lines are frequently fitted as a strong/weak pair at sub-resolution
separation, where the weak partner may be a genuine second transition or may
merely absorb lineshape-floor error (the SNR² model-fidelity deficit, D10).

## Problem

On the succinimide fixture, 144 fitted pairs sit within 2.5 resolution
elements (1/T_active = 77 kHz bins); only 20 have both members matching
distinct catalog lines. The seven brightest doublets split cleanly: five
(SNR 194–1069, separations 21–51 kHz, amplitude ratios 0.18–0.40) have no
catalog counterpart — the vinyl-fluoride pattern previously seen on the home
instrument — while two (w264 at 39.9 kHz, w611 at 83.4 kHz) are genuinely
catalogued blends. The same spectrum carries true catalog doublets at 13 and
22 kHz separations, so a hard merge rule on separation alone would destroy
real spectroscopy. The existing sub-resolution merge tier
(`stage5-subresolution-overfit` arc: resolution floor k=1.0 + amp-ratio
band) does not cover this regime — these pairs survive it.

A raw-χ² comparison cannot arbitrate: under an SNR ~10²–10³ line, a
sub-percent lineshape/τ deficit is hundreds of σ per bin, so adding the
partner always wins raw χ². The win is real arithmetic but may price only
the model's own shape error.

## Why no independent data domain exists (beat-test result)

The FID-domain "beat" check that adjudicated the w942 blend (0.23–0.56 MHz
splits: 3+ beat cycles across the active window) was tested on the
succinimide doublets and is **under-powered at sub-resolution separations**:
a pair-isolating demodulated envelope of bandwidth B over record length T
carries only ~T·B independent complex samples (~6 here), every 4–8-parameter
envelope model fits, and a known-true catalog doublet misclassifies
(`scratch/doublet-beat/`). At separations below ~1.5 beat cycles
(~115 kHz on a 13 µs record) the time domain is information-equivalent to
the few FT bins the window fit already uses. Coherent frame averaging at
import removes any per-frame angle, and both hypotheses are deterministic
and frame-stable anyway. Conclusion: discrimination must come from model
structure on the window bins, not from a new data domain. The beat view
remains valid as a *visual* check for splits ≳150 kHz.

## Design

A post-fit, per-window adjudication pass over the final fitted table. No
acceptance behavior changes — the pass *records* an alternative and two
statistics; consumption (annotation, user adjudication, report surfacing)
belongs to the reports / user-interaction project.

### Trigger

Every fitted pair in a window's final table with separation
`<= k_res * (1/T_active)` (default `k_res = 1.5`, knob) and weak/strong
amplitude ratio `>= r_min` (default 0.05, knob — below that the partner is
skirt-level and the existing cleanup owns it). Chains (3+ peaks pairwise
within the floor) adjudicate pairwise on adjacent members.

### Alternative fit

Refit the window with the pair collapsed to a single peak (seeded at the
amplitude-weighted centroid, all other window peaks and the frozen
background unchanged; same masks, same baseline policy, same τ treatment as
the production fit). Record the merged fit's parameter set alongside the
production doublet fit.

### Statistics recorded per pair

1. **Fidelity-floor flag** — `eps_single = sqrt(max(chi2r_merged - F, 0)) /
   SNR_max` with the D10 constants (`F = 3.0`, κ = 0.05 as the comparison
   bar). If `eps_single <= kappa`, the merged fit is within the lineshape
   floor: the doublet's raw-χ² win bought nothing the known shape deficit
   does not explain → flag `doublet_not_required`. Conversely
   `eps_single > kappa` with `eps_doublet <= kappa` is positive evidence the
   second component is load-bearing.
2. **Orthogonal-evidence score** — project the merged fit's residual onto
   the partner template after projecting out the parent's shape-derivative
   subspace `{h, dh/df, dh/dtau}` (the nuisance-projection machinery of the
   line-evidence escape, reused with the parent's modes as nuisance
   columns). First-order lineshape/τ error lives in that subspace; a genuine
   second transition retains an orthogonal component. Report the score in σ
   units; its acceptance threshold is a calibration output, not a design
   input.
3. Bookkeeping: separation in resolution elements, amplitude ratio, raw
   Δχ² between the two fits, ΔAICc.

### Persistence

The alternative fit and statistics persist with the window's audit records
in `/stage5_fitting` (same pattern as knockout/rescue audit payloads), so
`fit show`, future reports, and the candidate-revival ledger can consume
them without refitting. Loaders tolerate their absence (older files).

### Calibration results

Measured on the post-D14 builds (`scratch/doublet-calibration/calibrate.py`;
1512 = vinyl-cyanide truth, 45 adjudicated pairs of which 30 are true
catalog doublets; succinimide = 97 pairs against the extrapolated catalog,
where "no catalog match" includes real uncatalogued cluster lines):

- **`eps_single > kappa` is a high-precision "doublet required" verdict.**
  On 1512 all 10 pairs exceeding the fidelity floor after merging are
  catalog-true (10/10); on succinimide it fires on the catalogued blend
  w611. It fires on a minority of true doublets (10/30 on 1512) — the
  fidelity-floor allowance at high SNR absorbs even genuine partners — so
  its complement (`doublet_not_required`) must be read as "χ² cannot
  adjudicate", never as "spurious".
- **The orthogonal evidence is a fraction, not an absolute.** The
  recorded `orth_evidence_delta_chi2` scales with partner SNR²; the
  cross-pair currency is `orth_frac = delta_chi2 / psnr²` (partner SNR
  from the production table). Verdict zones: *real* when `orth_frac`
  clearly exceeds its χ²₃ noise null (`~3/psnr²`, so partner SNR ≳ 15 is
  needed for any power) — e.g. the catalogued 16914 blend reads 39%;
  *parent-shaped* when the fraction is tiny at high partner SNR — the
  five catalog-unsupported succinimide bright doublets read 0.003–4%
  (the SNR-1069 12321 pair: 0.03%), confirming the lineshape-error
  reading; *uninformative* otherwise.
- **Two measured degradation regimes** (both honest physics, recorded for
  consumers): below ~0.5 resolution elements the partner template is
  near-representable by the parent + derivatives, so `orth_frac` fades
  regardless of truth (catalog-true pairs at 0.2–0.5 elements read
  ≤ 1%); and in many-peak windows the nuisance basis (3 columns per
  merged-model peak + background, truncated at `m − 2` on the support
  slice) can nearly span the support, suppressing real partners
  (1512's w227/w277/w289 true doublets read ~0 despite partner SNR
  10²–10³ — `support_bins` together with the merged-model peak count
  flags the regime).
- **Cost**: succinimide end-to-end fit 111 s → 116 s with 97
  adjudications (~4%), within the one-refit-per-pair budget.

The reports / user-interaction layer should therefore present, per pair:
the `eps` pair, `orth_frac` with its noise null, and the degradation-regime
markers — and treat catalog cross-match (where available) as the decisive
external arbiter, as it was in this calibration.

### Calibration plan

- **1512 (vinyl cyanide truth)**: enumerate all triggered pairs in the
  post-D14 build; classify each against the truth list (both members
  distinct truth lines vs not). Measures the true/false discrimination of
  `eps_single` and the orthogonal-evidence score, and sets the score
  threshold empirically.
- **Succinimide**: the 7 bright doublets (5 catalog-unsupported, 2
  supported) as the cross-instrument check; the catalog's own close pairs
  (13–22 kHz) as must-not-flag cases at the trigger boundary.
- **655 / 363**: regression — the pass must be observation-only (zero table
  changes) and its runtime cost bounded (one extra constrained refit per
  triggered pair; triggered pairs are rare outside dense fixtures).

### Out of scope (deliberately)

- Automatic merging or any acceptance change driven by these statistics.
- Presentation/UX: which windows get surfaced for user attention, how
  adjudication decisions (add/remove lines, accept a merge) are captured
  and logged, and where those decisions live between Stage 5 and report
  artifacts — that is the reports / user-interaction project
  (ROADMAP priorities 4–5), which this feature feeds but does not depend
  on. The candidate-revival ledger
  ([`stage5-candidate-revival.md`](stage5-candidate-revival.md)) is the
  natural carrier for the flip operation.

## Interface surface

Internal to `fit run` (a settings-gated pass, default on, observation-only;
`StageFitSettings` gains a `doublet_alternative` sub-block: `enabled`,
`k_res`, `r_min`). No new CLI verbs; `fit check` gains the per-pair
statistics in its output, and `fit show`'s per-window detail may later
annotate the alternative once the presentation questions are settled.

## Test plan

- Unit: trigger enumeration (separation/ratio/chain cases), merged-refit
  parameter handling (frozen background, masks, τ policy), statistics
  computation against synthetic windows with (a) a true doublet, (b) a
  single line with injected τ error, (c) a single line with injected shape
  asymmetry — (b) and (c) must flag `doublet_not_required` and low
  orthogonal evidence; (a) the reverse.
- Integration: cross-interface consistency (the pass runs identically via
  CLI/Pipeline/API); persistence round-trip; observation-only invariant
  (fitted table byte-identical with the pass on vs off).
- Calibration artifacts under `scratch/` per the calibration plan, reported
  back into this document on completion.
