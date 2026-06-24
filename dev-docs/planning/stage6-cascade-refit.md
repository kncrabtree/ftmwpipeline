# Stage 6 — contributor-edit cascade (window-level dependency resolution)

Status: **prerequisite investigation complete; production prototype on branch
`stage6-cascade-refit`.** The investigation (recorded below) resolved the
dependency-model question and reshaped the cascade design. Headline conclusions:
the dependency treatment is **moot at moderate SNR for parameter edits** (≤0.02σ;
the order-4 baseline and a contributor are interchangeable substitutes for a
smooth skirt) but **material at extreme SNR and for structural edits** (near a
multi-line ~1e6 source, and when curating an overfit). The cycle-break should be
driven by **skirt curvature (~1/Δf³), not level** (which drives the spurious
fully-cyclic map); that orients edges strong→weak (validated 100–300×
asymmetry, acyclic), with **thaw** for close strong–strong seed pairs.
Edge-bearing propagation is only safe on a **curated** source — an overfit's
anti-phase skirt cancels in the fit but not in the data, so naive edge-bearing
propagates the artifact (the overfit-skirt trap). See the findings sections for
the evidence; the prototype scope is in **"Production prototype"** below. The
normative requirements remain in the `*_STRATEGY.md` specs; this plan is
normative only for the work it tracks.

Related: [`stage6-finalization.md`](stage6-finalization.md) (the read/edit
surface this builds on), [`stage6-peak-survival.md`](stage6-peak-survival.md)
and [`stage4-leakage-contributor-subtraction.md`](stage4-leakage-contributor-subtraction.md)
(the contributor / frozen-skirt machinery), and the documented limitation in
`docs/source/stage6_review.rst` ("No automatic cascade").

## Production prototype (branch `stage6-cascade-refit`)

The validated design, in dependency order. Each piece is to be built as a real
`_internal` change behind the dual-interface invariant, validated against the
7-fixture baseline, *not* a scratch harness — but assessed incrementally, with
the scratch harnesses (below) as the measurement instruments.

**A. Overfit detection + merge-veto fix (curation correctness; the prerequisite
for safe edge-bearing).** w1006's anti-phase degenerate doublet should have
auto-merged but did not — it slipped past both the canceling-phase degenerate
penalty (`window_fit.py:143/1943/2131/2355`) and the structural tier of
`merge_close_peaks_cleanup` (`residual_rescue.py:216`, 3 tiers: structural /
AICc-gated / amplitude-ratio; `_merge_cluster` collapses via the complex sum
`|Σ Aⱼe^{iφⱼ}|`, which for an anti-phase pair is *small*). First task: find out
*why* (the χ²ᵣ-jump veto reads a single-line refit of an anti-phase pair as
catastrophic even though the pair is degenerate). New signals to fold into the
decision tree:
  - **Degeneracy-gated anti-phase metric.** Sub-resolution separation (high
    amplitude VIF, `validation.py:1649`) **AND** Δφ≈π **AND** comparable
    amplitude ⇒ canceling overfit, merge unconditionally. *Guard
    ([[no-frequency-phase-relationship]]):* phase value/difference is meaningless
    for *resolved* doublets — the anti-phase signal is diagnostic **only** when
    gated on degeneracy (co-location), never on its own.
  - **Edge-bearing vs edge-free skirt disagreement** as an independent overfit
    detector: when a source's *fitted* (edge-bearing) skirt into a dependent
    cancels but its *data* (edge-free) skirt does not, the source fit carries
    structure the data lacks → overfit flag. Self-gating and physically grounded.

**B. Curvature-driven cycle break (Stage-4 window planning).** Replace the blunt
"drop all cyclic edges → demote to edge-free" with: seed from the strongest
windows, fan out, orient each edge strong→weak, and keep an edge as a real
(edge-bearing, on a *curated* source) contributor only when its *curvature
residual* `S_resid(p)` (the part an order-p baseline cannot absorb — see the
findings) is significant; otherwise drop it to the baseline. The level/curvature
asymmetry makes the oriented graph acyclic. Resolve the **joint-arbitration
entanglement** (flipping one edge re-arbitrates the others — the w1005 caveat).

**C. Thaw for close strong–strong seed pairs.** The one residual cycle the
orientation cannot resolve (peer seeds, neither upstream) is exactly thaw's job;
reactivate it there (it is dormant today only because everything is edge-free).

**D. Cascade.** With a meaningful edge-bearing graph on curated sources, a Stage 6
edit recomputes the transitive descendant closure in topological order (the
design in **"The decided design"** below). Structural edits (overfit merge on a
strong contributor) are where it has real bite (~20σ skirt change at w1006).

Open the assessment with **A** (self-contained, testable on w1006 immediately,
and the prerequisite that makes edge-bearing safe), then **B**.

### Step A diagnosis: why w1006's anti-phase overfit survives (root cause)

Traced on the fresh 655 w1006 (`scratch/cascade/merge_diag.py`,
`_mergetrace.py`, `_mergesum.py`). The merge machinery *does* find the pair —
`merge_close_peaks_cleanup` selects it as the closest pair (Δ=0.3 kHz, Tier 1,
`cancel=0.99`), and the blend-escape is correctly *blocked* (cancellation 0.97 ≥
`DEFAULT_PAIR_CANCELLATION_MAX`=0.75). Yet `n_merged=0`. The cause is the Tier-1
`(K−1)` refit and the `if not refit.success: break` guard:

- `_merge_cluster` seeds the merged amplitude as the **phasor sum**
  `|Σ Aⱼe^{iφⱼ}|` — for this anti-phase pair (0.44∠−1.35 + 0.44∠+1.76) that is
  **~0.012**, a near-degenerate seed. Setting it to the **magnitude sum** (the
  natural fix for the seed) still does not merge.
- The deeper reason: a **single Lorentzian at this position is χ²ᵣ ~3×10⁶**
  *regardless of seed* — the irreducible lineshape floor of an SNR-10⁶ line. The
  merged `(K−1)` refit converges *alone* (K=1) but **fails to converge with any
  companion peaks present** (K≥2 → `success=False`) against that astronomical
  floor residual, so the `break` fires and the merge is abandoned. The anti-phase
  pair (χ²ᵣ 525) is the fitter's least-bad absorption of the line's *shape
  distortion*, not two real lines.

**Conclusion:** the merge veto is *confounded by the lineshape floor* for
ultra-strong lines — any χ²- or refit-success-based gate blocks collapsing a
strong line's overfit, because the post-merge fit always looks catastrophic. The
floor is present for 1 *or* 2 lines and is **never** evidence for the second.

**Fix (two fronts):**
1. *Downstream (merge):* in Tier 1, a confirmed **canceling** pair (cancellation
   ≥ cmax) is an artifact by construction — merge it **regardless of the merged
   refit's χ²/success** (do not `break`); seed robustly (magnitude sum, drop
   near-zero companions). Degeneracy is the authority, not the floor-dominated χ².
2. *Upstream (prevent):* the canceling-phase penalty
   (`window_fit.py:143/1943/2131/2355`) was meant to stop this pair forming and
   did not — likely its weight is swamped by the 10⁶-scale floor residual, or the
   pair is created in a rescue/blend-split path the penalty does not cover.
   Preventing formation is cleaner (no artifact to detect); the merge fix is the
   safety net. New harnesses: `merge_diag.py`, plus `_mergetrace.py` /
   `_mergesum.py` (monkeypatch traces of the Tier-1 decision).

### Step A, generalized: SNR-aware (floor-aware) acceptance criteria

The w1006 overfit is a symptom of a general defect the user flagged: **peak
acceptance ultimately rests on a raw χ²ᵣ decrease, while the lineshape floor is a
real, accept-able source of χ²ᵣ loss** that no peak should be rewarded for
absorbing. (Prior n_eff-based weightings were tried and never conditioned the
tests sensibly.) So the upstream re-examination of penalties and acceptance
criteria is the *first* prototype workstream, ahead of the merge fix.

**Empirical floor (`scratch/cascade/floor_scan.py`, npk=1 windows):** χ²ᵣ ≈ 1 up
to SNR ~200–300, then climbs as **(κ·SNR)²** with **κ ~ 1.7–7×10⁻³** (varies a
few× by line — it is a real shape error, not universal). Decisive confirmation of
the mechanism: among the worst windows the persisted χ²ᵣ is *anti*-correlated
with SNR — w1006 (SNR 1e6, **6 peaks**) χ²ᵣ=385 vs w1102 (SNR 44k, **4 peaks**)
χ²ᵣ=**260 913** — the higher-SNR line has the lower χ²ᵣ because it was split into
more peaks that absorb the floor. **Raw-χ² acceptance rewards floor-absorbing
overfits.**

**The σ_eff floor model already exists** — this is not a new concept to invent.
`validation.sigma_eff_chi2` scores against

> σ_eff(bin)² = σ(bin)² + (κ·|model(bin)|)²,  κ = `DEFAULT_GATE_SIGMA_EFF_KAPPA` = 0.05

(plus κ_skirt=0.4 for frozen backgrounds), and the SNR-aware window-quality
criterion χ²ᵣ ≤ F + (κ·SNR_max)² is the documented gate (methods note
`methods/stage5_fitting.rst`, fig2_sigma_eff_localization.png; the
[[stage5-context-invariant-gate]] work). It is already applied in **both** the
add-one-peak gate **and** the blend-straddle trigger (`window_fit.py:1983` —
`_trigger_rchi2` uses `sigma_eff_chi2`). The measured floor κ ~ 1.7–7×10⁻³ is
*below* the deployed 0.05, so the budget is conservative — **calibration is not
the gap; coverage is.** (A global SNR^(−1/2) on χ²ᵣ would not flatten an SNR²
floor anyway — σ_eff is the right localized form.)

**So the real Step-A-general question:** since the two main entry gates are
floor-aware, w1006's canceling pair entered through a **floor-blind path**. The
work is to *complete the coverage*, not introduce the concept:

1. *Trace w1006's entry path* — candidates: Stage-3 double-detection at the giant
   core; the residual-reseed **nomination** (`window_fit.py:2099`, fires at raw
   `4·σ` — floor-blind, though its acceptance is floor-aware); `doublet_alternative`;
   an NLS canceling basin the anti-cancellation penalty (`window_fit.py:143/…`)
   failed to suppress.
2. *Audit* every χ²ᵣ / Δχ² site for raw-σ vs σ_eff: classify add-gate (σ_eff ✓),
   blend trigger (σ_eff ✓), window accept (σ_eff ✓), blend-escape `evidence_floor`
   (floor-scaled ✓) vs `knockout_test`, rescue-accept, the merge tiers, the
   residual-reseed nomination (raw σ — to fix).
3. *Extend* σ_eff to the floor-blind sites found.
4. *Fix the floor-blind removal*: the Tier-1 merge of a confirmed canceling pair
   must not be vetoed by the floor-dominated `(K−1)` refit (the `refit.success`
   break) — degeneracy is the authority. (Front 1 above.)
5. *Validate*: w1006/w1102/w79 overfits no longer formed/accepted; real
   sub-resolution multiplets (1512 ¹⁴N hyperfine, 363/360 methyl A/E — see
   [[fixture-splitting-physics]]) still pass; 7-fixture re-baseline.

Empirical aid: `scratch/cascade/floor_scan.py` (χ²ᵣ-vs-SNR floor, κ recovery)
demonstrated the symptom — raw χ²ᵣ is *anti*-correlated with SNR among the worst
windows (w1006 6-peak χ²ᵣ=385 < w1102 4-peak χ²ᵣ=260 913) because more peaks
absorb more floor: raw-χ² acceptance rewards floor-absorbing overfits wherever a
gate is not yet floor-aware.

### Step A SOURCE FOUND: σ_eff under-covers a bright line's own far skirt

Traced the actual entry path (`scratch/cascade/_entry.py` audit trail,
`_trigger.py` κ sweep). w1006's canceling pair is created by the **conservative
seeder's blend-straddle escalation**: from a single Stage-3 candidate at the
giant core it escalates K=1→2→3 (`seed-blend`, aicc_δ −816 / −458), and the joint
NLS collapses the straddle into the anti-phase pair. Crucially it is **not** a
floor-blind path — the escalation *trigger* (`_trigger_rchi2`) and the
acceptance AICc gate **both** carry σ_eff. The hole is in σ_eff's **coverage**:

- The giant's K=1 σ_eff-χ²ᵣ = **4.56 > 1.5** (the seeder trigger) at κ=0.05; a κ
  sweep closes it at **κ=0.10** (σ_eff-χ²ᵣ=1.14). So the gap is ~2× the core κ.
- **83% of the σ_eff residual is at 3+ FWHM from the line** (8% in the core). The
  lineshape floor concentrates in the **far skirt**, where the true-shape-vs-`h_T`
  deviation is fractionally ~10% — but σ_eff there is κ·|model| with |model|
  small, so the core κ=0.05 under-budgets it.
- A skirt-fidelity budget already exists — **κ_skirt = `DEFAULT_GATE_SIGMA_EFF_KAPPA_SKIRT`
  = 0.4** — but it is wired only to the *frozen out-of-window contributor*
  background (`gate_background`), **not** a window's own bright in-window lines'
  skirts. That is the hole: an in-window bright line's own far skirt falls through
  to the too-small core κ.

**The plug (source, not bucket):** extend the skirt-region fidelity budget to a
window's *own* bright in-window lines' far skirts (the same κ_skirt concept). It
raises σ_eff in the far-skirt bins so the floor is covered, the seeder stops
escalating on it, and the σ_eff-AICc gate stops accepting the straddle — threaded
uniformly through every σ_eff gate (trigger, add, merge) via the weights, with
the **core κ=0.05 unchanged** so real close blends (which live within ~1 FWHM)
are unaffected; the skirt budget only bites at 3+ FWHM, where a real line must
clear the bright line's skirt floor regardless. Validate on the 7-fixture
baseline (w1006/w1102/w79 no longer over-split; 1512/363/360 real multiplets
preserved). New harnesses: `_entry.py` (conservative_fit audit trail),
`_trigger.py` (σ_eff-χ²ᵣ vs κ + far-skirt residual localization).

**Approach (a) — implemented + validated (not committed).** `window_fit.py`:
`in_window_skirt_budget()` (κ_skirt=0.4 on bright in-window lines' core-masked
far skirts; consts `DEFAULT_IN_WINDOW_SKIRT_KAPPA`=0.4 / `_MIN_SNR`=100 /
`_CORE_FWHM`=1.0) + `_combine_budget()` quadrature, threaded into the seeder's
`_gate_budget()` → trigger + escalation. Full-pipeline 655 refit: w1006's
canceling doublet **eliminated** (seeder emits K=1 only; snr 987k overfit → one
honest line snr 48645, npk 6→3, χ²ᵣ at the honest floor within the (κ·SNR)²
window allowance); w124 also de-overfit; **resolved** real pairs (w1102/184/918,
154–220 kHz apart) and all neighbors unchanged; net −11 lines on 655; 469 fitting
unit tests pass; mypy clean. **7-fixture A/B re-baseline** (`scratch/cascade/batch_ab.py`, env
`FTMW_NO_IN_WINDOW_SKIRT` toggles the arms; reports under `ab/report_<name>/`).
At the initial `min_snr=100`: real multiplets preserved (1512 ¹⁴N hyperfine +0,
360 methyl +0, 1231 +0), canceling pairs 655 1→0, deltas tiny (1019 +1, 363 −3,
2638 +1, 655 −11). **But the comparison surfaced a regression**: 363 w357 (two
*real resolved* lines, snr 76/140, 194 kHz apart, healthy χ²ᵣ=0.65) was emptied —
`min_snr=100` applied the κ_skirt budget to the snr-140 line, inflating σ_eff in
its near-skirt where the snr-76 neighbor (≈2.3 FWHM, just past the 1-FWHM core
mask) sits, killing both. **Recalibrated `DEFAULT_IN_WINDOW_SKIRT_MIN_SNR` →
500** (verified: w357 restored, w1006 still collapsed, 363 back to its 1731-line
pre-fix baseline, 655 −6). The clear canceling overfits are all snr > 500 (w1006
48645, w124 18628, w905 1140, w892 505); the snr 100–500 changes were collateral
on benign-floor lines. So min_snr=500 catches the real overfits and spares
moderate lines. **chi2_eff reassurance stat already exists**: ε =
`shape_error_fraction` = √(max(χ²ᵣ−F,0))/SNR_max, computed per window and shown in
the report's SNR-aware-gate section (`report_impl.py:1083`); w1006's χ²ᵣ=53936 is
ε≈0.5% with a pass. **Approach (b)** (per-bin blended-κ; `mode='b'`, env
`FTMW_SKIRT_MODE=b`) prototyped: it tracked **(a)** *identically* on all 7
fixtures' peak counts → (a) favored (simpler).

**κ RE-CALIBRATION (the decisive fix; user-caught regression).** The
`DEFAULT_IN_WINDOW_SKIRT_KAPPA`=0.4 above was wrong: 0.4 is the *frozen
out-of-window contributor* value (its skirt is subtracted from the data and
carries 10–40% extrapolation error), whereas an **in-window** bright line's own
skirt is fit jointly and its only error is the `h_T`-vs-true-shape lineshape
floor (~5%). At 0.4 the budget **buried a real snr-1535 line** (37903.57) riding
the 1e6-SNR giant's skirt in 655 w1006. **Lowered to κ=0.10** (the in-window
skirt floor): recovers that line *and* still collapses the canceling overfit.
Validated three ways — (1) 7-fixture A/B @κ=0.10: 6/7 essentially unchanged, 655
the only meaningful change, **no real-line loss**; (2) **catalog** (655 = 1512's
molecule, vinyl cyanide): every κ=0.10 fitted line matches a catalog line
(37903.6 / 37904.8 / 37904.9 / 37906.5), and the 37904.0 nomination matches
*nothing*; (3) **Blackman-Harris apodized** spectrum (leakage-suppressed,
catalog-independent — covers a possible missing isotopologue): 37903.6 / 37906.5
survive as clean peaks, 37904.0 shows only the giant's residual near-wing. So
37903.57 is real (recovered) and 37904.0 is skirt floor (correctly rejected).
**Lesson: read the data/catalog, do not rationalize a χ²ᵣ residual as "floor."**
`DEFAULT_IN_WINDOW_SKIRT_MIN_SNR`=500, `_CORE_FWHM`=1.0 unchanged. Harnesses:
`log_view.py` (log-mag data/model/residual), `bh_view.py` (BH-apodized overlay),
`validate_k.py` (per-window lost-line hunt at the current κ).

### Step A, the overfit is unresolved hyperfine → VIF-collapse consolidation

The user audited more 655 windows past w1006 and found the same pathology is
pervasive and has a deeper cause. In a set of windows (655 w124 / w186 / w187 /
w769 / w892 / w905) each strong line is fit as a **sub-resolution anti-phase
amplitude-VIF doublet** (and the tallest line in w124 as a **quartet**). These
are **not noise sculpting** — they are the prior-free fitter representing **real
unresolved hyperfine**. Catalog cross-reference (655 = vinyl cyanide,
`dev-docs/fixtures/1512-vinyl-cyanide-truth/catalogs/*.cat`,
`combined_lines.csv`) shows every flagged window sits on a real catalog
hyperfine multiplet; the fitted doublets are co-located with the catalog
components but their splittings/amplitude ratios disagree with the catalog by
several σ — the data cannot honestly constrain the multiplicity.

**The decisive reframe (do not relitigate):** at the calibrated τ a single
Lorentzian per feature leaves a *high* χ²ᵣ (655 w187 K=3 → χ²ᵣ≈525; w124 → ~1e4;
fitting K=3 at τ=τ_maj gives χ²ᵣ≈7400 vs the doublets' ~22). The data genuinely
supports more structure, so **no χ²-based gate (raw or σ_eff) prefers the honest
single line** — verified by direct experiment (a σ_eff-aware Tier-1 merge escape
collapsed the pair but the round-install raw-χ² gate reverted it, and the
conservative K=5 is itself an overfit). Removing the doublets is therefore a
**prior**, not a fit-quality decision: *sub-resolution + anti-phase +
amplitude-VIF-degenerate ⇒ unconstrained multiplicity ⇒ report one centroid
line*. The honest output is one line per resolvable feature, carrying (1) an
**inflated effective frequency uncertainty** ≈ the multiplet spread, and (2) the
high χ²ᵣ surfaced as `shape_error_fraction` ε (≈1 % even where χ²ᵣ≈1e5), **not**
raw χ²ᵣ. ε already exists (`validation.shape_error_fraction`,
`report_impl.py`); the "effective fit error that doesn't look catastrophic" the
program wants is ε plus the spread-inflated σ_f.

**The machinery already existed** — `apply_vif_collapse`
(`stage5_impl.py`, the final Stage-5 pass *after* discovery/rescue/prune; flow:
`execute_plan` → `annotate_lattice` → survival prune → **VIF collapse** → window
cleanup → persist). It selects degenerate sub-resolution pairs (amplitude VIF >
`vif_collapse_threshold`=4 within `collapse_max_separation_res`=1.0 res) and
merges them, overriding χ²/AICc. The single blocker was the **catastrophic-merge
veto** `merge_chi2_veto`=100: it reverted exactly the high-χ²ᵣ unresolved-
hyperfine collapses we want. The pass *had already tried and reverted* all six
flagged windows.

**Fixes (in code on the branch, uncommitted):**

1. **Separation-gate the veto** — new knob
   `peak_survival.merge_chi2_veto_min_separation_res` (default **0.5**). The veto
   no longer applies when *every* collapsing pair in a window is below 0.5 res:
   an unresolvable pair (every calibration-truth doublet sits ≥ 0.82 res) cannot
   be two resolvable lines, so its high post-merge χ²ᵣ is the irreducible
   unresolved-structure floor, force-collapse it. The veto still guards the
   marginally-resolvable 0.5–1.0 res band, where the real-doublet protection
   lives (1512 w228 @ 0.727 res, post-merge χ²ᵣ 4939 — **preserved**; 1512
   w21/w192 sub-0.5-res hyperfine — collapsed).
2. **Effective σ_f for collapsed lines** — `frequency_error` inflated to
   `sqrt(formal² + spread²)`, spread = amplitude-weighted RMS of the merged
   component frequencies (a clean single line has spread 0 → unchanged). Recorded
   in the `vif_collapse` diagnostic (`unresolved_spread_mhz`); note
   `FittedPeak.extra_errors` does **not** serialize, so the persisted carrier is
   `frequency_error` itself plus the diagnostic.

**The deeper mechanism bug (user-caught on w905): the collapse REFIT was not
surgical.** `refit_window_core` re-fits the **whole window all-free** after a
merge, so a single collapse re-optimizes *every* peak. At high SNR the lineshape
floor makes that unstable: a real shoulder drifts to the window edge, the merged
line and a neighbour collapse together, and a 1e5 line clobbers a weak merged
line — w905 went 6→4 (corrupted, χ²ᵣ 3→409) and w124 *lost a real line*
(27766.43). Durable lessons:

- **The merge seed matters.** The magnitude/phasor-sum merged seed is too strong
  for an anti-phase pair, whose *true net is weak*; over-seeding drops the NLS
  into the bad basin. (Experiment `scratch/cascade/exp_w905.py`: all-free from
  the magnitude-sum seed → χ²ᵣ 396 collapsed; converge the merged line *alone*
  first → it finds the weak net → relax from that seed → χ²ᵣ 18, peaks stay
  separated. **The faithful standalone refit needs `baseline_order` set** — w905
  carries an order-4 leakage baseline; without it τ collapses and χ²ᵣ is
  garbage, the trap that made the first experiment runs unfaithful.)
- **Multiple simultaneous merges + an outer iteration drift weak lines into a
  dominant line.** The old flow found *all* pairs in a window and merged them in
  *one* all-free refit, outer-iterating up to 5×; a quartet only resolved across
  outer iterations, and the inter-iteration relaxation walked w124's weak merged
  line into the giant.
- **Fix (Part 3, sequential):** merge **one** pair → **frozen intermediate
  refit** (only the new merged line free; *every* inherited peak pinned, so a
  dominant line can never clobber a weak one mid-sequence) → repeat until no
  degenerate pair → **one final all-free refit** for honest covariance, seeded
  from the fully-converged frozen state. A quartet folds 4→3→2→1 within the
  window pass; the outer iteration is gone. `refit_window_core` gained a
  `freeze_inherited` mode; `apply_vif_collapse` drives the sequence.
- **IMPLEMENTATION TRAP (cost a debug cycle):** the frozen refit reused the
  thawed-line hold-out path (freeze in background + re-append verbatim). For a
  window's **own** inherited peaks that pollutes `fixed_parameters` — they
  persist as *contributors*, so the next refit subtracts them as background
  **and** fits them as peaks (double-count); w187 grew its own 3 peaks into its
  contributor list, χ²ᵣ 525→13760. Fix: **restore the original
  `fixed_parameters` after a frozen refit** (the held peaks live only in
  `fitted_peaks`). After the fix: w124 **10→4 with all four catalog lines**
  (27766.43 recovered), w905 6→5 (χ²ᵣ 18), w187 6→3 (χ²ᵣ 525), w186 8→5.

**Sequential collapse validated (655 trace, `init_K/frozen_K/relaxed_K`):**
w124 10→(6 merges)→4, w187 6→3, w905 6→5, w769 **11**→(2 merges)→9. **No window
grows** (`relaxed_K` never exceeds `init_K`) — the collapse never adds peaks and
the **final all-free relax never walks**. The apparent "w769 6→9 regression" was
a *misread*: the fresh fit has 11 peaks pre-collapse; the new code merges only
the 2 genuinely-degenerate pairs (→9), while the old all-at-once + outer-
iteration code drifted weak peaks together and over-merged to 6. Less aggressive
here is *more* correct.

**Trap that cost a cycle:** a debug `os.environ.get(...)` print in
`apply_vif_collapse` referenced `os`, which `stage5_impl` did not import →
`NameError` crashed every fit, leaving the copied-from-`ab` file unchanged and
mimicking "stale code / uncollapsed result". `import os` added; the print must be
**removed before commit**.

**Open at handoff:** (a) **Full 7-fixture audit** with the sequential code
(`scratch/cascade/refit_after.py` then `audit.py`); the last `audit/audit.txt` is
from the superseded per-call-2-step state. (b) **Re-run targeted suites** after
the rewrite (only the veto-gate stage was tested). (c) **Remove the
`FTMW_DEBUG_COLLAPSE` print** + decide whether to keep `import os`. (d) Reserve,
only if a future fixture's final relax walks: a **locality / parameter-
correlation-block-gated final relax** (thaw only the merged line's correlation
block, freeze distant strong lines); compute the block on the *pre-collapse* fit
(post-collapse it is degenerate). Not needed for the current 7 fixtures.

## Prerequisite finding: the dependency model is inconsistent across SNR

Characterizing the contributor / dependency structure of fresh and recent
fixtures (`scratch/cascade/characterize.py`) surfaced a latent inconsistency
that must be resolved before a cascade is meaningful. A dependent window only
goes stale on a *parameter* (amplitude/phase) edit when its contributor is
**edge-bearing** (reads the primary's *fitted* skirt). An **edge-free**
contributor reads (amp, phase) self-contained from the active FT (data), so it
is immune to a fit edit and self-consistent with the data by construction; a
**thaw** co-fit would hold a joint-optimum copy that does go stale.

Measured structure (windows / dependency_edges / edge-bearing / edge-free):

| Fixture | windows | edges | edge-bearing | edge-free |
|---|---|---|---|---|
| 360  | 331  | 114 | 234 | 1097 |
| 2638 | 375  | 73  | 130 | 265  |
| 363  | 486  | 62  | 119 | 1556 |
| 1231 | 305  | 63  | 115 | 798  |
| 1019 | 88   | 0   | 0   | 315  |
| 1512 | 267  | 0   | 0   | 962  |
| 655  | 1241 | 0   | 0   | 8372 |

The three **extreme-SNR** fixtures (655, 1512, 1019) — which by density have the
*most* contributor relationships — end up with **zero** surviving dependency
edges: every contributor is edge-free. On the fresh 655, the cycle-breaker
(`window_planning.py` Step 7) **dropped 9 523 cyclic dependency edges** and
converted the dominant orphaned contributors per window to edge-free (8 063
total), dropping the rest. So the densest spectra are fit as **independent
windows**, while the moderate fixtures (2638/360/363/1231) retain the
edge-bearing fit-ordering machinery. That is the inconsistency: the cases with
the heaviest coupling get the *least* sophisticated treatment.

**Thaw is dormant.** Thaw (the co-fit handshake intended to resolve coupling) is
gated to edge-bearing contributors only (`thawable = [fp for fp in fixed_peaks
if not fp.edge_free]`, `plan_execution.py:1373`), so once the cycle-breaker has
made everything edge-free there is nothing to thaw. On the fresh 655, all 132
thaw attempts reject with `reason='no frozen contributor on the flagged edge
side'`; thaw-accept is **0 on every fixture checked** (655, 2638, 360). Thaw may
be effectively dead code in production.

**Hypothesis (to test):** the cycle-breaker is too aggressive — dropping all
cyclic edges and demoting to edge-free pushes leakage-skirt modeling onto the
complex/leakage-wing **baseline**, which on the dense fixtures runs at order 4
in ~99% of windows and may be modeling leakage skirts that contributors should
carry. (The earlier visual impression that 655 was "deeply pathological" was a
display bug — the magnitude panels plotted a misaligned padded FT that erased
narrow lines — now fixed and committed. With a trustworthy display, 655 reads
healthy in bulk; the open question is the baseline-vs-contributor one, not a
broken fit.)

## Investigation plan

Done so far: fresh 655/1512/1019 built and reported; contributor structure
characterized (the SNR table above); the display bug found and fixed; the
faithful skirt A/B prototyped. The dependency-model question itself is **open**
and is the next session's work.

1. **Baseline-magnitude diagnostic (build this).** When a contributor skirt is
   modeled physically, the leakage-wing baseline should be a *small correction*,
   not a data-scale term. So compare each window's fitted baseline magnitude
   (and its slope) against a robust estimate of the average and slope of the
   real/imaginary data — a baseline whose magnitude is the same order as the data
   marks a **leakage-touched window that should be coupled** to a neighbor. Use
   this to map which windows the baseline is silently carrying (an investigation
   instrument, not necessarily a production flag).
2. **The contributor-model fix + cascade test on 1019 w79/w80** (the clean
   experiment). 1019 w79 has a strong peak over-split into a doublet (a merge in
   curation); w80 next to it has a weak feature riding w79's skirt, currently
   modeled by baseline alone (edge dropped → `ncon=0`). Fix the contributor model
   (window-level resolution) so w80 carries w79's fitted line(s) as contributors,
   then: merge w79's doublet and confirm the change *cascades* into w80's skirt
   and moves its fit. This is the decisive "does the dependency matter" test on a
   real, defensible curation decision.
3. **Decide the resolution**, consistent across SNR:
   - **Better cyclic-dependency resolution** than "drop all edges" — break the
     minimal feedback arc set, or resolve dense cycles with a real (accepted)
     thaw co-fit rather than edge-free demotion.
   - **Declare dependencies unnecessary** — if the baseline genuinely models the
     leakage adequately (the baseline-magnitude diagnostic says it is small),
     retire contributor edges uniformly so moderate fixtures match the dense ones.

Only once the dependency model is settled does the cascade design below apply:
the cascade propagates edits *along the dependency graph*, so its value and even
its existence depend on whether that graph should exist.

### Findings: item 1 (baseline-magnitude diagnostic) — done

`scratch/cascade/baseline_magnitude.py` reconstructs each window's fitted
leakage-wing baseline `B(u)` from the persisted `baseline_coeff{k}_re/im`
(no re-fit) and compares it to the active-FT data and the noise floor on the
window. The carried-leakage signature is the conjunction of three things:
`r_data = max|B|/max|X| ≳ 0.2` (baseline is data-scale), `r_noise =
median|B|/median σ_c ≳ 5` (and is real structure, not the noise envelope of a
noise-only window), and `ncon = 0` (no contributor is carrying it). `edge_frac`
(the location of `max|B|` as a fraction of the window half-width) distinguishes
a neighbor's wing (`≈1`, peaks at the edge) from line-vs-baseline degeneracy
(`≈0`, peaks at center).

The diagnostic confirms the SNR-inconsistency, quantitatively and decisively:

| Fixture | windows | baseline-fired | carried-leakage | of which ncon=0 | edge_frac med |
|---|---|---|---|---|---|
| 655  | 719 | 715 | 393 | **304 (42%)** | 1.00 |
| 1019 | 64  | 42  | 13  | 13            | 1.00 |
| 1512 | 80  | 29  | 6   | 4             | — |

On the dense 655, **304 windows (42%)** carry a data-scale, far-above-noise
leakage wing on the order-4 polynomial baseline with **zero contributors** —
`r_noise` reaches 451× and `r_data ≈ 1.0` (the baseline equals the data at the
edge) on windows that themselves hold real SNR-60–145 lines (w917, w1104,
w246, w125). The carry concentrates in low SNR: 272/491 (55%) of <30-SNR
windows, falling to 3/55 (5%) at 100–300 SNR and 0 above. The moderate 1512
silently carries on only **4** windows and *retains* contributors (ncon 6–12)
on its other leakage-touched windows. So the baseline on the dense fixture is
**not** a small correction — it is materially modeling a neighbor's wing that a
contributor should carry. This favors the **"better cyclic-dependency
resolution"** branch (§3) over "declare dependencies unnecessary."

### Findings: item 2 setup (1019 w79/w80) — structure confirmed, harness next

Inspecting the fresh 1019 (`scratch/cascade/_inspect.py 1019.ftmw 76 83`) the
w79/w80 pair is real but richer than first described:

- **w79** [39470.6, 39482.1] (a wide 11.5-MHz window) holds the spectrum's
  dominant line — **SNR 27 265** at 39477.980 (amp 3.0e-3) — over-split into a
  doublet with a SNR-2116 satellite 82 kHz away (sub-resolution at this
  ~93-kHz element). Its `chi²ᵣ = 2986` is **not** a fit defect — it is the
  expected lineshape-floor inflation for an SNR-27 000 line (σ is tiny relative
  to the signal, so any in-tolerance shape error blows χ²ᵣ up enormously); it is
  independent of any cascade. The merge candidate is the doublet.
- **w80** [39485.0, 39490.9] holds an SNR-130 line at 39487.980 with
  `ncon_qm = 0` — the giant line's skirt (~10 MHz away, estimated amplitude
  ~1.3e-5, *comparable to w80's own 1.6e-5 line*) is carried by the order-4
  baseline, not a contributor.
- The persisted edge into w80 (and w78/w81/w82) references w79 at **stale
  Stage-3 candidate frequencies** (39473.2, 39473.5, 39479.1) that do **not**
  match w79's actual fitted lines (39477.98, 39478.06) — the snapshot-staleness
  defect, observed live, and the direct motivation for window-level resolution
  (resolve against the source window's *current fitted* above-threshold peaks).

This validates w79→w80 as the cascade test. `scratch/cascade/wl_cascade.py`
implements it: it mutates the persisted Stage-4 plan and re-runs the production
fit so the evidence gate decides, comparing the target window across the
persisted fit, a clean control re-fit, and the mutated re-fit. With
`--edge-bearing` the contributors are imposed (`edge_free=False`): the dependent
subtracts the skirt synthesized from the source's *fitted* parameters (zero free
parameters in the dependent), which is the only treatment a curation edit on the
source can propagate through; `fit_peaks` loads the persisted plan as-is
(`stage5_impl.py:1359`, no cycle-break on load), so a one-directional artificial
edge survives.

### Findings: item 2 (cascade bite on 1019) — done, surprising

The dependency has **near-zero bite on the reported lines**, across every
treatment, and the reason is mechanistic, not a null fixture:

| treatment of w78←w79 | ncon | baseline max\|B\| | r_data | reported line | χ²ᵣ |
|---|---|---|---|---|---|
| current (edge-free / baseline) | 0 | 2.97e-5 | 0.907 | 39467.9790 | 1.08 |
| edge-bearing (impose w79 fit)  | 2 | **2.09e-6** | 0.064 | 39467.9790 | 1.08 |

- **Edge-free** contributors read (amp, phase) from the *data* via LSQ, so they
  are self-consistent with the data by construction: cleaning w80's stale
  contributor frequencies to w79's actual fitted peaks, or dropping the giant
  contributor entirely, moves w80's line by **≤0.02σ** (χ²ᵣ 1.35→1.36). They
  *are* applied when the arbitration gate adopts them (w80 ncon=6) — "edge-free"
  is about *where (amp,phase) comes from*, not *whether the skirt is subtracted*
  — but they cannot reveal the strong line's influence because they re-fit
  themselves to the data.
- **Edge-bearing** (impose w79's *fitted* skirt, zero free parameters) *also*
  leaves the neighbor line byte-identical (Δf=0). The decisive evidence is the
  baseline: explicitly subtracting w79's skirt **collapses w78's order-4
  baseline 14×** (max|B| 2.97e-5 → 2.09e-6, r_data 0.91 → 0.06) while the line
  does not move. The order-4 leakage baseline (essentially always on) and a
  physical contributor are **interchangeable substitutes** for the skirt — the
  edge just moves the skirt from the polynomial to the contributor.

So the strong line's effect on neighbors is real and *dominant in magnitude*
(item 1: r_data≈1.0) but it is a **smooth pedestal** over the ~6 MHz neighbor
window that the baseline removes as cleanly as a constrained skirt. All three
mechanisms (edge-free, edge-bearing, baseline) leave the same residual → the
same line. The cycle-breaker dropping edges does **not bias the reported
frequencies**; a cascade would move a dependent's *baseline*, not its line.

**The one untested regime** (where the verdict could still flip): a window
*close enough* to a strong line that the skirt has curvature/oscillation beyond
order-4 (sinc fringes near the line), or a **weak real line riding a structured
part of the skirt** that the baseline would absorb (masking). 1019's w79 is a
wide window, so every neighbor is ≥7 MHz out — the smooth-tail regime — and no
line was masked (npk unchanged). Settling item 3 requires probing the
structured-skirt geometry (a narrow window a few resolution elements from a
strong line, e.g. on a moderate fixture).

### Findings: item 2, uncertainty / correlation dimension — the real value

Central values are moot, but the choice of skirt treatment does move the per-line
*uncertainties* through line↔baseline correlation (the joint covariance is the
peaks+tau+baseline inverse JᵀJ; the baseline "flexibility is priced into the
per-line uncertainties"). `scratch/cascade/corr_experiment.py` fits the target
in-process across baseline order p∈{0..4}, with and without the edge-bearing
skirt pre-subtracted, reading σ_f, σ_amp, and the max line↔baseline correlation
straight from the covariance. w78 (skirt = 96% of its data), at the χ²ᵣ≈1
operating points (order-0-without-skirt is χ²ᵣ=207, a garbage fit, excluded):

| treatment | χ²ᵣ | σ_f (kHz) | σ_amp (%) | corr(amp,base) | corr(f,base) |
|---|---|---|---|---|---|
| current: order-4, no explicit skirt | 1.08 | 0.385 | 0.79 | 0.42 | 0.37 |
| edge-bearing skirt + order-0/1       | 1.1–1.9 | 0.36 | 0.71 | 0.22 | 0.19 |

- Without the skirt the baseline **needs order ≥2** for χ²ᵣ≈1; the skirt is what
  forces the high order. With the skirt subtracted, order 0/1 suffices.
- The correlations are real and grow with order (amp↔baseline reaches 0.42 at
  order 4). Carrying the skirt as a zero-parameter contributor lets the baseline
  drop to order 0/1, which **halves the correlation (0.42→0.22)** and **tightens
  σ_f ~7% and σ_amp ~10%**.
- The magnitude is *modest* (≤10%): a line's frequency is nearly orthogonal to a
  smooth polynomial (variance inflation 1/√(1−ρ²) ≈ 8% at ρ=0.37), and even its
  amplitude only loses ~10% to an order-4 baseline.

**Refined verdict (moderate SNR):** at SNR ~27k (1019) the edge-bearing/cascade
model's value is *not* moved central values (≤0.02σ) but **tighter error bars via
reduced baseline DOF** — ~7–10% σ, halved nuisance correlation — which the
current pipeline leaves on the table (it holds the baseline at order 4 even when a
physical skirt is available). Honesty caveat: the with-skirt σ treats the source
fit as exact (zero-parameter skirt); fine for a strong source (negligible
propagated uncertainty), illusory near the freeze threshold.

### Findings: extreme SNR (655 w1006, SNR ~1e6) — the verdict is SNR-dependent

The user's "immune by construction" holds *geometrically* (the structured
near-line region is inside the strong line's own window; w1006's giant doublet at
37904.87 sits in a wide 12.6 MHz window, so the nearest neighbor windows w1005 /
w1007 start ≥6 MHz out). **But the skirt magnitude alone breaks the
baseline-substitute equivalence at 1e6**, and real data contains the case — no
synthetic fixture needed.

- **Baseline-only is insufficient** (`corr_experiment.py` on w1005, skirt = 87%
  of its data, no contributors): even at order 4 χ²ᵣ stays **13–31** (never ~1)
  and σ is unstable (order-3 σ_f blows to 3444 kHz). An order-4 polynomial cannot
  absorb a 1e6 line's skirt to noise level — its gentle curvature is still many σ
  when the skirt is that large. (1019's order-4 baseline reached χ²ᵣ≈1; 655's
  cannot.)
- **Edge-free is also insufficient, and edge-bearing materially beats it.** Full
  pipeline, flipping only the dominant w1006 edge to edge-bearing (the other
  contributors left edge-free): w1007 goes npk 2 → 4, χ²ᵣ **6.31 → 2.02**, and
  **unmasks two weak lines** (snr 8, 10) the edge-free/baseline treatment had
  absorbed. Production already carries 12 edge-free contributors here yet is
  underfit at χ²ᵣ=6.31 — a material defect, not a ≤0.02σ nudge.
- **Mechanism:** w1006 has 6 lines packed in 12 MHz; their skirts into w1007 are
  nearly **collinear**, so an edge-free read (skirt amplitudes fit from the
  *dependent's* data) cannot separate them and gets the composite skirt wrong.
  Edge-bearing uses w1006's *fitted* amplitudes — well-determined where those
  lines are *resolved*, in w1006's own window. **Edge-free fails exactly when a
  strong source has multiple close lines**; that is the cascade/edge-bearing
  model's real, SNR-gated value.
- **Implementation complication (w1005 caveat):** the same mutation made w1005
  *worse* (χ²ᵣ 2.55 → 19, ncon 12 → 3) — not because edge-bearing is bad, but
  because flipping one edge changed the **joint arbitration** of the other 11
  edge-free contributors (the gate rejected 9). The dense contributor set is
  arbitrated *interdependently*, so a clean one-edge swap is not possible through
  the current machinery — likely why the cycle-breaker took the blunt
  "demote-everything-to-edge-free" route. Any edge-bearing cascade must resolve
  this joint-arbitration entanglement, not just add an edge.

**Net for item 3:** the dependency treatment is **moot at moderate SNR, material
at extreme SNR** (line-list and χ²ᵣ changes near a multi-line ~1e6 source).
"Declare dependencies unnecessary" is falsified at the extreme. The path is some
form of **better cyclic resolution** that makes a dominant source's edge
edge-bearing (resolving the joint-arbitration entanglement), at least for the
multi-line strong-source case where edge-free demonstrably fails.

### Findings: the curvature cycle-breaking criterion + the overfit-skirt trap

A leakage skirt's *level* decays ~1/Δf (significant everywhere → the fully
cyclic dependency map) but its *curvature* — the part a low-order baseline
cannot absorb — decays ~1/Δf³, so it is steeply local. Gating edges on
curvature (not level) should collapse the graph to local strong→weak arcs that
are naturally acyclic. `curvature_edge.py` / `curvature_window.py` measure, in
units of the dependent's σ_c, `S_resid(p)` = what survives an order-p polynomial
fit of a projected skirt.

- **Strong→weak asymmetry is decisive (the cycle-breaker):** forward skirt
  significance S_level = 1100 (1019 w79→w80) / 5051 (655 w1006→w1007); the
  *reverse* arc is 7.7 / 17.5 — a 100–300× asymmetry, and the reverse residual
  is negligible at every order (S_resid(4) = 0.01 / 0.06). Seed from the
  strongest windows, fan out; level/curvature orients the arcs one-way.
- **Curvature decays ~6× per baseline order** (655: 1121→223→44→8.5→1.63), so a
  low-order baseline absorbs most of a *smooth* skirt.
- **Thaw is the third tier (user):** the one-way orientation cannot order two
  *close strong seeds* (peers, neither upstream) — that residual cycle is
  exactly what the **thaw** co-fit handshake is for. Thaw is dormant today only
  because the cycle-breaker demotes everything to edge-free (nothing
  edge-bearing remains to thaw); under this design it reactivates precisely on
  close seed–seed pairs.

**The overfit-skirt trap (decisive).** 655 w1006's dominant "line" is an
*overfit doublet*: two members 0.7 kHz apart (≪ 85 kHz element), equal amplitude
(0.180 / 0.177), **π out of phase** (Δφ = 3.08). Their far skirts **cancel** —
S_resid(4) into w1007 is 21.8 for *one* member but **1.38 for both**. So the
overfit's *fitted* skirt is a phantom (1.38σ) that disagrees with the *data*
(one real line, ~20σ+). Consequences:

- An **edge-bearing** dependent inherits the phantom cancelling skirt
  (under-subtracts → underfits); an **edge-free** dependent reads the true skirt
  from data and is immune. This is a strong, previously-hidden argument that
  edge-free is the *safe* fallback — it does not propagate source fit artifacts
  — and that naive edge-bearing is dangerous on an un-curated source.
- This is the **cascade-with-real-bite** case the investigation was missing:
  merging the overfit (a defensible curation) changes w1006's edge-bearing skirt
  into w1007 from 1.38σ → ~20σ — a ~20σ change that *must* refit w1007. The
  earlier "moot" verdict tested *parameter* edits on already-good fits; a
  *structural merge of an overfit on a strong contributor* is material.

**Forced design ordering:** curation **first** (merge the overfit), edge-bearing
propagation **second**. Edge-bearing is only safe on a *trusted* (curated)
source fit; the cascade is the mechanism that makes the dependent honor the
correction. NEXT: actually merge w1006 → single line, refit, and measure the
cascade into w1007 (edge-bearing) — the culmination on a defensible curation.

### Handoff state (scratch harnesses, reusable)

Built this session under `scratch/cascade/` (gitignored): `build_655.py
<name>` (fresh fixture build via `run_pipeline`), `characterize.py` (edge-free
vs edge-bearing contributor structure + closures), `pathology_scan.py`
(per-window baseline order + post-fit residual edge-coherence), `ab_skirt.py`
and `ab_faithful.py` (skirt-vs-baseline A/B; the *faithful* one restores a
dropped contributor into the persisted plan as edge-free and re-runs the
production fit so the evidence gate decides), `show_w24.py` (native Re/Im/|X|
decomposition), `baseline_magnitude.py` (item-1 diagnostic: `r_data` /
`r_noise` / `edge_frac` carried-leakage signature, no re-fit), `_inspect.py
<file> <lo> <hi>` (per-window peaks + contributors + chi²ᵣ + baseline order for
an id range), `wl_cascade.py <file> <target> <source> [--edge-bearing]`
(window-level resolution / cascade test: mutate plan + re-run production fit),
`corr_experiment.py <file> <target> <source>` (in-process σ_f / σ_amp /
line↔baseline correlation vs baseline order, with/without the edge-bearing
skirt), `_neighbors.py <file> <wid>` (rank windows by frequency gap to a target
+ per-window npk/snr/chi²ᵣ/ncon/order), `curvature_edge.py <file> <dep> <src>`
(S_level / S_resid(p) of a projected skirt + the strong→weak vs reverse
asymmetry), `curvature_window.py <file> <dep>` (per-contributor vs combined
S_resid — the shared-baseline load). Fresh fixtures:
`scratch/cascade/{655,1512,1019}.ftmw` and their
reports under `scratch/cascade/reports/`. Rebuild fresh per the rebuild-fixtures
rule if code changes (a stale fixture carries build-time settings).

### Deferred follow-up (diagnostic surfacing)

Independent of the resolution decision, the investigation showed these are
currently invisible to a reviewer and should be surfaced — at least the
**out-of-band contributor count** and the **leakage-wing baseline order** — in
both `fit show` and a prominent place in the HTML report. A window whose edge is
dominated by a neighbor's leakage wing modeled by an order-4 polynomial baseline
with zero contributors reads as "clean" today; the count + order would expose
that. Relatedly, the per-window panels should plot the native-bin values as
authoritative markers (not only an interpolated curve), so a reviewer reads the
measured spectrum directly. (Design thoughts pending; tracked here so it is not
lost.)

## The problem

A strong line fit freely in its own window contributes its finite-T leakage
**skirt** to neighboring windows as a frozen (non-re-fit) component — a
`FixedContributor`. When the strong line is **edited during Stage 6 curation**,
that edit does not reach the dependent windows. The dependent windows keep the
skirt they were given at Stage 5 fit time.

### What happens today, precisely

The frozen background a dependent window **D** subtracts is a **snapshot**, not
a live reference. At Stage 5 fit time each window persists its own
`fixed_parameters` dict — the contributor's `(frequency, amplitude, phase)`
frozen at the value found in the primary window **W**. Every later operation
(`_reconstruct_frozen_peaks` in `_internal/stage6_impl.py`) rebuilds D's frozen
background **from D's own persisted snapshot**, never from W's current fit.

There are two distinct "rebuild" paths and **neither** propagates an edit:

- **`review run` (decision replay).** Each decision is replayed window-locally
  (`refit_window_core` is explicitly "no cascade"). Editing contributor C in W
  replays onto W only; D is not in the decision log, is never touched, and keeps
  its **pre-edit** snapshot of C.
- **`fit run` (full Stage 5 re-fit).** Wipes Stage 5 and re-fits from the
  *data*. The dependency DAG fits W before D and D reads C live from W's
  converged fit — but that is the **auto** value of C, because the Stage 6 edit
  is not a Stage 5 input. After `fit run` the edit is gone until `review run`
  replays it, and replay again touches W only.

So a Stage 6 edit to a contributor **never reaches the dependent windows by any
path a user currently has.** The file is left internally inconsistent: C's own
line-list entry reflects the edit; every dependent window's *model of C*
reflects the pre-edit value. Because a line is only a contributor when it is
strong (`min_freeze_snr`) and its predicted skirt into D was large enough to
keep, the inconsistency is non-trivial by construction whenever a contributor
exists — the only open variable is how much the edit moved C.

### Documentation discrepancy to resolve

`docs/source/stage6_review.rst` ("Limitations") states: *"Stage 6 reports which
neighbors reference the window but does not re-fit them."* The attention-reason
machinery (`_compute_attention_reasons`, kinds `auto_merged_review`,
`worst_eps`, `overfit_vif`, `candidate_bearing`, `spur_adjacent`,
`edge_boundary`) and the report layer implement **no such neighbor-reference
reporting** — the staleness is silent today. This is a code-vs-doc divergence;
the cascade work resolves it (the doc text changes to describe the cascade).

## The decided design

### 1. Window-level dependency resolution (a baseline change, for the better)

Replace per-peak contributor resolution with **window-level** resolution. A
dependency edge points at a **source window**, not a specific peak. At fit time,
the dependent window includes the skirts of **all fitted peaks in the source
window above `min_freeze_snr`**.

This is the foundation that makes propagation well-defined, and it dissolves the
peak-identity problems:

- **Split** of a strong contributor into C1/C2 → both are above threshold →
  both skirts included automatically (no nearest-frequency ambiguity).
- **Deletion** of the contributor → no peaks above threshold in the source →
  the edge resolves to **zero skirt, harmlessly**.
- **Frequency / amplitude shift** → picked up automatically.
- A **newly strong** peak appearing in the source under a refit (e.g. a rescue
  add) → its skirt is now included, which is correct.

The semantics of *how* a peak changed no longer matter — the dependent always
subtracts "whatever the source window currently fits, above threshold."

This aligns with the existing data model rather than adding to it:
`WindowPlan.dependency_edges` is **already** `(window, depends_on_window)`. The
window-level edge exists; it is the per-peak `FixedContributor.peak_index` that
is the redundant, brittle layer that this change can retire.

**Cost / scope.** This changes the **baseline** contributor evaluation, not just
the cascade. Today's plan is more selective than "all above-threshold peaks": it
applies a predicted-skirt magnitude threshold and an edge-free top-3 cap
(`DEFAULT_MAX_EDGE_FREE_NEIGHBORS`). Switching to window-level resolution will
move baseline fits (possibly not byte-identical) and must be reconciled with the
edge-free / cycle-break path. **Treat it as a foundational Stage-5 change,
validated against the 7-fixture baseline first**; the cascade then rides on top.

**Edge-free contributors are unaffected in spirit.** Cycle-broken edges are
resolved `edge_free` — read from the active FT, not from a fitted neighbor — so
they are **cascade-immune by construction**, and that is correct: the data did
not change, so the skirt should not. Under window-level semantics the edge-free
read still uses the source window's above-threshold peak *frequencies* to know
where to read; amplitude/phase still come from the data.

### 2. Cascade is obligatory, logged, and reversible

The curation layer is a **single level** on top of the automatic baseline — not
iterative versions. State it precisely:

> curated fit = f(baseline, decision_log, dependency_plan), computed by one
> deterministic top-down window-plan walk.

Auto-cascade does not add a concept; it widens the set of windows that one
revision recomputes. When an edit changes a contributor, the walk recomputes the
**transitive closure of the edited windows' descendants** in the dependency DAG,
in topological order, seeded from the baseline everywhere except where edits or
updated upstream skirts intrude. This is `execute_plan` restricted to the
affected subgraph.

- **Obligatory**, because a partially-cascaded file *is* the inconsistent state
  we are eliminating; making it optional reintroduces the danger.
- **Logged as revisions** for every window the cascade actually re-fits (parent
  and dependents), so the change is never silent.
- **Reversible** via re-walk: persist only `(baseline, decision_log)`. The
  curated layer is fully derived; undo = drop the decision entry and re-walk from
  baseline. No per-window pre-cascade snapshots (they would be a redundant second
  source of truth that can drift). A full re-walk is acceptable (minutes);
  subgraph-limited re-walk is a later optimization not worth the bug risk now.

### 3. Composition of cascade refits with direct edits

The curation layer is no longer a set of independent window-local edits; the
cascade couples them. The deterministic rule:

> For each window in the affected closure, in topological order, fit it with
> *(updated upstream skirts)* ∧ *(its own direct user edits)*.

A single topological pass over the union of all directly-edited windows'
descendant closures resolves every ordering question — a window that is both a
cascade target and the subject of a direct edit (E2) is fit once, honoring E2
and the updated upstream skirt together. There is no "who wins" ambiguity
because there is no second pass. The implementation requirement is exactly that:
**one merged topo-ordered pass**, not cascade-refit and direct-edit as two
layered code paths.

### 4. Discovery in the cascade: NLS-only plus one bounded rescue round

`refit_window_core` is NLS-only (no conservative discovery, no rescue, no thaw,
no replan), which keeps the cascade deterministic and bounded. Consequence: a
pure-NLS cascade can shift/shrink a dependent's peaks but cannot *gain* one — so
it cannot recover a line in D that had been masked by the wrong skirt.

Decision: allow **a single residual-rescue round** in a cascade refit *when an
out-of-band contributor changed*. Rescue is deterministic (residual-max seed,
raw-χ² accept), so one round preserves determinism, and gating it on "a
contributor actually changed" keeps it from running everywhere.

**Guard:** rescue must **not re-add a line at a user-suppressed frequency.** User
removals already carry prune-immune provenance; this is the symmetric guard, or
an edit on W could resurrect, via a rescue in D, a peak the user explicitly
deleted in D. Rescue-added lines carry their own provenance and surface through
the peak-count flag (below).

### 5. Thawed-peak ownership folds into the same fix

`refit_window_core` holds **thawed** peaks (owned by a primary, co-fit into D)
frozen at their persisted values. A naive cascade would refresh the frozen
*skirt* but leave a *thawed* peak stale — the same inconsistency in the co-fit
channel. The topo-walk must refresh both channels from the re-fit primary. This
is not extra scope; it is the same defect (latent in the single-window model)
wearing a co-fit hat, fixed by the same mechanism.

### 6. Attention flags: surface only scientifically significant changes

The cascade report flags a dependent window for a second look — deliberately
**not** every re-fit window, since σ is typically ≪ bin size and small shifts
are invisible by eye. Two families:

- **Result changed.**
  - A peak's parameter moves by more than *k* × **max(σ_before, σ_after)** for
    that parameter (max, not a quadrature "mutual" of two highly-correlated
    fits — same data, slightly different background → ρ≈1, and quadrature
    inflates the bar in a way that is hard to reason about). "Did the reported
    number move by ≳ *k* × its stated precision" matches the downstream
    consumer's experience. *k* (2, 3, …) is **tunable after seeing the
    implementation** — pick what reads as scientifically significant.
  - A parameter's **uncertainty inflates** by more than ~2× (the fit became
    ill-conditioned; also catches a shift that only looks small because σ blew
    up).
  - **Peak-count change** in a dependent (under NLS-only this can only decrease —
    a collapse — except where the bounded rescue round adds one).
- **Fit quality changed.** Flag on χ²ᵣ change, **DOF-gated**: a 20% χ²ᵣ change is
  overwhelming on a high-DOF window and pure noise on a 4-DOF one (χ²ᵣ sampling
  spread ≈ √(2/DOF)). Use `|Δχ²ᵣ|/χ²ᵣ > max(20%, c·√(2/DOF))`, or equivalently a
  change in χ² significance. A χ²ᵣ *drop* can be legitimate (a better skirt gives
  a more precise determination), so this replaces a naive "σ deflation" rule on
  that side.

**Development aid:** expose the ability to **rank peaks / windows by
baseline-vs-curated difference** (the same metrics above). At least during
development this is how we find where the "scientifically significant" line
should be drawn before fixing thresholds.

### UX notes

- Even though the cascade is obligatory, an **edit-time hint** ("this line feeds
  windows [X, Y]; they will be re-fit") is cheap orientation, not a gate.
- **Obligatory bundling** means a user cannot keep an edit while rejecting its
  cascade consequence. If the cascade makes a dependent worse (flagged via
  χ²ᵣ), the recourse is to undo the original edit or to **directly curate the
  flagged dependent** — which is consistent with the model (the dependent then
  becomes a directly-curated window).

## Experiment plan (gates the build)

Build a scratch harness that implements **window-level resolution** and a
cascade so we measure the magnitude/decay *and* validate the foundational
semantics before either touches Stage 5. There is no cascade entry point today,
and `_reconstruct_frozen_peaks` reads the stale snapshot, so the harness must
rebuild a dependent's frozen background from the source window's current
above-threshold peaks and re-fit.

Two fixtures, two purposes (build **fresh** per the rebuild-fresh rule):

- **655 → closure size / fan-out (the cost question).** Dense, high-SNR, many
  strong contributors with real dependents and a known leakage pedestal — the
  worst case for cascade radius. Pick the strongest line that is a contributor
  to ≥1 window (rank by predicted skirt into the dependent so an effect is
  visible) and measure how many hops the cascade reaches before every flag goes
  quiet. Confirms the "obligatory" cost is bounded (the 1–2-hop hypothesis).
- **1512 → scientifically-defensible structural edit (the magnitude question).**
  The ¹⁴N hyperfine doublets give a physically motivated split/merge on a strong
  line that is also plausibly a contributor — a *reasonable* curation decision,
  not a contrived one.

For each, run two edit classes on the chosen contributor's window W:

- **(a) Insidious / indirect.** Add or remove a *blend satellite* in W, leaving
  the strong line nominally alone, to see how much the contributor moves
  *indirectly* (joint-NLS correlation) and whether that trips a dependent.
- **(b) Structural.** Split the strong contributor — the upper bound.

Per hop record: Δfreq / σ_f, σ_after / σ_before, Δχ²ᵣ, peak-count change. Track
the decay across hops.

**Headline to watch for.** If even a structural split of a strong contributor
moves its top dependent by ≪ σ_f, the feature is a correctness/honesty fix with
rare practical bite (still worth doing — the file should never lie about its own
consistency) rather than a frequently-material recomputation. Either result is
useful and informs how much machinery is justified.

## Interface surface (to design during build)

All three interfaces must stay identical (the dual-interface invariant). The
cascade is internal to the edit verbs (`review edit/merge/split/accept` and
`review apply`) — it is not a new user verb; editing a contributor simply
re-fits more windows and records more revisions. New/changed surface to specify:

- The **ranking** development aid (likely `review rank --by <diff-metric>`,
  extending the existing `review rank`).
- The **cascade report** section (which windows were re-fit, which flagged, and
  why) in `review show` / the HTML report.
- Decision-log provenance for cascade-induced refits and rescue-added lines,
  distinct from direct user edits.

## Serialization

- Retire / deprecate `FixedContributor.peak_index` as the resolution key in
  favor of the window-level edge (keep the edge list; resolve dynamically).
- No per-window pre-cascade snapshots. Persisted truth stays `(baseline,
  decision_log)`; the curated fit is derived.
- Window-status / decision-log entries gain cascade provenance and the new
  attention-flag kinds.

## Test plan

- Cross-interface consistency tests for any verb whose behavior changed.
- A determinism test: the curated fit is byte-stable under re-walk (same
  baseline + decision log → identical result), including undo → re-apply.
- Composition test: a direct edit on a window that is also a cascade target is
  honored exactly once (single topo pass).
- Guard test: a cascade rescue round does not re-add a user-suppressed line.
- The window-level resolution change re-baselines the 7-fixture suite; the new
  baseline is the reference (document the deltas, expect small).

## Open questions (do not relitigate without a factual contradiction)

- Final flag thresholds (*k* for the shift, the χ²ᵣ gate constant) — tune after
  the experiment.
- Whether the predicted-skirt magnitude threshold and the edge-free top-3 cap
  survive unchanged under window-level resolution, or need re-tuning — settle
  from the 7-fixture re-baseline.
- Whether the single bounded rescue round is enough, or whether any structural
  edit needs more — settle from the experiment's structural-edit case.
