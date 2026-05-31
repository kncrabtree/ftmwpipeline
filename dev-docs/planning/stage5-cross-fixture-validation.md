# Stage 5 — Cross-fixture validation and per-dataset calibration

Status: **planning** (fixtures acquired; premise reconciled — see the
reconciliation section next). This document was opened after the
residual-rescue and AICc-gate work
([`stage5-residual-rescue.md`](stage5-residual-rescue.md),
[`../research/residual-rescue/report.md`](../research/residual-rescue/report.md))
landed on the 2638 fixture. That work made several decisions
calibrated to 2638-specific numbers (most importantly the
`shape_error_epsilon` constant); those calibrations need to be
verified or re-derived on other fixtures before any of them can be
treated as production defaults. This document also captures the
**lineshape model deficit** discovery from that work.

The keystone fixture blocker (#1) is now closed: seven same-instrument fixtures
build canonically, two with ground truth (1512 — 115 clean vinyl-cyanide lines;
655 — 328-line 5-species union, ~120× SNR). Validating against them **overturned
the physics premise this doc was built on**; the reconciliation below is
normative for execution and supersedes the ε/Voigt framing in the body.

Three parts of this doc have been overtaken and should be read with the notes
below:

- The **§Dataset-wide tau calibration** and **§Broken-initial-fit /
  freeze-at-consensus** proposals are superseded by the shipped data-driven τ
  calibration in [`stage2b-tau-calibration.md`](stage2b-tau-calibration.md)
  (`τ_maj ± σ_τ`, per-band routing, bidirectional Gaussian τ penalty) — see the
  in-section notes there.
- The **lineshape model deficit → "treat as irreducible, inflate σ via ε"**
  framing is **superseded** (not merely "partly overtaken"): the 655 ground
  truth shows the line shape is **Lorentzian, not Voigt** (see §Reconciliation
  point 1). The Gaussian shape path
  ([`stage5-gaussian-shape.md`](stage5-gaussian-shape.md)) remains available but
  did not win spectrum-wide on 2638 (#4). The Tier 1/2/3 acceptance *framework*
  remains the valid plan; the *lever* it calibrates has changed.

## Reconciliation: 655 + 1512 supersede the ε/Voigt premise (read first)

The fixtures arrived and two carry ground truth. Validating against them
overturned or refined the three premises this doc hangs on. **Read this before
executing the Tier 1/2/3 plan below — large parts of the `shape_error_epsilon` /
Voigt machinery are superseded.** Provenance: `655-vycn-validation` and
`issue1-1512-fitting-test` memories; `../research/noise-snr-scaling/report.md`.

1. **The line shape is Lorentzian, not Voigt/Gaussian.** On 655 (the high-SNR
   discriminator) `recommend_shape`'s per-line L/G/V AICc vote is **exp 88% /
   voigt 10% / gauss 2%** (Lorentzian, margin ~86%). The "Voigt-like deficit"
   that motivated the entire `shape_error_epsilon` apparatus (§The lineshape
   model deficit, §How to measure ε) was a **misread**: it came from fitting
   default-Lorentzian *without* engaging the shape recommender, plus mistaking
   beating on a near-degenerate doublet (38847) for envelope curvature.
   **Consequence:** `shape_error_epsilon` is most likely *not* the
   generalisation lever this doc frames it as. Before investing in cross-fixture
   ε calibration, re-test whether ε is needed *at all* once τ and noise are
   correct (Theme T4 below). Treat the ε sections as historical.

2. **The χ²ᵣ driver is τ-calibration bias, not an irreducible shape deficit.**
   STFT `calibrate_tau` is biased — *low* on dense/high-SNR spectra (655:
   `τ_maj` 2.5 µs vs ~4 µs truth from `calibrate_tau_G` and the model-free
   demodulated envelope), and *high* on low-SNR (1512: 8.3 µs). Stage 2b's τ
   penalty anchors on the biased value and broadens per-window models → strong-
   line χ²ᵣ blowups (655 w224 χ²ᵣ 8.3 → 1.2 under a `tau_maj_override_us=4.0`).
   Stage 2b shipped the *mechanism* (data-driven `τ_maj ± σ_τ`, per-band
   routing, bidirectional Gaussian penalty) but the *bias* is the live cross-
   fixture question: on dense spectra prefer `calibrate_tau_G` / envelope τ over
   STFT. This is Theme T2 / issue #3 and is the biggest χ²ᵣ lever.

3. **The Stage 2 noise keystone is fixed.** The "inflated noise hides the
   deficit" failure mode is resolved: `estimate_noise_scatter` (`method=
   "scatter"`, now default) replaced the pedestal-pinned MAD that overestimated
   σ 5–6× at extreme SNR. With honest noise, 655's predicted χ²ᵣ rises
   0.17 → 3.0 — exposing the τ deficit (point 2), not a shape deficit. The
   "Symptom: chi²ᵣ median > 1.5 → investigate Stage 2 first" guidance below now
   means *confirm the scatter estimator is engaged*, not the legacy adaptive
   one.

4. **New headline the doc omits — uncertainty honesty has two layers.** 1512's
   stated deliverable ("does σ_f match reality?") resolves on 655 into: σ_f is an
   honest LSQ *precision* (~1 kHz on bright lines) but there is a fixed
   **instrument absolute-accuracy floor ~10 kHz** (655 detrended residual
   scatter 8.9 kHz, 1512 6.6 kHz, SNR-independent) from free-running digitizer
   clock drift (1–2 ppm, removable as a linear-in-`f_RF` slope + a constant
   offset). σ_f *structurally cannot* capture it (σ_f ∝ 1/SNR; the floor is
   constant), so reported σ_f is honest as precision yet 8–44× overconfident *as
   accuracy* on strong lines. This belongs in Tier 3. **Deliverable:**
   characterize the floor per fixture, decide whether reported uncertainty
   should carry a quadrature instrument-accuracy term, and how to calibrate it
   (per-acquisition ppm-slope removal). Theme T3.

**What survives unchanged:** the Tier 1 (distribution health) / Tier 2 (gate
firing) / Tier 3 (ground-truth recall/precision/amplitude/uncertainty)
acceptance *skeleton*. What changes is the lever it calibrates: **τ-source and
the σ_f accuracy floor, not a Voigt ε.**

### Execution themes (post-reconciliation)

| theme | what | issues | status |
|---|---|---|---|
| T1 | Cross-fixture characterization harness (tracked recipe: build→fit all 7, emit Tier-1 χ²ᵣ health + Tier-2 gate-firing + τ-source comparison + σ_f floor), → roll-up | backbone for #2/#3/#4 | **landed** (`../research/stage5-cross-fixture/stage5_cross_fixture.py`; 2638 χ²ᵣ med 1.31 anchor); re-baseline pending per-fixture shape |
| T2 | τ-calibration robustness → **asymmetric long-anchor τ penalty** for swallowed-hyperfine blends (start narrow, broaden cheap, narrow expensive). STFT band-averaged-low / `calibrate_tau_G`-high confirmed; band-dependent τ reconfirmed | #3 | **in progress**: mechanism implemented in `window_fit` (back-compat, 295 tests green) + window-level validated (655 w29 χ²ᵣ 44865→327); production plumbing + full `fit_peaks` A/B next. See `../research/stage5-cross-fixture/report.md` |
| T3 | Uncertainty honesty: precision-vs-accuracy gap on 1512+655; calibrate instrument accuracy floor; decide σ_f_floor term | #2 Tier 3 | unblocked; shippable |
| T4 | Shape/ε reconciliation: re-run Gaussian acceptance bar on 1512/655; decide if `shape_error_epsilon` is still needed once τ+noise are right | #3/#4 | unblocked; likely retires ε |
| T5 | Land `validate-stage5-shape-error` CLI + per-fixture `dev-docs/fixtures/<n>.md` (dual-interface) | #2 | after T1 logic proven |

Blocked behind a *longer-T* / *different-instrument* fixture: **#5** (3-way
L/G/V — 2638-class T cannot separate τ_L from τ_G) and **#6's calibration half**
(instrument-sensitive `Y`-knobs). Do not plan these here.

## Why this matters: the per-dataset ε calibration is the generalisation lever

The Phase 1 work added a `shape_error_epsilon` parameter to
`rescue_and_consolidate` that drives a position-dependent noise floor
inflation under existing strong peaks. The fitted value on the 2638
fixture is `ε ≈ 0.05` (5% per-bin residual at line centre relative to
parent amplitude). This single number is the key per-dataset knob —
*if* the gates and penalties beneath it (AICc-with-`n_eff`, two-tier
merge, etc.) are dataset-invariant, then porting Stage 5 to a new
instrument requires only:

1. Run the chi²_r vs SNR² regression on a few hundred windows of the
   new fixture's persisted fit.
2. Read off ε from the regression slope (with the per-bin vs
   chi²_r-aggregated factor below).
3. Set `shape_error_epsilon` to that value in the new fixture's API
   parameters; ship.

If this turns out to work, it's a major generalisation win. If not —
if the gates also need per-dataset tuning — the work to port a new
instrument grows significantly. The validation plan below is designed
to discover which of the two regimes we're in as early as possible.

## The lineshape model deficit

**Superseded — see §Reconciliation point 1.** The Voigt-like residual described
here did not survive the 655 high-SNR ground truth once the fit engaged
`recommend_shape` (88% Lorentzian). Retained for the reasoning and the residual-
signature diagnostics, which are still a useful *test* — but the conclusion
("treat as irreducible, inflate σ via ε") no longer holds; the residual it
attributed to shape is largely the τ-bias of point 2.

This is the physics finding that motivates the calibration.
Discovered during validation of the AICc-with-`n_eff` merge gate on
the 2638 fixture; see
`scratch/stage5-validation/voigt_hypothesis.py` and the
"Shape-error sigma inflation" section of the residual-rescue
research report.

The Stage 5 model is a finite-T damped cosine — Lorentzian-like
magnitude profile, sharp central frequency. The true molecular
emission profile is **Voigt-like**: a Lorentzian (from transit-time
broadening) convolved with a kinematic-Doppler distribution from the
supersonic expansion's transverse velocity component. The Gaussian
component is *not* Maxwellian — it arises from the cos^N(θ) angular
intensity profile of the molecular beam, and so is geometry-driven
rather than thermal. Per-instrument it depends on:

- Skimmer / nozzle geometry (sets the angular distribution shape)
- Horn coupling vs frequency (sets the effective interaction time)
- Beam-LOS alignment

The signature in the post-fit residual:

1. **chi²_r ~ SNR²**: a model-shape mismatch leaves residual amplitude
   proportional to the parent line amplitude, so chi² contribution per
   bin under the line is proportional to SNR², and chi²_r ~ SNR²
   (modulo a noise-floor intercept ~ 1).
2. **In-phase residual under each line** (after rotating into the
   line's reference frame): negative central / positive shoulder
   structure (second derivative of the model magnitude — the canonical
   Voigt-minus-Lorentzian signature).
3. **Asymmetric Re + opposite-sign Im components** of comparable
   magnitude: the LSQ trades a small frequency offset δf to flatten
   the symmetric Re part, leaving antisymmetric residue split between
   Re and Im channels. Consistent with a *non*-symmetric kinematic
   distribution (cos²θ geometry rather than Gaussian Doppler).
4. **Stable scaling across windows**: per-bin ε is roughly fixed
   per dataset; what varies is the parent line amplitude, so the
   absolute residual magnitude scales linearly with line strength.

A simple Voigt extension to the model would *partly* address this
(captures the symmetric Re part) but not fully (the asymmetric
Re/Im components from non-Gaussian kinematic broadening would still
need a more elaborate model). For now, Stage 5 treats this as
irreducible-by-the-current-model and inflates the noise floor under
existing peaks via `shape_error_epsilon`.

## How to measure ε on a new fixture

**Superseded — see §Reconciliation point 1.** This procedure presumes ε is the
per-dataset generalisation lever; the 655 ground truth indicates it is not.
Theme T4 first re-tests whether `shape_error_epsilon` is needed at all once τ
(T2) and the scatter noise are correct. If T4 finds ε still earns its keep on
some fixture, this measurement procedure is the starting point — until then it
is historical.

Two related but distinct ε values; the relationship between them
matters for porting between fixtures.

### ε_chi²: the chi²_r-aggregated regression slope

The diagnostic in `scratch/stage5-validation/diag_voigt_hypothesis.py`
fits

```
chi²_r_post_rescue = a · SNR² + b
```

across the survey windows (a few-dozen-to-hundreds of windows is
enough). `a = ε_chi²²` and `b ≈ 1` is the noise floor. On the 2638
fixture, the fitted values (post-Phase-1) are:

```
a = 3.16e-4    →   ε_chi² ≈ 0.018  (1.8%)
b = 0.89       (close to the 1.0 noise floor)
```

This regression *is* the per-dataset calibration; the slope is what
shifts when the lineshape model deficit changes.

### ε_per-bin: the sigma-inflation knob

The number actually passed to `rescue_and_consolidate(...,
shape_error_epsilon=...)`. This is the per-bin residual amplitude at
the line centre, NOT the chi²_r-aggregated number. The relationship
is:

```
chi²_r ≈ (ε_per-bin · |model|)² / σ_c²  averaged over peak bins
       ≈ ε_per-bin² · E[|model|²/σ_c²] · N_peak_bins / dof
       ≈ 1.5 · ε_per-bin² · SNR² / dof_typical
```

For 2638 with dof_typical ≈ 100: `ε_chi²² ≈ 1.5 · ε_per-bin² / 100`,
i.e. `ε_per-bin ≈ 8 · ε_chi²`. The empirical sweep on the 2638
fixture (`scratch/stage5-validation/diag_phase1_merge_gate.py`)
landed on `ε_per-bin = 0.05` after exercising the merge-cycle stop
criterion on w148/w269/w198 — about 3× the analytic relation
predicts, suggesting the relation has the right *scaling* but not
exact prefactor (depends on per-window peak count, FWHM-in-bins,
and dof distribution).

**Recommended procedure on a new fixture:**

1. Run the post-Phase-1 rescue on the new fixture with
   `shape_error_epsilon = 0.0` (canonical noise model, no
   inflation). Record the survey chi²_r distribution.
2. Run the diagnostic regression. Fit `chi²_r = a · SNR² + b`.
3. Compute the analytic per-bin estimate `ε_per-bin ≈ 8 · √a`
   (or carry through the exact dof-dependent factor if precision
   matters).
4. Run the harness on the target windows (the analogues of
   w148/w269/w198) with `ε_per-bin` set to the analytic value; sweep
   around it ±50% to find the smallest value that breaks the rescue-
   merge limit cycle on strong-line windows without affecting weak-
   peak rescues.
5. Lock in the swept value as the production setting for that
   fixture.

If the analytic-vs-empirical factor is consistent across fixtures
(say, both ~3× the simple-aggregation estimate), it can be folded
into a single helper that derives `ε_per-bin` from the regression
fit directly. That's a future cleanup — keep the sweep step
explicit until at least two fixtures confirm the relationship.

## Cross-fixture acceptance metrics

When validating Phase 1+ against a new fixture, the deliverables are:

### Tier 1: distribution health (must pass)

A new fixture is "Stage 5 healthy" iff the post-rescue chi²_r
distribution looks like the 2638 reference:

- **chi²_r median** ≤ 1.5 (i.e. the typical window converges near
  noise floor)
- **chi²_r p95** ≤ 4 (extreme cases at most a few times noise)
- **chi²_r max** ≤ 10 (no catastrophic stuck cases — if there are,
  investigate them individually before declaring healthy)
- **No regression on clean controls**: pick the new fixture's
  analogues of w63/w64 (low-K, weak-SNR, well-fit by the persisted
  pipeline) and verify the rescue accepts no candidates.

### Tier 2: gate firing rates

These tell us whether the algorithm is doing meaningful work on the
new fixture, or whether it's mostly idle (the latter being fine for
a fixture whose persisted fit was already clean, suspicious for a
fixture that was over-fitting):

- **Merge fire rate**: fraction of windows where ≥1 merge fires
  during the rescue chain. On 2638 this is ~50%; on a different
  instrument it could legitimately be higher or lower.
- **Rescue-origin failsafe** (`n_pruned_rescue_origin`): how often
  the knockout sweep drops a peak the rescue just added. v1 logs
  this; persistent firing signals a pathological basin the joint
  refit isn't escaping.
- **Limit-cycle indicator**: rounds where chi² doesn't move but the
  rescue still adds peaks and the merge collapses them. Indicates
  the shape-error inflation may be too small for that fixture's
  ε; bump `ε_per-bin` and re-run.

### Tier 3: known-line ground truth (when available)

Where the new fixture has independent line assignments (e.g. from
the original spectroscopy analysis), compare the consolidated peak
list to the assignment list:

- **Recall**: fraction of known lines recovered within ~FWHM/2 of
  their assigned frequency.
- **Precision**: fraction of fitted peaks corresponding to a known
  line (vs spurious / artifact).
- **Amplitude consistency**: for matched peaks, compare
  fitted amplitudes to assignment-list intensities (slope tracks
  any per-fixture intensity calibration; scatter tracks fitting
  precision).

Ground truth is rare for FTMW data, so this tier may not be
available; Tiers 1+2 are the operational acceptance gate.

#### Borderline real-vs-noise: the standing open question

A class of windows on the 2638 fixture sit at the edge of
detectability: the initial fit lands at K=1, the rescue or
seeder nominates a borderline second peak, and the AICc gates
either accept or reject it depending on the exact n_eff
calibration. Canonical examples on 2638: w16, w104, w127,
w337. With the current gate configuration these end at K=1
(post-rescue iterative cleanup rejects the second peak).

The single-fixture data cannot resolve whether the rejected
second peak is a real weak line the gates over-reject, or noise
the gates correctly reject. Tier 3 is the discriminator: if a
new fixture has independent line assignments and the borderline
peaks appear in the reference list, the gates are calibrated too
strictly and the n_eff weighting needs to be loosened; if the
borderline peaks don't appear, the gates are calibrated right.

Per-fixture record this against:

- The list of borderline K=1-vs-K=2 windows in the fixture
  (those whose audit trail shows a candidate that was accepted-
  then-rejected by iterative cleanup, or rejected at the
  conservative-loop gate with `aicc_delta` near zero).
- Each window's AICc gate verdict under the production gate
  configuration; if the audit shows borderline numerics
  (`abs(aicc_delta) < small`), flag for ground-truth check.
- Whether each borderline peak was a Stage 3 candidate
  (suggests detector evidence) vs rescue-only (suggests it's
  chasing residual structure that may be shape-error rather
  than a line).

## What to bring back from a new fixture

Bare minimum:

1. The persisted `.ftmw` file (after running through Stages 0–4).
2. ~10 spot-check windows: 2–3 clean controls (low-K, well-fit), 2–3
   borderline (single weak peak, chi²_r ~ 1), 2–3 strong-line
   (single high-SNR line), 2–3 multi-line / hard cases.
3. Run the post-Phase-1 rescue with `shape_error_epsilon=0.0` to
   get the un-calibrated baseline.
4. Run `diag_voigt_hypothesis.py` (adjusted for the new fixture path)
   to measure ε_chi² and inspect the residual-shape stack.
5. Run `diag_phase1_merge_gate.py` (target windows updated) to
   sweep ε_per-bin and find the cycle-breaking value.

The Tier 1/2 numbers can come back in one report. Tier 3 if
assignments exist.

## Expected / acceptable failure modes per fixture

Not every fixture will Tier-1-pass cleanly. Symptoms and remediation:

### Symptom: chi²_r p95 high (> 5) but distribution otherwise OK

Most likely: shape-error ε is larger than 2638's because the
instrument has stronger kinematic Doppler (different beam geometry).
Re-fit ε_chi² and bump `shape_error_epsilon` proportionally.

### Symptom: chi²_r median > 1.5

The noise model is wrong somewhere upstream — Stage 2 noise
estimator may be giving an under-estimate of σ on the new
spectrum (different baseline structure, different stationarity).
Investigate stage 2 first, not stage 5.

### Symptom: many windows show "stuck-with-shape-error-inflation"

Either (a) ε_per-bin is too small (bump it), or (b) the lineshape
model deficit is structurally different from 2638's (e.g. the new
instrument has a *truly* asymmetric beam profile, not just a small
δf-offset like 2638). Inspect the in-phase / quadrature residual
stack from `diag_voigt_hypothesis.py`; if the Im channel is much
larger than the Re channel, the assumption that "the LSQ absorbs
the asymmetric part as a δf-offset and the in-phase residual is
the symmetric model deficit" is breaking. May need an asymmetric-
Lorentzian model extension; that's structural work, not just
calibration.

### Symptom: gate fires aggressively (merges in every window)

Either the persisted Stage 4 fit was wrong (too many initial
peaks per window — investigate the Stage 4 plan), or the new
fixture has many naturally-occurring close pairs (e.g. hyperfine
multiplets in a different molecular system). Inspect a few
specific windows; if the merges are over-aggressive on real close
pairs, the structural-merge factor (currently 0.5 FWHM) may need
loosening for that fixture.

### Symptom: weak-peak rescues fail

If w16/w104/w127/w337 analogues lose their borderline second
peaks under Phase 1+1b+1c, the screening pipeline is too strict.
Check whether ε_per-bin was set too high (it inflates noise floor
even at low-SNR sites if those sites happen to sit under some
weak existing model — though this is rare).

## Per-fixture record-keeping

For each fixture validated, capture the following in a per-fixture
artifact under `dev-docs/fixtures/<fixture-name>.md`:

- Origin: instrument, date, sample, person.
- Stages 0–4 parameters used (settings.json snapshot).
- Phase 1 calibration:
  - `ε_chi²` from the regression
  - `ε_per-bin` from the sweep
  - `n_eff_kind` choice
  - `structural_merge_factor` if non-default
  - `tau_consensus_us` and `tau_consensus_window_count` (the
    median of strong-window tau values and how many windows
    contributed)
- Tier 1 distribution numbers (post-Phase-1).
- Tier 2 firing rates.
- Tier 3 ground truth comparison if available.
- Per-window detail for the spot-check windows.

This artifact is the source of truth for "what does this fixture
look like under the current Stage 5 pipeline" and becomes the
regression baseline for future Stage 5 changes.

## Cross-fixture roll-up

Once two or more fixtures are characterised, summarise the
calibrated parameters per fixture in a single table:

Post-reconciliation the per-fixture invariants worth tracking are the **τ
source/value**, the **σ_f accuracy floor**, and the Tier-1 χ²ᵣ health — not ε
(retained as a column only if T4 keeps it). `n_eff_kind` is now globally
`perplexity_log1p_snr` (the `kish_mag` label is stale — see the
`neff-kind-perplexity-global` memory), so it is no longer a per-fixture knob.

```
fixture  instrument  tau_src      tau_us  sigma_f_floor_khz  chi2r_p50  notes
2638     BlackChirp  stage2b      ~3      (tbd)              ~1.3       reference
655      BlackChirp  tau_G/env    ~4.0    8.9                (tbd)      STFT biased low
1512     BlackChirp  (tbd)        ~8.3?   6.6                (tbd)      STFT biased high
<next>   ...         ...          ...     ...                ...        ...
```

The cross-fixture pattern is what tells us whether the gate-design
work is dataset-invariant. Two reasonable outcomes:

- **`ε` varies, everything else is constant.** The big win: per-
  fixture porting is a one-number calibration. Document the
  procedure and ship.
- **Multiple parameters drift per fixture.** Phase 1 was over-fit
  to 2638. Need to either find a more invariant formulation or
  accept that Stage 5 needs per-instrument calibration. Hopefully
  not, but the validation framework above will tell us either way.

## Dataset-wide tau calibration (related deferred work)

**Superseded** by [`stage2b-tau-calibration.md`](stage2b-tau-calibration.md):
the data-driven `τ_maj ± σ_τ` STFT calibration with per-band routing and a
bidirectional Gaussian τ penalty is the production answer to the per-window-τ
inconsistency described below. Retained for the reasoning.

Independent of the shape-error epsilon calibration but
conceptually adjacent: tau (the shared decay constant in the
finite-T line shape model) is currently fit per-window. Strong-line
windows converge to consistent values (~3 µs on 2638); weak windows
hold tau at the apodization ceiling because there's no information
to fit it from a few-σ peak. The weak-window behaviour is a bias,
not a fit.

A natural cleanup: after the per-window fits converge, compute a
consensus tau from the strong-window distribution (median, or
amplitude-weighted median, of windows where `tau_error <
tolerance`), then re-fit weak windows with tau locked at the
consensus. Improves weak-window amplitude/offset precision without
changing the strong-window results.

Per-fixture relevance: each instrument's beam geometry and natural
linewidth determine its consensus tau. Recording this number per-
fixture gives another simple invariant to track across fixtures —
if a new fixture's strong-window tau distribution looks very
different from 2638's, that's a signal about the instrument setup
worth investigating before assuming Phase 1 settings transfer.
Add `tau_consensus_us` and `tau_consensus_window_count` to the
per-fixture record-keeping below.

### Broken-initial-fit pathology and the majority-vote-freeze proposal

**Superseded** by [`stage2b-tau-calibration.md`](stage2b-tau-calibration.md):
the data-driven τ anchor + bidirectional Gaussian penalty addresses the
tau-collapse pathology described here, rather than this section's post-hoc
two-pass `tau_locked` consensus freeze (which was not implemented). Retained
for the reasoning.

The 2638 fixture's w198 surfaced a sharper version of the weak-window
tau bias: when the initial fit cannot represent the true number of
peaks (Stage 3's candidate offsets under-count the window's actual
K), LSQ narrows tau to broaden each modelled peak to absorb the
unmodeled-peak residual. The K=2 initial fit on w198 pegs tau at
1 µs (the apodization-override lower bound) instead of the dataset
consensus of ~3 µs. The rescue chain's joint refit warm-starts
from the apod-override at 5 µs but converges to ~2.4 µs — a
closer-but-still-wrong basin. The K=7 outcome at tau ≈ 3 µs is
the right answer; the K=3 outcome at tau ≈ 2.4 µs (the current
production result) is a different LSQ basin altogether.

This argues for **freeze-at-consensus** rather than soft re-fitting
on the weak windows the prior section covers: when a window's
initial fit lands far from the dataset consensus (e.g. > 2σ away),
lock tau at the consensus for all subsequent passes on that window
— initial fit, rescue's conservative loop, joint refit, knockout,
merge. The free-fit basin LSQ falls into when the initial K is
wrong is a pathological local minimum; locking tau keeps LSQ in
the physical basin and lets the rescue do its job.

Implementation sketch:

1. Run all per-window fits with tau free (current behaviour).
2. Compute the consensus tau and its variance from windows where
   `tau_error / tau < threshold` (e.g. 5%).
3. Identify windows whose fitted tau is > Nσ from the consensus.
4. Re-fit those windows with `fit_tau=False, tau0_us=tau_consensus`
   end-to-end — applied uniformly to the initial fit, the rescue's
   internal `conservative_fit`, and the rescue's joint refit
   (currently the joint refit thaws tau regardless).
5. Iterate (step 2 may shift consensus slightly after the re-fits).

The implementation involves threading a `tau_locked: bool` flag
through `rescue_and_consolidate` so the joint refit honours it
on flagged windows, and extending the orchestrator in
`_internal/stage5_impl.py` with the two-pass logic. Resolves the
w198 follow-up open after the residual-rescue restructure.

## Next steps

Superseded by the post-reconciliation themes (T1–T5) above. The original
sequence below is retained for the different-instrument arc, which is still
blocked.

1. ~~Acquire a second fixture from the same instrument.~~ *(Done — #1 closed;
   seven same-instrument fixtures, two with ground truth.)*
2. Acquire a fixture from a **different** instrument (longer `T` especially).
   Tests the actual generalisation hypothesis and unblocks the 3-way L/G/V
   shape test (#5) and the instrument-`Y`-knob calibration half (#6). **Still
   blocked** — no such fixture in hand.
3. ~~Move ε into a self-tuning `SpectrumFit.parameters` entry.~~ *Reconsidered
   — Theme T4 first decides whether ε survives at all (655 indicates it may
   not); do not build self-tuning for a knob that may be retired.*
4. Land the cross-fixture validation harness as a CLI subcommand
   (`ftmwpipeline validate-stage5-shape-error`) — Theme T5, after the T1 harness
   logic is proven across the seven fixtures.
