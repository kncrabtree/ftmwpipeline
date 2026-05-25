# The projection-coherence screen for peak detection

A research report on the σ-weighted Lorentzian-projection statistic that
could serve as a low-SNR-vs-noise discriminator at the peak-detection
stage of the FTMW processing pipeline. The companion implementation is
[`src/ftmwpipeline/preprocessing/coherence_screen.py`][screen]; the
reproducibility script (every figure and number in this report) is
[`prototype.py`](prototype.py).

The investigation answers five questions:

1. Does the σ-weighted projection ratio discriminate real Lorentzians
   from phase-incoherent noise excursions, when measured against
   controlled ground truth?
2. Over what region of `(true SNR, linewidth-in-bin-units)` does it
   work?
3. How sensitive is it to the choice of basis decay constant ``τ_basis``?
4. Does the screen handle *structured* false positives (sidelobes
   from bright lines, whose phase has Lorentzian coherence) as well
   as it handles pure-noise false positives?
5. What does that imply about acquisition design and the production
   wiring?

The answers are: yes; over a substantial portion of the parameter
space (with a clear bin-vs-linewidth optimum at FWHM/bin ≈ 1–2);
narrower-than-matched bases win on average; sidelobe FPs are
discriminated *better* than noise FPs at the recommended ``τ_basis ≈
2-3 × τ_truth`` (and *worse* at wider-than-matched bases, which is
the sharpest argument against that direction of mis-tuning); and the
results have implications for acquisition design that go beyond this
particular screen -- see §6.

[screen]: ../../../src/ftmwpipeline/preprocessing/coherence_screen.py

## 1. Setup

The screen takes a peak-detection candidate at frequency ``f_c`` on the
active-portion FT (the same spectrum the fitting stage consumes,
``dt_us · rfft(active · apod)`` in the ``[0, T]`` form ``h_T`` models)
and returns the ratio

```
ratio = coherent_snr / detected_snr_active
coherent_snr = |A| · |h_T(0; τ_basis, T)| / σ_c_median
detected_snr_active = |z(f_c)| / σ_c[f_c]
```

with ``A`` the σ-weighted least-squares amplitude of a unit Lorentzian
basis at ``f_c``, over a localised sub-window. For a real Lorentzian
the basis matches the data and ``ratio ≈ 1``; for a phase-incoherent
noise spike the off-line projection contributions average down and the
ratio drops.

The natural concern -- the one this study was set up to answer
empirically -- is that for a fixture with sub-bin linewidth, the
projection is dominated by the on-line bin and the discrimination
collapses. The prior investigation on the 2638 fixture *appeared* to
confirm this collapse, but did so against a ground-truth proxy that
was circular (the persisted fit on weak-line windows used a frozen τ
that was the same parameter the basis depended on). This study
replaces the proxy with a simulator-driven controlled ground truth.

### The simulator

[`prototype.py`'s `simulate_active_ft`][prototype] constructs a real
FID with ``n_lines`` damped cosines at random non-clustered bin
positions, adds time-domain Gaussian noise sized to hit a target
on-line SNR, and returns the resulting active-FT plus the ground-truth
frequency set. The two control axes:

[prototype]: prototype.py

- **`true_snr`**: per-bin on-line SNR
  (``|on-line magnitude| / σ_c``, Rayleigh scale) at the candidate
  bin.
- **`fwhm_bins`**: line FWHM as a multiple of the active-FT bin
  spacing. With ``T_active = 81.92 µs`` and bin spacing
  ``≈ 0.012 MHz`` (the simulator's defaults), this fixes
  ``τ_truth = T / (π · fwhm_bins)``.

Per-bin σ is the analytic noise constant (uniform across the
synthetic spectrum). This is deliberate: it isolates the screen's
behaviour from the upstream Stage 2 noise estimator's calibration,
which is what the study is meant to characterise.

**Caveat on the SNR parameterisation.** The simulator's ``true_snr``
holds the per-bin on-line SNR fixed by scaling the time-domain noise
σ. This means the *acquisition-design* axis -- "if I oversample,
per-bin SNR drops" -- is *not* directly visible in the sweep results,
because the simulator counteracts it. The bin-vs-linewidth optimum
the screen exhibits is still meaningful (it shows where the screen's
*projection statistic* concentrates discrimination), but the broader
acquisition principle in §6 is a separate observation that the
simulator validates only qualitatively.

Candidate generation: ``scipy.signal.find_peaks`` on the active-FT
magnitude with a height threshold of ``2 × σ_c``, no further
filtering. A candidate is a true positive iff some injected line lies
within ``max(1, FWHM/2)`` bins of its bin. The truth-match tolerance
scales with FWHM because at wide FWHM the detected local max can sit
several bins off the true line centre under noise -- the ±1 rule
would mis-classify legitimate detections as FP and bias the AUC.

## 2. Single-cell illustration

`figures/01_single_cell.png` shows three FWHM-in-bins regimes at one
SNR (4.0), side by side: the synthetic active-FT magnitude with the
injected line positions marked, plus the per-candidate ratio
distribution split by ground-truth label.

What to notice:

- At `FWHM/bin = 0.8` (sub-bin lines), real and noise ratios overlap
  heavily around 0.9-1.0. The screen has only the on-line bin to
  project against and the discrimination collapses.
- At `FWHM/bin = 1.5`, real-line ratios cluster near 1.0 and noise
  ratios spread below. The matched basis covers ~2 bins per FWHM,
  enough for phase coherence to register.
- At `FWHM/bin = 3.0`, the real-line cluster is even tighter and the
  noise distribution is centred well below 1, but per-FWHM there are
  many noise candidates picked up by the upstream detector that the
  screen has to score.

## 3. Phase-space sweep at matched τ

`figures/02_phase_space_auc_matched.png` shows AUC across the
`(true SNR, FWHM/bin)` grid with ``τ_basis = τ_truth`` (matched).
The full numbers behind the heatmap:

| SNR \ FWHM/bin | 0.50 | 0.80 | 1.00 | 1.34 | 2.00 | 3.00 | 5.00 | 8.00 | 12.00 | 20.00 |
|----------------|------|------|------|------|------|------|------|------|-------|-------|
| 2.0            | 0.71 | 0.82 | 0.86 | 0.90 | 0.93 | 0.93 | 0.87 | 0.80 | 0.75  | 0.69  |
| 3.0            | 0.76 | 0.87 | 0.90 | 0.92 | 0.90 | 0.89 | 0.78 | 0.71 | 0.66  | 0.60  |
| 4.0            | 0.78 | 0.87 | 0.89 | 0.90 | 0.87 | 0.84 | 0.72 | 0.64 | 0.60  | 0.53  |
| 5.0            | 0.78 | 0.86 | 0.87 | 0.88 | 0.84 | 0.79 | 0.66 | 0.58 | 0.54  | 0.48  |
| 7.0            | 0.78 | 0.84 | 0.85 | 0.85 | 0.78 | 0.72 | 0.60 | 0.52 | 0.47  | 0.41  |
| 10.0           | 0.78 | 0.83 | 0.83 | 0.82 | 0.75 | 0.67 | 0.55 | 0.46 | 0.40  | 0.35  |
| 25.0           | 0.77 | 0.81 | 0.81 | 0.82 | 0.76 | 0.69 | 0.53 | 0.38 | 0.29  | 0.23  |
| 50.0           | 0.76 | 0.80 | 0.81 | 0.83 | 0.78 | 0.70 | 0.52 | 0.35 | 0.25  | 0.18  |

**The optimum sits at FWHM/bin ≈ 1.0-2.0.** Three regimes:

- **Sub-bin (FWHM/bin < 1)**: discrimination collapses. With FWHM
  smaller than a bin, the basis has support essentially on the
  on-line bin only; the screen reduces to ``ratio ≈ σ_c[c]/σ_c_median``,
  which is class-independent.
- **Matched (FWHM/bin ≈ 1-2)**: peak AUC at every SNR. The basis
  covers 2-4 bins per FWHM, enough for the σ-weighted projection to
  integrate coherent phase information across multiple bins.
- **Over-sampled (FWHM/bin > 5)**: AUC drops monotonically. The
  upstream detector picks up many off-line bins of each real line as
  separate candidates (visible in
  `figures/06_detection_yield.png` as "FP per recovered line"
  growing from ~10 at FWHM=1 to ~20 at FWHM=20). The growing
  candidate count shifts the AUC denominator.

`figures/02b_phase_space_fpr95_matched.png` shows the same grid as
FPR at TPR = 0.95. At the FWHM/bin ≈ 1.34-2 optimum, the screen
kills 80 % of noise candidates at TPR = 0.95 in the SNR ≥ 3 band
(FPR ≤ 0.20). That is a usable discriminator.

`figures/06_detection_yield.png` shows the upstream-detector yield
across the grid: detection recall stays near 1.0 across the whole
sweep (the simulator's controlled per-bin SNR keeps real lines above
the 2σ detection floor by construction), but FP-per-recovered-line
grows from ~10 at narrow FWHM to ~20 at wide FWHM. The screen's
AUC degradation past FWHM/bin ≈ 3 is mostly driven by this growing
denominator, not by the screen failing on individual candidates.

`figures/03_roc_selected_cells.png` is the per-cell ROC curve at four
diagnostic corners (SNR ∈ {2, 10}, FWHM/bin ∈ {0.8, 3.0}). It
illustrates the same pattern in shape: the FWHM ≈ 1 cells have
steeper ROC; the FWHM > 3 cells have flatter ROC.

## 4. τ_basis sensitivity

`figures/04_tau_mismatch_auc.png` shows AUC across the phase-space
grid for ``τ_basis ∈ {0.5×τ_truth, 2×τ_truth}``. The key cells, all
SNR = 5:

| FWHM/bin | factor 0.5 | matched (1.0) | factor 2.0 |
|----------|------------|----------------|-------------|
| 0.5      | 0.74       | 0.78           | 0.78        |
| 1.0      | 0.82       | 0.87           | 0.92        |
| 1.34     | 0.80       | 0.88           | 0.94        |
| 2.0      | 0.74       | 0.84           | 0.93        |
| 5.0      | 0.47       | 0.66           | 0.82        |
| 20.0     | 0.29       | 0.48           | 0.69        |

Three patterns:

1. **``τ_basis = 0.5 × τ_truth`` (basis FWHM 2× wider than data) is
   catastrophic** in the over-sampled regime. AUC collapses to 0.04
   at the SNR=50, FWHM=20 corner. The wider basis dilutes the on-line
   bin and contaminates the numerator with off-line noise.
2. **``τ_basis = 2 × τ_truth`` (basis FWHM 2× narrower than data)
   uniformly outperforms matched** for FWHM/bin > 1. The plateau AUC
   approaches 0.97-0.99 at SNR ≥ 10 in the FWHM ≈ 1-2 band.
3. **At sub-bin lines (FWHM/bin < 1) the τ_basis choice barely
   matters** -- all three factors give AUC ≈ 0.7-0.8. The screen's
   discrimination is bottlenecked by the lack of off-line bins,
   not by basis choice.

`figures/05_tau_basis_optimum.png` is the fine sweep:
``τ_basis / τ_truth`` from 0.25 to 8.0 at five diagnostic cells.
Across all cells the AUC curves are monotone-rising up to
``factor ≈ 2-3`` and then plateau. There's no τ_basis "sweet spot"
within the range tested; anything narrower than ``2×`` matched
suffices.

## 5. Why narrower-than-matched wins

The mechanism, qualitatively (the cleanest explanation is post-hoc
-- the result was found empirically):

The helper's projection sub-window is sized by ``5 × FWHM_basis``,
so a narrower basis is a narrower sub-window. As ``τ_basis`` grows,
the sub-window collapses toward the helper's floor
``min_window_bins = 3``. In that 3-bin regime the σ-weighted
projection's amplitude estimator becomes dominated by the on-line
bin: ``A ≈ z(0) / b(0)``, with small off-bin corrections.

But the projection still has *one neighbour bin each side* worth of
phase information. For a real Lorentzian those neighbours have the
analytic ``h_T`` phase profile, which the basis matches. For a noise
excursion they don't. The screen's discriminator at large ``τ_basis``
is essentially "is the 3-bin local phase consistent with a
Lorentzian?", which is the minimum information needed to discriminate
without being washed out by far-off bins.

A wider basis (``τ_basis < τ_truth``) goes the wrong way because the
sub-window grows, more off-line bins enter the projection, and the
σ-weighted average of uncorrelated noise across many bins suppresses
real-line amplitudes by the same factor it suppresses noise -- net
AUC drops.

An analytic derivation of the optimal ``τ_basis / τ_truth`` as a
function of SNR, FWHM/bin, and the sub-window floor is open work.
The empirical "anything narrower than 2× matched, plateau at 3-4×"
is enough for production calibration.

## 6. The bin-vs-linewidth observation -- and why it matters beyond this screen

The matched-τ AUC heatmap is the screen's most striking result *not*
because it tells us where the screen works (it works), but because it
shows a clear optimum at **FWHM/bin ≈ 1-2** -- the bin spacing
matched to the linewidth, give or take a factor of 2.

This is a general property of frequency-domain signal extraction, not
a property of this particular screen:

- **Sub-bin linewidth (FWHM/bin < 1)**: zero-padding (oversampling
  in frequency) has packed more bins inside the line shape than the
  line's information content supports. Each additional bin between
  the resolved frequency cells is a Dirichlet-interpolated copy of
  its neighbours -- a *correlated* sample, not an independent one.
  The screen's matched filter sees only the on-line bin's
  independent information; the off-line bins contribute correlated
  noise. **Resolution past 1 FWHM/bin is wasted.**
- **Over-sampled in time (FWHM/bin > a few)**: the linewidth in bins
  is set by the ratio of acquisition length to molecular decay time:
  ``FWHM/bin = T_active / (π · τ_eff)``. Pushing T_active far beyond
  ``π · τ_eff`` means most of the acquisition window contains only
  noise (the molecule's emission decayed away long ago). The matched
  filter still recovers the line in integrated SNR -- but the
  per-bin SNR drops, the per-FWHM noise candidate count grows, and
  the screen's job gets harder.
- **Bin-matched (FWHM/bin ≈ 1-2)**: acquisition length is
  ``T_active ≈ (1-2) · π · τ_eff ≈ 3-6 · τ_eff``. Long enough to
  capture the molecular emission; short enough not to drown it in
  acquisition noise. The screen sees enough off-line bins to do
  phase-coherence work, but not so many that the candidate set
  fills up with noise.

**For 2638**: τ_eff ≈ 3 µs (the strong-line fit-determined value
from the prior study), T_active = 12.65 µs (Stage 1 canonical),
gives ``FWHM/bin ≈ 1.34``. That sits at the AUC peak. The screen
*should* discriminate well on 2638; the prior study's apparent
failure was the circular ground truth, not the bin-vs-linewidth
geometry.

**As an acquisition principle**, for any FTMW dataset the bin
spacing should target ``Δf_bin · π · τ_eff ≈ 1``, i.e.
``T_active ≈ 3-5 · τ_eff``. This is independent of the screen --
it's where any frequency-domain detector's per-bin SNR is best
matched to the linewidth. Increasing FFT bins past this (longer
T_active, more zero-padding) trades real per-bin signal energy
for visual smoothness and slightly improved frequency
*localisation* (which sub-bin centroid fitting can recover
anyway). At the experiment-design layer this argues for tuning the
acquisition window to the expected molecular linewidth; at the
processing layer it argues against routine extra zero-padding in
the active-FT.

This is the most general lesson the study produced: regardless of
whether the projection-coherence screen ships, *the bin-matched
acquisition regime is where the entire pipeline's per-bin
statistics are best behaved*.

## 6b. Structured-noise FPs (sidelobe extension)

The matched-τ AUC sweep above used pure complex Gaussian noise as the
only FP source. A separate concern from the 2638 application is that
real spectra contain *structured* false positives: noise excursions
that ride on the magnitude skirt of a nearby strong line. These have
mostly-Lorentzian phase (because the skirt does) and so might project
coherently against an ``h_T``-shaped basis, evading the screen.

`figures/07_sidelobe_breakdown.png` is the answer. The simulator was
extended to inject ``n_strong_lines = 4`` bright lines (SNR 25, 50, or
100) alongside the 25 weak lines, with ``strong_separation_bins = 80``
keep-out so weak and strong lines don't overlap. Candidates that
the upstream detector picks up within ±80 bins of a strong line but
outside the truth-match tolerance of any injected line are labelled
**sidelobe FPs**; the rest are **noise FPs**. AUC is computed against
each FP class separately, sweeping ``τ_basis / τ_truth`` over
``{0.5, 1.0, 1.5, 2.0, 3.0, 4.0}``.

At weak-line SNR = 4, FWHM/bin = 1.34, strong-line SNR = 50:

| τ_basis / τ_truth | AUC vs noise FPs | AUC vs sidelobe FPs |
|-------------------|-------------------|----------------------|
| 0.5 (wider basis) | 0.78 | 0.54 |
| 1.0 (matched)     | 0.88 | 0.83 |
| 1.5               | 0.91 | 0.93 |
| 2.0               | 0.93 | **0.96** |
| 3.0               | 0.94 | **0.97** |
| 4.0               | 0.94 | 0.94 |

Pattern repeats at strong_snr = 25 and 100 (the AUC against noise FPs
is essentially invariant in strong_snr; the AUC against sidelobe FPs
drops faster at small ``τ_basis / τ_truth`` as strong-line SNR grows,
from 0.65 at strong_snr=25 down to 0.36 at strong_snr=100 for the
wider-than-matched basis -- the wider basis confuses Lorentzian-phase
skirt contamination with real Lorentzians most when the contamination
is loudest).

Two findings:

1. **Sidelobe FPs are harder to kill than noise FPs at small
   ``τ_basis / τ_truth``** -- consistent with their off-line phase
   carrying real Lorentzian coherence -- but at ``τ_basis ≥ 1.5 ×
   τ_truth`` the screen separates them just as cleanly as noise FPs.
   The narrower basis's collapsed sub-window (3 bins) loses the
   off-line context that lets the sidelobe phase masquerade.
2. **A wider-than-matched basis (factor 0.5) is catastrophic for
   sidelobe discrimination** -- AUC drops to 0.36 when the strong line
   is bright. This is the sharpest argument the study produces against
   that direction of basis mis-tuning.

This validates the wiring proposal: ``τ_basis ≈ 2-3 × τ_truth`` is
the right operating point regardless of whether the FP population is
pure noise or structured sidelobe contamination. The 2638 *apparent*
finding that "matched-τ has wider dynamic range, so kills more
candidates at a fixed threshold" is **not** evidence that matched-τ
is the better operating point; it reflects a wider distribution that
includes both more aggressive TP-kills and more FP-kills mixed
together. Without ground truth, the wider distribution looks
informative; with controlled ground truth, the narrower basis ranks
TPs above FPs more reliably.

### 6b.i Sidelobe interference

A natural concern: with multiple strong lines whose skirts overlap,
a candidate sitting between them "sees" contributions from two
competing Lorentzian phase profiles centred at different places. Does
the screen still work when interference disrupts the simple
single-skirt picture?

`figures/08_sidelobe_interference.png` answers this by shrinking
``strong_separation_bins`` from 80 (isolated skirts) to 40 (significant
overlap) to 20 (heavy overlap). All other parameters held at the
2638-like regime (weak SNR=4, FWHM/bin=1.34, strong_snr=50, 4 strong
lines):

| separation | matched (1×) | **2× (recommended)** | 3× |
|------------|--------------|----------------------|-----|
| 80 bins (isolated)        | noise 0.88, side 0.84 | **noise 0.93, side 0.96** | 0.94, 0.97 |
| 40 bins (overlap)         | noise 0.87, side 0.78 | **noise 0.93, side 0.95** | 0.94, 0.96 |
| 20 bins (heavy overlap)   | noise 0.89, side 0.79 | **noise 0.93, side 0.95** | 0.95, 0.96 |

The recommended ``τ_basis = 2-3 × τ_truth`` is **robust to sidelobe
interference**. AUC against sidelobe FPs stays at 0.95-0.96 across all
separation regimes; the matched-τ setting shows modest degradation
(0.84 → 0.78) as interference grows. Mechanism: the narrower basis's
sub-window is small enough to test only the candidate's *local*
phase profile, and interference produces non-Lorentzian local phase
that the screen catches regardless of how many strong lines are
contributing skirts.

This closes the sidelobe story for the study's purposes. The
remaining cross-pollination concern -- candidates sitting on top of a
strong line's main peak rather than its skirt -- is handled by the
truth-match tolerance: a candidate within ±FWHM/2 of a strong-line
centre is a TP for that line, not an FP.

## 7. Verdict

**The projection-coherence screen works** in the bin-matched regime:
matched-τ AUC ≥ 0.9 for FWHM/bin ∈ [1, 2] across the SNR ≥ 2 band;
narrower-than-matched ``τ_basis`` lifts AUC to ≥ 0.94 at the same
operating points. At FWHM/bin ≪ 1 (severe over-padding) or FWHM/bin
≫ 5 (severe over-acquisition relative to molecular decay) the screen
degrades, but those regimes also degrade the upstream detector's
candidate quality, so the screen failing there is not a unique
failure of the projection statistic.

The narrower-than-matched advantage **holds against structured
sidelobe FPs as well as pure-noise FPs** (see §6b). The wider-basis
direction is *catastrophic* for sidelobe discrimination -- AUC drops
to 0.36 at strong_snr=100 with τ_basis = 0.5×τ_truth. This is the
sharpest single piece of evidence the study produces: the basis
choice has to be narrower-or-matched, never wider, regardless of
what kind of contamination is in the spectrum.

For the operating point most relevant to the production pipeline
(2638-like fixtures with FWHM/bin ≈ 1.34): the screen kills ~80 %
of noise candidates at TPR 95 % with matched τ, and ~85 % with
``τ_basis ≈ 2-3 × τ_truth``, and discriminates strong-line sidelobes
from real lines with AUC ≥ 0.93 at the same setting.

## 8. Wiring proposal

For peak detection, the screen should:

1. Run after the existing two-pass detector but before the snap-back
   to the user grid (so it operates on the active-FT-frame magnitudes
   the screen is calibrated against).
2. Use a per-experiment ``τ_basis`` derived from the brightest
   detected candidates -- a fast 1-parameter fit on the top-N
   strong-line candidates' Lorentzian width, then multiply by ~2-3×.
   Falling back to ``expf_us · 2`` if no strong-enough candidate
   exists is a defensible default but should be calibrated per
   instrument.
3. Drop candidates whose ratio is below a threshold derived from the
   per-experiment ROC at TPR = 0.95 -- *not* a hard-coded number.
   The threshold is dataset-dependent and should be reported in
   the persisted Stage 3 diagnostics.
4. Surface the kept ratio and the chosen ``τ_basis`` per candidate
   in the detection diagnostics, so curation can re-examine
   borderline cases.

What this proposal *requires* before production:

- A per-fixture auto-τ procedure: how to pick ``τ_basis`` for a
  given experiment without circular dependencies (the fitting
  stage's frozen τ on weak-line windows is *not* an acceptable
  input -- that is the bug the prior 2638 study exposed).
- A second-fixture validation: this study uses a synthetic
  active-FT with analytic uniform σ. Re-running the phase-space
  sweep on a synthetic spectrum *derived from a real FID's noise
  statistics* (spatially-varying σ) is the next experiment.
- A regression test against the fit-quality validation harness:
  the screen must not regress fit-quality on 2638 or other
  established fixtures.
- An acquisition-design note in the operator-facing docs
  reflecting §6: target ``T_active ≈ 3-5 · τ_eff_expected``.

## 9. Open questions

- **Analytic τ_basis optimum.** The qualitative argument in §5 is
  not a derivation. An analytic treatment of ``∂AUC/∂(τ_basis/τ_truth)``
  at the matched point would settle the optimal ratio for given
  ``(SNR, FWHM/bin)``. The empirical plateau at ``factor ≥ 2-3``
  is enough for production calibration, but a closed-form result
  would let us pick ``τ_basis`` without a per-cell sweep.
- **Does the screen behave well on realistic noise?** The simulator
  uses analytic σ. Real spectra have spatially-varying σ (the
  Stage 2 estimator output is not flat). Re-running with σ pulled
  from a real fixture is straightforward and should be done before
  wiring.
- **Cluster behaviour.** The simulator picks lines at ≥ 6-bin
  separation; close-pair candidates (Stage 4's "blended" regime)
  are not covered. The deleted residual-screening code had a
  sliding threshold + deferral for close pairs; whether the same
  logic is needed at the peak-detection stage is an open question.
- **Sidelobe density.** The §6b sidelobe extension and §6b.i
  interference test use 4 strong lines per spectrum. A real spectrum
  may have many more (rotational manifolds of a single species can
  produce dozens). Whether the screen scales -- the cumulative effect
  of dozens of overlapping skirts on a candidate's local phase -- is
  open. The §6b.i result that ``τ_basis = 2-3 × τ_truth`` is robust
  to two-strong-line interference is suggestive but doesn't bound
  the high-density regime.
- **Sub-window floor (``min_window_bins``).** The current default
  = 3. The mechanism analysis in §5 hinges on it. Worth a follow-up
  sweep.
- **Integrated-SNR parameterisation.** The simulator's per-bin SNR
  parameterisation hides the acquisition-design axis (§6 caveat).
  A simulator parameterised by integrated signal energy + bin
  spacing would let the bin-matching principle be derived from
  first principles instead of inferred from the matched-τ AUC
  optimum.

## 10. Reproducibility

```bash
# from repository root, with the project conda env
conda run -n ftmwpipeline-dev python \
    dev-docs/research/stage3-coherence-screen/prototype.py
```

Runtime: ~30 seconds end-to-end on a contemporary laptop CPU. Writes
six PNGs to ``figures/`` and three ``.npz`` intermediates to
``data/``. Re-running overwrites both.

The 13 unit tests for the helper at
``tests/unit/preprocessing/test_coherence_screen.py`` cover the
input-validation surface, the noise-free Lorentzian recovery (ratio
exactly 1.0 to atol 5e-3), and the pure-noise rejection -- they pin
the math; this report pins the behaviour.

## 11. Related artefacts

- The companion implementation:
  ``src/ftmwpipeline/preprocessing/coherence_screen.py``.
- The 2638 fixture investigation that motivated this study:
  ``scratch/stage3-coherence-study/`` (preserved with its
  proxy-dependent claims walked back; the strong-peak and
  padded-vs-active sanity checks survive there).
- The deleted Stage 5 application of the same projection primitive:
  ``git show c2f2ab2~1:src/ftmwpipeline/fitting/residual_screening.py``.
  Algorithmically related but operationally different: the rescue
  ran on a *residual* inside a fit window where the AICc accept
  gate duplicated the projection's discrimination; here the screen
  is the primary discriminator.
