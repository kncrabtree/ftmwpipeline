.. index::
   single: peak fitting; acceptance gate
   single: context-invariant gate
   single: penalized chi-squared
   single: effective sample size
   single: F-test; denominator bias
   single: blend recovery
   single: residual rescue
   single: SNR-aware acceptance
   single: shape-error fraction

The Stage 5 fit: gates, blends, and SNR-aware acceptance
========================================================

This note explains the statistical machinery behind :doc:`Stage 5
<../stage5_fitting>` — how the conservative add-one-peak loop decides a line is
real, why that decision is *immune to window size*, how it handles blended
lines, and on what basis the converged fit can be trusted across three decades
of signal-to-noise. The stage page describes the mechanisms at user altitude;
this note derives the acceptance gate, shows the failure mode it was built to
avoid, and validates the whole stage on synthetic ground truth and the seven
checked-in experiments. The figures and the numbers are regenerated from the
shipped fitting code by ``docs/source/methods/stage5_fitting/generate.py``; see
:ref:`stage5-fitting-reproducing`.

The decision the gate has to make
---------------------------------

Each window is fit by growing its model one line at a time, and every step asks
the same question: does the data demand another line, or is the apparent
improvement just the extra freedom of a larger model fitting noise? Getting that
gate right is the whole game. Too lenient and the fit sprouts spurious lines on
every strong line's skirt and every noise excursion; too strict and it misses
the weak lines that detection worked to find. The gate must also behave the same
way in a tight two-bin feature and in a wide window padded with hundreds of
quiet bins — otherwise the *windowing* decisions of :doc:`Stage 4
<../stage4_windows>` would silently change which lines survive.

The natural first choice — and the one the early prototype used — is the
classical nested-model **F-test**: the more-complex model adds parameters and
lowers the chi-squared, and the F-statistic

.. math::

   F = \frac{\Delta\chi^2 / \Delta k}{\chi^2_{K+1} / (n_\text{data} - k_{K+1})}

with its p-value gates the addition. This is the textbook test, and it has a
structural bias that makes it the wrong tool here.

The window-size bias
--------------------

The numerator of :math:`F` is the *local* evidence for the candidate — the
chi-squared drop :math:`\Delta\chi^2`, which lives in the handful of bins the new
line informs. The denominator is the more-complex model's reduced chi-squared
over the **whole window**. On a real spectrum a converged window is dominated by
quiet bins sitting at the noise, so that denominator is driven toward one as the
window grows — and the F-distribution's second degree of freedom grows with it,
concentrating the distribution. Both effects push the same way: for a *fixed*
amount of local evidence, padding the window with quiet noise lowers the
p-value. Significance is manufactured from empty bins.

.. figure:: stage5_fitting/figures/fig1_gate_window_invariance.png
   :width: 80%
   :align: center

   F-test p-value against window size for two candidates of fixed local
   evidence, padded with progressively more quiet bins. A marginal candidate
   that is absorbing a bright line's irreducible lineshape misfit (red) starts
   right at :math:`\alpha = 0.05` in a tight window and is driven to high
   significance purely by padding — the F-test would accept it on a wide window.
   The shipped penalized gate compares the same local :math:`\Delta\chi^2` to a
   fixed bar and returns the same verdict (reject the absorber, accept the real
   line) at every window size.

The figure makes the failure concrete. A candidate carrying a fixed
:math:`\Delta\chi^2 = 20` — a line that is really just absorbing the lineshape
floor under a bright neighbor — has an F-test p-value of about ``0.048`` in the
tightest window, just inside significance, and about ``2\times10^{-4}`` once the
window is padded to a thousand bins. Same line, same evidence; the verdict flips
with the window geometry. This is exactly the bias the residual-rescue
investigation traced on real windows, where weak peaks cleared the F-test
"because there are so many points in a window."

The context-invariant gate
--------------------------

The fix is to gate on the quantity that does *not* depend on window size. Because
the :math:`K`-line and :math:`(K{+}1)`-line models compared at a step share the
same window bins, the chi-squared *difference* :math:`\Delta\chi^2` is purely the
local likelihood-ratio statistic of the added line — the quiet bins contribute
equally to both models and cancel. Stage 5 accepts the more-complex model when

.. math::

   \Delta\chi^2 \;>\; 2\,\lambda\,\Delta k,

a penalized-chi-squared bar with a single global strictness :math:`\lambda`.
At :math:`\lambda = 1` this is the textbook AIC rule (:math:`\Delta\chi^2 > 2\,
\Delta k`); the shipped :math:`\lambda = 5` is a stricter, BIC-like bar,
calibrated once so the reference experiment reproduces its expected line count.
One added line costs :math:`\Delta k = 3` parameters (amplitude, offset, phase;
the decay time is shared and window-level), so the bar is :math:`\Delta\chi^2 >
30`. The bar is a constant — it has no :math:`n_\text{data}`, no
:math:`n_\text{eff}`, no window geometry in it at all, which is the property the
F-test lacked.

This gate replaced an intermediate form worth recording, because the stage's
audit trail still carries its fields. The first fix for the F-test bias kept the
AIC framework but shrank its sample size: an **information-weighted effective
sample size** :math:`n_\text{eff}` that weights each bin by
:math:`\log(1 + \text{SNR})` — counting only the bins a line actually informs —
fed into the small-sample-corrected AICc. That removed the dilution but left the
bar drifting with the weight distribution; scoring the window-size-independent
:math:`\Delta\chi^2` directly is the cleaner statement of the same idea, and is
what the gate does today. The classical F-statistic and p-value are still
computed and recorded at every step as a familiar diagnostic, but they no longer
gate the decision.

Localizing the gate against the lineshape floor
-----------------------------------------------

A bare :math:`\Delta\chi^2` bar has one remaining blind spot. At high
signal-to-noise the finite-acquisition lineshape is never a perfect fit to a real
line — a sub-percent core residual is irreducible — and that misfit is
**reducible-looking**: a spurious extra component placed on a bright line's core
can soak up the residual and earn a large :math:`\Delta\chi^2`, clearing the bar
for the wrong reason. The discriminator is *location*. Residual sitting under a
bright model component is the irreducible lineshape floor; residual sitting where
the model is small is a genuine missing line. Stage 5 scores the gate's
chi-squared against a fidelity-inflated per-bin noise

.. math::

   \sigma_\text{eff}^2(f) = \sigma^2(f) + \bigl(\kappa\,|\text{model}(f)|\bigr)^2,

with :math:`\kappa` the tolerated fractional model deficit (the same
:math:`\kappa` as the window-level SNR-aware gate below). Where the model is
bright, :math:`\kappa\,|\text{model}|` dominates and the residual's evidence is
shrunk to the fidelity budget; where the model is small,
:math:`\sigma_\text{eff} \to \sigma` and the evidence keeps full weight. A
frozen-skirt budget covers bright structure that was subtracted from the data and
is invisible to :math:`|\text{model}|`.

.. figure:: stage5_fitting/figures/fig2_sigma_eff_localization.png
   :width: 75%
   :align: center

   The same candidate evidence, scored two ways. Both candidates carry an
   identical raw :math:`\Delta\chi^2` well above the accept bar (gold). Under the
   :math:`\sigma_\text{eff}` weighting (blue), a real line over quiet bins keeps
   its full evidence and is accepted, while a candidate absorbing the lineshape
   floor under a bright line (on-line SNR 400) is discounted to a fraction of the
   bar and rejected.

In the synthetic test both candidates start at raw :math:`\Delta\chi^2 = 96`.
Over quiet bins the :math:`\sigma_\text{eff}` evidence is unchanged at ``96`` —
comfortably above the bar of ``30``. Under the bright line's core it collapses to
``0.24``, three orders of magnitude below the bar. The gate keeps the real line
and rejects the absorber, on the same nominal evidence, by reading where the
residual sits.

The same gate at every site
---------------------------

This penalized, localized gate is a single function applied at all four places
Stage 5 weighs adding or removing a line: the conservative add-one-peak accept
test, the blend-aware seeder's :math:`K{=}2`/:math:`K{=}3` escalation, the
per-line knockout test, and the merge of close pairs. Using one criterion
everywhere is what makes the fit's behavior coherent — a line the accept gate
admits is one the knockout gate will not immediately remove, because they ask the
same question with the same bar.

Blends: always detectable, sometimes mis-initialized
----------------------------------------------------

The hardest thing a prior-free fitter does is resolve a blend — two lines closer
than a linewidth. Fitting the unwindowed spectrum (the reason the
:doc:`canonical FT <../stage1_ft>` is left unapodized) is what makes this
possible at all, but it raises two separate questions that have very different
answers: *can* a close blend be recovered, and *will* the sequential loop find
it.

.. figure:: stage5_fitting/figures/fig3_blend_recovery_detectability.png
   :width: 80%
   :align: center

   A 1:1 in-phase blend swept over separation, in line widths. Top: the
   frequency error of a joint two-line fit started at the true positions —
   recovery holds at the sub-kHz level all the way down to 0.3 of a linewidth.
   Bottom: the reduced chi-squared of a single-cosine fit to the same blend
   (red) is elevated by one to two orders of magnitude across the whole range,
   while the joint two-line fit (blue) sits at one — so a blend always leaves an
   unmistakable single-line signature.

**Recovery is excellent.** Given the right number of lines, a joint fit recovers
a 1:1 blend at on-line SNR 120 to better than ``0.6`` kHz even at 0.3 of a
linewidth, and to ``0.2``–``0.3`` kHz from half a linewidth outward — far below
the nominal resolution. **Detectability is never the limit.** A single damped
cosine has one fixed shape; a blend it cannot match leaves a reduced chi-squared
of ``19`` at 0.3 FWHM rising to nearly ``290`` at 2 FWHM, against ``1`` for the
correct two-line fit. The elevation is large, smooth, and present at every
separation.

The limit is **initialization**. The conservative loop fits one cosine, lets it
drift to the blend centroid, then tries to add a second from that drifted state —
and for a tight, near-equal, in-phase blend that sequential path lands in a basin
where the two cosines collapse together. The blend was always recoverable (top
panel) and always detectable (bottom panel); the loop simply did not start the
two-line fit well enough. This is why Stage 5 carries a **blend-aware seeder**:
when a single-cosine fit leaves an elevated reduced chi-squared, it retries with
two (then up to three) lines *straddling* the feature rather than stacked at the
centroid. That escalation, not a large patience budget, is what makes the
conservative loop resolve real doublets — and the per-window peak count is
otherwise unbounded, so nothing else caps the lines a window may grow (a width
bound, not a count, holds the forest).

The residual-rescue chain
-------------------------

The detector runs on the magnitude spectrum and trusts its candidate list. When
that list is under-complete — a hyperfine multiplet whose strongest member masked
the others, a line on a strong neighbor's shoulder — the conservative loop
converges to a model that absorbs the missing lines, leaving an above-noise,
phase-coherent residual. The **rescue** is a second pass that re-examines that
residual.

Its structure matters. Each round computes the residual against a properly
**joint** refit of every line accepted so far — never a recursion on a shrinking
residual against an ever-more-wrong model. The alternative, recursing on the
residual of the initial fit, propagates that fit's errors (its wrong decay, its
wrong amplitudes) forward as structure the rescue then chases; the joint refit
makes each round's "what is left" question well posed. The rescue nominates
generously from residual prominence, folds the candidates into the joint fit, and
gates each one through the same penalized gate the main loop uses, with a
remove-and-refit cleanup that drops any line the joint fit no longer supports.
The decay time used to seed the rescue is the converged value, except when the
initial fit pinned it at the bound (the signature of a broken fit absorbing
unmodeled residual), in which case it falls back to the calibrated
:doc:`Stage 2b <../stage2b_tau>` value.

Two further pieces close the dense, high-dynamic-range case. A **leakage-wing
baseline** — a low-order complex polynomial, added only on an F-test or
edge-coherence trigger — carries the smooth pedestal that hundreds of distant
lines' summed leakage leaves under a window, which is otherwise un-modelable from
the discrete contributor set. And close pairs that the fit split are reconciled
by a two-tier merge that is **conservative by default**: below half a linewidth a
pair is merged, and above it a merge is gated by the same penalized criterion.
The bias toward merging is deliberate — in the ambiguous sub-resolution band an
LSQ split is far more often a cancelling near-duplicate artifact (the recurring
pathology where two cosines collapse onto one position) than a real doublet, so
the safe default is to collapse it and surface the window for review rather than
claim a split the data do not compel.

This does **not** contradict the sub-linewidth recovery above. The blend study
shows the *fitter* can resolve a pair to 0.3 of a linewidth when it is told there
are two lines; the merge governs whether the pipeline *keeps* a sub-resolution
split by default. The two are reconciled by a **blend escape**: a sub-linewidth
pair survives the structural merge when collapsing it would cost overwhelming
chi-squared *and* its members are physically constructive — distinct phases
producing a composite a single line shape cannot match, the signature of a
genuine blend rather than a numerical duplicate. A tight doublet the data truly
demand is preserved; an over-split of noise is not. (A phase-coherence projection
screen was built for the rescue and then removed — a counterfactual on every
window of the reference experiment showed it was a no-op against the penalized
gate and the :math:`\sigma_\text{eff}` weighting, which already do its job — so
the rescue carries no separate coherence stage.)

The SNR-aware health gate
-------------------------

How do you know a converged fit is good? The textbook answer, reduced
chi-squared near one, fails at high signal-to-noise — and fails in a way that
matters, because it would condemn the best fits in the spectrum. At on-line SNR
of :math:`10^4`–:math:`10^5` the per-bin noise is so small that the irreducible
sub-percent lineshape misfit is hundreds of standard deviations per bin, so a
perfectly good fit of a bright line has a reduced chi-squared in the thousands.
The honest health metric has two regimes:

.. math::

   \chi^2_r \;\le\; F + (\kappa\cdot\text{SNR}_\text{max})^2,

the **SNR-aware acceptance gate**. At low SNR the deficit term vanishes and the
gate is the noise-regime allowance :math:`\chi^2_r \le F` (with :math:`F = 3`
budgeting for the sampling scatter of a good fit); at high SNR the
:math:`(\kappa\cdot\text{SNR})^2` term governs and a bright line that fits to its
lineshape floor passes. The fractional core residual :math:`\varepsilon =
\sqrt{\max(\chi^2_r - F, 0)}/\text{SNR}_\text{max}` is the deficit-regime
quantity — the honest, SNR-independent measure of how well the model matches the
line. This is the same fidelity budget :math:`\kappa` that localizes the accept
gate above; the window-level allowance and the per-bin :math:`\sigma_\text{eff}`
are two views of one number.

.. figure:: stage5_fitting/figures/fig4_snr_aware_gate.png
   :width: 85%
   :align: center

   Per-window reduced chi-squared against per-window max SNR for every window of
   the seven checked-in experiments, fit each in its recommended line shape. The
   reference defaults hold across roughly three decades of signal-to-noise: the
   cloud follows the :math:`F + (\kappa\cdot\text{SNR})^2` gate (blue), flat at
   the noise floor where lines are faint and rising as the lineshape floor at the
   bright cores. The same window cloud is what makes the bulk look "bad" under a
   flat :math:`\chi^2_r \approx 1` standard and healthy under the SNR-aware one.

Cross-fixture validation
------------------------

The defaults are tuned on one experiment (the gaussian-shaped reference at SNR up
to a few hundred), so the load-bearing question is whether they generalize. The
seven checked-in fixtures span vinyl cyanide at SNR above :math:`30{,}000`, three
methyl-rotor spectra, and several sparse ones — built fresh through Stage 5, each
in the line shape its :doc:`Stage 2b <../stage2b_tau>` vote recommends, with no
hand-tuning. Across all seven the SNR-aware gate clears at least 95 % of every
fixture's windows (pass rate ``0.95``–``1.00``), and the per-window reduced
chi-squared median stays between ``1.1`` and ``1.6`` — from the sparse low-SNR
spectra up to vinyl cyanide at SNR above :math:`30{,}000`. The defaults
calibrated on one experiment hold across the whole set without adjustment.

On the two vinyl-cyanide fixtures a ground-truth catalog is available, so
detection completeness can be measured directly: the fit recovers ``34`` % of
the catalog's in-band lines on the sparse fixture and ``52`` % on the dense,
extreme-SNR one. The recall is
bounded by *detectability*, not by fit quality: the catalog includes lines below
the spectrum's noise that no fit can recover. Where a line is detected, the fit's
frequency is recovered to the instrument's accuracy floor — and that floor, a few
kilohertz of clock drift, is a property of the instrument, not the pipeline (the
formal fit precision is far tighter; the gap between them is the subject of the
:doc:`clock declaration <../clock_declaration>` and the Stage 6 reports).

Caveats
-------

* **The gate is calibrated, not derived from first principles.** The strictness
  :math:`\lambda = 5` and the fidelity budget :math:`\kappa = 0.05` are set so
  the reference experiment reproduces its expected line list; the cross-fixture
  validation is the evidence that those values transfer. A genuinely different
  instrument family could warrant its own calibration, which is why both are
  settings.

* **Recovery assumes the right number of lines.** The blend study's sub-kHz
  recovery is for a joint fit told there are two lines; on real data the seeder
  has to *discover* that count, and where the evidence for a sub-resolution split
  is ambiguous Stage 5 merges and flags for review rather than guessing. The
  doublet-versus-single adjudication is a curation decision, not a fit decision.

* **The SNR-aware gate is a fidelity floor, not a defect.** A window failing
  :math:`\chi^2_r \le F + (\kappa\cdot\text{SNR})^2` at high SNR is reporting that
  the analytic lineshape departs from the real line at the part-in-:math:`10^5`
  level — real, but not something the fit can drive to zero. The fractional
  residual :math:`\varepsilon`, not the raw :math:`\chi^2_r`, is the quantity to
  read there.

.. _stage5-fitting-reproducing:

Reproducing this note
---------------------

The synthetic gate, :math:`\sigma_\text{eff}`, and blend studies call the shipped
:func:`~ftmwpipeline.fitting.validation.calculate_chi_squared_improvement`,
:func:`~ftmwpipeline.fitting.validation.sigma_eff_chi2`, and
:func:`~ftmwpipeline.fitting.window_fit.fit_window`; the cross-fixture roll-up
drives the shipped Stage 5 fit and the SNR-aware validator. Regenerate the
figures and ``results.json`` from the repository root with::

    python docs/source/methods/stage5_fitting/generate.py                  # full
    python docs/source/methods/stage5_fitting/generate.py --no-cross-fixture

The synthetic half is a few seconds; the cross-fixture roll-up builds all seven
fixtures through Stage 5 and takes several minutes. Three fast, data-free
invariants are exercised by the unit suite — the gate's window-size invariance
against the F-test bias, the :math:`\sigma_\text{eff}` discounting of an absorber,
and the single-cosine blend signature — and the cross-fixture snapshot is guarded
against drift by a ``slow``-marked test.
