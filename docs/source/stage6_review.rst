.. index::
   single: review
   single: finalization
   single: candidate ledger
   single: decision log
   single: attention routing
   single: final products
   single: frequency calibration
   single: uncertainty budget
   single: catalog cross-reference
   single: reports

Stage 6: Review, Reports, and Finalization
==========================================

Stage 6 is where a *person* enters. The pipeline through
:doc:`Stage 5 <stage5_fitting>` produces an automatic spectral model with a
complete audit record; Stage 6 turns that model into a **finalized, reportable
line list**. It does two things: it captures human decisions about the automatic
fit — adding a missed line, removing a spurious one, adjudicating a blend — and
it **consolidates and calibrates** the final data products that the reports
present.

The boundary that runs through the whole pipeline holds here too. Stages 0–5
extract the best model the data support with no molecular-physics priors; Stage 6
records *human* decisions about that model. A user decision is recorded
provenance, not a physics prior injected into the fit. Model-as-prior fitting —
fitting a molecular Hamiltonian to assign the lines — belongs to a separate
program above this one; Stage 6 defines the data products and the decisions that
program would build on.

The review surface is a CLI stage object, **``review``**, mirrored on the Python
interfaces, and the reports are a second object, **``report``**. Both are
**read-only over the fit**: a review decision triggers a single-window refit but
never silently changes another window, and a report renders the persisted record
without ever recomputing it.

Two flags, no global lock
-------------------------

There is **no global "finalize" action** that freezes the record. Stage 6 is a
curation layer over the Stage 5 fit, carrying two **independent** flags per
window, shown as one combined label:

- **provenance** — ``auto`` (the fit produced it and no human has touched it),
  ``reviewed`` (a human looked and accepted the automatic fit unchanged), or
  ``user-edited`` (a decision was applied);
- **attention** — none, or ``needs-attention`` with one or more justified
  reasons. Attention is **advisory**: it never blocks a report. A window may
  legitimately stay ``auto · needs-attention``.

The two axes are orthogonal — a window can be ``user-edited · needs-attention``
(edited, but a neighboring spur still flags it) or ``user-edited`` with nothing
left to flag. Each **peak** additionally carries an ``origin`` of ``auto`` or
``user`` that survives serialization, so a hand-added line is visibly marked
wherever it appears.

Report-readiness is **computed, not asserted**. The *only* hard bar to generating
a report is a ``user-edited`` window whose decision has been **invalidated** by
re-running an upstream stage (see :ref:`stage6-decisions` below).
``needs-attention`` flags are reported honestly but never block.

Attention routing
-----------------

``review run`` builds one consolidated, ranked "needs attention" surface rather
than a scatter of per-feature lists, so the analyst has a single worklist. A
window is flagged for any of:

- **worst-ε** — the window's reduced :math:`\chi^2` fails the
  signal-to-noise-aware acceptance gate (the same standard ``fit check`` uses);
- **over-split** — a sub-resolution pair was auto-merged, or a member's amplitude
  variance-inflation factor flags a degenerate split for the analyst to confirm;
- **candidate-bearing** — the :ref:`candidate ledger <stage6-ledger>` holds a
  revivable rejected line with strong residual evidence (gated by a stiffer bar
  than the display ledger, so the surface stays actionable);
- **spur-adjacent** — a fitted line sits next to a masked clock/LO spur;
- **edge/boundary** — a line sits at a window edge, where leakage coupling is
  hardest.

A flag clears one of two ways: an **edit** (the window becomes ``user-edited``)
or an explicit ``review accept`` (the window becomes ``reviewed``, recording that
a human looked and was satisfied). Routing never forces a resolution — there is
no rubber-stamp — and ``review rank --by <metric>`` re-sorts the worklist by a
chosen diagnostic (minimum signal-to-noise, maximum VIF, reduced :math:`\chi^2`,
candidate evidence, edge distance, spur proximity, or the per-line determinacy
score).

.. _stage6-ledger:

The candidate ledger
--------------------

Every line the Stage 5 fit *considered and rejected* is still in the record — in
the per-window audit trail and rescue history. Stage 6 derives a **candidate
ledger** from that data on demand: it collects the add-loop rejects, the rescue
candidates, the failed blend-split trials, and the separation rejects, and
normalizes them to **one entry per distinct molecular frequency** (deduplicated
across rescue rounds), each carrying its best evidence seen, its rejection
reasons, and the recorded seed needed to revive it.

A **display bar** keeps gate-killed dust off the worklist — only candidates with
real residual or rescue evidence surface. The ledger is a pure function of the
already-persisted Stage 5 data, so ``review show --candidates`` renders it
without re-fitting and without touching the Stage 5 fit. It is the menu the
analyst draws from when a window is flagged candidate-bearing: a near-miss line
the conservative gate held back, ready to revive with its original seed.

Editing a window
----------------

The edit verbs each perform a **single-window refit** from that window's own
persisted state — its fitted peaks as seeds, its frozen contributors
reconstructed from its own record — using the production nonlinear-least-squares
primitive. No neighbor is re-fit and no upstream stage re-runs.

- ``review edit --window N --add F`` / ``--remove F`` — add or remove a line.
  ``--add`` snaps to the nearest ledger candidate within tolerance (reviving its
  recorded seed) or seeds a fresh line at ``F``; ``--remove`` snaps to the
  nearest fitted peak.
- ``review merge --window N --peaks F1,F2`` — collapse a set of lines to one
  (amplitudes summed, frequency the signal-to-noise-weighted mean). When the set
  matches a pair for which the observation-only doublet pass already recorded a
  merged-single alternative, ``merge`` snaps to that recorded seed.
- ``review split --window N --peak F [--into K]`` — replace one line with ``K``
  (default 2) straddling ``F`` by a fraction of a resolution element.
- ``review accept --window N`` — the "looked, no change" dismissal (→
  ``reviewed``); it also accepts a revived candidate or a recorded doublet merge.

``merge`` and ``split`` are physics-aware-reseed sugar over add-plus-remove. All
edits carry ``user`` provenance.

**A user edit bypasses the accept gate but not the optimizer.** The analyst is
the gate — a user-added line is kept without the F-test the automatic loop
applies — but it still faces the fit honestly: if the optimizer drives it to
zero amplitude or collapses it onto a neighbor, the result reports that rather
than keeping a phantom. And subsequent automatic passes are bound to the
decision: the survival prune, the merge cleanup, and the rescue pass will **not**
prune a user-added line nor re-add a user-removed one, because either would make
the verb feel broken.

.. _stage6-decisions:

The decision log and replay
---------------------------

Every edit appends an entry to an **ordered decision log** persisted in the file,
so the ``.ftmw`` reproduces the curated analysis with no side channel. Each entry
is **anchored** to a window identity and a molecular frequency, and records its
kind, its ``user`` provenance, and an evidence snapshot. ``review log`` lists the
history; ``review undo`` reverts the most recent decisions.

Because the decisions are anchored rather than baked in, re-running an upstream
stage does not silently discard them. **Replay re-applies each decision wherever
its anchor still resolves** and surfaces a comparison — reduced :math:`\chi^2`
with and without the decision, the peak-count and frequency deltas — so the
analyst can decide whether to revisit it. A decision is **invalidated** only when
its anchor no longer resolves: the window is gone, or the frequency has fallen
out of any window or out of band. An invalidated ``user-edited`` window is the
sole hard bar to report generation; the report, when generated, logs the Stage 6
decisions for honesty. Throughout, the curated result stays separable from the
automatic one, so a curated fixture never silently masquerades as an automatic
benchmark.

The final products and the frequency budget
-------------------------------------------

``review run`` also consolidates the **final-products table** — the single
canonical, calibrated line list the reports present, persisted in the file. For
each accepted line it records the frequency, the amplitude, the phase, the
signal-to-noise, and the uncertainty budget on the frequency. The raw
(uncalibrated) fitted frequency lives in the per-window detail as a drill-down,
not as a second peer table; the two sit at different altitudes, which removes the
redundancy.

The frequency uncertainty is composed as three independent terms in quadrature:

.. math::

   \sigma_f = \sqrt{\sigma_\text{stat}^2 \;+\; (\sigma_\varepsilon\, f_\text{baseband})^2
              \;+\; \sigma_\text{floor}^2}.

- :math:`\sigma_\text{stat}` is the **statistical precision** from the fit
  covariance — the per-line frequency error Stage 5 reports.
- :math:`\sigma_\varepsilon\, f_\text{baseband}` is the **timebase-calibration
  residual**: a fractional digitizer-clock scale error :math:`\varepsilon`
  multiplies the line's baseband offset from the probe, so it grows with distance
  from the local oscillator. It is zero unless self-calibration ran.
- :math:`\sigma_\text{floor}` is a **user-settable systematic floor**
  (``review run --sigma-floor``, default ``0`` kHz), the home for an instrument's
  irreducible run-to-run accuracy term that the clock lattice cannot pin. The
  default of zero keeps the reported budget honest about precision and leaves any
  accuracy claim to the analyst who knows the instrument.

**Calibration is a reported state, never a report gate.** Whether the frequency
axis needs self-calibration is a property of the instrument's clock reference,
declared alongside the :doc:`clock tree <clock_declaration>`, and defaulting to
"assume the axis is absolutely calibrated" when nothing says otherwise. The table
and the report state one of three cases plainly:

- **Rb-locked** (declared, or the default) — the axis is absolutely calibrated by
  a frequency standard; :math:`\varepsilon \equiv 0` and self-calibration is a
  null operation. Frequencies are trusted as-is.
- **Free-running, self-calibrated** — ``timebase_calibration`` ran; the measured
  :math:`\varepsilon` is applied and its residual folded into the budget.
- **Free-running, not self-calibrated** — frequencies are reported uncalibrated
  with a strong recommendation to self-calibrate.

The timebase self-calibration handles exactly one topology: all signal-chain
clocks locked, with the **digitizer the single free-running source**, whose
fractional scale error it fits from how the locked spurs appear to drift. A
declaration outside that topology — a locked digitizer with some other
free-running clock — reports as free-running with self-cal *unavailable* rather
than producing a wrong :math:`\varepsilon`. This is a stated limitation, not a
gap to be filled in this layer; ``timebase_calibration`` is therefore a *soft*
input, strongly recommended for a declared free-running instrument and a null op
for a locked one, and never a hard bar to a report.

The determinacy score
---------------------

Each line in the report and the per-window detail carries the per-line **``qual``
determinacy score** introduced in :doc:`Stage 5 <stage5_fitting>` — how many of
four independent checks the line clearly passes, written ``k/4`` (detected with
margin, amplitude identifiable, position pinned, isolated). It also serves as a
``review rank --by`` key. The framing is the same here: it measures how firmly
the data *determine* a line, **not** whether the line is a real, assignable
transition. A high score can still attach to an unmasked spur or an unassigned
feature, so it informs curation rather than gating it.

Catalog cross-reference
-----------------------

Every report level accepts an optional ``--catalog`` of expected frequencies. For
each reported line the report flags the nearest catalog entry within a geometric
tolerance :math:`N\sqrt{\sigma_f^2 + \sigma_\text{cat}^2}` (``--catalog-nsigma``,
default ``3``) and echoes its **opaque label** — proximity annotation only,
**never an assignment**. The pipeline emits unassigned lines; the cross-reference
is a cross-check, never a fit input. The reader accepts a CSV/whitespace table
(``frequency_mhz`` plus an optional uncertainty and label, with the uncertainty
unit read from the header) and the Pickett/SPCAT ``.cat`` predicted-line catalog.

A catalog also unlocks the **pull calibration** — the distribution of
:math:`(f_\text{fit} - f_\text{cat}) / \sigma_f`, which should be a unit normal
when the :math:`\sigma_f` budget is honest. The report summarizes the pull
(mean, spread) and flags a spread much greater than one (an optimistic budget) or
much less than one (a conservative one). It is a tool to *validate* the budget,
not a shipped gate.

The reports
-----------

The ``report`` object renders the finalized record; it never recomputes the fit.

- ``report run`` is the default deliverable: the calibrated line table (CSV) and
  a self-contained HTML report, written into an output directory. The HTML is one
  portable file — the stylesheet inlined and every figure embedded — with a
  full-spectrum index that shades the attention windows, a methods-and-results
  page carrying the per-stage algorithm prose plus this experiment's numbers and
  distribution histograms, and one page per fit window (the ``fit show`` figure,
  the fitted lines with raw and calibrated frequencies, the covariance, the
  candidate ledger, the recorded decisions, and the fit history). ``--summary``
  keeps the index and methods only; ``--windows attention`` folds in only the
  flagged windows for a spectrum with thousands of them.
- ``report table`` exports the final-products table on its own as CSV, JSON, or
  LaTeX (a ``booktabs`` table for a paper's supplementary material). Spectroscopic
  fitting formats (Pickett ``.lin`` / SPFIT) are out of scope by design: the
  pipeline emits *unassigned* lines.

The HTML report opens **read-only**; a ``Curate`` toggle reveals inline edit
controls that collect adds, removes, splits, and merges into a cart and export
them as a curation file that ``review apply`` consumes — a way to triage a report
in the browser and replay the decisions against the file, without ever editing
the file from the page.

Running it
----------

Stage 6 requires :doc:`Stage 5 <stage5_fitting>`; the timebase calibration is a
soft input. ``review run`` builds the curation layer and the final-products table
in one pass.

.. code-block:: console

   $ ftmwpipeline review run exp_2638.ftmw
   $ ftmwpipeline review show --attention exp_2638.ftmw
   $ ftmwpipeline review edit --window 217 --add 31214.20 exp_2638.ftmw
   $ ftmwpipeline report run exp_2638.ftmw --output-dir report/

The same operations on the Python interfaces:

.. code-block:: python

   import ftmwpipeline.api as ftmw

   ftmw.review_run("exp_2638.ftmw")
   for w in ftmw.rank_windows("exp_2638.ftmw", by="chi2r", top=10):
       print(w.window_id, w.metric, w.value)

   ftmw.review_edit("exp_2638.ftmw", window_id=217, add=[31214.20])
   ftmw.report_run("exp_2638.ftmw", output_dir="report/")

   # or, object-oriented
   from ftmwpipeline import Pipeline
   pipe = Pipeline.open("exp_2638.ftmw")
   pipe.review_run()
   pipe.report_table(fmt="csv", output="lines.csv")

.. figure:: figures/stage6_review.png
   :width: 95%
   :align: center

   The Stage 6 review surface over the example experiment: the report's
   full-spectrum index overview. The finalized active spectrum (magnitude) runs
   across the whole band, with the windows the review flagged for attention shaded
   — the analyst's worklist at a glance. The bulk of the spectrum is settled
   ``auto`` model; attention is drawn only to the few windows where a close call,
   a candidate, or an edge-coherent residual warrants a human look.

Limitations
-----------

- **No automatic cascade.** A single-window refit changes only that window. If
  the edited window is a *contributor primary* whose frozen skirt reaches its
  neighbors, those neighbors become stale; Stage 6 reports which neighbors
  reference the window but does not re-fit them. A cascading re-fit is future
  work.
- **Curation is recorded, not interpreted.** The decisions and their provenance
  persist, but adjudicating a genuine sub-resolution doublet against an over-split
  remains the analyst's call — supported by the observation-only doublet
  statistics and a catalog where one exists.
- **Precision, not accuracy, unless self-calibration ran.** The reported budget
  carries the formal precision plus whatever calibration the declared clock tree
  supports. An undeclared systematic offset is left to ``--sigma-floor``, not
  invented by the pipeline.

With the line list reviewed, calibrated, and finalized, the report renders the
analysis a reader can trust — the line positions, their uncertainty budget, and
the provenance of every decision that produced them.
