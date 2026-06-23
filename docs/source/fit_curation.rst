.. index::
   single: curation
   single: curation file
   single: curation cart
   single: HTML report
   single: report
   single: batch editing
   single: review apply

Fit Curation
============

The pipeline through :doc:`Stage 6 <stage6_review>` produces a fitted line list
and renders it as a self-contained report. Curation is the act of correcting
that fit (adding a line the detector missed, removing a spurious one,
adjudicating a blend) and folding the corrections back into the ``.ftmw`` file.
The Stage 6 page covers the individual edit verbs and the rule that binds them:
every edit re-runs the affected window's nonlinear least-squares fit, so a
curated line is fit as honestly as an automatic one, and each decision is
recorded with its provenance.

This page documents the two surfaces built for curating at production scale,
where a spectrum may carry hundreds of windows and thousands of lines: the
portable HTML report a reader works through in a browser, and the curation file
that batches a session of edits into one reproducible command.

Anatomy of the HTML report
--------------------------

``report run`` writes one portable ``<stem>_report.html``. It is assembled
internally as a set of linked pages, then folded into a single document with the
stylesheet inlined and every figure base64-embedded, so the cross-page links
become in-document anchors. The result has no external dependencies: it opens in
any browser, archives beside the ``.ftmw`` file, and mails to a collaborator as
a single attachment, with no server and no asset directory.

The report carries three kinds of content:

- **An index.** A summary block (the experiment's calibration state, line count,
  and worklist tally), a clickable full-spectrum overview with the
  attention-flagged windows shaded, a windows table, and the finalized line
  list.
- **A methods-and-results page.** The per-stage algorithm prose with this
  experiment's own numbers folded in, distribution histograms of the fitted
  parameters, and rendered display math.
- **One page per fit window.** The fit panels (real, imaginary, and magnitude
  data with the model and residuals), the fitted lines with their raw and
  calibrated frequencies, the parameter covariance, the candidate ledger, the
  recorded decisions, and the fit history.

Three flags scope the output for large spectra, where rendering a detail page
for every window is neither fast nor useful:

- ``--summary`` keeps the index and the methods page only, with no per-window
  detail.
- ``--windows attention`` renders detail pages only for the windows the review
  flagged, so a spectrum with thousands of windows still produces a report sized
  to the analyst's worklist.
- ``--level1-only`` writes the data table alone, with no HTML; ``--no-table``
  writes the HTML alone.

.. code-block:: console

   $ ftmwpipeline report run exp_2638.ftmw
   report run: wrote table to exp_2638_lines.csv
   report run: wrote self-contained full HTML report to exp_2638_report.html

Curating in the browser
-----------------------

The report opens **read-only**. A ``Curate`` toggle in the navigation bar flips
it into curation mode, revealing an inline control on each fitted-line and
candidate-ledger row and making the per-window plots clickable. Every control
only *queues* an edit; nothing edits the ``.ftmw`` file from the page. Queued
edits collect in a docked **cart**, grouped by window, and each one draws a
marker on its window plot, color-coded by action and removable with a click:

.. figure:: figures/fit_curation_annotations.png
   :width: 100%
   :align: center

   The report's own magnitude panels for two windows, overlaid with a curation
   cart exported from the browser (the marker colors match the in-report
   controls). Each panel shows the fitted model on the display grid above a
   residual strip, with the per-peak labels the report assigns. On the left
   window, a **split** marker (orange) divides a weak line in two and an **add**
   (green) seeds a missed line in the gap beside it; on the right window, a
   **merge** marker (purple) collapses a resolved close pair into one. The
   markers are queued intentions, not yet applied: exporting the cart writes them
   to a curation file that ``review apply`` refits.

The cart's **Download .csv** and **Copy** controls export the queued edits as a
**curation file** (``<stem>_curation.csv``) and print the ``review apply``
command that replays it. The frequency a control emits is the line's raw
Stage 5 model frequency, the value the edit verbs match on, not the calibrated
value shown in the table. The browser never writes to the file; it only composes
the curation file that the command-line tool applies.

Curation files
--------------

A curation file is a small CSV that records an ordered sequence of edits, one
per row. It is the canonical interchange for a curation session: diffable,
hand-editable, and independent of the report that may have authored it. The
browser cart writes one, and a curation file is equally well written by hand or
generated by a script.

The header is ``action,window,freqs,params``. Each row names one action, the
integer ``window`` id it targets, the molecular frequencies it carries (a
``;``-separated list, in MHz), and any ``;``-separated ``key=value`` modifiers.
Blank lines and lines beginning with ``#`` are ignored, and a leading header row
is optional.

.. list-table::
   :header-rows: 1
   :widths: 14 14 40 32

   * - ``action``
     - ``freqs``
     - Effect
     - ``params``
   * - ``add``
     - one
     - Revive the nearest ledger candidate within tolerance, or seed a fresh
       line at the frequency.
     - none
   * - ``remove``
     - one
     - Drop the fitted line nearest the frequency.
     - none
   * - ``merge``
     - two or more
     - Collapse the named lines to one (amplitudes summed, frequency the
       signal-to-noise-weighted mean).
     - none
   * - ``split``
     - one
     - Replace the named line with ``K`` straddling it.
     - ``into=K`` (default ``2``)
   * - ``accept``
     - none
     - Dismiss the window as reviewed with no change, or revive a named
       candidate.
     - ``candidate=F`` to revive

A representative curation file:

.. code-block:: text

   action,window,freqs,params
   remove,42,26613.6131,
   add,42,26614.20,
   merge,17,9001.10;9001.18,
   split,5,12000.50,into=3
   accept,8,,candidate=15001.4

Edits on one window are **coalesced** before they are applied. A maximal run of
``add`` and ``remove`` rows on the same window collapses into a single refit
rather than one refit per row, so the two rows for window 42 above become one
edit. A ``merge``, ``split``, or ``accept`` on a window flushes that window's
pending edit first, because each carries its own physics-aware seeding. Windows
are independent, so an edit interleaved on another window does not break the
coalescing of a pending group.

Applying a curation file
------------------------

``review apply`` replays a curation file through the same single-window refit
that the interactive verbs call, so a batch of edits produces exactly the result
of running the resolved plan by hand. ``--dry-run`` prints the resolved,
coalesced plan and any frequency-resolution warnings without touching the file:

.. code-block:: console

   $ ftmwpipeline review apply exp_2638.ftmw exp_2638_curation.csv --dry-run
   review apply (dry run): resolved plan
       1. edit window 42: add 26614.2000; remove 26613.6131
       2. merge window 17: peaks 9001.1000, 9001.1800
       3. split window 5: peak 12000.5000 into 3
       4. accept window 8: candidate 15001.4000
   4 action(s) would be applied (nothing written).

The warnings catch the two ways a target frequency fails to resolve: a
``remove``, ``merge``, or ``split`` frequency that matches no fitted peak within
tolerance (the edit would fail), or one that sits within tolerance of more than
one peak (the nearest is taken, which may not be the intended line). Previewing
them before the refit is the reason ``--dry-run`` exists. ``add`` and revived
candidates create peaks, so they are not checked.

Dropping ``--dry-run`` applies the plan, refitting each affected window in place:

.. code-block:: console

   $ ftmwpipeline review apply exp_2638.ftmw exp_2638_curation.csv
   review apply: plan
       1. edit window 42: add 26614.2000; remove 26613.6131
       ...
   applied 4 action(s).

Each applied edit appends an anchored entry to the
:ref:`Stage 6 decision log <stage6-decisions>`, exactly as the interactive verbs
do, so a curation file's effects carry their provenance and survive re-running
an upstream stage. The same Python entry points are available on the functional
API and the ``Pipeline`` class:

.. code-block:: python

   import ftmwpipeline.api as ftmw

   preview = ftmw.review_apply("exp_2638.ftmw", "exp_2638_curation.csv", dry_run=True)
   for action in preview.plan:
       print(action.kind, action.window_id)
   print(preview.warnings)

   result = ftmw.review_apply("exp_2638.ftmw", "exp_2638_curation.csv")
   print(result.applied)   # number of actions refit
