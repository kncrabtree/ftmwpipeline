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
  attention-flagged windows shaded, a windows table, an **applied-edits** table
  when the file already carries curation decisions (see below), and the finalized
  line list.
- **A methods-and-results page.** The per-stage algorithm prose with this
  experiment's own numbers folded in, a diagnostic figure for each stage —
  including the Stage 3 detections drawn over both the active FT and the
  primary-pass (Blackman-Harris) detection spectrum the detector localizes on,
  so a leakage-dominated regime is legible — distribution histograms of the
  fitted parameters, and rendered display math.
- **One page per fit window.** The fit panels (real, imaginary, and magnitude
  data with the model and residuals), the fitted lines with their raw and
  calibrated frequencies, the parameter covariance, the candidate ledger, the
  recorded decisions, and the fit history. Where a window carries an attention
  flag with a definite locus, the magnitude panel is annotated with a small
  caret and one-letter tag at that frequency — **C** for a missed-line
  (candidate) residual, **S** for a line sitting on a gated spur, **M** for an
  auto-merged pair — so it is obvious *where* to look; hover the caret for the
  reason detail.

The single-file report carries a sticky navigation bar. Besides the section
links and the **Jump to window** picker, a **Freq MHz** box jumps to the window
nearest a typed frequency, and a pair of **window** buttons step to the previous
/ next window. Keyboard shortcuts mirror these: ``j`` / ``k`` move to the next /
previous window. Both the buttons and the keys skip windows hidden by the tag
filter, so the filter selects which window types you step through. The controls
are inert when scripting is disabled; the anchors still work. Each window's own
title bar also carries **index** / prev / next buttons and a **Full spectrum**
toggle that reveals — hidden by default — the clickable full-spectrum overview
inside the window header, with this window highlighted.

Each window header carries a row of **tag chips** classifying the window at a
glance — ``attention`` (in the review queue), ``edited`` / ``reviewed`` (its
curation provenance), ``cascade-edit`` (changed only because another window's
edit propagated into it), ``merged`` (an auto-merged degenerate pair),
``high-χ²ᵣ`` / ``high-ε`` (fit-quality outliers), and ``catalog-match`` (a
catalog hit, when a catalog was supplied). A **Filter** menu in the navigation
bar lists the tags present in the report; ticking one or more hides every window
whose tags do not include any of the ticked ones (``Show all`` clears the
filter). The keyboard and window-step navigation skip the hidden windows while a
filter is active.

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
   residual strip, with the per-peak labels the report assigns. Add and remove
   are the cart's whole grammar. On the left window, an **add** (green, solid)
   seeds a line the detector missed in the gap between two faint fitted lines;
   on the right, a **remove** (red, dashed) drops an SNR 5.2 line the reviewer
   declines to claim, well away from the window's bright line. The markers are
   queued intentions, not yet applied: exporting the cart writes them to a
   curation file that ``review apply`` refits.

The cart's **Download .csv** and **Copy** controls export the queued edits as a
**curation file** (``<stem>_curation.csv``) and print the ``review apply``
command that replays it. A **remove** exports the line's ``peak_uid`` as a
``uid:N`` token, which names that peak exactly rather than by proximity: a
neighboring line inside the snap tolerance cannot be matched instead, and the
frame does not enter into it. Everything else a control emits -- an **add**, and
a **remove** on a fit predating ``peak_uid`` -- is the line's raw Stage 5 model
frequency, the value the edit verbs match on, not the calibrated value shown in
the table; the exported file declares that frame in its own ``# frame:`` header.
The browser never writes to the file; it only composes the curation file that
the command-line tool applies.

Several convenience controls speed a long worklist. Each window's title bar
carries **Mark reviewed**, **Reviewed & next** (marks it reviewed and advances to
the next window, honoring the tag filter), and **Clear window edits** (drops just
that window's queued ops); the index windows table gains a per-row **reviewed**
button, in its own column on attention windows, so they can be triaged from the
overview without scrolling to each. Marking a window reviewed (from either place)
also clears its attention tint from the overview, and the two buttons stay in
sync. A cart entry is clickable — it scrolls to its originating window and flashes
the row. In curation mode the
magnitude plots take keyboard shortcuts that act on the peak nearest the pointer,
mirroring click-to-add: hover the plot near a line, then press ``r`` to remove
it; ``a`` arms (and disarms) click-to-add on that plot. The affected line flashes
and its marker appears on the plot. The cart lists these keys for reference.

There is no separate split or merge control, in the row, on the plot, or on the
keyboard: the cart's whole grammar is add and remove. Clicking a point on the
armed plot near an existing line queues an add there, and that add reads as a
split of the line once ``review apply`` runs it (see :ref:`stage6-edit` on the
Stage 6 page) — which is deliberately the common path, since a reader will more
often point at a plot than type a frequency. Checking two close lines' rows and
removing both, then adding one frequency in their span, reads as a merge the
same way.

Applied edits and rollback
--------------------------

When a report is generated from a ``.ftmw`` that already carries curation edits,
the index lists them in an **Applied edits** table — the file's recorded decision
log, in execution order, with each edit's window, action, and frequency anchor.
This is a read-only record of the file's curation state and is always shown. In
curation mode each row gains an **Undo** button; queuing one does not enter the
curation file (a rollback is a different operation) but instead makes the cart
surface a ``review undo`` command alongside the ``review apply`` one:

.. code-block:: console

   ftmwpipeline review undo exp_2638.ftmw --id 3 5

``review undo`` restores the automatic-fit baseline snapshot and replays every
surviving decision, so undoing by id is exact and order-independent; the ids
shown in the table are the ones to pass.

.. _curation-frames:

Frames: which frequency you are typing
--------------------------------------

Every curation verb resolves a frequency you supply against the lines already
in the file. That only works if both sides agree on what the number *means*,
and there are two frequencies for every line:

``raw``
   The frame the Stage 5 fit, the Stage 3 candidate ledger, and every persisted
   value (the decision log, created windows) live in. Values are stored raw
   permanently, because ``epsilon`` is recomputable and a stored calibrated
   value would silently change meaning after a timebase re-run.

``calibrated``
   ``f_corr = probe + (f_raw - probe) / (1 + eps)`` — the frame the final
   products table and the reports present.

Every frequency-bearing entry point across the three interfaces — ``add`` /
``remove``, ``accept``'s ``candidate_freq``, ``create``'s anchor, and a curation
file's frequencies — takes a ``frame`` argument of type
:data:`~ftmwpipeline.core.curation.Frame`, defaulting to ``"raw"``:

.. code-block:: python

   import ftmwpipeline.api as ftmw

   # The same line of exp_2638, named in each frame
   ftmw.review_edit("exp_2638.ftmw", 5, remove=[26879.5009], frame="raw")
   ftmw.review_edit("exp_2638.ftmw", 5, remove=[26879.5317], frame="calibrated")

On an ``rb_locked`` or ``uncalibrated`` file ``epsilon`` is ``0.0``, the two
frames coincide, and ``frame`` is inert. On a ``self_calibrated`` file it is
not — and the failure is a quiet one. A calibrated frequency submitted as raw
still resolves, and to the *right* peak, but seeds or anchors it wrong by
``probe_freq * eps/(1 + eps)``: under the snap tolerance, so it matches, and
over the statistical σ, so the error shows up in the result without ever
announcing itself.

The two numbers above are 30.9 kHz apart, on a file whose snap tolerance is
49.1 kHz (``epsilon`` = 2.19 × 10⁻⁶, probe 40960 MHz). Submitting the
calibrated one as raw finds the same line — and then seeds it 30.9 kHz off.

Because that mistake is invisible, **omitting ``frame`` on a frequency-bearing
call is a hard error on a ``self_calibrated`` file** rather than a silent
assumption:

.. code-block:: text

   ValueError: frame is required on a self_calibrated file: pass frame="raw"
   or frame="calibrated" explicitly rather than relying on the default. A
   calibrated frequency submitted as raw still resolves to the right peak, but
   is wrong by probe_freq * eps/(1+eps) -- under the snap tolerance and over
   the statistical uncertainty, so the mistake would be silent.

Passing ``frame="raw"`` explicitly is never an error on any file, so a script
that always declares its frame works everywhere. Use
:func:`~ftmwpipeline.api.frequency_calibration` (or ``ftmwpipeline timebase
state``) to read a file's calibration state and epsilon before deciding.

A curation file declares its frame in the file itself, since it is the one
place a calibrated frequency becomes a durable artifact — see
:ref:`curation-file-header` below.

One further safety net applies to a whole batch. If a plan's targets all
resolve with a residual matching what *this* file's epsilon predicts for an
omitted conversion — at least three matched targets, all displaced the same
direction, each within 25 % of the predicted offset — ``review apply`` and
``review preview`` emit an advisory naming the suspicion. It never blocks
anything: it is a heuristic, and a heuristic that refused would be worse than
none.

.. _curation-snap-tolerance:

The snap tolerance
------------------

A ``remove`` does not need the exact fitted frequency, and an ``add`` beside an
existing line is read as a split of it. Both readings are governed by one
constant, the **snap tolerance** — and it is defined in **active-FT bins**, not
in MHz:

.. code-block:: console

   $ ftmwpipeline review snap-tolerance exp_2638.ftmw
   snap tolerance: 0.049097 MHz (49.1 kHz)
     0.625 active-FT bins @ df = 0.078555 MHz
     T_active = 12.73000 us

:data:`~ftmwpipeline.core.curation.REFIT_SNAP_TOL_BINS` is ``0.625`` bins, so
the frequency it works out to is a property of one file rather than a fixed
number: ~49 kHz at a 12.7 µs acquisition, but ~6.3 kHz at 100 µs. There is no
such thing as "the" snap tolerance in MHz.

**Read the resolved value; do not multiply the bin count by a spacing of your
own.** Three reads give the same answer, all derived from the same active
region the curation verbs themselves consult:

.. code-block:: python

   import ftmwpipeline.api as ftmw
   from ftmwpipeline import Pipeline

   ftmw.refit_snap_tol_mhz("exp_2638.ftmw")            # -> 0.049097...
   Pipeline.open("exp_2638.ftmw").refit_snap_tol_mhz()  # the same value

``ftmwpipeline review snap-tolerance --format json`` returns the same values
plus the bin count, bin spacing, and ``T_active`` for a script to record.

Within the tolerance a requested frequency resolves to the **nearest** fitted
peak or ledger candidate; beyond it, ``remove`` is a hard error naming the
closest peak and its distance, while ``add`` seeds a fresh line at the typed
frequency — which is what a genuine new line wants anyway. The tolerance is
deliberately looser than the 0.25-bin candidate-dedup window, so two fitted
peaks *can* both fall inside it; that is resolved nearest-wins, and
``review apply --dry-run`` reports the multi-match as an advisory so the
ambiguity is visible before the batch runs. Addressing the line by
``uid:N`` instead sidesteps the question entirely.

Every verb's ``snap_tol_mhz`` parameter defaults to the resolved value for the
file it is called on; passing your own overrides it for that call only, in MHz.

Curation files
--------------

A curation file is a small CSV that records an ordered sequence of edits, one
per row. It is the canonical interchange for a curation session: diffable,
hand-editable, and independent of the report that may have authored it. The
browser cart writes one, and a curation file is equally well written by hand or
generated by a script.

The header is ``action,window,freqs,params``. Each row names one action, the
integer ``window`` id it targets, the molecular frequency it carries (in MHz),
and any ``;``-separated ``key=value`` modifiers. Blank lines and lines
beginning with ``#`` are ignored, and a leading header row is optional.

On an ``add`` or ``remove`` row, the ``window`` column is optional: leave it
blank (or write ``new``, ``auto``, or ``-``) and the window is derived from
the row's own frequency (or ``uid:N``, below) instead of named -- the live
window whose range covers it, since windows never overlap. A ``remove`` whose
frequency (or ``uid:N``) no live window covers is an error naming it, rather
than a silent do-nothing: ``remove`` never implies creating a window, since
there is nothing there to remove. An ``add`` whose frequency no live window
covers instead **mints the window it needs** (or widens an adjacent one, when
the gap is too narrow to hold a new one) and applies the add into it --
exactly what a separate ``review create`` row followed by an ``add`` row
would produce, except recorded as the ONE decision the user actually made
rather than two: the create is a mechanism the engine chose, not something the
user decided. This only fires when the ``window`` column was OMITTED; naming a
window explicitly is still an assertion, and naming one whose range does not
cover the row's frequency is still an error even if some other live window
would have. A run of several omitted-window rows that resolve to the same
EXISTING live window still coalesces into one edit, exactly as a run of
explicitly-named rows does; each row whose frequency implies its own create
gets its own window and its own edit, even when two such rows are adjacent.
``accept`` and ``create`` still require the window column: there it names the
window being acted on (or, for ``create``, may take the same blank/``new``/
``auto``/``-`` tokens to mean "mint a new one"), not a coordinate derived from
a frequency.

A ``remove`` row may name its target by identifier instead of by frequency,
writing ``uid:N`` for the line whose
:attr:`~ftmwpipeline.core.data_structures.FittedPeak.peak_uid` is ``N`` --
``remove,12,uid:15425022,``. That names the line exactly rather than by
proximity, so a neighbor inside the snap tolerance cannot be matched instead
and the file's frame does not enter into it. The browser cart writes this form;
the identifier is also the ``peak_uid`` column of ``report table``. Every other
action is frequency-only, and a ``uid:N`` token elsewhere is refused at parse
time -- an identifier names a line that already exists, and ``add``,
``create`` and ``accept``'s ``candidate=`` all name one that does not.

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
     - Drop the fitted line nearest the frequency, or -- given a ``uid:N``
       token instead of a frequency -- the line carrying that ``peak_uid``.
     - none
   * - ``accept``
     - none
     - Dismiss the window as reviewed with no change, or revive a named
       candidate.
     - ``candidate=F`` to revive

.. _curation-file-header:

Declaring the file's frame
~~~~~~~~~~~~~~~~~~~~~~~~~~

A curation file is the one place a calibrated frequency becomes a **durable**
artifact — everywhere else ``frame`` is a per-call argument that leaves no
trace. So the file declares its own frame, as a whole-line comment anywhere in
it:

.. code-block:: text

   # frame: raw
   action,window,freqs,params
   remove,5,26879.5009,

A ``# frame:`` directive declares the frame every frequency in that file is
expressed in, which is what makes the file self-describing: it can be mailed to
a colleague, committed to a repository, or replayed a year later without the
frame having to be remembered separately. The browser cart writes
``# frame: raw`` for exactly this reason, and a file carrying the directive
needs no ``frame`` argument even on a ``self_calibrated`` file.

The header and the ``frame`` argument do not override one another. Whichever is
given alone decides; if both are given and **disagree**, the apply refuses
rather than picking a winner, naming both. Both directives are optional — a
file with no header falls back to the ordinary per-call resolution described in
:ref:`curation-frames`.

When the frame is ``calibrated``, the file **must** also stamp the epsilon it
was written under:

.. code-block:: text

   # frame: calibrated
   # epsilon: 2.2e-6

That stamp is what lets a later apply or preview notice that the calibration
has moved — a timebase re-run, say — and refuse rather than silently resolving
against the wrong peaks:

.. code-block:: text

   ValueError: curation file frame drift: this file was staged
   frame=calibrated at epsilon=2.200000e-06, but the target file's current
   epsilon is 2.310000e-06. The calibration has changed since this file was
   written (e.g. a timebase re-run) -- re-stage the curation file against the
   current calibration rather than applying it as-is.

``# epsilon:`` without ``# frame: calibrated`` is rejected at parse time: an
epsilon stamp is meaningless with no calibrated-frame declaration to attach it
to.

Inferred actions
~~~~~~~~~~~~~~~~

``merge`` and ``split`` are **not** row actions -- a file naming either is
refused, at parse time, before anything is touched. Both are read from what an
``add``/``remove`` combination does to a window's peak set, not typed:

- An ``add`` within snap tolerance of a fitted peak that is **not itself being
  removed** is applied as a **split** of that peak into two, seeded at (the
  existing peak's position, the requested position).
- Removing two or more mutually-close fitted peaks (within snap tolerance of
  each other -- the components of one feature) while adding exactly one
  frequency inside their span is applied as a **merge**, seeded at the
  requested frequency (or at a recorded doublet-alternative seed, when the
  removed pair matches one).

The decision log records which reading was used (``kind="split"``/``"merge"``)
and stamps the frequency that was requested, so the log reports the
reinterpretation rather than hiding it. See :ref:`stage6-edit` on the Stage 6
page for the exact rule.

A representative curation file:

.. code-block:: text

   action,window,freqs,params
   remove,42,26613.6131,
   add,42,26614.20,
   remove,17,9001.100,
   add,5,12000.4875,
   remove,17,9001.130,
   add,17,9001.115,
   accept,8,,candidate=15001.4

Window 42's ``add`` seeds a fresh line clear of its other peaks, an ordinary
add. Window 5's ``add`` is unrelated, interleaved between window 17's rows.
Window 17's two ``remove`` rows and its ``add`` between them remove a pair of
lines 30 kHz apart -- mutually inside a 49.4 kHz snap tolerance, so the two
components of one feature -- and add one frequency in their span, which is read
as a merge. The interleaved window 5 row does not break that run, because
coalescing tracks each window independently.

Edits on one window are **coalesced** before they are applied. A maximal run of
``add`` and ``remove`` rows on one window collapses into a single refit rather
than one refit per row, so the two rows for window 42 become one edit, the
three (non-adjacent) rows for window 17 become one edit, and window 5's row
becomes its own edit -- three edits from six add/remove rows. An ``accept``
(or ``create``) on a window flushes that window's pending edit first, since
it changes the file in its own right; a plain add/remove never does, even
when curation-intent inference will read the coalesced result as a split or a
merge once the batch actually runs.

Applying a curation file
------------------------

``review apply`` executes a curation file's resolved plan as a single batch. It
loads the Stage 5 fit and builds the active-FT fit context once, applies every
action to that fit in memory, cascades the dependents of all directly-edited
windows in one combined pass, and persists the result once. Nothing is written
until every action has succeeded, so a plan that fails partway through leaves
the file untouched.

A curated file therefore holds only two states: the base it started from — the
automatic fit, or the automatic baseline when ``review undo`` is replaying onto
it — and the revised state the whole edit set produces. The set is applied in
one canonical order rather than the order the rows happen to be written in.
``create`` rows run first, since they install the structure later rows name, and
the remaining actions run grouped by ascending window id. Within a single window
the specified order is preserved, because an ``accept`` composes on the peak set
a preceding coalesced add/remove group left behind. Two curation files listing
the same per-window edits in different row orders reach the same fitted state
and the same decision log.

That end state is what the guarantee covers. The batch reproduces the *result*
of running the resolved plan by hand, not its sequence of intermediate refits;
and because an edit changes the leakage skirt its neighbors froze, the peaks of
one edit are not guaranteed to land where they would have had that edit been
applied on its own. Edits are reproducible together, not independent of each
other.

``--dry-run`` prints the resolved, coalesced plan in the file's own row order,
along with any frequency-resolution warnings, without touching the file. The
plan is pre-inference — every coalesced add/remove group prints as ``edit``,
even one that will read as a split or a merge once the batch actually runs, so
this preview shows what was typed, not yet the reinterpretation:

.. code-block:: console

   $ ftmwpipeline review apply exp_2638.ftmw exp_2638_curation.csv --dry-run
   review apply (dry run): resolved plan
       1. edit window 42: add 26614.2000; remove 26613.6131
       2. edit window 17: add 9001.1150; remove 9001.1000, 9001.1300
       3. edit window 5: add 12000.4875
       4. accept window 8: candidate 15001.4000
   4 action(s) would be applied (nothing written).

The warnings catch the ways an action fails to resolve against the file. For
``remove`` that is the target frequency: one that matches no fitted peak within
tolerance (the edit would fail), or one that sits within tolerance of more than
one peak (the nearest is taken, which may not be the intended line). A
``uid:N`` target is checked the same way but by identity rather than by
distance -- the window either holds a fitted peak carrying that ``peak_uid`` or
it does not, so there is no ambiguity case, only an unmatched one (which
includes a window fitted before ``peak_uid`` existed, where nothing can carry
an identifier). An ``add``
creates its peak and so has no target to match, but it does name a *window*,
and that is what goes stale — window ids are
reassigned whenever Stage 4 re-plans, and a plan window whose peaks all failed
their Stage 5 gate carries no fit to edit at all. So an ``add`` is checked for
the two conditions the refit will enforce: the window is live, and the frequency
lies on its data (with the snap tolerance allowed as slack, so only an add that
cannot land however it snaps is flagged). An ``add`` into a window a ``create``
in the same file installs is left to the apply, since its geometry does not
exist yet. Revived candidates are not checked: the window's own ledger supplies
the frequency.

Previewing all of this before the refit is the reason ``--dry-run`` exists — but
the preview is advisory, not a gate. The apply is what enforces; it just does so
without writing anything unless every action succeeds.

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
an upstream stage. This is where a split or merge reading actually shows up —
the plan printed above and by ``--dry-run`` is pre-inference, but ``review
log`` afterward reports window 17's coalesced edit as ``kind=merge`` and
window 5's as ``kind=split``, each with the frequency that was requested. The
same Python entry points are available on the functional API and the
``Pipeline`` class:

.. code-block:: python

   import ftmwpipeline.api as ftmw

   preview = ftmw.review_apply("exp_2638.ftmw", "exp_2638_curation.csv", dry_run=True)
   for action in preview.plan:
       print(action.kind, action.window_id)
   print(preview.warnings)

   result = ftmw.review_apply("exp_2638.ftmw", "exp_2638_curation.csv")
   print(result.applied)   # number of actions refit

.. _curation-preview:

Previewing the fitted outcome
-----------------------------

``--dry-run`` answers "does this plan resolve against the file?" — it prints the
coalesced actions and any resolution warnings, and stops there. It never fits
anything, so it cannot tell you whether the edits *improve* the fit.

``review preview`` answers that second question. It runs the resolved plan to
completion in memory — every applier, the one combined cascade, and the same
derive step a live apply's persist would have used — and reports the fitted
outcome, without writing a byte:

.. code-block:: console

   $ ftmwpipeline review preview exp_2638.ftmw exp_2638_curation.csv
   review preview (nothing written):
     window    1  [  direct]  actions=2         peaks 2->3  chi2r 1.066->1.013
     window    5  [  direct]  actions=1         peaks 4->3  chi2r 1.029->1.339

Each row is one affected window, read *after* the cascade rather than
per-action: which plan actions touched it, whether the batch reached it
``direct`` (an action named it) or as a cascaded dependent, how its peak count
changed, and χ²ᵣ before and after. Here the window 1 add improves the fit and
the window 5 remove degrades it — the judgment the dry run cannot offer. A
window this batch *created* has no "before", and prints ``-`` on that side
rather than a ``0.000`` that would read as a perfect fit.

An ``add`` whose frequency no live window covers implies the window it needs
rather than erroring (above, under "Curation files"), and a file's own
``create`` row installs one explicitly. Either way the preview reports what
got built, not just that something did — the acceptance condition for
letting an ``add`` create structure on its own is that a typo'd frequency
shows up here as a stray window rather than silently landing somewhere
plausible:

.. code-block:: console

   $ ftmwpipeline review preview exp_2638.ftmw exp_2638_curation.csv
   review preview (nothing written):
     window    1  [  direct]  actions=2         peaks 2->3  chi2r 1.066->1.013
     window  298  [  direct]  actions=2         peaks 0->1  chi2r -->1.204
       Window 298 created: [30850.5734, 30850.6520] MHz (12 points, 2 frozen contributor(s))

The extra line only appears on a window the batch created or widened —
``PreviewWindowResult.created_window_mode`` is ``"created"`` or ``"widened"``
there, and ``None`` (no line) for a window the batch only edited, merged,
split, accepted, or cascaded into. When present, ``created_window_freq_range``
/ ``created_window_n_points`` / ``created_window_n_contributors`` /
``created_window_depends_on`` carry the rest of the structural facts, straight
off the create that ran in memory — a UI can render "this add creates a
window at A–B MHz" from these fields alone, without re-deriving anything.
``review apply --dry-run`` reports the same facts for a plan that implies (or
explicitly names) a create, since its own resolved-plan echo cannot know a
create's extent without running it.

A preview is not a weaker apply. It shares the appliers, so it raises the same
per-action error on the same failures, and it is epoch-gated by the same check,
so it cannot show you numbers whose apply is guaranteed to refuse. The one
thing it does not do is take the undo baseline snapshot, because it writes
nothing to snapshot against. A plan of nothing but bare ``accept`` rows touches
no fit and previews as ``(no fit-mutating actions; nothing to preview)``.

.. code-block:: python

   import ftmwpipeline.api as ftmw

   preview = ftmw.review_preview("exp_2638.ftmw", "exp_2638_curation.csv")
   for wid, w in sorted(preview.windows.items()):
       print(wid, w.n_peaks_before, "->", w.n_peaks_after, w.chi2r_after)
       if w.created_window_mode is not None:
           lo, hi = w.created_window_freq_range
           print(f"  {w.created_window_mode}: {lo:.4f}-{hi:.4f} MHz")

The three form a ladder, each answering the next question: ``--dry-run`` —
does the plan resolve? ``review preview`` — what does it fit to?
``review apply`` — commit it.

.. _curation-sessions:

Review sessions
---------------

Every fit-mutating Stage 6 verb rebuilds the same active-FT context before it
can do anything: roughly 420 ms of the ~516 ms an interactive single-window
edit costs cold. Stepping through a worklist one window at a time pays that
over and over.

A **review session** builds it once and reuses it across every verb issued
against the same file:

.. code-block:: python

   from ftmwpipeline import Pipeline

   # worklist: [(window_id, "uid:N" of the line to drop), ...]
   with Pipeline.open("exp_2638.ftmw").review_session() as session:
       for wid, target in worklist:
           result = session.review_edit(wid, remove=[target], frame="raw")
           print(wid, result.chi2r_before, "->", result.chi2r_after)
       session.review_accept(12)

The session hosts the whole verb set — ``review_edit``, ``review_accept``,
``review_create``, ``review_undo``, ``review_preview`` and ``review_apply`` —
not just the batch door, since an interactive click otherwise pays the full
price. Each takes the same arguments and returns the same result as the
``Pipeline`` method of the same name. The session is always obtained from
``review_session()``, never constructed, but its class is exported as
:class:`ftmwpipeline.ReviewSession` so you can annotate a function that takes
one. Warm-up happens synchronously in ``__enter__``; the session retains
about 26 MB of active-FT arrays for its lifetime, which is entirely yours to
control. Use the ``with`` block, or call ``close()``. There is no module-level
cache, so a session never opened costs nothing.

**Correctness never depends on the reuse.** Before every verb the session
re-reads a cheap on-disk fingerprint (~0.8 ms); if anything has moved — a
foreign writer touched the file — it rebuilds from scratch, which is
byte-for-byte the rebuild a sessionless caller gets on every call anyway. After
each of the session's own writes the fingerprint is re-read from disk rather
than predicted, so a foreign writer landing in the same instant is still caught
on the next call. A session holds no lock and does not protect the file from a
second writer: single-writer discipline per file is yours, exactly as it is
without a session.

Preview then apply
~~~~~~~~~~~~~~~~~~

Inside a session the preview above becomes more than advisory. ``review_preview``
stages its finished, cascaded, never-persisted outcome, so an immediately
following ``review_apply`` of the *identical* plan against an unmoved base
persists that result directly instead of computing it a second time:

.. code-block:: python

   with Pipeline.open("exp_2638.ftmw").review_session() as session:
       preview = session.review_preview("exp_2638_curation.csv")
       fitted = [w.chi2r_after for w in preview.windows.values()
                 if w.chi2r_after is not None]
       if fitted and max(fitted) < 2.0:
           result = session.review_apply("exp_2638_curation.csv")

The bytes persisted are then guaranteed to be exactly the ones the preview
showed, rather than a second computation trusted to agree with the first.

The staging is dropped the moment it stops being valid: a different curation
file, a different ``frame``, a resolved plan that differs, or a base that moved
(a foreign write, or another mutating verb issued on the session in between).
Any of those falls back to a full ordinary apply, identical to the sessionless
one — nothing is lost but the saving. When a staged preview was dropped
specifically because the base moved, the result's ``base_changed`` flag says
so, which is the cue to re-run the preview and look again before trusting it.
