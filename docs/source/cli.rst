.. index::
   single: command-line interface
   single: CLI
   single: object-verb grammar
   single: ftmwpipeline command

Command-Line Reference
======================

The ``ftmwpipeline`` command-line interface follows an **object-verb** grammar::

    ftmwpipeline <object> <verb> <file.ftmw> [options]

The *object* names a pipeline stage or a cross-cutting capability; the *verb* is
the action. Every stage object accepts ``run`` (compute and persist) and
``show`` (visualize), and a few add a third verb. Each stage object also has a
``stageN`` synonym that is interchangeable with its name — ``ft run`` and
``stage1 run`` are the same command.

This page is a map of the surface. It mirrors the Python API (:doc:`api/index`)
and the :class:`~ftmwpipeline.pipeline.Pipeline` class: a CLI verb, the matching
API function, and the class method run the same shared implementation and
produce identical results. The conceptual detail for each stage — what the
algorithm does and how to read its output — lives on the stage pages
(:doc:`stage0_import` through :doc:`stage6_review`); this page documents the
command structure and the options that shape it.

The full, authoritative option list for any command is always its ``--help``::

    ftmwpipeline ft run --help

Conventions
-----------

* **File argument.** Most commands take the ``.ftmw`` file path as the trailing
  positional argument; the ``.ftmw`` extension is added if omitted. ``data
  import`` and ``run`` instead take a raw data *source*.
* **Per-knob flags vs. presets.** A stage's ``run`` verb exposes its tuning
  knobs as ``--flag`` options and also accepts ``--preset``. The flags are the
  explicit resolution layer; the preset sits beneath the file's persisted
  values. They compose. The knob meanings are documented on the stage pages and
  in :doc:`settings_and_presets`; the same knobs are reachable by dotted path
  through ``settings set`` and ``scan``.
* **run persists, show does not.** ``run`` writes its result and parameters into
  the file (invalidating downstream stages when canonical settings change);
  ``show`` only renders and never persists.
* **Visualization output.** ``show`` verbs display interactively by default and
  write an image file when given ``--output`` / ``-o`` (often with
  ``--no-interactive``). Direct plots to an untracked location.
* **Verbosity.** ``-v`` / ``--verbose`` enables detailed logging on every
  command.

Command summary
---------------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Object (synonym)
     - Verbs
     - Purpose
   * - ``data`` (``stage0``)
     - ``import``, ``show``
     - Import a raw source; view the FID
   * - ``start``
     - ``run``, ``show``
     - Detect and stamp the FID ``start_us``
   * - ``ft`` (``stage1``)
     - ``run``, ``show``
     - Compute the canonical Fourier transform
   * - ``noise`` (``stage2``)
     - ``run``, ``show``
     - Estimate per-bin noise
   * - ``tau`` (``stage2b``)
     - ``run``, ``recommend``, ``show``
     - Decay-time calibration; line-shape vote
   * - ``timebase``
     - ``run``, ``show``
     - Digitizer-clock scale-error calibration
   * - ``peaks`` (``stage3``)
     - ``run``, ``show``
     - Detect and classify peaks
   * - ``windows`` (``stage4``)
     - ``run``, ``show``
     - Plan fit windows
   * - ``fit`` (``stage5``)
     - ``run``, ``show``, ``check``
     - Fit lines; assess the fit
   * - ``review`` (``stage6``)
     - | ``run``, ``show``, ``rank``, ``edit``, ``merge``,
       | ``split``, ``accept``, ``apply``, ``log``, ``undo``
     - Curate the fitted model
   * - ``report``
     - ``table``, ``run``
     - Export the finalized line list and report
   * - ``scan``
     - ``list``, ``run``, ``all``
     - Sweep tunable knobs for an instrument
   * - ``settings``
     - ``show``, ``set``, ``export``
     - Inspect / persist / export resolved settings
   * - ``clocks``
     - ``show``, ``set``, ``add``, ``remove``, ``clear``
     - Declare instrument clock fundamentals
   * - ``run``
     - —
     - Drive a raw source through every stage
   * - ``info``
     - —
     - Provenance and stage status of a file
   * - ``formats``
     - —
     - List available input-format loaders
   * - ``validate``
     - —
     - Check installation and dependencies
   * - ``version``
     - —
     - Show version and package information

Stage commands
--------------

``data`` — import and view the FID (Stage 0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``data import <file.ftmw> <source>`` loads a raw source into a new ``.ftmw``
file. The format auto-detects (``--format`` to force one of ``blackchirp``,
``csv``, ``ftmw-hdf5``, ``keysight-mat``). Re-importing the same source is safe;
``--force`` overwrites a file built from a different source. Generic inputs take
acquisition metadata from flags (``--spacing_us``, ``--probe_freq_mhz``,
``--sideband``, ``--column`` for a CSV) or a ``--metadata`` sidecar; segmented
raw-scope records add the ``--frame-period-us`` / ``--n-frames`` /
``--interleave-factors`` family. See :doc:`stage0_import`, :doc:`input_formats`,
and :doc:`scope_record_import`.

``data show <file.ftmw>`` plots the cached FID; ``--show-metadata`` prints the
acquisition record.

``start`` — start-time detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``start run`` sweeps the FID window start time, finds the chirp-end collapse,
and stamps a recommended ``start_us`` (chirp end plus a guard margin) into the
Stage 0 recommended layer, so a later ``ft run`` with no explicit ``--start-us``
inherits it. ``--no-stamp`` reports without writing. ``start show`` draws the
Σ\|FT\|-vs-``start_us`` sweep diagnostic. Knobs: ``--sweep-max-us``,
``--step-us``, ``--guard-margin-us``, ``--floor-factor``, ``--band``, plus the
less commonly tuned ``--floor-tail-us`` and ``--min-chirp-drop-ratio``.

``ft`` — Fourier transform (Stage 1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ft run`` computes and persists the canonical spectrum. The canonical FT is
unconditionally unapodized and native-length; the only settings are data
selection — ``--start-us`` / ``--end-us`` (the active FID window) and ``--trim
MIN:MAX`` (the analysis band in MHz) — plus ``--units-power`` scaling. ``--trim``
is persisted as canonical and binds every downstream stage. ``ft show`` renders
the FID-to-spectrum panels for parameter exploration and never persists. See
:doc:`stage1_ft`.

``noise`` — noise estimation (Stage 2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``noise run`` estimates the per-bin noise with the scatter estimator. Knobs
cover the region-aware scatter-MAD (``--window-mhz``, ``--pedestal-mhz``,
``--line-k``, ``--n-iter``, ``--region-aware`` / ``--no-region-aware``) and the
broad σ smoothing (``--smoothing-mhz``, ``--smoothing-percentile``,
``--convolve-mhz``). ``noise show`` draws the σ(f) diagnostic. See
:doc:`stage2_noise`.

``tau`` — decay-time calibration (Stage 2b)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tau run`` extracts the majority-vote decay constant ``tau_maj`` and its spread
from a sliding-window STFT on the raw FID. ``--gaussian`` runs the independent
Gaussian τ\ :sub:`G` twin instead; the two coexist on one file. ``tau
recommend`` runs the 3-way Lorentzian/Gaussian/Voigt shape vote and stamps the
winner that Stages 3 and 5 read (run automatically by ``tau run`` when
auto-recommend is on). ``tau show --kind heatmap|distribution`` (add
``--gaussian`` to view the τ\ :sub:`G` group) draws the diagnostics. See
:doc:`stage2b_tau`.

``timebase`` — digitizer-clock calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``timebase run`` measures the digitizer-clock fractional scale error ε from the
Rb-locked spur lattice and persists it; the clock declaration comes from the
Stage 5 ``spur.clocks`` settings (declare it first via ``clocks`` or a preset).
``--kappa-sys`` and ``--snr-min`` tune the per-tone budget and detection gate.
``timebase show`` prints ε ± σ and the per-tone table. See
:doc:`clock_declaration`.

``peaks`` — peak detection (Stage 3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``peaks run`` runs the two-pass detector on the canonical spectrum and
classifies peaks by SNR. Knobs: the promotion / classification SNR boundaries
(``--min-snr``, ``--weak-medium-snr``, ``--medium-strong-snr``), the
Savitzky-Golay locator (``--sg-window``, ``--sg-order``), and the matched-filter
gap pass (``--primary-window``, ``--min-exclusion-mhz``, ``--gap-pass`` /
``--no-gap-pass``). ``peaks show`` overlays the classified peaks; ``--snr-histogram``
adds the promotion-cutoff curation panel. See :doc:`stage3_peaks`.

``windows`` — window assignment (Stage 4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``windows run`` turns the promoted peaks into a plan of disjoint fit windows
with their frozen-leakage contributors and fit dependency order. Knobs cover the
edge-coherence statistic (``--edge-m``, ``--trim-m``, ``--edge-threshold``), the
window-width and margin caps (``--max-window-width-mhz`` /
``--max-window-width-points``, ``--min-window-half-width-points``,
``--max-peaks-per-window``), and contributor attachment (``--min-freeze-snr``,
``--magnitude-attachment-threshold``, ``--tau-us``). ``windows show`` overlays
the plan. See :doc:`stage4_windows`.

``fit`` — peak fitting (Stage 5)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``fit run`` fits each window's lines via the conservative add-one-peak loop with
the shared per-window decay and frozen contributors, then the residual
edge-coherence handshake. ``--shape {lorentzian,gaussian}`` selects the line
shape (default Lorentzian); ``--tau0-us``, ``--fit-tau`` / ``--no-fit-tau``,
``--max-decay-factor``, and ``--per-band-tau`` control the decay; the rescue and
thaw/replan caps (``--max-residual-rescue-rounds``, ``--max-thaw-rounds``,
``--max-replan-rounds``, …) bound the iterative passes. ``--tau-maj-override`` /
``--sigma-tau-override`` force a decay anchor for A/B work. ``-j`` / ``--jobs``
sets the cross-window worker pool (see :doc:`performance`).

``fit show`` draws the fit: with no selector, the spectrum-wide overview;
with window selectors (``--window``, ``--window-list``, ``--freq``,
``--freq-list``, ``--random``, ``--top-snr``, ``--all-windows``), a consolidated
per-window detail figure for each. ``--apodize WINDOW`` adds a diagnostic
apodized data-vs-model comparison, and ``--rescue`` a residual-rescue summary
(data/model, the residual with each round's nominations, and the per-round
:math:`\chi^2` descent and peak budget). ``fit check`` is a read-only health
assessment against the SNR-aware acceptance gate, with a Tier-3 catalog
comparison under ``--ground-truth``. See :doc:`stage5_fitting`.

``review`` — curation (Stage 6)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``review run`` builds the attention-routing layer (per-window advisory reasons),
preserving existing provenance and the decision log; ``--sigma-floor`` declares
the systematic frequency-accuracy floor folded into the σ\ :sub:`f` budget.
``review show`` lists per-window summaries or the candidate ledger
(``--candidates``); ``review rank --by METRIC`` ranks windows worst-first by a
persisted statistic. The editing verbs each re-fit the affected window and
record the change in the decision log: ``edit`` (``--add`` / ``--remove``),
``merge`` (``--peaks``), ``split`` (``--peak`` / ``--into``), and ``accept``.
``apply`` replays a curation CSV of batched edits; ``log`` lists the decision
log; ``undo --id N`` rolls a decision back by replay-from-baseline. See
:doc:`stage6_review`.

``report`` — finalized deliverables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``report`` renders the finalized record and never recomputes the fit; it
requires ``review run`` to have consolidated the calibrated final products.
``report table`` exports the Level-1 line table as ``--format`` ``csv`` /
``json`` / ``latex``. ``report run`` writes the default deliverables — the
Level-1 CSV and the self-contained Level-3 HTML report — into ``--output-dir``
(the current directory by default); trim with ``--level1-only``, ``--no-table``,
``--windows attention``, or ``--summary``, and parallelize the figure rendering
with ``-j`` / ``--jobs``. ``report table`` and ``report run`` accept
``--catalog`` / ``--catalog-nsigma`` to proximity-flag each line against a
frequency catalog (label echo only, never an assignment). ``report diff`` writes
a self-contained ``<stem>_diff.html`` comparing the automatic fit with the
current curated fit, side by side, for every window a curation (or its cascade)
changed materially — a before/after review aid for vetting edits before
committing them, without juggling two files. See :doc:`stage6_review`.

Cross-cutting commands
----------------------

``settings`` — resolved settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``settings show`` reports, per setting, the value in effect for a file and the
layer that supplied it (file / preset / recommended / default); ``--all``
includes advanced-tier knobs and ``--preset`` previews a preset's contribution.
``settings set <file> <knob> <value>`` persists a dotted-path knob (e.g.
``stage2.window_mhz``) and invalidates the affected and downstream stages.
``settings export`` writes the file's persisted Stage 2–5 values to a reusable
``.yml`` preset (Stage 1 FT settings are excluded). See
:doc:`settings_and_presets`.

``scan`` — knob sweeps
~~~~~~~~~~~~~~~~~~~~~~~

``scan list`` enumerates the tunable knobs grouped by stage; ``scan run --knob
KNOB --grid …`` sweeps one knob across a grid on a copy of the file and reports a
metric table (plus a CSV and, where available, a plot); ``scan all [selector]``
batches that sweep over a whole stage or sub-block. The Stage 5 sampling flags
(``--fit-top-snr``, ``--fit-sample``, ``--fit-all``, …) bound how many windows a
fitting-stage sweep re-fits.

``clocks`` — instrument clock declaration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``clocks`` declares the instrument clock fundamentals the Stage 5 spur gate
builds its lattice prior from, writing the recommended declaration layer.
``show`` lists them; ``set`` replaces and ``add`` appends sources in the
``freq[:locked|free[:label]]`` token form (e.g. ``5760:locked:synth``);
``remove`` drops sources by frequency and ``clear`` empties the declaration. See
:doc:`clock_declaration`.

Whole-pipeline command
----------------------

``run <source>`` drives a raw source through every stage in sequence — import →
FT → noise → tau → peaks → windows → fit → timebase → review — with live
per-stage progress. ``--trim MIN:MAX`` (the active band) is required;
``--output`` names the destination file. Start detection and tau calibration run
by default; timebase calibration runs by default but is non-fatal and skippable
with ``--no-cal``. ``--report`` also emits the Level-1 table and Level-3 HTML
report. ``--preset`` forwards a settings preset to the stages that accept one,
and ``--clocks`` declares the instrument clocks for the spur gate and timebase
calibration. Every stage's individual knobs are also reachable as namespaced
``--<stage>.<knob>`` flags. See :doc:`run` for the full reference, including
Stage 0 start-time control and worked preset examples.

Utility commands
----------------

* ``info <file.ftmw>`` — print the provenance record and stage-completion status;
  ``--format json`` for machine-readable output.
* ``formats`` — list the registered input-format loaders; ``--format NAME`` for
  one loader's details.
* ``validate`` — check the installation and its dependencies.
* ``version`` — show the version and optional-dependency status.
