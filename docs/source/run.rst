.. index::
   single: run command
   single: whole-pipeline command
   single: passthrough; per-stage knobs

The ``run`` command
====================

``run`` drives a raw data source through the entire pipeline — import, Fourier
transform, noise estimation, decay-time calibration, peak detection, window
assignment, fitting, timebase calibration, and review — in a single call, with
live per-stage progress. It is the primary entry point for processing an
experiment: use it when you want a finalized, review-ready ``.ftmw`` file (and
optionally its report) without driving each stage by hand. :doc:`quickstart`
shows it first for exactly that reason; drive the stages individually instead
when you need to inspect or tune an intermediate result before moving on (see
the stage pages, :doc:`stage0_import` through :doc:`stage6_review`).

``run`` is orchestration only — it adds no analysis of its own. Every stage
runs through the same shared implementation the standalone ``<stage> run``
subcommands use, so a ``run`` build and the equivalent sequence of per-stage
commands produce identical results.

Basic use
---------

The active-band trim is the one experiment-specific input ``run`` cannot infer
— there is no active-band auto-detector — so it is required:

.. code-block:: console

   $ ftmwpipeline run examples/blackchirp_data/2638/ \
       --trim 26500:40000 \
       --output exp_2638.ftmw

The same build through the class API:

.. code-block:: python

   from ftmwpipeline import Pipeline

   result = Pipeline.build(
       "examples/blackchirp_data/2638/",
       trim=(26500, 40000),
       output="exp_2638.ftmw",
   )
   pipe = Pipeline.open(result["pipeline_file"])

or the functional API:

.. code-block:: python

   import ftmwpipeline.api as ftmw

   result = ftmw.run_pipeline(
       "examples/blackchirp_data/2638/",
       output="exp_2638.ftmw",
       trim=(26500, 40000),
   )

All three return (or produce, at the CLI, a one-line summary of) a structured
result: ``pipeline_file``, ``status`` (``"success"`` / ``"error"``),
``completed_stages``, ``failed_stage`` and ``error`` (when it failed),
``timebase`` (``"calibrated"`` / ``"skipped"`` / ``"not_requested"``),
``report`` (the report paths, or ``None``), and ``elapsed_s``. A failing build
stops at the first stage that raises; ``completed_stages`` names everything
that finished before it.

What it runs
------------

.. code-block:: text

   import -> [start detection] -> FT -> noise -> tau -> peaks -> windows
     -> fit -> [timebase] -> review -> [report]

* **Start detection** and **tau calibration** run by default. Start detection
  stamps a recommended FID ``start_us`` that the FT inherits (see
  :ref:`run-stage0` below); tau calibration stamps the majority-vote decay
  constant and the recommended line shape that the fit consumes — skipping it
  silently degrades the fit to a naive ``T_active/3`` decay and a default
  shape, so there is no ``--no-tau`` toggle.
* **Timebase calibration** runs by default but is **non-fatal**: it resolves
  the instrument clock declaration (explicit ``--clocks`` > a preset's
  persisted ``spur.clocks`` > the clock sources auto-extracted at import) and,
  when none is resolvable, is skipped with a warning rather than failing the
  build — frequencies are then reported as precision-only. ``--no-cal`` skips
  it deliberately, without the warning. See :doc:`clock_declaration`.
* **Review** always runs last (before an optional report) to consolidate the
  finalized line list.
* ``--no-force`` refuses to overwrite an existing ``.ftmw`` built from a
  different source; the default is a fresh build every time, since persisted
  settings on a stale file would otherwise silently outrank the current
  defaults (see :doc:`settings_and_presets`).

Top-level options
------------------

* ``--output PATH`` — the destination ``.ftmw`` file; derived from the source
  name in the current directory if omitted.
* ``--trim MIN:MAX`` — the active-band FT range in MHz (**required**);
  forwarded to the FT stage exactly as ``ft run --trim`` would take it.
* ``--preset NAME`` — a packaged preset name or a path to a ``.yaml`` /
  ``.yml`` file, forwarded to every stage that accepts one (noise, tau, peaks,
  windows, fit). See `Reusable configurations with YAML presets`_ below.
* ``--sigma-floor KHZ`` — the systematic frequency-accuracy floor folded into
  the review stage's :math:`\sigma_f` budget.
* ``--clocks SPEC`` — an explicit instrument clock declaration for timebase
  calibration and the Stage 5 spur gate: comma-separated MHz fundamentals,
  with ``:u`` appended to mark one unlocked (e.g. ``--clocks 5120,5760,6250:u``).
  Overrides auto-detection. See :doc:`clock_declaration`.
* ``--report`` — also emit the Level-1 line-list table and the Level-3 HTML
  report once review finishes; ``--report-dir DIR`` places them (default
  ``<stem>_report/`` next to the output file).
* ``--format NAME`` / ``--fid-index N`` — force an input-format loader instead
  of auto-detecting one, and pick a FID index for multi-FID formats (e.g.
  Blackchirp). See :doc:`input_formats`.
* ``--quiet`` — suppress the live per-stage progress display (errors and
  warnings still print).
* ``--no-start-detect`` — skip start-time detection; the FT then uses whatever
  ``start_us`` a preset, a persisted value, or the hard default supplies.
* ``--no-cal`` — skip timebase calibration deliberately (no warning).
* ``--no-force`` — refuse to overwrite a file built from a different source
  instead of the default fresh rebuild.

Per-stage knobs (passthrough)
------------------------------

Beyond the always-on flags above, ``run`` also exposes every stage's
individual tuning knobs as **namespaced flags**, ``--<stage>.<knob>``, for the
seven stages that have a settings dataclass: ``start``, ``ft``, ``noise``,
``tau``, ``peaks``, ``windows``, and ``fit``. Each group is generated from the
exact same settings class the stage's own ``<stage> run`` subcommand uses (see
:doc:`cli` and :doc:`settings_and_presets`), so ``run --fit.foo`` and
``fit run --foo`` mean the same knob — ``run`` mirrors each stage's own
flags rather than keeping a separate copy that could drift from them. A
sub-block field is reached by
chaining another dot: ``--fit.tau.max-decay-factor``,
``--peaks.promotion.min-snr``, ``--windows.contributor.skirt-level-keep``.

Two realistic examples:

.. code-block:: shell

   # Tighten the peak-promotion floor and cap the fit's decay-time bounds
   ftmwpipeline run examples/blackchirp_data/2638/ \
     --trim 26500:40000 \
     --peaks.promotion.min-snr 6.0 \
     --fit.tau.max-decay-factor 3.0

.. code-block:: shell

   # Widen the noise scatter window and disable the matched-filter gap pass
   ftmwpipeline run examples/blackchirp_data/2638/ \
     --trim 26500:40000 \
     --noise.window-mhz 150.0 \
     --no-peaks.gap_pass.run-gap-pass

``--trim`` above is the canonical way to set the active-band range;
``--ft.trim`` is accepted as an alias of the same flag (not a separate,
independent namespaced knob) for consistency with the ``ft.`` namespace.
Every other ``FTSettings`` field (``--ft.start-us``, ``--ft.end-us``,
``--ft.units-power``) is reachable through the ``ft.`` namespace. ``timebase``,
``review``, and ``report`` have no namespaced knobs — they expose no per-stage
settings surface for ``run`` to mirror.

An explicit ``--<stage>.<knob>`` flag and the other top-level flags win over
``--preset``, which in turn wins over a file's persisted values, an upstream
stage's recommendation, and the hard defaults — the same five-layer
resolution chain every stage uses (:doc:`settings_and_presets`). ``--preset``
and the namespaced flags **compose**: a preset sets the baseline recipe and an
explicit flag overrides one field of it, exactly as `Reusable configurations
with YAML presets`_ demonstrates below.

The full generated list is long and always available from
``ftmwpipeline run --help``; it is not reproduced here.

.. _run-stage0:

Controlling Stage 0 (start time)
-----------------------------------

By default ``run`` performs start-time detection before the FT and stamps a
recommended FID ``start_us`` — the detected chirp end plus a guard margin —
which the FT inherits. From ``run`` you reach for one of two levers:

* ``--start.guard-margin-us`` retunes the margin added past the chirp end for
  the switch-bounce ring-down, adjusting the *recommendation* (inert if you
  also pass ``--ft.start-us``).
* ``--ft.start-us`` overrides the FID window start outright; an explicit value
  always wins over the stamped recommendation.
* ``--no-start-detect`` skips detection entirely, so the FT falls through to
  whatever a preset, a persisted value, or the hard default supplies.

The full detector mechanism and the complete ``--start.*`` knob reference live
on the Stage 0 page: see :ref:`stage0-detector-settings`.

Reusable configurations with YAML presets
--------------------------------------------

``--preset`` accepts a packaged name (``defaults``, a documentation copy of
the hard defaults) or a path to a YAML file you wrote yourself. The full
preset schema — the five stage blocks
(``stage2`` / ``stage2b`` / ``stage3`` / ``stage4`` / ``stage5``), their
sub-blocks, and the resolution precedence — is documented in
:doc:`settings_and_presets`; this section shows it applied through ``run``.

Presets deliberately do not carry FT or start-detection settings (``stage1``
and Stage 0 are outside the preset schema), which is exactly why the
namespaced ``--ft.*`` / ``--start.*`` flags matter alongside a preset: the
preset pins the recurring instrument recipe for noise through fitting, and the
namespaced flags cover the per-run, source-specific start time and FT window.

A small recipe for a noisy instrument that needs a looser peak-promotion floor
and a shorter fit decay bound:

.. code-block:: yaml

   # my_lab_recipe.yaml
   name: my_lab_recipe
   description: tuned for the bench 2 receiver (higher noise floor)

   stage3:
     promotion:
       min_snr: 6.0
   stage5:
     tau:
       max_decay_factor: 3.0

Run it:

.. code-block:: shell

   ftmwpipeline run examples/blackchirp_data/2638/ \
     --trim 26500:40000 \
     --preset my_lab_recipe.yaml

And a variant showing a preset and an explicit override composing — the
recipe sets the baseline rescue budget, and the per-run flag tightens it
further for one difficult spectrum, without editing the YAML:

.. code-block:: shell

   ftmwpipeline run examples/blackchirp_data/2638/ \
     --trim 26500:40000 \
     --preset my_lab_recipe.yaml \
     --fit.rescue.max-rounds 3

The explicit ``--fit.rescue.max-rounds`` lands in the top resolution layer;
the preset seeds everything else beneath it (and beneath anything already
persisted on the file from a prior run). See :doc:`settings_and_presets` for
the complete precedence chain and the full preset field reference.
