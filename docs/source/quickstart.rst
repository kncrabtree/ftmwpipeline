.. index::
   single: quickstart
   single: run command
   single: example data

Quickstart
==========

This guide processes one experiment from a raw free-induction decay to a fitted
line list, first in a single command and then stage by stage. It uses the
example experiment included with the package.

Example data
------------

``examples/blackchirp_data/2638/`` is a real Blackchirp experiment provided for
testing and exploration: a 15 µs FID of 750,000 points, probe frequency
40.96 GHz, lower sideband. Its active spectral band is 26500–40000 MHz, so the
Fourier transform is trimmed to that range. The transform itself is
unapodized and native-length.

The commands below write a new ``exp_2638.ftmw`` file in the current directory.

Running the whole pipeline
--------------------------

The ``run`` command drives a raw source through every stage in order — import,
start-time detection, Fourier transform, timebase calibration, noise,
decay-time calibration, peak detection, window assignment, fitting, and
review — with live per-stage progress. The active-band trim is required:

.. code-block:: bash

   ftmwpipeline run examples/blackchirp_data/2638/ \
       --trim 26500:40000 \
       --output exp_2638.ftmw

Add ``--report`` to also emit the line-list table and an HTML report. See
:doc:`run` for the full option reference — the per-stage knob passthrough,
controlling the Stage 0 start time, and worked ``--preset`` examples.

The same end-to-end build from Python, through the class API:

.. code-block:: python

   from ftmwpipeline import Pipeline

   result = Pipeline.build(
       "examples/blackchirp_data/2638/",
       trim=(26500, 40000),
       output="exp_2638.ftmw",
   )
   pipe = Pipeline.open(result["pipeline_file"])

or through the functional API:

.. code-block:: python

   import ftmwpipeline.api as ftmw

   ftmw.run_pipeline(
       "examples/blackchirp_data/2638/",
       "exp_2638.ftmw",
       trim=(26500, 40000),
   )

Timebase calibration and start detection run by default; timebase calibration
is non-fatal and is skipped with a warning when no instrument clock declaration
is available. The build stops at the first stage that fails. It does not raise
for a stage failure: it returns a result dict whose ``status`` is
``"success"`` or ``"error"``, with ``failed_stage`` (the canonical stage name,
``None`` when the failing step is not a stage), ``failed_step`` (the failing
step's progress label) and ``error`` (the failure as an ``ftmw/error@1`` dict,
see :doc:`machine_contract`) set on failure, and ``completed_stages`` listing
the canonical stages written. Every completed stage stays written in the file. Interrupting the
build (Ctrl-C, or a cancel token) raises ``OperationCancelledError`` instead,
whose ``completed_stages`` names the stages that finished.

Running the stages individually
-------------------------------

Driving the stages one at a time gives control over each step's parameters and
lets you inspect the intermediate results. The three interfaces are
interchangeable; the same sequence is shown in each.

At the command line, every stage runs with ``<stage> run``. This sequence
reproduces the ``run`` command above exactly:

.. code-block:: bash

   ftmwpipeline data import   exp_2638.ftmw examples/blackchirp_data/2638/
   ftmwpipeline start run     exp_2638.ftmw
   ftmwpipeline ft run        exp_2638.ftmw --trim 26500:40000
   ftmwpipeline timebase run  exp_2638.ftmw
   ftmwpipeline noise run     exp_2638.ftmw
   ftmwpipeline tau run       exp_2638.ftmw
   ftmwpipeline peaks run     exp_2638.ftmw
   ftmwpipeline windows run   exp_2638.ftmw
   ftmwpipeline fit run       exp_2638.ftmw
   ftmwpipeline review run    exp_2638.ftmw

``start run`` and ``timebase run`` are not dependencies of the later stages,
but leaving them out changes the result: without ``start run`` the Fourier
transform starts from whatever start the import recommended rather than the
detected one, so the spectrum, the peaks, and the fitted windows differ; without
``timebase run`` the reported frequencies stay uncalibrated.

With the ``Pipeline`` class, an instance is bound to one file:

.. code-block:: python

   from ftmwpipeline import Pipeline

   pipe = Pipeline.create("exp_2638.ftmw", source="examples/blackchirp_data/2638/")
   pipe.detect_start_time()
   pipe.compute_ft(trim=(26500, 40000))
   pipe.calibrate_timebase()
   pipe.estimate_noise()
   pipe.calibrate_tau()
   pipe.detect_peaks()
   pipe.assign_windows()
   pipe.fit_peaks()
   pipe.review_run()

With the functional API, each call takes the file path:

.. code-block:: python

   import ftmwpipeline.api as ftmw

   ftmw.import_data("exp_2638.ftmw", source="examples/blackchirp_data/2638/")
   ftmw.detect_start_time("exp_2638.ftmw")
   ftmw.compute_ft("exp_2638.ftmw", trim=(26500, 40000))
   ftmw.calibrate_timebase("exp_2638.ftmw")
   ftmw.estimate_noise("exp_2638.ftmw")
   ftmw.calibrate_tau("exp_2638.ftmw")
   ftmw.detect_peaks("exp_2638.ftmw")
   ftmw.assign_windows("exp_2638.ftmw")
   ftmw.fit_peaks("exp_2638.ftmw")
   ftmw.review_run("exp_2638.ftmw")

Each stage requires its predecessor to be complete and otherwise fails with a
``stage_not_run`` error naming the missing dependency. Re-running a stage with
new parameters is safe; re-running an earlier stage discards the results that
depended on it, and the command names the discarded stages
(``Invalidated (re-run to refresh): ...``).

Inspecting the result
---------------------

Check provenance and which stages have run:

.. code-block:: bash

   ftmwpipeline info exp_2638.ftmw

Visualize a stage's output with ``<stage> show`` — for example the fitted model:

.. code-block:: bash

   ftmwpipeline fit show exp_2638.ftmw

The equivalent introspection from Python:

.. code-block:: python

   pipe = Pipeline.open("exp_2638.ftmw")
   print(pipe.info())

Producing a report
------------------

``review run`` (the last step of both sequences above) consolidates the
finalized line list; ``report run`` then writes it alongside an HTML report:

.. code-block:: bash

   ftmwpipeline report run exp_2638.ftmw

Where to go next
----------------

* :doc:`settings_and_presets` — tune a stage's parameters and save recipes.
* The :doc:`stage pages <stage0_import>` — what each stage does and how to read
  its output.
* :doc:`cli` — the full command reference.
