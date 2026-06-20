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
canonical Fourier transform is trimmed to that range. The transform itself is
unapodized and native-length.

The commands below write a new ``exp_2638.ftmw`` file in the current directory.

Running the whole pipeline
--------------------------

The ``run`` command drives a raw source through every stage in order — import,
Fourier transform, noise, decay-time calibration, peak detection, window
assignment, fitting, timebase calibration, and review — with live per-stage
progress. The active-band trim is required:

.. code-block:: bash

   ftmwpipeline run examples/blackchirp_data/2638/ \
       --trim 26500:40000 \
       --output exp_2638.ftmw

Add ``--report`` to also emit the line-list table and an HTML report.

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
is available. The build stops at the first stage that fails.

Running the stages individually
-------------------------------

Driving the stages one at a time gives control over each step's parameters and
lets you inspect the intermediate results. The three interfaces are
interchangeable; the same sequence is shown in each.

At the command line, every stage runs with ``<stage> run``:

.. code-block:: bash

   ftmwpipeline data import  exp_2638.ftmw examples/blackchirp_data/2638/
   ftmwpipeline ft run       exp_2638.ftmw --trim 26500:40000
   ftmwpipeline noise run    exp_2638.ftmw
   ftmwpipeline tau run      exp_2638.ftmw
   ftmwpipeline peaks run    exp_2638.ftmw
   ftmwpipeline windows run  exp_2638.ftmw
   ftmwpipeline fit run      exp_2638.ftmw

With the ``Pipeline`` class, an instance is bound to one file:

.. code-block:: python

   from ftmwpipeline import Pipeline

   pipe = Pipeline.create("exp_2638.ftmw", source="examples/blackchirp_data/2638/")
   pipe.compute_ft(trim=(26500, 40000))
   pipe.estimate_noise()
   pipe.calibrate_tau()
   pipe.detect_peaks()
   pipe.assign_windows()
   pipe.fit_peaks()

With the functional API, each call takes the file path:

.. code-block:: python

   import ftmwpipeline.api as ftmw

   ftmw.import_data("exp_2638.ftmw", source="examples/blackchirp_data/2638/")
   ftmw.compute_ft("exp_2638.ftmw", trim=(26500, 40000))
   ftmw.estimate_noise("exp_2638.ftmw")
   ftmw.calibrate_tau("exp_2638.ftmw")
   ftmw.detect_peaks("exp_2638.ftmw")
   ftmw.assign_windows("exp_2638.ftmw")
   ftmw.fit_peaks("exp_2638.ftmw")

Each stage requires its predecessor to be complete and fails with a message
naming the missing dependency otherwise. Re-running a stage with new parameters
is safe; re-running an earlier stage invalidates the results that depended on
it.

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

After the review stage, the finalized line list and an HTML report are written
by the ``report`` command:

.. code-block:: bash

   ftmwpipeline review run exp_2638.ftmw
   ftmwpipeline report run exp_2638.ftmw

Where to go next
----------------

* :doc:`settings_and_presets` — tune a stage's parameters and save recipes.
* The :doc:`stage pages <stage0_import>` — what each stage does and how to read
  its output.
* :doc:`cli` — the full command reference.
