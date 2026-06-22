.. index::
   single: overview
   single: design philosophy
   single: pipeline stages
   single: interfaces; command-line, Pipeline, functional

Overview
========

``ftmwpipeline`` turns a raw Fourier transform microwave (FTMW) free-induction
decay into a calibrated, uncertainty-bearing list of spectral lines. Every stage
records what it did, the settings it used, and the statistical basis for its
output, so the result is fully inspectable and reproducible.

Purpose
-------

A single experiment enters as a time-domain FID — typically hundreds of
thousands of samples from a chirped-pulse spectrometer — and leaves as a table
of fitted center frequencies, amplitudes, and decay times (τ), each
with an error estimate and a calibrated frequency. Between those two endpoints
the pipeline computes the spectrum, characterizes the noise, calibrates the
molecular decay time, detects peaks, groups them into analysis windows, fits
each window with a finite-duration line-shape model, and consolidates the
result into a reviewed, reportable line list.

The analysis is **prior-free**: the pipeline does not assume a molecular model,
a catalog, or an expected line pattern. It reports what the data support, with
honest uncertainties, and leaves spectroscopic assignment to the user. Catalog
comparison is available as an optional, clearly labeled cross-reference that
never feeds back into the fit.

Design philosophy
-----------------

**One file, one experiment.** Each experiment is a single, self-contained,
portable ``.ftmw`` file (an HDF5 container) holding the raw FID, every stage's
result, the settings each stage used, and the provenance of the source data. The
raw record is preserved losslessly and never mutated, and exploring processing
parameters never re-imports or alters it. A stage's resolved settings are stamped
into the file, so a shared ``.ftmw`` reproduces the same result from the file
alone, independent of any local configuration; moving the analysis between
machines, or handing it to a collaborator, is a matter of copying one file. See
:doc:`file_format`.

**Staged and incremental.** The analysis is a sequence of stages with declared
dependencies, each persisting its result before the next begins. A stage reads
its inputs from the file, writes its output back, and records that it ran.
Re-running a stage with new parameters is safe and cheap, and re-running an
upstream stage invalidates the downstream results that depended on it, so the
file is never left in a silently inconsistent state.

**Unbiased spectrum, valid statistics.** The canonical spectrum that every later
stage measures is computed without apodization, time-domain windowing, or
zero-padding, so its per-bin noise and reduced χ² stay valid. Those operations
would trade frequency resolution, bias the line shape, or interpolate bins and
break the noise model that detection thresholds and fitted uncertainties rest on.
The intended way to trade variance for robustness is the per-window fit, never a
transform-level window.

**Honest, correlated uncertainties.** Every fitted quantity carries an
uncertainty, propagated with its correlations rather than as an independent error
bar. A reported precision reflects what the data constrain; an uncertainty the
data cannot determine is surfaced as such, never silently assumed away or
fabricated. The per-bin noise measured in :doc:`Stage 2 <stage2_noise>` and the
count of statistically independent samples in the band — not heuristics — back
every acceptance decision and reported uncertainty.

**Three interfaces, one implementation.** The pipeline is usable three ways — a
command-line interface, a ``Pipeline`` class, and a stateless functional API —
and all three produce identical results because they share a single underlying
implementation. The choice between them is a matter of workflow, not
capability.

The three interfaces
--------------------

**Command-line interface.** An object-verb grammar: the first token names a
pipeline stage (or a cross-cutting tool), the second names the action. Every
stage runs with ``<stage> run`` and is visualized with ``<stage> show``::

   ftmwpipeline data import exp.ftmw path/to/source/
   ftmwpipeline ft run      exp.ftmw --trim 26500:40000
   ftmwpipeline noise run   exp.ftmw

A full experiment can be driven end to end with the ``run`` command. The CLI is
covered in :doc:`cli`.

**Pipeline class.** A ``Pipeline`` instance is bound to one ``.ftmw`` file and
exposes a method per stage:

.. code-block:: python

   from ftmwpipeline import Pipeline

   pipe = Pipeline.create("exp.ftmw", source="path/to/source/")
   pipe.compute_ft(trim=(26500, 40000))
   pipe.estimate_noise()

**Functional API.** A stateless surface where every function takes the
``.ftmw`` path as its first argument — the natural fit for scripting and batch
work:

.. code-block:: python

   import ftmwpipeline.api as ftmw

   ftmw.import_data("exp.ftmw", source="path/to/source/")
   ftmw.compute_ft("exp.ftmw", trim=(26500, 40000))
   ftmw.estimate_noise("exp.ftmw")

The functional namespace is ``ftmwpipeline.api``; the package top level exports
the ``Pipeline`` class, the ``api`` module, and the core data structures.

The stage pipeline at a glance
------------------------------

An experiment progresses through the stages below. Each has a dedicated page
describing its algorithm, its assumptions, and how to read and trust its
output.

.. list-table::
   :header-rows: 1
   :widths: 10 26 64

   * - Stage
     - Name
     - Role
   * - 0
     - :doc:`Data import <stage0_import>`
     - Load a raw FID from an instrument format and record its provenance.
   * - 1
     - :doc:`Fourier transform <stage1_ft>`
     - Compute the canonical, unapodized, native-length spectrum over the
       active band.
   * - 2
     - :doc:`Noise estimation <stage2_noise>`
     - Measure the per-bin noise that every later stage uses as its
       statistical reference.
   * - 2b
     - :doc:`Decay-time calibration <stage2b_tau>`
     - Extract the molecular decay time from the FID and recommend a line
       shape.
   * - 3
     - :doc:`Peak detection <stage3_peaks>`
     - Locate peaks on the spectrum.
   * - 4
     - :doc:`Window assignment <stage4_windows>`
     - Group peaks into disjoint analysis windows for the fit.
   * - 5
     - :doc:`Peak fitting <stage5_fitting>`
     - Fit each window with a finite-duration line-shape model and accept lines
       on statistical grounds.
   * - 6
     - :doc:`Review, reports, finalization <stage6_review>`
     - Review the model, calibrate frequencies, and produce the final line list
       and reports.

Stage 2b is a calibration step rather than a transformation of the spectrum; it
runs after noise estimation and feeds both peak detection and the fit. Several
stages have tunable parameters; how those parameters are resolved across
keyword arguments, presets, and the values persisted in the file is described
in :doc:`settings_and_presets`.

Where to go next
----------------

* :doc:`installation` — set up the environment.
* :doc:`quickstart` — run an experiment from raw data to a line list.
* :doc:`file_format` — what the ``.ftmw`` file holds and how stages track their
  dependencies.
