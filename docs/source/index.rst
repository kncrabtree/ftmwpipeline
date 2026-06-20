.. ftmwpipeline documentation master file.
   The toctrees below define the sidebar navigation. Captioned
   toctrees become sidebar section headers.

.. toctree::
   :hidden:
   :caption: Getting Started

   overview
   installation
   quickstart

.. toctree::
   :hidden:
   :caption: Concepts

   settings_and_presets
   file_format
   input_formats

.. toctree::
   :hidden:
   :caption: Pipeline Stages

   stage0_import
   stage1_ft
   stage2_noise
   stage2b_tau
   stage3_peaks
   stage4_windows
   stage5_fitting
   stage6_review

.. toctree::
   :hidden:
   :caption: Advanced

   clock_declaration
   scope_record_import
   performance

.. toctree::
   :hidden:
   :caption: Reference

   cli
   api/index
   changelog

ftmwpipeline Documentation
==========================

``ftmwpipeline`` processes Fourier transform microwave (FTMW) spectroscopy
data, from a raw free-induction decay to a calibrated table of fitted spectral
lines. Each experiment is one self-contained, portable ``.ftmw`` file that
progresses through a sequence of stages — import, Fourier transform, noise
estimation, decay-time calibration, peak detection, window assignment, peak
fitting, and review — with every stage's result and its provenance recorded in
the file.

The pipeline is built for spectroscopists who need to know not only how to run
an analysis but what each stage does and why its results can be trusted. The
stage pages describe the algorithms, the assumptions behind them, and the
statistical basis for the reported peak parameters and uncertainties.

Where to start
==============

* :doc:`overview` — the purpose and design philosophy, the ``.ftmw`` file
  model, the three user-facing interfaces, and the stage pipeline at a glance.
* :doc:`installation` — install the package and its dependencies.
* :doc:`quickstart` — process an experiment end to end.
* :doc:`settings_and_presets` — how stage parameters are resolved across
  keyword arguments, presets, and the values persisted in the file.
* :doc:`input_formats` — bring data from any instrument into the pipeline: the
  native HDF5 and CSV input formats, the metadata sidecar, and declaring
  instrument clock sources.

The pipeline stages, in the order an experiment moves through them:

* :doc:`stage0_import` — load a raw FID from an instrument format.
* :doc:`stage1_ft` — compute the canonical frequency-domain spectrum.
* :doc:`stage2_noise` — estimate the per-bin noise.
* :doc:`stage2b_tau` — calibrate the molecular decay time and recommend a line
  shape.
* :doc:`stage3_peaks` — detect peaks.
* :doc:`stage4_windows` — assign disjoint analysis windows.
* :doc:`stage5_fitting` — fit the peaks in each window.
* :doc:`stage6_review` — review, report, and finalize the line list.

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
