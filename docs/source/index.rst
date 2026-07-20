.. ftmwpipeline documentation master file.
   The toctrees below define the sidebar navigation. Captioned
   toctrees become sidebar section headers.

.. toctree::
   :hidden:
   :caption: Getting Started

   overview
   installation
   quickstart
   run

.. toctree::
   :hidden:
   :caption: Concepts

   settings_and_presets
   file_format
   input_formats
   fit_curation

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
   :caption: Methods & Validation

   methods/noise_snr_scaling
   methods/matched_filter_detection
   methods/edge_coherence
   methods/stage5_fitting
   methods/timebase_calibration

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
* :doc:`stage1_ft` — compute the standard frequency-domain spectrum.
* :doc:`stage2_noise` — estimate the per-bin noise.
* :doc:`stage2b_tau` — calibrate the molecular decay time and recommend a line
  shape.
* :doc:`stage3_peaks` — detect peaks.
* :doc:`stage4_windows` — assign disjoint analysis windows.
* :doc:`stage5_fitting` — fit the peaks in each window.
* :doc:`stage6_review` — review, report, and finalize the line list.

Methods and validation notes go deeper on why specific algorithmic choices can
be trusted, with figures and numbers regenerated from the example data:

* :doc:`methods/noise_snr_scaling` — why naive noise estimation fails on
  high signal-to-noise, line-dense spectra, and how the scatter estimator is
  validated.
* :doc:`methods/matched_filter_detection` — why the weak-line gap pass detects
  with an exponentially-apodized transform, derived and validated on synthetic
  ground truth and the example experiment.
* :doc:`methods/edge_coherence` — the phase-coherent edge test behind window
  assignment: its closed-form null, threshold calibration, and the active-FT
  frame it must be scored in.
* :doc:`methods/stage5_fitting` — the peak-fitting acceptance gate: why it is
  window-size invariant, how it handles blends, and the SNR-aware health check
  validated across the example experiments.
* :doc:`methods/timebase_calibration` — measuring the digitizer scale error from
  the clock spurs' phase, its Cramér–Rao precision bound, and how the frequency
  correction reaches the line list.

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
