.. index::
   single: Python API
   single: API reference
   single: Pipeline class
   single: functional API

Python API Reference
====================

``ftmwpipeline`` exposes two Python interfaces over one shared implementation:

* the :class:`~ftmwpipeline.pipeline.Pipeline` class, bound to a single
  ``.ftmw`` file for its lifetime, and
* a stateless **functional API** (``import ftmwpipeline.api as ftmw``) whose
  every function takes a ``.ftmw`` path as its first argument.

The two are thin wrappers around the same internal stage code, so they produce
identical results for identical inputs; the functional API delegates to the
``Pipeline`` class internally. Use the class when you hold one experiment open
and step it through the stages interactively; use the functional API for batch
and scripting work where each call stands alone. Both surfaces persist their
results into the file, so the choice of interface never changes what is stored.
The command-line interface (:doc:`../cli`) is the third wrapper over the same
core.

Stage parameters resolve through the layered chain described in
:doc:`../settings_and_presets` (explicit argument > value persisted in the file
> preset > recommended default). Passing an explicit override to a stage method
persists it as the file's new state and invalidates the downstream
stages that depended on the old value.

.. contents:: On this page
   :local:
   :depth: 1

Operations by stage
-------------------

Each pipeline stage offers a matching pair of operations on both surfaces — a
``Pipeline`` method and a module-level function of the same name. The compute
operation runs the stage and persists its result; the ``load_*`` operation
reads a previously persisted result back; the ``visualize_*`` operation renders
a diagnostic figure.

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - Stage
     - Compute (persist)
     - Read back / visualize
   * - Import (Stage 0)
     - :func:`~ftmwpipeline.api.import_data`
     - :func:`~ftmwpipeline.api.load_fid`
   * - Start detection
     - :func:`~ftmwpipeline.api.detect_start_time`
     - :func:`~ftmwpipeline.api.visualize_start_detection`
   * - Fourier transform (Stage 1)
     - :func:`~ftmwpipeline.api.compute_ft`
     - :func:`~ftmwpipeline.api.visualize_ft`
   * - Noise (Stage 2)
     - :func:`~ftmwpipeline.api.estimate_noise`
     - :func:`~ftmwpipeline.api.visualize_noise`
   * - Decay time (Stage 2b)
     - :func:`~ftmwpipeline.api.calibrate_tau`
     - :func:`~ftmwpipeline.api.load_tau_calibration`, :func:`~ftmwpipeline.api.recommend_shape`
   * - Timebase calibration
     - :func:`~ftmwpipeline.api.calibrate_timebase`
     - :func:`~ftmwpipeline.api.load_timebase_calibration`
   * - Peaks (Stage 3)
     - :func:`~ftmwpipeline.api.detect_peaks`
     - :func:`~ftmwpipeline.api.load_peaks`, :func:`~ftmwpipeline.api.visualize_peaks`
   * - Windows (Stage 4)
     - :func:`~ftmwpipeline.api.assign_windows`
     - :func:`~ftmwpipeline.api.load_windows`, :func:`~ftmwpipeline.api.visualize_windows`
   * - Fit (Stage 5)
     - :func:`~ftmwpipeline.api.fit_peaks`
     - :func:`~ftmwpipeline.api.load_fit`, :func:`~ftmwpipeline.api.visualize_fit`, :func:`~ftmwpipeline.api.show_fit`
   * - Review (Stage 6)
     - :func:`~ftmwpipeline.api.review_run`
     - :func:`~ftmwpipeline.api.get_review_status`, :func:`~ftmwpipeline.api.report_run`, :func:`~ftmwpipeline.api.report_diff`

To drive a raw source through every stage in one call, use
:func:`~ftmwpipeline.api.run_pipeline` (functional) or
:meth:`Pipeline.build <ftmwpipeline.pipeline.Pipeline.build>` (class).

Pipeline class
--------------

.. autoclass:: ftmwpipeline.pipeline.Pipeline
   :members:
   :member-order: bysource

Functional API
--------------

.. automodule:: ftmwpipeline.api
   :members:
   :member-order: bysource

Data structures
---------------

The domain types returned and consumed by the stage operations. They are
defined in :mod:`ftmwpipeline.core.data_structures` unless noted.

.. autoclass:: ftmwpipeline.core.data_structures.FID
   :members:

.. autoclass:: ftmwpipeline.core.data_structures.ComplexFT
   :members:

.. autoclass:: ftmwpipeline.core.data_structures.Sideband
   :members:

.. autoclass:: ftmwpipeline.preprocessing.noise_estimation.NoiseResult
   :members:

.. autoclass:: ftmwpipeline.core.data_structures.Peak
   :members:

.. autoclass:: ftmwpipeline.core.data_structures.FittedPeak
   :members:

.. autoclass:: ftmwpipeline.core.data_structures.SpectralWindow
   :members:

.. autoclass:: ftmwpipeline.core.data_structures.WindowPlan
   :members:

.. autoclass:: ftmwpipeline.core.data_structures.FittingResult
   :members:

.. autoclass:: ftmwpipeline.core.data_structures.SpectrumFit
   :members:

Final products and review state
-------------------------------

The consolidated, calibrated outputs of the review stage.

.. autoclass:: ftmwpipeline.core.data_structures.FinalProducts
   :members:

.. autoclass:: ftmwpipeline.core.data_structures.FinalPeak
   :members:

.. autoclass:: ftmwpipeline.core.data_structures.FrequencyCalibration
   :members:

.. autoclass:: ftmwpipeline.core.data_structures.Stage6Review
   :members:

.. autoclass:: ftmwpipeline.core.data_structures.LedgerCandidate
   :members:

Calibration results
--------------------

Result objects returned by the start-detection, decay-time, and timebase
calibrations.

.. autoclass:: ftmwpipeline.preprocessing.start_detection.StartDetectionResult
   :members:

.. autoclass:: ftmwpipeline.fitting.tau_calibration.TauCalibrationResult
   :members:

.. autoclass:: ftmwpipeline.fitting.tau_calibration.ShapeRecommendation
   :members:

.. autoclass:: ftmwpipeline.fitting.timebase_calibration.TimebaseCalibrationResult
   :members:

Settings objects
----------------

Each stage takes an optional settings dataclass bundling its knobs; fields left
``None`` fall through the resolution chain. See :doc:`../settings_and_presets`
for how these compose with presets and the persisted layer.

.. autoclass:: ftmwpipeline.core.start_detection_settings.StartDetectionSettings
   :members:

.. autoclass:: ftmwpipeline.core.noise_settings.NoiseSettings
   :members:

.. autoclass:: ftmwpipeline.core.tau_calibration_settings.TauCalibrationSettings
   :members:

.. autoclass:: ftmwpipeline.core.peak_detection_settings.PeakDetectionSettings
   :members:

.. autoclass:: ftmwpipeline.core.window_planning_settings.WindowPlanningSettings
   :members:

.. autoclass:: ftmwpipeline.core.stage_fit_settings.StageFitSettings
   :members:

.. autoclass:: ftmwpipeline.core.stage_fit_settings.ClockSource
   :members:

Exceptions
----------

The file-management error family, all subclasses of
:class:`~ftmwpipeline.file_manager.PipelineFileError`.

.. autoexception:: ftmwpipeline.file_manager.PipelineFileError

.. autoexception:: ftmwpipeline.file_manager.PipelineExistsError

.. autoexception:: ftmwpipeline.file_manager.PipelineCorruptionError

.. autoexception:: ftmwpipeline.file_manager.StageDependencyError
