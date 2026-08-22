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
     - :func:`~ftmwpipeline.api.load_timebase_calibration`,
       :func:`~ftmwpipeline.api.frequency_calibration`
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

Read-only access to persisted data
----------------------------------

The ``load_*`` operations above rebuild the complete persisted record — audit
trails, thaw and rescue histories, fixed-contributor records, covariance blocks
— because a curator editing that record needs all of it. A consumer that only
wants a few columns should not pay for it: the cost is per-item HDF5 overhead
paid thousands of times, not the handful of values kept.

:func:`~ftmwpipeline.api.read_table` is the narrow counterpart. It exposes the
persisted stage artifacts as tables of bulk columns — each requested column is
one whole-dataset read, nothing is recomputed, and the file is opened read-only.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Table
     - Contents
   * - ``tau_bands``
     - Stage 2b per-band decay times, with uncertainties and frequency ranges —
       the per-window anchors Stage 5 may consume
   * - ``tau_thirds``
     - The low/mid/high split with a median decay time per third: the
       does-tau-drift-with-frequency diagnostic
   * - ``tau_contributors``
     - One row per STFT bin that survived the gates and voted on the decay time
   * - ``tau_spurs``
     - One row per excluded spur cluster (the ragged member-bin lists stay with
       the full loader)
   * - ``tau_g_*``
     - The same four for the Gaussian twin calibration, which lives in its own
       group and may coexist with the primary one
   * - ``peaks``
     - Stage 3 detected peaks, including the derived ``promoted`` flag
   * - ``windows``
     - Stage 4 planned-window bounds, batch, and contributor counts
   * - ``window_free_peaks`` / ``window_contributors``
     - The plan's ragged per-window sets in long form: which Stage 3 peaks a
       window fits freely, and which it holds fixed from a neighbor
   * - ``fit_peaks``
     - Stage 5 fitted peaks, ordered by molecular frequency, carrying the
       owning window's line ``shape`` so no join is needed
   * - ``fit_windows``
     - Stage 5 per-window fit scalars (bounds, tau, cost, quality)
   * - ``fit_audit`` / ``fit_doublets``
     - The per-window decision record: every candidate the conservative add loop
       tried and how it ruled, and every doublet alternative it weighed
   * - ``fit_thaw`` / ``fit_replans`` / ``fit_rescues``
     - The plan-level histories: contributors released back to free, window
       boundaries redrawn mid-fit, and residual re-searches

Two caveats on "cheap". First, not every table is here because its loader was
slow: Stages 4 and 5 fan out over hundreds of window groups, so the narrow read
is a large win there, while Stages 2b and 3 persist a single group and already
load in milliseconds. Those entries exist so that every persisted artifact is
reachable through one surface rather than only the ones that happened to be
expensive. Second, the event-log tables (``fit_audit``, ``fit_doublets``,
``fit_thaw``, ``fit_replans``, ``fit_rescues``) read JSON-encoded records,
because that is how the fit persists its narrative; reaching them means parsing,
and there is no narrower path. They still cost far less than a full
:func:`~ftmwpipeline.api.load_fit`, but they are not the whole-dataset reads the
rest of this surface is.

Stages 1 and 2 have no table at all: the canonical FT is recomputed from the FID
rather than persisted, and the Stage 2 noise model is a reconstruction over the
persisted bins, not a column to read off.

Column names are singular — a header names one row's field, not the stored
array — even where the file's own dataset is plural (``tau_us`` for
``taus_us``, and so on).

:func:`~ftmwpipeline.api.read_tables` lists the tables a file carries and their
columns. :func:`~ftmwpipeline.api.read_metadata` returns the cheap top-level
scalars as dotted keys — ``file.`` / ``source.`` provenance, the ``fid.``
acquisition, the ``start.`` detection sweep outcome, the canonical ``ft.``
window, the ``tau.`` / ``tau_g.`` calibrations and their line-shape vote, the
``stage3.`` / ``stage4.`` / ``stage5.`` counts, and the ``timebase.`` scale
error. A section is absent when its stage has not been run, so read with
``.get()``.

``ft.acquisition_us`` is the active record length every later stage's Fourier
resolution element ``1 / T`` follows from. It is fixed at Stage 1, so it is
readable long before a fit exists; ``stage5.acquisition_us`` is what the fit
actually recorded.

Every ``ft.`` key is also emitted as ``stage1.``, the two being the same section
under the two interchangeable names the CLI already gives that stage — which is
also the spelling ``settings show`` uses for these same persisted knobs
(``stage1.units_power``). Bind whichever reads better; they are the same value.

The two agree whenever the fit ran on the canonical Stage 1 window, which is the
usual case — but editing the processing settings between runs separates them,
and only the Stage 5 value is the window the fit measured its decay times over.
Pair a fitted ``tau`` against ``stage5.acquisition_us``; reach for
``ft.acquisition_us`` when you want the resolution element before a fit exists.
Note that ``1 / T`` is the resolution element, not a line width: the FWHM of the
finite-``T`` line shape is :func:`~ftmwpipeline.fitting.validation.feature_fwhm`,
which is a factor of order two wider at typical decay times.

Values keep the persisted sentinels rather than the ``None`` the full loaders
substitute: NaN for an absent float, ``-1`` for an absent id, and tri-state
small integers for the knockout and tau flags. Use the full loaders when you
need the reconstructed objects; use these when you need columns.

``fit_peaks.window_id`` is the exception, and is safe to group by: a peak is
stored inside a window group, so its owning id is always known and is filled in
over any ``-1`` an older file wrote. It is never absent, and always the window
the full loader puts the peak in.

.. code-block:: python

    import ftmwpipeline.api as ftmw

    lines = ftmw.read_table(
        "exp_2638.ftmw", "fit_peaks",
        columns=["frequency_mhz", "decay_rate", "shape"],
    )
    resolution_mhz = 1.0 / ftmw.read_metadata("exp_2638.ftmw")["ft.acquisition_us"]

The CLI equivalent is the ``read`` object (:doc:`../cli`), which dumps the same
tables as CSV, TSV, or JSON.

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

.. autoclass:: ftmwpipeline.core.data_structures.FTMWData
   :members:

.. autoclass:: ftmwpipeline.core.data_structures.FID
   :members:

.. autoclass:: ftmwpipeline.core.data_structures.FIDProcessingParameters
   :members:

.. autoclass:: ftmwpipeline.core.data_structures.ComplexFT
   :members:

.. autoclass:: ftmwpipeline.core.data_structures.Sideband
   :members:

.. autoclass:: ftmwpipeline.preprocessing.noise_estimation.NoiseResult
   :members:

.. autoclass:: ftmwpipeline.core.data_structures.PeakClassification
   :members:

.. autoclass:: ftmwpipeline.core.data_structures.Peak
   :members:

.. autoclass:: ftmwpipeline.core.data_structures.FittedPeak
   :members:

.. autoclass:: ftmwpipeline.core.data_structures.SpectralWindow
   :members:

.. autoclass:: ftmwpipeline.core.data_structures.FixedContributor
   :members:

.. autoclass:: ftmwpipeline.core.data_structures.FitWindow
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

The next two are different in kind from those above: they are not the record
of a calibration *run*, but the answer to "what frame is this file's frequency
data in right now, and by how much is it corrected".
:func:`~ftmwpipeline.api.frequency_calibration` /
:meth:`~ftmwpipeline.pipeline.Pipeline.frequency_calibration` /
``timebase state`` (:doc:`../clock_declaration`) *derive* that answer on every
call from the clock declaration and the live timebase calibration, so it is
readable at any stage — including before Stage 5 — and, being derived rather
than persisted, it cannot go stale.

.. autoclass:: ftmwpipeline.core.calibration.CalibrationStamp
   :members:

.. autodata:: ftmwpipeline.core.calibration.CalibrationState
   :annotation:

Curation operations
-------------------

The Stage 6 editing surface. Each is a module-level function taking a path and
a matching :class:`~ftmwpipeline.pipeline.Pipeline` method; see
:doc:`../fit_curation` for the narrative.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Operation
     - What it does
   * - :func:`~ftmwpipeline.api.review_edit`
     - Re-fit one window with ``add`` / ``remove`` edits. A ``remove`` token
       may be a frequency or a ``"uid:N"`` identifier.
   * - :func:`~ftmwpipeline.api.review_accept`
     - Accept a window as reviewed, or revive a named ledger candidate.
   * - :func:`~ftmwpipeline.api.review_create`
     - Install a fit window for a line no window covers.
   * - :func:`~ftmwpipeline.api.review_apply`
     - Apply a curation file as one batch (``dry_run=True`` resolves the plan
       without fitting or writing).
   * - :func:`~ftmwpipeline.api.review_preview`
     - Run a curation file's plan to completion in memory and report the
       fitted outcome, writing nothing.
   * - :func:`~ftmwpipeline.api.review_undo`
     - Roll recorded decisions back by id, replaying the survivors from the
       automatic baseline.
   * - :func:`~ftmwpipeline.api.review_log`
     - Read the persisted decision log.
   * - :func:`~ftmwpipeline.api.refit_snap_tol_mhz`
     - The MHz snap tolerance this file's curation verbs resolve to. Read it
       rather than deriving it — see
       :data:`~ftmwpipeline.core.curation.REFIT_SNAP_TOL_BINS`.

For a worklist that issues many verbs against one file,
:meth:`Pipeline.review_session <ftmwpipeline.pipeline.Pipeline.review_session>`
builds the shared fit context once and reuses it across every verb, and lets a
preview's outcome be persisted directly by an immediately following apply. The
:class:`~ftmwpipeline.ReviewSession` it yields offers the same six verbs with
the same signatures, and is covered in :ref:`curation-sessions`.

Review sessions
---------------

The object :meth:`Pipeline.review_session
<ftmwpipeline.pipeline.Pipeline.review_session>` yields. It is obtained from
that method, never constructed directly, but the class is exported at the
package top level so a caller can annotate it::

    from ftmwpipeline import Pipeline, ReviewSession

    def drain(session: ReviewSession, worklist) -> None:
        for wid, target in worklist:
            session.review_edit(wid, remove=[target], frame="raw")

.. autoclass:: ftmwpipeline.ReviewSession
   :members:
   :member-order: bysource

Curation vocabulary
--------------------

Public vocabulary shared by every Stage 6 curation verb (``review edit`` /
``create`` / ``accept``, and ``review apply``), defined in
:mod:`ftmwpipeline.core.curation` so an external tool can pair against the
exact values the pipeline itself uses rather than a separately maintained
copy. ``merge`` and ``split`` are not verbs; they are read from what an
``edit``'s ``add``/``remove`` does to a window's peak set (see
:doc:`../stage6_review`).

.. autodata:: ftmwpipeline.core.curation.Frame
   :annotation:

.. autodata:: ftmwpipeline.core.curation.REFIT_SNAP_TOL_BINS
   :annotation:

.. autofunction:: ftmwpipeline.core.curation.parse_peak_token

.. autoclass:: ftmwpipeline.core.curation.PeakUidToken
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

.. autoexception:: ftmwpipeline.file_manager.PipelineCompatibilityError

.. autoexception:: ftmwpipeline.file_manager.AnalysisEpochMismatchError
