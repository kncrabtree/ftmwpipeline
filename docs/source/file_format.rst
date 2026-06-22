.. index::
   single: .ftmw file
   single: file format
   single: HDF5
   single: provenance
   single: stage tracking
   single: dependencies; pipeline stages
   single: reproducibility

Inside the .ftmw file
=====================

Every experiment is one self-contained, portable file with a ``.ftmw``
extension. The file is an HDF5 container that holds the raw measurement, every
stage's result, the settings each stage used, and the provenance of the source
data. Moving an analysis between machines, archiving it, or handing it to a
collaborator is a matter of copying this one file; reproducing the analysis
requires nothing else.

What the file holds
-------------------

A ``.ftmw`` file accumulates content as the pipeline runs. At any point it
contains some prefix of:

* **The raw FID** — the time-series and its acquisition metadata (sample
  spacing, probe frequency, sideband, shot count, point count, duration),
  stored losslessly, together with any source-format *recommended* processing
  parameters.
* **The canonical FT settings** — the data-selection and display parameters
  that define the spectrum every later stage works on (see
  :ref:`persisted-vs-reconstructed` below).
* **The noise estimate** — the per-bin noise from :doc:`Stage 2 <stage2_noise>`.
* **The decay-time calibrations** — the exponential and Gaussian τ twins and the
  recommended line shape from :doc:`Stage 2b <stage2b_tau>`, and the optional
  scope-timebase self-calibration.
* **The detected peaks, the window plan, and the fitted model** — from
  :doc:`Stages 3 <stage3_peaks>`, :doc:`4 <stage4_windows>`, and
  :doc:`5 <stage5_fitting>`, the last including the per-window fitted-parameter
  covariance.
* **The review record and final products** — the consolidated, calibrated line
  list and the human-review decisions from :doc:`Stage 6 <stage6_review>`.
* **The settings each stage used** and the **source provenance** record.

The exact HDF5 group and attribute names are an implementation detail; the file
is meant to be inspected through the :doc:`cli` (``ftmwpipeline info``) and the
:meth:`~ftmwpipeline.pipeline.Pipeline.info` method rather than by reaching into
fixed paths. Because the container is plain HDF5, it is also readable with any
HDF5 tool (``h5dump``, ``h5py``) for ad hoc inspection.

Stage tracking and dependencies
-------------------------------

The pipeline is a sequence of stages with declared dependencies. The file
records which stages have completed, and each stage checks that its inputs are
present before it runs. The dependency graph is:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Stage
     - Requires
   * - :doc:`Stage 0 — import <stage0_import>`
     - nothing (it creates the file)
   * - :doc:`Stage 1 — FT <stage1_ft>`
     - Stage 0
   * - :doc:`Stage 2 — noise <stage2_noise>`
     - Stage 1
   * - :doc:`Stage 2b — τ calibration <stage2b_tau>`
     - Stages 0, 1, 2
   * - :doc:`Stage 3 — peak detection <stage3_peaks>`
     - Stages 1, 2 (Stage 2b recommended)
   * - :doc:`Stage 4 — window assignment <stage4_windows>`
     - Stage 3
   * - :doc:`Stage 5 — fitting <stage5_fitting>`
     - Stages 0, 4 (Stage 2b recommended)
   * - :doc:`Stage 6 — review <stage6_review>`
     - Stage 5

Running a stage whose required predecessor has not completed fails with an error
naming the missing dependency, rather than producing a result from incomplete
inputs. Where Stage 2b is *recommended* but not required, the consuming stage
runs without it and falls back to a documented default (Stage 3's gap-pass decay
basis, Stage 5's per-window starting τ); supplying the calibration improves the
result without being a precondition. The scope-timebase self-calibration depends
only on the raw FID and feeds the report's frequency uncertainty rather than any
fit.

Re-running a stage with new parameters is safe. Re-running an *earlier* stage
(or changing the canonical FT settings, or re-importing the source) invalidates
the downstream results that depended on it, so the file is never left in a
silently inconsistent state: the invalidated stages must be re-run.

.. _persisted-vs-reconstructed:

What is persisted, and what is recomputed
-----------------------------------------

The file stores the expensive, hard-to-reproduce products and recomputes the
cheap ones on demand. The most visible case is the spectrum itself: **the
canonical FT is not stored**. Only the processing parameters that define it are
persisted, and the frequency-domain spectrum is recomputed from the raw FID plus
those parameters whenever a stage needs it. A completed Stage 1 is therefore
proven by the presence of its persisted parameters, not by a stored spectrum
array.

The canonical FT is unconditionally unapodized, un-windowed, and native-length.
The persisted Stage 1 parameters are the data selection (start time, end time,
and the frequency trim range) plus display scaling; there are no apodization or
zero-padding settings (apodization trades resolution and biases the line shape,
and zero-padding corrupts the noise and χ² statistics that later stages depend
on). Every later stage operates on the single spectrum these parameters define.
How those parameters are resolved across explicit overrides, presets, and the
persisted values is described in :doc:`settings_and_presets`.

The noise estimate, peaks, window plan, fitted parameters (with covariance), and
the finalized line list are all persisted, because they are expensive to
recompute and are the scientific record of the analysis.

Provenance and safe re-import
-----------------------------

When the file is created, it records the provenance of the data it was built
from: the source path, the source modification time, a content hash, the import
timestamp, the format name, and the loader parameters. This record is what makes
re-import safe and reproducible.

Re-importing the *same* source onto an existing file is detected by the recorded
provenance and is non-destructive — the existing analysis is reused rather than
reprocessed, so re-running an import cell in a notebook does not discard
downstream work. Importing a *different* source onto an existing file is refused
unless overwriting is requested explicitly, so a finished analysis is never
silently replaced.

Error conditions
----------------

The file model surfaces problems explicitly rather than guessing:

* Opening a file that does not exist raises ``FileNotFoundError`` with guidance
  on how to create one.
* Opening a file that is present but not a readable pipeline file raises
  ``PipelineCorruptionError``.
* Creating over a file built from a different source raises
  ``PipelineExistsError`` with the options for proceeding.
* Running a stage whose dependency is missing raises ``StageDependencyError``
  naming the unmet stage.

All of these are subclasses of a common ``PipelineFileError``, so a script can
catch the whole family at once.
