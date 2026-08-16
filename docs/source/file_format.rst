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
* **The persisted FT settings** — the data-selection and display parameters
  that define the active region and analysis band every later stage rebuilds
  its working spectrum from (see :ref:`persisted-vs-reconstructed` below).
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
(or changing the persisted FT settings, or re-importing the source) invalidates
the downstream results that depended on it, so the file is never left in a
silently inconsistent state: the invalidated stages must be re-run.

.. _persisted-vs-reconstructed:

What is persisted, and what is recomputed
-----------------------------------------

The file stores the expensive, hard-to-reproduce products and recomputes the
cheap ones on demand. The most visible case is the spectrum itself: **the
FT is not stored**. Only the processing parameters that define it are
persisted, and the frequency-domain spectrum is recomputed from the raw FID plus
those parameters whenever a stage needs it. A completed Stage 1 is therefore
proven by the presence of its persisted parameters, not by a stored spectrum
array.

The FT is unconditionally unapodized, un-windowed, and native-length.
The persisted Stage 1 parameters are the data selection (start time, end time,
and the frequency trim range) plus display scaling; there are no apodization or
zero-padding settings (apodization trades resolution and biases the line shape,
and zero-padding corrupts the noise and χ² statistics that later stages depend
on). Every later stage rebuilds its own working spectrum (the active FT) from
these same parameters, rather than consuming a stored one.
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

.. _ftmw-analysis-environment:

The analysis environment
------------------------

Reproducibility needs more than the data and the settings: it needs to know what
*ran*. Each stage, as it completes, records the environment that produced it —
the ``ftmwpipeline`` version, the analysis epoch (below), the Python, numpy,
scipy and h5py versions, the runtime BLAS vendor and thread count, and the
platform.

The record is **per stage, not per file**, because the pipeline is sequential
and stateful: stages are run at different times, often across an upgrade, and
the situation worth detecting is a file whose Stage 5 fit and Stage 6 curation
came from different code. A single file-level stamp cannot express that. Stage 1
is deliberately absent — it persists no artifact, since the FT is recomputed on
demand from the persisted settings, so it is always the current environment.

``ftmwpipeline info`` prints the record, and says ``MIXED`` with a per-stage
breakdown when the stages disagree. The exported line table and the HTML report
carry it too, so a result is never read without the environment that produced
it. A mixed file is *not* invalid — upgrading mid-analysis is normal and often
harmless — but it is not reproducible from any single version, and the record
says so rather than implying otherwise.

``info`` also compares the record against the environment *running it*, and
reports that separately (``current_environment`` and
``runtime_environment_drift`` in the dictionary form). The two questions are
independent: a file every stage of which was stamped by one release has no
internal drift at all, and may still have been written by a different version
than the one about to edit it. Only the cross-stage comparison can be made from
the file alone, so only it is stored; the comparison against the running
interpreter is made fresh on every call.

A Stage 6 edit deliberately does **not** re-stamp the stage it edits.
``stage5_fitting``'s record names the environment that produced the automatic
fit — the artifact every edit splices into and gates against — so re-stamping
per edit would replace that reference with the editor's own environment and the
gate would compare each edit against the previous edit instead of against the
fit. The consequence worth knowing: a file that predates stamping stays
unstamped through any amount of curation, and its first stamp requires
re-running the producing stage (``fit run``).

The analysis epoch
~~~~~~~~~~~~~~~~~~

One integer, ``ANALYSIS_EPOCH``, is the only machine-checkable compatibility
signal. It is bumped by hand, and only when a change alters the *numerical
output* of a stage — not for refactors, new commands, or documentation. This is
deliberate: at ``0.x`` the package's minor version bumps for ordinary feature
work, so keying compatibility on the version string would fire constantly on
changes that alter nothing. The epoch mirrors what the container's format
version already does — a version whose meaning this project controls.

The interpreter and library versions are recorded but **advisory**: they can
shift the last bits of a fit, they are rarely under the user's independent
control, and a conclusion that depends on them is already fragile. Only the
epoch ever blocks anything.

What a version difference permits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The policy distinguishes three things by what they can silently corrupt:

* **Reading** — inspecting, loading, and reporting — is *never* gated. An
  archived file always opens.
* **Extending forward**, i.e. running a stage that has not run yet, is allowed
  and *warns*. Its inputs are finished artifacts and the new stage is
  self-consistently produced by the current environment, so the result is
  honest as long as the file says the stages differ.
* **Splicing into an existing artifact** — the Stage 6 edit verbs — is
  **refused** across an epoch change. ``review edit`` re-fits one window and
  writes it back into a fit whose other windows came from the earlier code, and
  the cascade then re-fits its dependents; the result would hold two different
  fitting models inside one product. That mixture is *within* the artifact, so
  no stamp on the file can express it and no report can caveat it honestly.

The usual fix is to re-run ``fit run`` so the whole fit comes from one
environment. If the mixture is acceptable, record that decision with
``ftmwpipeline review acknowledge-environment``. The acknowledgement is stored
in the file rather than passed as a command-line flag — on the same principle as
the frequency-accuracy floor, that a fact which changes how a result should be
read must be reproducible from the record alone. It names the environment it was
given under, so a later upgrade asks again instead of inheriting a stale
acceptance, and the reports state that the curation crossed an epoch boundary.

A file written before environment stamping existed carries no record. Its epoch
is *unknown*, which is treated as compatible, never as incompatible: refusing to
work on an existing file because it predates the stamp would punish users for an
upgrade they did not choose. That leniency has one cost, and the pipeline names
it out loud: re-running a stage over an unstamped result logs that
reproducibility against the original run cannot be verified, because nothing in
the file says which version produced the numbers being replaced. The warning
fires once, on the re-run itself; ``info`` shows the same situation as an
unrecorded environment alongside the version running now.

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
