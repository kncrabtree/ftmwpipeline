.. index::
   single: changelog
   single: version history

Changelog
=========

Notable changes to ``ftmwpipeline``, newest first. Versions follow
`semantic versioning <https://semver.org/>`_.

Version 0.1.0 (unreleased)
--------------------------

The first development release. It provides the full free-induction-decay to
fitted-line-list pipeline behind three interchangeable interfaces over one
shared implementation: the command-line tool (:doc:`cli`), the
:class:`~ftmwpipeline.pipeline.Pipeline` class, and the stateless functional API
(:doc:`api/index`).

Pipeline stages
~~~~~~~~~~~~~~~

* **Import (Stage 0).** Read a raw experiment into a self-contained ``.ftmw``
  HDF5 file with full provenance and safe, idempotent re-import. Pluggable input
  loaders for Blackchirp, the native self-describing ``ftmw-hdf5`` format, CSV
  columns, and segmented Keysight ``.mat`` scope records, with a metadata
  sidecar for no-code generic input. Start-time detection stamps a recommended
  FID window from the chirp-end collapse.
* **Fourier transform (Stage 1).** The canonical spectrum is unconditionally
  unapodized and native-length; the user controls only data selection (the
  active FID window and the analysis band) and display scaling. The persisted
  trim binds every downstream stage.
* **Noise estimation (Stage 2).** A region-aware, Rician-corrected scatter-MAD
  estimator yields the per-bin complex-RMS σ that every later stage scores
  against, immune to the leakage-pedestal inflation that defeats level
  estimators on high-SNR, line-dense spectra.
* **Decay-time calibration (Stage 2b).** A fit-free sliding-window STFT recovers
  the molecular decay constant and its spread, with independent Lorentzian and
  Gaussian variants and a three-way line-shape vote that selects the shape the
  fitter uses.
* **Peak detection (Stage 3).** A two-pass detector — a robust primary pass for
  strong-line positions plus a shape-aware matched-filter gap pass for weak-line
  recovery — classifies peaks by SNR on the canonical spectrum. SNR is the
  magnitude's excess over the local coherent-leakage pedestal, so a dense leakage
  pedestal cannot float pedestal noise above the promotion cutoff and flood the
  later stages.
* **Window assignment (Stage 4).** Promoted peaks are grouped into disjoint fit
  windows with frozen-leakage contributors and a fit dependency order, bounded
  by a phase-coherent edge test.
* **Peak fitting (Stage 5).** A conservative add-one-peak loop fits each window
  with a finite-acquisition line-shape model (leakage is fit, not apodized),
  shared per-window decay, an evidence-triggered leakage baseline, residual
  rescue, a final add-from-convergence pass that recovers close companion lines
  a mid-fit seed collapsed, spur masking, and a four-cut survival pass
  (SNR-floor prune, degenerate-pair collapse, bright-neighbor lineshape-sidelobe
  prune, and a chi-squared-gated degenerate merge trial), with the cross-window
  fit parallelized. The sidelobe prune removes a bright line's lineshape
  artifacts from the line list even at the cost of a higher reduced chi-squared —
  the residual lineshape error is reported honestly through the shape-error
  fraction rather than absorbed by a spurious line.
* **Review and reporting (Stage 6).** A read/edit curation surface with
  attention routing, a candidate ledger, an anchored decision log with undo, and
  catalog cross-referencing; calibrated final products with a three-term
  frequency-uncertainty budget; Level-1 (table), Level-2 (Markdown), and
  Level-3 (HTML) reports; and a post-curation diff report (``report diff``)
  showing every materially-changed window before and after, side by side.

Instrument calibration
~~~~~~~~~~~~~~~~~~~~~~~

* Clock-source declaration and a spur-lattice prior for the fitting-stage spur
  gate (:doc:`clock_declaration`).
* Digitizer-clock timebase self-calibration recovering the fractional scale
  error from the Rb-locked spur lattice.

Interfaces and tooling
~~~~~~~~~~~~~~~~~~~~~~~

* An object-verb command-line interface, a file-bound ``Pipeline`` class, and a
  stateless functional API, all sharing one implementation and producing
  identical results.
* A layered settings model (explicit override > persisted > preset > recommended
  > default) with ``settings`` inspection/persistence, portable presets, and a
  ``scan`` knob-sweep surface (:doc:`settings_and_presets`).
* User-controllable parallelism for the fitting and reporting stages via
  ``--jobs`` / ``FTMW_MAX_WORKERS`` (:doc:`performance`).
* ``matplotlib`` as the single visualization backend, with diagnostic figures
  for every stage.
