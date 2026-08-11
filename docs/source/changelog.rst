.. index::
   single: changelog
   single: version history

Changelog
=========

Notable changes to ``ftmwpipeline``, newest first. Versions follow
`semantic versioning <https://semver.org/>`_.

Version 0.1.0b4 (2026-08-10)
----------------------------

A read-access and linewidth beta over 0.1.0b3. As a pre-release it still
installs only when explicitly requested: ``pip install --pre ftmwpipeline``.

Mostly a second way to *read* what a ``.ftmw`` already holds. One change is not
additive, and it moves fitted numbers: the finite-``T`` linewidth is now solved
for rather than measured on a fixed grid, which shifts widths by about 1e-4
relative and so can flip a peak-separation decision at a boundary.
**``ANALYSIS_EPOCH`` is 2**, so splicing a Stage 6 edit into a fit produced
under 0.1.0b3 or earlier is refused until re-fit or acknowledged; reading is
never gated.

* **A read-only tap on persisted data.** The ``load_*`` operations rebuild the
  complete persisted record — audit trails, thaw and rescue histories,
  fixed-contributor records, covariance blocks — which is what a curator
  editing that record needs, and was until now the only way to reach a few
  columns. A consumer keeping two floats per window paid for all of it: the
  cost is per-item HDF5 overhead paid thousands of times, not the values kept.
  :func:`~ftmwpipeline.api.read_table` exposes the same persisted artifacts as
  tables of bulk columns, reading only the columns asked for and reconstructing
  nothing: the Stage 2b calibration (per-band decay times, frequency thirds,
  per-bin contributors, excluded spur clusters — and the same four again for
  the Gaussian twin), the Stage 3 peak list, the Stage 4 window plan with its
  free-peak and fixed-contributor assignments in long form, and the Stage 5
  fitted peaks, per-window scalars, and decision record (the add-loop audit,
  the doublet alternatives, and the thaw, replan and rescue histories).
  Measured on a 262-window file, a three-column fitted-peak read costs about a
  tenth of ``load_fit``'s HDF5 traffic; the Stage 2b and Stage 3 tables are
  there for uniform access and CSV export rather than speed, since those
  loaders were never the bottleneck, and the event-log tables cost a JSON parse
  because that is how the fit persists its narrative. Column names are
  singular, even where the file's own dataset is plural.
  :func:`~ftmwpipeline.api.read_metadata` reaches the top-level scalars from
  group attributes alone — provenance, the FID acquisition, the start-detection
  sweep outcome, the canonical FT window (with the derived
  ``ft.acquisition_us``, so the Fourier resolution element is readable as soon
  as Stage 1 has run rather than only after a fit), the decay-time calibrations
  and their line-shape vote, the per-stage counts, and the timebase scale
  error. The ``read`` CLI object (``read list`` / ``read table`` / ``read
  meta``) dumps the same tables as CSV, TSV, or JSON, to stdout or a file.
  Values keep the persisted sentinels; the calibrated, presentation-ready line
  list remains ``report table``.
* **A fitted peak's ``window_id`` is now always its owning window's.** The
  per-peak ``window_id`` column was written as ``-1`` whenever the in-memory
  ``FittedPeak.window_id`` was ``None``, while the window group it is stored
  under always carries a real id — so the column and the group could disagree,
  and the full loader grouped by the group. A consumer that grouped by the
  column instead would silently drop such a row out of its window: no
  exception, just a line with no fit behind it. The writer now stamps the
  owning window's id, and both readers backfill it over a stored ``-1``, so
  files written by earlier versions group identically. ``window_id`` has no
  absent case and the ``-1`` sentinel is retired from that column.
* **The finite-``T`` linewidth is solved for, not measured on a grid.**
  :func:`~ftmwpipeline.fitting.validation.feature_fwhm` gridded ``|h_T|`` on a
  fixed 200001-point ``linspace(-1, 1)`` in absolute MHz and read off the
  outermost points above half maximum. But the width does not depend on ``T``
  and the grid did: the model has only two length scales, ``tau`` and ``T``,
  and frequency enters solely as ``f*T``, so ``FWHM * T = W(tau/T, shape)``
  exactly. The new
  :func:`~ftmwpipeline.fitting.validation.fwhm_dimensionless` computes that
  ``W`` by bracketing and bisecting the half-maximum crossing, and
  ``feature_fwhm`` is ``W / T``. Three consequences: the width is now accurate
  to the solver tolerance at every ``T`` rather than losing precision as ``T``
  grows; it no longer clips, where the old grid silently returned its own 2 MHz
  span once the true width outran it (below ``T = 0.92 us`` for a Lorentzian at
  ``tau/T = 0.3``); and it is ~40x faster, which a consumer computing a width
  per line feels directly. Widths move by about 1e-4 relative, so fitted output
  is not bit-identical to 0.1.0b3 — hence the epoch bump.
* **``read_metadata`` spells the Stage 1 section both ways.** The CLI has always
  treated ``ft`` and ``stage1`` as interchangeable object names, but this view
  named the canonical data selection ``ft.units_power`` where ``settings show``
  names the same persisted knob ``stage1.units_power``. Every key in that
  section is now emitted under both prefixes, so a consumer need not know which
  surface a name came from.

Version 0.1.0b3 (2026-08-03)
----------------------------

A correctness and provenance beta over 0.1.0b2. As a pre-release it still
installs only when explicitly requested: ``pip install --pre ftmwpipeline``.

Results are unchanged on a file whose fit already ran: the numeric changes
below are either latent-correctness fixes that no line in the reference
fixture reached, or improvements that only engage on shorter records and on
spectra whose clock spurs the old match window missed.

* **Per-stage analysis environment, with an editing gate.** Every persisted
  stage now stamps the environment that produced it — package version,
  analysis epoch, Python, numpy, scipy, h5py, the runtime BLAS vendor, and the
  platform — instead of one version stamp written at import and never
  refreshed. Compatibility is keyed on a hand-maintained ``ANALYSIS_EPOCH``
  bumped only when numerical output changes, not on the package version.
  Reading is never gated; running a new stage forward warns; splicing a Stage 6
  edit into a fit produced under a different epoch is refused, because the
  mixture would live *inside* one artifact where no stamp could describe it.
  ``review acknowledge-environment`` records acceptance in the ``.ftmw``
  itself, so the reports say so permanently rather than only in the session
  where it happened. ``info``, ``validate``, the CSV/JSON provenance header,
  and the HTML report all surface the record and any mixed-environment
  warning. Files written before stamping existed have an unknown epoch, which
  is treated as compatible.
* **``review create``: a window for a line the detector missed.** Reaching a
  missed line previously meant lowering the detection threshold, which re-runs
  Stage 3 and discards Stages 5 and 6 — the entire curated edit set. The new
  verb installs the missing window instead, with deterministic bounds, freshly
  appended ids that never renumber existing windows, and no cascade or thaw.
  It is purely structural: the window is fit with an empty peak set, and
  putting a line in it is a separate ``review edit --add``, so the decision log
  records the two operations distinctly. When the gap is too narrow to hold a
  fittable window, the adjacent window is widened and re-fit instead, reported
  and logged rather than done silently.
* **Out-of-range ``review edit --add`` is now rejected.** An add outside the
  named window's extent used to be seeded and fit against data the window does
  not cover, with the optimizer quietly pinning it at the nearest edge — so an
  off-by-one resolving a click to a window produced a wrong fit instead of an
  error. The post-snap frequency is now checked against the window's range
  (with half a bin of slack), and the message names the range and points at
  ``review create``.
* **Per-peak derivation tag.** ``FittedPeak`` / ``FinalPeak`` carry the
  ``derivation`` index of the decision that created or last altered a peak
  (absent when it came through a refit unchanged), exported in the CSV and JSON
  tables. A consumer binding external state to individual peaks can read it
  instead of pairing peak sets across an edit.
* **Parallel fitting fixed and much faster.** Multi-worker Stage 5 fits aborted
  outright on many-core machines: each forked worker re-scanned loaded native
  libraries to pin BLAS threads, which is not fork-safe under a multithreaded
  parent. BLAS is now pinned once in the parent around the fit — fork-safe,
  scoped to the fit, and restored on return, so importing ``ftmwpipeline`` no
  longer changes a caller's own BLAS thread count. On a 32-core box the
  reference fit drops from 148.6 s to 23.2 s at 30 workers, byte-identical to
  the serial result.
* **Timebase calibration runs earlier (after Stage 1, before noise).** Its
  measured clock scale error ``eps`` is now available to every later stage, and
  its declared dependencies are honest about needing the Stage 1 FT settings.
  The ``eps`` measurement itself is unchanged.
* **Correct, ``eps``-aware spur matching (Stage 5).** The nearest-bin spur
  match now derives its window from the active-FT bin spacing (the former fixed
  0.04 MHz became an absolute floor — it was half a bin only at one particular
  record length), honors each lattice point's own window in lattice mode, and
  shifts the search anchor to the clock-corrected position ``f(1+eps)`` rather
  than merely widening the tolerance, since a displaced tone lands in a
  *different* bin that no widening reaches. Across the seven reference
  fixtures this recovers genuine on-lattice clock spurs the old window missed,
  with reduced-chi-squared holding or improving.
* **More stable timebase noise reference.** The ``eps`` fit's noise floor uses
  32 off-lattice probes instead of 8; the 25th-percentile floor was
  high-variance and could under-estimate the noise, inflating reported SNRs.
* **Stage 5 co-fit statistics installed on the thaw path.** An accepted thaw
  overwrote a window's peaks and decay time but left uncertainties, covariance
  and reduced chi-squared as carry-over from the superseded independent fit —
  stale numbers that feed the per-window log, Stage 6 attention routing, and
  ``fit check`` grading. Per-line uncertainties and covariance now come from
  the joint fit; chi-squared is recomputed per window so it stays a per-window
  diagnostic.
* **Peak-detection edge padding.** The Savitzky–Golay boundary padding
  block-copied samples in forward order instead of holding the edge value,
  seaming a discontinuity into the second derivative near each band edge and
  biasing the concave-down test there. Interior detection is unaffected.
* **Stage 1 FT visualization shows what later stages use.** The FT panels now
  render the zero-padded active-band display spectrum (the same surface the
  Stage 5 report draws), and the second FID panel is the Active FID — the
  DC-removed ``[start_us, end_us]`` slice on its true time axis — replacing the
  full-length, mostly-zero "Preprocessed FID" panel. The plot still follows the
  settings resolved for the call, so the ``ft show --start-us/--end-us/--trim``
  preview workflow is unchanged.
* Documentation: the STFT decay-time calibration derivation is now written into
  :doc:`stage2b_tau`; the fitting methods note is corrected on two points where
  it disagreed with the shipped code (residual rescue seeds from the calibrated
  decay time, and a frozen contributor's leakage skirt is drawn at the window's
  shared decay time). Per-experiment provenance notes and the vinyl-cyanide
  reference catalog moved into ``examples/blackchirp_data/``, and
  development-only planning docs and tuning scripts were retired from the
  distribution.

Version 0.1.0b2 (2026-07-20)
----------------------------

A refinement beta over 0.1.0b1. As a pre-release it still installs only when
explicitly requested: ``pip install --pre ftmwpipeline``.

* **Zero-padded display spectrum (Stage 1).** ``compute_display_ft`` renders a
  smoothly interpolated view of the standard spectrum for display, trimmed to
  the same analysis band as the standard transform so the two stay aligned. The
  standard, native-length spectrum that every downstream stage's settings
  derive from is unchanged.
* **Per-stage knobs on ``run``.** The single-command ``run`` pipeline exposes
  each stage's tuning knobs as namespaced command-line flags, so a full run can
  be steered from the command line without a settings file, with accompanying
  ``run`` documentation.
* **Richer provenance reporting.** The Stage 0 import report is reorganized
  around chirp-end and start-time provenance, and the frequency-calibration
  report gains a reproducibility grid.
* Documentation: a Read the Docs badge and link, plus minor edits.

Version 0.1.0b1 (2026-06-28)
----------------------------

The first public beta. As a pre-release it installs only when explicitly
requested: ``pip install --pre ftmwpipeline``.

It provides the full free-induction-decay to fitted-line-list pipeline behind
three interchangeable interfaces over one shared implementation: the
command-line tool (:doc:`cli`), the
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
* **Fourier transform (Stage 1).** The standard spectrum is unconditionally
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
  recovery — classifies peaks by SNR on the active FT. SNR is the
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
