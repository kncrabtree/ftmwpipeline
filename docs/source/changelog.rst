.. index::
   single: changelog
   single: version history

Changelog
=========

Notable changes to ``ftmwpipeline``, newest first. Versions follow
`semantic versioning <https://semver.org/>`_.

Unreleased
----------

Accumulating toward ``1.0.0``. ``0.1.0b4`` is the last published release;
``0.1.0b5`` and ``0.1.0b6`` are development versions that were never cut, so
everything below is reachable only from a source checkout. No further beta is
planned — these entries fold into the ``1.0.0`` section when it is dated.

**``ANALYSIS_EPOCH`` moves 2 → 3.** Every tolerance that expresses a spectral
distance is now defined in active-FT bins rather than in MHz (see the first
entry below), which moves fitted output on every existing file. A file fitted
under 0.1.0b4 must therefore be re-fit, or have the mismatch accepted with
``review acknowledge-environment``, before Stage 6 will splice an edit into it.
That refusal is the epoch gate working as intended, not a regression.

Everything else here is unchanged numerically: the Stage 6 rework is
bit-identical on a real fixture, and the remaining changes widen a read surface,
add a name, or move a message.

Much of what is new is *contract*: two values a downstream consumer previously
had to reach into ``_internal`` or parse out of prose to obtain are now
published, and the Stage 6 edit paths that carry them were collapsed onto one
engine so they cannot answer differently.

* **``review preview`` / ``review apply --dry-run`` / ``review edit`` now
  report the structural consequence of a create, implied or explicit, on the
  results they already return -- the acceptance condition an implied create
  (immediately below) shipped under: a typo'd frequency shows up as a stray
  window in the preview rather than erroring, so the preview is what replaces
  the error as the typo guard.** ``PreviewWindowResult`` gains
  ``created_window_mode`` / ``created_window_freq_range`` /
  ``created_window_n_points`` / ``created_window_n_contributors`` /
  ``created_window_depends_on``, additive and ``None`` on every window the
  batch did not create or widen -- keyed by the same in-memory-minted window
  id the preview already reports under, so no re-derivation is needed to
  render "this add creates a window at A-B MHz". ``RefitWindowResult`` gains
  the same five fields, so a caller of ``review_edit`` learns a window was
  built (or widened), and where, from the result it already holds rather than
  only from the preview. ``review preview`` and ``review apply --dry-run``
  print the extent, grid points, and frozen-contributor count for any window
  a plan created or widened, and so does a live ``review apply`` for the
  window it just installed -- which was previously silent about it.
  ``CurationApplyResult`` gains ``created_windows``, one entry per window the
  plan installs or grows (mode, extent, grid points, contributors,
  dependencies, and the anchor it was resolved for), filled on a dry run and
  on a live apply alike, so all three interfaces carry the facts the CLI
  renders rather than only the CLI. A dry run resolves them from the window
  planner directly -- the extent comes from the planner rather than from plan
  resolution, and asking it costs the proposal alone, with no fit and no
  second in-memory pass -- and only when the plan actually contains a create,
  so an ordinary dry run never opens the engine at all. Because that proposal
  is the apply's own, a dry run now *refuses* what the apply would refuse (an
  anchor outside the analysis band, a create whose window cannot be placed),
  with the same per-action attribution: a dry run that returns is a pre-flight
  rather than a plan echo. ``review edit``
  prints the same when the call it services implied a create. The
  preview-then-apply staged-reuse guarantee on a :class:`ReviewSession`
  already covered a plan containing an implied create -- the resolved plan
  the comparison keys on already carried it -- verified rather than assumed,
  with a test.
* **An ``add`` whose frequency no live window covers now implies a create,
  instead of erroring.** Building directly on the window-derivation above: an
  ``add`` -- through ``review edit``, its ``api``/``Pipeline``/CLI forms, or a
  curation file's ``add`` row -- that resolves to no live window now mints the
  window it needs (or widens an adjacent one, when the gap is too narrow to
  hold a new one) and applies the add into it, in one call. This only fires
  when the window was OMITTED; a *named* window that does not cover the
  frequency is still an error, exactly as before. ``remove`` is unchanged and
  permanent: a remove never implies a create. Recorded as **ONE** decision-log
  entry (the ``add`` the caller actually asked for), not a separate
  ``create_window`` entry plus an ``add`` -- the create is a mechanism the
  engine chose, not a user decision, and the structural consequence (``mode``,
  the installed range, its contributors and dependencies) rides on the add's
  own evidence instead. Undoing that one entry therefore removes the peak
  *and* the window it implied together, leaving no stray empty window behind
  -- the reason this shape was chosen over a two-entry alternative. A later,
  unrelated decision that also lands in the implied window (e.g. a second
  ``add`` the coverage check now routes into it) is protected by the same
  orphan guard ``review undo`` already enforces for an explicit
  ``create_window``: undoing the implying entry without the dependent one is
  refused, naming both. A resolved plan shows the pending create explicitly
  (``ReviewPreviewResult.plan`` / ``review apply --dry-run``) via a late-bound
  placeholder id the execution engine resolves to the create's real minted id
  before anything is recorded. An implied create that resolves to
  ``mode="widened"`` cascades to its dependents exactly as an explicit one now
  does (the entry above); an anchor outside the analysis band still refuses.
  One case is refused rather than recorded: if the implying ``add`` is
  reinterpreted against the window it landed in as an inferred merge or split,
  that applier records its own decision, which carries no record of the create
  -- and replaying it would add into the window at its pre-widening width.
  Since an implied create must always be reconstructible from its one log
  entry, that combination raises, naming the window and pointing at the
  explicit ``review create`` plus ``review edit`` spelling. Reaching it needs
  the anchor within snap tolerance of an existing peak in the very window a
  too-narrow gap just widened; the default window margin is ~51x the snap
  tolerance, so a peak sits far inside its own window's edge, and the real
  2638 fit contains no such configuration (the nearest is 40x the tolerance
  away).

* **A curation edit's window is now derived, not required.** ``window_id`` on
  the ``add``/``remove`` targets of ``review edit`` -- ``api.review_edit``,
  ``Pipeline.review_edit``, ``ReviewSession.review_edit``, and the CLI's
  ``review edit --window`` -- and a curation file's ``add``/``remove`` rows is
  now optional: the Python signatures take ``window_id: Optional[int] = None``
  (the last positional, so this is backward compatible), ``--window`` is no
  longer ``required=True``, and a curation-file row's window column accepts
  the tokens already reserved for an unpinned ``create`` (``new`` / ``auto`` /
  ``-`` / an empty cell). Omitting the window derives it from the target's own
  frequency (or ``uid:N``) by live-window coverage: windows are disjoint (a
  hard Stage 4 invariant), so a frequency covers at most one live window and
  the derivation is total and unambiguous. Resolution runs at plan time, before
  a curation file's rows are coalesced by window id, so a run of several
  omitted-window rows for the same window still collapses into one refit
  instead of costing one per row; ``ReviewPreviewResult.plan`` and
  ``review apply --dry-run`` show the resolved window id. A *named* window is
  still checked -- naming the wrong one is still an error exactly as before --
  and the bounded, snap-tolerance-limited match once a window is settled on
  (named or derived) is completely unchanged: deriving the window never widens
  a search. A frequency no live window covers is an error naming it and
  pointing at ``review create``; for ``remove`` this is permanent, since a
  remove never implies creating a window. A bare ``review edit`` with no
  ``add``/``remove`` (an identity refit) still requires ``window_id``
  explicitly, as do ``review accept`` and ``review create``, since in all
  three the window is the operand rather than a coordinate for something
  else. Several omitted-window targets resolving to different windows within
  one ``review edit`` call is refused, naming both, since one call is scoped
  to one window's refit.

* **A widened window now cascades to its dependents.** ``plan_stage6_window``
  returns ``mode="widened"`` rather than ``mode="created"`` when a gap cannot
  hold ``DEFAULT_STAGE6_MIN_WINDOW_HALF_WIDTH_POINTS`` (8) points of new
  window a side: an *existing* Stage 4 window grows instead, and its existing
  peak set is refit on the wider grid. ``_batch_apply_create`` marked that
  window mutated but never dirty, so the combined-cascade pass never touched
  it -- correct for ``mode="created"``, where a fresh window is a leaf with no
  outbound dependency edge onto any neighbor, but wrong for ``mode="widened"``,
  where the window is an established one that can be a freeze source for its
  neighbors and whose fit just moved on the wider grid. A widened window is
  now added to ``dirty_wids`` alongside ``mutated_wids`` so the same batch's
  cascade reaches any dependent that had frozen on the leakage skirt the
  widening just removed; a created window is unchanged -- mutated but not
  dirty, still no cascade. ``create_window_impl``'s "No cascade, in either
  direction" documentation now says this is true of ``mode="created"`` only.

* **Stage 6 offers an amortized review session, and its type is published.**
  Every fit-mutating Stage 6 verb rebuilds the same active-FT fit context
  before it can do anything -- about 420 ms of the ~516 ms an interactive
  single-window edit costs cold -- so stepping through a worklist one window
  at a time paid that over and over.
  :meth:`Pipeline.review_session <ftmwpipeline.pipeline.Pipeline.review_session>`
  builds it once and reuses it across every verb issued through the session:
  ``review_edit``, ``review_accept``, ``review_create``, ``review_undo``,
  ``review_preview`` and ``review_apply``. Hosting the whole verb set rather
  than only the batch door is the point -- a session that sped up only
  ``apply`` would leave every interactive click paying full price. Inside a
  session, a ``review_preview`` followed by a ``review_apply`` of the
  identical plan against an unmoved base persists the preview's
  already-cascaded outcome directly instead of computing it a second time, so
  the bytes written are the ones the preview showed rather than a second
  computation trusted to agree with the first; any mismatch falls back to a
  full ordinary apply, and ``CurationApplyResult.base_changed`` says when a
  staged preview was dropped because the base moved. Correctness never
  depends on any of the reuse: every verb re-validates a cheap on-disk
  fingerprint first and rebuilds from scratch on a mismatch, which is
  byte-for-byte the rebuild a sessionless caller gets on every call anyway,
  and the fingerprint is re-read from disk after each of the session's own
  writes rather than predicted from them. **The class is now exported as**
  ``ftmwpipeline.ReviewSession``. It was reachable only as the return of
  ``Pipeline.review_session()`` while its methods were already a public
  contract -- named in this changelog and specified in ``API_STRATEGY.md`` --
  so a caller writing a type annotation for the session object had to import
  from ``_internal``, which is exactly the reach the export exists to
  prevent. The name is for annotating and isinstance-checking the object the
  ``with`` block yields; the session is still obtained from
  ``Pipeline.review_session()`` and never constructed directly. This remains
  a ``Pipeline``-class-only surface: the functional API is stateless by
  contract and a CLI invocation is a fresh process, so neither has a
  process-lifetime handle to hold open. That is not a cross-interface
  divergence -- a session changes latency, never results.

* **A curation verb can address a peak by its ``peak_uid``.** ``remove`` on
  ``review edit`` -- through ``api.review_edit``, ``Pipeline.review_edit``,
  ``ReviewSession.review_edit``, and the CLI's ``--remove`` -- and a curation
  file's ``remove`` row now accept a ``"uid:N"`` token wherever they
  previously took only a molecular frequency. The Python ``add``/``remove``
  signatures widen to ``Sequence[Union[float, str]]`` so a frequency (as
  ``float`` or a numeric ``str``) and a ``"uid:N"`` identifier may be mixed
  freely in one call, e.g. ``remove=["uid:15425022", 27549.3259]``; a curation
  file's ``remove`` row keeps its existing one-token-per-row grammar (shared
  with ``add``, unchanged) and that one token may now be ``"uid:N"``, e.g.
  ``remove,12,uid:15425022,`` -- a run of ``remove`` rows on one window still
  coalesces into a single edit, exactly as a run of frequencies always has.
  Resolution is a substitution, not a second matching path: a uid is looked
  up against the named window's fitted peaks and replaced with that peak's
  own current ``frequency_mhz`` before the existing nearest-fitted-peak snap
  ever runs, so it is always an exact match at distance 0 -- including when
  another fitted peak sits inside the ordinary snap tolerance of it, which
  frequency addressing alone cannot disambiguate. An unmatched uid raises
  ``ValueError`` naming both the identifier and the window, both live and
  from ``review apply --dry-run``'s preview, which checks a ``"uid:N"``
  target for existence in its window exactly as it already checked a
  frequency target for a snap match; a fit predating ``peak_uid`` (every peak
  ``None``) hits the same refusal, since nothing there can match. ``add`` and
  ``accept --candidate`` stay frequency-only -- a ``"uid:N"`` token passed to
  ``add`` is refused outright, since a uid names a peak that already exists
  and ``add`` has none, and ``LedgerCandidate`` (what ``accept`` resolves
  against) carries no ``peak_uid`` at all. The token grammar itself
  (``ftmwpipeline.core.curation.parse_peak_token``) is published alongside
  ``Frame`` and ``REFIT_SNAP_TOL_BINS`` so an external tool parses the same
  ``"uid:N"`` prefix the pipeline does rather than reimplementing it. One
  consequence on the CLI: ``--add`` and ``--remove`` now take a string rather
  than a ``float``, so a malformed token is refused by the pipeline with a
  message naming it (``Error: ...``, exit 1) instead of by ``argparse``
  (exit 2).

* **The HTML report's curation cart exports a remove by identifier.** Clicking
  **Remove** on a fitted line now queues that peak's ``peak_uid``, and the
  cart's **Download .csv** / **Copy** export writes it as a ``uid:N`` token
  rather than a frequency -- so the report's own export addresses the peak
  exactly, and a neighboring line inside the snap tolerance can no longer be
  matched instead. Fitted-line rows carry the identifier as a ``data-uid``
  attribute beside the existing ``data-freq``; the two are kept side by side
  deliberately, since every in-page display path (the queued-edit
  strikethrough, the on-plot markers, jump-to-edit) still keys on the
  frequency. A peak from a fit predating ``peak_uid`` carries no ``data-uid``
  and exports a frequency exactly as before, and the export's ``# frame: raw``
  header stays unconditional because ``add`` rows still carry frequencies.

* **Breaking: ``split`` and ``merge`` are removed from every input surface.**
  A curation action is now read by what it does to a window's peak set, not
  by the verb typed: an ``add`` within snap tolerance of a fitted peak that is
  not itself being removed is applied as a split of that peak, seeded at (the
  parent's position, the requested position); removing >= 2 mutually-close
  peaks while adding one frequency in their span is applied as a merge,
  seeded at the requested frequency. Both stamp ``inferred`` and the
  requested frequency on the decision's evidence, so the log reports what was
  read. Since both are now reachable through ``add``/``remove`` alone,
  ``split``/``merge`` stop being verbs a caller types: ``api.review_merge``,
  ``api.review_split``, ``Pipeline.review_merge``, and ``Pipeline.review_split``
  are removed (call ``api.review_edit`` / ``Pipeline.review_edit`` with
  ``add``/``remove`` instead); the CLI ``review merge`` and ``review split``
  subcommands are removed (use ``review edit --add`` / ``--remove``); a
  curation-file row naming ``merge`` or ``split`` is now refused at parse
  time, with the add/remove spelling to write instead named in the refusal;
  and the report cart's per-row Split button, ``into=K`` field, per-row merge
  checkbox, "Merge selected" button, and their ``m``/``s`` plot shortcuts are
  removed -- the cart's Add control (typed, or a click on the armed plot)
  reaches both, since a click that lands near an existing peak is exactly the
  add curation-intent inference reads as a split. A decision recorded before
  this change, and ``review undo``'s replay of it, are unaffected: the
  decision log and the internal appliers still carry ``"merge"``/``"split"``
  kinds regardless of how a decision was produced.

* **Every fitted peak carries a stable identifier.** ``FittedPeak`` /
  ``FinalPeak`` now have a ``peak_uid``: the peak's point-space position in
  the active FT (hundredths of a point), stamped once when the peak is born
  (a Stage 3 detection, a blend escalation, a curated ``add``, a
  ``split``/``merge`` product, ...) and carried unchanged through every
  subsequent fit and Stage 6 edit -- including a refit, where the fitted
  frequency moves but the identifier does not. It is never recomputed from a
  fitted value, so it survives exactly the operations that make frequency
  matching unreliable. Persisted in the Stage 5 HDF5 peak columns and exported
  in the CSV/JSON final-products tables; ``None`` on a fit produced before
  this field existed (no backfill). Valid only within the one Stage 5 fit
  lineage it was stamped in: tracking a peak's identity across Stage 6
  curation is the entire purpose of the field and its only supported use. A
  fresh ``fit run`` re-stamps every identifier, and any upstream change that
  moves a seed or the active-FT geometry moves the identifiers with it, so no
  cross-run meaning is promised. ``review edit --remove`` (and a curation
  file's ``remove`` row) now accepts one in place of a frequency -- see above;
  ``add`` and ``accept --candidate`` stay frequency-only. What
  ``review undo`` promises is replay equivalence: it replays the surviving decisions
  from the automatic baseline, each as its own action against the state the
  previous ones left, so the identifiers afterward are exactly those that
  sequence produces. That is deliberately **not** the same as a fresh
  curation file naming the same surviving frequencies, which coalesces a run
  of add/remove rows into one action. Undoing everything restores the
  automatic fit's identifiers exactly, and a peak the replay re-creates
  reissues its old identifier only if the replay seed is unchanged -- so a
  matching identifier across an undo is not by itself proof that it is the
  same peak; ``derivation`` is.

* **A colliding ``peak_uid`` is disambiguated, not refused.** Two seeds can
  land on the same hundredth of a point: blend escalation pushes each
  detection's seeds outward by about one point, so neighboring detections'
  seed sets interleave, and the residual re-seed places a seed at the residual
  maximum, which nothing constrains. The earlier argument that this could not
  happen bounded the spacing between *detections*, not between the seeds those
  detections spawn, and a collision aborted the whole fit with an "invariant
  violation". A window's fitted peaks are now made distinct by moving the later
  of two colliding peaks to the nearest free identifier -- deterministic, and
  idempotent across a refit, since the warm-started set is already distinct.
  Uniqueness is the property the field exists to provide; treat a recomputed
  point position as agreeing with an identifier to within a unit or two rather
  than exactly. A *curated* ``add`` landing on a birth position is still
  refused rather than nudged, because that one has a meaningful answer, and it
  now raises ``ValueError`` from the add path, naming both identifiers, rather
  than ``RuntimeError`` from the conversion path. (Curation-intent inference,
  above, has since narrowed what reaches that refusal: an add beside an existing
  *fitted* peak is read as a split of it, so what is left is two simultaneous
  adds at one identical, not-yet-fitted frequency -- neither has an existing
  peak to be read as a split of.)

* **``FittedPeak.peak_id`` is renamed to ``detection_index``.** The field is
  provenance -- the index into the persisted Stage 3 promoted-peak list of the
  detection nearest a fitted line -- and it is explicitly non-unique: several
  fitted lines in a blend share one value. The old name read as an identifier,
  which invited confusion with ``peak_uid`` above, the field that actually is
  one. This is a rename only; the value and its non-uniqueness are unchanged.
  The published ``fit_peaks`` read column and the ``review show`` column
  header follow the same rename. A Stage 5 HDF5 file written before this
  change still loads -- the old ``peak_id`` column is read as
  ``detection_index`` -- but every file written from here on carries the new
  column name only.

  Two published methods change signature with it:
  ``FittingResult.set_shared_parameter`` and
  ``FittingResult.set_fixed_parameter`` took a ``peak_ids`` keyword and wrote a
  ``peak_ids`` key, both now ``detection_indices``. Neither is called anywhere
  in the package, and the key was never persisted -- ``shared_parameters`` is
  rebuilt in memory on load rather than serialized -- so this reaches only a
  caller that constructs a ``FittingResult`` by hand.

* **``Pipeline`` methods let a typed error propagate instead of flattening it
  into ``RuntimeError``.** Eighteen methods (``compute_ft``, ``fit_peaks``,
  ``estimate_noise``, ``calibrate_tau``, ``calibrate_timebase``, and the rest
  of the stage-driving surface) wrapped their whole body in
  ``except Exception as e: raise RuntimeError(f"Failed to ...: {e}") from e``,
  so a caller could route on nothing but a substring of the message. That
  flattened a ``ValueError`` from a stage's own input validation — a mistake in
  what the caller asked for — into the same type as a genuine internal failure,
  and it flattened the whole ``PipelineFileError`` family, including
  ``AnalysisEpochMismatchError``, which was given its own type precisely so a
  consumer could route on it rather than match a message. The refusal added
  below for an ``end_us`` past the record is the case that motivated fixing
  this now: it surfaced as ``RuntimeError: Failed to compute FT: end_us=100 us
  is past the end of the recording (12.65 us)`` — the words survived, the type
  did not. ``ValueError``, the ``PipelineFileError`` family, and
  ``FileNotFoundError`` (a missing pipeline file or preset path is the same
  class of user mistake, and is not a ``ValueError``) now propagate with their
  own type; the ``RuntimeError`` wrap remains only for an exception none of
  those methods' own contracts name.

  This is a **behavior change for an existing caller that catches
  ``RuntimeError`` around one of these methods** — that ``except`` clause may
  stop matching once the underlying failure is a ``ValueError`` or a
  ``PipelineFileError`` subclass instead. Nothing is lost in the process: every
  wrap still chains with ``from e``, and every propagated error is exactly what
  the failing stage raised, so a caller that widens to ``except Exception`` (or
  adds the specific type it now needs) sees the same message it always did.

* **The curation cart's CSV export declares its own frame.** The interactive
  report's cart wrote a bare ``action,window,freqs,params`` header with no
  ``# frame:`` directive. Since omitting the frame on a frequency-bearing
  curation file became a hard error on a ``self_calibrated`` file, the cart's
  own export — and the ``review apply`` command the cart prints beside it —
  could not be applied to such a file at all, not even with ``--dry-run``. The
  cart emits raw Stage 5 model frequencies deliberately: the value the edit
  verbs match on, not the calibrated value shown in the table. So the export
  now leads with ``# frame: raw``, which is the honest declaration of what it
  carries. The printed command is still left without ``--frame`` on purpose —
  the file's own header answers the question now, and an explicit ``--frame``
  that disagreed with the header would be refused.

* **An active region can no longer claim to be longer than the recording.**
  ``end_us`` was validated only for being non-negative and greater than
  ``start_us`` — never against the length of the FID. The sample slice clamps
  itself to the record, but the *declared* length did not, so an ``end_us``
  past the end of the data made the two disagree without limit. Since the
  active region's length sets the active-FT bin spacing that every
  bin-relative tolerance resolves against, the whole pipeline then measured
  against the bin spacing of a spectrum that does not exist: on a 12.65 µs
  record, ``compute_ft(end_us=100.0)`` made the published snap tolerance
  report 6.25 kHz instead of 49.4 kHz — 8× too tight, with the candidate-dedup
  window, the spur integer gate, the timebase scan and the frame-mismatch
  floor all wrong by the same factor, silently and with every number finite.

  Stage 1 now **refuses** such a window, naming both the requested end and the
  record's duration. It is refused rather than trimmed because omitting
  ``end_us`` already means "to the end of the record", so there is a correct
  path that costs the caller nothing and an out-of-range value can only be a
  mistake. Independently, ``active_acquisition_us`` now clamps to the record,
  so no derived quantity can describe more data than exists whatever route it
  arrives by — the refusal stops bad input at the door, and the clamp is what
  makes the refusal unnecessary to trust.

  ``validate`` reports a file that already carries such a window rather than
  silently correcting it: current reads of that file are now right, but its
  persisted Stage 3–5 results were computed against the over-long length, and
  the honest thing is to say so and let the owner re-run Stage 1. No epoch
  implication — the refusal moves no number on any valid file, and an affected
  file is named rather than quietly changed.

* **A preview window with no fit reports its absence, not a zero.**
  ``PreviewWindowResult.chi2r_before`` / ``chi2r_after`` were plain ``float``
  defaulting to ``0.0``, so a window the batch itself *creates* -- which never
  had a "before" fit -- reported a real, finite ``chi2r_before = 0.0``. A
  consumer rendering "before → after" showed ``0.00 → 1.4`` and read it as a
  perfect fit that got worse; χ²ᵣ = 0.0 is also a value a genuine fit
  essentially never produces, so the fabricated number was indistinguishable
  from an extraordinary one. Both fields are now ``Optional[float]`` and are
  ``None`` when that side carries no fit, the same absence-versus-plausible-
  number distinction ``CalibrationStamp`` already makes for ``probe_freq_mhz``.
  ``review preview`` prints ``-`` for an absent side. ``n_peaks_before == 0``
  was the available tell for this case and remains true, but it was a default
  riding alongside rather than a contract -- and a window can legitimately be
  emptied to zero peaks by an edit. ``RefitWindowResult``'s same-named fields
  are unchanged: an interactive verb always refits a window that already
  existed.

* **Spectral tolerances are defined in active-FT bins, not in MHz**
  (``ANALYSIS_EPOCH`` 2 → 3). A tolerance that expresses a distance *in a
  spectrum* is a property of the resolution, so freezing one in MHz encodes a
  single laboratory's acquisition length and is wrong everywhere else — too
  coarse at long acquisitions, where it merges genuinely resolved lines, and
  too fine at short ones. Each such constant is now defined as a multiple of
  the active-FT bin spacing ``1 / (end_us - start_us)``, resolved per file
  through the one accessor ``fitting.active_ft.active_ft_bin_spacing_mhz``:
  the candidate-dedup window (0.25 bins), the spur integer-MHz gate (0.5) and
  the separate spur-merge tolerance it had been sharing a number with by
  accident (0.5), the timebase sub-bin scan range and step (1.25 and 0.00125),
  the Stage 6 frame-mismatch floor (0.05), and the curation snap tolerance
  (0.625).

  Every one of them turned out to be an exact round bin count against a
  *nominal* 80 kHz spacing, so these were designed in bins and written down in
  MHz; the conversion recovers the original definitions rather than inventing
  new ones. The true reference spacing is 79.052 kHz, so **each has been
  running about 1.2 % off its intended value**, and correcting that is what
  moves fitted output on an existing file. Two of them additionally lose an
  absolute floor that used to win outright at long acquisitions: the spur
  integer tolerance was ``max(0.04 MHz, 0.5 bins)``, pinned at 40 kHz and
  spanning many bins once the record was long enough, and the snap tolerance
  was a flat 50 kHz. Both now tighten with resolution as they were meant to.
  A file's persisted ``spur.integer_tol_mhz`` knob migrates to bins at read
  time, exactly and per file — the file carries its own active region, so no
  default-and-hope is involved.

  Not everything absolute was converted, and the reasons are recorded at each
  definition rather than left to be re-derived: the two clock-frequency
  comparison tolerances (``1e-6``) are float comparisons on a value a person
  typed, not distances on any grid; and the Stage 2 scatter widths and the
  Stage 4 window bounds are genuine spectral widths — what they must be wide
  relative to is how fast the receiver's noise floor varies and how wide a
  line-dense band is, both measured in MHz. The window bounds' bin-relative
  form already existed and is already the default (``*_POINTS``), with the MHz
  form as the fallback a caller selects by zeroing it.

  **The published surface changes shape.** ``REFIT_SNAP_TOL_MHZ`` is deleted,
  with no deprecation alias — an alias would preserve exactly the
  "resolve it yourself" surface this work removes. In its place
  ``REFIT_SNAP_TOL_BINS`` (0.625) is the definition, and
  ``api.refit_snap_tol_mhz`` / ``Pipeline.refit_snap_tol_mhz`` /
  ``ftmwpipeline review snap-tolerance`` resolve it for a named file: 49.4 kHz
  at the 12.65 µs reference active region, 6.3 kHz at 100 µs. Read the
  accessor rather than multiplying the bin count by a spacing of your own —
  it derives from the same active region the curation verbs consult, so it
  cannot disagree with what a ``review apply`` on that file will snap with.
  Every verb's ``snap_tol_mhz`` parameter now defaults to ``None`` (resolve
  for this file) instead of to a number; passing an explicit MHz value still
  overrides it for that call.

  The snap tolerance keeps **no absolute floor**, which is a deliberate
  reversal of its own recorded rationale. The old 50 kHz was justified by
  *input* precision — "the input is a frequency a person typed off a plot or a
  line list" — and human typing does not get finer as the acquisition gets
  longer. It is overruled because the plot the person reads has exactly this
  resolution, an FTMW line list is quoted to ~1 kHz, and a miss is loud
  (``remove`` raises and names the closest peak and its distance) rather than
  silently wrong.

* **``fit show`` stops holding every rendered figure open.** Each selected window
  renders one figure, three with ``--apodize`` and ``--rescue``, and all of them
  were retained so an interactive caller could display them — including when the
  caller had asked for ``--output-dir`` and was only going to be told the paths.
  On a line-dense file ``--all-windows`` therefore built hundreds of live
  matplotlib figures to write hundreds of PNGs, and tripped matplotlib's own
  ``figure.max_open_warning`` on the way. The CLI now declines the figures when
  it has an output directory or is non-interactive, and each is closed as soon as
  it is on disk, bounding live figures at one instead of three per window. The
  written PNGs, the log and the reported paths are unchanged, and the ``Pipeline``
  and functional-API contract is unchanged: they still return the figures, since
  a caller holding a reference is the case the retention exists for.

* **The curation snap tolerance is public.** Matching "the peak at *f*" the way
  a curation refit matches it requires the refit's snap tolerance, which existed
  only as a private constant — so a consumer pairing peaks across a refit had to
  reach into ``_internal`` or hardcode a copy, and either way could disagree
  with the file about which peak was meant. It is now exported from the
  top-level package and defined once in a new stdlib-only ``core.curation``.
  (The published spelling has since become ``REFIT_SNAP_TOL_BINS`` plus a
  per-file accessor — see the bin-relative-tolerances entry below, which
  supersedes the absolute ``REFIT_SNAP_TOL_MHZ`` described here.) That single definition is the point:
  the value had been spelled as a bare literal in all eight public signatures
  while only the internal implementations referenced the constant, so publishing
  the name without collapsing the copies would have moved the drift up a level
  rather than closing it. Two asymmetries surfaced by the same audit are closed
  with it — ``review_accept`` now exposes the ``snap_tol_mhz`` its implementation
  always had, and the CLI gained the matching flag, so the parameter reaches all
  three interfaces. ``review apply`` deliberately gains no tolerance knob, since
  an override there would recreate the disagreement the published value exists
  to prevent. The private ``_internal.stage6_impl._REFIT_SNAP_TOL_MHZ`` alias is
  **not** kept: a private read that keeps working is how it survives in an
  integration forever.

* **The analysis-epoch refusal has a type.** Splicing a Stage 6 edit into a fit
  produced under a different epoch has been refused since ``0.1.0b3``, but it
  refused with a bare ``ValueError`` whose only distinguishing feature was its
  wording — so a caller routing that one refusal to a recovery flow (offer a
  re-run under the current environment, rather than a generic "the edit failed")
  had nothing to key on but a substring of a message that is ours to rephrase.
  ``AnalysisEpochMismatchError`` subclasses both ``PipelineFileError`` and
  ``ValueError``: the ``ValueError`` half keeps every existing catch site working
  unchanged, and the message text is byte-identical. It carries both epochs and
  both environment records, so the mismatch can be displayed without parsing the
  message it replaces. The whole ``PipelineFileError`` family is now exported at
  top level rather than only the new member. Relatedly, the ``"<field_name>: "``
  prefix on each drift line from ``describe_environment_drift`` /
  ``describe_runtime_drift`` is now documented as a contract — while stating
  plainly that the prose after the colon is not stable.

* **``run --help`` shows the flags that decide a run.** ``run`` mirrors every
  stage's knob surface, so argparse dumped all ~150 options and printed 315
  lines, drowning the dozen flags that actually determine what a run does. The
  per-stage knobs are now hidden from the default help and grouped behind
  ``--help-knobs`` (bare for everything, or scoped to one stage), with the epilog
  naming each prefix and the per-stage command that documents the same knobs
  un-prefixed. Hidden is not disabled — every flag parses exactly as before, and
  a test pins that — because an option that should not be used should be deleted,
  not concealed. 315 lines to 77.

* **``info`` compares the file against the environment running it.** The
  environment report answered only "were this file's stages produced by the same
  code", which a file stamped uniformly by one release always passes — including
  when the interpreter about to edit it is a different release entirely. That is
  the mismatch a caller of ``info`` most needs, and it was the one thing the
  report could not express, since it never looked at the running process. ``info``
  now carries ``current_environment`` and ``runtime_environment_drift`` alongside
  the existing cross-stage ``environment_drift``, and warns when the *epoch*
  differs, which is the difference that will refuse a Stage 6 edit. The two
  comparisons stay separate rather than pooled: a file can disagree with itself,
  with the running code, or with both, and the fix differs in each case.

* **A stage re-run on a file that predates environment stamping says so.** An
  unknown epoch is treated as compatible — refusing to work on a legacy file
  would punish users for an upgrade they did not choose — but that leniency spans
  an arbitrary version gap in silence, and re-running a stage over an unstamped
  result is the one case where no gate, no drift report and no epoch check can
  say whether the new numbers match the old ones. That re-run now logs that
  reproducibility against the original run cannot be verified. It fires only on a
  genuine re-run of an unstamped stage: a stage running for the first time has no
  earlier result to disagree with, and a stamped file is checked by the epoch gate
  as before. Relatedly, the record's semantics are now stated where they are
  discoverable: a Stage 6 edit deliberately does not re-stamp ``stage5_fitting``,
  because that stamp names the automatic fit every edit gates against.

* **``review apply --dry-run`` validates ``add`` actions.** The preview resolved
  ``remove`` / ``merge`` / ``split`` targets against the fitted peaks but said
  nothing about an ``add``, on the reasoning that an add creates its peak and so
  has no target to match. It does name a *window*, though, and that is exactly
  what goes stale: window ids are reassigned whenever Stage 4 re-plans, and a plan
  window whose peaks all failed their Stage 5 gate carries no fit to edit at all
  (a large fraction of the plan on a line-dense file). A curation file written
  against an earlier state could therefore pass a clean dry run and then fail on
  the apply. The dry run now checks each add for the two conditions the refit
  enforces — the window is live, and the frequency lies on its data, with the snap
  tolerance allowed as slack so only an add that cannot land however it snaps is
  flagged. An add into a window a ``create`` in the same file installs is left to
  the apply, since its geometry does not exist yet.

* **The Stage 2 noise methods note quantifies the Rician ``C(R)``
  monotonization.** The isotonic replacement of the raw Monte-Carlo table (before
  ``0.1.0b1``) was characterized by its effect on a typical bin, which is under
  half a percent and understates it: the two curves differ most where ``R``
  saturates at the Rayleigh limit, i.e. in the noise-only bins that dominate a
  full-spectrum median. On one line-dense real file the median RMS moved +4.5%,
  pruning about a tenth of the final fitted peaks — all threshold-marginal, with
  the strong lines untouched. The note now states the effect on the noise-floor
  median and the resulting line count, which is the honest quantity for a change
  in the correction curve.

* **``read_metadata`` reads a JSON-blob ``ft_processing`` record.** Early
  records stored the Stage 1 settings bundle as one JSON ``parameters``
  attribute rather than as individual attributes, a shape Stage 1 has always
  accepted. This view read only the individual attributes, so a blob-only file
  reported no ``ft.units_power`` at all while the display transform read one
  from the blob — the same question answered two ways depending on which reader
  a consumer held. The fold now lives once, in
  ``_internal.shared_utils.fold_settings_blob``, and both readers go through it,
  so they cannot drift apart again. Individual attributes still win where both
  are present, and this only ever widens what is readable.

* **Every Stage 6 edit runs through one engine.** The interactive verbs
  (``review edit`` / ``merge`` / ``split`` / ``accept`` / ``create``) each held
  their own copy of the load-edit-cascade-persist sequence alongside the batch
  engine's, which is how the analysis-epoch gate came to be missed on one path.
  They are now the batch engine applied to a single action, so the epoch gate,
  the undo baseline, the per-band τ anchor, the spur-catalog replay, the cascade
  and the persist have one definition each and reach every caller at once.
  Fitted results are unchanged — bit-identical on a real fixture. Two
  user-visible consequences: ``review merge`` and ``review split`` no longer
  load the FID twice, and an identity refit (``review edit`` with nothing added
  or removed) now refreshes the calibrated final-products table, which it
  previously left stale after re-converging the window.

* **``review apply`` and ``review undo`` apply a curation plan as one batch.**
  Each action used to rebuild the whole Stage 5 fit context from scratch — its
  own FID load, FT, noise estimation and spur-catalog replay, its own
  read-modify-write of ``/stage5_fitting``, and its own cascade — so a session of
  edits paid that setup once per action. Measured on a three-edit file, about
  three quarters of each action was redundant setup. The engine now builds the
  context once, applies every action to one in-memory ``SpectrumFit``, cascades
  the dependents of all directly-edited windows in a single combined pass, and
  persists once; on that file the FT rebuilds went 3 → 1 and the wall time 3.9 s
  → 2.3 s. Nothing is written unless every action succeeds, so a plan that fails
  partway through now leaves the file untouched rather than holding part of
  itself. The fitted result is unchanged — bit-identical on a real fixture — and
  the guarantee is stated as equivalence of outcome: cross-window execution
  order is canonical (creates first, then ascending window id), so two curation
  files listing the same per-window edits in different row orders reach the same
  fitted state and the same decision log. Order within one window is still the
  order specified, since a ``merge`` or ``split`` composes on what a preceding
  ``add`` or ``remove`` left behind.

* **The fit stage's progress percentage counts finished windows.** Under the
  dependency-gated parallel walk each worker logged the scheduling position it
  had been handed at submission time, not a completion count, and windows finish
  in a different order than they are dispatched. The percentage therefore jumped
  around and settled wherever the last window to finish happened to sit in the
  schedule — a 255-window fit ended its stage displaying ``1% (5/255)``. The
  count is now emitted by the parent process as each result lands, so it rises
  monotonically and terminates at 100%; the per-window detail line is logged
  separately and identified by window id. ``StageProgress`` additionally holds a
  per-stage high-water mark, so a stage that legitimately re-runs a batch of
  sub-steps cannot drive the bar backwards. The CLI, the ``Pipeline`` class and
  the functional API share one reporter and all three are fixed by this.

* **Stage 5 no longer warns that the Stage 2b pre-conditions did not pass.** The
  warning fired at fit time, named Stage 2b, and then said the fit would consume
  the calibration anyway — a failure report with no failure and no action behind
  it. Its most common trigger is a bimodal τ histogram, which is the expected
  signature of a decay with several genuine τ populations rather than a fault.
  The pre-condition result is unchanged and still reaches the user where it can
  be acted on: ``tau run`` prints the failing notes in its summary, the report
  raises it as a Stage 2b concern alongside the per-band τ variation, and
  ``read`` still exposes ``preconditions_passed``. Stage 5 still warns when a
  Gaussian fit finds no τ_G calibration at all, which does name a remedy.

Version 0.1.0b4 (2026-08-10)
----------------------------

A read-access and linewidth beta over 0.1.0b3. As a pre-release it still
installs only when explicitly requested: ``pip install --pre ftmwpipeline``.

Mostly a second way to *read* what a ``.ftmw`` already holds. One change is not
additive, and it moves fitted numbers: the finite-``T`` linewidth is now solved
for rather than measured on a fixed grid, which shifts widths by up to about
2e-4 relative at a typical acquisition length and so can flip a
peak-separation decision at a boundary. **``ANALYSIS_EPOCH`` is 2.** Reading is
never gated and running a stage forward only warns, but splicing a Stage 6 edit
into a fit stamped with epoch 1 — that is, one produced by 0.1.0b3 — is refused
until it is re-fit or acknowledged. Fits written before 0.1.0b3 carry no epoch
stamp at all; an unknown epoch is treated as compatible and is not refused.

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
  to the solver tolerance at every ``T``, where the old grid's step was fixed in
  absolute frequency and so cost relative precision as the line narrowed with
  increasing ``T``; it no longer clips, where the old grid silently returned its
  own 2 MHz span once the true width outran it (below ``T = 0.92 us`` for a
  Lorentzian at ``tau/T = 0.3``); and it is about 25x faster (5.2 ms to 0.20 ms
  per call), which the fit itself feels, since the separation constraint and the
  blend-aware seeder both call it. Widths move relative to 0.1.0b3 by up to
  7.6e-5 at ``T = 6 us``, 1.8e-4 at 11.7, 3.3e-4 at 25 and 7.2e-4 at 60 —
  growing with ``T``, as the mechanism implies — so fitted output is not
  bit-identical, hence the epoch bump. No reference-fixture result in the suite
  actually changed; the gate is on the possibility, not on one fixture.
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
