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

**A remove is logged at the peak it removed; ``CONTRACT_VERSION`` moves 15 → 16.**
A ``remove`` decision-log row's ``frequency_mhz``, and a merge's ``merged_from``,
are now the fitted frequencies (raw frame) of the peaks the request resolved to,
not the frequencies sent. An undo or log-prefix replay runs at the file's snap
tolerance, so a remove made with a widened ``snap_tol_mhz`` (``--snap-tol-mhz``)
that only the wider tolerance could resolve made every later undo on the file
fail; the replay now finds the logged peak at no distance. Rows written before
hold the frequencies sent. A remove sent in the wrong frame still resolves to
the same peak, and its log row now names that peak rather than the shifted
frequency.

**Decision-log action groups, empty-window ``converged``, canonical ``run_pipeline`` and
provenance names; ``CONTRACT_VERSION`` moves 14 → 15.** Four changes from the
documentation audit. ``action_index`` and ``failed_step`` are additions; the
values of ``run_pipeline``'s ``completed_stages`` and ``failed_stage``, a bare
accept's evidence, the shape recommendation's environment key and ``converged``
on a window with no peak change.

* **Undo and log-prefix replay go one user action at a time.** Every decision-log
  row now carries ``evidence["action_index"]``: the ``order_index`` of the first
  row of the same user action (a single-row action, a bare accept included,
  carries its own; a bare accept's evidence is no longer ``{}``). ``review undo``
  and ``review apply --log-prefix`` replay a group's surviving rows jointly as
  one action, one joint refit, so undoing part of a group replays the rest
  jointly, and a log prefix that cuts through a group replays the in-prefix rows
  jointly. A file written before the key existed carries none; its groups are
  inferred (consecutive ``add``/``remove`` rows on one window with identical
  non-empty evidence and no ``created_window``). The undo dry-run plan lists one
  edit per group. This fixes a later undo failing after a multi-line edit: each
  row used to replay as its own refit, the separate refits drifted the fitted
  peaks, and a later remove no longer snapped to the peak it named. The key is
  part of ``evidence``, so a hash of the log taken over ``evidence`` differs from
  one taken before.
* **``converged`` is ``Absent.UNDEFINED`` for a window left with no peak.**
  ``RefitWindowResult``, ``PreviewWindowResult`` and ``AppliedWindowResult``
  report ``converged`` as *undefined* (wire: ``null`` with
  ``"converged_absent": "undefined"``) when the window is created empty or has
  every peak removed: no solver ran. No non-convergence warning is raised, and
  the report's ``n_nonconverged`` (and its "N window(s) did not converge" line)
  no longer counts a window with no peak, a Stage 5 window knocked down to the
  null model included. The persisted ``success`` is unchanged.
* **``run_pipeline`` names stages canonically.** ``completed_stages`` is now the
  canonical stages written, in order and de-duplicated, the list a cancel's
  ``completed_stages`` holds; start detection and the report are steps, not
  stages, and add nothing. A Gaussian tau run lists ``tau_g`` (``tau`` only if a
  twin was built). ``failed_stage`` is the canonical stage of the failing step,
  ``None`` for start detection or the report, and ``tau_g`` for a failing
  Gaussian tau step. A new additive ``failed_step`` holds the progress label of
  the failing step (``"import"``, ``"start detection"``, ``"FT"``, ...,
  ``"report"``), ``None`` on success; ``run --json`` carries it as a scalar
  (``completed_stages`` stays one comma-joined string). The human failure line
  reads ``failed at step '<label>'``.
* **The shape recommendation is published as ``tau_shape``.** Its entry in
  ``stage_environments`` (``get_pipeline_info``, ``validate_pipeline``) used to
  appear under its storage name; the storage path
  ``processing_parameters/stage2b_shape_recommendation`` is unchanged. Every
  published drift or environment surface now names stages canonically: the
  ``environment_drift`` event message, the report table's ``environment_mixed``
  row, the HTML report's environment table and the re-run drift log. The
  contract exports ``canonical_provenance_name`` and ``PROVENANCE_NAMES``
  beside ``stage_for_key`` / ``key_for_stage``.

**Fixes from the documentation audit.**

* A source its format's loader refuses at import — an unknown sidecar key, a
  CSV without ``spacing_us``, an unknown ``--column`` — is now ``bad_setting``
  (``path`` ``"source"``) with the loader's message, as promised, instead of
  ``RuntimeError("Failed to load FID data")``; ``data import --json`` prints the
  ``ftmw/error@1`` dict. ``preview_source`` given a source its named (or
  detected) format refuses raises the same error instead of a plain
  ``ValueError``. ``data import`` no longer turns a second Ctrl-C into exit 1.
* ``review undo --id`` takes one or more ids per flag and repeats
  (``--id 3 5`` or ``--id 3 --id 5``), so the report's curation cart command
  runs.
* ``report run --summary`` no longer builds window pages it then drops: it
  draws only each window's magnitude panel (the index's hover thumbnail). The
  file is unchanged; on ``exp_2638`` it takes about half the time it did.
* ``start run`` no longer says a later FT run will inherit the stamped start
  once Stage 1 has run; it says how to adopt it (``ft run --start-us``). The
  ``target_outside_window`` message names the implied create of an add with no
  window.
* The invalidation warning log line names stages canonically (``noise``,
  ``fit``), as the ``Invalidated`` event does, not by storage key.
* Typed errors print as JSON under ``--json`` and the ``--format json``
  synonyms only; ``report run --format json`` (the table file's format) and a
  dump written with ``--output`` keep text errors.
* ``run --help``, the top-level ``--help`` and the ``run_pipeline`` /
  ``Pipeline.build`` docstrings state the real run order and verbs.
* ``noise run`` with any knob flag (``--window-mhz 60``) no longer crashes
  echoing the settings it was given.
* The ``frame`` refusal on a ``self_calibrated`` file states the frame mix-up
  as ``|f - probe_freq| * eps/(1+eps)``, the line's own offset, and quotes its
  largest value over the file's fitted windows (31.4 kHz on ``exp_2638``),
  instead of the ``probe_freq * eps/(1+eps)`` constant (89.7 kHz there), which
  no line is off by.
* The ``defaults`` preset carries every knob with a package default: it gained
  ``stage5.tau.fit_tau`` and ``stage5.rescue.final_add_snr_threshold``.
  Applying it still changes nothing.
* ``review create`` suggests a ``review edit`` that carries the ``--frame`` it
  was given, so the suggestion runs on a ``self_calibrated`` file.
* ``timebase run --help`` says the clock declaration falls back to the one
  recommended at import. ``review undo --help`` and the ``baseline_unavailable``
  message name the real cause of a missing baseline (edits recorded before
  ``review undo`` existed), not a fit re-run, which discards the decisions too.

**A window the fit leaves empty while its edge stays coherent is flagged for
review; ``CONTRACT_VERSION`` moves 13 → 14.** Stage 5 can finish a window of its
plan with no line while the residual on the window's edge stays coherent. The
structural replan records that flag as ``not merged`` (an empty window has no
fitted line straddling its boundary), and until now nothing pointed the user at
the window. ``review run`` now flags such a window: a window of the fitted plan
the fit holds no line in, not taken over by a created window or edited, whose
edge Stage 5's thaw or replan left flagged above the fit's own
``residual_edge_threshold``. When at least one Stage 3 peak in the window sits
off every gated spur, the item is ``empty_window_residual`` and queues the
window. When every one sits on a gated spur, the residual is consistent with
the spur's skirt beyond its mask, and the item is the advisory
``empty_window_spur`` (shown on the window's status and in the report, not
queued), naming the spur.

* ``AttentionReason`` gains ``evidence``, a dict of the kind's declared keys.
  The new items name the flagged ``edges`` (``side``, ``s_coh``, and the
  distance to the nearest fitted line beyond the edge), the
  ``residual_edge_threshold``, and the ``candidates``: the Stage 3 peaks the
  plan put in the window, with their SNR, whether each sits on a gated spur, and
  that spur's centre and source. Every record carries every key, with
  ``Absent`` for a missing value (a degenerate Stage 3 SNR reads
  ``undefined``).
  Its ``severity`` is the strongest edge's ``S_coh`` over the threshold; its
  ``locations`` are the candidates' frequencies.
* The contract declares ``AttentionReason`` (``kind``, ``detail``,
  ``severity``, ``locations``, ``evidence``) and the ``attention_kind``
  vocabulary. ``review show --attention --json`` rows also carry every reason
  under ``reasons``, and ``review show --window N --json`` gives each reason's
  ``locations`` and ``evidence``.
* The window has no Stage 5 result, so its status is one ``load_fit`` has no
  window for. ``ReviewRunResult.n_windows`` counts it whether queued or
  advisory; ``n_attention`` counts it only when queued. ``review show``
  reports it (no fitted line, ``reduced_chi2`` undefined), its ``--output``
  renders draw the data with nothing fitted, and the report gives it a page
  (under ``--windows attention`` only when queued): the data and residual
  (equal, since nothing is fitted) with the candidates marked **E**.
* The item reads only records of the window's current fit: a replan record
  measured before a structural merge re-fit the window is ignored.
* To fit the line, create a window at it and add the line to the new window;
  once a created window covers the flagged peak, it takes the empty one over and
  the item (and a status left with nothing) is dropped in the same call. A
  created window that only overlaps the empty one leaves the item.
  ``review accept`` may now name such a window (a bare accept, marking it
  reviewed), alone or in any batch, preview or undo replay; before, a window the
  fit holds no line in was ``not_found``.

On 2638 (cleanup-golden recipe) windows 100, 158 and 227 are flagged in both
line shapes, edges ``S_coh`` 17.6 / 11.4, 10.8 / 10.8 and 23.5 / 18.2 against a
threshold of 8. Each window's only Stage 3 peak (30719.94, 32960.00 and
35839.97 MHz; SNR 35, 69 and 109) sits on a spur the fit gated as saturated, so
all three are advisory ``empty_window_spur``; nothing is queued. ``review run``
now reports 265 windows instead of 262 (Gaussian) and 273 instead of 270
(Lorentzian), with ``n_attention`` unchanged (6 and 1). Fitted numbers, final products and the analysis
fingerprint are unchanged, so ``ANALYSIS_EPOCH`` stays 5.

**Stage 5 structural replan fixed, merged windows carried to the end;
``ANALYSIS_EPOCH`` moves 4 → 5 and ``CONTRACT_VERSION`` 12 → 13.** Stage 5's
structural replan applies merges again (before this, a round in which any window
was named twice was rejected whole, which on 2638 happened in every fit), and a
fit that applied one now reports and edits the windows it was made on.

*The replan.* When a window edge still carried a coherent residual with no
contributor to thaw, the fit asked to merge the window with the nearest window
on that side *at any distance* and sent every request to Stage 4 at once. A
window flagged on both edges was named in two requests, Stage 4 rejected the
second (it named a window the first had absorbed), and the whole round was
recorded as failed. A round whose requests named disjoint windows did apply,
which is why fits with merged windows exist from before this change. On the 2638
fixture a window was named twice in every fit, Gaussian and Lorentzian, and most of the requested partners were 5 to 114 MHz away, so
applying them as asked would have built windows far wider than the plan allows.
Now:

* A merge is asked for only from a window whose fit holds at least one line (a
  window the cleanup emptied has no fitted line straddling its boundary).
* The partner must **touch** the flagged window: at most one active-FT bin
  between them, as the planner leaves windows it split or whose margins just
  miss.
* The merged window must fit the plan's width cap and, when the plan sets one,
  its ``max_peaks_per_window``.
* Each round applies a disjoint set of pairs, strongest flagged edge first. A
  request that shares a window with a merge already chosen is deferred; next
  round the triggering window's current fit is scanned again against the revised
  plan and the request is judged afresh.
* A request Stage 4 rejects is recorded as failed and the round's set is chosen
  again without it, so a request that waited behind it gets its turn.
* A merge drops the thaw, rescue and cleanup records of every window it re-fits
  or absorbs.
* A survivor's per-band decay-time anchor is resolved again from its merged
  range before it is re-fit, so the merged window is fit on the anchor a Stage 6
  refit of it resolves (before, it kept its Stage 4 window's band, and a no-op
  edit of a survivor whose merged centre lay in another band moved its lines).
* Parallel and sequential fits choose the same merges.
* Every replan record that is not accepted gives its reason under one of four
  prefixes, ``not merged:``, ``refused:``, ``deferred:`` or ``failed:`` (a
  Stage 4 rejection read ``replan failed:`` before).

On 2638 no merge qualifies, in either recipe or line shape. On the
cleanup-golden plan the flags come from windows 100, 158 and 227, whose fits the
cleanup emptied, and (Gaussian only) from windows 59 and 179, which no window
touches: 9 records Gaussian and 6 Lorentzian, all ``not merged``. The fitted
windows and lines are bit-identical to epoch 4 in both shapes (262 windows and
511 lines Gaussian, 270 and 644 Lorentzian); only the replan records' reasons
change. The epoch moves because a merge can now apply wherever a pair
qualifies. A file fitted under epoch 4 must be re-fit, or have the mismatch
accepted, before Stage 6 will splice an edit into it, and a partial fit written
under epoch 4 starts over instead of resuming.

*Windows after a merge.* A merge leaves the fit on windows Stage 4 did not plan:
the survivor keeps the lower id and the merged range, and the absorbed id is
gone. Stage 5 now stores that plan with the fit (``/stage5_fitting/fitted_plan``,
only when a merge revised it), and everything that reads window geometry after a
complete fit reads it. Before this, Stage 6 kept refitting a survivor on its
narrow Stage 4 range, so even an edit that changed nothing dragged the absorbed
window's lines to the edge, and ``window_status`` reported the Stage 4 rows.
Now:

* ``window_status`` reports the fitted plan, and ``WindowStatusRow`` gains
  ``merged_from``: the ids a survivor absorbed, ascending, and empty for every
  other row (a tuple, so rows stay hashable; an array on the wire). The
  ``window_status`` table gains the column as JSON text (``"[]"``,
  ``"[101]"``).
* ``load_windows`` and the ``windows``, ``window_free_peaks`` and
  ``window_contributors`` tables stay the Stage 4 plan as planned; a program
  joining fit rows to windows joins them to ``window_status``.
* Every Stage 6 call resolves, edits, refits and plans against the fitted plan,
  and the undo baseline carries it. A created window never takes an absorbed
  id; a curation call naming one is ``not_found`` (kind ``window``), and a
  pinned create that would take one is ``replay_conflict``.
* A fit made before this record, in which a merge was applied, keeps working:
  ``window_status`` reports the merges its replan record names, and Stage 6
  refuses to refit the merged windows (or the windows reading their lines as
  fixed contributors), to cascade an edit into a merged window whose fit read
  the edited window's lines, or to create a window inside a merged range, with
  the new
  ``curation_conflict`` reason ``fit_plan_unavailable``. Re-running ``fit run``
  clears it. Fits no merge revised, including every earlier fit of 2638, store
  nothing extra and edit exactly as before.

**``ANALYSIS_EPOCH`` moves 3 → 4.** A dependent window's frozen contributors
now follow the window's fitted decay time *during* the fit, as the Stage 5 model
equation already wrote them (see :doc:`stage5_fitting`). Previously their
leakage skirt was subtracted once at the starting decay time and stayed there
while the decay time was fitted, so a free-decay window with frozen
contributors was fit to a different model from the one it reported. Only
those windows move: ones whose decay time is free *and* that hold frozen
contributors (a dependent window's neighbour lines, and the inherited lines the
Stage 5 collapse merge parks while the decay time is free). Windows whose decay
time is held are bit-identical, and so are windows with nothing frozen. On the
2638 fixture 6 of 262 windows moved, all through the merge's parked lines, by
at most about 1e-3 us in decay time. A file fitted under epoch 3 must be
re-fit, or have the mismatch accepted, before Stage 6 will splice an edit into
it. The same epoch also corrects a local thaw's joint co-fit, which drew the
dependent window's *other* frozen contributors shifted by the distance between
the two window centres; their skirt now sits at the lines' true frequencies.

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

* **Machine contract, cleanup wave: typed curation and import refusals.**
  ``CONTRACT_VERSION`` is now ``12`` (for the whole cleanup wave). A new code, ``curation_conflict``
  (``CurationConflictError``, a ``ValueError``; ``reason``, ``ids``), reports a
  valid curation request that conflicts with the file's review state; its
  reasons are listed in :doc:`machine_contract`. Curation-file syntax errors
  raise ``bad_setting`` with ``path`` ``curation[line <n>].<column>``, and a
  refused field of an ``actions=`` batch ``actions[<i>].<field>`` (including
  an action frame that disagrees with ``frame=``, which was ``"frame"``; a
  frequency with no frame at all on a ``self_calibrated`` file is still
  ``"frame"``). A ``create`` anchor refused inside a batch names the cell or
  field it came from (``curation[line <n>].freqs``, ``actions[<i>].freq_mhz``;
  was ``anchor_mhz``), and ``review_edit``'s implied create names ``add``. A
  frequency that matches no fitted peak is ``not_found`` (kind ``peak``), one
  no live window covers is ``not_found`` (kind ``window``), and unknown
  ``review_undo`` ids are ``not_found`` (kind ``decision``), every id at once;
  a frequency in ``ids`` is the one the caller wrote, in its frame. An ``add``
  that falls outside the window it names is ``curation_conflict``
  (``target_outside_window``). ``import_data`` raises
  ``not_found`` (kind ``file``) for a missing source and ``bad_setting``
  (``path`` ``source``) for one its format's loader refuses. Every one of
  these is still the ``ValueError`` (or ``FileNotFoundError``) it replaced,
  and messages are unchanged apart from a refused action naming its index.
  The frame-mismatch advisory is now judged per action, so a batch's raw
  actions are diagnosed when others in it are calibrated.
* **Machine contract, cleanup wave: status, validation and settings honesty.**

  * ``ComplexFT`` gains the wire field ``invalidated`` (the canonical names of
    the stages the run that produced it discarded; ``[]`` for a display or
    loaded spectrum). A ``PeakList`` still serializes as the plain list.
  * **Behaviour change:** ``api.save_ft_parameters`` returns the canonical
    names, in re-run order, of the stages that saving a changed Stage 1 record
    invalidated (``[]`` when nothing changed), where it returned ``None``. The
    call has no ``events`` argument, so no ``Invalidated`` event is delivered;
    the warning log line names them too. ``visualize_ft(save_params=True)``
    still returns its figure and now names the stages in its log line.
  * **Behaviour change:** ``get_pipeline_info``, ``Pipeline.info``,
    ``list_available_stages``, ``ftmwpipeline info`` and the ``validate_pipeline``
    report name stages by their canonical names (``data``, ``ft``, ``noise``,
    ...) in re-run order, where they listed storage keys such as
    ``stage1_complex_ft`` in alphabetical order. The report's
    ``stage_environments`` keys, drift lines and "Missing data for completed
    stage" errors follow. Code that tested ``"stage1_complex_ft" in
    list_available_stages(path)`` must test ``"ft"``.
  * **Behaviour change:** ``validate_pipeline`` / ``Pipeline.validate`` raise
    the typed open error (``not_found``, ``file_corrupt``, ``file_incompatible``)
    for a file that cannot be opened, and for one that cannot be read while the
    report is built, where they returned ``{"valid": False, ...}``. An
    ``OSError`` from any read while the report is built (the FID included,
    which used to be reported as "Cannot load FID data") raises as opening the
    file would: a permission failure or HDF5 lock refusal as the original
    ``OSError``, any other as ``file_corrupt``. A readable file still gets a
    report of its integrity problems.
  * **Behaviour change:** ``read_metadata``'s ``file.completed_stages`` lists
    canonical stage names (``data``, ``ft``, ...) in re-run order, where it
    listed sorted storage keys such as ``stage1_complex_ft``; a recorded key
    no stage of this version owns is left out, as in the status calls.
  * **Behaviour change:** ``settings set`` / ``settings_set`` refuse a value
    outside the ``choices`` or ``bounds`` a settings row declares
    (``bad_setting`` with the knob as ``path``, the declaration as ``expected``,
    the caller's value as ``value``), leaving the file untouched. Only
    ``stage5.conservative.n_eff_kind`` declares choices today
    (``perplexity_log1p_snr``, ``kish_mag_sq``, ``kish_mag``, ``hard_radius``);
    no setting declares bounds yet. Previously any string was stored and the
    fit failed later.
  * **Behaviour change:** every stage holds its resolved settings to the same
    declared ``choices`` and ``bounds``, whichever layer supplied the value --
    a ``settings=`` object, a preset or a persisted record -- and refuses a
    violation as ``bad_setting`` naming the registry path (for example
    ``stage5.conservative.n_eff_kind``); the file is left as it was. Only the
    winning value is checked; ``settings show`` still displays a bad persisted
    value and ``settings set`` repairs it.
  * **Behaviour change:** ``read_metadata`` reports ``Absent.NOT_RUN`` for a
    count, creation time, plan revision or shape attribute a stage group does
    not carry, where it returned ``0``, ``"unknown"`` or ``"lorentzian"``, and
    for ``stage4.n_dependency_edges`` when the plan carries no
    ``dependency_edges`` record, where it counted zero edges.

* **Machine contract, cleanup wave: degenerate statistics are undefined.**
  A statistic that has no value used to
  be stored as an ordinary number; it is now stored as ``nan`` and reads as
  undefined (``read_table`` status ``2``, value ``nan``). This covers an
  add-loop or knockout F-test with no residual degrees of freedom or a
  non-positive chi-squared (was ``f_statistic`` 0, ``p_value`` 1; a genuine
  non-improvement keeps those numbers), a doublet's orthogonal evidence
  when the weak partner had no usable support (was 0), a residual edge
  coherence of an empty residual or a band with no positive noise (was 0),
  and a Stage 3 SNR without positive local noise (was 0). Older files read
  the same way wherever what they store shows the statistic was degenerate:
  ``fit_audit``, ``fit_doublets``, ``fit_windows`` and ``peaks``. An older
  file's degenerate *knockout* p-value cannot be recognized and keeps its
  ``1.0``. **Behaviour change:** ``fit_windows`` ``edge_coherence_low`` /
  ``_high`` ``nan`` is now status ``2`` for a window the fit evaluated (it
  was always ``1``), and ``peaks`` ``internal_snr`` ``nan`` is ``2`` when the
  internal pass contributed the peak. ``fit_thaw``'s ``edge_coherence_before``
  / ``_after`` stay the values the thaw gate read (an undefined edge is
  ``0.0``), so ``edge_coherence_after`` ``nan`` still means only that the
  joint co-fit produced no usable fit. A window the fit never evaluated keeps
  reading ``1`` after a Stage 6 edit to another window rewrites the fit.
  Every gate decision and fitted number is unchanged; on the 2638 fixture the
  whole fit is bit-identical.
* **Stage 6 refits mask spurs at full precision.** A Stage 6 refit (every
  review verb that refits a window) replays the Stage 5 fit's gated spur
  catalog instead of re-detecting it. It used to replay the spur centers as ``parameters["spur_centers_mhz"]``
  stores them, rounded to 4 decimals for display, while the fit itself and
  the window and spectrum models masked at full precision; it now reads the
  full-precision catalog from the fit's ``diagnostics["gated_spurs"]``, the
  same way the models do. A spur's mask is a whole-bin half-width, so the
  rounding (at most 0.05 kHz) moves a masked bin only when a mask edge falls
  within it of a bin; on the 2638 fixture no window's mask changed and every
  refit is bit-identical. ``ANALYSIS_EPOCH`` is unchanged.
* **Machine contract, Wave 5.2: Stage 5 partial fits.** ``CONTRACT_VERSION``
  is now ``11``. A cancelled (or callback-failed) fit no longer throws its work
  away: the windows that had finished are kept as a partial fit, in the call's
  one atomic write, and ``cancelled.completed_windows`` lists them (as does
  ``callback_failed.completed_windows``, a new field that is ``[]`` everywhere
  else; no window finished: nothing is written, as before). The write's
  invalidations arrive as one ``Invalidated`` before the error is raised.
  **Behaviour change:** that write discards the previous fit and everything
  built on it, where a cancelled fit used to leave the file exactly as it was.
  **Behaviour change:** a bare ``review_accept`` (and a curation file of bare
  ``accept`` rows, applied or previewed) and ``review_undo`` now refuse with
  ``stage_not_run`` when the file holds no complete fit, like every other
  curation call; a bare accept used to record a decision against the Stage 4
  plan. ``status`` reports ``fit`` as
  ``partial``; ``window_status`` reports the kept windows; every fit and
  final-products accessor behaves as before Stage 5. The next ``fit_peaks`` /
  ``fit run`` resumes it, fitting only the remaining windows, and produces the
  fit an uninterrupted run produces; ``restart=True`` / ``--restart`` starts
  over. The ``fit run`` summary gains ``resumed``, ``windows_carried`` and
  ``restart_reason`` (``restart_requested``, ``settings_changed``,
  ``incomplete_provenance``, ``thaw_refit``, or ``null``; the new
  ``restart_reason`` vocabulary). Anything that discards a fit discards a
  partial one (``settings set`` / ``unset``, an upstream re-run, a forced
  re-import; clocks and the timebase leave it). The partial fit is stored
  without pickling: reading a file never runs code from it, and a partial fit
  that cannot be read back is not resumed (``incomplete_provenance``).
* **Machine contract, Wave 5.1b: crash safety.** ``CONTRACT_VERSION`` is now
  ``10``. Every call that writes a ``.ftmw`` file -- a stage run, curation,
  ``settings set`` / ``unset``, ``clocks``, ``start run``, a stamp -- now writes
  atomically: it does all its writes in a temporary copy beside the file
  (``.<file name>.ftmw-tmp.<hostname>.<pid>``) and replaces the original with
  one ``os.replace``. A process killed at any point, ``SIGKILL`` included,
  leaves the file as it was before the call or as the call completed it, never
  a mix and never a file that will not open; a cancel, a ``callback_failed`` or
  any other failure discards the copy, so the file is left exactly as it was
  (byte for byte, where it used to be equal in content). ``StageFinished`` is
  emitted once the replace has happened. Within ``run_pipeline`` each stage is
  its own atomic write, so a kill keeps every stage that finished. A new code,
  ``write_conflict`` (``WriteConflictError``, attribute ``path``, exit ``1``),
  is raised when another process wrote the file after the call's copy was
  taken; the other write stands and this call's changes are discarded. Writes
  from one process to one file are serialized. A copy left behind by a killed
  process is removed by the next write to the file from the same host (never
  another host's copy, and never one whose pid is alive). A reader that already
  has the file open keeps the version it opened, and a pure read (a preview, the
  import of an identical source, ``compute_ft`` from saved parameters) now
  leaves the file untouched. **Behaviour changes:** compaction happens when the
  copy is made, so a file carries the dead space of its last write until the
  next write reclaims it (a stage run no longer ends by repacking, and
  ``run_pipeline`` no longer repacks once at the end); and a failed
  ``review apply --log-prefix N`` row now leaves the file exactly as it was
  before the call, where it used to leave it aligned at the prefix.
* **Machine contract, Wave 5.1: events and cancellation.**
  ``CONTRACT_VERSION`` is now ``9``. Every long operation (each stage run, the
  curation calls, ``report_run``, ``scan_run`` / ``scan_all`` and
  ``run_pipeline``) takes ``events=`` (a callback receiving
  ``StageStarted``, ``StageFinished``, ``WindowProgress``, ``ScanProgress``,
  ``Invalidated`` and ``PipelineWarning`` events, all exported from
  ``ftmwpipeline``) and ``cancel=`` (anything with ``is_set()``) on the API and
  ``Pipeline``; every long CLI verb takes ``--events`` (one JSON line per event
  on stderr), and the first Ctrl-C cancels it (exit ``130``). A cancel raises
  ``cancelled`` and a failing callback ``callback_failed``; a cancelled stage
  or curation batch leaves the file as it was. **Breaking:** ``run_pipeline``'s
  ``error`` is now the failure's ``ftmw/error@1`` dict (it was a string) and
  ``failed_stage`` a canonical stage name; a cancel raises instead of being
  reported in the result. ``Pipeline.create`` now runs the same import as ``data import``,
  and ``import_data``'s ``invalidated`` includes stages a moved start hint
  dropped. The stage start and end lines, the per-window fit lines and the
  invalidation warning are now logged from the events, with the same text.
* **Machine contract, Wave 8: machine-readable CLI output and full
  capabilities.** ``CONTRACT_VERSION`` is now ``8``. Every CLI verb accepts
  ``--json``: stdout is then exactly one JSON document, serialized through
  ``to_jsonable`` (no ``NaN``, no ``default=str``), logs stay on stderr, and a
  typed error is the ``ftmw/error@1`` dict on stderr. Stage-running and curation
  verbs print ``ftmw/run_result@1`` (``verb``, canonical ``stage`` or ``null``,
  ``invalidated``, and a ``summary`` of the scalars the human output reports);
  plot verbs print the paths they wrote; other verbs print their natural
  payload. ``--format json`` stays a synonym where a verb already had it, and
  ``--json`` never changes the format of a file a verb writes. Human output is
  unchanged. ``capabilities()`` now carries every manifest group --
  ``metadata_keys``, ``tables``, ``fields``, ``vocabularies``, ``file_bound``,
  ``pipeline_names`` -- alongside the schemas, accessors, codes and stages, and
  declares ``CurationAction`` and ``SettingRow``.
* **Machine contract, Wave 7: status, stage mappings, no stale results, typed
  settings rows.** ``CONTRACT_VERSION`` is now ``7``. ``status(path)`` (API,
  ``Pipeline``, ``read status``) returns ``ftmw/status@1``: each stage's
  ``complete`` / ``partial`` / ``not_run`` state and dependencies, the stages
  runnable now, and the full re-run order. ``capabilities()["stages"]`` maps
  every canonical stage to its storage key, settings prefix and knob prefix
  (``tau`` and ``tau_g`` share ``stage2b``). Every stage-running call reports
  the stages it invalidated (``invalidated``, canonical names in re-run order;
  the CLI prints them), and the no-stale rule now holds for every write:
  ``clocks set`` / ``clear`` rebuild the final-products table when they move
  the calibration state, a start stamp that moves a pre-provenance Stage 1
  record's spectrum invalidates what was built on it, and
  ``compute_ft(..., from_saved_params=True)`` no longer writes the file.
  ``settings_set`` / ``unset``'s ``invalidated`` now holds canonical names, not
  storage keys. ``settings_show`` / ``settings_defaults`` rows gain ``type``,
  ``nullable``, ``units``, ``choices`` and ``bounds`` (``null`` where the knob
  metadata states none), and their values round-trip as typed JSON. Every verb
  that takes a ``.ftmw`` now refuses a path that is not a readable pipeline file
  with ``file_corrupt`` (exit 2), and a missing one with ``not_found``.
* **Machine contract, Wave 6: curation as data.** ``CONTRACT_VERSION`` is now
  ``6``. ``CurationAction`` (exported from ``ftmwpipeline``) is one curation-file
  row as a typed, JSON-able value (``ftmw/curation_action@1``): ``action``,
  ``window_id``, ``freq_mhz``, ``peak_uid``, ``candidate_mhz``, ``frame`` and,
  for a calibrated frequency, an optional ``epsilon`` stamp. ``review_apply`` /
  ``review_preview`` (API, ``Pipeline``, review session) take ``actions=`` in
  place of ``curation_path`` (exactly one), and the CLI takes ``--actions
  FILE`` (a JSON array; ``-`` for stdin). The same plan as a file and as data
  gives equal results, decision logs and files, with the same validation,
  frame handling and drift check. Each action resolves its own frame
  (``None`` takes the call's ``frame=``), so a batch may mix raw and
  calibrated frequencies; an explicit action frame that disagrees with an
  explicit ``frame=`` is refused.
* **Machine contract, Wave 4: typed refusals (breaking).** Refusals a program
  can act on now raise the typed family with a stable ``code``, still as a
  subclass of the built-in they replaced, so existing ``except`` clauses keep
  working. ``bad_setting`` (``BadSettingError``: ``path``, ``expected``,
  ``value``) covers ``settings set`` / ``unset``, presets, clock declarations,
  shapes, and the value checks of every stage call (``compute_ft`` start/end/
  trim, noise, tau, timebase, peaks, fit, ``show_fit``, report and read
  choices, curation arguments); ``path`` is the registry path of a registry
  setting (``stage1.trim``, ``stage5.shape``, ``stage5.spur.clocks``) and the
  argument name otherwise. An unknown or undetectable source format is
  ``bad_setting`` with ``path`` ``"format"`` on import and ``preview_source``
  alike (``preview_source`` raised ``not_found`` before). ``not_found`` now
  lists every unknown window or peak uid of a curation batch at once, and is
  ``NotFoundValueError`` (also a ``ValueError``) where the call refused with
  ``ValueError`` before; typed errors inside a curation batch propagate with
  their own type instead of a flattened ``ValueError``. Reading the noise
  result before ``noise run`` is ``stage_not_run``. ``validate_pipeline`` now
  raises the typed open error for a file it cannot open, as ``Pipeline.open``
  does, instead of returning ``{"valid": False}``. Every CLI verb lets typed
  errors reach one dispatch in ``main``, which applies the documented exit
  codes and writes the error dict to stderr under ``--format json``.
  ``algorithm_failed`` is declared and reserved (not raised yet).
* **Machine contract, Wave 3: absence is ``Absent`` everywhere on the
  contract (breaking).** Pre-contract fields that encoded "no value" as
  ``None``, ``nan``, ``-1``, ``""`` or a plausible-looking ``0.0`` now carry
  ``Absent.NOT_RUN`` (never computed in this file) or ``Absent.UNDEFINED``
  (computed, no value); on the wire they are ``null`` plus a
  ``"<field>_absent"`` sibling, never a bare ``null``. Storage is unchanged:
  status is derived at read time through one set of rules, so a quantity
  reports the same status on every surface (``FinalPeak`` and ``fit_peaks``,
  preview and edit results). Affected: ``FinalPeak`` (phase, snr, the
  uncertainties, window id, peak uid, derivation, clock lattice, the knockout
  fields), ``CalibrationStamp`` (probe frequency, sideband), the curation
  results (``chi2r_before`` / ``chi2r_after``, ``converged``, the created-window
  fields), ``read_metadata`` values present without a value (an unrun stage's
  keys are still omitted), ``get_pipeline_info``'s environment fields, typed
  error ``command``, ``read_tables`` ``n_rows``, and a ``uint8``
  ``<column>__status`` companion on the ``fit_peaks``, ``fit_windows``,
  ``fit_audit``, ``fit_doublets`` and ``peaks`` columns that can be absent.
  ``Absent`` is truthy: test it with ``isinstance``, never ``if x``. Two
  values change: a line with no statistical frequency error now reports
  ``sigma_stat_khz`` and ``sigma_f_khz`` as undefined instead of a total that
  silently dropped the statistical term, and a ``fit_audit`` separation
  reject's placeholder ``f_statistic`` 0.0 / ``p_value`` 1.0 (no test ran) read
  as ``nan`` with status not run. Lines from a Stage 6 refit are now annotated
  against the fit's clock lattice, as Stage 5's lines are. See
  :doc:`machine_contract`.
* **Machine contract: Stage 1's stored settings are authoritative.** Stage 1
  now records the concrete values it ran with in
  ``processing_parameters/ft_processing`` (field-set version 2): ``start_us``
  is ``0.0`` when there was no windowing, ``end_us`` is the FID duration when
  unset, and an unset ``trim`` means "no trim". Every reader -- Stages 2-5,
  Stage 2b, the shape recommendation, the timebase, the report and the
  ``read_metadata`` ``ft.start_us`` / ``ft.end_us`` keys (which now report
  ``0.0`` and the duration, not ``None``) -- resolves them through one
  resolver and never falls through to the recommended layer. A later
  ``start run`` or chirp-window stamp still updates the recommendation for
  display but changes nothing Stage 1 is read as having used; ``start run``
  now warns that adopting the new start takes ``ft run --start-us``. A fresh
  run is bit-identical. Old files keep today's behaviour, except that Stage 2b,
  the shape recommendation and the timebase now agree with Stages 2-5 when an
  old record has an unset window or trim and a recommendation was stamped
  afterwards (they previously read ``0.0`` / the duration).
* **Machine contract, Wave 2b: one Stage 2b settings record per producer.**
  The Lorentzian calibration, the Gaussian calibration and the shape
  recommendation each persist their own resolved settings record
  (``stage2b_tau_calibration``, ``stage2b_tau_G_calibration``,
  ``stage2b_shape_recommendation``), stamped with a field-set version, so each
  says what its own result used and none overwrites another. The shared
  ``stage2b_tau`` recipe is unchanged: it is still what ``settings show stage2b``
  reports and what the next run resolves against. The recommendation's record
  also holds the Stage 1 values it ran on, its effective decay-time clip and its
  verdict, and is written even when no calibration exists yet; a fresh
  calibration withdraws it along with the verdict. ``stage2b.stft.tau_max_factor``
  now reaches the classifier through all three producers (it was previously
  ignored), and the shape recommendation now also honours
  ``stft.relative_gate_fraction`` and ``stft.sigma_x_full`` (it ran at the
  kernel defaults). Every default equals the value used before, so a fresh run
  is bit-identical; only a file whose persisted recipe or preset sets one of
  these knobs away from its default changes when re-run. Files written earlier have no producer
  records and read as such; nothing is invented for them. See :doc:`stage2b_tau`.
* **Machine contract: stages record the upstream values they used.** Stage 3
  records the gap-pass decay time and shape it took from Stage 2b, and Stage 5
  the decay-time anchor source (with the per-band majorities table when it
  routed per band), the timebase ε it used for the spur window, the Stage 2b
  spur clusters its spur gate took as nominees, and its
  effective peak-survival floor, each under ``consumed`` in its own settings
  record. The timebase record now persists its resolved clock declaration as a
  typed list, and the Stage 6 final-products table records the declaration its
  calibration state was derived from (see :ref:`consumed-values`). Nothing is
  invalidated by this; a fresh run is bit-identical to before. Two behaviours
  change on particular files only. (1) ``tau.fit_tau`` now persists as
  ``True``, and a Stage 6 refit keeps a window's decay time fixed where the fit
  held it fixed. Stage 5 always read an explicit ``True`` the same as unset (free
  where the SNR gate frees it), but a refit used to free every window's decay
  time when the record held an explicit ``True``; it now follows the fit.
  (2) ``review run`` adds ``/frequency_calibration`` to a file that lacks it.
  Files written earlier
  open and run unchanged; their new records read as absent.

* **Machine contract: the analysis fingerprint.** ``CONTRACT_VERSION`` is now
  ``5``. ``analysis_fingerprint`` (API, ``Pipeline`` and ``read
  analysis_fingerprint``) returns ``ftmw/analysis_fingerprint@1``, a SHA-256
  digest of every input that shaped the file's results as the stages recorded
  them: their settings at the values they ran with, the values they took from
  other stages, the acquisition parameters and stored acquisition segments, the
  accuracy floor, the clock declaration the calibration state is derived from,
  and each stage's analysis epoch. The same digest means the same results. A file whose stages
  predate recording everything they used raises ``incomplete_provenance``
  listing every missing input; re-running those stages records them (see
  :doc:`machine_contract`).

* **Machine contract, Wave 2: the fitted model, evaluated.**
  ``CONTRACT_VERSION`` is now ``4``. ``window_model`` returns one window's
  fitted model with the data, the frozen-neighbour and baseline terms, and --
  on the active grid Stage 5 fits -- the noise and spur mask the fit used, so
  the fit's chi-squared is reproducible from the payload alone.
  ``spectrum_model`` returns the whole fitted spectrum, its data and residual.
  Both evaluate on the active grid or on ``compute_display_ft``'s grid, through
  the same evaluator the fit plots use (see :doc:`machine_contract`).

* **``compute_display_ft``'s band is the active grid's own.** It now runs from
  the first to the last active-FT bin inside the Stage 1 trim. Its edges used
  to be cut against ``compute_ft``'s band with a quarter-bin tolerance, which
  admitted one padded bin below the first active bin on the 2638 fixture
  (343,710 bins, now 343,709 = 2 x 171,855 - 1). Its docstrings no longer
  claim it differs from ``compute_ft`` only in bin density: it is the
  active-portion FT, at a different spacing, scale and phase origin.

* **The Stage 5 fit plots draw the fitted model.** ``visualize_fit``'s overview and
  per-window figures carried their own model, which ignored the line shape and
  the baseline; both now use the shared evaluator. On a window fitted with a
  Gaussian shape and a baseline the drawn residual now matches the fit (2638
  window 179: chi-squared 45213.95 before, 23023.91 now, equal to the fit's).

* **Machine contract, Wave 2: per-line fit fields on the final products.**
  ``CONTRACT_VERSION`` is now ``3``.
  Each ``FinalPeak`` now carries, from the Stage 5 fit of its window,
  ``decay_time_us``, ``decay_time_error_us``, ``shape``, ``fwhm_mhz``,
  ``detection_index`` and ``fit_window_mhz`` (see :doc:`machine_contract`).
  ``fwhm_mhz`` is ``feature_fwhm(tau, stage5.acquisition_us, shape)`` exactly,
  and ``fit_window_mhz`` is in the calibrated frame like ``frequency_mhz``.
  They are ``Absent`` when they have no value (an error for a ``tau`` held
  fixed, a line no Stage 3 detection seeded, a line with no fit record) and
  are stored losslessly. A final-products table written before them is
  rebuilt in memory on read, without writing the file. ``report table``
  (CSV and JSON) and the HTML report's line tables gain the matching columns,
  after the existing ones; an absent field is left empty, as missing values
  already are. ``ftmwpipeline.Absent`` is unchanged (it is now defined in a
  core module and re-exported, so the data structures can use it).

* **Machine contract, Wave 1: the read surface, one rule for every accessor.**
  ``CONTRACT_VERSION`` is now ``2``.
  The sixteen accessors added to ``MANIFEST`` after ``capabilities`` share one set of
  rules (see :doc:`machine_contract`).

  * *One CLI verb per accessor:* ``ftmwpipeline read <name>``, spelled exactly
    as the API name (``read frequency_calibration``, ``read get_pipeline_info``,
    ``read read_table FILE TABLE --columns a,b``, ``read settings_show``, ...).
    It prints a schema-stamped JSON envelope: a dict or dataclass result is
    stamped directly, a list or tuple is ``{"schema", "items": [...]}``, a
    scalar is ``{"schema", "value": x}``, and ``read get_final_products``
    before Stage 6 prints ``"value": null`` with ``"value_absent": "not_run"``
    (the Python result is still ``None``). The human verbs (``info``, ``review
    log``, ``settings show``, ``timebase state``, ``report table``, ``read
    table`` / ``meta`` / ``list``) are unchanged and are not contract.
  * *The schema is stamped in Python as well:* ``fid_samples``,
    ``display_units``, ``fit_thresholds``, ``window_status``,
    ``preview_source`` and ``capabilities`` return dicts with a ``"schema"``
    key; ``CalibrationStamp`` and ``FinalProducts`` declare
    ``__ftmw_schema__``. The dicts of ``read_metadata``, ``read_tables``,
    ``read_table`` and ``get_pipeline_info`` keep their Python shape (no
    ``"schema"`` key) and are stamped by the verb. Schema names are constants
    in ``ftmwpipeline.contract``.
  * *Entity tables are lists of records:* ``window_status`` returns
    ``{"schema", "windows": [WindowStatusRow, ...]}`` and ``preview_source``
    a list of ``FidPreviewRow`` under ``fids``, so an absent field travels per
    row (``null`` plus ``"<field>_absent"``). The columnar form, with
    ``__status`` columns, is the ``window_status`` table of ``read_table``,
    built from the same rows.
  * *Refusals name a bare verb:* ``fid_samples`` refuses with ``data import``,
    and ``read_table`` now raises ``StageDependencyError`` (still a
    ``ValueError``) for every table whose stage has not run, with
    ``command`` ``tau run``, ``tau run --gaussian``, ``peaks run``,
    ``windows run`` or ``fit run``.
  * ``get_pipeline_info`` now always carries ``warnings`` (an empty list when
    there are none), and ``to_jsonable`` serializes ``ComplexFT``. The declared
    elements also include the ``read_metadata`` keys, the ``fit_peaks`` /
    ``windows`` columns, the fields of ``FinalPeak``, ``DecisionLogEntry`` and
    ``ComplexFT``, and the decision-log ``kind`` and ``provenance``
    vocabularies.

  New in Wave 1:

  * ``fid_samples(path)``: the stored Stage 0 samples as a 1-D ``float64``
    array (values equal the stored ones, promoted losslessly), from a single
    dataset read; samples go to ``samples.npy`` under ``--output``.
  * ``display_units(path)``: ``amplitude_scale``, ``units_label`` and
    ``units_power``. The scale and label are exactly the pair
    ``compute_display_ft`` applies, at every stage including before Stage 1;
    ``units_power`` is always an ``int``.
  * ``fit_thresholds(path)``: the ``peak_survival_snr_floor`` and
    ``vif_collapse_threshold`` the persisted Stage 5 fit applied, read from its
    recorded diagnostics; each is ``Absent.NOT_RUN`` when there is no fit or
    the fit never recorded it, never a guessed default. Separately, the figures
    that grade old fits (``fit show`` detail, the rescue summary) no longer use
    a fixed VIF threshold of ``4.0`` or a fixed floor: for a fit that did not
    record a threshold they use the file's own resolved settings (the persisted
    Stage 3 promotion cutoff times the survival factor, or an explicit floor
    override; the resolved ``vif_collapse_threshold``). This is a display
    reference only; the accessor never reports it.
  * ``window_status(path)``: one row per Stage 4 plan window and per Stage 6
    created window, with ``window_id``, the bounds, ``created``,
    ``n_fitted_peaks`` and ``live`` (a Stage 5 line is held in the window).
    Before Stage 5 the two fit-derived fields are ``Absent.NOT_RUN``; before
    Stage 4 it raises ``StageDependencyError`` naming ``windows run``.
  * ``preview_source(source)``: the format and the FID table of a data source
    without importing it (``validate_source`` is unchanged), plus the declared
    chirp window. Each ``FidPreviewRow`` holds ``index``, ``n_points``,
    ``spacing_us``, ``probe_freq_mhz``, ``sideband`` (``"upper"`` /
    ``"lower"``), ``shots`` and ``channel``. Only source-declared values are
    reported: a field the source does not declare is *not run*, never import's
    default; a Keysight file reports one row per ``Channel_*`` group, whose
    ``channel`` is the value import's ``--channel`` takes; a declared chirp
    window the code cannot read is *undefined*.

* **The machine contract has its foundations: a contract version, ``Absent``,
  typed errors with codes, a JSON serializer, and ``capabilities``.**
  ``ftmwpipeline.CONTRACT_VERSION`` (``1``, the first published contract) is the
  integer a program gates on. ``ftmwpipeline.Absent`` (``NOT_RUN`` /
  ``UNDEFINED``) is the one vocabulary for "no value", written on the wire as
  ``null`` plus a ``"<field>_absent"`` sibling (also for a non-finite float)
  and in arrays as a ``uint8`` ``<column>__status`` column.
  ``ftmwpipeline.Stage`` is the canonical stage vocabulary (``data``, ``ft``,
  ... ``review``) every payload uses to name a stage. Every error a program may
  route on is a ``PipelineFileError`` with a stable ``code`` and ``to_dict()``,
  and every one pickles; three new members join the family:
  ``NotFoundError`` (also a ``KeyError``), ``PipelineFileNotFoundError`` (also
  a ``FileNotFoundError``, raised for a ``.ftmw`` path that does not exist) and
  ``IncompleteProvenanceError`` (also a ``ValueError``).
  ``PipelineCorruptionError`` is now also a ``RuntimeError`` and an
  ``OSError``; it replaces the bare ``RuntimeError`` ``Pipeline.open`` raised
  and the h5py ``OSError`` the read path let escape for an unopenable file,
  message unchanged. Permission and HDF5 file-lock failures still propagate
  as the original ``OSError``.
  No existing exception lost a built-in base or changed its message.
  ``ftmwpipeline.to_jsonable`` converts results to strict JSON, and
  ``capabilities()`` (``api.capabilities``, ``Pipeline.capabilities``,
  ``ftmwpipeline read capabilities``) reports what this installation declares.
  The error code set is introduced wave by wave: ``capabilities()`` lists the
  codes implemented today; the ``read`` accessors (and ``read table`` /
  ``meta`` under ``--format json``) emit error JSON.
  Under those verbs a contract error is printed as its ``to_dict()`` JSON on
  stderr, with exit code 2 for ``file_corrupt``, 130 for an interrupt, and 1
  for every other code; ``read table`` / ``meta`` / ``list`` use the same
  mapping (a file that exists but is not HDF5 now raises
  ``PipelineCorruptionError`` from ``api.read_table`` / ``read_metadata``, where
  h5py's ``OSError`` escaped before, and exits 2 from the verbs). Array results are
  written as ``.npy`` files into the directory named by ``--output``. An enum
  is serialized as its value, and a complex number as ``{"real", "imag"}``.
  See :doc:`machine_contract`.

* **``review apply --log-prefix N`` (``log_prefix=N`` on ``Pipeline.review_apply``,
  ``api.review_apply`` and ``ReviewSession.review_apply``) applies a curation
  file as if the decision log ended after its first ``N`` decisions.** The
  later decisions are dropped, the automatic fit is restored, and the kept
  decisions are replayed together with the file as one batch -- one cascade,
  one persist -- so the outcome is that of ``review undo`` of the dropped ids
  followed by ``review apply``, in one replay instead of two. The file's
  frequencies and omitted window ids (``uid:N`` targets included) resolve
  against the in-memory state the kept decisions leave, not the file as it
  stood before the restore or the bare baseline after it; a failing row leaves
  the file aligned at the prefix. ``N`` equal to the log's length is the
  ordinary apply; a shorter ``N`` refuses ``--dry-run``. Built for a client
  that keeps its own position in the log and aligns the file only on its next
  real edit.

* **A ``.ftmw`` no longer grows with every curation write.** HDF5 never
  reclaims the space a rewritten attribute or a deleted variable-length
  dataset leaves behind, which is what a stage re-run and every Stage 6 write
  churn: measured on a 6.5 MB build, each single-window edit added 0.7 MB and
  each undo 1.2 MB, permanently, until a curated file was three times its
  content. (Free-space tracking at file creation was measured and does not
  help -- that space is on no free list.) Every stage run and every curation
  write now ends by repacking the file in place (an object-by-object copy that
  atomically replaces the original, tens of milliseconds at this size);
  ``ftmwpipeline run`` repacks once at the end. Existing files shrink on their
  next stage run or curation write. There is no verb and nothing to configure.

* **Stage 6 ``review run`` now logs like the other stages, and a slow Stage 5
  window is its own log record.** ``review run`` emits a start line
  (``Stage 6 review: routing attention for %d windows in %s``) and a
  completion summary (``Saved Stage 6 review to %s: %d windows, %d need
  attention``) on the ``ftmwpipeline`` logger, where it used to emit nothing.
  Stage 5's per-window detail line (``w%d [%.1f-%.1f MHz]: %d peaks,
  chi2r=%.3g, %.1fs``) is now always INFO; a window over the slow threshold
  (``plan_execution.SLOW_WINDOW_WARNING_S``, 60 s) ADDITIONALLY logs a WARNING
  under its own template (``slow window w%d [%.1f-%.1f MHz]: %.1fs (over
  %.0fs)``), so a consumer filtering by level and one matching on the message
  template agree on which records mean "slow" -- previously the one detail
  template was promoted to WARNING, which neither could tell apart.

* **Fixed: an auto-merged line's frequency uncertainty no longer collapses when
  its window is curated.** Stage 5 widens a merged multiplet's
  ``frequency_error`` to ``sqrt(formal**2 + spread**2)``, but recorded the
  spread only in ``FittedPeak.extra_errors``, which is not serialized -- so
  the first Stage 6 refit of that window recomputed the error from its own
  covariance and the widening vanished. On the 2638 reference file the six
  auto-merged lines carry a spread 13x to 131x their formal error, so the
  reported ``sigma_f`` on a blended line dropped by one to two orders of
  magnitude, claiming a precision the data do not support. It did not take an
  edit to the line: a cascade refit triggered by an edit in a *neighboring*
  window did it too, invisibly.

  ``FittedPeak`` gains ``unresolved_spread_mhz``, persisted as a peaks-table
  column (absent or NaN -> ``None``, i.e. not a merged multiplet), and every
  Stage 6 refit re-applies the widening to the formal error it just computed.
  ``frequency_error`` still stores the widened value, so no consumer has to
  add the term in. The spread travels with the line and is dropped with it: a
  removed line takes it along, and added or split-product lines carry none.
  A fit written before the column recovers its spreads at the curation door
  from the window-level ``vif_collapse`` diagnostics, which were always
  persisted -- matched by nearest frequency within the file's own snap
  tolerance, skipping any line a Stage 6 decision has altered. That recovery
  seeds the field only; it never rewrites a stored error, since on an
  uncurated legacy file the stored value already carries the widening and
  re-applying it would count the spread twice.

* **The live apply reports its per-window outcome.**
  ``CurationApplyResult`` gains ``windows``, keyed by window id: the counts,
  the chi2r pair, convergence, the ``direct``/``cascaded`` origin and the
  action indices, for every window the plan touched or the cascade reached
  (:class:`AppliedWindowResult`). Previously only ``review preview`` carried
  that block, so a caller wanting the counts off a live apply had to re-read
  the file or the decision log's per-decision evidence. Built from the
  batch's own finished in-memory fit -- no extra read and no extra fit -- and
  narrowed to what an apply can answer for free: the preview's ``peaks`` are
  not repeated (an apply persists the final-products table), and installed
  structure stays on ``created_windows``, which a dry run fills too. Empty on
  a dry run and on a bare-``accept`` plan, both of which fit nothing. A
  session's staged apply reports the staged preview's own numbers rather than
  a second derivation of them. ``review apply`` prints the block in the shape
  ``review preview`` already prints it.

* **Doc fix:** the Stage 6 page said the per-line ``qual`` determinacy score
  "also serves as a ``review rank --by`` key". It does not -- ``RANK_METRICS``
  is per-window and ``qual`` is per-line -- and the page now says so.

* **Per-peak significance now crosses the curation-result boundary.**
  ``FinalPeak`` gains ``knockout_p_value``, ``knockout_supported`` and
  ``knockout_aicc_delta``, copied from the Stage 5 fitted peak's knockout
  result when the final-products table is built. The knockout test already ran
  on every curation refit and was already persisted per peak, but was dropped
  at the ``FinalPeak`` boundary, so a caller holding a ``review_preview``
  result -- which by design persists nothing -- had no significance statistics
  and no file to recover them from. Additive and backward-compatible: all
  three are ``None`` for a peak with no knockout result and for a table
  written before this existed, and ``nan`` (a float, distinct from ``None``)
  when the test ran but its refit did not converge. The exported table's
  columns are unchanged.

* **A failed refit is now visible on the curation results.**
  ``RefitWindowResult`` and ``PreviewWindowResult`` gain ``converged``,
  mirroring ``FittingResult.success``. A joint fit that fails returns the
  window's seeds verbatim with an infinite chi-squared, which was previously
  reachable only as an implausible ``chi2r_after`` -- convergence was the one
  outcome dimension the curation door did not report. ``review edit``,
  ``review accept`` and ``review preview`` print a warning line for such a
  window. ``PreviewWindowResult.converged`` is read off the same post-cascade
  in-memory fit ``chi2r_after`` is, and is ``None`` on a window with no fit on
  the after side, exactly where ``chi2r_after`` is; ``RefitWindowResult.converged``
  is a plain ``bool``, since a refit that returns has a fit either way.

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

* **``ReviewPreviewResult`` gains ``created_windows``.** ``review preview``
  computed the list of windows its plan installs or grows and then dropped it,
  so only ``review apply`` (dry run or live) published it. It is now on the
  preview result too, carrying exactly what ``CurationApplyResult.created_windows``
  carries -- window id, anchor, mode, extent, grid point count, contributor
  count, ``depends_on`` -- for a plan run to completion in memory. Additive
  field; nothing else changes. The per-window ``created_window_*`` fields on
  ``PreviewWindowResult`` are unaffected and remain the right read when you
  have a window id in hand; what they cannot carry is the **anchor**, which is
  a property of the row that implied the create rather than of the window it
  landed in -- and with coalescing one created window can hold two ``add``
  rows, so extent containment marks both while the anchor marks the one that
  caused it. A session staging a preview as an apply now reads the same list
  rather than a second copy of it.

* **Two omitted-window ``add`` rows in one gap, within a single curation
  plan, no longer refuse each other.** Each row resolves against *live*
  windows only, so two adds in one window-free gap used to each imply their
  own create; the second create's anchor then fell inside the first create's
  just-minted window, and the whole plan refused -- dry run and live apply
  alike -- even though the identical edits applied sequentially (two separate
  ``review edit`` calls) already succeeded. Now, before a fresh implied
  create mints anything, it is checked against the batch's own state so far
  (this batch's earlier creates included); a hit coalesces it away entirely
  -- no second window, nothing installed -- and its paired ``add`` applies as
  an ordinary edit into the window the earlier create already built, which is
  exactly the log shape (one ``add`` carrying ``created_window`` evidence,
  the other an ordinary ``add``) the sequential path already produced. This
  is not a search: the coalescing check reuses the window planner's own
  live-window-coverage predicate -- the same one ``plan_stage6_window``
  refuses on, extracted so there is one copy of the rule -- asked one step
  earlier, with no tolerance, nearest-peak search, or radius of its own. Only a FRESH
  implied create coalesces -- a *replayed* implied create (undo/redo
  reconstructing a decision-log entry) and an *explicit* ``create`` row both
  keep the original refusal, since both name a specific window rather than
  merely deriving one.

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
  from the automatic baseline, one user action at a time (the entries one multi-line
  edit logged replay jointly; see the Unreleased action-group entry) against the
  state the previous ones left, so the identifiers afterward are exactly those that
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
