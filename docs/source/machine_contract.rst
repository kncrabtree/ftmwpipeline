.. index::
   single: machine contract
   single: CONTRACT_VERSION
   single: Absent
   single: capabilities
   single: error codes
   single: exit codes
   single: JSON output

The machine contract
====================

A script, notebook helper or front end sometimes needs more than a human does:
a guarantee that a name, a field or a code will still be there next release,
and a way to branch on a condition without reading text. The **machine
contract** is that guarantee. It is the same surface interactive users already
have, exposed identically through the Python API, the ``Pipeline`` class and the
command line, with the extra promises a program needs. Anything not listed in
the contract -- log wording, exception message text, the layout of the
``.ftmw`` file, anything under ``ftmwpipeline._internal`` -- may change without
notice.

.. note::

   Before ``1.0.0`` the contract may still change incompatibly; each release
   that changes it raises the contract version.

This page is a reference. It covers, in order: the version and the
capabilities manifest; the rules every accessor follows; missing values; stage
names; typed errors and exit codes; machine-readable command-line output; the
read accessors; and what the calls that write a file guarantee.

.. _contract-version:

The contract version
--------------------

``ftmwpipeline.CONTRACT_VERSION`` is a single integer. Gate on it, never on
``__version__``::

    import ftmwpipeline
    if ftmwpipeline.CONTRACT_VERSION < 16:
        raise RuntimeError("needs a newer ftmwpipeline")

The first published contract is version ``1``; this release is version ``16``.
An addition (a new accessor, field or code) raises the version by one and never
breaks an existing field. Every machine-readable payload also carries a
**schema name** of the form ``ftmw/<payload>@<n>``; a schema name never changes
meaning.

.. _contract-capabilities:

Discovering what is offered: ``capabilities``
---------------------------------------------

``capabilities()`` returns the version and everything the installation
declares. It needs no file and is the same on every interface:

.. code-block:: python

   import ftmwpipeline.api as ftmw
   from ftmwpipeline import Pipeline

   ftmw.capabilities()          # functional API
   Pipeline.capabilities()      # Pipeline class (static)

.. code-block:: console

   $ ftmwpipeline read capabilities | python -m json.tool

The payload is ``{"schema": "ftmw/capabilities@1", "contract_version": int,
"schemas": [...], "accessors": [...], "codes": [...], "stages": [...],
"metadata_keys": [...], "tables": {name: [columns]}, "fields": {type: [fields]},
"vocabularies": {name: [values]}, "file_bound": {accessor: bool},
"pipeline_names": {accessor: name}}``. Every group of the manifest
(``ftmwpipeline.contract.MANIFEST``) is present, in manifest order, so a client
can discover the whole surface without importing the package:

* ``accessors`` -- every read accessor (:ref:`contract-accessors`);
  ``file_bound`` says whether each takes a file, and ``pipeline_names`` names
  the ``Pipeline`` method serving each one (every accessor is listed,
  defaulting to its own name: ``get_final_products`` is
  ``Pipeline.final_products``, ``get_pipeline_info`` is ``Pipeline.info``).
* ``codes`` -- the error codes the installation declares
  (:ref:`contract-errors`).
* ``stages`` -- one record per stage (:ref:`contract-stage-names`).
* ``metadata_keys`` and ``tables`` -- the declared ``read_metadata`` keys and
  ``read_table`` columns.
* ``fields`` -- the declared fields of the result types (``FinalPeak``,
  ``DecisionLogEntry``, ``AttentionReason``, ``PipelineInfo``, ``ComplexFT``
  and its ``metadata``, the ``converged`` flag of the curation results, ...).
  An event's fields are its wire keys; a ``PipelineWarning``'s own fields are
  listed per code under ``PipelineWarning.<code>``, such as
  ``PipelineWarning.slow_window``.
* ``vocabularies`` -- the closed value sets: ``decision_kind`` and
  ``decision_provenance`` (the ``DecisionLogEntry`` ``kind`` and
  ``provenance``), ``stage_state``, ``warning_code``, ``restart_reason`` and
  ``attention_kind`` (:ref:`contract-attention`).

.. _contract-rules:

Rules every accessor follows
----------------------------

* **One CLI verb per accessor:** ``ftmwpipeline read <name>``, spelled exactly
  as the API name (``read window_status``, ``read read_table``, ``read
  get_pipeline_info``). It prints the accessor's JSON envelope. An accessor
  that reads a file takes the path as its first argument on the API and as the
  ``read`` verb's file argument, and is an instance method of an opened
  ``Pipeline``; one that needs no file (like ``capabilities``) takes no path
  anywhere. The human-facing verbs (``info``, ``review log``, ``settings
  show``, ``timebase state``, ``report table``, ``read table`` / ``meta`` /
  ``list``) print output that is not contract.
* **The envelope has one of three forms**, by the kind of result:

  - a dict or dataclass is *stamped directly*: ``{"schema": ..., <fields>}``;
  - a list or tuple is *wrapped as items*: ``{"schema": ..., "items": [...]}``
    (``settings_show``, ``settings_defaults``, ``review_log``);
  - a scalar is *wrapped as value*: ``{"schema": ..., "value": x}``
    (``refit_snap_tol_mhz``).

  An absent result is an ``Absent`` in a named field, like any other absence:
  ``get_final_products`` returns ``None`` in Python before Stage 6 and prints
  ``{"schema": "ftmw/final_products@1", "value": null, "value_absent":
  "not_run"}``.
* **The payload carries its schema in Python too**, wherever the Python type
  can: a dict payload has a ``"schema"`` key and a dataclass payload declares
  ``__ftmw_schema__``, so the API, ``Pipeline`` and the CLI return the same
  stamped object. The plain dicts of ``read_metadata``, ``read_tables``,
  ``read_table`` and ``get_pipeline_info`` (and ``ComplexFT``) are stamped by
  the verb only.
* **Entity tables are lists of records**, not parallel columns: an accessor
  that returns one row per window, FID or line returns a list of dataclasses or
  dicts, so an absent field travels as ``Absent`` per row (``null`` plus its
  ``_absent`` sibling). Numeric series (samples, spectra) are arrays and go to
  ``.npy`` files under ``--output``. The columnar form of a table, with
  ``<column>__status`` columns, belongs to ``read_table``.
* **A refusal names the command to run as a bare CLI verb** (``windows run``,
  ``data import``, ``tau run --gaussian``), never a full shell line.
* **Serialization.** An enum is written as its ``.value``
  (``PeakShape.LORENTZIAN`` is ``"lorentzian"``, also as a mapping key); a
  complex number is ``{"real": x, "imag": y}``; a non-finite float with no field
  or list to hold it (the top level) is an error rather than a silent ``null``.
  Python code produces the same JSON with :func:`ftmwpipeline.to_jsonable`,
  which applies the ``Absent`` rule (:ref:`contract-absent`), stamps a
  ``schema=`` name, and can hand arrays to a sink instead of inlining them.

.. _contract-absent:

Missing values: ``Absent``
--------------------------

"No value" has two distinct meanings, and the contract never confuses them with
``None``, ``nan``, ``-1`` or an empty string:

.. list-table::
   :header-rows: 1
   :widths: 20 20 30 30

   * - Meaning
     - Python
     - JSON
     - Array column
   * - present
     - the value
     - the value
     - value, status ``0``
   * - **not run** (never computed or tested in this file)
     - ``Absent.NOT_RUN``
     - ``null`` plus ``"<field>_absent": "not_run"``
     - ``nan`` (or the column's fill), status ``1``
   * - **undefined** (computed, but has no value)
     - ``Absent.UNDEFINED``
     - ``null`` plus ``"<field>_absent": "undefined"``
     - ``nan``, status ``2``

Compare with ``is``: ``x is Absent.NOT_RUN``. ``Absent`` is truthy, so test a
flag that may be absent with ``is False``, never ``not flag``. On the wire a
field keeps one type (its value type or ``null``); the sibling key appears only
when the value is absent, and a key ending in ``_absent`` is never anything
else. JSON has no ``nan`` or ``inf``: a field whose value is not finite is
written as ``null`` with ``"<field>_absent": "undefined"``, and a non-finite
element of an inline JSON list is ``null``. Arrays written as ``.npy`` files
keep their ``nan`` values. Array columns that can be absent come with a
``uint8`` column named ``<column>__status``. A setting that is merely unset
reads as ``None``; that is not an absent value.

.. _contract-stage-names:

Stage names
-----------

Every contract payload that names a stage uses the canonical vocabulary
``ftmwpipeline.Stage``, whose values are the CLI object names: ``data``,
``ft``, ``noise``, ``tau``, ``tau_g``, ``timebase``, ``peaks``, ``windows``,
``fit``, ``review``. ``ftmwpipeline.contract.stage_for_key`` and
``key_for_stage`` map to and from the internal storage keys (for example
``stage1_complex_ft`` is ``ft``). The Python attribute
``StageDependencyError.missing_dependencies`` holds the internal keys; its
``to_dict()`` publishes the canonical names.

The status calls use the same vocabulary, in re-run order: the
``completed_stages`` and ``next_available_stages`` of ``get_pipeline_info`` /
``Pipeline.info`` (and ``ftmwpipeline info``), the result of
``list_available_stages``, the ``stages`` entry of the ``validate_pipeline`` /
``Pipeline.validate`` report, and ``read_metadata``'s
``file.completed_stages``. The report's ``stage_environments`` keys, the stage
names inside its drift lines and its "Missing data for completed stage" errors
are canonical too, and so is every other published environment surface: the
``environment_drift`` event message, the report table's ``environment_mixed``
row, the HTML report's environment table and the re-run drift log. The one
environment key that is not a stage is the Stage 2b shape recommendation's
entry, published as ``tau_shape`` (it is stored under
``processing_parameters/stage2b_shape_recommendation``, which does not change).
An environment key that is neither a known stage's storage key nor that one is
kept as the file recorded it, and a completed-stage key no stage of this
version owns is left out of the canonical lists.

**Re-run order** is the topological order of the stage dependencies with ties
broken by the order of ``Stage`` (the order listed above). Every list of stages
a call reports -- what it invalidated, what is runnable, what a refresh
re-runs -- is in this order.

Each spelling of a stage has one read-only mapping in
``ftmwpipeline.contract``:

* ``STAGE_KEYS``: the storage key.
* ``STAGE_SETTINGS_PREFIX``: the settings and preset prefix (``stage1`` ...
  ``stage5``). ``tau`` and ``tau_g`` share ``stage2b``. ``None`` for ``data``,
  ``timebase`` and ``review``, which have no settings record.
* ``STAGE_KNOB_PREFIX``: the tuning-registry knob prefix. ``data`` is
  ``stage0`` (the start-detection knobs). ``tau`` and ``tau_g`` share
  ``stage2b``: the twins run one STFT classifier recipe, so its knobs feed
  both. ``None`` for ``timebase`` and ``review``.

``PROVENANCE_NAMES`` maps the one provenance key that is not a stage's storage
key to its published name (``stage2b_shape_recommendation`` to ``tau_shape``),
and ``canonical_provenance_name(key)`` names any per-stage provenance key: the
canonical stage name, ``tau_shape``, or the key unchanged when it is neither.

``stage_for_key`` / ``key_for_stage`` are the one-to-one inverse; the settings
and knob mappings are not one-to-one and have none.
``capabilities()["stages"]`` lists them all as ``[{"stage", "storage_key",
"settings_prefix", "knob_prefix", "depends_on"}]`` in enum order, with
``depends_on`` in canonical names.

.. _contract-errors:

Typed errors
------------

Every refusal a program may need to route on is a member of the exception
family rooted at :class:`~ftmwpipeline.file_manager.PipelineFileError`. Each
carries a stable ``code`` and typed attributes, and ``to_dict()`` returns::

    {"schema": "ftmw/error@1", "code": "not_found",
     "message": "...", "kind": "window", "ids": [4, 9]}

.. list-table::
   :header-rows: 1
   :widths: 24 28 48

   * - Code
     - Exception
     - Extra fields
   * - ``stage_not_run``
     - ``StageDependencyError``
     - ``missing_dependencies`` (canonical stage names), ``command`` (``null``
       with ``command_absent`` ``not_run`` when the refusal names no producing
       verb; the Python attribute is ``None``)
   * - ``not_found``
     - ``NotFoundError``
     - ``kind`` (``"window"``, ``"peak"``, ``"file"`` or ``"decision"``),
       ``ids`` (every id the request named that does not exist; a peak or
       window named by frequency is reported by that frequency in MHz)
   * - ``not_found``
     - ``PipelineFileNotFoundError`` (a path that does not exist)
     - ``kind`` (``"file"``), ``ids`` (the path)
   * - ``incomplete_provenance``
     - ``IncompleteProvenanceError``
     - ``missing``
   * - ``file_incompatible``
     - ``PipelineCompatibilityError``
     - ``file_version``, ``supported_version``
   * - ``file_corrupt``
     - ``PipelineCorruptionError``
     - none
   * - ``epoch_mismatch``
     - ``AnalysisEpochMismatchError``
     - ``file_epoch``, ``current_epoch`` (``null`` with ``_absent:
       "not_run"`` when the file never recorded one)
   * - ``file_exists``
     - ``PipelineExistsError``
     - none
   * - ``bad_setting``
     - ``BadSettingError`` (a ``ValueError``)
     - ``path`` (the registry path of the setting, or the argument name),
       ``expected`` (what would have been accepted), ``value`` (what was
       given); see :ref:`contract-bad-settings`
   * - ``algorithm_failed``
     - ``AlgorithmFailedError`` (a ``RuntimeError``)
     - ``stage`` (the canonical stage whose algorithm could not produce a
       result from valid inputs). Declared, but no call raises it yet
   * - ``cancelled``
     - ``OperationCancelledError``
     - ``stage`` (the stage interrupted, or ``null`` between stages and for a
       step that is not a stage), ``completed_stages`` (canonical names, in
       order), ``completed_windows`` (the windows a cancelled fit kept as a
       partial fit, sorted; ``[]`` otherwise); see
       :ref:`machine-contract-events`
   * - ``callback_failed``
     - ``CallbackFailedError``
     - ``event_schema`` (the event being delivered), ``completed_windows``
       (as for ``cancelled``); the callback's exception is the ``__cause__``
   * - ``write_conflict``
     - ``WriteConflictError``
     - ``path`` (the file another process wrote while this call was writing
       it; this call's changes were discarded and the other write stands); see
       :ref:`contract-crash-safety`
   * - ``curation_conflict``
     - ``CurationConflictError`` (a ``ValueError``)
     - ``reason`` (a stable slug, see :ref:`contract-curation-refusals`),
       ``ids`` (the windows, peaks or decisions involved, per ``reason``;
       ``[]`` when it names none)
   * - ``pipeline_error``
     - ``PipelineFileError`` (the base class)
     - none. The declared fallback: a direct raise of the base class carries
       it, and ``run_pipeline`` reports a failure that is not a typed error
       under it (:ref:`machine-contract-events`)

``capabilities()["codes"]`` lists exactly these codes: the codes the
installation *declares*. A declared code is stable, but not every declared code
is raised by some call; ``algorithm_failed`` is declared and not yet raised.

Route on ``code`` (or the class); the ``message`` text is for people. Each typed
error is also a subclass of the built-in exception a caller would otherwise
catch (most are ``ValueError``; ``NotFoundError`` is a ``KeyError``;
``PipelineFileNotFoundError`` is also a ``FileNotFoundError``;
``PipelineCorruptionError`` is also a ``RuntimeError`` and an ``OSError``), so
an ``except`` clause written for the built-in still catches it. Every typed
error pickles, so it survives a process pool.

On the command line every verb reports a typed error through one mapping:
``Error: <message>`` on stderr, or the ``ftmw/error@1`` dict on stderr under
``--json`` (and under ``--format json`` on a verb that takes it), always as the
last stderr line. A ``read`` accessor always prints the dict. The exit code
follows the code (:ref:`contract-exit-codes`).

.. _contract-exit-codes:

Exit codes
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Condition
     - Exit
     - Notes
   * - success
     - ``0``
     -
   * - ``file_corrupt``
     - ``2``
     - a file that exists but cannot be opened as a pipeline file
       (:ref:`contract-file-errors`)
   * - ``algorithm_failed``
     - ``2``
     - declared; not yet raised
   * - a command-line usage error
     - ``2``
     - an unknown option or a value the option parser refuses (``--trim``
       that is not ``MIN:MAX``, for example); reported by the parser as a
       usage message, never as JSON
   * - ``cancelled``, or Ctrl-C
     - ``130``
     - the first Ctrl-C on a long verb cancels it (the ``cancelled`` error);
       a second one interrupts at once
   * - every other code (``not_found``, ``stage_not_run``, ``bad_setting``,
       ``callback_failed``, ``file_incompatible``, ``write_conflict``, ...)
       and any other error
     - ``1``
     -

.. _contract-bad-settings:

Bad settings
~~~~~~~~~~~~

A refusal of a setting value raises ``BadSettingError`` (``bad_setting``; also a
``ValueError``) before any work starts. ``path`` names the setting, so a client
can point at the offending knob. The calls that raise it:

* ``settings_set`` / ``settings_unset`` (API, ``Pipeline`` and ``settings
  set`` / ``unset``): an unknown or malformed path (``path`` is the knob), a
  value of the wrong type or one that does not parse, a bad ``stage5.shape``
  choice, invalid ``stage5.spur.clocks``, a value outside the ``choices`` or
  ``bounds`` the field's settings row declares (:ref:`contract-settings-rows`),
  or unsetting a field that is not optional. The file is not touched. For a
  violated declaration ``path`` is the knob, ``value`` is what the caller
  passed (before coercion), and ``expected`` states the declaration: ``one of
  'a', 'b', ...`` for ``choices``, and ``a value in [lo, hi)`` for ``bounds``
  (a square bracket includes that end, a round one excludes it; an open end
  reads ``-inf`` / ``inf``). Bounds apply to a numeric value, or to every
  element of a pair or list, and ``nan`` is refused wherever bounds exist.
  ``None`` (the unset request) is never checked against either.
* Every stage call, when it resolves its settings: a resolved value outside
  the ``choices`` or ``bounds`` its field declares, whichever layer supplied
  it -- a ``settings=`` object, a preset or a persisted record (one edited by
  hand, or written by a release that did not check). ``path`` is the registry
  path (``stage5.conservative.n_eff_kind``), ``expected`` and the bracket
  notation are as for ``settings_set``, and ``value`` is the resolved value.
  Only the value that wins is checked: a bad value in a layer a higher one
  overrides is never used and never refused, and an unset field falls through
  to its default. ``settings_show`` still displays a bad persisted value, and
  ``settings_set`` repairs it.
* ``preset=`` / ``--preset`` on every stage that takes one (and
  ``settings_show`` / ``settings_defaults``): a preset whose root or block is
  not a mapping, or that names an unknown field or key (``path`` is
  ``preset``, the block name, or ``block.field``).
* ``set_clock_sources`` and the ``clocks=`` argument of ``fit``: an invalid
  clock declaration (``path`` ``stage5.spur.clocks``); the ``shape=`` argument:
  ``path`` ``stage5.shape``.
* ``trim=`` / ``--trim`` with a value that is not ``min:max`` with
  ``max > min`` (``path`` ``stage1.trim``). On the CLI the option parser
  reports a bad ``--trim`` as a usage error (exit 2).
* ``scan_run`` / ``scan_batch`` / ``scan run`` with an unregistered knob
  (``path`` is the knob). This one is also a ``KeyError``, and its ``str()`` is
  the plain message.
* The curation calls, with ``path`` naming what the caller wrote
  (:ref:`contract-curation-refusals`), and ``CurationAction`` construction
  (:ref:`curation-as-data-contract`).

The stage calls that refuse an argument or a resolved setting:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Public call
     - ``bad_setting`` ``path``
   * - ``import_data`` (and ``Pipeline.create``, ``data import``),
       ``preview_source``, ``load_fid`` / source validation
     - ``format`` (an unknown ``format_name``, or auto-detection found no
       format; ``value`` the name given, or ``null``); ``source`` (a source
       the resolved format's loader does not accept, including one that does
       not fit a format named explicitly)
   * - ``compute_ft``
     - ``stage1.start_us`` (negative), ``stage1.end_us`` (not after
       ``start_us``, or past the end of the recording), ``stage1.trim`` (no
       data points in the range)
   * - ``estimate_noise``
     - ``stage2.n_iter`` (< 1), ``stage2.smoothing_percentile`` (outside
       [0, 100])
   * - ``calibrate_tau``, ``recommend_shape``
     - ``shape`` (not ``lorentzian`` / ``gaussian``), ``stage1.trim`` (value
       ``null``: no trim persisted by ``compute_ft``), ``stage2b.stft.n_seg``,
       ``stage2b.stft.sigma_time``, ``stage2b.band.band_edges_mhz``,
       ``stage2b.band.band_labels``, ``stage2b.gaussian.tau_G_bound_hi``,
       ``stage2b.gaussian.tau_G_upper_fraction``,
       ``stage2b.recommendation.tau_bound_hi``
   * - ``calibrate_timebase``
     - ``kappa_sys`` (negative or non-finite), ``snr_min`` (<= 0 or
       non-finite), ``stage5.spur.clocks`` (a malformed ``clocks=`` argument,
       no declaration, or no locked source)
   * - ``detect_peaks``
     - ``stage3.promotion.min_snr``, ``stage3.promotion.internal_min_snr``,
       ``stage3.promotion.weak_medium_snr``, ``stage3.savgol.sg_order``,
       ``stage3.savgol.sg_window``, ``stage3.primary_pass.primary_window``
       (unknown apodization)
   * - ``detect_start_time``
     - ``stage0.step_us`` (FID too short for the sweep step)
   * - ``fit_peaks``
     - ``stage5.shape``, ``tau_maj_override_us`` / ``sigma_tau_override_us``
       (only one of the pair, or non-positive), ``stage5.tau.tau0_us``,
       ``stage5.tau.max_decay_factor`` (must exceed 1)
   * - ``show_fit``
     - ``freqs`` (``value`` lists every frequency in no window at once),
       ``apodize`` / ``apodize_us``; an unknown window id is ``not_found``
       (kind ``window``, every unknown id at once)
   * - ``review_create``
     - ``anchor_mhz`` (outside the analysis band, or already inside a window)
   * - ``run_pipeline``
     - ``stage1.trim`` (not given)

A source path that does not exist is ``not_found`` (kind ``"file"``, ``ids``
the source path; a ``PipelineFileNotFoundError``) for ``import_data`` and
``preview_source`` alike. Reading the noise result before ``noise run`` (or
``ft run``) raises ``StageDependencyError`` (``stage_not_run``, ``command``
``noise run`` / ``ft run``). A typed error raised inside a stage propagates
with its own type; it is never re-wrapped as a ``RuntimeError``.

.. _contract-curation-refusals:

Curation refusals
~~~~~~~~~~~~~~~~~

The review calls (``review_edit``, ``review_apply``, ``review_preview``,
``review_create``, ``review_accept``, ``review_undo`` and their ``Pipeline``,
session and CLI spellings) refuse with typed errors. A batch refusal that names
one action keeps its type and adds the action to the message (``curation action
<n> (...) failed: ...``).

* ``bad_setting``, with ``path`` naming what the caller wrote:

  * a curation-file cell is ``curation[line <n>].<column>``, ``<column>``
    one of ``action``, ``window``, ``freqs``, ``params`` (an unknown or
    ``merge`` / ``split`` action, a window id that is not an integer or is
    missing, a non-numeric frequency, a malformed ``uid:N``, the wrong number
    of frequencies, unexpected or malformed parameters). A ``# frame:`` or
    ``# epsilon:`` directive is the cell ``frame`` / ``epsilon`` of its line:
    a bad or conflicting value, ``frame: calibrated`` without an epsilon
    (``frame``), an epsilon without ``frame: calibrated``, and an epsilon
    stamp that no longer matches the file's current epsilon (``epsilon``;
    *frame drift*);
  * a field of the i-th action of ``actions=`` is ``actions[<i>].<field>``:
    a dict ``CurationAction.from_dict`` refuses (an unknown key is the path's
    field), an item that is neither an action nor a dict (``actions[<i>]``),
    an action ``frame`` that disagrees with the call's ``frame=``
    (``actions[<i>].frame``), and an ``epsilon`` stamp that has drifted
    (``actions[<i>].epsilon``). A frequency with no frame at all on a
    ``self_calibrated`` file (neither the action's nor the call's) is
    ``frame``, the call's argument, as for a curation file without a
    ``# frame:`` header; the message names the action;
  * a ``create``'s anchor refused inside a batch (outside the analysis band,
    or already inside a window) is the cell or field the anchor came from:
    ``curation[line <n>].freqs`` or ``actions[<i>].freq_mhz``, also for the
    create an uncovered ``add`` implies. ``review_create`` names its argument,
    ``anchor_mhz``, and ``review_edit``'s implied create names ``add``;
  * ``review_edit``'s tokens are ``add`` / ``remove`` (a malformed token, or
    a ``uid:N`` given to ``add``).

* ``not_found``. Every frequency in ``ids`` is the one the caller wrote, in the
  frame it was written in (a calibrated request is answered in calibrated
  MHz), so a client can match it against its request.

  * ``kind`` ``"peak"``: a ``remove`` / merge / split frequency that matches
    no fitted peak within the snap tolerance (``ids`` every such frequency of
    the action), and a ``uid:N`` no fitted peak carries;
  * ``kind`` ``"window"``: a window id the fit does not have (every unknown
    id of a batch at once, including when the batch also creates windows:
    only an id one of the batch's ``create`` rows could mint is left to the
    per-action check; an id a structural merge absorbed is never one), and an
    omitted-window ``add`` / ``remove`` target that no live window covers
    (``ids`` the uncovered frequencies);
  * ``kind`` ``"decision"``: ``review_undo`` ids the decision log does not
    hold, every one of them (on a file with no recorded decisions, every
    requested id).

* ``curation_conflict`` (``CurationConflictError``), a valid request that
  conflicts with the file's review state. ``reason`` is one of:

  .. list-table::
     :header-rows: 1
     :widths: 30 70

     * - ``reason``
       - When (``ids``)
     * - ``line_already_fitted``
       - an ``add`` seeds at the birth position of a line already fitted
         (that line's ``peak_uid``)
     * - ``targets_span_windows``
       - the targets of one ``review_edit`` without a window resolve to
         different windows (those window ids)
     * - ``orphans_created_window``
       - ``review_undo`` would drop a window that surviving decisions act on
         (the decisions to undo with it)
     * - ``baseline_unavailable``
       - ``review_undo``, or an apply at a ``log_prefix``, needs the
         automatic-fit baseline the first edit snapshots, and the file's
         edits were recorded without one (by a version before ``review
         undo``, or the snapshot was removed) (``[]``)
     * - ``replay_conflict``
       - replaying a recorded window creation no longer reproduces its
         window: it would widen another window, or its id is taken (the
         recorded id, then the widened window's id)
     * - ``replay_diverged``
       - ``review_undo``, or an apply at a ``log_prefix``, would replay a
         surviving decision as a different action (another ``kind`` or
         window) than the one recorded; only a decision recorded with a
         per-call snap tolerance, before contract 16, can (that decision)
     * - ``target_outside_window``
       - an ``add`` whose seed, after snapping to a ledger candidate, falls
         outside the range of the window it names (that window)
     * - ``fit_plan_unavailable``
       - the edit would refit a window a Stage 5 structural merge changed, in a
         fit made before the fit stored its plan, or create a window inside or
         against such a merge: the windows the fit was made on are not in the
         file (the windows the edit would have refit; for a create inside a
         merged range, every affected window). Re-running ``fit run`` clears it

  Reasons are only ever added.

.. _contract-file-errors:

Missing, corrupt and incompatible files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every call and every verb that takes a ``.ftmw`` file refuses an unopenable one
the same way, from its first read:

* a path that does not exist raises ``PipelineFileNotFoundError``
  (``not_found`` with ``kind`` ``"file"``; also a ``FileNotFoundError``;
  exit 1);
* a path that exists but cannot be opened as a pipeline file -- not HDF5,
  unreadable, missing its source metadata -- raises ``PipelineCorruptionError``
  (``file_corrupt``; also a ``RuntimeError`` and an ``OSError``, chained from
  the underlying error; exit 2);
* a file from a newer MAJOR format raises ``PipelineCompatibilityError``
  (``file_incompatible``; exit 1).

A permission failure, or HDF5's refusal while another process holds the file
open for writing, is not corruption: it propagates as the original ``OSError``
so a client can retry. The check is the open and format-version gate only; it
does not require the provenance record. ``Pipeline.open``, the read accessors,
``read table`` / ``meta`` / ``list``, ``info`` and every stage, ``review``,
``report``, ``settings``, ``scan`` and ``clocks`` verb agree on this, and each
refuses before it writes anything.

``validate_pipeline`` (``Pipeline.validate``) reports problems in a file that
opens. For one that does not, it raises the open error instead of returning
``{"valid": False}``, and so it does for a file that opens but cannot be read
while the report is built: an ``OSError`` from any read -- the FID, the
environment record, the stage-data check -- raises exactly as opening the file
would. Only integrity problems in a readable file go in the report: an empty or
undecodable FID, a completed stage without its data, and any other failure
while the report is built (``valid: False`` with a ``Validation failed``
error); the report never swallows a typed error. ``Pipeline.info`` /
``get_pipeline_info`` / ``list_available_stages`` raise for a file they cannot
open, so the status calls and the report cannot disagree.

.. _machine-contract-cli-json:

Machine-readable CLI output: ``--json``
---------------------------------------

Every CLI verb accepts ``--json``. It is one uniform switch, because
``--format`` already means the report file format (``report``) or the source
format (``data import``, ``run``) on some verbs. ``--format json`` is a full
synonym for ``--json`` on ``info``, ``timebase state`` and ``review
snap-tolerance``. On ``read table``, ``read meta`` and ``report table``,
``--format json`` selects the JSON rendering of the table, and a typed error is
then also printed as its dict. Under ``--json``:

* stdout carries **exactly one JSON document and nothing else**: the verb's
  human printing is suppressed, and logging stays on stderr. A verb that fails
  after printing a message prints that message on stderr instead.
* every number is finite: a non-finite value of a named field is ``null`` with
  a ``"<field>_absent"`` sibling (:ref:`contract-absent`).
* an error is the ``ftmw/error@1`` dict on stderr, with the exit code of
  :ref:`contract-exit-codes`.

**Read accessors.** A ``read <accessor>`` verb takes ``--json`` and ``--format
json`` (its only and default format, so neither changes the output),
``-o/--output DIR`` and ``-v``. It prints its envelope (:ref:`contract-rules`)
with or without the flag; the JSON is always strictly valid. ``--output`` names
a **directory** (created if needed). Each array-valued field is written there
as ``<field path>.npy`` (for example ``samples.npy`` or ``components.0.npy``;
dtype, shape and byte order are self-describing) and the JSON names the file in
the array's place. An accessor whose result holds arrays refuses to run without
``--output`` (exit ``1``); a result with no arrays writes nothing and does not
create the directory. This differs from ``read table`` and ``read meta``, where
``--output`` names a single text file.

**Stage-running and curation verbs** print the ``ftmw/run_result@1``
envelope::

    {"schema": "ftmw/run_result@1", "verb": "noise run", "stage": "noise",
     "invalidated": ["tau", "peaks", "windows", "fit", "review"],
     "summary": {"total_points": 152760, "noise_points": 150374, ...}}

``verb`` is ``"<object> <verb>"`` (the canonical object name, never the
``stageN`` synonym). ``stage`` is the canonical stage name (``tau_g`` for ``tau
run --gaussian``) or ``null`` for a verb that is not one stage (``start run``,
``settings``, ``clocks``, ``report run``, ``run``). ``invalidated`` is the
result's own ``invalidated`` (canonical names, in re-run order; ``[]`` when
none; :ref:`contract-invalidation`). ``summary`` holds the scalars the human
output reports (counts, chosen values, paths written, a dict of counts), never
an array. A value with no measurement (an undefined ``epsilon``) is ``null``
with its ``"<field>_absent"`` sibling.

.. list-table::
   :header-rows: 1
   :widths: 28 14 58

   * - Verb
     - ``stage``
     - ``--json`` prints
   * - ``data import``
     - ``data``
     - run_result: ``pipeline_file``, ``source_format``, ``n_points``,
       ``duration_us``, ``probe_freq_mhz``, ``sideband``, ``shots``,
       ``file_size_mb``
   * - ``start run``
     - ``null``
     - run_result: band, ``chirp_detected``, chirp-end values, ``start_us``,
       ``stamped``
   * - ``ft run``
     - ``ft``
     - run_result: ``fid_points``, ``preprocessed_points``,
       ``frequency_points`` (the stored spectrum, after the trim)
   * - ``noise run``
     - ``noise``
     - run_result: point counts, ``noise_fraction``, frequency range, RMS
       statistics, estimator diagnostics
   * - ``tau run`` / ``tau recommend``
     - ``tau`` (``tau_g`` with ``--gaussian``)
     - run_result: ``tau_maj_us``, ``sigma_tau_us``, contributors, bimodality,
       preconditions / the recommended shape and vote rates
   * - ``timebase run``
     - ``timebase``
     - run_result: ``epsilon``, ``sigma_epsilon``, ``lattice_g_mhz``, tones
       used / detected, preconditions
   * - ``peaks run``, ``windows run``, ``fit run``
     - ``peaks``, ``windows``, ``fit``
     - run_result: the counts the human output reports (``fit run`` adds the
       resume fields of :ref:`contract-partial-fits`)
   * - ``review run`` / ``apply`` / ``edit`` / ``create`` / ``accept`` /
       ``undo``
     - ``review``
     - run_result: window / action counts, peak counts and reduced chi-squared
       before and after, ``converged``, created-window mode
   * - ``settings set`` / ``unset``
     - ``null``
     - run_result: ``path`` (and the new ``value`` for ``set``)
   * - ``clocks set`` / ``add`` / ``remove`` / ``clear``
     - ``null``
     - run_result: ``n_clock_sources`` (no stage is invalidated)
   * - ``run``
     - ``null``
     - run_result: ``status``, ``pipeline_file``, ``n_completed_stages``,
       ``completed_stages`` (one comma-separated string of canonical names),
       ``failed_stage`` (null for a step that is not a stage), ``failed_step``
       (the failing step's progress label, null on success), ``error_code`` and
       ``error`` (the failure's code and message), ``timebase``, ``elapsed_s``, ``report_table``,
       ``report_html``. A failed run still prints its envelope, with
       ``status`` ``"error"``, writes the failure's ``ftmw/error@1`` dict to
       stderr and exits 1
   * - ``report run``
     - ``null``
     - run_result: ``scope``, ``table`` and ``html`` paths
   * - ``data show``, ``start show``, ``ft show``, ``noise show``, ``tau show``,
       ``peaks show``, ``windows show``, ``fit show``, ``review show`` (with
       ``--output`` / ``--output-dir``)
     - n/a
     - ``{"paths": [...]}``: the image files written (empty when none)
   * - ``review show``
     - n/a
     - the table of the chosen mode: ``{"windows": [...]}``, ``{"attention":
       [...]}`` (:ref:`contract-attention`), ``{"bar", "window_id",
       "candidates": [...]}`` or one window's detail
   * - ``review rank`` / ``log`` / ``preview`` / ``snap-tolerance`` /
       ``acknowledge-environment``
     - n/a
     - ``{"windows": [...]}`` / ``{"entries": [...]}`` / ``{"warnings",
       "created_windows", "windows"}`` / the snap tolerance / the
       acknowledgement
   * - ``settings show`` / ``defaults``
     - n/a
     - ``{"settings": [...]}``, the rows of the table
       (:ref:`contract-settings-rows`); ``settings export`` prints
       ``{"out_path", "paths"}``
   * - ``clocks show``
     - n/a
     - ``{"clocks": [{"freq_mhz", "locked", "label"}, ...]}``
   * - ``scan list`` / ``scan run`` / ``scan all``
     - n/a
     - ``{"knobs": [...]}`` / ``{"result": {...}}`` / ``{"items": [...],
       "n_ok", "n_failed", "output_dir"}``; a sweep is its table, never the
       per-value stage results
   * - ``info``, ``timebase show`` / ``state``, ``fit check``
     - n/a
     - the object these verbs report, through ``to_jsonable``
   * - ``read table`` / ``read meta`` / ``read list``
     - n/a
     - the table (a list of records) / the metadata / ``{"tables": {...}}``;
       with ``--output`` the table text is written there and the payload is
       ``{"what", "path"}``
   * - ``report table`` / ``report diff``
     - n/a
     - the table as JSON, or ``{"format", "path"}`` with ``--output`` /
       ``{"path"}``
   * - ``formats``, ``validate``, ``version``
     - n/a
     - ``{"formats": [...]}`` (or one format's info) / ``{"components", "ok"}``
       / ``{"version", "description", "has_matplotlib"}``

``review merge`` and ``review split`` are not verbs (an ``edit --add`` /
``--remove`` is read as one), so they have no envelope of their own.

.. _contract-accessors:

Read accessors
--------------

Every accessor in ``capabilities()["accessors"]`` exists on the API, on
``Pipeline`` and as exactly one CLI verb, ``ftmwpipeline read <name>``. None of
them writes the file. Every file-bound one raises
``PipelineFileNotFoundError`` (``not_found``) for a missing file and
``PipelineCorruptionError`` (``file_corrupt``, exit 2) for a file that is not
HDF5 (:ref:`contract-file-errors`).

.. _contract-declared-surface:

Persisted data, provenance and products
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``frequency_calibration``, ``refit_snap_tol_mhz``, ``read_metadata``,
``read_tables``, ``read_table``, ``settings_defaults``, ``settings_show``,
``get_final_products``, ``review_log``, ``get_pipeline_info`` and
``compute_display_ft`` each have a schema name, a constant in
``ftmwpipeline.contract`` (``CALIBRATION_SCHEMA``, ``SNAP_TOLERANCE_SCHEMA``,
...): ``ftmw/calibration@1``, ``ftmw/snap_tolerance@1``, ``ftmw/metadata@1``,
``ftmw/tables@1``, ``ftmw/table@1``, ``ftmw/settings_defaults@1``,
``ftmw/settings@1``, ``ftmw/final_products@1``, ``ftmw/review_log@1``,
``ftmw/pipeline_info@1`` and ``ftmw/display_ft@1``. (``ftmw/curation_action@1``,
also declared, names a request rather than a result: see
:ref:`curation-as-data-contract`.) ``CalibrationStamp`` and ``FinalProducts``
declare theirs as ``__ftmw_schema__``; the ``read`` verb stamps the rest by the
envelope rules. ``get_pipeline_info`` always carries ``warnings``.

Each, with its absence cases:

* ``read_metadata`` / ``read_tables`` / ``read_table`` -- the persisted scalars
  and table columns, raw. ``tau.`` / ``tau_g.`` / ``timebase.`` keys appear only
  once that calibration has run, so a file without it simply lacks them; an
  unknown table or column is a ``ValueError``. A table whose stage has not run
  raises ``StageDependencyError`` (``stage_not_run``, also a ``ValueError``)
  with ``missing_dependencies`` the stage group and ``command`` the verb that
  produces it: ``tau run``, ``tau run --gaussian``, ``peaks run``,
  ``windows run`` or ``fit run``, for every table. ``read read_table FILE TABLE
  [--columns a,b]`` writes each column to ``<column>.npy`` under ``--output``.

  ``read_tables`` lists, per table, ``available``, ``n_rows``, ``columns`` and
  ``group``. ``n_rows`` is ``Absent.NOT_RUN`` when the producing stage has not
  run, and otherwise always an ``int``: a table whose stage records no count is
  counted from the table itself. (A stored table that cannot be read at all,
  such as a pre-1.0 fit layout, reports ``Absent.UNDEFINED``.) ``columns``
  includes every ``<column>__status`` companion.

  **Status columns.** The ``fit_peaks``, ``fit_windows``, ``fit_audit``,
  ``fit_doublets`` and ``peaks`` tables carry a ``uint8``
  ``<column>__status`` companion (``0`` present, ``1`` not run, ``2``
  undefined) for each column below, inserted right after it; column selection
  accepts them. The value column keeps its stored fill (``nan``, ``inf``,
  ``-1`` or ``""``), which a program must not read when the status is not
  ``0``. A degenerate statistic (below) reads as ``nan`` with status ``2``
  whether the file stores ``nan`` or an earlier release's ordinary number.
  A column that predates the file reads as its fill with status ``1``,
  except in the ``fit_peaks`` knockout block, where the knockout rule below
  decides (a test that ran on a file predating ``knockout_p_value`` reads
  ``2``, as on ``FinalPeak``).
  The rules are the ones ``FinalPeak`` uses for the same quantity.

  * ``fit_peaks``. Knockout: ``knockout_supported``, ``knockout_delta_chi2``,
    ``knockout_expected_delta_chi2``, ``chi_squared``, ``knockout_p_value``,
    ``knockout_n_eff`` and ``knockout_aicc_delta`` are ``1`` when the test did
    not run (no record: ``knockout_supported`` is ``-1`` or the delta chi2 is
    ``nan``); when it ran, a non-finite value is ``2`` (``knockout_p_value``
    is ``nan`` for a degenerate F-test; an earlier release stored ``1.0``
    there, which the file cannot distinguish). ``frequency_error``,
    ``amplitude_error``, ``phase``, ``phase_error``, ``decay_rate``,
    ``decay_rate_error`` (also when tau was held fixed) and ``snr``: a
    non-finite value is ``2``. ``detection_index``: ``-1`` (no Stage 3
    detection seeded the line) is ``2``. ``derivation`` and ``peak_uid``:
    ``-1`` is ``1``. ``unresolved_spread_mhz``: ``nan`` (never a merged
    multiplet) is ``1``. ``clock_lattice``: ``1`` when the fit recorded no
    clock declaration, ``2`` when it did and the line is off the lattice
    (``""``), else ``0``.
  * ``fit_windows``. ``freq_min``, ``freq_max``: ``nan`` is ``1``.
    ``edge_coherence_low``, ``edge_coherence_high``: ``nan`` is ``2`` when the
    fit computed the edge and it is undefined (an empty residual, or a band
    with no positive noise; an earlier release stored ``0.0`` there, which
    reads as ``nan``), and ``1`` when the fit never evaluated the window.
    ``tau_fitted``: ``-1`` is ``1``.
    ``tau_error``: ``2`` for any non-finite value (held fixed or singular).
    ``tau_us`` (also when not positive), ``aic``, ``reduced_chi2``: a
    non-finite value is ``2``.
  * ``fit_audit``. ``n_eff`` and ``aicc_delta``: ``1`` on a ``seed`` or
    ``spur-drop`` step and on a separation reject (``separation_ok`` false with
    no ``n_eff``), else ``2`` if non-finite. ``f_statistic``: ``1`` on
    ``spur-drop``, ``knockout-null`` and separation rejects; ``p_value``: ``1``
    on ``spur-drop`` and separation rejects (a ``nan`` ``p_value`` on
    ``knockout-null`` is ``2``). For a separation reject no test ran, so the
    stored placeholder ``f_statistic`` 0.0 and ``p_value`` 1.0 read as ``nan``
    with status ``1``. A degenerate F-test (no added parameter, no residual
    degrees of freedom, or a non-positive ``chi2_after``) has no value:
    ``f_statistic`` and ``p_value`` are ``2``. An earlier release stored
    such a test as ``f_statistic`` 0.0 and ``p_value`` 1.0; those read as
    ``nan`` with status ``2`` wherever the stored chi-squared values rule out
    a genuine non-improvement (``chi2_after`` below ``chi2_before``, or not
    positive). ``chi2_after`` and ``aic_after``: ``1`` on ``spur-drop``,
    else ``2`` if non-finite.
  * ``fit_doublets``. ``chi2r_merged``, ``delta_chi2_raw``, ``delta_aicc``,
    ``merged_frequency_mhz``, ``merged_amplitude``, ``merged_phase``,
    ``merged_tau_us`` and ``orth_evidence_delta_chi2``: a non-finite value is
    ``2`` (the merged refit was attempted). ``orth_evidence_delta_chi2`` is
    also ``2`` when the weak partner had no usable support
    (``support_bins`` 0), where an earlier release stored ``0.0``, which
    reads as ``nan``. (An earlier release's ``0.0`` from a disabled
    line-evidence escape sits on a usable support, so the file cannot tell it
    from a measured zero; it reads as present.)
  * ``peaks``. ``internal_snr``, ``internal_frequency``, ``leakage_pedestal``:
    ``nan`` (the internal pass did not contribute the peak) is ``1``, except
    that ``internal_snr`` is ``2`` when the internal pass did contribute it
    (``internal_frequency`` is finite) and its SNR has no value (no positive
    internal noise; an earlier release stored ``0.0``, which reads as
    ``nan``). ``snr`` and ``noise_std_local``: non-finite is ``2``; ``snr``
    is also ``2`` where ``noise_std_local`` is not positive (an earlier
    release stored ``0.0``, which reads as ``nan``). ``index`` ``-1`` and
    ``classification`` ``""`` are ``1``. ``promoted``: ``1`` for every row when
    the file records no promotion cutoff (it predates the record), else ``0``.

  ``fit_thaw`` carries no status columns. Its ``edge_coherence_before`` and
  ``edge_coherence_after`` are the values the thaw gate read, so an undefined
  edge (an empty residual, or a band with no positive noise) is ``0.0`` there,
  not ``nan``; ``edge_coherence_after`` is ``nan`` only when the joint co-fit
  produced no usable fit (it did not converge, or returned the wrong number
  of peaks).

  ``fit_replans`` carries no status columns either. A row with ``accepted``
  false has a ``reason`` that starts with exactly one of ``not merged:``
  (the flagged window's fit holds no line, or no window touches it),
  ``refused:`` (the merge would break the plan's width or peak cap),
  ``deferred:`` (the pair waited behind another merge that round) or
  ``failed:`` (Stage 4 could not apply it); an accepted row's ``reason``
  describes the flagged edge. ``revision_after`` equals ``revision_before`` on
  every row that is not accepted.

  The ``window_status`` table is described under :ref:`contract-window-status`.

  ``read_metadata`` has two kinds of absence. A key of a stage that has not
  run is *omitted* (read with ``.get()``). A key that is present without a
  value is ``Absent`` (``null`` plus ``"<key>_absent"`` on the wire), never
  ``None``:

  - *not run* (the file predates the record): ``file.format_version``,
    ``file.created_with``, ``source.source_path`` / ``format_name`` /
    ``import_timestamp`` / ``source_hash`` (an unset import field),
    ``stage5.acquisition_us`` (the fit recorded none),
    ``stage3.promotion_min_snr`` / ``stage3.internal_min_snr``, and any count,
    creation time, plan revision or shape attribute that a stage group does not
    carry: ``stage3.n_peaks`` / ``creation_time``, ``stage4.n_windows`` /
    ``creation_time``, ``stage4.n_dependency_edges`` (the group carries no
    ``dependency_edges`` record), and ``stage5.n_windows`` / ``n_fitted_peaks``
    / ``final_plan_revision`` / ``shape`` / ``creation_time``. A reader never
    fills a missing attribute with ``0``, ``"unknown"`` or ``"lorentzian"``; a
    recorded ``0`` (an empty edge list included) is a value, not an absence.
  - *undefined* (computed, no value): ``start.chirp_end_us`` when no chirp
    was detected; ``timebase.epsilon`` and ``timebase.sigma_epsilon`` when no
    lattice tone was used; ``timebase.lattice_g_mhz`` when no locked lattice
    exists; ``stage5.acquisition_us`` when the recorded value is not a positive
    finite number; ``tau.`` / ``tau_g.`` scalars that are not finite (a decay
    time with zero contributors, a bimodality or correlation statistic on too
    few points); and ``tau.recommended_shape`` when the vote had no winner.

  The ``ft.`` / ``stage1.`` settings echoes (an unset bound or trim) stay
  ``None``, as do ``start.resolved_band_min_mhz`` / ``resolved_band_max_mhz``
  (no band restriction).
* ``get_final_products`` -- the persisted final-products table, or ``None``
  before Stage 6 (``"value": null, "value_absent": "not_run"`` on the wire).
  Each ``FinalPeak`` carries ``peak_uid``, ``window_id``, ``origin``,
  ``derivation``, ``clock_lattice``, the ``knockout_*`` fields, the frequency
  and its sigma. Every field that can lack a value is ``Absent``, never
  ``None`` or ``nan``:

  * ``sigma_stat_khz`` and ``sigma_f_khz`` are *undefined* when the fit left
    the line without a finite frequency error; the total is never computed
    with its statistical term dropped. ``sigma_eps_khz`` and
    ``sigma_floor_khz`` are always present.
  * ``phase``, ``snr``, ``amplitude_error``, ``phase_error`` and ``snr_error``
    are *undefined* when the fit gave no finite value (``snr_error`` also when
    ``snr`` or ``amplitude_error`` is absent, or the amplitude is zero).
  * ``window_id`` and ``peak_uid`` are *not run* when the source peak records
    none (a fit written before peak identity existed); ``derivation`` is *not
    run* for a line no Stage 6 decision created or altered.
  * ``clock_lattice`` is the lattice identity of an on-lattice line; *not run*
    when the Stage 5 fit recorded no clock declaration (``spur.clocks``), so
    the lattice test never ran; *undefined* when a declaration was recorded and
    the line is off-lattice.
  * ``knockout_p_value``, ``knockout_supported`` and ``knockout_aicc_delta``
    are all *not run* when the knockout test never ran for the line (no
    record, or a ``nan`` Δχ²). When it ran, ``knockout_p_value`` and
    ``knockout_aicc_delta`` are *undefined* if not finite (a refit that did
    not converge, or an AICc with too few effective points);
    ``knockout_supported`` is then always a bool.

  The same quantity reports one status on every surface: these rules are the
  ones ``read_table``'s ``fit_peaks`` status columns follow. The stored table
  keeps each value with a ``<field>__status`` code; a table stored by an
  earlier release without those codes is decoded field by field by the rules
  above (its ``None``, ``nan`` and ``sigma_stat_khz`` ``0.0`` sentinels),
  without a rebuild and without writing the file.

  It also carries the per-line fit fields, joined from the Stage 5 fit of the
  line's window: ``decay_time_us`` and ``decay_time_error_us`` (the window's
  ``tau`` and its 1-sigma error), ``shape``, ``fwhm_mhz``,
  ``detection_index`` and ``fit_window_mhz``. ``fwhm_mhz`` is exactly
  ``fitting.validation.feature_fwhm(decay_time_us,
  read_metadata(path)["stage5.acquisition_us"], shape=shape)``, so a width a
  client computed that way does not move. ``fit_window_mhz`` is the window's
  ``(low, high)`` in the calibrated frame, like ``frequency_mhz`` (a
  two-element array on the wire); a window created during review reports its
  own bounds. ``decay_time_error_us`` is *undefined* when ``tau`` was held
  fixed, ``detection_index`` is *undefined* for a line no Stage 3 detection
  seeded, ``fwhm_mhz`` is *not run* when the fit recorded no
  ``stage5.acquisition_us``, and every one is *undefined* for a line with no
  fit record behind it. A table stored before these fields existed is rebuilt
  in memory when read (the file is not written); the next write that stores
  the table (``review run``, a curation edit, ``set_sigma_floor``, a timebase
  refresh) persists them.
* ``review_log`` -- the ``DecisionLogEntry`` rows in execution order (an empty
  list when nothing was edited). ``kind`` is one of ``add``, ``remove``,
  ``merge``, ``split``, ``accept``, ``create_window``; ``provenance`` is
  ``user``. ``evidence`` is a free-form snapshot dict and keeps its floats in
  Python: a non-finite ``chi2r_before`` / ``chi2r_after`` (no degrees of
  freedom, a fit that did not converge) stays ``inf`` there and is written as
  ``null`` with ``"<key>_absent": "undefined"`` on the wire. Every row also
  carries ``action_index`` in ``evidence``: the ``order_index`` of the first row
  of the same user action (a one-row action, a bare accept included, carries its
  own ``order_index``; a bare accept's evidence is no longer ``{}``). Rows with
  the same ``action_index`` were applied as one joint refit. Files written before
  the key existed have none; ``review_undo`` infers their groups (see below). A
  ``remove`` row's ``frequency_mhz``, and a merge's ``merged_from``, are the
  fitted frequencies (raw frame) of the peaks the request resolved to, not the
  frequencies sent; rows written before contract 16 hold the frequencies sent.
  A replay (``review_undo``, an apply at a ``log_prefix``) re-records every
  surviving row with its own ``frequency_mhz`` and ``merged_from`` verbatim;
  only ``order_index`` and ``action_index`` are renumbered after an undo. The
  snap tolerance is a property of the file (``refit_snap_tol_mhz``): from
  contract 16 no call takes one.
* ``get_pipeline_info`` -- the status dict. ``warnings`` is always present (an
  empty list when there are none). The environment fields hold ``Absent``
  rather than ``None`` / ``{}`` / ``[]``:

  - *not run*: ``format_version``, ``created_with`` and ``last_written_with``
    when the file carries no stamp; ``stage_environments`` when no stage was
    stamped (a file that predates the record); ``environment_drift`` and
    ``runtime_environment_drift`` when there are no stamps to compare (with
    stamps, ``[]`` means no drift); ``analysis_epoch`` and any other field of an
    environment record that was not captured, including inside
    ``current_environment``.
  - *undefined*: when validation could not read the stamps, every file-derived
    environment key (``format_version``, ``created_with``, ``stage_environments``,
    ``last_written_with``, both drift lists, ``environment_acknowledged``).
    ``current_environment`` does not depend on the file and is always reported.

  ``ftmwpipeline info`` prints ``Absent`` values as its usual "(unknown)" /
  "(not recorded)" placeholders, and ``info --json`` writes them as ``null``
  plus a ``"<key>_absent"`` sibling.
* ``frequency_calibration`` -- the ``CalibrationStamp``. ``probe_freq_mhz`` and
  ``sideband`` are *not run* when the file carries no Stage 0 FID acquisition
  header to read them from (no frame conversion is possible then); ``state``,
  ``epsilon``, ``sigma_epsilon`` and ``sigma_floor_khz`` are always present,
  with ``state`` saying why ``epsilon`` is ``0.0``.
* ``refit_snap_tol_mhz`` -- the Stage 6 curation snap tolerance in MHz, a
  scalar with no absence case: a file it cannot resolve raises
  ``StageDependencyError``.
* ``settings_show``, ``settings_defaults`` -- the settings rows
  (:ref:`contract-settings-rows`); ``settings_defaults`` needs no file.
* The curation results ``RefitWindowResult`` (``review edit`` / ``accept`` /
  merge / split), ``PreviewWindowResult`` (``review preview``) and
  ``AppliedWindowResult`` (``review apply``) carry ``Absent`` the same way:
  ``chi2r_before`` / ``chi2r_after`` are *not run* on a side with no fit (a
  window the batch created has no "before") and *undefined* when the fit's
  value is not finite; ``converged`` is *not run* exactly where
  ``chi2r_after`` is, and *undefined* for a window left with no peak (created
  empty, or every peak removed): no solver ran, so there is no convergence
  outcome (the wire form is ``null`` with ``"converged_absent": "undefined"``,
  and no non-convergence warning is raised); the
  five ``created_window_*`` fields are all *not run* on a window the batch did
  not create or widen (an empty ``created_window_depends_on`` list is a
  present value).
* ``compute_display_ft`` -- ``freq_array`` and ``complex_spectrum`` plus
  ``metadata`` (``amplitude_scale``, ``units_label``, ``pad_factor``). It is
  the active-portion FT Stage 5 fits, zero-filled to ``pad_factor`` times its
  density, from the first to the last active bin inside the Stage 1 trim, so
  it contains every bin the fit sees. It has no absence case: a file without
  Stage 1 raises ``StageDependencyError``. For example:

  .. code-block:: console

     $ ftmwpipeline read compute_display_ft run.ftmw --output ft/ --pad-factor 2

  The envelope (``ftmw/display_ft@1``) names ``freq_array.npy`` and
  ``complex_spectrum.npy`` (a complex128 array) under ``--output`` and carries
  ``metadata`` inline.

Reading the FID: ``fid_samples``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``fid_samples(path)`` returns the Stage 0 samples exactly as stored, as
``{"schema": "ftmw/fid_samples@1", "samples": ..., "stored_dtype": ...}``:

.. code-block:: python

   res = ftmw.fid_samples("exp.ftmw")      # or Pipeline.open(path).fid_samples()
   res["samples"]       # 1-D float64 ndarray, stored order
   res["stored_dtype"]  # what is on disk, e.g. "float64"

The values equal the stored ones: no scaling, windowing or mean removal. The
pipeline stores ``float64``; a narrower stored dtype is promoted losslessly and
``stored_dtype`` still names the on-disk type. The samples are write-once, so
the array is stable for the life of the file. The call reads one dataset. The
pipeline publishes no digest of the samples; a client that wants a spectrum
identity hashes the array itself.

.. code-block:: console

   $ ftmwpipeline read fid_samples exp.ftmw --output out/

writes ``out/samples.npy`` and prints ``{"schema": "ftmw/fid_samples@1",
"samples": "samples.npy", "stored_dtype": "float64"}``.

A file with no Stage 0 FID data raises ``StageDependencyError`` (code
``stage_not_run``, ``missing_dependencies`` ``["data"]``, ``command``
``data import``). There is no ``Absent`` value in this payload.

Display units: ``display_units``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``display_units(path)`` returns ``{"schema": "ftmw/display_units@1",
"amplitude_scale": float, "units_label": str, "units_power": int}``:

.. code-block:: console

   $ ftmwpipeline read display_units exp.ftmw

``amplitude_scale`` and ``units_label`` are exactly the pair that
``compute_display_ft`` applies to its spectrum, at every stage, including
before Stage 1 has persisted anything. The value is resolved through the Stage 1
chain (persisted, then the import-time recommendation, then the hard default),
the same chain that selects the spectrum being labelled, so a displayed
magnitude is ``abs(spectrum) * amplitude_scale`` in ``units_label``. Because the
chain ends in a hard default, ``units_power`` always resolves to an ``int``; it
carries no ``Absent`` marker.

Reading the fit thresholds: ``fit_thresholds``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``fit_thresholds(path)`` reports the thresholds the persisted Stage 5 fit
actually applied, by name. It reads only the recorded fit diagnostics (no fit is
deserialized).

.. code-block:: python

   ftmw.fit_thresholds("exp.ftmw")             # functional API
   Pipeline.open("exp.ftmw").fit_thresholds()  # Pipeline instance method

.. code-block:: console

   $ ftmwpipeline read fit_thresholds exp.ftmw

The payload is ``{"schema": "ftmw/fit_thresholds@1",
"peak_survival_snr_floor": float, "vif_collapse_threshold": float}``:

* ``peak_survival_snr_floor`` -- the survival SNR floor the fit used (the Stage 3
  promotion cutoff times the survival factor, unless overridden).
* ``vif_collapse_threshold`` -- the variance-inflation threshold of the collapse
  pass.

Either field is ``Absent.NOT_RUN`` when the file has no Stage 5 fit, when the
fit predates the recording of that threshold, or when the fit's peak-survival
pass was disabled (it then recorded neither). A default is never substituted:
the accessor reports what the fit applied or says it cannot. (The figures that
grade old fits for display fall back to the file's own resolved settings; the
accessor does not.)

Previewing a source: ``preview_source``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``preview_source(source, format_name=None)`` says what a data source holds
without importing it: the detected format and the source's FID table, one row
per FID the source holds (a Blackchirp experiment may hold many). It takes a
source path, not a ``.ftmw`` file, so it is file-less.

.. code-block:: python

   ftmw.preview_source("exp_2638")          # functional API
   Pipeline.preview_source("exp_2638")      # Pipeline class (static)

.. code-block:: console

   $ ftmwpipeline read preview_source exp_2638 [--source-format blackchirp]

``--source-format NAME`` selects the loader instead of auto-detection (the
verb's ``--format`` is the output format). The payload is
``ftmw/source_preview@1``: ``source``, ``format``, ``n_fids``, ``fids`` and
``chirp_window``. ``fids`` is a list of ``ftmwpipeline.FidPreviewRow`` records,
one per FID (``index``, ``n_points``, ``spacing_us``, ``probe_freq_mhz``,
``sideband`` -- ``"upper"`` or ``"lower"``, whatever the source encodes it as
-- ``shots`` and ``channel``, the source's channel identifier spelled exactly as
import's ``--channel`` takes it, *not run* for a source without channels).
Every field but ``index`` may be ``Absent``: a value the source does not
declare is *not run*; one that depends on load-time parameters (a Keysight
record's point and shot counts) is *undefined*. ``chirp_window`` is the window
the source declares (``chirp_start_us``, ``chirp_end_us``, ``start_margin_us``,
each a finite number or absent) or, when it declares none, ``null`` with
``chirp_window_absent: "not_run"``. A window the source declares but the code
cannot read (a non-numeric or non-finite value, a block that is not a mapping,
an unparsable Blackchirp ``chirps.csv``) is *undefined*, never *not run*; the
parse error is logged at debug level.

What each source reports:

* **Blackchirp** -- one row per ``fid/fidparams.csv`` row, indexed by row
  position (as ``load_fid`` does). The ``sideband`` cell may be the enum name
  (``LowerSideband`` / ``UpperSideband``) or the integer code (``1`` lower,
  ``0`` upper); both decode to ``"lower"`` / ``"upper"``. A column an older
  file lacks is *not run*; a cell that is not a finite number, or a sideband
  that is neither encoding, is *undefined*. The chirp window comes from
  ``chirps.csv`` and ``header.csv``.
* **CSV and native HDF5** -- one row. Fields come from the sidecar (and, for
  HDF5, the embedded attributes). A field neither the source nor its sidecar
  declares (``spacing_us``, probe frequency, sideband, shots) is *not run*;
  import's own defaults (``0``, ``"upper"``, ``1``) are never reported as if
  the source had declared them.
* **Keysight MATLAB** -- one row per ``Channel_*`` group (sorted), whose
  ``channel`` is the group name to pass as ``--channel``; spacing is that
  channel's sampling interval. ``n_points`` and ``shots`` are *undefined*
  because they depend on load-time layout parameters; probe frequency and
  sideband are *not run* (the record does not declare them).

Refusals are the ones ``import_data`` gives (:ref:`contract-bad-settings`): a
path that does not exist is ``not_found`` with kind ``"file"``; an unknown
``format_name``, or a source no format recognises, is ``bad_setting`` with
``path`` ``"format"`` (``value`` the name given, or ``null`` when detection
found none); a source that does not fit the named or detected format is
``bad_setting`` with ``path`` ``"source"``. The accessor reads only the source
and creates no ``.ftmw`` file. ``validate_source`` describes FID 0 alone.

.. _contract-settings-rows:

Typed settings rows: ``settings_show`` and ``settings_defaults``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each ``SettingRow`` (and each item of ``settings show`` / ``settings defaults``
``--json``) carries, besides ``path``, ``value``, ``source``,
``hard_default``, ``tier`` and ``help``, five fields that say how to edit it:

* ``type`` -- ``"float"``, ``"int"``, ``"bool"``, ``"str"``, ``"choice"``,
  ``"float_pair"``, ``"float_list"``, ``"str_list"``, ``"shape_spec"`` or
  ``"clock_sources"``, taken from the field's declared type (``"choice"`` is a
  string field that declares ``choices``).
* ``nullable`` -- true when the setting can legitimately resolve to no value
  (its hard default is ``None``, for example ``stage1.trim``), so ``value`` may
  be ``null``. Every setting can still be unset with ``settings_unset``.
* ``units`` -- ``"MHz"`` or ``"us"``, or ``None``.
* ``choices`` -- the allowed values as a list, or ``None``.
* ``bounds`` -- ``{"min", "max", "min_inclusive", "max_inclusive"}``, or
  ``None``.

``units``, ``choices`` and ``bounds`` are reported only where the setting's
declaration states them; ``None`` means "not stated", never "unrestricted". At
present ``stage5.conservative.n_eff_kind`` is the one ``choice``, and no setting
states bounds. ``settings_set`` / ``settings set`` enforces whatever a row
states, and so does every stage when it resolves its settings from a
``settings=`` object, a preset or the file (:ref:`contract-bad-settings`), so a
``choice`` row's ``choices`` are exactly the strings a run accepts.

``value`` and ``hard_default`` are typed JSON: a pair or list is an array, a
``ShapeSpec`` is ``{"kind": "gaussian"}``, and clock sources are an array of
``{"freq_mhz", "locked", "label"}`` objects. The same JSON is accepted back by
``settings_set`` / ``settings set``, so a row's ``value`` round-trips.

.. _contract-status:

Per-stage state: ``status``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``status(path)`` is file-bound on every interface (``ftmw.status(path)``,
``Pipeline.open(path).status()``, ``ftmwpipeline read status FILE``)::

   {"schema": "ftmw/status@1",
    "stages": [{"stage": "data", "state": "complete", "depends_on": []}, ...],
    "runnable": ["tau", ...],
    "rerun_order": ["data", "ft", ...]}

``state`` is one of ``complete``, ``partial`` or ``not_run`` (the
``stage_state`` vocabulary). ``partial`` is a Stage 5 fit interrupted by a
cancel (or a raising callback) that kept its finished windows
(:ref:`contract-partial-fits`); such a fit is still ``runnable`` (running it
resumes it). ``runnable`` lists, in enum order, the stages that are not
complete but whose dependencies all are. ``rerun_order`` lists every stage in
re-run order (:ref:`contract-stage-names`); it does not depend on the file.

.. _contract-window-status:

Window status: ``window_status``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``window_status(path)`` returns ``{"schema": "ftmw/window_status@1",
"windows": [...]}``, one ``ftmwpipeline.WindowStatusRow`` per window of the
plan the fit was made on and per window Stage 6 created, with ``window_id``,
``freq_min_mhz``, ``freq_max_mhz``, ``created``, ``n_fitted_peaks``, ``live``
and ``merged_from``. A window is **live** when the Stage 5 fit holds at least
one fitted line in it. Rows ascend by ``freq_min_mhz`` and then ``window_id``.
A created window that reuses a plan ``window_id`` (the narrow-gap widening
case) replaces that plan row, with its own bounds and ``created`` true.

**Windows after a structural merge.** Before a complete Stage 5 fit (a partial
fit included), and after a fit no structural merge revised, the plan is the
Stage 4 plan. After a merge (``final_plan_revision`` above 0) it is the plan
the fit was made on: the survivor keeps the lower id and the merged range, its
``merged_from`` lists the ids it absorbed (ascending), and an absorbed id has no
row. Every other row's ``merged_from`` is empty (``()``; ``[]`` on the wire).
The same plan is what the window model reports and what every Stage 6 call
resolves, edits and refits; a window Stage 6 creates never takes an absorbed
id, and a curation call naming one is ``not_found`` (kind ``"window"``). A fit
made before the fit stored its plan reports the merges its replan record names
(each survivor spanning the union of its windows); Stage 6 refuses to refit
those windows (``curation_conflict``, ``fit_plan_unavailable``).

The Stage 4 product stays the plan as planned: ``load_windows`` and the
``read_table`` tables ``windows``, ``window_free_peaks`` and
``window_contributors`` report the Stage 4 windows whatever a fit later merged.
The windows a fit was made on are reported by ``window_status`` (and by the
fit's own tables); a program that joins fit rows to windows joins them to
``window_status``.

Absence and refusals:

* Before Stage 5, each row's ``n_fitted_peaks`` and ``live`` are
  ``Absent.NOT_RUN``. With a partial fit, the rows of the windows it kept carry
  their counts and the others stay ``Absent.NOT_RUN``.
* Once Stage 5 exists, a window it holds no entry for (for example a created
  window not yet re-fit) reports ``0`` and ``False`` with status ``0``.
* Before Stage 4 it raises ``StageDependencyError`` (``stage_not_run``) with
  ``command`` ``windows run``.

Its columnar form is the ``window_status`` table of ``read_table`` (and of
``read table``), built from the same rows: the seven columns plus
``n_fitted_peaks__status`` and ``live__status`` (``1`` before Stage 5, where
the value columns hold the fill ``0`` / ``False``, which a program must not
read). ``merged_from`` is a text column holding each row's list as JSON
(``"[]"``, ``"[101]"``). Column selection works. Before Stage 4 ``read_table``
raises the same ``StageDependencyError``. The CLI prints the records inline:

.. code-block:: console

   $ ftmwpipeline read window_status experiment.ftmw

The fitted model: ``window_model`` and ``spectrum_model``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``window_model(path, window_id, grid="active", components=False)`` returns
``{"schema": "ftmw/window_model@1", "window_id", "grid", "frame",
"frequency_mhz", "data", "model", "fixed", "baseline", "sigma", "excluded",
"components"}`` over the window's fit range, so a program can draw the fit, and
check it, without reimplementing the line shape. ``frame`` is always ``"raw"``.

* ``model`` is everything the fit compared with the data: the window's fitted
  lines, the frozen neighbours it held fixed, and its baseline. ``fixed`` is the
  frozen neighbours alone, drawn at the window's fitted decay time (as the fit
  draws them); ``baseline`` is the baseline alone, or ``Absent.NOT_RUN`` when
  the window was fitted without one.
* On ``grid="active"`` -- the native active-portion FT Stage 5 fits -- ``data``
  is exactly what the fit compared with, ``sigma`` the per-bin noise it
  weighted by and ``excluded`` the bins it left out (gated spurs), so
  ``sum(|data - model|**2 / (sigma**2 / 2))`` over the bins not excluded is the
  fit's chi-squared. On ``grid="display"`` (``compute_display_ft``'s grid,
  which contains every active bin) ``sigma`` and ``excluded`` are
  ``Absent.UNDEFINED``.
* ``components=True`` adds one array per fitted line, keyed by ``peak_uid``;
  ``model = sum(components) + fixed + baseline``. Unrequested, the field is
  ``Absent.NOT_RUN``.

``spectrum_model(path, grid="active")`` returns ``{"schema":
"ftmw/spectrum_model@1", "grid", "frame", "frequency_mhz", "data", "model",
"residual"}`` over the whole grid: every fitted line once, at its window's
fitted decay time, plus each window's baseline only inside that window's fit
range (the nearest window centre where ranges overlap); ``residual`` is
``data - model``. Inside a window it differs from ``window_model`` by design --
it carries every neighbour's full line rather than the frozen part the fit held.

Both raise ``StageDependencyError`` (``command`` ``fit run``) without a Stage 5
fit; an unknown ``window_id`` raises ``NotFoundError`` (``not_found``).
Amplitudes are in the units of the underlying spectrum, so ``display_units``
applies to both grids. Through the CLI the arrays go to ``.npy`` under
``--output``:

.. code-block:: console

   $ ftmwpipeline read window_model experiment.ftmw 179 --components --output w179/
   $ ftmwpipeline read spectrum_model experiment.ftmw --grid display --output model/

.. _contract-fingerprint:

Analysis identity: ``analysis_fingerprint``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``analysis_fingerprint(path)`` returns ``{"schema":
"ftmw/analysis_fingerprint@1", "digest": "<64 hex>"}``: a SHA-256 digest of
every input that shaped the file's scientific output, as each stage recorded
it. Two files with the same FID samples and the same digest produce the same
results.

.. code-block:: python

   ftmw.analysis_fingerprint("exp.ftmw")
   Pipeline.open("exp.ftmw").analysis_fingerprint()

.. code-block:: console

   $ ftmwpipeline read analysis_fingerprint exp.ftmw

**What it covers.** For every stage (``data``, ``ft``, ``noise``, ``tau``,
``tau_g``, ``timebase``, ``peaks``, ``windows``, ``fit``, ``review``):

* the settings the stage ran with, at the values it resolved (every field of
  its settings record, including values taken from a recommendation, such as a
  detected start time or the Stage 5 shape);
* the values it took from another stage's result (Stage 3's gap-pass decay time
  and shape, Stage 5's decay-time anchor, timebase epsilon, spur nominees and
  survival floor);
* the timebase's knobs, active region and clock declaration;
* the acquisition parameters the analysis read (probe frequency, sideband,
  sample spacing), after any import-time override, and the stored acquisition
  segments (a scope record's pre-record and interleave patterns, which the
  Stage 5 spur gate reads), as content digests;
* the accuracy floor ``sigma_floor_khz`` (``0.0`` when none was declared);
* the clock declaration the final products' calibration state is derived from.
  The products derive it on read, so a ``clocks set`` or ``clocks clear``
  after ``review run`` changes the digest, as it changes the products;
* the analysis epoch each stage was produced under.

A stage that has not run is part of the digest as *not run*, so the digest
changes as stages complete. A stage counts as run only when it is complete:
a stage that was invalidated (for example by ``settings set``) is *not run*
whatever records it left behind. When a completed review has no accuracy floor
on record, the floor is ``0.0``, the value Stage 6 applies in that case. The
Stage 2b shape recommendation is not hashed itself; its verdict is covered
where it was used (the Stage 3 gap-pass shape and the Stage 5 fit shape).

Not covered: the FID samples themselves (hash ``fid_samples`` if you need a
spectrum identity), curation decisions (hash ``review_log`` if you need an
edit-set identity), the attention-routing arguments of ``review run``, write
timestamps, preset names, and the package version and environment (code
changes that move results are marked by the analysis epoch, which is covered).

**The guarantee runs one way.** The same digest means the same results.
Different digests do not imply different results: every recorded input is
hashed, including a knob that another setting made ineffective in a given run,
and nothing is inferred about which knobs mattered.

``@1`` is frozen: its definition never changes. A different definition will be
published as ``@2`` beside it.

**Refusals.** A file written before its stages recorded everything they used
cannot be fingerprinted. The call then raises ``IncompleteProvenanceError``
(code ``incomplete_provenance``) instead of hashing incomplete inputs. Its
``missing`` attribute lists every gap, across all stages, as dotted keys such as
``ft.analysis_epoch``, ``tau.stft``, ``peaks.consumed`` or
``timebase.clock_sources``. Re-running the stages named records them.

Writing a file
--------------

.. _contract-invalidation:

No stale results, and what a run invalidated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``.ftmw`` file never holds a result that disagrees with its own inputs. A
write that would leave one either rebuilds it in the same call or deletes it
together with everything built on it. A client therefore never has to detect
staleness; it only has to notice what was deleted.

- **Rebuilt in place.** The Stage 6 final-products table is the one stored
  result rebuilt rather than deleted. A ``timebase run``, ``set_sigma_floor``
  or a ``clocks`` write (``set``, ``add``, ``remove``, ``clear``, or a reused
  import's loader declaration) that changes the file's calibration rewrites
  the stored table in the same call. The curation record is left alone.
- **Deleted with its dependents.** A changed Stage 1 window or trim deletes
  every stage built on the spectrum. A changed noise recipe deletes Stage 2b
  onward. ``settings set`` and ``settings unset`` delete the stage they
  configure and its dependents. Re-running peaks, windows or the fit deletes
  the stages after it, the review and its decision log included. A forced
  re-import discards every stage the file held. On a file whose Stage 1 record
  predates provenance, a ``start run`` stamp that moves the start that record
  still follows deletes the stages built on the spectrum.
  ``save_peak_parameters`` with different values deletes the peaks it would
  otherwise misdescribe, and their dependents.
- **Unaffected by design.** A timebase change does not invalidate the fit: the
  fit used epsilon only to classify clock spurs and records the value it used.
  Stage 2b and the shape recommendation invalidate nothing: their consumers
  record the decay time and shape they took. An identical re-run (the same
  resolved Stage 1 or noise settings) invalidates nothing.
- **Calls that only read never write.** ``compute_ft(..., from_saved_params=True)``
  recomputes the spectrum from the stored settings without touching the file
  (it does not complete Stage 1 on a file that has none).

**Reporting.** Every stage-running call says which stages it invalidated, as
canonical stage names in re-run order. The list is empty when nothing was
invalidated.

- Results that are dicts carry the key ``"invalidated"`` (a list):
  ``import_data``, and the ``_internal`` stage results the CLI reads.
- Dataclass results carry a field ``invalidated`` (a tuple, default ``()``,
  excluded from equality): ``NoiseResult``, ``TauCalibrationResult``,
  ``ShapeRecommendation``, ``TimebaseCalibrationResult``, ``WindowPlan``,
  ``SpectrumFit``, ``StartDetectionResult``, the Stage 6 results
  (``ReviewRunResult``, ``CurationApplyResult``, ``RefitWindowResult``,
  ``CreateWindowResult``, ``UndoResult``) and ``SetResult`` (from
  ``settings_set`` / ``settings_unset``).
- ``compute_ft`` returns a ``ComplexFT`` whose ``invalidated`` attribute
  carries the list; ``detect_peaks`` returns a ``PeakList``, a ``list`` of
  ``Peak`` with the same attribute. A ``ComplexFT`` on the wire is an object
  of ``freq_array``, ``complex_spectrum``, ``metadata`` and ``invalidated`` (a
  JSON array of canonical names, ``[]`` for a display or loaded spectrum); a
  ``PeakList`` is the plain list of its peaks, without the attribute.
- ``save_ft_parameters`` (``ftmwpipeline.api``) returns the list: the stages
  that saving a changed Stage 1 record discarded (``[]`` when the record did
  not change, or nothing was built on it). It takes no ``events`` argument, so
  no ``Invalidated`` event is delivered; the warning log line names the stages
  as well. ``visualize_ft(save_params=True)`` returns its figure, and names the
  stages in its "Saved N processing parameters" log line.
- On the CLI, a run that invalidated anything prints ``Invalidated (re-run to
  refresh): noise, peaks, ...``; ``settings set`` and ``settings unset`` print
  their own line, with canonical names. Under ``--json`` the list is the
  ``invalidated`` member of the ``ftmw/run_result@1`` envelope
  (:ref:`machine-contract-cli-json`).

A loaded result (``load_fit``, ``load_windows``, ...) carries ``()``: the field
describes the call that produced the object, not the file.

.. _machine-contract-events:

Events and cancellation
~~~~~~~~~~~~~~~~~~~~~~~

Every long operation takes two optional keyword arguments on the API and on
``Pipeline``: ``events``, a callable that receives each event, and ``cancel``,
anything with an ``is_set()`` method (a ``threading.Event`` will do). The long
operations are every stage run (``import_data`` / ``Pipeline.create``,
``detect_start_time``, ``compute_ft``, ``estimate_noise``, ``calibrate_tau``,
``recommend_shape``, ``calibrate_timebase``, ``detect_peaks``,
``assign_windows``, ``fit_peaks``, ``review_run``), the curation calls
(``review_apply``, ``review_preview``, ``review_accept``, ``review_edit``,
``review_create``, ``review_undo``), ``report_run``, ``scan_run``,
``scan_all`` and ``run_pipeline`` (``Pipeline.build``).

The callback runs on the calling thread, never in a worker process. Each event
is a frozen dataclass exported from ``ftmwpipeline`` and serializes through
``to_jsonable``; every one carries ``schema``, ``operation`` (the CLI verb,
such as ``"fit run"``, or ``"run"`` for ``run_pipeline``) and ``stage`` (a
canonical stage name, or ``null`` for a step that is not a stage: start
detection, the report, a scan):

* ``StageStarted`` (``ftmw/stage_started@1``) opens a stage, after the cancel
  check.
* ``StageFinished`` (``ftmw/stage_finished@1``: ``elapsed_s``, ``summary``)
  closes it once its results are written. ``summary`` has the keys of the same
  verb's ``ftmw/run_result@1`` summary.
* ``WindowProgress`` (``ftmw/window_progress@1``: ``phase``, ``round``,
  ``index``, ``total``, ``window_id``, ``n_peaks``, ``chi2r``, ``elapsed_s``,
  ``dropped``) follows each window the Stage 5 walk fits, each window a Stage 6
  call re-fits (``stage: "review"``) and each window the report renders.
  Events come in **passes**, each named by its ``(phase, round)`` pair, and
  within a pass ``index`` counts finished windows from 1 up to ``total``,
  which is fixed when the pass begins. ``phase`` is ``"initial"`` (the fit's
  first walk, a Stage 6 call's directly edited windows, the report's windows),
  ``"replan"`` (a structural replan round), ``"fallback"`` (a sequential
  re-walk after that round's parallel walk fell back, reporting windows the
  round already reported; see ``walk_fallback``) or ``"cascade"`` (Stage 6's
  re-fit of dependent windows). ``round`` is ``0`` for the initial walk, its
  fallback and every Stage 6 pass, and the replan round's number (from 1) for
  a replan round and its fallback.
* ``ScanProgress`` (``ftmw/scan_progress@1``: ``knob``, ``value``, ``index``,
  ``total``) follows each scanned value.
* ``Invalidated`` (``ftmw/invalidated@1``: ``stages``) is emitted once per
  call that drops downstream stages, after the call's write is durable and
  just before ``StageFinished``, and equals the result's ``invalidated``.
* ``PipelineWarning`` (``ftmw/warning@1``: ``code``, ``message`` and the
  code's own fields, flattened beside them; ``capabilities()["fields"]`` lists
  them as ``PipelineWarning.<code>``) with ``code`` one of ``slow_window``,
  ``walk_fallback``, ``epoch_acknowledged``, ``frame_mismatch`` (the curation
  advisory, which stays in the result's ``warnings`` too),
  ``environment_drift`` (once per operation, when the file's recorded
  environment differs from the running one) and ``timebase_skipped``
  (``run_pipeline`` could not calibrate the timebase and went on without it).

A callback that raises aborts the operation with ``callback_failed``. What is
left in the file is what a cancel at that point would leave: a callback that
raises on ``Invalidated`` or ``StageFinished`` -- both delivered after the
write is durable -- leaves the write in place, and no ``StageFinished``
follows; one that raises while ``review_undo`` replays (where a cancel is not
honoured) lets the replay complete and be written, and then fails the call.

``cancel`` is checked before every stage, between the windows of the Stage 5
walk, of a Stage 6 refit and its cascade and of the report's rendering, and
between scan values. A cancel raises ``cancelled``. Every stage the operation
completed stays as written; the interrupted stage leaves the file as it was
before it began -- except the fit, which keeps its finished windows as a
partial fit (:ref:`contract-partial-fits`). A curation batch (``review_apply``,
and every edit with its cascade) is one unit: a cancel discards all of it.
``review_undo`` (and an apply with ``log_prefix``) honours a cancel only before
it restores the automatic fit; once the restore has begun, the replay
completes. A stage that has begun its final write completes, and the cancel is
honoured at the next check point.

The replay of ``review_undo`` (and of an apply with ``log_prefix``) goes one user
action at a time, not one decision at a time: the surviving rows of one action
group (the rows sharing an ``action_index`` in the ``review_log`` evidence)
replay together as one joint refit, so undoing part of a group replays the rest
jointly, and a ``log_prefix`` that cuts through a group replays the in-prefix
rows jointly. A row without the key (a file written before it existed) is
grouped with the row before it when both are ``add``/``remove`` rows on the
same window with identical non-empty evidence and no ``created_window``. The
dry-run plan lists one edit per group. Replaying each row as its own refit
let a later undo fail after a multi-line edit, because the separate refits
drifted the fitted peaks beyond snap tolerance of the line a later remove named.

A Python client that shows progress, stops on request and routes on
``cancelled`` and ``callback_failed``::

   import json
   import threading
   import ftmwpipeline.api as ftmw
   from ftmwpipeline import (
       CallbackFailedError, OperationCancelledError, StageFinished,
       WindowProgress, to_jsonable,
   )

   stop = threading.Event()          # set it from any thread to cancel

   def on_event(event):              # runs on this thread, never in a worker
       if isinstance(event, WindowProgress):
           print(f"window {event.window_id}: {event.index}/{event.total}")
       elif isinstance(event, StageFinished):
           print(event.stage.value, "finished in", f"{event.elapsed_s:.1f} s")
       print(json.dumps(to_jsonable(event)))           # the wire form

   try:
       ftmw.fit_peaks("exp.ftmw", events=on_event, cancel=stop)
   except OperationCancelledError as err:
       # err.stage == "fit" (None between stages); the finished windows are
       # kept as a partial fit, which the next fit_peaks resumes
       print("cancelled", err.stage, err.completed_stages, err.completed_windows)
   except CallbackFailedError as err:
       print("on_event raised while handling", err.event_schema, err.__cause__)

**The whole pipeline.** ``run_pipeline`` reports every stage under
``operation="run"``. A cancel or a failing callback raises, so the stages it
had finished are on the error (``completed_stages``, canonical names), not in a
result dict. Any other failure is a result with ``status == "error"`` whose
``error`` is the failure's ``ftmw/error@1`` dict (a failure that is not a typed
error is reported under ``pipeline_error``) and whose ``failed_stage`` is the
canonical stage of the step that failed (``None`` for start detection and the
report, which are steps, not stages; ``"tau_g"`` for a failing Gaussian tau
step). The new additive ``failed_step`` is the progress label of the failing
step (``"import"``, ``"start detection"``, ``"FT"``, ..., ``"report"``), ``None``
on success. The result's own ``completed_stages`` is the canonical stages
written, in order and each once, the same list a cancel's ``err.completed_stages``
holds; start detection and the report add nothing, and a Gaussian tau run lists
``"tau_g"`` (``"tau"`` only if a twin was built)::

   try:
       result = ftmw.run_pipeline(src, "exp.ftmw", trim=(26500, 40000),
                                  events=on_event, cancel=stop)
   except OperationCancelledError as err:
       print("kept:", err.completed_stages)          # e.g. ["data", "ft", "noise"]
   else:
       if result["status"] == "error":
           print(result["failed_stage"], result["error"]["code"])

**On the command line** every long verb takes ``--events``, which writes each
event to stderr as one JSON line. The first Ctrl-C cancels: the verb stops at
its next check point and exits ``130`` with the ``cancelled`` error (its
``ftmw/error@1`` dict as the last stderr line under ``--json``). A second
Ctrl-C interrupts at once::

   $ ftmwpipeline fit run exp.ftmw --events --json 2> stderr.jsonl > result.json
   $ head -2 stderr.jsonl
   {"schema": "ftmw/stage_started@1", "operation": "fit run", "stage": "fit"}
   {"schema": "ftmw/window_progress@1", "operation": "fit run", "stage": "fit", "phase": "initial", "round": 0, "index": 1, "total": 382, "window_id": 3, "n_peaks": 2, "chi2r": 1.04, "elapsed_s": 0.9, "dropped": false}
   $ # Ctrl-C:
   $ echo $?
   130
   $ tail -1 stderr.jsonl
   {"schema": "ftmw/error@1", "code": "cancelled", "message": "...", "stage": "fit", "completed_stages": [], "completed_windows": [3, 4, 7]}

.. _contract-crash-safety:

Crash safety
~~~~~~~~~~~~

Every call that writes a ``.ftmw`` file -- a stage run, curation, ``settings
set``/``unset``, ``clocks``, ``start run``, a stamp -- writes atomically: it
does all its writes in a temporary copy beside the file and replaces the file
with that copy in one ``os.replace`` when it finishes.

* *Kill.* A process killed at any point, ``SIGKILL`` included, leaves the file
  exactly as it was before the call or as the call completed it, never a mix, a
  stage marked complete over missing or partial results, or a file that will not
  open. A reader that already has the file open keeps the version it opened.
* *Failure.* A cancel, a ``callback_failed`` or any other failure discards the
  copy, so the file is left exactly as it was before the call (a fit that kept
  finished windows writes them as a partial fit in its one replace first). If
  the replace itself fails (a platform that refuses to replace a file another
  process holds open, for example) the call raises and the file is unchanged.
* *Events.* ``Invalidated`` and then ``StageFinished`` are emitted once the
  replace has happened, so an invalidation that never landed is never
  announced. A callback reading the file from inside either one sees the
  call's write.
* *Pipelines.* Within ``run_pipeline`` each stage is its own atomic write, so a
  kill keeps every stage that finished before it.
* *Concurrent writers.* The copy is taken when the call's write begins. If
  another process wrote the file after that, the call raises ``write_conflict``
  (``WriteConflictError``, attribute ``path``, exit ``1``) instead of replacing
  it; the other write stands and nothing of this call is kept. Re-run the call
  to apply it to the file as it now is. Writes from one process to one file are
  serialized.
* *Size.* The copy is written compacted, so the space a write frees is
  reclaimed by the next write.
* *Temporary copies.* The copy is in the target's own directory (so the replace
  stays on one filesystem) and is named
  ``.<target basename>.ftmw-tmp.<hostname>.<pid>``, the basename including its
  extension and ``<pid>`` the writer's decimal process id. A kill between making
  the copy and the replace leaves it behind. Before making its own copy, every
  write removes the leftover copies of the same target made on the same host by
  a process that is no longer running. Copies from other hosts are never
  touched, and neither are copies whose pid is alive, even if that pid has been
  reused. A client that knows no write to the file is in progress may delete
  every file matching the pattern.

.. _contract-partial-fits:

Stage 5 partial fits
~~~~~~~~~~~~~~~~~~~~

A cancel or a ``callback_failed`` during the fit keeps the windows whose whole
per-window pass had run as a partial fit, in one atomic write that also
discards the previous fit and everything downstream of it;
``completed_windows`` (of either error) lists them. The write's invalidations
are delivered as one ``Invalidated`` (``fit`` and everything downstream) once
the write is durable and before the error is raised -- never to a callback that
has just failed. When no window finished, nothing is written and any previous
fit is kept. Nothing is written during the walk, so a killed fit leaves the
file as it was before the call, and a partial fit survives a kill of the run
that resumes it.

While a partial fit is present, ``status`` reports ``fit`` as ``partial`` (and
runnable) and ``review`` as ``not_run``; ``window_status`` rows of the kept
windows carry ``n_fitted_peaks`` and ``live`` (the others stay ``not_run``);
every accessor of the fit or the final products behaves as before Stage 5, and
``review run`` and every curation call refuse with ``stage_not_run``. Anything
that invalidates a complete fit discards a partial one (an upstream re-run,
``settings set`` / ``unset`` of a ``stage5`` setting, a forced re-import; it is
then reported as an invalidated ``fit``); clocks and the timebase leave it
alone.

``fit_peaks`` (``fit run``) resumes a partial fit by default: it fits only the
remaining windows -- ``WindowProgress.index`` continues from the kept count,
``total`` stays the full window count -- then finishes as usual, and the result
equals an uninterrupted fit with the same settings to the same standard as the
parallel and sequential walks (the same lines, ``peak_uid`` values and window
structure; parameters to floating-point rounding). ``restart=True``
(``--restart``) starts over. The ``fit run`` summary (and its
``StageFinished.summary``) carries ``resumed``, ``windows_carried`` (``0``
unless resumed) and ``restart_reason``: ``null`` (a clean resume, or nothing to
resume) or one of the ``restart_reason`` vocabulary -- ``restart_requested``,
``settings_changed`` (the resolved Stage 5 settings, the values consumed from
other stages or ``ANALYSIS_EPOCH`` differ from the partial fit's),
``incomplete_provenance`` (the partial fit lacks what that comparison needs) or
``thaw_refit`` (an accepted thaw: every window is refit sequentially). It never
resumes on a guess. A change in a value the fit consumes from another stage, the
timebase epsilon among them, is a ``settings_changed``.

``run_pipeline`` does not resume a partial fit: it re-imports the source, which
discards it (:doc:`run`).

.. _curation-as-data-contract:

Curation as data: ``CurationAction``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``review_apply`` and ``review_preview`` accept the curation batch as data:
``actions=``, a sequence of :class:`ftmwpipeline.CurationAction` (also exported
from ``ftmwpipeline.contract``), in place of a curation-file path. The file path
remains for people; both spellings take the same validation and frame handling
and give equal results, decision logs and files.

.. code-block:: python

   from ftmwpipeline import CurationAction
   import ftmwpipeline.api as ftmw

   ftmw.review_apply("exp.ftmw", actions=[
       CurationAction("remove", peak_uid=15425022),
       CurationAction("add", freq_mhz=26880.3, frame="raw"),
   ])

A ``CurationAction`` is a frozen dataclass, one curation-file row:

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - Field
     - Type
     - Meaning
   * - ``action``
     - ``"add"`` | ``"remove"`` | ``"accept"`` | ``"create"``
     - The row action.
   * - ``window_id``
     - ``int`` or ``None``
     - The window. ``None`` means "derive it" on ``add`` / ``remove`` and "a
       new window" on ``create``. Required on ``accept``.
   * - ``freq_mhz``
     - ``float`` or ``None``
     - The one frequency: ``add`` / ``remove``, or the ``create`` anchor.
   * - ``peak_uid``
     - ``int`` or ``None``
     - ``remove`` only, in place of ``freq_mhz``.
   * - ``candidate_mhz``
     - ``float`` or ``None``
     - ``accept`` only: a ledger candidate to revive.
   * - ``frame``
     - ``"raw"`` | ``"calibrated"`` or ``None``
     - The frame of ``freq_mhz`` / ``candidate_mhz``. An explicit frame that
       disagrees with an explicit call ``frame=`` is refused.
   * - ``epsilon``
     - ``float`` or ``None``
     - Only with ``frame="calibrated"``: the epsilon the frequency was computed
       under (a file's ``# epsilon:``). It must match the file's current
       epsilon at apply time, else the action is refused as calibration drift.

**Validation.** Construction enforces the parser's rules for a row: one
frequency, or one ``peak_uid`` on ``remove``; no frequency on ``accept``; no
``peak_uid`` outside ``remove``; no ``candidate_mhz`` outside ``accept``. A
violation raises ``BadSettingError`` (``bad_setting``) whose ``path`` is the
field. A dict refused inside an ``actions=`` batch reports the field as
``actions[<i>].<field>``.

**Wire form.** ``to_dict()`` returns ``{"schema": "ftmw/curation_action@1",
"action", "window_id", "freq_mhz", "peak_uid", "candidate_mhz", "frame",
"epsilon"}``. Every key is always present and an unused field is ``null``.
These are request fields, so ``null`` here is "not given", not an ``Absent``
result. ``CurationAction.from_dict()`` is the inverse. It also accepts a dict
without ``schema`` or without unused keys, and refuses an unknown key
(``bad_setting`` naming it). ``to_row()`` returns the action as a curation-file
CSV row. A file's rows parse to the actions ``from_dict`` gives of their dicts.

**Frames.** Each action's frame is resolved on its own. ``None`` takes the
call's ``frame=``, and then the call's rule applies: raw on a file whose
``epsilon`` is 0, and ``bad_setting`` (``path`` ``"frame"``, the call's
missing ``frame=``, with the message naming the action) on a
``self_calibrated`` file when the action carries a frequency. A batch may mix
frames. The pipeline converts each action to raw before resolving anything. A
client never converts frequencies itself.

**Calls.** ``review_apply(path, curation_path=None, *, actions=None, ...)`` and
``review_preview(path, curation_path=None, *, actions=None, frame=None)`` take
exactly one of ``curation_path`` and ``actions``. Both or neither raises
``bad_setting`` with ``path`` ``"actions"``. ``Pipeline`` and the review session
take the same arguments, without the path. ``actions`` may hold
``CurationAction`` objects or their dicts. On the command line, ``review apply``
and ``review preview`` take ``--actions FILE``, a JSON array of action dicts
(``-`` for standard input), in place of the curation CSV:

.. code-block:: console

   $ ftmwpipeline review apply exp.ftmw --actions edits.json --frame raw

**Frames are explicit everywhere.** Every review call that takes a frequency
(``review_edit``'s ``add`` / ``remove``, ``review_create``'s anchor,
``review_accept``'s ``candidate_freq``, and a curation batch) has a typed
``frame`` parameter whose default ``None`` means raw on an ``epsilon == 0``
file and is refused on a ``self_calibrated`` file. The conversion stays
inside the pipeline.

The decision log lives with the fit it edits: re-running ``fit run`` or any
earlier stage discards the review, its decision log and the automatic-fit
baseline (:ref:`contract-invalidation`). To carry curation across such a
re-run, keep the batch (a curation file or its ``actions``) and apply it again.

.. _contract-attention:

Review attention: ``AttentionReason``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``review run`` routes windows to a human. ``get_review_status(path)``
(``Pipeline.review_status()``) returns the review, whose ``window_statuses``
map a window id to a ``WindowReviewStatus``; each status's
``attention_reasons`` are ``AttentionReason`` records with:

* ``kind``, from the frozen vocabulary ``attention_kind``: ``worst_eps``,
  ``auto_merged_review``, ``candidate_bearing``, ``spur_adjacent``,
  ``edge_boundary``, ``flat_decay``, ``empty_window_residual``,
  ``empty_window_spur``. Kinds are only ever added;
* ``severity``, a float; higher asks for a look sooner;
* ``locations``, the molecular frequencies (MHz) the reason points at, empty
  for a window-wide reason;
* ``evidence``, a dict of the kind's declared keys, empty for a kind that
  declares none (only the two empty-window kinds declare any, below);
* ``detail``, a sentence for a person. Its text is not contract.

``auto_merged_review``, ``flat_decay`` and ``empty_window_spur`` are advisory:
they stay on the status but do not put the window in the queue
(``needs_attention``) on their own. Attention is advice: it never changes a
fitted number, a final product or the analysis fingerprint.

On the command line, ``review show --attention --json`` prints ``{"attention":
[...]}``, one row per queued window, worst first, with its top reason
(``window_id``, ``label``, ``kind``, ``severity``, ``detail``) and every reason
in ``reasons`` (``kind``, ``severity``, ``detail``, ``locations``,
``evidence``). ``review show --window N --json`` lists the window's
``attention_reasons`` in the same form.

**A window the fit holds no line in.** The kinds ``empty_window_residual``
and ``empty_window_spur`` cover one case. Stage 5 can finish a window of its plan with no line
(its seeds rejected, gated as spurs or pruned) while the residual on the
window's edge is still coherent. Such a window is flagged when:

* it is a window of the fitted plan, not one Stage 6 created, and no created
  window has taken it over (below);
* the current fit holds no line in it;
* no decision that changes the fit (``add``, ``remove``, ``merge``, ``split``,
  ``create_window``) has been recorded on it;
* Stage 5's residual edge-coherence handshake left one of its edges flagged: a
  structural-replan record the window triggered, or a thaw record of the window
  that was not accepted, with ``S_coh`` above the fit's own
  ``residual_edge_threshold``. An edge a later accepted thaw resolved does not
  count, nor does a replan record measured before the window was last re-fit:
  a structural merge re-fits its survivor and every window that transitively
  depends on it, so a record whose ``revision_before`` precedes that merge's
  ``revision_after`` describes a fit that is gone. The current fit's thaws are
  read before the replan rounds that scanned it.

The kind depends on the Stage 3 peaks the plan put in the window. When every
one sits on a gated spur, the edge residual is consistent with that spur's
skirt beyond its mask: the reason is the advisory ``empty_window_spur``, and its
detail names the spur. Otherwise (a peak off every gated spur, or no peak) a
line may be missing: the reason is ``empty_window_residual``, which queues the
window. Both carry the same fields.

The trigger reads only what Stage 5 recorded, so a fit run with the thaw and
the replan disabled raises none. Its ``severity`` is the strongest flagged
edge's ``S_coh`` divided by the threshold, its ``locations`` the frequencies of
the Stage 3 peaks the plan put in the window, and its ``evidence`` is as
follows. Every record in it carries every key; a missing value is ``Absent``
(``null`` plus ``"<field>_absent"`` on the wire):

* ``edges``: one ``{"side", "s_coh", "neighbour_line_distance_mhz"}`` per
  flagged edge, low first. ``neighbour_line_distance_mhz`` is the distance
  from the edge to the nearest fitted line beyond it, ``undefined`` when the
  fit holds none on that side;
* ``residual_edge_threshold``: the threshold the fit applied;
* ``candidates``: one ``{"detection_index", "frequency_mhz", "snr",
  "gated_spur", "spur_center_mhz", "spur_source"}`` per Stage 3 peak the plan
  put in the window, ascending in frequency:

  .. list-table::
     :header-rows: 1

     * - key
       - value
       - absent when
     * - ``snr``
       - the Stage 3 SNR
       - ``undefined``: Stage 3 recorded none, or it is degenerate (a local
         noise that is not positive, stored by earlier writers as ``0.0``)
     * - ``gated_spur``
       - true when the peak lies within the ``spur_adjacent`` tolerance of a
         spur the fit gated
       - never
     * - ``spur_center_mhz``
       - the nearest such spur's centre
       - ``undefined``: the peak is on no gated spur
     * - ``spur_source``
       - that spur's recorded source (for example ``flat+saturated``)
       - ``undefined``: on no gated spur; ``not_run``: the fit gated the spur
         without recording its source

The fit has no window result for such a window, so ``window_statuses`` can hold
an id ``load_fit`` does not, and ``ReviewRunResult.n_windows`` counts it.
``window_status`` lists it with ``live`` false. ``review show --window N``
reports it with no fitted line and a ``reduced_chi2`` that is
``Absent.UNDEFINED``. The report gives it a page under the ``all`` window
filter, and under ``attention`` only when it is queued.

The item clears when created windows take the window over: a created window
with the same id, or created windows that together cover every flagged Stage 3
peak (every flagged edge, when the window holds none) -- ``review create`` at
the line, then ``review edit`` on the new window to add it. It is dropped in the
batch that creates the covering window, and with it the window's status when
that then carries nothing, reviewed or not. A created window that only overlaps
the empty one leaves the item. ``review accept`` may name the window, alone or
in any batch (with other edits, in ``review preview``, replayed by ``review
undo``), and marks it reviewed as for any kind.

The reference for every name is in :doc:`api/index`.
