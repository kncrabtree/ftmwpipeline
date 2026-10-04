.. index::
   single: machine contract
   single: CONTRACT_VERSION
   single: Absent
   single: capabilities
   single: error codes
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

The contract version
--------------------

``ftmwpipeline.CONTRACT_VERSION`` is a single integer. Gate on it, never on
``__version__``::

    import ftmwpipeline
    if ftmwpipeline.CONTRACT_VERSION < 1:
        raise RuntimeError("needs a newer ftmwpipeline")

The first published contract is version ``1``; this release is version ``5``. Additions (a new accessor,
field or code) raise the version by one and never break an existing field. Every machine-readable payload also carries a **schema name**
of the form ``ftmw/<payload>@<n>``; a schema name never changes meaning.

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

   $ ftmwpipeline read capabilities --format json

The payload is ``{"schema": "ftmw/capabilities@1", "contract_version": int,
"schemas": [...], "accessors": [...], "codes": [...]}``. Every accessor listed
exists on the API, on ``Pipeline`` and as exactly one CLI verb,
``ftmwpipeline read <name>``, spelled as the API name. An accessor that
reads a file takes the path as its first argument on the API and as the
``read`` verb's file argument, and is an instance method of an opened
``Pipeline``; one that needs no file (like ``capabilities``) takes no path
anywhere.

Rules every accessor follows
----------------------------

* **One CLI verb per accessor:** ``ftmwpipeline read <name>``, spelled exactly
  as the API name (``read window_status``, ``read read_table``, ``read
  get_pipeline_info``). It prints the accessor's JSON envelope. Verbs that
  predate the contract (``info``, ``review log``, ``settings show``, ...) keep
  their human output and are not contract.
* **The envelope has one of three forms**, by the kind of result: a dict or
  dataclass is *stamped directly* (``{"schema": ..., <fields>}``); a list or
  tuple is *wrapped as items* (``{"schema": ..., "items": [...]}``); a scalar
  is *wrapped as value* (``{"schema": ..., "value": x}``). An absent result is
  an ``Absent`` in a named field, like any other absence: a pre-contract
  accessor that returns ``None`` in Python (``get_final_products`` before
  Stage 6) prints ``{"schema": "ftmw/final_products@1", "value": null,
  "value_absent": "not_run"}``.
* **The payload carries its schema in Python too**, wherever the Python type
  can: a dict payload has a ``"schema"`` key, a dataclass payload declares
  ``__ftmw_schema__``, so the API, ``Pipeline`` and the CLI return the same
  stamped object. The dicts of ``read_metadata``, ``read_tables``,
  ``read_table`` and ``get_pipeline_info`` (and ``ComplexFT``) keep their
  existing Python shape and are stamped by the verb only.
* **Entity tables are lists of records**, not parallel columns: an accessor
  that returns one row per window, FID or line returns a list of dataclasses or
  dicts, so an absent field travels as ``Absent`` per row (``null`` plus its
  ``_absent`` sibling). Numeric series (samples, spectra) are arrays and go to
  ``.npy`` files under ``--output``. The columnar form of a table, with
  ``<column>__status`` columns, belongs to ``read_table``.
* **A refusal names the command to run as a bare CLI verb** (``windows run``,
  ``data import``, ``tau run --gaussian``), never a full shell line.

Declared existing surface
-------------------------

Accessors that predate the contract are declared in
``ftmwpipeline.contract.MANIFEST`` without changing their behaviour:
``frequency_calibration``, ``refit_snap_tol_mhz``, ``read_metadata``,
``read_tables``, ``read_table``, ``settings_defaults``, ``settings_show``,
``get_final_products``, ``review_log``, ``get_pipeline_info`` and
``compute_display_ft``. Each is served on the command line by ``read <name>``
(``read frequency_calibration``, ``read read_table``, ``read
get_pipeline_info``, ...); the older human-facing verbs (``info``, ``review
log``, ``settings show``, ``timebase state``, ``report table``, ``read table`` /
``meta`` / ``list``) keep their output and are not contract.
``MANIFEST.pipeline_names`` names the ``Pipeline`` method
(``get_final_products`` is ``Pipeline.final_products``, ``get_pipeline_info``
is ``Pipeline.info``), ``MANIFEST.metadata_keys`` and ``MANIFEST.tables`` the
declared ``read_metadata`` keys and ``read_table`` columns, ``MANIFEST.fields``
the declared fields of the result types (``FinalPeak``, ``DecisionLogEntry``,
``PipelineInfo``, ``ComplexFT`` and its ``metadata``, the ``converged`` flag of
the curation results), and ``MANIFEST.vocabularies`` the frozen
``DecisionLogEntry`` ``kind`` and ``provenance`` values.

Their Python results keep their types, except that ``get_pipeline_info`` always
carries ``warnings``. Each has a schema name, a constant in
``ftmwpipeline.contract`` (``CALIBRATION_SCHEMA``, ``SNAP_TOLERANCE_SCHEMA``,
...): ``ftmw/calibration@1``, ``ftmw/snap_tolerance@1``, ``ftmw/metadata@1``,
``ftmw/tables@1``, ``ftmw/table@1``, ``ftmw/settings_defaults@1``,
``ftmw/settings@1``, ``ftmw/final_products@1``, ``ftmw/review_log@1``,
``ftmw/pipeline_info@1`` and ``ftmw/display_ft@1``. ``CalibrationStamp`` and
``FinalProducts`` declare theirs as ``__ftmw_schema__``; the ``read`` verb
stamps the rest by the envelope rules above: a dict result is stamped directly,
a list or tuple (``settings_show``, ``settings_defaults``, ``review_log``)
becomes ``{"schema", "items": [...]}``, and a scalar (``refit_snap_tol_mhz``)
becomes ``{"schema", "value": x}``.

Each declared accessor, with its absence cases:

* ``read_metadata`` / ``read_tables`` / ``read_table`` -- the persisted scalars
  and table columns, raw. ``tau.`` / ``tau_g.`` / ``timebase.`` keys appear only
  once that calibration has run, so a file without it simply lacks them; an
  unknown table or column is a ``ValueError``. A table whose stage has not run
  raises ``StageDependencyError`` (``stage_not_run``, also a ``ValueError``)
  with ``missing_dependencies`` the stage group and ``command`` the verb that
  produces it: ``tau run``, ``tau run --gaussian``, ``peaks run``,
  ``windows run`` or ``fit run``, for every table. ``read read_table FILE TABLE [--columns a,b]`` writes each column to
  ``<column>.npy`` under ``--output``.

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
  ``0``. A column that predates the file reads as its fill with status ``1``.
  The rules are the ones ``FinalPeak`` uses for the same quantity.

  * ``fit_peaks``. Knockout: ``knockout_supported``, ``knockout_delta_chi2``,
    ``knockout_expected_delta_chi2``, ``chi_squared``, ``knockout_p_value``,
    ``knockout_n_eff`` and ``knockout_aicc_delta`` are ``1`` when the test did
    not run (no record: ``knockout_supported`` is ``-1`` or the delta chi2 is
    ``nan``); when it ran, a non-finite value is ``2``. ``frequency_error``,
    ``amplitude_error``, ``phase``, ``phase_error``, ``decay_rate``,
    ``decay_rate_error`` (also when tau was held fixed) and ``snr``: a
    non-finite value is ``2``. ``detection_index``: ``-1`` (no Stage 3
    detection seeded the line) is ``2``. ``derivation`` and ``peak_uid``:
    ``-1`` is ``1``. ``unresolved_spread_mhz``: ``nan`` (never a merged
    multiplet) is ``1``. ``clock_lattice``: ``1`` when the fit recorded no
    clock declaration, ``2`` when it did and the line is off the lattice
    (``""``), else ``0``.
  * ``fit_windows``. ``freq_min``, ``freq_max``, ``edge_coherence_low``,
    ``edge_coherence_high``: ``nan`` is ``1``. ``tau_fitted``: ``-1`` is ``1``.
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
    with status ``1``. ``chi2_after`` and ``aic_after``: ``1`` on
    ``spur-drop``, else ``2`` if non-finite.
  * ``fit_doublets``. ``chi2r_merged``, ``delta_chi2_raw``, ``delta_aicc``,
    ``merged_frequency_mhz``, ``merged_amplitude``, ``merged_phase``,
    ``merged_tau_us`` and ``orth_evidence_delta_chi2``: a non-finite value is
    ``2`` (the merged refit was attempted).
  * ``peaks``. ``internal_snr``, ``internal_frequency``, ``leakage_pedestal``:
    ``nan`` (the internal pass did not contribute the peak) is ``1``. ``snr``
    and ``noise_std_local``: non-finite is ``2``. ``index`` ``-1`` and
    ``classification`` ``""`` are ``1``. ``promoted``: ``1`` for every row when
    the file records no promotion cutoff (it predates the record), else ``0``.

  ``read_metadata`` has two kinds of absence. A key of a stage that has not
  run is *omitted* (read with ``.get()``). A key that is present without a
  value is ``Absent`` (``null`` plus ``"<key>_absent"`` on the wire), never
  ``None``:

  - *not run* (the file predates the record): ``file.format_version``,
    ``file.created_with``, ``source.source_path`` / ``format_name`` /
    ``import_timestamp`` / ``source_hash`` (an unset import field),
    ``stage5.acquisition_us`` (the fit recorded none), and
    ``stage3.promotion_min_snr`` / ``stage3.internal_min_snr``.
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
  before Stage 6. The Python return stays ``None``; ``read
  get_final_products`` prints ``"value": null, "value_absent": "not_run"``.
  Each ``FinalPeak`` carries ``peak_uid``, ``window_id``, ``origin``,
  ``derivation``, ``clock_lattice``, the ``knockout_*`` fields, the frequency
  and its sigma. Every field that can lack a value is ``Absent``, never
  ``None`` or ``nan``:

  * ``sigma_stat_khz`` and ``sigma_f_khz`` are *undefined* when the fit left
    the line without a finite frequency error. The total is never computed
    with its statistical term dropped (before this rule it read
    ``sigma_stat_khz`` ``0.0`` and a total without that term).
    ``sigma_eps_khz`` and ``sigma_floor_khz`` are always present.
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
  keeps each value with a ``<field>__status`` code; a table stored before those
  codes existed is decoded field by field by the rules above (its ``None``,
  ``nan`` and ``sigma_stat_khz`` ``0.0`` sentinels), without a rebuild and
  without writing the file.
  It also carries the per-line fit fields, joined from the Stage 5 fit of the
  line's window: ``decay_time_us`` and ``decay_time_error_us`` (the window's
  ``tau`` and its 1-sigma error), ``shape``, ``fwhm_mhz``,
  ``detection_index`` and ``fit_window_mhz``. ``fwhm_mhz`` is exactly
  ``fitting.validation.feature_fwhm(decay_time_us,
  read_metadata(path)["stage5.acquisition_us"], shape=shape)``, so a width a
  client computed that way does not move. ``fit_window_mhz`` is the window's
  ``(low, high)`` in the calibrated frame, like ``frequency_mhz`` (a
  two-element array on the wire); a window created during review reports its
  own bounds. These fields use ``Absent`` from the start:
  ``decay_time_error_us`` is *undefined* when ``tau`` was held fixed,
  ``detection_index`` is *undefined* for a line no Stage 3 detection seeded,
  ``fwhm_mhz`` is *not run* when the fit recorded no
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
  ``null`` with ``"<key>_absent": "undefined"`` on the wire.
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
  "(not recorded)" placeholders, and ``info --format json`` writes them as
  ``null`` plus a ``"<key>_absent"`` sibling.
* ``frequency_calibration`` -- the ``CalibrationStamp``. ``probe_freq_mhz`` and
  ``sideband`` are *not run* when the file carries no Stage 0 FID acquisition
  header to read them from (no frame conversion is possible then); ``state``,
  ``epsilon``, ``sigma_epsilon`` and ``sigma_floor_khz`` are always present,
  with ``state`` saying why ``epsilon`` is ``0.0``.
* ``refit_snap_tol_mhz``, ``settings_show``, ``settings_defaults`` --
  unchanged; ``settings_defaults`` needs no file. ``refit_snap_tol_mhz`` has no
  absence case: a file it cannot resolve raises ``StageDependencyError``.
* The curation results ``RefitWindowResult`` (``review edit`` / ``accept`` /
  merge / split), ``PreviewWindowResult`` (``review preview``) and
  ``AppliedWindowResult`` (``review apply``) carry ``Absent`` the same way:
  ``chi2r_before`` / ``chi2r_after`` are *not run* on a side with no fit (a
  window the batch created has no "before") and *undefined* when the fit's
  value is not finite; ``converged`` is *not run* exactly where
  ``chi2r_after`` is (``RefitWindowResult.converged`` is always a bool); the
  five ``created_window_*`` fields are all *not run* on a window the batch did
  not create or widen (an empty ``created_window_depends_on`` list is a
  present value). ``Absent`` is truthy: test a convergence flag with ``is
  False``, never ``not converged``.
* ``compute_display_ft`` -- ``freq_array`` and ``complex_spectrum`` plus
  ``metadata`` (``amplitude_scale``, ``units_label``, ``pad_factor``). It is
  the active-portion FT Stage 5 fits, zero-filled to ``pad_factor`` times its
  density, from the first to the last active bin inside the Stage 1 trim, so
  it contains every bin the fit sees. It has no absence case: a file without
  Stage 1 raises ``StageDependencyError``.

Every one raises ``PipelineFileNotFoundError`` (code ``not_found``) for a
missing file and ``PipelineCorruptionError`` (``file_corrupt``, exit 2) for a
file that is not HDF5, and none of them writes the file. For example:

.. code-block:: console

   $ ftmwpipeline read compute_display_ft run.ftmw --format json \
         --output ft/ --pad-factor 2

The JSON envelope (stamped ``ftmw/display_ft@1`` by the verb) names
``freq_array.npy`` and ``complex_spectrum.npy`` (a complex128 array) under
``--output`` and carries ``metadata`` inline; without ``--output`` the verb
exits 1, since arrays are never inlined.

Reading the FID: ``fid_samples``
--------------------------------

``fid_samples(path)`` returns the Stage 0 samples exactly as stored, as
``{"schema": "ftmw/fid_samples@1", "samples": ..., "stored_dtype": ...}``:

.. code-block:: python

   res = ftmw.fid_samples("exp.ftmw")      # or Pipeline.open(path).fid_samples()
   res["samples"]       # 1-D float64 ndarray, stored order
   res["stored_dtype"]  # what is on disk, e.g. "float64"

The values equal the stored ones: no scaling, windowing or mean removal. The
pipeline stores ``float64``; a narrower stored dtype is promoted losslessly and
``stored_dtype`` still names the on-disk type. The samples are write-once, so
the array is stable for the life of the file. The call reads one dataset and
never writes. The pipeline publishes no digest of the samples; a client that
wants a spectrum identity hashes the array itself.

.. code-block:: console

   $ ftmwpipeline read fid_samples exp.ftmw --output out/

writes ``out/samples.npy`` and prints ``{"schema": "ftmw/fid_samples@1",
"samples": "samples.npy", "stored_dtype": "float64"}``. Without ``--output`` it
exits ``1``, because the result holds an array.

A file with no Stage 0 FID data raises ``StageDependencyError`` (code
``stage_not_run``, ``missing_dependencies`` ``["data"]``, ``command``
``data import``). There is no ``Absent`` value in this payload.


Display units: ``display_units``
--------------------------------

``display_units(path)`` returns ``{"schema": "ftmw/display_units@1",
"amplitude_scale": float, "units_label": str, "units_power": int}``:

.. code-block:: console

   $ ftmwpipeline read display_units exp.ftmw

``amplitude_scale`` and ``units_label`` are exactly the pair that
``compute_display_ft`` applies to its spectrum, at every stage, including
before Stage 1 has persisted anything. The value is resolved through the Stage 1
chain (persisted, then the import-time recommendation, then the hard default),
the same chain that selects the spectrum being labelled, so a displayed
magnitude is ``abs(spectrum) * amplitude_scale`` in ``units_label``. The call
only reads. Because the chain ends in a hard default, ``units_power`` always
resolves to an ``int``; it carries no ``Absent`` marker.

Reading the fit thresholds: ``fit_thresholds``
----------------------------------------------

``fit_thresholds(path)`` reports the thresholds the persisted Stage 5 fit
actually applied, by name. It reads only the recorded fit diagnostics (no fit is
deserialized, and a read never writes the file).

.. code-block:: python

   ftmw.fit_thresholds("exp.ftmw")           # functional API
   Pipeline.open("exp.ftmw").fit_thresholds()  # Pipeline instance method

.. code-block:: console

   $ ftmwpipeline read fit_thresholds exp.ftmw --format json

The payload is ``{"schema": "ftmw/fit_thresholds@1",
"peak_survival_snr_floor": float, "vif_collapse_threshold": float}``:

* ``peak_survival_snr_floor`` -- the survival SNR floor the fit used (the Stage 3
  promotion cutoff times the survival factor, unless overridden).
* ``vif_collapse_threshold`` -- the variance-inflation threshold of the collapse
  pass.

Either field is ``Absent.NOT_RUN`` (``null`` plus ``"<field>_absent":
"not_run"`` on the wire) when the file has no Stage 5 fit, when the fit predates
the recording of that threshold, or when the fit's peak-survival pass was
disabled (it then recorded neither). A default is never substituted: the
accessor reports what the fit applied or says it cannot. (The figures that grade
old fits for display fall back to the file's own resolved settings; the
accessor does not.) A missing file raises
``PipelineFileNotFoundError`` (code ``not_found``).

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

Compare with ``is``: ``x is Absent.NOT_RUN``. On the wire a field keeps one type
(its value type or ``null``); the sibling key appears only when the value is
absent, and a key ending in ``_absent`` is never anything else. JSON has no
``nan`` or ``inf``: a field whose value is not finite is written as ``null``
with ``"<field>_absent": "undefined"``, and a non-finite element of an inline
JSON list is ``null``. Arrays written as ``.npy`` files keep their ``nan``
values. Array columns that can be absent come with a ``uint8`` column named
``<column>__status``. A setting that is merely unset reads as ``None``; that is
not an absent value.

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
     - ``missing_dependencies`` (canonical stage names, see below),
       ``command`` (``null`` with ``command_absent`` ``not_run`` when the
       refusal names no producing verb; the Python attribute is ``None``)
   * - ``not_found``
     - ``NotFoundError``
     - ``kind``, ``ids`` (every id the request named that does not exist)
   * - ``not_found``
     - ``PipelineFileNotFoundError`` (a ``.ftmw`` path that does not exist)
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

The code set is introduced **wave by wave**. ``capabilities()`` lists the
codes this installation currently implements, and a client should rely on that
list rather than on this page. Today the ``read`` accessors, and ``read table``
/ ``meta`` under ``--format json``, emit the error JSON described below; every
other verb still reports a failure as ``Error: ...`` text with its usual exit
code (the remaining verbs move to the error JSON in a later release).

Route on ``code`` (or the class); the ``message`` text is for people. Each typed
error is still a subclass of the built-in it replaced (most are
``ValueError``; ``NotFoundError`` is a ``KeyError``;
``PipelineFileNotFoundError`` is also a ``FileNotFoundError``;
``PipelineCorruptionError`` is also a ``RuntimeError`` and an ``OSError``), so existing ``except``
clauses keep working. Every typed error pickles, so it survives a process
pool.

**Missing and corrupt files.** A path that does not exist raises
``PipelineFileNotFoundError`` (``not_found`` with ``kind`` ``"file"``; also a
``FileNotFoundError``). A path that exists but cannot be opened as a pipeline
file -- not HDF5, unreadable, missing its source metadata -- raises
``PipelineCorruptionError`` (``file_corrupt``; also a ``RuntimeError`` and an
``OSError``, chained from the underlying error). A permission failure, or
HDF5's refusal while another process holds the file open for writing, is not
corruption: it propagates as the original ``OSError`` so a client can retry. ``Pipeline.open`` and the ``read`` entry points
(``api.read_table``, ``api.read_metadata``, ``read table`` / ``meta`` /
``list``) agree on this.

Previewing a source: ``preview_source``
---------------------------------------

``preview_source(source, format_name=None)`` says what a data source holds
without importing it: the detected format and the source's FID table, one row
per FID the source holds (a Blackchirp experiment may hold many). It takes a
source path, not a ``.ftmw`` file, so it is file-less.

.. code-block:: python

   ftmw.preview_source("exp_2638")          # functional API
   Pipeline.preview_source("exp_2638")      # Pipeline class (static)

.. code-block:: console

   $ ftmwpipeline read preview_source exp_2638 [--source-format blackchirp]

The payload is ``ftmw/source_preview@1``: ``source``, ``format``, ``n_fids``,
``fids`` and ``chirp_window``. ``fids`` is a list of
``ftmwpipeline.FidPreviewRow`` records, one per FID (``index``, ``n_points``,
``spacing_us``, ``probe_freq_mhz``, ``sideband`` -- ``"upper"`` or
``"lower"``, whatever the source encodes it as -- ``shots`` and ``channel``,
the source's channel identifier spelled exactly as import's ``--channel`` takes
it, *not run* for a source without channels). Every field
but ``index`` may be ``Absent`` (on the wire, ``null`` plus its ``_absent``
sibling): a value the source does not declare is *not run*; one that depends
on load-time parameters (a Keysight record's point and shot counts) is
*undefined*. ``chirp_window`` is the
window the source declares (``chirp_start_us``, ``chirp_end_us``,
``start_margin_us``, each a finite number or absent) or, when it declares none,
``null`` with ``chirp_window_absent: "not_run"``. A window the source declares
but the code cannot read (a non-numeric or non-finite value, a block that is not
a mapping, an unparsable Blackchirp ``chirps.csv``) is *undefined*, never *not
run*; the parse error is logged at debug level. A path that does not exist raises
``not_found`` with kind ``"file"``; an unknown ``format_name``, or a source no
format recognises, raises ``not_found`` with kind ``"format"``.

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

``--source-format NAME`` selects the loader instead of auto-detection (the
CLI's ``--format`` is the output format). A source that exists but does not fit
the named format exits ``1`` with a plain-text error, not an error object.
The accessor reads only the source: it creates no ``.ftmw`` file, and
``validate_source`` is unchanged (it still describes FID 0 alone).

Stage names
-----------

Every contract payload that names a stage uses the canonical vocabulary
``ftmwpipeline.Stage``, whose values are the CLI object names: ``data``,
``ft``, ``noise``, ``tau``, ``tau_g``, ``timebase``, ``peaks``, ``windows``,
``fit``, ``review``. ``ftmwpipeline.contract.stage_for_key`` and
``key_for_stage`` map to and from the internal storage keys (for example
``stage1_complex_ft`` is ``ft``). The Python attribute
``StageDependencyError.missing_dependencies`` keeps the internal keys; its
``to_dict()`` publishes the canonical names.

JSON from the command line
--------------------------

The contract accessors under ``ftmwpipeline read`` (one ``read <name>`` per
accessor, such as ``read capabilities``) take ``--format json`` (the only, and
default, format), ``-o/--output DIR`` and ``-v``.

* The result is printed to stdout as one schema-stamped envelope: a dict or
  dataclass result stamped directly, a list or tuple as ``{"schema", "items"}``,
  a scalar as ``{"schema", "value"}`` (see the rules above). It is JSON that is
  always strictly valid
  (non-finite floats follow the ``Absent`` rule above).
* ``--output`` names a **directory** (created if needed). Each array-valued
  field is written there as ``<field path>.npy`` (for example
  ``samples.npy`` or ``components.0.npy``; dtype, shape and byte order are
  self-describing) and the JSON names the file in the array's place. An
  accessor whose result holds arrays refuses to run without ``--output``
  (exit ``1``); a result with no arrays writes nothing and does not create the
  directory. This differs from ``read table`` and ``read meta``, where
  ``--output`` names a single text file.
* A contract error is printed to **stderr** as its ``to_dict()`` JSON, always
  for a ``read`` accessor, with the exit code from the table below. The same
  mapping sets the exit code of ``read table``, ``read meta`` and ``read
  list``, whose printed text is unchanged (``Error: ...``).

.. list-table:: Exit codes
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
     - a file that exists but cannot be opened
   * - ``algorithm_failed``
     - ``2``
     - not yet raised; reserved for a later wave
   * - ``cancelled``, or Ctrl-C
     - ``130``
     - ``cancelled`` is not yet raised; an interrupt exits ``130`` now
   * - every other code (``not_found``, ``stage_not_run``,
       ``file_incompatible``, ...) and any other user error
     - ``1``
     -

A code that appears in this table but not in ``capabilities()["codes"]`` is
reserved and not yet implemented.

.. code-block:: console

   $ ftmwpipeline read capabilities | python -m json.tool

Window status: ``window_status``
--------------------------------

``window_status(path)`` returns ``{"schema": "ftmw/window_status@1",
"windows": [...]}``, one ``ftmwpipeline.WindowStatusRow`` per Stage 4 plan
window and per window Stage 6 created, with ``window_id``, ``freq_min_mhz``,
``freq_max_mhz``, ``created``, ``n_fitted_peaks`` and ``live``. A window is **live** when the Stage 5 fit
holds at least one fitted line in it. Rows ascend by ``freq_min_mhz`` and then
``window_id``. A created window that reuses a plan ``window_id`` (the
narrow-gap widening case) replaces that plan row, with its own bounds and
``created`` true.

Absence and refusals:

* Before Stage 5, each row's ``n_fitted_peaks`` and ``live`` are
  ``Absent.NOT_RUN`` (``null`` plus ``"<field>_absent": "not_run"`` on the
  wire).
* Once Stage 5 exists, a window it holds no entry for (for example a created
  window not yet re-fit) reports ``0`` and ``False`` with status ``0``.
* Before Stage 4 it raises ``StageDependencyError`` (``stage_not_run``) with
  ``command`` ``windows run``.
* A path that does not exist raises ``PipelineFileNotFoundError``.

Its columnar form is the ``window_status`` table of ``read_table``, built from
the same rows: the six columns plus ``n_fitted_peaks__status`` and
``live__status`` (``1`` before Stage 5, where the value columns hold the fill
``0`` / ``False``, which a program must not read); column selection works.
Before Stage 4 ``read_table`` raises the same ``StageDependencyError``. The
read never writes the file. The CLI prints the records inline:

.. code-block:: console

   $ ftmwpipeline read window_status experiment.ftmw

The fitted model: ``window_model`` and ``spectrum_model``
---------------------------------------------------------

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
fit; an unknown ``window_id`` raises ``NotFoundError`` (``not_found``). Neither
writes the file. Amplitudes are in the units of the underlying spectrum, so
``display_units`` applies to both grids. Through the CLI the arrays go to
``.npy`` under ``--output``:

.. code-block:: console

   $ ftmwpipeline read window_model experiment.ftmw 179 --components --output w179/
   $ ftmwpipeline read spectrum_model experiment.ftmw --grid display --output model/

Analysis identity: ``analysis_fingerprint``
-------------------------------------------

``analysis_fingerprint(path)`` returns ``{"schema":
"ftmw/analysis_fingerprint@1", "digest": "<64 hex>"}``: a SHA-256 digest of
every input that shaped the file's scientific output, as each stage recorded
it. Two files with the same FID samples and the same digest produce the same
results. The call only reads.

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
where it was used (the Stage 3 gap-pass shape and the Stage 5 fit shape). Not covered: the FID samples themselves (hash
``fid_samples`` if you need a spectrum identity), curation decisions (hash
``review_log`` if you need an edit-set identity), the attention-routing
arguments of ``review run``, write timestamps, preset names, and the package
version and environment (code changes that move results are marked by the
analysis epoch, which is covered).

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
``timebase.clock_sources``. Re-running the stages named records them. A missing
file raises ``PipelineFileNotFoundError`` (``not_found``). Through the CLI either
error is written to stderr as JSON and the command exits 1.

Serialization rules worth knowing: an enum is written as its ``.value``
(``PeakShape.LORENTZIAN`` is ``"lorentzian"``, also as a mapping key); a
complex number is ``{"real": x, "imag": y}``; a non-finite float with no field
or list to hold it (the top level) is an error rather than a silent ``null``.

Python code can produce the same JSON with
:func:`ftmwpipeline.to_jsonable`, which applies the ``Absent`` rule, stamps a
``schema=`` name, and can hand arrays to a sink instead of inlining them.

The reference for every name is in :doc:`api/index`.
