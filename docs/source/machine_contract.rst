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

The first published contract is version ``1``; this release is version ``12``. Additions (a new accessor,
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
"schemas": [...], "accessors": [...], "codes": [...], "stages": [...],
"metadata_keys": [...], "tables": {name: [columns]}, "fields": {type: [fields]},
"vocabularies": {name: [values]}, "file_bound": {accessor: bool},
"pipeline_names": {accessor: name}}``
(``stages`` is described under `Stage names`_). Every group of the manifest the
contract tests check is present, in manifest order, so a client can discover
the whole surface without importing the package: ``tables`` and ``fields``
list the declared columns and result-type fields (an event's fields are its
wire keys; a ``PipelineWarning``'s own fields are listed per code under
``PipelineWarning.<code>``, such as ``PipelineWarning.slow_window``),
``vocabularies`` the closed value sets, ``file_bound`` whether each accessor takes a file, and
``pipeline_names`` the ``Pipeline`` method serving each accessor (every
accessor is listed, defaulting to its own name). Every accessor listed
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
``ftmw/pipeline_info@1`` and ``ftmw/display_ft@1``. (``ftmw/curation_action@1``,
also declared, names a request rather than a result: see
:ref:`curation-as-data-contract`.) ``CalibrationStamp`` and
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

.. _machine-contract-cli-json:

Machine-readable CLI output: ``--json``
---------------------------------------

Every CLI verb accepts ``--json``. It is one uniform switch, because
``--format`` already means the report file format (``report``) or the source
format (``data import``, ``run``) on some verbs. Where a verb accepts
``--format json`` today (``info``, ``timebase state``, ``review
snap-tolerance``, ``read table`` / ``read meta``, ``report table``) that stays a
synonym. Under ``--json``:

* stdout carries **exactly one JSON document and nothing else**: the verb's
  human printing is suppressed, and logging stays on stderr. A verb that fails
  after printing a message prints that message on stderr instead.
* every number is finite: JSON has no ``NaN`` or ``Infinity``. A non-finite value
  of a named field is ``null`` with a ``"<field>_absent"`` sibling, as in the
  rest of the contract.
* an error is the ``ftmw/error@1`` dict on stderr (exit code as in the
  exit-code table).

A ``read <accessor>`` verb prints its envelope exactly as it does without the
flag. A **stage-running or curation verb** prints the ``ftmw/run_result@1``
envelope::

    {"schema": "ftmw/run_result@1", "verb": "noise run", "stage": "noise",
     "invalidated": ["tau", "peaks", "windows", "fit", "review"],
     "summary": {"total_points": 152760, "noise_points": 150374, ...}}

``verb`` is ``"<object> <verb>"`` (the canonical object name, never the
``stageN`` synonym). ``stage`` is the canonical stage name (a
``ftmwpipeline.Stage`` value: ``tau_g`` for ``tau run --gaussian``) or ``null``
for a verb that is not one stage (``start run``, ``settings``, ``clocks``,
``report run``, ``run``). ``invalidated`` is the result's own ``invalidated``
(canonical names, in rerun order; ``[]`` when none). ``summary`` holds the
scalars the human output reports (counts, chosen values, paths written, a
dict of counts), never an array. A value with no measurement (an undefined
``epsilon``) is ``null`` with its ``"<field>_absent"`` sibling.

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
       ``frequency_points``, ``trimmed_points`` (when trimmed)
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
     - run_result: the counts the human output reports
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
       ``failed_stage``, ``error``, ``timebase``, ``elapsed_s``, report paths. A
       failed run still prints its envelope, with ``status`` ``"error"``, and
       exits 1
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
       [...]}``, ``{"bar", "window_id", "candidates": [...]}`` or one window's
       detail
   * - ``review rank`` / ``log`` / ``preview`` / ``snap-tolerance`` /
       ``acknowledge-environment``
     - n/a
     - ``{"windows": [...]}`` / ``{"entries": [...]}`` / ``{"warnings",
       "created_windows", "windows"}`` / the snap tolerance / the
       acknowledgement
   * - ``settings show`` / ``defaults``
     - n/a
     - ``{"settings": [...]}``, the rows of the table; ``settings export``
       prints ``{"out_path", "paths"}``
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
     - ``kind`` (``"window"``, ``"peak"``, ``"file"`` or ``"decision"``),
       ``ids`` (every id the request named that does not exist; a peak or
       window named by frequency is reported by that frequency in MHz)
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
   * - ``bad_setting``
     - ``BadSettingError`` (a ``ValueError``)
     - ``path`` (the registry path of the setting, or the argument name),
       ``expected`` (what would have been accepted), ``value`` (what was given)
   * - ``cancelled``
     - ``OperationCancelledError``
     - ``stage`` (the stage interrupted, or ``null`` between stages and for a
       step that is not a stage), ``completed_stages``, ``completed_windows``
       (the windows a cancelled fit kept as a partial fit, sorted; ``[]``
       otherwise); see *Events and cancellation* below
   * - ``callback_failed``
     - ``CallbackFailedError``
     - ``event_schema`` (the event being delivered), ``completed_windows``
       (as for ``cancelled``: the windows an interrupted fit kept as a partial
       fit, sorted; ``[]`` otherwise); the callback's exception is the
       ``__cause__``
   * - ``write_conflict``
     - ``WriteConflictError``
     - ``path`` (the file another process wrote while this call was writing
       it; this call's changes were discarded and the other write stands); see
       *Crash safety* below
   * - ``curation_conflict``
     - ``CurationConflictError`` (a ``ValueError``)
     - ``reason`` (a stable slug, see *Curation refusals* below), ``ids``
       (the windows, peaks or decisions involved, per ``reason``; ``[]`` when
       it names none)
   * - ``pipeline_error``
     - ``PipelineFileError`` (the base class)
     - none. The declared fallback: a direct raise of the base class carries
       it, and ``run_pipeline`` reports a failure that is not a typed error
       under it (see *Events and cancellation* below)

The code set is introduced **wave by wave**. ``capabilities()`` lists the
codes this installation currently implements, and a client should rely on that
list rather than on this page. Every CLI verb reports a typed error through one
mapping in ``main`` (see the exit-code table): ``Error: ...`` text on stderr, or
the error JSON under ``--json`` (or ``--format json`` where a verb takes it).

Route on ``code`` (or the class); the ``message`` text is for people. Each typed
error is still a subclass of the built-in it replaced (most are
``ValueError``; ``NotFoundError`` is a ``KeyError``;
``PipelineFileNotFoundError`` is also a ``FileNotFoundError``;
``PipelineCorruptionError`` is also a ``RuntimeError`` and an ``OSError``), so existing ``except``
clauses keep working. Every typed error pickles, so it survives a process
pool.

**Bad settings.** A refusal of a setting value raises ``BadSettingError``
(``bad_setting``; also a ``ValueError``, with the message text unchanged). The
public calls that raise it:

* ``settings_set`` / ``settings_unset`` (api, ``Pipeline`` and ``settings
  set`` / ``unset``): an unknown or malformed path (``path`` is the knob), a
  value of the wrong type or one that does not parse, a bad ``stage5.shape``
  choice, invalid ``stage5.spur.clocks``, a value outside the ``choices`` or
  ``bounds`` the field's settings row declares, or unsetting a field that is
  not optional. The file is not touched. For a violated declaration ``path`` is
  the knob, ``value`` is what the caller passed (before coercion), and
  ``expected`` states the declaration: ``one of 'a', 'b', ...`` for
  ``choices``, and ``a value in [lo, hi)`` for ``bounds`` (a square bracket
  includes that end, a round one excludes it; an open end reads ``-inf`` /
  ``inf``). Bounds apply to a numeric value, or to every element of a pair or
  list, and ``nan`` is refused wherever bounds exist. ``None`` (the unset
  request) is never checked against either.
* ``set_clock_sources`` and the ``clocks=`` argument of ``fit``: an invalid
  clock declaration (``path`` ``stage5.spur.clocks``); the ``shape=`` argument:
  ``path`` ``stage5.shape``.
* ``preset=`` / ``--preset`` on every stage that takes one (and
  ``settings_show`` / ``settings_defaults``): a preset whose root or block is
  not a mapping, or that names an unknown field or key (``path`` is
  ``preset``, the block name, or ``block.field``).
* Every stage call, when it resolves its settings: a resolved value outside
  the ``choices`` or ``bounds`` its field declares, whichever layer supplied
  it -- a ``settings=`` object, a preset or a persisted record (one written
  before the check existed, or edited by hand). ``path`` is the registry path
  (``stage5.conservative.n_eff_kind``), ``expected`` and the bracket notation
  are as for ``settings_set``, and ``value`` is the resolved value. Only the
  value that wins is checked: a bad value in a layer a higher one overrides is
  never used and never refused, and an unset field falls through to its
  default. ``settings_show`` still displays a bad persisted value, and
  ``settings_set`` repairs it.
* ``trim=`` / ``--trim`` with a value that is not ``min:max`` with
  ``max > min`` (``path`` ``stage1.trim``). On the CLI the option is parsed by
  argparse, which reports a bad ``--trim`` as a usage error (exit 2).
* ``scan_run`` / ``scan_batch`` / ``scan run`` with an unregistered knob
  (``path`` is the knob). This one is also a ``KeyError``, and its ``str()`` is
  the plain message.

Range and choice checks at set time are not yet made: an out-of-range number
is accepted by ``settings_set`` and fails only when a stage runs.

**Curation refusals.** The review calls (``review_edit``, ``review_apply``,
``review_preview``, ``review_create``, ``review_accept``, ``review_undo`` and
their ``Pipeline``, session and CLI spellings) refuse with typed errors. A
batch refusal that names one action keeps its type and adds the action to
the message (``curation action <n> (...) failed: ...``).

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

* ``not_found``:

  Every frequency in ``ids`` is the one the caller wrote, in the frame it
  was written in (a calibrated request is answered in calibrated MHz), so a
  client can match it against its request.

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
         automatic-fit baseline and the file no longer holds it, e.g. the fit
         was re-run after editing (``[]``)
     * - ``replay_conflict``
       - replaying a recorded window creation no longer reproduces its
         window: it would widen another window, or its id is taken (the
         recorded id, then the widened window's id)
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

**Import refusals.** ``import_data`` (and ``Pipeline.create``, ``data
import``) raises ``not_found`` (kind ``"file"``, ``ids`` the source path; a
``PipelineFileNotFoundError``, also a ``FileNotFoundError``) for a source
path that does not exist, and ``bad_setting`` with ``path`` ``"source"`` for a
source the resolved format's loader does not accept.

**Refused stage settings.** A stage call given a value it cannot use raises
``BadSettingError`` (``bad_setting``) before any work starts; ``path`` names the
setting, so a client can point at the offending knob. The message text is
unchanged from the ``ValueError`` it replaced.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Public call
     - ``bad_setting`` ``path``
   * - ``import_data`` (stage-level, auto-detection found nothing or an unknown
       ``format_name``), ``load_fid`` / source validation
     - ``format``; ``source`` when ``import_data``'s source does not validate
       under the resolved format
   * - ``compute_ft``
     - ``stage1.start_us`` (negative), ``stage1.end_us`` (not after ``start_us``, or
       past the end of the recording), ``stage1.trim`` (no data points in the range)
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
     - ``stage5.shape``, ``tau_maj_override_us`` / ``sigma_tau_override_us`` (only one
       of the pair, or non-positive), ``stage5.tau.tau0_us``,
       ``stage5.tau.max_decay_factor`` (must exceed 1)
   * - ``show_fit``
     - ``freqs`` (``value`` lists every frequency in no window at once),
       ``apodize`` / ``apodize_us``; an unknown window id is ``not_found``
       (kind ``window``, every unknown id at once)
   * - ``review_create``
     - ``anchor_mhz`` (outside the analysis band, or already inside a window)
   * - ``run_pipeline``
     - ``stage1.trim`` (not given)

Reading the noise result before ``noise run`` (or ``ft run``) raises
``StageDependencyError`` (``stage_not_run``, ``command`` ``noise run`` /
``ft run``). Typed ``PipelineFileError`` subclasses raised inside the import,
FT and noise stages propagate with their own type instead of being re-wrapped
as ``RuntimeError``.

**Missing and corrupt files.** A path that does not exist raises
``PipelineFileNotFoundError`` (``not_found`` with ``kind`` ``"file"``; also a
``FileNotFoundError``). A path that exists but cannot be opened as a pipeline
file -- not HDF5, unreadable, missing its source metadata -- raises
``PipelineCorruptionError`` (``file_corrupt``; also a ``RuntimeError`` and an
``OSError``, chained from the underlying error). A permission failure, or
HDF5's refusal while another process holds the file open for writing, is not
corruption: it propagates as the original ``OSError`` so a client can retry. ``Pipeline.open``, ``api.validate_pipeline`` and the ``read`` entry points
(``api.read_table``, ``api.read_metadata``, ``read table`` / ``meta`` /
``list``) agree on this. ``validate_pipeline`` reports problems in a file that
opens; for one that does not it raises the open error (``not_found``,
``file_corrupt`` or ``file_incompatible``) instead of returning
``{"valid": False}``, and so does a file that opens but cannot be read while
the report is built: an ``OSError`` from any read -- the FID, the environment
record, the stage-data check -- raises exactly as opening the file would (a
permission failure or HDF5 lock refusal as the original ``OSError``, any other
as ``file_corrupt`` chained from it), and the report never swallows a typed
error. Only integrity problems in a readable file go in the report: an empty or
undecodable FID, a completed stage without its data, and any other non-``OSError``
failure while the report is built (``valid: False`` with a ``Validation
failed`` error). ``Pipeline.validate`` is
``validate_pipeline``, and ``Pipeline.info`` / ``get_pipeline_info`` /
``list_available_stages`` raise for a file they cannot open, so the status
calls and the report cannot disagree. Validation reports problems in a pipeline
file; it does not stand in for opening one.

Every verb that takes a ``.ftmw`` file refuses an unopenable one the same
way, from its first read: a path that does not exist is ``not_found`` (exit
1), a path that is not an HDF5 file is ``file_corrupt`` (exit 2), and a file
from a newer MAJOR format is ``file_incompatible``. This holds for the stage
verbs (``noise``, ``tau``, ``timebase``, ``peaks``, ``windows``, ``fit``),
``review``, ``report``, ``settings`` and ``scan`` as well as ``Pipeline.open``;
none of them lets a raw ``OSError`` escape or re-wraps the refusal as a
``RuntimeError("... failed: Unable to open ...")``. The check is the open and
format-version gate only: it does not demand the provenance record, so a file
that worked before still works.

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
format recognises, raises ``bad_setting`` with ``path`` ``"format"`` (``value``
the name given, or ``null`` when detection found none) -- the same refusal
``import_data`` gives.

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

Typed settings rows: ``settings_show`` and ``settings_defaults``
-----------------------------------------------------------------

Each ``SettingRow`` (and each item of ``settings show`` / ``settings defaults``
``--format json``) carries, besides ``path``, ``value``, ``source``,
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
``settings=`` object, a preset or the file (see *Bad settings* above), so a
``choice`` row's ``choices`` are exactly the strings a run accepts.

``value`` and ``hard_default`` are typed JSON: a pair or list is an array, a
``ShapeSpec`` is ``{"kind": "gaussian"}``, and clock sources are an array of
``{"freq_mhz", "locked", "label"}`` objects. The same JSON is accepted back by
``settings_set`` / ``settings set``, so a row's ``value`` round-trips.

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

The status calls use the same vocabulary, in re-run order: the
``completed_stages`` and ``next_available_stages`` of ``get_pipeline_info`` /
``Pipeline.info`` (and ``ftmwpipeline info``), the result of
``list_available_stages``, and the ``stages`` entry of the
``validate_pipeline`` / ``Pipeline.validate`` report. The report's
``stage_environments`` keys, the stage names inside its drift lines and its
"Missing data for completed stage" errors are canonical too; an environment key
that is not a known stage's storage key is kept as the file recorded it, and a
completed-stage key no stage of this version owns is left out of the canonical
lists. ``read_metadata``'s ``file.completed_stages`` follows the same rule:
canonical names in re-run order.

No stale results, and what a run invalidated
--------------------------------------------

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
  the stages after it. A forced re-import discards every stage the file held.
  On a file whose Stage 1 record predates provenance, a ``start run`` stamp
  that moves the start that record still follows deletes the stages built on
  the spectrum. ``save_peak_parameters`` with different values deletes the
  peaks it would otherwise misdescribe, and their dependents.
- **Unaffected by design.** A timebase change does not invalidate the fit: the
  fit used epsilon only to classify clock spurs and records the value it used.
  Stage 2b and the shape recommendation invalidate nothing: their consumers
  record the decay time and shape they took. An identical re-run (the same
  resolved Stage 1 or noise settings) invalidates nothing.
- **Calls that only read never write.** ``compute_ft(..., from_saved_params=True)``
  recomputes the spectrum from the stored settings without touching the file
  (it does not complete Stage 1 on a file that has none).

**Reporting.** Every stage-running call says which stages it invalidated, as
canonical stage names in re-run order: the topological order of the stage
dependencies with ties broken by the order of ``Stage`` (``data``, ``ft``,
``noise``, ``tau``, ``tau_g``, ``timebase``, ``peaks``, ``windows``, ``fit``,
``review``). The list is empty when nothing was invalidated.

- Results that are dicts gain the key ``"invalidated"`` (a list):
  ``import_data``, and the ``_internal`` stage results the CLI reads.
- Dataclass results gain a field ``invalidated`` (a tuple, default ``()``,
  excluded from equality): ``NoiseResult``, ``TauCalibrationResult``,
  ``ShapeRecommendation``, ``TimebaseCalibrationResult``, ``WindowPlan``,
  ``SpectrumFit``, ``StartDetectionResult`` and the Stage 6 results
  (``ReviewRunResult``, ``CurationApplyResult``, ``RefitWindowResult``,
  ``CreateWindowResult``, ``UndoResult``). ``SetResult.invalidated`` (from
  ``settings_set`` / ``settings_unset``) now uses the same canonical names.
- ``compute_ft`` returns a ``ComplexFT`` whose ``invalidated`` attribute
  carries the list; ``detect_peaks`` returns a ``PeakList``, a ``list`` of
  ``Peak`` with the same attribute. A ``ComplexFT`` on the wire is an object
  of ``freq_array``, ``complex_spectrum``, ``metadata`` and ``invalidated`` (a
  JSON array of canonical names, ``[]`` for a display or loaded spectrum); a
  ``PeakList`` is the plain list of its peaks, without the attribute.
- ``save_ft_parameters`` (``ftmwpipeline.api``) returns the list: the canonical
  names, in re-run order, of the stages that saving a changed Stage 1 record
  discarded (``[]`` when the record did not change, or nothing was built on
  it). It takes no ``events`` argument, so no ``Invalidated`` event is
  delivered; the warning log line names the stages as well.
  ``visualize_ft(save_params=True)`` still returns its figure, and names the
  stages in its "Saved N processing parameters" log line.
- On the CLI, a run that invalidated anything prints
  ``Invalidated (re-run to refresh): noise, peaks, ...``. ``settings set`` and
  ``settings unset`` keep their own line, now with canonical names. Under
  ``--json`` the list is the ``invalidated`` member of the ``ftmw/run_result@1``
  envelope (see :ref:`machine-contract-cli-json`).

A loaded result (``load_fit``, ``load_windows``, ...) carries ``()``: the field
describes the call that produced the object, not the file.

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

``stage_for_key`` / ``key_for_stage`` are the one-to-one inverse; the settings
and knob mappings are not one-to-one and have none.
``capabilities()["stages"]`` lists them all as
``[{"stage", "storage_key", "settings_prefix", "knob_prefix",
"depends_on"}]`` in enum order, with ``depends_on`` in canonical names.

.. _machine-contract-events:

Events and cancellation
-----------------------

Every long operation takes two optional keyword arguments on the API and on
``Pipeline``: ``events``, a callable that receives each event, and ``cancel``,
anything with an ``is_set()`` method (a ``threading.Event`` will do). The long
operations are every stage run (``import_data`` / ``Pipeline.create``,
``detect_start_time``, ``compute_ft``, ``estimate_noise``, ``calibrate_tau``,
``recommend_shape``, ``calibrate_timebase``, ``detect_peaks``,
``assign_windows``, ``fit_peaks``, ``review_run``), the curation calls
(``review_apply``, ``review_preview``, ``review_accept``, ``review_edit``,
``review_create``, ``review_undo``), ``report_run``, ``scan_run``,
``scan_all`` and ``run_pipeline``.

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
  them as ``PipelineWarning.<code>``) with ``code`` one of ``slow_window``, ``walk_fallback``,
  ``epoch_acknowledged``, ``frame_mismatch`` (the curation advisory, which
  stays in the result's ``warnings`` too), ``environment_drift`` (once per
  operation, when the file's recorded environment differs from the running
  one) and ``timebase_skipped``.

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
partial fit (*Stage 5 partial fits* below). A curation batch (``review_apply``, and every edit with its
cascade) is one unit: a cancel discards all of it. ``review_undo`` (and an
apply with ``log_prefix``) honours a cancel only before it restores the
automatic fit; once the restore has begun, the replay completes. A stage that
has begun its final write completes, and the cancel is honoured at the next
check point.

**Crash safety.** Every call that writes a ``.ftmw`` file -- a stage run,
curation, ``settings set``/``unset``, ``clocks``, ``start run``, a stamp --
writes atomically: it does all its writes in a temporary copy beside the file
and replaces the file with that copy in one ``os.replace`` when it finishes.

* *Kill.* A process killed at any point, ``SIGKILL`` included, leaves the file
  exactly as it was before the call or as the call completed it, never a mix, a
  stage marked complete over missing or partial results, or a file that will not
  open. A reader that already has the file open keeps the version it opened.
* *Failure.* A cancel, a ``callback_failed`` or any other failure discards the
  copy, so the file is left exactly as it was before the call (a fit that kept
  finished windows writes them as a partial fit in its one replace first). If
  the replace
  itself fails (a platform that refuses to replace a file another process holds
  open, for example) the call raises and the file is unchanged.
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

On the command line every long verb takes ``--events``, which writes each event
to stderr as one JSON line. The first Ctrl-C cancels: the verb stops at its
next check point and exits ``130`` with the ``cancelled`` error (its
``ftmw/error@1`` dict on stderr under ``--json``). A second Ctrl-C interrupts
at once.

A Python client that shows progress, stops on request and routes on the two new
codes::

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

``run_pipeline`` reports every stage under ``operation="run"``. A cancel
raises, so the stages it had finished are on the error, not in a result dict;
any other failure is a result with ``status == "error"`` whose ``error`` is the
failure's ``ftmw/error@1`` dict (a plain string before contract version 9) and
whose ``failed_stage`` is a canonical stage name::

   try:
       result = ftmw.run_pipeline(src, "exp.ftmw", trim=(26500, 40000),
                                  events=on_event, cancel=stop)
   except OperationCancelledError as err:
       print("kept:", err.completed_stages)          # e.g. ["data", "ft", "noise"]
   else:
       if result["status"] == "error":
           print(result["failed_stage"], result["error"]["code"])

On the command line, ``--events`` writes the same events to stderr, one JSON
line each, and the first Ctrl-C cancels::

   $ ftmwpipeline fit run exp.ftmw --events --json 2> stderr.jsonl > result.json
   $ head -2 stderr.jsonl
   {"schema": "ftmw/stage_started@1", "operation": "fit run", "stage": "fit"}
   {"schema": "ftmw/window_progress@1", "operation": "fit run", "stage": "fit", "phase": "initial", "round": 0, "index": 1, "total": 382, "window_id": 3, "n_peaks": 2, "chi2r": 1.04, "elapsed_s": 0.9, "dropped": false}
   $ # Ctrl-C:
   $ echo $?
   130
   $ tail -1 stderr.jsonl
   {"schema": "ftmw/error@1", "code": "cancelled", "message": "...", "stage": "fit", "completed_stages": [], "completed_windows": [3, 4, 7]}

**Stage 5 partial fits.** A cancel or a ``callback_failed`` during the fit
keeps the windows whose whole per-window pass had run as a partial fit, in one
atomic write that also discards the previous fit and everything downstream of
it; ``completed_windows`` (of either error) lists them. The write's
invalidations are delivered as one ``Invalidated`` (``fit`` and everything
downstream) once the write is durable and before the error is raised -- never
to a callback that has just failed. No window finished: nothing is written,
any previous fit is kept. Nothing is written during the walk, so a killed fit
leaves the file as it was. While a partial fit is present, ``status`` reports
``fit`` as ``partial`` (and runnable) and ``review`` as ``not_run``;
``window_status`` rows of the kept windows carry ``n_fitted_peaks`` and
``live`` (the others stay ``not_run``); every accessor of the fit or the final
products behaves as before Stage 5, and ``review run`` and every curation call
refuse with ``stage_not_run``. Anything that invalidates a complete fit
discards a partial one (an upstream re-run, ``settings set`` / ``unset`` of a
``stage5`` setting, a forced re-import; it is then reported as an invalidated
``fit``); clocks and the timebase leave it alone.

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
timebase epsilon among them, is a ``settings_changed``. Nothing is written during the walk: a process killed
during the fit leaves the file as it was before the call, and a partial fit
survives a kill of the run that resumes it.

Per-stage state: ``status``
---------------------------

``status(path)`` is file-bound on every interface (``ftmw.status(path)``,
``Pipeline.open(path).status()``, ``ftmwpipeline read status FILE``) and never
writes::

   {"schema": "ftmw/status@1",
    "stages": [{"stage": "data", "state": "complete", "depends_on": []}, ...],
    "runnable": ["tau", ...],
    "rerun_order": ["data", "ft", ...]}

``state`` is one of ``complete``, ``partial`` or ``not_run``. ``partial`` is
a Stage 5 fit interrupted by a cancel (or a raising callback) that kept its
finished windows; such a fit is still ``runnable`` (running it resumes it).
``runnable`` lists, in enum order,
the stages that are not complete but whose dependencies all are. ``rerun_order``
lists every stage in the order a full refresh follows: a fixed topological
order, ties broken by the ``Stage`` enum's order (it does not depend on the
file). A file that cannot be opened raises the usual typed error.

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
  for a ``read`` accessor, with the exit code from the table below. Every other
  verb lets a typed error reach one dispatch in ``main``, which applies the same
  mapping: the error JSON under ``--format json``, ``Error: <message>`` on
  stderr otherwise.

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
     - a file that exists but cannot be opened, as detected by the typed opener
       (every verb that takes a ``.ftmw``: ``read`` accessors, ``info``,
       ``data``, ``ft``, ``noise``, ``tau``, ``timebase``, ``peaks``,
       ``windows``, ``fit``, ``review``, ``report``, ``settings``, ``scan``,
       ``clocks``)
   * - ``algorithm_failed``
     - ``2``
     - not yet raised; reserved for a later wave
   * - ``cancelled``, or Ctrl-C
     - ``130``
     - the first Ctrl-C on a long verb cancels it (the ``cancelled`` error);
       a second one interrupts at once
   * - every other code (``not_found``, ``stage_not_run``, ``callback_failed``,
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
"windows": [...]}``, one ``ftmwpipeline.WindowStatusRow`` per window of the
plan the fit was made on and per window Stage 6 created, with ``window_id``,
``freq_min_mhz``, ``freq_max_mhz``, ``created``, ``n_fitted_peaks``, ``live``
and ``merged_from``. A window is **live** when the Stage 5 fit
holds at least one fitted line in it. Rows ascend by ``freq_min_mhz`` and then
``window_id``. A created window that reuses a plan ``window_id`` (the
narrow-gap widening case) replaces that plan row, with its own bounds and
``created`` true.

**Windows after a structural merge.** Before a complete Stage 5 fit (a partial
fit included), and after a fit no structural merge revised, the plan is the
Stage 4 plan. After a merge (``final_plan_revision`` above 0) it is the plan
the fit was made on: the survivor keeps the lower id and the merged range, its
``merged_from`` lists the ids it absorbed (ascending), and an absorbed id has no
row. Every other row's ``merged_from`` is ``[]``. The same plan is what the
window model reports and what every Stage 6 call resolves, edits and refits; a
window Stage 6 creates never takes an absorbed id, and a curation call naming
one is ``not_found`` (kind ``"window"``). A fit made before the fit stored its
plan reports the merges its replan record names (each survivor spanning the
union of its windows); Stage 6 refuses to refit those windows
(``curation_conflict``, ``fit_plan_unavailable``).

Absence and refusals:

* Before Stage 5, each row's ``n_fitted_peaks`` and ``live`` are
  ``Absent.NOT_RUN`` (``null`` plus ``"<field>_absent": "not_run"`` on the
  wire). With a partial fit, the rows of the windows it kept carry their counts
  and the others stay ``Absent.NOT_RUN``.
* Once Stage 5 exists, a window it holds no entry for (for example a created
  window not yet re-fit) reports ``0`` and ``False`` with status ``0``.
* Before Stage 4 it raises ``StageDependencyError`` (``stage_not_run``) with
  ``command`` ``windows run``.
* A path that does not exist raises ``PipelineFileNotFoundError``.

Its columnar form is the ``window_status`` table of ``read_table``, built from
the same rows: the seven columns plus ``n_fitted_peaks__status`` and
``live__status`` (``1`` before Stage 5, where the value columns hold the fill
``0`` / ``False``, which a program must not read). ``merged_from`` is a text
column holding each row's list as JSON (``"[]"``, ``"[101]"``). Column
selection works.
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

.. _curation-as-data-contract:

Curation as data: ``CurationAction``
------------------------------------

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
"epsilon"}``. Every
key is always present and an unused field is ``null``. These are request fields,
so ``null`` here is "not given", not an ``Absent`` result.
``CurationAction.from_dict()`` is the inverse. It also accepts a dict without
``schema`` or without unused keys, and refuses an unknown key (``bad_setting``
naming it). ``to_row()`` returns the action as a curation-file CSV row. A
file's rows parse to the actions ``from_dict`` gives of their dicts.

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
