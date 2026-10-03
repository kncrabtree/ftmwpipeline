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

The first published contract is version ``1``. Additions (a new accessor,
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

Their Python results are unchanged. Each has a schema name, a constant in
``ftmwpipeline.contract``: ``ftmw/calibration@1``, ``ftmw/snap_tolerance@1``,
``ftmw/metadata@1``, ``ftmw/tables@1``, ``ftmw/table@1``,
``ftmw/settings_defaults@1``, ``ftmw/settings@1``, ``ftmw/final_products@1``,
``ftmw/review_log@1``, ``ftmw/pipeline_info@1`` and ``ftmw/display_ft@1``.
``CalibrationStamp`` and ``FinalProducts`` declare theirs as
``__ftmw_schema__``; the ``read`` verb stamps the rest: a dict result is
stamped directly, a list (``settings_show``, ``settings_defaults``,
``review_log``) becomes ``{"schema", "items": [...]}``, and a scalar
(``refit_snap_tol_mhz``) becomes ``{"schema", "value": x}``.

Each declared accessor, with its absence cases:

* ``read_metadata`` / ``read_tables`` / ``read_table`` -- the persisted scalars
  and table columns, raw. ``tau.`` / ``tau_g.`` / ``timebase.`` keys appear only
  once that calibration has run, so a file without it simply lacks them; an
  unknown table or column is a ``ValueError``. A table whose stage has not run
  raises ``StageDependencyError`` (``stage_not_run``, also a ``ValueError``)
  with ``command`` the verb that produces it (``fit run``, ``windows run``,
  ...). ``read read_table FILE TABLE [--columns a,b]`` writes each column to
  ``<column>.npy`` under ``--output``.
* ``get_final_products`` -- the persisted final-products table, or ``None``
  before Stage 6 (``None`` becomes ``Absent`` in a later wave; ``read
  get_final_products`` already prints ``"items": null, "items_absent":
  "not_run"``). Each
  ``FinalPeak`` carries ``peak_uid``, ``window_id``, ``origin``, ``derivation``,
  ``clock_lattice``, the ``knockout_*`` fields, the frequency and its sigma.
* ``review_log`` -- the ``DecisionLogEntry`` rows in execution order (an empty
  list when nothing was edited). ``kind`` is one of ``add``, ``remove``,
  ``merge``, ``split``, ``accept``, ``create_window``; ``provenance`` is
  ``user``.
* ``get_pipeline_info`` -- the status dict. ``warnings`` is always present (an
  empty list when there are none).
* ``frequency_calibration``, ``refit_snap_tol_mhz``, ``settings_show``,
  ``settings_defaults`` -- unchanged; ``settings_defaults`` needs no file.
* ``compute_display_ft`` -- ``freq_array`` and ``complex_spectrum`` plus
  ``metadata`` (``amplitude_scale``, ``units_label``, ``pad_factor``). It has no
  absence case: a file without Stage 1 raises ``StageDependencyError``.

Every one raises ``PipelineFileNotFoundError`` (code ``not_found``) for a
missing file and ``PipelineCorruptionError`` (``file_corrupt``, exit 2) for a
file that is not HDF5, and none of them writes the file. For example:

.. code-block:: console

   $ ftmwpipeline read compute_display_ft run.ftmw --format json \
         --output ft/ --pad-factor 2

The JSON envelope names ``freq_array.npy`` and ``complex_spectrum.npy`` (a
complex128 array) under ``--output`` and carries ``metadata`` inline; without
``--output`` the verb exits 1, since arrays are never inlined.
Reading the FID: ``fid_samples``
--------------------------------

``fid_samples(path)`` returns the Stage 0 samples exactly as stored:

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
``data import``). There is no
``Absent`` value in this payload.

Display units: ``display_units``
--------------------------------

``display_units(path)`` returns ``{"amplitude_scale": float, "units_label":
str, "units_power": int}`` (schema ``ftmw/display_units@1``):

.. code-block:: console

   $ ftmwpipeline read display_units exp.ftmw

``amplitude_scale`` and ``units_label`` are exactly the pair that
``compute_display_ft`` applies to its spectrum, at every stage, including
before Stage 1 has persisted anything. The value is resolved through the Stage 1
chain (persisted, then the import-time recommendation, then the hard default),
the same chain that selects the spectrum being labelled, so a displayed
magnitude is ``abs(spectrum) * amplitude_scale`` in ``units_label``. The call
only reads. Because the chain ends in a hard default, ``units_power`` is
always an integer in practice; it carries no ``Absent`` marker.
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
accessor reports what the fit applied or says it cannot. A missing file raises
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
       ``command``
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
``"lower"``, whatever the source encodes it as -- and ``shots``). Every field
but ``index`` may be ``Absent`` (on the wire, ``null`` plus its ``_absent``
sibling): a value the source does not declare is *not run*; one that depends
on load-time parameters (a Keysight record's point and shot counts) is
*undefined*. ``chirp_window`` is the
window the source declares (``chirp_start_us``, ``chirp_end_us``,
``start_margin_us``, each possibly absent) or, when it declares none, ``null``
with ``chirp_window_absent: "not_run"``. A path that does not exist raises
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
  HDF5, the embedded attributes). ``spacing_us`` is *not run* when nothing
  declares it; probe frequency, sideband and shots report the import's own
  defaults (``0``, ``"upper"``, ``1``) when the source is silent.
* **Keysight MATLAB** -- one row: spacing is the scope's sampling interval,
  probe frequency ``0`` and sideband ``"upper"``; ``n_points`` and ``shots``
  are *undefined* because they depend on load-time layout parameters.

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

* The result is printed to stdout as JSON that is always strictly valid
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

Serialization rules worth knowing: an enum is written as its ``.value``
(``PeakShape.LORENTZIAN`` is ``"lorentzian"``, also as a mapping key); a
complex number is ``{"real": x, "imag": y}``; a non-finite float with no field
or list to hold it (the top level) is an error rather than a silent ``null``.

Python code can produce the same JSON with
:func:`ftmwpipeline.to_jsonable`, which applies the ``Absent`` rule, stamps a
``schema=`` name, and can hand arrays to a sink instead of inlining them.

The reference for every name is in :doc:`api/index`.
