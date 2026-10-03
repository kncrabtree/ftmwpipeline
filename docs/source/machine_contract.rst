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
exists on the API, on ``Pipeline`` and as a ``read`` verb. An accessor that
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
``compute_display_ft``. ``MANIFEST.cli_verbs`` names the CLI verb that serves
each (an existing verb such as ``review log`` or ``info``, else
``read <name>``), ``MANIFEST.pipeline_names`` the ``Pipeline`` method
(``get_final_products`` is ``Pipeline.final_products``, ``get_pipeline_info``
is ``Pipeline.info``), ``MANIFEST.metadata_keys`` and ``MANIFEST.tables`` the
declared ``read_metadata`` keys and ``read_table`` columns, ``MANIFEST.fields``
the declared fields of the result types (``FinalPeak``, ``DecisionLogEntry``,
``PipelineInfo``, ``ComplexFT`` and its ``metadata``, the ``converged`` flag of
the curation results), and ``MANIFEST.vocabularies`` the frozen
``DecisionLogEntry`` ``kind`` and ``provenance`` values.

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

The contract accessors under ``ftmwpipeline read`` (such as ``read
capabilities``) take ``--format json`` (the only, and default, format),
``-o/--output DIR`` and ``-v``.

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

Serialization rules worth knowing: an enum is written as its ``.value``
(``PeakShape.LORENTZIAN`` is ``"lorentzian"``, also as a mapping key); a
complex number is ``{"real": x, "imag": y}``; a non-finite float with no field
or list to hold it (the top level) is an error rather than a silent ``null``.

Python code can produce the same JSON with
:func:`ftmwpipeline.to_jsonable`, which applies the ``Absent`` rule, stamps a
``schema=`` name, and can hand arrays to a sink instead of inlining them.

The reference for every name is in :doc:`api/index`.
