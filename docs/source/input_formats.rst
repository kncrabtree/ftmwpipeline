.. index::
   single: input formats
   single: bring your own data
   single: data loaders
   single: ftmw-hdf5 format
   single: CSV input
   single: sidecar metadata
   single: clock declaration; input
   single: custom loader

Input Formats
=============

A new analysis begins by importing a raw free-induction decay (FID) into a
``.ftmw`` file. ``ftmwpipeline`` reads several formats, and it provides a
documented, no-code path for data it has never seen: a single column of voltage
samples, or a self-describing HDF5 file, optionally accompanied by a small
metadata file. This page is the reference for those formats and for declaring
instrument clock sources on the imported file.

The mechanics of running an import — the command, the provenance record,
start-time detection — are described on :doc:`stage0_import`. This page is about
the *shape* the data must take.

Three ways in
-------------

#. **A built-in instrument loader.** A native Blackchirp experiment directory or
   a Keysight oscilloscope record is read directly, with the acquisition
   parameters taken from the instrument's own metadata. Raw oscilloscope records
   have their own page, :doc:`scope_record_import`.
#. **A generic format.** Data from any other instrument is imported by shaping
   it into one of two documented layouts — a :ref:`native HDF5 file
   <input-ftmw-hdf5>` or a :ref:`CSV column <input-csv>` — with acquisition
   metadata supplied in the file, as load options, or in a
   :ref:`sidecar file <input-sidecar>`. No code is required.
#. **A custom loader.** A proprietary binary format is supported by
   :ref:`writing a small loader class <input-custom-loader>` and registering it.

Whichever path is used, the result is identical: one FID, with its sample
spacing, probe frequency, sideband, and shot count, written into a new ``.ftmw``
file.

The acquisition metadata
------------------------

Every path resolves the same small set of acquisition parameters. The two
required values have no default; the rest fall back as shown.

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Parameter
     - Default
     - Meaning
   * - ``spacing_us``
     - *required*
     - Sample period in microseconds.
   * - ``probe_freq_mhz``
     - ``0``
     - Probe/local-oscillator frequency in MHz. The default ``0`` describes a
       direct-sampling instrument, for which the baseband frequency *is* the
       molecular frequency.
   * - ``sideband``
     - ``upper``
     - ``upper`` (molecular = probe + baseband) or ``lower``
       (molecular = probe − baseband).
   * - ``shots``
     - ``1``
     - Number of averaged shots; recorded for provenance.

When the same parameter is set in more than one place, the highest-priority
layer wins:

.. code-block:: text

   explicit load option  >  sidecar file  >  embedded in the data file  >  default

This mirrors the resolution philosophy used for stage settings
(:doc:`settings_and_presets`): an explicit value always wins, and a value
carried inside the data file overrides only the built-in default.

.. _input-ftmw-hdf5:

Native HDF5 (``ftmw-hdf5``)
---------------------------

The recommended no-code format is a self-describing HDF5 file. It carries the
samples, the acquisition metadata, and — optionally — the instrument clock
declarations, so everything travels in one portable file and no load options are
needed. Any HDF5 tool can write it: ``h5py``, MATLAB (``-v7.3``), or a C/Fortran
HDF5 library.

The layout (version 1):

.. list-table::
   :header-rows: 1
   :widths: 32 16 52

   * - Object
     - Type
     - Meaning
   * - ``ftmw_input_version`` *(root attr)*
     - int
     - Format marker and version; **required** (currently ``1``).
   * - ``spacing_us`` *(root attr)*
     - float
     - **Required.**
   * - ``probe_freq_mhz`` *(root attr)*
     - float
     - Optional, default ``0`` (a direct sampler).
   * - ``sideband`` *(root attr)*
     - string
     - Optional, default ``"upper"``.
   * - ``shots`` *(root attr)*
     - int
     - Optional, default ``1``.
   * - ``chirp_end_us`` *(root attr)*
     - float
     - Optional; enables a recommended FID start (see
       :ref:`input-chirp-window`).
   * - ``chirp_start_us`` / ``start_margin_us`` *(root attrs)*
     - float
     - Optional companions to ``chirp_end_us``.
   * - ``description`` *(root attr)*
     - string
     - Optional free-text provenance.
   * - ``/fid``
     - float[N]
     - **Required.** One-dimensional real voltage samples.
   * - ``/clock_sources/`` *(group)*
     - —
     - Optional; see :ref:`input-clock-declaration`.

The file is recognized by the ``ftmw_input_version`` attribute, so it is never
confused with an unrelated HDF5 file. A ``/fid`` that is complex is rejected:
the pipeline transform is a real FFT, so store real voltage samples.

.. note::

   MATLAB ``-v7.3`` files are HDF5 but use the ``.mat`` extension, which the
   Keysight loader claims. Write a native-input file with a ``.h5`` (or
   ``.hdf5``) extension, or import it with an explicit ``--format ftmw-hdf5``.

A minimal writer:

.. code-block:: python

   import h5py
   import numpy as np

   volts = ...  # 1-D real array of samples

   with h5py.File("mydata.h5", "w") as f:
       f.attrs["ftmw_input_version"] = 1
       f.attrs["spacing_us"] = 0.02
       f.attrs["probe_freq_mhz"] = 40960.0
       f.attrs["sideband"] = "lower"
       f.attrs["shots"] = 100
       f.create_dataset("fid", data=volts)

.. code-block:: console

   $ ftmwpipeline data import exp.ftmw mydata.h5 --format ftmw-hdf5

Acquisition values embedded in the file are the lowest-priority layer: a load
option (``--spacing_us``, ``--sideband``, …) or a sidecar overrides them, which
is convenient for correcting one field without rewriting the file.

.. _input-csv:

CSV (``csv``)
-------------

The simplest path for data that is already a column of numbers. The CSV carries
only samples; the acquisition metadata is supplied as load options or in a
sidecar.

* **Layout.** One or more columns, with an optional header row. The voltage
  column is chosen with ``--column`` — an integer index (0-based) or, when there
  is a header row, a column name. Omitted, the first column is used; other
  columns are ignored.
* **Required metadata.** ``spacing_us`` must be supplied, because a CSV cannot
  carry it. ``probe_freq_mhz`` is optional and defaults to ``0`` (a direct
  sampler); supply it for a heterodyne instrument.

.. code-block:: console

   # samples in the first column, metadata as options
   $ ftmwpipeline data import exp.ftmw data.csv --format csv \
       --spacing_us 0.02 --probe_freq_mhz 40960

   # select a named column and read metadata from a sidecar
   $ ftmwpipeline data import exp.ftmw data.csv --format csv \
       --column volts --metadata data.meta.json

A single column cannot carry clock declarations; supply those in a
:ref:`sidecar <input-sidecar>` or with the :ref:`clocks command
<input-clock-declaration>` after import.

.. _input-sidecar:

The sidecar metadata file
-------------------------

A sidecar is a small JSON or YAML file holding acquisition metadata and,
optionally, the chirp window and clock declarations. It is the no-code home for
metadata that a format cannot embed (a CSV), and a convenient override layer for
one that can.

.. code-block:: json

   {
     "spacing_us": 0.02,
     "probe_freq_mhz": 40960.0,
     "sideband": "lower",
     "shots": 100,
     "chirp_window": { "chirp_end_us": 1.5, "start_margin_us": 0.5 },
     "clock_sources": [
       { "freq_mhz": 5760.0, "locked": true,  "label": "synth" },
       { "freq_mhz": 6250.0, "locked": false, "label": "digitizer" }
     ]
   }

The sidecar is found in one of two ways: an explicit ``--metadata <path>``, or
auto-discovery of a file named ``<source>.ftmwmeta.json`` (or ``.yaml`` /
``.yml``) next to the data source. Every key is optional, and an unrecognized
key is reported as an error rather than ignored, so a typo surfaces immediately.

.. _input-chirp-window:

The chirp window and start time
-------------------------------

A chirped-pulse experiment records the excitation chirp and switch ring-down
before the molecular signal, so the FID is processed from a start time past the
chirp. When a chirp window is declared — ``chirp_end_us`` with an optional
``start_margin_us`` — the recommended start is set to
``chirp_end_us + start_margin_us`` at import, and the automatic start detector
(described on :doc:`stage0_import`) runs only as a cross-check. Declaring the
chirp end is the reliable way to fix the start when the detector has trouble, for
instance on very high signal-to-noise data.

.. _input-clock-declaration:

Declaring instrument clock sources
----------------------------------

The Stage 5 spur gate can use a declaration of the instrument's clock
*fundamentals* to build a lattice of predicted clock-harmonic frequencies and
recognize spurs that fall on it (see :doc:`clock_declaration`). A clock source
has three fields:

.. list-table::
   :header-rows: 1
   :widths: 18 12 70

   * - Field
     - Type
     - Meaning
   * - ``freq_mhz``
     - float
     - The chain **fundamental** in MHz. Declare 5760, not the 11520 or 40960
       products — harmonics are generated, so the products follow.
   * - ``locked``
     - bool
     - ``true`` for a source referenced to the frequency standard (it spans the
       intermodulation lattice); ``false`` for a free-running source such as an
       unlocked digitizer clock (it predicts a drifting tone family).
   * - ``label``
     - string
     - A human-readable identifier; informational.

There are three no-code ways to declare clocks, all writing the same record:

#. **Embedded** in a native HDF5 file, as a ``/clock_sources`` group of three
   equal-length parallel datasets — ``freq_mhz`` (float), ``locked`` (int,
   1 or 0), and ``label`` (string):

   .. code-block:: python

      g = f.create_group("clock_sources")
      g.create_dataset("freq_mhz", data=[5760.0, 6250.0])
      g.create_dataset("locked", data=[1, 0])
      g.create_dataset("label",
                       data=["synth", "digitizer"],
                       dtype=h5py.string_dtype("utf-8"))

#. **In a sidecar**, as the ``clock_sources`` array shown above.

#. **On an existing file**, with the ``clocks`` command — useful for any format,
   or for revising a declaration after import:

   .. code-block:: console

      $ ftmwpipeline clocks show exp.ftmw
      $ ftmwpipeline clocks set exp.ftmw 5760:locked:synth 6250:free:digitizer
      $ ftmwpipeline clocks add exp.ftmw 16000:locked:awg
      $ ftmwpipeline clocks remove exp.ftmw 6250
      $ ftmwpipeline clocks clear exp.ftmw

   Each token is ``freq[:locked|free[:label]]`` — a fundamental in MHz, an
   optional lock state (default ``locked``), and an optional label. The same
   operation is available on the Python interfaces as
   ``ftmw.set_clock_sources(...)`` / ``ftmw.get_clock_sources(...)`` and the
   matching :class:`~ftmwpipeline.pipeline.Pipeline` methods.

A declaration is a *recommendation*: it is the layer the Stage 5 resolver
consults last, below an explicit fit-time choice and below any clock setting
already persisted in the file. Declaring or editing clocks therefore never
silently rewrites a finished, reproducible fit; re-run the fit for the
declaration to take effect.

.. _input-custom-loader:

Writing a custom loader
-----------------------

A format that cannot be reshaped into the generic layouts — a proprietary binary
container, an instrument with embedded segmentation — is supported by writing a
loader. A loader is a subclass of ``ftmwpipeline.io.data_loaders.BaseLoader``
that implements:

* ``can_load(path)`` — recognize the format (cheaply);
* ``validate_source(path)`` — report structure and available options;
* ``load_fid(path, **params)`` — return a populated ``FID``;
* ``get_required_parameters()`` / ``get_optional_parameters()`` — declare the
  load options.

Registering it makes it available to every interface, including auto-detection:

.. code-block:: python

   from ftmwpipeline.io.data_loaders import BaseLoader, register_loader

   class MyLoader(BaseLoader):
       format_name = "mylab"
       file_extensions = [".dat"]
       ...

   register_loader("mylab", MyLoader)

A loader can attach clock declarations by populating
``fid.metadata["clock_sources"]`` with the field set above, and a chirp window
by populating ``fid.metadata["chirp_window"]``; the import step persists both,
exactly as for the generic formats.

Raw oscilloscope records — segmentation into a pre-record and repeated frames,
ADC interleave-offset cleanup, coherent averaging — are deliberately outside the
generic formats, which accept an already-reduced FID. Those are handled by
dedicated loaders; see :doc:`scope_record_import`.
