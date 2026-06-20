.. index::
   single: installation
   single: dependencies
   single: conda environment

Installation
============

``ftmwpipeline`` is a Python package (Python 3.9 or later).

From PyPI
---------

The standard installation is from PyPI:

.. code-block:: bash

   pip install ftmwpipeline

This pulls in the runtime dependencies — the scientific Python stack (NumPy,
SciPy, Matplotlib, pandas, h5py, PyYAML, tqdm) and the ``blackchirp`` module
that the Blackchirp data loader uses for format-tolerant reading of
spectrometer data.

Optional dependency groups are available as extras:

* ``[dev]`` — testing, linting, and type-checking tools.
* ``[docs]`` — the Sphinx toolchain for building this documentation.
* ``[viz]`` — additional plotting backends.

.. code-block:: bash

   pip install "ftmwpipeline[viz]"

From source
-----------

To work from a clone — for development, or to track the latest changes — conda
manages the scientific-computing dependencies. Two environment files are
provided, and both install the package itself in editable mode.

The minimal runtime environment:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate ftmwpipeline

The development environment, a superset that adds the test suite, linters, type
checker, documentation tooling, and visualization extras:

.. code-block:: bash

   conda env create -f environment-dev.yml
   conda activate ftmwpipeline-dev

If you manage your own environment, install the package in editable mode
directly:

.. code-block:: bash

   pip install -e ".[dev,docs]"

Verifying the installation
--------------------------

Confirm that the command-line entry point and its dependencies are in place:

.. code-block:: bash

   ftmwpipeline validate
   ftmwpipeline version

``validate`` reports the status of the installation and its dependencies;
``version`` prints the package version and the availability of the optional
extras.
