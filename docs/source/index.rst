ftmwpipeline Documentation
=========================

Welcome to the ftmwpipeline documentation! This package provides tools for 
processing Fourier Transform Microwave (FTMW) spectroscopy data, including 
baseline estimation, peak detection, window assignment, and advanced fitting algorithms.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index
   examples/index
   changelog

Features
--------

* **Data Loading**: Support for BlackChirp and generic FID formats
* **Preprocessing**: Automated baseline and noise estimation
* **Peak Detection**: Advanced algorithms with clustering and iterative subtraction
* **Window Assignment**: Physics-based analysis window optimization
* **Peak Fitting**: Time-domain and conservative fitting with statistical validation
* **Visualization**: Comprehensive plotting and diagnostic tools
* **Configuration**: Flexible parameter management and algorithm selection

Quick Start
-----------

.. code-block:: python

   import ftmwpipeline as fmw

   # Process a single experiment
   results = fmw.process_experiment('data/experiment.h5')

   # Or use the Pipeline class for more control
   pipeline = fmw.Pipeline()
   results = pipeline.process_experiment('data/experiment.h5')

   # Batch processing
   results = fmw.batch_process_experiments(['exp1.h5', 'exp2.h5'])

Installation
------------

Install from PyPI:

.. code-block:: bash

   pip install ftmwpipeline

Or install from source:

.. code-block:: bash

   git clone https://github.com/ftmw-pipeline/ftmwpipeline.git
   cd ftmwpipeline
   pip install -e .

For development:

.. code-block:: bash

   pip install -e ".[dev,docs,viz]"

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`