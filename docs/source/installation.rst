Installation
============

Requirements
------------

ftmwpipeline requires Python 3.9 or later and the following dependencies:

* numpy >= 1.21.0
* scipy >= 1.7.0
* matplotlib >= 3.5.0
* pandas >= 1.3.0
* h5py >= 3.1.0
* pyyaml >= 6.0
* tqdm >= 4.60.0

Installation Methods
--------------------

From PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install ftmwpipeline

From Source
~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/ftmw-pipeline/ftmwpipeline.git
   cd ftmwpipeline
   pip install .

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For development work:

.. code-block:: bash

   git clone https://github.com/ftmw-pipeline/ftmwpipeline.git
   cd ftmwpipeline
   pip install -e ".[dev,docs,viz]"

This installs the package in editable mode with development dependencies.

Optional Dependencies
---------------------

Visualization
~~~~~~~~~~~~~

For enhanced visualization capabilities:

.. code-block:: bash

   pip install ftmwpipeline[viz]

This includes plotly, bokeh, and seaborn.

Documentation
~~~~~~~~~~~~~

For building documentation:

.. code-block:: bash

   pip install ftmwpipeline[docs]

Jupyter Notebooks
~~~~~~~~~~~~~~~~~

For running example notebooks:

.. code-block:: bash

   pip install ftmwpipeline[notebook]

Verifying Installation
----------------------

To verify your installation is working correctly:

.. code-block:: python

   import ftmwpipeline as fmw
   
   # Check validation results
   validation = fmw.workflows.validate_installation()
   print(validation)
   
   # Should show all components as True
   
   # Test basic functionality
   pipeline = fmw.Pipeline()
   print(pipeline.get_summary())

Common Issues
-------------

Missing Dependencies
~~~~~~~~~~~~~~~~~~~~

If you encounter import errors, ensure all required dependencies are installed:

.. code-block:: bash

   pip install --upgrade numpy scipy matplotlib pandas h5py pyyaml tqdm

Version Conflicts
~~~~~~~~~~~~~~~~~~

If you have version conflicts, consider using a virtual environment:

.. code-block:: bash

   python -m venv ftmw_env
   source ftmw_env/bin/activate  # On Windows: ftmw_env\Scripts\activate
   pip install ftmwpipeline

Conda Installation
~~~~~~~~~~~~~~~~~~

For conda users:

.. code-block:: bash

   conda install -c conda-forge numpy scipy matplotlib pandas h5py pyyaml tqdm
   pip install ftmwpipeline