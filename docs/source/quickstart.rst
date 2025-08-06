Quick Start Guide
=================

This guide will get you started with ftmwpipeline for FTMW spectroscopy data analysis.

Basic Usage
-----------

Simple Processing
~~~~~~~~~~~~~~~~~

The easiest way to process FTMW data:

.. code-block:: python

   import ftmwpipeline as fmw
   
   # Process a single experiment file
   results = fmw.process_experiment('path/to/data.h5')
   
   print(f"Found {len(results['peaks'])} peaks")
   print(f"Processing status: {results['status']}")

Using the Pipeline Class
~~~~~~~~~~~~~~~~~~~~~~~~

For more control over the processing:

.. code-block:: python

   import ftmwpipeline as fmw
   
   # Create pipeline with custom configuration
   config = {
       'peak_detection': {'method': 'hybrid', 'threshold_factor': 4.0},
       'fitting': {'method': 'conservative', 'max_iterations': 200}
   }
   
   pipeline = fmw.Pipeline(config=config)
   results = pipeline.process_experiment('path/to/data.h5')

Batch Processing
~~~~~~~~~~~~~~~~

Process multiple experiments:

.. code-block:: python

   import ftmwpipeline as fmw
   
   data_files = ['exp1.h5', 'exp2.h5', 'exp3.h5']
   
   # Process all files
   results = fmw.batch_process_experiments(data_files)
   
   for i, result in enumerate(results):
       print(f"Experiment {i+1}: {result['status']}")

Step-by-Step Processing
-----------------------

For detailed control, you can run each step individually:

.. code-block:: python

   import ftmwpipeline as fmw
   
   # Load data
   data = fmw.load_blackchirp_data('experiment.h5')
   
   # Estimate baseline and noise
   baseline, noise = fmw.estimate_baseline_noise(
       data['frequencies'], data['intensities']
   )
   
   # Detect peaks
   peaks = fmw.locate_peaks_hybrid(
       data['frequencies'], data['intensities'], 
       noise_level=noise
   )
   
   # Classify peaks by SNR
   classified_peaks = fmw.classify_peaks(peaks)
   
   # Assign analysis windows
   windows = fmw.assign_analysis_windows(classified_peaks)
   
   # Fit peaks in each window
   fitting_results = []
   for window in windows:
       result = fmw.fit_time_domain_peaks(window)
       fitting_results.append(result)

Configuration
-------------

Default Settings
~~~~~~~~~~~~~~~~

ftmwpipeline comes with sensible defaults, but you can customize behavior:

.. code-block:: python

   # View default settings
   from ftmwpipeline.config import DEFAULT_SETTINGS
   print(DEFAULT_SETTINGS)

Custom Configuration
~~~~~~~~~~~~~~~~~~~~

Create a custom configuration:

.. code-block:: python

   config = {
       'preprocessing': {
           'baseline_method': 'polynomial',
           'baseline_order': 3,
           'noise_estimation_method': 'robust_std'
       },
       'peak_detection': {
           'method': 'hybrid',
           'threshold_factor': 3.5,
           'min_separation': 0.05,  # MHz
           'clustering_bandwidth': 0.1
       },
       'window_assignment': {
           'method': 'greedy', 
           'fwhm_factor': 5.0,
           'overlap_threshold': 0.15
       },
       'fitting': {
           'method': 'conservative',
           'max_iterations': 150,
           'convergence_threshold': 1e-7,
           'use_physics_constraints': True,
           'f_test_threshold': 0.05
       }
   }
   
   pipeline = fmw.Pipeline(config=config)

Saving Configuration
~~~~~~~~~~~~~~~~~~~~

Save your configuration for reuse:

.. code-block:: python

   from ftmwpipeline.config import save_config
   
   save_config(config, 'my_config.yaml')
   
   # Load it later
   from ftmwpipeline.config import load_config
   loaded_config = load_config('my_config.yaml')

Visualization
-------------

Basic Plotting
~~~~~~~~~~~~~~

.. code-block:: python

   import ftmwpipeline as fmw
   
   # Process data
   results = fmw.process_experiment('data.h5')
   
   # Plot spectrum with detected peaks
   fmw.visualization.plot_spectrum(
       results['frequencies'], 
       results['intensities'],
       peaks=results['peaks']
   )
   
   # Plot fitting results
   for result in results['fitting_results']:
       fmw.visualization.plot_fit_results(result)

Generate Report
~~~~~~~~~~~~~~~

Create a comprehensive analysis report:

.. code-block:: python

   from ftmwpipeline.visualization import generate_fit_report
   
   report = generate_fit_report(results, output_file='analysis_report.html')

Working with Results
--------------------

Understanding Results
~~~~~~~~~~~~~~~~~~~~~

The results dictionary contains:

.. code-block:: python

   results = {
       'status': 'success',  # Processing status
       'data_path': 'path/to/data.h5',
       'fid_parameters': {...},  # Experimental parameters
       'frequencies': [...],  # Frequency array
       'intensities': [...],  # Intensity array
       'baseline': [...],  # Estimated baseline
       'noise_level': 0.05,  # Estimated noise
       'peaks': [...],  # Detected peaks
       'windows': [...],  # Analysis windows
       'fitting_results': [...],  # Fit results
       'metadata': {...}  # Additional information
   }

Exporting Results
~~~~~~~~~~~~~~~~~

Save results in various formats:

.. code-block:: python

   # Save pipeline cache (recommended for large datasets)
   fmw.io.save_pipeline_cache('experiment_id', complex_ft, noise_result)
   
   # Note: CSV export will be available in a future release
   # For now, use the cache system for result persistence
   
   # Load pipeline cache later
   loaded_cache = fmw.io.load_pipeline_cache('experiment_id')
   complex_ft = loaded_cache['complex_ft']
   noise_result = loaded_cache['noise_result']

Next Steps
----------

* Read the :doc:`api/index` for detailed function documentation
* Check out :doc:`examples/index` for more complex workflows
* See the full :doc:`changelog` for version history

For questions or issues, visit our `GitHub repository <https://github.com/ftmw-pipeline/ftmwpipeline>`_.