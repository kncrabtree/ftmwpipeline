Changelog
=========

This document tracks changes and updates to ftmwpipeline.

Version 0.1.0 (In Development)
-------------------------------

Initial release of ftmwpipeline package.

**Added**

* Core package infrastructure
* Basic Pipeline class for orchestrating analysis
* Placeholder modules for all major components:

  * ``core`` - Data structures (SpectralWindow, Peak, FittingResult)
  * ``preprocessing`` - Data loading and baseline estimation
  * ``peak_detection`` - Peak detection algorithms
  * ``window_assignment`` - Analysis window assignment
  * ``fitting`` - Peak fitting algorithms
  * ``visualization`` - Plotting and diagnostics
  * ``io`` - Input/output and logging
  * ``config`` - Configuration management
  * ``utils`` - Signal processing utilities

* Modern Python packaging with pyproject.toml
* Comprehensive test suite structure with pytest
* Sphinx documentation framework
* Development tools configuration (black, isort, flake8, mypy)

**Features Planned**

* BlackChirp data format support
* Baseline and noise estimation algorithms
* Basic and hybrid peak detection
* Greedy window assignment algorithm
* Time-domain and conservative fitting
* Physics-based parameter validation
* Comprehensive visualization tools
* Batch processing capabilities

**Development Status**

* Phase 1 (Infrastructure): ✅ Complete
* Phase 2 (Core Data Structures): 🔄 In Progress
* Phase 3 (Preprocessing): ⏳ Planned
* Phase 4 (Peak Detection): ⏳ Planned
* Phase 5 (Window Assignment): ⏳ Planned
* Phase 6 (Fitting Algorithms): ⏳ Planned
* Phase 7 (Logging and IO): ⏳ Planned
* Phase 8 (Visualization): ⏳ Planned
* Phase 9 (Configuration): ⏳ Planned
* Phase 10 (Testing): ⏳ Planned
* Phase 11 (Packaging): ⏳ Planned