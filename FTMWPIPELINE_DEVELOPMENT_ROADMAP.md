# FTMW Pipeline Development Roadmap

## Project Overview

Create a standalone Python package `ftmwpipeline` for FTMW spectroscopy signal processing and peak fitting, extracted from the experimental work in `/home/kncrabtree/github/bcfitting/newfitting/`.

## Reference Source Files (bcfitting repository)

**IMPORTANT**: All source files for migration are located under `/home/kncrabtree/github/bcfitting/`

### Core Algorithm Files to Extract From:
- `/home/kncrabtree/github/bcfitting/src/bcfitting/ftmwfitting.py` (lines 1095-2900+)
  - `estimate_baseline_noise()` - Baseline/noise estimation
  - `locate_peaks()` - Basic peak detection  
  - `locate_peaks_hybrid()` - Advanced peak detection
  - `fit_weak_window_conservative_time_domain()` - Conservative fitting algorithm
  
- `/home/kncrabtree/github/bcfitting/newfitting/time_domain_fitting_unified.py`
  - `fit_time_domain_peaks()` - Unified time-domain fitting
  - `validate_fit_results()` - Physics-based validation
  - Complete time-domain modeling with decay constraints

- `/home/kncrabtree/github/bcfitting/newfitting/complex_ft.py`
  - `ComplexFT` class - Data container for spectral data
  - `FitResult` class - Fitting result container
  
- `/home/kncrabtree/github/bcfitting/newfitting/peak_classification.py`
  - `find_and_classify_peaks()` - SNR-based peak classification
  - Peak data structures and classification logic

- `/home/kncrabtree/github/bcfitting/newfitting/window_assignment.py`
  - `assign_analysis_windows()` - Greedy window assignment algorithm
  - Window optimization and boundary calculation

- `/home/kncrabtree/github/bcfitting/newfitting/conservative_fitting_logger.py`
  - `ConservativeFittingLogger` class - Comprehensive logging system
  - JSON logging and decision tracking

- `/home/kncrabtree/github/bcfitting/newfitting/unified_fitting_visualization.py`
  - Visualization functions for fit diagnostics
  - Multi-panel plotting and result analysis

### Utility Files:
- `/home/kncrabtree/github/bcfitting/newfitting/ftmw_utils.py` - Signal processing utilities
- `/home/kncrabtree/github/bcfitting/newfitting/blackchirp-test/` - Experimental data loading

### Test Data and Examples:
- `/home/kncrabtree/github/bcfitting/examples/blackchirp_data/2638/` - Real experimental data
- `/home/kncrabtree/github/bcfitting/newfitting/output/window_assignments.csv` - Window definitions
- `/home/kncrabtree/github/bcfitting/newfitting/test_conservative_fitting.py` - Pipeline integration example

## Package Structure Plan

```
ftmwpipeline/
├── pyproject.toml                 # Modern Python packaging
├── README.md                      # Package documentation  
├── LICENSE                        # MIT License
├── CHANGELOG.md                   # Version history
├── src/
│   └── ftmwpipeline/
│       ├── __init__.py           # Clean public API
│       ├── core/                 # Core data structures
│       │   ├── __init__.py
│       │   ├── data_structures.py    # SpectralWindow, Peak, FittingResult
│       │   ├── fid_parameters.py     # FID parameter management
│       │   └── fit_metrics.py        # Statistical metrics calculation
│       ├── preprocessing/        # Data preparation
│       │   ├── __init__.py
│       │   ├── data_loading.py       # BlackChirp data ingestion
│       │   ├── baseline_estimation.py # Frequency-dependent baseline/noise
│       │   └── data_validation.py    # Input validation
│       ├── peak_detection/       # Peak finding
│       │   ├── __init__.py
│       │   ├── basic_detection.py    # locate_peaks implementation
│       │   ├── hybrid_detection.py   # locate_peaks_hybrid implementation
│       │   └── classification.py     # SNR-based classification
│       ├── window_assignment/    # Analysis windows
│       │   ├── __init__.py
│       │   ├── greedy_assignment.py  # Current algorithm
│       │   └── window_optimization.py # Boundary optimization
│       ├── fitting/              # Peak fitting
│       │   ├── __init__.py
│       │   ├── time_domain.py        # Unified time-domain fitting
│       │   ├── conservative.py       # Conservative iterative fitting
│       │   └── validation.py         # Physics-based validation
│       ├── visualization/        # Optional plotting
│       │   ├── __init__.py
│       │   ├── pipeline_plots.py     # Pipeline stage plots
│       │   ├── fit_diagnostics.py    # Fitting diagnostics
│       │   └── summary_reports.py    # Result summaries
│       ├── io/                   # Input/output
│       │   ├── __init__.py
│       │   ├── experimental_formats.py # Data format readers
│       │   ├── result_serialization.py # Save/load results
│       │   └── logging.py            # Structured logging
│       ├── config/               # Configuration
│       │   ├── __init__.py
│       │   ├── pipeline_config.py    # Pipeline settings
│       │   └── default_settings.py   # Default parameters
│       └── utils/                # Utilities
│           ├── __init__.py
│           ├── signal_processing.py  # Signal processing
│           ├── statistical_tests.py  # F-tests, AIC, etc.
│           └── physics_utils.py      # Physics calculations
├── tests/                        # Test suite
│   ├── unit/                     # Unit tests
│   ├── integration/             # Integration tests
│   ├── performance/             # Benchmarks
│   └── fixtures/                # Test data
├── examples/                     # Usage examples
│   ├── basic_usage.py           # Simple pipeline
│   ├── batch_processing.py      # Multiple experiments
│   └── notebooks/               # Jupyter examples
├── docs/                        # Documentation
│   ├── source/                  # Sphinx source
│   └── api/                     # API documentation
└── scripts/                     # Utility scripts
    ├── benchmark.py             # Performance testing
    └── validate_installation.py # Installation check
```

## Implementation Phases

### Phase 1: Package Infrastructure (2-3 days)
**Status**: ✅ **COMPLETE** (2025-08-01)

**Completed Tasks**:
1. ✅ Create package directory structure with all modules
2. ✅ Set up `pyproject.toml` with modern Python packaging
3. ✅ Create basic `__init__.py` files with placeholder APIs for all modules
4. ✅ Set up pytest testing framework with fixtures and test structure
5. ✅ Initialize Sphinx documentation with API reference and examples
6. ✅ Create conda environment files (`environment.yml`, `environment-dev.yml`)
7. ✅ Implement command-line interface (`ftmwpipeline` command)
8. ✅ Create all placeholder modules and functions for clean imports
9. ✅ Validate installation and test framework functionality
10. ✅ Create development examples and basic usage demonstrations

**Key Files to Create**:
- `pyproject.toml` - Package configuration
- `src/ftmwpipeline/__init__.py` - Main API
- `tests/conftest.py` - Pytest configuration
- `docs/source/conf.py` - Sphinx configuration

### Phase 2: Core Data Structures (1 week)
**Status**: 🔄 **READY TO BEGIN**

**Source Files to Extract From**:
- `/home/kncrabtree/github/bcfitting/newfitting/complex_ft.py`
- `/home/kncrabtree/github/bcfitting/newfitting/peak_classification.py`

**Tasks**:
1. Create `core/data_structures.py`:
   - `SpectralWindow` class (from ComplexFT)
   - `Peak` class (from ClassifiedPeak)
   - `FittingResult` class (from FitResult)
   - `FIDParameters` class (extracted from ComplexFT)

2. Create `core/fit_metrics.py`:
   - Statistical metric calculations (chi-squared, AIC, F-tests)
   - Noise-weighted statistics
   - Confidence interval calculations

3. Comprehensive unit tests for all data structures

### Phase 3: Preprocessing Pipeline (1 week)  
**Status**: Pending

**Source Files to Extract From**:
- `/home/kncrabtree/github/bcfitting/src/bcfitting/ftmwfitting.py` (estimate_baseline_noise)
- `/home/kncrabtree/github/bcfitting/newfitting/ftmw_utils.py`
- `/home/kncrabtree/github/bcfitting/newfitting/blackchirp-test/`

**Tasks**:
1. Create `preprocessing/baseline_estimation.py`:
   - Extract `estimate_baseline_noise()` from ftmwfitting.py
   - Implement frequency-dependent baseline/noise estimation
   - Add robust outlier detection and filtering

2. Create `preprocessing/data_loading.py`:
   - BlackChirp data format reader
   - Generic FID data loading interface
   - FID parameter extraction and validation

3. Create `preprocessing/data_validation.py`:
   - Input data sanity checks
   - FID parameter validation
   - Frequency range and resolution checks

### Phase 4: Peak Detection (1 week)
**Status**: Pending

**Source Files to Extract From**:  
- `/home/kncrabtree/github/bcfitting/src/bcfitting/ftmwfitting.py` (locate_peaks, locate_peaks_hybrid)
- `/home/kncrabtree/github/bcfitting/newfitting/peak_classification.py`

**Tasks**:
1. Create `peak_detection/basic_detection.py`:
   - Extract `locate_peaks()` from ftmwfitting.py
   - Second derivative-based peak finding
   - Threshold-based peak selection

2. Create `peak_detection/hybrid_detection.py`:
   - Extract `locate_peaks_hybrid()` from ftmwfitting.py  
   - Advanced clustering and iterative subtraction
   - Multi-pass peak detection with refinement

3. Create `peak_detection/classification.py`:
   - Extract from peak_classification.py
   - SNR-based peak classification (strong/medium/weak)
   - Peak validation and filtering

### Phase 5: Window Assignment (1 week)
**Status**: Pending

**Source Files to Extract From**:
- `/home/kncrabtree/github/bcfitting/newfitting/window_assignment.py`
- `/home/kncrabtree/github/bcfitting/newfitting/unwindowed_peak_analysis.py`

**Tasks**:
1. Create `window_assignment/greedy_assignment.py`:
   - Extract greedy window assignment algorithm
   - Physics-based window sizing (FWHM calculations)
   - Peak grouping and window optimization

2. Create `window_assignment/window_optimization.py`:
   - Window boundary refinement
   - Overlap resolution
   - Coverage optimization

### Phase 6: Fitting Algorithms (2 weeks)
**Status**: Pending - Most Complex Phase

**Source Files to Extract From**:
- `/home/kncrabtree/github/bcfitting/newfitting/time_domain_fitting_unified.py`
- `/home/kncrabtree/github/bcfitting/src/bcfitting/ftmwfitting.py` (fit_weak_window_conservative_time_domain)

**Tasks**:
1. Create `fitting/time_domain.py`:
   - Extract `fit_time_domain_peaks()` from time_domain_fitting_unified.py
   - Unified damped cosine fitting with shared parameters
   - Complex spectrum fitting with LSB correction
   - Parameter constraint handling

2. Create `fitting/conservative.py`:
   - Extract `fit_weak_window_conservative_time_domain()` from ftmwfitting.py
   - Iterative peak addition with statistical validation
   - F-test and AIC-based model selection
   - Peak separation and significance testing

3. Create `fitting/validation.py`:
   - Extract `validate_fit_results()` from time_domain_fitting_unified.py
   - Physics-based parameter validation
   - Frequency bounds, decay rate, and amplitude checks
   - Post-fit quality assessment

### Phase 7: Logging and IO (3-4 days)
**Status**: Pending

**Source Files to Extract From**:
- `/home/kncrabtree/github/bcfitting/newfitting/conservative_fitting_logger.py`

**Tasks**:
1. Create `io/logging.py`:
   - Extract ConservativeFittingLogger class
   - JSON-based decision logging
   - Structured algorithm tracking

2. Create `io/result_serialization.py`:
   - HDF5, JSON, and CSV export formats
   - Result loading and caching
   - Metadata preservation

3. Create `io/experimental_formats.py`:
   - BlackChirp format support
   - Generic FID data readers
   - Format auto-detection

### Phase 8: Visualization (3-4 days)
**Status**: Pending

**Source Files to Extract From**:
- `/home/kncrabtree/github/bcfitting/newfitting/unified_fitting_visualization.py`

**Tasks**:
1. Create `visualization/fit_diagnostics.py`:
   - Extract visualization functions
   - Multi-panel fit diagnostics
   - Residual analysis plots

2. Create `visualization/pipeline_plots.py`:
   - Pipeline stage visualization
   - Peak detection and classification plots
   - Window assignment visualization

3. Create `visualization/summary_reports.py`:
   - HTML and PDF report generation
   - Batch processing summaries
   - Performance metrics visualization

### Phase 9: Configuration and Pipeline (2-3 days)
**Status**: Pending

**Tasks**:
1. Create `config/pipeline_config.py`:
   - YAML/JSON configuration management
   - Parameter validation and defaults
   - Algorithm selection and tuning

2. Create main `Pipeline` class:
   - High-level orchestration
   - Step-by-step and end-to-end processing
   - Error handling and recovery

3. Create convenience functions:
   - One-line processing functions
   - Batch processing utilities
   - Common workflow shortcuts

### Phase 10: Testing and Documentation (1 week)
**Status**: Pending

**Test Data Source**:
- `/home/kncrabtree/github/bcfitting/examples/blackchirp_data/2638/`
- `/home/kncrabtree/github/bcfitting/newfitting/output/`

**Tasks**:
1. Comprehensive unit tests (target >90% coverage)
2. Integration tests with real experimental data
3. Performance benchmarks vs. current implementation
4. Complete API documentation with Sphinx
5. User guide with tutorials and examples

### Phase 11: Packaging and Release (2-3 days)
**Status**: Pending

**Tasks**:
1. PyPI package preparation and upload
2. Conda package creation
3. GitHub releases with automated CI/CD
4. Installation guides and quick-start documentation

## Success Criteria

### Functional Requirements
- [ ] Successfully processes BlackChirp experimental data
- [ ] Reproduces results from current bcfitting/newfitting pipeline
- [ ] Handles 150+ weak spectral windows robustly
- [ ] Provides comprehensive fit diagnostics and validation
- [ ] Supports batch processing of multiple experiments

### Performance Requirements  
- [ ] Processing time within 10% of current implementation
- [ ] Memory usage scales linearly with data size
- [ ] Handles large datasets (>1M frequency points) efficiently

### Quality Requirements
- [ ] >90% test coverage across all modules
- [ ] Comprehensive API documentation
- [ ] User guide with working examples
- [ ] Professional packaging and distribution

### Integration Requirements
- [ ] Clean API for integration with bcfitting
- [ ] Standalone command-line interface
- [ ] Jupyter notebook compatibility
- [ ] Configurable algorithm selection

## Key Implementation Notes

1. **Data Source**: All source code is under `/home/kncrabtree/github/bcfitting/`
2. **Experimental Data**: Use `/home/kncrabtree/github/bcfitting/examples/blackchirp_data/2638/` for testing
3. **Reference Results**: Compare against existing JSON logs in `/home/kncrabtree/github/bcfitting/newfitting/output/fitting_logs/`
4. **Algorithm Validation**: Ensure statistical tests (F-tests, AIC) match current implementation
5. **Physics Constraints**: Preserve all physics-based validation from time_domain_fitting_unified.py

## Current Status

- **Repository**: ✅ Created with complete package infrastructure
- **Development Phase**: Phase 2 (Core Data Structures) - Ready to begin
- **Reference Code**: Available in `/home/kncrabtree/github/bcfitting/`
- **Test Data**: Available and validated
- **Timeline**: Estimated 5-6 weeks remaining for complete implementation

## Next Immediate Steps

1. ✅ ~~Create new directory for ftmwpipeline package~~
2. ✅ ~~Initialize git repository~~
3. ✅ ~~Set up package structure with pyproject.toml~~
4. ✅ ~~Begin Phase 1: Package Infrastructure~~
5. **Begin Phase 2**: Extract and implement core data structures from bcfitting codebase

---

**Last Updated**: 2025-08-01  
**Status**: Phase 1 Complete - Ready for Phase 2