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
│       │   ├── spectrum_visualization.py # Spectrum and ComplexFT plotting
│       │   ├── noise_visualization.py   # Noise estimation diagnostics
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
**Status**: ✅ **COMPLETE** (2025-08-02)

**Completed Tasks**:
1. ✅ Created comprehensive `core/data_structures.py` with modern architecture:
   - `FTMWData` - Top-level experiment container
   - `FID` - Time-domain data with BlackChirp-compatible processing
   - `ComplexFT` - Frequency-domain data with real FFT implementation
   - `SpectralWindow` - Analysis subset of ComplexFT for targeted fitting
   - `Peak` - Pre-fitting detected peak representation
   - `FittedPeak` - Post-fitting results with uncertainties
   - `FittingResult` - Flexible container for fitting analysis
   - `FIDProcessingParameters` - Complete processing configuration

2. ✅ Implemented BlackChirp data loading (`io/experimental_formats.py`):
   - `load_blackchirp_experiment()` - Complete experiment loading
   - `load_blackchirp_fid()` - FID-specific loading with parameter conversion
   - Base-36 integer conversion and voltage scaling
   - Processing parameter translation and metadata extraction

3. ✅ Added interactive visualization (`visualization/spectrum_visualization.py`):
   - `plot_complex_ft()` - Dual backend (matplotlib/plotly) plotting
   - Two-panel layout (magnitude + real/imaginary) with wide aspect ratio
   - `plot_spectral_window()` - Window visualization with peak annotations
   - `plot_complex_ft_from_cache()` - Cache-based visualization without recomputation
   - Validated with real experimental data (375k frequency points)

4. ✅ Enhanced ComplexFT with analysis methods:
   - `trim_to_range()` - Frequency range trimming for noise removal
   - `extract_window()` - Spectral window creation
   - Lazy computation and caching for performance

5. ✅ Added example experimental data:
   - BlackChirp experiment 2638 (750k FID points, 15 μs duration)
   - Processing guidance and best practices documented in CLAUDE.md
   - Recommended parameters: zpf=1, expf_us=5.0, activity region 26.5-40 GHz

**Key Architecture Decisions**:
- Hierarchical design: FTMWData → FID/ComplexFT → SpectralWindow → Peak
- Clear separation: Peak (pre-fitting) vs FittedPeak (post-fitting)  
- Real FFT processing (rfft/rfftfreq) for computational efficiency
- BlackChirp compatibility with generic extensibility
- Interactive visualization ready for pipeline development

**Validation**: All components tested with real experimental data

### Phase 3: Preprocessing Pipeline (1 week)  
**Status**: ✅ **SUBSTANTIALLY COMPLETE** (2025-08-04)

**Source Files Extracted From**:
- `/home/kncrabtree/github/bcfitting/src/bcfitting/ftmwfitting.py` (estimate_baseline_noise)
- `/home/kncrabtree/github/bcfitting/newfitting/ftmw_utils.py`
- `/home/kncrabtree/github/bcfitting/newfitting/blackchirp-test/`

**Completed Tasks**:
1. ✅ **Advanced Noise Estimation** (`preprocessing/baseline_estimation.py`):
   - ✅ Extracted and completely redesigned `estimate_baseline_noise()` algorithm
   - ✅ **Statistical rigor**: Implemented Z-test and F-test for subdivision decisions
   - ✅ **Adaptive binning**: Recursive binary subdivision based on post-filtering statistics
   - ✅ **Performance optimization**: Comprehensive caching reduces redundant calculations
   - ✅ **User-friendly parameters**: MHz-based smoothing window, verbose debugging flag
   - ✅ **Robust filtering**: Skewness-based noise identification with configurable targets
   - ✅ **Result**: Creates ~26 statistically distinct bins (vs 32 uniform bins originally)

2. ✅ **Data Loading Infrastructure** (`io/experimental_formats.py`):
   - ✅ Complete BlackChirp data format reader with FT processing
   - ✅ FID parameter extraction and processing (zpf, exponential filtering)
   - ✅ Frequency trimming and range selection capabilities
   - ✅ Integration with experiment 2638 test data

3. ✅ **Visualization and Diagnostics** (`visualization/noise_visualization.py`):
   - ✅ Comprehensive noise estimation diagnostic plots
   - ✅ Statistical threshold visualization (3×RMS, 5×RMS significance levels)
   - ✅ Adaptive bin boundary visualization
   - ✅ Both matplotlib and plotly backend support
   - ✅ **Dual API Support**: Direct object plotting and cache-based visualization

4. ✅ **Development Tools**:
   - ✅ Test script for algorithm validation (`scripts/development/test_noise_simple.py`)
   - ✅ Proper output handling with gitignore patterns
   - ✅ Performance benchmarking and comparison capabilities
   - ✅ **Dual API demonstration**: Both direct object and cache-based visualization workflows

**Remaining Tasks**:
- ⏳ `preprocessing/data_validation.py`: Input data sanity checks and FID parameter validation
- ⏳ Update `preprocessing/__init__.py` with new functions  
- ⏳ Create comprehensive unit tests for preprocessing components

**Key Technical Achievements**:
- **Statistical Foundation**: 20% difference thresholds, 5σ Z-test significance, F≥2.0 variance tests
- **Algorithm Efficiency**: O(log n) recursive subdivision with cached statistics 
- **Spectroscopic Intelligence**: Uses post-filtering statistics for meaningful region detection
- **Production Ready**: Verbose controls, MHz-based parameters, comprehensive error handling
- **Dual API Architecture**: Direct object visualization + cache-based session independence
- **Storage Optimization**: Cache-based visualization leverages 78.7% storage reduction from HDF5 serialization

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

2. Create `visualization/spectrum_visualization.py`:
   - ComplexFT spectrum visualization with dual API (direct + cache-based)
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
- **Development Phase**: ✅ Phase 2 (Core Data Structures) **COMPLETE**
- **Current Progress**: Phase 3 (Preprocessing Pipeline) - Ready to begin
- **Reference Code**: Available in `/home/kncrabtree/github/bcfitting/`
- **Test Data**: ✅ BlackChirp experiment 2638 integrated with processing guidance
- **Interactive Tools**: ✅ Visualization and data loading capabilities operational
- **Timeline**: Estimated 4-5 weeks remaining for complete implementation

## Next Immediate Steps

1. ✅ ~~Create new directory for ftmwpipeline package~~
2. ✅ ~~Initialize git repository~~
3. ✅ ~~Set up package structure with pyproject.toml~~
4. ✅ ~~Begin Phase 1: Package Infrastructure~~
5. ✅ ~~Phase 2: Extract and implement core data structures from bcfitting codebase~~
6. **Begin Phase 3**: Implement preprocessing pipeline (baseline/noise estimation, data validation)

---

**Last Updated**: 2025-08-02  
**Status**: Phase 2 Complete - Ready for Phase 3 (Preprocessing Pipeline)