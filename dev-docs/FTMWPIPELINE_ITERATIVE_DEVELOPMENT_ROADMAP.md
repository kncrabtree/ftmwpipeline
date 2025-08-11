# FTMW Pipeline - Iterative Stage-Based Development Roadmap

## Project Vision

The FTMW Pipeline provides a **dual-interface, file-centric** approach to FTMW spectroscopy data analysis, offering both Python API and CLI workflows built around portable `.ftmw` pipeline data files. Each experiment is self-contained in a single file that progresses through analysis stages, enabling reproducible science and collaborative analysis.

### Core Design Principles

1. **File-Centric Design**: Each `.ftmw` file contains one complete experiment analysis (FID → FT → Noise → Peaks → Fitting)
2. **Dual Interface Access**: Identical functionality via Python API and CLI commands
3. **Stage-Based Processing**: Sequential pipeline stages with clear dependencies and outputs
4. **Portable Analysis**: Self-contained `.ftmw` files can be shared, archived, and reproduced anywhere
5. **Interactive Parameter Exploration**: Safe re-execution and parameter optimization for research workflows
6. **No Code Duplication**: Shared implementation between Python API and CLI ensures consistency

### User Workflows

**Python API Workflow** (Interactive Research):
```python
from ftmwpipeline import Pipeline

# Create new analysis
pipe = Pipeline.create("my_experiment.ftmw", source="data/experiment_2638/")

# Interactive parameter exploration
pipe.visualize_ft(zpf=1, expf_us=5.0)
pipe.visualize_ft(zpf=2, expf_us=3.0, save_params=True)  # Save optimal params

# Process subsequent stages
pipe.estimate_noise()
pipe.detect_peaks(algorithm='hybrid')
```

**CLI Workflow** (Automation & Scripting):
```bash
# Create pipeline data file
ftmwpipeline import-data my_experiment.ftmw --source data/experiment_2638/

# Process with saved parameters  
ftmwpipeline visualize-ft my_experiment.ftmw --trim 26500:40000
ftmwpipeline estimate-noise my_experiment.ftmw
ftmwpipeline detect-peaks my_experiment.ftmw --algorithm hybrid
```

**Architecture Documentation**: Detailed strategies documented in:
- [`API_STRATEGY.md`](API_STRATEGY.md) - Python API design and file management
- [`CLI_STRATEGY.md`](CLI_STRATEGY.md) - Command-line interface principles  
- [`SERIALIZATION_STRATEGY.md`](SERIALIZATION_STRATEGY.md) - HDF5 storage and optimization
- [`TESTING_STRATEGY.md`](TESTING_STRATEGY.md) - Comprehensive testing approach for dual-interface architecture

## Development Pattern

Each pipeline stage follows a comprehensive **6-step development cycle**:

### 1. Core Logic Implementation
- Extract and refactor algorithms from source codebase
- Create clean data structures and processing functions
- Focus on scientific accuracy and performance

### 2. Visualization Development
- Direct object plotting functions 
- Interactive and static plotting options
- Comprehensive diagnostic outputs

### 3. Testing & API Stabilization
- **Unit Tests**: Core algorithm testing with real experiment data
- **Integration Tests**: End-to-end workflows for all three interfaces (CLI, Pipeline class, functional API)
- **Cross-Interface Consistency**: Verify identical results across interfaces
- **Parameter Validation**: Edge cases and error handling across all interfaces
- **File Management Tests**: `.ftmw` file creation, opening, validation, and error scenarios

### 4. Serialization Implementation
- HDF5-based caching with storage optimization
- Bit-perfect reconstruction algorithms
- Unit tests for serialization round-trips

### 5. Dual Interface Implementation
- Shared core implementation functions
- Python API methods (Pipeline class + functional API)  
- CLI command wrappers with consistent parameters
- Error handling and validation across interfaces

### 6. Integration & Workflow Testing
- Pipeline data file management and portability
- Interactive parameter exploration and persistence
- End-to-end workflow testing (Python API + CLI)
- Cross-interface consistency validation

This pattern ensures each stage is **fully functional, tested, and integrated** before proceeding to the next stage.

---

## Pipeline Architecture

### Single-File Pipeline Design
```
Raw Data → experiment.ftmw → Analysis Results
           ┌──────────────┐
           │ .ftmw File   │
           │              │
           │ Stage 0: FID │ ←→ Python API / CLI
           │ Stage 1: ComplexFT │ ←→ pipe.compute_ft() / ftmwpipeline compute-ft
           │ Stage 2: NoiseResult │ ←→ pipe.estimate_noise() / ftmwpipeline estimate-noise  
           │ Stage 3: Peak[] │ ←→ pipe.detect_peaks() / ftmwpipeline detect-peaks
           │ Stage 4: Window[] │ ←→ pipe.assign_windows() / ftmwpipeline assign-windows
           │ Stage 5: FittedPeak[] │ ←→ pipe.fit_peaks() / ftmwpipeline fit-peaks
           │              │
           │ + Metadata   │
           │ + Parameters │
           │ + Source Info│
           └──────────────┘
```

### Stage Processing Model
Each stage in the `.ftmw` file:
- **Loads**: Data from previous stages within the same file
- **Processes**: Applies algorithms with user-configurable parameters
- **Stores**: Results and metadata for subsequent stages
- **Visualizes**: Provides both Python and CLI visualization interfaces
- **Validates**: Dependencies and parameter consistency

### Dual Interface Access
Every stage operation available through both interfaces:
```python
# Python API
pipe = Pipeline.open("experiment.ftmw") 
result = pipe.compute_ft(zpf=2, expf_us=5.0)
```
```bash
# CLI  
ftmwpipeline compute-ft experiment.ftmw --zpf 2 --expf_us 5.0
```

**Stage 0 (Data Import)**: Multi-format data ingestion that creates portable `.ftmw` files from various experimental formats (BlackChirp, CSV, HDF5, etc.) with complete metadata and source traceability.

---

## Current Status Overview

| Stage | Core Logic | Visualization | Testing | Serialization | Python API | CLI | Status |
|-------|-----------|---------------|---------|---------------|-------------|-----|--------|
| **Stage 0: Data Loading** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | **Ready for API** |
| **Stage 1: FT Processing** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | **Ready for API** |
| **Stage 2: Noise Estimation** | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | **Ready for Integration** |
| Stage 3: Peak Detection | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | Not Started |
| Stage 4: Window Assignment | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | Not Started |
| Stage 5: Fitting | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | Not Started |

**Current Focus**: **High-Level API Implementation** - Implement Python API and updated CLI for Stages 0-1

### Latest Status: API Strategy Design Complete ✅

**Architecture Documentation Complete**:
- **API Strategy**: [`API_STRATEGY.md`](API_STRATEGY.md) - Pipeline class, functional API, and .ftmw file management
- **CLI Strategy**: [`CLI_STRATEGY.md`](CLI_STRATEGY.md) - File-centric commands and shared implementation
- **Serialization Strategy**: [`SERIALIZATION_STRATEGY.md`](SERIALIZATION_STRATEGY.md) - HDF5 optimization and stage storage

**Next Implementation Priority**: 
1. **Python API Implementation** (Stages 0-1): `Pipeline` class, functional API, and file manager
2. **Updated CLI Commands** (Stages 0-1): File-centric commands using shared implementation  
3. **Stage 2 Integration**: Extend dual-interface pattern to noise estimation

---

## Stage 0: Data Loading (FID)

### ✅ **COMPLETE** - Core Logic Implementation
**Target**: Multi-format data ingestion with standardized FID output
- **Implementation**: Extensible loader architecture with format registry
- **Data Structure**: Standardized `FID` objects with preserved metadata
- **Formats**: BlackChirp (✅), CSV (✅), HDF5 (✅), extensible registry system

**Completed Tasks**:
- ✅ Created `src/ftmwpipeline/io/data_loaders/` package structure
- ✅ Implemented base loader interface and format registry
- ✅ Refactored BlackChirp loader into new architecture
- ✅ Added CSV format support for generic time-series data
- ✅ Implemented auto-format detection capabilities

### ✅ **COMPLETE** - Visualization Development
**Target**: FID visualization for data validation
- **Location**: `src/ftmwpipeline/visualization/fid_visualization.py`
- **Functions**: `plot_fid()`, `plot_fid_from_cache()`
- **Features**: Time-domain plots, metadata display, acquisition parameter validation
- **Diagnostics**: Signal quality checks, timing validation

### ✅ **COMPLETE** - Testing & API Stabilization
**Target**: Multi-format validation with real data
- **Real Data**: Validated with experiment 2638 data
- **Format Detection**: Auto-detection working for all supported formats
- **Error Handling**: Comprehensive validation and error reporting

### ✅ **COMPLETE** - Serialization Implementation
**Target**: FID caching with metadata preservation
- **Location**: `src/ftmwpipeline/io/fid_serialization.py`
- **Implementation**: Complete HDF5 serialization with bit-perfect reconstruction
- **Storage**: Optimized HDF5 storage with metadata preservation
- **Validation**: Bit-perfect FID reconstruction verified with real data

**HDF5 Structure**:
```
/fid_data/
├── time_series_data              [dataset: real voltage data, float64]
├── acquisition/                  [group: acquisition parameters]
│   ├── spacing_seconds          [attr: float, time spacing in seconds]
│   ├── probe_freq_mhz           [attr: float, probe/LO frequency]
│   ├── sideband                 [attr: str, 'upper' or 'lower']
│   ├── shots                    [attr: int, number of shots averaged]
│   ├── n_points                 [attr: int, number of time points]
│   └── duration_us              [attr: float, FID duration in microseconds]
├── recommended_processing/       [group: optional format-specific defaults]
│   ├── description              [attr: str, explains these are suggestions]
│   ├── zpf                      [attr: int, suggested zero padding factor]
│   ├── expf_us                  [attr: float or None, suggested exp filter]
│   ├── winf                     [attr: str or None, suggested window function]
│   ├── start_us/end_us          [attr: float or None, suggested time range]
│   ├── autoscale_MHz            [attr: float or None, suggested autoscale]
│   └── units_power              [attr: int, suggested scaling units]
└── metadata/                     [group: source and experimental metadata]
    ├── source_info              [dataset: JSON string with source metadata]
    └── experimental_data        [dataset: JSON string with experimental metadata]
```

**Stage 0-1 Refinement Completed (2025-08-07 → 2025-08-08)**:
- ✅ **Data Storage Corrections**: FID data correctly stored as real-valued (not complex)
- ✅ **Format Standardization**: Point spacing displayed in scientific notation (`.4e` format) and stored in seconds
- ✅ **Architecture Decoupling**: Processing parameters moved from `processing/` to `recommended_processing/` group
- ✅ **Cache Portability**: FID cache now independent of specific FT processing choices
- ✅ **Backward Compatibility**: Code handles both old and new cache format gracefully
- ✅ **Enhanced ft-visualize Parameter Persistence**: Complete parameter sets (preprocessing + postprocessing) saved as defaults
- ✅ **Revised Tool Descriptions**: Clear distinction between ft-process (validation) and ft-visualize (exploration)
- ✅ **Complete CLI Parameter Coverage**: Added missing units-power parameter to both ft-process and ft-visualize

**Phase 1 Complete - Critical Bug Fixes and Test Infrastructure (2025-08-08)**:
- ✅ **Parameter Validation Enhancement**: Added expf_us > 0 validation in FIDProcessingParameters
- ✅ **CLI Parameter Merging**: Fixed CLI commands to properly merge user parameters with cached defaults
- ✅ **Critical log2(0) Bug Fix**: Fixed overflow bug in FID preprocessing for empty data edge case
- ✅ **Unit Test Alignment**: Updated all unit tests to align with Stage 0-1 architecture:
  - Fixed `test_three_stage_workflow.py` parameter assumptions and empty data handling
  - Updated `test_data_loaders.py` to match corrected API behavior 
  - Removed performance/memory tests from `test_fid_serialization.py` and `test_complex_ft_serialization.py`
  - Eliminated backward compatibility tests that no longer apply
- ✅ **Test Infrastructure Refinement**: Focused unit test suite on validating correct behavior rather than documenting bugs
- ✅ **Architecture Validation**: Confirmed Stage 0-1 separation works correctly with proper error handling

### ✅ **COMPLETE** - Pipeline Integration
**Target**: CLI commands for data loading
```bash
ftmwpipeline data-load exp_2638 --source examples/blackchirp_data/2638 --format blackchirp
ftmwpipeline data-load exp_2638 --source data.csv --format csv
ftmwpipeline data-load exp_2638 --source path/to/data  # Auto-detect format
ftmwpipeline data-visualize exp_2638 --show-metadata
```

**Completed Tasks**:
- ✅ Created `ftmwpipeline.cli.data_commands` module
- ✅ Added format-specific parameter parsing
- ✅ Implemented auto-format detection logic
- ✅ Added metadata validation and error reporting
- ✅ Integrated with cache management system

### ✅ **COMPLETE** - Interactive Workflow
**Target**: Interactive format selection and parameter validation
- **Implementation**: Complete cache-based visualization workflow
- **Format Selection**: Auto-detect with manual override options working
- **Parameter Validation**: Format-specific parameter validation implemented
- **Quality Assurance**: Visual validation via `data-visualize` command

---

## Stage 1: FT Processing (ComplexFT)

### ✅ **COMPLETE** - Core Logic Implementation  
- **Location**: `src/ftmwpipeline/core/data_structures.py` (Three-stage workflow)
- **Architecture**: FID.preprocess() → PreprocessedFID.compute_fft() → ComplexFT.from_spectrum()
- **Algorithms**: Corrected preprocessing (zeroing, filtering, windowing, DC removal), FFT with zero-padding
- **Data Structures**: `PreprocessedFID` (intermediate) + `ComplexFT` (final result)
- **Performance**: ~1 second for experiment 2638 (on-demand calculation)

### ✅ **COMPLETE** - Visualization Development  
- **Location**: `src/ftmwpipeline/visualization/spectrum_visualization.py`
- **Functions**: `plot_complex_ft()`, `plot_complex_ft_from_cache()`
- **Features**: Magnitude + real/imaginary plots, matplotlib + plotly backends
- **Cache Integration**: Direct visualization from HDF5 cache

### ✅ **COMPLETE** - Testing & API Stabilization
- **Unit Tests**: 32 tests covering FFT algorithms, parameter validation
- **Real Data**: Extensive testing with experiment 2638
- **Edge Cases**: Trimming, parameter overrides, sideband configurations
- **API**: Stable interface with backward compatibility

### ✅ **COMPLETE** - On-Demand Processing Implementation
- **Architecture**: ComplexFT calculated on-demand from cached FID data
- **Three-Stage Workflow**: FID.preprocess() → compute_fft() → ComplexFT.from_spectrum()
- **Storage Strategy**: No ComplexFT caching - calculated fresh with current parameters (~1s)
- **Benefits**: Unlimited parameter exploration, reduced storage requirements, interactive workflow

### ✅ **COMPLETE** - Pipeline Integration
**Target**: CLI subcommands for FT processing (Stage 0 → Stage 1 workflow)
```bash
# Stage-based workflow (on-demand FT calculation)
ftmwpipeline data-load exp_2638 --source examples/blackchirp_data/2638
ftmwpipeline ft-process exp_2638 --zpf 2 --expf_us 3.0 --trim 26500:40000
ftmwpipeline ft-visualize exp_2638 --start-us 2.0 --end-us 12.0 --expf_us 5.0 --trim 26500:40000
```

**Completed Implementation**:
- ✅ Complete `ft-process` and `ft-visualize` commands with full parameter support
- ✅ CLI package structure with modular command organization
- ✅ Full integration with Stage 0 FID caching system
- ✅ On-demand ComplexFT calculation - no ComplexFT storage required
- ✅ Enhanced parameter validation and error handling
- ✅ Clean stage separation - no direct data loading in ft-process

**Working Pipeline**:
- ✅ Stage 0 → Stage 1 workflow fully functional with on-demand calculation
- ✅ FID caching working, ComplexFT calculated on-demand per session
- ✅ Interactive parameter exploration with immediate visual feedback
- ✅ Error handling guides users through proper workflow

### ✅ **COMPLETE** - Interactive Workflow
- **Cache Visualization**: `plot_complex_ft_from_cache()` working
- **Integration Testing**: Cache-based workflow validated
- **Parameter Exploration**: Both direct and cached visualization APIs
- **Interactive Parameter Persistence**: ft-visualize saves complete parameter sets (preprocessing + postprocessing)
- **Complete CLI Parameter Coverage**: Both tools now support all processing parameters including units-power
- **Enhanced FID Visualization**: Multi-panel layout showing complete FID-to-spectrum processing workflow
- **Corrected Preprocessing**: Fixed FID processing order with DC removal after windowing
- **Parameter Consistency**: Resolved windowing bounds display inconsistencies between raw and preprocessed panels

**Latest Enhancements (2025-08-08)**:
- ✅ **Enhanced Matplotlib Layout**: 16:9 aspect ratio with 3 equally-sized rows for comprehensive workflow visualization
- ✅ **Raw FID Panel**: Shows original time-domain data with red dashed lines at windowing boundaries (start_us/end_us)
- ✅ **Preprocessed FID Panel**: Displays effects of filtering, windowing, and zero-padding transformations
- ✅ **Corrected Processing Order**: Fixed critical bug where DC removal occurred before windowing (now correctly after)
- ✅ **Parameter Consistency Fix**: Raw FID panel now shows user's current parameters instead of cached parameters
- ✅ **Layout Optimization**: Always calls fig.tight_layout() for proper panel spacing and readability

---

## Stage 2: Noise Estimation

### ✅ **COMPLETE** - Core Logic Implementation
- **Location**: `src/ftmwpipeline/preprocessing/noise_estimation.py`
- **Algorithms**: Adaptive binning, variance-based noise identification
- **Data Structure**: `NoiseResult` with RMS estimates and noise masks
- **Performance**: ~3-10 seconds for experiment 2638

### ✅ **COMPLETE** - Visualization Development
- **Location**: `src/ftmwpipeline/visualization/noise_visualization.py` 
- **Functions**: `plot_noise_estimation()`, `plot_noise_estimation_from_cache()`
- **Features**: Spectrum + noise points + RMS estimates + diagnostics
- **Cache Integration**: Automatic cache loading and visualization

### ✅ **COMPLETE** - Testing & API Stabilization  
- **Unit Tests**: 24 tests covering adaptive algorithms, parameter ranges
- **Real Data**: Multiple parameter combinations with experiment 2638
- **Edge Cases**: Different dataset sizes, numerical edge cases
- **API**: Stable interface with comprehensive parameter validation

### ✅ **COMPLETE** - Serialization Implementation
- **Location**: `src/ftmwpipeline/io/noise_result_serialization.py`
- **Innovation**: Signal indices + convolution reconstruction (95.4% reduction)
- **Storage**: ~155 KB per experiment (vs 3.4 MB original)
- **Validation**: Bit-perfect RMS reconstruction via convolution method

### ❌ **TODO** - Pipeline Integration  
**Target**: CLI subcommand for noise estimation
```bash
ftmwpipeline noise-estimate exp_2638 --from-cache
ftmwpipeline noise-estimate exp_2638 --skew-target 0.7 --min-bin-fraction 0.025
ftmwpipeline noise-visualize exp_2638 --show-bins --y-max-factor 15
```

**Implementation Tasks**:
- [ ] Create `ftmwpipeline.cli.noise_commands` module
- [ ] Add parameter parsing for noise algorithms
- [ ] Integrate with ComplexFT cache loading
- [ ] Configuration file support for algorithm parameters
- [ ] Validation and error handling

### ✅ **COMPLETE** - Interactive Workflow
- **Cache Visualization**: `plot_noise_estimation_from_cache()` working
- **Integration Testing**: End-to-end cache workflow validated  
- **Dual APIs**: Both direct object and cache-based visualization

---

## Stage 3: Peak Detection

### ❌ **TODO** - Core Logic Implementation
**Target**: Extract peak detection algorithms from source
- **Source**: `/home/kncrabtree/github/bcfitting/newfitting/peak_classification.py`
- **Algorithms**: Second derivative detection, hybrid clustering
- **Data Structure**: `Peak` objects with frequency, intensity, SNR
- **Dependencies**: Requires NoiseResult for SNR calculations

**Implementation Tasks**:
- [ ] Create `src/ftmwpipeline/peak_detection/` module
- [ ] Extract `locate_peaks()` and `locate_peaks_hybrid()` functions
- [ ] Create `Peak` data structure with properties
- [ ] Implement SNR calculation using noise estimates
- [ ] Parameter validation and algorithm selection

### ❌ **TODO** - Visualization Development
**Target**: Peak detection diagnostic plots
- **Functions**: `plot_peaks()`, `plot_peaks_from_cache()`
- **Features**: Spectrum + detected peaks + SNR annotations
- **Diagnostics**: Algorithm comparison, parameter sensitivity

### ❌ **TODO** - Testing & API Stabilization
**Target**: Comprehensive testing with experiment 2638
- **Unit Tests**: Algorithm accuracy, parameter validation
- **Real Data**: Peak count validation against known results
- **Edge Cases**: Weak signals, dense regions, noise thresholds

### ❌ **TODO** - Serialization Implementation  
**Target**: Direct Peak object storage (minimal optimization needed)
- **Strategy**: Store complete Peak arrays (~75 KB per experiment)
- **Benefits**: Direct loading, no reconstruction overhead
- **Structure**: HDF5 arrays for frequencies, intensities, SNR values

### ❌ **TODO** - Pipeline Integration
**Target**: CLI subcommands for peak detection
```bash
ftmwpipeline peak-detect exp_2638 --algorithm hybrid --snr-threshold 5
ftmwpipeline peak-visualize exp_2638 --annotate-snr --highlight-strong
```

### ❌ **TODO** - Interactive Workflow
**Target**: Interactive peak detection parameter exploration
- **Cache Integration**: Load ComplexFT + NoiseResult from cache
- **Parameter Tuning**: SNR thresholds, algorithm selection
- **Visual Feedback**: Real-time peak annotations

---

## Stage 4: Window Assignment

### ❌ **TODO** - Core Logic Implementation  
**Target**: Extract spectral window assignment algorithms
- **Source**: `/home/kncrabtree/github/bcfitting/newfitting/window_assignment.py`
- **Algorithms**: Greedy assignment, FWHM-based windowing
- **Data Structure**: `SpectralWindow` objects with peak assignments
- **Dependencies**: Requires Peak objects and ComplexFT data

### ❌ **TODO** - Visualization Development
**Target**: Window assignment visualization
- **Functions**: `plot_windows()`, `plot_windows_from_cache()`  
- **Features**: Spectrum + window boundaries + peak assignments
- **Diagnostics**: Window statistics, assignment efficiency

### ❌ **TODO** - Testing & API Stabilization
**Target**: Window assignment validation
- **Unit Tests**: Assignment algorithms, window optimization
- **Real Data**: Window count and coverage validation
- **Edge Cases**: Dense peak regions, isolated peaks

### ❌ **TODO** - Serialization Implementation
**Target**: Index-based SpectralWindow reconstruction (99.6% reduction)
- **Strategy**: Store window definitions + peak assignments
- **Reconstruction**: Slice ComplexFT using stored indices  
- **Storage**: ~35 KB vs 9 MB (99.6% reduction)

### ❌ **TODO** - Pipeline Integration
**Target**: Window assignment CLI commands
```bash
ftmwpipeline window-assign exp_2638 --algorithm greedy --max-peaks-per-window 5
ftmwpipeline window-visualize exp_2638 --show-assignments --highlight-overlaps
```

### ❌ **TODO** - Interactive Workflow
**Target**: Interactive window assignment tuning
- **Parameter Exploration**: Window size limits, assignment strategies
- **Visual Validation**: Window boundaries with peak assignments

---

## Stage 5: Fitting

### ❌ **TODO** - Core Logic Implementation
**Target**: Extract time-domain fitting algorithms  
- **Source**: `/home/kncrabtree/github/bcfitting/newfitting/time_domain_fitting_unified.py`
- **Algorithms**: Conservative fitting, decay constraint optimization
- **Data Structure**: `FittedPeak` with parameters + uncertainties
- **Dependencies**: Requires SpectralWindow objects

### ❌ **TODO** - Visualization Development
**Target**: Fitting diagnostics and residual analysis
- **Functions**: `plot_fit_results()`, `plot_residuals()`, cache variants
- **Features**: Original + fitted + residual spectra
- **Diagnostics**: Parameter uncertainties, fit quality metrics

### ❌ **TODO** - Testing & API Stabilization
**Target**: Fitting algorithm validation
- **Unit Tests**: Parameter accuracy, uncertainty quantification
- **Real Data**: Fit quality validation against known results
- **Edge Cases**: Weak signals, overlapped peaks, convergence issues

### ❌ **TODO** - Serialization Implementation
**Target**: Parameter-based spectrum reconstruction (96.5% reduction)
- **Strategy**: Store fitted parameters + uncertainties + metadata
- **Reconstruction**: Generate fitted spectra from damped cosine model
- **Storage**: ~330 KB vs 9.4 MB (96.5% reduction)

### ❌ **TODO** - Pipeline Integration
**Target**: Fitting CLI commands
```bash
ftmwpipeline fit exp_2638 --algorithm conservative --max-iterations 100
ftmwpipeline fit-visualize exp_2638 --show-residuals --show-uncertainties  
```

### ❌ **TODO** - Interactive Workflow
**Target**: Interactive fitting parameter optimization
- **Parameter Tuning**: Convergence criteria, constraint settings
- **Visual Validation**: Fit quality assessment, residual inspection

---

## Dual Interface Vision

### Current File-Centric CLI Commands (Implemented Stages 0-1)
```bash
# Create new pipeline data file from experimental data
ftmwpipeline import-data my_experiment.ftmw --source examples/blackchirp_data/2638/

# FT processing with parameter exploration
ftmwpipeline visualize-ft my_experiment.ftmw --zpf 1 --expf_us 5.0
ftmwpipeline visualize-ft my_experiment.ftmw --zpf 2 --expf_us 3.0 --save-params
ftmwpipeline compute-ft my_experiment.ftmw --from-saved-params

# File inspection and management
ftmwpipeline info my_experiment.ftmw
ftmwpipeline validate my_experiment.ftmw
```

### Planned File-Centric Extensions (Future Stages)
```bash
# Stage 2: Noise Estimation
ftmwpipeline estimate-noise my_experiment.ftmw --skew-target 0.7
ftmwpipeline visualize-noise my_experiment.ftmw

# Stage 3: Peak Detection  
ftmwpipeline detect-peaks my_experiment.ftmw --algorithm hybrid --snr-threshold 5
ftmwpipeline visualize-peaks my_experiment.ftmw --annotate-snr

# Stage 4: Window Assignment
ftmwpipeline assign-windows my_experiment.ftmw --max-peaks-per-window 5
ftmwpipeline visualize-windows my_experiment.ftmw --show-assignments

# Stage 5: Fitting
ftmwpipeline fit-peaks my_experiment.ftmw --algorithm conservative
ftmwpipeline visualize-fits my_experiment.ftmw --show-residuals

# Batch processing and utilities
ftmwpipeline batch-process config.json --parallel 4
ftmwpipeline export-results my_experiment.ftmw --format csv
```

### Python API Integration
**Jupyter Notebook Workflow** (matches CLI functionality):
```python
from ftmwpipeline import Pipeline

# Create or open pipeline
pipe = Pipeline.create("my_experiment.ftmw", source="examples/blackchirp_data/2638/")
# pipe = Pipeline.open("my_experiment.ftmw")  # For existing files

# Interactive parameter exploration (matches CLI)
pipe.visualize_ft(zpf=1, expf_us=5.0)
pipe.visualize_ft(zpf=2, expf_us=3.0, save_params=True)

# Process subsequent stages
pipe.estimate_noise(skew_target=0.7)
pipe.detect_peaks(algorithm='hybrid', snr_threshold=5)
pipe.assign_windows(max_peaks_per_window=5)
pipe.fit_peaks(algorithm='conservative')
```

### Interactive Mode (Future Enhancement)
```bash
$ ftmwpipeline interactive my_experiment.ftmw

FTMW Pipeline Interactive Mode - my_experiment.ftmw
===================================================

Pipeline Status:
✅ Stage 0: Data imported (750k points, 15.0 μs)
✅ Stage 1: FT processed (zpf=2, expf_us=3.0)
❌ Stage 2: Noise estimation pending

Next available operations:
[1] Re-process FT with different parameters
[2] Estimate noise and continue pipeline
[3] Visualize current results  
[4] Export current data

Choice [2]: 2
Processing noise estimation... ✓ Complete
Continue to peak detection? (Y/n): Y
```

**Architecture References**: 
- Full CLI design in [`CLI_STRATEGY.md`](CLI_STRATEGY.md)
- Python API patterns in [`API_STRATEGY.md`](API_STRATEGY.md)

### Configuration File Support
```json
{
  "experiment_id": "exp_2638",
  "data_source": "examples/blackchirp_data/2638",
  "cache_dir": "cache/",
  "stages": {
    "ft_processing": {
      "zpf": 2,
      "expf_us": 3.0,
      "trim_range": [26500, 40000]
    },
    "noise_estimation": {
      "skew_target": 0.7,
      "min_bin_fraction": 0.025,
      "smoothing_window_mhz": 1000
    },
    "peak_detection": {
      "algorithm": "hybrid",
      "snr_threshold": 5.0
    }
  }
}
```

### Batch Processing Mode
```bash
ftmwpipeline batch config.json --stages ft,noise,peaks --interactive-plots
ftmwpipeline batch config.json --stage noise --reprocess --show-comparison
```

---

## Updated CLI Architecture Design

### File-Centric Command Structure
```
ftmwpipeline
├── interactive <file.ftmw>          # Interactive mode for specific file
├── batch-process <config.json>      # Batch processing multiple files
├── import-data <file.ftmw>          # Stage 0: Create pipeline from raw data
├── compute-ft <file.ftmw>           # Stage 1: FT processing
├── visualize-ft <file.ftmw>         # Stage 1: FT visualization
├── estimate-noise <file.ftmw>       # Stage 2: Noise estimation  
├── visualize-noise <file.ftmw>      # Stage 2: Noise visualization
├── detect-peaks <file.ftmw>         # Stage 3: Peak detection
├── visualize-peaks <file.ftmw>      # Stage 3: Peak visualization
├── assign-windows <file.ftmw>       # Stage 4: Window assignment
├── visualize-windows <file.ftmw>    # Stage 4: Window visualization
├── fit-peaks <file.ftmw>            # Stage 5: Fitting
├── visualize-fits <file.ftmw>       # Stage 5: Fit visualization
├── info <file.ftmw>                 # Pipeline file information
├── validate <file.ftmw>             # Validate pipeline integrity  
├── export-results <file.ftmw>       # Export analysis results
└── formats                          # List available data formats
```

**Key Changes from Previous Design**:
- **File-Centric**: All commands operate on `.ftmw` files instead of experiment IDs + cache directories
- **Explicit Import**: `import-data` clearly distinguishes creation from analysis operations
- **Simplified Management**: No separate cache management needed - files are self-contained
- **Consistent Naming**: Verb-object pattern with clear stage alignment

**Implementation Strategy**: See [`CLI_STRATEGY.md`](CLI_STRATEGY.md) for complete design details

### Dual-Interface Implementation Structure  
```
src/ftmwpipeline/
├── api.py                        # Functional Python API
├── pipeline.py                   # Pipeline class API  
├── file_manager.py               # .ftmw file operations
├── cli/
│   ├── main.py                   # Entry point, command routing
│   ├── commands.py               # All CLI commands (file-centric)
│   ├── interactive.py            # Interactive mode (future)
│   ├── batch.py                  # Batch processing (future)
│   └── utils.py                  # CLI utilities and helpers
├── _internal/                    # Shared implementation (CLI + API)
│   ├── stage0_impl.py            # Data import implementation
│   ├── stage1_impl.py            # FT processing implementation
│   ├── stage2_impl.py            # Noise estimation implementation
│   ├── stage3_impl.py            # Peak detection (future)
│   ├── stage4_impl.py            # Window assignment (future)
│   └── stage5_impl.py            # Fitting (future)
└── config/
    ├── __init__.py
    ├── parameter_defaults.py     # Shared default values
    └── validation.py             # Parameter validation
```

**Key Architectural Changes**:
- **Shared Implementation**: `_internal/` modules contain core logic used by both API and CLI
- **Unified Commands**: Single `commands.py` with file-centric operations instead of scattered command files  
- **API-First Design**: CLI commands are thin wrappers around API functions
- **File Manager**: Centralized `.ftmw` file operations with dependency checking

**Code Reuse Pattern**: All interfaces (Pipeline class, functional API, CLI) use the same `_internal/` implementations to eliminate duplication

---

## Next Development Priorities

### **Immediate (Next 2 weeks) - HIGH-LEVEL API IMPLEMENTATION**

**Priority 1: Python API Foundation (Stages 0-1)**
1. **File Manager Implementation** (`file_manager.py`)
   - `.ftmw` file creation, opening, and validation
   - Source metadata tracking and smart re-import detection
   - Stage dependency checking and error handling

2. **Pipeline Class** (`pipeline.py`)  
   - `Pipeline.create()` and `Pipeline.open()` with safe file management
   - Stage 0: `load_data()` → wrapper around existing data loaders
   - Stage 1: `compute_ft()` and `visualize_ft()` → wrapper around existing FT processing

3. **Functional API** (`api.py`)
   - `import_data()`, `compute_ft()`, `visualize_ft()` functions  
   - File-based parameter management and validation

**Priority 2: Updated CLI Commands (Stages 0-1)**
4. **File-Centric CLI** (`cli/commands.py`)
   - `import-data`, `compute-ft`, `visualize-ft`, `info` commands
   - Thin wrappers around Python API functions
   - Migration from current experiment ID + cache-dir pattern

5. **Shared Implementation Extraction** (`_internal/stage0_impl.py`, `_internal/stage1_impl.py`)  
   - Extract core logic from existing CLI implementations
   - Enable code reuse between Python API and CLI

**Priority 3: Testing Infrastructure Updates**
6. **Unit Test Updates**
   - Update existing unit tests to use `.ftmw` file extensions
   - Ensure test fixtures generate proper pipeline data files
   - Do not add any methods for backward compatability. This is all new development with no legacy usage to support.

7. **Integration Test Redesign**
   - **CLI Integration Tests**: End-to-end workflow using file-centric commands
   - **Pipeline Class Integration Tests**: Complete workflows using `Pipeline.create()` → `Pipeline.open()` pattern
   - **Functional API Integration Tests**: Stateless function-based workflows
   - **Cross-Interface Consistency Tests**: Verify identical results across CLI, Pipeline class, and functional API

**Priority 4: Extension to Stage 2**
8. **Stage 2 Dual Interface** 
   - Extend Pipeline class and functional API to noise estimation
   - Implement file-centric CLI commands for noise processing
   - Validate complete Stage 0 → 1 → 2 workflow in both interfaces

### **Short Term (1 month)**
3. **Stage 3: Peak Detection**
   - Full 6-step development cycle
   - Integration with Stages 1 & 2 via cache

4. **Enhanced Interactive Mode**
   - Visual parameter exploration
   - Comparison between parameter sets
   - Configuration export from interactive sessions

### **Medium Term (2-3 months)**  
5. **Stages 4 & 5: Window Assignment + Fitting**
   - Complete pipeline implementation
   - Full end-to-end testing

6. **Batch Processing & Performance**
   - Multi-experiment processing
   - Parallel processing capabilities
   - Performance optimization

### **Long Term (3+ months)**
7. **Advanced Features**
   - Parameter sensitivity analysis
   - Automated parameter optimization
   - Integration with external tools

---

## Success Metrics

### Technical Milestones
- [ ] **Stage Independence**: Each stage runs standalone with cached inputs
- [ ] **Interactive Workflow**: Complete query/response interface implemented  
- [ ] **Configuration Management**: JSON-based batch processing working
- [ ] **Storage Efficiency**: >75% cache storage reduction maintained across all stages
- [ ] **Performance**: Processing times within 10% of original implementation

### User Experience Goals
- [ ] **Learning Curve**: New users can process experiment 2638 in <30 minutes
- [ ] **Parameter Exploration**: Interactive parameter tuning with immediate visual feedback
- [ ] **Reproducibility**: Configuration files enable exact result reproduction
- [ ] **Flexibility**: Both interactive and batch modes support all use cases

### Quality Assurance
- [ ] **Test Coverage**: >90% coverage maintained across all implemented stages
- [ ] **Documentation**: Complete API documentation with examples
- [ ] **Validation**: All algorithms validated against original implementation results
- [ ] **Error Handling**: Graceful handling of all error conditions with helpful messages

---

**Last Updated**: 2025-08-06  
**Status**: Stages 1 & 2 ready for pipeline integration  
**Next Milestone**: CLI commands for FT processing and noise estimation