# FTMW Pipeline - Iterative Stage-Based Development Roadmap

## Project Vision

The FTMW Pipeline implements a **stage-based, cacheable, interactive** approach to FTMW spectroscopy data analysis. Each pipeline stage can be executed independently, with results cached for efficient parameter exploration and workflow flexibility.

### Core Design Principles

1. **Stage Independence**: Each stage operates on cached data from previous stages
2. **Interactive Workflow**: Query/response interface for parameter exploration  
3. **Cacheable Results**: HDF5 serialization enables efficient reprocessing
4. **Standalone Commands**: Individual CLI subcommands per pipeline stage
5. **Flexible Configuration**: Interactive mode + config.json for batch processing

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
- Unit tests with real experiment data
- Parameter validation and edge case handling
- API refinement based on testing results

### 4. Serialization Implementation
- HDF5-based caching with storage optimization
- Bit-perfect reconstruction algorithms
- Unit tests for serialization round-trips

### 5. Pipeline Integration
- CLI subcommand for stage execution
- Cache loading/saving interfaces
- Parameter configuration management

### 6. Interactive Workflow
- Cache-based visualization functions
- Interactive parameter exploration
- Integration testing with full pipeline

This pattern ensures each stage is **fully functional, tested, and integrated** before proceeding to the next stage.

---

## Pipeline Architecture

```
Raw Data → Stage 0 → Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5 → Results
           (Load)    (FT)     (Noise)   (Peaks)   (Windows) (Fitting)
             ↓         ↓         ↓         ↓         ↓         ↓
           FID       ComplexFT  NoiseResult Peak[]   Window[]  FittedPeak[]
           Cache     Cache      Cache      Cache     Cache     Cache
             ↓         ↓         ↓         ↓         ↓         ↓
           Visualize Visualize Visualize Visualize Visualize Visualize
```

Each stage:
- **Loads**: Cached data from previous stages
- **Processes**: Applies algorithms with configurable parameters  
- **Caches**: Results for subsequent stages and visualization
- **Visualizes**: Both direct objects and cached data

**Stage 0 (Data Loading)**: Multi-format data ingestion layer that creates standardized FID objects from various experimental formats (BlackChirp, CSV, HDF5, etc.) with full metadata preservation.

---

## Current Status Overview

| Stage | Core Logic | Visualization | Testing | Serialization | Pipeline | Interactive | Status |
|-------|-----------|---------------|---------|---------------|----------|-------------|--------|
| **Stage 0: Data Loading** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **COMPLETE** |
| **Stage 1: FT Processing** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | **COMPLETE** |
| **Stage 2: Noise Estimation** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | **Ready for Integration** |
| Stage 3: Peak Detection | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | Not Started |
| Stage 4: Window Assignment | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | Not Started |
| Stage 5: Fitting | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | Not Started |

**Current Focus**: Stage 2 (Noise Estimation) CLI integration - **Phase 1 Complete**

**Latest Update**: **Phase 1 Complete** - Stage 0-1 refinement finished with critical bug fixes, parameter validation, CLI improvements, and comprehensive unit test alignment

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

## Interactive CLI Vision

### Query/Response Workflow
```
$ ftmwpipeline interactive

Welcome to FTMW Pipeline Interactive Mode
=========================================

[1] Select input FID data:
> examples/blackchirp_data/2638

✓ Loaded FID: 750k points, 15 μs duration, 40.96 GHz probe

[2] FT Processing Parameters:
   Zero padding factor (zpf): 1
   Exponential filter (expf_us): 5.0 μs
   Frequency trim range: Full spectrum
   
   Modify parameters? (y/N): y
   > zpf = 2
   > expf_us = 3.0
   > trim = 26500:40000
   
✓ Updated parameters

[3] Execute FT processing? (Y/n): Y
   Processing... ✓ Complete (2.3s)
   
[4] Review FT results:
   [Interactive plot window opens]
   
   Accept results and cache? (Y/n): Y
   ✓ Cached FT results

[5] Noise Estimation Parameters:
   Skew target: 0.631
   Minimum bin fraction: 0.03125
   Smoothing window: 1000 MHz
   
   Modify parameters? (y/N): N
   
[6] Execute noise estimation? (Y/n): Y
   Processing... ✓ Complete (4.1s)
   
[7] Review noise estimation:
   [Interactive diagnostic plot opens]
   
   Accept results and cache? (Y/n): Y
   ✓ Cached noise results

[8] Continue to peak detection? (Y/n): Y
   ...
```

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

## CLI Architecture Design

### Command Structure
```
ftmwpipeline
├── interactive                    # Interactive mode
├── batch <config.json>           # Batch processing  
├── config
│   ├── generate-template         # Create config template
│   └── validate <config.json>    # Validate config file
├── data-load <exp_id>            # Stage 0: Data loading
├── data-visualize <exp_id>       # Stage 0: Visualization
├── ft-process <exp_id>           # Stage 1: FT processing
├── ft-visualize <exp_id>         # Stage 1: Visualization
├── noise-estimate <exp_id>       # Stage 2: Noise estimation  
├── noise-visualize <exp_id>      # Stage 2: Visualization
├── peak-detect <exp_id>          # Stage 3: Peak detection
├── peak-visualize <exp_id>       # Stage 3: Visualization
├── window-assign <exp_id>        # Stage 4: Window assignment
├── window-visualize <exp_id>     # Stage 4: Visualization
├── fit <exp_id>                  # Stage 5: Fitting
├── fit-visualize <exp_id>        # Stage 5: Visualization
└── cache
    ├── info <exp_id>             # Cache information
    ├── list                      # List cached experiments
    └── clear <exp_id>            # Clear cache data
```

### Implementation Structure
```
src/ftmwpipeline/
├── cli/
│   ├── __init__.py
│   ├── main.py                   # Entry point, command routing
│   ├── interactive.py            # Interactive mode implementation
│   ├── batch.py                  # Batch processing 
│   ├── config_commands.py        # Configuration management
│   ├── data_commands.py          # Stage 0 CLI commands
│   ├── ft_commands.py            # Stage 1 CLI commands
│   ├── noise_commands.py         # Stage 2 CLI commands
│   ├── peak_commands.py          # Stage 3 CLI commands (future)
│   ├── window_commands.py        # Stage 4 CLI commands (future)
│   ├── fitting_commands.py       # Stage 5 CLI commands (future)
│   └── cache_commands.py         # Cache management
└── config/
    ├── __init__.py
    ├── config_schema.py          # JSON schema validation
    ├── templates.py              # Configuration templates
    └── parameter_defaults.py     # Default parameter values
```

---

## Next Development Priorities

### **Immediate (Next 2 weeks)**
1. **Stage 0: Data Loading Layer**
   - Implement multi-format data loading infrastructure
   - Create `data-load` and `data-visualize` CLI commands
   - Add FID serialization/caching system

2. **Stage 1: FT Processing CLI Refinement**
   - Remove direct data loading from `ft-process` command
   - Implement cache-first workflow (`--from-cache` only)
   - Integrate with Stage 0 FID caching system

3. **Stage 2: Noise Estimation CLI Integration**
   - Implement `noise-estimate` and `noise-visualize` commands
   - Full Stage 0 → Stage 1 → Stage 2 workflow functional

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