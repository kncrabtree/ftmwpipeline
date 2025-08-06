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
Raw Data → Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5 → Results
           (FT)     (Noise)   (Peaks)   (Windows) (Fitting)
             ↓         ↓         ↓         ↓         ↓
           Cache     Cache     Cache     Cache     Cache
             ↓         ↓         ↓         ↓         ↓
           Visualize Visualize Visualize Visualize Visualize
```

Each stage:
- **Loads**: Cached data from previous stages
- **Processes**: Applies algorithms with configurable parameters  
- **Caches**: Results for subsequent stages and visualization
- **Visualizes**: Both direct objects and cached data

---

## Current Status Overview

| Stage | Core Logic | Visualization | Testing | Serialization | Pipeline | Interactive | Status |
|-------|-----------|---------------|---------|---------------|----------|-------------|--------|
| **Stage 1: FT Processing** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | **Ready for Integration** |
| **Stage 2: Noise Estimation** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | **Ready for Integration** |
| Stage 3: Peak Detection | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | Not Started |
| Stage 4: Window Assignment | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | Not Started |
| Stage 5: Fitting | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | Not Started |

**Next Priority**: Pipeline integration for Stages 1 & 2

---

## Stage 1: FT Processing (ComplexFT)

### ✅ **COMPLETE** - Core Logic Implementation
- **Location**: `src/ftmwpipeline/core/data_structures.py:FID.ft()`
- **Algorithms**: FFT with zero-padding, exponential filtering, sideband conversion
- **Data Structure**: `ComplexFT` with frequency arrays and complex spectra
- **Performance**: ~2-5 seconds for experiment 2638 (566k points)

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

### ✅ **COMPLETE** - Serialization Implementation
- **Location**: `src/ftmwpipeline/io/complex_ft_serialization.py`
- **Innovation**: Frequency array reconstruction (99.998% storage reduction)
- **Storage**: ~6 MB per experiment with HDF5 compression
- **Validation**: Bit-perfect reconstruction verified with real data

### ❌ **TODO** - Pipeline Integration
**Target**: CLI subcommand for FT processing
```bash
ftmwpipeline ft-process exp_2638 --source examples/blackchirp_data/2638 
ftmwpipeline ft-process exp_2638 --zpf 2 --expf_us 3.0 --trim 26500:40000
ftmwpipeline ft-visualize exp_2638 --freq-range 26500:40000
```

**Implementation Tasks**:
- [ ] Create `ftmwpipeline.cli.ft_commands` module
- [ ] Add argument parsing for FT parameters  
- [ ] Integrate with existing caching system
- [ ] Add configuration file support
- [ ] Error handling and validation

### ✅ **COMPLETE** - Interactive Workflow
- **Cache Visualization**: `plot_complex_ft_from_cache()` working
- **Integration Testing**: Cache-based workflow validated
- **Parameter Exploration**: Both direct and cached visualization APIs

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
1. **Pipeline Integration for Stages 1 & 2**
   - Implement CLI commands for FT processing and noise estimation
   - Create configuration file support
   - Add cache management commands

2. **Interactive Mode Framework**
   - Basic query/response interface
   - Parameter modification workflows  
   - Integration with existing visualization

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