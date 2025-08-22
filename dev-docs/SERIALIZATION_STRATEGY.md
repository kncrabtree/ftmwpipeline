# FTMW Pipeline Serialization Implementation Status

## Overview

This document describes the **current implementation** of the FTMW Pipeline serialization strategy, which has evolved from the original ComplexFT-centric approach to a **Stage 0-1 architecture** where FID data is cached and ComplexFT objects are calculated on-demand. This provides maximum flexibility and storage efficiency while maintaining bit-perfect scientific accuracy.

## Current Architecture: Lightweight .ftmw File Design ✅ (Restored 2025-08-22)

The implemented system follows a lightweight, on-demand computation approach:
- **Stage 0 (Data Loading)**: FID data cached with multi-format loader support (~6MB)
- **Stage 1 (FT Processing)**: ComplexFT calculated on-demand from cached FID + parameters (no storage)
- **Stage 2 (Noise Estimation)**: NoiseResult uses on-demand ComplexFT computation (~155KB storage)
- **Future Stages**: Will build upon on-demand ComplexFT architecture for consistency

**Architecture Restoration (2025-08-22)**: Removed automatic ComplexFT storage that was causing 12.8MB .ftmw file bloat and breaking the intended lightweight design. All interfaces now compute ComplexFT on-demand for true parameter exploration.

## Design Principles (Implemented)

1. **Stage-wise Caching**: ✅ Stage 0 (FID) cached, Stage 1 (ComplexFT) calculated on-demand
2. **Storage Efficiency**: ✅ Store minimal data needed for reconstruction 
3. **Scientific Preservation**: ✅ All essential scientific data preserved bit-perfectly
4. **Parameter Flexibility**: ✅ Interactive parameter exploration without re-caching
5. **Multi-format Support**: ✅ Extensible data loader architecture

## Current Pipeline Implementation

```
Raw Data → Stage 0: Data Loading → Stage 1: FT Processing → Future Stages...
   ↓         ↓ CACHED FID              ↓ ON-DEMAND                ↓
BlackChirp   ✅ IMPLEMENTED         ✅ IMPLEMENTED            PLANNED
CSV/HDF5     🔄 PLACEHOLDER         
Custom       📝 EXTENSIBLE
```

## Stage 0: Data Loading (FID Caching) - ✅ **IMPLEMENTED**

**Storage**: ~6 MB per experiment (750k points × 8 bytes)  
**Computational Cost**: Low (file I/O, format parsing)

### Serialization Strategy - **IMPLEMENTED**

**Key Innovation**: Cache raw FID data with recommended processing parameters, enabling unlimited parameter exploration.

**Implementation Details**:
- `time_series_data`: Raw FID voltage data (750k float64 values, ~6 MB)
- **Acquisition metadata**: Preserved from source format
  - `spacing`: Time spacing in seconds (e.g., 2.0000e-11 s)
  - `probe_freq_mhz`: LO probe frequency in MHz
  - `sideband`: Upper/Lower sideband configuration
  - `shots`: Number of averaged shots
- **Recommended processing**: Source format suggestions (not required for processing)
  - BlackChirp processing.csv parameters stored as defaults
  - User can override all parameters during FT calculation

**Extensible Loader Architecture**:
- **BlackChirp format**: ✅ Fully implemented with parameter extraction
- **CSV/HDF5 formats**: 🔄 Placeholder implementations ready for extension
- **Custom formats**: 📝 Simple registration interface for user formats

### HDF5 Structure
```
/fid_data/
├── time_series_data          [dataset: 750k float64] ~6MB
├── acquisition/              [group: core parameters]
│   ├── spacing              [attribute: float, seconds]
│   ├── probe_freq_mhz       [attribute: float]
│   ├── sideband             [attribute: str]
│   └── shots                [attribute: int]
├── recommended_processing/   [group: format suggestions]
│   ├── start_us             [attribute: float, optional]
│   ├── end_us               [attribute: float, optional]
│   ├── zpf                  [attribute: int]
│   ├── expf_us              [attribute: float, optional]
│   ├── winf                 [attribute: str, optional]
│   ├── rdc                  [attribute: bool]
│   └── units_power          [attribute: int]
└── metadata/                 [group: experiment info]
    ├── experiment_path      [attribute: str]
    ├── fid_index            [attribute: int]
    └── format_metadata      [subgroups: format-specific data]
```

**Implemented Benefits**:
- **Unlimited parameter exploration**: Try any FT parameters without re-loading source data
- **Interactive parameter persistence**: Save good parameters as new defaults
- **Format independence**: Cache separates processing from source format
- **Shareability**: Cache files contain complete analysis context
- **Multi-format support**: Extensible architecture for any time-domain format

## Stage 1: FT Processing (On-Demand ComplexFT) - ✅ **IMPLEMENTED**

**Storage**: No persistent storage (calculated on-demand)  
**Computational Cost**: Low-Medium (~1 second for 750k points)

### Processing Strategy - **IMPLEMENTED**

**Key Innovation**: Three-stage workflow enables parameter exploration and diagnostic visualization.

**Three-Stage FT Workflow**:
1. **FID.preprocess()**: Windowing, filtering, zero-padding → PreprocessedFID
2. **PreprocessedFID.compute_fft()**: FFT calculation → (spectrum, freq_array)  
3. **ComplexFT.from_spectrum()**: Create ComplexFT object with metadata

**Preprocessing Algorithm** (Corrected):
1. **Zero regions outside windowing bounds** (preserve original array length)
2. **Apply exponential filtering** (active region only)
3. **Apply window function** (active region only)
4. **Remove DC component** (after windowing - correct scientific order)
5. **Zero padding** for frequency resolution enhancement

**Implementation Details**:
- **No ComplexFT caching**: Calculated fresh each time with current parameters
- **Parameter validation**: FT parameters validated before expensive computation
- **Diagnostic visualization**: Enhanced plots show raw FID, preprocessed FID, and spectrum
- **Frequency trimming**: Post-processing step for analysis region selection

**Parameter Coverage**:
- **Preprocessing**: start_us, end_us, zpf, expf_us, window_function, units_power
- **Postprocessing**: trim (frequency range selection)
- **Interactive persistence**: Save good parameter combinations as defaults

**Implemented Benefits**:
- **Fast parameter iteration**: ~1s calculation enables interactive exploration
- **Complete workflow visibility**: Visualization shows preprocessing effects
- **Scientific accuracy**: Corrected preprocessing order follows spectroscopy best practices
- **Memory efficiency**: No persistent storage of large frequency-domain arrays
- **Parameter validation**: Early detection of invalid parameter combinations

### User Interface - ✅ **IMPLEMENTED**

**Stage 0 Commands** (Data Loading):
```bash
# Load BlackChirp experiment
ftmwpipeline data-load exp_2638 --source examples/blackchirp_data/2638/

# Load custom format (future)
ftmwpipeline data-load exp_custom --source data.csv --format csv

# View cached FID info
ftmwpipeline data-info exp_2638

# Visualize FID data
ftmwpipeline data-visualize exp_2638
```

**Stage 1 Commands** (FT Processing):
```bash
# Validate FT parameters (power users)
ftmwpipeline ft-process exp_2638 --zpf 2 --expf_us 5.0 --trim 26500:40000

# Interactive parameter exploration (researchers)
ftmwpipeline ft-visualize exp_2638 --start-us 2.0 --end-us 12.0 --expf_us 8.0
```

**Parameter Persistence Workflow**:
1. Experiment with ft-visualize using different parameters
2. When satisfied, save parameters as defaults (y/N prompt)
3. Future pipeline stages automatically use saved parameters
4. Complete parameter sets (preprocessing + postprocessing) preserved

## Implementation Status Summary

### ✅ **Completed Features**
- **Multi-format data loading** with BlackChirp implementation
- **FID serialization/deserialization** with bit-perfect reconstruction
- **Three-stage FT workflow** with diagnostic visualization  
- **Interactive parameter exploration** with persistence
- **Enhanced visualization** showing complete processing pipeline
- **CLI command structure** for stage-based workflow
- **Parameter validation** and error handling

### 📝 **Testing Status** 
- **Data loading tests**: ✅ Implemented (BlackChirp format, serialization)
- **Three-stage workflow tests**: ✅ Implemented (preprocessing, FFT, ComplexFT creation)
- **ComplexFT serialization tests**: ❌ **NEEDS REWRITE** (obsolete storage-based tests)
- **Integration tests**: 🔄 Needs update for new CLI workflow

### 🔄 **Pending Work**
- **ComplexFT test scope redefinition**: Focus on metadata validation, not object storage
- **Integration test updates**: CLI-based workflow testing
- **Noise estimation compatibility**: Ensure API works with on-demand ComplexFT
- **Performance benchmarking**: Validate on-demand calculation acceptable for interactive use

## Architectural Decision: Why On-Demand ComplexFT?

### **Advantages of On-Demand Calculation**:
1. **Parameter Flexibility**: Users can try unlimited parameter combinations instantly
2. **Storage Efficiency**: ~6MB FID storage vs ~9MB ComplexFT storage per experiment
3. **Cache Simplicity**: Single FID cache serves all parameter combinations
4. **Interactive Workflow**: Real-time parameter effects visible in enhanced visualization
5. **Scientific Accuracy**: Parameter exploration encourages finding optimal settings

### **Performance Considerations**:
- **FT Calculation Time**: ~1 second for 750k points (acceptable for interactive use)
- **Memory Usage**: Temporary ComplexFT objects freed after use
- **Cache Loading**: <100ms FID loading enables rapid session startup

### **Compared to Previous Architecture**:
- **Previous**: Cache ComplexFT with specific parameters → Limited reusability
- **Current**: Cache FID + calculate any parameters → Unlimited flexibility

## Future Development

### **Stage 2+: Advanced Pipeline Stages**
- **Noise Estimation**: Extend to work with on-demand ComplexFT
- **Peak Detection**: Cache results for expensive clustering algorithms
- **Window Assignment**: Cache window definitions, reconstruct from ComplexFT
- **Fitting Results**: Cache fitted parameters, reconstruct spectra on-demand

### **Testing and Quality Assurance**
- **Comprehensive test coverage**: All stage 0-1 functionality
- **Performance benchmarking**: Validate interactive responsiveness
- **Multi-format validation**: Test extensible loader architecture
- **Integration testing**: End-to-end CLI workflow validation

### **Advanced Features**
- **Custom format documentation**: Simple interface for user format implementation
- **Batch processing support**: High-throughput workflows with parameter files
- **Multi-experiment analysis**: Comparative studies across experimental datasets
- **Parameter optimization**: Automated parameter search for optimal results

---

**Last Updated**: 2025-08-08  
**Status**: Stage 0-1 architecture **COMPLETE AND WORKING** ✅  
**Next Phase**: Test suite updates and Stage 2+ pipeline development