# FTMW Pipeline Serialization Strategy

## Overview

This document defines the comprehensive serialization strategy for the FTMW Pipeline, designed to enable efficient caching and resumption of computationally expensive pipeline stages. The strategy achieves massive storage reductions while preserving all essential scientific information.

## Design Principles

1. **Stage-wise Caching**: Each pipeline stage can be cached and resumed independently
2. **Storage Efficiency**: Exploit data relationships and redundancy for optimal storage
3. **Scientific Preservation**: All essential scientific data is preserved exactly
4. **Reconstruction Capability**: Non-essential data can be reconstructed on-demand
5. **User Flexibility**: Support both automatic caching and explicit control

## Pipeline Stages Overview

```
FTMWData → ComplexFT → NoiseResult → Peak[] → SpectralWindow[] → FittedPeak[]
    ↓         ↓           ↓           ↓           ↓              ↓
   skip     CACHE      CACHE       CACHE      CACHE          CACHE
   (low     (6 MB)    (155 KB)    (75 KB)    (35 KB)        (330 KB)
   cost)
```

**Total Pipeline Cache**: ~6.6 MB per experiment

## Stage 1: Data Loading (FTMWData)

**Decision**: No caching (low computational cost, file I/O only)

**Rationale**: 
- Loading BlackChirp data is fast (~100ms)
- Source files are already persistent storage
- Minimal computational processing involved

## Stage 2: FT Processing (ComplexFT)

**Storage**: ~6 MB per experiment  
**Computational Cost**: High (FFT with zero-padding, filtering)

### Serialization Strategy

**Key Insight**: Frequency arrays are deterministically generated and don't need storage.

**Store**:
- `complex_spectrum`: ~375k complex128 values (~6 MB)
- Frequency reconstruction parameters (6 scalars, ~48 bytes):
  - `n_fid_padded`: Length of zero-padded FID
  - `spacing_us`: FID time spacing in microseconds  
  - `probe_freq_mhz`: LO probe frequency in MHz
  - `sideband`: "UPPER" or "LOWER"
  - `autoscale_MHz`: DC suppression cutoff
  - `n_spectrum`: Length validation

**Reconstruct**:
- `freq_array`: Generated using `rfftfreq()` and sideband conversion

**Benefits**:
- 99.998% reduction in frequency array storage (3 MB → 48 bytes)
- Bit-perfect frequency array reconstruction
- All processing parameters preserved

### HDF5 Structure
```
/complex_ft/
├── complex_spectrum        [dataset: ~375k complex128] ~6MB
├── freq_reconstruction/    [group]
│   ├── n_fid_padded       [attribute: int]
│   ├── spacing_us         [attribute: float]
│   ├── probe_freq_mhz     [attribute: float] 
│   ├── sideband           [attribute: str]
│   ├── autoscale_MHz      [attribute: float]
│   └── n_spectrum         [attribute: int]
├── processing_params/      [group: zpf, expf_us, etc.]
└── metadata/              [group: experiment info]
```

## Stage 3: Noise Estimation (NoiseResult)

**Storage**: ~155 KB per experiment (95.4% reduction from 3.4 MB)  
**Computational Cost**: High (adaptive binning, statistical analysis)

### Serialization Strategy

**Key Insight**: 90%+ of spectrum points are noise, so store signal indices instead of full boolean mask.

**Store**:
- `signal_indices`: ~37.5k int32 values (~150 KB) - 10% of points  
- `rms_poly_coeffs`: Polynomial approximation (~104 bytes)
- `smoothing_params`: Convolution parameters (~100 bytes)
- `bin_info`: Adaptive binning metadata (~5 KB)

**Reconstruct**:
- `noise_mask`: Rebuild from signal indices
- `rms_noise`: Exact reconstruction via convolution OR fast polynomial approximation

**Benefits**:
- 95.4% storage reduction (3.4 MB → 155 KB)
- Exact RMS reconstruction capability
- Fast polynomial approximation available
- All statistical analysis preserved

### HDF5 Structure
```
/noise_result/
├── signal_indices         [dataset: ~37.5k int32] ~150KB
├── rms_poly_coeffs        [dataset: polynomial coefficients] ~104 bytes
├── smoothing_params/      [group: convolution parameters]
├── bin_info/             [group: adaptive binning metadata]
└── algorithm_info/       [group: method parameters]
```

### Reconstruction Method Decision

**Investigation Completed**: Polynomial approaches (Chebyshev, splines) were investigated but showed edge behavior issues and numerical constraints with real experimental data.

**Final Decision**: Use convolution-based exact reconstruction as the primary method. This provides:
- Proven reliability and robustness
- Excellent edge behavior across the full spectrum
- Exact reproducibility of original RMS curves
- Simpler implementation and maintenance

## Stage 4: Peak Detection (Peak[])

**Storage**: ~75 KB per experiment  
**Computational Cost**: Medium (second derivatives, thresholding, clustering)

### Serialization Strategy

**Decision**: Store complete Peak objects (no optimization needed)

**Rationale**:
- Output is already compact (200-500 peaks vs 375k spectrum points)  
- Algorithm complexity makes reconstruction expensive relative to storage
- Parameter sensitivity makes caching valuable for tuning workflows
- Direct object loading optimal for downstream processing

**Store**:
- Complete Peak array with all properties
- Algorithm metadata and parameters
- Performance metrics

**Benefits**:
- Direct loading (~1-5ms)
- No reconstruction overhead
- Algorithm comparison capability
- Debugging preservation

### HDF5 Structure
```
/peak_detection/
├── peaks/                  [group]
│   ├── frequencies        [dataset: N×float64]
│   ├── intensities        [dataset: N×float64]
│   ├── snr_values         [dataset: N×float64]
│   ├── classifications    [dataset: N×str]
│   └── properties/        [subgroups for additional data]
├── algorithm_info/        [group: method, parameters]
└── dependencies/          [group: validation checksums]
```

## Stage 5: Window Assignment (SpectralWindow[])

**Storage**: ~35 KB per experiment (99.6% reduction from 9 MB)  
**Computational Cost**: Medium (greedy assignment, FWHM calculations)

### Serialization Strategy

**Key Insight**: SpectralWindows are views of ComplexFT data, not independent copies.

**Store**:
- Window definitions with frequency ranges and indices
- Peak assignments to windows  
- Algorithm metadata

**Reconstruct**:
- SpectralWindow objects by slicing parent ComplexFT using stored indices

**Benefits**:
- 99.6% storage reduction (9 MB → 35 KB)
- Fast reconstruction via index-based slicing (~1-10ms)
- Clear dependency relationships
- Multiple assignment algorithm results can be cached

### HDF5 Structure
```
/window_assignment/
├── window_definitions/     [group]
│   ├── window_ids         [dataset: 150×str]
│   ├── freq_ranges        [dataset: 150×2×float64] 
│   ├── start_indices      [dataset: 150×int32]
│   ├── end_indices        [dataset: 150×int32]
│   └── assignment_metadata [attributes]
├── window_peaks/          [group: peak assignments per window]
└── algorithm_info/        [group: method, parameters]
```

## Stage 6: Fitting Results (FittedPeak[])

**Storage**: ~330 KB per experiment (96.5% reduction from 9.4 MB)  
**Computational Cost**: Very High (nonlinear optimization, iterative fitting)

### Serialization Strategy

**Key Insight**: Fitted spectra and residuals can be reconstructed from fitted parameters since we have the window definitions and ComplexFT cached.

**Store**:
- Fitted parameters for all peaks (frequencies, amplitudes, decay rates, phases)
- Parameter uncertainties and correlations  
- Covariance matrices (valuable for uncertainty analysis)
- Quality metrics (AIC, χ², cost function)
- Algorithm metadata

**Don't Store**:
- Fitted spectra (reconstruct from parameters)
- Residuals (compute as original - fitted)
- Window references (reconstruct from window_id)

**Reconstruct**:
- Fitted spectra using damped cosine model from parameters
- Residuals by subtracting fitted from original spectra

**Benefits**:
- 96.5% storage reduction (9.4 MB → 330 KB)
- All essential scientific data preserved
- Fast parameter access (~1ms)
- On-demand spectrum reconstruction (~1-5ms per window)
- Uncertainty quantification preserved

### HDF5 Structure
```
/fitting_results/
├── fitted_peaks/           [group: all fitted peaks]
│   ├── peak_ids           [dataset: 500×str]
│   ├── frequencies_mhz    [dataset: 500×float64]
│   ├── amplitudes         [dataset: 500×float64]
│   ├── decay_rates        [dataset: 500×float64]
│   ├── phases             [dataset: 500×float64]
│   ├── frequency_errors   [dataset: 500×float64]
│   ├── amplitude_errors   [dataset: 500×float64]
│   ├── decay_rate_errors  [dataset: 500×float64]
│   ├── phase_errors       [dataset: 500×float64]
│   ├── snr_values         [dataset: 500×float64]
│   └── chi_squared_values [dataset: 500×float64]
├── window_results/         [group: per-window metadata]
│   ├── window_000/        [subgroup]
│   │   ├── success        [attribute: bool]
│   │   ├── aic            [attribute: float]
│   │   ├── reduced_chi2   [attribute: float]
│   │   ├── cost_function  [attribute: float]
│   │   ├── n_iterations   [attribute: int]
│   │   ├── peak_indices   [dataset: indices into fitted_peaks]
│   │   ├── covariance_matrix [dataset: N_params×N_params float64]
│   │   ├── shared_params  [subgroup]
│   │   └── fixed_params   [subgroup]
│   └── ...
└── algorithm_info/         [group: fitting method, parameters]
```

## File Format: HDF5

**Advantages**:
- Excellent for scientific data (large arrays, mixed types)
- Self-describing with metadata support
- Cross-platform compatibility  
- Supports compression (~50% additional reduction)
- Fast random access for large datasets
- Hierarchical organization matches our data structure

**File Naming Convention**:
```
cache/
├── experiment_2638_cache.h5    # Complete pipeline cache
├── experiment_2639_cache.h5
└── ...
```

## User Interface Design

### Automatic Caching (Default)
```python
# Pipeline automatically caches expensive operations
pipeline = Pipeline(experiment_path="2638/", cache_dir="cache/")

# Automatically resumes from cached results if available  
results = pipeline.run_full()

# Run from specific stage
noise_result = pipeline.run_from_stage("noise_estimation")
```

### Explicit Cache Control
```python
# Force recalculation of specific stages
pipeline.run_full(recalculate=["noise_estimation", "fitting"])

# Save/load intermediate results explicitly
complex_ft = pipeline.load_stage_result("complex_ft", "2638")
pipeline.save_stage_result("complex_ft", complex_ft, "2638")
```

### Visualization from Cache
```python
# Visualization works directly on cached results
from ftmwpipeline.visualization import plot_complex_ft, plot_noise_estimation

# Load cached data for plotting
complex_ft = Pipeline.load_cached_result("complex_ft", "2638")
noise_result = Pipeline.load_cached_result("noise_result", "2638")

# Generate plots without recomputation
plot_complex_ft(complex_ft, backend="plotly")
plot_noise_estimation(complex_ft.freq_array, complex_ft.magnitude, noise_result)
```

### On-Demand Reconstruction
```python
# Fast parameter access
results = pipeline.load_fitting_results("2638")
frequencies = [peak.frequency_mhz for peak in results.fitted_peaks]

# Reconstruct specific diagnostics when needed
fitted_spectrum = pipeline.reconstruct_fitted_spectrum("2638", window_id=5)
residuals = pipeline.compute_residuals("2638", window_id=5)
```

## Performance Summary

| Stage | Original Size | Cached Size | Reduction | Reconstruction Time |
|-------|---------------|-------------|-----------|-------------------|
| Data Loading | N/A | No cache | N/A | ~100ms (file I/O) |
| ComplexFT | ~9 MB | ~6 MB | 33% | <1ms (freq array) |
| NoiseResult | 3.4 MB | 155 KB | 95.4% | 10-50ms (exact) |
| Peak Detection | 75 KB | 75 KB | 0% | <1ms (direct load) |
| Window Assignment | 9 MB | 35 KB | 99.6% | 1-10ms (index slice) |
| Fitting Results | 9.4 MB | 330 KB | 96.5% | 1-5ms per window |
| **Total Pipeline** | **~31 MB** | **~6.6 MB** | **78.7%** | **Variable** |

## Implementation Benefits

### Performance
- Skip expensive FFT recalculation (~2-5 seconds saved)
- Skip adaptive noise estimation (~3-10 seconds saved)  
- Enable rapid parameter tuning on later stages
- Batch processing with checkpointing

### User Experience  
- Resume interrupted analysis sessions
- Separate computation from visualization
- Share intermediate results between collaborators
- Interactive parameter exploration

### Development
- Test later pipeline stages without recomputing early stages
- Debug specific stages in isolation
- Compare algorithm variations efficiently
- Comprehensive algorithm benchmarking

## Future Considerations

### Noise Modeling Research
Polynomial-based noise modeling was investigated but determined to be less suitable than convolution-based approaches for this application. The current signal indices + convolution reconstruction method provides optimal balance of storage efficiency and scientific accuracy.

### Cache Management
- Automatic cleanup of outdated caches
- Size-based LRU eviction policies
- Dependency tracking and validation
- Multi-experiment cache optimization

### Extended Serialization
- Batch processing results
- Algorithm comparison datasets  
- User annotation and metadata
- Version control integration

---

**Last Updated**: 2025-08-04  
**Status**: Complete serialization strategy defined