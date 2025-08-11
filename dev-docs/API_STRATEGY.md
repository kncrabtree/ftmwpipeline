# FTMW Pipeline - High-Level Python API Strategy

## Overview

This document outlines the design strategy for the high-level Python API for ftmwpipeline Stages 0-1, providing clean abstractions that mirror CLI functionality while preventing code duplication and ensuring safe file management.

## Design Philosophy

### Core Principles
1. **Single File = Single Experiment**: Each `.ftmw` pipeline data file contains one complete FID analysis
2. **User-Controlled File Locations**: Users specify file paths, internal HDF5 structure is hidden
3. **Explicit Intent**: Clear distinction between creating new analyses vs. working with existing files
4. **Jupyter-Safe**: Designed for safe re-execution in interactive environments
5. **No Code Duplication**: CLI and API share core implementation logic

### File-Centric Design
- Pipeline data files use `.ftmw` extension for clarity
- Files are completely portable and self-contained
- Each file progresses through pipeline stages: FID → FT → Noise → Peaks → Fitting
- Users work with file paths, not experiment IDs or cache directories

## API Design

### 1. Pipeline Class (Primary Interface)

The Pipeline class represents a single experiment analysis bound to a specific file:

```python
from ftmwpipeline import Pipeline

# Create new pipeline from raw data
pipe = Pipeline.create("exp_2638.ftmw", source='examples/blackchirp_data/2638/')

# Open existing pipeline for analysis
pipe = Pipeline.open("exp_2638.ftmw")

# Smart constructor (convenience method)
pipe = Pipeline("exp_2638.ftmw")  # Opens if exists, clear error if not

# Stage 0: Data import is handled during creation
# (no separate import step needed after Pipeline.create)

# Stage 1: FT Processing
pipe.compute_ft(zpf=2, expf_us=5.0, trim=(26500, 40000))
pipe.visualize_ft(zpf=1, expf_us=3.0, save_params=True)

# Future stages (extensible design)
pipe.estimate_noise()
pipe.detect_peaks()
pipe.assign_windows()
pipe.fit_peaks()
```

**Key Features**:
- No experiment IDs needed - the instance IS the experiment
- Methods don't require repetitive file/experiment parameters
- Clear creation vs. opening semantics
- Extensible to all pipeline stages

### 2. Functional API (File-Based)

For users who prefer functional interfaces or need to work with multiple files:

```python
import ftmwpipeline as ftmw

# Stage 0: Import raw data into new pipeline file
ftmw.import_data("exp_2638.ftmw", source='examples/blackchirp_data/2638/')

# Stage 1: FT processing with existing pipeline file
complex_ft = ftmw.compute_ft("exp_2638.ftmw", zpf=2, expf_us=5.0)
ftmw.visualize_ft("exp_2638.ftmw", trim=(26500, 40000))

# Work with multiple experiments
ftmw.import_data("experiment_A.ftmw", source='data_A/')
ftmw.import_data("experiment_B.ftmw", source='data_B/')
result_A = ftmw.compute_ft("experiment_A.ftmw", zpf=2)
result_B = ftmw.compute_ft("experiment_B.ftmw", zpf=1)
```

**Key Features**:
- First parameter is always the pipeline file path
- Stateless functions for batch processing workflows
- Same underlying implementation as Pipeline class

## Safe File Management

### The Re-Import Problem
**Issue**: In Jupyter notebooks, users often re-run cells. Re-importing data could invalidate hours of downstream analysis.

**Solution**: Explicit creation vs. opening with smart source detection.

### Creation vs. Opening Semantics

#### Pipeline.create() - Explicit New Analysis
```python
# Case 1: File doesn't exist - create new
pipe = Pipeline.create("new_exp.ftmw", source='data/')  # ✅ Creates file

# Case 2: File exists with identical source - load existing  
pipe = Pipeline.create("exp_2638.ftmw", source='examples/blackchirp_data/2638/')
# ✅ Detects identical source, loads existing with info message:
# "ℹ️ Found existing pipeline with identical source. Loading existing data."

# Case 3: File exists with different source - explicit choice required
pipe = Pipeline.create("exp_2638.ftmw", source='different_data/')  
# ❌ Raises PipelineExistsError with clear options

# Case 4: Explicit overwrite when needed
pipe = Pipeline.create("exp_2638.ftmw", source='different_data/', force=True)
# ⚠️ Overwrites with warning
```

#### Pipeline.open() - Work with Existing Analysis
```python
# Opens existing file for analysis (safe to repeat)
pipe = Pipeline.open("exp_2638.ftmw")

# Clear error if file doesn't exist
pipe = Pipeline.open("missing.ftmw")  # FileNotFoundError with guidance
```

### Source Tracking System
```python
# Internal metadata stored in pipeline file:
class SourceMetadata:
    source_path: Path
    source_mtime: float      # File modification time
    source_hash: str         # Quick hash of key properties
    import_timestamp: datetime
    format_name: str
    loader_parameters: dict
```

**Smart Behavior**:
- Compare source path, modification time, and loader parameters
- If identical: load existing with informative message
- If different: require explicit user choice
- No silent behavior - users always informed of actions

## Jupyter-Friendly Patterns

### Pattern A: Explicit Creation (Recommended)
```python
# Cell 1: Create new analysis (run once)
pipe = Pipeline.create("my_analysis.ftmw", source='raw_data/')

# Cell 2: Open for analysis (safe to re-run)
pipe = Pipeline.open("my_analysis.ftmw")
pipe.visualize_ft(zpf=2)
```

### Pattern B: Smart Constructor (Convenience)
```python
# Cell 1: Smart creation/opening
try:
    pipe = Pipeline.open("my_analysis.ftmw")
    print("📂 Opened existing analysis")
except FileNotFoundError:
    pipe = Pipeline.create("my_analysis.ftmw", source='raw_data/')
    print("📥 Created new analysis from raw data")

# Cell 2: Analysis (always safe to re-run)
pipe.visualize_ft(zpf=2)
```

### Pattern C: Development Helper
```python
# Single cell that's safe to re-run during development
def get_or_create_pipeline(filename, source_path):
    try:
        return Pipeline.open(filename)
    except FileNotFoundError:
        return Pipeline.create(filename, source=source_path)

pipe = get_or_create_pipeline("my_analysis.ftmw", "raw_data/")
```

## CLI Integration

### Updated Command Structure
```bash
# Import raw data (creates new .ftmw file)
ftmwpipeline import-data exp_2638.ftmw --source examples/blackchirp_data/2638/

# Work with existing pipeline file  
ftmwpipeline visualize-ft exp_2638.ftmw --zpf 2
ftmwpipeline compute-ft exp_2638.ftmw --zpf 2 --expf_us 5.0

# Future stages
ftmwpipeline estimate-noise exp_2638.ftmw
ftmwpipeline detect-peaks exp_2638.ftmw
```

**Changes from Current CLI**:
- `data-load` → `import-data` (clearer intent)
- Experiment ID + cache-dir → single .ftmw filename
- Consistent file-based interface across all commands

### CLI-API Code Sharing
```python
# Internal implementation pattern
def _compute_ft_impl(pipeline_file: Path, **kwargs):
    """Shared implementation for both CLI and API"""
    # Parameter validation, FT computation, etc.
    pass

# CLI command
def cmd_compute_ft(args):
    return _compute_ft_impl(Path(args.pipeline_file), 
                           zpf=args.zpf, expf_us=args.expf_us, ...)

# Pipeline class method  
def compute_ft(self, **kwargs):
    return _compute_ft_impl(self._file_path, **kwargs)

# Functional API
def compute_ft(pipeline_file: str, **kwargs):
    return _compute_ft_impl(Path(pipeline_file), **kwargs)
```

## Implementation Structure

### Directory Organization
```
src/ftmwpipeline/
├── api.py              # Functional API (file-based functions)
├── pipeline.py         # Pipeline class (file-based instance)  
├── file_manager.py     # Single file operations abstraction
├── cli/
│   └── commands.py     # All CLI commands (filename-based)
└── _internal/          # Shared implementation functions
    ├── stage0_impl.py  # Data import implementation
    ├── stage1_impl.py  # FT processing implementation
    └── ...
```

### Code Reuse Strategy
1. **Shared Implementation Functions**: Core logic in `_internal/` modules
2. **Thin Interface Layers**: API, Pipeline class, and CLI are thin wrappers
3. **Consistent Error Handling**: Same exceptions and messages across interfaces
4. **Parameter Validation**: Shared validation logic prevents divergence

## Extension to Future Stages

This design naturally extends to future pipeline stages:

```python
# Stage 2: Noise Estimation
pipe.estimate_noise(skew_target=0.7, min_bin_fraction=0.025)
ftmw.estimate_noise("exp.ftmw", skew_target=0.7)

# Stage 3: Peak Detection  
peaks = pipe.detect_peaks(algorithm='hybrid', snr_threshold=5.0)
peaks = ftmw.detect_peaks("exp.ftmw", algorithm='hybrid')

# Stage 4: Window Assignment
windows = pipe.assign_windows(max_peaks_per_window=5)
windows = ftmw.assign_windows("exp.ftmw", max_peaks_per_window=5)

# Stage 5: Fitting
results = pipe.fit_peaks(algorithm='conservative')
results = ftmw.fit_peaks("exp.ftmw", algorithm='conservative')
```

Each stage follows the same pattern:
- Takes parameters for that specific stage
- Loads dependencies from previous stages automatically
- Stores results in the same pipeline file
- Provides both class method and functional interfaces

## Benefits Summary

### User Experience
- **Intuitive**: One file per experiment, clear creation/opening semantics
- **Safe**: No accidental data loss from re-running import commands
- **Flexible**: Choose between class-based or functional interfaces
- **Jupyter-Friendly**: Safe cell re-execution patterns

### Code Quality  
- **No Duplication**: CLI and API share implementation
- **Maintainable**: Clear separation between interface and implementation
- **Extensible**: Consistent pattern for all pipeline stages
- **Testable**: Shared logic can be thoroughly unit tested

### Data Management
- **Portable**: Self-contained .ftmw files can be shared and moved
- **Traceable**: Source metadata enables provenance tracking  
- **Efficient**: HDF5-based storage with stage-specific optimizations
- **Reliable**: Explicit file operations prevent silent data loss

## Implementation Timeline

This API strategy supports the current roadmap:

1. **Immediate (Stages 0-1)**: Implement Pipeline class and functional API for data import and FT processing
2. **Short Term (Stage 2)**: Extend to noise estimation with same patterns
3. **Medium Term (Stages 3-5)**: Complete pipeline with peak detection, window assignment, and fitting
4. **Long Term**: Advanced features like batch processing, parameter optimization, and integration tools

The file-centric design provides a solid foundation that will scale naturally as the pipeline grows in complexity.

---

**Last Updated**: 2025-01-11  
**Status**: Design complete, ready for implementation  
**Next Steps**: Implement Pipeline class and functional API for Stages 0-1