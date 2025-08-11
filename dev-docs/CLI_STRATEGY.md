# FTMW Pipeline - CLI Design Strategy

## Overview

This document outlines the design strategy for the ftmwpipeline command-line interface (CLI), which serves as a thin wrapper around the high-level Python API while providing an optimized user experience for terminal-based workflows.

## Design Philosophy

### Core Principles
1. **Thin API Wrapper**: CLI commands are lightweight wrappers around Python API functions
2. **File-Centric Interface**: All commands operate on `.ftmw` pipeline data files
3. **Single Responsibility**: Each command performs one specific pipeline stage or operation
4. **Explicit Operations**: Clear distinction between data import and analysis operations
5. **Progressive Enhancement**: Commands build upon previous stages stored in the data file
6. **No Code Duplication**: CLI and Python API share identical core implementation

### CLI as Service Interface
The CLI provides a command-line service interface to the ftmwpipeline functionality:
- **Stateless Operations**: Each command execution is independent
- **File-Based State**: Pipeline state persisted in `.ftmw` data files
- **Composable Workflow**: Commands can be chained and scripted
- **Batch Processing**: Suitable for automation and high-throughput analysis

## Command Architecture

### File-Centric Design
All CLI commands follow a consistent pattern:
```bash
ftmwpipeline <operation> <pipeline_file.ftmw> [options]
```

**Key Features**:
- First parameter is always the pipeline data file
- Operations are self-contained and can be run independently
- Clear error messages when dependencies missing
- Consistent option naming across all commands

### Command Categories

#### Stage Operations
Each pipeline stage has dedicated commands:
```bash
# Stage 0: Data Import
ftmwpipeline import-data exp_2638.ftmw --source examples/blackchirp_data/2638/

# Stage 1: FT Processing  
ftmwpipeline compute-ft exp_2638.ftmw --zpf 2 --expf_us 5.0
ftmwpipeline visualize-ft exp_2638.ftmw --trim 26500:40000

# Stage 2: Noise Estimation (future)
ftmwpipeline estimate-noise exp_2638.ftmw --skew-target 0.7

# Stage 3: Peak Detection (future)
ftmwpipeline detect-peaks exp_2638.ftmw --algorithm hybrid

# Stage 4: Window Assignment (future)  
ftmwpipeline assign-windows exp_2638.ftmw --max-peaks-per-window 5

# Stage 5: Fitting (future)
ftmwpipeline fit-peaks exp_2638.ftmw --algorithm conservative
```

#### Utility Operations
Supporting commands for pipeline management:
```bash
# Information and diagnostics
ftmwpipeline info exp_2638.ftmw              # Pipeline file info
ftmwpipeline validate exp_2638.ftmw          # Validate pipeline integrity
ftmwpipeline formats                          # Available data formats

# Visualization and export
ftmwpipeline export-data exp_2638.ftmw --format csv   # Export results
ftmwpipeline plot exp_2638.ftmw --stage ft --save     # Generate plots
```

## Implementation Strategy

### Code Reuse Architecture
```python
# Shared implementation pattern:
# _internal/stage1_impl.py
def _compute_ft_impl(pipeline_file: Path, zpf: int, expf_us: float, **kwargs):
    """Core FT computation logic shared by CLI and API."""
    # Implementation here
    return complex_ft

# CLI command (cli/commands.py)  
def cmd_compute_ft(args):
    """CLI wrapper for FT computation."""
    result = _compute_ft_impl(
        pipeline_file=Path(args.pipeline_file),
        zpf=args.zpf,
        expf_us=args.expf_us,
        trim=parse_trim_range(args.trim),
        verbose=args.verbose
    )
    print("✅ FT computation complete")
    return 0

# Python API (api.py)
def compute_ft(pipeline_file: str, zpf: int = 1, expf_us: float = 5.0, **kwargs):
    """Python API for FT computation.""" 
    return _compute_ft_impl(Path(pipeline_file), zpf=zpf, expf_us=expf_us, **kwargs)

# Pipeline class (pipeline.py)
def compute_ft(self, zpf: int = 1, expf_us: float = 5.0, **kwargs):
    """Pipeline class method for FT computation."""
    return _compute_ft_impl(self._file_path, zpf=zpf, expf_us=expf_us, **kwargs)
```

### Parameter Consistency
- **Shared Validation**: Common parameter validation logic
- **Consistent Naming**: Same parameter names across CLI and API
- **Type Conversion**: CLI handles string→type conversion, API uses native types
- **Default Values**: Shared default parameter definitions

### Error Handling
```python
# Shared exception hierarchy
class PipelineError(Exception): pass
class PipelineFileNotFoundError(PipelineError): pass  
class PipelineStageError(PipelineError): pass
class PipelineDataError(PipelineError): pass

# CLI error translation
def handle_cli_error(e: Exception) -> int:
    """Convert Python exceptions to CLI exit codes and messages."""
    if isinstance(e, PipelineFileNotFoundError):
        print(f"❌ Pipeline file not found: {e}")
        print("💡 Use 'import-data' to create a new pipeline file")
        return 1
    elif isinstance(e, PipelineStageError):
        print(f"❌ Stage dependency missing: {e}")
        return 1
    # ... etc
```

## User Experience Design

### Command Naming Strategy
- **Verb-Object Pattern**: `import-data`, `compute-ft`, `detect-peaks`
- **Clear Intent**: Distinguish creation (`import-data`) vs. analysis (`visualize-ft`)
- **Stage Alignment**: Command names reflect pipeline stage purposes
- **Consistent Prefixes**: Related commands share prefixes (`visualize-*`, `export-*`)

### Progress and Feedback
```bash
# Informative output with progress indicators
$ ftmwpipeline compute-ft exp_2638.ftmw --zpf 2 --expf_us 5.0

🔍 Loading FID data from exp_2638.ftmw...
📊 Found FID with 750,000 points (15.0 μs duration)
⚡ Computing FT with parameters:
   zpf: 2, expf_us: 5.0 μs
🔄 Processing... ████████████████ 100%
✅ FT computation complete (1.2s)
💾 Results stored in exp_2638.ftmw
```

### Error Messages and Guidance
```bash
# Clear error messages with actionable guidance
$ ftmwpipeline compute-ft missing.ftmw

❌ Pipeline file not found: missing.ftmw
💡 To create a new pipeline file:
   ftmwpipeline import-data missing.ftmw --source path/to/data/

$ ftmwpipeline detect-peaks exp_2638.ftmw

❌ Missing dependency: FT processing stage not completed
💡 Run FT processing first:  
   ftmwpipeline compute-ft exp_2638.ftmw --zpf 2 --expf_us 5.0
```

## Pipeline Data File Integration

### .ftmw File Convention
- **Single Source of Truth**: All pipeline state in one file per experiment
- **Portable Analysis**: Files can be shared, moved, and archived
- **Self-Contained**: No external dependencies once created
- **Version Tracking**: Internal metadata tracks pipeline version and operations

### File Operations
```bash
# CLI handles file paths transparently
ftmwpipeline import-data /path/to/experiment.ftmw --source data/
ftmwpipeline compute-ft ./analysis/exp_2638.ftmw --zpf 2
ftmwpipeline visualize-ft ~/results/final_analysis.ftmw

# File info and validation  
ftmwpipeline info experiment.ftmw
# Shows: stages completed, file size, creation date, source info, etc.
```

### Stage Dependencies
```python
# Automatic dependency checking
def check_stage_dependencies(pipeline_file: Path, required_stages: List[str]):
    """Verify required stages are complete before proceeding."""
    # Implementation checks HDF5 file for completed stages
    # Raises clear error if dependencies missing
```

## Scripting and Automation Support

### Batch Processing
```bash
# Process multiple experiments
for exp in experiments/*.ftmw; do
    ftmwpipeline compute-ft "$exp" --zpf 2 --expf_us 5.0
    ftmwpipeline detect-peaks "$exp" --algorithm hybrid
done

# Pipeline configuration files (future)
ftmwpipeline batch-process config.json --parallel 4
```

### Return Codes and Scripting
- **Exit Code 0**: Successful completion
- **Exit Code 1**: User error (wrong parameters, missing files, etc.)
- **Exit Code 2**: Data processing error (corrupted data, algorithm failure)
- **Exit Code 130**: User interruption (Ctrl+C)

### JSON Output Mode
```bash
# Machine-readable output for integration
ftmwpipeline info exp_2638.ftmw --format json
{
  "file": "exp_2638.ftmw",
  "size_mb": 6.2,
  "stages_complete": ["import", "ft"],
  "source_info": {...},
  "ft_params": {...}
}
```

## Comparison with Current CLI

### Migration Path
```bash
# Current CLI (experiment ID + cache directory)
ftmwpipeline data-load exp_2638 --source examples/blackchirp_data/2638/ --cache-dir cache/
ftmwpipeline ft-visualize exp_2638 --zpf 2 --cache-dir cache/

# New CLI (single pipeline file)  
ftmwpipeline import-data exp_2638.ftmw --source examples/blackchirp_data/2638/
ftmwpipeline visualize-ft exp_2638.ftmw --zpf 2
```

### Benefits of New Design
1. **Simpler**: No cache directory management
2. **Clearer**: File-based operations easier to understand
3. **Portable**: Pipeline files can be moved and shared
4. **Safer**: No accidental cache overwrites
5. **Scriptable**: Better automation support

## Implementation Plan

### Phase 1: Core Commands (Stages 0-1)
```bash
ftmwpipeline import-data <file.ftmw> --source <path> [--format <fmt>]
ftmwpipeline compute-ft <file.ftmw> [--zpf N] [--expf_us X] [--trim X:Y]
ftmwpipeline visualize-ft <file.ftmw> [options] [--save] [--output <path>]
ftmwpipeline info <file.ftmw> [--format json]
```

### Phase 2: Extended Commands (Stages 2-3)
```bash
ftmwpipeline estimate-noise <file.ftmw> [--skew-target X] [options]
ftmwpipeline visualize-noise <file.ftmw> [options]
ftmwpipeline detect-peaks <file.ftmw> [--algorithm <alg>] [--snr-threshold X]
ftmwpipeline visualize-peaks <file.ftmw> [options]
```

### Phase 3: Complete Pipeline (Stages 4-5)
```bash
ftmwpipeline assign-windows <file.ftmw> [--max-peaks-per-window N]
ftmwpipeline fit-peaks <file.ftmw> [--algorithm <alg>] [options]
ftmwpipeline export-results <file.ftmw> --format <fmt> --output <path>
```

## Integration with Development Workflow

### Testing Strategy
- **CLI Integration Tests**: Test complete command workflows
- **Shared Logic Tests**: Core implementation thoroughly unit tested  
- **Error Handling Tests**: Verify proper error codes and messages
- **File Format Tests**: Ensure .ftmw file compatibility

### Documentation
- **Man Pages**: Traditional Unix documentation
- **Help Text**: Comprehensive --help output for each command
- **Examples**: Real workflow examples in documentation
- **Tutorial**: Step-by-step getting started guide

## Future Enhancements

### Advanced Features (Future)
```bash
# Interactive mode
ftmwpipeline interactive exp_2638.ftmw

# Comparison tools
ftmwpipeline compare exp_A.ftmw exp_B.ftmw --stage ft

# Parameter optimization  
ftmwpipeline optimize-params exp_2638.ftmw --stage ft --metric snr

# Batch processing
ftmwpipeline batch-process pattern*.ftmw --config batch_config.json
```

### Integration Points
- **Workflow Managers**: Nextflow, Snakemake integration
- **Container Support**: Docker/Singularity containers
- **Cloud Processing**: Support for cloud storage and compute
- **Database Export**: Direct export to spectroscopy databases

## Benefits Summary

### For Users
- **Intuitive**: File-based operations match mental model
- **Reliable**: Clear error messages and dependency checking
- **Scriptable**: Easy automation and batch processing
- **Portable**: Self-contained analysis files

### For Developers
- **Maintainable**: Shared implementation prevents code duplication
- **Testable**: Clear separation of concerns enables thorough testing
- **Extensible**: Consistent patterns for new commands
- **Debuggable**: Clear error propagation and logging

This CLI strategy provides a solid foundation for user-friendly terminal workflows while maintaining clean architecture and code reuse with the Python API.

---

**Last Updated**: 2025-01-11  
**Status**: Design complete, ready for implementation  
**Dependencies**: Requires API_STRATEGY.md implementation as foundation