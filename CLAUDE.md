# FTMW Pipeline Project - Claude Assistant Guide

## Project Overview

This project is developing `ftmwpipeline`, a standalone Python package for FTMW (Fourier Transform Microwave) spectroscopy signal processing and peak fitting. The goal is to extract and refactor algorithms from the experimental work in `/home/kncrabtree/github/bcfitting/newfitting/` into a clean, well-structured Python package.

## Repository Structure

This project uses git worktrees, allowing multiple working directories for different branches:
- **Bare repository**: `../ftmwpipeline.git/` (contains git objects and refs)
- **Working directories**: Each branch can have its own worktree directory
- **This directory**: Current working directory for whichever branch you're on

### Typical Structure
```
ftmwpipeline/
├── ftmwpipeline.git/           # Bare git repository
├── main/                       # Main branch worktree
├── feature-xyz/                # Feature branch worktree (example)
└── dev/                        # Development branch worktree (example)
```

### Current Directory Contents
```
├── FTMWPIPELINE_DEVELOPMENT_ROADMAP.md    # Comprehensive project roadmap
├── CLAUDE.md                              # This file - assistant guide
├── LICENSE                                # Project license
└── README.md                              # Basic project description
```

**Recommended Workflow**: Always start Claude Code sessions from within a worktree directory (like this one) for development work.

## Source Code Location

**IMPORTANT**: All source code to be migrated is located at:
`/home/kncrabtree/github/bcfitting/`

### Key Source Files to Extract:
- `/home/kncrabtree/github/bcfitting/src/bcfitting/ftmwfitting.py` (lines 1095-2900+)
- `/home/kncrabtree/github/bcfitting/newfitting/time_domain_fitting_unified.py`
- `/home/kncrabtree/github/bcfitting/newfitting/complex_ft.py`
- `/home/kncrabtree/github/bcfitting/newfitting/peak_classification.py`
- `/home/kncrabtree/github/bcfitting/newfitting/window_assignment.py`
- `/home/kncrabtree/github/bcfitting/newfitting/conservative_fitting_logger.py`
- `/home/kncrabtree/github/bcfitting/newfitting/unified_fitting_visualization.py`

### Test Data Location:
- `/home/kncrabtree/github/bcfitting/examples/blackchirp_data/2638/` - Real experimental data
- `/home/kncrabtree/github/bcfitting/newfitting/output/` - Reference outputs
- **Example data**: `examples/blackchirp_data/2638/` - Local copy for testing

### Experiment 2638 Processing Notes:
**Recommended Processing Parameters**:
- `zpf=1` (zero padding factor for improved frequency resolution)
- `expf_us=5.0` (5 μs exponential apodization filter for sensitivity enhancement)
- **Activity region**: 26500-40000 MHz (focus analysis in this range)
- **IMPORTANT**: FT should be trimmed to 26500-40000 MHz range before analysis to remove noise regions
- FID specs: 750k points, 15 μs duration, 40.96 GHz probe, Lower Sideband

**Example Usage**:
```python
# Load and process experiment 2638
ftmw_data = load_blackchirp_experiment("examples/blackchirp_data/2638", fid_index=0)
complex_ft = ftmw_data.fid.ft(zpf=1, expf_us=5.0)

# Trim to activity region before analysis
trimmed_ft = complex_ft.trim_to_range(26500, 40000)
```

## Target Package Structure

The final package will follow this structure (see roadmap for complete details):
```
src/ftmwpipeline/
├── core/                    # Data structures (SpectralWindow, Peak, FittingResult)
├── preprocessing/           # Data loading, baseline estimation, validation
├── peak_detection/          # Basic and hybrid peak detection algorithms
├── window_assignment/       # Greedy assignment and optimization
├── fitting/                 # Time-domain and conservative fitting algorithms
├── visualization/           # Plotting and diagnostics (optional)
├── io/                      # Input/output and logging
├── config/                  # Configuration management
└── utils/                   # Signal processing and statistical utilities
```

## Development Phases

**Current Status**: Phase 1 ✅ **COMPLETE** - Phase 2 **READY TO BEGIN**

The project is organized into 11 development phases. For complete details, current status, task breakdowns, and timeline, see the [**Development Roadmap**](FTMWPIPELINE_DEVELOPMENT_ROADMAP.md).

## Build/Test Instructions

### Current Implementation ✅
The package infrastructure is now complete with:

- **Package Manager**: Modern Python packaging with `pyproject.toml`
- **Testing Framework**: pytest with fixtures and coverage
- **Documentation**: Sphinx with API reference
- **Build Tools**: Standard Python build tools
- **Environment**: Conda environment files

### Development Workflow
```bash
# Set up environment
conda env create -f environment.yml
conda activate ftmwpipeline

# Install in development mode
pip install -e .

# Run tests
pytest                             # Run all tests
pytest --cov=ftmwpipeline         # Run tests with coverage
ftmwpipeline validate             # Validate installation

# Build documentation (future)
sphinx-build docs/source docs/build
```

## Key Algorithms to Migrate

### Peak Detection
- `locate_peaks()` - Basic second derivative-based detection
- `locate_peaks_hybrid()` - Advanced clustering and iterative subtraction

### Fitting Algorithms
- `fit_time_domain_peaks()` - Unified time-domain fitting with decay constraints
- `fit_weak_window_conservative_time_domain()` - Conservative iterative fitting
- `validate_fit_results()` - Physics-based validation

### Data Processing
- `estimate_baseline_noise()` - Frequency-dependent baseline/noise estimation
- `assign_analysis_windows()` - Greedy window assignment algorithm
- `find_and_classify_peaks()` - SNR-based peak classification

## Performance Requirements

- Processing time within 10% of current implementation
- Handle 150+ weak spectral windows robustly
- Support large datasets (>1M frequency points)
- Memory usage scales linearly with data size

## Quality Standards

- Target >90% test coverage
- Comprehensive API documentation
- Professional packaging for PyPI
- Clean separation of concerns
- Physics-based validation preserved

## Next Steps

1. **Phase 1**: Create package directory structure in `main/`
2. Set up `pyproject.toml` with dependencies
3. Initialize basic module structure
4. Set up testing framework
5. Begin extracting core data structures

## Important Notes

- This is a **defensive security** project focused on scientific data analysis
- All algorithms are for spectroscopy signal processing (non-malicious)
- Source code extraction involves scientific computing functions only
- Focus on clean architecture and maintainable code
- Preserve all physics-based constraints and validation logic