# ftmwpipeline

A Python package for FTMW (Fourier Transform Microwave) spectroscopy signal processing and peak fitting.

## Overview

This package provides a comprehensive pipeline for processing FTMW spectroscopy data, including:

- **Data Loading**: Support for BlackChirp and generic FID formats
- **Preprocessing**: Automated baseline and noise estimation
- **Peak Detection**: Advanced algorithms with clustering and iterative subtraction
- **Window Assignment**: Physics-based analysis window optimization  
- **Peak Fitting**: Time-domain and conservative fitting with statistical validation
- **Visualization**: Comprehensive plotting and diagnostic tools
- **Configuration**: Flexible parameter management and algorithm selection

## Installation

### Using Conda (Recommended)

Create a conda environment with all dependencies:

```bash
# Full environment with all features
conda env create -f environment.yml
conda activate ftmwpipeline

# Or minimal development environment  
conda env create -f environment-dev.yml
conda activate ftmwpipeline-dev

# Install the package in development mode
pip install -e .
```

### Using pip

```bash
pip install ftmwpipeline
```

### Development Installation

```bash
git clone https://github.com/ftmw-pipeline/ftmwpipeline.git
cd ftmwpipeline

# Create conda environment
conda env create -f environment.yml
conda activate ftmwpipeline

# Install in development mode
pip install -e .
```

## Quick Start

```python
import ftmwpipeline as fmw

# Process a single experiment
results = fmw.process_experiment('data/experiment.h5')

# Or use the Pipeline class for more control
pipeline = fmw.Pipeline()
results = pipeline.process_experiment('data/experiment.h5')

# Batch processing
results = fmw.batch_process_experiments(['exp1.h5', 'exp2.h5'])
```

## Development Status

**Current Phase**: Phase 1 (Infrastructure) ✅ **COMPLETE** - Phase 2 (Core Data Structures) **READY TO BEGIN**

This project is being developed in 11 phases. For complete development status, timeline, and detailed task breakdown, see the [**Development Roadmap**](FTMWPIPELINE_DEVELOPMENT_ROADMAP.md).

## Validation

To verify your installation:

```bash
# Command line
ftmwpipeline validate

# Or in Python
python scripts/validate_installation.py
```

## Documentation

- [Installation Guide](docs/source/installation.rst)
- [Quick Start Guide](docs/source/quickstart.rst)
- [API Reference](docs/source/api/index.rst)
- [Examples](examples/)

## Contributing

This project is in active development. See [CHANGELOG.md](CHANGELOG.md) for detailed development progress.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
