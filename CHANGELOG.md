# Changelog

All notable changes to the ftmwpipeline project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-08-01 (In Development)

### Added

#### Phase 1: Package Infrastructure ✅ COMPLETE
- Complete package directory structure under `src/ftmwpipeline/`
- Modern Python packaging with `pyproject.toml` configuration
- Comprehensive module structure with placeholder APIs:
  - `core/` - Data structures (SpectralWindow, Peak, FittingResult, FIDParameters)
  - `preprocessing/` - Data loading, baseline estimation, validation
  - `peak_detection/` - Basic and hybrid peak detection algorithms  
  - `window_assignment/` - Greedy assignment and optimization
  - `fitting/` - Time-domain and conservative fitting algorithms
  - `visualization/` - Plotting and diagnostics
  - `io/` - Input/output and logging
  - `config/` - Configuration management
  - `utils/` - Signal processing and statistical utilities

- Main Pipeline class for orchestrating analysis workflow
- Workflow convenience functions (`process_experiment`, `batch_process_experiments`, `quick_fit`)
- Complete pytest testing framework with fixtures and test structure
- Sphinx documentation framework with API reference and examples
- Command-line interface (`ftmwpipeline` command)
- Development tools configuration (black, isort, flake8, mypy, pre-commit)
- Installation validation utilities

#### Dependencies
- Core: numpy, scipy, matplotlib, pandas, h5py, pyyaml, tqdm
- Development: pytest, black, isort, flake8, mypy, pre-commit
- Documentation: sphinx, sphinx-rtd-theme, nbsphinx
- Optional: plotly, bokeh, seaborn (visualization extras)

#### Project Structure
```
ftmwpipeline/
├── pyproject.toml              # Modern packaging configuration
├── src/ftmwpipeline/           # Source code
├── tests/                      # Test suite (unit, integration, performance)
├── docs/                       # Sphinx documentation
├── examples/                   # Usage examples and notebooks
├── scripts/                    # Utility scripts
├── CHANGELOG.md               # This file
├── README.md                  # Project description
└── LICENSE                    # MIT License
```

### Development Roadmap

#### Upcoming Phases

- **Phase 2** (1 week): Core Data Structures
  - SpectralWindow, Peak, FittingResult, FIDParameters classes
  - Statistical metrics and fit validation

- **Phase 3** (1 week): Preprocessing Pipeline  
  - BlackChirp data loading
  - Baseline and noise estimation
  - Data validation

- **Phase 4** (1 week): Peak Detection
  - Basic second derivative method
  - Hybrid clustering and iterative subtraction
  - SNR-based classification

- **Phase 5** (1 week): Window Assignment
  - Greedy window assignment algorithm
  - Physics-based sizing and optimization

- **Phase 6** (2 weeks): Fitting Algorithms ⚠️ Most Complex
  - Unified time-domain fitting with decay constraints
  - Conservative iterative fitting with statistical validation
  - Physics-based parameter validation

- **Phase 7** (3-4 days): Logging and IO
  - Comprehensive decision logging
  - HDF5, JSON, CSV serialization
  - Experimental data format support

- **Phase 8** (3-4 days): Visualization
  - Multi-panel fit diagnostics
  - Pipeline stage visualization  
  - HTML/PDF report generation

- **Phase 9** (2-3 days): Configuration and Pipeline
  - YAML/JSON configuration management
  - Complete Pipeline orchestration
  - Error handling and recovery

- **Phase 10** (1 week): Testing and Documentation
  - >90% test coverage target
  - Performance benchmarks vs. current implementation
  - Complete API documentation and user guide

- **Phase 11** (2-3 days): Packaging and Release
  - PyPI package preparation
  - GitHub releases and CI/CD
  - Installation guides

### Source Code Migration

All algorithms will be extracted from `/home/kncrabtree/github/bcfitting/`:

#### Key Files for Migration:
- `src/bcfitting/ftmwfitting.py` (lines 1095-2900+) - Core algorithms
- `newfitting/time_domain_fitting_unified.py` - Advanced fitting
- `newfitting/complex_ft.py` - Data structures
- `newfitting/peak_classification.py` - Peak detection
- `newfitting/window_assignment.py` - Window assignment
- `newfitting/conservative_fitting_logger.py` - Logging system
- `newfitting/unified_fitting_visualization.py` - Visualization

#### Test Data:
- `examples/blackchirp_data/2638/` - Real experimental data
- `newfitting/output/` - Reference results for validation

### Performance Targets

- Processing time within 10% of current bcfitting implementation
- Handle 150+ weak spectral windows robustly  
- Memory usage scales linearly with data size
- Support large datasets (>1M frequency points)

### Quality Standards

- Target >90% test coverage across all modules
- Comprehensive API documentation with examples
- Professional packaging for PyPI distribution
- Clean separation of concerns and modular design
- Preserve all physics-based validation logic

---

**Note**: This is a scientific data analysis package focused on defensive security applications only. All algorithms are for spectroscopy signal processing and contain no malicious functionality.