"""
FTMW Pipeline - A Python package for FTMW spectroscopy signal processing and peak fitting.

This package provides tools for processing Fourier Transform Microwave (FTMW) 
spectroscopy data, including baseline estimation, peak detection, window assignment,
and advanced fitting algorithms.
"""

__version__ = "0.1.0"
__author__ = "FTMW Pipeline Contributors"

# Core data structures
from ftmwpipeline.core.data_structures import (
    FTMWData,
    FID,
    ComplexFT,
    SpectralWindow,
    Peak,
    FittedPeak,
    FittingResult,
    FIDProcessingParameters,
    PeakClassification,
    Sideband,
)

# Main pipeline interface
from ftmwpipeline.pipeline import Pipeline

# Functional API - can be imported as "import ftmwpipeline.api as ftmw"
from ftmwpipeline import api

# Convenience workflow functions (thin wrappers over Pipeline)
from ftmwpipeline.workflows import process_experiment, batch_process_experiments

# TODO: Fix imports for other modules when they're implemented
# # Preprocessing functions
# from ftmwpipeline.preprocessing.noise_estimation import estimate_noise_adaptive
# from ftmwpipeline.preprocessing.data_loading import load_blackchirp_data

# # Peak detection
# from ftmwpipeline.peak_detection.basic_detection import locate_peaks
# from ftmwpipeline.peak_detection.hybrid_detection import locate_peaks_hybrid
# from ftmwpipeline.peak_detection.classification import classify_peaks

# # Window assignment
# from ftmwpipeline.window_assignment.greedy_assignment import assign_analysis_windows

# # Fitting algorithms
# from ftmwpipeline.fitting.time_domain import fit_time_domain_peaks
# from ftmwpipeline.fitting.conservative import fit_conservative_time_domain
# from ftmwpipeline.fitting.validation import validate_fit_results

# # Import submodules to make them accessible
# from ftmwpipeline import (
#     core,
#     preprocessing, 
#     peak_detection,
#     window_assignment,
#     fitting,
#     visualization,
#     io,
#     config,
#     utils,
# )

__all__ = [
    # Version info
    "__version__",
    "__author__",
    
    # Core data structures
    "FTMWData",
    "FID",
    "ComplexFT", 
    "SpectralWindow",
    "Peak",
    "FittedPeak",
    "FittingResult",
    "FIDProcessingParameters",
    "PeakClassification",
    "Sideband",
    
    # Main pipeline interface
    "Pipeline",

    # Functional API module
    "api",

    # Convenience workflows
    "process_experiment",
    "batch_process_experiments",
]

# Package-level configuration
import logging

# Configure default logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Optional imports with graceful fallbacks
try:
    import matplotlib
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

try:
    import plotly
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

# Package metadata
PACKAGE_INFO = {
    "name": "ftmwpipeline",
    "version": __version__,
    "description": "FTMW spectroscopy signal processing and peak fitting",
    "has_matplotlib": _HAS_MATPLOTLIB,
    "has_plotly": _HAS_PLOTLY,
}