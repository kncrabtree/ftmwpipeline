"""
FTMW Pipeline - A Python package for FTMW spectroscopy signal processing and peak fitting.

This package provides tools for processing Fourier Transform Microwave (FTMW)
spectroscopy data, including baseline estimation, peak detection, window assignment,
and advanced fitting algorithms.
"""

__version__ = "0.1.0b2"
__author__ = "Kyle N. Crabtree"

# Functional API - can be imported as "import ftmwpipeline.api as ftmw"
from ftmwpipeline import api

# Core data structures
from ftmwpipeline.core.data_structures import (
    FID,
    ComplexFT,
    FIDProcessingParameters,
    FittedPeak,
    FittingResult,
    FitWindow,
    FixedContributor,
    FTMWData,
    Peak,
    PeakClassification,
    Sideband,
    SpectralWindow,
    WindowPlan,
)

# Main pipeline interface
from ftmwpipeline.pipeline import Pipeline

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
    "FixedContributor",
    "FitWindow",
    "WindowPlan",
    # Main pipeline interface
    "Pipeline",
    # Functional API module
    "api",
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

# Package metadata
PACKAGE_INFO = {
    "name": "ftmwpipeline",
    "version": __version__,
    "description": "FTMW spectroscopy signal processing and peak fitting",
    "has_matplotlib": _HAS_MATPLOTLIB,
}
