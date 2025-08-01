"""
Core data structures and classes for FTMW pipeline.

This module contains the fundamental data structures used throughout the pipeline:
- SpectralWindow: Container for frequency domain data
- Peak: Individual spectral peak representation
- FittingResult: Results from fitting algorithms
- FIDParameters: Free induction decay parameters
"""

from .data_structures import (
    SpectralWindow,
    Peak, 
    FittingResult,
    FIDParameters,
)

from .fit_metrics import (
    calculate_chi_squared,
    calculate_aic,
    calculate_f_statistic,
    calculate_confidence_intervals,
)

__all__ = [
    "SpectralWindow",
    "Peak",
    "FittingResult", 
    "FIDParameters",
    "calculate_chi_squared",
    "calculate_aic",
    "calculate_f_statistic",
    "calculate_confidence_intervals",
]