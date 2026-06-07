"""
Utility functions for FTMW pipeline.

This module provides:
- Signal processing utilities
- Statistical tests and calculations
- Physics-based calculations and conversions
"""

from .physics_utils import (
    calculate_line_strength,
    doppler_broadening,
    pressure_broadening,
)
from .signal_processing import (
    APODIZATION_EXAMPLES,
    apodize_fid,
    make_apodization,
    matched_filter_window,
)
from .statistical_tests import aic_comparison, chi_squared_test, f_test

__all__ = [
    "APODIZATION_EXAMPLES",
    "apodize_fid",
    "make_apodization",
    "matched_filter_window",
    "f_test",
    "aic_comparison",
    "chi_squared_test",
    "calculate_line_strength",
    "doppler_broadening",
    "pressure_broadening",
]
