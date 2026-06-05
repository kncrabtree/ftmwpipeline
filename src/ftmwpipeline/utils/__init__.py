"""
Utility functions for FTMW pipeline.

This module provides:
- Signal processing utilities
- Statistical tests and calculations
- Physics-based calculations and conversions
"""

from .signal_processing import (
    apply_window_function,
    calculate_fwhm,
    frequency_to_time_domain,
    time_to_frequency_domain,
)
from .statistical_tests import f_test, aic_comparison, chi_squared_test
from .physics_utils import (
    calculate_line_strength,
    doppler_broadening,
    pressure_broadening,
)

__all__ = [
    "apply_window_function",
    "calculate_fwhm",
    "frequency_to_time_domain",
    "time_to_frequency_domain",
    "f_test",
    "aic_comparison",
    "chi_squared_test",
    "calculate_line_strength",
    "doppler_broadening",
    "pressure_broadening",
]
