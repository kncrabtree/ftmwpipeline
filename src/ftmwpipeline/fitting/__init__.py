"""
Peak fitting algorithms for FTMW spectroscopy.

This module provides:
- Unified time-domain fitting with damped cosines
- Conservative iterative fitting with statistical validation
- Physics-based parameter validation
"""

from .time_domain import fit_time_domain_peaks
from .conservative import fit_conservative_time_domain
from .validation import validate_fit_results, check_physics_constraints

__all__ = [
    "fit_time_domain_peaks",
    "fit_conservative_time_domain",
    "validate_fit_results",
    "check_physics_constraints",
]