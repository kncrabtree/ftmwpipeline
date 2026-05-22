"""
Peak fitting algorithms for FTMW spectroscopy.

This module provides:
- The finite-acquisition line-shape model (``peak_model``)
- The per-window least-squares core (``window_fit``)
- Conservative iterative fitting with statistical validation
- Physics-based parameter validation
"""

from .peak_model import (
    ModelPeak,
    baseband_offset,
    effective_tau,
    h_T,
    h_T_jacobian,
    model_spectrum,
    molecular_frequency,
    sideband_sign,
    to_baseband_frame,
)
from .window_fit import ParameterErrors, WindowFitResult, fit_window, model_jacobian
from .conservative import fit_conservative_time_domain
from .validation import validate_fit_results, check_physics_constraints

__all__ = [
    "ModelPeak",
    "baseband_offset",
    "effective_tau",
    "h_T",
    "h_T_jacobian",
    "model_spectrum",
    "molecular_frequency",
    "sideband_sign",
    "to_baseband_frame",
    "ParameterErrors",
    "WindowFitResult",
    "fit_window",
    "model_jacobian",
    "fit_conservative_time_domain",
    "validate_fit_results",
    "check_physics_constraints",
]
