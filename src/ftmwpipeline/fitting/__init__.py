"""
Peak fitting algorithms for FTMW spectroscopy.

This module provides:
- The finite-acquisition line-shape model (``peak_model``)
- The per-window least-squares core and conservative add-one-peak loop
  (``window_fit``)
- Statistical-test and linewidth-physics helpers (``validation``)
- Plan-level execution: fixed-contributor evaluation, DAG/batch walk, and
  local thaw renegotiation (``plan_execution``)
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
from .validation import (
    calculate_aic,
    calculate_chi_squared_improvement,
    calculate_hwhm_from_apodization,
    calculate_noise_weighted_chi2,
    calculate_rms_residuals,
    feature_fwhm,
    passes_significance_test,
    validate_peak_separation,
)
from .plan_execution import (
    DEFAULT_MAX_THAW_ROUNDS,
    DEFAULT_RESIDUAL_EDGE_M,
    DEFAULT_RESIDUAL_EDGE_THRESHOLD,
    FrozenPeak,
    PlanFitOutcome,
    ThawEvent,
    WindowOutcome,
    attempt_thaw_round,
    evaluate_fixed_contributor,
    execute_plan,
    fit_window_with_fixed_contributors,
    local_thaw_cofit,
    residual_edge_coherence,
    select_contributor_to_thaw,
    subtract_frozen_background,
)
from .window_fit import (
    AddStep,
    ConservativeFitResult,
    KnockoutResult,
    ParameterErrors,
    WindowFitResult,
    conservative_fit,
    fit_window,
    knockout_test,
    model_jacobian,
)

__all__ = [
    # peak_model
    "ModelPeak",
    "baseband_offset",
    "effective_tau",
    "h_T",
    "h_T_jacobian",
    "model_spectrum",
    "molecular_frequency",
    "sideband_sign",
    "to_baseband_frame",
    # validation
    "calculate_aic",
    "calculate_chi_squared_improvement",
    "calculate_hwhm_from_apodization",
    "calculate_noise_weighted_chi2",
    "calculate_rms_residuals",
    "feature_fwhm",
    "passes_significance_test",
    "validate_peak_separation",
    # window_fit
    "AddStep",
    "ConservativeFitResult",
    "KnockoutResult",
    "ParameterErrors",
    "WindowFitResult",
    "conservative_fit",
    "fit_window",
    "knockout_test",
    "model_jacobian",
    # plan_execution
    "DEFAULT_MAX_THAW_ROUNDS",
    "DEFAULT_RESIDUAL_EDGE_M",
    "DEFAULT_RESIDUAL_EDGE_THRESHOLD",
    "FrozenPeak",
    "PlanFitOutcome",
    "ThawEvent",
    "WindowOutcome",
    "attempt_thaw_round",
    "evaluate_fixed_contributor",
    "execute_plan",
    "fit_window_with_fixed_contributors",
    "local_thaw_cofit",
    "residual_edge_coherence",
    "select_contributor_to_thaw",
    "subtract_frozen_background",
]
