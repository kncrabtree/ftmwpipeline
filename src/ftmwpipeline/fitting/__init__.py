"""
Peak fitting algorithms for FTMW spectroscopy.

This module provides:
- The finite-acquisition line-shape model (``peak_model``)
- The active-portion FT used as the Stage 5 fit frame (``active_ft``)
- The per-window least-squares core and conservative add-one-peak loop
  (``window_fit``)
- Statistical-test and linewidth-physics helpers (``validation``)
- Plan-level execution: fixed-contributor evaluation, DAG/batch walk, and
  local thaw renegotiation (``plan_execution``)
"""

from .active_ft import (
    ActiveFTResult,
    compute_active_ft,
)
from .peak_model import (
    ModelPeak,
    baseband_offset,
    effective_tau,
    h_T,
    h_T_jacobian,
    model_spectrum,
    molecular_frequency,
    sideband_sign,
    to_baseband_offset,
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
    materialize_window,
    residual_edge_coherence,
    select_contributor_to_thaw,
    subtract_frozen_background,
)
from .result_conversion import (
    plan_fit_outcome_to_spectrum_fit,
    window_outcome_to_fitting_result,
    window_outcome_to_spectral_window,
)
from .residual_rescue import (
    DEFAULT_RESCUE_MAX_ROUNDS,
    DEFAULT_RESCUE_PROMINENCE_THRESHOLD,
    DEFAULT_RESCUE_SNR_THRESHOLD,
    RescueOutcome,
    attempt_residual_rescue,
)
from .residual_screening import (
    DEFAULT_COHERENCE_CLUSTER_FWHM,
    DEFAULT_COHERENCE_RATIO_THRESHOLD,
    ResidualPeakCandidate,
    filter_by_phase_coherence,
    find_residual_peaks,
)
from .window_fit import (
    AddStep,
    ConservativeFitResult,
    KnockoutResult,
    ParameterErrors,
    WindowFitConstraints,
    WindowFitResult,
    conservative_fit,
    derive_window_fit_constraints,
    fit_window,
    knockout_test,
    model_jacobian,
)

__all__ = [
    # active_ft
    "ActiveFTResult",
    "compute_active_ft",
    # peak_model
    "ModelPeak",
    "baseband_offset",
    "effective_tau",
    "h_T",
    "h_T_jacobian",
    "model_spectrum",
    "molecular_frequency",
    "sideband_sign",
    "to_baseband_offset",
    # validation
    "calculate_aic",
    "calculate_chi_squared_improvement",
    "calculate_hwhm_from_apodization",
    "calculate_noise_weighted_chi2",
    "calculate_rms_residuals",
    "feature_fwhm",
    "passes_significance_test",
    "validate_peak_separation",
    # residual_rescue
    "DEFAULT_RESCUE_MAX_ROUNDS",
    "DEFAULT_RESCUE_PROMINENCE_THRESHOLD",
    "DEFAULT_RESCUE_SNR_THRESHOLD",
    "RescueOutcome",
    "attempt_residual_rescue",
    # residual_screening
    "DEFAULT_COHERENCE_CLUSTER_FWHM",
    "DEFAULT_COHERENCE_RATIO_THRESHOLD",
    "ResidualPeakCandidate",
    "filter_by_phase_coherence",
    "find_residual_peaks",
    # window_fit
    "AddStep",
    "ConservativeFitResult",
    "KnockoutResult",
    "ParameterErrors",
    "WindowFitConstraints",
    "WindowFitResult",
    "conservative_fit",
    "derive_window_fit_constraints",
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
    "materialize_window",
    "residual_edge_coherence",
    "select_contributor_to_thaw",
    "subtract_frozen_background",
    # result_conversion
    "plan_fit_outcome_to_spectrum_fit",
    "window_outcome_to_fitting_result",
    "window_outcome_to_spectral_window",
]
