"""
Visualization and plotting functions for FTMW pipeline.

This module provides:
- Pipeline stage visualization
- Fit diagnostics and residual analysis
- Summary reports and batch processing visualization
"""

from .spectrum_visualization import (
    plot_complex_ft, plot_spectral_window, plot_spectrum, plot_peaks, plot_windows,
    plot_complex_ft_from_cache, plot_spectrum_from_cache
)
from .fit_diagnostics import plot_fit_results, plot_residuals, plot_time_domain_fit
from .summary_reports import generate_fit_report, create_batch_summary
from .noise_visualization import plot_noise_estimation, plot_noise_estimation_from_cache

__all__ = [
    # Spectrum visualization (direct and cache-based)
    "plot_complex_ft",
    "plot_complex_ft_from_cache",
    "plot_spectral_window",
    "plot_spectrum",
    "plot_spectrum_from_cache", 
    "plot_peaks", 
    "plot_windows",
    # Noise visualization (direct and cache-based)
    "plot_noise_estimation",
    "plot_noise_estimation_from_cache",
    # Fitting visualization
    "plot_fit_results",
    "plot_residuals",
    "plot_time_domain_fit",
    # Summary reports
    "generate_fit_report",
    "create_batch_summary",
]