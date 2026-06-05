"""
Data preprocessing and preparation for FTMW pipeline.

This module handles:
- Loading experimental data (BlackChirp format)
- Baseline and noise estimation
- Data validation and quality checks
"""

from .noise_estimation import estimate_noise_scatter, NoiseResult
from .peak_detection import (
    locate_peaks,
    PeakResult,
    classify_by_snr,
    detect_peaks,
)
from .leakage import estimate_leakage_reach
from .data_loading import load_blackchirp_data, load_fid_data
from .data_validation import validate_fid_data, validate_frequency_data

__all__ = [
    "estimate_noise_scatter",
    "NoiseResult",
    "locate_peaks",
    "PeakResult",
    "classify_by_snr",
    "detect_peaks",
    "estimate_leakage_reach",
    "load_blackchirp_data",
    "load_fid_data",
    "validate_fid_data",
    "validate_frequency_data",
]
