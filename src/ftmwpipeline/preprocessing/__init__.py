"""
Data preprocessing and preparation for FTMW pipeline.

This module handles:
- Baseline and noise estimation
- Peak detection and leakage analysis
"""

from .leakage import estimate_leakage_reach
from .noise_estimation import NoiseResult, estimate_noise_scatter
from .peak_detection import (
    PeakResult,
    classify_by_snr,
    detect_peaks,
    locate_peaks,
)

__all__ = [
    "estimate_noise_scatter",
    "NoiseResult",
    "locate_peaks",
    "PeakResult",
    "classify_by_snr",
    "detect_peaks",
    "estimate_leakage_reach",
]
