"""
Peak detection algorithms for FTMW spectroscopy.

This module provides:
- Basic second derivative-based peak detection
- Hybrid clustering and iterative subtraction methods
- SNR-based peak classification
"""

from .basic_detection import locate_peaks
from .hybrid_detection import locate_peaks_hybrid
from .classification import classify_peaks, find_and_classify_peaks

__all__ = [
    "locate_peaks",
    "locate_peaks_hybrid",
    "classify_peaks",
    "find_and_classify_peaks",
]