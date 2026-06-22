"""
Utility functions for FTMW pipeline.

This module provides signal-processing utilities (apodization windows and the
matched filter).
"""

from .signal_processing import (
    APODIZATION_EXAMPLES,
    apodize_fid,
    make_apodization,
    matched_filter_window,
)

__all__ = [
    "APODIZATION_EXAMPLES",
    "apodize_fid",
    "make_apodization",
    "matched_filter_window",
]
