"""
Analysis window assignment algorithms.

This module handles:
- Greedy window assignment for spectral peaks
- Physics-based window sizing (FWHM calculations)
- Window boundary optimization and overlap resolution
"""

from .greedy_assignment import assign_analysis_windows
from .window_optimization import optimize_window_boundaries, resolve_overlaps

__all__ = [
    "assign_analysis_windows",
    "optimize_window_boundaries",
    "resolve_overlaps",
]
