"""
Core data structures and classes for FTMW pipeline.

This module contains the fundamental data structures used throughout the pipeline:
- FTMWData: Top-level container for complete FTMW experiment
- FID: Time domain data with processing parameters
- ComplexFT: Frequency domain data
- SpectralWindow: Analysis window (subset of ComplexFT)
- Peak: Pre-fitting detected peak representation
- FittedPeak: Post-fitting peak results
- FittingResult: Container for fitting results
- FIDProcessingParameters: FID processing configuration
"""

from .data_structures import (
    FTMWData,
    FID,
    ComplexFT,
    SpectralWindow,
    Peak,
    FittedPeak,
    FittingResult,
    FIDProcessingParameters,
    PeakClassification,
    Sideband,
    WindowDifficulty,
    FixedContributor,
    FitWindow,
    WindowPlan,
)

__all__ = [
    "FTMWData",
    "FID",
    "ComplexFT",
    "SpectralWindow",
    "Peak",
    "FittedPeak",
    "FittingResult",
    "FIDProcessingParameters",
    "PeakClassification",
    "Sideband",
    "WindowDifficulty",
    "FixedContributor",
    "FitWindow",
    "WindowPlan",
]