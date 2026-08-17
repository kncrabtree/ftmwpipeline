"""
FTMW Pipeline - A Python package for FTMW spectroscopy signal processing and peak fitting.

This package provides tools for processing Fourier Transform Microwave (FTMW)
spectroscopy data, including baseline estimation, peak detection, window assignment,
and advanced fitting algorithms.
"""

__version__ = "0.1.0b6"
__author__ = "Kyle N. Crabtree"

# Note: BLAS/OpenMP thread pinning for the Stage 5 fork pool is deliberately NOT
# done here (a process-wide env var would throttle a caller's own BLAS work in
# the same interpreter). It is scoped to the fit instead: ``fit_peaks_impl``
# wraps the whole Stage 5 call in ``threadpoolctl.threadpool_limits(1)`` in the
# *parent* process, so OpenBLAS is at one thread when the window pool forks and
# the children inherit that (no oversubscription, and no fork-unsafe threadpoolctl
# call in the children); the limit is restored when the fit returns. See
# ``_internal/stage5_impl.py`` and ``fitting/plan_execution.py``.

# Functional API - can be imported as "import ftmwpipeline.api as ftmw"
from ftmwpipeline import api

# Public curation tolerances (the values the Stage 6 verbs themselves pair at)
from ftmwpipeline.core.curation import REFIT_SNAP_TOL_MHZ

# Core data structures
from ftmwpipeline.core.data_structures import (
    FID,
    ComplexFT,
    FIDProcessingParameters,
    FittedPeak,
    FittingResult,
    FitWindow,
    FixedContributor,
    FTMWData,
    Peak,
    PeakClassification,
    Sideband,
    SpectralWindow,
    WindowPlan,
)

# Exception family - typed errors callers are expected to catch and route on
from ftmwpipeline.file_manager import (
    AnalysisEpochMismatchError,
    PipelineCompatibilityError,
    PipelineCorruptionError,
    PipelineExistsError,
    PipelineFileError,
    StageDependencyError,
)

# Main pipeline interface
from ftmwpipeline.pipeline import Pipeline

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Core data structures
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
    "FixedContributor",
    "FitWindow",
    "WindowPlan",
    # Public curation tolerances
    "REFIT_SNAP_TOL_MHZ",
    # Exception family
    "PipelineFileError",
    "PipelineExistsError",
    "StageDependencyError",
    "PipelineCorruptionError",
    "PipelineCompatibilityError",
    "AnalysisEpochMismatchError",
    # Main pipeline interface
    "Pipeline",
    # Functional API module
    "api",
]

# Package-level configuration
import logging

# Configure default logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Optional imports with graceful fallbacks
try:
    import matplotlib

    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

# Package metadata
PACKAGE_INFO = {
    "name": "ftmwpipeline",
    "version": __version__,
    "description": "FTMW spectroscopy signal processing and peak fitting",
    "has_matplotlib": _HAS_MATPLOTLIB,
}
