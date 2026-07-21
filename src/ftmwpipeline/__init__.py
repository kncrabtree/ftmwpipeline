"""
FTMW Pipeline - A Python package for FTMW spectroscopy signal processing and peak fitting.

This package provides tools for processing Fourier Transform Microwave (FTMW)
spectroscopy data, including baseline estimation, peak detection, window assignment,
and advanced fitting algorithms.
"""

__version__ = "0.1.0b2"
__author__ = "Kyle N. Crabtree"

# --- BLAS / OpenMP thread pinning (MUST precede the numpy/scipy imports below) -
# Stage 5 fits fan windows across a ``fork()`` process pool: parallelism here is
# process-level, so every process runs single-threaded BLAS. The only fork-safe
# way to pin the thread count is via the environment *at import time*, before the
# native BLAS/OpenMP runtimes initialize. This (a) stops the parent from ever
# spawning a multi-threaded OpenBLAS pool -- whose lingering worker threads make
# a later ``fork()`` unsafe (locked malloc/linker mutexes -> SIGABRT in the
# child) -- and (b) removes N-workers x M-BLAS-threads oversubscription.
#
# Pinning at fit time is too late (the runtime is already initialized), and
# pinning inside a forked worker via ``threadpoolctl.threadpool_limits`` aborts:
# its ``dlopen`` library scan is not fork-safe in a fork child of a
# multithreaded parent. ``setdefault`` so an explicit user override still wins.
import os as _os

for _v in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    _os.environ.setdefault(_v, "1")
del _os, _v

# Functional API - can be imported as "import ftmwpipeline.api as ftmw"
from ftmwpipeline import api

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
