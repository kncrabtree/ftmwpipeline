"""
Input/output and logging functions for FTMW pipeline.

This module handles:
- Experimental data format readers
- Result serialization (HDF5, JSON, CSV)
- Structured logging and decision tracking
"""

from .experimental_formats import (
    load_blackchirp_experiment,
    load_blackchirp_fid,
    load_generic_fid,
)
from .fid_serialization import (
    load_fid_cache,
    save_fid_cache,
    update_fid_processing_defaults,
)
from .logging import FittingLogger, setup_logging
from .noise_result_serialization import (
    load_noise_result_from_hdf5,
    save_noise_result_to_hdf5,
)

__all__ = [
    "load_blackchirp_experiment",
    "load_blackchirp_fid",
    "load_generic_fid",
    # Logging
    "setup_logging",
    "FittingLogger",
    # Individual serialization functions
    "save_noise_result_to_hdf5",
    "load_noise_result_from_hdf5",
    # FID caching (Stage 0)
    "save_fid_cache",
    "load_fid_cache",
    "update_fid_processing_defaults",
]
