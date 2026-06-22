"""
Input/output and logging functions for FTMW pipeline.

This module handles:
- Result serialization (HDF5, JSON, CSV)
- Structured logging and decision tracking

Input data formats are loaded through the pluggable registry in
:mod:`ftmwpipeline.io.data_loaders`.
"""

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
