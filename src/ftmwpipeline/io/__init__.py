"""
Input/output and logging functions for FTMW pipeline.

This module handles:
- Experimental data format readers
- Result serialization (HDF5, JSON, CSV)
- Structured logging and decision tracking
"""

from .experimental_formats import load_blackchirp_experiment, load_blackchirp_fid, load_generic_fid
from .result_serialization import (
    save_pipeline_cache, 
    load_pipeline_cache,
    save_stage_result,
    load_stage_result,
    get_cache_info,
    clear_cache
)
from .logging import setup_logging, FittingLogger
from .complex_ft_serialization import save_complex_ft_to_hdf5, load_complex_ft_from_hdf5
from .noise_result_serialization import save_noise_result_to_hdf5, load_noise_result_from_hdf5

__all__ = [
    "load_blackchirp_experiment",
    "load_blackchirp_fid", 
    "load_generic_fid",
    # Unified pipeline cache interface
    "save_pipeline_cache",
    "load_pipeline_cache", 
    "save_stage_result",
    "load_stage_result",
    "get_cache_info",
    "clear_cache",
    # Logging
    "setup_logging",
    "FittingLogger",
    # Individual serialization functions
    "save_complex_ft_to_hdf5",
    "load_complex_ft_from_hdf5",
    "save_noise_result_to_hdf5",
    "load_noise_result_from_hdf5",
]