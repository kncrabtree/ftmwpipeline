"""
Data loaders package for multi-format FTMW data ingestion.

This package implements an extensible loader architecture that can handle various
experimental data formats (BlackChirp, CSV, HDF5, etc.) and produces standardized
FID objects with proper metadata preservation.

Architecture:
- BaseLoader: Abstract interface for all loaders
- FormatRegistry: Format detection and loader selection
- Specific loaders: BlackChirp, CSV, HDF5, etc.
"""

from .base import BaseLoader, LoaderError
from .blackchirp import BlackChirpLoader
from .csv import CSVLoader
from .hdf5 import HDF5Loader
from .keysight_mat import KeysightMatLoader
from .registry import (
    FormatRegistry,
    detect_format,
    get_format_info,
    list_formats,
    load_fid,
    register_loader,
    validate_source,
)

# Register loaders in precedence order.
#
# ``keysight-mat`` must be registered BEFORE ``hdf5`` because MATLAB v7.3
# files are HDF5 containers: both loaders can h5py-open a .mat file, but
# the generic HDF5Loader.can_load gates on ``fid_data``/``voltage_data``
# groups that Keysight files do not have, so there is no actual ambiguity.
# Registering keysight-mat first is belt-and-suspenders insurance against
# any future relaxation of HDF5Loader.can_load.
register_loader("blackchirp", BlackChirpLoader)
register_loader("csv", CSVLoader)
register_loader("keysight-mat", KeysightMatLoader)
register_loader("hdf5", HDF5Loader)

# Export main interface
__all__ = [
    "BaseLoader",
    "LoaderError",
    "FormatRegistry",
    "register_loader",
    "detect_format",
    "validate_source",
    "load_fid",
    "list_formats",
    "get_format_info",
    "BlackChirpLoader",
    "CSVLoader",
    "HDF5Loader",
    "KeysightMatLoader",
]
