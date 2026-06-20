"""
Data loaders package for multi-format FTMW data ingestion.

This package implements an extensible loader architecture that can handle various
experimental data formats (Blackchirp, CSV, HDF5, etc.) and produces standardized
FID objects with proper metadata preservation.

Architecture:
- BaseLoader: Abstract interface for all loaders
- FormatRegistry: Format detection and loader selection
- Specific loaders: Blackchirp, CSV, HDF5, etc.
"""

from .base import BaseLoader, LoaderError
from .blackchirp import BlackChirpLoader
from .csv import CSVLoader
from .ftmw_hdf5 import FtmwHdf5Loader
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
# ``keysight-mat`` is registered BEFORE ``ftmw-hdf5`` because MATLAB v7.3 files
# are HDF5 containers: a Keysight ``.mat`` and a native ``.h5`` are both
# h5py-openable, but the two loaders gate on disjoint signatures
# (``Channel_*/XInc`` vs a root ``ftmw_input_version`` attribute), so there is
# no real ambiguity.  Registering keysight-mat first is insurance against any
# future relaxation of those signatures.
register_loader("blackchirp", BlackChirpLoader)
register_loader("csv", CSVLoader)
register_loader("keysight-mat", KeysightMatLoader)
register_loader("ftmw-hdf5", FtmwHdf5Loader)

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
    "FtmwHdf5Loader",
    "KeysightMatLoader",
]
