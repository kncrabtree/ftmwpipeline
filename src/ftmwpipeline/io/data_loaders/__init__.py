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
from .registry import (
    FormatRegistry, register_loader, detect_format, validate_source, 
    load_fid, list_formats, get_format_info
)
from .blackchirp import BlackChirpLoader
from .csv import CSVLoader
from .hdf5 import HDF5Loader

# Register available loaders
register_loader('blackchirp', BlackChirpLoader)
register_loader('csv', CSVLoader)
register_loader('hdf5', HDF5Loader)

# Export main interface
__all__ = [
    'BaseLoader',
    'LoaderError', 
    'FormatRegistry',
    'register_loader',
    'detect_format',
    'validate_source',
    'load_fid',
    'list_formats',
    'get_format_info',
    'BlackChirpLoader',
    'CSVLoader',
    'HDF5Loader'
]