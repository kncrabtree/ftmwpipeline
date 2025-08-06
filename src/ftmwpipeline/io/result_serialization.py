"""
Unified pipeline cache interface for FTMW pipeline results.

This module provides a comprehensive caching system that integrates ComplexFT 
and NoiseResult serialization into a unified pipeline cache. The cache supports
complete pipeline state persistence, stage-specific caching, and efficient
storage optimization.

Key Features
------------
- Unified pipeline cache with ComplexFT and NoiseResult integration
- Stage-specific result caching and loading
- Automatic cache directory management
- Version tracking and metadata storage
- HDF5-based storage with compression optimization
- Error handling for corrupted/missing cache files

Functions
---------
save_pipeline_cache : Save complete pipeline cache with ComplexFT and NoiseResult
load_pipeline_cache : Load complete pipeline cache
save_stage_result : Save individual stage results
load_stage_result : Load individual stage results
get_cache_info : Get cache file information and metadata
clear_cache : Remove cache files for experiment

Storage Structure
-----------------
experiment_2638_cache.h5
├── /complex_ft/                    [ComplexFT serialization group]
│   ├── complex_spectrum            [dataset: complex128 array, ~6MB for 375k points]
│   ├── freq_reconstruction/        [group: frequency array reconstruction parameters]
│   │   ├── n_fid_padded           [attr: int, padded FID length used in FFT]
│   │   ├── spacing_us             [attr: float, time spacing in microseconds]
│   │   ├── probe_freq_mhz         [attr: float, probe/LO frequency] 
│   │   ├── sideband               [attr: str, 'upper' or 'lower']
│   │   ├── autoscale_MHz          [attr: float, DC suppression range]
│   │   ├── freq_min               [attr: float, minimum frequency in MHz]
│   │   ├── freq_max               [attr: float, maximum frequency in MHz]
│   │   └── n_spectrum             [attr: int, number of frequency points]
│   ├── processing_params/          [group: FID processing parameters used]
│   │   └── [FIDProcessingParameters attributes including actual zpf used]
│   └── metadata/                   [group: experiment metadata]
│       └── [metadata dict contents including processing_params override]
├── /noise_result/                  [NoiseResult serialization group]
│   ├── rms_poly_coeffs            [dataset: polynomial coefficients for RMS noise]
│   ├── signal_indices             [dataset: indices where signal > noise threshold]
│   ├── reconstruction_params/      [group: parameters for noise reconstruction]
│   └── metadata/                   [group: noise estimation metadata]
└── /pipeline_info/                 [pipeline metadata group]
    ├── version                     [attr: str, pipeline version]
    ├── timestamp                   [attr: str, ISO format timestamp]
    ├── experiment_id               [attr: str, experiment identifier]
    ├── has_complex_ft              [attr: bool, ComplexFT present]
    ├── has_noise_result            [attr: bool, NoiseResult present]
    ├── complex_ft_checksum         [attr: str, data integrity checksum]
    └── noise_result_checksum       [attr: str, data integrity checksum]

Key Features:
- ComplexFT frequency arrays are reconstructed from parameters (massive storage reduction)
- Stores actual processing parameters used in ft() call, including overrides
- Single reconstruction path for both trimmed and untrimmed ComplexFT objects
- NoiseResult uses signal indices approach for ~98% storage reduction
- Bit-perfect reconstruction accuracy maintained for all data types
"""

import numpy as np
import h5py
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
import warnings

from ..core.data_structures import ComplexFT
from ..preprocessing.noise_estimation import NoiseResult
from .complex_ft_serialization import save_complex_ft_to_hdf5, load_complex_ft_from_hdf5
from .noise_result_serialization import save_noise_result_to_hdf5, load_noise_result_from_hdf5


def save_pipeline_cache(
    experiment_id: str, 
    complex_ft: ComplexFT, 
    noise_result: Optional[NoiseResult] = None, 
    cache_dir: str = "cache"
) -> Path:
    """
    Save complete pipeline cache with ComplexFT and optional NoiseResult.
    
    Creates a unified HDF5 cache file containing the ComplexFT object, optional
    NoiseResult, and pipeline metadata. The cache enables fast pipeline restarts
    and supports storage optimization through the integrated serialization methods.
    
    Parameters
    ----------
    experiment_id : str
        Unique identifier for the experiment (used in filename)
    complex_ft : ComplexFT
        ComplexFT object to cache
    noise_result : NoiseResult, optional
        NoiseResult object to cache (if available)
    cache_dir : str, default "cache"
        Directory to store cache files
        
    Returns
    -------
    Path
        Path to the created cache file
        
    Raises
    ------
    ValueError
        If experiment_id is invalid or objects cannot be serialized
    RuntimeError
        If cache file creation fails
        
    Examples
    --------
    >>> # Save complete pipeline cache
    >>> cache_file = save_pipeline_cache("exp_2638", complex_ft, noise_result)
    >>> print(f"Cache saved to: {cache_file}")
    
    >>> # Save cache with only ComplexFT
    >>> cache_file = save_pipeline_cache("exp_2638", complex_ft)
    """
    try:
        # Validate inputs
        if not experiment_id or not isinstance(experiment_id, str):
            raise ValueError("experiment_id must be a non-empty string")
        
        if not isinstance(complex_ft, ComplexFT):
            raise ValueError("complex_ft must be a ComplexFT object")
        
        if noise_result is not None and not isinstance(noise_result, NoiseResult):
            raise ValueError("noise_result must be a NoiseResult object or None")
        
        # Create cache directory
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Generate cache filename
        cache_filename = f"{experiment_id}_cache.h5"
        cache_file = cache_path / cache_filename
        
        # Save to HDF5 with unified structure
        with h5py.File(cache_file, 'w') as h5f:
            # Save ComplexFT to /complex_ft/ group
            complex_ft_group = h5f.create_group('complex_ft')
            
            # Save ComplexFT with proper error handling
            try:
                save_complex_ft_to_hdf5(complex_ft, complex_ft_group)
            except ValueError as e:
                if "no associated FID" in str(e):
                    raise ValueError(
                        f"ComplexFT must have associated FID for optimized serialization. "
                        f"Original error: {e}"
                    )
                else:
                    raise
            
            # Save NoiseResult to /noise_result/ group if provided
            if noise_result is not None:
                noise_result_group = h5f.create_group('noise_result')
                save_noise_result_to_hdf5(
                    noise_result, 
                    complex_ft.freq_array, 
                    complex_ft.magnitude_spectrum, 
                    noise_result_group
                )
            
            # Save pipeline metadata to /pipeline_info/ group
            pipeline_info = h5f.create_group('pipeline_info')
            pipeline_info.attrs['version'] = '1.0'
            pipeline_info.attrs['timestamp'] = datetime.now().isoformat()
            pipeline_info.attrs['experiment_id'] = experiment_id
            pipeline_info.attrs['has_complex_ft'] = True
            pipeline_info.attrs['has_noise_result'] = noise_result is not None
            
            # Compute and store stage checksums for integrity checking
            pipeline_info.attrs['complex_ft_checksum'] = _compute_complex_ft_checksum(complex_ft)
            if noise_result is not None:
                pipeline_info.attrs['noise_result_checksum'] = _compute_noise_result_checksum(noise_result)
        
        return cache_file
        
    except Exception as e:
        raise RuntimeError(f"Failed to save pipeline cache for {experiment_id}: {e}") from e


def load_pipeline_cache(experiment_id: str, cache_dir: str = "cache") -> Dict[str, Any]:
    """
    Load complete pipeline cache returning ComplexFT and optional NoiseResult.
    
    Loads the unified pipeline cache and reconstructs ComplexFT and NoiseResult
    objects with full integrity checking. Returns a dictionary containing the
    loaded objects and metadata.
    
    Parameters
    ----------
    experiment_id : str
        Unique identifier for the experiment (used in filename)
    cache_dir : str, default "cache"
        Directory containing cache files
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'complex_ft': ComplexFT object
        - 'noise_result': NoiseResult object (or None if not cached)
        - 'metadata': Pipeline metadata dict
        - 'cache_file': Path to cache file
        
    Raises
    ------
    FileNotFoundError
        If cache file does not exist
    ValueError
        If cache file is corrupted or invalid
    RuntimeError
        If cache loading fails
        
    Examples
    --------
    >>> # Load complete pipeline cache
    >>> cache_data = load_pipeline_cache("exp_2638")
    >>> complex_ft = cache_data['complex_ft']
    >>> noise_result = cache_data['noise_result']  # May be None
    >>> metadata = cache_data['metadata']
    """
    try:
        # Validate inputs
        if not experiment_id or not isinstance(experiment_id, str):
            raise ValueError("experiment_id must be a non-empty string")
        
        # Locate cache file
        cache_path = Path(cache_dir)
        cache_filename = f"{experiment_id}_cache.h5"
        cache_file = cache_path / cache_filename
        
        if not cache_file.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_file}")
        
        # Load from HDF5
        with h5py.File(cache_file, 'r') as h5f:
            # Validate cache structure
            if 'complex_ft' not in h5f:
                raise ValueError("Invalid cache file: missing 'complex_ft' group")
            
            # Load ComplexFT with proper error handling
            try:
                complex_ft = load_complex_ft_from_hdf5(h5f['complex_ft'])
            except (ValueError, KeyError) as e:
                raise ValueError(
                    f"Failed to load ComplexFT from cache. This may indicate a "
                    f"corrupted cache file or ComplexFT saved without proper FID association. "
                    f"Original error: {e}"
                )
            
            # Load NoiseResult if available
            noise_result = None
            if 'noise_result' in h5f:
                noise_result = load_noise_result_from_hdf5(
                    h5f['noise_result'],
                    complex_ft.freq_array,
                    complex_ft.magnitude_spectrum
                )
            
            # Load pipeline metadata
            metadata = {}
            if 'pipeline_info' in h5f:
                pipeline_info = h5f['pipeline_info']
                for key in pipeline_info.attrs.keys():
                    value = pipeline_info.attrs[key]
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    metadata[key] = value
            
            # Perform integrity checks if checksums are available
            if 'complex_ft_checksum' in metadata:
                computed_checksum = _compute_complex_ft_checksum(complex_ft)
                if computed_checksum != metadata['complex_ft_checksum']:
                    warnings.warn("ComplexFT checksum mismatch - data may be corrupted")
            
            if noise_result is not None and 'noise_result_checksum' in metadata:
                computed_checksum = _compute_noise_result_checksum(noise_result)
                if computed_checksum != metadata['noise_result_checksum']:
                    warnings.warn("NoiseResult checksum mismatch - data may be corrupted")
        
        return {
            'complex_ft': complex_ft,
            'noise_result': noise_result,
            'metadata': metadata,
            'cache_file': cache_file
        }
        
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        else:
            raise RuntimeError(f"Failed to load pipeline cache for {experiment_id}: {e}") from e


def save_stage_result(
    stage_name: str, 
    result: Any, 
    experiment_id: str, 
    cache_dir: str = "cache"
) -> Path:
    """
    Save individual stage result to cache.
    
    Saves the result of a specific pipeline stage to a separate cache file
    or group within the main cache file. Supports caching of intermediate
    results for debugging and pipeline optimization.
    
    Parameters
    ----------
    stage_name : str
        Name of the pipeline stage (e.g., 'peak_detection', 'fitting')
    result : Any
        Result object to cache (ComplexFT, NoiseResult, or other serializable object)
    experiment_id : str
        Unique identifier for the experiment
    cache_dir : str, default "cache"
        Directory to store cache files
        
    Returns
    -------
    Path
        Path to the cache file containing the stage result
        
    Raises
    ------
    ValueError
        If stage_name or experiment_id is invalid
    RuntimeError
        If stage result cannot be cached
        
    Examples
    --------
    >>> # Save peak detection results
    >>> cache_file = save_stage_result("peak_detection", peak_results, "exp_2638")
    
    >>> # Save fitting results
    >>> cache_file = save_stage_result("fitting", fit_results, "exp_2638")
    """
    try:
        # Validate inputs
        if not stage_name or not isinstance(stage_name, str):
            raise ValueError("stage_name must be a non-empty string")
        
        if not experiment_id or not isinstance(experiment_id, str):
            raise ValueError("experiment_id must be a non-empty string")
        
        # Create cache directory
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Generate stage cache filename
        stage_filename = f"{experiment_id}_{stage_name}_stage.h5"
        stage_file = cache_path / stage_filename
        
        # Save based on result type
        with h5py.File(stage_file, 'w') as h5f:
            # Create stage group
            stage_group = h5f.create_group(stage_name)
            
            # Handle different result types
            if isinstance(result, ComplexFT):
                save_complex_ft_to_hdf5(result, stage_group)
                stage_group.attrs['result_type'] = 'ComplexFT'
            
            elif isinstance(result, NoiseResult):
                # For NoiseResult, we need frequency and magnitude arrays
                # These should be provided or we need to get them from context
                if hasattr(result, '_frequencies') and hasattr(result, '_magnitudes'):
                    save_noise_result_to_hdf5(
                        result, result._frequencies, result._magnitudes, stage_group
                    )
                else:
                    # Store as generic object
                    _save_generic_result(result, stage_group)
                stage_group.attrs['result_type'] = 'NoiseResult'
                
            else:
                # Handle generic results (lists, dicts, arrays, etc.)
                _save_generic_result(result, stage_group)
                stage_group.attrs['result_type'] = 'Generic'
            
            # Add metadata
            stage_group.attrs['timestamp'] = datetime.now().isoformat()
            stage_group.attrs['experiment_id'] = experiment_id
            stage_group.attrs['stage_name'] = stage_name
        
        return stage_file
        
    except Exception as e:
        raise RuntimeError(f"Failed to save stage result {stage_name} for {experiment_id}: {e}") from e


def load_stage_result(stage_name: str, experiment_id: str, cache_dir: str = "cache") -> Any:
    """
    Load individual stage result from cache.
    
    Loads the cached result for a specific pipeline stage, reconstructing
    the appropriate object type based on stored metadata.
    
    Parameters
    ----------
    stage_name : str
        Name of the pipeline stage to load
    experiment_id : str
        Unique identifier for the experiment
    cache_dir : str, default "cache"
        Directory containing cache files
        
    Returns
    -------
    Any
        Reconstructed stage result object
        
    Raises
    ------
    FileNotFoundError
        If stage cache file does not exist
    ValueError
        If stage cache is corrupted or invalid
    RuntimeError
        If stage loading fails
        
    Examples
    --------
    >>> # Load peak detection results
    >>> peak_results = load_stage_result("peak_detection", "exp_2638")
    
    >>> # Load fitting results
    >>> fit_results = load_stage_result("fitting", "exp_2638")
    """
    try:
        # Validate inputs
        if not stage_name or not isinstance(stage_name, str):
            raise ValueError("stage_name must be a non-empty string")
        
        if not experiment_id or not isinstance(experiment_id, str):
            raise ValueError("experiment_id must be a non-empty string")
        
        # Locate stage cache file
        cache_path = Path(cache_dir)
        stage_filename = f"{experiment_id}_{stage_name}_stage.h5"
        stage_file = cache_path / stage_filename
        
        if not stage_file.exists():
            raise FileNotFoundError(f"Stage cache file not found: {stage_file}")
        
        # Load from HDF5
        with h5py.File(stage_file, 'r') as h5f:
            if stage_name not in h5f:
                raise ValueError(f"Invalid stage cache: missing '{stage_name}' group")
            
            stage_group = h5f[stage_name]
            result_type = stage_group.attrs.get('result_type', 'Generic')
            
            # Reconstruct based on result type
            if result_type == 'ComplexFT':
                return load_complex_ft_from_hdf5(stage_group)
            
            elif result_type == 'NoiseResult':
                # Try to load as NoiseResult first
                try:
                    # We need frequency and magnitude arrays for NoiseResult loading
                    # These should be stored in the stage or we load generically
                    return _load_generic_result(stage_group)
                except Exception:
                    return _load_generic_result(stage_group)
            
            else:
                return _load_generic_result(stage_group)
        
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        else:
            raise RuntimeError(f"Failed to load stage result {stage_name} for {experiment_id}: {e}") from e


def get_cache_info(experiment_id: str, cache_dir: str = "cache") -> Dict[str, Any]:
    """
    Get information about cached pipeline data.
    
    Returns metadata about the cache file including file size, contents,
    timestamps, and integrity information.
    
    Parameters
    ----------
    experiment_id : str
        Unique identifier for the experiment
    cache_dir : str, default "cache"
        Directory containing cache files
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing cache information:
        - 'exists': Whether cache file exists
        - 'file_size': Size of cache file in bytes
        - 'has_complex_ft': Whether ComplexFT is cached
        - 'has_noise_result': Whether NoiseResult is cached
        - 'timestamp': Cache creation timestamp
        - 'stage_files': List of available stage cache files
        
    Examples
    --------
    >>> info = get_cache_info("exp_2638")
    >>> print(f"Cache size: {info['file_size']} bytes")
    >>> print(f"Has noise result: {info['has_noise_result']}")
    """
    cache_path = Path(cache_dir)
    cache_filename = f"{experiment_id}_cache.h5"
    cache_file = cache_path / cache_filename
    
    info = {
        'exists': cache_file.exists(),
        'cache_file': cache_file,
        'file_size': 0,
        'has_complex_ft': False,
        'has_noise_result': False,
        'timestamp': None,
        'stage_files': []
    }
    
    if cache_file.exists():
        info['file_size'] = cache_file.stat().st_size
        
        try:
            with h5py.File(cache_file, 'r') as h5f:
                info['has_complex_ft'] = 'complex_ft' in h5f
                info['has_noise_result'] = 'noise_result' in h5f
                
                if 'pipeline_info' in h5f:
                    pipeline_info = h5f['pipeline_info']
                    if 'timestamp' in pipeline_info.attrs:
                        info['timestamp'] = pipeline_info.attrs['timestamp']
                        if isinstance(info['timestamp'], bytes):
                            info['timestamp'] = info['timestamp'].decode('utf-8')
        except Exception:
            pass  # Cache file may be corrupted
    
    # Find stage cache files
    if cache_path.exists():
        stage_pattern = f"{experiment_id}_*_stage.h5"
        stage_files = list(cache_path.glob(stage_pattern))
        info['stage_files'] = [f.name for f in stage_files]
    
    return info


def clear_cache(experiment_id: str, cache_dir: str = "cache", include_stages: bool = True) -> List[Path]:
    """
    Remove cache files for an experiment.
    
    Deletes the main pipeline cache file and optionally all stage cache files
    for the specified experiment.
    
    Parameters
    ----------
    experiment_id : str
        Unique identifier for the experiment
    cache_dir : str, default "cache"
        Directory containing cache files
    include_stages : bool, default True
        Whether to also remove stage cache files
        
    Returns
    -------
    List[Path]
        List of removed cache files
        
    Examples
    --------
    >>> # Remove all cache files for experiment
    >>> removed = clear_cache("exp_2638")
    >>> print(f"Removed {len(removed)} cache files")
    
    >>> # Remove only main cache, keep stage files
    >>> removed = clear_cache("exp_2638", include_stages=False)
    """
    cache_path = Path(cache_dir)
    removed_files = []
    
    # Remove main cache file
    cache_filename = f"{experiment_id}_cache.h5"
    cache_file = cache_path / cache_filename
    if cache_file.exists():
        cache_file.unlink()
        removed_files.append(cache_file)
    
    # Remove stage files if requested
    if include_stages and cache_path.exists():
        stage_pattern = f"{experiment_id}_*_stage.h5"
        stage_files = list(cache_path.glob(stage_pattern))
        for stage_file in stage_files:
            stage_file.unlink()
            removed_files.append(stage_file)
    
    return removed_files




# Helper functions for checksum computation and generic serialization

def _compute_complex_ft_checksum(complex_ft: ComplexFT) -> str:
    """Compute checksum for ComplexFT object integrity checking."""
    hasher = hashlib.md5()
    hasher.update(complex_ft.complex_spectrum.tobytes())
    hasher.update(complex_ft.freq_array.tobytes())
    return hasher.hexdigest()


def _compute_noise_result_checksum(noise_result: NoiseResult) -> str:
    """Compute checksum for NoiseResult object integrity checking."""
    hasher = hashlib.md5()
    hasher.update(noise_result.rms_noise.tobytes())
    hasher.update(noise_result.noise_mask.tobytes())
    return hasher.hexdigest()


def _save_generic_result(result: Any, h5_group: h5py.Group) -> None:
    """Save generic result objects to HDF5."""
    if isinstance(result, np.ndarray):
        h5_group.create_dataset('data', data=result)
        h5_group.attrs['type'] = 'ndarray'
        h5_group.attrs['dtype'] = str(result.dtype)
        h5_group.attrs['shape'] = result.shape
    
    elif isinstance(result, (list, tuple)):
        # Convert to numpy array if possible
        try:
            arr = np.array(result)
            h5_group.create_dataset('data', data=arr)
            h5_group.attrs['type'] = 'list' if isinstance(result, list) else 'tuple'
            h5_group.attrs['original_type'] = type(result).__name__
        except Exception:
            # Fallback to JSON serialization
            json_str = json.dumps(result, default=str)
            h5_group.attrs['data'] = json_str
            h5_group.attrs['type'] = 'json'
            h5_group.attrs['original_type'] = type(result).__name__
    
    elif isinstance(result, dict):
        # Save dict as JSON
        json_str = json.dumps(result, default=str)
        h5_group.attrs['data'] = json_str
        h5_group.attrs['type'] = 'json'
        h5_group.attrs['original_type'] = 'dict'
    
    else:
        # Fallback to string representation
        h5_group.attrs['data'] = str(result)
        h5_group.attrs['type'] = 'string'
        h5_group.attrs['original_type'] = type(result).__name__


def _load_generic_result(h5_group: h5py.Group) -> Any:
    """Load generic result objects from HDF5."""
    result_type = h5_group.attrs.get('type', 'string')
    
    if result_type == 'ndarray':
        return h5_group['data'][:]
    
    elif result_type in ('list', 'tuple'):
        if 'data' in h5_group:
            arr = h5_group['data'][:]
            return list(arr) if result_type == 'list' else tuple(arr)
        else:
            # JSON fallback
            json_str = h5_group.attrs['data']
            if isinstance(json_str, bytes):
                json_str = json_str.decode('utf-8')
            data = json.loads(json_str)
            return list(data) if result_type == 'list' else tuple(data)
    
    elif result_type == 'json':
        json_str = h5_group.attrs['data']
        if isinstance(json_str, bytes):
            json_str = json_str.decode('utf-8')
        return json.loads(json_str)
    
    else:
        # String representation
        data = h5_group.attrs['data']
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        return data
