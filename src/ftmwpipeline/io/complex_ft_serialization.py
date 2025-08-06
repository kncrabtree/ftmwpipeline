"""
ComplexFT serialization to HDF5 format.

This module provides efficient serialization and deserialization of ComplexFT objects
to HDF5 format. The key optimization is frequency array reconstruction using parameters
instead of storing the full frequency array, reducing storage from ~3MB to ~48 bytes
(untrimmed) or ~64 bytes (trimmed objects) while maintaining bit-perfect reconstruction accuracy.

Functions
---------
save_complex_ft_to_hdf5 : Save ComplexFT to HDF5 group
load_complex_ft_from_hdf5 : Load ComplexFT from HDF5 group
_extract_frequency_reconstruction_params : Extract parameters for frequency reconstruction
_reconstruct_frequency_array : Reconstruct frequency array from parameters
"""

import numpy as np
import scipy.fft as sfft
import h5py
from typing import Dict, Any, Optional
import warnings

from ..core.data_structures import ComplexFT, FID, Sideband, FIDProcessingParameters


def save_complex_ft_to_hdf5(complex_ft: ComplexFT, h5_group: h5py.Group) -> None:
    """
    Save ComplexFT object to HDF5 group with optimized frequency storage.
    
    Instead of storing the full frequency array (~3MB), this function extracts
    reconstruction parameters (~48 bytes) that allow bit-perfect reconstruction
    of the frequency array using scipy.fft.rfftfreq.
    
    HDF5 Structure
    --------------
    /complex_ft/
    ├── complex_spectrum        [dataset: complex128 array, ~6MB for 375k points]
    ├── freq_reconstruction/    [group with frequency reconstruction parameters]
    │   ├── n_fid_padded       [attr: int] 
    │   ├── spacing_us         [attr: float]
    │   ├── probe_freq_mhz     [attr: float]
    │   ├── sideband           [attr: str]
    │   ├── autoscale_MHz      [attr: float or None]
    │   ├── n_spectrum         [attr: int]
    │   ├── is_trimmed         [attr: bool] (optional, trimmed objects only)
    │   ├── trim_freq_min      [attr: float] (optional, trimmed objects only)
    │   └── trim_freq_max      [attr: float] (optional, trimmed objects only)
    ├── processing_params/      [group with FID processing parameters]
    │   └── [FIDProcessingParameters attributes]
    └── metadata/              [group with experiment metadata]
        └── [metadata dict contents]
    
    Parameters
    ----------
    complex_ft : ComplexFT
        ComplexFT object to serialize
    h5_group : h5py.Group
        HDF5 group to write data to
        
    Raises
    ------
    ValueError
        If frequency reconstruction parameters cannot be extracted
    RuntimeError
        If HDF5 write operation fails
    """
    try:
        # Store complex spectrum data (~6MB for typical 375k points)
        h5_group.create_dataset(
            'complex_spectrum', 
            data=complex_ft.complex_spectrum,
            compression='gzip',
            compression_opts=6,
            shuffle=True
        )
        
        # Extract and store frequency reconstruction parameters (~48 bytes)
        freq_params = _extract_frequency_reconstruction_params(complex_ft)
        freq_group = h5_group.create_group('freq_reconstruction')
        
        for key, value in freq_params.items():
            if value is not None:
                freq_group.attrs[key] = value
            else:
                # Handle None values explicitly
                freq_group.attrs[key] = "None"
        
        # Store FID processing parameters if available
        if complex_ft.fid is not None:
            proc_group = h5_group.create_group('processing_params')
            proc = complex_ft.fid.processing
            
            # Store all FIDProcessingParameters attributes
            for attr_name in ['start_us', 'end_us', 'winf', 'zpf', 'rdc', 
                             'expf_us', 'autoscale_MHz', 'units_power']:
                value = getattr(proc, attr_name)
                if value is not None:
                    proc_group.attrs[attr_name] = value
                else:
                    proc_group.attrs[attr_name] = "None"
        
        # Store metadata
        if complex_ft.metadata:
            meta_group = h5_group.create_group('metadata')
            for key, value in complex_ft.metadata.items():
                if key == 'processing_params':
                    # Skip processing_params as it's stored separately
                    continue
                    
                try:
                    if isinstance(value, (int, float, str, bool)):
                        meta_group.attrs[key] = value
                    elif key == 'trimmed_range' and isinstance(value, (tuple, list)) and len(value) == 2:
                        # Handle trimmed_range tuple specially
                        meta_group.attrs[f'{key}_min'] = float(value[0])
                        meta_group.attrs[f'{key}_max'] = float(value[1])
                    else:
                        # Convert complex objects to string representation
                        meta_group.attrs[key] = str(value)
                except Exception as e:
                    warnings.warn(f"Could not serialize metadata key '{key}': {e}")
        
    except Exception as e:
        raise RuntimeError(f"Failed to save ComplexFT to HDF5: {e}") from e


def load_complex_ft_from_hdf5(h5_group: h5py.Group) -> ComplexFT:
    """
    Load ComplexFT object from HDF5 group with frequency array reconstruction.
    
    Reconstructs the frequency array from stored parameters using scipy.fft.rfftfreq
    to achieve bit-perfect accuracy while using minimal storage space.
    
    Parameters
    ----------
    h5_group : h5py.Group
        HDF5 group containing serialized ComplexFT data
        
    Returns
    -------
    ComplexFT
        Reconstructed ComplexFT object
        
    Raises
    ------
    KeyError
        If required data is missing from HDF5 group
    ValueError
        If frequency reconstruction fails or data is invalid
    RuntimeError
        If HDF5 read operation fails
    """
    try:
        # Load complex spectrum
        complex_spectrum = h5_group['complex_spectrum'][:]
        
        # Reconstruct frequency array from parameters
        freq_group = h5_group['freq_reconstruction']
        freq_params = {}
        
        # All parameters 
        for key in ['n_fid_padded', 'spacing_us', 'probe_freq_mhz', 
                   'sideband', 'autoscale_MHz', 'n_spectrum', 'freq_min', 'freq_max']:
            value = freq_group.attrs[key]
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            if value == "None":
                value = None
            freq_params[key] = value
        
        
        freq_array = _reconstruct_frequency_array(freq_params)
        
        # Validate reconstruction
        if len(freq_array) != len(complex_spectrum):
            raise ValueError(
                f"Frequency array length mismatch: expected {len(complex_spectrum)}, "
                f"got {len(freq_array)}"
            )
        
        # Reconstruct FID if processing parameters are available
        fid = None
        if 'processing_params' in h5_group:
            proc_group = h5_group['processing_params']
            proc_params = {}
            
            for attr_name in ['start_us', 'end_us', 'winf', 'zpf', 'rdc', 
                             'expf_us', 'autoscale_MHz', 'units_power']:
                if attr_name in proc_group.attrs:
                    value = proc_group.attrs[attr_name]
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    if value == "None":
                        value = None
                    proc_params[attr_name] = value
            
            # Create FIDProcessingParameters object
            processing = FIDProcessingParameters(**proc_params)
            
            # Create minimal FID object (note: actual FID data is not stored)
            # This preserves the processing parameters for reference
            fid = FID(
                data=np.array([0.0]),  # Placeholder data
                spacing=freq_params['spacing_us'] * 1e-6,  # Convert back to seconds
                probe_freq_mhz=freq_params['probe_freq_mhz'],
                sideband=freq_params['sideband'],
                processing=processing
            )
        
        # Load metadata
        metadata = {}
        if 'metadata' in h5_group:
            meta_group = h5_group['metadata']
            
            # Handle regular metadata
            for key in meta_group.attrs.keys():
                if key.endswith('_min') or key.endswith('_max'):
                    continue  # Handle these separately
                    
                value = meta_group.attrs[key]
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                metadata[key] = value
            
            # Reconstruct trimmed_range tuple if present
            if 'trimmed_range_min' in meta_group.attrs and 'trimmed_range_max' in meta_group.attrs:
                freq_min = float(meta_group.attrs['trimmed_range_min'])
                freq_max = float(meta_group.attrs['trimmed_range_max'])
                metadata['trimmed_range'] = (freq_min, freq_max)
        
        # Create and return ComplexFT object
        return ComplexFT(
            freq_array=freq_array,
            complex_spectrum=complex_spectrum,
            fid=fid,
            metadata=metadata
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to load ComplexFT from HDF5: {e}") from e


def _extract_frequency_reconstruction_params(complex_ft: ComplexFT) -> Dict[str, Any]:
    """
    Extract parameters needed to reconstruct the frequency array.
    
    This function stores the original FID parameters needed to reconstruct the full
    frequency array, plus the actual frequency range of this ComplexFT object.
    Upon deserialization, the full frequency array is reconstructed and then 
    filtered to the stored range.
    
    Parameters
    ----------
    complex_ft : ComplexFT
        ComplexFT object to extract parameters from
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing reconstruction parameters
        
    Raises
    ------
    ValueError
        If required parameters cannot be extracted from ComplexFT or its FID
    """
    if complex_ft.fid is None:
        raise ValueError(
            "Cannot extract frequency reconstruction parameters: "
            "ComplexFT has no associated FID object"
        )
    
    fid = complex_ft.fid
    
    # Store the actual frequency range of this ComplexFT object
    freq_min = float(np.min(complex_ft.freq_array))
    freq_max = float(np.max(complex_ft.freq_array))
    n_spectrum = len(complex_ft.freq_array)
    
    # Get the actual processing parameters that were used for this ComplexFT
    # These are stored in metadata and include any overrides from ft() call
    if 'processing_params' in complex_ft.metadata:
        actual_proc = complex_ft.metadata['processing_params']
        zpf = actual_proc.zpf if actual_proc.zpf is not None else 0
    else:
        # Fallback to FID's processing parameters  
        zpf = fid.processing.zpf if fid.processing.zpf is not None else 0
    
    # Calculate the padded FID length that was actually used in the FFT
    original_fid_length = len(fid.data)
    if zpf > 0:
        n_fid_padded = 2 ** (int(np.log2(original_fid_length)) + 1 + zpf)
    else:
        n_fid_padded = original_fid_length
    
    # Store parameters needed for reconstruction
    params = {
        'n_fid_padded': n_fid_padded,
        'spacing_us': fid.spacing * 1e6,  # Convert seconds to microseconds
        'probe_freq_mhz': fid.probe_freq_mhz,
        'sideband': fid.sideband.value,
        'autoscale_MHz': fid.processing.autoscale_MHz,
        'freq_min': freq_min,
        'freq_max': freq_max,
        'n_spectrum': n_spectrum
    }
    
    return params


def _reconstruct_frequency_array(params: Dict[str, Any]) -> np.ndarray:
    """
    Reconstruct frequency array from extracted parameters.
    
    Reconstructs the full frequency array from the stored padded FID length, then
    filters to the stored frequency range. This single approach works for both
    trimmed and untrimmed ComplexFT objects.
    
    Parameters
    ----------
    params : Dict[str, Any]
        Dictionary containing reconstruction parameters from _extract_frequency_reconstruction_params
        
    Returns
    -------
    np.ndarray
        Reconstructed frequency array in MHz
        
    Raises
    ------
    ValueError
        If reconstruction parameters are invalid or inconsistent
    KeyError
        If required parameters are missing
    """
    try:
        # Extract parameters
        n_fid_padded = int(params['n_fid_padded'])
        spacing_us = float(params['spacing_us'])
        probe_freq_mhz = float(params['probe_freq_mhz'])
        sideband_str = str(params['sideband'])
        freq_min = float(params['freq_min'])
        freq_max = float(params['freq_max'])
        n_spectrum = int(params['n_spectrum'])
        
        # Generate full scope frequencies using rfftfreq
        full_scope_freqs = sfft.rfftfreq(n_fid_padded, d=spacing_us * 1e-6) / 1e6  # MHz
        
        # Apply sideband conversion to get full molecular frequencies
        sideband = Sideband(sideband_str)
        if sideband in (Sideband.LOWER, Sideband.LSB):
            full_mol_freqs = probe_freq_mhz - full_scope_freqs
        else:
            full_mol_freqs = probe_freq_mhz + full_scope_freqs
        
        # Filter to the stored frequency range
        mask = (full_mol_freqs >= freq_min) & (full_mol_freqs <= freq_max)
        filtered_freqs = full_mol_freqs[mask]
        
        # Validate reconstructed array length
        if len(filtered_freqs) != n_spectrum:
            raise ValueError(
                f"Frequency array length mismatch: expected {n_spectrum}, got {len(filtered_freqs)}. "
                f"Range: {freq_min:.3f}-{freq_max:.3f} MHz"
            )
        
        return filtered_freqs
        
    except Exception as e:
        raise ValueError(f"Failed to reconstruct frequency array: {e}") from e