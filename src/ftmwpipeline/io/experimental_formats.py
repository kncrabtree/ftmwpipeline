"""
Experimental data format readers.

This module contains data format readers for various experimental systems.
Currently implements BlackChirp data loading.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Union

from ..core.data_structures import FTMWData, FID, FIDProcessingParameters, Sideband


def load_blackchirp_experiment(experiment_path: Union[str, Path], 
                              fid_index: int = 0) -> FTMWData:
    """
    Load a complete BlackChirp experiment into FTMWData structure.
    
    Parameters
    ----------
    experiment_path : str or Path
        Path to BlackChirp experiment directory
    fid_index : int, optional
        Index of FID to load (default: 0)
        
    Returns
    -------
    FTMWData
        Loaded experiment data
    """
    experiment_path = Path(experiment_path)
    
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment path does not exist: {experiment_path}")
    
    # Load FID data
    fid = load_blackchirp_fid(experiment_path, fid_index)
    
    # Load experiment metadata
    metadata = _load_blackchirp_metadata(experiment_path)
    
    # Create experiment ID from path
    experiment_id = experiment_path.name
    
    return FTMWData(
        fid=fid,
        experiment_id=experiment_id,
        metadata=metadata
    )


def load_blackchirp_fid(experiment_path: Union[str, Path], 
                       fid_index: int = 0) -> FID:
    """
    Load BlackChirp FID data into FID object.
    
    Parameters
    ----------
    experiment_path : str or Path
        Path to BlackChirp experiment directory
    fid_index : int, optional
        Index of FID to load (default: 0)
        
    Returns
    -------
    FID
        Loaded FID data with processing parameters
    """
    experiment_path = Path(experiment_path)
    fid_dir = experiment_path / "fid"
    
    if not fid_dir.exists():
        raise FileNotFoundError(f"FID directory not found: {fid_dir}")
    
    # Load FID parameters
    fidparams_file = fid_dir / "fidparams.csv"
    if not fidparams_file.exists():
        raise FileNotFoundError(f"FID parameters file not found: {fidparams_file}")
    
    fidparams_df = pd.read_csv(fidparams_file, sep=';')
    
    if fid_index >= len(fidparams_df):
        raise ValueError(f"FID index {fid_index} not found. Available indices: 0-{len(fidparams_df)-1}")
    
    fid_params = fidparams_df.iloc[fid_index]
    
    # Load FID data
    fid_file = fid_dir / f"{fid_index}.csv"
    if not fid_file.exists():
        raise FileNotFoundError(f"FID data file not found: {fid_file}")
    
    # Read FID data - BlackChirp stores as base-36 integers
    fid_df = pd.read_csv(fid_file, header=0, dtype=str, keep_default_na=False)
    
    # Convert from base-36 to integers, then to voltage
    raw_data = np.array([int(val, 36) for val in fid_df.iloc[:, 0]])
    
    # Convert to voltage using parameters
    voltage_data = raw_data * fid_params['vmult'] / fid_params['shots']
    
    # Load processing parameters
    processing_params = _load_blackchirp_processing(fid_dir)
    
    # Convert sideband string to enum
    sideband_str = fid_params['sideband'].lower()
    if 'lower' in sideband_str:
        sideband = Sideband.LOWER
    else:
        sideband = Sideband.UPPER
    
    return FID(
        data=voltage_data,
        spacing=float(fid_params['spacing']),  # seconds
        probe_freq_mhz=float(fid_params['probefreq']),  # MHz
        sideband=sideband,
        shots=int(fid_params['shots']),
        processing=processing_params,
        metadata={
            'experiment_path': str(experiment_path),
            'fid_index': fid_index,
            'blackchirp_params': fid_params.to_dict()
        }
    )


def _load_blackchirp_processing(fid_dir: Path) -> FIDProcessingParameters:
    """Load BlackChirp processing parameters."""
    processing_file = fid_dir / "processing.csv"
    
    if not processing_file.exists():
        # Return default parameters if file not found
        return FIDProcessingParameters()
    
    # Load processing parameters
    proc_df = pd.read_csv(processing_file, sep=';', index_col='ObjKey')
    proc_dict = proc_df['Value'].to_dict()
    
    # Convert to our parameter structure
    end_us_val = float(proc_dict.get('FidEndUs', 0))
    expf_us_val = float(proc_dict.get('FidExpfUs', 0))
    autoscale_val = float(proc_dict.get('AutoscaleIgnoreMHz', 0))
    
    return FIDProcessingParameters(
        start_us=float(proc_dict.get('FidStartUs', 0)),
        end_us=end_us_val if end_us_val > 0 else None,
        winf=_convert_window_function(proc_dict.get('FidWindowFunction', 'None')),
        zpf=int(proc_dict.get('FidZeroPadFactor', 0)),
        rdc=proc_dict.get('FidRemoveDC', 'false').lower() == 'true',
        expf_us=expf_us_val if expf_us_val > 0 else None,
        units_power=int(proc_dict.get('FtUnits', 6))
    )


def _convert_window_function(blackchirp_winf: str) -> Optional[str]:
    """Convert BlackChirp window function name to scipy compatible name."""
    winf_map = {
        'None': None,
        'Bartlett': 'bartlett',
        'Blackman': 'blackman', 
        'BlackmanHarris': 'blackmanharris',
        'Hamming': 'hamming',
        'Hanning': 'hann',
        'KaiserBessel': ('kaiser', 14.0)
    }
    
    return winf_map.get(blackchirp_winf, None)


def _load_blackchirp_metadata(experiment_path: Path) -> Dict[str, Any]:
    """Load BlackChirp experiment metadata from various files."""
    metadata = {}
    
    # Load header information if available
    header_file = experiment_path / "header.csv"
    if header_file.exists():
        try:
            header_df = pd.read_csv(header_file, sep=';')
            metadata['header'] = header_df.to_dict('records')[0] if len(header_df) > 0 else {}
        except Exception:
            pass
    
    # Load hardware configuration
    hardware_file = experiment_path / "hardware.csv" 
    if hardware_file.exists():
        try:
            hardware_df = pd.read_csv(hardware_file, sep=';')
            metadata['hardware'] = hardware_df.to_dict('records')
        except Exception:
            pass
    
    # Load version information
    version_file = experiment_path / "version.csv"
    if version_file.exists():
        try:
            with open(version_file, 'r') as f:
                metadata['version'] = f.read().strip()
        except Exception:
            pass
    
    return metadata


def load_generic_fid(*args, **kwargs):
    """Placeholder for generic FID loading."""
    raise NotImplementedError("Generic FID loading will be implemented later")
