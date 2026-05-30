"""
BlackChirp data format loader.

This module implements the loader for BlackChirp experimental data format,
handling FID data extraction from BlackChirp directory structures with
proper metadata preservation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union, List, Optional, TYPE_CHECKING

from .base import BaseLoader, LoaderError

if TYPE_CHECKING:
    from ...core.data_structures import FID, FIDProcessingParameters, Sideband
else:
    # Runtime imports to avoid circular dependencies
    def _get_fid_classes():
        from ...core.data_structures import FID, FIDProcessingParameters, Sideband
        return FID, FIDProcessingParameters, Sideband


class BlackChirpLoader(BaseLoader):
    """
    Loader for BlackChirp experimental data format.
    
    BlackChirp stores FTMW experiments in directory structures containing
    FID data, processing parameters, and experimental metadata.
    """
    
    format_name = "blackchirp"
    file_extensions = []  # BlackChirp uses directories, not files
    directory_indicators = ["fid", "fidparams.csv"]  # Files that indicate BlackChirp format
    
    def can_load(self, source_path: Union[str, Path]) -> bool:
        """
        Check if source is a BlackChirp experiment directory.
        
        BlackChirp experiments have the following structure:
        experiment_dir/
        ├── fid/
        │   ├── fidparams.csv
        │   ├── processing.csv (optional)
        │   ├── 0.csv
        │   ├── 1.csv (if multiple FIDs)
        │   └── ...
        """
        source_path = Path(source_path)
        
        if not source_path.exists():
            return False
        
        if not source_path.is_dir():
            return False
        
        # Check for FID directory
        fid_dir = source_path / "fid"
        if not fid_dir.exists() or not fid_dir.is_dir():
            return False
        
        # Check for fidparams.csv
        fidparams_file = fid_dir / "fidparams.csv"
        if not fidparams_file.exists():
            return False
        
        # Check for at least one FID data file
        fid_files = list(fid_dir.glob("*.csv"))
        fid_data_files = [f for f in fid_files if f.name.replace('.csv', '').isdigit()]
        
        return len(fid_data_files) > 0
    
    def validate_source(self, source_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        """
        Validate BlackChirp experiment directory.
        
        Returns information about available FIDs, processing parameters,
        and experimental metadata.
        """
        result = {
            'valid': False,
            'metadata': {},
            'options': {},
            'errors': []
        }
        
        source_path = Path(source_path)
        
        try:
            # Check basic structure
            if not self.can_load(source_path):
                result['errors'].append("Not a valid BlackChirp experiment directory")
                return result
            
            fid_dir = source_path / "fid"
            
            # Load and validate FID parameters
            fidparams_file = fid_dir / "fidparams.csv"
            try:
                fidparams_df = pd.read_csv(fidparams_file, sep=';')
                result['metadata']['n_fids'] = len(fidparams_df)
                result['options']['available_fid_indices'] = list(range(len(fidparams_df)))
                
                # Extract key parameters from first FID
                if len(fidparams_df) > 0:
                    first_fid = fidparams_df.iloc[0]
                    result['metadata']['probe_freq_mhz'] = float(first_fid['probefreq'])
                    result['metadata']['spacing_us'] = float(first_fid['spacing']) * 1e6  # Convert to μs
                    result['metadata']['sideband'] = first_fid['sideband']
                    result['metadata']['shots'] = int(first_fid['shots'])
                    
            except Exception as e:
                result['errors'].append(f"Failed to read FID parameters: {e}")
                return result
            
            # Check FID data files
            missing_files = []
            for i in range(len(fidparams_df)):
                fid_file = fid_dir / f"{i}.csv"
                if not fid_file.exists():
                    missing_files.append(str(fid_file))
            
            if missing_files:
                result['errors'].append(f"Missing FID data files: {missing_files}")
                return result
            
            # Load processing parameters if available
            processing_file = fid_dir / "processing.csv"
            if processing_file.exists():
                try:
                    proc_df = pd.read_csv(processing_file, sep=';', index_col='ObjKey')
                    proc_dict = proc_df['Value'].to_dict()
                    result['metadata']['processing_parameters'] = proc_dict
                except Exception as e:
                    result['errors'].append(f"Warning: Could not read processing parameters: {e}")
            
            # Load experiment metadata
            metadata_files = {
                'header.csv': 'header',
                'hardware.csv': 'hardware', 
                'version.csv': 'version'
            }
            
            for filename, key in metadata_files.items():
                metadata_file = source_path / filename
                if metadata_file.exists():
                    try:
                        if key == 'version':
                            with open(metadata_file, 'r') as f:
                                result['metadata'][key] = f.read().strip()
                        else:
                            df = pd.read_csv(metadata_file, sep=';')
                            if key == 'header':
                                result['metadata'][key] = df.to_dict('records')[0] if len(df) > 0 else {}
                            else:
                                result['metadata'][key] = df.to_dict('records')
                    except Exception as e:
                        result['errors'].append(f"Warning: Could not read {filename}: {e}")
            
            result['valid'] = True
            return result
            
        except Exception as e:
            result['errors'].append(f"Validation failed: {e}")
            return result
    
    def load_fid(self, source_path: Union[str, Path], 
                 fid_index: int = 0, **kwargs) -> 'FID':
        """
        Load FID data from BlackChirp experiment.
        
        Parameters
        ----------
        source_path : str or Path
            Path to BlackChirp experiment directory
        fid_index : int, optional
            Index of FID to load (default: 0)
        **kwargs
            Additional parameters (ignored for BlackChirp)
            
        Returns
        -------
        FID
            Loaded FID object with BlackChirp metadata
            
        Raises
        ------
        LoaderError
            If loading fails
        """
        try:
            # Get runtime imports
            FID, FIDProcessingParameters, Sideband = _get_fid_classes()
            
            source_path = Path(source_path)
            fid_dir = source_path / "fid"
            
            # Validate source first
            validation = self.validate_source(source_path)
            if not validation['valid']:
                raise LoaderError(f"Invalid BlackChirp source: {validation['errors']}")
            
            # Load FID parameters
            fidparams_file = fid_dir / "fidparams.csv"
            fidparams_df = pd.read_csv(fidparams_file, sep=';')
            
            if fid_index >= len(fidparams_df):
                raise LoaderError(f"FID index {fid_index} not found. Available indices: 0-{len(fidparams_df)-1}")
            
            fid_params = fidparams_df.iloc[fid_index]
            
            # Load FID data
            fid_file = fid_dir / f"{fid_index}.csv"
            if not fid_file.exists():
                raise LoaderError(f"FID data file not found: {fid_file}")
            
            # Read FID data - BlackChirp stores as base-36 integers
            try:
                fid_df = pd.read_csv(fid_file, header=0, dtype=str, keep_default_na=False)
                raw_data = np.array([int(val, 36) for val in fid_df.iloc[:, 0]])
            except Exception as e:
                raise LoaderError(f"Failed to read FID data from {fid_file}: {e}")
            
            # Convert to voltage using parameters
            voltage_data = raw_data * fid_params['vmult'] / fid_params['shots']
            
            # Load processing parameters
            processing_params = self._load_processing_parameters(fid_dir)
            
            # Convert sideband cell to enum (version-tolerant; see helper)
            sideband = self._resolve_sideband(fid_params['sideband'])
            
            # Create source metadata
            source_metadata = self._create_source_metadata(
                source_path, 
                fid_index=fid_index,
                blackchirp_params=fid_params.to_dict()
            )
            
            # Add validation metadata  
            source_metadata.update(validation['metadata'])
            
            return FID(
                data=voltage_data,
                spacing=float(fid_params['spacing']),  # seconds
                probe_freq_mhz=float(fid_params['probefreq']),  # MHz
                sideband=sideband,
                shots=int(fid_params['shots']),
                processing=processing_params,
                metadata=source_metadata
            )
            
        except LoaderError:
            raise
        except Exception as e:
            raise LoaderError(f"Failed to load BlackChirp FID: {e}") from e
    
    def get_required_parameters(self) -> List[str]:
        """BlackChirp loader has no required parameters."""
        return []
    
    def get_optional_parameters(self) -> Dict[str, Any]:
        """Get optional parameters for BlackChirp loading."""
        return {
            'fid_index': 0  # Which FID to load if multiple are available
        }
    
    @staticmethod
    def _resolve_sideband(value: Any):
        """Resolve a BlackChirp ``sideband`` cell to a :class:`Sideband`.

        The on-disk encoding varies across BlackChirp versions: newer files
        store the canonical Q_ENUM name (``"LowerSideband"`` /
        ``"UpperSideband"``), older ones the underlying enum integer
        (``1`` = lower, ``0`` = upper, matching the ``blackchirp`` module's
        ``_SIDEBAND_INT_MAP``). Both forms -- and numeric-string variants --
        resolve here so either generation of fixture imports cleanly.
        """
        _, _, Sideband = _get_fid_classes()

        # Integer enum code (int, numpy integer/float, or numeric string).
        if not isinstance(value, str):
            return Sideband.LOWER if int(value) == 1 else Sideband.UPPER
        s = value.strip()
        try:
            return Sideband.LOWER if int(s) == 1 else Sideband.UPPER
        except ValueError:
            pass

        low = s.lower()
        if "lower" in low:
            return Sideband.LOWER
        if "upper" in low:
            return Sideband.UPPER
        raise LoaderError(f"Unrecognised BlackChirp sideband value: {value!r}")

    def _load_processing_parameters(self, fid_dir: Path):
        """Load BlackChirp processing parameters."""
        # Get runtime imports
        FID, FIDProcessingParameters, Sideband = _get_fid_classes()
        
        processing_file = fid_dir / "processing.csv"
        
        if not processing_file.exists():
            # Return default parameters if file not found
            return FIDProcessingParameters()
        
        try:
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
                winf=self._convert_window_function(proc_dict.get('FidWindowFunction', 'None')),
                zpf=int(proc_dict.get('FidZeroPadFactor', 0)),
                rdc=proc_dict.get('FidRemoveDC', 'false').lower() == 'true',
                expf_us=expf_us_val if expf_us_val > 0 else None,
                units_power=int(proc_dict.get('FtUnits', 6))
            )
            
        except Exception as e:
            raise LoaderError(f"Failed to load processing parameters: {e}")
    
    def _convert_window_function(self, blackchirp_winf: str) -> Optional[str]:
        """Convert BlackChirp window function name to scipy compatible name."""
        winf_map = {
            'None': None,
            'Bartlett': 'bartlett',
            'Blackman': 'blackman', 
            'BlackmanHarris': 'blackmanharris',
            'Hamming': 'hamming',
            'Hanning': 'hann',
            'KaiserBessel': ('kaiser', 14.0)  # Note: scipy needs parameters for Kaiser
        }
        
        return winf_map.get(blackchirp_winf, None)