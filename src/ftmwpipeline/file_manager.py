"""
FTMW Pipeline File Manager - Foundation for dual-interface architecture.

This module provides the foundational file management capabilities for .ftmw 
pipeline data files. It handles file creation, opening, validation, source 
metadata tracking, and stage dependency checking.

Key Features:
- Safe creation vs. opening semantics with smart re-import detection
- Source metadata tracking for reproducibility and conflict prevention
- Stage dependency validation with clear error messages
- File integrity checking and corruption recovery guidance
- Jupyter-safe patterns for interactive development

This file manager serves as the foundation for both the Pipeline class and 
functional API, enabling consistent file operations across interfaces.

Architecture Integration:
- Used by Pipeline.create() and Pipeline.open() methods
- Used by functional API functions (import_data, compute_ft, etc.)
- Leverages existing serialization infrastructure in io/fid_serialization.py
- Provides error handling patterns used throughout the dual-interface system
"""

import hashlib
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import h5py

from .core.data_structures import FID
from .io.fid_serialization import save_fid_to_hdf5, load_fid_from_hdf5


# Custom exceptions for clear error handling
class PipelineFileError(Exception):
    """Base exception for pipeline file operations."""
    pass


class PipelineExistsError(PipelineFileError):
    """Raised when attempting to create a pipeline file that already exists with different source."""
    
    def __init__(self, filepath: Path, existing_source: str, requested_source: str):
        self.filepath = filepath
        self.existing_source = existing_source
        self.requested_source = requested_source
        
        super().__init__(
            f"Pipeline file {filepath} already exists with different source.\n"
            f"  Existing source: {existing_source}\n"
            f"  Requested source: {requested_source}\n\n"
            f"Options:\n"
            f"  1. Use Pipeline.open('{filepath}') to work with existing data\n"
            f"  2. Use Pipeline.create('{filepath}', source='{requested_source}', force=True) to overwrite\n"
            f"  3. Choose a different filename for the new analysis"
        )


class PipelineStageError(PipelineFileError):
    """Raised when attempting operations that require incomplete pipeline stages."""
    
    def __init__(self, stage_name: str, missing_dependencies: list, filepath: Path):
        self.stage_name = stage_name
        self.missing_dependencies = missing_dependencies
        self.filepath = filepath
        
        deps_str = ", ".join(missing_dependencies)
        super().__init__(
            f"Cannot perform {stage_name} - missing required dependencies: {deps_str}\n"
            f"Pipeline file: {filepath}\n\n"
            f"Complete the missing stages first:\n" +
            "\n".join([f"  - {dep}" for dep in missing_dependencies])
        )


class PipelineCorruptedError(PipelineFileError):
    """Raised when a pipeline file appears to be corrupted."""
    
    def __init__(self, filepath: Path, corruption_details: str):
        self.filepath = filepath
        self.corruption_details = corruption_details
        
        super().__init__(
            f"Pipeline file appears to be corrupted: {filepath}\n"
            f"Details: {corruption_details}\n\n"
            f"Recovery options:\n"
            f"  1. Restore from backup if available\n"
            f"  2. Re-create from original source data\n"
            f"  3. Contact support if data is critical"
        )


class SourceMetadata:
    """Source metadata for tracking data provenance and preventing accidental re-imports."""
    
    def __init__(self, source_path: Union[str, Path], format_name: str, 
                 loader_parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize source metadata.
        
        Parameters
        ----------
        source_path : str or Path
            Path to the source data
        format_name : str
            Name of the data format (e.g., 'blackchirp', 'csv', 'hdf5')
        loader_parameters : dict, optional
            Parameters used by the data loader
        """
        self.source_path = Path(source_path).resolve()
        self.format_name = format_name
        self.loader_parameters = loader_parameters or {}
        self.import_timestamp = datetime.now()
        
        # Compute source hash for quick comparison
        self.source_hash = self._compute_source_hash()
        
        # Get file modification time if source is a file
        if self.source_path.is_file():
            self.source_mtime = self.source_path.stat().st_mtime
        elif self.source_path.is_dir():
            # For directories, use the most recent modification time
            self.source_mtime = max(
                (f.stat().st_mtime for f in self.source_path.rglob("*") if f.is_file()),
                default=0
            )
        else:
            self.source_mtime = 0
    
    def _compute_source_hash(self) -> str:
        """Compute a hash of key source properties for quick comparison."""
        hash_input = {
            'source_path': str(self.source_path),
            'format_name': self.format_name,
            'loader_parameters': self.loader_parameters,
            'source_mtime': getattr(self, 'source_mtime', 0)
        }
        
        # Create a stable JSON representation
        json_str = json.dumps(hash_input, sort_keys=True, default=str)
        return hashlib.md5(json_str.encode('utf-8')).hexdigest()
    
    def matches(self, other: 'SourceMetadata') -> bool:
        """Check if this source metadata matches another."""
        return (
            self.source_path == other.source_path and
            self.format_name == other.format_name and
            self.loader_parameters == other.loader_parameters and
            abs(self.source_mtime - other.source_mtime) < 1.0  # 1 second tolerance
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'source_path': str(self.source_path),
            'format_name': self.format_name,
            'loader_parameters': self.loader_parameters,
            'import_timestamp': self.import_timestamp.isoformat(),
            'source_mtime': self.source_mtime,
            'source_hash': self.source_hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SourceMetadata':
        """Create from dictionary (deserialization)."""
        metadata = cls(
            source_path=data['source_path'],
            format_name=data['format_name'],
            loader_parameters=data.get('loader_parameters', {})
        )
        
        # Override computed values with stored ones
        metadata.import_timestamp = datetime.fromisoformat(data['import_timestamp'])
        metadata.source_mtime = data['source_mtime']
        metadata.source_hash = data['source_hash']
        
        return metadata


class PipelineStageTracker:
    """Tracks completion status of pipeline stages and validates dependencies."""
    
    # Define stage dependencies
    STAGE_DEPENDENCIES = {
        'fid_import': [],                    # Stage 0: No dependencies
        'ft_processing': ['fid_import'],     # Stage 1: Requires FID data
        'noise_estimation': ['ft_processing'], # Stage 2: Requires ComplexFT
        'peak_detection': ['noise_estimation'], # Stage 3: Requires noise estimation
        'window_assignment': ['peak_detection'], # Stage 4: Requires peaks
        'fitting': ['window_assignment']     # Stage 5: Requires windows
    }
    
    def __init__(self, completed_stages: Optional[list] = None):
        """
        Initialize stage tracker.
        
        Parameters
        ----------
        completed_stages : list, optional
            List of completed stage names
        """
        self.completed_stages = set(completed_stages or [])
    
    def mark_completed(self, stage_name: str) -> None:
        """Mark a stage as completed."""
        if stage_name not in self.STAGE_DEPENDENCIES:
            raise ValueError(f"Unknown stage: {stage_name}")
        self.completed_stages.add(stage_name)
    
    def is_completed(self, stage_name: str) -> bool:
        """Check if a stage is completed."""
        return stage_name in self.completed_stages
    
    def validate_dependencies(self, stage_name: str) -> None:
        """
        Validate that all dependencies for a stage are met.
        
        Parameters
        ----------
        stage_name : str
            Name of the stage to validate
            
        Raises
        ------
        PipelineStageError
            If required dependencies are missing
        """
        if stage_name not in self.STAGE_DEPENDENCIES:
            raise ValueError(f"Unknown stage: {stage_name}")
        
        required_deps = self.STAGE_DEPENDENCIES[stage_name]
        missing_deps = [dep for dep in required_deps if dep not in self.completed_stages]
        
        if missing_deps:
            raise PipelineStageError(stage_name, missing_deps, Path("unknown"))  # filepath added by caller
    
    def get_next_available_stages(self) -> list:
        """Get list of stages that can be run next."""
        available = []
        for stage_name, dependencies in self.STAGE_DEPENDENCIES.items():
            if stage_name not in self.completed_stages:
                if all(dep in self.completed_stages for dep in dependencies):
                    available.append(stage_name)
        return available
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'completed_stages': list(self.completed_stages),
            'next_available': self.get_next_available_stages()
        }


class PipelineFileManager:
    """
    Core file manager for .ftmw pipeline data files.
    
    Provides safe file operations, source tracking, and stage dependency management
    for the dual-interface architecture (Pipeline class + functional API + CLI).
    """
    
    def __init__(self):
        """Initialize the file manager."""
        self.logger = logging.getLogger(__name__)
    
    def create_pipeline_file(self, filepath: Union[str, Path], fid: FID, 
                           source_metadata: SourceMetadata, 
                           force: bool = False) -> Path:
        """
        Create a new pipeline file with FID data and source metadata.
        
        Parameters
        ----------
        filepath : str or Path
            Path for the new pipeline file (should have .ftmw extension)
        fid : FID
            FID object to store as Stage 0 data
        source_metadata : SourceMetadata
            Source provenance information
        force : bool, default False
            If True, overwrite existing file even with different source
            
        Returns
        -------
        Path
            Path to the created pipeline file
            
        Raises
        ------
        PipelineExistsError
            If file exists with different source and force=False
        ValueError
            If inputs are invalid
        RuntimeError
            If file creation fails
        """
        filepath = Path(filepath)
        
        # Add .ftmw extension if not present
        if filepath.suffix != '.ftmw':
            filepath = filepath.with_suffix('.ftmw')
        
        # Check for existing file
        if filepath.exists() and not force:
            existing_metadata = self._load_source_metadata(filepath)
            if existing_metadata and not existing_metadata.matches(source_metadata):
                raise PipelineExistsError(
                    filepath, 
                    str(existing_metadata.source_path), 
                    str(source_metadata.source_path)
                )
            elif existing_metadata and existing_metadata.matches(source_metadata):
                self.logger.info(
                    f"ℹ️  Found existing pipeline with identical source. Loading existing data: {filepath}"
                )
                return filepath
        
        # Create parent directory if needed
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Create new HDF5 file
            with h5py.File(filepath, 'w') as h5f:
                # Store file metadata
                h5f.attrs['file_type'] = 'ftmw_pipeline'
                h5f.attrs['file_version'] = '1.0'
                h5f.attrs['creation_timestamp'] = datetime.now().isoformat()
                
                # Store source metadata
                source_group = h5f.create_group('source_metadata')
                source_dict = source_metadata.to_dict()
                for key, value in source_dict.items():
                    if isinstance(value, dict):
                        # Store dictionaries as JSON strings
                        source_group.attrs[key] = json.dumps(value, default=str)
                    else:
                        source_group.attrs[key] = value
                
                # Initialize stage tracker
                stages_group = h5f.create_group('pipeline_stages')
                stages_group.attrs['completed_stages'] = json.dumps(['fid_import'])
                
                # Store Stage 0 data (FID)
                stage0_group = h5f.create_group('stage0_fid_data')
                save_fid_to_hdf5(fid, stage0_group)
            
            self.logger.info(f"✅ Created pipeline file: {filepath}")
            
            if force and filepath.exists():
                self.logger.warning(f"⚠️  Overwrote existing pipeline file: {filepath}")
            
            return filepath
            
        except Exception as e:
            # Clean up partial file on failure
            if filepath.exists():
                try:
                    filepath.unlink()
                except Exception:
                    pass
            raise RuntimeError(f"Failed to create pipeline file {filepath}: {e}") from e
    
    def open_pipeline_file(self, filepath: Union[str, Path]) -> Tuple[Path, SourceMetadata, PipelineStageTracker]:
        """
        Open an existing pipeline file and return metadata.
        
        Parameters
        ----------
        filepath : str or Path
            Path to the pipeline file
            
        Returns
        -------
        tuple
            (filepath, source_metadata, stage_tracker)
            
        Raises
        ------
        FileNotFoundError
            If pipeline file doesn't exist
        PipelineCorruptedError
            If file appears to be corrupted
        ValueError
            If file format is invalid
        """
        filepath = Path(filepath)
        
        # Check file existence
        if not filepath.exists():
            raise FileNotFoundError(
                f"Pipeline file not found: {filepath}\n\n"
                f"To create a new pipeline:\n"
                f"  Pipeline.create('{filepath}', source='path/to/data/')\n"
                f"  # or\n"
                f"  ftmwpipeline import-data {filepath} --source path/to/data/"
            )
        
        try:
            with h5py.File(filepath, 'r') as h5f:
                # Validate file format
                if h5f.attrs.get('file_type') != 'ftmw_pipeline':
                    raise ValueError(f"Not a valid pipeline file: {filepath}")
                
                # Load source metadata
                source_metadata = self._load_source_metadata(filepath, h5f)
                
                # Load stage tracker
                stage_tracker = self._load_stage_tracker(filepath, h5f)
                
                return filepath, source_metadata, stage_tracker
                
        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError)):
                raise
            else:
                raise PipelineCorruptedError(
                    filepath, 
                    f"Unable to read file structure: {e}"
                )
    
    def validate_pipeline_file(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate pipeline file integrity and return validation report.
        
        Parameters
        ----------
        filepath : str or Path
            Path to the pipeline file
            
        Returns
        -------
        dict
            Validation report with status and details
        """
        filepath = Path(filepath)
        report = {
            'filepath': str(filepath),
            'valid': False,
            'errors': [],
            'warnings': [],
            'stages': {},
            'source_info': {}
        }
        
        try:
            # Check file existence and access
            if not filepath.exists():
                report['errors'].append("File does not exist")
                return report
            
            if not os.access(filepath, os.R_OK):
                report['errors'].append("File is not readable")
                return report
            
            # Open and validate HDF5 structure
            with h5py.File(filepath, 'r') as h5f:
                # Check file type
                if h5f.attrs.get('file_type') != 'ftmw_pipeline':
                    report['errors'].append("Not a valid pipeline file (wrong file_type)")
                    return report
                
                # Check required groups
                required_groups = ['source_metadata', 'pipeline_stages', 'stage0_fid_data']
                for group_name in required_groups:
                    if group_name not in h5f:
                        report['errors'].append(f"Missing required group: {group_name}")
                
                if report['errors']:
                    return report
                
                # Validate source metadata
                try:
                    source_metadata = self._load_source_metadata(filepath, h5f)
                    report['source_info'] = {
                        'source_path': str(source_metadata.source_path),
                        'format': source_metadata.format_name,
                        'import_time': source_metadata.import_timestamp.isoformat(),
                        'source_exists': source_metadata.source_path.exists()
                    }
                    
                    if not source_metadata.source_path.exists():
                        report['warnings'].append(f"Source data no longer exists: {source_metadata.source_path}")
                        
                except Exception as e:
                    report['errors'].append(f"Invalid source metadata: {e}")
                
                # Validate stage data
                try:
                    stage_tracker = self._load_stage_tracker(filepath, h5f)
                    report['stages'] = stage_tracker.to_dict()
                    
                    # Check FID data integrity
                    if 'fid_import' in stage_tracker.completed_stages:
                        try:
                            self.load_fid_data(filepath)
                            report['stages']['fid_data_valid'] = True
                        except Exception as e:
                            report['errors'].append(f"FID data corrupted: {e}")
                            report['stages']['fid_data_valid'] = False
                    
                except Exception as e:
                    report['errors'].append(f"Invalid stage data: {e}")
            
            # Set overall validity
            report['valid'] = len(report['errors']) == 0
            
        except Exception as e:
            report['errors'].append(f"File validation failed: {e}")
        
        return report
    
    def load_fid_data(self, filepath: Union[str, Path]) -> FID:
        """
        Load FID data from pipeline file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to the pipeline file
            
        Returns
        -------
        FID
            The FID object stored in the pipeline
            
        Raises
        ------
        FileNotFoundError
            If pipeline file doesn't exist
        PipelineStageError
            If FID data stage is not complete
        RuntimeError
            If loading fails
        """
        filepath = Path(filepath)
        
        # Validate file and check stage completion
        _, _, stage_tracker = self.open_pipeline_file(filepath)
        if not stage_tracker.is_completed('fid_import'):
            raise PipelineStageError('fid_import', ['data_import'], filepath)
        
        try:
            with h5py.File(filepath, 'r') as h5f:
                if 'stage0_fid_data' not in h5f:
                    raise RuntimeError("FID data group missing from pipeline file")
                
                return load_fid_from_hdf5(h5f['stage0_fid_data'])
                
        except Exception as e:
            raise RuntimeError(f"Failed to load FID data from {filepath}: {e}") from e
    
    def check_stage_dependencies(self, filepath: Union[str, Path], 
                                stage_name: str) -> None:
        """
        Check if a pipeline stage can be executed.
        
        Parameters
        ----------
        filepath : str or Path
            Path to the pipeline file
        stage_name : str
            Name of the stage to check
            
        Raises
        ------
        PipelineStageError
            If stage dependencies are not met
        FileNotFoundError
            If pipeline file doesn't exist
        """
        filepath = Path(filepath)
        _, _, stage_tracker = self.open_pipeline_file(filepath)
        
        try:
            stage_tracker.validate_dependencies(stage_name)
        except PipelineStageError as e:
            # Update the filepath in the exception
            e.filepath = filepath
            raise
    
    def mark_stage_completed(self, filepath: Union[str, Path], 
                           stage_name: str) -> None:
        """
        Mark a pipeline stage as completed.
        
        Parameters
        ----------
        filepath : str or Path
            Path to the pipeline file
        stage_name : str
            Name of the completed stage
            
        Raises
        ------
        ValueError
            If stage name is invalid
        RuntimeError
            If update fails
        """
        filepath = Path(filepath)
        
        try:
            with h5py.File(filepath, 'r+') as h5f:
                stage_tracker = self._load_stage_tracker(filepath, h5f)
                stage_tracker.mark_completed(stage_name)
                
                # Update stored stage information
                stages_group = h5f['pipeline_stages']
                stages_group.attrs['completed_stages'] = json.dumps(
                    list(stage_tracker.completed_stages)
                )
                stages_group.attrs['last_updated'] = datetime.now().isoformat()
                
        except Exception as e:
            raise RuntimeError(f"Failed to update stage completion for {filepath}: {e}") from e
    
    def _load_source_metadata(self, filepath: Path, 
                            h5f: Optional[h5py.File] = None) -> Optional[SourceMetadata]:
        """Load source metadata from pipeline file."""
        should_close = h5f is None
        if h5f is None:
            h5f = h5py.File(filepath, 'r')
        
        try:
            if 'source_metadata' not in h5f:
                return None
            
            source_group = h5f['source_metadata']
            source_dict = {}
            
            for key in source_group.attrs.keys():
                value = source_group.attrs[key]
                if key == 'loader_parameters':
                    # Parse JSON string back to dict
                    source_dict[key] = json.loads(value)
                else:
                    source_dict[key] = value
            
            return SourceMetadata.from_dict(source_dict)
            
        finally:
            if should_close:
                h5f.close()
    
    def _load_stage_tracker(self, filepath: Path, 
                          h5f: Optional[h5py.File] = None) -> PipelineStageTracker:
        """Load stage tracker from pipeline file."""
        should_close = h5f is None
        if h5f is None:
            h5f = h5py.File(filepath, 'r')
        
        try:
            if 'pipeline_stages' not in h5f:
                return PipelineStageTracker()
            
            stages_group = h5f['pipeline_stages']
            completed_stages_json = stages_group.attrs.get('completed_stages', '[]')
            completed_stages = json.loads(completed_stages_json)
            
            return PipelineStageTracker(completed_stages)
            
        finally:
            if should_close:
                h5f.close()