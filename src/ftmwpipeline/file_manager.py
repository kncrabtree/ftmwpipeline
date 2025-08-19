"""
FTMW Pipeline File Manager - Foundation for dual-interface architecture.

This module provides utility functions for .ftmw pipeline data files, handling
file creation, opening, validation, source metadata tracking, and stage
dependency checking.

Key Features:
- Safe creation vs. opening semantics with smart re-import detection
- Source metadata tracking for reproducibility and conflict prevention
- Stage dependency validation with clear error messages
- File integrity checking and corruption recovery guidance
- Jupyter-safe patterns for interactive development

These utility functions serve as the foundation for both the Pipeline class and 
functional API, enabling consistent file operations across interfaces.
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


# Module-level logger for file manager operations
logger = logging.getLogger(__name__)


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
            f"Pipeline file already exists with different source:\n"
            f"  File: {filepath}\n"
            f"  Existing source: {existing_source}\n"
            f"  Requested source: {requested_source}\n\n"
            f"Options:\n"
            f"  1. Use force=True to overwrite: Pipeline.create('{filepath}', source='{requested_source}', force=True)\n"
            f"  2. Use different filename: Pipeline.create('new_name.ftmw', source='{requested_source}')\n"
            f"  3. Open existing file: Pipeline.open('{filepath}')"
        )


class StageDependencyError(PipelineFileError):
    """Raised when attempting to execute a stage without required dependencies."""
    
    def __init__(self, stage_name: str, missing_dependencies: list, filepath: Path):
        self.stage_name = stage_name
        self.missing_dependencies = missing_dependencies
        self.filepath = filepath
        super().__init__(
            f"Cannot execute {stage_name} - missing dependencies: {missing_dependencies}\n"
            f"File: {filepath}\n"
            f"Complete the required stages first."
        )


class PipelineCorruptionError(PipelineFileError):
    """Raised when pipeline file is corrupted or invalid."""
    
    def __init__(self, filepath: Path, corruption_details: str):
        self.filepath = filepath
        self.corruption_details = corruption_details
        super().__init__(
            f"Pipeline file is corrupted: {filepath}\n"
            f"Details: {corruption_details}\n"
            f"Try recreating from original data source."
        )


class SourceMetadata:
    """Metadata about the source data for a pipeline file."""
    
    def __init__(self, source_path: Union[str, Path], format_name: str, 
                 loader_parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize source metadata.
        
        Parameters
        ----------
        source_path : str or Path
            Path to the original data source
        format_name : str
            Name of the data format
        loader_parameters : dict, optional
            Parameters used with the data loader
        """
        self.source_path = Path(source_path)
        self.format_name = format_name
        self.loader_parameters = loader_parameters or {}
        self.source_mtime = self.source_path.stat().st_mtime if self.source_path.exists() else 0
        self.source_hash = self._compute_source_hash()
        self.import_timestamp = datetime.now()
    
    def _compute_source_hash(self) -> str:
        """Compute a quick hash for source comparison."""
        # Simple hash based on path, mtime, and loader params
        content = f"{self.source_path}:{self.source_mtime}:{json.dumps(self.loader_parameters, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()[:16]  # Short hash for metadata storage
    
    def matches(self, other: 'SourceMetadata') -> bool:
        """Check if this metadata matches another (same source, same parameters)."""
        return (self.source_path == other.source_path and 
                self.format_name == other.format_name and
                self.loader_parameters == other.loader_parameters and
                abs(self.source_mtime - other.source_mtime) < 1.0)  # Allow 1 second tolerance
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'source_path': str(self.source_path),
            'format_name': self.format_name,
            'loader_parameters': self.loader_parameters,
            'source_mtime': self.source_mtime,
            'source_hash': self.source_hash,
            'import_timestamp': self.import_timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SourceMetadata':
        """Create from dictionary."""
        metadata = cls(
            source_path=data['source_path'],
            format_name=data['format_name'],
            loader_parameters=data['loader_parameters']
        )
        metadata.source_mtime = data['source_mtime']
        metadata.source_hash = data['source_hash']
        metadata.import_timestamp = datetime.fromisoformat(data['import_timestamp'])
        return metadata


class PipelineStageTracker:
    """Tracks completion status of pipeline stages and validates dependencies."""
    
    # Define stage dependencies
    STAGE_DEPENDENCIES = {
        'stage0_fid_data': [],  # Stage 0 has no dependencies
        'stage1_complex_ft': ['stage0_fid_data'],  # Stage 1 requires Stage 0
        'stage2_noise_result': ['stage1_complex_ft'],  # Stage 2 requires Stage 1
        # Future stages...
    }
    
    def __init__(self, completed_stages: Optional[list] = None):
        """Initialize stage tracker."""
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
        """Validate that all dependencies for a stage are completed."""
        if stage_name not in self.STAGE_DEPENDENCIES:
            raise ValueError(f"Unknown stage: {stage_name}")
        
        required = self.STAGE_DEPENDENCIES[stage_name]
        missing = [dep for dep in required if dep not in self.completed_stages]
        
        if missing:
            raise StageDependencyError(stage_name, missing, Path("unknown"))
    
    def get_available_stages(self) -> list:
        """Get list of stages that can be executed next."""
        available = []
        for stage_name, dependencies in self.STAGE_DEPENDENCIES.items():
            if stage_name not in self.completed_stages:
                if all(dep in self.completed_stages for dep in dependencies):
                    available.append(stage_name)
        return available
    
    def get_next_available_stages(self) -> list:
        """Get stages that can be executed immediately."""
        return self.get_available_stages()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'completed_stages': list(self.completed_stages),
            'next_available': self.get_next_available_stages()
        }


def create_pipeline_file(filepath: Union[str, Path], fid: FID, 
                       source_metadata: SourceMetadata, 
                       force: bool = False) -> Path:
    """
    Create a new pipeline file with FID data and source metadata.
    
    Parameters
    ----------
    filepath : str or Path
        Path for the new pipeline file (will add .ftmw extension if needed)
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
        existing_metadata = _load_source_metadata(filepath)
        if existing_metadata and not existing_metadata.matches(source_metadata):
            raise PipelineExistsError(
                filepath, 
                str(existing_metadata.source_path), 
                str(source_metadata.source_path)
            )
        elif existing_metadata and existing_metadata.matches(source_metadata):
            logger.info(
                f"ℹ️  Found existing pipeline with identical source. Loading existing data: {filepath}"
            )
            return filepath
    
    # Create parent directory if needed
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with h5py.File(filepath, 'w') as h5f:
            # Store source metadata
            source_group = h5f.create_group('source_metadata')
            source_dict = source_metadata.to_dict()
            for key, value in source_dict.items():
                if key == 'loader_parameters':
                    # Store as JSON string
                    source_group.attrs[key] = json.dumps(value)
                else:
                    source_group.attrs[key] = value
            
            # Initialize stage tracking
            stages_group = h5f.create_group('pipeline_stages')
            stage_tracker = PipelineStageTracker()
            stage_tracker.mark_completed('stage0_fid_data')  # Stage 0 completed by creating file
            stages_group.attrs['completed_stages'] = json.dumps(list(stage_tracker.completed_stages))
            stages_group.attrs['created'] = datetime.now().isoformat()
            stages_group.attrs['last_updated'] = datetime.now().isoformat()
            
            # Store Stage 0 data (FID)
            stage0_group = h5f.create_group('stage0_fid_data')
            save_fid_to_hdf5(fid, stage0_group)
        
        logger.info(f"✅ Created pipeline file: {filepath}")
        
        if force and filepath.exists():
            logger.warning(f"⚠️  Overwrote existing pipeline file: {filepath}")
        
        return filepath
        
    except Exception as e:
        # Clean up partial file on error
        if filepath.exists():
            try:
                filepath.unlink()
            except Exception:
                pass
        raise RuntimeError(f"Failed to create pipeline file {filepath}: {e}") from e


def open_pipeline_file(filepath: Union[str, Path]) -> Tuple[Path, SourceMetadata, PipelineStageTracker]:
    """
    Open an existing pipeline file and return metadata.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the .ftmw pipeline file
        
    Returns
    -------
    tuple
        (filepath, source_metadata, stage_tracker)
        
    Raises
    ------
    FileNotFoundError
        If pipeline file does not exist
    PipelineCorruptionError
        If file is corrupted or invalid
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
            # Load metadata and stage information
            source_metadata = _load_source_metadata(filepath, h5f)
            stage_tracker = _load_stage_tracker(filepath, h5f)
            
            if source_metadata is None:
                raise PipelineCorruptionError(filepath, "Missing source metadata")
            
            return filepath, source_metadata, stage_tracker
            
    except h5py.Error as e:
        raise PipelineCorruptionError(filepath, f"HDF5 error: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to open pipeline file {filepath}: {e}") from e


def validate_pipeline_file(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate a pipeline file for integrity and correctness.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the .ftmw pipeline file
        
    Returns
    -------
    dict
        Validation results with status and any issues found
    """
    filepath = Path(filepath)
    errors = []
    warnings = []
    
    # Check file existence
    if not filepath.exists():
        return {
            'valid': False,
            'errors': [f"File does not exist: {filepath}"],
            'warnings': []
        }
    
    try:
        # Open and validate structure
        filepath_obj, source_metadata, stage_tracker = open_pipeline_file(filepath)
        
        # Validate source metadata
        if not source_metadata.source_path.exists():
            warnings.append(f"Original source file no longer exists: {source_metadata.source_path}")
        
        # Validate stage data
        with h5py.File(filepath, 'r') as h5f:
            for stage in stage_tracker.completed_stages:
                if stage not in h5f:
                    errors.append(f"Missing data for completed stage: {stage}")
        
        # Try loading FID data
        try:
            fid = load_fid_from_hdf5(filepath)
            if fid.n_points == 0:
                errors.append("FID data is empty")
        except Exception as e:
            errors.append(f"Cannot load FID data: {e}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'file_size': filepath.stat().st_size,
            'stages': stage_tracker.to_dict()
        }
        
    except Exception as e:
        return {
            'valid': False,
            'errors': [f"Validation failed: {e}"],
            'warnings': warnings
        }


def _load_source_metadata(filepath: Path, h5f: Optional[h5py.File] = None) -> Optional[SourceMetadata]:
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


def _load_stage_tracker(filepath: Path, h5f: Optional[h5py.File] = None) -> PipelineStageTracker:
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