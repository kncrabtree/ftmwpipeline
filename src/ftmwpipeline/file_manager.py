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
from .io.fid_serialization import load_fid_from_hdf5, save_fid_to_hdf5

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

    def __init__(
        self,
        source_path: Union[str, Path],
        format_name: str,
        loader_parameters: Optional[Dict[str, Any]] = None,
    ):
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
        self.source_mtime = (
            self.source_path.stat().st_mtime if self.source_path.exists() else 0
        )
        self.source_hash = self._compute_source_hash()
        self.import_timestamp = datetime.now()

    def _compute_source_hash(self) -> str:
        """Compute a quick hash for source comparison."""
        # Simple hash based on path, mtime, and loader params
        content = f"{self.source_path}:{self.source_mtime}:{json.dumps(self.loader_parameters, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()[
            :16
        ]  # Short hash for metadata storage

    def matches(self, other: "SourceMetadata") -> bool:
        """Check if this metadata matches another (same source, same parameters)."""
        return (
            self.source_path == other.source_path
            and self.format_name == other.format_name
            and self.loader_parameters == other.loader_parameters
            and abs(self.source_mtime - other.source_mtime) < 1.0
        )  # Allow 1 second tolerance

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_path": str(self.source_path),
            "format_name": self.format_name,
            "loader_parameters": self.loader_parameters,
            "source_mtime": self.source_mtime,
            "source_hash": self.source_hash,
            "import_timestamp": self.import_timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceMetadata":
        """Create from dictionary."""
        metadata = cls(
            source_path=data["source_path"],
            format_name=data["format_name"],
            loader_parameters=data["loader_parameters"],
        )
        metadata.source_mtime = data["source_mtime"]
        metadata.source_hash = data["source_hash"]
        metadata.import_timestamp = datetime.fromisoformat(data["import_timestamp"])
        return metadata


class PipelineStageTracker:
    """Tracks completion status of pipeline stages and validates dependencies."""

    # Define stage dependencies
    STAGE_DEPENDENCIES = {
        "stage0_fid_data": [],  # Stage 0 has no dependencies
        "stage1_complex_ft": ["stage0_fid_data"],  # Stage 1 requires Stage 0
        "stage2_noise_result": ["stage1_complex_ft"],  # Stage 2 requires Stage 1
        # Stage 2b runs the data-driven sliding-active-window STFT tau
        # calibration on the raw FID. It needs Stage 1 settings (start_us,
        # end_us, trim) to slice the FID and Stage 2 noise as the canonical
        # σ reference so the calibration matches the user spectrum.
        "stage2b_tau_calibration": [
            "stage0_fid_data",
            "stage1_complex_ft",
            "stage2_noise_result",
        ],
        # Gaussian-shape twin of Stage 2b: per-bin Voigt fits on the STFT
        # contributor pool yield the per-band τ_G majority that the
        # Stage 5 Gaussian path consumes. Independent of the pure-exp
        # Stage 2b (both can coexist on one file). Same dependencies.
        "stage2b_tau_G_calibration": [
            "stage0_fid_data",
            "stage1_complex_ft",
            "stage2_noise_result",
        ],
        # Stage 3 requires Stage 1 (FT) and Stage 2 (noise). Stage 2b is a
        # recommended dependency but not enforced as required: the gap pass
        # falls back to ``tau_basis_us = 5.0`` when no calibration is
        # present, preserving the legacy single-stage path during the
        # rollout of Stage 2b.
        "stage3_peaks": ["stage1_complex_ft", "stage2_noise_result"],
        # Stage 4 (window assignment) requires Stage 3 (peaks).
        "stage4_windows": ["stage3_peaks"],
        # Stage 5 (per-window fitting) consumes the Stage 4 plan AND the raw
        # FID -- the active-portion FT the fit runs on is computed on demand
        # from stage0_fid_data plus the canonical Stage 1 settings, so a
        # change to Stage 1 settings or a re-import invalidates Stage 5
        # through the existing canonical-settings/Stage 0 path. Stage 5 also
        # reads the Stage 2b calibration when present (the bidirectional
        # tau-anchoring penalty and rescue τ); same recommended-but-not-
        # required policy as Stage 3.
        "stage5_fitting": ["stage0_fid_data", "stage4_windows"],
        # Future stages...
    }

    # HDF5 location whose presence proves a completed stage's data is stored.
    # Note Stage 1 is intentionally lightweight: ComplexFT is recomputed
    # on-demand, so only its processing parameters are persisted (there is no
    # 'stage1_complex_ft' group). Stages whose key is absent here are validated
    # by a group named after the stage itself.
    STAGE_DATA_PATHS = {
        "stage0_fid_data": "stage0_fid_data",
        "stage1_complex_ft": "processing_parameters/ft_processing",
        "stage2_noise_result": "stage2_noise_result",
        "stage2b_tau_calibration": "stage2b_tau_calibration",
        "stage2b_tau_G_calibration": "stage2b_tau_G_calibration",
        "stage3_peaks": "stage3_peaks",
        "stage4_windows": "stage4_windows",
        "stage5_fitting": "stage5_fitting",
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
            "completed_stages": list(self.completed_stages),
            "next_available": self.get_next_available_stages(),
        }


def create_pipeline_file(
    filepath: Union[str, Path],
    fid: FID,
    source_metadata: SourceMetadata,
    force: bool = False,
) -> Path:
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
    if filepath.suffix != ".ftmw":
        filepath = filepath.with_suffix(".ftmw")

    # Check for existing file
    if filepath.exists() and not force:
        existing_metadata = _load_source_metadata(filepath)
        if existing_metadata and not existing_metadata.matches(source_metadata):
            raise PipelineExistsError(
                filepath,
                str(existing_metadata.source_path),
                str(source_metadata.source_path),
            )
        elif existing_metadata and existing_metadata.matches(source_metadata):
            logger.info(
                f"Found existing pipeline with identical source. Loading existing data: {filepath}"
            )
            return filepath

    # Create parent directory if needed
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        with h5py.File(filepath, "w") as h5f:
            # Store source metadata
            source_group = h5f.create_group("source_metadata")
            source_dict = source_metadata.to_dict()
            for key, value in source_dict.items():
                if key == "loader_parameters":
                    # Store as JSON string
                    source_group.attrs[key] = json.dumps(value)
                else:
                    source_group.attrs[key] = value

            # Initialize stage tracking
            stages_group = h5f.create_group("pipeline_stages")
            stage_tracker = PipelineStageTracker()
            stage_tracker.mark_completed(
                "stage0_fid_data"
            )  # Stage 0 completed by creating file
            stages_group.attrs["completed_stages"] = json.dumps(
                list(stage_tracker.completed_stages)
            )
            stages_group.attrs["created"] = datetime.now().isoformat()
            stages_group.attrs["last_updated"] = datetime.now().isoformat()

            # Store Stage 0 data (FID)
            stage0_group = h5f.create_group("stage0_fid_data")
            save_fid_to_hdf5(fid, stage0_group)

        logger.info(f"Created pipeline file: {filepath}")

        if force and filepath.exists():
            logger.warning(f"Overwrote existing pipeline file: {filepath}")

        return filepath

    except Exception as e:
        # Clean up partial file on error
        if filepath.exists():
            try:
                filepath.unlink()
            except Exception:
                pass
        raise RuntimeError(f"Failed to create pipeline file {filepath}: {e}") from e


def open_pipeline_file(
    filepath: Union[str, Path],
) -> Tuple[Path, SourceMetadata, PipelineStageTracker]:
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
        with h5py.File(filepath, "r") as h5f:
            # Load metadata and stage information
            source_metadata = _load_source_metadata(filepath, h5f)
            stage_tracker = _load_stage_tracker(filepath, h5f)

            if source_metadata is None:
                raise PipelineCorruptionError(filepath, "Missing source metadata")

            return filepath, source_metadata, stage_tracker

    except Exception as e:
        if "h5py" in str(type(e)).lower() or "hdf5" in str(e).lower():
            raise PipelineCorruptionError(filepath, f"HDF5 error: {e}") from e
        else:
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
            "valid": False,
            "errors": [f"File does not exist: {filepath}"],
            "warnings": [],
        }

    try:
        # Open and validate structure
        filepath_obj, source_metadata, stage_tracker = open_pipeline_file(filepath)

        # Validate source metadata
        if not source_metadata.source_path.exists():
            warnings.append(
                f"Original source file no longer exists: {source_metadata.source_path}"
            )

        # Validate stage data. Each completed stage must have its persisted
        # data present at its known HDF5 location (which is not always a group
        # named after the stage - e.g. Stage 1 is lightweight).
        with h5py.File(filepath, "r") as h5f:
            for stage in stage_tracker.completed_stages:
                data_path = PipelineStageTracker.STAGE_DATA_PATHS.get(stage, stage)
                if data_path not in h5f:
                    errors.append(f"Missing data for completed stage: {stage}")

        # Try loading FID data
        try:
            with h5py.File(filepath, "r") as h5f:
                stage0_group = h5f["stage0_fid_data"]
                fid = load_fid_from_hdf5(stage0_group)
                if fid.n_points == 0:
                    errors.append("FID data is empty")
        except Exception as e:
            errors.append(f"Cannot load FID data: {e}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "file_size": filepath.stat().st_size,
            "stages": stage_tracker.to_dict(),
        }

    except Exception as e:
        return {
            "valid": False,
            "errors": [f"Validation failed: {e}"],
            "warnings": warnings,
        }


def _load_source_metadata(
    filepath: Path, h5f: Optional[h5py.File] = None
) -> Optional[SourceMetadata]:
    """Load source metadata from pipeline file."""
    should_close = h5f is None
    if h5f is None:
        h5f = h5py.File(filepath, "r")

    try:
        if "source_metadata" not in h5f:
            return None

        source_group = h5f["source_metadata"]
        source_dict = {}

        for key in source_group.attrs.keys():
            value = source_group.attrs[key]
            if key == "loader_parameters":
                # Parse JSON string back to dict
                source_dict[key] = json.loads(value)
            else:
                source_dict[key] = value

        return SourceMetadata.from_dict(source_dict)

    finally:
        if should_close:
            h5f.close()


def _load_stage_tracker(
    filepath: Path, h5f: Optional[h5py.File] = None
) -> PipelineStageTracker:
    """Load stage tracker from pipeline file."""
    should_close = h5f is None
    if h5f is None:
        h5f = h5py.File(filepath, "r")

    try:
        if "pipeline_stages" not in h5f:
            return PipelineStageTracker()

        stages_group = h5f["pipeline_stages"]
        completed_stages_json = stages_group.attrs.get("completed_stages", "[]")
        completed_stages = json.loads(completed_stages_json)

        return PipelineStageTracker(completed_stages)

    finally:
        if should_close:
            h5f.close()


def invalidate_downstream_stages(filepath: Union[str, Path], stage_name: str) -> list:
    """Drop persisted data and completion for every stage that depends on ``stage_name``.

    Called when a stage is re-run (e.g. Stage 3 re-detection) so that stale
    downstream results -- computed against the now-superseded output -- do not
    linger. The stage itself is *not* touched, only its transitive dependents.

    Parameters
    ----------
    filepath : str or Path
        Path to the .ftmw pipeline file.
    stage_name : str
        The stage that was just (re-)run.

    Returns
    -------
    list of str
        The stage names that were invalidated (sorted), empty if none.
    """
    deps = PipelineStageTracker.STAGE_DEPENDENCIES
    paths = PipelineStageTracker.STAGE_DATA_PATHS

    dependents: set = set()
    changed = True
    while changed:
        changed = False
        for stage, required in deps.items():
            if stage in dependents:
                continue
            if any(req == stage_name or req in dependents for req in required):
                dependents.add(stage)
                changed = True

    if not dependents:
        return []

    with h5py.File(filepath, "a") as h5f:
        if "pipeline_stages" not in h5f:
            return []
        stages_group = h5f["pipeline_stages"]
        completed = json.loads(stages_group.attrs.get("completed_stages", "[]"))
        invalidated = []
        for stage in dependents:
            data_path = paths.get(stage, stage)
            if data_path in h5f:
                del h5f[data_path]
            if stage in completed:
                completed.remove(stage)
                invalidated.append(stage)
        if invalidated:
            stages_group.attrs["completed_stages"] = json.dumps(completed)
            stages_group.attrs["last_updated"] = datetime.now().isoformat()
            logger.warning(
                "Stage %s re-run; invalidated downstream stage(s) %s -- "
                "re-run them to refresh.",
                stage_name,
                sorted(invalidated),
            )
        return sorted(invalidated)


def update_processing_parameters(
    filepath: Union[str, Path], parameters: Dict[str, Any]
) -> None:
    """
    Update processing parameters in the pipeline file's recommended_processing section.

    This function updates the parameters stored in the FID's recommended_processing
    group, allowing parameter persistence from interactive sessions.

    Parameters
    ----------
    filepath : str or Path
        Path to the pipeline file
    parameters : dict
        Processing parameters to save. Keys can include:
        start_us, end_us, zpf, expf_us, window_function, units_power

    Raises
    ------
    FileNotFoundError
        If pipeline file does not exist
    ValueError
        If file format is invalid or parameters are invalid
    RuntimeError
        If parameter update fails

    Example
    -------
    >>> update_processing_parameters("exp_2638.ftmw", {
    ...     'zpf': 2,
    ...     'expf_us': 5.0,
    ...     'start_us': 2.0,
    ...     'end_us': 12.0
    ... })
    """
    import json

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Pipeline file does not exist: {filepath}")

    try:
        with h5py.File(filepath, "r+") as h5f:
            # Locate the recommended_processing group
            fid_group_path = "stage0_fid_data/recommended_processing"
            if fid_group_path not in h5f:
                raise ValueError(f"Invalid pipeline file: missing {fid_group_path}")

            rec_proc_group = h5f[fid_group_path]

            # Update attributes with new parameters
            for param_name, param_value in parameters.items():
                # Handle None values (HDF5 can't store None directly)
                if param_value is None:
                    attr_value = "__None__"
                else:
                    attr_value = param_value

                # Set the attribute
                rec_proc_group.attrs[param_name] = attr_value

            # Update last modified timestamp
            if "pipeline_stages" in h5f:
                stages_group = h5f["pipeline_stages"]
                from datetime import datetime

                stages_group.attrs["last_updated"] = datetime.now().isoformat()

        logger.info(f"Updated {len(parameters)} processing parameters in {filepath}")

    except Exception as e:
        raise RuntimeError(f"Failed to update processing parameters: {e}") from e
