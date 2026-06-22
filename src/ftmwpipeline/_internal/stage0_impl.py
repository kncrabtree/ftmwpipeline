"""
Shared implementation for Stage 0: Data Import and FID Operations.

This module contains the core implementation functions for data loading,
FID caching, and FID visualization that are shared between CLI, Pipeline class,
and functional API interfaces.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..core.data_structures import FID, ChirpWindow
from ..core.stage_fit_settings import coerce_clock_sources
from ..core.start_detection_settings import StartDetectionSettings
from ..file_manager import (
    SourceMetadata,
    create_pipeline_file,
    open_pipeline_file,
    update_processing_parameters,
    validate_pipeline_file,
)
from ..io.data_loaders import (
    detect_format,
    get_format_info,
    list_formats,
    load_fid,
    validate_source,
)
from ..io.fid_serialization import load_fid_from_hdf5
from ..io.stage_fit_settings_serialization import (
    write_recommended_chirp_window,
    write_recommended_clock_sources,
)

logger = logging.getLogger(__name__)


def _coerce_chirp_window(value: Any) -> ChirpWindow:
    """Coerce a loader-injected chirp-window value to a :class:`ChirpWindow`.

    Accepts a :class:`ChirpWindow` directly or a mapping with at least a
    ``chirp_end_us`` key.  Raises ``ValueError`` on unrecognized input.
    """
    if isinstance(value, ChirpWindow):
        return value
    if isinstance(value, dict):
        chirp_end_us = float(value["chirp_end_us"])
        chirp_start_us = (
            float(value["chirp_start_us"])
            if value.get("chirp_start_us") is not None
            else None
        )
        start_margin_us = (
            float(value["start_margin_us"])
            if value.get("start_margin_us") is not None
            else None
        )
        return ChirpWindow(
            chirp_end_us=chirp_end_us,
            chirp_start_us=chirp_start_us,
            start_margin_us=start_margin_us,
        )
    raise ValueError(f"Cannot coerce {type(value)} to ChirpWindow")


def persist_chirp_window_metadata(file_path: str, fid: FID) -> None:
    """Persist a loader-declared chirp window and derive the start hint.

    When the loader attached ``fid.metadata["chirp_window"]``, write it to the
    ``recommended_chirp_window`` attr and -- unless the loader already recorded
    an experimenter start (e.g. Blackchirp ``FidStartUs``, which outranks the
    derived value) -- stamp ``recommended_processing.start_us = chirp_end +
    margin`` so a later FT inherits a physically grounded start without
    running the sweep detector.  Failures are non-fatal: the declaration is
    advisory metadata.
    """
    raw_chirp = fid.metadata.get("chirp_window")
    if raw_chirp is None:
        return
    try:
        chirp_window = _coerce_chirp_window(raw_chirp)
        write_recommended_chirp_window(file_path, chirp_window)
        logger.info(
            "Persisted declared chirp window: chirp_end_us=%.3f, " "chirp_start_us=%s.",
            chirp_window.chirp_end_us,
            (
                f"{chirp_window.chirp_start_us:.3f}"
                if chirp_window.chirp_start_us is not None
                else "None"
            ),
        )
        if fid.processing.start_us is None:
            margin = chirp_window.start_margin_us
            if margin is None:
                margin = StartDetectionSettings().guard_margin_us
            recommended_start = chirp_window.chirp_end_us + margin
            update_processing_parameters(file_path, {"start_us": recommended_start})
            logger.info(
                "Derived recommended start_us = %.3f us "
                "(chirp_end %.3f + margin %.3f) from chirp window.",
                recommended_start,
                chirp_window.chirp_end_us,
                margin,
            )
    except Exception as exc:
        logger.warning("Could not persist declared chirp window: %s", exc)


def import_data_impl(
    file_path: str,
    source: str,
    format_name: Optional[str] = None,
    force: bool = False,
    **format_params: Any,
) -> Dict[str, Any]:
    """
    Shared implementation for data import into .ftmw pipeline files.

    This function handles the complete data import workflow:
    1. Format detection/validation
    2. Data loading from source
    3. Pipeline file creation with source metadata

    Parameters
    ----------
    file_path : str
        Path to the .ftmw pipeline file to create/update
    source : str
        Path to data source (file or directory)
    format_name : str, optional
        Data format name. If None, auto-detection is attempted
    force : bool, default False
        Overwrite an existing file even with a different source or layout.
    **format_params
        Format-specific loading parameters

    Returns
    -------
    dict
        Result information including FID metadata, file paths, and status

    Raises
    ------
    FileNotFoundError
        If source path does not exist
    ValueError
        If format detection fails or validation errors occur
    """
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")

    # Format detection or validation
    if format_name is None:
        logger.info("Auto-detecting data format...")
        format_name = detect_format(source_path)
        if format_name is None:
            available_formats = ", ".join(list_formats())
            raise ValueError(
                f"Could not detect data format for: {source_path}. "
                f"Available formats: {available_formats}"
            )
        logger.info(f"Detected format: {format_name}")
    else:
        logger.info(f"Using specified format: {format_name}")

    # Validate source with detected/specified format
    logger.info(f"Validating source with {format_name} loader...")
    validation = validate_source(source_path, format_name)

    if not validation["valid"]:
        error_details = "; ".join(validation["errors"])
        raise ValueError(f"Source validation failed: {error_details}")

    logger.info("Source validation passed")

    # Load FID data
    logger.info("Loading FID data...")
    try:
        fid = load_fid(source_path, format_name, **format_params)
        logger.info(
            f"FID data loaded successfully: {fid.n_points:,} points, {fid.duration_us:.1f} μs"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load FID data: {e}")

    # Create pipeline file with source metadata
    logger.info("Creating pipeline file...")
    try:
        # Create source metadata
        source_metadata = SourceMetadata(
            source_path=source_path,
            format_name=format_name,
            loader_parameters=format_params,
        )

        # Create the pipeline file
        pipeline_file = create_pipeline_file(
            filepath=file_path, fid=fid, source_metadata=source_metadata, force=force
        )
        logger.info(f"Pipeline file created: {pipeline_file}")

        # Persist any instrument clock declaration extracted by the loader.
        # This is written after create_pipeline_file so stage0_fid_data exists.
        raw_clocks = fid.metadata.get("clock_sources")
        if raw_clocks is not None:
            try:
                clock_tuple = coerce_clock_sources(raw_clocks)
                write_recommended_clock_sources(str(pipeline_file), clock_tuple)
                logger.info(
                    "Persisted %d recommended clock source(s) from loader metadata.",
                    len(clock_tuple) if clock_tuple is not None else 0,
                )
            except Exception as exc:
                logger.warning("Could not persist recommended clock sources: %s", exc)

        persist_chirp_window_metadata(str(pipeline_file), fid)
    except Exception as e:
        raise RuntimeError(f"Failed to create pipeline file: {e}")

    # Return comprehensive result information
    result = {
        "pipeline_file": str(pipeline_file),
        "source_path": str(source_path),
        "format_name": format_name,
        "fid_metadata": {
            "n_points": fid.n_points,
            "duration_us": fid.duration_us,
            "probe_freq_mhz": fid.probe_freq_mhz,
            "sideband": fid.sideband.value,
            "shots": fid.shots,
            "spacing": fid.spacing,
        },
        "validation_metadata": validation.get("metadata", {}),
        "loader_parameters": format_params,
        "status": "success",
    }

    return result


def load_fid_from_pipeline_impl(file_path: str) -> FID:
    """
    Shared implementation for loading FID data from .ftmw pipeline files.

    Parameters
    ----------
    file_path : str
        Path to the .ftmw pipeline file

    Returns
    -------
    FID
        The loaded FID object

    Raises
    ------
    FileNotFoundError
        If pipeline file does not exist
    ValueError
        If file is corrupted or invalid
    """
    try:
        # Open and validate the pipeline file
        file_path_obj, source_metadata, stage_tracker = open_pipeline_file(file_path)

        # Load FID data from the validated file
        import h5py

        with h5py.File(file_path_obj, "r") as h5f:
            if "stage0_fid_data" not in h5f:
                raise ValueError("Invalid pipeline file: missing 'fid_data' group")
            fid = load_fid_from_hdf5(h5f["stage0_fid_data"])
        return fid
    except Exception as e:
        raise RuntimeError(f"Failed to load FID from pipeline file {file_path}: {e}")


def visualize_fid_impl(
    file_path: str,
    show_metadata: bool = False,
    title: Optional[str] = None,
    **plot_kwargs: Any,
) -> Any:
    """
    Shared implementation for FID visualization from .ftmw pipeline files.

    Parameters
    ----------
    file_path : str
        Path to the .ftmw pipeline file
    show_metadata : bool, default False
        Whether to display metadata information
    title : str, optional
        Plot title. If None, auto-generated from file info
    **plot_kwargs
        Additional plotting parameters

    Returns
    -------
    matplotlib.Figure
        The created figure object

    Raises
    ------
    FileNotFoundError
        If pipeline file does not exist
    ImportError
        If visualization module is not available
    """
    # Load FID from pipeline file
    fid = load_fid_from_pipeline_impl(file_path)

    # Import visualization function
    try:
        from ..visualization.fid_visualization import plot_fid
    except ImportError:
        raise ImportError(
            "FID visualization not available - visualization module missing"
        )

    # Generate title if not provided
    if title is None:
        pipeline_name = Path(file_path).stem
        title = f"Pipeline {pipeline_name} - FID Data"

    # Create FID plot
    try:
        fig = plot_fid(fid, show_metadata=show_metadata, title=title, **plot_kwargs)
        logger.info("FID visualization completed successfully")
        return fig
    except Exception as e:
        raise RuntimeError(f"Failed to create FID visualization: {e}")


def get_pipeline_info_impl(file_path: str) -> Tuple[Path, SourceMetadata, Any, FID]:
    """
    Shared implementation for getting pipeline file information.

    Returns the raw objects rather than repackaging into a dict.
    Higher-level APIs can decide how to present this information.

    Parameters
    ----------
    file_path : str
        Path to the .ftmw pipeline file

    Returns
    -------
    tuple
        (file_path, source_metadata, stage_tracker, fid)
        Raw objects with all pipeline information

    Raises
    ------
    FileNotFoundError
        If pipeline file does not exist
    RuntimeError
        If file cannot be read
    """
    try:
        file_path_obj, source_metadata, stage_tracker = open_pipeline_file(file_path)
        fid = load_fid_from_hdf5(file_path_obj)
        return file_path_obj, source_metadata, stage_tracker, fid
    except Exception as e:
        raise RuntimeError(f"Failed to get pipeline info for {file_path}: {e}")


def validate_pipeline_file_impl(file_path: str) -> Dict[str, Any]:
    """
    Shared implementation for pipeline file validation.

    Parameters
    ----------
    file_path : str
        Path to the .ftmw pipeline file

    Returns
    -------
    dict
        Validation results with status and any issues found
    """
    try:
        return validate_pipeline_file(file_path)
    except Exception as e:
        return {
            "valid": False,
            "errors": [f"Failed to validate pipeline file: {e}"],
            "warnings": [],
        }
