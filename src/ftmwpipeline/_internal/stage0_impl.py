"""
Shared implementation for Stage 0: Data Import and FID Operations.

This module contains the core implementation functions for data loading,
FID caching, and FID visualization that are shared between CLI, Pipeline class,
and functional API interfaces.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple

from ..contract import CancelToken, EventCallback, Stage, stage_for_key
from ..core.data_structures import FID, ChirpWindow
from ..core.stage_fit_settings import coerce_clock_sources
from ..core.start_detection_settings import StartDetectionSettings
from ..file_manager import (
    BadSettingError,
    PipelineFileError,
    PipelineFileNotFoundError,
    SourceMetadata,
    canonical_invalidated,
    create_pipeline_file,
    open_pipeline_file,
    pipeline_file_path,
    stages_an_import_replaces,
)
from ..io.data_loaders import (
    detect_format,
    list_formats,
    load_fid,
    validate_source,
)
from ..io.fid_serialization import load_fid_from_hdf5
from ..io.stage_fit_settings_serialization import write_recommended_chirp_window
from .atomic import atomic_write, h5open

if TYPE_CHECKING:
    from .events import StageScope

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


def persist_chirp_window_metadata(
    file_path: str, fid: FID, *, events: Optional["StageScope"] = None
) -> List[str]:
    """Persist a loader-declared chirp window and derive the start hint.

    When the loader attached ``fid.metadata["chirp_window"]``, write it to the
    ``recommended_chirp_window`` attr and -- unless the loader already recorded
    an experimenter start (e.g. Blackchirp ``FidStartUs``, which outranks the
    derived value) -- stamp ``recommended_processing.start_us = chirp_end +
    margin`` so a later FT inherits a physically grounded start without
    running the sweep detector.  Failures are non-fatal: the declaration is
    advisory metadata. Returns the storage keys of the stages the start hint
    invalidated (only ever on a reused file with a pre-provenance Stage 1
    record; see :func:`~.stage1_impl.write_recommended_ft_params`).
    """
    raw_chirp = fid.metadata.get("chirp_window")
    if raw_chirp is None:
        return []
    invalidated: List[str] = []
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
            from .stage1_impl import write_recommended_ft_params

            margin = chirp_window.start_margin_us
            if margin is None:
                margin = StartDetectionSettings().guard_margin_us
            recommended_start = chirp_window.chirp_end_us + margin
            invalidated = write_recommended_ft_params(
                file_path, {"start_us": recommended_start}, events=events
            )
            logger.info(
                "Derived recommended start_us = %.3f us "
                "(chirp_end %.3f + margin %.3f) from chirp window.",
                recommended_start,
                chirp_window.chirp_end_us,
                margin,
            )
    except Exception as exc:
        logger.warning("Could not persist declared chirp window: %s", exc)
    return invalidated


def persist_loader_metadata(
    file_path: str, fid: FID, *, events: Optional["StageScope"] = None
) -> List[str]:
    """Persist what the loader declared beside the samples: the instrument
    clock declaration and the chirp window (with its derived start hint).

    Both are advisory metadata, so a failure is logged, never raised. The
    clock declaration is written through
    :func:`~.clocks_impl.write_declaration`, which keeps a stored
    final-products table consistent with it when the import reuses an
    existing file. Returns the storage keys of the stages the chirp window's
    start hint invalidated (:func:`persist_chirp_window_metadata`), reported
    through ``events`` (the import's stage scope) when given.
    """
    raw_clocks = fid.metadata.get("clock_sources")
    if raw_clocks is not None:
        try:
            from .clocks_impl import write_declaration

            clock_tuple = coerce_clock_sources(raw_clocks)
            write_declaration(str(file_path), clock_tuple)
            logger.info(
                "Persisted %d recommended clock source(s) from loader metadata.",
                len(clock_tuple) if clock_tuple is not None else 0,
            )
        except Exception as exc:
            logger.warning("Could not persist recommended clock sources: %s", exc)

    return persist_chirp_window_metadata(str(file_path), fid, events=events)


#: The ``ftmw/run_result@1`` summary keys of ``data import`` -- also the keys of
#: the import's ``StageFinished.summary``. The one builder is
#: :func:`data_import_summary`.
DATA_IMPORT_SUMMARY_KEYS: Tuple[str, ...] = (
    "pipeline_file",
    "source_format",
    "n_points",
    "duration_us",
    "probe_freq_mhz",
    "sideband",
    "shots",
    "file_size_mb",
)


def data_import_summary(result: Mapping[str, Any]) -> Dict[str, Any]:
    """The scalar summary of an :func:`import_data_impl` result."""
    fid_meta = result["fid_metadata"]
    return {
        "pipeline_file": str(result["pipeline_file"]),
        "source_format": result["format_name"],
        "n_points": fid_meta["n_points"],
        "duration_us": fid_meta["duration_us"],
        "probe_freq_mhz": fid_meta["probe_freq_mhz"],
        "sideband": fid_meta["sideband"],
        "shots": fid_meta["shots"],
        "file_size_mb": Path(result["pipeline_file"]).stat().st_size / (1024 * 1024),
    }


def import_data_impl(
    file_path: str,
    source: str,
    format_name: Optional[str] = None,
    force: bool = False,
    *,
    fid_index: Optional[int] = None,
    events: Optional[EventCallback] = None,
    cancel: Optional[CancelToken] = None,
    **format_params: Any,
) -> Dict[str, Any]:
    """
    Shared implementation for data import into .ftmw pipeline files.

    The one implementation behind ``data import``, :meth:`Pipeline.create` and
    ``import_data``. Reports as the ``data`` stage of the ``data import``
    operation (or of the enclosing operation whose events are passed):
    ``StageStarted``, ``Invalidated`` (once, for everything an overwrite or the
    loader's start hint dropped), then ``StageFinished`` with
    :func:`data_import_summary` once the file is written. ``cancel`` is checked
    before the import starts; once it starts it completes.

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
    fid_index : int, optional
        FID index for a multi-FID Blackchirp source (added to the loader
        parameters when the resolved format is ``blackchirp``).
    events, cancel
        The long-operation event callback and cancel token.
    **format_params
        Format-specific loading parameters

    Returns
    -------
    dict
        Result information including FID metadata, file paths, and status

    Raises
    ------
    PipelineFileNotFoundError
        (``not_found``, kind ``"file"``; also a ``FileNotFoundError``) If the
        source path does not exist.
    BadSettingError
        (``bad_setting``; also a ``ValueError``) ``path`` ``"format"`` if
        format detection fails or the format is unknown; ``path`` ``"source"``
        if the source does not validate under the resolved format.
    """
    from .events import operation_events

    ops = operation_events("data import", events, cancel)
    with ops.stage(Stage.DATA, verb="data import") as scope:
        # One atomic write of the file the import creates (or overwrites):
        # ``create_pipeline_file`` adds the ``.ftmw`` suffix, so the
        # transaction is on the suffixed path. StageFinished follows the
        # replace.
        with atomic_write(pipeline_file_path(file_path)):
            result = _import_data(
                file_path,
                source,
                format_name,
                force,
                fid_index=fid_index,
                events=scope,
                format_params=format_params,
            )
        scope.finish(data_import_summary(result))
    return result


def _import_data(
    file_path: str,
    source: str,
    format_name: Optional[str],
    force: bool,
    *,
    fid_index: Optional[int],
    events: "StageScope",
    format_params: Dict[str, Any],
) -> Dict[str, Any]:
    """The body of :func:`import_data_impl` (inside the ``data`` stage scope)."""
    source_path = Path(source)
    if not source_path.exists():
        # not_found, kind "file" (also a FileNotFoundError), as preview_source.
        raise PipelineFileNotFoundError(
            source_path, message=f"Source path does not exist: {source_path}"
        )

    # Format detection or validation
    if format_name is None:
        logger.info("Auto-detecting data format...")
        format_name = detect_format(source_path)
        if format_name is None:
            available_formats = ", ".join(list_formats())
            raise BadSettingError(
                "format",
                f"one of: {available_formats} (auto-detection found none)",
                None,
                message=(
                    f"Could not detect data format for: {source_path}. "
                    f"Available formats: {available_formats}"
                ),
            )
        logger.info(f"Detected format: {format_name}")
    else:
        known_formats = list_formats()
        if format_name not in known_formats:
            raise BadSettingError(
                "format",
                f"one of: {', '.join(known_formats)}",
                format_name,
                message=(
                    f"Unknown format '{format_name}'. "
                    f"Available formats: {known_formats}"
                ),
            )
        logger.info(f"Using specified format: {format_name}")

    # Validate source with detected/specified format
    logger.info(f"Validating source with {format_name} loader...")
    validation = validate_source(source_path, format_name)

    if not validation["valid"]:
        error_details = "; ".join(validation["errors"])
        raise BadSettingError(
            "source",
            f"a source the {format_name} loader can read",
            str(source_path),
            message=f"Source validation failed: {error_details}",
        )

    logger.info("Source validation passed")

    if format_name == "blackchirp" and fid_index is not None:
        format_params["fid_index"] = fid_index

    # Load FID data
    logger.info("Loading FID data...")
    try:
        fid = load_fid(source_path, format_name, **format_params)
        logger.info(
            f"FID data loaded successfully: {fid.n_points:,} points, {fid.duration_us:.1f} μs"
        )
    except (ValueError, PipelineFileError, FileNotFoundError):
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to load FID data: {e}") from e

    # Create pipeline file with source metadata
    logger.info("Creating pipeline file...")
    try:
        # Create source metadata
        source_metadata = SourceMetadata(
            source_path=source_path,
            format_name=format_name,
            loader_parameters=format_params,
        )

        # Create the pipeline file. Overwriting a file discards every stage it
        # held, which the result reports as invalidated. The overwrite and the
        # loader's start hint are reported as ONE Invalidated event (the
        # stage scope merges them and delivers it after the replace).
        invalidated = stages_an_import_replaces(file_path, force)
        pipeline_file = create_pipeline_file(
            filepath=file_path,
            fid=fid,
            source_metadata=source_metadata,
            force=force,
        )
        logger.info(f"Pipeline file created: {pipeline_file}")
        # The overwrite logs no warning line of its own (it never did); its
        # stages join the one Invalidated event.
        events.invalidated(
            [stage_for_key(k) for k in invalidated],
            reason=None,
        )

        # Written after create_pipeline_file so stage0_fid_data exists.
        invalidated += persist_loader_metadata(str(pipeline_file), fid, events=events)
    except (PipelineFileError, ValueError, OSError):
        # OSError (PermissionError, a full disk, ...) propagates unwrapped, as
        # creating the file always did on the Python interfaces.
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to create pipeline file: {e}") from e

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
        "invalidated": list(canonical_invalidated(invalidated)),
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

        with h5open(file_path_obj, "r") as h5f:
            if "stage0_fid_data" not in h5f:
                raise ValueError("Invalid pipeline file: missing 'fid_data' group")
            fid = load_fid_from_hdf5(h5f["stage0_fid_data"])
        return fid
    except PipelineFileError:
        raise
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
    except PipelineFileError:
        raise
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
    except PipelineFileError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to get pipeline info for {file_path}: {e}")
