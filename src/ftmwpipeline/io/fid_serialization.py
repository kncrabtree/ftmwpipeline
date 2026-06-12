"""
FID serialization for pipeline caching.

This module implements HDF5-based serialization for FID objects, preserving all
time-domain data, acquisition metadata, and source information needed for
proper FT processing and pipeline traceability.

The serialization stores:
- Complete time-domain voltage data (real-valued)
- Core acquisition parameters (spacing, probe frequency, sideband, shots)
- Optional format-specific processing recommendations (NOT requirements)
- Source metadata (file path, format, loading timestamp)
- Complete experimental metadata
- Acquisition segments (pre-record, tail, and optional per-frame data) when the
  source was a segmented scope record (keysight-mat loader).

This design decouples FID cache from specific processing choices, making cache files
portable and shareable. Processing parameters are stored as optional defaults that
can be overridden during FT processing.

Storage is compact since FID data is inherently small compared to frequency-domain data.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

import h5py
import numpy as np

from ..core.data_structures import FID, FIDProcessingParameters, Sideband


@dataclass
class AcquisitionSegments:
    """Persisted acquisition segments from a segmented scope record.

    Attributes
    ----------
    pre_record : np.ndarray
        Quiet pre-record voltage samples (float64).  When interleave cleanup
        was applied these are the post-subtraction values (same units as the
        science FID).
    tail : np.ndarray
        Trailing dead-time samples after the last frame (float64, may be
        empty).  Post-cleanup when interleave factors were given.
    frames : np.ndarray or None
        Per-frame array of shape ``(n_frames, frame_samples)`` (float64),
        present only when ``keep_frames`` was ``True`` at import time.
    pre_record_us : float
        Duration of the pre-record in µs (from the operator-supplied layout).
    frame_period_us : float
        Frame repetition period in µs.
    n_frames : int
        Number of frames.
    frame_selection : int or None
        Single-frame index that was used as the science FID, or ``None``
        when all frames were averaged.
    sample_dt : float
        Sample interval in seconds.
    interleave_patterns : dict[int, np.ndarray] or None
        Per-phase offset patterns estimated during interleave-offset cleanup,
        keyed by the interleave factor M.  ``None`` when no cleanup was
        applied.  Each value is a float64 array of length M (raw sample-unit
        DC offsets, not scaled to volts).
    """

    pre_record: np.ndarray
    tail: np.ndarray
    frames: Optional[np.ndarray]
    pre_record_us: float
    frame_period_us: float
    n_frames: int
    frame_selection: Optional[int]
    sample_dt: float
    interleave_patterns: Optional[Dict[int, np.ndarray]] = None


def save_fid_to_hdf5(fid: FID, h5_group: h5py.Group) -> None:
    """
    Save FID object to HDF5 group with complete metadata preservation.

    Parameters
    ----------
    fid : FID
        FID object to serialize
    h5_group : h5py.Group
        HDF5 group to save FID data to

    Raises
    ------
    ValueError
        If FID object is invalid
    RuntimeError
        If serialization fails

    HDF5 Structure
    --------------
    /fid_data/
    ├── time_series_data              [dataset: real voltage data, float64]
    ├── acquisition/                  [group: acquisition parameters]
    │   ├── spacing_seconds          [attr: float, time spacing in seconds]
    │   ├── probe_freq_mhz           [attr: float, probe/LO frequency]
    │   ├── sideband                 [attr: str, 'upper' or 'lower']
    │   ├── shots                    [attr: int, number of shots averaged]
    │   ├── n_points                 [attr: int, number of time points]
    │   └── duration_us              [attr: float, FID duration in microseconds]
    ├── recommended_processing/       [group: optional format-specific defaults]
    │   ├── description              [attr: str, explains these are suggestions]
    │   ├── start_us                 [attr: float or None, suggested start time]
    │   ├── end_us                   [attr: float or None, suggested end time]
    │   ├── rdc                      [attr: bool, suggested DC removal]
    │   └── units_power              [attr: int, suggested scaling units]
    └── metadata/                     [group: source and experimental metadata]
        ├── source_info              [dataset: JSON string with source metadata]
        └── experimental_data        [dataset: JSON string with experimental metadata]
    """
    try:
        if not isinstance(fid, FID):
            raise ValueError("Input must be a FID object")

        # Save time series data (this is the core FID data)
        h5_group.create_dataset(
            "time_series_data",
            data=fid.data,
            dtype=np.float64,
            compression="gzip",
            compression_opts=9,
            shuffle=True,
        )

        # Create acquisition parameters group
        acq_group = h5_group.create_group("acquisition")
        acq_group.attrs["spacing_seconds"] = fid.spacing
        acq_group.attrs["probe_freq_mhz"] = fid.probe_freq_mhz
        acq_group.attrs["sideband"] = fid.sideband.value
        acq_group.attrs["shots"] = fid.shots
        acq_group.attrs["n_points"] = fid.n_points
        acq_group.attrs["duration_us"] = fid.duration_us

        # Create recommended processing defaults group (NOT requirements)
        # These are format-specific suggestions, not cache requirements
        defaults_group = h5_group.create_group("recommended_processing")
        defaults_group.attrs["description"] = (
            "Format-specific processing recommendations (not requirements)"
        )
        defaults_group.attrs["start_us"] = _serialize_optional_float(
            fid.processing.start_us
        )
        defaults_group.attrs["end_us"] = _serialize_optional_float(
            fid.processing.end_us
        )
        defaults_group.attrs["rdc"] = fid.processing.rdc
        defaults_group.attrs["units_power"] = fid.processing.units_power

        # Create metadata group and save as JSON strings
        meta_group = h5_group.create_group("metadata")

        # Save source metadata (path, format, timestamp, etc.)
        source_metadata = {}
        if fid.metadata:
            # Extract source-related metadata
            source_keys = [
                "source_path",
                "source_format",
                "loader_class",
                "load_timestamp",
                "loader_parameters",
                "source_size",
                "source_modified",
                "fid_index",
                "blackchirp_params",
            ]
            for key in source_keys:
                if key in fid.metadata:
                    source_metadata[key] = fid.metadata[key]

        source_json = json.dumps(source_metadata, default=str, indent=2)
        meta_group.create_dataset(
            "source_info",
            data=source_json.encode("utf-8"),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

        # Save experimental metadata (everything else). Underscore-prefixed
        # keys are private in-memory transport (e.g. the loader-injected
        # ``_sliced_*`` segment arrays consumed by
        # ``save_acquisition_segments_to_hdf5``), not persistable metadata --
        # JSON-dumping a numpy array via ``default=str`` would store its
        # truncated repr.
        experimental_metadata = {}
        if fid.metadata:
            for key, value in fid.metadata.items():
                if key not in source_metadata and not key.startswith("_"):
                    experimental_metadata[key] = value

        experimental_json = json.dumps(experimental_metadata, default=str, indent=2)
        meta_group.create_dataset(
            "experimental_data",
            data=experimental_json.encode("utf-8"),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )

        # Persist acquisition segments when the source was a segmented scope
        # record (keysight-mat loader injects the required metadata keys).
        save_acquisition_segments_to_hdf5(fid, h5_group)

        # Add serialization metadata for portability
        h5_group.attrs["serialization_version"] = "1.0"
        h5_group.attrs["serialization_timestamp"] = datetime.now().isoformat()
        h5_group.attrs["object_type"] = "FID"
        h5_group.attrs["cache_description"] = (
            "Portable FID cache - contains all data needed for independent analysis"
        )

        # Add quick-access summary for cache portability
        h5_group.attrs["summary_probe_freq_mhz"] = fid.probe_freq_mhz
        h5_group.attrs["summary_sideband"] = fid.sideband.value
        h5_group.attrs["summary_duration_us"] = fid.duration_us
        h5_group.attrs["summary_n_points"] = fid.n_points

    except Exception as e:
        raise RuntimeError(f"Failed to serialize FID to HDF5: {e}") from e


def load_fid_from_hdf5(h5_group: h5py.Group) -> FID:
    """
    Load FID object from HDF5 group with complete metadata reconstruction.

    Parameters
    ----------
    h5_group : h5py.Group
        HDF5 group containing FID data

    Returns
    -------
    FID
        Reconstructed FID object with all original metadata

    Raises
    ------
    ValueError
        If HDF5 group structure is invalid
    RuntimeError
        If deserialization fails
    """
    try:
        # Validate HDF5 structure
        required_datasets = ["time_series_data"]
        required_groups = [
            "acquisition",
            "metadata",
        ]  # Note: processing is now optional 'recommended_processing'

        for dataset in required_datasets:
            if dataset not in h5_group:
                raise ValueError(f"Missing required dataset: {dataset}")

        for group in required_groups:
            if group not in h5_group:
                raise ValueError(f"Missing required group: {group}")

        # Load time series data
        time_series_data = h5_group["time_series_data"][:]

        # Load acquisition parameters
        acq_group = h5_group["acquisition"]
        spacing = float(acq_group.attrs["spacing_seconds"])
        probe_freq_mhz = float(acq_group.attrs["probe_freq_mhz"])
        sideband_str = acq_group.attrs["sideband"]
        if isinstance(sideband_str, bytes):
            sideband_str = sideband_str.decode("utf-8")
        sideband = Sideband(sideband_str)
        shots = int(acq_group.attrs["shots"])

        # Load processing parameters (now optional defaults)
        # Default to basic parameters if not found (for older cache files)
        if "recommended_processing" in h5_group:
            proc_group = h5_group["recommended_processing"]
        elif "processing" in h5_group:
            # Backward compatibility with older cache files
            proc_group = h5_group["processing"]
        else:
            # No processing defaults stored - use minimal defaults
            proc_group = None

        if proc_group is not None:
            # Legacy files may carry retired apodization keys (winf / zpf /
            # expf_us) in this group; they are ignored -- the canonical FT is
            # unconditionally unapodized and native-length.
            processing = FIDProcessingParameters(
                start_us=_deserialize_optional_float(proc_group.attrs["start_us"]),
                end_us=_deserialize_optional_float(proc_group.attrs["end_us"]),
                rdc=bool(proc_group.attrs["rdc"]),
                units_power=int(proc_group.attrs["units_power"]),
            )
        else:
            # Use minimal default processing parameters
            processing = FIDProcessingParameters()

        # Load metadata
        meta_group = h5_group["metadata"]
        metadata = {}

        # Load source metadata
        if "source_info" in meta_group:
            source_json_bytes = meta_group["source_info"][()]
            if isinstance(source_json_bytes, bytes):
                source_json = source_json_bytes.decode("utf-8")
            else:
                source_json = str(source_json_bytes)
            source_metadata = json.loads(source_json)
            metadata.update(source_metadata)

        # Load experimental metadata
        if "experimental_data" in meta_group:
            exp_json_bytes = meta_group["experimental_data"][()]
            if isinstance(exp_json_bytes, bytes):
                exp_json = exp_json_bytes.decode("utf-8")
            else:
                exp_json = str(exp_json_bytes)
            experimental_metadata = json.loads(exp_json)
            metadata.update(experimental_metadata)

        # Create and return FID object
        return FID(
            data=time_series_data,
            spacing=spacing,
            probe_freq_mhz=probe_freq_mhz,
            sideband=sideband,
            shots=shots,
            processing=processing,
            metadata=metadata,
        )

    except Exception as e:
        raise RuntimeError(f"Failed to deserialize FID from HDF5: {e}") from e


def save_acquisition_segments_to_hdf5(
    fid: "FID",
    h5_group: h5py.Group,
) -> None:
    """Persist acquisition segments alongside the FID when present.

    Called from :func:`save_fid_to_hdf5`.  The segments are extracted from
    ``fid.metadata`` keys written by the keysight-mat loader.  If those
    keys are absent (non-scope-record sources) this function is a no-op.

    HDF5 Structure
    --------------
    /acquisition_segments/
    ├── [attrs] pre_record_us, frame_period_us, n_frames, frame_selection,
    │           sample_dt
    ├── pre_record                 [dataset: float64, gzip]
    ├── tail                       [dataset: float64, gzip]
    ├── frames                     [dataset: float64, gzip; shape (n_frames, frame_samples),
    │                               only when keep_frames=True]
    └── interleave_pattern_<M>     [dataset: float64; one per factor M when
                                    interleave cleanup was applied]
    """
    layout_dict: Optional[Dict] = fid.metadata.get("acquisition_layout")
    pre_record: Optional[np.ndarray] = fid.metadata.get("_sliced_pre_record")
    tail: Optional[np.ndarray] = fid.metadata.get("_sliced_tail")

    if layout_dict is None or pre_record is None or tail is None:
        return

    seg_group = h5_group.create_group("acquisition_segments")

    # Segment map attributes
    seg_group.attrs["pre_record_us"] = float(layout_dict["pre_record_us"])
    seg_group.attrs["frame_period_us"] = float(layout_dict["frame_period_us"])
    seg_group.attrs["n_frames"] = int(layout_dict["n_frames"])
    frame_sel = layout_dict.get("frame")
    seg_group.attrs["frame_selection"] = (
        int(frame_sel) if frame_sel is not None else -1
    )  # -1 sentinel for None
    # sample_dt is the FID spacing (same clock)
    seg_group.attrs["sample_dt"] = float(fid.spacing)

    # Segment datasets (float64, gzip-compressed)
    _ds_kwargs = {"dtype": np.float64, "compression": "gzip", "compression_opts": 5}
    seg_group.create_dataset(
        "pre_record", data=np.asarray(pre_record, dtype=np.float64), **_ds_kwargs
    )
    seg_group.create_dataset(
        "tail", data=np.asarray(tail, dtype=np.float64), **_ds_kwargs
    )

    frames: Optional[np.ndarray] = fid.metadata.get("_sliced_frames")
    if frames is not None:
        seg_group.create_dataset(
            "frames", data=np.asarray(frames, dtype=np.float64), **_ds_kwargs
        )

    # Persist interleave-offset patterns for audit.  Each pattern is a small
    # float64 array of length M stored in raw sample units (before YInc
    # voltage scaling) for fidelity to the estimation domain.
    interleave_patterns: Optional[Dict] = fid.metadata.get("_interleave_patterns")
    if interleave_patterns:
        for m, pattern in interleave_patterns.items():
            seg_group.create_dataset(
                f"interleave_pattern_{m}",
                data=np.asarray(pattern, dtype=np.float64),
            )


def load_acquisition_segments_from_hdf5(
    h5_group: h5py.Group,
) -> Optional["AcquisitionSegments"]:
    """Load acquisition segments from an HDF5 group.

    Returns ``None`` if the ``acquisition_segments`` sub-group is absent
    (old files without segments load cleanly without error).

    Parameters
    ----------
    h5_group : h5py.Group
        The ``stage0_fid_data`` group.

    Returns
    -------
    AcquisitionSegments or None
    """
    if "acquisition_segments" not in h5_group:
        return None

    seg = h5_group["acquisition_segments"]

    pre_record_us = float(seg.attrs["pre_record_us"])
    frame_period_us = float(seg.attrs["frame_period_us"])
    n_frames = int(seg.attrs["n_frames"])
    frame_sel_raw = int(seg.attrs.get("frame_selection", -1))
    frame_selection: Optional[int] = None if frame_sel_raw == -1 else frame_sel_raw
    sample_dt = float(seg.attrs["sample_dt"])

    pre_record = np.asarray(seg["pre_record"][:], dtype=np.float64)
    tail = np.asarray(seg["tail"][:], dtype=np.float64)
    frames: Optional[np.ndarray] = None
    if "frames" in seg:
        frames = np.asarray(seg["frames"][:], dtype=np.float64)

    # Load interleave patterns when present.  Dataset names follow the
    # convention ``interleave_pattern_<M>`` written by
    # ``save_acquisition_segments_to_hdf5``.
    interleave_patterns: Optional[Dict[int, np.ndarray]] = None
    for key in seg.keys():
        if key.startswith("interleave_pattern_"):
            suffix = key[len("interleave_pattern_") :]
            try:
                m = int(suffix)
            except ValueError:
                continue
            if interleave_patterns is None:
                interleave_patterns = {}
            interleave_patterns[m] = np.asarray(seg[key][:], dtype=np.float64)

    return AcquisitionSegments(
        pre_record=pre_record,
        tail=tail,
        frames=frames,
        pre_record_us=pre_record_us,
        frame_period_us=frame_period_us,
        n_frames=n_frames,
        frame_selection=frame_selection,
        sample_dt=sample_dt,
        interleave_patterns=interleave_patterns,
    )


def save_fid_cache(experiment_id: str, fid: FID, cache_dir: str = "cache") -> Path:
    """
    Save FID to standalone cache file.

    Parameters
    ----------
    experiment_id : str
        Unique identifier for the experiment
    fid : FID
        FID object to cache
    cache_dir : str, default "cache"
        Directory to store cache files

    Returns
    -------
    Path
        Path to created cache file

    Raises
    ------
    ValueError
        If experiment_id is invalid
    RuntimeError
        If cache creation fails
    """
    try:
        if not experiment_id or not isinstance(experiment_id, str):
            raise ValueError("experiment_id must be a non-empty string")

        if not isinstance(fid, FID):
            raise ValueError("fid must be a FID object")

        # Create cache directory
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        # Generate cache filename
        cache_filename = f"{experiment_id}_fid.h5"
        cache_file = cache_path / cache_filename

        # Save to HDF5
        with h5py.File(cache_file, "w") as h5f:
            fid_group = h5f.create_group("fid_data")
            save_fid_to_hdf5(fid, fid_group)

            # Add cache metadata
            h5f.attrs["cache_type"] = "FID"
            h5f.attrs["experiment_id"] = experiment_id
            h5f.attrs["cache_timestamp"] = datetime.now().isoformat()

        return cache_file

    except Exception as e:
        raise RuntimeError(f"Failed to save FID cache for {experiment_id}: {e}") from e


def load_fid_cache(experiment_id: str, cache_dir: str = "cache") -> FID:
    """
    Load FID from standalone cache file.

    Parameters
    ----------
    experiment_id : str
        Unique identifier for the experiment
    cache_dir : str, default "cache"
        Directory containing cache files

    Returns
    -------
    FID
        Loaded FID object

    Raises
    ------
    FileNotFoundError
        If cache file does not exist
    ValueError
        If cache file is corrupted
    RuntimeError
        If cache loading fails
    """
    try:
        if not experiment_id or not isinstance(experiment_id, str):
            raise ValueError("experiment_id must be a non-empty string")

        # Locate cache file
        cache_path = Path(cache_dir)
        cache_filename = f"{experiment_id}_fid.h5"
        cache_file = cache_path / cache_filename

        if not cache_file.exists():
            raise FileNotFoundError(f"FID cache file not found: {cache_file}")

        # Load from HDF5
        with h5py.File(cache_file, "r") as h5f:
            if "fid_data" not in h5f:
                raise ValueError("Invalid FID cache file: missing 'fid_data' group")

            return load_fid_from_hdf5(h5f["fid_data"])

    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        else:
            raise RuntimeError(
                f"Failed to load FID cache for {experiment_id}: {e}"
            ) from e


def update_fid_processing_defaults(
    experiment_id: str, new_params: dict, cache_dir: str = "cache"
) -> None:
    """
    Update the recommended_processing section of FID cache with new defaults.

    This function allows updating the processing parameters stored in the FID cache
    to serve as defaults for future FT processing, enabling parameter persistence
    from interactive sessions.

    Parameters
    ----------
    experiment_id : str
        Unique identifier for the experiment
    new_params : dict
        New processing parameters to save as defaults.
        Keys can include: start_us, end_us, rdc, units_power
    cache_dir : str, default "cache"
        Directory containing cache files

    Raises
    ------
    FileNotFoundError
        If cache file does not exist
    ValueError
        If cache file is corrupted or parameters are invalid
    RuntimeError
        If cache update fails

    Example
    -------
    >>> # Update default parameters from interactive session
    >>> new_params = {'start_us': 1.0, 'end_us': 10.0}
    >>> update_fid_processing_defaults('exp_2638', new_params, 'cache/')
    """
    try:
        if not experiment_id or not isinstance(experiment_id, str):
            raise ValueError("experiment_id must be a non-empty string")

        if not isinstance(new_params, dict) or not new_params:
            raise ValueError("new_params must be a non-empty dictionary")

        # Locate cache file
        cache_path = Path(cache_dir)
        cache_filename = f"{experiment_id}_fid.h5"
        cache_file = cache_path / cache_filename

        if not cache_file.exists():
            raise FileNotFoundError(f"FID cache file not found: {cache_file}")

        # Update cache file in place
        with h5py.File(cache_file, "r+") as h5f:
            if "fid_data" not in h5f:
                raise ValueError("Invalid FID cache file: missing 'fid_data' group")

            fid_group = h5f["fid_data"]

            # Ensure recommended_processing group exists
            if "recommended_processing" not in fid_group:
                defaults_group = fid_group.create_group("recommended_processing")
                defaults_group.attrs["description"] = (
                    "Format-specific processing recommendations (not requirements)"
                )
            else:
                defaults_group = fid_group["recommended_processing"]

            # Update attributes with new parameters. The retired apodization
            # knobs (window_function / winf / zpf / expf_us) are not accepted --
            # the canonical FT is unconditionally unapodized and native-length.
            for param_name, param_value in new_params.items():
                if param_name in ("start_us", "end_us"):
                    # Optional float parameters
                    defaults_group.attrs[param_name] = _serialize_optional_float(
                        param_value
                    )
                elif param_name == "units_power":
                    defaults_group.attrs[param_name] = int(param_value)
                elif param_name == "rdc":
                    defaults_group.attrs[param_name] = bool(param_value)

    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        else:
            raise RuntimeError(
                f"Failed to update FID processing defaults for {experiment_id}: {e}"
            ) from e


# Helper functions for optional value serialization


def _serialize_optional_float(value: Optional[float]) -> Union[float, str]:
    """Serialize optional float value for HDF5 attribute storage."""
    if value is None:
        return "__None__"
    return float(value)


def _deserialize_optional_float(value: Union[float, str, bytes]) -> Optional[float]:
    """Deserialize optional float value from HDF5 attribute."""
    if isinstance(value, (bytes, str)):
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if value == "__None__":
            return None
    return float(value)


def _serialize_optional_str(value: Optional[str]) -> str:
    """Serialize optional string value for HDF5 attribute storage."""
    if value is None:
        return "__None__"
    return str(value)


def _deserialize_optional_str(value: Union[str, bytes]) -> Optional[str]:
    """Deserialize optional string value from HDF5 attribute."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if value == "__None__":
        return None
    return str(value)
