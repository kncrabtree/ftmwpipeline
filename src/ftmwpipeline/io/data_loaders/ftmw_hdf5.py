"""Native ftmwpipeline HDF5 input loader.

The canonical no-code "bring your own data" format: a self-describing HDF5 file
the user writes with h5py, MATLAB (``-v7.3``), or any HDF5 tool.  One file
carries the time-domain samples, the acquisition metadata, and -- optionally --
the instrument clock declarations the Stage 5 spur gate consumes.

Layout (version 1)::

    mydata.h5
      attrs:
        ftmw_input_version  int      = 1          # format marker + version (required)
        spacing_us          float64  = 0.02        # required
        probe_freq_mhz      float64  = 40960.0     # optional, default 0.0 (a direct sampler)
        sideband            string   = "lower"     # optional, default "upper"
        shots               int      = 100         # optional, default 1
        chirp_end_us        float64  = 1.5         # optional (enables a recommended start)
        chirp_start_us      float64  = 0.5         # optional
        start_margin_us     float64  = 0.5         # optional
        description         string   = "..."       # optional provenance, free text
      /fid                  float64[N]             # required: 1-D real voltage samples
      /clock_sources/                              # optional group
        freq_mhz            float64[K]
        locked              int8[K]                # 1 = referenced/locked, 0 = free-running
        label               string[K]             # variable-length UTF-8

Acquisition metadata embedded here is the lowest-precedence layer: an explicit
load parameter or a ``--metadata`` sidecar overrides it (see
:mod:`ftmwpipeline.io.input_metadata`).
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import h5py
import numpy as np

from ..input_metadata import (
    build_fid_metadata,
    explicit_layer_from_kwargs,
    find_sidecar,
    load_sidecar,
    resolve_input_metadata,
)
from .base import BaseLoader, LoaderError

if TYPE_CHECKING:
    from ...core.data_structures import FID

#: Highest input-format version this loader understands.
SUPPORTED_VERSION = 1

#: Root attribute that marks a file as the native input format.
VERSION_ATTR = "ftmw_input_version"


def _decode(value: Any) -> Any:
    """Decode an HDF5 attribute/scalar that may come back as bytes."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


class FtmwHdf5Loader(BaseLoader):
    """Loader for the native ftmwpipeline HDF5 input format.

    Identified by the ``.h5``/``.hdf5`` extension *and* a root
    ``ftmw_input_version`` attribute, so it never collides with arbitrary HDF5
    files.  Accepts the acquisition fields as explicit overrides and a
    ``metadata`` sidecar path; everything else is read from the file.
    """

    format_name = "ftmw-hdf5"
    file_extensions = [".h5", ".hdf5"]
    directory_indicators: List[str] = []

    def can_load(self, source_path: Union[str, Path]) -> bool:
        source_path = Path(source_path)
        if not source_path.is_file():
            return False
        if source_path.suffix.lower() not in self.file_extensions:
            return False
        try:
            with h5py.File(source_path, "r") as h5f:
                return VERSION_ATTR in h5f.attrs
        except Exception:
            return False

    def validate_source(
        self, source_path: Union[str, Path], **kwargs: Any
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "valid": False,
            "metadata": {},
            "options": {},
            "errors": [],
        }
        source_path = Path(source_path)

        if not self.can_load(source_path):
            result["errors"].append(
                "Not a native ftmwpipeline HDF5 file "
                f"(missing root '{VERSION_ATTR}' attribute)"
            )
            return result

        try:
            with h5py.File(source_path, "r") as h5f:
                version = int(h5f.attrs[VERSION_ATTR])
                if version > SUPPORTED_VERSION:
                    result["errors"].append(
                        f"Unsupported {VERSION_ATTR}={version} "
                        f"(this build understands up to {SUPPORTED_VERSION})"
                    )
                    return result
                if "fid" not in h5f:
                    result["errors"].append("Missing required '/fid' dataset")
                    return result

                n_points = int(h5f["fid"].shape[-1])
                embedded = self._read_embedded(h5f)
                result["metadata"]["n_points"] = n_points
                result["metadata"]["input_version"] = version
                for key in ("spacing_us", "probe_freq_mhz", "sideband", "shots"):
                    if embedded.get(key) is not None:
                        result["metadata"][key] = embedded[key]
                if embedded.get("spacing_us") is not None:
                    result["metadata"]["duration_us"] = n_points * float(
                        embedded["spacing_us"]
                    )
                clocks = embedded.get("clock_sources")
                result["metadata"]["n_clock_sources"] = len(clocks) if clocks else 0
                result["valid"] = True
        except Exception as exc:
            result["errors"].append(f"Validation failed: {exc}")

        return result

    def load_fid(self, source_path: Union[str, Path], **kwargs: Any) -> "FID":
        from ...core.data_structures import FID, FIDProcessingParameters

        source_path = Path(source_path)
        sidecar_path = kwargs.pop("metadata", None)

        try:
            with h5py.File(source_path, "r") as h5f:
                if VERSION_ATTR not in h5f.attrs:
                    raise LoaderError(
                        f"{source_path.name} is not a native ftmwpipeline HDF5 "
                        f"file (missing root '{VERSION_ATTR}' attribute)"
                    )
                version = int(h5f.attrs[VERSION_ATTR])
                if version > SUPPORTED_VERSION:
                    raise LoaderError(
                        f"Unsupported {VERSION_ATTR}={version} in "
                        f"{source_path.name} (this build understands up to "
                        f"{SUPPORTED_VERSION})"
                    )
                if "fid" not in h5f:
                    raise LoaderError(
                        f"Missing required '/fid' dataset in {source_path.name}"
                    )
                fid_dataset = h5f["fid"]
                if np.issubdtype(fid_dataset.dtype, np.complexfloating):
                    raise LoaderError(
                        f"'/fid' in {source_path.name} is complex; the pipeline "
                        "FT is a real FFT -- store real voltage samples"
                    )
                data = np.asarray(fid_dataset[...], dtype=np.float64).ravel()
                embedded = self._read_embedded(h5f)
        except LoaderError:
            raise
        except Exception as exc:
            raise LoaderError(f"Failed to read {source_path.name}: {exc}") from exc

        sidecar: Optional[Dict[str, Any]] = None
        found = find_sidecar(source_path, sidecar_path)
        if found is not None:
            sidecar = load_sidecar(found)

        explicit = explicit_layer_from_kwargs(kwargs)
        resolved = resolve_input_metadata(
            explicit=explicit, sidecar=sidecar, embedded=embedded
        )

        metadata = build_fid_metadata(resolved)
        metadata.update(
            self._create_source_metadata(
                source_path, metadata=str(found) if found else None
            )
        )

        return FID(
            data=data,
            spacing=resolved.spacing_s,
            probe_freq_mhz=resolved.probe_freq_mhz,
            sideband=resolved.sideband,
            shots=resolved.shots,
            processing=FIDProcessingParameters(),
            metadata=metadata,
        )

    def get_required_parameters(self) -> List[str]:
        return []

    def get_optional_parameters(self) -> Dict[str, Any]:
        return {
            "metadata": None,
            "spacing_us": None,
            "probe_freq_mhz": None,
            "sideband": None,
            "shots": None,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_embedded(h5f: "h5py.File") -> Dict[str, Any]:
        """Read the embedded acquisition layer (attrs + clock-source group)."""
        attrs = h5f.attrs
        embedded: Dict[str, Any] = {}
        for key in ("spacing_us", "probe_freq_mhz", "sideband", "shots"):
            if key in attrs:
                embedded[key] = _decode(attrs[key])

        chirp_end = attrs.get("chirp_end_us")
        if chirp_end is not None:
            chirp_window: Dict[str, Any] = {"chirp_end_us": float(chirp_end)}
            if attrs.get("chirp_start_us") is not None:
                chirp_window["chirp_start_us"] = float(attrs["chirp_start_us"])
            if attrs.get("start_margin_us") is not None:
                chirp_window["start_margin_us"] = float(attrs["start_margin_us"])
            embedded["chirp_window"] = chirp_window

        if "clock_sources" in h5f:
            embedded["clock_sources"] = _read_clock_group(h5f["clock_sources"])

        return embedded


def _read_clock_group(group: "h5py.Group") -> List[Dict[str, Any]]:
    """Read the optional ``/clock_sources`` parallel-dataset group."""
    if "freq_mhz" not in group:
        raise LoaderError("/clock_sources group is missing the 'freq_mhz' dataset")
    freqs = np.asarray(group["freq_mhz"][...], dtype=np.float64).ravel()
    k = len(freqs)
    if "locked" in group:
        locked = np.asarray(group["locked"][...]).ravel()
    else:
        locked = np.ones(k, dtype=np.int8)
    labels: List[str]
    if "label" in group:
        labels = [_decode(v) for v in np.asarray(group["label"][...]).ravel().tolist()]
    else:
        labels = [""] * k
    if not (len(locked) == k and len(labels) == k):
        raise LoaderError(
            "/clock_sources datasets freq_mhz/locked/label must be equal length"
        )
    return [
        {
            "freq_mhz": float(freqs[i]),
            "locked": bool(locked[i]),
            "label": str(labels[i]),
        }
        for i in range(k)
    ]
