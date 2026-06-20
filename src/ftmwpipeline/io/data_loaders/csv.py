"""CSV data format loader.

Imports a single (averaged) FID from a column of time-domain voltage samples.
A CSV cannot embed acquisition metadata, so ``spacing_us`` and
``probe_freq_mhz`` must come from explicit load parameters or a ``--metadata``
sidecar (see :mod:`ftmwpipeline.io.input_metadata`).  Clock declarations -- which
a single column cannot carry -- come from the sidecar or the ``clocks`` command
on the imported ``.ftmw`` file.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

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


class CSVLoader(BaseLoader):
    """Loader for CSV files holding a column of voltage samples.

    The voltage column is selected with ``column`` -- an integer index (0-based)
    or, when the file has a header row, a column name.  Omitted, the first
    column is used; other columns are ignored.  Acquisition metadata is resolved
    from explicit parameters and an optional sidecar.
    """

    format_name = "csv"
    file_extensions = [".csv"]
    directory_indicators: List[str] = []

    def can_load(self, source_path: Union[str, Path]) -> bool:
        """Check the source is a readable ``.csv`` file (structure only)."""
        source_path = Path(source_path)
        if not source_path.is_file():
            return False
        return source_path.suffix.lower() == ".csv"

    def validate_source(
        self, source_path: Union[str, Path], **kwargs: Any
    ) -> Dict[str, Any]:
        """Validate file structure; acquisition metadata is checked at load time.

        ``validate_source`` runs before load parameters and the sidecar are
        known, so it confirms only that the file is a readable CSV with a
        numeric voltage column -- ``load_fid`` raises the targeted error if
        ``spacing_us`` / ``probe_freq_mhz`` are ultimately missing.
        """
        result: Dict[str, Any] = {
            "valid": False,
            "metadata": {},
            "options": {},
            "errors": [],
        }
        source_path = Path(source_path)

        if not self.can_load(source_path):
            result["errors"].append("Not a CSV file (expected a .csv extension)")
            return result

        try:
            frame = pd.read_csv(source_path)
        except Exception as exc:
            result["errors"].append(f"Failed to read CSV file: {exc}")
            return result

        if frame.shape[1] < 1 or len(frame) == 0:
            result["errors"].append("CSV file has no data rows")
            return result

        result["metadata"]["n_points"] = int(len(frame))
        result["metadata"]["n_columns"] = int(frame.shape[1])
        result["options"]["columns"] = [str(c) for c in frame.columns]
        result["valid"] = True
        return result

    def load_fid(self, source_path: Union[str, Path], **kwargs: Any) -> "FID":
        from ...core.data_structures import FID, FIDProcessingParameters

        source_path = Path(source_path)
        sidecar_path = kwargs.pop("metadata", None)
        column = kwargs.pop("column", None)

        try:
            frame = pd.read_csv(source_path)
        except Exception as exc:
            raise LoaderError(f"Failed to read {source_path.name}: {exc}") from exc

        series = self._select_column(frame, column, source_path.name)
        if not pd.api.types.is_numeric_dtype(series):
            raise LoaderError(
                f"Selected CSV column does not hold numeric voltage values "
                f"in {source_path.name}"
            )
        data = np.asarray(series.to_numpy(), dtype=np.float64).ravel()
        if data.size == 0:
            raise LoaderError(f"No samples in selected column of {source_path.name}")

        sidecar: Optional[Dict[str, Any]] = None
        found = find_sidecar(source_path, sidecar_path)
        if found is not None:
            sidecar = load_sidecar(found)

        explicit = explicit_layer_from_kwargs(kwargs)
        resolved = resolve_input_metadata(explicit=explicit, sidecar=sidecar)

        metadata = build_fid_metadata(resolved)
        metadata.update(
            self._create_source_metadata(
                source_path,
                column=column,
                metadata=str(found) if found else None,
            )
        )

        return FID(
            data=data,
            spacing=resolved.spacing_s,
            probe_freq_mhz=resolved.probe_freq_mhz,
            sideband=resolved.sideband,
            shots=resolved.shots,
            processing=FIDProcessingParameters(rdc=resolved.rdc),
            metadata=metadata,
        )

    def get_required_parameters(self) -> List[str]:
        # spacing_us / probe_freq_mhz are required, but may arrive via the
        # sidecar rather than as load parameters, so they are validated in
        # load_fid (the shared resolver) rather than declared required here.
        return []

    def get_optional_parameters(self) -> Dict[str, Any]:
        return {
            "column": None,
            "metadata": None,
            "spacing_us": None,
            "probe_freq_mhz": None,
            "sideband": None,
            "shots": None,
            "rdc": None,
        }

    @staticmethod
    def _select_column(frame: "pd.DataFrame", column: Any, name: str) -> "pd.Series":
        """Resolve the ``column`` selector (index, name, or default first)."""
        if column is None:
            return frame.iloc[:, 0]
        # Integer index (also accept an int-valued string like "2").
        if isinstance(column, int) or (
            isinstance(column, str) and column.lstrip("-").isdigit()
        ):
            idx = int(column)
            if not -frame.shape[1] <= idx < frame.shape[1]:
                raise LoaderError(
                    f"Column index {idx} out of range for {name} "
                    f"({frame.shape[1]} column(s))"
                )
            return frame.iloc[:, idx]
        if column in frame.columns:
            return frame[column]
        raise LoaderError(
            f"Column {column!r} not found in {name}; available: "
            f"{[str(c) for c in frame.columns]}"
        )
