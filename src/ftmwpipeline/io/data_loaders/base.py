"""
Base loader interface for FTMW data formats.

This module defines the abstract base class that all data format loaders must
implement. It provides a consistent interface for loading FID data from various
experimental formats with proper metadata preservation.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from ...contract import Absent

if TYPE_CHECKING:
    from ...core.data_structures import FID


class LoaderError(Exception):
    """Exception raised when data loading fails."""

    pass


@dataclass(frozen=True)
class FidInfo:
    """One row of a source's FID table (see :meth:`BaseLoader.preview_fids`).

    Every field except ``index`` is a value or an
    :class:`~ftmwpipeline.contract.Absent`: ``NOT_RUN`` when the source does not
    declare it, ``UNDEFINED`` when it cannot be stated without load parameters
    (or is not finite).
    """

    index: int
    n_points: Union[int, Absent] = Absent.NOT_RUN
    spacing_us: Union[float, Absent] = Absent.NOT_RUN
    probe_freq_mhz: Union[float, Absent] = Absent.NOT_RUN
    sideband: Union[str, Absent] = Absent.NOT_RUN  # "upper" | "lower"
    shots: Union[int, Absent] = Absent.NOT_RUN


@dataclass(frozen=True)
class SourcePreview:
    """What a source holds, found without importing (see ``preview_source``).

    ``chirp_window`` is the declared window (``chirp_end_us`` and, when
    declared, ``chirp_start_us`` / ``start_margin_us``) or ``None``.
    """

    format: str
    fids: List[FidInfo]
    chirp_window: Optional[Dict[str, Any]] = None


def finite_or_absent(value: Any) -> Union[float, Absent]:
    """``float(value)`` when finite; ``NOT_RUN`` for ``None``; else ``UNDEFINED``."""
    if value is None:
        return Absent.NOT_RUN
    try:
        out = float(value)
    except (TypeError, ValueError):
        return Absent.UNDEFINED
    return out if math.isfinite(out) else Absent.UNDEFINED


def count_or_absent(value: Any) -> Union[int, Absent]:
    """``int(value)`` when it is finite; see :func:`finite_or_absent`."""
    number = finite_or_absent(value)
    return number if isinstance(number, Absent) else int(number)


def sideband_or_absent(value: Any) -> Union[str, Absent]:
    """``"upper"``/``"lower"`` from a name, ``NOT_RUN`` for ``None``, else ``UNDEFINED``."""
    if value is None:
        return Absent.NOT_RUN
    text = str(value).strip().lower()
    return text if text in ("upper", "lower") else Absent.UNDEFINED


class BaseLoader(ABC):
    """
    Abstract base class for all data format loaders.

    This class defines the interface that all specific format loaders must
    implement. It handles the common aspects of data loading while allowing
    format-specific implementations to handle their unique requirements.
    """

    # Class attributes that subclasses should define
    format_name: str = "unknown"
    file_extensions: List[str] = []
    directory_indicators: List[str] = []  # Files/dirs that indicate this format

    def __init__(self) -> None:
        """Initialize base loader."""
        pass

    @abstractmethod
    def can_load(self, source_path: Union[str, Path]) -> bool:
        """
        Check if this loader can handle the given source path.

        Parameters
        ----------
        source_path : str or Path
            Path to data source (file or directory)

        Returns
        -------
        bool
            True if this loader can handle the source
        """
        pass

    @abstractmethod
    def validate_source(
        self, source_path: Union[str, Path], **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Validate the source and return available metadata/options.

        Parameters
        ----------
        source_path : str or Path
            Path to data source
        **kwargs
            Format-specific validation parameters

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'valid': bool indicating if source is valid
            - 'metadata': dict with available metadata
            - 'options': dict with loading options (e.g., available FID indices)
            - 'errors': list of validation errors if any
        """
        pass

    @abstractmethod
    def load_fid(self, source_path: Union[str, Path], **kwargs: Any) -> "FID":
        """
        Load FID data from the source.

        Parameters
        ----------
        source_path : str or Path
            Path to data source
        **kwargs
            Format-specific loading parameters

        Returns
        -------
        FID
            Loaded FID object with all metadata

        Raises
        ------
        LoaderError
            If loading fails for any reason
        """
        pass

    def preview_fids(self, source_path: Union[str, Path]) -> List[FidInfo]:
        """
        Describe every FID the source holds, without importing anything.

        The default is the single-FID description built from
        :meth:`validate_source` metadata (``n_points``, ``spacing_us``,
        ``probe_freq_mhz``, ``sideband``, ``shots``); a field the loader does
        not report is ``Absent.NOT_RUN``. Loaders override it to say more. A
        loader whose source can hold several FIDs must return a row for each.

        Raises
        ------
        LoaderError
            If the source is not valid for this loader.
        """
        validation = self.validate_source(source_path)
        if not validation.get("valid"):
            raise LoaderError(
                f"Invalid {self.format_name} source: {validation.get('errors')}"
            )
        meta = validation.get("metadata", {})
        return [
            FidInfo(
                index=0,
                n_points=count_or_absent(meta.get("n_points")),
                spacing_us=finite_or_absent(meta.get("spacing_us")),
                probe_freq_mhz=finite_or_absent(meta.get("probe_freq_mhz")),
                sideband=sideband_or_absent(meta.get("sideband")),
                shots=count_or_absent(meta.get("shots")),
            )
        ]

    def preview_chirp_window(
        self, source_path: Union[str, Path]
    ) -> Optional[Dict[str, Any]]:
        """The chirp window the source declares, or ``None`` when it has none.

        A dict with ``chirp_end_us`` and, when declared, ``chirp_start_us`` and
        ``start_margin_us`` (µs). Reads only; never imports.
        """
        return None

    def get_required_parameters(self) -> List[str]:
        """
        Get list of parameters required for this loader.

        Returns
        -------
        List[str]
            List of parameter names that must be provided to load_fid()
        """
        return []

    def get_optional_parameters(self) -> Dict[str, Any]:
        """
        Get dictionary of optional parameters with their defaults.

        Returns
        -------
        Dict[str, Any]
            Dictionary mapping parameter names to default values
        """
        return {}

    def validate_parameters(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Validate loading parameters and provide defaults.

        Parameters
        ----------
        **kwargs
            Parameters to validate

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'valid': bool indicating if parameters are valid
            - 'parameters': dict with validated/defaulted parameters
            - 'errors': list of parameter errors if any

        Raises
        ------
        LoaderError
            If required parameters are missing or invalid
        """
        result: Dict[str, Any] = {"valid": True, "parameters": {}, "errors": []}

        # Check required parameters
        required = self.get_required_parameters()
        for param in required:
            if param not in kwargs:
                result["errors"].append(f"Required parameter '{param}' not provided")
                result["valid"] = False
            else:
                result["parameters"][param] = kwargs[param]

        # Add optional parameters with defaults
        optional = self.get_optional_parameters()
        for param, default_value in optional.items():
            result["parameters"][param] = kwargs.get(param, default_value)

        # Add any extra parameters
        for param, value in kwargs.items():
            if param not in result["parameters"]:
                result["parameters"][param] = value

        if not result["valid"]:
            raise LoaderError(
                f"Parameter validation failed: {'; '.join(result['errors'])}"
            )

        return result

    def _create_source_metadata(
        self, source_path: Union[str, Path], **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Create source metadata for FID object.

        Parameters
        ----------
        source_path : str or Path
            Path to data source
        **kwargs
            Additional metadata

        Returns
        -------
        Dict[str, Any]
            Source metadata dictionary
        """
        from datetime import datetime

        source_path = Path(source_path)

        metadata: Dict[str, Any] = {
            "source_path": str(source_path.resolve()),
            "source_format": self.format_name,
            "loader_class": self.__class__.__name__,
            "load_timestamp": datetime.now().isoformat(),
            "loader_parameters": kwargs.copy(),
        }

        # Add file/directory information
        if source_path.exists():
            stat = source_path.stat()
            metadata["source_size"] = stat.st_size if source_path.is_file() else None
            metadata["source_modified"] = datetime.fromtimestamp(
                stat.st_mtime
            ).isoformat()

        return metadata

    def __repr__(self) -> str:
        """String representation of loader."""
        extensions = ", ".join(self.file_extensions) if self.file_extensions else "N/A"
        return f"{self.__class__.__name__}(format='{self.format_name}', extensions=[{extensions}])"
