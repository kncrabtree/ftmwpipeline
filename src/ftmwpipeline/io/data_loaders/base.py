"""
Base loader interface for FTMW data formats.

This module defines the abstract base class that all data format loaders must
implement. It provides a consistent interface for loading FID data from various
experimental formats with proper metadata preservation.
"""

import math
import numbers
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

from ...contract import Absent, FidPreviewRow

if TYPE_CHECKING:
    from ...core.data_structures import FID


class LoaderError(Exception):
    """Exception raised when data loading fails."""

    pass


class LoadParameterError(LoaderError):
    """A load parameter the loader cannot honour.

    Raised for one named parameter (``parameter``), with what would be accepted
    (``expected``) and the value given (``value``). The import path reports it
    as ``bad_setting`` with ``path`` the parameter name (not ``"source"``).
    """

    def __init__(self, parameter: str, expected: str, value: Any, message: str) -> None:
        self.parameter = parameter
        self.expected = expected
        self.value = value
        super().__init__(message)


#: The explicit chirp-window load parameters every loader accepts (µs).
CHIRP_PARAMETERS: Tuple[str, ...] = (
    "chirp_start_us",
    "chirp_end_us",
    "start_margin_us",
)

#: The chirp parameters the derived start (``chirp_end_us + start_margin_us``)
#: is computed from. ``chirp_start_us`` is provenance only and never feeds it.
CHIRP_START_PARAMETERS: Tuple[str, ...] = ("chirp_end_us", "start_margin_us")

#: ``fid.metadata`` transport key: ``True`` when the attached ``chirp_window``
#: carries an explicitly passed value the derived start is computed from (one of
#: :data:`CHIRP_START_PARAMETERS`). Underscore-prefixed, so it is
#: in-memory provenance only and never persisted as metadata (the explicit
#: values themselves are persisted in the source's ``loader_parameters``).
CHIRP_WINDOW_EXPLICIT_KEY = "_chirp_window_explicit"


def resolve_chirp_window(params: Mapping[str, Any], declared: Any) -> Tuple[Any, bool]:
    """Merge the explicit chirp parameters over the window the source declares.

    With no explicit (non-``None``) chirp parameter in ``params``, ``declared``
    is returned unchanged (malformed or not -- persisting it is advisory and
    reports its own failure). Otherwise, field by field, an explicit
    ``chirp_start_us`` / ``chirp_end_us`` / ``start_margin_us`` replaces the
    declared value, and the declared fields not given explicitly are kept when
    they are finite numbers (a malformed declared field is dropped, so it
    cannot sink the explicit window).

    Returns the window (``None`` when neither declares one) and whether an
    explicit value feeds the derived start (``chirp_end_us`` or
    ``start_margin_us``, :data:`CHIRP_START_PARAMETERS`); an explicit
    ``chirp_start_us`` alone replaces the window's start field but does not
    reorder start precedence.

    Raises
    ------
    LoadParameterError
        For an explicit value that is not a finite number, or for an explicit
        ``chirp_start_us`` / ``start_margin_us`` when no chirp end is known
        from any layer (there is nothing to attach it to).
    """
    explicit: Dict[str, float] = {}
    for key in CHIRP_PARAMETERS:
        raw = params.get(key)
        if raw is None:
            continue
        value = chirp_float(raw)
        if not isinstance(value, float):
            raise LoadParameterError(
                key,
                "a finite number of microseconds",
                raw,
                f"{key} must be a finite number of microseconds; got {raw!r}",
            )
        explicit[key] = value
    if not explicit:
        return declared, False

    # A declared block that is not a mapping cannot contribute a field.
    base: Mapping[str, Any] = declared if isinstance(declared, Mapping) else {}
    merged: Dict[str, Any] = {}
    for key in CHIRP_PARAMETERS:
        value = chirp_float(base.get(key))
        if isinstance(value, float):
            merged[key] = value
    merged.update(explicit)
    if merged.get("chirp_end_us") is None:
        name = "start_margin_us" if "start_margin_us" in explicit else "chirp_start_us"
        raise LoadParameterError(
            name,
            "a chirp_end_us to attach to (from the call, a sidecar or the source)",
            params.get(name),
            f"{name} cannot be honoured: no chirp end is declared by the source "
            "and none was given; pass chirp_end_us as well",
        )
    return merged, any(key in explicit for key in CHIRP_START_PARAMETERS)


def has_chirp_end(block: Any) -> bool:
    """Whether a declared ``chirp_window`` block holds a finite ``chirp_end_us``."""
    return isinstance(block, Mapping) and isinstance(
        chirp_float(block.get("chirp_end_us")), float
    )


@dataclass(frozen=True)
class SourcePreview:
    """What a source holds, found without importing (see ``preview_source``).

    ``chirp_window`` is the declared window (``chirp_end_us`` and, when
    declared, ``chirp_start_us`` / ``start_margin_us``) or ``None``.
    """

    format: str
    fids: List[FidPreviewRow]
    chirp_window: Union[None, Dict[str, Any], Absent] = None


def chirp_float(value: Any) -> Union[float, Absent]:
    """A declared chirp time: a finite real number, else ``UNDEFINED``.

    Booleans and strings are not numbers here; ``None`` is ``NOT_RUN``.
    """
    if value is None:
        return Absent.NOT_RUN
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        return Absent.UNDEFINED
    out = float(value)
    return out if math.isfinite(out) else Absent.UNDEFINED


def validated_chirp_window(block: Any) -> Union[Dict[str, Any], Absent]:
    """Validate a declared ``chirp_window`` block read from a source.

    A block that is not a mapping is ``UNDEFINED`` (declared, unreadable); in a
    mapping each of the three chirp times is a finite float or ``UNDEFINED``.
    A declared window without ``chirp_end_us`` is ``UNDEFINED`` as a whole.
    """
    if not isinstance(block, dict) or block.get("chirp_end_us") is None:
        return Absent.UNDEFINED
    out: Dict[str, Any] = {}
    for key in ("chirp_start_us", "chirp_end_us", "start_margin_us"):
        value = chirp_float(block.get(key))
        if value is not Absent.NOT_RUN:
            out[key] = value
    return out


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

    def preview_fids(self, source_path: Union[str, Path]) -> List[FidPreviewRow]:
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
            FidPreviewRow(
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
    ) -> Union[None, Dict[str, Any], Absent]:
        """The chirp window the source declares, or ``None`` when it has none.

        A dict with ``chirp_end_us`` and, when declared, ``chirp_start_us`` and
        ``start_margin_us`` (µs; each a finite float or ``Absent.UNDEFINED``).
        ``Absent.UNDEFINED`` means the source declares a window that cannot be
        read. Reads only; never imports.
        """
        return None

    def accepted_parameters(self) -> Tuple[str, ...]:
        """Every load parameter this loader accepts, in declaration order.

        The one declaration is :meth:`get_required_parameters` plus the keys of
        :meth:`get_optional_parameters`; the import path refuses any other
        parameter passed for this format.
        """
        names = list(self.get_required_parameters())
        names += [n for n in self.get_optional_parameters() if n not in names]
        return tuple(names)

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
