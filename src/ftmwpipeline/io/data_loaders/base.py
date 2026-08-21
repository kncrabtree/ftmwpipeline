"""
Base loader interface for FTMW data formats.

This module defines the abstract base class that all data format loaders must
implement. It provides a consistent interface for loading FID data from various
experimental formats with proper metadata preservation.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Union

if TYPE_CHECKING:
    from ...core.data_structures import FID


class LoaderError(Exception):
    """Exception raised when data loading fails."""

    pass


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
