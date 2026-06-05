"""
Format registry for automatic format detection and loader selection.

This module provides a centralized registry for all data format loaders,
enabling automatic format detection and appropriate loader selection.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

from .base import BaseLoader, LoaderError

if TYPE_CHECKING:
    from ...core.data_structures import FID

logger = logging.getLogger(__name__)


class FormatRegistry:
    """
    Central registry for data format loaders.

    This class manages the available loaders and provides format detection
    and loader selection functionality.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._loaders: Dict[str, Type[BaseLoader]] = {}
        self._loader_instances: Dict[str, BaseLoader] = {}

    def register_loader(self, format_name: str, loader_class: Type[BaseLoader]) -> None:
        """
        Register a new loader class.

        Parameters
        ----------
        format_name : str
            Unique name for the format
        loader_class : Type[BaseLoader]
            Loader class that inherits from BaseLoader

        Raises
        ------
        ValueError
            If format_name already exists or loader_class is invalid
        """
        if not isinstance(format_name, str) or not format_name:
            raise ValueError("format_name must be a non-empty string")

        if not issubclass(loader_class, BaseLoader):
            raise ValueError("loader_class must inherit from BaseLoader")

        if format_name in self._loaders:
            logger.warning(f"Overriding existing loader for format '{format_name}'")

        self._loaders[format_name] = loader_class

        # Create instance for use
        try:
            self._loader_instances[format_name] = loader_class()
            logger.debug(
                f"Registered loader for format '{format_name}': {loader_class.__name__}"
            )
        except Exception as e:
            logger.error(
                f"Failed to instantiate loader for format '{format_name}': {e}"
            )
            # Remove from registry if instantiation fails
            if format_name in self._loaders:
                del self._loaders[format_name]
            raise LoaderError(
                f"Failed to register loader for format '{format_name}': {e}"
            ) from e

    def get_loader(self, format_name: str) -> BaseLoader:
        """
        Get loader instance for a specific format.

        Parameters
        ----------
        format_name : str
            Name of the format

        Returns
        -------
        BaseLoader
            Loader instance for the format

        Raises
        ------
        ValueError
            If format is not registered
        """
        if format_name not in self._loader_instances:
            available = list(self._loaders.keys())
            raise ValueError(
                f"Unknown format '{format_name}'. Available formats: {available}"
            )

        return self._loader_instances[format_name]

    def detect_format(self, source_path: Union[str, Path]) -> Optional[str]:
        """
        Automatically detect the format of the source.

        Parameters
        ----------
        source_path : str or Path
            Path to data source

        Returns
        -------
        Optional[str]
            Detected format name, or None if no format can handle the source
        """
        source_path = Path(source_path)

        if not source_path.exists():
            logger.warning(f"Source path does not exist: {source_path}")
            return None

        # Try each loader's can_load method
        for format_name, loader in self._loader_instances.items():
            try:
                if loader.can_load(source_path):
                    logger.debug(f"Format '{format_name}' can load: {source_path}")
                    return format_name
            except Exception as e:
                logger.debug(f"Format '{format_name}' failed can_load check: {e}")
                continue

        logger.warning(f"No loader found for source: {source_path}")
        return None

    def validate_source(
        self, source_path: Union[str, Path], format_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate a source with automatic or specified format detection.

        Parameters
        ----------
        source_path : str or Path
            Path to data source
        format_name : str, optional
            Specific format to use, or None for auto-detection

        Returns
        -------
        Dict[str, Any]
            Validation result dictionary containing:
            - 'format': detected or specified format name
            - 'valid': bool indicating if source is valid
            - 'metadata': dict with available metadata
            - 'options': dict with loading options
            - 'errors': list of validation errors
        """
        result = {
            "format": format_name,
            "valid": False,
            "metadata": {},
            "options": {},
            "errors": [],
        }

        # Auto-detect format if not specified
        if format_name is None:
            format_name = self.detect_format(source_path)
            result["format"] = format_name

            if format_name is None:
                result["errors"].append("Could not detect data format")
                return result

        # Validate with specific loader
        try:
            loader = self.get_loader(format_name)
            validation_result = loader.validate_source(source_path)

            result.update(validation_result)
            return result

        except ValueError as e:
            result["errors"].append(str(e))
            return result
        except Exception as e:
            result["errors"].append(f"Validation failed: {e}")
            return result

    def load_fid(
        self, source_path: Union[str, Path], format_name: Optional[str] = None, **kwargs
    ) -> "FID":
        """
        Load FID data with automatic or specified format detection.

        Parameters
        ----------
        source_path : str or Path
            Path to data source
        format_name : str, optional
            Specific format to use, or None for auto-detection
        **kwargs
            Format-specific loading parameters

        Returns
        -------
        FID
            Loaded FID object

        Raises
        ------
        LoaderError
            If format detection or loading fails
        """
        # Auto-detect format if not specified
        if format_name is None:
            format_name = self.detect_format(source_path)
            if format_name is None:
                raise LoaderError(f"Could not detect data format for: {source_path}")

        # Load with specific format
        try:
            loader = self.get_loader(format_name)
            return loader.load_fid(source_path, **kwargs)
        except ValueError as e:
            raise LoaderError(str(e)) from e
        except Exception as e:
            raise LoaderError(
                f"Failed to load data with format '{format_name}': {e}"
            ) from e

    def list_formats(self) -> List[str]:
        """
        Get list of registered format names.

        Returns
        -------
        List[str]
            List of registered format names
        """
        return list(self._loaders.keys())

    def get_format_info(self, format_name: str) -> Dict[str, Any]:
        """
        Get information about a registered format.

        Parameters
        ----------
        format_name : str
            Name of the format

        Returns
        -------
        Dict[str, Any]
            Dictionary with format information:
            - 'name': format name
            - 'loader_class': loader class name
            - 'file_extensions': supported file extensions
            - 'directory_indicators': directory indicators
            - 'required_parameters': required loading parameters
            - 'optional_parameters': optional loading parameters
        """
        if format_name not in self._loader_instances:
            raise ValueError(f"Unknown format '{format_name}'")

        loader = self._loader_instances[format_name]

        return {
            "name": format_name,
            "loader_class": loader.__class__.__name__,
            "file_extensions": getattr(loader, "file_extensions", []),
            "directory_indicators": getattr(loader, "directory_indicators", []),
            "required_parameters": loader.get_required_parameters(),
            "optional_parameters": loader.get_optional_parameters(),
        }

    def __repr__(self) -> str:
        """String representation of registry."""
        formats = list(self._loaders.keys())
        return f"FormatRegistry(formats={formats})"


# Global registry instance
_global_registry = FormatRegistry()


def register_loader(format_name: str, loader_class: Type[BaseLoader]) -> None:
    """
    Register a loader with the global registry.

    Parameters
    ----------
    format_name : str
        Unique name for the format
    loader_class : Type[BaseLoader]
        Loader class that inherits from BaseLoader
    """
    _global_registry.register_loader(format_name, loader_class)


def get_loader(format_name: str) -> BaseLoader:
    """Get loader from global registry."""
    return _global_registry.get_loader(format_name)


def detect_format(source_path: Union[str, Path]) -> Optional[str]:
    """Detect format using global registry."""
    return _global_registry.detect_format(source_path)


def validate_source(
    source_path: Union[str, Path], format_name: Optional[str] = None
) -> Dict[str, Any]:
    """Validate source using global registry."""
    return _global_registry.validate_source(source_path, format_name)


def load_fid(
    source_path: Union[str, Path], format_name: Optional[str] = None, **kwargs
) -> "FID":
    """Load FID using global registry."""
    return _global_registry.load_fid(source_path, format_name, **kwargs)


def list_formats() -> List[str]:
    """List formats from global registry."""
    return _global_registry.list_formats()


def get_format_info(format_name: str) -> Dict[str, Any]:
    """Get format info from global registry."""
    return _global_registry.get_format_info(format_name)
