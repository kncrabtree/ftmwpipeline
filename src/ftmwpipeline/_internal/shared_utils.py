"""
Shared utilities for internal implementations.

Common functionality used across multiple pipeline stages and interfaces.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def require_resolved(
    value: Any,
    name: str,
    *,
    cast: Optional[Callable[[Any], Any]] = None,
    owner: str,
) -> Any:
    """Assert a post-resolve settings field is filled, optionally casting it.

    Every stage impl reads its knobs from a *resolved* settings bundle whose
    fields should all be non-``None`` (the resolver fills any gap from the
    hard defaults). A ``None`` here means a hard default is missing -- a
    programming error, not user input -- so this raises ``AssertionError``
    naming ``{owner}.{name}``. When ``cast`` is given the returned value is
    coerced through it (``int``/``float``/``bool``/``str``); otherwise the
    value passes through unchanged.
    """
    if value is None:
        raise AssertionError(f"resolved {owner}.{name} is None; missing hard default")
    return cast(value) if cast is not None else value


def parse_frequency_range(range_str: str) -> Tuple[float, float]:
    """
    Parse frequency range string in format "min:max".

    Parameters
    ----------
    range_str : str
        Frequency range as "min:max" in MHz

    Returns
    -------
    tuple of float
        (min_freq, max_freq) in MHz

    Raises
    ------
    ValueError
        If range string format is invalid
    """
    try:
        parts = range_str.split(":")
        if len(parts) != 2:
            raise ValueError("Range must be in format 'min:max'")

        min_freq = float(parts[0])
        max_freq = float(parts[1])

        if min_freq >= max_freq:
            raise ValueError("Minimum frequency must be less than maximum frequency")

        if min_freq < 0 or max_freq < 0:
            raise ValueError("Frequencies must be non-negative")

        return (min_freq, max_freq)

    except ValueError as e:
        if "could not convert" in str(e):
            raise ValueError(
                f"Invalid frequency values in range '{range_str}': must be numbers"
            )
        raise


def validate_file_path(file_path: str, must_exist: bool = True) -> Path:
    """
    Validate and normalize file path.

    Parameters
    ----------
    file_path : str
        Path to validate
    must_exist : bool, default True
        If True, file must exist

    Returns
    -------
    Path
        Validated Path object

    Raises
    ------
    FileNotFoundError
        If must_exist=True and file doesn't exist
    ValueError
        If path is invalid
    """
    if not file_path:
        raise ValueError("File path cannot be empty")

    path = Path(file_path)

    if must_exist and not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")

    return path


def setup_logging_for_internal(verbose: bool = False) -> None:
    """
    Setup logging for internal implementations.

    Parameters
    ----------
    verbose : bool, default False
        Enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def format_processing_summary(params: Dict[str, Any]) -> str:
    """
    Format processing parameters for display.

    Parameters
    ----------
    params : dict
        Processing parameters

    Returns
    -------
    str
        Formatted parameter summary
    """
    lines = ["Processing parameters:"]
    for param, value in params.items():
        formatted_value = "None (default)" if value is None else str(value)
        lines.append(f"  {param}: {formatted_value}")

    return "\n".join(lines)


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Parameters
    ----------
    size_bytes : int
        Size in bytes

    Returns
    -------
    str
        Formatted size string
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.1f} MB"
    else:
        return f"{size_bytes / 1024**3:.1f} GB"


def extract_experiment_id(file_path: str) -> str:
    """
    Extract experiment ID from pipeline file path.

    Parameters
    ----------
    file_path : str
        Path to .ftmw pipeline file

    Returns
    -------
    str
        Experiment ID (filename without extension)
    """
    return Path(file_path).stem


def ensure_ftmw_extension(file_path: str) -> str:
    """
    Ensure file path has .ftmw extension.

    Parameters
    ----------
    file_path : str
        File path

    Returns
    -------
    str
        File path with .ftmw extension
    """
    path = Path(file_path)
    if path.suffix.lower() != ".ftmw":
        return str(path.with_suffix(".ftmw"))
    return file_path


def create_result_summary(
    operation: str, file_path: str, **metadata: Any
) -> Dict[str, Any]:
    """
    Create standardized result summary for operations.

    Parameters
    ----------
    operation : str
        Operation name
    file_path : str
        Pipeline file path
    **metadata
        Additional metadata

    Returns
    -------
    dict
        Standardized result summary
    """
    result = {
        "operation": operation,
        "file_path": str(file_path),
        "experiment_id": extract_experiment_id(file_path),
        "timestamp": None,  # Could add datetime.now() if needed
        "status": "success",
    }
    result.update(metadata)
    return result
