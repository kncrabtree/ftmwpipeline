"""
Utilities for CLI parameter parsing and validation.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def parse_frequency_range(range_str: str) -> Tuple[float, float]:
    """
    Parse frequency range string in format 'min:max'.

    Parameters
    ----------
    range_str : str
        Frequency range as 'min:max' in MHz

    Returns
    -------
    tuple of float
        (min_freq, max_freq) in MHz

    Raises
    ------
    ValueError
        If range_str format is invalid
    """
    try:
        min_freq, max_freq = range_str.split(":")
        min_freq = float(min_freq)
        max_freq = float(max_freq)

        if min_freq >= max_freq:
            raise ValueError("Minimum frequency must be less than maximum frequency")

        return min_freq, max_freq
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"Invalid frequency range format '{range_str}'. Expected 'min:max' in MHz"
        ) from e


def validate_experiment_source(source_path: str) -> Path:
    """
    Validate experiment source path.

    Parameters
    ----------
    source_path : str
        Path to experiment data directory

    Returns
    -------
    Path
        Validated path object

    Raises
    ------
    FileNotFoundError
        If source path does not exist
    """
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment source path does not exist: {source_path}")
    if not path.is_dir():
        raise NotADirectoryError(
            f"Experiment source path is not a directory: {source_path}"
        )
    return path


def validate_cache_dir(cache_dir: str) -> Path:
    """
    Validate and create cache directory if needed.

    Parameters
    ----------
    cache_dir : str
        Path to cache directory

    Returns
    -------
    Path
        Validated cache directory path
    """
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def print_error(message: str, exit_code: int = 1) -> None:
    """Print error message and exit."""
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(exit_code)


def print_processing_params(
    zpf: int, expf_us: float, trim_range: Optional[Tuple[float, float]]
) -> None:
    """Print processing parameters for user confirmation."""
    print("Processing parameters:")
    print(f"  Zero padding factor: {zpf}")
    print(f"  Exponential filter: {expf_us} μs")
    if trim_range:
        print(f"  Frequency range: {trim_range[0]:.1f}-{trim_range[1]:.1f} MHz")
    else:
        print("  Frequency range: Full spectrum")
