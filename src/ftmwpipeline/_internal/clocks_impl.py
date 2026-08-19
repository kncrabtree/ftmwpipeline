"""Shared implementation for the ``clocks`` declaration surface.

Declaring instrument clock sources on an already-imported ``.ftmw`` is the
no-code path for any data that did not embed clocks (a CSV, a native-HDF5 file
written without a ``/clock_sources`` group) or for revising a declaration after
import.  All three interfaces (CLI ``clocks``, ``api.set_clock_sources`` /
``Pipeline.set_clock_sources``) delegate here.

The declaration is written to the **recommended** layer
(``stage0_fid_data@recommended_clock_sources``) -- the same layer a loader-
injected declaration uses.  The Stage 5 resolver still ranks it below an
explicit ``--clocks`` and below persisted Stage 5 settings, so editing the
declaration never silently rewrites a reproducible fit (the D11 guarantee).
"""

import logging
from typing import Any, List, Optional, Sequence, Tuple

import h5py

from ..core.stage_fit_settings import ClockSource, coerce_clock_sources
from ..file_manager import open_pipeline_file
from ..io.stage_fit_settings_serialization import (
    read_recommended_clock_sources,
    write_recommended_clock_sources,
)

logger = logging.getLogger(__name__)

# Frequency match tolerance for ``remove`` (MHz).  Matches the 6-decimal
# rounding that ``ClockSource.to_dict`` persists.
#
# Legitimately absolute (dev-docs/SCIENCE_STRATEGY.md Requirement 8): this is
# a float-comparison tolerance on a user-declared clock frequency (a value a
# person typed when declaring an instrument clock source), not a spectral
# distance on any FT grid -- it owes nothing to the active-FT bin spacing and
# must not be converted to a bin-relative quantity.
_FREQ_TOL_MHZ = 1e-6


def _validate_pipeline(file_path: str) -> None:
    """Raise if ``file_path`` is not a readable pipeline file with Stage 0 data."""
    open_pipeline_file(file_path)  # raises PipelineFileError on a bad file


def get_clock_sources_impl(file_path: str) -> Optional[Tuple[ClockSource, ...]]:
    """Return the declared (recommended) clock sources, or ``None`` if unset."""
    _validate_pipeline(file_path)
    return read_recommended_clock_sources(file_path)


def stage5_fit_present(file_path: str) -> bool:
    """Return ``True`` if a Stage 5 fit exists (so a clock edit post-dates it)."""
    try:
        with h5py.File(file_path, "r") as h5f:
            return "stage5_fitting" in h5f
    except OSError:
        return False


def set_clock_sources_impl(
    file_path: str,
    clocks: Any,
    *,
    replace: bool = True,
) -> Tuple[ClockSource, ...]:
    """Declare clock sources, replacing or appending to the current declaration.

    ``clocks`` is any clocks-like value :func:`coerce_clock_sources` accepts
    (a sequence of :class:`ClockSource` or ``{freq_mhz, locked, label}`` dicts).
    With ``replace=False`` the new sources are appended to the existing
    declaration.  Returns the resulting declaration.
    """
    _validate_pipeline(file_path)
    new = coerce_clock_sources(clocks) or ()
    if replace:
        result = tuple(new)
    else:
        existing = read_recommended_clock_sources(file_path) or ()
        result = tuple(existing) + tuple(new)
    write_recommended_clock_sources(file_path, result if result else None)
    logger.info("Declared %d clock source(s) on %s", len(result), file_path)
    return result


def remove_clock_sources_impl(
    file_path: str,
    freqs_mhz: Sequence[float],
) -> Tuple[ClockSource, ...]:
    """Remove declared clock sources whose frequency matches any of ``freqs_mhz``.

    Frequencies are matched within ``_FREQ_TOL_MHZ``.  Returns the remaining
    declaration; a frequency that matches nothing is reported via the log.
    """
    _validate_pipeline(file_path)
    existing = read_recommended_clock_sources(file_path) or ()
    targets = [float(f) for f in freqs_mhz]
    kept: List[ClockSource] = []
    matched = {t: False for t in targets}
    for clock in existing:
        hit = next(
            (t for t in targets if abs(clock.freq_mhz - t) <= _FREQ_TOL_MHZ), None
        )
        if hit is None:
            kept.append(clock)
        else:
            matched[hit] = True
    for t, was_matched in matched.items():
        if not was_matched:
            logger.warning("No declared clock source near %.6f MHz to remove", t)
    result = tuple(kept)
    write_recommended_clock_sources(file_path, result if result else None)
    return result


def clear_clock_sources_impl(file_path: str) -> None:
    """Clear the clock-source declaration (writes the no-recommendation sentinel)."""
    _validate_pipeline(file_path)
    write_recommended_clock_sources(file_path, None)
    logger.info("Cleared clock-source declaration on %s", file_path)
