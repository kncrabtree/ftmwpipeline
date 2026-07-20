"""
Persistence for :class:`~ftmwpipeline.core.peak_detection_settings.PeakDetectionSettings`.

The persisted record for a Stage 3 run's resolved knobs lives under
``processing_parameters/stage3_peaks``. The layout uses one HDF5 subgroup
per sub-dataclass so each block is independently inspectable with
``h5dump -p``:

.. code-block::

    processing_parameters/
      stage3_peaks/
        @creation_time
        @preset_name              (optional audit attr)
        promotion/
          @min_snr
          @internal_min_snr
          ...
        savgol/  primary_pass/  gap_pass/

Unset (Optional-None) fields encode as the ``__None__`` sentinel string,
matching :mod:`ftmwpipeline.io.stage_fit_settings_serialization`,
:mod:`ftmwpipeline.io.tau_calibration_settings_serialization`, and
:mod:`ftmwpipeline.io.noise_settings_serialization`.

The path is intentionally distinct from the existing root-level
``/stage3_peaks`` group (the detected peak list -- frequencies,
intensities, SNRs, classifications). Settings (knobs) live here under
``processing_parameters/``; results live at the root. Same pattern
Stages 5 and 2 use (``processing_parameters/stage5_fit`` vs
``/stage5_fitting``, ``processing_parameters/stage2_noise`` vs
``/stage2_noise_result``).

The legacy ``processing_parameters/peak_detection`` group (carrying the
JSON-encoded ``parameters_used`` dict the older Stage 3 impl persisted)
is unrelated to this module; it stays in place as a back-compat shim
and is owned by ``_internal/stage3_impl.save_peak_parameters_impl``.
"""

from __future__ import annotations

from typing import Optional

from ..core.peak_detection_settings import (
    PeakDetectionSettings,
)
from ..core.peak_detection_settings import from_attrs as peak_from_attrs
from ..core.peak_detection_settings import to_attrs as peak_to_attrs
from ._settings_serialization import (
    load_subblock_settings,
    save_settings,
    settings_block_present,
)

STAGE3_PEAKS_SETTINGS_PATH = "processing_parameters/stage3_peaks"

_SUB_NAMES = ("promotion", "savgol", "primary_pass", "gap_pass")


def save_peak_detection_settings_to_h5(
    file_path: str,
    settings: PeakDetectionSettings,
    *,
    preset_name: Optional[str] = None,
) -> None:
    """Persist a resolved :class:`PeakDetectionSettings` to ``processing_parameters/stage3_peaks``.

    Overwrites any prior group at that path. ``preset_name`` (if given) is
    recorded as a top-level attr for audit/reproducibility.
    """
    save_settings(
        file_path,
        STAGE3_PEAKS_SETTINGS_PATH,
        peak_to_attrs(settings),
        preset_name=preset_name,
    )


def load_peak_detection_settings_from_h5(
    file_path: str,
) -> Optional[PeakDetectionSettings]:
    """Return the persisted :class:`PeakDetectionSettings`, or ``None`` if absent.

    Tolerates missing sub-blocks (a partial group still loads); fields not
    present default to ``None``.
    """
    return load_subblock_settings(
        file_path, STAGE3_PEAKS_SETTINGS_PATH, _SUB_NAMES, peak_from_attrs
    )


def peak_detection_settings_present(file_path: str) -> bool:
    """Lightweight: does the file have a persisted ``stage3_peaks`` settings block?"""
    return settings_block_present(file_path, STAGE3_PEAKS_SETTINGS_PATH)


__all__ = [
    "STAGE3_PEAKS_SETTINGS_PATH",
    "save_peak_detection_settings_to_h5",
    "load_peak_detection_settings_from_h5",
    "peak_detection_settings_present",
]
