"""
Persistence for :class:`~ftmwpipeline.core.peak_detection_settings.PeakDetectionSettings`.

The canonical record for a Stage 3 run's resolved knobs lives under
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

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import h5py

from ..core.peak_detection_settings import (
    PeakDetectionSettings,
    from_attrs as peak_from_attrs,
    to_attrs as peak_to_attrs,
)

logger = logging.getLogger(__name__)

STAGE3_PEAKS_SETTINGS_PATH = "processing_parameters/stage3_peaks"

_SUB_NAMES = ("promotion", "savgol", "primary_pass", "gap_pass")


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _read_sub_attrs(grp: h5py.Group) -> Dict[str, Any]:
    return {key: _decode_attr(raw) for key, raw in grp.attrs.items()}


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
    attrs = peak_to_attrs(settings)
    with h5py.File(file_path, "a") as h5f:
        if STAGE3_PEAKS_SETTINGS_PATH in h5f:
            del h5f[STAGE3_PEAKS_SETTINGS_PATH]
        grp = h5f.create_group(STAGE3_PEAKS_SETTINGS_PATH)
        grp.attrs["creation_time"] = datetime.now().isoformat()
        if preset_name is not None:
            grp.attrs["preset_name"] = preset_name
        for sub_name in _SUB_NAMES:
            sub_grp = grp.create_group(sub_name)
            for field_name, value in attrs[sub_name].items():
                sub_grp.attrs[field_name] = value


def load_peak_detection_settings_from_h5(
    file_path: str,
) -> Optional[PeakDetectionSettings]:
    """Return the persisted :class:`PeakDetectionSettings`, or ``None`` if absent.

    Tolerates missing sub-blocks (a partial group still loads); fields not
    present default to ``None``.
    """
    with h5py.File(file_path, "r") as h5f:
        if STAGE3_PEAKS_SETTINGS_PATH not in h5f:
            return None
        grp = h5f[STAGE3_PEAKS_SETTINGS_PATH]
        attrs_dict: Dict[str, Any] = {}
        for sub_name in _SUB_NAMES:
            if sub_name in grp and isinstance(grp[sub_name], h5py.Group):
                attrs_dict[sub_name] = _read_sub_attrs(grp[sub_name])
            else:
                attrs_dict[sub_name] = {}
    return peak_from_attrs(attrs_dict)


def peak_detection_settings_present(file_path: str) -> bool:
    """Lightweight: does the file have a persisted ``stage3_peaks`` settings block?"""
    try:
        with h5py.File(file_path, "r") as h5f:
            return STAGE3_PEAKS_SETTINGS_PATH in h5f
    except (OSError, KeyError):
        return False


__all__ = [
    "STAGE3_PEAKS_SETTINGS_PATH",
    "save_peak_detection_settings_to_h5",
    "load_peak_detection_settings_from_h5",
    "peak_detection_settings_present",
]
