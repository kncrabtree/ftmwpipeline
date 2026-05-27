"""
Persistence for :class:`~ftmwpipeline.core.tau_calibration_settings.TauCalibrationSettings`.

The canonical record for a Stage 2b run's resolved knobs lives under
``processing_parameters/stage2b_tau``. The layout uses one HDF5 subgroup
per sub-dataclass so each block is independently inspectable with
``h5dump -p``:

.. code-block::

    processing_parameters/
      stage2b_tau/
        @creation_time
        @preset_name              (optional audit attr)
        stft/
          @n_seg
          @t_sigma
          ...
        polish/  aggregation/  band/  gaussian/  recommendation/

Unset (Optional-None) fields encode as the ``__None__`` sentinel string,
matching :mod:`ftmwpipeline.io.stage_fit_settings_serialization`.
Tuple-valued fields (``tau_G_seeds``, ``band_edges_mhz``,
``band_labels``) round-trip via 1-D HDF5 datasets stored as attrs.

The Lorentzian and Gaussian τ twins share this one persisted settings
record -- they are alternative outputs of the same algorithm under
different shape selections; the shape itself lives on
:class:`~ftmwpipeline.core.stage_fit_settings.StageFitSettings` (Stage 5).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import h5py

from ..core.tau_calibration_settings import (
    TauCalibrationSettings,
    from_attrs as tau_settings_from_attrs,
    to_attrs as tau_settings_to_attrs,
)

logger = logging.getLogger(__name__)

STAGE2B_TAU_SETTINGS_PATH = "processing_parameters/stage2b_tau"

_SUB_NAMES = (
    "stft",
    "polish",
    "aggregation",
    "band",
    "gaussian",
    "recommendation",
)

# Tuple-valued attrs encoded as Python lists; HDF5 stores them as 1-D
# numpy arrays under the hood. The serialization module flattens these
# back to plain tuples on load via ``from_attrs``.
_TUPLE_FIELDS = {"tau_G_seeds", "band_edges_mhz", "band_labels"}

_NONE_SENTINEL = "__None__"


def _decode_attr(value: Any) -> Any:
    """Decode an HDF5 attribute value (handles bytes -> str, ndarray -> list)."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _write_attr(grp: h5py.Group, name: str, value: Any) -> None:
    """Write one attr, lifting lists/tuples to numpy 1-D arrays."""
    if isinstance(value, list):
        if all(isinstance(v, str) for v in value):
            grp.attrs[name] = list(value)
        else:
            grp.attrs[name] = list(value)
    else:
        grp.attrs[name] = value


def _read_sub_attrs(grp: h5py.Group) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, raw in grp.attrs.items():
        value = _decode_attr(raw)
        if key in _TUPLE_FIELDS:
            # HDF5 may return numpy arrays for sequence attrs; pass them
            # through as iterables so the dataclass decoder can coerce to
            # tuples.
            if isinstance(value, str) and value == _NONE_SENTINEL:
                out[key] = _NONE_SENTINEL
                continue
            out[key] = [_decode_attr(v) for v in value]
        else:
            out[key] = value
    return out


def save_tau_calibration_settings_to_h5(
    file_path: str,
    settings: TauCalibrationSettings,
    *,
    preset_name: Optional[str] = None,
) -> None:
    """Persist a resolved :class:`TauCalibrationSettings` to ``processing_parameters/stage2b_tau``.

    Overwrites any prior group at that path. ``preset_name`` (if given) is
    recorded as a top-level attr for audit/reproducibility -- useful when
    a calibration was driven by a named preset.
    """
    attrs = tau_settings_to_attrs(settings)
    with h5py.File(file_path, "a") as h5f:
        if STAGE2B_TAU_SETTINGS_PATH in h5f:
            del h5f[STAGE2B_TAU_SETTINGS_PATH]
        grp = h5f.create_group(STAGE2B_TAU_SETTINGS_PATH)
        grp.attrs["creation_time"] = datetime.now().isoformat()
        if preset_name is not None:
            grp.attrs["preset_name"] = preset_name
        for sub_name in _SUB_NAMES:
            sub_grp = grp.create_group(sub_name)
            for field_name, value in attrs[sub_name].items():
                _write_attr(sub_grp, field_name, value)


def load_tau_calibration_settings_from_h5(
    file_path: str,
) -> Optional[TauCalibrationSettings]:
    """Return the persisted :class:`TauCalibrationSettings`, or ``None`` if absent.

    Tolerates missing sub-blocks (a partial group still loads); fields
    not present default to ``None``.
    """
    with h5py.File(file_path, "r") as h5f:
        if STAGE2B_TAU_SETTINGS_PATH not in h5f:
            return None
        grp = h5f[STAGE2B_TAU_SETTINGS_PATH]
        attrs_dict: Dict[str, Any] = {}
        for sub_name in _SUB_NAMES:
            if sub_name in grp and isinstance(grp[sub_name], h5py.Group):
                attrs_dict[sub_name] = _read_sub_attrs(grp[sub_name])
            else:
                attrs_dict[sub_name] = {}
    return tau_settings_from_attrs(attrs_dict)


def tau_calibration_settings_present(file_path: str) -> bool:
    """Lightweight: does the file have a persisted ``stage2b_tau`` block?"""
    try:
        with h5py.File(file_path, "r") as h5f:
            return STAGE2B_TAU_SETTINGS_PATH in h5f
    except (OSError, KeyError):
        return False


__all__ = [
    "STAGE2B_TAU_SETTINGS_PATH",
    "save_tau_calibration_settings_to_h5",
    "load_tau_calibration_settings_from_h5",
    "tau_calibration_settings_present",
]
