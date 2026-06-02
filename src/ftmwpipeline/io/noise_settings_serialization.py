"""
Persistence for :class:`~ftmwpipeline.core.noise_settings.NoiseSettings`.

The canonical record for a Stage 2 run's resolved knobs lives under
``processing_parameters/stage2_noise``. The layout uses one HDF5 subgroup
per sub-dataclass so each block is independently inspectable with
``h5dump -p``:

.. code-block::

    processing_parameters/
      stage2_noise/
        @creation_time
        @preset_name              (optional audit attr)
        binning/
          @subdivision_threshold
          @abs_min_bin_size
          ...
        skewness/  smoothing/  skirt_exclusion/

Unset (Optional-None) fields encode as the ``__None__`` sentinel string,
matching :mod:`ftmwpipeline.io.stage_fit_settings_serialization` and
:mod:`ftmwpipeline.io.tau_calibration_settings_serialization`.

The path is intentionally distinct from the existing ``/stage2_noise_result``
group (the noise-estimate payload — the σ_x array + bin diagnostics).
Settings (knobs) live here under ``processing_parameters/``; results live
at the root. Same pattern Stage 5 uses
(``processing_parameters/stage5_fit`` vs ``/stage5_fitting``).

The legacy ``processing_parameters/noise_estimation`` group (carrying the
four user-tunable kwargs the older Stage 2 impl persisted to drive
``from_saved_params=True``) is unrelated to this module; it stays in place
as a back-compat shim and is owned by ``_internal/stage2_impl``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import h5py

from ..core.noise_settings import (
    NoiseSettings,
    _SUB_NAMES,
    from_attrs as noise_from_attrs,
    to_attrs as noise_to_attrs,
)

logger = logging.getLogger(__name__)

STAGE2_NOISE_SETTINGS_PATH = "processing_parameters/stage2_noise"


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _read_sub_attrs(grp: h5py.Group) -> Dict[str, Any]:
    return {key: _decode_attr(raw) for key, raw in grp.attrs.items()}


def save_noise_settings_to_h5(
    file_path: str,
    settings: NoiseSettings,
    *,
    preset_name: Optional[str] = None,
) -> None:
    """Persist a resolved :class:`NoiseSettings` to ``processing_parameters/stage2_noise``.

    Overwrites any prior group at that path. ``preset_name`` (if given) is
    recorded as a top-level attr for audit/reproducibility.
    """
    attrs = noise_to_attrs(settings)
    with h5py.File(file_path, "a") as h5f:
        if STAGE2_NOISE_SETTINGS_PATH in h5f:
            del h5f[STAGE2_NOISE_SETTINGS_PATH]
        grp = h5f.create_group(STAGE2_NOISE_SETTINGS_PATH)
        grp.attrs["creation_time"] = datetime.now().isoformat()
        if preset_name is not None:
            grp.attrs["preset_name"] = preset_name
        for sub_name in _SUB_NAMES:
            sub_grp = grp.create_group(sub_name)
            for field_name, value in attrs[sub_name].items():
                sub_grp.attrs[field_name] = value


def load_noise_settings_from_h5(file_path: str) -> Optional[NoiseSettings]:
    """Return the persisted :class:`NoiseSettings`, or ``None`` if absent.

    Tolerates missing sub-blocks (a partial group still loads); fields not
    present default to ``None``.
    """
    with h5py.File(file_path, "r") as h5f:
        if STAGE2_NOISE_SETTINGS_PATH not in h5f:
            return None
        grp = h5f[STAGE2_NOISE_SETTINGS_PATH]
        attrs_dict: Dict[str, Any] = {}
        for sub_name in _SUB_NAMES:
            if sub_name in grp and isinstance(grp[sub_name], h5py.Group):
                attrs_dict[sub_name] = _read_sub_attrs(grp[sub_name])
            else:
                attrs_dict[sub_name] = {}
    return noise_from_attrs(attrs_dict)


def noise_settings_present(file_path: str) -> bool:
    """Lightweight: does the file have a persisted ``stage2_noise`` block?"""
    try:
        with h5py.File(file_path, "r") as h5f:
            return STAGE2_NOISE_SETTINGS_PATH in h5f
    except (OSError, KeyError):
        return False


__all__ = [
    "STAGE2_NOISE_SETTINGS_PATH",
    "save_noise_settings_to_h5",
    "load_noise_settings_from_h5",
    "noise_settings_present",
]
