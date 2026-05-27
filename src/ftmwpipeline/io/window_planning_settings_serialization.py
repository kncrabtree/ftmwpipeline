"""
Persistence for :class:`~ftmwpipeline.core.window_planning_settings.WindowPlanningSettings`.

The canonical record for a Stage 4 run's resolved knobs lives under
``processing_parameters/stage4_windows``. The layout uses one HDF5
subgroup per sub-dataclass so each block is independently inspectable
with ``h5dump -p``:

.. code-block::

    processing_parameters/
      stage4_windows/
        @creation_time
        @preset_name              (optional audit attr)
        coherence/
          @edge_m
          @trim_m
          @edge_threshold
        clustering/  contributor/  leakage/

Unset (Optional-None) fields encode as the ``__None__`` sentinel string,
matching :mod:`ftmwpipeline.io.stage_fit_settings_serialization`,
:mod:`ftmwpipeline.io.tau_calibration_settings_serialization`,
:mod:`ftmwpipeline.io.noise_settings_serialization`, and
:mod:`ftmwpipeline.io.peak_detection_settings_serialization`.

The path is intentionally distinct from the existing root-level
``/stage4_windows`` group (the resolved :class:`WindowPlan` -- windows,
fixed contributors, dependency DAG, batches). Settings (knobs) live here
under ``processing_parameters/``; results live at the root. Same pattern
Stages 5, 2, and 3 use.

The legacy ``processing_parameters/window_assignment`` group (carrying
the JSON-encoded ``parameters_used`` dict the older Stage 4 impl
persisted) is unrelated to this module; it stays in place as a
back-compat shim and is owned by
``_internal/stage4_impl.save_window_parameters_impl``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import h5py

from ..core.window_planning_settings import (
    WindowPlanningSettings,
    from_attrs as window_from_attrs,
    to_attrs as window_to_attrs,
)

logger = logging.getLogger(__name__)

STAGE4_WINDOWS_SETTINGS_PATH = "processing_parameters/stage4_windows"

_SUB_NAMES = ("coherence", "clustering", "contributor", "leakage")


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _read_sub_attrs(grp: h5py.Group) -> Dict[str, Any]:
    return {key: _decode_attr(raw) for key, raw in grp.attrs.items()}


def save_window_planning_settings_to_h5(
    file_path: str,
    settings: WindowPlanningSettings,
    *,
    preset_name: Optional[str] = None,
) -> None:
    """Persist a resolved :class:`WindowPlanningSettings` to ``processing_parameters/stage4_windows``.

    Overwrites any prior group at that path. ``preset_name`` (if given) is
    recorded as a top-level attr for audit/reproducibility.
    """
    attrs = window_to_attrs(settings)
    with h5py.File(file_path, "a") as h5f:
        if STAGE4_WINDOWS_SETTINGS_PATH in h5f:
            del h5f[STAGE4_WINDOWS_SETTINGS_PATH]
        grp = h5f.create_group(STAGE4_WINDOWS_SETTINGS_PATH)
        grp.attrs["creation_time"] = datetime.now().isoformat()
        if preset_name is not None:
            grp.attrs["preset_name"] = preset_name
        for sub_name in _SUB_NAMES:
            sub_grp = grp.create_group(sub_name)
            for field_name, value in attrs[sub_name].items():
                sub_grp.attrs[field_name] = value


def load_window_planning_settings_from_h5(
    file_path: str,
) -> Optional[WindowPlanningSettings]:
    """Return the persisted :class:`WindowPlanningSettings`, or ``None`` if absent.

    Tolerates missing sub-blocks (a partial group still loads); fields not
    present default to ``None``.
    """
    with h5py.File(file_path, "r") as h5f:
        if STAGE4_WINDOWS_SETTINGS_PATH not in h5f:
            return None
        grp = h5f[STAGE4_WINDOWS_SETTINGS_PATH]
        attrs_dict: Dict[str, Any] = {}
        for sub_name in _SUB_NAMES:
            if sub_name in grp and isinstance(grp[sub_name], h5py.Group):
                attrs_dict[sub_name] = _read_sub_attrs(grp[sub_name])
            else:
                attrs_dict[sub_name] = {}
    return window_from_attrs(attrs_dict)


def window_planning_settings_present(file_path: str) -> bool:
    """Lightweight: does the file have a persisted ``stage4_windows`` settings block?"""
    try:
        with h5py.File(file_path, "r") as h5f:
            return STAGE4_WINDOWS_SETTINGS_PATH in h5f
    except (OSError, KeyError):
        return False


__all__ = [
    "STAGE4_WINDOWS_SETTINGS_PATH",
    "save_window_planning_settings_to_h5",
    "load_window_planning_settings_from_h5",
    "window_planning_settings_present",
]
