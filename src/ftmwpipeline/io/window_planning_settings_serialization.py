"""
Persistence for :class:`~ftmwpipeline.core.window_planning_settings.WindowPlanningSettings`.

The persisted record for a Stage 4 run's resolved knobs lives under
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

from typing import Optional

from ..core.window_planning_settings import (
    WindowPlanningSettings,
)
from ..core.window_planning_settings import from_attrs as window_from_attrs
from ..core.window_planning_settings import to_attrs as window_to_attrs
from ._settings_serialization import (
    load_subblock_settings,
    save_settings,
    settings_block_present,
)

STAGE4_WINDOWS_SETTINGS_PATH = "processing_parameters/stage4_windows"

_SUB_NAMES = ("coherence", "clustering", "contributor", "leakage")


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
    save_settings(
        file_path,
        STAGE4_WINDOWS_SETTINGS_PATH,
        window_to_attrs(settings),
        preset_name=preset_name,
    )


def load_window_planning_settings_from_h5(
    file_path: str,
) -> Optional[WindowPlanningSettings]:
    """Return the persisted :class:`WindowPlanningSettings`, or ``None`` if absent.

    Tolerates missing sub-blocks (a partial group still loads); fields not
    present default to ``None``.
    """
    return load_subblock_settings(
        file_path, STAGE4_WINDOWS_SETTINGS_PATH, _SUB_NAMES, window_from_attrs
    )


def window_planning_settings_present(file_path: str) -> bool:
    """Lightweight: does the file have a persisted ``stage4_windows`` settings block?"""
    return settings_block_present(file_path, STAGE4_WINDOWS_SETTINGS_PATH)


__all__ = [
    "STAGE4_WINDOWS_SETTINGS_PATH",
    "save_window_planning_settings_to_h5",
    "load_window_planning_settings_from_h5",
    "window_planning_settings_present",
]
