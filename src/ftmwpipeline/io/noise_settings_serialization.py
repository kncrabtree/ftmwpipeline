"""
Persistence for :class:`~ftmwpipeline.core.noise_settings.NoiseSettings`.

The canonical record for a Stage 2 run's resolved knobs lives under
``processing_parameters/stage2_noise``. Stage 2 has a single estimator, so the
fields are stored as attrs directly on the group (no sub-block layer),
inspectable with ``h5dump -p``:

.. code-block::

    processing_parameters/
      stage2_noise/
        @creation_time
        @preset_name              (optional audit attr)
        @window_mhz
        @pedestal_mhz
        ...

Unset (Optional-None) fields encode as the ``__None__`` sentinel string,
matching :mod:`ftmwpipeline.io.stage_fit_settings_serialization` and
:mod:`ftmwpipeline.io.tau_calibration_settings_serialization`.

The path is intentionally distinct from the existing ``/stage2_noise_result``
group (the noise-estimate payload — the σ_x array + bin diagnostics).
Settings (knobs) live here under ``processing_parameters/``; results live
at the root. Same pattern Stage 5 uses
(``processing_parameters/stage5_fit`` vs ``/stage5_fitting``).
"""

from __future__ import annotations

from typing import Optional

from ..core.noise_settings import (
    NoiseSettings,
)
from ..core.noise_settings import from_attrs as noise_from_attrs
from ..core.noise_settings import to_attrs as noise_to_attrs
from ._settings_serialization import (
    load_flat_settings,
    save_settings,
    settings_block_present,
)

STAGE2_NOISE_SETTINGS_PATH = "processing_parameters/stage2_noise"

# Group-level bookkeeping attrs that are not NoiseSettings fields.
_AUDIT_ATTRS = ("creation_time", "preset_name")


def save_noise_settings_to_h5(
    file_path: str,
    settings: NoiseSettings,
    *,
    preset_name: Optional[str] = None,
) -> None:
    """Persist a resolved :class:`NoiseSettings` to ``processing_parameters/stage2_noise``.

    The settings fields are stored as attrs directly on the group (Stage 2 has
    a single estimator, so there is no sub-block layer). Overwrites any prior
    group at that path. ``preset_name`` (if given) is recorded as a top-level
    attr for audit/reproducibility.
    """
    save_settings(
        file_path,
        STAGE2_NOISE_SETTINGS_PATH,
        noise_to_attrs(settings),
        preset_name=preset_name,
    )


def load_noise_settings_from_h5(file_path: str) -> Optional[NoiseSettings]:
    """Return the persisted :class:`NoiseSettings`, or ``None`` if absent.

    Fields not present default to ``None``; the audit attrs are skipped.
    """
    return load_flat_settings(
        file_path,
        STAGE2_NOISE_SETTINGS_PATH,
        noise_from_attrs,
        audit_attrs=_AUDIT_ATTRS,
    )


def noise_settings_present(file_path: str) -> bool:
    """Lightweight: does the file have a persisted ``stage2_noise`` block?"""
    return settings_block_present(file_path, STAGE2_NOISE_SETTINGS_PATH)


__all__ = [
    "STAGE2_NOISE_SETTINGS_PATH",
    "save_noise_settings_to_h5",
    "load_noise_settings_from_h5",
    "noise_settings_present",
]
