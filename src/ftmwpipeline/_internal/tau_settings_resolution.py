"""Shared resolver glue for the Stage 2b user-facing impls.

The three Stage 2b orchestrators (``calibrate_tau_impl``,
``calibrate_tau_G_impl``, ``recommend_shape_impl``) all build an explicit
:class:`TauCalibrationSettings` from their legacy per-knob kwargs and
walk the four-layer resolution chain (``explicit > preset > persisted >
recommended > hard default``). The shared scaffolding lives here so a
single resolver-shape change updates every consumer.

The recommended layer is currently always ``None`` for Stage 2b -- the
stage is the originator of the L/G recommendation, not a consumer. The
slot stays for parity with Stage 5's resolver shape; a future Stage-2
σ-driven recommender for ``snr_min`` will land in this layer without
API churn.
"""

from __future__ import annotations

from typing import Optional, Tuple

from ..core.tau_calibration_settings import (
    TauCalibrationSettings,
)
from ..core.tau_calibration_settings import load_preset as load_tau_preset
from ..core.tau_calibration_settings import resolve as resolve_tau_settings
from ..io.tau_calibration_settings_serialization import (
    load_tau_calibration_settings_from_h5,
)


def _required_int(value: Optional[int], name: str) -> int:
    """Coerce a post-resolve field that must be filled into ``int``."""
    if value is None:
        raise AssertionError(
            f"resolved TauCalibrationSettings.{name} is None; missing hard default"
        )
    return int(value)


def _required_float(value: Optional[float], name: str) -> float:
    """Coerce a post-resolve field that must be filled into ``float``."""
    if value is None:
        raise AssertionError(
            f"resolved TauCalibrationSettings.{name} is None; missing hard default"
        )
    return float(value)


def _required_bool(value: Optional[bool], name: str) -> bool:
    """Coerce a post-resolve field that must be filled into ``bool``."""
    if value is None:
        raise AssertionError(
            f"resolved TauCalibrationSettings.{name} is None; missing hard default"
        )
    return bool(value)


def resolve_with_preset_and_persisted(
    file_path: str,
    *,
    explicit: TauCalibrationSettings,
    settings: Optional[TauCalibrationSettings],
    preset: Optional[str],
) -> Tuple[TauCalibrationSettings, Optional[str]]:
    """Walk the Stage 2b resolution chain and return ``(resolved, preset_name)``.

    ``settings`` and ``preset`` are mutually exclusive (the rule mirrors
    :func:`ftmwpipeline._internal.stage5_impl.fit_peaks_impl`). When
    ``preset`` is a bare name or YAML path, it is loaded via
    :func:`ftmwpipeline.core.tau_calibration_settings.load_preset` and
    its name is returned for the audit attr on the persisted settings
    record. ``persisted`` comes from
    :func:`load_tau_calibration_settings_from_h5`; the *recommended*
    layer is reserved (always ``None`` today).
    """
    if preset is not None and settings is not None:
        raise ValueError(
            "'preset' and 'settings' are alternative ways to populate "
            "the preset layer of the tau-calibration settings chain; "
            "pass exactly one (or override individual fields via "
            "explicit kwargs)"
        )
    preset_layer = settings
    preset_name: Optional[str] = None
    if preset is not None:
        preset_layer = load_tau_preset(preset)
        preset_name = str(preset)
    persisted = load_tau_calibration_settings_from_h5(file_path)
    resolved = resolve_tau_settings(
        explicit=explicit,
        preset=preset_layer,
        persisted=persisted,
        recommended=None,
    )
    return resolved, preset_name


__all__ = [
    "_required_int",
    "_required_float",
    "_required_bool",
    "resolve_with_preset_and_persisted",
]
