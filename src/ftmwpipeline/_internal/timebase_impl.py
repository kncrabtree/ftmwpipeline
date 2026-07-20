"""Shared implementation for scope-timebase self-calibration.

Wraps :func:`calibrate_timebase_from_fid` so the three user-facing interfaces
(CLI, Pipeline class, functional API) share one orchestration layer. The
calibration runs on the raw FID, reads its active region from the persisted
Stage 1 settings, resolves the instrument clock declaration, and persists the
measured timebase scale error ``eps`` to ``/timebase_calibration``.

The calibration depends only on Stage 0 (the raw FID) plus the persisted
Stage 1 active-region bounds. It measures and persists ``eps`` only; applying
the correction to the frequency axis is out of scope here.

Clock-declaration resolution lives in one small function
(:func:`_resolve_clock_sources`): explicit argument > persisted Stage 5
``spur.clocks`` > the loader's import-time recommendation.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

from ..core.settings import FT_PROCESSING_PATH, FTSettings
from ..core.stage_fit_settings import ClockSource, coerce_clock_sources
from ..file_manager import StageDependencyError
from ..fitting.timebase_calibration import (
    DEFAULT_KAPPA_SYS,
    DEFAULT_SNR_MIN,
    TimebaseCalibrationResult,
    calibrate_timebase_from_fid,
)
from ..io.stage_fit_settings_serialization import (
    load_stage_fit_settings_from_h5,
    read_recommended_clock_sources,
)
from ..io.timebase_serialization import GROUP_PATH as TIMEBASE_GROUP_PATH
from ..io.timebase_serialization import (
    load_timebase_calibration_from_hdf5,
    save_timebase_calibration_to_hdf5,
)
from .stage0_impl import load_fid_from_pipeline_impl
from .stage1_impl import _read_settings_layer
from .stage2_impl import _update_stage_completion

logger = logging.getLogger(__name__)

STAGE_NAME = "timebase_calibration"


def _read_persisted_ft_settings(file_path: str) -> FTSettings:
    """Resolve the persisted Stage 1 FT settings the calibration consumes."""
    settings = _read_settings_layer(file_path, FT_PROCESSING_PATH)
    if settings is None:
        raise StageDependencyError(
            STAGE_NAME,
            ["stage1_complex_ft"],
            Path(file_path),
        )
    return settings


def _resolve_clock_sources(
    file_path: str,
    clocks: Optional[Any],
) -> Tuple[ClockSource, ...]:
    """Resolve the instrument clock declaration for the calibration.

    Resolution order: explicit ``clocks`` argument (coerced) > persisted
    Stage 5 ``spur.clocks`` > the import-time recommended declaration the
    loader extracted from the instrument metadata (mirroring the Stage 5
    settings chain's explicit > persisted > recommended precedence). If no
    layer yields a non-empty declaration, raise ``ValueError`` -- the
    calibration cannot build the Rb-locked lattice without declared locked
    clocks and must not silently fall back.
    """
    if clocks is not None:
        coerced = coerce_clock_sources(clocks)
        if coerced:
            return coerced

    persisted = load_stage_fit_settings_from_h5(file_path)
    if persisted is not None and persisted.spur.clocks:
        return tuple(persisted.spur.clocks)

    recommended = read_recommended_clock_sources(file_path)
    if recommended:
        return recommended

    raise ValueError(
        "No instrument clock declaration found. Timebase self-calibration "
        "needs the Rb-locked clock fundamentals to build the spur lattice. "
        "Declare them via spur.clocks (e.g. StageFitSettings with "
        "spur.clocks=[ClockSource(freq_mhz=5120, locked=True), ...]) or pass "
        "clocks=... to this call."
    )


def _split_locked_unlocked(
    clocks: Tuple[ClockSource, ...],
) -> Tuple[List[float], List[float]]:
    """Split a clock declaration into locked / unlocked fundamental lists."""
    locked = [float(c.freq_mhz) for c in clocks if c.locked]
    unlocked = [float(c.freq_mhz) for c in clocks if not c.locked]
    return locked, unlocked


def calibrate_timebase_impl(
    file_path: str,
    *,
    clocks: Optional[Any] = None,
    kappa_sys: Optional[float] = None,
    snr_min: Optional[float] = None,
) -> Dict[str, Any]:
    """Run scope-timebase self-calibration and persist the result.

    Requires Stage 0 (the raw FID); the persisted Stage 1 settings supply the
    active-region bounds. Resolves the instrument clock declaration via
    :func:`_resolve_clock_sources` (explicit ``clocks`` > persisted
    ``spur.clocks``); raises ``ValueError`` if neither yields a non-empty
    declaration.

    Returns
    -------
    dict
        ``{"timebase_calibration": TimebaseCalibrationResult,
        "parameters_used": dict, "status": "success"}``.
    """
    file_path_obj = Path(file_path)
    with h5py.File(file_path, "r") as h5f:
        if "stage0_fid_data" not in h5f:
            raise StageDependencyError(
                STAGE_NAME,
                ["stage0_fid_data"],
                file_path_obj,
            )

    resolved_clocks = _resolve_clock_sources(file_path, clocks)
    locked, unlocked = _split_locked_unlocked(resolved_clocks)
    if not locked:
        raise ValueError(
            "The clock declaration has no locked sources. Timebase "
            "self-calibration needs at least one Rb-locked clock fundamental "
            "(ClockSource(..., locked=True)) to build the spur lattice."
        )

    ft_settings = _read_persisted_ft_settings(file_path)
    fid = load_fid_from_pipeline_impl(file_path)
    sample_dt_us = float(fid.spacing * 1e6)

    start_us = float(ft_settings.start_us) if ft_settings.start_us is not None else 0.0
    end_us = (
        float(ft_settings.end_us)
        if ft_settings.end_us is not None
        else float(fid.duration_us)
    )

    kappa_v = float(kappa_sys) if kappa_sys is not None else DEFAULT_KAPPA_SYS
    snr_min_v = float(snr_min) if snr_min is not None else DEFAULT_SNR_MIN

    result: TimebaseCalibrationResult = calibrate_timebase_from_fid(
        np.asarray(fid.data, dtype=float),
        sample_dt_us,
        start_us=start_us,
        end_us=end_us,
        locked_freqs_mhz=locked,
        unlocked_freqs_mhz=unlocked,
        kappa_sys=kappa_v,
        snr_min=snr_min_v,
    )

    parameters_used: Dict[str, Any] = {
        "kappa_sys": kappa_v,
        "snr_min": snr_min_v,
        "start_us": start_us,
        "end_us": end_us,
        "locked_freqs_mhz": locked,
        "unlocked_freqs_mhz": unlocked,
        "lattice_g_mhz": result.lattice_g_mhz,
    }
    save_timebase_calibration_impl(file_path, result, parameters_used=parameters_used)
    _update_stage_completion(file_path, STAGE_NAME)

    return {
        "status": "success",
        "timebase_calibration": result,
        "parameters_used": parameters_used,
    }


def save_timebase_calibration_impl(
    file_path: str,
    result: TimebaseCalibrationResult,
    *,
    parameters_used: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a :class:`TimebaseCalibrationResult` into ``/timebase_calibration``."""
    with h5py.File(file_path, "a") as h5f:
        if TIMEBASE_GROUP_PATH in h5f:
            del h5f[TIMEBASE_GROUP_PATH]
        grp = h5f.create_group(TIMEBASE_GROUP_PATH)
        save_timebase_calibration_to_hdf5(result, grp)
        grp.attrs["creation_time"] = datetime.now().isoformat()
        if parameters_used is not None:
            grp.attrs["parameters_used"] = json.dumps(parameters_used, default=str)


def load_timebase_calibration_impl(file_path: str) -> Dict[str, Any]:
    """Load the persisted :class:`TimebaseCalibrationResult` from ``file_path``.

    Raises
    ------
    ValueError
        If the timebase calibration has not been completed.
    """
    with h5py.File(file_path, "r") as h5f:
        if TIMEBASE_GROUP_PATH not in h5f:
            raise ValueError(
                "Timebase calibration has not been completed for "
                f"{file_path}. Run calibrate_timebase() / 'timebase run' first."
            )
        grp = h5f[TIMEBASE_GROUP_PATH]
        result = load_timebase_calibration_from_hdf5(grp)
        creation_time = grp.attrs.get("creation_time", "unknown")
        parameters_used: Dict[str, Any] = {}
        if "parameters_used" in grp.attrs:
            try:
                parameters_used = json.loads(grp.attrs["parameters_used"])
            except (json.JSONDecodeError, TypeError):
                logger.warning("Could not parse saved timebase-calibration parameters")
    return {
        "timebase_calibration": result,
        "creation_time": creation_time,
        "parameters_used": parameters_used,
    }


def timebase_calibration_present(file_path: str) -> bool:
    """Lightweight: does the ``.ftmw`` file have a persisted timebase result?"""
    try:
        with h5py.File(file_path, "r") as h5f:
            return TIMEBASE_GROUP_PATH in h5f
    except (OSError, KeyError):
        return False
