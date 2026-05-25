"""Shared implementation for Stage 2b: data-driven tau calibration.

Wraps :func:`extract_tau_majority` so the three user-facing interfaces (CLI,
Pipeline class, functional API) share one orchestration layer. The
calibration runs on the raw FID (no apodization), reads its active region
and Stage 1 trim from the persisted Stage 1 settings, and persists the
result to ``/stage2b_tau_calibration``.

Stage 2b sits between Stage 2 (noise estimation) and Stage 3 (peak
detection): Stage 3's gap pass and Stage 5's tau anchor both consume
``tau_maj``. Re-running Stage 2b invalidates Stages 3-5 downstream.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np

from ..core.settings import FT_PROCESSING_PATH, FTSettings
from ..file_manager import (
    PipelineStageTracker,
    StageDependencyError,
    invalidate_downstream_stages,
)
from ..fitting.tau_calibration import (
    DEFAULT_N_SEG,
    DEFAULT_RSS_GATE_FACTOR,
    DEFAULT_T_SIGMA,
    TauCalibrationResult,
    extract_tau_majority,
)
from ..io.tau_calibration_serialization import (
    GROUP_PATH as TAU_GROUP_PATH,
    load_tau_calibration_from_hdf5,
    save_tau_calibration_to_hdf5,
)
from .stage0_impl import load_fid_from_pipeline_impl
from .stage1_impl import _read_settings_layer
from .stage2_impl import _update_stage_completion

logger = logging.getLogger(__name__)

STAGE_NAME = "stage2b_tau_calibration"


def _read_canonical_ft_settings(file_path: str) -> FTSettings:
    """Resolve the canonical Stage 1 FT settings the calibration consumes."""
    settings = _read_settings_layer(file_path, FT_PROCESSING_PATH)
    if settings is None:
        raise StageDependencyError(
            STAGE_NAME, ["stage1_complex_ft"], Path(file_path),
        )
    return settings


def calibrate_tau_impl(
    file_path: str,
    *,
    n_seg: Optional[int] = None,
    t_sigma: Optional[float] = None,
    tau_max_us: Optional[float] = None,
    rss_gate_factor: Optional[float] = None,
    sigma_time: Optional[float] = None,
    min_contributors: Optional[int] = None,
    sigma_tau_fraction_max: Optional[float] = None,
    bimodality_dominant_fraction: Optional[float] = None,
    compute_band_majorities: bool = False,
    min_contributors_per_band: Optional[int] = None,
) -> Dict[str, Any]:
    """Run STFT tau calibration and persist the result to ``file_path``.

    Requires Stages 0-2 to be completed (Stage 1 owns the active-region and
    trim parameters the calibration consumes; Stage 2's σ is not strictly
    needed but is the natural noise reference for downstream consistency).

    Parameters
    ----------
    file_path : str
        Path to the ``.ftmw`` pipeline file.
    n_seg, t_sigma, tau_max_us, rss_gate_factor : optional
        STFT knobs. Defaults match the Phase-1 acceptance gate.
    sigma_time : float, optional
        Time-domain σ_t reference. ``None`` (default) lets
        :func:`extract_tau_majority` measure it from the FID active-region
        tail.
    min_contributors, sigma_tau_fraction_max, bimodality_dominant_fraction
        Acceptance pre-conditions; calibrations that fail any pre-condition
        still persist (downstream consumers gate on ``preconditions_passed``).
    compute_band_majorities : bool, default False
        When True, also compute per-band SNR-weighted majority tau on an
        arithmetic three-band split of the trim range (low / mid / high)
        and persist as ``band_majorities``. Stage 5 may then consume these
        as per-window tau anchors via ``fit_peaks(per_band_tau=True)``.
    min_contributors_per_band : int, optional
        Threshold below which a band falls back to the band-wide majority.
        Defaults to 50 inside :func:`compute_band_majorities`.

    Returns
    -------
    dict
        ``{"tau_calibration": TauCalibrationResult, "parameters_used": dict,
        "status": "success"}``.
    """
    file_path_obj = Path(file_path)
    with h5py.File(file_path, "r") as h5f:
        if "stage2_noise_result" not in h5f:
            raise StageDependencyError(
                STAGE_NAME, ["stage2_noise_result"], file_path_obj,
            )

    settings = _read_canonical_ft_settings(file_path)
    fid = load_fid_from_pipeline_impl(file_path)
    sample_dt_us = float(fid.spacing * 1e6)

    start_us = (
        float(settings.start_us) if settings.start_us is not None else 0.0
    )
    end_us = (
        float(settings.end_us)
        if settings.end_us is not None
        else float(fid.duration_us)
    )
    if settings.trim is None:
        raise ValueError(
            "Stage 1 canonical FT settings have no frequency trim; tau "
            "calibration uses the persisted trim range to match the user "
            "spectrum. Set trim on compute_ft() first."
        )
    trim_lo_mhz, trim_hi_mhz = settings.trim

    n_seg_v = DEFAULT_N_SEG if n_seg is None else int(n_seg)
    t_sigma_v = DEFAULT_T_SIGMA if t_sigma is None else float(t_sigma)
    rss_gate_v = (
        DEFAULT_RSS_GATE_FACTOR if rss_gate_factor is None else float(rss_gate_factor)
    )

    extra_kwargs: Dict[str, Any] = {}
    if min_contributors is not None:
        extra_kwargs["min_contributors"] = int(min_contributors)
    if sigma_tau_fraction_max is not None:
        extra_kwargs["sigma_tau_fraction_max"] = float(sigma_tau_fraction_max)
    if bimodality_dominant_fraction is not None:
        extra_kwargs["bimodality_dominant_fraction"] = float(
            bimodality_dominant_fraction
        )
    extra_kwargs["compute_band_majorities_flag"] = bool(compute_band_majorities)
    if min_contributors_per_band is not None:
        extra_kwargs["min_contributors_per_band"] = int(min_contributors_per_band)

    sideband = (
        fid.sideband.value if hasattr(fid.sideband, "value") else str(fid.sideband)
    )

    result: TauCalibrationResult = extract_tau_majority(
        np.asarray(fid.data, dtype=float),
        sample_dt_us,
        start_us=start_us,
        end_us=end_us,
        probe_freq_mhz=float(fid.probe_freq_mhz),
        sideband=sideband,
        trim_lo_mhz=float(trim_lo_mhz),
        trim_hi_mhz=float(trim_hi_mhz),
        sigma_time=sigma_time,
        n_seg=n_seg_v,
        t_sigma=t_sigma_v,
        tau_max_us=tau_max_us,
        rss_gate_factor=rss_gate_v,
        **extra_kwargs,
    )

    parameters_used: Dict[str, Any] = {
        "n_seg": n_seg_v,
        "t_sigma": t_sigma_v,
        "tau_max_us": result.tau_max_us,
        "rss_gate_factor": rss_gate_v,
        "sigma_time_supplied": sigma_time is not None,
        "start_us": start_us,
        "end_us": end_us,
        "trim_lo_mhz": float(trim_lo_mhz),
        "trim_hi_mhz": float(trim_hi_mhz),
        "min_contributors": extra_kwargs.get("min_contributors"),
        "sigma_tau_fraction_max": extra_kwargs.get("sigma_tau_fraction_max"),
        "bimodality_dominant_fraction": extra_kwargs.get(
            "bimodality_dominant_fraction"
        ),
        "compute_band_majorities": bool(compute_band_majorities),
        "min_contributors_per_band": extra_kwargs.get("min_contributors_per_band"),
    }
    save_tau_calibration_impl(file_path, result, parameters_used=parameters_used)
    _update_stage_completion(file_path, STAGE_NAME)
    invalidated = invalidate_downstream_stages(file_path, STAGE_NAME)
    if invalidated:
        logger.info(
            "Stage 2b re-run invalidated downstream stages: %s",
            invalidated,
        )

    return {
        "status": "success",
        "tau_calibration": result,
        "parameters_used": parameters_used,
        "invalidated_stages": invalidated,
    }


def save_tau_calibration_impl(
    file_path: str,
    result: TauCalibrationResult,
    *,
    parameters_used: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a :class:`TauCalibrationResult` into ``/stage2b_tau_calibration``."""
    with h5py.File(file_path, "a") as h5f:
        if TAU_GROUP_PATH in h5f:
            del h5f[TAU_GROUP_PATH]
        grp = h5f.create_group(TAU_GROUP_PATH)
        save_tau_calibration_to_hdf5(result, grp)
        grp.attrs["creation_time"] = datetime.now().isoformat()
        if parameters_used is not None:
            grp.attrs["parameters_used"] = json.dumps(
                parameters_used, default=str
            )


def load_tau_calibration_impl(file_path: str) -> Dict[str, Any]:
    """Load the persisted :class:`TauCalibrationResult` from ``file_path``.

    Raises
    ------
    ValueError
        If Stage 2b has not been completed.
    """
    with h5py.File(file_path, "r") as h5f:
        if TAU_GROUP_PATH not in h5f:
            raise ValueError(
                "Stage 2b (tau calibration) has not been completed for "
                f"{file_path}. Run calibrate_tau() / calibrate-tau first."
            )
        grp = h5f[TAU_GROUP_PATH]
        result = load_tau_calibration_from_hdf5(grp)
        creation_time = grp.attrs.get("creation_time", "unknown")
        parameters_used: Dict[str, Any] = {}
        if "parameters_used" in grp.attrs:
            try:
                parameters_used = json.loads(grp.attrs["parameters_used"])
            except (json.JSONDecodeError, TypeError):
                logger.warning("Could not parse saved tau-calibration parameters")
    return {
        "tau_calibration": result,
        "creation_time": creation_time,
        "parameters_used": parameters_used,
    }


def tau_calibration_present(file_path: str) -> bool:
    """Lightweight: does the ``.ftmw`` file have a persisted Stage 2b result?"""
    try:
        with h5py.File(file_path, "r") as h5f:
            return TAU_GROUP_PATH in h5f
    except (OSError, KeyError):
        return False
