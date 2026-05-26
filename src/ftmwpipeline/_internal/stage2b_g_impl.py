"""Shared implementation for the Stage 2b Gaussian-shape τ_G calibration.

Twin of :mod:`stage2b_impl`. Wraps :func:`extract_tau_G_majority` so the
three user-facing interfaces (CLI, Pipeline class, functional API) share
one orchestration layer. The calibration runs on the raw FID, reads its
active region and Stage 1 trim from the persisted Stage 1 settings, and
persists the result to ``/stage2b_tau_G_calibration``.

The pure-exp Stage 2b (``calibrate_tau``) and the Gaussian Stage 2b
(``calibrate_tau_G``) are independent: a user running
``fit_peaks(shape='lorentzian')`` consumes the pure-exp calibration; a
user running ``fit_peaks(shape='gaussian')`` consumes this Gaussian
calibration. Both can coexist on the same ``.ftmw`` file.

The serialised payload reuses :class:`TauCalibrationResult` (same struct,
different group path); every ``τ`` / ``tau`` field carries ``τ_G`` when
loaded from this twin's group, disambiguated by the path.
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
    StageDependencyError,
    invalidate_downstream_stages,
)
from ..fitting.tau_calibration import (
    DEFAULT_N_SEG,
    DEFAULT_RSS_GATE_FACTOR,
    DEFAULT_T_SIGMA,
    DEFAULT_TAU_G_BOUND_HI,
    DEFAULT_TAU_G_BOUND_LO,
    DEFAULT_TAU_G_DELTA_CHI2R_MIN,
    DEFAULT_TAU_G_MIN_CONTRIBUTORS,
    DEFAULT_TAU_G_SEEDS,
    DEFAULT_TAU_G_SNR_MIN,
    DEFAULT_TAU_G_UPPER_FRACTION,
    TauCalibrationResult,
    extract_tau_G_majority,
)
from ..io.tau_calibration_serialization import (
    load_tau_calibration_from_hdf5,
    save_tau_calibration_to_hdf5,
)
from .stage0_impl import load_fid_from_pipeline_impl
from .stage1_impl import _read_settings_layer
from .stage2_impl import _update_stage_completion

logger = logging.getLogger(__name__)

STAGE_NAME = "stage2b_tau_G_calibration"
GROUP_PATH = "stage2b_tau_G_calibration"


def _read_canonical_ft_settings(file_path: str) -> FTSettings:
    """Resolve the canonical Stage 1 FT settings the calibration consumes."""
    settings = _read_settings_layer(file_path, FT_PROCESSING_PATH)
    if settings is None:
        raise StageDependencyError(
            STAGE_NAME, ["stage1_complex_ft"], Path(file_path),
        )
    return settings


def calibrate_tau_G_impl(
    file_path: str,
    *,
    n_seg: Optional[int] = None,
    t_sigma: Optional[float] = None,
    tau_max_us: Optional[float] = None,
    rss_gate_factor: Optional[float] = None,
    sigma_time: Optional[float] = None,
    snr_min: Optional[float] = None,
    tau_G_bound_lo: Optional[float] = None,
    tau_G_bound_hi: Optional[float] = None,
    delta_chi2r_min: Optional[float] = None,
    tau_G_upper_fraction: Optional[float] = None,
    min_contributors: Optional[int] = None,
    sigma_tau_fraction_max: Optional[float] = None,
    bimodality_dominant_fraction: Optional[float] = None,
    compute_band_majorities: bool = True,
    min_contributors_per_band: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the Gaussian-shape τ_G calibration and persist it to ``file_path``.

    Requires Stages 0-2 to be completed (Stage 1 owns the active-region and
    trim parameters; Stage 2 is the canonical noise reference for the
    sliding-active-window STFT). Parameters left as ``None`` use the
    documented defaults from :mod:`ftmwpipeline.fitting.tau_calibration`.

    Returns
    -------
    dict
        ``{"tau_G_calibration": TauCalibrationResult, "parameters_used":
        dict, "status": "success", "invalidated_stages": list}``.
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
            "Stage 1 canonical FT settings have no frequency trim; τ_G "
            "calibration uses the persisted trim range to match the user "
            "spectrum. Set trim on compute_ft() first."
        )
    trim_lo_mhz, trim_hi_mhz = settings.trim

    n_seg_v = DEFAULT_N_SEG if n_seg is None else int(n_seg)
    t_sigma_v = DEFAULT_T_SIGMA if t_sigma is None else float(t_sigma)
    rss_gate_v = (
        DEFAULT_RSS_GATE_FACTOR if rss_gate_factor is None else float(rss_gate_factor)
    )
    snr_min_v = DEFAULT_TAU_G_SNR_MIN if snr_min is None else float(snr_min)
    bound_lo_v = (
        DEFAULT_TAU_G_BOUND_LO if tau_G_bound_lo is None else float(tau_G_bound_lo)
    )
    bound_hi_v = (
        DEFAULT_TAU_G_BOUND_HI if tau_G_bound_hi is None else float(tau_G_bound_hi)
    )
    delta_min_v = (
        DEFAULT_TAU_G_DELTA_CHI2R_MIN
        if delta_chi2r_min is None
        else float(delta_chi2r_min)
    )
    upper_frac_v = (
        DEFAULT_TAU_G_UPPER_FRACTION
        if tau_G_upper_fraction is None
        else float(tau_G_upper_fraction)
    )
    min_contrib_v = (
        DEFAULT_TAU_G_MIN_CONTRIBUTORS
        if min_contributors is None
        else int(min_contributors)
    )

    extra_kwargs: Dict[str, Any] = {}
    if sigma_tau_fraction_max is not None:
        extra_kwargs["sigma_tau_fraction_max"] = float(sigma_tau_fraction_max)
    if bimodality_dominant_fraction is not None:
        extra_kwargs["bimodality_dominant_fraction"] = float(
            bimodality_dominant_fraction
        )
    if min_contributors_per_band is not None:
        extra_kwargs["min_contributors_per_band"] = int(min_contributors_per_band)

    sideband = (
        fid.sideband.value if hasattr(fid.sideband, "value") else str(fid.sideband)
    )

    result: TauCalibrationResult = extract_tau_G_majority(
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
        snr_min=snr_min_v,
        tau_G_bound_lo=bound_lo_v,
        tau_G_bound_hi=bound_hi_v,
        tau_G_seeds=DEFAULT_TAU_G_SEEDS,
        delta_chi2r_min=delta_min_v,
        tau_G_upper_fraction=upper_frac_v,
        min_contributors=min_contrib_v,
        compute_band_majorities_flag=bool(compute_band_majorities),
        **extra_kwargs,
    )

    parameters_used: Dict[str, Any] = {
        "n_seg": n_seg_v,
        "t_sigma": t_sigma_v,
        "tau_max_us": result.tau_max_us,
        "rss_gate_factor": rss_gate_v,
        "sigma_time_supplied": sigma_time is not None,
        "snr_min": snr_min_v,
        "tau_G_bound_lo": bound_lo_v,
        "tau_G_bound_hi": bound_hi_v,
        "delta_chi2r_min": delta_min_v,
        "tau_G_upper_fraction": upper_frac_v,
        "tau_G_seeds": list(DEFAULT_TAU_G_SEEDS),
        "min_contributors": min_contrib_v,
        "sigma_tau_fraction_max": extra_kwargs.get("sigma_tau_fraction_max"),
        "bimodality_dominant_fraction": extra_kwargs.get(
            "bimodality_dominant_fraction"
        ),
        "start_us": start_us,
        "end_us": end_us,
        "trim_lo_mhz": float(trim_lo_mhz),
        "trim_hi_mhz": float(trim_hi_mhz),
        "compute_band_majorities": bool(compute_band_majorities),
        "min_contributors_per_band": extra_kwargs.get("min_contributors_per_band"),
    }
    save_tau_G_calibration_impl(file_path, result, parameters_used=parameters_used)
    _update_stage_completion(file_path, STAGE_NAME)
    invalidated = invalidate_downstream_stages(file_path, STAGE_NAME)
    if invalidated:
        logger.info(
            "Stage 2b τ_G re-run invalidated downstream stages: %s", invalidated,
        )

    return {
        "status": "success",
        "tau_G_calibration": result,
        "parameters_used": parameters_used,
        "invalidated_stages": invalidated,
    }


def save_tau_G_calibration_impl(
    file_path: str,
    result: TauCalibrationResult,
    *,
    parameters_used: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a :class:`TauCalibrationResult` into ``/stage2b_tau_G_calibration``."""
    with h5py.File(file_path, "a") as h5f:
        if GROUP_PATH in h5f:
            del h5f[GROUP_PATH]
        grp = h5f.create_group(GROUP_PATH)
        save_tau_calibration_to_hdf5(result, grp)
        grp.attrs["creation_time"] = datetime.now().isoformat()
        grp.attrs["shape"] = "gaussian"
        if parameters_used is not None:
            grp.attrs["parameters_used"] = json.dumps(
                parameters_used, default=str
            )


def load_tau_G_calibration_impl(file_path: str) -> Dict[str, Any]:
    """Load the persisted Gaussian Stage 2b :class:`TauCalibrationResult`."""
    with h5py.File(file_path, "r") as h5f:
        if GROUP_PATH not in h5f:
            raise ValueError(
                "Stage 2b Gaussian τ_G calibration has not been completed "
                f"for {file_path}. Run calibrate_tau_G() / calibrate-tau-G "
                "first."
            )
        grp = h5f[GROUP_PATH]
        result = load_tau_calibration_from_hdf5(grp)
        creation_time = grp.attrs.get("creation_time", "unknown")
        parameters_used: Dict[str, Any] = {}
        if "parameters_used" in grp.attrs:
            try:
                parameters_used = json.loads(grp.attrs["parameters_used"])
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Could not parse saved τ_G-calibration parameters"
                )
    return {
        "tau_G_calibration": result,
        "creation_time": creation_time,
        "parameters_used": parameters_used,
    }


def tau_G_calibration_present(file_path: str) -> bool:
    """Lightweight: does the ``.ftmw`` file have a persisted Gaussian Stage 2b result?"""
    try:
        with h5py.File(file_path, "r") as h5f:
            return GROUP_PATH in h5f
    except (OSError, KeyError):
        return False
