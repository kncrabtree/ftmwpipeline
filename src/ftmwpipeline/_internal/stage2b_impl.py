"""Shared implementation for Stage 2b: data-driven tau calibration.

Wraps :func:`extract_tau_majority` so the three user-facing interfaces (CLI,
Pipeline class, functional API) share one orchestration layer. The
calibration runs on the raw FID (no apodization), reads its active region
and Stage 1 trim from the persisted Stage 1 settings, and persists the
result to ``/stage2b_tau_calibration``.

Stage 2b sits between Stage 2 (noise estimation) and Stage 3 (peak
detection): Stage 3's gap pass and Stage 5's tau anchor both consume
``tau_maj``. Re-running Stage 2b invalidates Stages 3-5 downstream.

Knob configuration follows the four-layer resolver pattern shared with
Stage 5: each call bundles its legacy per-knob kwargs into an explicit
:class:`TauCalibrationSettings`, which composes against the optional
``settings=`` / ``preset=`` layer, the persisted ``stage2b_tau`` settings
group, and the hard-default table. The resolved settings drive the
kernel call and are stamped back onto the file so a follow-up no-kwargs
call inherits the same recipe.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import h5py
import numpy as np

from ..core.settings import FT_PROCESSING_PATH, FTSettings
from ..core.tau_calibration_settings import TauCalibrationSettings
from ..file_manager import (
    PipelineStageTracker,
    StageDependencyError,
    invalidate_downstream_stages,
)
from ..fitting.tau_calibration import (
    TauCalibrationResult,
    extract_tau_majority,
)
from ..io.tau_calibration_serialization import GROUP_PATH as TAU_GROUP_PATH
from ..io.tau_calibration_serialization import (
    load_tau_calibration_from_hdf5,
    save_tau_calibration_to_hdf5,
)
from ..io.tau_calibration_settings_serialization import (
    save_tau_calibration_settings_to_h5,
)
from .deprecation import warn_legacy_kwargs
from .shape_recommendation_impl import recommend_shape_impl
from .stage0_impl import load_fid_from_pipeline_impl
from .stage1_impl import _read_settings_layer
from .stage2_impl import _update_stage_completion
from .tau_settings_resolution import (
    _required_bool,
    _required_float,
    _required_int,
    resolve_with_preset_and_persisted,
)

logger = logging.getLogger(__name__)

STAGE_NAME = "stage2b_tau_calibration"


def _read_canonical_ft_settings(file_path: str) -> FTSettings:
    """Resolve the canonical Stage 1 FT settings the calibration consumes."""
    settings = _read_settings_layer(file_path, FT_PROCESSING_PATH)
    if settings is None:
        raise StageDependencyError(
            STAGE_NAME,
            ["stage1_complex_ft"],
            Path(file_path),
        )
    return settings


def _build_explicit_from_kwargs(
    *,
    n_seg: Optional[int],
    t_sigma: Optional[float],
    tau_max_us: Optional[float],
    rss_gate_factor: Optional[float],
    sigma_time: Optional[float],
    min_contributors: Optional[int],
    sigma_tau_fraction_max: Optional[float],
    bimodality_dominant_fraction: Optional[float],
    compute_band_majorities: Optional[bool],
    min_contributors_per_band: Optional[int],
) -> TauCalibrationSettings:
    """Bundle legacy per-knob kwargs into an explicit-layer settings instance.

    Any kwarg that is ``None`` leaves the corresponding field unset so
    the resolver can fall through to higher layers.
    """
    explicit = TauCalibrationSettings()
    explicit.stft.n_seg = n_seg
    explicit.stft.t_sigma = t_sigma
    explicit.stft.tau_max_us = tau_max_us
    explicit.stft.rss_gate_factor = rss_gate_factor
    explicit.stft.sigma_time = sigma_time
    explicit.aggregation.min_contributors = min_contributors
    explicit.aggregation.sigma_tau_fraction_max = sigma_tau_fraction_max
    explicit.aggregation.bimodality_dominant_fraction = bimodality_dominant_fraction
    explicit.band.compute_band_majorities = compute_band_majorities
    explicit.band.min_contributors_per_band = min_contributors_per_band
    return explicit


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
    compute_band_majorities: Optional[bool] = None,
    min_contributors_per_band: Optional[int] = None,
    settings: Optional[TauCalibrationSettings] = None,
    preset: Optional[str] = None,
) -> Dict[str, Any]:
    """Run STFT tau calibration and persist the result to ``file_path``.

    Requires Stages 0-2 to be completed (Stage 1 owns the active-region and
    trim parameters the calibration consumes; Stage 2's σ is not strictly
    needed but is the natural noise reference for downstream consistency).

    Parameters left as ``None`` fall through the resolution chain
    (``explicit > persisted > preset > recommended > hard default``); the
    resolved settings are stamped to ``processing_parameters/stage2b_tau``
    so a follow-up no-kwargs call on the same file inherits them. Pass
    ``settings=`` to drive the calibration from a Python dataclass, or
    ``preset=NAME_OR_PATH`` to load from packaged YAML; they are mutually
    exclusive.

    Returns
    -------
    dict
        ``{"tau_calibration": TauCalibrationResult, "parameters_used": dict,
        "status": "success"}``.
    """
    warn_legacy_kwargs(
        func_name="calibrate_tau",
        legacy_kwargs={
            "n_seg": n_seg,
            "t_sigma": t_sigma,
            "tau_max_us": tau_max_us,
            "rss_gate_factor": rss_gate_factor,
            "sigma_time": sigma_time,
            "min_contributors": min_contributors,
            "sigma_tau_fraction_max": sigma_tau_fraction_max,
            "bimodality_dominant_fraction": bimodality_dominant_fraction,
            "compute_band_majorities": compute_band_majorities,
            "min_contributors_per_band": min_contributors_per_band,
        },
        migration_hint=(
            "use settings=TauCalibrationSettings(...) or preset='name' to "
            "drive Stage 2b from the settings resolver"
        ),
    )

    file_path_obj = Path(file_path)
    with h5py.File(file_path, "r") as h5f:
        if "stage2_noise_result" not in h5f:
            raise StageDependencyError(
                STAGE_NAME,
                ["stage2_noise_result"],
                file_path_obj,
            )

    explicit = _build_explicit_from_kwargs(
        n_seg=n_seg,
        t_sigma=t_sigma,
        tau_max_us=tau_max_us,
        rss_gate_factor=rss_gate_factor,
        sigma_time=sigma_time,
        min_contributors=min_contributors,
        sigma_tau_fraction_max=sigma_tau_fraction_max,
        bimodality_dominant_fraction=bimodality_dominant_fraction,
        compute_band_majorities=compute_band_majorities,
        min_contributors_per_band=min_contributors_per_band,
    )
    resolved, preset_name = resolve_with_preset_and_persisted(
        file_path,
        explicit=explicit,
        settings=settings,
        preset=preset,
    )

    ft_settings = _read_canonical_ft_settings(file_path)
    fid = load_fid_from_pipeline_impl(file_path)
    sample_dt_us = float(fid.spacing * 1e6)

    start_us = float(ft_settings.start_us) if ft_settings.start_us is not None else 0.0
    end_us = (
        float(ft_settings.end_us)
        if ft_settings.end_us is not None
        else float(fid.duration_us)
    )
    if ft_settings.trim is None:
        raise ValueError(
            "Stage 1 canonical FT settings have no frequency trim; tau "
            "calibration uses the persisted trim range to match the user "
            "spectrum. Set trim on compute_ft() first."
        )
    trim_lo_mhz, trim_hi_mhz = ft_settings.trim
    sideband = (
        fid.sideband.value if hasattr(fid.sideband, "value") else str(fid.sideband)
    )

    # Lift resolved settings into the kernel-call kwargs bag. Every field
    # with a hard default is guaranteed non-None after resolve().
    stft = resolved.stft
    aggr = resolved.aggregation
    band = resolved.band
    polish = resolved.polish
    n_seg_v = _required_int(stft.n_seg, "stft.n_seg")
    t_sigma_v = _required_float(stft.t_sigma, "stft.t_sigma")
    rss_gate_v = _required_float(stft.rss_gate_factor, "stft.rss_gate_factor")
    relative_gate_v = _required_float(
        stft.relative_gate_fraction, "stft.relative_gate_fraction"
    )
    min_contrib_v = _required_int(aggr.min_contributors, "aggregation.min_contributors")
    sigma_tau_frac_v = _required_float(
        aggr.sigma_tau_fraction_max, "aggregation.sigma_tau_fraction_max"
    )
    bimod_frac_v = _required_float(
        aggr.bimodality_dominant_fraction,
        "aggregation.bimodality_dominant_fraction",
    )
    spur_mult_v = _required_float(
        aggr.spur_cluster_multiplier, "aggregation.spur_cluster_multiplier"
    )
    compute_bands_v = _required_bool(
        band.compute_band_majorities, "band.compute_band_majorities"
    )
    min_per_band_v = _required_int(
        band.min_contributors_per_band, "band.min_contributors_per_band"
    )
    polish_v = _required_bool(polish.polish, "polish.polish")
    polish_n_iter_v = _required_int(polish.polish_n_iter, "polish.polish_n_iter")
    polish_debias_v = _required_bool(
        polish.polish_noise_debias, "polish.polish_noise_debias"
    )
    # polish_snr_cap and polish_top_n legitimately accept None (None on
    # polish_snr_cap means "polish every contributor"; None on
    # polish_top_n means "no top-N restriction").

    kernel_kwargs: Dict[str, Any] = dict(
        n_seg=n_seg_v,
        t_sigma=t_sigma_v,
        tau_max_us=stft.tau_max_us,
        rss_gate_factor=rss_gate_v,
        relative_gate_fraction=relative_gate_v,
        spur_cluster_multiplier=spur_mult_v,
        min_contributors=min_contrib_v,
        sigma_tau_fraction_max=sigma_tau_frac_v,
        bimodality_dominant_fraction=bimod_frac_v,
        polish=polish_v,
        polish_n_iter=polish_n_iter_v,
        polish_top_n=polish.polish_top_n,
        polish_snr_cap=polish.polish_snr_cap,
        polish_noise_debias=polish_debias_v,
        sigma_x_full=stft.sigma_x_full,
        compute_band_majorities_flag=compute_bands_v,
        band_edges_mhz=band.band_edges_mhz,
        min_contributors_per_band=min_per_band_v,
    )
    if band.band_labels is not None:
        kernel_kwargs["band_labels"] = band.band_labels

    result: TauCalibrationResult = extract_tau_majority(
        np.asarray(fid.data, dtype=float),
        sample_dt_us,
        start_us=start_us,
        end_us=end_us,
        probe_freq_mhz=float(fid.probe_freq_mhz),
        sideband=sideband,
        trim_lo_mhz=float(trim_lo_mhz),
        trim_hi_mhz=float(trim_hi_mhz),
        sigma_time=stft.sigma_time,
        **kernel_kwargs,
    )

    parameters_used: Dict[str, Any] = {
        "n_seg": n_seg_v,
        "t_sigma": t_sigma_v,
        "tau_max_us": result.tau_max_us,
        "rss_gate_factor": rss_gate_v,
        "relative_gate_fraction": relative_gate_v,
        "spur_cluster_multiplier": spur_mult_v,
        "sigma_time_supplied": stft.sigma_time is not None,
        "start_us": start_us,
        "end_us": end_us,
        "trim_lo_mhz": float(trim_lo_mhz),
        "trim_hi_mhz": float(trim_hi_mhz),
        "min_contributors": min_contrib_v,
        "sigma_tau_fraction_max": sigma_tau_frac_v,
        "bimodality_dominant_fraction": bimod_frac_v,
        "polish": polish_v,
        "polish_n_iter": polish_n_iter_v,
        "polish_top_n": polish.polish_top_n,
        "polish_snr_cap": polish.polish_snr_cap,
        "polish_noise_debias": polish_debias_v,
        "compute_band_majorities": compute_bands_v,
        "min_contributors_per_band": min_per_band_v,
        "band_edges_mhz": (
            list(band.band_edges_mhz) if band.band_edges_mhz is not None else None
        ),
        "band_labels": (
            list(band.band_labels) if band.band_labels is not None else None
        ),
    }
    save_tau_calibration_impl(file_path, result, parameters_used=parameters_used)
    save_tau_calibration_settings_to_h5(
        file_path,
        resolved,
        preset_name=preset_name,
    )
    _update_stage_completion(file_path, STAGE_NAME)
    invalidated = invalidate_downstream_stages(file_path, STAGE_NAME)
    if invalidated:
        logger.info(
            "Stage 2b re-run invalidated downstream stages: %s",
            invalidated,
        )

    if resolved.recommendation.auto_recommend:
        # Run the 3-way L/G/V shape recommendation as part of the calibration
        # so Stage 5's resolver inherits the verdict on every fresh Stage 2b
        # run. Re-uses the just-persisted settings via the no-kwargs call.
        logger.info("auto_recommend on: running compute_shape_recommendation")
        recommend_shape_impl(file_path)

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
    """Write a :class:`TauCalibrationResult` into ``/stage2b_tau_calibration``.

    Also stamps a ``recommended_shape`` attribute on the group. The
    Stage 5 resolver reads this attr as the *recommended* layer of the
    fit-settings chain; it carries the ``__None__`` sentinel (no
    recommendation) unless a Stage 2b L/G discriminator supplies one.
    """
    from ..io.stage_fit_settings_serialization import (
        write_stage2b_recommended_shape,
    )

    with h5py.File(file_path, "a") as h5f:
        if TAU_GROUP_PATH in h5f:
            del h5f[TAU_GROUP_PATH]
        grp = h5f.create_group(TAU_GROUP_PATH)
        save_tau_calibration_to_hdf5(result, grp)
        grp.attrs["creation_time"] = datetime.now().isoformat()
        if parameters_used is not None:
            grp.attrs["parameters_used"] = json.dumps(parameters_used, default=str)
    write_stage2b_recommended_shape(file_path, shape=None)


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
                f"{file_path}. Run calibrate_tau() / 'tau run' first."
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
