"""Shared implementation for Stage 2b: data-driven tau calibration.

Wraps the STFT tau extractors so the three user-facing interfaces (CLI,
Pipeline class, functional API) share one orchestration layer. The
calibration runs on the raw FID (no apodization), reads its active region
and Stage 1 trim from the persisted Stage 1 settings, and persists the
result to a per-shape group.

The calibration has two shape variants, selected by the ``shape`` argument:

* ``"lorentzian"`` -- the pure-exponential decay (:func:`extract_tau_majority`),
  persisted to ``/stage2b_tau_calibration``. Stage 5 consumes it for a
  ``shape='lorentzian'`` fit.
* ``"gaussian"`` -- the pure-Gaussian envelope (:func:`extract_tau_G_majority`),
  persisted to ``/stage2b_tau_G_calibration``. Stage 5 consumes it for a
  ``shape='gaussian'`` fit.

Both variants are the same algorithm under a different bin classifier and
contributor-eligibility gate; they share one canonical settings record and can
coexist on the same ``.ftmw`` file. The serialized payload reuses
:class:`TauCalibrationResult` (same struct, different group path); every ``tau``
field carries ``tau_G`` when loaded from the Gaussian group, disambiguated by
the path.

Stage 2b sits between Stage 2 (noise estimation) and Stage 3 (peak detection):
Stage 3's gap pass consumes the band-wide ``tau_maj``, and Stage 5's per-window
tau anchor resolves through :func:`~ftmwpipeline._internal.stage5_impl.resolve_window_tau_anchor`
(``tau_maj`` is only its degenerate single-band fallback). Re-running Stage 2b
invalidates Stages 3-5 downstream.

Knob configuration follows the four-layer resolver pattern shared with Stage 5:
the optional ``settings=`` / ``preset=`` layer composes against the persisted
``stage2b_tau`` settings group and the hard-default table. The resolved settings
drive the kernel call and are stamped back onto the file so a follow-up no-arg
call inherits the same recipe.
"""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np

from ..core.settings import FT_PROCESSING_PATH, FTSettings
from ..core.tau_calibration_settings import TauCalibrationSettings
from ..file_manager import (
    StageDependencyError,
    invalidate_downstream_stages,
)
from ..fitting.tau_calibration import (
    TauCalibrationResult,
    extract_tau_G_majority,
    extract_tau_majority,
)
from ..io.tau_calibration_serialization import GROUP_PATH as LORENTZIAN_GROUP_PATH
from ..io.tau_calibration_serialization import (
    load_tau_calibration_from_hdf5,
    save_tau_calibration_to_hdf5,
)
from ..io.tau_calibration_settings_serialization import (
    save_tau_calibration_settings_to_h5,
)
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

LORENTZIAN_STAGE_NAME = "stage2b_tau_calibration"
GAUSSIAN_GROUP_PATH = "stage2b_tau_G_calibration"
GAUSSIAN_STAGE_NAME = "stage2b_tau_G_calibration"

_VALID_SHAPES = ("lorentzian", "gaussian")


def _check_shape(shape: str) -> str:
    if shape not in _VALID_SHAPES:
        raise ValueError(f"shape must be one of {_VALID_SHAPES}; got {shape!r}")
    return shape


def _group_path_for_shape(shape: str) -> str:
    return GAUSSIAN_GROUP_PATH if shape == "gaussian" else LORENTZIAN_GROUP_PATH


def _stage_name_for_shape(shape: str) -> str:
    return GAUSSIAN_STAGE_NAME if shape == "gaussian" else LORENTZIAN_STAGE_NAME


def _read_canonical_ft_settings(file_path: str, stage_name: str) -> FTSettings:
    """Resolve the canonical Stage 1 FT settings the calibration consumes."""
    settings = _read_settings_layer(file_path, FT_PROCESSING_PATH)
    if settings is None:
        raise StageDependencyError(
            stage_name,
            ["stage1_complex_ft"],
            Path(file_path),
        )
    return settings


def _route_min_contributors_for_gaussian(
    settings: Optional[TauCalibrationSettings],
) -> Optional[TauCalibrationSettings]:
    """Move an explicit ``aggregation.min_contributors`` onto the Gaussian block.

    The shared ``min_contributors`` knob has one CLI/scan flag that lands on
    ``aggregation.min_contributors`` (the Lorentzian block), but the Gaussian
    twin's precondition reads ``gaussian.min_contributors``. When only the
    aggregation field is set, route it across so every interface (CLI,
    Pipeline, functional API) treats ``--min-contributors`` the same way for a
    Gaussian run. Operates on a copy so the caller's bundle is untouched.
    """
    if (
        settings is None
        or settings.aggregation.min_contributors is None
        or settings.gaussian.min_contributors is not None
    ):
        return settings
    routed = copy.deepcopy(settings)
    routed.gaussian.min_contributors = routed.aggregation.min_contributors
    routed.aggregation.min_contributors = None
    return routed


def calibrate_tau_impl(
    file_path: str,
    *,
    shape: str = "lorentzian",
    settings: Optional[TauCalibrationSettings] = None,
    preset: Optional[str] = None,
    _run_recommendation: bool = True,
    _ensure_recommended_twin: bool = True,
) -> Dict[str, Any]:
    """Run STFT tau calibration for ``shape`` and persist it to ``file_path``.

    Requires Stages 0-2 to be completed (Stage 1 owns the active-region and
    trim parameters the calibration consumes; Stage 2 is the natural noise
    reference for the sliding-active-window STFT). ``shape`` selects the
    pure-exponential (``"lorentzian"``) or pure-Gaussian (``"gaussian"``)
    variant and its persistence group.

    Settings resolve through the chain (``settings`` / ``preset`` > persisted >
    hard default); the resolved settings are stamped to
    ``processing_parameters/stage2b_tau`` -- the single record both shape
    variants share -- so a follow-up no-arg call inherits them. Pass
    ``settings=`` to drive the calibration from a :class:`TauCalibrationSettings`
    dataclass, or ``preset=NAME_OR_PATH`` to load from packaged YAML; they may be
    combined. A ``settings`` bundle is the explicit override that outranks the
    persisted record, while a ``preset`` seeds only the fields neither the
    explicit layer nor the persisted record has fixed (the persisted record
    outranks the preset, per D11).

    Returns
    -------
    dict
        ``{"tau_calibration": TauCalibrationResult, "parameters_used": dict,
        "status": "success", "invalidated_stages": list}``. The
        ``tau_calibration`` value carries ``tau_G`` semantics when
        ``shape="gaussian"``.
    """
    _check_shape(shape)
    stage_name = _stage_name_for_shape(shape)
    file_path_obj = Path(file_path)
    with h5py.File(file_path, "r") as h5f:
        if "stage2_noise_result" not in h5f:
            raise StageDependencyError(
                stage_name,
                ["stage2_noise_result"],
                file_path_obj,
            )

    if shape == "gaussian":
        settings = _route_min_contributors_for_gaussian(settings)

    explicit = TauCalibrationSettings()
    resolved, preset_name = resolve_with_preset_and_persisted(
        file_path,
        explicit=explicit,
        settings=settings,
        preset=preset,
    )

    ft_settings = _read_canonical_ft_settings(file_path, stage_name)
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

    # Lift resolved settings into the kernel-call kwargs bag. Every field with a
    # hard default is guaranteed non-None after resolve().
    stft = resolved.stft
    aggr = resolved.aggregation
    band = resolved.band
    n_seg_v = _required_int(stft.n_seg, "stft.n_seg")
    t_sigma_v = _required_float(stft.t_sigma, "stft.t_sigma")
    rss_gate_v = _required_float(stft.rss_gate_factor, "stft.rss_gate_factor")
    relative_gate_v = _required_float(
        stft.relative_gate_fraction, "stft.relative_gate_fraction"
    )
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
    sigma_tau_floor_v = _required_float(
        aggr.sigma_tau_floor_us, "aggregation.sigma_tau_floor_us"
    )
    compute_bands_v = _required_bool(
        band.compute_band_majorities, "band.compute_band_majorities"
    )
    min_per_band_v = _required_int(
        band.min_contributors_per_band, "band.min_contributors_per_band"
    )

    kernel_kwargs: Dict[str, Any] = dict(
        n_seg=n_seg_v,
        t_sigma=t_sigma_v,
        tau_max_us=stft.tau_max_us,
        rss_gate_factor=rss_gate_v,
        relative_gate_fraction=relative_gate_v,
        spur_cluster_multiplier=spur_mult_v,
        sigma_tau_fraction_max=sigma_tau_frac_v,
        bimodality_dominant_fraction=bimod_frac_v,
        sigma_x_full=stft.sigma_x_full,
        compute_band_majorities_flag=compute_bands_v,
        band_edges_mhz=band.band_edges_mhz,
        min_contributors_per_band=min_per_band_v,
        sigma_tau_floor_us=sigma_tau_floor_v,
    )
    if band.band_labels is not None:
        kernel_kwargs["band_labels"] = band.band_labels

    parameters_used: Dict[str, Any] = {
        "shape": shape,
        "n_seg": n_seg_v,
        "t_sigma": t_sigma_v,
        "rss_gate_factor": rss_gate_v,
        "relative_gate_fraction": relative_gate_v,
        "spur_cluster_multiplier": spur_mult_v,
        "sigma_time_supplied": stft.sigma_time is not None,
        "start_us": start_us,
        "end_us": end_us,
        "trim_lo_mhz": float(trim_lo_mhz),
        "trim_hi_mhz": float(trim_hi_mhz),
        "sigma_tau_fraction_max": sigma_tau_frac_v,
        "bimodality_dominant_fraction": bimod_frac_v,
        "sigma_tau_floor_us": sigma_tau_floor_v,
        "compute_band_majorities": compute_bands_v,
        "min_contributors_per_band": min_per_band_v,
        "band_edges_mhz": (
            list(band.band_edges_mhz) if band.band_edges_mhz is not None else None
        ),
        "band_labels": (
            list(band.band_labels) if band.band_labels is not None else None
        ),
    }

    fid_data = np.asarray(fid.data, dtype=float)
    common_call: Dict[str, Any] = dict(
        sample_dt_us=sample_dt_us,
        start_us=start_us,
        end_us=end_us,
        probe_freq_mhz=float(fid.probe_freq_mhz),
        sideband=sideband,
        trim_lo_mhz=float(trim_lo_mhz),
        trim_hi_mhz=float(trim_hi_mhz),
        sigma_time=stft.sigma_time,
    )

    if shape == "gaussian":
        gauss = resolved.gaussian
        snr_min_v = _required_float(gauss.snr_min, "gaussian.snr_min")
        bound_lo_v = _required_float(gauss.tau_G_bound_lo, "gaussian.tau_G_bound_lo")
        bound_hi_v = _required_float(gauss.tau_G_bound_hi, "gaussian.tau_G_bound_hi")
        delta_min_v = _required_float(gauss.delta_chi2r_min, "gaussian.delta_chi2r_min")
        upper_frac_v = _required_float(
            gauss.tau_G_upper_fraction, "gaussian.tau_G_upper_fraction"
        )
        min_contrib_v = _required_int(
            gauss.min_contributors, "gaussian.min_contributors"
        )
        if gauss.tau_G_seeds is None:
            raise ValueError(
                "resolved TauCalibrationSettings.gaussian.tau_G_seeds is None; "
                "missing hard default"
            )
        seeds_v = tuple(float(v) for v in gauss.tau_G_seeds)
        kernel_kwargs.update(
            min_contributors=min_contrib_v,
            snr_min=snr_min_v,
            tau_G_bound_lo=bound_lo_v,
            tau_G_bound_hi=bound_hi_v,
            tau_G_seeds=seeds_v,
            delta_chi2r_min=delta_min_v,
            tau_G_upper_fraction=upper_frac_v,
        )
        parameters_used.update(
            min_contributors=min_contrib_v,
            snr_min=snr_min_v,
            tau_G_bound_lo=bound_lo_v,
            tau_G_bound_hi=bound_hi_v,
            delta_chi2r_min=delta_min_v,
            tau_G_upper_fraction=upper_frac_v,
            tau_G_seeds=list(seeds_v),
        )
        result: TauCalibrationResult = extract_tau_G_majority(
            fid_data, **common_call, **kernel_kwargs
        )
    else:
        polish = resolved.polish
        min_contrib_v = _required_int(
            aggr.min_contributors, "aggregation.min_contributors"
        )
        polish_v = _required_bool(polish.polish, "polish.polish")
        polish_n_iter_v = _required_int(polish.polish_n_iter, "polish.polish_n_iter")
        polish_debias_v = _required_bool(
            polish.polish_noise_debias, "polish.polish_noise_debias"
        )
        # polish_snr_cap legitimately accepts None (None = polish every
        # contributor).
        kernel_kwargs.update(
            min_contributors=min_contrib_v,
            polish=polish_v,
            polish_n_iter=polish_n_iter_v,
            polish_snr_cap=polish.polish_snr_cap,
            polish_noise_debias=polish_debias_v,
        )
        parameters_used.update(
            min_contributors=min_contrib_v,
            polish=polish_v,
            polish_n_iter=polish_n_iter_v,
            polish_snr_cap=polish.polish_snr_cap,
            polish_noise_debias=polish_debias_v,
        )
        result = extract_tau_majority(fid_data, **common_call, **kernel_kwargs)

    parameters_used["tau_max_us"] = result.tau_max_us

    save_tau_calibration_impl(
        file_path,
        result,
        shape=shape,
        parameters_used=parameters_used,
        reset_recommendation=_run_recommendation,
    )
    save_tau_calibration_settings_to_h5(
        file_path,
        resolved,
        preset_name=preset_name,
    )
    _update_stage_completion(file_path, stage_name)
    invalidated = invalidate_downstream_stages(file_path, stage_name)
    if invalidated:
        logger.info(
            "Stage 2b (%s) re-run invalidated downstream stages: %s",
            shape,
            invalidated,
        )

    if _run_recommendation and resolved.recommendation.auto_recommend:
        # Run the 3-way L/G/V shape recommendation as part of the calibration so
        # Stage 5's resolver inherits the verdict on every fresh Stage 2b run.
        logger.info("auto_recommend on: running compute_shape_recommendation")
        recommend_shape_impl(file_path)

        # Self-consistency: Stage 5 fits the *recommended* shape and consumes the
        # matching tau twin. When the vote names the shape this call did not
        # build, build that twin too so Stage 5 never silently falls back to the
        # T_active/3 default.
        if _ensure_recommended_twin:
            _build_recommended_twin(file_path, built=shape)

    return {
        "status": "success",
        "tau_calibration": result,
        "parameters_used": parameters_used,
        "invalidated_stages": invalidated,
    }


def _build_recommended_twin(file_path: str, *, built: str) -> None:
    """Build the Stage 2b tau twin matching the recommended shape, if needed.

    *built* is the shape this call already produced (``"lorentzian"`` or
    ``"gaussian"``). If the persisted ``recommended_shape`` names the *other*
    shape, build that twin too (skipping its own recommendation re-run and its
    own cross-build, so the pair is computed exactly once). A vote of ``None``
    (no clear winner -> Stage 5 defaults to Lorentzian) builds nothing extra.
    """
    from ..io.stage_fit_settings_serialization import read_stage2b_recommended_shape

    recommended = read_stage2b_recommended_shape(file_path)
    if recommended is None or recommended == built:
        return
    if recommended in _VALID_SHAPES:
        logger.info(
            "Stage 2b vote = %s; building the matching tau twin so Stage 5 has "
            "its calibration",
            recommended,
        )
        calibrate_tau_impl(
            file_path,
            shape=recommended,
            _run_recommendation=False,
            _ensure_recommended_twin=False,
        )


def save_tau_calibration_impl(
    file_path: str,
    result: TauCalibrationResult,
    *,
    shape: str = "lorentzian",
    parameters_used: Optional[Dict[str, Any]] = None,
    reset_recommendation: bool = True,
) -> None:
    """Write a :class:`TauCalibrationResult` into the per-shape Stage 2b group.

    ``reset_recommendation`` clears the file's ``recommended_shape`` attr to the
    ``__None__`` sentinel after the write. A primary calibration call resets so
    a fresh run never carries a stale vote; the cross-build that follows a
    just-computed recommendation passes ``reset_recommendation=False`` so it does
    not wipe the verdict it is acting on. The Stage 5 resolver reads the attr as
    the *recommended* layer of its fit-settings chain.
    """
    from ..io.stage_fit_settings_serialization import (
        write_stage2b_recommended_shape,
    )

    _check_shape(shape)
    group_path = _group_path_for_shape(shape)
    with h5py.File(file_path, "a") as h5f:
        if group_path in h5f:
            del h5f[group_path]
        grp = h5f.create_group(group_path)
        save_tau_calibration_to_hdf5(result, grp)
        grp.attrs["creation_time"] = datetime.now().isoformat()
        grp.attrs["shape"] = shape
        if parameters_used is not None:
            grp.attrs["parameters_used"] = json.dumps(parameters_used, default=str)
    if reset_recommendation:
        write_stage2b_recommended_shape(file_path, shape=None)


def load_tau_calibration_impl(
    file_path: str, *, shape: str = "lorentzian"
) -> Dict[str, Any]:
    """Load the persisted :class:`TauCalibrationResult` for ``shape``.

    Raises
    ------
    ValueError
        If the requested Stage 2b shape variant has not been completed.
    """
    _check_shape(shape)
    group_path = _group_path_for_shape(shape)
    with h5py.File(file_path, "r") as h5f:
        if group_path not in h5f:
            verb = "tau run --gaussian" if shape == "gaussian" else "tau run"
            raise ValueError(
                f"Stage 2b ({shape}) tau calibration has not been completed for "
                f"{file_path}. Run calibrate_tau(shape={shape!r}) / '{verb}' first."
            )
        grp = h5f[group_path]
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


def tau_calibration_present(file_path: str, *, shape: str = "lorentzian") -> bool:
    """Lightweight: does the ``.ftmw`` file carry the ``shape`` Stage 2b result?"""
    _check_shape(shape)
    group_path = _group_path_for_shape(shape)
    try:
        with h5py.File(file_path, "r") as h5f:
            return group_path in h5f
    except (OSError, KeyError):
        return False
