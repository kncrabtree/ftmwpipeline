"""Shared implementation for the 3-way L/G/V shape-recommendation hook.

Wraps :func:`~ftmwpipeline.fitting.tau_calibration.compute_shape_recommendation`
so the three user-facing interfaces (CLI, Pipeline class, functional API)
share one orchestration layer. Reads the raw FID + persisted Stage 1
settings from a ``.ftmw`` file, runs the 3-way per-bin AICc vote across
exp / gauss / voigt models, writes the verdict to the
``recommended_shape`` attr on every persisted Stage 2b group, and
returns the :class:`ShapeRecommendation` for inspection.

Requires Stage 1 (the active region + frequency trim live there) but
does *not* require either Stage 2b twin to have run -- the
recommendation is computed directly from the raw FID via the same STFT
classifier. It writes to whichever Stage 2b groups exist; if neither is
present the verdict is returned but no attr is stamped (callers can
re-run after ``calibrate_tau`` (either shape) if they
want the persisted contract for Stage 5 to fire).

Knob configuration follows the four-layer resolver pattern shared with
the τ twins; the optional ``settings=`` / ``preset=`` layer composes against
the same persisted ``processing_parameters/stage2b_tau`` settings record that
drives the recommender (Stage 2b is one stage, one settings block, three
consumers).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..core.settings import FT_PROCESSING_PATH, FTSettings
from ..core.tau_calibration_settings import TauCalibrationSettings
from ..file_manager import StageDependencyError
from ..fitting.tau_calibration import (
    ShapeRecommendation,
    compute_shape_recommendation,
)
from ..io.stage_fit_settings_serialization import (
    write_stage2b_recommended_shape,
)
from ..io.tau_calibration_settings_serialization import (
    save_tau_calibration_settings_to_h5,
)
from .stage0_impl import load_fid_from_pipeline_impl
from .stage1_impl import _read_settings_layer
from .tau_settings_resolution import (
    _required_float,
    _required_int,
    resolve_with_preset_and_persisted,
)

logger = logging.getLogger(__name__)

STAGE_NAME = "shape_recommendation"


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


def recommend_shape_impl(
    file_path: str,
    *,
    settings: Optional[TauCalibrationSettings] = None,
    preset: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the 3-way shape recommendation on ``file_path`` and persist it.

    Requires Stage 1 (FT settings + trim) to have completed. Reads the
    raw FID, runs the same STFT classifier the τ calibrations use, fits
    exp / gauss / voigt per contributor bin, computes the SNR-weighted
    majority vote, and stamps the verdict's ``recommended_shape`` onto
    every Stage 2b group present on the file (Lorentzian-twin
    ``stage2b_tau_calibration`` and/or Gaussian-twin
    ``stage2b_tau_G_calibration``). Stage 5's resolver picks the attr up
    automatically as its *recommended* layer.

    Settings resolve through the chain (``settings`` / ``preset`` > persisted >
    hard default); the resolved settings are stamped to
    ``processing_parameters/stage2b_tau`` so a follow-up no-arg call inherits
    the same recipe. Pass ``settings=`` to drive the recommender from a
    :class:`TauCalibrationSettings` dataclass, or ``preset=NAME_OR_PATH`` to
    load from packaged YAML; they may be combined. A ``settings`` bundle is the
    explicit override that outranks the persisted record, while a ``preset``
    seeds only the fields neither the explicit layer nor the persisted record
    has fixed (the persisted record outranks the preset, per D11).

    Returns
    -------
    dict
        ``{"status": "success", "shape_recommendation": ShapeRecommendation,
        "groups_written": list[str]}``. ``groups_written`` lists the
        Stage 2b group paths whose attr was updated -- empty if no
        Stage 2b twin has been calibrated yet (the verdict is still
        returned, but Stage 5 won't pick it up until a calibration is
        present).
    """
    file_path_obj = Path(file_path)

    explicit = TauCalibrationSettings()
    resolved, preset_name = resolve_with_preset_and_persisted(
        file_path,
        explicit=explicit,
        settings=settings,
        preset=preset,
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
    if ft_settings.trim is None:
        raise ValueError(
            "Stage 1 persisted FT settings have no frequency trim; the "
            "shape recommendation uses the persisted trim range to match "
            "the user spectrum. Set trim on compute_ft() first."
        )
    trim_lo_mhz, trim_hi_mhz = ft_settings.trim
    sideband = (
        fid.sideband.value if hasattr(fid.sideband, "value") else str(fid.sideband)
    )

    stft = resolved.stft
    rec = resolved.recommendation
    n_seg_v = _required_int(stft.n_seg, "stft.n_seg")
    t_sigma_v = _required_float(stft.t_sigma, "stft.t_sigma")
    rss_gate_v = _required_float(stft.rss_gate_factor, "stft.rss_gate_factor")
    snr_min_v = _required_float(rec.snr_min, "recommendation.snr_min")
    bound_lo_v = _required_float(rec.tau_bound_lo, "recommendation.tau_bound_lo")
    bound_hi_v = _required_float(rec.tau_bound_hi, "recommendation.tau_bound_hi")
    margin_v = _required_float(
        rec.pure_margin_threshold, "recommendation.pure_margin_threshold"
    )
    if rec.tau_G_seeds is None:
        raise AssertionError(
            "resolved TauCalibrationSettings.recommendation.tau_G_seeds is "
            "None; missing hard default"
        )
    seeds_v = tuple(float(v) for v in rec.tau_G_seeds)

    verdict: ShapeRecommendation = compute_shape_recommendation(
        np.asarray(fid.data, dtype=float),
        sample_dt_us,
        start_us=start_us,
        end_us=end_us,
        probe_freq_mhz=float(fid.probe_freq_mhz),
        sideband=sideband,
        trim_lo_mhz=float(trim_lo_mhz),
        trim_hi_mhz=float(trim_hi_mhz),
        sigma_time=stft.sigma_time,
        n_seg=n_seg_v,
        t_sigma=t_sigma_v,
        tau_max_us=stft.tau_max_us,
        rss_gate_factor=rss_gate_v,
        snr_min=snr_min_v,
        tau_bound_lo=bound_lo_v,
        tau_bound_hi=bound_hi_v,
        tau_G_seeds=seeds_v,
        pure_margin_threshold=margin_v,
    )

    # Stamp the verdict onto every Stage 2b group that exists. The
    # ``__None__`` sentinel is written when the verdict's
    # ``recommended_shape`` is None so the attr explicitly reflects
    # "no clear winner" rather than carrying a stale prior value.
    import h5py

    groups_written: list[str] = []
    write_stage2b_recommended_shape(
        file_path,
        verdict.recommended_shape,
        vote_rates=verdict.vote_rates,
    )
    with h5py.File(file_path, "r") as h5f:
        for path in (
            "stage2b_tau_calibration",
            "stage2b_tau_G_calibration",
        ):
            if path in h5f:
                groups_written.append(path)
    if not groups_written:
        logger.warning(
            "Shape recommendation computed on %s but neither Stage 2b "
            "group is present; the verdict is returned but Stage 5's "
            "resolver will not see it until a τ calibration is run.",
            file_path_obj,
        )

    # Persist the resolved Stage 2b settings as the persisted record
    # for this run -- recommend_shape is one of three consumers that
    # share the same settings block.
    save_tau_calibration_settings_to_h5(
        file_path,
        resolved,
        preset_name=preset_name,
    )

    return {
        "status": "success",
        "shape_recommendation": verdict,
        "groups_written": groups_written,
    }
