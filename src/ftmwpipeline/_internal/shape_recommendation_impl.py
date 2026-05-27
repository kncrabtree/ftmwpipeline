"""Shared implementation for the 3-way L/G/V shape-recommendation hook.

Wraps :func:`~ftmwpipeline.fitting.tau_calibration.compute_shape_recommendation`
so the three user-facing interfaces (CLI, Pipeline class, functional API)
share one orchestration layer. Reads the raw FID + canonical Stage 1
settings from a ``.ftmw`` file, runs the 3-way per-bin AICc vote across
exp / gauss / voigt models, writes the verdict to the
``recommended_shape`` attr on every persisted Stage 2b group, and
returns the :class:`ShapeRecommendation` for inspection.

Requires Stage 1 (the active region + frequency trim live there) but
does *not* require either Stage 2b twin to have run -- the
recommendation is computed directly from the raw FID via the same STFT
classifier. It writes to whichever Stage 2b groups exist; if neither is
present the verdict is returned but no attr is stamped (callers can
re-run after :func:`calibrate_tau` / :func:`calibrate_tau_G` if they
want the persisted contract for Stage 5 to fire).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from ..core.settings import FT_PROCESSING_PATH, FTSettings
from ..file_manager import StageDependencyError
from ..fitting.tau_calibration import (
    DEFAULT_N_SEG,
    DEFAULT_RSS_GATE_FACTOR,
    DEFAULT_SHAPE_RECOMMENDATION_PURE_MARGIN,
    DEFAULT_T_SIGMA,
    DEFAULT_TAU_G_BOUND_HI,
    DEFAULT_TAU_G_BOUND_LO,
    DEFAULT_TAU_G_SEEDS,
    DEFAULT_TAU_G_SNR_MIN,
    ShapeRecommendation,
    compute_shape_recommendation,
)
from ..io.stage_fit_settings_serialization import (
    write_stage2b_recommended_shape,
)
from .stage0_impl import load_fid_from_pipeline_impl
from .stage1_impl import _read_settings_layer

logger = logging.getLogger(__name__)

STAGE_NAME = "shape_recommendation"


def _read_canonical_ft_settings(file_path: str) -> FTSettings:
    """Resolve the canonical Stage 1 FT settings the calibration consumes."""
    settings = _read_settings_layer(file_path, FT_PROCESSING_PATH)
    if settings is None:
        raise StageDependencyError(
            STAGE_NAME, ["stage1_complex_ft"], Path(file_path),
        )
    return settings


def recommend_shape_impl(
    file_path: str,
    *,
    n_seg: Optional[int] = None,
    t_sigma: Optional[float] = None,
    tau_max_us: Optional[float] = None,
    rss_gate_factor: Optional[float] = None,
    sigma_time: Optional[float] = None,
    snr_min: Optional[float] = None,
    tau_bound_lo: Optional[float] = None,
    tau_bound_hi: Optional[float] = None,
    tau_G_seeds: Optional[Sequence[float]] = None,
    pure_margin_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Run the 3-way shape recommendation on ``file_path`` and persist it.

    Requires Stage 1 (FT settings + trim) to have completed. Reads the
    raw FID, runs the same STFT classifier the τ calibrations use, fits
    exp / gauss / voigt per contributor bin, computes the SNR-weighted
    majority vote, and stamps the verdict's ``recommended_shape`` onto
    every Stage 2b group present on the file (Lorentzian-twin
    ``stage2b_tau_calibration`` and/or Gaussian-twin
    ``stage2b_tau_G_calibration``). Stage 5's resolver picks the attr
    up automatically as its *recommended* layer.

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
            "Stage 1 canonical FT settings have no frequency trim; the "
            "shape recommendation uses the persisted trim range to match "
            "the user spectrum. Set trim on compute_ft() first."
        )
    trim_lo_mhz, trim_hi_mhz = settings.trim
    sideband = (
        fid.sideband.value if hasattr(fid.sideband, "value") else str(fid.sideband)
    )

    n_seg_v = DEFAULT_N_SEG if n_seg is None else int(n_seg)
    t_sigma_v = DEFAULT_T_SIGMA if t_sigma is None else float(t_sigma)
    rss_gate_v = (
        DEFAULT_RSS_GATE_FACTOR if rss_gate_factor is None else float(rss_gate_factor)
    )
    snr_min_v = DEFAULT_TAU_G_SNR_MIN if snr_min is None else float(snr_min)
    bound_lo_v = (
        DEFAULT_TAU_G_BOUND_LO if tau_bound_lo is None else float(tau_bound_lo)
    )
    bound_hi_v = (
        DEFAULT_TAU_G_BOUND_HI if tau_bound_hi is None else float(tau_bound_hi)
    )
    seeds_v = (
        DEFAULT_TAU_G_SEEDS if tau_G_seeds is None else tuple(tau_G_seeds)
    )
    margin_v = (
        DEFAULT_SHAPE_RECOMMENDATION_PURE_MARGIN
        if pure_margin_threshold is None
        else float(pure_margin_threshold)
    )

    verdict: ShapeRecommendation = compute_shape_recommendation(
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
    write_stage2b_recommended_shape(file_path, verdict.recommended_shape)
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

    return {
        "status": "success",
        "shape_recommendation": verdict,
        "groups_written": groups_written,
    }
