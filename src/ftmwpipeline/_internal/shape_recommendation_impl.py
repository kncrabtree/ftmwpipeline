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
present no attr is stamped (callers can re-run after ``calibrate_tau``
(either shape) if they want the persisted contract for Stage 5 to fire).
Either way the verdict is persisted in the recommendation's own record,
``processing_parameters/stage2b_shape_recommendation``, together with the
knobs and Stage 1 values it used.

Knob configuration follows the four-layer resolver pattern shared with
the τ twins; the optional ``settings=`` / ``preset=`` layer composes against
the same persisted ``processing_parameters/stage2b_tau`` recipe the twins
resolve against (Stage 2b is one stage, one settings recipe, three
consumers). What the recommender used is recorded in its own record, which no
twin writes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from ..contract import CancelToken, EventCallback, Stage
from ..core.tau_calibration_settings import TauCalibrationSettings
from ..file_manager import BadSettingError, requires_pipeline_file
from ..fitting.tau_calibration import (
    ShapeRecommendation,
    compute_shape_recommendation,
)
from ..io.provenance import stamp_stage_epoch_in_file
from ..io.stage_fit_settings_serialization import (
    write_stage2b_recommended_shape,
)
from ..io.tau_calibration_settings_serialization import (
    save_shape_recommendation_record,
    save_tau_calibration_settings_to_h5,
)
from .stage0_impl import load_fid_from_pipeline_impl
from .stage1_impl import persisted_ft_settings
from .tau_settings_resolution import (
    _required_float,
    _required_int,
    resolve_with_preset_and_persisted,
)

logger = logging.getLogger(__name__)

STAGE_NAME = "shape_recommendation"

#: The key the recommendation's analysis epoch is stamped under. Not a tracked
#: stage, but its verdict feeds Stage 3's gap pass and Stage 5's shape, so it
#: records the epoch it was produced under like any stage does.
SHAPE_RECOMMENDATION_EPOCH_KEY = "stage2b_shape_recommendation"


def tau_recommend_summary(result: Mapping[str, Any]) -> Dict[str, Any]:
    """The scalar ``ftmw/run_result@1`` summary of ``tau recommend`` -- also its
    ``StageFinished.summary`` -- from a :func:`recommend_shape_impl` result."""
    rec = result["shape_recommendation"]
    rates = rec.vote_rates
    return {
        "recommended_shape": rec.recommended_shape,
        "vote_rate_exp": rates["exp"],
        "vote_rate_gauss": rates["gauss"],
        "vote_rate_voigt": rates["voigt"],
        "n_contributors": rec.n_contributors,
        "stamped_onto": ", ".join(result.get("groups_written") or ()),
    }


@requires_pipeline_file()
def recommend_shape_impl(
    file_path: str,
    *,
    settings: Optional[TauCalibrationSettings] = None,
    preset: Optional[str] = None,
    events: Optional[EventCallback] = None,
    cancel: Optional[CancelToken] = None,
) -> Dict[str, Any]:
    """Run the 3-way shape recommendation on ``file_path`` and persist it.

    A long operation (``tau recommend``), reported under the ``tau`` stage
    (its ``run_result`` stage): ``StageStarted``, then ``StageFinished`` with
    :func:`tau_recommend_summary` once the verdict is written. It invalidates
    nothing. ``cancel`` is checked before it starts.

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
    the same recipe. The knobs the recommender consumed, the Stage 1 values it
    ran on, the effective tau clip and the verdict are recorded in
    ``processing_parameters/stage2b_shape_recommendation`` -- also when no
    Stage 2b group exists yet. Pass ``settings=`` to drive the recommender from a
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
    from .events import operation_events

    ops = operation_events("tau recommend", events, cancel)
    with ops.stage(Stage.TAU, verb="tau recommend", file_path=file_path) as scope:
        result = run_shape_recommendation(file_path, settings=settings, preset=preset)
        scope.finish(tau_recommend_summary(result))
    return result


def run_shape_recommendation(
    file_path: str,
    *,
    settings: Optional[TauCalibrationSettings] = None,
    preset: Optional[str] = None,
) -> Dict[str, Any]:
    """The body of :func:`recommend_shape_impl`, without the operation's
    events: what ``tau run``'s auto-recommendation calls inside its own
    stage."""
    file_path_obj = Path(file_path)

    explicit = TauCalibrationSettings()
    resolved, preset_name = resolve_with_preset_and_persisted(
        file_path,
        explicit=explicit,
        settings=settings,
        preset=preset,
    )

    fid = load_fid_from_pipeline_impl(file_path)
    # What Stage 1 is read as having used, through the one Stage 1 resolver
    # (the same values Stages 2-5 rebuild the FT from).
    ft_settings = persisted_ft_settings(file_path, STAGE_NAME, float(fid.duration_us))
    sample_dt_us = float(fid.spacing * 1e6)
    start_us, end_us = ft_settings.active_window_us()
    if ft_settings.trim is None:
        raise BadSettingError(
            "stage1.trim",
            "a persisted (min, max) MHz trim range (set trim on compute_ft)",
            None,
            message=(
                "Stage 1 persisted FT settings have no frequency trim; the "
                "shape recommendation uses the persisted trim range to match "
                "the user spectrum. Set trim on compute_ft() first."
            ),
        )
    trim_lo_mhz, trim_hi_mhz = ft_settings.trim
    sideband = (
        fid.sideband.value if hasattr(fid.sideband, "value") else str(fid.sideband)
    )

    stft = resolved.stft
    rec = resolved.recommendation
    n_seg_v = _required_int(stft.n_seg, "stft.n_seg")
    t_sigma_v = _required_float(stft.t_sigma, "stft.t_sigma")
    tau_max_factor_v = _required_float(stft.tau_max_factor, "stft.tau_max_factor")
    rss_gate_v = _required_float(stft.rss_gate_factor, "stft.rss_gate_factor")
    relative_gate_v = _required_float(
        stft.relative_gate_fraction, "stft.relative_gate_fraction"
    )
    snr_min_v = _required_float(rec.snr_min, "recommendation.snr_min")
    bound_lo_v = _required_float(rec.tau_bound_lo, "recommendation.tau_bound_lo")
    bound_hi_v = _required_float(rec.tau_bound_hi, "recommendation.tau_bound_hi")
    margin_v = _required_float(
        rec.pure_margin_threshold, "recommendation.pure_margin_threshold"
    )
    if bound_hi_v <= bound_lo_v:
        raise BadSettingError(
            "stage2b.recommendation.tau_bound_hi",
            f"a number greater than tau_bound_lo ({bound_lo_v})",
            bound_hi_v,
            message=(
                f"tau_bound_hi ({bound_hi_v}) must exceed tau_bound_lo "
                f"({bound_lo_v})"
            ),
        )
    n_active = min(int(round(end_us / sample_dt_us)), len(fid.data)) - max(
        int(round(start_us / sample_dt_us)), 0
    )
    if n_active < 4 * int(n_seg_v):
        raise BadSettingError(
            "stage2b.stft.n_seg",
            f"an integer <= {n_active // 4} (4 samples per segment in the "
            f"{n_active}-sample active region)",
            n_seg_v,
            message=(
                f"active region has too few samples ({n_active}) for "
                f"n_seg={n_seg_v}"
            ),
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
        tau_max_factor=tau_max_factor_v,
        rss_gate_factor=rss_gate_v,
        relative_gate_fraction=relative_gate_v,
        sigma_x_full=stft.sigma_x_full,
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
            "Shape recommendation computed and recorded on %s, but neither "
            "Stage 2b group is present; Stage 5's resolver will not see the "
            "verdict until a τ calibration is run.",
            file_path_obj,
        )

    # Persist the resolved Stage 2b settings as the recipe for the next run --
    # recommend_shape is one of three consumers that resolve against it.
    save_tau_calibration_settings_to_h5(
        file_path,
        resolved,
        preset_name=preset_name,
    )
    # Record what this recommendation used and decided, in its own record.
    save_shape_recommendation_record(
        file_path,
        resolved,
        consumed={
            "start_us": start_us,
            "end_us": end_us,
            "trim_lo_mhz": float(trim_lo_mhz),
            "trim_hi_mhz": float(trim_hi_mhz),
        },
        tau_max_us=verdict.tau_max_us,
        recommended_shape=verdict.recommended_shape,
        vote_rates=verdict.vote_rates,
        preset_name=preset_name,
    )
    # Recording the epoch is part of producing the recommendation; a stamp
    # that cannot be written raises.
    stamp_stage_epoch_in_file(file_path, SHAPE_RECOMMENDATION_EPOCH_KEY)

    # The verdict reaches the science only through its consumers, which record
    # the shape they took, so a new verdict invalidates no stage.
    return {
        "status": "success",
        "shape_recommendation": verdict,
        "groups_written": groups_written,
        "invalidated": [],
    }
