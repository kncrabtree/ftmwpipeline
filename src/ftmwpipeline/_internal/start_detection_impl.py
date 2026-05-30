"""File-bound orchestration for data-driven start-time detection.

Runs on the raw Stage 0 FID (no Stage 1 dependency -- detection *informs* the
Stage 1 ``start_us``). Resolves the integration band from the canonical Stage 1
frequency trim when available, runs
:func:`ftmwpipeline.preprocessing.start_detection.detect_start_time`, and
(optionally) stamps the recommended ``start_us`` into the Stage 0
``recommended_processing`` layer so a subsequent ``compute_ft`` with no explicit
``start_us`` inherits it through the existing resolution chain
(``explicit > persisted > recommended``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..core.settings import FT_PROCESSING_PATH, RECOMMENDED_PATH
from ..core.start_detection_settings import StartDetectionSettings
from ..file_manager import update_processing_parameters
from ..preprocessing.start_detection import StartDetectionResult, detect_start_time
from .stage0_impl import load_fid_from_pipeline_impl
from .stage1_impl import _read_settings_layer

logger = logging.getLogger(__name__)


def _resolve_band(
    file_path: str,
    settings: StartDetectionSettings,
) -> Optional[Tuple[float, float]]:
    """Integration band: explicit override > persisted trim > recommended trim.

    Returns ``None`` (integrate the full positive spectrum) when no band can be
    resolved -- the chirp collapse dominates Σ|FT| either way, but a trim band
    sharpens the floor.
    """
    if settings.band_min_mhz is not None and settings.band_max_mhz is not None:
        return (settings.band_min_mhz, settings.band_max_mhz)
    for group_path in (FT_PROCESSING_PATH, RECOMMENDED_PATH):
        ft_settings = _read_settings_layer(file_path, group_path)
        if ft_settings is not None and ft_settings.trim is not None:
            return ft_settings.trim
    return None


def detect_start_time_impl(
    file_path: str,
    *,
    settings: Optional[StartDetectionSettings] = None,
    stamp: bool = True,
) -> Dict[str, Any]:
    """Detect a good ``start_us`` from the FID and (optionally) stamp it.

    Parameters
    ----------
    file_path :
        Path to a ``.ftmw`` file with Stage 0 (FID) imported.
    settings :
        Detection knobs; defaults to :class:`StartDetectionSettings`.
    stamp :
        When ``True`` (default), write the recommended ``start_us`` to the
        Stage 0 ``recommended_processing`` layer so later ``compute_ft`` calls
        inherit it. When ``False``, only return the result.

    Returns
    -------
    dict
        ``{"status": "success", "start_detection": StartDetectionResult,
        "start_us": float, "stamped": bool}``.
    """
    settings = settings or StartDetectionSettings()
    fid = load_fid_from_pipeline_impl(file_path)
    band = _resolve_band(file_path, settings)

    result: StartDetectionResult = detect_start_time(fid, band=band, settings=settings)

    if not result.chirp_detected:
        logger.warning(
            "Start detection found no chirp collapse (plateau/floor = %.1f < %.1f); "
            "start_us could not be inferred from the data.",
            result.plateau / result.floor if result.floor else float("inf"),
            settings.min_chirp_drop_ratio,
        )

    stamped = False
    if stamp and result.chirp_detected:
        update_processing_parameters(file_path, {"start_us": float(result.start_us)})
        stamped = True
        logger.info(
            "Stamped recommended start_us = %.3f us (chirp_end %.3f + margin %.3f) "
            "to %s.",
            result.start_us,
            result.chirp_end_us,
            settings.guard_margin_us,
            Path(file_path).name,
        )

    return {
        "status": "success",
        "start_detection": result,
        "start_us": float(result.start_us),
        "stamped": stamped,
    }


__all__ = ["detect_start_time_impl"]
