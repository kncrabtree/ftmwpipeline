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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ..core.data_structures import ChirpWindow
from ..core.settings import FT_PROCESSING_PATH, RECOMMENDED_PATH
from ..core.start_detection_settings import StartDetectionSettings
from ..file_manager import update_processing_parameters
from ..io.stage_fit_settings_serialization import (
    read_recommended_chirp_window,
    read_recommended_start_detection,
    write_recommended_start_detection,
)
from ..preprocessing.start_detection import StartDetectionResult, detect_start_time
from .stage0_impl import load_fid_from_pipeline_impl
from .stage1_impl import _read_settings_layer

logger = logging.getLogger(__name__)

# Maximum allowed absolute difference between a declared chirp_end and the
# detector's estimate before a cross-check warning is emitted (µs).
_CHIRP_END_DISAGREE_THRESHOLD_US = 0.5


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

    When the file carries a declared chirp-window (persisted at import time),
    the declaration governs the recommended start:
    ``declared chirp_end + margin`` where margin is the declared
    ``start_margin_us`` when set, else ``settings.guard_margin_us``.
    The sweep detector still runs as a cross-check; a ``WARNING`` is logged
    when the two chirp-end estimates disagree by more than
    :data:`_CHIRP_END_DISAGREE_THRESHOLD_US` or the detector finds no chirp.

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
        "start_us": float, "stamped": bool,
        "chirp_end_declared_us": float | None,
        "chirp_end_detected_us": float,
        "declaration_used": bool}``.
    """
    settings = settings or StartDetectionSettings()
    fid = load_fid_from_pipeline_impl(file_path)
    band = _resolve_band(file_path, settings)

    # Read any import-time declared chirp-window.
    declared = read_recommended_chirp_window(file_path)

    result: StartDetectionResult = detect_start_time(fid, band=band, settings=settings)

    declaration_used = False
    final_start_us: float

    if declared is not None:
        # Declaration governs: use declared chirp_end + margin.
        margin = (
            declared.start_margin_us
            if declared.start_margin_us is not None
            else settings.guard_margin_us
        )
        final_start_us = declared.chirp_end_us + margin
        declaration_used = True

        # Cross-check: warn when the detector disagrees with the declaration.
        threshold = max(
            _CHIRP_END_DISAGREE_THRESHOLD_US, 2.0 * settings.guard_margin_us
        )
        if not result.chirp_detected:
            logger.warning(
                "Declared chirp_end=%.3f us but the sweep detector found no "
                "chirp collapse (plateau/floor = %.1f < %.1f). "
                "Using declaration.",
                declared.chirp_end_us,
                result.plateau / result.floor if result.floor else float("inf"),
                settings.min_chirp_drop_ratio,
            )
        elif abs(result.chirp_end_us - declared.chirp_end_us) > threshold:
            logger.warning(
                "Declared chirp_end=%.3f us but sweep detector found %.3f us "
                "(difference %.3f us > threshold %.3f us). "
                "Using declaration; verify instrument config.",
                declared.chirp_end_us,
                result.chirp_end_us,
                abs(result.chirp_end_us - declared.chirp_end_us),
                threshold,
            )
    else:
        # No declaration: fall back to sweep detector behavior.
        if not result.chirp_detected:
            logger.warning(
                "Start detection found no chirp collapse (plateau/floor = %.1f < %.1f); "
                "start_us could not be inferred from the data.",
                result.plateau / result.floor if result.floor else float("inf"),
                settings.min_chirp_drop_ratio,
            )
        final_start_us = result.start_us

    # Build a result whose start_us reflects the effective recommendation so
    # all three interfaces see a consistent value.
    effective_result = replace(
        result,
        start_us=final_start_us,
        chirp_end_declared_us=declared.chirp_end_us if declared is not None else None,
        declaration_used=declaration_used,
    )

    if stamp:
        # Always record the settings actually given to the sweep detector and
        # what it actually found -- independent of whether a declaration
        # governs the final start_us -- so a report can replay this exact
        # sweep later instead of re-running it with guessed default knobs.
        write_recommended_start_detection(file_path, settings, result)

    stamped = False
    can_stamp = declaration_used or result.chirp_detected
    if stamp and can_stamp:
        update_processing_parameters(file_path, {"start_us": float(final_start_us)})
        stamped = True
        if declaration_used and declared is not None:
            _margin: float = margin if margin is not None else settings.guard_margin_us
            logger.info(
                "Stamped declaration-derived start_us = %.3f us "
                "(declared chirp_end %.3f + margin %.3f) to %s.",
                final_start_us,
                declared.chirp_end_us,
                _margin,
                Path(file_path).name,
            )
        else:
            logger.info(
                "Stamped recommended start_us = %.3f us (chirp_end %.3f + margin %.3f) "
                "to %s.",
                final_start_us,
                result.chirp_end_us,
                settings.guard_margin_us,
                Path(file_path).name,
            )

    return {
        "status": "success",
        "start_detection": effective_result,
        "start_us": float(final_start_us),
        "stamped": stamped,
        "chirp_end_declared_us": (
            declared.chirp_end_us if declared is not None else None
        ),
        "chirp_end_detected_us": float(result.chirp_end_us),
        "declaration_used": declaration_used,
    }


# Two Stage 0 -> Stage 1 layer values agree (the recommendation was inherited
# unchanged) unless they differ by more than floating-point copy noise.
_INHERITED_TOLERANCE_US = 1e-6


@dataclass(frozen=True)
class StartProvenance:
    """How the effective Stage 1 ``start_us`` was arrived at, resolved purely
    from the persisted record (no recompute of the fit).

    ``source`` is one of:

    * ``"manual"`` -- the persisted, resolved Stage 1 ``start_us`` differs
      from the Stage 0 recommendation (or there was no recommendation at all):
      the user set it explicitly (``ft process --start-us`` / ``ft run``).
    * ``"declared"`` -- inherited from an import-time instrument declaration:
      a declared chirp window, or (when no chirp-window metadata is on
      record) a loader-set experimenter start such as Blackchirp's
      ``FidStartUs``.
    * ``"auto_detected"`` -- inherited from the Σ|FT|-vs-start sweep detector
      (``start run``), which found a chirp collapse.
    * ``"no_chirp_found"`` -- the sweep detector ran but found no chirp
      collapse; the recommendation could not be inferred and start_us stayed
      unset (full record used from t = 0).
    * ``"none"`` -- nothing was ever recommended, detected, or set.

    ``detection_settings`` / ``detection_band_mhz`` are present whenever a
    :class:`~ftmwpipeline.preprocessing.start_detection.StartDetectionRecord`
    is on file (i.e. ``start run`` has been invoked at least once), even when
    a chirp-window declaration governs the final ``start_us`` -- the sweep
    still ran as a cross-check, and its exact settings are enough to replay
    it for a diagnostic figure.
    """

    source: str
    start_us: Optional[float]
    recommended_start_us: Optional[float]
    chirp_end_us: Optional[float]
    chirp_start_us: Optional[float]
    guard_margin_us: Optional[float]
    chirp_detected: Optional[bool]
    detection_settings: Optional[StartDetectionSettings]
    detection_band_mhz: Optional[Tuple[float, float]]
    plateau: Optional[float]
    floor: Optional[float]


def resolve_start_provenance(file_path: str) -> StartProvenance:
    """Resolve how ``start_us`` was chosen for ``file_path``, from persisted state only.

    Reads the resolved Stage 1 ``ft_processing`` layer (the value every stage
    after Stage 0 actually uses), the Stage 0 ``recommended_processing``
    layer, the import-time chirp-window declaration (if any), and the
    start-detection provenance record (if ``start run`` has ever been
    invoked) -- then classifies the result per :class:`StartProvenance`.
    """
    ft = _read_settings_layer(file_path, FT_PROCESSING_PATH)
    rec = _read_settings_layer(file_path, RECOMMENDED_PATH)
    effective_start = ft.start_us if ft is not None else None
    recommended_start = rec.start_us if rec is not None else None

    declared: Optional[ChirpWindow] = read_recommended_chirp_window(file_path)
    detection = read_recommended_start_detection(file_path)

    chirp_end_us: Optional[float]
    chirp_start_us: Optional[float]
    guard_margin_us: Optional[float]
    chirp_detected: Optional[bool] = (
        detection.chirp_detected if detection is not None else None
    )

    if declared is not None:
        stage0_source = "declared"
        chirp_end_us = declared.chirp_end_us
        chirp_start_us = declared.chirp_start_us
        margin = declared.start_margin_us
        if margin is None:
            margin = (
                detection.settings.guard_margin_us
                if detection is not None
                else StartDetectionSettings().guard_margin_us
            )
        guard_margin_us = margin
    elif detection is not None:
        stage0_source = (
            "auto_detected" if detection.chirp_detected else "no_chirp_found"
        )
        chirp_end_us = detection.chirp_end_us if detection.chirp_detected else None
        chirp_start_us = None
        guard_margin_us = detection.settings.guard_margin_us
    elif recommended_start is not None:
        # A recommendation exists but neither a chirp-window declaration nor a
        # start-detection record explains it -- a loader-set experimenter
        # start with no chirp-window metadata (or a legacy file).
        stage0_source = "declared"
        chirp_end_us = None
        chirp_start_us = None
        guard_margin_us = None
    else:
        stage0_source = "none"
        chirp_end_us = None
        chirp_start_us = None
        guard_margin_us = None

    inherited = (recommended_start is None and effective_start is None) or (
        recommended_start is not None
        and effective_start is not None
        and abs(effective_start - recommended_start) < _INHERITED_TOLERANCE_US
    )
    source = stage0_source if inherited else "manual"

    return StartProvenance(
        source=source,
        start_us=effective_start,
        recommended_start_us=recommended_start,
        chirp_end_us=chirp_end_us,
        chirp_start_us=chirp_start_us,
        guard_margin_us=guard_margin_us,
        chirp_detected=chirp_detected,
        detection_settings=detection.settings if detection is not None else None,
        detection_band_mhz=detection.band_mhz if detection is not None else None,
        plateau=detection.plateau if detection is not None else None,
        floor=detection.floor if detection is not None else None,
    )


__all__ = ["detect_start_time_impl", "StartProvenance", "resolve_start_provenance"]
