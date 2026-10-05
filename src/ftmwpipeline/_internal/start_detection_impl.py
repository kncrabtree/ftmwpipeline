"""File-bound orchestration for data-driven start-time detection.

Runs on the raw Stage 0 FID (no Stage 1 dependency -- detection *informs* the
Stage 1 ``start_us``). Resolves the integration band from the persisted Stage 1
frequency trim when available, runs
:func:`ftmwpipeline.preprocessing.start_detection.detect_start_time`, and
(optionally) stamps the recommended ``start_us`` into the Stage 0
``recommended_processing`` layer so a first ``compute_ft`` with no explicit
``start_us`` inherits it through the resolution chain.

Once Stage 1 has persisted its record, the stamp no longer changes what Stage 1
is read as having used (the persisted record is authoritative; see
``stage1_impl.resolve_ft_settings_h5``). The recommendation is still stored and
shown; adopting it is an explicit ``ft run --start-us`` re-run, which
invalidates every stage built on the old spectrum.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple

from ..contract import CancelToken, EventCallback
from ..core.data_structures import ChirpWindow
from ..core.settings import FT_PROCESSING_PATH, RECOMMENDED_PATH
from ..core.start_detection_settings import StartDetectionSettings
from ..file_manager import canonical_invalidated
from ..io.stage_fit_settings_serialization import (
    read_recommended_chirp_window,
    read_recommended_start_detection,
    write_recommended_start_detection,
)
from ..preprocessing.start_detection import StartDetectionResult, detect_start_time
from .atomic import atomic_write, h5open
from .stage0_impl import load_fid_from_pipeline_impl
from .stage1_impl import (
    _read_settings_layer,
    _resolve_settings,
    ft_record_is_authoritative,
    write_recommended_ft_params,
)

if TYPE_CHECKING:
    from .events import StageScope

logger = logging.getLogger(__name__)

# Maximum allowed absolute difference between a declared chirp_end and the
# detector's estimate before a cross-check warning is emitted (µs).
_CHIRP_END_DISAGREE_THRESHOLD_US = 0.5


def _resolve_band(
    file_path: str,
    settings: StartDetectionSettings,
) -> Optional[Tuple[float, float]]:
    """Integration band: explicit override > the Stage 1 trim.

    The Stage 1 trim comes from the one Stage 1 resolver: the persisted trim,
    falling through to the recommended one only before Stage 1 has written an
    authoritative record. Returns ``None`` (integrate the full positive
    spectrum) when no band can be resolved -- the chirp collapse dominates
    Σ|FT| either way, but a trim band sharpens the floor.
    """
    if settings.band_min_mhz is not None and settings.band_max_mhz is not None:
        return (settings.band_min_mhz, settings.band_max_mhz)
    return _resolve_settings(file_path, None).trim


def start_run_summary(out: Mapping[str, Any]) -> Dict[str, Any]:
    """The scalar ``ftmw/run_result@1`` summary of ``start run`` -- also its
    ``StageFinished.summary`` -- from a :func:`detect_start_time_impl` result."""
    r = out["start_detection"]
    band = (
        f"{r.band_mhz[0]:.0f}-{r.band_mhz[1]:.0f} MHz"
        if r.band_mhz is not None
        else "full spectrum"
    )
    return {
        "integration_band": band,
        "chirp_detected": bool(r.chirp_detected),
        "plateau_floor_ratio": None if not r.floor else r.plateau / r.floor,
        "chirp_end_declared_us": out["chirp_end_declared_us"],
        "chirp_end_detected_us": out["chirp_end_detected_us"],
        "chirp_end_us": r.chirp_end_us,
        "declaration_used": bool(out["declaration_used"]),
        "start_us": out["start_us"],
        "stamped": bool(out["stamped"]),
    }


def detect_start_time_impl(
    file_path: str,
    *,
    settings: Optional[StartDetectionSettings] = None,
    stamp: bool = True,
    events: Optional[EventCallback] = None,
    cancel: Optional[CancelToken] = None,
) -> Dict[str, Any]:
    """Detect a good ``start_us`` from the FID and (optionally) stamp it.

    A long operation (``start run``). Start detection stamps a recommendation
    on the data stage rather than running a stage of its own, so its events
    carry ``stage: null``: ``StageStarted``, ``Invalidated`` (when the stamp
    moved a pre-provenance Stage 1 record's settings), then ``StageFinished``
    with :func:`start_run_summary`. ``cancel`` is checked before it starts.

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
        "declaration_used": bool, "invalidated": list}``. ``invalidated``
        (also on the result's own field) is empty unless the stamp moved the
        settings a pre-provenance Stage 1 record falls through to (see
        :func:`~.stage1_impl.write_recommended_ft_params`).
    """
    from .events import operation_events

    ops = operation_events("start run", events, cancel)
    with ops.stage(None, verb="start run", file_path=file_path) as scope:
        with atomic_write(file_path):
            out = _detect_start_time(
                file_path, settings=settings, stamp=stamp, events=scope
            )
        scope.finish(start_run_summary(out))
    return out


def _detect_start_time(
    file_path: str,
    *,
    settings: Optional[StartDetectionSettings],
    stamp: bool,
    events: "StageScope",
) -> Dict[str, Any]:
    """The body of :func:`detect_start_time_impl` (inside its scope)."""
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
    invalidated: List[str] = []
    can_stamp = declaration_used or result.chirp_detected
    if stamp and can_stamp:
        invalidated = write_recommended_ft_params(
            file_path, {"start_us": float(final_start_us)}, events=events
        )
        stamped = True
        _note_stage1_unaffected(file_path, float(final_start_us))
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

    canonical = canonical_invalidated(invalidated)
    return {
        "status": "success",
        "start_detection": replace(effective_result, invalidated=canonical),
        "start_us": float(final_start_us),
        "stamped": stamped,
        "invalidated": list(canonical),
        "chirp_end_declared_us": (
            declared.chirp_end_us if declared is not None else None
        ),
        "chirp_end_detected_us": float(result.chirp_end_us),
        "declaration_used": declaration_used,
    }


# Two Stage 0 -> Stage 1 layer values agree (the recommendation was inherited
# unchanged) unless they differ by more than floating-point copy noise.
_INHERITED_TOLERANCE_US = 1e-6


def stage1_start_in_force(file_path: str) -> Tuple[bool, Optional[float]]:
    """``(authoritative, start_us)`` of the file's Stage 1 record.

    ``authoritative`` is ``True`` once Stage 1 has persisted the record every
    stage reads; a new recommended ``start_us`` then no longer reaches it.
    ``start_us`` is that record's persisted start (``None`` when it records
    none, or when there is no authoritative record).
    """
    with h5open(file_path, "r") as h5f:
        if not ft_record_is_authoritative(h5f):
            return False, None
    persisted = _read_settings_layer(file_path, FT_PROCESSING_PATH)
    return True, persisted.start_us if persisted is not None else None


def stage1_uses_start(current: Optional[float], start_us: float) -> bool:
    """Whether a persisted Stage 1 ``start_us`` is ``start_us`` (to copy noise)."""
    return current is not None and abs(current - start_us) < _INHERITED_TOLERANCE_US


def stamped_start_note(file_path: str, start_us: float) -> str:
    """What a just-stamped recommended ``start_us`` changes, in one sentence.

    Before Stage 1 has persisted its record, a later ``ft run`` with no
    explicit ``--start-us`` inherits the recommendation. Once it has, Stage 1's
    record is authoritative: the recommendation is stored but changes nothing
    until ``ft run --start-us`` adopts it.
    """
    authoritative, current = stage1_start_in_force(file_path)
    if not authoritative:
        return "A later 'ft run' with no explicit --start-us will inherit it."
    if stage1_uses_start(current, start_us):
        return "Stage 1 already runs with this start_us."
    shown = "unset" if current is None else f"{current:.3f} us"
    return (
        f"Stage 1 has already run (start_us = {shown}) and keeps its start; "
        f"to adopt the new value, re-run 'ft run {file_path} --start-us "
        f"{start_us:.3f}'."
    )


def _note_stage1_unaffected(file_path: str, recommended_start_us: float) -> None:
    """Say so when a new recommendation will not reach an existing Stage 1 run.

    After Stage 1 has persisted an authoritative record, the recommended layer
    no longer feeds it: every stage keeps reading the start Stage 1 ran with.
    That is deliberate (no stage can silently move under results computed from
    the old spectrum), but a user who just ran ``start run`` should hear that
    adopting the new value is an explicit Stage 1 re-run.
    """
    authoritative, current = stage1_start_in_force(file_path)
    if not authoritative or stage1_uses_start(current, recommended_start_us):
        return
    logger.warning(
        "Stage 1 has already run with start_us = %s us; the new recommended "
        "start_us = %.3f us is stored but does not change it. To adopt it, "
        "re-run 'ft run --start-us %.3f' (this invalidates every stage built "
        "on the current spectrum).",
        current,
        recommended_start_us,
        recommended_start_us,
    )


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
      unset (full record used from t = 0; an authoritative Stage 1 record
      spells that as ``start_us = 0.0``).
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
    with h5open(file_path, "r") as h5f:
        authoritative = ft_record_is_authoritative(h5f)
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

    if authoritative and effective_start is not None:
        # An authoritative record spells "no windowing" as a concrete 0.0, so
        # no recommendation inherits as a start of 0.0.
        inherited_start = 0.0 if recommended_start is None else recommended_start
        inherited = abs(effective_start - inherited_start) < _INHERITED_TOLERANCE_US
    else:
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
