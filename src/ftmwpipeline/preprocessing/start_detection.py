"""Data-driven FID start-time detection.

CP-FTMW experiments record the excitation chirp and the switch-bounce ringdown
that follows it before the molecular free-induction decay. Analysis should start
after both. This module infers a good ``start_us`` directly from the FID, with
no metadata, by sweeping the window start time and integrating the FT magnitude
over the active band:

  * While the start cut still includes any of the broadband chirp, Σ|FT| sits on
    a high plateau; the instant the cut clears the chirp it collapses ~2-3
    decades to a floor. The collapse point is the **chirp end**, found robustly
    by a level crossing.
  * The empirically-good start sits a fixed guard margin past the chirp end --
    the switch-bounce ringdown settling time. ``start_us = chirp_end +
    guard_margin_us`` is the primary recommendation.

See :class:`~ftmwpipeline.core.start_detection_settings.StartDetectionSettings`
for the knobs and :mod:`ftmwpipeline._internal.start_detection_impl` for the
file-bound orchestration (band resolution + recommended-``start_us`` stamping).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..core.data_structures import FID
from ..core.start_detection_settings import StartDetectionSettings


@dataclass(frozen=True)
class StartDetectionResult:
    """Outcome of :func:`detect_start_time` (sweep-only) or the file-bound
    orchestration in :mod:`ftmwpipeline._internal.start_detection_impl`.

    Attributes
    ----------
    start_us :
        Effective recommended FID window start.  When a chirp-window
        declaration is present this is the declaration-derived value
        (``declared chirp_end + margin``); otherwise it is the
        detector-derived ``chirp_end_us + guard_margin_us``.
    chirp_end_us :
        Start time at which Σ|FT| collapses to the post-chirp floor
        (sweep-detector result; may differ from ``chirp_end_declared_us``).
    chirp_detected :
        Whether a chirp collapse (plateau/floor ratio above the configured
        minimum) was present. When ``False`` the start time could not be
        inferred and ``start_us`` falls back to ``guard_margin_us``.
    floor :
        Robust deep-tail Σ|FT| floor.
    plateau :
        Σ|FT| on the pre-chirp plateau.
    band_mhz :
        Integration band actually used (``None`` = full positive spectrum).
    starts_us, sum_magnitude :
        The full sweep, retained for visualization.
    chirp_end_declared_us :
        Declared chirp end (µs) from the import-time chirp-window record,
        or ``None`` when no declaration is present.
    declaration_used :
        ``True`` when ``start_us`` was derived from a chirp-window
        declaration rather than from the sweep detector.
    """

    start_us: float
    chirp_end_us: float
    chirp_detected: bool
    floor: float
    plateau: float
    band_mhz: Optional[Tuple[float, float]]
    starts_us: np.ndarray
    sum_magnitude: np.ndarray
    # Declaration fields — absent on pure detector results (back-compat default).
    chirp_end_declared_us: Optional[float] = None
    declaration_used: bool = False


@dataclass(frozen=True)
class StartDetectionRecord:
    """Persisted settings + sweep outcome from the last ``start run`` invocation.

    Written by :func:`ftmwpipeline._internal.start_detection_impl.detect_start_time_impl`
    whenever it stamps (``stage0_fid_data/recommended_start_detection``), and
    read back by ``resolve_start_provenance`` so a report or visualization can
    replay the exact sweep that produced the recommendation instead of
    re-running it with guessed defaults -- the sweep arrays themselves are not
    persisted (cheap and deterministic to regenerate from the raw FID plus
    these settings).
    """

    settings: StartDetectionSettings
    chirp_end_us: float
    chirp_detected: bool
    floor: float
    plateau: float
    band_mhz: Optional[Tuple[float, float]]


def detect_start_time(
    fid: FID,
    *,
    band: Optional[Tuple[float, float]] = None,
    settings: Optional[StartDetectionSettings] = None,
) -> StartDetectionResult:
    """Infer a good ``start_us`` from the FID via the Σ|FT|-vs-start sweep.

    Parameters
    ----------
    fid :
        Raw FID to analyze.
    band :
        ``(min_mhz, max_mhz)`` integration band. When ``None`` and the settings
        carry no band override, the full positive spectrum is integrated.
    settings :
        Detection knobs; defaults to :class:`StartDetectionSettings`.

    Returns
    -------
    StartDetectionResult
    """
    settings = settings or StartDetectionSettings()
    if settings.band_min_mhz is not None and settings.band_max_mhz is not None:
        band = (settings.band_min_mhz, settings.band_max_mhz)

    # Sweep start times up to the configured max, capped well inside the FID.
    sweep_max = min(settings.sweep_max_us, max(fid.duration_us - 1.0, settings.step_us))
    starts = np.round(np.arange(0.0, sweep_max + 1e-9, settings.step_us), 6)
    if starts.size < 5:
        raise ValueError(
            "FID too short for start detection "
            f"(duration {fid.duration_us:.3f} us, step {settings.step_us} us)"
        )

    summag = np.empty(starts.shape, dtype=float)
    for i, s in enumerate(starts):
        preprocessed = fid.preprocess(start_us=float(s))
        spectrum, freqs = preprocessed.compute_fft()
        if band is not None:
            mask = (freqs >= band[0]) & (freqs <= band[1])
            mag = np.abs(spectrum[mask])
        else:
            mag = np.abs(spectrum)
        summag[i] = float(mag.sum())

    # Robust floor (deep tail) and pre-chirp plateau.
    tail = starts > starts.max() - settings.floor_tail_us
    floor = float(np.median(summag[tail])) if tail.any() else float(summag[-1])
    head = starts < min(0.3, sweep_max * 0.2)
    plateau = float(np.median(summag[head])) if head.any() else float(summag[0])

    drop_ratio = plateau / floor if floor > 0 else np.inf
    chirp_detected = bool(drop_ratio >= settings.min_chirp_drop_ratio)

    # Chirp end: first start where Σ|FT| settles to within floor_factor x floor.
    settled = np.where(summag < settings.floor_factor * floor)[0]
    if chirp_detected and settled.size:
        chirp_end = float(starts[settled[0]])
    else:
        chirp_end = 0.0

    # With a chirp present, skip past it plus the ringdown guard margin; with no
    # chirp collapse there is nothing to exclude, so recommend the full FID.
    start_us = chirp_end + settings.guard_margin_us if chirp_detected else 0.0

    return StartDetectionResult(
        start_us=start_us,
        chirp_end_us=chirp_end,
        chirp_detected=chirp_detected,
        floor=floor,
        plateau=plateau,
        band_mhz=band,
        starts_us=starts,
        sum_magnitude=summag,
    )


__all__ = ["StartDetectionResult", "StartDetectionRecord", "detect_start_time"]
