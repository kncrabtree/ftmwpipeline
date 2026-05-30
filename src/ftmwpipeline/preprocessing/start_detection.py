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
  * As a confirmatory diagnostic, the post-chirp floor itself has two decay
    regimes (fast ringdown, then slow molecular tail); the Kneedle elbow between
    them lands near the same place when the ringdown is separable. Strong
    molecular FIDs bury the ringdown and yield no knee (strength ~0), so the
    knee is reported with a strength score rather than used as the estimate.

See :class:`~ftmwpipeline.core.start_detection_settings.StartDetectionSettings`
for the knobs and :mod:`ftmwpipeline._internal.start_detection_impl` for the
file-bound orchestration (band resolution + recommended-``start_us`` stamping).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d

from ..core.data_structures import FID
from ..core.start_detection_settings import StartDetectionSettings


@dataclass(frozen=True)
class StartDetectionResult:
    """Outcome of :func:`detect_start_time`.

    Attributes
    ----------
    start_us :
        Recommended FID window start (``chirp_end_us + guard_margin_us``).
    chirp_end_us :
        Start time at which Σ|FT| collapses to the post-chirp floor.
    knee_us :
        Kneedle elbow of the post-chirp floor (ringdown -> molecular). A
        diagnostic; equals the chirp-end window left edge when no knee is found.
    knee_strength :
        Normalized elbow strength (0..1); larger is a sharper, more separable
        ringdown knee.
    knee_confident :
        ``knee_strength >= knee_strength_min``.
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
    """

    start_us: float
    chirp_end_us: float
    knee_us: float
    knee_strength: float
    knee_confident: bool
    chirp_detected: bool
    floor: float
    plateau: float
    band_mhz: Optional[Tuple[float, float]]
    starts_us: np.ndarray
    sum_magnitude: np.ndarray


def _kneedle_elbow(x: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
    """Kneedle elbow of a decreasing curve.

    Returns the index of maximum chord-minus-curve distance after min-max
    normalization, plus that distance (0 for a straight line, larger for a
    sharper elbow). ``x`` must be increasing and ``y`` decreasing overall.
    """
    if x.size < 3:
        return 0, 0.0
    xn = (x - x.min()) / max(x.max() - x.min(), 1e-12)
    yn = (y - y.min()) / max(y.max() - y.min(), 1e-12)
    diff = (1.0 - xn) - yn  # chord of a decreasing curve is (1 - xn)
    ki = int(np.argmax(diff))
    return ki, float(diff[ki])


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
        preprocessed = fid.preprocess(start_us=float(s), zpf=settings.zpf, expf_us=None)
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

    # Confirmatory Kneedle elbow on the post-chirp floor (skip the shoulder).
    logs = uniform_filter1d(np.log(summag), size=5, mode="nearest")
    win = (starts >= chirp_end + settings.shoulder_skip_us) & (
        starts <= chirp_end + settings.knee_window_us
    )
    if win.sum() >= 3:
        ki, knee_strength = _kneedle_elbow(starts[win], logs[win])
        knee_us = float(starts[win][ki])
    else:
        knee_us, knee_strength = chirp_end, 0.0
    knee_confident = bool(knee_strength >= settings.knee_strength_min)

    start_us = chirp_end + settings.guard_margin_us

    return StartDetectionResult(
        start_us=start_us,
        chirp_end_us=chirp_end,
        knee_us=knee_us,
        knee_strength=knee_strength,
        knee_confident=knee_confident,
        chirp_detected=chirp_detected,
        floor=floor,
        plateau=plateau,
        band_mhz=band,
        starts_us=starts,
        sum_magnitude=summag,
    )


__all__ = ["StartDetectionResult", "detect_start_time"]
