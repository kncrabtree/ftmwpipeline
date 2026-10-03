"""Evaluate a persisted Stage 5 window fit as a complex model spectrum.

The one evaluator of a stored :class:`~ftmwpipeline.core.data_structures.FittingResult`
on an arbitrary molecular-frequency grid. The fit-detail figures draw with it,
and any accessor that hands a fitted model to a client must call it too, so the
drawn model and the delivered one cannot drift apart.

The model of a window is what the Stage 5 fit compared with the active-FT data
(:mod:`ftmwpipeline.fitting.window_fit`):

- the window's fitted lines,
- the frozen out-of-window contributors the fit held fixed and subtracted as a
  background (``frozen_peak_*`` entries of ``fixed_parameters``), and
- the optional leakage-wing baseline ``B(u)`` fitted jointly with the lines
  (recorded in ``quality_metrics``),

all on the signed baseband offset ``u = s * (f - f_c)`` from the window center
``f_c`` (the midpoint of the window's ``freq_range``), with the window's shared
``tau`` and the fit's active acquisition length ``T``.
"""

from __future__ import annotations

from typing import List, Sequence, Union, cast

import numpy as np

from ..core.data_structures import FittingResult, Sideband
from .peak_model import ModelPeak, model_spectrum, sideband_sign

SidebandLike = Union[Sideband, str]

#: Prefix of the ``fixed_parameters`` keys that record a frozen contributor.
FROZEN_PEAK_PREFIX = "frozen_peak_"


def evaluate_window_baseline(
    window_fit: FittingResult, u_offset_mhz: np.ndarray
) -> np.ndarray:
    """Evaluate the persisted leakage-wing baseline ``B(u)`` on an offset grid.

    Reads the per-window baseline audit from ``quality_metrics``
    (``baseline_applied`` / ``baseline_order`` / ``baseline_offset_scale`` /
    ``baseline_coeff{k}_re`` / ``baseline_coeff{k}_im``) and returns the complex
    ``B(u) = sum_{k<=p} (a_k + i b_k) (u/u_s)^k`` on the signed baseband offset
    from the window center. Returns zeros when no baseline fired, so callers can
    add it unconditionally. The Stage 5 fit applies the baseline jointly with
    the de-biased lines, so the faithful model is
    ``model_spectrum(peaks) + B(u)``.
    """
    qa = window_fit.quality_metrics or {}
    u = np.asarray(u_offset_mhz, dtype=float)
    zeros = cast(np.ndarray, np.zeros(u.shape, dtype=np.complex128))
    if float(qa.get("baseline_applied", 0.0)) < 0.5:
        return zeros
    u_s = float(qa.get("baseline_offset_scale", 0.0))
    order = int(qa.get("baseline_order", 0))
    if not u_s > 0.0:
        return zeros
    x = u / u_s
    b = np.zeros(u.shape, dtype=np.complex128)
    for k in range(order + 1):
        a_k = float(qa.get(f"baseline_coeff{k}_re", 0.0))
        b_k = float(qa.get(f"baseline_coeff{k}_im", 0.0))
        b = b + (a_k + 1j * b_k) * x**k
    return cast(np.ndarray, b)


def fitted_model_peaks(
    window_fit: FittingResult, sideband: SidebandLike, center_mhz: float
) -> List[ModelPeak]:
    """The window's fitted lines as offset :class:`ModelPeak` terms, in order."""
    s = sideband_sign(sideband)
    return [
        ModelPeak(
            amplitude=float(p.amplitude),
            offset_mhz=float(s * (p.frequency_mhz - center_mhz)),
            phase=float(p.phase if p.phase is not None else 0.0),
        )
        for p in window_fit.fitted_peaks
    ]


def frozen_model_peaks(
    window_fit: FittingResult, sideband: SidebandLike, center_mhz: float
) -> List[ModelPeak]:
    """The frozen out-of-window contributors as offset :class:`ModelPeak` terms.

    The frozen contributors (``frozen_peak_*`` in ``fixed_parameters``) are the
    strong neighboring lines whose leakage skirt reaches into this window; the
    fit subtracts them as a fixed background, so a faithful model must add
    them back. The offset is re-derived from the persisted molecular frequency
    exactly as the fit derived it (``s * (f - f_c)``).
    """
    s = sideband_sign(sideband)
    peaks: List[ModelPeak] = []
    for key, fp in window_fit.fixed_parameters.items():
        if not key.startswith(FROZEN_PEAK_PREFIX):
            continue
        peaks.append(
            ModelPeak(
                amplitude=float(fp["amplitude"]),
                offset_mhz=float(s * (float(fp["frequency_mhz"]) - center_mhz)),
                phase=float(fp.get("phase", 0.0) or 0.0),
            )
        )
    return peaks


def window_model_peaks(
    window_fit: FittingResult, sideband: SidebandLike, center_mhz: float
) -> List[ModelPeak]:
    """Fitted lines followed by the frozen contributors, as offset ModelPeaks."""
    return fitted_model_peaks(window_fit, sideband, center_mhz) + frozen_model_peaks(
        window_fit, sideband, center_mhz
    )


def evaluate_window_model(
    freqs: np.ndarray,
    window_fit: FittingResult,
    peaks: Sequence[ModelPeak],
    tau_us: float,
    acquisition_us: float,
    sideband: SidebandLike,
    center_mhz: float,
    shape: str,
) -> np.ndarray:
    """Model spectrum (``peaks`` + the window's baseline) on a frequency grid.

    ``peaks`` is normally :func:`window_model_peaks`; ``freqs`` is any
    molecular-frequency grid (MHz). With no peaks, or a non-positive ``tau``,
    only the baseline is returned.
    """
    s = sideband_sign(sideband)
    u = s * (np.asarray(freqs, dtype=float) - center_mhz)
    if peaks and tau_us > 0.0:
        model = model_spectrum(u, peaks, tau_us, acquisition_us, shape=shape)
    else:
        model = np.zeros(u.shape, dtype=np.complex128)
    return cast(np.ndarray, model + evaluate_window_baseline(window_fit, u))


__all__ = [
    "FROZEN_PEAK_PREFIX",
    "evaluate_window_baseline",
    "fitted_model_peaks",
    "frozen_model_peaks",
    "window_model_peaks",
    "evaluate_window_model",
]
