"""Evaluate a persisted Stage 5 window fit as a complex model spectrum.

The one evaluator of a stored :class:`~ftmwpipeline.core.data_structures.FittingResult`
on an arbitrary molecular-frequency grid. The fit-detail figures draw with it,
and any accessor that hands a fitted model to a client must call it too, so the
drawn model and the delivered one cannot drift apart.

The model of a window is what the Stage 5 fit compared with the active-FT data
(:mod:`ftmwpipeline.fitting.window_fit`):

- the window's fitted lines,
- the frozen out-of-window contributors the fit held fixed
  (``frozen_peak_*`` entries of ``fixed_parameters``), drawn at the window's
  shared fitted ``tau`` exactly as the fit drew them (ROADMAP D18), and
- the optional leakage-wing baseline ``B(u)`` fitted jointly with the lines
  (recorded in ``quality_metrics``),

all on the signed baseband offset ``u = s * (f - f_c)`` from the window center
``f_c`` (the midpoint of the window's ``freq_range``), with the window's shared
``tau`` and the fit's active acquisition length ``T``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union, cast

import numpy as np

from ..core.data_structures import FittingResult, Sideband, SpectrumFit
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


def window_center_mhz(window_fit: FittingResult) -> float:
    """The window's reference frequency ``f_c``: the midpoint of its fit range.

    The fit's own convention (``plan_execution.materialize_window``).
    """
    if window_fit.window is None:
        raise ValueError(
            f"window {window_fit.window_id} has no attached SpectralWindow -- "
            "its fit range is unknown"
        )
    lo, hi = window_fit.window.freq_range
    return 0.5 * (float(lo) + float(hi))


def window_fit_range_mhz(window_fit: FittingResult) -> tuple[float, float]:
    """The window's fit range ``(low, high)`` in MHz (molecular, raw frame)."""
    if window_fit.window is None:
        raise ValueError(
            f"window {window_fit.window_id} has no attached SpectralWindow -- "
            "its fit range is unknown"
        )
    lo, hi = window_fit.window.freq_range
    return float(min(lo, hi)), float(max(lo, hi))


def window_tau_us(window_fit: FittingResult) -> float:
    """The window's shared fitted decay time (``tau`` or ``tau_G``), in us."""
    return float(window_fit.shared_parameters.get("tau_us", {}).get("value", 0.0))


def window_shape(window_fit: FittingResult) -> str:
    """The line shape the window was fitted with."""
    return str(getattr(window_fit, "shape", "lorentzian"))


def window_has_baseline(window_fit: FittingResult) -> bool:
    """Whether the fit carried a leakage-wing baseline in this window."""
    qa = window_fit.quality_metrics or {}
    return float(qa.get("baseline_applied", 0.0)) >= 0.5


@dataclass(frozen=True)
class WindowModelTerms:
    """A persisted window fit evaluated term by term on one frequency grid.

    Attributes
    ----------
    model : np.ndarray
        Everything the fit compared with the data: the fitted lines, the
        frozen contributors and the baseline -- :func:`evaluate_window_model`
        on :func:`window_model_peaks`, the path every plot draws.
    fixed : np.ndarray
        The frozen contributors alone, at the window's shared fitted ``tau``.
    baseline : np.ndarray or None
        The fitted baseline alone; ``None`` when the window has none.
    components : list of np.ndarray or None
        One array per fitted line, in ``fitted_peaks`` order; ``None`` unless
        requested.
    """

    model: np.ndarray
    fixed: np.ndarray
    baseline: Optional[np.ndarray]
    components: Optional[List[np.ndarray]]


def evaluate_window_terms(
    freqs: np.ndarray,
    window_fit: FittingResult,
    *,
    acquisition_us: float,
    sideband: SidebandLike,
    components: bool = False,
) -> WindowModelTerms:
    """Evaluate a persisted window fit, and each of its terms, on ``freqs``.

    ``freqs`` is any molecular-frequency grid (MHz). Every term is evaluated at
    the window's shared fitted ``tau`` on the window's own offset frame.
    """
    center = window_center_mhz(window_fit)
    tau_us = window_tau_us(window_fit)
    shape = window_shape(window_fit)
    s = sideband_sign(sideband)
    f = np.asarray(freqs, dtype=float)
    u = s * (f - center)
    model = evaluate_window_model(
        f,
        window_fit,
        window_model_peaks(window_fit, sideband, center),
        tau_us,
        acquisition_us,
        sideband,
        center,
        shape,
    )
    frozen = frozen_model_peaks(window_fit, sideband, center)
    if frozen and tau_us > 0.0:
        fixed = model_spectrum(u, frozen, tau_us, acquisition_us, shape=shape)
    else:
        fixed = np.zeros(u.shape, dtype=np.complex128)
    baseline = (
        evaluate_window_baseline(window_fit, u)
        if window_has_baseline(window_fit)
        else None
    )
    lines: Optional[List[np.ndarray]] = None
    if components:
        lines = [
            (
                model_spectrum(u, [pk], tau_us, acquisition_us, shape=shape)
                if tau_us > 0.0
                else np.zeros(u.shape, dtype=np.complex128)
            )
            for pk in fitted_model_peaks(window_fit, sideband, center)
        ]
    return WindowModelTerms(
        model=model,
        fixed=fixed,
        baseline=baseline,
        components=lines,
    )


def spectrum_line_owners(fit: SpectrumFit) -> Dict[int, int]:
    """For a line fitted in more than one window, the window that evaluates it.

    A thaw copies a contributor's line into the dependent window under the
    primary's ``peak_uid``, so the same line can appear in two windows' fits.
    A whole-spectrum model evaluates it once: in the primary window of an
    accepted thaw when one names it, otherwise in the lowest window id.
    Returns ``{peak_uid: window_id}`` for the duplicated identities only.
    """
    seen: Dict[int, List[int]] = {}
    for wf in fit.window_fits:
        if wf.window_id is None:
            continue
        for p in wf.fitted_peaks:
            uid = getattr(p, "peak_uid", None)
            if uid is not None:
                seen.setdefault(int(uid), []).append(int(wf.window_id))
    dup = {uid: wids for uid, wids in seen.items() if len(set(wids)) > 1}
    owners: Dict[int, int] = {}
    if not dup:
        return owners
    primaries = {
        int(te.primary_window_id)
        for wf in fit.window_fits
        for te in (wf.thaw_events or [])
        if te.accepted
    }
    for uid, wids in dup.items():
        named = sorted(w for w in set(wids) if w in primaries)
        owners[uid] = named[0] if named else min(wids)
    return owners


def _baseline_owners(freqs: np.ndarray, windows: Sequence[FittingResult]) -> np.ndarray:
    """Per bin of ``freqs``, the index into ``windows`` whose baseline covers
    it, or ``-1``.

    A baseline is a local leakage-wing polynomial and lives only inside its own
    window's fit range; where fit ranges overlap, a bin takes the window whose
    centre is nearest.
    """
    owner = np.full(freqs.shape, -1, dtype=int)
    best = np.full(freqs.shape, np.inf)
    for i, wf in enumerate(windows):
        lo, hi = window_fit_range_mhz(wf)
        idx = np.flatnonzero((freqs >= lo) & (freqs <= hi))
        if not idx.size:
            continue
        dist = np.abs(freqs[idx] - window_center_mhz(wf))
        take = dist < best[idx]
        owner[idx[take]] = i
        best[idx[take]] = dist[take]
    return owner


def evaluate_spectrum_model(
    freqs: np.ndarray,
    fit: SpectrumFit,
    *,
    acquisition_us: float,
    sideband: SidebandLike,
) -> np.ndarray:
    """The whole persisted fit as one model spectrum on a frequency grid.

    Every line of the fit once (:func:`spectrum_line_owners`), over the whole
    grid at its window's fitted ``tau`` and shape, plus each window's baseline
    only inside that window's fit range (the nearest window centre where fit
    ranges overlap). Frozen contributors are not added: every neighbour's full
    line is already in the sum.
    """
    f = np.asarray(freqs, dtype=float)
    s = sideband_sign(sideband)
    owners = spectrum_line_owners(fit)
    windows = [
        wf
        for wf in sorted(
            fit.window_fits,
            key=lambda w: -1 if w.window_id is None else int(w.window_id),
        )
        if wf.window is not None
    ]
    model = np.zeros(f.shape, dtype=np.complex128)
    for wf in windows:
        tau_us = window_tau_us(wf)
        if not tau_us > 0.0:
            continue
        center = window_center_mhz(wf)
        wid = int(wf.window_id) if wf.window_id is not None else -1
        uids = [getattr(p, "peak_uid", None) for p in wf.fitted_peaks]
        lines = [
            mp
            for uid, mp in zip(uids, fitted_model_peaks(wf, sideband, center))
            if uid is None or owners.get(int(uid), wid) == wid
        ]
        if lines:
            model += model_spectrum(
                s * (f - center),
                lines,
                tau_us,
                acquisition_us,
                shape=window_shape(wf),
            )
    owner = _baseline_owners(f, windows)
    for i, wf in enumerate(windows):
        if not window_has_baseline(wf):
            continue
        idx = np.flatnonzero(owner == i)
        if idx.size:
            model[idx] += evaluate_window_baseline(
                wf, s * (f[idx] - window_center_mhz(wf))
            )
    return model


__all__ = [
    "FROZEN_PEAK_PREFIX",
    "evaluate_spectrum_model",
    "spectrum_line_owners",
    "WindowModelTerms",
    "evaluate_window_terms",
    "window_center_mhz",
    "window_fit_range_mhz",
    "window_has_baseline",
    "window_shape",
    "window_tau_us",
    "evaluate_window_baseline",
    "fitted_model_peaks",
    "frozen_model_peaks",
    "window_model_peaks",
    "evaluate_window_model",
]
