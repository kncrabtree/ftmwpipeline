"""Detect candidate peaks in a window's complex residual.

Given a single window's complex residual on its frequency / offset grid plus
the per-bin complex noise, this module returns a list of candidates whose
``|residual|`` rises above ``snr_threshold * sigma_c`` (Rayleigh scale) with
enough prominence to outrank single-bin noise spikes. The intent is to feed
a *rescue pass* (:func:`ftmwpipeline.fitting.residual_rescue.attempt_residual_rescue`):
the conservative add-one-peak loop only ever sees Stage 3 candidates, so any
line Stage 3 missed (or that knockout dropped) shows up in the residual and
needs a second nomination.

Stage 3's full peak-finder machinery is *not* reused here. Stage 3 operates
on the zero-padded persisted spectrum with savgol smoothing whose window
length is calibrated for the persisted bin count; on a short active-FT
window slice the same smoothing would over-smear lines. The threshold
*concept* (multiples of a local sigma) is shared; the smoothing and bin
scale are window-local. This detector is deliberately minimal -- a
``scipy.signal.find_peaks`` pass with a sigma-relative height + prominence
cut -- and leans on downstream F-test / AIC gating to weed out false
positives.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.signal import find_peaks

__all__ = [
    "ResidualPeakCandidate",
    "find_residual_peaks",
]


@dataclass(frozen=True)
class ResidualPeakCandidate:
    """A peak candidate detected in a window's ``|residual|`` spectrum.

    ``frequency_mhz`` is in whatever frame the input grid uses -- the
    molecular-frequency grid for visualization or the signed baseband-offset
    grid for production rescue (the detector is grid-agnostic; the caller's
    convention is preserved). ``near_existing`` records whether this
    candidate sits within ``near_separation_factor * fwhm_mhz`` of an already
    fitted (or frozen) peak; a true value usually means "this is fit-quality
    residual at a known line", not a missed nomination.

    Attributes
    ----------
    bin_index : int
        Index into the residual slice.
    frequency_mhz : float
        Grid position (MHz) at the residual peak.
    magnitude : float
        ``|residual|`` at the bin.
    snr : float
        ``magnitude / sigma_c`` at the bin, with ``sigma_c = sigma / sqrt(2)``.
    prominence_sigma_c : float
        Prominence in units of the window's median ``sigma_c``.
    nearest_existing_detection_index, nearest_existing_freq_mhz, nearest_existing_separation_mhz
        Bookkeeping for the nearest fitted/frozen peak (if any were supplied).
    near_existing : bool
        ``True`` when ``nearest_existing_separation_mhz <=
        near_separation_factor * fwhm_mhz`` -- the candidate is co-located
        with an existing line.
    """

    bin_index: int
    frequency_mhz: float
    magnitude: float
    snr: float
    prominence_sigma_c: float
    nearest_existing_detection_index: Optional[int]
    nearest_existing_freq_mhz: Optional[float]
    nearest_existing_separation_mhz: Optional[float]
    near_existing: bool


def find_residual_peaks(
    freq_slice: np.ndarray,
    residual_complex: np.ndarray,
    sigma_slice: np.ndarray,
    *,
    snr_threshold: float = 2.5,
    prominence_threshold: float = 2.0,
    min_separation_mhz: Optional[float] = None,
    fwhm_mhz: Optional[float] = None,
    existing_freqs_mhz: Sequence[float] = (),
    existing_detection_indices: Sequence[int] = (),
    near_separation_factor: float = 0.5,
) -> List[ResidualPeakCandidate]:
    """Detect ``|residual|`` peaks above a sigma-relative threshold.

    Parameters
    ----------
    freq_slice
        Per-bin grid (MHz). Need not be sorted; the detector reports
        positions as-is.
    residual_complex
        Complex residual (data - model) on the grid.
    sigma_slice
        Per-bin complex noise RMS (|X| scale). Per-component sigma is
        ``sigma_slice / sqrt(2)``; the |residual| is Rayleigh-distributed
        with that scale.
    snr_threshold
        Minimum ``|residual| / sigma_c`` at the bin (Rayleigh scale). Default
        ``2.5`` is generous on purpose -- downstream F-test / AIC gating
        ultimately decides which candidates become peaks; the detector should
        nominate borderline cases rather than silently discard them.
    prominence_threshold
        Minimum prominence (in units of the median ``sigma_c``) so isolated
        single-bin noise spikes are suppressed.
    min_separation_mhz
        Minimum spacing between adjacent candidates. When ``None`` and
        ``fwhm_mhz`` is given, defaults to ``fwhm_mhz`` (no two candidates
        within one line width); otherwise the only constraint is the
        ``find_peaks`` one-bin default.
    fwhm_mhz
        Expected line FWHM (``1 / (pi * tau_us)`` for the local tau). Used
        for the default ``min_separation_mhz`` *and* the ``near_existing``
        flag.
    existing_freqs_mhz, existing_detection_indices
        Currently fitted (or frozen) peaks in this window. Each candidate
        gets the nearest one attached. Lengths may differ -- positions
        without ids leave ``nearest_existing_detection_index`` as ``None``.
    near_separation_factor
        Multiplier on ``fwhm_mhz`` for the ``near_existing`` tag. Ignored
        when ``fwhm_mhz`` is missing.
    """
    mag = np.abs(np.asarray(residual_complex)).astype(np.float64)
    sigma_arr = np.asarray(sigma_slice, dtype=np.float64)
    if mag.size == 0 or sigma_arr.size != mag.size:
        return []
    sigma_c = sigma_arr / np.sqrt(2.0)
    median_sigma_c = float(np.median(sigma_c))
    if not np.isfinite(median_sigma_c) or median_sigma_c <= 0.0:
        return []

    height = snr_threshold * median_sigma_c
    prominence = prominence_threshold * median_sigma_c

    freq_arr = np.asarray(freq_slice, dtype=np.float64)
    sep_mhz = min_separation_mhz
    if sep_mhz is None and fwhm_mhz is not None and fwhm_mhz > 0.0:
        sep_mhz = float(fwhm_mhz)
    if sep_mhz is not None and sep_mhz > 0.0 and freq_arr.size > 1:
        df = float(abs(np.median(np.diff(freq_arr))))
        distance = max(1, int(round(sep_mhz / df))) if df > 0.0 else 1
    else:
        distance = 1

    indices, props = find_peaks(
        mag,
        height=height,
        prominence=prominence,
        distance=distance,
    )

    existing_freqs_arr = np.asarray(existing_freqs_mhz, dtype=np.float64)
    near_threshold = (
        near_separation_factor * float(fwhm_mhz)
        if fwhm_mhz is not None and fwhm_mhz > 0.0
        else 0.0
    )

    candidates: List[ResidualPeakCandidate] = []
    for i, peak_idx in enumerate(indices):
        bin_sigma_c = float(sigma_c[peak_idx])
        candidate_mag = float(mag[peak_idx])
        candidate_snr = (
            candidate_mag / bin_sigma_c if bin_sigma_c > 0.0 else float("inf")
        )
        candidate_prom = float(props["prominences"][i]) / median_sigma_c
        candidate_freq = float(freq_arr[peak_idx])

        near_id: Optional[int] = None
        near_freq: Optional[float] = None
        near_sep: Optional[float] = None
        near_flag = False
        if existing_freqs_arr.size > 0:
            seps = np.abs(existing_freqs_arr - candidate_freq)
            j = int(np.argmin(seps))
            near_freq = float(existing_freqs_arr[j])
            near_sep = float(seps[j])
            if near_threshold > 0.0 and near_sep <= near_threshold:
                near_flag = True
                if j < len(existing_detection_indices):
                    near_id = int(existing_detection_indices[j])

        candidates.append(
            ResidualPeakCandidate(
                bin_index=int(peak_idx),
                frequency_mhz=candidate_freq,
                magnitude=candidate_mag,
                snr=candidate_snr,
                prominence_sigma_c=candidate_prom,
                nearest_existing_detection_index=near_id,
                nearest_existing_freq_mhz=near_freq,
                nearest_existing_separation_mhz=near_sep,
                near_existing=near_flag,
            )
        )
    return candidates
