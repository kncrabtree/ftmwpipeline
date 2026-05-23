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
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

from .peak_model import ModelPeak, model_spectrum


__all__ = [
    "DEFAULT_COHERENCE_CLUSTER_FWHM",
    "DEFAULT_COHERENCE_RATIO_THRESHOLD",
    "ResidualPeakCandidate",
    "filter_by_phase_coherence",
    "find_residual_peaks",
]


# Two candidates closer than this multiple of the FWHM are considered
# "in a cluster" -- their projections cross-contaminate, so the phase-
# coherence test is skipped for clustered candidates and the blend-aware
# seeder in :func:`conservative_fit` handles them.
DEFAULT_COHERENCE_CLUSTER_FWHM = 1.0
# A candidate's "coherent SNR" (projection magnitude * peak Lorentzian
# value / median sigma_c) must be at least this fraction of its detected
# magnitude SNR to count as a real Lorentzian peak. A phase-rotation
# artifact has high magnitude SNR but near-zero coherent SNR (the complex
# projection cancels). 0.5 means "at least half the magnitude SNR is
# coherently aligned with a Lorentzian basis at this offset".
DEFAULT_COHERENCE_RATIO_THRESHOLD = 0.5


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
    nearest_existing_peak_id, nearest_existing_freq_mhz, nearest_existing_separation_mhz
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
    nearest_existing_peak_id: Optional[int]
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
    existing_peak_ids: Sequence[int] = (),
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
    existing_freqs_mhz, existing_peak_ids
        Currently fitted (or frozen) peaks in this window. Each candidate
        gets the nearest one attached. Lengths may differ -- positions
        without ids leave ``nearest_existing_peak_id`` as ``None``.
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
        mag, height=height, prominence=prominence, distance=distance,
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
                if j < len(existing_peak_ids):
                    near_id = int(existing_peak_ids[j])

        candidates.append(
            ResidualPeakCandidate(
                bin_index=int(peak_idx),
                frequency_mhz=candidate_freq,
                magnitude=candidate_mag,
                snr=candidate_snr,
                prominence_sigma_c=candidate_prom,
                nearest_existing_peak_id=near_id,
                nearest_existing_freq_mhz=near_freq,
                nearest_existing_separation_mhz=near_sep,
                near_existing=near_flag,
            )
        )
    return candidates


def filter_by_phase_coherence(
    candidates: Sequence[ResidualPeakCandidate],
    offset_grid_mhz: np.ndarray,
    residual_complex: np.ndarray,
    rms_noise: np.ndarray,
    tau_us: float,
    acquisition_us: float,
    *,
    fwhm_mhz: float,
    cluster_threshold_fwhm: float = DEFAULT_COHERENCE_CLUSTER_FWHM,
    coherence_ratio_threshold: float = DEFAULT_COHERENCE_RATIO_THRESHOLD,
) -> Tuple[List[ResidualPeakCandidate], List[ResidualPeakCandidate]]:
    """Drop candidates whose residual signal is incoherent with a Lorentzian.

    For each isolated candidate, compute the optimal complex amplitude
    ``A = <basis, residual> / <basis, basis>`` (sigma-weighted) where
    ``basis(f) = h_T(f - f_c, tau, T)`` is a unit-amplitude Lorentzian
    centred at the candidate's position. For a real molecular line, ``|A|``
    matches what an amp+phase-only fit would give and the coherent SNR
    matches the detected magnitude SNR. For a phase-rotation artifact
    (e.g. residual leftover from a slightly-wrong neighbour fit), the
    complex projection cancels across bins and ``|A|`` collapses toward
    zero -- the magnitude SNR was inflated by incoherent contributions.

    Candidates that have another candidate within ``cluster_threshold_fwhm
    * fwhm`` are *not* tested -- close pairs cross-contaminate each
    other's projection, so the test would mis-classify legitimate close
    real peaks. Those candidates pass through unchanged for
    :func:`conservative_fit` 's blend-aware seeder to handle.

    Parameters
    ----------
    candidates
        Output of :func:`find_residual_peaks`.
    offset_grid_mhz, residual_complex, rms_noise
        Same arguments passed to :func:`find_residual_peaks`. Must be
        consistent with the candidate ``frequency_mhz`` frame.
    tau_us, acquisition_us
        Line-shape parameters for the Lorentzian basis. Use the frozen tau
        the rescue will fit at -- typically the initial fit's ``tau_us``.
    fwhm_mhz
        Expected line FWHM (``feature_fwhm(tau, T)``). Used for the
        cluster check and as a contextual scale.
    cluster_threshold_fwhm
        Skip the coherence test for candidates within this multiple of the
        FWHM of any other candidate.
    coherence_ratio_threshold
        Required ratio of coherent SNR to detected magnitude SNR.

    Returns
    -------
    (kept, rejected)
        Two candidate lists. ``kept`` includes every clustered candidate
        plus every isolated candidate that passed the coherence test;
        ``rejected`` is the isolated candidates that failed (the
        phase-rotation artifacts).
    """
    if not candidates:
        return [], []
    u = np.asarray(offset_grid_mhz, dtype=float)
    z = np.asarray(residual_complex, dtype=np.complex128)
    sigma = np.asarray(rms_noise, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(u.size, float(sigma))

    sigma_c = sigma / np.sqrt(2.0)
    sigma_c_safe = np.where(sigma_c > 0.0, sigma_c, 1.0)
    weights = 1.0 / sigma_c_safe**2
    median_sigma_c = float(np.median(sigma_c[sigma_c > 0.0])) if np.any(sigma_c > 0.0) else 0.0
    cluster_threshold = cluster_threshold_fwhm * fwhm_mhz

    kept: list[ResidualPeakCandidate] = []
    rejected: list[ResidualPeakCandidate] = []
    candidate_freqs = np.array([c.frequency_mhz for c in candidates], dtype=float)
    for i, c in enumerate(candidates):
        # Is the candidate clustered with any other candidate?
        if len(candidates) > 1:
            others = np.delete(candidate_freqs, i)
            nearest_other = float(np.min(np.abs(others - c.frequency_mhz)))
        else:
            nearest_other = float("inf")
        if nearest_other < cluster_threshold:
            # Clustered -- let conservative_fit's blend-aware seeder decide.
            kept.append(c)
            continue

        # Build the unit-amplitude Lorentzian basis at this offset and
        # compute the sigma-weighted complex projection.
        basis = model_spectrum(
            u,
            [ModelPeak(amplitude=1.0, offset_mhz=c.frequency_mhz, phase=0.0)],
            tau_us,
            acquisition_us,
        )
        denominator = float(np.sum((np.abs(basis) ** 2) * weights))
        if denominator <= 0.0 or median_sigma_c <= 0.0:
            kept.append(c)
            continue
        numerator = np.sum(np.conj(basis) * z * weights)
        projection = numerator / denominator
        coherent_amp = float(abs(projection))
        peak_idx = int(np.argmin(np.abs(u - c.frequency_mhz)))
        h_t_peak = float(abs(basis[peak_idx]))
        coherent_snr = (coherent_amp * h_t_peak) / median_sigma_c
        # Compare to the detected magnitude SNR. For a real peak the ratio
        # should be ~1; for a phase-rotation artifact, near 0.
        if c.snr > 0 and (coherent_snr / c.snr) < coherence_ratio_threshold:
            rejected.append(c)
        else:
            kept.append(c)
    return kept, rejected
