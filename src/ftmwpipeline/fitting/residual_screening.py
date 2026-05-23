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
    "DEFAULT_COHERENCE_CLOSE_THRESHOLD",
    "DEFAULT_COHERENCE_ISOLATED_FWHM",
    "DEFAULT_COHERENCE_ISOLATED_THRESHOLD",
    "ResidualPeakCandidate",
    "filter_by_phase_coherence",
    "find_residual_peaks",
]


# Cluster floor: a candidate whose nearest neighbour (other candidate OR
# already-fitted peak) is closer than this multiple of the FWHM defers
# entirely -- the basis projection cannot disambiguate a sub-FWHM blend
# from a real isolated peak, so we hand the candidate to
# :func:`conservative_fit` 's blend-aware seeder.
DEFAULT_COHERENCE_CLUSTER_FWHM = 1.0
# Distance (in FWHM units) above which a candidate is considered fully
# isolated and the strictest coherence threshold applies. Between the
# cluster floor and this value, the threshold ramps linearly from the
# close anchor to the isolated anchor.
DEFAULT_COHERENCE_ISOLATED_FWHM = 5.0
# "Coherent SNR" / "detected magnitude SNR" floor at the cluster boundary.
# A candidate this close to an existing peak/candidate has its projection
# basis contaminated by the neighbour's Lorentzian skirt (and any neighbour
# fit error), so we accept a small coherent fraction (>= 20%) as evidence
# of a real line.
DEFAULT_COHERENCE_CLOSE_THRESHOLD = 0.2
# Coherent / detected ratio required for a fully isolated candidate (no
# neighbour within DEFAULT_COHERENCE_ISOLATED_FWHM). A real isolated
# Lorentzian's coherent SNR essentially equals its magnitude SNR, so the
# ratio sits near 1.0; a phase-rotation artifact collapses near 0.
# 0.8 keeps real isolated peaks but rejects partially-coherent leakage
# (which can legitimately reach ~0.5 ratio for a 1%-level fit error).
DEFAULT_COHERENCE_ISOLATED_THRESHOLD = 0.8


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


def _sliding_coherence_threshold(
    min_distance_fwhm: float,
    *,
    cluster_threshold_fwhm: float,
    isolated_threshold_fwhm: float,
    close_threshold: float,
    isolated_threshold: float,
) -> float:
    """Linear interpolation of the coherence-ratio threshold by neighbour
    proximity (in FWHM units). Three bands:

    * ``min_distance_fwhm < cluster_threshold_fwhm``: returns ``nan``
      (the caller's signal to defer entirely -- a sub-cluster candidate
      should not be coherence-tested).
    * ``cluster_threshold_fwhm <= min_distance_fwhm <
      isolated_threshold_fwhm``: linear ramp from ``close_threshold`` at
      the cluster boundary up to ``isolated_threshold`` at the isolated
      boundary.
    * ``min_distance_fwhm >= isolated_threshold_fwhm``: returns
      ``isolated_threshold``.

    The ramp tracks the physical contamination level loosely -- a
    candidate one FWHM from a neighbour has its basis support filled
    with ~50% of the neighbour's Lorentzian magnitude, so any fit error
    in that neighbour drags the coherent projection down; at 5 FWHM the
    skirt is ~6% and the projection is essentially clean. A more
    principled functional form (e.g. ``|basis(neighbour - candidate)|``
    directly) is straightforward to swap in later, but the linear ramp
    is easy to tune and reason about.
    """
    if min_distance_fwhm < cluster_threshold_fwhm:
        return float("nan")
    if min_distance_fwhm >= isolated_threshold_fwhm:
        return float(isolated_threshold)
    span = isolated_threshold_fwhm - cluster_threshold_fwhm
    if span <= 0.0:
        return float(isolated_threshold)
    frac = (min_distance_fwhm - cluster_threshold_fwhm) / span
    return float(close_threshold + frac * (isolated_threshold - close_threshold))


def filter_by_phase_coherence(
    candidates: Sequence[ResidualPeakCandidate],
    offset_grid_mhz: np.ndarray,
    residual_complex: np.ndarray,
    rms_noise: np.ndarray,
    tau_us: float,
    acquisition_us: float,
    *,
    fwhm_mhz: float,
    fitted_peak_offsets: Sequence[float] = (),
    cluster_threshold_fwhm: float = DEFAULT_COHERENCE_CLUSTER_FWHM,
    isolated_threshold_fwhm: float = DEFAULT_COHERENCE_ISOLATED_FWHM,
    close_threshold: float = DEFAULT_COHERENCE_CLOSE_THRESHOLD,
    isolated_threshold: float = DEFAULT_COHERENCE_ISOLATED_THRESHOLD,
) -> Tuple[List[ResidualPeakCandidate], List[ResidualPeakCandidate]]:
    """Drop candidates whose residual signal is incoherent with a Lorentzian.

    For each non-clustered candidate, compute the optimal complex amplitude
    ``A = <basis, residual> / <basis, basis>`` (sigma-weighted) where
    ``basis(f) = h_T(f - f_c, tau, T)`` is a unit-amplitude Lorentzian
    centred at the candidate's position. The candidate's ``coherent SNR``
    is ``|A| * |basis(f_c)| / median_sigma_c``. For a real molecular line
    this matches the detected magnitude SNR; for a phase-rotation artifact
    (e.g. residual leftover from a slightly-wrong neighbour fit), the
    complex projection cancels across bins and the ratio collapses.

    A candidate's **acceptance threshold** is a sliding function of its
    distance to the nearest "other peak" -- where "other peak" means
    either another candidate in the same batch OR an already-fitted peak
    in ``fitted_peak_offsets``. Three bands (see
    :func:`_sliding_coherence_threshold`):

    * ``Δ < cluster_threshold_fwhm * fwhm``: the candidate defers entirely
      to :func:`conservative_fit` 's blend-aware seeder (no projection
      test). A sub-cluster candidate's basis projection is contaminated
      by the neighbour to a degree the coherence ratio cannot
      disambiguate.
    * ``Δ`` between cluster and isolated: ramped threshold from
      ``close_threshold`` (lenient -- the projection is expected to be
      contaminated by the neighbour's skirt) up to ``isolated_threshold``
      (strict -- a far candidate has no excuse for low coherence).
    * ``Δ ≥ isolated_threshold_fwhm * fwhm``: full ``isolated_threshold``.

    Including ``fitted_peak_offsets`` in the proximity check is the
    structural change vs the prior candidate-only cluster test: a
    candidate sitting in a *fitted* peak's skirt (e.g. w198 R0 at
    ~1.1 FWHM from a freshly-fit line) was previously coherence-tested
    at the strict 0.5 ratio and rejected; with the sliding scheme it
    gets the appropriate lenient threshold for its distance band.

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
    fitted_peak_offsets
        Offsets (same frame as ``offset_grid_mhz`` / candidate
        ``frequency_mhz``) of already-fitted peaks in the current
        consolidated model. Combined with other candidates' positions
        to form the nearest-neighbour distance that drives the sliding
        threshold. Empty disables the fitted-peak contribution (purely
        candidate-only proximity).
    cluster_threshold_fwhm
        Cluster floor (in FWHM units). Candidates closer than this to any
        other candidate or fitted peak are deferred.
    isolated_threshold_fwhm
        Isolated ceiling (in FWHM units). At and above this distance the
        ``isolated_threshold`` applies.
    close_threshold, isolated_threshold
        Anchor values of the sliding ratio threshold.

    Returns
    -------
    (kept, rejected)
        ``kept`` includes every clustered candidate plus every
        non-clustered candidate that passed the sliding test; ``rejected``
        is the non-clustered candidates that failed.
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

    kept: list[ResidualPeakCandidate] = []
    rejected: list[ResidualPeakCandidate] = []
    candidate_freqs = np.array([c.frequency_mhz for c in candidates], dtype=float)
    fitted_freqs = np.asarray(fitted_peak_offsets, dtype=float)

    for i, c in enumerate(candidates):
        # Min distance to "other peak": any other candidate AND any
        # already-fitted peak. The fitted-peak inclusion is the key fix
        # for candidates that sit in a fitted peak's skirt (w198 R0).
        nearest_other = float("inf")
        if len(candidates) > 1:
            others = np.delete(candidate_freqs, i)
            nearest_other = float(np.min(np.abs(others - c.frequency_mhz)))
        if fitted_freqs.size > 0:
            nearest_fit = float(np.min(np.abs(fitted_freqs - c.frequency_mhz)))
            nearest_other = min(nearest_other, nearest_fit)
        nearest_other_fwhm = (
            nearest_other / fwhm_mhz if fwhm_mhz > 0.0 else float("inf")
        )

        ratio_threshold = _sliding_coherence_threshold(
            nearest_other_fwhm,
            cluster_threshold_fwhm=cluster_threshold_fwhm,
            isolated_threshold_fwhm=isolated_threshold_fwhm,
            close_threshold=close_threshold,
            isolated_threshold=isolated_threshold,
        )
        if not np.isfinite(ratio_threshold):
            # Sub-cluster: defer to the blend-aware seeder.
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
        # Compare to the detected magnitude SNR using the proximity-aware
        # threshold. A close candidate gets a permissive threshold (its
        # projection is expected to be partially contaminated by the
        # neighbour's skirt); a far candidate must clear the strict floor.
        if c.snr > 0 and (coherent_snr / c.snr) < ratio_threshold:
            rejected.append(c)
        else:
            kept.append(c)
    return kept, rejected
