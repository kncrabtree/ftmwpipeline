"""
Peak detection for FTMW spectroscopy (Stage 3).

This module contains the Stage 3 detection algorithm:

* ``locate_peaks`` -- the low-level peak locator, a cleaned, type-annotated
  port of the surviving reference (``bcfitting.ftmwfitting``); numeric
  behavior preserved exactly.
* ``classify_by_snr`` -- SNR-only weak/medium/strong binning.
* ``detect_peaks`` -- the two-pass driver: a primary pass on the apodized
  leakage-suppressed spectrum for the robust coarse list, then a gap pass on a
  second spectrum to recover weak lines the primary apodization suppressed. Both
  passes share a continuous leakage-aware floor that raises the detection
  threshold by the local coherent-leakage amplitude, so a strong line's skirt
  ripple is not re-detected as weak lines.

It operates on already-computed spectra so it stays pure and unit-testable: the
two spectra are just the ``primary_*`` and ``gap_*`` arrays. In the Stage 3
pipeline the gap spectrum is the shape-aware matched filter (see
``ftmwpipeline._internal.stage3_impl``), but ``detect_peaks`` itself makes no
assumption about how either spectrum was built. File orchestration --
recomputing the two spectra from the FID and Stage 1 params, supplying
per-point noise on each grid, persistence -- lives in that module.

The detector is a Savitzky-Golay smoothed second-derivative search:

1. Filter and differentiate ``y`` with a Savitzky-Golay filter (2nd and 1st
   derivatives).
2. Flag local minima of the (negative-clipped) second derivative -- a peak
   must have a concave-down second derivative, which rejects spurious noise
   shoulders on the side of a strong line.
3. With an optional per-point threshold, drop sub-threshold detections and
   apply the strong-peak split/merge heuristic: when the second derivative has
   no zero-crossing between two consecutive detections *and* the first
   derivatives at those points have opposite signs, the pair straddles a single
   strong feature and is merged to its midpoint.
"""

from typing import Dict, List, NamedTuple, Optional, Tuple, cast

import numpy as np
import scipy.signal as spsig

from ..core.data_structures import Peak, PeakClassification

# Absolute SNR classification boundaries. Cross-instrument validated across
# seven fixtures spanning ~3 orders of magnitude in line SNR: the detected-peak
# SNR distribution is anchored at the detection floor with a heavy upper tail,
# so all three tiers stay populated in every regime and these fixed *absolute*
# cutoffs generalize where percentile-based ones would not. Only the
# medium/strong boundary has a downstream consumer (Stage 4 marks a window HARD
# when it holds a STRONG line); the weak/medium boundary is curation labeling.
# Configurable per instrument. See dev-docs/research/stage3-snr-corner/report.md
# section 8.
DEFAULT_WEAK_MEDIUM_SNR = 10.0
DEFAULT_MEDIUM_STRONG_SNR = 50.0
# Default user-facing *promotion* cutoff: which peaks (by user-grid SNR) move
# on to Stage 4.
DEFAULT_MIN_SNR = 3.0
# Fixed *internal* detection floor. Detection runs this aggressively on the
# zpf=1 grids regardless of the promotion cutoff: a 2638 benchmark showed
# detecting at 3.0 then re-measuring on the user grid loses ~190 peaks that
# genuinely clear 3.0 there, while ~2.0 recovers them and then plateaus
# (below 2.0 is almost pure noise, no extra survivors). Cost is flat in the
# floor (detection is bound by the fixed adaptive-noise step), so we always
# detect at <=2.0 and let the promotion cutoff filter afterwards.
DEFAULT_INTERNAL_MIN_SNR = 2.0


class PeakResult(NamedTuple):
    """Return shape of :func:`locate_peaks`.

    Attributes
    ----------
    freqs : np.ndarray
        x values (frequencies, MHz) of the located peaks.
    intensities : np.ndarray
        y values (magnitudes) of the located peaks.
    indices : np.ndarray
        Integer indices of the located peaks in the input arrays.
    """

    freqs: np.ndarray
    intensities: np.ndarray
    indices: np.ndarray


def locate_peaks(
    x: np.ndarray,
    y: np.ndarray,
    window: int = 7,
    order: int = 5,
    thresh: Optional[np.ndarray] = None,
) -> PeakResult:
    """Locate approximate peak positions via a smoothed second-derivative search.

    The ``y`` array is filtered and differentiated with a Savitzky-Golay
    filter, then peaks are located using :func:`scipy.signal.argrelmin` with an
    order of half the Savitzky-Golay window size. Peaks are only flagged where
    the second derivative is negative; this prevents finding spurious "peaks"
    due to noise on the side of a strong peak.

    For strong peaks the second-derivative test occasionally locates a peak on
    each side of the main feature instead of one at the center. To prevent
    this, an additional test is performed: if there is no zero-crossing in the
    second derivative between consecutive peaks, and the first derivatives have
    opposite signs, the two peaks are merged into a single feature at their
    average position.

    Parameters
    ----------
    x : np.ndarray
        1D x array for peak detection (assumed uniformly spaced).
    y : np.ndarray
        1D y array for peak detection.
    window : int, optional
        Window size for the Savitzky-Golay filter. Must be odd, positive, and
        greater than ``order``. A good choice is comparable to the expected
        linewidth of a spectral line in points (default: 7).
    order : int, optional
        Polynomial order for the Savitzky-Golay filter. Must be positive
        (default: 5).
    thresh : np.ndarray, optional
        If provided, an array the same size as ``y`` giving the minimum ``y``
        value required for a peak at each point (default: None).

    Returns
    -------
    PeakResult
        Named tuple ``(freqs, intensities, indices)`` of the located peaks.

    Raises
    ------
    ValueError
        If ``x``/``y`` are not 1D and the same length, if ``window`` is not an
        odd positive integer greater than ``order``, if ``order`` is not a
        positive integer, or if ``thresh`` is provided with the wrong shape.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1-dimensional")
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length")
    if not isinstance(order, (int, np.integer)) or order < 1:
        raise ValueError("order must be a positive integer")
    if not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError("window must be a positive integer")
    if window % 2 == 0:
        raise ValueError("window must be odd")
    if window <= order:
        raise ValueError("window must be greater than order")
    if thresh is not None:
        thresh = np.asarray(thresh)
        if thresh.shape != y.shape:
            raise ValueError("thresh must have the same shape as y")

    half = window // 2

    # Edge padding: replicate the first/last `half` samples so the
    # Savitzky-Golay convolution is well defined at the boundaries. (Faithful
    # to the reference: this is sample replication, not reflection.)
    y_pre = y[0:half]
    y_post = y[-half:]
    y_pad = np.concatenate([y_pre, y, y_post])

    delta = x[1] - x[0]
    coeffs_d2 = spsig.savgol_coeffs(window, order, deriv=2, delta=delta)
    coeffs_d1 = spsig.savgol_coeffs(window, order, deriv=1, delta=delta)
    sg_d2_pad = spsig.oaconvolve(y_pad, coeffs_d2, mode="same")
    sg_d1_pad = spsig.oaconvolve(y_pad, coeffs_d1, mode="same")
    d2 = sg_d2_pad[half:-half]
    d1 = sg_d1_pad[half:-half]

    # Concave-down test: keep only the negative part of the 2nd derivative,
    # then find its local minima (the sharpest concave-down points = peaks).
    neg = np.where(d2 < 0, d2, np.zeros_like(d2))
    locs = spsig.argrelmin(neg, order=half)

    if thresh is None:
        return PeakResult(x[locs], y[locs], np.asarray(locs).flatten())

    # Per-point threshold: drop sub-threshold detections.
    y_at_locs = y[locs]
    t_at_locs = thresh[locs]
    candidate_idx = np.asarray(locs).flatten()
    pos = candidate_idx[np.where(y_at_locs > t_at_locs)]

    if len(pos) <= 1:
        return PeakResult(x[pos], y[pos], pos)

    # Strong-peak split/merge: a strong line can be detected as a pair
    # straddling its center. Merge such a pair to its midpoint when the 2nd
    # derivative does not cross zero between them (no genuine valley) AND the
    # 1st derivatives have opposite signs (a single peak between them).
    merged_pos = []
    last_merged = False
    for p1, p2 in zip(pos[:-1], pos[1:]):
        if last_merged:
            last_merged = False
            continue
        if 0.0 not in neg[p1:p2] and d1[p1] * d1[p2] < 0.0:
            merged_pos.append((p1 + p2) // 2)
            last_merged = True
        else:
            merged_pos.append(p1)
            last_merged = False

    if not last_merged:
        merged_pos.append(pos[-1])

    final_idx = np.asarray(merged_pos)
    return PeakResult(x[final_idx], y[final_idx], final_idx)


def classify_by_snr(
    snr: float,
    weak_medium_snr: float = DEFAULT_WEAK_MEDIUM_SNR,
    medium_strong_snr: float = DEFAULT_MEDIUM_STRONG_SNR,
) -> PeakClassification:
    """Classify a peak by SNR alone.

    Bin edges follow the Stage 3 spec ``weak < t1 <= medium < t2 <= strong``:
    ``snr < weak_medium_snr`` is WEAK, ``weak_medium_snr <= snr <
    medium_strong_snr`` is MEDIUM, ``snr >= medium_strong_snr`` is STRONG.

    Parameters
    ----------
    snr : float
        Peak signal-to-noise ratio.
    weak_medium_snr : float
        Weak/medium boundary ``t1`` (validated default; see module constants).
    medium_strong_snr : float
        Medium/strong boundary ``t2`` (validated default; see module constants).

    Returns
    -------
    PeakClassification

    Raises
    ------
    ValueError
        If the thresholds are not ``0 < weak_medium_snr <= medium_strong_snr``.
    """
    if not (0.0 < weak_medium_snr <= medium_strong_snr):
        raise ValueError(
            "thresholds must satisfy 0 < weak_medium_snr <= medium_strong_snr"
        )
    if snr < weak_medium_snr:
        return PeakClassification.WEAK
    if snr < medium_strong_snr:
        return PeakClassification.MEDIUM
    return PeakClassification.STRONG


def _merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge a list of ``(lo, hi)`` intervals into disjoint sorted intervals."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged: List[Tuple[float, float]] = [ordered[0]]
    for lo, hi in ordered[1:]:
        last_lo, last_hi = merged[-1]
        if lo <= last_hi:
            merged[-1] = (last_lo, max(last_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


def _covered_mask(
    freqs: np.ndarray, intervals: List[Tuple[float, float]]
) -> np.ndarray:
    """Boolean mask: True where ``freqs`` falls inside any merged interval."""
    covered = np.zeros(freqs.shape, dtype=bool)
    for lo, hi in intervals:
        covered |= (freqs >= lo) & (freqs <= hi)
    return cast(np.ndarray, covered)


def _apex_snap(mag: np.ndarray, idx: int, radius: int) -> int:
    """Refine a detection to the local magnitude maximum within ``±radius``.

    ``locate_peaks`` returns the ``argrelmin`` of the smoothed second
    derivative, which for ultra-narrow lines lands a few points off the true
    apex (under-reporting amplitude/SNR). Snapping to the nearest local max on
    the *scoring* spectrum recovers the correct peak height.
    """
    lo = max(0, idx - radius)
    hi = min(len(mag), idx + radius + 1)
    return int(lo + np.argmax(mag[lo:hi]))


def _nearest_indices(ref_freq: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Index in ``ref_freq`` of the closest frequency to each ``query`` value.

    Works for ascending or descending ``ref_freq`` (2638 is descending).
    """
    order = np.argsort(ref_freq)
    sorted_f = ref_freq[order]
    pos = np.searchsorted(sorted_f, query)
    pos = np.clip(pos, 1, len(sorted_f) - 1)
    left = sorted_f[pos - 1]
    right = sorted_f[pos]
    choose_left = np.abs(query - left) <= np.abs(query - right)
    nearest_sorted = np.where(choose_left, pos - 1, pos)
    return cast(np.ndarray, order[nearest_sorted])


def detect_peaks(
    primary_freq: np.ndarray,
    primary_mag: np.ndarray,
    primary_sd: np.ndarray,
    gap_freq: Optional[np.ndarray] = None,
    gap_mag: Optional[np.ndarray] = None,
    gap_sd: Optional[np.ndarray] = None,
    *,
    min_snr: float = DEFAULT_MIN_SNR,
    weak_medium_snr: float = DEFAULT_WEAK_MEDIUM_SNR,
    medium_strong_snr: float = DEFAULT_MEDIUM_STRONG_SNR,
    sg_window: int = 11,
    gap_sg_window: Optional[int] = None,
    sg_order: int = 3,
    primary_leakage_amp: Optional[np.ndarray] = None,
    gap_leakage_amp: Optional[np.ndarray] = None,
    min_exclusion_mhz: float = 0.0,
    run_gap_pass: bool = True,
) -> List[Peak]:
    """Two-pass peak detection scored on the gap (reference) spectrum.

    The two passes only *find positions*; every peak's amplitude, SNR and
    classification are then measured on the ``gap_*`` reference spectrum, giving
    one consistent SNR scale across both passes. (In the Stage 3 pipeline these
    intermediate scores are themselves superseded by a snap-back onto the
    canonical active FT; here ``gap_*`` is simply the common reference.)

    * **Pass 1 (primary)** runs :func:`locate_peaks` on the *apodized*,
      leakage-suppressed spectrum (robust, few sidelobe false positives).
    * **Pass 2 (gap)** runs it on the ``gap_*`` spectrum -- the shape-aware
      matched filter in the Stage 3 pipeline -- to recover weak lines the
      primary apodization smeared away. A strong line's coherent
      truncation-leakage skirt would otherwise re-detect as spurious weak
      lines; the same continuous leakage-aware floor used by the primary pass
      (``gap_leakage_amp``) raises the gap threshold by the local coherent-
      leakage amplitude so the skirt ripple stays below it while a genuine
      line clears it. Detections within ``±min_exclusion_mhz`` of a primary
      peak are dropped as re-finds.

    Every detected position is snapped to the nearest local maximum of the
    ``gap_*`` magnitude (:func:`_apex_snap`) and de-duplicated by snapped
    index, so the split-strong-line triplets and the two passes do not double
    count. If the ``gap_*`` spectrum is not supplied, scoring falls back to the
    apodized primary spectrum (degraded) and the gap pass is skipped.

    Parameters
    ----------
    primary_freq, primary_mag, primary_sd : np.ndarray
        Apodized primary spectrum (position finding), equal length, 1D.
    gap_freq, gap_mag, gap_sd : np.ndarray, optional
        Gap-pass spectrum -- the scoring/reference spectrum and the gap-pass
        detector input (the matched filter in Stage 3). Strongly recommended;
        omitted -> score on the primary spectrum.
    min_snr : float, default 3.0
        Detection floor in SNR units (both passes; SNR on the scoring grid).
    weak_medium_snr, medium_strong_snr : float
        SNR classification boundaries (validated defaults).
    sg_window, sg_order : int
        Savitzky-Golay window/order for :func:`locate_peaks`; the apex-snap
        radius is ``sg_window // 2``.
    primary_leakage_amp, gap_leakage_amp : np.ndarray, optional
        Per-bin additive amplitudes (same shape as ``primary_sd`` / ``gap_sd``)
        raising each pass's detection floor to ``min_snr * sigma + leakage_amp``
        in regions carrying coherent truncation leakage. Each is the local
        leakage estimate ``k * (S_coh / sqrt(M)) * sigma`` (a per-bin
        coherent-leakage amplitude scaled by ``k``) on that pass's own grid: a
        genuine line towers over it while a strong line's skirt ripple -- which
        *is* that leakage -- does not, so neither pass re-detects the skirt as
        weak lines once its noise floor is honest (scatter). Being a continuous
        floor, it never removes the coherence-generating lines themselves
        (a hard ``S_coh`` mask would). If None, that pass's floor is the plain
        ``min_snr * sigma``.
    min_exclusion_mhz : float, default 0.0
        Minimum exclusion half-width around every primary peak.
    run_gap_pass : bool, default True
        If False, only the primary pass runs (O3 -- keep switchable).

    Returns
    -------
    list of Peak
        Classified peaks sorted by frequency ascending; each carries
        ``properties['detection_pass']`` of ``'primary'`` or ``'gap'``.

    Raises
    ------
    ValueError
        If ``min_snr <= 0`` or array lengths within a spectrum disagree.
    """
    if min_snr <= 0:
        raise ValueError("min_snr must be positive")

    primary_freq = np.asarray(primary_freq, dtype=float)
    primary_mag = np.asarray(primary_mag, dtype=float)
    primary_sd = np.asarray(primary_sd, dtype=float)
    if not (primary_freq.shape == primary_mag.shape == primary_sd.shape):
        raise ValueError("primary freq/mag/sd must have the same shape")

    have_gap = gap_freq is not None and gap_mag is not None and gap_sd is not None
    if have_gap:
        ref_freq = np.asarray(gap_freq, dtype=float)
        ref_mag = np.asarray(gap_mag, dtype=float)
        ref_sd = np.asarray(gap_sd, dtype=float)
        if not (ref_freq.shape == ref_mag.shape == ref_sd.shape):
            raise ValueError("gap freq/mag/sd must have the same shape")
    else:
        # No unapodized spectrum -> degrade to scoring on the apodized one.
        ref_freq, ref_mag, ref_sd = primary_freq, primary_mag, primary_sd

    gap_sg = sg_window if gap_sg_window is None else int(gap_sg_window)
    radius = gap_sg // 2

    def _score(ref_idx: int, pass_name: str) -> Tuple[int, Peak]:
        sd_local = ref_sd[ref_idx]
        intensity = float(ref_mag[ref_idx])
        snr = float(intensity / sd_local) if sd_local > 0 else 0.0
        return ref_idx, Peak(
            frequency=float(ref_freq[ref_idx]),
            intensity=intensity,
            index=int(ref_idx),
            snr=snr,
            noise_std_local=float(sd_local),
            classification=classify_by_snr(snr, weak_medium_snr, medium_strong_snr),
            detection_pass=pass_name,
        )

    # idx -> Peak; primary inserted first so it wins ties over the gap pass.
    by_index: Dict[int, Peak] = {}

    # --- Pass 1: positions from the apodized primary spectrum ------------
    primary_thresh = min_snr * primary_sd
    if primary_leakage_amp is not None:
        primary_thresh = primary_thresh + np.asarray(primary_leakage_amp, dtype=float)
    primary = locate_peaks(
        primary_freq,
        primary_mag,
        window=sg_window,
        order=sg_order,
        thresh=primary_thresh,
    )
    exclusions: List[Tuple[float, float]] = []
    if len(primary.freqs):
        ref_idx_arr = _nearest_indices(ref_freq, primary.freqs)
        for ref_i in ref_idx_arr:
            snapped = _apex_snap(ref_mag, int(ref_i), radius)
            if snapped in by_index:
                continue
            _, peak = _score(snapped, "primary")
            by_index[snapped] = peak
            if min_exclusion_mhz > 0.0:
                exclusions.append(
                    (
                        peak.frequency - min_exclusion_mhz,
                        peak.frequency + min_exclusion_mhz,
                    )
                )

    # --- Pass 2: gap-pass detector on the reference (gap) spectrum ---------
    if run_gap_pass and have_gap:
        gap_thresh = min_snr * ref_sd
        if gap_leakage_amp is not None:
            gap_thresh = gap_thresh + np.asarray(gap_leakage_amp, dtype=float)
        gap = locate_peaks(
            ref_freq,
            ref_mag,
            window=gap_sg,
            order=sg_order,
            thresh=gap_thresh,
        )
        merged = _merge_intervals(exclusions)
        if len(gap.freqs):
            covered = _covered_mask(gap.freqs, merged)
            for idx, is_cov in zip(gap.indices, covered):
                if is_cov:
                    continue
                snapped = _apex_snap(ref_mag, int(idx), radius)
                if snapped in by_index:
                    continue
                _, peak = _score(snapped, "gap")
                by_index[snapped] = peak

    peaks = sorted(by_index.values(), key=lambda p: p.frequency)
    return peaks
