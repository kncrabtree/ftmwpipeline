"""
Noise estimation for FTMW spectroscopy data.

Adaptive frequency-dependent noise estimator. The spectrum is recursively
bisected into bins where the underlying Rayleigh σ appears constant
(median+MAD on raw |X| as the subdivision criterion — robust to spectral-
line outliers and free of the trim-conflation flaw the prior post-trim
mean/var criterion suffered from). Inside each final bin a skewness trim
identifies the noise samples; a moving sqrt-of-mean-squares of those
noise samples gives the per-point σ_x estimate, which tracks σ(f)
continuously at the smoothing-window scale.

The subdivision criterion is documented in
``dev-docs/research/noise-grid-invariance/report.md``. The bin-size floor,
smoothing-window-sample target, and skewness-trim heuristics are
documented in the prior audit at
``dev-docs/research/noise-heuristic-audit/report.md``.
"""

import logging

import numpy as np
import scipy.signal as spsig
from typing import NamedTuple, Tuple, Optional, Dict, Union, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Module-level tuning constants.
#
# DEFAULT_SMOOTHING_MHZ: smoothing window in physical MHz (grid-invariant
# coverage). Real σ(f) on the underlying hardware varies on the
# ~few-hundred-MHz scale (empirical, from no-signal noise-only acquisitions);
# features finer than that — especially ones tracking Lorentzian skirts of
# strong lines — are not physical. 300 MHz averages enough of the surrounding
# spectrum that local skirt-leakage artefacts in the noise mask wash out.
# Translates to ~25 000 noise samples on the 2638 user grid (≈ 0.3 % RMS
# stability per the Rayleigh δ-method formula) — over-tightened relative to
# the audit's 1 % target, but stability is no longer the binding constraint;
# physical smoothness is.
#
# ABS_MIN_BIN_SIZE: minimum bin size below which Rayleigh sample-MAD/median
# fluctuations dominate the subdivision criterion. At N = 300 the sample
# MAD has relative std ≈ 5 %, well below the T = 0.08 subdivision threshold.
#
# SUBDIVISION_THRESHOLD: relative threshold for the median/MAD subdivision
# criterion. Calibrated against the noise-heuristic-audit §2 test bed
# (5000 Rayleigh samples per half, 2000 trials per condition):
# zero false-positives on truly homogeneous noise; 94.6 % detection at
# σ_R/σ_L = 1.10; 100 % at ≥ 1.20. See
# ``scratch/mad-calibration/calibrate_thresholds.py``.
#
# RAYLEIGH_MAD_TO_SC: MAD-to-scale ratio for a Rayleigh distribution.
# Rayleigh(σ_c) has MAD(|X|) ≈ 0.4485 σ_c (computed numerically from the
# Rayleigh CDF; matches the noise-grid-invariance §5 derivation).
# σ_x = σ_c·√2 is the |X|-RMS convention the downstream consumers expect.
# Used to derive an outlier-robust per-bin σ_x reference for diagnostic
# comparisons; not the production output σ_x.
#
# STRONG_PEAK_SNR / SKIRT_EXCLUSION_K / MAX_SKIRT_EXCLUSION_MHZ:
# parameters for the explicit Lorentzian-skirt-exclusion refinement. The
# Lorentzian magnitude skirt of an exp-damped sinusoid decays as
# X_peak·γ/|Δf| far from the line centre (γ = HWHM). Excluding a radius
# Δf_exclude = γ · SNR / k around each strong peak removes the contiguous
# skirt region whose magnitudes would bias the moving-median noise
# estimator. STRONG_PEAK_SNR is the SNR threshold above which we treat a
# peak as worth excluding (peaks below this contribute negligibly to the
# bias). SKIRT_EXCLUSION_K picks the radius where the predicted skirt
# drops to k · σ_x. MAX_SKIRT_EXCLUSION_MHZ caps individual exclusions
# against pathologically strong peaks.
DEFAULT_SMOOTHING_MHZ = 300.0
ABS_MIN_BIN_SIZE = 300
SUBDIVISION_THRESHOLD = 0.08
RAYLEIGH_MAD_TO_SC = 0.4485
STRONG_PEAK_SNR = 20.0
SKIRT_EXCLUSION_K = 1.5
MAX_SKIRT_EXCLUSION_MHZ = 500.0


@dataclass
class NoiseResult:
    """Result container for noise estimation.

    Attributes:
        rms_noise: σ_x estimate across the full frequency grid (= σ_c·√2).
        noise_mask: Boolean mask of points classified as noise by the
            per-bin skewness trim. Used by downstream consumers that need
            to discriminate noise vs signal samples; not used in the σ
            computation itself.
        bin_info: Dictionary of binning + smoothing diagnostics.
    """
    rms_noise: np.ndarray
    noise_mask: np.ndarray
    bin_info: Dict[str, Union[np.ndarray, int, float, str]]


def estimate_noise_adaptive(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    skew_target: float = 0.631,
    inc: float = 0.01,
    min_bin_fraction: float = 1 / 64,
    smoothing_window_mhz: Optional[float] = None,
    min_noise_fraction: float = 2 / 3,
    verbose: bool = False,
) -> NoiseResult:
    """Estimate frequency-dependent σ via adaptive binning + MAD scaling.

    Algorithm:

    1. Recursively bisect the magnitude grid. Split iff the medians of
       raw |X| in the two halves differ by ≥ ``SUBDIVISION_THRESHOLD``
       relatively, OR the MADs do (see
       ``dev-docs/research/noise-grid-invariance/report.md``). Both
       halves must independently exceed ``min_bin_size`` and have
       ≥ ``min_noise_fraction`` noise samples (skewness trim).
    2. Inside each final bin a skewness-targeted trim identifies the
       initial noise samples (kept magnitudes are Rayleigh-like,
       sample skewness < ``skew_target``); their union forms the
       initial noise mask.
    3. σ_x at every grid point is sqrt of a moving mean of |X|² over
       the noise-masked sequence, target window ``DEFAULT_SMOOTHING_MHZ``
       on the full grid. The output is interpolated back to the full
       frequency grid, so σ(f) varies continuously.
    4. Refine the noise mask by excluding Lorentzian skirts of strong
       lines. The strongest peak's HWHM is measured from the data;
       each peak with ``|X|/σ_x > STRONG_PEAK_SNR`` then has a
       neighborhood ``±Δf_exclude = γ · SNR / SKIRT_EXCLUSION_K`` masked
       out (capped at ``MAX_SKIRT_EXCLUSION_MHZ``) — this is the radius
       at which the 1/f-decaying skirt drops below ``k·σ``. Recompute
       σ_x on the refined mask.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency values (MHz).
    magnitudes : np.ndarray
        Magnitude spectrum (= |X|).
    skew_target : float, default 0.631
        Sample-skewness target for the noise-mask trim (Rayleigh value).
    inc : float, default 0.01
        Rank-step granularity for the noise-mask trim.
    min_bin_fraction : float, default 1/64
        Minimum bin size as a fraction of total data length.
    smoothing_window_mhz : float, optional
        Smoothing window in MHz. Defaults to ``DEFAULT_SMOOTHING_MHZ``.
    min_noise_fraction : float, default 2/3
        Minimum trimmed-noise fraction per half required for subdivision.
    verbose : bool
        Emit recursive subdivision decisions to the module logger.

    Returns
    -------
    NoiseResult
        σ_x estimate, noise mask, and bin/smoothing diagnostics.
    """
    if frequencies.shape != magnitudes.shape:
        raise ValueError("frequencies and magnitudes must have the same shape")
    if frequencies.ndim != 1 or magnitudes.ndim != 1:
        raise ValueError("Input arrays must be 1-dimensional")

    n_points = len(frequencies)
    min_bin_size = max(int(n_points * min_bin_fraction), ABS_MIN_BIN_SIZE)

    bin_edges = _compute_mad_based_bins(
        magnitudes,
        min_bin_size=min_bin_size,
        skew_target=skew_target,
        inc=inc,
        min_noise_fraction=min_noise_fraction,
        frequencies=frequencies,
        verbose=verbose,
    )

    noise_mask = _build_noise_mask(magnitudes, bin_edges, skew_target, inc)

    freq_step = abs(frequencies[1] - frequencies[0]) if n_points > 1 else 1.0
    target_mhz = DEFAULT_SMOOTHING_MHZ if smoothing_window_mhz is None else smoothing_window_mhz
    smoothing_window_points = max(int(target_mhz / freq_step), 10)

    # First pass σ_x — needed to find strong peaks for skirt exclusion.
    rms_initial = compute_rms_noise_convolution(
        frequencies, magnitudes, noise_mask, smoothing_window_points
    )
    noise_mask, n_dropped, line_hwhm_mhz = _exclude_strong_line_skirts(
        frequencies, magnitudes, noise_mask, rms_initial
    )
    rms_noise = compute_rms_noise_convolution(
        frequencies, magnitudes, noise_mask, smoothing_window_points
    )

    bin_info: Dict[str, Union[np.ndarray, int, float, str]] = {
        "bin_edges": np.array(bin_edges),
        "n_bins": len(bin_edges) - 1,
        "algorithm": "mad_median_subdivision",
        "noise_fraction": float(np.sum(noise_mask) / max(n_points, 1)),
        "smoothing_window_mhz": float(target_mhz),
        "smoothing_window_points": int(smoothing_window_points),
        "skirt_excluded": int(n_dropped),
        "skirt_line_hwhm_mhz": float(line_hwhm_mhz),
    }
    return NoiseResult(rms_noise, noise_mask, bin_info)


def _mad(values: np.ndarray) -> float:
    """Median absolute deviation (robust spread estimator)."""
    if values.size == 0:
        return 0.0
    return float(np.median(np.abs(values - np.median(values))))


def _compute_mad_based_bins(
    magnitudes: np.ndarray,
    *,
    min_bin_size: int,
    skew_target: float,
    inc: float,
    min_noise_fraction: float,
    frequencies: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> List[int]:
    """Recursively bisect the spectrum using median+MAD on raw |X|.

    Subdivision is rejected when either:
      * the region is too small to split (size < 2·min_bin_size),
      * median and MAD agree across halves within ``SUBDIVISION_THRESHOLD``,
      * either half's noise fraction (skewness trim) falls below
        ``min_noise_fraction``.

    The noise-fraction check still uses the skewness trim — this is the
    "enough noise points to call this a noise bin" guard, distinct from
    the subdivision criterion proper.
    """
    n = magnitudes.shape[0]
    edges: List[int] = [0]

    def noise_fraction(start: int, end: int) -> float:
        bin_mag = magnitudes[start:end]
        idx = np.arange(start, end)
        noise_idx, _ = _filter_by_skewness_cached(
            bin_mag, idx, skew_target, inc, (start, end)
        )
        return len(noise_idx) / max(end - start, 1)

    def should_subdivide(start: int, end: int) -> bool:
        size = end - start
        if size < 2 * min_bin_size:
            if verbose:
                logger.debug(f"  [{start}:{end}] size {size} < {2 * min_bin_size}; no split")
            return False
        mid = (start + end) // 2
        if mid - start < min_bin_size:
            mid = start + min_bin_size
        if end - mid < min_bin_size:
            mid = end - min_bin_size
        if not (start < mid < end):
            return False

        left, right = magnitudes[start:mid], magnitudes[mid:end]
        med_l, med_r = float(np.median(left)), float(np.median(right))
        mad_l, mad_r = _mad(left), _mad(right)
        med_diff = abs(med_l - med_r) / max(0.5 * (med_l + med_r), 1e-12)
        mad_diff = abs(mad_l - mad_r) / max(0.5 * (mad_l + mad_r), 1e-12)
        if verbose:
            f_info = ""
            if frequencies is not None:
                f_lo = float(frequencies[start])
                f_hi = float(frequencies[min(end - 1, n - 1)])
                f_info = f" f=[{f_lo:.0f}:{f_hi:.0f}]"
            logger.debug(
                f"  [{start}:{end}]{f_info} med_diff={med_diff:.3f} mad_diff={mad_diff:.3f}"
            )
        if med_diff < SUBDIVISION_THRESHOLD and mad_diff < SUBDIVISION_THRESHOLD:
            return False

        left_frac = noise_fraction(start, mid)
        right_frac = noise_fraction(mid, end)
        if left_frac < min_noise_fraction or right_frac < min_noise_fraction:
            if verbose:
                logger.debug(
                    f"    noise frac too low: L={left_frac:.3f} R={right_frac:.3f} "
                    f"(need ≥ {min_noise_fraction:.3f})"
                )
            return False
        return True

    def recurse(start: int, end: int) -> None:
        if not should_subdivide(start, end):
            return
        mid = (start + end) // 2
        if mid - start < min_bin_size:
            mid = start + min_bin_size
        if end - mid < min_bin_size:
            mid = end - min_bin_size
        if not (start < mid < end):
            return
        edges.append(mid)
        recurse(start, mid)
        recurse(mid, end)

    recurse(0, n)
    edges.append(n)
    return sorted(set(edges))


def _build_noise_mask(
    magnitudes: np.ndarray,
    bin_edges: List[int],
    skew_target: float,
    inc: float,
) -> np.ndarray:
    """Boolean noise mask: skewness-trim each final bin and union the kept points."""
    n = magnitudes.shape[0]
    noise_mask = np.zeros(n, dtype=bool)
    for i in range(len(bin_edges) - 1):
        a, b = bin_edges[i], bin_edges[i + 1]
        if b <= a:
            continue
        bin_mag = magnitudes[a:b]
        idx = np.arange(a, b)
        noise_idx, _ = _filter_by_skewness_cached(
            bin_mag, idx, skew_target, inc, (a, b)
        )
        noise_mask[noise_idx] = True
    return noise_mask


def _exclude_strong_line_skirts(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    noise_mask: np.ndarray,
    sigma_x: np.ndarray,
) -> Tuple[np.ndarray, int, float]:
    """Exclude Lorentzian-skirt neighborhoods of strong lines from the mask.

    For an exp-damped sinusoid the FT magnitude is Lorentzian:
    ``|X|(Δf) = X_peak · γ / sqrt(Δf² + γ²)`` (γ = HWHM). Far from the
    line centre this reduces to ``X_peak · γ / |Δf|``, so the skirt drops
    below ``k · σ_x`` at ``|Δf| > γ · SNR / k`` where SNR = X_peak/σ_x.
    Below that radius the skirt biases the moving-median noise estimator
    even when individual samples pass the skewness trim — so we mask the
    radius out of the noise mask entirely and recompute σ.

    γ is measured from the data: the FWHM of the strongest peak via
    ``scipy.signal.peak_widths`` at the half-maximum height. This avoids
    threading FT processing parameters into the noise estimator while
    still using a physical width.

    Returns ``(refined_mask, n_excluded, line_hwhm_mhz)``. If no peaks
    exceed STRONG_PEAK_SNR the mask is returned unchanged with
    n_excluded = 0; HWHM is reported as 0.0 in that case.
    """
    n = magnitudes.shape[0]
    if n < 3:
        return noise_mask, 0, 0.0
    freq_step = abs(frequencies[1] - frequencies[0])
    if freq_step <= 0.0:
        return noise_mask, 0, 0.0

    safe_sigma = np.where(sigma_x > 0.0, sigma_x, np.inf)
    snr = magnitudes / safe_sigma
    if not np.any(snr > STRONG_PEAK_SNR):
        return noise_mask, 0, 0.0

    peak_idx, _ = spsig.find_peaks(magnitudes, height=None)
    strong = peak_idx[snr[peak_idx] > STRONG_PEAK_SNR]
    if strong.size == 0:
        return noise_mask, 0, 0.0

    # Measure the strongest peak's HWHM in MHz.
    strongest = int(strong[np.argmax(magnitudes[strong])])
    widths_bins, _, _, _ = spsig.peak_widths(magnitudes, [strongest], rel_height=0.5)
    line_hwhm_mhz = float(widths_bins[0] / 2.0 * freq_step)
    if not np.isfinite(line_hwhm_mhz) or line_hwhm_mhz <= 0.0:
        return noise_mask, 0, 0.0

    max_radius_bins = int(MAX_SKIRT_EXCLUSION_MHZ / freq_step)
    exclusion = np.zeros(n, dtype=bool)
    for p in strong:
        peak_snr = float(snr[p])
        radius_mhz = line_hwhm_mhz * peak_snr / SKIRT_EXCLUSION_K
        radius_bins = min(int(radius_mhz / freq_step), max_radius_bins)
        if radius_bins <= 0:
            continue
        lo = max(0, p - radius_bins)
        hi = min(n, p + radius_bins + 1)
        exclusion[lo:hi] = True

    refined_mask = noise_mask & ~exclusion
    n_excluded = int(np.sum(noise_mask & exclusion))
    return refined_mask, n_excluded, line_hwhm_mhz


class BinStats(NamedTuple):
    """Sufficient statistics of the noise-filtered subset of a bin.

    Retained for use by ``_filter_by_skewness_cached``; the subdivision
    criterion no longer reads its fields, but the serialization round-trip
    and the noise-mask trim do.
    """

    nobs: int
    mean: float
    variance: float
    skewness: float


def _filter_by_skewness_cached(
    bin_magnitudes: np.ndarray,
    bin_indices: np.ndarray,
    skew_target: float,
    inc: float,
    cache_key: tuple,
) -> Tuple[np.ndarray, BinStats]:
    """Identify noise points by trimming the bin's high-magnitude tail until
    the kept distribution is Rayleigh-like (sample skewness < ``skew_target``).

    Sorts the bin once and computes cumulative ``x``, ``x^2``, ``x^3`` so the
    kept-data skewness at every candidate rank cutoff is O(1) to evaluate.
    A 1%-rank cutoff grid is scanned; the first cutoff whose kept skewness
    drops below the target is the chosen mask, and its mean/variance/skewness
    are returned. The 1%-rank step, the 90%-trimmed cutoff guard, and the
    bottom-10% fallback are documented in
    ``dev-docs/research/noise-heuristic-audit/report.md``.
    """
    n = bin_magnitudes.shape[0]
    if n < 3:
        return bin_indices, _bin_stats_from(bin_magnitudes)

    sorted_mag = np.sort(bin_magnitudes)
    cs1 = np.cumsum(sorted_mag, dtype=np.float64)
    cs2 = np.cumsum(sorted_mag * sorted_mag, dtype=np.float64)
    cs3 = np.cumsum(sorted_mag * sorted_mag * sorted_mag, dtype=np.float64)

    def stats_at(keep_n: int) -> BinStats:
        k = float(keep_n)
        m1 = cs1[keep_n - 1] / k
        var = cs2[keep_n - 1] / k - m1 * m1
        if var <= 0:
            return BinStats(keep_n, float(m1), float(var), float("inf"))
        m3 = cs3[keep_n - 1] / k - 3.0 * m1 * (cs2[keep_n - 1] / k) + 2.0 * m1 ** 3
        skew = m3 / var ** 1.5
        return BinStats(keep_n, float(m1), float(var), float(skew))

    n_steps = int(np.floor(0.9 / inc)) + 1
    for step in range(n_steps):
        cutoff = step * inc
        if cutoff == 0.0:
            keep_n = n
        else:
            keep_n = max(3, n - int(np.ceil(cutoff * n)))
        st = stats_at(keep_n)
        if st.skewness < skew_target:
            if cutoff == 0.0:
                return bin_indices, st
            threshold = sorted_mag[keep_n - 1]
            mask = bin_magnitudes <= threshold
            return bin_indices[mask], st

    keep_n = max(1, n // 10)
    threshold = sorted_mag[keep_n - 1]
    mask = bin_magnitudes <= threshold
    fallback_indices = bin_indices[mask] if np.any(mask) else bin_indices[:1]
    return fallback_indices, stats_at(max(3, keep_n))


def _bin_stats_from(values: np.ndarray) -> BinStats:
    """BinStats from a full slice (no trimming). Used for degenerate
    small-bin paths where the skewness scan would be ill-conditioned."""
    n = int(values.shape[0])
    if n == 0:
        return BinStats(0, 0.0, 0.0, float("nan"))
    mean = float(np.mean(values))
    var = float(np.var(values))
    if n < 3 or var <= 0:
        return BinStats(n, mean, var, float("nan"))
    centred = values - mean
    skew = float(np.mean(centred ** 3) / var ** 1.5)
    return BinStats(n, mean, var, skew)


# For a Rayleigh distribution with scale σ_c, median(|X|) = σ_c · √(2 ln 2);
# σ_x = σ_c · √2 = median(|X|) · √(1 / ln 2) ≈ median(|X|) · 1.2011. The
# moving-median estimator outputs σ_x by scaling the moving-median of |X|
# in the noise-mask window by this constant.
_RAYLEIGH_MEDIAN_TO_SIGMA_X = 1.0 / np.sqrt(np.log(2.0))


def compute_rms_noise_convolution(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    noise_mask: np.ndarray,
    bl_bin: int,
) -> np.ndarray:
    """Moving-median σ_x estimator over the noise-masked magnitude sequence.

    Slides a median filter of width ``bl_bin`` along ``magnitudes[noise_mask]``
    (boundary-block padded), converts each median to σ_x via the Rayleigh
    quantile relation σ_x = median(|X|) · √(1/ln 2), and interpolates back
    to the full frequency grid. The moving-median is robust against
    Lorentzian-skirt residuals that survive the per-bin skewness trim:
    median is unaffected by contamination up to 50 % of the window, and a
    300 MHz window contains 100× more samples than any single line's skirt
    so skirt contamination is well below the breakdown point.

    The function name is historical (this used to compute moving sqrt-of-
    mean-squared) but the signature and σ_x output convention are
    unchanged; callers that just need σ_x on the full grid (production
    estimator, HDF5 round-trip) do not need to change.
    """
    noise_magnitudes = magnitudes[noise_mask]
    noise_frequencies = frequencies[noise_mask]

    if len(noise_magnitudes) == 0:
        return np.full_like(frequencies, np.min(magnitudes))

    bl_bin = max(bl_bin, 1)
    # scipy median_filter handles boundaries via 'reflect' (same convention
    # as the previous bl_pre/bl_post block-padding) so no explicit padding
    # is needed here.
    from scipy.ndimage import median_filter
    median_values = median_filter(
        noise_magnitudes.astype(np.float64, copy=False),
        size=bl_bin,
        mode="reflect",
    )
    sigma_x_at_mask = median_values * _RAYLEIGH_MEDIAN_TO_SIGMA_X

    if len(noise_frequencies) == len(frequencies) and np.allclose(noise_frequencies, frequencies):
        return sigma_x_at_mask

    if frequencies[0] > frequencies[-1]:
        return np.interp(frequencies, noise_frequencies[::-1], sigma_x_at_mask[::-1])
    return np.interp(frequencies, noise_frequencies, sigma_x_at_mask)
