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
from scipy.ndimage import median_filter, percentile_filter, gaussian_filter1d
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
    *,
    subdivision_threshold: float = SUBDIVISION_THRESHOLD,
    abs_min_bin_size: int = ABS_MIN_BIN_SIZE,
    strong_peak_snr: float = STRONG_PEAK_SNR,
    skirt_exclusion_k: float = SKIRT_EXCLUSION_K,
    max_skirt_exclusion_mhz: float = MAX_SKIRT_EXCLUSION_MHZ,
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
    min_bin_size = max(int(n_points * min_bin_fraction), int(abs_min_bin_size))

    bin_edges = _compute_mad_based_bins(
        magnitudes,
        min_bin_size=min_bin_size,
        skew_target=skew_target,
        inc=inc,
        min_noise_fraction=min_noise_fraction,
        subdivision_threshold=float(subdivision_threshold),
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
        frequencies, magnitudes, noise_mask, rms_initial,
        strong_peak_snr=float(strong_peak_snr),
        skirt_exclusion_k=float(skirt_exclusion_k),
        max_skirt_exclusion_mhz=float(max_skirt_exclusion_mhz),
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
    subdivision_threshold: float = SUBDIVISION_THRESHOLD,
    frequencies: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> List[int]:
    """Recursively bisect the spectrum using median+MAD on raw |X|.

    Subdivision is rejected when either:
      * the region is too small to split (size < 2·min_bin_size),
      * median and MAD agree across halves within ``subdivision_threshold``
        (defaults to the module-level ``SUBDIVISION_THRESHOLD``),
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
        if med_diff < subdivision_threshold and mad_diff < subdivision_threshold:
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
    *,
    strong_peak_snr: float = STRONG_PEAK_SNR,
    skirt_exclusion_k: float = SKIRT_EXCLUSION_K,
    max_skirt_exclusion_mhz: float = MAX_SKIRT_EXCLUSION_MHZ,
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

    ``strong_peak_snr``, ``skirt_exclusion_k`` and
    ``max_skirt_exclusion_mhz`` default to the module-level constants of
    the same name (upper-case), kept as the readable canonical source.

    Returns ``(refined_mask, n_excluded, line_hwhm_mhz)``. If no peaks
    exceed ``strong_peak_snr`` the mask is returned unchanged with
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
    if not np.any(snr > strong_peak_snr):
        return noise_mask, 0, 0.0

    peak_idx, _ = spsig.find_peaks(magnitudes, height=None)
    strong = peak_idx[snr[peak_idx] > strong_peak_snr]
    if strong.size == 0:
        return noise_mask, 0, 0.0

    # Measure the strongest peak's HWHM in MHz.
    strongest = int(strong[np.argmax(magnitudes[strong])])
    widths_bins, _, _, _ = spsig.peak_widths(magnitudes, [strongest], rel_height=0.5)
    line_hwhm_mhz = float(widths_bins[0] / 2.0 * freq_step)
    if not np.isfinite(line_hwhm_mhz) or line_hwhm_mhz <= 0.0:
        return noise_mask, 0, 0.0

    max_radius_bins = int(max_skirt_exclusion_mhz / freq_step)
    exclusion = np.zeros(n, dtype=bool)
    for p in strong:
        peak_snr = float(snr[p])
        radius_mhz = line_hwhm_mhz * peak_snr / skirt_exclusion_k
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


# ---------------------------------------------------------------------------
# Scatter (high-pass), region-aware noise estimator — Stage 2 alternative.
#
# The canonical FT is raw and un-apodized (boxcar), so on high-SNR, line-dense
# spectra the summed far-wings of strong lines form a smooth leakage *pedestal*
# that fills every quiet bin. A level-based estimator (the adaptive one above)
# measures that pedestal, not the random noise, and over-reports σ by up to ~6×
# at SNR ~10⁵–10⁶. The pedestal is constant in shot count N while the noise
# averages down as 1/√N, so the error is a pure SNR-scaling failure.
#
# The fix is a high-pass along the frequency axis: the noise is the white,
# bin-uncorrelated part of |X|; the pedestal is the smooth part.
#
#   σ(f) = C(R) · 1.4826 · MAD( |X| − medfilt_pedestal(|X|) )  over non-line bins
#
# |X| is Rician, so the magnitude-scatter relates to the underlying complex σ by
# a regime-dependent factor running from 1.0 under strong lines (Rician → Gaussian)
# to 1.47 in quiet Rayleigh regions. The dimensionless ratio R = scatter/pedestal
# is a monotone function of the regime alone, so a single 1-D lookup C(R) recovers
# the regime-correct factor from one spectrum — no frames, no fixed mid-regime
# bias. Stage 2 runs before Stage 3, so the estimator self-masks lines iteratively
# rather than consuming a peak list.
#
# Finally the per-region σ is smoothed in two passes. First a broad moving median:
# line skirts and boxcar sidelobes can only *add* to the residual scatter (never
# subtract), so the true noise floor is the lower envelope of the local estimates;
# a wide median is unbiased where the spectrum is clean yet, being robust to up to
# 50 % per-window contamination, rides the floor straight through line-dense bands
# instead of bumping up under them. (The noise floor is a slowly varying receiver
# property, so a broad window does not erase real structure.) Lower
# ``smoothing_percentile`` below 50 for a more aggressive lower-envelope at the
# cost of a clean-region low bias. The median is robust but leaves a staircase, so
# a second Gaussian convolution (``convolve_mhz``) removes the steps — applied to
# the already-de-inflated median output, so it cannot re-inflate under lines.
#
# The full derivation, the 1/√N validation, and the region-aware C(R) calibration
# against frame-difference truth across the multi-frame fixtures are documented in
# ``dev-docs/research/noise-snr-scaling/report.md`` (§4.1, §9). The estimator's
# instrument-family-dependent knobs are tracked in
# ``dev-docs/planning/instrument-tunable-knobs.md``.

# Default knobs (instrument-family-dependent; see the planning doc above).
SCATTER_WINDOW_MHZ = 80.0      # full width of the per-region scatter-MAD window
SCATTER_PEDESTAL_MHZ = 20.0    # running-median width isolating the leakage pedestal
SCATTER_LINE_K = 8.0           # robust-σ multiple above which a bin is flagged a line
SCATTER_N_ITER = 3             # self-mask refinement iterations
SCATTER_MIN_WINDOW_SAMPLES = 30  # minimum surviving non-line bins per region window
SCATTER_SMOOTHING_MHZ = 800.0  # broad moving-percentile σ smoothing width (0 = off)
SCATTER_SMOOTHING_PERCENTILE = 50.0  # 50 = median (unbiased); lower = lower-envelope
SCATTER_CONVOLVE_MHZ = 200.0   # Gaussian σ (MHz) of the 2nd, step-removing pass (0 = off)

# Fixed mid-regime σ_c/scatter factor used when ``region_aware=False`` (the
# Rayleigh-to-under-line endpoints span 1.0–1.47; 1.20 is the validated
# mid-regime compromise — biased ~±15 % in a regime-dependent way, which is
# exactly what the region-aware lookup removes).
FIXED_SCATTER_FACTOR = 1.20

# Region-aware Rician correction C(R) = σ_c / scatter, where R = scatter / pedestal
# selects the local pedestal/noise regime. Frozen Monte-Carlo reference table
# (M = 100 000, seed = 0); regenerable verbatim via the noise-snr-scaling
# prototype's ``_build_CR``. The Rayleigh-end MC scatter in C is part of the
# validated reference and is averaged over by the per-region windowing.
_SCATTER_R_TAB = np.array(
    [
        0.03986596, 0.04082170, 0.04199363, 0.04314672, 0.04392716, 0.04513807,
        0.04599797, 0.04714164, 0.04834466, 0.05017728, 0.05161228, 0.05315595,
        0.05441469, 0.05617445, 0.05765688, 0.05981985, 0.06225409, 0.06448111,
        0.06661682, 0.06929284, 0.07214518, 0.07538205, 0.07831997, 0.08230764,
        0.08582522, 0.08997978, 0.09478602, 0.10021766, 0.10627878, 0.11308372,
        0.12010165, 0.12783041, 0.13848745, 0.15058242, 0.16410868, 0.17921095,
        0.19831460, 0.22243956, 0.25294978, 0.29016246, 0.30734432, 0.31370355,
        0.31995688, 0.32878855, 0.33599066, 0.34114749, 0.35159426, 0.36126187,
        0.36920461, 0.37736863, 0.38960624, 0.39594634, 0.40604791, 0.41872537,
        0.42767292, 0.43793798, 0.45029773, 0.46199273, 0.47202251, 0.48345762,
        0.49670391, 0.50579551, 0.51433488, 0.52263698, 0.52946218, 0.53538209,
        0.54391309, 0.54984083, 0.55411757, 0.55956841, 0.56129992, 0.56202714,
        0.56286181, 0.56287035, 0.56367008, 0.56378198, 0.56442523, 0.56460626,
        0.56461257, 0.56562746,
    ]
)

_SCATTER_C_TAB = np.array(
    [
        1.00237940, 1.00153229, 0.99632158, 0.99259848, 0.99923264, 0.99649888,
        1.00330809, 1.00483687, 1.00623928, 0.99691948, 0.99696346, 0.99684148,
        1.00328269, 1.00233410, 1.00819421, 1.00435034, 0.99827075, 0.99864254,
        1.00268474, 1.00080615, 1.00037696, 0.99759210, 1.00167353, 0.99724406,
        1.00217515, 1.00379491, 1.00360366, 1.00173589, 1.00012860, 0.99966342,
        1.00400361, 1.01078569, 1.00565146, 1.00119711, 1.00323225, 1.01000980,
        1.01273321, 1.01408565, 1.01710622, 1.02781026, 1.02871528, 1.03128887,
        1.03469078, 1.03405033, 1.03531733, 1.04496886, 1.04100395, 1.04260191,
        1.04645777, 1.05194337, 1.04575131, 1.06128545, 1.06651893, 1.06300956,
        1.07346248, 1.08055241, 1.08495677, 1.09310003, 1.10386198, 1.11462597,
        1.12209720, 1.13899321, 1.16268326, 1.18071045, 1.20397788, 1.22673695,
        1.24711990, 1.27198521, 1.29583333, 1.32547974, 1.46727774, 1.50577996,
        1.34660719, 1.37852501, 1.48722828, 1.40654222, 1.50457734, 1.46599646,
        1.49454816, 1.42803168,
    ]
)

# C(R) recovers the per-quadrature σ_c; the canonical Stage 2 ``rms_noise`` is the
# complex RMS σ_x = σ_c·√2 (real/imag each carry σ_x²/2 — see
# ``fitting/validation.py``). The estimator scales its σ_c output to σ_x so it is
# a drop-in for :func:`estimate_noise_adaptive` and feeds the same χ² weighting.
_QUADRATURE_TO_COMPLEX_RMS = float(np.sqrt(2.0))

# Algorithm tag carried in ``NoiseResult.bin_info``. Serialization keys off it to
# store the scatter σ verbatim (it is not reproducible from the moving-median
# reconstruction the adaptive estimator's round-trip uses).
SCATTER_ALGORITHM = "scatter_highpass_region_aware"


def _robust_sigma(values: np.ndarray) -> float:
    """1.4826 · MAD — robust Gaussian-σ estimator (0.0 on an empty input)."""
    if values.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def estimate_noise_scatter(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    *,
    window_mhz: float = SCATTER_WINDOW_MHZ,
    pedestal_mhz: float = SCATTER_PEDESTAL_MHZ,
    line_k: float = SCATTER_LINE_K,
    n_iter: int = SCATTER_N_ITER,
    region_aware: bool = True,
    smoothing_mhz: float = SCATTER_SMOOTHING_MHZ,
    smoothing_percentile: float = SCATTER_SMOOTHING_PERCENTILE,
    convolve_mhz: float = SCATTER_CONVOLVE_MHZ,
) -> NoiseResult:
    """Estimate frequency-dependent σ via a high-pass, region-aware scatter MAD.

    Drop-in replacement for :func:`estimate_noise_adaptive` that is immune to the
    leakage-pedestal over-estimation on high-SNR, line-dense spectra (see the
    module-level note and ``dev-docs/research/noise-snr-scaling/report.md``).

    Algorithm:

    1. Estimate the smooth leakage *pedestal* as a broad running median of |X|
       (width ``pedestal_mhz``). Line bins are interpolated over before each
       median pass so strong-line power does not pull the pedestal up near lines.
    2. The high-passed residual ``|X| − pedestal`` is white where there is only
       noise. Flag bins whose residual exceeds ``line_k`` robust-σ as lines and
       iterate (``n_iter`` passes) to refine the self-mask. Stage 2 precedes
       Stage 3, so there is no peak list to lean on.
    3. In each sliding ``window_mhz`` region take the MAD of the residual over the
       surviving non-line bins (the scatter), and convert it to the underlying
       complex σ. With ``region_aware`` the conversion uses the Rician lookup
       ``C(R)`` (``R = scatter / pedestal``); otherwise a fixed mid-regime factor.
    4. Interpolate σ across region centres and line positions, then smooth in two
       passes: a broad moving percentile (median by default) followed by a
       Gaussian convolution. Line skirts/sidelobes only *add* to the local
       scatter, so the noise floor is the lower envelope; a wide median is
       unbiased on clean spectrum yet rides the floor through line-dense bands
       instead of bumping up under them. The Gaussian then removes the median's
       staircase steps for a smooth floor without re-inflating it.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency values (MHz). Ascending or descending; the estimator works on
        the index grid, so either orientation is fine.
    magnitudes : np.ndarray
        Magnitude spectrum |X|.
    window_mhz : float, default ``SCATTER_WINDOW_MHZ``
        Full width of the per-region scatter-MAD window.
    pedestal_mhz : float, default ``SCATTER_PEDESTAL_MHZ``
        Running-median width that isolates the smooth leakage pedestal.
    line_k : float, default ``SCATTER_LINE_K``
        Robust-σ multiple above which a residual bin is flagged as a line.
    n_iter : int, default ``SCATTER_N_ITER``
        Self-mask refinement iterations.
    region_aware : bool, default True
        Use the Rician ``C(R)`` lookup (True) or the fixed mid-regime factor.
    smoothing_mhz : float, default ``SCATTER_SMOOTHING_MHZ``
        Width of the broad moving-percentile σ smoothing. ``0`` (or non-positive)
        disables smoothing and returns the raw per-region σ.
    smoothing_percentile : float, default ``SCATTER_SMOOTHING_PERCENTILE``
        Percentile of the smoothing filter. ``50`` is the median (robust to
        ≤50 % per-window line contamination, unbiased on clean spectrum); lower
        values give a more aggressive lower-envelope that de-inflates wide
        line-dense bands at the cost of a small clean-region low bias.
    convolve_mhz : float, default ``SCATTER_CONVOLVE_MHZ``
        Gaussian σ (MHz) of the second smoothing pass, applied after the median
        to remove its staircase steps. Because it acts on the already-de-inflated
        median output it cannot re-inflate σ under lines. ``0`` (or non-positive)
        disables this pass. Ignored when ``smoothing_mhz`` is ``0``.

    Returns
    -------
    NoiseResult
        ``rms_noise`` is the per-bin complex RMS σ_x (= σ_c·√2) on the full grid;
        ``noise_mask`` is True on the bins kept as noise by the self-mask;
        ``bin_info`` carries the algorithm tag, the knob values, and diagnostics.
    """
    if frequencies.shape != magnitudes.shape:
        raise ValueError("frequencies and magnitudes must have the same shape")
    if frequencies.ndim != 1 or magnitudes.ndim != 1:
        raise ValueError("Input arrays must be 1-dimensional")
    if n_iter < 1:
        raise ValueError("n_iter must be >= 1")
    if not 0.0 <= smoothing_percentile <= 100.0:
        raise ValueError("smoothing_percentile must be in [0, 100]")

    freqs = np.asarray(frequencies, dtype=float)
    mag = np.abs(np.asarray(magnitudes, dtype=float))
    n = freqs.size

    if n < 3:
        sigma = np.full(n, _robust_sigma(mag) * _QUADRATURE_TO_COMPLEX_RMS)
        return NoiseResult(
            rms_noise=sigma,
            noise_mask=np.ones(n, dtype=bool),
            bin_info=_scatter_bin_info(
                window_mhz,
                pedestal_mhz,
                line_k,
                n_iter,
                region_aware,
                smoothing_mhz=smoothing_mhz,
                smoothing_percentile=smoothing_percentile,
                convolve_mhz=convolve_mhz,
                n_line_bins=0,
                n_points=n,
                n_region_windows=0,
            ),
        )

    df = abs(float(np.mean(np.diff(freqs))))
    ped_size = max(7, int(round(pedestal_mhz / df)) | 1)
    half = max(1, int(round(0.5 * window_mhz / df)))

    # Iterative self-mask: interpolate masked (line) bins before estimating the
    # pedestal so strong-line power does not pull the pedestal up near lines.
    keep = np.ones(n, dtype=bool)
    ped = mag.copy()
    resid = mag.copy()
    for _ in range(n_iter):
        magc = mag.copy()
        if (~keep).any() and keep.any():
            magc[~keep] = np.interp(
                np.flatnonzero(~keep), np.flatnonzero(keep), mag[keep]
            )
        ped = median_filter(magc, size=ped_size)
        resid = mag - ped
        scale = _robust_sigma(resid[keep])
        if scale <= 0.0:
            scale = _robust_sigma(resid)
        keep = resid < line_k * scale

    sigma = np.full(n, np.nan)
    n_region_windows = 0
    for c in np.arange(0, n, half):
        lo, hi = max(0, c - half), min(n, c + half)
        m = keep[lo:hi]
        seg = resid[lo:hi][m]
        if seg.size < SCATTER_MIN_WINDOW_SAMPLES:
            continue
        scatter = _robust_sigma(seg)
        if region_aware:
            ped_lvl = float(np.median(ped[lo:hi][m]))
            R = scatter / ped_lvl if ped_lvl > 0 else float(_SCATTER_R_TAB.max())
            sigma[lo:hi] = scatter * float(np.interp(R, _SCATTER_R_TAB, _SCATTER_C_TAB))
        else:
            sigma[lo:hi] = scatter * FIXED_SCATTER_FACTOR
        n_region_windows += 1

    good = np.isfinite(sigma)
    if good.any():
        sigma = np.interp(np.arange(n), np.flatnonzero(good), sigma[good])
    else:
        # No window had enough surviving bins — fall back to a global scatter.
        sigma = np.full(
            n, _robust_sigma(resid[keep]) if keep.any() else _robust_sigma(resid)
        )

    sigma = sigma * _QUADRATURE_TO_COMPLEX_RMS

    # Broad lower-envelope smoothing: median (or lower percentile) over a wide
    # window rides the true noise floor through line-dense bands. Robust to the
    # ≤50 % per-window line contamination that bumps the raw per-region σ up.
    if smoothing_mhz > 0.0:
        smooth_size = max(3, int(round(smoothing_mhz / df)) | 1)
        sigma = percentile_filter(
            sigma, percentile=float(smoothing_percentile), size=smooth_size,
            mode="nearest",
        )
        # Second pass: a Gaussian removes the median's staircase. Acting on the
        # de-inflated median output, it smooths without re-inflating under lines.
        if convolve_mhz > 0.0:
            sigma = gaussian_filter1d(sigma, sigma=convolve_mhz / df, mode="nearest")

    bin_info = _scatter_bin_info(
        window_mhz,
        pedestal_mhz,
        line_k,
        n_iter,
        region_aware,
        smoothing_mhz=smoothing_mhz,
        smoothing_percentile=smoothing_percentile,
        convolve_mhz=convolve_mhz,
        n_line_bins=int(np.sum(~keep)),
        n_points=n,
        n_region_windows=n_region_windows,
    )
    return NoiseResult(rms_noise=sigma, noise_mask=keep, bin_info=bin_info)


def _scatter_bin_info(
    window_mhz: float,
    pedestal_mhz: float,
    line_k: float,
    n_iter: int,
    region_aware: bool,
    *,
    smoothing_mhz: float,
    smoothing_percentile: float,
    convolve_mhz: float,
    n_line_bins: int,
    n_points: int,
    n_region_windows: int,
) -> Dict[str, Union[np.ndarray, int, float, str]]:
    """Assemble the ``NoiseResult.bin_info`` diagnostics for the scatter estimator."""
    return {
        "algorithm": SCATTER_ALGORITHM,
        "window_mhz": float(window_mhz),
        "pedestal_mhz": float(pedestal_mhz),
        "line_k": float(line_k),
        "n_iter": int(n_iter),
        "region_aware": bool(region_aware),
        "smoothing_mhz": float(smoothing_mhz),
        "smoothing_percentile": float(smoothing_percentile),
        "convolve_mhz": float(convolve_mhz),
        "n_line_bins": int(n_line_bins),
        "n_region_windows": int(n_region_windows),
        "noise_fraction": float((n_points - n_line_bins) / max(n_points, 1)),
    }
