"""
Noise estimation for FTMW spectroscopy data.

Frequency-dependent noise estimator. The canonical FT is raw and unapodized,
so on high-SNR, line-dense spectra the far-wings of strong lines form a smooth
leakage *pedestal* that a level-based estimator would mistake for noise. The
:func:`estimate_noise_scatter` estimator high-passes the magnitude (subtracting
a broad running-median pedestal) and takes a region-aware, Rician-corrected
MAD of the residual over self-masked non-line bins, recovering the true σ(f)
floor. It emits the per-bin complex-RMS σ_x every later stage consumes.

The derivation, the 1/√N validation, and the region-aware C(R) calibration are
documented in ``dev-docs/research/noise-snr-scaling/report.md``. (That report
also contrasts the high-pass estimator against the retired level-based
"adaptive" estimator, whose minimal form survives only as a comparison
reference at ``dev-docs/research/noise-snr-scaling/legacy_adaptive.py``.)
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, cast

import numpy as np
import scipy.signal as spsig
from scipy.ndimage import median_filter, percentile_filter

logger = logging.getLogger(__name__)


@dataclass
class NoiseResult:
    """Result container for noise estimation.

    Attributes:
        rms_noise: σ_x estimate across the full frequency grid (= σ_c·√2).
        noise_mask: Boolean mask of points classified as noise by the
            estimator's self-mask. Used by downstream consumers that need
            to discriminate noise vs signal samples; not used in the σ
            computation itself.
        bin_info: Dictionary of estimator diagnostics.
    """

    rms_noise: np.ndarray
    noise_mask: np.ndarray
    bin_info: Dict[str, Union[np.ndarray, int, float, str]]


# ---------------------------------------------------------------------------
# Scatter (high-pass), region-aware noise estimator — the Stage 2 estimator.
#
# The canonical FT is raw and un-apodized (boxcar), so on high-SNR, line-dense
# spectra the summed far-wings of strong lines form a smooth leakage *pedestal*
# that fills every quiet bin. A level-based estimator measures that pedestal,
# not the random noise, and over-reports σ by up to ~6×
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
SCATTER_WINDOW_MHZ = 80.0  # full width of the per-region scatter-MAD window
SCATTER_PEDESTAL_MHZ = 20.0  # running-median width isolating the leakage pedestal
SCATTER_LINE_K = 8.0  # robust-σ multiple above which a bin is flagged a line
SCATTER_N_ITER = 3  # self-mask refinement iterations
SCATTER_MIN_WINDOW_SAMPLES = 30  # minimum surviving non-line bins per region window
SCATTER_SMOOTHING_MHZ = 800.0  # broad moving-percentile σ smoothing width (0 = off)
SCATTER_SMOOTHING_PERCENTILE = 50.0  # 50 = median (unbiased); lower = lower-envelope
SCATTER_CONVOLVE_MHZ = (
    200.0  # Gaussian σ (MHz) of the 2nd, step-removing pass (0 = off)
)

# Fixed mid-regime σ_c/scatter factor used when ``region_aware=False`` (the
# Rayleigh-to-under-line endpoints span 1.0–1.47; 1.20 is the validated
# mid-regime compromise — biased ~±15 % in a regime-dependent way, which is
# exactly what the region-aware lookup removes).
FIXED_SCATTER_FACTOR = 1.20

# Region-aware Rician correction C(R) = sigma_c / scatter, where R = scatter / pedestal
# selects the local pedestal/noise regime. The true C(R) is smooth and monotone
# increasing (Rician theory): ~1.0 under strong lines (pedestal >> sigma, Rician ->
# Gaussian) rising to ~1.5 in quiet Rayleigh regions. R saturates near the Rayleigh
# end so the raw Monte-Carlo cloud is noisy there; the baked table is its isotonic
# (monotone) fit resampled onto a clean ascending R grid. Regenerable verbatim via
# ``docs/source/methods/noise_snr_scaling/generate.py`` (build_cr_table, M=1e6, seed=0).
_SCATTER_R_TAB = np.array(
    [
        0.03994648,
        0.04659835,
        0.05325021,
        0.05990208,
        0.06655395,
        0.07320581,
        0.07985768,
        0.08650955,
        0.09316142,
        0.09981328,
        0.10646515,
        0.11311702,
        0.11976888,
        0.12642075,
        0.13307262,
        0.13972449,
        0.14637635,
        0.15302822,
        0.15968009,
        0.16633196,
        0.17298382,
        0.17963569,
        0.18628756,
        0.19293942,
        0.19959129,
        0.20624316,
        0.21289503,
        0.21954689,
        0.22619876,
        0.23285063,
        0.23950249,
        0.24615436,
        0.25280623,
        0.25945810,
        0.26610996,
        0.27276183,
        0.27941370,
        0.28606557,
        0.29271743,
        0.29936930,
        0.30602117,
        0.31267303,
        0.31932490,
        0.32597677,
        0.33262864,
        0.33928050,
        0.34593237,
        0.35258424,
        0.35923610,
        0.36588797,
        0.37253984,
        0.37919171,
        0.38584357,
        0.39249544,
        0.39914731,
        0.40579918,
        0.41245104,
        0.41910291,
        0.42575478,
        0.43240664,
        0.43905851,
        0.44571038,
        0.45236225,
        0.45901411,
        0.46566598,
        0.47231785,
        0.47896972,
        0.48562158,
        0.49227345,
        0.49892532,
        0.50557718,
        0.51222905,
        0.51888092,
        0.52553279,
        0.53218465,
        0.53883652,
        0.54548839,
        0.55214025,
        0.55879212,
        0.56544399,
    ]
)

_SCATTER_C_TAB = np.array(
    [
        1.00018072,
        1.00037659,
        1.00062868,
        1.00072329,
        1.00072329,
        1.00072329,
        1.00227117,
        1.00227117,
        1.00227117,
        1.00227117,
        1.00227117,
        1.00229212,
        1.00291084,
        1.00457574,
        1.00457574,
        1.00457574,
        1.00515989,
        1.00597999,
        1.00634478,
        1.00739121,
        1.00806536,
        1.00848672,
        1.00904552,
        1.00962742,
        1.01039346,
        1.01115950,
        1.01221210,
        1.01328489,
        1.01396977,
        1.01426013,
        1.01455048,
        1.01501093,
        1.01560437,
        1.01619780,
        1.01698782,
        1.01905726,
        1.02112671,
        1.02319615,
        1.02489027,
        1.02568037,
        1.02647047,
        1.02667095,
        1.02988343,
        1.03297633,
        1.03498834,
        1.03608236,
        1.03840777,
        1.03866617,
        1.04214364,
        1.04531648,
        1.04719353,
        1.04893509,
        1.05064931,
        1.05262505,
        1.05539757,
        1.05842083,
        1.06208205,
        1.06585226,
        1.06870058,
        1.07404263,
        1.08162002,
        1.08410347,
        1.08600320,
        1.09141726,
        1.10141929,
        1.10667228,
        1.11189625,
        1.11825784,
        1.12597351,
        1.13708362,
        1.14532592,
        1.15964385,
        1.17570200,
        1.18920254,
        1.20846742,
        1.22482229,
        1.25543050,
        1.28587148,
        1.35401785,
        1.49890732,
    ]
)

# C(R) recovers the per-quadrature σ_c; the canonical Stage 2 ``rms_noise`` is the
# complex RMS σ_x = σ_c·√2 (real/imag each carry σ_x²/2 — see
# ``fitting/validation.py``). The estimator scales its σ_c output to σ_x, the
# convention the downstream χ² weighting consumes.
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


def _gaussian_smooth_1d(
    x: np.ndarray, sigma: float, truncate: float = 4.0
) -> np.ndarray:
    """1-D Gaussian smoothing equivalent to ``scipy.ndimage.gaussian_filter1d``
    (order 0, ``mode="nearest"``) but evaluated by FFT convolution.

    The broad σ-smoothing of the scatter estimator uses a Gaussian whose width
    is a fixed *frequency* span (``convolve_mhz``); on a fine detection grid
    that is tens of thousands of bins, where the direct spatial correlation in
    ``gaussian_filter1d`` is O(N · kernel) and dominates the whole estimator.
    Replicating ``mode="nearest"`` by edge-padding and convolving the *same*
    normalized Gaussian kernel via FFT is O(N log N) and matches the direct
    result to floating-point round-off (~1e-14 relative).
    """
    sigma = float(sigma)
    if sigma <= 0.0:
        return cast(np.ndarray, np.asarray(x, dtype=float))
    radius = int(truncate * sigma + 0.5)
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    # ``mode="nearest"`` == replicate the edge value over the kernel radius.
    padded = np.pad(np.asarray(x, dtype=float), radius, mode="edge")
    return cast(np.ndarray, spsig.fftconvolve(padded, kernel, mode="valid"))


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

    The Stage 2 noise estimator. Immune to the
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
    4. Interpolate σ across region centers and line positions, then smooth in two
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
    # Clamp filter windows to the data length: a smoothing window can never
    # exceed the spectrum, and an oversized rank-filter footprint (on a coarse
    # grid where the MHz width spans more bins than exist) is O(N·window) and
    # blows up. The Gaussian pass is FFT-based and needs no such clamp.
    ped_size = min(max(7, int(round(pedestal_mhz / df)) | 1), n)
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
        smooth_size = min(max(3, int(round(smoothing_mhz / df)) | 1), n)
        sigma = percentile_filter(
            sigma,
            percentile=float(smoothing_percentile),
            size=smooth_size,
            mode="nearest",
        )
        # Second pass: a Gaussian removes the median's staircase. Acting on the
        # de-inflated median output, it smooths without re-inflating under lines.
        if convolve_mhz > 0.0:
            sigma = _gaussian_smooth_1d(sigma, sigma=convolve_mhz / df)

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


def estimate_active_ft_noise(
    freq_mhz: np.ndarray,
    complex_spectrum: np.ndarray,
    **scatter_kwargs: Any,
) -> NoiseResult:
    """Estimate per-bin σ on an active-FT spectrum, in its native bin order.

    The single noise-authority surface: it runs the scatter estimator
    (:func:`estimate_noise_scatter`) on the magnitude of an active-portion FT
    (the ``dt_us * rfft(active)`` spectrum :func:`compute_active_ft` produces)
    and returns the result re-expressed on the *input* bin order. The scatter
    estimator works on a monotonic frequency axis, but an active FT for a lower
    sideband is descending; this wrapper sorts to ascending, estimates, then
    un-sorts ``rms_noise`` and ``noise_mask`` back onto ``freq_mhz``'s order so
    the σ array lines up with ``complex_spectrum`` element-for-element.

    Parameters
    ----------
    freq_mhz : np.ndarray
        Molecular frequency grid of the active FT (ascending or descending).
    complex_spectrum : np.ndarray
        Complex active FT on ``freq_mhz`` (``dt_us * rfft`` convention).
    **scatter_kwargs
        Forwarded verbatim to :func:`estimate_noise_scatter` (the resolved
        Stage 2 scatter knobs).

    Returns
    -------
    NoiseResult
        ``rms_noise`` (per-bin σ_x) and ``noise_mask`` on ``freq_mhz``'s bin
        order; ``bin_info`` carries the scatter diagnostics unchanged.
    """
    freq = np.asarray(freq_mhz, dtype=float)
    mag = np.abs(np.asarray(complex_spectrum))
    if freq.shape != mag.shape:
        raise ValueError("freq_mhz and complex_spectrum must have the same shape")

    sort_idx = np.argsort(freq)
    sorted_freq = np.ascontiguousarray(freq[sort_idx])
    sorted_mag = np.ascontiguousarray(mag[sort_idx])

    result = estimate_noise_scatter(sorted_freq, sorted_mag, **scatter_kwargs)

    unsort = np.argsort(sort_idx)
    return NoiseResult(
        rms_noise=np.asarray(result.rms_noise, dtype=float)[unsort],
        noise_mask=np.asarray(result.noise_mask, dtype=bool)[unsort],
        bin_info=result.bin_info,
    )


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
