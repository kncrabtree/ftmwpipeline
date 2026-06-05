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
from typing import Any, Dict, Optional, Tuple, Union

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

# Region-aware Rician correction C(R) = σ_c / scatter, where R = scatter / pedestal
# selects the local pedestal/noise regime. Frozen Monte-Carlo reference table
# (M = 100 000, seed = 0); regenerable verbatim via the noise-snr-scaling
# prototype's ``_build_CR``. The Rayleigh-end MC scatter in C is part of the
# validated reference and is averaged over by the per-region windowing.
_SCATTER_R_TAB = np.array(
    [
        0.03986596,
        0.04082170,
        0.04199363,
        0.04314672,
        0.04392716,
        0.04513807,
        0.04599797,
        0.04714164,
        0.04834466,
        0.05017728,
        0.05161228,
        0.05315595,
        0.05441469,
        0.05617445,
        0.05765688,
        0.05981985,
        0.06225409,
        0.06448111,
        0.06661682,
        0.06929284,
        0.07214518,
        0.07538205,
        0.07831997,
        0.08230764,
        0.08582522,
        0.08997978,
        0.09478602,
        0.10021766,
        0.10627878,
        0.11308372,
        0.12010165,
        0.12783041,
        0.13848745,
        0.15058242,
        0.16410868,
        0.17921095,
        0.19831460,
        0.22243956,
        0.25294978,
        0.29016246,
        0.30734432,
        0.31370355,
        0.31995688,
        0.32878855,
        0.33599066,
        0.34114749,
        0.35159426,
        0.36126187,
        0.36920461,
        0.37736863,
        0.38960624,
        0.39594634,
        0.40604791,
        0.41872537,
        0.42767292,
        0.43793798,
        0.45029773,
        0.46199273,
        0.47202251,
        0.48345762,
        0.49670391,
        0.50579551,
        0.51433488,
        0.52263698,
        0.52946218,
        0.53538209,
        0.54391309,
        0.54984083,
        0.55411757,
        0.55956841,
        0.56129992,
        0.56202714,
        0.56286181,
        0.56287035,
        0.56367008,
        0.56378198,
        0.56442523,
        0.56460626,
        0.56461257,
        0.56562746,
    ]
)

_SCATTER_C_TAB = np.array(
    [
        1.00237940,
        1.00153229,
        0.99632158,
        0.99259848,
        0.99923264,
        0.99649888,
        1.00330809,
        1.00483687,
        1.00623928,
        0.99691948,
        0.99696346,
        0.99684148,
        1.00328269,
        1.00233410,
        1.00819421,
        1.00435034,
        0.99827075,
        0.99864254,
        1.00268474,
        1.00080615,
        1.00037696,
        0.99759210,
        1.00167353,
        0.99724406,
        1.00217515,
        1.00379491,
        1.00360366,
        1.00173589,
        1.00012860,
        0.99966342,
        1.00400361,
        1.01078569,
        1.00565146,
        1.00119711,
        1.00323225,
        1.01000980,
        1.01273321,
        1.01408565,
        1.01710622,
        1.02781026,
        1.02871528,
        1.03128887,
        1.03469078,
        1.03405033,
        1.03531733,
        1.04496886,
        1.04100395,
        1.04260191,
        1.04645777,
        1.05194337,
        1.04575131,
        1.06128545,
        1.06651893,
        1.06300956,
        1.07346248,
        1.08055241,
        1.08495677,
        1.09310003,
        1.10386198,
        1.11462597,
        1.12209720,
        1.13899321,
        1.16268326,
        1.18071045,
        1.20397788,
        1.22673695,
        1.24711990,
        1.27198521,
        1.29583333,
        1.32547974,
        1.46727774,
        1.50577996,
        1.34660719,
        1.37852501,
        1.48722828,
        1.40654222,
        1.50457734,
        1.46599646,
        1.49454816,
        1.42803168,
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
        return np.asarray(x, dtype=float)
    radius = int(truncate * sigma + 0.5)
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    # ``mode="nearest"`` == replicate the edge value over the kernel radius.
    padded = np.pad(np.asarray(x, dtype=float), radius, mode="edge")
    return spsig.fftconvolve(padded, kernel, mode="valid")


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
