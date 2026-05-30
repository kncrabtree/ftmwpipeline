"""Self-contained scatter-based (high-pass) noise estimator.

Stage 2 runs BEFORE Stage 3, so the noise estimator cannot use a peak list -- it
self-masks. The thermal noise is the white (bin-uncorrelated) part of |X|; the
leakage pedestal is the smooth part. High-pass (|X| minus a broad running median)
isolates the noise; sharp excursions above the local scale are lines and are
masked iteratively. Robust (MAD) throughout.

  sigma(f) = DETREND_FACTOR * 1.4826 * MAD( |X| - medfilt(|X|) )  over non-line
             bins, in sliding windows so sigma tracks the real frequency profile.

DETREND_FACTOR maps |X|-scatter to the underlying complex-Gaussian sigma (Rician
high-pedestal limit ~1.0; Rayleigh limit ~0.93). Single mid-regime value here;
a regime-aware correction is the documented refinement.
"""
import numpy as np
from scipy.ndimage import median_filter

MAD_K = 1.4826
DETREND_FACTOR = 1.20


def _rmad(x):
    return MAD_K * np.median(np.abs(x - np.median(x)))


def estimate_sigma(f, spec, window_mhz=80.0, pedestal_mhz=20.0,
                   line_k=8.0, n_iter=3, return_mask=False):
    f = np.asarray(f, float)
    mag = np.abs(np.asarray(spec))
    n = f.size
    df = abs(np.mean(np.diff(f)))
    ped_size = max(7, int(round(pedestal_mhz / df)) | 1)
    half = int(round(0.5 * window_mhz / df))

    keep = np.ones(n, bool)
    resid = mag.copy()
    for _ in range(n_iter):
        magc = mag.copy()
        if (~keep).any() and keep.any():
            magc[~keep] = np.interp(np.flatnonzero(~keep),
                                    np.flatnonzero(keep), mag[keep])
        ped = median_filter(magc, size=ped_size)
        resid = mag - ped
        s = _rmad(resid[keep])
        keep = resid < line_k * s           # mask sharp (line) excursions

    sigma = np.full(n, np.nan)
    for c in np.arange(0, n, max(1, half)):
        lo, hi = max(0, c - half), min(n, c + half)
        seg = resid[lo:hi][keep[lo:hi]]
        if seg.size < 30:
            continue
        sigma[lo:hi] = MAD_K * np.median(np.abs(seg - np.median(seg))) * DETREND_FACTOR
    good = np.isfinite(sigma)
    if good.any():
        sigma = np.interp(np.arange(n), np.arange(n)[good], sigma[good])
    return (sigma, keep) if return_mask else sigma
