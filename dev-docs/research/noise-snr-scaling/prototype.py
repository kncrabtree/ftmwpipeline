"""Self-contained scatter-based (high-pass), region-aware noise estimator.

Stage 2 runs BEFORE Stage 3, so the noise estimator cannot use a peak list -- it
self-masks. The thermal noise is the white (bin-uncorrelated) part of |X|; the
leakage pedestal is the smooth part. High-pass (|X| minus a broad running median)
isolates the noise; sharp excursions above the local scale are lines and are
masked iteratively. Robust (MAD) throughout.

  sigma(f) = C * 1.4826 * MAD( |X| - medfilt_pedestal(|X|) )   over non-line bins

The factor C maps the magnitude-scatter to the underlying complex-Gaussian sigma.
|X| is Rician, so that factor depends on the local pedestal/noise regime -- 1.0
under strong lines (Rician -> Gaussian), 1.47 in quiet Rayleigh regions. The
RATIO R = scatter/pedestal is a monotone function of the regime alone, so a
single 1-D lookup C(R) recovers the regime-correct factor from one spectrum (no
frames, no iteration). A fixed mid-regime factor is biased ~+-15% in a
regime-dependent way (validated against frame-difference truth); the region-aware
C(R) centers it. Production should bake the (R_TAB, C_TAB) arrays as constants
rather than simulating C(R) at import.
"""
import numpy as np
from scipy.ndimage import median_filter

MAD_K = 1.4826
FIXED_FACTOR = 1.20      # fallback mid-regime value when region_aware=False


def _rmad(x):
    return MAD_K * np.median(np.abs(x - np.median(x)))


def _build_CR(M=100000, seed=0):
    """C(R) by simulating the Rician magnitude over a grid of pedestal/noise
    ratios theta (sigma=1): R = scatter/pedestal, C = sigma/scatter = 1/scatter.
    Deterministic given the seed; bake the result as constants in production."""
    rng = np.random.default_rng(seed)
    thetas = np.concatenate([np.linspace(0.0, 3.0, 40), np.linspace(3.2, 25.0, 40)])
    Rs, Cs = [], []
    for th in thetas:
        mag = np.sqrt((th + rng.standard_normal(M)) ** 2 + rng.standard_normal(M) ** 2)
        Rs.append(_rmad(mag) / np.median(mag))
        Cs.append(1.0 / _rmad(mag))
    o = np.argsort(Rs)
    return np.asarray(Rs)[o], np.asarray(Cs)[o]


R_TAB, C_TAB = _build_CR()


def estimate_sigma(f, spec, window_mhz=80.0, pedestal_mhz=20.0, line_k=8.0,
                   n_iter=3, region_aware=True, return_mask=False):
    f = np.asarray(f, float)
    mag = np.abs(np.asarray(spec))
    n = f.size
    df = abs(np.mean(np.diff(f)))
    ped_size = max(7, int(round(pedestal_mhz / df)) | 1)
    half = int(round(0.5 * window_mhz / df))

    # iterative self-mask: interpolate masked (line) bins before estimating the
    # pedestal so strong-line power does not pull the pedestal up near lines
    keep = np.ones(n, bool)
    ped = mag.copy()
    resid = mag.copy()
    for _ in range(n_iter):
        magc = mag.copy()
        if (~keep).any() and keep.any():
            magc[~keep] = np.interp(np.flatnonzero(~keep),
                                    np.flatnonzero(keep), mag[keep])
        ped = median_filter(magc, size=ped_size)
        resid = mag - ped
        keep = resid < line_k * _rmad(resid[keep])

    sigma = np.full(n, np.nan)
    for c in np.arange(0, n, max(1, half)):
        lo, hi = max(0, c - half), min(n, c + half)
        m = keep[lo:hi]
        seg = resid[lo:hi][m]
        if seg.size < 30:
            continue
        scatter = MAD_K * np.median(np.abs(seg - np.median(seg)))
        if region_aware:
            ped_lvl = np.median(ped[lo:hi][m])
            R = scatter / ped_lvl if ped_lvl > 0 else R_TAB.max()
            sigma[lo:hi] = scatter * np.interp(R, R_TAB, C_TAB)
        else:
            sigma[lo:hi] = scatter * FIXED_FACTOR

    good = np.isfinite(sigma)
    if good.any():
        sigma = np.interp(np.arange(n), np.arange(n)[good], sigma[good])
    return (sigma, keep) if return_mask else sigma
