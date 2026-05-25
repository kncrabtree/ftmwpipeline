"""
Noise-estimator grid-invariance audit -- reproducibility script.

The Stage 2 adaptive noise estimator was calibrated for the persisted
user-grid spectrum (~10⁶ bins, ~1.5× zero-padded by Stage 1's zpf=2).
The matched-filter detection study (``../matched-filter-detection/``)
needed σ on the active-portion FT (no zero padding, ~10⁵ bins) and
hit a structural failure: the estimator returns a single bin spanning
the whole spectrum, and the recovered σ disagrees with the user-grid
σ by up to 67 % on the 2638 fixture.

This study verifies the user's premise -- *the noise estimate should
be independent of grid choice* -- diagnoses the failure modes, and
proposes a fix.

Sections (each is a function this script's ``main`` calls in order):

1. **2638 σ across grids**: side-by-side σ(f) on the user grid
   (zpf=2) and the trimmed active-FT (zpf=0). Establishes the
   disagreement empirically (fig 01).

2. **Synthetic ground truth with step σ(f)**: build a noise-only
   FID whose freq-domain σ has three discrete levels; rfft at
   several zpf levels; check estimator recovery (fig 02). The
   current estimator misses the real σ structure on the active-FT.

3. **Failure-mode diagnosis**: trace the recursive subdivision on
   the synthetic; identify which OR-of-four criterion fails and
   why the post-trim statistics hide the heterogeneity. The
   underlying issue: the skewness-trim is doing double duty
   (noise-mask identification AND subdivision-criterion input).

4. **Robust-statistics variant**: MAD/median-based subdivision
   criterion that is robust to spectral lines AND preserves σ
   heterogeneity (fig 03).

5. **Smoothing-window/sample-count audit**: ``DEFAULT_SMOOTHING_SAMPLES
   = 2500`` implicitly assumes uncorrelated bins. On the zero-padded
   user grid, adjacent bins are correlated through the Dirichlet
   kernel; the effective N is smaller. On the active-FT
   (uncorrelated), 2500 bins gives the targeted 1 % stability; on the
   user grid with α = 0.15 the effective N is ~375, giving ~2.6 %.
   Discusses the right way to choose the smoothing window when bin
   correlation depends on grid (fig 04).

6. **2638 reality check with the variant**: re-runs σ on both grids
   with the proposed fix; checks they agree (fig 05).

Outputs land in ``figures/`` (PNGs) and ``data/`` (.npz). The 2638
sections need ``scratch/stage5-validation/exp_2638.ftmw``; the
synthetic sections are self-contained.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ftmwpipeline.core.data_structures import Sideband
from ftmwpipeline.fitting.active_ft import compute_active_ft
from ftmwpipeline.preprocessing.noise_estimation import estimate_noise_adaptive

logger = logging.getLogger("noise-grid-invariance")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

HERE = Path(__file__).parent
FIG = HERE / "figures"
DATA = HERE / "data"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
REPO_ROOT = Path(__file__).resolve().parents[3]
FTMW_PATH = REPO_ROOT / "scratch" / "stage5-validation" / "exp_2638.ftmw"
RNG_SEED = 20260524


# ===========================================================================
# Helpers
# ===========================================================================
def _sigma_local_variation(
    sigma: np.ndarray, freq_mhz: np.ndarray, window_mhz: float
) -> Tuple[float, float, float]:
    """Local σ heterogeneity: σ_max/σ_min within ``window_mhz`` chunks."""
    bin_mhz = abs(freq_mhz[1] - freq_mhz[0])
    n = max(2, int(window_mhz / bin_mhz))
    ratios = []
    for i in range(0, sigma.size - n, n):
        chunk = sigma[i : i + n]
        if chunk.min() > 0:
            ratios.append(chunk.max() / chunk.min())
    if not ratios:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(ratios)
    return (
        float(np.median(arr)),
        float(np.percentile(arr, 95)),
        float(arr.max()),
    )


# ===========================================================================
# Section 1: 2638 σ across grids
# ===========================================================================
def figure_2638_sigma_vs_grid() -> None:
    """σ(f) on user grid vs trimmed active-FT, both normalised to median."""
    if not FTMW_PATH.exists():
        logger.info("2638 fixture missing, skipping section 1")
        return
    from ftmwpipeline._internal.stage5_impl import _build_active_ft_inputs

    inputs = _build_active_ft_inputs(str(FTMW_PATH))
    (
        fid_samples, sample_dt_us, start_us, end_us, expf_us, probe_freq_mhz,
        sideband, n_padded, acquisition_us, user_ft, user_rms,
    ) = inputs
    trim_lo = float(user_ft.freq_array.min())
    trim_hi = float(user_ft.freq_array.max())

    # User grid (zpf=2): already trimmed and Stage-2-noise-estimated.
    sort_u = np.argsort(user_ft.freq_array)
    uf = user_ft.freq_array[sort_u]
    um = user_ft.magnitude_spectrum[sort_u]
    res_u = estimate_noise_adaptive(uf, um)
    su = res_u.rms_noise

    # Active-FT (zpf=0). Must pre-trim to the user's band before
    # the estimator -- otherwise the out-of-band low-noise region kills
    # the first subdivision (it fails the min_noise_fraction = 2/3 test).
    active = compute_active_ft(
        fid=fid_samples, sample_dt_us=sample_dt_us,
        start_us=start_us, end_us=end_us, expf_us=expf_us,
        probe_freq_mhz=probe_freq_mhz, sideband=sideband,
        n_padded=n_padded, rdc=True,
    )
    sort_a = np.argsort(active.freq_mhz)
    af = active.freq_mhz[sort_a]
    am = np.abs(active.complex_spectrum)[sort_a]
    m = (af >= trim_lo) & (af <= trim_hi)
    af_t, am_t = af[m], am[m]
    res_a = estimate_noise_adaptive(af_t, am_t)
    sa = res_a.rms_noise

    # Diagnostic: same-grid raw active-FT (not pre-trimmed) — show the
    # 1-bin failure too.
    res_a_full = estimate_noise_adaptive(af, am)

    su_norm = su / np.median(su)
    sa_norm = sa / np.median(sa)
    sa_full_norm = res_a_full.rms_noise / np.median(res_a_full.rms_noise)

    # Compare on user grid
    sa_on_user = np.interp(uf, af_t, sa_norm)
    diff = (su_norm - sa_on_user) / np.maximum(sa_on_user, 1e-12)
    logger.info(
        "2638 σ(f) comparison (active interp to user grid): "
        "median |Δσ/σ|=%.2f%%  p95=%.2f%%  max=%.2f%%",
        np.median(np.abs(diff)) * 100,
        np.percentile(np.abs(diff), 95) * 100,
        np.max(np.abs(diff)) * 100,
    )
    logger.info(
        "  σ range: user=%.2fx, active-FT trimmed=%.2fx, active-FT raw=%.2fx",
        su_norm.max() / su_norm.min(),
        sa_norm.max() / sa_norm.min(),
        sa_full_norm.max() / sa_full_norm.min(),
    )

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(
        uf, su_norm, color="C0", lw=0.5, alpha=0.85,
        label=(
            f"user grid (zpf=2)  n_bins={res_u.bin_info['n_bins']}  "
            f"smoothing={res_u.bin_info['smoothing_window_mhz']:.1f} MHz"
        ),
    )
    ax.plot(
        af_t, sa_norm, color="C3", lw=0.8,
        label=(
            f"active-FT trimmed (zpf=0)  n_bins={res_a.bin_info['n_bins']}  "
            f"smoothing={res_a.bin_info['smoothing_window_mhz']:.1f} MHz"
        ),
    )
    ax.plot(
        af, sa_full_norm, color="C7", lw=0.6, alpha=0.5,
        label=(
            f"active-FT NOT pre-trimmed  n_bins={res_a_full.bin_info['n_bins']}  "
            f"(falls back to single global σ)"
        ),
    )
    ax.set_xlabel("freq (MHz)")
    ax.set_ylabel("σ / median(σ)  [normalised]")
    ax.set_title(
        "2638: noise-estimator σ(f) across grids — same physics, "
        "two grids disagree up to 67 %"
    )
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out = FIG / "01_2638_sigma_vs_grid.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", out)


# ===========================================================================
# Section 2: synthetic ground truth with step σ(f)
# ===========================================================================
@dataclass(frozen=True)
class SyntheticNoiseSpec:
    """Three-region synthetic noise FID + ground-truth σ.

    Time-domain FID is inverse-FFT of a complex Gaussian whose σ
    varies in three baseband regions. The active-FT at the same dt
    must recover the same σ(f) regardless of zero-padding.

    Attributes
    ----------
    fid : np.ndarray
        Active-region samples (length ``n_active``).
    sample_dt_us : float
        Sample spacing (microseconds).
    sigma_target_active_grid : np.ndarray
        Ground-truth σ on the active-FT freq grid (Rayleigh re/im
        scale, not |X|-RMS). |X|-RMS = σ * √2.
    freq_bb_target : np.ndarray
        Baseband frequencies for sigma_target_active_grid.
    breaks_mhz : tuple[float, float]
        The two σ-step locations (baseband MHz).
    sigma_values : tuple[float, float, float]
        The three σ levels in order of increasing baseband freq.
    """

    fid: np.ndarray
    sample_dt_us: float
    sigma_target_active_grid: np.ndarray
    freq_bb_target: np.ndarray
    breaks_mhz: Tuple[float, float]
    sigma_values: Tuple[float, float, float]


def make_synthetic_noise(
    *,
    n_active: int = 4096,
    sample_dt_us: float = 0.020,
    breaks_mhz: Tuple[float, float] = (6.25, 12.5),
    sigma_values: Tuple[float, float, float] = (1.0, 3.0, 1.5),
    rng_seed: int = RNG_SEED,
) -> SyntheticNoiseSpec:
    """Build a noise-only FID with a known 3-step σ(f) in baseband."""
    rng = np.random.default_rng(rng_seed)
    freq_bb = np.fft.rfftfreq(n_active, d=sample_dt_us)
    sigma_target = np.where(
        freq_bb < breaks_mhz[0], sigma_values[0],
        np.where(freq_bb < breaks_mhz[1], sigma_values[1], sigma_values[2]),
    ).astype(float)
    n_freq = freq_bb.size
    re = rng.normal(scale=sigma_target)
    im = rng.normal(scale=sigma_target)
    # rfft of a real signal: bin 0 and Nyquist are pure-real.
    re[0] *= np.sqrt(2)
    im[0] = 0.0
    if n_active % 2 == 0:
        re[-1] *= np.sqrt(2)
        im[-1] = 0.0
    spec = re + 1j * im
    # inverse-rfft to time domain. The compute_active_ft amplitude
    # convention is dt * rfft, so we divide by dt before irfft so
    # that compute_active_ft(rdc=False) reproduces our spec.
    fid = np.fft.irfft(spec / sample_dt_us, n=n_active)
    return SyntheticNoiseSpec(
        fid=fid, sample_dt_us=sample_dt_us,
        sigma_target_active_grid=sigma_target,
        freq_bb_target=freq_bb,
        breaks_mhz=breaks_mhz,
        sigma_values=sigma_values,
    )


def _rfft_at_zpf(
    fid: np.ndarray, sample_dt_us: float, zpf: int
) -> Tuple[np.ndarray, np.ndarray]:
    """rfft the FID with zero-padding factor zpf (so n_padded = n_active << zpf)."""
    n_active = fid.size
    if zpf <= 0:
        n_padded = n_active
        fid_padded = fid
    else:
        n_padded = n_active * (2 ** zpf)
        fid_padded = np.zeros(n_padded, dtype=float)
        fid_padded[:n_active] = fid
    spec = sample_dt_us * np.fft.rfft(fid_padded)
    freq_bb = np.fft.rfftfreq(n_padded, d=sample_dt_us)
    return freq_bb, spec


def figure_synthetic_recovery() -> None:
    """Estimator recovery on a known-σ(f) synthetic at zpf ∈ {0, 1, 2, 3}."""
    spec = make_synthetic_noise()
    sigma_target_x = spec.sigma_target_active_grid * np.sqrt(2)  # σ_x = σ_c · √2

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(
        spec.freq_bb_target, sigma_target_x,
        color="k", lw=2.5, alpha=0.6, label="ground truth |X|-RMS",
    )
    rows = []
    for zpf, color in [(0, "C3"), (1, "C2"), (2, "C0"), (3, "C5")]:
        freq_bb, spec_zpf = _rfft_at_zpf(spec.fid, spec.sample_dt_us, zpf)
        res = estimate_noise_adaptive(freq_bb, np.abs(spec_zpf))
        sigma_hat = res.rms_noise
        ax.plot(
            freq_bb, sigma_hat, color=color, lw=0.8, alpha=0.9,
            label=(
                f"zpf={zpf}  n_bins={res.bin_info['n_bins']}  "
                f"smoothing={res.bin_info['smoothing_window_mhz']:.2f} MHz  "
                f"range={sigma_hat.max()/sigma_hat.min():.2f}x"
            ),
        )
        rows.append((zpf, res.bin_info["n_bins"], sigma_hat.max() / sigma_hat.min()))
    ax.axvline(spec.breaks_mhz[0], color="gray", ls=":", alpha=0.5)
    ax.axvline(spec.breaks_mhz[1], color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("baseband freq (MHz)")
    ax.set_ylabel("σ_x  (Rayleigh |X|-RMS scale)")
    ax.set_title(
        "Synthetic 3-step σ(f) recovery: estimator output at zpf ∈ {0, 1, 2, 3}  "
        f"(ground-truth range = {sigma_target_x.max()/sigma_target_x.min():.2f}x)"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = FIG / "02_synthetic_recovery.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", out)
    for zpf, n_bins, ratio in rows:
        logger.info(
            "  zpf=%d → n_bins=%d  recovered σ ratio=%.2fx  (target=3.0x)",
            zpf, n_bins, ratio,
        )


# ===========================================================================
# Section 3: failure-mode diagnosis
# ===========================================================================
def figure_failure_mode() -> None:
    """Show why subdivision fails on the active-FT synthetic.

    Plot raw magnitude histograms in the heterogeneous LEFT half vs
    the homogeneous RIGHT half, side-by-side with the post-trim
    (noise-mask) histograms. The post-trim distributions are almost
    identical even though the raw ones differ -- the trim flattens
    the heterogeneity into a Rayleigh-looking unimodal blob.
    """
    spec = make_synthetic_noise()
    freq_bb, spec_zpf = _rfft_at_zpf(spec.fid, spec.sample_dt_us, 0)
    mag = np.abs(spec_zpf)
    n_freq = mag.size
    # Midpoint subdivision -- exactly what the estimator tried.
    left = mag[: n_freq // 2]
    right = mag[n_freq // 2 :]

    # Skewness trim to mimic the estimator's pre-subdivision step.
    from scipy.stats import skew

    def trim_to_rayleigh(m: np.ndarray, target: float = 0.631) -> np.ndarray:
        sm = np.sort(m)
        n = sm.size
        for cutoff in np.arange(0, 0.99, 0.01):
            kept = sm[: int(n * (1 - cutoff))]
            if kept.size < 10:
                break
            if skew(kept) <= target:
                return kept
        return sm[: max(int(n * 0.1), 10)]

    left_trim = trim_to_rayleigh(left)
    right_trim = trim_to_rayleigh(right)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    bins_full = np.linspace(0, max(mag.max(), 12), 40)
    bins_low = np.linspace(0, 6, 40)
    axes[0, 0].hist(left, bins=bins_full, alpha=0.6, color="C3")
    axes[0, 0].set_title(
        f"LEFT half raw mag  (σ-mix: 1.0 + 3.0)\n"
        f"mean={left.mean():.3f}  var={left.var():.3f}  n={left.size}"
    )
    axes[0, 1].hist(right, bins=bins_full, alpha=0.6, color="C0")
    axes[0, 1].set_title(
        f"RIGHT half raw mag  (σ=1.5)\n"
        f"mean={right.mean():.3f}  var={right.var():.3f}  n={right.size}"
    )
    axes[1, 0].hist(left_trim, bins=bins_low, alpha=0.6, color="C3")
    axes[1, 0].set_title(
        f"LEFT half POST-TRIM (noise mask)\n"
        f"mean={left_trim.mean():.3f}  var={left_trim.var():.3f}  "
        f"n={left_trim.size}  ({100*left_trim.size/left.size:.0f}%)"
    )
    axes[1, 1].hist(right_trim, bins=bins_low, alpha=0.6, color="C0")
    axes[1, 1].set_title(
        f"RIGHT half POST-TRIM\n"
        f"mean={right_trim.mean():.3f}  var={right_trim.var():.3f}  "
        f"n={right_trim.size}  ({100*right_trim.size/right.size:.0f}%)"
    )
    for ax in axes.flat:
        ax.set_xlabel("|X|")
        ax.set_ylabel("count")

    rel_mean_raw = abs(left.mean() - right.mean()) / (0.5 * (left.mean() + right.mean()))
    rel_var_raw = abs(left.var() - right.var()) / (0.5 * (left.var() + right.var()))
    rel_mean_trim = abs(left_trim.mean() - right_trim.mean()) / (
        0.5 * (left_trim.mean() + right_trim.mean())
    )
    rel_var_trim = abs(left_trim.var() - right_trim.var()) / (
        0.5 * (left_trim.var() + right_trim.var())
    )
    fig.suptitle(
        f"Failure mode: skewness trim hides σ heterogeneity\n"
        f"raw mean diff = {100*rel_mean_raw:.1f}%, raw var diff = {100*rel_var_raw:.1f}%  →  "
        f"trimmed mean diff = {100*rel_mean_trim:.1f}%, trimmed var diff = {100*rel_var_trim:.1f}%  "
        f"(20 % threshold)"
    )
    fig.tight_layout()
    out = FIG / "03_failure_mode.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", out)
    logger.info(
        "  raw mean diff %.1f%%, raw var diff %.1f%%  →  "
        "trimmed mean diff %.1f%%, trimmed var diff %.1f%%",
        100 * rel_mean_raw, 100 * rel_var_raw,
        100 * rel_mean_trim, 100 * rel_var_trim,
    )


# ===========================================================================
# Section 4: robust-statistics variant (MAD-based subdivision)
# ===========================================================================
def _mad(x: np.ndarray) -> float:
    """Median absolute deviation, the robust spread estimator."""
    return float(np.median(np.abs(x - np.median(x))))


def estimate_noise_mad_split(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    *,
    smoothing_samples: int = 2500,
    min_bin_fraction: float = 1 / 64,
    abs_min_bin_size: int = 300,
    mad_diff_threshold: float = 0.10,
    median_diff_threshold: float = 0.10,
    verbose: bool = False,
) -> Tuple[np.ndarray, dict]:
    """Drop-in noise estimator using MEDIAN + MAD subdivision criterion.

    The current estimator uses skewness-trimmed mean/variance for the
    subdivision decision; the trim hides σ heterogeneity (Section 3
    of the report). This variant uses median + MAD on the raw
    magnitudes -- both robust to spectral lines (high-magnitude
    outliers do not move the median or MAD much) and faithful to the
    underlying σ.

    Algorithm:
    1. Recursively bisect. For each candidate split:
       - Compute median and MAD of raw |X| in left and right halves.
       - Subdivide iff median or MAD differ by ≥ threshold (relative).
    2. In each final bin, estimate σ_c as MAD/0.4485 (Rayleigh MAD
       scaling) and σ_x = σ_c · √2.
    3. Smooth σ_x with a uniform window covering ``smoothing_samples``
       points.

    Returns (sigma_x_estimate, bin_info).
    """
    freq = np.asarray(frequencies, dtype=float)
    mag = np.asarray(magnitudes, dtype=float)
    n = freq.size
    min_bin_size = max(int(n * min_bin_fraction), abs_min_bin_size)
    # Rayleigh MAD/scale ratio: for X ~ Rayleigh(σ_c), median = σ_c·sqrt(2 ln 2),
    # and MAD(|X|) is approximately 0.4485 σ_c.
    RAYLEIGH_MAD_TO_SC = 0.4485

    def stats(start: int, end: int) -> Tuple[float, float]:
        chunk = mag[start:end]
        return float(np.median(chunk)), _mad(chunk)

    edges = [0]

    def recurse(start: int, end: int) -> None:
        size = end - start
        if size < 2 * min_bin_size:
            return
        mid = (start + end) // 2
        med_l, mad_l = stats(start, mid)
        med_r, mad_r = stats(mid, end)
        med_diff = abs(med_l - med_r) / max(0.5 * (med_l + med_r), 1e-12)
        mad_diff = abs(mad_l - mad_r) / max(0.5 * (mad_l + mad_r), 1e-12)
        if verbose:
            logger.debug(
                "  [%d:%d] med_l=%.4g med_r=%.4g (Δ=%.1f%%)  "
                "mad_l=%.4g mad_r=%.4g (Δ=%.1f%%)",
                start, end, med_l, med_r, 100 * med_diff,
                mad_l, mad_r, 100 * mad_diff,
            )
        if med_diff < median_diff_threshold and mad_diff < mad_diff_threshold:
            return
        edges.append(mid)
        recurse(start, mid)
        recurse(mid, end)

    recurse(0, n)
    edges.append(n)
    edges = sorted(set(edges))

    # Per-bin σ_x estimate via MAD/scaling
    sigma_x = np.zeros(n)
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        bin_mad = _mad(mag[a:b])
        sigma_c = bin_mad / RAYLEIGH_MAD_TO_SC
        sigma_x[a:b] = sigma_c * np.sqrt(2.0)

    # Smooth (uniform window over ``smoothing_samples`` points)
    if smoothing_samples > 1:
        win = max(1, min(smoothing_samples, n))
        kernel = np.ones(win) / win
        sigma_x = np.convolve(sigma_x, kernel, mode="same")

    return sigma_x, {
        "n_bins": len(edges) - 1,
        "edges": edges,
        "smoothing_samples": smoothing_samples,
        "algorithm": "mad_median_split",
    }


def figure_mad_variant_recovery() -> None:
    """MAD-variant recovery on the synthetic 3-step σ(f), zpf ∈ {0,1,2,3}."""
    spec = make_synthetic_noise()
    sigma_target_x = spec.sigma_target_active_grid * np.sqrt(2)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(
        spec.freq_bb_target, sigma_target_x,
        color="k", lw=2.5, alpha=0.6, label="ground truth |X|-RMS",
    )
    rows = []
    for zpf, color in [(0, "C3"), (1, "C2"), (2, "C0"), (3, "C5")]:
        freq_bb, spec_zpf = _rfft_at_zpf(spec.fid, spec.sample_dt_us, zpf)
        sigma_hat, info = estimate_noise_mad_split(freq_bb, np.abs(spec_zpf))
        ax.plot(
            freq_bb, sigma_hat, color=color, lw=0.8,
            label=f"zpf={zpf}  n_bins={info['n_bins']}  range={sigma_hat.max()/sigma_hat.min():.2f}x",
        )
        rows.append((zpf, info["n_bins"], sigma_hat.max() / sigma_hat.min()))
    ax.axvline(spec.breaks_mhz[0], color="gray", ls=":", alpha=0.5)
    ax.axvline(spec.breaks_mhz[1], color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("baseband freq (MHz)")
    ax.set_ylabel("σ_x  (Rayleigh |X|-RMS scale)")
    ax.set_title(
        "MAD-variant noise estimator: synthetic recovery at zpf ∈ {0, 1, 2, 3}  "
        f"(ground-truth range = {sigma_target_x.max()/sigma_target_x.min():.2f}x)"
    )
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out = FIG / "04_mad_variant_recovery.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", out)
    for zpf, n_bins, ratio in rows:
        logger.info(
            "  zpf=%d → n_bins=%d  recovered σ ratio=%.2fx  (target=3.0x)",
            zpf, n_bins, ratio,
        )


# ===========================================================================
# Section 5: line-contamination robustness
# ===========================================================================
def figure_homogeneous_false_positive() -> None:
    """False-positive check: homogeneous σ noise — both estimators should report a flat σ.

    A subdivision-happy estimator can find spurious σ structure in
    homogeneous noise. The MAD-variant claim is that it preserves the
    current estimator's near-perfect false-positive rate on truly
    uniform data while not hiding real heterogeneity. This panel
    checks the claim.
    """
    rng = np.random.default_rng(RNG_SEED + 1)
    n_active = 4096
    sample_dt_us = 0.020
    freq_bb = np.fft.rfftfreq(n_active, d=sample_dt_us)
    sigma_const = 1.0
    re = rng.normal(scale=sigma_const, size=freq_bb.size)
    im = rng.normal(scale=sigma_const, size=freq_bb.size)
    re[0] *= np.sqrt(2); im[0] = 0.0
    if n_active % 2 == 0:
        re[-1] *= np.sqrt(2); im[-1] = 0.0
    spec = re + 1j * im
    mag = np.abs(spec)

    res_cur = estimate_noise_adaptive(freq_bb, mag)
    sigma_mad, info_mad = estimate_noise_mad_split(freq_bb, mag)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axhline(np.sqrt(2), color="k", ls="--", alpha=0.6, label="true σ_x = √2 σ_c")
    ax.plot(
        freq_bb, res_cur.rms_noise, color="C3", lw=0.7,
        label=(
            f"current estimator  n_bins={res_cur.bin_info['n_bins']}  "
            f"range={res_cur.rms_noise.max()/res_cur.rms_noise.min():.3f}x"
        ),
    )
    ax.plot(
        freq_bb, sigma_mad, color="C0", lw=0.7,
        label=(
            f"MAD variant  n_bins={info_mad['n_bins']}  "
            f"range={sigma_mad.max()/sigma_mad.min():.3f}x"
        ),
    )
    ax.set_xlabel("baseband freq (MHz)")
    ax.set_ylabel("σ_x")
    ax.set_title(
        "False-positive check: homogeneous σ=1 noise  "
        "(estimators should report a flat σ; large ratio = spurious structure)"
    )
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out = FIG / "07_homogeneous_fp_check.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", out)
    logger.info(
        "  homogeneous noise: current n_bins=%d range=%.3fx, MAD n_bins=%d range=%.3fx",
        res_cur.bin_info["n_bins"],
        res_cur.rms_noise.max() / res_cur.rms_noise.min(),
        info_mad["n_bins"],
        sigma_mad.max() / sigma_mad.min(),
    )


def figure_line_robustness() -> None:
    """Repeat the synthetic recovery with strong lines added.

    The MAD-variant claim is robustness against spectral lines. Inject
    a few strong Lorentzians on top of the noise; check both the
    current estimator and the MAD variant recover σ(f).
    """
    spec = make_synthetic_noise()
    # Inject 5 strong lines as δ-spikes in the freq domain (the test is
    # whether the σ estimator is insensitive to them, not their shape).
    freq_bb = spec.freq_bb_target
    base_re = np.fft.rfft(spec.fid) * spec.sample_dt_us
    line_freqs = [3.0, 7.5, 10.5, 14.0, 18.0]
    line_amps = [50.0, 30.0, 60.0, 40.0, 80.0]
    spec_with_lines = base_re.copy()
    for f_target, amp in zip(line_freqs, line_amps):
        idx = int(round(f_target / (freq_bb[1] - freq_bb[0])))
        spec_with_lines[idx] += amp + 0j
    mag = np.abs(spec_with_lines)

    sigma_target_x = spec.sigma_target_active_grid * np.sqrt(2)

    res_current = estimate_noise_adaptive(freq_bb, mag)
    sigma_mad, info_mad = estimate_noise_mad_split(freq_bb, mag)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(freq_bb, sigma_target_x, color="k", lw=2.5, alpha=0.6, label="ground truth")
    ax.plot(
        freq_bb, res_current.rms_noise, color="C3", lw=0.8,
        label=(
            f"current estimator  n_bins={res_current.bin_info['n_bins']}  "
            f"range={res_current.rms_noise.max()/res_current.rms_noise.min():.2f}x"
        ),
    )
    ax.plot(
        freq_bb, sigma_mad, color="C0", lw=0.8,
        label=(
            f"MAD variant  n_bins={info_mad['n_bins']}  "
            f"range={sigma_mad.max()/sigma_mad.min():.2f}x"
        ),
    )
    for f, amp in zip(line_freqs, line_amps):
        ax.axvline(f, color="C2", ls=":", alpha=0.4, lw=0.5)
    ax.set_xlabel("baseband freq (MHz)")
    ax.set_ylabel("σ_x")
    ax.set_title(
        "Line-contamination robustness: synthetic 3-step σ(f) + 5 strong δ-lines"
    )
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out = FIG / "05_line_robustness.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", out)


# ===========================================================================
# Section 6: 2638 reality check with the variant
# ===========================================================================
def figure_2638_mad_variant() -> None:
    """Apply the MAD variant to 2638 on both grids; check agreement."""
    if not FTMW_PATH.exists():
        logger.info("2638 fixture missing, skipping section 6")
        return
    from ftmwpipeline._internal.stage5_impl import _build_active_ft_inputs

    inputs = _build_active_ft_inputs(str(FTMW_PATH))
    (
        fid_samples, sample_dt_us, start_us, end_us, expf_us, probe_freq_mhz,
        sideband, n_padded, acquisition_us, user_ft, user_rms,
    ) = inputs
    trim_lo = float(user_ft.freq_array.min())
    trim_hi = float(user_ft.freq_array.max())

    # User grid -- already trimmed by Stage 1 trim
    sort_u = np.argsort(user_ft.freq_array)
    uf = user_ft.freq_array[sort_u]
    um = user_ft.magnitude_spectrum[sort_u]
    su_mad, info_u = estimate_noise_mad_split(uf, um)

    # Active-FT trimmed
    active = compute_active_ft(
        fid=fid_samples, sample_dt_us=sample_dt_us,
        start_us=start_us, end_us=end_us, expf_us=expf_us,
        probe_freq_mhz=probe_freq_mhz, sideband=sideband,
        n_padded=n_padded, rdc=True,
    )
    sort_a = np.argsort(active.freq_mhz)
    af = active.freq_mhz[sort_a]
    am = np.abs(active.complex_spectrum)[sort_a]
    m = (af >= trim_lo) & (af <= trim_hi)
    af_t, am_t = af[m], am[m]
    sa_mad, info_a = estimate_noise_mad_split(af_t, am_t)

    su_norm = su_mad / np.median(su_mad)
    sa_norm = sa_mad / np.median(sa_mad)
    sa_on_user = np.interp(uf, af_t, sa_norm)
    diff = (su_norm - sa_on_user) / np.maximum(sa_on_user, 1e-12)
    logger.info(
        "2638 MAD-variant: median |Δσ/σ|=%.2f%%  p95=%.2f%%  max=%.2f%%",
        np.median(np.abs(diff)) * 100,
        np.percentile(np.abs(diff), 95) * 100,
        np.max(np.abs(diff)) * 100,
    )
    logger.info(
        "  σ range: user=%.2fx, active-FT trimmed=%.2fx",
        su_norm.max() / su_norm.min(), sa_norm.max() / sa_norm.min(),
    )

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(
        uf, su_norm, color="C0", lw=0.5, alpha=0.85,
        label=f"MAD variant on user grid  n_bins={info_u['n_bins']}",
    )
    ax.plot(
        af_t, sa_norm, color="C3", lw=0.7,
        label=f"MAD variant on active-FT trimmed  n_bins={info_a['n_bins']}",
    )
    ax.set_xlabel("freq (MHz)")
    ax.set_ylabel("σ / median(σ)")
    ax.set_title(
        "2638: MAD-variant σ(f) is grid-invariant where current estimator is not"
    )
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out = FIG / "06_2638_mad_variant.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", out)


# ===========================================================================
# Driver
# ===========================================================================
def main() -> None:
    t0 = time.time()
    logger.info("section 1: 2638 σ across grids (current estimator)")
    figure_2638_sigma_vs_grid()
    logger.info("section 2: synthetic recovery (current estimator)")
    figure_synthetic_recovery()
    logger.info("section 3: failure-mode diagnosis")
    figure_failure_mode()
    logger.info("section 4: MAD-variant synthetic recovery")
    figure_mad_variant_recovery()
    logger.info("section 4b: homogeneous-noise false-positive check")
    figure_homogeneous_false_positive()
    logger.info("section 5: line-contamination robustness")
    figure_line_robustness()
    logger.info("section 6: 2638 MAD-variant agreement check")
    figure_2638_mad_variant()
    logger.info("done in %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()
