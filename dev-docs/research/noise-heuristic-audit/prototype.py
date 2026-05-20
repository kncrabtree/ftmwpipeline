"""
Noise-estimation heuristic audit — reproducibility script.

Regenerates every figure under ``figures/`` and the empirical numbers
cited in ``report.md``. From the repository root, with the project
conda env:

    conda run -n ftmwpipeline-dev python \\
        dev-docs/research/noise-heuristic-audit/prototype.py

Sections:

1. Sample-skewness stability vs bin size N (drives the ABS_MIN_BIN_SIZE
   floor).
2. Rayleigh-RMS estimator stability vs N (drives the
   DEFAULT_SMOOTHING_SAMPLES target).
3. Bin-subdivision criterion calibration (current OR-of-four vs
   alternative F-tests, on synthetic σ-step data).
4. 1%-rank step vs per-point cutoff on the 2638 fixture.
5. Fallback-branch reachability on 2638 (instrumented run).
6. σ(f) on 2638 across smoothing-window choices (the figure that
   motivated retiring the 2×bin-width default).

Inputs: the 2638 fixture at ``scratch/exp_2638.ftmw`` (untracked).
Synthetic sections need no inputs.
"""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps

import ftmwpipeline.api as ftmw
from ftmwpipeline.preprocessing import noise_estimation as ne

HERE = Path(__file__).parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
# dev-docs/research/<topic>/prototype.py → repo root = parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = REPO_ROOT / "scratch" / "exp_2638.ftmw"
RNG = np.random.default_rng(20260520)

RAYLEIGH_SKEW = (2 * np.sqrt(np.pi) * (np.pi - 3)) / (4 - np.pi) ** 1.5  # ≈ 0.6311

print(f"Rayleigh theoretical skewness: {RAYLEIGH_SKEW:.6f}")


# ---------------------------------------------------------------------------
# 1. Sample skewness stability vs N
# ---------------------------------------------------------------------------
def study_skewness_stability(n_trials: int = 5000) -> None:
    print("\n=== 1. Sample-skewness stability vs N ===")
    Ns = (50, 100, 150, 200, 300, 500, 1000, 2000, 5000, 10000)
    rows = []
    for N in Ns:
        skews = []
        for _ in range(n_trials):
            x = RNG.rayleigh(scale=1.0, size=N)
            m = x.mean()
            var = x.var()
            if var <= 0:
                continue
            m3 = ((x - m) ** 3).mean()
            skews.append(m3 / var ** 1.5)
        s = np.asarray(skews)
        rows.append((N, s.mean(), s.std(), np.quantile(s, 0.05), np.quantile(s, 0.95)))
        print(
            f"  N={N:5d}  mean={s.mean():.4f}  std={s.std():.4f}  "
            f"5%={np.quantile(s, 0.05):+.3f}  95%={np.quantile(s, 0.95):+.3f}  "
            f"bias={s.mean() - RAYLEIGH_SKEW:+.4f}"
        )
    arr = np.asarray(rows)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.errorbar(arr[:, 0], arr[:, 1], yerr=arr[:, 2], fmt="o-", label="sample skewness")
    ax.axhline(RAYLEIGH_SKEW, color="k", ls="--", label=f"Rayleigh truth ({RAYLEIGH_SKEW:.3f})")
    ax.axhspan(0.5, 0.7, color="C0", alpha=0.08, label="±0.1 of truth")
    ax.set_xscale("log")
    ax.set_xlabel("bin size N")
    ax.set_ylabel("sample skewness")
    ax.set_title("Rayleigh sample-skewness vs bin size  (mean ± 1σ over 5000 trials)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "01_skewness_stability.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Rayleigh-RMS estimator stability
# ---------------------------------------------------------------------------
def study_rms_stability(n_trials: int = 5000) -> None:
    print("\n=== 2. Rayleigh-RMS estimator stability vs N ===")
    # Theoretical: Var(RMS)/RMS^2 ≈ 1/(4N)  → relative σ ≈ 1/(2√N).
    Ns = (50, 100, 200, 500, 1000, 2000, 5000, 10000, 25000)
    rows = []
    for N in Ns:
        rms_vals = []
        for _ in range(n_trials):
            x = RNG.rayleigh(scale=1.0, size=N)
            rms_vals.append(np.sqrt(np.mean(x * x)))
        r = np.asarray(rms_vals)
        true_rms = np.sqrt(2.0)  # E[X^2] = 2 s^2 with s = scale = 1
        rel_std = r.std() / true_rms
        predicted = 1.0 / (2.0 * np.sqrt(N))
        rows.append((N, r.mean(), rel_std, predicted))
        print(
            f"  N={N:5d}  mean(RMS)={r.mean():.5f}  rel_std={rel_std*100:.3f}%  "
            f"predicted 1/(2√N)={predicted*100:.3f}%  ratio={rel_std/predicted:.3f}"
        )
    arr = np.asarray(rows)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.loglog(arr[:, 0], arr[:, 2] * 100, "o-", label="empirical σ(RMS)/RMS")
    ax.loglog(arr[:, 0], arr[:, 3] * 100, "k--", label="theory 1/(2√N)")
    for tol, label in [(0.01, "1% target"), (0.05, "5% target")]:
        n_req = (1.0 / (2.0 * tol)) ** 2
        ax.axvline(n_req, color="C3", ls=":", alpha=0.5)
        ax.text(n_req * 1.1, 6, f"{label}: N≈{int(n_req)}", fontsize=8, rotation=90, va="top")
    ax.set_xlabel("samples in smoothing window N")
    ax.set_ylabel("relative std of RMS estimator (%)")
    ax.set_title("Rayleigh-RMS stability — predicts smoothing window size")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "02_rms_stability.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Subdivision criterion calibration
# ---------------------------------------------------------------------------
def _current_subdivision_decision(left_mags: np.ndarray, right_mags: np.ndarray) -> bool:
    """Reimplement the OR-of-four criterion from the production code."""
    # Use the same skewness-filtered subsets as the code does.
    li, ls = ne._filter_by_skewness_cached(left_mags, np.arange(len(left_mags)), 0.631, 0.01, (0, 0))
    ri, rs = ne._filter_by_skewness_cached(right_mags, np.arange(len(right_mags)), 0.631, 0.01, (0, 0))
    lm, lv, ln = ls.mean, ls.variance, len(li)
    rm, rv, rn = rs.mean, rs.variance, len(ri)
    mean_diff_pct = abs(lm - rm) / (0.5 * (lm + rm) + 1e-10) * 100
    var_diff_pct = abs(lv - rv) / (0.5 * (lv + rv) + 1e-10) * 100
    l_se = np.sqrt(lv / max(ln, 1))
    r_se = np.sqrt(rv / max(rn, 1))
    z = abs(lm - rm) / (np.hypot(l_se, r_se) + 1e-10)
    f_stat = max(lv, rv) / (min(lv, rv) + 1e-10)
    mean_sig = (mean_diff_pct >= 20) or (z >= 5)
    var_sig = (var_diff_pct >= 20) or (f_stat >= 2.0)
    nf_l = ln / len(left_mags)
    nf_r = rn / len(right_mags)
    return (mean_sig or var_sig) and nf_l >= 2 / 3 and nf_r >= 2 / 3


def _ftest_subdivision_decision(left_mags: np.ndarray, right_mags: np.ndarray, alpha: float = 1e-3) -> bool:
    """Cleaner alternative: F-test on the noise-trimmed variances at calibrated α."""
    li, ls = ne._filter_by_skewness_cached(left_mags, np.arange(len(left_mags)), 0.631, 0.01, (0, 0))
    ri, rs = ne._filter_by_skewness_cached(right_mags, np.arange(len(right_mags)), 0.631, 0.01, (0, 0))
    lv, rv = ls.variance, rs.variance
    ln, rn = ls.nobs, rs.nobs
    if lv <= 0 or rv <= 0 or ln < 2 or rn < 2:
        return False
    if lv >= rv:
        F = lv / rv
        df1, df2 = ln - 1, rn - 1
    else:
        F = rv / lv
        df1, df2 = rn - 1, ln - 1
    p_value = 2 * (1 - sps.f.cdf(F, df1, df2))  # two-sided
    return p_value < alpha


def study_subdivision_calibration(n_trials: int = 200, N: int = 5000) -> None:
    print("\n=== 3. Subdivision criterion calibration ===")
    # Generate Rayleigh magnitudes (the magnitude of complex Gaussian noise).
    # "scale ratio" parametrises σ_right / σ_left; ratio 1.0 is homogeneous.
    ratios = (1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0)
    print(f"  {'σ ratio':>9}  {'curr':>8}  {'F α=1e-3':>10}  {'F α=1e-4':>10}")
    rows = []
    for ratio in ratios:
        cur, ft1, ft2 = 0, 0, 0
        for _ in range(n_trials):
            left = RNG.rayleigh(scale=1.0, size=N)
            right = RNG.rayleigh(scale=ratio, size=N)
            cur += int(_current_subdivision_decision(left, right))
            ft1 += int(_ftest_subdivision_decision(left, right, alpha=1e-3))
            ft2 += int(_ftest_subdivision_decision(left, right, alpha=1e-4))
        rows.append((ratio, cur / n_trials, ft1 / n_trials, ft2 / n_trials))
        print(f"  {ratio:>9.2f}  {cur/n_trials:>7.1%}  {ft1/n_trials:>10.1%}  {ft2/n_trials:>10.1%}")

    arr = np.asarray(rows)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(arr[:, 0], arr[:, 1] * 100, "o-", label="current criterion")
    ax.plot(arr[:, 0], arr[:, 2] * 100, "s--", label="F-test α=1e-3")
    ax.plot(arr[:, 0], arr[:, 3] * 100, "^:", label="F-test α=1e-4")
    ax.axhline(5, color="k", ls=":", alpha=0.5, label="5% false-positive line")
    ax.set_xlabel("σ_right / σ_left  (1.0 = homogeneous)")
    ax.set_ylabel("subdivision rate (%)")
    ax.set_title(f"Subdivision criterion vs noise-σ step  (N={N} per half)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "03_subdivision_calibration.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. 1%-step vs per-point cutoff on 2638
# ---------------------------------------------------------------------------
def _filter_per_point(bin_magnitudes: np.ndarray, bin_indices: np.ndarray, skew_target: float) -> tuple[np.ndarray, ne.BinStats]:
    """Per-point granularity variant: scan every keep count from N down to N/10."""
    n = bin_magnitudes.shape[0]
    if n < 3:
        return bin_indices, ne._bin_stats_from(bin_magnitudes)
    s = np.sort(bin_magnitudes)
    cs1 = np.cumsum(s, dtype=np.float64)
    cs2 = np.cumsum(s * s, dtype=np.float64)
    cs3 = np.cumsum(s * s * s, dtype=np.float64)
    counts = np.arange(1, n + 1, dtype=np.float64)
    means = cs1 / counts
    var = cs2 / counts - means * means
    m3 = cs3 / counts - 3 * means * (cs2 / counts) + 2 * means ** 3
    with np.errstate(divide="ignore", invalid="ignore"):
        skews = np.where(var > 0, m3 / var ** 1.5, np.inf)
    # Largest keep_n with skews[k-1] < target, scanning from top.
    target_floor = max(3, n // 10)
    valid = skews < skew_target
    # Find largest k in [target_floor, n] satisfying valid; if none, fallback.
    valid_window = valid[target_floor - 1 : n]
    if not valid_window.any():
        keep_n = max(1, n // 10)
        threshold = s[keep_n - 1]
        mask = bin_magnitudes <= threshold
        return bin_indices[mask] if mask.any() else bin_indices[:1], ne._bin_stats_from(s[:keep_n])
    # Last True in valid_window:
    rel_k = int(np.where(valid_window)[0].max())
    keep_n = target_floor + rel_k
    threshold = s[keep_n - 1]
    mask = bin_magnitudes <= threshold
    return bin_indices[mask], ne.BinStats(
        nobs=keep_n,
        mean=float(means[keep_n - 1]),
        variance=float(var[keep_n - 1]),
        skewness=float(skews[keep_n - 1]),
    )


def study_step_granularity() -> None:
    print("\n=== 4. 1%-step vs per-point cutoff on 2638 ===")
    ft = ftmw.compute_ft(str(FIXTURE_PATH))
    freqs = np.asarray(ft.freq_array, dtype=float)
    mags = np.abs(np.asarray(ft.complex_spectrum, dtype=np.complex128))
    # Original.
    base = ne.estimate_noise_adaptive(freqs, mags)
    # Per-point: patch the inner function.
    orig = ne._filter_by_skewness_cached

    def patched(bin_magnitudes, bin_indices, skew_target, inc, cache_key):
        return _filter_per_point(bin_magnitudes, bin_indices, skew_target)

    ne._filter_by_skewness_cached = patched  # type: ignore[assignment]
    try:
        per_point = ne.estimate_noise_adaptive(freqs, mags)
    finally:
        ne._filter_by_skewness_cached = orig  # type: ignore[assignment]
    mask_agree = (base.noise_mask == per_point.noise_mask).mean()
    rel = np.abs(base.rms_noise - per_point.rms_noise) / np.maximum(np.abs(base.rms_noise), 1e-12)
    print(f"  1%-step: bins={base.bin_info['n_bins']}, noise_frac={base.bin_info['noise_fraction']:.5f}")
    print(f"  per-pt:  bins={per_point.bin_info['n_bins']}, noise_frac={per_point.bin_info['noise_fraction']:.5f}")
    print(f"  noise_mask agreement: {mask_agree*100:.4f}%")
    print(f"  rms_noise rel-diff: median={np.median(rel):.3e}, p95={np.quantile(rel, 0.95):.3e}, max={rel.max():.3e}")


# ---------------------------------------------------------------------------
# 5. Fallback reachability on 2638
# ---------------------------------------------------------------------------
def study_fallback_reachability() -> None:
    print("\n=== 5. Fallback reachability on 2638 ===")
    ft = ftmw.compute_ft(str(FIXTURE_PATH))
    freqs = np.asarray(ft.freq_array, dtype=float)
    mags = np.abs(np.asarray(ft.complex_spectrum, dtype=np.complex128))
    counts = Counter()
    chosen_cutoffs = []
    orig = ne._filter_by_skewness_cached

    def instrumented(bin_magnitudes, bin_indices, skew_target, inc, cache_key):
        # Re-implement the same logic but record which branch fired.
        n = bin_magnitudes.shape[0]
        if n < 3:
            counts["tiny_bin"] += 1
            return orig(bin_magnitudes, bin_indices, skew_target, inc, cache_key)
        s = np.sort(bin_magnitudes)
        cs1 = np.cumsum(s, dtype=np.float64)
        cs2 = np.cumsum(s * s, dtype=np.float64)
        cs3 = np.cumsum(s * s * s, dtype=np.float64)
        n_steps = int(np.floor(0.9 / inc)) + 1
        for step in range(n_steps):
            cutoff = step * inc
            keep_n = n if cutoff == 0.0 else max(3, n - int(np.ceil(cutoff * n)))
            k = float(keep_n)
            m1 = cs1[keep_n - 1] / k
            var = cs2[keep_n - 1] / k - m1 * m1
            if var <= 0:
                continue
            m3 = cs3[keep_n - 1] / k - 3.0 * m1 * (cs2[keep_n - 1] / k) + 2.0 * m1 ** 3
            skew = m3 / var ** 1.5
            if skew < skew_target:
                counts["converged"] += 1
                chosen_cutoffs.append(cutoff)
                return orig(bin_magnitudes, bin_indices, skew_target, inc, cache_key)
        counts["fallback_10pct"] += 1
        return orig(bin_magnitudes, bin_indices, skew_target, inc, cache_key)

    ne._filter_by_skewness_cached = instrumented  # type: ignore[assignment]
    try:
        ne.estimate_noise_adaptive(freqs, mags)
    finally:
        ne._filter_by_skewness_cached = orig  # type: ignore[assignment]
    print(f"  total calls: {sum(counts.values())}")
    for k, v in counts.items():
        print(f"    {k}: {v}")
    if chosen_cutoffs:
        c = np.asarray(chosen_cutoffs)
        print(f"  chosen cutoffs: median={np.median(c):.3f}, p95={np.quantile(c, 0.95):.3f}, max={c.max():.3f}")
        print(f"  cutoff distribution: {np.bincount((c * 100).astype(int))[:20]}")


# ---------------------------------------------------------------------------
# 6. σ(f) on 2638 across smoothing-window choices
# ---------------------------------------------------------------------------
def study_smoothing_window_sweep() -> None:
    """Show σ(f) on 2638 for a range of smoothing-window widths.

    The 614 MHz entry is hard-coded — it reproduces the pre-audit
    ``2 × avg_bin_width`` default the algorithm used to compute
    automatically. Hard-coding keeps this figure as a historical
    comparison against the post-audit sample-count default, regardless
    of what ``smoothing_window_mhz=None`` resolves to today.
    """
    print("\n=== 6. σ(f) on 2638 vs smoothing window ===")
    ft = ftmw.compute_ft(str(FIXTURE_PATH))
    freqs = np.asarray(ft.freq_array, dtype=float)
    mags = np.abs(np.asarray(ft.complex_spectrum, dtype=np.complex128))
    # 614 MHz reproduces the prior 2×avg_bin_width default on 2638.
    windows = (614, 200, 100, 50, 20)
    results = {}
    for w in windows:
        r = ne.estimate_noise_adaptive(freqs, mags, smoothing_window_mhz=w)
        results[w] = r
        eff = r.bin_info["smoothing_window_mhz"]
        npts = r.bin_info["smoothing_window_points"]
        print(
            f"  win={w:>5} MHz  effective={eff:6.1f} MHz  ({npts:6d} pts)  "
            f"σ med={np.median(r.rms_noise):.5g}  "
            f"range=[{r.rms_noise.min():.4g}, {r.rms_noise.max():.4g}]"
        )

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.semilogy(freqs, mags, lw=0.3, color="0.7", label="|FT|")
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(windows)))
    for c, w in zip(colors, windows):
        r = results[w]
        eff = r.bin_info["smoothing_window_mhz"]
        ax.plot(freqs, r.rms_noise, lw=0.9, color=c, label=f"win={eff:.0f} MHz")
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("|FT|  /  RMS noise")
    ax.set_title("σ(f) on 2638 vs smoothing-window choice")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "04_smoothing_window_sweep.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.perf_counter()
    study_skewness_stability()
    study_rms_stability()
    study_subdivision_calibration()
    study_step_granularity()
    study_fallback_reachability()
    study_smoothing_window_sweep()
    print(f"\nTotal: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
