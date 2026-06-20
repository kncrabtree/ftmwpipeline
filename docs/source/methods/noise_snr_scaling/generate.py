"""Regenerate the figures and key results for the high-SNR noise note.

This harness reproduces, from the checked-in Blackchirp fixtures, the
quantitative claims in :doc:`noise_snr_scaling`:

* the per-bin **scatter** noise estimate
  (:func:`ftmwpipeline.preprocessing.noise_estimation.estimate_noise_scatter`)
  averages down as ``1/sqrt(N)`` with shot count, while a naive level estimate
  (a running median of ``|X|``) flattens at high signal-to-noise because it
  measures the leakage pedestal, not the noise;
* across the fixtures the naive level estimate over-reports the noise by a
  factor that grows with signal-to-noise (up to ~6x), while the scatter
  estimate stays close to the line-free frame-difference reference;
* the analytic Rician correction table ``C(R)`` the estimator ships is
  reproducible from a fixed-seed Monte-Carlo simulation.

Run as a script to (re)write ``figures/*.png`` and ``results.json`` beside this
file::

    python generate.py                 # all fixtures (a few seconds)
    python generate.py --fixtures 2638  # one fixture

The committed ``results.json`` is the regression target for
``tests/integration/test_noise_snr_report.py`` (the ``slow`` marker); the
figures are embedded by the note. ``build_cr_table`` and
``synthetic_sqrtn_slopes`` back the fast, data-free unit tests in
``tests/unit/preprocessing/test_noise_snr_invariants.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import median_filter

from ftmwpipeline.fitting.active_ft import compute_active_ft
from ftmwpipeline.io.data_loaders import load_fid
from ftmwpipeline.preprocessing.noise_estimation import estimate_noise_scatter

# --------------------------------------------------------------------------
# Fixtures (all from one spectrometer; see dev-docs/fixtures/README.md). The
# recommended active-region start per fixture; the active region runs to the end
# of the 15 us record and the analysis band is the canonical 26500-40000 MHz.
# --------------------------------------------------------------------------
TRIM = (26500.0, 40000.0)
FIXTURES: Dict[str, float] = {
    "2638": 2.35,
    "1019": 4.35,
    "655": 3.35,
    "360": 4.35,
    "363": 4.35,
    "1512": 3.35,
    "1231": 4.35,
}
EXAMPLES = Path("examples/blackchirp_data")

# Naive level estimator: a broad running median of |X|, converted to the
# complex-RMS sigma_x via the Rayleigh median factor (median|X| = sigma_c*sqrt(ln4)).
_RAYLEIGH_MEDIAN_TO_SIGMA_X = 1.0 / np.sqrt(np.log(2.0))


def _frame_shots(fixture: str) -> List[Tuple[int, int]]:
    """Return ``[(frame_index, shot_count), ...]`` from ``fid/fidparams.csv``."""
    import csv

    out: List[Tuple[int, int]] = []
    with open(EXAMPLES / fixture / "fid" / "fidparams.csv") as fh:
        for row in csv.DictReader(fh, delimiter=";"):
            out.append((int(row["index"]), int(row["shots"])))
    return out


def active_magnitude(fixture: str, frame: int) -> Tuple[np.ndarray, np.ndarray]:
    """Active-FT (band-trimmed) frequency grid and magnitude for one frame."""
    fid = load_fid(str(EXAMPLES / fixture), format_name="blackchirp", fid_index=frame)
    aft = compute_active_ft(
        np.asarray(fid.data, dtype=float),
        fid.spacing * 1e6,
        start_us=FIXTURES[fixture],
        end_us=fid.duration_us,
        probe_freq_mhz=fid.probe_freq_mhz,
        sideband=fid.sideband,
        n_padded=len(fid.data),
    )
    order = np.argsort(aft.freq_mhz)
    freq = aft.freq_mhz[order]
    mag = np.abs(aft.complex_spectrum[order])
    keep = (freq >= TRIM[0]) & (freq <= TRIM[1])
    return freq[keep], mag[keep]


def naive_level_sigma(
    freq: np.ndarray, mag: np.ndarray, width_mhz: float = 300.0
) -> np.ndarray:
    """Per-bin sigma_x from a running median of |X| (the naive level estimate)."""
    df = abs(float(np.mean(np.diff(freq))))
    width = max(3, int(round(width_mhz / df)) | 1)
    med = median_filter(mag, size=width, mode="nearest")
    return np.asarray(med * _RAYLEIGH_MEDIAN_TO_SIGMA_X, dtype=float)  # type: ignore[no-any-return]


def scatter_sigma(freq: np.ndarray, mag: np.ndarray) -> np.ndarray:
    """Per-bin sigma_x from the production scatter estimator."""
    sigma = estimate_noise_scatter(freq, mag).rms_noise
    return np.asarray(sigma, dtype=float)  # type: ignore[no-any-return]


def _slope(n: np.ndarray, sigma: np.ndarray) -> float:
    """Log-log slope of sigma vs shot count (−0.5 for pure 1/sqrt(N) noise)."""
    order = np.argsort(n)
    return float(np.polyfit(np.log(n[order]), np.log(sigma[order]), 1)[0])


def per_fixture_scaling(fixture: str) -> Dict[str, Any]:
    """Per-frame scaling: shot count, peak SNR, the two sigmas, and overestimate.

    Recording the peak SNR and the naive/scatter overestimate *per backup frame*
    lets the note plot the overestimate against SNR *within one fixture* (line
    content held fixed), which isolates the signal-to-noise dependence from the
    spectrum's line density.
    """
    frames = _frame_shots(fixture)
    n: List[int] = []
    scat: List[float] = []
    naive: List[float] = []
    snr: List[float] = []
    over: List[float] = []
    for idx, shots in frames:
        freq, mag = active_magnitude(fixture, idx)
        sc = float(np.median(scatter_sigma(freq, mag)))
        nv = float(np.median(naive_level_sigma(freq, mag)))
        n.append(shots)
        scat.append(sc)
        naive.append(nv)
        snr.append(float(mag.max() / sc))
        over.append(nv / sc)
    n_a = np.array(n, dtype=float)
    primary = int(np.argmax(n_a))  # the full cumulative frame (max shots)
    out: Dict[str, Any] = {
        "n_shots": n,
        "sigma_scatter": scat,
        "sigma_naive": naive,
        "snr_per_frame": snr,
        "overestimate_per_frame": over,
        "max_shots": int(n_a[primary]),
        # Peak SNR and naive/scatter overestimate at the primary (max-N) frame.
        "snr": snr[primary],
        "overestimate_naive_over_scatter": over[primary],
    }
    if len(frames) >= 2:
        out["slope_scatter"] = _slope(n_a, np.array(scat))
        out["slope_naive"] = _slope(n_a, np.array(naive))
    return out


def floor_decomposition(scaling_655: Dict[str, Any]) -> Dict[str, float]:
    """Fit ``scatter(N)^2 = c/N + f^2`` for 655 (the additive weak-line floor).

    A positive intercept ``f`` is the constant contamination floor (weak lines +
    pedestal residual) that flattens the slope above the pure −0.5 of ``c/N``.
    """
    n = np.array(scaling_655["n_shots"], dtype=float)
    s = np.array(scaling_655["sigma_scatter"], dtype=float)
    x = 1.0 / n
    a, b = np.polyfit(x, s**2, 1)  # s^2 = a*(1/N) + b
    f = float(np.sqrt(max(b, 0.0)))
    return {"c": float(a), "f2": float(b), "floor_f": f}


def _pav_increasing(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: the least-squares non-decreasing fit of ``y``."""
    blocks: List[Tuple[float, float]] = []  # (value, weight)
    for v in np.asarray(y, dtype=float):
        cur_v, cur_w = float(v), 1.0
        while blocks and blocks[-1][0] > cur_v:
            pv, pw = blocks.pop()
            cur_v = (pv * pw + cur_v * cur_w) / (pw + cur_w)
            cur_w = pw + cur_w
        blocks.append((cur_v, cur_w))
    out: List[float] = []
    for value, weight in blocks:
        out.extend([value] * int(round(weight)))
    return np.array(out[: len(y)])


def raw_cr_samples(
    m: int = 1_000_000, seed: int = 0, n_theta: int = 60
) -> Tuple[np.ndarray, np.ndarray]:
    """Raw Monte-Carlo ``(R, C)`` samples of the Rician magnitude (for the figure).

    ``theta = pedestal / sigma_c`` sweeps the regime; ``R = scatter/pedestal`` and
    ``C = sigma_c/scatter`` (``sigma_c = 1``). The cloud is noisy near the
    Rayleigh limit (small ``theta``) because ``R`` saturates there, which is why
    the baked table is the monotone fit below, not these points.
    """
    rng = np.random.default_rng(seed)
    thetas = np.concatenate(
        [np.linspace(0.0, 3.0, n_theta), np.linspace(3.2, 25.0, n_theta)]
    )
    r: List[float] = []
    c: List[float] = []
    for th in thetas:
        mag = np.sqrt((th + rng.standard_normal(m)) ** 2 + rng.standard_normal(m) ** 2)
        scatter = 1.4826 * np.median(np.abs(mag - np.median(mag)))
        r.append(scatter / float(np.median(mag)))
        c.append(1.0 / scatter)  # sigma_c = 1 by construction
    return np.array(r), np.array(c)


def build_cr_table(
    m: int = 1_000_000, seed: int = 0, n_out: int = 80
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the monotone Rician correction ``C(R) = sigma_c / scatter`` lookup.

    Deterministic given the seed; this is the exact procedure that produced the
    ``_SCATTER_R_TAB`` / ``_SCATTER_C_TAB`` constants baked into
    :mod:`ftmwpipeline.preprocessing.noise_estimation`. ``R = scatter / pedestal``
    indexes the local pedestal/noise regime. The true ``C(R)`` is smooth and
    monotone increasing (Rician theory), so the raw Monte-Carlo cloud
    (:func:`raw_cr_samples`) is reduced to its isotonic (monotone) fit and
    resampled onto a clean ascending ``R`` grid for use as an ``np.interp``
    lookup.
    """
    r, c = raw_cr_samples(m, seed)
    order = np.argsort(r)
    r_sorted = r[order]
    c_mono = _pav_increasing(c[order])
    # Collapse near-degenerate R (the saturated Rayleigh end) and resample onto a
    # strictly ascending grid so np.interp is well-posed.
    r_unique, first = np.unique(np.round(r_sorted, 6), return_index=True)
    c_unique = np.maximum.accumulate(c_mono[first])
    grid = np.linspace(float(r_sorted.min()), float(r_sorted.max()), n_out)
    return grid, np.interp(grid, r_unique, c_unique)


def synthetic_sqrtn_slopes(seed: int = 0) -> Dict[str, float]:
    """Slopes on synthetic spectra: a 1/sqrt(N) noise on a constant pedestal.

    The decisive, data-free invariant: the scatter estimator must follow the
    noise (slope ≈ −0.5) while a naive level estimate flattens (slope ≈ 0)
    because it tracks the constant pedestal.
    """
    rng = np.random.default_rng(seed)
    n_bins = 6000
    freq = np.linspace(26500.0, 40000.0, n_bins)
    # A constant leakage-pedestal stand-in that dominates the (1/sqrt(N))
    # noise across the whole shot-count range, so a naive level estimate
    # tracks the constant pedestal (slope ~0) while the scatter estimator
    # follows the noise (slope ~-0.5).
    pedestal = 5.0
    shots = np.array([2.0e4, 1.0e5, 5.0e5, 2.5e6, 1.0e7])
    sigma0 = 1.0
    scat: List[float] = []
    naive: List[float] = []
    for nsh in shots:
        sigma = sigma0 / np.sqrt(nsh / shots[0])  # 1/sqrt(N)
        re = rng.normal(0.0, sigma, n_bins)
        im = rng.normal(0.0, sigma, n_bins)
        mag = np.sqrt((pedestal + re) ** 2 + im**2)
        scat.append(float(np.median(scatter_sigma(freq, mag))))
        naive.append(float(np.median(naive_level_sigma(freq, mag))))
    return {
        "slope_scatter": _slope(shots, np.array(scat)),
        "slope_naive": _slope(shots, np.array(naive)),
    }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _figdir() -> Path:
    d = Path(__file__).resolve().parent / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _loglog_fit_line(ax: Any, n: np.ndarray, y: np.ndarray, color: str) -> None:
    """Overlay the power-law (log-log linear) fit of ``y`` vs ``n`` as a line."""
    order = np.argsort(n)
    slope, intercept = np.polyfit(np.log(n[order]), np.log(y[order]), 1)
    grid = np.array([n[order][0], n[order][-1]], dtype=float)
    ax.plot(grid, np.exp(intercept) * grid**slope, "-", color=color, lw=1.5, zorder=2)


def make_figures(results: Dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ftmwpipeline.visualization.report_style import (
        AGGIE_BLUE,
        BRAND_CYCLE,
        POPPY,
        apply_bare_style,
        apply_color_cycle,
    )

    apply_color_cycle(matplotlib)
    figdir = _figdir()
    per = results["per_fixture"]
    multi = {k: v for k, v in per.items() if "slope_scatter" in v}
    # House style: no titles (the figure caption labels each plot); the brand
    # categorical cycle and the bare spine-free look match the pipeline reports.

    # fig0: 655 shot-count scaling -- data as points, the power-law fit as a line.
    fig, ax = plt.subplots(figsize=(7, 5))
    n655 = np.array(per["655"]["n_shots"], float)
    sc655 = np.array(per["655"]["sigma_scatter"], float)
    nv655 = np.array(per["655"]["sigma_naive"], float)
    c_scatter, c_naive = BRAND_CYCLE[0], BRAND_CYCLE[1]
    ax.scatter(n655, sc655, color=c_scatter, zorder=3, label="scatter")
    _loglog_fit_line(ax, n655, sc655, c_scatter)
    ax.scatter(n655, nv655, color=c_naive, marker="s", zorder=3, label="naive level")
    _loglog_fit_line(ax, n655, nv655, c_naive)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("shot count N")
    ax.set_ylabel(r"median $\sigma$ (active FT)")
    apply_bare_style(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "fig0_sqrtN_655.png", dpi=130)
    plt.close(fig)

    # fig1: scatter scaling across fixtures -- points + per-fixture power-law fit.
    fig, ax = plt.subplots(figsize=(7, 5))
    for (fx, d), color in zip(multi.items(), BRAND_CYCLE):
        nn = np.array(d["n_shots"], float)
        ss = np.array(d["sigma_scatter"], float)
        ax.scatter(
            nn / nn.max(),
            ss / ss.max(),
            color=color,
            zorder=3,
            label=f"{fx} ({d['slope_scatter']:.2f})",
        )
        _loglog_fit_line(ax, nn / nn.max(), ss / ss.max(), color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("relative shot count N / N_max")
    ax.set_ylabel(r"relative scatter $\sigma$")
    apply_bare_style(ax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "fig1_sqrtN_all.png", dpi=130)
    plt.close(fig)

    # fig2: within-fixture overestimate vs SNR. Each multi-frame fixture traces
    # its own backup-frame series, so line content is fixed along a curve and the
    # rise is purely the signal-to-noise effect (not line density).
    fig, ax = plt.subplots(figsize=(7, 5))
    for (fx, d), color in zip(multi.items(), BRAND_CYCLE):
        s = np.array(d["snr_per_frame"], float)
        o = np.array(d["overestimate_per_frame"], float)
        order = np.argsort(s)
        ax.loglog(s[order], o[order], "o-", ms=4, color=color, label=fx)
    ax.axhline(1.0, color="#999999", lw=0.8, ls=":")  # brand neutral gray
    ax.set_xlabel("peak signal-to-noise (per backup frame)")
    ax.set_ylabel("naive level / scatter")
    apply_bare_style(ax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "fig2_overestimate_vs_snr.png", dpi=130)
    plt.close(fig)

    # fig3: floor decomposition on 655 -- points + the c/N + f^2 fit line.
    fl = results["floor_655"]
    fig, ax = plt.subplots(figsize=(7, 5))
    x = 1.0 / n655
    ax.scatter(x, sc655**2, color=AGGIE_BLUE, zorder=3, label="scatter²")
    xs = np.linspace(0, x.max(), 100)
    ax.plot(
        xs,
        fl["c"] * xs + fl["f2"],
        "-",
        color=BRAND_CYCLE[1],
        lw=1.5,
        label=f"c/N + f²  (f={fl['floor_f']:.4g})",
    )
    ax.set_xlabel("1 / N")
    ax.set_ylabel(r"scatter$^2$")
    apply_bare_style(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "fig3_floor_decomp.png", dpi=130)
    plt.close(fig)

    # fig5: the Rician correction C(R) -- raw MC cloud and the monotone fit.
    r_raw, c_raw = raw_cr_samples()
    r_tab, c_tab = build_cr_table()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        r_raw,
        c_raw,
        s=30,
        color=POPPY,
        alpha=0.7,
        zorder=3,
        label="raw Monte-Carlo samples",
    )
    ax.plot(
        r_tab,
        c_tab,
        "-",
        color=AGGIE_BLUE,
        lw=2,
        zorder=4,
        label="monotone (isotonic) fit — baked",
    )
    ax.set_xlabel("R = scatter / pedestal")
    ax.set_ylabel(r"C(R) = $\sigma_c$ / scatter")
    apply_bare_style(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "fig5_region_aware.png", dpi=130)
    plt.close(fig)


def run(fixtures: Optional[List[str]] = None, figures: bool = True) -> Dict[str, Any]:
    names = fixtures or list(FIXTURES)
    per: Dict[str, Any] = {}
    for fx in names:
        per[fx] = per_fixture_scaling(fx)
        d = per[fx]
        msg = f"{fx}: SNR={d['snr']:.3g}  naive/scatter={d['overestimate_naive_over_scatter']:.2f}"
        if "slope_scatter" in d:
            msg += f"  slope_scatter={d['slope_scatter']:.3f}  slope_naive={d['slope_naive']:.3f}"
        print(msg, flush=True)
    results: Dict[str, Any] = {"per_fixture": per}
    if "655" in per:
        results["floor_655"] = floor_decomposition(per["655"])
    r_tab, c_tab = build_cr_table()
    results["cr_table"] = {
        "r_max": float(r_tab.max()),
        "c_max": float(c_tab.max()),
        "n_points": int(r_tab.size),
    }
    results["synthetic_sqrtn"] = synthetic_sqrtn_slopes()
    if figures and fixtures is None:
        make_figures(results)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", nargs="*", default=None, help="subset of fixtures")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument(
        "--out", default=str(Path(__file__).resolve().parent / "results.json")
    )
    args = ap.parse_args()
    results = run(fixtures=args.fixtures, figures=not args.no_figures)
    if args.fixtures is None:
        Path(args.out).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
