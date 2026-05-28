"""Probe 1 follow-up: Stage 5 χ²ᵣ comparison across Stage 4 τ variants.

Probe 1 (``probe_leakage_tau.py``) showed Stage 4-internal differences
between the boxcar default and τ-fed plans: τ-fed plans add hard
windows and fixed contributors. Those *might* be improvements (extra
contributors resolving out-of-band-line bleed-through) or *might* be
overhead with no benefit. Only the Stage 5 fit can tell.

This probe runs ``fit_peaks`` on the 6 variant fixtures from probe 1
(3 τ-variants × 2 shape fixtures) and compares per-window χ²ᵣ across
variants. For each shape, the three variants are matched by
``window_id`` (probe 1 confirmed window boundaries don't depend on
``leakage.tau_us``; only contributor attachment changes).

Variants compared per shape:

* **boxcar** -- ``tau_us = None`` (current default; widest reach).
* **tau_maj** -- Stage 2b Lorentzian τ_maj (5.958 µs on 2638).
* **tau_G_maj** -- Stage 2b Gaussian τ_G_maj (6.411 µs on 2638).

Note: the Stage 5 fit shape is set automatically from
``recommended_shape`` on each fixture, so the Lorentzian fixture is
fit with shape=lorentzian and the Gaussian fixture with
shape=gaussian. The Stage 2b τ that drives the fit's prior and the
rescue is the shape-matching twin (per-band routing); the
``leakage.tau_us`` variant only changes which strong lines were
attached to each window as fixed_contributors at Stage 4.

Outputs
-------

* ``data/probe_leakage_tau_p5_per_window.csv`` -- per-window per-variant.
* ``data/probe_leakage_tau_p5_pairs.csv`` -- pairwise (variant_a,
  variant_b, window_id, Δχ²ᵣ, Δn_fixed).
* ``data/probe_leakage_tau_p5_summary.json`` -- aggregate stats per
  shape / pair.
* ``figures/probe_leakage_tau_p5.png`` -- panel figure.

Wall-clock: ~12 min (6× fit_peaks on full 2638).
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import ftmwpipeline.api as ftmw
from ftmwpipeline.io.stage_fit_settings_serialization import (
    write_stage2b_recommended_shape,
)

DATA = HERE / "data"
FIG = HERE / "figures"
DATA.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = REPO_ROOT / "scratch" / "stage4-gaussian-audit" / "runs"
P5_RUNS_DIR = REPO_ROOT / "scratch" / "stage4-gaussian-audit" / "runs_p5"
P5_RUNS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("probe-leakage-tau-p5")

SHAPES = ("lorentzian", "gaussian")
VARIANTS = ("boxcar", "tau_maj", "tau_G_maj")


@dataclass
class WindowRecord:
    shape: str
    variant: str
    window_id: int
    freq_lo: float
    freq_hi: float
    n_fitted_peaks: int
    n_fixed_contributors: int
    reduced_chi2: float
    aic: float
    success: bool


@dataclass
class PairRecord:
    shape: str
    variant_a: str
    variant_b: str
    window_id: int
    freq_lo: float
    freq_hi: float
    chi2r_a: float
    chi2r_b: float
    delta_chi2r: float  # b - a
    n_fixed_a: int
    n_fixed_b: int
    delta_n_fixed: int  # b - a
    aic_a: float
    aic_b: float
    delta_aic: float  # b - a


def _run_fit(shape: str, variant: str) -> List[WindowRecord]:
    src = RUNS_DIR / f"{shape}_leakage.tau_us_{variant}.ftmw"
    if not src.exists():
        raise SystemExit(f"missing source fixture: {src}")
    work = P5_RUNS_DIR / f"{shape}_{variant}.ftmw"
    shutil.copy(src, work)
    write_stage2b_recommended_shape(str(work), shape=shape)

    logger.info("fit_peaks(%s, %s) ...", shape, variant)
    t0 = time.perf_counter()
    fit = ftmw.fit_peaks(str(work))
    dt = time.perf_counter() - t0
    logger.info(
        "  done in %.1f s: %d windows, %d fitted peaks",
        dt, fit.n_windows, fit.n_fitted_peaks,
    )

    rows: List[WindowRecord] = []
    for wf in fit.window_fits:
        win = wf.window
        wid = wf.window_id if wf.window_id is not None else (
            win.window_id if win is not None else -1
        )
        if win is not None:
            fr = win.freq_range
            freq_lo, freq_hi = float(fr[0]), float(fr[1])
        else:
            freq_lo, freq_hi = float("nan"), float("nan")
        # ``fixed_parameters`` holds the frozen-contributor summary the
        # window fit used. The number of distinct fixed contributors is
        # the number of entries; each value is one frozen line's tuple.
        n_fixed = len(wf.fixed_parameters)
        rows.append(WindowRecord(
            shape=shape,
            variant=variant,
            window_id=int(wid),
            freq_lo=freq_lo,
            freq_hi=freq_hi,
            n_fitted_peaks=int(wf.n_peaks_fitted),
            n_fixed_contributors=int(n_fixed),
            reduced_chi2=float(wf.reduced_chi2),
            aic=float(wf.aic),
            success=bool(wf.success),
        ))
    return rows


def _make_pairs(
    by_variant: Dict[str, List[WindowRecord]],
    shape: str,
    variant_a: str,
    variant_b: str,
) -> List[PairRecord]:
    a_map = {r.window_id: r for r in by_variant[variant_a]}
    b_map = {r.window_id: r for r in by_variant[variant_b]}
    shared = sorted(set(a_map) & set(b_map))
    out: List[PairRecord] = []
    for wid in shared:
        a = a_map[wid]
        b = b_map[wid]
        if not (np.isfinite(a.reduced_chi2) and np.isfinite(b.reduced_chi2)):
            continue
        out.append(PairRecord(
            shape=shape,
            variant_a=variant_a,
            variant_b=variant_b,
            window_id=wid,
            freq_lo=a.freq_lo,
            freq_hi=a.freq_hi,
            chi2r_a=a.reduced_chi2,
            chi2r_b=b.reduced_chi2,
            delta_chi2r=b.reduced_chi2 - a.reduced_chi2,
            n_fixed_a=a.n_fixed_contributors,
            n_fixed_b=b.n_fixed_contributors,
            delta_n_fixed=b.n_fixed_contributors - a.n_fixed_contributors,
            aic_a=a.aic,
            aic_b=b.aic,
            delta_aic=b.aic - a.aic,
        ))
    return out


def _summarise_pair(pairs: List[PairRecord]) -> Dict[str, Any]:
    if not pairs:
        return {"n": 0}
    d_chi = np.array([p.delta_chi2r for p in pairs])
    d_nf = np.array([p.delta_n_fixed for p in pairs])
    d_aic = np.array([p.delta_aic for p in pairs])
    cnt_same = int(np.sum(d_nf == 0))
    cnt_more_b = int(np.sum(d_nf > 0))
    cnt_more_a = int(np.sum(d_nf < 0))
    # Conditional Δχ²ᵣ by Δn_fixed sign.
    def _stats(mask):
        sub = d_chi[mask]
        if sub.size == 0:
            return {"n": 0}
        return {
            "n": int(sub.size),
            "median": float(np.median(sub)),
            "mean": float(np.mean(sub)),
            "p25": float(np.percentile(sub, 25)),
            "p75": float(np.percentile(sub, 75)),
            "p95": float(np.percentile(sub, 95)),
            "min": float(sub.min()),
            "max": float(sub.max()),
        }
    return {
        "n_pairs": len(pairs),
        "n_b_lower_chi2r": int(np.sum(d_chi < 0)),
        "n_b_higher_chi2r": int(np.sum(d_chi > 0)),
        "delta_chi2r_median": float(np.median(d_chi)),
        "delta_chi2r_mean": float(np.mean(d_chi)),
        "delta_aic_median": float(np.median(d_aic)),
        "delta_aic_mean": float(np.mean(d_aic)),
        "n_fixed_same": cnt_same,
        "n_fixed_more_b": cnt_more_b,
        "n_fixed_more_a": cnt_more_a,
        "delta_chi2r_by_same_fixed": _stats(d_nf == 0),
        "delta_chi2r_by_more_b_fixed": _stats(d_nf > 0),
        "delta_chi2r_by_more_a_fixed": _stats(d_nf < 0),
    }


def _write_per_window_csv(rows: List[WindowRecord], path: Path) -> None:
    import csv
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "shape", "variant", "window_id", "freq_lo_mhz", "freq_hi_mhz",
            "n_fitted_peaks", "n_fixed_contributors",
            "reduced_chi2", "aic", "success",
        ])
        for r in rows:
            w.writerow([
                r.shape, r.variant, r.window_id,
                f"{r.freq_lo:.4f}", f"{r.freq_hi:.4f}",
                r.n_fitted_peaks, r.n_fixed_contributors,
                f"{r.reduced_chi2:.6f}", f"{r.aic:.4f}", int(r.success),
            ])
    logger.info("Wrote %s", path)


def _write_pairs_csv(pairs: List[PairRecord], path: Path) -> None:
    import csv
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "shape", "variant_a", "variant_b", "window_id",
            "freq_lo_mhz", "freq_hi_mhz",
            "chi2r_a", "chi2r_b", "delta_chi2r",
            "n_fixed_a", "n_fixed_b", "delta_n_fixed",
            "aic_a", "aic_b", "delta_aic",
        ])
        for p in pairs:
            w.writerow([
                p.shape, p.variant_a, p.variant_b, p.window_id,
                f"{p.freq_lo:.4f}", f"{p.freq_hi:.4f}",
                f"{p.chi2r_a:.6f}", f"{p.chi2r_b:.6f}",
                f"{p.delta_chi2r:.6f}",
                p.n_fixed_a, p.n_fixed_b, p.delta_n_fixed,
                f"{p.aic_a:.4f}", f"{p.aic_b:.4f}", f"{p.delta_aic:.4f}",
            ])
    logger.info("Wrote %s", path)


def _plot(
    by_shape_variant: Dict[Tuple[str, str], List[WindowRecord]],
    pairs_by_key: Dict[Tuple[str, str, str], List[PairRecord]],
    out: Path,
) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Top row: per-variant aggregate χ²ᵣ distribution by shape.
    for col, shape in enumerate(SHAPES):
        ax = axes[0, col]
        for variant in VARIANTS:
            rows = by_shape_variant.get((shape, variant), [])
            chi2r = np.array([
                r.reduced_chi2 for r in rows if np.isfinite(r.reduced_chi2)
            ])
            if chi2r.size == 0:
                continue
            ax.hist(
                np.clip(chi2r, 0, 20), bins=40, alpha=0.45,
                label=f"{variant} (med={np.median(chi2r):.2f}, p95={np.percentile(chi2r, 95):.2f})",
            )
        ax.set_xlabel("χ²ᵣ (clipped at 20)")
        ax.set_ylabel("n_windows")
        ax.set_title(f"{shape}: χ²ᵣ distribution by variant")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Bottom-left/middle: pairwise Δχ²ᵣ vs Δn_fixed per shape.
    for col, shape in enumerate(SHAPES):
        ax = axes[1, col]
        for variant_b, colour in (
            ("tau_maj", "tab:blue"),
            ("tau_G_maj", "tab:orange"),
        ):
            key = (shape, "boxcar", variant_b)
            pairs = pairs_by_key.get(key, [])
            if not pairs:
                continue
            d_chi = np.array([p.delta_chi2r for p in pairs])
            d_nf = np.array([p.delta_n_fixed for p in pairs])
            # Add jitter to d_nf for visibility.
            jitter = (np.random.default_rng(0).random(d_nf.size) - 0.5) * 0.3
            ax.scatter(
                d_nf + jitter, np.clip(d_chi, -10, 10),
                s=14, alpha=0.4, color=colour,
                label=f"boxcar → {variant_b} (n={len(pairs)})",
            )
        ax.axhline(0, color="k", lw=0.7, linestyle="--")
        ax.axvline(0, color="k", lw=0.5, linestyle=":")
        ax.set_xlabel("Δn_fixed_contributors (variant − boxcar)")
        ax.set_ylabel("Δχ²ᵣ (clipped ±10)")
        ax.set_title(f"{shape}: Δχ²ᵣ vs Δcontributors")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Top-right + bottom-right: per-shape Δχ²ᵣ histograms split by
    # contributor-set delta sign.
    ax_top = axes[0, 2]
    ax_bot = axes[1, 2]
    for col_ax, shape in zip((ax_top, ax_bot), SHAPES):
        for variant_b, colour in (
            ("tau_maj", "tab:blue"),
            ("tau_G_maj", "tab:orange"),
        ):
            key = (shape, "boxcar", variant_b)
            pairs = pairs_by_key.get(key, [])
            if not pairs:
                continue
            d_chi_more = np.array([
                p.delta_chi2r for p in pairs if p.delta_n_fixed > 0
            ])
            if d_chi_more.size:
                col_ax.hist(
                    np.clip(d_chi_more, -5, 5), bins=40, alpha=0.45,
                    color=colour,
                    label=(
                        f"{shape}: boxcar→{variant_b} "
                        f"(Δfc>0, n={d_chi_more.size}, "
                        f"med Δχ²ᵣ={np.median(d_chi_more):+.3f})"
                    ),
                )
        col_ax.axvline(0, color="k", lw=0.7, linestyle="--")
        col_ax.set_xlabel("Δχ²ᵣ on windows that gained a contributor")
        col_ax.set_ylabel("n_windows")
        col_ax.set_title(f"{shape}: Δχ²ᵣ where variant added a contributor")
        col_ax.legend(fontsize=7, loc="upper left")
        col_ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Stage 4 leakage.tau_us p5 follow-up: per-window χ²ᵣ across τ variants",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    logger.info("Wrote %s", out)
    return out


def main():
    by_shape_variant: Dict[Tuple[str, str], List[WindowRecord]] = {}
    all_rows: List[WindowRecord] = []
    for shape in SHAPES:
        for variant in VARIANTS:
            rows = _run_fit(shape, variant)
            by_shape_variant[(shape, variant)] = rows
            all_rows.extend(rows)

    _write_per_window_csv(all_rows, DATA / "probe_leakage_tau_p5_per_window.csv")

    pairs_by_key: Dict[Tuple[str, str, str], List[PairRecord]] = {}
    all_pairs: List[PairRecord] = []
    for shape in SHAPES:
        by_variant = {
            v: by_shape_variant[(shape, v)] for v in VARIANTS
        }
        for (a, b) in (
            ("boxcar", "tau_maj"),
            ("boxcar", "tau_G_maj"),
            ("tau_maj", "tau_G_maj"),
        ):
            pairs = _make_pairs(by_variant, shape, a, b)
            pairs_by_key[(shape, a, b)] = pairs
            all_pairs.extend(pairs)

    _write_pairs_csv(all_pairs, DATA / "probe_leakage_tau_p5_pairs.csv")

    summary = {"per_shape": {}, "pairs": {}}
    for shape in SHAPES:
        summary["per_shape"][shape] = {}
        for variant in VARIANTS:
            rows = by_shape_variant[(shape, variant)]
            chi2r = np.array([
                r.reduced_chi2 for r in rows if np.isfinite(r.reduced_chi2)
            ])
            nf = np.array([r.n_fixed_contributors for r in rows])
            n_peaks = np.array([r.n_fitted_peaks for r in rows])
            summary["per_shape"][shape][variant] = {
                "n_windows": int(len(rows)),
                "n_finite": int(chi2r.size),
                "chi2r_median": float(np.median(chi2r)) if chi2r.size else float("nan"),
                "chi2r_p95": float(np.percentile(chi2r, 95)) if chi2r.size else float("nan"),
                "chi2r_max": float(chi2r.max()) if chi2r.size else float("nan"),
                "n_chi2r_gt_5": int(np.sum(chi2r > 5.0)),
                "n_chi2r_gt_10": int(np.sum(chi2r > 10.0)),
                "n_fitted_peaks_total": int(n_peaks.sum()),
                "n_fixed_total": int(nf.sum()),
            }
    for key, pairs in pairs_by_key.items():
        summary["pairs"]["__".join(key)] = _summarise_pair(pairs)

    (DATA / "probe_leakage_tau_p5_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    logger.info("Wrote %s", DATA / "probe_leakage_tau_p5_summary.json")

    _plot(by_shape_variant, pairs_by_key, FIG / "probe_leakage_tau_p5.png")

    print()
    print("====== Stage 4 p5 follow-up: per-variant aggregate ======")
    for shape in SHAPES:
        for variant in VARIANTS:
            s = summary["per_shape"][shape][variant]
            print(
                f"  {shape:>10s} {variant:>10s}: "
                f"chi2r med={s['chi2r_median']:.3f} "
                f"p95={s['chi2r_p95']:.3f} "
                f"max={s['chi2r_max']:.2f} "
                f"n>5={s['n_chi2r_gt_5']:>3d} "
                f"n>10={s['n_chi2r_gt_10']:>3d} "
                f"fc_total={s['n_fixed_total']:>4d} "
                f"peaks={s['n_fitted_peaks_total']:>4d}"
            )
    print()
    print("====== Pairwise: median Δχ²ᵣ (b - a) by Δn_fixed_contributors ======")
    for key, pairs in pairs_by_key.items():
        shape, a, b = key
        s = _summarise_pair(pairs)
        print(f"  {shape} {a} -> {b}: n={s.get('n_pairs', 0)}")
        for tag in (
            "delta_chi2r_by_same_fixed",
            "delta_chi2r_by_more_b_fixed",
            "delta_chi2r_by_more_a_fixed",
        ):
            sub = s.get(tag, {})
            label = tag.replace("delta_chi2r_by_", "").replace("_", " ")
            n = sub.get("n", 0)
            if n == 0:
                continue
            print(
                f"    {label:>18s}: n={n:>4d}  "
                f"med Δχ²ᵣ={sub['median']:+.4f}  "
                f"mean={sub['mean']:+.4f}  "
                f"p95={sub['p95']:+.4f}"
            )


if __name__ == "__main__":
    main()
