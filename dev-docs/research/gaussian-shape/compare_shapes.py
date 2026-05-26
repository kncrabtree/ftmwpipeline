"""Compare Stage 5 ``fit_peaks(shape='lorentzian')`` vs ``shape='gaussian'``.

Drives the Stage 5 fit twice on the same ``.ftmw`` fixture -- once per
shape -- and emits a per-window comparison: χ²ᵣ, AICc, fitted peak counts,
per-peak frequency / amplitude scatter on shared peaks.

Run from the repository root::

    conda run -n ftmwpipeline-dev python \
        dev-docs/research/gaussian-shape/compare_shapes.py \
        scratch/stage5-tau-calibration-lsq/exp_2638_unapodized.ftmw

Outputs (under ``dev-docs/research/gaussian-shape/data/`` and ``figures/``):

* ``data/<stem>_summary.json``      -- per-window comparison records +
                                       aggregate stats (median χ²ᵣ,
                                       AICc, peak-count deltas, peak
                                       scatter on shared lines)
* ``data/<stem>_per_window.csv``    -- per-window record (one row per
                                       window per shape)
* ``data/<stem>_per_peak.csv``      -- per-peak matched pairs (Lorentzian
                                       peak ↔ nearest Gaussian peak in
                                       the same window)
* ``figures/<stem>_panel.png``      -- 2-panel comparison figure
                                       (per-window χ²ᵣ scatter +
                                       per-peak frequency/amplitude
                                       scatter on shared lines)

The fixture must have Stages 0-4 completed. The script runs the Stage 2b
Gaussian-twin calibration (``calibrate_tau_G``) on a working copy if it
isn't already present, then runs both fits on independent working copies
so the persisted ``stage5_fitting`` artifacts don't trample each other.

Acceptance bar (from ``dev-docs/planning/stage5-gaussian-shape.md``):

* Median Gaussian χ²ᵣ < median Lorentzian χ²ᵣ.
* 95th percentile Gaussian χ²ᵣ < 5 (vs ~ 50+ tail on Lorentzian).
* Peak frequencies agree on shared lines to within the per-peak frequency
  error bar (no shape-induced centre bias).
* AICc on the Part A shape-error windows (w141, w213, w310, w355) prefers
  Gaussian by Δ > 5.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.stage2b_g_impl import tau_G_calibration_present

HERE = Path(__file__).parent
FIG = HERE / "figures"
DATA = HERE / "data"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

# Windows Part A identified as shape-error-bearing on 2638. The Gaussian
# fit should win AICc by Δ > 5 on each.
PART_A_SHAPE_ERROR_WINDOWS: tuple[int, ...] = (141, 213, 310, 355)

# Window-pair AICc constant: k=2 (Gaussian / Lorentzian both add one
# decay constant per window). For two fits with the same parameter
# count and same n_eff, AICc difference is just n_eff·log(χ²_a / χ²_b).
# We use the persisted per-window χ²ᵣ scalar (post-rescue, post-thaw) as
# the comparison anchor.

logger = logging.getLogger("gaussian-shape-compare")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


@dataclass
class WindowRecord:
    window_id: int
    n_peaks_lorentz: int
    n_peaks_gauss: int
    chi2r_lorentz: float
    chi2r_gauss: float
    aic_lorentz: float
    aic_gauss: float
    freq_lo_mhz: float
    freq_hi_mhz: float


@dataclass
class MatchedPeak:
    window_id: int
    freq_lorentz_mhz: float
    freq_gauss_mhz: float
    amp_lorentz: float
    amp_gauss: float
    snr_lorentz: float
    snr_gauss: float
    freq_err_lorentz: Optional[float]
    freq_err_gauss: Optional[float]


def _ensure_tau_G_calibration(fp: Path) -> None:
    """Run calibrate_tau_G on ``fp`` unless already present."""
    if tau_G_calibration_present(str(fp)):
        logger.info("%s: τ_G calibration already present", fp.name)
        return
    logger.info("%s: running calibrate_tau_G ...", fp.name)
    tc = ftmw.calibrate_tau_G(str(fp))
    logger.info(
        "%s: τ_G_maj=%.3f us, σ_τ_G=%.3f us, n_eligible=%d (preconditions=%s)",
        fp.name, tc.tau_maj_us, tc.sigma_tau_us, tc.n_contributors,
        "pass" if tc.preconditions_passed else "fail",
    )


def _run_fit(fp: Path, *, shape: str) -> None:
    logger.info("%s: running fit_peaks(shape=%s) ...", fp.name, shape)
    fit = ftmw.fit_peaks(str(fp), shape=shape)
    logger.info(
        "%s: shape=%s -> %d windows, %d fitted peaks (final plan revision %d)",
        fp.name, shape, fit.n_windows, fit.n_fitted_peaks,
        fit.final_plan_revision,
    )


def _match_peaks(
    peaks_l, peaks_g, *, window_id: int, tol_mhz: float = 0.5,
) -> list[MatchedPeak]:
    """Nearest-neighbour match between two peak lists in one window.

    Greedy: walk peaks_l in ascending frequency, pop the closest peaks_g
    within ``tol_mhz``. Unmatched peaks are dropped (peak-count deltas
    handled separately at the window-aggregate level).
    """
    gauss_sorted = sorted(peaks_g, key=lambda p: p.frequency_mhz)
    used = [False] * len(gauss_sorted)
    matches: list[MatchedPeak] = []
    for pl in sorted(peaks_l, key=lambda p: p.frequency_mhz):
        best_idx = -1
        best_d = tol_mhz
        for i, pg in enumerate(gauss_sorted):
            if used[i]:
                continue
            d = abs(pg.frequency_mhz - pl.frequency_mhz)
            if d <= best_d:
                best_d = d
                best_idx = i
        if best_idx >= 0:
            pg = gauss_sorted[best_idx]
            used[best_idx] = True
            matches.append(
                MatchedPeak(
                    window_id=window_id,
                    freq_lorentz_mhz=float(pl.frequency_mhz),
                    freq_gauss_mhz=float(pg.frequency_mhz),
                    amp_lorentz=float(pl.amplitude),
                    amp_gauss=float(pg.amplitude),
                    snr_lorentz=float(pl.snr) if pl.snr is not None else float("nan"),
                    snr_gauss=float(pg.snr) if pg.snr is not None else float("nan"),
                    freq_err_lorentz=(
                        float(pl.frequency_error)
                        if pl.frequency_error is not None
                        else None
                    ),
                    freq_err_gauss=(
                        float(pg.frequency_error)
                        if pg.frequency_error is not None
                        else None
                    ),
                )
            )
    return matches


def _aggregate(
    rows: list[WindowRecord], peaks: list[MatchedPeak],
) -> dict:
    if not rows:
        return {"n_windows": 0}
    chi2r_l = np.array([r.chi2r_lorentz for r in rows], dtype=float)
    chi2r_g = np.array([r.chi2r_gauss for r in rows], dtype=float)
    aic_l = np.array([r.aic_lorentz for r in rows], dtype=float)
    aic_g = np.array([r.aic_gauss for r in rows], dtype=float)
    n_l = np.array([r.n_peaks_lorentz for r in rows])
    n_g = np.array([r.n_peaks_gauss for r in rows])

    finite_l = np.isfinite(chi2r_l)
    finite_g = np.isfinite(chi2r_g)
    finite_both = finite_l & finite_g
    # Restrict per-window aggregates to windows where both fits returned
    # finite χ²ᵣ; the rare degenerate-window case is reported separately.
    chi2r_l_f = chi2r_l[finite_both]
    chi2r_g_f = chi2r_g[finite_both]
    aic_l_f = aic_l[finite_both]
    aic_g_f = aic_g[finite_both]

    # Per-peak shape consistency.
    if peaks:
        f_l = np.array([p.freq_lorentz_mhz for p in peaks], dtype=float)
        f_g = np.array([p.freq_gauss_mhz for p in peaks], dtype=float)
        a_l = np.array([p.amp_lorentz for p in peaks], dtype=float)
        a_g = np.array([p.amp_gauss for p in peaks], dtype=float)
        freq_residual_mhz = f_g - f_l
        amp_ratio = np.where(a_l != 0, a_g / a_l, np.nan)
    else:
        freq_residual_mhz = np.array([])
        amp_ratio = np.array([])

    # Part A shape-error window subset.
    pa_windows = []
    for r in rows:
        if r.window_id in PART_A_SHAPE_ERROR_WINDOWS:
            delta_aic = float(r.aic_lorentz - r.aic_gauss)
            pa_windows.append({
                "window_id": int(r.window_id),
                "chi2r_lorentz": float(r.chi2r_lorentz),
                "chi2r_gauss": float(r.chi2r_gauss),
                "aic_lorentz": float(r.aic_lorentz),
                "aic_gauss": float(r.aic_gauss),
                "delta_aic": delta_aic,  # positive -> Gaussian preferred
                "gaussian_preferred": bool(delta_aic > 5.0),
            })

    return {
        "n_windows": int(len(rows)),
        "n_windows_finite_both": int(finite_both.sum()),
        "chi2r_lorentz_median": float(np.median(chi2r_l_f)) if chi2r_l_f.size else float("nan"),
        "chi2r_gauss_median": float(np.median(chi2r_g_f)) if chi2r_g_f.size else float("nan"),
        "chi2r_lorentz_p95": float(np.percentile(chi2r_l_f, 95)) if chi2r_l_f.size else float("nan"),
        "chi2r_gauss_p95": float(np.percentile(chi2r_g_f, 95)) if chi2r_g_f.size else float("nan"),
        "chi2r_gauss_lower_count": int(np.sum(chi2r_g_f < chi2r_l_f)),
        "aic_lorentz_median": float(np.median(aic_l_f)) if aic_l_f.size else float("nan"),
        "aic_gauss_median": float(np.median(aic_g_f)) if aic_g_f.size else float("nan"),
        "delta_aic_median": (
            float(np.median(aic_l_f - aic_g_f)) if aic_l_f.size else float("nan")
        ),
        "n_peaks_total_lorentz": int(n_l.sum()),
        "n_peaks_total_gauss": int(n_g.sum()),
        "n_peaks_matched": int(len(peaks)),
        "freq_residual_mhz_median": (
            float(np.median(freq_residual_mhz))
            if freq_residual_mhz.size else float("nan")
        ),
        "freq_residual_mhz_rms": (
            float(np.sqrt(np.mean(freq_residual_mhz ** 2)))
            if freq_residual_mhz.size else float("nan")
        ),
        "amp_ratio_median": (
            float(np.nanmedian(amp_ratio)) if amp_ratio.size else float("nan")
        ),
        "part_a_shape_error_windows": pa_windows,
    }


def _build_rows(
    fit_lorentz, fit_gauss,
) -> tuple[list[WindowRecord], list[MatchedPeak]]:
    by_l = {wf.window_id: wf for wf in fit_lorentz.window_fits}
    by_g = {wf.window_id: wf for wf in fit_gauss.window_fits}
    shared_ids = sorted(set(by_l) & set(by_g))
    only_l = sorted(set(by_l) - set(by_g))
    only_g = sorted(set(by_g) - set(by_l))
    if only_l or only_g:
        logger.warning(
            "Window-id mismatch between shapes: %d Lorentzian-only, %d "
            "Gaussian-only (replan revisions diverged). Restricting "
            "comparison to %d shared ids.",
            len(only_l), len(only_g), len(shared_ids),
        )
    rows: list[WindowRecord] = []
    matched_peaks: list[MatchedPeak] = []
    for wid in shared_ids:
        wl = by_l[wid]
        wg = by_g[wid]
        fr_lo = float(wl.window.freq_range[0]) if wl.window is not None else float("nan")
        fr_hi = float(wl.window.freq_range[1]) if wl.window is not None else float("nan")
        rows.append(
            WindowRecord(
                window_id=int(wid),
                n_peaks_lorentz=int(len(wl.fitted_peaks)),
                n_peaks_gauss=int(len(wg.fitted_peaks)),
                chi2r_lorentz=float(wl.reduced_chi2),
                chi2r_gauss=float(wg.reduced_chi2),
                aic_lorentz=float(wl.aic),
                aic_gauss=float(wg.aic),
                freq_lo_mhz=fr_lo,
                freq_hi_mhz=fr_hi,
            )
        )
        matched_peaks.extend(
            _match_peaks(
                wl.fitted_peaks, wg.fitted_peaks, window_id=wid, tol_mhz=0.5,
            )
        )
    return rows, matched_peaks


def _save_csvs(
    stem: str, rows: list[WindowRecord], peaks: list[MatchedPeak],
) -> tuple[Path, Path]:
    win_path = DATA / f"{stem}_per_window.csv"
    pk_path = DATA / f"{stem}_per_peak.csv"
    with win_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "window_id", "freq_lo_mhz", "freq_hi_mhz",
            "n_peaks_lorentz", "n_peaks_gauss",
            "chi2r_lorentz", "chi2r_gauss",
            "aic_lorentz", "aic_gauss",
            "delta_aic_lorentz_minus_gauss",
        ])
        for r in rows:
            w.writerow([
                r.window_id, f"{r.freq_lo_mhz:.6f}", f"{r.freq_hi_mhz:.6f}",
                r.n_peaks_lorentz, r.n_peaks_gauss,
                f"{r.chi2r_lorentz:.4f}", f"{r.chi2r_gauss:.4f}",
                f"{r.aic_lorentz:.4f}", f"{r.aic_gauss:.4f}",
                f"{(r.aic_lorentz - r.aic_gauss):.4f}",
            ])
    with pk_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "window_id",
            "freq_lorentz_mhz", "freq_gauss_mhz", "freq_residual_mhz",
            "amp_lorentz", "amp_gauss",
            "snr_lorentz", "snr_gauss",
        ])
        for p in peaks:
            w.writerow([
                p.window_id,
                f"{p.freq_lorentz_mhz:.6f}", f"{p.freq_gauss_mhz:.6f}",
                f"{(p.freq_gauss_mhz - p.freq_lorentz_mhz):.6f}",
                f"{p.amp_lorentz:.6f}", f"{p.amp_gauss:.6f}",
                f"{p.snr_lorentz:.2f}", f"{p.snr_gauss:.2f}",
            ])
    return win_path, pk_path


def _save_figure(
    stem: str, rows: list[WindowRecord], peaks: list[MatchedPeak],
) -> Path:
    fig, (ax_chi, ax_peak) = plt.subplots(1, 2, figsize=(13.5, 6.0))

    # Panel 1: per-window χ²ᵣ scatter.
    if rows:
        chi2r_l = np.array([r.chi2r_lorentz for r in rows], dtype=float)
        chi2r_g = np.array([r.chi2r_gauss for r in rows], dtype=float)
        finite = np.isfinite(chi2r_l) & np.isfinite(chi2r_g)
        wid = np.array([r.window_id for r in rows])
        ax_chi.scatter(
            chi2r_l[finite], chi2r_g[finite], s=10, alpha=0.55,
            c="tab:blue", edgecolor="none", label=f"all windows (N={int(finite.sum())})",
        )
        # Highlight Part A shape-error windows.
        pa_mask = np.array([w in PART_A_SHAPE_ERROR_WINDOWS for w in wid]) & finite
        if pa_mask.any():
            ax_chi.scatter(
                chi2r_l[pa_mask], chi2r_g[pa_mask], s=80,
                facecolor="none", edgecolor="tab:red", linewidth=1.5,
                label=f"Part A shape-error windows (N={int(pa_mask.sum())})",
            )
        lim_lo = 0.3
        lim_hi = max(
            np.percentile(chi2r_l[finite], 99) if finite.any() else 10,
            np.percentile(chi2r_g[finite], 99) if finite.any() else 10,
            10.0,
        ) * 1.1
        ax_chi.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=0.7,
                    label="Gaussian = Lorentzian")
        ax_chi.axhline(1.0, color="tab:green", lw=0.7, alpha=0.7)
        ax_chi.set_xlim(lim_lo, lim_hi)
        ax_chi.set_ylim(lim_lo, lim_hi)
        ax_chi.set_xscale("log")
        ax_chi.set_yscale("log")
    ax_chi.set_xlabel(r"Lorentzian $\chi^2_r$")
    ax_chi.set_ylabel(r"Gaussian $\chi^2_r$")
    ax_chi.set_title("Per-window χ²ᵣ: Lorentzian vs Gaussian")
    ax_chi.grid(True, which="both", alpha=0.3)
    ax_chi.legend(loc="lower right", fontsize=8)

    # Panel 2: per-peak frequency residual scatter on shared lines.
    if peaks:
        freq_resid_khz = 1000.0 * np.array(
            [p.freq_gauss_mhz - p.freq_lorentz_mhz for p in peaks], dtype=float
        )
        snr_l = np.array(
            [p.snr_lorentz if np.isfinite(p.snr_lorentz) else 1.0 for p in peaks],
            dtype=float,
        )
        # Two-stage scatter to show frequency-error context.
        ax_peak.scatter(
            snr_l, freq_resid_khz, s=10, alpha=0.55, c="tab:purple",
            edgecolor="none",
            label=f"matched lines (N={len(peaks)})",
        )
        ax_peak.axhline(0.0, color="k", lw=0.7, linestyle="--")
        ax_peak.set_xscale("log")
        ax_peak.set_xlim(left=1.0)
    ax_peak.set_xlabel("Lorentzian SNR")
    ax_peak.set_ylabel(r"$f_{\rm Gauss} - f_{\rm Lorentz}$ (kHz)")
    ax_peak.set_title("Per-peak frequency residual on shared lines")
    ax_peak.grid(True, alpha=0.3)
    ax_peak.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"Stage 5 shape comparison: Lorentzian vs Gaussian -- {stem}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIG / f"{stem}_panel.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def _console_summary(stem: str, summary: dict) -> None:
    print()
    print(f"====== Gaussian-shape comparison: {stem} ======")
    print(f"  Windows compared (both shapes finite): {summary.get('n_windows_finite_both', 0)} "
          f"/ {summary.get('n_windows', 0)}")
    print()
    print(f"  Median χ²ᵣ -- Lorentzian: {summary.get('chi2r_lorentz_median', float('nan')):>7.2f}  "
          f"Gaussian: {summary.get('chi2r_gauss_median', float('nan')):>7.2f}  "
          f"(Gaussian < Lorentzian: {summary.get('chi2r_gauss_lower_count', 0)} / "
          f"{summary.get('n_windows_finite_both', 0)} windows)")
    print(f"  p95 χ²ᵣ    -- Lorentzian: {summary.get('chi2r_lorentz_p95', float('nan')):>7.2f}  "
          f"Gaussian: {summary.get('chi2r_gauss_p95', float('nan')):>7.2f}")
    print(f"  Median AIC -- Lorentzian: {summary.get('aic_lorentz_median', float('nan')):>10.2f}  "
          f"Gaussian: {summary.get('aic_gauss_median', float('nan')):>10.2f}  "
          f"ΔAIC median: {summary.get('delta_aic_median', float('nan')):>8.2f}")
    print()
    print(f"  Peaks total -- Lorentzian: {summary.get('n_peaks_total_lorentz', 0):>5d}  "
          f"Gaussian: {summary.get('n_peaks_total_gauss', 0):>5d}  "
          f"matched: {summary.get('n_peaks_matched', 0):>5d}")
    print(f"  Peak frequency residual (G − L): "
          f"median = {summary.get('freq_residual_mhz_median', float('nan')):.4e} MHz, "
          f"RMS = {summary.get('freq_residual_mhz_rms', float('nan')):.4e} MHz")
    print(f"  Peak amplitude ratio  (G / L): "
          f"median = {summary.get('amp_ratio_median', float('nan')):.3f}")
    print()
    pa = summary.get("part_a_shape_error_windows", [])
    if pa:
        print("  Part A shape-error windows (acceptance: ΔAIC > 5):")
        for entry in pa:
            mark = "✓" if entry["gaussian_preferred"] else "·"
            print(
                f"    {mark} w{entry['window_id']:>3d}  "
                f"χ²ᵣ L={entry['chi2r_lorentz']:.2f}  G={entry['chi2r_gauss']:.2f}  "
                f"AIC L={entry['aic_lorentz']:.1f}  G={entry['aic_gauss']:.1f}  "
                f"ΔAIC={entry['delta_aic']:+.2f}"
            )
    print("==========================================")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Stage 5 fit_peaks(shape='lorentzian') vs "
                    "fit_peaks(shape='gaussian') on the same .ftmw fixture."
    )
    parser.add_argument(
        "fixture",
        type=Path,
        help="Path to a .ftmw file with Stages 0-4 completed.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("scratch/gaussian-shape-compare"),
        help="Working directory for per-shape copies (default: "
             "scratch/gaussian-shape-compare).",
    )
    args = parser.parse_args()

    if not args.fixture.exists():
        raise SystemExit(f"Fixture missing: {args.fixture}")
    args.work_dir.mkdir(parents=True, exist_ok=True)

    stem = args.fixture.stem
    lorentz_fp = args.work_dir / f"{stem}_lorentzian.ftmw"
    gauss_fp = args.work_dir / f"{stem}_gaussian.ftmw"
    logger.info("Copying fixture into per-shape work files ...")
    shutil.copy(args.fixture, lorentz_fp)
    shutil.copy(args.fixture, gauss_fp)

    # Ensure τ_G calibration on the Gaussian copy (Lorentzian copy already
    # has /stage2b_tau_calibration from the fixture itself).
    _ensure_tau_G_calibration(gauss_fp)

    _run_fit(lorentz_fp, shape="lorentzian")
    _run_fit(gauss_fp, shape="gaussian")

    fit_l = ftmw.load_fit(str(lorentz_fp))
    fit_g = ftmw.load_fit(str(gauss_fp))
    rows, peaks = _build_rows(fit_l, fit_g)

    summary = {
        "fixture": str(args.fixture),
        "lorentzian_file": str(lorentz_fp),
        "gaussian_file": str(gauss_fp),
        "lorentzian_summary": {
            "n_windows": int(fit_l.n_windows),
            "n_fitted_peaks": int(fit_l.n_fitted_peaks),
            "final_plan_revision": int(fit_l.final_plan_revision),
        },
        "gaussian_summary": {
            "n_windows": int(fit_g.n_windows),
            "n_fitted_peaks": int(fit_g.n_fitted_peaks),
            "final_plan_revision": int(fit_g.final_plan_revision),
        },
        "comparison": _aggregate(rows, peaks),
    }
    summary_path = DATA / f"{stem}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s", summary_path)

    win_csv, peak_csv = _save_csvs(stem, rows, peaks)
    logger.info("Wrote %s and %s", win_csv, peak_csv)

    fig_path = _save_figure(stem, rows, peaks)
    logger.info("Wrote %s", fig_path)

    _console_summary(stem, summary["comparison"])


if __name__ == "__main__":
    main()
