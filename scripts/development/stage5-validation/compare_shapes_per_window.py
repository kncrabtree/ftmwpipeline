"""Direct side-by-side Lorentzian-vs-Gaussian per-window comparison figures.

Stage 5's validation harness produces parallel detail.png artifacts under
``scratch/gaussian-shape-validation/{lorentzian,gaussian}/window_NNN/`` but
reading those by flipping back and forth is slow and biases the eye. This
script overlays both models on the same axes per window so the difference
shows up at a glance:

* Top panel: data |X| + Lorentzian model + Gaussian model on shared axes
  (different colours).
* Middle panel: |residual_L| and |residual_G| overlaid + noise band.
* Bottom-left: residual histograms (L vs G).
* Bottom-right: peak listing — fitted peaks from both shapes side-by-side.

Selects windows via three preset modes:

* ``--mode strong-isolated`` (default): K_L = K_G = 1, no fixed
  contributors in either fit, max SNR > 50, min χ²ᵣ < 2. The cleanest
  shape-discrimination signal on the 2638 fixture.
* ``--mode broader-isolated``: same K=1 / no-fixed gate but SNR > 20 and
  min χ²ᵣ < 3. Wider net.
* ``--mode windows``: explicit ``--window-id N`` (repeatable). Overrides
  the preset filter.

Outputs under ``scratch/gaussian-shape-validation/per-window-compare/``:

* ``compare_NNN.png`` — one comparison figure per selected window.
* ``INDEX.md``        — table of selected windows with comparison stats.

Run from repo root::

    conda run -n ftmwpipeline-dev python \
        scripts/development/stage5-validation/compare_shapes_per_window.py \
        [--mode strong-isolated|broader-isolated|windows] \
        [--window-id N ...] \
        [--output-dir scratch/gaussian-shape-validation/per-window-compare]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.stage5_impl import _build_active_ft_inputs
from ftmwpipeline.fitting.active_ft import compute_active_ft
from ftmwpipeline.fitting.peak_model import (
    ModelPeak,
    PeakShape,
    model_spectrum,
    sideband_sign,
)
from ftmwpipeline.preprocessing.noise_estimation import estimate_noise_adaptive

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CSV_PATH = (
    REPO_ROOT / "dev-docs" / "research" / "gaussian-shape" / "data"
    / "exp_2638_unapodized_per_window.csv"
)
LORENTZ_FIXTURE = (
    REPO_ROOT / "scratch" / "gaussian-shape-compare"
    / "exp_2638_unapodized_lorentzian.ftmw"
)
GAUSS_FIXTURE = (
    REPO_ROOT / "scratch" / "gaussian-shape-compare"
    / "exp_2638_unapodized_gaussian.ftmw"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "scratch" / "gaussian-shape-validation" / "per-window-compare"
)

logger = logging.getLogger("compare-shapes-per-window")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


@dataclass
class WindowRow:
    wid: int
    flo: float
    fhi: float
    np_L: int
    np_G: int
    c2_L: float
    c2_G: float
    aic_L: float
    aic_G: float
    dAIC: float
    snr_L: float = 0.0
    snr_G: float = 0.0
    fixed_L: int = 0
    fixed_G: int = 0
    tau_L: float = float("nan")
    tau_G: float = float("nan")


def _load_window_rows() -> dict[int, WindowRow]:
    rows: dict[int, WindowRow] = {}
    with CSV_PATH.open() as fh:
        for r in csv.DictReader(fh):
            wid = int(r["window_id"])
            rows[wid] = WindowRow(
                wid=wid,
                flo=float(r["freq_lo_mhz"]),
                fhi=float(r["freq_hi_mhz"]),
                np_L=int(r["n_peaks_lorentz"]),
                np_G=int(r["n_peaks_gauss"]),
                c2_L=float(r["chi2r_lorentz"]),
                c2_G=float(r["chi2r_gauss"]),
                aic_L=float(r["aic_lorentz"]),
                aic_G=float(r["aic_gauss"]),
                dAIC=float(r["delta_aic_lorentz_minus_gauss"]),
            )

    def annotate(fixture: Path, key: str) -> None:
        with h5py.File(fixture, "r") as f:
            for wname in f["stage5_fitting/windows"]:
                wid = int(wname.split("_")[1])
                if wid not in rows:
                    continue
                g = f[f"stage5_fitting/windows/{wname}"]
                fp_raw = g.attrs.get("fixed_parameters", "{}")
                fp_str = (
                    fp_raw if isinstance(fp_raw, str) else fp_raw.decode("utf-8")
                )
                fp = json.loads(fp_str)
                n_fixed = sum(1 for k in fp if k.startswith("frozen_peak_"))
                setattr(rows[wid], f"fixed_{key}", n_fixed)
                if "peaks/snr" in g:
                    snr_arr = g["peaks/snr"][:]
                    if len(snr_arr) > 0:
                        setattr(rows[wid], f"snr_{key}", float(np.max(snr_arr)))
                setattr(
                    rows[wid], f"tau_{key}",
                    float(g.attrs.get("tau_us", float("nan"))),
                )

    annotate(LORENTZ_FIXTURE, "L")
    annotate(GAUSS_FIXTURE, "G")
    return rows


def _select_windows(
    rows: dict[int, WindowRow], mode: str, explicit_ids: Sequence[int],
) -> List[int]:
    if mode == "windows":
        if not explicit_ids:
            raise SystemExit(
                "--mode windows requires at least one --window-id"
            )
        missing = [w for w in explicit_ids if w not in rows]
        if missing:
            raise SystemExit(f"window ids not in fixture: {missing}")
        return list(explicit_ids)
    if mode == "strong-isolated":
        snr_floor, chi2_cap = 50.0, 2.0
    elif mode == "broader-isolated":
        snr_floor, chi2_cap = 20.0, 3.0
    else:
        raise SystemExit(f"unknown --mode: {mode}")
    out = []
    for wid, r in rows.items():
        if r.np_L != 1 or r.np_G != 1:
            continue
        if r.fixed_L > 0 or r.fixed_G > 0:
            continue
        if r.snr_L < snr_floor:
            continue
        if min(r.c2_L, r.c2_G) > chi2_cap:
            continue
        out.append(wid)
    out.sort(key=lambda w: -max(rows[w].snr_L, rows[w].snr_G))
    return out


def _build_window_model(
    wf, freq_slice: np.ndarray, sideband, acquisition_us: float,
) -> Tuple[np.ndarray, float, str]:
    """Sum the per-peak model on ``freq_slice`` for one shape's window fit.

    Returns ``(model_complex, tau_us, shape_str)``. ``shape_str`` is
    ``'lorentzian'`` or ``'gaussian'`` as carried by the persisted fit.
    """
    if wf.window is None:
        raise ValueError(f"window fit {wf.window_id} has no SpectralWindow")
    s = sideband_sign(sideband)
    lo, hi = wf.window.freq_range
    center = 0.5 * (lo + hi)
    u_slice = s * (freq_slice - center)
    tau_us = float(wf.shared_parameters.get("tau_us", {}).get("value", 0.0))
    shape_str = getattr(wf, "shape", "lorentzian")
    peaks = [
        ModelPeak(
            amplitude=float(p.amplitude),
            offset_mhz=float(s * (p.frequency_mhz - center)),
            phase=float(p.phase if p.phase is not None else 0.0),
        )
        for p in wf.fitted_peaks
    ]
    for key, fp_data in wf.fixed_parameters.items():
        if not key.startswith("frozen_peak_"):
            continue
        peaks.append(
            ModelPeak(
                amplitude=float(fp_data["amplitude"]),
                offset_mhz=float(
                    s * (float(fp_data["frequency_mhz"]) - center)
                ),
                phase=float(fp_data.get("phase", 0.0) or 0.0),
            )
        )
    if not peaks or tau_us <= 0:
        return np.zeros_like(freq_slice, dtype=np.complex128), tau_us, shape_str
    model = model_spectrum(
        u_slice, peaks, tau_us, acquisition_us, shape=shape_str,
    )
    return model, tau_us, shape_str


def _format_peak_row(p) -> str:
    f_str = f"{float(p.frequency_mhz):.4f}"
    if p.frequency_error is not None and np.isfinite(p.frequency_error):
        f_str += f" ±{float(p.frequency_error):.4f}"
    a_str = f"{float(p.amplitude):.3e}"
    snr_str = f"{float(p.snr):.1f}" if p.snr is not None else "—"
    return f"{f_str} MHz, A={a_str}, SNR={snr_str}"


def _save_compare_figure(
    out_path: Path,
    *,
    wid: int,
    freq_slice: np.ndarray,
    z_slice: np.ndarray,
    sigma_slice: np.ndarray,
    model_L: np.ndarray,
    model_G: np.ndarray,
    tau_L: float,
    tau_G: float,
    wf_L,
    wf_G,
    row: WindowRow,
    amp_scale: float = 1e6,
    units_lbl: str = "µV",
) -> None:
    """Emit one ``compare_NNN.png`` with both models overlaid.

    Layout (top to bottom):
      Row 1: |X| data + L model + G model overlaid (one axes).
      Row 2: Re and Im residuals overlaid (two axes side-by-side).
      Row 3: |residual_L| - |residual_G| (positive ⇒ L worse at that bin),
             with band lines at ±3σ_c for reference.
      Row 4: residual histograms (L and G) overlaid + Rayleigh
             curve at σ_c = median(σ)/√2.
    """
    res_L = z_slice - model_L
    res_G = z_slice - model_G
    sigma_c_med = float(np.median(sigma_slice)) / np.sqrt(2.0)

    fig = plt.figure(figsize=(12.5, 10.5))
    suptitle = (
        f"Window {wid}  [{row.flo:.2f}–{row.fhi:.2f}] MHz   "
        f"L: K={row.np_L} χ²ᵣ={row.c2_L:.2f} τ={tau_L:.2f} µs  |  "
        f"G: K={row.np_G} χ²ᵣ={row.c2_G:.2f} τ={tau_G:.2f} µs   "
        f"ΔAIC={row.dAIC:+.1f}"
    )
    fig.suptitle(suptitle, fontsize=11)
    gs = GridSpec(
        nrows=4, ncols=2, figure=fig,
        height_ratios=[2.0, 1.4, 1.4, 1.2],
        hspace=0.32, wspace=0.22,
        left=0.07, right=0.97, top=0.94, bottom=0.06,
    )
    ax_mag = fig.add_subplot(gs[0, :])
    ax_re = fig.add_subplot(gs[1, 0], sharex=ax_mag)
    ax_im = fig.add_subplot(gs[1, 1], sharex=ax_mag)
    ax_diff = fig.add_subplot(gs[2, :], sharex=ax_mag)
    ax_hist = fig.add_subplot(gs[3, 0])
    ax_peaks = fig.add_subplot(gs[3, 1])
    ax_peaks.set_axis_off()

    f_mhz = freq_slice
    # --- Row 1: magnitude with both model overlays ---------------------
    ax_mag.plot(
        f_mhz, np.abs(z_slice) * amp_scale, color="0.25", lw=0.6,
        zorder=1, label="data |X|",
    )
    ax_mag.plot(
        f_mhz, np.abs(z_slice) * amp_scale, marker="o", linestyle="None",
        markersize=2.0, markerfacecolor="0.1", markeredgecolor="0.1",
        zorder=2,
    )
    ax_mag.plot(
        f_mhz, np.abs(model_L) * amp_scale, color="tab:red", lw=1.4,
        zorder=4,
        label=f"Lorentzian (χ²ᵣ={row.c2_L:.2f})",
    )
    ax_mag.plot(
        f_mhz, np.abs(model_G) * amp_scale, color="tab:blue", lw=1.4,
        ls="--", zorder=4,
        label=f"Gaussian (χ²ᵣ={row.c2_G:.2f})",
    )
    # Mark fitted-peak centres for each shape
    for p in wf_L.fitted_peaks:
        ax_mag.axvline(
            float(p.frequency_mhz), color="tab:red", lw=0.6, ls=":",
            alpha=0.5, zorder=0,
        )
    for p in wf_G.fitted_peaks:
        ax_mag.axvline(
            float(p.frequency_mhz), color="tab:blue", lw=0.6, ls=":",
            alpha=0.5, zorder=0,
        )
    ax_mag.set_ylabel(f"|X| ({units_lbl})", fontsize=10)
    ax_mag.set_title("data + both models", fontsize=9)
    ax_mag.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_mag.grid(True, alpha=0.25)
    ax_mag.tick_params(axis="x", labelbottom=False)

    # --- Row 2: Re/Im residuals overlaid -------------------------------
    band = 3.0 * sigma_c_med * amp_scale
    for ax, comp, label in (
        (ax_re, np.real, "Re"),
        (ax_im, np.imag, "Im"),
    ):
        ax.axhline(0.0, color="0.5", lw=0.4)
        ax.axhline(band, color="0.3", lw=0.5, ls="--")
        ax.axhline(-band, color="0.3", lw=0.5, ls="--")
        ax.plot(
            f_mhz, comp(res_L) * amp_scale, color="tab:red", lw=0.7,
            label="L residual",
        )
        ax.plot(
            f_mhz, comp(res_G) * amp_scale, color="tab:blue", lw=0.7,
            ls="--", label="G residual",
        )
        ax.set_ylabel(f"{label} residual ({units_lbl})", fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="x", labelbottom=False)
    ax_re.legend(loc="upper right", fontsize=7, framealpha=0.9)

    # --- Row 3: |residual_L| - |residual_G| (signed) -------------------
    diff = (np.abs(res_L) - np.abs(res_G)) * amp_scale
    ax_diff.axhline(0.0, color="0.4", lw=0.6)
    ax_diff.plot(
        f_mhz, diff, color="tab:purple", lw=0.7, label="|res L| − |res G|",
    )
    ax_diff.fill_between(
        f_mhz, 0.0, diff, where=(diff > 0), color="tab:red", alpha=0.18,
        label="G fits better here",
    )
    ax_diff.fill_between(
        f_mhz, 0.0, diff, where=(diff < 0), color="tab:blue", alpha=0.18,
        label="L fits better here",
    )
    ax_diff.set_ylabel(f"|res| diff ({units_lbl})", fontsize=9)
    ax_diff.set_xlabel("frequency (MHz)", fontsize=10)
    ax_diff.grid(True, alpha=0.25)
    ax_diff.legend(loc="upper right", fontsize=7, framealpha=0.9)

    # --- Row 4 left: residual histograms -------------------------------
    mag_L = np.abs(res_L) * amp_scale
    mag_G = np.abs(res_G) * amp_scale
    sigma_c = sigma_c_med * amp_scale
    if mag_L.size > 0 and sigma_c > 0:
        x_hi = max(float(mag_L.max()), float(mag_G.max()), 5.0 * sigma_c)
        bins = np.linspace(0.0, x_hi, 30)
        ax_hist.hist(
            mag_L, bins=bins, density=True, color="tab:red", alpha=0.45,
            edgecolor="tab:red", label=f"L (median {np.median(mag_L):.2f})",
        )
        ax_hist.hist(
            mag_G, bins=bins, density=True, color="tab:blue", alpha=0.45,
            edgecolor="tab:blue", label=f"G (median {np.median(mag_G):.2f})",
        )
        x = np.linspace(0.0, x_hi, 200)
        rayleigh = (x / (sigma_c ** 2)) * np.exp(-(x ** 2) / (2.0 * sigma_c ** 2))
        ax_hist.plot(
            x, rayleigh, color="0.25", lw=1.0,
            label=r"Rayleigh($\sigma_c$)",
        )
        ax_hist.axvline(
            3.0 * sigma_c, color="0.4", lw=0.5, ls="--",
        )
    ax_hist.set_xlabel(f"|residual| ({units_lbl})", fontsize=9)
    ax_hist.set_ylabel("density", fontsize=9)
    ax_hist.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax_hist.set_title("|residual| distributions", fontsize=9)
    ax_hist.grid(True, alpha=0.25)

    # --- Row 4 right: peak listing -------------------------------------
    ax_peaks.set_title("Fitted peaks", fontsize=9, loc="left")
    lines = ["Lorentzian:"]
    for p in wf_L.fitted_peaks:
        lines.append("  " + _format_peak_row(p))
    lines.append("")
    lines.append("Gaussian:")
    for p in wf_G.fitted_peaks:
        lines.append("  " + _format_peak_row(p))
    ax_peaks.text(
        0.02, 0.97, "\n".join(lines),
        transform=ax_peaks.transAxes, family="monospace", fontsize=8.0,
        va="top",
    )

    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _slice_window(
    wf, freqs_sorted: np.ndarray, spec_sorted: np.ndarray,
    rms_sorted: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, hi = wf.window.freq_range
    mask = (freqs_sorted >= min(lo, hi)) & (freqs_sorted <= max(lo, hi))
    return freqs_sorted[mask], spec_sorted[mask], rms_sorted[mask]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("strong-isolated", "broader-isolated", "windows"),
        default="strong-isolated",
        help="Window selection mode (default: strong-isolated).",
    )
    parser.add_argument(
        "--window-id", type=int, action="append", dest="window_ids",
        default=[],
        help=(
            "Used with --mode windows: restrict to this id. Repeat for "
            "multiple."
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    if not CSV_PATH.exists():
        raise SystemExit(
            f"missing {CSV_PATH}; run compare_shapes.py first."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_window_rows()
    selected = _select_windows(rows, args.mode, args.window_ids)
    logger.info(
        "Selected %d windows (mode=%s)", len(selected), args.mode,
    )

    # Load both fits + a shared active-FT (the L and G fixtures share the
    # same FID + canonical Stage 1 settings, so the active-FT is identical
    # bit-for-bit between them; we just use the Lorentzian fixture as the
    # source of truth for the spectrum grid).
    fit_L = ftmw.load_fit(str(LORENTZ_FIXTURE))
    fit_G = ftmw.load_fit(str(GAUSS_FIXTURE))
    by_L = {wf.window_id: wf for wf in fit_L.window_fits}
    by_G = {wf.window_id: wf for wf in fit_G.window_fits}

    (
        fid_samples, sample_dt_us, start_us, end_us, expf_us,
        probe_freq_mhz, sideband_enum, n_padded,
        acquisition_us, _user_ft, _user_rms,
    ) = _build_active_ft_inputs(str(LORENTZ_FIXTURE))
    active_ft = compute_active_ft(
        fid_samples, sample_dt_us,
        start_us=start_us, end_us=end_us, expf_us=expf_us,
        probe_freq_mhz=probe_freq_mhz, sideband=sideband_enum,
        n_padded=n_padded,
    )
    sort_idx = np.argsort(active_ft.freq_mhz)
    freqs_sorted = np.ascontiguousarray(active_ft.freq_mhz[sort_idx])
    spec_sorted = np.ascontiguousarray(active_ft.complex_spectrum[sort_idx])
    active_noise = estimate_noise_adaptive(
        freqs_sorted, np.abs(spec_sorted).astype(np.float64),
    )
    rms_sorted = np.asarray(active_noise.rms_noise, dtype=float)

    index_lines: List[str] = [
        "# Lorentzian vs Gaussian per-window direct comparison",
        "",
        f"Mode: `{args.mode}`. Selected {len(selected)} windows.",
        "",
        "Each row links to a single comparison figure that overlays both "
        "models on the same axes (data + L + G), with a signed "
        "|residual| difference panel (red shading = Gaussian fits better, "
        "blue = Lorentzian fits better) and residual histograms.",
        "",
        "| wid | freq range (MHz) | SNR (L / G) | χ²ᵣ (L / G) | ΔAIC | τ (L / G) µs | figure |",
        "|---:|---|:---:|:---:|---:|:---:|---|",
    ]
    for wid in selected:
        if wid not in by_L or wid not in by_G:
            logger.warning("window %d missing from one fixture; skipping", wid)
            continue
        wf_L = by_L[wid]
        wf_G = by_G[wid]
        f_slice, z_slice, sig_slice = _slice_window(
            wf_L, freqs_sorted, spec_sorted, rms_sorted,
        )
        model_L, tau_L, _ = _build_window_model(
            wf_L, f_slice, sideband_enum, acquisition_us,
        )
        model_G, tau_G, _ = _build_window_model(
            wf_G, f_slice, sideband_enum, acquisition_us,
        )
        out_path = args.output_dir / f"compare_{wid:03d}.png"
        _save_compare_figure(
            out_path,
            wid=wid,
            freq_slice=f_slice, z_slice=z_slice, sigma_slice=sig_slice,
            model_L=model_L, model_G=model_G,
            tau_L=tau_L, tau_G=tau_G,
            wf_L=wf_L, wf_G=wf_G,
            row=rows[wid],
        )
        logger.info("wrote %s", out_path.name)
        r = rows[wid]
        index_lines.append(
            f"| {wid} | {r.flo:.2f}–{r.fhi:.2f} | "
            f"{r.snr_L:.0f} / {r.snr_G:.0f} | "
            f"{r.c2_L:.2f} / {r.c2_G:.2f} | "
            f"{r.dAIC:+.1f} | "
            f"{r.tau_L:.2f} / {r.tau_G:.2f} | "
            f"[compare]({out_path.name}) |"
        )

    (args.output_dir / "INDEX.md").write_text("\n".join(index_lines) + "\n")
    print(f"wrote {args.output_dir / 'INDEX.md'}")


if __name__ == "__main__":
    main()
