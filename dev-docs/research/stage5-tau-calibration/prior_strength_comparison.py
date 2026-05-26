"""Three-mode comparison of Stage 5 tau under different Stage 2b priors.

Runs Stage 5 on the unapodized 2638 fixture under three configurations:

* **no-prior**: Stage 2b stripped from the fixture so no calibration
  drives the fit; tau bounded only by the legacy apodization ceiling.
* **band-wide prior**: Stage 2b persisted but
  ``fit_peaks(per_band_tau=False)`` -- every window anchors on the
  same band-wide ``(tau_maj, sigma_tau)``.
* **per-band prior**: Stage 2b persisted with band majorities and
  ``fit_peaks(per_band_tau=True)`` (the production default) -- each
  window anchors on the band-local ``(tau_maj_band, sigma_tau_band)``
  computed at calibration time.

For each mode the script reports the per-band τ central tendency
across the strict-pool windows (the same gate
:mod:`lsq_comparison.py` uses) and the chi-squared-reduced
distribution across all windows. Emits a 3-panel version of
``figures/13_lsq_freq_third.png`` saved as ``15_prior_strength.png``
and a summary JSON in ``data/prior_strength_comparison.json``.

Run from the repository root:

    conda run -n ftmwpipeline-dev python \\
        dev-docs/research/stage5-tau-calibration/prior_strength_comparison.py

Output files:

- ``data/prior_strength_comparison.json``
- ``figures/15_prior_strength.png``
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ftmwpipeline.api as ftmw

HERE = Path(__file__).parent
FIG = HERE / "figures"
DATA = HERE / "data"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

SCRATCH = Path("scratch/stage5-tau-calibration-lsq")
SOURCE_FIXTURE = SCRATCH / "exp_2638_unapodized.ftmw"

# Same filter gates as lsq_comparison.py.
SNR_GATE = 10.0
TAU_REL_ERR_GATE = 0.10
RCHI2_GATE = 3.0
TAU_BOUND_LO = 6.325 / 5.0
TAU_BOUND_HI = 6.325 * 5.0
TAU_LO_SATURATE_FACTOR = 1.05
TAU_HI_SATURATE_FACTOR = 0.95

TRIM_LO_MHZ = 26500.0
TRIM_HI_MHZ = 40000.0

ARITHMETIC_THIRDS = [
    ("low",  TRIM_LO_MHZ, TRIM_LO_MHZ + (TRIM_HI_MHZ - TRIM_LO_MHZ) / 3.0),
    ("mid",  TRIM_LO_MHZ + (TRIM_HI_MHZ - TRIM_LO_MHZ) / 3.0,
             TRIM_LO_MHZ + 2.0 * (TRIM_HI_MHZ - TRIM_LO_MHZ) / 3.0),
    ("high", TRIM_LO_MHZ + 2.0 * (TRIM_HI_MHZ - TRIM_LO_MHZ) / 3.0, TRIM_HI_MHZ),
]

logger = logging.getLogger("prior-strength")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


@dataclass
class WindowRow:
    window_id: int
    center_mhz: float
    tau_us: float
    tau_err_us: Optional[float]
    reduced_chi2: float
    n_fixed_contributors: int
    tau_was_fit: bool
    passes_strict: bool


def _strict_pass(
    *,
    tau_us: float,
    tau_err: Optional[float],
    reduced_chi2: float,
    n_fixed: int,
    tau_was_fit: bool,
) -> bool:
    if not tau_was_fit:
        return False
    if n_fixed != 0:
        return False
    if tau_err is None or not np.isfinite(tau_err) or tau_err <= 0.0:
        return False
    if tau_us <= 0.0 or tau_err / tau_us >= TAU_REL_ERR_GATE:
        return False
    if not (np.isfinite(reduced_chi2) and reduced_chi2 < RCHI2_GATE):
        return False
    if tau_us < TAU_LO_SATURATE_FACTOR * TAU_BOUND_LO:
        return False
    if tau_us > TAU_HI_SATURATE_FACTOR * TAU_BOUND_HI:
        return False
    return True


def collect(file_path: Path) -> list[WindowRow]:
    rows: list[WindowRow] = []
    with h5py.File(file_path, "r") as h5f:
        windows = h5f["stage5_fitting/windows"]
        for name in sorted(windows.keys()):
            wg = windows[name]
            wid = int(wg.attrs["window_id"])
            tau_us = float(wg.attrs["tau_us"])
            tau_err_raw = float(wg.attrs["tau_error"])
            tau_err = tau_err_raw if np.isfinite(tau_err_raw) else None
            rchi2 = float(wg.attrs["reduced_chi2"])
            tau_was_fit = int(wg.attrs.get("tau_fitted", -1)) == 1
            freq_lo = float(wg.attrs["freq_min"])
            freq_hi = float(wg.attrs["freq_max"])
            n_fixed = len(json.loads(wg.attrs.get("fixed_parameters", "{}") or "{}"))
            rows.append(
                WindowRow(
                    window_id=wid,
                    center_mhz=0.5 * (freq_lo + freq_hi),
                    tau_us=tau_us,
                    tau_err_us=tau_err,
                    reduced_chi2=rchi2,
                    n_fixed_contributors=n_fixed,
                    tau_was_fit=tau_was_fit,
                    passes_strict=_strict_pass(
                        tau_us=tau_us, tau_err=tau_err, reduced_chi2=rchi2,
                        n_fixed=n_fixed, tau_was_fit=tau_was_fit,
                    ),
                )
            )
    return rows


def _build_fixture(label: str, mode: str) -> Path:
    """Build a working fixture for the given mode.

    Modes:
        - ``"no_prior"``: copy SOURCE_FIXTURE, strip Stage 2b and Stage 5,
          re-fit (Stage 2b absent -> no prior).
        - ``"band_wide"``: copy SOURCE_FIXTURE, ensure Stage 2b has band
          majorities, then re-fit with per_band_tau=False.
        - ``"per_band"``: copy SOURCE_FIXTURE, ensure Stage 2b has band
          majorities, then re-fit with per_band_tau=True (default).
    """
    dst = SCRATCH / f"exp_2638_unapodized_priorcmp_{label}.ftmw"
    shutil.copy(SOURCE_FIXTURE, dst)

    with h5py.File(dst, "a") as h5f:
        if "stage5_fitting" in h5f:
            del h5f["stage5_fitting"]
        completed = json.loads(h5f["pipeline_stages"].attrs["completed_stages"])
        completed = [s for s in completed if s != "stage5_fitting"]
        if mode == "no_prior":
            if "stage2b_tau_calibration" in h5f:
                del h5f["stage2b_tau_calibration"]
            completed = [s for s in completed if s != "stage2b_tau_calibration"]
        h5f["pipeline_stages"].attrs["completed_stages"] = json.dumps(completed)

    # Make sure Stage 2b's band_majorities are populated for the prior modes.
    if mode in ("band_wide", "per_band"):
        cal = ftmw.load_tau_calibration(dst)
        if not cal.band_majorities:
            logger.info("Re-running calibrate_tau on %s to populate band_majorities", dst)
            ftmw.calibrate_tau(dst, compute_band_majorities=True)

    if mode == "per_band":
        ftmw.fit_peaks(dst, tau0_us=6.325, per_band_tau=True)
    elif mode == "band_wide":
        ftmw.fit_peaks(dst, tau0_us=6.325, per_band_tau=False)
    elif mode == "no_prior":
        ftmw.fit_peaks(dst, tau0_us=6.325)
    else:
        raise ValueError(f"unknown mode: {mode!r}")
    return dst


def _summarize(label: str, rows: list[WindowRow]) -> dict:
    rchi2_all = np.array(
        [r.reduced_chi2 for r in rows if np.isfinite(r.reduced_chi2)], dtype=float,
    )
    strict = [r for r in rows if r.passes_strict]
    summary: dict = {
        "label": label,
        "n_total": len(rows),
        "n_strict": len(strict),
        "chi2r_all": {
            "median": float(np.median(rchi2_all)) if rchi2_all.size else float("nan"),
            "mean": float(rchi2_all.mean()) if rchi2_all.size else float("nan"),
            "p90": float(np.percentile(rchi2_all, 90)) if rchi2_all.size else float("nan"),
            "p99": float(np.percentile(rchi2_all, 99)) if rchi2_all.size else float("nan"),
        },
        "bands": {},
    }
    if strict:
        taus_all = np.array([r.tau_us for r in strict])
        rch_all = np.array([r.reduced_chi2 for r in strict])
        summary["strict_overall"] = {
            "tau_mean": float(taus_all.mean()),
            "tau_median": float(np.median(taus_all)),
            "tau_std": float(taus_all.std(ddof=1)) if len(taus_all) > 1 else float("nan"),
            "chi2r_median": float(np.median(rch_all)),
            "chi2r_mean": float(rch_all.mean()),
        }
    else:
        summary["strict_overall"] = {}

    for bname, lo, hi in ARITHMETIC_THIRDS:
        band_rows = [r for r in strict if lo <= r.center_mhz < hi]
        if band_rows:
            taus = np.array([r.tau_us for r in band_rows])
            errs = np.array(
                [r.tau_err_us if r.tau_err_us is not None else float("nan")
                 for r in band_rows]
            )
            rchs = np.array([r.reduced_chi2 for r in band_rows])
            summary["bands"][bname] = {
                "band_lo_mhz": float(lo),
                "band_hi_mhz": float(hi),
                "n": len(band_rows),
                "tau_mean": float(taus.mean()),
                "tau_median": float(np.median(taus)),
                "tau_std": float(taus.std(ddof=1)) if len(taus) > 1 else float("nan"),
                "tau_err_median": float(np.nanmedian(errs)),
                "chi2r_median": float(np.median(rchs)),
            }
        else:
            summary["bands"][bname] = {
                "band_lo_mhz": float(lo),
                "band_hi_mhz": float(hi),
                "n": 0,
                "tau_mean": float("nan"),
                "tau_median": float("nan"),
                "tau_std": float("nan"),
                "tau_err_median": float("nan"),
                "chi2r_median": float("nan"),
            }
    return summary


def _draw_panel(
    ax,
    title: str,
    rows: list[WindowRow],
    summary: dict,
    *,
    show_ylabel: bool = False,
):
    strict = [r for r in rows if r.passes_strict]
    centers = np.array([r.center_mhz for r in strict])
    taus = np.array([r.tau_us for r in strict])
    errs = np.array(
        [r.tau_err_us if r.tau_err_us is not None else float("nan") for r in strict],
        dtype=float,
    )
    if strict:
        ax.errorbar(
            centers, taus, yerr=errs,
            fmt="o", ms=4.5, alpha=0.7, color="tab:blue", ecolor="tab:blue",
            elinewidth=0.8, capsize=2, zorder=3,
            label=f"strict windows (N={len(strict)})",
        )
    band_colors = ("tab:orange", "tab:green", "tab:red")
    for (bname, info), color in zip(summary["bands"].items(), band_colors):
        lo, hi = info["band_lo_mhz"], info["band_hi_mhz"]
        med = info["tau_median"]
        n = info["n"]
        if np.isfinite(med):
            ax.hlines(
                med, lo, hi, color=color, lw=2.5,
                label=f"{bname}: median={med:.2f} (N={n})",
            )
    ax.set_xlabel("window centre frequency (MHz)")
    if show_ylabel:
        ax.set_ylabel(r"$\tau$ (µs)")
    ax.set_title(title)
    ax.set_ylim(2.5, 12.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)


def main() -> None:
    logger.info("Building no-prior fixture ...")
    fp_no = _build_fixture("noprior", "no_prior")
    logger.info("Building band-wide-prior fixture ...")
    fp_band = _build_fixture("bandwide", "band_wide")
    logger.info("Building per-band-prior fixture ...")
    fp_per = _build_fixture("perband", "per_band")

    rows_no = collect(fp_no)
    rows_band = collect(fp_band)
    rows_per = collect(fp_per)

    summaries = {
        "no_prior": _summarize("no prior (Stage 2b stripped)", rows_no),
        "band_wide": _summarize("band-wide prior", rows_band),
        "per_band": _summarize("per-band prior (production default)", rows_per),
    }

    out_json = DATA / "prior_strength_comparison.json"
    out_json.write_text(json.dumps(summaries, indent=2))
    logger.info("Wrote %s", out_json)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0), sharey=True)
    _draw_panel(axes[0], "no prior", rows_no, summaries["no_prior"], show_ylabel=True)
    _draw_panel(axes[1], "band-wide prior", rows_band, summaries["band_wide"])
    _draw_panel(
        axes[2], "per-band prior (production default)", rows_per, summaries["per_band"],
    )
    fig.suptitle(
        r"Stage 2b prior strength on 2638 -- per-window $\tau$ across the band",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png = FIG / "15_prior_strength.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    logger.info("Wrote %s", out_png)

    print()
    print("====== prior strength comparison (2638 unapodized) ======")
    header = (
        f"  {'mode':<22} {'N strict':>8} {'low τ':>14} {'mid τ':>14} "
        f"{'high τ':>14} {'all χ²ᵣ med':>12}"
    )
    print(header)
    for key in ("no_prior", "band_wide", "per_band"):
        s = summaries[key]
        bands = s["bands"]
        print(
            f"  {s['label']:<22} {s['n_strict']:>8d} "
            f"{bands['low']['tau_mean']:>6.2f}+/-{bands['low']['tau_std']:>5.2f} "
            f"{bands['mid']['tau_mean']:>6.2f}+/-{bands['mid']['tau_std']:>5.2f} "
            f"{bands['high']['tau_mean']:>6.2f}+/-{bands['high']['tau_std']:>5.2f} "
            f"{s['chi2r_all']['median']:>12.3f}"
        )
    print("==========================================================")


if __name__ == "__main__":
    main()
