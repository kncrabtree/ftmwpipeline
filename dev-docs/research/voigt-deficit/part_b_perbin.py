"""Per-bin Voigt fits on STFT bins to test the Voigt-shape hypothesis (Part B).

Independent of Part A. Tests whether the per-bin ``|S_n(a)|`` time evolution
across the Stage 2b STFT frames is consistent with a Voigt envelope

    |S_n(a)| = C * exp(-a / tau_L) * exp(-(a / tau_G)**2)

vs the pure exponential ``C * exp(-a / tau_L)`` the production calibration
assumes. If Voigt is the right physical shape, fitting it on per-bin time
series should:

1. Reduce the per-bin chi-squared meaningfully relative to pure-exp.
2. Recover a per-band ``tau_G`` consistent across bins (the horn-coupling
   geometry should produce a 1/f trend across the band).
3. Yield a per-band median ``tau_G`` that Part A can consume as a soft prior
   on the per-window joint ``(tau_L, tau_G)`` LSQ.

Two bin pools tested:

* **Bad-fit hi-SNR bins** (the original planning target: STFT classification
  ``2`` AND per-bin SNR > 100). The planning hypothesis was that the
  pure-exp gate fires *because* these bins are Voigt-shaped. On 2638 this
  hypothesis is **contradicted** -- the high-SNR bad-fit bins are dominated
  by line-blend interference (non-monotonic |S_n(a)|; two close lines
  beating inside one bin's resolution) and no monotonic Voigt fits them
  either. Reported for completeness.
* **Strong contributor bins** (STFT classification ``3`` AND per-bin
  SNR > ``SNR_CONTRIB_GATE``). These pass the pure-exp gate but a joint
  Voigt fit still reduces per-bin chi-squared by 2-5x on the strongest
  bins, with recovered ``tau_G`` clustering in a physically sensible
  range. The per-band ``tau_G`` calibration artefact comes from this pool.

The per-band ``tau_G`` calibration produced by this script is the artefact
Part A consumes as a soft prior on the joint ``(tau_L, tau_G)`` LSQ.

Run from the repository root:

    conda run -n ftmwpipeline-dev python \
        dev-docs/research/voigt-deficit/part_b_perbin.py

Outputs:
- ``data/part_b_perbin_summary.json``   -- per-band stats + global diagnostics
                                            for both pools
- ``data/tau_G_band_majorities.json``   -- (label, freq_lo, freq_hi, tau_G,
                                            sigma_tau_G, n) from the
                                            **improved-Voigt-fit subset** of
                                            the contributor pool -- Part A
                                            prior input
- ``data/part_b_perbin_detail.csv``     -- per-bin records (both pools)
- ``figures/01_voigt_vs_exp_chi2.png``  -- chi2r scatter: pure-exp vs Voigt,
                                            one panel per pool
- ``figures/02_tauG_vs_freq.png``       -- recovered tau_G vs frequency on
                                            the contributor pool, with per-
                                            band medians overlaid
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from ftmwpipeline._internal.stage0_impl import load_fid_from_pipeline_impl
from ftmwpipeline.fitting.tau_calibration import (
    DEFAULT_N_SEG,
    DEFAULT_RSS_GATE_FACTOR,
    DEFAULT_T_SIGMA,
    stft_calibration,
)

HERE = Path(__file__).parent
FIG = HERE / "figures"
DATA = HERE / "data"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

SCRATCH = Path("scratch/stage5-tau-calibration-lsq")
SOURCE_FIXTURE = SCRATCH / "exp_2638_unapodized.ftmw"

# Pool-1 (bad-fit) SNR threshold matches the original planning-doc target.
SNR_BADFIT_GATE = 100.0
# Pool-2 (contributor) SNR threshold: max contributor SNR on 2638 is ~82, so
# 20 admits ~400 bins (p90 ≈ 19) -- enough statistics, and high enough that
# the chi-squared discriminator between Voigt and pure-exp has signal/noise.
SNR_CONTRIB_GATE = 20.0
TRIM_LO_MHZ = 26500.0
TRIM_HI_MHZ = 40000.0

TAU_BOUND_LO = 0.5
TAU_BOUND_HI = 100.0
# Multi-start grid for tau_G seeds. A bin that is pure-Lorentzian lands at
# the upper edge; one with finite Gaussian content lands at the right basin.
VOIGT_TAU_G_SEEDS = (100.0, 50.0, 20.0, 10.0, 5.0, 3.0)

# Calibration-pool gate: only feed bins into the per-band tau_G calibration
# when (a) Voigt actually beats pure-exp by at least this many chi^2_r units
# and (b) tau_G is finite and away from the upper bound (so the bin carries
# meaningful Gaussian content).
CAL_DELTA_CHI2R_MIN = 1.0
CAL_TAU_G_UPPER_FRACTION = 0.7  # tau_G < 0.7 * TAU_BOUND_HI

logger = logging.getLogger("voigt-deficit-part-b")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


@dataclass
class BinFit:
    pool: str  # "bad_fit" or "contributor"
    bin_index: int
    freq_mhz: float
    snr: float
    sigma_frame: float
    # Pure-exp NLS polish
    C_exp: float
    tau_L_exp: float
    rss_exp: float
    chi2r_exp: float
    # Voigt (multi-start)
    C_voigt: float
    tau_L_voigt: float
    tau_G_voigt: float
    rss_voigt: float
    chi2r_voigt: float
    voigt_converged: bool
    cond_number_voigt: float
    seed_tau_G_best: float
    delta_chi2r: float  # chi2r_exp - chi2r_voigt


def _exp_residuals(params: np.ndarray, a: np.ndarray, y: np.ndarray) -> np.ndarray:
    C, tau_L = params
    return C * np.exp(-a / tau_L) - y


def _voigt_residuals(
    params: np.ndarray, a: np.ndarray, y: np.ndarray
) -> np.ndarray:
    C, tau_L, tau_G = params
    return C * np.exp(-a / tau_L) * np.exp(-((a / tau_G) ** 2)) - y


def _fit_exp_nls(
    a_centers: np.ndarray, mag: np.ndarray, C0: float, tau0: float
) -> tuple[np.ndarray, float, bool]:
    """Polish the log-linear exp seed with NLS. ``params = (C, tau_L)``."""
    C0 = max(C0, 1e-30)
    tau0 = float(np.clip(tau0, TAU_BOUND_LO * 1.05, TAU_BOUND_HI * 0.95))
    res = least_squares(
        _exp_residuals,
        x0=np.array([C0, tau0]),
        bounds=([0.0, TAU_BOUND_LO], [np.inf, TAU_BOUND_HI]),
        args=(a_centers, mag),
        method="trf",
        max_nfev=200,
    )
    rss = float(np.sum(res.fun ** 2))
    return res.x, rss, bool(res.success)


def _fit_voigt_nls_one(
    a_centers: np.ndarray,
    mag: np.ndarray,
    C0: float,
    tau_L0: float,
    tau_G0: float,
) -> tuple[np.ndarray, float, bool, float]:
    """Single-start Voigt fit. Returns ``(params, rss, ok, cond)``.

    ``params = (C, tau_L, tau_G)``; ``cond`` is the squared singular-value
    ratio of the Jacobian at the solution (``inf`` for rank-deficient).
    """
    C0 = max(C0, 1e-30)
    tau_L0 = float(np.clip(tau_L0, TAU_BOUND_LO * 1.05, TAU_BOUND_HI * 0.95))
    tau_G0 = float(np.clip(tau_G0, TAU_BOUND_LO * 1.05, TAU_BOUND_HI * 0.95))
    res = least_squares(
        _voigt_residuals,
        x0=np.array([C0, tau_L0, tau_G0]),
        bounds=([0.0, TAU_BOUND_LO, TAU_BOUND_LO],
                [np.inf, TAU_BOUND_HI, TAU_BOUND_HI]),
        args=(a_centers, mag),
        method="trf",
        max_nfev=400,
    )
    rss = float(np.sum(res.fun ** 2))
    J = res.jac
    try:
        s = np.linalg.svd(J, compute_uv=False)
        cond = float((s.max() / s.min()) ** 2) if s.min() > 0 else float("inf")
    except np.linalg.LinAlgError:
        cond = float("inf")
    return res.x, rss, bool(res.success), cond


def _fit_voigt_multistart(
    a_centers: np.ndarray,
    mag: np.ndarray,
    C0: float,
    tau_L0: float,
) -> tuple[np.ndarray, float, bool, float, float]:
    """Multi-start Voigt fit; returns the best-RSS basin.

    Returns ``(params, rss, ok, cond, seed_tau_G_best)``.
    """
    best_rss = np.inf
    best: Optional[tuple[np.ndarray, float, bool, float, float]] = None
    for tG0 in VOIGT_TAU_G_SEEDS:
        try:
            params, rss, ok, cond = _fit_voigt_nls_one(
                a_centers, mag, C0, tau_L0, tG0
            )
        except Exception:  # noqa: BLE001
            continue
        if rss < best_rss:
            best_rss = rss
            best = (params, rss, ok, cond, tG0)
    if best is None:
        # Pathological seed grid -- return a sentinel.
        return (
            np.array([C0, tau_L0, TAU_BOUND_HI * 0.95]),
            float("inf"),
            False,
            float("inf"),
            float("nan"),
        )
    return best


def _fit_pool(
    pool_name: str,
    bin_indices: np.ndarray,
    cal,
    freq_mol_mhz: np.ndarray,
    sigma_frame: float,
    n_seg: int,
) -> List[BinFit]:
    rows: List[BinFit] = []
    a = cal.a_centers_us.astype(float)
    for idx in bin_indices:
        mag_bin = cal.mag[:, idx].astype(float)
        snr = float(cal.snr_per_bin[idx])
        freq = float(freq_mol_mhz[idx])

        C_seed = float(cal.C_per_bin[idx])
        tau_seed = float(cal.tau_per_bin[idx])

        exp_params, rss_exp, _ok_exp = _fit_exp_nls(a, mag_bin, C_seed, tau_seed)
        v_params, rss_v, ok_v, cond_v, seed_tG = _fit_voigt_multistart(
            a, mag_bin, exp_params[0], exp_params[1]
        )

        dof_exp = max(n_seg - 2, 1)
        dof_voigt = max(n_seg - 3, 1)
        chi2r_exp = rss_exp / (sigma_frame ** 2) / dof_exp
        chi2r_voigt = rss_v / (sigma_frame ** 2) / dof_voigt

        rows.append(BinFit(
            pool=pool_name,
            bin_index=int(idx),
            freq_mhz=freq,
            snr=snr,
            sigma_frame=sigma_frame,
            C_exp=float(exp_params[0]),
            tau_L_exp=float(exp_params[1]),
            rss_exp=float(rss_exp),
            chi2r_exp=float(chi2r_exp),
            C_voigt=float(v_params[0]),
            tau_L_voigt=float(v_params[1]),
            tau_G_voigt=float(v_params[2]),
            rss_voigt=float(rss_v),
            chi2r_voigt=float(chi2r_voigt),
            voigt_converged=ok_v,
            cond_number_voigt=cond_v,
            seed_tau_G_best=float(seed_tG),
            delta_chi2r=float(chi2r_exp - chi2r_voigt),
        ))
    return rows


def _per_band_calibration(
    rows: list[BinFit],
    *,
    bands: tuple[tuple[str, float, float], ...],
    delta_chi2r_min: float,
    tau_G_upper: float,
) -> list[dict]:
    """Per-band tau_G stats from the *calibration-eligible* bin subset.

    A row is calibration-eligible iff Voigt converged, finite tau_G < ``tau_G_upper``,
    AND Voigt beats pure-exp by at least ``delta_chi2r_min`` chi^2_r units.
    Bins where Voigt does not meaningfully improve the fit get filtered out
    so their (often saturated-at-upper-bound) tau_G doesn't pull the median.
    """
    out: list[dict] = []
    for label, lo, hi in bands:
        sub = [
            r for r in rows
            if lo <= r.freq_mhz < hi
            and r.voigt_converged
            and np.isfinite(r.tau_G_voigt)
            and r.tau_G_voigt < tau_G_upper
            and r.delta_chi2r >= delta_chi2r_min
        ]
        if not sub:
            out.append({
                "label": label,
                "freq_lo_mhz": float(lo),
                "freq_hi_mhz": float(hi),
                "n": 0,
                "tau_G_median_us": float("nan"),
                "tau_G_iqr_us": float("nan"),
                "tau_G_sigma_robust_us": float("nan"),
                "tau_L_median_us": float("nan"),
                "chi2r_voigt_median": float("nan"),
                "chi2r_exp_median": float("nan"),
                "delta_chi2r_median": float("nan"),
            })
            continue
        tau_G = np.array([r.tau_G_voigt for r in sub])
        tau_L = np.array([r.tau_L_voigt for r in sub])
        ch_v = np.array([r.chi2r_voigt for r in sub])
        ch_e = np.array([r.chi2r_exp for r in sub])
        dch = np.array([r.delta_chi2r for r in sub])
        q75, q25 = np.percentile(tau_G, [75, 25])
        iqr = float(q75 - q25)
        sigma_robust = iqr / 1.349
        out.append({
            "label": label,
            "freq_lo_mhz": float(lo),
            "freq_hi_mhz": float(hi),
            "n": int(len(sub)),
            "tau_G_median_us": float(np.median(tau_G)),
            "tau_G_iqr_us": iqr,
            "tau_G_sigma_robust_us": sigma_robust,
            "tau_L_median_us": float(np.median(tau_L)),
            "chi2r_voigt_median": float(np.median(ch_v)),
            "chi2r_exp_median": float(np.median(ch_e)),
            "delta_chi2r_median": float(np.median(dch)),
        })
    return out


def _pool_summary(rows: list[BinFit]) -> dict:
    if not rows:
        return {"n": 0}
    ch_e = np.array([r.chi2r_exp for r in rows])
    ch_v = np.array([r.chi2r_voigt for r in rows])
    dch = np.array([r.delta_chi2r for r in rows])
    n_improved = int(np.sum(dch > 0))
    n_meaningfully_improved = int(np.sum(dch >= CAL_DELTA_CHI2R_MIN))
    return {
        "n": int(len(rows)),
        "chi2r_exp_median": float(np.median(ch_e)),
        "chi2r_voigt_median": float(np.median(ch_v)),
        "improvement_factor_median": float(
            np.median(ch_e) / max(np.median(ch_v), 1e-30)
        ),
        "n_voigt_improves": n_improved,
        "n_voigt_improves_by_at_least_1": n_meaningfully_improved,
        "fraction_voigt_improves": float(n_improved) / len(rows),
    }


def main() -> None:
    if not SOURCE_FIXTURE.exists():
        raise FileNotFoundError(
            f"Source fixture missing: {SOURCE_FIXTURE}. "
            "Re-run dev-docs/research/stage5-tau-calibration setup first."
        )

    logger.info("Loading FID from %s", SOURCE_FIXTURE)
    fid = load_fid_from_pipeline_impl(str(SOURCE_FIXTURE))
    sample_dt_us = float(fid.spacing * 1e6)

    with h5py.File(SOURCE_FIXTURE, "r") as h5f:
        params = json.loads(h5f["stage2b_tau_calibration"].attrs["parameters_used"])
        sigma_x_full_persisted = float(
            h5f["stage2b_tau_calibration/scalars"].attrs["sigma_x_full"]
        )
    start_us = float(params["start_us"])
    end_us = float(params["end_us"])
    n_seg = int(params.get("n_seg", DEFAULT_N_SEG))
    t_sigma = float(params.get("t_sigma", DEFAULT_T_SIGMA))
    rss_gate_factor = float(params.get("rss_gate_factor", DEFAULT_RSS_GATE_FACTOR))
    trim_lo_mhz = float(params.get("trim_lo_mhz", TRIM_LO_MHZ))
    trim_hi_mhz = float(params.get("trim_hi_mhz", TRIM_HI_MHZ))
    sideband = (
        fid.sideband.value if hasattr(fid.sideband, "value") else str(fid.sideband)
    ).lower()
    probe_freq_mhz = float(fid.probe_freq_mhz)

    fid_arr = np.asarray(fid.data, dtype=float)
    start_idx = max(int(round(start_us / sample_dt_us)), 0)
    end_idx = min(int(round(end_us / sample_dt_us)), fid_arr.size)
    active = fid_arr[start_idx:end_idx]
    new_size = (active.size // n_seg) * n_seg
    active = active[:new_size]

    logger.info(
        "Active region: %.3f-%.3f us, %d samples, n_seg=%d (Nw=%d, T_w=%.3f us)",
        start_us, end_us, active.size, n_seg, active.size // n_seg,
        (active.size // n_seg) * sample_dt_us,
    )

    cal = stft_calibration(
        active, sample_dt_us, sigma_time=1.0,  # placeholder; sigma_x_full overrides
        n_seg=n_seg,
        t_sigma=t_sigma,
        rss_gate_factor=rss_gate_factor,
        sigma_x_full=sigma_x_full_persisted,
    )
    sigma_frame = float(cal.sigma_frame)
    logger.info(
        "STFT: sigma_x_full=%.3e, sigma_frame=%.3e, tau_max=%.2f us",
        cal.sigma_x_full, sigma_frame, cal.tau_max_us,
    )

    sign = -1.0 if sideband == "lower" else 1.0
    freq_mol_mhz = probe_freq_mhz + sign * cal.freq_bb_mhz
    in_trim = (freq_mol_mhz >= trim_lo_mhz) & (freq_mol_mhz <= trim_hi_mhz)

    # Pool 1: bad-fit hi-SNR bins (the planning-doc target).
    bad_mask = (cal.classification == 2) & in_trim & (cal.snr_per_bin > SNR_BADFIT_GATE)
    bad_bins = np.where(bad_mask)[0]
    bad_bins = bad_bins[np.argsort(freq_mol_mhz[bad_bins])]
    logger.info(
        "Pool 1 (bad-fit, SNR > %.0f, in trim): %d bins",
        SNR_BADFIT_GATE, bad_bins.size,
    )

    # Pool 2: strong contributor bins (the pivot).
    contrib_mask = (
        (cal.classification == 3) & in_trim & (cal.snr_per_bin > SNR_CONTRIB_GATE)
    )
    contrib_bins = np.where(contrib_mask)[0]
    contrib_bins = contrib_bins[np.argsort(freq_mol_mhz[contrib_bins])]
    logger.info(
        "Pool 2 (contributors, SNR > %.0f, in trim): %d bins",
        SNR_CONTRIB_GATE, contrib_bins.size,
    )

    rows_bad = _fit_pool("bad_fit", bad_bins, cal, freq_mol_mhz, sigma_frame, n_seg)
    rows_contrib = _fit_pool(
        "contributor", contrib_bins, cal, freq_mol_mhz, sigma_frame, n_seg,
    )
    all_rows = rows_bad + rows_contrib
    logger.info(
        "Voigt fits done. Bad-fit pool: %d, contributor pool: %d.",
        len(rows_bad), len(rows_contrib),
    )

    # Bands: arithmetic-third partition matching Stage 2b on this fixture.
    step = (trim_hi_mhz - trim_lo_mhz) / 3.0
    bands = (
        ("low",  trim_lo_mhz,             trim_lo_mhz + step),
        ("mid",  trim_lo_mhz + step,      trim_lo_mhz + 2 * step),
        ("high", trim_lo_mhz + 2 * step,  trim_hi_mhz),
    )

    band_stats_contrib = _per_band_calibration(
        rows_contrib,
        bands=bands,
        delta_chi2r_min=CAL_DELTA_CHI2R_MIN,
        tau_G_upper=CAL_TAU_G_UPPER_FRACTION * TAU_BOUND_HI,
    )
    band_stats_bad = _per_band_calibration(
        rows_bad,
        bands=bands,
        delta_chi2r_min=CAL_DELTA_CHI2R_MIN,
        tau_G_upper=CAL_TAU_G_UPPER_FRACTION * TAU_BOUND_HI,
    )

    summary = {
        "fixture": str(SOURCE_FIXTURE),
        "snr_gate_bad_fit": SNR_BADFIT_GATE,
        "snr_gate_contributor": SNR_CONTRIB_GATE,
        "calibration_delta_chi2r_min": CAL_DELTA_CHI2R_MIN,
        "calibration_tau_G_upper_us": CAL_TAU_G_UPPER_FRACTION * TAU_BOUND_HI,
        "sigma_frame": sigma_frame,
        "sigma_x_full": float(cal.sigma_x_full),
        "n_seg": int(n_seg),
        "pool_summary": {
            "bad_fit": _pool_summary(rows_bad),
            "contributor": _pool_summary(rows_contrib),
        },
        "bands_contributor": band_stats_contrib,
        "bands_bad_fit": band_stats_bad,
    }
    summary_path = DATA / "part_b_perbin_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s", summary_path)

    # Per-band tau_G calibration artifact from the contributor pool.
    tau_G_band_majorities = {
        "fixture": str(SOURCE_FIXTURE),
        "source_pool": "contributor",
        "snr_gate": SNR_CONTRIB_GATE,
        "calibration_delta_chi2r_min": CAL_DELTA_CHI2R_MIN,
        "calibration_tau_G_upper_us": CAL_TAU_G_UPPER_FRACTION * TAU_BOUND_HI,
        "trim_lo_mhz": trim_lo_mhz,
        "trim_hi_mhz": trim_hi_mhz,
        "bands": [
            {
                "label": b["label"],
                "freq_lo_mhz": b["freq_lo_mhz"],
                "freq_hi_mhz": b["freq_hi_mhz"],
                "n_bins": b["n"],
                "tau_G_us": b["tau_G_median_us"],
                "sigma_tau_G_us": b["tau_G_sigma_robust_us"],
            }
            for b in band_stats_contrib
        ],
    }
    tauG_path = DATA / "tau_G_band_majorities.json"
    tauG_path.write_text(json.dumps(tau_G_band_majorities, indent=2))
    logger.info("Wrote %s", tauG_path)

    detail_path = DATA / "part_b_perbin_detail.csv"
    with detail_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "pool", "bin_index", "freq_mhz", "snr",
            "tau_L_exp_us", "chi2r_exp",
            "tau_L_voigt_us", "tau_G_voigt_us", "chi2r_voigt", "delta_chi2r",
            "voigt_converged", "cond_number_voigt", "seed_tau_G_best",
        ])
        for r in all_rows:
            w.writerow([
                r.pool, r.bin_index, f"{r.freq_mhz:.6f}", f"{r.snr:.2f}",
                f"{r.tau_L_exp:.4f}", f"{r.chi2r_exp:.4f}",
                f"{r.tau_L_voigt:.4f}", f"{r.tau_G_voigt:.4f}",
                f"{r.chi2r_voigt:.4f}", f"{r.delta_chi2r:.4f}",
                int(r.voigt_converged), f"{r.cond_number_voigt:.2e}",
                f"{r.seed_tau_G_best:.2f}",
            ])
    logger.info("Wrote %s", detail_path)

    # Figure 1: chi2r scatter, pure-exp vs Voigt, one panel per pool.
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.0))
    for ax, rows, title in (
        (axes[0], rows_bad,
         f"Bad-fit pool (SNR > {SNR_BADFIT_GATE:.0f}, N={len(rows_bad)})"),
        (axes[1], rows_contrib,
         f"Contributor pool (SNR > {SNR_CONTRIB_GATE:.0f}, N={len(rows_contrib)})"),
    ):
        if not rows:
            ax.set_title(title + "  -- (empty)")
            continue
        ch_e = np.array([r.chi2r_exp for r in rows])
        ch_v = np.array([r.chi2r_voigt for r in rows])
        ax.scatter(ch_e, ch_v, s=10, alpha=0.55, c="tab:blue", edgecolor="none")
        lim_lo = 0.3
        lim_hi = max(ch_e.max(), ch_v.max(), 10.0) * 1.1
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=0.7,
                label="Voigt = pure-exp")
        ax.axhline(1.0, color="tab:green", lw=0.7, alpha=0.7,
                   label=r"noise floor $\chi^2_r = 1$")
        ax.axhline(2.0, color="tab:orange", lw=0.7, alpha=0.7,
                   label=r"acceptance gate $\chi^2_r = 2$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_xlabel(r"pure-exp $\chi^2_r$")
        ax.set_ylabel(r"Voigt $\chi^2_r$")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Voigt-deficit Part B: per-bin Voigt vs pure-exp $\\chi^2_r$ on 2638",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out1 = FIG / "01_voigt_vs_exp_chi2.png"
    fig.savefig(out1, dpi=140)
    plt.close(fig)
    logger.info("Wrote %s", out1)

    # Figure 2: tau_G vs molecular frequency on the contributor pool.
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    cap = CAL_TAU_G_UPPER_FRACTION * TAU_BOUND_HI
    clean = [
        r for r in rows_contrib
        if r.voigt_converged
        and np.isfinite(r.tau_G_voigt)
        and r.tau_G_voigt < cap
    ]
    improved = [r for r in clean if r.delta_chi2r >= CAL_DELTA_CHI2R_MIN]
    saturated = [
        r for r in rows_contrib
        if r.voigt_converged
        and np.isfinite(r.tau_G_voigt)
        and r.tau_G_voigt >= cap
    ]
    if saturated:
        ax.scatter(
            [r.freq_mhz for r in saturated],
            [r.tau_G_voigt for r in saturated],
            s=10, c="#cccccc", alpha=0.4, edgecolor="none",
            label=fr"$\tau_G \geq {cap:.0f}$ us (no Voigt content)",
        )
    if clean:
        ax.scatter(
            [r.freq_mhz for r in clean],
            [r.tau_G_voigt for r in clean],
            c=[np.log10(max(r.snr, 1.0)) for r in clean],
            cmap="viridis", s=18, alpha=0.7, edgecolor="none",
            label=r"$\tau_G < {:.0f}$ us (Voigt-shaped)".format(cap),
        )
    if improved:
        sc = ax.scatter(
            [r.freq_mhz for r in improved],
            [r.tau_G_voigt for r in improved],
            c="none", s=42, edgecolor="black", linewidth=0.5,
            label=fr"$\Delta\chi^2_r \geq {CAL_DELTA_CHI2R_MIN}$ "
                  fr"(calibration eligible, N={len(improved)})",
        )
    band_colors = ("tab:orange", "tab:green", "tab:red")
    for b, color in zip(band_stats_contrib, band_colors):
        med = b["tau_G_median_us"]
        n = b["n"]
        if np.isfinite(med):
            ax.hlines(
                med, b["freq_lo_mhz"], b["freq_hi_mhz"],
                color=color, lw=2.5,
                label=f"{b['label']}: median={med:.2f} us (N={n})",
            )
    ax.set_xlabel("molecular frequency (MHz)")
    ax.set_ylabel(r"$\tau_G$ (us)")
    ax.set_title(
        "Voigt-deficit Part B: per-bin $\\tau_G$ across the band "
        "(2638 contributor pool)"
    )
    ax.set_xlim(trim_lo_mhz, trim_hi_mhz)
    ax.set_ylim(0.0, TAU_BOUND_HI)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out2 = FIG / "02_tauG_vs_freq.png"
    fig.savefig(out2, dpi=140)
    plt.close(fig)
    logger.info("Wrote %s", out2)

    # Console summary.
    print()
    print("====== Voigt-deficit Part B (2638) ======")
    pb = summary["pool_summary"]["bad_fit"]
    pc = summary["pool_summary"]["contributor"]
    print(f"  sigma_frame:                        {sigma_frame:.3e}")
    print()
    print("  Pool 1 -- bad-fit hi-SNR bins (planning-doc target):")
    if pb["n"] > 0:
        print(f"    N:                                 {pb['n']}")
        print(f"    chi2r_exp   median:                {pb['chi2r_exp_median']:>8.2f}")
        print(f"    chi2r_voigt median:                {pb['chi2r_voigt_median']:>8.2f}")
        print(f"    improvement factor (median):       {pb['improvement_factor_median']:.2f}x")
        print(f"    fraction with Voigt > pure-exp:    {pb['fraction_voigt_improves']:.0%}")
    else:
        print("    N: 0 (no qualifying bins)")
    print()
    print("  Pool 2 -- strong contributor bins (the pivot):")
    print(f"    N:                                 {pc['n']}")
    print(f"    chi2r_exp   median:                {pc['chi2r_exp_median']:>8.2f}")
    print(f"    chi2r_voigt median:                {pc['chi2r_voigt_median']:>8.2f}")
    print(f"    improvement factor (median):       {pc['improvement_factor_median']:.2f}x")
    print(f"    fraction with Voigt > pure-exp:    {pc['fraction_voigt_improves']:.0%}")
    print(f"    N with Δchi2r >= "
          f"{CAL_DELTA_CHI2R_MIN}: {pc['n_voigt_improves_by_at_least_1']}")
    print()
    print("  Per-band tau_G (contributor pool, calibration-eligible subset):")
    print(f"    {'band':<6} {'N':>5} {'tau_G med (us)':>14} "
          f"{'sigma (us)':>11} {'rel scatter':>11} "
          f"{'tau_L med (us)':>14} {'Δchi2r med':>11}")
    for b in band_stats_contrib:
        if b["n"] > 0 and np.isfinite(b["tau_G_median_us"]):
            rel = b["tau_G_sigma_robust_us"] / max(b["tau_G_median_us"], 1e-30)
            print(f"    {b['label']:<6} {b['n']:>5d} "
                  f"{b['tau_G_median_us']:>14.2f} "
                  f"{b['tau_G_sigma_robust_us']:>11.2f} "
                  f"{rel:>11.2%} "
                  f"{b['tau_L_median_us']:>14.2f} "
                  f"{b['delta_chi2r_median']:>11.2f}")
        else:
            print(f"    {b['label']:<6} {b['n']:>5d}      (no calibration-eligible bins)")
    print("==========================================")


if __name__ == "__main__":
    main()
