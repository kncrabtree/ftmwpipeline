"""Polish refinement sweep: SNR-cap × bad-fit gate cross-validation.

Calibrates the production ``polish_snr_cap`` default by sweeping the
cap against the LSQ-fit-and-histogram per-band reference on the
unapodized 2638 fixture. Apply the Gauss-Newton polish only to
contributors whose per-bin SNR is below a cap, leaving high-SNR
contributors on their (already near-unbiased) log-linear seed.

Acceptance: a cap exists where all three arithmetic-third LSQ medians
(low 7.87, mid 6.27, high 5.16 µs from
[`lsq_comparison.py`](lsq_comparison.py)) land within 5 % of the
corresponding STFT polished per-third median.

Surprise finding: on 2638 the per-bin contributor SNR maxes at ~82
with a median ~8 — the report's predicted cap range (100-200) sits
above the entire distribution, and the cap range that actually does
gate bins (5-30) is too narrow to differentially correct the per-band
bias. The narrow contributor SNR range is itself an artefact of the
**bad-fit classification gate** in `stft_calibration`: strong on-line
bins (per-frame SNR 240-360) are excluded because their `rss_exp`
exceeds the relative gate `5·n_seg·(0.05·mean|S|)²` by ~10-20× — the
single-exponential model leaves more residual than the gate allows
because real molecular lines aren't pure single-exponentials (line
shape, Doppler, saturation). The strong bins are nevertheless
correctly identified as exponential by AICc.

So this script sweeps BOTH axes: the bad-fit gate
``relative_gate_fraction`` (currently 0.05) and ``polish_snr_cap``.
The clean operating point on 2638 is:

    relative_gate_fraction = 0.30 (vs current 0.05)
    polish = True
    polish_snr_cap = 10

which lands all three per-band SNR-weighted majorities within ±2.8 %
of the LSQ reference (low -0.0 %, mid +2.8 %, high -2.8 %), vs ±8-10 %
for the current production default or pure polish=False.

Run from the repository root:

    conda run -n ftmwpipeline-dev python \
        dev-docs/research/stage5-tau-calibration/polish_snr_cap_validation.py

Outputs:

- ``data/polish_snr_cap.json`` — full cross-sweep table (caps × gates).
- ``figures/14_polish_snr_cap.png`` — worst-case Δ heatmap.

Assumes the unapodized fixture at
``scratch/stage5-tau-calibration-lsq/exp_2638_unapodized.ftmw`` has
Stages 0-1 populated. The script only calls extract_tau_majority
directly; it does not need Stage 2-5 outputs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ftmwpipeline.api as ftmw
from ftmwpipeline.fitting.tau_calibration import extract_tau_majority
from ftmwpipeline._internal.stage0_impl import load_fid_from_pipeline_impl

HERE = Path(__file__).parent
FIG = HERE / "figures"
DATA = HERE / "data"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

FIXTURE = Path("scratch/stage5-tau-calibration-lsq/exp_2638_unapodized.ftmw")

TRIM_LO_MHZ = 26500.0
TRIM_HI_MHZ = 40000.0

ARITHMETIC_THIRDS = [
    ("low",  TRIM_LO_MHZ, TRIM_LO_MHZ + (TRIM_HI_MHZ - TRIM_LO_MHZ) / 3.0),
    ("mid",  TRIM_LO_MHZ + (TRIM_HI_MHZ - TRIM_LO_MHZ) / 3.0,
             TRIM_LO_MHZ + 2.0 * (TRIM_HI_MHZ - TRIM_LO_MHZ) / 3.0),
    ("high", TRIM_LO_MHZ + 2.0 * (TRIM_HI_MHZ - TRIM_LO_MHZ) / 3.0, TRIM_HI_MHZ),
]

# Phase 4 LSQ expanded medians (lsq_comparison.py output on the same fixture):
# low 7.87 µs (N=27), mid 6.27 µs (N=33), high 5.16 µs (N=30).
LSQ_REFERENCE = {"low": 7.87, "mid": 6.27, "high": 5.16}
LSQ_BAND_WIDE = 6.26  # band-wide LSQ expanded mean from lsq_comparison.py

# Bad-fit gate (relative branch): rss_exp > rss_gate_factor · n_seg ·
# (relative_gate_fraction · mean|S|)². Shipping default is 0.05. Above
# 0.30 the contributor set saturates on this fixture.
GATE_SWEEP = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

# Polish SNR cap: polish runs only on contributors with snr_per_bin < cap.
# None = polish all contributors (legacy polish=True behaviour).
CAP_SWEEP: list[Optional[float]] = [None, 30.0, 20.0, 15.0, 12.0, 11.0, 10.0, 9.0, 8.0]

# 5 % per-third acceptance gate.
PER_THIRD_TOLERANCE = 0.05

logger = logging.getLogger("polish-snr-cap")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _load_fid_inputs() -> tuple[np.ndarray, float, float, float, float, str]:
    fid = load_fid_from_pipeline_impl(str(FIXTURE))
    sample_dt_us = float(fid.spacing * 1e6)
    sideband = (
        fid.sideband.value if hasattr(fid.sideband, "value") else str(fid.sideband)
    )
    ft = ftmw.compute_ft(str(FIXTURE))
    pp = ft.metadata["processing_params"]
    start_us = float(pp.start_us)
    end_us = float(pp.end_us)
    probe_mhz = float(fid.probe_freq_mhz)
    return np.asarray(fid.data, dtype=float), sample_dt_us, start_us, end_us, probe_mhz, sideband


def _per_third(
    freqs: np.ndarray, taus: np.ndarray,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, lo, hi in ARITHMETIC_THIRDS:
        mask = (freqs >= lo) & (freqs < hi)
        s = taus[mask]
        out[name] = {
            "n": int(s.size),
            "median_us": float(np.median(s)) if s.size else float("nan"),
            "iqr_us": (
                float(np.percentile(s, 75) - np.percentile(s, 25))
                if s.size else float("nan")
            ),
        }
    return out


def _run(
    fid: np.ndarray,
    sample_dt_us: float,
    start_us: float,
    end_us: float,
    probe_mhz: float,
    sideband: str,
    *,
    rgate: float,
    polish: bool,
    polish_snr_cap: Optional[float],
) -> dict:
    cal = extract_tau_majority(
        fid, sample_dt_us,
        start_us=start_us, end_us=end_us,
        probe_freq_mhz=probe_mhz, sideband=sideband,
        trim_lo_mhz=TRIM_LO_MHZ, trim_hi_mhz=TRIM_HI_MHZ,
        relative_gate_fraction=rgate,
        polish=polish, polish_snr_cap=polish_snr_cap,
        compute_band_majorities_flag=True,
    )
    freqs = np.asarray(cal.contributor_freqs_mhz, dtype=float)
    taus = np.asarray(cal.contributor_taus_us, dtype=float)
    thirds = _per_third(freqs, taus)
    band_majorities: dict[str, dict[str, float]] = {}
    for bm in cal.band_majorities:
        band_majorities[bm.label] = {
            "tau_us": float(bm.tau_maj_us),
            "sigma_us": float(bm.sigma_tau_us),
            "n": int(bm.n),
        }
    rels_median = {
        n: (thirds[n]["median_us"] - LSQ_REFERENCE[n]) / LSQ_REFERENCE[n]
        for n in LSQ_REFERENCE
    }
    rels_majority = {
        n: (band_majorities[n]["tau_us"] - LSQ_REFERENCE[n]) / LSQ_REFERENCE[n]
        for n in LSQ_REFERENCE
        if n in band_majorities
    }
    worst_median = max(abs(rels_median[n]) for n in LSQ_REFERENCE)
    worst_majority = (
        max(abs(rels_majority[n]) for n in LSQ_REFERENCE)
        if rels_majority else float("nan")
    )
    return {
        "rgate": rgate,
        "polish": polish,
        "polish_snr_cap": polish_snr_cap,
        "tau_maj_us": float(cal.tau_maj_us),
        "sigma_tau_us": float(cal.sigma_tau_us),
        "n_contributors": int(cal.n_contributors),
        "thirds": thirds,
        "band_majorities": band_majorities,
        "rels_median_pct": {n: float(v * 100) for n, v in rels_median.items()},
        "rels_majority_pct": {n: float(v * 100) for n, v in rels_majority.items()},
        "worst_abs_rel_median_pct": float(worst_median * 100),
        "worst_abs_rel_majority_pct": float(worst_majority * 100),
        "passes_5pct_majority": bool(worst_majority < PER_THIRD_TOLERANCE),
    }


def main() -> None:
    logger.info("Loading FID + Stage 1 settings from %s", FIXTURE)
    fid, sample_dt_us, start_us, end_us, probe_mhz, sideband = _load_fid_inputs()

    # Three baselines + the cap × gate cross-sweep.
    runs: list[dict] = []
    for rgate in GATE_SWEEP:
        # polish=False reference at each gate.
        legacy = _run(
            fid, sample_dt_us, start_us, end_us, probe_mhz, sideband,
            rgate=rgate, polish=False, polish_snr_cap=None,
        )
        legacy["label"] = f"polish=False rel={rgate:.2f}"
        runs.append(legacy)
        for cap in CAP_SWEEP:
            cell = _run(
                fid, sample_dt_us, start_us, end_us, probe_mhz, sideband,
                rgate=rgate, polish=True, polish_snr_cap=cap,
            )
            cap_s = "no cap" if cap is None else f"cap<{cap:g}"
            cell["label"] = f"polish=True rel={rgate:.2f} {cap_s}"
            runs.append(cell)

    # Pick best by SNR-weighted band-majority worst-case (the production
    # quantity Stage 5 routes on).
    valid = [r for r in runs if r["polish"] and np.isfinite(r["worst_abs_rel_majority_pct"])]
    best = min(valid, key=lambda r: r["worst_abs_rel_majority_pct"])

    headline = {
        "fixture": str(FIXTURE),
        "lsq_reference_per_third": LSQ_REFERENCE,
        "lsq_band_wide_us": LSQ_BAND_WIDE,
        "per_third_tolerance": PER_THIRD_TOLERANCE,
        "gate_sweep": GATE_SWEEP,
        "cap_sweep_inf_as_null": [None if c is None else float(c) for c in CAP_SWEEP],
        "runs": runs,
        "best_polish_run": {
            "label": best["label"],
            "rgate": best["rgate"],
            "polish_snr_cap": best["polish_snr_cap"],
            "worst_abs_rel_majority_pct": best["worst_abs_rel_majority_pct"],
            "passes_5pct_majority": best["passes_5pct_majority"],
            "band_majorities": best["band_majorities"],
            "rels_majority_pct": best["rels_majority_pct"],
        },
    }
    out_json = DATA / "polish_snr_cap.json"
    out_json.write_text(json.dumps(headline, indent=2))
    logger.info("Wrote %s", out_json)

    # Heatmap of worst-case Δ on per-band SNR-weighted majority.
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    polish_runs = [r for r in runs if r["polish"]]
    cap_axis = [c if c is not None else np.inf for c in CAP_SWEEP]
    grid = np.full((len(GATE_SWEEP), len(CAP_SWEEP)), np.nan)
    for r in polish_runs:
        i = GATE_SWEEP.index(r["rgate"])
        cap_val = np.inf if r["polish_snr_cap"] is None else float(r["polish_snr_cap"])
        j = cap_axis.index(cap_val)
        grid[i, j] = r["worst_abs_rel_majority_pct"]

    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=12)
    ax.set_xticks(range(len(CAP_SWEEP)))
    ax.set_xticklabels(["∞" if c is None else f"{c:g}" for c in CAP_SWEEP])
    ax.set_yticks(range(len(GATE_SWEEP)))
    ax.set_yticklabels([f"{g:.2f}" for g in GATE_SWEEP])
    ax.set_xlabel("polish_snr_cap (∞ = no cap)")
    ax.set_ylabel("relative_gate_fraction (bad-fit gate; current default 0.05)")
    ax.set_title(
        "Worst-case |Δ| on per-band SNR-weighted majority τ (vs LSQ, 2638)\n"
        "green = ≤5% on all three bands; red = ≥10% on at least one"
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("worst |Δ| (%) across low/mid/high majority τ")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        color="black" if v < 6 else "white", fontsize=8)
    # Mark best.
    i_best = GATE_SWEEP.index(best["rgate"])
    cap_best = np.inf if best["polish_snr_cap"] is None else float(best["polish_snr_cap"])
    j_best = cap_axis.index(cap_best)
    ax.scatter([j_best], [i_best], marker="*", s=180, edgecolor="black",
               facecolor="white", zorder=5, label=f"best: {best['label']}")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig_path = FIG / "14_polish_snr_cap.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    logger.info("Wrote %s", fig_path)

    # Stdout summary.
    print()
    print("===== polish_snr_cap × relative_gate_fraction sweep on 2638 (unapodized) =====")
    print(
        f"  LSQ ref per-third: low={LSQ_REFERENCE['low']:.2f}  "
        f"mid={LSQ_REFERENCE['mid']:.2f}  high={LSQ_REFERENCE['high']:.2f}  "
        f"band-wide={LSQ_BAND_WIDE:.2f}"
    )
    print(
        f"  Worst-case |Δ| on per-band SNR-weighted majority τ "
        f"(what Stage 5 routes on):"
    )
    print()
    header = (
        f"  {'label':<42} {'tau_maj':>8} {'n_ctr':>6} | "
        f"{'low maj':>9} {'Δ%':>6} | {'mid maj':>9} {'Δ%':>6} | "
        f"{'high maj':>10} {'Δ%':>6} | {'worst%':>7} {'pass5%':>6}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in runs:
        bm = r["band_majorities"]
        rm = r["rels_majority_pct"]
        print(
            f"  {r['label']:<42} {r['tau_maj_us']:>8.3f} {r['n_contributors']:>6d} | "
            f"{bm.get('low',{}).get('tau_us', float('nan')):>9.2f} "
            f"{rm.get('low', float('nan')):>+5.1f} | "
            f"{bm.get('mid',{}).get('tau_us', float('nan')):>9.2f} "
            f"{rm.get('mid', float('nan')):>+5.1f} | "
            f"{bm.get('high',{}).get('tau_us', float('nan')):>10.2f} "
            f"{rm.get('high', float('nan')):>+5.1f} | "
            f"{r['worst_abs_rel_majority_pct']:>6.1f} "
            f"{'YES' if r['passes_5pct_majority'] else 'no':>6}"
        )
    print()
    print(
        f"  Best polish=True operating point: {best['label']}  "
        f"worst |Δ| = {best['worst_abs_rel_majority_pct']:.1f} %  "
        f"-> acceptance: "
        f"{'PASS' if best['passes_5pct_majority'] else 'FAIL'}"
    )
    print("===============================================================================")


if __name__ == "__main__":
    main()
