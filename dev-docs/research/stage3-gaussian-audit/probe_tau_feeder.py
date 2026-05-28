"""Probe 1: Stage 3 gap-pass τ-feeder substitution impact.

Asks: when Stage 2b stamps ``recommended_shape='gaussian'``, does
swapping the gap-pass matched-filter τ from the pure-exp ``τ_maj``
to the Gaussian-twin ``τ_G_maj`` materially change what the gap pass
detects on 2638?

The Phase A τ-feeder fix in
``src/ftmwpipeline/_internal/stage3_impl.py`` already implements the
substitution. This probe quantifies what changed.

What it does
------------

Runs ``detect_peaks`` four times against the cached 2638
``stage3-gaussian-audit`` fixtures:

* L_anchor — Lorentzian shape, τ_maj feed (status quo on Lorentzian
  path; baseline).
* G_old_tau — Gaussian shape with the τ_maj feed forced (simulates
  pre-Phase-A behaviour on Gaussian-recommended data; forced by
  stamping ``recommended_shape='lorentzian'`` on the otherwise
  Gaussian-shape file).
* G_new_tau — Gaussian shape with the τ_G_maj feed (post-Phase-A
  behaviour). The default branch when ``recommended_shape='gaussian'``
  and the Gaussian twin is present.
* L_with_tau_G — Lorentzian shape forced to use τ_G_maj (the
  symmetric inverse of G_old_tau; isolates whether the τ value
  matters independent of which downstream shape consumes it).

For each variant the probe captures the per-pass peak counts
(primary / gap), the SNR distribution of detected peaks, the
classification breakdown, and the sidelobe-suspect proxy count.

Outputs
-------

* ``data/probe_tau_feeder.csv`` — one row per variant.
* ``figures/probe_tau_feeder.png`` — 3-panel comparison:
  per-pass peak counts, SNR p25/median/p75/p95 box, sidelobe proxy.
* Console summary with the four variants' headline counts.

Run from the repository root::

    conda run -n ftmwpipeline-dev python \\
        dev-docs/research/stage3-gaussian-audit/probe_tau_feeder.py
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from harness import (
    SCRATCH_DIR,
    Stage3RunResult,
    prepare_fixture,
    run_detect_peaks,
    summarise,
    write_csv,
)

from ftmwpipeline.io.stage_fit_settings_serialization import (
    write_stage2b_recommended_shape,
)

DATA = HERE / "data"
FIG = HERE / "figures"
DATA.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("probe-tau-feeder")


def _variant(
    base_fixture: Path,
    *,
    label: str,
    recommended_shape: str,
) -> Stage3RunResult:
    """Run a single τ-feeder variant.

    ``recommended_shape`` is what gets stamped on the working copy
    before detect_peaks runs; it drives the τ branch the feeder picks.
    ``label`` is the variant tag used in the CSV / figure / log output.
    """
    work_fp = SCRATCH_DIR / "runs" / f"tau_feeder_{label}.ftmw"
    work_fp.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(base_fixture, work_fp)
    write_stage2b_recommended_shape(str(work_fp), shape=recommended_shape)

    logger.info(
        "Variant %s (recommended_shape=%s) ...", label, recommended_shape,
    )
    t0 = time.perf_counter()
    from ftmwpipeline._internal.stage3_impl import detect_peaks_impl
    result = detect_peaks_impl(str(work_fp))
    runtime = time.perf_counter() - t0

    summary = summarise(
        result,
        shape=recommended_shape,
        knob_label="tau_feeder_variant",
        knob_value=label,
        runtime_s=runtime,
    )
    logger.info(
        "  %s: τ=%.3f µs (%s), n_peaks=%d, n_promoted=%d "
        "(primary=%d, gap=%d), SNR median=%.2f, p95=%.2f, sidelobe-proxy=%d",
        label, summary.tau_basis_us,
        result["parameters_used"].get("tau_basis_source", "?"),
        summary.n_peaks, summary.n_promoted, summary.n_primary, summary.n_gap,
        summary.snr_median, summary.snr_p95, summary.n_internal_high_user_low,
    )
    return summary


def _plot(rows: List[Stage3RunResult], out: Path) -> Path:
    labels = [r.knob_value for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    ax_cnt, ax_snr, ax_side = axes

    # Panel 1: per-pass peak counts (stacked bar).
    primary = np.array([r.n_primary for r in rows])
    gap = np.array([r.n_gap for r in rows])
    x = np.arange(len(rows))
    ax_cnt.bar(x, primary, label="primary", color="tab:blue")
    ax_cnt.bar(x, gap, bottom=primary, label="gap", color="tab:orange")
    promoted = [r.n_promoted for r in rows]
    for i, v in enumerate(promoted):
        ax_cnt.text(
            i, primary[i] + gap[i] + 5, f"prom={v}", ha="center", fontsize=8,
        )
    ax_cnt.set_xticks(x)
    ax_cnt.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax_cnt.set_ylabel("peak count")
    ax_cnt.set_title("Per-pass peak counts (promoted noted above)")
    ax_cnt.legend(fontsize=8)
    ax_cnt.grid(True, alpha=0.3, axis="y")

    # Panel 2: SNR distribution (p25 / median / p75 / p95 as a box+whisker
    # equivalent; we don't carry the full distribution).
    snr_med = np.array([r.snr_median for r in rows])
    snr_p25 = np.array([r.snr_p25 for r in rows])
    snr_p75 = np.array([r.snr_p75 for r in rows])
    snr_p95 = np.array([r.snr_p95 for r in rows])
    ax_snr.errorbar(
        x, snr_med, yerr=[snr_med - snr_p25, snr_p75 - snr_med],
        fmt="o", color="tab:blue", capsize=4, label="median ± IQR",
    )
    ax_snr.scatter(x, snr_p95, marker="^", color="tab:red", label="p95")
    ax_snr.set_xticks(x)
    ax_snr.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax_snr.set_yscale("log")
    ax_snr.set_ylabel("SNR")
    ax_snr.set_title("SNR distribution of detected peaks")
    ax_snr.legend(fontsize=8)
    ax_snr.grid(True, which="both", alpha=0.3)

    # Panel 3: sidelobe-suspect proxy (high internal SNR but un-promoted on
    # user grid). A large value here flags gap-pass leakage.
    side = np.array([r.n_internal_high_user_low for r in rows])
    ax_side.bar(x, side, color="tab:red")
    ax_side.set_xticks(x)
    ax_side.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax_side.set_ylabel("n peaks (internal SNR ≥ 2×floor, user SNR < floor)")
    ax_side.set_title("Sidelobe-suspect proxy count")
    ax_side.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Stage 3 τ-feeder substitution impact (2638 unapodized)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=140)
    plt.close(fig)
    logger.info("Wrote %s", out)
    return out


def main() -> None:
    logger.info("Preparing per-shape fixtures (cached) ...")
    lor_fp = prepare_fixture("lorentzian")
    gauss_fp = prepare_fixture("gaussian")

    rows: List[Stage3RunResult] = []
    rows.append(_variant(lor_fp, label="L_anchor", recommended_shape="lorentzian"))
    rows.append(_variant(gauss_fp, label="G_old_tau", recommended_shape="lorentzian"))
    rows.append(_variant(gauss_fp, label="G_new_tau", recommended_shape="gaussian"))
    rows.append(_variant(lor_fp, label="L_with_tau_G", recommended_shape="gaussian"))

    csv_path = DATA / "probe_tau_feeder.csv"
    write_csv(rows, csv_path)
    json_path = DATA / "probe_tau_feeder.json"
    json_path.write_text(json.dumps([asdict(r) for r in rows], indent=2))
    logger.info("Wrote %s", json_path)
    _plot(rows, FIG / "probe_tau_feeder.png")

    print()
    print("====== Stage 3 τ-feeder probe summary ======")
    fmt = (
        "  {label:>14s}   τ={tau:>6.3f} µs   "
        "n_peaks={np:>4d}   n_promoted={pr:>4d}   "
        "primary={pri:>4d}   gap={gap:>4d}   "
        "SNR_med={sm:>5.2f}   p95={sp:>6.2f}   "
        "sidelobe_proxy={sl:>3d}"
    )
    for r in rows:
        print(fmt.format(
            label=r.knob_value, tau=r.tau_basis_us or float("nan"),
            np=r.n_peaks, pr=r.n_promoted,
            pri=r.n_primary, gap=r.n_gap,
            sm=r.snr_median, sp=r.snr_p95,
            sl=r.n_internal_high_user_low,
        ))


if __name__ == "__main__":
    main()
