"""Probe 1: ``leakage.tau_us`` for Stage 4.

Asks: should Stage 4's analytic leakage reach (``estimate_leakage_reach``)
default to the boxcar (undamped) limit, or should it consume the Stage 2b
``τ_maj`` / ``τ_G_maj`` anchor when available?

Background
----------

``estimate_leakage_reach`` predicts the half-width ``Δf`` (MHz) at which
a strong line's truncation-leakage sidelobe falls below ``min_snr·σ``,
given the peak SNR, the acquisition length ``T``, and an assumed natural
decay time constant ``τ``. The leakage reach feeds into:

* The lower bound on isolated-peak window half-widths (so a window is
  guaranteed at least as wide as the strongest in-window line's
  leakage reach).
* The tier-1 contributor-attachment predicted skirt magnitude.

Default = ``None`` → boxcar (τ_eff = T, env_factor = 2.0). This was
chosen as the "safe most leakage-prone" case; in practice on 2638
(T ≈ 12.65 µs, τ ≈ 6 µs) the τ-fed reach is actually *wider* than
boxcar because the smaller τ_eff dominates the reach formula
``reach ∝ env_factor / tau_eff``. The probe quantifies the impact on
plan composition and decides whether to swap the default.

Variants
--------

* ``boxcar`` -- ``tau_us = None`` (current default).
* ``tau_maj`` -- Stage 2b Lorentzian anchor (pure-exp envelope).
* ``tau_G_maj`` -- Stage 2b Gaussian anchor.
* ``tau_3`` / ``tau_12`` -- bracketing values (3 µs and 12 µs) to
  isolate the τ-dependence shape independent of the Stage 2b value.

Each variant runs against both the Lorentzian and Gaussian shape
fixtures (8 runs total). The "shape" axis here only controls
``recommended_shape`` for the Stage 3 τ-feeder upstream -- the
Stage 4 ``tau_us`` is what the variant pins.

Outputs
-------

* ``data/probe_leakage_tau.csv``
* ``figures/probe_leakage_tau.png`` -- 4-panel summary.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from harness import (
    Stage4RunResult,
    get_stage2b_tau,
    prepare_fixture,
    run_assign_windows,
    write_csv,
)

from ftmwpipeline.core.window_planning_settings import WindowPlanningSettings

DATA = HERE / "data"
FIG = HERE / "figures"
DATA.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("probe-leakage-tau")


def _settings_for(tau_us):
    s = WindowPlanningSettings()
    s.leakage.tau_us = tau_us
    return s


def _run_variant(fp, *, shape, label, tau_us):
    s = _settings_for(tau_us)
    r = run_assign_windows(
        fp, shape=shape, settings=s,
        knob_label="leakage.tau_us", knob_value=label,
    )
    logger.info(
        "  %s/%s (τ=%s) → n_windows=%d (hard=%d), n_fixed=%d, width p95=%.2f MHz",
        shape, label,
        f"{tau_us:.3f}" if tau_us is not None else "None",
        r.n_windows, r.n_hard, r.n_fixed_contributors, r.width_p95_mhz,
    )
    return r


def _plot(rows, out: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    (ax_n, ax_h), (ax_w, ax_c) = axes

    # Group by variant label, two bars per group (L / G).
    labels = sorted({r.knob_value for r in rows}, key=str)
    x = np.arange(len(labels))
    width = 0.35

    for offs, shape, colour in (
        (-width / 2, "lorentzian", "tab:blue"),
        (width / 2, "gaussian", "tab:orange"),
    ):
        sub = {r.knob_value: r for r in rows if r.shape == shape}
        n_win = [sub[l].n_windows if l in sub else 0 for l in labels]
        n_hard = [sub[l].n_hard if l in sub else 0 for l in labels]
        w95 = [sub[l].width_p95_mhz if l in sub else 0.0 for l in labels]
        n_fc = [
            sub[l].n_fixed_contributors if l in sub else 0 for l in labels
        ]
        ax_n.bar(x + offs, n_win, width, color=colour, label=shape)
        ax_h.bar(x + offs, n_hard, width, color=colour, label=shape)
        ax_w.bar(x + offs, w95, width, color=colour, label=shape)
        ax_c.bar(x + offs, n_fc, width, color=colour, label=shape)

    for ax, ylab in (
        (ax_n, "n_windows"),
        (ax_h, "n_hard"),
        (ax_w, "width p95 (MHz)"),
        (ax_c, "n_fixed_contributors"),
    ):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Stage 4 `leakage.tau_us` variants (2638 unapodized)", fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    logger.info("Wrote %s", out)
    return out


def main() -> None:
    lor_fp = prepare_fixture("lorentzian")
    gauss_fp = prepare_fixture("gaussian")
    tau_L = get_stage2b_tau(lor_fp, shape="lorentzian")
    tau_G = get_stage2b_tau(gauss_fp, shape="gaussian")
    logger.info("τ_maj (Lorentzian) = %.3f µs", tau_L)
    logger.info("τ_G_maj (Gaussian) = %.3f µs", tau_G)

    rows: list[Stage4RunResult] = []
    for shape, fp in (("lorentzian", lor_fp), ("gaussian", gauss_fp)):
        rows.append(_run_variant(fp, shape=shape, label="boxcar", tau_us=None))
        rows.append(_run_variant(fp, shape=shape, label="tau_maj", tau_us=tau_L))
        rows.append(_run_variant(fp, shape=shape, label="tau_G_maj", tau_us=tau_G))
        rows.append(_run_variant(fp, shape=shape, label="tau_3", tau_us=3.0))
        rows.append(_run_variant(fp, shape=shape, label="tau_12", tau_us=12.0))

    write_csv(rows, DATA / "probe_leakage_tau.csv")
    _plot(rows, FIG / "probe_leakage_tau.png")

    print()
    print("====== Stage 4 leakage.tau_us probe summary ======")
    fmt = (
        "  {shape:>10s}  {label:>10s}  τ={tau:>6s} µs   "
        "n_win={nw:>4d}  hard={nh:>3d}  fc={nc:>5d}  "
        "w_p95={w95:>6.2f}MHz  w_max={wmax:>6.2f}MHz"
    )
    for r in rows:
        tau = (
            f"{r.leakage_tau_us:.3f}"
            if r.leakage_tau_us is not None else "None"
        )
        print(fmt.format(
            shape=r.shape, label=r.knob_value, tau=tau,
            nw=r.n_windows, nh=r.n_hard, nc=r.n_fixed_contributors,
            w95=r.width_p95_mhz, wmax=r.width_max_mhz,
        ))


if __name__ == "__main__":
    main()
