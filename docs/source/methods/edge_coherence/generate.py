"""Regenerate the figures and key results for the edge-coherence note.

This harness reproduces, from a fixed-seed simulator and the checked-in 2638
fixture, the quantitative claims in :doc:`edge_coherence`:

* the **closed-form null** of the complex-edge coherence statistic
  ``S_coh = |sum z| / (sigma sqrt(M))`` -- mean ``sqrt(pi/4) ~= 0.886`` and
  ``M``-independent, so one threshold works at every band width;
* the **sqrt(M) discrimination gain** -- against a coherent leakage tail of
  fixed per-bin amplitude ``L`` the statistic grows as ``(L/sigma) sqrt(M)``;
* the **active-FT application** on the reference experiment -- the
  leakage-touched coverage, the local-sigma requirement, the absence of a
  pedestal, and the de-ramp pitfall (de-ramping the already-[0, T] active grid
  collapses the statistic to the null);
* a **strong-line illustration** (``fig3``) over the centered 36350 MHz line: the
  real/imaginary leakage skirt the magnitude hides, and the rolling statistic
  towering above threshold and crossing back below it on either side.

The statistic, the rolling map, and the de-ramp are the shipped
``ftmwpipeline.preprocessing.edge_coherence`` /
``ftmwpipeline.preprocessing.leakage`` functions, so the note validates the
production code.

Run as a script to (re)write ``figures/*.png`` and ``results.json`` beside this
file::

    python generate.py                 # synthetic + 2638 (a few seconds)
    python generate.py --no-2638       # synthetic only

The committed ``results.json`` is the regression target for
``tests/integration/test_edge_coherence_report.py`` (the ``slow`` marker); the
figures are embedded by the note. ``null_calibration`` and ``signal_growth``
back the fast unit checks in
``tests/unit/preprocessing/test_edge_coherence_invariants.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ftmwpipeline.preprocessing.edge_coherence import (
    NULL_MEAN,
    active_edge_coherence,
    coherence_statistic,
)

SEED = 20260519
EXAMPLES = Path("examples/blackchirp_data")
TRIM = (26500.0, 40000.0)
BAND_WIDTHS = (8, 16, 32, 64, 128)
N_TRIALS = 4000
EDGE_THRESHOLD = 8.0


def _complex_noise(n: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Complex Gaussian band with E[|n|^2] = sigma^2."""
    s = sigma / np.sqrt(2.0)
    return rng.normal(0.0, s, n) + 1j * rng.normal(0.0, s, n)


def null_calibration(seed: int = SEED) -> Dict[str, Any]:
    """Null distribution of S_coh across band widths (clean complex noise)."""
    rng = np.random.default_rng(seed)
    out: Dict[str, Any] = {
        "band_widths": list(BAND_WIDTHS),
        "mean": {},
        "std": {},
        "p99": {},
    }
    sigma = 0.02
    for m in BAND_WIDTHS:
        vals = np.array(
            [
                coherence_statistic(_complex_noise(m, sigma, rng), sigma)
                for _ in range(N_TRIALS)
            ]
        )
        out["mean"][str(m)] = float(vals.mean())
        out["std"][str(m)] = float(vals.std())
        out["p99"][str(m)] = float(np.percentile(vals, 99))
    return out


def signal_growth(seed: int = SEED) -> Dict[str, Any]:
    """S_coh of a coherent band (constant phase) vs band width, by L/sigma.

    A coherent leakage tail of per-bin amplitude ``L`` adds in phase, so the
    statistic grows as ``(L/sigma) sqrt(M)`` -- the sqrt(M) discrimination gain.
    """
    rng = np.random.default_rng(seed + 1)
    sigma = 0.02
    l_over_sigma = [0.5, 1.0, 2.0]
    out: Dict[str, Any] = {
        "band_widths": list(BAND_WIDTHS),
        "l_over_sigma": l_over_sigma,
        "value": {},
    }
    for los in l_over_sigma:
        row: List[float] = []
        for m in BAND_WIDTHS:
            trials = []
            for _ in range(200):
                band = los * sigma * np.exp(1j * 0.7) + _complex_noise(m, sigma, rng)
                trials.append(coherence_statistic(band, sigma))
            row.append(float(np.mean(trials)))
        out["value"][str(los)] = row
    return out


def _build_2638_active(workdir: Optional[Path] = None) -> Dict[str, Any]:
    """Build 2638 through noise estimation; return the active-FT arrays.

    Returns the ascending-frequency active FT (``freq``, ``spec``, ``sigma``),
    the rolling edge-coherence map, and the de-ramped counterpart, shared by the
    scalar summary and the strong-line figure.
    """
    import tempfile

    import ftmwpipeline.api as ftmw
    from ftmwpipeline._internal.active_ft_support import build_active_grid_with_noise
    from ftmwpipeline._internal.stage0_impl import load_fid_from_pipeline_impl
    from ftmwpipeline._internal.stage1_impl import compute_ft_impl
    from ftmwpipeline.preprocessing.leakage import deramp_to_active_start

    tmp = workdir or Path(tempfile.mkdtemp(prefix="edge-coh-note-"))
    tmp.mkdir(parents=True, exist_ok=True)
    path = str(tmp / "exp_2638.ftmw")
    ftmw.import_data(path, source=str(EXAMPLES / "2638"), force=True)
    ftmw.detect_start_time(path, band=TRIM, stamp=True)
    ftmw.compute_ft(path, trim=TRIM)
    ftmw.estimate_noise(path)

    cft, rms = build_active_grid_with_noise(path, TRIM)
    order = np.argsort(cft.freq_array)
    freq = cft.freq_array[order]
    spec = cft.complex_spectrum[order]
    sigma = rms[order]
    rolling = active_edge_coherence(spec, sigma, band_m=64)

    # De-ramp pitfall: de-ramping the already-[0, T] active grid winds the band
    # phase and collapses the statistic toward the null.
    fid = load_fid_from_pipeline_impl(path)
    pp = compute_ft_impl(file_path=path)["complex_ft"].metadata["processing_params"]
    start_us = float(pp.start_us or 0.0)
    probe = float(fid.probe_freq_mhz)
    deramped = deramp_to_active_start(freq, spec, probe, start_us)
    rolling_dr = active_edge_coherence(deramped, sigma, band_m=64)
    return {
        "freq": freq,
        "spec": spec,
        "sigma": sigma,
        "rolling": rolling,
        "rolling_dr": rolling_dr,
        "start_us": start_us,
    }


def _summarize_2638(a: Dict[str, Any]) -> Dict[str, Any]:
    """Scalar regression summary from the active-FT arrays."""
    freq, spec, sigma = a["freq"], a["spec"], a["sigma"]
    rolling, rolling_dr = a["rolling"], a["rolling_dr"]
    finite = rolling[np.isfinite(rolling)]
    finite_dr = rolling_dr[np.isfinite(rolling_dr)]

    # Strong line at 36350: peak value and the contiguous above-threshold run.
    i350 = int(np.argmin(np.abs(freq - 36350.0)))
    gap = (freq > 36352.0) & (freq < 36388.0)

    # Quiet region near 29008: no pedestal -> median Re/Im consistent with zero.
    quiet = (freq > 28983.0) & (freq < 29033.0)
    sig_q = float(np.median(sigma[quiet]))
    re_q = float(np.median(spec[quiet].real)) / sig_q
    im_q = float(np.median(spec[quiet].imag)) / sig_q

    return {
        "start_us": float(a["start_us"]),
        "n_bins": int(freq.size),
        "coverage_active": float(np.mean(finite >= EDGE_THRESHOLD)),
        "median_active": float(np.median(finite)),
        "median_deramped": float(np.median(finite_dr)),
        "coverage_deramped": float(np.mean(finite_dr >= EDGE_THRESHOLD)),
        "scoh_at_36350": float(rolling[i350]),
        "scoh_gap_36352_36388_median": float(np.nanmedian(rolling[gap])),
        "sigma_min": float(sigma.min()),
        "sigma_max": float(sigma.max()),
        "sigma_ratio": float(sigma.max() / sigma.min()),
        "quiet_re_over_sigma": re_q,
        "quiet_im_over_sigma": im_q,
    }


def run_2638(workdir: Optional[Path] = None) -> Dict[str, Any]:
    """Score the shipped edge-coherence map on the reference active FT."""
    return _summarize_2638(_build_2638_active(workdir))


def _figdir() -> Path:
    d = Path(__file__).resolve().parent / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_figures(
    results: Dict[str, Any], panel: Optional[Dict[str, Any]] = None
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ftmwpipeline.visualization.report_style import (
        AGGIE_BLUE,
        AGGIE_GOLD,
        BRAND_CYCLE,
        CABERNET,
        DOUBLE_DECKER,
        GUNROCK,
        apply_bare_style,
        apply_color_cycle,
    )

    apply_color_cycle(matplotlib)
    figdir = _figdir()
    nul = results["null"]
    grw = results["signal_growth"]
    m = np.array(nul["band_widths"], float)

    # fig1: null mean + 99th percentile vs band width (M-independence).
    fig, ax = plt.subplots(figsize=(7, 5))
    mean = np.array([nul["mean"][str(int(x))] for x in m])
    p99 = np.array([nul["p99"][str(int(x))] for x in m])
    ax.plot(m, mean, "o-", color=AGGIE_BLUE, lw=1.8, label="null mean")
    ax.axhline(NULL_MEAN, color=AGGIE_BLUE, ls=":", lw=1.0)
    ax.plot(m, p99, "s-", color=AGGIE_GOLD, lw=1.8, label="null 99th pct")
    ax.text(m[-1], NULL_MEAN, r"  $\sqrt{\pi/4}$", va="center", color=AGGIE_BLUE)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("band width M")
    ax.set_ylabel(r"$S_\mathrm{coh}$ (clean noise)")
    ax.set_ylim(0, max(p99) * 1.15)
    apply_bare_style(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "fig1_null_m_independence.png", dpi=130)
    plt.close(fig)

    # fig2: signal grows as sqrt(M); the null is flat. The discrimination gain.
    fig, ax = plt.subplots(figsize=(7, 5))
    for los, color in zip(grw["l_over_sigma"], BRAND_CYCLE):
        ax.plot(
            m,
            np.array(grw["value"][str(los)]),
            "o-",
            color=color,
            lw=1.8,
            label=rf"$L/\sigma={los:g}$",
        )
    ax.axhline(NULL_MEAN, color="#999999", ls=":", lw=1.0, label="null")
    ax.axhline(
        EDGE_THRESHOLD,
        color=DOUBLE_DECKER,
        ls="--",
        lw=1.2,
        label=r"$T_\mathrm{edge}=8$",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("band width M")
    ax.set_ylabel(r"$S_\mathrm{coh}$")
    apply_bare_style(ax)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "fig2_signal_growth.png", dpi=130)
    plt.close(fig)

    if panel is None:
        return

    # fig3: a strong line on the reference active FT, in local-SNR units, with the
    # real and imaginary components carrying the coherent leakage skirt that the
    # magnitude alone hides, and the rolling S_coh beneath it. Window the 36350 MHz
    # line and ~12 MHz of its skirt.
    freq = panel["freq"]
    sigma = panel["sigma"]
    z_over_s = panel["spec"] / sigma
    rolling = panel["rolling"]
    mag_all = np.abs(z_over_s)

    # Center on the strong line and widen until the rolling statistic falls back
    # below the threshold on both sides, so the figure shows the full
    # above-threshold run and both crossings, with the line centered.
    core_idxs = np.flatnonzero((freq >= 36348.0) & (freq <= 36352.0))
    core_i = int(core_idxs[int(np.argmax(mag_all[core_idxs]))])
    lo_i = hi_i = core_i
    while (
        lo_i > 0
        and np.isfinite(rolling[lo_i - 1])
        and rolling[lo_i - 1] >= EDGE_THRESHOLD
        and freq[core_i] - freq[lo_i - 1] < 40.0
    ):
        lo_i -= 1
    while (
        hi_i < freq.size - 1
        and np.isfinite(rolling[hi_i + 1])
        and rolling[hi_i + 1] >= EDGE_THRESHOLD
        and freq[hi_i + 1] - freq[core_i] < 40.0
    ):
        hi_i += 1
    half = max(freq[core_i] - freq[lo_i], freq[hi_i] - freq[core_i]) + 2.0
    sel = (freq >= freq[core_i] - half) & (freq <= freq[core_i] + half)
    f = freq[sel]
    re, im, mag = z_over_s[sel].real, z_over_s[sel].imag, np.abs(z_over_s[sel])
    sc = rolling[sel]

    fig, (axT, axB) = plt.subplots(
        2, 1, sharex=True, figsize=(8.5, 5.8), height_ratios=[3, 2]
    )
    # Top: Re / Im / |X| in local-SNR units; clip so the off-scale core does not
    # flatten the skirt structure the figure is about.
    axT.axhline(0.0, color="#c8ced6", lw=0.8, zorder=0)
    axT.plot(f, re, color=DOUBLE_DECKER, lw=0.9, label=r"$\mathrm{Re}/\sigma$")
    axT.plot(f, im, color=GUNROCK, lw=0.9, label=r"$\mathrm{Im}/\sigma$")
    axT.plot(f, mag, color=CABERNET, lw=1.5, label=r"$|X|/\sigma$")
    clip = float(np.nanpercentile(mag, 90)) * 1.8
    axT.set_ylim(-clip, clip)
    axT.text(
        0.015,
        0.96,
        "strong lines clipped to skirt scale",
        transform=axT.transAxes,
        fontsize=8,
        va="top",
        color="0.35",
    )
    axT.set_ylabel(r"amplitude / $\sigma$")
    axT.legend(fontsize=8, ncol=3, loc="upper right")
    apply_bare_style(axT)

    # Bottom: the rolling coherent-sum statistic over the same span.
    axB.plot(f, sc, color=AGGIE_BLUE, lw=1.5)
    axB.fill_between(
        f, EDGE_THRESHOLD, sc, where=sc >= EDGE_THRESHOLD, color=AGGIE_GOLD, alpha=0.30
    )
    axB.axhline(
        EDGE_THRESHOLD,
        color=DOUBLE_DECKER,
        ls="--",
        lw=1.2,
        label=r"$T_\mathrm{edge}=8$",
    )
    axB.axhline(
        NULL_MEAN, color="#999999", ls=":", lw=1.0, label=r"null $\sqrt{\pi/4}$"
    )
    axB.set_ylim(bottom=0.0)
    axB.set_ylabel(r"$S_\mathrm{coh}$ ($M{=}64$)")
    axB.set_xlabel("frequency (MHz)")
    axB.legend(fontsize=8, loc="upper right")
    apply_bare_style(axB)

    fig.tight_layout()
    fig.savefig(figdir / "fig3_strong_line_coherence.png", dpi=130)
    plt.close(fig)


def run(do_2638: bool = True, figures: bool = True) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    results["null"] = null_calibration()
    results["signal_growth"] = signal_growth()
    print(
        f"null mean @M=64: {results['null']['mean']['64']:.3f} "
        f"(sqrt(pi/4)={NULL_MEAN:.3f})",
        flush=True,
    )
    panel: Optional[Dict[str, Any]] = None
    if do_2638:
        panel = _build_2638_active()
        results["2638"] = _summarize_2638(panel)
        d = results["2638"]
        print(
            f"2638: active coverage {100*d['coverage_active']:.1f}% median "
            f"{d['median_active']:.2f}; de-ramped median {d['median_deramped']:.2f} "
            f"(collapse); 36350 S_coh {d['scoh_at_36350']:.0f}",
            flush=True,
        )
    if figures:
        make_figures(results, panel)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-2638", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument(
        "--out", default=str(Path(__file__).resolve().parent / "results.json")
    )
    args = ap.parse_args()
    results = run(do_2638=not args.no_2638, figures=not args.no_figures)
    if not args.no_2638:
        Path(args.out).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
