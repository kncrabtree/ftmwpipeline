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
  collapses the statistic to the null).

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


def run_2638(workdir: Optional[Path] = None) -> Dict[str, Any]:
    """Score the shipped edge-coherence map on the reference active FT."""
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
    finite = rolling[np.isfinite(rolling)]
    coverage = float(np.mean(finite >= EDGE_THRESHOLD))

    # De-ramp pitfall: de-ramping the already-[0, T] active grid winds the band
    # phase and collapses the statistic toward the null.
    fid = load_fid_from_pipeline_impl(path)
    pp = compute_ft_impl(file_path=path)["complex_ft"].metadata["processing_params"]
    start_us = float(pp.start_us or 0.0)
    probe = float(fid.probe_freq_mhz)
    deramped = deramp_to_active_start(freq, spec, probe, start_us)
    rolling_dr = active_edge_coherence(deramped, sigma, band_m=64)
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
        "start_us": start_us,
        "n_bins": int(freq.size),
        "coverage_active": coverage,
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


def _figdir() -> Path:
    d = Path(__file__).resolve().parent / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_figures(results: Dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ftmwpipeline.visualization.report_style import (
        AGGIE_BLUE,
        AGGIE_GOLD,
        BRAND_CYCLE,
        DOUBLE_DECKER,
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


def run(do_2638: bool = True, figures: bool = True) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    results["null"] = null_calibration()
    results["signal_growth"] = signal_growth()
    print(
        f"null mean @M=64: {results['null']['mean']['64']:.3f} "
        f"(sqrt(pi/4)={NULL_MEAN:.3f})",
        flush=True,
    )
    if do_2638:
        results["2638"] = run_2638()
        d = results["2638"]
        print(
            f"2638: active coverage {100*d['coverage_active']:.1f}% median "
            f"{d['median_active']:.2f}; de-ramped median {d['median_deramped']:.2f} "
            f"(collapse); 36350 S_coh {d['scoh_at_36350']:.0f}",
            flush=True,
        )
    if figures:
        make_figures(results)
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
