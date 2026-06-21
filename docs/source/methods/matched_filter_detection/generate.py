"""Regenerate the figures and key results for the matched-filter detection note.

This harness reproduces, from a fixed-seed simulator and the checked-in 2638
fixture, the quantitative claims in :doc:`matched_filter_detection`:

* the **projection-equals-apodized-FFT identity** — a sigma-weighted complex
  projection of the active-region spectrum onto a finite-decay Lorentzian basis
  at every frequency is one exponentially-apodized FFT, not N projections, so
  the matched filter is an ``O(N log N)`` operation;
* the **synthetic recall/false-positive sweep** — across the
  signal-to-noise x line-width-per-bin plane, the matched-filter gap-pass
  kernel holds near-complete weak-line recall through the narrow-line regime
  (FWHM/bin ~ 1-2) real FTMW data sits in, where a strong-window
  (Blackman-Harris) detector collapses;
* the **2638 application** — on the reference experiment the matched-filter gap
  pass recovers real lines (Stage 5 fit peaks) that the apodized primary pass
  misses, at the cost of more raw candidates.

The synthetic detectors call the shipped
:func:`ftmwpipeline.utils.signal_processing.matched_filter_window` and
:func:`ftmwpipeline.preprocessing.peak_detection.locate_peaks`, so the note
validates the production kernels rather than a private prototype.

Run as a script to (re)write ``figures/*.png`` and ``results.json`` beside this
file::

    python generate.py                 # synthetic + 2638 (a few minutes)
    python generate.py --no-2638       # synthetic only (a few seconds)

The committed ``results.json`` is the regression target for
``tests/integration/test_matched_filter_report.py`` (the ``slow`` marker); the
figures are embedded by the note. ``projection_fft_identity`` and
``synthetic_sweep`` back the fast unit checks in
``tests/unit/preprocessing/test_matched_filter_invariants.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks
from scipy.signal.windows import blackmanharris

from ftmwpipeline.core.data_structures import Sideband
from ftmwpipeline.preprocessing.peak_detection import locate_peaks
from ftmwpipeline.utils.signal_processing import matched_filter_window

SEED = 20260524
EXAMPLES = Path("examples/blackchirp_data")
TRIM = (26500.0, 40000.0)

# Synthetic sweep grid. The regime real FTMW data sits in is FWHM/bin ~ 1-2,
# SNR ~ 3-10; the grid brackets it on both sides.
SWEEP_SNR: Tuple[float, ...] = (2.0, 3.0, 4.0, 5.0, 7.0, 10.0)
SWEEP_FWHM_BINS: Tuple[float, ...] = (0.8, 1.0, 1.34, 2.0, 3.0, 5.0)
SWEEP_TRIALS = 4


# --------------------------------------------------------------------------
# Synthetic active-FT simulator (weak Lorentzians, white time-domain noise)
# --------------------------------------------------------------------------
def simulate_fid(
    *,
    fwhm_bins: float,
    true_snr: float,
    seed: int,
    n_lines: int = 25,
    n_active: int = 4096,
    sample_dt_us: float = 0.020,
    probe_freq_mhz: float = 40000.0,
    sideband: Sideband = Sideband.LOWER,
    edge_guard_bins: int = 20,
    min_line_separation_bins: float = 6.0,
) -> Dict[str, Any]:
    """One synthetic active region: a real damped-cosine FID plus white noise.

    The line decay is realized directly in the FID (no apodization). The
    line FWHM is fixed in active-FT bin units by ``tau_truth = T /
    (pi * fwhm_bins)``, and the time-domain noise scale is set so the median
    injected line reaches ``true_snr`` on the unapodized active-FT magnitude
    against the analytic per-bin complex-RMS noise.
    """
    rng = np.random.default_rng(seed)
    t_active = n_active * sample_dt_us
    tau_truth = t_active / (np.pi * float(fwhm_bins))

    valid = np.arange(edge_guard_bins, n_active // 2 - edge_guard_bins)
    chosen: List[int] = []
    attempts = 0
    while len(chosen) < n_lines and attempts < 200 * n_lines:
        attempts += 1
        b = int(rng.choice(valid))
        if any(abs(b - c) < min_line_separation_bins for c in chosen):
            continue
        chosen.append(b)
    if len(chosen) < n_lines:
        raise RuntimeError("could not place all synthetic lines")
    line_bins = np.sort(np.array(chosen, dtype=int))

    t = np.arange(n_active) * sample_dt_us
    f_bb = line_bins.astype(float) / t_active
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_lines)
    fid: np.ndarray = np.zeros(n_active, dtype=float)
    for i in range(n_lines):
        fid += np.cos(2 * np.pi * f_bb[i] * t + phases[i]) * np.exp(-t / tau_truth)

    # Noise-free on-line magnitude -> size the time-domain noise for true_snr.
    spec0 = sample_dt_us * np.fft.rfft(fid)
    on_line = float(np.median(np.abs(spec0[line_bins])))
    target_sigma_x = on_line / true_snr
    sigma_time = target_sigma_x / np.sqrt(t_active * sample_dt_us)
    fid_noisy = fid + rng.normal(scale=sigma_time, size=n_active)

    s = -1.0 if sideband == Sideband.LOWER else 1.0
    freq_bb = np.fft.rfftfreq(n_active, d=sample_dt_us)
    freq_mhz = probe_freq_mhz + s * freq_bb
    truth_mhz = probe_freq_mhz + s * f_bb

    return {
        "fid": fid_noisy,
        "sample_dt_us": sample_dt_us,
        "t_active": t_active,
        "tau_truth_us": tau_truth,
        "freq_mhz": freq_mhz,
        "truth_mhz": truth_mhz,
        "sigma_x": target_sigma_x,
        "bin_mhz": 1.0 / t_active,
        "fwhm_bins": float(fwhm_bins),
    }


# --------------------------------------------------------------------------
# Detectors (synthetic): the matched filter and a Blackman-Harris baseline
# --------------------------------------------------------------------------
def _apodized_sigma_c(sigma_x_unapodized: float, window: np.ndarray) -> float:
    """Per-component noise sigma of an apodized active-FT bin.

    For white time-domain noise the per-bin complex-RMS of the unapodized
    active-FT is ``sigma_x = dt * sigma_time * sqrt(N)``. Apodizing by
    ``window`` scales the per-component (real or imaginary) sigma to
    ``dt * sigma_time * sqrt(sum(w^2)/2)``. The matched filter's
    signal-to-noise gain lives entirely in this denominator, so the synthetic
    detector must use the window's own noise, not the unapodized sigma.
    """
    n = window.size
    sigma_time_dt = float(sigma_x_unapodized) / np.sqrt(n)
    return sigma_time_dt * np.sqrt(float(np.sum(window**2)) / 2.0)


def matched_filter_candidates(
    sim: Dict[str, Any], *, tau_basis_us: float, detection_snr: float = 4.0
) -> np.ndarray:
    """Matched-filter candidate frequencies: exp-apodized FFT + per-bin SNR.

    Mirrors the shipped gap-pass kernel: window the active FID by the
    Lorentzian matched filter (the shipped ``matched_filter_window``), FFT,
    divide the magnitude by the per-component sigma, and keep local maxima
    above ``detection_snr``. No concavity test — the per-bin threshold alone.
    """
    fid = np.asarray(sim["fid"], dtype=float)
    dt = sim["sample_dt_us"]
    t_rel = np.arange(fid.size) * dt
    w = matched_filter_window(t_rel, tau_basis_us, shape="lorentzian")
    mag = np.abs(dt * np.fft.rfft(fid * w))
    snr = mag / _apodized_sigma_c(sim["sigma_x"], w)
    sep = max(2, int(round(sim["fwhm_bins"])))
    bins, _ = find_peaks(snr, height=detection_snr, distance=sep)
    return np.asarray(sim["freq_mhz"][bins], dtype=float)


def blackman_harris_candidates(
    sim: Dict[str, Any], *, min_snr: float = 2.0, sg_window: int = 11
) -> np.ndarray:
    """Blackman-Harris primary-pass baseline: strong window + concavity locator.

    Apodizes the active region with a Blackman-Harris window (heavy sidelobe
    suppression) and runs the shipped Savitzky-Golay concavity locator
    (``locate_peaks``) against the per-bin magnitude threshold.
    """
    fid = np.asarray(sim["fid"], dtype=float)
    dt = sim["sample_dt_us"]
    bh = blackmanharris(fid.size)
    mag = np.abs(dt * np.fft.rfft(fid * bh))
    freq_bb = np.fft.rfftfreq(fid.size, d=dt)
    # The Blackman-Harris window changes the per-bin noise too; threshold on
    # the window's own analytic per-component sigma.
    thresh = np.full(mag.shape, min_snr * _apodized_sigma_c(sim["sigma_x"], bh))
    res = locate_peaks(freq_bb, mag, window=sg_window, order=3, thresh=thresh)
    # Map baseband -> molecular with the simulator's probe / LOWER sideband.
    mol = 40000.0 - np.asarray(res.freqs, dtype=float)
    return mol


def _recall_fp(
    cand_mhz: np.ndarray, truth_mhz: np.ndarray, tol_mhz: float
) -> Tuple[float, int]:
    """Recall (fraction of truths matched) and false-positive count."""
    if truth_mhz.size == 0:
        return 1.0, int(cand_mhz.size)
    matched_truth = 0
    for f in truth_mhz:
        if cand_mhz.size and np.min(np.abs(cand_mhz - f)) <= tol_mhz:
            matched_truth += 1
    fp = 0
    for c in cand_mhz:
        if truth_mhz.size == 0 or np.min(np.abs(truth_mhz - c)) > tol_mhz:
            fp += 1
    return matched_truth / truth_mhz.size, fp


def synthetic_sweep() -> Dict[str, Any]:
    """Recall + false-positive grid for the matched filter and the BH baseline."""
    detectors = ("matched_filter", "blackman_harris")
    recall: Dict[str, List[List[float]]] = {d: [] for d in detectors}
    fp: Dict[str, List[List[float]]] = {d: [] for d in detectors}
    for snr in SWEEP_SNR:
        rows: Dict[str, List[float]] = {d: [] for d in detectors}
        fprows: Dict[str, List[float]] = {d: [] for d in detectors}
        for fwhm in SWEEP_FWHM_BINS:
            cell: Dict[str, Tuple[List[float], List[int]]] = {
                d: ([], []) for d in detectors
            }
            for trial in range(SWEEP_TRIALS):
                sim = simulate_fid(
                    fwhm_bins=fwhm, true_snr=snr, seed=SEED + 1000 * trial
                )
                tol = max(1.0, fwhm / 2.0) * sim["bin_mhz"]
                truth = sim["truth_mhz"]
                mf = matched_filter_candidates(
                    sim, tau_basis_us=2.0 * sim["tau_truth_us"]
                )
                bh = blackman_harris_candidates(sim)
                for name, cand in (("matched_filter", mf), ("blackman_harris", bh)):
                    r, f = _recall_fp(cand, truth, tol)
                    cell[name][0].append(r)
                    cell[name][1].append(f)
            for d in detectors:
                rows[d].append(float(np.mean(cell[d][0])))
                fprows[d].append(float(np.mean(cell[d][1])))
        for d in detectors:
            recall[d].append(rows[d])
            fp[d].append(fprows[d])
    return {
        "snr_axis": list(SWEEP_SNR),
        "fwhm_axis": list(SWEEP_FWHM_BINS),
        "recall": recall,
        "false_positives": fp,
    }


def projection_fft_identity(seed: int = SEED) -> Dict[str, float]:
    """Verify the projection == apodized-FFT identity to numerical tolerance.

    The unweighted correlation of the active-region signal with the
    matched-filter basis ``exp(-t/tau) * exp(-i 2 pi f_c t)`` at every bin
    frequency, computed as the explicit O(N^2) sum, equals the apodized FFT
    ``dt * rfft(exp(-t/tau) * fid)`` to floating-point precision. This is the
    load-bearing claim that makes the matched filter one FFT, not N
    projections.
    """
    rng = np.random.default_rng(seed)
    n = 512
    dt = 0.020
    tau = 3.0
    t = np.arange(n) * dt
    fid = rng.standard_normal(n)
    w = np.exp(-t / tau)
    apodized = w * fid
    fft_path = dt * np.fft.rfft(apodized)
    # Explicit projection at each rfft bin frequency.
    freqs = np.fft.rfftfreq(n, d=dt)
    basis = np.exp(-2j * np.pi * np.outer(freqs, t))  # (n_freq, n)
    proj_path = dt * basis @ apodized
    err = float(np.max(np.abs(fft_path - proj_path)))
    scale = float(np.max(np.abs(fft_path)))
    return {"max_abs_error": err, "scale": scale, "rel_error": err / scale}


# --------------------------------------------------------------------------
# 2638 application (shipped pipeline; gap pass vs primary pass on real data)
# --------------------------------------------------------------------------
def run_2638(workdir: Optional[Path] = None) -> Dict[str, Any]:
    """Build 2638 through Stage 5 and measure the gap pass's real-line yield.

    The Stage 5 fitted lines are the outside reference for "real". For each
    fit peak, find the nearest promoted Stage 3 peak within tolerance and note
    which pass found it. The metric is how many real lines are recovered by the
    primary pass alone vs primary + matched-filter gap pass, and how many are
    recovered *only* by the gap pass.
    """
    import tempfile

    import ftmwpipeline.api as ftmw

    tmp = workdir or Path(tempfile.mkdtemp(prefix="mf-note-"))
    tmp.mkdir(parents=True, exist_ok=True)
    path = str(tmp / "exp_2638.ftmw")
    ftmw.import_data(path, source=str(EXAMPLES / "2638"), force=True)
    ftmw.detect_start_time(path, band=TRIM, stamp=True)
    ftmw.compute_ft(path, trim=TRIM)
    ftmw.estimate_noise(path)
    ftmw.calibrate_tau(path)
    ftmw.detect_peaks(path)
    ftmw.assign_windows(path)
    fit = ftmw.fit_peaks(path)
    peaks = ftmw.load_peaks(path)

    promoted = [p for p in peaks if p.properties.get("promoted")]
    prim = np.array(
        [
            p.frequency
            for p in promoted
            if p.properties.get("detection_pass") == "primary"
        ],
        dtype=float,
    )
    gap = np.array(
        [p.frequency for p in promoted if p.properties.get("detection_pass") == "gap"],
        dtype=float,
    )

    fit_freqs = np.array([fp.frequency_mhz for fp in fit.fitted_peaks], dtype=float)
    # Tolerance: a few active-FT bins (active length ~12.65 us -> bin ~0.079 MHz).
    tol_mhz = 0.25

    def _recovered(ref: np.ndarray, cand: np.ndarray) -> np.ndarray:
        if cand.size == 0:
            return np.zeros(ref.size, dtype=bool)
        return np.array([np.min(np.abs(cand - f)) <= tol_mhz for f in ref], dtype=bool)

    by_primary = _recovered(fit_freqs, prim)
    by_gap = _recovered(fit_freqs, gap)
    by_both = by_primary | by_gap
    gap_only = by_gap & ~by_primary

    n_fit = int(fit_freqs.size)
    return {
        "n_fit_peaks": n_fit,
        "n_promoted_primary": int(prim.size),
        "n_promoted_gap": int(gap.size),
        "recall_primary_only": float(by_primary.sum() / n_fit) if n_fit else 0.0,
        "recall_primary_plus_gap": float(by_both.sum() / n_fit) if n_fit else 0.0,
        "n_gap_only_real_lines": int(gap_only.sum()),
        "gap_only_freqs": [float(f) for f in fit_freqs[gap_only]],
    }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
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
        POPPY,
        aggie_blue_cmap,
        apply_bare_style,
        apply_color_cycle,
    )

    apply_color_cycle(matplotlib)
    figdir = _figdir()
    sweep = results["synthetic_sweep"]
    snr_axis = np.array(sweep["snr_axis"], float)
    fwhm_axis = np.array(sweep["fwhm_axis"], float)
    mf_recall = np.array(sweep["recall"]["matched_filter"], float)
    bh_recall = np.array(sweep["recall"]["blackman_harris"], float)

    # fig1: matched-filter recall heatmap over SNR x FWHM/bin.
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        mf_recall,
        origin="lower",
        aspect="auto",
        cmap=aggie_blue_cmap(),
        vmin=0.0,
        vmax=1.0,
        extent=(0, len(fwhm_axis), 0, len(snr_axis)),
    )
    ax.set_xticks(np.arange(len(fwhm_axis)) + 0.5)
    ax.set_xticklabels([f"{x:g}" for x in fwhm_axis])
    ax.set_yticks(np.arange(len(snr_axis)) + 0.5)
    ax.set_yticklabels([f"{y:g}" for y in snr_axis])
    ax.set_xlabel("line width FWHM / bin")
    ax.set_ylabel("signal-to-noise")
    for i in range(len(snr_axis)):
        for j in range(len(fwhm_axis)):
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{mf_recall[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if mf_recall[i, j] > 0.5 else "black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, label="weak-line recall")
    apply_bare_style(ax)
    fig.tight_layout()
    fig.savefig(figdir / "fig1_mf_recall_heatmap.png", dpi=130)
    plt.close(fig)

    # fig2: recall vs SNR at the narrow-line regime real data sits in.
    j = list(fwhm_axis).index(1.34)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        snr_axis,
        mf_recall[:, j],
        "o-",
        color=AGGIE_BLUE,
        lw=1.8,
        label="matched filter",
    )
    ax.plot(
        snr_axis,
        bh_recall[:, j],
        "s-",
        color=POPPY,
        lw=1.8,
        label="Blackman-Harris primary",
    )
    ax.set_xlabel("signal-to-noise")
    ax.set_ylabel("weak-line recall (FWHM/bin = 1.34)")
    ax.set_ylim(-0.03, 1.03)
    apply_bare_style(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "fig2_recall_vs_snr.png", dpi=130)
    plt.close(fig)

    # fig3: 2638 real-line recovery, primary alone vs primary + gap. The axis is
    # truncated near the top (disclosed in the caption) so the +13-line lift on an
    # already-high primary recall is legible; bars carry the absolute counts.
    if "2638" in results:
        d = results["2638"]
        n_fit = d["n_fit_peaks"]
        vals = [d["recall_primary_only"], d["recall_primary_plus_gap"]]
        counts = [int(round(v * n_fit)) for v in vals]
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ["primary\nonly", "primary +\nmatched-filter gap"]
        ax.bar(bars, vals, color=[POPPY, AGGIE_GOLD], width=0.6)
        for i, (v, c) in enumerate(zip(vals, counts)):
            ax.text(
                i,
                v + 0.001,
                f"{v:.3f}\n{c}/{n_fit}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        ax.set_ylabel("fraction of Stage 5 fit peaks recovered")
        ax.set_ylim(0.90, 1.0)
        apply_bare_style(ax)
        fig.tight_layout()
        fig.savefig(figdir / "fig3_2638_gap_recovery.png", dpi=130)
        plt.close(fig)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run(do_2638: bool = True, figures: bool = True) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    results["identity"] = projection_fft_identity()
    print(f"identity: rel_error={results['identity']['rel_error']:.2e}", flush=True)
    results["synthetic_sweep"] = synthetic_sweep()
    mf = np.array(results["synthetic_sweep"]["recall"]["matched_filter"], float)
    bh = np.array(results["synthetic_sweep"]["recall"]["blackman_harris"], float)
    print(
        f"synthetic: MF recall min={mf.min():.2f} BH recall min={bh.min():.2f}",
        flush=True,
    )
    if do_2638:
        results["2638"] = run_2638()
        d = results["2638"]
        print(
            f"2638: primary {d['recall_primary_only']:.3f} -> +gap "
            f"{d['recall_primary_plus_gap']:.3f} "
            f"({d['n_gap_only_real_lines']} gap-only real lines)",
            flush=True,
        )
    if figures:
        make_figures(results)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-2638", action="store_true", help="skip the fixture build")
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
