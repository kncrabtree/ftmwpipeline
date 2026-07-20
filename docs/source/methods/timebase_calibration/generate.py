"""Regenerate the figures and key results for the timebase-calibration note.

This harness reproduces, from a fixed-seed simulator and the checked-in 2638
fixture, the quantitative claims in :doc:`timebase_calibration`:

* the **estimator recovers a known scale error** -- a synthetic record with
  Rb-locked lattice tones planted at ``k g (1 + eps_true)`` is calibrated back
  to ``eps_true`` within a fraction of its formal uncertainty, and an
  off-nominal tone is rejected by the shared-eps consistency test;
* the **Cramer-Rao position bound** -- each tone's reported uncertainty equals
  the single-tone ML bound ``sigma_f = (sqrt(6)/pi) / (T snr)`` (with the
  systematic floor switched off), the relation that makes the phase read so
  much sharper than a line width;
* the **2638 application** -- the measured ``epsilon`` and its uncertainty, the
  lattice spacing, and the kept/rejected/drift-control tone counts, plus the
  two embedded figures (the phase-ramp measurement on the strongest tone and
  the per-tone shared-eps scatter).

The estimator, the block demodulation, and the consistency rejection are the
shipped ``ftmwpipeline.fitting.timebase_calibration`` functions, so the note
validates the production code.

Run as a script to (re)write ``figures/*.png`` and ``results.json`` beside this
file::

    python generate.py                 # synthetic + 2638 (a few seconds)
    python generate.py --no-2638       # synthetic only

The committed ``results.json`` is the regression target for
``tests/integration/test_timebase_calibration_report.py`` (the ``slow``
marker); the figures are embedded by the note. ``synthetic_recovery`` and
``crb_check`` back the fast unit checks in
``tests/unit/fitting/test_timebase_calibration_invariants.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ftmwpipeline.fitting.timebase_calibration import (
    TimebaseCalibrationResult,
    calibrate_timebase_from_fid,
)

SEED = 20260622
EXAMPLES = Path("examples/blackchirp_data")
TRIM = (26500.0, 40000.0)

# Synthetic record: three decades of Rb-locked lattice tones on a 320 MHz comb,
# stretched by a known scale error, plus a decaying molecular line and an
# off-nominal spur that the consistency test must reject.
G = 320.0  # MHz lattice fundamental (gcd of the locked clocks below)
EPS_TRUE = 2.0e-6
DT_US = 2.0e-5  # 50 GSa/s
N = 200_000
LOCKED = [5120.0, 5440.0]  # gcd = 320 MHz
CRB_COEFF = np.sqrt(6.0) / np.pi  # ML single-tone position-bound coefficient


def _synthetic_record(
    *,
    eps: float = EPS_TRUE,
    seed: int = 0,
    noise: float = 0.05,
    off_nominal_offset_mhz: float = -5.0e-3,
) -> np.ndarray:
    """Build a synthetic FID: locked lattice tones, a molecular line, a spur.

    The locked tones (k = 10, 20, 30, 40, 56) are scaled by ``(1 + eps)``; a
    decaying off-lattice tone tests robustness; one fundamental tone carries a
    fixed kHz offset that is *not* eps-scaled (the rejection target).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N) * DT_US
    x = np.zeros(N, dtype=float)
    for k, amp in ((10, 5.0), (20, 4.5), (30, 4.0), (40, 5.5), (56, 6.0)):
        f = k * G * (1.0 + eps)
        x += amp * np.cos(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    tau = 1.0
    x += (
        8.0
        * np.exp(-t / tau)
        * np.cos(2 * np.pi * 3360.0 * t + rng.uniform(0, 2 * np.pi))
    )
    f_off = G + off_nominal_offset_mhz
    x += 3.5 * np.cos(2 * np.pi * f_off * t + rng.uniform(0, 2 * np.pi))
    x += rng.normal(0.0, noise, size=N)
    return x


def synthetic_recovery(seed: int = SEED) -> Dict[str, Any]:
    """Recover ``eps_true`` from the synthetic record at a few noise levels."""
    out: Dict[str, Any] = {"eps_true": EPS_TRUE, "noise": [], "recovered": []}
    for noise in (0.05, 0.15, 0.45):
        result = calibrate_timebase_from_fid(
            _synthetic_record(seed=seed, noise=noise),
            DT_US,
            start_us=0.0,
            end_us=N * DT_US,
            locked_freqs_mhz=LOCKED,
        )
        out["noise"].append(noise)
        out["recovered"].append(
            {
                "epsilon": result.epsilon,
                "sigma_epsilon": result.sigma_epsilon,
                "n_used": result.n_used,
                "n_detected": result.n_detected,
            }
        )
    return out


def crb_check(seed: int = SEED) -> Dict[str, Any]:
    """Each kept tone's reported sigma equals the ML position bound.

    With the systematic floor switched off (``kappa_sys = 0``) the per-tone
    uncertainty is purely ``sigma_f = (sqrt(6)/pi) / (T snr)``; the ratio of the
    reported value to that bound is one for every tone.
    """
    result = calibrate_timebase_from_fid(
        _synthetic_record(seed=seed),
        DT_US,
        start_us=0.0,
        end_us=N * DT_US,
        locked_freqs_mhz=LOCKED,
        kappa_sys=0.0,
    )
    T = result.span_us
    ratios = [
        tr.sigma_mhz / (CRB_COEFF / (T * tr.snr))
        for tr in result.tone_reads
        if tr.used and tr.snr > 0
    ]
    return {"span_us": T, "ratios": ratios, "n": len(ratios)}


def _build_2638(
    workdir: Optional[Path] = None,
) -> Tuple[TimebaseCalibrationResult, str]:
    """Build 2638 far enough to calibrate the timebase; return result + path.

    Calibration needs only the raw FID and the persisted active-region bounds,
    so the fixture is taken through start detection and the FT, not the full
    fit. The Blackchirp loader populates the clock declaration at import.
    """
    import tempfile

    import ftmwpipeline.api as ftmw

    tmp = workdir or Path(tempfile.mkdtemp(prefix="timebase-note-"))
    tmp.mkdir(parents=True, exist_ok=True)
    path = str(tmp / "exp_2638.ftmw")
    ftmw.import_data(path, source=str(EXAMPLES / "2638"), force=True)
    ftmw.detect_start_time(path, band=TRIM, stamp=True)
    ftmw.compute_ft(path, trim=TRIM)
    result = ftmw.calibrate_timebase(path)
    return result, path


def _summarize_2638(result: TimebaseCalibrationResult) -> Dict[str, Any]:
    """Scalar regression summary from the 2638 calibration."""
    kept = [t for t in result.tone_reads if t.used]
    rejected = [t for t in result.tone_reads if not t.used and not t.drift_control]
    controls = [t for t in result.tone_reads if t.drift_control]
    strongest = max(kept, key=lambda t: t.snr) if kept else None
    return {
        "epsilon_ppm": result.epsilon * 1e6,
        "sigma_epsilon_ppm": result.sigma_epsilon * 1e6,
        "lattice_g_mhz": result.lattice_g_mhz,
        "span_us": result.span_us,
        "n_used": len(kept),
        "n_detected": result.n_detected,
        "n_rejected": len(rejected),
        "n_controls": len(controls),
        "f_bb_max_mhz": max((t.f_bb_mhz for t in result.tone_reads), default=0.0),
        "strongest_f_bb_mhz": float(strongest.f_bb_mhz) if strongest else 0.0,
        "strongest_snr": float(strongest.snr) if strongest else 0.0,
        "strongest_df_khz": float(strongest.df_mhz * 1e3) if strongest else 0.0,
    }


def run_2638(workdir: Optional[Path] = None) -> Dict[str, Any]:
    """Calibrate the timebase on the reference experiment; return the summary."""
    result, _ = _build_2638(workdir)
    return _summarize_2638(result)


def _figdir() -> Path:
    d = Path(__file__).resolve().parent / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _plot_timebase(result: Any) -> Any:
    """Per-tone shared-eps scatter with the fitted-epsilon line and its band."""
    import matplotlib.pyplot as plt

    from ftmwpipeline.visualization.report_style import (
        AGGIE_BLUE,
        DOUBLE_DECKER,
        POPPY,
        apply_bare_style,
    )

    tones = list(result.tone_reads)
    kept = [t for t in tones if t.used]
    rej = [t for t in tones if not t.used]
    eps = result.epsilon
    sig = result.sigma_epsilon

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    # Offset in kHz = epsilon * f_bb_MHz * 1e3, drawn with its 1-sigma band.
    f_max = max((t.f_bb_mhz for t in tones), default=1.0)
    xs = np.linspace(0.0, f_max * 1.02, 200)
    ax.fill_between(
        xs / 1e3,
        (eps - sig) * xs * 1e3,
        (eps + sig) * xs * 1e3,
        color=AGGIE_BLUE,
        alpha=0.15,
        linewidth=0,
    )
    ax.plot(
        xs / 1e3,
        eps * xs * 1e3,
        color=AGGIE_BLUE,
        linewidth=1.6,
        label=rf"$\varepsilon = {eps*1e6:+.2f} \pm {sig*1e6:.2f}$ ppm",
    )
    if rej:
        ax.scatter(
            [t.f_bb_mhz / 1e3 for t in rej],
            [t.df_mhz * 1e3 for t in rej],
            facecolors="none",
            edgecolors=POPPY,
            s=34,
            linewidths=1.2,
            label="rejected (off-line / inconsistent)",
        )
    if kept:
        ax.scatter(
            [t.f_bb_mhz / 1e3 for t in kept],
            [t.df_mhz * 1e3 for t in kept],
            color=DOUBLE_DECKER,
            s=38,
            label="kept (shared-$\\varepsilon$ fit)",
            zorder=3,
        )
    ax.axhline(0.0, color="#9aa3ad", linewidth=0.8, zorder=0)
    ax.set_xlabel("Baseband frequency (GHz)")
    ax.set_ylabel("Measured tone offset (kHz)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    apply_bare_style(ax)
    return fig


def _plot_phase_ramp(path: str, result: Any) -> Any:
    """The phase-ramp measurement on the strongest clock tone.

    Reconstructs the calibration's active segment and demodulates it at the
    tone's nominal baseband frequency (exactly as ``calibrate_timebase_from_fid``
    does), then shows the residual rotation as (left) the block-averaged phase
    advancing linearly in time and (right) the coherent-sum scan whose peak is
    the measured offset.
    """
    import matplotlib.pyplot as plt

    from ftmwpipeline._internal.stage0_impl import load_fid_from_pipeline_impl
    from ftmwpipeline.fitting.timebase_calibration import _build_block_demod
    from ftmwpipeline.visualization.report_style import (
        AGGIE_BLUE,
        DOUBLE_DECKER,
        POPPY,
        apply_bare_style,
    )

    tone = max((t for t in result.tone_reads if t.used), key=lambda t: t.snr)
    fbb = float(tone.f_bb_mhz)

    fid = load_fid_from_pipeline_impl(path)
    x = np.asarray(fid.data, dtype=float)
    dt = float(result.sample_dt_us)
    i0 = max(int(result.start_us / dt), 0)
    i1 = min(int(result.end_us / dt), x.size)
    seg = x[i0:i1]
    t = (np.arange(i0, i1) - i0) * dt

    z = seg * np.exp(-2j * np.pi * fbb * t)

    grid, _t_blocks, scan_matrix = _build_block_demod(t, 4096, 0.1, 0.0001)
    zb = np.array([c.mean() for c in np.array_split(z, 4096)])
    amp = np.abs(scan_matrix @ zb)
    df = float(grid[int(np.argmax(amp))])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)

    # Left: the residual phase ramp. The demodulated tone rotates at the offset,
    # so the slope of its phase in time is the offset. For a positive offset df
    # the phase advances as +360*df*t degrees; coarser blocks only smooth it.
    n_disp = 160
    zc = np.array([c.mean() for c in np.array_split(z, n_disp)])
    tc = np.array([c.mean() for c in np.array_split(t, n_disp)])
    phase_deg = np.degrees(np.unwrap(np.angle(zc)))
    slope_line = phase_deg[0] + 360.0 * df * (tc - tc[0])
    axL.plot(
        tc,
        phase_deg,
        color=DOUBLE_DECKER,
        linewidth=1.4,
        label=f"residual phase ({fbb/1e3:.2f} GHz tone)",
    )
    axL.plot(
        tc,
        slope_line,
        color=AGGIE_BLUE,
        linewidth=1.2,
        linestyle="--",
        label=rf"slope $\Rightarrow \delta f = {df*1e3:+.1f}$ kHz",
    )
    axL.set_xlabel("Time (µs)")
    axL.set_ylabel("Demodulated phase (degrees)")
    axL.legend(loc="best", frameon=False, fontsize=9)
    apply_bare_style(axL)

    # Right: the coherent-sum scan. Its peak is the maximum-likelihood offset.
    axR.plot(grid * 1e3, amp / amp.max(), color=AGGIE_BLUE, linewidth=1.4)
    axR.axvline(0.0, color="#9aa3ad", linewidth=0.8, label="nominal frequency")
    axR.axvline(
        df * 1e3,
        color=POPPY,
        linewidth=1.2,
        linestyle="--",
        label=rf"peak $\Rightarrow \delta f = {df*1e3:+.1f}$ kHz",
    )
    axR.set_xlabel("Candidate offset from nominal (kHz)")
    axR.set_ylabel("Coherent-sum amplitude (normalized)")
    axR.legend(loc="upper left", frameon=False, fontsize=9)
    apply_bare_style(axR)
    return fig


def make_figures(result: Any, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")

    from ftmwpipeline.visualization.report_style import apply_color_cycle

    apply_color_cycle(matplotlib)
    figdir = _figdir()
    fig_tb = _plot_timebase(result)
    fig_tb.savefig(figdir / "clock_timebase.png", dpi=130, bbox_inches="tight")
    fig_pr = _plot_phase_ramp(path, result)
    fig_pr.savefig(figdir / "clock_phase_ramp.png", dpi=130, bbox_inches="tight")


def run(do_2638: bool = True, figures: bool = True) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    results["synthetic_recovery"] = synthetic_recovery()
    results["crb_check"] = crb_check()
    rec = results["synthetic_recovery"]["recovered"][0]
    print(
        f"synthetic: eps {rec['epsilon']*1e6:+.3f} ppm vs truth "
        f"{EPS_TRUE*1e6:+.3f} ppm; CRB ratio "
        f"{np.mean(results['crb_check']['ratios']):.3f}",
        flush=True,
    )
    if do_2638:
        result, path = _build_2638()
        results["2638"] = _summarize_2638(result)
        d = results["2638"]
        print(
            f"2638: eps {d['epsilon_ppm']:+.3f} +/- {d['sigma_epsilon_ppm']:.3f} ppm; "
            f"g={d['lattice_g_mhz']:.0f} MHz; {d['n_used']} kept / "
            f"{d['n_rejected']} rejected / {d['n_controls']} controls",
            flush=True,
        )
        if figures:
            make_figures(result, path)
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
