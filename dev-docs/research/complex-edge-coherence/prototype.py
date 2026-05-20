"""
Complex-edge coherence statistic — research prototype.

Reproducibility script for the report in ``report.md``. Calibrates the
candidate edge statistics against synthetic ground truth and the 2638
fixture, then regenerates every figure in ``figures/``.

Run from the repo root via the project conda env:

    conda run -n ftmwpipeline-dev python \\
        dev-docs/research/complex-edge-coherence/prototype.py

Inputs: the 2638 fixture at ``scratch/exp_2638.ftmw`` (untracked; rebuild via
``ftmw.import_data(...) → ftmw.compute_ft(...)`` from
``examples/blackchirp_data/2638/`` if missing).

Outputs land in ``figures/`` alongside this script. The synthetic and 2638
sweep ``.npz`` blobs are written next to the script; they are large
(~50 MB combined) and intentionally gitignored — the figures and this
script are what the report depends on.

Math conventions
----------------
Spectrum model for a damped cosine x(t) = A cos(2π f0 t + φ) e^{-t/τ}
truncated to [0, T] is the rfft-domain response near +f0:

    X(f) ≈ (A/2) · e^{iφ} · h_T(f - f0; τ)
    h_T(Δf; τ) = [1 - exp(-(1/τ + i 2π Δf) T)] / (1/τ + i 2π Δf)

At line centre |h_T(0)| = τ_eff = τ(1 - e^{-T/τ}); the leakage envelope
far from centre is |h_T(Δf)| ≈ (1 + e^{-T/τ}) / (2π |Δf|) — the same
model used by ``preprocessing/leakage.py`` for the leakage-reach
predictor.

Complex Gaussian noise convention: each spectrum bin is
N(0, σ/√2) + i N(0, σ/√2), so E[|n|²] = σ². ``σ`` here is the per-bin
complex RMS, matching the per-point noise the noise-estimation stage
reports on the persisted user grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ----- physical defaults broadly matching the 2638 fixture -----------------
DEFAULT_T_US = 15.0       # FID duration
DEFAULT_FS_MHZ = 50.0     # 1/dt; 2638-ish digitizer
DEFAULT_N = 750           # samples in the FID; df = 1/T_us → MHz spacing
RNG = np.random.default_rng(20260519)

HERE = Path(__file__).parent
FIGDIR = HERE / "figures"
# dev-docs/research/<topic>/prototype.py → repo root = parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = REPO_ROOT / "scratch" / "exp_2638.ftmw"


# ---------------------------------------------------------------------------
# Spectrum model
# ---------------------------------------------------------------------------
def finite_T_response(delta_f_mhz: np.ndarray, T_us: float, tau_us: float | None) -> np.ndarray:
    """Complex h_T(Δf; τ) on a frequency-offset grid Δf in MHz."""
    T = T_us * 1e-6
    df_hz = delta_f_mhz * 1e6
    if tau_us is None:
        # Boxcar/undamped limit: h_T(Δf) = T · sinc(Δf · T) · exp(-i π Δf T)
        # Equivalent to the closed-form with 1/τ → 0; do it directly for stability.
        x = np.pi * df_hz * T
        # np.sinc(y) = sin(πy)/(πy); we want sin(x)/x = sinc(x/π)
        sinc = np.sinc(df_hz * T)
        return T * sinc * np.exp(-1j * x)
    tau = tau_us * 1e-6
    denom = (1.0 / tau) + 1j * 2.0 * np.pi * df_hz
    return (1.0 - np.exp(-denom * T)) / denom


@dataclass
class Line:
    f_mhz: float
    amp: float
    phase: float = 0.0
    tau_us: float | None = None


def make_spectrum(
    freqs_mhz: np.ndarray,
    lines: Iterable[Line],
    T_us: float,
    sigma: float,
    rng: np.random.Generator,
    pedestal: complex = 0.0 + 0.0j,
) -> np.ndarray:
    """Sum of analytic line responses + complex Gaussian noise on the grid."""
    X = np.zeros_like(freqs_mhz, dtype=np.complex128)
    for ln in lines:
        h = finite_T_response(freqs_mhz - ln.f_mhz, T_us, ln.tau_us)
        X += 0.5 * ln.amp * np.exp(1j * ln.phase) * h
    if pedestal != 0:
        X += pedestal
    n = rng.normal(0.0, sigma / np.sqrt(2.0), size=X.shape) + 1j * rng.normal(
        0.0, sigma / np.sqrt(2.0), size=X.shape
    )
    return X + n


def grid(T_us: float = DEFAULT_T_US, n_zpf: int = 2, fs_mhz: float = DEFAULT_FS_MHZ, N: int = DEFAULT_N) -> np.ndarray:
    """Frequency grid (MHz) for an N-sample FID, zero-padded by ``n_zpf`` and rfft'd.

    rfft frequency spacing = 1/(N_pad · dt). Returns positive frequencies only.
    """
    N_pad = N * n_zpf
    dt = 1.0 / (fs_mhz * 1e6)
    return np.fft.rfftfreq(N_pad, d=dt) / 1e6  # MHz


# ---------------------------------------------------------------------------
# Statistic variants
# ---------------------------------------------------------------------------
StatFn = Callable[[np.ndarray, float], float]


def stat_coherent_sum(z: np.ndarray, sigma: float) -> float:
    """|Σ z| / (σ · √M). Mean ~ √(π/4) ≈ 0.886 on clean complex Gaussian."""
    return float(np.abs(np.sum(z)) / (sigma * np.sqrt(len(z))))


def stat_max_cumsum(z: np.ndarray, sigma: float) -> float:
    """max_{1≤t≤M} |Σ_{i≤t} z| / (σ · √t). CUSUM-style local-coherence detector."""
    csum = np.cumsum(z)
    ts = np.arange(1, len(z) + 1, dtype=float)
    return float(np.max(np.abs(csum) / (sigma * np.sqrt(ts))))


def stat_realimag_z(z: np.ndarray, sigma: float) -> float:
    """max(|Σ Re|, |Σ Im|) / (σ · √(M/2)). Per-component z-score."""
    re = np.abs(np.sum(z.real))
    im = np.abs(np.sum(z.imag))
    return float(max(re, im) / (sigma * np.sqrt(len(z) / 2.0)))


STATS: dict[str, StatFn] = {
    "coherent_sum": stat_coherent_sum,
    "max_cumsum": stat_max_cumsum,
    "realimag_z": stat_realimag_z,
}


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------
@dataclass
class SweepResult:
    name: str
    rows: list[dict]  # tagged scalar rows; small enough to keep in-memory


def edge_band(freqs: np.ndarray, X: np.ndarray, side: str, M: int) -> np.ndarray:
    """Take an M-point band at the low (`'low'`) or high (`'high'`) edge."""
    if side == "low":
        return X[:M]
    return X[-M:]


def run_synthetic_sweep(
    n_trials: int = 200,
    snr_grid: tuple[float, ...] = (10.0, 30.0, 100.0, 300.0, 1000.0),
    tau_grid: tuple[float | None, ...] = (None, 30.0, 10.0, 3.0),
    distance_grid_mhz: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0),
    M_grid: tuple[int, ...] = (8, 16, 32, 64, 128, 256),
) -> SweepResult:
    """Cases (a) clean / (b) out-of-band line / (c) in-band centred / (d) pedestal.

    Returns one row per (case, trial, sweep-point) combination. For each row,
    every statistic in ``STATS`` is evaluated on the edge band.
    """
    freqs = grid()
    df = float(np.median(np.diff(freqs)))
    # Place the FT subband generously inside the rfft range so both edges live in the band.
    f_lo, f_hi = 2.0, 22.0  # MHz
    band = (freqs >= f_lo) & (freqs <= f_hi)
    fband = freqs[band]
    rows: list[dict] = []

    # Treat σ as the per-bin complex RMS; SNR is line peak height / σ.
    sigma = 1.0
    T = DEFAULT_T_US

    for tau_us in tau_grid:
        h0 = finite_T_response(np.array([0.0]), T, tau_us)[0]
        peak_per_amp = 0.5 * np.abs(h0)  # |X(f0)| for unit-amplitude line
        for snr in snr_grid:
            amp = snr * sigma / peak_per_amp
            for M in M_grid:
                if M >= 0.4 * len(fband):
                    continue  # band too small for this M
                # ---- (a) clean ----
                for trial in range(n_trials):
                    X = make_spectrum(fband, [], T, sigma, RNG)
                    for sname, sfn in STATS.items():
                        rows.append(
                            dict(
                                case="clean",
                                tau_us=tau_us,
                                snr=snr,
                                M=M,
                                distance_mhz=np.nan,
                                stat=sname,
                                value=sfn(edge_band(fband, X, "low", M), sigma),
                                trial=trial,
                            )
                        )
                # ---- (b) out-of-band line above the band ----
                for d in distance_grid_mhz:
                    f_oob = f_hi + d
                    line = Line(f_mhz=f_oob, amp=amp, phase=0.0, tau_us=tau_us)
                    for trial in range(n_trials):
                        X = make_spectrum(fband, [line], T, sigma, RNG)
                        for sname, sfn in STATS.items():
                            rows.append(
                                dict(
                                    case="oob_high",
                                    tau_us=tau_us,
                                    snr=snr,
                                    M=M,
                                    distance_mhz=d,
                                    stat=sname,
                                    value=sfn(edge_band(fband, X, "high", M), sigma),
                                    trial=trial,
                                )
                            )
                # ---- (c) in-band centred line at the band middle ----
                f_centre = 0.5 * (f_lo + f_hi)
                line_c = Line(f_mhz=f_centre, amp=amp, phase=0.0, tau_us=tau_us)
                for trial in range(n_trials):
                    X = make_spectrum(fband, [line_c], T, sigma, RNG)
                    # Use the same low-edge band as (a); for case (c) the
                    # statistic only sees the centred sinc's distant skirt.
                    for sname, sfn in STATS.items():
                        rows.append(
                            dict(
                                case="inband_centre",
                                tau_us=tau_us,
                                snr=snr,
                                M=M,
                                distance_mhz=f_centre - f_lo,
                                stat=sname,
                                value=sfn(edge_band(fband, X, "low", M), sigma),
                                trial=trial,
                            )
                        )

    # ---- (d) injected flat pedestal (single sweep over magnitude) ----
    for ped_mag in (0.0, 0.5, 1.0, 3.0, 10.0):
        for M in M_grid:
            if M >= 0.4 * len(fband):
                continue
            for trial in range(n_trials):
                X = make_spectrum(fband, [], T, sigma, RNG, pedestal=ped_mag + 0.0j)
                for sname, sfn in STATS.items():
                    rows.append(
                        dict(
                            case="pedestal",
                            tau_us=None,
                            snr=ped_mag,  # repurpose: pedestal magnitude in σ units
                            M=M,
                            distance_mhz=np.nan,
                            stat=sname,
                            value=sfn(edge_band(fband, X, "low", M), sigma),
                            trial=trial,
                        )
                    )

    _ = df  # silence linter
    return SweepResult(name="synthetic", rows=rows)


# ---------------------------------------------------------------------------
# Plotting helpers (synthetic)
# ---------------------------------------------------------------------------
def _agg(rows: list[dict], **filters) -> np.ndarray:
    out = []
    for r in rows:
        ok = True
        for k, v in filters.items():
            if r[k] != v:
                ok = False
                break
        if ok:
            out.append(r["value"])
    return np.asarray(out)


def plot_spatial_profile(outdir: Path) -> None:
    """One-shot spatial-profile demo: out-of-band line vs in-band centred line.

    Shows the rolling ``coherent_sum`` statistic across the band for a
    *high-SNR* case where both leakage skirts are well above noise. The OOB
    profile is monotone toward the line edge; the centred profile is
    peaked at the centre and decays both ways.
    """
    freqs = grid()
    f_lo, f_hi = 2.0, 22.0
    band = (freqs >= f_lo) & (freqs <= f_hi)
    fband = freqs[band]
    sigma = 1.0
    T = DEFAULT_T_US
    tau_us = None
    snr = 5000.0  # large enough that the centred line's far skirt clears noise

    h0 = finite_T_response(np.array([0.0]), T, tau_us)[0]
    amp = snr * sigma / (0.5 * np.abs(h0))

    oob_line = Line(f_mhz=f_hi + 0.5, amp=amp, phase=0.0, tau_us=tau_us)
    centre_line = Line(f_mhz=0.5 * (f_lo + f_hi), amp=amp, phase=0.0, tau_us=tau_us)

    X_oob = make_spectrum(fband, [oob_line], T, sigma, RNG)
    X_centre = make_spectrum(fband, [centre_line], T, sigma, RNG)

    M = 32
    def rolling(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        win = np.ones(M)
        s = np.convolve(X.real, win, mode="valid") + 1j * np.convolve(X.imag, win, mode="valid")
        return fband[M // 2 : M // 2 + len(s)], np.abs(s) / (sigma * np.sqrt(M))

    f_oob, stat_oob = rolling(X_oob)
    f_centre, stat_centre = rolling(X_centre)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    ax1.plot(fband, np.abs(X_oob), lw=0.5, color="C1", label="|X| oob")
    ax1.plot(fband, np.abs(X_centre), lw=0.5, color="C2", label="|X| centre")
    ax1.set_yscale("log")
    ax1.set_ylabel("|X|")
    ax1.legend(fontsize=8)
    ax2.plot(f_oob, stat_oob, lw=0.8, color="C1", label="OOB line at f=22.5")
    ax2.plot(f_centre, stat_centre, lw=0.8, color="C2", label="centred line at f=12")
    ax2.axhline(3.0, color="k", ls="--", lw=0.8, label="thr=3")
    ax2.set_yscale("log")
    ax2.set_xlabel("frequency in band (MHz)")
    ax2.set_ylabel("coherent_sum statistic")
    ax2.set_title(f"shape: OOB monotone vs centred-line peaked  (SNR={snr:g}, M={M})")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "03b_spatial_profile.png", dpi=130)
    plt.close(fig)


def plot_synthetic(result: SweepResult, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    stats = list(STATS)
    # Panel 1: clean-edge null distribution by M, per statistic
    fig, axes = plt.subplots(1, len(stats), figsize=(4.5 * len(stats), 3.6), sharey=True)
    Ms = sorted({r["M"] for r in result.rows if r["case"] == "clean"})
    for ax, sname in zip(axes, stats):
        for M in Ms:
            vals = _agg(result.rows, case="clean", M=M, stat=sname)
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=40, alpha=0.4, label=f"M={M}", density=True)
        ax.set_title(f"clean edge — {sname}")
        ax.set_xlabel("statistic")
        ax.axvline(3.0, color="k", ls="--", lw=0.8, label="thr=3")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("density")
    fig.tight_layout()
    fig.savefig(outdir / "01_clean_null_distributions.png", dpi=130)
    plt.close(fig)

    # Panel 2: out-of-band power vs distance, by SNR (tau = None for clarity)
    fig, axes = plt.subplots(1, len(stats), figsize=(4.5 * len(stats), 3.6))
    Mfix = 32
    for ax, sname in zip(axes, stats):
        for snr in sorted({r["snr"] for r in result.rows if r["case"] == "oob_high"}):
            ds, means = [], []
            for d in sorted(
                {r["distance_mhz"] for r in result.rows if r["case"] == "oob_high"}
            ):
                vals = _agg(
                    result.rows,
                    case="oob_high",
                    snr=snr,
                    M=Mfix,
                    distance_mhz=d,
                    stat=sname,
                    tau_us=None,
                )
                if len(vals) == 0:
                    continue
                ds.append(d)
                means.append(np.median(vals))
            ax.plot(ds, means, marker="o", label=f"SNR={snr:g}")
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_title(f"out-of-band line — {sname}  (M={Mfix}, τ=∞)")
        ax.set_xlabel("edge–line distance (MHz)")
        ax.set_ylabel("median statistic")
        ax.axhline(3.0, color="k", ls="--", lw=0.8)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "02_oob_distance_sweep.png", dpi=130)
    plt.close(fig)

    # Panel 3: in-band centred vs out-of-band — shape discrimination at fixed SNR
    fig, axes = plt.subplots(1, len(stats), figsize=(4.5 * len(stats), 3.6), sharey=True)
    snr_fix = 100.0
    Ms_fix = sorted({r["M"] for r in result.rows if r["case"] == "inband_centre"})
    for ax, sname in zip(axes, stats):
        # Compare distributions of the statistic for (b) at the *closest* distance
        # vs (c) at its (one fixed) centre distance, vs (a).
        a_vals = _agg(result.rows, case="clean", M=Mfix, stat=sname)
        b_vals = _agg(
            result.rows,
            case="oob_high",
            snr=snr_fix,
            M=Mfix,
            distance_mhz=min(
                {r["distance_mhz"] for r in result.rows if r["case"] == "oob_high"}
            ),
            stat=sname,
            tau_us=None,
        )
        c_vals = _agg(
            result.rows,
            case="inband_centre",
            snr=snr_fix,
            M=Mfix,
            stat=sname,
            tau_us=None,
        )
        bins = np.linspace(0, max(a_vals.max(initial=1), b_vals.max(initial=1), c_vals.max(initial=1)), 50)
        ax.hist(a_vals, bins=bins, alpha=0.4, label="clean (a)", density=True)
        ax.hist(b_vals, bins=bins, alpha=0.4, label="oob @ min d (b)", density=True)
        ax.hist(c_vals, bins=bins, alpha=0.4, label="centre line (c)", density=True)
        ax.set_title(f"shape discrim — {sname}  (M={Mfix}, SNR={snr_fix:g}, τ=∞)")
        ax.set_xlabel("statistic")
        ax.axvline(3.0, color="k", ls="--", lw=0.8)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("density")
    fig.tight_layout()
    fig.savefig(outdir / "03_shape_discrim.png", dpi=130)
    plt.close(fig)

    # Panel 4: pedestal sensitivity
    fig, axes = plt.subplots(1, len(stats), figsize=(4.5 * len(stats), 3.6))
    for ax, sname in zip(axes, stats):
        mags = sorted({r["snr"] for r in result.rows if r["case"] == "pedestal"})
        for M in Ms_fix:
            ys = []
            for m in mags:
                vals = _agg(result.rows, case="pedestal", snr=m, M=M, stat=sname)
                if len(vals) == 0:
                    ys.append(np.nan)
                else:
                    ys.append(np.median(vals))
            ax.plot(mags, ys, marker="o", label=f"M={M}")
        ax.set_title(f"pedestal — {sname}")
        ax.set_xlabel("pedestal magnitude (σ units)")
        ax.set_ylabel("median statistic")
        ax.axhline(3.0, color="k", ls="--", lw=0.8)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "04_pedestal_sensitivity.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2638 real-data sweep
# ---------------------------------------------------------------------------
def run_2638(figdir: Path, npz_dir: Path) -> None:
    """Evaluate the chosen statistic along the 2638 user-grid spectrum."""
    from ftmwpipeline import api as ftmw

    pipeline_path = FIXTURE_PATH
    if not pipeline_path.exists():
        print(f"[2638] skipping — {pipeline_path} not found")
        return

    # Use the canonical persisted settings (Stage 3+ contract). compute_ft with
    # no overrides resolves from the file's persisted FTSettings.
    ft = ftmw.compute_ft(str(pipeline_path))
    noise = ftmw.estimate_noise(str(pipeline_path))

    freqs = np.asarray(ft.freq_array, dtype=float)
    spec = np.asarray(ft.complex_spectrum, dtype=np.complex128)
    sigma_per_point_arr = np.asarray(noise.rms_noise, dtype=float)  # frequency-dependent
    sigma_med = float(np.median(sigma_per_point_arr))
    print(
        f"[2638] points={len(freqs):,}, range={freqs[0]:.1f}–{freqs[-1]:.1f} MHz, "
        f"σ median={sigma_med:.3g}, min/max={sigma_per_point_arr.min():.3g}/"
        f"{sigma_per_point_arr.max():.3g}"
    )

    # Rolling statistic with band width M.
    M = 64
    z = spec
    win = np.ones(M)
    # |Σ z over a length-M window| / (σ_local · √M); use local σ averaged in the window.
    sums = np.convolve(z.real, win, mode="valid") + 1j * np.convolve(z.imag, win, mode="valid")
    sigma_local = np.convolve(sigma_per_point_arr, win / M, mode="valid")  # mean σ in window
    stat_vals = np.abs(sums) / (sigma_local * np.sqrt(M))
    f_centres = freqs[M // 2 : M // 2 + len(stat_vals)]

    fig, axes = plt.subplots(3, 1, figsize=(13, 7.5), sharex=True)
    axes[0].plot(freqs, np.abs(spec), lw=0.5, color="k")
    axes[0].set_ylabel("|FT|")
    axes[0].set_yscale("log")
    axes[0].set_title("2638 spectrum (magnitude, log)")
    axes[1].plot(freqs, spec.real, lw=0.4, label="Re", color="C0")
    axes[1].plot(freqs, spec.imag, lw=0.4, label="Im", color="C1", alpha=0.7)
    axes[1].set_ylabel("Re/Im")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[2].plot(f_centres, stat_vals, lw=0.6, color="C3")
    axes[2].axhline(3.0, color="k", ls="--", lw=0.8, label="thr=3")
    axes[2].set_yscale("log")
    axes[2].set_ylabel(f"|Σ z| / (σ·√M),  M={M}")
    axes[2].set_xlabel("frequency (MHz)")
    axes[2].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "05_2638_statistic.png", dpi=130)
    plt.close(fig)

    # A focused look at one strong-line skirt and one isolated quiet region.
    # Pick the global max-magnitude point and a quiet region by lowest local max.
    i_strong = int(np.argmax(np.abs(spec)))
    f_strong = freqs[i_strong]
    # Quiet: pick the bin where rolling stat is lowest, with at least M/2 spacing from strong.
    safe = np.abs(f_centres - f_strong) > 200.0  # MHz
    if np.any(safe):
        i_quiet = int(np.argmin(np.where(safe, stat_vals, np.inf)))
        f_quiet = f_centres[i_quiet]
    else:
        f_quiet = freqs[len(freqs) // 2]
    print(f"[2638] strongest line @ {f_strong:.2f} MHz, quiet sample @ {f_quiet:.2f} MHz")

    # Zoomed panels: ±50 MHz around each, with statistic overlay
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5))
    for ax, fc, label in [(axes[0], f_strong, "strong-line neighbourhood"),
                           (axes[1], f_quiet, "quiet region")]:
        m = (freqs > fc - 50) & (freqs < fc + 50)
        ax.plot(freqs[m], np.abs(spec)[m], lw=0.7, color="k", label="|FT|")
        ax2 = ax.twinx()
        m2 = (f_centres > fc - 50) & (f_centres < fc + 50)
        ax2.plot(f_centres[m2], stat_vals[m2], lw=0.6, color="C3", label="statistic")
        ax2.axhline(3.0, color="C3", ls="--", lw=0.6)
        ax.set_title(f"{label} (centre={fc:.2f} MHz)")
        ax.set_ylabel("|FT|")
        ax2.set_ylabel("statistic")
        ax.set_xlabel("MHz")
    fig.tight_layout()
    fig.savefig(figdir / "06_2638_zoom.png", dpi=130)
    plt.close(fig)

    # Pedestal check: median Re/Im in the lowest-statistic region as a fraction of σ.
    if np.any(safe):
        m_quiet = (f_centres > f_quiet - 25) & (f_centres < f_quiet + 25)
        re_med = float(np.median(spec.real[M // 2 : M // 2 + len(stat_vals)][m_quiet]))
        im_med = float(np.median(spec.imag[M // 2 : M // 2 + len(stat_vals)][m_quiet]))
        print(
            f"[2638] median Re/Im in quiet region (σ units, median σ): "
            f"{re_med/sigma_med:+.3f}, {im_med/sigma_med:+.3f}"
        )

    np.savez(
        npz_dir / "2638_statistic.npz",
        freqs=freqs,
        complex_spectrum=spec,
        sigma_per_point=sigma_per_point_arr,
        M=M,
        stat_freqs=f_centres,
        stat_values=stat_vals,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    print(f"[synth] running sweep → figures in {FIGDIR}")
    result = run_synthetic_sweep()
    plot_synthetic(result, FIGDIR)
    plot_spatial_profile(FIGDIR)
    np.savez(
        HERE / "synthetic_sweep.npz",
        rows=np.asarray(result.rows, dtype=object),
    )
    # Quick numeric summary for the findings note.
    for sname in STATS:
        clean = _agg(result.rows, case="clean", stat=sname, M=64)
        if len(clean):
            print(
                f"[synth] clean stat={sname} M=64: "
                f"mean={clean.mean():.3f} std={clean.std():.3f} "
                f"p99={np.quantile(clean, 0.99):.3f}"
            )
    print("[2638] running")
    run_2638(FIGDIR, HERE)
    print("done")


if __name__ == "__main__":
    main()
