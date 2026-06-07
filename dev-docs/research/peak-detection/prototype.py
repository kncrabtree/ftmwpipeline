"""
Peak-detection algorithm investigation — reproducibility script.

Regenerates every figure under ``figures/`` and the empirical numbers cited
in ``report.md``. From the repository root, with the project conda env:

    conda run -n ftmwpipeline-dev python \\
        dev-docs/research/peak-detection/prototype.py

Sections
--------
1. ``locate_peaks`` cost breakdown (Sav-Gol filter, argrelmin, threshold,
   merge) — where the wall time goes, and how it scales with N.
2. Two-pass driver wall-clock breakdown on 2638 — how much of Stage 3 is
   spent inside ``locate_peaks`` vs the FFT recomputes, noise estimation,
   apex-snap, and snap-back.
3. Synthetic correctness sweep — ground-truth spectra with controlled
   line counts, SNRs and separations; TP/FP rates of ``locate_peaks`` on
   the unapodized boxcar spectrum vs five common apodizations (Blackman-
   Harris, Hann, Kaiser-Bessel, Tukey, exponential 5 µs).
4. Anatomy of the unapodized false positives — where they sit relative
   to the nearest stronger detected peak (distance, height ratio); how
   well the closed-form leakage-reach mask covers them; what is left
   over.
5. Cheap suppression candidates run on the unapodized synthetic
   spectrum:
       (a) prominence threshold (scipy.signal.peak_prominences),
       (b) "stronger-neighbour" ratio (height ≥ α·max within ±R),
       (c) phase-coherence consistency vs the dominant nearby line,
       (d) closed-form leakage-reach mask alone (current Stage 3 gap).
   ROC-style curve (TP retained vs FP removed) for each.
6. 2638 reality check — apply the chosen suppression to the gap pass
   and quantify the reduction in low-SNR detections that the windowing
   stage will otherwise have to absorb.

Inputs: the 2638 fixture at ``scratch/exp_2638.ftmw`` (untracked).
Synthetic sections need no inputs; sections 2 and 6 do.

Fixture build (canonical):

    import ftmwpipeline.api as ftmw
    ftmw.import_data("scratch/exp_2638.ftmw", source="examples/blackchirp_data/2638", force=True)
    ftmw.detect_start_time("scratch/exp_2638.ftmw", band=(26500,40000), stamp=True)
    ftmw.compute_ft("scratch/exp_2638.ftmw", trim=(26500,40000))
    ftmw.estimate_noise("scratch/exp_2638.ftmw")
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as spsig

import scipy.fft as sfft
import ftmwpipeline.api as ftmw
from ftmwpipeline.core.data_structures import ComplexFT, FID, Sideband
from ftmwpipeline.preprocessing.noise_estimation import estimate_noise_scatter
from ftmwpipeline.preprocessing.leakage import estimate_leakage_reach
from ftmwpipeline.preprocessing.peak_detection import (
    detect_peaks as pd_detect,
    locate_peaks,
)
from ftmwpipeline.utils.signal_processing import apodize_fid

HERE = Path(__file__).parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
# dev-docs/research/<topic>/prototype.py → repo root = parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = REPO_ROOT / "scratch" / "exp_2638.ftmw"
RNG = np.random.default_rng(20260520)


# ---------------------------------------------------------------------------
# Synthetic spectrum helper
# ---------------------------------------------------------------------------
def synthetic_spectrum(
    n_lines: int,
    snrs: np.ndarray,
    freqs_mhz: np.ndarray,
    *,
    sigma: float = 1.0,
    T_us: float = 12.65,
    f_lo_mhz: float = 26500.0,
    f_hi_mhz: float = 40000.0,
    df_mhz: float = 0.0238,  # ≈ 2638 user grid spacing on the lower SB
    tau_us: Optional[float] = None,
    rng: np.random.Generator = RNG,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a complex FT analytically on a uniform frequency grid.

    Each line ``i`` contributes (1/2)·A_i·exp(iφ_i)·h_T(Δf;τ) on the rfft
    grid, then complex Gaussian noise of per-bin RMS ``sigma`` (so that
    |z| is Rayleigh with mean ≈ sigma·sqrt(π/4)). Returns the magnitude
    and complex spectra on the grid.

    ``snrs`` are line peak SNRs (≈ A_i·τ_eff/(2·sigma·sqrt(2))) — the
    "1/2" prefactor of the rfft is folded into the SNR definition so the
    user does not need to track it.
    """
    f = np.arange(f_lo_mhz, f_hi_mhz, df_mhz)
    n = len(f)
    T = T_us * 1e-6
    if tau_us is None:
        tau_eff = T
        # For the synthetic, we directly synthesise sinc-shaped responses
        # since real and synthetic noise statistics match analytically.
        edge = 0.0
    else:
        tau = tau_us * 1e-6
        edge = np.exp(-T / tau)
        tau_eff = tau * (1.0 - edge)

    spec = np.zeros(n, dtype=complex)
    for k in range(n_lines):
        f0 = freqs_mhz[k]
        snr = snrs[k]
        phi = rng.uniform(0, 2 * np.pi)
        # peak |z| ≈ sigma * snr; back out A via A * tau_eff / 2 = sigma * snr
        amp_peak = 2.0 * sigma * snr / tau_eff if tau_eff > 0 else 0.0
        df = (f - f0) * 1e6  # Hz
        if tau_us is None:
            # boxcar: h_T(df) = (1 - exp(-i2π df T)) / (i2π df), |h_T(0)|=T
            denom = 1j * 2.0 * np.pi * df
            denom[np.abs(df) < 1e-12] = 1.0  # avoid 0/0; set on-line value below
            ht = (1.0 - np.exp(-denom * T)) / denom
            # exact on-line value: T
            ht[np.abs(df) < 1e-12] = T
        else:
            denom = (1.0 / tau) + 1j * 2.0 * np.pi * df
            ht = (1.0 - np.exp(-denom * T)) / denom
        spec += 0.5 * amp_peak * np.exp(1j * phi) * ht

    noise = rng.normal(0, sigma / np.sqrt(2.0), size=n) + 1j * rng.normal(
        0, sigma / np.sqrt(2.0), size=n
    )
    spec += noise
    return f, spec


# ---------------------------------------------------------------------------
# 1. locate_peaks cost breakdown
# ---------------------------------------------------------------------------
def study_locate_peaks_cost() -> None:
    print("\n=== 1. locate_peaks cost breakdown ===")
    Ns = (10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000)
    sg_window = 11
    sg_order = 3
    rows = []
    for N in Ns:
        x = np.linspace(0.0, 1.0, N)
        y = RNG.normal(0, 1.0, size=N).astype(float)
        # warm-up
        locate_peaks(x, y, window=sg_window, order=sg_order)
        t0 = time.perf_counter()
        for _ in range(3):
            locate_peaks(x, y, window=sg_window, order=sg_order)
        wall = (time.perf_counter() - t0) / 3
        # per-step
        delta = x[1] - x[0]
        half = sg_window // 2
        y_pad = np.concatenate([y[:half], y, y[-half:]])
        coeffs_d2 = spsig.savgol_coeffs(sg_window, sg_order, deriv=2, delta=delta)
        coeffs_d1 = spsig.savgol_coeffs(sg_window, sg_order, deriv=1, delta=delta)
        t = time.perf_counter()
        for _ in range(3):
            d2pad = spsig.oaconvolve(y_pad, coeffs_d2, mode="same")
            d1pad = spsig.oaconvolve(y_pad, coeffs_d1, mode="same")
        t_filter = (time.perf_counter() - t) / 3
        d2 = d2pad[half:-half]
        d1 = d1pad[half:-half]
        t = time.perf_counter()
        for _ in range(3):
            neg = np.where(d2 < 0, d2, np.zeros_like(d2))
            spsig.argrelmin(neg, order=half)
        t_argrelmin = (time.perf_counter() - t) / 3
        rows.append((N, wall * 1000, t_filter * 1000, t_argrelmin * 1000))
        print(
            f"  N={N:>8d}  total={wall*1000:7.2f} ms   "
            f"sg_filter={t_filter*1000:7.2f} ms   "
            f"argrelmin={t_argrelmin*1000:7.2f} ms   "
            f"other={ (wall - t_filter - t_argrelmin)*1000:7.2f} ms"
        )
    arr = np.asarray(rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(arr[:, 0], arr[:, 1], "o-", label="locate_peaks total")
    ax.loglog(arr[:, 0], arr[:, 2], "s--", label="Sav-Gol filter (d1+d2)")
    ax.loglog(arr[:, 0], arr[:, 3], "^--", label="argrelmin")
    ax.set_xlabel("N (spectrum points)")
    ax.set_ylabel("wall time per call (ms)")
    ax.set_title("locate_peaks cost breakdown")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "01_locate_peaks_cost.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Stage 3 wall-clock breakdown on 2638
# ---------------------------------------------------------------------------
def study_pipeline_breakdown() -> None:
    if not FIXTURE_PATH.exists():
        print("\n=== 2. Stage 3 wall-clock breakdown — SKIPPED (no fixture) ===")
        return
    print("\n=== 2. Stage 3 wall-clock breakdown on 2638 ===")
    fid = ftmw.load_fid(str(FIXTURE_PATH))
    user_ft = ftmw.compute_ft(str(FIXTURE_PATH))
    base_pp = user_ft.metadata["processing_params"]
    trim = (user_ft.freq_array.min(), user_ft.freq_array.max())
    acq_us = (base_pp.end_us or fid.duration_us) - (base_pp.start_us or 0.0)

    # The canonical FT is unconditionally unapodized; apodized spectra for
    # the detection passes are built via apodize_fid (the common helper that
    # replicates the removed FID.preprocess apodization chain) + a manual
    # rfft replicating PreprocessedFID.compute_fft's freq-axis construction.
    original_length = len(fid.data)
    scale_factor = 10 ** base_pp.units_power

    def _recompute(expf_us: Optional[float], window_function: Optional[str] = None) -> ComplexFT:
        """Rebuild an apodized detection spectrum using apodize_fid + rfft.

        Replicates the removed fid.preprocess(...).compute_fft() path:
        active-region extraction, optional exp matched filter, optional
        symmetric window, DC removal, zero-pad (zpf=1), rfft, probe/sideband
        fold, normalization, and amplitude scaling.
        """
        processed = apodize_fid(
            fid.data,
            fid.time_array_us(),
            start_us=base_pp.start_us,
            end_us=base_pp.end_us,
            expf_us=expf_us,
            window_function=window_function,
            zpf=1,
            rdc=base_pp.rdc,
        )
        ft_data = sfft.rfft(processed)
        scope_freqs = sfft.rfftfreq(len(processed), d=fid.spacing) / 1e6  # MHz
        # Probe/sideband fold (matches PreprocessedFID.apply_molecular_frequency)
        if fid.sideband in (Sideband.LOWER, Sideband.LSB):
            mol_freqs = fid.probe_freq_mhz - scope_freqs
        else:
            mol_freqs = fid.probe_freq_mhz + scope_freqs
        ft_data = ft_data / original_length * scale_factor
        cft = ComplexFT.from_spectrum(ft_data, mol_freqs)
        return cft.trim_to_range(trim[0], trim[1])

    # Historical research default: 5 µs exponential for the primary pass.
    # The pipeline's canonical FT is now unapodized; 5 µs is used here only
    # as the research apodization benchmark described in §6 of the report.
    PRIMARY_EXPF_US = 5.0

    t = time.perf_counter()
    primary_ft = _recompute(PRIMARY_EXPF_US)
    t_primary_ft = time.perf_counter() - t

    t = time.perf_counter()
    gap_ft = _recompute(None)
    t_gap_ft = time.perf_counter() - t

    t = time.perf_counter()
    primary_noise = estimate_noise_scatter(
        primary_ft.freq_array, primary_ft.magnitude_spectrum
    )
    t_primary_noise = time.perf_counter() - t

    t = time.perf_counter()
    gap_noise = estimate_noise_scatter(
        gap_ft.freq_array, gap_ft.magnitude_spectrum
    )
    t_gap_noise = time.perf_counter() - t

    t = time.perf_counter()
    primary_res = locate_peaks(
        primary_ft.freq_array,
        primary_ft.magnitude_spectrum,
        window=11,
        order=3,
        thresh=2.0 * primary_noise.rms_noise,
    )
    t_primary_locate = time.perf_counter() - t

    t = time.perf_counter()
    gap_res = locate_peaks(
        gap_ft.freq_array,
        gap_ft.magnitude_spectrum,
        window=11,
        order=3,
        thresh=2.0 * gap_noise.rms_noise,
    )
    t_gap_locate = time.perf_counter() - t

    t = time.perf_counter()
    internal_peaks = pd_detect(
        primary_ft.freq_array,
        primary_ft.magnitude_spectrum,
        primary_noise.rms_noise,
        gap_ft.freq_array,
        gap_ft.magnitude_spectrum,
        gap_noise.rms_noise,
        min_snr=2.0,
        sg_window=11,
        sg_order=3,
        run_gap_pass=True,
    )
    t_full = time.perf_counter() - t

    print(f"  primary FT recompute:  {t_primary_ft*1000:7.1f} ms")
    print(f"  gap FT recompute:      {t_gap_ft*1000:7.1f} ms")
    print(f"  primary noise:         {t_primary_noise*1000:7.1f} ms")
    print(f"  gap noise:             {t_gap_noise*1000:7.1f} ms")
    print(f"  primary locate_peaks:  {t_primary_locate*1000:7.1f} ms")
    print(f"  gap locate_peaks:      {t_gap_locate*1000:7.1f} ms")
    print(
        f"  full two-pass driver:  {t_full*1000:7.1f} ms  "
        f"(produced {len(internal_peaks)} internal peaks)"
    )
    # Stage 3 wallclock summary as bar chart
    labels = [
        "primary FT",
        "gap FT",
        "primary noise",
        "gap noise",
        "primary locate",
        "gap locate",
    ]
    times = [
        t_primary_ft,
        t_gap_ft,
        t_primary_noise,
        t_gap_noise,
        t_primary_locate,
        t_gap_locate,
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, np.asarray(times) * 1000)
    ax.set_ylabel("wall time (ms)")
    ax.set_title(
        f"Stage 3 wall-clock breakdown on 2638 (N={len(primary_ft.freq_array)}, two-pass)"
    )
    for b, v in zip(bars, np.asarray(times) * 1000):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{v:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "02_pipeline_breakdown.png", dpi=110)
    plt.close(fig)
    return {
        "primary_ft": primary_ft,
        "gap_ft": gap_ft,
        "primary_noise": primary_noise,
        "gap_noise": gap_noise,
        "internal_peaks": internal_peaks,
        "acq_us": acq_us,
        "fid": fid,
        "base_pp": base_pp,
        "trim": trim,
        "recompute": _recompute,
    }


# ---------------------------------------------------------------------------
# 3. Synthetic TP/FP sweep: unapodized vs apodized
# ---------------------------------------------------------------------------
APODIZATIONS = [
    ("boxcar (unapodized)", None),
    ("exp 5 µs", "exp_5us"),
    ("Hann", "hann"),
    ("Blackman-Harris", "blackmanharris"),
    ("Kaiser β=8.6", "kaiser_8p6"),
]


def _apod_envelope(name: Optional[str], t_us: np.ndarray, T_us: float) -> np.ndarray:
    """Return the time-domain weighting w(t) applied by each apodization on [0,T]."""
    if name is None:
        return np.ones_like(t_us)
    if name == "exp_5us":
        return np.exp(-t_us / 5.0)
    if name == "hann":
        n = len(t_us)
        return spsig.windows.hann(n)
    if name == "blackmanharris":
        n = len(t_us)
        return spsig.windows.blackmanharris(n)
    if name == "kaiser_8p6":
        n = len(t_us)
        return spsig.windows.kaiser(n, beta=8.6)
    raise ValueError(name)


def study_synthetic_correctness(n_trials: int = 4) -> Dict:
    print("\n=== 3. Synthetic TP/FP sweep ===")
    # Acquisition + grid (baseband). Mirrors a typical FTMW transient class:
    # 15 µs acquisition → 1/T ≈ 67 kHz natural linewidth.
    T_us = 15.0
    # Baseband: 0.5 .. 200 MHz. Sampled above Nyquist; rfft to the full band.
    f_lo, f_hi = 0.5, 200.0
    # Time-domain sample rate well above the highest line frequency:
    # Nyquist 220 MHz → dt = 1/(2·220e6) ≈ 2.27 ns
    fs_hz = 2.0 * 220e6
    dt = 1.0 / fs_hz
    n_samples = int(round(T_us * 1e-6 / dt))
    dt = T_us * 1e-6 / n_samples
    fs_hz = 1.0 / dt
    # Zero-pad to give ~5 kHz/bin in the FFT → N_fft = 1 / (5e3 · dt)
    n_fft = int(round(1.0 / (5e3 * dt)))
    # round up to next power of two for speed
    n_fft = 1 << (int(np.ceil(np.log2(n_fft))))
    t_us = np.arange(n_samples) * dt * 1e6
    freqs_full = np.fft.rfftfreq(n_fft, d=dt) * 1e-6  # MHz
    mask = (freqs_full >= f_lo) & (freqs_full <= f_hi)
    f_band = freqs_full[mask]
    n_pts = len(f_band)
    # 30 lines, log-spaced SNR 5..500, frequencies uniform across band with min separation
    n_lines = 30
    truth_snr = np.logspace(np.log10(5), np.log10(500), n_lines)
    rng_lines = np.random.default_rng(20260601)
    truth_freq = rng_lines.uniform(f_lo + 20.0, f_hi - 20.0, size=n_lines)
    truth_freq = np.sort(truth_freq)
    for i in range(1, n_lines):
        if truth_freq[i] - truth_freq[i - 1] < 3.0:
            truth_freq[i] = truth_freq[i - 1] + 3.0
    print(
        f"  truth: {n_lines} lines, SNR in [{truth_snr.min():.1f}, {truth_snr.max():.1f}], "
        f"baseband 0–{f_hi:.0f} MHz, n_samples={n_samples}, n_fft={n_fft}, df={f_band[1]-f_band[0]:.4f} MHz"
    )

    # Pre-compute apodization envelopes on the *time* grid
    envelopes: Dict[str, np.ndarray] = {
        name: _apod_envelope(apod_name, t_us, T_us)
        for name, apod_name in APODIZATIONS
    }
    # Time-domain noise std calibrated so the rfft per-bin complex noise RMS is sigma=1.
    # For real Gaussian white noise of std σ_t over T = n_samples·dt, rfft (n_fft bins)
    # has per-bin variance = σ_t² · n_samples / 2 (for two-sided rfft).
    # → σ_bin = σ_t · √(n_samples/2). To make σ_bin = 1, σ_t = 1/√(n_samples/2).
    sigma_t = float(np.sqrt(2.0 / n_samples))

    metrics: Dict[str, Dict] = {}
    rows_summary = []

    for name, _ in APODIZATIONS:
        w_t = envelopes[name]
        TPs, FPs, FNs = [], [], []
        for trial in range(n_trials):
            rng_trial = np.random.default_rng(20260700 + trial)
            # Build time-domain signal: sum of damped (here undamped) cosines.
            # peak rfft magnitude of A·cos(2π f₀ t)·rect_T is A·n_samples/2 (per-bin)
            # → SNR_peak ≈ A · n_samples / 2 / σ_bin = A · n_samples / 2 / 1.
            sig = np.zeros(n_samples)
            for k in range(n_lines):
                A = 2.0 * truth_snr[k] / n_samples
                phi = rng_trial.uniform(0, 2 * np.pi)
                sig += A * np.cos(2.0 * np.pi * truth_freq[k] * 1e6 * t_us * 1e-6 + phi)
            # Physical model: noise enters the transient, then the whole
            # record (signal + noise) is apodized.
            noise_t = rng_trial.normal(0, sigma_t, n_samples)
            spec_c_full = np.fft.rfft((sig + noise_t) * w_t, n=n_fft)
            spec_c = spec_c_full[mask]
            mag = np.abs(spec_c)
            c_band = spec_c
            noise = estimate_noise_scatter(f_band, mag)
            res = locate_peaks(
                f_band, mag, window=11, order=3, thresh=3.0 * noise.rms_noise
            )
            # Match detections to truth: 0.2 MHz tolerance
            det_f = res.freqs
            det_h = res.intensities
            tol = 0.5  # MHz, generous since boxcar peak shift small
            matched_truth = np.zeros(n_lines, dtype=bool)
            is_tp = np.zeros(len(det_f), dtype=bool)
            for j, fd in enumerate(det_f):
                k = int(np.argmin(np.abs(truth_freq - fd)))
                if abs(truth_freq[k] - fd) <= tol and not matched_truth[k]:
                    matched_truth[k] = True
                    is_tp[j] = True
            TPs.append(int(is_tp.sum()))
            FPs.append(int((~is_tp).sum()))
            FNs.append(int((~matched_truth).sum()))
            # save spec for last trial of each apodization for follow-on figures
            if trial == n_trials - 1:
                metrics[name] = {
                    "f_band": f_band,
                    "mag": mag,
                    "complex": c_band,
                    "rms": noise.rms_noise,
                    "det_f": det_f,
                    "det_h": det_h,
                    "is_tp": is_tp,
                    "matched_truth": matched_truth,
                    "truth_freq": truth_freq,
                    "truth_snr": truth_snr,
                    "T_us": T_us,
                }
        tp_med = float(np.median(TPs))
        fp_med = float(np.median(FPs))
        fn_med = float(np.median(FNs))
        rows_summary.append((name, tp_med, fp_med, fn_med))
        print(
            f"  {name:>22s}: TP={tp_med:5.1f}  FP={fp_med:6.1f}  FN={fn_med:5.1f}  (medians over {n_trials} trials)"
        )

    # Figure: TP and FP per apodization (median across trials)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    names = [r[0] for r in rows_summary]
    tps = [r[1] for r in rows_summary]
    fps = [r[2] for r in rows_summary]
    fns = [r[3] for r in rows_summary]
    axes[0].barh(names, tps, color="C2", label="TP")
    axes[0].barh(names, fns, left=tps, color="C3", alpha=0.6, label="FN")
    axes[0].axvline(n_lines, color="k", lw=0.7, ls="--", label=f"truth = {n_lines}")
    axes[0].set_xlabel("counts")
    axes[0].set_title("True positives & false negatives")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, axis="x", alpha=0.3)
    axes[1].barh(names, fps, color="C3")
    axes[1].set_xlabel("false positives (sinc sidelobes etc.)")
    axes[1].set_title("False positives")
    axes[1].grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "03_apodization_tp_fp.png", dpi=110)
    plt.close(fig)
    return metrics


# ---------------------------------------------------------------------------
# 4. Anatomy of unapodized false positives
# ---------------------------------------------------------------------------
def study_fp_anatomy(metrics: Dict) -> Dict:
    print("\n=== 4. Anatomy of unapodized false positives ===")
    case = metrics["boxcar (unapodized)"]
    det_f = case["det_f"]
    det_h = case["det_h"]
    is_tp = case["is_tp"]
    truth_freq = case["truth_freq"]
    truth_snr = case["truth_snr"]
    T_us = case["T_us"]

    fp_f = det_f[~is_tp]
    fp_h = det_h[~is_tp]
    tp_f = det_f[is_tp]
    tp_h = det_h[is_tp]
    print(f"  TP={len(tp_f)}  FP={len(fp_f)}")
    # For each FP, find nearest stronger detection (any source)
    # Use the "nearest line in truth" for clarity:
    fp_nearest_truth_dist = np.full(len(fp_f), np.nan)
    fp_nearest_truth_snr = np.full(len(fp_f), np.nan)
    fp_height_ratio = np.full(len(fp_f), np.nan)
    for j, (ff, fh) in enumerate(zip(fp_f, fp_h)):
        k = int(np.argmin(np.abs(truth_freq - ff)))
        fp_nearest_truth_dist[j] = abs(truth_freq[k] - ff)
        fp_nearest_truth_snr[j] = truth_snr[k]
        # ratio: FP height / truth line peak height in the same spectrum
        # truth line peak height ≈ truth_snr * rms_noise (approx)
        rms_at_line = float(np.interp(truth_freq[k], case["f_band"], case["rms"]))
        truth_h = truth_snr[k] * rms_at_line
        fp_height_ratio[j] = fh / truth_h if truth_h > 0 else np.nan
    # Closed-form leakage reach mask: for each truth line, mask within ±reach
    reach = np.array(
        [
            float(estimate_leakage_reach(s, T_us, min_snr=3.0, tau_us=None))
            for s in truth_snr
        ]
    )
    is_inside_reach = np.zeros(len(fp_f), dtype=bool)
    for j, ff in enumerate(fp_f):
        d_to_lines = np.abs(truth_freq - ff)
        is_inside_reach[j] = bool(np.any(d_to_lines <= reach))
    print(
        f"  FPs inside ±reach of some truth line: {int(is_inside_reach.sum())} / {len(fp_f)} "
        f"({100*is_inside_reach.mean():.1f}%)"
    )
    print(
        f"  FP distance to nearest truth line:  median {np.nanmedian(fp_nearest_truth_dist):.3f} MHz, "
        f"p95 {np.nanpercentile(fp_nearest_truth_dist, 95):.3f} MHz"
    )
    print(
        f"  FP height ratio (FP/truth peak):    median {np.nanmedian(fp_height_ratio):.3f}, "
        f"p95 {np.nanpercentile(fp_height_ratio, 95):.3f}"
    )

    # Figure: scatter of FP positions vs distance to nearest truth line, colour by truth SNR
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(
        fp_nearest_truth_dist,
        fp_height_ratio,
        c=fp_nearest_truth_snr,
        cmap="viridis",
        norm=matplotlib.colors.LogNorm(),
        edgecolors="k",
        linewidths=0.3,
        s=24,
    )
    # Overlay analytic |sinc| envelope: sidelobe peaks at (k+0.5)/T,
    # envelope 1/(π|Δf|T) for boxcar (relative to peak T)
    df_plot = np.linspace(0.05, 25.0, 500)
    env = 1.0 / (np.pi * df_plot * (T_us))  # in same units (1/MHz·µs = MHz⁻¹·µs)
    ax.plot(df_plot, env, "k--", lw=1.2, label="1/(πΔfT) envelope")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("distance to nearest truth line (MHz)")
    ax.set_ylabel("FP height / truth peak height")
    ax.set_title(
        f"Anatomy of unapodized false positives "
        f"({len(fp_f)} FPs, last trial)"
    )
    plt.colorbar(sc, ax=ax, label="truth SNR of nearest line")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "04_fp_anatomy.png", dpi=110)
    plt.close(fig)
    return {
        "fp_f": fp_f,
        "fp_h": fp_h,
        "fp_dist": fp_nearest_truth_dist,
        "fp_snr": fp_nearest_truth_snr,
        "fp_ratio": fp_height_ratio,
        "is_inside_reach": is_inside_reach,
    }


# ---------------------------------------------------------------------------
# 5. Suppression candidates
# ---------------------------------------------------------------------------
def _prominence_test(
    mag: np.ndarray, det_idx: np.ndarray, alpha: float = 3.0, rms: Optional[np.ndarray] = None
) -> np.ndarray:
    """Keep detections whose prominence exceeds ``alpha`` × local σ."""
    pr = spsig.peak_prominences(mag, det_idx)[0]
    if rms is None:
        thresh = np.full(det_idx.shape, alpha * np.median(mag))
    else:
        thresh = alpha * rms[det_idx]
    return pr >= thresh


def _stronger_neighbour_ratio(
    mag: np.ndarray,
    det_idx: np.ndarray,
    radius_pts: int,
    alpha: float = 0.30,
) -> np.ndarray:
    """Keep detections where det_h ≥ α · max(mag in ±radius_pts window).

    Rejects sinc sidelobes: a sidelobe is much smaller than its parent line.
    A separate genuine line that happens to be near a stronger one will only
    be rejected when it is *both* close and significantly smaller — exactly
    the worst case for a sidelobe.
    """
    n = len(mag)
    keep = np.zeros(len(det_idx), dtype=bool)
    for j, i in enumerate(det_idx):
        lo = max(0, int(i) - radius_pts)
        hi = min(n, int(i) + radius_pts + 1)
        local_max = mag[lo:hi].max()
        keep[j] = mag[int(i)] >= alpha * local_max
    return keep


def _phase_coherence_with_neighbour(
    cspec: np.ndarray,
    det_idx: np.ndarray,
    radius_pts: int,
    threshold: float = 0.85,
) -> np.ndarray:
    """For each detection, compute the *normalized inner product* between the
    complex spectrum patch around it and a phase-aligned model of the local
    dominant peak. A sidelobe has a phase that flips by π between adjacent
    sinc lobes, so the leakage skirt at the sidelobe location is
    *anti-phase* with the main lobe. We test sidelobe-ness by comparing the
    complex value at the candidate's apex with the complex value at the
    apex of the nearest stronger peak — sidelobes have a ~π phase offset.

    Returns True for detections kept (i.e., not flagged as sidelobes).
    """
    n = len(cspec)
    mag = np.abs(cspec)
    keep = np.zeros(len(det_idx), dtype=bool)
    for j, i in enumerate(det_idx):
        i = int(i)
        # Find the strongest peak within ±radius_pts (other than itself)
        lo = max(0, i - radius_pts)
        hi = min(n, i + radius_pts + 1)
        window = mag[lo:hi].copy()
        local_pos = i - lo
        # exclude self
        window[local_pos] = 0
        if window.max() <= 0:
            keep[j] = True
            continue
        argmax_local = int(np.argmax(window))
        dom_i = lo + argmax_local
        if mag[dom_i] <= mag[i]:
            # candidate is the local max → not a sidelobe of anything stronger
            keep[j] = True
            continue
        # Phase of candidate vs phase of dominant
        z_cand = cspec[i]
        z_dom = cspec[dom_i]
        # Phase consistency: |Re(z_cand · conj(z_dom) / (|z_cand||z_dom|))|
        coh = (z_cand * np.conj(z_dom)) / (abs(z_cand) * abs(z_dom) + 1e-30)
        # A peak that is the same physical line component would have coh near +1.
        # A sinc sidelobe is anti-phase: coh near -1.
        # A genuinely independent peak has coh distributed roughly uniformly on the unit circle.
        # The discriminating cut is coh.real < -threshold → reject as sidelobe.
        keep[j] = coh.real >= -threshold
    return keep


def _reach_mask(
    det_f: np.ndarray,
    det_h: np.ndarray,
    rms_at_det: np.ndarray,
    T_us: float,
    min_snr: float = 3.0,
) -> np.ndarray:
    """Closed-form leakage-reach mask: drop a detection if a stronger detection
    sits within its leakage reach. Mirrors the Stage 3 gap-pass exclusion but
    self-consistently from the detection list (no separate primary pass).
    """
    n = len(det_f)
    snr = det_h / np.maximum(rms_at_det, 1e-30)
    reach = np.array(
        [float(estimate_leakage_reach(s, T_us, min_snr=min_snr, tau_us=None)) for s in snr]
    )
    keep = np.ones(n, dtype=bool)
    for j in range(n):
        # find any other detection within the *other*'s reach
        for k in range(n):
            if k == j or snr[k] <= snr[j]:
                continue
            if abs(det_f[k] - det_f[j]) <= reach[k]:
                keep[j] = False
                break
    return keep


def _apodized_veto(
    apod_mag: np.ndarray,
    apod_rms: np.ndarray,
    apod_freq: np.ndarray,
    det_f: np.ndarray,
    k_sigma: float = 2.0,
) -> np.ndarray:
    """Keep a detection only if the apodized spectrum at that frequency is
    itself above ``k_sigma`` × local noise.

    A sinc sidelobe is suppressed by ~60–90 dB under a strong apodization, so
    it collapses into the noise. A genuine weak line is only *broadened* by
    apodization — its peak amplitude drops by a modest factor (≈1.5–2×) but it
    stays well above the noise floor. The veto exploits this asymmetry. It is
    a *pointwise measurement* on a spectrum the pipeline already computes (the
    apodized primary), not an analytic model.
    """
    idx = np.searchsorted(apod_freq, det_f)
    idx = np.clip(idx, 1, len(apod_freq) - 1)
    left = apod_freq[idx - 1]
    right = apod_freq[idx]
    pick = np.where(np.abs(det_f - left) <= np.abs(det_f - right), idx - 1, idx)
    return apod_mag[pick] >= k_sigma * apod_rms[pick]


def study_suppressions(metrics: Dict) -> Dict:
    print("\n=== 5. Suppression candidates on unapodized synthetic ===")
    case = metrics["boxcar (unapodized)"]
    f = case["f_band"]
    mag = case["mag"]
    c = case["complex"]
    rms = case["rms"]
    det_f = case["det_f"]
    det_h = case["det_h"]
    is_tp = case["is_tp"]
    T_us = case["T_us"]
    df = f[1] - f[0]  # MHz per point
    # Apodized companion: same trial, same lines & noise seed (Blackman-Harris).
    bh = metrics["Blackman-Harris"]
    hann = metrics["Hann"]

    # Map detections to indices on the band grid
    det_idx = np.searchsorted(f, det_f)
    det_idx = np.clip(det_idx, 0, len(f) - 1)
    rms_at_det = rms[det_idx]
    print(f"  starting from {len(det_f)} detections ({int(is_tp.sum())} TP, {int((~is_tp).sum())} FP)")

    suppressions = {}

    # --- (a) Prominence threshold sweep ---
    print("  (a) prominence threshold sweep:")
    alphas = (2.0, 3.0, 5.0, 8.0, 12.0)
    rows_a = []
    for alpha in alphas:
        keep = _prominence_test(mag, det_idx, alpha=alpha, rms=rms)
        tp_kept = int((keep & is_tp).sum())
        fp_kept = int((keep & ~is_tp).sum())
        rows_a.append((alpha, tp_kept, fp_kept))
        print(f"     α={alpha:4.1f}σ: TP_kept={tp_kept:3d}  FP_kept={fp_kept:4d}")
    suppressions["prominence"] = rows_a

    # --- (b) Stronger-neighbour ratio sweep ---
    print("  (b) stronger-neighbour height-ratio (R=20 MHz):")
    radius_pts = int(20.0 / df)
    ratios = (0.05, 0.10, 0.20, 0.30, 0.50)
    rows_b = []
    for r in ratios:
        keep = _stronger_neighbour_ratio(mag, det_idx, radius_pts, alpha=r)
        tp_kept = int((keep & is_tp).sum())
        fp_kept = int((keep & ~is_tp).sum())
        rows_b.append((r, tp_kept, fp_kept))
        print(f"     α={r:4.2f}:  TP_kept={tp_kept:3d}  FP_kept={fp_kept:4d}")
    suppressions["stronger_neighbour"] = rows_b

    # --- (c) Phase-coherence consistency ---
    print("  (c) phase-coherence with nearest stronger (R=20 MHz):")
    rows_c = []
    for thr in (0.5, 0.7, 0.85, 0.95):
        keep = _phase_coherence_with_neighbour(c, det_idx, radius_pts, threshold=thr)
        tp_kept = int((keep & is_tp).sum())
        fp_kept = int((keep & ~is_tp).sum())
        rows_c.append((thr, tp_kept, fp_kept))
        print(f"     thr=-{thr:4.2f}: TP_kept={tp_kept:3d}  FP_kept={fp_kept:4d}")
    suppressions["phase_coh"] = rows_c

    # --- (d) Closed-form leakage-reach mask (self-applied) ---
    print("  (d) closed-form leakage-reach mask:")
    rows_d = []
    for mn_snr in (2.0, 3.0, 5.0, 8.0):
        keep = _reach_mask(det_f, det_h, rms_at_det, T_us, min_snr=mn_snr)
        tp_kept = int((keep & is_tp).sum())
        fp_kept = int((keep & ~is_tp).sum())
        rows_d.append((mn_snr, tp_kept, fp_kept))
        print(f"     min_snr={mn_snr:4.1f}: TP_kept={tp_kept:3d}  FP_kept={fp_kept:4d}")
    suppressions["reach"] = rows_d

    # --- (e) Apodized-amplitude veto (Blackman-Harris & Hann companions) ---
    print("  (e) apodized-amplitude veto:")
    rows_e = []
    for label, comp in (("BH", bh), ("Hann", hann)):
        for k in (1.5, 2.0, 3.0, 5.0):
            keep = _apodized_veto(
                comp["mag"], comp["rms"], comp["f_band"], det_f, k_sigma=k
            )
            tp_kept = int((keep & is_tp).sum())
            fp_kept = int((keep & ~is_tp).sum())
            rows_e.append((f"{label} {k}σ", tp_kept, fp_kept))
            print(
                f"     {label:>4s} k={k:4.1f}σ: TP_kept={tp_kept:3d}  FP_kept={fp_kept:4d}"
            )
    suppressions["apodized_veto"] = rows_e

    # ROC-style figure
    fig, ax = plt.subplots(figsize=(8, 5.5))
    n_tp_total = int(is_tp.sum())
    n_fp_total = int((~is_tp).sum())

    def _curve(rows, col, marker, label):
        # rows: (param, tp_kept, fp_kept)
        tp_kept = np.array([r[1] for r in rows])
        fp_kept = np.array([r[2] for r in rows])
        ax.plot(
            fp_kept / max(n_fp_total, 1),
            tp_kept / max(n_tp_total, 1),
            marker=marker,
            color=col,
            label=label,
            lw=1.2,
            markersize=7,
        )
        for r, x, y in zip(rows, fp_kept / max(n_fp_total, 1), tp_kept / max(n_tp_total, 1)):
            ax.annotate(f"{r[0]}", (x, y), fontsize=8, alpha=0.7)

    _curve(rows_a, "C0", "o", "prominence (α·σ)")
    _curve(rows_b, "C1", "s", "stronger-neighbour ratio")
    _curve(rows_c, "C2", "^", "phase coherence (anti)")
    _curve(rows_d, "C3", "D", "reach mask (min_snr)")
    _curve(
        [(r[0], r[1], r[2]) for r in rows_e if r[0].startswith("BH")],
        "C4",
        "P",
        "apodized veto (BH)",
    )
    _curve(
        [(r[0], r[1], r[2]) for r in rows_e if r[0].startswith("Hann")],
        "C5",
        "*",
        "apodized veto (Hann)",
    )
    ax.plot([0, 1], [0, 1], "k--", lw=0.5)
    ax.set_xlabel("FP retained fraction")
    ax.set_ylabel("TP retained fraction")
    ax.set_title(
        f"Suppression candidates on unapodized synthetic "
        f"(TP_total={n_tp_total}, FP_total={n_fp_total})"
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.4, 1.02)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "05_suppression_roc.png", dpi=110)
    plt.close(fig)
    return suppressions


# ---------------------------------------------------------------------------
# 6. 2638 reality check
# ---------------------------------------------------------------------------
def study_2638_reality(pipe_state: Optional[Dict]) -> None:
    if pipe_state is None:
        print("\n=== 6. 2638 reality check — SKIPPED (no fixture) ===")
        return
    print("\n=== 6. 2638 reality check ===")
    gap_ft = pipe_state["gap_ft"]
    gap_noise = pipe_state["gap_noise"]
    acq_us = pipe_state["acq_us"]
    f = gap_ft.freq_array
    mag = gap_ft.magnitude_spectrum
    c = np.asarray(gap_ft.complex_spectrum) if hasattr(gap_ft, "complex_spectrum") else None
    if c is None:
        # ComplexFT stores real/imag separately
        c = gap_ft.real_spectrum + 1j * gap_ft.imag_spectrum
    rms = gap_noise.rms_noise

    # Run an independent unapodized detection (mirror the gap pass)
    res = locate_peaks(f, mag, window=11, order=3, thresh=2.0 * rms)
    det_idx = res.indices
    det_f = res.freqs
    det_h = res.intensities
    df_mhz = abs(f[1] - f[0])
    radius_pts = max(1, int(round(20.0 / df_mhz)))
    print(f"  unapodized locate_peaks on 2638 gap grid: {len(det_idx)} detections")
    rms_at_det = rms[det_idx]
    snr_at_det = det_h / np.maximum(rms_at_det, 1e-30)
    print(
        f"  SNR distribution of detections: <3 → {(snr_at_det<3).sum()}, "
        f"3–10 → {((snr_at_det>=3)&(snr_at_det<10)).sum()}, "
        f">=10 → {(snr_at_det>=10).sum()}"
    )

    # Apodized companion: the primary (apodized) spectrum the pipeline already
    # computed for the primary pass.
    primary_ft = pipe_state["primary_ft"]
    primary_noise = pipe_state["primary_noise"]
    apod_mag = primary_ft.magnitude_spectrum
    apod_rms = primary_noise.rms_noise
    apod_freq = primary_ft.freq_array
    # apodized veto needs ascending frequency for searchsorted
    a_order = np.argsort(apod_freq)
    apod_freq_s = apod_freq[a_order]
    apod_mag_s = apod_mag[a_order]
    apod_rms_s = apod_rms[a_order]

    # Apply each suppression at one "reasonable" operating point
    keep_prom = _prominence_test(mag, det_idx, alpha=5.0, rms=rms)
    keep_neigh = _stronger_neighbour_ratio(mag, det_idx, radius_pts, alpha=0.10)
    keep_phase = _phase_coherence_with_neighbour(c, det_idx, radius_pts, threshold=0.85)
    keep_reach = _reach_mask(det_f, det_h, rms_at_det, acq_us, min_snr=3.0)
    keep_apod = _apodized_veto(apod_mag_s, apod_rms_s, apod_freq_s, det_f, k_sigma=2.0)

    print(
        f"  retained after  prominence  (5σ):  {int(keep_prom.sum()):5d}  ({100*keep_prom.mean():.1f}%)"
    )
    print(
        f"  retained after  neighbour  (α=0.10): {int(keep_neigh.sum()):5d}  ({100*keep_neigh.mean():.1f}%)"
    )
    print(
        f"  retained after  phase coh (-0.85):  {int(keep_phase.sum()):5d}  ({100*keep_phase.mean():.1f}%)"
    )
    print(
        f"  retained after  reach mask (3σ):   {int(keep_reach.sum()):5d}  ({100*keep_reach.mean():.1f}%)"
    )
    print(
        f"  retained after  apodized veto (2σ): {int(keep_apod.sum()):5d}  ({100*keep_apod.mean():.1f}%)"
    )

    # Figure: overlay spectrum + retained detections per suppression
    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
    z_band = (f > 36340) & (f < 36420)  # the strongest line region
    for ax, keep, label in zip(
        axes,
        [keep_prom, keep_neigh, keep_phase, keep_reach, keep_apod],
        [
            "prominence 5σ",
            "neighbour α=0.10",
            "phase coh thr=-0.85",
            "reach mask 3σ",
            "apodized veto 2σ",
        ],
    ):
        ax.plot(f[z_band], mag[z_band], color="0.4", lw=0.8)
        ax.plot(f[z_band], 3 * rms[z_band], color="C3", lw=0.6, ls="--", alpha=0.5)
        kept_in_zone = (det_f >= f[z_band].min()) & (det_f <= f[z_band].max())
        kept = kept_in_zone & keep
        rej = kept_in_zone & ~keep
        ax.plot(det_f[kept], det_h[kept], "v", color="C2", markersize=6, label="kept")
        ax.plot(det_f[rej], det_h[rej], "x", color="C3", markersize=6, label="dropped")
        ax.set_yscale("log")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("frequency (MHz)")
    axes[0].set_title("2638 strongest-line cluster (36350/36389): suppressions applied")
    fig.tight_layout()
    fig.savefig(FIG / "06_2638_suppression.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. 2638 primary-pass apodization comparison
# ---------------------------------------------------------------------------
def study_2638_primary_apodization(pipe_state: Optional[Dict]) -> None:
    """Recompute the 2638 *primary* pass under different apodizations.

    Measures, on the real 2638 spectrum, how many primary detections the
    research-baseline exp-5µs apodization produces vs Blackman-Harris /
    Blackman, and estimates the sidelobe-contamination fraction via the
    self-consistent reach mask. The pipeline's shipped primary pass uses
    Blackman-Harris internally; this section is the calibration study that
    motivated that choice (§6 of the report).
    """
    if pipe_state is None:
        print("\n=== 7. 2638 primary-pass apodization — SKIPPED (no fixture) ===")
        return
    print("\n=== 7. 2638 primary-pass apodization comparison ===")
    recompute = pipe_state["recompute"]
    acq_us = pipe_state["acq_us"]

    cases = [
        ("exp 5 µs (research baseline)", dict(expf_us=5.0, window_function=None)),
        ("Blackman-Harris", dict(expf_us=None, window_function="blackmanharris")),
        ("Blackman", dict(expf_us=None, window_function="blackman")),
        ("Hann", dict(expf_us=None, window_function="hann")),
    ]
    rows = []
    for label, kw in cases:
        cft = recompute(kw["expf_us"], window_function=kw["window_function"])
        f = cft.freq_array
        mag = cft.magnitude_spectrum
        noise = estimate_noise_scatter(f, mag)
        rms = noise.rms_noise
        res = locate_peaks(f, mag, window=11, order=3, thresh=2.0 * rms)
        det_idx = res.indices
        det_f = res.freqs
        det_h = res.intensities
        rms_at = rms[det_idx]
        # self-consistent reach mask → flag detections that are sidelobe-suspects
        keep = _reach_mask(det_f, det_h, rms_at, acq_us, min_snr=2.0)
        n_total = len(det_f)
        n_suspect = int((~keep).sum())
        rows.append((label, n_total, n_suspect))
        print(
            f"  {label:>28s}: {n_total:5d} detections, "
            f"{n_suspect:5d} ({100*n_suspect/max(n_total,1):.1f}%) flagged sidelobe-suspect"
        )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = [r[0] for r in rows]
    totals = np.array([r[1] for r in rows])
    suspects = np.array([r[2] for r in rows])
    clean = totals - suspects
    ax.barh(labels, clean, color="C2", label="not reach-flagged")
    ax.barh(labels, suspects, left=clean, color="C3", label="sidelobe-suspect (reach-flagged)")
    for i, (c, s) in enumerate(zip(clean, suspects)):
        ax.text(c + s, i, f"  {c+s}", va="center", fontsize=9)
    ax.set_xlabel("primary-pass detections on 2638 (internal grid)")
    ax.set_title("2638 primary-pass detections vs apodization")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "07_2638_primary_apodization.png", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    study_locate_peaks_cost()
    pipe_state = study_pipeline_breakdown()
    metrics = study_synthetic_correctness()
    study_fp_anatomy(metrics)
    study_suppressions(metrics)
    study_2638_reality(pipe_state)
    study_2638_primary_apodization(pipe_state)
    print("\nDone. Figures under:", FIG)
