"""Per-window joint (τ_L, τ_G) Voigt LSQ on shape-error windows (Part A).

Picks the Stage-5 shape-error candidates from the per-band production fit
on 2638 (windows where the unprior-on-tau-only single-exponential fit
still leaves a large chi^2 residual) and re-fits each with a Voigt
envelope

    s_i(t) = A_i · exp(-t/τ_L) · exp(-(t/τ_G)²) · cos(2π f_i t + φ_i)

where τ_L (Lorentzian decay) and τ_G (Gaussian decay) are *shared* per
window. The per-peak (A, Δf, φ) parameters and per-window (τ_L, τ_G)
are fit jointly by a constrained nonlinear least-squares step on the
window's active-FT slice.

The Voigt envelope's finite-T rfft has no closed form, but for our
problem dimensions (~10 us active region, ~few peaks per window) a
quadrature evaluation on a 1024-point time grid is cheap. The script
uses direct integration; conditioning + (τ_L, τ_G) covariance reads
from the LSQ Jacobian SVD at the solution.

Baseline for comparison: the *same per-peak seed* re-fit under the
pure-exponential model (current Stage 5 form) on the same active-FT
subset, with τ free (no Stage 2b prior) -- so the comparison isolates
the Voigt shape benefit from the prior anchoring. The persisted Stage
5 chi^2 (which carries the Stage 2b prior) is reported alongside for
context.

Run from the repository root:

    conda run -n ftmwpipeline-dev python \
        dev-docs/research/voigt-deficit/part_a_perwindow.py

Outputs:
- ``data/part_a_perwindow_summary.json`` -- per-window stats + global summary
- ``data/part_a_perwindow_detail.csv``   -- per-window records
- ``figures/03_voigt_vs_exp_chi2_perwindow.png`` -- chi2_r scatter (exp vs Voigt)
- ``figures/04_perwindow_tauG.png``      -- (τ_L, τ_G) recovered per window
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from ftmwpipeline._internal.stage5_impl import _build_active_ft_inputs
from ftmwpipeline.fitting.active_ft import compute_active_ft
from ftmwpipeline.fitting.peak_model import baseband_offset, sideband_sign
from ftmwpipeline.preprocessing.noise_estimation import estimate_noise_scatter

HERE = Path(__file__).parent
FIG = HERE / "figures"
DATA = HERE / "data"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

SCRATCH = Path("scratch/stage5-tau-calibration-lsq")
SOURCE_FIXTURE = SCRATCH / "exp_2638_unapodized.ftmw"

# Shape-error candidate selection: per-band production chi^2_r > 10
# AND multi-peak (single-peak high-chi^2 windows are prior-penalty
# driven, not shape-error -- their unprior chi^2 is already near 1).
CHI2R_SHAPE_ERROR_GATE = 10.0
MIN_PEAKS_FOR_CANDIDATE = 2

# Quadrature density for the Voigt envelope FT. 256 gives relative
# quadrature error ~ (T/N)^2/12 ~ 8e-5 for T=12.7us -- well below the
# per-bin noise.
N_QUAD = 256

TAU_BOUND_LO = 0.5
TAU_BOUND_HI = 100.0
# Multi-start grid on tau_G; large seed = "essentially pure-Lorentzian".
VOIGT_TAU_G_SEEDS = (100.0, 30.0, 12.0, 6.0)

logger = logging.getLogger("voigt-deficit-part-a")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


@dataclass
class PeakSeed:
    freq_mhz: float
    amplitude: float
    phase: float


@dataclass
class WindowFitOut:
    window_id: int
    freq_min: float
    freq_max: float
    n_peaks: int
    n_bins: int
    # Baseline (re-run pure-exp, tau free, no prior)
    tau_baseline_us: float
    chi2r_baseline: float
    chi2_baseline: float
    # Production (from persisted fit, per-band prior)
    tau_persisted_us: float
    chi2r_persisted: float
    # Voigt joint
    tau_L_us: float
    tau_G_us: float
    chi2r_voigt: float
    chi2_voigt: float
    voigt_converged: bool
    # Conditioning
    cond_number: float
    smallest_sv: float
    seed_tau_G_best: float


def _voigt_envelope_t(
    tau_L_us: float, tau_G_us: float, acquisition_us: float, n_quad: int = N_QUAD,
) -> tuple[np.ndarray, np.ndarray]:
    """Time-domain envelope vector + the corresponding time grid.

    Computed once per model call; the per-peak FT then reuses these arrays.
    """
    t = np.linspace(0.0, acquisition_us, n_quad)
    env = np.exp(-t / tau_L_us - (t / tau_G_us) ** 2)
    return t, env


def _envelope_ft_at(
    df_mhz: np.ndarray,
    env_t: np.ndarray,
    t_grid: np.ndarray,
    acquisition_us: float,
) -> np.ndarray:
    """FT of a precomputed envelope at the offsets ``df_mhz`` via trapezoid.

    ``env_t`` is the envelope sampled on ``t_grid`` (length ``n_quad``); both
    are shared across peaks in one model call. Returns a complex array of
    shape ``df_mhz``.
    """
    df = np.asarray(df_mhz, dtype=float)
    n_quad = t_grid.size
    dt = acquisition_us / (n_quad - 1)
    # Outer product (n_freq, n_t) complex.
    integrand = env_t[None, :] * np.exp(
        -1j * 2.0 * np.pi * df[:, None] * t_grid[None, :]
    )
    integral = (
        integrand.sum(axis=1) - 0.5 * (integrand[:, 0] + integrand[:, -1])
    ) * dt
    return integral.astype(np.complex128)


def _exp_envelope_ft(
    df_mhz: np.ndarray, tau_us: float, acquisition_us: float
) -> np.ndarray:
    """Closed-form finite-T FT of ``exp(-t/τ)`` -- the pure-exp baseline shape.

    ``h_T(Δf; τ) = (1 - exp(-(1/τ + i 2π Δf) T)) / (1/τ + i 2π Δf)``.
    """
    df = np.asarray(df_mhz, dtype=float)
    denom = (1.0 / tau_us) + 1j * 2.0 * np.pi * df
    return (1.0 - np.exp(-denom * acquisition_us)) / denom


def _pack_exp(seeds: List[PeakSeed], tau_us: float) -> np.ndarray:
    """Pack ``(A_i, Δf_i, φ_i)*K, τ)`` for the pure-exp model."""
    params = []
    for s in seeds:
        params.extend([s.amplitude, 0.0, s.phase])  # Δf relative to seed freq
    params.append(tau_us)
    return np.asarray(params, dtype=float)


def _pack_voigt(seeds: List[PeakSeed], tau_L_us: float, tau_G_us: float) -> np.ndarray:
    """Pack ``(A_i, Δf_i, φ_i)*K, τ_L, τ_G)`` for the Voigt model."""
    params = []
    for s in seeds:
        params.extend([s.amplitude, 0.0, s.phase])
    params.extend([tau_L_us, tau_G_us])
    return np.asarray(params, dtype=float)


def _exp_model(
    params: np.ndarray,
    seeds: List[PeakSeed],
    df_grid_mhz_per_peak: List[np.ndarray],
    acquisition_us: float,
) -> np.ndarray:
    """Pure-exp window model on the window's frequency grid.

    ``params = [A0, Δf0, φ0, A1, Δf1, φ1, ..., τ]``. ``df_grid_mhz_per_peak[i]``
    is the (window frequency grid) minus (seed i's molecular freq), in MHz --
    so the model evaluates ``h_T(df_grid[i] - Δf_i; τ)`` per peak. The grid
    direction (lower-sideband: f decreasing in MHz with bin index) is already
    baked in by passing the grid in molecular-frequency space.
    """
    n_peaks = len(seeds)
    tau = params[-1]
    spectrum = np.zeros(df_grid_mhz_per_peak[0].shape, dtype=np.complex128)
    for i in range(n_peaks):
        A, df_shift, phi = params[3 * i: 3 * i + 3]
        df = df_grid_mhz_per_peak[i] - df_shift
        spectrum += 0.5 * A * np.exp(1j * phi) * _exp_envelope_ft(
            df, tau, acquisition_us
        )
    return spectrum


def _voigt_model(
    params: np.ndarray,
    seeds: List[PeakSeed],
    df_grid_mhz_per_peak: List[np.ndarray],
    acquisition_us: float,
    *,
    n_quad: int = N_QUAD,
) -> np.ndarray:
    """Voigt window model on the window's frequency grid.

    ``params = [A0, Δf0, φ0, A1, Δf1, φ1, ..., τ_L, τ_G]``. The time-domain
    envelope is computed once per call and reused across peaks.
    """
    n_peaks = len(seeds)
    tau_L, tau_G = params[-2], params[-1]
    t_grid, env_t = _voigt_envelope_t(tau_L, tau_G, acquisition_us, n_quad=n_quad)
    spectrum = np.zeros(df_grid_mhz_per_peak[0].shape, dtype=np.complex128)
    for i in range(n_peaks):
        A, df_shift, phi = params[3 * i: 3 * i + 3]
        df = df_grid_mhz_per_peak[i] - df_shift
        spectrum += 0.5 * A * np.exp(1j * phi) * _envelope_ft_at(
            df, env_t, t_grid, acquisition_us,
        )
    return spectrum


def _stack_resid(
    model_complex: np.ndarray, data_complex: np.ndarray, sigma_per_bin: np.ndarray,
) -> np.ndarray:
    """Stack Re/Im residual with ``sigma/sqrt(2)`` weighting per Stage 5 convention."""
    resid = model_complex - data_complex
    w = 1.0 / (sigma_per_bin / np.sqrt(2.0))
    return np.concatenate([(resid.real * w), (resid.imag * w)])


def _exp_residuals(
    params: np.ndarray,
    seeds: List[PeakSeed],
    df_grid_mhz_per_peak: List[np.ndarray],
    acquisition_us: float,
    data_complex: np.ndarray,
    sigma_per_bin: np.ndarray,
) -> np.ndarray:
    model = _exp_model(params, seeds, df_grid_mhz_per_peak, acquisition_us)
    return _stack_resid(model, data_complex, sigma_per_bin)


def _voigt_residuals(
    params: np.ndarray,
    seeds: List[PeakSeed],
    df_grid_mhz_per_peak: List[np.ndarray],
    acquisition_us: float,
    data_complex: np.ndarray,
    sigma_per_bin: np.ndarray,
) -> np.ndarray:
    model = _voigt_model(params, seeds, df_grid_mhz_per_peak, acquisition_us)
    return _stack_resid(model, data_complex, sigma_per_bin)


def _fit_exp_window(
    seeds: List[PeakSeed],
    df_grid_mhz_per_peak: List[np.ndarray],
    acquisition_us: float,
    data_complex: np.ndarray,
    sigma_per_bin: np.ndarray,
    tau0: float = 6.0,
    df_bound: float = 1.0,
    tau_bound_hi: float = TAU_BOUND_HI,
) -> tuple[np.ndarray, float, bool]:
    """Fit pure-exp model with τ free. Returns ``(params, chi2, ok)``."""
    n_peaks = len(seeds)
    x0 = _pack_exp(seeds, tau0)
    lo: list[float] = []
    hi: list[float] = []
    for _ in range(n_peaks):
        lo.extend([0.0, -df_bound, -np.pi])
        hi.extend([np.inf, df_bound, np.pi])
    lo.append(TAU_BOUND_LO)
    hi.append(tau_bound_hi)
    res = least_squares(
        _exp_residuals,
        x0=x0,
        bounds=(lo, hi),
        args=(seeds, df_grid_mhz_per_peak, acquisition_us,
              data_complex, sigma_per_bin),
        method="trf",
        max_nfev=400,
    )
    chi2 = float(np.sum(res.fun ** 2))
    return res.x, chi2, bool(res.success)


def _fit_voigt_window_one(
    seeds: List[PeakSeed],
    df_grid_mhz_per_peak: List[np.ndarray],
    acquisition_us: float,
    data_complex: np.ndarray,
    sigma_per_bin: np.ndarray,
    tau_L0: float,
    tau_G0: float,
    df_bound: float = 1.0,
) -> tuple[np.ndarray, float, bool, float, float]:
    """Single-start Voigt fit. Returns ``(params, chi2, ok, cond, sv_min)``."""
    n_peaks = len(seeds)
    x0 = _pack_voigt(seeds, tau_L0, tau_G0)
    lo: list[float] = []
    hi: list[float] = []
    for _ in range(n_peaks):
        lo.extend([0.0, -df_bound, -np.pi])
        hi.extend([np.inf, df_bound, np.pi])
    lo.extend([TAU_BOUND_LO, TAU_BOUND_LO])
    hi.extend([TAU_BOUND_HI, TAU_BOUND_HI])
    res = least_squares(
        _voigt_residuals,
        x0=x0,
        bounds=(lo, hi),
        args=(seeds, df_grid_mhz_per_peak, acquisition_us,
              data_complex, sigma_per_bin),
        method="trf",
        max_nfev=600,
    )
    chi2 = float(np.sum(res.fun ** 2))
    # Conditioning from final Jacobian SVD.
    try:
        s = np.linalg.svd(res.jac, compute_uv=False)
        sv_min = float(s.min())
        cond = float((s.max() / sv_min) ** 2) if sv_min > 0 else float("inf")
    except np.linalg.LinAlgError:
        cond = float("inf")
        sv_min = 0.0
    return res.x, chi2, bool(res.success), cond, sv_min


def _fit_voigt_multistart(
    seeds: List[PeakSeed],
    df_grid_mhz_per_peak: List[np.ndarray],
    acquisition_us: float,
    data_complex: np.ndarray,
    sigma_per_bin: np.ndarray,
    tau_L0: float,
    df_bound: float = 1.0,
) -> tuple[np.ndarray, float, bool, float, float, float]:
    """Multi-start Voigt fit on tau_G; returns best basin."""
    best = None
    best_chi2 = np.inf
    best_seed = float("nan")
    for tG0 in VOIGT_TAU_G_SEEDS:
        try:
            params, chi2, ok, cond, sv = _fit_voigt_window_one(
                seeds, df_grid_mhz_per_peak, acquisition_us,
                data_complex, sigma_per_bin, tau_L0, tG0, df_bound,
            )
        except Exception:  # noqa: BLE001
            continue
        if chi2 < best_chi2:
            best_chi2 = chi2
            best = (params, chi2, ok, cond, sv)
            best_seed = tG0
    if best is None:
        n_peaks = len(seeds)
        params = _pack_voigt(seeds, tau_L0, TAU_BOUND_HI * 0.95)
        return params, float("inf"), False, float("inf"), 0.0, float("nan")
    params, chi2, ok, cond, sv = best
    return params, chi2, ok, cond, sv, best_seed


def _load_candidate_windows(
    file_path: Path, gate: float
) -> list[tuple[int, float, float, float, float, list[PeakSeed]]]:
    """Pull shape-error candidate windows from the persisted Stage 5 fit.

    Returns ``(window_id, freq_min, freq_max, tau_persisted, chi2r_persisted, seeds)``
    for windows with ``reduced_chi2 > gate`` and ``tau_fitted == 1``
    (so a meaningful comparison against a free-tau baseline exists).
    """
    out = []
    with h5py.File(file_path, "r") as h5f:
        windows = h5f["stage5_fitting/windows"]
        for name in sorted(windows.keys()):
            wg = windows[name]
            chi2r = float(wg.attrs["reduced_chi2"])
            if not (np.isfinite(chi2r) and chi2r > gate):
                continue
            if int(wg.attrs.get("tau_fitted", -1)) != 1:
                continue
            wid = int(wg.attrs["window_id"])
            f_lo = float(wg.attrs["freq_min"])
            f_hi = float(wg.attrs["freq_max"])
            tau = float(wg.attrs["tau_us"])
            seeds: list[PeakSeed] = []
            if "peaks" in wg and "peak_id" in wg["peaks"]:
                freqs = wg["peaks/frequency_mhz"][:]
                amps = wg["peaks/amplitude"][:]
                phases = wg["peaks/phase"][:]
                for f, a, p in zip(freqs, amps, phases):
                    seeds.append(PeakSeed(
                        freq_mhz=float(f), amplitude=float(a), phase=float(p),
                    ))
            # Single-peak high-chi^2 windows are prior-penalty driven, not
            # shape-error. Drop them so the Voigt comparison isn't dominated
            # by long-tau saturation basins.
            if len(seeds) < MIN_PEAKS_FOR_CANDIDATE:
                continue
            out.append((wid, f_lo, f_hi, tau, chi2r, seeds))
    return out


def main() -> None:
    if not SOURCE_FIXTURE.exists():
        raise FileNotFoundError(f"Source fixture missing: {SOURCE_FIXTURE}")

    logger.info("Reading active-FT inputs from %s", SOURCE_FIXTURE)
    (
        fid_arr, sample_dt_us, start_us, end_us,
        probe_freq_mhz, sideband, n_padded, acquisition_us,
        _user_ft, _trim_range,
    ) = _build_active_ft_inputs(str(SOURCE_FIXTURE))
    logger.info(
        "Active region: %.3f-%.3f us (T_active=%.3f us), sideband=%s",
        start_us, end_us, acquisition_us, sideband,
    )

    # Unapodized active FT -- consistent with what Stage 5 fits on.  Apodization
    # would convolve in the window's spectral response and bias the (tau_L, tau_G)
    # line-shape recovery, so it is intentionally omitted here.
    active_ft = compute_active_ft(
        fid_arr, sample_dt_us,
        start_us=start_us, end_us=end_us,
        probe_freq_mhz=probe_freq_mhz,
        sideband=sideband, n_padded=n_padded,
    )
    logger.info(
        "Active-FT: %d bins, bin spacing %.4f MHz",
        active_ft.freq_mhz.size,
        abs(active_ft.freq_mhz[1] - active_ft.freq_mhz[0]),
    )

    # Per-bin noise on the active-FT (same path Stage 5 uses).
    sort_idx = np.argsort(active_ft.freq_mhz)
    sorted_freq = np.ascontiguousarray(active_ft.freq_mhz[sort_idx])
    sorted_mag = np.ascontiguousarray(np.abs(active_ft.complex_spectrum)[sort_idx])
    active_noise = estimate_noise_scatter(sorted_freq, sorted_mag)
    unsort = np.argsort(sort_idx)
    sigma_per_bin_all = np.asarray(active_noise.rms_noise, dtype=float)[unsort]

    candidates = _load_candidate_windows(
        SOURCE_FIXTURE, gate=CHI2R_SHAPE_ERROR_GATE,
    )
    logger.info(
        "Shape-error candidates (chi2r > %.0f, tau_was_fit=1): %d windows",
        CHI2R_SHAPE_ERROR_GATE, len(candidates),
    )

    s_sign = sideband_sign(sideband)
    rows: List[WindowFitOut] = []
    for (wid, f_lo, f_hi, tau_pers, chi2r_pers, seeds) in candidates:
        if not seeds:
            logger.warning("Window %d has no peak seeds; skipping", wid)
            continue
        in_win = (
            (active_ft.freq_mhz >= f_lo) & (active_ft.freq_mhz <= f_hi)
        )
        if in_win.sum() < 3:
            logger.warning("Window %d: %d bins < 3; skipping", wid, in_win.sum())
            continue
        win_freq = active_ft.freq_mhz[in_win]
        win_data = active_ft.complex_spectrum[in_win]
        win_sigma = sigma_per_bin_all[in_win]

        # For each peak, the (window frequency grid) minus (peak molecular f)
        # in baseband-offset units. Sideband sign is baked into the FT axis;
        # baseband_offset() handles the conversion for the model.
        df_grid_per_peak: list[np.ndarray] = []
        for sd in seeds:
            # baseband offset of each window bin relative to the peak freq.
            df_grid_per_peak.append(
                baseband_offset(win_freq, sd.freq_mhz, sideband)
            )

        # Baseline: free-tau pure-exp re-fit on this active-FT subset.
        try:
            exp_params, chi2_e, ok_e = _fit_exp_window(
                seeds, df_grid_per_peak, acquisition_us,
                win_data, win_sigma, tau0=tau_pers,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Window %d baseline fit failed: %s", wid, exc)
            continue
        tau_baseline = float(exp_params[-1])
        n_data = 2 * win_data.size  # Re+Im
        n_params_exp = 3 * len(seeds) + 1
        dof_exp = max(n_data - n_params_exp, 1)
        chi2r_baseline = chi2_e / dof_exp

        # Voigt joint fit, multi-start on tau_G.
        try:
            v_params, chi2_v, ok_v, cond_v, sv_v, seed_tG = (
                _fit_voigt_multistart(
                    seeds, df_grid_per_peak, acquisition_us,
                    win_data, win_sigma, tau_L0=tau_baseline,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Window %d Voigt fit failed: %s", wid, exc)
            continue
        tau_L = float(v_params[-2])
        tau_G = float(v_params[-1])
        n_params_v = 3 * len(seeds) + 2
        dof_v = max(n_data - n_params_v, 1)
        chi2r_voigt = chi2_v / dof_v

        rows.append(WindowFitOut(
            window_id=wid,
            freq_min=float(f_lo),
            freq_max=float(f_hi),
            n_peaks=len(seeds),
            n_bins=int(win_data.size),
            tau_baseline_us=tau_baseline,
            chi2r_baseline=float(chi2r_baseline),
            chi2_baseline=float(chi2_e),
            tau_persisted_us=float(tau_pers),
            chi2r_persisted=float(chi2r_pers),
            tau_L_us=tau_L,
            tau_G_us=tau_G,
            chi2r_voigt=float(chi2r_voigt),
            chi2_voigt=float(chi2_v),
            voigt_converged=ok_v,
            cond_number=float(cond_v),
            smallest_sv=float(sv_v),
            seed_tau_G_best=float(seed_tG),
        ))
        logger.info(
            "w%03d f=[%.1f,%.1f] N=%d bins=%d | baseline tau=%.2f chi2r=%.2f | "
            "voigt tL=%.2f tG=%.2f chi2r=%.2f (cond=%.1e seed=%.0f)",
            wid, f_lo, f_hi, len(seeds), win_data.size,
            tau_baseline, chi2r_baseline,
            tau_L, tau_G, chi2r_voigt, cond_v, seed_tG,
        )

    logger.info("Fits done: %d windows", len(rows))

    # Summary stats.
    n = len(rows)
    n_improved_50 = sum(
        1 for r in rows
        if r.chi2r_voigt < 0.5 * r.chi2r_baseline
    )
    n_improved_any = sum(1 for r in rows if r.chi2r_voigt < r.chi2r_baseline)
    summary = {
        "fixture": str(SOURCE_FIXTURE),
        "chi2r_shape_error_gate": CHI2R_SHAPE_ERROR_GATE,
        "n_candidates": n,
        "n_voigt_improves_any": n_improved_any,
        "n_voigt_improves_by_50pct_or_more": n_improved_50,
        "fraction_meeting_50pct_gate": (n_improved_50 / n) if n > 0 else 0.0,
        "median_chi2r_baseline": (
            float(np.median([r.chi2r_baseline for r in rows])) if rows else float("nan")
        ),
        "median_chi2r_voigt": (
            float(np.median([r.chi2r_voigt for r in rows])) if rows else float("nan")
        ),
        "median_improvement_factor": (
            float(np.median([r.chi2r_baseline / max(r.chi2r_voigt, 1e-9) for r in rows]))
            if rows else float("nan")
        ),
        "median_tau_L_us": (
            float(np.median([r.tau_L_us for r in rows])) if rows else float("nan")
        ),
        "median_tau_G_us": (
            float(np.median([r.tau_G_us for r in rows])) if rows else float("nan")
        ),
        "median_cond_number": (
            float(np.median(
                [r.cond_number for r in rows if np.isfinite(r.cond_number)]
            )) if rows else float("nan")
        ),
        "rows": [
            {
                "window_id": r.window_id,
                "freq_min": r.freq_min,
                "freq_max": r.freq_max,
                "n_peaks": r.n_peaks,
                "n_bins": r.n_bins,
                "tau_baseline_us": r.tau_baseline_us,
                "chi2r_baseline": r.chi2r_baseline,
                "tau_persisted_us": r.tau_persisted_us,
                "chi2r_persisted": r.chi2r_persisted,
                "tau_L_us": r.tau_L_us,
                "tau_G_us": r.tau_G_us,
                "chi2r_voigt": r.chi2r_voigt,
                "voigt_converged": r.voigt_converged,
                "cond_number": r.cond_number,
                "seed_tau_G_best": r.seed_tau_G_best,
            }
            for r in rows
        ],
    }
    summary_path = DATA / "part_a_perwindow_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote %s", summary_path)

    detail_path = DATA / "part_a_perwindow_detail.csv"
    with detail_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "window_id", "freq_min", "freq_max", "n_peaks", "n_bins",
            "tau_persisted_us", "chi2r_persisted",
            "tau_baseline_us", "chi2r_baseline",
            "tau_L_us", "tau_G_us", "chi2r_voigt", "voigt_converged",
            "cond_number", "seed_tau_G_best",
        ])
        for r in rows:
            w.writerow([
                r.window_id, f"{r.freq_min:.3f}", f"{r.freq_max:.3f}",
                r.n_peaks, r.n_bins,
                f"{r.tau_persisted_us:.4f}", f"{r.chi2r_persisted:.4f}",
                f"{r.tau_baseline_us:.4f}", f"{r.chi2r_baseline:.4f}",
                f"{r.tau_L_us:.4f}", f"{r.tau_G_us:.4f}",
                f"{r.chi2r_voigt:.4f}", int(r.voigt_converged),
                f"{r.cond_number:.2e}", f"{r.seed_tau_G_best:.2f}",
            ])
    logger.info("Wrote %s", detail_path)

    # Figure 3: chi2r scatter (baseline pure-exp vs Voigt).
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    if rows:
        x = np.array([r.chi2r_baseline for r in rows])
        y = np.array([r.chi2r_voigt for r in rows])
        sizes = np.clip(8.0 * np.array([r.n_peaks for r in rows]), 12, 90)
        sc = ax.scatter(x, y, s=sizes, alpha=0.7, edgecolor="black", linewidth=0.5)
        for r in rows:
            ax.annotate(f"w{r.window_id}", (r.chi2r_baseline, r.chi2r_voigt),
                        fontsize=7, alpha=0.7,
                        xytext=(4, 2), textcoords="offset points")
        lim_lo, lim_hi = 0.5, max(x.max(), y.max()) * 1.2
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=0.7,
                label=r"Voigt $\chi^2_r$ = baseline $\chi^2_r$")
        # 50% reduction line
        ax.plot([lim_lo, lim_hi], [0.5 * lim_lo, 0.5 * lim_hi], "g--", lw=0.7,
                label=r"50% reduction")
        ax.axhline(1.0, color="tab:green", lw=0.6, alpha=0.7,
                   label=r"noise floor $\chi^2_r=1$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel(r"baseline (pure-exp, free-$\tau$) $\chi^2_r$")
    ax.set_ylabel(r"Voigt joint $\chi^2_r$")
    ax.set_title(
        f"Voigt-deficit Part A: shape-error windows on 2638 "
        f"(N={len(rows)}, gate $\\chi^2_r>{CHI2R_SHAPE_ERROR_GATE:.0f}$)"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out1 = FIG / "03_voigt_vs_exp_chi2_perwindow.png"
    fig.savefig(out1, dpi=140)
    plt.close(fig)
    logger.info("Wrote %s", out1)

    # Figure 4: recovered (tau_L, tau_G) by window center.
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    if rows:
        centers = np.array([0.5 * (r.freq_min + r.freq_max) for r in rows])
        tau_L = np.array([r.tau_L_us for r in rows])
        tau_G = np.array([r.tau_G_us for r in rows])
        chi2_imp = np.array(
            [r.chi2r_baseline / max(r.chi2r_voigt, 1e-9) for r in rows]
        )
        ax.scatter(centers, tau_L, marker="o", s=50, alpha=0.7, c="tab:blue",
                   label=r"$\tau_L$ (Lorentzian)")
        sc = ax.scatter(
            centers, tau_G, marker="^", s=70, alpha=0.8,
            c=np.log10(np.clip(chi2_imp, 1.0, None)), cmap="viridis",
            edgecolor="black", linewidth=0.4,
            label=r"$\tau_G$ (Gaussian)",
        )
        plt.colorbar(sc, ax=ax,
                     label=r"$\log_{10}$ (baseline $\chi^2_r$ / Voigt $\chi^2_r$)")
        for r in rows:
            ax.annotate(f"w{r.window_id}",
                        (0.5 * (r.freq_min + r.freq_max), r.tau_G_us),
                        fontsize=7, alpha=0.7,
                        xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("window center frequency (MHz)")
    ax.set_ylabel(r"$\tau$ (us)")
    ax.set_title(
        "Voigt-deficit Part A: recovered $(\\tau_L, \\tau_G)$ on shape-error windows"
    )
    ax.set_yscale("log")
    ax.set_ylim(TAU_BOUND_LO * 0.9, TAU_BOUND_HI * 1.1)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out2 = FIG / "04_perwindow_tauG.png"
    fig.savefig(out2, dpi=140)
    plt.close(fig)
    logger.info("Wrote %s", out2)

    # Console summary.
    print()
    print("====== Voigt-deficit Part A (2638) ======")
    print(f"  N candidates (chi2r > {CHI2R_SHAPE_ERROR_GATE:.0f}, tau_was_fit=1): {n}")
    print(f"  N with Voigt < baseline chi2r:          {n_improved_any}")
    print(f"  N with Voigt chi2r < 0.5 * baseline:    {n_improved_50}")
    if rows:
        print(f"  median chi2r baseline:                  "
              f"{summary['median_chi2r_baseline']:.2f}")
        print(f"  median chi2r Voigt:                     "
              f"{summary['median_chi2r_voigt']:.2f}")
        print(f"  median improvement factor:              "
              f"{summary['median_improvement_factor']:.2f}x")
        print(f"  median tau_L (us):                      "
              f"{summary['median_tau_L_us']:.2f}")
        print(f"  median tau_G (us):                      "
              f"{summary['median_tau_G_us']:.2f}")
        print(f"  median cond number (J^TJ):              "
              f"{summary['median_cond_number']:.2e}")
        print()
        print("  per-window (sorted by baseline chi2r):")
        print(f"    {'wid':>4} {'fmin':>7} {'fmax':>7} {'K':>3} "
              f"{'baseline':>10} {'voigt':>9} {'tau_L':>7} {'tau_G':>7} "
              f"{'factor':>7} {'cond':>10}")
        for r in sorted(rows, key=lambda x: -x.chi2r_baseline):
            factor = r.chi2r_baseline / max(r.chi2r_voigt, 1e-9)
            print(f"    w{r.window_id:03d} {r.freq_min:>7.1f} {r.freq_max:>7.1f} "
                  f"{r.n_peaks:>3d} "
                  f"{r.chi2r_baseline:>10.2f} {r.chi2r_voigt:>9.2f} "
                  f"{r.tau_L_us:>7.2f} {r.tau_G_us:>7.2f} "
                  f"{factor:>6.1f}x {r.cond_number:>10.2e}")
    print("==========================================")


if __name__ == "__main__":
    main()
