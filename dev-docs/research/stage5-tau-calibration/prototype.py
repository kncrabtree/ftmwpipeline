"""Stage-5 tau calibration via sliding-active-window STFT — Phase 1 research.

Reproduces every figure under ``figures/`` and every datum cited in
``report.md``. Run from the repository root with the project conda env:

    conda run -n ftmwpipeline-dev python \
        dev-docs/research/stage5-tau-calibration/prototype.py

The script validates a data-driven tau calibration scheme on controlled
synthetic FIDs. The seven Phase-1 cases from the planning doc:

1. Single isolated strong line — tau_truth sweep × T_full sweep.
2. SNR sweep at fixed tau.
3. Isolated clock spur.
4. Spur adjacent to a real line.
5. Dense cluster (Stage 4 mega-window analogue).
6. Bimodal tau population (velocity slip).
7. Voigt-deficit cos^2 theta shape error.

Plus two pathological corner studies:

- Long tau at short T_full (decay invisible in the FID window).
- Short tau at long T_full (signal fully decayed before later frames).

Outputs:

- figures/   one PNG per case + diagnostic overviews.
- data/      .npz cache for fast figure regeneration.
- report.md  worked through cases with empirical answers.

See ``dev-docs/planning/stage2b-tau-calibration.md`` for the design
narrative and Phase 1 acceptance gate.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
FIG = HERE / "figures"
DATA = HERE / "data"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
REPO_ROOT = Path(__file__).resolve().parents[3]
RNG_SEED = 20260525

# Sample dt and probe matched to 2638 (CLAUDE.md).
SAMPLE_DT_US = 0.020
PROBE_MHZ = 40000.0

logger = logging.getLogger("tau-calibration-research")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# FID synthesis helpers
# ---------------------------------------------------------------------------
def synth_fid(
    n_samples: int,
    sample_dt_us: float,
    line_bins: Sequence[int],
    line_taus_us: Sequence[float],
    line_snrs: Sequence[float],
    *,
    rng: np.random.Generator,
    spur_bins: Sequence[int] = (),
    spur_snrs: Sequence[float] = (),
    voigt_tau_g_us: Optional[float] = None,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Construct a noisy real-valued FID with controlled lines and optional spurs.

    Returns ``(fid_noisy, sigma_time, sigma_x_per_bin)`` where ``sigma_time``
    is the time-domain RMS noise added and ``sigma_x_per_bin`` is the
    analytic |X|-RMS noise on the full-record rfft (per-bin, single scalar
    since white time-domain noise is uniform across the rfft grid).

    Parameters
    ----------
    n_samples : int
        FID length (number of samples).
    sample_dt_us : float
        Sample spacing in microseconds.
    line_bins : sequence of int
        Baseband bin indices for the damped-cosine lines.
    line_taus_us : sequence of float
        Per-line tau_truth (µs).
    line_snrs : sequence of float
        Per-line peak SNR (= peak |X| / sigma_x). Matches the matched-
        filter prototype's convention.
    spur_bins, spur_snrs : sequence
        CW tones (no decay) injected at integer baseband bins.
    voigt_tau_g_us : float, optional
        Voigt-style Gaussian decay time (µs). When set, each line decays
        as ``exp(-t/tau) · exp(-(t/tau_g)²)`` instead of pure exponential.
        Models the Voigt deficit (Lorentzian × Gaussian convolution in
        frequency domain ↔ product in time domain).
    """
    T_full_us = n_samples * sample_dt_us
    t_us = np.arange(n_samples) * sample_dt_us

    # |X|-RMS noise per bin for white time-domain noise: σ_x = σ_t · √(T·dt).
    # Pick σ_t so each line achieves its requested on-line SNR (= |X|_peak/σ_x).
    fid = np.zeros(n_samples, dtype=float)
    line_bins = np.asarray(line_bins, dtype=int)
    line_taus_us = np.asarray(line_taus_us, dtype=float)
    line_snrs = np.asarray(line_snrs, dtype=float)
    spur_bins = np.asarray(spur_bins, dtype=int)
    spur_snrs = np.asarray(spur_snrs, dtype=float)

    if line_bins.size > 0:
        f_bb_lines = line_bins.astype(float) / T_full_us  # MHz
        tau_eff = line_taus_us * (1.0 - np.exp(-T_full_us / line_taus_us))
        # Normalise amplitudes so each line's on-line |X| ∝ SNR_i.
        # On-line |X|_i ≈ amp_i · tau_eff_i / 2 (with the dt·rfft convention,
        # absorbed below into σ_x). Choose amp_i = SNR_i / tau_eff_i; the
        # absolute scale is set by σ_x calibration after the FID is built.
        amps = line_snrs / tau_eff
        phases = rng.uniform(0, 2.0 * np.pi, size=line_bins.size)
        for i in range(line_bins.size):
            envelope = np.exp(-t_us / line_taus_us[i])
            if voigt_tau_g_us is not None and voigt_tau_g_us > 0.0:
                envelope = envelope * np.exp(-((t_us / voigt_tau_g_us) ** 2))
            fid += (
                amps[i]
                * np.cos(2.0 * np.pi * f_bb_lines[i] * t_us + phases[i])
                * envelope
            )
        # σ_x calibrated against the maximum-SNR line so its on-line |X|
        # equals max_snr · σ_x. Other lines inherit the same calibration
        # since amps scale by SNR_i / tau_eff_i.
        spec_clean = sample_dt_us * np.fft.rfft(fid)
        i_strongest = int(np.argmax(line_snrs))
        on_line_mag = np.abs(spec_clean[line_bins[i_strongest]])
        sigma_x = on_line_mag / float(line_snrs[i_strongest])
    else:
        sigma_x = 1.0

    # σ_t from σ_x: σ_x = σ_t · √(T_full·dt) · dt (since dt·rfft amplitude
    # convention). Re-derive: variance of dt·rfft bin = dt² · variance of
    # rfft bin = dt² · n_samples/2 · σ_t² (for the bulk bins). So
    # σ_x = dt · σ_t · √(n_samples/2). Solve for σ_t.
    sigma_time = sigma_x / (sample_dt_us * np.sqrt(n_samples / 2.0))

    # Add spurs (CW tones, no decay).
    if spur_bins.size > 0:
        f_bb_spurs = spur_bins.astype(float) / T_full_us
        spur_phases = rng.uniform(0, 2.0 * np.pi, size=spur_bins.size)
        for j in range(spur_bins.size):
            # Spur amplitude tuned so its on-line |X| = snr · σ_x.
            # CW tone of amplitude A_s contributes A_s · T_full / 2 to the
            # on-line bin's |X| (rectangle of length T_full * dt with the
            # dt·rfft normalisation gives A_s · (n·dt) / 2).
            A_s = 2.0 * spur_snrs[j] * sigma_x / (n_samples * sample_dt_us)
            fid += A_s * np.cos(2.0 * np.pi * f_bb_spurs[j] * t_us + spur_phases[j])

    noise = rng.normal(scale=sigma_time, size=n_samples)
    return fid + noise, sigma_time, np.full_like(np.fft.rfftfreq(n_samples, d=sample_dt_us), sigma_x)


# ---------------------------------------------------------------------------
# STFT calibration core
# ---------------------------------------------------------------------------
@dataclass
class STFTCalibrationResult:
    """Per-bin STFT calibration output."""
    n_seg: int
    sample_dt_us: float
    a_centers_us: np.ndarray         # (n_seg,) frame mid-times
    freq_bb_mhz: np.ndarray          # (n_bins,) baseband frequencies (MHz)
    mag: np.ndarray                  # (n_seg, n_bins)
    sigma_frame: np.ndarray          # (n_bins,) per-frame |X|-RMS per bin
    tau_per_bin: np.ndarray          # (n_bins,) recovered tau
    C_per_bin: np.ndarray            # (n_bins,) recovered C
    rss_exp: np.ndarray              # exponential RSS per bin
    rss_const: np.ndarray            # constant RSS per bin
    aicc_exp: np.ndarray
    aicc_const: np.ndarray
    classification: np.ndarray       # 0=discard, 1=spur, 2=bad-fit, 3=contributor
    snr_per_bin: np.ndarray          # max(mag) / mean(sigma_frame)
    t_sigma: float
    tau_max_us: float
    contributor_bins: np.ndarray
    contributor_taus: np.ndarray
    contributor_snrs: np.ndarray
    spur_bins: np.ndarray


def sliding_stft(
    fid: np.ndarray, sample_dt_us: float, n_seg: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding-active-window STFT (zero-pad to full record).

    Returns ``(mag, a_centers_us, freq_bb_mhz)``. ``mag`` is shape
    ``(n_seg, n_bins)``.
    """
    N = fid.size
    Nw = N // n_seg
    if Nw < 4:
        raise ValueError(f"n_seg={n_seg} too large for N={N} (Nw={Nw})")
    a_centers = np.empty(n_seg, dtype=float)
    n_bins = N // 2 + 1
    mag = np.empty((n_seg, n_bins), dtype=float)
    padded = np.zeros(N, dtype=fid.dtype)
    for k in range(n_seg):
        a_start = k * Nw
        a_end = a_start + Nw
        padded[:] = 0.0
        padded[a_start:a_end] = fid[a_start:a_end]
        spec = sample_dt_us * np.fft.rfft(padded)
        mag[k] = np.abs(spec)
        a_centers[k] = (a_start + (Nw - 1) * 0.5) * sample_dt_us
    freq_bb = np.fft.rfftfreq(N, d=sample_dt_us)  # MHz since dt is µs
    return mag, a_centers, freq_bb


def _fit_exp_per_bin(
    mag: np.ndarray, a_centers_us: np.ndarray,
    *, tau_clip_us: Tuple[float, float] = (0.1, 1e4),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted log-linear fit ``log|S_n| = log C - a/τ`` per bin.

    Weights are ``|S_n|^2`` (variance of log|S_n| under Gaussian noise on
    |S_n| is ≈ σ^2 / |S_n|^2). Returns ``(tau, C, rss_lin)`` where rss_lin
    is the residual SS in linear |S_n| space (so it's comparable against
    the constant-model RSS).
    """
    n_seg, n_bins = mag.shape
    safe = np.clip(mag, 1e-300, None)
    log_m = np.log(safe)
    w = mag ** 2
    a = a_centers_us[:, None]
    Sw = w.sum(axis=0)
    Swa = (w * a).sum(axis=0)
    Swl = (w * log_m).sum(axis=0)
    Swaa = (w * a * a).sum(axis=0)
    Swal = (w * a * log_m).sum(axis=0)
    denom = Sw * Swaa - Swa ** 2
    good = denom > 1e-30
    slope = np.where(good, (Sw * Swal - Swa * Swl) / np.where(good, denom, 1.0), 0.0)
    intercept = np.where(Sw > 0, (Swl - slope * Swa) / np.where(Sw > 0, Sw, 1.0), 0.0)
    # τ from slope; clip; slope >= 0 → no decay → assign tau_max.
    tau = np.where(slope < 0, -1.0 / np.where(slope < 0, slope, -1.0), tau_clip_us[1])
    tau = np.clip(tau, tau_clip_us[0], tau_clip_us[1])
    C = np.exp(np.clip(intercept, -50.0, 50.0))
    pred = C[None, :] * np.exp(-a / tau[None, :])
    rss = ((mag - pred) ** 2).sum(axis=0)
    return tau, C, rss


def _aicc(rss: np.ndarray, n: int, k: int) -> np.ndarray:
    """Small-sample-corrected AIC under Gaussian residuals.

    Uses ``AIC = n*log(RSS/n) + 2k`` and the small-n correction
    ``+ 2k(k+1)/(n-k-1)`` (NaN if n <= k+1).
    """
    if n - k - 1 <= 0:
        correction = np.inf
    else:
        correction = 2.0 * k * (k + 1) / (n - k - 1)
    rss_safe = np.where(rss > 0, rss, 1e-300)
    return n * np.log(rss_safe / n) + 2 * k + correction


def stft_calibration(
    fid: np.ndarray,
    sample_dt_us: float,
    *,
    n_seg: int = 10,
    sigma_time: Optional[float] = None,
    t_sigma: float = 5.0,
    tau_max_us: Optional[float] = None,
    rss_gate_factor: float = 5.0,
) -> STFTCalibrationResult:
    """Run the sliding-active-window STFT calibration on a single FID.

    Parameters
    ----------
    fid : np.ndarray
        Raw FID samples (length N).
    sample_dt_us : float
        Sample spacing, microseconds.
    n_seg : int
        Number of non-overlapping STFT frames. T_w = T_full / n_seg.
    sigma_time : float, optional
        Time-domain white-noise RMS. When None, estimate from the
        off-line bins of the full-record FT (MAD on the upper half of
        the magnitude distribution). For synthetic studies pass the
        known σ_t.
    t_sigma : float
        Above-threshold gate factor on per-frame noise.
    tau_max_us : float, optional
        Upper bound on τ; bins fitting above 0.95·tau_max are flagged
        as spur-candidates. Default = 5·T_full.
    rss_gate_factor : float
        Per-frame χ² gate for "bad-fit" classification. RSS_exp must
        be ≤ rss_gate_factor · n_seg · σ_frame² (mean across bins).
    """
    N = fid.size
    T_full_us = N * sample_dt_us
    if tau_max_us is None:
        tau_max_us = 5.0 * T_full_us
    mag, a_centers_us, freq_bb_mhz = sliding_stft(fid, sample_dt_us, n_seg)

    # Per-bin noise estimate.
    if sigma_time is None:
        # Estimate σ_x_full from MAD of the full-record FT magnitude on
        # the upper-frequency baseband (a rough off-line region).
        spec_full = sample_dt_us * np.fft.rfft(fid)
        mag_full = np.abs(spec_full)
        # Pick the highest-frequency third as off-line proxy.
        i_lo = (2 * mag_full.size) // 3
        sigma_x_full = 1.4826 * np.median(np.abs(mag_full[i_lo:] - np.median(mag_full[i_lo:])))
    else:
        # Analytic: σ_x = σ_t · dt · √(N/2).
        sigma_x_full = sigma_time * sample_dt_us * np.sqrt(N / 2.0)
    # Per-frame noise: σ_frame = σ_x_full / √n_seg (zero-padded T_w-frame
    # has noise variance T_w / T_full times the full-record bin variance).
    sigma_frame_scalar = sigma_x_full / np.sqrt(n_seg)
    sigma_frame = np.full(mag.shape[1], sigma_frame_scalar, dtype=float)

    # Per-bin fits.
    tau, C, rss_exp = _fit_exp_per_bin(
        mag, a_centers_us, tau_clip_us=(0.1, tau_max_us)
    )
    mean_m = mag.mean(axis=0)
    rss_const = ((mag - mean_m[None, :]) ** 2).sum(axis=0)

    aicc_exp = _aicc(rss_exp, n_seg, k=2)
    aicc_const = _aicc(rss_const, n_seg, k=1)

    # Bin SNR for the above-threshold gate.
    max_m = mag.max(axis=0)
    snr_per_bin = max_m / sigma_frame
    above = snr_per_bin >= t_sigma

    # Spur: constant wins by AICc OR tau saturates upper bound.
    spur_by_aicc = aicc_const + 2.0 < aicc_exp  # ΔAICc ≥ 2 favouring const
    spur_by_tau = tau >= 0.95 * tau_max_us
    is_spur = above & (spur_by_aicc | spur_by_tau)

    # Bad fit: RSS_exp too large relative to the signal level. Use a
    # *relative* gate (rss_exp > rss_gate_factor · (0.05 · mag_mean)²·n_seg)
    # in parallel with an absolute noise-floor gate, and only flag if BOTH
    # fail. The absolute gate alone over-classifies high-SNR clean fits
    # as bad-fit because the log-linear weighted regression doesn't
    # minimise linear-space RSS — its prediction error scales with the
    # signal level, not the noise level.
    mag_mean = mag.mean(axis=0)
    rss_gate_abs = rss_gate_factor * n_seg * sigma_frame ** 2
    rss_gate_rel = rss_gate_factor * n_seg * (0.05 * mag_mean) ** 2
    rss_gate = np.maximum(rss_gate_abs, rss_gate_rel)
    bad_fit = above & ~is_spur & (rss_exp > rss_gate)

    # Contributor: above threshold, not spur, not bad fit.
    contributor = above & ~is_spur & ~bad_fit

    classification = np.where(
        contributor, 3,
        np.where(bad_fit, 2, np.where(is_spur, 1, 0))
    )

    contributor_bins = np.where(contributor)[0]
    contributor_taus = tau[contributor_bins]
    contributor_snrs = snr_per_bin[contributor_bins]
    spur_bin_idx = np.where(is_spur)[0]

    return STFTCalibrationResult(
        n_seg=n_seg, sample_dt_us=sample_dt_us,
        a_centers_us=a_centers_us, freq_bb_mhz=freq_bb_mhz,
        mag=mag, sigma_frame=sigma_frame,
        tau_per_bin=tau, C_per_bin=C,
        rss_exp=rss_exp, rss_const=rss_const,
        aicc_exp=aicc_exp, aicc_const=aicc_const,
        classification=classification, snr_per_bin=snr_per_bin,
        t_sigma=t_sigma, tau_max_us=tau_max_us,
        contributor_bins=contributor_bins,
        contributor_taus=contributor_taus,
        contributor_snrs=contributor_snrs,
        spur_bins=spur_bin_idx,
    )


def majority_tau(
    contributor_taus: np.ndarray,
    contributor_snrs: Optional[np.ndarray] = None,
    *, weighted: bool = True,
) -> Tuple[float, float]:
    """Robust majority τ and spread σ_τ from contributor histogram.

    SNR-weighted by default — the empirical Phase-1 case-1 result showed
    that strong-line skirts cluster tightly around the truth while
    near-threshold bins are biased high by noisy log-linear fits, so the
    weighted-median is what we want. Returns ``(tau_maj, sigma_tau)``
    with σ_τ = (weighted) IQR / 1.349.
    """
    if contributor_taus.size == 0:
        return float("nan"), float("nan")
    if weighted and contributor_snrs is not None and contributor_snrs.sum() > 0:
        order = np.argsort(contributor_taus)
        t = contributor_taus[order]
        w = contributor_snrs[order]
        cdf = np.cumsum(w) / w.sum()
        tau_maj = float(np.interp(0.5, cdf, t))
        q25 = float(np.interp(0.25, cdf, t))
        q75 = float(np.interp(0.75, cdf, t))
    else:
        tau_maj = float(np.median(contributor_taus))
        q25, q75 = np.percentile(contributor_taus, [25, 75])
    sigma_tau = (q75 - q25) / 1.349
    return tau_maj, sigma_tau


def gmm_bimodality(
    contributor_taus: np.ndarray, max_iter: int = 200,
) -> dict:
    """Fit a 1- and 2-component Gaussian mixture; report AIC preference.

    Hand-rolled EM (no sklearn dependency). Returns dict with components
    of each fit and the AIC difference (positive → 2-component preferred).
    """
    x = np.asarray(contributor_taus, dtype=float)
    n = x.size
    if n < 20:
        return {"n": n, "delta_aic": float("nan"), "two_component_preferred": False}

    # 1-component MLE: μ, σ.
    mu1 = float(np.mean(x))
    var1 = float(np.var(x))
    var1 = max(var1, 1e-12)
    ll1 = -0.5 * n * (np.log(2 * np.pi * var1) + 1.0)
    aic1 = 2 * 2 - 2 * ll1  # 2 params (μ, σ).

    # 2-component EM init: split at median.
    med = float(np.median(x))
    mu_a, mu_b = float(np.mean(x[x < med])), float(np.mean(x[x >= med]))
    var_a = float(np.var(x[x < med])) or var1
    var_b = float(np.var(x[x >= med])) or var1
    pi_a = 0.5
    eps = 1e-12
    for _ in range(max_iter):
        # E-step.
        ga = pi_a / np.sqrt(2 * np.pi * var_a) * np.exp(-0.5 * (x - mu_a) ** 2 / var_a)
        gb = (1 - pi_a) / np.sqrt(2 * np.pi * var_b) * np.exp(-0.5 * (x - mu_b) ** 2 / var_b)
        denom = ga + gb + eps
        wa = ga / denom
        wb = gb / denom
        Na = wa.sum() + eps
        Nb = wb.sum() + eps
        # M-step.
        mu_a_new = (wa * x).sum() / Na
        mu_b_new = (wb * x).sum() / Nb
        var_a_new = max((wa * (x - mu_a_new) ** 2).sum() / Na, 1e-12)
        var_b_new = max((wb * (x - mu_b_new) ** 2).sum() / Nb, 1e-12)
        pi_a_new = Na / n
        if (abs(mu_a_new - mu_a) + abs(mu_b_new - mu_b)) < 1e-9:
            mu_a, mu_b, var_a, var_b, pi_a = mu_a_new, mu_b_new, var_a_new, var_b_new, pi_a_new
            break
        mu_a, mu_b, var_a, var_b, pi_a = mu_a_new, mu_b_new, var_a_new, var_b_new, pi_a_new
    # 2-comp log-likelihood.
    g_a = pi_a / np.sqrt(2 * np.pi * var_a) * np.exp(-0.5 * (x - mu_a) ** 2 / var_a)
    g_b = (1 - pi_a) / np.sqrt(2 * np.pi * var_b) * np.exp(-0.5 * (x - mu_b) ** 2 / var_b)
    ll2 = np.sum(np.log(g_a + g_b + eps))
    aic2 = 2 * 5 - 2 * ll2  # 5 params (μ_a, σ_a, μ_b, σ_b, π_a).
    delta_aic = aic1 - aic2  # positive → 2-comp preferred.
    return {
        "n": int(n),
        "mu1": mu1, "sigma1": float(np.sqrt(var1)),
        "mu_a": float(mu_a), "sigma_a": float(np.sqrt(var_a)),
        "mu_b": float(mu_b), "sigma_b": float(np.sqrt(var_b)),
        "pi_a": float(pi_a),
        "aic1": float(aic1), "aic2": float(aic2),
        "delta_aic": float(delta_aic),
        "two_component_preferred": bool(delta_aic > 4.0),
    }


# ---------------------------------------------------------------------------
# Synthetic case helpers
# ---------------------------------------------------------------------------
def make_single_line_fid(
    T_full_us: float, tau_truth_us: float, snr: float,
    *, rng: np.random.Generator, sample_dt_us: float = SAMPLE_DT_US,
    n_seg: int = 10,
) -> Tuple[np.ndarray, dict]:
    """Build a single-line FID at a random bin near the band centre."""
    N = int(round(T_full_us / sample_dt_us))
    # Ensure N is divisible by n_seg (so frames don't drop samples).
    N = (N // n_seg) * n_seg
    # Pick a bin away from edges.
    bin_idx = int(N // 4 + rng.integers(0, N // 4))
    fid, sigma_t, _ = synth_fid(
        n_samples=N, sample_dt_us=sample_dt_us,
        line_bins=[bin_idx], line_taus_us=[tau_truth_us], line_snrs=[snr],
        rng=rng,
    )
    return fid, {
        "n_samples": N, "T_full_us": N * sample_dt_us,
        "tau_truth_us": tau_truth_us, "snr": snr,
        "line_bin": bin_idx, "sigma_time": sigma_t,
    }


def make_multi_line_fid(
    T_full_us: float, line_taus_us: Sequence[float], line_snrs: Sequence[float],
    *, rng: np.random.Generator, sample_dt_us: float = SAMPLE_DT_US,
    n_seg: int = 10, min_sep_bins: int = 25,
    spur_bins: Sequence[int] = (), spur_snrs: Sequence[float] = (),
    voigt_tau_g_us: Optional[float] = None,
    cluster_centre_bin: Optional[int] = None,
    cluster_span_bins: Optional[int] = None,
) -> Tuple[np.ndarray, dict]:
    """Build a multi-line FID with controllable line placement."""
    N = int(round(T_full_us / sample_dt_us))
    N = (N // n_seg) * n_seg
    n_lines = len(line_taus_us)
    edge = max(20, N // 16)
    if cluster_centre_bin is not None and cluster_span_bins is not None:
        # Pack lines uniformly inside the cluster span.
        lo = cluster_centre_bin - cluster_span_bins // 2
        hi = cluster_centre_bin + cluster_span_bins // 2
        line_bins = np.linspace(lo, hi, n_lines).round().astype(int).tolist()
    else:
        line_bins: list[int] = []
        attempts = 0
        while len(line_bins) < n_lines and attempts < 5000:
            attempts += 1
            b = int(rng.integers(edge, N // 2 - edge))
            if any(abs(b - c) < min_sep_bins for c in line_bins):
                continue
            line_bins.append(b)
        line_bins.sort()
    # Trim line_taus_us / line_snrs to match if placement fell short.
    placed = len(line_bins)
    line_taus_us = list(line_taus_us)[:placed]
    line_snrs = list(line_snrs)[:placed]
    fid, sigma_t, _ = synth_fid(
        n_samples=N, sample_dt_us=sample_dt_us,
        line_bins=line_bins, line_taus_us=line_taus_us, line_snrs=line_snrs,
        rng=rng, spur_bins=spur_bins, spur_snrs=spur_snrs,
        voigt_tau_g_us=voigt_tau_g_us,
    )
    return fid, {
        "n_samples": N, "T_full_us": N * sample_dt_us,
        "line_bins": line_bins, "line_taus": list(line_taus_us),
        "line_snrs": list(line_snrs), "sigma_time": sigma_t,
        "spur_bins": list(spur_bins), "spur_snrs": list(spur_snrs),
        "voigt_tau_g_us": voigt_tau_g_us,
    }


# ===========================================================================
# Phase 1: case studies
# ===========================================================================
def case1_tau_sweep() -> dict:
    """Case 1: single isolated strong line. τ_truth × T_full × N_seg sweep.

    For each (T_full, τ_truth, n_seg) cell, run the calibration on the
    line's bin and report recovery error vs ground truth. The output
    decides the operating-point N_seg.
    """
    logger.info("=== Case 1: single-line tau sweep ===")
    T_fulls = [5.0, 10.0, 12.65, 20.0, 30.0, 40.0]
    tau_truths = [3.0, 5.0, 7.5, 12.0, 20.0]
    n_segs = [6, 8, 10, 12, 16, 20]
    snr = 100.0
    n_trials = 8

    rows = []
    for T_full in T_fulls:
        for tau_truth in tau_truths:
            for n_seg in n_segs:
                # Skip degenerate cells: Nw < 4 samples.
                N = int(round(T_full / SAMPLE_DT_US))
                if (N // n_seg) < 4:
                    continue
                err_pcts = []
                maj_err_pcts = []
                cluster_size = []
                for trial in range(n_trials):
                    rng = np.random.default_rng(RNG_SEED + 1000 * trial + int(T_full * 100))
                    fid, meta = make_single_line_fid(
                        T_full, tau_truth, snr, rng=rng, n_seg=n_seg,
                    )
                    cal = stft_calibration(
                        fid, SAMPLE_DT_US, n_seg=n_seg,
                        sigma_time=meta["sigma_time"],
                    )
                    if cal.contributor_taus.size == 0:
                        continue
                    # SNR-weighted majority tau (primary metric):
                    tau_maj, _ = majority_tau(
                        cal.contributor_taus, cal.contributor_snrs,
                    )
                    maj_err_pcts.append(100.0 * (tau_maj - tau_truth) / tau_truth)
                    # On-line bin (diagnostic):
                    bin_idx = meta["line_bin"]
                    if cal.classification[bin_idx] == 3:
                        tau_hat = cal.tau_per_bin[bin_idx]
                        err_pcts.append(100.0 * (tau_hat - tau_truth) / tau_truth)
                    cluster_size.append(cal.contributor_taus.size)
                if maj_err_pcts:
                    rows.append({
                        "T_full": T_full, "tau_truth": tau_truth, "n_seg": n_seg,
                        "median_err_pct": float(np.median(maj_err_pcts)),
                        "iqr_err_pct": float(np.percentile(maj_err_pcts, 75)
                                             - np.percentile(maj_err_pcts, 25)),
                        "median_online_err_pct": float(np.median(err_pcts))
                            if err_pcts else float("nan"),
                        "trials": len(maj_err_pcts),
                        "mean_contributors": float(np.mean(cluster_size)),
                    })
    arr = rows  # keep as list of dicts
    np.savez(DATA / "case1_tau_sweep.npz", rows=np.asarray(json.dumps(arr)))

    # Figure: heatmap of median error for each (T_full, tau_truth) at the
    # recommended N_seg = 10, plus a side panel showing how error varies
    # with N_seg at one cell (T_full=12.65, tau=7.5).
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    n_seg_target = 10
    heat_T = T_fulls
    heat_tau = tau_truths
    Z = np.full((len(heat_tau), len(heat_T)), np.nan)
    for r in rows:
        if r["n_seg"] != n_seg_target:
            continue
        i = heat_tau.index(r["tau_truth"])
        j = heat_T.index(r["T_full"])
        Z[i, j] = r["median_err_pct"]
    im = axes[0].imshow(Z, aspect="auto", origin="lower",
                        cmap="RdBu_r", vmin=-30, vmax=30,
                        extent=[-0.5, len(heat_T) - 0.5, -0.5, len(heat_tau) - 0.5])
    axes[0].set_xticks(range(len(heat_T)))
    axes[0].set_xticklabels([f"{t:g}" for t in heat_T])
    axes[0].set_yticks(range(len(heat_tau)))
    axes[0].set_yticklabels([f"{t:g}" for t in heat_tau])
    axes[0].set_xlabel("T_full (µs)")
    axes[0].set_ylabel("tau_truth (µs)")
    axes[0].set_title(f"Median tau recovery error (%) at N_seg={n_seg_target}")
    for i in range(len(heat_tau)):
        for j in range(len(heat_T)):
            if not np.isnan(Z[i, j]):
                axes[0].text(j, i, f"{Z[i, j]:+.1f}",
                             ha="center", va="center",
                             color="white" if abs(Z[i, j]) > 15 else "black",
                             fontsize=8)
    plt.colorbar(im, ax=axes[0], label="median error (%)")

    # Side panel: N_seg sweep at the canonical 2638-like cell.
    for tau_t in tau_truths:
        ax_data = [(r["n_seg"], r["median_err_pct"]) for r in rows
                   if r["T_full"] == 12.65 and r["tau_truth"] == tau_t]
        ax_data.sort()
        if ax_data:
            xs, ys = zip(*ax_data)
            axes[1].plot(xs, ys, "o-", label=f"τ={tau_t} µs")
    axes[1].axhline(5, color="grey", ls=":", lw=0.8, alpha=0.7)
    axes[1].axhline(-5, color="grey", ls=":", lw=0.8, alpha=0.7)
    axes[1].set_xlabel("N_seg")
    axes[1].set_ylabel("median tau recovery error (%)")
    axes[1].set_title("N_seg sensitivity at T_full=12.65 µs")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "01_tau_sweep.png", dpi=110)
    plt.close(fig)

    return {"rows": rows, "n_seg_chosen": n_seg_target}


def case2_snr_sweep() -> dict:
    """Case 2: SNR sweep at τ=7.5 µs, T_full=12.65 µs, N_seg=10."""
    logger.info("=== Case 2: SNR sweep ===")
    snrs = [5, 10, 20, 50, 100, 500, 1000, 10000]
    tau_truth = 7.5
    T_full = 12.65
    n_seg = 10
    n_trials = 20

    rows = []
    for snr in snrs:
        maj_errs = []
        online_errs = []
        fit_rates = []
        contribs = []
        for trial in range(n_trials):
            rng = np.random.default_rng(RNG_SEED + 5000 + trial * 13)
            fid, meta = make_single_line_fid(
                T_full, tau_truth, float(snr), rng=rng, n_seg=n_seg,
            )
            cal = stft_calibration(
                fid, SAMPLE_DT_US, n_seg=n_seg, sigma_time=meta["sigma_time"],
            )
            bin_idx = meta["line_bin"]
            fit_rates.append(1.0 if cal.classification[bin_idx] == 3 else 0.0)
            contribs.append(cal.contributor_bins.size)
            if cal.contributor_bins.size > 0:
                tau_maj, _ = majority_tau(cal.contributor_taus, cal.contributor_snrs)
                maj_errs.append(100.0 * (tau_maj - tau_truth) / tau_truth)
            if cal.classification[bin_idx] == 3:
                online_errs.append(
                    100.0 * (cal.tau_per_bin[bin_idx] - tau_truth) / tau_truth
                )
        rows.append({
            "snr": snr,
            "median_err_pct": float(np.median(maj_errs)) if maj_errs else float("nan"),
            "iqr_pct": float(np.percentile(maj_errs, 75) - np.percentile(maj_errs, 25))
                if len(maj_errs) >= 4 else float("nan"),
            "median_online_err_pct": float(np.median(online_errs))
                if online_errs else float("nan"),
            "fit_rate": float(np.mean(fit_rates)),
            "mean_contributors": float(np.mean(contribs)),
            "n_fit": len(maj_errs),
        })
    snr_floor = next((r["snr"] for r in rows if r["fit_rate"] >= 0.9), None)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].errorbar(
        [r["snr"] for r in rows], [r["median_err_pct"] for r in rows],
        yerr=[r["iqr_pct"] / 2 if not np.isnan(r["iqr_pct"]) else 0 for r in rows],
        fmt="o-",
    )
    axes[0].axhspan(-5, 5, color="green", alpha=0.15)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("line SNR")
    axes[0].set_ylabel("tau recovery error (%)")
    axes[0].set_title("τ_recover vs SNR (single line, τ=7.5 µs, T=12.65 µs)")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].plot([r["snr"] for r in rows], [r["fit_rate"] for r in rows], "o-")
    axes[1].set_xscale("log")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].axhline(0.9, color="grey", ls=":", lw=0.8)
    axes[1].set_xlabel("line SNR")
    axes[1].set_ylabel("contributor classification rate")
    axes[1].set_title(f"On-line classification rate (floor ≈ SNR {snr_floor})")
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "02_snr_sweep.png", dpi=110)
    plt.close(fig)
    return {"rows": rows, "snr_floor": snr_floor}


def case3_isolated_spur() -> dict:
    """Case 3: lone CW spur — verify spur classification fires."""
    logger.info("=== Case 3: isolated clock spur ===")
    T_full = 12.65
    n_seg = 10
    N = (int(round(T_full / SAMPLE_DT_US)) // n_seg) * n_seg
    spur_bin = N // 3
    rng = np.random.default_rng(RNG_SEED + 9001)
    # Include one real line so sigma normalisation makes sense.
    line_bin = N // 5
    fid, sigma_t, _ = synth_fid(
        n_samples=N, sample_dt_us=SAMPLE_DT_US,
        line_bins=[line_bin], line_taus_us=[7.5], line_snrs=[50.0],
        rng=rng, spur_bins=[spur_bin], spur_snrs=[100.0],
    )
    cal = stft_calibration(fid, SAMPLE_DT_US, n_seg=n_seg, sigma_time=sigma_t)

    spur_cls = int(cal.classification[spur_bin])
    line_cls = int(cal.classification[line_bin])
    tau_line = float(cal.tau_per_bin[line_bin])
    tau_spur = float(cal.tau_per_bin[spur_bin])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(cal.a_centers_us, cal.mag[:, spur_bin], "C3o-", label=f"spur bin (cls={spur_cls})")
    axes[0].plot(cal.a_centers_us, cal.mag[:, line_bin], "C0o-", label=f"line bin (cls={line_cls})")
    axes[0].set_xlabel("frame centre a_c (µs)")
    axes[0].set_ylabel("|S_n|")
    axes[0].set_title("Per-bin STFT decay (spur stays flat, line decays)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    # Classification map on the spectrum.
    cls = cal.classification
    colors = {0: "lightgrey", 1: "red", 2: "orange", 3: "green"}
    axes[1].plot(cal.freq_bb_mhz, cal.mag.mean(axis=0), "k-", lw=0.5)
    for c, col in colors.items():
        mask = cls == c
        if mask.any():
            axes[1].plot(cal.freq_bb_mhz[mask], cal.mag.mean(axis=0)[mask], ".", color=col,
                         ms=2, label={0: "discard", 1: "spur", 2: "bad-fit", 3: "contributor"}[c])
    axes[1].set_xlabel("baseband freq (MHz)")
    axes[1].set_ylabel("mean |S_n|")
    axes[1].set_title("Per-bin classification")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "03_isolated_spur.png", dpi=110)
    plt.close(fig)

    tau_maj_w, sigma_tau_w = majority_tau(
        cal.contributor_taus, cal.contributor_snrs,
    )
    return {
        "spur_classification": spur_cls, "line_classification": line_cls,
        "tau_spur": tau_spur, "tau_line": tau_line, "tau_max_us": cal.tau_max_us,
        "n_contributors": int(cal.contributor_bins.size),
        "n_spurs": int(cal.spur_bins.size),
        "tau_maj_weighted": tau_maj_w,
        "sigma_tau_weighted": sigma_tau_w,
    }


def case4_spur_near_line() -> dict:
    """Case 4: CW spur near a real damped cosine.

    The STFT method's relevant "near" radius is set by the STFT-frame
    rectangular sinc skirt (width 1/T_w = n_seg/T_full), NOT the natural
    Lorentzian FWHM (which is 1/(π·τ), much narrower at τ ≥ 1 µs). The
    planning doc's "±5 FWHM" criterion underestimates the contamination
    radius by ~n_seg/π. We test two separations:

    - Far: spur at line_bin + 60 (= 6 STFT bins for n_seg=10), well
      outside the spur's sinc-skirt envelope. Expectation: spur
      classified as spur, line classified as contributor with τ recovery
      within a few percent.
    - Close: spur at line_bin + 5 (= 0.5 STFT bins), inside the spur's
      sinc skirt. Expectation: line bin contaminated, classified as
      bad-fit. Mitigation downstream: extend the spur-bin exclusion
      radius by ±n_seg STFT bins around each detected spur.
    """
    logger.info("=== Case 4: spur next to real line ===")
    T_full = 12.65
    n_seg = 10
    N = (int(round(T_full / SAMPLE_DT_US)) // n_seg) * n_seg
    rng = np.random.default_rng(RNG_SEED + 9201)
    line_bin = N // 4
    results: dict = {}
    for label, offset in [("far", 60), ("close", 5)]:
        spur_bin = line_bin + offset
        rng2 = np.random.default_rng(RNG_SEED + 9201 + offset)
        fid, sigma_t, _ = synth_fid(
            n_samples=N, sample_dt_us=SAMPLE_DT_US,
            line_bins=[line_bin], line_taus_us=[7.5], line_snrs=[100.0],
            rng=rng2, spur_bins=[spur_bin], spur_snrs=[100.0],
        )
        cal = stft_calibration(fid, SAMPLE_DT_US, n_seg=n_seg, sigma_time=sigma_t)
        results[label] = {
            "offset_full_record_bins": offset,
            "offset_stft_bins": offset / n_seg,
            "spur_classification": int(cal.classification[spur_bin]),
            "line_classification": int(cal.classification[line_bin]),
            "tau_line": float(cal.tau_per_bin[line_bin]),
            "tau_spur": float(cal.tau_per_bin[spur_bin]),
            "n_contributors": int(cal.contributor_bins.size),
            "n_spurs": int(cal.spur_bins.size),
        }
    return results


def case5_dense_cluster() -> dict:
    """Case 5: six lines, all τ=7.5 µs, in a cluster spanning ~5 STFT bins.

    Lines spaced by 10 full-record bins (= 1 STFT bin at n_seg=10) so
    the cluster spans roughly 5 STFT bins — meaningfully blended on the
    STFT grid but still individually resolvable at the full-record Δf.
    Test: per-bin τ recovery on the on-line bins (should be ≈ 7.5);
    between-line bins should be classified as bad-fit (overlapping
    skirts).
    """
    logger.info("=== Case 5: dense cluster ===")
    T_full = 12.65
    n_seg = 10
    N = (int(round(T_full / SAMPLE_DT_US)) // n_seg) * n_seg
    n_lines = 6
    spacing_full_record_bins = 10  # ≈ 1 STFT bin at n_seg=10
    base_bin = N // 3
    line_bins = [base_bin + k * spacing_full_record_bins for k in range(n_lines)]
    rng = np.random.default_rng(RNG_SEED + 9501)
    fid, sigma_t, _ = synth_fid(
        n_samples=N, sample_dt_us=SAMPLE_DT_US,
        line_bins=line_bins,
        line_taus_us=[7.5] * n_lines, line_snrs=[100.0] * n_lines,
        rng=rng,
    )
    cal = stft_calibration(fid, SAMPLE_DT_US, n_seg=n_seg, sigma_time=sigma_t)
    on_taus = [float(cal.tau_per_bin[b]) for b in line_bins]
    on_cls = [int(cal.classification[b]) for b in line_bins]
    between_bins = [(line_bins[i] + line_bins[i + 1]) // 2 for i in range(n_lines - 1)]
    between_taus = [float(cal.tau_per_bin[b]) for b in between_bins]
    between_cls = [int(cal.classification[b]) for b in between_bins]
    return {
        "line_bins": line_bins, "on_taus": on_taus, "on_cls": on_cls,
        "between_bins": between_bins, "between_taus": between_taus,
        "between_cls": between_cls,
        "spacing_stft_bins": spacing_full_record_bins / n_seg,
        "n_contributors": int(cal.contributor_bins.size),
        "median_contributor_tau": float(np.median(cal.contributor_taus))
            if cal.contributor_bins.size else float("nan"),
        "tau_maj_weighted": majority_tau(
            cal.contributor_taus, cal.contributor_snrs,
        ),
    }


def case6_bimodal_population() -> dict:
    """Case 6: half lines at τ=5, half at τ=10. GMM bimodality test."""
    logger.info("=== Case 6: bimodal population ===")
    T_full = 20.0  # Longer T_full so τ=5 vs 10 are clearly separable.
    n_seg = 10
    rng = np.random.default_rng(RNG_SEED + 9701)
    n_total = 24
    line_taus = [5.0] * (n_total // 2) + [10.0] * (n_total // 2)
    rng.shuffle(line_taus)
    line_snrs = [200.0] * n_total
    fid, meta = make_multi_line_fid(
        T_full, line_taus_us=line_taus, line_snrs=line_snrs,
        rng=rng, n_seg=n_seg, min_sep_bins=15,
    )
    cal = stft_calibration(
        fid, SAMPLE_DT_US, n_seg=n_seg, sigma_time=meta["sigma_time"],
    )
    bm = gmm_bimodality(cal.contributor_taus)
    tau_maj_unimodal, sigma_tau_unimodal = majority_tau(
        cal.contributor_taus, cal.contributor_snrs,
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(cal.contributor_taus, bins=40, alpha=0.6, color="C0",
            label=f"contributors (n={cal.contributor_taus.size})")
    ax.axvline(5.0, color="C3", ls="--", label="τ_truth = 5 µs")
    ax.axvline(10.0, color="C2", ls="--", label="τ_truth = 10 µs")
    if not np.isnan(bm.get("mu_a", float("nan"))):
        ax.axvline(bm["mu_a"], color="C3", lw=0.8, alpha=0.7,
                   label=f"GMM μ_a={bm['mu_a']:.2f}")
        ax.axvline(bm["mu_b"], color="C2", lw=0.8, alpha=0.7,
                   label=f"GMM μ_b={bm['mu_b']:.2f}")
    ax.set_xlabel("recovered τ (µs)")
    ax.set_ylabel("count")
    ax.set_title(f"Bimodal population (ΔAIC = {bm.get('delta_aic', float('nan')):.1f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "06_bimodal_population.png", dpi=110)
    plt.close(fig)
    return {
        "n_contributors": int(cal.contributor_taus.size),
        "tau_maj_unimodal": tau_maj_unimodal,
        "sigma_tau_unimodal": sigma_tau_unimodal,
        "gmm": bm,
    }


def case7_voigt_deficit() -> dict:
    """Case 7: Voigt-deficit shape error (Lorentzian × Gaussian in time).

    Injects ``exp(-(t/τ_G)²)`` on top of the pure exponential decay. A
    pure Lorentzian (τ_G → ∞) is the no-deficit reference; smaller
    τ_G adds a Gaussian "early decay" component. The STFT fit returns
    a pure-exponential τ; we measure how the recovery shifts and
    whether per-bin τ values correlate with SNR (the cross-fixture-
    validation prediction: stronger lines show more bias because their
    deficit residual escapes the noise floor).

    Restricts the τ-vs-SNR correlation to the actual on-line bins
    (one per planted line) so the skirt-noise artefact from case 1
    doesn't mask the real signal.
    """
    logger.info("=== Case 7: Voigt-deficit shape error ===")
    T_full = 12.65
    n_seg = 10
    n_lines = 20
    line_snrs = (10 ** np.linspace(1.0, 3.0, n_lines)).tolist()  # SNR 10 → 1000
    line_taus = [7.5] * n_lines

    results = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, tau_g in zip(axes, [None, 30.0, 12.0]):
        rng2 = np.random.default_rng(RNG_SEED + 9901 + int((tau_g or 999) * 100))
        fid, meta = make_multi_line_fid(
            T_full, line_taus_us=line_taus, line_snrs=line_snrs,
            rng=rng2, n_seg=n_seg, min_sep_bins=15,
            voigt_tau_g_us=tau_g,
        )
        cal = stft_calibration(
            fid, SAMPLE_DT_US, n_seg=n_seg, sigma_time=meta["sigma_time"],
        )
        # On-line bin τ for each planted line.
        on_line_taus = []
        on_line_snrs = []
        for b in meta["line_bins"]:
            if cal.classification[b] == 3:
                on_line_taus.append(cal.tau_per_bin[b])
                on_line_snrs.append(cal.snr_per_bin[b])
        on_line_taus = np.asarray(on_line_taus)
        on_line_snrs = np.asarray(on_line_snrs)
        if on_line_taus.size >= 5:
            x = np.log10(np.clip(on_line_snrs, 1e-6, None))
            r_on = float(np.corrcoef(x, on_line_taus)[0, 1])
        else:
            r_on = float("nan")
        if cal.contributor_taus.size >= 5:
            x = np.log10(np.clip(cal.contributor_snrs, 1e-6, None))
            r_all = float(np.corrcoef(x, cal.contributor_taus)[0, 1])
        else:
            r_all = float("nan")
        tau_maj, sig_tau = majority_tau(cal.contributor_taus, cal.contributor_snrs)
        results[f"tau_g_{tau_g}"] = {
            "tau_g_us": tau_g,
            "n_on_line_fit": int(on_line_taus.size),
            "n_contributors": int(cal.contributor_taus.size),
            "pearson_r_on_line": r_on,
            "pearson_r_all_contribs": r_all,
            "tau_maj_weighted": tau_maj, "sigma_tau_weighted": sig_tau,
            "on_line_median_tau": float(np.median(on_line_taus))
                if on_line_taus.size else float("nan"),
            "on_line_iqr_tau": float(
                np.percentile(on_line_taus, 75) - np.percentile(on_line_taus, 25)
            ) if on_line_taus.size >= 4 else float("nan"),
        }
        ax.scatter(cal.contributor_snrs, cal.contributor_taus, s=4, alpha=0.3,
                   color="C0", label="contributors")
        ax.scatter(on_line_snrs, on_line_taus, s=30, marker="D", color="C3",
                   label="on-line bins")
        ax.set_xscale("log")
        ax.axhline(7.5, color="k", ls="--", lw=0.8)
        ax.set_xlabel("on-line SNR")
        if ax is axes[0]:
            ax.set_ylabel("recovered τ (µs)")
        ax.set_title(
            f"τ_G = {'∞ (pure Lorentzian)' if tau_g is None else f'{tau_g} µs'}"
            f"\nr_on={r_on:.2f}  τ_maj={tau_maj:.2f}"
        )
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "07_voigt_deficit.png", dpi=110)
    plt.close(fig)
    return results


def pathological_corners() -> dict:
    """Corner cases: long τ at short T_full, short τ at long T_full."""
    logger.info("=== Pathological corners ===")
    n_seg = 10
    n_trials = 12

    def _trial_block(T_full, tau_truth, snr=100.0):
        errs = []
        contribs = []
        on_line_fits = 0
        for trial in range(n_trials):
            rng = np.random.default_rng(RNG_SEED + 12000 + trial)
            fid, meta = make_single_line_fid(
                T_full, tau_truth, snr, rng=rng, n_seg=n_seg,
            )
            cal = stft_calibration(
                fid, SAMPLE_DT_US, n_seg=n_seg, sigma_time=meta["sigma_time"],
            )
            bin_idx = meta["line_bin"]
            if cal.classification[bin_idx] == 3:
                on_line_fits += 1
                errs.append(100.0 * (cal.tau_per_bin[bin_idx] - tau_truth) / tau_truth)
            contribs.append(cal.contributor_taus.size)
        return {
            "T_full": T_full, "tau_truth": tau_truth,
            "n_trials": n_trials, "on_line_fit_rate": on_line_fits / n_trials,
            "median_err_pct": float(np.median(errs)) if errs else float("nan"),
            "iqr_err_pct": float(
                np.percentile(errs, 75) - np.percentile(errs, 25)
            ) if len(errs) >= 4 else float("nan"),
            "mean_contributors": float(np.mean(contribs)),
        }

    long_tau_short_T = _trial_block(T_full=5.0, tau_truth=20.0)
    short_tau_long_T = _trial_block(T_full=40.0, tau_truth=3.0)
    return {
        "long_tau_short_T": long_tau_short_T,
        "short_tau_long_T": short_tau_long_T,
    }


# ===========================================================================
# Phase 2: 2638 application
# ===========================================================================
FIXTURE_PATH = (
    REPO_ROOT / "scratch" / "stage5-tau-calibration" / "exp_2638_unapodized.ftmw"
)


def phase2_2638() -> dict:
    """Phase 2: apply STFT calibration to the 2638 fixture.

    Loads the FID from the unapodized fixture (which must be built
    separately via ``ftmwpipeline.api.compute_ft(..., expf_us=None)``),
    extracts the active region [2.35, 15] µs, runs the calibration with
    N_seg = 10, and produces 2D STFT heatmap + per-bin τ histogram +
    diagnostics + statistical tests on the contributor distribution.
    """
    if not FIXTURE_PATH.exists():
        logger.warning("Phase 2 skipped — fixture missing: %s", FIXTURE_PATH)
        return {"skipped": True, "reason": "fixture missing"}

    logger.info("=== Phase 2: 2638 application ===")
    import ftmwpipeline.api as ftmw_api  # local import: avoids module-load cost when only Phase 1 runs.

    fid = ftmw_api.load_fid(str(FIXTURE_PATH))
    sample_dt_us = float(fid.spacing * 1e6)
    start_us = 2.35
    end_us = 15.0
    start_idx = int(round(start_us / sample_dt_us))
    end_idx = int(round(end_us / sample_dt_us))
    active = np.asarray(fid.data[start_idx:end_idx], dtype=float)
    T_full_us = active.size * sample_dt_us
    logger.info(
        "FID active region: %d samples, T_full = %.4f µs, sample_dt = %.4f ns",
        active.size, T_full_us, sample_dt_us * 1e3,
    )

    n_seg = 10
    # Trim active size to be divisible by n_seg.
    new_size = (active.size // n_seg) * n_seg
    active = active[:new_size]
    T_full_us = active.size * sample_dt_us

    # Estimate σ_t empirically from the FID tail (last ~30 % of the active
    # region). The Stage 2 noise estimator runs on the apodized
    # (expf_us=5) user FT by default — its |X|-RMS values are not in the
    # same amplitude convention as the unapodized STFT magnitudes here,
    # so we fall back to a direct time-domain noise measurement. The
    # active region's last few µs decays past detection for all but the
    # longest-lived lines (τ ≥ T_active) and is dominated by noise.
    tail_start = int(0.7 * active.size)
    tail = active[tail_start:]
    sigma_time = float(np.std(tail - tail.mean()))
    sigma_x_full = sigma_time * sample_dt_us * np.sqrt(active.size / 2.0)
    logger.info(
        "Empirical σ_t from tail %.2f-%.2f µs: %.5e",
        tail_start * sample_dt_us, active.size * sample_dt_us, sigma_time,
    )
    logger.info("Derived σ_x_full (unapodized): %.5e", sigma_x_full)

    logger.info("Running STFT (%d samples × %d frames)…", active.size, n_seg)
    t0 = time.perf_counter()
    cal = stft_calibration(
        active, sample_dt_us, n_seg=n_seg, sigma_time=sigma_time,
    )
    logger.info("STFT complete: %.1f sec", time.perf_counter() - t0)

    # Restrict analysis to the canonical molecular trim 26500-40000 MHz
    # (lower sideband, probe 40960). Baseband range:
    #   f_bb = probe - f_mol = 40960 - 40000 = 960 MHz (low end)
    #   f_bb = 40960 - 26500 = 14460 MHz (high end)
    probe_mhz = float(fid.probe_freq_mhz)
    f_bb_lo = probe_mhz - 40000.0  # 960 MHz
    f_bb_hi = probe_mhz - 26500.0  # 14460 MHz
    in_trim = (cal.freq_bb_mhz >= f_bb_lo) & (cal.freq_bb_mhz <= f_bb_hi)
    # Sub-set the contributors to the trim region.
    contrib_mask = np.zeros(cal.classification.size, dtype=bool)
    contrib_mask[cal.contributor_bins] = True
    contrib_in_trim = np.where(contrib_mask & in_trim)[0]
    taus_in_trim = cal.tau_per_bin[contrib_in_trim]
    snrs_in_trim = cal.snr_per_bin[contrib_in_trim]
    freqs_in_trim_mhz = probe_mhz - cal.freq_bb_mhz[contrib_in_trim]  # molecular
    logger.info("Contributors in trim: %d", contrib_in_trim.size)

    # Spurs in trim.
    spur_in_trim = cal.spur_bins[np.isin(cal.spur_bins, np.where(in_trim)[0])]
    spur_freqs_mol_mhz = probe_mhz - cal.freq_bb_mhz[spur_in_trim]
    logger.info("Spurs in trim: %d", spur_in_trim.size)

    # Statistics.
    tau_maj_w, sigma_tau_w = majority_tau(taus_in_trim, snrs_in_trim)
    bm = gmm_bimodality(taus_in_trim)
    # τ vs SNR (Pearson)
    if taus_in_trim.size >= 5:
        r_snr = float(np.corrcoef(np.log10(np.clip(snrs_in_trim, 1e-6, None)),
                                  taus_in_trim)[0, 1])
        r_freq = float(np.corrcoef(freqs_in_trim_mhz, taus_in_trim)[0, 1])
    else:
        r_snr = r_freq = float("nan")
    # Frequency thirds.
    third_med: dict = {}
    edges = np.percentile(freqs_in_trim_mhz, [0, 33.33, 66.66, 100]) if freqs_in_trim_mhz.size else [0, 0, 0, 0]
    for i, label in enumerate(("low_third", "mid_third", "high_third")):
        m = (freqs_in_trim_mhz >= edges[i]) & (freqs_in_trim_mhz <= edges[i + 1])
        if m.any():
            third_med[label] = {
                "n": int(m.sum()),
                "median_tau": float(np.median(taus_in_trim[m])),
                "freq_range": (float(edges[i]), float(edges[i + 1])),
            }

    # ---------------------------------------------------------------------
    # Figures
    # ---------------------------------------------------------------------
    # 2D STFT heatmap (trim region only).
    trim_idx = np.where(in_trim)[0]
    mag_trim = cal.mag[:, trim_idx]
    freqs_mol = probe_mhz - cal.freq_bb_mhz[trim_idx]
    sort = np.argsort(freqs_mol)
    fig, ax = plt.subplots(figsize=(14, 4.5))
    im = ax.imshow(
        np.log10(np.clip(mag_trim[:, sort], 1e-12, None)),
        aspect="auto", origin="lower",
        extent=[freqs_mol[sort][0], freqs_mol[sort][-1],
                cal.a_centers_us[0], cal.a_centers_us[-1]],
        cmap="viridis",
    )
    ax.set_xlabel("molecular freq (MHz)")
    ax.set_ylabel("STFT frame centre a_c (µs)")
    ax.set_title("2638 2D STFT heatmap (log10 |S(a, f)|) — trim region only")
    plt.colorbar(im, ax=ax, label="log10 |S_n|")
    fig.tight_layout()
    fig.savefig(FIG / "08_2638_stft_heatmap.png", dpi=110)
    plt.close(fig)

    # τ histogram.
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    ax = axes[0, 0]
    ax.hist(taus_in_trim, bins=80, color="C0", alpha=0.7,
            label=f"contributors (n={taus_in_trim.size})")
    ax.axvline(tau_maj_w, color="C3", ls="--", lw=2,
               label=f"τ_maj = {tau_maj_w:.2f} µs (σ_τ = {sigma_tau_w:.2f})")
    ax.set_xlabel("recovered τ_k (µs)")
    ax.set_ylabel("count")
    ax.set_title("2638 per-bin τ histogram")
    ax.legend()
    ax.grid(alpha=0.3)

    # τ vs SNR.
    ax = axes[0, 1]
    ax.scatter(snrs_in_trim, taus_in_trim, s=2, alpha=0.4)
    ax.set_xscale("log")
    ax.axhline(tau_maj_w, color="C3", ls="--", lw=1)
    ax.set_xlabel("contributor on-line SNR (per-frame)")
    ax.set_ylabel("τ_k (µs)")
    ax.set_title(f"τ vs SNR (Pearson r = {r_snr:.3f})")
    ax.grid(alpha=0.3)

    # τ vs molecular freq.
    ax = axes[1, 0]
    ax.scatter(freqs_in_trim_mhz, taus_in_trim, s=2, alpha=0.4)
    ax.axhline(tau_maj_w, color="C3", ls="--", lw=1)
    ax.set_xlabel("molecular freq (MHz)")
    ax.set_ylabel("τ_k (µs)")
    ax.set_title(f"τ vs frequency (Pearson r = {r_freq:.3f})")
    ax.grid(alpha=0.3)

    # GMM overlay.
    ax = axes[1, 1]
    ax.hist(taus_in_trim, bins=80, color="C0", alpha=0.5, density=True)
    if not np.isnan(bm.get("mu_a", float("nan"))):
        xs = np.linspace(0, tau_maj_w * 3, 400)
        ya = bm["pi_a"] / np.sqrt(2 * np.pi * bm["sigma_a"] ** 2) \
             * np.exp(-0.5 * ((xs - bm["mu_a"]) / bm["sigma_a"]) ** 2)
        yb = (1 - bm["pi_a"]) / np.sqrt(2 * np.pi * bm["sigma_b"] ** 2) \
             * np.exp(-0.5 * ((xs - bm["mu_b"]) / bm["sigma_b"]) ** 2)
        ax.plot(xs, ya, "C3", lw=1, label=f"GMM μ_a={bm['mu_a']:.2f}, π_a={bm['pi_a']:.2f}")
        ax.plot(xs, yb, "C2", lw=1, label=f"GMM μ_b={bm['mu_b']:.2f}")
        ax.plot(xs, ya + yb, "k", lw=1, alpha=0.6)
    ax.set_xlabel("τ (µs)")
    ax.set_ylabel("density")
    ax.set_title(
        f"GMM 1 vs 2 component (ΔAIC = {bm.get('delta_aic', float('nan')):.1f}, "
        f"bimodal={bm.get('two_component_preferred', False)})"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG / "09_2638_distribution_analysis.png", dpi=110)
    plt.close(fig)

    # Save artefacts.
    np.savez(
        DATA / "phase2_2638.npz",
        taus=taus_in_trim, snrs=snrs_in_trim, freqs_mhz=freqs_in_trim_mhz,
        spur_bins=spur_in_trim, spur_freqs_mhz=spur_freqs_mol_mhz,
        a_centers_us=cal.a_centers_us, sigma_frame_scalar=cal.sigma_frame[0],
    )

    # Verdict against Phase 2 acceptance.
    # tau_obs ≈ 3 µs + tau_apod = 5 µs ⇒ tau_mol ≈ 7.5 µs.
    # Acceptance: τ_maj within ±20 % of [6.0, 9.0] µs i.e. [4.8, 10.8].
    implied_low, implied_high = 6.0, 9.0
    accept_low, accept_high = implied_low * 0.80, implied_high * 1.20
    in_range = accept_low <= tau_maj_w <= accept_high
    out = {
        "n_active_samples": int(active.size),
        "T_full_us": T_full_us,
        "sample_dt_ns": sample_dt_us * 1e3,
        "n_seg": n_seg,
        "n_contributors": int(taus_in_trim.size),
        "n_spurs_in_trim": int(spur_in_trim.size),
        "n_spurs_total": int(cal.spur_bins.size),
        "tau_maj_us": float(tau_maj_w),
        "sigma_tau_us": float(sigma_tau_w),
        "pearson_r_log_snr_vs_tau": r_snr,
        "pearson_r_freq_vs_tau": r_freq,
        "frequency_thirds": third_med,
        "gmm": bm,
        "implied_tau_range_us": (implied_low, implied_high),
        "phase2_acceptance_range_us": (accept_low, accept_high),
        "phase2_acceptance_pass": in_range,
        "spur_freqs_mhz_sample": [
            float(f) for f in np.sort(spur_freqs_mol_mhz)[:20]
        ],
    }
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(run_case2_only: bool = False, run_phase2: bool = True) -> dict:
    t0 = time.perf_counter()
    out = {}
    out["case1"] = case1_tau_sweep()
    logger.info("case1 elapsed: %.1fs", time.perf_counter() - t0)
    out["case2"] = case2_snr_sweep()
    logger.info("case2 elapsed: %.1fs", time.perf_counter() - t0)
    out["case3"] = case3_isolated_spur()
    out["case4"] = case4_spur_near_line()
    out["case5"] = case5_dense_cluster()
    out["case6"] = case6_bimodal_population()
    out["case7"] = case7_voigt_deficit()
    out["pathological"] = pathological_corners()
    elapsed = time.perf_counter() - t0
    out["elapsed_sec"] = elapsed
    logger.info("Phase 1 total elapsed: %.1f sec", elapsed)
    with open(DATA / "phase1_summary.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    if run_phase2:
        out["phase2"] = phase2_2638()
        with open(DATA / "phase2_summary.json", "w") as f:
            json.dump(out["phase2"], f, indent=2,
                      default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    return out


if __name__ == "__main__":
    main()
