"""Data-driven tau calibration via sliding-active-window STFT.

Produces a global majority-vote molecular decay constant ``tau_maj`` (with
robust spread ``sigma_tau``) from the raw FID, without running any LSQ fit.
The calibration extracts tau by zero-padding the full record and sliding a
``T_w = T_full / n_seg``-long active sub-window across it, then reading the
magnitude at every frequency bin across the resulting STFT frames. A real
molecular line at ``f_0`` decays as ``exp(-a/tau_mol)`` vs the window start
``a``; a CW clock spur stays constant; noise bins fail the goodness-of-fit
gate. Per-bin exponential fits give a tau histogram of thousands of bins,
robust to single-window pathologies (blends, shape error, fixed-contributor
coupling) that complicate the LSQ-fit-and-histogram alternative.

The math for a single damped cosine
``s(t) = A * exp(-t/tau_mol) * cos(2*pi*f_0*t + phi)`` extracted on
``[a, a + T_w]`` and zero-padded to the full record before rfft:

    |S(a, f_0)| = (A * tau_mol / 2) * exp(-a/tau_mol) * (1 - exp(-T_w/tau_mol))
                = constant * exp(-a/tau_mol)

so sliding ``a`` traces a pure exponential decay whose rate is ``1/tau_mol``
directly -- single-parameter fit, no LSQ ambiguity.

The operating points (``n_seg = 10``, ``T_sigma = 5``, SNR-weighted majority,
hybrid bad-fit gate, GMM threshold ``delta_aicc > 2``) match the Phase 1
research prototype in ``dev-docs/research/stage5-tau-calibration/prototype.py``.
See ``dev-docs/research/stage5-tau-calibration/report.md`` for the synthetic
acceptance gate and ``report-2638.md`` for the 2638 application.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Operating points (Phase 1 acceptance gate; see report.md).
DEFAULT_N_SEG = 10
DEFAULT_T_SIGMA = 5.0
DEFAULT_TAU_MAX_FACTOR = 5.0  # tau_max = 5 * T_full
DEFAULT_RSS_GATE_FACTOR = 5.0
DEFAULT_RELATIVE_GATE_FRACTION = 0.05
DEFAULT_GMM_DELTA_AICC = 2.0
DEFAULT_BIMODALITY_DOMINANT_FRACTION = 0.70
DEFAULT_MIN_CONTRIBUTORS = 200
DEFAULT_SIGMA_TAU_FRACTION_MAX = 0.20
DEFAULT_SIGMA_TAU_FLOOR_US = 0.5
DEFAULT_SPUR_CLUSTER_MULTIPLIER = 1.0  # cluster gap in units of n_seg full-record bins


__all__ = [
    "TauCalibrationResult",
    "SpurCluster",
    "extract_tau_majority",
    "sliding_stft",
    "stft_calibration",
    "majority_tau",
    "gmm_bimodality",
    "group_spur_bins",
    "DEFAULT_N_SEG",
    "DEFAULT_T_SIGMA",
    "DEFAULT_TAU_MAX_FACTOR",
    "DEFAULT_RSS_GATE_FACTOR",
    "DEFAULT_GMM_DELTA_AICC",
    "DEFAULT_BIMODALITY_DOMINANT_FRACTION",
    "DEFAULT_MIN_CONTRIBUTORS",
    "DEFAULT_SIGMA_TAU_FRACTION_MAX",
    "DEFAULT_SIGMA_TAU_FLOOR_US",
]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SpurCluster:
    """One CW spur after collapsing adjacent spur-classified bins.

    A single clock spur produces approximately ``n_seg`` adjacent spur-bins
    (the spur's STFT-rectangular-window sinc-skirt cluster shares its parent's
    constant time-dependence and so is also classified as a spur). Grouping
    those satellites into one entry keeps the persisted spur catalogue
    human-auditable.

    Attributes
    ----------
    center_freq_mhz : float
        Molecular frequency of the cluster's peak-magnitude bin (MHz).
    peak_bin_index : int
        Index of the cluster's peak-magnitude bin in the per-bin arrays.
    n_bins : int
        Number of adjacent spur-classified bins in the cluster.
    bin_indices : tuple of int
        Indices of every spur-classified bin in this cluster (sorted).
    """

    center_freq_mhz: float
    peak_bin_index: int
    n_bins: int
    bin_indices: Tuple[int, ...]


@dataclass(frozen=True)
class GMMBimodality:
    """1- vs 2-component Gaussian-mixture preference on the contributor tau histogram.

    Attributes
    ----------
    n : int
        Contributor count fed into the GMM.
    mu1, sigma1 : float
        1-component MLE parameters.
    mu_a, sigma_a, mu_b, sigma_b : float
        Two-component EM parameters (``mu_a`` is the lower-tau component).
    pi_a : float
        Mixture weight on the lower-tau component (``0..1``).
    aic1, aic2 : float
        Akaike information criteria for the two models.
    delta_aic : float
        ``aic1 - aic2``; positive means 2-component preferred.
    two_component_preferred : bool
        ``delta_aic > DEFAULT_GMM_DELTA_AICC``.
    dominant_weight : float
        ``max(pi_a, 1 - pi_a)`` -- weight of the majority cluster.
    """

    n: int
    mu1: float
    sigma1: float
    mu_a: float
    sigma_a: float
    mu_b: float
    sigma_b: float
    pi_a: float
    aic1: float
    aic2: float
    delta_aic: float
    two_component_preferred: bool
    dominant_weight: float


@dataclass(frozen=True)
class FrequencyThird:
    """Median tau in one band-third (low / mid / high) of the contributor range."""

    label: str
    freq_lo_mhz: float
    freq_hi_mhz: float
    n: int
    median_tau_us: float


@dataclass(frozen=True)
class TauCalibrationResult:
    """Outcome of one STFT tau-calibration pass.

    Attributes
    ----------
    tau_maj_us : float
        SNR-weighted-majority molecular decay constant (microseconds).
    sigma_tau_us : float
        Robust spread (weighted IQR / 1.349) of the contributor tau histogram.
    n_contributors : int
        Number of "contributor" (= above-threshold, not-spur, not-bad-fit)
        bins inside the analysis frequency range.
    n_spur_bins : int
        Total number of spur-classified bins (pre-clustering).
    spur_clusters : tuple of SpurCluster
        Grouped CW spur catalogue (adjacent spur-bins collapsed).
    bimodality : GMMBimodality
        1- vs 2-component GMM diagnostic on the contributor histogram.
    pearson_r_log_snr_vs_tau : float
        Pearson correlation of ``log10(SNR)`` and tau on the contributor set
        (NaN if fewer than 5 contributors).
    pearson_r_freq_vs_tau : float
        Pearson correlation of molecular frequency and tau on the contributor
        set (NaN if fewer than 5 contributors).
    frequency_thirds : tuple of FrequencyThird
        Per-band-third median tau (low / mid / high).
    contributor_bin_indices : np.ndarray
        Indices of contributor bins in the per-bin arrays (sorted by frequency).
    contributor_taus_us : np.ndarray
        Recovered tau per contributor bin.
    contributor_snrs : np.ndarray
        On-line SNR (= max-frame magnitude / per-frame sigma) per contributor bin.
    contributor_freqs_mhz : np.ndarray
        Molecular frequency per contributor bin.
    n_seg : int
        Number of non-overlapping STFT frames used.
    t_sigma : float
        Above-threshold gate factor on per-frame noise (``max |S_n| >= t_sigma * sigma_frame``).
    tau_max_us : float
        Upper clip on recovered tau (saturation -> spur candidate).
    rss_gate_factor : float
        Bad-fit gate strength (relative-or-absolute hybrid).
    sample_dt_us : float
        FID sample spacing the calibration used (microseconds).
    start_us : float
        FID active-region start time (microseconds).
    end_us : float
        FID active-region end time (microseconds).
    probe_freq_mhz : float
        Probe frequency the calibration used to convert baseband bins to
        molecular frequencies (MHz).
    sideband : str
        ``"lower"`` or ``"upper"``.
    trim_lo_mhz, trim_hi_mhz : float
        Analysis frequency range (molecular, MHz). Bins outside are not
        considered for tau extraction.
    sigma_x_full : float
        |X|-RMS noise floor on the full-record FT (per-bin, scalar).
    sigma_frame : float
        Per-frame noise floor (= ``sigma_x_full / sqrt(n_seg)``).
    snr_weighted : bool
        Whether the majority tau is SNR-weighted (always True in the
        production extractor; kept for forensic clarity).
    preconditions_passed : bool
        Whether the calibration satisfies the acceptance pre-conditions
        (>= 200 contributors AND (not strongly bimodal OR dominant >= 0.70)
        AND sigma_tau / tau_maj < 0.20). When False, downstream consumers
        should fall back to a conservative default.
    preconditions_notes : tuple of str
        Per-precondition diagnostic message ("ok" or the failing reason).
    """

    tau_maj_us: float
    sigma_tau_us: float
    n_contributors: int
    n_spur_bins: int
    spur_clusters: Tuple[SpurCluster, ...]
    bimodality: GMMBimodality
    pearson_r_log_snr_vs_tau: float
    pearson_r_freq_vs_tau: float
    frequency_thirds: Tuple[FrequencyThird, ...]
    contributor_bin_indices: np.ndarray
    contributor_taus_us: np.ndarray
    contributor_snrs: np.ndarray
    contributor_freqs_mhz: np.ndarray
    n_seg: int
    t_sigma: float
    tau_max_us: float
    rss_gate_factor: float
    sample_dt_us: float
    start_us: float
    end_us: float
    probe_freq_mhz: float
    sideband: str
    trim_lo_mhz: float
    trim_hi_mhz: float
    sigma_x_full: float
    sigma_frame: float
    snr_weighted: bool
    preconditions_passed: bool
    preconditions_notes: Tuple[str, ...]


# ---------------------------------------------------------------------------
# Per-bin STFT core (pure NumPy)
# ---------------------------------------------------------------------------
def sliding_stft(
    fid: np.ndarray,
    sample_dt_us: float,
    n_seg: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding-active-window STFT (zero-pad outside the active sub-window).

    Returns
    -------
    mag : np.ndarray
        ``(n_seg, n_bins)`` magnitude spectrogram. Each row is the rfft of a
        full-record-length signal in which only the active sub-interval
        ``[k * T_w, (k + 1) * T_w)`` is non-zero. Zero-padding preserves the
        full-record bin spacing ``Delta f = 1 / T_full`` so all frames share
        the same frequency grid and per-bin time series are 1:1 comparable.
    a_centers_us : np.ndarray
        ``(n_seg,)`` frame midpoint times (microseconds).
    freq_bb_mhz : np.ndarray
        ``(n_bins,)`` baseband frequencies (MHz; assumes ``sample_dt_us`` in
        microseconds).
    """
    fid_arr = np.asarray(fid, dtype=float)
    N = fid_arr.size
    Nw = N // n_seg
    if Nw < 4:
        raise ValueError(
            f"n_seg={n_seg} too large for fid length N={N} (Nw={Nw} < 4)"
        )
    n_bins = N // 2 + 1
    mag = np.empty((n_seg, n_bins), dtype=float)
    a_centers = np.empty(n_seg, dtype=float)
    padded = np.zeros(N, dtype=float)
    for k in range(n_seg):
        a_start = k * Nw
        a_end = a_start + Nw
        padded[:] = 0.0
        padded[a_start:a_end] = fid_arr[a_start:a_end]
        spec = sample_dt_us * np.fft.rfft(padded)
        mag[k] = np.abs(spec)
        a_centers[k] = (a_start + (Nw - 1) * 0.5) * sample_dt_us
    freq_bb_mhz = np.fft.rfftfreq(N, d=sample_dt_us)
    return mag, a_centers, freq_bb_mhz


def _fit_exp_per_bin(
    mag: np.ndarray,
    a_centers_us: np.ndarray,
    *,
    tau_clip_us: Tuple[float, float] = (0.1, 1e4),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted log-linear fit ``log |S_n| = log C - a / tau`` per bin.

    Weights are ``|S_n|^2`` (variance of ``log |S_n|`` under Gaussian noise on
    ``|S_n|`` is ``~ sigma^2 / |S_n|^2``). The +3-5% systematic bias at
    intermediate ``T_full / tau`` ratios documented in
    ``report.md`` § Case 1 traces to this weighting choice; the per-bin
    estimate biases consistently and the SNR-weighted-majority over many
    contributors absorbs the systematic. Returns
    ``(tau, C, rss_lin)``; the RSS is computed in linear ``|S_n|`` space so
    it is comparable to the constant-model RSS.
    """
    n_seg, n_bins = mag.shape
    safe = np.clip(mag, 1e-300, None)
    log_m = np.log(safe)
    w = mag * mag
    a = a_centers_us[:, None]
    Sw = w.sum(axis=0)
    Swa = (w * a).sum(axis=0)
    Swl = (w * log_m).sum(axis=0)
    Swaa = (w * a * a).sum(axis=0)
    Swal = (w * a * log_m).sum(axis=0)
    denom = Sw * Swaa - Swa * Swa
    good = denom > 1e-30
    slope = np.where(good, (Sw * Swal - Swa * Swl) / np.where(good, denom, 1.0), 0.0)
    intercept = np.where(
        Sw > 0, (Swl - slope * Swa) / np.where(Sw > 0, Sw, 1.0), 0.0
    )
    # Slope >= 0 means no exponential decay (constant or growing) -> assign
    # tau_max so the bin is later classified as a spur candidate.
    tau = np.where(slope < 0, -1.0 / np.where(slope < 0, slope, -1.0), tau_clip_us[1])
    tau = np.clip(tau, tau_clip_us[0], tau_clip_us[1])
    C = np.exp(np.clip(intercept, -50.0, 50.0))
    pred = C[None, :] * np.exp(-a / tau[None, :])
    rss = ((mag - pred) ** 2).sum(axis=0)
    return tau, C, rss


def _aicc(rss: np.ndarray, n: int, k: int) -> np.ndarray:
    """Small-sample-corrected AIC under Gaussian residuals.

    ``AICc = n * log(RSS / n) + 2 k + 2 k (k + 1) / (n - k - 1)``. Returns
    ``+inf`` for ``n - k - 1 <= 0`` (model not identifiable at this sample
    size).
    """
    if n - k - 1 <= 0:
        return np.full_like(rss, np.inf, dtype=float)
    correction = 2.0 * k * (k + 1) / (n - k - 1)
    rss_safe = np.where(rss > 0, rss, 1e-300)
    return n * np.log(rss_safe / n) + 2 * k + correction


@dataclass(frozen=True)
class _STFTClassification:
    """Internal: per-bin classification from one STFT calibration pass."""

    mag: np.ndarray
    a_centers_us: np.ndarray
    freq_bb_mhz: np.ndarray
    tau_per_bin: np.ndarray
    rss_exp: np.ndarray
    rss_const: np.ndarray
    aicc_exp: np.ndarray
    aicc_const: np.ndarray
    classification: np.ndarray  # 0=discard, 1=spur, 2=bad-fit, 3=contributor
    snr_per_bin: np.ndarray
    sigma_x_full: float
    sigma_frame: float
    tau_max_us: float


def stft_calibration(
    fid: np.ndarray,
    sample_dt_us: float,
    sigma_time: float,
    *,
    n_seg: int = DEFAULT_N_SEG,
    t_sigma: float = DEFAULT_T_SIGMA,
    tau_max_us: Optional[float] = None,
    rss_gate_factor: float = DEFAULT_RSS_GATE_FACTOR,
    relative_gate_fraction: float = DEFAULT_RELATIVE_GATE_FRACTION,
) -> _STFTClassification:
    """Run the sliding-active-window STFT and classify every frequency bin.

    Per-bin classification:

    * **0 / discard**: ``max_n |S_n(k)| < t_sigma * sigma_frame(k)``. Most
      bins land here.
    * **1 / spur**: AICc prefers the constant model OR the exponential fit
      saturates at ``0.95 * tau_max_us`` (CW tone, ``tau -> infinity``).
    * **2 / bad-fit**: the exponential RSS exceeds
      ``rss_gate_factor * n_seg * max(sigma_frame^2, (relative_gate_fraction *
      mean(|S_n|))^2)`` (overlapping skirts, mid-blend bins).
    * **3 / contributor**: above threshold, not a spur, not a bad-fit. These
      bins enter the tau histogram.

    Parameters
    ----------
    fid : np.ndarray
        Raw FID samples (length ``N``).
    sample_dt_us : float
        Sample spacing (microseconds).
    sigma_time : float
        Time-domain white-noise RMS of the FID. Used to derive the
        analytic per-bin |X|-RMS (``sigma_x_full = sigma_t * dt * sqrt(N/2)``)
        and the per-frame floor (``sigma_frame = sigma_x_full / sqrt(n_seg)``).
        Must be positive.
    n_seg : int, default :data:`DEFAULT_N_SEG`
        Number of non-overlapping frames; ``T_w = T_full / n_seg``.
    t_sigma : float, default :data:`DEFAULT_T_SIGMA`
        Above-threshold gate factor (the contributor floor on per-frame SNR).
    tau_max_us : float, optional
        Upper clip on recovered tau; saturation = spur candidate. Defaults to
        ``DEFAULT_TAU_MAX_FACTOR * T_full``.
    rss_gate_factor : float, default :data:`DEFAULT_RSS_GATE_FACTOR`
        Bad-fit gate strength (relative-or-absolute hybrid).
    relative_gate_fraction : float, default :data:`DEFAULT_RELATIVE_GATE_FRACTION`
        Relative branch of the bad-fit gate (``rss > rss_gate_factor * n_seg *
        (relative_gate_fraction * mean(|S_n|))^2``). The relative branch is
        necessary for high-SNR clean fits not to over-classify as bad-fit; the
        log-linear weighted regression does not minimise linear-space RSS so
        its prediction error scales with the signal level, not the noise level.
    """
    if sigma_time <= 0.0:
        raise ValueError("sigma_time must be positive")
    fid_arr = np.asarray(fid, dtype=float)
    N = fid_arr.size
    T_full_us = N * sample_dt_us
    if tau_max_us is None:
        tau_max_us = DEFAULT_TAU_MAX_FACTOR * T_full_us

    mag, a_centers_us, freq_bb_mhz = sliding_stft(fid_arr, sample_dt_us, n_seg)

    # Analytic per-bin noise estimate on the full-record FT, then per-frame.
    sigma_x_full = sigma_time * sample_dt_us * np.sqrt(N / 2.0)
    sigma_frame = sigma_x_full / np.sqrt(n_seg)

    tau, _C, rss_exp = _fit_exp_per_bin(
        mag, a_centers_us, tau_clip_us=(0.1, tau_max_us)
    )
    mean_m = mag.mean(axis=0)
    rss_const = ((mag - mean_m[None, :]) ** 2).sum(axis=0)
    aicc_exp = _aicc(rss_exp, n_seg, k=2)
    aicc_const = _aicc(rss_const, n_seg, k=1)

    max_m = mag.max(axis=0)
    snr_per_bin = max_m / sigma_frame
    above = snr_per_bin >= t_sigma

    spur_by_aicc = aicc_const + 2.0 < aicc_exp
    spur_by_tau = tau >= 0.95 * tau_max_us
    is_spur = above & (spur_by_aicc | spur_by_tau)

    rss_gate_abs = rss_gate_factor * n_seg * (sigma_frame ** 2)
    rss_gate_rel = rss_gate_factor * n_seg * (relative_gate_fraction * mean_m) ** 2
    rss_gate = np.maximum(rss_gate_abs, rss_gate_rel)
    bad_fit = above & ~is_spur & (rss_exp > rss_gate)

    contributor = above & ~is_spur & ~bad_fit
    classification = np.where(
        contributor, 3,
        np.where(bad_fit, 2, np.where(is_spur, 1, 0))
    ).astype(np.int8)

    return _STFTClassification(
        mag=mag,
        a_centers_us=a_centers_us,
        freq_bb_mhz=freq_bb_mhz,
        tau_per_bin=tau,
        rss_exp=rss_exp,
        rss_const=rss_const,
        aicc_exp=aicc_exp,
        aicc_const=aicc_const,
        classification=classification,
        snr_per_bin=snr_per_bin,
        sigma_x_full=float(sigma_x_full),
        sigma_frame=float(sigma_frame),
        tau_max_us=float(tau_max_us),
    )


# ---------------------------------------------------------------------------
# Aggregation: majority tau, GMM bimodality, spur clustering
# ---------------------------------------------------------------------------
def majority_tau(
    contributor_taus_us: np.ndarray,
    contributor_snrs: Optional[np.ndarray] = None,
    *,
    weighted: bool = True,
) -> Tuple[float, float]:
    """Robust majority tau and spread sigma_tau from the contributor histogram.

    SNR-weighted by default. The Phase-1 case-1 result showed that strong-line
    skirt bins cluster tightly around the truth while near-threshold bins are
    biased high by noisy log-linear fits; the SNR-weighted median collapses
    onto the on-line bins and matches the truth. Returns ``(tau_maj_us,
    sigma_tau_us)`` with ``sigma_tau_us = (weighted) IQR / 1.349``.
    """
    taus = np.asarray(contributor_taus_us, dtype=float)
    if taus.size == 0:
        return float("nan"), float("nan")
    if weighted and contributor_snrs is not None:
        snrs = np.asarray(contributor_snrs, dtype=float)
        if snrs.sum() > 0:
            order = np.argsort(taus)
            t = taus[order]
            w = snrs[order]
            cdf = np.cumsum(w) / w.sum()
            tau_maj = float(np.interp(0.5, cdf, t))
            q25 = float(np.interp(0.25, cdf, t))
            q75 = float(np.interp(0.75, cdf, t))
            return tau_maj, (q75 - q25) / 1.349
    tau_maj = float(np.median(taus))
    q25, q75 = np.percentile(taus, [25, 75])
    return tau_maj, float((q75 - q25) / 1.349)


def gmm_bimodality(
    contributor_taus_us: np.ndarray, *, max_iter: int = 200,
) -> GMMBimodality:
    """Fit 1- and 2-component Gaussian mixtures and report the AIC preference.

    Hand-rolled EM (no sklearn dependency, matching the research prototype).
    Returns the per-component MLE parameters plus
    ``delta_aic = aic1 - aic2``; positive means the 2-component model is
    preferred. The Phase-2 default of ``delta_aic > 2`` is slightly looser
    than the planning doc's original ``> 4`` because the prototype's
    realistic-bimodal case (case 6, ``ΔAIC = +53``) clears either threshold
    comfortably while real-world contributor counts of ~50-150 sometimes
    sit in the gap.

    Returns a degenerate result (NaN parameters, ``delta_aic = NaN``,
    ``two_component_preferred = False``) when fewer than 20 contributors are
    available.
    """
    x = np.asarray(contributor_taus_us, dtype=float)
    n = int(x.size)
    if n < 20:
        return GMMBimodality(
            n=n,
            mu1=float("nan"), sigma1=float("nan"),
            mu_a=float("nan"), sigma_a=float("nan"),
            mu_b=float("nan"), sigma_b=float("nan"),
            pi_a=float("nan"),
            aic1=float("nan"), aic2=float("nan"),
            delta_aic=float("nan"),
            two_component_preferred=False,
            dominant_weight=float("nan"),
        )

    mu1 = float(np.mean(x))
    var1 = max(float(np.var(x)), 1e-12)
    ll1 = -0.5 * n * (np.log(2.0 * np.pi * var1) + 1.0)
    aic1 = 2 * 2 - 2 * ll1  # 2 params

    med = float(np.median(x))
    lo_mask = x < med
    hi_mask = ~lo_mask
    mu_a = float(np.mean(x[lo_mask])) if lo_mask.any() else med
    mu_b = float(np.mean(x[hi_mask])) if hi_mask.any() else med
    var_a = max(float(np.var(x[lo_mask])) if lo_mask.any() else var1, 1e-12)
    var_b = max(float(np.var(x[hi_mask])) if hi_mask.any() else var1, 1e-12)
    pi_a = 0.5
    eps = 1e-12

    for _ in range(max_iter):
        ga = pi_a / np.sqrt(2.0 * np.pi * var_a) * np.exp(-0.5 * (x - mu_a) ** 2 / var_a)
        gb = (1.0 - pi_a) / np.sqrt(2.0 * np.pi * var_b) * np.exp(-0.5 * (x - mu_b) ** 2 / var_b)
        denom = ga + gb + eps
        wa = ga / denom
        wb = gb / denom
        Na = wa.sum() + eps
        Nb = wb.sum() + eps
        mu_a_new = float((wa * x).sum() / Na)
        mu_b_new = float((wb * x).sum() / Nb)
        var_a_new = max(float((wa * (x - mu_a_new) ** 2).sum() / Na), 1e-12)
        var_b_new = max(float((wb * (x - mu_b_new) ** 2).sum() / Nb), 1e-12)
        pi_a_new = float(Na / n)
        if (abs(mu_a_new - mu_a) + abs(mu_b_new - mu_b)) < 1e-9:
            mu_a, mu_b, var_a, var_b, pi_a = (
                mu_a_new, mu_b_new, var_a_new, var_b_new, pi_a_new,
            )
            break
        mu_a, mu_b, var_a, var_b, pi_a = (
            mu_a_new, mu_b_new, var_a_new, var_b_new, pi_a_new,
        )

    g_a = pi_a / np.sqrt(2.0 * np.pi * var_a) * np.exp(-0.5 * (x - mu_a) ** 2 / var_a)
    g_b = (1.0 - pi_a) / np.sqrt(2.0 * np.pi * var_b) * np.exp(-0.5 * (x - mu_b) ** 2 / var_b)
    ll2 = float(np.sum(np.log(g_a + g_b + eps)))
    aic2 = 2 * 5 - 2 * ll2
    delta_aic = aic1 - aic2

    # Enforce mu_a <= mu_b for output stability (the EM initialisation already
    # primes this but small-n mixtures can swap).
    if mu_a > mu_b:
        mu_a, mu_b = mu_b, mu_a
        var_a, var_b = var_b, var_a
        pi_a = 1.0 - pi_a

    return GMMBimodality(
        n=n,
        mu1=mu1, sigma1=float(np.sqrt(var1)),
        mu_a=float(mu_a), sigma_a=float(np.sqrt(var_a)),
        mu_b=float(mu_b), sigma_b=float(np.sqrt(var_b)),
        pi_a=float(pi_a),
        aic1=float(aic1), aic2=float(aic2),
        delta_aic=float(delta_aic),
        two_component_preferred=bool(delta_aic > DEFAULT_GMM_DELTA_AICC),
        dominant_weight=float(max(pi_a, 1.0 - pi_a)),
    )


def group_spur_bins(
    spur_bin_indices: np.ndarray,
    mean_mag: np.ndarray,
    freqs_mhz: np.ndarray,
    *,
    n_seg: int,
    cluster_multiplier: float = DEFAULT_SPUR_CLUSTER_MULTIPLIER,
) -> Tuple[SpurCluster, ...]:
    """Collapse adjacent spur-classified bins into one entry per CW source.

    A single CW tone produces approximately ``n_seg`` adjacent spur bins (the
    spur's STFT-rectangular-window sinc-skirt cluster). Bins whose index
    separation is at most ``cluster_multiplier * n_seg`` are grouped into a
    single :class:`SpurCluster`; the cluster's representative is the bin with
    the largest mean magnitude.

    Empirically: Phase 2 on 2638 reported 649 raw spur bins that collapse to
    ~50-100 clusters under this rule, which matches the expected count for
    that instrument's clock harmonics.
    """
    if spur_bin_indices.size == 0:
        return ()
    idx = np.sort(np.asarray(spur_bin_indices, dtype=np.int64))
    gap = max(int(round(cluster_multiplier * n_seg)), 1)
    groups: list[list[int]] = [[int(idx[0])]]
    for b in idx[1:]:
        b_int = int(b)
        if b_int - groups[-1][-1] <= gap:
            groups[-1].append(b_int)
        else:
            groups.append([b_int])
    clusters: list[SpurCluster] = []
    for g in groups:
        g_arr = np.asarray(g, dtype=np.int64)
        peak_idx = int(g_arr[int(np.argmax(mean_mag[g_arr]))])
        clusters.append(
            SpurCluster(
                center_freq_mhz=float(freqs_mhz[peak_idx]),
                peak_bin_index=peak_idx,
                n_bins=len(g),
                bin_indices=tuple(int(b) for b in g),
            )
        )
    clusters.sort(key=lambda c: c.center_freq_mhz)
    return tuple(clusters)


# ---------------------------------------------------------------------------
# Time-domain noise estimate from the FID tail
# ---------------------------------------------------------------------------
def estimate_sigma_time_from_tail(
    fid: np.ndarray, *, tail_fraction: float = 0.30,
) -> float:
    """Empirical FID-tail sigma_t for the calibration noise reference.

    The last ``tail_fraction`` of the (active-region) FID is dominated by
    noise: any line with ``tau <= acquisition / 1.5`` has decayed below
    ``exp(-1.5) ≈ 22 %`` of its peak by then, so the sample standard
    deviation of the tail is a robust direct measurement of the time-domain
    white-noise RMS. Matches the prototype's Phase-2 noise reference path.
    """
    arr = np.asarray(fid, dtype=float)
    if arr.size == 0:
        raise ValueError("fid must be non-empty")
    if not 0.0 < tail_fraction < 1.0:
        raise ValueError("tail_fraction must lie in (0, 1)")
    tail_start = max(int(arr.size * (1.0 - tail_fraction)), 0)
    tail = arr[tail_start:]
    if tail.size < 2:
        raise ValueError("fid tail too short to estimate sigma_time")
    return float(np.std(tail - tail.mean()))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def extract_tau_majority(
    fid: np.ndarray,
    sample_dt_us: float,
    *,
    start_us: float,
    end_us: float,
    probe_freq_mhz: float,
    sideband: str,
    trim_lo_mhz: float,
    trim_hi_mhz: float,
    sigma_time: Optional[float] = None,
    n_seg: int = DEFAULT_N_SEG,
    t_sigma: float = DEFAULT_T_SIGMA,
    tau_max_us: Optional[float] = None,
    rss_gate_factor: float = DEFAULT_RSS_GATE_FACTOR,
    relative_gate_fraction: float = DEFAULT_RELATIVE_GATE_FRACTION,
    spur_cluster_multiplier: float = DEFAULT_SPUR_CLUSTER_MULTIPLIER,
    min_contributors: int = DEFAULT_MIN_CONTRIBUTORS,
    sigma_tau_fraction_max: float = DEFAULT_SIGMA_TAU_FRACTION_MAX,
    bimodality_dominant_fraction: float = DEFAULT_BIMODALITY_DOMINANT_FRACTION,
) -> TauCalibrationResult:
    """End-to-end STFT tau calibration: FID -> ``TauCalibrationResult``.

    Slices the FID to ``[start_us, end_us)``, runs the sliding-active-window
    STFT, classifies bins, computes the SNR-weighted majority tau over the
    trim region, fits a 1- vs 2-component GMM, groups spur clusters, and
    evaluates the three Phase-2 pre-conditions.

    Parameters
    ----------
    fid : np.ndarray
        Raw FID samples (full record; the function slices to ``[start_us,
        end_us)`` itself).
    sample_dt_us : float
        Sample spacing in microseconds.
    start_us, end_us : float
        FID active region. The STFT runs on this slice; the noise reference
        (when ``sigma_time`` is not supplied) is measured on its tail.
    probe_freq_mhz : float
        Probe frequency (MHz) used to map baseband to molecular frequencies.
    sideband : {"lower", "upper"}
        Mixing sideband. Sets the sign of the baseband-to-molecular map.
    trim_lo_mhz, trim_hi_mhz : float
        Molecular-frequency analysis range. Contributors and spurs outside
        the range are dropped (mirrors the Stage 1 user-grid trim so the
        calibration matches the spectrum the user analyses).
    sigma_time : float, optional
        Time-domain white-noise RMS. When ``None``, estimated from the FID
        active-region tail (see :func:`estimate_sigma_time_from_tail`).
    n_seg, t_sigma, tau_max_us, rss_gate_factor, relative_gate_fraction
        STFT calibration knobs; defaults match the Phase 1 acceptance gate.
    spur_cluster_multiplier : float
        Cluster-gap multiplier in units of ``n_seg`` full-record bins.
    min_contributors, sigma_tau_fraction_max, bimodality_dominant_fraction
        Acceptance pre-conditions (Phase 2). Calibrations that fail any
        pre-condition still return a result; downstream consumers gate on
        :attr:`TauCalibrationResult.preconditions_passed`.

    Notes
    -----
    Stage 1 owns the *user-grid* trim; the calibration accepts the
    ``trim_*`` range as parameters rather than re-reading the FT settings
    so the same routine can be applied to non-pipeline FIDs (e.g. from the
    Phase-1 synthetic study) without dependency on the file format.
    """
    sb = sideband.strip().lower()
    if sb not in ("lower", "upper"):
        raise ValueError(f"sideband must be 'lower' or 'upper', got {sideband!r}")
    if sample_dt_us <= 0.0:
        raise ValueError("sample_dt_us must be positive")
    if end_us <= start_us:
        raise ValueError("end_us must be strictly greater than start_us")
    if trim_hi_mhz <= trim_lo_mhz:
        raise ValueError("trim_hi_mhz must be strictly greater than trim_lo_mhz")

    fid_arr = np.asarray(fid, dtype=float)
    start_idx = int(round(start_us / sample_dt_us))
    end_idx = int(round(end_us / sample_dt_us))
    start_idx = max(start_idx, 0)
    end_idx = min(end_idx, fid_arr.size)
    if end_idx - start_idx < 4 * n_seg:
        raise ValueError(
            f"active region [{start_us}, {end_us}) us has too few samples "
            f"({end_idx - start_idx}) for n_seg={n_seg}"
        )
    active = fid_arr[start_idx:end_idx]
    # Trim to a multiple of n_seg so frames don't drop samples.
    new_size = (active.size // n_seg) * n_seg
    active = active[:new_size]

    sigma_t = (
        float(sigma_time)
        if sigma_time is not None
        else estimate_sigma_time_from_tail(active)
    )
    if sigma_t <= 0.0:
        raise ValueError("sigma_time must be positive")

    cal = stft_calibration(
        active, sample_dt_us, sigma_t,
        n_seg=n_seg,
        t_sigma=t_sigma,
        tau_max_us=tau_max_us,
        rss_gate_factor=rss_gate_factor,
        relative_gate_fraction=relative_gate_fraction,
    )

    # Baseband -> molecular conversion: lower sideband -> f_mol = probe - f_bb,
    # upper sideband -> f_mol = probe + f_bb.
    sign = -1.0 if sb == "lower" else +1.0
    freq_mol_mhz = probe_freq_mhz + sign * cal.freq_bb_mhz

    in_trim = (freq_mol_mhz >= trim_lo_mhz) & (freq_mol_mhz <= trim_hi_mhz)
    contributor_mask = (cal.classification == 3) & in_trim
    spur_mask_full = (cal.classification == 1) & in_trim

    contributor_bins = np.where(contributor_mask)[0]
    # Sort contributors by molecular frequency (stable, helpful for serialization).
    contributor_bins = contributor_bins[np.argsort(freq_mol_mhz[contributor_bins])]
    contributor_taus = cal.tau_per_bin[contributor_bins]
    contributor_snrs = cal.snr_per_bin[contributor_bins]
    contributor_freqs = freq_mol_mhz[contributor_bins]

    tau_maj, sigma_tau = majority_tau(contributor_taus, contributor_snrs)
    bm = gmm_bimodality(contributor_taus)

    if contributor_taus.size >= 5:
        log_snr = np.log10(np.clip(contributor_snrs, 1e-6, None))
        r_log_snr = float(np.corrcoef(log_snr, contributor_taus)[0, 1])
        r_freq = float(np.corrcoef(contributor_freqs, contributor_taus)[0, 1])
    else:
        r_log_snr = float("nan")
        r_freq = float("nan")

    thirds: list[FrequencyThird] = []
    if contributor_freqs.size > 0:
        edges = np.percentile(contributor_freqs, [0.0, 33.333, 66.667, 100.0])
        for i, label in enumerate(("low", "mid", "high")):
            lo, hi = float(edges[i]), float(edges[i + 1])
            mask = (contributor_freqs >= lo) & (contributor_freqs <= hi)
            if mask.any():
                thirds.append(
                    FrequencyThird(
                        label=label,
                        freq_lo_mhz=lo,
                        freq_hi_mhz=hi,
                        n=int(mask.sum()),
                        median_tau_us=float(np.median(contributor_taus[mask])),
                    )
                )

    spur_bin_indices = np.where(spur_mask_full)[0]
    spur_clusters = group_spur_bins(
        spur_bin_indices,
        cal.mag.mean(axis=0),
        freq_mol_mhz,
        n_seg=n_seg,
        cluster_multiplier=spur_cluster_multiplier,
    )

    # Pre-condition checks: report each one independently.
    notes: list[str] = []
    cond_count = contributor_bins.size >= min_contributors
    notes.append(
        "ok"
        if cond_count
        else f"only {contributor_bins.size} contributors (< {min_contributors})"
    )
    cond_bimodal = (
        (not bm.two_component_preferred)
        or (bm.dominant_weight >= bimodality_dominant_fraction)
    )
    notes.append(
        "ok"
        if cond_bimodal
        else (
            f"strongly bimodal (delta_aic={bm.delta_aic:.1f}) and dominant cluster "
            f"weight {bm.dominant_weight:.2f} < {bimodality_dominant_fraction:.2f}"
        )
    )
    if tau_maj > 0:
        sigma_ratio = sigma_tau / tau_maj
        cond_spread = sigma_ratio < sigma_tau_fraction_max
        notes.append(
            "ok"
            if cond_spread
            else f"sigma_tau/tau_maj={sigma_ratio:.2f} >= {sigma_tau_fraction_max:.2f}"
        )
    else:
        cond_spread = False
        notes.append("tau_maj non-positive; spread test undefined")

    all_passed = bool(cond_count and cond_bimodal and cond_spread)

    logger.info(
        "STFT tau calibration: tau_maj=%.3f sigma_tau=%.3f us "
        "(n_contrib=%d, n_spur_bins=%d, n_clusters=%d, bimodal=%s, "
        "preconditions=%s)",
        tau_maj, sigma_tau, contributor_bins.size, spur_bin_indices.size,
        len(spur_clusters), bm.two_component_preferred,
        "pass" if all_passed else "fail",
    )

    return TauCalibrationResult(
        tau_maj_us=float(tau_maj),
        sigma_tau_us=float(sigma_tau),
        n_contributors=int(contributor_bins.size),
        n_spur_bins=int(spur_bin_indices.size),
        spur_clusters=spur_clusters,
        bimodality=bm,
        pearson_r_log_snr_vs_tau=r_log_snr,
        pearson_r_freq_vs_tau=r_freq,
        frequency_thirds=tuple(thirds),
        contributor_bin_indices=contributor_bins.astype(np.int64),
        contributor_taus_us=contributor_taus.astype(np.float64),
        contributor_snrs=contributor_snrs.astype(np.float64),
        contributor_freqs_mhz=contributor_freqs.astype(np.float64),
        n_seg=n_seg,
        t_sigma=float(t_sigma),
        tau_max_us=float(cal.tau_max_us),
        rss_gate_factor=float(rss_gate_factor),
        sample_dt_us=float(sample_dt_us),
        start_us=float(start_us),
        end_us=float(end_us),
        probe_freq_mhz=float(probe_freq_mhz),
        sideband=sb,
        trim_lo_mhz=float(trim_lo_mhz),
        trim_hi_mhz=float(trim_hi_mhz),
        sigma_x_full=float(cal.sigma_x_full),
        sigma_frame=float(cal.sigma_frame),
        snr_weighted=True,
        preconditions_passed=all_passed,
        preconditions_notes=tuple(notes),
    )
