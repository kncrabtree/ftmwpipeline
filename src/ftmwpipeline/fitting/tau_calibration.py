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
hybrid bad-fit gate, GMM threshold ``delta_aicc > 2``) match the research
prototype in ``dev-docs/research/stage5-tau-calibration/prototype.py``.
``dev-docs/research/stage5-tau-calibration/report.md`` consolidates the
method, synthetic validation, 2638 application, LSQ cross-validation, and
polish design (including the ``polish_snr_cap=9`` calibration).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

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
# Polish gate: skip the Gauss-Newton step on contributors whose per-bin
# SNR is at or above this cap (they keep the log-linear seed). The
# log-linear +3-5 % bias the polish targets concentrates at modest SNR;
# above the cap the seed is already near-unbiased so applying the polish
# there over-corrects per-band. On 2638 the cap range [8, 12] yields
# per-band SNR-weighted majority τ within ±5 % of the LSQ reference
# across low/mid/high arithmetic thirds; 9 lands worst-case 3.2 %. See
# `dev-docs/research/stage5-tau-calibration/polish_snr_cap_validation.py`
# and `report.md` § "Polish design" for the sweep.
DEFAULT_POLISH_SNR_CAP = 9.0

# Gaussian-twin operating points (see ``extract_tau_G_majority``). Per-bin
# pure-Gaussian-fit gates decide which bins enter the τ_G calibration:
# every condition must hold or the bin is filtered out before per-band
# aggregation. The estimator is a pure-Gaussian model
# ``|S| = C exp(-(a/τ_G)²)``, matching the Stage 5 ``shape='gaussian'``
# window fit -- a Voigt decomposition would extract τ_G as the Gaussian
# component *after* the Lorentzian decay is absorbed into a separate τ_L,
# which over-estimates the envelope's effective Gaussian τ relative to
# what the window fit recovers. ``_voigt_residuals`` and
# ``_fit_voigt_nls_multistart`` are exported alongside for the 3-way
# L/G/V shape-recommendation comparison.
DEFAULT_TAU_G_SNR_MIN = 20.0
DEFAULT_TAU_G_BOUND_LO = 0.5
DEFAULT_TAU_G_BOUND_HI = 100.0
# Minimum Δχ²ᵣ = χ²ᵣ(exp) − χ²ᵣ(gauss) required for a bin to be kept:
# bins where the data is more pure-Lorentzian than pure-Gaussian are
# filtered out (their τ_G saturates against the upper bound). With
# matched parameter counts (k=2 for both models) the threshold reduces
# to a direct "Gaussian beats exp by Δχ²ᵣ ≥ 1.0" test.
DEFAULT_TAU_G_DELTA_CHI2R_MIN = 1.0
DEFAULT_TAU_G_UPPER_FRACTION = 0.7
DEFAULT_TAU_G_SEEDS: Tuple[float, ...] = (100.0, 50.0, 20.0, 10.0, 5.0, 3.0)
# τ_G calibrations live on far fewer eligible bins than the pure-exp pool
# (2638: ~210 pure-Gauss-eligible vs ~2k pure-exp contributors); the
# acceptance floor scales accordingly so the Gaussian preconditions can
# actually pass on a typical fixture.
DEFAULT_TAU_G_MIN_CONTRIBUTORS = 50


__all__ = [
    "TauCalibrationResult",
    "SpurCluster",
    "BandMajority",
    "ShapeRecommendation",
    "extract_tau_majority",
    "extract_tau_G_majority",
    "compute_shape_recommendation",
    "sliding_stft",
    "stft_calibration",
    "majority_tau",
    "gmm_bimodality",
    "group_spur_bins",
    "compute_band_majorities",
    "band_majority_for_frequency",
    "DEFAULT_BAND_LABELS",
    "DEFAULT_N_SEG",
    "DEFAULT_T_SIGMA",
    "DEFAULT_TAU_MAX_FACTOR",
    "DEFAULT_RSS_GATE_FACTOR",
    "DEFAULT_GMM_DELTA_AICC",
    "DEFAULT_BIMODALITY_DOMINANT_FRACTION",
    "DEFAULT_MIN_CONTRIBUTORS",
    "DEFAULT_SIGMA_TAU_FRACTION_MAX",
    "DEFAULT_SIGMA_TAU_FLOOR_US",
    "DEFAULT_POLISH_SNR_CAP",
    "DEFAULT_TAU_G_SNR_MIN",
    "DEFAULT_TAU_G_BOUND_LO",
    "DEFAULT_TAU_G_BOUND_HI",
    "DEFAULT_TAU_G_DELTA_CHI2R_MIN",
    "DEFAULT_TAU_G_UPPER_FRACTION",
    "DEFAULT_TAU_G_SEEDS",
    "DEFAULT_TAU_G_MIN_CONTRIBUTORS",
    "DEFAULT_SHAPE_RECOMMENDATION_PURE_MARGIN",
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
class BandMajority:
    """SNR-weighted majority tau over one frequency band.

    Stage 5 consumes this as a per-window override for ``(tau_maj_us,
    sigma_tau_us)`` when ``Pipeline.fit_peaks(..., per_band_tau=True)`` is
    set. The band is identified by ``[freq_lo_mhz, freq_hi_mhz)`` and
    holds the band-local majority + spread, computed by re-running
    :func:`majority_tau` on the contributor subset whose frequency falls
    inside the band. ``n`` is the contributor count in the band; for
    small-N bands the spread may be wider than the band-wide ``sigma_tau``.

    Wide bands (entire trim range = one band) reduce to the band-wide
    ``(tau_maj_us, sigma_tau_us)``; the struct is intentionally identical
    in shape so callers can fall back transparently.
    """

    label: str
    freq_lo_mhz: float
    freq_hi_mhz: float
    n: int
    tau_maj_us: float
    sigma_tau_us: float


# Operating points for the 3-way L/G/V per-bin shape-recommendation test.
# The verdict aggregator picks the dominant *pure* shape (exp ⇒
# Lorentzian, gauss ⇒ Gaussian) when one beats the other by at least
# ``DEFAULT_SHAPE_RECOMMENDATION_PURE_MARGIN`` of the SNR-weighted total
# vote mass; Voigt votes are tabulated but never become the recommendation
# (the production Stage 5 line-shape selector supports L and G only, with
# Voigt reserved for future per-window unification). If neither pure
# shape clears the margin, the recommendation is ``None`` and the Stage 5
# resolver's *recommended* layer falls through to the next layer.
DEFAULT_SHAPE_RECOMMENDATION_PURE_MARGIN = 0.10


@dataclass(frozen=True)
class ShapeRecommendation:
    """3-way per-bin model preference verdict for the Stage 5 shape selector.

    Attributes
    ----------
    recommended_shape : str or None
        ``"lorentzian"``, ``"gaussian"``, or ``None`` for "no clear
        winner". Written to ``stage2b_tau_calibration/.attrs/
        recommended_shape`` (and the Gaussian-twin's equivalent) so the
        Stage 5 resolver's *recommended* layer picks it up automatically.
    vote_rates : dict[str, float]
        SNR-weighted vote rate per model. Keys: ``"exp"``, ``"gauss"``,
        ``"voigt"``. Values sum to 1.0 over eligible contributors. The
        per-bin verdict is ``argmin AICc(exp, gauss, voigt)``; the SNR
        weighting collapses many low-SNR ambiguous bins onto the few
        strong on-line bins that actually constrain the shape.
    median_d_aicc : dict[str, float]
        Per-row-pooled median ΔAICc. Keys:

        * ``"gauss_vs_exp"``: ``AICc(gauss) − AICc(exp)``; negative ⇒
          Gaussian beats exp.
        * ``"voigt_vs_exp"``: ``AICc(voigt) − AICc(exp)``; negative ⇒
          Voigt beats exp.
        * ``"voigt_vs_gauss"``: ``AICc(voigt) − AICc(gauss)``;
          negative ⇒ Voigt beats Gaussian.

        Negative AICc differences below about −2 are conventionally
        called "strong preference" for the model on the left of the
        difference.
    n_contributors : int
        Number of bins that entered the verdict (cls=3 ∧ SNR > snr_min,
        with all three fits converged and finite τ inside the bound).
    notes : tuple of str
        Diagnostic strings, one per acceptance / tiebreaker step.
    """

    recommended_shape: Optional[str]
    vote_rates: Dict[str, float]
    median_d_aicc: Dict[str, float]
    n_contributors: int
    notes: Tuple[str, ...]


# Arithmetic-third edges (low / mid / high), default partition for
# ``compute_band_majorities`` when no caller-supplied edges are passed.
DEFAULT_BAND_LABELS = ("low", "mid", "high")


def compute_band_majorities(
    contributor_freqs_mhz: np.ndarray,
    contributor_taus_us: np.ndarray,
    contributor_snrs: np.ndarray,
    *,
    trim_lo_mhz: float,
    trim_hi_mhz: float,
    band_edges_mhz: Optional[Tuple[float, ...]] = None,
    band_labels: Tuple[str, ...] = DEFAULT_BAND_LABELS,
    min_contributors_per_band: int = 50,
    sigma_floor_us: float = 0.5,
) -> Tuple[BandMajority, ...]:
    """SNR-weighted majority tau on each band of an arithmetic partition.

    Default partition is the three-band arithmetic split of
    ``[trim_lo_mhz, trim_hi_mhz)`` (the same split used by
    [`dev-docs/research/stage5-tau-calibration/lsq_comparison.py`](../../dev-docs/research/stage5-tau-calibration/lsq_comparison.py)
    for the Phase-4 LSQ comparison). Caller can pass explicit interior
    edges via ``band_edges_mhz`` (a 1-D sequence of strictly-increasing
    interior boundaries; outer edges are taken from ``trim_lo_mhz`` /
    ``trim_hi_mhz``).

    Bands with fewer than ``min_contributors_per_band`` are still returned
    but their ``(tau_maj_us, sigma_tau_us)`` collapse to the band-wide
    majority computed on *all* contributors. This is the safe fallback:
    a band that lacks information should not introduce a fresh tau anchor
    that the Stage 5 fits will follow into a worse local optimum.

    ``sigma_floor_us`` floors any band-local sigma below the floor at the
    floor; useful when a band has many contributors but they happen to
    cluster very tightly (e.g. all on one strong line). Stage 5's
    bidirectional Gaussian-prior penalty would otherwise become
    over-confident.
    """
    freqs = np.asarray(contributor_freqs_mhz, dtype=float)
    taus = np.asarray(contributor_taus_us, dtype=float)
    snrs = np.asarray(contributor_snrs, dtype=float)
    if freqs.size == 0:
        return tuple()

    if band_edges_mhz is None:
        n_bands = len(band_labels)
        step = (trim_hi_mhz - trim_lo_mhz) / float(n_bands)
        interior = tuple(trim_lo_mhz + (i + 1) * step for i in range(n_bands - 1))
    else:
        interior = tuple(float(e) for e in band_edges_mhz)
        if any(e <= trim_lo_mhz or e >= trim_hi_mhz for e in interior):
            raise ValueError(
                f"band_edges_mhz must lie strictly inside "
                f"({trim_lo_mhz}, {trim_hi_mhz}); got {interior}"
            )
        if not all(interior[i] < interior[i + 1] for i in range(len(interior) - 1)):
            raise ValueError(
                f"band_edges_mhz must be strictly increasing; got {interior}"
            )
        if len(interior) + 1 != len(band_labels):
            raise ValueError(
                f"band_labels has {len(band_labels)} entries but "
                f"band_edges_mhz implies {len(interior) + 1} bands"
            )
    edges: Tuple[float, ...] = (float(trim_lo_mhz),) + interior + (float(trim_hi_mhz),)

    # Band-wide fallback for short-contributor bands.
    fallback_tau, fallback_sigma = majority_tau(taus, snrs, weighted=True)

    out: list[BandMajority] = []
    for i, label in enumerate(band_labels):
        lo, hi = edges[i], edges[i + 1]
        mask = (freqs >= lo) & (freqs < hi)
        n_in = int(mask.sum())
        if n_in >= min_contributors_per_band:
            tau_b, sigma_b = majority_tau(
                taus[mask], snrs[mask], weighted=True,
            )
            if not np.isfinite(tau_b) or tau_b <= 0:
                tau_b, sigma_b = fallback_tau, fallback_sigma
        else:
            tau_b, sigma_b = fallback_tau, fallback_sigma
        if np.isfinite(sigma_b):
            sigma_b = float(max(sigma_b, sigma_floor_us))
        out.append(
            BandMajority(
                label=str(label),
                freq_lo_mhz=float(lo),
                freq_hi_mhz=float(hi),
                n=n_in,
                tau_maj_us=float(tau_b),
                sigma_tau_us=float(sigma_b),
            )
        )
    return tuple(out)


def band_majority_for_frequency(
    band_majorities: Tuple["BandMajority", ...],
    freq_mhz: float,
) -> Optional["BandMajority"]:
    """Return the band whose ``[freq_lo, freq_hi)`` contains ``freq_mhz``.

    Returns ``None`` if ``band_majorities`` is empty or ``freq_mhz`` lies
    outside every band. Callers should fall back to the band-wide
    ``(tau_maj_us, sigma_tau_us)`` in that case (the band-wide values are
    what would be active without per-band routing).
    """
    if not band_majorities:
        return None
    for bm in band_majorities:
        if bm.freq_lo_mhz <= freq_mhz < bm.freq_hi_mhz:
            return bm
    # Allow the high edge of the last band to match (closed-closed convention
    # at the very top so a contributor exactly at trim_hi_mhz still routes).
    last = band_majorities[-1]
    if freq_mhz == last.freq_hi_mhz:
        return last
    return None


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
        Per-band-third median tau (low / mid / high). Diagnostic only —
        carries the median per band, not the SNR-weighted majority.
    band_majorities : tuple of BandMajority
        Per-band SNR-weighted majority tau + spread. Empty tuple when
        ``compute_band_majorities=False`` was passed to
        :func:`extract_tau_majority`. Bands with fewer than
        ``min_contributors_per_band`` contributors fall back to the band-
        wide ``(tau_maj_us, sigma_tau_us)``; see
        :func:`compute_band_majorities` for the policy.
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
    # Optional per-band SNR-weighted majority tau. Default empty so existing
    # construction sites (and tests) continue to work without supplying it.
    # Stage 5 consumes this when fit_peaks_impl is called with
    # ``per_band_tau=True``; populated by extract_tau_majority when
    # ``compute_band_majorities_flag=True``.
    band_majorities: Tuple[BandMajority, ...] = field(default_factory=tuple)


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


def _nls_polish_step(
    mag: np.ndarray,
    a_centers_us: np.ndarray,
    tau: np.ndarray,
    C: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
    tau_clip_us: Tuple[float, float] = (0.1, 1e4),
    n_iter: int = 1,
    sigma_frame: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """``n_iter`` Gauss-Newton steps on ``|S_n| = C * exp(-a_n / tau)`` per bin.

    Removes the log-linear-weighting bias documented in Phase 1 § Case 1
    (a persistent +3-5 % positive shift at intermediate ``T_full / tau``
    ratios). The seed ``(tau, C)`` comes from :func:`_fit_exp_per_bin`'s
    weighted log-linear regression; Gauss-Newton converges quadratically
    near a well-conditioned minimum, so ``n_iter = 1`` lands at sub-1 %
    on top of the seed for most cells.

    When ``sigma_frame`` is supplied, the polish replaces ``|S_n|`` with
    the Rician-unbiased magnitude estimator
    ``sqrt(max(0, |S_n|^2 - 2 * sigma_frame^2))``: at high SNR this is
    ``~ |S_n|`` (signal dominates), at SNR ~ 1 it removes the
    ``E[|S_n|^2] = S^2 + 2 sigma^2`` noise contribution, at SNR << 1 it
    drives apparent signal to 0. The debiasing closes the residual
    +1-2 % bias that survives pure-NLS polish on intermediate-``T_full /
    tau`` cells (late frames sit at signal ~ noise where the noise
    contribution to ``|S_n|`` tilts apparent tau upward).

    Vectorised over the masked bins (default: every bin). Bins whose
    normal-equations determinant is degenerate (~zero) are returned
    unchanged so the polish never makes a bad fit worse. Updated ``tau``
    is clipped to ``tau_clip_us``; ``C`` is floored at ``1e-300``.

    Returns ``(tau_polished, C_polished)`` of the same shape as the inputs.
    """
    tau_arr = np.asarray(tau, dtype=float).copy()
    C_arr = np.asarray(C, dtype=float).copy()
    n_seg, n_bins = mag.shape
    if mask is None:
        mask = np.ones(n_bins, dtype=bool)
    sel = np.asarray(mask, dtype=bool)
    if not sel.any() or n_iter <= 0:
        return tau_arr, C_arr

    m_sel = mag[:, sel]
    if sigma_frame is not None and sigma_frame > 0.0:
        two_var = 2.0 * float(sigma_frame) ** 2
        m_sel = np.sqrt(np.maximum(m_sel * m_sel - two_var, 0.0))
    a = a_centers_us[:, None]

    t_sel = tau_arr[sel].copy()
    c_sel = C_arr[sel].copy()
    for _ in range(int(n_iter)):
        inv_t = 1.0 / np.where(t_sel > 0, t_sel, 1.0)  # (n_active,)
        e = np.exp(-a * inv_t[None, :])                # (n_seg, n_active)
        pred = c_sel[None, :] * e
        resid = m_sel - pred

        j_c = e
        j_t = (c_sel[None, :] * a * (inv_t ** 2)[None, :]) * e

        jtj_00 = (j_c * j_c).sum(axis=0)
        jtj_01 = (j_c * j_t).sum(axis=0)
        jtj_11 = (j_t * j_t).sum(axis=0)
        jtr_0 = (j_c * resid).sum(axis=0)
        jtr_1 = (j_t * resid).sum(axis=0)

        det = jtj_00 * jtj_11 - jtj_01 * jtj_01
        well_cond = np.abs(det) > 1e-30
        safe_det = np.where(well_cond, det, 1.0)
        d_c = (jtj_11 * jtr_0 - jtj_01 * jtr_1) / safe_det
        d_t = (-jtj_01 * jtr_0 + jtj_00 * jtr_1) / safe_det

        c_sel = np.where(well_cond, c_sel + d_c, c_sel)
        t_sel = np.where(well_cond, t_sel + d_t, t_sel)
        t_sel = np.clip(t_sel, tau_clip_us[0], tau_clip_us[1])
        c_sel = np.maximum(c_sel, 1e-300)

    tau_arr[sel] = t_sel
    C_arr[sel] = c_sel
    return tau_arr, C_arr


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
    C_per_bin: np.ndarray  # log-linear amplitude seed; polish input
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
    sigma_x_full: Optional[float] = None,
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
    sigma_x_full : float, optional
        Override for the per-bin full-record FT noise floor (in ``dt * rfft``
        units). When supplied, replaces the analytic
        ``sigma_t * dt * sqrt(N/2)`` derivation. Pass this when an
        independent spectral noise estimate (e.g. Stage 2's per-bin
        ``rms_noise`` median) is more accurate than the FID-tail
        ``sigma_t`` derivation -- on real fixtures the tail can carry
        residual signal that inflates the analytic value (2638: ~2x
        overestimate).
    """
    if sigma_time <= 0.0:
        raise ValueError("sigma_time must be positive")
    fid_arr = np.asarray(fid, dtype=float)
    N = fid_arr.size
    T_full_us = N * sample_dt_us
    if tau_max_us is None:
        tau_max_us = DEFAULT_TAU_MAX_FACTOR * T_full_us

    mag, a_centers_us, freq_bb_mhz = sliding_stft(fid_arr, sample_dt_us, n_seg)

    # Per-bin full-record FT noise floor, then per-frame. The override beats
    # the analytic ``sigma_t * dt * sqrt(N/2)`` derivation; pass the override
    # when an independent (e.g. Stage 2) spectral noise estimate is more
    # accurate than the FID-tail sigma_t.
    if sigma_x_full is not None and sigma_x_full > 0.0:
        sigma_x_full_v = float(sigma_x_full)
    else:
        sigma_x_full_v = sigma_time * sample_dt_us * np.sqrt(N / 2.0)
    sigma_frame = sigma_x_full_v / np.sqrt(n_seg)

    tau, C, rss_exp = _fit_exp_per_bin(
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
        C_per_bin=C,
        rss_exp=rss_exp,
        rss_const=rss_const,
        aicc_exp=aicc_exp,
        aicc_const=aicc_const,
        classification=classification,
        snr_per_bin=snr_per_bin,
        sigma_x_full=float(sigma_x_full_v),
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
    polish: bool = True,
    polish_n_iter: int = 1,
    polish_top_n: Optional[int] = None,
    polish_snr_cap: Optional[float] = DEFAULT_POLISH_SNR_CAP,
    polish_noise_debias: bool = False,
    sigma_x_full: Optional[float] = None,
    compute_band_majorities_flag: bool = False,
    band_edges_mhz: Optional[Tuple[float, ...]] = None,
    band_labels: Tuple[str, ...] = DEFAULT_BAND_LABELS,
    min_contributors_per_band: int = 50,
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
    polish : bool, default True
        Apply Gauss-Newton NLS step(s) to each contributor bin's
        ``(C, tau)`` seed before computing the majority. Closes the +3-5 %
        log-linear-weighting bias documented in Phase 1 § Case 1 to
        sub-percent on the case-1 grid; opt-out left as a forensic
        switch for A/B comparison against the legacy log-linear-only path.
    polish_n_iter : int, default 1
        Number of Gauss-Newton steps when ``polish`` is on. One step
        already lands at sub-1 % for most cells; bumping to 2-3 closes
        the residual on the intermediate-``T_full / tau`` regime
        documented in Phase 1 § Case 1 (a 1-µs run takes < 100 ms).
    polish_top_n : int, optional
        When set, the polish runs on only the ``polish_top_n`` highest-SNR
        contributor bins (the on-line bins of the strongest lines).
        Default ``None`` polishes every contributor. Limiting to top-N
        keeps the polish's correction local to high-confidence anchors
        and avoids shifting weak-skirt bins whose per-bin SNR is too low
        for one Gauss-Newton step to reliably improve.
    polish_snr_cap : float, optional
        When set, the polish runs only on contributors whose per-bin SNR
        is **below** ``polish_snr_cap``; high-SNR contributors retain the
        unpolished log-linear seed (the +3-5 % log-linear bias the polish
        targets concentrates at modest SNR, so applying it to high-SNR
        bins over-corrects). Pass ``None`` to disable the cap and polish
        every contributor (the legacy polish=True behaviour). Default is
        :data:`DEFAULT_POLISH_SNR_CAP`, calibrated against the
        LSQ-fit-and-histogram per-band reference on 2638 to land per-band
        SNR-weighted majority τ within ±5 % of the LSQ low/mid/high
        thirds. See
        ``dev-docs/research/stage5-tau-calibration/polish_snr_cap_validation.py``
        for the sweep and ``report.md`` § "Polish design" for the
        underlying rationale.
    polish_noise_debias : bool, default False
        Replace ``|S_n|`` with the Rician-unbiased magnitude
        ``sqrt(|S_n|^2 - 2 sigma^2)`` in the polish step. Theoretically
        correct for Gaussian complex noise; closes the single-isolated-
        line case-1 grid to sub-percent. But on multi-line spectra
        (including 2638-shape synthetics with realistic
        tau(f)/SNR(f) variation) the debiasing over-corrects -- the
        per-bin noise in real spectra includes inter-line skirt
        interference that the Rician model does not capture -- and
        biases ``tau_maj`` low. Left as an opt-in for forensic/case-1
        comparison; default off for production multi-line spectra.
    sigma_x_full : float, optional
        Override for the per-bin full-record FT noise floor (in
        ``dt * rfft`` amplitude units). When supplied, beats the FID-tail
        ``sigma_t`` derivation. Pass this when an independent spectral
        noise estimate (e.g. Stage 2's per-bin ``rms_noise`` median in
        the trim band, converted to raw amplitude units) is more
        accurate than the FID tail -- on real fixtures the tail can
        carry residual signal that inflates the analytic value (2638:
        ~2x overestimate). When the polish noise-debias kicks in
        (sigma_frame derivation), a too-large sigma over-subtracts and
        biases tau low.

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
        sigma_x_full=sigma_x_full,
    )

    # Baseband -> molecular conversion: lower sideband -> f_mol = probe - f_bb,
    # upper sideband -> f_mol = probe + f_bb.
    sign = -1.0 if sb == "lower" else +1.0
    freq_mol_mhz = probe_freq_mhz + sign * cal.freq_bb_mhz

    in_trim = (freq_mol_mhz >= trim_lo_mhz) & (freq_mol_mhz <= trim_hi_mhz)
    contributor_mask = (cal.classification == 3) & in_trim
    spur_mask_full = (cal.classification == 1) & in_trim

    # Single NLS Gauss-Newton step on the contributor bins removes the
    # +3-5 % log-linear-weighting bias before majority/histogram aggregation
    # (see Phase 1 § Case 1; ``_nls_polish_step``).
    tau_per_bin = cal.tau_per_bin
    if polish and contributor_mask.any():
        # Pure NLS polish: a single Gauss-Newton step on
        # ``|S_n| = C exp(-a/tau)`` per contributor bin. On a 2638-shape
        # multi-line synthetic with controlled tau(f), SNR(f) it shifts
        # the SNR-weighted majority from +2.7 % above truth (legacy
        # log-linear) to -2.2 % below truth -- a real improvement, though
        # it slightly overshoots. ``polish_noise_debias`` reaches
        # sub-1 % on the single-isolated-line case-1 grid but over-
        # corrects on multi-line spectra (inter-line skirt interference
        # is not Rician-Gaussian); default off.
        polish_sigma = (
            float(cal.sigma_frame) if polish_noise_debias else None
        )
        polish_mask = contributor_mask
        if polish_snr_cap is not None and polish_snr_cap > 0.0:
            # Polish only the contributors whose per-bin SNR sits below the
            # cap: the log-linear bias the polish targets is concentrated
            # at modest SNR, so high-SNR contributors keep the already-
            # near-unbiased log-linear seed.
            polish_mask = polish_mask & (cal.snr_per_bin < float(polish_snr_cap))
        if polish_mask.any():
            tau_polished, _C_polished = _nls_polish_step(
                cal.mag, cal.a_centers_us, cal.tau_per_bin, cal.C_per_bin,
                mask=polish_mask,
                tau_clip_us=(0.1, float(cal.tau_max_us)),
                n_iter=int(polish_n_iter),
                sigma_frame=polish_sigma,
            )
            tau_per_bin = tau_polished

    contributor_bins = np.where(contributor_mask)[0]
    # Sort contributors by molecular frequency (stable, helpful for serialization).
    contributor_bins = contributor_bins[np.argsort(freq_mol_mhz[contributor_bins])]
    contributor_taus = tau_per_bin[contributor_bins]
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

    if compute_band_majorities_flag and contributor_freqs.size > 0:
        bands = compute_band_majorities(
            contributor_freqs,
            contributor_taus,
            contributor_snrs,
            trim_lo_mhz=trim_lo_mhz,
            trim_hi_mhz=trim_hi_mhz,
            band_edges_mhz=band_edges_mhz,
            band_labels=band_labels,
            min_contributors_per_band=min_contributors_per_band,
        )
    else:
        bands = tuple()

    logger.info(
        "STFT tau calibration: tau_maj=%.3f sigma_tau=%.3f us "
        "(n_contrib=%d, n_spur_bins=%d, n_clusters=%d, bimodal=%s, "
        "preconditions=%s, n_bands=%d)",
        tau_maj, sigma_tau, contributor_bins.size, spur_bin_indices.size,
        len(spur_clusters), bm.two_component_preferred,
        "pass" if all_passed else "fail",
        len(bands),
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
        band_majorities=bands,
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


# ---------------------------------------------------------------------------
# Gaussian-twin calibration: per-bin pure-Gaussian fit -> τ_G majority.
# Voigt residual + multi-start helpers sit alongside for the 3-way
# L/G/V shape-recommendation comparator; the production
# ``stage2b_tau_G_calibration`` group is driven by the pure-Gaussian
# estimator so its τ_G matches the Stage 5 ``shape='gaussian'`` envelope.
# ---------------------------------------------------------------------------
def _voigt_residuals(
    params: np.ndarray, a: np.ndarray, y: np.ndarray
) -> np.ndarray:
    C, tau_L, tau_G = params
    return C * np.exp(-a / tau_L) * np.exp(-((a / tau_G) ** 2)) - y


def _gauss_residuals(
    params: np.ndarray, a: np.ndarray, y: np.ndarray
) -> np.ndarray:
    C, tau_G = params
    return C * np.exp(-((a / tau_G) ** 2)) - y


def _exp_residuals(
    params: np.ndarray, a: np.ndarray, y: np.ndarray
) -> np.ndarray:
    C, tau_L = params
    return C * np.exp(-a / tau_L) - y


def _fit_exp_nls_single(
    a: np.ndarray,
    y: np.ndarray,
    C0: float,
    tau0: float,
    *,
    tau_lo: float,
    tau_hi: float,
) -> Tuple[np.ndarray, float, bool]:
    """Polish the log-linear exp seed with a single-start NLS fit."""
    C0 = max(float(C0), 1e-30)
    tau0 = float(np.clip(tau0, tau_lo * 1.05, tau_hi * 0.95))
    res = least_squares(
        _exp_residuals,
        x0=np.array([C0, tau0]),
        bounds=([0.0, tau_lo], [np.inf, tau_hi]),
        args=(a, y),
        method="trf",
        max_nfev=200,
    )
    rss = float(np.sum(res.fun ** 2))
    return res.x, rss, bool(res.success)


def _fit_voigt_nls_multistart(
    a: np.ndarray,
    y: np.ndarray,
    C0: float,
    tau_L0: float,
    *,
    tau_lo: float,
    tau_hi: float,
    tau_G_seeds: Sequence[float],
) -> Tuple[np.ndarray, float, bool, float]:
    """Multi-start Voigt fit; returns the best-RSS basin.

    Returns ``(params, rss, converged, seed_tau_G_best)`` with
    ``params = (C, tau_L, tau_G)``.
    """
    C0 = max(float(C0), 1e-30)
    tau_L0 = float(np.clip(tau_L0, tau_lo * 1.05, tau_hi * 0.95))
    best_rss = np.inf
    best: Optional[Tuple[np.ndarray, float, bool, float]] = None
    for tG0 in tau_G_seeds:
        tG0 = float(np.clip(tG0, tau_lo * 1.05, tau_hi * 0.95))
        try:
            res = least_squares(
                _voigt_residuals,
                x0=np.array([C0, tau_L0, tG0]),
                bounds=(
                    [0.0, tau_lo, tau_lo],
                    [np.inf, tau_hi, tau_hi],
                ),
                args=(a, y),
                method="trf",
                max_nfev=400,
            )
        except Exception:  # noqa: BLE001
            continue
        rss = float(np.sum(res.fun ** 2))
        if rss < best_rss:
            best_rss = rss
            best = (res.x, rss, bool(res.success), tG0)
    if best is None:
        return (
            np.array([C0, tau_L0, tau_hi * 0.95]),
            float("inf"),
            False,
            float("nan"),
        )
    return best


def _fit_gauss_nls_multistart(
    a: np.ndarray,
    y: np.ndarray,
    C0: float,
    *,
    tau_lo: float,
    tau_hi: float,
    tau_G_seeds: Sequence[float],
) -> Tuple[np.ndarray, float, bool, float]:
    """Multi-start pure-Gaussian fit ``|S| = C exp(-(a/τ_G)²)``.

    Returns ``(params, rss, converged, seed_tau_G_best)`` with
    ``params = (C, tau_G)``. Mirrors :func:`_fit_voigt_nls_multistart`
    but with one fewer parameter -- the pure-Gaussian model is the
    Stage 5 ``shape='gaussian'`` window-fit envelope, so its recovered
    τ_G is what the downstream Gaussian Stage 5 fit will see.
    """
    C0 = max(float(C0), 1e-30)
    best_rss = np.inf
    best: Optional[Tuple[np.ndarray, float, bool, float]] = None
    for tG0 in tau_G_seeds:
        tG0 = float(np.clip(tG0, tau_lo * 1.05, tau_hi * 0.95))
        try:
            res = least_squares(
                _gauss_residuals,
                x0=np.array([C0, tG0]),
                bounds=([0.0, tau_lo], [np.inf, tau_hi]),
                args=(a, y),
                method="trf",
                max_nfev=400,
            )
        except Exception:  # noqa: BLE001
            continue
        rss = float(np.sum(res.fun ** 2))
        if rss < best_rss:
            best_rss = rss
            best = (res.x, rss, bool(res.success), tG0)
    if best is None:
        return (
            np.array([C0, tau_hi * 0.95]),
            float("inf"),
            False,
            float("nan"),
        )
    return best


def extract_tau_G_majority(
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
    snr_min: float = DEFAULT_TAU_G_SNR_MIN,
    tau_G_bound_lo: float = DEFAULT_TAU_G_BOUND_LO,
    tau_G_bound_hi: float = DEFAULT_TAU_G_BOUND_HI,
    tau_G_seeds: Sequence[float] = DEFAULT_TAU_G_SEEDS,
    delta_chi2r_min: float = DEFAULT_TAU_G_DELTA_CHI2R_MIN,
    tau_G_upper_fraction: float = DEFAULT_TAU_G_UPPER_FRACTION,
    min_contributors: int = DEFAULT_TAU_G_MIN_CONTRIBUTORS,
    sigma_tau_fraction_max: float = DEFAULT_SIGMA_TAU_FRACTION_MAX,
    bimodality_dominant_fraction: float = DEFAULT_BIMODALITY_DOMINANT_FRACTION,
    sigma_x_full: Optional[float] = None,
    compute_band_majorities_flag: bool = True,
    band_edges_mhz: Optional[Tuple[float, ...]] = None,
    band_labels: Tuple[str, ...] = DEFAULT_BAND_LABELS,
    min_contributors_per_band: int = 10,
) -> TauCalibrationResult:
    """End-to-end STFT τ_G calibration for the Gaussian-shape Stage 5 fit.

    Twin of :func:`extract_tau_majority`. Algorithm:

    1. Run the same sliding-active-window STFT and bin classifier.
    2. Restrict to strong contributor bins (``classification == 3`` AND
       per-bin ``SNR > snr_min``).
    3. Per-bin: polish the log-linear pure-exp seed (NLS), then multi-start
       pure-Gaussian fit ``|S_n(a)| = C exp(-(a/τ_G)²)`` with the
       ``tau_G_seeds`` grid; keep the best-RSS basin. The pure-Gaussian
       model matches the Stage 5 ``shape='gaussian'`` window-fit envelope,
       so the recovered ``τ_G`` is what the downstream window fit will
       see. A Voigt decomposition's ``τ_G`` would describe the Gaussian
       component *after* the Lorentzian decay had been absorbed into a
       separate ``τ_L`` -- larger than the envelope-equivalent τ that the
       window fit recovers.
    4. Calibration-eligible mask: pure-Gauss converged, finite τ_G away
       from the upper bound (``τ_G < tau_G_upper_fraction *
       tau_G_bound_hi``), and pure-Gauss beats pure-exp by at least
       ``delta_chi2r_min`` χ²ᵣ units. Bins where pure-exp wins are
       pure-Lorentzian and contribute no τ_G information.
    5. Persisted "contributors" = the eligible subset. The SNR-weighted
       majority over them gives ``τ_G_maj`` and ``σ_τ_G``; the per-band
       majorities (when ``compute_band_majorities_flag``) use the same
       eligibility filter inside each band.

    The result reuses :class:`TauCalibrationResult` so the existing HDF5
    serialisation can persist it unchanged (under a different group path).
    Semantic interpretation: every ``τ`` / ``tau`` field carries ``τ_G``
    when this twin is the producer; the group path
    ``/stage2b_tau_G_calibration`` disambiguates from the pure-exp twin.

    Parameters
    ----------
    fid, sample_dt_us, start_us, end_us, probe_freq_mhz, sideband,
    trim_lo_mhz, trim_hi_mhz, sigma_time, n_seg, t_sigma, tau_max_us,
    rss_gate_factor, relative_gate_fraction, spur_cluster_multiplier,
    sigma_x_full
        STFT calibration knobs; defaults match the pure-exp twin so the
        same bin classifier produces the same contributor pool.
    snr_min : float, default :data:`DEFAULT_TAU_G_SNR_MIN`
        Per-bin SNR floor on the contributor pool. Below this the
        pure-Gauss vs pure-exp χ²ᵣ discriminator has too little signal-
        to-noise to be informative.
    tau_G_bound_lo, tau_G_bound_hi : float
        Bounds on the pure-Gaussian ``τ_G`` parameter (microseconds).
        The upper bound is the saturation ceiling whose proximity flags
        a bin as Gaussian-uninformative.
    tau_G_seeds : sequence of float
        Multi-start grid for the pure-Gaussian fit's τ_G seed; a bin
        whose true τ_G lies far from any seed can still recover the
        correct basin through the multi-start sweep.
    delta_chi2r_min : float, default :data:`DEFAULT_TAU_G_DELTA_CHI2R_MIN`
        Minimum χ²ᵣ improvement (pure-exp − pure-Gauss) required for a
        bin to enter the calibration. With matched parameter counts
        (k=2 for both models) the test reduces directly to "data is
        more pure-Gaussian than pure-Lorentzian on this bin".
    tau_G_upper_fraction : float, default
        :data:`DEFAULT_TAU_G_UPPER_FRACTION`
        Bins whose recovered τ_G ≥ ``tau_G_upper_fraction * tau_G_bound_hi``
        are saturated against the bound and dropped (no Gaussian content).
    min_contributors, sigma_tau_fraction_max, bimodality_dominant_fraction
        Acceptance pre-conditions. ``min_contributors`` is lower here
        than in the pure-exp twin (defaults differ) because the eligible
        pool is naturally smaller.
    compute_band_majorities_flag, band_edges_mhz, band_labels,
    min_contributors_per_band
        Per-band SNR-weighted majority τ_G control. Default ``True`` so
        the Gaussian Stage 5 path can pick up per-band anchors out of the
        box.

    Raises
    ------
    ValueError
        On the same input-validation failures as
        :func:`extract_tau_majority` (sideband, sample_dt_us, end_us,
        trim_hi_mhz).
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
    if tau_G_bound_hi <= tau_G_bound_lo:
        raise ValueError(
            f"tau_G_bound_hi ({tau_G_bound_hi}) must exceed tau_G_bound_lo "
            f"({tau_G_bound_lo})"
        )
    if not 0.0 < tau_G_upper_fraction < 1.0:
        raise ValueError(
            f"tau_G_upper_fraction must lie in (0, 1); got {tau_G_upper_fraction}"
        )

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
        sigma_x_full=sigma_x_full,
    )

    sign = -1.0 if sb == "lower" else +1.0
    freq_mol_mhz = probe_freq_mhz + sign * cal.freq_bb_mhz
    in_trim = (freq_mol_mhz >= trim_lo_mhz) & (freq_mol_mhz <= trim_hi_mhz)

    contributor_mask = (
        (cal.classification == 3) & in_trim & (cal.snr_per_bin > float(snr_min))
    )
    bin_indices = np.where(contributor_mask)[0]
    bin_indices = bin_indices[np.argsort(freq_mol_mhz[bin_indices])]

    a_centers = cal.a_centers_us.astype(float)
    sigma_frame = float(cal.sigma_frame)
    dof_exp = max(int(n_seg) - 2, 1)
    dof_gauss = max(int(n_seg) - 2, 1)
    sigma_frame_sq = max(sigma_frame * sigma_frame, 1e-300)
    tau_G_cap = float(tau_G_upper_fraction) * float(tau_G_bound_hi)
    seeds = tuple(float(s) for s in tau_G_seeds)

    eligible_bins: list[int] = []
    eligible_tau_G: list[float] = []
    eligible_snr: list[float] = []
    eligible_freq: list[float] = []

    for idx in bin_indices:
        mag_bin = cal.mag[:, int(idx)].astype(float)
        snr_bin = float(cal.snr_per_bin[int(idx)])
        freq_bin = float(freq_mol_mhz[int(idx)])

        C_seed = float(cal.C_per_bin[int(idx)])
        tau_seed = float(cal.tau_per_bin[int(idx)])
        exp_params, rss_exp, _ok_exp = _fit_exp_nls_single(
            a_centers, mag_bin, C_seed, tau_seed,
            tau_lo=float(tau_G_bound_lo), tau_hi=float(tau_G_bound_hi),
        )
        g_params, rss_gauss, ok_gauss, _seed_best = _fit_gauss_nls_multistart(
            a_centers, mag_bin, float(exp_params[0]),
            tau_lo=float(tau_G_bound_lo), tau_hi=float(tau_G_bound_hi),
            tau_G_seeds=seeds,
        )
        chi2r_exp = rss_exp / sigma_frame_sq / dof_exp
        chi2r_gauss = rss_gauss / sigma_frame_sq / dof_gauss
        delta_chi2r = chi2r_exp - chi2r_gauss
        tau_G = float(g_params[1])
        if (
            ok_gauss
            and np.isfinite(tau_G)
            and tau_G < tau_G_cap
            and delta_chi2r >= float(delta_chi2r_min)
        ):
            eligible_bins.append(int(idx))
            eligible_tau_G.append(tau_G)
            eligible_snr.append(snr_bin)
            eligible_freq.append(freq_bin)

    contributor_bins = np.asarray(eligible_bins, dtype=np.int64)
    contributor_taus = np.asarray(eligible_tau_G, dtype=np.float64)
    contributor_snrs = np.asarray(eligible_snr, dtype=np.float64)
    contributor_freqs = np.asarray(eligible_freq, dtype=np.float64)

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

    spur_mask_full = (cal.classification == 1) & in_trim
    spur_bin_indices = np.where(spur_mask_full)[0]
    spur_clusters = group_spur_bins(
        spur_bin_indices,
        cal.mag.mean(axis=0),
        freq_mol_mhz,
        n_seg=n_seg,
        cluster_multiplier=spur_cluster_multiplier,
    )

    notes: list[str] = []
    cond_count = contributor_bins.size >= int(min_contributors)
    notes.append(
        "ok"
        if cond_count
        else f"only {contributor_bins.size} eligible bins (< {min_contributors})"
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
        notes.append("tau_G non-positive; spread test undefined")
    all_passed = bool(cond_count and cond_bimodal and cond_spread)

    if compute_band_majorities_flag and contributor_freqs.size > 0:
        bands = compute_band_majorities(
            contributor_freqs,
            contributor_taus,
            contributor_snrs,
            trim_lo_mhz=trim_lo_mhz,
            trim_hi_mhz=trim_hi_mhz,
            band_edges_mhz=band_edges_mhz,
            band_labels=band_labels,
            min_contributors_per_band=int(min_contributors_per_band),
        )
    else:
        bands = tuple()

    logger.info(
        "STFT τ_G calibration: tau_G_maj=%.3f sigma_tau_G=%.3f us "
        "(n_eligible=%d / contributor_pool=%d, n_spur_bins=%d, "
        "n_clusters=%d, preconditions=%s, n_bands=%d)",
        tau_maj, sigma_tau, contributor_bins.size, bin_indices.size,
        spur_bin_indices.size, len(spur_clusters),
        "pass" if all_passed else "fail", len(bands),
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
        band_majorities=bands,
        contributor_bin_indices=contributor_bins,
        contributor_taus_us=contributor_taus,
        contributor_snrs=contributor_snrs,
        contributor_freqs_mhz=contributor_freqs,
        n_seg=int(n_seg),
        t_sigma=float(t_sigma),
        tau_max_us=float(tau_G_bound_hi),
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


# ---------------------------------------------------------------------------
# 3-way L / G / V shape-recommendation hook (per-bin AICc vote)
# ---------------------------------------------------------------------------
def _fit_per_bin_three_way(
    cal: _STFTClassification,
    bin_indices: np.ndarray,
    freq_mol_mhz: np.ndarray,
    *,
    tau_lo: float,
    tau_hi: float,
    tau_G_seeds: Sequence[float],
    n_seg: int,
) -> List[Dict[str, Any]]:
    """For each contributor bin, fit exp / gauss / voigt and compute AICc.

    Returns a list of dicts (one per converged bin). Bins where any of
    the three fits fails to converge are skipped; the caller's
    aggregator divides over the survivors.
    """
    a_centers = cal.a_centers_us.astype(float)
    rows: List[Dict[str, Any]] = []
    seeds = tuple(float(s) for s in tau_G_seeds)
    for idx in bin_indices:
        mag_bin = cal.mag[:, int(idx)].astype(float)
        snr_bin = float(cal.snr_per_bin[int(idx)])
        freq_bin = float(freq_mol_mhz[int(idx)])
        C_seed = float(cal.C_per_bin[int(idx)])
        tau_seed = float(cal.tau_per_bin[int(idx)])
        exp_params, rss_exp, ok_exp = _fit_exp_nls_single(
            a_centers, mag_bin, C_seed, tau_seed,
            tau_lo=tau_lo, tau_hi=tau_hi,
        )
        g_params, rss_gauss, ok_gauss, _ = _fit_gauss_nls_multistart(
            a_centers, mag_bin, float(exp_params[0]),
            tau_lo=tau_lo, tau_hi=tau_hi, tau_G_seeds=seeds,
        )
        v_params, rss_voigt, ok_voigt, _ = _fit_voigt_nls_multistart(
            a_centers, mag_bin, float(exp_params[0]), float(exp_params[1]),
            tau_lo=tau_lo, tau_hi=tau_hi, tau_G_seeds=seeds,
        )
        if not (ok_exp and ok_gauss and ok_voigt):
            continue
        aicc_exp = float(_aicc(np.array([rss_exp]), n_seg, k=2)[0])
        aicc_gauss = float(_aicc(np.array([rss_gauss]), n_seg, k=2)[0])
        aicc_voigt = float(_aicc(np.array([rss_voigt]), n_seg, k=3)[0])
        scores = {"exp": aicc_exp, "gauss": aicc_gauss, "voigt": aicc_voigt}
        verdict = min(scores, key=scores.get)
        rows.append(dict(
            idx=int(idx), freq=freq_bin, snr=snr_bin,
            tau_L_exp=float(exp_params[1]),
            tau_G_gauss=float(g_params[1]),
            tau_L_voigt=float(v_params[1]),
            tau_G_voigt=float(v_params[2]),
            aicc_exp=aicc_exp, aicc_gauss=aicc_gauss, aicc_voigt=aicc_voigt,
            d_aicc_gauss_exp=aicc_gauss - aicc_exp,
            d_aicc_voigt_exp=aicc_voigt - aicc_exp,
            d_aicc_voigt_gauss=aicc_voigt - aicc_gauss,
            verdict=verdict,
        ))
    return rows


def _shape_recommendation_bin_clean(
    row: Dict[str, Any], tau_cap_us: float,
) -> bool:
    """Per-bin acceptance gate for the 3-way recommendation pool.

    A bin contributes to the vote when at least one of the three
    candidate τ values lies strictly inside ``[0, tau_cap_us)``. A bin
    where every fit's τ saturates against the upper bound has no
    informative shape to vote on (its time series is essentially
    flat or noise-dominated) and gets dropped before the SNR-weighted
    tally.
    """
    return (
        row["tau_L_exp"] < tau_cap_us
        or row["tau_G_gauss"] < tau_cap_us
        or row["tau_L_voigt"] < tau_cap_us
        or row["tau_G_voigt"] < tau_cap_us
    )


def _aggregate_shape_verdict(
    rows: List[Dict[str, Any]],
    *,
    pure_margin_threshold: float = DEFAULT_SHAPE_RECOMMENDATION_PURE_MARGIN,
) -> ShapeRecommendation:
    """Aggregate per-bin 3-way AICc rows into a single ShapeRecommendation.

    Algorithm:

    1. SNR-weighted vote rates over the per-bin ``argmin AICc`` verdicts.
    2. If neither pure shape beats the other by ``pure_margin_threshold``
       of the total weight, return ``recommended_shape=None`` -- the data
       does not strongly favour one pure shape over the other and the
       Stage 5 resolver's *recommended* layer falls through. Otherwise
       recommend the dominant pure shape (``"lorentzian"`` for exp,
       ``"gaussian"`` for gauss). Voigt vote mass is reported but does
       not enter the recommendation because the production Stage 5
       line-shape selector supports L and G only.
    """
    if not rows:
        return ShapeRecommendation(
            recommended_shape=None,
            vote_rates={"exp": 0.0, "gauss": 0.0, "voigt": 0.0},
            median_d_aicc={
                "gauss_vs_exp": float("nan"),
                "voigt_vs_exp": float("nan"),
                "voigt_vs_gauss": float("nan"),
            },
            n_contributors=0,
            notes=("no contributor bins survived the per-bin three-way fit",),
        )
    snrs = np.asarray([r["snr"] for r in rows], dtype=float)
    total_w = float(np.sum(snrs))
    vote_rates: Dict[str, float] = {}
    for m in ("exp", "gauss", "voigt"):
        mask = np.array([r["verdict"] == m for r in rows], dtype=bool)
        vote_rates[m] = (
            float(np.sum(snrs[mask]) / total_w) if total_w > 0 else 0.0
        )

    median_d_aicc = {
        "gauss_vs_exp": float(np.median([r["d_aicc_gauss_exp"] for r in rows])),
        "voigt_vs_exp": float(np.median([r["d_aicc_voigt_exp"] for r in rows])),
        "voigt_vs_gauss": float(np.median([r["d_aicc_voigt_gauss"] for r in rows])),
    }

    notes: List[str] = [
        f"n_contributors={len(rows)}; "
        f"vote rates exp={vote_rates['exp']*100:.1f}% "
        f"gauss={vote_rates['gauss']*100:.1f}% "
        f"voigt={vote_rates['voigt']*100:.1f}%"
    ]
    exp_rate = vote_rates["exp"]
    gauss_rate = vote_rates["gauss"]
    pure_margin = abs(exp_rate - gauss_rate)
    if pure_margin >= pure_margin_threshold:
        recommendation = "lorentzian" if exp_rate > gauss_rate else "gaussian"
        notes.append(
            f"pure-shape margin {pure_margin*100:.1f}% >= "
            f"{pure_margin_threshold*100:.0f}% threshold; "
            f"recommend {recommendation!r}"
        )
    else:
        recommendation = None
        notes.append(
            f"pure-shape margin {pure_margin*100:.1f}% < "
            f"{pure_margin_threshold*100:.0f}% threshold; "
            f"no clear winner (recommended_shape=None)"
        )

    return ShapeRecommendation(
        recommended_shape=recommendation,
        vote_rates=vote_rates,
        median_d_aicc=median_d_aicc,
        n_contributors=len(rows),
        notes=tuple(notes),
    )


def compute_shape_recommendation(
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
    snr_min: float = DEFAULT_TAU_G_SNR_MIN,
    tau_bound_lo: float = DEFAULT_TAU_G_BOUND_LO,
    tau_bound_hi: float = DEFAULT_TAU_G_BOUND_HI,
    tau_G_seeds: Sequence[float] = DEFAULT_TAU_G_SEEDS,
    pure_margin_threshold: float = DEFAULT_SHAPE_RECOMMENDATION_PURE_MARGIN,
    include_bad_fit_bins: bool = False,
    sigma_x_full: Optional[float] = None,
) -> ShapeRecommendation:
    """Compute a 3-way L / G / V shape recommendation from a raw FID.

    Runs the same sliding-active-window STFT classifier as
    :func:`extract_tau_majority` and :func:`extract_tau_G_majority`,
    then on every contributor bin (``classification == 3`` ∧
    ``SNR > snr_min``) fits all three single-shape models -- pure
    exponential (``|S| = C exp(-a/τ_L)``, k=2), pure Gaussian
    (``|S| = C exp(-(a/τ_G)²)``, k=2), and Voigt
    (``|S| = C exp(-a/τ_L) exp(-(a/τ_G)²)``, k=3) -- with the small-
    sample-corrected AICc. The per-bin verdict is ``argmin AICc``; the
    SNR-weighted majority vote across all converged bins drives the
    recommendation.

    The Voigt model is the most expressive of the three (one extra
    parameter), and on real instrumental envelopes will often win the
    raw vote without one pure shape being a poor description of the
    data. The aggregator picks between the two *pure* shapes (exp ⇒
    ``"lorentzian"``, gauss ⇒ ``"gaussian"``) based on their relative
    vote mass; the Voigt votes are reported as diagnostic but do not
    enter the recommendation. When neither pure shape wins by at least
    ``pure_margin_threshold`` of the total weight, the recommendation
    is ``None`` (the data does not strongly favour one pure shape and
    the Stage 5 resolver's *recommended* layer falls through to the
    next layer).

    The STFT classifier (:func:`stft_calibration`) labels bins as
    ``cls=3`` (clean contributor) using a *pure-exp* bad-fit gate;
    strong on-line bins on Gaussian-envelope fixtures often fail that
    gate and end up as ``cls=2`` (bad-fit per the exp model) even
    though their pure-Gauss fit is clean. The default pool is
    ``cls=3`` (matching the τ calibrations) so the recommendation and
    the persisted τ are computed on the same contributor set. Setting
    ``include_bad_fit_bins=True`` expands the pool to ``cls ∈ {2, 3}``;
    on fixtures with many skirt / overlapping-line ``cls=2`` bins (e.g.
    2638) the expansion dilutes the strong on-line contribution and
    can flip the verdict to "no clear winner" -- a shape-aware
    classifier or a Stage-3-peak-anchored contributor pool would be
    the more principled fix and is tracked in
    ``dev-docs/planning/stage5-gaussian-shape.md``.

    See :class:`ShapeRecommendation` for the returned struct.

    Parameters
    ----------
    fid, sample_dt_us, start_us, end_us, probe_freq_mhz, sideband,
    trim_lo_mhz, trim_hi_mhz, sigma_time, n_seg, t_sigma, tau_max_us,
    rss_gate_factor, relative_gate_fraction, sigma_x_full
        STFT classifier knobs (identical defaults to
        :func:`extract_tau_majority` /
        :func:`extract_tau_G_majority` so the same contributor pool
        feeds all three).
    snr_min
        Per-bin SNR floor on the contributor pool. Below this the
        per-bin AICc discriminator has too little signal-to-noise to
        be informative.
    tau_bound_lo, tau_bound_hi
        Bounds on τ_L and τ_G inside the fits.
    tau_G_seeds
        Multi-start seed grid for the Gaussian and Voigt fits.
    pure_margin_threshold
        Minimum SNR-weighted vote-rate margin between exp and gauss
        for a pure-shape recommendation to fire.
    include_bad_fit_bins
        When ``True``, expand the candidate pool to include
        ``classification == 2`` bins in addition to ``cls=3``. Default
        is ``False`` so the recommendation pool matches the τ
        calibrations' pool. On fixtures with many skirt ``cls=2``
        bins (the typical case) the expansion dilutes the strong
        on-line contribution; the kwarg is exposed for diagnostic
        experiments rather than as a principled improvement to the
        verdict.
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
    if tau_bound_hi <= tau_bound_lo:
        raise ValueError(
            f"tau_bound_hi ({tau_bound_hi}) must exceed tau_bound_lo "
            f"({tau_bound_lo})"
        )

    fid_arr = np.asarray(fid, dtype=float)
    start_idx = max(int(round(start_us / sample_dt_us)), 0)
    end_idx = min(int(round(end_us / sample_dt_us)), fid_arr.size)
    if end_idx - start_idx < 4 * n_seg:
        raise ValueError(
            f"active region [{start_us}, {end_us}) us has too few samples "
            f"({end_idx - start_idx}) for n_seg={n_seg}"
        )
    active = fid_arr[start_idx:end_idx]
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
        sigma_x_full=sigma_x_full,
    )

    sign = -1.0 if sb == "lower" else +1.0
    freq_mol_mhz = probe_freq_mhz + sign * cal.freq_bb_mhz
    in_trim = (freq_mol_mhz >= trim_lo_mhz) & (freq_mol_mhz <= trim_hi_mhz)
    above_snr = cal.snr_per_bin > float(snr_min)
    if include_bad_fit_bins:
        contributor_mask = (
            ((cal.classification == 3) | (cal.classification == 2))
            & in_trim & above_snr
        )
    else:
        contributor_mask = (
            (cal.classification == 3) & in_trim & above_snr
        )
    bin_indices = np.where(contributor_mask)[0]
    bin_indices = bin_indices[np.argsort(freq_mol_mhz[bin_indices])]

    rows = _fit_per_bin_three_way(
        cal, bin_indices, freq_mol_mhz,
        tau_lo=float(tau_bound_lo), tau_hi=float(tau_bound_hi),
        tau_G_seeds=tau_G_seeds, n_seg=int(n_seg),
    )
    # Per-bin acceptance: keep bins where at least one of the three
    # candidates produces a τ that lies inside the bound, so genuine
    # noise / blend bins (where every fit saturates against the upper
    # bound or fails to converge) drop out of the vote.
    tau_cap_us = float(DEFAULT_TAU_G_UPPER_FRACTION) * float(tau_bound_hi)
    rows = [r for r in rows if _shape_recommendation_bin_clean(r, tau_cap_us)]
    verdict = _aggregate_shape_verdict(
        rows, pure_margin_threshold=float(pure_margin_threshold),
    )

    logger.info(
        "3-way shape recommendation: n_contributors=%d, vote rates "
        "exp=%.1f%% gauss=%.1f%% voigt=%.1f%%, median ΔAICc(gauss-exp)=%.2f, "
        "ΔAICc(voigt-exp)=%.2f, ΔAICc(voigt-gauss)=%.2f -> recommendation=%s",
        verdict.n_contributors,
        verdict.vote_rates["exp"] * 100,
        verdict.vote_rates["gauss"] * 100,
        verdict.vote_rates["voigt"] * 100,
        verdict.median_d_aicc["gauss_vs_exp"],
        verdict.median_d_aicc["voigt_vs_exp"],
        verdict.median_d_aicc["voigt_vs_gauss"],
        verdict.recommended_shape,
    )
    return verdict
