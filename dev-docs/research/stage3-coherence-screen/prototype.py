"""
Stage 3 projection-coherence screen -- reproducibility script.

Regenerates every figure under ``figures/`` and the empirical numbers
cited in ``report.md``. From the repository root, with the project
conda env:

    conda run -n ftmwpipeline-dev python \\
        dev-docs/research/stage3-coherence-screen/prototype.py

The screen takes a Stage 3 candidate, projects a small sub-window of
the active-portion FT onto a unit-amplitude finite-T Lorentzian basis
centred at the candidate, and returns the σ-weighted ratio of the
coherent-projection SNR to the on-line detected SNR. A real Lorentzian
projects to ratio ≈ 1; a phase-incoherent noise excursion drops below
1. The companion module is
``src/ftmwpipeline/preprocessing/coherence_screen.py``.

The investigation has four sections:

1. **Simulator**. A synthetic-FID active-FT generator parameterised by
   ``true_snr`` (per-line, on the on-line active-FT bin) and
   ``fwhm_bins`` (FWHM as a multiple of the active-FT bin spacing).
   Ground truth: injected line frequencies. The simulator returns the
   active-FT, per-bin σ, and the ground-truth frequency set, in the
   natural ``dt_us * rfft(active)`` convention the helper expects.

2. **Single-cell illustration**. A panel of three example regimes
   (sub-bin, ~1.5 bins, ~3 bins per FWHM) at a single SNR to show
   visually what the ratio is doing — the basis overlaid on the data,
   the per-candidate ratio cloud, and where the threshold cuts.

3. **Phase-space sweep**. A grid over (true SNR, FWHM/bin) recording
   per-candidate ratio + ground-truth label + ROC. Produces an AUC
   heatmap, a precision-at-recall=0.95 heatmap, and ROC curves for
   selected cells.

4. **τ_basis mismatch sensitivity**. The same sweep with the basis τ
   set to 0.5×τ_truth and 2×τ_truth, to quantify how much an
   incorrectly-defaulted basis costs the screen.

Outputs are written to ``figures/`` (PNGs) and ``data/`` (.npz
intermediates). The script is idempotent: re-running overwrites
existing artefacts. Synthetic sections need no inputs; the 2638
reality check (section 5, optional) reads
``scratch/stage5-validation/exp_2638.ftmw`` if it exists.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from ftmwpipeline.core.data_structures import Sideband
from ftmwpipeline.fitting.active_ft import compute_active_ft
from ftmwpipeline.fitting.peak_model import h_T
from ftmwpipeline.preprocessing.coherence_screen import project_candidates
from ftmwpipeline.preprocessing.noise_estimation import estimate_noise_adaptive

logger = logging.getLogger("coherence-screen-research")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

HERE = Path(__file__).parent
FIG = HERE / "figures"
DATA = HERE / "data"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
REPO_ROOT = Path(__file__).resolve().parents[3]
RNG_SEED = 20260524

# The simulator's active region must be wide enough that even the
# widest FWHM in the sweep fits ``n_lines`` non-clustered Lorentzians.
# With ``n_lines=25``, ``min_line_separation_bins=6`` and
# ``fwhm_bins=20`` the rfft band needs ~25*(20+6) + edge_guard = ~670
# bins. n_active=4096 gives 2049 rfft bins -- comfortable headroom.
SIM_N_ACTIVE = 4096


# ===========================================================================
# Section 1: simulator
# ===========================================================================
@dataclass(frozen=True)
class SyntheticActiveFT:
    """One realised synthetic active-FT plus ground truth.

    Attributes
    ----------
    freq_mhz : np.ndarray
        Active-FT molecular frequency grid (MHz), ascending.
    spectrum : np.ndarray
        Complex active-FT in ``dt_us * rfft(active)`` units.
    sigma : np.ndarray
        Per-bin |X| RMS noise (Stage 2 estimator output applied to
        ``|spectrum|``).
    truth_freqs_mhz : np.ndarray
        True injected line frequencies (MHz), molecular axis. Each
        corresponds to an injected Lorentzian of unit amplitude.
    truth_snrs : np.ndarray
        Target true SNR per injected line (the parameter the simulator
        was asked to hit). The realised on-line SNR fluctuates around
        this with Rayleigh-scale noise.
    tau_truth_us : float
        Effective decay constant of the injected lines (the data's true
        ``τ_eff``).
    acquisition_us : float
        Active-region length ``T``.
    bin_mhz : float
        Active-FT bin spacing ``1/T``.
    fwhm_bins : float
        FWHM of an injected line in bin units (``1/(π τ_truth T)``).
    """

    freq_mhz: np.ndarray
    spectrum: np.ndarray
    sigma: np.ndarray
    truth_freqs_mhz: np.ndarray
    truth_snrs: np.ndarray
    tau_truth_us: float
    acquisition_us: float
    bin_mhz: float
    fwhm_bins: float
    # Strong-line / sidelobe-FP extension. ``strong_freqs_mhz`` may be an
    # empty array when no strong lines are injected (the matched-τ sweep's
    # original use case).
    strong_freqs_mhz: np.ndarray
    strong_snrs: np.ndarray


def _adaptive_noise(freq: np.ndarray, mag: np.ndarray) -> np.ndarray:
    res = estimate_noise_adaptive(freq, mag)
    return np.asarray(res.rms_noise, dtype=float)


def simulate_active_ft(
    *,
    fwhm_bins: float,
    true_snr: float,
    n_lines: int = 25,
    n_active: int = 4096,
    sample_dt_us: float = 0.020,
    probe_freq_mhz: float = 40000.0,
    sideband: Sideband = Sideband.LOWER,
    edge_guard_bins: int = 20,
    min_line_separation_bins: float = 6.0,
    rng_seed: int = 0,
    n_strong_lines: int = 0,
    strong_snr: float = 50.0,
    strong_separation_bins: float = 30.0,
) -> SyntheticActiveFT:
    """Generate a synthetic active-FT with ``n_lines`` injected Lorentzians.

    Time-domain construction (no apodization -- the line decay is
    realised by setting the FID's effective ``τ`` directly). Real FID
    samples with Gaussian noise; the noise variance is set so the
    requested ``true_snr`` matches the on-line magnitude over the
    Stage-2-estimated per-bin σ.

    Parameters
    ----------
    fwhm_bins : float
        Target line FWHM in active-FT bin units. With
        ``Δf_bin = 1/(n_active · sample_dt_us)``, this fixes
        ``τ_truth = n_active · sample_dt_us / (π · fwhm_bins)``.
    true_snr : float
        On-line SNR of each injected line (against the active-FT |X|
        RMS noise from the Stage 2 adaptive estimator).
    n_lines : int
        Number of injected lines.
    n_active : int
        Active-region length in samples. With the default
        ``sample_dt_us = 0.020`` this gives ``T_active = 20.48 µs``,
        ``bin = 0.0488 MHz`` -- enough bandwidth for ~50 MHz of
        spectrum at the synthetic probe.
    sample_dt_us : float
        FID sample spacing (microseconds).
    probe_freq_mhz : float
        Probe frequency (MHz). Sets the molecular-frequency axis.
    sideband : Sideband
        Sideband convention.
    edge_guard_bins : int
        Keep injected lines at least this many bins away from the
        spectrum edges so the projection sub-window never clips.
    min_line_separation_bins : float
        Minimum separation between injected lines, in bin units.
        Keeps the ground-truth set unambiguous on close-pair edges.
    rng_seed : int
        Deterministic seed.
    n_strong_lines : int, default 0
        Number of *strong* lines to inject in addition to the
        ``n_lines`` weak ones, for the sidelobe-FP study. Strong lines
        are placed at least ``strong_separation_bins`` from any weak
        line and from each other, so the upstream detector picks up
        their main peak unambiguously plus any noise-bumps that ride
        on their skirts (the sidelobe-FP class). Default 0 disables
        the strong-line extension (backward-compatible with the
        original sweep).
    strong_snr : float, default 50
        Per-bin on-line SNR of each strong line, in the same
        convention as ``true_snr``. A strong line's skirt magnitude
        at offset Δ_bins ≈ ``true_snr * (FWHM/2) / Δ_bins`` (Lorentzian
        falloff), so SNR=50 strong lines produce above-threshold
        skirt bumps for ~10-20 bins around the line.
    strong_separation_bins : float, default 30
        Minimum bin distance between a strong line and any weak line
        / other strong line. Should be larger than
        ``min_line_separation_bins`` because strong-line skirts extend
        much further than weak-line ones.

    Returns
    -------
    SyntheticActiveFT
        Carrier of the realised spectrum + truth. The ``truth_freqs_mhz``
        / ``truth_snrs`` fields cover the weak-line set only; the
        ``strong_freqs_mhz`` / ``strong_snrs`` fields cover the
        strong-line set separately so the candidate classifier can
        flag sidelobe candidates vs noise candidates.
    """
    rng = np.random.default_rng(rng_seed)
    T_active = n_active * sample_dt_us
    bin_mhz = 1.0 / T_active
    tau_truth = T_active / (np.pi * float(fwhm_bins))

    # Pick injected line bin positions, respecting edge guard + separation.
    # Strong lines are placed first (they need a larger keep-out radius)
    # so the weak-line placement respects their exclusion zones.
    valid_bins = np.arange(edge_guard_bins, n_active // 2 - edge_guard_bins)
    strong_chosen: list[int] = []
    attempts = 0
    while len(strong_chosen) < n_strong_lines and attempts < 200 * max(
        n_strong_lines, 1
    ):
        attempts += 1
        b = int(rng.choice(valid_bins))
        if any(abs(b - c) < strong_separation_bins for c in strong_chosen):
            continue
        strong_chosen.append(b)
    if len(strong_chosen) < n_strong_lines:
        raise RuntimeError(
            f"could not place {n_strong_lines} strong lines with separation "
            f"{strong_separation_bins} bins on a {n_active}-sample grid"
        )

    chosen: list[int] = []
    attempts = 0
    while len(chosen) < n_lines and attempts < 200 * n_lines:
        attempts += 1
        b = int(rng.choice(valid_bins))
        if any(abs(b - c) < min_line_separation_bins for c in chosen):
            continue
        if any(abs(b - c) < strong_separation_bins for c in strong_chosen):
            continue
        chosen.append(b)
    if len(chosen) < n_lines:
        raise RuntimeError(
            f"could not place {n_lines} lines with separation "
            f"{min_line_separation_bins} bins on a {n_active}-sample grid"
        )
    chosen.sort()
    strong_chosen.sort()
    line_bins = np.array(chosen, dtype=int)
    strong_bins = np.array(strong_chosen, dtype=int)

    # Baseband freq for each line: at bin k the baseband freq is k/T_active.
    f_bb_lines = line_bins.astype(float) / T_active
    f_bb_strong = strong_bins.astype(float) / T_active

    # Build the FID: real damped cosines. The weak lines all have unit
    # amplitude; the strong lines are scaled so that *after* the σ_time
    # is set to give weak lines SNR = ``true_snr``, the strong lines'
    # on-line SNR comes out to ``strong_snr``. Since both classes share
    # the same τ_truth (and hence the same on-line h_T(0)), the strong
    # amplitude scales linearly: A_strong / A_weak = strong_snr / true_snr.
    t = np.arange(n_active) * sample_dt_us
    fid = np.zeros(n_active, dtype=float)
    weak_amp = 1.0
    strong_amp = (strong_snr / true_snr) * weak_amp if n_strong_lines else 0.0
    weak_phases = rng.uniform(0, 2 * np.pi, size=n_lines)
    strong_phases = rng.uniform(0, 2 * np.pi, size=n_strong_lines)
    for i in range(n_lines):
        fid += weak_amp * np.cos(
            2 * np.pi * f_bb_lines[i] * t + weak_phases[i]
        ) * np.exp(-t / tau_truth)
    for i in range(n_strong_lines):
        fid += strong_amp * np.cos(
            2 * np.pi * f_bb_strong[i] * t + strong_phases[i]
        ) * np.exp(-t / tau_truth)

    # Pre-compute the *noise-free* active-FT to derive the per-line on-line
    # magnitude, then size the time-domain noise so the realised on-line
    # SNR matches ``true_snr`` against the Stage 2 adaptive estimator.
    # Use only weak-line bins for the calibration (the strong lines'
    # on-line magnitude is much larger; we want the weak lines to hit
    # ``true_snr`` while the strong lines come out at ``strong_snr``).
    spec_signal = sample_dt_us * np.fft.rfft(fid)
    on_line_mag = float(np.median(np.abs(spec_signal[line_bins])))

    # Active-FT |X| RMS noise scales as σ_time * sqrt(T · dt) (matched-filter
    # convention). Set σ_time so |X|-RMS = on_line_mag / true_snr.
    target_sigma_x = on_line_mag / true_snr
    sigma_time = target_sigma_x / np.sqrt(T_active * sample_dt_us)
    noise_t = rng.normal(scale=sigma_time, size=n_active)
    fid_noisy = fid + noise_t

    # Active-FT of the noisy FID.
    active = sample_dt_us * np.fft.rfft(fid_noisy)
    # Molecular frequency axis.
    s = -1.0 if sideband == Sideband.LOWER else 1.0
    f_bb = np.fft.rfftfreq(n_active, d=sample_dt_us)
    freq_mhz = probe_freq_mhz + s * f_bb
    truth_freqs_mhz = probe_freq_mhz + s * f_bb_lines
    strong_freqs_mhz = probe_freq_mhz + s * f_bb_strong

    # Sort ascending for analysis convenience.
    sort_idx = np.argsort(freq_mhz)
    freq_sorted = np.ascontiguousarray(freq_mhz[sort_idx])
    spec_sorted = np.ascontiguousarray(active[sort_idx])
    # Per-bin |X| RMS is uniform across the synthetic spectrum: white
    # time-domain noise gives constant per-bin variance in the active-FT.
    # ``target_sigma_x`` is exactly that constant (set above). Skip the
    # adaptive Stage-2 estimator here -- it's calibrated for the
    # production grid (millions of bins) and degrades when the per-bin
    # smoothing window exceeds the spectrum length. Using the analytic
    # σ also isolates the screen's behaviour from upstream noise-estimator
    # quirks, which is what the study is supposed to characterise.
    sigma_sorted = np.full(freq_sorted.shape, target_sigma_x, dtype=float)
    # Hand back the sorted view (ascending molecular freq) since
    # ``project_candidates`` re-sorts internally anyway and the analysis
    # downstream is easier on monotone grids.
    return SyntheticActiveFT(
        freq_mhz=freq_sorted,
        spectrum=spec_sorted,
        sigma=sigma_sorted,
        truth_freqs_mhz=np.sort(truth_freqs_mhz),
        truth_snrs=np.full(n_lines, true_snr),
        tau_truth_us=float(tau_truth),
        acquisition_us=float(T_active),
        bin_mhz=float(bin_mhz),
        fwhm_bins=float(fwhm_bins),
        strong_freqs_mhz=np.sort(strong_freqs_mhz)
        if n_strong_lines
        else np.array([], dtype=float),
        strong_snrs=np.full(n_strong_lines, strong_snr)
        if n_strong_lines
        else np.array([], dtype=float),
    )


# ===========================================================================
# Candidate generation + classification
# ===========================================================================
@dataclass(frozen=True)
class CandidateRow:
    freq_mhz: float
    bin: int
    detected_snr: float
    ratio: float
    coherent_snr: float
    is_true_positive: bool
    nearest_truth_separation_bins: float
    # Sidelobe-FP extension. ``in_strong_skirt`` is True when the
    # candidate is NOT a true positive AND lies within ``sidelobe_range``
    # bins of an injected strong line's centre. Pure-noise FPs have
    # ``in_strong_skirt = False``.
    in_strong_skirt: bool = False
    nearest_strong_separation_bins: float = float("inf")


def candidates_with_classification(
    sim: SyntheticActiveFT,
    *,
    tau_basis_us: float | None,
    detect_min_snr: float = 2.0,
    truth_match_bins: int | None = None,
    sidelobe_range_bins: float = 30.0,
) -> list[CandidateRow]:
    """Run a magnitude-prominence detector + the projection screen.

    The detector is a single-pass ``scipy.signal.find_peaks`` on the
    active-FT magnitude with height = ``detect_min_snr * σ_c`` (Rayleigh
    scale) -- the simplest possible peak-detection stand-in.
    Classification: a candidate is a true positive iff some injected
    line lies within ``truth_match_bins`` of its bin. When
    ``truth_match_bins`` is None it defaults to
    ``max(1, round(fwhm_bins / 2))`` so wide-FWHM lines (whose detected
    local max can sit several bins off the true centre under noise)
    are not spuriously labelled FP.

    ``tau_basis_us = None`` triggers the matched basis
    ``tau_basis = tau_truth``; otherwise the supplied τ is used (the
    mismatch-sensitivity sweep passes 0.5×τ_truth and 2×τ_truth).
    """
    if truth_match_bins is None:
        truth_match_bins = max(1, int(round(sim.fwhm_bins / 2.0)))
    mag = np.abs(sim.spectrum)
    sigma_c = sim.sigma / np.sqrt(2.0)
    # Median over the noise floor is a stable height anchor.
    median_sigma = float(np.median(sim.sigma[sim.sigma > 0.0]))
    height = detect_min_snr * (median_sigma / np.sqrt(2.0))
    bins, _ = find_peaks(mag, height=height, distance=1)

    # The simulator returns an ascending freq grid; the candidate bins are
    # along that grid.
    cand_freqs = sim.freq_mhz[bins].tolist()
    tau_used = tau_basis_us if tau_basis_us is not None else sim.tau_truth_us
    projections = project_candidates(
        sim.freq_mhz,
        sim.spectrum,
        sim.sigma,
        cand_freqs,
        tau_us=tau_used,
        acquisition_us=sim.acquisition_us,
        sideband=Sideband.LOWER,  # the simulator's convention
    )

    # Build the truth-bin sets on the ascending grid. Weak lines drive
    # the TP label; strong lines (if any) also count as TPs at their
    # centres and drive the sidelobe-FP flag in their skirts.
    truth_bins = np.searchsorted(sim.freq_mhz, sim.truth_freqs_mhz)
    truth_bins = np.clip(truth_bins, 0, sim.freq_mhz.size - 1)
    if sim.strong_freqs_mhz.size:
        strong_bins = np.searchsorted(sim.freq_mhz, sim.strong_freqs_mhz)
        strong_bins = np.clip(strong_bins, 0, sim.freq_mhz.size - 1)
    else:
        strong_bins = np.array([], dtype=int)

    rows: list[CandidateRow] = []
    for cand_bin, pr in zip(bins, projections):
        seps = np.abs(truth_bins - int(cand_bin))
        nearest_sep = int(seps.min()) if seps.size else sim.freq_mhz.size
        if strong_bins.size:
            strong_seps = np.abs(strong_bins - int(cand_bin))
            nearest_strong = int(strong_seps.min())
        else:
            nearest_strong = sim.freq_mhz.size
        is_tp_weak = nearest_sep <= truth_match_bins
        is_tp_strong = (
            strong_bins.size > 0 and nearest_strong <= truth_match_bins
        )
        is_tp = bool(is_tp_weak or is_tp_strong)
        in_skirt = (
            (not is_tp)
            and strong_bins.size > 0
            and nearest_strong <= sidelobe_range_bins
        )

        bin_sigma_c = float(sigma_c[cand_bin])
        det_snr = (
            float(mag[cand_bin]) / bin_sigma_c if bin_sigma_c > 0 else 0.0
        )
        rows.append(
            CandidateRow(
                freq_mhz=float(sim.freq_mhz[cand_bin]),
                bin=int(cand_bin),
                detected_snr=det_snr,
                ratio=pr.ratio,
                coherent_snr=pr.coherent_snr,
                is_true_positive=is_tp,
                nearest_truth_separation_bins=float(nearest_sep),
                in_strong_skirt=bool(in_skirt),
                nearest_strong_separation_bins=float(nearest_strong),
            )
        )
    return rows


# ===========================================================================
# Section 3: phase-space sweep
# ===========================================================================
@dataclass
class CellResult:
    snr: float
    fwhm_bins: float
    n_candidates: int
    n_tp: int
    n_fp: int
    auc: float
    fpr_at_tpr_95: float
    median_ratio_tp: float
    median_ratio_fp: float
    rows: list[CandidateRow]
    # Detection-yield diagnostics:
    n_injected_total: int
    n_injected_recovered: int  # truths with >=1 candidate within ±truth_match_bins
    detection_recall: float    # n_injected_recovered / n_injected_total
    fp_per_line: float         # n_fp / n_injected_recovered (or NaN)


def _auc(ratios: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    if labels.sum() == 0 or (~labels).sum() == 0:
        return float("nan"), float("nan")
    order = np.argsort(-ratios)  # high ratio first (most-like-Lorentzian first)
    sorted_labels = labels[order].astype(int)
    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    tpr = tp / labels.sum()
    fpr = fp / (~labels).sum()
    auc = float(np.trapezoid(tpr, fpr))
    elig = np.where(tpr >= 0.95)[0]
    fpr_95 = float(fpr[elig[0]]) if elig.size else float("nan")
    return auc, fpr_95


def run_cell(
    *,
    snr: float,
    fwhm_bins: float,
    n_trials: int = 4,
    n_lines: int = 25,
    n_active: int = SIM_N_ACTIVE,
    rng_seed: int = RNG_SEED,
    tau_basis_factor: float = 1.0,
    detect_min_snr: float = 2.0,
) -> CellResult:
    """Aggregate ``n_trials`` simulator realisations into one cell."""
    rows: list[CandidateRow] = []
    n_injected_total = 0
    n_injected_recovered = 0
    for k in range(n_trials):
        sim = simulate_active_ft(
            fwhm_bins=fwhm_bins,
            true_snr=snr,
            n_lines=n_lines,
            n_active=n_active,
            rng_seed=rng_seed + k,
        )
        tau_basis = (
            sim.tau_truth_us * tau_basis_factor
            if tau_basis_factor != 1.0
            else None  # None → matched τ
        )
        trial_rows = candidates_with_classification(
            sim,
            tau_basis_us=tau_basis,
            detect_min_snr=detect_min_snr,
        )
        rows.extend(trial_rows)
        # Per-trial detection yield: how many of the injected lines had
        # at least one candidate within the truth-match tolerance.
        truth_match = max(1, int(round(sim.fwhm_bins / 2.0)))
        # Translate truth freqs to ascending-grid bins (sim already
        # sorts ascending).
        truth_bins = np.searchsorted(sim.freq_mhz, sim.truth_freqs_mhz)
        truth_bins = np.clip(truth_bins, 0, sim.freq_mhz.size - 1)
        cand_bins = np.array([r.bin for r in trial_rows])
        n_injected_total += sim.truth_freqs_mhz.size
        if cand_bins.size:
            for tb in truth_bins:
                if np.any(np.abs(cand_bins - tb) <= truth_match):
                    n_injected_recovered += 1
    ratios = np.array([r.ratio for r in rows])
    labels = np.array([r.is_true_positive for r in rows])
    auc, fpr95 = _auc(ratios, labels)
    n_tp = int(labels.sum())
    n_fp = int((~labels).sum())
    detection_recall = (
        n_injected_recovered / n_injected_total if n_injected_total else float("nan")
    )
    fp_per_line = (
        n_fp / n_injected_recovered if n_injected_recovered else float("nan")
    )
    return CellResult(
        snr=snr,
        fwhm_bins=fwhm_bins,
        n_candidates=len(rows),
        n_tp=n_tp,
        n_fp=n_fp,
        auc=auc,
        fpr_at_tpr_95=fpr95,
        median_ratio_tp=float(np.median(ratios[labels])) if n_tp else float("nan"),
        median_ratio_fp=float(np.median(ratios[~labels])) if n_fp else float("nan"),
        rows=rows,
        n_injected_total=n_injected_total,
        n_injected_recovered=n_injected_recovered,
        detection_recall=detection_recall,
        fp_per_line=fp_per_line,
    )


# ===========================================================================
# Section 2: single-cell illustration
# ===========================================================================
def figure_single_cell(snr: float = 4.0) -> None:
    """Three FWHM/bin regimes at one SNR, side-by-side panels."""
    fwhms = [0.8, 1.5, 3.0]
    fig, axes = plt.subplots(
        2, 3, figsize=(15, 8), gridspec_kw={"height_ratios": [2, 1]}
    )
    for col, fwhm in enumerate(fwhms):
        sim = simulate_active_ft(
            fwhm_bins=fwhm, true_snr=snr, n_lines=12, rng_seed=RNG_SEED + col
        )
        ax = axes[0, col]
        ax.plot(sim.freq_mhz, np.abs(sim.spectrum), color="C0", lw=0.7)
        for f in sim.truth_freqs_mhz:
            ax.axvline(f, color="C2", lw=0.6, alpha=0.6, ls="--")
        ax.set_title(
            f"FWHM = {fwhm:.1f} bins,  injected SNR = {snr:.1f}\n"
            f"τ_truth = {sim.tau_truth_us:.2f} µs"
        )
        ax.set_ylabel("|active-FT|")
        ax.set_xlabel("freq (MHz)")

        rows = candidates_with_classification(sim, tau_basis_us=None)
        ratios_tp = np.array([r.ratio for r in rows if r.is_true_positive])
        ratios_fp = np.array([r.ratio for r in rows if not r.is_true_positive])
        ax2 = axes[1, col]
        bins_h = np.linspace(0.0, max(2.5, np.max([*ratios_tp, *ratios_fp, 1.0]) + 0.2), 30)
        if ratios_fp.size:
            ax2.hist(ratios_fp, bins=bins_h, alpha=0.5, color="C3", label="noise")
        if ratios_tp.size:
            ax2.hist(ratios_tp, bins=bins_h, alpha=0.5, color="C0", label="real")
        ax2.set_xlabel("projection ratio")
        ax2.set_ylabel("count")
        ax2.legend(loc="upper right", fontsize=9)
        ax2.axvline(1.0, color="k", ls=":", alpha=0.5)

    fig.suptitle(
        "Single-cell illustration: how the projection ratio behaves as "
        "linewidth in bin units grows"
    )
    fig.tight_layout()
    out = FIG / "01_single_cell.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", out)


# ===========================================================================
# Section 3 main: AUC heatmap + ROC curves
# ===========================================================================
SNR_GRID = np.array([2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 25.0, 50.0])
FWHM_GRID = np.array([0.5, 0.8, 1.0, 1.34, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0])


def run_sweep(
    *, n_trials: int = 4, tau_basis_factor: float = 1.0, label: str
) -> np.ndarray:
    """Run the phase-space sweep and return a structured ``CellResult`` grid."""
    grid = np.empty((SNR_GRID.size, FWHM_GRID.size), dtype=object)
    t0 = time.time()
    for i, snr in enumerate(SNR_GRID):
        for j, fwhm in enumerate(FWHM_GRID):
            cell = run_cell(
                snr=float(snr),
                fwhm_bins=float(fwhm),
                n_trials=n_trials,
                tau_basis_factor=tau_basis_factor,
            )
            grid[i, j] = cell
        logger.info(
            "[%s] row snr=%.1f done in %.1fs (cumulative)",
            label, snr, time.time() - t0,
        )
    return grid


def _heatmap(
    grid: np.ndarray, attr: str, *, vmin: float, vmax: float, cmap: str, title: str
) -> np.ndarray:
    arr = np.array(
        [[getattr(c, attr) for c in row] for row in grid], dtype=float
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        arr,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        extent=(0, FWHM_GRID.size, 0, SNR_GRID.size),
    )
    ax.set_xticks(np.arange(FWHM_GRID.size) + 0.5)
    ax.set_xticklabels([f"{x:.1f}" for x in FWHM_GRID])
    ax.set_yticks(np.arange(SNR_GRID.size) + 0.5)
    ax.set_yticklabels([f"{x:.1f}" for x in SNR_GRID])
    ax.set_xlabel("FWHM (bins)")
    ax.set_ylabel("injected SNR")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if v < (vmin + vmax) / 2 else "black",
                )
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=attr)
    return arr, fig


def figure_phase_space_matched() -> np.ndarray:
    grid = run_sweep(n_trials=6, tau_basis_factor=1.0, label="matched")
    auc_arr, fig = _heatmap(
        grid,
        "auc",
        vmin=0.4,
        vmax=1.0,
        cmap="viridis",
        title="Phase-space AUC -- matched τ_basis (n=6 trials per cell)",
    )
    out = FIG / "02_phase_space_auc_matched.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", out)
    fpr_arr, fig2 = _heatmap(
        grid,
        "fpr_at_tpr_95",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis_r",
        title="Phase-space FPR @ TPR≥0.95 -- matched τ_basis",
    )
    out2 = FIG / "02b_phase_space_fpr95_matched.png"
    fig2.tight_layout()
    fig2.savefig(out2, dpi=120)
    plt.close(fig2)
    logger.info("wrote %s", out2)
    np.savez(
        DATA / "phase_space_matched.npz",
        snr_grid=SNR_GRID,
        fwhm_grid=FWHM_GRID,
        auc=auc_arr,
        fpr95=fpr_arr,
    )
    return grid


def figure_detection_yield(grid: np.ndarray) -> None:
    """Heatmaps of detection recall and FP-per-line across the grid.

    Shows where the upstream detector stops seeing real lines (per-bin
    SNR collapses as line energy spreads over many bins) and where it
    is flooded with noise candidates. Pairs with the AUC heatmap to
    make the wide-FWHM collapse interpretable.
    """
    recall = np.array(
        [[c.detection_recall for c in row] for row in grid], dtype=float
    )
    fp_per_line = np.array(
        [[c.fp_per_line for c in row] for row in grid], dtype=float
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, arr, title, vmin, vmax, cmap in zip(
        axes,
        [recall, fp_per_line],
        ["Detection recall (n_recovered / n_injected)",
         "FP candidates per recovered line"],
        [0.0, 0.0],
        [1.0, max(20.0, float(np.nanmax(fp_per_line)) if np.isfinite(fp_per_line).any() else 1.0)],
        ["viridis", "inferno"],
    ):
        im = ax.imshow(
            arr,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            extent=(0, FWHM_GRID.size, 0, SNR_GRID.size),
        )
        ax.set_xticks(np.arange(FWHM_GRID.size) + 0.5)
        ax.set_xticklabels([f"{x:.1f}" for x in FWHM_GRID], rotation=0)
        ax.set_yticks(np.arange(SNR_GRID.size) + 0.5)
        ax.set_yticklabels([f"{x:.1f}" for x in SNR_GRID])
        ax.set_xlabel("FWHM (bins)")
        ax.set_ylabel("injected SNR")
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                if np.isfinite(v):
                    label = f"{v:.2f}" if title.startswith("Detection") else f"{v:.1f}"
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        label,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if v < (vmin + vmax) / 2 else "black",
                    )
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    fig.suptitle("Upstream detector yield (matched τ_basis sweep)")
    fig.tight_layout()
    out = FIG / "06_detection_yield.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", out)


def figure_roc_selected_cells(grid: np.ndarray) -> None:
    """ROC curves for four diagnostic cells: low/high SNR × narrow/wide FWHM."""
    picks = [(2.0, 0.8), (2.0, 3.0), (10.0, 0.8), (10.0, 3.0)]
    fig, ax = plt.subplots(figsize=(7, 7))
    for (snr_pick, fwhm_pick) in picks:
        i = int(np.argmin(np.abs(SNR_GRID - snr_pick)))
        j = int(np.argmin(np.abs(FWHM_GRID - fwhm_pick)))
        cell = grid[i, j]
        if cell.n_tp == 0 or cell.n_fp == 0:
            continue
        ratios = np.array([r.ratio for r in cell.rows])
        labels = np.array([r.is_true_positive for r in cell.rows])
        order = np.argsort(-ratios)
        tp = np.cumsum(labels[order])
        fp = np.cumsum(~labels[order])
        tpr = tp / labels.sum()
        fpr = fp / (~labels).sum()
        ax.plot(
            fpr, tpr,
            label=f"SNR={snr_pick:.0f}, FWHM={fwhm_pick:.1f} bins (AUC={cell.auc:.2f})",
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="random")
    ax.set_xlabel("FPR (kept noise / total noise)")
    ax.set_ylabel("TPR (kept real / total real)")
    ax.set_title("ROC: keep candidates with ratio ≥ threshold (matched τ_basis)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    out = FIG / "03_roc_selected_cells.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", out)


# ===========================================================================
# Section 4: τ_basis mismatch sensitivity
# ===========================================================================
def _per_class_auc(rows: list[CandidateRow]) -> dict[str, float]:
    """Compute AUC against three FP populations: all-FP, noise-only-FP,
    sidelobe-only-FP. Returns dict of AUCs.

    The TP population is the same in all three (candidates within
    truth-match of a real line, weak or strong). FPs are partitioned by
    ``in_strong_skirt``.
    """
    ratios = np.array([r.ratio for r in rows])
    is_tp = np.array([r.is_true_positive for r in rows])
    in_skirt = np.array([r.in_strong_skirt for r in rows])

    def _auc_for_mask(tp_mask: np.ndarray, fp_mask: np.ndarray) -> float:
        if tp_mask.sum() == 0 or fp_mask.sum() == 0:
            return float("nan")
        tp_r = ratios[tp_mask]
        fp_r = ratios[fp_mask]
        # Mann-Whitney U → AUC, "higher ratio = more like a Lorentzian".
        from scipy.stats import mannwhitneyu
        u, _ = mannwhitneyu(tp_r, fp_r, alternative="greater")
        return float(u / (tp_r.size * fp_r.size))

    is_fp = ~is_tp
    return {
        "auc_all": _auc_for_mask(is_tp, is_fp),
        "auc_noise_only": _auc_for_mask(is_tp, is_fp & ~in_skirt),
        "auc_sidelobe_only": _auc_for_mask(is_tp, is_fp & in_skirt),
    }


def figure_sidelobe_breakdown() -> None:
    """Compare matched vs narrower τ_basis against noise-FPs and sidelobe-FPs.

    Runs the simulator with strong lines injected to produce a
    structured-noise FP population, then computes per-class AUC for
    several τ_basis settings. Tests whether the 2638 finding (matched
    has more dynamic range on real data, suggesting structured-noise
    contamination matters) replicates under controlled simulation.
    """
    fwhm_bins = 1.34   # the 2638-like bin-matched regime
    weak_snr = 4.0     # detection-threshold-ish
    n_lines = 25
    n_strong_lines = 4
    strong_snr_values = [25.0, 50.0, 100.0]
    factors = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
    n_trials = 6

    fig, axes = plt.subplots(
        1, len(strong_snr_values), figsize=(15, 5), sharey=True
    )

    summary_rows: list[str] = []
    for ax, strong_snr in zip(axes, strong_snr_values):
        aucs_noise: list[float] = []
        aucs_sidelobe: list[float] = []
        aucs_all: list[float] = []
        per_trial_skirt_counts: list[int] = []
        per_trial_noise_counts: list[int] = []
        for factor in factors:
            all_rows: list[CandidateRow] = []
            for k in range(n_trials):
                sim = simulate_active_ft(
                    fwhm_bins=fwhm_bins,
                    true_snr=weak_snr,
                    n_lines=n_lines,
                    n_strong_lines=n_strong_lines,
                    strong_snr=strong_snr,
                    strong_separation_bins=80.0,
                    rng_seed=RNG_SEED + k,
                )
                tau_basis = (
                    sim.tau_truth_us * factor if factor != 1.0 else None
                )
                all_rows.extend(
                    candidates_with_classification(
                        sim,
                        tau_basis_us=tau_basis,
                        sidelobe_range_bins=80.0,
                    )
                )
            stats = _per_class_auc(all_rows)
            aucs_all.append(stats["auc_all"])
            aucs_noise.append(stats["auc_noise_only"])
            aucs_sidelobe.append(stats["auc_sidelobe_only"])
            n_skirt = sum(1 for r in all_rows if r.in_strong_skirt)
            n_noise = sum(
                1 for r in all_rows if not r.is_true_positive and not r.in_strong_skirt
            )
            per_trial_skirt_counts.append(n_skirt)
            per_trial_noise_counts.append(n_noise)

        ax.plot(factors, aucs_noise, marker="o", label="AUC vs noise-only FPs", color="C0")
        ax.plot(factors, aucs_sidelobe, marker="s", label="AUC vs sidelobe FPs", color="C3")
        ax.plot(factors, aucs_all, marker="^", label="AUC vs all FPs", color="k", alpha=0.5)
        ax.axvline(1.0, color="gray", ls=":", alpha=0.5)
        ax.set_xlabel("τ_basis / τ_truth")
        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(
            f"strong_snr={strong_snr:.0f}\n"
            f"n_sidelobe_FP={per_trial_skirt_counts[0]}, "
            f"n_noise_FP={per_trial_noise_counts[0]}"
        )
        axes[0].set_ylabel("AUC (real lines vs FP class)")
        ax.legend(loc="lower right", fontsize=9)
        summary_rows.append(
            f"strong_snr={strong_snr:.0f}: "
            + " | ".join(
                f"{factor:.1f}: noise={n:.2f} side={s:.2f}"
                for factor, n, s in zip(factors, aucs_noise, aucs_sidelobe)
            )
        )

    fig.suptitle(
        "Structured-FP AUC: matched vs narrower τ_basis on simulated spectra "
        f"with strong-line sidelobes (weak SNR={weak_snr}, FWHM/bin={fwhm_bins})"
    )
    fig.tight_layout()
    out = FIG / "07_sidelobe_breakdown.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", out)
    for line in summary_rows:
        logger.info(line)


def figure_sidelobe_interference() -> None:
    """Sidelobe interference: do overlapping strong-line skirts break the screen?

    Sweeps ``strong_separation_bins ∈ {80, 40, 20}`` while holding
    ``strong_snr = 50`` and a fixed number of strong lines (4),
    weak SNR = 4, FWHM/bin = 1.34. At ``strong_separation = 80`` each
    strong line's skirt is essentially isolated; at 40 the skirts
    overlap substantially; at 20 a candidate in the middle "sees"
    contributions from two strong lines whose phases compete.

    The recommended ``τ_basis = 2 × τ_truth`` setting should hold up
    because its sub-window is small enough that the *local* phase
    profile (a few bins around the candidate) is what the screen
    tests, and interference produces non-Lorentzian local phase the
    screen flags. This figure confirms or refutes that.
    """
    fwhm_bins = 1.34
    weak_snr = 4.0
    strong_snr = 50.0
    n_lines = 25
    n_strong_lines = 4
    n_trials = 6
    separations = [80.0, 40.0, 20.0]
    factors = np.array([1.0, 1.5, 2.0, 3.0])

    fig, axes = plt.subplots(
        1, len(separations), figsize=(15, 5), sharey=True
    )
    summary: list[str] = []
    for ax, sep in zip(axes, separations):
        # Sidelobe-range tracking the actual sidelobe extent. With
        # ``strong_separation_bins = 20`` two strong lines' skirts
        # genuinely overlap if we set the sidelobe label range to
        # the same 80-bin span, so a candidate between two close
        # strong lines is labelled sidelobe-FP w.r.t. both -- which
        # is exactly the interference regime we want to test.
        sidelobe_range = 80.0
        aucs_noise: list[float] = []
        aucs_sidelobe: list[float] = []
        skirt_counts: list[int] = []
        for factor in factors:
            all_rows: list[CandidateRow] = []
            for k in range(n_trials):
                sim = simulate_active_ft(
                    fwhm_bins=fwhm_bins,
                    true_snr=weak_snr,
                    n_lines=n_lines,
                    n_strong_lines=n_strong_lines,
                    strong_snr=strong_snr,
                    strong_separation_bins=sep,
                    rng_seed=RNG_SEED + 1000 + k,
                )
                tau_basis = (
                    sim.tau_truth_us * factor if factor != 1.0 else None
                )
                all_rows.extend(
                    candidates_with_classification(
                        sim,
                        tau_basis_us=tau_basis,
                        sidelobe_range_bins=sidelobe_range,
                    )
                )
            stats = _per_class_auc(all_rows)
            aucs_noise.append(stats["auc_noise_only"])
            aucs_sidelobe.append(stats["auc_sidelobe_only"])
            n_skirt = sum(1 for r in all_rows if r.in_strong_skirt)
            skirt_counts.append(n_skirt)

        ax.plot(factors, aucs_noise, marker="o", label="AUC vs noise FPs", color="C0")
        ax.plot(factors, aucs_sidelobe, marker="s", label="AUC vs sidelobe FPs", color="C3")
        ax.axvline(2.0, color="gray", ls=":", alpha=0.5, label="recommended 2×")
        ax.set_xlabel("τ_basis / τ_truth")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(
            f"strong_separation={sep:.0f} bins\n"
            f"(sidelobe FP count = {skirt_counts[0]})"
        )
        axes[0].set_ylabel("AUC (real lines vs FP class)")
        ax.legend(loc="lower right", fontsize=9)
        summary.append(
            f"sep={sep:.0f}: "
            + " | ".join(
                f"factor={f:.1f}: noise={n:.2f} side={s:.2f}"
                for f, n, s in zip(factors, aucs_noise, aucs_sidelobe)
            )
        )

    fig.suptitle(
        "Sidelobe interference test: AUC as strong-line separation shrinks "
        "(strong_snr=50, weak SNR=4, FWHM/bin=1.34)"
    )
    fig.tight_layout()
    out = FIG / "08_sidelobe_interference.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", out)
    for line in summary:
        logger.info(line)


def figure_tau_basis_optimum() -> None:
    """Fine-grained τ_basis sweep at a few diagnostic cells.

    Picks (SNR, FWHM) cells covering the matched-τ AUC range and walks
    ``τ_basis = factor · τ_truth`` over a wide grid (factor 0.25 → 8.0)
    to locate the empirical optimum. The matched-τ default is one
    operating point; production τ_basis should be set from the curve.
    """
    cells = [(2.0, 1.0), (3.0, 1.34), (5.0, 1.0), (10.0, 1.0), (25.0, 1.0)]
    factors = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
    fig, ax = plt.subplots(figsize=(9, 6))
    for snr, fwhm in cells:
        aucs: list[float] = []
        for f in factors:
            cell = run_cell(
                snr=snr,
                fwhm_bins=fwhm,
                n_trials=4,
                tau_basis_factor=float(f),
            )
            aucs.append(cell.auc)
        ax.plot(factors, aucs, marker="o", label=f"SNR={snr:.0f}, FWHM={fwhm:.2f}")
    ax.axvline(1.0, color="k", ls=":", alpha=0.5, label="matched τ")
    ax.set_xlabel("τ_basis / τ_truth")
    ax.set_ylabel("AUC")
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("τ_basis optimum: AUC vs (τ_basis / τ_truth)")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    out = FIG / "05_tau_basis_optimum.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", out)


def figure_tau_mismatch() -> None:
    factors = {"τ_basis = 0.5×τ_truth": 0.5, "τ_basis = 2×τ_truth": 2.0}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (label, factor) in zip(axes, factors.items()):
        grid = run_sweep(n_trials=4, tau_basis_factor=factor, label=label)
        arr = np.array(
            [[c.auc for c in row] for row in grid], dtype=float
        )
        im = ax.imshow(
            arr,
            origin="lower",
            cmap="viridis",
            vmin=0.4,
            vmax=1.0,
            aspect="auto",
            extent=(0, FWHM_GRID.size, 0, SNR_GRID.size),
        )
        ax.set_xticks(np.arange(FWHM_GRID.size) + 0.5)
        ax.set_xticklabels([f"{x:.1f}" for x in FWHM_GRID])
        ax.set_yticks(np.arange(SNR_GRID.size) + 0.5)
        ax.set_yticklabels([f"{x:.1f}" for x in SNR_GRID])
        ax.set_xlabel("FWHM (bins)")
        ax.set_ylabel("injected SNR")
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                if np.isfinite(v):
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if v < 0.7 else "black",
                    )
        ax.set_title(label)
        np.savez(
            DATA / f"phase_space_factor_{factor:.1f}.npz",
            snr_grid=SNR_GRID,
            fwhm_grid=FWHM_GRID,
            auc=arr,
        )
    fig.suptitle(
        "AUC under τ_basis mismatch -- compare to matched in 02_phase_space_auc_matched.png"
    )
    fig.colorbar(im, ax=axes, label="AUC")
    out = FIG / "04_tau_mismatch_auc.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", out)


# ===========================================================================
# Driver
# ===========================================================================
def main() -> None:
    t0 = time.time()
    logger.info("section 2: single-cell illustration")
    figure_single_cell()
    logger.info("section 3: phase-space sweep (matched τ_basis)")
    grid = figure_phase_space_matched()
    figure_detection_yield(grid)
    figure_roc_selected_cells(grid)
    logger.info("section 4: τ_basis mismatch sensitivity")
    figure_tau_mismatch()
    logger.info("section 5: τ_basis fine sweep")
    figure_tau_basis_optimum()
    logger.info("section 6: sidelobe-FP breakdown")
    figure_sidelobe_breakdown()
    logger.info("section 7: sidelobe interference")
    figure_sidelobe_interference()
    logger.info("done in %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()
