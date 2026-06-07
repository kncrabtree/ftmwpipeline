"""
Matched-filter peak detection -- reproducibility script.

Regenerates every figure under ``figures/`` and the empirical numbers
cited in ``report.md``. From the repository root with the project
conda env:

    conda run -n ftmwpipeline-dev python \\
        dev-docs/research/matched-filter-detection/prototype.py

The 2638 sections require ``scratch/stage5-validation/exp_2638.ftmw``.
To build it canonically::

    python - <<'EOF'
    import ftmwpipeline.api as ftmw
    P = "scratch/stage5-validation/exp_2638.ftmw"
    ftmw.import_data(P, source="examples/blackchirp_data/2638", force=True)
    ftmw.detect_start_time(P, band=(26500, 40000), stamp=True)
    ftmw.compute_ft(P, trim=(26500, 40000))
    ftmw.estimate_noise(P)
    EOF

The study tests whether replacing the peak-detection stage's primary
pass (Savitzky-Golay second-derivative locator on a Blackman-Harris-
apodized spectrum) with a Lorentzian matched-filter detector
(exponentially-apodized FFT with per-bin SNR thresholding) followed
by the σ-weighted Lorentzian-projection screen is a net win on
synthetic data and the 2638 fixture. See ``report.md`` for the full
narrative.

Sections:

1. **Reference implementation**: ``matched_filter_detect`` --
   exponentially-apodized active-FT, per-bin SNR thresholding, local
   maxima above threshold. The σ-weighted Lorentzian projection at
   ``f_c`` is, by Parseval, an apodized FT divided by per-bin σ; the
   matched filter under Gaussian noise is therefore an FFT, not N
   projections (one per candidate).

2. **Smoke test**: single synthetic cell at SNR=4, FWHM/bin=1.34.
   Verify the matched filter finds the injected lines, the
   production-like detector finds them too, and the candidate sets
   are similar in the easy regime.

3. **Phase-space sweep**: two detectors (production-like Sav-Gol on
   BH-apodized, matched filter on exp-apodized) across the
   ``(SNR, FWHM/bin)`` grid the screen study used, with and without
   the projection screen applied as a post-filter. Compares TP recall
   and FP counts (noise-only and sidelobe FPs separately).

4. **2638 application**: build the active-FT at the auto-calibrated
   τ_basis, run the matched filter, apply the screen, and compare
   against the production Stage 3 output and the persisted Stage 5
   fit.

5. **Architectural questions**: gap-pass redundancy and leakage-mask
   redundancy in the matched-filter design.

Outputs land in ``figures/`` (PNGs) and ``data/`` (.npz). The 2638
sections need ``scratch/stage5-validation/exp_2638.ftmw``; the rest
are synthetic and need no inputs.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, get_window

from ftmwpipeline.core.data_structures import Sideband
from ftmwpipeline.fitting.active_ft import ActiveFTResult, compute_active_ft
from ftmwpipeline.fitting.peak_model import h_T, sideband_sign
from ftmwpipeline.preprocessing.coherence_screen import project_candidates
from ftmwpipeline.preprocessing.noise_estimation import estimate_noise_scatter
from ftmwpipeline.preprocessing.peak_detection import locate_peaks
from ftmwpipeline.utils.signal_processing import matched_filter_window

# Reuse the screen study's simulator so improvements there propagate here.
# Load via importlib to avoid the "prototype" module-name collision.
HERE = Path(__file__).parent
import importlib.util as _ilu

_screen_spec = _ilu.spec_from_file_location(
    "_screen_prototype",
    HERE.parent / "stage3-coherence-screen" / "prototype.py",
)
assert _screen_spec is not None and _screen_spec.loader is not None
_screen_module = _ilu.module_from_spec(_screen_spec)
sys.modules["_screen_prototype"] = _screen_module
_screen_spec.loader.exec_module(_screen_module)
SyntheticActiveFT = _screen_module.SyntheticActiveFT  # type: ignore
simulate_active_ft = _screen_module.simulate_active_ft  # type: ignore

logger = logging.getLogger("matched-filter-research")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

FIG = HERE / "figures"
DATA = HERE / "data"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
REPO_ROOT = Path(__file__).resolve().parents[3]
FTMW_PATH = REPO_ROOT / "scratch" / "stage5-validation" / "exp_2638.ftmw"
RNG_SEED = 20260524

SIM_N_ACTIVE = 4096  # mirrors the screen study's simulator
SIM_SAMPLE_DT_US = 0.020
SIM_PROBE_MHZ = 40000.0
SIM_SIDEBAND = Sideband.LOWER


# ===========================================================================
# Section 1: reference matched-filter implementation
# ===========================================================================
@dataclass(frozen=True)
class MatchedFilterResult:
    """Output of ``matched_filter_detect``.

    Attributes
    ----------
    active_ft : ActiveFTResult
        The exponentially-apodized active-FT (active samples windowed by
        ``exp(-t/tau_basis_us)`` before the rfft).
    sigma : np.ndarray
        Per-bin |X| RMS noise on the active-FT magnitude, same shape
        as ``active_ft.freq_mhz``.
    per_bin_snr : np.ndarray
        ``|X| / σ_c`` with ``σ_c = σ / √2`` -- the per-bin Rayleigh-
        scale SNR. Above-threshold local maxima are candidates.
    candidate_bins : np.ndarray
        Integer indices into ``active_ft.freq_mhz`` of the kept candidates,
        sorted ascending.
    detection_snr : float
        Threshold used; candidates have ``per_bin_snr[bin] >= detection_snr``.
    """

    active_ft: ActiveFTResult
    sigma: np.ndarray
    per_bin_snr: np.ndarray
    candidate_bins: np.ndarray
    detection_snr: float


def matched_filter_detect(
    fid_samples: np.ndarray,
    sample_dt_us: float,
    *,
    start_us: float,
    end_us: float,
    tau_basis_us: float,
    probe_freq_mhz: float,
    sideband: Sideband,
    n_padded: int,
    detection_snr: float = 4.0,
    sigma: Optional[np.ndarray] = None,
    min_separation_bins: int = 1,
) -> MatchedFilterResult:
    """Matched-filter peak detector for a Lorentzian signal model.

    Implements the Neyman-Pearson detector for a damped-sinusoid signal
    of decay constant ``tau_basis_us`` under Gaussian time-domain
    noise: apply ``exp(-(t-t0)/tau_basis_us)`` apodization to the
    active region, FFT, divide the magnitude by per-bin σ, and find
    local maxima above ``detection_snr``.

    The Parseval equivalence -- σ-weighted Lorentzian projection at
    every ``f_c`` is one apodized FFT, not N projections -- makes this
    O(N log N). See the theory section in ``report.md``.

    Parameters
    ----------
    fid_samples : np.ndarray
        Raw FID samples (real, 1-D).
    sample_dt_us : float
        FID sample spacing (microseconds).
    start_us, end_us : float
        Active region bounds (microseconds).
    tau_basis_us : float
        Exponential apodization time constant (microseconds). This is
        the matched-filter weight: ``w(t) = exp(-(t-t0)/tau_basis_us)``.
        For a known molecular decay ``τ_truth``, the optimal matched-
        filter τ_basis equals τ_truth (under white time-domain noise);
        the screen study recommends ``2×τ_truth`` for narrower lines
        that suppress strong-line sidelobes.
    probe_freq_mhz, sideband, n_padded : passthrough to
        :func:`compute_active_ft`.
    detection_snr : float, default 4.0
        Local-maxima SNR threshold. 4 is the textbook single-bin
        detection threshold (Rayleigh tail probability ~3e-4); 2-3 is
        more aggressive (catches more real lines, more FPs).
    sigma : np.ndarray, optional
        Per-bin |X| RMS noise. When ``None`` the function estimates σ
        on the resulting active-FT magnitude via
        :func:`estimate_noise_scatter`. Synthetic callers can pass an
        analytic constant σ to isolate the detector's behaviour from
        the scatter estimator's quirks.
    min_separation_bins : int, default 1
        Forwarded to ``find_peaks`` as the ``distance`` argument. The
        production locator uses Savitzky-Golay concavity to suppress
        adjacent-bin duplicates of one main lobe; the matched filter
        has no such mechanism, so passing a separation of order the
        apodized FWHM keeps the candidate count comparable. The
        default ``1`` is permissive (any non-adjacent local maxima
        kept); callers that know the FWHM can pass a higher value.

    Returns
    -------
    MatchedFilterResult
        The active-FT, per-bin σ, per-bin SNR, and the kept candidate
        bin indices.
    """
    if tau_basis_us <= 0.0:
        raise ValueError("tau_basis_us must be positive")
    if detection_snr <= 0.0:
        raise ValueError("detection_snr must be positive")

    # Build the exponentially-apodized active-region FT.  The current API has
    # no expf_us knob on compute_active_ft (canonical FT is unapodized); to
    # reproduce the matched-filter exp-apodized FFT, multiply the active-region
    # samples by the window before calling compute_active_ft.  This mirrors
    # stage3_impl._mf_gap_spectrum exactly: extract the active slice, multiply
    # by matched_filter_window(t_rel, tau_basis_us), then rfft via
    # compute_active_ft on the windowed copy.
    fid_arr = np.asarray(fid_samples, dtype=float)
    time_us = np.arange(fid_arr.size) * sample_dt_us
    start_idx = int(np.searchsorted(time_us, start_us))
    end_idx = int(np.searchsorted(time_us, end_us))
    end_idx = min(end_idx, fid_arr.size)
    n_active = end_idx - start_idx
    t_rel = np.arange(n_active) * sample_dt_us
    w = matched_filter_window(t_rel, tau_basis_us, shape="lorentzian")
    fid_windowed = fid_arr.copy()
    fid_windowed[start_idx:end_idx] *= w
    # Zero out samples outside the active region so compute_active_ft sees
    # only the windowed active samples (everything else is dead weight that
    # rdc would subtract anyway, but be explicit).
    fid_windowed[:start_idx] = 0.0
    fid_windowed[end_idx:] = 0.0

    active_ft = compute_active_ft(
        fid=fid_windowed,
        sample_dt_us=sample_dt_us,
        start_us=start_us,
        end_us=end_us,
        probe_freq_mhz=probe_freq_mhz,
        sideband=sideband,
        n_padded=n_padded,
        rdc=True,
    )
    mag = np.abs(active_ft.complex_spectrum)

    if sigma is None:
        noise = estimate_noise_scatter(active_ft.freq_mhz, mag)
        sigma_arr = np.asarray(noise.rms_noise, dtype=float)
    else:
        sigma_arr = np.asarray(sigma, dtype=float)
        if sigma_arr.shape != mag.shape:
            raise ValueError(
                f"sigma shape {sigma_arr.shape} != active-FT shape {mag.shape}"
            )

    sigma_c = sigma_arr / np.sqrt(2.0)
    safe_sigma_c = np.where(sigma_c > 0.0, sigma_c, 1.0)
    per_bin_snr = mag / safe_sigma_c
    per_bin_snr = np.where(sigma_c > 0.0, per_bin_snr, 0.0)

    # Local-maxima detection -- no Sav-Gol smoothing, no concavity test.
    # The per-bin SNR threshold is the discriminator. ``distance`` mimics
    # the production locator's concavity-driven dedup of adjacent main-
    # lobe bins: peaks within ``min_separation_bins`` are not both kept.
    cand_bins, _ = find_peaks(
        per_bin_snr, height=detection_snr, distance=max(1, int(min_separation_bins))
    )

    return MatchedFilterResult(
        active_ft=active_ft,
        sigma=sigma_arr,
        per_bin_snr=per_bin_snr,
        candidate_bins=np.asarray(cand_bins, dtype=int),
        detection_snr=float(detection_snr),
    )


# ===========================================================================
# Section 1b: production-like reference detector for the comparison
# ===========================================================================
@dataclass(frozen=True)
class ProductionLikeResult:
    """Output of ``production_like_detect``.

    Mirrors the Stage 3 primary pass: Blackman-Harris time-domain
    apodization, FFT, Sav-Gol second-derivative locator with the
    production defaults.
    """

    active_ft: ActiveFTResult
    sigma: np.ndarray
    candidate_bins: np.ndarray
    sg_window: int
    sg_order: int
    min_snr: float


def production_like_detect(
    fid_samples: np.ndarray,
    sample_dt_us: float,
    *,
    start_us: float,
    end_us: float,
    probe_freq_mhz: float,
    sideband: Sideband,
    n_padded: int,
    sg_window: int = 11,
    sg_order: int = 3,
    min_snr: float = 2.0,
    sigma: Optional[np.ndarray] = None,
) -> ProductionLikeResult:
    """Stand-in for the Stage 3 primary pass on the active-FT grid.

    Apply Blackman-Harris in time domain (matches the production
    primary apodization; see ``dev-docs/research/peak-detection/report.md``
    §6), FFT the active region with no zero-padding, then run the
    Sav-Gol second-derivative locator with the production defaults
    (``sg_window=11, sg_order=3, min_snr=2.0``).

    The production pipeline applies the window inside Stage 3's
    ``FID.preprocess`` call; here we replicate the geometry directly
    so the comparison runs on the same active-FT grid the matched
    filter sees -- otherwise the bin counts and frequency axes would
    not be apples-to-apples.
    """
    fid_arr = np.asarray(fid_samples, dtype=float)
    if sample_dt_us <= 0.0:
        raise ValueError("sample_dt_us must be positive")

    n_total = fid_arr.size
    time_us = np.arange(n_total) * sample_dt_us
    start_idx = int(np.searchsorted(time_us, start_us))
    end_idx = int(np.searchsorted(time_us, end_us))
    if end_idx <= start_idx:
        raise ValueError("empty active region")
    if end_idx > n_total:
        end_idx = n_total
    active = fid_arr[start_idx:end_idx].astype(float, copy=True)
    n_active = active.size

    # Blackman-Harris in time domain.
    window = get_window("blackmanharris", n_active)
    active *= window
    active -= active.mean()

    spectrum = sample_dt_us * np.fft.rfft(active)
    f_bb = np.fft.rfftfreq(n_active, d=sample_dt_us)
    s = sideband_sign(sideband)
    freq_mhz = probe_freq_mhz + s * f_bb

    # The active-FT for the production-like detector lives on the same
    # baseband grid as the matched filter's, but is reordered into
    # ascending molecular frequency below.
    if n_padded < n_active:
        raise ValueError("n_padded must be >= n_active")
    alpha = float(n_active) / float(n_padded)
    active_ft = ActiveFTResult(
        freq_mhz=freq_mhz.astype(float),
        complex_spectrum=spectrum.astype(np.complex128),
        alpha=alpha,
        n_active=int(n_active),
        n_padded=int(n_padded),
    )
    mag = np.abs(spectrum)

    if sigma is None:
        # estimate_noise_scatter works on ascending or descending axes.
        noise_sorted = estimate_noise_scatter(freq_mhz, mag)
        sigma_arr = np.asarray(noise_sorted.rms_noise, dtype=float)
    else:
        sigma_arr = np.asarray(sigma, dtype=float)
        if sigma_arr.shape != mag.shape:
            raise ValueError("sigma shape mismatch")

    # locate_peaks expects a uniformly-spaced x axis; the molecular grid
    # is uniform after sorting (constant bin spacing == 1/T_active).
    sort_idx = np.argsort(freq_mhz)
    x_sorted = freq_mhz[sort_idx]
    y_sorted = mag[sort_idx]
    sd_sorted = sigma_arr[sort_idx]
    locator = locate_peaks(
        x_sorted,
        y_sorted,
        window=sg_window,
        order=sg_order,
        thresh=min_snr * sd_sorted,
    )
    # Apex-snap: SavGol's argrelmin of the 2nd derivative on narrow
    # lines lands several bins off the true peak (the smoothing kernel
    # spans more bins than the FWHM). Production recovers the apex by
    # snapping to the local magnitude maximum within ±sg_window/2.
    radius = sg_window // 2
    snapped: list[int] = []
    for idx in locator.indices:
        lo = max(0, int(idx) - radius)
        hi = min(y_sorted.size, int(idx) + radius + 1)
        snapped.append(int(lo + np.argmax(y_sorted[lo:hi])))
    cand_sorted = np.unique(np.array(snapped, dtype=int))
    cand_bins_unsorted = sort_idx[cand_sorted]

    return ProductionLikeResult(
        active_ft=active_ft,
        sigma=sigma_arr,
        candidate_bins=np.asarray(cand_bins_unsorted, dtype=int),
        sg_window=int(sg_window),
        sg_order=int(sg_order),
        min_snr=float(min_snr),
    )


# ===========================================================================
# Section 1b': hybrid -- matched apodization + Sav-Gol concavity locator
# ===========================================================================
@dataclass(frozen=True)
class HybridResult:
    """Output of ``matched_filter_concavity_detect``.

    The hybrid pairs the matched-filter exponential apodization
    (best per-bin SNR for a damped sinusoid) with the production
    Sav-Gol concavity locator (rejects monotonic skirt bins of strong
    lines). Same field semantics as ``MatchedFilterResult`` minus the
    per-bin SNR array (the locator does its own thresholding).
    """

    active_ft: ActiveFTResult
    sigma: np.ndarray
    candidate_bins: np.ndarray
    tau_basis_us: float
    sg_window: int
    sg_order: int
    min_snr: float


def matched_filter_concavity_detect(
    fid_samples: np.ndarray,
    sample_dt_us: float,
    *,
    start_us: float,
    end_us: float,
    tau_basis_us: float,
    probe_freq_mhz: float,
    sideband: Sideband,
    n_padded: int,
    sg_window: int = 11,
    sg_order: int = 3,
    min_snr: float = 2.0,
    sigma: Optional[np.ndarray] = None,
) -> HybridResult:
    """Matched-filter apodization + Sav-Gol concavity locator.

    The reasoning: ``matched_filter_detect`` over-produces near strong
    lines because the per-bin SNR threshold accepts every above-4σ
    bin in the Lorentzian skirt (§5 of the matched-filter report).
    The Sav-Gol locator's concavity test rejects monotonic skirt bins,
    leaving only the line centre. Apply the locator on the
    *matched-filter spectrum* (exp-apodized at τ_basis) instead of
    the Blackman-Harris one to keep the per-bin SNR advantage on weak
    damped lines while gaining the structural FP suppression.

    No zero-padding -- the active region only, FFT length ``N_active``.
    Apodization referenced to ``start_us`` (``t=0`` at the
    active-region turn-on). ``min_snr`` is the magnitude threshold the
    locator applies; the matched-filter spectrum's σ is the noise
    reference.
    """
    if tau_basis_us <= 0.0:
        raise ValueError("tau_basis_us must be positive")
    if min_snr <= 0.0:
        raise ValueError("min_snr must be positive")

    # Windowed active-region FT: same construction as matched_filter_detect.
    # Multiply active-region samples by exp(-t/tau_basis_us) before rfft so
    # the apodized spectrum is the matched filter at tau_basis_us.
    fid_arr_h = np.asarray(fid_samples, dtype=float)
    time_us_h = np.arange(fid_arr_h.size) * sample_dt_us
    start_idx_h = int(np.searchsorted(time_us_h, start_us))
    end_idx_h = int(np.searchsorted(time_us_h, end_us))
    end_idx_h = min(end_idx_h, fid_arr_h.size)
    n_active_h = end_idx_h - start_idx_h
    t_rel_h = np.arange(n_active_h) * sample_dt_us
    w_h = matched_filter_window(t_rel_h, tau_basis_us, shape="lorentzian")
    fid_windowed_h = fid_arr_h.copy()
    fid_windowed_h[start_idx_h:end_idx_h] *= w_h
    fid_windowed_h[:start_idx_h] = 0.0
    fid_windowed_h[end_idx_h:] = 0.0

    active_ft = compute_active_ft(
        fid=fid_windowed_h,
        sample_dt_us=sample_dt_us,
        start_us=start_us,
        end_us=end_us,
        probe_freq_mhz=probe_freq_mhz,
        sideband=sideband,
        n_padded=n_padded,
        rdc=True,
    )
    mag = np.abs(active_ft.complex_spectrum)
    if sigma is None:
        noise = estimate_noise_scatter(active_ft.freq_mhz, mag)
        sigma_arr = np.asarray(noise.rms_noise, dtype=float)
    else:
        sigma_arr = np.asarray(sigma, dtype=float)
        if sigma_arr.shape != mag.shape:
            raise ValueError(
                f"sigma shape {sigma_arr.shape} != active-FT shape {mag.shape}"
            )

    # locate_peaks expects ascending x; sort, run, snap, map back.
    sort_idx = np.argsort(active_ft.freq_mhz)
    x_sorted = active_ft.freq_mhz[sort_idx]
    y_sorted = mag[sort_idx]
    sd_sorted = sigma_arr[sort_idx]
    locator = locate_peaks(
        x_sorted, y_sorted,
        window=sg_window, order=sg_order, thresh=min_snr * sd_sorted,
    )
    radius = sg_window // 2
    snapped: list[int] = []
    for idx in locator.indices:
        lo = max(0, int(idx) - radius)
        hi = min(y_sorted.size, int(idx) + radius + 1)
        snapped.append(int(lo + np.argmax(y_sorted[lo:hi])))
    cand_sorted = np.unique(np.array(snapped, dtype=int))
    cand_bins = sort_idx[cand_sorted] if cand_sorted.size else np.array([], dtype=int)

    return HybridResult(
        active_ft=active_ft,
        sigma=sigma_arr,
        candidate_bins=np.asarray(cand_bins, dtype=int),
        tau_basis_us=float(tau_basis_us),
        sg_window=int(sg_window),
        sg_order=int(sg_order),
        min_snr=float(min_snr),
    )


# ===========================================================================
# Section 1c: simulator wrapping -- the screen study's simulator returns
# an active-FT computed from the noiseless+noise FID without any extra
# apodization. To run the matched filter against the *FID samples*, we
# rebuild the FID from the simulator's parameters.
# ===========================================================================
def regenerate_fid_for_sim(
    sim: SyntheticActiveFT,
    *,
    n_active: int = SIM_N_ACTIVE,
    sample_dt_us: float = SIM_SAMPLE_DT_US,
    probe_freq_mhz: float = SIM_PROBE_MHZ,
    sideband: Sideband = SIM_SIDEBAND,
    rng_seed: int = 0,
    strong_separation_bins: float = 80.0,
) -> Tuple[np.ndarray, float]:
    """Rebuild the noisy FID that gave rise to a SyntheticActiveFT.

    The screen study's simulator returns the active-FT but not the
    underlying FID -- we need the FID to feed the matched-filter
    detector (which apodizes in time domain). This rebuilds it from
    the simulator's parameters using the same construction logic so
    the regenerated active-FT matches ``sim.spectrum`` bit-for-bit.

    Returns ``(fid_noisy, time_domain_sigma)``. The time-domain σ
    is what was used to size the noise floor; the analytic per-bin
    |X| RMS is ``time_domain_sigma * sqrt(T_active * dt)``.
    """
    # Reproduce the construction in screen-study prototype.py::simulate_active_ft.
    # We can't pull the noisy FID out without re-running the same
    # parameters + seed, so we accept the redundancy.
    rng = np.random.default_rng(rng_seed)
    T_active = n_active * sample_dt_us
    tau_truth = sim.tau_truth_us
    fwhm_bins = sim.fwhm_bins
    n_lines = sim.truth_snrs.size
    n_strong = sim.strong_snrs.size
    weak_snr = float(sim.truth_snrs[0]) if n_lines else 1.0
    strong_snr = float(sim.strong_snrs[0]) if n_strong else 0.0

    edge_guard_bins = 20
    min_line_separation_bins = 6.0
    valid_bins = np.arange(edge_guard_bins, n_active // 2 - edge_guard_bins)

    strong_chosen: list[int] = []
    attempts = 0
    while len(strong_chosen) < n_strong and attempts < 200 * max(n_strong, 1):
        attempts += 1
        b = int(rng.choice(valid_bins))
        if any(abs(b - c) < strong_separation_bins for c in strong_chosen):
            continue
        strong_chosen.append(b)
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
    chosen.sort()
    strong_chosen.sort()
    line_bins = np.array(chosen, dtype=int)
    strong_bins = np.array(strong_chosen, dtype=int)
    f_bb_lines = line_bins.astype(float) / T_active
    f_bb_strong = strong_bins.astype(float) / T_active

    t = np.arange(n_active) * sample_dt_us
    fid = np.zeros(n_active, dtype=float)
    weak_amp = 1.0
    strong_amp = (strong_snr / weak_snr) * weak_amp if n_strong else 0.0
    weak_phases = rng.uniform(0, 2 * np.pi, size=n_lines)
    strong_phases = rng.uniform(0, 2 * np.pi, size=n_strong)
    for i in range(n_lines):
        fid += weak_amp * np.cos(
            2 * np.pi * f_bb_lines[i] * t + weak_phases[i]
        ) * np.exp(-t / tau_truth)
    for i in range(n_strong):
        fid += strong_amp * np.cos(
            2 * np.pi * f_bb_strong[i] * t + strong_phases[i]
        ) * np.exp(-t / tau_truth)

    spec_signal = sample_dt_us * np.fft.rfft(fid)
    on_line_mag = float(np.median(np.abs(spec_signal[line_bins])))
    target_sigma_x = on_line_mag / weak_snr
    sigma_time = target_sigma_x / np.sqrt(T_active * sample_dt_us)
    noise_t = rng.normal(scale=sigma_time, size=n_active)
    fid_noisy = fid + noise_t
    return fid_noisy, sigma_time


# ===========================================================================
# Section 2: smoke test
# ===========================================================================
def figure_smoke_test() -> None:
    """Single-cell illustration: both detectors at SNR=4, FWHM/bin=1.34.

    Visualises:
    - The exp-apodized active-FT with the matched-filter SNR threshold.
    - The BH-apodized active-FT with the SavGol candidate set.
    - Ground truth markers and the candidate sets overlaid.
    """
    sim = simulate_active_ft(
        fwhm_bins=1.34,
        true_snr=4.0,
        n_lines=15,
        n_active=SIM_N_ACTIVE,
        sample_dt_us=SIM_SAMPLE_DT_US,
        probe_freq_mhz=SIM_PROBE_MHZ,
        sideband=SIM_SIDEBAND,
        rng_seed=RNG_SEED,
    )
    fid, sigma_time = regenerate_fid_for_sim(sim, rng_seed=RNG_SEED)

    T_active = SIM_N_ACTIVE * SIM_SAMPLE_DT_US
    # Analytic per-bin |X| RMS for the exp-apodized spectrum: with white
    # time-domain noise of σ_time the bin variance is σ_time² · T · dt;
    # apodization scales the effective bin variance by Σ w²(t) / N.
    # The screen study uses uniform analytic σ derived from the
    # noise-free FFT; we apply the same logic here with the apodization-
    # adjusted scale.
    tau_basis = 2.0 * sim.tau_truth_us  # screen study's recommended setting
    # Noise on the exp-apodized FT: σ_x_apod = σ_time * sqrt(Σ w²(t) · dt²).
    t_us = np.arange(SIM_N_ACTIVE) * SIM_SAMPLE_DT_US
    w_exp = np.exp(-t_us / tau_basis)
    sigma_x_exp = sigma_time * np.sqrt(np.sum(w_exp**2)) * SIM_SAMPLE_DT_US
    sigma_exp = np.full(SIM_N_ACTIVE // 2 + 1, sigma_x_exp, dtype=float)

    mf = matched_filter_detect(
        fid,
        SIM_SAMPLE_DT_US,
        start_us=0.0,
        end_us=T_active,
        tau_basis_us=tau_basis,
        probe_freq_mhz=SIM_PROBE_MHZ,
        sideband=SIM_SIDEBAND,
        n_padded=SIM_N_ACTIVE,
        detection_snr=4.0,
        sigma=sigma_exp,
        min_separation_bins=max(2, int(round(sim.fwhm_bins))),
    )

    # Production-like: BH window, no exp filter.
    w_bh = get_window("blackmanharris", SIM_N_ACTIVE)
    sigma_x_bh = sigma_time * np.sqrt(np.sum(w_bh**2)) * SIM_SAMPLE_DT_US
    sigma_bh = np.full(SIM_N_ACTIVE // 2 + 1, sigma_x_bh, dtype=float)
    pl = production_like_detect(
        fid,
        SIM_SAMPLE_DT_US,
        start_us=0.0,
        end_us=T_active,
        probe_freq_mhz=SIM_PROBE_MHZ,
        sideband=SIM_SIDEBAND,
        n_padded=SIM_N_ACTIVE,
        sg_window=11,
        sg_order=3,
        min_snr=2.0,
        sigma=sigma_bh,
    )

    # Map all axes/spectra into a common ascending molecular-frequency view.
    mf_sort = np.argsort(mf.active_ft.freq_mhz)
    pl_sort = np.argsort(pl.active_ft.freq_mhz)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    ax = axes[0]
    ax.plot(
        mf.active_ft.freq_mhz[mf_sort],
        mf.per_bin_snr[mf_sort],
        color="C0",
        lw=0.7,
        label="matched filter SNR  (τ_basis = 2·τ_truth)",
    )
    ax.axhline(mf.detection_snr, color="C0", ls="--", alpha=0.5, label=f"thr={mf.detection_snr:.1f}")
    cand_freqs_mf = mf.active_ft.freq_mhz[mf.candidate_bins]
    ax.scatter(
        cand_freqs_mf,
        mf.per_bin_snr[mf.candidate_bins],
        color="C0",
        s=30,
        marker="x",
        label=f"MF candidates ({len(cand_freqs_mf)})",
    )
    for f in sim.truth_freqs_mhz:
        ax.axvline(f, color="C2", lw=0.6, alpha=0.5, ls=":")
    ax.set_ylabel("per-bin SNR  |X|/σ_c")
    ax.set_title(
        f"smoke test:  injected SNR={sim.truth_snrs[0]:.1f}, "
        f"FWHM/bin={sim.fwhm_bins:.2f}, "
        f"τ_truth={sim.tau_truth_us:.2f} µs, n_lines={sim.truth_snrs.size}"
    )
    ax.legend(loc="upper right", fontsize=9)

    ax2 = axes[1]
    pl_mag = np.abs(pl.active_ft.complex_spectrum)
    ax2.plot(
        pl.active_ft.freq_mhz[pl_sort],
        pl_mag[pl_sort],
        color="C3",
        lw=0.7,
        label="BH-apodized |X|",
    )
    cand_freqs_pl = pl.active_ft.freq_mhz[pl.candidate_bins]
    ax2.scatter(
        cand_freqs_pl,
        pl_mag[pl.candidate_bins],
        color="C3",
        s=30,
        marker="x",
        label=f"SavGol candidates ({len(cand_freqs_pl)})",
    )
    for f in sim.truth_freqs_mhz:
        ax2.axvline(f, color="C2", lw=0.6, alpha=0.5, ls=":")
    ax2.set_xlabel("freq (MHz)")
    ax2.set_ylabel("|X| (BH-apodized)")
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    out = FIG / "01_smoke_test.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", out)

    # Smoke-test sanity: report TP/FP for both detectors in the log.
    def _classify(cand_freqs: np.ndarray, truth_freqs: np.ndarray) -> Tuple[int, int]:
        tol_mhz = max(1.0, sim.fwhm_bins / 2.0) * sim.bin_mhz
        n_tp = 0
        for tf in truth_freqs:
            if np.any(np.abs(cand_freqs - tf) <= tol_mhz):
                n_tp += 1
        # Avoid double-counting: a single candidate near two truths still
        # counts as one TP. FP = candidates with no truth within tolerance.
        n_fp = 0
        for cf in cand_freqs:
            if not np.any(np.abs(sim.truth_freqs_mhz - cf) <= tol_mhz):
                n_fp += 1
        return n_tp, n_fp

    n_tp_mf, n_fp_mf = _classify(cand_freqs_mf, sim.truth_freqs_mhz)
    n_tp_pl, n_fp_pl = _classify(cand_freqs_pl, sim.truth_freqs_mhz)
    logger.info(
        "smoke test: MF  TP=%d/%d  FP=%d  ||  PL  TP=%d/%d  FP=%d",
        n_tp_mf, sim.truth_snrs.size, n_fp_mf,
        n_tp_pl, sim.truth_snrs.size, n_fp_pl,
    )


# ===========================================================================
# Section 3: phase-space sweep
# ===========================================================================
SNR_GRID = np.array([2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 25.0])
FWHM_GRID = np.array([0.8, 1.0, 1.34, 2.0, 3.0, 5.0])

# High-SNR extension: real FTMW data has strong lines at SNR up to 10⁴,
# orders of magnitude above the main sweep's ceiling. The matched filter's
# per-bin SNR threshold accepts every above-4σ bin of a Lorentzian skirt;
# at SNR=1000 that's ~15 bins per side, at SNR=10000 ~50 bins per side.
# This sweep tests whether the Sav-Gol concavity rejection scales (it
# should, because skirts are concave-up, not concave-down).
SNR_GRID_HIGH = np.array([50.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0])


@dataclass
class DetectorCell:
    """Per-cell aggregated detector + screen statistics."""

    snr: float
    fwhm_bins: float
    detector: str  # "matched_filter" | "production_like"
    # Pre-screen counts
    n_candidates: int
    n_tp: int  # candidates within tol of any injected weak line
    n_fp_noise: int  # candidates not TP and not in strong skirt
    n_fp_sidelobe: int  # candidates not TP and within strong skirt
    n_injected_recovered: int  # how many of the weak truths have >=1 candidate
    n_injected_total: int
    # Post-screen counts (at the same threshold for both detectors)
    n_post_tp: int
    n_post_fp_noise: int
    n_post_fp_sidelobe: int
    n_post_recovered: int
    # Screen AUC (TP vs all FPs combined; nan if either class empty)
    screen_auc_all: float
    screen_auc_noise: float
    screen_auc_sidelobe: float
    # Threshold the screen used to produce the post counts
    screen_threshold: float


def _classify_candidates(
    cand_bins: np.ndarray,
    sim: SyntheticActiveFT,
    ascending_freq: np.ndarray,
    sidelobe_range_bins: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (is_tp, in_strong_skirt) arrays for each candidate bin.

    Both arrays are computed on the ascending grid passed in (the
    detector's natural grid; sorted by the caller).
    """
    truth_match_bins = max(1, int(round(sim.fwhm_bins / 2.0)))
    truth_bins = np.searchsorted(ascending_freq, sim.truth_freqs_mhz)
    truth_bins = np.clip(truth_bins, 0, ascending_freq.size - 1)
    if sim.strong_freqs_mhz.size:
        strong_bins = np.searchsorted(ascending_freq, sim.strong_freqs_mhz)
        strong_bins = np.clip(strong_bins, 0, ascending_freq.size - 1)
    else:
        strong_bins = np.array([], dtype=int)
    is_tp = np.zeros(cand_bins.size, dtype=bool)
    in_skirt = np.zeros(cand_bins.size, dtype=bool)
    for k, cb in enumerate(cand_bins):
        seps_w = np.abs(truth_bins - int(cb)) if truth_bins.size else None
        seps_s = np.abs(strong_bins - int(cb)) if strong_bins.size else None
        tp_w = bool(seps_w is not None and seps_w.min() <= truth_match_bins)
        tp_s = bool(seps_s is not None and seps_s.min() <= truth_match_bins)
        is_tp[k] = tp_w or tp_s
        if not is_tp[k] and seps_s is not None:
            in_skirt[k] = bool(seps_s.min() <= sidelobe_range_bins)
    return is_tp, in_skirt


def _ascending_view(
    freq_mhz: np.ndarray, spec: np.ndarray, sigma: np.ndarray, cand_bins: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sort detector output ascending and translate candidate bins."""
    sort_idx = np.argsort(freq_mhz)
    inv = np.argsort(sort_idx)
    freq_asc = freq_mhz[sort_idx]
    spec_asc = spec[sort_idx]
    sigma_asc = sigma[sort_idx]
    cand_asc = inv[cand_bins]
    return freq_asc, spec_asc, sigma_asc, np.asarray(cand_asc, dtype=int)


def _auc(scores_tp: np.ndarray, scores_fp: np.ndarray) -> float:
    """Mann-Whitney U-based AUC: P(score_tp > score_fp)."""
    if scores_tp.size == 0 or scores_fp.size == 0:
        return float("nan")
    from scipy.stats import mannwhitneyu

    u, _ = mannwhitneyu(scores_tp, scores_fp, alternative="greater")
    return float(u / (scores_tp.size * scores_fp.size))


def _evaluate_detector(
    sim: SyntheticActiveFT,
    detector: str,
    fid: np.ndarray,
    sigma_time: float,
    *,
    sidelobe_range_bins: float,
    screen_factor: float,
    screen_threshold: float,
    detection_snr: float,
    min_snr_pl: float,
) -> DetectorCell:
    """Run one detector + the screen on one sim realisation and aggregate."""
    T_active = SIM_N_ACTIVE * SIM_SAMPLE_DT_US
    t_us = np.arange(SIM_N_ACTIVE) * SIM_SAMPLE_DT_US

    if detector == "matched_filter":
        tau_basis = sim.tau_truth_us * 2.0  # screen-recommended operating point
        w = np.exp(-t_us / tau_basis)
        sigma_x = sigma_time * np.sqrt(np.sum(w**2)) * SIM_SAMPLE_DT_US
        sigma_arr = np.full(SIM_N_ACTIVE // 2 + 1, sigma_x, dtype=float)
        # Apodized FWHM ≈ fwhm_bins for the 2·τ_truth basis (data decay
        # still dominates). Pass that as min_separation_bins so adjacent
        # main-lobe bins of one line aren't both reported. Floor at 2
        # because distance=1 still allows immediate neighbours.
        min_sep = max(2, int(round(sim.fwhm_bins)))
        res = matched_filter_detect(
            fid,
            SIM_SAMPLE_DT_US,
            start_us=0.0,
            end_us=T_active,
            tau_basis_us=tau_basis,
            probe_freq_mhz=SIM_PROBE_MHZ,
            sideband=SIM_SIDEBAND,
            n_padded=SIM_N_ACTIVE,
            detection_snr=detection_snr,
            sigma=sigma_arr,
            min_separation_bins=min_sep,
        )
        freq = res.active_ft.freq_mhz
        spec = res.active_ft.complex_spectrum
        sigma_used = res.sigma
        cand_bins = res.candidate_bins
    elif detector == "production_like":
        w = get_window("blackmanharris", SIM_N_ACTIVE)
        sigma_x = sigma_time * np.sqrt(np.sum(w**2)) * SIM_SAMPLE_DT_US
        sigma_arr = np.full(SIM_N_ACTIVE // 2 + 1, sigma_x, dtype=float)
        res = production_like_detect(
            fid,
            SIM_SAMPLE_DT_US,
            start_us=0.0,
            end_us=T_active,
            probe_freq_mhz=SIM_PROBE_MHZ,
            sideband=SIM_SIDEBAND,
            n_padded=SIM_N_ACTIVE,
            sg_window=11,
            sg_order=3,
            min_snr=min_snr_pl,
            sigma=sigma_arr,
        )
        freq = res.active_ft.freq_mhz
        spec = res.active_ft.complex_spectrum
        sigma_used = res.sigma
        cand_bins = res.candidate_bins
    elif detector == "production_two_pass":
        # Production primary: BH window + SavGol locator.
        w_bh = get_window("blackmanharris", SIM_N_ACTIVE)
        sigma_x_bh = sigma_time * np.sqrt(np.sum(w_bh**2)) * SIM_SAMPLE_DT_US
        sigma_bh = np.full(SIM_N_ACTIVE // 2 + 1, sigma_x_bh, dtype=float)
        prim = production_like_detect(
            fid, SIM_SAMPLE_DT_US,
            start_us=0.0, end_us=T_active,
            probe_freq_mhz=SIM_PROBE_MHZ, sideband=SIM_SIDEBAND,
            n_padded=SIM_N_ACTIVE,
            sg_window=11, sg_order=3, min_snr=min_snr_pl, sigma=sigma_bh,
        )
        # Production gap: unapodized FFT + SavGol locator. Build the
        # active-FT with no apodization at all.
        unapod = compute_active_ft(
            fid=fid, sample_dt_us=SIM_SAMPLE_DT_US,
            start_us=0.0, end_us=T_active,
            probe_freq_mhz=SIM_PROBE_MHZ, sideband=SIM_SIDEBAND,
            n_padded=SIM_N_ACTIVE, rdc=True,
        )
        sigma_x_un = sigma_time * np.sqrt(T_active * SIM_SAMPLE_DT_US)
        sigma_un = np.full(SIM_N_ACTIVE // 2 + 1, sigma_x_un, dtype=float)
        sort_idx = np.argsort(unapod.freq_mhz)
        inv = np.argsort(sort_idx)
        x_sorted = unapod.freq_mhz[sort_idx]
        y_sorted = np.abs(unapod.complex_spectrum)[sort_idx]
        sd_sorted = sigma_un[sort_idx]
        gap_loc = locate_peaks(
            x_sorted, y_sorted,
            window=11, order=3, thresh=min_snr_pl * sd_sorted,
        )
        gap_bins = sort_idx[gap_loc.indices]
        # Union of primary + gap positions on the BH grid (use BH spectrum
        # for the ratio downstream so the screen sees the same grid for
        # both detectors).
        cand_union = np.union1d(prim.candidate_bins, np.asarray(gap_bins, dtype=int))
        freq = prim.active_ft.freq_mhz
        spec = prim.active_ft.complex_spectrum
        sigma_used = prim.sigma
        cand_bins = cand_union
    elif detector == "mf_concavity":
        # The hybrid: matched-filter apodization + Sav-Gol concavity locator.
        tau_basis = sim.tau_truth_us * 2.0
        w = np.exp(-t_us / tau_basis)
        sigma_x = sigma_time * np.sqrt(np.sum(w**2)) * SIM_SAMPLE_DT_US
        sigma_arr = np.full(SIM_N_ACTIVE // 2 + 1, sigma_x, dtype=float)
        res = matched_filter_concavity_detect(
            fid, SIM_SAMPLE_DT_US,
            start_us=0.0, end_us=T_active,
            tau_basis_us=tau_basis,
            probe_freq_mhz=SIM_PROBE_MHZ, sideband=SIM_SIDEBAND,
            n_padded=SIM_N_ACTIVE,
            sg_window=11, sg_order=3, min_snr=min_snr_pl,
            sigma=sigma_arr,
        )
        freq = res.active_ft.freq_mhz
        spec = res.active_ft.complex_spectrum
        sigma_used = res.sigma
        cand_bins = res.candidate_bins
    elif detector == "production_with_mf_gap":
        # Production primary (BH+SavGol) + matched-filter gap (MF+SavGol).
        # The gap pass uses exp-apodization at τ_basis instead of boxcar.
        # No leakage mask in synthetic comparison -- production uses one on
        # real data, the synthetic doesn't need it (no leakage from finite
        # acquisition because lines decay before T_active ends in this regime).
        w_bh = get_window("blackmanharris", SIM_N_ACTIVE)
        sigma_x_bh = sigma_time * np.sqrt(np.sum(w_bh**2)) * SIM_SAMPLE_DT_US
        sigma_bh = np.full(SIM_N_ACTIVE // 2 + 1, sigma_x_bh, dtype=float)
        prim = production_like_detect(
            fid, SIM_SAMPLE_DT_US,
            start_us=0.0, end_us=T_active,
            probe_freq_mhz=SIM_PROBE_MHZ, sideband=SIM_SIDEBAND,
            n_padded=SIM_N_ACTIVE,
            sg_window=11, sg_order=3, min_snr=min_snr_pl, sigma=sigma_bh,
        )
        tau_basis = sim.tau_truth_us * 2.0
        w_mf = np.exp(-t_us / tau_basis)
        sigma_x_mf = sigma_time * np.sqrt(np.sum(w_mf**2)) * SIM_SAMPLE_DT_US
        sigma_mf = np.full(SIM_N_ACTIVE // 2 + 1, sigma_x_mf, dtype=float)
        gap = matched_filter_concavity_detect(
            fid, SIM_SAMPLE_DT_US,
            start_us=0.0, end_us=T_active,
            tau_basis_us=tau_basis,
            probe_freq_mhz=SIM_PROBE_MHZ, sideband=SIM_SIDEBAND,
            n_padded=SIM_N_ACTIVE,
            sg_window=11, sg_order=3, min_snr=min_snr_pl,
            sigma=sigma_mf,
        )
        # The primary and gap detect on DIFFERENT grids (BH vs MF). We need
        # to report candidates on a single grid for the screen + truth match.
        # Use the MF grid as the reporting reference (it's the one the screen
        # is calibrated against). Translate primary's BH-grid bin → MF-grid
        # nearest bin via frequency.
        prim_freqs = prim.active_ft.freq_mhz[prim.candidate_bins]
        mf_freq = gap.active_ft.freq_mhz
        sort_mf = np.argsort(mf_freq)
        mf_asc = mf_freq[sort_mf]
        prim_on_mf = []
        for f in prim_freqs:
            pos = int(np.searchsorted(mf_asc, f))
            pos = min(max(pos, 1), mf_asc.size - 1)
            if abs(f - mf_asc[pos - 1]) <= abs(f - mf_asc[pos]):
                pos -= 1
            prim_on_mf.append(int(sort_mf[pos]))
        cand_union = np.union1d(
            np.asarray(prim_on_mf, dtype=int), gap.candidate_bins
        )
        freq = gap.active_ft.freq_mhz
        spec = gap.active_ft.complex_spectrum
        sigma_used = gap.sigma
        cand_bins = cand_union
    else:
        raise ValueError(f"unknown detector: {detector}")

    freq_asc, spec_asc, sigma_asc, cand_asc = _ascending_view(
        freq, spec, sigma_used, cand_bins
    )
    is_tp, in_skirt = _classify_candidates(
        cand_asc, sim, freq_asc, sidelobe_range_bins
    )
    n_tp = int(is_tp.sum())
    n_fp_noise = int(np.sum(~is_tp & ~in_skirt))
    n_fp_sidelobe = int(np.sum(~is_tp & in_skirt))
    # Detection-yield: how many injected weak lines have at least one candidate.
    truth_match_bins = max(1, int(round(sim.fwhm_bins / 2.0)))
    truth_bins = np.searchsorted(freq_asc, sim.truth_freqs_mhz)
    truth_bins = np.clip(truth_bins, 0, freq_asc.size - 1)
    n_recovered = 0
    if cand_asc.size:
        for tb in truth_bins:
            if np.any(np.abs(cand_asc - tb) <= truth_match_bins):
                n_recovered += 1

    # Apply the screen at tau_basis = screen_factor * tau_truth.
    cand_freqs = freq_asc[cand_asc].tolist()
    tau_screen = sim.tau_truth_us * screen_factor
    projections = project_candidates(
        freq_asc,
        spec_asc,
        sigma_asc,
        cand_freqs,
        tau_us=tau_screen,
        acquisition_us=sim.acquisition_us,
        sideband=SIM_SIDEBAND,
    )
    ratios = np.array([p.ratio for p in projections])
    keep = ratios >= screen_threshold
    post_tp = int(np.sum(is_tp & keep))
    post_fp_noise = int(np.sum(~is_tp & ~in_skirt & keep))
    post_fp_sidelobe = int(np.sum(~is_tp & in_skirt & keep))
    # Post-screen recall
    post_recovered = 0
    if cand_asc.size:
        kept_bins = cand_asc[keep]
        for tb in truth_bins:
            if kept_bins.size and np.any(np.abs(kept_bins - tb) <= truth_match_bins):
                post_recovered += 1
    auc_all = _auc(ratios[is_tp], ratios[~is_tp])
    auc_noise = _auc(ratios[is_tp], ratios[~is_tp & ~in_skirt])
    auc_side = _auc(ratios[is_tp], ratios[~is_tp & in_skirt])

    return DetectorCell(
        snr=float(sim.truth_snrs[0]) if sim.truth_snrs.size else float("nan"),
        fwhm_bins=float(sim.fwhm_bins),
        detector=detector,
        n_candidates=int(cand_asc.size),
        n_tp=n_tp,
        n_fp_noise=n_fp_noise,
        n_fp_sidelobe=n_fp_sidelobe,
        n_injected_recovered=int(n_recovered),
        n_injected_total=int(sim.truth_snrs.size),
        n_post_tp=post_tp,
        n_post_fp_noise=post_fp_noise,
        n_post_fp_sidelobe=post_fp_sidelobe,
        n_post_recovered=post_recovered,
        screen_auc_all=auc_all,
        screen_auc_noise=auc_noise,
        screen_auc_sidelobe=auc_side,
        screen_threshold=float(screen_threshold),
    )


def _aggregate_cells(cells: list[DetectorCell]) -> DetectorCell:
    """Sum counts across simulator trials, average AUCs."""
    first = cells[0]
    sums = {
        "n_candidates": 0, "n_tp": 0, "n_fp_noise": 0, "n_fp_sidelobe": 0,
        "n_injected_recovered": 0, "n_injected_total": 0,
        "n_post_tp": 0, "n_post_fp_noise": 0, "n_post_fp_sidelobe": 0,
        "n_post_recovered": 0,
    }
    aucs_all, aucs_noise, aucs_side = [], [], []
    for c in cells:
        for k in sums:
            sums[k] += getattr(c, k)
        if np.isfinite(c.screen_auc_all):
            aucs_all.append(c.screen_auc_all)
        if np.isfinite(c.screen_auc_noise):
            aucs_noise.append(c.screen_auc_noise)
        if np.isfinite(c.screen_auc_sidelobe):
            aucs_side.append(c.screen_auc_sidelobe)
    return DetectorCell(
        snr=first.snr,
        fwhm_bins=first.fwhm_bins,
        detector=first.detector,
        n_candidates=sums["n_candidates"],
        n_tp=sums["n_tp"],
        n_fp_noise=sums["n_fp_noise"],
        n_fp_sidelobe=sums["n_fp_sidelobe"],
        n_injected_recovered=sums["n_injected_recovered"],
        n_injected_total=sums["n_injected_total"],
        n_post_tp=sums["n_post_tp"],
        n_post_fp_noise=sums["n_post_fp_noise"],
        n_post_fp_sidelobe=sums["n_post_fp_sidelobe"],
        n_post_recovered=sums["n_post_recovered"],
        screen_auc_all=float(np.mean(aucs_all)) if aucs_all else float("nan"),
        screen_auc_noise=float(np.mean(aucs_noise)) if aucs_noise else float("nan"),
        screen_auc_sidelobe=float(np.mean(aucs_side)) if aucs_side else float("nan"),
        screen_threshold=first.screen_threshold,
    )


def run_sweep(
    *,
    n_trials: int = 4,
    n_lines: int = 25,
    n_strong_lines: int = 0,
    strong_snr: float = 50.0,
    strong_separation_bins: float = 80.0,
    sidelobe_range_bins: float = 80.0,
    screen_factor: float = 2.0,
    screen_threshold: float = 0.6,
    detection_snr: float = 4.0,
    min_snr_pl: float = 2.0,
    rng_seed: int = RNG_SEED,
    snr_grid: Optional[np.ndarray] = None,
    fwhm_grid: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Sweep all detectors across SNR × FWHM, return result grids.

    Default grids are ``SNR_GRID × FWHM_GRID``; callers can override
    for a high-SNR extension (``SNR_GRID_HIGH``) or other slicings.
    Returns a dict keyed by detector name; each value is a 2-D object
    array of aggregated :class:`DetectorCell`.
    """
    snr_grid_use = SNR_GRID if snr_grid is None else snr_grid
    fwhm_grid_use = FWHM_GRID if fwhm_grid is None else fwhm_grid
    detector_names = (
        "matched_filter",
        "production_like",
        "production_two_pass",
        "mf_concavity",
        "production_with_mf_gap",
    )
    grids: dict[str, np.ndarray] = {
        name: np.empty((snr_grid_use.size, fwhm_grid_use.size), dtype=object)
        for name in detector_names
    }
    t0 = time.time()
    for i, snr in enumerate(snr_grid_use):
        for j, fwhm in enumerate(fwhm_grid_use):
            per_det_cells: dict[str, list[DetectorCell]] = {
                name: [] for name in detector_names
            }
            for k in range(n_trials):
                sim = simulate_active_ft(
                    fwhm_bins=float(fwhm),
                    true_snr=float(snr),
                    n_lines=n_lines,
                    n_active=SIM_N_ACTIVE,
                    sample_dt_us=SIM_SAMPLE_DT_US,
                    probe_freq_mhz=SIM_PROBE_MHZ,
                    sideband=SIM_SIDEBAND,
                    n_strong_lines=n_strong_lines,
                    strong_snr=strong_snr,
                    strong_separation_bins=strong_separation_bins,
                    rng_seed=rng_seed + k,
                )
                fid, sigma_time = regenerate_fid_for_sim(
                    sim,
                    rng_seed=rng_seed + k,
                    strong_separation_bins=strong_separation_bins,
                )
                for detector in detector_names:
                    cell = _evaluate_detector(
                        sim,
                        detector,
                        fid,
                        sigma_time,
                        sidelobe_range_bins=sidelobe_range_bins,
                        screen_factor=screen_factor,
                        screen_threshold=screen_threshold,
                        detection_snr=detection_snr,
                        min_snr_pl=min_snr_pl,
                    )
                    per_det_cells[detector].append(cell)
            for detector, cells in per_det_cells.items():
                grids[detector][i, j] = _aggregate_cells(cells)
        logger.info(
            "  row snr=%.1f done (cumulative %.1fs)", snr, time.time() - t0
        )
    return grids


# ===========================================================================
# Figure 2: phase-space comparison heatmap (pre-screen recall + FPs)
# ===========================================================================
def _heatmap(
    arr: np.ndarray, ax, *, vmin, vmax, cmap, fmt, title,
    snr_grid: Optional[np.ndarray] = None,
    fwhm_grid: Optional[np.ndarray] = None,
) -> None:
    """Draw a SNR × FWHM heatmap with cell annotations on the given Axes."""
    snr_g = SNR_GRID if snr_grid is None else snr_grid
    fwhm_g = FWHM_GRID if fwhm_grid is None else fwhm_grid
    im = ax.imshow(
        arr,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        extent=(0, fwhm_g.size, 0, snr_g.size),
    )
    ax.set_xticks(np.arange(fwhm_g.size) + 0.5)
    ax.set_xticklabels([f"{x:.1f}" for x in fwhm_g])
    ax.set_yticks(np.arange(snr_g.size) + 0.5)
    ax.set_yticklabels([f"{x:.0f}" if x >= 100 else f"{x:.1f}" for x in snr_g])
    ax.set_xlabel("FWHM (bins)")
    ax.set_ylabel("injected SNR")
    midpoint = 0.5 * (vmin + vmax)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    fmt.format(v),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if v < midpoint else "black",
                )
    ax.set_title(title)
    return im


DETECTOR_LABELS = {
    "matched_filter": "Matched filter (τ_basis=2·τ_truth, thr=4σ)",
    "production_like": "Production primary (BH+SavGol, thr=2σ)",
    "production_two_pass": "Production two-pass (primary + gap union)",
    "mf_concavity": "Hybrid: MF apodization + SavGol concavity",
    "production_with_mf_gap": "Production primary + MF gap (replacing unapod)",
}


def _grid_recall(grid: np.ndarray) -> np.ndarray:
    return np.array([
        [c.n_injected_recovered / c.n_injected_total if c.n_injected_total else np.nan
         for c in row]
        for row in grid
    ])


def _grid_post_recall(grid: np.ndarray) -> np.ndarray:
    return np.array([
        [c.n_post_recovered / c.n_injected_total if c.n_injected_total else np.nan
         for c in row]
        for row in grid
    ])


def _grid_fp(grid: np.ndarray) -> np.ndarray:
    return np.array(
        [[c.n_fp_noise + c.n_fp_sidelobe for c in row] for row in grid],
        dtype=float,
    )


def _grid_post_fp(grid: np.ndarray) -> np.ndarray:
    return np.array(
        [[c.n_post_fp_noise + c.n_post_fp_sidelobe for c in row] for row in grid],
        dtype=float,
    )


def _grid_post_fp_side(grid: np.ndarray) -> np.ndarray:
    return np.array(
        [[c.n_post_fp_sidelobe for c in row] for row in grid],
        dtype=float,
    )


def figure_recall_and_fp(grids: dict[str, np.ndarray], suffix: str = "") -> None:
    """6-panel heatmap: pre-screen recall (top) + FP counts (bottom)
    across the three detectors (columns)."""
    detectors = list(DETECTOR_LABELS.keys())
    fig, axes = plt.subplots(2, len(detectors), figsize=(5 * len(detectors), 10))
    recalls = [_grid_recall(grids[d]) for d in detectors]
    fps = [_grid_fp(grids[d]) for d in detectors]
    vmax_fp = float(max(np.nanmax(arr) for arr in fps))
    for col, det in enumerate(detectors):
        _heatmap(
            recalls[col], axes[0, col],
            vmin=0.0, vmax=1.0, cmap="viridis", fmt="{:.2f}",
            title=f"{DETECTOR_LABELS[det]}\nweak-line recall (pre-screen)",
        )
        _heatmap(
            fps[col], axes[1, col],
            vmin=0.0, vmax=vmax_fp, cmap="inferno", fmt="{:.0f}",
            title=f"{DETECTOR_LABELS[det]}\ntotal FPs (pre-screen)",
        )
    fig.suptitle(f"Pre-screen recall + FP comparison{suffix}")
    fig.tight_layout()
    out = FIG / f"02_recall_fp_comparison{suffix}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", out)


def figure_post_screen(grids: dict[str, np.ndarray], suffix: str = "") -> None:
    """6-panel heatmap: post-screen recall + FPs across three detectors."""
    detectors = list(DETECTOR_LABELS.keys())
    fig, axes = plt.subplots(2, len(detectors), figsize=(5 * len(detectors), 10))
    recalls = [_grid_post_recall(grids[d]) for d in detectors]
    fps = [_grid_post_fp(grids[d]) for d in detectors]
    vmax_fp = float(max(np.nanmax(arr) for arr in fps) or 1.0)
    for col, det in enumerate(detectors):
        _heatmap(
            recalls[col], axes[0, col],
            vmin=0.0, vmax=1.0, cmap="viridis", fmt="{:.2f}",
            title=f"{DETECTOR_LABELS[det]}\npost-screen recall",
        )
        _heatmap(
            fps[col], axes[1, col],
            vmin=0.0, vmax=vmax_fp, cmap="inferno", fmt="{:.0f}",
            title=f"{DETECTOR_LABELS[det]}\ntotal FPs (post-screen)",
        )
    fig.suptitle(f"Post-screen recall + FP comparison{suffix}  (ratio ≥ 0.6)")
    fig.tight_layout()
    out = FIG / f"03_post_screen_comparison{suffix}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", out)


def figure_high_snr_fp(
    grids: dict[str, np.ndarray], suffix: str = ""
) -> None:
    """High-SNR FP-count comparison: how does each detector scale at SNR ≥ 50?

    Real FTMW data has strong lines at SNR up to 10⁴. The matched filter's
    per-bin SNR threshold accepts ~30 above-4σ bins per side of a
    SNR=10⁴ Lorentzian (the skirt extent scales as √SNR); the Sav-Gol
    concavity test rejects monotonic skirts at any SNR. This figure
    answers: does the hybrid stay clean at high SNR, or does it blow up
    too?
    """
    detectors = ("matched_filter", "mf_concavity", "production_like",
                 "production_two_pass", "production_with_mf_gap")
    fig, axes = plt.subplots(1, len(detectors), figsize=(5 * len(detectors), 5))
    fps = [_grid_fp(grids[d]) for d in detectors]
    vmax = float(max(np.nanmax(a) for a in fps))
    for col, det in enumerate(detectors):
        _heatmap(
            fps[col], axes[col], vmin=0.0, vmax=vmax,
            cmap="inferno", fmt="{:.0f}",
            title=f"{DETECTOR_LABELS[det]}\ntotal FPs",
            snr_grid=SNR_GRID_HIGH, fwhm_grid=FWHM_GRID,
        )
    fig.suptitle(f"High-SNR FP scaling{suffix}  (SNR up to 10⁴)")
    fig.tight_layout()
    out = FIG / f"09_high_snr_fp_scaling{suffix}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", out)


def figure_screen_auc(grids: dict[str, np.ndarray], suffix: str = "") -> None:
    """3-panel heatmap: screen AUC against all-FPs across three detectors."""
    detectors = list(DETECTOR_LABELS.keys())
    fig, axes = plt.subplots(1, len(detectors), figsize=(5 * len(detectors), 6))
    for col, det in enumerate(detectors):
        arr = np.array(
            [[c.screen_auc_all for c in row] for row in grids[det]], dtype=float
        )
        _heatmap(
            arr, axes[col],
            vmin=0.4, vmax=1.0, cmap="viridis", fmt="{:.2f}",
            title=f"{DETECTOR_LABELS[det]}\nscreen AUC (TP vs all FPs)",
        )
    fig.suptitle(f"Screen AUC by detector{suffix}")
    fig.tight_layout()
    out = FIG / f"04_screen_auc_by_detector{suffix}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", out)


# ===========================================================================
# Section 4: 2638 application
# ===========================================================================
def auto_calibrate_tau_basis(
    fid_samples: np.ndarray,
    sample_dt_us: float,
    *,
    start_us: float,
    end_us: float,
    probe_freq_mhz: float,
    sideband: Sideband,
    n_padded: int,
    expf_us: float,
    top_n: int = 10,
) -> Tuple[float, float, np.ndarray]:
    """Per-experiment τ_basis auto-calibration.

    Two-pass procedure:
    1. Build the active-FT at the production-canonical apodization
       (expf_us, the Stage 1 setting), find the top_n brightest
       candidates (no SavGol, just |X| local maxima above 10σ).
    2. For each, fit a Lorentzian half-width by walking outward
       from the bin until |X| crosses half-max. The median full-width
       gives ``τ_eff_observed = 1 / (π · FWHM_observed)``.
    3. The matched-filter τ_basis is set to 2 × τ_eff_observed (the
       screen study's recommended operating point for the screen;
       co-opted here as the matched-filter detector's working point).

    Returns ``(tau_basis_us, tau_eff_observed_us, fwhm_observed_mhz_array)``.
    """
    # First pass: detect on the exp-apodized active-FT (matched filter at expf_us).
    # Build the windowed active FT by multiplying the active-region samples before
    # passing to compute_active_ft (expf_us knob removed from API).
    fid_arr_ac = np.asarray(fid_samples, dtype=float)
    time_us_ac = np.arange(fid_arr_ac.size) * sample_dt_us
    start_idx_ac = int(np.searchsorted(time_us_ac, start_us))
    end_idx_ac = int(np.searchsorted(time_us_ac, end_us))
    end_idx_ac = min(end_idx_ac, fid_arr_ac.size)
    n_active_ac = end_idx_ac - start_idx_ac
    t_rel_ac = np.arange(n_active_ac) * sample_dt_us
    w_ac = matched_filter_window(t_rel_ac, expf_us, shape="lorentzian")
    fid_windowed_ac = fid_arr_ac.copy()
    fid_windowed_ac[start_idx_ac:end_idx_ac] *= w_ac
    fid_windowed_ac[:start_idx_ac] = 0.0
    fid_windowed_ac[end_idx_ac:] = 0.0

    active = compute_active_ft(
        fid=fid_windowed_ac,
        sample_dt_us=sample_dt_us,
        start_us=start_us,
        end_us=end_us,
        probe_freq_mhz=probe_freq_mhz,
        sideband=sideband,
        n_padded=n_padded,
        rdc=True,
    )
    mag = np.abs(active.complex_spectrum)
    noise = estimate_noise_scatter(active.freq_mhz, mag)
    sigma = np.asarray(noise.rms_noise, dtype=float)
    sigma_c = sigma / np.sqrt(2.0)
    snr = np.where(sigma_c > 0.0, mag / sigma_c, 0.0)
    cand_bins, _ = find_peaks(snr, height=10.0, distance=2)
    if cand_bins.size == 0:
        # Fallback: 2 × expf_us. The mild apodization is a defensible
        # default when no strong line is bright enough to fit.
        return 2.0 * expf_us, expf_us, np.array([])
    # Top-N brightest by per-bin SNR
    order = np.argsort(-snr[cand_bins])
    top = cand_bins[order][:top_n]

    bin_mhz = abs(active.freq_mhz[1] - active.freq_mhz[0])
    fwhms_mhz: list[float] = []
    for b in top:
        # Half-max walk on the magnitude
        half = 0.5 * mag[b]
        lo = b
        while lo > 0 and mag[lo] > half:
            lo -= 1
        hi = b
        while hi < mag.size - 1 and mag[hi] > half:
            hi += 1
        fwhms_mhz.append(float((hi - lo) * bin_mhz))
    fwhms_arr = np.array(fwhms_mhz, dtype=float)
    fwhm_med = float(np.median(fwhms_arr))
    if fwhm_med <= 0.0:
        return 2.0 * expf_us, expf_us, fwhms_arr
    tau_eff = 1.0 / (np.pi * fwhm_med)
    return 2.0 * tau_eff, tau_eff, fwhms_arr


def _candidate_diff_2638(
    cand_freqs_mf: np.ndarray,
    cand_freqs_pl: np.ndarray,
    cand_freqs_ratios_mf: np.ndarray,
    cand_freqs_ratios_pl: np.ndarray,
    fitted_freqs: np.ndarray,
    bin_mhz: float,
    fwhm_active_bins: float,
) -> dict:
    """Set-theoretic comparison of matched-filter, production, and fit.

    Match within ±FWHM/2 on the active-FT grid. Returns a dict with
    intersection / mf-only / pl-only frequency arrays, the screen-ratio
    distribution for each set, and the fit-cross-reference counts.
    """
    tol_mhz = max(1.0, fwhm_active_bins / 2.0) * bin_mhz

    def _match(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Bool mask over a: True iff some entry of b is within tol_mhz."""
        if a.size == 0 or b.size == 0:
            return np.zeros(a.size, dtype=bool)
        out = np.zeros(a.size, dtype=bool)
        for i, x in enumerate(a):
            if np.any(np.abs(b - x) <= tol_mhz):
                out[i] = True
        return out

    mf_in_pl = _match(cand_freqs_mf, cand_freqs_pl)
    pl_in_mf = _match(cand_freqs_pl, cand_freqs_mf)
    both = cand_freqs_mf[mf_in_pl]
    mf_only = cand_freqs_mf[~mf_in_pl]
    pl_only = cand_freqs_pl[~pl_in_mf]
    mf_only_ratios = cand_freqs_ratios_mf[~mf_in_pl]
    pl_only_ratios = cand_freqs_ratios_pl[~pl_in_mf]

    # Cross-reference against the persisted Stage 5 fit.
    fit_in_mf = _match(fitted_freqs, cand_freqs_mf)
    fit_in_pl = _match(fitted_freqs, cand_freqs_pl)
    return {
        "n_both": int(both.size),
        "n_mf_only": int(mf_only.size),
        "n_pl_only": int(pl_only.size),
        "both_freqs": both,
        "mf_only_freqs": mf_only,
        "pl_only_freqs": pl_only,
        "mf_only_ratios": mf_only_ratios,
        "pl_only_ratios": pl_only_ratios,
        "n_fit_total": int(fitted_freqs.size),
        "n_fit_in_mf": int(np.sum(fit_in_mf)),
        "n_fit_in_pl": int(np.sum(fit_in_pl)),
        "tol_mhz": float(tol_mhz),
    }


def run_2638_application() -> Optional[dict]:
    """Apply the matched filter + screen to the 2638 fixture.

    Returns a dict with the comparison statistics, or None if the
    fixture is not present.
    """
    if not FTMW_PATH.exists():
        logger.info("2638 fixture missing at %s, skipping 2638 section", FTMW_PATH)
        return None

    # Use the api / pipeline to avoid duplicating Stage 1 setup.
    import ftmwpipeline.api as ftmw
    from ftmwpipeline._internal.stage5_impl import _build_active_ft_inputs

    (
        fid_samples,
        sample_dt_us,
        start_us,
        end_us,
        probe_freq_mhz,
        sideband,
        n_padded,
        acquisition_us,
        user_ft,
        trim_range,
    ) = _build_active_ft_inputs(str(FTMW_PATH))

    logger.info(
        "2638: start=%.2f end=%.2f acq=%.3f us  probe=%.1f MHz  zpf-padded=%d",
        start_us, end_us, acquisition_us, probe_freq_mhz, n_padded,
    )

    # The user-facing spectrum is trimmed to the Stage 1 trim range
    # (26500–40000 MHz on 2638); the active-FT alone covers the full
    # [probe ± fs/2] band, so candidates outside the trim region are
    # noise-only and irrelevant to the production comparison. Pull the
    # trim from the canonical trim_range returned by _build_active_ft_inputs
    # (falls back to the user-FT axis bounds for pre-trim fixtures).
    if trim_range is not None:
        trim_lo, trim_hi = float(trim_range[0]), float(trim_range[1])
    else:
        trim_lo = float(user_ft.freq_array.min())
        trim_hi = float(user_ft.freq_array.max())
    logger.info("2638: trim range = (%.1f, %.1f) MHz", trim_lo, trim_hi)

    # Step 1: auto-calibrate τ_basis using expf_us=5.0 (the historical default
    # Stage 1 apodization for 2638; the auto-calibrate first-pass apodizes
    # the active region to find bright candidates).
    _expf_us_default = 5.0
    tau_basis, tau_eff, fwhms_obs = auto_calibrate_tau_basis(
        fid_samples,
        sample_dt_us,
        start_us=start_us,
        end_us=end_us,
        probe_freq_mhz=probe_freq_mhz,
        sideband=sideband,
        n_padded=n_padded,
        expf_us=_expf_us_default,
        top_n=10,
    )
    logger.info(
        "2638: auto-τ:  τ_eff_observed=%.3f µs,  τ_basis=2·τ_eff=%.3f µs  "
        "(top-N FWHM median=%.4f MHz)",
        tau_eff, tau_basis,
        float(np.median(fwhms_obs)) if fwhms_obs.size else float("nan"),
    )

    # Step 2: matched-filter detection on the active-FT with τ_basis.
    mf = matched_filter_detect(
        fid_samples,
        sample_dt_us,
        start_us=start_us,
        end_us=end_us,
        tau_basis_us=tau_basis,
        probe_freq_mhz=probe_freq_mhz,
        sideband=sideband,
        n_padded=n_padded,
        detection_snr=4.0,
        sigma=None,  # adaptive estimator on the resulting active-FT
        min_separation_bins=3,
    )
    # Restrict candidates to the persisted trim band -- everything outside
    # is noise-only and not part of the production comparison.
    cand_freqs_mf_all = mf.active_ft.freq_mhz[mf.candidate_bins]
    in_trim = (cand_freqs_mf_all >= trim_lo) & (cand_freqs_mf_all <= trim_hi)
    cand_freqs_mf = cand_freqs_mf_all[in_trim]
    cand_bins_mf_trimmed = mf.candidate_bins[in_trim]
    logger.info(
        "2638: matched filter at τ_basis=%.3f µs found %d candidates total, "
        "%d inside the trim band (thr=4σ)",
        tau_basis, mf.candidate_bins.size, cand_freqs_mf.size,
    )
    projections_mf = project_candidates(
        mf.active_ft.freq_mhz,
        mf.active_ft.complex_spectrum,
        mf.sigma,
        cand_freqs_mf.tolist(),
        tau_us=tau_basis,
        acquisition_us=acquisition_us,
        sideband=sideband,
    )
    ratios_mf = np.array([p.ratio for p in projections_mf])

    # Step 4: load production Stage 3 peaks + persisted Stage 5 fit.
    prod_peaks = ftmw.load_peaks(str(FTMW_PATH))
    fit = ftmw.load_fit(str(FTMW_PATH))
    prod_freqs = np.array([p.frequency for p in prod_peaks if p.properties.get("promoted")])
    # Persisted Stage 5 fitted peak list:
    fitted_freqs = np.array([p.frequency_mhz for p in fit.fitted_peaks])
    logger.info(
        "2638: production promoted peaks=%d, persisted fit peaks=%d",
        prod_freqs.size, fitted_freqs.size,
    )

    # Production candidates: re-snap to the *active-FT* grid (so the diff
    # uses the same frequency axis as the matched filter). Production
    # peaks are on the user grid; we need to re-locate them on the
    # active-FT grid by nearest-neighbour. Restrict to the trim band
    # (production peaks are already inside it, but be defensive).
    prod_freqs = prod_freqs[(prod_freqs >= trim_lo) & (prod_freqs <= trim_hi)]
    fitted_freqs = fitted_freqs[(fitted_freqs >= trim_lo) & (fitted_freqs <= trim_hi)]
    mf_freq_asc = np.sort(mf.active_ft.freq_mhz)
    nearest = lambda fs: np.array([mf_freq_asc[np.argmin(np.abs(mf_freq_asc - f))] for f in fs])
    prod_freqs_on_active = nearest(prod_freqs)
    fitted_freqs_on_active = nearest(fitted_freqs)
    cand_freqs_pl_active = prod_freqs_on_active

    # For the production set, run the screen on the active-FT at the
    # candidates' active-FT bins so the post-screen comparison is
    # apples-to-apples.
    projections_pl = project_candidates(
        mf.active_ft.freq_mhz,
        mf.active_ft.complex_spectrum,
        mf.sigma,
        cand_freqs_pl_active.tolist(),
        tau_us=tau_basis,
        acquisition_us=acquisition_us,
        sideband=sideband,
    )
    ratios_pl = np.array([p.ratio for p in projections_pl])

    # Step 5: set-theoretic diff
    bin_mhz = abs(mf.active_ft.freq_mhz[1] - mf.active_ft.freq_mhz[0])
    fwhm_active = 1.0 / (np.pi * tau_eff * bin_mhz)  # in bin units
    diff = _candidate_diff_2638(
        np.asarray(cand_freqs_mf, dtype=float),
        np.asarray(cand_freqs_pl_active, dtype=float),
        ratios_mf,
        ratios_pl,
        fitted_freqs_on_active,
        bin_mhz,
        fwhm_active,
    )
    logger.info(
        "2638: diff  both=%d  mf_only=%d  pl_only=%d  "
        "fit_in_mf=%d/%d  fit_in_pl=%d/%d  tol=%.4f MHz",
        diff["n_both"], diff["n_mf_only"], diff["n_pl_only"],
        diff["n_fit_in_mf"], diff["n_fit_total"],
        diff["n_fit_in_pl"], diff["n_fit_total"],
        diff["tol_mhz"],
    )

    # Apply a screen threshold to each detector and report post-screen.
    SCREEN_THR = 0.6
    keep_mf = ratios_mf >= SCREEN_THR
    keep_pl = ratios_pl >= SCREEN_THR
    cand_freqs_mf_kept = cand_freqs_mf[keep_mf]
    cand_freqs_pl_kept = cand_freqs_pl_active[keep_pl]
    fit_in_mf_kept = sum(
        np.any(np.abs(cand_freqs_mf_kept - f) <= diff["tol_mhz"])
        for f in fitted_freqs_on_active
    )
    fit_in_pl_kept = sum(
        np.any(np.abs(cand_freqs_pl_kept - f) <= diff["tol_mhz"])
        for f in fitted_freqs_on_active
    )
    logger.info(
        "2638: post-screen (ratio>=%.2f)  mf_kept=%d/%d  pl_kept=%d/%d  "
        "fit_in_mf_kept=%d  fit_in_pl_kept=%d",
        SCREEN_THR,
        int(keep_mf.sum()), keep_mf.size,
        int(keep_pl.sum()), keep_pl.size,
        fit_in_mf_kept, fit_in_pl_kept,
    )

    # Persist the 2638 candidate table for downstream visual inspection.
    table_path = REPO_ROOT / "scratch" / "matched-filter-detection" / "2638_candidates.npz"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        table_path,
        cand_freqs_mf=np.asarray(cand_freqs_mf),
        ratios_mf=ratios_mf,
        keep_mf=keep_mf,
        cand_freqs_pl_active=cand_freqs_pl_active,
        ratios_pl=ratios_pl,
        keep_pl=keep_pl,
        fitted_freqs_on_active=fitted_freqs_on_active,
        tau_basis=tau_basis,
        tau_eff=tau_eff,
        screen_threshold=SCREEN_THR,
        tol_mhz=diff["tol_mhz"],
    )
    logger.info("2638: candidate table written to %s", table_path)

    return {
        "tau_basis": tau_basis,
        "tau_eff": tau_eff,
        "n_mf": int(cand_freqs_mf.size),
        "n_pl": int(cand_freqs_pl_active.size),
        "n_both": diff["n_both"],
        "n_mf_only": diff["n_mf_only"],
        "n_pl_only": diff["n_pl_only"],
        "n_fit_total": diff["n_fit_total"],
        "n_fit_in_mf": diff["n_fit_in_mf"],
        "n_fit_in_pl": diff["n_fit_in_pl"],
        "n_fit_in_mf_kept": int(fit_in_mf_kept),
        "n_fit_in_pl_kept": int(fit_in_pl_kept),
        "n_mf_kept": int(keep_mf.sum()),
        "n_pl_kept": int(keep_pl.sum()),
        "screen_threshold": SCREEN_THR,
        "tol_mhz": diff["tol_mhz"],
    }


def figure_2638_ratio_histogram() -> None:
    """Histogram of screen ratios on 2638, split by proximity to fit peaks.

    A clean separation would put "near-fit-peak" candidates at ratio ≈ 1
    and "not-near-fit-peak" at ratio < 0.6 -- the screen threshold the
    synthetic study calibrated. On 2638 the two populations overlap, so
    the screen cannot kill the surplus matched-filter candidates.
    """
    if not FTMW_PATH.exists():
        return
    import ftmwpipeline.api as ftmw
    from ftmwpipeline._internal.stage5_impl import _build_active_ft_inputs

    inputs = _build_active_ft_inputs(str(FTMW_PATH))
    (
        fid_samples, sample_dt_us, start_us, end_us, probe_freq_mhz,
        sideband, n_padded, acquisition_us, user_ft, trim_range,
    ) = inputs
    if trim_range is not None:
        trim_lo, trim_hi = float(trim_range[0]), float(trim_range[1])
    else:
        trim_lo = float(user_ft.freq_array.min())
        trim_hi = float(user_ft.freq_array.max())
    fit = ftmw.load_fit(str(FTMW_PATH))
    fitted_freqs = np.array([p.frequency_mhz for p in fit.fitted_peaks])
    fitted_freqs = fitted_freqs[(fitted_freqs >= trim_lo) & (fitted_freqs <= trim_hi)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, tau_basis in zip(axes, [2.30, 6.0]):
        mf = matched_filter_detect(
            fid_samples, sample_dt_us, start_us=start_us, end_us=end_us,
            tau_basis_us=tau_basis, probe_freq_mhz=probe_freq_mhz,
            sideband=sideband, n_padded=n_padded, detection_snr=4.0,
            min_separation_bins=3,
        )
        cand_freqs = mf.active_ft.freq_mhz[mf.candidate_bins]
        in_trim = (cand_freqs >= trim_lo) & (cand_freqs <= trim_hi)
        cand_freqs = cand_freqs[in_trim]
        cand_bins_trim = mf.candidate_bins[in_trim]
        projections = project_candidates(
            mf.active_ft.freq_mhz, mf.active_ft.complex_spectrum, mf.sigma,
            cand_freqs.tolist(), tau_us=tau_basis,
            acquisition_us=acquisition_us, sideband=sideband,
        )
        ratios = np.array([p.ratio for p in projections])
        bin_mhz = abs(mf.active_ft.freq_mhz[1] - mf.active_ft.freq_mhz[0])
        tol = bin_mhz * 2.0
        fit_match = np.array(
            [np.any(np.abs(fitted_freqs - cf) <= tol) for cf in cand_freqs]
        )
        bins_h = np.linspace(0.5, 1.6, 40)
        ax.hist(ratios[fit_match], bins=bins_h, alpha=0.6, color="C2",
                label=f"near fit peak (n={int(fit_match.sum())})")
        ax.hist(ratios[~fit_match], bins=bins_h, alpha=0.6, color="C3",
                label=f"not near fit peak (n={int((~fit_match).sum())})")
        ax.axvline(0.6, ls="--", color="k", alpha=0.5, label="thr=0.6")
        ax.set_xlabel("screen ratio")
        ax.set_ylabel("candidate count")
        ax.set_title(f"τ_basis = {tau_basis:.2f} µs, n_cand = {cand_freqs.size}")
        ax.legend(loc="upper left", fontsize=9)
    fig.suptitle(
        "2638 screen-ratio distributions: candidates near a fit peak vs not"
    )
    fig.tight_layout()
    out = FIG / "06_2638_ratio_histogram.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", out)


def figure_2638_threshold_sweep() -> None:
    """Trade-off curve: detection threshold vs (fit recall, candidate count).

    Sweeps ``detection_snr`` from 2 to 15 σ at two τ_basis settings and
    compares against the production primary's promoted-peak count and
    fit recall. Shows where the matched filter would have to operate to
    match production -- and the cost in recall.
    """
    if not FTMW_PATH.exists():
        return
    import ftmwpipeline.api as ftmw
    from ftmwpipeline._internal.stage5_impl import _build_active_ft_inputs

    inputs = _build_active_ft_inputs(str(FTMW_PATH))
    (
        fid_samples, sample_dt_us, start_us, end_us, probe_freq_mhz,
        sideband, n_padded, acquisition_us, user_ft, trim_range,
    ) = inputs
    if trim_range is not None:
        trim_lo, trim_hi = float(trim_range[0]), float(trim_range[1])
    else:
        trim_lo = float(user_ft.freq_array.min())
        trim_hi = float(user_ft.freq_array.max())
    prod_peaks = ftmw.load_peaks(str(FTMW_PATH))
    fit = ftmw.load_fit(str(FTMW_PATH))
    prod_freqs = np.array(
        [p.frequency for p in prod_peaks if p.properties.get("promoted")]
    )
    fitted_freqs = np.array([p.frequency_mhz for p in fit.fitted_peaks])
    prod_freqs = prod_freqs[(prod_freqs >= trim_lo) & (prod_freqs <= trim_hi)]
    fitted_freqs = fitted_freqs[(fitted_freqs >= trim_lo) & (fitted_freqs <= trim_hi)]
    # Production fit recall against itself (sanity reference, often ~1)
    n_prod_in_fit = sum(
        1 for f in fitted_freqs if np.any(np.abs(prod_freqs - f) <= 0.5)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    thresholds = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0])
    for ax, tau_basis in zip(axes, [2.30, 6.0]):
        nc, recall = [], []
        for thr in thresholds:
            mf = matched_filter_detect(
                fid_samples, sample_dt_us, start_us=start_us, end_us=end_us,
                tau_basis_us=tau_basis, probe_freq_mhz=probe_freq_mhz,
                sideband=sideband, n_padded=n_padded, detection_snr=float(thr),
                min_separation_bins=3,
            )
            cf = mf.active_ft.freq_mhz[mf.candidate_bins]
            cf = cf[(cf >= trim_lo) & (cf <= trim_hi)]
            bin_mhz = abs(mf.active_ft.freq_mhz[1] - mf.active_ft.freq_mhz[0])
            tol = bin_mhz
            recovered = sum(
                1 for f in fitted_freqs if np.any(np.abs(cf - f) <= tol)
            )
            nc.append(cf.size)
            recall.append(recovered / fitted_freqs.size)
        ax2 = ax.twinx()
        ax.plot(thresholds, recall, marker="o", color="C0", label="MF fit recall")
        ax2.plot(thresholds, nc, marker="s", color="C3", label="MF n_candidates")
        ax.axhline(
            n_prod_in_fit / fitted_freqs.size,
            color="C0", ls="--", alpha=0.4,
            label=f"production fit recall ({n_prod_in_fit}/{fitted_freqs.size})",
        )
        ax2.axhline(
            prod_freqs.size, color="C3", ls="--", alpha=0.4,
            label=f"production n_promoted ({prod_freqs.size})",
        )
        ax.set_xlabel("detection threshold (σ)")
        ax.set_ylabel("MF fit recall", color="C0")
        ax2.set_ylabel("MF n_candidates", color="C3")
        ax.set_title(f"τ_basis = {tau_basis:.2f} µs")
        ax.set_ylim(0, 1.0)
        ax2.set_yscale("log")
        ax.legend(loc="upper right", fontsize=9)
        ax2.legend(loc="center right", fontsize=9)
    fig.suptitle("2638: matched-filter detection threshold trade-off")
    fig.tight_layout()
    out = FIG / "07_2638_threshold_sweep.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", out)


def run_2638_hybrid_application() -> Optional[dict]:
    """Apply the hybrid + production-with-MF-gap variants to 2638.

    Mirrors ``run_2638_application`` but for the new detector variants
    flagged by the matched-filter report's §6.3. Reports each variant's
    candidate count, fit-peak recovery, and where it sits relative to
    production and pure MF.
    """
    if not FTMW_PATH.exists():
        logger.info("2638 fixture missing, skipping hybrid 2638 section")
        return None
    import ftmwpipeline.api as ftmw
    from ftmwpipeline._internal.stage5_impl import _build_active_ft_inputs

    inputs = _build_active_ft_inputs(str(FTMW_PATH))
    (
        fid_samples, sample_dt_us, start_us, end_us, probe_freq_mhz,
        sideband, n_padded, acquisition_us, user_ft, trim_range,
    ) = inputs
    if trim_range is not None:
        trim_lo, trim_hi = float(trim_range[0]), float(trim_range[1])
    else:
        trim_lo = float(user_ft.freq_array.min())
        trim_hi = float(user_ft.freq_array.max())
    prod_peaks = ftmw.load_peaks(str(FTMW_PATH))
    fit = ftmw.load_fit(str(FTMW_PATH))
    prod_freqs = np.array(
        [p.frequency for p in prod_peaks if p.properties.get("promoted")]
    )
    fitted_freqs = np.array([p.frequency_mhz for p in fit.fitted_peaks])
    prod_freqs = prod_freqs[(prod_freqs >= trim_lo) & (prod_freqs <= trim_hi)]
    fitted_freqs = fitted_freqs[(fitted_freqs >= trim_lo) & (fitted_freqs <= trim_hi)]
    n_fit = fitted_freqs.size
    n_prod = prod_freqs.size

    summary: dict = {
        "trim_lo": trim_lo,
        "trim_hi": trim_hi,
        "n_fit": n_fit,
        "n_prod": n_prod,
    }

    # Reuse the prompt's recommended τ_basis = 2 × τ_eff (3 µs).
    tau_basis = 6.0

    # 1) Hybrid: MF apodization + Sav-Gol concavity locator
    hyb = matched_filter_concavity_detect(
        fid_samples, sample_dt_us,
        start_us=start_us, end_us=end_us,
        tau_basis_us=tau_basis,
        probe_freq_mhz=probe_freq_mhz, sideband=sideband,
        n_padded=n_padded, sg_window=11, sg_order=3, min_snr=2.0,
        sigma=None,
    )
    cf_hyb = hyb.active_ft.freq_mhz[hyb.candidate_bins]
    cf_hyb = cf_hyb[(cf_hyb >= trim_lo) & (cf_hyb <= trim_hi)]
    bin_mhz = abs(hyb.active_ft.freq_mhz[1] - hyb.active_ft.freq_mhz[0])
    tol = bin_mhz
    fit_in_hyb = sum(1 for f in fitted_freqs if np.any(np.abs(cf_hyb - f) <= tol))
    logger.info(
        "2638 hybrid (MF+SavGol, τ_basis=%.2f µs): n_cand=%d  fit_in=%d/%d",
        tau_basis, cf_hyb.size, fit_in_hyb, n_fit,
    )
    summary["hybrid_tau_basis"] = tau_basis
    summary["hybrid_n_cand"] = int(cf_hyb.size)
    summary["hybrid_n_fit_in"] = int(fit_in_hyb)

    # 2) Production primary + MF gap. Production primary on the user grid
    # is the canonical Stage 3 design; we use ``load_peaks`` for it.
    # MF gap = matched_filter_concavity_detect on the active-FT.
    cf_mf_gap = cf_hyb  # same single-pass call as hybrid -- no leakage mask
    # The union of production peaks + hybrid candidates on the active-FT grid
    prod_freqs_on_active = np.array([
        hyb.active_ft.freq_mhz[np.argmin(np.abs(hyb.active_ft.freq_mhz - f))]
        for f in prod_freqs
    ])
    union = np.unique(
        np.concatenate(
            [
                np.round(prod_freqs_on_active / bin_mhz).astype(int),
                np.round(cf_hyb / bin_mhz).astype(int),
            ]
        )
    )
    # Translate back to MHz approximately: the union is in "bin index" space.
    cf_union = union.astype(float) * bin_mhz
    # That works only approximately; for fit-recovery we just need to know
    # whether each fit peak is in either set.
    fit_in_union = sum(
        1 for f in fitted_freqs
        if np.any(np.abs(prod_freqs - f) <= 0.5)
        or np.any(np.abs(cf_hyb - f) <= tol)
    )
    n_union = int(prod_freqs.size + cf_hyb.size - sum(
        1 for f in cf_hyb if np.any(np.abs(prod_freqs - f) <= tol)
    ))
    logger.info(
        "2638 prod+MF-gap union: n_cand≈%d  fit_in_union=%d/%d",
        n_union, fit_in_union, n_fit,
    )
    summary["prod_mf_gap_n_cand"] = int(n_union)
    summary["prod_mf_gap_n_fit_in"] = int(fit_in_union)

    # Headline comparison table
    logger.info(
        "2638 summary (τ_basis=%.2f µs):  "
        "production %d/%d (%.1f%%, n_promoted=%d),  "
        "pure MF %d/%d (%.1f%%, n_cand≈3449),  "
        "hybrid %d/%d (%.1f%%, n_cand=%d),  "
        "prod+MF-gap union %d/%d (%.1f%%, n_cand≈%d)",
        tau_basis,
        sum(1 for f in fitted_freqs if np.any(np.abs(prod_freqs - f) <= 0.5)),
        n_fit, 100 * sum(1 for f in fitted_freqs if np.any(np.abs(prod_freqs - f) <= 0.5)) / n_fit,
        n_prod,
        # MF numbers come from run_2638_application; quote indicative values
        577, n_fit, 100 * 577 / n_fit,
        fit_in_hyb, n_fit, 100 * fit_in_hyb / n_fit, cf_hyb.size,
        fit_in_union, n_fit, 100 * fit_in_union / n_fit, n_union,
    )
    return summary


def figure_2638_hybrid_comparison() -> None:
    """Bar chart: candidate count vs fit-peak recovery for each variant."""
    if not FTMW_PATH.exists():
        return
    import ftmwpipeline.api as ftmw
    from ftmwpipeline._internal.stage5_impl import _build_active_ft_inputs

    inputs = _build_active_ft_inputs(str(FTMW_PATH))
    (
        fid_samples, sample_dt_us, start_us, end_us, probe_freq_mhz,
        sideband, n_padded, acquisition_us, user_ft, trim_range,
    ) = inputs
    if trim_range is not None:
        trim_lo, trim_hi = float(trim_range[0]), float(trim_range[1])
    else:
        trim_lo = float(user_ft.freq_array.min())
        trim_hi = float(user_ft.freq_array.max())
    prod_peaks = ftmw.load_peaks(str(FTMW_PATH))
    fit = ftmw.load_fit(str(FTMW_PATH))
    prod_freqs = np.array(
        [p.frequency for p in prod_peaks if p.properties.get("promoted")]
    )
    fitted_freqs = np.array([p.frequency_mhz for p in fit.fitted_peaks])
    prod_freqs = prod_freqs[(prod_freqs >= trim_lo) & (prod_freqs <= trim_hi)]
    fitted_freqs = fitted_freqs[(fitted_freqs >= trim_lo) & (fitted_freqs <= trim_hi)]
    n_fit = fitted_freqs.size

    tau_basis = 6.0

    # Pure MF
    mf = matched_filter_detect(
        fid_samples, sample_dt_us, start_us=start_us, end_us=end_us,
        tau_basis_us=tau_basis, probe_freq_mhz=probe_freq_mhz, sideband=sideband,
        n_padded=n_padded, detection_snr=4.0, min_separation_bins=3,
    )
    cf_mf = mf.active_ft.freq_mhz[mf.candidate_bins]
    cf_mf = cf_mf[(cf_mf >= trim_lo) & (cf_mf <= trim_hi)]
    bin_mhz = abs(mf.active_ft.freq_mhz[1] - mf.active_ft.freq_mhz[0])
    tol = bin_mhz
    fit_in_mf = sum(1 for f in fitted_freqs if np.any(np.abs(cf_mf - f) <= tol))

    # Hybrid
    hyb = matched_filter_concavity_detect(
        fid_samples, sample_dt_us, start_us=start_us, end_us=end_us,
        tau_basis_us=tau_basis,
        probe_freq_mhz=probe_freq_mhz, sideband=sideband,
        n_padded=n_padded, sg_window=11, sg_order=3, min_snr=2.0,
    )
    cf_hyb = hyb.active_ft.freq_mhz[hyb.candidate_bins]
    cf_hyb = cf_hyb[(cf_hyb >= trim_lo) & (cf_hyb <= trim_hi)]
    fit_in_hyb = sum(1 for f in fitted_freqs if np.any(np.abs(cf_hyb - f) <= tol))

    # Production primary (counts as one variant)
    fit_in_prod = sum(1 for f in fitted_freqs if np.any(np.abs(prod_freqs - f) <= 0.5))

    # Prod + MF gap union (union of production peaks + hybrid candidates)
    in_prod = np.array(
        [np.any(np.abs(prod_freqs - f) <= 0.5) for f in cf_hyb]
    )
    n_union = int(prod_freqs.size + cf_hyb.size - in_prod.sum())
    fit_in_union = sum(
        1 for f in fitted_freqs
        if np.any(np.abs(prod_freqs - f) <= 0.5)
        or np.any(np.abs(cf_hyb - f) <= tol)
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    variants = ["production\nprimary", "pure MF", "hybrid\n(MF+SavGol)", "prod + MF\ngap"]
    n_cand = [int(prod_freqs.size), int(cf_mf.size), int(cf_hyb.size), n_union]
    n_fit_in = [fit_in_prod, fit_in_mf, fit_in_hyb, fit_in_union]
    colors = ["C0", "C3", "C2", "C1"]

    ax = axes[0]
    bars = ax.bar(variants, n_cand, color=colors)
    ax.set_ylabel("candidate count")
    ax.set_yscale("log")
    ax.set_title("Candidate count (lower is cleaner)")
    for b, v in zip(bars, n_cand):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.1, str(v),
                ha="center", va="bottom", fontsize=10)
    ax = axes[1]
    recall = [v / n_fit for v in n_fit_in]
    bars = ax.bar(variants, recall, color=colors)
    ax.set_ylabel(f"fit recall (of {n_fit} fitted peaks)")
    ax.axhline(1.0, color="k", ls=":", alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.set_title("Fit-peak recovery (higher is better)")
    for b, v, n in zip(bars, recall, n_fit_in):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02,
                f"{v:.2f}\n({n}/{n_fit})",
                ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        f"2638: hybrid vs production vs pure MF  (τ_basis = {tau_basis:.1f} µs)"
    )
    fig.tight_layout()
    out = FIG / "08_2638_hybrid_comparison.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", out)


def figure_2638_candidate_overlay() -> None:
    """Overlay matched-filter, production, and fit candidates on the active-FT."""
    if not FTMW_PATH.exists():
        return
    table_path = REPO_ROOT / "scratch" / "matched-filter-detection" / "2638_candidates.npz"
    if not table_path.exists():
        logger.info("2638 candidate table missing -- run run_2638_application first")
        return
    d = np.load(table_path, allow_pickle=True)
    cand_freqs_mf = d["cand_freqs_mf"]
    ratios_mf = d["ratios_mf"]
    cand_freqs_pl = d["cand_freqs_pl_active"]
    ratios_pl = d["ratios_pl"]
    fit_freqs = d["fitted_freqs_on_active"]
    tau_basis = float(d["tau_basis"])
    thr = float(d["screen_threshold"])

    # Rebuild the active-FT for the panel
    from ftmwpipeline._internal.stage5_impl import _build_active_ft_inputs

    (
        fid_samples,
        sample_dt_us,
        start_us,
        end_us,
        probe_freq_mhz,
        sideband,
        n_padded,
        _acq,
        _user_ft,
        _trim_range,
    ) = _build_active_ft_inputs(str(FTMW_PATH))
    mf = matched_filter_detect(
        fid_samples,
        sample_dt_us,
        start_us=start_us,
        end_us=end_us,
        tau_basis_us=tau_basis,
        probe_freq_mhz=probe_freq_mhz,
        sideband=sideband,
        n_padded=n_padded,
        detection_snr=4.0,
    )
    freq_asc = np.sort(mf.active_ft.freq_mhz)
    mag_asc = np.abs(mf.active_ft.complex_spectrum[np.argsort(mf.active_ft.freq_mhz)])
    snr_asc = mf.per_bin_snr[np.argsort(mf.active_ft.freq_mhz)]

    # Pick a representative band: full sweep is hard to read, so two zoom panels.
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    bands = [(26500, 33000), (33000, 40000)]
    for ax, (lo, hi) in zip(axes, bands):
        sel = (freq_asc >= lo) & (freq_asc <= hi)
        ax.plot(freq_asc[sel], snr_asc[sel], color="C0", lw=0.5, alpha=0.7, label="MF per-bin SNR")
        ax.axhline(4.0, color="C0", ls="--", alpha=0.4, label="thr=4σ")
        keep_mf = ratios_mf >= thr
        in_band = (cand_freqs_mf >= lo) & (cand_freqs_mf <= hi)
        ax.scatter(
            cand_freqs_mf[in_band & keep_mf],
            np.interp(cand_freqs_mf[in_band & keep_mf], freq_asc, snr_asc),
            color="C0", s=18, marker="o",
            label=f"MF kept (ratio≥{thr:.1f})" if lo == bands[0][0] else None,
        )
        ax.scatter(
            cand_freqs_mf[in_band & ~keep_mf],
            np.interp(cand_freqs_mf[in_band & ~keep_mf], freq_asc, snr_asc),
            color="C0", s=12, marker="x", alpha=0.5,
            label="MF dropped by screen" if lo == bands[0][0] else None,
        )
        # Production candidates (re-snapped to active-FT)
        in_band_pl = (cand_freqs_pl >= lo) & (cand_freqs_pl <= hi)
        ax.scatter(
            cand_freqs_pl[in_band_pl],
            np.interp(cand_freqs_pl[in_band_pl], freq_asc, snr_asc),
            color="C3", s=12, marker="+", alpha=0.6,
            label="production peaks" if lo == bands[0][0] else None,
        )
        # Fit peaks
        in_band_fit = (fit_freqs >= lo) & (fit_freqs <= hi)
        for f in fit_freqs[in_band_fit]:
            ax.axvline(f, color="C2", lw=0.4, alpha=0.4)
        ax.set_xlabel("freq (MHz)")
        ax.set_ylabel("MF per-bin SNR")
        ax.set_title(f"{lo}–{hi} MHz")
        if lo == bands[0][0]:
            ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        f"2638 matched filter vs production + fit (τ_basis={tau_basis:.2f} µs, screen ≥ {thr:.1f})"
    )
    fig.tight_layout()
    out = FIG / "05_2638_overlay.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", out)


# ===========================================================================
# Driver
# ===========================================================================
def main(skip_2638: bool = False) -> None:
    t0 = time.time()
    logger.info("section 2: smoke test")
    figure_smoke_test()

    logger.info("section 3a: phase-space sweep (no strong lines)")
    grids_clean = run_sweep(n_trials=4, n_strong_lines=0)
    figure_recall_and_fp(grids_clean, suffix="_clean")
    figure_post_screen(grids_clean, suffix="_clean")
    figure_screen_auc(grids_clean, suffix="_clean")

    logger.info("section 3b: phase-space sweep (4 strong lines, SNR=50)")
    grids_strong = run_sweep(
        n_trials=4, n_strong_lines=4, strong_snr=50.0,
        strong_separation_bins=80.0, sidelobe_range_bins=80.0,
    )
    figure_recall_and_fp(grids_strong, suffix="_strong")
    figure_post_screen(grids_strong, suffix="_strong")
    figure_screen_auc(grids_strong, suffix="_strong")

    # Save sweep results for the report writeup
    def _grid_to_dict(grids: dict[str, np.ndarray]) -> dict:
        out: dict[str, np.ndarray] = {}
        for k, arr in grids.items():
            out[k + "_n_cand"] = np.array(
                [[c.n_candidates for c in row] for row in arr], dtype=int
            )
            out[k + "_recall"] = np.array(
                [[c.n_injected_recovered / max(c.n_injected_total, 1) for c in row] for row in arr]
            )
            out[k + "_fp_noise"] = np.array(
                [[c.n_fp_noise for c in row] for row in arr], dtype=int
            )
            out[k + "_fp_side"] = np.array(
                [[c.n_fp_sidelobe for c in row] for row in arr], dtype=int
            )
            out[k + "_post_recall"] = np.array(
                [[c.n_post_recovered / max(c.n_injected_total, 1) for c in row] for row in arr]
            )
            out[k + "_post_fp_noise"] = np.array(
                [[c.n_post_fp_noise for c in row] for row in arr], dtype=int
            )
            out[k + "_post_fp_side"] = np.array(
                [[c.n_post_fp_sidelobe for c in row] for row in arr], dtype=int
            )
            out[k + "_auc"] = np.array(
                [[c.screen_auc_all for c in row] for row in arr]
            )
        return out
    np.savez(
        DATA / "sweep_clean.npz",
        snr_grid=SNR_GRID, fwhm_grid=FWHM_GRID, **_grid_to_dict(grids_clean),
    )
    np.savez(
        DATA / "sweep_strong.npz",
        snr_grid=SNR_GRID, fwhm_grid=FWHM_GRID, **_grid_to_dict(grids_strong),
    )

    if not skip_2638:
        logger.info("section 4: 2638 application")
        summary = run_2638_application()
        if summary is not None:
            (DATA / "2638_summary.txt").write_text(
                "\n".join(f"{k}: {v}" for k, v in summary.items()) + "\n"
            )
            figure_2638_candidate_overlay()
            figure_2638_ratio_histogram()
            figure_2638_threshold_sweep()
        logger.info("section 5: 2638 hybrid investigation")
        run_2638_hybrid_application()
        figure_2638_hybrid_comparison()

    logger.info("section 6: high-SNR FP scaling")
    grids_highsnr = run_sweep(
        n_trials=3, n_strong_lines=0,
        snr_grid=SNR_GRID_HIGH, fwhm_grid=FWHM_GRID,
    )
    figure_high_snr_fp(grids_highsnr)
    np.savez(
        DATA / "sweep_high_snr.npz",
        snr_grid=SNR_GRID_HIGH, fwhm_grid=FWHM_GRID,
        **{
            f"{k}_n_cand": np.array(
                [[c.n_candidates for c in row] for row in arr], dtype=int
            ) for k, arr in grids_highsnr.items()
        },
        **{
            f"{k}_fp_total": np.array(
                [[c.n_fp_noise + c.n_fp_sidelobe for c in row] for row in arr],
                dtype=int,
            ) for k, arr in grids_highsnr.items()
        },
        **{
            f"{k}_recall": np.array(
                [[c.n_injected_recovered / max(c.n_injected_total, 1)
                  for c in row] for row in arr]
            ) for k, arr in grids_highsnr.items()
        },
    )

    logger.info("done in %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()
