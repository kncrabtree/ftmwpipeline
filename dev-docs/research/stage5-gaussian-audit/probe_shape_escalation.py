"""Shape-escalation diagnostic for the Stage 5 ``shape_error`` bucket.

The Step 5 cross-reference (``report.md`` § Step 5) found the high-chi2r tail
is structural, not knob-tunable, and that the dominant bucket is
``shape_error`` on strong lines: a visually excellent fit reads as chi2r 8-90
while the residual is only ~1-5 % of ``|X|``, because chi2 weights by
``1/sigma^2`` but line-shape model error scales with amplitude ``A``. Two
escalation targets are on the table for structural work item 2:

* **per-peak tau** -- "the strong line has the wrong decay constant": let the
  dominant line carry its own tau (still inside the calibrated band, still
  penalized toward ``tau_maj``) while weak lines share the window tau.
* **Voigt** -- "the strong line has the wrong functional form": a
  Gaussian-anchored ``tau_G`` core plus a small free Lorentzian width, for the
  Gaussian-core + Lorentzian-wing residual a single pure shape cannot carry.

Per-peak tau plateaus at the same ~5 % wall if the residual is genuinely
Voigt-shaped (it relocates the mismatch rather than removing it), so this probe
*measures* which lever the residual responds to before either refactor is
specced. It is a **throwaway diagnostic**: the per-peak-tau plumbing and the
Voigt line shape are prototyped here, not in ``window_fit.py`` / ``peak_model.py``.

Method
------
For each probed window we take the *shipped* converged free-peak set as the
starting model and run a fixed-K joint refit under three model variants,
holding the frozen-contributor background fixed (evaluated once at the
persisted shared tau and subtracted from the data) so all three variants see
identical data and identical free-peak starts -- only the free-peak model form
differs. This isolates the model-form question from the peak-selection / rescue
machinery (out of scope here).

* ``baseline``    -- shared-tau single ``PeakShape`` (the persisted shape),
  reproducing the shipped fit's floor.
* ``per_peak_tau`` -- the dominant (highest-SNR) free line carries its own tau
  (calibrated band, bidirectional-Gaussian penalty toward ``tau_maj``); the
  rest share the window tau.
* ``voigt``       -- the dominant line is refit with a finite-T Voigt shape
  (shared ``tau_G`` Gaussian core + its own free Lorentzian ``tau_L``); the
  rest stay Gaussian on the shared ``tau_G``.

The shared tau (and per-peak tau) carry the same calibrated bounds + tau penalty
the production fit uses; the phase / amplitude penalties are dropped (the shipped
peaks are already well-separated and sensibly-amplituded, so those penalties are
near-zero and irrelevant to the residual floor under test).

Metric
------
The residual floor each variant reaches, as chi2r *and* as
``max |residual| / |X|`` near the dominant line (a ~1-5 % relative residual is
the shape-mismatch signature, more interpretable than chi2r for strong lines).
Emits ``data/shape_escalation_per_window.csv``.

Run via the conda env::

    conda run -n ftmwpipeline-dev python \
        dev-docs/research/stage5-gaussian-audit/probe_shape_escalation.py
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.special import wofz

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.stage5_impl import _build_active_ft_inputs
from ftmwpipeline.core.data_structures import WindowPlan, SpectrumFit, FitWindow
from ftmwpipeline.core.peak_shape import PeakShape
from ftmwpipeline.fitting.active_ft import compute_active_ft
from ftmwpipeline.fitting.peak_model import (
    ModelPeak,
    h_T_gaussian,
    h_T_shape,
    model_spectrum,
    sideband_sign,
)
from ftmwpipeline.fitting.plan_execution import materialize_window
from ftmwpipeline.fitting.validation import feature_fwhm
from ftmwpipeline.preprocessing.noise_estimation import estimate_noise_scatter

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIXTURE = (
    REPO_ROOT
    / "scratch"
    / "stage5-gaussian-audit"
    / "runs"
    / "rescue_prominence_threshold__1p5__gaussian.ftmw"
)
DATA_DIR = Path(__file__).parent / "data"
OUT_CSV = DATA_DIR / "shape_escalation_per_window.csv"

# Worst strong-line shape_error windows to probe, plus w360 as a "large chi2r
# but excellent fit" control -- the diagnostic should *improve* it, not perturb.
DEFAULT_WINDOWS: Tuple[int, ...] = (236, 308, 318, 368, 360)

# Band half-width for the calibrated tau bounds, in units of sigma_tau. Matches
# window_fit.DEFAULT_TAU_PENALTY_N_SIGMA.
TAU_N_SIGMA = 5.0
TAU_MAX_DECAY_FACTOR = 5.0
# Tau anchoring penalty weight; matches window_fit.DEFAULT_TAU_PENALTY_LAMBDA.
TAU_PENALTY_LAMBDA = 50.0
# Free Lorentzian tau bounds for the Voigt dominant line. Large tau_L => narrow
# Lorentzian wing (near-pure Gaussian); small tau_L => broad wing. Start near
# pure-Gaussian and let the optimizer pull tau_L down only if wings help.
VOIGT_TAU_L_BOUNDS: Tuple[float, float] = (0.5, 500.0)
VOIGT_TAU_L_START = 200.0
# Relative-residual band: bins within this many FWHM of the dominant line.
REL_RESID_BAND_FWHM = 3.0

logger = logging.getLogger("shape-escalation")

FIELDS: List[str] = [
    "window_id",
    "freq_lo_mhz",
    "freq_hi_mhz",
    "variant",
    "chi2r",
    "max_rel_resid_pct",
    "dominant_line_snr",
    "fitted_tau_us",
]


# ---------------------------------------------------------------------------
# Prototype Voigt line shape (throwaway -- not for peak_model.py)
# ---------------------------------------------------------------------------
def h_T_voigt(
    delta_f_mhz: np.ndarray,
    tau_G_us: float,
    tau_L_us: float,
    acquisition_us: float,
) -> np.ndarray:
    """Finite-T FFT of a Gaussian*Lorentzian-windowed cosine over ``[0, T]``.

    The time envelope is ``exp(-(t/tau_G)^2) * exp(-t/tau_L)`` (a Gaussian
    core with a Lorentzian wing). Integrating ``env(t) e^{-i2pi df t}`` over
    ``[0, T]`` via complete-the-square gives the same closed form as
    :func:`h_T_gaussian` but with the purely-imaginary ``beta = i pi df tau_G``
    generalized to the complex ``gamma = tau_G/(2 tau_L) + beta``:

        h_T = (tau_G sqrt(pi)/2) [wofz(i gamma)
              - exp(-(T/tau_G)^2) exp(-T/tau_L) exp(-i 2pi df T) wofz(i(T/tau_G + gamma))]

    The wofz rewrite cancels the overflowing ``exp(gamma^2)`` prefactor exactly,
    mirroring :func:`h_T_gaussian`'s stable form (which is the ``tau_L -> inf``
    limit of this). Reuses the existing scipy ``wofz`` path per the prompt.
    """
    if tau_G_us <= 0.0:
        raise ValueError("tau_G_us must be positive")
    if tau_L_us <= 0.0:
        raise ValueError("tau_L_us must be positive")
    if acquisition_us <= 0.0:
        raise ValueError("acquisition_us must be positive")
    df = np.asarray(delta_f_mhz, dtype=float)
    beta = 1j * np.pi * df * tau_G_us
    gamma = (tau_G_us / (2.0 * tau_L_us)) + beta
    t_over_tau = acquisition_us / tau_G_us
    w_lo = wofz(1j * gamma)
    w_hi = wofz(1j * (t_over_tau + gamma))
    damping_g = np.exp(-(t_over_tau ** 2))
    damping_l = np.exp(-acquisition_us / tau_L_us)
    phase = np.exp(-1j * 2.0 * np.pi * df * acquisition_us)
    response = (tau_G_us * np.sqrt(np.pi) / 2.0) * (
        w_lo - damping_g * damping_l * phase * w_hi
    )
    return response.astype(np.complex128)


def _assert_voigt_reduces_to_gaussian(acquisition_us: float) -> None:
    """Sanity check: Voigt with very large tau_L matches the pure Gaussian."""
    # The Voigt -> Gaussian limit is approached linearly in 1/tau_L (a vanishing
    # Lorentzian wing), so a very large tau_L is needed for tight agreement.
    df = np.linspace(-0.5, 0.5, 257)
    tau_G = 8.0
    voigt = h_T_voigt(df, tau_G, 1.0e8, acquisition_us)
    gauss = h_T_gaussian(df, tau_G, acquisition_us)
    if not np.allclose(voigt, gauss, rtol=1e-6, atol=1e-9):
        raise AssertionError(
            "h_T_voigt(tau_L->inf) does not reduce to h_T_gaussian"
        )


# ---------------------------------------------------------------------------
# Per-window inputs
# ---------------------------------------------------------------------------
@dataclass
class WindowInputs:
    window_id: int
    freq_lo_mhz: float
    freq_hi_mhz: float
    offset_grid: np.ndarray
    z_data: np.ndarray  # data minus frozen background, fit frame
    sig_ri: np.ndarray  # sigma / sqrt(2) per bin (unit-variance weighting)
    sig_c: np.ndarray   # complex per-bin sigma
    free_peaks: List[ModelPeak]
    dominant_index: int
    dominant_snr: float
    tau_shared0: float
    tau_bounds: Tuple[float, float]
    tau_maj: float
    sigma_tau: float
    acquisition_us: float
    shape: PeakShape


@dataclass
class FixtureContext:
    plan: WindowPlan
    fit: SpectrumFit
    active_ft: object
    active_noise_arr: np.ndarray
    sideband: object
    acquisition_us: float
    tau0_us: float
    tau_maj: Optional[float]
    sigma_tau: Optional[float]
    shape: PeakShape


def load_fixture(fixture: Path) -> FixtureContext:
    plan: WindowPlan = ftmw.load_windows(str(fixture))
    fit: SpectrumFit = ftmw.load_fit(str(fixture))
    (
        fid_samples,
        sample_dt_us,
        start_us,
        end_us,
        probe_freq_mhz,
        sideband_enum,
        n_padded,
        acquisition_us,
        _user_ft,
        _trim_range,
    ) = _build_active_ft_inputs(str(fixture))
    active_ft = compute_active_ft(
        fid_samples,
        sample_dt_us,
        start_us=start_us,
        end_us=end_us,
        probe_freq_mhz=probe_freq_mhz,
        sideband=sideband_enum,
        n_padded=n_padded,
    )
    # Active-FT noise, measured the way fit_peaks_impl does: Stage 2 estimator
    # on the sorted active-FT magnitude, re-indexed to native bin order for
    # materialize_window.
    sort_idx = np.argsort(active_ft.freq_mhz)
    unsort_idx = np.argsort(sort_idx)
    freqs_sorted = np.ascontiguousarray(active_ft.freq_mhz[sort_idx])
    spec_sorted = np.ascontiguousarray(active_ft.complex_spectrum[sort_idx])
    active_noise = estimate_noise_scatter(
        freqs_sorted, np.abs(spec_sorted).astype(np.float64)
    )
    active_noise_arr = np.asarray(active_noise.rms_noise, dtype=float)[unsort_idx]

    params = fit.parameters or {}
    tau0_us = float(params.get("tau0_us", acquisition_us / 3.0))
    shape = PeakShape.coerce(str(params.get("shape", "lorentzian")))
    tau_maj = params.get("tau_maj_us")
    sigma_tau = params.get("sigma_tau_us")
    return FixtureContext(
        plan=plan,
        fit=fit,
        active_ft=active_ft,
        active_noise_arr=active_noise_arr,
        sideband=sideband_enum,
        acquisition_us=acquisition_us,
        tau0_us=tau0_us,
        tau_maj=float(tau_maj) if tau_maj is not None else None,
        sigma_tau=float(sigma_tau) if sigma_tau is not None else None,
        shape=shape,
    )


def _tau_band(tau_maj: float, max_decay_factor: float, n_sigma: float,
              sigma_tau: float) -> Tuple[float, float]:
    lo = max(tau_maj - n_sigma * sigma_tau, tau_maj / max_decay_factor)
    hi = min(tau_maj + n_sigma * sigma_tau, tau_maj * max_decay_factor)
    if not lo < hi:
        lo, hi = tau_maj / max_decay_factor, tau_maj * max_decay_factor
    return lo, hi


def build_window_inputs(
    ctx: FixtureContext, window_id: int
) -> Optional[WindowInputs]:
    plan_by_id = {w.window_id: w for w in ctx.plan.windows}
    fit_by_id = {wf.window_id: wf for wf in ctx.fit.window_fits}
    if window_id not in plan_by_id or window_id not in fit_by_id:
        logger.warning("window %d not present in fixture; skipping", window_id)
        return None
    window: FitWindow = plan_by_id[window_id]
    wf = fit_by_id[window_id]

    _freq_slice, offset_grid, z_offset, sig_slice, center_mhz = materialize_window(
        window, ctx.active_ft, ctx.active_noise_arr, sideband=ctx.sideband
    )
    s = sideband_sign(ctx.sideband)
    tau_shared0 = float(wf.shared_parameters.get("tau_us", {}).get("value", 0.0))
    if tau_shared0 <= 0.0:
        logger.warning("window %d has no shared tau; skipping", window_id)
        return None

    free_peaks = [
        ModelPeak(
            amplitude=float(p.amplitude),
            offset_mhz=float(s * (p.frequency_mhz - center_mhz)),
            phase=float(p.phase if p.phase is not None else 0.0),
        )
        for p in wf.fitted_peaks
    ]
    if not free_peaks:
        logger.warning("window %d has no free peaks; skipping", window_id)
        return None

    # Dominant line = highest SNR (fall back to on-line response A*tau_eff).
    snrs = [
        (p.snr if p.snr is not None else float("nan")) for p in wf.fitted_peaks
    ]
    if all(np.isnan(snrs)):
        snrs = [float(p.amplitude) for p in wf.fitted_peaks]
    dominant_index = int(np.nanargmax(snrs))
    dominant_snr = float(snrs[dominant_index])

    # Frozen-contributor background, evaluated once at the persisted shared tau
    # and subtracted so every variant fits identical data.
    frozen_peaks: List[ModelPeak] = []
    for key, fp in wf.fixed_parameters.items():
        if not key.startswith("frozen_peak_"):
            continue
        frozen_peaks.append(
            ModelPeak(
                amplitude=float(fp["amplitude"]),
                offset_mhz=float(s * (float(fp["frequency_mhz"]) - center_mhz)),
                phase=float(fp.get("phase", 0.0) or 0.0),
            )
        )
    if frozen_peaks:
        frozen_bg = model_spectrum(
            offset_grid, frozen_peaks, tau_shared0, ctx.acquisition_us,
            shape=ctx.shape,
        )
    else:
        frozen_bg = np.zeros_like(z_offset)
    z_data = z_offset - frozen_bg

    sig_c = np.asarray(sig_slice, dtype=float)
    sig_ri = sig_c / np.sqrt(2.0)

    # Tau band + anchor. Prefer the calibrated (tau_maj, sigma_tau); else a
    # factor-k band around the persisted shared tau with tau_maj = tau_shared0.
    if ctx.tau_maj is not None and ctx.sigma_tau is not None and ctx.sigma_tau > 0:
        tau_maj = ctx.tau_maj
        sigma_tau = ctx.sigma_tau
    else:
        tau_maj = tau_shared0
        sigma_tau = tau_shared0 / TAU_N_SIGMA  # band spans +/- tau_maj/k
    tau_bounds = _tau_band(tau_maj, TAU_MAX_DECAY_FACTOR, TAU_N_SIGMA, sigma_tau)

    lo, hi = window.freq_range
    return WindowInputs(
        window_id=window_id,
        freq_lo_mhz=float(min(lo, hi)),
        freq_hi_mhz=float(max(lo, hi)),
        offset_grid=np.asarray(offset_grid, dtype=float),
        z_data=np.asarray(z_data, dtype=np.complex128),
        sig_ri=sig_ri,
        sig_c=sig_c,
        free_peaks=free_peaks,
        dominant_index=dominant_index,
        dominant_snr=dominant_snr,
        tau_shared0=float(np.clip(tau_shared0, *tau_bounds)),
        tau_bounds=tau_bounds,
        tau_maj=tau_maj,
        sigma_tau=sigma_tau,
        acquisition_us=ctx.acquisition_us,
        shape=ctx.shape,
    )


# ---------------------------------------------------------------------------
# Variant refits (fixed K, starting from the shipped peaks)
# ---------------------------------------------------------------------------
@dataclass
class VariantResult:
    variant: str
    chi2r: float
    max_rel_resid_pct: float
    fitted_tau_us: float


def _pack_peaks(peaks: Sequence[ModelPeak]) -> List[float]:
    out: List[float] = []
    for pk in peaks:
        out += [pk.amplitude, pk.offset_mhz, pk.phase]
    return out


def _unpack_peaks(p: np.ndarray, k: int) -> List[ModelPeak]:
    return [
        ModelPeak(
            amplitude=float(p[3 * i]),
            offset_mhz=float(p[3 * i + 1]),
            phase=float(p[3 * i + 2]),
        )
        for i in range(k)
    ]


def _peak_bounds(
    k: int, offset_lo: float, offset_hi: float
) -> Tuple[List[float], List[float]]:
    phase_bound = 4.0 * np.pi
    lo: List[float] = []
    hi: List[float] = []
    for _ in range(k):
        lo += [0.0, offset_lo, -phase_bound]
        hi += [np.inf, offset_hi, phase_bound]
    return lo, hi


def _chi2(z_data: np.ndarray, model: np.ndarray, sig_ri: np.ndarray) -> float:
    r = (z_data - model) / sig_ri
    return float(np.sum(r.real ** 2 + r.imag ** 2))


def _max_rel_resid_pct(win: WindowInputs, model: np.ndarray) -> float:
    """max|residual| / max|data+frozen-equivalent signal| near the dominant line.

    Restricted to bins within ``REL_RESID_BAND_FWHM`` FWHM of the dominant
    offset, so the metric reads the shape mismatch where ``|X|`` is large and
    is not diluted by the noise-dominated wings.
    """
    fwhm = feature_fwhm(win.tau_shared0, win.acquisition_us, shape=win.shape)
    dom_off = win.free_peaks[win.dominant_index].offset_mhz
    band = np.abs(win.offset_grid - dom_off) <= REL_RESID_BAND_FWHM * fwhm
    if not np.any(band):
        band = np.ones_like(win.offset_grid, dtype=bool)
    resid = win.z_data - model
    sig_band = np.max(np.abs(win.z_data[band]))
    if sig_band <= 0.0:
        return float("nan")
    return 100.0 * float(np.max(np.abs(resid[band])) / sig_band)


def fit_baseline(win: WindowInputs) -> VariantResult:
    """Shared-tau single-shape joint refit -- reproduces the shipped floor."""
    k = len(win.free_peaks)
    p0 = np.array(_pack_peaks(win.free_peaks) + [win.tau_shared0], dtype=float)
    lo, hi = _peak_bounds(k, float(win.offset_grid.min()), float(win.offset_grid.max()))
    lo.append(win.tau_bounds[0])
    hi.append(win.tau_bounds[1])
    sqrt_lam = np.sqrt(TAU_PENALTY_LAMBDA)

    def residual(p: np.ndarray) -> np.ndarray:
        peaks = _unpack_peaks(p, k)
        tau = float(p[3 * k])
        model = model_spectrum(
            win.offset_grid, peaks, tau, win.acquisition_us, shape=win.shape
        )
        r = (win.z_data - model) / win.sig_ri
        pen = sqrt_lam * (tau - win.tau_maj) / win.sigma_tau
        return np.concatenate([r.real, r.imag, [pen]])

    sol = least_squares(
        residual, np.clip(p0, lo, hi), bounds=(lo, hi),
        method="trf", max_nfev=4000,
    )
    peaks = _unpack_peaks(sol.x, k)
    tau = float(sol.x[3 * k])
    model = model_spectrum(
        win.offset_grid, peaks, tau, win.acquisition_us, shape=win.shape
    )
    chi2 = _chi2(win.z_data, model, win.sig_ri)
    dof = max(2 * win.offset_grid.size - (3 * k + 1), 1)
    return VariantResult(
        "baseline", chi2 / dof, _max_rel_resid_pct(win, model), tau
    )


def fit_per_peak_tau(win: WindowInputs) -> VariantResult:
    """Dominant line carries its own tau; weak lines share the window tau."""
    k = len(win.free_peaks)
    d = win.dominant_index
    p0 = np.array(
        _pack_peaks(win.free_peaks) + [win.tau_shared0, win.tau_shared0],
        dtype=float,
    )
    lo, hi = _peak_bounds(k, float(win.offset_grid.min()), float(win.offset_grid.max()))
    lo += [win.tau_bounds[0], win.tau_bounds[0]]
    hi += [win.tau_bounds[1], win.tau_bounds[1]]
    sqrt_lam = np.sqrt(TAU_PENALTY_LAMBDA)

    def _models(p: np.ndarray) -> np.ndarray:
        peaks = _unpack_peaks(p, k)
        tau_w = float(p[3 * k])
        tau_d = float(p[3 * k + 1])
        weak = [pk for i, pk in enumerate(peaks) if i != d]
        model = model_spectrum(
            win.offset_grid, weak, tau_w, win.acquisition_us, shape=win.shape
        )
        model = model + model_spectrum(
            win.offset_grid, [peaks[d]], tau_d, win.acquisition_us,
            shape=win.shape,
        )
        return model

    def residual(p: np.ndarray) -> np.ndarray:
        model = _models(p)
        r = (win.z_data - model) / win.sig_ri
        tau_w = float(p[3 * k])
        tau_d = float(p[3 * k + 1])
        pen_w = sqrt_lam * (tau_w - win.tau_maj) / win.sigma_tau
        pen_d = sqrt_lam * (tau_d - win.tau_maj) / win.sigma_tau
        return np.concatenate([r.real, r.imag, [pen_w, pen_d]])

    sol = least_squares(
        residual, np.clip(p0, lo, hi), bounds=(lo, hi),
        method="trf", max_nfev=4000,
    )
    model = _models(sol.x)
    chi2 = _chi2(win.z_data, model, win.sig_ri)
    dof = max(2 * win.offset_grid.size - (3 * k + 2), 1)
    tau_w = float(sol.x[3 * k])
    tau_d = float(sol.x[3 * k + 1])
    logger.info(
        "  window %d per_peak_tau: tau_w=%.4g us  tau_d=%.4g us  "
        "(baseline shared tau0=%.4g, tau_maj=%.4g)",
        win.window_id, tau_w, tau_d, win.tau_shared0, win.tau_maj,
    )
    return VariantResult(
        "per_peak_tau", chi2 / dof, _max_rel_resid_pct(win, model), tau_d
    )


def fit_voigt(win: WindowInputs) -> VariantResult:
    """Dominant line refit as Voigt (shared tau_G core + free Lorentzian tau_L)."""
    k = len(win.free_peaks)
    d = win.dominant_index
    p0 = np.array(
        _pack_peaks(win.free_peaks) + [win.tau_shared0, VOIGT_TAU_L_START],
        dtype=float,
    )
    lo, hi = _peak_bounds(k, float(win.offset_grid.min()), float(win.offset_grid.max()))
    lo += [win.tau_bounds[0], VOIGT_TAU_L_BOUNDS[0]]
    hi += [win.tau_bounds[1], VOIGT_TAU_L_BOUNDS[1]]
    sqrt_lam = np.sqrt(TAU_PENALTY_LAMBDA)

    def _models(p: np.ndarray) -> np.ndarray:
        peaks = _unpack_peaks(p, k)
        tau_g = float(p[3 * k])
        tau_l = float(p[3 * k + 1])
        weak = [pk for i, pk in enumerate(peaks) if i != d]
        # Weak lines: shared Gaussian (the persisted shape is gaussian on the
        # probed fixture; h_T_shape routes correctly either way).
        model = model_spectrum(
            win.offset_grid, weak, tau_g, win.acquisition_us, shape=win.shape
        )
        pk = peaks[d]
        phasor = 0.5 * pk.amplitude * np.exp(1j * pk.phase)
        model = model + phasor * h_T_voigt(
            win.offset_grid - pk.offset_mhz, tau_g, tau_l, win.acquisition_us
        )
        return model

    def residual(p: np.ndarray) -> np.ndarray:
        model = _models(p)
        r = (win.z_data - model) / win.sig_ri
        tau_g = float(p[3 * k])
        pen_g = sqrt_lam * (tau_g - win.tau_maj) / win.sigma_tau
        return np.concatenate([r.real, r.imag, [pen_g]])

    sol = least_squares(
        residual, np.clip(p0, lo, hi), bounds=(lo, hi),
        method="trf", max_nfev=4000,
    )
    model = _models(sol.x)
    chi2 = _chi2(win.z_data, model, win.sig_ri)
    # tau_G shared + tau_L free => 2 extra params (matches per_peak_tau count).
    dof = max(2 * win.offset_grid.size - (3 * k + 2), 1)
    tau_g = float(sol.x[3 * k])
    tau_l = float(sol.x[3 * k + 1])
    logger.info(
        "  window %d voigt: tau_G=%.4g us  tau_L=%.4g us", win.window_id,
        tau_g, tau_l,
    )
    return VariantResult(
        "voigt", chi2 / dof, _max_rel_resid_pct(win, model), tau_g
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(fixture: Path, window_ids: Sequence[int]) -> List[Tuple]:
    ctx = load_fixture(fixture)
    _assert_voigt_reduces_to_gaussian(ctx.acquisition_us)
    rows: List[Tuple] = []
    for wid in window_ids:
        win = build_window_inputs(ctx, wid)
        if win is None:
            continue
        logger.info(
            "window %d  [%.2f, %.2f] MHz  K=%d  dominant#%d snr=%.1f  tau0=%.4g",
            wid, win.freq_lo_mhz, win.freq_hi_mhz, len(win.free_peaks),
            win.dominant_index, win.dominant_snr, win.tau_shared0,
        )
        for fn in (fit_baseline, fit_per_peak_tau, fit_voigt):
            vr = fn(win)
            logger.info(
                "    %-13s chi2r=%.3f  max_rel_resid=%.2f%%  tau=%.4g",
                vr.variant, vr.chi2r, vr.max_rel_resid_pct, vr.fitted_tau_us,
            )
            rows.append(
                (
                    wid,
                    f"{win.freq_lo_mhz:.4f}",
                    f"{win.freq_hi_mhz:.4f}",
                    vr.variant,
                    f"{vr.chi2r:.6f}",
                    f"{vr.max_rel_resid_pct:.4f}",
                    f"{win.dominant_snr:.4f}",
                    f"{vr.fitted_tau_us:.6f}",
                )
            )
    return rows


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    ap.add_argument(
        "--windows", type=int, nargs="+", default=list(DEFAULT_WINDOWS)
    )
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    args = ap.parse_args()

    if not args.fixture.exists():
        raise SystemExit(f"missing fixture {args.fixture}")
    rows = run(args.fixture, args.windows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS)
        w.writerows(rows)
    logger.info("wrote %d rows -> %s", len(rows), args.out)


if __name__ == "__main__":
    main()
