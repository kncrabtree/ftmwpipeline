"""
Per-window least-squares core for Stage 5 fitting.

This module recreates the contract of the lost ``fit_time_domain_peaks``
engine, inferred from its surviving call sites
(``dev-docs/planning/stage5-fitting.md``, "Reuse map"):

    (complex window, initial peaks, shared decay, bounds)
        -> success, fitted complex spectrum, per-peak {frequency, amplitude,
           decay}, cost, AIC, reduced chi-squared, parameter covariance.

:func:`fit_window` is the *fixed-K* core: given a window of complex FT data and
a starting set of ``K`` lines it refines all line parameters at once by
complex-domain least squares. The conservative add-one-peak loop that *decides*
K -- and the F-test / AIC / blend-aware seeding around it -- is Stage 5 task 4
and builds on this core; it is not here.

The fit frame
-------------
The core works entirely in the demodulated *fit frame*: the window grid is
the signed baseband offset ``u`` and each line is parameterised by
``(amplitude, offset_mhz, phase)`` (:class:`~ftmwpipeline.fitting.peak_model.ModelPeak`)
plus a window-shared decay ``tau``. The data passed in is a slice of the
active-portion FT (:mod:`ftmwpipeline.fitting.active_ft`), already in the
``[0, T]`` reference frame ``h_T`` models; the molecular<->offset grid
relabel is :func:`~ftmwpipeline.fitting.peak_model.to_baseband_offset`,
applied by the orchestration *before* calling :func:`fit_window`.

The residual and its weighting
------------------------------
The residual is in the complex-FT domain: model versus active-FT window data,
identical point counts, real and imaginary parts stacked into one real vector.
The canonical Stage 2 ``rms_noise`` is a per-bin *complex* RMS ``sigma``; the
real and imaginary parts each carry variance ``sigma**2 / 2``, so every stacked
element is weighted by ``sigma / sqrt(2)`` to be unit-variance. Then the cost
is a proper chi-squared (reduced chi-squared ~ 1) and -- once task 4 adds it --
the F-test is calibrated (D-8; the prototype confirmed weighting by ``sigma``
itself leaves reduced chi-squared ~ 0.5).

tau handling
------------
``tau`` is shared by all lines in the window. ``fit_tau=True`` makes it a free
shared parameter; ``fit_tau=False`` holds it at the supplied default (the
weak-only-window path, O5-4). The free-vs-fixed *decision* is the caller's.

Jacobian and covariance
-----------------------
The solver uses the analytic Jacobian assembled by :func:`model_jacobian` from
``h_T`` and its derivatives -- no finite-difference phase. ``inv(J^T J)`` at the
solution is the parameter covariance directly (the residual is unit-variance by
construction), giving real frequency / amplitude / phase uncertainties.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Optional, Union, cast

import numpy as np
from scipy.optimize import least_squares

from .peak_model import ModelPeak, effective_tau, h_T, h_T_jacobian, model_spectrum
from .validation import (
    DEFAULT_CONSERVATIVE_N_EFF_KIND,
    DEFAULT_N_EFF_KIND,
    calculate_aic,
    calculate_aicc,
    calculate_chi_squared_improvement,
    calculate_noise_weighted_chi2,
    effective_sample_size,
    feature_fwhm,
    validate_peak_separation,
)

__all__ = [
    "ParameterErrors",
    "WindowFitResult",
    "AddStep",
    "KnockoutResult",
    "ConservativeFitResult",
    "WindowFitConstraints",
    "derive_window_fit_constraints",
    "model_jacobian",
    "fit_window",
    "knockout_test",
    "conservative_fit",
]

NoiseLike = Union[float, np.ndarray]

# Default solver evaluation cap. Originally 400 (ported from the bcfitting
# reference shell), bumped to 2000 after the residual-rescue work surfaced
# partial-capture cases where LSQ converges (chi-squared stops moving) but
# scipy still flags ``success=False`` because it hit the iteration cap --
# which then trips the early-exit guards in
# :func:`_blend_aware_seed` / :func:`conservative_fit` (both bail on
# ``not fit.success``). 2000 covers every case seen in the 2638 fixture;
# the cap exists only as a runaway-safety, not a quality criterion.
DEFAULT_MAX_NFEV = 2000
# Default tau bound factor k: tau in [tau_default / k, tau_default * k] (O5-4).
DEFAULT_MAX_DECAY_FACTOR = 5.0
# Phase bound: wide enough that wrapping never clips a free phase.
_PHASE_BOUND = 4.0 * np.pi

# Conservative add-one-peak loop defaults.
DEFAULT_SIGNIFICANCE = 0.05
DEFAULT_MAX_PEAKS = 8
DEFAULT_PATIENCE = 1
DEFAULT_MIN_SEPARATION_FACTOR = 1.0
# Blend-aware seeder: a single-cosine seed fit whose reduced chi-squared
# exceeds this is treated as an unresolved blend and re-seeded at K=2/K=3.
DEFAULT_SEEDER_RCHI2 = 1.5
# Straddle of the re-seeded inits, in units of the feature FWHM.
DEFAULT_SEEDER_STRADDLE_FACTOR = 1.0
DEFAULT_SEEDER_MAX_K = 3
# Soft phase-difference penalty: weak at sep = phase_penalty_cutoff_fwhm * fwhm,
# growing linearly to (lambda * |sin(d_phase/2)|^2) at sep = 0. Catches the
# degenerate "two peaks collapsed to the same offset with cancelling phases"
# blend-aware re-seed pathology.
DEFAULT_PHASE_PENALTY_LAMBDA = 100.0
DEFAULT_PHASE_PENALTY_CUTOFF_FWHM = 2.0
# Soft amplitude-floor penalty: linear hinge that adds sqrt(lambda) * max(0,
# 1 - A/A_floor) for each peak, with A_floor scaled to the noise level so
# noise-amplitude peaks are pushed toward 0.
DEFAULT_AMP_PENALTY_LAMBDA = 10.0
# Hard upper bound on a peak amplitude as a multiple of 2 * max(|data|) /
# tau_eff_min (the strongest physically-plausible amplitude). 3x leaves
# headroom for blended-peak superposition while still rejecting the
# degenerate ~1000x-inflated amplitudes from the cancelling-pair pathology.
DEFAULT_AMP_MAX_HEADROOM = 3.0
# Post-fit sanity check in _blend_aware_seed: reject a K>=2 escalation if any
# two of its fitted peaks collapsed to within this fraction of a FWHM.
DEFAULT_MIN_PAIR_SEPARATION_FACTOR = 0.5
# Tau policy: the canonical apodization (``expf_us``) sets a hard upper bound
# on ``tau`` -- the data cannot decay slower than the apodization itself.
# Decreasing ``tau`` below the apodization broadens the line, so the LSQ
# can buy chi^2 by under-fitting amplitude and over-broadening to absorb
# unmodeled-peak residual; the penalty discourages this with a stiff
# quadratic hinge. Weak-only windows (no candidate clears
# ``DEFAULT_WEAK_WINDOW_SNR_THRESHOLD``) hold ``tau`` fixed entirely.
DEFAULT_TAU_PENALTY_LAMBDA = 500.0
DEFAULT_WEAK_WINDOW_SNR_THRESHOLD = 10.0


# ---------------------------------------------------------------------------
# Derived per-window fit constraints
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WindowFitConstraints:
    """Per-window state derived from a window's data + caller-level knobs.

    Returned by :func:`derive_window_fit_constraints` and consumed by
    :func:`conservative_fit` and the residual-rescue orchestrator so both
    enforce the same tau bounds / amplitude bounds / penalty weights.
    """

    tau_bounds: tuple[float, float]
    fwhm: float
    min_separation: float
    amp_max: Optional[float]
    amp_floor: Optional[float]
    fit_tau_eff: bool
    tau_penalty_reference: Optional[float]
    effective_tau_penalty_lambda: float
    fit_kwargs_inner: dict


def derive_window_fit_constraints(
    complex_spectrum: np.ndarray,
    rms_noise: np.ndarray,
    tau0_us: float,
    acquisition_us: float,
    *,
    fit_tau: bool = True,
    min_separation_factor: float = DEFAULT_MIN_SEPARATION_FACTOR,
    max_decay_factor: float = DEFAULT_MAX_DECAY_FACTOR,
    amp_max_headroom: float = DEFAULT_AMP_MAX_HEADROOM,
    phase_penalty_lambda: float = DEFAULT_PHASE_PENALTY_LAMBDA,
    amp_penalty_lambda: float = DEFAULT_AMP_PENALTY_LAMBDA,
    phase_penalty_cutoff_fwhm: float = DEFAULT_PHASE_PENALTY_CUTOFF_FWHM,
    tau_penalty_lambda: float = DEFAULT_TAU_PENALTY_LAMBDA,
    weak_window_snr_threshold: float = DEFAULT_WEAK_WINDOW_SNR_THRESHOLD,
    tau_apodization_us: Optional[float] = None,
) -> WindowFitConstraints:
    """Derive a window's tau / amplitude / penalty constraints from its data.

    The tau policy (apodization-bounded above, stiff-penalty-toward-it below,
    forced-fixed for weak-only windows) and the amplitude bounds (max from
    the strongest in-window data and the tightest tau bound, floor from the
    noise level) are derived purely from the window's complex spectrum and
    the noise estimate -- no peak fitting involved. Returned bounds match the
    legacy inline derivation in :func:`conservative_fit` exactly.
    """
    z = np.asarray(complex_spectrum, dtype=np.complex128)
    sigma = np.asarray(rms_noise, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(z.size, float(sigma))

    tau_upper = tau0_us * max_decay_factor
    if tau_apodization_us is not None and tau_apodization_us > 0.0:
        tau_upper = min(tau_upper, float(tau_apodization_us))
    tau_bounds = (tau0_us / max_decay_factor, tau_upper)
    fwhm = feature_fwhm(tau0_us, acquisition_us)
    min_separation = min_separation_factor * fwhm

    tau_eff_min = effective_tau(tau_bounds[0], acquisition_us)
    tau_eff_nom = effective_tau(tau0_us, acquisition_us)
    max_abs = float(np.max(np.abs(z))) if z.size else 0.0
    if max_abs > 0.0 and tau_eff_min > 0.0:
        amp_max: Optional[float] = float(
            amp_max_headroom * 2.0 * max_abs / tau_eff_min
        )
    else:
        amp_max = None
    sig_median = float(np.median(sigma)) if sigma.size else 0.0
    if sig_median > 0.0 and tau_eff_nom > 0.0:
        amp_floor: Optional[float] = float(2.0 * sig_median / tau_eff_nom)
    else:
        amp_floor = None

    fit_tau_eff = fit_tau
    snr_proxy = (max_abs / sig_median) if sig_median > 0.0 else 0.0
    if fit_tau_eff and snr_proxy < weak_window_snr_threshold:
        fit_tau_eff = False

    tau_penalty_ref = (
        float(tau_apodization_us)
        if tau_apodization_us is not None and tau_apodization_us > 0.0
        else None
    )
    effective_tau_penalty_lambda = (
        tau_penalty_lambda if tau_penalty_ref is not None else 0.0
    )

    fit_kwargs_inner = dict(
        fit_tau=fit_tau_eff,
        tau_bounds=tau_bounds,
        amp_max=amp_max,
        amp_floor=amp_floor,
        fwhm_mhz=fwhm,
        phase_penalty_lambda=phase_penalty_lambda,
        amp_penalty_lambda=amp_penalty_lambda,
        phase_penalty_cutoff_fwhm=phase_penalty_cutoff_fwhm,
        tau_penalty_lambda=effective_tau_penalty_lambda,
        tau_penalty_reference=tau_penalty_ref,
    )

    return WindowFitConstraints(
        tau_bounds=tau_bounds,
        fwhm=fwhm,
        min_separation=min_separation,
        amp_max=amp_max,
        amp_floor=amp_floor,
        fit_tau_eff=fit_tau_eff,
        tau_penalty_reference=tau_penalty_ref,
        effective_tau_penalty_lambda=effective_tau_penalty_lambda,
        fit_kwargs_inner=fit_kwargs_inner,
    )


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class ParameterErrors:
    """1-sigma uncertainties of one line's fitted parameters.

    Parallels :class:`~ftmwpipeline.fitting.peak_model.ModelPeak`. Entries are
    ``nan`` when the covariance matrix was unavailable (a singular ``J^T J``).

    Attributes
    ----------
    amplitude : float
        Uncertainty of the line amplitude ``A``.
    offset_mhz : float
        Uncertainty of the signed baseband offset ``delta`` (MHz). Because the
        molecular frequency is ``f = f_c + s*delta`` with ``|s| = 1``, this is
        also the line's frequency uncertainty.
    phase : float
        Uncertainty of the line phase (radians).
    """

    amplitude: float
    offset_mhz: float
    phase: float


@dataclass
class WindowFitResult:
    """Outcome of a fixed-K :func:`fit_window` call.

    Attributes
    ----------
    success : bool
        Whether the solver reported convergence.
    peaks : list of ModelPeak
        The fitted lines, in fit-frame (baseband-offset) coordinates, ordered
        as the initial peaks were. Phases are wrapped to ``(-pi, pi]``.
    peak_errors : list of ParameterErrors
        Per-line 1-sigma uncertainties, parallel to ``peaks``.
    tau_us : float
        The shared decay constant -- fitted if ``fit_tau`` else the input.
    tau_error : float or None
        Uncertainty of ``tau_us``; ``None`` when ``tau`` was held fixed.
    fit_tau : bool
        Whether ``tau`` was a free parameter.
    cost : float
        ``scipy`` least-squares cost ``0.5 * sum(r**2)`` of the weighted
        residual.
    chi_squared : float
        Noise-weighted ``chi-squared = sum(r**2)`` over the unit-variance
        stacked residual (``= 2 * cost``).
    n_data : int
        Number of real residual elements (``2 * M`` for an ``M``-bin window).
    n_params : int
        Number of free parameters (``3 * K`` plus one if ``fit_tau``).
    n_function_evals : int
        Solver residual-function evaluations.
    fitted_spectrum : np.ndarray
        The fitted complex model on the window grid.
    residual : np.ndarray
        ``data - fitted_spectrum`` (complex, *unweighted*).
    covariance : np.ndarray or None
        Parameter covariance matrix (order: ``A, delta, phase`` per peak, then
        ``tau`` if fitted), or ``None`` if ``J^T J`` was singular.
    """

    success: bool
    peaks: list[ModelPeak]
    peak_errors: list[ParameterErrors]
    tau_us: float
    tau_error: Optional[float]
    fit_tau: bool
    cost: float
    chi_squared: float
    n_data: int
    n_params: int
    n_function_evals: int
    fitted_spectrum: np.ndarray
    residual: np.ndarray
    covariance: Optional[np.ndarray] = field(default=None)

    @property
    def n_peaks(self) -> int:
        """Number of fitted lines ``K``."""
        return len(self.peaks)

    @property
    def reduced_chi2(self) -> float:
        """Reduced chi-squared ``chi-squared / (n_data - n_params)``.

        Close to 1 for a good fit with the canonical Stage 2 noise and the
        ``sigma / sqrt(2)`` weighting. ``inf`` when there are no degrees of
        freedom.
        """
        dof = self.n_data - self.n_params
        if dof <= 0:
            return float("inf")
        return self.chi_squared / dof

    @property
    def aic(self) -> float:
        """Akaike information criterion ``2k + n * ln(chi-squared / n)``.

        The form ported from the bcfitting reference shell; used by the
        conservative add-one-peak loop (task 4) to compare nested models.
        ``inf`` for a degenerate (non-positive ``chi-squared``) fit.
        """
        return calculate_aic(self.chi_squared, self.n_params, self.n_data)


# ---------------------------------------------------------------------------
# Analytic model Jacobian
# ---------------------------------------------------------------------------
def model_jacobian(
    offset_grid_mhz: np.ndarray,
    peaks: Sequence[ModelPeak],
    tau_us: float,
    acquisition_us: float,
    *,
    include_tau: bool = False,
) -> np.ndarray:
    """Analytic Jacobian of :func:`model_spectrum` w.r.t. the fit parameters.

    Returns ``d(model)/d(p)`` -- one complex column per parameter, evaluated on
    the window grid. With ``model = sum_j 0.5 A_j e^{i phi_j} h_T(u - delta_j)``
    the per-line columns are

        d/dA      = 0.5 e^{i phi} h_T(u - delta)
        d/ddelta  = -0.5 A e^{i phi} (dh_T/d(Delta f))(u - delta)
        d/dphi    = 0.5j A e^{i phi} h_T(u - delta)

    and, when ``include_tau`` is set, a final shared column
    ``d/dtau = sum_j 0.5 A_j e^{i phi_j} (dh_T/dtau)(u - delta_j)``.

    Parameters
    ----------
    offset_grid_mhz : np.ndarray
        Baseband-offset grid ``u`` for the window (MHz), 1-D.
    peaks : sequence of ModelPeak
        The ``K`` lines.
    tau_us : float
        Shared decay constant ``tau`` (microseconds, ``> 0``).
    acquisition_us : float
        Active acquisition length ``T`` (microseconds, ``> 0``).
    include_tau : bool, default False
        Append the shared ``d/dtau`` column.

    Returns
    -------
    np.ndarray
        Complex array of shape ``(M, 3K)`` -- or ``(M, 3K + 1)`` with
        ``include_tau`` -- where ``M = len(offset_grid_mhz)``.
    """
    u = np.asarray(offset_grid_mhz, dtype=float)
    k = len(peaks)
    n_params = 3 * k + (1 if include_tau else 0)
    jac = np.zeros((u.size, n_params), dtype=np.complex128)
    dmodel_dtau = np.zeros(u.size, dtype=np.complex128)

    for i, pk in enumerate(peaks):
        du = u - pk.offset_mhz
        shape = h_T(du, tau_us, acquisition_us)
        d_shape_df, d_shape_dtau = h_T_jacobian(du, tau_us, acquisition_us)
        phasor = np.exp(1j * pk.phase)
        jac[:, 3 * i] = 0.5 * phasor * shape
        jac[:, 3 * i + 1] = -0.5 * pk.amplitude * phasor * d_shape_df
        jac[:, 3 * i + 2] = 0.5j * pk.amplitude * phasor * shape
        dmodel_dtau += 0.5 * pk.amplitude * phasor * d_shape_dtau

    if include_tau:
        jac[:, 3 * k] = dmodel_dtau
    return cast(np.ndarray, jac)


# ---------------------------------------------------------------------------
# Parameter packing
# ---------------------------------------------------------------------------
def _pack(peaks: Sequence[ModelPeak], tau_us: float, fit_tau: bool) -> np.ndarray:
    """Flatten ``(peaks, tau)`` into the solver parameter vector."""
    values: list[float] = []
    for pk in peaks:
        values += [pk.amplitude, pk.offset_mhz, pk.phase]
    if fit_tau:
        values.append(tau_us)
    return cast(np.ndarray, np.asarray(values, dtype=float))


def _unpack(
    params: np.ndarray, k: int, tau_fixed: float, fit_tau: bool
) -> tuple[list[ModelPeak], float]:
    """Inverse of :func:`_pack`."""
    peaks = [
        ModelPeak(
            amplitude=float(params[3 * i]),
            offset_mhz=float(params[3 * i + 1]),
            phase=float(params[3 * i + 2]),
        )
        for i in range(k)
    ]
    tau = float(params[3 * k]) if fit_tau else tau_fixed
    return peaks, tau


def _wrap_phase(phase: float) -> float:
    """Wrap a phase into ``(-pi, pi]``."""
    return float((phase + np.pi) % (2.0 * np.pi) - np.pi)


def _parameter_errors(
    covariance: Optional[np.ndarray], k: int, fit_tau: bool
) -> tuple[list[ParameterErrors], Optional[float]]:
    """Per-line 1-sigma errors and the tau error from the covariance diagonal."""
    if covariance is None:
        peak_errors = [
            ParameterErrors(float("nan"), float("nan"), float("nan")) for _ in range(k)
        ]
        return peak_errors, (float("nan") if fit_tau else None)

    diag = np.diag(covariance)
    # A tiny negative diagonal entry is numerical noise -> report it as nan.
    sigma = np.sqrt(np.where(diag > 0.0, diag, np.nan))
    peak_errors = [
        ParameterErrors(
            amplitude=float(sigma[3 * i]),
            offset_mhz=float(sigma[3 * i + 1]),
            phase=float(sigma[3 * i + 2]),
        )
        for i in range(k)
    ]
    tau_error = float(sigma[3 * k]) if fit_tau else None
    return peak_errors, tau_error


# ---------------------------------------------------------------------------
# The fixed-K least-squares core
# ---------------------------------------------------------------------------
def _penalty_count(
    k: int,
    phase_penalty_lambda: float,
    amp_penalty_lambda: float,
    tau_penalty_lambda: float = 0.0,
    fit_tau: bool = False,
) -> int:
    """Number of penalty residual elements for a window with ``k`` peaks.

    Always-emitted shape (zero when inactive) so scipy's least-squares sees a
    constant residual length across iterations; otherwise the optimiser
    silently breaks when peaks cross the cutoff or amplitude floor mid-fit.
    """
    n = 0
    if phase_penalty_lambda > 0.0 and k > 1:
        n += k * (k - 1) // 2
    if amp_penalty_lambda > 0.0:
        n += k
    if tau_penalty_lambda > 0.0 and fit_tau:
        n += 1
    return n


def _penalty_residuals_and_jacobian(
    params: np.ndarray,
    k: int,
    tau0_us: float,
    fit_tau: bool,
    *,
    phase_penalty_lambda: float,
    amp_penalty_lambda: float,
    amp_floor: Optional[float],
    fwhm_mhz: Optional[float],
    phase_penalty_cutoff_fwhm: float,
    tau_penalty_lambda: float = 0.0,
    tau_penalty_reference: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Penalty residuals + analytic Jacobian rows for the augmented LSQ.

    Three soft penalties, all built so ``sum(r**2)`` matches the cost the
    scipy solver minimises in addition to the data residual:

    * **Pair phase penalty** -- one residual element per unordered pair
      ``(i, j)``. The penalty is
      ``sqrt(lambda) * weight * sin((phi_i - phi_j) / 2)`` with
      ``weight = max(0, 1 - sep / cutoff)``: zero for in-phase peaks and at
      or beyond the cutoff separation, maximal for anti-phase peaks at zero
      separation. The slot is *always emitted* (with value zero when
      ``weight = 0``) so the residual vector has constant length across
      solver iterations.
    * **Amplitude floor penalty** -- one residual element per peak,
      ``sqrt(lambda) * max(0, 1 - A_i / amp_floor)``. A linear hinge that
      pushes spurious noise-amplitude peaks toward zero. Always emitted
      (zero above the floor).
    * **Tau lower-side penalty** -- one residual element when ``fit_tau``
      and ``tau_penalty_lambda > 0``:
      ``sqrt(lambda) * max(0, (tau_ref - tau) / tau_ref)``. The applied
      apodization (``expf_us``) sets a hard upper bound on tau (data can't
      decay slower than the apodization); the penalty discourages tau from
      drifting *below* it so the LSQ can't broaden the line to absorb
      unmodeled-peak residual.
    """
    n_params = 3 * k + (1 if fit_tau else 0)
    n_pen = _penalty_count(
        k,
        phase_penalty_lambda,
        amp_penalty_lambda,
        tau_penalty_lambda,
        fit_tau,
    )
    if n_pen == 0:
        return np.zeros(0, dtype=float), np.zeros((0, n_params), dtype=float)
    res = np.zeros(n_pen, dtype=float)
    jac = np.zeros((n_pen, n_params), dtype=float)
    peaks, _tau = _unpack(params, k, tau0_us, fit_tau)

    row = 0
    if phase_penalty_lambda > 0.0 and k > 1:
        cutoff = (
            phase_penalty_cutoff_fwhm * fwhm_mhz
            if fwhm_mhz is not None
            else 0.0
        )
        sqrt_lambda = float(np.sqrt(phase_penalty_lambda))
        for i in range(k):
            for j in range(i + 1, k):
                if cutoff > 0.0:
                    off_i = peaks[i].offset_mhz
                    off_j = peaks[j].offset_mhz
                    sep = off_i - off_j
                    abs_sep = abs(sep)
                    if abs_sep < cutoff:
                        weight = 1.0 - abs_sep / cutoff
                        half_diff = 0.5 * (peaks[i].phase - peaks[j].phase)
                        sin_half = float(np.sin(half_diff))
                        cos_half = float(np.cos(half_diff))
                        res[row] = sqrt_lambda * weight * sin_half
                        sgn = (sep / abs_sep) if abs_sep > 0.0 else 0.0
                        dweight_doffi = -sgn / cutoff
                        dweight_doffj = sgn / cutoff
                        jac[row, 3 * i + 1] = (
                            sqrt_lambda * dweight_doffi * sin_half
                        )
                        jac[row, 3 * j + 1] = (
                            sqrt_lambda * dweight_doffj * sin_half
                        )
                        jac[row, 3 * i + 2] = (
                            sqrt_lambda * weight * 0.5 * cos_half
                        )
                        jac[row, 3 * j + 2] = (
                            -sqrt_lambda * weight * 0.5 * cos_half
                        )
                row += 1

    if amp_penalty_lambda > 0.0 and amp_floor is not None and amp_floor > 0.0:
        sqrt_lambda = float(np.sqrt(amp_penalty_lambda))
        for i, pk in enumerate(peaks):
            ratio = pk.amplitude / amp_floor
            if ratio < 1.0:
                res[row] = sqrt_lambda * (1.0 - ratio)
                jac[row, 3 * i] = -sqrt_lambda / amp_floor
            row += 1

    if (
        tau_penalty_lambda > 0.0
        and fit_tau
        and tau_penalty_reference is not None
        and tau_penalty_reference > 0.0
    ):
        # Tau is the last packed parameter when fit_tau is True.
        tau_value = float(params[3 * k])
        ratio = (tau_penalty_reference - tau_value) / tau_penalty_reference
        if ratio > 0.0:
            sqrt_lambda = float(np.sqrt(tau_penalty_lambda))
            res[row] = sqrt_lambda * ratio
            # d/d(tau) of (ref - tau)/ref = -1/ref
            jac[row, 3 * k] = -sqrt_lambda / tau_penalty_reference
        row += 1

    return res, jac


def fit_window(
    offset_grid_mhz: np.ndarray,
    complex_spectrum: np.ndarray,
    rms_noise: NoiseLike,
    initial_peaks: Sequence[ModelPeak],
    tau0_us: float,
    acquisition_us: float,
    *,
    fit_tau: bool = True,
    max_decay_factor: float = DEFAULT_MAX_DECAY_FACTOR,
    tau_bounds: Optional[tuple[float, float]] = None,
    offset_bounds: Optional[tuple[float, float]] = None,
    max_nfev: int = DEFAULT_MAX_NFEV,
    amp_max: Optional[float] = None,
    phase_penalty_lambda: float = 0.0,
    amp_penalty_lambda: float = 0.0,
    amp_floor: Optional[float] = None,
    fwhm_mhz: Optional[float] = None,
    phase_penalty_cutoff_fwhm: float = DEFAULT_PHASE_PENALTY_CUTOFF_FWHM,
    tau_penalty_lambda: float = 0.0,
    tau_penalty_reference: Optional[float] = None,
) -> WindowFitResult:
    """Fit a fixed number of lines to one window by complex least squares.

    Refines every line's ``(amplitude, offset_mhz, phase)`` -- and optionally
    the shared ``tau`` -- against the active-FT complex window data, using the
    analytic Jacobian. This is the fixed-K core; choosing K is task 4.

    Parameters
    ----------
    offset_grid_mhz : np.ndarray
        Baseband-offset grid ``u`` for the window (MHz), 1-D. The window data
        must already be in the active-FT fit frame
        (:func:`~ftmwpipeline.fitting.peak_model.to_baseband_offset`).
    complex_spectrum : np.ndarray
        Complex active-FT window data on ``offset_grid_mhz``, same shape.
    rms_noise : float or np.ndarray
        Per-bin *complex* noise RMS ``sigma`` (canonical Stage 2 noise). A
        scalar is broadcast across the window; an array must match the grid.
    initial_peaks : sequence of ModelPeak
        Starting lines, in fit-frame coordinates. An empty sequence yields a
        null-model result (``success=False``) carrying the data's chi-squared.
    tau0_us : float
        Starting / default shared decay constant ``tau`` (microseconds).
    acquisition_us : float
        Active acquisition length ``T`` (microseconds).
    fit_tau : bool, default True
        Fit ``tau`` as a free shared parameter; if ``False`` hold it at
        ``tau0_us`` (the weak-only-window path, O5-4).
    max_decay_factor : float, default 5.0
        Bound factor ``k``: ``tau`` is bounded to
        ``[tau0_us / k, tau0_us * k]`` when ``tau_bounds`` is not given.
    tau_bounds : tuple of float, optional
        Explicit ``(lo, hi)`` bounds for ``tau``; overrides ``max_decay_factor``.
    offset_bounds : tuple of float, optional
        ``(lo, hi)`` bounds for every line offset; defaults to the grid span.
    max_nfev : int, default 400
        Solver residual-evaluation cap.
    amp_max : float, optional
        Hard upper bound on every peak amplitude (replaces the default
        ``+inf``). Typically set by the caller from the in-window data
        magnitude with a small headroom factor.
    phase_penalty_lambda : float, default 0.0
        Weight of the pair-phase penalty (see
        :func:`_penalty_residuals_and_jacobian`). ``0`` disables it.
    amp_penalty_lambda : float, default 0.0
        Weight of the amplitude-floor penalty. ``0`` disables it.
    amp_floor : float, optional
        Amplitude scale for the floor penalty (required when
        ``amp_penalty_lambda > 0``).
    fwhm_mhz : float, optional
        Line-shape FWHM (MHz) used by the phase penalty (required when
        ``phase_penalty_lambda > 0``).
    phase_penalty_cutoff_fwhm : float, default 2.0
        Pair-separation cutoff for the phase penalty, in FWHM units. The
        penalty linearly ramps from 0 at this separation to its full
        ``sqrt(lambda) * |sin(d_phase/2)|`` at zero separation.
    tau_penalty_lambda : float, default 0.0
        Weight of the lower-side tau penalty. ``0`` disables it.
    tau_penalty_reference : float, optional
        Reference tau (typically the apodization ``expf_us``) that the
        lower-side penalty pulls tau toward. Required when
        ``tau_penalty_lambda > 0``.

    Returns
    -------
    WindowFitResult
        Fitted lines, uncertainties, the shared ``tau``, fit statistics, the
        fitted complex spectrum, and the parameter covariance.

    Raises
    ------
    ValueError
        If the arrays are not 1-D or differ in length, ``rms_noise`` is not
        positive or mis-shaped, ``tau0_us`` / ``acquisition_us`` is not
        positive, or the ``tau`` bounds are degenerate.
    """
    u = np.asarray(offset_grid_mhz, dtype=float)
    z = np.asarray(complex_spectrum, dtype=np.complex128)
    if u.ndim != 1:
        raise ValueError("offset_grid_mhz must be 1-dimensional")
    if u.shape != z.shape:
        raise ValueError("offset_grid_mhz and complex_spectrum must match shape")
    if tau0_us <= 0.0:
        raise ValueError("tau0_us must be positive")
    if acquisition_us <= 0.0:
        raise ValueError("acquisition_us must be positive")

    m = u.size
    sigma = np.asarray(rms_noise, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(m, float(sigma))
    elif sigma.shape != u.shape:
        raise ValueError("rms_noise must be a scalar or match the window grid")
    if np.any(sigma <= 0.0):
        raise ValueError("rms_noise must be strictly positive")
    # The complex noise has per-bin RMS sigma, so the stacked Re/Im residual is
    # weighted by sigma / sqrt(2) for every element to be unit-variance (D-8).
    sig_ri = sigma / np.sqrt(2.0)

    k = len(initial_peaks)
    n_data = 2 * m

    # --- null model: nothing to fit, but report the data's chi-squared. -----
    if k == 0:
        r0 = z / sig_ri
        chi2 = float(np.sum(r0.real**2 + r0.imag**2))
        return WindowFitResult(
            success=False,
            peaks=[],
            peak_errors=[],
            tau_us=tau0_us,
            tau_error=None,
            fit_tau=False,
            cost=0.5 * chi2,
            chi_squared=chi2,
            n_data=n_data,
            n_params=0,
            n_function_evals=0,
            fitted_spectrum=np.zeros(m, dtype=np.complex128),
            residual=z.copy(),
            covariance=None,
        )

    if tau_bounds is None:
        if max_decay_factor <= 1.0:
            raise ValueError("max_decay_factor must be greater than 1")
        tau_bounds = (tau0_us / max_decay_factor, tau0_us * max_decay_factor)
    if not tau_bounds[0] < tau_bounds[1]:
        raise ValueError("tau_bounds must be an increasing (lo, hi) pair")
    if offset_bounds is None:
        offset_bounds = (float(u.min()), float(u.max()))

    amp_upper = float(amp_max) if amp_max is not None and amp_max > 0.0 else np.inf
    if amp_penalty_lambda > 0.0 and amp_floor is None:
        raise ValueError("amp_floor is required when amp_penalty_lambda > 0")
    if phase_penalty_lambda > 0.0 and fwhm_mhz is None:
        raise ValueError("fwhm_mhz is required when phase_penalty_lambda > 0")
    if (
        tau_penalty_lambda > 0.0
        and fit_tau
        and (tau_penalty_reference is None or tau_penalty_reference <= 0.0)
    ):
        raise ValueError(
            "tau_penalty_reference (>0) is required when tau_penalty_lambda > 0 "
            "and fit_tau is True"
        )

    # --- bounds, in the packed parameter order ------------------------------
    lo: list[float] = []
    hi: list[float] = []
    for _ in range(k):
        lo += [0.0, offset_bounds[0], -_PHASE_BOUND]
        hi += [amp_upper, offset_bounds[1], _PHASE_BOUND]
    if fit_tau:
        lo.append(tau_bounds[0])
        hi.append(tau_bounds[1])
    lo_arr = np.asarray(lo, dtype=float)
    hi_arr = np.asarray(hi, dtype=float)
    p0 = np.clip(_pack(initial_peaks, tau0_us, fit_tau), lo_arr, hi_arr)

    penalty_kw = dict(
        phase_penalty_lambda=phase_penalty_lambda,
        amp_penalty_lambda=amp_penalty_lambda,
        amp_floor=amp_floor,
        fwhm_mhz=fwhm_mhz,
        phase_penalty_cutoff_fwhm=phase_penalty_cutoff_fwhm,
        tau_penalty_lambda=tau_penalty_lambda,
        tau_penalty_reference=tau_penalty_reference,
    )
    penalties_active = (
        phase_penalty_lambda > 0.0
        or amp_penalty_lambda > 0.0
        or (tau_penalty_lambda > 0.0 and fit_tau)
    )

    def residual(params: np.ndarray) -> np.ndarray:
        peaks, tau = _unpack(params, k, tau0_us, fit_tau)
        model = model_spectrum(u, peaks, tau, acquisition_us)
        r = (z - model) / sig_ri
        data_r = np.concatenate([r.real, r.imag])
        if not penalties_active:
            return cast(np.ndarray, data_r)
        pen_r, _ = _penalty_residuals_and_jacobian(
            params, k, tau0_us, fit_tau, **penalty_kw
        )
        return cast(np.ndarray, np.concatenate([data_r, pen_r]))

    def jacobian(params: np.ndarray) -> np.ndarray:
        peaks, tau = _unpack(params, k, tau0_us, fit_tau)
        # d(residual)/d(p) = -(d(model)/d(p)) / sig_ri, Re stacked over Im.
        dmodel = model_jacobian(u, peaks, tau, acquisition_us, include_tau=fit_tau)
        weighted = -dmodel / sig_ri[:, np.newaxis]
        data_jac = np.concatenate([weighted.real, weighted.imag], axis=0)
        if not penalties_active:
            return cast(np.ndarray, data_jac)
        _, pen_jac = _penalty_residuals_and_jacobian(
            params, k, tau0_us, fit_tau, **penalty_kw
        )
        return cast(np.ndarray, np.concatenate([data_jac, pen_jac], axis=0))

    try:
        sol = least_squares(
            residual,
            p0,
            jac=jacobian,
            bounds=(lo_arr, hi_arr),
            method="trf",
            max_nfev=max_nfev,
        )
    except (ValueError, np.linalg.LinAlgError):
        return WindowFitResult(
            success=False,
            peaks=list(initial_peaks),
            peak_errors=[
                ParameterErrors(float("nan"), float("nan"), float("nan"))
                for _ in range(k)
            ],
            tau_us=tau0_us,
            tau_error=float("nan") if fit_tau else None,
            fit_tau=fit_tau,
            cost=float("inf"),
            chi_squared=float("inf"),
            n_data=n_data,
            n_params=3 * k + (1 if fit_tau else 0),
            n_function_evals=0,
            fitted_spectrum=np.zeros(m, dtype=np.complex128),
            residual=z.copy(),
            covariance=None,
        )

    peaks, tau = _unpack(np.asarray(sol.x, dtype=float), k, tau0_us, fit_tau)
    for pk in peaks:
        pk.phase = _wrap_phase(pk.phase)

    fitted = model_spectrum(u, peaks, tau, acquisition_us)
    # Reported statistics are data-only (penalties act like a prior on the
    # parameters; the F-test / AIC across K stays calibrated only if chi^2
    # counts the data residual alone).
    data_resid = (z - fitted) / sig_ri
    chi2 = float(np.sum(data_resid.real**2 + data_resid.imag**2))
    cost_data = 0.5 * chi2

    covariance: Optional[np.ndarray] = None
    try:
        # Data-only Jacobian for uncertainty: penalties bias parameter errors
        # smaller (they're effectively a prior). Caller wants the data
        # likelihood's covariance.
        dmodel_sol = model_jacobian(
            u, peaks, tau, acquisition_us, include_tau=fit_tau
        )
        weighted_sol = -dmodel_sol / sig_ri[:, np.newaxis]
        data_jac = np.concatenate(
            [weighted_sol.real, weighted_sol.imag], axis=0
        )
        jtj = data_jac.T @ data_jac
        covariance = cast(np.ndarray, np.linalg.inv(jtj))
    except np.linalg.LinAlgError:
        covariance = None
    peak_errors, tau_error = _parameter_errors(covariance, k, fit_tau)

    return WindowFitResult(
        success=bool(sol.success),
        peaks=peaks,
        peak_errors=peak_errors,
        tau_us=tau,
        tau_error=tau_error,
        fit_tau=fit_tau,
        cost=cost_data,
        chi_squared=chi2,
        n_data=n_data,
        n_params=int(p0.size),
        n_function_evals=int(sol.nfev),
        fitted_spectrum=fitted,
        residual=z - fitted,
        covariance=covariance,
    )


# ---------------------------------------------------------------------------
# Conservative add-one-peak loop: audit trail and result types
# ---------------------------------------------------------------------------
@dataclass
class AddStep:
    """One decision in the conservative add-one-peak audit trail.

    Every iteration of :func:`conservative_fit` -- the seed, each blend-aware
    re-seed, and each candidate tested -- records an :class:`AddStep` so the
    loop's accept/reject behaviour can be validated and curated.

    Attributes
    ----------
    n_peaks_before : int
        Number of accepted lines before this step.
    candidate_offset_mhz : float
        Baseband offset of the line tested at this step.
    chi2_before, chi2_after : float
        Noise-weighted chi-squared of the model before / with the candidate.
    f_statistic, p_value : float
        Diagnostic nested-model F-test of the chi-squared improvement.
        Kept as a familiar statistic; the accept gate is
        AICc-with-``n_eff`` (see ``n_eff`` / ``aicc_delta``).
    aic_before, aic_after : float
        Diagnostic AIC at the raw ``n_data``.
    separation_ok : bool
        Whether the candidate cleared the peak-separation constraint.
    decision : str
        ``"seed"``, ``"seed-blend"``, ``"accept"``, ``"promote"``,
        ``"tentative"`` or ``"reject"``.
    reason : str
        Free-text note on the decision.
    n_eff : float
        Effective sample size shared by the K-vs-(K+1) AICc evaluation;
        computed once from the K+1 (trial) model magnitude (kind set by
        ``n_eff_kind`` on :func:`conservative_fit`). ``nan`` on steps
        that do not run the gate (the K=1 seed and separation-rejected
        candidates).
    aicc_delta : float
        ``AICc(K+1) - AICc(K)`` at the shared ``n_eff``; negative values
        mean the gate accepted (the K+1 model is preferred). The
        REJECT-on-tie convention reads ``aicc_delta >= 0`` as "no
        evidence for the more-complex model -- preserve K"; this is
        opposite to the merge/knockout gates' REJECT-on-tie because the
        comparison runs in the other direction (those test K-vs-(K-1)
        and prefer the more-complex K). ``nan`` on the same steps as
        ``n_eff``.
    """

    n_peaks_before: int
    candidate_offset_mhz: float
    chi2_before: float
    chi2_after: float
    f_statistic: float
    p_value: float
    aic_before: float
    aic_after: float
    separation_ok: bool
    decision: str
    reason: str = ""
    n_eff: float = float("nan")
    aicc_delta: float = float("nan")


@dataclass
class KnockoutResult:
    """Per-line knockout-test outcome.

    Removing a genuinely supported line and re-fitting the surviving (K-1)
    peaks (tau locked at the K-fit value) grows the chi-squared by enough
    that AICc, evaluated on an effective sample size that collapses to the
    bins the model actually informs, prefers the K-peak model. A duplicate
    or noise-amplitude line shows roughly indistinguishable chi-squared
    between K and (K-1), and the AICc gate prefers K-1.

    Attributes
    ----------
    peak_index : int
        Index of the knocked-out line in the fitted-peak list.
    offset_mhz : float
        Baseband offset of the line.
    delta_chi2 : float
        Diagnostic: chi-squared increase when the line is removed and every
        other parameter is held frozen at the K-fit value. The "energy
        carried by this line" check; meaningful in isolation but no longer
        the gate (frozen others leave duplicate twins half-fit and produce
        spurious large delta_chi2).
    expected_delta_chi2 : float
        Diagnostic: the line's own noise-weighted energy -- the increase a
        real line should produce under the freeze-others convention.
    supported : bool
        Whether the AICc-with-n_eff gate prefers the K-peak fit
        (``aicc_delta >= 0``; REJECT-on-tie matches the merge-cleanup
        convention). A peak whose removal-and-refit produces a strictly
        better AICc has ``supported = False``.
    p_value : float
        Diagnostic F-test p-value of the K-peak fit vs the (K-1)-peak
        refit (not freeze-others); kept as a familiar statistic but no
        longer the decision rule. ``nan`` for an empty fit or when the
        refit failed to converge.
    n_eff : float
        Effective sample size used by the AICc gate; computed once from
        the K-fit model magnitude (kind set by ``n_eff_kind``) and shared
        across all peak comparisons in this sweep.
    aicc_delta : float
        ``AICc(K-1 refit) - AICc(K)`` at the shared ``n_eff``. Negative
        values say the simpler model is preferred (peak is redundant);
        ``supported = aicc_delta < 0`` reads "the more complex model is
        not preferred". ``nan`` when the refit failed to converge.
    """

    peak_index: int
    offset_mhz: float
    delta_chi2: float
    expected_delta_chi2: float
    supported: bool
    p_value: float = float("nan")
    n_eff: float = float("nan")
    aicc_delta: float = float("nan")


@dataclass
class ConservativeFitResult:
    """Outcome of the conservative add-one-peak loop.

    Attributes
    ----------
    fit : WindowFitResult
        The final fixed-K fit of the accepted lines.
    audit_trail : list of AddStep
        Every seed / re-seed / candidate decision, in order.
    knockouts : list of KnockoutResult
        Per-line knockout validation of the final fit.
    """

    fit: WindowFitResult
    audit_trail: list[AddStep]
    knockouts: list[KnockoutResult]

    @property
    def success(self) -> bool:
        """Whether the final fit converged."""
        return self.fit.success

    @property
    def peaks(self) -> list[ModelPeak]:
        """The accepted lines (fit-frame coordinates)."""
        return self.fit.peaks

    @property
    def n_peaks(self) -> int:
        """Number of accepted lines."""
        return self.fit.n_peaks

    @property
    def n_iterations(self) -> int:
        """Number of recorded add-one-peak decisions."""
        return len(self.audit_trail)


# ---------------------------------------------------------------------------
# Knockout validation
# ---------------------------------------------------------------------------
def knockout_test(
    offset_grid_mhz: np.ndarray,
    complex_spectrum: np.ndarray,
    rms_noise: NoiseLike,
    fit: WindowFitResult,
    acquisition_us: float,
    *,
    fit_kwargs_inner: Optional[dict[str, Any]] = None,
    n_eff_kind: str = DEFAULT_N_EFF_KIND,
    significance: float = DEFAULT_SIGNIFICANCE,
) -> list[KnockoutResult]:
    """Per-line knockout validation of a converged window fit.

    For each fitted line: remove it from the model, refit the surviving
    (K-1) peaks freely starting from their K-fit parameters, with tau
    locked at the K-fit value. Compare AICc on a shared effective sample
    size; the peak is **supported** when AICc prefers the K-peak model
    (REJECT-on-tie matches the merge-cleanup convention).

    The refit (not freeze-others) is the structural fix for the
    duplicate-pair pathology: if peak A and C are both fit at half the
    line's true amplitude at the same physical offset, freezing C while
    removing A leaves a half-fit residual and produces a huge
    delta_chi2 that flags A as supported -- and symmetrically for C.
    With the refit, C re-converges to full amplitude when A is removed
    and the (K-1) chi-squared matches the K chi-squared, so AICc prefers
    K-1 and both duplicates flip to ``supported = False``. See
    `dev-docs/planning/stage5-residual-rescue.md`, Phase 2.

    Tau is locked (``fit_tau=False``) in the (K-1) refit because tau is
    effectively a dataset-shared parameter (transit time x natural
    lifetime); a single-window (K-1) refit must not get the extra knob
    of broadening tau to compensate for the removed peak.

    Parameters
    ----------
    offset_grid_mhz : np.ndarray
        Baseband-offset grid of the window.
    complex_spectrum : np.ndarray
        Complex active-FT window data.
    rms_noise : float or np.ndarray
        Per-bin complex noise RMS.
    fit : WindowFitResult
        The converged K-peak fit to validate.
    acquisition_us : float
        Active acquisition length ``T`` (microseconds).
    fit_kwargs_inner : dict, optional
        Tau / amplitude / penalty constraints to enforce in the (K-1)
        refit -- the same bag :func:`derive_window_fit_constraints`
        produces for :func:`conservative_fit`. The refit forces
        ``fit_tau=False`` (tau locked at ``fit.tau_us``) regardless of
        what this bag says. ``None`` defaults to the empty bag (only the
        tau lock is applied); production callers should pass the
        constraints derived for the K-fit.
    n_eff_kind : str, default :data:`DEFAULT_N_EFF_KIND`
        Effective-sample-size weighting kind passed to
        :func:`effective_sample_size`. Shared across all peaks in the
        sweep (n_eff is computed once from the K-fit model magnitude).
    significance : float, default :data:`DEFAULT_SIGNIFICANCE`
        Threshold for the diagnostic F-test ``p_value`` only; ``supported``
        keys off ``aicc_delta`` (REJECT-on-tie).

    Returns
    -------
    list of KnockoutResult
        One entry per fitted line, in fitted-peak order. ``delta_chi2`` /
        ``expected_delta_chi2`` keep the freeze-others "energy carried by
        this line" diagnostic; ``p_value`` is now the refit-based F-test
        diagnostic; ``supported`` / ``aicc_delta`` are the new gate.
    """
    peaks = fit.peaks
    if not peaks:
        return []

    u = np.asarray(offset_grid_mhz, dtype=float)
    z = np.asarray(complex_spectrum, dtype=np.complex128)
    sigma = np.asarray(rms_noise, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(u.size, float(sigma))

    tau = fit.tau_us
    full_model = model_spectrum(u, peaks, tau, acquisition_us)
    full_chi2 = calculate_noise_weighted_chi2(complex_spectrum, rms_noise, full_model)

    # n_eff is keyed on the K-fit's model magnitude -- the same value the
    # merge cleanup uses for its AICc test -- and shared across all peaks
    # in the sweep so per-peak comparisons sit on a common scale.
    n_eff = effective_sample_size(
        fit.fitted_spectrum, kind=n_eff_kind, sigma=rms_noise,
    )
    aicc_k = calculate_aicc(fit.chi_squared, fit.n_params, n_eff)

    refit_kwargs: dict[str, Any] = dict(fit_kwargs_inner or {})
    refit_kwargs["fit_tau"] = False  # tau locked at the K-fit value

    results: list[KnockoutResult] = []
    for i, pk in enumerate(peaks):
        kept = [p for j, p in enumerate(peaks) if j != i]
        # Diagnostic: freeze-others delta_chi2 + expected line energy.
        kept_model = model_spectrum(u, kept, tau, acquisition_us)
        chi2_without = calculate_noise_weighted_chi2(
            complex_spectrum, rms_noise, kept_model
        )
        delta = chi2_without - full_chi2
        line_model = model_spectrum(u, [pk], tau, acquisition_us)
        expected = calculate_noise_weighted_chi2(line_model, rms_noise)

        if not kept:
            # K=1 -> K=0 refit is the null model; no fit_window call needed.
            # The null chi-squared is the data's own noise-weighted energy.
            null_chi2 = calculate_noise_weighted_chi2(complex_spectrum, rms_noise)
            aicc_km1 = calculate_aicc(null_chi2, 0, n_eff)
            # Compare aicc_km1 to aicc_k directly so the both-+inf case
            # (model not identifiable at n_eff for either K or K-1) reads
            # as a tie and preserves the K-peak fit -- matches the merge
            # gate's REJECT-on-tie convention.
            supported = aicc_km1 >= aicc_k
            p_value, _, _ = calculate_chi_squared_improvement(
                null_chi2, fit.chi_squared, 3, fit.n_data, fit.n_params
            )
            results.append(
                KnockoutResult(
                    peak_index=i,
                    offset_mhz=pk.offset_mhz,
                    delta_chi2=delta,
                    expected_delta_chi2=expected,
                    supported=bool(supported),
                    p_value=float(p_value),
                    n_eff=float(n_eff),
                    aicc_delta=float(aicc_km1 - aicc_k),
                )
            )
            continue

        refit = fit_window(
            u, z, sigma, kept, tau, acquisition_us, **refit_kwargs,
        )
        if not refit.success:
            # Refit failure -> the AICc gate is unable to express a
            # preference. Treat the peak as supported (REJECT-on-tie's
            # conservative direction) and report NaN for the gate fields
            # so downstream readers can distinguish "we tried and failed"
            # from a legitimate decision.
            results.append(
                KnockoutResult(
                    peak_index=i,
                    offset_mhz=pk.offset_mhz,
                    delta_chi2=delta,
                    expected_delta_chi2=expected,
                    supported=True,
                    p_value=float("nan"),
                    n_eff=float(n_eff),
                    aicc_delta=float("nan"),
                )
            )
            continue

        aicc_km1 = calculate_aicc(refit.chi_squared, refit.n_params, n_eff)
        # Diagnostic p_value: refit-based F-test (K-1 simpler vs K complex).
        p_value, _, _ = calculate_chi_squared_improvement(
            refit.chi_squared, fit.chi_squared, 3, fit.n_data, fit.n_params
        )
        # REJECT-on-tie: supported iff AICc(K-1) is not strictly better
        # than AICc(K). Comparing the two AICc values directly (rather
        # than via the subtraction) makes the both-+inf case (model not
        # identifiable at n_eff for either K or K-1) read as a tie and
        # preserve the K-peak fit; ``inf - inf`` would be NaN and silently
        # drop the peak. Matches the merge-cleanup convention.
        supported = aicc_km1 >= aicc_k
        results.append(
            KnockoutResult(
                peak_index=i,
                offset_mhz=pk.offset_mhz,
                delta_chi2=delta,
                expected_delta_chi2=expected,
                supported=bool(supported),
                p_value=float(p_value),
                n_eff=float(n_eff),
                aicc_delta=float(aicc_km1 - aicc_k),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Conservative add-one-peak loop
# ---------------------------------------------------------------------------
def _seed_peak(
    offset_mhz: float,
    offset_grid_mhz: np.ndarray,
    residual_spectrum: np.ndarray,
    tau0_us: float,
    acquisition_us: float,
) -> ModelPeak:
    """Initial-guess line at ``offset_mhz`` from the local residual.

    Amplitude from the residual magnitude divided by the on-line gain
    ``tau_eff``, phase from the residual phase. ``offset_grid_mhz`` must be
    ascending (``np.interp`` requirement).
    """
    mag = float(
        np.abs(np.interp(offset_mhz, offset_grid_mhz, np.abs(residual_spectrum)))
    )
    gain = max(effective_tau(tau0_us, acquisition_us), 1e-9)
    phase = float(np.interp(offset_mhz, offset_grid_mhz, np.angle(residual_spectrum)))
    return ModelPeak(
        amplitude=max(2.0 * mag / gain, 1e-6), offset_mhz=offset_mhz, phase=phase
    )


def _blend_aware_seed(
    offset_grid_mhz: np.ndarray,
    complex_spectrum: np.ndarray,
    rms_noise: NoiseLike,
    seed_offset_mhz: float,
    tau0_us: float,
    acquisition_us: float,
    *,
    fit_tau: bool,
    tau_bounds: tuple[float, float],
    fwhm_mhz: float,
    null_chi2: float,
    null_aic: float,
    significance: float,
    rchi2_threshold: float,
    straddle_factor: float,
    max_k: int,
    amp_max: Optional[float] = None,
    amp_floor: Optional[float] = None,
    phase_penalty_lambda: float = 0.0,
    amp_penalty_lambda: float = 0.0,
    phase_penalty_cutoff_fwhm: float = DEFAULT_PHASE_PENALTY_CUTOFF_FWHM,
    min_pair_separation_factor: float = DEFAULT_MIN_PAIR_SEPARATION_FACTOR,
    tau_penalty_lambda: float = 0.0,
    tau_penalty_reference: Optional[float] = None,
    n_eff_kind: str = DEFAULT_CONSERVATIVE_N_EFF_KIND,
) -> tuple[WindowFitResult, list[AddStep]]:
    """Seed the window fit, escalating K=1 -> K=2 -> K=3 on an elevated chi².

    A single-cosine seed fit that leaves an elevated reduced chi-squared is
    treated as an unresolved blend (the prototype's key finding -- the failure
    is *initialisation*, not detectability). The fit is then retried with K
    lines initialised at positions straddling the seed, accepting each
    escalation only on the AICc-with-``n_eff`` gate (REJECT-on-tie: the
    K+1 model must score strictly better; a tied or worse score preserves
    the simpler K-peak fit). ``offset_grid_mhz`` must be ascending.

    Each K=2/K=3 trial fit is also post-checked for the "two peaks collapsed
    onto the same offset with cancelling phases" degenerate solution
    (``min_pair_separation_factor * fwhm`` is the minimum allowed pair
    separation); collapsed escalations are rejected even when the gate would
    accept them.

    ``significance`` is not used as a gate threshold; the F-test
    ``p_value`` is computed and recorded on each :class:`AddStep` as a
    familiar diagnostic.
    """
    fit_kwargs = dict(
        fit_tau=fit_tau,
        tau_bounds=tau_bounds,
        amp_max=amp_max,
        amp_floor=amp_floor,
        fwhm_mhz=fwhm_mhz,
        phase_penalty_lambda=phase_penalty_lambda,
        amp_penalty_lambda=amp_penalty_lambda,
        phase_penalty_cutoff_fwhm=phase_penalty_cutoff_fwhm,
        tau_penalty_lambda=tau_penalty_lambda,
        tau_penalty_reference=tau_penalty_reference,
    )
    fit1 = fit_window(
        offset_grid_mhz,
        complex_spectrum,
        rms_noise,
        [
            _seed_peak(
                seed_offset_mhz,
                offset_grid_mhz,
                complex_spectrum,
                tau0_us,
                acquisition_us,
            )
        ],
        tau0_us,
        acquisition_us,
        **fit_kwargs,
    )
    p1, f1, _ = calculate_chi_squared_improvement(
        null_chi2, fit1.chi_squared, fit1.n_params, fit1.n_data, fit1.n_params
    )
    audit = [
        AddStep(
            n_peaks_before=0,
            candidate_offset_mhz=seed_offset_mhz,
            chi2_before=null_chi2,
            chi2_after=fit1.chi_squared,
            f_statistic=f1,
            p_value=p1,
            aic_before=null_aic,
            aic_after=fit1.aic,
            separation_ok=True,
            decision="seed",
            reason="K=1 seed fit",
        )
    ]

    best = fit1
    if not fit1.success or fit1.reduced_chi2 <= rchi2_threshold:
        return best, audit

    # Elevated reduced chi-squared -> retry as a straddled blend.
    straddle = straddle_factor * fwhm_mhz
    min_pair_sep = min_pair_separation_factor * fwhm_mhz
    prev = fit1
    for k in range(2, max_k + 1):
        positions = seed_offset_mhz + (np.arange(k) - 0.5 * (k - 1)) * straddle
        init = [
            _seed_peak(
                float(pos), offset_grid_mhz, complex_spectrum, tau0_us, acquisition_us
            )
            for pos in positions
        ]
        trial = fit_window(
            offset_grid_mhz,
            complex_spectrum,
            rms_noise,
            init,
            tau0_us,
            acquisition_us,
            **fit_kwargs,
        )
        p_value, f_stat, _ = calculate_chi_squared_improvement(
            prev.chi_squared,
            trial.chi_squared,
            trial.n_params - prev.n_params,
            trial.n_data,
            trial.n_params,
        )
        # Post-fit sanity check: reject escalations whose peaks collapsed onto
        # the same offset (the cancelling-phase degenerate solution).
        collapsed = False
        if trial.success and len(trial.peaks) >= 2 and min_pair_sep > 0.0:
            offs = np.asarray([pk.offset_mhz for pk in trial.peaks], dtype=float)
            for ii in range(offs.size):
                for jj in range(ii + 1, offs.size):
                    if abs(offs[ii] - offs[jj]) < min_pair_sep:
                        collapsed = True
                        break
                if collapsed:
                    break
        # AICc-with-n_eff gate. The K+1 trial model is the magnitude
        # basis: its fitted_spectrum defines the informative bins, and
        # both AICc evaluations share that ``n_eff`` so they sit on a
        # common scale. REJECT-on-tie: ``aicc_trial < aicc_prev`` is
        # strictly less, so the both-+inf case (model not identifiable
        # at n_eff for either K or K+1) reads as a tie and preserves
        # the K-peak fit. Same conservative principle as the merge /
        # knockout gates (a tie always preserves K), with the sign
        # inverted because here K+1 is the more-complex model.
        if trial.success:
            n_eff = effective_sample_size(
                trial.fitted_spectrum, kind=n_eff_kind, sigma=rms_noise,
            )
            aicc_prev = calculate_aicc(prev.chi_squared, prev.n_params, n_eff)
            aicc_trial = calculate_aicc(trial.chi_squared, trial.n_params, n_eff)
            aicc_delta = aicc_trial - aicc_prev
            gate_accepts = aicc_trial < aicc_prev
        else:
            n_eff = float("nan")
            aicc_delta = float("nan")
            gate_accepts = False
        accepted = trial.success and gate_accepts and not collapsed
        reason = f"K={k} straddled re-seed"
        if collapsed and trial.success:
            reason += " (rejected: peaks collapsed within min separation)"
        audit.append(
            AddStep(
                n_peaks_before=prev.n_peaks,
                candidate_offset_mhz=seed_offset_mhz,
                chi2_before=prev.chi_squared,
                chi2_after=trial.chi_squared,
                f_statistic=f_stat,
                p_value=p_value,
                aic_before=prev.aic,
                aic_after=trial.aic,
                separation_ok=not collapsed,
                decision="seed-blend" if accepted else "reject",
                reason=reason,
                n_eff=float(n_eff),
                aicc_delta=float(aicc_delta),
            )
        )
        if not accepted:
            break
        best = trial
        prev = trial
        if trial.reduced_chi2 <= rchi2_threshold:
            break
    return best, audit


def conservative_fit(
    offset_grid_mhz: np.ndarray,
    complex_spectrum: np.ndarray,
    rms_noise: NoiseLike,
    candidate_offsets: Sequence[float],
    tau0_us: float,
    acquisition_us: float,
    *,
    fit_tau: bool = True,
    significance: float = DEFAULT_SIGNIFICANCE,
    min_separation_factor: float = DEFAULT_MIN_SEPARATION_FACTOR,
    max_peaks: int = DEFAULT_MAX_PEAKS,
    patience: int = DEFAULT_PATIENCE,
    max_decay_factor: float = DEFAULT_MAX_DECAY_FACTOR,
    seeder_rchi2_threshold: float = DEFAULT_SEEDER_RCHI2,
    seeder_straddle_factor: float = DEFAULT_SEEDER_STRADDLE_FACTOR,
    seeder_max_k: int = DEFAULT_SEEDER_MAX_K,
    phase_penalty_lambda: float = DEFAULT_PHASE_PENALTY_LAMBDA,
    amp_penalty_lambda: float = DEFAULT_AMP_PENALTY_LAMBDA,
    phase_penalty_cutoff_fwhm: float = DEFAULT_PHASE_PENALTY_CUTOFF_FWHM,
    amp_max_headroom: float = DEFAULT_AMP_MAX_HEADROOM,
    min_pair_separation_factor: float = DEFAULT_MIN_PAIR_SEPARATION_FACTOR,
    tau_penalty_lambda: float = DEFAULT_TAU_PENALTY_LAMBDA,
    weak_window_snr_threshold: float = DEFAULT_WEAK_WINDOW_SNR_THRESHOLD,
    tau_apodization_us: Optional[float] = None,
    n_eff_kind: str = DEFAULT_CONSERVATIVE_N_EFF_KIND,
    knockout_n_eff_kind: str = DEFAULT_N_EFF_KIND,
) -> ConservativeFitResult:
    """Conservative incremental peak fitting of one window.

    Seeds with the strongest candidate (escalating to a straddled K=2/K=3 fit
    if the single-cosine seed leaves an elevated reduced chi-squared -- the
    blend-aware seeder), then repeatedly trial-fits the strongest remaining
    residual candidate, accepting it only when the AICc-with-``n_eff`` gate
    (REJECT-on-tie: ``AICc(K+1) < AICc(K)`` strictly) prefers the K+1 model.
    A candidate that violates the peak-separation constraint is dropped; a
    rejected candidate is held tentatively and the loop tolerates
    ``patience`` consecutive rejections (a tentative batch is promoted whole
    if it later becomes jointly significant). The converged fit is validated
    by :func:`knockout_test`.

    Parameters
    ----------
    offset_grid_mhz : np.ndarray
        Baseband-offset grid of the window (any order; sorted internally).
    complex_spectrum : np.ndarray
        Complex active-FT window data on the grid.
    rms_noise : float or np.ndarray
        Per-bin complex noise RMS.
    candidate_offsets : sequence of float
        Baseband offsets of the promoted peaks to consider for this window.
    tau0_us : float
        Default / starting shared decay constant (microseconds).
    acquisition_us : float
        Active acquisition length ``T`` (microseconds).
    fit_tau : bool, default True
        Fit the shared ``tau`` freely; ``False`` holds it (weak-only windows).
    significance : float, default 0.05
        Diagnostic F-test threshold. The accept gate is AICc-with-``n_eff``;
        the F-test ``p_value`` is recorded on each :class:`AddStep` as a
        familiar diagnostic but is not used to decide acceptance.
    min_separation_factor : float, default 1.0
        Minimum peak separation as a multiple of the feature FWHM.
    max_peaks : int, default 8
        Cap on the number of lines fit.
    patience : int, default 1
        Consecutive candidate rejections tolerated before the loop stops.
    max_decay_factor : float, default 5.0
        ``tau`` bound factor (see :func:`fit_window`).
    seeder_rchi2_threshold : float, default 1.5
        Reduced chi-squared above which the seed is re-fit as a blend.
    seeder_straddle_factor : float, default 1.0
        Straddle of the blend re-seed, in feature-FWHM units.
    seeder_max_k : int, default 3
        Largest K the blend-aware seeder escalates to.
    phase_penalty_lambda : float, default :data:`DEFAULT_PHASE_PENALTY_LAMBDA`
        Soft pair-phase penalty weight (see
        :func:`_penalty_residuals_and_jacobian`). Catches the degenerate
        "two peaks collapsed onto the same offset with cancelling phases"
        blend-aware re-seed pathology. ``0`` disables.
    amp_penalty_lambda : float, default :data:`DEFAULT_AMP_PENALTY_LAMBDA`
        Soft amplitude-floor penalty weight; pushes noise-amplitude peaks
        toward zero. ``0`` disables.
    phase_penalty_cutoff_fwhm : float, default
        :data:`DEFAULT_PHASE_PENALTY_CUTOFF_FWHM`
        Pair-separation cutoff for the phase penalty, in FWHM units.
    amp_max_headroom : float, default :data:`DEFAULT_AMP_MAX_HEADROOM`
        Hard amplitude upper bound is set to
        ``amp_max_headroom * 2 * max(|z|) / tau_eff(tau_bounds[0], T)`` -- a
        few times the strongest physically-plausible amplitude at the tightest
        bound, which still rejects the cancelling-pair pathology's inflated
        amplitudes.
    min_pair_separation_factor : float, default
        :data:`DEFAULT_MIN_PAIR_SEPARATION_FACTOR`
        Post-fit sanity-check threshold in FWHM units; the blend-aware seeder
        rejects an escalation whose fitted peaks ended up within
        ``min_pair_separation_factor * fwhm`` of each other.
    tau_penalty_lambda : float, default :data:`DEFAULT_TAU_PENALTY_LAMBDA`
        Weight of the lower-side tau penalty (see
        :func:`_penalty_residuals_and_jacobian`). ``0`` disables. The
        penalty pulls tau toward ``tau_apodization_us`` (the apodization
        ceiling); the LSQ otherwise tends to broaden the line by lowering
        tau to absorb unmodeled-peak residual.
    weak_window_snr_threshold : float, default
        :data:`DEFAULT_WEAK_WINDOW_SNR_THRESHOLD`
        Windows whose strongest in-window magnitude is below this multiple
        of the median active-FT noise hold tau fixed entirely (``fit_tau``
        is forced to ``False`` regardless of the input). Weak windows
        carry no information to fit tau and would otherwise pin it at the
        lower bound.
    tau_apodization_us : float, optional
        Apodization ``expf_us`` (the hard upper bound on tau). When set:
        (a) tau is bounded above by ``min(tau0_us * max_decay_factor,
        tau_apodization_us)``; and (b) the tau penalty is referenced to it.
        When ``None`` the tau penalty is disabled and the upper bound stays
        ``tau0_us * max_decay_factor``.
    n_eff_kind : str, default :data:`DEFAULT_CONSERVATIVE_N_EFF_KIND`
        Effective-sample-size kind for the conservative add-one-peak accept
        gate (this loop and :func:`_blend_aware_seed` 's K=2/K=3
        escalation). The K-vs-(K+1) comparisons use a per-bin information
        weight so n_eff stays in the AICc-identifiable regime on narrow
        features.
    knockout_n_eff_kind : str, default :data:`DEFAULT_N_EFF_KIND`
        Effective-sample-size kind for the final :func:`knockout_test`
        sweep. The K-vs-(K-1) comparisons use the magnitude-concentrated
        weight so the structural divergence of AICc at small ``n_eff``
        falls through to "preserve K" (do not drop the peak).

    Returns
    -------
    ConservativeFitResult
        The final fit, the add-one-peak audit trail, and the knockout results.
    """
    u = np.asarray(offset_grid_mhz, dtype=float)
    z = np.asarray(complex_spectrum, dtype=np.complex128)
    sigma = np.asarray(rms_noise, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(u.size, float(sigma))
    # np.interp needs an ascending grid; sort the window once on entry.
    order = np.argsort(u)
    u, z, sigma = u[order], z[order], sigma[order]

    constraints = derive_window_fit_constraints(
        z,
        sigma,
        tau0_us,
        acquisition_us,
        fit_tau=fit_tau,
        min_separation_factor=min_separation_factor,
        max_decay_factor=max_decay_factor,
        amp_max_headroom=amp_max_headroom,
        phase_penalty_lambda=phase_penalty_lambda,
        amp_penalty_lambda=amp_penalty_lambda,
        phase_penalty_cutoff_fwhm=phase_penalty_cutoff_fwhm,
        tau_penalty_lambda=tau_penalty_lambda,
        weak_window_snr_threshold=weak_window_snr_threshold,
        tau_apodization_us=tau_apodization_us,
    )
    tau_bounds = constraints.tau_bounds
    fwhm = constraints.fwhm
    min_separation = constraints.min_separation
    amp_max = constraints.amp_max
    amp_floor = constraints.amp_floor
    fit_tau_eff = constraints.fit_tau_eff
    tau_penalty_ref = constraints.tau_penalty_reference
    effective_tau_penalty_lambda = constraints.effective_tau_penalty_lambda
    fit_kwargs_inner = constraints.fit_kwargs_inner

    remaining = sorted(
        candidate_offsets,
        key=lambda o: -abs(float(np.interp(o, u, np.abs(z)))),
    )
    if not remaining:
        empty = fit_window(u, z, sigma, [], tau0_us, acquisition_us)
        return ConservativeFitResult(empty, [], [])

    null = fit_window(u, z, sigma, [], tau0_us, acquisition_us)
    seed = remaining.pop(0)
    current, audit = _blend_aware_seed(
        u,
        z,
        sigma,
        seed,
        tau0_us,
        acquisition_us,
        fit_tau=fit_tau_eff,
        tau_bounds=tau_bounds,
        fwhm_mhz=fwhm,
        null_chi2=null.chi_squared,
        null_aic=null.aic,
        significance=significance,
        rchi2_threshold=seeder_rchi2_threshold,
        straddle_factor=seeder_straddle_factor,
        max_k=seeder_max_k,
        amp_max=amp_max,
        amp_floor=amp_floor,
        phase_penalty_lambda=phase_penalty_lambda,
        amp_penalty_lambda=amp_penalty_lambda,
        phase_penalty_cutoff_fwhm=phase_penalty_cutoff_fwhm,
        min_pair_separation_factor=min_pair_separation_factor,
        tau_penalty_lambda=effective_tau_penalty_lambda,
        tau_penalty_reference=tau_penalty_ref,
        n_eff_kind=n_eff_kind,
    )
    if not current.success:
        return ConservativeFitResult(current, audit, [])

    tentative: list[ModelPeak] = []
    consecutive_rejects = 0
    while remaining and current.n_peaks + len(tentative) < max_peaks:
        in_model = list(current.peaks) + tentative
        residual = z - model_spectrum(u, in_model, current.tau_us, acquisition_us)
        cand = max(
            remaining,
            key=lambda o: abs(float(np.interp(o, u, np.abs(residual)))),
        )
        remaining.remove(cand)

        existing = [pk.offset_mhz for pk in in_model]
        sep_ok, _ = validate_peak_separation(
            np.asarray(existing + [cand]), min_separation
        )
        if not sep_ok:
            audit.append(
                AddStep(
                    n_peaks_before=current.n_peaks,
                    candidate_offset_mhz=cand,
                    chi2_before=current.chi_squared,
                    chi2_after=current.chi_squared,
                    f_statistic=0.0,
                    p_value=1.0,
                    aic_before=current.aic,
                    aic_after=current.aic,
                    separation_ok=False,
                    decision="reject",
                    reason="peak-separation constraint",
                )
            )
            continue

        # Trial = accepted + the whole tentative batch + this candidate, tested
        # against the last accepted model so a jointly-significant batch is
        # promoted even when its members are not individually significant.
        trial_init = (
            list(current.peaks)
            + tentative
            + [_seed_peak(cand, u, residual, current.tau_us, acquisition_us)]
        )
        trial = fit_window(
            u,
            z,
            sigma,
            trial_init,
            tau0_us,
            acquisition_us,
            **fit_kwargs_inner,
        )
        p_value, f_stat, _ = calculate_chi_squared_improvement(
            current.chi_squared,
            trial.chi_squared,
            trial.n_params - current.n_params,
            trial.n_data,
            trial.n_params,
        )
        # AICc-with-n_eff gate. The trial (K+1) model is the magnitude
        # basis: its fitted_spectrum defines the informative bins, and
        # both AICc evaluations share that ``n_eff`` so they sit on a
        # common scale. REJECT-on-tie: ``aicc_trial < aicc_current`` is
        # strictly less, so the both-+inf case (model not identifiable
        # at n_eff for either K or K+1) reads as a tie and preserves
        # the K-peak fit. Same conservative principle as the merge /
        # knockout gates (a tie always preserves K), with the sign
        # inverted because here K+1 is the more-complex model.
        if trial.success:
            n_eff = effective_sample_size(
                trial.fitted_spectrum, kind=n_eff_kind, sigma=sigma,
            )
            aicc_current = calculate_aicc(
                current.chi_squared, current.n_params, n_eff
            )
            aicc_trial = calculate_aicc(trial.chi_squared, trial.n_params, n_eff)
            aicc_delta = aicc_trial - aicc_current
            passes = aicc_trial < aicc_current
        else:
            n_eff = float("nan")
            aicc_delta = float("nan")
            passes = False
        if passes:
            decision = "promote" if tentative else "accept"
            audit.append(
                AddStep(
                    n_peaks_before=current.n_peaks,
                    candidate_offset_mhz=cand,
                    chi2_before=current.chi_squared,
                    chi2_after=trial.chi_squared,
                    f_statistic=f_stat,
                    p_value=p_value,
                    aic_before=current.aic,
                    aic_after=trial.aic,
                    separation_ok=True,
                    decision=decision,
                    reason=f"+{len(tentative) + 1} line(s)",
                    n_eff=float(n_eff),
                    aicc_delta=float(aicc_delta),
                )
            )
            current = trial
            tentative = []
            consecutive_rejects = 0
        else:
            audit.append(
                AddStep(
                    n_peaks_before=current.n_peaks,
                    candidate_offset_mhz=cand,
                    chi2_before=current.chi_squared,
                    chi2_after=trial.chi_squared,
                    f_statistic=f_stat,
                    p_value=p_value,
                    aic_before=current.aic,
                    aic_after=trial.aic,
                    separation_ok=True,
                    decision="tentative",
                    reason="held pending a jointly-significant batch",
                    n_eff=float(n_eff),
                    aicc_delta=float(aicc_delta),
                )
            )
            tentative.append(
                _seed_peak(cand, u, residual, current.tau_us, acquisition_us)
            )
            consecutive_rejects += 1
            if consecutive_rejects > patience:
                break

    knockouts = knockout_test(
        u, z, sigma, current, acquisition_us,
        fit_kwargs_inner=fit_kwargs_inner,
        n_eff_kind=knockout_n_eff_kind,
        significance=significance,
    )
    return ConservativeFitResult(current, audit, knockouts)
