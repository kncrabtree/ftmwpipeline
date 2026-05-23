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
from typing import Optional, Union, cast

import numpy as np
from scipy.optimize import least_squares

from .peak_model import ModelPeak, effective_tau, h_T, h_T_jacobian, model_spectrum
from .validation import (
    calculate_aic,
    calculate_chi_squared_improvement,
    calculate_noise_weighted_chi2,
    feature_fwhm,
    validate_peak_separation,
)

__all__ = [
    "ParameterErrors",
    "WindowFitResult",
    "AddStep",
    "KnockoutResult",
    "ConservativeFitResult",
    "model_jacobian",
    "fit_window",
    "knockout_test",
    "conservative_fit",
]

NoiseLike = Union[float, np.ndarray]

# Default solver evaluation cap, ported from the bcfitting reference shell.
DEFAULT_MAX_NFEV = 400
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

    # --- bounds, in the packed parameter order ------------------------------
    lo: list[float] = []
    hi: list[float] = []
    for _ in range(k):
        lo += [0.0, offset_bounds[0], -_PHASE_BOUND]
        hi += [np.inf, offset_bounds[1], _PHASE_BOUND]
    if fit_tau:
        lo.append(tau_bounds[0])
        hi.append(tau_bounds[1])
    lo_arr = np.asarray(lo, dtype=float)
    hi_arr = np.asarray(hi, dtype=float)
    p0 = np.clip(_pack(initial_peaks, tau0_us, fit_tau), lo_arr, hi_arr)

    def residual(params: np.ndarray) -> np.ndarray:
        peaks, tau = _unpack(params, k, tau0_us, fit_tau)
        model = model_spectrum(u, peaks, tau, acquisition_us)
        r = (z - model) / sig_ri
        return cast(np.ndarray, np.concatenate([r.real, r.imag]))

    def jacobian(params: np.ndarray) -> np.ndarray:
        peaks, tau = _unpack(params, k, tau0_us, fit_tau)
        # d(residual)/d(p) = -(d(model)/d(p)) / sig_ri, Re stacked over Im.
        dmodel = model_jacobian(u, peaks, tau, acquisition_us, include_tau=fit_tau)
        weighted = -dmodel / sig_ri[:, np.newaxis]
        return cast(np.ndarray, np.concatenate([weighted.real, weighted.imag], axis=0))

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
    chi2 = 2.0 * float(sol.cost)  # scipy cost = 0.5 * sum(r**2)

    covariance: Optional[np.ndarray] = None
    try:
        jtj = np.asarray(sol.jac, dtype=float).T @ np.asarray(sol.jac, dtype=float)
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
        cost=float(sol.cost),
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
        Nested-model F-test of the chi-squared improvement.
    aic_before, aic_after : float
        AIC before / with the candidate.
    separation_ok : bool
        Whether the candidate cleared the peak-separation constraint.
    decision : str
        ``"seed"``, ``"seed-blend"``, ``"accept"``, ``"promote"``,
        ``"tentative"`` or ``"reject"``.
    reason : str
        Free-text note on the decision.
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


@dataclass
class KnockoutResult:
    """Per-line knockout-test outcome.

    Removing a genuinely supported line from the converged model -- holding
    every other parameter frozen -- grows the chi-squared by very nearly that
    line's own weighted energy. A line that can be knocked out without the
    expected response was not supported by the data and is flagged.

    Attributes
    ----------
    peak_index : int
        Index of the knocked-out line in the fitted-peak list.
    offset_mhz : float
        Baseband offset of the line.
    delta_chi2 : float
        Observed chi-squared increase when the line is removed.
    expected_delta_chi2 : float
        The line's own noise-weighted energy -- the increase a real line
        should produce.
    supported : bool
        Whether removing the line significantly worsens the fit (F-test).
    """

    peak_index: int
    offset_mhz: float
    delta_chi2: float
    expected_delta_chi2: float
    supported: bool


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
    significance: float = DEFAULT_SIGNIFICANCE,
) -> list[KnockoutResult]:
    """Per-line knockout validation of a converged window fit.

    For each fitted line: remove it from the model, hold every other parameter
    frozen, and measure the chi-squared increase. A supported line grows the
    chi-squared by ~its own weighted energy and the increase is statistically
    significant (an F-test against the 3 parameters the line carries).

    Parameters
    ----------
    offset_grid_mhz : np.ndarray
        Baseband-offset grid of the window.
    complex_spectrum : np.ndarray
        Complex active-FT window data.
    rms_noise : float or np.ndarray
        Per-bin complex noise RMS.
    fit : WindowFitResult
        The converged fit to validate.
    acquisition_us : float
        Active acquisition length ``T`` (microseconds).
    significance : float, default 0.05
        F-test p-value threshold for ``supported``.

    Returns
    -------
    list of KnockoutResult
        One entry per fitted line, in fitted-peak order.
    """
    peaks = fit.peaks
    if not peaks:
        return []

    u = np.asarray(offset_grid_mhz, dtype=float)
    tau = fit.tau_us
    full_model = model_spectrum(u, peaks, tau, acquisition_us)
    full_chi2 = calculate_noise_weighted_chi2(complex_spectrum, rms_noise, full_model)

    results: list[KnockoutResult] = []
    for i, pk in enumerate(peaks):
        kept = [p for j, p in enumerate(peaks) if j != i]
        kept_model = model_spectrum(u, kept, tau, acquisition_us)
        chi2_without = calculate_noise_weighted_chi2(
            complex_spectrum, rms_noise, kept_model
        )
        delta = chi2_without - full_chi2
        # The line's own weighted energy -- the increase a real line should
        # produce when it is removed with every other parameter held frozen.
        line_model = model_spectrum(u, [pk], tau, acquisition_us)
        expected = calculate_noise_weighted_chi2(line_model, rms_noise)
        # Removing one line drops its 3 parameters (the shared tau stays).
        p_value, _, _ = calculate_chi_squared_improvement(
            chi2_without, full_chi2, 3, fit.n_data, fit.n_params
        )
        results.append(
            KnockoutResult(
                peak_index=i,
                offset_mhz=pk.offset_mhz,
                delta_chi2=delta,
                expected_delta_chi2=expected,
                supported=p_value < significance,
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
) -> tuple[WindowFitResult, list[AddStep]]:
    """Seed the window fit, escalating K=1 -> K=2 -> K=3 on an elevated chi².

    A single-cosine seed fit that leaves an elevated reduced chi-squared is
    treated as an unresolved blend (the prototype's key finding -- the failure
    is *initialisation*, not detectability). The fit is then retried with K
    lines initialised at positions straddling the seed, accepting each
    escalation only on the F-test and AIC. ``offset_grid_mhz`` must be
    ascending.
    """
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
        fit_tau=fit_tau,
        tau_bounds=tau_bounds,
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
            fit_tau=fit_tau,
            tau_bounds=tau_bounds,
        )
        p_value, f_stat, _ = calculate_chi_squared_improvement(
            prev.chi_squared,
            trial.chi_squared,
            trial.n_params - prev.n_params,
            trial.n_data,
            trial.n_params,
        )
        accepted = trial.success and p_value < significance and trial.aic < prev.aic
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
                separation_ok=True,
                decision="seed-blend" if accepted else "reject",
                reason=f"K={k} straddled re-seed",
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
) -> ConservativeFitResult:
    """Conservative incremental peak fitting of one window.

    Seeds with the strongest candidate (escalating to a straddled K=2/K=3 fit
    if the single-cosine seed leaves an elevated reduced chi-squared -- the
    blend-aware seeder), then repeatedly trial-fits the strongest remaining
    residual candidate, accepting it only when the chi-squared improvement
    passes a nested-model F-test *and* the AIC decreases. A candidate that
    violates the peak-separation constraint is dropped; a rejected candidate is
    held tentatively and the loop tolerates ``patience`` consecutive rejections
    (a tentative batch is promoted whole if it later becomes jointly
    significant). The converged fit is validated by :func:`knockout_test`.

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
        F-test p-value threshold for accepting a line.
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

    tau_bounds = (tau0_us / max_decay_factor, tau0_us * max_decay_factor)
    fwhm = feature_fwhm(tau0_us, acquisition_us)
    min_separation = min_separation_factor * fwhm

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
        fit_tau=fit_tau,
        tau_bounds=tau_bounds,
        fwhm_mhz=fwhm,
        null_chi2=null.chi_squared,
        null_aic=null.aic,
        significance=significance,
        rchi2_threshold=seeder_rchi2_threshold,
        straddle_factor=seeder_straddle_factor,
        max_k=seeder_max_k,
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
            fit_tau=fit_tau,
            tau_bounds=tau_bounds,
        )
        p_value, f_stat, _ = calculate_chi_squared_improvement(
            current.chi_squared,
            trial.chi_squared,
            trial.n_params - current.n_params,
            trial.n_data,
            trial.n_params,
        )
        passes = trial.success and p_value < significance and trial.aic < current.aic
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
                )
            )
            tentative.append(
                _seed_peak(cand, u, residual, current.tau_us, acquisition_us)
            )
            consecutive_rejects += 1
            if consecutive_rejects > patience:
                break

    knockouts = knockout_test(
        u, z, sigma, current, acquisition_us, significance=significance
    )
    return ConservativeFitResult(current, audit, knockouts)
