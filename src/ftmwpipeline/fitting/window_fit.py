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
The core works entirely in the de-ramped, demodulated *fit frame*: the window
grid is the signed baseband offset ``u`` and each line is parameterised by
``(amplitude, offset_mhz, phase)`` (:class:`~ftmwpipeline.fitting.peak_model.ModelPeak`)
plus a window-shared decay ``tau``. The de-ramp and the molecular<->offset
conversion are :func:`~ftmwpipeline.fitting.peak_model.to_baseband_frame`,
applied by the orchestration *before* calling :func:`fit_window`.

The residual and its weighting
------------------------------
The residual is in the complex-FT domain: model versus de-ramped window data,
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

from .peak_model import ModelPeak, h_T, h_T_jacobian, model_spectrum

__all__ = [
    "ParameterErrors",
    "WindowFitResult",
    "model_jacobian",
    "fit_window",
]

NoiseLike = Union[float, np.ndarray]

# Default solver evaluation cap, ported from the bcfitting reference shell.
DEFAULT_MAX_NFEV = 400
# Default tau bound factor k: tau in [tau_default / k, tau_default * k] (O5-4).
DEFAULT_MAX_DECAY_FACTOR = 5.0
# Phase bound: wide enough that wrapping never clips a free phase.
_PHASE_BOUND = 4.0 * np.pi


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
        if self.chi_squared <= 0.0 or self.n_data <= 0:
            return float("inf")
        penalty = self.n_data * np.log(self.chi_squared / self.n_data)
        return 2 * self.n_params + float(penalty)


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
    the shared ``tau`` -- against the de-ramped complex window data, using the
    analytic Jacobian. This is the fixed-K core; choosing K is task 4.

    Parameters
    ----------
    offset_grid_mhz : np.ndarray
        Baseband-offset grid ``u`` for the window (MHz), 1-D. The window data
        must already be in the de-ramped fit frame
        (:func:`~ftmwpipeline.fitting.peak_model.to_baseband_frame`).
    complex_spectrum : np.ndarray
        De-ramped complex window data on ``offset_grid_mhz``, same shape.
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
