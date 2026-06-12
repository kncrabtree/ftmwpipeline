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

import os as import_os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional, Union, cast

import numpy as np
from scipy.optimize import least_squares

from . import validation
from .peak_model import (
    ModelPeak,
    PeakShape,
    effective_tau_shape,
    h_T_shape,
    h_T_shape_jacobian,
    model_spectrum,
)
from .spur_detection import SpurMaskSpec
from .validation import (
    DEFAULT_N_EFF_KIND,
    calculate_aic,
    calculate_aicc,
    calculate_chi_squared_improvement,
    calculate_noise_weighted_chi2,
    effective_sample_size,
    feature_fwhm,
    gate_aicc_pair,
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
    "baseline_basis",
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
DEFAULT_MAX_PEAKS = 0  # 0 = no cap; the add-loop is bounded by the candidate set
DEFAULT_PATIENCE = 1
DEFAULT_MIN_SEPARATION_FACTOR = 1.0
# Blend-aware seeder: a single-cosine seed fit whose reduced chi-squared
# exceeds this is treated as an unresolved blend and re-seeded at K=2/K=3.
DEFAULT_SEEDER_RCHI2 = 1.5
# Straddle of the re-seeded inits, in units of the feature FWHM.
DEFAULT_SEEDER_STRADDLE_FACTOR = 1.0
DEFAULT_SEEDER_MAX_K = 3
# Soft phase-difference penalty: weak at sep = phase_penalty_cutoff_fwhm * fwhm,
# growing linearly to (lambda * cos(d_phase)^2) at sep = 0. Penalises both the
# in-phase degeneracy basin (two peaks at the same offset with aligned phases
# summing to a single feature's amplitude) and the anti-phase cancellation
# basin (cancelling phases producing inflated amplitudes). The penalty is zero
# only in quadrature (Δφ = π/2) -- the configuration where two close peaks
# carry independent information.
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
# Resolution-referenced floor on the minimum allowed pair separation, in units
# of the active-FT Fourier resolution element ``1/T_active`` (= ``1 /
# acquisition_us`` MHz). The effective minimum pair separation is
# ``max(min_pair_separation_factor * fwhm, this_factor / acquisition_us)``: two
# lines closer than one resolution element are fundamentally unresolvable, so a
# pair below this floor is a numerical artifact regardless of the per-window
# FWHM (which depends on the fitted decay tau and can fall below the resolution
# limit). Guards the sub-resolution duplicate-overfit pathology the FWHM-only
# floor licenses (GitHub issue #13).
DEFAULT_MIN_PAIR_SEPARATION_RESOLUTION_FACTOR = 1.0
# Blend-split trial: a candidate inside the pre-fit peak-separation floor of
# an existing peak still gets a trial fit when the residual at its position
# carries at least this many sigma of magnitude evidence (0 disables; the
# candidate is then rejected outright as before). An existing peak parked at
# a blend's compromise position leaves its residual maximum *inside* its own
# separation dead zone, so the outright rejection forecloses ever resolving
# the blend -- the trial NLS, free to move both the candidate and the
# blocking peak, splits it instead (655 w880: a 119-kHz doublet modeled as
# one line held a 41-sigma residual; the split drops window chi2 8503->221).
# The post-fit collapse check (with its blend_pair_escape) and the AICc gate
# arbitrate the outcome exactly as for an ordinary candidate; a failed
# blend-split trial is rejected outright (never held tentative, never counted
# against patience) so loop termination matches the legacy skip.
DEFAULT_BLEND_SPLIT_MIN_SNR = 4.0
# Tau policy: the canonical apodization (``expf_us``) sets a hard upper bound
# on ``tau`` -- the data cannot decay slower than the apodization itself.
# Decreasing ``tau`` below the apodization broadens the line, so the LSQ
# can buy chi^2 by under-fitting amplitude and over-broadening to absorb
# unmodeled-peak residual; the penalty discourages this with a stiff
# quadratic hinge. Weak-only windows (no candidate clears
# ``DEFAULT_WEAK_WINDOW_SNR_THRESHOLD``) hold ``tau`` fixed entirely.
DEFAULT_TAU_PENALTY_LAMBDA = 50.0
DEFAULT_WEAK_WINDOW_SNR_THRESHOLD = 10.0
# τ is freed only in windows clearing *both* the general weak-window floor
# (``weak_window_snr_threshold``) and this τ-specific bar; the effective floor is
# their max. Defaulting it to the weak-window floor keeps the gate where it
# historically sat (a single SNR-10 cutoff) until tuned upward.
DEFAULT_FIT_TAU_MIN_SNR = 10.0


def _effective_min_pair_separation(
    fwhm_mhz: float,
    acquisition_us: float,
    min_pair_separation_factor: float,
    min_pair_separation_resolution_factor: float,
) -> float:
    """Minimum allowed separation between two fitted lines, in MHz.

    The larger of the FWHM-referenced floor
    (``min_pair_separation_factor * fwhm_mhz``) and the active-FT resolution
    floor (``min_pair_separation_resolution_factor / acquisition_us``, where
    ``1/acquisition_us`` MHz is the Fourier resolution element ``1/T_active``).
    The resolution term is what catches sub-resolution duplicate pairs the
    FWHM-only floor licenses on narrow features (GitHub issue #13); a
    non-positive ``acquisition_us`` (or resolution factor) leaves only the
    FWHM term.
    """
    fwhm_floor = min_pair_separation_factor * fwhm_mhz
    if acquisition_us > 0.0 and min_pair_separation_resolution_factor > 0.0:
        resolution_floor = min_pair_separation_resolution_factor / acquisition_us
        return max(fwhm_floor, resolution_floor)
    return fwhm_floor


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
    tau_penalty_sigma_us: Optional[float]
    tau_penalty_sigma_lo_us: Optional[float]
    effective_tau_penalty_lambda: float
    fit_kwargs_inner: dict


# Tau-penalty bounds factor: half-width of the calibrated tau bounds in units
# of sigma_tau. Wide enough to keep the bidirectional Gaussian prior weakly
# binding inside the band while still rejecting the long-tau (clock-spur)
# basin and the over-broadened-tau basin.
DEFAULT_TAU_PENALTY_N_SIGMA = 5.0


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
    tau_penalty_n_sigma: float = DEFAULT_TAU_PENALTY_N_SIGMA,
    weak_window_snr_threshold: float = DEFAULT_WEAK_WINDOW_SNR_THRESHOLD,
    fit_tau_min_snr: float = DEFAULT_FIT_TAU_MIN_SNR,
    tau_apodization_us: Optional[float] = None,
    tau_maj_us: Optional[float] = None,
    sigma_tau_us: Optional[float] = None,
    tau_anchor_us: Optional[float] = None,
    tau_penalty_sigma_lo_factor: float = 1.0,
    shape: PeakShape | str = PeakShape.LORENTZIAN,
) -> WindowFitConstraints:
    """Derive a window's tau / amplitude / penalty constraints from its data.

    Tau policy: when a Stage 2b calibration is available, the
    ``(tau_maj_us, sigma_tau_us)`` pair drives both the bounds (a band of
    ``± tau_penalty_n_sigma * sigma_tau`` around ``tau_maj``, intersected
    with the ``max_decay_factor`` hard cap) and the bidirectional Gaussian
    prior in :func:`_penalty_residuals_and_jacobian`. When only the legacy
    ``tau_apodization_us`` is provided (calibration not yet run), the
    apodization is the hard upper bound and the (one-sided) penalty pulls
    tau toward it. Weak-only windows (in-window SNR below
    ``weak_window_snr_threshold``) hold tau fixed entirely.

    Parameters
    ----------
    tau_maj_us : float, optional
        Calibrated majority tau (from Stage 2b). When set with
        ``sigma_tau_us``, replaces ``tau_apodization_us`` as the tau
        anchor and switches the penalty to the bidirectional form.
    sigma_tau_us : float, optional
        Calibrated robust spread of the tau distribution (Stage 2b). The
        bidirectional penalty's width parameter; must be positive when
        ``tau_maj_us`` is supplied.
    tau_penalty_n_sigma : float, default :data:`DEFAULT_TAU_PENALTY_N_SIGMA`
        Half-width of the calibrated tau bounds in units of ``sigma_tau``.
    tau_anchor_us : float, optional
        Explicit long (narrow-line) anchor for the bidirectional prior; when
        ``None`` the anchor is the calibrated majority ``tau_maj_us``. Setting
        it above the true tau implements a "start narrow, broaden cheaply"
        policy (see ``tau_penalty_sigma_lo_factor``).
    tau_penalty_sigma_lo_factor : float, default 1.0
        Multiplier on ``sigma_tau`` for the *below-anchor* side of the penalty.
        ``> 1`` makes broadening (decreasing tau) cheap while the stiff
        ``sigma_tau`` above the anchor blocks tau-runaway; ``1.0`` reproduces
        the symmetric bidirectional prior.

        DORMANT: this asymmetric long-anchor penalty is a prototype with **no
        production caller** -- nothing in the settings resolver,
        ``fit_peaks_impl``, ``plan_execution``, or ``residual_rescue`` sets
        ``tau_anchor_us`` or a non-unity ``tau_penalty_sigma_lo_factor``, so
        every shipped fit runs the symmetric prior. It was prototyped to keep
        partially-resolved hyperfine that an over-broad start would swallow
        resolvable, but cross-fixture validation found it is **not a per-window
        chi2r lever on the dense bulk** (the bulk tau is already at its
        data-preferred per-band value and does not relax under the long
        anchor); its apparent wins were confounded by the dense-spectrum
        mega-windows the Stage 4 window peak-count cap eliminates. The knobs
        and tests are kept for a possible future genuine-blend use case; absent
        one, the settings/orchestrator/dual-interface wiring is omitted.
        See ``dev-docs/research/stage5-cross-fixture/report.md`` (Phase 2).
    """
    shape_resolved = PeakShape.coerce(shape)
    z = np.asarray(complex_spectrum, dtype=np.complex128)
    sigma = np.asarray(rms_noise, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(z.size, float(sigma))

    fwhm = feature_fwhm(tau0_us, acquisition_us, shape=shape_resolved)
    min_separation = min_separation_factor * fwhm

    use_calibrated = (
        tau_maj_us is not None
        and tau_maj_us > 0.0
        and sigma_tau_us is not None
        and sigma_tau_us > 0.0
    )

    if use_calibrated:
        assert tau_maj_us is not None  # guaranteed by use_calibrated check above
        assert sigma_tau_us is not None  # guaranteed by use_calibrated check above
        tm = float(tau_maj_us)
        st = float(sigma_tau_us)
        n_sig = float(tau_penalty_n_sigma)
        # Anchor the prior at the explicit long (narrow-line) anchor when given,
        # else at the calibrated majority. The long anchor implements the "start
        # narrow" policy: it sits above the true tau so the soft low side can
        # broaden down onto it while the stiff high side blocks tau-runaway.
        anchor = (
            float(tau_anchor_us)
            if tau_anchor_us is not None and tau_anchor_us > 0.0
            else tm
        )
        tau_lo = max(tm - n_sig * st, tm / max_decay_factor)
        # The high bound must reach the anchor (the seed starts there); the
        # stiff penalty -- not the bound -- discourages narrowing past it.
        tau_hi = min(max(tm + n_sig * st, anchor), tm * max_decay_factor)
        # Ensure the band is non-degenerate (sigma_tau larger than tm
        # would otherwise push tau_lo below the factor-k floor).
        if not tau_lo < tau_hi:
            tau_lo, tau_hi = tm / max_decay_factor, max(tm * max_decay_factor, anchor)
        tau_bounds = (tau_lo, tau_hi)
        tau_penalty_ref: Optional[float] = anchor
        tau_penalty_sigma_us: Optional[float] = st
        # Soft broadening side: a wider effective sigma below the anchor makes
        # decreasing tau (broadening) cheap; 1.0 reproduces the symmetric prior.
        tau_penalty_sigma_lo_us: Optional[float] = (
            st * float(tau_penalty_sigma_lo_factor)
            if tau_penalty_sigma_lo_factor and tau_penalty_sigma_lo_factor > 0.0
            else st
        )
    else:
        tau_upper = tau0_us * max_decay_factor
        if tau_apodization_us is not None and tau_apodization_us > 0.0:
            tau_upper = min(tau_upper, float(tau_apodization_us))
        tau_bounds = (tau0_us / max_decay_factor, tau_upper)
        tau_penalty_ref = (
            float(tau_apodization_us)
            if tau_apodization_us is not None and tau_apodization_us > 0.0
            else None
        )
        tau_penalty_sigma_us = None
        tau_penalty_sigma_lo_us = None

    tau_eff_min = effective_tau_shape(shape_resolved, tau_bounds[0], acquisition_us)
    tau_eff_nom = effective_tau_shape(shape_resolved, tau0_us, acquisition_us)
    max_abs = float(np.max(np.abs(z))) if z.size else 0.0
    if max_abs > 0.0 and tau_eff_min > 0.0:
        amp_max: Optional[float] = float(amp_max_headroom * 2.0 * max_abs / tau_eff_min)
    else:
        amp_max = None
    sig_median = float(np.median(sigma)) if sigma.size else 0.0
    if sig_median > 0.0 and tau_eff_nom > 0.0:
        amp_floor: Optional[float] = float(2.0 * sig_median / tau_eff_nom)
    else:
        amp_floor = None

    fit_tau_eff = fit_tau
    snr_proxy = (max_abs / sig_median) if sig_median > 0.0 else 0.0
    # τ is freed only above *both* the general weak-window floor and the
    # τ-specific bar, so the effective free-τ floor is their max. With both at
    # their default of 10 this is the historical single SNR-10 cutoff; raising
    # ``fit_tau_min_snr`` tightens τ freedom without touching the weak-window
    # floor that governs the rest of the conservative treatment.
    tau_free_floor = max(weak_window_snr_threshold, fit_tau_min_snr)
    if fit_tau_eff and snr_proxy < tau_free_floor:
        fit_tau_eff = False

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
        tau_penalty_sigma_us=tau_penalty_sigma_us,
        tau_penalty_sigma_lo_us=tau_penalty_sigma_lo_us,
    )

    return WindowFitConstraints(
        tau_bounds=tau_bounds,
        fwhm=fwhm,
        min_separation=min_separation,
        amp_max=amp_max,
        amp_floor=amp_floor,
        fit_tau_eff=fit_tau_eff,
        tau_penalty_reference=tau_penalty_ref,
        tau_penalty_sigma_us=tau_penalty_sigma_us,
        tau_penalty_sigma_lo_us=tau_penalty_sigma_lo_us,
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
    # Provenance flag: did this window's pipeline ever determine tau via LSQ?
    # ``fit_tau`` is the mechanical "was tau a free parameter in *this*
    # least-squares call" -- True on the originating conservative_fit when
    # snr_proxy clears the weak-window gate, False on cleanup refits that lock
    # tau by design. ``tau_was_fit`` propagates the originating answer
    # through the cleanup/rescue chain so downstream consumers can tell
    # frozen-by-gate (tau held at tau0_us) from data-driven-but-cleaned-up
    # (tau came from the joint refit / original conservative_fit).
    # ``None`` in the constructor defers to ``fit_tau``; cleanup paths
    # explicitly override after refit so the cleaned WindowFitResult
    # advertises the original determination.
    tau_was_fit: Optional[bool] = field(default=None)
    # Line-shape selector used for this window's fit. ``tau_us`` is the
    # exponential τ under LORENTZIAN, the Gaussian τ_G under GAUSSIAN;
    # the model evaluation, Jacobian, and effective_tau all route through
    # the matching closed form. Defaults to LORENTZIAN so existing call
    # sites and persisted-file readers behave unchanged.
    shape: PeakShape = field(default=PeakShape.LORENTZIAN)
    # Optional low-order complex baseline ``B(u) = Σ_{k≤p} (a_k + i b_k)
    # (u/u_s)^k`` jointly fit with the peaks to absorb a neighbour's
    # mismodeled leakage wing (the leakage-wing baseline nuisance term).
    # ``baseline_order`` is ``p`` (0 = const, 1 = linear); ``None`` means
    # no baseline was fit. ``baseline_coeffs`` is the length-``(p+1)``
    # complex coefficient vector ``a_k + i b_k`` and ``baseline_offset_scale``
    # is ``u_s`` (the conditioning scale ``max|u|``). The reported peak
    # uncertainties already include the baseline's degrees of freedom: the
    # covariance is the joint (peaks + tau + baseline) inverse ``J^T J``.
    baseline_order: Optional[int] = field(default=None)
    baseline_coeffs: Optional[np.ndarray] = field(default=None)
    baseline_offset_scale: Optional[float] = field(default=None)

    def __post_init__(self) -> None:
        if self.tau_was_fit is None:
            object.__setattr__(self, "tau_was_fit", self.fit_tau)

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
    shape: PeakShape | str = PeakShape.LORENTZIAN,
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
        Shared decay constant (microseconds, ``> 0``). For
        ``shape=LORENTZIAN`` this is ``τ``; for ``shape=GAUSSIAN`` it is
        ``τ_G``.
    acquisition_us : float
        Active acquisition length ``T`` (microseconds, ``> 0``).
    include_tau : bool, default False
        Append the shared ``d/dtau`` column.
    shape : PeakShape or str, default LORENTZIAN
        Line-shape selector. Routes the per-peak ``h_T`` and
        ``h_T_jacobian`` evaluations through the matching Lorentzian or
        Gaussian closed forms.

    Returns
    -------
    np.ndarray
        Complex array of shape ``(M, 3K)`` -- or ``(M, 3K + 1)`` with
        ``include_tau`` -- where ``M = len(offset_grid_mhz)``.
    """
    s = PeakShape.coerce(shape)
    u = np.asarray(offset_grid_mhz, dtype=float)
    k = len(peaks)
    n_params = 3 * k + (1 if include_tau else 0)
    jac = np.zeros((u.size, n_params), dtype=np.complex128)
    if k == 0:
        return cast(np.ndarray, jac)

    # Evaluate every line's shape and its derivatives in one broadcast over the
    # (K, M) offset grid (one ``h_T_shape`` / ``h_T_shape_jacobian`` call,
    # shape coerced once) instead of 2K per-peak calls. Byte-identical to the
    # per-peak loop; only the line-shape calls are batched.
    offsets = np.fromiter((pk.offset_mhz for pk in peaks), dtype=float, count=k)
    amps = np.fromiter((pk.amplitude for pk in peaks), dtype=float, count=k)
    phases = np.fromiter((pk.phase for pk in peaks), dtype=float, count=k)
    phasors = np.exp(1j * phases)  # (K,)
    du = u[np.newaxis, :] - offsets[:, np.newaxis]  # (K, M)
    line = h_T_shape(s, du, tau_us, acquisition_us)  # (K, M)
    d_shape_df, d_shape_dtau = h_T_shape_jacobian(s, du, tau_us, acquisition_us)
    ph = phasors[:, np.newaxis]  # (K, 1)
    amp = amps[:, np.newaxis]
    # Columns, peak-major: [A_0, δ_0, φ_0, A_1, δ_1, φ_1, ...].
    jac[:, 0 : 3 * k : 3] = (0.5 * ph * line).T
    jac[:, 1 : 3 * k : 3] = (-0.5 * amp * ph * d_shape_df).T
    jac[:, 2 : 3 * k : 3] = (0.5j * amp * ph * line).T
    if include_tau:
        jac[:, 3 * k] = np.sum(0.5 * amp * ph * d_shape_dtau, axis=0)
    return cast(np.ndarray, jac)


# ---------------------------------------------------------------------------
# Optional low-order complex baseline (leakage-wing nuisance term)
# ---------------------------------------------------------------------------
def baseline_basis(
    offset_grid_mhz: np.ndarray, order: int, offset_scale: float
) -> np.ndarray:
    """Real polynomial basis ``[(u/u_s)^0, ..., (u/u_s)^order]`` for the baseline.

    The complex baseline ``B(u) = Σ_{k=0..p} (a_k + i b_k)(u/u_s)^k`` shares
    one real basis column ``(u/u_s)^k`` between its real coefficient ``a_k``
    and imaginary coefficient ``b_k``. ``offset_scale`` (``u_s``, the window's
    ``max|u|``) normalises the abscissa so the design matrix stays well
    conditioned across windows of different widths.

    Parameters
    ----------
    offset_grid_mhz : np.ndarray
        Baseband-offset grid ``u`` (MHz), 1-D.
    order : int
        Baseline polynomial order ``p`` (``0`` = constant, ``1`` = linear).
    offset_scale : float
        Conditioning scale ``u_s`` (``> 0``).

    Returns
    -------
    np.ndarray
        Real array of shape ``(M, order + 1)``; column ``k`` is ``(u/u_s)^k``.
    """
    if order < 0:
        raise ValueError("baseline order must be non-negative")
    if not offset_scale > 0.0:
        raise ValueError("offset_scale must be positive")
    u = np.asarray(offset_grid_mhz, dtype=float)
    x = u / float(offset_scale)
    return cast(np.ndarray, np.vander(x, order + 1, increasing=True))


def evaluate_baseline(
    fit: "WindowFitResult", offset_grid_mhz: np.ndarray
) -> np.ndarray:
    """A fit's complex-baseline contribution on a grid (zeros when none).

    The baseline is part of the fit's *model* (``fitted_spectrum`` includes
    it), but consumers that re-evaluate the model from ``fit.peaks`` via
    :func:`~ftmwpipeline.fitting.peak_model.model_spectrum` -- the rescue's
    residual detector, the knockout sweep's K-fit chi-squared -- would
    otherwise drop it and see the carried pedestal as residual.
    """
    u = np.asarray(offset_grid_mhz, dtype=float)
    if (
        fit.baseline_order is None
        or fit.baseline_coeffs is None
        or not fit.baseline_offset_scale
    ):
        return cast(np.ndarray, np.zeros(u.size, dtype=np.complex128))
    basis = baseline_basis(u, int(fit.baseline_order), float(fit.baseline_offset_scale))
    return cast(
        np.ndarray, basis @ np.asarray(fit.baseline_coeffs, dtype=np.complex128)
    )


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
    tau_penalty_sigma_us: Optional[float] = None,
    tau_penalty_sigma_lo_us: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Penalty residuals + analytic Jacobian rows for the augmented LSQ.

    Three soft penalties, all built so ``sum(r**2)`` matches the cost the
    scipy solver minimises in addition to the data residual:

    * **Pair phase penalty** -- one residual element per unordered pair
      ``(i, j)``. The penalty is
      ``sqrt(lambda) * weight * cos(phi_i - phi_j)`` with
      ``weight = max(0, 1 - sep / cutoff)``: zero at or beyond the cutoff
      separation, ``±sqrt(lambda)`` at zero separation. The cosine fires at
      both the in-phase degeneracy basin (Δφ = 0 → cos = +1) and the
      anti-phase cancellation basin (Δφ = π → cos = -1); it is zero only at
      quadrature (Δφ = π/2), the configuration where two close peaks carry
      independent information. The slot is *always emitted* (with value zero
      when ``weight = 0``) so the residual vector has constant length across
      solver iterations.
    * **Amplitude floor penalty** -- one residual element per peak,
      ``sqrt(lambda) * max(0, 1 - A_i / amp_floor)``. A linear hinge that
      pushes spurious noise-amplitude peaks toward zero. Always emitted
      (zero above the floor).
    * **Tau anchoring penalty** -- one residual element when ``fit_tau``
      and ``tau_penalty_lambda > 0``. Two forms:

      - **(A)symmetric Gaussian prior** (``tau_penalty_sigma_us`` set, > 0):
        ``sqrt(lambda) * (tau - tau_ref) / sigma_eff``. Pulls tau toward
        ``tau_ref`` at strength ``lambda / sigma_eff^2``. ``sigma_eff`` is
        ``tau_penalty_sigma_us`` for ``tau >= tau_ref`` (the stiff *narrowing*
        side) and ``tau_penalty_sigma_lo_us`` for ``tau < tau_ref`` (the soft
        *broadening* side); when the latter is None both sides share
        ``tau_penalty_sigma_us`` and the prior is the symmetric bidirectional
        form. The asymmetry implements the "start narrow, broaden cheaply,
        narrow expensively" policy: with ``tau_ref`` seeded at a deliberately
        long (narrow-line) anchor, a soft low side lets the fit broaden down
        onto the true tau (and keeps partially-blended components resolvable
        rather than swallowed by an over-broad start), while the stiff high
        side blocks tau-runaway toward the CW-spur basin / clean-feature
        overfit. This is the form used when a Stage 2b calibration is
        available.
      - **One-sided hinge** (``tau_penalty_sigma_us`` is None): legacy form
        for the no-calibration path -- ``sqrt(lambda) * max(0, (tau_ref -
        tau) / tau_ref)``. The applied apodization (``expf_us``) sets a hard
        upper bound on tau and the penalty pulls tau back up toward it.
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
    res: np.ndarray = np.zeros(n_pen, dtype=float)
    jac: np.ndarray = np.zeros((n_pen, n_params), dtype=float)
    peaks, _tau = _unpack(params, k, tau0_us, fit_tau)

    row = 0
    if phase_penalty_lambda > 0.0 and k > 1:
        cutoff = phase_penalty_cutoff_fwhm * fwhm_mhz if fwhm_mhz is not None else 0.0
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
                        d_phase = peaks[i].phase - peaks[j].phase
                        cos_d = float(np.cos(d_phase))
                        sin_d = float(np.sin(d_phase))
                        res[row] = sqrt_lambda * weight * cos_d
                        sgn = (sep / abs_sep) if abs_sep > 0.0 else 0.0
                        dweight_doffi = -sgn / cutoff
                        dweight_doffj = sgn / cutoff
                        jac[row, 3 * i + 1] = sqrt_lambda * dweight_doffi * cos_d
                        jac[row, 3 * j + 1] = sqrt_lambda * dweight_doffj * cos_d
                        # d/d(phi_i) cos(phi_i - phi_j) = -sin(phi_i - phi_j);
                        # d/d(phi_j) is +sin(phi_i - phi_j).
                        jac[row, 3 * i + 2] = -sqrt_lambda * weight * sin_d
                        jac[row, 3 * j + 2] = sqrt_lambda * weight * sin_d
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
        sqrt_lambda = float(np.sqrt(tau_penalty_lambda))
        if tau_penalty_sigma_us is not None and tau_penalty_sigma_us > 0.0:
            # (A)symmetric Gaussian prior: stiff above tau_ref (narrowing),
            # soft below (broadening) when a distinct low-side sigma is given.
            below = tau_value < tau_penalty_reference
            sigma_eff = (
                float(tau_penalty_sigma_lo_us)
                if below
                and tau_penalty_sigma_lo_us is not None
                and tau_penalty_sigma_lo_us > 0.0
                else float(tau_penalty_sigma_us)
            )
            res[row] = sqrt_lambda * (tau_value - tau_penalty_reference) / sigma_eff
            jac[row, 3 * k] = sqrt_lambda / sigma_eff
        else:
            # Legacy one-sided hinge.
            ratio = (tau_penalty_reference - tau_value) / tau_penalty_reference
            if ratio > 0.0:
                res[row] = sqrt_lambda * ratio
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
    tau_penalty_sigma_us: Optional[float] = None,
    tau_penalty_sigma_lo_us: Optional[float] = None,
    shape: PeakShape | str = PeakShape.LORENTZIAN,
    spur_mask: Optional[SpurMaskSpec] = None,
    baseline_order: Optional[int] = None,
    baseline_offset_scale: Optional[float] = None,
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
        Reference tau (the calibrated ``tau_maj`` or, in the legacy path,
        the apodization ``expf_us``) the tau penalty centres on. Required
        when ``tau_penalty_lambda > 0``.
    tau_penalty_sigma_us : float, optional
        When set with ``tau_penalty_reference``, switches the tau penalty
        to the bidirectional Gaussian-prior form (centred on the
        reference, width ``sigma_tau``). ``None`` keeps the legacy
        one-sided hinge form.
    spur_mask : SpurMaskSpec, optional
        Clock/LO-spur bins to exclude from the weighted residual, Jacobian,
        and chi-squared. The model is still evaluated on the full grid (so
        ``fitted_spectrum`` / ``residual`` stay full-length for downstream
        plotting and edge-coherence), but masked bins do not contribute to
        the cost and ``n_data`` is reduced by twice the masked-bin count so
        reduced chi-squared stays calibrated. A spur is a single-bin CW
        delta no finite-T line shape can represent; excluding it stops it
        detonating the window's chi-squared. ``None`` (default) masks
        nothing.
    baseline_order : int, optional
        Order ``p`` of an optional complex baseline ``B(u) = Σ_{k=0..p}
        (a_k + i b_k)(u/u_s)^k`` fit jointly with the lines (``0`` = const,
        ``1`` = linear). ``None`` (default) fits no baseline. The baseline
        absorbs a neighbour's mismodeled leakage wing without representing a
        narrow line (it is too smooth to do so); its ``2(p+1)`` real
        coefficients enter the covariance, so the reported per-line
        uncertainties honestly price the added flexibility. The fitted
        coefficients are returned on :attr:`WindowFitResult.baseline_coeffs`.
    baseline_offset_scale : float, optional
        Conditioning scale ``u_s`` for the baseline abscissa; defaults to
        ``max|u|`` (or ``1.0`` for an all-zero grid). Only used when
        ``baseline_order`` is set.

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

    # Spur-bin exclusion: ``keep`` is True for bins that contribute to the
    # cost. A spur is a single-bin CW delta no line shape can represent;
    # dropping it from the residual / Jacobian / chi^2 stops it detonating
    # the window's chi^2, and reducing n_data by 2*masked keeps chi^2_r
    # calibrated. The model is still evaluated everywhere so the returned
    # fitted/residual arrays stay full-length.
    if spur_mask is not None:
        keep = ~spur_mask.bin_mask(u)
        if not keep.any():
            keep = np.ones(m, dtype=bool)
    else:
        keep = np.ones(m, dtype=bool)
    n_keep = int(keep.sum())

    k = len(initial_peaks)
    n_data = 2 * n_keep

    # --- null model: nothing to fit, but report the data's chi-squared. -----
    if k == 0:
        r0 = (z / sig_ri)[keep]
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
            # The null model has no lines, but it can stand as a window's
            # *final* fit (seed-knockout enforcement); the persisted
            # per-window shape attribute must still record the run's shape.
            shape=PeakShape.coerce(shape),
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

    # --- optional complex baseline -----------------------------------------
    # The baseline contributes ``2 * (baseline_order + 1)`` real parameters
    # (a_k, b_k per order), packed after the peak / tau parameters so the
    # peak / tau covariance slices stay at their existing offsets. Its design
    # columns are shared by a_k and b_k: ``d(model)/d(a_k) = (u/u_s)^k`` and
    # ``d(model)/d(b_k) = i (u/u_s)^k``.
    baseline_active = baseline_order is not None and baseline_order >= 0
    order_resolved = baseline_order if baseline_order is not None else 0
    n_base = (order_resolved + 1) if baseline_active else 0
    if baseline_active:
        u_s = (
            float(baseline_offset_scale)
            if baseline_offset_scale is not None and baseline_offset_scale > 0.0
            else float(np.max(np.abs(u)))
        )
        if not u_s > 0.0:
            u_s = 1.0
        basis = baseline_basis(u, order_resolved, u_s)  # (M, n_base) real
        # Both real- and imag-coefficient columns of the complex model.
        base_cols = np.concatenate([basis, 1j * basis], axis=1)  # (M, 2 n_base)
    else:
        u_s = None
        basis = None
        base_cols = None
    base_start = 3 * k + (1 if fit_tau else 0)

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
    if baseline_active:
        # Baseline coefficients are unbounded and seed at zero.
        lo_arr = np.concatenate([lo_arr, np.full(2 * n_base, -np.inf)])
        hi_arr = np.concatenate([hi_arr, np.full(2 * n_base, np.inf)])
        p0 = np.concatenate([p0, np.zeros(2 * n_base, dtype=float)])

    penalty_kw: dict[str, Any] = dict(
        phase_penalty_lambda=phase_penalty_lambda,
        amp_penalty_lambda=amp_penalty_lambda,
        amp_floor=amp_floor,
        fwhm_mhz=fwhm_mhz,
        phase_penalty_cutoff_fwhm=phase_penalty_cutoff_fwhm,
        tau_penalty_lambda=tau_penalty_lambda,
        tau_penalty_reference=tau_penalty_reference,
        tau_penalty_sigma_us=tau_penalty_sigma_us,
        tau_penalty_sigma_lo_us=tau_penalty_sigma_lo_us,
    )
    penalties_active = (
        phase_penalty_lambda > 0.0
        or amp_penalty_lambda > 0.0
        or (tau_penalty_lambda > 0.0 and fit_tau)
    )

    shape_resolved = PeakShape.coerce(shape)

    def _model_with_baseline(
        peaks: Sequence[ModelPeak], tau: float, params: np.ndarray
    ) -> np.ndarray:
        model = model_spectrum(u, peaks, tau, acquisition_us, shape=shape_resolved)
        if baseline_active:
            model = model + base_cols @ params[base_start:]
        return model

    def residual(params: np.ndarray) -> np.ndarray:
        peaks, tau = _unpack(params, k, tau0_us, fit_tau)
        model = _model_with_baseline(peaks, tau, params)
        r = (z - model) / sig_ri
        data_r = np.concatenate([r.real[keep], r.imag[keep]])
        if not penalties_active:
            return cast(np.ndarray, data_r)
        pen_r, _ = _penalty_residuals_and_jacobian(
            params, k, tau0_us, fit_tau, **penalty_kw
        )
        return cast(np.ndarray, np.concatenate([data_r, pen_r]))

    def _weighted_model_jacobian(peaks: Sequence[ModelPeak], tau: float) -> np.ndarray:
        # d(residual)/d(p) = -(d(model)/d(p)) / sig_ri, complex, columns =
        # peaks (+ tau), then the baseline a_k / b_k columns.
        dmodel = model_jacobian(
            u,
            peaks,
            tau,
            acquisition_us,
            include_tau=fit_tau,
            shape=shape_resolved,
        )
        if baseline_active:
            dmodel = np.concatenate([dmodel, base_cols], axis=1)
        return cast(np.ndarray, -dmodel / sig_ri[:, np.newaxis])

    def jacobian(params: np.ndarray) -> np.ndarray:
        peaks, tau = _unpack(params, k, tau0_us, fit_tau)
        weighted = _weighted_model_jacobian(peaks, tau)
        data_jac = np.concatenate([weighted.real[keep], weighted.imag[keep]], axis=0)
        if not penalties_active:
            return cast(np.ndarray, data_jac)
        _, pen_jac = _penalty_residuals_and_jacobian(
            params, k, tau0_us, fit_tau, **penalty_kw
        )
        # Penalty rows do not touch the baseline parameters: pad with zeros.
        if baseline_active and pen_jac.shape[0] > 0:
            pen_jac = np.concatenate(
                [pen_jac, np.zeros((pen_jac.shape[0], 2 * n_base))], axis=1
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
            # Scale the trust region by the Jacobian column norms each
            # iteration. The fit parameters are wildly different in magnitude
            # -- amplitude ~1e3, baseband offset ~1e-2 MHz, phase ~1, shared
            # tau ~5 us -- so the default uniform scaling conditions the step
            # poorly and wastes iterations crawling along the cramped
            # directions. ``'jac'`` cuts solver evaluations by ~1/3 on the
            # dense high-K windows (and a few percent everywhere) for an
            # identical converged minimum; the fitted line set is unchanged
            # within solver tolerance across the cross-fixture metric.
            x_scale="jac",
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
            shape=shape_resolved,
        )

    sol_x = np.asarray(sol.x, dtype=float)
    peaks, tau = _unpack(sol_x, k, tau0_us, fit_tau)
    for pk in peaks:
        pk.phase = _wrap_phase(pk.phase)
    baseline_coeffs: Optional[np.ndarray] = None
    if baseline_active:
        a = sol_x[base_start : base_start + n_base]
        b = sol_x[base_start + n_base : base_start + 2 * n_base]
        baseline_coeffs = a + 1j * b

    fitted = _model_with_baseline(peaks, tau, sol_x)
    # Reported statistics are data-only (penalties act like a prior on the
    # parameters; the F-test / AIC across K stays calibrated only if chi^2
    # counts the data residual alone). Spur-masked bins are excluded from the
    # chi^2 sum to match the reduced n_data.
    data_resid = ((z - fitted) / sig_ri)[keep]
    chi2 = float(np.sum(data_resid.real**2 + data_resid.imag**2))
    cost_data = 0.5 * chi2

    covariance: Optional[np.ndarray] = None
    try:
        # Data-only Jacobian for uncertainty: penalties bias parameter errors
        # smaller (they're effectively a prior). Caller wants the data
        # likelihood's covariance. When a baseline is fit, its columns are
        # part of the Jacobian, so the inverse is the *joint* covariance and
        # the per-line errors honestly include the baseline's flexibility.
        weighted_sol = _weighted_model_jacobian(peaks, tau)
        data_jac = np.concatenate(
            [weighted_sol.real[keep], weighted_sol.imag[keep]], axis=0
        )
        jtj = data_jac.T @ data_jac
        covariance = cast(np.ndarray, np.linalg.inv(jtj))
    except np.linalg.LinAlgError:
        covariance = None
    # _parameter_errors slices the peak (front) and tau (3*k) diagonal
    # entries; baseline coefficients sit after them, so the slicing is
    # unchanged while the covariance already reflects the joint fit.
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
        shape=shape_resolved,
        baseline_order=order_resolved if baseline_active else None,
        baseline_coeffs=baseline_coeffs,
        baseline_offset_scale=u_s if baseline_active else None,
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
        ``"tentative"``, ``"reject"`` or ``"knockout-null"`` (the lone
        seed removed at exit by its own K=1-vs-null knockout verdict, see
        :data:`validation.DEFAULT_ENFORCE_SEED_KNOCKOUT`).
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
    weighted_gate_chi2: Optional[bool] = None,
    significance: float = DEFAULT_SIGNIFICANCE,
    spur_mask: Optional[SpurMaskSpec] = None,
    gate_budget_extra: Optional[np.ndarray] = None,
    refit_sink: Optional[dict[int, WindowFitResult]] = None,
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
    refit_sink : dict, optional
        When provided, every (K-1) ``fit_window`` refit computed for peak
        index ``i`` is stored as ``refit_sink[i] = refit`` -- including
        failed refits (``refit.success == False``). The K=1 -> K=0 null
        branch performs no refit and stores nothing. The caller may pass
        this dict to :func:`iterative_aicc_cleanup` as ``initial_refits``
        to avoid recomputing the same fits in the cleanup's first iteration.

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

    # Exclude spur bins from every chi^2 in the sweep so the AICc gate sees
    # the same masked data the K-fit's chi_squared was computed on.
    if spur_mask is not None:
        keep = ~spur_mask.bin_mask(u)
        if not keep.any():
            keep = np.ones(u.size, dtype=bool)
    else:
        keep = np.ones(u.size, dtype=bool)
    z_keep = z[keep]
    sigma_keep = sigma[keep]
    budget_keep: Optional[np.ndarray] = None
    if gate_budget_extra is not None:
        budget_keep = np.asarray(gate_budget_extra, dtype=float)[keep]

    tau = fit.tau_us
    shape_resolved = fit.shape
    # The K-fit model must carry the fit's baseline term (when one was fit
    # jointly): the (K-1) refits below inherit it via ``fit_kwargs_inner``,
    # so a peaks-only K side would unfairly carry the pedestal as misfit.
    full_model = model_spectrum(
        u, peaks, tau, acquisition_us, shape=shape_resolved
    ) + evaluate_baseline(fit, u)
    full_chi2 = calculate_noise_weighted_chi2(z_keep, sigma_keep, full_model[keep])

    # n_eff is keyed on the K-fit's model magnitude -- the same value the
    # merge cleanup uses for its AICc test -- and shared across all peaks
    # in the sweep so per-peak comparisons sit on a common scale.
    n_eff = effective_sample_size(
        fit.fitted_spectrum,
        kind=n_eff_kind,
        sigma=sigma,
    )
    # Context-invariant gate: score the K vs K-1 AICc on the information-
    # weighted chi-squared (weights from the more-complex K-fit model) so a
    # peak's support depends on its local evidence, not the window's bin count.
    weighted = (
        validation.DEFAULT_WEIGHTED_GATE_CHI2
        if weighted_gate_chi2 is None
        else weighted_gate_chi2
    )
    fit_residual_keep = np.asarray(fit.residual)[keep]
    weight_model_keep = np.asarray(fit.fitted_spectrum)[keep]

    refit_kwargs: dict[str, Any] = dict(fit_kwargs_inner or {})
    refit_kwargs["fit_tau"] = False  # tau locked at the K-fit value
    refit_kwargs.setdefault("shape", shape_resolved)

    results: list[KnockoutResult] = []
    for i, pk in enumerate(peaks):
        kept = [p for j, p in enumerate(peaks) if j != i]
        # Diagnostic: freeze-others delta_chi2 + expected line energy.
        kept_model = model_spectrum(u, kept, tau, acquisition_us, shape=shape_resolved)
        chi2_without = calculate_noise_weighted_chi2(
            z_keep, sigma_keep, kept_model[keep]
        )
        delta = chi2_without - full_chi2
        line_model = model_spectrum(u, [pk], tau, acquisition_us, shape=shape_resolved)
        expected = calculate_noise_weighted_chi2(line_model[keep], sigma_keep)

        if not kept:
            # K=1 -> K=0 refit is the null model; no fit_window call needed.
            # The null chi-squared is the data's own noise-weighted energy.
            null_chi2 = calculate_noise_weighted_chi2(z_keep, sigma_keep)
            aicc_k, aicc_km1 = gate_aicc_pair(
                n_eff,
                more_n_params=fit.n_params,
                less_n_params=0,
                more_chi2_raw=fit.chi_squared,
                less_chi2_raw=null_chi2,
                weighted=weighted,
                more_residual=fit_residual_keep,
                less_residual=z_keep,
                rms_noise=sigma_keep,
                weight_model=weight_model_keep,
                n_eff_kind=n_eff_kind,
                ref_reduced_chi2=fit.reduced_chi2,
                budget_extra=budget_keep,
            )
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
            u,
            z,
            sigma,
            kept,
            tau,
            acquisition_us,
            spur_mask=spur_mask,
            **refit_kwargs,
        )
        if refit_sink is not None:
            refit_sink[i] = refit
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

        aicc_k, aicc_km1 = gate_aicc_pair(
            n_eff,
            more_n_params=fit.n_params,
            less_n_params=refit.n_params,
            more_chi2_raw=fit.chi_squared,
            less_chi2_raw=refit.chi_squared,
            weighted=weighted,
            more_residual=fit_residual_keep,
            less_residual=np.asarray(refit.residual)[keep],
            rms_noise=sigma_keep,
            weight_model=weight_model_keep,
            n_eff_kind=n_eff_kind,
            budget_extra=budget_keep,
        )
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
    *,
    shape: PeakShape | str = PeakShape.LORENTZIAN,
) -> ModelPeak:
    """Initial-guess line at ``offset_mhz`` from the local residual.

    Amplitude from the residual magnitude divided by the on-line gain
    ``tau_eff``, phase from the residual phase. ``offset_grid_mhz`` must be
    ascending (``np.interp`` requirement).
    """
    mag = float(
        np.abs(np.interp(offset_mhz, offset_grid_mhz, np.abs(residual_spectrum)))
    )
    gain = max(effective_tau_shape(shape, tau0_us, acquisition_us), 1e-9)
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
    min_pair_separation_resolution_factor: float = (
        DEFAULT_MIN_PAIR_SEPARATION_RESOLUTION_FACTOR
    ),
    tau_penalty_lambda: float = 0.0,
    tau_penalty_reference: Optional[float] = None,
    tau_penalty_sigma_us: Optional[float] = None,
    tau_penalty_sigma_lo_us: Optional[float] = None,
    n_eff_kind: str = DEFAULT_N_EFF_KIND,
    weighted_gate_chi2: Optional[bool] = None,
    shape: PeakShape | str = PeakShape.LORENTZIAN,
    spur_mask: Optional[SpurMaskSpec] = None,
    gate_budget_extra: Optional[np.ndarray] = None,
    baseline_order: Optional[int] = None,
    baseline_offset_scale: Optional[float] = None,
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
    (the minimum allowed pair separation is
    ``max(min_pair_separation_factor * fwhm, min_pair_separation_resolution_factor
    / acquisition_us)`` -- the larger of the FWHM-referenced floor and the
    active-FT resolution element ``1/T_active``); collapsed escalations are
    rejected even when the gate would accept them.

    ``significance`` is not used as a gate threshold; the F-test
    ``p_value`` is computed and recorded on each :class:`AddStep` as a
    familiar diagnostic.
    """
    shape_resolved = PeakShape.coerce(shape)
    weighted = (
        validation.DEFAULT_WEIGHTED_GATE_CHI2
        if weighted_gate_chi2 is None
        else weighted_gate_chi2
    )
    u_grid = np.asarray(offset_grid_mhz, dtype=float)
    sigma_arr = np.asarray(rms_noise, dtype=float)
    if sigma_arr.ndim == 0:
        sigma_arr = np.full(u_grid.size, float(sigma_arr))
    if spur_mask is not None:
        keep = ~spur_mask.bin_mask(u_grid)
        if not keep.any():
            keep = np.ones(u_grid.size, dtype=bool)
    else:
        keep = np.ones(u_grid.size, dtype=bool)
    sigma_keep = sigma_arr[keep]
    budget_keep: Optional[np.ndarray] = None
    if gate_budget_extra is not None:
        budget_keep = np.asarray(gate_budget_extra, dtype=float)[keep]

    def _trigger_rchi2(res_fit: WindowFitResult) -> float:
        # Escalation-trigger reduced chi-squared. Under the sigma_eff gate
        # (:data:`validation.DEFAULT_GATE_SIGMA_EFF_KAPPA`) the trigger is
        # computed against the fidelity-inflated noise of the fit's own
        # model: a bright line sitting at its lineshape floor reads ~1 and
        # does not escalate (the floor is irreducible -- straddled absorber
        # peaks are not the remedy), while a genuine unresolved blend leaves
        # reducible misfit at bins the model under-covers and still fires.
        kappa = validation.DEFAULT_GATE_SIGMA_EFF_KAPPA
        if kappa is None:
            return float(res_fit.reduced_chi2)
        chi2_eff = validation.sigma_eff_chi2(
            np.asarray(res_fit.residual)[keep],
            sigma_keep,
            np.asarray(res_fit.fitted_spectrum)[keep],
            kappa,
            extra=budget_keep,
        )
        dof = max(res_fit.n_data - res_fit.n_params, 1)
        return chi2_eff / dof

    fit_kwargs: dict[str, Any] = dict(
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
        tau_penalty_sigma_us=tau_penalty_sigma_us,
        tau_penalty_sigma_lo_us=tau_penalty_sigma_lo_us,
        shape=shape_resolved,
        spur_mask=spur_mask,
    )
    if baseline_order is not None:
        fit_kwargs["baseline_order"] = int(baseline_order)
        fit_kwargs["baseline_offset_scale"] = baseline_offset_scale
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
                shape=shape_resolved,
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
    if not fit1.success or _trigger_rchi2(fit1) <= rchi2_threshold:
        return best, audit

    # Elevated reduced chi-squared -> retry as a straddled blend.
    straddle = straddle_factor * fwhm_mhz
    min_pair_sep = _effective_min_pair_separation(
        fwhm_mhz,
        acquisition_us,
        min_pair_separation_factor,
        min_pair_separation_resolution_factor,
    )
    prev = fit1
    for k in range(2, max_k + 1):
        positions = seed_offset_mhz + (np.arange(k) - 0.5 * (k - 1)) * straddle
        init = [
            _seed_peak(
                float(pos),
                offset_grid_mhz,
                complex_spectrum,
                tau0_us,
                acquisition_us,
                shape=shape_resolved,
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
        # Residual re-seed alternative: the symmetric straddle only reaches
        # ~straddle_factor * FWHM, so a multi-FWHM blend whose Stage 3
        # detection was pulled off-position (a steep-skirt shoulder shifts
        # both apparent maxima) is out of its basin. Seed the extra
        # component at the previous fit's residual-magnitude maximum
        # instead -- the data says where the unmodeled line is -- and keep
        # whichever trial converges better. Attempted only on real local
        # evidence (the blend-split bar); the collapse check and the AICc
        # gate below judge the winning trial exactly as before.
        res_prev = np.abs(np.asarray(prev.residual))
        res_prev = np.where(keep, res_prev, 0.0)
        i_res = int(np.argmax(res_prev))
        if res_prev[i_res] >= DEFAULT_BLEND_SPLIT_MIN_SNR * sigma_arr[i_res]:
            init_res = list(prev.peaks) + [
                _seed_peak(
                    float(u_grid[i_res]),
                    offset_grid_mhz,
                    np.asarray(prev.residual),
                    prev.tau_us,
                    acquisition_us,
                    shape=shape_resolved,
                )
            ]
            trial_res = fit_window(
                offset_grid_mhz,
                complex_spectrum,
                rms_noise,
                init_res,
                tau0_us,
                acquisition_us,
                **fit_kwargs,
            )
            if trial_res.success and (
                not trial.success or trial_res.chi_squared < trial.chi_squared
            ):
                trial = trial_res
        p_value, f_stat, _ = calculate_chi_squared_improvement(
            prev.chi_squared,
            trial.chi_squared,
            trial.n_params - prev.n_params,
            trial.n_data,
            trial.n_params,
        )
        # Post-fit sanity check: reject escalations whose peaks collapsed onto
        # the same offset (the cancelling-phase degenerate solution). A
        # genuine sub-separation BLEND earns the escape: when the escalation's
        # raw chi-squared win is overwhelming and every violating pair is
        # constructive, the straddle resolved physical structure, not the
        # pathology (see :data:`validation.DEFAULT_PAIR_CANCELLATION_MAX`;
        # measured K=2 escalations 6-29x better in raw chi-squared were
        # vetoed here on 363 w157/w100 and 360 w56).
        collapsed = False
        if trial.success and len(trial.peaks) >= 2 and min_pair_sep > 0.0:
            violating: list[tuple[int, int]] = []
            offs = np.asarray([pk.offset_mhz for pk in trial.peaks], dtype=float)
            for ii in range(offs.size):
                for jj in range(ii + 1, offs.size):
                    if abs(offs[ii] - offs[jj]) < min_pair_sep:
                        violating.append((ii, jj))
            if violating:
                collapsed = True
                evidence = prev.chi_squared - trial.chi_squared
                dk = max(trial.n_params - prev.n_params, 1)
                if all(
                    validation.blend_pair_escape(
                        evidence,
                        dk,
                        trial.peaks[ii].amplitude,
                        trial.peaks[ii].phase,
                        trial.peaks[jj].amplitude,
                        trial.peaks[jj].phase,
                        # Relative lane: the escalation splits the feature the
                        # accepted seed explains, so its evidence scale is the
                        # seed's own null-referenced chi-squared win.
                        feature_evidence=null_chi2 - prev.chi_squared,
                    )
                    for ii, jj in violating
                ):
                    collapsed = False
                for ii, jj in violating:
                    validation.debug_fringe_dump(
                        "seeder",
                        cand_offset=seed_offset_mhz,
                        pair_off_a=trial.peaks[ii].offset_mhz,
                        pair_off_b=trial.peaks[jj].offset_mhz,
                        pair_amp_a=trial.peaks[ii].amplitude,
                        pair_amp_b=trial.peaks[jj].amplitude,
                        pair_cancellation=validation.pair_cancellation_fraction(
                            trial.peaks[ii].amplitude,
                            trial.peaks[ii].phase,
                            trial.peaks[jj].amplitude,
                            trial.peaks[jj].phase,
                        ),
                        tau_us=trial.tau_us,
                        acquisition_us=acquisition_us,
                        raw_chi2_more=trial.chi_squared,
                        raw_chi2_less=prev.chi_squared,
                        null_chi2=null_chi2,
                        n_params_delta=dk,
                        passes=int(not collapsed),
                    )
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
                trial.fitted_spectrum,
                kind=n_eff_kind,
                sigma=sigma_arr,
            )
            aicc_trial, aicc_prev = gate_aicc_pair(
                n_eff,
                more_n_params=trial.n_params,
                less_n_params=prev.n_params,
                more_chi2_raw=trial.chi_squared,
                less_chi2_raw=prev.chi_squared,
                weighted=weighted,
                more_residual=np.asarray(trial.residual)[keep],
                less_residual=np.asarray(prev.residual)[keep],
                rms_noise=sigma_keep,
                weight_model=np.asarray(trial.fitted_spectrum)[keep],
                n_eff_kind=n_eff_kind,
                ref_reduced_chi2=trial.reduced_chi2,
                budget_extra=budget_keep,
            )
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
        if _trigger_rchi2(trial) <= rchi2_threshold:
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
    min_pair_separation_resolution_factor: float = (
        DEFAULT_MIN_PAIR_SEPARATION_RESOLUTION_FACTOR
    ),
    blend_split_min_snr: float = DEFAULT_BLEND_SPLIT_MIN_SNR,
    tau_penalty_lambda: float = DEFAULT_TAU_PENALTY_LAMBDA,
    tau_penalty_n_sigma: float = DEFAULT_TAU_PENALTY_N_SIGMA,
    weak_window_snr_threshold: float = DEFAULT_WEAK_WINDOW_SNR_THRESHOLD,
    fit_tau_min_snr: float = DEFAULT_FIT_TAU_MIN_SNR,
    tau_apodization_us: Optional[float] = None,
    tau_maj_us: Optional[float] = None,
    sigma_tau_us: Optional[float] = None,
    tau_anchor_us: Optional[float] = None,
    tau_penalty_sigma_lo_factor: float = 1.0,
    n_eff_kind: str = DEFAULT_N_EFF_KIND,
    weighted_gate_chi2: Optional[bool] = None,
    shape: PeakShape | str = PeakShape.LORENTZIAN,
    spur_mask: Optional[SpurMaskSpec] = None,
    gate_budget_extra: Optional[np.ndarray] = None,
    gate_background: Optional[np.ndarray] = None,
    gate_line_escape: bool = True,
    baseline_order: Optional[int] = None,
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
    max_peaks : int, default 0
        Cap on the number of lines fit; ``0`` (the default) disables the cap so the
        add-loop is bounded by the candidate set (and ``patience`` / separation).
        The AICc-with-``n_eff`` accept gate self-regulates K, so a hard cap only
        truncated dense clusters the gate would otherwise resolve. A positive value
        restores an explicit cap.
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
        rejects an escalation whose fitted peaks ended up within the minimum
        pair separation of each other.
    min_pair_separation_resolution_factor : float, default
        :data:`DEFAULT_MIN_PAIR_SEPARATION_RESOLUTION_FACTOR`
        Resolution-referenced floor on the minimum pair separation, in units
        of the active-FT resolution element ``1/T_active`` (=
        ``1/acquisition_us`` MHz). The effective minimum pair separation is
        ``max(min_pair_separation_factor * fwhm,
        min_pair_separation_resolution_factor / acquisition_us)`` -- the
        resolution term catches sub-resolution duplicate pairs the FWHM-only
        floor licenses on narrow features (the per-window FWHM can fall below
        the Fourier limit). See GitHub issue #13.
    blend_split_min_snr : float, default :data:`DEFAULT_BLEND_SPLIT_MIN_SNR`
        Residual-magnitude evidence (in local sigma) above which a candidate
        failing the pre-fit peak-separation constraint still gets a trial
        fit -- the blend-split trial (see
        :data:`DEFAULT_BLEND_SPLIT_MIN_SNR`). ``0`` restores the outright
        rejection.
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
    tau_anchor_us, tau_penalty_sigma_lo_factor : optional
        Asymmetric long-anchor tau penalty, forwarded to
        :func:`derive_window_fit_constraints`. **Dormant**: the defaults
        (``None`` / ``1.0``) reproduce the symmetric prior and no production
        path sets them -- see that function's docstring for why the wiring is
        deliberately omitted.
    n_eff_kind : str, default :data:`DEFAULT_N_EFF_KIND`
        Effective-sample-size kind shared by every gate this fit runs --
        the conservative add-one-peak accept gate (main loop and
        :func:`_blend_aware_seed`'s K=2/K=3 escalation) and the final
        :func:`knockout_test` sweep. The default information-weighted
        kind keeps ``n_eff`` in the AICc-identifiable regime on narrow
        features; gates that do diverge to ``+inf`` fall through to
        their REJECT-on-tie branch (preserve the simpler model).

    spur_mask : SpurMaskSpec, optional
        Clock/LO-spur bins to exclude from every inner fit's residual /
        chi-squared (forwarded verbatim to :func:`fit_window` and
        :func:`knockout_test`). Keeps a spur from inflating the AICc gate
        and the knockout sweep. ``None`` masks nothing.

    gate_budget_extra : np.ndarray, optional
        Per-bin amplitude budget (aligned with ``offset_grid_mhz``) added in
        quadrature to the gate noise by the sigma_eff gate variant -- the
        fidelity allowance ``kappa_skirt * |frozen background|`` for skirt
        structure subtracted from this window's data (see
        :data:`~ftmwpipeline.fitting.validation.DEFAULT_GATE_SIGMA_EFF_KAPPA_SKIRT`).
        Threaded to the seeder, the add-loop gate, and the knockout sweep;
        inert unless the sigma_eff gate is active. The NLS objective and all
        reported chi-squared stay on the raw Stage 2 noise.

    Returns
    -------
    ConservativeFitResult
        The final fit, the add-one-peak audit trail, and the knockout results.
    """
    shape_resolved = PeakShape.coerce(shape)
    u = np.asarray(offset_grid_mhz, dtype=float)
    z = np.asarray(complex_spectrum, dtype=np.complex128)
    sigma = np.asarray(rms_noise, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(u.size, float(sigma))
    # np.interp needs an ascending grid; sort the window once on entry.
    order = np.argsort(u)
    u, z, sigma = u[order], z[order], sigma[order]
    budget: Optional[np.ndarray] = None
    if gate_budget_extra is not None:
        budget = np.asarray(gate_budget_extra, dtype=float)[order]
    background: Optional[np.ndarray] = None
    if gate_background is not None:
        background = np.asarray(gate_background, dtype=np.complex128)[order]

    weighted = (
        validation.DEFAULT_WEIGHTED_GATE_CHI2
        if weighted_gate_chi2 is None
        else weighted_gate_chi2
    )
    if spur_mask is not None:
        keep = ~spur_mask.bin_mask(u)
        if not keep.any():
            keep = np.ones(u.size, dtype=bool)
    else:
        keep = np.ones(u.size, dtype=bool)
    sigma_keep = sigma[keep]
    budget_keep: Optional[np.ndarray] = None if budget is None else budget[keep]

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
        tau_penalty_n_sigma=tau_penalty_n_sigma,
        weak_window_snr_threshold=weak_window_snr_threshold,
        fit_tau_min_snr=fit_tau_min_snr,
        tau_apodization_us=tau_apodization_us,
        tau_maj_us=tau_maj_us,
        sigma_tau_us=sigma_tau_us,
        tau_anchor_us=tau_anchor_us,
        tau_penalty_sigma_lo_factor=tau_penalty_sigma_lo_factor,
        shape=shape_resolved,
    )
    tau_bounds = constraints.tau_bounds
    fwhm = constraints.fwhm
    min_separation = constraints.min_separation
    amp_max = constraints.amp_max
    amp_floor = constraints.amp_floor
    fit_tau_eff = constraints.fit_tau_eff
    tau_penalty_ref = constraints.tau_penalty_reference
    tau_penalty_sigma_us = constraints.tau_penalty_sigma_us
    effective_tau_penalty_lambda = constraints.effective_tau_penalty_lambda
    fit_kwargs_inner = dict(constraints.fit_kwargs_inner)
    fit_kwargs_inner.setdefault("shape", shape_resolved)
    # Joint complex-baseline nuisance term for every inner fit (seeder
    # escalations, add-loop trials, knockout refits). On a window whose data
    # carries a smooth leakage pedestal the peaks-only model otherwise buys
    # chi-squared by collapsing the shared tau onto the pedestal, which
    # poisons every downstream decision (separations measured in a ballooned
    # FWHM, the rescue consolidating onto bound-pinned absorbers). The
    # caller triggers this (see ``fit_window_with_fixed_contributors``).
    baseline_offset_scale: Optional[float] = None
    if baseline_order is not None and u.size:
        baseline_offset_scale = float(np.max(np.abs(u))) or 1.0
        fit_kwargs_inner["baseline_order"] = int(baseline_order)
        fit_kwargs_inner["baseline_offset_scale"] = baseline_offset_scale

    remaining = sorted(
        candidate_offsets,
        key=lambda o: -abs(float(np.interp(o, u, np.abs(z)))),
    )
    if not remaining:
        empty = fit_window(
            u,
            z,
            sigma,
            [],
            tau0_us,
            acquisition_us,
            shape=shape_resolved,
            spur_mask=spur_mask,
        )
        return ConservativeFitResult(empty, [], [])

    null = fit_window(
        u,
        z,
        sigma,
        [],
        tau0_us,
        acquisition_us,
        shape=shape_resolved,
        spur_mask=spur_mask,
    )
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
        min_pair_separation_resolution_factor=(min_pair_separation_resolution_factor),
        tau_penalty_lambda=effective_tau_penalty_lambda,
        tau_penalty_reference=tau_penalty_ref,
        tau_penalty_sigma_us=tau_penalty_sigma_us,
        tau_penalty_sigma_lo_us=constraints.tau_penalty_sigma_lo_us,
        n_eff_kind=n_eff_kind,
        weighted_gate_chi2=weighted,
        shape=shape_resolved,
        spur_mask=spur_mask,
        gate_budget_extra=budget,
        baseline_order=baseline_order,
        baseline_offset_scale=baseline_offset_scale,
    )
    if not current.success:
        return ConservativeFitResult(current, audit, [])

    tentative: list[ModelPeak] = []
    consecutive_rejects = 0
    while remaining and (
        max_peaks <= 0 or current.n_peaks + len(tentative) < max_peaks
    ):
        in_model = list(current.peaks) + tentative
        residual = z - model_spectrum(
            u,
            in_model,
            current.tau_us,
            acquisition_us,
            shape=shape_resolved,
        )
        cand = max(
            remaining,
            key=lambda o: abs(float(np.interp(o, u, np.abs(residual)))),
        )
        remaining.remove(cand)

        # Test only the candidate's distance to the peaks already in the
        # model. Existing-vs-existing pairs are policed by the blend seeder's
        # own collapse check and the merge tiers -- and those legitimately
        # admit pairs down to 0.5 FWHM, below this check's 1.0 FWHM bar, so
        # including them here would poison the check and reject every later
        # candidate wholesale (the deep-skirt windows lost their real
        # Stage 3 lines exactly this way).
        existing = [pk.offset_mhz for pk in in_model]
        sep_ok = all(abs(e - cand) >= min_separation for e in existing)
        sep_waived = False
        if not sep_ok:
            # Blend-split trial: an existing peak parked at a blend's
            # compromise position leaves the residual maximum *inside* its
            # own separation dead zone, so rejecting here forecloses ever
            # resolving the blend. When the residual at the candidate
            # carries real magnitude evidence, run the trial anyway -- the
            # NLS is free to move both the candidate and the blocking peak
            # and splits the blend when the data supports it; the post-fit
            # collapse check and the AICc gate arbitrate as usual.
            res_snr = abs(float(np.interp(cand, u, np.abs(residual)))) / max(
                float(np.interp(cand, u, sigma)), float(np.finfo(float).tiny)
            )
            if blend_split_min_snr > 0.0 and res_snr >= blend_split_min_snr:
                sep_waived = True
            else:
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
            + [
                _seed_peak(
                    cand,
                    u,
                    residual,
                    current.tau_us,
                    acquisition_us,
                    shape=shape_resolved,
                )
            ]
        )
        trial = fit_window(
            u,
            z,
            sigma,
            trial_init,
            tau0_us,
            acquisition_us,
            spur_mask=spur_mask,
            **fit_kwargs_inner,
        )
        # Post-fit collapse check on the *candidate*: the NLS can migrate a
        # legitimately-separated candidate onto an existing bright core and
        # converge to the cancelling near-duplicate pair (huge opposite-phase
        # amplitudes buying raw chi-squared) -- the same degenerate solution
        # the seeder rejects on its escalations. Scope deliberately narrow:
        # only pairs involving the candidate's fitted position (a transient
        # collapse among *other* trial members must not veto this candidate),
        # and only below HALF the effective separation floor (the structural
        # scale of the merge tiers) -- real close pairs legitimately fit just
        # under the floor and the merge/knockout machinery owns that band.
        # Drop the candidate outright (NOT into the tentative batch, where it
        # would re-collapse inside every later trial).
        if trial.success and trial.n_peaks >= 2:
            sep_eff = _effective_min_pair_separation(
                fwhm,
                acquisition_us,
                min_pair_separation_factor,
                min_pair_separation_resolution_factor,
            )
            trial_offsets = [pk.offset_mhz for pk in trial.peaks]
            # The candidate is structurally the LAST trial peak (trial_init
            # appends it; the NLS preserves parameter order). Nearest-to-seed
            # matching misidentifies it when it migrates past an existing
            # peak -- on a blend-split trial it routinely converges closer
            # to the bright core than to its own seed.
            ci = len(trial_offsets) - 1
            sep_ok_post = all(
                abs(o - trial_offsets[ci]) >= 0.5 * sep_eff
                for i, o in enumerate(trial_offsets)
                if i != ci
            )
            if not sep_ok_post:
                # Blend escape: a candidate that converged sub-separation
                # beside an existing peak with overwhelming raw evidence and
                # a constructive pair is an unresolved blend, not the
                # cancelling absorber this check targets.
                cj = min(
                    (i for i in range(len(trial_offsets)) if i != ci),
                    key=lambda i: abs(trial_offsets[i] - trial_offsets[ci]),
                )
                if validation.blend_pair_escape(
                    current.chi_squared - trial.chi_squared,
                    max(trial.n_params - current.n_params, 1),
                    trial.peaks[ci].amplitude,
                    trial.peaks[ci].phase,
                    trial.peaks[cj].amplitude,
                    trial.peaks[cj].phase,
                ):
                    sep_ok_post = True
            if not sep_ok_post:
                audit.append(
                    AddStep(
                        n_peaks_before=current.n_peaks,
                        candidate_offset_mhz=cand,
                        chi2_before=current.chi_squared,
                        chi2_after=trial.chi_squared,
                        f_statistic=0.0,
                        p_value=1.0,
                        aic_before=current.aic,
                        aic_after=trial.aic,
                        separation_ok=False,
                        decision="reject",
                        reason="trial fit collapsed peaks within min separation",
                    )
                )
                continue
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
                trial.fitted_spectrum,
                kind=n_eff_kind,
                sigma=sigma,
            )
            aicc_trial, aicc_current = gate_aicc_pair(
                n_eff,
                more_n_params=trial.n_params,
                less_n_params=current.n_params,
                more_chi2_raw=trial.chi_squared,
                less_chi2_raw=current.chi_squared,
                weighted=weighted,
                more_residual=np.asarray(trial.residual)[keep],
                less_residual=np.asarray(current.residual)[keep],
                rms_noise=sigma_keep,
                weight_model=np.asarray(trial.fitted_spectrum)[keep],
                n_eff_kind=n_eff_kind,
                ref_reduced_chi2=trial.reduced_chi2,
                budget_extra=budget_keep,
            )
            aicc_delta = aicc_trial - aicc_current
            passes = aicc_trial < aicc_current
            escape_dchi2 = 0.0
            # Structural index, not nearest-to-seed (see the collapse check).
            ci_cand = len(trial.peaks) - 1
            # cand_template is consumed only by the line-evidence escape branch
            # and by debug_fringe_dump (a no-op unless FTMW_DEBUG_FRINGE_DIR is
            # set).  Compute it lazily to avoid the model_spectrum eval on the
            # majority of candidates where neither path runs.
            _need_template = (
                not passes
                and gate_line_escape
                and validation.DEFAULT_GATE_LINE_ESCAPE_LAMBDA is not None
            ) or bool(import_os.environ.get("FTMW_DEBUG_FRINGE_DIR"))
            cand_template: Optional[np.ndarray] = None
            if _need_template:
                cand_template = model_spectrum(
                    u,
                    [trial.peaks[ci_cand]],
                    trial.tau_us,
                    acquisition_us,
                    shape=shape_resolved,
                )
            # Line-evidence escape hatch: the sigma_eff currency discounts
            # evidence under the trial's own bright candidate (the shared
            # weight model includes the very component being tested) and the
            # fidelity floor scales the bar with the *remaining* unmodeled
            # misfit -- on a multi-line blend window both conspire to reject
            # a real bright line at raw delta-chi2 ~ 1e4-1e5 (655 w527). A
            # rejected candidate whose disputed evidence survives the
            # matched-filter test in raw currency -- beyond the span of the
            # established peaks' lineshape-error modes and the frozen
            # background's skirt-error modes -- is accepted anyway. See
            # :data:`validation.DEFAULT_GATE_LINE_ESCAPE_LAMBDA`.
            if (
                not passes
                and gate_line_escape
                and validation.DEFAULT_GATE_LINE_ESCAPE_LAMBDA is not None
                and cand_template is not None
            ):
                evidence = np.asarray(trial.residual) + cand_template
                others = [pk for i, pk in enumerate(trial.peaks) if i != ci_cand]
                tpl_keep = cand_template[keep]
                sl = validation.line_escape_support_slice(tpl_keep)
                if sl is None:
                    # Template is empty/zero or support too narrow: escape
                    # takes the early-return path (amax<=0 / e.size<3 / m<3)
                    # regardless of columns -- pass empty columns for speed.
                    escaped, escape_dchi2 = validation.line_evidence_escape(
                        evidence[keep],
                        sigma_keep,
                        tpl_keep,
                        [],
                        n_params_peak=max(trial.n_params - current.n_params, 1),
                    )
                else:
                    # Build background columns on the full grid; peak columns
                    # on the support subgrid (pointwise -- byte-identical to
                    # full-grid eval then slicing).
                    bg_cols = validation.line_escape_background_columns(u, background)
                    pk_cols = validation.line_escape_peak_columns(
                        u[keep][sl],
                        others,
                        trial.tau_us,
                        acquisition_us,
                        shape=shape_resolved,
                    )
                    nuisance_sliced = [col[keep][sl] for col in bg_cols] + pk_cols
                    escaped, escape_dchi2 = validation.line_evidence_escape(
                        evidence[keep][sl],
                        sigma_keep[sl],
                        tpl_keep[sl],
                        nuisance_sliced,
                        n_params_peak=max(trial.n_params - current.n_params, 1),
                    )
                if escaped:
                    passes = True
            validation.debug_fringe_dump(
                "addloop",
                u_keep=u[keep],
                less_res=(
                    np.asarray(trial.residual)[keep] + cand_template[keep]
                    if cand_template is not None
                    else None
                ),
                more_res=np.asarray(trial.residual)[keep],
                cur_res=np.asarray(current.residual)[keep],
                sigma=sigma_keep,
                budget=budget_keep,
                weight_model=np.asarray(trial.fitted_spectrum)[keep],
                template=cand_template[keep] if cand_template is not None else None,
                cand_offset=cand,
                fitted_offset=trial.peaks[ci_cand].offset_mhz,
                tau_us=trial.tau_us,
                acquisition_us=acquisition_us,
                raw_chi2_more=trial.chi_squared,
                raw_chi2_less=current.chi_squared,
                aicc_delta=aicc_delta,
                escape_dchi2=escape_dchi2,
                passes=int(passes),
            )
        else:
            n_eff = float("nan")
            aicc_delta = float("nan")
            escape_dchi2 = 0.0
            passes = False
        if passes:
            decision = "promote" if tentative else "accept"
            reason = f"+{len(tentative) + 1} line(s)"
            if escape_dchi2 and aicc_trial >= aicc_current:
                reason += f" (line-evidence escape dchi2={escape_dchi2:.0f})"
            if sep_waived:
                reason += " (blend-split trial)"
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
                    separation_ok=not sep_waived,
                    decision=decision,
                    reason=reason,
                    n_eff=float(n_eff),
                    aicc_delta=float(aicc_delta),
                )
            )
            current = trial
            tentative = []
            consecutive_rejects = 0
        elif sep_waived:
            # A failed blend-split trial dies outright: it must not join the
            # tentative batch (a sub-separation peak there would re-collapse
            # inside every later trial) and must not consume patience (the
            # legacy path skipped these candidates without stopping the loop).
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
                    separation_ok=False,
                    decision="reject",
                    reason="blend-split trial failed the gate",
                    n_eff=float(n_eff),
                    aicc_delta=float(aicc_delta),
                )
            )
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
        u,
        z,
        sigma,
        current,
        acquisition_us,
        fit_kwargs_inner=fit_kwargs_inner,
        n_eff_kind=n_eff_kind,
        weighted_gate_chi2=weighted,
        significance=significance,
        spur_mask=spur_mask,
        gate_budget_extra=budget,
    )
    seed_unsupported = False
    if validation.DEFAULT_ENFORCE_SEED_KNOCKOUT and current.n_peaks == 1 and knockouts:
        # The enforcement currency is deliberately *raw*: under the
        # penalized gate the lone seed must clear the plain evidence bar
        # ``Delta chi2_raw > 2*lambda*k`` against the null fit, WITHOUT
        # the sigma_eff/skirt budget. The budget discounts evidence under
        # bright frozen structure -- right for incremental adds chasing
        # subtraction error, but a window's single dominant feature is
        # routinely a real line riding that same pedestal (the dense
        # forest case), and budget currency would delete it wholesale.
        # Sub-bar dust (raw Delta chi2 ~ 10-20 at SNR ~ 2) still dies.
        lam = validation.DEFAULT_GATE_PENALTY_LAMBDA
        if lam is not None:
            delta_raw = null.chi_squared - current.chi_squared
            seed_unsupported = delta_raw <= 2.0 * float(lam) * float(current.n_params)
        else:
            seed_unsupported = not knockouts[0].supported
    if seed_unsupported:
        # Enforce the K=1-vs-null verdict. The seed is installed without
        # facing the accept gate, and the consolidation sweeps that act
        # on knockout verdicts only run inside an accepted rescue round
        # -- so without this a quiet window keeps a lone unsupported
        # peak (see :data:`validation.DEFAULT_ENFORCE_SEED_KNOCKOUT`).
        audit.append(
            AddStep(
                n_peaks_before=1,
                candidate_offset_mhz=float(current.peaks[0].offset_mhz),
                chi2_before=current.chi_squared,
                chi2_after=null.chi_squared,
                f_statistic=float("nan"),
                p_value=knockouts[0].p_value,
                aic_before=current.aic,
                aic_after=null.aic,
                separation_ok=True,
                decision="knockout-null",
                reason="lone seed unsupported by its K=1-vs-null knockout",
                n_eff=knockouts[0].n_eff,
                aicc_delta=knockouts[0].aicc_delta,
            )
        )
        return ConservativeFitResult(null, audit, [])
    return ConservativeFitResult(current, audit, knockouts)
