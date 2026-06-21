"""
Statistical-test and linewidth-physics helpers for Stage 5 fitting.

These are the small, pure helpers the conservative add-one-peak loop
(:mod:`ftmwpipeline.fitting.window_fit`) leans on -- the nested-model F-test,
the AIC, the noise-weighted chi-squared, the peak-separation constraint, and
the apodization / finite-T linewidth. Most are ported from the surviving
bcfitting reference shell (``dev-docs/planning/stage5-fitting.md``, "Reuse
map"); :func:`feature_fwhm` is new -- the exact ``h_T`` linewidth, used in
preference to the analytic apodization estimate for the model's own
resolution scale.

Noise convention
----------------
The canonical Stage 2 ``rms_noise`` is a per-bin *complex* RMS ``sigma``: the
real and imaginary parts each carry variance ``sigma**2 / 2``. The
noise-weighted chi-squared therefore divides every stacked Re/Im residual
element by ``sigma / sqrt(2)`` (D-8). The bcfitting reference applied a 1.53
magnitude-to-complex factor for the same reason; with a genuine complex per-bin
sigma the correct factor is exactly ``sqrt(2)`` and nothing else.
"""

from __future__ import annotations

import itertools
import math
import os
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
from scipy.stats import f as f_distribution

from ..core.data_structures import FittedPeak
from .peak_model import ModelPeak, PeakShape, h_T_shape

__all__ = [
    "DEFAULT_N_EFF_KIND",
    "DEFAULT_WEIGHTED_GATE_CHI2",
    "DEFAULT_GATE_PENALTY_LAMBDA",
    "DEFAULT_GATE_FLOOR_SCALING",
    "DEFAULT_GATE_SIGMA_EFF_KAPPA",
    "DEFAULT_GATE_SIGMA_EFF_KAPPA_SKIRT",
    "DEFAULT_GATE_LINE_ESCAPE_LAMBDA",
    "DEFAULT_BLEND_RELATIVE_EVIDENCE_FRACTION",
    "DEFAULT_PAIR_CANCELLATION_MAX",
    "line_evidence_escape",
    "line_escape_nuisance_columns",
    "line_escape_background_columns",
    "line_escape_peak_columns",
    "line_escape_support_slice",
    "pair_cancellation_fraction",
    "blend_pair_escape",
    "DEFAULT_SHAPE_ERROR_KAPPA",
    "DEFAULT_CHI2R_NOISE_FLOOR",
    "calculate_hwhm_from_apodization",
    "feature_fwhm",
    "calculate_rms_residuals",
    "calculate_noise_weighted_chi2",
    "calculate_aic",
    "calculate_aicc",
    "sigma_eff_chi2",
    "calculate_chi_squared_improvement",
    "effective_sample_size",
    "gate_information_weights",
    "information_weighted_chi2",
    "gate_aicc_pair",
    "passes_significance_test",
    "validate_peak_separation",
    "shape_error_fraction",
    "snr_aware_chi2_pass",
    "amplitude_vif",
    "peak_quality_score",
    "PEAK_QUALITY_MAX",
]

NoiseLike = Union[float, np.ndarray]

# Effective-sample-size weighting kind shared by every Stage 5 AICc gate
# (conservative add-one-peak accept, blend-aware K=2/K=3 escalation, merge
# cleanup, knockout, iterative cleanup). Each bin is weighted by
# ``log(1 + |model|/sigma)`` -- the per-bin Shannon information of a signal-
# vs-noise detection -- and ``n_eff`` is the perplexity ``exp(H(p))`` of the
# normalized distribution. On a Lorentzian peak with peak SNR ~ 100 this
# returns ~50 bins (the bins where the skirt is significant) rather than
# the ~5 FWHM-in-bins a magnitude-concentrated weight gives. The gate then
# stays in the AICc-identifiable regime for the realistic K-vs-(K+/-1)
# transitions Stage 5 makes and only diverges to ``+inf`` when the model is
# genuinely under-determined; in the divergent case the REJECT-on-tie at
# each gate falls through to "preserve the simpler model" (do not add /
# do not merge / do not drop the peak), which is the conservative direction.
DEFAULT_N_EFF_KIND = "perplexity_log1p_snr"

# Whether the Stage 5 AICc gates score K vs K+/-1 on the information-weighted
# chi-squared (:func:`information_weighted_chi2`) instead of the raw
# noise-weighted chi-squared. The raw chi-squared is ~``2M`` for a unit-variance
# fit over ``M`` bins, so the per-peak accept benefit ``n_eff * dchi2 / (2M)``
# scales as ``1/M`` -- a window-size dependence the AICc gate should not have
# (whether a line is real is local, not a function of how many empty bins
# surround it). Weighting chi-squared by the *same* per-bin information weights
# ``w_f = log1p(|model|/sigma)`` that define ``n_eff`` makes ``chi2_w ~ n_eff``
# for a good fit regardless of ``M``, so the gate weights evidence by
# information on both sides of the ledger and the window-size sensitivity
# vanishes. Only the internal AICc gates switch; the reported chi-squared, the
# reduced chi-squared, the SNR-aware pass metric, and the F-test stay on the raw
# chi-squared (they are calibrated and externally meaningful). Read at call time
# (module attribute) so the gate is A/B-toggleable; the gate functions also take
# an explicit ``weighted_gate_chi2`` override for direct unit tests.
#
# Disabled by default: validated as over-permissive -- on the 2638 calibration
# fixture the weighted gate roughly doubled the accepted line count at fixed
# windowing, because replacing the ``~2M`` raw-chi-squared denominator with
# ``chi2_w ~ n_eff`` removes the ``n_eff/(2M)`` shrinkage the legacy gate leans
# on as its (window-size-dependent) evidence bar. The window-INDEPENDENT
# recalibration that holds is the penalized raw-chi-squared gate below.
DEFAULT_WEIGHTED_GATE_CHI2 = False

# Penalty multiplier ``lambda`` of the window-independent AIC-style gate. When
# not ``None`` the Stage 5 AICc gates score each model by
# ``score = chi2_raw + 2 * lambda * k`` and prefer the lower score
# (REJECT-on-tie). Because the two models compared share the same window bins,
# the chi-squared *difference* ``Delta chi2`` is purely the local likelihood-
# ratio statistic of the added/removed peak -- independent of the window bin
# count ``M`` and of how many empty bins surround the feature. Accept the more
# complex model iff ``Delta chi2 > lambda * 2 * Delta k``: ``lambda = 1`` is the
# textbook AIC bar (``Delta chi2 > 2 Delta k``); larger ``lambda`` is a stricter
# (BIC-like) bar. ``lambda`` is the single calibration knob, set so the 2638
# control reproduces its shipped line count; unlike the legacy ``n_eff`` gate the
# bar no longer drifts with window size. ``None`` selects the legacy
# AICc-with-``n_eff`` gate. Read at call time so it is A/B-toggleable. Takes
# precedence over :data:`DEFAULT_WEIGHTED_GATE_CHI2`.
DEFAULT_GATE_PENALTY_LAMBDA: Optional[float] = 5.0

# Whether the penalized gate scales its bar by the model-fidelity floor
# ``max(1, ref_reduced_chi2)`` (the more-complex model's reduced chi-squared).
# Intended to restore the SNR scaling the legacy fractional benefit carried
# (suppressing high-SNR lineshape-absorber over-add), but MEASURED to backfire:
# a dense under-fit window has a high reduced chi-squared for a *reducible*
# reason (missing real lines), so scaling by it raises the bar exactly where
# recovery is needed and re-collapses the fit to K=1 (363 cap=8 w0230: K=6 ->
# K=1, chi2_r 52.7). Reducible (dense under-fit) and irreducible (high-SNR
# lineshape floor) both raise reduced chi-squared but want opposite bars, so
# reduced chi-squared alone cannot discriminate them. Left OFF (floor=1, the
# pure penalized bar) pending a discriminator that is not the current
# reduced chi-squared. Read at call time.
DEFAULT_GATE_FLOOR_SCALING = False

# Per-bin model-fidelity noise budget ``kappa`` for the penalized gate's
# chi-squared. When not ``None`` (and :data:`DEFAULT_GATE_PENALTY_LAMBDA` is
# set) the penalized gate computes both models' chi-squared against the
# inflated per-bin noise ``sigma_eff_f**2 = sigma_f**2 + (kappa*|model_f|)**2``
# (:func:`sigma_eff_chi2`, weights model = the more-complex model, shared by
# both sides like ``n_eff``). This is the per-bin localisation of the
# SNR-aware window allowance ``chi2r <= F + (kappa*SNR_max)**2``
# (:func:`snr_aware_chi2_pass`): residual evidence sitting *under a bright
# model component* is discounted by the lineshape-fidelity budget
# ``kappa*|model|`` (irreducible misfit -- a candidate absorbing it gains
# ~O(dk) in chi2_eff and cannot clear the ``2*lambda*dk`` bar), while residual
# evidence at bins where the model is small keeps its full noise weighting
# (reducible misfit -- a missing real line still clears the bar). The
# discriminator the chi2r-floor variant lacked is *location*: dense under-fit
# and the high-SNR lineshape floor both raise the window's reduced
# chi-squared, but only the floor's residuals sit under bright model bins.
# ``None`` keeps the penalized gate on the raw chi-squared. Read at call time
# so it is A/B-toggleable; the default tracks :data:`DEFAULT_SHAPE_ERROR_KAPPA`
# (one fidelity budget in the system).
DEFAULT_GATE_SIGMA_EFF_KAPPA: Optional[float] = 0.05

# Fractional fidelity of a *frozen contributor background* (the subtracted
# skirt of a bright out-of-window line), for the sigma_eff gate budget. The
# local ``kappa * |model|`` term cannot see subtracted structure: the
# contributor's skirt is removed from the window data *before* fitting, so
# its extrapolation error -- coherent wing fringes that run ~10-40% of the
# subtracted amplitude tens of MHz from the line (the frozen (amp, freq,
# phase) is read at the line and ``h_T`` is evaluated with the dependent
# window's tau) -- arrives as genuinely significant residual structure on
# bins where the *local* model is near zero. When set (and
# :data:`DEFAULT_GATE_SIGMA_EFF_KAPPA` is active) the plan executor budgets
# it per bin: ``extra_f = kappa_skirt * |background_f|`` enters the gate's
# sigma_eff in quadrature, so skirt-error fringes are discounted while a
# real line riding the skirt (amplitude >> kappa_skirt * |skirt|) keeps its
# evidence. ``None`` adds no background budget. This is a fixed fidelity
# constant times a *model* amplitude -- not a residual-derived local noise
# estimate (the Stage 2 sigma stays the only noise authority). 0.4 sits at the
# measured knee of the bright-band fringe sweep (1512 spurious in-band lines:
# 0.2 -> 61, 0.4 -> 13) while the dense 363 anchors hold to 0.6.
DEFAULT_GATE_SIGMA_EFF_KAPPA_SKIRT: Optional[float] = 0.4

# Evidence bar (per peak-parameter, in gate-lambda units) for the
# line-evidence escape hatch (:func:`line_evidence_escape`). The sigma_eff
# fidelity currency has two structural blind spots, both measured on real
# windows: (1) the kappa*|model| weights are built from the more-complex
# model, so a bright *candidate*'s own component inflates sigma_eff under
# the very evidence for it -- on a multi-line blend window with a high
# fidelity floor the discounted evidence saturates near ``2/kappa**2`` per
# bin and a raw delta-chi2 of ~7e4 reads as worse-by-thousands (655 w527);
# (2) the kappa_skirt*|background| budget is an amplitude band, blind to
# *shape* -- a real line riding a frozen skirt at comparable magnitude sits
# inside the band and is pruned (655 w1055). The escape hatch re-tests a
# peak the gate is about to reject/drop with a matched-filter statistic in
# RAW noise currency: the disputed evidence (residual of the model without
# the peak, on the peak's support neighborhood) is fit as
# ``alpha*template + sum_j beta_j*nuisance_j``, where the nuisance columns
# span exactly the error modes the fidelity currency exists to tolerate --
# the frozen background and its frequency derivative (skirt amplitude /
# position error) and each established overlapping peak's component with
# its frequency and tau derivatives (lineshape error under a bright line,
# which keeps high-SNR absorber candidates dead). The template's marginal
# delta-chi2 beyond that nuisance span is line-evidence: fringe error is
# absorbed (bg-shaped, measured absorb ~0.7-1.0), a real line at its own
# center is not (absorb ~0). The escape fires when
# ``delta_chi2_line > 2 * lambda_escape * dk``; ``lambda_escape = 50``
# demands 10x the accept gate's lambda=5 evidence, so only
# overwhelmingly-supported peaks overrule the fidelity currency. The
# escape is strictly additive: it can flip a reject/drop into an
# accept/keep, never the reverse. ``None`` disables it.
DEFAULT_GATE_LINE_ESCAPE_LAMBDA: Optional[float] = 50.0

# Maximum phasor-cancellation fraction for the sub-separation blend escape
# (:func:`pair_cancellation_fraction`). The unconditional anti-collapse
# layers (the merge cleanup's sub-resolution tier, the seeder's straddle
# veto, the add-loop's post-fit collapse check) exist to kill ONE pathology:
# the cancelling near-duplicate pair, two large opposite-phase amplitudes
# buying chi-squared by synthesizing structure no physical pair of lines
# produces. But the same layers also destroy genuine unresolved blends the
# complex-domain evidence supports overwhelmingly -- measured rescue joint
# refits 4-58x better in raw chi-squared collapsed back by the merge tier
# and the rounds then rejected (363 w157/w100, 360 w56/w116, 1231 w51,
# 655 w306). A physical blend's members share the molecular phase field:
# their phasor sum is mostly constructive (cancellation ~ 0), while the
# pathology is destructive by construction (cancellation -> 1). A
# sub-separation pair is therefore KEPT when its raw delta-chi-squared
# evidence clears the line-escape bar (``2 * lambda_escape * dk``, the same
# overwhelming-evidence currency as :func:`line_evidence_escape`) AND its
# cancellation fraction stays below this threshold. The threshold vetoes
# only near-anti-aligned pairs: the pathology is destructive by
# construction (cancellation ~ 1), while a real blend's members can sit at
# any moderate relative phase (a measured balanced 1-resolution-element
# doublet with a 21x raw win reads 0.55 -- the chirp phase field and
# member-tau compensation legitimately rotate fitted phases). The evidence
# bar, not this veto, is the primary gate. ``None`` disables the escape
# (legacy unconditional collapse).
DEFAULT_PAIR_CANCELLATION_MAX: Optional[float] = 0.75

# Relative-evidence lane of the sub-separation blend escape. The absolute
# bar above (``2 * lambda_escape * dk`` ~ 300) is sized against high-SNR
# pathologies, but a low-SNR feature can never reach it: its WHOLE line
# carries less raw chi-squared than the bar (363 w87: a constructive
# 0.8-resolution-element doublet at snr_max 9 whose second component is
# worth 169 -- 64% of the feature's entire evidence of 263). The relative
# lane keeps a sub-separation pair when its raw delta-chi-squared clears
# this fraction of the *feature's own evidence* (the chi-squared the whole
# feature explains vs the model without it) AND the plain accept-gate bar
# ``2 * DEFAULT_GATE_PENALTY_LAMBDA * dk`` (kills dust pairs outright).
# Scale-invariance argument: the kappa-floor lineshape error a shape-error
# absorber can soak is ~ n_eff * (kappa * snr)^2, a few percent of the
# feature evidence (~ snr^2 * n_support) at ANY snr with kappa = 0.05 --
# so demanding 25% relative evidence excludes shape-error absorbers
# without an SNR cutoff while admitting genuine doublets whose members
# split the feature's information. Call sites opt in by passing
# ``feature_evidence``; ``None`` disables the lane (absolute bar only).
DEFAULT_BLEND_RELATIVE_EVIDENCE_FRACTION: Optional[float] = 0.25

# Whether ``conservative_fit`` enforces the knockout verdict on a lone seed.
# The K=1 seed is the one path into a window's accepted peak set that never
# faces the accept gate (the blend-aware seeder installs it unconditionally),
# and the consolidation sweeps that act on knockout verdicts only run inside
# an accepted residual-rescue round -- so a quiet window whose only peak
# fails its own K=1-vs-null knockout keeps it anyway. On a small-window plan
# every skirt-side window contributes one such sub-threshold "dust" line
# (SNR ~ 2, zero catalog matches). With this enabled, a single-peak fit whose
# knockout reads unsupported is replaced by the empty (null) fit at
# ``conservative_fit`` exit. Enforcement reads the knockout in *raw* penalized
# currency: a sigma_eff-budget currency was measured to remove real lines (a
# window's lone dominant feature is often a genuine line riding the pedestal
# the budget discounts), while the raw-currency removals are catalog-verified
# noise (655 A/B: 0% catalog matches among removed, median SNR 1.3).
DEFAULT_ENFORCE_SEED_KNOCKOUT = True

# Tolerated per-bin fractional model deficit in the SNR-aware acceptance gate
# (:func:`snr_aware_chi2_pass`). At extreme SNR the per-window reduced
# chi-squared is a model-fidelity floor, not a noise statistic: a sub-percent
# lineshape/tau deficit becomes hundreds of sigma per bin under a SNR ~ 1e4-1e5
# line, so chi-squared_r tracks SNR^2 even on a line fit to part-in-1e5. The
# dense vinyl-cyanide bulk sits at an honest ~1-3% Lorentzian + low-order-
# baseline fidelity floor (ruled not a defect: window sizing, cross-window
# leakage, and tau were each falsified as levers). kappa = 0.05 sits just above
# that measured floor so a cleanly-fit dense window passes and the gate flags
# only genuinely bad ones; the fractional residual
# (:func:`shape_error_fraction`) is reported alongside so the gate stays
# auditable. A second, non-vinyl-cyanide instrument is the real calibration
# ceiling for this number.
DEFAULT_SHAPE_ERROR_KAPPA = 0.05

# Noise-regime allowance ``F`` in the SNR-aware gate ``chi2r <= F +
# (kappa*SNR_max)**2``. At low SNR the ``(kappa*SNR_max)**2`` deficit term is
# negligible, so the gate reduces to ``chi2r <= F``; ``F`` must therefore budget
# for the sampling distribution of a *good* fit's reduced chi-squared, which has
# mean ~1 but variance ~2/dof (95th percentile ~1.5-2 at the typical Stage-5
# active-FT window dof of ~10-30) plus a small constant bias from the active-FT
# bin correlation. Without it the gate would reject the normal upward scatter of
# a healthy noise-dominated window. F = 3 admits that scatter (and the elevated
# but non-deficit low-SNR bulk) while the catastrophic chi2r tail still fails;
# at high SNR the deficit term dominates and F is negligible.
DEFAULT_CHI2R_NOISE_FLOOR = 3.0


# ---------------------------------------------------------------------------
# Linewidth physics
# ---------------------------------------------------------------------------
def calculate_hwhm_from_apodization(
    apodization_us: float,
    spectrum_type: str = "magnitude",
    include_natural_broadening: bool = True,
) -> float:
    """Analytic HWHM of an exponentially apodized line.

    For an exponential apodization of time constant ``tau`` the absorption
    HWHM is ``1 / (2 pi tau)``; a magnitude spectrum widens it by ``sqrt(3)``
    and natural (quadrature) broadening adds a further ``sqrt(2)``. This is the
    cheap apodization-only estimate; :func:`feature_fwhm` is the exact ``h_T``
    linewidth and is preferred for the model's resolution scale.

    Parameters
    ----------
    apodization_us : float
        Apodization time constant in microseconds (``> 0``).
    spectrum_type : str, default "magnitude"
        ``"magnitude"`` applies the ``sqrt(3)`` factor; ``"absorption"`` does
        not.
    include_natural_broadening : bool, default True
        Apply the ``sqrt(2)`` natural-broadening factor.

    Returns
    -------
    float
        Half-width at half-maximum in MHz.

    Raises
    ------
    ValueError
        If ``apodization_us`` is not positive or ``spectrum_type`` is unknown.
    """
    if apodization_us <= 0.0:
        raise ValueError("apodization_us must be positive")
    if spectrum_type not in ("magnitude", "absorption"):
        raise ValueError("spectrum_type must be 'magnitude' or 'absorption'")

    hwhm_hz = 1.0 / (2.0 * np.pi * apodization_us * 1e-6)
    if spectrum_type == "magnitude":
        hwhm_hz *= np.sqrt(3.0)
    if include_natural_broadening:
        hwhm_hz *= np.sqrt(2.0)
    return float(hwhm_hz * 1e-6)


def feature_fwhm(
    tau_us: float,
    acquisition_us: float,
    *,
    shape: "PeakShape | str" = "lorentzian",
) -> float:
    """Full width at half maximum of the magnitude line shape ``|h_T|``.

    Measured numerically on a fine grid -- the exact resolution scale of the
    finite-T model, including boxcar truncation, which the analytic
    :func:`calculate_hwhm_from_apodization` estimate omits. Used to set the
    conservative loop's peak-separation constraint and the blend-aware seeder's
    straddle (the prototype's ``feature_fwhm_mhz``; ~122 kHz at the 2638
    scale on Lorentzian).

    The Gaussian variant uses :func:`h_T_gaussian` instead. The Gaussian
    FWHM for τ_G = 8 µs / T = 12.65 µs is ~90 kHz -- noticeably narrower
    in the unwindowed limit than the Lorentzian with the same τ at the
    same acquisition.

    Parameters
    ----------
    tau_us : float
        Decay time constant (microseconds, ``> 0``). ``τ`` under
        ``shape='lorentzian'``; ``τ_G`` under ``shape='gaussian'``.
    acquisition_us : float
        Active acquisition length ``T`` in microseconds (``> 0``).
    shape : PeakShape or str, default 'lorentzian'
        Line-shape selector. The FWHM is shape-dependent.

    Returns
    -------
    float
        FWHM of ``|h_T|`` in MHz.

    Raises
    ------
    ValueError
        If ``tau_us`` or ``acquisition_us`` is not positive.
    """
    if tau_us <= 0.0:
        raise ValueError("tau_us must be positive")
    if acquisition_us <= 0.0:
        raise ValueError("acquisition_us must be positive")

    grid = np.linspace(-1.0, 1.0, 200001)
    mag = np.abs(h_T_shape(shape, grid, tau_us, acquisition_us))
    above = np.where(mag >= 0.5 * mag.max())[0]
    return float(grid[above[-1]] - grid[above[0]])


# ---------------------------------------------------------------------------
# Residual statistics
# ---------------------------------------------------------------------------
def calculate_rms_residuals(
    complex_spectrum: np.ndarray,
    fitted_spectrum: Union[np.ndarray, None] = None,
) -> float:
    """RMS of the stacked real+imaginary residuals (unweighted).

    Parameters
    ----------
    complex_spectrum : np.ndarray
        Complex window data.
    fitted_spectrum : np.ndarray, optional
        Fitted complex model; ``None`` means the zero model (the residual is
        the data itself).

    Returns
    -------
    float
        RMS over the concatenated Re/Im residual.
    """
    residual = complex_spectrum
    if fitted_spectrum is not None:
        residual = complex_spectrum - fitted_spectrum
    stacked = np.concatenate([np.real(residual), np.imag(residual)])
    return float(np.sqrt(np.mean(stacked**2)))


def calculate_noise_weighted_chi2(
    complex_spectrum: np.ndarray,
    rms_noise: NoiseLike,
    fitted_spectrum: Union[np.ndarray, None] = None,
) -> float:
    """Noise-weighted chi-squared of a model against complex window data.

    ``chi-squared = sum_k |r_k / (sigma_k / sqrt(2))|**2`` over the stacked
    Re/Im residual, so a good fit with the canonical Stage 2 noise gives a
    reduced chi-squared near 1 (D-8).

    Parameters
    ----------
    complex_spectrum : np.ndarray
        Complex window data.
    rms_noise : float or np.ndarray
        Per-bin complex noise RMS ``sigma`` (scalar or per-bin array).
    fitted_spectrum : np.ndarray, optional
        Fitted complex model; ``None`` means the zero model.

    Returns
    -------
    float
        The noise-weighted chi-squared.

    Raises
    ------
    ValueError
        If ``rms_noise`` is not strictly positive.
    """
    sigma = np.asarray(rms_noise, dtype=float)
    if np.any(sigma <= 0.0):
        raise ValueError("rms_noise must be strictly positive")
    residual = complex_spectrum
    if fitted_spectrum is not None:
        residual = complex_spectrum - fitted_spectrum
    sig_ri = sigma / np.sqrt(2.0)
    r_re = np.real(residual) / sig_ri
    r_im = np.imag(residual) / sig_ri
    return float(np.sum(r_re**2) + np.sum(r_im**2))


def sigma_eff_chi2(
    residual: np.ndarray,
    rms_noise: NoiseLike,
    model_spectrum: np.ndarray,
    kappa: float,
    extra: Optional[np.ndarray] = None,
) -> float:
    """Chi-squared against the fidelity-inflated noise ``sigma_eff``.

    ``sigma_eff_f**2 = sigma_f**2 + (kappa * |model_f|)**2`` -- the per-bin
    noise budget with the tolerated fractional model deficit ``kappa``
    (:data:`DEFAULT_SHAPE_ERROR_KAPPA`) added in quadrature. The chi-squared
    follows the same stacked Re/Im convention as
    :func:`calculate_noise_weighted_chi2` (each component carries variance
    ``sigma_eff**2 / 2``), so for ``kappa * |model| << sigma`` it reduces
    exactly to the raw noise-weighted chi-squared of the residual.

    The effect is regime-selective by *location*: bins under a bright model
    component (``|model|/sigma >> 1/kappa``) have their residual evidence
    shrunk to the fidelity budget -- a model fit to its lineshape floor
    contributes ~1 per bin there instead of ``(kappa*SNR)**2`` -- while bins
    where the model is small keep their full noise weighting. This is the
    per-bin localisation of the window-aggregate SNR-aware allowance
    ``F + (kappa * SNR_max)**2`` (:func:`snr_aware_chi2_pass`).

    Parameters
    ----------
    residual : np.ndarray
        Complex per-bin residual ``data - model`` on the (already
        spur-masked) grid; a zero-model null is simply the data.
    rms_noise : float or np.ndarray
        Per-bin complex noise RMS ``sigma`` aligned with ``residual``.
    model_spectrum : np.ndarray
        Complex model spectrum the fidelity budget is referenced to. In a
        K-vs-K±1 gate this is the **more-complex** model for both sides (the
        shared-basis discipline ``n_eff`` and the gate weights follow).
    kappa : float
        Tolerated per-bin fractional model deficit (``>= 0``).
    extra : np.ndarray, optional
        Additional per-bin amplitude budget added in quadrature (e.g.
        ``kappa_skirt * |frozen background|``,
        :data:`DEFAULT_GATE_SIGMA_EFF_KAPPA_SKIRT`) -- fidelity of bright
        structure that was *subtracted* from the data and is therefore
        invisible to ``|model_spectrum|``. Aligned with ``residual``.

    Returns
    -------
    float
        The sigma_eff-weighted chi-squared.

    Raises
    ------
    ValueError
        If ``rms_noise`` is not strictly positive.
    """
    sigma = np.asarray(rms_noise, dtype=float)
    if np.any(sigma <= 0.0):
        raise ValueError("rms_noise must be strictly positive")
    res = np.asarray(residual)
    if sigma.ndim == 0:
        sigma = np.full(res.shape, float(sigma))
    mag = np.abs(np.asarray(model_spectrum)).astype(float)
    sig_eff_sq = sigma**2 + (float(kappa) * mag) ** 2
    if extra is not None:
        ex = np.asarray(extra, dtype=float)
        sig_eff_sq = sig_eff_sq + ex**2
    r2 = (np.real(res) ** 2 + np.imag(res) ** 2) / (sig_eff_sq / 2.0)
    return float(np.sum(r2))


def line_escape_background_columns(
    offset_grid_mhz: np.ndarray,
    background: Optional[np.ndarray],
) -> List[np.ndarray]:
    """Background nuisance columns for :func:`line_evidence_escape`.

    Returns the frozen-contributor ``background`` column and its frequency
    gradient as complex regressors on the full ``offset_grid_mhz``. These
    must be built on the full grid (``np.gradient`` uses edge-adjacent
    differences at the boundaries and is not pointwise) and sliced by the
    caller after assembly.

    Returns an empty list when ``background`` is absent or all-zero.
    """
    u = np.asarray(offset_grid_mhz, dtype=float)
    cols: List[np.ndarray] = []
    if background is not None:
        bg = np.asarray(background, dtype=np.complex128)
        if bg.size == u.size and np.any(bg != 0):
            cols.append(bg)
            if u.size >= 3:
                cols.append(np.gradient(bg, u))
    return cols


def line_escape_peak_columns(
    offset_grid_mhz: np.ndarray,
    peaks: Sequence[ModelPeak],
    tau_us: float,
    acquisition_us: float,
    *,
    shape: "PeakShape | str" = PeakShape.LORENTZIAN,
) -> List[np.ndarray]:
    """Per-peak nuisance columns for :func:`line_evidence_escape`.

    For each established peak (the model WITHOUT the disputed peak): its
    component, its frequency derivative, and its tau derivative -- 3 complex
    columns evaluated on ``offset_grid_mhz``. Derivatives are central finite
    differences (a quarter grid-bin in frequency, 5% in tau).

    ``model_spectrum`` is pointwise in the frequency grid, so these columns
    may be built on any subgrid of the window -- evaluating on a subgrid is
    byte-identical to building on the full grid and slicing.
    """
    from .peak_model import model_spectrum

    u = np.asarray(offset_grid_mhz, dtype=float)
    cols: List[np.ndarray] = []
    if u.size >= 2:
        df = float(np.median(np.abs(np.diff(np.sort(u))))) or 1e-3
    else:
        df = 1e-3
    delta_f = 0.25 * df
    delta_tau = 0.05 * float(tau_us) if tau_us > 0 else 0.0
    for pk in peaks:
        comp = model_spectrum(u, [pk], tau_us, acquisition_us, shape=shape)
        cols.append(comp)
        plus = ModelPeak(pk.amplitude, pk.offset_mhz + delta_f, pk.phase)
        minus = ModelPeak(pk.amplitude, pk.offset_mhz - delta_f, pk.phase)
        d_f = (
            model_spectrum(u, [plus], tau_us, acquisition_us, shape=shape)
            - model_spectrum(u, [minus], tau_us, acquisition_us, shape=shape)
        ) / (2.0 * delta_f)
        cols.append(d_f)
        if delta_tau > 0.0:
            d_tau = (
                model_spectrum(u, [pk], tau_us + delta_tau, acquisition_us, shape=shape)
                - model_spectrum(
                    u, [pk], tau_us - delta_tau, acquisition_us, shape=shape
                )
            ) / (2.0 * delta_tau)
            cols.append(d_tau)
    return cols


def line_escape_nuisance_columns(
    offset_grid_mhz: np.ndarray,
    peaks: Sequence[ModelPeak],
    tau_us: float,
    acquisition_us: float,
    *,
    shape: "PeakShape | str" = PeakShape.LORENTZIAN,
    background: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """Nuisance columns for :func:`line_evidence_escape`.

    The span of the error modes the sigma_eff fidelity currency tolerates,
    as complex regressors on ``offset_grid_mhz``:

    - the frozen contributor ``background`` and its frequency derivative
      (skirt amplitude / position extrapolation error);
    - each established peak's own component plus its frequency and tau
      derivatives (lineshape error under a bright line -- the modes a
      high-SNR absorber candidate feeds on).

    ``peaks`` are the established :class:`~.peak_model.ModelPeak` entries
    of the model *without* the disputed peak. Derivatives are central
    finite differences (a quarter grid-bin in frequency, 5% in tau).

    Composed from :func:`line_escape_background_columns` (full-grid, contains
    the gradient) and :func:`line_escape_peak_columns` (pointwise, safe on a
    subgrid) -- the column ORDER is background, background-gradient, then
    per-peak comp/d_f/d_tau.
    """
    u = np.asarray(offset_grid_mhz, dtype=float)
    bg_cols = line_escape_background_columns(u, background)
    pk_cols = line_escape_peak_columns(u, peaks, tau_us, acquisition_us, shape=shape)
    return bg_cols + pk_cols


def line_escape_support_slice(
    template: np.ndarray,
    *,
    support_fraction: float = 0.15,
    support_dilate: int = 2,
) -> Optional[slice]:
    """Support slice for :func:`line_evidence_escape`.

    Returns the contiguous ``[lo, hi)`` slice covering all bins where
    ``|template|`` is at least ``support_fraction`` of its maximum, dilated
    by ``support_dilate`` bins on each side. Returns ``None`` when the
    template is empty/zero or the resulting slice spans fewer than 3 bins.
    """
    tpl = np.asarray(template, dtype=np.complex128)
    amax = float(np.abs(tpl).max()) if tpl.size else 0.0
    if amax <= 0.0 or tpl.size < 3:
        return None
    sup = np.abs(tpl) >= support_fraction * amax
    idx = np.where(sup)[0]
    lo = max(int(idx.min()) - support_dilate, 0)
    hi = min(int(idx.max()) + support_dilate + 1, tpl.size)
    if hi - lo < 3:
        return None
    return slice(lo, hi)


def line_evidence_escape(
    evidence: np.ndarray,
    rms_noise: NoiseLike,
    template: np.ndarray,
    nuisance_columns: List[np.ndarray],
    *,
    n_params_peak: int,
    penalty_lambda: Optional[float] = None,
    support_fraction: float = 0.15,
    support_dilate: int = 2,
) -> Tuple[bool, float]:
    """Matched-filter line-evidence test for a disputed peak (raw currency).

    On the peak's support neighborhood (bins where ``|template|`` is at
    least ``support_fraction`` of its maximum, dilated by ``support_dilate``
    bins), fit the disputed ``evidence`` (complex residual of the model
    WITHOUT the peak) by weighted complex least squares twice -- nuisance
    columns only, then nuisance plus the peak's ``template`` -- with raw
    per-bin noise weights (stacked Re/Im convention). The difference

        delta_chi2_line = chi2(nuisance) - chi2(nuisance + template)

    is the template's marginal evidence beyond every tolerated error mode
    (see :func:`line_escape_nuisance_columns`). Returns
    ``(fires, delta_chi2_line)`` with ``fires = delta_chi2_line >
    2 * penalty_lambda * n_params_peak``;
    ``penalty_lambda`` defaults to :data:`DEFAULT_GATE_LINE_ESCAPE_LAMBDA`
    and ``(False, 0.0)`` is returned when that is ``None`` (disabled).

    Callers may pre-restrict all arrays to the template's support slice
    (via :func:`line_escape_support_slice`) before calling: passing
    already-sliced arrays is byte-identical because re-deriving the support
    on a pre-sliced template yields the identity slice (dilation re-clips to
    array bounds), so the same ``[lo, hi)`` window is selected.
    """
    lam = DEFAULT_GATE_LINE_ESCAPE_LAMBDA if penalty_lambda is None else penalty_lambda
    if lam is None:
        return False, 0.0
    e = np.asarray(evidence, dtype=np.complex128)
    tpl = np.asarray(template, dtype=np.complex128)
    sigma = np.asarray(rms_noise, dtype=float)
    if sigma.ndim == 0:
        sigma = np.full(e.shape, float(sigma))
    amax = float(np.abs(tpl).max()) if tpl.size else 0.0
    if amax <= 0.0 or e.size < 3:
        return False, 0.0
    sup = np.abs(tpl) >= support_fraction * amax
    idx = np.where(sup)[0]
    lo = max(int(idx.min()) - support_dilate, 0)
    hi = min(int(idx.max()) + support_dilate + 1, e.size)
    sl = slice(lo, hi)
    m = hi - lo
    if m < 3:
        return False, 0.0

    w = np.sqrt(2.0) / sigma[sl]
    b = e[sl] * w
    t_col = tpl[sl] * w

    cols = []
    for c in nuisance_columns:
        seg = np.asarray(c, dtype=np.complex128)[sl] * w
        n2 = float(np.sum(np.abs(seg) ** 2))
        if n2 > 4.0:  # below ~2 sigma aggregate the column cannot matter
            cols.append((n2, seg))
    # Keep the design overdetermined: at most m-2 nuisance columns (each
    # complex column spends 2 real parameters of the 2m real observations),
    # strongest support-norm first.
    cols.sort(key=lambda t: -t[0])
    kept = [seg for _, seg in cols[: max(m - 2, 0)]]

    def _chi2(design: List[np.ndarray]) -> float:
        if not design:
            return float(np.sum(np.abs(b) ** 2))
        A = np.stack(design, axis=1)
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        r = b - A @ coef
        return float(np.sum(np.abs(r) ** 2))

    chi2_nui = _chi2(kept)
    chi2_full = _chi2(kept + [t_col])
    delta = chi2_nui - chi2_full
    fires = delta > 2.0 * float(lam) * float(max(n_params_peak, 1))
    return fires, float(delta)


def pair_cancellation_fraction(
    amplitude_a: float,
    phase_a: float,
    amplitude_b: float,
    phase_b: float,
) -> float:
    """Destructive-interference fraction of a peak pair's phasor sum.

    ``1 - |A_a e^{i phi_a} + A_b e^{i phi_b}| / (A_a + A_b)`` -- 0 for
    perfectly constructive members (a physical unresolved blend sharing the
    molecular phase field), 1 for the cancelling near-duplicate pathology
    (two large opposite-phase amplitudes synthesizing structure no pair of
    real lines produces).
    """
    a = abs(float(amplitude_a))
    b = abs(float(amplitude_b))
    total = a + b
    if total <= 0.0:
        return 0.0
    phasor = a * np.exp(1j * float(phase_a)) + b * np.exp(1j * float(phase_b))
    return float(1.0 - np.abs(phasor) / total)


def blend_pair_escape(
    delta_chi2_raw: float,
    n_params_delta: int,
    amplitude_a: float,
    phase_a: float,
    amplitude_b: float,
    phase_b: float,
    *,
    evidence_floor: float = 1.0,
    feature_evidence: Optional[float] = None,
) -> bool:
    """Whether a sub-separation pair earns exemption from the collapse layers.

    ``delta_chi2_raw`` is the raw chi-squared cost of collapsing the pair
    (merged/simpler fit minus pair fit; positive when the pair is better).
    The pair is kept when that evidence clears the overwhelming-evidence bar
    ``2 * DEFAULT_GATE_LINE_ESCAPE_LAMBDA * n_params_delta * evidence_floor``
    AND the pair is constructive (:func:`pair_cancellation_fraction` below
    :data:`DEFAULT_PAIR_CANCELLATION_MAX`). Either constant set to ``None``
    disables the escape.

    ``evidence_floor`` scales the bar by the model-fidelity level (pass
    ``max(1, reduced_chi2)``, the same floor the penalized gate uses) at
    call sites whose target pathology is the *shape-error absorber* rather
    than the cancelling pair: an absorber's chi-squared win is bounded by
    the lineshape-fidelity floor it soaks, so demanding evidence far above
    that floor keeps high-SNR absorbers collapsed while a genuine blend
    (whose win is reducible structure the single-line model cannot
    represent at any fidelity) still escapes.

    ``feature_evidence`` (the raw chi-squared the whole feature explains
    against the model without it) opts the call site into the
    relative-evidence lane: a constructive pair whose ``delta_chi2_raw``
    clears :data:`DEFAULT_BLEND_RELATIVE_EVIDENCE_FRACTION` of the feature's
    own evidence -- and the plain accept-gate bar
    ``2 * DEFAULT_GATE_PENALTY_LAMBDA * n_params_delta`` -- also escapes,
    even below the absolute bar. A low-SNR doublet's whole feature carries
    less evidence than the absolute bar, so without this lane no low-SNR
    blend can ever be kept (363 w87).
    """
    lam = DEFAULT_GATE_LINE_ESCAPE_LAMBDA
    cmax = DEFAULT_PAIR_CANCELLATION_MAX
    if lam is None or cmax is None:
        return False
    bar = (
        2.0
        * float(lam)
        * float(max(n_params_delta, 1))
        * float(max(evidence_floor, 1.0))
    )
    clears = delta_chi2_raw > bar
    if not clears:
        frac = DEFAULT_BLEND_RELATIVE_EVIDENCE_FRACTION
        lam_gate = DEFAULT_GATE_PENALTY_LAMBDA
        if (
            frac is not None
            and lam_gate is not None
            and feature_evidence is not None
            and feature_evidence > 0.0
        ):
            gate_bar = 2.0 * float(lam_gate) * float(max(n_params_delta, 1))
            clears = (
                delta_chi2_raw > float(frac) * float(feature_evidence)
                and delta_chi2_raw > gate_bar
            )
    if not clears:
        return False
    return pair_cancellation_fraction(amplitude_a, phase_a, amplitude_b, phase_b) < cmax


_fringe_dump_counter = itertools.count()
_fringe_window_ctx: Optional[dict[str, float]] = None


def debug_fringe_dump(site: str, **arrays: Any) -> None:
    """``FTMW_DEBUG_FRINGE_DIR=<dir>``: dump disputed-evidence arrays at a
    gate decision point to ``<dir>/<site>_<n>.npz`` for offline analysis.
    ``FTMW_DEBUG_FRINGE_MIN`` overrides the minimum window-level raw
    delta-chi2 an event must carry to be written (default 50)."""
    out_dir = os.environ.get("FTMW_DEBUG_FRINGE_DIR")
    if not out_dir:
        return
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    n = next(_fringe_dump_counter)
    payload = {k: np.asarray(v) for k, v in arrays.items() if v is not None}
    try:
        evidence = float(payload["raw_chi2_less"]) - float(payload["raw_chi2_more"])
        if evidence < float(os.environ.get("FTMW_DEBUG_FRINGE_MIN", "50")):
            return
    except KeyError:
        pass
    if _fringe_window_ctx is not None:
        for k, v in _fringe_window_ctx.items():
            payload[f"ctx_{k}"] = np.asarray(v)
    np.savez(path / f"{site}_{n:05d}.npz", **payload)


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------
def calculate_aic(chi2: float, n_params: int, n_data: int) -> float:
    """Akaike information criterion ``2k + n * ln(chi-squared / n)``.

    Parameters
    ----------
    chi2 : float
        Noise-weighted chi-squared of the fit.
    n_params : int
        Number of fitted parameters ``k``.
    n_data : int
        Number of (real) data points ``n``.

    Returns
    -------
    float
        The AIC; ``inf`` for a degenerate (non-positive ``chi2``) fit. Lower
        is better.
    """
    if chi2 <= 0.0 or n_data <= 0:
        return float("inf")
    return 2 * n_params + n_data * float(np.log(chi2 / n_data))


def gate_information_weights(
    model_spectrum: np.ndarray,
    *,
    kind: str = DEFAULT_N_EFF_KIND,
    cutoff_fraction: float = 0.1,
    sigma: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-bin weight vector ``w_f`` shared by ``n_eff`` and the weighted chi².

    This is the single source of the bin-weighting the Stage 5 gate uses on
    *both* sides of the AICc ledger: :func:`effective_sample_size` reduces this
    vector to a scalar ``n_eff`` (its perplexity / Kish form), and
    :func:`information_weighted_chi2` weights the per-bin squared residual by the
    same vector so ``chi2_w`` and ``n_eff`` are computed from one weight
    distribution. Factoring it out keeps the two in lock-step (the
    context-invariant-gate fix relies on the weights matching exactly).

    The weight family follows ``kind`` (see :func:`effective_sample_size` for
    the rationale of each):

    - ``"perplexity_log1p_snr"`` -> ``w_f = log1p(|model_f| / sigma_f)`` (needs
      ``sigma``),
    - ``"kish_mag_sq"`` -> ``w_f = |model_f|**2``,
    - ``"kish_mag"`` -> ``w_f = |model_f|``,
    - ``"hard_radius"`` -> ``w_f = 1`` where ``|model_f| > cutoff_fraction *
      max|model|`` else ``0``.

    Parameters
    ----------
    model_spectrum : np.ndarray
        Complex (or real) model spectrum on the window grid. Only the magnitude
        is consulted.
    kind : str, default :data:`DEFAULT_N_EFF_KIND`
        Weighting scheme (see above).
    cutoff_fraction : float, default 0.1
        ``"hard_radius"`` active-region fraction; ignored by other kinds.
    sigma : np.ndarray, optional
        Per-bin complex noise RMS. Required for ``"perplexity_log1p_snr"``.

    Returns
    -------
    np.ndarray
        The non-negative per-bin weight vector, same length as the model.

    Raises
    ------
    ValueError
        If ``kind`` is unknown, or ``"perplexity_log1p_snr"`` is selected
        without ``sigma``.
    """
    mag = np.abs(np.asarray(model_spectrum)).astype(float)
    n_data = int(mag.size)
    if n_data == 0:
        return np.zeros(0, dtype=float)
    if kind == "kish_mag_sq":
        return cast(np.ndarray, np.asarray(mag**2, dtype=float))
    if kind == "kish_mag":
        return cast(np.ndarray, np.asarray(mag, dtype=float))
    if kind == "hard_radius":
        max_mag = float(mag.max())
        if max_mag <= 0.0:
            return np.zeros(n_data, dtype=float)
        return cast(
            np.ndarray, np.asarray(mag > cutoff_fraction * max_mag, dtype=float)
        )
    if kind == "perplexity_log1p_snr":
        if sigma is None:
            raise ValueError(
                "kind='perplexity_log1p_snr' requires sigma "
                "(per-bin complex noise RMS)"
            )
        sig = np.asarray(sigma, dtype=float)
        if sig.ndim == 0:
            sig = np.full(n_data, float(sig))
        snr = mag / np.maximum(sig, 1e-30)
        return cast(np.ndarray, np.asarray(np.log1p(snr), dtype=float))
    raise ValueError(f"unknown kind {kind!r}")


def information_weighted_chi2(
    residual: np.ndarray,
    rms_noise: NoiseLike,
    weights: np.ndarray,
    n_eff: float,
) -> float:
    """Information-weighted chi-squared for the context-invariant AICc gate.

    ``chi2_w = n_eff * ( sum_f w_f * r_f^2 ) / ( sum_f w_f )``, where
    ``r_f^2 = |residual_f / sigma_f|^2`` is the per-bin unit-variance squared
    residual (real+imag) -- the same quantity the raw noise-weighted chi²
    sums, but here it averages to ~1 (using the *complex* RMS ``sigma_f``, not
    ``sigma_f / sqrt(2)``). For a good fit ``r_f^2 ~ 1`` everywhere, so the
    weighted mean is ~1 and ``chi2_w ~ n_eff`` -- ``chi2_w / n_eff ~ 1``,
    **independent of the window bin count ``M``**.

    Feeding ``chi2_w`` (with the same ``n_eff``) to :func:`calculate_aicc` in
    place of the raw chi-squared makes the gate's log-likelihood term compare
    weighted-mean residuals: a real peak that reduces ``r^2`` in its
    high-``w`` bins drops ``chi2_w`` sharply (accept); a spurious peak that
    only reduces ``r^2`` in low-``w`` noise bins barely moves it (reject).
    Adding empty (``w ~ 0``) bins to widen a window moves neither ``chi2_w``
    nor ``n_eff``, so the gate is window-size invariant.

    Parameters
    ----------
    residual : np.ndarray
        Complex per-bin residual ``data - model`` on the (already spur-masked)
        grid. The zero model -- the null -- is simply ``residual = data``.
    rms_noise : float or np.ndarray
        Per-bin complex noise RMS ``sigma`` aligned with ``residual``.
    weights : np.ndarray
        The per-bin information weights from :func:`gate_information_weights`,
        built from the **more-complex** model so both AICc sides share them,
        aligned with ``residual``.
    n_eff : float
        The effective sample size (:func:`effective_sample_size`) the gate
        already computed; the multiplier that puts ``chi2_w`` on the same scale
        as the raw chi-squared the AICc formula expects.

    Returns
    -------
    float
        ``chi2_w``. When the weights sum to zero (an all-noise model with no
        informative bins) the weighted mean is undefined and this returns
        ``n_eff`` (weighted mean defaulted to 1), a neutral value at which the
        AICc log-likelihood term vanishes.

    Raises
    ------
    ValueError
        If ``rms_noise`` is not strictly positive.
    """
    sigma = np.asarray(rms_noise, dtype=float)
    if np.any(sigma <= 0.0):
        raise ValueError("rms_noise must be strictly positive")
    res = np.asarray(residual)
    sig = sigma
    if sig.ndim == 0:
        sig = np.full(res.shape, float(sigma))
    r2 = (np.real(res) ** 2 + np.imag(res) ** 2) / (sig**2)
    w = np.asarray(weights, dtype=float)
    sum_w = float(w.sum())
    if sum_w <= 0.0:
        return float(n_eff)
    weighted_mean_r2 = float((w * r2).sum() / sum_w)
    return float(n_eff) * weighted_mean_r2


def effective_sample_size(
    model_spectrum: np.ndarray,
    *,
    kind: str = "kish_mag_sq",
    cutoff_fraction: float = 0.1,
    sigma: Optional[np.ndarray] = None,
) -> float:
    """Effective number of bins the model actually informs.

    The raw bin count ``n_data`` is the wrong scale for hypothesis tests that
    distinguish K-peak from (K±1)-peak models on a narrow feature: only the
    handful of bins under the feature carry information about the parameter
    change. The effective sample size collapses a flat spectrum to ``n_data``
    and a delta to ``1``; for a localised feature it returns roughly the
    extent over which the feature is informative. Feeding ``n_eff`` into
    :func:`calculate_aicc` makes the small-sample correction kick in on
    narrow features and naturally rejects spurious K growth.

    Two families of weighting are supported:

    - **Magnitude-concentrated** (``kish_mag_sq``, ``kish_mag``,
      ``hard_radius``). The weight is a function of ``|model|`` alone; on a
      Lorentzian peak the result is roughly the FWHM-in-bins. The structural
      divergence of AICc at the resulting small ``n_eff`` pushes hard in the
      conservative direction (REJECT a merge / preserve a peak) -- helpful for
      a K-vs-(K-1) test in isolation, but it over-rejects real K-vs-(K+1)
      escalations on narrow features.
    - **Information-weighted** (``perplexity_log1p_snr``). The weight is
      ``log(1 + |model|/sigma)`` (per-bin Shannon information of a signal-
      vs-noise detection at that SNR), aggregated as the perplexity
      ``exp(H(p))`` of the normalized weight distribution. It keeps ``n_eff``
      in the AICc-identifiable regime across the realistic K-vs-(K±1)
      transitions and only diverges when the model is genuinely
      under-determined.

    Stage 5 threads a single kind -- :data:`DEFAULT_N_EFF_KIND`
    (``perplexity_log1p_snr``) -- into *every* gate (conservative
    add-one-peak accept, blend-aware escalation, merge cleanup, knockout,
    iterative cleanup), including the merge / knockout K-vs-(K-1) sweeps. This
    is the deliberate, cross-gate choice (GitHub issue #9, resolved): the
    magnitude-concentrated ``kish_*`` alternative at the K-vs-(K-1) sites gave
    a marginally worse survey distribution and added a per-site rationale to
    maintain without supporting evidence, so one information-weighted default
    governs every gate. (The merge K-vs-(K-1) AICc test is itself inert under
    the shipped defaults -- the Tier-2 band is closed, also per #9 -- so its
    kind choice only matters to a caller that re-opens the band.) The
    ``kish_*`` / ``hard_radius`` kinds remain available for callers that select
    them explicitly.

    Parameters
    ----------
    model_spectrum : np.ndarray
        Complex (or real) model spectrum on the window grid. Only the
        magnitude is consulted.
    kind : str, default "kish_mag_sq"
        Weighting scheme. This default applies only to direct callers; every
        Stage 5 gate passes :data:`DEFAULT_N_EFF_KIND`
        (``"perplexity_log1p_snr"``) explicitly. ``"kish_mag_sq"`` uses
        ``w_f = |model(f)|²``
        (Fisher-information density for a Gaussian likelihood). ``"kish_mag"``
        uses ``w_f = |model(f)|`` -- softer concentration. ``"hard_radius"``
        counts bins where ``|model(f)| > cutoff_fraction * max|model|`` --
        threshold-sensitive but simpler to reason about.
        ``"perplexity_log1p_snr"`` uses ``w_f = log1p(|model(f)|/sigma(f))``
        as a per-bin information weight and returns the perplexity of the
        normalized distribution; this requires ``sigma``.
    cutoff_fraction : float, default 0.1
        Fraction of ``max|model|`` used by ``"hard_radius"`` to delimit the
        active region. Ignored by the other kinds.
    sigma : np.ndarray, optional
        Per-bin complex noise RMS. Required for ``"perplexity_log1p_snr"``;
        ignored by the magnitude-only kinds.

    Returns
    -------
    float
        The effective sample size, in ``[1, n_data]``. Returns ``n_data`` for
        an all-zero or all-flat model (the "no concentration" limit).

    Raises
    ------
    ValueError
        If ``kind`` is unknown, or if a kind that needs ``sigma`` is selected
        without it.
    """
    mag = np.abs(np.asarray(model_spectrum))
    n_data = mag.size
    if n_data == 0:
        return 0.0
    max_mag = float(mag.max())
    if max_mag <= 0.0:
        return float(n_data)
    # The per-bin weight vector is shared with the gate's weighted chi-squared
    # (:func:`gate_information_weights`); only the scalar reduction differs by
    # family.
    w = gate_information_weights(
        mag, kind=kind, cutoff_fraction=cutoff_fraction, sigma=sigma
    )
    if kind == "hard_radius":
        n_eff = float(int(w.sum()))
        return n_eff if n_eff > 0.0 else 1.0
    if kind == "perplexity_log1p_snr":
        sum_w = float(w.sum())
        if sum_w <= 0.0:
            return 1.0
        p = w / sum_w
        pp = p[p > 0.0]
        h = float(-np.sum(pp * np.log(pp)))
        n_eff = float(np.exp(h))
        return float(min(max(n_eff, 1.0), float(n_data)))
    # Kish kinds (``kish_mag_sq`` / ``kish_mag``): n_eff = (sum w)^2 / sum w^2.
    sum_w = float(w.sum())
    sum_w2 = float((w * w).sum())
    if sum_w2 <= 0.0:
        return float(n_data)
    n_eff = (sum_w * sum_w) / sum_w2
    return float(min(max(n_eff, 1.0), float(n_data)))


def calculate_aicc(
    chi2: float,
    n_params: int,
    n_eff: float,
) -> float:
    """AIC with a small-sample correction, evaluated on the effective
    sample size ``n_eff``.

    The standard Burnham-Anderson AICc treats ``n`` as one quantity: it
    appears in the Gaussian-MLE variance estimator (``σ̂² = chi²/n``, which
    drives the log-likelihood term) *and* in the small-sample correction
    (``2k(k+1)/(n - k - 1)``). Substituting an effective sample size
    ``n_eff`` (see :func:`effective_sample_size`) for ``n`` uniformly
    keeps the formula self-consistent: smaller ``n_eff`` simultaneously
    rescales how much chi² improvements count for in the log-likelihood
    term AND grows the small-sample correction. Hybridising (``n_data``
    in the log term, ``n_eff`` in the correction) would not correspond to
    any standard AICc derivation.

    The formula:

        AICc = 2k + n_eff * log(chi² / n_eff) + 2k(k+1) / (n_eff - k - 1)

    Reduces to :func:`calculate_aic` (in the ``n_eff`` scaling) as
    ``n_eff → ∞``. When ``n_eff ≤ k + 1`` the model is not identifiable
    at the effective sample size and the function returns ``+inf`` -- this
    is the structural rejection on narrow features the AICc-with-n_eff
    design relies on.

    Parameters
    ----------
    chi2 : float
        Noise-weighted chi-squared of the fit.
    n_params : int
        Number of fitted parameters ``k``.
    n_eff : float
        Effective sample size (:func:`effective_sample_size`).

    Returns
    -------
    float
        The AICc. ``+inf`` for ``chi2 ≤ 0``, ``n_eff ≤ 0``, or
        ``n_eff - k - 1 ≤ 0`` (model not identifiable on this evidence).
        Lower is better.

    Notes
    -----
    Tie semantics matter at decision points: when comparing AICc(K) to
    AICc(K-1) and both diverge to ``+inf`` (neither model identifiable),
    the comparison ``aicc_K-1 > aicc_K`` evaluates to ``False`` in
    Python, so a "reject K if AICc(K-1) > AICc(K)" gate falls through to
    accept the simpler model -- the conservative call.
    """
    if chi2 <= 0.0 or n_eff <= 0.0:
        return float("inf")
    denom = n_eff - float(n_params) - 1.0
    if denom <= 0.0:
        return float("inf")
    log_term = n_eff * float(np.log(chi2 / n_eff))
    correction = 2.0 * n_params * (n_params + 1) / denom
    return 2.0 * n_params + log_term + correction


def gate_aicc_pair(
    n_eff: float,
    *,
    more_n_params: int,
    less_n_params: int,
    more_chi2_raw: float,
    less_chi2_raw: float,
    weighted: bool,
    more_residual: Optional[np.ndarray] = None,
    less_residual: Optional[np.ndarray] = None,
    rms_noise: Optional[NoiseLike] = None,
    weight_model: Optional[np.ndarray] = None,
    n_eff_kind: str = DEFAULT_N_EFF_KIND,
    ref_reduced_chi2: Optional[float] = None,
    budget_extra: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """AICc pair ``(aicc_more, aicc_less)`` for a K vs K±1 accept/merge gate.

    A single switch point for the context-invariant gate. When ``weighted`` is
    ``False`` the pair is the legacy raw-chi-squared AICc -- exactly
    ``(calculate_aicc(more_chi2_raw, ...), calculate_aicc(less_chi2_raw, ...))``.
    When ``True`` both sides are re-scored on the information-weighted
    chi-squared (:func:`information_weighted_chi2`), with the per-bin weights
    built **once** from the more-complex model (``weight_model``) so the K and
    K±1 comparisons share one weight distribution -- the same discipline
    ``n_eff`` already follows.

    The weighted branch -- and the penalized branch's sigma_eff variant
    (:data:`DEFAULT_GATE_SIGMA_EFF_KAPPA`) -- need the per-bin residuals and
    noise. All of ``more_residual`` / ``less_residual`` / ``rms_noise`` /
    ``weight_model`` must be aligned and pre-restricted to the bins the gate
    scores (spur bins already removed); ``less_residual`` for a K→0 null is
    simply the data.

    Parameters
    ----------
    n_eff : float
        Shared effective sample size (:func:`effective_sample_size`).
    more_n_params, less_n_params : int
        Parameter counts of the more- and less-complex model.
    more_chi2_raw, less_chi2_raw : float
        Raw noise-weighted chi-squared of each model (used when
        ``weighted=False`` and as the legacy reference).
    weighted : bool
        Select the information-weighted chi-squared gate.
    more_residual, less_residual : np.ndarray, optional
        Complex per-bin residuals ``data - model`` of each model (required when
        ``weighted=True``).
    rms_noise : float or np.ndarray, optional
        Per-bin complex noise RMS aligned with the residuals (required when
        ``weighted=True``).
    weight_model : np.ndarray, optional
        The more-complex model's complex spectrum on the scored bins; the
        information weights are built from it (required when ``weighted=True``).
    n_eff_kind : str, default :data:`DEFAULT_N_EFF_KIND`
        Weighting kind for the shared weights -- match the kind used for
        ``n_eff``.
    ref_reduced_chi2 : float, optional
        The more-complex model's reduced chi-squared, used (floored at 1) to
        scale the penalized gate's evidence bar by the model-fidelity floor so
        the bar rises with SNR. Shared by both scores. Only consulted by the
        penalized branch (:data:`DEFAULT_GATE_PENALTY_LAMBDA` set); ``None``
        leaves the bar unscaled (floor = 1).
    budget_extra : np.ndarray, optional
        Additional per-bin amplitude budget for the sigma_eff variant
        (``kappa_skirt * |frozen background|``; see
        :data:`DEFAULT_GATE_SIGMA_EFF_KAPPA_SKIRT` and
        :func:`sigma_eff_chi2`). Aligned with the residuals; ignored unless
        the sigma_eff branch is active.

    Returns
    -------
    tuple of float
        ``(aicc_more, aicc_less)``. Compare them with the same REJECT-on-tie
        convention the raw gate used.
    """
    # Window-independent penalized-chi-squared gate takes precedence: a single
    # global ``lambda`` (:data:`DEFAULT_GATE_PENALTY_LAMBDA`) replaces the
    # n_eff-shrunk AICc and the information-weighted variant. ``score = chi2_raw
    # + 2*lambda*k*floor``; the lower score wins, so the more-complex model is
    # accepted iff ``Delta chi2 > 2*lambda*Delta k * floor``. ``Delta chi2`` (the
    # two models share the window's bins) is the local likelihood-ratio
    # statistic -- window-size-independent. The ``floor`` is the model-fidelity
    # level ``max(1, ref_reduced_chi2)``: it is ~1 in the noise-dominated regime
    # (the bar reduces to the textbook ``Delta chi2 > 2*lambda*Delta k``, which
    # recovers dense under-fit windows) and scales up as ``(kappa*SNR)^2`` under
    # a bright line (the chi2_r ~ SNR^2 lineshape-fidelity floor), so the bar
    # rises with SNR and rejects the lineshape-absorber peaks an absolute bar
    # over-accepts at high SNR. This restores the SNR scaling the legacy
    # ``n_eff * Delta chi2 / chi2_K`` benefit carried (chi2_K ~ 2M * chi2_r),
    # without its window-size ``2M/n_eff`` drift. ``ref_reduced_chi2`` is the
    # more-complex model's reduced chi-squared (shared by both scores, like
    # ``n_eff``); ``None`` falls back to floor=1 (the un-scaled penalized bar).
    penalty_lambda = DEFAULT_GATE_PENALTY_LAMBDA
    if penalty_lambda is not None:
        # sigma_eff variant: score both models' chi-squared against the
        # fidelity-inflated per-bin noise (:func:`sigma_eff_chi2`,
        # :data:`DEFAULT_GATE_SIGMA_EFF_KAPPA`) so residual evidence under a
        # bright model component is discounted by the lineshape-fidelity
        # budget while evidence at empty bins keeps full noise weighting.
        # The budget is referenced to the more-complex model
        # (``weight_model``) on both sides -- the shared-basis discipline.
        kappa = DEFAULT_GATE_SIGMA_EFF_KAPPA
        if kappa is not None:
            if (
                more_residual is None
                or less_residual is None
                or rms_noise is None
                or weight_model is None
            ):
                raise ValueError(
                    "sigma_eff gate requires more_residual, less_residual, "
                    "rms_noise, and weight_model"
                )
            more_chi2 = sigma_eff_chi2(
                more_residual, rms_noise, weight_model, kappa, extra=budget_extra
            )
            less_chi2 = sigma_eff_chi2(
                less_residual, rms_noise, weight_model, kappa, extra=budget_extra
            )
        else:
            more_chi2 = float(more_chi2_raw)
            less_chi2 = float(less_chi2_raw)
        floor = (
            max(1.0, float(ref_reduced_chi2))
            if (
                DEFAULT_GATE_FLOOR_SCALING
                and ref_reduced_chi2 is not None
                and np.isfinite(ref_reduced_chi2)
            )
            else 1.0
        )
        pen = 2.0 * float(penalty_lambda) * floor
        return (
            more_chi2 + pen * float(more_n_params),
            less_chi2 + pen * float(less_n_params),
        )
    if not weighted:
        return (
            calculate_aicc(more_chi2_raw, more_n_params, n_eff),
            calculate_aicc(less_chi2_raw, less_n_params, n_eff),
        )
    if (
        more_residual is None
        or less_residual is None
        or rms_noise is None
        or weight_model is None
    ):
        raise ValueError(
            "weighted gate requires more_residual, less_residual, rms_noise, "
            "and weight_model"
        )
    weights = gate_information_weights(
        weight_model, kind=n_eff_kind, sigma=np.asarray(rms_noise, dtype=float)
    )
    chi2_more = information_weighted_chi2(more_residual, rms_noise, weights, n_eff)
    chi2_less = information_weighted_chi2(less_residual, rms_noise, weights, n_eff)
    return (
        calculate_aicc(chi2_more, more_n_params, n_eff),
        calculate_aicc(chi2_less, less_n_params, n_eff),
    )


def calculate_chi_squared_improvement(
    old_chi2: float,
    new_chi2: float,
    dof_change: int,
    n_data: int,
    n_params_new: int,
) -> Tuple[float, float, float]:
    """Nested-model F-test for a chi-squared improvement.

    The more complex model adds ``dof_change`` parameters and lowers the
    chi-squared by ``old_chi2 - new_chi2``. The F-statistic is

        F = (chi2_diff / dof_change) / (new_chi2 / (n_data - n_params_new))

    and the p-value is its upper tail. A non-improvement (or degenerate
    degrees of freedom) returns ``(1.0, 0.0, chi2_diff)``.

    Parameters
    ----------
    old_chi2, new_chi2 : float
        Chi-squared of the simpler and the more complex model.
    dof_change : int
        Number of parameters the complex model adds (``> 0``).
    n_data : int
        Number of (real) data points.
    n_params_new : int
        Parameter count of the complex model.

    Returns
    -------
    tuple of float
        ``(p_value, f_statistic, chi2_diff)``.
    """
    chi2_diff = old_chi2 - new_chi2
    df_residual = n_data - n_params_new
    if dof_change <= 0 or chi2_diff <= 0.0 or df_residual <= 0 or new_chi2 <= 0.0:
        return 1.0, 0.0, chi2_diff
    f_statistic = (chi2_diff / dof_change) / (new_chi2 / df_residual)
    p_value = float(1.0 - f_distribution.cdf(f_statistic, dof_change, df_residual))
    return p_value, float(f_statistic), chi2_diff


def passes_significance_test(
    old_chi2: float,
    new_chi2: float,
    dof_change: int,
    n_data: int,
    n_params_new: int,
    significance: float = 0.05,
) -> bool:
    """Whether a chi-squared improvement clears the F-test significance level.

    Thin wrapper over :func:`calculate_chi_squared_improvement`: ``True`` when
    the F-test p-value is below ``significance``. (Unlike the bcfitting
    reference, this takes the real ``dof_change`` rather than assuming 2.)

    Parameters
    ----------
    old_chi2, new_chi2 : float
        Chi-squared of the simpler and the more complex model.
    dof_change : int
        Parameters added by the complex model.
    n_data : int
        Number of (real) data points.
    n_params_new : int
        Parameter count of the complex model.
    significance : float, default 0.05
        P-value threshold.

    Returns
    -------
    bool
        ``True`` if the improvement is statistically significant.
    """
    p_value, _, _ = calculate_chi_squared_improvement(
        old_chi2, new_chi2, dof_change, n_data, n_params_new
    )
    return p_value < significance


# ---------------------------------------------------------------------------
# SNR-aware acceptance (the lineshape-fidelity floor)
# ---------------------------------------------------------------------------
def shape_error_fraction(
    reduced_chi2: float,
    snr_max: float,
    noise_floor: float = DEFAULT_CHI2R_NOISE_FLOOR,
) -> float:
    """Per-bin fractional model deficit implied by a window's reduced chi-squared.

    Inverts the deficit-dominated regime of the SNR-aware gate: a fractional
    lineshape deficit ``eps`` under a line of peak SNR ``snr_max`` contributes
    ``(eps * snr_max)**2`` to the reduced chi-squared above the noise-regime
    allowance ``F`` (:data:`DEFAULT_CHI2R_NOISE_FLOOR`), so

        eps = sqrt(max(reduced_chi2 - F, 0)) / snr_max .

    Subtracting ``F`` rather than 1 makes ``eps`` a clean deficit estimate: in
    the noise-dominated regime (``reduced_chi2 <= F``) it returns 0 (no
    measurable deficit), and at high SNR the ``F`` offset is negligible against
    ``(eps * snr_max)**2``. On the high-SNR vinyl-cyanide fixtures it reads
    ~0.1% on the bright cores and ~1-2% on the moderate forest -- the honest
    fidelity of the analytic line shape plus the low-order leakage baseline, not
    a defect to drive to zero. It is the deficit-regime readout reported next to
    the pass/fail gate, and is only meaningful where the deficit term clears the
    noise scatter (``snr_max`` of order 100+).

    Parameters
    ----------
    reduced_chi2 : float
        Per-window reduced chi-squared of the fit.
    snr_max : float
        Maximum peak SNR in the window (the brightest line drives the deficit).
    noise_floor : float, default :data:`DEFAULT_CHI2R_NOISE_FLOOR`
        Noise-regime allowance ``F`` subtracted before taking the root.

    Returns
    -------
    float
        The fractional deficit ``eps``; ``0.0`` when ``snr_max <= 0`` (no line
        to resolve a deficit against) or the reduced chi-squared is at/below the
        noise-regime allowance.
    """
    if snr_max <= 0.0:
        return 0.0
    excess = max(float(reduced_chi2) - float(noise_floor), 0.0)
    return float(np.sqrt(excess) / snr_max)


def snr_aware_chi2_pass(
    reduced_chi2: float,
    snr_max: float,
    kappa: float = DEFAULT_SHAPE_ERROR_KAPPA,
    noise_floor: float = DEFAULT_CHI2R_NOISE_FLOOR,
) -> bool:
    """Unified SNR-aware Stage 5 window acceptance gate.

    A window passes iff

        reduced_chi2 <= F + (kappa * snr_max)**2 ,

    where ``kappa`` is the tolerated per-bin fractional model deficit
    (:data:`DEFAULT_SHAPE_ERROR_KAPPA`) and ``F`` is the noise-regime allowance
    (:data:`DEFAULT_CHI2R_NOISE_FLOOR`). The gate has two physical regimes that
    the raw ``reduced_chi2 <= threshold`` gate conflates:

    - **Noise-dominated** (low ``snr_max``): the ``(kappa * snr_max)**2`` term
      is negligible, the gate collapses to ``reduced_chi2 <= F``, and a model
      deficit is invisible below the noise. ``F`` (rather than 1) budgets for
      the sampling scatter of a good fit's reduced chi-squared so a healthy
      noise-dominated window is not rejected for normal upward fluctuation.
    - **Deficit-dominated** (high ``snr_max``): the allowance grows as
      ``snr_max**2``, matching the measured chi-squared_r ~ SNR^2 floor, so a
      bright line fit to its lineshape-fidelity limit passes rather than failing
      purely for being bright; ``F`` is negligible there.

    Parameters
    ----------
    reduced_chi2 : float
        Per-window reduced chi-squared of the fit.
    snr_max : float
        Maximum peak SNR in the window.
    kappa : float, default :data:`DEFAULT_SHAPE_ERROR_KAPPA`
        Tolerated fractional model deficit.
    noise_floor : float, default :data:`DEFAULT_CHI2R_NOISE_FLOOR`
        Noise-regime allowance ``F``.

    Returns
    -------
    bool
        ``True`` if the window is acceptable under the SNR-aware gate.
    """
    if not np.isfinite(reduced_chi2):
        return False
    allowance = float(noise_floor) + (kappa * max(float(snr_max), 0.0)) ** 2
    return bool(float(reduced_chi2) <= allowance)


def validate_peak_separation(
    offsets_mhz: np.ndarray,
    min_separation_mhz: float,
) -> Tuple[bool, List[Tuple[int, int]]]:
    """Check that every pair of peaks is at least ``min_separation`` apart.

    Parameters
    ----------
    offsets_mhz : array-like
        Peak positions (baseband offsets or frequencies, MHz).
    min_separation_mhz : float
        Minimum allowed separation between any two peaks (MHz).

    Returns
    -------
    tuple
        ``(is_valid, too_close_pairs)`` -- ``is_valid`` is ``True`` when no
        pair is closer than ``min_separation``; ``too_close_pairs`` lists the
        offending ``(i, j)`` index pairs.
    """
    offsets = np.asarray(offsets_mhz, dtype=float)
    too_close: List[Tuple[int, int]] = []
    for i in range(offsets.size):
        for j in range(i + 1, offsets.size):
            if abs(offsets[i] - offsets[j]) < min_separation_mhz:
                too_close.append((i, j))
    return len(too_close) == 0, too_close


# ---------------------------------------------------------------------------
# Per-peak quality (determinacy) score
# ---------------------------------------------------------------------------
# Number of independent determinacy checks a fitted line is graded on.
PEAK_QUALITY_MAX = 4


def amplitude_vif(peak: FittedPeak) -> Optional[float]:
    """Diagonal amplitude variance-inflation factor ``(amp_err / amp) * snr``.

    The overfit discriminant: ~1 for an identifiable line, >> 1 when a line is
    degenerate with a sub-resolution neighbor (the pair *sum* is constrained,
    neither amplitude individually). A pure function of already-persisted
    per-peak fields -- no covariance matrix needed. Returns ``None`` when any
    input is missing / non-finite or the amplitude is zero.
    """
    amp = float(peak.amplitude)
    amp_err = peak.amplitude_error
    snr = peak.snr
    if amp_err is None or snr is None:
        return None
    if not (math.isfinite(amp) and math.isfinite(amp_err) and math.isfinite(snr)):
        return None
    if abs(amp) <= 0.0:
        return None
    return abs(amp_err / amp) * float(snr)


def peak_quality_score(
    peak: FittedPeak,
    *,
    peer_freqs_mhz: Sequence[float],
    acquisition_us: float,
    survival_floor: float,
    vif_collapse_threshold: float = 4.0,
) -> int:
    """Grade how well the fit *determines* a line, in ``0..PEAK_QUALITY_MAX``.

    A coarse, prior-free determinacy tier: it counts how many of four
    independent checks the line clearly passes, each reusing a threshold the
    pipeline already trusts so no new tuning is introduced. It says how firmly
    the data pin the line, **not** whether the line is a real, assignable
    transition -- a high score can still attach to an unflagged spur or an
    unassigned feature.

    The four checks:

    #. **Detected with margin** -- ``snr >= 2 * survival_floor`` (well clear of
       the survival floor, not a marginal survivor).
    #. **Amplitude identifiable** -- :func:`amplitude_vif` ``<= 0.5 *
       vif_collapse_threshold`` (the amplitude is individually determined, far
       from the degenerate-collapse bar).
    #. **Position pinned** -- ``frequency_error <= 0.1 / acquisition_us`` (the
       1-sigma frequency error is under a tenth of a resolution element).
    #. **Isolated** -- the nearest other fitted line is ``>= 1 /
       acquisition_us`` away (not inside a sub-resolution blend); a lone line
       passes by default.

    A check whose inputs are missing or non-finite does not count as a pass.
    """
    score = 0

    snr = peak.snr
    if snr is not None and math.isfinite(snr) and snr >= 2.0 * survival_floor:
        score += 1

    vif = amplitude_vif(peak)
    if vif is not None and vif <= 0.5 * vif_collapse_threshold:
        score += 1

    res_element = 1.0 / acquisition_us if acquisition_us > 0.0 else math.inf
    ferr = peak.frequency_error
    if ferr is not None and math.isfinite(ferr) and ferr <= 0.1 * res_element:
        score += 1

    others = [float(f) for f in peer_freqs_mhz if float(f) != float(peak.frequency_mhz)]
    if (
        not others
        or min(abs(float(peak.frequency_mhz) - f) for f in others) >= res_element
    ):
        score += 1

    return score
