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

from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.stats import f as f_distribution

from .peak_model import PeakShape, h_T_shape

__all__ = [
    "DEFAULT_N_EFF_KIND",
    "calculate_hwhm_from_apodization",
    "feature_fwhm",
    "calculate_rms_residuals",
    "calculate_noise_weighted_chi2",
    "calculate_aic",
    "calculate_aicc",
    "calculate_chi_squared_improvement",
    "effective_sample_size",
    "passes_significance_test",
    "validate_peak_separation",
]

NoiseLike = Union[float, np.ndarray]

# Effective-sample-size weighting kind shared by every Stage 5 AICc gate
# (conservative add-one-peak accept, blend-aware K=2/K=3 escalation, merge
# cleanup, knockout, iterative cleanup). Each bin is weighted by
# ``log(1 + |model|/sigma)`` -- the per-bin Shannon information of a signal-
# vs-noise detection -- and ``n_eff`` is the perplexity ``exp(H(p))`` of the
# normalised distribution. On a Lorentzian peak with peak SNR ~ 100 this
# returns ~50 bins (the bins where the skirt is significant) rather than
# the ~5 FWHM-in-bins a magnitude-concentrated weight gives. The gate then
# stays in the AICc-identifiable regime for the realistic K-vs-(K+/-1)
# transitions Stage 5 makes and only diverges to ``+inf`` when the model is
# genuinely under-determined; in the divergent case the REJECT-on-tie at
# each gate falls through to "preserve the simpler model" (do not add /
# do not merge / do not drop the peak), which is the conservative direction.
DEFAULT_N_EFF_KIND = "perplexity_log1p_snr"


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
      Lorentzian peak the result is roughly the FWHM-in-bins. Right for
      K-vs-(K-1) tests (merge / knockout), where the question is "are these
      K peaks each individually informative" and the structural divergence
      of AICc at small ``n_eff`` *helps* the conservative direction (REJECT
      a merge / preserve a peak).
    - **Information-weighted** (``perplexity_log1p_snr``). The weight is
      ``log(1 + |model|/sigma)`` (per-bin Shannon information of a signal-
      vs-noise detection at that SNR), aggregated as the perplexity
      ``exp(H(p))`` of the normalised weight distribution. Right for
      K-vs-(K+1) tests (the conservative add-one-peak loop), where the
      same divergence over-rejects real escalations on narrow features.

    Parameters
    ----------
    model_spectrum : np.ndarray
        Complex (or real) model spectrum on the window grid. Only the
        magnitude is consulted.
    kind : str, default "kish_mag_sq"
        Weighting scheme. ``"kish_mag_sq"`` uses ``w_f = |model(f)|²``
        (Fisher-information density for a Gaussian likelihood). ``"kish_mag"``
        uses ``w_f = |model(f)|`` -- softer concentration. ``"hard_radius"``
        counts bins where ``|model(f)| > cutoff_fraction * max|model|`` --
        threshold-sensitive but simpler to reason about.
        ``"perplexity_log1p_snr"`` uses ``w_f = log1p(|model(f)|/sigma(f))``
        as a per-bin information weight and returns the perplexity of the
        normalised distribution; this requires ``sigma``.
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
    if kind == "kish_mag_sq":
        w = mag.astype(float) ** 2
    elif kind == "kish_mag":
        w = mag.astype(float)
    elif kind == "hard_radius":
        active = mag > cutoff_fraction * max_mag
        n_eff = float(int(active.sum()))
        return n_eff if n_eff > 0.0 else 1.0
    elif kind == "perplexity_log1p_snr":
        if sigma is None:
            raise ValueError(
                "kind='perplexity_log1p_snr' requires sigma "
                "(per-bin complex noise RMS)"
            )
        sig = np.asarray(sigma, dtype=float)
        if sig.ndim == 0:
            sig = np.full(n_data, float(sig))
        snr = mag / np.maximum(sig, 1e-30)
        w = np.log1p(snr)
        sum_w = float(w.sum())
        if sum_w <= 0.0:
            return 1.0
        p = w / sum_w
        pp = p[p > 0.0]
        h = float(-np.sum(pp * np.log(pp)))
        n_eff = float(np.exp(h))
        return float(min(max(n_eff, 1.0), float(n_data)))
    else:
        raise ValueError(f"unknown kind {kind!r}")
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
