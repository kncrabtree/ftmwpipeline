"""
Stage 3 projection-coherence screen.

Given a set of Stage 3 candidates on the active-portion FT, score each one by
how well the data on a localised sub-window around the candidate projects onto
a unit-amplitude finite-T Lorentzian basis at the candidate's frequency. The
output is the *coherent-to-detected SNR ratio* per candidate: ratio ≈ 1 for a
real Lorentzian (the basis matches the data), ratio ≪ 1 for a phase-incoherent
noise excursion (the σ-weighted complex projection cancels across bins).

Background and scope
--------------------
This is the same projection primitive a prior session removed from the Stage 5
residual rescue (commit ``c2f2ab2``). At Stage 5 the AICc accept gate inside
``conservative_fit`` already does the same discrimination, so the screen there
was redundant. Stage 3 has no AICc gate behind it; a candidate is promoted or
not on magnitude / prominence alone. The projection ratio is the right
discriminator for "is this magnitude excursion the rfft of a Lorentzian?", and
that is exactly what a Stage 3 low-SNR screen needs.

This module is an unwired diagnostic for the Stage 3 coherence study (see
``scratch/stage3-coherence-study/plan.md``). It will become wired into
``_internal/stage3_impl.py`` only if the cross-tabulation analysis shows the
ratio cleanly separates ``became_fitted_peak`` outcomes; until that decision
lands it ships as a pure helper with unit tests but no callers in ``src/``.

The math
--------
For a candidate at molecular frequency ``f_c`` with sideband sign ``s``, the
basis on a sub-window's signed baseband-offset grid ``u = s·(f - f_c)`` is
``b(u) = h_T(u; τ_apod, T_active)`` from :mod:`ftmwpipeline.fitting.peak_model`.
The σ-weighted complex projection of the active-FT slice ``z(u)`` onto the
basis is the closed-form least-squares amplitude

    A = ⟨b, z⟩_σ / ⟨b, b⟩_σ
    ⟨x, y⟩_σ = Σ_k conj(x_k) · y_k / σ_c[k]²       with σ_c = σ / √2.

The coherent SNR is ``|A| · |h_T(0)| / σ_c_median`` -- the on-line response of
the projected Lorentzian against the local Rayleigh noise scale. The detected
SNR is ``|z(0)| / σ_c[bin_c]`` (active-FT magnitude / per-bin noise at the
candidate's bin). The *ratio* is what the screen emits.

Choosing the basis ``τ``
------------------------
``τ_basis`` controls the basis FWHM ``1 / (π τ_basis)`` and is *not*
fixed by the helper -- the caller must supply a value matched to the
expected line shape of the experiment. The apodization time constant
``expf_us`` is the **upper bound** on the data's observed
``τ_eff`` (``1/τ_eff = 1/τ_true + 1/τ_apod``, so ``τ_eff ≤ τ_apod``
always), which means a basis at ``τ_basis = τ_apod`` is the
**narrowest** matched basis the data can produce: it matches only when
the molecular ``τ_true ≫ τ_apod`` so the apodization dominates the
linewidth. When molecular decay is comparable to or shorter than the
apodization (the common case for real samples), a basis built with
``τ_basis = τ_apod`` is *narrower* than the actual line shape and the
projection becomes dominated by the on-line bin alone, which is what
removes the screen's discrimination power.

Practical guidance: for production use, take ``τ_basis`` from a
data-derived estimate -- e.g. a per-experiment τ_eff fitted on the
brightest detected lines, or an explicit override -- not from
``expf_us`` blindly. The research project at
``dev-docs/research/stage3-coherence-screen/`` characterises the
screen's behaviour against linewidth-in-bin-units; consult its
report for how sensitive the screen is to ``τ_basis`` being off by
a factor of 2.

Why a localised projection window
---------------------------------
The active-FT is the full-spectrum complex spectrum; running the projection on
the whole range would let any far-away strong line's coherent skirt
contaminate the numerator for every candidate. Limiting the projection to a
small sub-window around the candidate (``window_fwhm_factor * FWHM`` per side)
keeps the projection local. Beyond ~5 FWHM the basis magnitude is essentially
gone, so the projection contribution drops to noise anyway -- but truncating
explicitly makes the contamination from strong lines elsewhere a non-issue.

This module deliberately omits the residual-screen's sliding threshold and
cluster-deferral logic. The intent here is to emit a *raw* ratio per
candidate; the discriminator analysis decides where (or whether) to threshold.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np

from ftmwpipeline.core.data_structures import Sideband
from ftmwpipeline.fitting.peak_model import h_T, sideband_sign

SidebandLike = Union[Sideband, str]

__all__ = [
    "ProjectionResult",
    "project_candidates",
]


@dataclass(frozen=True)
class ProjectionResult:
    """Per-candidate output of :func:`project_candidates`.

    Attributes
    ----------
    candidate_freq_mhz : float
        Input candidate frequency (MHz, molecular).
    active_bin : int
        Index into ``active_freq_mhz`` of the candidate's nearest active-FT
        bin (the bin used as the on-line reference).
    coherent_amp : float
        ``|A|`` from the σ-weighted least-squares projection -- the
        best-fit complex amplitude of a unit Lorentzian at the candidate.
    coherent_snr : float
        ``|A| · |h_T(0)| / σ_c_median`` on the projection sub-window: the
        SNR-equivalent on-line response of the projected Lorentzian.
    detected_snr_active : float
        ``|z(f_c)| / σ_c[bin_c]`` -- the active-FT magnitude SNR at the
        candidate's bin (the per-bin reference the ratio normalises against).
    ratio : float
        ``coherent_snr / detected_snr_active``. ≈ 1 for a real Lorentzian;
        ≪ 1 for a phase-incoherent excursion.
    projection_n_bins : int
        Number of active-FT bins inside the projection sub-window.
    window_half_width_mhz : float
        Half-width of the projection sub-window (= ``window_fwhm_factor *
        FWHM_mhz``).
    fwhm_mhz : float
        Effective Lorentzian FWHM (``1 / (π · tau_us)``) used for the
        sub-window width.
    """

    candidate_freq_mhz: float
    active_bin: int
    coherent_amp: float
    coherent_snr: float
    detected_snr_active: float
    ratio: float
    projection_n_bins: int
    window_half_width_mhz: float
    fwhm_mhz: float


def _nearest_bin(sorted_freq: np.ndarray, value: float) -> int:
    """Index of the entry in ``sorted_freq`` closest to ``value``.

    ``sorted_freq`` must be strictly ascending. The caller is expected to have
    sorted the active-FT grid (it is ascending baseband, which maps to a
    monotone molecular axis -- ascending for the upper sideband, descending
    for the lower; the descending case is handled by the caller passing a
    pre-sorted ascending view and translating the index back if needed).
    """
    pos = int(np.searchsorted(sorted_freq, value))
    if pos == 0:
        return 0
    if pos >= sorted_freq.size:
        return int(sorted_freq.size - 1)
    if abs(value - sorted_freq[pos - 1]) <= abs(value - sorted_freq[pos]):
        return pos - 1
    return pos


def project_candidates(
    active_freq_mhz: np.ndarray,
    active_spectrum: np.ndarray,
    active_sigma: np.ndarray,
    candidate_freqs_mhz: Sequence[float],
    *,
    tau_us: float,
    acquisition_us: float,
    sideband: SidebandLike,
    window_fwhm_factor: float = 5.0,
    min_window_bins: int = 3,
) -> List[ProjectionResult]:
    """Score each candidate by its σ-weighted Lorentzian projection ratio.

    Parameters
    ----------
    active_freq_mhz : np.ndarray
        Molecular frequency axis of the active-FT (MHz), 1-D. May be
        ascending or descending -- the function sorts internally.
    active_spectrum : np.ndarray
        Complex active-FT on ``active_freq_mhz`` in the natural
        ``dt_us * rfft(active * apod)`` convention (see
        :mod:`ftmwpipeline.fitting.active_ft`). Same shape as
        ``active_freq_mhz``.
    active_sigma : np.ndarray
        Per-bin |X| RMS noise on the same grid (output of Stage 2's
        adaptive noise estimator applied to the active-FT magnitude).
    candidate_freqs_mhz : sequence of float
        Candidate frequencies (MHz, molecular).
    tau_us : float
        Lorentzian basis decay constant (microseconds). Set this to a
        data-derived estimate of the experiment's observed ``τ_eff``
        (e.g. the fitted τ on the brightest lines). ``expf_us`` is the
        *upper bound* on the observed ``τ_eff`` and corresponds to the
        *narrowest* possible matched basis -- it is the correct choice
        only when the molecular ``τ_true`` is much longer than
        ``expf_us``. See the module docstring's "Choosing the basis τ"
        section.
    acquisition_us : float
        Active acquisition length ``T`` (microseconds).
    sideband : Sideband or str
        Sideband convention (``"lower"`` / ``"upper"`` or the enum).
        Used to translate the molecular grid to the signed baseband
        offset the basis is parameterised in.
    window_fwhm_factor : float, default 5.0
        Half-width of the projection sub-window in units of the
        Lorentzian FWHM. 5 FWHM is the point at which the basis
        magnitude has decayed to a few percent of the on-line value --
        beyond it the projection adds only noise to the numerator. The
        residual-screen reference used the same number as the boundary
        between "neighbour contaminating" and "isolated" candidates.
    min_window_bins : int, default 3
        Floor on the sub-window bin count. For instruments / FT settings
        where ``FWHM`` ≲ 2 bins the FWHM-scaled half-width may pick only
        one bin per side; enforce at least ``min_window_bins`` total bins
        so the projection has degrees of freedom beyond the on-line bin.

    Returns
    -------
    list of ProjectionResult
        One entry per candidate, in input order.

    Raises
    ------
    ValueError
        If ``tau_us`` or ``acquisition_us`` is not positive, if the input
        shapes disagree, or if any of ``active_freq_mhz``,
        ``active_spectrum``, ``active_sigma`` is empty.
    """
    if tau_us <= 0.0:
        raise ValueError("tau_us must be positive")
    if acquisition_us <= 0.0:
        raise ValueError("acquisition_us must be positive")

    freq = np.asarray(active_freq_mhz, dtype=float)
    spec = np.asarray(active_spectrum, dtype=np.complex128)
    sigma = np.asarray(active_sigma, dtype=float)
    if freq.ndim != 1:
        raise ValueError("active_freq_mhz must be 1-D")
    if spec.shape != freq.shape or sigma.shape != freq.shape:
        raise ValueError(
            "active_spectrum and active_sigma must have the same shape as "
            "active_freq_mhz"
        )
    if freq.size == 0:
        raise ValueError("active spectrum must be non-empty")

    s = sideband_sign(sideband)

    # Sort the grid ascending in molecular frequency. The active-FT is
    # ascending baseband; for the lower sideband that maps to descending
    # molecular frequency. The projection is grid-orientation-independent
    # (it sums over bins inside a frequency band) but the sub-window slice
    # is much easier on a monotone grid.
    sort_idx = np.argsort(freq)
    freq_s = freq[sort_idx]
    spec_s = spec[sort_idx]
    sigma_s = sigma[sort_idx]
    inv_sort = np.argsort(sort_idx)  # to translate sorted bin → input bin

    # Per-component noise scale: |X| is Rayleigh with scale σ_c = σ / √2.
    sigma_c = sigma_s / np.sqrt(2.0)
    safe_sigma_c = np.where(sigma_c > 0.0, sigma_c, 1.0)
    weights = 1.0 / safe_sigma_c**2

    fwhm_mhz = 1.0 / (np.pi * tau_us)  # Lorentzian FWHM in MHz
    half_width_mhz = float(window_fwhm_factor) * fwhm_mhz
    h_T_on_line = float(abs(h_T(np.array([0.0]), tau_us, acquisition_us)[0]))

    results: List[ProjectionResult] = []
    for f_c in candidate_freqs_mhz:
        f_c = float(f_c)
        c_sorted = _nearest_bin(freq_s, f_c)
        c_input = int(inv_sort[c_sorted])

        # Sub-window: |f - f_c| <= half_width_mhz, with a floor of
        # min_window_bins to keep the projection well-conditioned.
        lo_freq = f_c - half_width_mhz
        hi_freq = f_c + half_width_mhz
        lo = int(np.searchsorted(freq_s, lo_freq, side="left"))
        hi = int(np.searchsorted(freq_s, hi_freq, side="right"))
        if hi - lo < min_window_bins:
            half_extra = max(min_window_bins // 2, 1)
            lo = max(0, c_sorted - half_extra)
            hi = min(freq_s.size, c_sorted + half_extra + 1)
        if hi <= lo:
            # Degenerate -- one-bin slice. Fall back to a trivial projection.
            results.append(
                ProjectionResult(
                    candidate_freq_mhz=f_c,
                    active_bin=c_input,
                    coherent_amp=0.0,
                    coherent_snr=0.0,
                    detected_snr_active=float(
                        abs(spec_s[c_sorted]) / safe_sigma_c[c_sorted]
                    ),
                    ratio=0.0,
                    projection_n_bins=0,
                    window_half_width_mhz=half_width_mhz,
                    fwhm_mhz=fwhm_mhz,
                )
            )
            continue

        u_slice = s * (freq_s[lo:hi] - f_c)  # signed baseband offset (MHz)
        z_slice = spec_s[lo:hi]
        w_slice = weights[lo:hi]
        sigma_c_slice = sigma_c[lo:hi]

        basis = h_T(u_slice, tau_us, acquisition_us)
        denom = float(np.sum((np.abs(basis) ** 2) * w_slice))
        if denom <= 0.0:
            results.append(
                ProjectionResult(
                    candidate_freq_mhz=f_c,
                    active_bin=c_input,
                    coherent_amp=0.0,
                    coherent_snr=0.0,
                    detected_snr_active=float(
                        abs(spec_s[c_sorted]) / safe_sigma_c[c_sorted]
                    ),
                    ratio=0.0,
                    projection_n_bins=int(hi - lo),
                    window_half_width_mhz=half_width_mhz,
                    fwhm_mhz=fwhm_mhz,
                )
            )
            continue

        numer = np.sum(np.conj(basis) * z_slice * w_slice)
        amp = numer / denom
        coherent_amp = float(abs(amp))

        # Median σ_c over the projection sub-window -- the Rayleigh scale
        # the coherent SNR normalises against. Robust to the slice's
        # bin-count being small (mean would be more sensitive to a single
        # noisy bin).
        finite_sigma_c = sigma_c_slice[sigma_c_slice > 0.0]
        if finite_sigma_c.size == 0:
            results.append(
                ProjectionResult(
                    candidate_freq_mhz=f_c,
                    active_bin=c_input,
                    coherent_amp=coherent_amp,
                    coherent_snr=0.0,
                    detected_snr_active=0.0,
                    ratio=0.0,
                    projection_n_bins=int(hi - lo),
                    window_half_width_mhz=half_width_mhz,
                    fwhm_mhz=fwhm_mhz,
                )
            )
            continue
        median_sigma_c = float(np.median(finite_sigma_c))
        coherent_snr = (coherent_amp * h_T_on_line) / median_sigma_c
        detected_snr_active = float(
            abs(spec_s[c_sorted]) / safe_sigma_c[c_sorted]
        )
        ratio = (
            coherent_snr / detected_snr_active
            if detected_snr_active > 0.0
            else 0.0
        )

        results.append(
            ProjectionResult(
                candidate_freq_mhz=f_c,
                active_bin=c_input,
                coherent_amp=coherent_amp,
                coherent_snr=coherent_snr,
                detected_snr_active=detected_snr_active,
                ratio=ratio,
                projection_n_bins=int(hi - lo),
                window_half_width_mhz=half_width_mhz,
                fwhm_mhz=fwhm_mhz,
            )
        )

    return results
