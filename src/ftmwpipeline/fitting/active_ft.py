"""
Active-portion FT for Stage 5 fitting.

The persisted Stage 1 spectrum is an rfft of the *whole* zero-padded FID
record (``zpf=2``, ``N_active << N_padded`` for 2638). Adjacent bins are
correlated by a Dirichlet kernel (the FFT of the zero-padding indicator), so
the effective number of independent samples in any band of ``M`` bins is
``M * alpha`` with ``alpha = N_active / N_padded`` (~0.42 for 2638). The naive
``N_dof = M - N_params`` overcounts by ``1/alpha``; reduced chi-squared, the
F-test, and AIC are all biased optimistic on the persisted spectrum.

Stage 5 dissolves the problem at source by fitting the **active-portion FT** --
the rfft of just the ``fid[t0 : t0+T]`` active samples with the canonical
Stage 1 apodization, no zero-padding. The result has

* independent bins (no Dirichlet correlation; ``alpha = 1`` by construction),
* no phase ramp (the active-FT is in the ``[0, T]`` form ``h_T`` models),
* the same molecular frequency axis convention as the persisted FT (so
  windows defined as frequency ranges translate directly), with a coarser
  bin spacing ``1/T_active`` instead of ``1/T_padded``.

This module owns the construction (:func:`compute_active_ft`). The active-FT
is internal to Stage 5: computed on demand from ``stage0_fid_data`` and the
canonical Stage 1 settings, not persisted in the ``.ftmw`` file.

Amplitude convention
--------------------
The complex spectrum is ``dt_us * rfft(active * apod)``, which is the
``[0, T]``-frame discrete approximation of the continuous Fourier transform
in MHz / microsecond units. A line of true amplitude ``A``, phase ``phi``, and
decay ``tau`` has on-line response ``0.5 * A * exp(i*phi) * h_T(0; tau, T)``,
i.e. exactly the form :mod:`ftmwpipeline.fitting.peak_model` models. The
synthetic-spectrum builder the Stage 5 prototype tests use
(``test_plan_execution.py: _synth_spectrum``) builds spectra in this same
convention, so unit tests can construct :class:`ActiveFTResult` directly.

Noise convention
----------------
Per-bin noise on the active-FT is measured directly by running the existing
Stage 2 adaptive noise estimator
(:func:`ftmwpipeline.preprocessing.noise_estimation.estimate_noise_adaptive`)
on the active-FT spectrum -- the *same* algorithm Stage 2 uses on the
persisted spectrum, just applied to the active-FT instead. No conversion
factor, no ``1/sqrt(alpha)`` rescale: the noise estimate comes from the same
spectrum the fit sees, so any normalization choices cancel by construction.

References
----------
* ``dev-docs/planning/stage5-fitting.md`` § "Spectral domain for the fit"
* ``dev-docs/ROADMAP.md`` divergence D9
* ``dev-docs/research/stage5-fitting/report.md`` § 3 "Calibration scope"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from ftmwpipeline.core.data_structures import Sideband

from .peak_model import sideband_sign

__all__ = [
    "ActiveFTResult",
    "compute_active_ft",
]

SidebandLike = Union[Sideband, str]


@dataclass
class ActiveFTResult:
    """The active-portion FT of an FID, in the Stage 5 fit frame.

    Attributes
    ----------
    freq_mhz : np.ndarray
        Molecular frequency grid of the active-FT, in MHz. Ascending for the
        upper sideband, descending for the lower (matching the persisted Stage
        1 convention). Bin spacing is ``1 / T_active`` MHz.
    complex_spectrum : np.ndarray
        Complex active-FT on ``freq_mhz``, in ``dt_us * rfft(active * apod)``
        units (natural ``h_T`` form -- see module docstring). Shape matches
        ``freq_mhz``.
    alpha : float
        ``N_active / N_padded`` -- the persisted bin-correlation factor. Equal
        to 1 only when no zero-padding is in play (test fixtures); for a real
        Stage 1 pipeline with ``zpf >= 1`` this is the bin-density / variance
        rescale factor that links persisted-FT noise to active-FT noise.
    n_active : int
        Number of FID samples in ``[t0, t0+T]`` -- the FFT input length.
    n_padded : int
        Length of the canonical Stage 1 zero-padded record (informational --
        the active-FT itself is not zero-padded). Tracked so callers can
        compute ``alpha`` exactly without re-deriving it from the FID.
    """

    freq_mhz: np.ndarray
    complex_spectrum: np.ndarray
    alpha: float
    n_active: int
    n_padded: int


def compute_active_ft(
    fid: np.ndarray,
    sample_dt_us: float,
    *,
    start_us: float,
    end_us: float,
    expf_us: Optional[float],
    probe_freq_mhz: float,
    sideband: SidebandLike,
    n_padded: int,
    rdc: bool = True,
) -> ActiveFTResult:
    """Compute the active-portion FT of an FID for Stage 5 fitting.

    Extracts the ``[start_us, end_us]`` active region from the FID, applies
    the exponential apodization ``exp(-(t - t0)/expf_us)`` relative to the
    active start ``t0 = start_us``, removes the DC component (matching the
    canonical ``rdc=True`` Stage 1 step), then rfft's just the active region
    -- no zero-padding. The result is in the ``[0, T]`` reference frame
    ``h_T`` models, so the fit needs no de-ramp.

    The amplitude convention is ``dt_us * rfft(active * apod)``: a damped
    cosine of true amplitude ``A``, phase ``phi``, decay ``tau`` has on-line
    response ``0.5 * A * exp(i*phi) * h_T(0; tau, T)`` (units: ``us * V`` if
    the FID is in volts), matching the prototype / synthetic-test convention.

    Parameters
    ----------
    fid : np.ndarray
        Raw FID samples (real, 1-D). Length is the original FID length
        ``N_total``; the active region is the index slice that maps to
        ``[start_us, end_us]``.
    sample_dt_us : float
        FID sample spacing in microseconds (= ``fid.spacing * 1e6``).
    start_us, end_us : float
        Active region start / end times in microseconds. ``start_us`` may be
        zero; ``end_us > start_us`` is required.
    expf_us : float or None
        Exponential apodization time constant in microseconds, applied as
        ``exp(-(t - t0)/expf_us)`` over the active region. ``None`` skips
        apodization entirely (matching the Stage 1 ``expf_us=None`` path).
    probe_freq_mhz : float
        Probe (LO) frequency in MHz. The molecular frequency grid is
        ``f = probe + s * f_bb`` with ``s`` from :func:`sideband_sign`.
    sideband : Sideband or str
        Sideband configuration (``"lower"`` / ``"upper"`` or the enum).
    n_padded : int
        Length of the canonical Stage 1 zero-padded record. Used only to
        record ``alpha = N_active / N_padded`` on the result -- the active-FT
        itself is not zero-padded. Pass ``N_active`` (so ``alpha = 1``) for
        synthetic tests where there is no persisted record to compare against.
    rdc : bool, default True
        Subtract the mean of the apodized active region (matches the
        canonical Stage 1 ``rdc=True``).

    Returns
    -------
    ActiveFTResult
        The active-FT in the natural ``h_T`` amplitude convention.

    Raises
    ------
    ValueError
        If ``fid`` is not 1-D, ``sample_dt_us`` is non-positive, the active
        region is empty or out of bounds, ``expf_us`` is provided and
        non-positive, or ``n_padded < n_active``.
    """
    fid_arr = np.asarray(fid, dtype=float)
    if fid_arr.ndim != 1:
        raise ValueError("fid must be 1-D")
    if sample_dt_us <= 0:
        raise ValueError("sample_dt_us must be positive")
    if start_us < 0:
        raise ValueError("start_us must be non-negative")
    if end_us <= start_us:
        raise ValueError("end_us must be greater than start_us")
    if expf_us is not None and expf_us <= 0:
        raise ValueError("expf_us must be positive when provided")

    n_total = fid_arr.size
    if n_total == 0:
        raise ValueError("fid must not be empty")

    # Active-region indexing: searchsorted on the FID time axis. The
    # convention matches Stage 1's preprocess (which uses
    # ``np.searchsorted(time_us, start_us)`` / ``end_us``) so the active
    # samples here are exactly those Stage 1 keeps before zero-padding.
    time_us = np.arange(n_total) * sample_dt_us
    start_idx = int(np.searchsorted(time_us, start_us))
    end_idx = int(np.searchsorted(time_us, end_us))
    if end_idx <= start_idx:
        raise ValueError(
            f"active region [{start_us}, {end_us}] us is empty in FID of "
            f"length {n_total} at sample_dt_us={sample_dt_us}"
        )
    if end_idx > n_total:
        end_idx = n_total

    active = fid_arr[start_idx:end_idx].astype(float, copy=True)
    n_active = active.size

    if n_padded < n_active:
        raise ValueError(f"n_padded ({n_padded}) must be >= n_active ({n_active})")

    # Apodization relative to the active start: t = 0 at start_us.
    if expf_us is not None:
        t_relative_us = np.arange(n_active) * sample_dt_us
        active *= np.exp(-t_relative_us / expf_us)

    # Match Stage 1's rdc step (mean removal) on the active region.
    if rdc:
        active -= active.mean()

    # Natural h_T convention: dt * rfft(active * apod). At bin spacing
    # 1/T_active MHz, this is the [0, T]-frame analogue of the continuous FT
    # ∫_0^T x(t) e^{-i2π Δf t} dt, with no further normalization.
    spectrum = sample_dt_us * np.fft.rfft(active)

    # Baseband frequency grid in MHz: bin k <-> k / (n_active * dt_us) MHz.
    f_bb_mhz = np.fft.rfftfreq(n_active, d=sample_dt_us)
    s = sideband_sign(sideband)
    freq_mhz = probe_freq_mhz + s * f_bb_mhz

    alpha = float(n_active) / float(n_padded)

    return ActiveFTResult(
        freq_mhz=freq_mhz.astype(float),
        complex_spectrum=spectrum.astype(np.complex128),
        alpha=alpha,
        n_active=int(n_active),
        n_padded=int(n_padded),
    )
