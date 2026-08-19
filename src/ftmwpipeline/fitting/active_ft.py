"""
Active-portion FT for Stage 5 fitting.

The persisted Stage 1 spectrum is an rfft of the *whole* FID record (the
active region embedded in a full-length array with the inactive samples
zeroed). The active samples occupy a fraction ``alpha = N_active / N_total``
of the record, so adjacent bins are correlated and the naive
``N_dof = M - N_params`` overcounts the independent samples; reduced
chi-squared, the F-test, and AIC are all biased optimistic on the persisted
spectrum.

Stage 5 dissolves the problem at source by fitting the **active-portion FT** --
the rfft of just the ``fid[t0 : t0+T]`` active samples, unapodized and with no
zero-padding. The result has

* independent bins (``alpha = 1`` by construction),
* no phase ramp (the active-FT is in the ``[0, T]`` form ``h_T`` models),
* the same molecular frequency axis convention as the persisted FT (so
  windows defined as frequency ranges translate directly), with a coarser
  bin spacing ``1/T_active`` instead of ``1/T_total``.

This module owns the construction (:func:`compute_active_ft`). The active-FT
is the working measurement domain for Stages 2, 4, and 5 (and Stage 3's
re-scoring pass): computed on demand from ``stage0_fid_data`` and the
persisted Stage 1 settings, not itself persisted in the ``.ftmw`` file.

Amplitude convention
--------------------
The complex spectrum is ``dt_us * rfft(active)``, which is the
``[0, T]``-frame discrete approximation of the continuous Fourier transform
in MHz / microsecond units. A line of true amplitude ``A``, phase ``phi``, and
decay ``tau`` has on-line response ``0.5 * A * exp(i*phi) * h_T(0; tau, T)``,
i.e. exactly the form :mod:`ftmwpipeline.fitting.peak_model` models. The
synthetic-spectrum builder the Stage 5 prototype tests use
(``test_plan_execution.py: _synth_spectrum``) builds spectra in this same
convention, so unit tests can construct :class:`ActiveFTResult` directly.

Noise convention
----------------
Per-bin noise on the active-FT is the Stage 2 scatter authority
(:func:`ftmwpipeline.preprocessing.noise_estimation.estimate_active_ft_noise`)
measured on the active-FT spectrum -- the *same* estimator and grid Stage 2
persists. No conversion factor, no ``1/sqrt(alpha)`` rescale: the noise
estimate comes from the same spectrum the fit sees, so any normalization
choices cancel by construction.

References
----------
* ``docs/source/methods/stage5_fitting.rst`` (calibration scope)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

from ftmwpipeline.core.data_structures import Sideband

from .peak_model import sideband_sign

__all__ = [
    "ActiveFTResult",
    "active_ft_bin_spacing_mhz",
    "active_region_bounds",
    "compute_active_ft",
]

SidebandLike = Union[Sideband, str]


def active_ft_bin_spacing_mhz(acquisition_us: float) -> float:
    """Active-FT bin spacing (MHz): ``1 / acquisition_us``.

    THE single resolved accessor for the active-FT bin spacing
    (``dev-docs/SCIENCE_STRATEGY.md`` Requirement 8). Every tolerance,
    threshold, window width, or increment that expresses a *spectral
    distance* as a multiple of the active-FT bin must compute that multiple
    through this function -- never by re-deriving ``1.0 / T`` inline at the
    call site. A duplicated inline form is exactly how a future edit could
    silently substitute the wrong quantity below, with nothing to catch it.
    It lives in this module (not a dependency-free utility module) because
    the active FT is what it is the spacing *of*: the two belong together,
    and every consumer of this function already needs to know what the
    active FT is.

    Parameters
    ----------
    acquisition_us : float
        The active region's duration, ``end_us - start_us``. This is the one
        spectrum every measuring stage shares -- Stage 2's noise estimate,
        Stage 3's peak detection, and Stage 5's fit all run on the active FT
        (see the module docstring and ``dev-docs/SCIENCE_STRATEGY.md``
        Requirements 2 and 8). The magnitude-display FT (2x zero-padded for
        visualization) and the raw / full-length / imported FT (the
        untrimmed original record, e.g. for Stage 0 start detection) are
        never measurement surfaces.

    **Never** pass ``ActiveFTResult.n_raw`` / ``Stage5FitContext.n_raw``, a
    sample count, or anything derived from the persisted Stage 1 full-length
    record here. ``n_raw`` is kept on the active-FT result only so a caller
    can recompute ``alpha = N_active / N_raw``; it is *not* the active FT's
    own length, and using it in place of ``acquisition_us`` gives a bin
    spacing wrong by ``alpha`` -- silently, since both are plain floats and
    the mistake has no type error to catch it. This function takes a
    *duration* (microseconds), never a sample count, specifically so that
    mistake cannot be made by passing the wrong argument of the same type in
    the wrong units.

    Raises
    ------
    ValueError
        If ``acquisition_us`` is not positive.
    """
    t = float(acquisition_us)
    if t <= 0.0:
        raise ValueError(f"acquisition_us must be positive, got {t}")
    return 1.0 / t


def active_region_bounds(
    n_total: int, sample_dt_us: float, start_us: float, end_us: float
) -> tuple[int, int]:
    """Half-open ``[start_idx, end_idx)`` slice of the active FID region.

    Single source of truth for the active-sample slice. Uses ``searchsorted`` on
    the FID time axis (matching Stage 1's preprocess), so the active samples are
    exactly those Stage 1 keeps before zero-padding. ``end_idx`` is clamped to
    ``n_total``.

    Both the fit's :func:`compute_active_ft` and the display-only padded FT
    (``_padded_active_display_ft``) extract via this helper, so the 2x-zero-pad
    of the display spectrum coincides with the native active FT bin-for-bin
    (``padded[2k] == native[k]``). Independent ``floor``/``ceil`` vs
    ``searchsorted`` extractions previously differed by one sample, which shifted
    the padded grid off the native grid and -- when the offset count landed a
    line on a mid-bin null -- erased the line from the magnitude display.
    """
    time_us = np.arange(int(n_total)) * sample_dt_us
    start_idx = int(np.searchsorted(time_us, start_us))
    end_idx = min(int(np.searchsorted(time_us, end_us)), int(n_total))
    return start_idx, end_idx


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
        Complex active-FT on ``freq_mhz``, in ``dt_us * rfft(active)`` units
        (natural ``h_T`` form -- see module docstring). Shape matches
        ``freq_mhz``.
    alpha : float
        ``N_active / N_raw`` -- the persisted bin-correlation factor. Equal
        to 1 only when the FFT input is exactly the active region (test
        fixtures); for the raw, full-length FT this is the active-sample
        fraction that links full-length-FT noise to active-FT noise.
    n_active : int
        Number of FID samples in ``[t0, t0+T]`` -- the FFT input length. This
        is the active FT's own length; its bin spacing is
        ``1 / acquisition_us`` (:func:`compute_active_ft`'s ``end_us -
        start_us``), NOT ``1 / (n_active * sample_dt_us)`` in general and
        never derived from ``n_raw`` below.
    n_raw : int
        Length of the raw, full-length FID record (informational; the Stage 1
        full-length FT is unpadded, so this is also that FT's input length).
        Tracked only so callers can recompute ``alpha`` without re-deriving it
        from the FID. **Never** use ``n_raw`` where the active FT's own length
        or bin spacing is wanted; see
        :func:`ftmwpipeline.fitting.active_ft.active_ft_bin_spacing_mhz`
        for the full statement of this trap.
    """

    freq_mhz: np.ndarray
    complex_spectrum: np.ndarray
    alpha: float
    n_active: int
    n_raw: int


def compute_active_ft(
    fid: np.ndarray,
    sample_dt_us: float,
    *,
    start_us: float,
    end_us: float,
    probe_freq_mhz: float,
    sideband: SidebandLike,
    n_raw: int,
    rdc: bool = True,
) -> ActiveFTResult:
    """Compute the active-portion FT of an FID for Stage 5 fitting.

    Extracts the ``[start_us, end_us]`` active region from the FID, removes
    the DC component (matching the persisted Stage 1 step, which is
    unconditional), then rfft's just the active region -- no apodization, no
    zero-padding. The
    result is in the ``[0, T]`` reference frame ``h_T`` models, so the fit
    needs no de-ramp.

    The amplitude convention is ``dt_us * rfft(active)``: a damped cosine of
    true amplitude ``A``, phase ``phi``, decay ``tau`` has on-line response
    ``0.5 * A * exp(i*phi) * h_T(0; tau, T)`` (units: ``us * V`` if the FID is
    in volts), matching the prototype / synthetic-test convention.

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
    probe_freq_mhz : float
        Probe (LO) frequency in MHz. The molecular frequency grid is
        ``f = probe + s * f_bb`` with ``s`` from :func:`sideband_sign`.
    sideband : Sideband or str
        Sideband configuration (``"lower"`` / ``"upper"`` or the enum).
    n_raw : int
        Length of the raw, full-length FID record (the Stage 1 full-length FT
        input). Used only to record ``alpha = N_active / N_raw`` on the
        result -- never to derive the active FT's own length or bin spacing
        (that is ``n_active`` / ``end_us - start_us``, computed independently
        of ``n_raw`` below). Pass ``N_active`` (so ``alpha = 1``) for
        synthetic tests where there is no full-length record to compare
        against.
    rdc : bool, default True
        Subtract the mean of the active region (matches the persisted Stage 1
        DC-removal step, which is unconditional).

    Returns
    -------
    ActiveFTResult
        The active-FT in the natural ``h_T`` amplitude convention.

    Raises
    ------
    ValueError
        If ``fid`` is not 1-D, ``sample_dt_us`` is non-positive, the active
        region is empty or out of bounds, or ``n_raw < n_active``.
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

    n_total = fid_arr.size
    if n_total == 0:
        raise ValueError("fid must not be empty")

    # Active-region indexing via the shared bounds helper so the display-only
    # padded FT extracts the *exact same* samples (see active_region_bounds).
    start_idx, end_idx = active_region_bounds(n_total, sample_dt_us, start_us, end_us)
    if end_idx <= start_idx:
        raise ValueError(
            f"active region [{start_us}, {end_us}] us is empty in FID of "
            f"length {n_total} at sample_dt_us={sample_dt_us}"
        )

    active = fid_arr[start_idx:end_idx].astype(float, copy=True)
    n_active = active.size

    if n_raw < n_active:
        raise ValueError(f"n_raw ({n_raw}) must be >= n_active ({n_active})")

    # Match Stage 1's unconditional DC-removal step (mean removal) on the
    # active region.
    if rdc:
        active -= active.mean()

    # Natural h_T convention: dt * rfft(active). At bin spacing 1/T_active MHz,
    # this is the [0, T]-frame analog of the continuous FT
    # ∫_0^T x(t) e^{-i2π Δf t} dt, with no further normalization.
    spectrum = sample_dt_us * np.fft.rfft(active)

    # Baseband frequency grid in MHz: bin k <-> k / (n_active * dt_us) MHz.
    f_bb_mhz = np.fft.rfftfreq(n_active, d=sample_dt_us)
    s = sideband_sign(sideband)
    freq_mhz = probe_freq_mhz + s * f_bb_mhz

    alpha = float(n_active) / float(n_raw)

    return ActiveFTResult(
        freq_mhz=freq_mhz.astype(float),
        complex_spectrum=spectrum.astype(np.complex128),
        alpha=alpha,
        n_active=int(n_active),
        n_raw=int(n_raw),
    )
