"""
Truncation-leakage detection and reach estimation (Stages 3-4).

Two complementary tools, shared by the Stage 3 gap-pass mask and Stage 4 window
assignment:

* :func:`deramp_to_active_start` + :func:`leakage_touched_intervals` -- the
  *measured* leakage-extent map. De-ramping the spectrum to the active-region
  turn-on restores the coherent-sum edge statistic, whose above-threshold runs
  are the authoritative leakage mask. See
  ``dev-docs/planning/leakage-detection-rework.md``.
* :func:`estimate_leakage_reach` -- the closed-form analytic reach (below), kept
  as a cheap *proposal* of a strong line's extent. It under-predicts the real
  cumulative skirt and is no longer the authoritative mask.

Truncation-leakage reach estimator (Stage 3 open question O1).

A strong line in a *boxcar-truncated* (unapodized) FTMW spectrum carries
sinc-shaped truncation sidelobes whose envelope decays only as ``1/Δf``. The
Stage 3 gap pass runs an unapodized detector at a low SNR floor to recover weak
lines hiding between the strong ones; without a mask it would re-detect those
sidelobes as spurious weak lines. This module provides the analytic estimate of
how far from a strong line its sidelobes stay above the gap-pass floor, so the
gap pass can be masked within ``±reach`` of every strong line.

The same estimate sets the initial window extent in Stage 4 (a window must be
at least as wide as the strongest in-window line's leakage reach), so it is
defined once here and imported by both stages.

Model
-----
Baseband response of an exponentially damped cosine (natural decay time
constant ``τ``) observed over a finite acquisition ``T``::

    S(Δf) = [1 - exp(-(1/τ + i2πΔf) T)] / (1/τ + i2πΔf)

At line center ``S(0) = τ_eff`` with ``τ_eff = τ (1 - exp(-T/τ))`` (``→ T`` in
the undamped/boxcar limit ``τ → ∞``). Far from center the envelope is::

    |S_env(Δf)| ≈ (1 + exp(-T/τ)) / (2π |Δf|)

so the sidelobe envelope, expressed as a fraction of the peak height, is
``(1 + exp(-T/τ)) / (2π |Δf| τ_eff)``. Equating that to the gap-pass floor
(``min_snr`` in SNR units) for a line of signal-to-noise ``peak_snr`` gives the
closed form

    reach = peak_snr · (1 + exp(-T/τ)) / (2π · τ_eff · min_snr)         (Hz)

Limits (sanity):

* Undamped boxcar (``τ → ∞``): ``τ_eff → T``, ``1 + exp(-T/τ) → 2`` so
  ``reach → peak_snr / (π · T · min_snr)`` — exactly the offset at which the
  ``1/(π|Δf|)`` sidelobe-peak envelope of ``T·sinc(Δf·T)`` crosses the floor.
* Heavily damped (``τ ≪ T``): ``τ_eff → τ``, envelope factor ``→ 1``; the
  edge term ``exp(-T/τ) → 0`` (the signal has decayed before truncation, so
  there is little genuine leakage) and the reach collapses toward the
  Lorentzian core width. The ``1/Δf`` form overestimates the true ``1/Δf²``
  Lorentzian wing here, i.e. the mask is deliberately conservative (slightly
  too wide rather than too narrow) in a regime where leakage is weak anyway.

The estimate is intentionally an envelope/order-of-magnitude bound, not a
per-sidelobe prediction; masking errs wide on purpose.
"""

from typing import List, Optional, Tuple, Union, cast

import numpy as np

from .edge_coherence import (
    DEFAULT_EDGE_M,
    DEFAULT_EDGE_THRESHOLD,
    above_threshold_intervals,
    rolling_coherence,
)

ArrayLike = Union[float, np.ndarray]


def estimate_leakage_reach(
    peak_snr: ArrayLike,
    acquisition_us: float,
    min_snr: float = 3.0,
    tau_us: Optional[float] = None,
) -> ArrayLike:
    """Estimate the truncation-leakage reach of a strong line.

    Returns the half-width ``Δf`` (MHz) beyond which a line of signal-to-noise
    ``peak_snr`` no longer has unapodized sidelobes above ``min_snr·σ``. The
    gap pass should be masked over ``[f0 - reach, f0 + reach]`` for each strong
    line; Stage 4 uses the same value as a lower bound on window extent.

    Parameters
    ----------
    peak_snr : float or np.ndarray
        Peak signal-to-noise ratio of the strong line(s). Scalars and arrays
        are both accepted; an array in gives an array out.
    acquisition_us : float
        Acquisition (FID) duration ``T`` in microseconds. Use the *active*
        FID duration actually transformed, not the digitizer record length.
    min_snr : float, default 3.0
        Gap-pass detection floor in SNR units (the threshold the gap pass runs
        at). Lower floors give a wider reach.
    tau_us : float, optional
        Assumed shared natural decay time constant ``τ`` in microseconds. If
        ``None`` (default), the undamped/boxcar limit is used (``τ_eff = T``,
        the most leakage-prone case and the safe default for masking).

    Returns
    -------
    float or np.ndarray
        Leakage reach in MHz, matching the shape of ``peak_snr``. Returns 0.0
        where ``peak_snr <= 0`` (no line, nothing to mask).

    Raises
    ------
    ValueError
        If ``acquisition_us`` or ``min_snr`` is not positive, or ``tau_us`` is
        given and not positive.
    """
    if acquisition_us <= 0:
        raise ValueError("acquisition_us must be positive")
    if min_snr <= 0:
        raise ValueError("min_snr must be positive")
    if tau_us is not None and tau_us <= 0:
        raise ValueError("tau_us must be positive when provided")

    snr = np.asarray(peak_snr, dtype=float)
    T = acquisition_us * 1e-6  # seconds

    if tau_us is None:
        # Undamped boxcar limit: tau_eff -> T, envelope factor -> 2.
        tau_eff = T
        env_factor = 2.0
    else:
        tau = tau_us * 1e-6
        edge = np.exp(-T / tau)  # exp(-T/τ): truncation-edge amplitude
        # τ_eff = τ(1 - e^{-T/τ}); -expm1(-x) = 1 - e^{-x} (stable for small x).
        tau_eff = tau * (-np.expm1(-T / tau))
        env_factor = 1.0 + edge

    reach_hz = snr * env_factor / (2.0 * np.pi * tau_eff * min_snr)
    reach_mhz = reach_hz * 1e-6

    # No line -> nothing to mask.
    reach_arr = cast(
        np.ndarray, np.asarray(np.where(snr > 0.0, reach_mhz, 0.0), dtype=float)
    )

    if reach_arr.ndim == 0:
        return float(reach_arr)
    return reach_arr


def deramp_to_active_start(
    freq_mhz: np.ndarray,
    complex_spectrum: np.ndarray,
    probe_freq_mhz: float,
    start_us: float,
) -> np.ndarray:
    """Reference a full-record-rfft complex spectrum to the active-region turn-on.

    The pipeline FT is an rfft of the *whole* (zero-padded) FID record, with the
    active signal occupying ``[start_us, end_us]``. A signal that begins at
    sample index ``s`` (time ``t0 = start_us``) carries a phase ramp
    ``exp(-i 2pi f_bb t0)`` on every rfft bin (numpy rfft convention), where
    ``f_bb`` is the *baseband* frequency. That ramp makes a strong line's
    truncation-leakage skirt oscillate, so the coherent edge statistic
    ``S_coh`` (see :mod:`ftmwpipeline.preprocessing.edge_coherence`) cancels on
    genuine leakage. Multiplying the spectrum by ``exp(+i 2pi f_bb t0)`` undoes
    the ramp and restores the non-oscillating leakage component a coherent sum
    can detect.

    The baseband frequency is ``f_bb = |f - f_probe|`` -- a single experiment is
    single-sideband, so every line lies on one side of the probe; the absolute
    value makes the de-ramp correct for both sidebands with no sideband
    argument. The leftover global constant phase ``exp(-/+ i 2pi f_probe t0)``
    does not affect any coherence statistic (it factors out of ``|sum z|``).

    Parameters
    ----------
    freq_mhz : np.ndarray
        Real (sideband-converted) frequency grid in MHz, 1-D.
    complex_spectrum : np.ndarray
        Complex spectrum on ``freq_mhz``, same shape.
    probe_freq_mhz : float
        Probe (LO) frequency in MHz; ``|f - probe|`` is the baseband frequency.
    start_us : float
        Active-region start time ``t0`` in microseconds (``>= 0``). ``0`` makes
        the de-ramp the identity.

    Returns
    -------
    np.ndarray
        The de-ramped complex spectrum (same shape; magnitudes unchanged).

    Raises
    ------
    ValueError
        If the arrays differ in shape, are not 1-D, or ``start_us`` is negative.
    """
    freq = np.asarray(freq_mhz, dtype=float)
    z = np.asarray(complex_spectrum, dtype=complex)
    if freq.shape != z.shape:
        raise ValueError("freq_mhz and complex_spectrum must have equal shape")
    if freq.ndim != 1:
        raise ValueError("freq_mhz must be 1-dimensional")
    if start_us < 0:
        raise ValueError("start_us must be non-negative")

    f_bb_hz = np.abs(freq - probe_freq_mhz) * 1e6
    t0_s = start_us * 1e-6
    return cast(np.ndarray, z * np.exp(2j * np.pi * f_bb_hz * t0_s))


def leakage_touched_intervals(
    freq_mhz: np.ndarray,
    complex_spectrum: np.ndarray,
    rms_noise: np.ndarray,
    probe_freq_mhz: float,
    start_us: float,
    band_m: int = DEFAULT_EDGE_M,
    threshold: float = DEFAULT_EDGE_THRESHOLD,
) -> List[Tuple[int, int]]:
    """Index runs where coherent truncation leakage is detectable.

    De-ramps the spectrum to the active-region turn-on
    (:func:`deramp_to_active_start`), runs the rolling complex-edge coherence
    statistic, and returns the contiguous index runs above ``threshold``. These
    are the spectrum's leakage-touched regions: Stage 3 masks its gap pass
    outside them; Stage 4 uses them for strong-cluster grouping and
    fixed-contributor attachment.

    Parameters
    ----------
    freq_mhz : np.ndarray
        Real frequency grid in MHz, 1-D.
    complex_spectrum : np.ndarray
        Complex spectrum on ``freq_mhz``, same shape.
    rms_noise : np.ndarray
        Per-point complex noise RMS on the same grid (canonical Stage 2 noise).
    probe_freq_mhz : float
        Probe (LO) frequency in MHz, passed to :func:`deramp_to_active_start`.
    start_us : float
        Active-region start time in microseconds, passed to
        :func:`deramp_to_active_start`.
    band_m : int, default ``DEFAULT_EDGE_M``
        Rolling coherence band width ``M``.
    threshold : float, default ``DEFAULT_EDGE_THRESHOLD``
        ``S_coh`` threshold ``T_edge``.

    Returns
    -------
    list of tuple of int
        ``(lo, hi)`` inclusive index runs, ordered by ``lo``.
    """
    z_ref = deramp_to_active_start(freq_mhz, complex_spectrum, probe_freq_mhz, start_us)
    rolling = rolling_coherence(z_ref, rms_noise, band_m)
    return above_threshold_intervals(rolling, threshold)
