"""
Truncation-leakage de-ramping (display path).

De-ramping the full-record-rfft spectrum to the active-region turn-on restores
the coherent-sum edge statistic (see :mod:`ftmwpipeline.preprocessing.edge_coherence`)
by cancelling the phase ramp a signal that starts partway through the record
otherwise imposes on every rfft bin:

* :func:`deramp_to_active_start` -- reference the spectrum to the active turn-on.

Stage 3 (:func:`ftmwpipeline.preprocessing.edge_coherence.active_edge_coherence`,
called from ``_internal/stage3_impl.py``) and Stage 4
(:mod:`ftmwpipeline.preprocessing.window_planning`) both compute their leakage
statistic directly on the active-FT slice, which needs no de-ramp -- the active
FT is already referenced to the active-region start. ``deramp_to_active_start``
is instead the display-path primitive: it lets
:func:`ftmwpipeline.preprocessing.edge_coherence.coherence_curve` (consumed by
``window_visualization`` / ``tuning.plots``) compute a comparable coherent-edge
curve on the full-record FT for visualization.

A closed-form analytic *reach* estimator preceded the measured active-FT
statistic and was found 7-25x too narrow vs the real cumulative skirt; the
finite-T leakage envelope it used survives only where Stage 4 still needs it,
in :func:`ftmwpipeline.preprocessing.window_planning._leakage_envelope_fraction`.
"""

from typing import cast

import numpy as np


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
