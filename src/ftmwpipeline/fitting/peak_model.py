"""
Finite-acquisition line-shape model for Stage 5 per-window fitting.

Stage 5 fits each analysis window to a sum of finite-acquisition damped
cosines. This module is the *model* layer: the closed-form line shape ``h_T``,
its analytic Jacobian, and the demodulation / sideband mapping that connects
the molecular-frequency grid the data lives on to the baseband offset the fit
is parameterised in.

It is a pure algorithm module (arrays in, arrays out, no file or pipeline
state), unit-tested in isolation. Orchestration lives in
:mod:`ftmwpipeline._internal.stage5_impl` and the per-window least-squares
solver in :mod:`ftmwpipeline.fitting.window_fit` (Stage 5 task 3). The model
and its conventions were established by the Stage 5 research prototype
(``dev-docs/research/stage5-fitting/``); the plan is
``dev-docs/planning/stage5-fitting.md``.

The model
---------
A molecular line is an exponentially damped cosine excited at the
active-region turn-on and observed over the finite acquisition ``[t0, t0+T]``.
In *baseband* frequency offset ``Δf`` (MHz) from line centre its complex-FT
response, in the de-ramped ``[0, T]`` frame, is

    X(Δf) ≈ ½ A e^{iφ} · h_T(Δf; τ)
    h_T(Δf; τ) = [1 - exp(-(1/τ + i2π Δf) T)] / (1/τ + i2π Δf)

``h_T`` is the exact rfft-domain line shape: at centre ``h_T(0) = τ_eff =
τ(1 - e^{-T/τ})`` and far from centre its magnitude decays as the ``1/|Δf|``
truncation-leakage skirt. A window's model is a sum of such terms -- see
:func:`model_spectrum`. Because the model carries leakage exactly, the fit can
be done on the *unwindowed* (boxcar-truncated) spectrum without apodization.

Units
-----
Everything is in **microseconds and MHz**. A frequency in MHz times a time in
µs is dimensionless, so ``h_T`` carries units of µs and ``h_T(0)`` is ``τ_eff``
in µs. Mixing in SI (Hz, s) silently rescales the on-line response by 10⁶.

The sideband mapping
--------------------
The persisted spectrum is on a *molecular* frequency grid ``f``; a line
physically sits at *baseband* frequency ``f_bb = s·(f - f_probe)`` with the
sideband sign ``s = -1`` (lower sideband) or ``s = +1`` (upper). A window is
fit about a reference molecular frequency ``f_c``; every line is fit by its
signed baseband offset ``δ = s·(f - f_c)`` -- a small (~MHz) signed number,
not an absolute ~36 GHz frequency -- and the window grid is converted to the
matching coordinate ``u = s·(f - f_c)`` (:func:`baseband_offset`). Recovered
offsets map back as ``f = f_c + s·δ`` (:func:`molecular_frequency`).

The sign is load-bearing: ``h_T(-Δf) = conj(h_T(Δf))`` exactly, so the wrong
``s`` conjugates every leakage skirt and biases fitted frequencies and phases
by 100s of kHz while the magnitude residual can still look plausible -- a
silent failure. The demodulation helpers are unit tested on synthetic lines of
both sidebands precisely to guard this.

The fit frame
-------------
``h_T`` is in the ``[0, T]`` form. Stage 5 fits on the **active-portion FT**
(:mod:`ftmwpipeline.fitting.active_ft`) -- the rfft of just the active samples
with the canonical apodization, which is already in the ``[0, T]`` form
natively. No de-ramp is needed: :func:`to_baseband_offset` only does the grid
conversion from molecular MHz to the signed baseband offset.

(Earlier drafts -- before D9 -- fitted on the persisted Stage 1 FT, which
carries an ``exp(-i2π f_bb t0)`` turn-on phase ramp; this module called
:func:`ftmwpipeline.preprocessing.leakage.deramp_to_active_start` to remove
it. With the active-FT contract the active samples are referenced to
``t = 0`` directly, so the deramp is a no-op. The de-ramp helper survives in
``preprocessing/leakage.py`` for Stage 4's edge-coherence work on the
persisted spectrum.)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Union, cast

import numpy as np
from scipy.special import wofz

from ftmwpipeline.core.data_structures import Sideband
from ftmwpipeline.core.peak_shape import PeakShape

__all__ = [
    "ModelPeak",
    "PeakShape",
    "sideband_sign",
    "h_T",
    "h_T_jacobian",
    "h_T_gaussian",
    "h_T_gaussian_jacobian",
    "h_T_shape",
    "h_T_shape_jacobian",
    "effective_tau",
    "effective_tau_gaussian",
    "effective_tau_shape",
    "model_spectrum",
    "baseband_offset",
    "molecular_frequency",
    "to_baseband_offset",
]


SidebandLike = Union[Sideband, str]


# ---------------------------------------------------------------------------
# Sideband sign
# ---------------------------------------------------------------------------
def sideband_sign(sideband: SidebandLike) -> float:
    """Return the sideband sign ``s`` connecting molecular and baseband axes.

    The baseband frequency of a line at molecular frequency ``f`` is
    ``f_bb = s·(f - f_probe)``: ``s = -1`` for the lower sideband (the
    molecular axis descends as baseband frequency rises) and ``s = +1`` for
    the upper.

    Parameters
    ----------
    sideband : Sideband or str
        Sideband configuration. A string is matched case-insensitively against
        ``"lower"``/``"lsb"`` and ``"upper"``/``"usb"``.

    Returns
    -------
    float
        ``-1.0`` for the lower sideband, ``+1.0`` for the upper.

    Raises
    ------
    ValueError
        If ``sideband`` is a string that is not a recognised sideband name.
    """
    if isinstance(sideband, str):
        key = sideband.strip().lower()
        if key in ("lower", "lsb"):
            return -1.0
        if key in ("upper", "usb"):
            return 1.0
        raise ValueError(f"unknown sideband: {sideband!r}")
    return -1.0 if sideband == Sideband.LOWER else 1.0


# ---------------------------------------------------------------------------
# The finite-T line shape and its Jacobian
# ---------------------------------------------------------------------------
def h_T(
    delta_f_mhz: np.ndarray,
    tau_us: float,
    acquisition_us: float,
) -> np.ndarray:
    """Closed-form complex FFT of a finite-T damped cosine.

    Evaluates ``h_T(Δf; τ)`` on a baseband frequency-offset grid -- the exact
    rfft-domain line shape, in the de-ramped ``[0, T]`` frame. At centre
    ``h_T(0) = τ_eff`` (see :func:`effective_tau`); far from centre the
    magnitude decays as the ``1/|Δf|`` truncation-leakage skirt with the
    coherent phase that lets a fitted line's skirt be subtracted exactly.

    Worked entirely in µs/MHz: a frequency in MHz times a time in µs is
    dimensionless, so the result carries units of µs.

    Parameters
    ----------
    delta_f_mhz : np.ndarray
        Baseband frequency offset ``Δf`` from line centre, in MHz. Scalars
        are accepted; the result is always an :class:`~numpy.ndarray`.
    tau_us : float
        Effective decay time constant ``τ`` in microseconds (``> 0``).
    acquisition_us : float
        Active acquisition length ``T`` in microseconds (``> 0``).

    Returns
    -------
    np.ndarray
        Complex line shape on ``delta_f_mhz``, in units of µs.

    Raises
    ------
    ValueError
        If ``tau_us`` or ``acquisition_us`` is not positive.
    """
    if tau_us <= 0.0:
        raise ValueError("tau_us must be positive")
    if acquisition_us <= 0.0:
        raise ValueError("acquisition_us must be positive")

    df = np.asarray(delta_f_mhz, dtype=float)
    denom = (1.0 / tau_us) + 1j * 2.0 * np.pi * df  # 1/µs
    response = (1.0 - np.exp(-denom * acquisition_us)) / denom  # µs
    return cast(np.ndarray, response)


def h_T_jacobian(
    delta_f_mhz: np.ndarray,
    tau_us: float,
    acquisition_us: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic derivatives of :func:`h_T` w.r.t. ``Δf`` (MHz) and ``τ`` (µs).

    With ``z = 1/τ + i2π Δf`` and ``e = exp(-zT)``, ``h_T = (1 - e)/z`` so

        dh/dz       = (T e)/z - (1 - e)/z²
        dh/d(Δf)    = dh/dz · (i2π)         since dz/d(Δf) = i2π
        dh/d(τ)     = dh/dz · (-1/τ²)       since dz/d(τ)  = -1/τ²

    The Stage 5 prototype verified this against central finite differences to
    a relative error of ~3×10⁻¹⁰, so the production fit uses the analytic
    Jacobian from the start (no finite-difference phase).

    Parameters
    ----------
    delta_f_mhz : np.ndarray
        Baseband frequency offset ``Δf`` from line centre, in MHz.
    tau_us : float
        Effective decay time constant ``τ`` in microseconds (``> 0``).
    acquisition_us : float
        Active acquisition length ``T`` in microseconds (``> 0``).

    Returns
    -------
    tuple of np.ndarray
        ``(dh/d(Δf_MHz), dh/d(τ_us))``, both complex arrays shaped like
        ``delta_f_mhz``.

    Raises
    ------
    ValueError
        If ``tau_us`` or ``acquisition_us`` is not positive.
    """
    if tau_us <= 0.0:
        raise ValueError("tau_us must be positive")
    if acquisition_us <= 0.0:
        raise ValueError("acquisition_us must be positive")

    df = np.asarray(delta_f_mhz, dtype=float)
    denom = (1.0 / tau_us) + 1j * 2.0 * np.pi * df
    e = np.exp(-denom * acquisition_us)
    dh_dz = (acquisition_us * e) / denom - (1.0 - e) / denom**2
    dh_ddf = dh_dz * (1j * 2.0 * np.pi)
    dh_dtau = dh_dz * (-1.0 / tau_us**2)
    return cast(np.ndarray, dh_ddf), cast(np.ndarray, dh_dtau)


def effective_tau(tau_us: float, acquisition_us: float) -> float:
    """On-line gain ``τ_eff = τ(1 - e^{-T/τ})`` -- the value of ``h_T`` at Δf=0.

    ``τ_eff`` is the (real) magnitude of the model response on resonance; it
    sets the SNR scale, since a line of amplitude ``A`` has on-line response
    ``½ A τ_eff``. In the undamped/boxcar limit ``τ → ∞`` it tends to ``T``.

    Parameters
    ----------
    tau_us : float
        Decay time constant ``τ`` in microseconds (``> 0``).
    acquisition_us : float
        Active acquisition length ``T`` in microseconds (``> 0``).

    Returns
    -------
    float
        ``τ_eff`` in microseconds.

    Raises
    ------
    ValueError
        If ``tau_us`` or ``acquisition_us`` is not positive.
    """
    if tau_us <= 0.0:
        raise ValueError("tau_us must be positive")
    if acquisition_us <= 0.0:
        raise ValueError("acquisition_us must be positive")
    # -expm1(-x) = 1 - e^{-x}, stable for small x.
    return tau_us * float(-np.expm1(-acquisition_us / tau_us))


# ---------------------------------------------------------------------------
# Gaussian envelope variant
# ---------------------------------------------------------------------------
def h_T_gaussian(
    delta_f_mhz: np.ndarray,
    tau_G_us: float,
    acquisition_us: float,
) -> np.ndarray:
    """Closed-form complex FFT of a finite-T Gaussian-windowed cosine.

    For envelope ``exp(-(t/τ_G)²)`` integrated over ``[0, T]``, the natural
    closed form via complete-the-square is

        h_T(Δf; τ_G) = (τ_G √π / 2) · exp(β²) · [erf(T/τ_G + β) − erf(β)]
        β = i π Δf τ_G  (purely imaginary).

    The naive evaluation overflows for large ``|Δf|·τ_G``: ``exp(β²) =
    exp(−π²·Δf²·τ_G²)`` underflows to zero while the bracketed
    ``erf``-difference grows like ``exp(π²·Δf²·τ_G²)``; the floating-point
    cancellation discards 100+ digits of precision.

    The stable form cancels the ``exp(β²)`` prefactor against the erfc /
    Faddeeva expansion ``erf(z) = 1 − exp(−z²) · wofz(i·z)``:

        h_T(Δf; τ_G) = (τ_G √π / 2) · [wofz(i β) −
            exp(−(T/τ_G)²) · exp(−i 2π Δf T) · wofz(i (T/τ_G + β))]

    Every intermediate is bounded: ``i β`` is purely real,
    ``i (T/τ_G + β)`` has bounded real and imaginary parts, the damping
    factor ``exp(−(T/τ_G)²) ≤ 1``, and the phase factor is unit-modulus.

    At centre Δf = 0, ``β = 0``, ``wofz(0) = 1``, the phase factor = 1, so

        h_T(0; τ_G) = (τ_G √π / 2) · [1 − exp(−(T/τ_G)²)·wofz(i T/τ_G)]
                    = (τ_G √π / 2) · erf(T/τ_G)
                    = effective_tau_gaussian(τ_G, T)

    matching the real-arg closed form.

    Worked in µs / MHz: ``β`` is dimensionless (MHz × µs cancels), and
    ``h_T`` carries units of µs, matching :func:`h_T`.

    Parameters
    ----------
    delta_f_mhz : np.ndarray
        Baseband frequency offset ``Δf`` from line centre, in MHz. Scalars
        are accepted; the result is always an :class:`~numpy.ndarray`.
    tau_G_us : float
        Gaussian decay time constant ``τ_G`` in microseconds (``> 0``).
    acquisition_us : float
        Active acquisition length ``T`` in microseconds (``> 0``).

    Returns
    -------
    np.ndarray
        Complex line shape on ``delta_f_mhz``, in units of µs.

    Raises
    ------
    ValueError
        If ``tau_G_us`` or ``acquisition_us`` is not positive.
    """
    if tau_G_us <= 0.0:
        raise ValueError("tau_G_us must be positive")
    if acquisition_us <= 0.0:
        raise ValueError("acquisition_us must be positive")
    df = np.asarray(delta_f_mhz, dtype=float)
    beta = 1j * np.pi * df * tau_G_us
    t_over_tau = acquisition_us / tau_G_us
    w_lo = wofz(1j * beta)
    w_hi = wofz(1j * (t_over_tau + beta))
    damping = np.exp(-(t_over_tau**2))
    phase = np.exp(-1j * 2.0 * np.pi * df * acquisition_us)
    response = (tau_G_us * np.sqrt(np.pi) / 2.0) * (w_lo - damping * phase * w_hi)
    return cast(np.ndarray, response.astype(np.complex128))


def h_T_gaussian_jacobian(
    delta_f_mhz: np.ndarray,
    tau_G_us: float,
    acquisition_us: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic derivatives of :func:`h_T_gaussian` w.r.t. ``Δf`` and ``τ_G``.

    Derived by direct ``d/d(Δf)`` and ``d/dτ_G`` of the closed-form
    integral. Like :func:`h_T_gaussian`, the natural form carries an
    ``exp(β²)`` prefactor that overflows for large ``|Δf|·τ_G``; the
    implementation cancels it analytically against the erf-to-wofz
    rewrite. With ``β = i π Δf τ_G``, ``T̃ = T/τ_G``, ``W_L = wofz(iβ)``,
    ``W_H = wofz(i(T̃ + β))``, ``D = exp(-T̃²)``, ``Φ = exp(-i 2π Δf T)``,
    the stable forms are

        E·(P−Q)  = W_L − D·Φ·W_H
        E·e_P    = D·Φ
        E·e_Q    = 1

        dh/d(Δf) = (τ_G √π / 2) · (i π τ_G) · {
            2 β (W_L − D·Φ·W_H) + (2/√π) (D·Φ − 1)
        }

        dh/dτ_G  = (√π/2)·(W_L − D·Φ·W_H)
                   − τ_G²·π²·Δf²·√π·(W_L − D·Φ·W_H)
                   − T̃·D·Φ + β·(D·Φ − 1)

    Verified against central finite differences in the test suite to
    relative error < 1e-6 over the operational range.

    Parameters
    ----------
    delta_f_mhz : np.ndarray
        Baseband frequency offset ``Δf`` from line centre, in MHz.
    tau_G_us : float
        Gaussian decay time constant ``τ_G`` in microseconds (``> 0``).
    acquisition_us : float
        Active acquisition length ``T`` in microseconds (``> 0``).

    Returns
    -------
    tuple of np.ndarray
        ``(dh/d(Δf_MHz), dh/d(τ_G_us))``, both complex arrays shaped like
        ``delta_f_mhz``.

    Raises
    ------
    ValueError
        If ``tau_G_us`` or ``acquisition_us`` is not positive.
    """
    if tau_G_us <= 0.0:
        raise ValueError("tau_G_us must be positive")
    if acquisition_us <= 0.0:
        raise ValueError("acquisition_us must be positive")
    df = np.asarray(delta_f_mhz, dtype=float)
    beta = 1j * np.pi * df * tau_G_us
    t_over_tau = acquisition_us / tau_G_us
    w_lo = wofz(1j * beta)
    w_hi = wofz(1j * (t_over_tau + beta))
    damping = np.exp(-(t_over_tau**2))
    phase = np.exp(-1j * 2.0 * np.pi * df * acquisition_us)
    sqrt_pi = np.sqrt(np.pi)

    e_pq = w_lo - damping * phase * w_hi  # exp(β²) · (P − Q), stable
    e_diff = damping * phase - 1.0  # exp(β²) · (e_P − e_Q)

    dh_ddf = (
        (tau_G_us * sqrt_pi / 2.0)
        * (1j * np.pi * tau_G_us)
        * (2.0 * beta * e_pq + (2.0 / sqrt_pi) * e_diff)
    )

    dh_dtau = (
        (sqrt_pi / 2.0) * e_pq
        - (tau_G_us**2) * (np.pi**2) * (df**2) * sqrt_pi * e_pq
        - t_over_tau * damping * phase
        + beta * e_diff
    )

    return (
        cast(np.ndarray, dh_ddf.astype(np.complex128)),
        cast(np.ndarray, dh_dtau.astype(np.complex128)),
    )


def effective_tau_gaussian(tau_G_us: float, acquisition_us: float) -> float:
    """On-line gain ``τ_eff = (τ_G √π / 2) · erf(T/τ_G)`` for the Gaussian.

    The Gaussian analog of :func:`effective_tau`: the (real) value of
    :func:`h_T_gaussian` at ``Δf = 0``. In the long-decay limit ``τ_G → ∞``
    it tends to ``T`` (boxcar acquisition); in the short-decay limit
    ``τ_G → 0`` it tends to ``(τ_G √π / 2)`` (the full Gaussian integral).

    Parameters
    ----------
    tau_G_us : float
        Gaussian decay time constant ``τ_G`` in microseconds (``> 0``).
    acquisition_us : float
        Active acquisition length ``T`` in microseconds (``> 0``).

    Returns
    -------
    float
        ``τ_eff`` in microseconds.

    Raises
    ------
    ValueError
        If ``tau_G_us`` or ``acquisition_us`` is not positive.
    """
    if tau_G_us <= 0.0:
        raise ValueError("tau_G_us must be positive")
    if acquisition_us <= 0.0:
        raise ValueError("acquisition_us must be positive")
    # erf is real for real argument; scipy.special.erf takes real input fine.
    from scipy.special import erf as _real_erf  # local import: only this path

    return float(tau_G_us * np.sqrt(np.pi) / 2.0 * _real_erf(acquisition_us / tau_G_us))


# ---------------------------------------------------------------------------
# Shape-aware dispatchers
# ---------------------------------------------------------------------------
def h_T_shape(
    shape: "PeakShape | str",
    delta_f_mhz: np.ndarray,
    tau_us: float,
    acquisition_us: float,
) -> np.ndarray:
    """Route to :func:`h_T` or :func:`h_T_gaussian` by shape.

    ``tau_us`` carries ``τ`` for ``PeakShape.LORENTZIAN`` and ``τ_G`` for
    ``PeakShape.GAUSSIAN``; both shapes are single-parameter in the decay
    constant.
    """
    s = PeakShape.coerce(shape)
    if s is PeakShape.LORENTZIAN:
        return h_T(delta_f_mhz, tau_us, acquisition_us)
    return h_T_gaussian(delta_f_mhz, tau_us, acquisition_us)


def h_T_shape_jacobian(
    shape: "PeakShape | str",
    delta_f_mhz: np.ndarray,
    tau_us: float,
    acquisition_us: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Route to :func:`h_T_jacobian` or :func:`h_T_gaussian_jacobian` by shape."""
    s = PeakShape.coerce(shape)
    if s is PeakShape.LORENTZIAN:
        return h_T_jacobian(delta_f_mhz, tau_us, acquisition_us)
    return h_T_gaussian_jacobian(delta_f_mhz, tau_us, acquisition_us)


def effective_tau_shape(
    shape: "PeakShape | str", tau_us: float, acquisition_us: float
) -> float:
    """Route to :func:`effective_tau` or :func:`effective_tau_gaussian` by shape."""
    s = PeakShape.coerce(shape)
    if s is PeakShape.LORENTZIAN:
        return effective_tau(tau_us, acquisition_us)
    return effective_tau_gaussian(tau_us, acquisition_us)


# ---------------------------------------------------------------------------
# The window model
# ---------------------------------------------------------------------------
@dataclass
class ModelPeak:
    """A single line in baseband-offset coordinates.

    The fit is parameterised per line by ``(amplitude, offset_mhz, phase)``:
    a real amplitude ``A``, the *signed* baseband offset ``δ`` from the window
    reference frequency (MHz), and a free phase ``φ`` (radians). The shared
    decay ``τ`` is a window-level parameter, not carried here.

    Attributes
    ----------
    amplitude : float
        Line amplitude ``A`` (non-negative; the on-line response is
        ``½ A τ_eff``).
    offset_mhz : float
        Signed baseband offset ``δ = s·(f - f_c)`` from the window reference
        frequency, in MHz.
    phase : float
        Line phase ``φ`` in radians.
    """

    amplitude: float
    offset_mhz: float
    phase: float


def model_spectrum(
    offset_grid_mhz: np.ndarray,
    peaks: Sequence[ModelPeak],
    tau_us: float,
    acquisition_us: float,
    *,
    shape: "PeakShape | str" = PeakShape.LORENTZIAN,
) -> np.ndarray:
    """Sum of finite-T damped-cosine responses on a baseband-offset grid.

    Evaluates the window model

        model(u) = Σ_j ½ A_j e^{iφ_j} h_T(u - δ_j; τ)

    on the offset grid ``u`` (MHz). Free peaks and frozen fixed contributors
    are both just :class:`ModelPeak` entries -- pass them in one list. All
    lines share the decay ``τ`` and the shape ``shape``.

    Parameters
    ----------
    offset_grid_mhz : np.ndarray
        Baseband-offset grid ``u`` for the window, in MHz (see
        :func:`baseband_offset`).
    peaks : sequence of ModelPeak
        The lines to sum. An empty sequence yields an all-zero spectrum.
    tau_us : float
        Shared decay time constant in microseconds (``> 0``). For
        ``shape=LORENTZIAN`` this is ``τ`` (exponential decay); for
        ``shape=GAUSSIAN`` this is ``τ_G`` (Gaussian decay).
    acquisition_us : float
        Active acquisition length ``T`` in microseconds (``> 0``).
    shape : PeakShape or str, default LORENTZIAN
        Line-shape selector. Routes the per-peak ``h_T`` evaluation through
        :func:`h_T` (Lorentzian) or :func:`h_T_gaussian`.

    Returns
    -------
    np.ndarray
        Complex model spectrum on ``offset_grid_mhz``.

    Raises
    ------
    ValueError
        If ``tau_us`` or ``acquisition_us`` is not positive.
    """
    if tau_us <= 0.0:
        raise ValueError("tau_us must be positive")
    if acquisition_us <= 0.0:
        raise ValueError("acquisition_us must be positive")

    s = PeakShape.coerce(shape)
    u = np.asarray(offset_grid_mhz, dtype=float)
    spectrum = np.zeros(u.shape, dtype=np.complex128)
    for pk in peaks:
        phasor = 0.5 * pk.amplitude * np.exp(1j * pk.phase)
        spectrum += phasor * h_T_shape(s, u - pk.offset_mhz, tau_us, acquisition_us)
    return cast(np.ndarray, spectrum)


# ---------------------------------------------------------------------------
# The demodulation / sideband mapping
# ---------------------------------------------------------------------------
def baseband_offset(
    freq_mhz: np.ndarray,
    center_mhz: float,
    sideband: SidebandLike,
) -> np.ndarray:
    """Convert a molecular frequency grid to the signed baseband offset ``u``.

    ``u = s·(f - f_c)`` with the sideband sign ``s`` from
    :func:`sideband_sign`. This is the coordinate the window fit runs in; it
    is paired with :func:`molecular_frequency`, its exact inverse.

    Parameters
    ----------
    freq_mhz : np.ndarray
        Molecular frequency grid (MHz). Ascending or descending order is fine;
        the conversion is element-wise.
    center_mhz : float
        Window reference (molecular) frequency ``f_c`` in MHz.
    sideband : Sideband or str
        Sideband configuration.

    Returns
    -------
    np.ndarray
        Signed baseband offset ``u`` (MHz), shaped like ``freq_mhz``.
    """
    s = sideband_sign(sideband)
    f = np.asarray(freq_mhz, dtype=float)
    return cast(np.ndarray, s * (f - center_mhz))


def molecular_frequency(
    offset_mhz: np.ndarray,
    center_mhz: float,
    sideband: SidebandLike,
) -> np.ndarray:
    """Convert a signed baseband offset back to molecular frequency.

    Inverts :func:`baseband_offset`: ``f = f_c + s·δ`` (since ``s = ±1``,
    ``1/s = s``). Used to map fitted offsets back onto the molecular axis.

    Parameters
    ----------
    offset_mhz : np.ndarray
        Signed baseband offset ``δ`` (MHz). Scalars are accepted.
    center_mhz : float
        Window reference (molecular) frequency ``f_c`` in MHz.
    sideband : Sideband or str
        Sideband configuration.

    Returns
    -------
    np.ndarray
        Molecular frequency ``f`` (MHz), shaped like ``offset_mhz``.
    """
    s = sideband_sign(sideband)
    delta = np.asarray(offset_mhz, dtype=float)
    return cast(np.ndarray, center_mhz + s * delta)


def to_baseband_offset(
    freq_mhz: np.ndarray,
    complex_spectrum: np.ndarray,
    *,
    center_mhz: float,
    sideband: SidebandLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Move an active-FT window into the fit frame: grid conversion only.

    Converts the molecular grid to the signed baseband offset ``u`` with
    :func:`baseband_offset` and returns the spectrum unchanged. The active-FT
    (:mod:`ftmwpipeline.fitting.active_ft`) is already in the ``[0, T]`` form
    ``h_T`` models -- there is no turn-on phase ramp to remove, so this is
    just a grid relabel.

    The point correspondence is preserved -- ``offset[i]`` and
    ``spectrum[i]`` describe the same bin -- but the grid is **not** reordered
    (it may be descending, as 2638's is). Callers that need a monotone grid
    should sort the returned pair together.

    Parameters
    ----------
    freq_mhz : np.ndarray
        Molecular frequency grid of the window (MHz), 1-D.
    complex_spectrum : np.ndarray
        Active-FT complex spectrum on ``freq_mhz``, same shape.
    center_mhz : float
        Window reference (molecular) frequency ``f_c`` in MHz.
    sideband : Sideband or str
        Sideband configuration.

    Returns
    -------
    tuple of np.ndarray
        ``(offset_grid_mhz, complex_spectrum)`` -- the ``[0, T]``-frame data
        ready for :func:`model_spectrum`.

    Raises
    ------
    ValueError
        If ``freq_mhz`` and ``complex_spectrum`` differ in shape.
    """
    freq = np.asarray(freq_mhz, dtype=float)
    z = np.asarray(complex_spectrum, dtype=np.complex128)
    if freq.shape != z.shape:
        raise ValueError("freq_mhz and complex_spectrum must have equal shape")
    offset_grid = baseband_offset(freq, center_mhz, sideband)
    return offset_grid, z
