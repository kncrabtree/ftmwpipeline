"""
Stage 5 per-window fit diagnostic plots.

Two modes:

* **Overview** (``window_id=None``): the persisted high-resolution magnitude
  spectrum overlaid with the fitted model (sum of every window's
  contribution), with each fit window's span shaded. The model is
  re-evaluated on the persisted grid via :func:`model_spectrum` -- the
  active-FT was a fit-time convenience; the model is grid-agnostic.
* **Per-window detail** (``window_id=int``): four panels for one window --
  real and imaginary parts of the model-on-data with their residuals, the
  magnitude with its residual, the time-domain envelope (intuition only),
  and a compact rendering of the conservative add-one-peak audit trail.

The time-domain envelope is a synthesized ``A_eff(t) = sum_j 0.5 A_j
exp(-(t - t0)/tau)`` envelope -- the prototype's "spectrum-IFFT vs the
time-domain model" panel was dropped per the planning doc as adding little
diagnostic value over the residual-on-data view.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union, cast

import matplotlib.pyplot as plt
import numpy as np

from ..core.data_structures import FittingResult, Sideband, SpectrumFit
from ..fitting.peak_model import ModelPeak, model_spectrum, sideband_sign

SidebandLike = Union[Sideband, str]


def _window_model_on_persisted_grid(
    frequencies: np.ndarray,
    fit: SpectrumFit,
    sideband: SidebandLike,
    acquisition_us: float,
) -> np.ndarray:
    """Sum the fitted model over every window, on the persisted grid.

    Each window's per-peak ``(amplitude, frequency_mhz, phase)`` and shared
    ``tau_us`` are converted into the window's signed baseband-offset
    parameterisation, then :func:`model_spectrum` is evaluated on the
    persisted-grid offsets (``u = s*(f - f_c)``). Frequencies outside any
    window contribute zero -- but the leakage skirt of each fitted line
    naturally reaches across the persisted grid through the closed-form
    ``h_T``.
    """
    s = sideband_sign(sideband)
    total = np.zeros(frequencies.shape, dtype=np.complex128)
    f = np.asarray(frequencies, dtype=float)
    for window_fit in fit.window_fits:
        if window_fit.window is None:
            continue
        center = 0.5 * (
            window_fit.window.freq_range[0] + window_fit.window.freq_range[1]
        )
        tau_us = float(window_fit.shared_parameters.get("tau_us", {}).get("value", 0.0))
        if tau_us <= 0:
            continue
        peaks = [
            ModelPeak(
                amplitude=float(p.amplitude),
                offset_mhz=float(s * (p.frequency_mhz - center)),
                phase=float(p.phase if p.phase is not None else 0.0),
            )
            for p in window_fit.fitted_peaks
        ]
        if not peaks:
            continue
        u = s * (f - center)
        total += model_spectrum(u, peaks, tau_us, acquisition_us)
    return cast(np.ndarray, total)


def _shade_windows(ax: plt.Axes, fit: SpectrumFit) -> None:
    for window_fit in fit.window_fits:
        if window_fit.window is None:
            continue
        lo, hi = window_fit.window.freq_range
        ax.axvspan(lo, hi, color="tab:green", alpha=0.08, linewidth=0)


def _plot_overview(
    frequencies: np.ndarray,
    complex_spectrum: np.ndarray,
    rms_noise: np.ndarray,
    fit: SpectrumFit,
    sideband: SidebandLike,
    acquisition_us: float,
    figsize: Tuple[float, float],
    title: str,
) -> plt.Figure:
    """Two-panel overview: spectrum + model overlay; magnitude residual."""
    model = _window_model_on_persisted_grid(frequencies, fit, sideband, acquisition_us)
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax_top.plot(
        frequencies, np.abs(complex_spectrum), color="0.4", lw=0.7, label="data |X|"
    )
    ax_top.plot(
        frequencies, np.abs(model), color="tab:orange", lw=0.9, label="model |X|"
    )
    _shade_windows(ax_top, fit)
    ax_top.set_ylabel("|X(f)|")
    ax_top.set_title(title)
    ax_top.legend(loc="upper right", fontsize=8)

    residual_mag = np.abs(complex_spectrum - model)
    ax_bot.plot(frequencies, residual_mag, color="tab:red", lw=0.6, label="|residual|")
    ax_bot.plot(
        frequencies, rms_noise, color="0.4", lw=0.6, ls="--", label="canonical sigma"
    )
    _shade_windows(ax_bot, fit)
    ax_bot.set_xlabel("frequency (MHz)")
    ax_bot.set_ylabel("|residual|")
    ax_bot.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def _plot_audit_trail(ax: plt.Axes, fit_window: FittingResult) -> None:
    """Compact rendering of the per-iteration conservative-loop decisions."""
    audit = fit_window.audit_trail
    if not audit:
        ax.text(0.5, 0.5, "no audit trail", ha="center", va="center")
        ax.set_axis_off()
        return
    colors = {
        "seed": "tab:blue",
        "seed-blend": "tab:cyan",
        "accept": "tab:green",
        "promote": "tab:olive",
        "tentative": "0.6",
        "reject": "tab:red",
    }
    for i, step in enumerate(audit):
        ax.barh(
            i,
            step.candidate_offset_mhz,
            color=colors.get(step.decision, "0.4"),
            edgecolor="black",
            linewidth=0.4,
        )
        label = step.decision
        ax.text(
            step.candidate_offset_mhz,
            i,
            f" {label} (p={step.p_value:.1e})",
            va="center",
            fontsize=7,
        )
    ax.axvline(0.0, color="black", linewidth=0.5)
    ax.set_yticks(range(len(audit)))
    ax.set_yticklabels([f"step {i}" for i in range(len(audit))], fontsize=7)
    ax.set_xlabel("candidate offset (MHz)")
    ax.set_title("audit trail", fontsize=9)


def _plot_time_envelope(
    ax: plt.Axes,
    fit_window: FittingResult,
    acquisition_us: float,
    n_samples: int = 256,
) -> None:
    """Synthesised time-domain envelope of the fitted lines (intuition only)."""
    t = np.linspace(0.0, acquisition_us, n_samples)
    tau = float(fit_window.shared_parameters.get("tau_us", {}).get("value", 0.0))
    if tau <= 0:
        ax.text(0.5, 0.5, "no tau", ha="center", va="center")
        ax.set_axis_off()
        return
    envelope = np.zeros_like(t)
    for peak in fit_window.fitted_peaks:
        envelope += 0.5 * float(peak.amplitude) * np.exp(-t / tau)
    ax.plot(t, envelope, color="tab:purple", lw=0.9)
    ax.set_xlabel("t (us)")
    ax.set_ylabel("|envelope|")
    ax.set_title("time-domain envelope (model)", fontsize=9)


def _plot_per_window_detail(
    frequencies: np.ndarray,
    complex_spectrum: np.ndarray,
    rms_noise: np.ndarray,
    fit: SpectrumFit,
    window_id: int,
    sideband: SidebandLike,
    acquisition_us: float,
    figsize: Tuple[float, float],
    title: str,
) -> plt.Figure:
    """Five panels for one window: re/im model+residual, magnitude+residual,
    audit trail, time envelope."""
    window_fit = fit.window_fit(window_id)
    if window_fit.window is None:
        raise ValueError(
            f"window {window_id} has no attached SpectralWindow -- the fit was "
            "loaded but the window context is missing (cannot plot detail)"
        )
    s = sideband_sign(sideband)
    lo, hi = window_fit.window.freq_range
    mask = (frequencies >= min(lo, hi)) & (frequencies <= max(lo, hi))
    f_slice = frequencies[mask]
    z_slice = complex_spectrum[mask]
    sigma_slice = rms_noise[mask]
    center = 0.5 * (lo + hi)
    u_slice = s * (f_slice - center)

    tau_us = float(window_fit.shared_parameters.get("tau_us", {}).get("value", 0.0))
    peaks = [
        ModelPeak(
            amplitude=float(p.amplitude),
            offset_mhz=float(s * (p.frequency_mhz - center)),
            phase=float(p.phase if p.phase is not None else 0.0),
        )
        for p in window_fit.fitted_peaks
    ]
    model_slice = (
        model_spectrum(u_slice, peaks, tau_us, acquisition_us)
        if peaks and tau_us > 0
        else np.zeros_like(z_slice)
    )
    residual = z_slice - model_slice

    fig = plt.figure(figsize=figsize)
    fig.suptitle(title)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
    ax_re = fig.add_subplot(gs[0, 0])
    ax_im = fig.add_subplot(gs[0, 1])
    ax_mag = fig.add_subplot(gs[1, 0])
    ax_env = fig.add_subplot(gs[1, 1])
    ax_audit = fig.add_subplot(gs[2, :])

    ax_re.plot(f_slice, np.real(z_slice), color="0.4", lw=0.6, label="data Re")
    ax_re.plot(
        f_slice, np.real(model_slice), color="tab:orange", lw=0.9, label="model Re"
    )
    ax_re.plot(f_slice, np.real(residual), color="tab:red", lw=0.5, label="residual Re")
    ax_re.set_xlabel("frequency (MHz)")
    ax_re.set_title("Re(X)", fontsize=9)
    ax_re.legend(loc="best", fontsize=7)

    ax_im.plot(f_slice, np.imag(z_slice), color="0.4", lw=0.6, label="data Im")
    ax_im.plot(
        f_slice, np.imag(model_slice), color="tab:orange", lw=0.9, label="model Im"
    )
    ax_im.plot(f_slice, np.imag(residual), color="tab:red", lw=0.5, label="residual Im")
    ax_im.set_xlabel("frequency (MHz)")
    ax_im.set_title("Im(X)", fontsize=9)
    ax_im.legend(loc="best", fontsize=7)

    ax_mag.plot(f_slice, np.abs(z_slice), color="0.4", lw=0.6, label="|data|")
    ax_mag.plot(
        f_slice, np.abs(model_slice), color="tab:orange", lw=0.9, label="|model|"
    )
    ax_mag.plot(f_slice, np.abs(residual), color="tab:red", lw=0.5, label="|residual|")
    ax_mag.plot(f_slice, sigma_slice, color="0.4", ls="--", lw=0.5, label="sigma")
    ax_mag.set_xlabel("frequency (MHz)")
    ax_mag.set_title("|X|", fontsize=9)
    ax_mag.legend(loc="best", fontsize=7)

    _plot_time_envelope(ax_env, window_fit, acquisition_us)
    _plot_audit_trail(ax_audit, window_fit)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


def plot_spectrum_fit(
    frequencies: np.ndarray,
    complex_spectrum: np.ndarray,
    rms_noise: np.ndarray,
    fit: SpectrumFit,
    sideband: SidebandLike,
    acquisition_us: float,
    figsize: Tuple[float, float] = (16, 10),
    title: Optional[str] = None,
    window_id: Optional[int] = None,
    backend: str = "matplotlib",
) -> plt.Figure:
    """Plot a Stage 5 fit -- overview or per-window detail.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency axis (MHz) of the persisted user spectrum.
    complex_spectrum : np.ndarray
        Complex FT of the user spectrum.
    rms_noise : np.ndarray
        Per-bin canonical noise (Stage 2) on the user grid.
    fit : SpectrumFit
        The persistent fit aggregate (loaded from ``/stage5_fitting``).
    sideband : Sideband or str
        Pipeline sideband.
    acquisition_us : float
        Active acquisition length ``T`` (microseconds).
    figsize : tuple of float, default ``(16, 10)``
        Figure size in inches.
    title : str, optional
        Custom plot title.
    window_id : int, optional
        When set, draw a per-window detail figure for the given window
        instead of the spectrum-wide overview.
    backend : str, default ``"matplotlib"``
        Plotting backend (only ``"matplotlib"`` supported today).
    """
    if backend != "matplotlib":
        raise NotImplementedError(
            f"backend {backend!r} is not implemented for fit visualization; "
            "only 'matplotlib' is supported"
        )
    if title is None:
        title = (
            f"Stage 5 fit (window {window_id})"
            if window_id is not None
            else f"Stage 5 fit ({fit.n_windows} windows, "
            f"{fit.n_fitted_peaks} peaks)"
        )
    if window_id is None:
        return _plot_overview(
            frequencies,
            complex_spectrum,
            rms_noise,
            fit,
            sideband,
            acquisition_us,
            figsize,
            title,
        )
    return _plot_per_window_detail(
        frequencies,
        complex_spectrum,
        rms_noise,
        fit,
        window_id,
        sideband,
        acquisition_us,
        figsize,
        title,
    )
