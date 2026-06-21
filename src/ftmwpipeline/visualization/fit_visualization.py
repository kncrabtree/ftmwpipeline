"""
Stage 5 per-window fit diagnostic plots.

Two modes:

* **Overview** (``window_id=None``): the persisted high-resolution magnitude
  spectrum overlaid with the fitted model (sum of every window's
  contribution), with each fit window's span shaded. The model is
  re-evaluated on the persisted grid via :func:`model_spectrum`, which is
  grid-agnostic, so the display grid is independent of the fit-time active-FT.
* **Per-window detail** (``window_id=int``): four panels for one window --
  real and imaginary parts of the model-on-data with their residuals, the
  magnitude with its residual, the time-domain envelope (intuition only),
  and a compact rendering of the conservative add-one-peak audit trail.

The time-domain envelope is a synthesized ``A_eff(t) = sum_j 0.5 A_j
exp(-(t - t0)/tau)`` envelope -- an intuition aid alongside the
residual-on-data view, which carries the diagnostic weight.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union, cast

import matplotlib.pyplot as plt
import numpy as np

from ..core.data_structures import FittingResult, Sideband, SpectrumFit
from ..fitting.peak_model import ModelPeak, model_spectrum, sideband_sign
from .report_style import (
    AGGIE_BLUE,
    AGGIE_GOLD,
    DOUBLE_DECKER,
    PINOT,
    apply_bare_style,
)

SidebandLike = Union[Sideband, str]


def _persisted_phase_ramp(
    frequencies: np.ndarray,
    sideband: SidebandLike,
    probe_freq_mhz: float,
    start_us: float,
) -> np.ndarray:
    """Per-bin phase factor that converts the active-FT phase frame to the
    persisted-FT phase frame.

    The active-FT lives in the de-ramped ``[0, T]`` frame (its first sample
    is the FID at ``t = start_us``). The persisted FT is the rfft of the
    full FID and lives in the ``[0, T_full]`` frame (its first sample is the
    FID at ``t = 0``). A line at baseband frequency ``f_bb`` picks up an
    ``exp(-i 2 pi f_bb start_us)`` phase between the two frames; multiplying
    the active-frame model by this factor puts it on the persisted-frame
    phase axis so the overlay matches the data's complex parts (not just
    its magnitude). Returns ``ones`` when ``start_us == 0``.
    """
    if start_us == 0.0:
        return cast(
            np.ndarray, np.ones(np.asarray(frequencies).shape, dtype=np.complex128)
        )
    s = sideband_sign(sideband)
    f_bb = s * (np.asarray(frequencies, dtype=float) - float(probe_freq_mhz))
    return cast(np.ndarray, np.exp(-1j * 2.0 * np.pi * f_bb * float(start_us)))


def _window_model_on_persisted_grid(
    frequencies: np.ndarray,
    fit: SpectrumFit,
    sideband: SidebandLike,
    acquisition_us: float,
    model_amplitude_scale: float = 1.0,
    persisted_phase_ramp: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Sum the fitted model over every window, on the persisted grid.

    Each window's per-peak ``(amplitude, frequency_mhz, phase)`` and shared
    ``tau_us`` are converted into the window's signed baseband-offset
    parameterisation, then :func:`model_spectrum` is evaluated on the
    persisted-grid offsets (``u = s*(f - f_c)``). Frequencies outside any
    window contribute zero -- but the leakage skirt of each fitted line
    naturally reaches across the persisted grid through the closed-form
    ``h_T``.

    ``model_amplitude_scale`` converts amplitudes from the fit-time active-FT
    amplitude units (``dt_us * rfft(active)``) to the persisted-FT units
    (``rfft(padded) / original_length * 10**units_power``). The scale is
    ``10**units_power / (original_length * dt_us)`` for the canonical Stage
    1 pipeline; the visualization caller computes it from the FID and
    persisted-FT metadata.

    ``persisted_phase_ramp`` (per-bin complex unit-modulus factor) converts
    the active-FT ``[0, T]`` phase frame to the persisted-FT
    ``[0, T_full]`` phase frame so the overlay matches the data's complex
    parts. See :func:`_persisted_phase_ramp`.
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
    if model_amplitude_scale != 1.0:
        total *= model_amplitude_scale
    if persisted_phase_ramp is not None:
        total *= persisted_phase_ramp
    return cast(np.ndarray, total)


def _shade_windows(ax: plt.Axes, fit: SpectrumFit) -> None:
    for window_fit in fit.window_fits:
        if window_fit.window is None:
            continue
        lo, hi = window_fit.window.freq_range
        ax.axvspan(lo, hi, color=AGGIE_GOLD, alpha=0.30, linewidth=0)


def _plot_overview(
    frequencies: np.ndarray,
    complex_spectrum: np.ndarray,
    rms_noise: np.ndarray,
    fit: SpectrumFit,
    sideband: SidebandLike,
    acquisition_us: float,
    figsize: Tuple[float, float],
    title: str,
    model_amplitude_scale: float = 1.0,
    persisted_phase_ramp: Optional[np.ndarray] = None,
) -> plt.Figure:
    """Two-panel overview: spectrum + model overlay; magnitude residual."""
    model = _window_model_on_persisted_grid(
        frequencies,
        fit,
        sideband,
        acquisition_us,
        model_amplitude_scale,
        persisted_phase_ramp,
    )
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax_top.plot(
        frequencies,
        np.abs(complex_spectrum),
        color=DOUBLE_DECKER,
        lw=1.0,
        label="data |X|",
    )
    ax_top.plot(frequencies, np.abs(model), color=AGGIE_BLUE, lw=0.9, label="model |X|")
    _shade_windows(ax_top, fit)
    ax_top.set_ylabel("|X(f)|")
    ax_top.set_title(title)
    ax_top.legend(loc="upper right", fontsize=8)
    apply_bare_style(ax_top)

    residual_mag = np.abs(complex_spectrum - model)
    ax_bot.plot(frequencies, residual_mag, color=PINOT, lw=0.6, label="|residual|")
    ax_bot.plot(
        frequencies,
        rms_noise,
        color="0.35",
        lw=0.7,
        ls="--",
        label="canonical sigma",
    )
    _shade_windows(ax_bot, fit)
    ax_bot.set_xlabel("frequency (MHz)")
    ax_bot.set_ylabel("|residual|")
    ax_bot.legend(loc="upper right", fontsize=8)
    apply_bare_style(ax_bot)
    fig.tight_layout()
    return fig


_AUDIT_COLORS = {
    "seed": "tab:blue",
    "seed-blend": "tab:cyan",
    "accept": "tab:green",
    "promote": "tab:olive",
    "tentative": "0.6",
    "reject": "tab:red",
    "knockout-null": "tab:purple",
}


def _plot_audit_trail_on_freq(
    ax: plt.Axes,
    fit_window: FittingResult,
    sideband: SidebandLike,
    center_mhz: float,
    freq_lo: float,
    freq_hi: float,
) -> None:
    """Audit trail on the molecular-frequency axis.

    Each step is a horizontal bar at ``y = step_index`` from the window
    center to the candidate's molecular frequency (so the bar's terminus is
    the line position on the same axis as the spectra above). Color =
    decision, with a short label ``decision (p=...)``.
    """
    audit = fit_window.audit_trail
    if not audit:
        ax.text(
            0.5,
            0.5,
            "no audit trail",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xlim(freq_lo, freq_hi)
        ax.set_xlabel("frequency (MHz)")
        ax.set_yticks([])
        return
    s = sideband_sign(sideband)
    window_width = freq_hi - freq_lo
    for i, step in enumerate(audit):
        candidate_mhz = center_mhz + s * step.candidate_offset_mhz
        # Bar from center to candidate, oriented along frequency axis.
        ax.plot(
            [center_mhz, candidate_mhz],
            [i, i],
            color=_AUDIT_COLORS.get(step.decision, "0.4"),
            lw=3.0,
            solid_capstyle="butt",
        )
        ax.plot(
            [candidate_mhz],
            [i],
            marker="o",
            markersize=5,
            color=_AUDIT_COLORS.get(step.decision, "0.4"),
        )
        # Text label sits just below the marker. Horizontal alignment
        # follows which third of the window the marker is in, so the text
        # block always grows toward the window's interior and doesn't spill
        # past the axis edge.
        if window_width > 0:
            frac = (candidate_mhz - freq_lo) / window_width
        else:
            frac = 0.5
        if frac < 1.0 / 3.0:
            ha = "left"
        elif frac > 2.0 / 3.0:
            ha = "right"
        else:
            ha = "center"
        ax.text(
            candidate_mhz,
            i - 0.3,
            f"{step.decision} (p={step.p_value:.1e})",
            va="top",
            ha=ha,
            fontsize=7,
        )
    ax.axvline(center_mhz, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xlim(freq_lo, freq_hi)
    # Extra bottom headroom for the text under the lowest marker.
    ax.set_ylim(-0.9, len(audit) - 0.5)
    ax.set_yticks(range(len(audit)))
    ax.set_yticklabels([f"step {i}" for i in range(len(audit))], fontsize=7)
    ax.set_xlabel("frequency (MHz)")
    ax.set_title("audit trail", fontsize=9)


def _plot_residual_histogram(
    ax: plt.Axes, residual: np.ndarray, sigma_slice: np.ndarray
) -> None:
    """Histogram of ``|residual|`` overlaid with the theoretical Rayleigh PDF.

    Complex Gaussian noise with per-bin complex RMS ``sigma`` has per-real-
    component standard deviation ``sigma_c = sigma / sqrt(2)``, and ``|noise|``
    is Rayleigh-distributed with scale ``sigma_c``. Deviations from the
    Rayleigh curve flag heavy-tail / mis-fit residual.
    """
    mag = np.abs(residual)
    sigma_c = float(np.median(sigma_slice)) / np.sqrt(2.0)
    if sigma_c <= 0.0 or mag.size == 0:
        ax.text(
            0.5,
            0.5,
            "no residual data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return
    n_bins = max(10, min(50, mag.size // 4))
    ax.hist(
        mag,
        bins=n_bins,
        density=True,
        color="0.7",
        edgecolor="0.3",
        linewidth=0.5,
        label="|residual|",
    )
    x_max = max(float(mag.max()), 5.0 * sigma_c)
    x = np.linspace(0.0, x_max, 400)
    rayleigh = (x / (sigma_c**2)) * np.exp(-(x**2) / (2.0 * sigma_c**2))
    ax.plot(
        x, rayleigh, color="tab:purple", lw=1.2, label=r"Rayleigh($\sigma/\sqrt{2}$)"
    )
    ax.axvline(
        3.0 * sigma_c,
        color="tab:red",
        lw=0.7,
        ls="--",
        label=r"$3\sigma_c$ (~99%)",
    )
    ax.set_xlabel("|residual|")
    ax.set_ylabel("density")
    ax.set_title("|residual| histogram vs noise", fontsize=9)
    ax.legend(loc="upper right", fontsize=7)


def _plot_data_overlay(
    ax: plt.Axes,
    f_slice: np.ndarray,
    data: np.ndarray,
    model: np.ndarray,
    model_color: str,
) -> None:
    """Data (black markers + faint connecting lines) and model on one axis."""
    ax.plot(
        f_slice,
        data,
        color="#00000044",
        lw=0.7,
        zorder=1,
    )
    ax.plot(
        f_slice,
        data,
        marker="o",
        linestyle="None",
        markersize=3,
        markerfacecolor="black",
        markeredgecolor="black",
        zorder=2,
    )
    ax.plot(
        f_slice,
        model,
        color=model_color,
        lw=1.2,
        zorder=3,
    )


def _plot_residual_with_band(
    ax: plt.Axes,
    f_slice: np.ndarray,
    residual: np.ndarray,
    band: float,
    color: str,
    is_magnitude: bool,
) -> None:
    """Residual line + horizontal ±band reference for re/im (or single +band for mag).

    Re/Im panels also draw a zero line to guide the eye.
    """
    if not is_magnitude:
        ax.axhline(0.0, color="0.5", lw=0.5)
    ax.plot(f_slice, residual, color=color, lw=0.8)
    if band > 0.0:
        ax.axhline(band, color="0.3", lw=0.6, ls="--")
        if is_magnitude:
            ax.text(
                float(f_slice[0]),
                band,
                " 3σ_c (~99%)",
                fontsize=7,
                va="bottom",
                ha="left",
                color="0.3",
            )
        else:
            ax.axhline(-band, color="0.3", lw=0.6, ls="--")
            ax.text(
                float(f_slice[0]),
                band,
                " ±3σ_c",
                fontsize=7,
                va="bottom",
                ha="left",
                color="0.3",
            )


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
    model_amplitude_scale: float = 1.0,
    persisted_phase_ramp: Optional[np.ndarray] = None,
) -> plt.Figure:
    """Per-window diagnostic: 3x2 grid of Re/Im/Mag (data+model | residual+band)
    plus a 4th row of audit trail on the freq axis and a |residual| histogram
    against the Rayleigh noise model."""
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
    # Add the frozen-contributor leakage: each FixedContributor names a
    # strong line fit freely in its primary window; the converted
    # ``fixed_parameters`` dict carries the line's refined molecular freq
    # plus amplitude / phase from the primary fit. We evaluate them in this
    # window's offset frame using THIS window's tau -- matching
    # ``subtract_frozen_background``, which the fit itself uses to build
    # the background it subtracts from the data.
    frozen_peaks: list[ModelPeak] = []
    for key, fp_data in window_fit.fixed_parameters.items():
        if not key.startswith("frozen_peak_"):
            continue
        contrib_freq = float(fp_data["frequency_mhz"])
        contrib_amp = float(fp_data["amplitude"])
        contrib_phase = float(fp_data.get("phase", 0.0) or 0.0)
        frozen_peaks.append(
            ModelPeak(
                amplitude=contrib_amp,
                offset_mhz=float(s * (contrib_freq - center)),
                phase=contrib_phase,
            )
        )
    all_peaks = peaks + frozen_peaks
    model_slice = (
        model_spectrum(u_slice, all_peaks, tau_us, acquisition_us)
        if all_peaks and tau_us > 0
        else np.zeros_like(z_slice)
    )
    if model_amplitude_scale != 1.0:
        model_slice = model_slice * model_amplitude_scale
    if persisted_phase_ramp is not None:
        model_slice = model_slice * persisted_phase_ramp[mask]
    residual = z_slice - model_slice

    # Per-component noise band: sigma_c = sigma_complex / sqrt(2).
    sigma_c_slice = sigma_slice / np.sqrt(2.0)
    band_re_im = 3.0 * float(np.median(sigma_c_slice))
    # Magnitude residual peak-detection threshold: 3*sigma_c covers ~99% of
    # the Rayleigh distribution (CDF at 3 sigma_c is ~0.989).
    band_mag = band_re_im

    # Spectrum + residual + audit panels all share the molecular-frequency
    # x-axis; the histogram is on a different x (|residual|) so it must
    # stay independent. ``sharex="col"`` on plt.subplots would link the
    # hist to the right-column residuals and squash both.
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=10)
    ax_re_data = fig.add_subplot(4, 2, 1)
    ax_re_res = fig.add_subplot(4, 2, 2, sharex=ax_re_data)
    ax_im_data = fig.add_subplot(4, 2, 3, sharex=ax_re_data)
    ax_im_res = fig.add_subplot(4, 2, 4, sharex=ax_re_data)
    ax_mag_data = fig.add_subplot(4, 2, 5, sharex=ax_re_data)
    ax_mag_res = fig.add_subplot(4, 2, 6, sharex=ax_re_data)
    ax_audit = fig.add_subplot(4, 2, 7, sharex=ax_re_data)
    ax_hist = fig.add_subplot(4, 2, 8)
    axes_freq_x = (
        ax_re_data,
        ax_re_res,
        ax_im_data,
        ax_im_res,
        ax_mag_data,
        ax_mag_res,
        ax_audit,
    )

    _plot_data_overlay(
        ax_re_data,
        f_slice,
        np.real(z_slice),
        np.real(model_slice),
        "tab:red",
    )
    _plot_residual_with_band(
        ax_re_res,
        f_slice,
        np.real(residual),
        band_re_im,
        "tab:red",
        is_magnitude=False,
    )
    _plot_data_overlay(
        ax_im_data,
        f_slice,
        np.imag(z_slice),
        np.imag(model_slice),
        "tab:blue",
    )
    _plot_residual_with_band(
        ax_im_res,
        f_slice,
        np.imag(residual),
        band_re_im,
        "tab:blue",
        is_magnitude=False,
    )
    _plot_data_overlay(
        ax_mag_data,
        f_slice,
        np.abs(z_slice),
        np.abs(model_slice),
        "tab:purple",
    )
    _plot_residual_with_band(
        ax_mag_res,
        f_slice,
        np.abs(residual),
        band_mag,
        "tab:purple",
        is_magnitude=True,
    )
    ax_re_data.set_ylabel("Re")
    ax_im_data.set_ylabel("Im")
    ax_mag_data.set_ylabel("|X|")
    ax_re_res.set_ylabel("Re residual")
    ax_im_res.set_ylabel("Im residual")
    ax_mag_res.set_ylabel("|residual|")

    # Column headers (top row only).
    ax_re_data.set_title("spectrum (black: data, color: model)", fontsize=9)
    ax_re_res.set_title("residual (data - model)", fontsize=9)

    _plot_audit_trail_on_freq(
        ax_audit,
        window_fit,
        sideband,
        center,
        float(min(lo, hi)),
        float(max(lo, hi)),
    )
    _plot_residual_histogram(ax_hist, residual, sigma_slice)

    # Hide redundant x-tick labels on the rows that share the freq axis
    # except for the bottom-most (audit) row, which carries the label.
    for ax in (ax_re_data, ax_re_res, ax_im_data, ax_im_res, ax_mag_data, ax_mag_res):
        ax.tick_params(axis="x", labelbottom=False)
    ax_audit.set_xlabel("frequency (MHz)", fontsize=9)
    for ax in axes_freq_x + (ax_hist,):
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
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
    model_amplitude_scale: float = 1.0,
    probe_freq_mhz: Optional[float] = None,
    start_us: float = 0.0,
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
    model_amplitude_scale : float, default 1.0
        Multiplicative factor applied to the fitted model when plotting on
        the persisted-FT grid. The fit's amplitudes are in active-FT units
        (``dt_us * rfft(active)``); converting to the persisted-FT amplitude
        convention (``rfft(padded) / original_length * 10**units_power``)
        requires multiplying by
        ``10**units_power / (original_length * dt_us)``.
    probe_freq_mhz : float, optional
        Probe (LO) frequency in MHz. Required together with ``start_us > 0``
        for the per-bin phase re-roll that maps the fit's active-FT
        ``[0, T]`` phase frame onto the persisted-FT ``[0, T_full]`` frame.
        Without it, complex parts of the model overlay will be off even if
        the magnitude is correct.
    start_us : float, default 0.0
        Active-region start time in microseconds (``[start_us, end_us]`` is
        the active portion of the FID). The model is multiplied by
        ``exp(-i 2 pi f_bb start_us)`` to put it on the persisted-FT phase
        frame. ``0`` skips the re-roll.
    """
    if title is None:
        title = (
            f"Stage 5 fit (window {window_id})"
            if window_id is not None
            else f"Stage 5 fit ({fit.n_windows} windows, "
            f"{fit.n_fitted_peaks} peaks)"
        )
    phase_ramp: Optional[np.ndarray] = None
    if start_us != 0.0:
        if probe_freq_mhz is None:
            raise ValueError(
                "probe_freq_mhz is required when start_us != 0 (the model "
                "phase frame depends on the per-bin baseband frequency)"
            )
        phase_ramp = _persisted_phase_ramp(
            frequencies, sideband, probe_freq_mhz, start_us
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
            model_amplitude_scale,
            phase_ramp,
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
        model_amplitude_scale,
        phase_ramp,
    )
