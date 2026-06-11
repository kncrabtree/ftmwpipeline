"""
Consolidated Stage 5 per-window detail figure.

A single landscape-letter figure summarising one fit window:

* Row 1 -- full-spectrum context (data magnitude) with the window highlighted.
* Row 2 -- Re / Im / |residual| panels with a vertical line at every fitted
  peak's molecular frequency. All three are on the fit's native grid so
  |residual| = sqrt(Re^2 + Im^2) bin-for-bin.
* Row 3 -- Re / Im / |z| data + model overlays. The model is drawn on an
  oversampled grid (smooth analytic line shape) *and* as small ``x`` markers at
  the native data bins, so the eye compares model-at-bin with data-at-bin and
  the smooth curve is never mistaken for a denser fit.
* Row 4 -- |residual| histogram against the Rayleigh noise model (left) and a
  fitted-peak table with PDG-style uncertainties (right).

Magnitude honesty
-----------------
The magnitude *data* (``|X|``, Row 3) is drawn on an exactly-2x zero-filled grid.
For a magnitude spectrum this is information-faithful, not cosmetic: the magnitude
operation discards the phase, and the half-bin magnitudes of a single zero-fill
recover it (Marshall & Verdun). The model in that overlay is its smooth analytic
curve, so the eye sees the data's between-bin truncation ringing against the
clean model -- the point of the panel.

The |residual| panel (Row 2), by contrast, stays on the native grid alongside
Re / Im, so |residual| = sqrt(Re^2 + Im^2) bin-for-bin. It must not reuse the 2x
data grid: subtracting the smooth analytic model from the sinc-interpolated data
off the native bins plots the data ringing the model lacks, a spurious magnitude
residual that can dwarf the true one near a strong line. Every *statistic* -- the
complex residual, the per-bin noise band, and the |residual| histogram -- is
native.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from ..core.data_structures import FittedPeak, FittingResult, Sideband
from ..fitting.peak_model import ModelPeak, model_spectrum, sideband_sign
from ..utils.signal_processing import APODIZATION_EXAMPLES, make_apodization

SidebandLike = Union[Sideband, str]

# Model curves are evaluated on a grid this many times finer than the native
# data grid so a narrow line renders as a smooth analytic shape rather than a
# few-point polyline. Display only -- every statistic uses the native grid.
MODEL_OVERSAMPLE = 8

# Exactly-2x zero-fill for the magnitude panels (the information limit for a
# magnitude spectrum; finer is pure interpolation).
DISPLAY_PAD_FACTOR = 2


# ---------------------------------------------------------------------------
# Spectroscopic formatting (PDG-style value(uncertainty))
# ---------------------------------------------------------------------------
def _format_spectroscopic(
    value: Optional[float], err: Optional[float], *, n_digits: int = 2
) -> str:
    """PDG-style ``value(err_digits)`` formatter.

    Uses 2 digits of error by default, bumped to 3 when the 2-digit error
    rounds into the leading-1 decade (10-19), where per-digit precision is
    worst. Value precision is matched to the error's last displayed digit.

    >>> _format_spectroscopic(3.06546, 0.0134)
    '3.0655(134)'
    >>> _format_spectroscopic(3.06546, 0.0234)
    '3.065(23)'
    """
    if value is None or not math.isfinite(value):
        return "-"
    if err is None or not math.isfinite(err) or err <= 0:
        return f"{value:.6g}"
    k = math.floor(math.log10(err))
    last_pos = k - n_digits + 1
    scale = 10.0**last_pos
    err_int = int(round(err / scale))
    if err_int >= 10**n_digits:  # round-up to next order (0.999 -> 100)
        last_pos += 1
        scale = 10.0**last_pos
        err_int = int(round(err / scale))
    if 10 <= err_int < 20:  # PDG leading-1: bump "1X" -> "1XY"
        last_pos -= 1
        scale = 10.0**last_pos
        err_int = int(round(err / scale))
    value_round = round(value / scale) * scale
    decimals = max(0, -last_pos)
    return f"{value_round:.{decimals}f}({err_int})"


def _format_spectroscopic_sci(
    value: Optional[float], err: Optional[float], *, n_digits: int = 2
) -> str:
    """Spectroscopic formatter with automatic scientific notation.

    Falls back to :func:`_format_spectroscopic` (plain decimal) when ``|value|``
    is in ``[1e-3, 1e6)``; otherwise factors out the order of magnitude and
    formats the mantissa spectroscopically.

    >>> _format_spectroscopic_sci(4.18e-6, 3.1e-7)
    '4.18(31)e-06'
    """
    if value is None or not math.isfinite(value):
        return "-"
    if value == 0.0 or 1e-3 <= abs(value) < 1e6:
        return _format_spectroscopic(value, err, n_digits=n_digits)
    exp = int(math.floor(math.log10(abs(value))))
    scale = 10.0**exp
    m_value = value / scale
    m_err = (
        err / scale if (err is not None and math.isfinite(err) and err > 0) else None
    )
    body = _format_spectroscopic(m_value, m_err, n_digits=n_digits)
    return f"{body}e{exp:+03d}"


def _peak_labels(n: int) -> List[str]:
    """Alpha labels: A..Z, then AA..AZ, BA..."""
    out = []
    for i in range(n):
        if i < 26:
            out.append(chr(ord("A") + i))
        else:
            j = i - 26
            out.append(chr(ord("A") + j // 26) + chr(ord("A") + j % 26))
    return out


def _peak_confidence(peak: FittedPeak) -> Optional[float]:
    """Per-peak confidence rating placeholder.

    No principled per-peak confidence statistic exists yet; the column is wired
    here so it can be filled in one place when one does. Returns ``None`` (the
    table renders ``-``) for now.
    """
    return None


def _eval_window_baseline(
    window_fit: FittingResult, u_offset_mhz: np.ndarray
) -> np.ndarray:
    """Evaluate the persisted leakage-wing baseline ``B(u)`` on an offset grid.

    Reads the per-window baseline audit from ``quality_metrics``
    (``baseline_applied`` / ``baseline_order`` / ``baseline_offset_scale`` /
    ``baseline_coeff{k}_re`` / ``baseline_coeff{k}_im``) and returns the complex
    ``B(u) = sum_{k<=p} (a_k + i b_k) (u/u_s)^k`` on the signed baseband offset
    from the window centre. Returns zeros when no baseline fired, so callers can
    add it unconditionally. The Stage 5 fit applies the baseline jointly with
    the de-biased lines, so the faithful plotted model is
    ``model_spectrum(peaks) + B(u)``.
    """
    qa = window_fit.quality_metrics or {}
    u = np.asarray(u_offset_mhz, dtype=float)
    zeros = cast(np.ndarray, np.zeros(u.shape, dtype=np.complex128))
    if float(qa.get("baseline_applied", 0.0)) < 0.5:
        return zeros
    u_s = float(qa.get("baseline_offset_scale", 0.0))
    order = int(qa.get("baseline_order", 0))
    if not u_s > 0.0:
        return zeros
    x = u / u_s
    b = np.zeros(u.shape, dtype=np.complex128)
    for k in range(order + 1):
        a_k = float(qa.get(f"baseline_coeff{k}_re", 0.0))
        b_k = float(qa.get(f"baseline_coeff{k}_im", 0.0))
        b = b + (a_k + 1j * b_k) * x**k
    return cast(np.ndarray, b)


# ---------------------------------------------------------------------------
# Model assembly
# ---------------------------------------------------------------------------
def _window_model_peaks(
    window_fit: FittingResult, sideband: SidebandLike, center_mhz: float
) -> List[ModelPeak]:
    """Fitted peaks + frozen out-of-window contributors as offset ModelPeaks.

    The frozen contributors (``frozen_peak_*`` in ``fixed_parameters``) are the
    strong neighbouring lines whose leakage skirt reaches into this window; the
    fit subtracts them as a fixed background, so a faithful overlay must add
    them back.
    """
    s = sideband_sign(sideband)
    peaks = [
        ModelPeak(
            amplitude=float(p.amplitude),
            offset_mhz=float(s * (p.frequency_mhz - center_mhz)),
            phase=float(p.phase if p.phase is not None else 0.0),
        )
        for p in window_fit.fitted_peaks
    ]
    for key, fp in window_fit.fixed_parameters.items():
        if not key.startswith("frozen_peak_"):
            continue
        peaks.append(
            ModelPeak(
                amplitude=float(fp["amplitude"]),
                offset_mhz=float(s * (float(fp["frequency_mhz"]) - center_mhz)),
                phase=float(fp.get("phase", 0.0) or 0.0),
            )
        )
    return peaks


def _eval_model(
    freqs: np.ndarray,
    window_fit: FittingResult,
    peaks: Sequence[ModelPeak],
    tau_us: float,
    acquisition_us: float,
    sideband: SidebandLike,
    center_mhz: float,
    shape: str,
) -> np.ndarray:
    """Model spectrum (lines + leakage-wing baseline) on a frequency grid."""
    s = sideband_sign(sideband)
    u = s * (np.asarray(freqs, dtype=float) - center_mhz)
    if peaks and tau_us > 0.0:
        model = model_spectrum(u, peaks, tau_us, acquisition_us, shape=shape)
    else:
        model = np.zeros(u.shape, dtype=np.complex128)
    return cast(np.ndarray, model + _eval_window_baseline(window_fit, u))


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------
def plot_consolidated_detail(
    window_fit: FittingResult,
    *,
    frequencies: np.ndarray,
    complex_spectrum: np.ndarray,
    rms_noise: np.ndarray,
    sideband: SidebandLike,
    acquisition_us: float,
    title: str,
    amplitude_scale: float = 1.0,
    units_label: str = "",
    trim_mhz: Optional[Tuple[float, float]] = None,
    freq_padded: Optional[np.ndarray] = None,
    spec_padded: Optional[np.ndarray] = None,
    figsize: Tuple[float, float] = (11, 8.5),
    spurs: Optional[Sequence[dict]] = None,
) -> plt.Figure:
    """Render the consolidated per-window detail figure (see module docstring).

    ``frequencies`` / ``complex_spectrum`` / ``rms_noise`` are the native active
    grid (ascending molecular frequency) the fit lives on. ``freq_padded`` /
    ``spec_padded`` are the exactly-2x zero-filled display grid for the
    magnitude panels; when omitted, the magnitude panels fall back to native.
    ``spurs`` is the fit's gated-spur catalogue
    (``SpectrumFit.diagnostics["gated_spurs"]``: dicts with ``center_mhz``
    and ``source``); in-window entries are marked on the data and residual
    panels so a masked tone is never mistaken for an un-fit line.
    """
    if window_fit.window is None:
        raise ValueError(
            "window_fit has no attached SpectralWindow -- cannot plot detail "
            "(the fit was loaded without window context)"
        )
    s = sideband_sign(sideband)
    lo, hi = window_fit.window.freq_range
    lo_f, hi_f = float(min(lo, hi)), float(max(lo, hi))
    center = 0.5 * (lo + hi)
    shape_str = str(getattr(window_fit, "shape", "lorentzian"))
    tau_us = float(window_fit.shared_parameters.get("tau_us", {}).get("value", 0.0))

    mask = (frequencies >= lo_f) & (frequencies <= hi_f)
    f_slice = frequencies[mask]
    z_slice = complex_spectrum[mask]
    sigma_slice = rms_noise[mask]

    peaks = _window_model_peaks(window_fit, sideband, center)
    model_slice = _eval_model(
        f_slice, window_fit, peaks, tau_us, acquisition_us, sideband, center, shape_str
    )
    residual = z_slice - model_slice  # native -- drives the histogram/stats

    # Fine analytic model for the smooth overlay (display only).
    if f_slice.size >= 2:
        n_fine = (f_slice.size - 1) * MODEL_OVERSAMPLE + 1
        f_fine = np.linspace(float(f_slice.min()), float(f_slice.max()), n_fine)
        model_fine = _eval_model(
            f_fine,
            window_fit,
            peaks,
            tau_us,
            acquisition_us,
            sideband,
            center,
            shape_str,
        )
    else:
        f_fine = f_slice.copy()
        model_fine = model_slice.copy()

    sigma_c_slice = sigma_slice / np.sqrt(2.0)
    band = 3.0 * float(np.median(sigma_c_slice)) if sigma_c_slice.size else 0.0

    amp = float(amplitude_scale)
    usuffix = f" ({units_label})" if units_label else ""

    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=11)
    gs = GridSpec(
        nrows=4,
        ncols=3,
        figure=fig,
        height_ratios=[1.0, 1.6, 1.6, 1.6],
        hspace=0.45,
        wspace=0.30,
        left=0.06,
        right=0.97,
        top=0.92,
        bottom=0.07,
    )
    ax_overview = fig.add_subplot(gs[0, :])
    ax_re_res = fig.add_subplot(gs[1, 0])
    ax_im_res = fig.add_subplot(gs[1, 1], sharex=ax_re_res)
    ax_mag_res = fig.add_subplot(gs[1, 2], sharex=ax_re_res)
    ax_re_dat = fig.add_subplot(gs[2, 0], sharex=ax_re_res)
    ax_im_dat = fig.add_subplot(gs[2, 1], sharex=ax_re_res)
    ax_mag_dat = fig.add_subplot(gs[2, 2], sharex=ax_re_res)
    ax_hist = fig.add_subplot(gs[3, 0])
    ax_peaks = fig.add_subplot(gs[3, 1:])
    ax_peaks.set_axis_off()

    _draw_overview(
        ax_overview, frequencies, complex_spectrum, lo_f, hi_f, amp, usuffix, trim_mhz
    )

    fitted_freqs = [float(p.frequency_mhz) for p in window_fit.fitted_peaks]
    labels = _peak_labels(len(fitted_freqs))
    _vline_plotter = _make_vline_plotter(fitted_freqs, labels, tau_us)

    # Row 2: residuals (native re / im / |residual|), all on the fit's native
    # grid so |residual| = sqrt(Re^2 + Im^2) bin-for-bin and the three panels
    # agree. (The |residual| must NOT be drawn on the 2x display grid: the
    # padded data is the sinc-interpolation of the spectrum, but the analytic
    # model is its smooth closed form, so subtracting them off the native bins
    # plots the data's truncation ringing the model lacks -- a spurious
    # magnitude residual that can dwarf the true one near a strong line. The 2x
    # grid is for the data |X| overlay only, where that ringing is the point.)
    band_s = band * amp

    def _res(ax: plt.Axes, vals: np.ndarray, color: str, mag: bool) -> None:
        if not mag:
            ax.axhline(0.0, color="0.5", lw=0.4)
        ax.plot(f_slice, vals * amp, color=color, lw=0.7)
        if band_s > 0.0:
            ax.axhline(band_s, color="0.3", lw=0.5, ls="--")
            if not mag:
                ax.axhline(-band_s, color="0.3", lw=0.5, ls="--")

    for ax in (ax_re_res, ax_im_res, ax_mag_res):
        _vline_plotter(ax, True)
    _res(ax_re_res, np.real(residual), "tab:red", mag=False)
    _res(ax_im_res, np.imag(residual), "tab:blue", mag=False)
    _res(ax_mag_res, np.abs(residual), "tab:purple", mag=True)
    ax_re_res.set_ylabel(f"Re residual{usuffix}", fontsize=9)
    ax_im_res.set_ylabel(f"Im residual{usuffix}", fontsize=9)
    ax_mag_res.set_ylabel(f"|residual|{usuffix}", fontsize=9)
    for ax in (ax_re_res, ax_im_res, ax_mag_res):
        ax.tick_params(axis="both", labelsize=8)
        ax.tick_params(axis="x", labelbottom=False)

    # Row 3: data + model (native re/im, 2x-zero-filled |X|), with model
    # markers at native bins on every panel.
    for ax in (ax_re_dat, ax_im_dat, ax_mag_dat):
        _vline_plotter(ax, False)
    _draw_data_model(
        ax_re_dat,
        f_slice,
        np.real(z_slice),
        f_fine,
        np.real(model_fine),
        np.real(model_slice),
        "tab:red",
        amp,
    )
    _draw_data_model(
        ax_im_dat,
        f_slice,
        np.imag(z_slice),
        f_fine,
        np.imag(model_fine),
        np.imag(model_slice),
        "tab:blue",
        amp,
    )
    _draw_mag_data(
        ax_mag_dat,
        f_slice,
        np.abs(z_slice),
        f_fine,
        np.abs(model_fine),
        np.abs(model_slice),
        lo_f,
        hi_f,
        amp,
        freq_padded,
        spec_padded,
    )
    ax_re_dat.set_ylabel(f"Re{usuffix}", fontsize=9)
    ax_im_dat.set_ylabel(f"Im{usuffix}", fontsize=9)
    ax_mag_dat.set_ylabel(f"|X|{usuffix}", fontsize=9)
    for ax in (ax_re_dat, ax_im_dat, ax_mag_dat):
        ax.tick_params(axis="both", labelsize=8)
        ax.set_xlabel("frequency (MHz)", fontsize=9)

    # Gated spurs in-window: mark the masked tone on every data/residual
    # panel (it is deliberately absent from the model and excluded from the
    # fit's chi-squared).
    in_window_spurs = [
        sp
        for sp in (spurs or [])
        if lo_f <= float(sp.get("center_mhz", float("nan"))) <= hi_f
    ]
    for i, sp in enumerate(in_window_spurs):
        f_sp = float(sp["center_mhz"])
        for ax in (ax_re_res, ax_im_res, ax_mag_res, ax_re_dat, ax_im_dat):
            ax.axvline(f_sp, color="tab:orange", lw=1.0, ls=":", alpha=0.9, zorder=1)
        ax_mag_dat.axvline(
            f_sp,
            color="tab:orange",
            lw=1.0,
            ls=":",
            alpha=0.9,
            zorder=1,
            label=f"spur ({sp.get('source', '?')})" if i == 0 else None,
        )
    if in_window_spurs:
        ax_mag_dat.legend(loc="upper right", fontsize=7, framealpha=0.85)

    _draw_residual_hist(ax_hist, residual, sigma_slice, amp, usuffix)
    _draw_peak_table(ax_peaks, window_fit.fitted_peaks, labels, amp, units_label)
    return fig


def _make_vline_plotter(
    fitted_freqs: Sequence[float], labels: Sequence[str], tau_us: float
) -> Callable[[plt.Axes, bool], None]:
    """Build the per-peak vertical-line annotator with row-staggered labels."""
    fwhm = 1.0 / (math.pi * tau_us) if tau_us > 0.0 else 0.0
    n_rows = 3
    row_of = [0] * len(fitted_freqs)
    last_in_row = [-math.inf] * n_rows
    for idx in sorted(range(len(fitted_freqs)), key=lambda i: fitted_freqs[i]):
        f_here = fitted_freqs[idx]
        chosen = 0
        if fwhm > 0:
            for r in range(n_rows):
                if f_here - last_in_row[r] >= fwhm:
                    chosen = r
                    break
            else:
                chosen = int(min(range(n_rows), key=lambda r: last_in_row[r]))
        row_of[idx] = chosen
        last_in_row[chosen] = f_here
    y_offsets = [2.0, 12.0, 22.0]

    def _plot(ax: plt.Axes, with_labels: bool) -> None:
        for f_pk in fitted_freqs:
            ax.axvline(f_pk, color="tab:gray", lw=0.7, alpha=0.45, zorder=1)
        if not with_labels:
            return
        for f_pk, lbl, row in zip(fitted_freqs, labels, row_of):
            ax.annotate(
                lbl,
                xy=(f_pk, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(0, y_offsets[row]),
                textcoords="offset points",
                fontsize=7,
                ha="center",
                va="bottom",
                color="0.25",
            )

    return _plot


def _draw_overview(
    ax: plt.Axes,
    frequencies: np.ndarray,
    complex_spectrum: np.ndarray,
    lo_f: float,
    hi_f: float,
    amp: float,
    usuffix: str,
    trim_mhz: Optional[Tuple[float, float]],
) -> None:
    """Row 1: full-spectrum data magnitude with the active window highlighted.

    Data magnitude only -- at full-spectrum scale a fitted line collapses to a
    one-pixel spike over the data spike, so the model overlay adds nothing here
    and is carried by the per-window panels below.
    """
    if trim_mhz is not None:
        t_lo, t_hi = float(min(trim_mhz)), float(max(trim_mhz))
        m = (frequencies >= t_lo) & (frequencies <= t_hi)
        ov_f, ov_d = frequencies[m], complex_spectrum[m]
    else:
        ov_f, ov_d = frequencies, complex_spectrum
    ax.plot(ov_f, np.abs(ov_d) * amp, color="0.3", lw=0.5, label="data |X|")
    ax.axvspan(lo_f, hi_f, color="tab:green", alpha=0.35, zorder=0, label="this window")
    ax.axvline(lo_f, color="tab:green", lw=0.9, alpha=0.75, zorder=1)
    ax.axvline(hi_f, color="tab:green", lw=0.9, alpha=0.75, zorder=1)
    if ov_f.size:
        ax.set_xlim(float(ov_f[0]), float(ov_f[-1]))
    ax.set_ylabel(f"|X(f)|{usuffix}", fontsize=9)
    ax.set_title("full-spectrum context", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)


def _draw_data_model(
    ax: plt.Axes,
    f_slice: np.ndarray,
    data: np.ndarray,
    f_fine: np.ndarray,
    model_fine: np.ndarray,
    model_native: np.ndarray,
    color: str,
    amp: float,
) -> None:
    """Data (markers + faint connector) + smooth model curve + model bin markers."""
    ax.plot(f_slice, data * amp, color="#00000044", lw=0.5, zorder=1)
    ax.plot(
        f_slice,
        data * amp,
        marker="o",
        linestyle="None",
        markersize=2.0,
        markerfacecolor="black",
        markeredgecolor="black",
        zorder=2,
    )
    ax.plot(f_fine, model_fine * amp, color=color, lw=1.2, zorder=3)
    ax.plot(
        f_slice,
        model_native * amp,
        marker="x",
        linestyle="None",
        markersize=4.0,
        markeredgewidth=0.9,
        color=color,
        zorder=4,
    )


def _draw_mag_data(
    ax: plt.Axes,
    f_slice: np.ndarray,
    data_mag: np.ndarray,
    f_fine: np.ndarray,
    model_mag_fine: np.ndarray,
    model_mag_native: np.ndarray,
    lo_f: float,
    hi_f: float,
    amp: float,
    freq_padded: Optional[np.ndarray],
    spec_padded: Optional[np.ndarray],
) -> None:
    """|X| panel: 2x-zero-filled data magnitude (or native fallback) + model."""
    if freq_padded is not None and spec_padded is not None:
        pm = (freq_padded >= lo_f) & (freq_padded <= hi_f)
        f_disp, z_disp = freq_padded[pm], spec_padded[pm]
        ax.plot(f_disp, np.abs(z_disp) * amp, color="#00000066", lw=0.6, zorder=1)
        ax.plot(
            f_disp,
            np.abs(z_disp) * amp,
            marker="o",
            linestyle="None",
            markersize=1.6,
            markerfacecolor="black",
            markeredgecolor="black",
            zorder=2,
        )
    else:
        ax.plot(f_slice, data_mag * amp, color="#00000044", lw=0.5, zorder=1)
        ax.plot(
            f_slice,
            data_mag * amp,
            marker="o",
            linestyle="None",
            markersize=2.0,
            markerfacecolor="black",
            markeredgecolor="black",
            zorder=2,
        )
    ax.plot(f_fine, model_mag_fine * amp, color="tab:purple", lw=1.2, zorder=3)
    ax.plot(
        f_slice,
        model_mag_native * amp,
        marker="x",
        linestyle="None",
        markersize=4.0,
        markeredgewidth=0.9,
        color="tab:purple",
        zorder=4,
    )


def _draw_residual_hist(
    ax: plt.Axes,
    residual: np.ndarray,
    sigma_slice: np.ndarray,
    amp: float,
    usuffix: str,
) -> None:
    """|residual| histogram vs the Rayleigh(sigma/sqrt2) noise model (native)."""
    mag = np.abs(residual) * amp
    sigma_c = (
        (float(np.median(sigma_slice)) / np.sqrt(2.0)) * amp
        if sigma_slice.size
        else 0.0
    )
    if sigma_c > 0.0 and mag.size > 0:
        n_bins = max(10, min(40, mag.size // 5))
        ax.hist(
            mag,
            bins=n_bins,
            density=True,
            color="0.75",
            edgecolor="0.3",
            linewidth=0.4,
            label="|residual|",
        )
        x = np.linspace(0.0, max(float(mag.max()), 5.0 * sigma_c), 400)
        rayleigh = (x / sigma_c**2) * np.exp(-(x**2) / (2.0 * sigma_c**2))
        ax.plot(
            x,
            rayleigh,
            color="tab:purple",
            lw=1.0,
            label=r"Rayleigh($\sigma/\sqrt{2}$)",
        )
        ax.axvline(
            3.0 * sigma_c, color="tab:red", lw=0.6, ls="--", label=r"$3\sigma_c$"
        )
        ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    ax.set_xlabel(f"|residual|{usuffix}", fontsize=9)
    ax.set_ylabel("density", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title("|residual| vs noise", fontsize=9)


def _draw_peak_table(
    ax: plt.Axes,
    fitted_peaks: Sequence[FittedPeak],
    labels: Sequence[str],
    amp: float,
    units_label: str,
) -> None:
    """Row 4 right: fitted-peak table with PDG-style uncertainties."""
    ax.set_title("Fitted peaks", fontsize=9, loc="left")
    n = len(fitted_peaks)
    line_h = 1.0 / max(n + 1, 8)
    amp_hdr = f"amplitude ({units_label})" if units_label else "amplitude"
    header = (
        f"  pk | {'frequency (MHz)':>18} | {amp_hdr:>14} | "
        f"{'phase (rad)':>12} | {'SNR':>5} | conf"
    )
    ax.text(
        0.02,
        0.98,
        header,
        transform=ax.transAxes,
        family="monospace",
        fontsize=8.0,
        va="top",
        color="0.3",
    )
    ax.text(
        0.02,
        0.98 - 0.5 * line_h,
        "  " + "-" * (len(header) - 2),
        transform=ax.transAxes,
        family="monospace",
        fontsize=8.0,
        va="top",
        color="0.5",
    )
    for i, (pk, lbl) in enumerate(zip(fitted_peaks, labels)):
        freq_s = _format_spectroscopic(float(pk.frequency_mhz), pk.frequency_error)
        amp_err = pk.amplitude_error * amp if pk.amplitude_error is not None else None
        amp_s = _format_spectroscopic_sci(float(pk.amplitude) * amp, amp_err)
        phase_s = (
            _format_spectroscopic(pk.phase, pk.phase_error)
            if pk.phase is not None
            else "-"
        )
        snr_s = f"{pk.snr:.2f}" if pk.snr is not None else "-"
        conf = _peak_confidence(pk)
        conf_s = f"{conf:.2f}" if conf is not None else "-"
        line = (
            f"  {lbl:>2} | {freq_s:>18} | {amp_s:>14} | {phase_s:>12} | "
            f"{snr_s:>5} | {conf_s:>4}"
        )
        ax.text(
            0.02,
            0.98 - (i + 1.5) * line_h,
            line,
            transform=ax.transAxes,
            family="monospace",
            fontsize=8.0,
            va="top",
        )


# ===========================================================================
# Windowed (apodized) fit view
# ===========================================================================
@dataclass
class WindowedView:
    """Precomputed arrays for the windowed comparison figure (one window)."""

    freq_native: np.ndarray  # ascending molecular freq over the window
    data_native: np.ndarray  # windowed data spectrum (native grid)
    model_native: np.ndarray  # windowed model spectrum (native grid)
    freq_data_2x: np.ndarray  # 2x grid for the |X| data magnitude
    data_2x: np.ndarray
    freq_model_fine: np.ndarray  # dense grid for the smooth model curve
    model_fine: np.ndarray


def plot_windowed_comparison(
    view: WindowedView,
    *,
    title: str,
    apodize_label: str,
    amplitude_scale: float = 1.0,
    units_label: str = "",
    figsize: Tuple[float, float] = (11, 4.2),
) -> plt.Figure:
    """Re/Im/|X| comparison of the data and model under the same apodization.

    Strictly diagnostic: the window changes the noise correlation, so this is
    *not* the fit metric -- no residuals or chi-squared are shown. The model is
    the persisted fit's lines re-synthesised and windowed identically to the
    data; the leakage-wing baseline is omitted because apodization suppresses
    the very skirt it compensates.
    """
    amp = float(amplitude_scale)
    usuffix = f" ({units_label})" if units_label else ""
    fig, (ax_re, ax_im, ax_mag) = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=10)

    _draw_data_model(
        ax_re,
        view.freq_native,
        np.real(view.data_native),
        view.freq_model_fine,
        np.real(view.model_fine),
        np.real(view.model_native),
        "tab:red",
        amp,
    )
    _draw_data_model(
        ax_im,
        view.freq_native,
        np.imag(view.data_native),
        view.freq_model_fine,
        np.imag(view.model_fine),
        np.imag(view.model_native),
        "tab:blue",
        amp,
    )
    _draw_mag_data(
        ax_mag,
        view.freq_native,
        np.abs(view.data_native),
        view.freq_model_fine,
        np.abs(view.model_fine),
        np.abs(view.model_native),
        float(view.freq_native.min()) if view.freq_native.size else 0.0,
        float(view.freq_native.max()) if view.freq_native.size else 0.0,
        amp,
        view.freq_data_2x,
        view.data_2x,
    )
    ax_re.set_ylabel(f"Re{usuffix}", fontsize=9)
    ax_im.set_ylabel(f"Im{usuffix}", fontsize=9)
    ax_mag.set_ylabel(f"|X|{usuffix}", fontsize=9)
    ax_re.set_title(
        f"apodized: {apodize_label} (diagnostic, not the fit metric)",
        fontsize=8,
        loc="left",
    )
    for ax in (ax_re, ax_im, ax_mag):
        ax.set_xlabel("frequency (MHz)", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
    fig.tight_layout()
    return fig
