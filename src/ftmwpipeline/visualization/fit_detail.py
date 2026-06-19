"""
Consolidated Stage 5 per-window detail figure.

A single ~16:9 figure summarising one fit window, built from reusable painters
(:func:`prepare_window_panels` + ``draw_*``) so the combined figure and the
modular per-panel figures (the HTML report's flexbox, :func:`plot_window_panels`)
share one source of truth:

* Row 1 -- full-spectrum context (data magnitude) with the window highlighted.
* Row 2 -- three fused Re / Im / |X| panels, each a thin residual strip over a
  data + model panel sharing one x-axis, with a vertical line at every fitted
  peak's molecular frequency. The residual stays on the fit's native grid so
  |residual| = sqrt(Re^2 + Im^2) bin-for-bin. The data + model overlay draws the
  model on an oversampled grid (smooth analytic line shape) *and* as small ``x``
  markers at the native data bins, so the eye compares model-at-bin with
  data-at-bin and the smooth curve is never mistaken for a denser fit.
* Row 3 -- |residual| histogram against the Rayleigh noise model (left) and a
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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecBase, GridSpecFromSubplotSpec
from matplotlib.ticker import ScalarFormatter

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

# Default combined-figure size, ~16:9 for a typical widescreen monitor (the
# detail figure is viewed on screen far more often than printed).
DEFAULT_FIGSIZE = (13.33, 7.5)


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


def frequency_sorted_labels(freqs: Sequence[float]) -> List[str]:
    """Alpha labels assigned by ascending frequency: the lowest-frequency peak
    is ``A``, the next ``B``, and so on.

    Returned in the *input* order -- ``labels[i]`` is the letter for
    ``freqs[i]`` -- so callers can zip it straight onto their peak list without
    re-sorting. This is the single source of the display letter so the figure
    annotations, the fit-log peak table, and the HTML fitted-lines table all
    agree.
    """
    n = len(freqs)
    letters = _peak_labels(n)
    out = [""] * n
    for rank, i in enumerate(sorted(range(n), key=lambda j: freqs[j])):
        out[i] = letters[rank]
    return out


def _peak_confidence(peak: FittedPeak) -> Optional[float]:
    """Per-peak confidence rating placeholder.

    No principled per-peak confidence statistic exists yet; the column is wired
    here so it can be filled in one place when one does. Returns ``None`` (the
    table renders ``-``).
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
# Panel data preparation (one source of truth for both assemblers)
# ---------------------------------------------------------------------------
@dataclass
class WindowPanelData:
    """Everything the per-window painters need, prepared once.

    Produced by :func:`prepare_window_panels` and consumed by the painters
    below, so the combined ``fit show`` figure and the modular per-panel figures
    (the HTML report's flexbox) draw from the same prepared arrays rather than
    each re-deriving the model and residual.
    """

    # Full-spectrum context (overview panel).
    frequencies: np.ndarray
    complex_spectrum: np.ndarray
    trim_mhz: Optional[Tuple[float, float]]
    # Window geometry.
    lo_f: float
    hi_f: float
    center: float
    # Native window slices (every statistic lives here).
    f_slice: np.ndarray
    z_slice: np.ndarray
    sigma_slice: np.ndarray
    model_slice: np.ndarray
    residual: np.ndarray
    # Oversampled smooth-model overlay (display only).
    f_fine: np.ndarray
    model_fine: np.ndarray
    # Display transforms / noise band.
    amp: float
    usuffix: str
    units_label: str
    band: float
    freq_padded: Optional[np.ndarray]
    spec_padded: Optional[np.ndarray]
    # Peak annotations and the fitted-peak table.
    fitted_peaks: List[FittedPeak]
    fitted_freqs: List[float]
    labels: List[str]
    vline_plotter: Callable[[plt.Axes, bool], None]
    in_window_spurs: List[dict]
    lattice_peaks: List[FittedPeak]
    tau_us: float
    shape: str
    title: str


# ---------------------------------------------------------------------------
# Spine-free / light-grid presentation (shared by the report and CLI panels)
# ---------------------------------------------------------------------------

# Distinct colour for the noise-band (+/- sigma) reference lines and the
# peak-position vlines: amber reads as a deliberate reference mark and stays
# clear of both the grey grid and the red / blue / purple residual traces
# (vlines are vertical, the noise bands horizontal dashed -- distinct by
# orientation).
_BARE_BAND_COLOR = "#d4920a"
# The zero baseline, emphasised so it pops out of the faint grid.
_BARE_ZERO_COLOR = "#2a2a2a"


def _apply_bare_style(ax: plt.Axes) -> None:
    """Strip an axes to a spine-free, tick-mark-free look with a light major grid.

    No spines (the top / right ones especially read as clutter), no tick marks
    (labels kept), and a faint major grid in their place, sitting behind the
    data. The reference marks (noise band, peak vlines, zero baseline) are
    coloured distinctly so they are not mistaken for grid lines.
    """
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.grid(True, which="major", color="#dbe0e6", lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def _draw_bare_zero(ax: plt.Axes) -> None:
    """Emphasised y = 0 baseline for a bare-style axes (darker than the grid)."""
    ax.axhline(0.0, color=_BARE_ZERO_COLOR, lw=0.9, alpha=0.65, zorder=2.5)


def prepare_window_panels(
    window_fit: FittingResult,
    *,
    frequencies: np.ndarray,
    complex_spectrum: np.ndarray,
    rms_noise: np.ndarray,
    sideband: SidebandLike,
    acquisition_us: float,
    title: str = "",
    amplitude_scale: float = 1.0,
    units_label: str = "",
    trim_mhz: Optional[Tuple[float, float]] = None,
    freq_padded: Optional[np.ndarray] = None,
    spec_padded: Optional[np.ndarray] = None,
    spurs: Optional[Sequence[dict]] = None,
    vline_color: str = _BARE_BAND_COLOR,
    vline_alpha: float = 0.7,
) -> WindowPanelData:
    """Prepare the per-window model, residual, and annotations once.

    The native window slice drives every statistic (residual, noise band,
    histogram); the oversampled ``model_fine`` and the 2x ``freq_padded`` /
    ``spec_padded`` grids are display-only overlays. See the module docstring
    for the magnitude-honesty rationale.
    """
    if window_fit.window is None:
        raise ValueError(
            "window_fit has no attached SpectralWindow -- cannot plot detail "
            "(the fit was loaded without window context)"
        )
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

    fitted_freqs = [float(p.frequency_mhz) for p in window_fit.fitted_peaks]
    labels = frequency_sorted_labels(fitted_freqs)
    vline_plotter = _make_vline_plotter(
        fitted_freqs, labels, tau_us, color=vline_color, alpha=vline_alpha
    )
    in_window_spurs = [
        sp
        for sp in (spurs or [])
        if lo_f <= float(sp.get("center_mhz", float("nan"))) <= hi_f
    ]
    lattice_peaks = [
        p
        for p in window_fit.fitted_peaks
        if getattr(p, "clock_lattice", None) is not None
    ]

    return WindowPanelData(
        frequencies=frequencies,
        complex_spectrum=complex_spectrum,
        trim_mhz=trim_mhz,
        lo_f=lo_f,
        hi_f=hi_f,
        center=center,
        f_slice=f_slice,
        z_slice=z_slice,
        sigma_slice=sigma_slice,
        model_slice=model_slice,
        residual=residual,
        f_fine=f_fine,
        model_fine=model_fine,
        amp=float(amplitude_scale),
        usuffix=f" ({units_label})" if units_label else "",
        units_label=units_label,
        band=band,
        freq_padded=freq_padded,
        spec_padded=spec_padded,
        fitted_peaks=list(window_fit.fitted_peaks),
        fitted_freqs=fitted_freqs,
        labels=labels,
        vline_plotter=vline_plotter,
        in_window_spurs=in_window_spurs,
        lattice_peaks=lattice_peaks,
        tau_us=tau_us,
        shape=shape_str,
        title=title,
    )


# ---------------------------------------------------------------------------
# Panel painters (each paints onto axes the assembler owns)
# ---------------------------------------------------------------------------

# Per data+residual component: (projection, line colour, data-panel label).
_COMPONENT_SPECS: Dict[str, Tuple[Callable[[np.ndarray], np.ndarray], str, str]] = {
    "re": (np.real, "tab:red", "Re"),
    "im": (np.imag, "tab:blue", "Im"),
    "mag": (np.abs, "tab:purple", "|X|"),
}


def _stacked_pair(
    fig: plt.Figure,
    spec: Optional[Any] = None,
    *,
    anchor: Optional[plt.Axes] = None,
) -> Tuple[plt.Axes, plt.Axes]:
    """Create a (residual strip, data panel) pair sharing one x-axis.

    ``spec`` is a :class:`~matplotlib.gridspec.SubplotSpec` cell of an outer grid
    (combined figure) or ``None`` for a standalone single-panel figure. The thin
    residual strip sits above the taller data panel; ``anchor`` links the x-axis
    to another data panel so a row of panels zoom together. Returns
    ``(ax_residual, ax_data)``.
    """
    gs: GridSpecBase
    if spec is None:
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 3], hspace=0.06)
    else:
        gs = GridSpecFromSubplotSpec(
            2, 1, subplot_spec=spec, height_ratios=[1, 3], hspace=0.06
        )
    ax_data = fig.add_subplot(gs[1], sharex=anchor)
    ax_resid = fig.add_subplot(gs[0], sharex=ax_data)
    return ax_resid, ax_data


def draw_overview(ax: plt.Axes, data: WindowPanelData) -> None:
    """Full-spectrum context panel with this window highlighted."""
    _draw_overview(
        ax,
        data.frequencies,
        data.complex_spectrum,
        data.lo_f,
        data.hi_f,
        data.amp,
        data.usuffix,
        data.trim_mhz,
    )
    # Spine-free / light-grid presentation; magnitude is positive, no zero line.
    _apply_bare_style(ax)


def _annotate_spurs_lattice(
    ax_resid: plt.Axes,
    ax_data: plt.Axes,
    data: WindowPanelData,
    *,
    with_legend: bool,
) -> None:
    """Mark in-window gated spurs (dotted) and lattice-matched lines (dashed).

    The legend is attached to the data panel only when ``with_legend`` so the
    re/im panels stay uncluttered and the |X| panel carries the key.
    """
    for i, sp in enumerate(data.in_window_spurs):
        f_sp = float(sp["center_mhz"])
        ax_resid.axvline(f_sp, color="tab:orange", lw=1.0, ls=":", alpha=0.9, zorder=1)
        ax_data.axvline(
            f_sp,
            color="tab:orange",
            lw=1.0,
            ls=":",
            alpha=0.9,
            zorder=1,
            label=(
                f"spur ({sp.get('source', '?')})" if (with_legend and i == 0) else None
            ),
        )
    for i, lp in enumerate(data.lattice_peaks):
        f_lp = float(lp.frequency_mhz)
        ax_resid.axvline(f_lp, color="tab:orange", lw=0.9, ls="--", alpha=0.7, zorder=2)
        ax_data.axvline(
            f_lp,
            color="tab:orange",
            lw=0.9,
            ls="--",
            alpha=0.7,
            zorder=2,
            label="lattice match" if (with_legend and i == 0) else None,
        )
    if with_legend and (data.in_window_spurs or data.lattice_peaks):
        ax_data.legend(loc="upper right", fontsize=7, framealpha=0.85)


def draw_component(
    ax_resid: plt.Axes,
    ax_data: plt.Axes,
    data: WindowPanelData,
    component: str,
    *,
    show_xlabel: bool = True,
) -> None:
    """Paint one component (``"re"`` / ``"im"`` / ``"mag"``).

    Residual strip on top, data + model below, sharing an x-axis. The residual
    stays native (so ``|residual|`` is bin-for-bin); the ``|X|`` data overlay
    uses the 2x display grid -- see the module docstring. Rendered spine-free
    over a light major grid (:func:`_apply_bare_style`): the noise band is amber
    dashed and the zero baseline is emphasised, both distinct from the grid. The
    strictly-positive magnitude panels carry no zero line.
    """
    proj, color, dlabel = _COMPONENT_SPECS[component]
    amp = data.amp
    usuffix = data.usuffix
    band_s = data.band * amp
    is_mag = component == "mag"

    # Peak vlines: labelled on the residual strip (top), plain on the data panel.
    data.vline_plotter(ax_resid, True)
    data.vline_plotter(ax_data, False)

    # Residual strip.
    if is_mag:
        ax_resid.plot(data.f_slice, np.abs(data.residual) * amp, color=color, lw=0.7)
        if band_s > 0.0:
            ax_resid.axhline(band_s, color=_BARE_BAND_COLOR, lw=0.9, ls="--", zorder=2)
        ax_resid.set_ylabel(f"|resid|{usuffix}", fontsize=8)
    else:
        ax_resid.plot(data.f_slice, proj(data.residual) * amp, color=color, lw=0.7)
        if band_s > 0.0:
            ax_resid.axhline(band_s, color=_BARE_BAND_COLOR, lw=0.9, ls="--", zorder=2)
            ax_resid.axhline(-band_s, color=_BARE_BAND_COLOR, lw=0.9, ls="--", zorder=2)
        ax_resid.set_ylabel(f"{dlabel} resid{usuffix}", fontsize=8)

    # Data + model panel.
    if is_mag:
        _draw_mag_data(
            ax_data,
            data.f_slice,
            np.abs(data.z_slice),
            data.f_fine,
            np.abs(data.model_fine),
            np.abs(data.model_slice),
            data.lo_f,
            data.hi_f,
            amp,
            data.freq_padded,
            data.spec_padded,
        )
    else:
        _draw_data_model(
            ax_data,
            data.f_slice,
            proj(data.z_slice),
            data.f_fine,
            proj(data.model_fine),
            proj(data.model_slice),
            color,
            amp,
        )
    ax_data.set_ylabel(f"{dlabel}{usuffix}", fontsize=9)

    _annotate_spurs_lattice(ax_resid, ax_data, data, with_legend=is_mag)

    ax_resid.tick_params(axis="both", labelsize=8)
    ax_resid.tick_params(axis="x", labelbottom=False)
    ax_data.tick_params(axis="both", labelsize=8)
    # Show absolute frequencies on the shared x-axis: no "+2.96e4"-style offset
    # and no scientific collapse, so each tick reads as a full MHz value.
    xfmt = ScalarFormatter(useOffset=False)
    xfmt.set_scientific(False)
    ax_data.xaxis.set_major_formatter(xfmt)
    if show_xlabel:
        ax_data.set_xlabel("frequency (MHz)", fontsize=9)

    _apply_bare_style(ax_resid)
    _apply_bare_style(ax_data)
    # Emphasise the zero baseline only where zero is meaningful: the dispersive
    # re / im residual and data traces cross it. The magnitude panels are
    # strictly positive, so no zero line.
    if not is_mag:
        _draw_bare_zero(ax_resid)
        _draw_bare_zero(ax_data)


def draw_residual_hist(ax: plt.Axes, data: WindowPanelData) -> None:
    """|residual| histogram against the Rayleigh noise model."""
    _draw_residual_hist(ax, data.residual, data.sigma_slice, data.amp, data.usuffix)
    # Spine-free / light-grid presentation, matching the spectral panels.
    _apply_bare_style(ax)


def draw_peak_table(ax: plt.Axes, data: WindowPanelData) -> None:
    """Fitted-peak table with PDG-style uncertainties (combined figure only)."""
    ax.set_axis_off()
    _draw_peak_table(ax, data.fitted_peaks, data.labels, data.amp, data.units_label)


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
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
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
    data = prepare_window_panels(
        window_fit,
        frequencies=frequencies,
        complex_spectrum=complex_spectrum,
        rms_noise=rms_noise,
        sideband=sideband,
        acquisition_us=acquisition_us,
        title=title,
        amplitude_scale=amplitude_scale,
        units_label=units_label,
        trim_mhz=trim_mhz,
        freq_padded=freq_padded,
        spec_padded=spec_padded,
        spurs=spurs,
    )

    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=11)
    outer = GridSpec(
        nrows=3,
        ncols=3,
        figure=fig,
        height_ratios=[1.0, 2.8, 1.5],
        hspace=0.5,
        wspace=0.26,
        left=0.06,
        right=0.975,
        top=0.91,
        bottom=0.08,
    )
    ax_overview = fig.add_subplot(outer[0, :])
    draw_overview(ax_overview, data)

    # Three fused panels: each a residual strip over a data + model panel,
    # sharing one x-axis so a strong line and its residual stay together (and a
    # whole row zooms together).
    ax_re_res, ax_re_dat = _stacked_pair(fig, outer[1, 0])
    ax_im_res, ax_im_dat = _stacked_pair(fig, outer[1, 1], anchor=ax_re_dat)
    ax_mag_res, ax_mag_dat = _stacked_pair(fig, outer[1, 2], anchor=ax_re_dat)
    draw_component(ax_re_res, ax_re_dat, data, "re")
    draw_component(ax_im_res, ax_im_dat, data, "im")
    draw_component(ax_mag_res, ax_mag_dat, data, "mag")

    ax_hist = fig.add_subplot(outer[2, 0])
    draw_residual_hist(ax_hist, data)
    ax_peaks = fig.add_subplot(outer[2, 1:])
    draw_peak_table(ax_peaks, data)
    return fig


def plot_window_panels(
    window_fit: FittingResult,
    *,
    frequencies: np.ndarray,
    complex_spectrum: np.ndarray,
    rms_noise: np.ndarray,
    sideband: SidebandLike,
    acquisition_us: float,
    amplitude_scale: float = 1.0,
    units_label: str = "",
    trim_mhz: Optional[Tuple[float, float]] = None,
    freq_padded: Optional[np.ndarray] = None,
    spec_padded: Optional[np.ndarray] = None,
    spurs: Optional[Sequence[dict]] = None,
    panel_figsize: Tuple[float, float] = (6.6, 4.4),
    overview_figsize: Tuple[float, float] = (12.0, 2.6),
    hist_figsize: Tuple[float, float] = (6.6, 4.4),
    include_overview: bool = True,
) -> Dict[str, plt.Figure]:
    """Render the per-window detail as separate, standalone panel figures.

    Returns a dict keyed ``"overview"`` / ``"re"`` / ``"im"`` / ``"mag"`` /
    ``"hist"``, each a self-contained :class:`~matplotlib.figure.Figure` (one PNG
    per panel) for the HTML report's flexbox. Uses the same painters as the
    combined :func:`plot_consolidated_detail`, so the two stay consistent. The
    in-figure peak table is omitted -- the HTML page renders its own table. The
    caller owns closing the figures.

    When ``include_overview`` is False the full-spectrum ``"overview"`` panel is
    not built (the HTML report discards it in favour of a single shared
    interactive overview, so building one per window is wasted work).
    """
    data = prepare_window_panels(
        window_fit,
        frequencies=frequencies,
        complex_spectrum=complex_spectrum,
        rms_noise=rms_noise,
        sideband=sideband,
        acquisition_us=acquisition_us,
        amplitude_scale=amplitude_scale,
        units_label=units_label,
        trim_mhz=trim_mhz,
        freq_padded=freq_padded,
        spec_padded=spec_padded,
        spurs=spurs,
    )
    figures: Dict[str, plt.Figure] = {}

    if include_overview:
        fig_ov = plt.figure(figsize=overview_figsize, constrained_layout=True)
        draw_overview(fig_ov.add_subplot(111), data)
        figures["overview"] = fig_ov

    for component in ("re", "im", "mag"):
        fig_c = plt.figure(figsize=panel_figsize, constrained_layout=True)
        ax_res, ax_dat = _stacked_pair(fig_c)
        draw_component(ax_res, ax_dat, data, component)
        figures[component] = fig_c

    fig_h = plt.figure(figsize=hist_figsize, constrained_layout=True)
    draw_residual_hist(fig_h.add_subplot(111), data)
    figures["hist"] = fig_h

    return figures


def plot_correlation_heatmap(
    covariance: np.ndarray,
    labels: Sequence[str],
    *,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Render the parameter *correlation* matrix as a divergent heatmap.

    Normalizes the covariance to correlation coefficients
    ``rho_ij = cov_ij / sqrt(cov_ii * cov_jj)`` and draws them on a fixed
    ``[-1, +1]`` divergent colour scale, so the off-diagonal structure (which
    parameter pairs trade off) is legible at a glance even for a wide window
    where the numeric matrix is unreadable. Zero-variance parameters (a
    degenerate diagonal entry) yield a zero correlation rather than a NaN. The
    diagonal is set to exactly 1.
    """
    arr = np.asarray(covariance, dtype=float)
    n = len(labels)
    diag = np.diag(arr).astype(float)
    scale = np.sqrt(np.clip(diag, 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = arr / np.outer(scale, scale)
    corr[~np.isfinite(corr)] = 0.0
    np.fill_diagonal(corr, 1.0)

    if figsize is None:
        side = max(4.0, min(12.0, 0.32 * n + 1.6))
        figsize = (side + 1.0, side)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="RdBu_r", aspect="equal")
    # Label every cell while the matrix is small enough to stay legible; thin
    # the ticks out as the parameter count grows.
    fontsize = 8.0 if n <= 20 else max(4.0, 8.0 - 0.12 * (n - 20))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=fontsize)
    ax.set_yticklabels(labels, fontsize=fontsize)
    ax.set_title("Parameter correlation", fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("correlation coefficient")
    cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    return fig


def plot_summary_histograms(
    specs: Sequence[Tuple[str, str, Sequence[float]]],
    *,
    ncols: int = 3,
    bins: int = 30,
) -> Optional[plt.Figure]:
    """Render a grid of histograms for the report's summary distributions.

    ``specs`` is a list of ``(title, xlabel, values)``. Each non-empty entry
    becomes one histogram panel; a wide strictly-positive range (max/min > 100)
    is drawn on a log x-axis so a heavy tail does not crush the bulk. Returns
    ``None`` when no spec has data (the caller then omits the figure).
    """
    panels = [
        (t, x, [float(v) for v in vals if np.isfinite(v)]) for t, x, vals in specs
    ]
    panels = [(t, x, v) for t, x, v in panels if v]
    if not panels:
        return None
    n = len(panels)
    ncols = max(1, min(ncols, n))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.4 * ncols, 3.1 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    flat = list(axes.flat)
    for ax, (title, xlabel, vals) in zip(flat, panels):
        arr = np.asarray(vals, dtype=float)
        lo, hi = float(arr.min()), float(arr.max())
        if lo > 0.0 and hi / lo > 100.0:
            edges = np.logspace(np.log10(lo), np.log10(hi), bins + 1)
            ax.set_xscale("log")
        else:
            edges = np.linspace(lo, hi if hi > lo else lo + 1.0, bins + 1)
        ax.hist(arr, bins=edges, color="tab:blue", alpha=0.8, edgecolor="white", lw=0.3)
        med = float(np.median(arr))
        ax.axvline(med, color="tab:red", lw=1.0, ls="--", label=f"median {med:.3g}")
        ax.set_title(f"{title}  (n={arr.size})", fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("count", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        _apply_bare_style(ax)
    for ax in flat[n:]:
        ax.set_axis_off()
    return fig


def plot_magnitude_histogram(
    magnitudes: np.ndarray,
    *,
    sigma_median: float,
    title: str,
    xlabel: str = "magnitude |X|",
    n_sigma: float = 3.0,
    bins: int = 40,
    figsize: Tuple[float, float] = (4.4, 3.3),
) -> Optional[plt.Figure]:
    """A single log-log histogram of active-FT bin magnitudes.

    The distribution of ``|X|`` over every bin in a band sits as a Rayleigh-like
    noise hump (the bulk) with a heavy tail of real lines; a log-log scale keeps
    both legible at once. Two reference verticals mark where the noise floor sits
    relative to that bulk: the per-bin median ``sigma_x`` (``SNR = 1``) and the
    ``n_sigma`` detection level (``n_sigma * median sigma_x``), both in the same
    scaled units as *magnitudes*. Returns ``None`` when too few positive
    magnitudes remain to histogram (the caller then omits the panel).
    """
    arr = np.asarray(magnitudes, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size < 5:
        return None
    lo, hi = float(arr.min()), float(arr.max())
    if not (hi > lo):
        return None
    edges = np.logspace(np.log10(lo), np.log10(hi), bins + 1)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.hist(arr, bins=edges, color="tab:blue", alpha=0.8, edgecolor="white", lw=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if sigma_median > 0.0:
        ax.axvline(
            sigma_median,
            color="#d08700",
            lw=1.1,
            ls="--",
            label=f"median σₓ {sigma_median:.3g}",
        )
        ax.axvline(
            n_sigma * sigma_median,
            color="tab:red",
            lw=1.1,
            ls=":",
            label=f"{n_sigma:g}σ {n_sigma * sigma_median:.3g}",
        )
    ax.set_title(f"{title}  (n={arr.size:,})", fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("count", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)
    _apply_bare_style(ax)
    return fig


def _make_vline_plotter(
    fitted_freqs: Sequence[float],
    labels: Sequence[str],
    tau_us: float,
    *,
    color: str = "tab:gray",
    alpha: float = 0.45,
    lw: float = 1.2,
) -> Callable[[plt.Axes, bool], None]:
    """Build the per-peak vertical-line annotator with row-staggered labels.

    ``color`` / ``alpha`` / ``lw`` style the peak lines -- the bare/grid panels
    pass the amber reference colour so the peak positions read distinctly from
    the grey major grid.
    """
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
            ax.axvline(f_pk, color=color, lw=lw, alpha=alpha, zorder=1)
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
    ax.plot(f_fine, model_fine * amp, color=color, lw=1.0, alpha=0.8, zorder=3)
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
    ax.plot(
        f_fine, model_mag_fine * amp, color="tab:purple", lw=1.0, alpha=0.8, zorder=3
    )
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
    """Row 4 right: fitted-peak table with PDG-style uncertainties.

    When any peak in the window carries a ``clock_lattice`` annotation a narrow
    "lattice" column is appended so the user can spot spur candidates at a
    glance.  The column is omitted entirely when no peak is annotated, keeping
    the layout clean for the common no-declaration case.
    """
    ax.set_title("Fitted peaks", fontsize=9, loc="left")
    n = len(fitted_peaks)
    line_h = 1.0 / max(n + 1, 8)
    amp_hdr = f"amplitude ({units_label})" if units_label else "amplitude"
    # Only emit the lattice column when at least one peak in this window is
    # annotated -- the column is always absent on files with no declaration.
    show_lattice = any(
        getattr(pk, "clock_lattice", None) is not None for pk in fitted_peaks
    )
    lattice_col_w = 14  # field width for the identity string
    if show_lattice:
        header = (
            f"  pk | {'frequency (MHz)':>18} | {amp_hdr:>14} | "
            f"{'phase (rad)':>12} | {'SNR':>5} | conf | {'lattice':>{lattice_col_w}}"
        )
    else:
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
        cl = getattr(pk, "clock_lattice", None)
        if show_lattice:
            lattice_s = (cl or "")[:lattice_col_w]
            line = (
                f"  {lbl:>2} | {freq_s:>18} | {amp_s:>14} | {phase_s:>12} | "
                f"{snr_s:>5} | {conf_s:>4} | {lattice_s:>{lattice_col_w}}"
            )
        else:
            line = (
                f"  {lbl:>2} | {freq_s:>18} | {amp_s:>14} | {phase_s:>12} | "
                f"{snr_s:>5} | {conf_s:>4}"
            )
        # Annotated lines are tinted orange (consistent with the gated-spur
        # convention) so they stand out without a separate legend entry.
        text_color = "tab:orange" if cl is not None else "black"
        ax.text(
            0.02,
            0.98 - (i + 1.5) * line_h,
            line,
            transform=ax.transAxes,
            family="monospace",
            fontsize=8.0,
            va="top",
            color=text_color,
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
