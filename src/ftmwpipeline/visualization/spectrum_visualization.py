"""
Pipeline plotting functions.

This module contains the matplotlib plotting functions for FTMW pipeline
visualization, styled with the shared house style in :mod:`report_style`.
"""

from typing import Any, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from ..core.data_structures import FID, ComplexFT, SpectralWindow
from .report_style import (
    AGGIE_BLUE,
    CABERNET,
    DOUBLE_DECKER,
    GUNROCK,
    PINOT,
    POPPY,
    apply_bare_style,
    resolve_title,
)


def plot_complex_ft(
    complex_ft: ComplexFT,
    freq_range: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    interactive: bool = True,
    figsize: Tuple[float, float] = (16, 9),
    fid: Optional[FID] = None,
    active_fid_time_us: Optional[np.ndarray] = None,
    active_fid_data: Optional[np.ndarray] = None,
    show_fid_panels: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Plot a ComplexFT with optional FID panels showing the processing stages.

    The figure includes up to three stacked regions:

    1. Raw FID data with the active-region bounds (if ``fid`` is provided and
       ``show_fid_panels=True``).
    2. Active FID data -- just the active-region slice (start_us to end_us),
       DC-removed (if ``active_fid_time_us``/``active_fid_data`` are provided
       and ``show_fid_panels=True``).
    3. The ComplexFT magnitude spectrum and its real/imaginary components
       (always shown). ``complex_ft`` is expected to be the zero-padded,
       active-band DISPLAY FT (e.g. from ``compute_display_ft``), not the
       full-record, zero-substituted Stage 1 FT -- this is a display surface
       only and must never be treated as the fitted spectrum. If
       ``complex_ft.metadata`` carries ``amplitude_scale`` / ``units_label``
       (as the display FT does), the magnitude and real/imaginary panels are
       scaled and labeled accordingly, matching the Stage 5 report/``fit
       show`` convention.

    Parameters
    ----------
    complex_ft : ComplexFT
        ComplexFT object to plot (typically the padded active-band display
        FT; see above).
    freq_range : tuple of float, optional
        (min_freq, max_freq) in MHz to display. If None, uses full range.
    title : str, optional
        Plot title. ``None`` uses an automatic title; ``""`` suppresses it
        (the documentation-figure convention, where the caption labels the
        figure).
    interactive : bool, optional
        Reserved flag for the caller's display logic (default: True).
    figsize : tuple of float, optional
        Figure size in inches (width, height). Default: (16, 9)
    fid : FID, optional
        Original FID data for the raw FID panel with active-region bounds
    active_fid_time_us : np.ndarray, optional
        Time axis (μs) of the active-region FID slice, for the Active FID
        panel. Must be paired with ``active_fid_data``.
    active_fid_data : np.ndarray, optional
        DC-removed voltage data of the active-region FID slice, for the
        Active FID panel. Must be paired with ``active_fid_time_us``.
    show_fid_panels : bool, optional
        Whether to show FID panels when FID data is provided (default: True)
    **kwargs
        Additional arguments passed to plotting functions

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """

    # Extract data
    freq = complex_ft.freq_array
    amp = float(complex_ft.metadata.get("amplitude_scale", 1.0))
    units_label = str(complex_ft.metadata.get("units_label", ""))
    magnitude = complex_ft.magnitude_spectrum * amp
    real_part = complex_ft.real_spectrum * amp
    imag_part = complex_ft.imag_spectrum * amp

    # Apply frequency range filter if specified
    if freq_range is not None:
        freq_min, freq_max = freq_range
        mask = (freq >= freq_min) & (freq <= freq_max)
        freq = freq[mask]
        magnitude = magnitude[mask]
        real_part = real_part[mask]
        imag_part = imag_part[mask]

    # Generate title if not provided
    if title is None:
        freq_min, freq_max = np.min(freq), np.max(freq)
        title = f"ComplexFT Spectrum ({freq_min:.1f} - {freq_max:.1f} MHz)"

    # Determine which panels to show
    show_raw_fid = show_fid_panels and fid is not None
    show_active_fid = (
        show_fid_panels
        and active_fid_time_us is not None
        and active_fid_data is not None
    )

    return _plot_complex_ft_matplotlib(
        freq,
        magnitude,
        real_part,
        imag_part,
        units_label,
        title,
        figsize,
        interactive,
        fid,
        active_fid_time_us,
        active_fid_data,
        show_raw_fid,
        show_active_fid,
        **kwargs,
    )


def _plot_complex_ft_matplotlib(
    freq: np.ndarray,
    magnitude: np.ndarray,
    real_part: np.ndarray,
    imag_part: np.ndarray,
    units_label: str,
    title: str,
    figsize: Tuple[float, float],
    interactive: bool,
    fid: Optional[FID],
    active_fid_time_us: Optional[np.ndarray],
    active_fid_data: Optional[np.ndarray],
    show_raw_fid: bool,
    show_active_fid: bool,
    **kwargs: Any,
) -> Any:
    """Create the matplotlib ComplexFT figure with optional FID panels."""

    # Check if we need FID panels
    need_fid_panels = show_raw_fid or show_active_fid

    amp_suffix = f" ({units_label})" if units_label else ""

    # Constrained layout handles the spanning magnitude/real-imag rows without
    # the spanning-axes warning that tight_layout raises here.
    fig = plt.figure(figsize=figsize, constrained_layout=True)

    if need_fid_panels:
        # Row 1: 2 columns (Raw FID | Active FID)
        # Row 2: Magnitude spectrum (spans both columns)
        # Row 3: Real/Imaginary (spans both columns)
        gs = gridspec.GridSpec(3, 2, figure=fig)
    else:
        # No FID panels: just 2 rows for spectrum data
        gs = gridspec.GridSpec(2, 1, figure=fig)

    if need_fid_panels:
        # Raw FID panel (left column)
        if show_raw_fid:
            assert fid is not None
            ax_raw = fig.add_subplot(gs[0, 0])
            time_us = fid.time_array_us()
            ax_raw.plot(
                time_us, fid.data, color=AGGIE_BLUE, linewidth=1, label="Raw FID"
            )
            ax_raw.set_ylabel("Voltage")
            ax_raw.set_xlabel("Time (μs)")
            ax_raw.set_title("Raw FID Data")

            # Active-region bounds
            if hasattr(fid, "processing") and fid.processing:
                if fid.processing.start_us is not None:
                    ax_raw.axvline(
                        fid.processing.start_us,
                        color=POPPY,
                        linestyle="--",
                        linewidth=2,
                        label=f"Start: {fid.processing.start_us:.1f} μs",
                    )
                if fid.processing.end_us is not None:
                    ax_raw.axvline(
                        fid.processing.end_us,
                        color=POPPY,
                        linestyle="--",
                        linewidth=2,
                        label=f"End: {fid.processing.end_us:.1f} μs",
                    )
            apply_bare_style(ax_raw)
            ax_raw.legend()

        # Active FID panel (right column) -- the active-region slice only,
        # DC-removed, not the full zero-substituted record.
        if show_active_fid:
            assert active_fid_time_us is not None
            assert active_fid_data is not None
            ax_active = fig.add_subplot(gs[0, 1])
            ax_active.plot(
                active_fid_time_us,
                active_fid_data,
                color=PINOT,
                linewidth=1,
                label="Active FID",
            )
            ax_active.set_ylabel("Voltage")
            ax_active.set_xlabel("Time (μs)")
            ax_active.set_title("Active FID")
            apply_bare_style(ax_active)
            ax_active.legend()

        ax_mag = fig.add_subplot(gs[1, :])
        ax_complex = fig.add_subplot(gs[2, :])
    else:
        ax_mag = fig.add_subplot(gs[0, 0])
        ax_complex = fig.add_subplot(gs[1, 0])

    # Magnitude spectrum
    ax_mag.plot(freq, magnitude, color=CABERNET, linewidth=1, label="Magnitude")
    ax_mag.set_ylabel(f"Magnitude{amp_suffix}")
    ax_mag.set_title("Magnitude Spectrum")
    apply_bare_style(ax_mag)
    ax_mag.legend()

    # Real and imaginary components
    ax_complex.plot(freq, real_part, color=DOUBLE_DECKER, linewidth=1, label="Real")
    ax_complex.plot(freq, imag_part, color=GUNROCK, linewidth=1, label="Imaginary")
    ax_complex.set_xlabel("Frequency (MHz)")
    ax_complex.set_ylabel(f"Amplitude{amp_suffix}")
    ax_complex.set_title("Real and Imaginary Components")
    apply_bare_style(ax_complex)
    ax_complex.legend()

    # Overall title (opt-in)
    resolved = resolve_title(title, title)
    if resolved:
        fig.suptitle(resolved, fontsize=16)

    return fig


def plot_spectral_window(
    window: SpectralWindow,
    title: Optional[str] = None,
    show_peaks: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Plot a spectral window with optional peak annotations.

    Parameters
    ----------
    window : SpectralWindow
        SpectralWindow object to plot
    title : str, optional
        Plot title. ``None`` uses an automatic title; ``""`` suppresses it.
    show_peaks : bool, optional
        Whether to annotate detected peaks (default: True)
    **kwargs
        Additional plotting arguments

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """

    if title is None:
        title = (
            f"Spectral Window {window.window_id or ''} "
            f"({window.freq_range[0]:.1f} - {window.freq_range[1]:.1f} MHz)"
        )

    fig, ax = plt.subplots(figsize=(12, 6))

    freq = window.freq_array
    magnitude = window.magnitude_spectrum

    # Main spectrum
    ax.plot(freq, magnitude, color=CABERNET, linewidth=1, label="Magnitude")

    # Add peak markers if requested
    if show_peaks and window.n_peaks > 0:
        peak_freqs = [peak.frequency for peak in window.peaks]
        peak_intensities = [peak.intensity for peak in window.peaks]
        ax.plot(
            peak_freqs,
            peak_intensities,
            "x",
            color=DOUBLE_DECKER,
            markersize=8,
            markeredgewidth=2,
            label="Detected Peaks",
        )

        for i, peak in enumerate(window.peaks):
            label = f"P{i+1}"
            if peak.snr is not None:
                label += f"\nSNR: {peak.snr:.1f}"
            ax.annotate(
                label,
                (peak.frequency, peak.intensity),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                ha="left",
            )

    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Magnitude")
    resolved = resolve_title(title, title)
    if resolved:
        ax.set_title(resolved)
    apply_bare_style(ax)
    ax.legend()

    fig.tight_layout()
    return fig
