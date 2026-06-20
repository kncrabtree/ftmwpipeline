"""
Pipeline plotting functions.

This module contains the matplotlib plotting functions for FTMW pipeline
visualization, styled with the shared house style in :mod:`report_style`.
"""

from typing import Any, Optional, Tuple

import numpy as np

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from ..core.data_structures import FID, ComplexFT, PreprocessedFID, SpectralWindow
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
    preprocessed_fid: Optional[PreprocessedFID] = None,
    show_fid_panels: bool = True,
    **kwargs: Any,
) -> Any:
    """
    Plot a ComplexFT with optional FID panels showing the processing stages.

    The figure includes up to three stacked regions:

    1. Raw FID data with the active-region bounds (if ``fid`` is provided and
       ``show_fid_panels=True``).
    2. Preprocessed FID data (if ``preprocessed_fid`` is provided and
       ``show_fid_panels=True``).
    3. The ComplexFT magnitude spectrum and its real/imaginary components
       (always shown).

    Parameters
    ----------
    complex_ft : ComplexFT
        ComplexFT object to plot
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
    preprocessed_fid : PreprocessedFID, optional
        Preprocessed FID data for the preprocessed FID panel
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
    magnitude = complex_ft.magnitude_spectrum
    real_part = complex_ft.real_spectrum
    imag_part = complex_ft.imag_spectrum

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
    show_preprocessed_fid = show_fid_panels and preprocessed_fid is not None

    return _plot_complex_ft_matplotlib(
        freq,
        magnitude,
        real_part,
        imag_part,
        title,
        figsize,
        interactive,
        fid,
        preprocessed_fid,
        show_raw_fid,
        show_preprocessed_fid,
        **kwargs,
    )


def _plot_complex_ft_matplotlib(
    freq: np.ndarray,
    magnitude: np.ndarray,
    real_part: np.ndarray,
    imag_part: np.ndarray,
    title: str,
    figsize: Tuple[float, float],
    interactive: bool,
    fid: Optional[FID],
    preprocessed_fid: Optional[PreprocessedFID],
    show_raw_fid: bool,
    show_preprocessed_fid: bool,
    **kwargs: Any,
) -> Any:
    """Create the matplotlib ComplexFT figure with optional FID panels."""

    # Check if we need FID panels
    need_fid_panels = show_raw_fid or show_preprocessed_fid

    # Constrained layout handles the spanning magnitude/real-imag rows without
    # the spanning-axes warning that tight_layout raises here.
    fig = plt.figure(figsize=figsize, constrained_layout=True)

    if need_fid_panels:
        # Row 1: 2 columns (Raw FID | Preprocessed FID)
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
            ax_raw.plot(time_us, fid.data, color=AGGIE_BLUE, linewidth=1, label="Raw FID")
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

        # Preprocessed FID panel (right column)
        if show_preprocessed_fid:
            assert preprocessed_fid is not None
            ax_preproc = fig.add_subplot(gs[0, 1])
            time_us = (
                np.arange(len(preprocessed_fid.data)) * preprocessed_fid.spacing * 1e6
            )
            ax_preproc.plot(
                time_us,
                preprocessed_fid.data,
                color=PINOT,
                linewidth=1,
                label="Preprocessed FID",
            )
            ax_preproc.set_ylabel("Voltage")
            ax_preproc.set_xlabel("Time (μs)")
            ax_preproc.set_title("Preprocessed FID Data")
            apply_bare_style(ax_preproc)
            ax_preproc.legend()

        ax_mag = fig.add_subplot(gs[1, :])
        ax_complex = fig.add_subplot(gs[2, :])
    else:
        ax_mag = fig.add_subplot(gs[0, 0])
        ax_complex = fig.add_subplot(gs[1, 0])

    # Magnitude spectrum
    ax_mag.plot(freq, magnitude, color=CABERNET, linewidth=1, label="Magnitude")
    ax_mag.set_ylabel("Magnitude")
    ax_mag.set_title("Magnitude Spectrum")
    apply_bare_style(ax_mag)
    ax_mag.legend()

    # Real and imaginary components
    ax_complex.plot(freq, real_part, color=DOUBLE_DECKER, linewidth=1, label="Real")
    ax_complex.plot(freq, imag_part, color=GUNROCK, linewidth=1, label="Imaginary")
    ax_complex.set_xlabel("Frequency (MHz)")
    ax_complex.set_ylabel("Amplitude")
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
