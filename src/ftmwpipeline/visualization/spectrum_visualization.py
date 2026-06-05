"""
Pipeline plotting functions.

This module contains interactive plotting functions for FTMW pipeline visualization.
Supports both matplotlib and plotly backends.
"""

import warnings
from typing import Any, Optional, Tuple, Union

import numpy as np

# Try to import plotting libraries
try:
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from ..core.data_structures import FID, ComplexFT, PreprocessedFID, SpectralWindow
from ..io.fid_serialization import load_fid_cache


def plot_complex_ft(
    complex_ft: ComplexFT,
    freq_range: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    backend: str = "plotly",
    interactive: bool = True,
    figsize: Tuple[float, float] = (16, 12),
    fid: Optional[FID] = None,
    preprocessed_fid: Optional[PreprocessedFID] = None,
    show_fid_panels: bool = True,
    **kwargs,
) -> Any:
    """
    Create interactive plot of ComplexFT with optional FID panels showing processing stages.

    Enhanced visualization includes up to 3 panels:
    1. Raw FID data with windowing bounds (if fid provided and show_fid_panels=True)
    2. Preprocessed FID data (if preprocessed_fid provided and show_fid_panels=True)
    3. ComplexFT magnitude and real/imaginary components (always shown)

    Parameters
    ----------
    complex_ft : ComplexFT
        ComplexFT object to plot
    freq_range : tuple of float, optional
        (min_freq, max_freq) in MHz to display. If None, uses full range.
    title : str, optional
        Plot title. If None, generates automatic title.
    backend : str, optional
        Plotting backend: 'plotly' (default) or 'matplotlib'
    interactive : bool, optional
        Whether to create interactive plot (default: True)
    figsize : tuple of float, optional
        Figure size in inches (width, height). Default: (16, 12)
    fid : FID, optional
        Original FID data for raw FID panel with windowing bounds
    preprocessed_fid : PreprocessedFID, optional
        Preprocessed FID data for preprocessed FID panel
    show_fid_panels : bool, optional
        Whether to show FID panels when FID data is provided (default: True)
    **kwargs
        Additional arguments passed to plotting functions

    Returns
    -------
    figure
        Plotly Figure or matplotlib Figure object

    Raises
    ------
    ImportError
        If required plotting library is not available
    """

    # Check backend availability
    if backend == "plotly" and not HAS_PLOTLY:
        if HAS_MATPLOTLIB:
            warnings.warn(
                "Plotly not available, falling back to matplotlib", UserWarning
            )
            backend = "matplotlib"
            interactive = False
        else:
            raise ImportError("Neither plotly nor matplotlib is available")
    elif backend == "matplotlib" and not HAS_MATPLOTLIB:
        if HAS_PLOTLY:
            warnings.warn(
                "Matplotlib not available, falling back to plotly", UserWarning
            )
            backend = "plotly"
        else:
            raise ImportError("Neither matplotlib nor plotly is available")

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

    if backend == "plotly":
        return _plot_complex_ft_plotly(
            freq,
            magnitude,
            real_part,
            imag_part,
            title,
            fid,
            preprocessed_fid,
            show_raw_fid,
            show_preprocessed_fid,
            **kwargs,
        )
    else:
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


def _plot_complex_ft_plotly(
    freq: np.ndarray,
    magnitude: np.ndarray,
    real_part: np.ndarray,
    imag_part: np.ndarray,
    title: str,
    fid,
    preprocessed_fid,
    show_raw_fid: bool,
    show_preprocessed_fid: bool,
    **kwargs,
):
    """Create plotly interactive plot of ComplexFT with optional FID panels."""

    # Determine number of rows and subplot titles
    panel_count = 2  # Always show magnitude and real/imaginary panels
    subplot_titles = []

    if show_raw_fid:
        subplot_titles.append("Raw FID Data")
        panel_count += 1
    if show_preprocessed_fid:
        subplot_titles.append("Preprocessed FID Data")
        panel_count += 1

    subplot_titles.extend(["Magnitude Spectrum", "Real and Imaginary Components"])

    # Create subplots
    fig = make_subplots(
        rows=panel_count,
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.04,
        specs=[[{"secondary_y": False}] for _ in range(panel_count)],
    )

    current_row = 1

    # Raw FID panel
    if show_raw_fid:
        time_us = fid.time_array_us()
        fig.add_trace(
            go.Scatter(
                x=time_us,
                y=fid.data,
                mode="lines",
                name="Raw FID",
                line=dict(color="black", width=1),
                hovertemplate="Time: %{x:.3f} μs<br>Voltage: %{y:.3e}<extra></extra>",
            ),
            row=current_row,
            col=1,
        )

        # Add vertical lines for windowing bounds
        if hasattr(fid, "processing") and fid.processing:
            y_range = [np.min(fid.data), np.max(fid.data)]
            if fid.processing.start_us is not None:
                fig.add_shape(
                    type="line",
                    x0=fid.processing.start_us,
                    x1=fid.processing.start_us,
                    y0=y_range[0],
                    y1=y_range[1],
                    line=dict(color="red", width=2, dash="dash"),
                    row=current_row,
                    col=1,
                )
            if fid.processing.end_us is not None:
                fig.add_shape(
                    type="line",
                    x0=fid.processing.end_us,
                    x1=fid.processing.end_us,
                    y0=y_range[0],
                    y1=y_range[1],
                    line=dict(color="red", width=2, dash="dash"),
                    row=current_row,
                    col=1,
                )

        fig.update_xaxes(title_text="Time (μs)", showgrid=True, row=current_row, col=1)
        fig.update_yaxes(title_text="Voltage", showgrid=True, row=current_row, col=1)
        current_row += 1

    # Preprocessed FID panel
    if show_preprocessed_fid:
        # Create time array for preprocessed data
        time_us = np.arange(len(preprocessed_fid.data)) * preprocessed_fid.spacing * 1e6
        fig.add_trace(
            go.Scatter(
                x=time_us,
                y=preprocessed_fid.data,
                mode="lines",
                name="Preprocessed FID",
                line=dict(color="purple", width=1),
                hovertemplate="Time: %{x:.3f} μs<br>Voltage: %{y:.3e}<extra></extra>",
            ),
            row=current_row,
            col=1,
        )

        fig.update_xaxes(title_text="Time (μs)", showgrid=True, row=current_row, col=1)
        fig.update_yaxes(title_text="Voltage", showgrid=True, row=current_row, col=1)
        current_row += 1

    # Magnitude plot
    fig.add_trace(
        go.Scatter(
            x=freq,
            y=magnitude,
            mode="lines",
            name="Magnitude",
            line=dict(color="blue", width=1),
            hovertemplate="Freq: %{x:.3f} MHz<br>Magnitude: %{y:.3e}<extra></extra>",
        ),
        row=current_row,
        col=1,
    )

    fig.update_xaxes(
        title_text="Frequency (MHz)", showgrid=True, row=current_row, col=1
    )
    fig.update_yaxes(title_text="Magnitude", showgrid=True, row=current_row, col=1)
    current_row += 1

    # Real and imaginary components
    fig.add_trace(
        go.Scatter(
            x=freq,
            y=real_part,
            mode="lines",
            name="Real",
            line=dict(color="red", width=1),
            hovertemplate="Freq: %{x:.3f} MHz<br>Real: %{y:.3e}<extra></extra>",
        ),
        row=current_row,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=freq,
            y=imag_part,
            mode="lines",
            name="Imaginary",
            line=dict(color="green", width=1),
            hovertemplate="Freq: %{x:.3f} MHz<br>Imaginary: %{y:.3e}<extra></extra>",
        ),
        row=current_row,
        col=1,
    )

    fig.update_xaxes(
        title_text="Frequency (MHz)", showgrid=True, row=current_row, col=1
    )
    fig.update_yaxes(title_text="Amplitude", showgrid=True, row=current_row, col=1)

    # Update layout
    height = 200 + 300 * panel_count  # Dynamic height based on number of panels
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        width=1200,
        height=height,
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def _plot_complex_ft_matplotlib(
    freq: np.ndarray,
    magnitude: np.ndarray,
    real_part: np.ndarray,
    imag_part: np.ndarray,
    title: str,
    figsize: Tuple[float, float],
    interactive: bool,
    fid,
    preprocessed_fid,
    show_raw_fid: bool,
    show_preprocessed_fid: bool,
    **kwargs,
):
    """Create matplotlib plot of ComplexFT with optional FID panels."""

    # Check if we need FID panels
    need_fid_panels = show_raw_fid or show_preprocessed_fid

    # Configure 16:9 aspect ratio (16x9 inches)
    fig_width = 16
    fig_height = 9
    fig = plt.figure(figsize=(fig_width, fig_height))

    if need_fid_panels:
        # 3 equally sized rows configuration:
        # Row 1: 2 columns (Raw FID | Preprocessed FID)
        # Row 2: 1 column (Magnitude spectrum)
        # Row 3: 1 column (Real/Imaginary)
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.2)
    else:
        # No FID panels: just 2 rows for spectrum data
        gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.3)

    if need_fid_panels:
        # Row 1: FID panels (2 columns)
        # Raw FID panel (left column)
        if show_raw_fid:
            ax_raw = fig.add_subplot(gs[0, 0])
            time_us = fid.time_array_us()
            ax_raw.plot(time_us, fid.data, "k-", linewidth=1, label="Raw FID")
            ax_raw.set_ylabel("Voltage")
            ax_raw.set_xlabel("Time (μs)")
            ax_raw.set_title("Raw FID Data")
            ax_raw.grid(True, alpha=0.3)

            # Add vertical lines for windowing bounds
            if hasattr(fid, "processing") and fid.processing:
                if fid.processing.start_us is not None:
                    ax_raw.axvline(
                        fid.processing.start_us,
                        color="r",
                        linestyle="--",
                        linewidth=2,
                        label=f"Start: {fid.processing.start_us:.1f} μs",
                    )
                if fid.processing.end_us is not None:
                    ax_raw.axvline(
                        fid.processing.end_us,
                        color="r",
                        linestyle="--",
                        linewidth=2,
                        label=f"End: {fid.processing.end_us:.1f} μs",
                    )
            ax_raw.legend()

        # Preprocessed FID panel (right column)
        if show_preprocessed_fid:
            ax_preproc = fig.add_subplot(gs[0, 1])
            time_us = (
                np.arange(len(preprocessed_fid.data)) * preprocessed_fid.spacing * 1e6
            )
            ax_preproc.plot(
                time_us,
                preprocessed_fid.data,
                "m-",
                linewidth=1,
                label="Preprocessed FID",
            )
            ax_preproc.set_ylabel("Voltage")
            ax_preproc.set_xlabel("Time (μs)")
            ax_preproc.set_title("Preprocessed FID Data")
            ax_preproc.grid(True, alpha=0.3)
            ax_preproc.legend()

        # Row 2: Magnitude plot (spans both columns)
        ax_mag = fig.add_subplot(gs[1, :])
        ax_mag.plot(freq, magnitude, "b-", linewidth=1, label="Magnitude")
        ax_mag.set_ylabel("Magnitude")
        ax_mag.set_title("Magnitude Spectrum")
        ax_mag.grid(True, alpha=0.3)
        ax_mag.legend()

        # Row 3: Real and imaginary components (spans both columns)
        ax_complex = fig.add_subplot(gs[2, :])
        ax_complex.plot(freq, real_part, "r-", linewidth=1, label="Real")
        ax_complex.plot(freq, imag_part, "g-", linewidth=1, label="Imaginary")
        ax_complex.set_xlabel("Frequency (MHz)")
        ax_complex.set_ylabel("Amplitude")
        ax_complex.set_title("Real and Imaginary Components")
        ax_complex.grid(True, alpha=0.3)
        ax_complex.legend()
    else:
        # No FID panels: traditional 2-row layout
        # Magnitude plot
        ax_mag = fig.add_subplot(gs[0, 0])
        ax_mag.plot(freq, magnitude, "b-", linewidth=1, label="Magnitude")
        ax_mag.set_ylabel("Magnitude")
        ax_mag.set_title("Magnitude Spectrum")
        ax_mag.grid(True, alpha=0.3)
        ax_mag.legend()

        # Real and imaginary components
        ax_complex = fig.add_subplot(gs[1, 0])
        ax_complex.plot(freq, real_part, "r-", linewidth=1, label="Real")
        ax_complex.plot(freq, imag_part, "g-", linewidth=1, label="Imaginary")
        ax_complex.set_xlabel("Frequency (MHz)")
        ax_complex.set_ylabel("Amplitude")
        ax_complex.set_title("Real and Imaginary Components")
        ax_complex.grid(True, alpha=0.3)
        ax_complex.legend()

    # Overall title
    fig.suptitle(title, fontsize=16, y=0.95)

    # Always call tight_layout for proper spacing
    fig.tight_layout()

    return fig


def plot_spectral_window(
    window: SpectralWindow,
    title: Optional[str] = None,
    backend: str = "plotly",
    show_peaks: bool = True,
    **kwargs,
) -> Any:
    """
    Plot a spectral window with optional peak annotations.

    Parameters
    ----------
    window : SpectralWindow
        SpectralWindow object to plot
    title : str, optional
        Plot title
    backend : str, optional
        Plotting backend: 'plotly' (default) or 'matplotlib'
    show_peaks : bool, optional
        Whether to annotate detected peaks (default: True)
    **kwargs
        Additional plotting arguments

    Returns
    -------
    figure
        Plotly Figure or matplotlib Figure object
    """

    # Generate title if not provided
    if title is None:
        title = (
            f"Spectral Window {window.window_id or ''} "
            f"({window.freq_range[0]:.1f} - {window.freq_range[1]:.1f} MHz)"
        )

    # Create ComplexFT-like object for plotting
    if backend == "plotly":
        fig = _plot_spectral_window_plotly(window, title, show_peaks, **kwargs)
    else:
        fig = _plot_spectral_window_matplotlib(window, title, show_peaks, **kwargs)

    return fig


def _plot_spectral_window_plotly(
    window: SpectralWindow, title: str, show_peaks: bool, **kwargs
):
    """Create plotly plot of spectral window."""

    freq = window.freq_array
    magnitude = window.magnitude_spectrum

    fig = go.Figure()

    # Main spectrum
    fig.add_trace(
        go.Scatter(
            x=freq,
            y=magnitude,
            mode="lines",
            name="Magnitude",
            line=dict(color="blue", width=1),
            hovertemplate="Freq: %{x:.3f} MHz<br>Magnitude: %{y:.3e}<extra></extra>",
        )
    )

    # Add peak markers if requested
    if show_peaks and window.n_peaks > 0:
        peak_freqs = [peak.frequency for peak in window.peaks]
        peak_intensities = [peak.intensity for peak in window.peaks]
        peak_labels = [
            (
                f"Peak {i+1}<br>SNR: {peak.snr:.1f}"
                if peak.snr is not None
                else f"Peak {i+1}"
            )
            for i, peak in enumerate(window.peaks)
        ]

        fig.add_trace(
            go.Scatter(
                x=peak_freqs,
                y=peak_intensities,
                mode="markers",
                name="Detected Peaks",
                marker=dict(color="red", size=8, symbol="x"),
                text=peak_labels,
                hovertemplate="%{text}<br>Freq: %{x:.3f} MHz<br>Intensity: %{y:.3e}<extra></extra>",
            )
        )

    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Frequency (MHz)",
        yaxis_title="Magnitude",
        width=800,
        height=400,
        hovermode="x unified",
        showlegend=True,
    )

    return fig


def _plot_spectral_window_matplotlib(
    window: SpectralWindow, title: str, show_peaks: bool, **kwargs
):
    """Create matplotlib plot of spectral window."""

    fig, ax = plt.subplots(figsize=(12, 6))

    freq = window.freq_array
    magnitude = window.magnitude_spectrum

    # Main spectrum
    ax.plot(freq, magnitude, "b-", linewidth=1, label="Magnitude")

    # Add peak markers if requested
    if show_peaks and window.n_peaks > 0:
        peak_freqs = [peak.frequency for peak in window.peaks]
        peak_intensities = [peak.intensity for peak in window.peaks]
        ax.plot(
            peak_freqs,
            peak_intensities,
            "rx",
            markersize=8,
            markeredgewidth=2,
            label="Detected Peaks",
        )

        # Annotate peaks
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
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    return fig


# Legacy placeholder functions for other plots - will be implemented in Phase 8
def plot_peaks(*args, **kwargs):
    """Placeholder for peak plotting."""
    raise NotImplementedError("Will be implemented in Phase 8")


def plot_windows(*args, **kwargs):
    """Placeholder for window plotting."""
    raise NotImplementedError("Will be implemented in Phase 8")
