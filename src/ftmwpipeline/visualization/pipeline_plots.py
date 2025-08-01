"""
Pipeline plotting functions.

This module contains interactive plotting functions for FTMW pipeline visualization.
Supports both matplotlib and plotly backends.
"""

import numpy as np
from typing import Optional, Union, Tuple, Any
import warnings

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from ..core.data_structures import ComplexFT, SpectralWindow


def plot_complex_ft(complex_ft: ComplexFT, 
                   freq_range: Optional[Tuple[float, float]] = None,
                   title: Optional[str] = None,
                   backend: str = 'plotly',
                   interactive: bool = True,
                   figsize: Tuple[float, float] = (16, 8),
                   **kwargs) -> Any:
    """
    Create interactive plot of ComplexFT showing magnitude and real/imaginary components.
    
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
        Figure size in inches (width, height). Default: (16, 8)
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
    if backend == 'plotly' and not HAS_PLOTLY:
        if HAS_MATPLOTLIB:
            warnings.warn("Plotly not available, falling back to matplotlib", UserWarning)
            backend = 'matplotlib'
            interactive = False
        else:
            raise ImportError("Neither plotly nor matplotlib is available")
    elif backend == 'matplotlib' and not HAS_MATPLOTLIB:
        if HAS_PLOTLY:
            warnings.warn("Matplotlib not available, falling back to plotly", UserWarning)
            backend = 'plotly'
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
    
    if backend == 'plotly':
        return _plot_complex_ft_plotly(freq, magnitude, real_part, imag_part, title, **kwargs)
    else:
        return _plot_complex_ft_matplotlib(freq, magnitude, real_part, imag_part, title, 
                                          figsize, interactive, **kwargs)


def _plot_complex_ft_plotly(freq: np.ndarray, magnitude: np.ndarray, 
                           real_part: np.ndarray, imag_part: np.ndarray, 
                           title: str, **kwargs):
    """Create plotly interactive plot of ComplexFT."""
    
    # Create subplots with wide aspect ratio
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Magnitude Spectrum', 'Real and Imaginary Components'),
        vertical_spacing=0.08,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Magnitude plot
    fig.add_trace(
        go.Scatter(
            x=freq, 
            y=magnitude,
            mode='lines',
            name='Magnitude',
            line=dict(color='blue', width=1),
            hovertemplate='Freq: %{x:.3f} MHz<br>Magnitude: %{y:.3e}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Real and imaginary components
    fig.add_trace(
        go.Scatter(
            x=freq,
            y=real_part,
            mode='lines', 
            name='Real',
            line=dict(color='red', width=1),
            hovertemplate='Freq: %{x:.3f} MHz<br>Real: %{y:.3e}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=freq,
            y=imag_part,
            mode='lines',
            name='Imaginary', 
            line=dict(color='green', width=1),
            hovertemplate='Freq: %{x:.3f} MHz<br>Imaginary: %{y:.3e}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        width=1200,
        height=600,
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Update axes
    fig.update_xaxes(title_text="Frequency (MHz)", showgrid=True, row=1, col=1)
    fig.update_xaxes(title_text="Frequency (MHz)", showgrid=True, row=2, col=1)
    fig.update_yaxes(title_text="Magnitude", showgrid=True, row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", showgrid=True, row=2, col=1)
    
    return fig


def _plot_complex_ft_matplotlib(freq: np.ndarray, magnitude: np.ndarray,
                               real_part: np.ndarray, imag_part: np.ndarray,
                               title: str, figsize: Tuple[float, float], 
                               interactive: bool, **kwargs):
    """Create matplotlib plot of ComplexFT."""
    
    # Create figure with wide aspect ratio
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.3)
    
    # Magnitude plot
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(freq, magnitude, 'b-', linewidth=1, label='Magnitude')
    ax1.set_ylabel('Magnitude')
    ax1.set_title('Magnitude Spectrum')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Real and imaginary components
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(freq, real_part, 'r-', linewidth=1, label='Real')
    ax2.plot(freq, imag_part, 'g-', linewidth=1, label='Imaginary')
    ax2.set_xlabel('Frequency (MHz)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Real and Imaginary Components')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Overall title
    fig.suptitle(title, fontsize=14)
    
    # Enable interactive features if requested
    if interactive:
        plt.tight_layout()
        
    return fig


def plot_spectral_window(window: SpectralWindow,
                        title: Optional[str] = None,
                        backend: str = 'plotly',
                        show_peaks: bool = True,
                        **kwargs) -> Any:
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
        title = f"Spectral Window {window.window_id or ''} " \
                f"({window.freq_range[0]:.1f} - {window.freq_range[1]:.1f} MHz)"
    
    # Create ComplexFT-like object for plotting
    if backend == 'plotly':
        fig = _plot_spectral_window_plotly(window, title, show_peaks, **kwargs)
    else:
        fig = _plot_spectral_window_matplotlib(window, title, show_peaks, **kwargs)
    
    return fig


def _plot_spectral_window_plotly(window: SpectralWindow, title: str, 
                                show_peaks: bool, **kwargs):
    """Create plotly plot of spectral window."""
    
    freq = window.freq_array
    magnitude = window.magnitude_spectrum
    
    fig = go.Figure()
    
    # Main spectrum
    fig.add_trace(
        go.Scatter(
            x=freq,
            y=magnitude,
            mode='lines',
            name='Magnitude',
            line=dict(color='blue', width=1),
            hovertemplate='Freq: %{x:.3f} MHz<br>Magnitude: %{y:.3e}<extra></extra>'
        )
    )
    
    # Add peak markers if requested
    if show_peaks and window.n_peaks > 0:
        peak_freqs = [peak.frequency for peak in window.peaks]
        peak_intensities = [peak.intensity for peak in window.peaks]
        peak_labels = [f"Peak {i+1}<br>SNR: {peak.snr:.1f}" if peak.snr is not None 
                      else f"Peak {i+1}" for i, peak in enumerate(window.peaks)]
        
        fig.add_trace(
            go.Scatter(
                x=peak_freqs,
                y=peak_intensities, 
                mode='markers',
                name='Detected Peaks',
                marker=dict(color='red', size=8, symbol='x'),
                text=peak_labels,
                hovertemplate='%{text}<br>Freq: %{x:.3f} MHz<br>Intensity: %{y:.3e}<extra></extra>'
            )
        )
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Frequency (MHz)",
        yaxis_title="Magnitude",
        width=800,
        height=400,
        hovermode='x unified',
        showlegend=True
    )
    
    return fig


def _plot_spectral_window_matplotlib(window: SpectralWindow, title: str,
                                    show_peaks: bool, **kwargs):
    """Create matplotlib plot of spectral window."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    freq = window.freq_array
    magnitude = window.magnitude_spectrum
    
    # Main spectrum
    ax.plot(freq, magnitude, 'b-', linewidth=1, label='Magnitude')
    
    # Add peak markers if requested
    if show_peaks and window.n_peaks > 0:
        peak_freqs = [peak.frequency for peak in window.peaks]
        peak_intensities = [peak.intensity for peak in window.peaks]
        ax.plot(peak_freqs, peak_intensities, 'rx', markersize=8, 
               markeredgewidth=2, label='Detected Peaks')
        
        # Annotate peaks
        for i, peak in enumerate(window.peaks):
            label = f"P{i+1}"
            if peak.snr is not None:
                label += f"\nSNR: {peak.snr:.1f}"
            ax.annotate(label, (peak.frequency, peak.intensity),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, ha='left')
    
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Magnitude')
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


# Alias for backward compatibility
plot_spectrum = plot_complex_ft
