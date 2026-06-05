"""
Noise estimation diagnostic plots for FTMW spectroscopy.

This module provides visualization functions for noise estimation algorithms,
including bin boundaries, noise masks, and RMS estimates.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union
from ..preprocessing.noise_estimation import NoiseResult


def plot_noise_estimation(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    noise_result: NoiseResult,
    y_max_factor: float = 20.0,
    figsize: Tuple[float, float] = (16, 6),
    title: Optional[str] = None,
    show_bin_boundaries: bool = True,
    show_noise_points: bool = True,
    backend: str = "matplotlib",
) -> Union[plt.Figure, object]:
    """
    Create diagnostic plot for noise estimation.

    Shows original spectrum, noise points, bin boundaries, and RMS noise estimate
    in a single wide plot for better visibility.

    Parameters:
    -----------
    frequencies : np.ndarray
        Frequency values (MHz)
    magnitudes : np.ndarray
        Magnitude spectrum values
    noise_result : NoiseResult
        Result from estimate_noise_scatter
    y_max_factor : float, default=20.0
        Y-axis maximum as multiple of median RMS noise
    figsize : tuple, default=(16, 6)
        Figure size (width, height) in inches - wide aspect ratio
    title : str, optional
        Custom title for the plot
    show_bin_boundaries : bool, default=True
        Whether to show adaptive bin boundaries as dotted lines
    show_noise_points : bool, default=True
        Whether to highlight noise points
    backend : str, default="matplotlib"
        Plotting backend ("matplotlib" or "plotly")

    Returns:
    --------
    matplotlib.Figure or plotly.Figure
        The created figure object
    """

    if backend == "plotly":
        return _plot_noise_estimation_plotly(
            frequencies,
            magnitudes,
            noise_result,
            y_max_factor,
            title,
            show_bin_boundaries,
            show_noise_points,
        )
    else:
        return _plot_noise_estimation_matplotlib(
            frequencies,
            magnitudes,
            noise_result,
            y_max_factor,
            figsize,
            title,
            show_bin_boundaries,
            show_noise_points,
        )


def _plot_noise_estimation_matplotlib(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    noise_result: NoiseResult,
    y_max_factor: float,
    figsize: Tuple[float, float],
    title: Optional[str],
    show_bin_boundaries: bool,
    show_noise_points: bool,
) -> plt.Figure:
    """Create matplotlib version of noise estimation plot - single wide panel."""

    # Create single subplot with wide aspect ratio
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.suptitle(
        title or "Noise Estimation Diagnostics", fontsize=14, fontweight="bold"
    )

    # Frequencies are already in MHz
    freq_mhz = frequencies

    # Calculate y-axis limits
    median_rms = np.median(noise_result.rms_noise)
    y_max = y_max_factor * median_rms

    # Plot full spectrum
    ax.plot(freq_mhz, magnitudes, "b-", linewidth=0.8, alpha=0.8, label="Spectrum")

    # Plot noise points using masked array approach (like original)
    if show_noise_points:
        # Create masked array where True = noise points
        noise_magnitudes = np.ma.array(magnitudes, mask=~noise_result.noise_mask)
        ax.plot(
            freq_mhz, noise_magnitudes, color="orange", alpha=0.7, label="Noise Points"
        )

    # Plot RMS noise estimate and multiples
    ax.plot(freq_mhz, noise_result.rms_noise, "g-", linewidth=2, label="RMS Noise")
    ax.plot(
        freq_mhz,
        3 * noise_result.rms_noise,
        "g--",
        linewidth=1,
        alpha=0.7,
        label="3×RMS",
    )
    ax.plot(
        freq_mhz,
        5 * noise_result.rms_noise,
        "g:",
        linewidth=1,
        alpha=0.7,
        label="5×RMS",
    )
    ax.fill_between(freq_mhz, 0, noise_result.rms_noise, alpha=0.2, color="green")

    # Add bin boundaries as thin dotted lines
    if show_bin_boundaries and "bin_edges" in noise_result.bin_info:
        bin_edges = noise_result.bin_info["bin_edges"]

        # Limit number of lines if there are too many
        if len(bin_edges) > 50:
            step = max(1, len(bin_edges) // 30)
            bin_edges_to_show = bin_edges[::step]
            # Add note about subsampling
            ax.text(
                0.02,
                0.98,
                f"Showing {len(bin_edges_to_show)}/{len(bin_edges)} bin edges",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
            )
        else:
            bin_edges_to_show = bin_edges

        for edge in bin_edges_to_show:
            if 0 <= edge < len(frequencies):
                ax.axvline(
                    freq_mhz[edge], color="red", linestyle=":", alpha=0.6, linewidth=0.5
                )

    # Set limits and labels
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Frequency (MHz)", fontsize=12)
    ax.set_ylabel("Magnitude", fontsize=12)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Add summary statistics as text
    stats_text = _compile_noise_statistics_summary(noise_result)
    ax.text(
        0.02,
        0.85,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8),
    )

    plt.tight_layout()
    return fig


def _plot_noise_estimation_plotly(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    noise_result: NoiseResult,
    y_max_factor: float,
    title: Optional[str],
    show_bin_boundaries: bool,
    show_noise_points: bool,
):
    """Create plotly version of noise estimation plot."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError(
            "plotly is required for interactive plotting. Install with: pip install plotly"
        )

    # Frequencies are already in MHz
    freq_mhz = frequencies

    # Calculate y-axis limits
    median_rms = np.median(noise_result.rms_noise)
    y_max = y_max_factor * median_rms

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Spectrum with Noise Points",
            f'Adaptive Bins ({noise_result.bin_info.get("algorithm", "unknown")})',
            "RMS Noise Estimate",
            "Statistics",
        ],
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"type": "table"}],
        ],
    )

    # Panel 1: Spectrum with noise points
    fig.add_trace(
        go.Scatter(
            x=freq_mhz,
            y=magnitudes,
            mode="lines",
            name="Spectrum",
            line=dict(color="blue", width=1),
            opacity=0.7,
        ),
        row=1,
        col=1,
    )

    if show_noise_points:
        noise_freq = freq_mhz[noise_result.noise_mask]
        noise_mag = magnitudes[noise_result.noise_mask]
        fig.add_trace(
            go.Scatter(
                x=noise_freq,
                y=noise_mag,
                mode="markers",
                name="Noise Points",
                marker=dict(color="red", size=2, opacity=0.6),
            ),
            row=1,
            col=1,
        )

    # Panel 2: Bin boundaries
    fig.add_trace(
        go.Scatter(
            x=freq_mhz,
            y=magnitudes,
            mode="lines",
            name="Spectrum",
            line=dict(color="blue", width=1),
            opacity=0.7,
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    if show_bin_boundaries and "bin_edges" in noise_result.bin_info:
        bin_edges = noise_result.bin_info["bin_edges"]

        # Limit number of lines if there are too many
        if len(bin_edges) > 50:
            step = max(1, len(bin_edges) // 30)
            bin_edges_to_show = bin_edges[::step]
        else:
            bin_edges_to_show = bin_edges

        for edge in bin_edges_to_show:
            if 0 <= edge < len(frequencies):
                fig.add_vline(
                    x=freq_mhz[edge],
                    line_dash="dot",
                    line_color="red",
                    opacity=0.8,
                    line_width=1,
                    row=1,
                    col=2,
                )

    # Panel 3: RMS noise estimate
    fig.add_trace(
        go.Scatter(
            x=freq_mhz,
            y=magnitudes,
            mode="lines",
            name="Spectrum",
            line=dict(color="blue", width=1),
            opacity=0.7,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=freq_mhz,
            y=noise_result.rms_noise,
            mode="lines",
            name="RMS Noise",
            line=dict(color="green", width=2),
            fill="tozeroy",
            fillcolor="rgba(0,255,0,0.3)",
        ),
        row=2,
        col=1,
    )

    # Panel 4: Statistics table
    stats_data = _compile_noise_statistics_table(noise_result, frequencies, magnitudes)

    fig.add_trace(
        go.Table(
            header=dict(values=["Parameter", "Value"], fill_color="lightgray"),
            cells=dict(values=[stats_data["parameters"], stats_data["values"]]),
        ),
        row=2,
        col=2,
    )

    # Update layout
    fig.update_layout(
        title=title or "Noise Estimation Diagnostics", height=800, showlegend=True
    )

    # Set y-axis ranges
    for row, col in [(1, 1), (1, 2), (2, 1)]:
        fig.update_yaxes(range=[0, y_max], row=row, col=col)

    # Set axis labels
    for col in [1, 2]:
        fig.update_xaxes(title_text="Frequency (MHz)", row=2, col=col)
        fig.update_yaxes(title_text="Magnitude", row=1, col=col)
        fig.update_yaxes(title_text="Magnitude", row=2, col=col)

    return fig


def _compile_noise_statistics(
    noise_result: NoiseResult, frequencies: np.ndarray, magnitudes: np.ndarray
) -> str:
    """Compile noise estimation statistics into formatted text."""

    stats = []
    stats.append("NOISE ESTIMATION STATISTICS")
    stats.append("=" * 30)

    # Basic statistics
    stats.append(f"Total points: {len(frequencies):,}")
    stats.append(f"Noise points: {np.sum(noise_result.noise_mask):,}")
    stats.append(
        f"Noise fraction: {noise_result.bin_info.get('noise_fraction', 0):.3f}"
    )
    stats.append("")

    # RMS noise statistics
    rms_mean = np.mean(noise_result.rms_noise)
    rms_std = np.std(noise_result.rms_noise)
    rms_min = np.min(noise_result.rms_noise)
    rms_max = np.max(noise_result.rms_noise)

    stats.append("RMS NOISE STATISTICS")
    stats.append("-" * 20)
    stats.append(f"Mean: {rms_mean:.2e}")
    stats.append(f"Std Dev: {rms_std:.2e}")
    stats.append(f"Min: {rms_min:.2e}")
    stats.append(f"Max: {rms_max:.2e}")
    stats.append(f"Range: {rms_max/rms_min:.1f}x")
    stats.append("")

    # Algorithm parameters
    stats.append("ALGORITHM PARAMETERS")
    stats.append("-" * 20)
    stats.append(f"Strategy: {noise_result.bin_info.get('algorithm', 'unknown')}")
    stats.append(f"Number of bins: {noise_result.bin_info.get('n_bins', 'unknown')}")
    stats.append(
        f"Smoothing window: {noise_result.bin_info.get('smoothing_window', 'unknown')}"
    )

    return "\n".join(stats)


def _compile_noise_statistics_table(
    noise_result: NoiseResult, frequencies: np.ndarray, magnitudes: np.ndarray
) -> dict:
    """Compile noise estimation statistics into table format for plotly."""

    # RMS noise statistics
    rms_mean = np.mean(noise_result.rms_noise)
    rms_std = np.std(noise_result.rms_noise)
    rms_min = np.min(noise_result.rms_noise)
    rms_max = np.max(noise_result.rms_noise)

    parameters = [
        "Total Points",
        "Noise Points",
        "Noise Fraction",
        "RMS Mean",
        "RMS Std Dev",
        "RMS Min",
        "RMS Max",
        "RMS Range",
        "Strategy",
        "Number of Bins",
        "Smoothing Window",
    ]

    values = [
        f"{len(frequencies):,}",
        f"{np.sum(noise_result.noise_mask):,}",
        f"{noise_result.bin_info.get('noise_fraction', 0):.3f}",
        f"{rms_mean:.2e}",
        f"{rms_std:.2e}",
        f"{rms_min:.2e}",
        f"{rms_max:.2e}",
        f"{rms_max/rms_min:.1f}x",
        f"{noise_result.bin_info.get('algorithm', 'unknown')}",
        f"{noise_result.bin_info.get('n_bins', 'unknown')}",
        f"{noise_result.bin_info.get('smoothing_window', 'unknown')}",
    ]

    return {"parameters": parameters, "values": values}


def _compile_noise_statistics_summary(noise_result: NoiseResult) -> str:
    """Compile brief noise statistics for single-plot display."""

    stats = []
    stats.append(f"Strategy: {noise_result.bin_info.get('algorithm', 'unknown')}")
    stats.append(f"Bins: {noise_result.bin_info.get('n_bins', 'unknown')}")
    stats.append(
        f"Noise fraction: {noise_result.bin_info.get('noise_fraction', 0):.3f}"
    )

    rms_mean = np.mean(noise_result.rms_noise)
    rms_std = np.std(noise_result.rms_noise)
    stats.append(f"RMS: {rms_mean:.2e} ± {rms_std:.2e}")

    return "\n".join(stats)
