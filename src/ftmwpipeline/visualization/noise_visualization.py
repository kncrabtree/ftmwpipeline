"""
Noise estimation diagnostic plots for FTMW spectroscopy.

This module provides visualization functions for the scatter noise estimator,
overlaying the noise mask and the per-bin σ estimate (with 3×/5×σ reference
levels) on the active-FT magnitude spectrum.
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..preprocessing.noise_estimation import NoiseResult
from .report_style import AGGIE_BLUE, POPPY, QUAD, apply_bare_style, resolve_title


def plot_noise_estimation(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    noise_result: NoiseResult,
    y_max_factor: float = 20.0,
    figsize: Tuple[float, float] = (16, 6),
    title: Optional[str] = None,
    show_noise_points: bool = True,
) -> plt.Figure:
    """
    Create the diagnostic plot for noise estimation.

    Shows the magnitude spectrum, the noise points, and the per-bin σ estimate
    (with 3×/5×σ reference levels) in a single wide plot.

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
        Custom title. ``None`` uses the default; ``""`` suppresses it.
    show_noise_points : bool, default=True
        Whether to highlight noise points

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """

    # Create single subplot with wide aspect ratio
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    resolved_title = resolve_title(title, "Noise Estimation Diagnostics")
    if resolved_title:
        fig.suptitle(resolved_title, fontsize=14, fontweight="bold")

    # Frequencies are already in MHz
    freq_mhz = frequencies

    # Calculate y-axis limits
    median_rms = np.median(noise_result.rms_noise)
    y_max = y_max_factor * median_rms

    # Plot full spectrum
    ax.plot(
        freq_mhz, magnitudes, color=AGGIE_BLUE, linewidth=0.8, alpha=0.8, label="Spectrum"
    )

    # Plot noise points using masked array approach
    if show_noise_points:
        # Create masked array where True = noise points
        noise_magnitudes = np.ma.array(magnitudes, mask=~noise_result.noise_mask)
        ax.plot(
            freq_mhz, noise_magnitudes, color=POPPY, alpha=0.7, label="Noise Points"
        )

    # Plot RMS noise estimate and multiples
    ax.plot(
        freq_mhz, noise_result.rms_noise, color=QUAD, linewidth=2, label="RMS Noise"
    )
    ax.plot(
        freq_mhz,
        3 * noise_result.rms_noise,
        color=QUAD,
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label="3×RMS",
    )
    ax.plot(
        freq_mhz,
        5 * noise_result.rms_noise,
        color=QUAD,
        linestyle=":",
        linewidth=1,
        alpha=0.7,
        label="5×RMS",
    )
    ax.fill_between(freq_mhz, 0, noise_result.rms_noise, alpha=0.2, color=QUAD)

    # Set limits and labels
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Frequency (MHz)", fontsize=12)
    ax.set_ylabel("Magnitude", fontsize=12)
    ax.legend(loc="upper right")
    apply_bare_style(ax)

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
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="white", edgecolor="#d0d4d9", alpha=0.85
        ),
    )

    plt.tight_layout()
    return fig


def _compile_noise_statistics_summary(noise_result: NoiseResult) -> str:
    """Compile brief noise statistics for single-plot display."""

    info = noise_result.bin_info
    stats = []
    stats.append(f"Strategy: {info.get('algorithm', 'unknown')}")
    stats.append(f"Region windows: {info.get('n_region_windows', 'unknown')}")
    stats.append(f"Noise fraction: {info.get('noise_fraction', 0):.3f}")

    rms_mean = np.mean(noise_result.rms_noise)
    rms_std = np.std(noise_result.rms_noise)
    stats.append(f"RMS: {rms_mean:.2e} ± {rms_std:.2e}")

    return "\n".join(stats)
