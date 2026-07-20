"""
Stage 4 window-assignment diagnostic plot.

Overlays the :class:`~ftmwpipeline.core.data_structures.WindowPlan` on the
user's magnitude spectrum: each fit window's span, its free peaks, its fixed
contributors, and -- in a lower panel -- the rolling complex-edge coherence
statistic with the ``T_edge`` threshold that drove the partition. Mirrors the
matplotlib pattern of :mod:`ftmwpipeline.visualization.peak_visualization`.
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..core.data_structures import Peak, WindowPlan
from ..preprocessing.edge_coherence import coherence_curve


def plot_window_plan(
    frequencies: np.ndarray,
    complex_spectrum: np.ndarray,
    rms_noise: np.ndarray,
    peaks: List[Peak],
    plan: WindowPlan,
    figsize: Tuple[float, float] = (16, 8),
    title: Optional[str] = None,
    y_max_factor: float = 25.0,
) -> plt.Figure:
    """Plot a Stage 4 window plan over the spectrum.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency axis (MHz) of the active-FT surface.
    complex_spectrum : np.ndarray
        Complex FT of the active-FT surface.
    rms_noise : np.ndarray
        Persisted Stage 2 per-point RMS noise.
    peaks : list of Peak
        The full Stage 3 peak list (``free_peak_indices`` index into it).
    plan : WindowPlan
        The Stage 4 window plan to display.
    figsize : tuple, default (16, 8)
    title : str, optional
    y_max_factor : float, default 25.0
        Spectrum-panel y-axis headroom above the tallest peak.

    Returns
    -------
    matplotlib.figure.Figure
    """
    magnitude = np.abs(np.asarray(complex_spectrum, dtype=complex))

    from .report_style import (
        AGGIE_BLUE,
        AGGIE_GOLD,
        DOUBLE_DECKER,
        PINOT,
        POPPY,
        apply_bare_style,
        resolve_title,
        set_log_spectrum_ylim,
    )

    # De-ramp to the active-region turn-on so the displayed S_coh matches the
    # statistic that drove the plan (see leakage-detection-rework).
    ordered_freq, rolling, threshold, edge_m = coherence_curve(
        frequencies, complex_spectrum, rms_noise, plan.parameters
    )

    fig, (ax, ax_stat) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax.plot(
        frequencies,
        magnitude,
        lw=0.5,
        color="0.45",
        label="magnitude spectrum",
        zorder=1,
    )

    # Fit-window spans, uniformly shaded (Stage 4 is purely structural; no
    # difficulty grading).
    span_done = False
    for w in plan.windows:
        ax.axvspan(
            w.freq_range[0],
            w.freq_range[1],
            color=AGGIE_GOLD,
            alpha=0.15,
            zorder=0,
            label=None if span_done else "fit window",
        )
        span_done = True

    # Free peaks and fixed contributors.
    free_done = fixed_done = False
    for w in plan.windows:
        for li in w.free_peak_indices:
            if 0 <= li < len(peaks):
                p = peaks[li]
                ax.scatter(
                    p.frequency,
                    p.intensity,
                    s=28,
                    color=AGGIE_BLUE,
                    marker="o",
                    zorder=5,
                    label=None if free_done else "free peak",
                )
                free_done = True
        for fc in w.fixed_contributors:
            if 0 <= fc.peak_index < len(peaks):
                p = peaks[fc.peak_index]
                ax.scatter(
                    p.frequency,
                    p.intensity,
                    s=70,
                    facecolors="none",
                    edgecolors=POPPY,
                    marker="s",
                    linewidths=1.5,
                    zorder=6,
                    label=None if fixed_done else "fixed contributor",
                )
                fixed_done = True

    top = float(np.max(magnitude)) if magnitude.size else 1.0
    set_log_spectrum_ylim(ax, rms_noise, top, y_max_factor)
    ax.set_ylabel("Magnitude")
    resolved_title = resolve_title(title, "Stage 4 Window Assignment")
    if resolved_title:
        ax.set_title(resolved_title)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    # Rolling coherence statistic panel.
    ax_stat.plot(ordered_freq, rolling, lw=0.6, color=PINOT)
    ax_stat.axhline(
        threshold,
        color=DOUBLE_DECKER,
        lw=1.0,
        ls="--",
        label=f"T_edge = {threshold:g}",
    )
    ax_stat.set_yscale("log")
    ax_stat.set_xlabel("Frequency (MHz)")
    ax_stat.set_ylabel(f"S_coh (M={edge_m})")
    ax_stat.legend(loc="upper right", fontsize=8)

    apply_bare_style(ax)
    apply_bare_style(ax_stat)
    fig.tight_layout()
    return fig
