"""
Stage 4 window-assignment diagnostic plot.

Overlays the :class:`~ftmwpipeline.core.data_structures.WindowPlan` on the
user's magnitude spectrum: each fit window's span (shaded by difficulty), its
free peaks, its fixed contributors, and -- in a lower panel -- the rolling
complex-edge coherence statistic with the ``T_edge`` threshold that drove the
partition. Mirrors the matplotlib pattern of
:mod:`ftmwpipeline.visualization.peak_visualization`.
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..core.data_structures import Peak, WindowDifficulty, WindowPlan
from ..preprocessing.edge_coherence import (
    DEFAULT_EDGE_M,
    DEFAULT_EDGE_THRESHOLD,
    rolling_coherence,
)
from ..preprocessing.leakage import deramp_to_active_start

_DIFFICULTY_COLOR = {
    WindowDifficulty.EASY: "tab:green",
    WindowDifficulty.HARD: "tab:red",
}


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
        Frequency axis (MHz) of the persisted user spectrum.
    complex_spectrum : np.ndarray
        Complex FT of the user spectrum.
    rms_noise : np.ndarray
        Canonical Stage 2 per-point RMS noise.
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
    edge_m = int(plan.parameters.get("edge_m", DEFAULT_EDGE_M))
    threshold = float(plan.parameters.get("edge_threshold", DEFAULT_EDGE_THRESHOLD))

    from .report_style import apply_bare_style, resolve_title

    order = np.argsort(frequencies)
    # De-ramp to the active-region turn-on so the displayed S_coh matches the
    # statistic that drove the plan (see leakage-detection-rework).
    referenced = deramp_to_active_start(
        np.asarray(frequencies, dtype=float),
        np.asarray(complex_spectrum, dtype=complex),
        float(plan.parameters.get("probe_freq_mhz", 0.0)),
        float(plan.parameters.get("start_us", 0.0)),
    )
    rolling = rolling_coherence(
        referenced[order],
        np.asarray(rms_noise)[order],
        band_m=edge_m,
    )
    ordered_freq = np.asarray(frequencies)[order]

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
        color="0.4",
        label="magnitude spectrum",
        zorder=1,
    )

    # Window spans shaded by difficulty.
    legended = set()
    for w in plan.windows:
        color = _DIFFICULTY_COLOR.get(w.difficulty, "tab:gray")
        label = None
        if w.difficulty not in legended:
            legended.add(w.difficulty)
            label = f"{w.difficulty.value} window"
        ax.axvspan(
            w.freq_range[0],
            w.freq_range[1],
            color=color,
            alpha=0.12,
            zorder=0,
            label=label,
        )
        if w.split_proposal is not None:
            ax.axvline(
                w.split_proposal,
                color="purple",
                lw=1.0,
                ls=":",
                zorder=3,
            )

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
                    color="black",
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
                    edgecolors="tab:blue",
                    marker="s",
                    linewidths=1.5,
                    zorder=6,
                    label=None if fixed_done else "fixed contributor",
                )
                fixed_done = True

    median_rms = float(np.median(rms_noise)) if len(rms_noise) else 1.0
    floor = max(median_rms * 0.1, 1e-12)
    top = float(np.max(magnitude)) if magnitude.size else 1.0
    ax.set_yscale("log")
    ax.set_ylim(floor, top * max(y_max_factor / 25.0, 1.2))
    ax.set_ylabel("Magnitude")
    resolved_title = resolve_title(title, "Stage 4 Window Assignment")
    if resolved_title:
        ax.set_title(resolved_title)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    # Rolling coherence statistic panel.
    ax_stat.plot(ordered_freq, rolling, lw=0.6, color="tab:purple")
    ax_stat.axhline(
        threshold,
        color="crimson",
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
