"""
Stage 3 peak-detection diagnostic plot.

Overlays the classified, detected peaks on the (full-resolution) magnitude
spectrum with the detection floor, colour-coded by SNR classification and
marker-styled by which pass found each peak. Mirrors the matplotlib pattern of
:mod:`ftmwpipeline.visualization.noise_visualization`.
"""

from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from ..core.data_structures import Peak, PeakClassification

_CLASS_COLOR = {
    PeakClassification.WEAK: "tab:green",
    PeakClassification.MEDIUM: "tab:orange",
    PeakClassification.STRONG: "tab:red",
    None: "tab:gray",
}


def plot_peak_detection(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    rms_noise: np.ndarray,
    peaks: List[Peak],
    figsize: Tuple[float, float] = (16, 6),
    title: Optional[str] = None,
    y_max_factor: float = 25.0,
    backend: str = "matplotlib",
) -> Union[plt.Figure, object]:
    """Plot detected/classified peaks over the spectrum.

    Parameters
    ----------
    frequencies, magnitudes, rms_noise : np.ndarray
        The spectrum the peaks were detected on and its per-point noise.
    peaks : list of Peak
        Classified peaks (each may carry ``properties['detection_pass']``).
    figsize : tuple, default (16, 6)
    title : str, optional
    y_max_factor : float, default 25.0
        Y-axis max as a multiple of the median RMS noise.
    backend : str, default "matplotlib"
        Only ``"matplotlib"`` is supported.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if backend != "matplotlib":
        raise ValueError(
            f"Unsupported backend {backend!r}; only 'matplotlib' is available"
        )

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        frequencies,
        magnitudes,
        lw=0.5,
        color="0.4",
        label="magnitude spectrum",
        zorder=1,
    )

    legended = set()
    for p in peaks:
        cls = (
            p.classification
            if isinstance(p.classification, PeakClassification)
            else None
        )
        color = _CLASS_COLOR.get(cls, "tab:gray")
        is_gap = p.properties.get("detection_pass") == "gap"
        marker = "^" if is_gap else "o"
        key = (cls, is_gap)
        label = None
        if key not in legended:
            legended.add(key)
            cls_name = cls.value if cls is not None else "unclassified"
            pass_name = "gap" if is_gap else "primary"
            label = f"{cls_name} ({pass_name})"
        ax.scatter(
            p.frequency,
            p.intensity,
            s=30,
            color=color,
            marker=marker,
            edgecolors="black",
            linewidths=0.3,
            zorder=5,
            label=label,
        )

    ax.plot(
        frequencies,
        rms_noise,
        lw=0.8,
        color="tab:blue",
        alpha=0.7,
        label="rms noise",
        zorder=2,
    )

    # Log y: noise floor and the 100s-of-x stronger lines (and their
    # markers, now at the true apex magnitude) are both legible. y_max_factor
    # only sets headroom above the tallest peak.
    median_rms = float(np.median(rms_noise)) if len(rms_noise) else 1.0
    floor = max(median_rms * 0.1, 1e-12)
    top = float(np.max(magnitudes)) if len(magnitudes) else 1.0
    if peaks:
        top = max(top, max(p.intensity for p in peaks))
    ax.set_yscale("log")
    ax.set_ylim(floor, top * max(y_max_factor / 25.0, 1.2))
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(title or "Stage 3 Peak Detection")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    return fig
