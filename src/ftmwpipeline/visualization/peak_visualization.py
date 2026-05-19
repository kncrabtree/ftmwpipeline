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


def _plot_snr_histogram(
    ax: plt.Axes,
    peaks: List[Peak],
    promotion_min_snr: Optional[float],
) -> None:
    """Draw the user-grid SNR distribution with the promotion cutoff marked.

    This is the curation view: it shows where the promotion threshold lands in
    the peak population (the noise hump vs the real-line tail) so the cutoff
    can be chosen deliberately before peaks move to Stage 4.
    """
    snrs = np.array(
        [p.snr for p in peaks if p.snr is not None and p.snr > 0], dtype=float
    )
    if snrs.size == 0:
        ax.text(0.5, 0.5, "no SNR data", ha="center", va="center",
                transform=ax.transAxes)
        return
    bins = np.logspace(
        np.log10(max(snrs.min(), 0.5)), np.log10(snrs.max()), 60
    )
    ax.hist(snrs, bins=bins, color="steelblue", alpha=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("peak SNR (user grid)")
    ax.set_ylabel("count")
    if promotion_min_snr is not None:
        ax.axvline(
            promotion_min_snr,
            color="crimson",
            lw=2,
            label=f"promotion cutoff = {promotion_min_snr:g}",
        )
        below = int((snrs < promotion_min_snr).sum())
        ax.text(
            0.02,
            0.95,
            f"{snrs.size - below:,} promoted / {snrs.size:,} detected "
            f"({below:,} below cutoff)",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title("SNR distribution")


def plot_peak_detection(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    rms_noise: np.ndarray,
    peaks: List[Peak],
    figsize: Tuple[float, float] = (16, 6),
    title: Optional[str] = None,
    y_max_factor: float = 25.0,
    backend: str = "matplotlib",
    snr_histogram: bool = False,
    promotion_min_snr: Optional[float] = None,
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
    snr_histogram : bool, default False
        If True, add a second panel below the overlay showing the user-grid
        SNR distribution with the promotion cutoff marked (curation view).
    promotion_min_snr : float, optional
        Promotion cutoff drawn on the SNR-histogram panel.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if backend != "matplotlib":
        raise ValueError(
            f"Unsupported backend {backend!r}; only 'matplotlib' is available"
        )

    if snr_histogram:
        fig, (ax, ax_hist) = plt.subplots(
            2,
            1,
            figsize=(figsize[0], figsize[1] + 4),
            gridspec_kw={"height_ratios": [3, 1]},
        )
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax_hist = None
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
    if ax_hist is not None:
        _plot_snr_histogram(ax_hist, peaks, promotion_min_snr)
    fig.tight_layout()
    return fig
