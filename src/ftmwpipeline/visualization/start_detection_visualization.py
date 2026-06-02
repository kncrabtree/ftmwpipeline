"""Visualization for data-driven start-time detection.

A two-panel diagnostic of the Σ|FT|-vs-start_us sweep:

* top: the full sweep on a log y-axis -- the pre-chirp plateau, the chirp-end
  collapse, and the post-chirp floor, with the detected chirp-end and
  recommended start marked.
* bottom: a linear zoom on the post-chirp floor where the ringdown shoulder
  settles into the molecular tail.

:func:`plot_start_detection_from_file` re-runs detection (cheap at zpf=0) so the
CLI / Pipeline / functional-API surfaces share one orchestration.
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt

from ..core.start_detection_settings import StartDetectionSettings
from ..preprocessing.start_detection import StartDetectionResult


def _figsize_or_default(
    figsize: Optional[Tuple[float, float]], default: Tuple[float, float]
) -> Tuple[float, float]:
    return figsize if figsize is not None else default


def plot_start_detection(
    result: StartDetectionResult,
    *,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """Render the start-detection sweep diagnostic from a result."""
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=_figsize_or_default(figsize, (10, 8))
    )
    starts = result.starts_us
    summag = result.sum_magnitude
    band = (
        f"{result.band_mhz[0]:.0f}-{result.band_mhz[1]:.0f} MHz"
        if result.band_mhz is not None
        else "full spectrum"
    )

    ax_top.semilogy(starts, summag, "-", lw=1, color="C0")
    ax_top.axvline(
        result.chirp_end_us,
        color="r",
        ls=":",
        label=f"chirp-end {result.chirp_end_us:.2f} us",
    )
    ax_top.axvline(
        result.start_us,
        color="g",
        ls="--",
        label=f"recommended start {result.start_us:.2f} us",
    )
    ax_top.set_ylabel("Σ|FT| over band")
    ax_top.legend(loc="upper right", fontsize=8)
    ax_top.set_title(title or f"Start-time detection (band {band}, zpf=0, unapodized)")
    if not result.chirp_detected:
        ax_top.text(
            0.02,
            0.05,
            "no chirp collapse detected",
            transform=ax_top.transAxes,
            color="red",
            fontsize=9,
        )

    # Linear zoom on the post-chirp floor; clip the collapse spike so the
    # ringdown shoulder settling into the molecular tail is legible.
    zoom = (starts > result.chirp_end_us - 0.1) & (starts < result.chirp_end_us + 3.0)
    ax_bot.plot(starts[zoom], summag[zoom], "-", lw=1, color="C0")
    ax_bot.axvline(result.start_us, color="g", ls="--")
    if result.floor > 0:
        ax_bot.set_ylim(0, result.floor * 8.0)
    ax_bot.set_xlabel("start_us")
    ax_bot.set_ylabel("Σ|FT| (post-chirp floor, linear)")

    fig.tight_layout()
    return fig


def plot_start_detection_from_file(
    file_path: str,
    *,
    settings: Optional[StartDetectionSettings] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """Run detection on ``file_path`` (no stamping) and plot the sweep."""
    from .._internal.start_detection_impl import detect_start_time_impl

    result = detect_start_time_impl(file_path, settings=settings, stamp=False)[
        "start_detection"
    ]
    return plot_start_detection(result, figsize=figsize, title=title)


__all__ = ["plot_start_detection", "plot_start_detection_from_file"]
