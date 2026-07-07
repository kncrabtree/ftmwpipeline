"""Visualization for data-driven start-time detection.

A two-panel diagnostic of the Σ|FT|-vs-start_us sweep:

* top: the full sweep on a log y-axis -- the pre-chirp plateau, the chirp-end
  collapse, and the post-chirp floor, with the detected chirp-end and
  recommended start marked.
* bottom: a linear zoom on the post-chirp floor where the ringdown shoulder
  settles into the molecular tail.

:func:`plot_start_detection_from_file` re-runs detection (cheap, native-length)
so the CLI / Pipeline / functional-API surfaces share one orchestration.

:func:`plot_stage0_overview_from_file` is the richer Stage 0 report
diagnostic: the raw FID (time domain) plus -- when start-time detection has
been run on the file -- the same sweep, both on aligned time axes and
annotated with the chirp end, the effective start, and the ring-down guard
margin, however each was actually selected (declared, auto-detected, or
manually overridden).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ..core.start_detection_settings import StartDetectionSettings
from ..preprocessing.start_detection import StartDetectionResult
from .report_style import AGGIE_BLUE, POPPY, QUAD, apply_bare_style, resolve_title

if TYPE_CHECKING:
    from .._internal.start_detection_impl import StartProvenance


def _figsize_or_default(
    figsize: Optional[Tuple[float, float]], default: Tuple[float, float]
) -> Tuple[float, float]:
    return figsize if figsize is not None else default


def _draw_sweep_top(
    ax: "matplotlib.axes.Axes",
    starts: np.ndarray,
    summag: np.ndarray,
    *,
    chirp_end_us: float,
    start_us: float,
    chirp_detected: bool,
    band_label: str,
    title: Optional[str],
    start_label: str = "recommended start",
    start_color: str = QUAD,
) -> None:
    """The full log-scale Σ|FT|-vs-start sweep with chirp-end/start markers."""
    ax.semilogy(starts, summag, "-", lw=1, color=AGGIE_BLUE)
    ax.axvline(
        chirp_end_us, color=POPPY, ls=":", label=f"chirp-end {chirp_end_us:.2f} us"
    )
    ax.axvline(
        start_us, color=start_color, ls="--", label=f"{start_label} {start_us:.2f} us"
    )
    ax.set_ylabel("Σ|FT| over band")
    ax.legend(loc="upper right", fontsize=8)
    resolved_title = resolve_title(
        title, f"Start-time detection (band {band_label}, unapodized)"
    )
    if resolved_title:
        ax.set_title(resolved_title)
    if not chirp_detected:
        ax.text(
            0.02,
            0.05,
            "no chirp collapse detected",
            transform=ax.transAxes,
            color="red",
            fontsize=9,
        )


def _draw_sweep_bottom(
    ax: "matplotlib.axes.Axes",
    starts: np.ndarray,
    summag: np.ndarray,
    *,
    chirp_end_us: float,
    start_us: float,
    floor: float,
    start_color: str = QUAD,
) -> None:
    """A linear zoom on the post-chirp floor where the ringdown shoulder settles."""
    zoom = (starts > chirp_end_us - 0.1) & (starts < chirp_end_us + 3.0)
    ax.plot(starts[zoom], summag[zoom], "-", lw=1, color=AGGIE_BLUE)
    ax.axvline(start_us, color=start_color, ls="--")
    if floor > 0:
        ax.set_ylim(0, floor * 8.0)
    ax.set_xlabel("start_us")
    ax.set_ylabel("Σ|FT| (post-chirp floor, linear)")


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
    band = (
        f"{result.band_mhz[0]:.0f}-{result.band_mhz[1]:.0f} MHz"
        if result.band_mhz is not None
        else "full spectrum"
    )
    _draw_sweep_top(
        ax_top,
        result.starts_us,
        result.sum_magnitude,
        chirp_end_us=result.chirp_end_us,
        start_us=result.start_us,
        chirp_detected=result.chirp_detected,
        band_label=band,
        title=title,
    )
    _draw_sweep_bottom(
        ax_bot,
        result.starts_us,
        result.sum_magnitude,
        chirp_end_us=result.chirp_end_us,
        start_us=result.start_us,
        floor=result.floor,
    )

    apply_bare_style(ax_top)
    apply_bare_style(ax_bot)
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


def _mark_chirp_and_start(
    ax: "matplotlib.axes.Axes",
    *,
    chirp_start_us: Optional[float],
    chirp_end_us: Optional[float],
    recommended_start_us: Optional[float],
    start_us: Optional[float],
    is_manual: bool,
) -> None:
    """Draw chirp-start/chirp-end/guard-margin/start markers on a time axis.

    ``recommended_start_us`` is the Stage 0 recommendation (chirp end + guard
    margin); ``start_us`` is the value Stage 1 actually used. The two coincide
    unless ``is_manual`` (a manual override), in which case both the guard
    margin (up to the recommendation) and the manually-set start are shown.
    """
    any_marker = False
    if chirp_start_us is not None:
        ax.axvline(
            chirp_start_us,
            color=AGGIE_BLUE,
            ls="-.",
            lw=1,
            label=f"chirp start {chirp_start_us:.2f} us",
        )
        any_marker = True
    if chirp_end_us is not None:
        ax.axvline(
            chirp_end_us, color=POPPY, ls=":", label=f"chirp end {chirp_end_us:.2f} us"
        )
        any_marker = True
    if (
        chirp_end_us is not None
        and recommended_start_us is not None
        and recommended_start_us > chirp_end_us
    ):
        ax.axvspan(
            chirp_end_us,
            recommended_start_us,
            color=QUAD,
            alpha=0.12,
            label=f"guard margin {recommended_start_us - chirp_end_us:.2f} us",
        )
    if start_us is not None:
        ax.axvline(
            start_us,
            color="0.15" if is_manual else QUAD,
            ls="--",
            label=(
                f"start (manual) {start_us:.2f} us"
                if is_manual
                else f"start {start_us:.2f} us"
            ),
        )
        any_marker = True
    if any_marker:
        ax.legend(loc="upper right", fontsize=7)


def plot_stage0_overview_from_file(
    file_path: str,
    *,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """Render the Stage 0 start-time-selection diagnostic for the report.

    Always shows the raw FID (time domain) so the excitation chirp,
    switch-bounce ring-down, and molecular onset are visible directly in the
    data. When start-time detection has been run on this file (``start run``
    / ``Pipeline.detect_start_time`` / ``api.detect_start_time``, at least
    once), the Σ|FT|-vs-start sweep it produced is replayed underneath on the
    same time axis, using the exact settings that were persisted -- not
    re-run with guessed defaults. All panels are annotated with the chirp
    end, the effective start, and the ring-down guard margin, however each
    was actually selected (declared, auto-detected, or manually overridden).
    """
    from .._internal.stage0_impl import load_fid_from_pipeline_impl
    from .._internal.start_detection_impl import resolve_start_provenance
    from ..preprocessing.start_detection import detect_start_time

    fid = load_fid_from_pipeline_impl(file_path)
    prov: "StartProvenance" = resolve_start_provenance(file_path)
    has_sweep = prov.detection_settings is not None
    is_manual = prov.source == "manual"

    if has_sweep:
        fig, axes = plt.subplots(3, 1, figsize=_figsize_or_default(figsize, (10, 11)))
        ax_fid, ax_top, ax_bot = axes
    else:
        fig, ax_fid = plt.subplots(
            1, 1, figsize=_figsize_or_default(figsize, (10, 3.5))
        )

    span_us = (
        prov.detection_settings.sweep_max_us
        if prov.detection_settings is not None
        else min(fid.duration_us, 7.5)
    )
    span_us = min(span_us, fid.duration_us)
    time_us = fid.time_array_us()
    mask = time_us <= span_us
    ax_fid.plot(time_us[mask], fid.data[mask], color=AGGIE_BLUE, lw=0.6)
    ax_fid.set_xlabel("Time (µs)")
    ax_fid.set_ylabel("Voltage (V)")
    _mark_chirp_and_start(
        ax_fid,
        chirp_start_us=prov.chirp_start_us,
        chirp_end_us=prov.chirp_end_us,
        recommended_start_us=prov.recommended_start_us,
        start_us=prov.start_us,
        is_manual=is_manual,
    )
    if prov.source == "none":
        ax_fid.text(
            0.02,
            0.92,
            "no chirp/start selection on record",
            transform=ax_fid.transAxes,
            color="red",
            fontsize=9,
        )
    elif prov.source == "no_chirp_found":
        ax_fid.text(
            0.02,
            0.92,
            "no chirp collapse detected",
            transform=ax_fid.transAxes,
            color="red",
            fontsize=9,
        )
    resolved_title = resolve_title(title, "Stage 0 -- start-time selection")
    if resolved_title:
        ax_fid.set_title(resolved_title)
    apply_bare_style(ax_fid)

    if has_sweep:
        assert prov.detection_settings is not None
        result = detect_start_time(
            fid, band=prov.detection_band_mhz, settings=prov.detection_settings
        )
        band_label = (
            f"{prov.detection_band_mhz[0]:.0f}-{prov.detection_band_mhz[1]:.0f} MHz"
            if prov.detection_band_mhz is not None
            else "full spectrum"
        )
        chirp_end_for_sweep = (
            prov.chirp_end_us if prov.chirp_end_us is not None else result.chirp_end_us
        )
        start_for_sweep = (
            prov.start_us if prov.start_us is not None else result.start_us
        )
        chirp_detected = (
            prov.chirp_detected
            if prov.chirp_detected is not None
            else result.chirp_detected
        )
        start_color = "0.15" if is_manual else QUAD
        _draw_sweep_top(
            ax_top,
            result.starts_us,
            result.sum_magnitude,
            chirp_end_us=chirp_end_for_sweep,
            start_us=start_for_sweep,
            chirp_detected=chirp_detected,
            band_label=band_label,
            title=None,
            start_label="start (manual)" if is_manual else "start",
            start_color=start_color,
        )
        _draw_sweep_bottom(
            ax_bot,
            result.starts_us,
            result.sum_magnitude,
            chirp_end_us=chirp_end_for_sweep,
            start_us=start_for_sweep,
            floor=result.floor,
            start_color=start_color,
        )
        ax_top.set_xlim(ax_fid.get_xlim())
        apply_bare_style(ax_top)
        apply_bare_style(ax_bot)

    fig.tight_layout()
    return fig


__all__ = [
    "plot_start_detection",
    "plot_start_detection_from_file",
    "plot_stage0_overview_from_file",
]
