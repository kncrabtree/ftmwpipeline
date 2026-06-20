"""Visualizations for the Stage 2b tau calibration.

Two figures are exposed:

* :func:`plot_tau_heatmap`: 2D ``log10 |S_n(f)|`` heatmap across the
  ``n_seg`` STFT frames and the trim frequency range. The STFT is
  recomputed on demand from the persisted FID + Stage 1 settings + the
  ``n_seg`` recorded in the persisted calibration; the heatmap data is too
  large to persist (~50 MB on 2638 even compressed).
* :func:`plot_tau_distribution`: tau histogram with the SNR-weighted
  majority overlay, plus tau-vs-SNR and tau-vs-molecular-frequency
  scatters and the 1- vs 2-component GMM overlay.

The two ``*_from_file`` wrappers handle the .ftmw -> figure plumbing so the
CLI / Pipeline / functional-API surfaces all share the same orchestration.
``shape`` selects the pure-exp (``"lorentzian"``) or Gaussian (``"gaussian"``)
calibration group.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ..fitting.tau_calibration import TauCalibrationResult, sliding_stft
from .report_style import (
    AGGIE_BLUE,
    DOUBLE_DECKER,
    GUNROCK,
    QUAD,
    aggie_blue_cmap,
    apply_bare_style,
    resolve_title,
)

__all__ = [
    "plot_tau_heatmap",
    "plot_tau_distribution",
    "plot_tau_heatmap_from_file",
    "plot_tau_distribution_from_file",
]


def _figsize_or_default(
    figsize: Optional[Tuple[float, float]], default: Tuple[float, float]
) -> Tuple[float, float]:
    return figsize if figsize is not None else default


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------
def plot_tau_heatmap(
    result: TauCalibrationResult,
    fid: np.ndarray,
    *,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """2D STFT magnitude heatmap restricted to the calibration's trim range.

    Reruns :func:`sliding_stft` on the active-region FID slice (parameters
    pulled from the persisted ``TauCalibrationResult``: ``start_us``,
    ``end_us``, ``sample_dt_us``, ``n_seg``) and plots ``log10 |S|`` over the
    trim molecular-frequency range. A real molecular line decays down the
    frame axis; a clock spur holds constant magnitude.
    """
    sample_dt_us = result.sample_dt_us
    start_idx = int(round(result.start_us / sample_dt_us))
    end_idx = int(round(result.end_us / sample_dt_us))
    fid_arr = np.asarray(fid, dtype=float)
    start_idx = max(start_idx, 0)
    end_idx = min(end_idx, fid_arr.size)
    active = fid_arr[start_idx:end_idx]
    new_size = (active.size // result.n_seg) * result.n_seg
    active = active[:new_size]
    mag, a_centers_us, freq_bb_mhz = sliding_stft(active, sample_dt_us, result.n_seg)

    # Map baseband to molecular and restrict to trim.
    sign = -1.0 if result.sideband == "lower" else +1.0
    freq_mol_mhz = result.probe_freq_mhz + sign * freq_bb_mhz
    in_trim = (freq_mol_mhz >= result.trim_lo_mhz) & (
        freq_mol_mhz <= result.trim_hi_mhz
    )
    trim_idx = np.where(in_trim)[0]
    freqs_mol = freq_mol_mhz[trim_idx]
    mag_trim = mag[:, trim_idx]
    order = np.argsort(freqs_mol)
    freqs_sorted = freqs_mol[order]
    mag_sorted = mag_trim[:, order]

    fig, ax = plt.subplots(
        figsize=_figsize_or_default(figsize, (14, 4.5)), layout="constrained"
    )
    im = ax.imshow(
        np.log10(np.clip(mag_sorted, 1e-30, None)),
        aspect="auto",
        origin="lower",
        extent=(
            float(freqs_sorted[0]),
            float(freqs_sorted[-1]),
            float(a_centers_us[0]),
            float(a_centers_us[-1]),
        ),
        cmap=aggie_blue_cmap(),
    )
    ax.set_xlabel("molecular frequency (MHz)")
    ax.set_ylabel("STFT frame center a_c (us)")
    default_title = (
        "STFT magnitude heatmap (log10 |S(a, f)|) — "
        f"trim {result.trim_lo_mhz:.0f}-{result.trim_hi_mhz:.0f} MHz, "
        f"n_seg={result.n_seg}, tau_maj={result.tau_maj_us:.2f} us"
    )
    resolved_title = resolve_title(title, default_title)
    if resolved_title:
        ax.set_title(resolved_title)
    fig.colorbar(im, ax=ax, label="log10 |S_n|")
    return fig


def plot_tau_heatmap_from_file(
    file_path: str,
    *,
    shape: str = "lorentzian",
    **kwargs: Any,
) -> "matplotlib.figure.Figure":
    """Load the FID + persisted ``shape`` calibration and plot the heatmap."""
    from .._internal.stage0_impl import load_fid_from_pipeline_impl
    from .._internal.stage2b_impl import load_tau_calibration_impl

    cal = load_tau_calibration_impl(file_path, shape=shape)["tau_calibration"]
    fid = load_fid_from_pipeline_impl(file_path)
    return plot_tau_heatmap(cal, np.asarray(fid.data, dtype=float), **kwargs)


# ---------------------------------------------------------------------------
# Distribution analysis
# ---------------------------------------------------------------------------
def _scatter_with_tau_cap(
    ax: Any, x: np.ndarray, y: np.ndarray, cap: Optional[float], *, color: str
) -> None:
    """Scatter ``(x, y)`` with the decay-time axis ``y`` capped at ``cap``.

    Decay times longer than the active-region length cannot be measured
    reliably, so the panel's y-axis stops at ``cap``; bins above it are drawn
    as open upward triangles along the ceiling to flag the off-screen data
    without letting it stretch the axis. ``cap=None`` falls back to a plain
    scatter.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if cap is None or not np.isfinite(cap) or cap <= 0:
        ax.scatter(x, y, s=2, alpha=0.4, color=color)
        return
    over = y > cap
    if (~over).any():
        ax.scatter(x[~over], y[~over], s=2, alpha=0.4, color=color)
    n_over = int(over.sum())
    if n_over:
        ax.scatter(
            x[over],
            np.full(n_over, cap),
            s=16,
            marker="^",
            facecolors="none",
            edgecolors=color,
            linewidths=0.6,
            alpha=0.7,
        )
        ax.text(
            0.02,
            0.97,
            f"△ {n_over} bins > {cap:.0f} us",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
            color=color,
        )
    ax.set_ylim(0.0, cap * 1.08)


def plot_tau_distribution(
    result: TauCalibrationResult,
    *,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    n_bins: int = 80,
    tau_cap_factor: float = 1.5,
) -> "matplotlib.figure.Figure":
    """Four-panel distribution analysis: histogram, tau-vs-SNR, tau-vs-freq, GMM.

    The two decay-time scatters cap their y-axis at ``tau_cap_factor`` times the
    active-region length (decay times beyond that are not reliably measurable);
    contributors above the cap are flagged as open upward triangles along the
    top edge rather than allowed to stretch the axis.
    """
    fig, axes = plt.subplots(2, 2, figsize=_figsize_or_default(figsize, (14, 8)))

    taus = np.asarray(result.contributor_taus_us)
    snrs = np.asarray(result.contributor_snrs)
    freqs = np.asarray(result.contributor_freqs_mhz)
    tau_maj = result.tau_maj_us
    sigma_tau = result.sigma_tau_us
    bm = result.bimodality

    # Decay times longer than ~the active-region length are not reliably
    # measurable; cap the decay-time scatters there and flag anything above.
    t_active = float(result.end_us - result.start_us)
    tau_cap = float(tau_cap_factor) * t_active if t_active > 0 else None

    # 1. Histogram
    ax = axes[0, 0]
    ax.hist(
        taus,
        bins=n_bins,
        color=AGGIE_BLUE,
        alpha=0.7,
        label=f"contributors (n={taus.size})",
    )
    ax.axvline(
        tau_maj,
        color=DOUBLE_DECKER,
        ls="--",
        lw=2,
        label=f"tau_maj = {tau_maj:.2f} us (sigma_tau = {sigma_tau:.2f})",
    )
    ax.set_xlabel("recovered tau (us)")
    ax.set_ylabel("count")
    ax.set_title("Per-bin tau histogram")
    ax.legend()
    apply_bare_style(ax)

    # 2. tau vs SNR
    ax = axes[0, 1]
    if snrs.size > 0:
        _scatter_with_tau_cap(ax, snrs, taus, tau_cap, color=AGGIE_BLUE)
        ax.set_xscale("log")
    ax.axhline(tau_maj, color=DOUBLE_DECKER, ls="--", lw=1)
    ax.set_xlabel("contributor on-line SNR (per-frame)")
    ax.set_ylabel("tau_k (us)")
    ax.set_title("tau vs SNR (Pearson r = " f"{result.pearson_r_log_snr_vs_tau:.3f})")
    apply_bare_style(ax)

    # 3. tau vs molecular freq
    ax = axes[1, 0]
    if freqs.size > 0:
        _scatter_with_tau_cap(ax, freqs, taus, tau_cap, color=AGGIE_BLUE)
    ax.axhline(tau_maj, color=DOUBLE_DECKER, ls="--", lw=1)
    ax.set_xlabel("molecular frequency (MHz)")
    ax.set_ylabel("tau_k (us)")
    ax.set_title(
        "tau vs frequency (Pearson r = " f"{result.pearson_r_freq_vs_tau:.3f})"
    )
    apply_bare_style(ax)
    # Annotate the frequency thirds when present.
    for third in result.frequency_thirds:
        ax.axhline(third.median_tau_us, color=QUAD, ls=":", lw=0.6, alpha=0.7)
        ax.text(
            0.5 * (third.freq_lo_mhz + third.freq_hi_mhz),
            third.median_tau_us,
            f"{third.label}: {third.median_tau_us:.2f}",
            fontsize=7,
            ha="center",
            va="bottom",
            color=QUAD,
        )

    # 4. GMM overlay
    ax = axes[1, 1]
    if taus.size > 0:
        ax.hist(taus, bins=n_bins, color=AGGIE_BLUE, alpha=0.5, density=True)
    if not np.isnan(bm.mu_a) and tau_maj > 0:
        xs = np.linspace(
            0.0, max(tau_maj * 3.0, taus.max() * 1.1 if taus.size else tau_maj), 400
        )
        ya = (
            bm.pi_a
            / np.sqrt(2 * np.pi * bm.sigma_a**2)
            * np.exp(-0.5 * ((xs - bm.mu_a) / bm.sigma_a) ** 2)
        )
        yb = (
            (1.0 - bm.pi_a)
            / np.sqrt(2 * np.pi * bm.sigma_b**2)
            * np.exp(-0.5 * ((xs - bm.mu_b) / bm.sigma_b) ** 2)
        )
        ax.plot(
            xs,
            ya,
            color=DOUBLE_DECKER,
            lw=1,
            label=f"GMM mu_a={bm.mu_a:.2f}, pi_a={bm.pi_a:.2f}",
        )
        ax.plot(xs, yb, color=QUAD, lw=1, label=f"GMM mu_b={bm.mu_b:.2f}")
        ax.plot(xs, ya + yb, color=GUNROCK, lw=1, alpha=0.6)
    ax.set_xlabel("tau (us)")
    ax.set_ylabel("density")
    ax.set_title(
        f"GMM 1 vs 2 component (delta_aic = {bm.delta_aic:.1f}, "
        f"bimodal={bm.two_component_preferred})"
    )
    ax.legend(fontsize=8)
    apply_bare_style(ax)

    resolved_title = resolve_title(title, "")
    if resolved_title:
        fig.suptitle(resolved_title)
    fig.tight_layout()
    return fig


def plot_tau_distribution_from_file(
    file_path: str,
    *,
    shape: str = "lorentzian",
    **kwargs: Any,
) -> "matplotlib.figure.Figure":
    """Load the persisted ``shape`` calibration and plot the distribution."""
    from .._internal.stage2b_impl import load_tau_calibration_impl

    cal = load_tau_calibration_impl(file_path, shape=shape)["tau_calibration"]
    return plot_tau_distribution(cal, **kwargs)
