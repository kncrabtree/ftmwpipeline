"""Visualizations for the Stage 2b tau calibration.

Three figures are exposed:

* :func:`plot_tau_heatmap`: 2D ``log10 |S_n(f)|`` heatmap across the
  ``n_seg`` STFT windows and a frequency range, with optional frequency-window
  and color-range clipping so per-line decays read clearly. The STFT is
  recomputed on demand from the persisted FID + the calibration's window
  parameters; the heatmap data is too large to persist (~50 MB on 2638 even
  compressed).
* :func:`plot_stft_decay_examples`: the per-bin magnitude-vs-window series for a
  strong line (with the exponential and Gaussian decay fits), a clock spur, and
  a noise bin — the time series the per-bin classifier sorts.
* :func:`plot_tau_distribution`: tau histogram with the SNR-weighted majority
  overlay, plus tau-vs-SNR and tau-vs-molecular-frequency scatters (with the
  per-band levels) and the 1- vs 2-component GMM overlay.

The ``*_from_file`` wrappers handle the .ftmw -> figure plumbing so the CLI /
Pipeline / functional-API surfaces all share the same orchestration. ``shape``
selects the pure-exp (``"lorentzian"``) or Gaussian (``"gaussian"``) calibration
group.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ..core.data_structures import Sideband
from ..fitting.tau_calibration import TauCalibrationResult, sliding_stft
from .report_style import (
    AGGIE_BLUE,
    DOUBLE_DECKER,
    GUNROCK,
    POPPY,
    QUAD,
    aggie_blue_cmap,
    apply_bare_style,
    resolve_title,
)

__all__ = [
    "plot_tau_heatmap",
    "plot_tau_distribution",
    "plot_stft_decay_examples",
    "plot_tau_heatmap_from_file",
    "plot_tau_distribution_from_file",
    "plot_stft_decay_examples_from_file",
]


def _figsize_or_default(
    figsize: Optional[Tuple[float, float]], default: Tuple[float, float]
) -> Tuple[float, float]:
    return figsize if figsize is not None else default


def _active_stft(
    result: TauCalibrationResult, fid: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recompute the calibration's sliding STFT and molecular-frequency axis.

    Reruns :func:`sliding_stft` on the active-region FID slice using the
    parameters pulled from the persisted :class:`TauCalibrationResult`
    (``start_us``, ``end_us``, ``sample_dt_us``, ``n_seg``, ``sideband``,
    ``probe_freq_mhz``). Returns ``(mag, a_centers_us, freq_mol_mhz)`` with
    ``mag`` of shape ``(n_seg, n_bins)``.
    """
    sample_dt_us = result.sample_dt_us
    fid_arr = np.asarray(fid, dtype=float)
    start_idx = max(int(round(result.start_us / sample_dt_us)), 0)
    end_idx = min(int(round(result.end_us / sample_dt_us)), fid_arr.size)
    active = fid_arr[start_idx:end_idx]
    new_size = (active.size // result.n_seg) * result.n_seg
    active = active[:new_size]
    mag, a_centers_us, freq_bb_mhz = sliding_stft(active, sample_dt_us, result.n_seg)
    sign = Sideband.coerce(result.sideband).sign
    freq_mol_mhz = result.probe_freq_mhz + sign * freq_bb_mhz
    return mag, a_centers_us, freq_mol_mhz


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------
def plot_tau_heatmap(
    result: TauCalibrationResult,
    fid: np.ndarray,
    *,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    freq_window: Optional[Tuple[float, float]] = None,
    clip_percentiles: Optional[Tuple[float, float]] = None,
) -> "matplotlib.figure.Figure":
    """2D STFT magnitude heatmap over molecular frequency and window position.

    Plots ``log10 |S|`` with molecular frequency on the x-axis and the sliding
    window center on the y-axis. A real molecular line decays down the frame
    axis (bright at the top, fading downward); a clock spur holds a constant
    magnitude. ``freq_window`` restricts the x-axis to a ``(lo, hi)`` MHz slice
    so the decays are resolvable; ``clip_percentiles`` sets the color limits to
    the given ``(low, high)`` percentiles of the displayed magnitudes, which
    pulls the faint-but-real decays out of the noise floor instead of letting a
    handful of strong bins set the whole color range.
    """
    mag, a_centers_us, freq_mol_mhz = _active_stft(result, fid)

    lo = result.trim_lo_mhz if freq_window is None else min(freq_window)
    hi = result.trim_hi_mhz if freq_window is None else max(freq_window)
    in_window = (freq_mol_mhz >= lo) & (freq_mol_mhz <= hi)
    sel = np.where(in_window)[0]
    freqs_sel = freq_mol_mhz[sel]
    mag_sel = mag[:, sel]
    order = np.argsort(freqs_sel)
    freqs_sorted = freqs_sel[order]
    mag_sorted = mag_sel[:, order]

    log_mag = np.log10(np.clip(mag_sorted, 1e-30, None))
    vmin = vmax = None
    if clip_percentiles is not None and log_mag.size:
        vmin = float(np.percentile(log_mag, min(clip_percentiles)))
        vmax = float(np.percentile(log_mag, max(clip_percentiles)))

    fig, ax = plt.subplots(
        figsize=_figsize_or_default(figsize, (14, 4.5)), layout="constrained"
    )
    im = ax.imshow(
        log_mag,
        aspect="auto",
        origin="upper",
        extent=(
            float(freqs_sorted[0]),
            float(freqs_sorted[-1]),
            float(a_centers_us[-1]),
            float(a_centers_us[0]),
        ),
        cmap=aggie_blue_cmap(),
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("molecular frequency (MHz)")
    ax.set_ylabel("STFT window center $a_c$ (µs)")
    span = (
        f"{lo:.0f}-{hi:.0f} MHz"
        if freq_window is not None
        else f"trim {result.trim_lo_mhz:.0f}-{result.trim_hi_mhz:.0f} MHz"
    )
    default_title = (
        "STFT magnitude heatmap (log10 |S(a, f)|) — "
        f"{span}, n_seg={result.n_seg}, tau_maj={result.tau_maj_us:.2f} us"
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
# Per-bin STFT decay examples
# ---------------------------------------------------------------------------
def _loglin_decay_fit(
    y: np.ndarray, regressor: np.ndarray
) -> Optional[Tuple[float, float]]:
    """Weighted log-linear decay fit; returns ``(C, decay_coeff)`` or ``None``.

    Fits ``log|S| = log C + b * regressor`` (``regressor = a`` for the
    exponential, ``a**2`` for the Gaussian), weighting by ``|S|**2`` as the
    classifier does. Returns ``None`` when the slope is non-negative (no decay).
    """
    safe = np.clip(y, 1e-30, None)
    w = y * y
    coeffs = np.polyfit(regressor, np.log(safe), 1, w=np.sqrt(w))
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    if slope >= 0:
        return None
    return float(np.exp(intercept)), slope


def plot_stft_decay_examples(
    result: TauCalibrationResult,
    fid: np.ndarray,
    *,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """The per-bin magnitude-vs-window-position series that drive classification.

    Three representative bins, recomputed from the persisted calibration's STFT:
    a strong molecular line (with the exponential and Gaussian decay fits
    overlaid, so the shape difference is visible), a clock spur (constant
    magnitude — no decay), and a noise bin (no clean decay). These are the
    time series the per-bin classifier fits to sort every frequency bin into
    contributor, spur, or discard.
    """
    mag, a_us, freq_mol_mhz = _active_stft(result, fid)
    mean_mag = mag.mean(axis=0)
    max_mag = mag.max(axis=0)
    a_fine = np.linspace(float(a_us[0]), float(a_us[-1]), 100)

    fig, axes = plt.subplots(
        1, 3, figsize=_figsize_or_default(figsize, (14, 4.2)), layout="constrained"
    )

    # 1. Strong molecular line: highest-SNR contributor, with both shape fits.
    ax = axes[0]
    if result.contributor_bin_indices.size:
        i = int(result.contributor_bin_indices[int(np.argmax(result.contributor_snrs))])
        y = mag[:, i]
        snr = float(result.contributor_snrs[int(np.argmax(result.contributor_snrs))])
        ax.scatter(a_us, y, s=24, color=AGGIE_BLUE, zorder=4, label="|S_n|")
        exp_fit = _loglin_decay_fit(y, a_us)
        if exp_fit is not None:
            c, slope = exp_fit
            tau = -1.0 / slope
            ax.plot(
                a_fine,
                c * np.exp(slope * a_fine),
                color=DOUBLE_DECKER,
                lw=1.6,
                label=f"exponential ($\\tau$ = {tau:.1f} µs)",
            )
        gauss_fit = _loglin_decay_fit(y, a_us**2)
        if gauss_fit is not None:
            c, slope = gauss_fit
            tau_g = 1.0 / np.sqrt(-slope)
            ax.plot(
                a_fine,
                c * np.exp(slope * a_fine**2),
                color=QUAD,
                lw=1.6,
                ls="--",
                label=f"Gaussian ($\\tau_G$ = {tau_g:.1f} µs)",
            )
        ax.set_title(f"molecular line @ {freq_mol_mhz[i]:.1f} MHz (SNR {snr:.0f})")
        ax.legend(fontsize=7)
    ax.set_xlabel("STFT window center $a_c$ (µs)")
    ax.set_ylabel("$|S_n|$")
    ax.set_ylim(bottom=0.0)
    apply_bare_style(ax)

    # 2. Clock spur: brightest *saturated* (flat, CW) cluster representative, so
    # the panel shows a genuine constant tone rather than an erratic beat bin.
    ax = axes[1]
    if result.spur_clusters:
        saturated = [c for c in result.spur_clusters if c.saturated]
        pool = saturated or list(result.spur_clusters)
        peaks = np.array([c.peak_bin_index for c in pool], dtype=int)
        s = int(peaks[int(np.argmax(mean_mag[peaks]))])
        ax.scatter(a_us, mag[:, s], s=24, color=GUNROCK, zorder=4)
        ax.axhline(float(mean_mag[s]), color=GUNROCK, ls=":", lw=1.0)
        ax.set_title(f"clock spur @ {freq_mol_mhz[s]:.1f} MHz")
        ax.text(
            0.5,
            0.08,
            "constant — no decay",
            transform=ax.transAxes,
            ha="center",
            fontsize=8,
            color=GUNROCK,
        )
    ax.set_xlabel("STFT window center $a_c$ (µs)")
    ax.set_ylabel("$|S_n|$")
    ax.set_ylim(bottom=0.0)
    apply_bare_style(ax)

    # 3. Noise bin: a representative low-magnitude in-band bin.
    ax = axes[2]
    in_trim = (freq_mol_mhz >= result.trim_lo_mhz) & (
        freq_mol_mhz <= result.trim_hi_mhz
    )
    trim_idx = np.where(in_trim)[0]
    if trim_idx.size:
        target = float(np.percentile(max_mag[trim_idx], 40.0))
        n = int(trim_idx[int(np.argmin(np.abs(max_mag[trim_idx] - target)))])
        ax.scatter(a_us, mag[:, n], s=24, color=AGGIE_BLUE, alpha=0.6, zorder=4)
        ax.set_title(f"noise bin @ {freq_mol_mhz[n]:.1f} MHz")
        ax.text(
            0.5,
            0.08,
            "no clean decay",
            transform=ax.transAxes,
            ha="center",
            fontsize=8,
            color=AGGIE_BLUE,
        )
    ax.set_xlabel("STFT window center $a_c$ (µs)")
    ax.set_ylabel("$|S_n|$")
    ax.set_ylim(bottom=0.0)
    apply_bare_style(ax)

    resolved_title = resolve_title(title, "")
    if resolved_title:
        fig.suptitle(resolved_title)
    return fig


def plot_stft_decay_examples_from_file(
    file_path: str,
    *,
    shape: str = "lorentzian",
    **kwargs: Any,
) -> "matplotlib.figure.Figure":
    """Load the FID + persisted ``shape`` calibration and plot the decay examples."""
    from .._internal.stage0_impl import load_fid_from_pipeline_impl
    from .._internal.stage2b_impl import load_tau_calibration_impl

    cal = load_tau_calibration_impl(file_path, shape=shape)["tau_calibration"]
    fid = load_fid_from_pipeline_impl(file_path)
    return plot_stft_decay_examples(cal, np.asarray(fid.data, dtype=float), **kwargs)


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
            f"△ {n_over} bins > {cap:.0f} µs",
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
        label=f"$\\tau_\\mathrm{{maj}}$ = {tau_maj:.2f} µs "
        f"($\\sigma_\\tau$ = {sigma_tau:.2f})",
    )
    ax.set_xlabel("recovered $\\tau$ (µs)")
    ax.set_ylabel("count")
    ax.set_title("Per-bin $\\tau$ histogram")
    ax.legend()
    apply_bare_style(ax)

    # 2. tau vs SNR
    ax = axes[0, 1]
    if snrs.size > 0:
        _scatter_with_tau_cap(ax, snrs, taus, tau_cap, color=AGGIE_BLUE)
        ax.set_xscale("log")
    ax.axhline(tau_maj, color=DOUBLE_DECKER, ls="--", lw=2)
    ax.set_xlabel("contributor on-line SNR (per-frame)")
    ax.set_ylabel("$\\tau_k$ (µs)")
    ax.set_title(
        "$\\tau$ vs SNR (Pearson $r$ = " f"{result.pearson_r_log_snr_vs_tau:.3f})"
    )
    apply_bare_style(ax)

    # 3. tau vs molecular freq
    ax = axes[1, 0]
    if freqs.size > 0:
        _scatter_with_tau_cap(ax, freqs, taus, tau_cap, color=AGGIE_BLUE)
    ax.axhline(tau_maj, color=DOUBLE_DECKER, ls="--", lw=2)
    ax.set_xlabel("molecular frequency (MHz)")
    ax.set_ylabel("$\\tau_k$ (µs)")
    ax.set_title(
        "$\\tau$ vs frequency (Pearson $r$ = " f"{result.pearson_r_freq_vs_tau:.3f})"
    )
    apply_bare_style(ax)
    # Per-band majority levels and boundaries, labeled legibly at the top edge.
    # Prefer the persisted per-band majorities (what the fit routes on); fall
    # back to the per-third medians when band majorities were not computed.
    band_specs: list[Tuple[str, float, float, float, Optional[float]]]
    if result.band_majorities:
        band_specs = [
            (b.label, b.freq_lo_mhz, b.freq_hi_mhz, b.tau_maj_us, b.sigma_tau_us)
            for b in result.band_majorities
        ]
    else:
        band_specs = [
            (t.label, t.freq_lo_mhz, t.freq_hi_mhz, t.median_tau_us, None)
            for t in result.frequency_thirds
        ]
    if band_specs:
        ymax = ax.get_ylim()[1]
        for i, (label, flo, fhi, level, sig) in enumerate(band_specs):
            if i > 0:  # interior boundary between bands
                ax.axvline(flo, color="0.6", ls="-", lw=0.7, alpha=0.8, zorder=1)
            # Shade the per-band majority's robust spread sigma_tau so the
            # uncertainty on each band's tau is visible, not just the level.
            if sig is not None and np.isfinite(sig) and sig > 0:
                ax.fill_between(
                    [flo, fhi],
                    level - sig,
                    level + sig,
                    color=POPPY,
                    alpha=0.18,
                    lw=0,
                    zorder=4,
                )
            ax.hlines(level, flo, fhi, color=POPPY, lw=2.5, zorder=5)
            band_txt = (
                f"{label}\n{level:.2f} µs"
                if sig is None
                else f"{label}\n{level:.2f}±{sig:.2f} µs"
            )
            ax.text(
                0.5 * (flo + fhi),
                ymax * 0.97,
                band_txt,
                ha="center",
                va="top",
                fontsize=9,
                fontweight="bold",
                color=POPPY,
                zorder=6,
            )

    # 4. GMM overlay
    ax = axes[1, 1]
    if taus.size > 0:
        ax.hist(taus, bins=n_bins, color=AGGIE_BLUE, alpha=0.7, density=True)
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
            lw=2,
            label=f"GMM $\\mu_a$={bm.mu_a:.2f}, $\\pi_a$={bm.pi_a:.2f}",
        )
        ax.plot(xs, yb, color=QUAD, lw=2, label=f"GMM $\\mu_b$={bm.mu_b:.2f}")
        ax.plot(xs, ya + yb, color="black", lw=2, alpha=0.85)
    ax.set_xlabel("$\\tau$ (µs)")
    ax.set_ylabel("density")
    ax.set_title(
        f"GMM 1 vs 2 component ($\\Delta$AIC = {bm.delta_aic:.1f}, "
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
