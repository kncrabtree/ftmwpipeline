"""Visualizations for the Stage 2b tau calibration.

Two figures are exposed, mirroring the Phase-2 research figures
(``figures/08_2638_stft_heatmap.png`` and
``figures/09_2638_distribution_analysis.png``):

* :func:`plot_tau_heatmap`: 2D ``log10 |S_n(f)|`` heatmap across the
  ``n_seg`` STFT frames and the trim frequency range. The STFT is
  recomputed on demand from the persisted FID + Stage 1 settings + the
  ``n_seg`` recorded in the persisted calibration; the heatmap data is too
  large to persist (~50 MB on 2638 even compressed).
* :func:`plot_tau_distribution`: tau histogram with the SNR-weighted
  majority overlay, plus tau-vs-SNR and tau-vs-molecular-frequency
  scatters and the 1- vs 2-component GMM overlay.

The two ``*_from_file`` wrappers handle the .ftmw -> figure plumbing so
the CLI / Pipeline / functional-API surfaces all share the same
orchestration.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ..fitting.tau_calibration import TauCalibrationResult, sliding_stft

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
    cmap: str = "viridis",
) -> "matplotlib.figure.Figure":
    """2D STFT magnitude heatmap restricted to the calibration's trim range.

    Reruns :func:`sliding_stft` on the active-region FID slice (parameters
    pulled from the persisted ``TauCalibrationResult``: ``start_us``,
    ``end_us``, ``sample_dt_us``, ``n_seg``) and plots ``log10 |S|`` over
    the trim molecular-frequency range. Same convention as
    ``research/figures/08_2638_stft_heatmap.png``.
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

    fig, ax = plt.subplots(figsize=_figsize_or_default(figsize, (14, 4.5)))
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
        cmap=cmap,
    )
    ax.set_xlabel("molecular frequency (MHz)")
    ax.set_ylabel("STFT frame centre a_c (us)")
    if title is None:
        title = (
            "STFT magnitude heatmap (log10 |S(a, f)|) — "
            f"trim {result.trim_lo_mhz:.0f}-{result.trim_hi_mhz:.0f} MHz, "
            f"n_seg={result.n_seg}, tau_maj={result.tau_maj_us:.2f} us"
        )
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, label="log10 |S_n|")
    fig.tight_layout()
    return fig


def plot_tau_heatmap_from_file(
    file_path: str,
    **kwargs: Any,
) -> "matplotlib.figure.Figure":
    """Convenience: load the FID + persisted calibration and call :func:`plot_tau_heatmap`."""
    from .._internal.stage0_impl import load_fid_from_pipeline_impl
    from .._internal.stage2b_impl import load_tau_calibration_impl

    cal = load_tau_calibration_impl(file_path)["tau_calibration"]
    fid = load_fid_from_pipeline_impl(file_path)
    return plot_tau_heatmap(cal, np.asarray(fid.data, dtype=float), **kwargs)


# ---------------------------------------------------------------------------
# Distribution analysis
# ---------------------------------------------------------------------------
def plot_tau_distribution(
    result: TauCalibrationResult,
    *,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    n_bins: int = 80,
) -> "matplotlib.figure.Figure":
    """Four-panel distribution analysis: histogram, tau-vs-SNR, tau-vs-freq, GMM overlay.

    Matches ``research/figures/09_2638_distribution_analysis.png``.
    """
    from .report_style import apply_bare_style

    fig, axes = plt.subplots(2, 2, figsize=_figsize_or_default(figsize, (14, 8)))

    taus = np.asarray(result.contributor_taus_us)
    snrs = np.asarray(result.contributor_snrs)
    freqs = np.asarray(result.contributor_freqs_mhz)
    tau_maj = result.tau_maj_us
    sigma_tau = result.sigma_tau_us
    bm = result.bimodality

    # 1. Histogram
    ax = axes[0, 0]
    ax.hist(
        taus, bins=n_bins, color="C0", alpha=0.7, label=f"contributors (n={taus.size})"
    )
    ax.axvline(
        tau_maj,
        color="C3",
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
        ax.scatter(snrs, taus, s=2, alpha=0.4)
        ax.set_xscale("log")
    ax.axhline(tau_maj, color="C3", ls="--", lw=1)
    ax.set_xlabel("contributor on-line SNR (per-frame)")
    ax.set_ylabel("tau_k (us)")
    ax.set_title("tau vs SNR (Pearson r = " f"{result.pearson_r_log_snr_vs_tau:.3f})")
    apply_bare_style(ax)

    # 3. tau vs molecular freq
    ax = axes[1, 0]
    if freqs.size > 0:
        ax.scatter(freqs, taus, s=2, alpha=0.4)
    ax.axhline(tau_maj, color="C3", ls="--", lw=1)
    ax.set_xlabel("molecular frequency (MHz)")
    ax.set_ylabel("tau_k (us)")
    ax.set_title(
        "tau vs frequency (Pearson r = " f"{result.pearson_r_freq_vs_tau:.3f})"
    )
    apply_bare_style(ax)
    # Annotate the frequency thirds when present.
    for third in result.frequency_thirds:
        ax.axhline(third.median_tau_us, color="C2", ls=":", lw=0.6, alpha=0.7)
        ax.text(
            0.5 * (third.freq_lo_mhz + third.freq_hi_mhz),
            third.median_tau_us,
            f"{third.label}: {third.median_tau_us:.2f}",
            fontsize=7,
            ha="center",
            va="bottom",
            color="C2",
        )

    # 4. GMM overlay
    ax = axes[1, 1]
    if taus.size > 0:
        ax.hist(taus, bins=n_bins, color="C0", alpha=0.5, density=True)
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
        ax.plot(xs, ya, "C3", lw=1, label=f"GMM mu_a={bm.mu_a:.2f}, pi_a={bm.pi_a:.2f}")
        ax.plot(xs, yb, "C2", lw=1, label=f"GMM mu_b={bm.mu_b:.2f}")
        ax.plot(xs, ya + yb, "k", lw=1, alpha=0.6)
    ax.set_xlabel("tau (us)")
    ax.set_ylabel("density")
    ax.set_title(
        f"GMM 1 vs 2 component (delta_aic = {bm.delta_aic:.1f}, "
        f"bimodal={bm.two_component_preferred})"
    )
    ax.legend(fontsize=8)
    apply_bare_style(ax)

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_tau_distribution_from_file(
    file_path: str,
    **kwargs: Any,
) -> "matplotlib.figure.Figure":
    """Convenience: load the persisted calibration and call :func:`plot_tau_distribution`."""
    from .._internal.stage2b_impl import load_tau_calibration_impl

    cal = load_tau_calibration_impl(file_path)["tau_calibration"]
    return plot_tau_distribution(cal, **kwargs)
