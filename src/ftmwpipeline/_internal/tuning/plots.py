"""Plot adapters for tuning knobs.

A plot adapter has signature ``(spec, rows, ctx) -> matplotlib Figure | None``
and is referenced from a
:class:`~ftmwpipeline._internal.tuning.registry.KnobSpec`. The engine calls it
after the sweep; ``rows`` carries each grid value's metrics and its full stage
``result``, and ``ctx`` (a :class:`~ftmwpipeline._internal.tuning.engine.PlotContext`)
carries the working ``.ftmw`` path so an adapter can load source data (e.g. the
FID) the per-value result does not hold.

Adapters import matplotlib lazily and never choose a backend — the engine/CLI
own display (file vs interactive).
"""

from __future__ import annotations

from typing import Any, List

# Target per-panel aspect for stacked "ladder" figures (width : height).
_LADDER_PANEL_ASPECT = 5.5
_LADDER_WIDTH_IN = 11.0
# Shared y-limit for the spectra ladder: this multiple of the largest per-panel
# median |FT| (p50). High enough that the chirp/ringdown fuzz fills the axis and
# its collapse across panels is visible; real lines clip off the top.
_LADDER_YMAX_P50_FACTOR = 10.0
# σ(f)-over-spectrum overlay: zoom the y-axis to this multiple of the largest
# per-bin σ so the noise band sits against the spectral floor and small,
# noise-scale features are legible; tall real lines clip off the top.
_NOISE_OVERLAY_YMAX_FACTOR = 9.0


def _value_colors(n: int) -> List[Any]:
    import matplotlib.pyplot as plt

    # ``plasma`` clipped to its lower 0.85: a dark-purple -> magenta -> orange
    # ramp that keeps the low->high value ordering legible. The bright-yellow
    # tail of perceptual maps washes out on a white background, so it is
    # dropped.
    cmap = plt.get_cmap("plasma")
    lo, hi = 0.0, 0.85
    if n <= 1:
        return [cmap(lo)]
    return [cmap(lo + (hi - lo) * i / (n - 1)) for i in range(n)]


def plot_start_detection(spec: Any, rows: List[Any], ctx: Any) -> Any:
    """Two panels: the FID with each value's detected chirp end marked (top),
    and the Σ|FT|-vs-start detection sweep (bottom).

    These knobs move the *chirp end* — the detector output — so the panels mark
    the located chirp end per value rather than a derived start (whether the
    user hardcodes ``start_us`` or adds the guard margin is downstream of
    detection). For the effect of the chosen start on the resulting spectrum,
    see the ``stage1.start_us`` / ``start.guard_margin_us`` knob.
    """
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r.result is not None]
    if not rows:
        return None

    leaf = spec.path.split(".")[-1]
    colors = _value_colors(len(rows))
    fig, (axf, axs) = plt.subplots(2, 1, figsize=(9.0, 8.0))

    # Top: the FID, zoomed to the chirp transient + start markers.
    try:
        import ftmwpipeline.api as ftmw  # lazy

        fid = ftmw.load_fid(ctx.ftmw_path)
        t = fid.time_array_us()
        axf.plot(t, fid.data, lw=0.3, color="0.4")
        for row, color in zip(rows, colors):
            ce = row.result.chirp_end_us
            axf.axvline(
                ce, color=color, ls="--", alpha=0.85,
                label=f"{leaf}={row.value:g}: chirp-end {ce:.2f} us",
            )
        xmax = max(r.result.chirp_end_us for r in rows) + 1.0
        axf.set_xlim(0.0, xmax)
        axf.set_xlabel("time (us)")
        axf.set_ylabel("FID amplitude")
        axf.set_title("FID with detected chirp ends")
        axf.legend(fontsize=7)
    except Exception:
        # FID unavailable — keep the detection panel useful on its own.
        axf.set_visible(False)

    # Bottom: the Σ|FT|-vs-start sweep with each value's detected chirp end marked.
    for row, color in zip(rows, colors):
        r = row.result
        axs.semilogy(r.starts_us, r.sum_magnitude, color=color, alpha=0.8, lw=1.2,
                     label=f"{leaf}={row.value:g}")
        axs.axvline(r.chirp_end_us, color=color, ls=":", alpha=0.7)
    axs.set_xlabel("FID window start (us)")
    axs.set_ylabel("Σ|FT| (integrated magnitude)")
    axs.set_title("Σ|FT| vs start")
    axs.legend(fontsize=7)

    fig.suptitle(f"Start-detection sweep: {spec.path}")
    fig.tight_layout()
    return fig


def plot_spectra_ladder(spec: Any, rows: List[Any], ctx: Any) -> Any:
    """The spectrum-vs-start view shared by ``stage1.start_us`` and
    ``start.guard_margin_us``.

    A top FID panel marks each value's window start (and the chirp end, when the
    knob references it), over a stack of active-band |FT| panels — one per value.
    The spectra are **linear** with a **shared** y-limit scaled to the floor
    (from the percentile metric) so the chirp/ringdown "fuzz" is visible and its
    collapse from panel to panel is obvious; real lines clip off the top.

    Each ``row.result`` is an ``FtAtStart`` (``.ft`` / ``.start_us`` /
    ``.chirp_end_us``). Height grows with the number of values.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r.result is not None]
    if not rows:
        return None

    leaf = spec.path.split(".")[-1]
    n = len(rows)
    colors = _value_colors(n)
    panel_h = _LADDER_WIDTH_IN / _LADDER_PANEL_ASPECT
    fig, axes_grid = plt.subplots(
        n + 1, 1, figsize=(_LADDER_WIDTH_IN, panel_h * (n + 1)), squeeze=False
    )
    axes = list(axes_grid[:, 0])
    fid_ax, spec_axes = axes[0], axes[1:]

    starts = [getattr(r.result, "start_us", r.value) for r in rows]
    chirp_end = next(
        (r.result.chirp_end_us for r in rows
         if getattr(r.result, "chirp_end_us", None) is not None),
        None,
    )

    # Top panel: the FID with the window-start positions (and chirp end) marked.
    try:
        import ftmwpipeline.api as ftmw  # lazy

        fid = ftmw.load_fid(ctx.ftmw_path)
        fid_ax.plot(fid.time_array_us(), fid.data, lw=0.3, color="0.4")
        if chirp_end is not None:
            fid_ax.axvline(chirp_end, color="k", ls=":", lw=1.3,
                           label=f"chirp-end {chirp_end:.2f} us")
        for r, color, s in zip(rows, colors, starts):
            fid_ax.axvline(s, color=color, ls="--", alpha=0.85,
                           label=f"{leaf}={r.value:g}: start {s:.2f} us")
        fid_ax.set_xlim(0.0, max(starts) + 1.0)
        fid_ax.set_xlabel("time (us)")
        fid_ax.set_ylabel("FID amplitude")
        fid_ax.set_title("FID with window-start positions")
        fid_ax.legend(fontsize=7)
    except Exception:
        fid_ax.set_visible(False)

    # Shared linear y scaled to the floor so the residue (not the lines) is read.
    p50s = [r.metrics.get("p50") for r in rows
            if isinstance(r.metrics.get("p50"), (int, float))]
    top = _LADDER_YMAX_P50_FACTOR * max(p50s) if p50s and max(p50s) > 0 else None

    for ax, row, color in zip(spec_axes, rows, colors):
        ft = row.result.ft
        f_ghz = np.asarray(ft.freq_array, dtype=float) / 1000.0
        ax.plot(f_ghz, np.abs(ft.complex_spectrum), lw=0.4, color=color)
        ax.set_ylabel(f"{leaf}={row.value:g}", fontsize=9)
        if top is not None:
            ax.set_ylim(0.0, top)
        ax.grid(True, alpha=0.2)
    spec_axes[0].set_title(
        f"active-band |FT| vs {leaf} (linear, shared y scaled to the floor)"
    )
    spec_axes[-1].set_xlabel("frequency (GHz)")
    fig.tight_layout()
    return fig


def _contrib_alpha(n: int) -> float:
    """Per-contributor line alpha keyed to the contributor count, so the
    overplotted decay cloud reads as a density regardless of population.

    Halves once per decade: byte ``2^(5 - log10 n)`` -> 0x08 at 100, 0x04 at
    1k, 0x02 at 10k, 0x01 at 100k. Capped at 0x10 for small populations and
    floored at 0x01 so even a huge cloud keeps one bit of opacity.
    """
    import math

    if n <= 1:
        return 0x10 / 255.0
    byte = 2.0 ** (5.0 - math.log10(n))
    byte = max(1.0, min(float(0x10), byte))
    return byte / 255.0


def _plot_contributor_decays(ax: Any, row: Any, leaf: str, fit_color: str) -> None:
    """One value's contributor population as faint per-contributor decay curves
    (``exp(-t/τ_i)``) with the majority fit ``exp(-t/τ_maj)`` and a ±σ_τ band
    overlaid. Hidden (axis off) when the result carries no contributor taus."""
    import numpy as np
    from matplotlib.collections import LineCollection

    res = row.result
    taus = np.asarray(getattr(res, "contributor_taus_us", []), dtype=float)
    taus = taus[np.isfinite(taus) & (taus > 0.0)]
    start = getattr(res, "start_us", None)
    end = getattr(res, "end_us", None)
    if taus.size == 0 or start is None or end is None or float(end) <= float(start):
        ax.set_visible(False)
        return

    span = float(end) - float(start)
    t = np.linspace(0.0, span, 80)
    # Vectorised cloud: one normalised exponential per contributor.
    curves = np.exp(-t[None, :] / taus[:, None])  # (n_contrib, n_t)
    xs_grid = np.broadcast_to(t, curves.shape)
    segments = np.stack([xs_grid, curves], axis=-1)  # (n_contrib, n_t, 2)
    ax.add_collection(
        LineCollection(segments, colors="black",
                       alpha=_contrib_alpha(taus.size), linewidths=0.5)
    )

    tau_maj = float(res.tau_maj_us)
    sig = float(res.sigma_tau_us)
    ax.plot(t, np.exp(-t / tau_maj), color=fit_color, lw=2.0,
            label=rf"fit $\tau_{{maj}}$={tau_maj:.2f} us  (n={taus.size})")
    lo_tau = max(tau_maj - sig, 1e-3)
    upper = np.exp(-t / (tau_maj + sig))
    lower = np.exp(-t / lo_tau)
    ax.fill_between(t, upper, lower, color=fit_color, alpha=0.12)
    # Dotted opaque edges demarcate the ±σ_τ envelope clearly over the cloud.
    ax.plot(t, upper, color=fit_color, ls=":", lw=1.3,
            label=rf"$\pm\sigma_\tau$={sig:.2f} us")
    ax.plot(t, lower, color=fit_color, ls=":", lw=1.3)
    ax.set_xlim(0.0, span)
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel(f"{leaf}={row.value:g}\nnorm. decay", fontsize=8)
    ax.legend(fontsize=7, loc="upper right")


def _plot_tau_vs_freq(ax: Any, row: Any, leaf: str) -> None:
    """One value's contributor τ vs molecular frequency, colored by log10(SNR),
    with the per-band majority steps and the global τ_maj overlaid. Hidden (axis
    off) when the result carries no contributor frequencies."""
    import numpy as np

    res = row.result
    freqs = np.asarray(getattr(res, "contributor_freqs_mhz", []), dtype=float)
    taus = np.asarray(getattr(res, "contributor_taus_us", []), dtype=float)
    snrs = np.asarray(getattr(res, "contributor_snrs", []), dtype=float)
    ok = (
        freqs.size > 0
        and freqs.size == taus.size
        and np.isfinite(freqs).any()
        and np.isfinite(taus).any()
    )
    if not ok:
        ax.set_visible(False)
        return

    f_ghz = freqs / 1000.0
    if snrs.size == freqs.size and np.isfinite(snrs).any():
        c = np.log10(np.clip(snrs, 1.0, None))
    else:
        c = "0.4"
    sc = ax.scatter(f_ghz, taus, c=c, s=6, alpha=0.5, cmap="viridis",
                    linewidths=0.0)
    if not isinstance(c, str):
        cb = ax.figure.colorbar(sc, ax=ax, pad=0.01, fraction=0.04)
        cb.set_label(r"$\log_{10}$ SNR", fontsize=7)
        cb.ax.tick_params(labelsize=6)

    tau_maj = float(res.tau_maj_us)
    ax.axhline(tau_maj, color="crimson", ls="--", lw=1.5,
               label=rf"$\tau_{{maj}}$={tau_maj:.2f} us")
    # Per-band SNR-weighted majority as horizontal segments spanning each band.
    bands = getattr(res, "band_majorities", ()) or ()
    for b in bands:
        ax.plot([b.freq_lo_mhz / 1000.0, b.freq_hi_mhz / 1000.0],
                [b.tau_maj_us, b.tau_maj_us], color="crimson", lw=2.4,
                solid_capstyle="butt")
    tau_max = getattr(res, "tau_max_us", None)
    if tau_max is not None and float(tau_max) > 0:
        ax.set_ylim(0.0, min(float(tau_max), float(np.nanmax(taus)) * 1.15))
    ax.set_ylabel(f"{leaf}={row.value:g}\nτ (us)", fontsize=8)
    ax.legend(fontsize=7, loc="upper right")


def plot_tau_trend(spec: Any, rows: List[Any], ctx: Any) -> Any:
    """tau_maj +/- sigma_tau vs the knob value with the contributor count on a
    twin axis (top, spanning), then per grid value a pair of panels: the
    contributor decay cloud with the majority fit (left) and contributor τ vs
    molecular frequency with the per-band majorities (right), so both the spread
    τ_maj summarises and any frequency-dependence are visible.
    Returns ``None`` for non-numeric knobs (table-only)."""
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r.result is not None]
    if not rows:
        return None
    try:
        xs = [float(r.value) for r in rows]
    except (TypeError, ValueError):
        return None

    leaf = spec.path.split(".")[-1]
    tau = [r.metrics.get("tau_maj_us") for r in rows]
    sigma = [r.metrics.get("sigma_tau_us") for r in rows]
    n_contrib = [r.metrics.get("n_contributors") for r in rows]

    n = len(rows)
    fig = plt.figure(figsize=(12.0, 4.8 + 2.6 * n))
    gs = fig.add_gridspec(n + 1, 2, height_ratios=[1.5] + [1.0] * n)
    ax = fig.add_subplot(gs[0, :])

    ax.errorbar(xs, tau, yerr=sigma, fmt="o-", color="tab:blue", capsize=3,
                label=r"$\tau_{maj} \pm \sigma_\tau$")
    ax.set_xlabel(leaf)
    ax.set_ylabel(r"$\tau_{maj}$ (us)", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    axn = ax.twinx()
    axn.plot(xs, n_contrib, "s--", color="tab:green", label="n_contributors")
    axn.set_ylabel("n_contributors", color="tab:green")
    axn.tick_params(axis="y", labelcolor="tab:green")
    ax.set_title(f"tau calibration vs {leaf}")

    decay_axes = [fig.add_subplot(gs[i + 1, 0]) for i in range(n)]
    freq_axes = [fig.add_subplot(gs[i + 1, 1]) for i in range(n)]
    for axd, axf, row in zip(decay_axes, freq_axes, rows):
        _plot_contributor_decays(axd, row, leaf, "crimson")
        _plot_tau_vs_freq(axf, row, leaf)
    for col_axes, xlabel in (
        (decay_axes, "time since active-region start (us)"),
        (freq_axes, "molecular frequency (GHz)"),
    ):
        for ax_ in reversed(col_axes):
            if ax_.get_visible():
                ax_.set_xlabel(xlabel)
                break

    fig.suptitle(f"tau calibration sweep: {spec.path}")
    fig.tight_layout()
    return fig


def plot_shape_vote(spec: Any, rows: List[Any], ctx: Any) -> Any:
    """exp / gauss / voigt SNR-weighted vote rate per grid value (grouped bars),
    annotated with the per-value recommended shape. For the shape-recommendation
    knobs. Returns ``None`` for non-numeric knobs (table-only)."""
    import numpy as np
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r.result is not None]
    if not rows:
        return None
    try:
        xs = [float(r.value) for r in rows]
    except (TypeError, ValueError):
        return None

    leaf = spec.path.split(".")[-1]
    models = ("exp", "gauss", "voigt")
    model_colors = {"exp": "tab:blue", "gauss": "crimson", "voigt": "tab:gray"}
    idx = np.arange(len(rows), dtype=float)
    width = 0.26

    fig, ax = plt.subplots(figsize=(max(7.0, 1.6 * len(rows)), 5.0))
    for k, model in enumerate(models):
        vals = [float(r.metrics.get(model) or 0.0) for r in rows]
        ax.bar(idx + (k - 1) * width, vals, width,
               color=model_colors[model], label=model)
    for i, r in enumerate(rows):
        rec = r.metrics.get("recommended_shape")
        ax.annotate("none" if rec in (None, "") else str(rec),
                    (idx[i], 1.02), ha="center", va="bottom", fontsize=8,
                    rotation=0)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{v:g}" for v in xs])
    ax.set_ylim(0.0, 1.15)
    ax.set_xlabel(leaf)
    ax.set_ylabel("SNR-weighted vote rate")
    ax.set_title(f"shape vote vs {leaf}  (label = recommended_shape)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def plot_noise_sweep(spec: Any, rows: List[Any], ctx: Any) -> Any:
    """Three panels: σ(f) per grid value (top-left), the scalar metric trend
    (top-right, median σ and noise-flagged fraction vs the knob), and a
    full-width view of each value's σ(f) superimposed on the actual spectrum
    (bottom), zoomed to the noise band so the impact on the spectrum is legible.

    The spectrum is invariant across the sweep — only the σ estimate moves — so
    it is loaded once from ``ctx.ftmw_path`` and the per-value σ(f) curves
    overlay it. If the spectrum cannot be loaded the bottom panel is hidden and
    the two trend panels stand on their own.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r.result is not None]
    if not rows:
        return None

    leaf = spec.path.split(".")[-1]
    colors = _value_colors(len(rows))
    fig = plt.figure(figsize=(12.0, 9.0))
    gs = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.05))
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    for row, color in zip(rows, colors):
        sigma = np.asarray(row.result.rms_noise, dtype=float)
        ax1.plot(
            np.arange(sigma.size), sigma, color=color, alpha=0.8, lw=1.0,
            label=f"{leaf}={row.value:g}",
        )
    ax1.set_xlabel("frequency bin (trimmed grid)")
    ax1.set_ylabel(r"$\sigma_x$ (rms_noise)")
    ax1.set_title(r"$\sigma(f)$ per knob value")
    ax1.legend(fontsize=8)

    vals = [row.value for row in rows]
    median = [row.metrics.get("median_sigma") for row in rows]
    frac = [row.metrics.get("noise_fraction") for row in rows]
    ax2.plot(vals, median, "o-", color="tab:blue", label="median σ")
    ax2.set_xlabel(leaf)
    ax2.set_ylabel("median σ", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2b = ax2.twinx()
    ax2b.plot(vals, frac, "s--", color="tab:orange", label="noise fraction")
    ax2b.set_ylabel("noise-flagged fraction", color="tab:orange")
    ax2b.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_title("metric trend")

    # Bottom (full width): σ(f) over the spectrum, zoomed to the noise band.
    try:
        import ftmwpipeline.api as ftmw  # lazy

        ft = ftmw.compute_ft(ctx.ftmw_path)
        f_ghz = np.asarray(ft.freq_array, dtype=float) / 1000.0
        mag = np.abs(np.asarray(ft.complex_spectrum, dtype=float))
        ax3.plot(f_ghz, mag, lw=0.4, color="0.6", label="|FT|", zorder=1)
        sigma_max = 0.0
        for row, color in zip(rows, colors):
            sigma = np.asarray(row.result.rms_noise, dtype=float)
            if sigma.size != f_ghz.size:
                continue
            ax3.plot(f_ghz, sigma, color=color, lw=1.3, zorder=2,
                     label=f"σ: {leaf}={row.value:g}")
            sigma_max = max(sigma_max, float(np.nanmax(sigma)))
        if sigma_max > 0.0:
            ax3.set_ylim(0.0, _NOISE_OVERLAY_YMAX_FACTOR * sigma_max)
        ax3.set_xlabel("frequency (GHz)")
        ax3.set_ylabel(r"amplitude (|FT|, $\sigma_x$)")
        ax3.set_title(
            f"σ(f) over the spectrum "
            f"(zoomed to {_NOISE_OVERLAY_YMAX_FACTOR:g}× max σ; real lines clip)"
        )
        ax3.legend(fontsize=7, ncol=2)
        ax3.grid(True, alpha=0.2)
    except Exception:
        ax3.set_visible(False)

    fig.suptitle(f"Noise sweep: {spec.path}")
    fig.tight_layout()
    return fig
