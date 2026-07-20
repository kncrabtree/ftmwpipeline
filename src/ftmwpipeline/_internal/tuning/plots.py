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

from typing import Any, List, cast

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

# Stage 3 peak-detection zoom panels: width of each zoom region (MHz), how many
# regions to show, and the y-limits of the log-scaled spectrum so the noise
# floor sits near the bottom of the axis (not filling it). The lower limit is
# this fraction of the region's median σ; the upper is this multiple of the
# region's tallest line.
_PEAK_REGION_WIDTH_MHZ = 100.0
_PEAK_N_REGIONS = 3
_PEAK_YMIN_SIGMA_FRACTION = 0.5
_PEAK_YMAX_PEAK_FACTOR = 1.5
# Frequency tolerance (MHz) for treating a promoted peak as the "same" line
# across swept values when coloring it by the last value at which it survives.
_PEAK_PERSIST_TOL_MHZ = 0.05


def _resolve_regions(
    ctx: Any,
    rows: List[Any],
    default_width: float,
    default_n: int,
    auto_select: Any,
) -> List[Any]:
    """Resolve the zoom regions for a region-based adapter, honouring the user's
    ``PlotContext`` zoom controls.

    Explicit ``ctx.zoom_regions`` win verbatim (the user pinned the windows);
    otherwise the adapter's ``auto_select(rows, width, n)`` runs with the user's
    ``zoom_width_mhz`` / ``n_zoom`` when set, else the adapter defaults.
    """
    explicit = getattr(ctx, "zoom_regions", ()) or ()
    if explicit:
        return [(float(lo), float(hi)) for lo, hi in explicit]
    width = getattr(ctx, "zoom_width_mhz", None) or default_width
    n = getattr(ctx, "n_zoom", None) or default_n
    return cast(List[Any], auto_select(rows, width, n))


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
                ce,
                color=color,
                ls="--",
                alpha=0.85,
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
        axs.semilogy(
            r.starts_us,
            r.sum_magnitude,
            color=color,
            alpha=0.8,
            lw=1.2,
            label=f"{leaf}={row.value:g}",
        )
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
    import matplotlib.pyplot as plt
    import numpy as np

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
        (
            r.result.chirp_end_us
            for r in rows
            if getattr(r.result, "chirp_end_us", None) is not None
        ),
        None,
    )

    # Top panel: the FID with the window-start positions (and chirp end) marked.
    try:
        import ftmwpipeline.api as ftmw  # lazy

        fid = ftmw.load_fid(ctx.ftmw_path)
        fid_ax.plot(fid.time_array_us(), fid.data, lw=0.3, color="0.4")
        if chirp_end is not None:
            fid_ax.axvline(
                chirp_end,
                color="k",
                ls=":",
                lw=1.3,
                label=f"chirp-end {chirp_end:.2f} us",
            )
        for r, color, s in zip(rows, colors, starts):
            fid_ax.axvline(
                s,
                color=color,
                ls="--",
                alpha=0.85,
                label=f"{leaf}={r.value:g}: start {s:.2f} us",
            )
        fid_ax.set_xlim(0.0, max(starts) + 1.0)
        fid_ax.set_xlabel("time (us)")
        fid_ax.set_ylabel("FID amplitude")
        fid_ax.set_title("FID with window-start positions")
        fid_ax.legend(fontsize=7)
    except Exception:
        fid_ax.set_visible(False)

    # Shared linear y scaled to the floor so the residue (not the lines) is read.
    p50s = [
        r.metrics.get("p50")
        for r in rows
        if isinstance(r.metrics.get("p50"), (int, float))
    ]
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


def plot_ft_band_stack(spec: Any, rows: List[Any], ctx: Any) -> Any:
    """Stack each value's active-band |FT| (one panel per value), for the FT
    band/window knobs (``trim_min_mhz`` / ``trim_max_mhz`` / ``end_us``) where
    the spectrum itself is the thing the knob changes.

    Linear, with a shared y-limit scaled to the floor (from the percentile
    metric) so the noise floor and its change across values are readable; real
    lines clip off the top. No FID panel — unlike the start ladder these knobs
    do not move the window start. Each ``row.result`` carries ``.ft``.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [r for r in rows if r.result is not None]
    if not rows:
        return None

    leaf = spec.path.split(".")[-1]
    n = len(rows)
    colors = _value_colors(n)
    panel_h = _LADDER_WIDTH_IN / _LADDER_PANEL_ASPECT
    fig, axes_grid = plt.subplots(
        n, 1, figsize=(_LADDER_WIDTH_IN, panel_h * n), squeeze=False
    )
    axes = list(axes_grid[:, 0])

    p50s = [
        r.metrics.get("p50")
        for r in rows
        if isinstance(r.metrics.get("p50"), (int, float))
    ]
    top = _LADDER_YMAX_P50_FACTOR * max(p50s) if p50s and max(p50s) > 0 else None

    for ax, row, color in zip(axes, rows, colors):
        ft = row.result.ft
        f_ghz = np.asarray(ft.freq_array, dtype=float) / 1000.0
        ax.plot(f_ghz, np.abs(ft.complex_spectrum), lw=0.4, color=color)
        ax.set_ylabel(f"{leaf}={row.value:g}", fontsize=9)
        if top is not None:
            ax.set_ylim(0.0, top)
        ax.grid(True, alpha=0.2)
    axes[0].set_title(
        f"active-band |FT| vs {leaf} (linear, shared y scaled to the floor)"
    )
    axes[-1].set_xlabel("frequency (GHz)")
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
    return float(byte / 255.0)


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
    # Vectorised cloud: one normalized exponential per contributor.
    curves = np.exp(-t[None, :] / taus[:, None])  # (n_contrib, n_t)
    xs_grid = np.broadcast_to(t, curves.shape)
    segments = np.stack([xs_grid, curves], axis=-1)  # (n_contrib, n_t, 2)
    ax.add_collection(
        LineCollection(
            segments, colors="black", alpha=_contrib_alpha(taus.size), linewidths=0.5
        )
    )

    tau_maj = float(res.tau_maj_us)
    sig = float(res.sigma_tau_us)
    ax.plot(
        t,
        np.exp(-t / tau_maj),
        color=fit_color,
        lw=2.0,
        label=rf"fit $\tau_{{maj}}$={tau_maj:.2f} us  (n={taus.size})",
    )
    lo_tau = max(tau_maj - sig, 1e-3)
    upper = np.exp(-t / (tau_maj + sig))
    lower = np.exp(-t / lo_tau)
    ax.fill_between(t, upper, lower, color=fit_color, alpha=0.12)
    # Dotted opaque edges demarcate the ±σ_τ envelope clearly over the cloud.
    ax.plot(
        t,
        upper,
        color=fit_color,
        ls=":",
        lw=1.3,
        label=rf"$\pm\sigma_\tau$={sig:.2f} us",
    )
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
    sc = ax.scatter(f_ghz, taus, c=c, s=6, alpha=0.5, cmap="viridis", linewidths=0.0)
    if not isinstance(c, str):
        cb = ax.figure.colorbar(sc, ax=ax, pad=0.01, fraction=0.04)
        cb.set_label(r"$\log_{10}$ SNR", fontsize=7)
        cb.ax.tick_params(labelsize=6)

    tau_maj = float(res.tau_maj_us)
    ax.axhline(
        tau_maj,
        color="crimson",
        ls="--",
        lw=1.5,
        label=rf"$\tau_{{maj}}$={tau_maj:.2f} us",
    )
    # Per-band SNR-weighted majority as horizontal segments spanning each band.
    bands = getattr(res, "band_majorities", ()) or ()
    for b in bands:
        ax.plot(
            [b.freq_lo_mhz / 1000.0, b.freq_hi_mhz / 1000.0],
            [b.tau_maj_us, b.tau_maj_us],
            color="crimson",
            lw=2.4,
            solid_capstyle="butt",
        )
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
    τ_maj summarizes and any frequency-dependence are visible.
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

    ax.errorbar(
        xs,
        tau,
        yerr=sigma,
        fmt="o-",
        color="tab:blue",
        capsize=3,
        label=r"$\tau_{maj} \pm \sigma_\tau$",
    )
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
    import matplotlib.pyplot as plt
    import numpy as np

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
        ax.bar(
            idx + (k - 1) * width, vals, width, color=model_colors[model], label=model
        )
    for i, r in enumerate(rows):
        rec = r.metrics.get("recommended_shape")
        ax.annotate(
            "none" if rec in (None, "") else str(rec),
            (idx[i], 1.02),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{v:g}" for v in xs])
    ax.set_ylim(0.0, 1.15)
    ax.set_xlabel(leaf)
    ax.set_ylabel("SNR-weighted vote rate")
    ax.set_title(f"shape vote vs {leaf}  (label = recommended_shape)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def _peaks_in(peaks: Any, lo: float, hi: float) -> List[Any]:
    return [p for p in peaks if lo <= float(p.frequency) <= hi]


def _select_peak_regions(rows: List[Any], width_mhz: float, k: int) -> List[Any]:
    """Pick up to ``k`` ~``width_mhz``-wide frequency windows to zoom into.

    Regions are ranked by how much the detected/promoted peak set *diverges*
    across the swept values (the variance, per window, of the per-value total
    and promoted peak counts) so the panels land where the knob actually
    changes the outcome. When nothing diverges (every value detects the same
    set), the ranking falls back to the *richest* windows (most peaks), so the
    panels still show where the action is.

    Returns a list of ``(lo_mhz, hi_mhz)`` tuples, low-frequency first.
    """
    import numpy as np

    fts = [r.result.get("active_ft") for r in rows if r.result is not None]
    fts = [ft for ft in fts if ft is not None]
    if not fts:
        return []
    freqs = np.asarray(fts[0].freq_array, dtype=float)
    f_lo, f_hi = float(freqs.min()), float(freqs.max())
    if not np.isfinite(f_lo) or f_hi <= f_lo:
        return []

    n_bins = max(1, int(np.ceil((f_hi - f_lo) / width_mhz)))
    edges = [
        (f_lo + i * width_mhz, min(f_lo + (i + 1) * width_mhz, f_hi))
        for i in range(n_bins)
    ]

    peak_sets = [r.result.get("peaks", []) for r in rows if r.result is not None]
    scored = []
    for lo, hi in edges:
        totals = np.array([len(_peaks_in(ps, lo, hi)) for ps in peak_sets], dtype=float)
        promoted = np.array(
            [
                sum(1 for p in _peaks_in(ps, lo, hi) if p.properties.get("promoted"))
                for ps in peak_sets
            ],
            dtype=float,
        )
        divergence = float(totals.var() + promoted.var())
        richness = float(totals.max()) if totals.size else 0.0
        if richness <= 0.0:
            continue  # empty window — nothing to show
        scored.append((divergence, richness, lo, hi))
    if not scored:
        return []

    # Primary key: divergence; tiebreak / fallback: richness. When no window
    # diverges (all divergence == 0) this reduces to the k richest windows.
    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
    chosen = scored[:k]
    chosen.sort(key=lambda s: s[2])  # low frequency first for reading order
    return [(lo, hi) for _, _, lo, hi in chosen]


def _draw_peak_panel(
    ax: Any,
    value_label: str,
    ft: Any,
    sigma: Any,
    peaks: Any,
    promotion_min_snr: float,
    lo: float,
    hi: float,
) -> None:
    """One value × one region: the (log) active-FT magnitude zoomed to the
    region, the per-bin promotion threshold (``min_snr·σ``), and the value's
    peaks marked by detection pass (primary / gap) and promotion state."""
    import numpy as np

    freqs = np.asarray(ft.freq_array, dtype=float)
    mag = np.abs(np.asarray(ft.complex_spectrum))
    sel = (freqs >= lo) & (freqs <= hi)
    if not sel.any():
        ax.set_visible(False)
        return
    f_sel, m_sel = freqs[sel], mag[sel]
    order = np.argsort(f_sel)
    f_sel, m_sel = f_sel[order], m_sel[order]

    ax.plot(f_sel, m_sel, lw=0.5, color="0.55", zorder=1)

    sig = np.asarray(sigma, dtype=float)
    median_sigma = float("nan")
    if sig.size == freqs.size:
        sig_sel = sig[sel][order]
        thresh = promotion_min_snr * sig_sel
        ax.plot(f_sel, thresh, color="tab:red", ls="--", lw=0.9, zorder=2)
        finite = sig_sel[np.isfinite(sig_sel)]
        if finite.size:
            median_sigma = float(np.median(finite))

    # Promoted peaks only, solid, colored by detection pass (primary vs gap).
    # Dropped (below-cutoff) peaks are intentionally not drawn — the dashed
    # threshold already shows the cutoff, and the band-wide persistence panel
    # above carries the survival story.
    pass_color = {"primary": "tab:blue", "gap": "tab:green"}
    for p in _peaks_in(peaks, lo, hi):
        if not p.properties.get("promoted"):
            continue
        dp = p.properties.get("detection_pass", "primary")
        color = pass_color.get(dp, "tab:gray")
        ax.scatter(
            [float(p.frequency)],
            [float(p.intensity)],
            s=34,
            zorder=3,
            marker="v",
            color=color,
            edgecolors=color,
            linewidths=1.1,
        )

    ax.set_yscale("log")
    region_max = float(m_sel.max()) if m_sel.size else 1.0
    ymin = (
        _PEAK_YMIN_SIGMA_FRACTION * median_sigma
        if np.isfinite(median_sigma) and median_sigma > 0.0
        else max(region_max * 1e-3, 1e-12)
    )
    ax.set_ylim(ymin, _PEAK_YMAX_PEAK_FACTOR * region_max)
    ax.set_xlim(lo, hi)
    ax.set_ylabel(value_label, fontsize=8)
    ax.grid(True, alpha=0.2, which="both")


def _peak_persistence(rows: List[Any], tol_mhz: float) -> List[Any]:
    """Categorise each promoted peak by the *last* swept value at which it is
    still promoted.

    Promoted peaks are matched across values by frequency (within ``tol_mhz``);
    for each matched line the highest swept-value index at which it is promoted
    is recorded. Returns ``[(frequency_mhz, intensity, last_index), ...]`` sorted
    by frequency — the band-wide "how deep into the sweep does this line survive"
    summary. ``last_index`` indexes into ``rows`` (the sweep order).
    """
    persistence: "dict[int, tuple[int, float, float]]" = {}
    for i, row in enumerate(rows):
        if row.result is None:
            continue
        for p in row.result.get("peaks", []):
            if not p.properties.get("promoted"):
                continue
            key = int(round(float(p.frequency) / tol_mhz))
            # Iterating in sweep order and always overwriting leaves the highest
            # index at which the line survives.
            persistence[key] = (i, float(p.frequency), float(p.intensity))
    return sorted(
        ((freq, inten, idx) for idx, freq, inten in persistence.values()),
        key=lambda t: t[0],
    )


def _plot_peak_persistence(
    ax: Any,
    rows: List[Any],
    regions: List[Any],
    leaf: str,
    labels: List[str],
) -> None:
    """Full-width log spectrum with every promoted peak colored by the last
    swept value it survives (early-drop → survives-throughout), and the zoom
    regions shaded. The active FT is invariant across the sweep, so the spectrum
    is drawn once from the first row."""
    import numpy as np

    ft = rows[0].result.get("active_ft")
    if ft is None:
        ax.set_visible(False)
        return
    freqs = np.asarray(ft.freq_array, dtype=float)
    mag = np.abs(np.asarray(ft.complex_spectrum))
    order = np.argsort(freqs)
    ax.plot(freqs[order], mag[order], lw=0.4, color="0.6", zorder=1)

    # Shade the zoom regions detailed below. A cool tint + edge lines so the
    # bands read clearly against the gray spectrum and the warm plasma peak
    # colors (a gray shade blended in and was easy to miss).
    for lo, hi in regions:
        ax.axvspan(lo, hi, color="#6baed6", alpha=0.28, zorder=0)
        for edge in (lo, hi):
            ax.axvline(edge, color="#3182bd", lw=0.7, alpha=0.6, zorder=0)

    n = len(rows)
    colors = _value_colors(n)
    persistence = _peak_persistence(rows, _PEAK_PERSIST_TOL_MHZ)
    # Group by survival bucket so the legend stays one entry per swept value.
    for idx in range(n):
        pts = [(f, inten) for (f, inten, li) in persistence if li == idx]
        if not pts:
            continue
        fs, ins = zip(*pts)
        ax.scatter(
            list(fs),
            list(ins),
            s=16,
            color=colors[idx],
            zorder=3,
            label=f"survives to {leaf}={labels[idx]}",
        )

    sigma = np.asarray(rows[0].result.get("active_rms", []), dtype=float)
    finite = sigma[np.isfinite(sigma)]
    region_top = float(mag.max()) if mag.size else 1.0
    ymin = (
        _PEAK_YMIN_SIGMA_FRACTION * float(np.median(finite))
        if finite.size and float(np.median(finite)) > 0.0
        else max(region_top * 1e-3, 1e-12)
    )
    ax.set_yscale("log")
    ax.set_ylim(ymin, _PEAK_YMAX_PEAK_FACTOR * region_top)
    ax.set_xlim(float(freqs.min()), float(freqs.max()))
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("|FT|")
    ax.grid(True, alpha=0.2, which="both")
    ax.legend(fontsize=7, ncol=min(n, 6), loc="upper right")
    ax.set_title(
        "promoted peaks colored by the last sweep value they survive "
        "(shaded = zoom regions below)"
    )


def plot_peak_detection(spec: Any, rows: List[Any], ctx: Any) -> Any:
    """Stage 3 sweep view: a by-SNR-band count trend, a band-wide peak-survival
    panel, then per-value × per-region zoom detail.

    The active FT is invariant across the sweep — only which peaks are found and
    promoted changes. The top trend tracks the peaks passed to Stage 4 (total and
    by weak/medium/strong SNR band) vs the swept value; the full-width panel
    below it draws the whole band once and colors every
    promoted peak by the last value at which it survives (so a glance shows which
    lines drop out first), shading the zoom regions. Each remaining row is one
    swept value and each column one auto-selected ~100 MHz region (chosen where
    the peak set diverges most across values, falling back to the richest
    regions): log-scaled with the noise floor near the axis bottom, promoted
    peaks marked solid by detection pass (primary = blue, gap = green), the
    per-bin promotion threshold (``min_snr·σ``) dashed in red.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    rows = [r for r in rows if r.result is not None]
    if not rows:
        return None

    leaf = spec.path.split(".")[-1]
    regions = _resolve_regions(
        ctx, rows, _PEAK_REGION_WIDTH_MHZ, _PEAK_N_REGIONS, _select_peak_regions
    )
    n = len(rows)
    ncol = max(1, len(regions))
    has_zoom = bool(regions)
    n_rows = (n + 2) if has_zoom else 2

    fig = plt.figure(figsize=(5.0 * ncol, 5.6 + 2.0 * n))
    height_ratios = [1.6, 1.5] + ([1.0] * n if has_zoom else [])
    gs = fig.add_gridspec(n_rows, ncol, height_ratios=height_ratios)

    # Row 0 (spanning): the peaks passed to Stage 4, total and by SNR band, vs
    # the swept value. Categorical x keeps non-numeric knobs (toggles, window
    # names) plottable.
    ax_trend = fig.add_subplot(gs[0, :])
    xs = np.arange(n, dtype=float)
    labels = [
        f"{r.value:g}" if isinstance(r.value, (int, float)) else str(r.value)
        for r in rows
    ]
    for col, color, marker in (
        ("n_total", "0.2", "o"),
        ("n_strong", "tab:red", "^"),
        ("n_medium", "tab:orange", "s"),
        ("n_weak", "tab:blue", "v"),
    ):
        ys = [r.metrics.get(col) for r in rows]
        ax_trend.plot(xs, ys, marker + "-", color=color, label=col)
    ax_trend.set_xticks(xs)
    ax_trend.set_xticklabels(labels)
    ax_trend.set_xlabel(leaf)
    ax_trend.set_ylabel("peaks passed to Stage 4")
    ax_trend.grid(True, alpha=0.3)
    ax_trend.legend(fontsize=8, ncol=4, loc="upper left")
    ax_trend.set_title(f"Stage 3 peaks passed to Stage 4 vs {leaf} (by SNR band)")

    # Row 1 (spanning): band-wide peak survival.
    _plot_peak_persistence(fig.add_subplot(gs[1, :]), rows, regions, leaf, labels)

    if not has_zoom:
        fig.suptitle(f"Peak-detection sweep: {spec.path}")
        fig.tight_layout()
        return fig

    for ri, row in enumerate(rows):
        res = row.result
        ft = res.get("active_ft")
        sigma = res.get("active_rms")
        peaks = res.get("peaks", [])
        pmin = float(res.get("promotion_min_snr", 0.0))
        for ci, (lo, hi) in enumerate(regions):
            ax = fig.add_subplot(gs[ri + 2, ci])
            if ft is None:
                ax.set_visible(False)
                continue
            label = f"{leaf}={labels[ri]}" if ci == 0 else ""
            _draw_peak_panel(ax, label, ft, sigma, peaks, pmin, lo, hi)
            if ri == 0:
                ax.set_title(f"{lo:.0f}–{hi:.0f} MHz", fontsize=9)
            if ri == n - 1:
                ax.set_xlabel("frequency (MHz)")

    handles = [
        Line2D(
            [],
            [],
            marker="v",
            color="tab:blue",
            ls="none",
            markerfacecolor="tab:blue",
            label="primary (promoted)",
        ),
        Line2D(
            [],
            [],
            marker="v",
            color="tab:green",
            ls="none",
            markerfacecolor="tab:green",
            label="gap / secondary (promoted)",
        ),
        Line2D(
            [],
            [],
            color="tab:red",
            ls="--",
            label=r"promotion threshold ($min\_snr\cdot\sigma$)",
        ),
    ]
    fig.legend(
        handles=handles,
        fontsize=8,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(f"Peak-detection sweep: {spec.path}")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


# Stage 4 window-planning zoom panels: each region is a touch wider than the
# Stage 3 ones (a fit window runs up to tens of MHz, so the panel must hold a
# few of them and show their boundaries move). y-floor + ceiling mirror the
# peak panels so the noise band sits near the axis bottom.
_WINDOW_REGION_WIDTH_MHZ = 150.0
_WINDOW_N_REGIONS = 3
_WINDOW_YMIN_SIGMA_FRACTION = 0.5
_WINDOW_YMAX_PEAK_FACTOR = 1.5
_WINDOW_SPAN_COLOR = "tab:blue"


def _coherence_curve(result: Any) -> Any:
    """Recompute the rolling complex-edge coherence S_coh and its T_edge
    threshold for one Stage 4 result — the statistic that drove the partition.

    Mirrors the single-value window visualizer: de-ramp to the active turn-on,
    then roll the coherence over the persisted ``edge_m`` band. Returns
    ``(freqs_ordered, S_coh, threshold)`` or ``None`` when the inputs are absent.
    """
    import numpy as np

    plan = result.get("plan")
    ft = result.get("active_ft")
    rms = result.get("active_rms")
    if plan is None or ft is None or rms is None:
        return None
    try:
        from ...preprocessing.edge_coherence import coherence_curve
    except Exception:
        return None
    params = getattr(plan, "parameters", {}) or {}
    ordered_freq, rolling, thr, _ = coherence_curve(
        ft.freq_array, ft.complex_spectrum, rms, params
    )
    return ordered_freq, np.asarray(rolling, dtype=float), thr


def _select_window_regions(rows: List[Any], width_mhz: float, k: int) -> List[Any]:
    """Pick up to ``k`` ~``width_mhz``-wide windows to zoom into, ranked by how
    much the *partition* diverges across the swept values.

    Per candidate window the score is the variance, across values, of the number
    of window edges falling inside it — so the panels land where the knob
    actually moves boundaries. When nothing diverges the ranking falls back to
    the richest windows (most edges), so the panels still show structure.
    Returns ``(lo_mhz, hi_mhz)`` tuples, low-frequency first.
    """
    import numpy as np

    fts = [r.result.get("active_ft") for r in rows if r.result is not None]
    fts = [ft for ft in fts if ft is not None]
    if not fts:
        return []
    freqs = np.asarray(fts[0].freq_array, dtype=float)
    f_lo, f_hi = float(freqs.min()), float(freqs.max())
    if not np.isfinite(f_lo) or f_hi <= f_lo:
        return []

    n_bins = max(1, int(np.ceil((f_hi - f_lo) / width_mhz)))
    edges = [
        (f_lo + i * width_mhz, min(f_lo + (i + 1) * width_mhz, f_hi))
        for i in range(n_bins)
    ]
    plans = [r.result.get("plan") for r in rows if r.result is not None]
    plans = [p for p in plans if p is not None]

    scored = []
    for lo, hi in edges:
        n_edges = []
        for p in plans:
            e_in = 0
            for w in p.windows:
                wlo, whi = w.freq_range
                if whi < lo or wlo > hi:
                    continue
                e_in += sum(1 for e in (wlo, whi) if lo <= e <= hi)
            n_edges.append(e_in)
        ne = np.asarray(n_edges, dtype=float)
        richness = float(ne.max()) if ne.size else 0.0
        if richness <= 0.0:
            continue  # no windows here — nothing to show
        scored.append((float(ne.var()), richness, lo, hi))
    if not scored:
        return []

    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
    chosen = scored[:k]
    chosen.sort(key=lambda s: s[2])  # low frequency first for reading order
    return [(lo, hi) for _, _, lo, hi in chosen]


def _plot_boundary_shift(
    ax: Any,
    rows: List[Any],
    regions: List[Any],
    leaf: str,
    labels: List[str],
) -> None:
    """Full-width log spectrum drawn once, with every swept value's window
    boundaries overlaid as vertical lines colored by value — so a glance shows
    how the partition walks as the knob changes. The zoom regions detailed below
    are shaded."""
    import numpy as np

    ft = rows[0].result.get("active_ft")
    if ft is None:
        ax.set_visible(False)
        return
    freqs = np.asarray(ft.freq_array, dtype=float)
    mag = np.abs(np.asarray(ft.complex_spectrum))
    order = np.argsort(freqs)
    ax.plot(freqs[order], mag[order], lw=0.4, color="0.45", zorder=1)

    n = len(rows)
    colors = _value_colors(n)
    for i, row in enumerate(rows):
        plan = row.result.get("plan")
        if plan is None:
            continue
        for w in plan.windows:
            lo, hi = w.freq_range
            for edge in (lo, hi):
                ax.axvline(edge, color=colors[i], lw=0.6, alpha=0.65, zorder=2)
        ax.plot([], [], color=colors[i], lw=1.4, label=f"{leaf}={labels[i]}")

    for lo, hi in regions:
        ax.axvspan(lo, hi, color="#6baed6", alpha=0.10, zorder=0)

    sigma = np.asarray(rows[0].result.get("active_rms", []), dtype=float)
    finite = sigma[np.isfinite(sigma)]
    top = float(mag.max()) if mag.size else 1.0
    ymin = (
        _WINDOW_YMIN_SIGMA_FRACTION * float(np.median(finite))
        if finite.size and float(np.median(finite)) > 0.0
        else max(top * 1e-3, 1e-12)
    )
    ax.set_yscale("log")
    ax.set_ylim(ymin, _WINDOW_YMAX_PEAK_FACTOR * top)
    ax.set_xlim(float(freqs.min()), float(freqs.max()))
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("|FT|")
    ax.grid(True, alpha=0.2, which="both")
    ax.legend(fontsize=7, ncol=min(n, 6), loc="upper right")
    ax.set_title(
        "window boundaries per swept value (colored by value; "
        "shaded = zoom regions below)"
    )


def _draw_window_panel(
    ax: Any,
    value_label: str,
    result: Any,
    peaks: Any,
    lo: float,
    hi: float,
) -> None:
    """One value × one region: the log active-FT magnitude zoomed to the region,
    the window spans shaded with their boundaries, free peaks (filled) vs fixed
    contributors (open square), and the S_coh coherence statistic with its
    T_edge threshold on a twin axis."""
    import numpy as np

    ft = result.get("active_ft")
    freqs = np.asarray(ft.freq_array, dtype=float)
    mag = np.abs(np.asarray(ft.complex_spectrum))
    sel = (freqs >= lo) & (freqs <= hi)
    if not sel.any():
        ax.set_visible(False)
        return
    f_sel, m_sel = freqs[sel], mag[sel]
    order = np.argsort(f_sel)
    f_sel, m_sel = f_sel[order], m_sel[order]
    ax.plot(f_sel, m_sel, lw=0.5, color="0.35", zorder=1)

    plan = result.get("plan")
    for w in plan.windows:
        wlo, whi = w.freq_range
        if whi < lo or wlo > hi:
            continue
        ax.axvspan(
            max(wlo, lo),
            min(whi, hi),
            color=_WINDOW_SPAN_COLOR,
            alpha=0.10,
            zorder=0,
        )
        for edge in (wlo, whi):
            if lo <= edge <= hi:
                ax.axvline(edge, color="0.4", lw=0.6, zorder=2)

    # Free peaks (filled black) vs fixed contributors (open blue square).
    for w in plan.windows:
        for li in w.free_peak_indices:
            if 0 <= li < len(peaks) and lo <= float(peaks[li].frequency) <= hi:
                p = peaks[li]
                ax.scatter(
                    [p.frequency],
                    [p.intensity],
                    s=18,
                    marker="o",
                    color="black",
                    zorder=5,
                )
        for fc in w.fixed_contributors:
            idx = fc.peak_index
            if 0 <= idx < len(peaks) and lo <= float(peaks[idx].frequency) <= hi:
                p = peaks[idx]
                ax.scatter(
                    [p.frequency],
                    [p.intensity],
                    s=46,
                    marker="s",
                    facecolors="none",
                    edgecolors="tab:blue",
                    linewidths=1.3,
                    zorder=6,
                )

    coh = _coherence_curve(result)
    if coh is not None:
        cf, cs, thr = coh
        csel = (cf >= lo) & (cf <= hi)
        if csel.any():
            axc = ax.twinx()
            axc.plot(
                cf[csel], cs[csel], lw=0.8, color="tab:purple", alpha=0.55, zorder=4
            )
            axc.axhline(thr, color="crimson", lw=0.9, ls="--", zorder=4)
            axc.set_yscale("log")
            axc.set_ylabel("S_coh", fontsize=7, color="tab:purple")
            axc.tick_params(axis="y", labelsize=6, colors="tab:purple")

    ax.set_yscale("log")
    sig = np.asarray(result.get("active_rms", []), dtype=float)
    finite = sig[np.isfinite(sig)]
    region_max = float(m_sel.max()) if m_sel.size else 1.0
    ymin = (
        _WINDOW_YMIN_SIGMA_FRACTION * float(np.median(finite))
        if finite.size and float(np.median(finite)) > 0.0
        else max(region_max * 1e-3, 1e-12)
    )
    ax.set_ylim(ymin, _WINDOW_YMAX_PEAK_FACTOR * region_max)
    ax.set_xlim(lo, hi)
    ax.set_ylabel(value_label, fontsize=8)
    ax.grid(True, alpha=0.2, which="both")


def plot_window_planning(spec: Any, rows: List[Any], ctx: Any) -> Any:
    """Stage 4 sweep view: a window-count trend, a band-wide boundary-shift
    overlay, then per-value × per-region zoom detail.

    The active FT is invariant across the sweep — only the partition changes.
    The top trend tracks the plan's shape (n_windows, n_fixed contributors,
    n_dep) vs the swept value; the full-width panel below draws the band once
    and overlays every value's window boundaries colored by value, shading the
    zoom regions. Each remaining row is one swept value and each column one
    auto-selected ~150 MHz region (chosen where the partition diverges most
    across values): log-scaled, window spans shaded, free peaks filled / fixed
    contributors open, and the S_coh coherence statistic with its T_edge
    threshold on a twin axis — the statistic that set the boundaries.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    rows = [r for r in rows if r.result is not None]
    if not rows:
        return None

    leaf = spec.path.split(".")[-1]
    # Peaks are the Stage 3 output — invariant across the Stage 4 sweep — so
    # load them once from the working copy for the free/fixed markers.
    try:
        import ftmwpipeline.api as ftmw

        peaks = ftmw.load_peaks(ctx.ftmw_path)
    except Exception:
        peaks = []

    regions = _resolve_regions(
        ctx,
        rows,
        _WINDOW_REGION_WIDTH_MHZ,
        _WINDOW_N_REGIONS,
        _select_window_regions,
    )
    n = len(rows)
    ncol = max(1, len(regions))
    has_zoom = bool(regions)
    n_rows = (n + 2) if has_zoom else 2

    fig = plt.figure(figsize=(5.0 * ncol, 5.6 + 2.0 * n))
    height_ratios = [1.6, 1.7] + ([1.1] * n if has_zoom else [])
    gs = fig.add_gridspec(n_rows, ncol, height_ratios=height_ratios)

    ax_trend = fig.add_subplot(gs[0, :])
    xs = np.arange(n, dtype=float)
    labels = [
        f"{r.value:g}" if isinstance(r.value, (int, float)) else str(r.value)
        for r in rows
    ]
    for col, color, marker in (
        ("n_windows", "0.2", "o"),
        ("n_fixed", "tab:blue", "s"),
        ("n_dep", "tab:purple", "D"),
    ):
        ys = [r.metrics.get(col) for r in rows]
        ax_trend.plot(xs, ys, marker + "-", color=color, label=col)
    ax_trend.set_xticks(xs)
    ax_trend.set_xticklabels(labels)
    ax_trend.set_xlabel(leaf)
    ax_trend.set_ylabel("plan counts")
    ax_trend.grid(True, alpha=0.3)
    ax_trend.legend(fontsize=8, ncol=4, loc="upper left")
    ax_trend.set_title(f"Stage 4 window plan vs {leaf}")

    _plot_boundary_shift(fig.add_subplot(gs[1, :]), rows, regions, leaf, labels)

    if not has_zoom:
        fig.suptitle(f"Window-planning sweep: {spec.path}")
        fig.tight_layout()
        return fig

    for ri, row in enumerate(rows):
        res = row.result
        for ci, (lo, hi) in enumerate(regions):
            ax = fig.add_subplot(gs[ri + 2, ci])
            if res.get("active_ft") is None:
                ax.set_visible(False)
                continue
            label = f"{leaf}={labels[ri]}" if ci == 0 else ""
            _draw_window_panel(ax, label, res, peaks, lo, hi)
            if ri == 0:
                ax.set_title(f"{lo:.0f}–{hi:.0f} MHz", fontsize=9)
            if ri == n - 1:
                ax.set_xlabel("frequency (MHz)")

    handles = [
        Line2D([], [], marker="o", color="black", ls="none", label="free peak"),
        Line2D(
            [],
            [],
            marker="s",
            color="tab:blue",
            ls="none",
            markerfacecolor="none",
            label="fixed contributor",
        ),
        Line2D([], [], color=_WINDOW_SPAN_COLOR, lw=6, alpha=0.4, label="fit window"),
        Line2D(
            [], [], color="tab:purple", lw=1.2, alpha=0.55, label=r"$S_{coh}$ coherence"
        ),
        Line2D(
            [], [], color="crimson", ls="--", label=r"$T_{edge}$ coherence threshold"
        ),
    ]
    fig.legend(
        handles=handles,
        fontsize=8,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(f"Window-planning sweep: {spec.path}")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig


def _fit_quality_rows(row: Any) -> List[Any]:
    """Per-window fit-quality dicts for one swept value (empty when absent)."""
    from .fit_support import window_fit_quality

    res = row.result
    if res is None:
        return []
    fit = res.get("fit")
    if fit is None or not getattr(fit, "window_fits", None):
        return []
    return [window_fit_quality(wf) for wf in fit.window_fits]


def plot_fit_quality(spec: Any, rows: List[Any], ctx: Any) -> Any:
    """Stage 5 fit-quality sweep view (the shared adapter for tau / conservative
    / penalties / seeder / baseline knobs).

    The honest fit-quality lens is the SNR-normalized shape-error fraction ε, not
    χ²ᵣ (which rides an SNR² floor). Three stacked sections:
    (1) an ε-percentile + fail-count / peak-count trend vs the swept value;
    (2) the headline **ε-vs-SNR scatter** — every fitted window, colored by
    swept value, with the pass boundary drawn as the flat line ε = κ (the gate
    ``χ²ᵣ ≤ F + (κ·SNR)²`` is exactly ``ε ≤ κ``): watch the knob push windows
    across the line; (3) a per-window **ε-vs-frequency strip** showing *where* on
    the band the knob moved the misfit, colored by value.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from ...fitting.validation import DEFAULT_SHAPE_ERROR_KAPPA

    rows = [r for r in rows if r.result is not None]
    per_value = [_fit_quality_rows(r) for r in rows]
    if not rows or not any(per_value):
        return None

    leaf = spec.path.split(".")[-1]
    n = len(rows)
    labels = [
        f"{r.value:g}" if isinstance(r.value, (int, float)) else str(r.value)
        for r in rows
    ]
    colors = _value_colors(n)
    kappa = float(DEFAULT_SHAPE_ERROR_KAPPA)

    fig = plt.figure(figsize=(12.0, 12.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.3, 1.6, 1.3])

    # (1) trend: eps percentiles + fail / peak counts vs value.
    ax = fig.add_subplot(gs[0, 0])
    xs = np.arange(n, dtype=float)
    e50 = [
        float(np.percentile([w["epsilon"] for w in pv], 50)) if pv else 0.0
        for pv in per_value
    ]
    e95 = [
        float(np.percentile([w["epsilon"] for w in pv], 95)) if pv else 0.0
        for pv in per_value
    ]
    ax.plot(xs, e50, "o-", color="tab:blue", label="ε p50")
    ax.plot(xs, e95, "s--", color="tab:blue", alpha=0.6, label="ε p95")
    ax.axhline(
        kappa, color="crimson", ls=":", lw=1.0, label=f"κ = {kappa:g} (pass ≤ κ)"
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel(leaf)
    ax.set_ylabel("shape-error ε")
    ax.grid(True, alpha=0.3)
    axc = ax.twinx()
    n_fail = [sum(1 for w in pv if not w["passed"]) for pv in per_value]
    n_peak = [sum(w["n_peaks"] for w in pv) for pv in per_value]
    axc.plot(xs, n_fail, "^-", color="tab:red", label="n_fail")
    axc.plot(xs, n_peak, "D-", color="0.4", label="n_peaks")
    axc.set_ylabel("window / peak counts")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axc.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, ncol=3, loc="upper left")
    ax.set_title(f"Stage 5 fit quality vs {leaf}")

    # (2) headline: ε vs SNR, colored by value, pass line at ε = κ.
    ax2 = fig.add_subplot(gs[1, 0])
    for i, pv in enumerate(per_value):
        snr = [w["snr_max"] for w in pv if w["snr_max"] > 0]
        eps = [w["epsilon"] for w in pv if w["snr_max"] > 0]
        if snr:
            ax2.scatter(
                snr, eps, s=20, color=colors[i], alpha=0.75, label=f"{leaf}={labels[i]}"
            )
    ax2.axhline(
        kappa,
        color="crimson",
        ls="--",
        lw=1.1,
        label=f"pass boundary ε = κ = {kappa:g}",
    )
    ax2.set_xscale("log")
    ax2.set_xlabel("window SNR_max (log)")
    ax2.set_ylabel("shape-error fraction ε")
    ax2.grid(True, alpha=0.25, which="both")
    ax2.legend(fontsize=8, ncol=min(n + 1, 5), loc="upper right")
    ax2.set_title(
        "ε vs SNR — points above κ are genuine misfit " "(SNR² floor removed)"
    )

    # (3) ε vs frequency: where on the band the knob moves the misfit.
    ax3 = fig.add_subplot(gs[2, 0])
    for i, pv in enumerate(per_value):
        fc = []
        eps = []
        for w in pv:
            wid = w["window_id"]
            fr = _window_center(rows[i].result, wid)
            if fr is not None:
                fc.append(fr)
                eps.append(w["epsilon"])
        if fc:
            ax3.scatter(fc, eps, s=18, color=colors[i], alpha=0.75)
    ax3.axhline(kappa, color="crimson", ls="--", lw=1.0)
    ax3.set_xlabel("window center frequency (MHz)")
    ax3.set_ylabel("shape-error ε")
    ax3.grid(True, alpha=0.25)
    ax3.set_title(
        "ε across the band by swept value " "(which windows the knob helps / hurts)"
    )

    fig.suptitle(f"Fit-quality sweep: {spec.path}")
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    return fig


def _window_center(result: Any, window_id: int) -> Any:
    """Center frequency (MHz) of a fitted window by id, or None."""
    fit = result.get("fit") if result else None
    if fit is None:
        return None
    for wf in fit.window_fits:
        if wf.window_id == window_id and wf.window is not None:
            lo, hi = wf.window.freq_range
            return 0.5 * (float(lo) + float(hi))
    return None


def _band_spectrum(ctx: Any) -> Any:
    """``(freqs_mhz, |FT|)`` (frequency-sorted) of the persisted FT, or
    ``None``. The band-overlay panels draw this once as a backdrop for the
    per-value rescue / spur / thaw provenance.

    Deliberately the *persisted* Stage 1 FT (loaded via ``ctx.ftmw_path``), not
    the Stage 5 ``active_ft``: the active FT is an rfft of the truncated FID, so
    it spans the full 0→Nyquist RF band (the trim is applied only downstream to
    the windows / peaks). The persisted FT is the standard analysis spectrum —
    the truncated FID *and* the trimmed analysis band — which is what these
    overlays should show. Markers (window centers, spur centers, candidates) are
    absolute MHz, so they register on it regardless. Mirrors ``plot_noise_sweep``.
    """
    import numpy as np

    path = getattr(ctx, "ftmw_path", None)
    if path is None:
        return None
    try:
        import ftmwpipeline.api as ftmw  # lazy

        ft = ftmw.compute_ft(path)
    except Exception:
        return None
    freqs = np.asarray(ft.freq_array, dtype=float)
    mag = np.abs(np.asarray(ft.complex_spectrum))
    order = np.argsort(freqs)
    return freqs[order], mag[order]


# ---------------------------------------------------------------------------
# Stage 5 knob-family plots (rescue / spur / thaw). Unlike the fit-quality
# adapter these read the persisted renegotiation histories rather than the
# per-window fit quality: each family *does* a distinct thing (rescue adds
# residual lines, spur masks integer-MHz tones, thaw re-co-fits contested
# window edges), so each gets its own provenance view.
# ---------------------------------------------------------------------------

# rescue.snr_threshold default — the constant gate line drawn on the candidate
# panel when the swept knob is not snr_threshold itself.
_RESCUE_DEFAULT_SNR_GATE = 2.5
# thaw.residual_edge_threshold default — the constant trigger line on the
# coherence-handshake panel when the swept knob is not the threshold itself.
_THAW_DEFAULT_EDGE_GATE = 8.0


def _sweep_labels(rows: List[Any]) -> List[str]:
    return [
        f"{r.value:g}" if isinstance(r.value, (int, float)) else str(r.value)
        for r in rows
    ]


def plot_rescue(spec: Any, rows: List[Any], ctx: Any) -> Any:
    """Stage 5 residual-rescue sweep view (the ``stage5.rescue.*`` adapter).

    Rescue runs a per-window B-loop that nominates residual peaks, jointly
    refits, then prunes by AICc. The persisted ``rescue_history`` carries each
    round's counts (added / rescue-origin-pruned / merged), the χ² before/after,
    and the detector candidates (offset + SNR) — but not a per-peak origin flag,
    so the view is *aggregate*: three co-equal panels —
    (1) a count + χ²-reduction trend vs the swept value (does the knob add lines,
    and do they lower the misfit or just churn — watch ``rescue-origin pruned``,
    the failsafe); (2) a band-wide **where-rescue-fires** raster, one row per
    value, marking the window centers of accepted rounds over the spectrum; and
    (3) the **candidate-SNR-vs-gate** strip — every nominated candidate's SNR per
    value with the ``snr_threshold`` cut drawn, so the detection gates visibly
    bite instead of looking inert.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [r for r in rows if r.result is not None]
    if not rows:
        return None
    leaf = spec.path.split(".")[-1]
    n = len(rows)
    labels = _sweep_labels(rows)
    colors = _value_colors(n)
    xs = np.arange(n, dtype=float)

    fig = plt.figure(figsize=(12.0, 12.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.1, 1.4, 1.2])

    # (1) trend: peaks added (accepted) / rescue-origin pruned + χ² reduction.
    ax = fig.add_subplot(gs[0, 0])
    added, pruned, drop = [], [], []
    for r in rows:
        rh = r.result["fit"].rescue_history
        acc = [x for x in rh if x.accepted]
        added.append(sum(x.n_rescue_added for x in acc))
        pruned.append(sum(x.n_pruned_rescue_origin for x in rh))
        vals = [
            (x.chi2_before - x.chi2_after) / x.chi2_before
            for x in acc
            if x.chi2_before > 0
            and np.isfinite(x.chi2_before)
            and np.isfinite(x.chi2_after)
        ]
        drop.append(float(np.median(vals)) * 100.0 if vals else 0.0)
    ax.plot(xs, added, "o-", color="tab:green", label="peaks added (accepted)")
    ax.plot(xs, pruned, "x--", color="tab:red", label="rescue-origin pruned (failsafe)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel(leaf)
    ax.set_ylabel("peak count")
    ax.grid(True, alpha=0.3)
    axc = ax.twinx()
    axc.plot(xs, drop, "D-", color="tab:blue", alpha=0.55, label="median χ² drop (%)")
    axc.set_ylabel("median χ² reduction (%)", color="tab:blue")
    axc.tick_params(axis="y", labelcolor="tab:blue")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axc.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, ncol=3, loc="upper left")
    ax.set_title(f"Stage 5 rescue vs {leaf}: lines added and whether they stick")

    # (2) where-on-band raster: accepted-round window centers, one row per value.
    ax2 = fig.add_subplot(gs[1, 0])
    spec_xy = _band_spectrum(ctx)
    if spec_xy is not None:
        f, mag = spec_xy
        axb = ax2.twinx()
        axb.plot(f, mag, lw=0.4, color="0.85", zorder=0)
        axb.set_ylabel("|FT|", color="0.6")
        axb.tick_params(axis="y", labelcolor="0.6")
        # A twin axis renders above its base regardless of zorder, so lift the
        # marker axis above the spectrum and clear its (opaque) background so the
        # FT still shows through — otherwise the markers are buried under |FT|.
        ax2.set_zorder(axb.get_zorder() + 1)
        ax2.patch.set_visible(False)
    for i, r in enumerate(rows):
        fcs = [
            _window_center(r.result, x.window_id)
            for x in r.result["fit"].rescue_history
            if x.accepted
        ]
        fcs = [c for c in fcs if c is not None]
        if fcs:
            ax2.scatter(
                fcs, np.full(len(fcs), i), s=26, color=colors[i], alpha=0.9, zorder=3
            )
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(labels)
    ax2.set_ylim(-0.5, n - 0.5)
    ax2.set_ylabel(leaf)
    ax2.set_xlabel("window center frequency (MHz)")
    ax2.set_title("Where rescue adds lines across the band (one row per value)")

    # (3) candidate SNR vs the detection gate.
    ax3 = fig.add_subplot(gs[2, 0])
    for i, r in enumerate(rows):
        snrs = np.asarray(
            [c.snr for x in r.result["fit"].rescue_history for c in x.candidates],
            dtype=float,
        )
        if snrs.size:
            xpos = i + np.linspace(-0.32, 0.32, snrs.size)
            ax3.scatter(xpos, snrs, s=9, color=colors[i], alpha=0.5, zorder=2)
        if leaf == "snr_threshold" and isinstance(r.value, (int, float)):
            ax3.hlines(
                float(r.value), i - 0.42, i + 0.42, color=colors[i], lw=2.0, zorder=3
            )
    if leaf != "snr_threshold":
        ax3.axhline(
            _RESCUE_DEFAULT_SNR_GATE,
            color="crimson",
            ls="--",
            lw=1.0,
            label=f"snr_threshold = {_RESCUE_DEFAULT_SNR_GATE:g}",
        )
        ax3.legend(fontsize=8, loc="upper right")
    ax3.set_xticks(range(n))
    ax3.set_xticklabels(labels)
    ax3.set_xlabel(leaf)
    ax3.set_ylabel("residual candidate SNR")
    ax3.grid(True, alpha=0.25)
    ax3.set_title(
        "Residual candidates nominated vs the detection gate "
        "(per-value gate in color when sweeping snr_threshold)"
    )

    fig.suptitle(f"Rescue sweep: {spec.path}")
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    return fig


def plot_spur(spec: Any, rows: List[Any], ctx: Any) -> Any:
    """Stage 5 spur-masking sweep view (the ``stage5.spur.*`` adapter).

    A spur is a persistent integer-MHz CW tone (clock/LO harmonic) no finite-T
    line shape can represent; the gate drops it from nomination and masks its
    bins out of the residual. The persisted band-level catalog
    (``spur_centers_mhz`` / ``spur_sources`` / ``spur_mask_half_width_bins``) is
    computed on the whole active FT, so spur sweeps are immune to the fit's plan
    reduction. Two panels: (1) a count-by-source trend (total / narrow /
    saturated gated vs the swept value); and (2) the **spectrum overlay** —
    ``|FT|`` drawn once with every gated spur as a vertical marker at its
    integer-MHz, its ±mask half-width shaded, colored by *how many* swept values
    gate it (a robustness ramp: a tone gated at every value is an unambiguous
    spur; one only the most permissive value catches is fragile). Robustness, not
    a last-surviving value, because spur gating is not monotonic in one direction
    across the different spur knobs — looser ``narrowness_ratio`` adds spurs while
    a higher ``snr_threshold`` removes them, so a count is the direction-agnostic
    readout.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [r for r in rows if r.result is not None]
    if not rows:
        return None
    leaf = spec.path.split(".")[-1]
    n = len(rows)
    labels = _sweep_labels(rows)

    def _params(r: Any) -> Any:
        return r.result["fit"].parameters

    per_value_centers = [
        [float(c) for c in (_params(r).get("spur_centers_mhz") or [])] for r in rows
    ]
    # Key spurs by nearest integer MHz (the gate's anchor); a spur is "gated at
    # value i" when an integer-equal center is in that value's catalog. Color
    # by the *number* of values that gate it (robustness) rather than a
    # last-surviving value: spur gating is not monotonic in one direction across
    # the spur knobs (looser narrowness_ratio adds spurs, higher snr_threshold
    # removes them), so a directional survival ramp would collapse to one color.
    keyed = [{round(c): c for c in cs} for cs in per_value_centers]
    all_keys = sorted({k for d in keyed for k in d})
    gate_count = {}
    rep_center = {}
    for k in all_keys:
        idxs = [i for i, d in enumerate(keyed) if k in d]
        gate_count[k] = len(idxs)
        rep_center[k] = keyed[idxs[0]][k] if idxs else float(k)

    fig = plt.figure(figsize=(12.0, 10.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.5])

    # (1) count-by-source trend.
    ax = fig.add_subplot(gs[0, 0])
    xs = np.arange(n, dtype=float)
    n_tot, n_narrow, n_sat = [], [], []
    for r in rows:
        srcs = list(_params(r).get("spur_sources") or [])
        n_tot.append(len(srcs))
        n_narrow.append(sum(1 for s in srcs if "narrow" in s))
        n_sat.append(sum(1 for s in srcs if "saturated" in s))
    ax.plot(xs, n_tot, "o-", color="0.2", label="gated total")
    ax.plot(xs, n_narrow, "s--", color="tab:blue", label="narrow")
    ax.plot(xs, n_sat, "^--", color="tab:orange", label="saturated")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel(leaf)
    ax.set_ylabel("spurs gated")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Stage 5 spur gating vs {leaf}")

    # (2) spectrum overlay with the survival ramp.
    ax2 = fig.add_subplot(gs[1, 0])
    spec_xy = _band_spectrum(ctx)
    bin_mhz = 0.0
    if spec_xy is not None:
        f, mag = spec_xy
        ax2.plot(f, mag, lw=0.5, color="0.6", zorder=1)
        if f.size > 1:
            bin_mhz = float(np.median(np.diff(f)))
    hw_bins = max(
        (int(_params(r).get("spur_mask_half_width_bins", 0) or 0) for r in rows),
        default=0,
    )
    hw_mhz = hw_bins * bin_mhz
    # Robustness ramp keyed on the gating count 1..n (dark = fragile, bright =
    # gated at every value); _value_colors(n) gives n distinct steps.
    ramp = _value_colors(n)
    for k in all_keys:
        c = rep_center[k]
        col = ramp[min(gate_count[k], n) - 1] if gate_count[k] else ramp[0]
        ax2.axvline(c, color=col, lw=1.2, zorder=3)
        if hw_mhz > 0:
            ax2.axvspan(c - hw_mhz, c + hw_mhz, color=col, alpha=0.15, zorder=2)
    ax2.set_xlabel("frequency (MHz)")
    ax2.set_ylabel("|FT|")
    ax2.set_title(
        "Gated spurs, colored by how many values gate them "
        f"(robustness; ±{hw_bins} bin mask shaded)"
    )
    handles = [plt.Line2D([0], [0], color=ramp[i], lw=2.5) for i in range(n)]
    ax2.legend(
        handles,
        [f"{i + 1}/{n} values" for i in range(n)],
        fontsize=7,
        ncol=min(n, 5),
        title="gated by",
        loc="upper right",
    )

    fig.suptitle(f"Spur sweep: {spec.path}")
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    return fig


def plot_thaw(spec: Any, rows: List[Any], ctx: Any) -> Any:
    """Stage 5 thaw / structural-replan sweep view (the ``stage5.thaw.*``
    adapter).

    A post-fit residual edge that stays coherent triggers the renegotiation
    handshake: unfreeze the neighboring fixed contributor and co-fit it (thaw),
    or, with no contributor to thaw, merge the adjacent window and re-plan
    (replan). The persisted ``thaw_history`` / ``replan_history`` record where
    each edge was flagged, the edge ``S_coh`` before and after, and accept/reject
    — but *not* literal original-vs-final window boundaries, so the view is the
    **coherence handshake**, not boundary geometry. Three panels:
    (1) an attempt / accept count + plan-revision trend vs the swept value;
    (2) a band-wide **contested-edge raster**, one row per value (thaw at the
    contributor frequency, replan at the surviving-window center; filled =
    accepted); and (3) the **before→after coherence scatter** with the trigger
    threshold drawn — points on the diagonal are edges the handshake left
    unchanged (the common case: thaw is near-dormant on clean spectra).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    rows = [r for r in rows if r.result is not None]
    if not rows:
        return None
    leaf = spec.path.split(".")[-1]
    n = len(rows)
    labels = _sweep_labels(rows)
    colors = _value_colors(n)
    xs = np.arange(n, dtype=float)

    fig = plt.figure(figsize=(12.0, 12.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.1, 1.3, 1.3])

    # (1) attempt / accept counts + plan revision.
    ax = fig.add_subplot(gs[0, 0])
    n_th, n_th_a, n_rp, n_rp_a, rev = [], [], [], [], []
    for r in rows:
        fit = r.result["fit"]
        n_th.append(len(fit.thaw_history))
        n_th_a.append(sum(1 for t in fit.thaw_history if t.accepted))
        n_rp.append(len(fit.replan_history))
        n_rp_a.append(sum(1 for x in fit.replan_history if x.accepted))
        rev.append(int(fit.final_plan_revision))
    ax.plot(xs, n_th, "o-", color="tab:blue", label="thaw attempts")
    ax.plot(xs, n_th_a, "o-", color="tab:green", label="thaw accepted")
    ax.plot(xs, n_rp, "s--", color="tab:purple", label="replan attempts")
    ax.plot(xs, n_rp_a, "s--", color="tab:red", label="replan accepted")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel(leaf)
    ax.set_ylabel("event count")
    ax.grid(True, alpha=0.3)
    axc = ax.twinx()
    axc.plot(xs, rev, "D:", color="0.4", label="final plan revision")
    axc.set_ylabel("plan revision", color="0.4")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axc.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, ncol=3, loc="upper left")
    ax.set_title(f"Stage 5 thaw / replan handshake vs {leaf}")

    # (2) contested-edge raster.
    ax2 = fig.add_subplot(gs[1, 0])
    spec_xy = _band_spectrum(ctx)
    if spec_xy is not None:
        f, mag = spec_xy
        axb = ax2.twinx()
        axb.plot(f, mag, lw=0.4, color="0.85", zorder=0)
        axb.set_ylabel("|FT|", color="0.6")
        axb.tick_params(axis="y", labelcolor="0.6")
        # Lift the marker axis above the spectrum twin (see plot_rescue).
        ax2.set_zorder(axb.get_zorder() + 1)
        ax2.patch.set_visible(False)
    for i, r in enumerate(rows):
        fit = r.result["fit"]
        for t in fit.thaw_history:
            fc = t.contributor_frequency_mhz
            if fc is None or not np.isfinite(fc):
                continue
            ax2.scatter(
                [fc],
                [i],
                s=30,
                marker="o",
                zorder=3,
                facecolors=colors[i] if t.accepted else "none",
                edgecolors=colors[i],
            )
        for x in fit.replan_history:
            c = _window_center(r.result, x.surviving_window_id) or _window_center(
                r.result, x.triggering_window_id
            )
            if c is not None:
                ax2.scatter(
                    [c],
                    [i],
                    s=48,
                    marker="^",
                    zorder=3,
                    facecolors=colors[i] if x.accepted else "none",
                    edgecolors=colors[i],
                )
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(labels)
    ax2.set_ylim(-0.5, n - 0.5)
    ax2.set_ylabel(leaf)
    ax2.set_xlabel("contested edge frequency (MHz)")
    ax2.set_title("Where edges are contested " "(○ thaw, △ replan; filled = accepted)")

    # (3) before→after coherence handshake.
    ax3 = fig.add_subplot(gs[2, 0])
    finite_b: List[float] = []
    for i, r in enumerate(rows):
        before, after = [], []
        for t in r.result["fit"].thaw_history:
            b, a = t.edge_coherence_before, t.edge_coherence_after
            if np.isfinite(b):
                before.append(float(b))
                after.append(float(a) if np.isfinite(a) else float(b))
        if before:
            finite_b.extend(before)
            ax3.scatter(
                before,
                after,
                s=22,
                color=colors[i],
                alpha=0.75,
                label=f"{leaf}={labels[i]}",
            )
    if finite_b:
        lo = min(finite_b + [_THAW_DEFAULT_EDGE_GATE])
        hi = max(finite_b + [_THAW_DEFAULT_EDGE_GATE])
        ax3.plot(
            [lo, hi],
            [lo, hi],
            color="0.5",
            ls=":",
            lw=1.0,
            label="no change (after = before)",
        )
    if leaf == "residual_edge_threshold":
        for i, r in enumerate(rows):
            if isinstance(r.value, (int, float)):
                ax3.axvline(float(r.value), color=colors[i], ls="--", lw=0.8)
    else:
        ax3.axvline(
            _THAW_DEFAULT_EDGE_GATE,
            color="crimson",
            ls="--",
            lw=1.0,
            label=f"trigger S_coh = {_THAW_DEFAULT_EDGE_GATE:g}",
        )
    ax3.set_xlabel("edge S_coh before co-fit")
    ax3.set_ylabel("edge S_coh after co-fit")
    ax3.grid(True, alpha=0.25)
    ax3.legend(fontsize=8, loc="upper left")
    ax3.set_title(
        "Coherence handshake — points on the diagonal were left "
        "unchanged (no improvement)"
    )

    fig.suptitle(f"Thaw sweep: {spec.path}")
    fig.tight_layout(rect=(0, 0, 1, 0.99))
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
    import matplotlib.pyplot as plt
    import numpy as np

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
            np.arange(sigma.size),
            sigma,
            color=color,
            alpha=0.8,
            lw=1.0,
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
        mag = np.abs(np.asarray(ft.complex_spectrum))
        ax3.plot(f_ghz, mag, lw=0.4, color="0.6", label="|FT|", zorder=1)
        sigma_max = 0.0
        for row, color in zip(rows, colors):
            sigma = np.asarray(row.result.rms_noise, dtype=float)
            if sigma.size != f_ghz.size:
                continue
            ax3.plot(
                f_ghz,
                sigma,
                color=color,
                lw=1.3,
                zorder=2,
                label=f"σ: {leaf}={row.value:g}",
            )
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
