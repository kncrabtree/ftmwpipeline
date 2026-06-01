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


def _value_colors(n: int) -> List[Any]:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("viridis")
    if n <= 1:
        return [cmap(0.5)]
    return [cmap(i / (n - 1)) for i in range(n)]


def plot_start_detection(spec: Any, rows: List[Any], ctx: Any) -> Any:
    """Two panels: the FID with the chirp-end and each value's recommended
    start marked (top), and the Σ|FT|-vs-start detection sweep (bottom).

    The top panel shows where the knob lands the start on the actual transient;
    the bottom shows the detector's collapse curve. For the effect of ``start_us``
    on the resulting spectrum, see the ``stage1.start_us`` knob.
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
        chirp = rows[0].result.chirp_end_us
        axf.axvline(chirp, color="k", ls=":", lw=1.3,
                    label=f"chirp-end {chirp:.2f} us")
        for row, color in zip(rows, colors):
            axf.axvline(
                row.result.start_us, color=color, ls="--", alpha=0.85,
                label=f"{leaf}={row.value:g}: start {row.result.start_us:.2f} us",
            )
        xmax = max(r.result.start_us for r in rows) + 1.0
        axf.set_xlim(0.0, xmax)
        axf.set_xlabel("time (us)")
        axf.set_ylabel("FID amplitude")
        axf.set_title("FID with chirp-end and recommended starts")
        axf.legend(fontsize=7)
    except Exception:
        # FID unavailable — keep the detection panel useful on its own.
        axf.set_visible(False)

    # Bottom: the Σ|FT|-vs-start sweep with each value's start marked.
    for row, color in zip(rows, colors):
        r = row.result
        axs.semilogy(r.starts_us, r.sum_magnitude, color=color, alpha=0.8, lw=1.2,
                     label=f"{leaf}={row.value:g}")
        axs.axvline(r.start_us, color=color, ls="--", alpha=0.7)
    axs.axvline(rows[0].result.chirp_end_us, color="k", ls=":", alpha=0.6,
                label=f"chirp-end ~{rows[0].result.chirp_end_us:.2f} us")
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


def plot_tau_trend(spec: Any, rows: List[Any], ctx: Any) -> Any:
    """tau_maj +/- sigma_tau vs the knob value, with the contributor count on a
    twin axis. Returns ``None`` for non-numeric knobs (table-only)."""
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

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
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
    fig.tight_layout()
    return fig


def plot_noise_sweep(spec: Any, rows: List[Any], ctx: Any) -> Any:
    """Two panels: σ(f) per grid value (left) and the scalar metric trend
    (right, median σ and noise-flagged fraction vs the knob)."""
    import numpy as np
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r.result is not None]
    if not rows:
        return None

    leaf = spec.path.split(".")[-1]
    colors = _value_colors(len(rows))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.8))

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
    fig.suptitle(f"Noise sweep: {spec.path}")
    fig.tight_layout()
    return fig
