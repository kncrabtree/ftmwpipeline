"""HTML report (``report run``): a local HTML view over the persisted record.

Reports render the persisted Stage 6 record; they never recompute the fit. The
report is an **assembler** over existing artifacts and renderers, not new
analysis: it reuses the summary model for the index, the ``fit show`` figure
renderer for the per-window plots, the candidate ledger, and the persisted
Stage 6 review status / decision log.

The output is one self-contained ``.html`` file. It is assembled internally as a
set of linked pages (an index, a methods-and-results page, and a page per fit
window) with their figures, then folded into a single document with the
stylesheet inlined and every figure base64-embedded
(:func:`_collapse_site_to_single_file`); cross-page links become in-document
anchors. The markup is dependency-light (hand-rolled HTML + matplotlib PNGs),
with the presentation isolated in one stylesheet (``_STYLESHEET``).
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np

from ..core.data_structures import (
    DecisionLogEntry,
    FinalPeak,
    FinalProducts,
    LedgerCandidate,
    WindowReviewStatus,
)
from ..fitting.peak_model import sideband_sign as _sideband_sign
from ..io.stage6_review_serialization import load_stage6_review_from_file
from ..utils.parallelism import resolve_worker_count
from .catalog_xref import CatalogCrossRef, CatalogMatch, load_cross_ref
from .report_impl import (
    _CAL_STATE_PHRASE,
    _amplitude_unit,
    _concise,
    _freq,
    _g,
    _md_num,
    _scaled,
    assemble_summary_model,
    report_table_impl,
)

# Per-window render progress is emitted as a ``"window %d/%d"`` INFO log; a
# ``StageProgress`` capture (the ``run --report`` orchestrator, or the standalone
# ``report run`` command) renders it as a percentage. See ``_internal/progress``.
logger = logging.getLogger(__name__)

# File extension per Level-1 table format (the ``report run`` table artifact).
_TABLE_EXT = {"csv": "csv", "json": "json", "latex": "tex"}

VALID_WINDOW_FILTERS = ("all", "attention")

# Above this many covariance parameters the matrix is summarized rather than
# rendered inline (a wide window's peak-major matrix is unreadable as a table;
# the full matrix stays available via the API / Stage 5 record). The compact
# parameter symbols keep a wider matrix legible than the old word-labels did.
_COVARIANCE_RENDER_CAP = 36


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------


def _esc(value: object) -> str:
    """HTML-escape any value's string form."""
    return html.escape("" if value is None else str(value))


def _table(
    headers: List[str],
    rows: List[List[str]],
    *,
    cls: str = "",
    row_attrs: Optional[List[str]] = None,
) -> str:
    """A simple HTML table; cell contents are emitted verbatim (pre-escaped).

    ``row_attrs`` optionally supplies a pre-formatted attribute string per row
    (parallel to *rows*, ``""`` for none) injected into the ``<tr>`` -- used to
    hang ``data-thumb`` / ``data-info`` on rows for the hover-preview popup.
    """
    cls_attr = f' class="{cls}"' if cls else ""
    out = [f"<table{cls_attr}>", "  <thead>", "    <tr>"]
    out += [f"      <th>{h}</th>" for h in headers]
    out += ["    </tr>", "  </thead>", "  <tbody>"]
    for i, r in enumerate(rows):
        attr = row_attrs[i] if row_attrs is not None else ""
        out.append(f"    <tr{attr}>")
        out += [f"      <td>{c}</td>" for c in r]
        out.append("    </tr>")
    out += ["  </tbody>", "</table>"]
    return "\n".join(out)


def _page(title: str, body: List[str], *, css_href: str, head_extra: str = "") -> str:
    """Wrap a body fragment list in a minimal HTML document.

    ``head_extra`` injects extra ``<head>`` markup (e.g. the MathJax loader on
    the methods page); empty for the plain pages.
    """
    head = [
        "<head>",
        '  <meta charset="utf-8">',
        '  <meta name="viewport" content="width=device-width, ' 'initial-scale=1">',
        f"  <title>{_esc(title)}</title>",
        f'  <link rel="stylesheet" href="{css_href}">',
    ]
    if head_extra:
        head.append(head_extra)
    head.append("</head>")
    return "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="en">',
            *head,
            "<body>",
            '<main class="report">',
            *body,
            "</main>",
            "</body>",
            "</html>",
            "",
        ]
    )


# MathJax (display math) for the methods page. Loaded from a CDN; when offline
# the raw ``\[...\]`` TeX source stays visible in the styled equation block, so
# the page degrades gracefully rather than breaking.
_MATHJAX_HEAD = (
    "  <script>window.MathJax={tex:{displayMath:[['$$','$$'],"
    "['\\\\[','\\\\]']]},options:{skipHtmlTags:['script','noscript','style',"
    "'textarea','pre','code']}};</script>\n"
    '  <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/'
    'tex-svg.js"></script>'
)


# The single stylesheet -- the presentation lives here, isolated from the HTML
# structure so the two can change independently.
_STYLESHEET = """/* ftmwpipeline report styling (report run).
   All presentation lives here; the HTML structure carries none. */
body { font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
       margin: 0; color: #1a1a1a; background: #f6f7f9; }
main.report { max-width: 1500px; margin: 0 auto; padding: 1.5rem 2rem 4rem; }
h1, h2, h3 { line-height: 1.2; }
a { color: #1559b3; }
table { border-collapse: collapse; margin: 0.5rem 0 1.25rem; font-size: 0.9rem; }
th, td { border: 1px solid #d0d4d9; padding: 0.25rem 0.6rem; text-align: right; }
th { background: #eceff2; }
td:first-child, th:first-child { text-align: left; }
/* Long data tables fill the column width (better use of horizontal space) and
   stay readable when they run to hundreds of rows: zebra rows, a row hover, and
   a header that sticks while scrolling. The compact methods-page parameter /
   percentile tables keep their content width. */
table.window-list, table.final-list, table.peak-list, table.audit,
table.ledger, table.covariance { width: 100%; }
tbody tr:nth-child(even) { background: #f2f4f7; }
tbody tr:hover { background: #e6edf4; }
/* Rows wired to the hover-preview popup (window list / final line list): a
   help cursor hints that hovering shows the window's magnitude thumbnail. */
tbody tr[data-thumb]:hover { cursor: help; }
thead th { position: sticky; top: 0; z-index: 1; }
pre { background: #11151a; color: #e6e6e6; padding: 0.75rem 1rem;
      overflow-x: auto; border-radius: 4px; font-size: 0.82rem; }
/* A sticky top navigation bar spans the viewport; its inner row is centered to
   the report column. Anchor jumps and the table sticky-headers are offset by the
   bar height so nothing lands hidden underneath it. */
.topnav { position: sticky; top: 0; z-index: 100; background: #11233a;
          box-shadow: 0 1px 4px rgba(0,0,0,0.25); }
.topnav-inner { max-width: 1500px; margin: 0 auto; padding: 0.5rem 2rem;
                display: flex; align-items: center; gap: 1.1rem; flex-wrap: wrap;
                font-size: 0.9rem; }
.topnav a { color: #cfe0f5; text-decoration: none; }
.topnav a:hover { color: #fff; text-decoration: underline; }
.topnav .brand { font-weight: 600; color: #fff; margin-right: auto; }
.topnav select { font-size: 0.85rem; padding: 0.1rem 0.3rem; border-radius: 3px;
                 border: 1px solid #2c4a6e; background: #fff; color: #1a1a1a; }
html.report-single { scroll-padding-top: 3.4rem; }
html.report-single thead th { top: 3.4rem; }
/* The major blocks (index, methods, each embedded window) stack as <section>
   children of main; space them apart and rule a divider between consecutive
   sections so the report reads as distinct blocks rather than one
   undifferentiated scroll. */
main.report > section { margin-bottom: 2.75rem; }
main.report > section + section { border-top: 2px solid #c4ccd4;
                                  padding-top: 2.25rem; }
section.embedded-window { margin-top: 0.5rem; }
/* Compact mode (single-file "Compact" topnav toggle): cap every report figure's
   height (width follows the aspect ratio, so a wide panel like the spectrum
   overview stays legible as a short strip rather than a tiny square) so a full
   all-windows report scrolls fast; clicking a figure expands just that one back
   to full size, and the chatty hint lines are hidden. Pure CSS over the
   already-embedded figures -- no extra payload -- and gated on a class the toggle
   script sets, so with scripting off every figure stays full size and the toggle
   is inert. */
.topnav .compact-toggle { font-size: 0.85rem; padding: 0.12rem 0.6rem;
    border-radius: 3px; border: 1px solid #2c4a6e; background: #cfe0f5;
    color: #11233a; cursor: pointer; }
.topnav .compact-toggle:hover { background: #fff; }
html.report-compact .fit-panels, html.report-compact .panel-grid {
    display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: flex-start; }
html.report-compact .winmap-hint { display: none; }
html.report-compact .fit-panels img,
html.report-compact .cov-heatmap img,
html.report-compact .hist img,
html.report-compact .maghist img {
    height: 130px; width: auto; max-width: 100%; cursor: zoom-in; }
html.report-compact img.zoom-expanded {
    height: auto; width: 100%; max-width: 100%; cursor: zoom-out; }
/* The interactive overview stays usable in compact: cap its height (the div
   hugs the shrunk SVG so the background image stays aligned), click to expand. */
html.report-compact .spectrum-ctx { width: fit-content; max-width: 100%;
                                    cursor: zoom-in; }
html.report-compact .spectrum-ctx-svg { height: 150px; width: auto;
                                        max-width: 100%; }
html.report-compact .spectrum-ctx.zoom-expanded { width: auto; cursor: zoom-out; }
html.report-compact .spectrum-ctx.zoom-expanded .spectrum-ctx-svg {
    height: auto; width: 100%; }
.badge { display: inline-block; padding: 0.05rem 0.45rem; border-radius: 3px;
         font-size: 0.78rem; background: #f0d9a8; color: #5a4300;
         margin-left: 0.35rem; }
/* Catalog proximity-match badge (a cross-check echo, not an assignment). */
.badge.cat { background: #cfe8d2; color: #1d5026; margin-left: 0; cursor: help; }
/* Clock-lattice badge (a line on the declared clock lattice -- a candidate
   instrumental artifact flagged for review, never auto-removed). */
.badge.lattice { background: #f3c9bd; color: #7a2810; margin-left: 0; cursor: help; }
.nav { margin: 1rem 0; }
/* The summary key-value list fills the row as a responsive multi-column grid
   (side-by-side on a wide screen) and collapses to a single column when narrow. */
ul.summary { list-style: none; padding: 0; display: grid; gap: 0.15rem 1.75rem;
             grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); }
.summary li { margin: 0.15rem 0; }
/* Fit panels: the overview spans full width on top; the Re/Im model+residual
   panels share a row and the |X| + residual histogram share the row below
   (a 1:2:2 stack that gives the model+residual panels the most space). On a
   narrow viewport the grid collapses to a single column, so Re, Im, and |X|
   each get the full width in turn. Each panel is its own PNG. */
.fit-panels { margin: 0.5rem 0 1.25rem; }
.fit-panels figure { margin: 0; }
.fit-panels img { width: 100%; height: auto; border: 1px solid #d0d4d9;
                  background: #fff; }
.panel-overview { margin-bottom: 0.9rem; }
.panel-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.9rem;
              align-items: start; }
@media (max-width: 900px) { .panel-grid { grid-template-columns: 1fr; } }
/* Covariance: a small variances table, the correlation heatmap, then the full
   numeric matrix below (the heatmap stays readable where the matrix does not). */
.cov-heatmap { margin: 0.5rem 0 1rem; }
.cov-heatmap img { max-width: 640px; width: 100%; height: auto;
                   border: 1px solid #d0d4d9; background: #fff; }
table.covariance { font-size: 0.78rem; }
/* The add-one-peak history: keep the free-text reason left-aligned. */
table.audit { font-size: 0.82rem; }
table.audit td:last-child, table.audit th:last-child { text-align: left; }
.cov-legend, .audit-legend { font-size: 0.82rem; color: #444;
                             margin: 0.25rem 0 0.75rem; }
/* Level-2 methods page: equation blocks (MathJax-rendered, raw TeX fallback)
   and the distributions figure. */
.equation { background: #f0f2f5; color: #1a1a1a; border: 1px solid #d0d4d9;
            border-radius: 4px; padding: 0.6rem 1rem; margin: 0.5rem 0 1.25rem;
            overflow-x: auto; font-size: 0.9rem; text-align: center; }
/* The interleaved distribution figures cap their natural width and center,
   rather than stretching a 1-2 panel group across the full column. */
.hist { margin: 0.5rem 0 1.25rem; }
.hist img { max-width: 100%; width: auto; height: auto; border: 1px solid #d0d4d9;
            background: #fff; }
/* Magnitude-distribution panels: a reflowing flex row -- each panel keeps a
   readable minimum width, panels share the row on a wide screen and stack on a
   narrow one. */
.maghist-grid { display: flex; flex-wrap: wrap; gap: 0.75rem; margin: 0.5rem 0 1.25rem; }
.maghist { flex: 1 1 320px; max-width: 480px; margin: 0; }
.maghist img { width: 100%; height: auto; border: 1px solid #d0d4d9; background: #fff; }
/* A short caption above an interleaved figure or figure group. */
.fig-note { font-size: 0.82rem; color: #555; margin: 0.5rem 0 0.2rem; }
/* Markdown *emphasis* (the "Results:" labels) -- styleable; bold for now. */
.md-em { font-weight: 600; }
/* Interactive full-spectrum overview: ONE shared full-spectrum image (rendered
   once, applied as the div's background by a per-build rule) carries an SVG
   overlay of one clickable rect per window -- so the image doubles as the
   quick-nav and is rendered/embedded once, not re-rendered per window. The
   overlay viewBox tracks the overview figure's axes box and stretches to the div
   (preserveAspectRatio="none"), so the rects stay aligned to the frequency axis
   the image draws. Rects are transparent hotspots (the image shows the spectrum
   and its attention shading); attention windows get a faint tint, hover a blue
   wash, and the current window -- on its own page -- a green "you are here". */
.spectrum-ctx { margin: 0.5rem 0 0.25rem; border: 1px solid #d0d4d9;
                background: #fff center / 100% 100% no-repeat; line-height: 0; }
.spectrum-ctx-svg { display: block; width: 100%; height: auto; }
.specnav-rect { fill: #1559b3; fill-opacity: 0; pointer-events: all;
                cursor: pointer; }
.specnav-rect.attn { fill: #f4a23b; fill-opacity: 0.2; }
.spectrum-ctx a:hover .specnav-rect, .specnav-rect:hover { fill: #1559b3;
                                                           fill-opacity: 0.28; }
.specnav-rect.current { fill: #2ca02c; fill-opacity: 0.42; }
.winmap-hint { font-size: 0.8rem; color: #666; margin: 0 0 1.25rem; }
.winmap-pop { position: fixed; z-index: 60; pointer-events: none; background: #fff;
              border: 1px solid #888; box-shadow: 0 2px 10px rgba(0,0,0,0.25);
              padding: 4px; border-radius: 4px; }
.winmap-pop img { display: block; width: 360px; height: auto; }
.winmap-pop-info { font-size: 0.76rem; color: #222; padding: 0.15rem 0.1rem 0; }
/* Narrow viewport: shrink the gutters and table type so nothing overflows. */
@media (max-width: 700px) {
  main.report { padding: 1rem 1rem 3rem; }
  table { font-size: 0.82rem; }
}
/* In-report curation. Every curation affordance is hidden until the `Curate`
   toggle adds `curation-enabled` to <html>, so the no-JS document and the
   default read-only view show the same clean report. A single class governs all
   of it: the per-row Curate column (always the trailing cell of the
   fitted-lines / ledger tables), the `.cur-only` per-window control strips, and
   the docked cart. The topnav toggle/badge stay visible (like the compact
   toggle) so the reader can flip into editing. */
html:not(.curation-enabled) .cur-only { display: none; }
html:not(.curation-enabled) #cur-cart { display: none; }
html:not(.curation-enabled) table.peak-list td:last-child,
html:not(.curation-enabled) table.peak-list th:last-child,
html:not(.curation-enabled) table.ledger td:last-child,
html:not(.curation-enabled) table.ledger th:last-child { display: none; }
.cur-cell { display: inline-flex; align-items: center; gap: 0.3rem;
            white-space: nowrap; }
.cur-cell .cur-btn, .cur-window-controls .cur-btn {
    font-size: 0.78rem; padding: 0.08rem 0.45rem; border-radius: 3px;
    border: 1px solid #2c4a6e; background: #cfe0f5; color: #11233a;
    cursor: pointer; }
.cur-cell .cur-btn:hover, .cur-window-controls .cur-btn:hover { background: #fff; }
.cur-cell .cur-k { width: 3rem; font-size: 0.78rem; }
.cur-mergebox { font-size: 0.78rem; color: #444; }
.cur-splitbadge { display: inline-block; padding: 0.02rem 0.35rem;
                  border-radius: 3px; background: #f0d9a8; color: #5a4300;
                  font-size: 0.74rem; }
/* Pending-edit feedback on the originating fitted-line rows. */
tr.cur-removed > td { text-decoration: line-through; opacity: 0.5; }
tr.cur-merge-grp > td { background: #fdeccb !important; }
tr.cur-added > td { background: #d8efdc !important; }
.cur-window-controls { display: flex; flex-wrap: wrap; align-items: center;
    gap: 0.6rem 1rem; margin: 0.25rem 0 1.25rem; font-size: 0.85rem; }
.cur-window-controls .cur-addfreq { width: 8rem; font-size: 0.82rem; }
.cur-range { color: #666; }
/* The docked cart: fixed bottom-right, collapsible, grouped per window. */
#cur-cart { position: fixed; right: 1rem; bottom: 1rem; z-index: 150;
    width: 340px; max-width: calc(100vw - 2rem); max-height: 60vh;
    display: flex; flex-direction: column; background: #fff;
    border: 1px solid #11233a; border-radius: 6px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.28); font-size: 0.85rem; }
.cur-cart-head { background: #11233a; border-radius: 6px 6px 0 0; }
.cur-cart-head .cur-min { width: 100%; text-align: left; background: none;
    border: 0; color: #fff; font-size: 0.9rem; font-weight: 600;
    padding: 0.45rem 0.7rem; cursor: pointer; }
.cur-cart-body { overflow-y: auto; padding: 0.4rem 0.7rem; }
#cur-cart.cur-collapsed .cur-cart-body,
#cur-cart.cur-collapsed .cur-cart-foot { display: none; }
.cur-grp-h { font-weight: 600; color: #11233a; margin: 0.35rem 0 0.1rem; }
.cur-entry { display: flex; justify-content: space-between; align-items: center;
    gap: 0.5rem; padding: 0.05rem 0 0.05rem 0.6rem; }
.cur-x { background: none; border: 0; color: #a11; cursor: pointer;
         font-size: 0.85rem; }
.cur-cart-foot { border-top: 1px solid #d0d4d9; padding: 0.5rem 0.7rem;
    display: flex; flex-wrap: wrap; gap: 0.35rem; }
.cur-cart-foot button { font-size: 0.8rem; padding: 0.15rem 0.5rem;
    border-radius: 3px; border: 1px solid #2c4a6e; background: #cfe0f5;
    color: #11233a; cursor: pointer; }
.cur-cart-foot button:hover { background: #eef4fb; }
.cur-cmd { width: 100%; margin: 0.4rem 0 0; white-space: pre-wrap;
           font-size: 0.72rem; padding: 0.4rem 0.5rem; }
.cur-ta { position: absolute; left: -9999px; width: 1px; height: 1px; }
.topnav .cur-toggle { font-size: 0.85rem; padding: 0.12rem 0.6rem;
    border-radius: 3px; border: 1px solid #2c4a6e; background: #cfe0f5;
    color: #11233a; cursor: pointer; }
.topnav .cur-toggle:hover { background: #fff; }
.topnav .cur-badge { font-size: 0.82rem; color: #cfe0f5; }
/* Click-on-plot: the |X| panel is wrapped so an SVG marker layer and a corner
   arm toggle sit over it. Click-to-add fires only while the panel is armed (so a
   tap meant to scroll or zoom never adds); the armed panel takes a crosshair.
   The SVG is click-transparent except the marker handles, so a plot click still
   reaches the image. Arm/markers are hidden outside curation and in compact. */
.cur-plot-wrap { position: relative; display: block; }
.cur-plot-svg { position: absolute; inset: 0; width: 100%; height: 100%;
                pointer-events: none; }
.cur-plot-svg .cur-marker { pointer-events: all; cursor: pointer; }
.cur-plot-arm { position: absolute; top: 8px; right: 8px; z-index: 2;
    font-size: 0.85rem; font-weight: 600; padding: 0.3rem 0.75rem;
    border-radius: 4px; border: 1px solid #2c4a6e;
    background: rgba(255,255,255,0.92); color: #11233a; cursor: pointer;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2); }
.cur-plot-wrap.cur-armed .cur-plot-arm { background: #1e8e3e; color: #fff;
    border-color: #14622a; }
.cur-plot-wrap.cur-armed img.cur-plot { cursor: crosshair; }
/* On a hover-capable pointer (desktop), keep the arm out of the way until you
   hover the panel, focus into it (keyboard), or it is armed. Touch devices
   report `hover: none`, so this guard never applies there and the button stays
   visible -- the only way to arm without a hover. */
@media (hover: hover) {
  .cur-plot-arm { opacity: 0; pointer-events: none;
                  transition: opacity 0.12s ease; }
  .cur-plot-wrap:hover .cur-plot-arm,
  .cur-plot-wrap:focus-within .cur-plot-arm,
  .cur-plot-wrap.cur-armed .cur-plot-arm { opacity: 1; pointer-events: auto; }
}
html.report-compact .cur-plot-arm, html.report-compact .cur-plot-svg {
    display: none; }
"""


# ---------------------------------------------------------------------------
# Window-name helpers
# ---------------------------------------------------------------------------


def _window_page_name(window_id: int) -> str:
    return f"window_{window_id:03d}.html"


# The per-window detail is rendered as separate panel images (overview / re /
# im / mag / hist) so the page can lay them out in a flexbox.
_PANEL_ORDER = ("overview", "re", "im", "mag", "hist")


def _panel_figure_name(stem: str, window_id: int, panel: str) -> str:
    return f"{stem}_window_{window_id:03d}_{panel}.png"


def _figure_png_bytes(fig: Any, *, dpi: int, **kw: Any) -> bytes:
    """Render a report figure to adaptive 256-color palette PNG *bytes*.

    The report's figures are line plots and small heatmaps with only a few
    hundred distinct colors, so an adaptive 256-color palette is visually
    indistinguishable from the RGBA original while cutting the PNG -- and its
    base64 embed in the single-file build -- by roughly two thirds. (Lowering the
    resolution or making the background transparent does *not* help: the panels
    already render below 800 px wide, downscaling re-introduces intermediate
    colors, and a transparent background only adds alpha variation at the
    anti-aliased edges.) Renders to a buffer, then quantizes. ``kw`` is forwarded
    to ``savefig`` (e.g. ``bbox_inches="tight"``). The encoding is sink-agnostic,
    so the parallel figure-render path (worker returns bytes) and the serial path
    (write to file) are byte-identical.
    """
    import io

    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, **kw)
    buf.seek(0)
    out = io.BytesIO()
    with Image.open(buf) as im:
        palette = im.convert("RGB").quantize(
            colors=256, method=Image.Quantize.FASTOCTREE
        )
        palette.save(out, format="PNG", optimize=True)
    return out.getvalue()


def _save_figure_png(fig: Any, path: Union[str, Path], *, dpi: int, **kw: Any) -> None:
    """Save a report figure as an adaptive 256-color palette PNG (see
    :func:`_figure_png_bytes`)."""
    Path(path).write_bytes(_figure_png_bytes(fig, dpi=dpi, **kw))


def _window_preview_info(
    window_id: int, lo: float, hi: float, k: int, chi2r: Optional[float]
) -> str:
    """The hover-preview info line for a window -- frequency range (1 decimal),
    peak count, and reduced chi-square. Shared verbatim by the window list and
    the final line list so both show identical text for a given window."""
    info = (
        f"window {window_id}: {min(lo, hi):.1f}–{max(lo, hi):.1f} MHz, "
        f"{k} peak{'s' if k != 1 else ''}"
    )
    if chi2r is not None:
        info += f", χ²ᵣ {chi2r:.2f}"
    return info


def _preview_row_attr(
    stem: str,
    window_id: int,
    info: str,
    *,
    has_page: bool,
    thumb_prefix: str = "figures/",
) -> str:
    """Pre-formatted ``<tr>`` attributes for the hover-preview popup.

    Returns ``data-thumb`` (the window's magnitude panel) plus ``data-info``
    when the window has a detail page (only those windows render a ``_mag.png``);
    otherwise ``""`` so the row gets no popup. Mirrors the window-map strip's
    hover-zoom, reusing the same panel image and the shared :data:`_WINMAP_JS`.
    """
    if not has_page:
        return ""
    thumb = f"{thumb_prefix}{_panel_figure_name(stem, window_id, 'mag')}"
    return f' data-thumb="{_esc(thumb)}" data-info="{_esc(info)}"'


# ---------------------------------------------------------------------------
# Level-2 summary -> HTML (small, targeted Markdown converter)
# ---------------------------------------------------------------------------

_MD_BOLD = re.compile(r"\*\*(.+?)\*\*")
_MD_EM = re.compile(r"\*(.+?)\*")
_MD_CODE = re.compile(r"`([^`]+?)`")


def _md_inline(text: str) -> str:
    """Escape one line of Markdown text, then apply bold / emphasis / code spans.

    ``*emphasis*`` (e.g. the ``*Results:*`` labels) becomes a CSS-styleable
    ``span.md-em`` rather than a bare ``<em>`` so the look is tunable from the
    stylesheet; bold (``**``) is resolved first so its inner ``*`` are gone."""
    out = _esc(text)
    out = _MD_CODE.sub(lambda m: f"<code>{m.group(1)}</code>", out)
    out = _MD_BOLD.sub(lambda m: f"<strong>{m.group(1)}</strong>", out)
    out = _MD_EM.sub(lambda m: f'<span class="md-em">{m.group(1)}</span>', out)
    return out


def _md_table(lines: List[str]) -> str:
    """Render a contiguous block of Markdown pipe-table lines as an HTML table."""

    def cells(row: str) -> List[str]:
        return [c.strip() for c in row.strip().strip("|").split("|")]

    header = cells(lines[0])
    body = [cells(r) for r in lines[2:]]  # lines[1] is the |---|---| separator
    head_html = "".join(f"<th>{_md_inline(h)}</th>" for h in header)
    rows_html = [
        "<tr>" + "".join(f"<td>{_md_inline(c)}</td>" for c in r) + "</tr>" for r in body
    ]
    return (
        "<table><thead><tr>"
        + head_html
        + "</tr></thead><tbody>"
        + "".join(rows_html)
        + "</tbody></table>"
    )


def _md_to_html(md: str) -> str:
    """Convert the report's Markdown subset to HTML.

    Handles only what the Level-2 summary emits: ATX headings, ``-`` bullet
    lists, pipe tables, ``$$``-delimited equation blocks (rendered as the LaTeX
    source in a styled block), bold / inline-code spans, and blank-line
    paragraphs. Deliberately small and dependency-free rather than a general
    Markdown engine.
    """
    lines = md.splitlines()
    out: List[str] = []
    i = 0
    n = len(lines)
    in_list = False

    def _close_list() -> None:
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    while i < n:
        line = lines[i]
        stripped = line.strip()
        # Equation block: collect verbatim until the closing $$.
        if stripped.startswith("$$"):
            _close_list()
            eq = [stripped]
            if not (len(stripped) > 2 and stripped.endswith("$$")):
                i += 1
                while i < n and "$$" not in lines[i]:
                    eq.append(lines[i])
                    i += 1
                if i < n:
                    eq.append(lines[i])
            body = "\n".join(eq).strip("$").strip()
            # Emit display math in \[...\] so MathJax (loaded on the methods
            # page) renders it; offline the raw TeX stays visible in the block.
            out.append(f'<div class="equation">\\[{_esc(body)}\\]</div>')
            i += 1
            continue
        # Pipe table: a header row followed by a |---| separator.
        if (
            stripped.startswith("|")
            and i + 1 < n
            and set(lines[i + 1].strip()) <= set("|-: ")
            and "-" in lines[i + 1]
        ):
            _close_list()
            block = [line]
            j = i + 1
            while j < n and lines[j].strip().startswith("|"):
                block.append(lines[j])
                j += 1
            out.append(_md_table(block))
            i = j
            continue
        if not stripped:
            _close_list()
            i += 1
            continue
        if stripped.startswith("#"):
            _close_list()
            level = len(stripped) - len(stripped.lstrip("#"))
            level = min(max(level, 1), 6)
            out.append(f"<h{level}>{_md_inline(stripped[level:].strip())}</h{level}>")
            i += 1
            continue
        if stripped.startswith("- ") or stripped.startswith("* "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_md_inline(stripped[2:])}</li>")
            i += 1
            continue
        _close_list()
        out.append(f"<p>{_md_inline(stripped)}</p>")
        i += 1
    _close_list()
    return "\n".join(out)


def _summary_distribution_groups(
    model: Any, xref: Optional[CatalogCrossRef] = None
) -> List[Tuple[str, str, List[Tuple[str, str, List[float]]]]]:
    """Histogram groups keyed to the percentile table they sit beside on L3.

    Each entry is ``(anchor_text, slug, specs)``: ``anchor_text`` is a substring
    of the bold caption above the table the figure follows (so the figure lands
    next to the numbers it summarizes), ``slug`` names the PNG, and ``specs`` are
    the ``plot_summary_histograms`` specs. Groups whose data is empty drop out
    when the figure is rendered.
    """
    groups: List[Tuple[str, str, List[Tuple[str, str, List[float]]]]] = [
        (
            "Peak-strength distribution",
            "snr",
            [("Promoted-peak SNR", "SNR", model.snr_values_promoted)],
        ),
        (
            "Fit-quality distributions",
            "fitquality",
            [
                ("Reduced χ² per window", "χ²_r", model.chi2r_values),
                ("Shape error ε per window", "ε (% / bin)", model.eps_values),
                ("Precision σ_stat", "σ_stat (kHz)", model.sigma_stat_values),
            ],
        ),
        (
            "σ_f budget distribution",
            "budget",
            [
                ("Timebase σ_ε", "σ_ε (kHz)", model.sigma_eps_values),
                ("Budget σ_f", "σ_f (kHz)", model.sigma_f_values),
            ],
        ),
    ]
    if xref is not None and xref.pull_values:
        groups.append(
            (
                "Largest pulls",
                "pull",
                [("Catalog pull (f_fit−f_cat)/σ_f", "pull", xref.pull_values)],
            )
        )
    return groups


def _inject_after(
    html: str, anchor: str, snippet: str, *, closing: str = "</table>"
) -> Tuple[str, bool]:
    """Insert *snippet* right after the first *closing* tag that follows *anchor*.

    *anchor* is a substring of the rendered HTML (a caption or results phrase);
    *closing* is the element it should land after (``</table>`` for a figure that
    annotates a table, ``</p>`` for one that follows a results paragraph).
    Returns ``(html, injected)`` -- ``injected`` is False when the anchor or its
    following *closing* tag is absent (the caller then falls the figure back to a
    trailing section).
    """
    pos = html.find(anchor)
    if pos < 0:
        return html, False
    end = html.find(closing, pos)
    if end < 0:
        return html, False
    cut = end + len(closing)
    return html[:cut] + "\n" + snippet + html[cut:], True


def _inject_after_table(html: str, anchor: str, snippet: str) -> Tuple[str, bool]:
    """Insert *snippet* right after the first ``</table>`` that follows *anchor*."""
    return _inject_after(html, anchor, snippet, closing="</table>")


def _inject_before(html: str, anchor: str, snippet: str) -> Tuple[str, bool]:
    """Insert *snippet* immediately before the first occurrence of *anchor*.

    Used to drop a stage's figure at the end of its section, anchored on the
    *next* stage's heading. Returns ``(html, injected)``; ``injected`` is False
    when the anchor is absent (caller falls the figure back to a trailing block).
    """
    pos = html.find(anchor)
    if pos < 0:
        return html, False
    return html[:pos] + snippet + "\n" + html[pos:], True


# How many equal-frequency segments the magnitude-distribution panel splits the
# active band into (plus one overall panel). Six reflows cleanly across the
# flex row on a wide screen and stacks on a narrow one.
_MAG_HIST_N_SEGMENTS = 6


def _magnitude_histogram_figures(
    bundle: Any,
    out_root: Path,
    stem: str,
    dpi: int,
    plot_fn: Any,
) -> Tuple[str, str]:
    """Render the magnitude-distribution figures for the methods page.

    A log-log histogram of active-FT bin magnitudes over the whole band, then
    one per equal-frequency segment, each with the median per-bin sigma and the
    3-sigma detection level marked. Returns ``(full_band_html, per_band_html)``:
    the overall figure (a single capped ``hist`` block, placed by the driver at
    the Stage 2 results) and the segment panels (a reflowing ``maghist-grid``,
    placed after the per-band noise table). Either is ``""`` when nothing usable
    rendered."""
    import matplotlib.pyplot as plt

    f = np.asarray(bundle.frequencies, dtype=float)
    spec = np.asarray(bundle.complex_spectrum)
    rms = np.asarray(bundle.rms_noise, dtype=float)
    amp = float(bundle.amplitude_scale)
    if bundle.trim_mhz is not None:
        lo, hi = float(min(bundle.trim_mhz)), float(max(bundle.trim_mhz))
        m = (f >= lo) & (f <= hi)
        f, spec, rms = f[m], spec[m], rms[m]
    if f.size == 0:
        return "", ""
    mag = np.abs(spec) * amp
    sigma = rms * amp
    units = bundle.units_label
    xlabel = f"magnitude |X| ({units})" if units else "magnitude |X|"
    band_lo, band_hi = float(f.min()), float(f.max())

    def _median_sigma(mask: np.ndarray) -> float:
        vals = sigma[mask]
        vals = vals[np.isfinite(vals) & (vals > 0.0)]
        return float(np.median(vals)) if vals.size else 0.0

    def _render(fname: str, title: str, mags: np.ndarray, sig: float) -> Optional[str]:
        fig = plot_fn(mags, sigma_median=sig, title=title, xlabel=xlabel)
        if fig is None:
            return None
        _save_figure_png(fig, out_root / "figures" / fname, dpi=dpi)
        plt.close(fig)
        return fname

    note = (
        '<p class="fig-note">Active-FT bin magnitudes on a log–log scale -- a '
        "noise hump with a heavy line tail. The amber line marks the median "
        "per-bin &sigma;<sub>x</sub> (SNR&nbsp;=&nbsp;1); the red line the "
        "3&sigma; detection level.</p>"
    )

    full_band_html = ""
    full_name = _render(
        f"{stem}_maghist_all.png",
        f"Full band {band_lo:.0f}–{band_hi:.0f} MHz",
        mag,
        _median_sigma(np.ones(f.shape, dtype=bool)),
    )
    if full_name is not None:
        full_band_html = (
            note + f'<div class="hist"><img src="figures/{full_name}" '
            f'alt="full-band magnitude distribution"></div>'
        )

    edges = np.linspace(band_lo, band_hi, _MAG_HIST_N_SEGMENTS + 1)
    cards: List[str] = []
    for i in range(_MAG_HIST_N_SEGMENTS):
        seg_lo, seg_hi = float(edges[i]), float(edges[i + 1])
        # Last segment is right-inclusive so the band's top bin is not dropped.
        seg_mask = (f >= seg_lo) & (
            f <= seg_hi if i == _MAG_HIST_N_SEGMENTS - 1 else f < seg_hi
        )
        if not seg_mask.any():
            continue
        name = _render(
            f"{stem}_maghist_seg{i:02d}.png",
            f"{seg_lo:.0f}–{seg_hi:.0f} MHz",
            mag[seg_mask],
            _median_sigma(seg_mask),
        )
        if name is not None:
            cards.append(
                f'<figure class="maghist"><img src="figures/{name}" '
                f'alt="{seg_lo:.0f}–{seg_hi:.0f} MHz magnitude distribution">'
                f"</figure>"
            )
    per_band_html = ""
    if cards:
        per_band_html = (
            '<p class="fig-note">Per-band magnitude distributions (same axes as '
            "above), where the noise floor shifts across the band.</p>"
            '<div class="maghist-grid">' + "".join(cards) + "</div>"
        )
    return full_band_html, per_band_html


def _methods_stage_figures(
    path: str, out_root: Path, stem: str, dpi: int, *, jobs: Optional[int] = None
) -> List[Tuple[str, str]]:
    """Render the per-stage diagnostic figures for the methods page.

    Each entry is ``(before_anchor, html)``: the figure(s) for one stage, to be
    inserted right before the *next* stage's heading (i.e. at the end of that
    stage's section). Every figure is gated -- a stage that was not run, or whose
    renderer raises, is silently skipped. Reuses the per-stage ``visualize_*``
    renderers so the report inherits the same grids the CLI draws."""
    from ..visualization.fid_visualization import plot_fid_overview
    from ..visualization.tau_calibration_visualization import (
        plot_tau_distribution_from_file,
        plot_tau_heatmap_from_file,
    )
    from .stage0_impl import load_fid_from_pipeline_impl
    from .stage2_impl import visualize_noise_impl
    from .stage3_impl import visualize_peaks_impl
    from .stage4_impl import visualize_windows_impl

    # These figures span the full content width, so render them sharper than the
    # compact per-window panels: more pixels stays crisp when scaled to the page.
    fig_dpi = int(dpi * 1.4)

    # (next-stage heading anchor, [(slug, caption, render thunk), ...]).
    groups: List[Tuple[str, List[Tuple[str, str, Any]]]] = [
        (
            "Stage 1 -- Fourier transform",
            [
                (
                    "stage0_fid",
                    "Raw FID -- full trace, early-time zoom, and voltage histogram.",
                    lambda: plot_fid_overview(load_fid_from_pipeline_impl(path)),
                )
            ],
        ),
        (
            "Stage 2b -- τ calibration",
            [
                (
                    "stage2_noise",
                    "Stage 2 noise estimate -- the per-bin σₓ across the spectrum "
                    "and the bins classed as noise.",
                    lambda: visualize_noise_impl(path, title="", interactive=False),
                )
            ],
        ),
        (
            "Stage 3 -- peak detection",
            [
                (
                    "stage2b_tau_heatmap",
                    "Stage 2b STFT magnitude heatmap (log |S(a, f)|) -- the time–"
                    "frequency map the τ majority is read from.",
                    lambda: plot_tau_heatmap_from_file(path, title=""),
                ),
                (
                    "stage2b_tau_dist",
                    "Stage 2b decay-time distribution -- τ histogram, τ vs SNR, τ vs "
                    "frequency, and the one- vs two-component test.",
                    lambda: plot_tau_distribution_from_file(path, title=""),
                ),
            ],
        ),
        (
            "Stage 4 -- window assignment",
            [
                (
                    "stage3_peaks",
                    "Stage 3 detections over the spectrum, colored by SNR class "
                    "and detection pass (promoted peaks filled, below-cutoff "
                    "candidates open), with the SNR distribution.",
                    lambda: visualize_peaks_impl(
                        path,
                        title="",
                        interactive=False,
                        show_snr_histogram=True,
                    ),
                )
            ],
        ),
        (
            "Stage 5 -- per-window fitting",
            [
                (
                    "stage4_windows",
                    "Stage 4 window plan -- the fit-window spans, free peaks and "
                    "fixed contributors, and the edge-coherence statistic.",
                    lambda: visualize_windows_impl(
                        path,
                        title="",
                        figsize=(15, 9.5),
                        interactive=False,
                    ),
                )
            ],
        ),
    ]

    # The per-stage figures are independent and a couple are expensive (the
    # 750k-point FID overview, the full-spectrum stage visualizations), so render
    # the thunks across a process pool, then write the bytes + build the HTML in
    # group order here. Bytes are byte-identical to a serial render at fixed DPI.
    flat: List[Tuple[int, str, str, Any]] = [
        (gi, slug, caption, thunk)
        for gi, (_anchor, figs) in enumerate(groups)
        for (slug, caption, thunk) in figs
    ]
    rendered = _render_methods_figures([t for (_, _, _, t) in flat], fig_dpi, jobs=jobs)

    group_blocks: Dict[int, List[str]] = {}
    for (gi, slug, caption, _thunk), data in zip(flat, rendered):
        if data is None:  # a stage that was not run, or whose renderer raised
            continue
        fname = f"{stem}_methods_{slug}.png"
        (out_root / "figures" / fname).write_bytes(data)
        group_blocks.setdefault(gi, []).append(
            f'<p class="fig-note">{caption}</p>'
            f'<div class="hist"><img src="figures/{fname}" alt="{_esc(slug)}"></div>'
        )

    out: List[Tuple[str, str]] = []
    for gi, (anchor, _figs) in enumerate(groups):
        blocks = group_blocks.get(gi)
        if blocks:
            out.append((anchor, "".join(blocks)))
    return out


_FORK_ITEM = TypeVar("_FORK_ITEM")
_FORK_RESULT = TypeVar("_FORK_RESULT")


def fork_map(
    items: List[_FORK_ITEM],
    worker: Callable[[_FORK_ITEM], _FORK_RESULT],
    *,
    jobs: Optional[int] = None,
    override: Optional[int] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[_FORK_RESULT]:
    """Map *worker* over *items*, in a forking process pool when worthwhile.

    Runs serially in-process when there are fewer than two items, fewer than two
    resolvable workers (``resolve_worker_count(jobs, override=override)``), or the
    platform lacks ``fork``; otherwise forks a ``ProcessPoolExecutor`` capped at
    ``min(workers, n)``. Either way the results come back in input order, and
    *progress* (if given) is called ``progress(i, n)`` once per completed item
    (1-based, in order) -- the ordered per-window progress log.

    The big read-only inputs a *worker* needs (file path, the active-FT bundle,
    unpicklable closures) must travel through a fork-inherited module global the
    caller sets **before** calling this and clears in a ``finally``; only the
    *items* entries cross the process boundary. ``fork_map`` does not manage that
    global.
    """
    import multiprocessing

    n = len(items)
    max_workers = resolve_worker_count(jobs, override=override)
    serial = (
        n < 2
        or max_workers < 2
        or "fork" not in multiprocessing.get_all_start_methods()
    )
    out: List[_FORK_RESULT] = []
    if serial:
        for i, item in enumerate(items, start=1):
            out.append(worker(item))
            if progress is not None:
                progress(i, n)
        return out

    from concurrent.futures import ProcessPoolExecutor

    ctx_mp = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=min(max_workers, n), mp_context=ctx_mp) as ex:
        # ex.map preserves input order and re-raises worker exceptions.
        for i, res in enumerate(ex.map(worker, items), start=1):
            out.append(res)
            if progress is not None:
                progress(i, n)
    return out


# Per-stage methods figures are rendered the same way as the per-window panels:
# a forking pool, thunks reached via a fork-inherited global (they are local
# closures over the file path, not picklable, so they must be inherited, never
# sent as task args). Workers return PNG bytes (or None when the render raises).
_METHODS_RENDER_CTX: Optional[Dict[str, Any]] = None


def _render_one_methods_figure(thunk: Any, fig_dpi: int) -> Optional[bytes]:
    """Run one methods-figure thunk to PNG bytes; ``None`` if it has no data."""
    import matplotlib.pyplot as plt

    try:
        fig = thunk()
    except Exception:  # optional diagnostic: skip a missing/failed stage
        return None
    if fig is None:
        return None
    data = _figure_png_bytes(fig, dpi=fig_dpi, bbox_inches="tight")
    plt.close(fig)
    return data


def _methods_figure_worker(idx: int) -> Tuple[int, Optional[bytes]]:
    ctx = _METHODS_RENDER_CTX
    assert ctx is not None
    return idx, _render_one_methods_figure(ctx["thunks"][idx], ctx["fig_dpi"])


def _render_methods_figures(
    thunks: List[Any], fig_dpi: int, *, jobs: Optional[int] = None
) -> List[Optional[bytes]]:
    """Render the methods-page figure thunks to PNG bytes, in parallel when
    worthwhile (same forking-pool / serial-fallback policy as the window
    figures; ``_FIGURE_RENDER_WORKERS`` pins the count for tests)."""
    global _METHODS_RENDER_CTX
    _METHODS_RENDER_CTX = {"thunks": thunks, "fig_dpi": fig_dpi}
    try:
        rendered = fork_map(
            list(range(len(thunks))),
            _methods_figure_worker,
            jobs=jobs,
            override=_FIGURE_RENDER_WORKERS,
        )
    finally:
        _METHODS_RENDER_CTX = None
    return [data for _idx, data in rendered]


def _summary_page(stem: str, md_html: str, leftover: List[str]) -> str:
    """The HTML methods + results page (Level-2 content; histograms interleaved).

    The per-distribution figures are injected next to their tables in *md_html*
    by the driver; any that could not be placed (anchor table absent) arrive in
    *leftover* and are shown in a trailing Distributions section.
    """
    body: List[str] = [
        '<div class="nav"><a href="index.html">&larr; index</a></div>',
        md_html,
    ]
    if leftover:
        body.append("<h2>Distributions</h2>")
        body += leftover
    return _page(
        f"{stem} methods",
        body,
        css_href="assets/style.css",
        head_extra=_MATHJAX_HEAD,
    )


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------


def _index_summary_block(model: Any, products: FinalProducts) -> List[str]:
    m = model
    state = products.calibration_state
    eps_ppm = products.epsilon * 1e6
    seps_ppm = products.sigma_epsilon * 1e6
    items = [
        f"<li><strong>Source:</strong> "
        f"<code>{_esc(m.source_path or '(unknown)')}</code> "
        f"({_esc(m.source_format or 'unknown')} format)</li>",
        f"<li><strong>Probe:</strong> {_esc(_md_num(m.probe_freq_mhz, 8))} MHz, "
        f"{_esc(m.sideband)} sideband</li>",
    ]
    if m.band_lo_mhz is not None and m.band_hi_mhz is not None:
        items.append(
            f"<li><strong>Active band:</strong> "
            f"{_esc(_md_num(m.band_lo_mhz, 7))}&ndash;"
            f"{_esc(_md_num(m.band_hi_mhz, 7))} MHz</li>"
        )
    items.append(
        f"<li><strong>Lines reported:</strong> {len(products.peaks):,} "
        f"({m.n_fitted_peaks:,} fitted across {m.n_windows_fit:,} windows)</li>"
    )
    items.append(
        f"<li><strong>Calibration state:</strong> <code>{_esc(state)}</code> "
        f"(&epsilon; = {eps_ppm:+.3f} &plusmn; {seps_ppm:.3f} ppm, "
        f"&sigma;<sub>floor</sub> = {products.sigma_floor_khz:.3f} kHz)</li>"
    )
    if m.chi2_median is not None:
        items.append(
            f"<li><strong>Reduced &chi;&sup2;:</strong> median "
            f"{_esc(_md_num(m.chi2_median, 3))} "
            f"(range {_esc(_md_num(m.chi2_min, 3))}&ndash;"
            f"{_esc(_md_num(m.chi2_max, 3))})</li>"
        )
    cal_phrase = _CAL_STATE_PHRASE.get(state, state).replace("**", "")
    return [
        "<h2>Summary</h2>",
        '<ul class="summary">',
        *items,
        "</ul>",
        f"<p>{_esc(cal_phrase)}</p>",
    ]


# Symmetric horizontal margin (figure fraction) shared by the overview figure's
# axes box and the window-map SVG, so the two line up. The margin is wide enough
# that the y-axis tick labels and title fit in the left gutter without shifting
# the (symmetric) axes box, and the edge frequency labels have room.
_OVERVIEW_MARGIN_FRAC = 0.07


def _plot_index_overview(
    bundle: Any,
    attention_ranges: List[Tuple[float, float]],
    *,
    highlight_range: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (13.0, 2.8),
) -> Any:
    """Full-spectrum magnitude, with attention windows (and optionally one
    current window) shaded.

    Renders the trimmed active-FT magnitude across the whole band. The index
    shades every review-flagged window (``attention_ranges``); a per-window
    page passes ``highlight_range`` to mark the current window in green. The
    axes box uses a fixed symmetric :data:`_OVERVIEW_MARGIN_FRAC` so it lines up
    with the window-map strip beneath it, with the y-axis labels living in the
    left gutter. Returns a matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    f = np.asarray(bundle.frequencies, dtype=float)
    spec = np.asarray(bundle.complex_spectrum)
    amp = float(bundle.amplitude_scale)
    if bundle.trim_mhz is not None:
        lo, hi = float(min(bundle.trim_mhz)), float(max(bundle.trim_mhz))
        m = (f >= lo) & (f <= hi)
        f, spec = f[m], spec[m]
    mag = np.abs(spec) * amp
    fig = plt.figure(figsize=figsize)
    # Explicit axes box: symmetric L/R margins (the y labels sit in the left
    # gutter), room below for the x labels and above for the title.
    ax = fig.add_axes(
        (_OVERVIEW_MARGIN_FRAC, 0.22, 1.0 - 2.0 * _OVERVIEW_MARGIN_FRAC, 0.60)
    )
    ax.plot(f, mag, color="0.3", lw=0.5)
    for lo_w, hi_w in attention_ranges:
        ax.axvspan(
            min(lo_w, hi_w), max(lo_w, hi_w), color="tab:orange", alpha=0.35, zorder=0
        )
    if highlight_range is not None:
        hlo, hhi = highlight_range
        ax.axvspan(
            min(hlo, hhi), max(hlo, hhi), color="tab:green", alpha=0.45, zorder=1
        )
    if f.size:
        ax.set_xlim(float(f[0]), float(f[-1]))
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("frequency (MHz)", fontsize=9)
    suffix = f" ({bundle.units_label})" if bundle.units_label else ""
    ax.set_ylabel(f"|X(f)|{suffix}", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    if title is None:
        title = "full spectrum"
        if attention_ranges:
            title += " (attention windows shaded)"
    ax.set_title(title, fontsize=10)
    # Spine-free / light-grid presentation, matching the per-window panels. The
    # magnitude is strictly positive, so no zero baseline.
    from ..visualization.report_style import apply_bare_style

    apply_bare_style(ax)
    return fig


# Interactive-overview geometry. The SVG overlay's viewBox matches the overview
# figure's pixel aspect (figsize 13.0 x 2.8), and the rects are placed by the
# figure's axes box (left margin _OVERVIEW_MARGIN_FRAC, the data band running
# bottom=0.22..top=0.82 of the figure height). preserveAspectRatio="none" stretches
# the overlay to the div, which carries the same image at background-size 100% 100%,
# so a rect at a given frequency lands over that frequency in the image.
_CTX_VIEW_W = 1300.0
_CTX_VIEW_H = 280.0
# Axes box in top-down figure fractions: top = 1 - (0.22 + 0.60), height = 0.60.
_CTX_PLOT_Y0 = 0.18
_CTX_PLOT_H = 0.60


def _spectrum_nav(
    rows: List[Tuple[int, float, float, int, Optional[float], bool, bool]],
    band: Tuple[float, float],
    stem: str,
    *,
    link_prefix: str,
    thumb_prefix: str,
    current_id: Optional[int] = None,
) -> str:
    """The interactive full-spectrum overview: clickable per-window rects on the
    shared overview image.

    A ``div.spectrum-ctx`` (its background is the shared overview image, bound by
    a per-build CSS rule) holding an SVG overlay with one ``<rect>`` per window,
    positioned by frequency over the image's frequency axis. Attention windows
    get a faint tint, the *current_id* window (on its own page) a green "you are
    here"; a window with a detail page is wrapped in an SVG ``<a>`` (click
    navigates) and carries ``data-thumb`` (its magnitude panel) plus a ``<title>``
    for the hover-zoom popup. ``rows`` are the index table's
    ``(wid, lo, hi, k, chi2r, has_page, attn)`` tuples; *link_prefix* /
    *thumb_prefix* resolve the page/figure paths from the index (``"windows/"`` /
    ``"figures/"``) or a window page (``""`` / ``"../figures/"``).
    """
    lo, hi = float(min(band)), float(max(band))
    span = hi - lo if hi > lo else 1.0
    left = _OVERVIEW_MARGIN_FRAC
    usable = 1.0 - 2.0 * left
    y = _CTX_PLOT_Y0 * _CTX_VIEW_H
    h = _CTX_PLOT_H * _CTX_VIEW_H

    def x_of(f: float) -> float:
        return (left + (float(f) - lo) / span * usable) * _CTX_VIEW_W

    parts: List[str] = [
        '<div class="spectrum-ctx">',
        f'<svg class="spectrum-ctx-svg" viewBox="0 0 {_CTX_VIEW_W:.0f} '
        f'{_CTX_VIEW_H:.0f}" preserveAspectRatio="none" role="img" '
        'aria-label="interactive full-spectrum overview">',
    ]
    for wid, wlo, whi, k, chi2r, has_page, attn in rows:
        x0, x1 = x_of(min(wlo, whi)), x_of(max(wlo, whi))
        w = max(x1 - x0, 2.0)
        cls = "specnav-rect"
        if attn:
            cls += " attn"
        if current_id is not None and wid == current_id:
            cls += " current"
        info = (
            f"window {wid}: {min(wlo, whi):.3f}–{max(wlo, whi):.3f} MHz, "
            f"{k} peak{'s' if k != 1 else ''}"
        )
        if chi2r is not None:
            info += f", χ²ᵣ {chi2r:.2f}"
        thumb = (
            f' data-thumb="{thumb_prefix}{stem}_window_{wid:03d}_mag.png"'
            if has_page
            else ""
        )
        rect = (
            f'<rect class="{cls}" x="{x0:.1f}" y="{y:.1f}" width="{w:.1f}" '
            f'height="{h:.1f}" data-window="{wid}"{thumb}>'
            f"<title>{_esc(info)}</title></rect>"
        )
        if has_page:
            parts.append(f'<a href="{link_prefix}{_window_page_name(wid)}">{rect}</a>')
        else:
            parts.append(rect)
    parts.append("</svg></div>")
    return "".join(parts)


def _spectrum_ctx_css(overview_name: Optional[str]) -> str:
    """The per-build CSS rule binding the shared overview image as the
    ``.spectrum-ctx`` background (``""`` when no overview rendered).

    The ``url`` is relative to the stylesheet (``assets/style.css``), so it
    resolves identically from the index and the per-window pages; the single-file
    collapse rewrites it to a deduplicated data URI.
    """
    if overview_name is None:
        return ""
    return (
        "\n/* Per build: the shared overview image as the interactive-overview "
        "background. */\n"
        f".spectrum-ctx {{ background-image: url(../figures/{overview_name}); }}\n"
    )


# Optional hover-zoom popup: a tiny, dependency-free script that shows a window's
# magnitude thumbnail at the cursor on hover. It drives both the window-map strip
# (rects carry data-thumb + a native <title>) and the index tables (window list /
# final line list rows carry data-thumb + data-info). Everything is fully usable
# without it (rects link + carry native tooltips; rows link to the window page),
# so this degrades gracefully when scripting is off.
_WINMAP_JS = """<script>
(function () {
  var rects = document.querySelectorAll('[data-window]');
  var rows = document.querySelectorAll('tr[data-thumb]');
  if (!rects.length && !rows.length) return;
  var pop = document.createElement('div');
  pop.className = 'winmap-pop';
  pop.style.display = 'none';
  document.body.appendChild(pop);
  function move(e) {
    var x = e.clientX + 16, y = e.clientY + 16;
    var w = pop.offsetWidth, h = pop.offsetHeight;
    if (x + w > window.innerWidth) x = e.clientX - w - 16;
    if (y + h > window.innerHeight) y = window.innerHeight - h - 8;
    pop.style.left = Math.max(4, x) + 'px';
    pop.style.top = Math.max(4, y) + 'px';
  }
  function show(e) {
    var t = e.currentTarget;
    var thumb = t.getAttribute('data-thumb');
    // data-thumb is a basename key into the deduplicated base64 map injected by
    // the single-file collapse; resolve it to the embedded image.
    if (thumb && window.__thumbs && window.__thumbs[thumb]) thumb = window.__thumbs[thumb];
    var info = t.getAttribute('data-info') || '';
    pop.innerHTML = (thumb ? '<img src="' + thumb + '" alt="">' : '') +
      '<div class="winmap-pop-info">' + info + '</div>';
    pop.style.display = 'block';
    move(e);
  }
  function hide() { pop.style.display = 'none'; }
  function bind(el) {
    // Move any native <title> into a data attribute so the popup is the only
    // tooltip when scripting is on; without this script the <title> stays and
    // the browser shows it natively. (Table rows store data-info directly.)
    var titleEl = el.querySelector('title');
    if (titleEl) {
      el.setAttribute('data-info', titleEl.textContent);
      el.removeChild(titleEl);
    }
    el.addEventListener('mouseenter', show);
    el.addEventListener('mousemove', move);
    el.addEventListener('mouseleave', hide);
  }
  rects.forEach(bind);
  rows.forEach(bind);
})();
</script>"""


# Optional compact-mode toggle. The "Compact" topnav
# button flips a class on <html>; in compact mode the stylesheet shrinks every
# report figure to a thumbnail (pure CSS over the already-embedded full images,
# so no extra bytes), and clicking a thumbnail expands just that figure. With
# scripting off the button does nothing and every figure stays full size.
_COMPACT_JS = """<script>
(function () {
  var root = document.documentElement;
  var btn = document.querySelector('.compact-toggle');
  if (!btn) return;
  var sel = '.fit-panels img, .cov-heatmap img, .hist img, .maghist img';
  function label() {
    btn.textContent =
      root.classList.contains('report-compact') ? 'Full view' : 'Compact';
  }
  btn.addEventListener('click', function () {
    root.classList.toggle('report-compact');
    label();
  });
  document.addEventListener('click', function (e) {
    if (!root.classList.contains('report-compact')) return;
    var t = e.target;
    if (t.tagName === 'IMG' && t.matches && t.matches(sel)) {
      t.classList.toggle('zoom-expanded');
      return;
    }
    // The interactive overview is an SVG over a CSS background, not an <img>;
    // expand the container. A click on a window-nav rect navigates instead, so
    // only a click on empty overview area (no data-window) toggles the zoom.
    var ctx = t.closest && t.closest('.spectrum-ctx');
    if (ctx && !(t.closest && t.closest('[data-window]'))) {
      ctx.classList.toggle('zoom-expanded');
    }
  });
  label();
})();
</script>"""


# In-report curation. The report opens read-only -- the same clean document the
# no-JS view shows -- and the ``Curate`` toggle adds ``curation-enabled`` to
# <html> to reveal the controls. The boot script builds a docked cart and wires
# the inline per-row / per-window controls by event delegation. Every control
# only *emits* an edit into the cart; the cart exports the
# ``action,window,freqs,params`` curation CSV that ``review apply`` consumes.
# The frequency a control emits is the row's raw Stage-5 model frequency
# (``data-freq``), which is what the edit verbs match on -- not the calibrated
# display value. Vanilla JS only, in the spirit of the compact toggle above.
_CURATION_JS = r"""<script>
(function () {
  var root = document.documentElement;
  var STEM = window.__stem || 'report';
  var ops = [];  // {action, window, freqs, params, label}

  // --- docked cart -----------------------------------------------------------
  var cart = document.createElement('div');
  cart.id = 'cur-cart';
  cart.innerHTML =
    '<div class="cur-cart-head"><button type="button" class="cur-min"' +
    ' title="collapse">Curation cart (<span class="cur-n">0</span>)</button></div>' +
    '<div class="cur-cart-body"></div>' +
    '<div class="cur-cart-foot">' +
    '<button type="button" class="cur-dl">Download .csv</button>' +
    '<button type="button" class="cur-copy">Copy</button>' +
    '<button type="button" class="cur-clear">Clear</button>' +
    '<pre class="cur-cmd"></pre>' +
    '<textarea class="cur-ta" aria-hidden="true"></textarea></div>';
  document.body.appendChild(cart);
  var cartBody = cart.querySelector('.cur-cart-body');
  var cmdPre = cart.querySelector('.cur-cmd');
  var copyTa = cart.querySelector('.cur-ta');

  function fmtCmd() {
    return 'ftmwpipeline review apply ' + STEM + '.ftmw ' + STEM +
      '_curation.csv\n  (add --dry-run to preview the resolved plan)';
  }
  function csvCell(s) { return (s == null) ? '' : String(s); }
  function toCsv() {
    var lines = ['action,window,freqs,params'];
    ops.forEach(function (o) {
      lines.push([o.action, o.window, csvCell(o.freqs),
                  csvCell(o.params)].join(','));
    });
    return lines.join('\n') + '\n';
  }

  function opKey(o) {
    return o.action + '|' + o.window + '|' + o.freqs + '|' + (o.params || '');
  }
  function findKey(key) {
    for (var i = 0; i < ops.length; i++) {
      if (opKey(ops[i]) === key) return i;
    }
    return -1;
  }

  function rowsFor(table, w, f) {
    return Array.prototype.slice.call(document.querySelectorAll(
      'table.' + table + ' tr[data-window="' + w + '"][data-freq="' + f + '"]'));
  }
  function refreshRows() {
    // Re-derive every row's pending state from the op list (idempotent).
    document.querySelectorAll('tr[data-freq]').forEach(function (tr) {
      tr.classList.remove('cur-removed', 'cur-split', 'cur-merge-grp',
                          'cur-added');
      var b = tr.querySelector('.cur-splitbadge');
      if (b) b.remove();
    });
    ops.forEach(function (o) {
      if (o.action === 'remove') {
        rowsFor('peak-list', o.window, o.freqs).forEach(function (tr) {
          tr.classList.add('cur-removed');
        });
      } else if (o.action === 'split') {
        rowsFor('peak-list', o.window, o.freqs).forEach(function (tr) {
          tr.classList.add('cur-split');
          var cell = tr.querySelector('.cur-cell');
          if (cell && !cell.querySelector('.cur-splitbadge')) {
            var k = (o.params || '').replace('into=', '') || '2';
            var s = document.createElement('span');
            s.className = 'cur-splitbadge';
            s.textContent = '→' + k;
            cell.appendChild(s);
          }
        });
      } else if (o.action === 'merge') {
        o.freqs.split(';').forEach(function (f) {
          rowsFor('peak-list', o.window, f).forEach(function (tr) {
            tr.classList.add('cur-merge-grp');
          });
        });
      } else if (o.action === 'add') {
        rowsFor('ledger', o.window, o.freqs).forEach(function (tr) {
          tr.classList.add('cur-added');
        });
      }
    });
  }

  // On-plot markers: one per queued edit, placed by mapping its frequency
  // forward (MHz -> pixel) through the same stamped data-axes box the click
  // inverts. The SVG is click-transparent except the marker handles, so plot
  // clicks still reach the image; clicking a handle drops that edit.
  var SVGNS = 'http://www.w3.org/2000/svg';
  var ACT_COLOR = { remove: '#c0392b', split: '#b9770e', add: '#1e8e3e',
                    merge: '#7d3c98' };
  function renderMarkers() {
    document.querySelectorAll('.cur-plot-svg').forEach(function (svg) {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var img = svg.parentNode.querySelector('img.cur-plot');
      if (!img) return;
      var w = svg.getAttribute('data-window');
      var px0 = parseFloat(img.getAttribute('data-axes-x0'));
      var px1 = parseFloat(img.getAttribute('data-axes-x1'));
      var y0 = parseFloat(img.getAttribute('data-axes-y0'));
      var y1 = parseFloat(img.getAttribute('data-axes-y1'));
      var flo = parseFloat(img.getAttribute('data-axes-flo'));
      var fhi = parseFloat(img.getAttribute('data-axes-fhi'));
      if (!(px1 > px0) || isNaN(flo) || isNaN(fhi)) return;
      var fmin = Math.min(flo, fhi), fmax = Math.max(flo, fhi);
      ops.forEach(function (o, i) {
        if (String(o.window) !== String(w)) return;
        var color = ACT_COLOR[o.action];
        if (!color || !o.freqs) return;
        o.freqs.split(';').forEach(function (fs) {
          var f = parseFloat(fs);
          if (isNaN(f) || f < fmin || f > fmax) return;
          var x = px0 + (f - flo) / (fhi - flo) * (px1 - px0);
          var line = document.createElementNS(SVGNS, 'line');
          line.setAttribute('x1', x); line.setAttribute('x2', x);
          line.setAttribute('y1', y0); line.setAttribute('y2', y1);
          line.setAttribute('stroke', color);
          line.setAttribute('stroke-width', '2');
          if (o.action === 'remove') line.setAttribute('stroke-dasharray', '6 4');
          svg.appendChild(line);
          var dot = document.createElementNS(SVGNS, 'circle');
          dot.setAttribute('cx', x); dot.setAttribute('cy', y0);
          dot.setAttribute('r', '7'); dot.setAttribute('fill', color);
          dot.setAttribute('class', 'cur-marker');
          dot.setAttribute('data-i', i);
          var title = document.createElementNS(SVGNS, 'title');
          title.textContent = o.label + ' — click to drop';
          dot.appendChild(title);
          svg.appendChild(dot);
        });
      });
    });
  }

  function renderCart() {
    var n = ops.length;
    document.querySelectorAll('.cur-badge').forEach(function (b) {
      b.textContent = 'cart (' + n + ')';
    });
    cart.querySelector('.cur-n').textContent = n;
    cart.classList.toggle('cur-empty', n === 0);
    var byWin = {}, order = [];
    ops.forEach(function (o, i) {
      if (!byWin[o.window]) { byWin[o.window] = []; order.push(o.window); }
      byWin[o.window].push({ op: o, i: i });
    });
    var html = '';
    order.forEach(function (w) {
      html += '<div class="cur-grp"><div class="cur-grp-h">window ' + w +
        '</div>';
      byWin[w].forEach(function (e) {
        html += '<div class="cur-entry"><span>' + e.op.label +
          '</span><button type="button" class="cur-x" data-i="' + e.i +
          '" title="drop">✕</button></div>';
      });
      html += '</div>';
    });
    cartBody.innerHTML = html ||
      '<div class="cur-grp-h">No queued edits.</div>';
    cmdPre.textContent = fmtCmd();
    copyTa.value = toCsv();
    refreshRows();
    renderMarkers();
  }

  function addOp(o) { ops.push(o); renderCart(); }
  function dropOp(i) { ops.splice(i, 1); renderCart(); }
  function clearCart() { ops = []; renderCart(); }

  // --- control wiring (event delegation) -------------------------------------
  document.addEventListener('click', function (e) {
    var t = e.target;
    if (t.closest && t.closest('.cur-x')) {
      dropOp(parseInt(t.closest('.cur-x').getAttribute('data-i'), 10));
      return;
    }
    if (t.classList && t.classList.contains('cur-toggle')) {
      root.classList.toggle('curation-enabled');
      t.textContent = root.classList.contains('curation-enabled') ?
        'Read-only' : 'Curate';
      return;
    }
    if (t.classList && t.classList.contains('cur-min')) {
      cart.classList.toggle('cur-collapsed');
      return;
    }
    if (t.classList && t.classList.contains('cur-dl')) {
      var blob = new Blob([toCsv()], { type: 'text/csv' });
      var url = URL.createObjectURL(blob);
      var a = document.createElement('a');
      a.href = url; a.download = STEM + '_curation.csv';
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(function () { URL.revokeObjectURL(url); }, 0);
      return;
    }
    if (t.classList && t.classList.contains('cur-copy')) {
      var text = toCsv();
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).catch(function () {
          copyTa.select(); document.execCommand('copy');
        });
      } else {
        copyTa.select(); document.execCommand('copy');
      }
      return;
    }
    if (t.classList && t.classList.contains('cur-clear')) { clearCart(); return; }

    // Drop a queued edit by clicking its on-plot marker handle.
    if (t.classList && t.classList.contains('cur-marker')) {
      e.stopPropagation();
      dropOp(parseInt(t.getAttribute('data-i'), 10));
      return;
    }
    // Arm toggle: click-to-add is live only on the single armed panel, so a tap
    // meant to scroll or zoom never adds. Arming one panel disarms the others.
    if (t.classList && t.classList.contains('cur-plot-arm')) {
      var wrap = t.closest('.cur-plot-wrap');
      var on = wrap && wrap.classList.contains('cur-armed');
      document.querySelectorAll('.cur-plot-wrap.cur-armed').forEach(function (w) {
        w.classList.remove('cur-armed');
      });
      if (wrap && !on) wrap.classList.add('cur-armed');
      document.querySelectorAll('.cur-plot-arm').forEach(function (b) {
        var w = b.closest('.cur-plot-wrap');
        b.textContent = (w && w.classList.contains('cur-armed')) ? '× done' : '+ add';
      });
      return;
    }

    // Click-on-plot: on the armed |X| panel, invert the click x to a molecular
    // MHz and queue an `add` seed there. The stamped data-axes box is in natural
    // PNG pixels; scale the click through the rendered <img> size. Compact mode
    // zooms the thumbnail on click, so defer to that.
    if (t.classList && t.classList.contains('cur-plot')) {
      if (!root.classList.contains('curation-enabled')) return;
      if (root.classList.contains('report-compact')) return;
      var awrap = t.closest('.cur-plot-wrap');
      if (!awrap || !awrap.classList.contains('cur-armed')) return;
      var px0 = parseFloat(t.getAttribute('data-axes-x0'));
      var px1 = parseFloat(t.getAttribute('data-axes-x1'));
      var flo = parseFloat(t.getAttribute('data-axes-flo'));
      var fhi = parseFloat(t.getAttribute('data-axes-fhi'));
      if (!(px1 > px0) || isNaN(flo) || isNaN(fhi) || !t.naturalWidth) return;
      var rect = t.getBoundingClientRect();
      var nat = (e.clientX - rect.left) / rect.width * t.naturalWidth;
      if (nat < px0 || nat > px1) return;  // outside the data axes
      var f = (flo + (nat - px0) / (px1 - px0) * (fhi - flo)).toFixed(4);
      addOp({ action: 'add', window: t.getAttribute('data-window'), freqs: f,
              params: '', label: 'add ' + f + ' (plot)' });
      return;
    }

    if (!(t.classList && t.classList.contains('cur-btn'))) return;
    var act = t.getAttribute('data-act');
    var sec = t.closest('section');
    if (act === 'merge-selected') {
      var w = t.getAttribute('data-window');
      var boxes = sec ? sec.querySelectorAll('.cur-merge:checked') : [];
      if (boxes.length < 2) { return; }
      var fs = [];
      Array.prototype.forEach.call(boxes, function (cb) {
        var tr = cb.closest('tr');
        fs.push(tr.getAttribute('data-freq'));
        cb.checked = false;
      });
      addOp({ action: 'merge', window: w, freqs: fs.join(';'), params: '',
              label: 'merge ' + fs.join(' + ') });
      return;
    }
    if (act === 'add-typed') {
      var w2 = t.getAttribute('data-window');
      var inp = t.parentNode.querySelector('.cur-addfreq');
      var v = inp && inp.value ? inp.value.trim() : '';
      if (!v) { return; }
      addOp({ action: 'add', window: w2, freqs: v, params: '',
              label: 'add ' + v });
      if (inp) inp.value = '';
      return;
    }
    if (act === 'accept') {
      addOp({ action: 'accept', window: t.getAttribute('data-window'),
              freqs: '', params: '', label: 'mark reviewed' });
      return;
    }

    // Per-row remove / split / ledger add: key off the closest row.
    var row = t.closest('tr');
    if (!row) return;
    var win = row.getAttribute('data-window');
    var freq = row.getAttribute('data-freq');
    if (act === 'remove') {
      var k = 'remove|' + win + '|' + freq + '|';
      var idx = findKey(k);
      if (idx >= 0) { dropOp(idx); }
      else { addOp({ action: 'remove', window: win, freqs: freq, params: '',
                     label: 'remove ' + freq }); }
      return;
    }
    if (act === 'split') {
      var kin = row.querySelector('.cur-k');
      var kk = kin && kin.value ? String(parseInt(kin.value, 10) || 2) : '2';
      var params = 'into=' + kk;
      var existing = -1;
      for (var i = 0; i < ops.length; i++) {
        if (ops[i].action === 'split' && ops[i].window === win &&
            ops[i].freqs === freq) { existing = i; break; }
      }
      if (existing >= 0) { dropOp(existing); }
      else { addOp({ action: 'split', window: win, freqs: freq,
                     params: params, label: 'split ' + freq + ' → ' + kk }); }
      return;
    }
    if (act === 'add') {  // ledger candidate add
      var ka = 'add|' + win + '|' + freq + '|';
      var ia = findKey(ka);
      if (ia >= 0) { dropOp(ia); }
      else { addOp({ action: 'add', window: win, freqs: freq, params: '',
                     label: 'add ' + freq }); }
      return;
    }
  });

  renderCart();
})();
</script>"""


def _index_window_table(
    rows: List[Tuple[int, float, float, int, Optional[float], bool, bool]],
    preview_attrs: Dict[int, str],
) -> str:
    """Build the index window list. Each row:
    ``(wid, lo, hi, k, chi2r, has_page, needs_attention)``.

    ``preview_attrs`` maps a window id to its pre-formatted ``<tr>`` attributes
    (``data-thumb`` / ``data-info``) so the shared hover-preview popup shows the
    magnitude panel on hover -- the same map feeds the final line list."""
    out_rows: List[List[str]] = []
    row_attrs: List[str] = []
    for wid, lo, hi, k, chi2r, has_page, attn in rows:
        label = f"window {wid}"
        if has_page:
            link = f'<a href="windows/{_window_page_name(wid)}">{label}</a>'
        else:
            link = label
        if attn:
            link += '<span class="badge">attention</span>'
        row_attrs.append(preview_attrs.get(wid, ""))
        out_rows.append(
            [
                link,
                f"{min(lo, hi):.3f}&ndash;{max(lo, hi):.3f}",
                f"{k:,}",
                "n/a" if chi2r is None else _esc(_md_num(chi2r, 3)),
            ]
        )
    return _table(
        ["Window", "Range (MHz)", "Peaks", "&chi;&sup2;<sub>r</sub>"],
        out_rows,
        cls="window-list",
        row_attrs=row_attrs,
    )


def _catalog_cell(m: Optional[CatalogMatch]) -> str:
    """A catalog-match badge cell: the opaque label, Δ/pull on hover, or empty.

    Proximity annotation only -- the label is echoed as a cross-check, never as
    an assignment.
    """
    if m is None:
        return ""
    title = f"Δ {_g(m.delta_khz, 3)} kHz, pull {_g(m.pull, 2)}"
    return f'<span class="badge cat" title="{_esc(title)}">{_esc(m.label)}</span>'


def _lattice_cell(p: FinalPeak) -> str:
    """A clock-lattice badge cell: the matched lattice identity, or empty.

    A non-empty cell marks a fitted line that lands on the declared clock
    lattice -- a candidate instrumental artifact that survived the spur gate.
    It is flagged for review, never an assignment and never auto-removed.
    """
    cl = getattr(p, "clock_lattice", None)
    if not cl:
        return ""
    title = (
        f"On the declared clock lattice ({cl}) -- possible instrumental "
        "artifact, not a molecular line"
    )
    return f'<span class="badge lattice" title="{_esc(title)}">{_esc(cl)}</span>'


def _index_final_table(
    products: FinalProducts,
    matches: Optional[List[Optional[CatalogMatch]]] = None,
    *,
    preview_attrs: Optional[Dict[int, str]] = None,
) -> str:
    """Build the index final line list.

    ``preview_attrs`` maps a window id to its pre-formatted ``<tr>`` attributes;
    each peak inherits its window's hover-preview popup (the same magnitude-panel
    thumbnail and identical info text as the window list)."""
    uname, uval = _amplitude_unit(products)
    with_cat = matches is not None
    rows: List[List[str]] = []
    row_attrs: List[str] = []
    for i, p in enumerate(products.peaks):
        wid = p.window_id
        win_cell = (
            f'<a href="windows/{_window_page_name(wid)}">{wid}</a>'
            if wid is not None
            else ""
        )
        row_attrs.append(
            preview_attrs.get(wid, "")
            if preview_attrs is not None and wid is not None
            else ""
        )
        row = [
            _esc(_freq(p.frequency_mhz)),
            _esc(_g(p.sigma_f_khz, 3)),
            _esc(_scaled(p.amplitude, uval)),
            _esc(_md_num(p.snr, 3)),
            _esc(p.origin),
            _lattice_cell(p),
            win_cell,
        ]
        if with_cat:
            row.append(_catalog_cell(matches[i]))  # type: ignore[index]
        rows.append(row)
    head = [
        "Frequency (MHz)",
        "&sigma;<sub>f</sub> (kHz)",
        f"Amplitude ({_esc(uname)})",
        "SNR",
        "Origin",
        "Lattice",
        "Window",
    ]
    if with_cat:
        head.append("Catalog")
    return _table(head, rows, cls="final-list", row_attrs=row_attrs)


# ---------------------------------------------------------------------------
# Per-window page
# ---------------------------------------------------------------------------


def _window_peak_table(
    peaks: List[FinalPeak],
    uname: str,
    uval: float,
    matches: Optional[List[Optional[CatalogMatch]]] = None,
    *,
    window_id: Optional[int] = None,
) -> str:
    """The per-window fitted-lines table.

    Each peak carries a display letter (assigned low-to-high frequency, shared
    with the figure annotations and the fit-log table). The calibrated
    frequency, amplitude, phase, and SNR are shown in concise ``value(unc)``
    notation (uncertainty in last-digit units) -- the raw frequency and the
    broken-out &sigma; columns stay in plain fixed/scientific form. When a
    catalog cross-reference is supplied, a proximity-match badge column is added.

    When *window_id* is supplied, each row carries ``data-window`` /
    ``data-freq`` (the **raw** Stage-5 model frequency, the value the edit verbs
    match on -- not the &epsilon;-calibrated display value), and a trailing
    curation control column (Remove / Split / merge-checkbox) is appended. The
    column and attributes are inert markup; the in-report curation script reads
    them, and the curation-only column is CSS-hidden outside curation mode. The
    no-window-id form (used by the pure unit tests) emits neither.
    """
    from ..visualization.fit_detail import frequency_sorted_labels

    with_cat = matches is not None
    curate = window_id is not None
    letters = frequency_sorted_labels([float(p.frequency_mhz) for p in peaks])
    rows: List[List[str]] = []
    row_attrs: List[str] = []
    for i, (lbl, p) in enumerate(zip(letters, peaks)):
        sigma_f_mhz = None if p.sigma_f_khz is None else float(p.sigma_f_khz) / 1e3
        amp = None if p.amplitude is None else float(p.amplitude) / uval
        amp_err = None if p.amplitude_error is None else float(p.amplitude_error) / uval
        row = [
            _esc(lbl),
            _esc(_concise(float(p.frequency_mhz), sigma_f_mhz)),
            _esc(_freq(p.frequency_raw_mhz)),
            _esc(_g(p.sigma_f_khz, 3)),
            _esc(_g(p.sigma_stat_khz, 3)),
            _esc(_g(p.sigma_eps_khz, 3)),
            _esc("" if amp is None else _concise(amp, amp_err)),
            _esc("" if p.phase is None else _concise(float(p.phase), p.phase_error)),
            _esc("" if p.snr is None else _concise(float(p.snr), p.snr_error)),
            _esc(p.origin),
            _lattice_cell(p),
        ]
        if with_cat:
            row.append(_catalog_cell(matches[i]))  # type: ignore[index]
        if curate:
            row.append(_peak_curation_cell())
            raw = _freq(p.frequency_raw_mhz)
            row_attrs.append(f' data-window="{window_id}" data-freq="{_esc(raw)}"')
        rows.append(row)
    head = [
        "Peak",
        "Calibrated f (MHz)",
        "Raw f (MHz)",
        "&sigma;<sub>f</sub> (kHz)",
        "&sigma;<sub>stat</sub>",
        "&sigma;<sub>&epsilon;</sub>",
        f"Amplitude ({_esc(uname)})",
        "Phase (rad)",
        "SNR",
        "Origin",
        "Lattice",
    ]
    if with_cat:
        head.append("Catalog")
    if curate:
        head.append('<span class="cur-col-h">Curate</span>')
    return _table(head, rows, cls="peak-list", row_attrs=row_attrs if curate else None)


def _peak_curation_cell() -> str:
    """The trailing per-row curation control cell (Remove / Split / merge).

    Pure inert markup -- the curation script wires it by event delegation off the
    ``cur-btn`` class and the row's ``data-window`` / ``data-freq``. The whole
    column is CSS-hidden when the report is not in curation mode.
    """
    return (
        '<span class="cur-cell">'
        '<button type="button" class="cur-btn" data-act="remove">Remove</button>'
        '<button type="button" class="cur-btn" data-act="split">Split</button>'
        '<input type="number" class="cur-k" min="2" value="2" '
        'title="split into K" aria-label="split into K">'
        '<label class="cur-mergebox" title="select to merge">'
        '<input type="checkbox" class="cur-merge"> merge</label>'
        "</span>"
    )


def _window_curation_controls(window_id: int, lo: float, hi: float) -> List[str]:
    """The per-window curation controls under the fitted-lines table.

    The merge-selected button, a typed ``+ Add peak at <MHz>`` input, and an
    optional bare ``Mark reviewed`` accept. All wrapped in ``cur-only`` so the
    whole block is CSS-hidden outside curation mode; the curation script binds
    them by ``data-act`` and the enclosing ``<section>`` (which scopes the merge
    selection to this one window).
    """
    rng = f"{lo:.4f}&ndash;{hi:.4f} MHz"
    return [
        '<div class="cur-only cur-window-controls">',
        f'<button type="button" class="cur-btn" data-act="merge-selected" '
        f'data-window="{window_id}">Merge selected</button>',
        '<label class="cur-add">+ Add peak at '
        '<input type="number" class="cur-addfreq" step="0.0001" '
        f'placeholder="MHz"> '
        f'<button type="button" class="cur-btn" data-act="add-typed" '
        f'data-window="{window_id}">Add</button></label>',
        f'<span class="cur-range">window range {rng}</span>',
        f'<button type="button" class="cur-btn" data-act="accept" '
        f'data-window="{window_id}">Mark reviewed</button>',
        "</div>",
    ]


# --- parameter-symbol display ------------------------------------------------
# Covariance parameter labels are persisted as words (``amplitude_3`` /
# ``offset_3`` / ``phase_3`` / ``tau`` / ``baseline_re_2`` …). For display we map
# them to compact symbols and tie the per-peak index to the peak's display
# letter, so the covariance reads in the same alphabet as the fitted-lines table
# and the figure annotations -- and so a wide matrix actually fits.


def _param_symbol_parts(label: str, peak_letters: List[str]) -> Tuple[str, str, str]:
    """Return ``(base, superscript, subscript)`` tokens for a parameter label.

    ``base`` is a neutral token (``"phi"`` / ``"tau"`` resolve to Greek in the
    HTML/mathtext renderers). The per-peak index becomes the peak's display
    letter; baseline coefficients keep their numeric index.
    """

    def _sub_for_index(idx: str) -> str:
        return (
            peak_letters[int(idx)]
            if idx.isdigit() and int(idx) < len(peak_letters)
            else idx
        )

    if label == "tau":
        return ("tau", "", "")
    for kind, base, sup in (
        ("amplitude_", "A", ""),
        ("offset_", "f", "o"),
        ("phase_", "phi", ""),
    ):
        if label.startswith(kind):
            return (base, sup, _sub_for_index(label[len(kind) :]))
    for kind, sup in (("baseline_re_", "Re"), ("baseline_im_", "Im")):
        if label.startswith(kind):
            return ("c", sup, label[len(kind) :])
    return (label, "", "")


_GREEK_HTML = {"phi": "&phi;", "tau": "&tau;"}
_GREEK_MATHTEXT = {"phi": r"\phi", "tau": r"\tau"}


def _param_symbol_html(label: str, peak_letters: List[str]) -> str:
    base, sup, sub = _param_symbol_parts(label, peak_letters)
    out = _GREEK_HTML.get(base, _esc(base))
    if sup:
        out += f"<sup>{_esc(sup)}</sup>"
    if sub:
        out += f"<sub>{_esc(sub)}</sub>"
    return out


def _param_symbol_mathtext(label: str, peak_letters: List[str]) -> str:
    base, sup, sub = _param_symbol_parts(label, peak_letters)
    s = _GREEK_MATHTEXT.get(base, base)
    if sup:
        s += "^{" + sup + "}"
    if sub:
        s += "_{" + sub + "}"
    return f"${s}$"


def _param_value(
    label: str, wf: Any, center_mhz: float, sideband_sign: float
) -> Optional[float]:
    """The fitted value of one covariance parameter (None when unavailable).

    Amplitude / baseband offset / phase / tau are recovered from the fitted
    peaks and shared parameters; baseline polynomial coefficients are read from
    the per-window ``quality_metrics`` (``baseline_coeff{k}_re`` / ``_im``),
    where the leakage-wing baseline persists them.
    """
    peaks = wf.fitted_peaks
    if label == "tau":
        tv = wf.shared_parameters.get("tau_us", {})
        v = tv.get("value")
        return float(v) if v is not None else None
    for kind in ("amplitude", "offset", "phase"):
        prefix = kind + "_"
        if label.startswith(prefix):
            idx = label[len(prefix) :]
            if not idx.isdigit() or int(idx) >= len(peaks):
                return None
            p = peaks[int(idx)]
            if kind == "amplitude":
                return float(p.amplitude)
            if kind == "offset":
                return float(sideband_sign) * (float(p.frequency_mhz) - center_mhz)
            return float(p.phase) if p.phase is not None else None
    qm = getattr(wf, "quality_metrics", {}) or {}
    for prefix, suffix in (("baseline_re_", "_re"), ("baseline_im_", "_im")):
        if label.startswith(prefix):
            k = label[len(prefix) :]
            v = qm.get(f"baseline_coeff{k}{suffix}")
            return float(v) if v is not None else None
    return None


_COV_LEGEND = (
    '<p class="cov-legend"><em>Symbols: '
    "A = amplitude, f<sup>o</sup> = baseband offset (MHz), &phi; = phase (rad), "
    "&tau; = decay constant (&micro;s), "
    "c<sup>Re</sup><sub>k</sub> / c<sup>Im</sup><sub>k</sub> = baseline "
    "polynomial coefficients. Per-peak subscripts are the peak letters from the "
    "fitted-lines table.</em></p>"
)


def _variance_table(
    arr: np.ndarray, symbols: List[str], values: List[Optional[float]]
) -> str:
    """Per-parameter diagonal: fitted value, variance, and 1&sigma; std. dev."""
    diag = np.diag(arr).astype(float)
    rows: List[List[str]] = []
    for sym, val, var in zip(symbols, values, diag):
        std = float(np.sqrt(var)) if var >= 0.0 else float("nan")
        val_s = "&mdash;" if val is None else _esc(_g(val, 4))
        rows.append([sym, val_s, _esc(_g(float(var), 3)), _esc(_g(std, 3))])
    return _table(
        ["Parameter", "Value", "Variance", "Std. dev. (1&sigma;)"],
        rows,
        cls="variances",
    )


def _covariance_block(
    wf: Any,
    heatmap_name: Optional[str],
    sideband: Any,
) -> List[str]:
    """Present the per-window parameter (co)variance.

    Most-readable first: the diagonal fitted values / variances / standard
    deviations, the off-diagonal correlation coefficients as a divergent
    ``[-1, +1]`` heatmap (always shown when a covariance exists, since it stays
    legible for wide windows), and the full numeric covariance matrix below
    (suppressed past :data:`_COVARIANCE_RENDER_CAP` parameters). Parameters use
    compact symbols keyed to the peak letters; see :func:`_param_symbol_html`.
    """
    from ..visualization.fit_detail import frequency_sorted_labels

    cov = getattr(wf, "covariance", None)
    labels = getattr(wf, "covariance_param_labels", None)
    if cov is None or labels is None or len(labels) == 0:
        return [
            "<p><em>No parameter covariance was persisted for this window.</em></p>"
        ]
    arr = np.asarray(cov, dtype=float)
    n = len(labels)

    peak_letters = frequency_sorted_labels(
        [float(p.frequency_mhz) for p in wf.fitted_peaks]
    )
    symbols = [_param_symbol_html(lbl, peak_letters) for lbl in labels]
    lo, hi = wf.window.freq_range
    center = 0.5 * (float(lo) + float(hi))
    s = _sideband_sign(sideband)
    values = [_param_value(lbl, wf, center, s) for lbl in labels]

    out: List[str] = [
        _COV_LEGEND,
        "<h3>Variances</h3>",
        _variance_table(arr, symbols, values),
        "<h3>Correlation</h3>",
    ]
    if heatmap_name is not None:
        out.append(
            '<div class="cov-heatmap">'
            f'<img src="../figures/{heatmap_name}" '
            'alt="parameter correlation heatmap"></div>'
        )
    else:
        out.append("<p><em>Correlation heatmap unavailable.</em></p>")
    out.append("<h3>Covariance matrix</h3>")
    if n > _COVARIANCE_RENDER_CAP:
        out.append(
            f"<p><em>The full covariance is a {n}&times;{n} matrix &mdash; too "
            "wide to render inline; see the correlation heatmap above. The "
            "numeric matrix is available via the API "
            "(<code>FittingResult.covariance</code>).</em></p>"
        )
    else:
        rows: List[List[str]] = []
        for i in range(n):
            rows.append(
                [symbols[i]] + [_esc(_g(float(arr[i, j]), 3)) for j in range(n)]
            )
        out.append(_table([""] + symbols, rows, cls="covariance"))
    return out


def _ledger_block(
    candidates: List[LedgerCandidate], *, window_id: Optional[int] = None
) -> List[str]:
    if not candidates:
        return ["<p><em>No revivable ledger candidates for this window.</em></p>"]
    curate = window_id is not None
    rows: List[List[str]] = []
    row_attrs: List[str] = []
    for c in candidates:
        row = [
            _esc(_freq(c.frequency_mhz)),
            _esc(f"{c.seed_offset_mhz:+.4f}"),
            _esc(_g(c.best_evidence, 3)),
            _esc(c.evidence_kind),
            _esc(", ".join(c.decision_sites)),
            _esc("; ".join(c.reasons)),
        ]
        if curate:
            row.append(
                '<span class="cur-cell">'
                '<button type="button" class="cur-btn" data-act="add">Add</button>'
                "</span>"
            )
            raw = _freq(c.frequency_mhz)
            row_attrs.append(f' data-window="{window_id}" data-freq="{_esc(raw)}"')
        rows.append(row)
    head = [
        "Frequency (MHz)",
        "Seed offset (MHz)",
        "Evidence",
        "Kind",
        "Decision sites",
        "Reasons",
    ]
    if curate:
        head.append('<span class="cur-col-h">Curate</span>')
    return [_table(head, rows, cls="ledger", row_attrs=row_attrs if curate else None)]


# Decisions that leave a peak in the model (advance K) vs. those that do not.
_AUDIT_ADDS = {"seed", "seed-blend", "accept"}

_AUDIT_LEGEND = (
    '<p class="audit-legend"><em>'
    "<strong>seed</strong>: the initial single-peak (K=1) fit at the brightest "
    "feature. <strong>seed-blend</strong>: a blend-aware re-seed that adds a "
    "straddling component to the same feature (raising K). <strong>accept</strong>: "
    "a further candidate line cleared the add-one gate and was kept. "
    "<strong>reject</strong>: a candidate was tried and removed, so K is unchanged. "
    "<strong>knockout-null</strong>: the lone seed failed its K=1-vs-null test and "
    "the window was emptied. The <em>K</em> column is the model size after the step; "
    "the run ends at the fitted-lines model listed above."
    "</em></p>"
)


_MERGE_LEGEND = (
    '<p class="audit-legend"><em>This window was rebuilt by an automatic '
    "degenerate-pair <strong>merge</strong> (VIF collapse): each row is a "
    "near-coincident line pair whose amplitudes were too correlated to resolve "
    "(variance-inflation factor VIF), replaced by the single merged line. The "
    "joint refit that follows a merge does not carry an add-one-peak history, so "
    "the merge record below is the provenance.</em></p>"
)


def _merge_block(merges: List[Dict[str, Any]]) -> List[str]:
    """Render a window's VIF-collapse merge records (the auto-merge provenance).

    Surfaced when the add-one-peak ``audit_trail`` is empty because the window
    was rebuilt by an auto-merge joint refit; the records come from the Stage 5
    ``vif_collapse`` diagnostics keyed by window.
    """
    rows: List[List[str]] = []
    for m in merges:
        vif = (
            f"{_g(m.get('vif_a'), 3)} / {_g(m.get('vif_b'), 3)}"
            if m.get("vif_a") is not None
            else "&mdash;"
        )
        rows.append(
            [
                _esc(_freq(m.get("merged_frequency_mhz"))),
                _esc(_freq(m.get("frequency_a_mhz"))),
                _esc(_freq(m.get("frequency_b_mhz"))),
                _esc(_g(m.get("separation_res"), 3)),
                vif,
            ]
        )
    head = [
        "Merged f (MHz)",
        "Line A (MHz)",
        "Line B (MHz)",
        "Separation (res. elem.)",
        "VIF A / B",
    ]
    return [_table(head, rows, cls="audit"), _MERGE_LEGEND]


def _audit_block(
    wf: Any,
    center_mhz: float,
    sideband_sign: float,
    merges: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """Render the conservative add-one-peak history as a legible table.

    One row per :class:`AuditStep`: the decision, the candidate's molecular
    frequency (and baseband offset), the running model size *K* after the step,
    the &chi;&sup2; it moved from / to, the F-test p-value, and -- most
    importantly -- the plain-language ``reason`` (which the raw fit-log text
    drops). A legend below explains the decision keywords and points at the final
    model so a seed/seed-blend sequence's outcome is unambiguous. When the
    window has no add-one history because it was rebuilt by an auto-merge,
    *merges* (its VIF-collapse records) are rendered instead.
    """
    audit = getattr(wf, "audit_trail", None) or []
    if not audit:
        if merges:
            return _merge_block(merges)
        return [
            "<p><em>No add-one-peak history was recorded for this window "
            "(e.g. a user-edited window rebuilt by a joint refit).</em></p>"
        ]
    rows: List[List[str]] = []
    k = 0
    for i, s in enumerate(audit):
        if s.decision == "knockout-null":
            k = 0
        elif s.decision in _AUDIT_ADDS:
            k += 1
        # reject / unknown: K unchanged.
        freq = center_mhz + float(sideband_sign) * float(s.candidate_offset_mhz)
        dchi2 = float(s.chi2_before) - float(s.chi2_after)
        rows.append(
            [
                str(i),
                _esc(s.decision),
                _esc(_freq(freq)),
                _esc(f"{s.candidate_offset_mhz:+.4f}"),
                str(k),
                f"{_esc(_g(s.chi2_before, 4))} &rarr; {_esc(_g(s.chi2_after, 4))}",
                _esc(_g(dchi2, 4)),
                _esc(_g(s.p_value, 2)),
                _esc(s.reason),
            ]
        )
    head = [
        "Step",
        "Decision",
        "Frequency (MHz)",
        "Offset (MHz)",
        "K",
        "&chi;&sup2; (before &rarr; after)",
        "&Delta;&chi;&sup2;",
        "p-value",
        "Reason",
    ]
    return [_table(head, rows, cls="audit"), _AUDIT_LEGEND]


def _decision_block(decisions: List[DecisionLogEntry]) -> List[str]:
    if not decisions:
        return []
    rows: List[List[str]] = []
    for d in decisions:
        rows.append(
            [
                f"{d.order_index}",
                _esc(d.kind),
                _esc(_freq(d.frequency_mhz)),
                _esc(d.provenance),
            ]
        )
    return [
        "<h3>User decisions</h3>",
        _table(["#", "Kind", "Anchor (MHz)", "Provenance"], rows, cls="decisions"),
    ]


def _attention_block(status: Optional[WindowReviewStatus]) -> List[str]:
    if status is None or not status.attention_reasons:
        return []
    items = [
        f"<li><strong>{_esc(r.kind)}</strong> &mdash; {_esc(r.detail)} "
        f"(severity {r.severity:.2g})</li>"
        for r in sorted(
            status.attention_reasons, key=lambda r: r.severity, reverse=True
        )
    ]
    return ["<h3>Attention</h3>", "<ul>", *items, "</ul>"]


def _mag_axes_geometry(fig: Any, *, dpi: int) -> Optional[Dict[str, float]]:
    """Capture the |X| panel's data-axes geometry for the click-to-add overlay.

    Returns the data-axes bounding box in *saved-PNG pixels* (the natural image
    size, ``figsize * dpi``) plus the molecular frequency at the box's left and
    right edges, so a click x over the rendered ``<img>`` inverts to a molecular
    MHz by linear interpolation -- carrying the axis sense (ascending or
    descending) implicitly in ``flo`` / ``fhi``. Must be called *after* the
    figure is drawn (``savefig`` forces the ``constrained_layout`` pass that
    finalizes ``ax.get_position()``); the panels are saved without
    ``bbox_inches="tight"``, so the figure-fraction box maps cleanly to the PNG.
    Returns ``None`` when the data axis can't be identified.
    """
    ax = next((a for a in fig.axes if a.get_xlabel().startswith("frequency")), None)
    if ax is None:
        return None
    pos = ax.get_position()  # figure-fraction, post-layout
    w_in, h_in = (float(v) for v in fig.get_size_inches())
    w_px, h_px = w_in * dpi, h_in * dpi
    flo, fhi = (float(v) for v in ax.get_xlim())
    return {
        "x0": pos.x0 * w_px,
        "x1": pos.x1 * w_px,
        # Image y grows downward; the matplotlib position fraction grows upward.
        "y0": (1.0 - pos.y1) * h_px,
        "y1": (1.0 - pos.y0) * h_px,
        "w": w_px,
        "h": h_px,
        "flo": flo,
        "fhi": fhi,
    }


# Per-window figure rendering is the dominant report cost (matplotlib savefig /
# constrained-layout over ~4 independent panels per window) and is embarrassingly
# parallel. The render result for one window: the panel PNG bytes keyed by panel,
# the |X| panel's post-layout axes geometry (for the curation overlay), and the
# optional correlation-heatmap PNG bytes. Bytes (not files) travel back so the
# parent owns all figure-file IO and the encoding stays byte-identical to the
# serial path. ``None`` workers/auto-sizing lets a test pin the worker count.
_WindowFigures = Tuple[
    int,  # window id
    Dict[str, bytes],  # panel name -> PNG bytes
    Optional[Dict[str, float]],  # |X| panel axes geometry
    Optional[bytes],  # correlation-heatmap PNG bytes
    Optional[bytes],  # hover thumbnail PNG bytes (mag panel, downscaled)
]
_FIGURE_RENDER_WORKERS: Optional[int] = (
    None  # None => auto (cpu_count - 2); 1 => serial
)
_WORKER_RENDER_CTX: Optional[Dict[str, Any]] = None


def _render_one_window_figures(
    wid: int, *, path: str, bundle: Any, dpi: int, stem: str
) -> _WindowFigures:
    """Render one window's zoomed panels + correlation heatmap to PNG bytes.

    Pure function of the read-only ``bundle`` (and the persisted spurs it
    carries); shared by the serial and the process-pool paths so both produce
    identical bytes.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..visualization.fit_detail import (
        frequency_sorted_labels,
        plot_correlation_heatmap,
    )
    from .stage5_impl import render_fit_panels_impl

    panels = render_fit_panels_impl(path, wid, bundle=bundle, with_overview=False)
    panel_bytes: Dict[str, bytes] = {}
    mag_geom: Optional[Dict[str, float]] = None
    for panel in _PANEL_ORDER:
        pfig = panels.get(panel)
        if pfig is None:
            continue
        if panel == "overview":
            # The full-spectrum context is the single shared interactive overview,
            # not a per-window image; discard any overview panel.
            plt.close(pfig)
            continue
        panel_bytes[panel] = _figure_png_bytes(pfig, dpi=dpi)
        # The |X| panel's data-axes geometry (post-savefig layout) powers the
        # curation click-to-add overlay; capture before closing the figure.
        if panel == "mag":
            mag_geom = _mag_axes_geometry(pfig, dpi=dpi)
        plt.close(pfig)

    # Correlation heatmap (when a covariance was persisted) -- a divergent
    # [-1, +1] image that stays readable where the numeric matrix does not.
    wf = bundle.fit.window_fit(wid)
    cov = getattr(wf, "covariance", None)
    cov_labels = getattr(wf, "covariance_param_labels", None)
    corr_bytes: Optional[bytes] = None
    if cov is not None and cov_labels:
        peak_letters = frequency_sorted_labels(
            [float(p.frequency_mhz) for p in wf.fitted_peaks]
        )
        sym_labels = [_param_symbol_mathtext(lbl, peak_letters) for lbl in cov_labels]
        hfig = plot_correlation_heatmap(cov, sym_labels)
        # bbox_inches="tight" so the wide mathtext axis labels (the baseline
        # coefficient stacks) are never clipped at the figure edge.
        corr_bytes = _figure_png_bytes(hfig, dpi=dpi, bbox_inches="tight")
        plt.close(hfig)

    # The hover-preview thumbnail is the |X| panel downscaled; doing it here (in
    # the render worker, off the panel bytes we already hold) parallelizes the
    # O(N_windows) downscale that the single-file collapse otherwise runs serially.
    thumb_bytes: Optional[bytes] = None
    mag_png = panel_bytes.get("mag")
    if mag_png is not None:
        thumb_bytes = _downscale_thumb_bytes(mag_png)
    return wid, panel_bytes, mag_geom, corr_bytes, thumb_bytes


def _render_window_worker(wid: int) -> _WindowFigures:
    """Process-pool entry point: render ``wid`` from the fork-inherited context."""
    ctx = _WORKER_RENDER_CTX
    assert ctx is not None  # set in the parent before the pool forks
    return _render_one_window_figures(
        wid, path=ctx["path"], bundle=ctx["bundle"], dpi=ctx["dpi"], stem=ctx["stem"]
    )


def _render_all_window_figures(
    *,
    path: str,
    bundle: Any,
    dpi: int,
    stem: str,
    page_ids: List[int],
    jobs: Optional[int] = None,
) -> Dict[int, _WindowFigures]:
    """Render every page window's figures, in parallel when worthwhile.

    Returns ``{wid: (wid, panel_bytes, mag_geom, corr_bytes)}``. Uses a forking
    process pool (the read-only ``bundle`` -- ~200k-1.2M-point arrays -- is
    inherited via fork, never pickled per task; only the int window id goes out
    and the PNG bytes come back) capped at ``cpu_count - 2``. Falls back to an
    in-process serial render for a single window, when only one worker is
    available, or when the platform lacks ``fork``. The figures are deterministic
    at fixed DPI, so the parallel and serial outputs are byte-identical.
    """
    n = len(page_ids)

    def _log_progress(i: int, _n: int) -> None:
        # The ordered per-window progress signal on long report runs.
        logger.info("window %d/%d", i, n)

    global _WORKER_RENDER_CTX
    _WORKER_RENDER_CTX = {"path": path, "bundle": bundle, "dpi": dpi, "stem": stem}
    try:
        rendered = fork_map(
            page_ids,
            _render_window_worker,
            jobs=jobs,
            override=_FIGURE_RENDER_WORKERS,
            progress=_log_progress,
        )
    finally:
        _WORKER_RENDER_CTX = None
    return {res[0]: res for res in rendered}


def _fit_panels_block(
    panel_files: Dict[str, str],
    *,
    window_id: Optional[int] = None,
    mag_geom: Optional[Dict[str, float]] = None,
) -> List[str]:
    """Lay the per-window panel PNGs out for the 1:2:2 responsive grid.

    The overview spans full width on top; the Re / Im model+residual panels
    share a two-column grid row and the |X| + residual histogram share the row
    below. The grid collapses to a single column on a narrow viewport (CSS), so
    each panel takes the full width in turn. Each entry is rendered only when its
    PNG was produced, so a degenerate window with missing panels still yields
    valid markup.

    When *window_id* and *mag_geom* are supplied, the magnitude panel becomes the
    curation click surface: its ``<img>`` carries ``data-window`` + the full
    ``data-axes-*`` geometry (pixel box, frequency limits) so the overlay can map
    between a click x and a molecular MHz both ways. The panel is wrapped with a
    corner *arm* toggle (click-to-add fires only on the armed plot, so routine
    navigation never adds) and an SVG layer the script fills with one marker per
    queued edit in this window.
    """

    def _fig(panel: str, alt: str, cls: str = "panel") -> List[str]:
        name = panel_files.get(panel)
        if name is None:
            return []
        return [
            f'  <figure class="{cls}">'
            f'<img src="../figures/{name}" alt="{_esc(alt)}"></figure>'
        ]

    def _mag_fig() -> List[str]:
        name = panel_files.get("mag")
        if name is None:
            return []
        if window_id is None or mag_geom is None:
            return _fig("mag", "magnitude + residual")
        g = mag_geom
        attrs = (
            f' class="cur-plot" data-window="{window_id}"'
            f' data-axes-x0="{g["x0"]:.2f}" data-axes-x1="{g["x1"]:.2f}"'
            f' data-axes-y0="{g["y0"]:.2f}" data-axes-y1="{g["y1"]:.2f}"'
            f' data-axes-w="{g["w"]:.2f}" data-axes-h="{g["h"]:.2f}"'
            f' data-axes-flo="{g["flo"]:.6f}" data-axes-fhi="{g["fhi"]:.6f}"'
        )
        return [
            '  <figure class="panel">'
            f'<div class="cur-plot-wrap" data-window="{window_id}">'
            f'<img src="../figures/{name}" alt="magnitude + residual"{attrs}>'
            '<button type="button" class="cur-plot-arm cur-only" '
            'title="arm click-to-add on this panel">+ add</button>'
            f'<svg class="cur-plot-svg cur-only" data-window="{window_id}" '
            f'viewBox="0 0 {g["w"]:.0f} {g["h"]:.0f}" preserveAspectRatio="none" '
            'aria-hidden="true"></svg>'
            "</div></figure>"
        ]

    out = ['<div class="fit-panels">']
    out += _fig("overview", "full-spectrum context", "panel panel-overview")
    grid = (
        _fig("re", "real part + residual")
        + _fig("im", "imaginary part + residual")
        + _mag_fig()
        + _fig("hist", "residual histogram")
    )
    if grid:
        out.append('  <div class="panel-grid">')
        out += grid
        out.append("  </div>")
    out.append("</div>")
    return out


def _window_page(
    *,
    stem: str,
    window_id: int,
    wf: Any,
    peaks: List[FinalPeak],
    uname: str,
    uval: float,
    status: Optional[WindowReviewStatus],
    ledger: List[LedgerCandidate],
    decisions: List[DecisionLogEntry],
    panel_files: Dict[str, str],
    cov_heatmap_name: Optional[str],
    sideband: Any,
    prev_id: Optional[int],
    next_id: Optional[int],
    catalog_matches: Optional[List[Optional[CatalogMatch]]] = None,
    merges: Optional[List[Dict[str, Any]]] = None,
    nav_rows: Optional[
        List[Tuple[int, float, float, int, Optional[float], bool, bool]]
    ] = None,
    band: Optional[Tuple[float, float]] = None,
    overview_name: Optional[str] = None,
    mag_geom: Optional[Dict[str, float]] = None,
) -> str:
    lo, hi = wf.window.freq_range
    tau = float(wf.shared_parameters.get("tau_us", {}).get("value", 0.0))
    shape = getattr(wf, "shape", "lorentzian")
    chi2r = float(wf.reduced_chi2)

    nav = ['<div class="nav">', '<a href="../index.html">&larr; index</a>']
    if prev_id is not None:
        nav.append(
            f'<a href="{_window_page_name(prev_id)}">&larr; window {prev_id}</a>'
        )
    if next_id is not None:
        nav.append(
            f'<a href="{_window_page_name(next_id)}">window {next_id} &rarr;</a>'
        )
    nav.append("</div>")

    # Spectrum-context section: the interactive full-spectrum overview with this
    # window highlighted -- the same shared image + clickable per-window overlay
    # the index uses (here this window gets the green "you are here"). Rendered
    # only when the shared overview image was produced.
    context: List[str] = []
    if nav_rows and band is not None and overview_name is not None:
        context.append("<h2>Spectrum context</h2>")
        context.append(
            _spectrum_nav(
                nav_rows,
                band,
                stem,
                link_prefix="",
                thumb_prefix="../figures/",
                current_id=window_id,
            )
        )
        context.append(
            '<p class="winmap-hint">The shaded bands are fit windows (orange = '
            "attention, green = this one); hover for details, click to jump.</p>"
        )

    body: List[str] = [
        *nav,
        f"<h1>{_esc(stem)} &mdash; window {window_id}</h1>",
        '<ul class="summary">',
        f"<li><strong>Range:</strong> {min(lo, hi):.4f}&ndash;{max(lo, hi):.4f} "
        "MHz</li>",
        f"<li><strong>Peaks:</strong> {len(wf.fitted_peaks):,}</li>",
        f"<li><strong>&chi;&sup2;<sub>r</sub>:</strong> "
        f"{_esc(_md_num(chi2r, 4))}</li>",
        f"<li><strong>&tau;:</strong> {_esc(_md_num(tau, 4))} &micro;s</li>",
        f"<li><strong>Shape:</strong> {_esc(shape)}</li>",
        "</ul>",
        *_attention_block(status),
        *context,
        "<h2>Fit</h2>",
        *_fit_panels_block(panel_files, window_id=window_id, mag_geom=mag_geom),
        "<h2>Fitted lines</h2>",
        _window_peak_table(peaks, uname, uval, catalog_matches, window_id=window_id),
        *_window_curation_controls(window_id, min(lo, hi), max(lo, hi)),
        "<h2>Parameter covariance</h2>",
        *_covariance_block(wf, cov_heatmap_name, sideband),
        "<h2>Fit history</h2>",
        *_audit_block(
            wf, 0.5 * (float(lo) + float(hi)), _sideband_sign(sideband), merges
        ),
        "<h2>Ledger candidates</h2>",
        *_ledger_block(ledger, window_id=window_id),
        *_decision_block(decisions),
    ]
    if context:
        body.append(_WINMAP_JS)  # hover-zoom popup; the map works without it
    return _page(f"{stem} window {window_id}", body, css_href="../assets/style.css")


# ---------------------------------------------------------------------------
# Single-file collapse (base64-embedded, self-contained HTML)
# ---------------------------------------------------------------------------

REPORT_SCOPES = ("summary", "full")

# Hover-preview thumbnails are downscaled to a little above the popup's display
# width (.winmap-pop img is 360px) before base64 embedding, so the deduplicated
# thumbnail map stays small.
_THUMB_MAX_W = 400


def _downscale_thumb_bytes(png_bytes: bytes) -> bytes:
    """Downscale a panel PNG to the hover-preview thumbnail width, as PNG bytes.

    Sink-agnostic (operates on bytes), so a render worker can produce the
    thumbnail off the panel it just drew -- parallelizing the O(N_windows)
    downscale that otherwise runs serially in the collapse -- and the result is
    byte-identical to downscaling the same PNG read back from disk.
    """
    import io

    from PIL import Image

    with Image.open(io.BytesIO(png_bytes)) as src:
        img = src.convert("RGB")
        if img.width > _THUMB_MAX_W:
            h = round(img.height * _THUMB_MAX_W / img.width)
            img = img.resize((_THUMB_MAX_W, h), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _collapse_site_to_single_file(
    site_dir: Path,
    *,
    mode: str,
    stem: str,
    figure_store: Dict[str, bytes],
    thumb_store: Dict[str, bytes],
    page_store: Dict[str, str],
    css: str,
) -> str:
    """Collapse the assembled report model into one self-contained HTML document.

    Inlines the stylesheet, base64-embeds every ``<img>`` figure, and rewrites
    cross-page links to in-document anchors. ``mode="summary"`` keeps the index
    and methods pages (the portable methods-and-results replacement);
    ``mode="full"`` also folds in every per-window page, reachable via
    ``#window-<id>``.

    The pages, per-window figures, thumbnails, and stylesheet resolve from the
    in-memory stores; only the O(1) methods-page figures live on disk under
    ``site_dir/figures`` (``read_png`` falls back there for them).

    The hover-zoom *thumbnails* (``data-thumb``) are keyed to a single
    deduplicated, downscaled base64 map (``window.__thumbs``) rather than inlined
    per reference -- each window thumbnail is referenced on many rows and every
    page, so a map keeps one small copy instead of hundreds.
    """
    import base64
    import json

    img_re = re.compile(r'src="([^"]+\.png)"')
    thumb_re = re.compile(r'data-thumb="[^"]*/([^"/]+\.png)"')
    script_re = re.compile(r"<script\b[^>]*>.*?</script>", re.DOTALL)
    main_open = '<main class="report">'
    thumb_keys: set[str] = set()

    # Per-window figures resolve from ``figure_store``; the few methods-page
    # figures stay on disk, so the resolver checks the store then the figure dir.
    def read_png(name: str) -> Optional[bytes]:
        if name in figure_store:
            return figure_store[name]
        fp = site_dir / "figures" / name
        if fp.exists():
            return fp.read_bytes()
        return None

    def data_uri_name(name: str) -> Optional[str]:
        raw = read_png(name)
        if raw is None:
            return None
        return f"data:image/png;base64,{base64.b64encode(raw).decode('ascii')}"

    # The interactive-overview background is a CSS ``url(../figures/...)``; embed
    # it once as a data URI (the rest of the stylesheet has no url() refs), so the
    # shared overview image is inlined a single time for the whole document.
    def _embed_css_url(m: "re.Match[str]") -> str:
        uri = data_uri_name(m.group(1))
        return f"url({uri})" if uri is not None else m.group(0)

    css = re.sub(r"url\((?:\.\./)?figures/([^)]+\.png)\)", _embed_css_url, css)

    def thumb_data_uri(name: str) -> str:
        # Prefer a worker-precomputed (already downscaled) thumbnail; otherwise
        # downscale the full panel here. Both yield identical bytes.
        tb = thumb_store.get(name)
        if tb is None:
            png = read_png(name)
            if png is None:
                return ""
            tb = _downscale_thumb_bytes(png)
        return f"data:image/png;base64,{base64.b64encode(tb).decode('ascii')}"

    def rewrite_links(frag: str) -> str:
        if mode == "full":
            frag = re.sub(
                r'href="(?:\.\./)?(?:windows/)?window_0*(\d+)\.html"',
                lambda m: f'href="#window-{int(m.group(1))}"',
                frag,
            )
        else:  # summary has no window pages -- send window links to the top
            frag = re.sub(
                r'href="(?:\.\./)?(?:windows/)?window_0*\d+\.html"',
                'href="#page-top"',
                frag,
            )
        frag = re.sub(r'href="(?:\.\./)?index\.html"', 'href="#page-top"', frag)
        frag = re.sub(r'href="(?:\.\./)?methods\.html"', 'href="#methods"', frag)
        return frag

    def to_thumb_key(m: "re.Match[str]") -> str:
        name = m.group(1)
        thumb_keys.add(name)
        return f'data-thumb="{name}"'

    def prep(html: str) -> str:
        i = html.find(main_open)
        j = html.rfind("</main>")
        frag = html[i + len(main_open) : j]
        frag = script_re.sub("", frag)  # one popup script is re-added globally
        frag = thumb_re.sub(to_thumb_key, frag)  # key into the dedup thumb map
        frag = rewrite_links(frag)

        def embed(m: "re.Match[str]") -> str:
            uri = data_uri_name(Path(m.group(1)).name)
            return f'src="{uri}"' if uri is not None else m.group(0)

        return img_re.sub(embed, frag)

    sections = [
        f'<section id="page-top">{prep(page_store.get("index.html") or "")}</section>'
    ]
    methods_text = page_store.get("methods.html")
    if methods_text is not None:
        sections.append(f'<section id="methods">{prep(methods_text)}</section>')
    window_ids: List[int] = []
    if mode == "full":
        win_pages = sorted(
            k for k in page_store if re.fullmatch(r"windows/window_\d+\.html", k)
        )
        for key in win_pages:
            wid = int(re.search(r"window_0*(\d+)\.html", key).group(1))  # type: ignore[union-attr]
            window_ids.append(wid)
            sections.append(
                f'<section id="window-{wid}" class="embedded-window">'
                f"{prep(page_store[key])}</section>"
            )

    # One deduplicated, downscaled base64 thumbnail per referenced window, looked
    # up by the hover popup (window.__thumbs); built after prep() has collected
    # every key.
    thumb_map = {name: thumb_data_uri(name) for name in sorted(thumb_keys)}
    thumb_script = (
        f"<script>window.__thumbs={json.dumps(thumb_map)};</script>"
        if thumb_map
        else ""
    )

    # Sticky section navigation. Overview, the window list, and the final line
    # list are on the index (present in both modes); Methods when that page is
    # folded in; the jump-to-window picker only when the per-window pages are.
    # Everything anchors to in-document section ids.
    nav_links = ['<a href="#page-top">Overview</a>']
    if methods_text is not None:
        nav_links.append('<a href="#methods">Methods</a>')
    nav_links.append('<a href="#window-list">Windows</a>')
    nav_links.append('<a href="#final-list">Line list</a>')
    if window_ids:
        opts = "".join(
            f'<option value="#window-{w}">window {w}</option>' for w in window_ids
        )
        nav_links.append(
            '<select class="winjump" aria-label="Jump to window" '
            'onchange="if(this.value)location.hash=this.value">'
            f'<option value="">Jump to window…</option>{opts}</select>'
        )
    # The compact toggle shrinks every figure to a thumbnail to speed scrolling;
    # the report opens in full view, and the button is inert without scripting.
    nav_links.append('<button class="compact-toggle" type="button">Compact</button>')
    # The curation toggle flips between the read-only (default) and editable
    # views; the badge tracks the cart size. Both are inert without scripting,
    # and the controls they reveal are CSS-hidden until the toggle is engaged.
    nav_links.append('<button class="cur-toggle" type="button">Curate</button>')
    nav_links.append('<span class="cur-badge">cart (0)</span>')
    topnav = (
        '<nav class="topnav"><div class="topnav-inner">'
        f'<a class="brand" href="#page-top">{_esc(stem)}</a>'
        f'{"".join(nav_links)}</div></nav>'
    )

    return "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="en" class="report-single">',
            "<head>",
            '  <meta charset="utf-8">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1">',
            f"  <title>{_esc(stem)} report</title>",
            f"  <style>{css}</style>",
            _MATHJAX_HEAD,
            "</head>",
            "<body>",
            topnav,
            '<main class="report">',
            *sections,
            "</main>",
            thumb_script,
            f"<script>window.__stem={json.dumps(stem)};</script>",
            _WINMAP_JS,
            _COMPACT_JS,
            _CURATION_JS,
            "</body>",
            "</html>",
            "",
        ]
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class _ReportModel:
    """In-memory result of assembling a report site.

    Carries the figure bytes (per-window panels + correlation heatmaps), the
    per-window hover thumbnails (downscaled in the render workers), the page
    HTML, and the stylesheet -- everything :func:`_collapse_site_to_single_file`
    needs to build the single self-contained file *without* a disk round-trip.
    The few methods-page figures (per-stage diagnostics) and the shared overview
    are O(1) and stay on disk under ``out_root``. ``stem`` is the file stem.
    """

    stem: str
    figure_store: Dict[str, bytes] = field(default_factory=dict)
    thumb_store: Dict[str, bytes] = field(default_factory=dict)
    page_store: Dict[str, str] = field(default_factory=dict)
    css: str = ""


def _assemble_report_site(
    file_path: Union[Path, str],
    *,
    out_root: Union[Path, str],
    windows: str = "all",
    dpi: int = 110,
    catalog: Optional[Union[Path, str]] = None,
    catalog_n_sigma: float = 3.0,
    jobs: Optional[int] = None,
) -> _ReportModel:
    """Assemble the report into an in-memory :class:`_ReportModel`.

    Builds the index page, the methods page, and one page per window, each as an
    HTML string in the returned model alongside the per-window figures (the O(N)
    bulk) and their hover thumbnails;
    :func:`_collapse_site_to_single_file` folds them into the one self-contained
    file :func:`report_full_impl` ships. Only the few O(1) methods-page figures
    (the shared spectrum overview, the distribution histograms, the per-stage
    diagnostics) land on disk under ``out_root/figures``; the collapse reads them
    back from there. See :func:`report_full_impl` for the parameter semantics.

    Raises ``ValueError`` if *windows* is unknown, no final-products table is
    present, or *catalog* is given but unreadable / empty.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..visualization.fit_detail import (
        plot_magnitude_histogram,
        plot_summary_histograms,
    )
    from .report_impl import _render_markdown
    from .stage5_impl import _resolve_detail_bundle
    from .stage6_impl import get_candidate_ledger_impl

    key = str(windows).lower()
    if key not in VALID_WINDOW_FILTERS:
        raise ValueError(
            f"unknown windows filter {windows!r}; choose one of "
            f"{VALID_WINDOW_FILTERS}"
        )

    path = str(file_path)
    review = load_stage6_review_from_file(path)
    products = review.final_products
    if products is None:
        raise ValueError(
            "No final-products table found in this file. Run 'review run' first "
            "to consolidate the calibrated final products."
        )

    model = assemble_summary_model(path)
    bundle = _resolve_detail_bundle(path)
    stem = bundle.file_stem
    uname, uval = _amplitude_unit(products)

    # Catalog cross-reference (optional): matches are parallel to
    # products.peaks; key them by peak identity so the same FinalPeak objects in
    # peaks_by_window resolve to their match without re-running the test.
    cross_ref = load_cross_ref(products.peaks, catalog, catalog_n_sigma)
    match_by_peak: Dict[int, Optional[CatalogMatch]] = {}
    if cross_ref is not None:
        match_by_peak = {
            id(p): cross_ref.matches[i] for i, p in enumerate(products.peaks)
        }

    # Available windows (those with attached context), ascending.
    win_fits = {
        int(wf.window_id): wf
        for wf in bundle.fit.window_fits
        if wf.window is not None and wf.window_id is not None
    }
    all_ids = sorted(win_fits)
    attention_ids = {
        wid
        for wid, st in review.window_statuses.items()
        if st.needs_attention and wid in win_fits
    }
    page_ids = all_ids if key == "all" else [w for w in all_ids if w in attention_ids]
    page_set = set(page_ids)

    # Window-map rows + band, built once and shared by the index and every
    # window page's quick-nav strip.
    index_rows: List[Tuple[int, float, float, int, Optional[float], bool, bool]] = [
        (
            wid,
            *_window_range(win_fits[wid]),
            len(win_fits[wid].fitted_peaks),
            _opt_chi2(win_fits[wid]),
            wid in page_set,
            wid in attention_ids,
        )
        for wid in all_ids
    ]
    nav_band: Optional[Tuple[float, float]] = None
    if index_rows:
        if bundle.trim_mhz is not None:
            nav_band = (float(min(bundle.trim_mhz)), float(max(bundle.trim_mhz)))
        else:
            nav_band = (
                min(min(r[1], r[2]) for r in index_rows),
                max(max(r[1], r[2]) for r in index_rows),
            )

    # Group consolidated final peaks + user decisions by window.
    peaks_by_window: Dict[int, List[FinalPeak]] = {}
    for p in products.peaks:
        if p.window_id is not None:
            peaks_by_window.setdefault(int(p.window_id), []).append(p)
    decisions_by_window: Dict[int, List[DecisionLogEntry]] = {}
    for d in review.decision_log:
        decisions_by_window.setdefault(int(d.window_id), []).append(d)

    # Auto-merge (VIF-collapse) provenance, keyed by window: a merged window's
    # joint refit carries no add-one-peak history, so this is its fit history.
    merges_by_window: Dict[int, List[Dict[str, Any]]] = {}
    for rec in (
        (bundle.fit.diagnostics or {}).get("vif_collapse", {}).get("collapses", [])
    ):
        wid_rec = rec.get("window_id")
        if wid_rec is not None:
            merges_by_window.setdefault(int(wid_rec), []).append(rec)

    out_root = Path(out_root)
    # Only the O(1) methods-page figures + the shared overview land on disk under
    # out_root/figures; the collapse reads them back from there. The per-window
    # pages and figures stay in the in-memory model.
    (out_root / "figures").mkdir(parents=True, exist_ok=True)

    site = _ReportModel(stem=stem)

    # The interactive full-spectrum overview image, rendered ONCE (attention
    # windows shaded into it): it backs both the index "Spectrum" section and
    # every window page's "Spectrum context" via a clickable per-window overlay,
    # so the full-spectrum image is rendered and embedded once rather than
    # re-rendered per window.
    attention_ranges = [_window_range(win_fits[w]) for w in sorted(attention_ids)]
    overview_name: Optional[str] = None
    if index_rows and nav_band is not None:
        try:
            ov_fig = _plot_index_overview(bundle, attention_ranges)
            overview_name = f"{stem}_overview.png"
            _save_figure_png(ov_fig, out_root / "figures" / overview_name, dpi=dpi)
            plt.close(ov_fig)
        except (ValueError, KeyError):
            overview_name = None
    # The stylesheet, plus the per-build rule binding that overview as the
    # interactive-overview background. The single-file path embeds it from the
    # model.
    site.css = _STYLESHEET + _spectrum_ctx_css(overview_name)

    # --- per-window figures (parallel) ----------------------------------
    # Figure rendering is the dominant report cost and the windows are
    # independent, so render them across a forking process pool; the returned
    # PNG + thumbnail bytes go into the model and the single-file path embeds
    # them straight from memory.
    n_pages = len(page_ids)
    logger.info("rendering %d window figure sets", n_pages)
    rendered = _render_all_window_figures(
        path=path, bundle=bundle, dpi=dpi, stem=stem, page_ids=page_ids, jobs=jobs
    )
    panel_files_by_wid: Dict[int, Dict[str, str]] = {}
    mag_geom_by_wid: Dict[int, Optional[Dict[str, float]]] = {}
    cov_heatmap_by_wid: Dict[int, Optional[str]] = {}
    for wid in page_ids:
        _, panel_bytes, mag_geom, corr_bytes, thumb_bytes = rendered[wid]
        panel_files: Dict[str, str] = {}
        for panel, data in panel_bytes.items():
            fname = _panel_figure_name(stem, wid, panel)
            site.figure_store[fname] = data
            panel_files[panel] = fname
        # The hover thumbnail (worker-downscaled off the |X| panel) keys on the
        # mag panel's basename, matching the data-thumb the pages emit.
        if thumb_bytes is not None and "mag" in panel_files:
            site.thumb_store[panel_files["mag"]] = thumb_bytes
        cov_heatmap_name: Optional[str] = None
        if corr_bytes is not None:
            cov_heatmap_name = _panel_figure_name(stem, wid, "corr")
            site.figure_store[cov_heatmap_name] = corr_bytes
        panel_files_by_wid[wid] = panel_files
        mag_geom_by_wid[wid] = mag_geom
        cov_heatmap_by_wid[wid] = cov_heatmap_name

    # --- per-window pages -----------------------------------------------
    for idx, wid in enumerate(page_ids):
        wf = win_fits[wid]
        panel_files = panel_files_by_wid[wid]
        mag_geom = mag_geom_by_wid[wid]
        cov_heatmap_name = cov_heatmap_by_wid[wid]
        ledger = get_candidate_ledger_impl(
            path, wid, spectrum_fit=bundle.fit, sideband=bundle.sideband
        )
        prev_id = page_ids[idx - 1] if idx > 0 else None
        next_id = page_ids[idx + 1] if idx + 1 < len(page_ids) else None
        win_peaks = peaks_by_window.get(wid, [])
        win_matches = (
            [match_by_peak.get(id(p)) for p in win_peaks]
            if cross_ref is not None
            else None
        )
        page_html = _window_page(
            stem=stem,
            window_id=wid,
            wf=wf,
            peaks=win_peaks,
            uname=uname,
            uval=uval,
            status=review.window_statuses.get(wid),
            ledger=ledger,
            decisions=decisions_by_window.get(wid, []),
            panel_files=panel_files,
            cov_heatmap_name=cov_heatmap_name,
            sideband=bundle.sideband,
            prev_id=prev_id,
            next_id=next_id,
            catalog_matches=win_matches,
            merges=merges_by_window.get(wid),
            nav_rows=index_rows,
            band=nav_band,
            overview_name=overview_name,
            mag_geom=mag_geom,
        )
        site.page_store[f"windows/{_window_page_name(wid)}"] = page_html

    # --- methods + results page (Level-2 content, HTML-ified) ---------------
    # Each distribution figure is injected next to the percentile table it
    # summarizes; any that cannot be placed fall back to a trailing section.
    methods_md = _render_markdown(model, path, cross_ref=cross_ref)
    methods_html = _md_to_html(methods_md)
    leftover: List[str] = []
    for anchor, slug, specs in _summary_distribution_groups(model, cross_ref):
        fig = plot_summary_histograms(specs, ncols=min(3, len(specs)))
        if fig is None:
            continue
        fname = f"{stem}_hist_{slug}.png"
        _save_figure_png(fig, out_root / "figures" / fname, dpi=dpi)
        plt.close(fig)
        snippet = (
            f'<div class="hist"><img src="figures/{fname}" '
            f'alt="{_esc(anchor)} distribution"></div>'
        )
        methods_html, injected = _inject_after_table(methods_html, anchor, snippet)
        if not injected:
            leftover.append(snippet)
    # Magnitude-distribution figures: the full-band panel lands at the Stage 2
    # results; the per-band panels after the per-band noise table. Either falls
    # back to a trailing Distributions block if its anchor is absent.
    full_band_html, per_band_html = _magnitude_histogram_figures(
        bundle, out_root, stem, dpi, plot_magnitude_histogram
    )
    if full_band_html:
        methods_html, ok = _inject_after(
            methods_html, "median σ_x", full_band_html, closing="</p>"
        )
        if not ok:
            leftover.append(full_band_html)
    if per_band_html:
        methods_html, ok = _inject_after_table(
            methods_html, "Per-band noise", per_band_html
        )
        if not ok:
            leftover.append(per_band_html)
    # Per-stage diagnostic figures, each dropped at the end of its section (before
    # the next stage's heading); a stage that was not run is skipped.
    for anchor, fig_html in _methods_stage_figures(
        path, out_root, stem, dpi, jobs=jobs
    ):
        methods_html, ok = _inject_before(methods_html, anchor, fig_html)
        if not ok:
            leftover.append(fig_html)
    methods_page = _summary_page(stem, methods_html, leftover)
    site.page_store["methods.html"] = methods_page

    # --- index ----------------------------------------------------------
    # Spectrum section: the interactive full-spectrum overview (the shared image
    # rendered above) with a clickable per-window overlay -- the quick-nav is the
    # image itself, so there is no separate strip.
    spectrum_section: List[str] = []
    if index_rows and nav_band is not None and overview_name is not None:
        spectrum_section.append("<h2>Spectrum</h2>")
        spectrum_section.append(
            _spectrum_nav(
                index_rows,
                nav_band,
                stem,
                link_prefix="windows/",
                thumb_prefix="figures/",
            )
        )
        spectrum_section.append(
            '<p class="winmap-hint">The shaded bands are fit windows (orange = '
            "flagged for attention); hover for details, click to open its page.</p>"
        )

    # Hover-preview attributes per window (magnitude thumbnail + identical info
    # text), shared by the window list and the final line list.
    preview_attrs: Dict[int, str] = {
        wid: _preview_row_attr(
            stem, wid, _window_preview_info(wid, lo, hi, k, chi2r), has_page=has_page
        )
        for wid, lo, hi, k, chi2r, has_page, _attn in index_rows
    }

    body: List[str] = [
        f"<h1>FTMW pipeline report &mdash; {_esc(stem)}</h1>",
        "<p>Generated by <code>ftmwpipeline</code>. This site renders the "
        "persisted analysis record; it does not recompute the fit.</p>",
        *_index_summary_block(model, products),
        '<p><a href="methods.html">Methods &amp; results &rarr;</a> '
        "&mdash; per-stage algorithm notes, parameters, statistics tables, and "
        "distribution histograms.</p>",
        *spectrum_section,
        '<h2 id="window-list">Windows</h2>',
        f"<p>{len(page_ids):,} of {len(all_ids):,} windows have a detail page "
        f"(<code>{_esc(key)}</code> filter); {len(attention_ids):,} flagged for "
        "attention.</p>",
        _index_window_table(index_rows, preview_attrs),
        '<h2 id="final-list">Final line list</h2>',
        f"<p>All {len(products.peaks):,} calibrated lines "
        f"(amplitude in {_esc(uname)})"
        + (
            f"; {cross_ref.n_matched:,} flagged against "
            f"<code>{_esc(Path(cross_ref.catalog_path).name)}</code> "
            f"within {cross_ref.n_sigma:g}&sigma; (proximity only)."
            if cross_ref is not None
            else "."
        )
        + "</p>",
        _index_final_table(
            products,
            cross_ref.matches if cross_ref is not None else None,
            preview_attrs=preview_attrs,
        ),
    ]
    # The hover-preview popup drives both the window-map strip (when present) and
    # the index tables' rows, so it is always included on the index page.
    body.append(_WINMAP_JS)
    index_html = _page(f"{stem} report", body, css_href="assets/style.css")
    site.page_store["index.html"] = index_html

    return site


def report_full_impl(
    file_path: Union[Path, str],
    *,
    output_dir: Union[Path, str],
    windows: str = "all",
    dpi: int = 110,
    catalog: Optional[Union[Path, str]] = None,
    catalog_n_sigma: float = 3.0,
    scope: str = "full",
    jobs: Optional[int] = None,
) -> str:
    """Render the Level-3 self-contained HTML report; return its path.

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` pipeline file.
    output_dir :
        Directory to write the report file into (created if absent).
    windows :
        ``"all"`` (a detail section per fit window) or ``"attention"`` (sections
        only for windows the Stage 6 review flagged). The index always lists
        every window, linking the ones with a generated section.
    dpi :
        Resolution for the per-window matplotlib figures.
    catalog :
        Optional frequency-catalog path. When given, a proximity-match badge
        column is added to the index and per-window line tables, the methods
        section gains the catalog cross-reference + pull-calibration content, and
        a σ_f pull histogram is added to the distributions (label echo only,
        never an assignment).
    catalog_n_sigma :
        Catalog match tolerance in combined sigmas (default ``3``).
    scope :
        Report content scope: ``"full"`` (default) folds in a detail section per
        window; ``"summary"`` keeps the index + methods only. Either way the
        output is a single self-contained HTML file.

    Returns
    -------
    str
        Path to the generated self-contained report file (``<stem>_report.html``,
        or ``<stem>_report_summary.html`` for ``scope="summary"``).

    Raises
    ------
    ValueError
        If *windows* or *scope* is unknown, no final-products table is present
        (the Stage 6 ``review run`` consolidation has not been run), or *catalog*
        is given but unreadable / empty.
    """
    import shutil
    import tempfile

    if scope not in REPORT_SCOPES:
        raise ValueError(
            f"unknown report scope {scope!r}; choose one of {REPORT_SCOPES}"
        )

    final_dir = Path(output_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    # Assemble straight to an in-memory model and collapse it into one
    # self-contained file (figures embedded, CSS inlined, cross-page links
    # rewritten to in-document anchors). The O(N_windows) per-window figures +
    # pages never touch disk -- only the O(1) methods-page figures land in the
    # scratch dir, which is discarded.
    out_root = Path(tempfile.mkdtemp(prefix="ftmw_report_"))
    try:
        site = _assemble_report_site(
            file_path,
            out_root=out_root,
            windows=windows,
            dpi=dpi,
            catalog=catalog,
            catalog_n_sigma=catalog_n_sigma,
            jobs=jobs,
        )
        suffix = "_summary" if scope == "summary" else ""
        single_path = final_dir / f"{site.stem}_report{suffix}.html"
        single_path.write_text(
            _collapse_site_to_single_file(
                out_root,
                mode=scope,
                stem=site.stem,
                figure_store=site.figure_store,
                thumb_store=site.thumb_store,
                page_store=site.page_store,
                css=site.css,
            )
        )
    finally:
        shutil.rmtree(out_root, ignore_errors=True)
    return str(single_path)


def report_run_impl(
    file_path: Union[Path, str],
    *,
    output_dir: Optional[Union[Path, str]] = None,
    windows: str = "all",
    emit_table: bool = True,
    emit_html: bool = True,
    table_format: str = "csv",
    scope: str = "full",
    catalog: Optional[Union[Path, str]] = None,
    catalog_n_sigma: float = 3.0,
    jobs: Optional[int] = None,
) -> Dict[str, Optional[str]]:
    """The default Stage 6 report run: the L1 table plus the L3 HTML report.

    Writes both deliverables into *output_dir* in one call, defaulting to the
    current working directory when *output_dir* is omitted. By default the
    artifacts are the Level-1 final-products table (``<stem>_lines.csv``) and the
    self-contained Level-3 report with every window folded in
    (``<stem>_report.html``).
    Either artifact can be suppressed (``emit_table`` / ``emit_html``); the HTML
    content follows *scope* (``"full"`` folds in every window, ``"summary"`` keeps
    the index + methods only). Renders the persisted record; never recomputes.

    Returns ``{"table": <path|None>, "html": <path|None>}`` -- the path of each
    artifact written, ``None`` when that artifact was suppressed.

    Raises
    ------
    ValueError
        If both artifacts are disabled, or the underlying table / HTML render
        raises (no final-products table, unknown format / windows / scope,
        unreadable catalog).
    """
    if not emit_table and not emit_html:
        raise ValueError(
            "nothing to do: both the table and the HTML report are disabled"
        )

    stem = Path(str(file_path)).stem
    out_dir = Path.cwd() if output_dir is None else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Optional[str]] = {"table": None, "html": None}

    if emit_table:
        ext = _TABLE_EXT.get(str(table_format).lower(), "txt")
        table_path = out_dir / f"{stem}_lines.{ext}"
        report_table_impl(
            file_path,
            fmt=table_format,
            output=str(table_path),
            catalog=catalog,
            catalog_n_sigma=catalog_n_sigma,
        )
        results["table"] = str(table_path)

    if emit_html:
        results["html"] = report_full_impl(
            file_path,
            output_dir=str(out_dir),
            windows=windows,
            catalog=catalog,
            catalog_n_sigma=catalog_n_sigma,
            scope=scope,
            jobs=jobs,
        )

    return results


def _window_range(wf: Any) -> Tuple[float, float]:
    lo, hi = wf.window.freq_range
    return float(lo), float(hi)


def _opt_chi2(wf: Any) -> Optional[float]:
    c = wf.reduced_chi2
    try:
        cf = float(c)
    except (TypeError, ValueError):
        return None
    return cf if np.isfinite(cf) else None
