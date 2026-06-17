"""Level 3 (``report full``): a local, linked HTML site over the persisted record.

Reports render the persisted Stage 6 record; they never recompute the fit. L3 is
an **assembler** over existing artifacts and renderers, not new analysis: it
reuses the L2 summary model for the index, the ``fit show`` figure renderer for
the per-window plots, the candidate ledger, and the persisted Stage 6 review
status / decision log. The output is a self-contained directory:

    <output_dir>/
        index.html                  -- summary + final table + window links
        assets/style.css            -- styling (hand-tuned later)
        figures/<stem>_window_NNN.png
        windows/window_NNN.html     -- one page per window

The HTML is dependency-light (hand-rolled markup + matplotlib PNGs); a separate
stylesheet (``assets/style.css``) carries the presentation so the structure and
the styling evolve independently. See ``dev-docs/planning/stage6-reports.md`` §C.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
)

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


def _table(headers: List[str], rows: List[List[str]], *, cls: str = "") -> str:
    """A simple HTML table; cell contents are emitted verbatim (pre-escaped)."""
    cls_attr = f' class="{cls}"' if cls else ""
    out = [f"<table{cls_attr}>", "  <thead>", "    <tr>"]
    out += [f"      <th>{h}</th>" for h in headers]
    out += ["    </tr>", "  </thead>", "  <tbody>"]
    for r in rows:
        out.append("    <tr>")
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


# A deliberately minimal stylesheet -- enough that the structure is legible;
# the visual polish is a follow-on pass that only edits this file.
_STYLESHEET = """/* ftmwpipeline report -- Level 3 (report full).
   Minimal baseline styling; refine here without touching the HTML. */
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
thead th { position: sticky; top: 0; z-index: 1; }
pre { background: #11151a; color: #e6e6e6; padding: 0.75rem 1rem;
      overflow-x: auto; border-radius: 4px; font-size: 0.82rem; }
.badge { display: inline-block; padding: 0.05rem 0.45rem; border-radius: 3px;
         font-size: 0.78rem; background: #f0d9a8; color: #5a4300;
         margin-left: 0.35rem; }
/* Catalog proximity-match badge (a cross-check echo, not an assignment). */
.badge.cat { background: #cfe8d2; color: #1d5026; margin-left: 0; cursor: help; }
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
/* The interleaved distribution figures cap their natural width and centre,
   rather than stretching a 1-2 panel group across the full column. */
.hist { margin: 0.5rem 0 1.25rem; }
.hist img { max-width: 100%; width: auto; height: auto; border: 1px solid #d0d4d9;
            background: #fff; }
/* The index full-spectrum overview spans the content width. */
.spectrum-overview { margin: 0.5rem 0 0.25rem; }
.spectrum-overview img { width: 100%; height: auto; border: 1px solid #d0d4d9;
                         background: #fff; }
/* Window-map navigation strip: one clickable bar per window under the overview,
   coloured by attention; hover highlights, the SVG <a> navigates, and the
   <title> gives a native tooltip. The optional script adds a zoom thumbnail. */
.winmap { margin: 0 0 0.4rem; }
.winmap-svg { width: 100%; height: auto; display: block; }
.winmap-rect { fill: #9bb0c4; stroke: #6b8298; stroke-width: 0.5; }
.winmap-rect.attn { fill: #f4a23b; stroke: #b9741a; }
/* "You are here" on a window page: a bold dark outline over the fill. */
.winmap-rect.current { stroke: #11151a; stroke-width: 2; }
.winmap a:hover .winmap-rect, .winmap-rect:hover { fill: #1559b3; stroke: #0d3f86;
                                                   cursor: pointer; }
.winmap-axis { stroke: #c0c7cf; stroke-width: 1; }
.winmap-tick { font-size: 11px; fill: #555; }
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


# ---------------------------------------------------------------------------
# Level-2 summary -> HTML (small, targeted Markdown converter)
# ---------------------------------------------------------------------------

_MD_BOLD = re.compile(r"\*\*(.+?)\*\*")
_MD_CODE = re.compile(r"`([^`]+?)`")


def _md_inline(text: str) -> str:
    """Escape one line of Markdown text, then apply bold / inline-code spans."""
    out = _esc(text)
    out = _MD_CODE.sub(lambda m: f"<code>{m.group(1)}</code>", out)
    out = _MD_BOLD.sub(lambda m: f"<strong>{m.group(1)}</strong>", out)
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


def _summary_distribution_specs(
    model: Any, xref: Optional[CatalogCrossRef] = None
) -> List[Tuple[str, str, List[float]]]:
    """The (title, xlabel, values) histogram specs drawn from the L2 model.

    When a catalog cross-reference is supplied, the σ_f pull distribution is
    appended (the calibration surface: ~unit-normal when σ_f is honest).
    """
    specs = [
        ("Reduced χ² per window", "χ²_r", model.chi2r_values),
        ("Shape error ε per window", "ε (% / bin)", model.eps_values),
        ("Precision σ_stat", "σ_stat (kHz)", model.sigma_stat_values),
        ("Timebase σ_ε", "σ_ε (kHz)", model.sigma_eps_values),
        ("Budget σ_f", "σ_f (kHz)", model.sigma_f_values),
        ("Promoted-peak SNR", "SNR", model.snr_values_promoted),
    ]
    if xref is not None and xref.pull_values:
        specs.append(("Catalog pull (f_fit−f_cat)/σ_f", "pull", xref.pull_values))
    return specs


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


def _inject_after_table(html: str, anchor: str, snippet: str) -> Tuple[str, bool]:
    """Insert *snippet* right after the first ``</table>`` that follows *anchor*.

    *anchor* is a caption substring; returns ``(html, injected)`` -- ``injected``
    is False when the anchor or a following table is absent (the caller then
    falls the figure back to a trailing section).
    """
    pos = html.find(anchor)
    if pos < 0:
        return html, False
    end = html.find("</table>", pos)
    if end < 0:
        return html, False
    cut = end + len("</table>")
    return html[:cut] + "\n" + snippet + html[cut:], True


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
    return fig


# Window-map SVG geometry (a fixed viewBox the browser scales to the column
# width; uniform scaling keeps the tick labels undistorted).
_WINMAP_W = 1400.0
_WINMAP_H = 64.0


def _window_map_svg(
    rows: List[Tuple[int, float, float, int, Optional[float], bool, bool]],
    band: Tuple[float, float],
    stem: str,
    *,
    link_prefix: str,
    thumb_prefix: str,
    current_id: Optional[int] = None,
) -> str:
    """A clickable SVG window-map strip spanning the band.

    One ``<rect>`` per window, positioned by frequency and coloured by attention
    (the *current_id* window, when on a window page, gets a "you are here"
    outline); a window with a detail page is wrapped in an SVG ``<a>`` (click
    navigates) and carries ``data-thumb`` (its magnitude panel) for the optional
    hover-zoom script. Each rect's ``<title>`` is the native tooltip (and the
    text the popup reuses). ``rows`` is the index table's
    ``(wid, lo, hi, k, chi2r, has_page, attn)`` tuples; *link_prefix* /
    *thumb_prefix* make the page/figure paths resolve from either the index
    (``"windows/"`` / ``"figures/"``) or a window page (``""`` / ``"../figures/"``).
    The horizontal margin matches the overview figure (:data:`_OVERVIEW_MARGIN_FRAC`)
    so the strip lines up with it and the edge frequency labels are not clipped.
    """
    lo, hi = float(min(band)), float(max(band))
    span = hi - lo if hi > lo else 1.0
    pad = _OVERVIEW_MARGIN_FRAC * _WINMAP_W
    top = 6.0
    strip_h = 40.0
    usable = _WINMAP_W - 2.0 * pad

    def x_of(f: float) -> float:
        return pad + (float(f) - lo) / span * usable

    parts: List[str] = [
        f'<svg class="winmap-svg" viewBox="0 0 {_WINMAP_W:.0f} {_WINMAP_H:.0f}" '
        'preserveAspectRatio="xMidYMid meet" role="img" '
        'aria-label="window navigation map">',
        f'<line class="winmap-axis" x1="{pad:.1f}" y1="{top + strip_h:.1f}" '
        f'x2="{_WINMAP_W - pad:.1f}" y2="{top + strip_h:.1f}"/>',
    ]
    for wid, wlo, whi, k, chi2r, has_page, attn in rows:
        x0, x1 = x_of(min(wlo, whi)), x_of(max(wlo, whi))
        w = max(x1 - x0, 2.5)
        cls = "winmap-rect"
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
            f'<rect class="{cls}" x="{x0:.1f}" y="{top:.1f}" width="{w:.1f}" '
            f'height="{strip_h:.1f}" data-window="{wid}"{thumb}>'
            f"<title>{_esc(info)}</title></rect>"
        )
        if has_page:
            parts.append(f'<a href="{link_prefix}{_window_page_name(wid)}">{rect}</a>')
        else:
            parts.append(rect)
    n_ticks = 7
    for i in range(n_ticks):
        f = lo + span * i / (n_ticks - 1)
        parts.append(
            f'<text class="winmap-tick" x="{x_of(f):.1f}" '
            f'y="{_WINMAP_H - 3:.1f}" text-anchor="middle">{f:.0f}</text>'
        )
    parts.append("</svg>")
    return '<div class="winmap">' + "".join(parts) + "</div>"


# Optional hover-zoom popup for the window map: a tiny, dependency-free script
# that shows a window's magnitude thumbnail at the cursor on hover. The map is
# fully usable without it (rects link + carry native <title> tooltips), so this
# degrades gracefully when scripting is off.
_WINMAP_JS = """<script>
(function () {
  var rects = document.querySelectorAll('.winmap [data-window]');
  if (!rects.length) return;
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
    var info = t.getAttribute('data-info') || '';
    pop.innerHTML = (thumb ? '<img src="' + thumb + '" alt="">' : '') +
      '<div class="winmap-pop-info">' + info + '</div>';
    pop.style.display = 'block';
    move(e);
  }
  function hide() { pop.style.display = 'none'; }
  rects.forEach(function (el) {
    // Move the native <title> into a data attribute so the popup (below) is the
    // only tooltip when scripting is on; without this script the <title> stays
    // and the browser shows it natively.
    var titleEl = el.querySelector('title');
    if (titleEl) {
      el.setAttribute('data-info', titleEl.textContent);
      el.removeChild(titleEl);
    }
    el.addEventListener('mouseenter', show);
    el.addEventListener('mousemove', move);
    el.addEventListener('mouseleave', hide);
  });
})();
</script>"""


def _index_window_table(
    rows: List[Tuple[int, float, float, int, Optional[float], bool, bool]],
) -> str:
    """Build the index window list. Each row:
    ``(wid, lo, hi, k, chi2r, has_page, needs_attention)``."""
    out_rows: List[List[str]] = []
    for wid, lo, hi, k, chi2r, has_page, attn in rows:
        label = f"window {wid}"
        if has_page:
            link = f'<a href="windows/{_window_page_name(wid)}">{label}</a>'
        else:
            link = label
        if attn:
            link += '<span class="badge">attention</span>'
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


def _index_final_table(
    products: FinalProducts, matches: Optional[List[Optional[CatalogMatch]]] = None
) -> str:
    uname, uval = _amplitude_unit(products)
    with_cat = matches is not None
    rows: List[List[str]] = []
    for i, p in enumerate(products.peaks):
        wid = p.window_id
        win_cell = (
            f'<a href="windows/{_window_page_name(wid)}">{wid}</a>'
            if wid is not None
            else ""
        )
        row = [
            _esc(_freq(p.frequency_mhz)),
            _esc(_g(p.sigma_f_khz, 3)),
            _esc(_scaled(p.amplitude, uval)),
            _esc(_md_num(p.snr, 3)),
            _esc(p.origin),
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
        "Window",
    ]
    if with_cat:
        head.append("Catalog")
    return _table(head, rows, cls="final-list")


# ---------------------------------------------------------------------------
# Per-window page
# ---------------------------------------------------------------------------


def _window_peak_table(
    peaks: List[FinalPeak],
    uname: str,
    uval: float,
    matches: Optional[List[Optional[CatalogMatch]]] = None,
) -> str:
    """The per-window fitted-lines table.

    Each peak carries a display letter (assigned low-to-high frequency, shared
    with the figure annotations and the fit-log table). The calibrated
    frequency, amplitude, phase, and SNR are shown in concise ``value(unc)``
    notation (uncertainty in last-digit units) -- the raw frequency and the
    broken-out &sigma; columns stay in plain fixed/scientific form. When a
    catalog cross-reference is supplied, a proximity-match badge column is added.
    """
    from ..visualization.fit_detail import frequency_sorted_labels

    with_cat = matches is not None
    letters = frequency_sorted_labels([float(p.frequency_mhz) for p in peaks])
    rows: List[List[str]] = []
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
        ]
        if with_cat:
            row.append(_catalog_cell(matches[i]))  # type: ignore[index]
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
    ]
    if with_cat:
        head.append("Catalog")
    return _table(head, rows, cls="peak-list")


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


def _ledger_block(candidates: List[LedgerCandidate]) -> List[str]:
    if not candidates:
        return ["<p><em>No revivable ledger candidates for this window.</em></p>"]
    rows: List[List[str]] = []
    for c in candidates:
        rows.append(
            [
                _esc(_freq(c.frequency_mhz)),
                _esc(f"{c.seed_offset_mhz:+.4f}"),
                _esc(_g(c.best_evidence, 3)),
                _esc(c.evidence_kind),
                _esc(", ".join(c.decision_sites)),
                _esc("; ".join(c.reasons)),
            ]
        )
    head = [
        "Frequency (MHz)",
        "Seed offset (MHz)",
        "Evidence",
        "Kind",
        "Decision sites",
        "Reasons",
    ]
    return [_table(head, rows, cls="ledger")]


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


def _fit_panels_block(panel_files: Dict[str, str]) -> List[str]:
    """Lay the per-window panel PNGs out for the 1:2:2 responsive grid.

    The overview spans full width on top; the Re / Im model+residual panels
    share a two-column grid row and the |X| + residual histogram share the row
    below. The grid collapses to a single column on a narrow viewport (CSS), so
    each panel takes the full width in turn. Each entry is rendered only when its
    PNG was produced, so a degenerate window with missing panels still yields
    valid markup.
    """

    def _fig(panel: str, alt: str, cls: str = "panel") -> List[str]:
        name = panel_files.get(panel)
        if name is None:
            return []
        return [
            f'  <figure class="{cls}">'
            f'<img src="../figures/{name}" alt="{_esc(alt)}"></figure>'
        ]

    out = ['<div class="fit-panels">']
    out += _fig("overview", "full-spectrum context", "panel panel-overview")
    grid = (
        _fig("re", "real part + residual")
        + _fig("im", "imaginary part + residual")
        + _fig("mag", "magnitude + residual")
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
    context_name: Optional[str] = None,
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

    # Spectrum-context section: the aligned full-spectrum figure (this window
    # highlighted) over the window-map quick-nav strip -- the same overview/strip
    # pairing the index uses, so they line up here too.
    context: List[str] = []
    if nav_rows and band is not None:
        context.append("<h2>Spectrum context</h2>")
        if context_name is not None:
            context.append(
                f'<div class="spectrum-overview"><img src="../figures/'
                f'{context_name}" alt="full-spectrum context"></div>'
            )
        context.append(
            _window_map_svg(
                nav_rows,
                band,
                stem,
                link_prefix="",
                thumb_prefix="../figures/",
                current_id=window_id,
            )
        )
        context.append(
            '<p class="winmap-hint">Quick-nav: each bar is a fit window (orange = '
            "attention, outlined = this one); hover for details, click to jump.</p>"
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
        *_fit_panels_block(panel_files),
        "<h2>Fitted lines</h2>",
        _window_peak_table(peaks, uname, uval, catalog_matches),
        "<h2>Parameter covariance</h2>",
        *_covariance_block(wf, cov_heatmap_name, sideband),
        "<h2>Fit history</h2>",
        *_audit_block(
            wf, 0.5 * (float(lo) + float(hi)), _sideband_sign(sideband), merges
        ),
        "<h2>Ledger candidates</h2>",
        *_ledger_block(ledger),
        *_decision_block(decisions),
    ]
    if context:
        body.append(_WINMAP_JS)  # hover-zoom popup; the map works without it
    return _page(f"{stem} window {window_id}", body, css_href="../assets/style.css")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def report_full_impl(
    file_path: Union[Path, str],
    *,
    output_dir: Union[Path, str],
    windows: str = "all",
    dpi: int = 110,
    catalog: Optional[Union[Path, str]] = None,
    catalog_n_sigma: float = 3.0,
) -> str:
    """Assemble the Level-3 linked-HTML report site and return the index path.

    Parameters
    ----------
    file_path :
        Path to the ``.ftmw`` pipeline file.
    output_dir :
        Directory to write the site into (created if absent).
    windows :
        ``"all"`` (a page per fit window) or ``"attention"`` (pages only for
        windows the Stage 6 review flagged for attention). The index always
        lists every window, hyperlinking the ones with a generated page.
    dpi :
        Resolution for the per-window matplotlib figures.
    catalog :
        Optional frequency-catalog path. When given, a proximity-match badge
        column is added to the index and per-window line tables, the methods
        page gains the catalog cross-reference + pull-calibration section, and a
        σ_f pull histogram is added to the distributions (label echo only, never
        an assignment).
    catalog_n_sigma :
        Catalog match tolerance in combined sigmas (default ``3``).

    Returns
    -------
    str
        Path to the generated ``index.html``.

    Raises
    ------
    ValueError
        If *windows* is unknown, no final-products table is present (the Stage 6
        ``review run`` consolidation has not been run), or *catalog* is given
        but unreadable / empty.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..visualization.fit_detail import (
        frequency_sorted_labels,
        plot_correlation_heatmap,
        plot_summary_histograms,
    )
    from .report_impl import _render_markdown
    from .stage5_impl import _resolve_detail_bundle, render_fit_panels_impl
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

    out_root = Path(output_dir)
    (out_root / "assets").mkdir(parents=True, exist_ok=True)
    (out_root / "figures").mkdir(parents=True, exist_ok=True)
    (out_root / "windows").mkdir(parents=True, exist_ok=True)
    (out_root / "assets" / "style.css").write_text(_STYLESHEET)

    # --- per-window pages + figures -------------------------------------
    for idx, wid in enumerate(page_ids):
        wf = win_fits[wid]
        # The full-spectrum context is rendered in the index-aligned style (this
        # window highlighted) so it lines up with the window-map strip beneath
        # it; the per-window panels supply the zoomed re / im / mag / hist.
        context_name: Optional[str] = None
        if nav_band is not None:
            try:
                ctx_fig = _plot_index_overview(
                    bundle,
                    [],
                    highlight_range=_window_range(wf),
                    title="full-spectrum context (this window highlighted)",
                )
                context_name = f"{stem}_window_{wid:03d}_ctx.png"
                ctx_fig.savefig(str(out_root / "figures" / context_name), dpi=dpi)
                plt.close(ctx_fig)
            except (ValueError, KeyError):
                context_name = None
        # The zoomed panels (the overview panel is superseded by the aligned
        # context figure above, so it is not saved).
        panels = render_fit_panels_impl(path, wid, bundle=bundle)
        panel_files: Dict[str, str] = {}
        for panel in _PANEL_ORDER:
            pfig = panels.get(panel)
            if pfig is None:
                continue
            if panel == "overview":
                plt.close(pfig)
                continue
            fname = _panel_figure_name(stem, wid, panel)
            pfig.savefig(str(out_root / "figures" / fname), dpi=dpi)
            plt.close(pfig)
            panel_files[panel] = fname
        # Correlation heatmap (when a covariance was persisted) -- a divergent
        # [-1, +1] image that stays readable where the numeric matrix does not.
        cov = getattr(wf, "covariance", None)
        cov_labels = getattr(wf, "covariance_param_labels", None)
        cov_heatmap_name: Optional[str] = None
        if cov is not None and cov_labels:
            peak_letters = frequency_sorted_labels(
                [float(p.frequency_mhz) for p in wf.fitted_peaks]
            )
            sym_labels = [
                _param_symbol_mathtext(lbl, peak_letters) for lbl in cov_labels
            ]
            hfig = plot_correlation_heatmap(cov, sym_labels)
            cov_heatmap_name = _panel_figure_name(stem, wid, "corr")
            # bbox_inches="tight" so the wide mathtext axis labels (the baseline
            # coefficient stacks) are never clipped at the figure edge.
            hfig.savefig(
                str(out_root / "figures" / cov_heatmap_name),
                dpi=dpi,
                bbox_inches="tight",
            )
            plt.close(hfig)
        ledger = get_candidate_ledger_impl(path, wid)
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
            context_name=context_name,
        )
        (out_root / "windows" / _window_page_name(wid)).write_text(page_html)

    # --- methods + results page (Level-2 content, HTML-ified) ---------------
    # Each distribution figure is injected next to the percentile table it
    # summarizes; any that cannot be placed fall back to a trailing section.
    methods_md = _render_markdown(model, path, include_table=False, cross_ref=cross_ref)
    methods_html = _md_to_html(methods_md)
    leftover: List[str] = []
    for anchor, slug, specs in _summary_distribution_groups(model, cross_ref):
        fig = plot_summary_histograms(specs, ncols=min(3, len(specs)))
        if fig is None:
            continue
        fname = f"{stem}_hist_{slug}.png"
        fig.savefig(str(out_root / "figures" / fname), dpi=dpi)
        plt.close(fig)
        snippet = (
            f'<div class="hist"><img src="figures/{fname}" '
            f'alt="{_esc(anchor)} distribution"></div>'
        )
        methods_html, injected = _inject_after_table(methods_html, anchor, snippet)
        if not injected:
            leftover.append(snippet)
    (out_root / "methods.html").write_text(_summary_page(stem, methods_html, leftover))

    # --- index ----------------------------------------------------------
    # Full-spectrum overview figure (attention windows shaded).
    attention_ranges = [_window_range(win_fits[w]) for w in sorted(attention_ids)]
    overview_name: Optional[str] = None
    try:
        ov_fig = _plot_index_overview(bundle, attention_ranges)
        overview_name = f"{stem}_overview.png"
        ov_fig.savefig(str(out_root / "figures" / overview_name), dpi=dpi)
        plt.close(ov_fig)
    except (ValueError, KeyError):
        overview_name = None

    # Spectrum section: the overview image (when rendered) over an interactive
    # window-map strip (one clickable bar per window).
    spectrum_section: List[str] = []
    if index_rows and nav_band is not None:
        spectrum_section.append("<h2>Spectrum</h2>")
        if overview_name is not None:
            spectrum_section.append(
                f'<div class="spectrum-overview"><img src="figures/'
                f'{overview_name}" alt="full-spectrum overview"></div>'
            )
        spectrum_section.append(
            _window_map_svg(
                index_rows,
                nav_band,
                stem,
                link_prefix="windows/",
                thumb_prefix="figures/",
            )
        )
        spectrum_section.append(
            '<p class="winmap-hint">Each bar is a fit window (orange = flagged for '
            "attention); hover for details, click to open its page.</p>"
        )

    body: List[str] = [
        f"<h1>FTMW pipeline report &mdash; {_esc(stem)}</h1>",
        "<p>Generated by <code>ftmwpipeline</code>. This site renders the "
        "persisted analysis record; it does not recompute the fit.</p>",
        *_index_summary_block(model, products),
        '<p><a href="methods.html">Methods &amp; results &rarr;</a> '
        "&mdash; per-stage algorithm notes, parameters, statistics tables, and "
        "distribution histograms.</p>",
        *spectrum_section,
        "<h2>Windows</h2>",
        f"<p>{len(page_ids):,} of {len(all_ids):,} windows have a detail page "
        f"(<code>{_esc(key)}</code> filter); {len(attention_ids):,} flagged for "
        "attention.</p>",
        _index_window_table(index_rows),
        "<h2>Final line list</h2>",
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
            products, cross_ref.matches if cross_ref is not None else None
        ),
    ]
    if spectrum_section:
        body.append(_WINMAP_JS)  # hover-zoom popup; the map works without it
    index_html = _page(f"{stem} report", body, css_href="assets/style.css")
    index_path = out_root / "index.html"
    index_path.write_text(index_html)
    return str(index_path)


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
