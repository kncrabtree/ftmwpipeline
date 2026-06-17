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


def _page(title: str, body: List[str], *, css_href: str) -> str:
    """Wrap a body fragment list in a minimal HTML document."""
    return "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="utf-8">',
            '  <meta name="viewport" content="width=device-width, ' 'initial-scale=1">',
            f"  <title>{_esc(title)}</title>",
            f'  <link rel="stylesheet" href="{css_href}">',
            "</head>",
            "<body>",
            '<main class="report">',
            *body,
            "</main>",
            "</body>",
            "</html>",
            "",
        ]
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
pre { background: #11151a; color: #e6e6e6; padding: 0.75rem 1rem;
      overflow-x: auto; border-radius: 4px; font-size: 0.82rem; }
.badge { display: inline-block; padding: 0.05rem 0.45rem; border-radius: 3px;
         font-size: 0.78rem; background: #f0d9a8; color: #5a4300;
         margin-left: 0.35rem; }
.nav { margin: 1rem 0; }
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
/* Level-2 methods page: equation source blocks and the distributions figure. */
pre.equation { background: #f0f2f5; color: #1a1a1a; border: 1px solid #d0d4d9;
               font-size: 0.85rem; }
.hist img { max-width: 100%; width: 100%; height: auto; border: 1px solid #d0d4d9;
            background: #fff; }
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
            out.append(f'<pre class="equation">{_esc(body)}</pre>')
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


def _summary_distribution_specs(model: Any) -> List[Tuple[str, str, List[float]]]:
    """The (title, xlabel, values) histogram specs drawn from the L2 model."""
    return [
        ("Reduced χ² per window", "χ²_r", model.chi2r_values),
        ("Shape error ε per window", "ε (% / bin)", model.eps_values),
        ("Precision σ_stat", "σ_stat (kHz)", model.sigma_stat_values),
        ("Timebase σ_ε", "σ_ε (kHz)", model.sigma_eps_values),
        ("Budget σ_f", "σ_f (kHz)", model.sigma_f_values),
        ("Promoted-peak SNR", "SNR", model.snr_values_promoted),
    ]


def _summary_page(stem: str, md_html: str, hist_name: Optional[str]) -> str:
    """The HTML methods + results page (Level-2 content plus histograms)."""
    body: List[str] = [
        '<div class="nav"><a href="index.html">&larr; index</a></div>',
        md_html,
    ]
    if hist_name is not None:
        body += [
            "<h2>Distributions</h2>",
            "<p>Histograms of the per-window / per-line statistics tabulated "
            "above (red dashed line = median).</p>",
            f'<div class="hist"><img src="figures/{hist_name}" '
            'alt="summary distributions"></div>',
        ]
    return _page(f"{stem} methods", body, css_href="assets/style.css")


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


def _index_final_table(products: FinalProducts) -> str:
    uname, uval = _amplitude_unit(products)
    rows: List[List[str]] = []
    for p in products.peaks:
        wid = p.window_id
        win_cell = (
            f'<a href="windows/{_window_page_name(wid)}">{wid}</a>'
            if wid is not None
            else ""
        )
        rows.append(
            [
                _esc(_freq(p.frequency_mhz)),
                _esc(_g(p.sigma_f_khz, 3)),
                _esc(_scaled(p.amplitude, uval)),
                _esc(_md_num(p.snr, 3)),
                _esc(p.origin),
                win_cell,
            ]
        )
    head = [
        "Frequency (MHz)",
        "&sigma;<sub>f</sub> (kHz)",
        f"Amplitude ({_esc(uname)})",
        "SNR",
        "Origin",
        "Window",
    ]
    return _table(head, rows, cls="final-list")


# ---------------------------------------------------------------------------
# Per-window page
# ---------------------------------------------------------------------------


def _window_peak_table(peaks: List[FinalPeak], uname: str, uval: float) -> str:
    """The per-window fitted-lines table.

    Each peak carries a display letter (assigned low-to-high frequency, shared
    with the figure annotations and the fit-log table). The calibrated
    frequency, amplitude, phase, and SNR are shown in concise ``value(unc)``
    notation (uncertainty in last-digit units) -- the raw frequency and the
    broken-out &sigma; columns stay in plain fixed/scientific form.
    """
    from ..visualization.fit_detail import frequency_sorted_labels

    letters = frequency_sorted_labels([float(p.frequency_mhz) for p in peaks])
    rows: List[List[str]] = []
    for lbl, p in zip(letters, peaks):
        sigma_f_mhz = None if p.sigma_f_khz is None else float(p.sigma_f_khz) / 1e3
        amp = None if p.amplitude is None else float(p.amplitude) / uval
        amp_err = None if p.amplitude_error is None else float(p.amplitude_error) / uval
        rows.append(
            [
                _esc(lbl),
                _esc(_concise(float(p.frequency_mhz), sigma_f_mhz)),
                _esc(_freq(p.frequency_raw_mhz)),
                _esc(_g(p.sigma_f_khz, 3)),
                _esc(_g(p.sigma_stat_khz, 3)),
                _esc(_g(p.sigma_eps_khz, 3)),
                _esc("" if amp is None else _concise(amp, amp_err)),
                _esc(
                    "" if p.phase is None else _concise(float(p.phase), p.phase_error)
                ),
                _esc("" if p.snr is None else _concise(float(p.snr), p.snr_error)),
                _esc(p.origin),
            ]
        )
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


def _audit_block(wf: Any, center_mhz: float, sideband_sign: float) -> List[str]:
    """Render the conservative add-one-peak history as a legible table.

    One row per :class:`AuditStep`: the decision, the candidate's molecular
    frequency (and baseband offset), the running model size *K* after the step,
    the &chi;&sup2; it moved from / to, the F-test p-value, and -- most
    importantly -- the plain-language ``reason`` (which the raw fit-log text
    drops). A legend below explains the decision keywords and points at the final
    model so a seed/seed-blend sequence's outcome is unambiguous.
    """
    audit = getattr(wf, "audit_trail", None) or []
    if not audit:
        return [
            "<p><em>No add-one-peak history was recorded for this window "
            "(e.g. an auto-merged or user-edited window rebuilt by a joint "
            "refit).</em></p>"
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
    report_text: str,
    panel_files: Dict[str, str],
    cov_heatmap_name: Optional[str],
    sideband: Any,
    prev_id: Optional[int],
    next_id: Optional[int],
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
        "<h2>Fit</h2>",
        *_fit_panels_block(panel_files),
        "<h2>Fitted lines</h2>",
        _window_peak_table(peaks, uname, uval),
        "<h2>Parameter covariance</h2>",
        *_covariance_block(wf, cov_heatmap_name, sideband),
        "<h2>Fit history (add-one-peak)</h2>",
        *_audit_block(wf, 0.5 * (float(lo) + float(hi)), _sideband_sign(sideband)),
        "<h2>Ledger candidates</h2>",
        *_ledger_block(ledger),
        *_decision_block(decisions),
        "<h2>Fit log</h2>",
        f"<pre>{_esc(report_text)}</pre>",
    ]
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

    Returns
    -------
    str
        Path to the generated ``index.html``.

    Raises
    ------
    ValueError
        If *windows* is unknown, or no final-products table is present (the
        Stage 6 ``review run`` consolidation has not been run).
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
    from .stage5_impl import (
        _resolve_detail_bundle,
        fit_window_report_text,
        render_fit_panels_impl,
    )
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

    # Group consolidated final peaks + user decisions by window.
    peaks_by_window: Dict[int, List[FinalPeak]] = {}
    for p in products.peaks:
        if p.window_id is not None:
            peaks_by_window.setdefault(int(p.window_id), []).append(p)
    decisions_by_window: Dict[int, List[DecisionLogEntry]] = {}
    for d in review.decision_log:
        decisions_by_window.setdefault(int(d.window_id), []).append(d)

    out_root = Path(output_dir)
    (out_root / "assets").mkdir(parents=True, exist_ok=True)
    (out_root / "figures").mkdir(parents=True, exist_ok=True)
    (out_root / "windows").mkdir(parents=True, exist_ok=True)
    (out_root / "assets" / "style.css").write_text(_STYLESHEET)

    # --- per-window pages + figures -------------------------------------
    for idx, wid in enumerate(page_ids):
        wf = win_fits[wid]
        # Each window's detail is a set of standalone panel figures (overview /
        # re / im / mag / hist) the page lays out in a flexbox.
        panels = render_fit_panels_impl(path, wid, bundle=bundle)
        panel_files: Dict[str, str] = {}
        for panel in _PANEL_ORDER:
            pfig = panels.get(panel)
            if pfig is None:
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
            hfig.savefig(str(out_root / "figures" / cov_heatmap_name), dpi=dpi)
            plt.close(hfig)
        # The add-one-peak audit is rendered as its own structured table on the
        # page; keep the <pre> fit log to the header + peak-parameter dump.
        report_text = fit_window_report_text(path, wid, bundle=bundle, show_audit=False)
        ledger = get_candidate_ledger_impl(path, wid)
        prev_id = page_ids[idx - 1] if idx > 0 else None
        next_id = page_ids[idx + 1] if idx + 1 < len(page_ids) else None
        page_html = _window_page(
            stem=stem,
            window_id=wid,
            wf=wf,
            peaks=peaks_by_window.get(wid, []),
            uname=uname,
            uval=uval,
            status=review.window_statuses.get(wid),
            ledger=ledger,
            decisions=decisions_by_window.get(wid, []),
            report_text=report_text,
            panel_files=panel_files,
            cov_heatmap_name=cov_heatmap_name,
            sideband=bundle.sideband,
            prev_id=prev_id,
            next_id=next_id,
        )
        (out_root / "windows" / _window_page_name(wid)).write_text(page_html)

    # --- methods + results page (Level-2 content, HTML-ified, + histograms) ---
    methods_md = _render_markdown(model, path, include_table=False)
    hist_fig = plot_summary_histograms(_summary_distribution_specs(model))
    hist_name: Optional[str] = None
    if hist_fig is not None:
        hist_name = f"{stem}_summary_histograms.png"
        hist_fig.savefig(str(out_root / "figures" / hist_name), dpi=dpi)
        plt.close(hist_fig)
    (out_root / "methods.html").write_text(
        _summary_page(stem, _md_to_html(methods_md), hist_name)
    )

    # --- index ----------------------------------------------------------
    page_set = set(page_ids)
    index_rows = [
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
    body: List[str] = [
        f"<h1>FTMW pipeline report &mdash; {_esc(stem)}</h1>",
        "<p>Generated by <code>ftmwpipeline</code>. This site renders the "
        "persisted analysis record; it does not recompute the fit.</p>",
        *_index_summary_block(model, products),
        '<p><a href="methods.html">Methods &amp; results &rarr;</a> '
        "&mdash; per-stage algorithm notes, parameters, statistics tables, and "
        "distribution histograms.</p>",
        "<h2>Windows</h2>",
        f"<p>{len(page_ids):,} of {len(all_ids):,} windows have a detail page "
        f"(<code>{_esc(key)}</code> filter); {len(attention_ids):,} flagged for "
        "attention.</p>",
        _index_window_table(index_rows),
        "<h2>Final line list</h2>",
        f"<p>All {len(products.peaks):,} calibrated lines "
        f"(amplitude in {_esc(uname)}).</p>",
        _index_final_table(products),
    ]
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
