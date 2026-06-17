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
from ..io.stage6_review_serialization import load_stage6_review_from_file
from .report_impl import (
    _CAL_STATE_PHRASE,
    _amplitude_unit,
    _freq,
    _g,
    _md_num,
    _scaled,
    assemble_summary_model,
)

VALID_WINDOW_FILTERS = ("all", "attention")

# Above this many covariance parameters the matrix is summarized rather than
# rendered inline (a wide window's peak-major matrix is unreadable as a table;
# the full matrix stays available via the API / Stage 5 record).
_COVARIANCE_RENDER_CAP = 24


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
main.report { max-width: 1100px; margin: 0 auto; padding: 1.5rem 2rem 4rem; }
h1, h2, h3 { line-height: 1.2; }
a { color: #1559b3; }
table { border-collapse: collapse; margin: 0.5rem 0 1.25rem; font-size: 0.9rem; }
th, td { border: 1px solid #d0d4d9; padding: 0.25rem 0.6rem; text-align: right; }
th { background: #eceff2; }
td:first-child, th:first-child { text-align: left; }
img.fit-figure { max-width: 100%; height: auto; border: 1px solid #d0d4d9; }
pre { background: #11151a; color: #e6e6e6; padding: 0.75rem 1rem;
      overflow-x: auto; border-radius: 4px; font-size: 0.82rem; }
.badge { display: inline-block; padding: 0.05rem 0.45rem; border-radius: 3px;
         font-size: 0.78rem; background: #f0d9a8; color: #5a4300;
         margin-left: 0.35rem; }
.nav { margin: 1rem 0; }
.summary li { margin: 0.15rem 0; }
"""


# ---------------------------------------------------------------------------
# Window-name helpers
# ---------------------------------------------------------------------------


def _window_page_name(window_id: int) -> str:
    return f"window_{window_id:03d}.html"


def _figure_name(stem: str, window_id: int) -> str:
    return f"{stem}_window_{window_id:03d}.png"


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
    rows: List[List[str]] = []
    for p in peaks:
        rows.append(
            [
                _esc(_freq(p.frequency_mhz)),
                _esc(_freq(p.frequency_raw_mhz)),
                _esc(_g(p.sigma_f_khz, 3)),
                _esc(_g(p.sigma_stat_khz, 3)),
                _esc(_g(p.sigma_eps_khz, 3)),
                _esc(_scaled(p.amplitude, uval)),
                _esc(_g(p.phase, 3)),
                _esc(_md_num(p.snr, 3)),
                _esc(p.origin),
            ]
        )
    head = [
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


def _covariance_block(
    cov: Optional[np.ndarray], labels: Optional[List[str]]
) -> List[str]:
    if cov is None or labels is None or len(labels) == 0:
        return [
            "<p><em>No parameter covariance was persisted for this window.</em></p>"
        ]
    n = len(labels)
    if n > _COVARIANCE_RENDER_CAP:
        return [
            f"<p><em>Parameter covariance is a {n}&times;{n} matrix &mdash; too "
            "wide to render inline. The full matrix is available via the API "
            "(<code>FittingResult.covariance</code>).</em></p>"
        ]
    arr = np.asarray(cov, dtype=float)
    rows: List[List[str]] = []
    for i, lbl in enumerate(labels):
        rows.append([_esc(lbl)] + [_esc(_g(float(arr[i, j]), 3)) for j in range(n)])
    return [_table([""] + [_esc(x) for x in labels], rows, cls="covariance")]


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
    figure_name: str,
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
        f'<img class="fit-figure" src="../figures/{figure_name}" '
        f'alt="window {window_id} fit detail">',
        "<h2>Fitted lines</h2>",
        _window_peak_table(peaks, uname, uval),
        "<h2>Parameter covariance</h2>",
        *_covariance_block(
            getattr(wf, "covariance", None),
            getattr(wf, "covariance_param_labels", None),
        ),
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

    from .stage5_impl import (
        _resolve_detail_bundle,
        fit_window_report_text,
        render_fit_detail_impl,
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
        fig_name = _figure_name(stem, wid)
        fig = render_fit_detail_impl(path, wid, bundle=bundle)
        fig.savefig(str(out_root / "figures" / fig_name), dpi=dpi)
        plt.close(fig)
        report_text = fit_window_report_text(path, wid, bundle=bundle, show_audit=True)
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
            figure_name=fig_name,
            prev_id=prev_id,
            next_id=next_id,
        )
        (out_root / "windows" / _window_page_name(wid)).write_text(page_html)

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
