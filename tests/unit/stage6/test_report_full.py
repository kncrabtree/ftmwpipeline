"""Tests for the Level-3 ``report full`` linked-HTML site.

Two layers: fast unit tests over the pure HTML helpers (no fixture fit needed),
and integration tests that run a real ``review run`` consolidation on the small
2638 fixture, assemble the site, and check the structure / well-formedness /
read-only / cross-interface (api == Pipeline == impl) guarantees.
"""

from __future__ import annotations

import hashlib
import html.parser
import shutil
from pathlib import Path
from typing import List

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.report_html_impl import (
    _PANEL_ORDER,
    _covariance_block,
    _esc,
    _assemble_report_site,
    _page,
    _panel_figure_name,
    _table,
    _window_page_name,
    _window_peak_table,
    report_full_impl,
)
from ftmwpipeline._internal.stage4_impl import load_windows_impl, save_window_plan_impl
from ftmwpipeline._internal.stage6_impl import review_run_impl
from ftmwpipeline.core.data_structures import FinalPeak
from ftmwpipeline.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Pure HTML-helper unit tests
# ---------------------------------------------------------------------------


def test_esc_escapes_markup():
    assert _esc("a<b>&'\"") == "a&lt;b&gt;&amp;&#x27;&quot;"
    assert _esc(None) == ""


def test_table_structure():
    out = _table(["A", "B"], [["1", "2"], ["3", "4"]], cls="x")
    assert '<table class="x">' in out
    assert "<th>A</th>" in out
    assert "<td>3</td>" in out
    assert out.count("<tr>") == 3  # header + 2 body rows


def test_page_is_wellformed_html():
    doc = _page("Title", ["<h1>hi</h1>"], css_href="assets/style.css")
    assert doc.startswith("<!DOCTYPE html>")
    assert '<link rel="stylesheet" href="assets/style.css">' in doc
    assert "<title>Title</title>" in doc
    html.parser.HTMLParser().feed(doc)  # no exception => parses


class _FakePeak:
    def __init__(self, frequency_mhz, amplitude=1.0, phase=0.5):
        self.frequency_mhz = frequency_mhz
        self.amplitude = amplitude
        self.phase = phase


class _FakeWindow:
    def __init__(self, freq_range):
        self.freq_range = freq_range


class _FakeWf:
    """Minimal stand-in for a per-window FittingResult covariance block."""

    def __init__(
        self, cov, labels, freqs, tau=5.0, freq_range=(10.0, 20.0), quality_metrics=None
    ):
        self.covariance = cov
        self.covariance_param_labels = labels
        self.fitted_peaks = [_FakePeak(f) for f in freqs]
        self.window = _FakeWindow(freq_range)
        self.shared_parameters = {"tau_us": {"value": tau}}
        self.quality_metrics = quality_metrics or {}


def test_covariance_block_renders_small_matrix():
    cov = np.array(
        [
            [1.0, 0.5, 0.0, 0.0],
            [0.5, 2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 0.5],
        ]
    )
    wf = _FakeWf(cov, ["amplitude_0", "offset_0", "phase_0", "tau"], [12.0])
    block = "\n".join(_covariance_block(wf, "corr.png", "lower"))
    # Symbolic parameter labels (not the raw words) + Greek tau.
    assert "amplitude_0" not in block
    assert "<sub>A</sub>" in block  # amplitude of peak A
    assert "&tau;" in block
    # Variances table (with values), the heatmap image, and the numeric matrix.
    assert "Variances" in block and "Value" in block
    assert "corr.png" in block
    assert 'class="covariance"' in block
    assert 'class="cov-legend"' in block


def test_covariance_block_summarizes_when_too_wide():
    n = 40
    cov = np.eye(n)
    labels = [f"amplitude_{i}" for i in range(n)]
    wf = _FakeWf(cov, labels, [10.0 + i for i in range(n)])
    block = "\n".join(_covariance_block(wf, "corr.png", "lower"))
    # The numeric matrix is suppressed, but the variances table and heatmap
    # (which stay readable) are still shown.
    assert "too" in block and "wide" in block
    assert 'class="covariance"' not in block
    assert 'class="variances"' in block
    assert "corr.png" in block


def test_covariance_block_handles_missing():
    wf = _FakeWf(None, None, [])
    block = "\n".join(_covariance_block(wf, None, "lower"))
    assert "No parameter covariance" in block


def test_param_symbol_and_value():
    from ftmwpipeline._internal.report_html_impl import (
        _param_symbol_html,
        _param_symbol_mathtext,
        _param_value,
    )

    letters = ["B", "A"]  # fit-peak 0 is the higher-frequency 'B'
    assert _param_symbol_html("amplitude_0", letters) == "A<sub>B</sub>"
    assert _param_symbol_html("offset_1", letters) == "f<sup>o</sup><sub>A</sub>"
    assert _param_symbol_html("phase_0", letters) == "&phi;<sub>B</sub>"
    assert _param_symbol_html("tau", letters) == "&tau;"
    assert _param_symbol_html("baseline_re_2", letters) == "c<sup>Re</sup><sub>2</sub>"
    assert _param_symbol_mathtext("phase_1", letters) == r"$\phi_{A}$"
    assert _param_symbol_mathtext("tau", letters) == r"$\tau$"

    wf = _FakeWf(
        np.eye(4),
        ["amplitude_0", "offset_0", "phase_0", "tau"],
        [12.0],
        quality_metrics={"baseline_coeff0_re": -8.4e-08, "baseline_coeff0_im": 1.2e-07},
    )
    wf.fitted_peaks[0].amplitude = 3.5
    # offset value = sign * (freq - center); center of (10, 20) = 15.
    assert _param_value("amplitude_0", wf, 15.0, 1.0) == pytest.approx(3.5)
    assert _param_value("offset_0", wf, 15.0, 1.0) == pytest.approx(12.0 - 15.0)
    assert _param_value("tau", wf, 15.0, 1.0) == pytest.approx(5.0)
    # Baseline coefficients come from quality_metrics, not a wf attribute.
    assert _param_value("baseline_re_0", wf, 15.0, 1.0) == pytest.approx(-8.4e-08)
    assert _param_value("baseline_im_0", wf, 15.0, 1.0) == pytest.approx(1.2e-07)
    assert _param_value("baseline_re_9", wf, 15.0, 1.0) is None


def test_md_to_html_subset():
    from ftmwpipeline._internal.report_html_impl import _md_to_html

    md = "\n".join(
        [
            "# Title",
            "",
            "Some **bold** and `code`.",
            "",
            "- one",
            "- two",
            "",
            "| A | B |",
            "| --- | --- |",
            "| 1 | 2 |",
            "",
            "$$",
            "x = y + z",
            "$$",
        ]
    )
    html_out = _md_to_html(md)
    assert "<h1>Title</h1>" in html_out
    assert "<strong>bold</strong>" in html_out
    assert "<code>code</code>" in html_out
    assert "<ul>" in html_out and "<li>one</li>" in html_out
    assert (
        "<table>" in html_out and "<th>A</th>" in html_out and "<td>1</td>" in html_out
    )
    assert 'class="equation"' in html_out and "x = y + z" in html_out


class _AuditStep:
    def __init__(self, decision, off, c0, c1, p, reason):
        self.decision = decision
        self.candidate_offset_mhz = off
        self.chi2_before = c0
        self.chi2_after = c1
        self.p_value = p
        self.reason = reason


def test_audit_block_frequency_and_k():
    from ftmwpipeline._internal.report_html_impl import _audit_block

    wf = _FakeWf(None, None, [])
    wf.audit_trail = [
        _AuditStep("seed", -0.5, 1e5, 5e4, 1e-9, "K=1 seed fit"),
        _AuditStep("seed-blend", -0.5, 5e4, 2e4, 1e-9, "K=2 re-seed"),
        _AuditStep("accept", +1.0, 2e4, 1e4, 1e-9, "+1 line(s)"),
        _AuditStep("reject", +2.0, 1e4, 1e4, 1.0, "collapsed"),
    ]
    # center 100, lower sideband sign -1: freq = center - offset.
    block = "\n".join(_audit_block(wf, 100.0, -1.0))
    assert "Frequency (MHz)" in block and "<th>K</th>" in block
    assert "audit-legend" in block
    # K column: seed->1, seed-blend->2, accept->3, reject stays 3.
    import re as _re

    krows = _re.findall(r"<tr>(.*?)</tr>", block, _re.S)[1:]  # skip header
    ks = [
        _re.findall(r"<td>(.*?)</td>", r, _re.S)[4].strip() for r in krows
    ]  # K is col index 4
    assert ks == ["1", "2", "3", "3"]
    # candidate frequency = 100 - (-0.5) = 100.5 for the seed.
    assert "100.500000" in block


def test_audit_block_falls_back_to_merge_history():
    from ftmwpipeline._internal.report_html_impl import _audit_block

    wf = _FakeWf(None, None, [])
    wf.audit_trail = []  # an auto-merged window carries no add-one history
    merges = [
        {
            "window_id": 222,
            "frequency_a_mhz": 37974.470,
            "frequency_b_mhz": 37974.511,
            "vif_a": 45.2,
            "vif_b": 42.7,
            "separation_res": 0.469,
            "merged_frequency_mhz": 37974.506,
        }
    ]
    block = "\n".join(_audit_block(wf, 100.0, -1.0, merges))
    assert "VIF" in block and "Merged f (MHz)" in block
    assert "37974.506" in block  # the merged frequency
    assert "merge" in block.lower()
    # Without merges the empty-history note is shown instead.
    empty = "\n".join(_audit_block(wf, 100.0, -1.0, None))
    assert "No add-one-peak history" in empty


def test_plot_summary_histograms():
    from ftmwpipeline.visualization.fit_detail import plot_summary_histograms

    fig = plot_summary_histograms(
        [
            ("chi2", "x", [1.0, 1.2, 1.5, 2.0, 5.0]),
            ("empty", "y", []),
            ("snr", "s", [3.0, 10.0, 100.0, 500.0]),
        ]
    )
    try:
        assert fig is not None
        # Only the two non-empty specs become panels.
        drawn = [ax for ax in fig.axes if ax.has_data()]
        assert len(drawn) == 2
    finally:
        plt.close(fig)
    assert plot_summary_histograms([("a", "x", []), ("b", "y", [])]) is None


def _final_peak(freq_mhz, **kw):
    defaults = dict(
        frequency_mhz=freq_mhz,
        frequency_raw_mhz=freq_mhz,
        f_baseband_mhz=abs(freq_mhz - 40000.0),
        sigma_f_khz=0.92,
        sigma_stat_khz=0.30,
        sigma_eps_khz=0.86,
        sigma_floor_khz=0.0,
        amplitude=6.0e-6,
        phase=1.73,
        snr=144.0,
    )
    defaults.update(kw)
    return FinalPeak(**defaults)


def test_window_peak_table_letters_low_to_high_and_bce():
    # Pass peaks out of frequency order; letters must still go low->high.
    peaks = [
        _final_peak(
            29148.12838,
            sigma_f_khz=0.936,
            amplitude=5.217e-6,
            amplitude_error=0.033e-6,
            phase=1.738,
            phase_error=0.010,
            snr=117.96,
            snr_error=0.75,
        ),
        _final_peak(
            29147.44669,
            sigma_f_khz=0.917,
            amplitude=6.377e-6,
            amplitude_error=0.036e-6,
            phase=1.7303,
            phase_error=0.0086,
            snr=144.17,
            snr_error=0.81,
        ),
    ]
    table = _window_peak_table(peaks, "uV", 1e-6)
    assert "<th>Peak</th>" in table
    # The lower-frequency peak (29147...) is 'A', the higher ('B'); the table
    # rows stay in the given order, so 'A' annotates the second row's peak.
    assert "<td>A</td>" in table and "<td>B</td>" in table
    a_pos = table.index("29147.44669")
    b_pos = table.index("29148.12838")
    # 'B' label appears before 'A' label in source (first row is the high peak),
    # but the A row holds the low frequency.
    assert table.index("<td>A</td>") < a_pos
    assert table.index("<td>B</td>") < b_pos
    # BCE value(unc) for calibrated f / amplitude / phase / snr.
    assert "29147.44669(92)" in table  # f with sigma_f 0.917 kHz
    assert "6.377(36)" in table  # amplitude in uV
    assert "1.7303(86)" in table  # phase
    assert "144.17(81)" in table  # snr
    # Raw frequency stays plain (not BCE).
    assert "29147.446690" in table or "29147.44669" in table


def test_window_peak_table_curation_markup():
    # Without a window id (the pure-unit form) the table carries no curation
    # markup at all -- the clean read-only document.
    peaks = [_final_peak(29148.0, frequency_raw_mhz=29148.001234)]
    plain = _window_peak_table(peaks, "uV", 1e-6)
    assert "data-window" not in plain and "cur-cell" not in plain
    assert "Curate" not in plain

    # With a window id, every row gains data-window / data-freq (the RAW model
    # frequency, the value the edit verbs match on) and a trailing control cell.
    cur = _window_peak_table(peaks, "uV", 1e-6, window_id=217)
    assert 'data-window="217"' in cur
    assert 'data-freq="29148.001234"' in cur  # raw, not the calibrated 29148.0
    assert 'data-act="remove"' in cur and 'data-act="split"' in cur
    assert 'class="cur-merge"' in cur
    assert "cur-col-h" in cur  # the gated Curate header label
    # The curation column is the trailing cell (CSS hides it via :last-child).
    assert cur.rfind("cur-cell") > cur.rfind('class="badge')


def test_ledger_block_curation_add():
    from ftmwpipeline._internal.report_html_impl import _ledger_block
    from ftmwpipeline.core.data_structures import LedgerCandidate

    c = LedgerCandidate(
        frequency_mhz=38502.7,
        seed_offset_mhz=0.12,
        seed_amplitude=None,
        best_evidence=7.3,
        evidence_kind="matched-filter",
        reasons=["below add-one gate"],
        decision_sites=["w309"],
        window_id=309,
    )
    plain = "\n".join(_ledger_block([c]))
    assert "data-window" not in plain and 'data-act="add"' not in plain

    cur = "\n".join(_ledger_block([c], window_id=309))
    assert 'data-window="309"' in cur
    assert 'data-freq="38502.700000"' in cur
    assert 'data-act="add"' in cur


def test_window_curation_controls():
    from ftmwpipeline._internal.report_html_impl import _window_curation_controls

    block = "\n".join(_window_curation_controls(24, 38449.0, 38451.0))
    assert 'class="cur-only cur-window-controls"' in block
    assert 'data-act="merge-selected"' in block and 'data-window="24"' in block
    assert 'data-act="add-typed"' in block and 'class="cur-addfreq"' in block
    assert 'data-act="accept"' in block  # the bare "Mark reviewed"
    assert "38449.0000" in block and "38451.0000" in block  # the window range


def test_catalog_cell_and_window_table_column():
    from ftmwpipeline._internal.catalog_xref import CatalogMatch
    from ftmwpipeline._internal.report_html_impl import _catalog_cell

    m = CatalogMatch("line-A", 29147.4, 0.0, 1.2, 1.0, 1.2)
    cell = _catalog_cell(m)
    assert 'class="badge cat"' in cell
    assert "line-A" in cell
    assert "pull" in cell  # tooltip
    assert _catalog_cell(None) == ""

    peaks = [_final_peak(29148.0), _final_peak(29147.4)]
    matches = [None, m]
    table = _window_peak_table(peaks, "uV", 1e-6, matches)
    assert "<th>Catalog</th>" in table
    assert "line-A" in table
    # Without matches, no catalog column.
    assert "<th>Catalog</th>" not in _window_peak_table(peaks, "uV", 1e-6)


def test_index_final_table_catalog_column():
    from ftmwpipeline._internal.catalog_xref import CatalogMatch
    from ftmwpipeline._internal.report_html_impl import _index_final_table
    from ftmwpipeline.core.data_structures import FinalProducts

    prod = FinalProducts(peaks=[_final_peak(30000.0), _final_peak(31000.0)])
    matches = [CatalogMatch("X", 30000.0, 0.0, 0.0, 1.0, 0.0), None]
    html_out = _index_final_table(prod, matches)
    assert "<th>Catalog</th>" in html_out
    assert "X" in html_out
    assert "<th>Catalog</th>" not in _index_final_table(prod)


def test_summary_distribution_specs_appends_pull():
    from types import SimpleNamespace

    from ftmwpipeline._internal.catalog_xref import CatalogCrossRef
    from ftmwpipeline._internal.report_html_impl import _summary_distribution_specs

    model = SimpleNamespace(
        chi2r_values=[1.0],
        eps_values=[1.0],
        sigma_stat_values=[1.0],
        sigma_eps_values=[1.0],
        sigma_f_values=[1.0],
        snr_values_promoted=[10.0],
    )
    assert len(_summary_distribution_specs(model)) == 6
    xref = CatalogCrossRef(
        catalog_path="c", n_sigma=3.0, n_catalog=2, matches=[], pull_values=[0.5, -0.3]
    )
    specs = _summary_distribution_specs(model, xref)
    assert len(specs) == 7
    assert specs[-1][2] == [0.5, -0.3]


def test_name_helpers_zero_pad():
    assert _window_page_name(7) == "window_007.html"
    assert _panel_figure_name("exp_2638", 7, "mag") == "exp_2638_window_007_mag.png"


def test_unknown_windows_filter_raises(tmp_path):
    fp = tmp_path / "empty.ftmw"
    with h5py.File(fp, "w") as h5f:
        h5f.attrs["x"] = 1
    with pytest.raises(ValueError, match="windows filter"):
        report_full_impl(str(fp), output_dir=str(tmp_path / "out"), windows="bogus")


def test_missing_products_raises(tmp_path):
    fp = tmp_path / "empty.ftmw"
    with h5py.File(fp, "w") as h5f:
        h5f.attrs["x"] = 1
    with pytest.raises(ValueError, match="review run"):
        report_full_impl(str(fp), output_dir=str(tmp_path / "out"))


# ---------------------------------------------------------------------------
# Integration on the small 2638 fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stage5_small_file(tmp_path_factory):
    data_path = Path("examples/blackchirp_data/2638")
    if not data_path.exists():
        pytest.skip("Experiment 2638 data not available")
    tmp = tmp_path_factory.mktemp("stage5_small_full")
    fp = tmp / "2638_full.ftmw"

    ftmw.import_data(fp, source=str(data_path))
    ftmw.compute_ft(fp, trim=(26500, 40000))
    ftmw.estimate_noise(fp)
    ftmw.calibrate_tau(fp)
    ftmw.detect_peaks(fp)
    ftmw.assign_windows(fp)

    plan = load_windows_impl(str(fp))["plan"]
    candidates: List[int] = []
    for wid in plan.topological_order:
        deps = [(a, b) for (a, b) in plan.dependency_edges if a == wid or b == wid]
        if all(a in candidates or a == wid for (a, _) in deps) and all(
            b in candidates or b == wid for (_, b) in deps
        ):
            candidates.append(wid)
        if len(candidates) >= 3:
            break
    if not candidates:
        candidates = list(plan.topological_order[:3])
    keep = set(candidates)
    plan.windows = [w for w in plan.windows if w.window_id in keep]
    plan.topological_order = [w for w in plan.topological_order if w in keep]
    plan.dependency_edges = [
        (a, b) for (a, b) in plan.dependency_edges if a in keep and b in keep
    ]
    save_window_plan_impl(str(fp), plan)

    ftmw.fit_peaks(str(fp))
    review_run_impl(str(fp))
    return fp


def _assert_wellformed(path: Path):
    for f in path.rglob("*.html"):
        html.parser.HTMLParser().feed(f.read_text())


@pytest.mark.integration
def test_full_site_structure(stage5_small_file, tmp_path):
    # The assembled working site (pre-collapse) carries the structural invariants;
    # report_full_impl folds this into one self-contained file.
    out = tmp_path / "site"
    _assemble_report_site(str(stage5_small_file), out_root=str(out))

    assert (out / "index.html").exists()
    assert (out / "assets" / "style.css").exists()
    pages = list((out / "windows").glob("*.html"))
    figures = list((out / "figures").glob("*.png"))
    assert pages and figures
    # Per window: the four zoomed panels (re/im/mag/hist) plus an optional
    # correlation heatmap ("_corr.png"). The full-spectrum context is the single
    # shared overview image -- there is NO per-window context PNG. The index has
    # one overview ("<stem>_overview.png") and one figure per distribution group
    # ("<stem>_hist_<slug>.png").
    zoom_suffixes = ("_re.png", "_im.png", "_mag.png", "_hist.png")
    corr_pngs = [f for f in figures if f.name.endswith("_corr.png")]
    panel_pngs = [f for f in figures if f.name.endswith(zoom_suffixes)]
    ctx_pngs = [f for f in figures if f.name.endswith("_ctx.png")]
    overview_pngs = [
        f
        for f in figures
        if f.name.endswith("_overview.png") and "_window_" not in f.name
    ]
    hist_pngs = [f for f in figures if "_hist_" in f.name]
    assert len(panel_pngs) == len(pages) * len(zoom_suffixes)
    assert len(ctx_pngs) == 0  # the shared overview is the context; no per-window PNG
    assert len(corr_pngs) <= len(pages)
    # The single shared full-spectrum overview image.
    assert len(overview_pngs) == 1
    # Distribution histograms (SNR / fit-quality / σ_f budget; pull only with a
    # catalog), one figure per group that has data.
    assert 1 <= len(hist_pngs) <= 4
    # The Level-2 methods page is built and linked from the index summary.
    assert (out / "methods.html").exists()
    _assert_wellformed(out)

    idx = (out / "index.html").read_text()
    assert "FTMW pipeline report" in idx
    assert "Final line list" in idx
    assert 'href="windows/window_' in idx  # links to the window pages
    assert 'href="methods.html"' in idx
    # The interactive full-spectrum overview: the shared image is the div's CSS
    # background (bound in the stylesheet, not an <img> on the page) carrying an
    # SVG overlay of clickable, attention-tinted per-window rects.
    assert "<h2>Spectrum</h2>" in idx
    assert 'class="spectrum-ctx"' in idx
    assert "specnav-rect" in idx
    assert "data-window=" in idx
    assert "<title>" in idx  # native tooltip per rect
    assert 'data-thumb="figures/' in idx  # thumbnail for the popup
    assert "winmap-pop" in idx  # the hover-zoom script wired in
    # Each window rect links to its page from inside the SVG.
    assert idx.count('<a href="windows/window_') >= 1
    # The overview image is bound once as the interactive-overview background.
    css = (out / "assets" / "style.css").read_text()
    assert "_overview.png" in css and ".spectrum-ctx { background-image:" in css
    methods = (out / "methods.html").read_text()
    # Histograms are interleaved beside their tables (not one trailing figure).
    assert 'class="hist"' in methods
    assert "_hist_" in methods
    # MathJax renders the methods-page equations (raw TeX stays as fallback).
    assert "MathJax" in methods
    assert 'class="equation"' in methods

    page = pages[0].read_text()
    # The spectrum context is the shared interactive overview with this window
    # highlighted -- no per-window context image.
    assert "<h2>Spectrum context</h2>" in page
    assert "_ctx.png" not in page
    assert 'class="spectrum-ctx"' in page and "specnav-rect" in page
    assert "<h2>Fit</h2>" in page
    # The fit detail is a responsive grid of the zoomed Re/Im/|X|/hist panels.
    assert 'class="fit-panels"' in page
    assert 'class="panel-grid"' in page
    for panel in ("re", "im", "mag", "hist"):
        assert f"_{panel}.png" in page
    assert "Fitted lines" in page
    assert "Parameter covariance" in page
    assert "Ledger candidates" in page
    assert "<h2>Fit history</h2>" in page
    # The raw fit-log dump was removed (it duplicated the structured tables).
    assert "Fit log" not in page
    # The interactive overview doubles as the quick-nav on each window page too,
    # with this window highlighted and figure/page paths relative to windows/.
    assert 'class="spectrum-ctx"' in page
    # The "you are here" highlight: the current window's rect carries the
    # `current` class (alongside `attn` when it is also flagged).
    assert "specnav-rect" in page and 'current"' in page
    assert 'data-thumb="../figures/' in page  # thumbnail path from a window page
    assert 'href="window_' in page  # same-dir links (no windows/ prefix)
    assert "winmap-pop" in page  # the hover-zoom script


@pytest.mark.integration
def test_full_windows_filter_attention_subset(stage5_small_file, tmp_path):
    all_out = tmp_path / "all"
    att_out = tmp_path / "att"
    _assemble_report_site(str(stage5_small_file), out_root=str(all_out), windows="all")
    _assemble_report_site(
        str(stage5_small_file), out_root=str(att_out), windows="attention"
    )

    n_all = len(list((all_out / "windows").glob("*.html")))
    n_att = len(list((att_out / "windows").glob("*.html")))
    # The attention filter never produces more pages than 'all'.
    assert 0 <= n_att <= n_all
    # The index still lists every window in both modes.
    assert "<code>attention</code> filter" in (att_out / "index.html").read_text()


@pytest.mark.integration
def test_full_with_catalog(stage5_small_file, tmp_path):
    from ftmwpipeline.io.stage6_review_serialization import load_stage6_review_from_file

    # Build a catalog on the actual fitted-line frequencies so every line matches.
    products = load_stage6_review_from_file(str(stage5_small_file)).final_products
    assert products is not None and products.peaks
    cat = tmp_path / "cat.csv"
    cat.write_text(
        "frequency_mhz,uncertainty_khz,label\n"
        + "".join(
            f"{p.frequency_mhz},0.0,cat-{i}\n" for i, p in enumerate(products.peaks)
        )
    )
    out = tmp_path / "site"
    _assemble_report_site(str(stage5_small_file), out_root=str(out), catalog=str(cat))

    idx = (out / "index.html").read_text()
    assert "<th>Catalog</th>" in idx
    assert "cat-0" in idx
    assert "proximity only" in idx
    methods = (out / "methods.html").read_text()
    assert "Catalog cross-reference" in methods
    # The catalog pull histogram is rendered as its own group figure.
    figures = [f.name for f in (out / "figures").glob("*.png")]
    assert any(f.endswith("_hist_pull.png") for f in figures)
    pages = list((out / "windows").glob("*.html"))
    assert any("<th>Catalog</th>" in p.read_text() for p in pages)
    _assert_wellformed(out)


@pytest.mark.integration
def test_full_is_read_only(stage5_small_file, tmp_path):
    fp = tmp_path / "ro.ftmw"
    shutil.copy(stage5_small_file, fp)
    before = hashlib.md5(fp.read_bytes()).hexdigest()
    report_full_impl(str(fp), output_dir=str(tmp_path / "out"))
    assert hashlib.md5(fp.read_bytes()).hexdigest() == before


@pytest.mark.integration
def test_single_file_summary_is_self_contained(stage5_small_file, tmp_path):
    out = tmp_path / "sf"
    path = report_full_impl(
        str(stage5_small_file), output_dir=str(out), scope="summary"
    )
    p = Path(path)
    assert p.name.endswith("_report_summary.html")
    assert p.exists() and p.parent == out
    # Exactly one self-contained file -- no scratch site dir / assets / figures.
    assert list(out.glob("*.html")) == [p]
    assert not (out / "figures").exists() and not (out / "assets").exists()

    doc = p.read_text()
    html.parser.HTMLParser().feed(doc)
    # No external asset references survive: CSS inlined, figures base64-embedded,
    # cross-page links rewritten to in-document anchors.
    import re as _re

    assert "<style>" in doc and 'rel="stylesheet"' not in doc
    assert 'src="figures/' not in doc and 'src="../figures/' not in doc
    assert "data:image/png;base64," in doc
    # No page-to-page file links survive (every href is an in-document anchor).
    assert not _re.search(r'href="[^"#][^"]*\.html"', doc)
    # Index + methods are both present, methods reachable by anchor.
    assert 'id="methods"' in doc and 'href="#methods"' in doc
    assert "Methods and results" in doc  # the methods-page content folded in
    # Summary carries no per-window section (the #window-list nav anchor is fine).
    assert not _re.search(r'id="window-\d', doc)
    # Hover previews survive via one deduplicated thumbnail map: data-thumb is a
    # bare basename key (no path) resolved against window.__thumbs.
    assert "window.__thumbs=" in doc
    assert _re.search(r'data-thumb="[^"/]+_mag\.png"', doc)
    assert not _re.search(r'data-thumb="[^"]*/', doc)  # no path-form keys leak
    # Sticky section nav: Overview/Methods/Windows/Line list, no window jump.
    assert 'class="topnav"' in doc and 'href="#methods"' in doc
    assert 'href="#window-list"' in doc and 'id="window-list"' in doc
    assert 'href="#final-list"' in doc and 'id="final-list"' in doc
    assert 'class="winjump"' not in doc


@pytest.mark.integration
def test_single_file_full_folds_in_window_pages(stage5_small_file, tmp_path):
    out = tmp_path / "sff"
    path = report_full_impl(
        str(stage5_small_file), output_dir=str(out), scope="full"
    )
    p = Path(path)
    assert p.name.endswith("_report.html") and not p.name.endswith("_summary.html")
    assert list(out.glob("*.html")) == [p]
    assert not (out / "windows").exists()

    doc = p.read_text()
    html.parser.HTMLParser().feed(doc)
    assert 'src="figures/' not in doc and "data:image/png;base64," in doc
    # Every per-window section is present and every window link is an anchor that
    # resolves to one of those sections (no dangling file links).
    import re as _re

    section_ids = set(_re.findall(r'id="window-(\d+)"', doc))
    link_ids = set(_re.findall(r'href="#window-(\d+)"', doc))
    assert section_ids and link_ids <= section_ids
    assert 'href="windows/window_' not in doc
    # The jump-to-window picker offers every window section as an anchor target.
    assert 'class="winjump"' in doc
    jump_ids = set(_re.findall(r'<option value="#window-(\d+)"', doc))
    assert jump_ids == section_ids
    # Nav links to the index window list and final line list.
    assert 'href="#window-list"' in doc and 'href="#final-list"' in doc


@pytest.mark.integration
def test_single_file_carries_curation_surface(stage5_small_file, tmp_path):
    out = tmp_path / "cur"
    path = report_full_impl(str(stage5_small_file), output_dir=str(out), scope="full")
    doc = Path(path).read_text()
    html.parser.HTMLParser().feed(doc)

    # The boot script + curation script + topnav chrome ship in the single file.
    assert "window.__stem=" in doc
    assert "curation-enabled" in doc  # the script adds it and the CSS gates on it
    assert 'class="cur-toggle"' in doc and 'class="cur-badge"' in doc
    # Per-row controls + raw-frequency data attributes on fitted-line rows.
    assert 'data-act="remove"' in doc and 'data-act="split"' in doc
    import re as _re

    assert _re.search(r'data-window="\d+" data-freq="[0-9.]+"', doc)
    # Per-window controls (merge selected / add / mark reviewed) gated by cur-only.
    assert 'class="cur-only cur-window-controls"' in doc
    assert 'data-act="merge-selected"' in doc and 'data-act="add-typed"' in doc
    # The curation CSS gating rule is present (single-class governance).
    assert "html:not(.curation-enabled)" in doc


def _peak_list_row(doc: str):
    """Return ``(window_id, raw_freq_str)`` of one fitted-line row in the report."""
    import re as _re

    for tbl in _re.findall(r'<table class="peak-list">.*?</table>', doc, _re.S):
        m = _re.search(r'data-window="(\d+)" data-freq="([0-9.]+)"', tbl)
        if m:
            return int(m.group(1)), m.group(2)
    pytest.skip("No fitted-line rows in the built subset")


@pytest.mark.integration
def test_curation_emitted_frequency_resolves(stage5_small_file, tmp_path):
    """The load-bearing rule: the raw frequency a control emits targets the
    intended peak when fed straight to ``review apply``."""
    from ftmwpipeline._internal.stage6_impl import apply_curation_impl
    from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5

    out = tmp_path / "lb"
    path = report_full_impl(str(stage5_small_file), output_dir=str(out), scope="full")
    doc = Path(path).read_text()
    wid, freq = _peak_list_row(doc)

    def _window_raw_freqs(fp):
        with h5py.File(str(fp), "r") as h5f:
            sf = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
        for wf in sf.window_fits:
            if int(wf.window_id) == wid:
                return [round(float(p.frequency_mhz), 6) for p in wf.fitted_peaks]
        return []

    target = tmp_path / "lb.ftmw"
    shutil.copy(stage5_small_file, target)
    before = _window_raw_freqs(target)
    assert round(float(freq), 6) in before  # the emitted attr is a real peak

    cur = tmp_path / "remove.csv"
    cur.write_text(f"action,window,freqs,params\nremove,{wid},{freq},\n")
    apply_curation_impl(str(target), str(cur))

    after = _window_raw_freqs(target)
    # That specific peak is gone; the refit did not simply re-add it.
    assert round(float(freq), 6) not in after
    assert len(after) == len(before) - 1


@pytest.mark.integration
def test_report_rejects_unknown_scope(stage5_small_file, tmp_path):
    with pytest.raises(ValueError, match="scope"):
        report_full_impl(
            str(stage5_small_file), output_dir=str(tmp_path / "x"), scope="bogus"
        )


@pytest.mark.integration
def test_full_cross_interface(stage5_small_file, tmp_path):
    fp = tmp_path / "copy.ftmw"
    shutil.copy(stage5_small_file, fp)

    impl_dir = tmp_path / "impl"
    api_dir = tmp_path / "api"
    pipe_dir = tmp_path / "pipe"

    from ftmwpipeline._internal.report_html_impl import report_full_impl as impl

    # The report is one self-contained file across every interface; the impl and
    # the report_run wrappers must render byte-identical HTML (paths inside are
    # in-document anchors / data URIs, so only the output directory differs).
    impl(str(fp), output_dir=str(impl_dir))
    ftmw.report_run(str(fp), output_dir=str(api_dir), emit_table=False)
    Pipeline.open(fp).report_run(output_dir=str(pipe_dir), emit_table=False)

    def _single(d):
        return next(p for p in Path(d).glob("*_report.html"))

    a = _single(impl_dir).read_text()
    b = _single(api_dir).read_text()
    c = _single(pipe_dir).read_text()
    assert a == b == c


# ---------------------------------------------------------------------------
# report run (the default Stage 6 deliverable: L1 table + L3 report)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_full_emits_per_window_progress_logs(stage5_small_file, tmp_path, caplog):
    """The render loop logs one ``window i/N`` per window for the progress bridge."""
    import logging

    logger_name = "ftmwpipeline._internal.report_html_impl"
    with caplog.at_level(logging.INFO, logger=logger_name):
        report_full_impl(str(stage5_small_file), output_dir=str(tmp_path / "out"))
    win_msgs = [
        r.getMessage()
        for r in caplog.records
        if r.name == logger_name and r.getMessage().startswith("window ")
    ]
    assert win_msgs and win_msgs[0].startswith("window 1/")
    total = win_msgs[0].split("/")[1]
    assert win_msgs[-1] == f"window {total}/{total}"  # the bar reaches 100%


@pytest.mark.integration
def test_run_default_writes_table_and_single_file(stage5_small_file, tmp_path):
    from ftmwpipeline._internal.report_html_impl import report_run_impl

    out = tmp_path / "run"
    result = report_run_impl(str(stage5_small_file), output_dir=str(out))

    table_p = Path(result["table"])
    html_p = Path(result["html"])
    assert table_p.exists() and table_p.suffix == ".csv"
    assert table_p.name.endswith("_lines.csv")
    # The default HTML is the self-contained full single-file report.
    assert html_p.exists() and html_p.name.endswith("_report.html")
    assert not html_p.name.endswith("_summary.html")
    doc = html_p.read_text()
    assert "data:image/png;base64," in doc and 'src="figures/' not in doc
    # The compact toggle and its script ship in the single-file build.
    assert 'class="compact-toggle"' in doc and "report-compact" in doc


@pytest.mark.integration
def test_run_level1_only_skips_html(stage5_small_file, tmp_path):
    from ftmwpipeline._internal.report_html_impl import report_run_impl

    out = tmp_path / "l1"
    result = report_run_impl(
        str(stage5_small_file), output_dir=str(out), emit_html=False
    )
    assert result["table"] is not None and result["html"] is None
    assert list(out.glob("*.html")) == []
    assert Path(result["table"]).exists()


@pytest.mark.integration
def test_run_no_table_skips_table(stage5_small_file, tmp_path):
    from ftmwpipeline._internal.report_html_impl import report_run_impl

    out = tmp_path / "htmlonly"
    result = report_run_impl(
        str(stage5_small_file), output_dir=str(out), emit_table=False
    )
    assert result["table"] is None and result["html"] is not None
    assert list(out.glob("*_lines.csv")) == []


def test_run_rejects_empty_output():
    from ftmwpipeline._internal.report_html_impl import report_run_impl

    with pytest.raises(ValueError, match="nothing to do"):
        report_run_impl(
            "ignored.ftmw", output_dir="x", emit_table=False, emit_html=False
        )


@pytest.mark.integration
def test_run_cross_interface(stage5_small_file, tmp_path):
    fp = tmp_path / "rcopy.ftmw"
    shutil.copy(stage5_small_file, fp)

    from ftmwpipeline._internal.report_html_impl import report_run_impl

    impl_out = tmp_path / "rimpl"
    api_out = tmp_path / "rapi"
    pipe_out = tmp_path / "rpipe"

    report_run_impl(str(fp), output_dir=str(impl_out))
    ftmw.report_run(str(fp), output_dir=str(api_out))
    Pipeline.open(fp).report_run(output_dir=str(pipe_out))

    def _single(d):
        return next(p for p in Path(d).glob("*_report.html"))

    a = _single(impl_out).read_text()
    b = _single(api_out).read_text()
    c = _single(pipe_out).read_text()
    assert a == b == c
