"""Tests for the Level-3 ``report full`` linked-HTML site.

Two layers: fast unit tests over the pure HTML helpers (no fixture fit needed),
and integration tests that run a real ``review run`` consolidation on the small
2638 fixture, assemble the site, and check the structure / well-formedness /
read-only / cross-interface (api == Pipeline == impl) guarantees.
"""

from __future__ import annotations

import collections
import hashlib
import html.parser
import shutil
from pathlib import Path
from typing import List

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import ftmwpipeline.api as ftmw  # noqa: E402
from ftmwpipeline._internal.report_html_impl import (  # noqa: E402
    _LIGHTBOX_JS,
    _STYLESHEET,
    _assemble_report_site,
    _covariance_block,
    _esc,
    _page,
    _panel_figure_name,
    _table,
    _window_page_name,
    _window_peak_table,
    report_full_impl,
)
from ftmwpipeline._internal.stage4_impl import (  # noqa: E402
    load_windows_impl,
    save_window_plan_impl,
)
from ftmwpipeline._internal.stage6_impl import (  # noqa: E402
    get_candidate_ledger_impl,
    review_run_impl,
)
from ftmwpipeline.core.data_structures import FinalPeak  # noqa: E402
from ftmwpipeline.pipeline import Pipeline  # noqa: E402

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
    # Full size is the default: the <html> element must NOT carry report-compact.
    html_tag = doc[
        doc.index("<!DOCTYPE html>") : doc.index(">", doc.index("<html")) + 1
    ]
    assert "report-compact" not in html_tag


def test_responsive_panel_grid_styles():
    """The primary plots reflow (auto-fit) and fill their cell -- no fixed
    two-column grid or fixed image height."""
    assert "repeat(auto-fit, minmax(440px, 1fr))" in _STYLESHEET
    assert ".fit-panels img { width: 100%" in _STYLESHEET
    assert "height: 340px" not in _STYLESHEET  # the old fixed panel height is gone
    # The full-size lightbox overlay is styled.
    assert ".lightbox" in _STYLESHEET
    assert ".lightbox.open" in _STYLESHEET


def test_lightbox_triggers_on_plots_and_yields_to_curation():
    """Clicking a primary plot opens the lightbox, except on an armed curation
    plot where the add-marker handler must win."""
    assert ".fit-panels img" in _LIGHTBOX_JS  # the click trigger
    assert "cur-plot-wrap.cur-armed" in _LIGHTBOX_JS  # curation add-marker gate


class _TagWf:
    """Minimal per-window stand-in for the tag classifier."""

    def __init__(self, chi2r, snrs):
        self.reduced_chi2 = chi2r
        self.fitted_peaks = [_TagPeak(s) for s in snrs]


class _TagPeak:
    def __init__(self, snr):
        self.snr = snr


class _TagStatus:
    def __init__(self, needs_attention=False, provenance="auto"):
        self._na = needs_attention
        self.provenance = provenance

    @property
    def needs_attention(self):
        return self._na


def test_window_tags_classification_and_precedence():
    from ftmwpipeline._internal.report_html_impl import _window_tags

    # A clean, untouched, well-fit window earns no tags.
    clean = _TagWf(1.0, [50.0])
    assert (
        _window_tags(
            wf=clean,
            status=_TagStatus(),
            edited=False,
            reviewed=False,
            merged=False,
            cascade=False,
            catalog_match=False,
        )
        == []
    )

    # A user edit supersedes both the cascade and reviewed provenance tags, and
    # the queue / merge / fit-quality / catalog tags ride alongside in order.
    tags = _window_tags(
        # χ²ᵣ past the misfit bar with a low-SNR peak, so ε = sqrt((χ²ᵣ-floor)/
        # floor)/snr clears 5% too (high SNR would shrink ε below the bar).
        wf=_TagWf(20.0, [15.0]),
        status=_TagStatus(needs_attention=True),
        edited=True,
        reviewed=True,
        cascade=True,
        merged=True,
        catalog_match=True,
    )
    assert "edited" in tags
    assert "cascade-edit" not in tags and "reviewed" not in tags
    assert {"attention", "merged", "high-chi2r", "high-eps", "catalog-match"} <= set(
        tags
    )
    # Canonical display order is preserved.
    from ftmwpipeline._internal.report_html_impl import _WINDOW_TAG_ORDER

    assert tags == [t for t in _WINDOW_TAG_ORDER if t in tags]

    # A passive cascade dependent (material change, no user edit) tags as such.
    casc = _window_tags(
        wf=_TagWf(1.0, [50.0]),
        status=_TagStatus(),
        edited=False,
        reviewed=True,
        cascade=True,
        merged=False,
        catalog_match=False,
    )
    assert casc == ["cascade-edit"]


def test_window_tag_chips_markup():
    from ftmwpipeline._internal.report_html_impl import _window_tag_chips

    assert _window_tag_chips([]) == ""
    out = _window_tag_chips(["attention", "merged"])
    assert 'class="win-tags"' in out
    assert 'class="win-tag win-tag-attention"' in out
    assert 'class="win-tag win-tag-merged"' in out


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
    # Only a Remove control lives there now: split/merge are inferred from an
    # add, not typed at the row (see _peak_curation_cell).
    cur = _window_peak_table(peaks, "uV", 1e-6, window_id=217)
    assert 'data-window="217"' in cur
    assert 'data-freq="29148.001234"' in cur  # raw, not the calibrated 29148.0
    assert 'data-act="remove"' in cur
    assert 'data-act="split"' not in cur and 'class="cur-merge"' not in cur
    assert "cur-col-h" in cur  # the gated Curate header label
    # The curation column is the trailing cell (CSS hides it via :last-child).
    assert cur.rfind("cur-cell") > cur.rfind('class="badge')


def test_window_peak_table_emits_peak_uid():
    """A curation row carries ``data-uid`` when the peak has an identifier.

    The cart exports it as a ``uid:N`` remove token, which addresses the peak
    exactly rather than by proximity. A peak from a fit predating ``peak_uid``
    carries no attribute at all, so the cart falls back to the frequency --
    the two cases must stay distinguishable in the markup.
    """
    stamped = _final_peak(29148.0, frequency_raw_mhz=29148.001234, peak_uid=15425022)
    cur = _window_peak_table([stamped], "uV", 1e-6, window_id=217)
    assert 'data-freq="29148.001234" data-uid="15425022"' in cur

    legacy = _final_peak(29148.0, frequency_raw_mhz=29148.001234)
    assert legacy.peak_uid is None
    old = _window_peak_table([legacy], "uV", 1e-6, window_id=217)
    assert "data-uid" not in old
    assert 'data-freq="29148.001234"' in old  # still addressable by frequency

    # Never emitted outside curation mode -- the read-only document stays clean.
    assert "data-uid" not in _window_peak_table([stamped], "uV", 1e-6)


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
    # No merge-selected control: split/merge are inferred from add/remove.
    assert 'data-act="merge-selected"' not in block
    assert 'data-act="add-typed"' in block and 'class="cur-addfreq"' in block
    assert 'data-window="24"' in block
    assert "38449.0000" in block and "38451.0000" in block  # the window range
    # The accept / accept-next / clear verbs now live in the title bar, not here.
    assert 'data-act="accept"' not in block
    assert 'data-act="accept-next"' not in block
    assert 'data-act="clear-window"' not in block


def test_window_header_bar():
    from ftmwpipeline._internal.report_html_impl import _window_header_bar

    bar = _window_header_bar(24, 23, 25)
    assert 'class="win-header-bar"' in bar
    # Navigation buttons (index + prev/next window), styled as buttons.
    assert 'class="win-navbtn" href="../index.html"' in bar
    assert "window 23" in bar and "window 25" in bar
    # Curation verbs moved here, curate-only, no flag glyph on "Reviewed & next".
    assert 'class="cur-only win-cur-actions"' in bar
    assert 'data-act="accept"' in bar and 'data-act="accept-next"' in bar
    assert 'data-act="clear-window"' in bar
    assert "&#9873;" not in bar  # no flag icon
    # Endpoints: no prev/next link when there is no neighbor.
    first = _window_header_bar(0, None, 5)
    assert "&larr; window" not in first and "window 5 &rarr;" in first


def test_applied_edits_section():
    from ftmwpipeline._internal.report_html_impl import _applied_edits_section
    from ftmwpipeline.core.data_structures import DecisionLogEntry

    # No decisions -> no section.
    assert _applied_edits_section([]) == []

    decisions = [
        DecisionLogEntry(
            order_index=0, window_id=24, frequency_mhz=38450.123, kind="remove"
        ),
        DecisionLogEntry(order_index=1, window_id=7, frequency_mhz=0.0, kind="accept"),
    ]
    html_block = "\n".join(_applied_edits_section(decisions))
    assert 'id="applied-edits"' in html_block
    # Each decision lists with an undo control carrying its id + window.
    assert 'data-act="undo"' in html_block
    assert 'data-edit-id="0"' in html_block and 'data-window="24"' in html_block
    assert "38450.1230" in html_block  # the remove anchor
    # The undo control is curate-only (never alters the read-only view).
    assert 'class="cur-only cur-btn cur-undo"' in html_block


def test_mag_axes_geometry_and_click_inversion():
    from ftmwpipeline._internal.report_html_impl import _mag_axes_geometry

    fig = plt.figure(figsize=(10.0, 4.0))
    ax = fig.add_axes([0.1, 0.2, 0.8, 0.7])  # left, bottom, width, height
    ax.set_xlim(26500.0, 40000.0)
    ax.set_xlabel("frequency (MHz)")
    fig.canvas.draw()  # finalize the layout the way savefig does
    try:
        geom = _mag_axes_geometry(fig, dpi=100)
        assert geom is not None
        # Natural width = 10 in * 100 dpi = 1000 px; the axes span x∈[0.1,0.9].
        assert geom["x0"] == pytest.approx(100.0)
        assert geom["x1"] == pytest.approx(900.0)
        assert geom["w"] == pytest.approx(1000.0)
        assert geom["flo"] == pytest.approx(26500.0)
        assert geom["fhi"] == pytest.approx(40000.0)

        # Replicate the overlay's linear inversion: a click x (natural px) -> MHz.
        def invert(nat):
            return geom["flo"] + (nat - geom["x0"]) / (geom["x1"] - geom["x0"]) * (
                geom["fhi"] - geom["flo"]
            )

        assert invert(100.0) == pytest.approx(26500.0)  # left edge
        assert invert(900.0) == pytest.approx(40000.0)  # right edge
        assert invert(500.0) == pytest.approx(33250.0)  # axes midpoint -> midband
    finally:
        plt.close(fig)

    # A descending x-axis carries its sense in flo/fhi, so the same inversion holds.
    fig2 = plt.figure(figsize=(10.0, 4.0))
    ax2 = fig2.add_axes([0.1, 0.2, 0.8, 0.7])
    ax2.set_xlim(40000.0, 26500.0)
    ax2.set_xlabel("frequency (MHz)")
    fig2.canvas.draw()
    try:
        g2 = _mag_axes_geometry(fig2, dpi=100)
        assert g2["flo"] == pytest.approx(40000.0) and g2["fhi"] == pytest.approx(
            26500.0
        )
    finally:
        plt.close(fig2)

    # No frequency axis -> no geometry (the painter that lacks the xlabel).
    fig3 = plt.figure(figsize=(4.0, 4.0))
    fig3.add_subplot(111).set_xlabel("|residual|")
    fig3.canvas.draw()
    try:
        assert _mag_axes_geometry(fig3, dpi=100) is None
    finally:
        plt.close(fig3)


def test_fit_panels_block_mag_geometry_attrs():
    from ftmwpipeline._internal.report_html_impl import _fit_panels_block

    files = {"re": "w_re.png", "mag": "w_mag.png", "hist": "w_hist.png"}
    geom = {
        "x0": 90.0,
        "x1": 910.0,
        "y0": 40.0,
        "y1": 360.0,
        "w": 1000.0,
        "h": 400.0,
        "flo": 26500.0,
        "fhi": 40000.0,
    }
    # Without a window id / geometry, the mag img is plain (no click surface).
    plain = "\n".join(_fit_panels_block(files))
    assert "cur-plot" not in plain and "data-axes" not in plain

    out = "\n".join(_fit_panels_block(files, window_id=24, mag_geom=geom))
    # The |X| panel is wrapped as the curation click surface with both-way axes.
    assert 'class="cur-plot-wrap" data-window="24"' in out
    assert 'class="cur-plot" data-window="24"' in out
    assert 'data-axes-x0="90.00"' in out and 'data-axes-x1="910.00"' in out
    assert 'data-axes-y0="40.00"' in out and 'data-axes-y1="360.00"' in out
    assert 'data-axes-w="1000.00"' in out and 'data-axes-h="400.00"' in out
    assert 'data-axes-flo="26500.000000"' in out
    assert 'data-axes-fhi="40000.000000"' in out
    # The corner arm toggle and the marker SVG layer (viewBox = natural px).
    assert 'class="cur-plot-arm cur-only"' in out
    assert 'class="cur-plot-svg cur-only" data-window="24"' in out
    assert 'viewBox="0 0 1000 400"' in out
    # The |X| panel alone is the click surface (Re/Im/hist stay plain images).
    assert out.count("cur-plot-wrap") == 1 and "w_re.png" in out


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


def test_lattice_cell_and_table_columns():
    """The clock-lattice annotation: a flagged badge for an on-lattice line,
    rendered as its own column in both the per-window and index tables."""
    from ftmwpipeline._internal.report_html_impl import (
        _index_final_table,
        _lattice_cell,
    )
    from ftmwpipeline.core.data_structures import FinalProducts

    on = _final_peak(30000.0, clock_lattice="320x6 (bb)")
    off = _final_peak(31000.0)  # clock_lattice defaults None
    cell = _lattice_cell(on)
    assert 'class="badge lattice"' in cell
    assert "320x6 (bb)" in cell
    assert "instrumental artifact" in cell  # tooltip framing
    assert _lattice_cell(off) == ""

    # Per-window detail table.
    win = _window_peak_table([on, off], "uV", 1e-6)
    assert "<th>Lattice</th>" in win
    assert 'class="badge lattice"' in win

    # Index line list.
    idx = _index_final_table(FinalProducts(peaks=[on, off]))
    assert "<th>Lattice</th>" in idx
    assert "320x6 (bb)" in idx


def test_summary_distribution_groups_appends_pull():
    from types import SimpleNamespace

    from ftmwpipeline._internal.catalog_xref import CatalogCrossRef
    from ftmwpipeline._internal.report_html_impl import _summary_distribution_groups

    model = SimpleNamespace(
        chi2r_values=[1.0],
        eps_values=[1.0],
        sigma_stat_values=[1.0],
        sigma_eps_values=[1.0],
        sigma_f_values=[1.0],
        snr_values_promoted=[10.0],
    )
    # Without a catalog: SNR / fit-quality / σ_f budget groups, no pull group.
    groups = _summary_distribution_groups(model)
    assert [g[1] for g in groups] == ["snr", "fitquality", "budget"]
    # With a catalog carrying pulls, the pull group is appended.
    xref = CatalogCrossRef(
        catalog_path="c", n_sigma=3.0, n_catalog=2, matches=[], pull_values=[0.5, -0.3]
    )
    groups = _summary_distribution_groups(model, xref)
    assert groups[-1][1] == "pull"
    assert groups[-1][2][0][2] == [0.5, -0.3]


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


@pytest.fixture(autouse=True)
def _serial_report_figures(monkeypatch):
    # The report fixture is tiny (a few windows); the production figure-render
    # fork pool is pure overhead at that size. Default the report tests to serial
    # rendering so the suite does not fork a pool per assemble. The one
    # production-vs-reference test flips back to parallel explicitly to exercise
    # the pool.
    import ftmwpipeline._internal.report_html_impl as rh

    monkeypatch.setattr(rh, "_FIGURE_RENDER_WORKERS", 1, raising=False)


_SharedFullReport = collections.namedtuple(
    "_SharedFullReport", "path out_dir doc source_unchanged window_log_msgs"
)


@pytest.fixture(scope="module")
def full_report_single_file(stage5_small_file, tmp_path_factory):
    """Build the default ``scope="full"`` single-file report ONCE for the module.

    Every report build re-renders the expensive methods-page figures (the
    750k-point FID overview + full-spectrum stage visualizations), so the tests
    that only *inspect* the default full report share this single build instead
    of each paying ~16 s for its own. The single build also captures the inputs
    for two sibling checks: the source ``.ftmw`` hash before/after (read-only
    guarantee) and the per-window render-progress log records.
    """
    import logging

    import ftmwpipeline._internal.report_html_impl as rh

    out = tmp_path_factory.mktemp("full_report_shared")
    src = Path(stage5_small_file)
    before = hashlib.md5(src.read_bytes()).hexdigest()

    records: List = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    cap = _Capture()
    rlogger = logging.getLogger("ftmwpipeline._internal.report_html_impl")
    rlogger.addHandler(cap)
    old_level = rlogger.level
    rlogger.setLevel(logging.INFO)
    prev = rh._FIGURE_RENDER_WORKERS
    rh._FIGURE_RENDER_WORKERS = 1  # serial: one tiny build, no fork overhead
    try:
        path = report_full_impl(str(src), output_dir=str(out), scope="full")
    finally:
        rh._FIGURE_RENDER_WORKERS = prev
        rlogger.removeHandler(cap)
        rlogger.setLevel(old_level)

    after = hashlib.md5(src.read_bytes()).hexdigest()
    win_msgs = [r.getMessage() for r in records if r.getMessage().startswith("window ")]
    p = Path(path)
    return _SharedFullReport(p, out, p.read_text(), before == after, win_msgs)


def _collapse(model, out_root, *, mode="full") -> str:
    """Fold an assembled report model into its single self-contained document."""
    from ftmwpipeline._internal.report_html_impl import _collapse_site_to_single_file

    return _collapse_site_to_single_file(
        Path(out_root),
        mode=mode,
        stem=model.stem,
        figure_store=model.figure_store,
        thumb_store=model.thumb_store,
        page_store=model.page_store,
        css=model.css,
        nav_windows=model.nav_windows,
        tags_by_wid=model.tags_by_wid,
    )


def _feed(*html_strings: str) -> None:
    """Parse each HTML string, asserting it is well-formed enough to tokenize."""
    for doc in html_strings:
        html.parser.HTMLParser().feed(doc)


def _window_pages(model) -> List[str]:
    """The per-window page HTML strings held in the assembled model."""
    return [v for k, v in model.page_store.items() if k.startswith("windows/")]


@pytest.mark.integration
def test_full_site_structure(stage5_small_file, tmp_path):
    # The assembled model (pre-collapse) carries the structural invariants;
    # report_full_impl folds it into one self-contained file. Pages and per-window
    # figures live in the in-memory stores; only the O(1) methods-page figures
    # (overview + histograms) land on disk under figures/.
    out = tmp_path / "site"
    model = _assemble_report_site(str(stage5_small_file), out_root=str(out))

    assert "index.html" in model.page_store
    assert "methods.html" in model.page_store  # methods page built + linked
    assert model.css
    pages = _window_pages(model)
    fig_names = list(model.figure_store)
    disk_figs = [f.name for f in (out / "figures").glob("*.png")]
    assert pages and fig_names
    # Per window: the four zoomed panels (re/im/mag/hist) plus an optional
    # correlation heatmap ("_corr.png") -- all in the figure store. The
    # full-spectrum context is the single shared overview image (on disk), NOT a
    # per-window context PNG. The methods page has one overview
    # ("<stem>_overview.png") and one figure per distribution group
    # ("<stem>_hist_<slug>.png"), both on disk.
    zoom_suffixes = ("_re.png", "_im.png", "_mag.png", "_hist.png")
    corr_pngs = [n for n in fig_names if n.endswith("_corr.png")]
    panel_pngs = [n for n in fig_names if n.endswith(zoom_suffixes)]
    assert len(panel_pngs) == len(pages) * len(zoom_suffixes)
    # The shared overview is the context; no per-window context PNG anywhere.
    assert not any(n.endswith("_ctx.png") for n in fig_names + disk_figs)
    assert len(corr_pngs) <= len(pages)
    # The per-window panels stay off disk; only methods-page figures land there.
    assert not [f for f in disk_figs if "_window_" in f]
    overview_pngs = [
        f for f in disk_figs if f.endswith("_overview.png") and "_window_" not in f
    ]
    hist_pngs = [f for f in disk_figs if "_hist_" in f]
    # The single shared full-spectrum overview image.
    assert len(overview_pngs) == 1
    # Distribution histograms (SNR / fit-quality / σ_f budget; pull only with a
    # catalog), one figure per group that has data.
    assert 1 <= len(hist_pngs) <= 4

    idx = model.page_store["index.html"]
    methods = model.page_store["methods.html"]
    page = pages[0]
    _feed(idx, methods, page, _collapse(model, out))
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
    assert "_overview.png" in model.css
    assert ".spectrum-ctx { background-image:" in model.css
    # The Stage 3 methods section carries both detection views: the active-FT
    # overlay and the primary-pass (Blackman-Harris) detection spectrum.
    assert "_methods_stage3_peaks.png" in methods
    assert "_methods_stage3_primary.png" in methods
    assert "primary-pass" in methods
    # Histograms are interleaved beside their tables (not one trailing figure).
    assert 'class="hist"' in methods
    assert "_hist_" in methods
    # MathJax renders the methods-page equations (raw TeX stays as fallback).
    assert "MathJax" in methods
    assert 'class="equation"' in methods

    # The spectrum context now lives in the window header, hidden by default and
    # revealed by the header bar's "Full spectrum" toggle -- no per-window context
    # image, no bottom <details>.
    assert "<summary>Spectrum context</summary>" not in page
    assert 'class="win-spectrum-ctx" hidden' in page
    assert 'class="win-navbtn spectrum-toggle"' in page
    assert ">Full spectrum<" in page
    assert "_ctx.png" not in page
    assert 'class="spectrum-ctx"' in page and "specnav-rect" in page
    assert "<h2>Fit</h2>" in page
    # The primary fit detail is a responsive grid of the zoomed Re/Im/|X| panels.
    # The residual histogram is in its own collapsed <details>, not in the grid.
    assert 'class="fit-panels"' in page
    assert 'class="panel-grid"' in page
    for panel in ("re", "im", "mag"):
        assert f"_{panel}.png" in page
    assert "_hist.png" in page  # still rendered, just in a <details>
    assert "<summary>Residual histogram</summary>" in page
    assert "Fitted lines" in page
    # Ledger is expanded (bare <h2>), not a <summary>.
    assert "<h2>Ledger candidates</h2>" in page
    assert "<summary>Ledger candidates</summary>" not in page
    # Fit-health sections wrapped in <details> (closed by default, no `open`).
    assert "<summary>Parameter covariance</summary>" in page
    assert "<summary>Fit history</summary>" in page
    assert '<details class="report-detail">' in page
    assert "<details" in page and "open" not in page.split("<details")[1].split(">")[0]
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
    # Full size is the default: <html> on multi-page site pages must NOT carry
    # report-compact.
    html_tag = page[page.index("<html") : page.index(">", page.index("<html")) + 1]
    assert "report-compact" not in html_tag


def test_collapse_stamps_tags_and_builds_filter_menu(tmp_path):
    """The single-file collapse stamps each tagged window section with its
    data-tags set and grows a topnav filter menu of exactly the tags present, in
    canonical order, with the "show all" reset. Driven by a minimal page set so
    the menu/data-attr/JS wiring is exercised without the heavy pipeline fixture
    (the classifier and the chip markup are covered by the pure unit tests)."""
    import re as _re

    from ftmwpipeline._internal.report_html_impl import (
        _FILTER_JS,
        _NAV_JS,
        _STYLESHEET,
        _WINDOW_TAG_ORDER,
        _collapse_site_to_single_file,
    )

    out = tmp_path / "site"
    (out / "figures").mkdir(parents=True)

    def _doc(body: str) -> str:
        return f'<html><body><main class="report">{body}</main></body></html>'

    page_store = {
        "index.html": _doc("<h1>index</h1>"),
        "windows/window_2.html": _doc("<h2>window 2</h2>"),
        "windows/window_5.html": _doc("<h2>window 5</h2>"),
    }
    # One window carries several out-of-canonical-order tags; the other none.
    tags_by_wid = {2: ["high-eps", "attention", "merged"], 5: []}
    nav_windows = [(2, 10.0, 11.0, True), (5, 12.0, 13.0, False)]
    expected = {"attention", "high-eps", "merged"}

    doc = _collapse_site_to_single_file(
        out,
        mode="full",
        stem="x",
        figure_store={},
        thumb_store={},
        page_store=page_store,
        css="",
        nav_windows=nav_windows,
        tags_by_wid=tags_by_wid,
    )
    _feed(doc)

    # Only the tagged window section carries data-tags; tokens are known tags.
    section_tags = _re.findall(r'class="embedded-window" data-tags="([^"]*)"', doc)
    assert len(section_tags) == 1
    assert set(section_tags[0].split()) == expected <= set(_WINDOW_TAG_ORDER)
    # The untagged window section gets no data-tags attribute.
    assert 'id="window-5" class="embedded-window">' in doc

    # The topnav filter menu lists exactly those tags, in canonical order.
    menu_vals = _re.findall(
        r'tagfilter-item"><input type="checkbox" value="([^"]+)"', doc
    )
    assert menu_vals == [t for t in _WINDOW_TAG_ORDER if t in expected]
    assert "tagfilter-clear" in doc  # the "show all" reset

    # The filter + nav-skip behavior is wired (inert without scripting): the
    # filter toggles .tag-hidden, the navigation skips hidden sections, and the
    # stylesheet hides a filtered-out section.
    assert "function apply" in _FILTER_JS and "tag-hidden" in _FILTER_JS
    assert "function vis(s)" in _NAV_JS
    assert "section.embedded-window.tag-hidden { display: none; }" in _STYLESHEET


@pytest.mark.integration
def test_window_page_layout_order_and_details(stage5_small_file, tmp_path):
    """Per-window page: section order, <details> collapse, full-size by default."""
    import re as _re

    out = tmp_path / "site"
    model = _assemble_report_site(str(stage5_small_file), out_root=str(out))
    pages = _window_pages(model)
    assert pages, "need at least one window page"
    page = pages[0]

    # --- section order ---
    # Primary plots (Re/Im/|X|) appear before the fitted-lines table.
    fit_pos = page.index('class="fit-panels"')
    table_pos = page.index('class="peak-list"')
    assert fit_pos < table_pos, "fit panels must precede the peak table"

    # Ledger section (expanded, not in <details>) follows the table.
    ledger_h2_pos = page.index("<h2>Ledger candidates</h2>")
    assert table_pos < ledger_h2_pos, "ledger must follow the peak table"

    # All <details> blocks appear after the ledger.
    details_pos = page.index("<details")
    assert ledger_h2_pos < details_pos, "ledger must precede <details> sections"

    # Attention block (when present) is between ledger and first <details>.
    attn_match = _re.search(r'id="attention"', page)
    if attn_match:
        attn_pos = attn_match.start()
        assert (
            ledger_h2_pos < attn_pos < details_pos
        ), "attention anchor must be after the ledger and before <details>"

    # Spectrum context now lives in the window header (hidden), revealed by the
    # "Full spectrum" toggle -- it precedes the Fit section, not a trailing <details>.
    assert "<summary>Spectrum context</summary>" not in page
    ctx_pos = page.find('class="win-spectrum-ctx"')
    assert ctx_pos != -1, "spectrum context must be in the window header"
    assert ctx_pos < fit_pos, "spectrum context must be in the header, before Fit"

    # --- header band ---
    # χ²ᵣ chip and ε chip (when snr_max > 0) appear in the header band.
    assert 'class="metric-chip"' in page
    assert "&chi;&sup2;<sub>r</sub>" in page
    # The residual histogram is inside a <details>, NOT in the primary panel grid.
    panels_html = page[fit_pos:table_pos]
    assert "_hist.png" not in panels_html, "hist must not be in the primary panel grid"
    # The hist is in its own <details> with the right summary.
    assert "<summary>Residual histogram</summary>" in page

    # --- <details> structure ---
    # Three closed <details class="report-detail"> elements: residual histogram,
    # parameter covariance, fit history. (Spectrum context moved to the header.)
    detail_tags = _re.findall(r"<details([^>]*)>", page)
    assert (
        len(detail_tags) == 3
    ), f"expected exactly 3 <details> elements, got {len(detail_tags)}"
    for attrs in detail_tags:
        assert "report-detail" in attrs
        assert "open" not in attrs.split(), "details must be closed by default"

    # Fit-health headings are <summary> elements.
    assert "<summary>Residual histogram</summary>" in page
    assert "<summary>Parameter covariance</summary>" in page
    assert "<summary>Fit history</summary>" in page
    # Ledger is expanded (bare <h2>), not a <summary>.
    assert "<h2>Ledger candidates</h2>" in page
    assert "<summary>Ledger candidates</summary>" not in page

    # --- full size by default ---
    # The multi-page <html> element must NOT carry the report-compact class.
    html_tag = page[page.index("<html") : page.index(">", page.index("<html")) + 1]
    assert "report-compact" not in html_tag

    # --- full-size lightbox ---
    # The lightbox script is included on the window page and yields to the
    # curation add-marker on an armed plot.
    assert ".fit-panels img" in page  # lightbox click trigger
    assert "cur-plot-wrap.cur-armed" in page  # gate: armed plot -> curation, not zoom


@pytest.mark.integration
def test_ledger_bundle_equals_self_loading(stage5_small_file):
    # 1a: the report's per-window hot path passes the already-loaded
    # ``spectrum_fit``/``sideband`` from the detail bundle to skip two HDF5
    # reloads. That fast path must derive an identical ledger to the standalone
    # self-loading ``(path, window_id)`` call the CLI/api expose.
    from ftmwpipeline._internal.stage5_impl import _resolve_detail_bundle

    path = str(stage5_small_file)
    bundle = _resolve_detail_bundle(path)
    for wf in bundle.fit.window_fits:
        if wf.window_id is None:
            continue
        wid = int(wf.window_id)
        self_loaded = get_candidate_ledger_impl(path, wid)
        bundled = get_candidate_ledger_impl(
            path, wid, spectrum_fit=bundle.fit, sideband=bundle.sideband
        )
        assert self_loaded == bundled


@pytest.mark.integration
@pytest.mark.slow
def test_report_serial_and_parallel_render_match(
    stage5_small_file, tmp_path, monkeypatch
):
    # The production report keeps every per-window figure + page in memory and
    # renders the figures across a forking process pool. A serial render and a
    # parallel render must fold into a byte-identical single-file document --
    # guarding the fork-pool plumbing (worker bytes round-trip) on top of plain
    # matplotlib determinism.
    import ftmwpipeline._internal.report_html_impl as rh

    monkeypatch.setattr(rh, "_FIGURE_RENDER_WORKERS", 1)
    serial_root = tmp_path / "serial"
    m_serial = _assemble_report_site(str(stage5_small_file), out_root=str(serial_root))
    html_serial = _collapse(m_serial, serial_root)

    monkeypatch.setattr(rh, "_FIGURE_RENDER_WORKERS", 4)
    par_root = tmp_path / "parallel"
    m_par = _assemble_report_site(str(stage5_small_file), out_root=str(par_root))
    html_par = _collapse(m_par, par_root)

    assert html_serial == html_par
    # The O(N) per-window figures + pages stay in the model, off disk (only the
    # O(1) methods/overview figures land under figures/).
    assert m_par.figure_store and m_par.page_store and m_par.thumb_store
    assert not (par_root / "windows").exists() and not (par_root / "assets").exists()
    # Per-window panels (``<stem>_window_NNN_<panel>.png``) are never written;
    # only methods-page figures (e.g. ``<stem>_methods_stage4_windows.png``) land.
    assert list((par_root / "figures").glob("*_window_*.png")) == []


@pytest.mark.integration
@pytest.mark.slow
def test_full_windows_filter_attention_subset(stage5_small_file, tmp_path):
    m_all = _assemble_report_site(
        str(stage5_small_file), out_root=str(tmp_path / "all"), windows="all"
    )
    m_att = _assemble_report_site(
        str(stage5_small_file), out_root=str(tmp_path / "att"), windows="attention"
    )

    n_all = len(_window_pages(m_all))
    n_att = len(_window_pages(m_att))
    # The attention filter never produces more pages than 'all'.
    assert 0 <= n_att <= n_all
    # The index still lists every window in both modes.
    assert "<code>attention</code> filter" in m_att.page_store["index.html"]


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
    model = _assemble_report_site(
        str(stage5_small_file), out_root=str(out), catalog=str(cat)
    )

    idx = model.page_store["index.html"]
    assert "<th>Catalog</th>" in idx
    assert "cat-0" in idx
    assert "proximity only" in idx
    methods = model.page_store["methods.html"]
    assert "Catalog cross-reference" in methods
    # The catalog pull histogram is rendered as its own group figure (on disk).
    figures = [f.name for f in (out / "figures").glob("*.png")]
    assert any(f.endswith("_hist_pull.png") for f in figures)
    pages = _window_pages(model)
    assert any("<th>Catalog</th>" in p for p in pages)
    _feed(idx, methods, *pages, _collapse(model, out))


@pytest.mark.integration
def test_full_is_read_only(full_report_single_file):
    # The shared fixture hashed the source .ftmw before/after its one build:
    # report_full_impl must not mutate the input file.
    assert full_report_single_file.source_unchanged


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
def test_single_file_full_folds_in_window_pages(full_report_single_file):
    p, out = full_report_single_file.path, full_report_single_file.out_dir
    assert p.name.endswith("_report.html") and not p.name.endswith("_summary.html")
    assert list(out.glob("*.html")) == [p]
    assert not (out / "windows").exists()

    doc = full_report_single_file.doc
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
def test_single_file_carries_curation_surface(full_report_single_file):
    doc = full_report_single_file.doc
    html.parser.HTMLParser().feed(doc)

    # The boot script + curation script + topnav chrome ship in the single file.
    assert "window.__stem=" in doc
    assert "curation-enabled" in doc  # the CSS gates on it; the toggle adds it
    assert 'class="cur-toggle"' in doc and 'class="cur-badge"' in doc
    # Per-row controls (Remove only -- split/merge are inferred from add) +
    # raw-frequency data attributes on fitted-line rows.
    assert 'data-act="remove"' in doc
    assert 'data-act="split"' not in doc and 'data-act="merge-selected"' not in doc
    import re as _re

    assert _re.search(r'data-window="\d+" data-freq="[0-9.]+"', doc)
    # Per-window controls (add / mark reviewed) gated by cur-only.
    assert 'class="cur-only cur-window-controls"' in doc
    assert 'data-act="add-typed"' in doc
    # The curation CSS gating rule is present (single-class governance).
    assert "html:not(.curation-enabled)" in doc
    # Click-on-plot: the |X| panel carries its data-axes geometry so the overlay
    # can invert a click to a molecular MHz. The geometry survives the single-file
    # collapse (only the img src is rewritten to a data URI).
    assert 'class="cur-plot"' in doc
    assert "data-axes-x0=" in doc and "data-axes-flo=" in doc
    geom = _re.search(
        r'data-axes-x0="([0-9.]+)" data-axes-x1="([0-9.]+)"',
        doc,
    )
    assert geom and float(geom.group(2)) > float(geom.group(1))  # x1 > x0
    # The arm toggle (touch-safe opt-in) and the marker SVG layer ship per panel.
    assert 'class="cur-plot-arm cur-only"' in doc
    assert "cur-plot-svg cur-only" in doc and "preserveAspectRatio" in doc
    # Curation convenience controls ship in the single file: reviewed-and-advance,
    # per-window clear, the index-list per-row accept, and the cart jump + keyboard
    # handlers in the script.
    assert 'data-act="accept-next"' in doc and 'data-act="clear-window"' in doc
    assert "cur-list-accept" in doc
    assert "function jumpToEdit" in doc and "function clearWindow" in doc
    assert "__navNext" in doc
    # Window nav + curate verbs live in the title bar; topnav steps plain windows
    # (no flagged-step buttons), relying on the tag filter for window types.
    assert 'class="win-header-bar"' in doc and 'class="win-navbtn"' in doc
    assert 'class="nav-step nav-step-prev"' in doc
    assert "nav-flag" not in doc and "&#9873;" not in doc
    # Plot keyboard shortcuts act on the magnitude plot under the pointer (not
    # table rows), mirroring click-to-add, and the cart shows a legend for them.
    assert "function nearestPeakRow" in doc and "function plotMhz" in doc
    assert 'class="cur-keys"' in doc
    # A queued "mark reviewed" drops the attention tint from the overview rect,
    # and syncs every accept button for that window (index list + window page).
    assert "function refreshReviewed" in doc
    assert ".specnav-rect.attn.cur-reviewed" in doc
    assert "querySelectorAll('[data-act=\"accept\"]')" in doc
    assert ".cur-btn.cur-queued" in doc
    # The overview attention tint is the SVG overlay (toggleable), not baked into
    # the shared overview PNG -- so reviewing can clear it client-side.
    assert ".specnav-rect.attn { fill:" in doc
    # Regression guard: the hover-preview popup must target the overview rects by
    # class, not the now-overloaded [data-window] (curation hangs data-window on
    # plot wraps/buttons/rows, and an attention plot wrap holds a descendant
    # <title> that broke the bind loop and killed the table-row previews).
    assert "querySelectorAll('.specnav-rect')" in doc
    assert "titleEl.parentNode === el" in doc


def test_curation_cart_csv_self_declares_frame(full_report_single_file):
    """D0 regression: the cart's ``toCsv()`` export must self-declare its
    frame. The cart deliberately emits raw Stage 5 model frequencies (see
    "Curating in the browser" in ``fit_curation.rst``), and since ``faf8e6e``
    omitting the frame on a frequency-bearing curation call is a hard
    ``ValueError`` on a ``self_calibrated`` file -- so without this header the
    cart's own export could not be applied to such a file at all.

    Still required now that a ``remove`` exports ``uid:N``: an ``add`` row
    carries a frequency, so the header is not vestigial.
    """
    import re as _re

    doc = full_report_single_file.doc
    m = _re.search(r"function toCsv\(\) \{(.*?)\n  \}", doc, _re.S)
    assert m, "toCsv() not found in the emitted report script"
    body = m.group(1)
    # The frame directive must be the CSV's first emitted line, ahead of the
    # action,window,freqs,params header row.
    assert _re.search(r"\[\s*'# frame: raw'\s*,\s*'action,window,freqs,params'", body)


def test_curation_cart_exports_uid_for_remove(full_report_single_file):
    """A remove exports the peak's identifier, not its frequency, when the fit
    carries one.

    ``uid:N`` addresses the peak exactly, so a neighbor inside the snap
    tolerance cannot be matched instead and the frame is irrelevant to it --
    which is the whole reason ``review edit --remove`` accepts the token. The
    fallback (no ``data-uid``, i.e. a fit predating ``peak_uid``) must stay a
    plain frequency, and the ``# frame: raw`` header must stay unconditional
    because ``add`` rows still carry frequencies.
    """
    import re as _re

    doc = full_report_single_file.doc
    m = _re.search(r"function csvFreqs\(o\) \{(.*?)\n  \}", doc, _re.S)
    assert m, "csvFreqs() not found in the emitted report script"
    body = m.group(1)
    assert "o.action === 'remove' && o.uid" in body
    assert "'uid:' + o.uid" in body
    assert "return csvCell(o.freqs);" in body  # the no-uid fallback

    # The remove control reads the attribute the table now emits, and keeps the
    # frequency in freqs so the display wiring is unchanged.
    assert "var uid = row.getAttribute('data-uid');" in doc
    assert "action: 'remove', window: win, freqs: freq, uid: uid," in doc


def _peak_list_row(doc: str):
    """Return ``(window_id, raw_freq_str)`` of one fitted-line row in the report."""
    import re as _re

    for tbl in _re.findall(r'<table class="peak-list">.*?</table>', doc, _re.S):
        m = _re.search(r'data-window="(\d+)" data-freq="([0-9.]+)"', tbl)
        if m:
            return int(m.group(1)), m.group(2)
    pytest.skip("No fitted-line rows in the built subset")


@pytest.mark.integration
def test_curation_emitted_frequency_resolves(
    stage5_small_file, full_report_single_file, tmp_path
):
    """The load-bearing rule: the raw frequency a control emits targets the
    intended peak when fed straight to ``review apply``."""
    from ftmwpipeline._internal.stage6_impl import apply_curation_impl
    from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5

    doc = full_report_single_file.doc
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
def test_curation_emitted_uid_resolves(
    stage5_small_file, full_report_single_file, tmp_path
):
    """The uid a control emits removes the intended peak when fed straight to
    ``review apply`` -- the uid half of the rule above.

    This is what makes the report's Remove button meaningful: without it the
    cart could export a ``uid:N`` the pipeline cannot resolve, and the failure
    would surface only at apply time on the user's file.
    """
    import re as _re

    from ftmwpipeline._internal.stage6_impl import apply_curation_impl
    from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5

    doc = full_report_single_file.doc
    m = None
    for tbl in _re.findall(r'<table class="peak-list">.*?</table>', doc, _re.S):
        m = _re.search(
            r'data-window="(\d+)" data-freq="([0-9.]+)" data-uid="(\d+)"', tbl
        )
        if m:
            break
    # A hard failure, not a skip: every fitted peak in a current build carries a
    # peak_uid, so an absent attribute means the report stopped emitting it.
    assert m, "no fitted-line row in the report carries data-uid"
    wid, freq, uid = int(m.group(1)), m.group(2), int(m.group(3))

    def _window_peaks(fp):
        with h5py.File(str(fp), "r") as h5f:
            sf = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
        for wf in sf.window_fits:
            if int(wf.window_id) == wid:
                return [
                    (p.peak_uid, round(float(p.frequency_mhz), 6))
                    for p in wf.fitted_peaks
                ]
        return []

    target = tmp_path / "uid.ftmw"
    shutil.copy(stage5_small_file, target)
    before = _window_peaks(target)
    # The emitted attribute pair names one real peak, consistently.
    assert (uid, round(float(freq), 6)) in before

    cur = tmp_path / "remove_uid.csv"
    cur.write_text(
        f"# frame: raw\naction,window,freqs,params\nremove,{wid},uid:{uid},\n"
    )
    apply_curation_impl(str(target), str(cur))

    after = _window_peaks(target)
    assert uid not in [u for u, _ in after]
    assert len(after) == len(before) - 1


@pytest.mark.integration
def test_report_rejects_unknown_scope(stage5_small_file, tmp_path):
    with pytest.raises(ValueError, match="scope"):
        report_full_impl(
            str(stage5_small_file), output_dir=str(tmp_path / "x"), scope="bogus"
        )


# Cross-interface report byte-identity (impl == api == pipe) is covered by
# test_run_cross_interface below, which exercises the default report_run path;
# report_run builds the same full single-file HTML internally, so a separate
# report_full cross-interface build is redundant.


# ---------------------------------------------------------------------------
# report run (the default Stage 6 deliverable: L1 table + L3 report)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_full_emits_per_window_progress_logs(full_report_single_file):
    """The render loop logs one ``window i/N`` per window for the progress bridge
    (captured during the shared build's serial render)."""
    win_msgs = full_report_single_file.window_log_msgs
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
def test_run_defaults_output_dir_to_cwd(stage5_small_file, tmp_path, monkeypatch):
    from ftmwpipeline._internal.report_html_impl import report_run_impl

    monkeypatch.chdir(tmp_path)
    result = report_run_impl(str(stage5_small_file))

    assert Path(result["table"]).resolve().parent == tmp_path.resolve()
    assert Path(result["html"]).resolve().parent == tmp_path.resolve()


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
