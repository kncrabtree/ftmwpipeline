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
import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.report_html_impl import (
    _PANEL_ORDER,
    _covariance_block,
    _esc,
    _page,
    _panel_figure_name,
    _table,
    _window_page_name,
    report_full_impl,
)
from ftmwpipeline._internal.stage4_impl import load_windows_impl, save_window_plan_impl
from ftmwpipeline._internal.stage6_impl import review_run_impl
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


def test_covariance_block_renders_small_matrix():
    cov = np.array([[1.0, 0.5], [0.5, 2.0]])
    block = "\n".join(_covariance_block(cov, ["amplitude_0", "offset_0"]))
    assert "amplitude_0" in block
    assert "offset_0" in block
    assert "<table" in block


def test_covariance_block_summarizes_when_too_wide():
    n = 30
    cov = np.eye(n)
    labels = [f"p{i}" for i in range(n)]
    block = "\n".join(_covariance_block(cov, labels))
    assert "too" in block and "wide" in block
    assert "<table" not in block


def test_covariance_block_handles_missing():
    block = "\n".join(_covariance_block(None, None))
    assert "No parameter covariance" in block


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
    out = tmp_path / "site"
    index = report_full_impl(str(stage5_small_file), output_dir=str(out))

    assert index == str(out / "index.html")
    assert (out / "index.html").exists()
    assert (out / "assets" / "style.css").exists()
    pages = list((out / "windows").glob("*.html"))
    figures = list((out / "figures").glob("*.png"))
    assert pages and figures
    # One detail page per window; one PNG per panel (overview/re/im/mag/hist).
    assert len(figures) == len(pages) * len(_PANEL_ORDER)
    _assert_wellformed(out)

    idx = (out / "index.html").read_text()
    assert "FTMW pipeline report" in idx
    assert "Final line list" in idx
    assert 'href="windows/window_' in idx  # links to the window pages

    page = pages[0].read_text()
    assert "<h2>Fit</h2>" in page
    # The fit detail is a flexbox of separate panel images, not one figure.
    assert 'class="fit-panels"' in page
    assert 'class="panel-row"' in page
    assert page.count("<img") >= len(_PANEL_ORDER)
    for panel in _PANEL_ORDER:
        assert f"_{panel}.png" in page
    assert "Fitted lines" in page
    assert "Parameter covariance" in page
    assert "Ledger candidates" in page
    assert "<h2>Fit log</h2>" in page


@pytest.mark.integration
def test_full_windows_filter_attention_subset(stage5_small_file, tmp_path):
    all_out = tmp_path / "all"
    att_out = tmp_path / "att"
    report_full_impl(str(stage5_small_file), output_dir=str(all_out), windows="all")
    report_full_impl(
        str(stage5_small_file), output_dir=str(att_out), windows="attention"
    )

    n_all = len(list((all_out / "windows").glob("*.html")))
    n_att = len(list((att_out / "windows").glob("*.html")))
    # The attention filter never produces more pages than 'all'.
    assert 0 <= n_att <= n_all
    # The index still lists every window in both modes.
    assert "<code>attention</code> filter" in (att_out / "index.html").read_text()


@pytest.mark.integration
def test_full_is_read_only(stage5_small_file, tmp_path):
    fp = tmp_path / "ro.ftmw"
    shutil.copy(stage5_small_file, fp)
    before = hashlib.md5(fp.read_bytes()).hexdigest()
    report_full_impl(str(fp), output_dir=str(tmp_path / "out"))
    assert hashlib.md5(fp.read_bytes()).hexdigest() == before


@pytest.mark.integration
def test_full_cross_interface(stage5_small_file, tmp_path):
    fp = tmp_path / "copy.ftmw"
    shutil.copy(stage5_small_file, fp)

    impl_dir = tmp_path / "impl"
    api_dir = tmp_path / "api"
    pipe_dir = tmp_path / "pipe"

    from ftmwpipeline._internal.report_html_impl import report_full_impl as impl

    impl(str(fp), output_dir=str(impl_dir))
    ftmw.report_full(str(fp), output_dir=str(api_dir))
    Pipeline.open(fp).report_full(output_dir=str(pipe_dir))

    # The rendered HTML is identical across interfaces (paths inside are
    # relative, so only the output directory differs).
    a = (impl_dir / "index.html").read_text()
    b = (api_dir / "index.html").read_text()
    c = (pipe_dir / "index.html").read_text()
    assert a == b == c
