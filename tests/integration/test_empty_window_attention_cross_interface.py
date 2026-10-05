"""``empty_window_residual`` is the same on API, Pipeline and CLI, and in the report.

Mandatory cross-interface category (``dev-docs/TESTING_STRATEGY.md``) for the
review attention item of ``dev-docs/CONTRACT_STRATEGY.md`` §Review attention, on
real 2638 data. The small Stage 5 baseline fits its two low-band windows to
empty; the fixture records the structural-replan flag Stage 5 writes when such
a window's edge stays coherent (as it does for windows 100, 158 and 227 of the
full 2638 fit), so the trigger runs on a real fit without a full-band build.
"""

from __future__ import annotations

import json
import shutil

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import Pipeline
from ftmwpipeline.cli.main import main
from ftmwpipeline.serialize import to_jsonable

pytestmark = [pytest.mark.integration]

KIND = "empty_window_residual"
S_COH = 20.0


def _flag_an_empty_window(path) -> int:
    """Record a not-merged replan flag on the low edge of the first window the
    fit holds no line in; return that window's id."""
    plan = ftmw.load_windows(str(path))
    fit = ftmw.load_fit(str(path))
    live = {int(w.window_id) for w in fit.window_fits if w.fitted_peaks}
    empty = sorted(int(w.window_id) for w in plan.windows if w.window_id not in live)
    assert empty, "the small baseline is expected to fit a window to empty"
    wid = empty[0]
    with h5py.File(path, "a") as h5f:
        grp = h5f["stage5_fitting"]
        records = json.loads(grp.attrs.get("replan_history", "[]"))
        records.append(
            {
                "triggering_window_id": wid,
                "partner_window_id": wid + 1,
                "surviving_window_id": wid,
                "edge_side": "low",
                "edge_coherence_before": S_COH,
                "revision_before": 0,
                "revision_after": 0,
                "accepted": False,
                "reason": "not merged: the window's fit holds no line",
            }
        )
        grp.attrs["replan_history"] = json.dumps(records)
    return wid


def _cli_json(argv, capsys):
    rc = main([*argv, "--json"])
    cap = capsys.readouterr()
    assert rc == 0, cap.err
    return json.loads(cap.out)


@pytest.fixture(scope="module")
def flagged_source(baseline_2638_stage5_small, tmp_path_factory):
    """Read-only: the small Stage 5 baseline with one empty window flagged."""
    path = tmp_path_factory.mktemp("empty_window_attn") / "flagged.ftmw"
    shutil.copy(baseline_2638_stage5_small, path)
    wid = _flag_an_empty_window(path)
    return path, wid


@pytest.fixture
def three_copies(flagged_source, tmp_path):
    src, wid = flagged_source
    paths = []
    for name in ("api", "pipeline", "cli"):
        p = tmp_path / f"{name}.ftmw"
        shutil.copy(src, p)
        paths.append(p)
    return paths, wid


def test_review_run_flags_the_window_on_every_interface(three_copies, capsys):
    (p_api, p_pipe, p_cli), wid = three_copies
    r_api = ftmw.review_run(str(p_api))
    r_pipe = Pipeline.open(str(p_pipe)).review_run()
    env = _cli_json(["review", "run", str(p_cli)], capsys)

    assert r_api.reason_counts.get(KIND) == 1
    assert r_api.reason_counts == r_pipe.reason_counts
    assert env["summary"]["reason_counts"] == r_api.reason_counts
    assert r_api.n_windows == r_pipe.n_windows == env["summary"]["n_windows"]
    assert r_api.n_attention == env["summary"]["n_attention"]

    statuses = [
        ftmw.get_review_status(str(p_api)).window_statuses,
        Pipeline.open(str(p_pipe)).review_status().window_statuses,
        ftmw.get_review_status(str(p_cli)).window_statuses,
    ]
    for st in statuses:
        assert st[wid] == statuses[0][wid]
    status = statuses[0][wid]
    assert status.provenance == "auto" and status.needs_attention
    (reason,) = [r for r in status.attention_reasons if r.kind == KIND]
    assert reason.evidence["edges"] == [{"side": "low", "s_coh": S_COH}]
    assert reason.severity == pytest.approx(
        S_COH / reason.evidence["residual_edge_threshold"]
    )
    assert reason.locations == [
        c["frequency_mhz"] for c in reason.evidence["candidates"]
    ]

    # The window the item names is a window of the fit's plan, not live.
    rows = {r.window_id: r for r in ftmw.window_status(str(p_api))["windows"]}
    assert rows[wid].live is False


def test_review_show_reports_the_same_reason(three_copies, capsys):
    (p_api, _p_pipe, _p_cli), wid = three_copies
    ftmw.review_run(str(p_api))
    (reason,) = [
        r
        for r in ftmw.get_review_status(str(p_api))
        .window_statuses[wid]
        .attention_reasons
        if r.kind == KIND
    ]
    wire = to_jsonable(
        {
            "kind": reason.kind,
            "severity": reason.severity,
            "detail": reason.detail,
            "locations": reason.locations,
            "evidence": reason.evidence,
        }
    )

    queue = _cli_json(["review", "show", str(p_api), "--attention"], capsys)
    (row,) = [r for r in queue["attention"] if r["window_id"] == wid]
    assert row["kind"] == KIND
    assert wire in row["reasons"]

    detail = _cli_json(["review", "show", str(p_api), "--window", str(wid)], capsys)
    assert detail["fitted_peaks"] == [] and detail["candidates"] == []
    assert detail["reduced_chi2"] is None
    assert detail["reduced_chi2_absent"] == "undefined"
    assert wire in detail["attention_reasons"]


def test_accept_marks_it_reviewed_on_every_interface(three_copies, capsys):
    (p_api, p_pipe, p_cli), wid = three_copies
    for p in (p_api, p_pipe, p_cli):
        ftmw.review_run(str(p))
    ftmw.review_accept(str(p_api), wid)
    Pipeline.open(str(p_pipe)).review_accept(wid)
    _cli_json(["review", "accept", str(p_cli), "--window", str(wid)], capsys)

    got = [ftmw.get_review_status(str(p)).window_statuses[wid] for p in three_copies[0]]
    assert got[0] == got[1] == got[2]
    assert got[0].provenance == "reviewed"
    assert [r.kind for r in got[0].attention_reasons] == [KIND]
    logs = [
        [(e.window_id, e.kind) for e in ftmw.review_log(str(p))]
        for p in three_copies[0]
    ]
    assert logs[0] == logs[1] == logs[2] == [(wid, "accept")]


def test_a_window_created_over_it_resolves_the_item(flagged_source, tmp_path):
    src, wid = flagged_source
    path = tmp_path / "create.ftmw"
    shutil.copy(src, path)
    ftmw.review_run(str(path))
    st = ftmw.get_review_status(str(path)).window_statuses[wid]
    anchor = next(r for r in st.attention_reasons if r.kind == KIND).locations[0]

    ftmw.review_create(str(path), anchor, frame="raw")
    assert wid not in ftmw.get_review_status(str(path)).window_statuses
    # And a fresh review run agrees: the created window took it over.
    ftmw.review_run(str(path))
    assert wid not in ftmw.get_review_status(str(path)).window_statuses


def test_fitted_numbers_do_not_move(flagged_source, tmp_path):
    """Attention is advice: the item never changes a fitted number."""
    src, _wid = flagged_source
    path = tmp_path / "numbers.ftmw"
    shutil.copy(src, path)
    before = ftmw.read_table(str(path), "fit_peaks")
    ftmw.review_run(str(path))
    after = ftmw.read_table(str(path), "fit_peaks")
    assert set(before) == set(after)
    for col in before:
        assert list(map(repr, before[col])) == list(map(repr, after[col])), col


def test_the_report_renders_the_empty_window(flagged_source, tmp_path):
    src, wid = flagged_source
    path = tmp_path / "report.ftmw"
    shutil.copy(src, path)
    ftmw.review_run(str(path))
    out = ftmw.report_run(
        str(path), output_dir=tmp_path / "rep", windows="attention", emit_table=False
    )
    html = open(out["html"], encoding="utf-8").read()
    assert f'id="window-{wid}"' in html
    section = html[html.index(f'id="window-{wid}"') :]
    section = section[: section.index("</section>")]
    assert "no fitted line" in section
    assert "attn-empty_window_residual" in section  # the candidate marker
    assert "Stage 3 peaks in this window" in section
    assert 'data-act="add-typed"' not in section  # no add on a window not fitted
