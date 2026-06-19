"""
Tests for the Stage 6 curation language: parsing, coalescing, and ``review
apply`` / ``review log``.

The parsing / coalescing / ambiguity logic is pure and tested without a built
file. ``review apply`` end-to-end and its cross-interface parity use a built
3-window 2638 Stage-5 subset (mirrors test_review_decisions.py).
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal import stage6_impl as s6
from ftmwpipeline._internal.stage4_impl import load_windows_impl, save_window_plan_impl
from ftmwpipeline._internal.stage6_impl import (
    STAGE5_BASELINE_GROUP,
    PlannedAction,
    apply_curation_impl,
    clear_stage5_baseline,
    describe_planned_action,
    parse_curation_file,
    refit_window_impl,
    review_log_impl,
    review_undo_impl,
    _decision_to_op,
    _resolve_curation_plan,
)
from ftmwpipeline.cli.review_commands import (
    cmd_review_apply,
    cmd_review_log,
    cmd_review_undo,
)
from ftmwpipeline.core.data_structures import DecisionLogEntry
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5
from ftmwpipeline.io.stage6_review_serialization import load_stage6_review_from_file
from ftmwpipeline.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Parsing (pure, no fixture)
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, text: str) -> str:
    p = tmp_path / "curation.csv"
    p.write_text(text)
    return str(p)


def test_parse_basic_rows(tmp_path):
    ops = parse_curation_file(
        _write(
            tmp_path,
            "remove,217,38450.123,\n"
            "add,217,38451.0,\n"
            "merge,438,38450.10;38450.40,\n"
            "split,24,38449.9,into=3\n"
            "accept,84,,\n"
            "accept,309,,candidate=38502.7\n",
        )
    )
    assert [o.action for o in ops] == [
        "remove",
        "add",
        "merge",
        "split",
        "accept",
        "accept",
    ]
    assert ops[2].freqs == [38450.10, 38450.40]
    assert ops[3].params == {"into": "3"}
    assert ops[5].params == {"candidate": "38502.7"}


def test_parse_skips_comments_blanks_and_header(tmp_path):
    ops = parse_curation_file(
        _write(
            tmp_path,
            "# a comment\n"
            "action,window,freqs,params\n"
            "\n"
            "   \n"
            "# another comment\n"
            "add,5,100.0,\n",
        )
    )
    assert len(ops) == 1
    assert ops[0].action == "add" and ops[0].window_id == 5


def test_parse_unknown_action_reports_line(tmp_path):
    with pytest.raises(ValueError, match="line 2: unknown action"):
        parse_curation_file(_write(tmp_path, "add,5,100.0,\nbogus,6,1.0,\n"))


def test_parse_non_integer_window_reports_line(tmp_path):
    with pytest.raises(ValueError, match="line 1: window id"):
        parse_curation_file(_write(tmp_path, "add,five,100.0,\n"))


def test_parse_non_numeric_frequency_reports_line(tmp_path):
    with pytest.raises(ValueError, match="line 1: non-numeric frequency"):
        parse_curation_file(_write(tmp_path, "add,5,abc,\n"))


def test_parse_add_requires_one_frequency(tmp_path):
    with pytest.raises(ValueError, match="add needs exactly one frequency"):
        parse_curation_file(_write(tmp_path, "add,5,100.0;101.0,\n"))


def test_parse_merge_requires_two(tmp_path):
    with pytest.raises(ValueError, match="merge needs at least two"):
        parse_curation_file(_write(tmp_path, "merge,5,100.0,\n"))


def test_parse_split_bad_into(tmp_path):
    with pytest.raises(ValueError, match="into must be >= 2"):
        parse_curation_file(_write(tmp_path, "split,5,100.0,into=1\n"))
    with pytest.raises(ValueError, match="into=.*is not an integer"):
        parse_curation_file(_write(tmp_path, "split,5,100.0,into=x\n"))


def test_parse_accept_rejects_frequency_column(tmp_path):
    with pytest.raises(ValueError, match="accept takes no frequency column"):
        parse_curation_file(_write(tmp_path, "accept,5,100.0,\n"))


def test_parse_malformed_param(tmp_path):
    with pytest.raises(ValueError, match="malformed parameter"):
        parse_curation_file(_write(tmp_path, "split,5,100.0,into\n"))


# ---------------------------------------------------------------------------
# Coalescing (pure)
# ---------------------------------------------------------------------------


def test_resolve_coalesces_consecutive_same_window(tmp_path):
    ops = parse_curation_file(
        _write(
            tmp_path,
            "add,5,100.0,\nremove,5,101.0,\nadd,5,102.0,\n",
        )
    )
    plan = _resolve_curation_plan(ops)
    assert len(plan) == 1
    a = plan[0]
    assert a.kind == "edit" and a.window_id == 5
    assert a.add == [100.0, 102.0] and a.remove == [101.0]


def test_resolve_same_window_barrier_flushes(tmp_path):
    # split on the SAME window splits the edit into before/after groups.
    ops = parse_curation_file(
        _write(
            tmp_path,
            "add,5,100.0,\nsplit,5,200.0,into=2\nadd,5,300.0,\n",
        )
    )
    plan = _resolve_curation_plan(ops)
    assert [a.kind for a in plan] == ["edit", "split", "edit"]
    assert plan[0].add == [100.0]
    assert plan[1].peak == 200.0 and plan[1].into == 2
    assert plan[2].add == [300.0]


def test_resolve_other_window_barrier_does_not_flush(tmp_path):
    # A merge on window 8 does not flush window 5's pending edit; window 5's
    # two adds stay coalesced (windows are independent).
    ops = parse_curation_file(
        _write(
            tmp_path,
            "add,5,100.0,\nmerge,8,1.0;2.0,\nadd,5,300.0,\n",
        )
    )
    plan = _resolve_curation_plan(ops)
    kinds = [a.kind for a in plan]
    assert kinds.count("edit") == 1 and "merge" in kinds
    edit = next(a for a in plan if a.kind == "edit")
    assert edit.window_id == 5 and edit.add == [100.0, 300.0]


def test_resolve_accept_candidate_and_bare(tmp_path):
    ops = parse_curation_file(
        _write(tmp_path, "accept,5,,\naccept,6,,candidate=42.5\n")
    )
    plan = _resolve_curation_plan(ops)
    assert plan[0].kind == "accept" and plan[0].candidate is None
    assert plan[1].kind == "accept" and plan[1].candidate == 42.5


def test_describe_planned_action_strings():
    assert "edit window 5" in describe_planned_action(
        PlannedAction(kind="edit", window_id=5, add=[1.0], remove=[2.0])
    )
    assert "merge window 8" in describe_planned_action(
        PlannedAction(kind="merge", window_id=8, peaks=[1.0, 2.0])
    )
    assert "into 3" in describe_planned_action(
        PlannedAction(kind="split", window_id=2, peak=1.0, into=3)
    )
    assert "candidate" in describe_planned_action(
        PlannedAction(kind="accept", window_id=1, candidate=9.0)
    )


# ---------------------------------------------------------------------------
# Ambiguity warnings (pure, monkeypatched fitted peaks)
# ---------------------------------------------------------------------------


def test_ambiguity_warnings(monkeypatch):
    monkeypatch.setattr(
        s6,
        "_fitted_freqs_by_window",
        lambda path: {5: [100.0, 100.02, 200.0]},
    )
    # Two peaks within 50 kHz of 100.01 -> ambiguous.
    warns = s6._curation_ambiguity_warnings(
        "x", [PlannedAction(kind="edit", window_id=5, remove=[100.01])]
    )
    assert any("within" in w and "2 fitted peaks" in w for w in warns)

    # No peak near 150.0 -> unmatched (will fail).
    warns = s6._curation_ambiguity_warnings(
        "x", [PlannedAction(kind="edit", window_id=5, remove=[150.0])]
    )
    assert any("no fitted peak within" in w for w in warns)

    # Unknown window -> no fitted peaks.
    warns = s6._curation_ambiguity_warnings(
        "x", [PlannedAction(kind="split", window_id=9, peak=1.0)]
    )
    assert any("no fitted peaks" in w for w in warns)

    # add frequencies are not checked (they create peaks).
    warns = s6._curation_ambiguity_warnings(
        "x", [PlannedAction(kind="edit", window_id=5, add=[150.0])]
    )
    assert warns == []


# ---------------------------------------------------------------------------
# Integration: review apply / log on a built Stage-5 file
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def exp_2638_data_path():
    data_path = Path("examples/blackchirp_data/2638")
    if not data_path.exists():
        pytest.skip("Experiment 2638 data not available")
    return str(data_path)


@pytest.fixture(scope="module")
def stage5_file(exp_2638_data_path, tmp_path_factory):
    """Build the 2638 pipeline through Stage 5 (3-window subset) once."""
    tmp = tmp_path_factory.mktemp("stage5_curation")
    fp = tmp / "2638_curation.ftmw"

    ftmw.import_data(fp, source=exp_2638_data_path)
    ftmw.compute_ft(fp, trim=(26500, 40000))
    ftmw.estimate_noise(fp)
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
    return fp


def _fitted_by_window(path: Path) -> Dict[int, List[float]]:
    with h5py.File(str(path), "r") as h5f:
        sf = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    return {
        int(wf.window_id): sorted(round(float(p.frequency_mhz), 6) for p in wf.fitted_peaks)
        for wf in sf.window_fits
        if wf.window_id is not None
    }


def _a_peak(path: Path) -> tuple[int, float]:
    """Return (window_id, frequency) of one fitted peak."""
    by_w = _fitted_by_window(path)
    for wid, freqs in by_w.items():
        if freqs:
            return wid, freqs[0]
    pytest.skip("No fitted peaks in the built subset")


@pytest.mark.integration
def test_apply_dry_run_does_not_mutate(stage5_file, tmp_path):
    fp = tmp_path / "dry.ftmw"
    shutil.copy(stage5_file, fp)
    wid, freq = _a_peak(fp)
    cur = tmp_path / "c.csv"
    cur.write_text(f"remove,{wid},{freq},\n")

    before = hashlib.md5(fp.read_bytes()).hexdigest()
    result = apply_curation_impl(fp, cur, dry_run=True)
    assert hashlib.md5(fp.read_bytes()).hexdigest() == before
    assert result.dry_run and result.applied == 0
    assert len(result.plan) == 1 and result.plan[0].kind == "edit"


@pytest.mark.integration
def test_apply_matches_handtyped_sequence(stage5_file, tmp_path):
    a = tmp_path / "a.ftmw"
    b = tmp_path / "b.ftmw"
    shutil.copy(stage5_file, a)
    shutil.copy(stage5_file, b)

    wid, freq = _a_peak(a)
    # Two add rows on one window coalesce into one refit; assert apply matches a
    # single hand-typed refit with the union of edits.
    f1, f2 = freq + 0.3, freq + 0.6
    cur = tmp_path / "c.csv"
    cur.write_text(f"add,{wid},{f1},\nadd,{wid},{f2},\n")

    apply_curation_impl(a, cur)
    refit_window_impl(str(b), wid, add=[f1, f2])

    assert _fitted_by_window(a) == _fitted_by_window(b)
    # The decision log records both adds.
    log = review_log_impl(a)
    kinds = sorted(e.kind for e in log if e.window_id == wid)
    assert kinds == ["add", "add"]


@pytest.mark.integration
def test_apply_unmatched_remove_fails(stage5_file, tmp_path):
    fp = tmp_path / "fail.ftmw"
    shutil.copy(stage5_file, fp)
    wid, _ = _a_peak(fp)
    cur = tmp_path / "c.csv"
    cur.write_text(f"remove,{wid},99999.0,\n")  # nowhere near a peak

    # Dry run surfaces the unmatched target as a warning, no mutation.
    dry = apply_curation_impl(fp, cur, dry_run=True)
    assert any("no fitted peak within" in w for w in dry.warnings)

    with pytest.raises(ValueError, match="curation action 1.*failed"):
        apply_curation_impl(fp, cur)


@pytest.mark.integration
def test_review_log_lists_decisions(stage5_file, tmp_path):
    fp = tmp_path / "log.ftmw"
    shutil.copy(stage5_file, fp)
    assert review_log_impl(fp) == []  # nothing recorded yet

    wid, freq = _a_peak(fp)
    apply_curation_impl(fp, _write_curation(tmp_path, f"remove,{wid},{freq},\n"))
    log = review_log_impl(fp)
    assert len(log) == 1
    assert log[0].kind == "remove" and log[0].window_id == wid


def _write_curation(tmp_path: Path, text: str) -> str:
    p = tmp_path / "log_c.csv"
    p.write_text(text)
    return str(p)


@pytest.mark.integration
def test_apply_cross_interface(stage5_file, tmp_path):
    """api / Pipeline / CLI apply produce identical fitted state."""
    paths = {k: tmp_path / f"{k}.ftmw" for k in ("api", "pipe", "cli")}
    for p in paths.values():
        shutil.copy(stage5_file, p)

    wid, freq = _a_peak(paths["api"])
    cur = tmp_path / "x.csv"
    cur.write_text(f"add,{wid},{freq + 0.4},\n")

    ftmw.review_apply(str(paths["api"]), str(cur))
    Pipeline.open(paths["pipe"]).review_apply(str(cur))
    rc = cmd_review_apply(
        argparse.Namespace(
            file_path=str(paths["cli"]), curation_file=str(cur), dry_run=False
        )
    )
    assert rc == 0

    ref = _fitted_by_window(paths["api"])
    assert _fitted_by_window(paths["pipe"]) == ref
    assert _fitted_by_window(paths["cli"]) == ref


@pytest.mark.integration
def test_cli_log_smoke(stage5_file, tmp_path, capsys):
    fp = tmp_path / "clilog.ftmw"
    shutil.copy(stage5_file, fp)
    rc = cmd_review_log(argparse.Namespace(file_path=str(fp)))
    assert rc == 0
    assert "review log" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# review undo: decision -> op conversion (pure)
# ---------------------------------------------------------------------------


def _entry(order, wid, kind, freq, evidence=None):
    return DecisionLogEntry(
        order_index=order,
        window_id=wid,
        frequency_mhz=freq,
        kind=kind,
        evidence=evidence or {},
    )


def test_decision_to_op_add_remove_accept():
    assert _decision_to_op(_entry(0, 5, "add", 100.0)).action == "add"
    assert _decision_to_op(_entry(1, 5, "remove", 101.0)).freqs == [101.0]
    acc = _decision_to_op(_entry(2, 5, "accept", 0.0))
    assert acc.action == "accept" and acc.freqs == []


def test_decision_to_op_merge_uses_merged_from():
    op = _decision_to_op(
        _entry(0, 7, "merge", 50.2, evidence={"merged_from": [50.1, 50.3]})
    )
    assert op.action == "merge" and op.freqs == [50.1, 50.3]


def test_decision_to_op_split_uses_into():
    op = _decision_to_op(
        _entry(0, 7, "split", 50.0, evidence={"split_into": 3})
    )
    assert op.action == "split" and op.freqs == [50.0] and op.params == {"into": "3"}


def test_decision_to_op_merge_without_peaks_raises():
    with pytest.raises(ValueError, match="missing its 'merged_from'"):
        _decision_to_op(_entry(0, 7, "merge", 50.2))


# ---------------------------------------------------------------------------
# review undo: integration
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_edit_snapshots_baseline(stage5_file, tmp_path):
    fp = tmp_path / "snap.ftmw"
    shutil.copy(stage5_file, fp)
    with h5py.File(str(fp), "r") as h5f:
        assert STAGE5_BASELINE_GROUP not in h5f  # none before any edit

    wid, freq = _a_peak(fp)
    apply_curation_impl(fp, _write_curation(tmp_path, f"add,{wid},{freq + 0.4},\n"))
    with h5py.File(str(fp), "r") as h5f:
        assert STAGE5_BASELINE_GROUP in h5f  # snapshotted on the first edit


@pytest.mark.integration
def test_undo_single_returns_to_baseline(stage5_file, tmp_path):
    fp = tmp_path / "u1.ftmw"
    shutil.copy(stage5_file, fp)
    baseline = _fitted_by_window(fp)

    wid, freq = _a_peak(fp)
    apply_curation_impl(fp, _write_curation(tmp_path, f"add,{wid},{freq + 0.4},\n"))
    assert _fitted_by_window(fp) != baseline  # the add changed the window
    log = review_log_impl(fp)
    assert len(log) == 1

    result = review_undo_impl(fp, [log[0].order_index])
    assert result.applied == 0 and len(result.removed) == 1
    assert _fitted_by_window(fp) == baseline  # restored exactly
    assert review_log_impl(fp) == []  # decision log cleared


@pytest.mark.integration
def test_undo_one_of_two_replays_other(stage5_file, tmp_path):
    by_w = _fitted_by_window(stage5_file)
    wids = [w for w, f in by_w.items() if f]
    if len(wids) < 2:
        pytest.skip("Need two windows with peaks")
    wa, wb = wids[0], wids[1]
    fa = by_w[wa][0] + 0.4
    fb = by_w[wb][0] + 0.4

    # Reference: only the wb add applied to the automatic fit.
    ref = tmp_path / "ref.ftmw"
    shutil.copy(stage5_file, ref)
    apply_curation_impl(ref, _write_curation(tmp_path, f"add,{wb},{fb},\n"))

    # Undone: both adds, then undo the wa add.
    both = tmp_path / "both.ftmw"
    shutil.copy(stage5_file, both)
    cur = tmp_path / "two.csv"
    cur.write_text(f"add,{wa},{fa},\nadd,{wb},{fb},\n")
    apply_curation_impl(both, cur)
    log = review_log_impl(both)
    wa_id = next(e.order_index for e in log if e.window_id == wa and e.kind == "add")
    review_undo_impl(both, [wa_id])

    assert _fitted_by_window(both) == _fitted_by_window(ref)


@pytest.mark.integration
def test_undo_dry_run_no_mutation(stage5_file, tmp_path):
    fp = tmp_path / "udry.ftmw"
    shutil.copy(stage5_file, fp)
    wid, freq = _a_peak(fp)
    apply_curation_impl(fp, _write_curation(tmp_path, f"add,{wid},{freq + 0.4},\n"))
    log = review_log_impl(fp)

    before = hashlib.md5(fp.read_bytes()).hexdigest()
    result = review_undo_impl(fp, [log[0].order_index], dry_run=True)
    assert hashlib.md5(fp.read_bytes()).hexdigest() == before
    assert result.dry_run and len(result.removed) == 1


@pytest.mark.integration
def test_undo_unknown_id_and_empty_log(stage5_file, tmp_path):
    fp = tmp_path / "uerr.ftmw"
    shutil.copy(stage5_file, fp)
    with pytest.raises(ValueError, match="no recorded decisions"):
        review_undo_impl(fp, [0])

    wid, freq = _a_peak(fp)
    apply_curation_impl(fp, _write_curation(tmp_path, f"add,{wid},{freq + 0.4},\n"))
    with pytest.raises(ValueError, match="unknown decision id"):
        review_undo_impl(fp, [999])


@pytest.mark.integration
def test_refit_clears_baseline_and_decisions(stage5_file, tmp_path):
    """Re-running the automatic fit resets the curation state (fresh start)."""
    fp = tmp_path / "urefit.ftmw"
    shutil.copy(stage5_file, fp)
    wid, freq = _a_peak(fp)
    apply_curation_impl(fp, _write_curation(tmp_path, f"add,{wid},{freq + 0.4},\n"))
    assert review_log_impl(fp)  # a decision was recorded

    ftmw.fit_peaks(str(fp))
    with h5py.File(str(fp), "r") as h5f:
        assert STAGE5_BASELINE_GROUP not in h5f  # snapshot dropped
    assert review_log_impl(fp) == []  # decisions reset


@pytest.mark.integration
def test_undo_refused_when_baseline_missing(stage5_file, tmp_path):
    """If the baseline is gone while fit-mutating decisions remain, undo refuses."""
    fp = tmp_path / "ubad.ftmw"
    shutil.copy(stage5_file, fp)
    wid, freq = _a_peak(fp)
    apply_curation_impl(fp, _write_curation(tmp_path, f"add,{wid},{freq + 0.4},\n"))
    log = review_log_impl(fp)

    clear_stage5_baseline(fp)  # simulate the snapshot becoming unavailable
    with pytest.raises(ValueError, match="baseline is unavailable"):
        review_undo_impl(fp, [log[0].order_index])


@pytest.mark.integration
def test_undo_cross_interface(stage5_file, tmp_path):
    paths = {k: tmp_path / f"u_{k}.ftmw" for k in ("api", "pipe", "cli")}
    for p in paths.values():
        shutil.copy(stage5_file, p)
    wid, freq = _a_peak(paths["api"])
    cur = tmp_path / "uc.csv"
    cur.write_text(f"add,{wid},{freq + 0.4},\nremove,{wid},{freq},\n")

    ids = {}
    for k, p in paths.items():
        apply_curation_impl(p, cur)
        ids[k] = review_log_impl(p)[0].order_index  # undo the first (the add)

    ftmw.review_undo(str(paths["api"]), [ids["api"]])
    Pipeline.open(paths["pipe"]).review_undo([ids["pipe"]])
    rc = cmd_review_undo(
        argparse.Namespace(file_path=str(paths["cli"]), ids=[ids["cli"]], dry_run=False)
    )
    assert rc == 0

    ref = _fitted_by_window(paths["api"])
    assert _fitted_by_window(paths["pipe"]) == ref
    assert _fitted_by_window(paths["cli"]) == ref
