"""
Decision-log action groups: one user action's rows replay as one action.

A ``review edit`` with several ``--add``/``--remove`` (or a run of add/remove
rows on one window in a curation file) is ONE joint refit that logs one row
per frequency. Every row carries the ``action_index`` evidence key -- the
``order_index`` of the action's first row -- and ``review undo`` /
``review apply --log-prefix`` replay each group as one joint action, exactly
like the original, instead of one refit per row (a different fit, after
which a later remove target can drift beyond snap tolerance). Rows recorded
before the key existed are grouped by inference.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.stage6_impl import (
    ACTION_INDEX_EVIDENCE_KEY,
    _decision_action_groups,
    _replay_plan,
    apply_curation_impl,
    refit_window_impl,
    review_log_impl,
    review_undo_impl,
)
from ftmwpipeline.cli.review_commands import cmd_review_undo
from ftmwpipeline.core.data_structures import DecisionLogEntry, Stage6Review
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5
from ftmwpipeline.io.stage6_review_serialization import (
    load_stage6_review_from_hdf5,
    save_stage6_review_to_hdf5,
)
from ftmwpipeline.pipeline import Pipeline
from tests.unit.stage6.test_curation import _clear_add_freq, _close

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fitted_by_window(path: Path) -> Dict[int, List[float]]:
    with h5py.File(str(path), "r") as h5f:
        sf = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    return {
        int(wf.window_id): sorted(
            round(float(p.frequency_mhz), 6) for p in wf.fitted_peaks
        )
        for wf in sf.window_fits
        if wf.window_id is not None
    }


def _multi_peak_window(path: Path) -> Tuple[int, List[float], int]:
    """``(wid, peaks, other)``: a window with at least three fitted peaks no
    two of which sit within snap tolerance of each other (so removing two of
    them is a plain joint edit, never an inferred merge), its sorted peak
    frequencies, and a different fitted window for an unrelated decision."""
    by_w = _fitted_by_window(path)
    tol = _snap_tol(path)
    for wid, freqs in by_w.items():
        if len(freqs) >= 3 and all(b - a > tol for a, b in zip(freqs, freqs[1:])):
            other = next((w for w in by_w if w != wid), None)
            if other is None:
                break
            return wid, freqs, other
    pytest.skip("Need a window with three separated peaks plus another window")


def _snap_tol(path: Path) -> float:
    from ftmwpipeline._internal import stage6_impl as s6

    return float(s6.refit_snap_tol_mhz_impl(str(path)))


def _log_shape(path: Path) -> List[Tuple[int, int, str, Optional[int]]]:
    return [
        (
            e.order_index,
            e.window_id,
            e.kind,
            e.evidence.get(ACTION_INDEX_EVIDENCE_KEY),
        )
        for e in review_log_impl(path)
    ]


def _strip_action_index(path: Path) -> None:
    """Rewrite the persisted decision log as a file written before the
    ``action_index`` key existed would hold it."""
    with h5py.File(str(path), "a") as h5f:
        grp = h5f["stage6_review"]["decision_log"]
        rows = json.loads(str(grp.attrs["data"]))
        for row in rows:
            row["evidence"].pop(ACTION_INDEX_EVIDENCE_KEY, None)
        grp.attrs["data"] = json.dumps(rows)


def _entry(
    order: int, wid: int, kind: str, freq: float, evidence: Dict[str, Any]
) -> DecisionLogEntry:
    return DecisionLogEntry(
        order_index=order,
        window_id=wid,
        frequency_mhz=freq,
        kind=kind,
        provenance="user",
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Pure: grouping and group replay
# ---------------------------------------------------------------------------


def test_groups_follow_action_index():
    ev = {"chi2r_after": 1.0}
    log = [
        _entry(0, 5, "remove", 10.0, {**ev, "action_index": 0}),
        _entry(1, 5, "remove", 11.0, {**ev, "action_index": 0}),
        # Same window, same kind, identical evidence apart from the key: a
        # different action all the same.
        _entry(2, 5, "remove", 12.0, {**ev, "action_index": 2}),
        _entry(3, 6, "accept", 0.0, {"action_index": 3}),
    ]
    groups = _decision_action_groups(log)
    assert [[e.order_index for e in g] for g in groups] == [[0, 1], [2], [3]]


def test_group_replays_as_one_edit_and_actions_stay_separate():
    ev = {"chi2r_after": 1.0}
    log = [
        _entry(0, 5, "add", 9.5, {**ev, "action_index": 0}),
        _entry(1, 5, "remove", 10.0, {**ev, "action_index": 0}),
        _entry(2, 5, "remove", 11.0, {**ev, "action_index": 0}),
        _entry(3, 5, "remove", 12.0, {"chi2r_after": 2.0, "action_index": 3}),
    ]
    plan = _replay_plan(log)
    assert [(a.kind, a.window_id, a.add, a.remove) for a in plan] == [
        ("edit", 5, [9.5], [10.0, 11.0]),
        ("edit", 5, [], [12.0]),
    ]


def test_partial_group_survivors_replay_jointly():
    ev = {"chi2r_after": 1.0, "action_index": 0}
    log = [
        _entry(0, 5, "remove", 10.0, ev),
        _entry(1, 5, "remove", 11.0, ev),
        _entry(2, 5, "remove", 12.0, ev),
    ]
    surviving = [log[0], log[2]]  # the middle row undone
    plan = _replay_plan(surviving)
    assert [(a.kind, a.remove) for a in plan] == [("edit", [10.0, 12.0])]


def test_legacy_rows_grouped_by_inference():
    """No ``action_index``: consecutive add/remove rows on one window with
    the SAME evidence dict (one refit's snapshot) are one action."""
    joint = {"chi2r_before": 3.0, "chi2r_after": 1.25, "n_peaks_before": 4}
    log = [
        _entry(0, 5, "add", 9.5, dict(joint)),
        _entry(1, 5, "remove", 10.0, dict(joint)),
        _entry(2, 5, "remove", 11.0, dict(joint)),
        # A later, separate refit of the same window: its own evidence.
        _entry(3, 5, "remove", 12.0, {**joint, "chi2r_after": 1.5}),
        # Identical evidence but another window: separate.
        _entry(4, 6, "remove", 20.0, {**joint, "chi2r_after": 1.5}),
        # Bare accepts never group.
        _entry(5, 6, "accept", 0.0, {}),
        _entry(6, 6, "accept", 0.0, {}),
        # An implied-create add is always its own action.
        _entry(7, 9, "add", 30.0, {"created_window": {"mode": "created"}}),
        _entry(8, 9, "add", 30.5, {"created_window": {"mode": "created"}}),
    ]
    groups = _decision_action_groups(log)
    assert [[e.order_index for e in g] for g in groups] == [
        [0, 1, 2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8],
    ]


def test_legacy_row_never_joins_a_stamped_one():
    ev = {"chi2r_after": 1.0}
    log = [
        _entry(0, 5, "remove", 10.0, dict(ev)),
        _entry(1, 5, "remove", 11.0, {**ev, "action_index": 1}),
    ]
    assert len(_decision_action_groups(log)) == 2


def test_action_index_round_trips_through_serialization(tmp_path):
    review = Stage6Review(
        decision_log=[
            _entry(0, 5, "remove", 10.0, {"chi2r_after": 1.0, "action_index": 0}),
            _entry(1, 5, "remove", 11.0, {"chi2r_after": 1.0, "action_index": 0}),
            _entry(2, 7, "accept", 0.0, {"action_index": 2}),
        ]
    )
    fp = tmp_path / "rt.h5"
    with h5py.File(str(fp), "w") as h5f:
        save_stage6_review_to_hdf5(review, h5f.create_group("stage6_review"))
    with h5py.File(str(fp), "r") as h5f:
        loaded = load_stage6_review_from_hdf5(h5f["stage6_review"])
    assert loaded.decision_log == review.decision_log
    values = [e.evidence[ACTION_INDEX_EVIDENCE_KEY] for e in loaded.decision_log]
    assert values == [0, 0, 2]
    assert all(type(v) is int for v in values)


# ---------------------------------------------------------------------------
# Integration: recording
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_recording_stamps_one_index_per_action(stage5_multi_file, tmp_path):
    path = stage5_multi_file
    wid, peaks, other = _multi_peak_window(path)
    fv = _clear_add_freq(path, other)

    refit_window_impl(str(path), wid, remove=[peaks[0], peaks[1]])  # one action
    cur = tmp_path / "c.csv"
    cur.write_text(f"add,{other},{fv},\n")
    apply_curation_impl(path, cur)
    ftmw.review_accept(str(path), other)

    assert _log_shape(path) == [
        (0, wid, "remove", 0),
        (1, wid, "remove", 0),
        (2, other, "add", 2),
        (3, other, "accept", 3),
    ]


# ---------------------------------------------------------------------------
# Integration: undo / log-prefix replay
# ---------------------------------------------------------------------------


def _joint_then_unrelated(path: Path, tmp_path: Path) -> Tuple[int, List[float], int]:
    """A joint two-remove edit of one window, then an unrelated add in
    another; returns ``(wid, peaks, other)`` and leaves the state after the
    joint edit in ``tmp_path / 'after_joint.json'``."""
    wid, peaks, other = _multi_peak_window(path)
    fv = _clear_add_freq(path, other)
    refit_window_impl(str(path), wid, remove=[peaks[0], peaks[1]])
    (tmp_path / "after_joint.json").write_text(
        json.dumps({str(k): v for k, v in _fitted_by_window(path).items()})
    )
    cur = tmp_path / "later.csv"
    cur.write_text(f"add,{other},{fv},\n")
    apply_curation_impl(path, cur)
    return wid, peaks, other


def _saved_state(tmp_path: Path) -> Dict[int, List[float]]:
    raw = json.loads((tmp_path / "after_joint.json").read_text())
    return {int(k): v for k, v in raw.items()}


@pytest.mark.integration
def test_undo_later_decision_after_joint_edit(stage5_multi_file, tmp_path):
    """(a) The real failure: a multi-remove edit, then an unrelated later
    decision. Undoing the later one replays the edit as ONE joint refit, so
    the window is exactly as it was before that later decision."""
    path = stage5_multi_file
    wid, peaks, other = _joint_then_unrelated(path, tmp_path)

    dry = review_undo_impl(path, [2], dry_run=True)
    assert [(a.kind, a.window_id, a.remove) for a in dry.plan] == [
        ("edit", wid, [peaks[0], peaks[1]])
    ]

    review_undo_impl(path, [2])

    assert _fitted_by_window(path) == _saved_state(tmp_path)
    assert _log_shape(path) == [(0, wid, "remove", 0), (1, wid, "remove", 0)]


@pytest.mark.integration
def test_undo_restamps_action_index_after_renumbering(stage5_multi_file, tmp_path):
    """(5) Replay re-records the log: a group behind an undone row takes the
    renumbered ``order_index`` of its first row as its ``action_index``."""
    path = stage5_multi_file
    wid, peaks, other = _multi_peak_window(path)
    fv = _clear_add_freq(path, other)
    cur = tmp_path / "first.csv"
    cur.write_text(f"add,{other},{fv},\n")
    apply_curation_impl(path, cur)
    refit_window_impl(str(path), wid, remove=[peaks[0], peaks[1]])
    assert _log_shape(path) == [
        (0, other, "add", 0),
        (1, wid, "remove", 1),
        (2, wid, "remove", 1),
    ]

    review_undo_impl(path, [0])
    assert _log_shape(path) == [(0, wid, "remove", 0), (1, wid, "remove", 0)]


@pytest.mark.integration
def test_undo_one_row_of_group_replays_survivors_jointly(stage5_multi_source, tmp_path):
    """(b) Undoing one row of a joint edit replays the group's other rows
    jointly: the fit equals a fresh joint edit of just those rows."""
    wid, peaks, other = _multi_peak_window(stage5_multi_source)
    fa = _clear_add_freq(stage5_multi_source, wid)

    path = tmp_path / "group.ftmw"
    shutil.copy(stage5_multi_source, path)
    refit_window_impl(str(path), wid, add=[fa], remove=[peaks[0], peaks[1]])
    shape = _log_shape(path)
    assert [(k, ai) for _, _, k, ai in shape] == [
        ("add", 0),
        ("remove", 0),
        ("remove", 0),
    ]

    ref = tmp_path / "ref.ftmw"
    shutil.copy(stage5_multi_source, ref)
    refit_window_impl(str(ref), wid, remove=[peaks[0], peaks[1]])

    result = review_undo_impl(path, [0])  # the add
    assert [(a.kind, a.add, a.remove) for a in result.plan] == [
        ("edit", [], [peaks[0], peaks[1]])
    ]
    assert _fitted_by_window(path) == _fitted_by_window(ref)
    assert _log_shape(path) == _log_shape(ref)


@pytest.mark.integration
def test_log_prefix_mid_group_replays_in_prefix_rows_jointly(
    stage5_multi_source, tmp_path
):
    """(c) A log prefix that cuts through a joint edit replays that edit's
    in-prefix rows together, as one action."""
    wid, peaks, other = _multi_peak_window(stage5_multi_source)
    fa = _clear_add_freq(stage5_multi_source, wid)
    fv = _clear_add_freq(stage5_multi_source, other)

    path = tmp_path / "prefix.ftmw"
    shutil.copy(stage5_multi_source, path)
    refit_window_impl(str(path), wid, add=[fa], remove=[peaks[0], peaks[1]])
    cur = tmp_path / "next.csv"
    cur.write_text(f"add,{other},{fv},\n")

    ref = tmp_path / "ref.ftmw"
    shutil.copy(stage5_multi_source, ref)
    refit_window_impl(str(ref), wid, add=[fa], remove=[peaks[0]])
    apply_curation_impl(ref, cur)

    apply_curation_impl(path, cur, log_prefix=2)  # keeps the add + first remove

    assert _close(_fitted_by_window(path), _fitted_by_window(ref))
    assert _log_shape(path) == [
        (0, wid, "add", 0),
        (1, wid, "remove", 0),
        (2, other, "add", 2),
    ]


@pytest.mark.integration
def test_legacy_log_without_action_index_is_grouped_by_inference(
    stage5_multi_file, tmp_path
):
    """(d) A file written before the key existed: the joint edit's rows are
    recognised by their shared evidence and still replay as one action."""
    path = stage5_multi_file
    wid, peaks, other = _joint_then_unrelated(path, tmp_path)
    _strip_action_index(path)
    assert all(ai is None for *_, ai in _log_shape(path))

    dry = review_undo_impl(path, [2], dry_run=True)
    assert [(a.kind, a.remove) for a in dry.plan] == [("edit", [peaks[0], peaks[1]])]
    review_undo_impl(path, [2])

    assert _fitted_by_window(path) == _saved_state(tmp_path)
    # The replay re-records the log, stamping the key from then on.
    assert _log_shape(path) == [(0, wid, "remove", 0), (1, wid, "remove", 0)]


@pytest.mark.integration
def test_group_undo_cross_interface(stage5_multi_source, tmp_path):
    """(f) api / Pipeline / CLI undo of a decision behind a joint edit agree,
    fit and log alike."""
    wid, peaks, other = _multi_peak_window(stage5_multi_source)
    fv = _clear_add_freq(stage5_multi_source, other)
    cur = tmp_path / "later.csv"
    cur.write_text(f"add,{other},{fv},\n")

    paths = {k: tmp_path / f"g_{k}.ftmw" for k in ("api", "pipe", "cli")}
    for p in paths.values():
        shutil.copy(stage5_multi_source, p)
        refit_window_impl(str(p), wid, remove=[peaks[0], peaks[1]])
        apply_curation_impl(p, cur)

    ftmw.review_undo(str(paths["api"]), [2])
    Pipeline.open(paths["pipe"]).review_undo([2])
    rc = cmd_review_undo(
        argparse.Namespace(file_path=str(paths["cli"]), ids=[2], dry_run=False)
    )
    assert rc == 0

    ref_fit = _fitted_by_window(paths["api"])
    ref_log = _log_shape(paths["api"])
    assert ref_log == [(0, wid, "remove", 0), (1, wid, "remove", 0)]
    for k in ("pipe", "cli"):
        assert _fitted_by_window(paths[k]) == ref_fit, k
        assert _log_shape(paths[k]) == ref_log, k
