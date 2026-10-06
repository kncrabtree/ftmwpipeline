"""
Decision-log action groups: one user action's rows replay as one action.

A ``review edit`` with several ``--add``/``--remove`` (or a run of add/remove
rows on one window in a curation file) is ONE joint refit that logs one row
per frequency. Every row carries the ``action_index`` evidence key -- the
``serial`` of the action's first row -- and ``review undo`` /
``review apply --log-prefix`` replay each group as one joint action, exactly
like the original, instead of one refit per row (a different fit, after
which a later remove target can drift beyond snap tolerance).
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
from ftmwpipeline.cli.review_commands import cmd_review_apply, cmd_review_undo
from ftmwpipeline.core.curation import PeakUidToken
from ftmwpipeline.core.data_structures import DecisionLogEntry, Stage6Review
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5
from ftmwpipeline.io.stage6_review_serialization import (
    load_stage6_review_from_hdf5,
    save_stage6_review_to_hdf5,
)
from ftmwpipeline.pipeline import Pipeline
from tests.unit.stage6.test_curation import _clear_add_freq, _close

# Design G1: every write here persists the reference replay of its log.
pytestmark = [pytest.mark.usefixtures("every_write_is_reference")]

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


def _entry(
    order: int,
    wid: int,
    kind: str,
    freq: float,
    evidence: Dict[str, Any],
    *,
    targets: Tuple[int, ...] = (),
    seeds: Tuple[float, ...] = (),
    born: Tuple[int, ...] = (),
) -> DecisionLogEntry:
    return DecisionLogEntry(
        order_index=order,
        window_id=wid,
        frequency_mhz=freq,
        kind=kind,
        provenance="user",
        evidence=evidence,
        serial=order,
        targets=targets,
        seeds_mhz=seeds,
        born_uids=born,
    )


def _uid_tokens(entries: List[DecisionLogEntry]) -> List[PeakUidToken]:
    """The ``remove`` a replay of *entries* reports: their targets, by uid."""
    return [PeakUidToken(int(t)) for e in entries for t in e.targets]


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
        _entry(0, 5, "add", 9.5, {**ev, "action_index": 0}, seeds=(9.5,), born=(95,)),
        _entry(1, 5, "remove", 10.0, {**ev, "action_index": 0}, targets=(100,)),
        _entry(2, 5, "remove", 11.0, {**ev, "action_index": 0}, targets=(110,)),
        _entry(
            3,
            5,
            "remove",
            12.0,
            {"chi2r_after": 2.0, "action_index": 3},
            targets=(120,),
        ),
    ]
    plan = _replay_plan(log)
    assert [(a.kind, a.window_id, a.add, a.remove) for a in plan] == [
        ("edit", 5, [9.5], [PeakUidToken(100), PeakUidToken(110)]),
        ("edit", 5, [], [PeakUidToken(120)]),
    ]


def test_partial_group_survivors_replay_jointly():
    ev = {"chi2r_after": 1.0, "action_index": 0}
    log = [
        _entry(0, 5, "remove", 10.0, ev, targets=(100,)),
        _entry(1, 5, "remove", 11.0, ev, targets=(110,)),
        _entry(2, 5, "remove", 12.0, ev, targets=(120,)),
    ]
    surviving = [log[0], log[2]]  # the middle row undone
    plan = _replay_plan(surviving)
    assert [(a.kind, a.remove) for a in plan] == [
        ("edit", [PeakUidToken(100), PeakUidToken(120)])
    ]


def test_a_row_without_the_key_never_joins_a_stamped_one():
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

    log = review_log_impl(path)
    dry = review_undo_impl(path, [2], dry_run=True)
    assert [(a.kind, a.window_id, a.remove) for a in dry.plan] == [
        ("edit", wid, _uid_tokens(log[:2]))
    ]

    review_undo_impl(path, [2])

    assert _fitted_by_window(path) == _saved_state(tmp_path)
    assert _log_shape(path) == [(0, wid, "remove", 0), (1, wid, "remove", 0)]


@pytest.mark.integration
def test_undo_keeps_action_index_when_positions_move(stage5_multi_file, tmp_path):
    """(5) Rows are immutable: a group behind an undone row moves up in the
    log (its ``order_index`` changes) but keeps its ``action_index``, the
    serial of its first row."""
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
    assert _log_shape(path) == [(0, wid, "remove", 1), (1, wid, "remove", 1)]
    assert [e.serial for e in review_log_impl(path)] == [1, 2]


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

    log = review_log_impl(path)
    result = review_undo_impl(path, [0])  # the add
    assert [(a.kind, a.add, a.remove) for a in result.plan] == [
        ("edit", [], _uid_tokens(log[1:]))
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
    # The new row takes the next serial above every one ever recorded (the
    # dropped remove held 2), never a reused one.
    assert _log_shape(path) == [
        (0, wid, "add", 0),
        (1, wid, "remove", 0),
        (2, other, "add", 3),
    ]


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


# ---------------------------------------------------------------------------
# Integration: inferred merge + residual rows in one keyed group
# ---------------------------------------------------------------------------


def _uids(path: Path, wid: int) -> List[str]:
    with h5py.File(str(path), "r") as h5f:
        sf = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    wf = next(w for w in sf.window_fits if w.window_id == wid)
    return sorted(str(p.peak_uid) for p in wf.fitted_peaks)


def _split_site(path: Path) -> Tuple[int, float, float, float, int]:
    """``(wid, parent, near, far, other)``: a window peak, an add frequency
    half a snap tolerance beside it (an inferred split) and one far from every
    peak (a residual plain add), and another window."""
    by_w = _fitted_by_window(path)
    tol = _snap_tol(path)
    for wid, freqs in by_w.items():
        other = next((w for w in by_w if w != wid), None)
        if other is None:
            continue
        far = _clear_add_freq(path, wid)
        for parent in freqs:
            near = parent + 0.5 * tol
            if all(abs(near - f) > tol for f in freqs if f != parent) and all(
                abs(far - f) > tol for f in freqs + [near]
            ):
                return wid, parent, near, far, other
    pytest.skip("Need a window with room for a split add and a clear add")


def _undo_later_unrelated(path: Path, tmp_path: Path, other: int) -> None:
    cur = tmp_path / "later.csv"
    cur.write_text(f"add,{other},{_clear_add_freq(path, other)},\n")
    apply_curation_impl(path, cur)
    review_undo_impl(path, [len(review_log_impl(path)) - 1])


@pytest.mark.integration
def test_undo_keyed_inferred_split_with_residual_rows(stage5_multi_file, tmp_path):
    """A keyed group holding an inferred split plus a residual add replays,
    after undoing a later unrelated decision, to the same fitted state, the
    same peak uids and the same inferred split."""
    path = stage5_multi_file
    wid, parent, near, far, other = _split_site(path)

    refit_window_impl(str(path), wid, add=[near, far])
    log = review_log_impl(path)
    assert sorted(e.kind for e in log) == ["add", "split"]
    assert {e.evidence.get(ACTION_INDEX_EVIDENCE_KEY) for e in log} == {0}
    assert next(e for e in log if e.kind == "split").evidence["inferred"] is True
    state, shape, uids = _fitted_by_window(path), _log_shape(path), _uids(path, wid)

    _undo_later_unrelated(path, tmp_path, other)

    assert _fitted_by_window(path) == state
    assert _log_shape(path) == shape
    assert _uids(path, wid) == uids
    split = next(e for e in review_log_impl(path) if e.kind == "split")
    assert split.evidence["inferred"] is True
    assert split.evidence["requested_freq_mhz"] == pytest.approx(near)


@pytest.mark.integration
def test_undo_keyed_inferred_merge_with_residual_rows(stage5_multi_file, tmp_path):
    """Same for an inferred merge: a split first makes a blend inside snap
    tolerance, then one edit removes both products, adds between and adds
    a far line."""
    path = stage5_multi_file
    wid, parent, near, far, other = _split_site(path)
    refit_window_impl(str(path), wid, add=[near])
    blend = sorted(
        f for f in _fitted_by_window(path)[wid] if abs(f - parent) < _snap_tol(path)
    )
    assert len(blend) == 2, blend
    refit_window_impl(
        str(path), wid, add=[0.5 * (blend[0] + blend[1]), far], remove=blend
    )
    log = review_log_impl(path)
    assert [(e.kind, e.evidence[ACTION_INDEX_EVIDENCE_KEY]) for e in log] == [
        ("split", 0),
        ("merge", 1),
        ("add", 1),
    ]
    assert log[1].evidence["inferred"] is True
    state, shape, uids = _fitted_by_window(path), _log_shape(path), _uids(path, wid)

    _undo_later_unrelated(path, tmp_path, other)

    assert _fitted_by_window(path) == state
    assert _log_shape(path) == shape
    assert _uids(path, wid) == uids


# ---------------------------------------------------------------------------
# Integration: stamping through the preview-then-apply session path
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_session_preview_then_apply_stamps_one_index_per_action(
    stage5_multi_file, tmp_path
):
    """``ReviewSession`` persists the preview's own decisions: they must be
    grouped by action exactly as a direct apply groups them (both are rows
    :func:`_request_rows` stamps, one action per plan action)."""
    path = stage5_multi_file
    wid, peaks, other = _multi_peak_window(path)
    fv = _clear_add_freq(path, other)
    cur = tmp_path / "c.csv"
    cur.write_text(
        f"remove,{wid},{peaks[0]},\nremove,{wid},{peaks[1]},\nadd,{other},{fv},\n"
    )

    with Pipeline.open(path).review_session() as session:
        session.review_preview(cur)
        session.review_apply(cur)

    # The plan runs windows in canonical order, so the add may precede the
    # removes; either way each action's rows carry its first row's index.
    shape = _log_shape(path)
    assert [(w, k) for _, w, k, _ in shape if w == wid] == [
        (wid, "remove"),
        (wid, "remove"),
    ]
    first = {w: min(o for o, ww, _, _ in shape if ww == w) for w in (wid, other)}
    assert [(o, ai) for o, w, _, ai in shape] == [(o, first[w]) for o, w, _, _ in shape]
    assert first[wid] != first[other]


# ---------------------------------------------------------------------------
# Integration: log-prefix apply
# ---------------------------------------------------------------------------


def _prefix_setup(source: Path, tmp_path: Path, name: str) -> Tuple[Path, Path, int]:
    """A copy holding a joint two-remove edit then an unrelated add, and the
    curation file that repeats that add."""
    path = tmp_path / f"{name}.ftmw"
    shutil.copy(source, path)
    wid, peaks, other = _multi_peak_window(path)
    cur = tmp_path / "later.csv"
    cur.write_text(f"add,{other},{_clear_add_freq(path, other)},\n")
    refit_window_impl(str(path), wid, remove=[peaks[0], peaks[1]])
    apply_curation_impl(path, cur)
    return path, cur, wid


@pytest.mark.integration
def test_group_log_prefix_apply_cross_interface(stage5_multi_source, tmp_path):
    """api / Pipeline / CLI ``review apply --log-prefix`` agree, fit and log
    alike, for a prefix that keeps a joint edit."""
    paths: Dict[str, Path] = {}
    for k in ("api", "pipe", "cli"):
        paths[k], cur, wid = _prefix_setup(stage5_multi_source, tmp_path, k)

    ftmw.review_apply(str(paths["api"]), cur, log_prefix=2)
    Pipeline.open(paths["pipe"]).review_apply(cur, log_prefix=2)
    rc = cmd_review_apply(
        argparse.Namespace(
            file_path=str(paths["cli"]),
            curation_file=str(cur),
            actions=None,
            dry_run=False,
            log_prefix=2,
        )
    )
    assert rc == 0

    ref_fit, ref_log = _fitted_by_window(paths["api"]), _log_shape(paths["api"])
    assert [(k, ai) for _, _, k, ai in ref_log] == [
        ("remove", 0),
        ("remove", 0),
        ("add", 3),
    ]
    for k in ("pipe", "cli"):
        assert _fitted_by_window(paths[k]) == ref_fit, k
        assert _log_shape(paths[k]) == ref_log, k


# ---------------------------------------------------------------------------
# Regression: a joint action must not replay as per-row refits
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_replay_refits_once_per_action_not_once_per_row(
    stage5_multi_file, tmp_path, monkeypatch, every_write_is_reference
):
    """The old per-row replay resolved each log row on its own: two refits
    for a two-remove edit, a different (sequential) fit. Count the refits an
    undo performs and pin the plan's shape. The engine's keys are dropped
    first, so the undo recomputes the joint edit's window from its rows
    rather than keeping its persisted fit."""
    # This counts work, which the G1 check's own replays would add to.
    every_write_is_reference.enabled = False
    from ftmwpipeline._internal import stage6_impl as s6
    from ftmwpipeline.io.stage6_engine_serialization import (
        save_stage6_engine_state,
    )

    path = stage5_multi_file
    wid, peaks, other = _joint_then_unrelated(path, tmp_path)
    log = review_log_impl(path)
    assert len(_replay_plan(log[:2])) == 1  # one edit for the two rows
    with h5py.File(path, "a") as h5f:
        save_stage6_engine_state(h5f, None)

    calls: List[Tuple[int, Tuple[int, ...]]] = []
    real = s6._apply_refit_step

    def counting(ctx: Any, step: Any, **kw: Any) -> Any:
        calls.append((step.window_id, tuple(t for r in step.rows for t in r.targets)))
        return real(ctx, step, **kw)

    monkeypatch.setattr(s6, "_apply_refit_step", counting)
    review_undo_impl(path, [2])

    assert calls == [(wid, tuple(t for e in log[:2] for t in e.targets))]
    assert _fitted_by_window(path) == _saved_state(tmp_path)


def _rewrite_rows(path: Path, fn: Any) -> None:
    """Rewrite the persisted decision log's rows in place (``fn(rows)``), to
    forge a row an older version recorded."""
    with h5py.File(str(path), "a") as h5f:
        grp = h5f["stage6_review"]["decision_log"]
        rows = json.loads(str(grp.attrs["data"]))
        fn(rows)
        grp.attrs["data"] = json.dumps(rows)


def _raw_rows(path: Path) -> List[Dict[str, Any]]:
    return [
        {"window_id": e.window_id, "kind": e.kind, "frequency_mhz": e.frequency_mhz}
        for e in review_log_impl(path)
    ]


@pytest.mark.integration
def test_remove_is_logged_at_the_peak_it_resolved_to(stage5_multi_file):
    """A remove sent half a snap tolerance off its peak is logged at the
    peak's fitted frequency, not at the frequency sent."""
    path = stage5_multi_file
    wid, peaks, _ = _multi_peak_window(path)
    with h5py.File(str(path), "r") as h5f:
        sf = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    wf = next(w for w in sf.window_fits if w.window_id == wid)
    target = min(float(p.frequency_mhz) for p in wf.fitted_peaks)

    refit_window_impl(str(path), wid, remove=[target + 0.5 * _snap_tol(path)])
    assert [(e.kind, e.frequency_mhz) for e in review_log_impl(path)] == [
        ("remove", target)
    ]


@pytest.mark.integration
def test_replay_keeps_every_surviving_row_verbatim(stage5_multi_file, tmp_path):
    """Undo and a log-prefix apply re-record the surviving rows with their own
    frequencies, byte for byte -- also an older row that holds the frequency
    sent rather than the peak it resolved to."""
    path = stage5_multi_file
    wid, peaks, other = _joint_then_unrelated(path, tmp_path)
    off = 0.5 * _snap_tol(path)

    def legacy(rows: List[Dict[str, Any]]) -> None:
        rows[0]["frequency_mhz"] = float(rows[0]["frequency_mhz"]) + off

    _rewrite_rows(path, legacy)
    before = _raw_rows(path)
    assert before[0]["frequency_mhz"] != peaks[0]

    review_undo_impl(path, [2])
    assert _raw_rows(path) == before[:2]

    review_undo_impl(path, [1])  # one member of the joint group
    assert _raw_rows(path) == before[:1]


@pytest.mark.integration
def test_log_prefix_apply_keeps_the_kept_rows_verbatim(stage5_multi_file, tmp_path):
    path = stage5_multi_file
    wid, peaks, other = _joint_then_unrelated(path, tmp_path)
    before = _raw_rows(path)
    cur = tmp_path / "next.csv"
    cur.write_text(f"add,{other},{_clear_add_freq(path, other)},\n")
    apply_curation_impl(path, cur, log_prefix=2)
    assert _raw_rows(path)[:2] == before[:2]


@pytest.mark.integration
def test_a_row_whose_target_is_gone_is_refused(stage5_multi_file, tmp_path):
    """A replay applies each row by peak identity: a row whose target the
    window does not hold at its place in the log (forged here) refuses the
    replay rather than act on another peak, before anything is fit, and
    leaves the file as it was."""
    from ftmwpipeline.file_manager import CurationConflictError

    path = stage5_multi_file
    wid, parent, near, far, other = _split_site(path)
    refit_window_impl(str(path), wid, add=[near])
    assert [e.kind for e in review_log_impl(path)] == ["split"]

    def forged(rows: List[Dict[str, Any]]) -> None:
        rows[0]["targets"] = [int(rows[0]["targets"][0]) + 7]

    cur = tmp_path / "later.csv"
    cur.write_text(f"add,{other},{_clear_add_freq(path, other)},\n")
    apply_curation_impl(path, cur)
    _rewrite_rows(path, forged)
    before = _raw_rows(path)
    state = _fitted_by_window(path)

    with pytest.raises(CurationConflictError) as exc:
        review_undo_impl(path, [1])
    assert exc.value.reason == "replay_diverged"
    assert exc.value.ids == [0]
    assert _raw_rows(path) == before
    assert _fitted_by_window(path) == state


@pytest.mark.integration
def test_replay_keeps_the_log_order_across_windows(stage5_multi_file, tmp_path):
    """Edits made in separate calls on a higher, then a lower window: an undo
    of a later decision replays them in log order, not canonical (ascending
    window) order, and re-records every surviving row as it was."""
    path = stage5_multi_file
    by_w = _fitted_by_window(path)
    wids = sorted(w for w, f in by_w.items() if f)
    if len(wids) < 3:
        pytest.skip("Need three fitted windows")
    lo, hi, third = wids[0], wids[-1], wids[1]
    refit_window_impl(str(path), hi, add=[_clear_add_freq(path, hi)])
    refit_window_impl(str(path), lo, add=[_clear_add_freq(path, lo)])
    before = _raw_rows(path)
    assert [r["window_id"] for r in before] == [hi, lo]
    state = _fitted_by_window(path)

    cur = tmp_path / "later.csv"
    cur.write_text(f"add,{third},{_clear_add_freq(path, third)},\n")
    apply_curation_impl(path, cur)
    review_undo_impl(path, [2])

    assert _raw_rows(path) == before
    assert _fitted_by_window(path) == state
