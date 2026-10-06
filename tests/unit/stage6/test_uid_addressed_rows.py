"""
uid-addressed decision rows.

A request is resolved, before anything is fit, against the state the user
sees (each window's persisted fit) into decision-log rows that name peaks by
identity: ``targets`` (the ``peak_uid`` values a row removes), ``seeds_mhz``
(where the peaks it births are seeded) and ``born_uids`` (the uid each was
stamped with, from its seed). A replay applies the rows as recorded -- never
re-resolving a frequency or re-inferring an edit -- so it acts on the same
peaks however far the cascade has moved the lines in between. Every refusal
(a gone or colliding uid, a target held twice, a seed off its window, a birth
on a held uid, an orphaned birth) is raised before the first fit.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal import stage6_impl as s6
from ftmwpipeline.core.data_structures import FittingResult
from ftmwpipeline.file_manager import CurationConflictError, NotFoundValueError
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5
from tests._events_support import content_digest
from tests.unit.stage6.test_curation import _clear_add_freq

pytestmark = [
    pytest.mark.integration,
    # Design G1: every write here persists the reference replay of its log.
    pytest.mark.usefixtures("every_write_is_reference"),
]


def _fits(path: Path) -> Dict[int, FittingResult]:
    with h5py.File(str(path), "r") as h5f:
        sf = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    return {int(w.window_id): w for w in sf.window_fits if w.window_id is not None}


def _uids(path: Path, wid: int) -> List[int]:
    return sorted(int(p.peak_uid) for p in _fits(path)[wid].fitted_peaks)


def _snap_tol(path: Path) -> float:
    return float(s6.refit_snap_tol_mhz_impl(str(path)))


def _windows_with_peaks(path: Path, n: int, min_peaks: int = 1) -> List[int]:
    fits = _fits(path)
    wids = [w for w in sorted(fits) if len(fits[w].fitted_peaks) >= min_peaks]
    if len(wids) < n:
        pytest.skip(f"Need {n} windows with {min_peaks}+ fitted peaks")
    return wids[:n]


def _split_site(path: Path) -> Tuple[int, float, float]:
    """``(wid, parent, near)``: a fitted peak and an add half a snap tolerance
    beside it that no other peak is within snap tolerance of (an inferred
    split)."""
    tol = _snap_tol(path)
    for wid, wf in _fits(path).items():
        freqs = sorted(float(p.frequency_mhz) for p in wf.fitted_peaks)
        for parent in freqs:
            near = parent + 0.5 * tol
            if all(abs(near - f) > tol for f in freqs if f != parent):
                return wid, parent, near
    pytest.skip("Need a window with room for a split add")


@pytest.fixture
def forbid_fits(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Call to make any later window refit fail the test: a refusal comes
    before the first fit."""

    def arm() -> None:
        def refit(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("a refused write fitted a window")

        monkeypatch.setattr(s6, "refit_window_core", refit)

    return arm


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def test_rows_record_targets_seeds_and_born_uids(stage5_multi_file):
    path = stage5_multi_file
    (wid,) = _windows_with_peaks(path, 1, min_peaks=2)
    wf = _fits(path)[wid]
    victim = min(wf.fitted_peaks, key=lambda p: float(p.frequency_mhz))
    clear = _clear_add_freq(path, wid)

    ftmw.review_edit(
        str(path), wid, add=[clear], remove=[float(victim.frequency_mhz)], frame="raw"
    )

    add, remove = ftmw.review_log(str(path))
    assert (add.kind, remove.kind) == ("add", "remove")
    assert add.targets == () and len(add.seeds_mhz) == 1 and len(add.born_uids) == 1
    assert add.frequency_mhz == pytest.approx(clear)
    assert remove.targets == (int(victim.peak_uid),)
    assert remove.seeds_mhz == () and remove.born_uids == ()
    # The remove is displayed at the peak it resolved to.
    assert remove.frequency_mhz == float(victim.frequency_mhz)
    after = {int(p.peak_uid): p for p in _fits(path)[wid].fitted_peaks}
    assert int(victim.peak_uid) not in after
    born = after[add.born_uids[0]]
    assert born.derivation == add.serial
    # The uid is the seed's: stamped once, from the recorded seed position.
    assert born.peak_uid == s6._mint_peak_uid(
        _ctx_for(path), _fit_window(path, wid), add.seeds_mhz[0]
    )


def _ctx_for(path: Path) -> Any:
    return s6._build_batch_ctx(str(path), snap_tol_mhz=_snap_tol(path))


def _fit_window(path: Path, wid: int) -> Any:
    return next(
        w for w in s6.effective_window_plan(str(path)).windows if w.window_id == wid
    )


def test_split_and_merge_rows_name_their_children(stage5_multi_file, tmp_path):
    """An inferred split records its parent and the uids of both children;
    a merge of those children records them as its targets and its one child's
    uid. An undo of a later unrelated decision replays both as recorded."""
    path = stage5_multi_file
    wid, parent, near = _split_site(path)
    other = next(w for w in _fits(path) if w != wid)
    parent_uid = next(
        int(p.peak_uid)
        for p in _fits(path)[wid].fitted_peaks
        if float(p.frequency_mhz) == parent
    )

    ftmw.review_edit(str(path), wid, add=[near], frame="raw")
    (split,) = ftmw.review_log(str(path))
    assert split.kind == "split" and split.targets == (parent_uid,)
    assert len(split.seeds_mhz) == len(split.born_uids) == 2
    kids = {int(p.peak_uid): p for p in _fits(path)[wid].fitted_peaks}
    assert set(split.born_uids) <= set(kids) and parent_uid not in kids
    assert {kids[u].derivation for u in split.born_uids} == {split.serial}

    a, b = split.born_uids
    between = 0.5 * sum(float(kids[u].frequency_mhz) for u in (a, b))
    if abs(float(kids[a].frequency_mhz) - float(kids[b].frequency_mhz)) > _snap_tol(
        path
    ):
        pytest.skip("split products did not converge within snap tolerance here")
    ftmw.review_edit(str(path), wid, add=[between], remove=[f"uid:{a}", f"uid:{b}"])
    merge = ftmw.review_log(str(path))[-1]
    assert merge.kind == "merge" and set(merge.targets) == {a, b}
    assert len(merge.born_uids) == 1
    assert merge.born_uids[0] in _uids(path, wid)

    state = _uids(path, wid)
    rows = [
        (e.serial, e.targets, e.seeds_mhz, e.born_uids) for e in ftmw.review_log(path)
    ]
    ftmw.review_edit(str(path), other, add=[_clear_add_freq(path, other)], frame="raw")
    ftmw.review_undo(str(path), [ftmw.review_log(str(path))[-1].serial])
    assert _uids(path, wid) == state
    assert [
        (e.serial, e.targets, e.seeds_mhz, e.born_uids) for e in ftmw.review_log(path)
    ] == rows


def test_a_later_action_resolves_against_the_newborns_seed(stage5_multi_file):
    """D9: in one request, an action after an add resolves against the
    newborn at its seed, not at a position only the in-request refit
    produced."""
    path = stage5_multi_file
    (wid,) = _windows_with_peaks(path, 1)
    clear = _clear_add_freq(path, wid)
    tol = _snap_tol(path)

    ftmw.review_apply(
        str(path),
        actions=[
            {"action": "add", "window_id": wid, "freq_mhz": clear},
            {"action": "accept", "window_id": wid},
            {"action": "remove", "window_id": wid, "freq_mhz": clear + 0.5 * tol},
        ],
        frame="raw",
    )
    add, accept, remove = ftmw.review_log(str(path))
    assert (add.kind, accept.kind, remove.kind) == ("add", "accept", "remove")
    assert remove.targets == add.born_uids
    assert remove.frequency_mhz == add.seeds_mhz[0]
    assert add.born_uids[0] not in _uids(path, wid)


# ---------------------------------------------------------------------------
# Refusals at recording, before any fit
# ---------------------------------------------------------------------------


def test_a_birth_on_a_held_uid_is_refused_not_nudged(stage5_multi_file, forbid_fits):
    path = stage5_multi_file
    (wid,) = _windows_with_peaks(path, 1)
    clear = _clear_add_freq(path, wid)
    before = content_digest(path)
    forbid_fits()
    with pytest.raises(CurationConflictError) as exc:
        ftmw.review_edit(str(path), wid, add=[clear, clear], frame="raw")
    assert exc.value.reason == "line_already_fitted"
    assert content_digest(path) == before


def test_a_target_the_window_holds_twice_is_ambiguous(stage5_multi_file, forbid_fits):
    path = stage5_multi_file
    (wid,) = _windows_with_peaks(path, 1, min_peaks=2)
    p0, p1 = sorted(_fits(path)[wid].fitted_peaks, key=lambda p: p.frequency_mhz)[:2]
    with h5py.File(str(path), "a") as h5f:
        peaks = h5f["stage5_fitting"]["peaks"]
        uid_col = peaks["peak_uid"][...]
        uid_col[uid_col == int(p1.peak_uid)] = int(p0.peak_uid)
        peaks["peak_uid"][...] = uid_col
    before = content_digest(path)
    forbid_fits()
    for remove in ([f"uid:{p0.peak_uid}"], [float(p1.frequency_mhz)]):
        with pytest.raises(CurationConflictError) as exc:
            ftmw.review_edit(str(path), wid, remove=remove, frame="raw")
        assert (exc.value.reason, exc.value.ids) == (
            "ambiguous_peak",
            [int(p0.peak_uid)],
        )
    assert content_digest(path) == before


def test_a_batch_refusal_comes_before_any_fit(stage5_multi_file, forbid_fits):
    """A curation batch whose last action is refused fits nothing: the whole
    batch is resolved before the first fit."""
    path = stage5_multi_file
    wa, wb = _windows_with_peaks(path, 2)
    before = content_digest(path)
    forbid_fits()
    with pytest.raises(NotFoundValueError):
        ftmw.review_apply(
            str(path),
            actions=[
                {
                    "action": "add",
                    "window_id": wa,
                    "freq_mhz": _clear_add_freq(path, wa),
                },
                {
                    "action": "remove",
                    "window_id": wb,
                    "freq_mhz": _clear_add_freq(path, wb),
                },
            ],
            frame="raw",
        )
    assert content_digest(path) == before


# ---------------------------------------------------------------------------
# Undo: orphans, re-births and gone uids
# ---------------------------------------------------------------------------


def test_undoing_a_birth_a_kept_row_removes_is_orphans_peak(
    stage5_multi_file, forbid_fits
):
    path = stage5_multi_file
    (wid,) = _windows_with_peaks(path, 1)
    ftmw.review_edit(str(path), wid, add=[_clear_add_freq(path, wid)], frame="raw")
    (add,) = ftmw.review_log(str(path))
    ftmw.review_edit(str(path), wid, remove=[f"uid:{add.born_uids[0]}"])
    remove = ftmw.review_log(str(path))[-1]
    before = content_digest(path)
    forbid_fits()
    for dry_run in (True, False):
        with pytest.raises(CurationConflictError) as exc:
            ftmw.review_undo(str(path), [add.serial], dry_run=dry_run)
        assert (exc.value.reason, exc.value.ids) == ("orphans_peak", [remove.serial])
    assert content_digest(path) == before


def test_orphans_peak_names_every_row_downstream(stage5_multi_file):
    """Transitive: a merge of a split's children, then a remove of the
    merge's child -- undoing the split orphans both."""
    path = stage5_multi_file
    wid, parent, near = _split_site(path)
    ftmw.review_edit(str(path), wid, add=[near], frame="raw")
    (split,) = ftmw.review_log(str(path))
    a, b = split.born_uids
    kids = {
        int(p.peak_uid): float(p.frequency_mhz) for p in _fits(path)[wid].fitted_peaks
    }
    if abs(kids[a] - kids[b]) > _snap_tol(path):
        pytest.skip("split products did not converge within snap tolerance here")
    ftmw.review_edit(
        str(path), wid, add=[0.5 * (kids[a] + kids[b])], remove=[f"uid:{a}", f"uid:{b}"]
    )
    merge = ftmw.review_log(str(path))[-1]
    ftmw.review_edit(str(path), wid, remove=[f"uid:{merge.born_uids[0]}"])
    last = ftmw.review_log(str(path))[-1]
    with pytest.raises(CurationConflictError) as exc:
        ftmw.review_undo(str(path), [split.serial])
    assert (exc.value.reason, exc.value.ids) == (
        "orphans_peak",
        [merge.serial, last.serial],
    )
    ftmw.review_undo(str(path), [split.serial, merge.serial, last.serial])
    assert ftmw.review_log(str(path)) == []


def test_a_peak_re_added_at_the_same_seed_is_born_under_the_same_uid(
    stage5_multi_file, forbid_fits
):
    """Add, remove, add again at the same frequency: the second birth takes
    the same uid (the uid is the seed's). Undoing the remove alone would make
    the first and second births one uid twice over: the replay refuses it
    (``replay_diverged``) before any fit; undoing the second add with it is
    fine."""
    path = stage5_multi_file
    (wid,) = _windows_with_peaks(path, 1)
    clear = _clear_add_freq(path, wid)
    ftmw.review_edit(str(path), wid, add=[clear], frame="raw")
    first = ftmw.review_log(str(path))[-1]
    ftmw.review_edit(str(path), wid, remove=[f"uid:{first.born_uids[0]}"])
    remove = ftmw.review_log(str(path))[-1]
    ftmw.review_edit(str(path), wid, add=[clear], frame="raw")
    second = ftmw.review_log(str(path))[-1]
    assert second.born_uids == first.born_uids
    assert second.born_uids[0] in _uids(path, wid)

    before = content_digest(path)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(s6, "refit_window_core", _no_fit)
        with pytest.raises(CurationConflictError) as exc:
            ftmw.review_undo(str(path), [remove.serial])
    assert (exc.value.reason, exc.value.ids) == ("replay_diverged", [second.serial])
    assert content_digest(path) == before

    ftmw.review_undo(str(path), [remove.serial, second.serial])
    assert first.born_uids[0] in _uids(path, wid)


def _no_fit(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("a refused write fitted a window")


def test_a_joint_edit_that_re_births_its_own_target_replays(stage5_multi_file):
    """One edit removes a peak and re-adds a line at its seed, so the birth
    takes the removed peak's uid. A joint refit removes before it births, and
    so does the symbolic pass over a replay: an unrelated undo replays it,
    and undoing the first birth orphans the edit's remove (bound to that
    birth, not to the edit's own add) before any fit."""
    path = stage5_multi_file
    wa, wb = _windows_with_peaks(path, 2)
    ftmw.review_edit(str(path), wa, add=[_clear_add_freq(path, wa)], frame="raw")
    (first,) = ftmw.review_log(str(path))
    (uid,) = first.born_uids
    ftmw.review_edit(
        str(path), wa, add=[first.seeds_mhz[0]], remove=[f"uid:{uid}"], frame="raw"
    )
    rows = {e.kind: e for e in ftmw.review_log(str(path))[1:]}
    if rows["add"].born_uids != (uid,):
        pytest.skip("the re-add's seed did not mint the removed peak's uid here")
    assert rows["remove"].targets == (uid,)
    ftmw.review_accept(str(path), wb)
    accept = ftmw.review_log(str(path))[-1]

    ftmw.review_undo(str(path), [accept.serial])
    assert uid in _uids(path, wa)

    before = content_digest(path)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(s6, "refit_window_core", _no_fit)
        with pytest.raises(CurationConflictError) as exc:
            ftmw.review_undo(str(path), [first.serial])
    assert (exc.value.reason, exc.value.ids) == (
        "orphans_peak",
        [rows["remove"].serial],
    )
    assert content_digest(path) == before

    ftmw.review_undo(str(path), [first.serial, rows["remove"].serial])
    assert uid in _uids(path, wa)
    assert [e.serial for e in ftmw.review_log(str(path))] == [rows["add"].serial]


def test_a_uid_gone_after_an_undo_is_not_found(stage5_multi_file):
    """Undo an add: its peak is gone, so a later request naming its uid is
    refused (``not_found``), and the undo itself recorded nothing that names
    it."""
    path = stage5_multi_file
    wa, wb = _windows_with_peaks(path, 2)
    ftmw.review_edit(str(path), wa, add=[_clear_add_freq(path, wa)], frame="raw")
    (add,) = ftmw.review_log(str(path))
    ftmw.review_accept(str(path), wb)
    ftmw.review_undo(str(path), [add.serial])
    assert add.born_uids[0] not in _uids(path, wa)
    with pytest.raises(NotFoundValueError) as exc:
        ftmw.review_edit(str(path), wa, remove=[f"uid:{add.born_uids[0]}"])
    assert exc.value.ids == [add.born_uids[0]]


def test_a_forged_born_uid_collision_is_replay_diverged(stage5_multi_file, forbid_fits):
    """Only the engine writes the log; a row whose recorded birth collides
    with a held uid is refused on replay, before any fit."""
    path = stage5_multi_file
    wa, wb = _windows_with_peaks(path, 2)
    held = _uids(path, wa)[0]
    ftmw.review_edit(str(path), wa, add=[_clear_add_freq(path, wa)], frame="raw")
    ftmw.review_accept(str(path), wb)
    with h5py.File(str(path), "a") as h5f:
        grp = h5f["stage6_review"]["decision_log"]
        rows = json.loads(str(grp.attrs["data"]))
        rows[0]["born_uids"] = [held]
        grp.attrs["data"] = json.dumps(rows)
    before = content_digest(path)
    forbid_fits()
    with pytest.raises(CurationConflictError) as exc:
        ftmw.review_undo(str(path), [1])
    assert (exc.value.reason, exc.value.ids) == ("replay_diverged", [0])
    assert content_digest(path) == before


# ---------------------------------------------------------------------------
# Replay: rows applied as recorded, every refusal before any fit
# ---------------------------------------------------------------------------


def _rewrite_rows(path: Path, fn: Any) -> None:
    """Rewrite the persisted decision log's rows in place, to forge a row the
    engine would not have written."""
    with h5py.File(str(path), "a") as h5f:
        grp = h5f["stage6_review"]["decision_log"]
        rows = json.loads(str(grp.attrs["data"]))
        fn(rows)
        grp.attrs["data"] = json.dumps(rows)


def _unavailable(
    monkeypatch: pytest.MonkeyPatch,
    unavailable: Tuple[int, ...],
    edges: Tuple[Tuple[int, int], ...] = (),
) -> None:
    """Mark *unavailable* windows as ones whose fitted geometry the file does
    not hold, and give the cascade the ``(source, dependent)`` *edges*."""
    import dataclasses

    real = s6._build_shared_fit_ctx

    def build(*args: Any, **kwargs: Any) -> Any:
        shared = real(*args, **kwargs)
        sources = dict(shared.base_cascade_sources)
        for src, dep in edges:
            sources[dep] = tuple(sources.get(dep, ())) + (src,)
        return dataclasses.replace(
            shared,
            base_cascade_sources=sources,
            unavailable_window_ids=frozenset(unavailable),
        )

    monkeypatch.setattr(s6, "_build_shared_fit_ctx", build)


def test_a_replay_acts_on_the_recorded_peaks_not_the_display_frequency(
    stage5_multi_file,
):
    """``frequency_mhz`` is display only: a replay removes the recorded
    target and births the recorded seed whatever frequency the row shows."""
    path = stage5_multi_file
    wa, wb = _windows_with_peaks(path, 2, min_peaks=2)
    victim = min(_fits(path)[wa].fitted_peaks, key=lambda p: float(p.frequency_mhz))
    ftmw.review_edit(
        str(path),
        wa,
        add=[_clear_add_freq(path, wa)],
        remove=[float(victim.frequency_mhz)],
        frame="raw",
    )
    ftmw.review_accept(str(path), wb)
    state = _uids(path, wa)
    kept = [
        (e.serial, e.targets, e.seeds_mhz, e.born_uids) for e in ftmw.review_log(path)
    ]

    def display_elsewhere(rows: List[Dict[str, Any]]) -> None:
        others = [
            float(p.frequency_mhz)
            for p in _fits(path)[wa].fitted_peaks
            if int(p.peak_uid) != int(victim.peak_uid)
        ]
        for row in rows[:2]:
            row["frequency_mhz"] = others[0]

    _rewrite_rows(path, display_elsewhere)
    ftmw.review_undo(str(path), [ftmw.review_log(str(path))[-1].serial])
    assert _uids(path, wa) == state
    assert int(victim.peak_uid) not in state
    assert [
        (e.serial, e.targets, e.seeds_mhz, e.born_uids)
        for e in ftmw.review_log(path)[:2]
    ] == kept[:2]


def test_a_forged_seed_off_its_window_is_refused_on_replay(
    stage5_multi_file, forbid_fits
):
    path = stage5_multi_file
    wa, wb = _windows_with_peaks(path, 2)
    ftmw.review_edit(str(path), wa, add=[_clear_add_freq(path, wa)], frame="raw")
    ftmw.review_accept(str(path), wb)
    other = _fits(path)[wb].window
    assert other is not None
    off = max(float(v) for v in other.freq_range) + 1.0e3
    _rewrite_rows(path, lambda rows: rows[0].update(seeds_mhz=[off]))
    before = content_digest(path)
    forbid_fits()
    with pytest.raises(CurationConflictError) as exc:
        ftmw.review_undo(str(path), [1])
    assert (exc.value.reason, exc.value.ids) == ("target_outside_window", [wa])
    assert content_digest(path) == before


def test_a_replayed_row_on_an_unavailable_window_is_refused(
    stage5_multi_file, forbid_fits, monkeypatch
):
    path = stage5_multi_file
    wa, wb = _windows_with_peaks(path, 2)
    ftmw.review_edit(str(path), wa, add=[_clear_add_freq(path, wa)], frame="raw")
    ftmw.review_accept(str(path), wb)
    before = content_digest(path)
    _unavailable(monkeypatch, (wa,))
    forbid_fits()
    with pytest.raises(CurationConflictError) as exc:
        ftmw.review_undo(str(path), [1])
    assert (exc.value.reason, exc.value.ids) == ("fit_plan_unavailable", [wa])
    assert content_digest(path) == before


def test_a_replayed_cascade_into_an_unavailable_window_is_refused(
    stage5_multi_file, forbid_fits, monkeypatch
):
    path = stage5_multi_file
    wa, wb = _windows_with_peaks(path, 2)
    ftmw.review_edit(str(path), wa, add=[_clear_add_freq(path, wa)], frame="raw")
    ftmw.review_accept(str(path), wb)
    before = content_digest(path)
    _unavailable(monkeypatch, (wb,), edges=((wa, wb),))
    forbid_fits()
    with pytest.raises(CurationConflictError) as exc:
        ftmw.review_undo(str(path), [1])
    assert (exc.value.reason, exc.value.ids) == ("fit_plan_unavailable", [wb])
    assert content_digest(path) == before


def test_a_forged_target_held_twice_is_ambiguous_on_replay(
    stage5_multi_file, forbid_fits, monkeypatch
):
    """A row whose target the automatic fit holds twice cannot say which
    peak it removes: ``ambiguous_peak``, as at recording."""
    import dataclasses

    path = stage5_multi_file
    wa, wb = _windows_with_peaks(path, 2, min_peaks=2)
    p0, p1 = sorted(_fits(path)[wa].fitted_peaks, key=lambda p: p.frequency_mhz)[:2]
    ftmw.review_edit(str(path), wa, remove=[f"uid:{p0.peak_uid}"])
    ftmw.review_accept(str(path), wb)
    real = s6._build_shared_fit_ctx

    def build(*args: Any, **kwargs: Any) -> Any:
        shared = real(*args, **kwargs)
        fits = dict(shared.baseline_fits)
        twin = copy.copy(fits[wa])
        twin.fitted_peaks = []
        for p in fits[wa].fitted_peaks:
            p = copy.copy(p)
            if int(p.peak_uid) == int(p1.peak_uid):
                p.peak_uid = p0.peak_uid
            twin.fitted_peaks.append(p)
        fits[wa] = twin
        return dataclasses.replace(shared, baseline_fits=fits)

    monkeypatch.setattr(s6, "_build_shared_fit_ctx", build)
    before = content_digest(path)
    forbid_fits()
    with pytest.raises(CurationConflictError) as exc:
        ftmw.review_undo(str(path), [1])
    assert (exc.value.reason, exc.value.ids) == ("ambiguous_peak", [int(p0.peak_uid)])
    assert content_digest(path) == before


def test_an_edit_of_an_unavailable_window_is_refused_before_any_fit(
    stage5_multi_file, forbid_fits, monkeypatch
):
    path = stage5_multi_file
    (wid,) = _windows_with_peaks(path, 1)
    clear = _clear_add_freq(path, wid)
    before = content_digest(path)
    _unavailable(monkeypatch, (wid,))
    forbid_fits()
    with pytest.raises(CurationConflictError) as exc:
        ftmw.review_edit(str(path), wid, add=[clear], frame="raw")
    assert (exc.value.reason, exc.value.ids) == ("fit_plan_unavailable", [wid])
    assert content_digest(path) == before
