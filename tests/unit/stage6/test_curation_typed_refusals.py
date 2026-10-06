"""Curation refusals a program can route on are typed (CONTRACT_STRATEGY, Errors).

The promises, each checked through the public surfaces (functional API,
``Pipeline``, the CLI under ``--json``) on a real multi-window Stage 5 fit:

* a frequency that matches no fitted peak is ``not_found`` (``kind="peak"``) and
  lists **every** such frequency of the request, not the first;
* a target no live window covers is ``not_found`` (``kind="window"``, ``ids`` the
  uncovered frequencies);
* ``review_undo`` ids the decision log does not hold are ``not_found``
  (``kind="decision"``), every one of them, on an empty log too;
* a window id a batch names that no create can install is ``not_found`` even
  when the batch holds an unpinned ``create`` -- all of them, at once;
* a valid request that conflicts with the file's review state is
  ``curation_conflict`` carrying a stable ``reason`` and the ``ids`` involved;
* inside a batch the refusal keeps its type and gains the action's tag;
* every refusal is still the ``ValueError`` it replaced and leaves the file
  exactly as it was.

Mutation: reverting a site to a bare ``ValueError`` fails the type check;
reporting only the first miss fails the ``ids`` check; a wrong ``reason`` slug or
``ids`` fails the conflict checks.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import h5py
import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import CurationAction
from ftmwpipeline._internal import stage6_impl as s6
from ftmwpipeline._internal.stage4_impl import load_windows_impl
from ftmwpipeline._internal.stage6_impl import (
    create_window_impl,
    merge_peaks_impl,
    refit_window_impl,
    review_log_impl,
    split_peak_impl,
)
from ftmwpipeline.cli.main import main
from ftmwpipeline.core.data_structures import FittedPeak, FittingResult
from ftmwpipeline.file_manager import (
    CurationConflictError,
    NotFoundError,
    PipelineFileError,
)
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5
from ftmwpipeline.pipeline import Pipeline
from tests._events_support import content_digest

pytestmark = [
    pytest.mark.integration,
    # Design G1: every write here persists the reference replay of its log.
    pytest.mark.usefixtures("every_write_is_reference"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_fit(path: Path):
    with h5py.File(str(path), "r") as h5f:
        return load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])


def _fitted_by_window(path: Path) -> Dict[int, List[float]]:
    return {
        int(wf.window_id): sorted(float(p.frequency_mhz) for p in wf.fitted_peaks)
        for wf in _load_fit(path).window_fits
        if wf.window_id is not None
    }


def _fitted_wf(path: Path, wid: int) -> FittingResult:
    return next(wf for wf in _load_fit(path).window_fits if wf.window_id == wid)


def _first_peak(path: Path) -> Tuple[int, FittedPeak]:
    for wf in _load_fit(path).window_fits:
        if wf.window_id is None:
            continue
        for p in wf.fitted_peaks:
            if p.peak_uid is not None:
                return int(wf.window_id), p
    raise AssertionError("fixture has no fitted peak with a uid")


def _two_window_ids(path: Path) -> Tuple[int, int]:
    wids = sorted(w for w, freqs in _fitted_by_window(path).items() if freqs)
    if len(wids) < 2:
        pytest.skip("Need at least two fitted windows")
    return wids[0], wids[1]


def _plan_ranges(path: Path) -> Dict[int, Tuple[float, float]]:
    plan = load_windows_impl(str(path))["plan"]
    return {
        int(w.window_id): (min(w.freq_range), max(w.freq_range)) for w in plan.windows
    }


def _live_ranges(path: Path) -> List[Tuple[int, float, float]]:
    """``(window_id, lo, hi)`` of every window carrying a Stage 5 fit."""
    ranges = _plan_ranges(path)
    return sorted(
        ((w, *ranges[w]) for w in _fitted_by_window(path) if w in ranges),
        key=lambda t: t[1],
    )


def _far_from_peaks(path: Path, wid: int) -> float:
    """The point of window ``wid``'s own range farthest from its fitted peaks:
    covered by the window, but a miss for any snap-bounded peak match."""
    lo, hi = _plan_ranges(path)[wid]
    peaks = np.asarray(
        [float(p.frequency_mhz) for p in _fitted_wf(path, wid).fitted_peaks]
    )
    grid = np.linspace(lo, hi, 512)[1:-1]
    dist = np.min(np.abs(grid[:, None] - peaks[None, :]), axis=1)
    return float(grid[int(np.argmax(dist))])


def _uncovered_freq(path: Path) -> float:
    return min(lo for lo, _ in _plan_ranges(path).values()) - 1000.0


def _free_anchor(path: Path) -> float:
    """A frequency clear of every fitted window, inside the analysis trim."""
    return _live_ranges(path)[-1][2] + 5.0


def _bin_width_mhz(path: Path) -> float:
    plan = load_windows_impl(str(path))["plan"]
    w = next(
        w
        for w in plan.windows
        if "grid_span" in w.diagnostics
        and w.diagnostics["grid_span"][1] > w.diagnostics["grid_span"][0]
    )
    lo, hi = w.freq_range
    span = w.diagnostics["grid_span"]
    return abs(hi - lo) / (int(span[1]) - int(span[0]))


def _write(tmp_path: Path, text: str, name: str = "cur.csv") -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


def _assert_not_found(
    exc: pytest.ExceptionInfo, kind: str, ids: Sequence[Any], *, approx: bool = False
) -> NotFoundError:
    err = exc.value
    assert isinstance(err, NotFoundError)
    assert isinstance(err, (ValueError, KeyError, PipelineFileError))
    assert err.kind == kind
    if approx:
        assert err.ids == pytest.approx(list(ids))
    else:
        assert err.ids == list(ids)
    d = err.to_dict()
    assert d["code"] == "not_found" and d["kind"] == kind
    return err


def _assert_conflict(
    exc: pytest.ExceptionInfo, reason: str, ids: Sequence[int]
) -> CurationConflictError:
    err = exc.value
    assert isinstance(err, CurationConflictError)
    assert isinstance(err, (ValueError, PipelineFileError))
    assert err.reason == reason
    assert err.ids == list(ids)
    d = err.to_dict()
    assert d["code"] == "curation_conflict"
    assert d["reason"] == reason and d["ids"] == list(ids)
    json.dumps(d, allow_nan=False)
    return err


def _cli_error(capsys, *argv: Any) -> Tuple[int, Dict[str, Any]]:
    """Run the CLI under ``--json``; return (exit code, the error dict)."""
    capsys.readouterr()
    rc = main([str(a) for a in argv] + ["--json"])
    err = capsys.readouterr().err
    # the error dict is the last line of stderr (any log lines precede it)
    last = [ln for ln in err.splitlines() if ln.strip()][-1]
    return rc, json.loads(last)


# ---------------------------------------------------------------------------
# A frequency that matches no fitted peak: not_found, kind "peak", every one
# ---------------------------------------------------------------------------


@pytest.fixture
def two_misses(stage5_multi_file) -> Tuple[Path, int, List[float]]:
    f = stage5_multi_file
    wid, _ = _first_peak(f)
    far = _far_from_peaks(f, wid)
    return f, wid, [far, far + 0.001]


@pytest.mark.parametrize("via", ["api", "pipeline"])
@pytest.mark.parametrize("named", [True, False])
def test_refit_remove_lists_every_unmatched_frequency(two_misses, via, named):
    f, wid, misses = two_misses
    before = content_digest(f)
    window = wid if named else None
    with pytest.raises(ValueError) as exc:
        if via == "api":
            ftmw.review_edit(f, window, remove=misses)
        else:
            Pipeline.open(f).review_edit(window, remove=misses)
    err = _assert_not_found(exc, "peak", misses, approx=True)
    # the historical per-frequency messages, joined
    assert str(err).count("no fitted peak within") == 2
    assert content_digest(f) == before


def test_one_hit_one_miss_reports_only_the_miss(stage5_multi_file):
    f = stage5_multi_file
    wid, peak = _first_peak(f)
    miss = _far_from_peaks(f, wid)
    with pytest.raises(ValueError) as exc:
        ftmw.review_edit(f, wid, remove=[float(peak.frequency_mhz), miss])
    _assert_not_found(exc, "peak", [miss], approx=True)


def test_a_batch_keeps_the_type_and_tags_the_action(two_misses):
    f, wid, misses = two_misses
    before = content_digest(f)
    actions = [
        CurationAction("remove", window_id=wid, freq_mhz=m, frame="raw") for m in misses
    ]
    for call in (ftmw.review_apply, ftmw.review_preview):
        with pytest.raises(ValueError) as exc:
            call(f, actions=actions)
        err = _assert_not_found(exc, "peak", misses, approx=True)
        # The two rows coalesce into one edit: both are named, by request
        # position.
        assert str(err).startswith("curation actions 1, 2 (")
        assert err.action_indices == [0, 1]
        assert "failed:" in str(err)
    assert content_digest(f) == before


def test_merge_lists_every_unmatched_frequency(two_misses):
    f, wid, misses = two_misses
    before = content_digest(f)
    with pytest.raises(ValueError) as exc:
        merge_peaks_impl(str(f), wid, misses)
    err = _assert_not_found(exc, "peak", misses, approx=True)
    assert str(err).count("merge: no fitted peak within") == 2
    assert content_digest(f) == before


def test_split_reports_its_one_peak(two_misses):
    f, wid, misses = two_misses
    before = content_digest(f)
    with pytest.raises(ValueError) as exc:
        split_peak_impl(str(f), wid, misses[0])
    _assert_not_found(exc, "peak", [misses[0]], approx=True)
    assert content_digest(f) == before


def test_peak_miss_through_the_cli(two_misses, capsys):
    f, wid, misses = two_misses
    rc, payload = _cli_error(
        capsys,
        "review",
        "edit",
        f,
        "--window",
        wid,
        "--remove",
        misses[0],
        "--remove",
        misses[1],
    )
    assert rc == 1
    assert payload["code"] == "not_found" and payload["kind"] == "peak"
    assert payload["ids"] == pytest.approx(misses)


# ---------------------------------------------------------------------------
# A target no live window covers: not_found, kind "window", ids the frequencies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("via", ["api", "pipeline"])
def test_review_edit_uncovered_remove_targets(stage5_multi_file, via):
    f = stage5_multi_file
    unc = _uncovered_freq(f)
    targets = [unc, unc - 1.0]
    before = content_digest(f)
    with pytest.raises(ValueError) as exc:
        if via == "api":
            ftmw.review_edit(f, None, remove=targets)
        else:
            Pipeline.open(f).review_edit(None, remove=targets)
    err = _assert_not_found(exc, "window", targets, approx=True)
    assert "not covered by any live window" in str(err)
    assert content_digest(f) == before


def test_a_covered_target_is_not_reported_alongside_the_uncovered(stage5_multi_file):
    f = stage5_multi_file
    wid, peak = _first_peak(f)
    unc = _uncovered_freq(f)
    with pytest.raises(ValueError) as exc:
        ftmw.review_edit(f, None, remove=[float(peak.frequency_mhz), unc])
    _assert_not_found(exc, "window", [unc], approx=True)


@pytest.mark.parametrize("call_name", ["review_apply", "review_preview"])
def test_curation_file_uncovered_targets(stage5_multi_file, tmp_path, call_name):
    f = stage5_multi_file
    unc = _uncovered_freq(f)
    cur = _write(tmp_path, f"remove,,{unc!r},\nremove,,{unc - 1.0!r},\n")
    before = content_digest(f)
    with pytest.raises(ValueError) as exc:
        getattr(ftmw, call_name)(f, cur)
    err = _assert_not_found(exc, "window", [unc, unc - 1.0], approx=True)
    assert "curation line 1:" in str(err) and "curation line 2:" in str(err)
    assert content_digest(f) == before


def test_curation_file_uncovered_targets_are_listed_once(stage5_multi_file, tmp_path):
    f = stage5_multi_file
    unc = _uncovered_freq(f)
    cur = _write(tmp_path, f"remove,,{unc!r},\nremove,,{unc!r},\n")
    with pytest.raises(ValueError) as exc:
        ftmw.review_apply(f, cur)
    _assert_not_found(exc, "window", [unc], approx=True)


# ---------------------------------------------------------------------------
# curation_conflict: a valid request against the file's review state
# ---------------------------------------------------------------------------


def _assert_line_already_fitted(
    exc: pytest.ExceptionInfo, peak: FittedPeak
) -> CurationConflictError:
    """``ids`` is the one uid the refused add collides on: the uid the message
    names, which is the fitted line's birth identity as the refit seeds it. The
    seed identity of a line can differ from its stored ``peak_uid`` by a few
    units (a stored uid is nudged to the nearest free value when two seeds
    land on one), so it is held to the message and to the stored uid's
    neighbourhood rather than to equality with the stored value."""
    err = exc.value
    assert isinstance(err, CurationConflictError)
    assert err.reason == "line_already_fitted"
    assert len(err.ids) == 1 and isinstance(err.ids[0], int)
    named = re.search(r"peak_uid=(\d+)", str(err))
    assert named is not None and err.ids == [int(named.group(1))]
    assert abs(err.ids[0] - int(peak.peak_uid)) <= 100
    d = err.to_dict()
    assert d["code"] == "curation_conflict" and d["reason"] == "line_already_fitted"
    return err


@pytest.mark.parametrize("via", ["api", "pipeline"])
def test_targets_span_windows(stage5_multi_file, via):
    f = stage5_multi_file
    wa, wb = _two_window_ids(f)
    remove = _fitted_by_window(f)[wa][0]
    add = _far_from_peaks(f, wb)
    before = content_digest(f)
    with pytest.raises(ValueError) as exc:
        if via == "api":
            ftmw.review_edit(f, None, add=[add], remove=[remove])
        else:
            Pipeline.open(f).review_edit(None, add=[add], remove=[remove])
    err = _assert_conflict(exc, "targets_span_windows", sorted([wa, wb]))
    assert "resolve to different windows" in str(err)
    assert content_digest(f) == before


def test_line_already_fitted(stage5_multi_file):
    f = stage5_multi_file
    wid, peak = _first_peak(f)
    before = content_digest(f)
    with pytest.raises(ValueError) as exc:
        ftmw.review_edit(f, wid, add=[float(peak.frequency_mhz)])
    err = _assert_line_already_fitted(exc, peak)
    assert "birth position" in str(err)
    assert content_digest(f) == before


def test_line_already_fitted_inside_a_batch_keeps_its_type(stage5_multi_file):
    f = stage5_multi_file
    wid, peak = _first_peak(f)
    before = content_digest(f)
    action = CurationAction(
        "add", window_id=wid, freq_mhz=float(peak.frequency_mhz), frame="raw"
    )
    with pytest.raises(ValueError) as exc:
        ftmw.review_apply(f, actions=[action])
    err = _assert_line_already_fitted(exc, peak)
    assert str(err).startswith("curation action 1 (")
    assert content_digest(f) == before


def test_replay_conflict_when_the_pinned_create_would_widen(
    stage5_multi_file, tmp_path
):
    f = stage5_multi_file
    step = _bin_width_mhz(f)
    wid, _lo, hi = _live_ranges(f)[-1]
    fresh = max(_plan_ranges(f)) + 1
    cur = _write(tmp_path, f"create,{fresh},{hi + 2 * step:.6f},\n")
    before = content_digest(f)
    with pytest.raises(ValueError) as exc:
        ftmw.review_apply(f, cur)
    err = _assert_conflict(exc, "replay_conflict", [fresh, wid])
    assert "widens window" in str(err)
    assert str(err).startswith("curation action 1 (")
    assert content_digest(f) == before


def test_replay_conflict_when_the_pinned_id_is_taken(stage5_multi_file, tmp_path):
    f = stage5_multi_file
    taken = _live_ranges(f)[0][0]
    cur = _write(tmp_path, f"create,{taken},{_free_anchor(f):.6f},\n")
    before = content_digest(f)
    with pytest.raises(ValueError) as exc:
        ftmw.review_apply(f, cur)
    err = _assert_conflict(exc, "replay_conflict", [taken])
    assert "already in use" in str(err)
    assert content_digest(f) == before


def test_widening_an_unfitted_window_is_an_untyped_guard(
    stage5_multi_file, monkeypatch
):
    """``plan_stage6_window`` only proposes widening a *live* window, so this
    guard is unreachable on real geometry: an invariant, not a route, so it
    stays the built-in ``ValueError`` (no ``curation_conflict`` reason). It is
    forced through the planner seam, as the implied-create reinterpretation
    guard is."""
    f = stage5_multi_file
    live = {w for w, _, _ in _live_ranges(f)}
    unfitted = sorted(set(_plan_ranges(f)) - live)
    if not unfitted:
        pytest.skip("every planned window carries a fit")
    wid, _lo, hi = _live_ranges(f)[-1]
    anchor = hi + 2 * _bin_width_mhz(f)

    orig = s6._plan_create

    def forced(shared, created_windows, anchor_mhz, *, replay_window_id, **kw):
        # The request plans it and the replay replans it: both see the forced
        # window, as the replay of the recorded create row would.
        proposal = orig(
            shared, created_windows, anchor_mhz, replay_window_id=None, **kw
        )
        assert proposal.mode == "widened"
        proposal.window.window_id = unfitted[0]
        return proposal

    monkeypatch.setattr(s6, "_plan_create", forced)
    before = content_digest(f)
    with pytest.raises(ValueError) as exc:
        create_window_impl(str(f), anchor)
    assert not isinstance(exc.value, PipelineFileError)
    assert "no Stage 5 fit to widen" in str(exc.value)
    assert content_digest(f) == before


def test_implied_create_reinterpreted_is_an_untyped_guard(
    stage5_multi_file, monkeypatch
):
    """The reinterpretation is not reachable on real data (it needs the anchor
    within snap tolerance of a peak in the very window it just created), so
    the guard stays the built-in ``ValueError``. It is forced through the
    resolution seam, as ``test_implied_create`` does."""
    f = stage5_multi_file
    anchor = _free_anchor(f)
    orig = s6._resolve_edit_steps

    def reinterpreting(ctx, state, window_id, add, remove, **kwargs):
        steps = orig(ctx, state, window_id, add, remove, **kwargs)
        # Stand in for inference reading the add as a split of a peak in the
        # window it landed in, which is what the refusal exists to catch.
        steps[0].kind = "split"
        steps[0].rows[0].kind = "split"
        return steps

    monkeypatch.setattr(s6, "_resolve_edit_steps", reinterpreting)
    before = content_digest(f)
    with pytest.raises(ValueError) as exc:
        refit_window_impl(str(f), None, add=[anchor])
    assert not isinstance(exc.value, PipelineFileError)
    assert "implies creating a window" in str(exc.value)
    assert content_digest(f) == before


@pytest.mark.parametrize("via", ["api", "pipeline"])
def test_target_outside_window(stage5_multi_file, via):
    """An add whose seed falls outside the window it names: the request is
    valid, the window is the wrong one."""
    f = stage5_multi_file
    wa, wb = _two_window_ids(f)
    add = _far_from_peaks(f, wb)  # covered by wb, nowhere near wa
    before = content_digest(f)
    with pytest.raises(ValueError) as exc:
        if via == "api":
            ftmw.review_edit(f, wa, add=[add])
        else:
            Pipeline.open(f).review_edit(wa, add=[add])
    err = _assert_conflict(exc, "target_outside_window", [wa])
    assert f"outside window {wa}'s range" in str(err)
    # The remedy names the implied create (an add with no window named), not
    # only 'review create'.
    assert "name no window" in str(err)
    assert "creates one there if none does" in str(err)
    assert content_digest(f) == before


def test_target_outside_window_inside_a_batch(stage5_multi_file, tmp_path):
    f = stage5_multi_file
    wa, wb = _two_window_ids(f)
    add = _far_from_peaks(f, wb)
    before = content_digest(f)
    actions = [CurationAction("add", window_id=wa, freq_mhz=add, frame="raw")]
    cur = _write(tmp_path, f"add,{wa},{add!r},\n")
    for call, source in (
        (ftmw.review_apply, {"actions": actions}),
        (ftmw.review_preview, {"actions": actions}),
        (ftmw.review_apply, {"curation_path": cur}),
    ):
        with pytest.raises(ValueError) as exc:
            call(f, **source)
        err = _assert_conflict(exc, "target_outside_window", [wa])
        assert str(err).startswith("curation action 1 (")
    assert content_digest(f) == before


def test_target_outside_window_through_the_cli(stage5_multi_file, capsys):
    f = stage5_multi_file
    wa, wb = _two_window_ids(f)
    rc, payload = _cli_error(
        capsys, "review", "edit", f, "--window", wa, "--add", _far_from_peaks(f, wb)
    )
    assert rc == 1
    assert payload["code"] == "curation_conflict"
    assert payload["reason"] == "target_outside_window"
    assert payload["ids"] == [wa]


@pytest.fixture
def created_then_edited(stage5_multi_file) -> Tuple[Path, int]:
    """A file whose log is ``[create_window, add]`` on a created window."""
    f = stage5_multi_file
    anchor = _free_anchor(f)
    wid = create_window_impl(str(f), anchor).window_id
    refit_window_impl(str(f), wid, add=[anchor])
    assert [e.kind for e in review_log_impl(str(f))] == ["create_window", "add"]
    return f, wid


@pytest.mark.parametrize("via", ["api", "pipeline"])
def test_undo_orphans_a_created_window(created_then_edited, via):
    f, _wid = created_then_edited
    before = content_digest(f)
    with pytest.raises(ValueError) as exc:
        if via == "api":
            ftmw.review_undo(f, [0])
        else:
            Pipeline.open(f).review_undo([0])
    # ids: the surviving decisions that act on the dropped window
    err = _assert_conflict(exc, "orphans_created_window", [1])
    assert "Undo them together" in str(err)
    assert content_digest(f) == before


def test_undoing_both_is_not_a_conflict(created_then_edited):
    f, _wid = created_then_edited
    ftmw.review_undo(f, [0, 1])
    assert review_log_impl(str(f)) == []


def test_conflict_through_the_cli(created_then_edited, capsys):
    f, _wid = created_then_edited
    rc, payload = _cli_error(capsys, "review", "undo", f, "--id", 0)
    assert rc == 1
    assert payload["schema"] == "ftmw/error@1"
    assert payload["code"] == "curation_conflict"
    assert payload["reason"] == "orphans_created_window"
    assert payload["ids"] == [1]


# ---------------------------------------------------------------------------
# review_undo: unknown ids are not_found, kind "decision"
# ---------------------------------------------------------------------------


@pytest.fixture
def one_decision(stage5_multi_file) -> Path:
    f = stage5_multi_file
    wid, peak = _first_peak(f)
    ftmw.review_edit(f, wid, remove=[f"uid:{peak.peak_uid}"])
    assert len(review_log_impl(str(f))) == 1
    return f


@pytest.mark.parametrize("via", ["api", "pipeline"])
def test_undo_lists_every_unknown_id_in_request_order(one_decision, via):
    f = one_decision
    before = content_digest(f)
    with pytest.raises(ValueError) as exc:
        if via == "api":
            ftmw.review_undo(f, [9, 0, 7, 9])
        else:
            Pipeline.open(f).review_undo([9, 0, 7, 9])
    err = _assert_not_found(exc, "decision", [9, 7])
    assert "unknown decision id(s)" in str(err)
    assert content_digest(f) == before


@pytest.mark.parametrize("via", ["api", "pipeline"])
def test_undo_on_an_empty_log_reports_every_requested_id(stage5_multi_file, via):
    f = stage5_multi_file
    with pytest.raises(ValueError) as exc:
        if via == "api":
            ftmw.review_undo(f, [5, 3])
        else:
            Pipeline.open(f).review_undo([5, 3])
    err = _assert_not_found(exc, "decision", [5, 3])
    assert "no recorded decisions to undo" in str(err)


def test_undo_with_no_ids_is_bad_setting_on_ids(stage5_multi_file, one_decision):
    from ftmwpipeline.file_manager import BadSettingError

    for f in (stage5_multi_file, one_decision):
        with pytest.raises(BadSettingError) as exc:
            ftmw.review_undo(f, [])
        assert exc.value.path == "ids"


def test_undo_unknown_id_through_the_cli(one_decision, capsys):
    rc, payload = _cli_error(
        capsys, "review", "undo", one_decision, "--id", 4, "--id", 8
    )
    assert rc == 1
    assert payload["code"] == "not_found" and payload["kind"] == "decision"
    assert payload["ids"] == [4, 8]


def test_undo_known_ids_still_work(one_decision):
    ftmw.review_undo(one_decision, [0])
    assert review_log_impl(str(one_decision)) == []


# ---------------------------------------------------------------------------
# Unknown window ids alongside an unpinned create
# ---------------------------------------------------------------------------


@pytest.fixture
def create_batch(stage5_multi_file) -> Dict[str, Any]:
    f = stage5_multi_file
    ranges = _plan_ranges(f)
    live = {w for w, _, _ in _live_ranges(f)}
    unfitted = sorted(set(ranges) - live)
    if not unfitted:
        pytest.skip("every planned window carries a fit")
    top = max(ranges)
    return {
        "path": f,
        "anchor": _free_anchor(f),
        "live": sorted(live)[0],
        "unfitted": unfitted[0],  # at or below the largest id: can never be minted
        "minted": top + 1,  # what the batch's one unpinned create installs
        "beyond": top + 2,  # past anything one create can mint
        "far": 987654,
    }


def _create_batch_file(tmp_path: Path, b: Dict[str, Any]) -> Path:
    return _write(
        tmp_path,
        f"create,new,{b['anchor']:.6f},\n"
        f"accept,{b['live']},,\n"
        f"accept,{b['unfitted']},,\n"
        f"accept,{b['far']},,\n"
        f"accept,{b['beyond']},,\n",
    )


@pytest.mark.parametrize("call_name", ["review_apply", "review_preview"])
def test_unknown_windows_are_all_reported_despite_an_unpinned_create(
    create_batch, tmp_path, call_name
):
    b = create_batch
    cur = _create_batch_file(tmp_path, b)
    before = content_digest(b["path"])
    with pytest.raises(ValueError) as exc:
        getattr(ftmw, call_name)(b["path"], cur)
    _assert_not_found(exc, "window", [b["unfitted"], b["far"], b["beyond"]])
    assert content_digest(b["path"]) == before


def test_unknown_windows_in_an_action_batch(create_batch):
    b = create_batch
    actions = [
        CurationAction("create", freq_mhz=b["anchor"], frame="raw"),
        CurationAction("accept", window_id=b["unfitted"]),
        CurationAction("accept", window_id=b["far"]),
    ]
    for call in (ftmw.review_apply, ftmw.review_preview):
        with pytest.raises(ValueError) as exc:
            call(b["path"], actions=actions)
        _assert_not_found(exc, "window", [b["unfitted"], b["far"]])


def test_unknown_windows_alongside_an_implied_create(create_batch, tmp_path):
    b = create_batch
    cur = _write(
        tmp_path,
        f"add,,{b['anchor']:.6f},\n"
        f"accept,{b['far']},,\n"
        f"accept,{b['unfitted']},,\n",
    )
    with pytest.raises(ValueError) as exc:
        ftmw.review_apply(b["path"], cur)
    _assert_not_found(exc, "window", [b["far"], b["unfitted"]])


def test_the_id_a_create_mints_is_still_accepted(create_batch, tmp_path):
    b = create_batch
    cur = _write(
        tmp_path,
        f"create,new,{b['anchor']:.6f},\nadd,{b['minted']},{b['anchor']:.6f},\n",
    )
    result = ftmw.review_apply(b["path"], cur)
    assert result.applied == 2
    assert [e.kind for e in review_log_impl(str(b["path"]))] == [
        "create_window",
        "add",
    ]
    assert review_log_impl(str(b["path"]))[1].window_id == b["minted"]


def test_an_unfitted_top_plan_window_is_unknown_and_unmintable():
    """The largest window id of the plan is unfitted: a create mints above it,
    so naming it is unknown even beside an unpinned create, while the id the
    create mints is still left to the per-action check. Hand-built, since the
    shared fixture's top plan window carries a fit."""
    known = {0, 1, 2}
    plan_window_ids = {0, 1, 2, 3}  # 3: planned, never fitted
    plan = [
        s6.PlannedAction(
            kind="create", window_id=s6._NEW_WINDOW_SENTINEL, anchor=100.0
        ),
        s6.PlannedAction(kind="accept", window_id=3),
        s6.PlannedAction(kind="accept", window_id=4),  # what the create mints
        s6.PlannedAction(kind="accept", window_id=5),  # past one mint
    ]
    assert s6._unknown_plan_window_ids(known, plan, plan_window_ids) == [3, 5]
    # A pinned create above every window raises the floor too.
    pinned = [
        s6.PlannedAction(kind="create", window_id=7, anchor=100.0),
        *plan,
        s6.PlannedAction(kind="accept", window_id=8),
    ]
    assert s6._unknown_plan_window_ids(known, pinned, plan_window_ids) == [3, 4, 5]
    # A pinned create placed *after* the unpinned one does not lift what it
    # mints: the create still mints 4, so naming 4 is not unknown.
    pinned_after = [
        *plan[:1],
        s6.PlannedAction(kind="create", window_id=7, anchor=200.0),
        *plan[1:],
        s6.PlannedAction(kind="accept", window_id=8),
    ]
    assert s6._unknown_plan_window_ids(known, pinned_after, plan_window_ids) == [
        3,
        5,
        8,
    ]


def test_without_a_create_every_unknown_window_is_reported(create_batch, tmp_path):
    b = create_batch
    cur = _write(
        tmp_path,
        f"accept,{b['far']},,\naccept,{b['live']},,\naccept,{b['beyond']},,\n",
    )
    with pytest.raises(ValueError) as exc:
        ftmw.review_apply(b["path"], cur)
    _assert_not_found(exc, "window", [b["far"], b["beyond"]])


# ---------------------------------------------------------------------------
# bad_setting paths through the engine (the file/actions never reach a fit)
# ---------------------------------------------------------------------------


def test_review_edit_tokens_are_named_by_their_argument(stage5_multi_file):
    from ftmwpipeline.file_manager import BadSettingError

    f = stage5_multi_file
    wid, _ = _first_peak(f)
    with pytest.raises(BadSettingError) as exc:
        ftmw.review_edit(f, wid, remove=["not-a-peak"])
    assert exc.value.path == "remove" and exc.value.value == "not-a-peak"
    with pytest.raises(BadSettingError) as exc:
        ftmw.review_edit(f, wid, add=["not-a-peak"])
    assert exc.value.path == "add" and exc.value.value == "not-a-peak"
    with pytest.raises(BadSettingError) as exc:
        ftmw.review_edit(f, wid, add=["uid:7"])
    assert exc.value.path == "add"
    with pytest.raises(BadSettingError) as exc:
        ftmw.review_edit(f, wid, remove=["uid:abc"])
    assert exc.value.path == "remove" and exc.value.value == "uid:abc"


@pytest.mark.parametrize("call_name", ["review_apply", "review_preview"])
def test_a_curation_file_refusal_names_its_cell_through_the_api(
    stage5_multi_file, tmp_path, call_name
):
    from ftmwpipeline.file_manager import BadSettingError

    f = stage5_multi_file
    cur = _write(tmp_path, "# note\naccept,3,,\nbogus,3,,\n")
    before = content_digest(f)
    with pytest.raises(BadSettingError) as exc:
        getattr(ftmw, call_name)(f, cur)
    assert exc.value.path == "curation[line 3].action"
    assert exc.value.value == "bogus"
    assert content_digest(f) == before


def test_a_curation_file_refusal_through_the_pipeline_and_cli(
    stage5_multi_file, tmp_path, capsys
):
    from ftmwpipeline.file_manager import BadSettingError

    f = stage5_multi_file
    cur = _write(tmp_path, "add,five,100.0,\n")
    with pytest.raises(BadSettingError) as exc:
        Pipeline.open(f).review_apply(cur)
    assert exc.value.path == "curation[line 1].window"
    rc, payload = _cli_error(capsys, "review", "apply", f, cur)
    assert rc == 1
    assert payload["code"] == "bad_setting"
    assert payload["path"] == "curation[line 1].window"
    assert payload["value"] == "five"


def test_a_bad_action_field_is_named_by_its_index_through_the_api(stage5_multi_file):
    from ftmwpipeline.file_manager import BadSettingError

    f = stage5_multi_file
    good = CurationAction("accept", window_id=2)
    with pytest.raises(BadSettingError) as exc:
        ftmw.review_apply(f, actions=[good, {"action": "add", "freq_mhz": 1.0, "x": 2}])
    assert exc.value.path == "actions[1].x"
    with pytest.raises(BadSettingError) as exc:
        Pipeline.open(f).review_preview(actions=[good, good, 42])
    assert exc.value.path == "actions[2]"


# ---------------------------------------------------------------------------
# A create's refused anchor names the cell or field the anchor came from
# ---------------------------------------------------------------------------

OFF_BAND_MHZ = 1000.0  # far below the fixture's analysis band


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize(
    "rows",
    [
        "accept,{live},,\ncreate,new,{anchor},\n",  # an explicit create
        "accept,{live},,\nadd,,{anchor},\n",  # the create an add implies
    ],
)
def test_a_batch_anchor_refusal_names_its_cell(
    stage5_multi_file, tmp_path, rows, dry_run
):
    from ftmwpipeline.file_manager import BadSettingError

    f = stage5_multi_file
    live = _live_ranges(f)[0][0]
    cur = _write(tmp_path, rows.format(live=live, anchor=OFF_BAND_MHZ))
    before = content_digest(f)
    with pytest.raises(BadSettingError) as exc:
        ftmw.review_apply(f, cur, dry_run=dry_run)
    assert exc.value.path == "curation[line 2].freqs"
    assert exc.value.value == pytest.approx(OFF_BAND_MHZ)
    assert "outside the analysis band" in str(exc.value)
    assert str(exc.value).startswith("curation action ")
    assert content_digest(f) == before


def test_an_action_batch_anchor_refusal_names_its_field(stage5_multi_file):
    from ftmwpipeline.file_manager import BadSettingError

    f = stage5_multi_file
    live = _live_ranges(f)[0][0]
    for anchor_action in (
        CurationAction("create", freq_mhz=OFF_BAND_MHZ, frame="raw"),
        CurationAction("add", freq_mhz=OFF_BAND_MHZ, frame="raw"),
    ):
        actions = [CurationAction("accept", window_id=live), anchor_action]
        for call in (ftmw.review_apply, ftmw.review_preview):
            with pytest.raises(BadSettingError) as exc:
                call(f, actions=actions)
            assert exc.value.path == "actions[1].freq_mhz"


def test_an_anchor_inside_a_window_names_its_cell(stage5_multi_file, tmp_path):
    from ftmwpipeline.file_manager import BadSettingError

    f = stage5_multi_file
    wid, lo, hi = _live_ranges(f)[0]
    cur = _write(tmp_path, f"create,new,{0.5 * (lo + hi)!r},\n")
    with pytest.raises(BadSettingError) as exc:
        ftmw.review_apply(f, cur)
    assert exc.value.path == "curation[line 1].freqs"
    assert "already falls inside window" in str(exc.value)


def test_review_edit_implied_create_anchor_is_named_add(stage5_multi_file):
    from ftmwpipeline.file_manager import BadSettingError

    f = stage5_multi_file
    before = content_digest(f)
    with pytest.raises(BadSettingError) as exc:
        ftmw.review_edit(f, None, add=[OFF_BAND_MHZ])
    assert exc.value.path == "add"
    assert "outside the analysis band" in str(exc.value)
    # review_create names its own argument
    with pytest.raises(BadSettingError) as exc:
        ftmw.review_create(f, OFF_BAND_MHZ)
    assert exc.value.path == "anchor_mhz"
    assert content_digest(f) == before


# ---------------------------------------------------------------------------
# A batch refusal and a dry-run advisory name the request's actions by their
# 0-based request positions -- not by the plan, which coalescing and canonical
# ordering renumber
# ---------------------------------------------------------------------------


def _session_apply(f: Path, **kw: Any) -> Any:
    with Pipeline.open(f).review_session() as session:
        return session.review_apply(**kw)


_BATCH_CALLS = {
    "api": lambda f, **kw: ftmw.review_apply(f, **kw),
    "pipeline": lambda f, **kw: Pipeline.open(f).review_apply(**kw),
    "session": _session_apply,
    "preview": lambda f, **kw: ftmw.review_preview(f, **kw),
}


@pytest.fixture
def split_coalesced(stage5_multi_file) -> Tuple[Path, int, List[float], List[Any]]:
    """Two misses on one window, with an accept of another window between
    them: the misses coalesce into one edit from request actions 0 and 2."""
    f = stage5_multi_file
    wa, wb = _two_window_ids(f)
    far = _far_from_peaks(f, wa)
    misses = [far, far + 0.001]
    actions = [
        CurationAction("remove", window_id=wa, freq_mhz=misses[0], frame="raw"),
        CurationAction("accept", window_id=wb),
        CurationAction("remove", window_id=wa, freq_mhz=misses[1], frame="raw"),
    ]
    return f, wa, misses, actions


@pytest.mark.parametrize("via", sorted(_BATCH_CALLS))
def test_a_coalesced_edit_names_every_request_action(split_coalesced, via):
    f, _wa, misses, actions = split_coalesced
    before = content_digest(f)
    with pytest.raises(ValueError) as exc:
        _BATCH_CALLS[via](f, actions=actions)
    err = _assert_not_found(exc, "peak", misses, approx=True)
    assert err.action_indices == [0, 2]
    assert err.to_dict()["action_indices"] == [0, 2]
    assert str(err).startswith("curation actions 1, 3 (")
    assert content_digest(f) == before


def test_a_curation_file_numbers_its_action_rows(split_coalesced, tmp_path):
    """A file's comment and blank lines are not actions: row i of the file
    and actions[i] of the same request agree."""
    f, wa, misses, actions = split_coalesced
    wb = actions[1].window_id
    cur = _write(
        tmp_path,
        f"# a comment\nremove,{wa},{misses[0]!r},\n\naccept,{wb},,\n"
        f"remove,{wa},{misses[1]!r},\n",
    )
    with pytest.raises(ValueError) as exc:
        ftmw.review_apply(f, cur)
    assert exc.value.action_indices == [0, 2]


def test_a_reordered_batch_names_the_request_position(stage5_multi_file):
    """A create runs first whatever its position, so a refused create after
    an edit is request action 1, though plan action 0."""
    f = stage5_multi_file
    step = _bin_width_mhz(f)
    wid, peak = _first_peak(f)
    _w, _lo, hi = _live_ranges(f)[-1]
    fresh = max(_plan_ranges(f)) + 1
    actions = [
        CurationAction(
            "remove", window_id=wid, freq_mhz=float(peak.frequency_mhz), frame="raw"
        ),
        CurationAction("create", window_id=fresh, freq_mhz=hi + 2 * step, frame="raw"),
    ]
    for call in _BATCH_CALLS.values():
        with pytest.raises(ValueError) as exc:
            call(f, actions=actions)
        assert isinstance(exc.value, CurationConflictError)
        assert exc.value.reason == "replay_conflict"
        assert exc.value.action_indices == [1]
        assert str(exc.value).startswith("curation action 2 (create window")


def test_a_window_sorted_batch_names_the_request_position(stage5_multi_file):
    """Edits run by ascending window: the later request action, on the lower
    window, fails first and is named as itself."""
    f = stage5_multi_file
    wa, wb = _two_window_ids(f)
    miss_a, miss_b = _far_from_peaks(f, wa), _far_from_peaks(f, wb)
    actions = [
        CurationAction("remove", window_id=wb, freq_mhz=miss_b, frame="raw"),
        CurationAction("remove", window_id=wa, freq_mhz=miss_a, frame="raw"),
    ]
    with pytest.raises(ValueError) as exc:
        ftmw.review_apply(f, actions=actions)
    err = _assert_not_found(exc, "peak", [miss_a], approx=True)
    assert err.action_indices == [1]
    assert str(err).startswith("curation action 2 (")


def test_a_batch_refusal_through_the_cli_names_the_request(
    split_coalesced, tmp_path, capsys
):
    f, _wa, _misses, actions = split_coalesced
    spec = tmp_path / "actions.json"
    spec.write_text(json.dumps([a.to_dict() for a in actions]))
    for verb in ("apply", "preview"):
        rc, payload = _cli_error(capsys, "review", verb, f, "--actions", spec)
        assert rc == 1
        assert payload["code"] == "not_found"
        assert payload["action_indices"] == [0, 2]


def test_a_single_verb_refusal_carries_no_action_indices(two_misses):
    f, wid, misses = two_misses
    with pytest.raises(ValueError) as exc:
        ftmw.review_edit(f, wid, remove=misses)
    err = _assert_not_found(exc, "peak", misses, approx=True)
    assert err.action_indices is None
    d = err.to_dict()
    assert d["action_indices"] is None and d["action_indices_absent"] == "not_run"


def test_dry_run_warnings_name_their_request_actions(split_coalesced, tmp_path):
    f, _wa, _misses, actions = split_coalesced
    for result in (
        ftmw.review_apply(f, actions=actions, dry_run=True),
        Pipeline.open(f).review_apply(actions=actions, dry_run=True),
    ):
        assert [d.message for d in result.warning_details] == result.warnings
        assert len(result.warning_details) == 2  # one per missed target
        for detail in result.warning_details:
            assert detail.action_indices == [0, 2]


def _cli_doc(capsys, *argv: Any) -> Dict[str, Any]:
    capsys.readouterr()
    assert main([str(a) for a in argv] + ["--json"]) == 0
    return json.loads(capsys.readouterr().out)


def test_review_apply_json_carries_the_results_advisories(
    split_coalesced, tmp_path, capsys
):
    """``review apply --json`` (dry run or not) carries ``warnings`` and
    ``warning_details`` beside the run_result summary, as the Python result
    does; the summary stays scalar-only."""
    f, _wa, _misses, actions = split_coalesced
    spec = tmp_path / "actions.json"
    spec.write_text(json.dumps([a.to_dict() for a in actions]))
    result = ftmw.review_apply(f, actions=actions, dry_run=True)
    doc = _cli_doc(capsys, "review", "apply", f, "--actions", spec, "--dry-run")
    assert doc["schema"] == "ftmw/run_result@1"
    assert doc["warnings"] == result.warnings and len(doc["warnings"]) == 2
    assert doc["warning_details"] == [
        {"message": d.message, "action_indices": list(d.action_indices)}
        for d in result.warning_details
    ]
    assert not any(isinstance(v, list) for v in doc["summary"].values())

    accept = tmp_path / "accept.json"
    accept.write_text(json.dumps([actions[1].to_dict()]))
    doc = _cli_doc(capsys, "review", "apply", f, "--actions", accept)
    assert doc["warnings"] == [] and doc["warning_details"] == []
    assert doc["summary"]["dry_run"] is False


# ---------------------------------------------------------------------------
# A refusal of the whole batch names no action: action_indices is undefined
# ---------------------------------------------------------------------------


def _assert_whole_batch(err: Any) -> None:
    assert err.action_indices is None
    d = err.to_dict()
    assert d["action_indices"] is None
    assert d["action_indices_absent"] == "undefined"


def test_unknown_windows_refuse_the_whole_batch(create_batch, tmp_path, capsys):
    b = create_batch
    actions = [
        CurationAction("create", freq_mhz=b["anchor"], frame="raw"),
        CurationAction("accept", window_id=b["far"]),
    ]
    bare = [CurationAction("accept", window_id=b["far"])]
    for plan in (actions, bare):
        for call in (
            ftmw.review_apply,
            ftmw.review_preview,
            lambda p, actions: ftmw.review_apply(p, actions=actions, dry_run=True),
        ):
            with pytest.raises(ValueError) as exc:
                call(b["path"], actions=plan)
            _assert_whole_batch(exc.value)
    spec = tmp_path / "actions.json"
    spec.write_text(json.dumps([a.to_dict() for a in actions]))
    for argv in (("apply",), ("apply", "--dry-run"), ("preview",)):
        rc, payload = _cli_error(capsys, "review", *argv, b["path"], "--actions", spec)
        assert rc == 1 and payload["code"] == "not_found"
        assert payload["action_indices"] is None
        assert payload["action_indices_absent"] == "undefined"


def test_uncovered_removes_refuse_the_whole_batch(stage5_multi_file, tmp_path):
    f = stage5_multi_file
    unc = _uncovered_freq(f)
    cur = _write(tmp_path, f"remove,,{unc!r},\n")
    for call in (ftmw.review_apply, ftmw.review_preview):
        with pytest.raises(ValueError) as exc:
            call(f, cur)
        _assert_whole_batch(exc.value)


def test_a_single_verb_unknown_window_is_still_not_run(stage5_multi_file):
    with pytest.raises(NotFoundError) as exc:
        ftmw.review_accept(stage5_multi_file, 987654)
    d = exc.value.to_dict()
    assert d["action_indices_absent"] == "not_run"
