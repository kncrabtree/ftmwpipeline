"""W4 -- the preview surfaces the implied structure.

BlackQuill's acceptance condition for W3's implicit window creation
(``scratch/intent-driven-windowing-plan.md``, W4, and
``scratch/bq-correspondence/ask-intent-driven-windowing.md``): letting an
``add`` mint the window it needs is acceptable to them *iff* ``review_preview``
/ ``apply --dry-run`` reports the implied create (mode, extent, ``depends_on``)
so a UI can render "this add creates a window at A-B MHz" before apply -- and
iff the preview-to-apply byte-for-byte guarantee covers the pair as one plan.

The data already existed and had no home: :class:`CreateWindowResult` (W3)
already carries ``mode``, ``freq_range``, ``n_points``, ``n_contributors`` and
``depends_on``, and the ``created_window`` decision-log evidence records the
same set. This adds five additive ``created_window_*`` fields to
``PreviewWindowResult`` and to ``RefitWindowResult`` -- ``None`` on any window
a batch did not create or widen -- and wires the CLI (``review preview``,
``review apply --dry-run``, and ``review edit``) to print them.

New module rather than an extension of ``test_review_preview.py``: this is a
narrow, self-contained surface (five fields on two dataclasses, plus their CLI
rendering) with the same fixture needs as W3's own module
(``test_implied_create.py``) rather than W3's -- keeping it separate mirrors
how W3 got its own module instead of growing ``test_derived_window.py``.

Uses ``stage5_multi_file`` (several live fitted windows), like
``test_implied_create.py``. Writes only to pytest ``tmp_path``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple

import h5py
import pytest

from ftmwpipeline._internal.stage4_impl import load_windows_impl
from ftmwpipeline._internal import stage6_impl as s6
from ftmwpipeline._internal.stage6_impl import (
    apply_curation_impl,
    refit_window_impl,
    review_log_impl,
    review_preview_impl,
)
from ftmwpipeline.cli.review_commands import cmd_review_apply, cmd_review_preview
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5
from ftmwpipeline.pipeline import Pipeline

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Helpers -- duplicated from test_implied_create.py per this suite's own
# precedent (tests/AGENTS.md; test_frame_parameter.py's docstring) rather
# than imported across test files.
# ---------------------------------------------------------------------------


def _load_spectrum_fit(path: Path):
    with h5py.File(str(path), "r") as h5f:
        return load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])


def _first_peak(path: Path) -> Tuple[int, int]:
    """``(window_id, peak_uid)`` of the first fitted peak carrying a stamped
    uid -- a target for an ORDINARY (non-implied-create) edit."""
    sf = _load_spectrum_fit(path)
    for wf in sf.window_fits:
        if wf.window_id is None:
            continue
        for p in wf.fitted_peaks:
            if p.peak_uid is not None:
                return int(wf.window_id), int(p.peak_uid)
    pytest.skip(
        "expected stage5_multi_file to carry at least one fitted peak with "
        "a stamped peak_uid"
    )


def _gap_anchor(path: Path) -> float:
    """A molecular frequency inside the analysis band but outside every
    PLANNED window (live or dead) -- reliably triggers ``mode="created"``
    rather than ``mode="widened"``. Mirrors ``test_implied_create.py``."""
    plan = load_windows_impl(str(path))["plan"]
    spans = sorted((min(w.freq_range), max(w.freq_range)) for w in plan.windows)
    for (_lo1, hi1), (lo2, _hi2) in zip(spans, spans[1:]):
        if lo2 - hi1 > 4.0:
            return 0.5 * (hi1 + lo2)
    pytest.skip("no gap between planned windows wide enough to create into")


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


def _live_ranges(path: Path) -> List[Tuple[int, float, float]]:
    """``(window_id, lo, hi)`` for every window that carries a Stage 5 fit,
    ascending by ``lo``. Mirrors ``test_implied_create.py``."""
    sf = _load_spectrum_fit(path)
    plan = load_windows_impl(str(path))["plan"]
    by_id = {int(w.window_id): w for w in plan.windows}
    out: List[Tuple[int, float, float]] = []
    for wf in sf.window_fits:
        if wf.window_id is None:
            continue
        w = by_id.get(int(wf.window_id))
        if w is None:
            continue
        lo, hi = w.freq_range
        out.append((int(wf.window_id), min(lo, hi), max(lo, hi)))
    return sorted(out, key=lambda t: t[1])


def _write_curation(tmp_path: Path, name: str, text: str) -> str:
    p = tmp_path / name
    p.write_text(text)
    return str(p)


# ---------------------------------------------------------------------------
# PreviewWindowResult: an implied create matches what the apply installs
# ---------------------------------------------------------------------------


def test_preview_of_implied_create_matches_the_apply(stage5_multi_file, tmp_path):
    base = stage5_multi_file
    freq = _gap_anchor(base)
    cur = _write_curation(tmp_path, "implied.csv", f"add,,{freq},\n")

    preview = review_preview_impl(base, cur)
    assert len(preview.windows) == 1
    wid = next(iter(preview.windows))
    w = preview.windows[wid]

    assert w.created_window_mode == "created"
    assert w.created_window_freq_range is not None
    assert w.created_window_n_points is not None and w.created_window_n_points > 0
    assert w.created_window_n_contributors is not None
    assert w.created_window_depends_on is not None
    # A preview never persists.
    assert review_log_impl(base) == []

    # Apply the SAME curation to an independent copy of the SAME base file
    # and compare against what actually landed -- never hard-coded.
    applied_path = tmp_path / "applied.ftmw"
    shutil.copy(base, applied_path)
    apply_curation_impl(applied_path, cur)

    log = review_log_impl(applied_path)
    assert len(log) == 1
    assert log[0].window_id == wid
    created = log[0].evidence["created_window"]
    assert created["mode"] == w.created_window_mode
    assert (
        created["freq_min_mhz"],
        created["freq_max_mhz"],
    ) == w.created_window_freq_range
    assert created["n_points"] == w.created_window_n_points
    assert created["n_contributors"] == w.created_window_n_contributors
    assert created["depends_on"] == w.created_window_depends_on


def test_preview_of_ordinary_edit_leaves_structure_none(stage5_multi_file, tmp_path):
    path = stage5_multi_file
    wid, uid = _first_peak(path)
    cur = _write_curation(tmp_path, "ordinary.csv", f"remove,{wid},uid:{uid},\n")

    preview = review_preview_impl(path, cur)
    assert wid in preview.windows
    w = preview.windows[wid]

    assert w.created_window_mode is None
    assert w.created_window_freq_range is None
    assert w.created_window_n_points is None
    assert w.created_window_n_contributors is None
    assert w.created_window_depends_on is None


def test_preview_widened_mode_is_distinguishable_from_created(
    stage5_multi_file, tmp_path
):
    path = stage5_multi_file
    step = _bin_width_mhz(path)
    live = _live_ranges(path)
    wid, _lo, hi = live[-1]
    anchor = hi + 2 * step
    cur = _write_curation(tmp_path, "widen.csv", f"add,,{anchor},\n")

    preview = review_preview_impl(path, cur)
    assert wid in preview.windows
    w = preview.windows[wid]

    assert w.created_window_mode == "widened"
    assert w.created_window_mode != "created"
    assert w.created_window_freq_range is not None
    assert w.created_window_n_points is not None
    assert w.created_window_n_contributors is not None
    assert w.created_window_depends_on is not None


# ---------------------------------------------------------------------------
# RefitWindowResult: review_edit's own result carries (or omits) the same
# ---------------------------------------------------------------------------


def test_refit_result_carries_structure_for_implied_create(stage5_multi_file):
    path = stage5_multi_file
    freq = _gap_anchor(path)

    result = refit_window_impl(str(path), None, add=[freq])

    assert result.created_window_mode == "created"
    assert result.created_window_freq_range is not None
    assert result.created_window_n_points is not None
    assert result.created_window_n_contributors is not None
    assert result.created_window_depends_on is not None

    entry = review_log_impl(path)[0]
    created = entry.evidence["created_window"]
    assert created["mode"] == result.created_window_mode
    assert (
        created["freq_min_mhz"],
        created["freq_max_mhz"],
    ) == result.created_window_freq_range
    assert created["n_points"] == result.created_window_n_points
    assert created["n_contributors"] == result.created_window_n_contributors
    assert created["depends_on"] == result.created_window_depends_on


def test_refit_result_leaves_structure_none_for_ordinary_edit(stage5_multi_file):
    path = stage5_multi_file
    wid, uid = _first_peak(path)

    result = refit_window_impl(str(path), wid, remove=[f"uid:{uid}"])

    assert result.created_window_mode is None
    assert result.created_window_freq_range is None
    assert result.created_window_n_points is None
    assert result.created_window_n_contributors is None
    assert result.created_window_depends_on is None


# ---------------------------------------------------------------------------
# Session staged reuse: the previewed outcome, byte for byte, for a plan
# containing an implied create.
# ---------------------------------------------------------------------------


def test_session_staged_reuse_persists_implied_create_from_preview(
    stage5_multi_file, tmp_path
):
    path = stage5_multi_file
    freq = _gap_anchor(path)
    cur = _write_curation(tmp_path, "implied.csv", f"add,,{freq},\n")

    with Pipeline.open(path).review_session() as session:
        preview = session.review_preview(cur)
        assert len(preview.windows) == 1
        wid = next(iter(preview.windows))
        w = preview.windows[wid]
        assert w.created_window_mode == "created"

        # Spy on the reuse path so the test proves it actually fired, not
        # merely that the numbers happen to agree (test_review_session.py's
        # own D4 pattern).
        calls = {"n": 0}
        orig_persist = session._persist_staged

        def spy(staged):
            calls["n"] += 1
            return orig_persist(staged)

        session._persist_staged = spy
        session.review_apply(cur)

    assert calls["n"] == 1

    log = review_log_impl(path)
    assert len(log) == 1
    assert log[0].window_id == wid
    created = log[0].evidence["created_window"]
    assert created["mode"] == w.created_window_mode
    assert (
        created["freq_min_mhz"],
        created["freq_max_mhz"],
    ) == w.created_window_freq_range
    assert created["n_points"] == w.created_window_n_points
    assert created["n_contributors"] == w.created_window_n_contributors
    assert created["depends_on"] == w.created_window_depends_on


# ---------------------------------------------------------------------------
# CLI: review preview / review apply --dry-run print the implied structure
# ---------------------------------------------------------------------------


def test_cli_review_preview_prints_the_implied_structure(
    stage5_multi_file, tmp_path, capsys
):
    path = stage5_multi_file
    freq = _gap_anchor(path)
    cur = _write_curation(tmp_path, "implied.csv", f"add,,{freq},\n")

    preview = review_preview_impl(path, cur)
    wid = next(iter(preview.windows))
    w = preview.windows[wid]
    assert w.created_window_freq_range is not None
    lo, hi = w.created_window_freq_range

    rc = cmd_review_preview(
        argparse.Namespace(file_path=str(path), curation_file=cur, verbose=False)
    )
    assert rc == 0

    out = capsys.readouterr().out
    assert "created" in out
    assert f"{lo:.4f}" in out
    assert f"{hi:.4f}" in out


def test_cli_review_apply_dry_run_prints_the_implied_structure(
    stage5_multi_file, tmp_path, capsys
):
    path = stage5_multi_file
    freq = _gap_anchor(path)
    cur = _write_curation(tmp_path, "implied.csv", f"add,,{freq},\n")

    preview = review_preview_impl(path, cur)
    wid = next(iter(preview.windows))
    w = preview.windows[wid]
    assert w.created_window_freq_range is not None
    lo, hi = w.created_window_freq_range

    rc = cmd_review_apply(
        argparse.Namespace(file_path=str(path), curation_file=cur, dry_run=True)
    )
    assert rc == 0

    out = capsys.readouterr().out
    assert "created" in out
    assert f"{lo:.4f}" in out
    assert f"{hi:.4f}" in out
    # A dry run still writes nothing.
    assert review_log_impl(path) == []


def test_cli_review_apply_dry_run_without_a_create_prints_nothing_extra(
    stage5_multi_file, tmp_path, capsys
):
    """The narrow-gating check: an ordinary dry run (no create in the plan)
    must not pay for (or print) the extra in-memory preview pass."""
    path = stage5_multi_file
    wid, uid = _first_peak(path)
    cur = _write_curation(tmp_path, "ordinary.csv", f"remove,{wid},uid:{uid},\n")

    rc = cmd_review_apply(
        argparse.Namespace(file_path=str(path), curation_file=cur, dry_run=True)
    )
    assert rc == 0

    out = capsys.readouterr().out
    assert "created" not in out
    assert "widened" not in out


# ---------------------------------------------------------------------------
# The dry run resolves the structure itself -- W4's follow-up
# (scratch/intent-driven-windowing-plan.md): the extent comes from the window
# planner, not from plan resolution, so reporting it used to cost a second
# full in-memory preview. It now costs the proposal alone, and rides on
# CurationApplyResult so all three interfaces carry it rather than only the
# CLI's own rendering.
# ---------------------------------------------------------------------------


def _structure_of(pw) -> Tuple:
    """A PlannedWindowResult reduced to the five facts every other surface
    reports, for comparison against a preview / a decision-log entry."""
    return (
        pw.window_id,
        pw.mode,
        pw.freq_range,
        pw.n_points,
        pw.n_contributors,
        pw.depends_on,
    )


def test_dry_run_reports_what_the_preview_reports(stage5_multi_file, tmp_path):
    """The typo guard is the same guard on both rungs of the ladder."""
    path = stage5_multi_file
    freq = _gap_anchor(path)
    cur = _write_curation(tmp_path, "implied.csv", f"add,,{freq},\n")

    dry = apply_curation_impl(path, cur, dry_run=True)
    preview = review_preview_impl(path, cur)

    assert len(dry.created_windows) == 1
    pw = dry.created_windows[0]
    w = preview.windows[pw.window_id]
    assert _structure_of(pw) == (
        pw.window_id,
        w.created_window_mode,
        w.created_window_freq_range,
        w.created_window_n_points,
        w.created_window_n_contributors,
        w.created_window_depends_on,
    )
    assert pw.anchor_mhz == pytest.approx(freq)
    # Neither wrote anything.
    assert review_log_impl(path) == []


def test_preview_publishes_created_windows_like_the_dry_run(
    stage5_multi_file, tmp_path
):
    """``ReviewPreviewResult.created_windows`` is the dry run's own list.

    The per-window ``created_window_*`` fields cannot carry the ANCHOR --
    they are keyed by the window the create landed in, and the anchor is a
    property of the row that implied it. This is the field a client reads to
    say WHICH add creates the window.
    """
    path = stage5_multi_file
    freq = _gap_anchor(path)
    cur = _write_curation(tmp_path, "implied.csv", f"add,,{freq},\n")

    dry = apply_curation_impl(path, cur, dry_run=True)
    preview = review_preview_impl(path, cur)

    assert len(preview.created_windows) == 1
    assert [_structure_of(pw) for pw in preview.created_windows] == [
        _structure_of(pw) for pw in dry.created_windows
    ]
    assert preview.created_windows[0].anchor_mhz == pytest.approx(freq)
    assert review_log_impl(path) == []


def test_preview_created_windows_is_empty_without_a_create(stage5_multi_file, tmp_path):
    """A plan that installs nothing reports nothing -- not a stale list."""
    path = stage5_multi_file
    wid, _detection_index = _first_peak(path)
    lo, hi = [(w_lo, w_hi) for w, w_lo, w_hi in _live_ranges(path) if w == wid][0]
    add_freq = lo + 0.25 * (hi - lo)
    cur = _write_curation(tmp_path, "plain.csv", f"add,{wid},{add_freq},\n")

    preview = review_preview_impl(path, cur)
    assert preview.created_windows == []


def test_preview_created_windows_carries_one_entry_for_a_coalesced_pair(
    stage5_multi_file, tmp_path
):
    """W3.1: two adds sharing one created window report ONE entry, anchored
    at the add that implied the create -- the same call the decision log
    makes by putting ``created_window`` on the first add's entry."""
    path = stage5_multi_file
    freq1 = _gap_anchor(path)
    probe = apply_curation_impl(
        path,
        _write_curation(tmp_path, "probe.csv", f"add,,{freq1},\n"),
        dry_run=True,
    )
    win_lo, win_hi = probe.created_windows[0].freq_range
    freq2 = win_lo + 0.15 * (win_hi - win_lo)

    cur = _write_curation(tmp_path, "pair.csv", f"add,,{freq1},\nadd,,{freq2},\n")
    preview = review_preview_impl(path, cur)

    assert len(preview.created_windows) == 1
    pw = preview.created_windows[0]
    # The anchor names the FIRST add, not the second and not the midpoint --
    # extent containment would mark both, which is exactly why the anchor is
    # the field a client needs.
    assert pw.anchor_mhz == pytest.approx(freq1)
    assert pw.freq_range[0] <= freq2 <= pw.freq_range[1]
    # Both adds landed in that one window.
    assert preview.windows[pw.window_id].n_peaks_after == 2


def test_dry_run_reports_what_the_apply_installs(stage5_multi_file, tmp_path):
    """The prediction is checked against what actually lands, on an
    independent copy of the same base -- never against a hard-coded extent."""
    base = stage5_multi_file
    freq = _gap_anchor(base)
    cur = _write_curation(tmp_path, "implied.csv", f"add,,{freq},\n")

    dry = apply_curation_impl(base, cur, dry_run=True)
    assert len(dry.created_windows) == 1
    pw = dry.created_windows[0]

    applied_path = tmp_path / "applied.ftmw"
    shutil.copy(base, applied_path)
    live = apply_curation_impl(applied_path, cur)

    # The live apply reports the same shape (question 2 of this work item):
    # created_windows means the same thing in both modes.
    assert [_structure_of(x) for x in live.created_windows] == [_structure_of(pw)]

    log = review_log_impl(applied_path)
    assert len(log) == 1
    created = log[0].evidence["created_window"]
    assert log[0].window_id == pw.window_id
    assert created["mode"] == pw.mode
    assert (created["freq_min_mhz"], created["freq_max_mhz"]) == pw.freq_range
    assert created["n_points"] == pw.n_points
    assert created["n_contributors"] == pw.n_contributors
    assert created["depends_on"] == pw.depends_on


def test_dry_run_predicts_a_widening_too(stage5_multi_file, tmp_path):
    """``mode="widened"`` is the other half of the prediction: a gap too
    narrow to hold a window grows a neighbor instead, and the dry run has to
    say which neighbor and how far."""
    base = stage5_multi_file
    step = _bin_width_mhz(base)
    wid, _lo, hi = _live_ranges(base)[-1]
    anchor = hi + 2 * step
    cur = _write_curation(tmp_path, "widen.csv", f"add,,{anchor},\n")

    dry = apply_curation_impl(base, cur, dry_run=True)
    assert len(dry.created_windows) == 1
    pw = dry.created_windows[0]
    assert pw.mode == "widened"
    assert pw.window_id == wid

    applied_path = tmp_path / "applied.ftmw"
    shutil.copy(base, applied_path)
    live = apply_curation_impl(applied_path, cur)
    assert [_structure_of(x) for x in live.created_windows] == [_structure_of(pw)]


def test_dry_run_does_not_fit_anything(stage5_multi_file, tmp_path, monkeypatch):
    """The point of the change: the structural report costs the window
    proposal, not a fit. ``refit_window_core`` is the engine's only fitter --
    a dry run that reaches it has paid what a preview pays."""
    path = stage5_multi_file
    freq = _gap_anchor(path)
    cur = _write_curation(tmp_path, "implied.csv", f"add,,{freq},\n")

    def no_fitting(*args, **kwargs):
        raise AssertionError("a dry run must not fit")

    monkeypatch.setattr(s6, "refit_window_core", no_fitting)

    result = apply_curation_impl(path, cur, dry_run=True)
    assert len(result.created_windows) == 1
    assert result.created_windows[0].mode == "created"


def test_dry_run_without_a_create_opens_no_engine(
    stage5_multi_file, tmp_path, monkeypatch
):
    """And a plan with no create pays nothing at all -- not even the
    active-FT rebuild, which is the expensive half of opening the engine."""
    path = stage5_multi_file
    wid, uid = _first_peak(path)
    cur = _write_curation(tmp_path, "ordinary.csv", f"remove,{wid},uid:{uid},\n")

    def no_engine(*args, **kwargs):
        raise AssertionError("an ordinary dry run must not open the engine")

    monkeypatch.setattr(s6, "_build_shared_fit_ctx", no_engine)

    result = apply_curation_impl(path, cur, dry_run=True)
    assert result.created_windows == []
    assert len(result.plan) == 1


def test_dry_run_refuses_a_create_the_apply_would_refuse(stage5_multi_file, tmp_path):
    """A dry run that returns is a pre-flight, not a plan echo: an anchor
    outside the analysis band is the apply's refusal, raised by the dry run
    with the same per-action attribution -- and still writing nothing."""
    path = stage5_multi_file
    cur = _write_curation(tmp_path, "offband.csv", "add,,1000.0,\n")

    with pytest.raises(ValueError, match="outside the analysis band"):
        apply_curation_impl(path, cur, dry_run=True)
    with pytest.raises(ValueError, match="outside the analysis band"):
        apply_curation_impl(path, cur)

    assert review_log_impl(path) == []


def test_dry_run_chains_two_creates_in_one_plan(stage5_multi_file, tmp_path):
    """Each proposal is folded into the state the next one plans against,
    exactly as the apply folds its own. Without that, both creates mint the
    same id (one past the base plan's highest) and the dry run predicts a
    collision that never happens."""
    base = stage5_multi_file
    plan = load_windows_impl(str(base))["plan"]
    spans = sorted((min(w.freq_range), max(w.freq_range)) for w in plan.windows)
    gap = None
    for (_lo1, hi1), (lo2, _hi2) in zip(spans, spans[1:]):
        if lo2 - hi1 > 8.0:
            gap = (hi1, lo2)
            break
    if gap is None:
        pytest.skip("no gap wide enough to create into twice")
    lo, hi = gap
    a1, a2 = lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo)
    cur = _write_curation(tmp_path, "two.csv", f"add,,{a1},\nadd,,{a2},\n")

    dry = apply_curation_impl(base, cur, dry_run=True)
    assert len(dry.created_windows) == 2
    first, second = dry.created_windows
    assert first.window_id != second.window_id
    assert {first.mode, second.mode} == {"created"}
    # Disjointness, predicted rather than assumed.
    assert first.freq_range[1] < second.freq_range[0]

    applied_path = tmp_path / "applied.ftmw"
    shutil.copy(base, applied_path)
    live = apply_curation_impl(applied_path, cur)
    assert [_structure_of(x) for x in live.created_windows] == [
        _structure_of(first),
        _structure_of(second),
    ]


def test_session_staged_apply_reports_the_structure_too(stage5_multi_file, tmp_path):
    """The staged fast path persists a preview instead of running the batch,
    so it has to carry the structure across rather than re-deriving it."""
    path = stage5_multi_file
    freq = _gap_anchor(path)
    cur = _write_curation(tmp_path, "implied.csv", f"add,,{freq},\n")

    with Pipeline.open(path).review_session() as session:
        preview = session.review_preview(cur)
        wid = next(iter(preview.windows))
        result = session.review_apply(cur)

    assert [_structure_of(x) for x in result.created_windows] == [
        (
            wid,
            preview.windows[wid].created_window_mode,
            preview.windows[wid].created_window_freq_range,
            preview.windows[wid].created_window_n_points,
            preview.windows[wid].created_window_n_contributors,
            preview.windows[wid].created_window_depends_on,
        )
    ]
    assert result.created_windows[0].anchor_mhz == pytest.approx(freq)


def test_cli_live_apply_prints_the_installed_structure(
    stage5_multi_file, tmp_path, capsys
):
    """A live apply used to install a window silently; it now renders the
    same block the dry run does, in the past tense."""
    path = stage5_multi_file
    freq = _gap_anchor(path)
    cur = _write_curation(tmp_path, "implied.csv", f"add,,{freq},\n")

    rc = cmd_review_apply(
        argparse.Namespace(file_path=str(path), curation_file=cur, dry_run=False)
    )
    assert rc == 0

    out = capsys.readouterr().out
    assert "installed:" in out
    assert "created" in out
    log = review_log_impl(path)
    created = log[0].evidence["created_window"]
    assert f"{created['freq_min_mhz']:.4f}" in out
