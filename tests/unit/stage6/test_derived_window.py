"""W2 -- the window becomes optional where it is a COORDINATE.

``review edit``'s ``add``/``remove`` targets (through ``api.review_edit``,
``Pipeline.review_edit``, ``ReviewSession.review_edit``, and the CLI's
``--window``) and a curation file's ``add``/``remove`` rows no longer require
a window id: it is derived from the target's own frequency (or ``uid:N``) by
live-window coverage. ``review accept``, ``review create``, and a bare
``review edit`` (no add/remove -- an identity refit) are the SUBJECT, not a
coordinate, and keep the window REQUIRED -- untouched by this file.

THE RULE THAT MUST NOT BE VIOLATED (see ``scratch/intent-driven-windowing-
plan.md``): deriving the window never widens the search. Resolution is always
(1) find which live window covers the frequency, THEN (2) run the existing,
unchanged, snap-tolerance-bounded match inside that one window -- never a
global nearest-peak search. ``test_remove_no_peak_within_snap_tolerance_still_
errors`` below is the test that pins this; see its docstring.

Uses ``stage5_multi_file`` (several live fitted windows) from ``conftest.py``
throughout -- the 3-window ``stage5_small_source`` build usually has only ONE
live window, which is not enough to test cross-window ambiguity. Writes only
to pytest ``tmp_path``.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.stage4_impl import load_windows_impl
from ftmwpipeline._internal.stage6_impl import (
    _DERIVE_WINDOW_SENTINEL,
    _NEW_WINDOW_SENTINEL,
    apply_curation_impl,
    parse_curation_file,
    refit_snap_tol_mhz_impl,
    refit_window_impl,
    review_log_impl,
    review_undo_impl,
)
from ftmwpipeline.cli.review_commands import cmd_review_edit
from ftmwpipeline.core.data_structures import FittedPeak, FittingResult
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5
from ftmwpipeline.pipeline import Pipeline

# ---------------------------------------------------------------------------
# parse_curation_file: the omitted-window grammar on add/remove rows, pure
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, text: str) -> str:
    p = tmp_path / "curation.csv"
    p.write_text(text)
    return str(p)


@pytest.mark.parametrize("token", ["", "new", "auto", "-"])
def test_parse_add_row_omitted_window_derives(tmp_path, token):
    ops = parse_curation_file(_write(tmp_path, f"add,{token},27549.3259,\n"))
    assert len(ops) == 1
    assert ops[0].action == "add"
    assert ops[0].window_id == _DERIVE_WINDOW_SENTINEL


@pytest.mark.parametrize("token", ["", "new", "auto", "-"])
def test_parse_remove_row_omitted_window_derives(tmp_path, token):
    ops = parse_curation_file(_write(tmp_path, f"remove,{token},27549.3259,\n"))
    assert len(ops) == 1
    assert ops[0].action == "remove"
    assert ops[0].window_id == _DERIVE_WINDOW_SENTINEL


def test_parse_create_row_still_unpinned_sentinel(tmp_path):
    """'create' keeps its own, unrelated sentinel -- the two must never be
    confused, since a 'create' with an omitted window MINTS a window, while
    an add/remove with an omitted window DERIVES an existing one."""
    ops = parse_curation_file(_write(tmp_path, "create,new,27549.0,\n"))
    assert ops[0].window_id == _NEW_WINDOW_SENTINEL
    assert ops[0].window_id != _DERIVE_WINDOW_SENTINEL


def test_parse_accept_row_still_requires_window(tmp_path):
    """accept is the SUBJECT, not a coordinate: an omitted window is still a
    hard parse error there, unaffected by W2's add/remove grammar."""
    with pytest.raises(ValueError, match="missing window id"):
        parse_curation_file(_write(tmp_path, "accept,,,\n"))


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _load_spectrum_fit(path: Path):
    with h5py.File(str(path), "r") as h5f:
        return load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])


def _fitted_by_window(path: Path) -> Dict[int, List[float]]:
    sf = _load_spectrum_fit(path)
    return {
        int(wf.window_id): sorted(
            round(float(p.frequency_mhz), 6) for p in wf.fitted_peaks
        )
        for wf in sf.window_fits
        if wf.window_id is not None
    }


def _fitted_wf(path: Path, wid: int) -> FittingResult:
    sf = _load_spectrum_fit(path)
    return next(wf for wf in sf.window_fits if wf.window_id == wid)


def _first_peak(path: Path) -> Tuple[int, FittedPeak]:
    """(window_id, FittedPeak) of the first fitted peak with a stamped uid."""
    sf = _load_spectrum_fit(path)
    for wf in sf.window_fits:
        if wf.window_id is None:
            continue
        for p in wf.fitted_peaks:
            if p.peak_uid is not None:
                return int(wf.window_id), p
    raise AssertionError(
        "expected stage5_multi_file to carry at least one fitted peak with "
        "a stamped peak_uid"
    )


def _two_window_ids(path: Path) -> Tuple[int, int]:
    """Two distinct window ids that each carry >=1 fitted peak."""
    by_w = _fitted_by_window(path)
    wids = sorted(wid for wid, freqs in by_w.items() if freqs)
    if len(wids) < 2:
        pytest.skip("Need at least two fitted windows")
    return wids[0], wids[1]


def _window_with_n_peaks(path: Path, n: int) -> int:
    by_w = _fitted_by_window(path)
    for wid, freqs in by_w.items():
        if len(freqs) >= n:
            return wid
    pytest.skip(f"Need a window with at least {n} fitted peaks")


def _clear_add_freq(path: Path, wid: int) -> float:
    """The point in ``wid``'s own range farthest from every fitted peak in
    it -- a safe ``add`` target (never re-birthing an existing identity) and,
    for the snap-tolerance test below, the best candidate for "far enough
    from every peak to miss the snap tolerance" without leaving the window's
    own range (staying inside it is what proves derivation resolved to THIS
    window rather than failing to resolve at all).
    """
    plan = load_windows_impl(str(path))["plan"]
    w = next(x for x in plan.windows if int(x.window_id) == wid)
    lo, hi = w.freq_range
    lo, hi = min(lo, hi), max(lo, hi)
    wf = _fitted_wf(path, wid)
    peaks = [float(p.frequency_mhz) for p in wf.fitted_peaks]
    if not peaks:
        return 0.5 * (lo + hi)
    grid = np.linspace(lo, hi, 512)[1:-1]
    dist = np.min(np.abs(grid[:, None] - np.asarray(peaks)[None, :]), axis=1)
    return float(grid[int(np.argmax(dist))])


def _uncovered_freq(path: Path) -> float:
    """A frequency well outside every window in the Stage 4 plan (live or
    dead) -- guaranteed uncovered by any live window a fortiori."""
    plan = load_windows_impl(str(path))["plan"]
    lo = min(min(w.freq_range) for w in plan.windows)
    return lo - 1000.0


def _write_curation(tmp_path: Path, name: str, text: str) -> str:
    p = tmp_path / name
    p.write_text(text)
    return str(p)


# ---------------------------------------------------------------------------
# review_edit: omitted window derives the same result a named call would
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_remove_omitted_window_matches_named(stage5_multi_file, tmp_path):
    wid, target = _first_peak(stage5_multi_file)
    token = f"uid:{target.peak_uid}"

    named = tmp_path / "named.ftmw"
    derived = tmp_path / "derived.ftmw"
    shutil.copy(stage5_multi_file, named)
    shutil.copy(stage5_multi_file, derived)

    refit_window_impl(str(named), wid, remove=[token])
    refit_window_impl(str(derived), None, remove=[token])

    assert _fitted_by_window(derived) == _fitted_by_window(named)


@pytest.mark.integration
def test_add_omitted_window_matches_named(stage5_multi_file, tmp_path):
    wid, _ = _first_peak(stage5_multi_file)
    freq = _clear_add_freq(stage5_multi_file, wid)

    named = tmp_path / "named.ftmw"
    derived = tmp_path / "derived.ftmw"
    shutil.copy(stage5_multi_file, named)
    shutil.copy(stage5_multi_file, derived)

    refit_window_impl(str(named), wid, add=[freq])
    refit_window_impl(str(derived), None, add=[freq])

    assert _fitted_by_window(derived) == _fitted_by_window(named)


@pytest.mark.integration
def test_remove_uncovered_frequency_errors_and_nothing_removed(stage5_multi_file):
    """A remove whose frequency no live window covers is an error -- for
    remove this is PERMANENT (a remove never implies a create, unlike a
    future add) -- and touches nothing anywhere in the file."""
    path = stage5_multi_file
    bad = _uncovered_freq(path)
    before = _fitted_by_window(path)

    with pytest.raises(ValueError, match=re.escape(f"{bad:.4f}")):
        refit_window_impl(str(path), None, remove=[bad])

    assert _fitted_by_window(path) == before


@pytest.mark.integration
def test_remove_no_peak_within_snap_tolerance_still_errors(stage5_multi_file):
    """THE MOST IMPORTANT TEST IN THIS FILE.

    Deriving the window must NEVER widen the search. Pick a frequency
    strictly inside a live window's own range but farther than the snap
    tolerance from every fitted peak in it: the derivation must resolve to
    THAT window (it is covered), and the existing, unchanged snap-bounded
    match inside it must then fail exactly as it would for an explicitly
    named window -- raising the ordinary "no fitted peak within tolerance"
    error rather than silently falling through to search a neighboring
    window for something closer. If this test ever needs weakening to pass,
    that is the regression this whole work item exists to prevent: a
    transposed digit must never delete the wrong peak.
    """
    path = stage5_multi_file
    wid, _ = _first_peak(path)
    far_freq = _clear_add_freq(path, wid)
    snap_tol = refit_snap_tol_mhz_impl(path)
    fitted = _fitted_by_window(path)[wid]
    if not fitted or min(abs(far_freq - f) for f in fitted) <= snap_tol:
        pytest.skip("window too densely packed to find a >snap_tol gap")

    before = _fitted_by_window(path)
    with pytest.raises(ValueError, match="no fitted peak within"):
        refit_window_impl(str(path), None, remove=[far_freq])

    assert _fitted_by_window(path) == before


@pytest.mark.integration
def test_named_window_still_checked_when_wrong(stage5_multi_file):
    """A NAMED window stays an assertion -- naming the wrong one is still an
    error with today's behavior, exactly as before. This also proves the
    derivation path is never silently substituted for an explicit,
    incorrect window_id."""
    path = stage5_multi_file
    wa, wb = _two_window_ids(path)
    target_freq = _fitted_by_window(path)[wa][0]
    before = _fitted_by_window(path)

    with pytest.raises(ValueError, match="no fitted peak within"):
        refit_window_impl(str(path), wb, remove=[target_freq])

    assert _fitted_by_window(path) == before


@pytest.mark.integration
def test_remove_by_uid_omitted_window_resolves(stage5_multi_file):
    path = stage5_multi_file
    wid, target = _first_peak(path)
    n_before = len(_fitted_wf(path, wid).fitted_peaks)

    result = refit_window_impl(str(path), None, remove=[f"uid:{target.peak_uid}"])

    assert result.window_id == wid
    assert result.n_peaks_after == n_before - 1
    assert target.peak_uid not in {p.peak_uid for p in result.fitted_peaks}


@pytest.mark.integration
def test_remove_by_uid_unknown_omitted_window_errors(stage5_multi_file):
    path = stage5_multi_file
    before = _fitted_by_window(path)

    with pytest.raises(ValueError, match="999999999"):
        refit_window_impl(str(path), None, remove=["uid:999999999"])

    assert _fitted_by_window(path) == before


@pytest.mark.integration
def test_review_edit_targets_different_windows_errors(stage5_multi_file):
    """add/remove targets that resolve to two different windows in ONE
    review_edit call are refused, naming both -- a single call is scoped to
    one window's refit (RefitWindowResult is a per-window result), so there
    is no "pick one" fallback."""
    path = stage5_multi_file
    wa, wb = _two_window_ids(path)
    remove_freq = _fitted_by_window(path)[wa][0]
    add_freq = _clear_add_freq(path, wb)
    before = _fitted_by_window(path)

    with pytest.raises(ValueError) as excinfo:
        refit_window_impl(str(path), None, add=[add_freq], remove=[remove_freq])

    msg = str(excinfo.value)
    assert str(wa) in msg
    assert str(wb) in msg
    assert _fitted_by_window(path) == before


@pytest.mark.integration
def test_bare_edit_omitted_window_requires_explicit(stage5_multi_file):
    """A bare edit (no add/remove -- an identity refit) is the SUBJECT, not a
    coordinate: window_id stays REQUIRED there, unaffected by W2."""
    path = stage5_multi_file
    before = _fitted_by_window(path)

    with pytest.raises(ValueError, match="window_id is required"):
        refit_window_impl(str(path), None)

    assert _fitted_by_window(path) == before


# ---------------------------------------------------------------------------
# Curation file: omitted-window add/remove rows
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_curation_file_omitted_window_rows_coalesce(stage5_multi_file, tmp_path):
    """Regression guard for resolve-before-coalesce: several omitted-window
    rows that resolve to the SAME window must still coalesce into ONE REFIT
    -- exactly as a run of explicitly-named rows does
    (``_resolve_curation_plan``'s docstring: "one refit instead of one per
    row"). ``_batch_apply_edit_plain`` records one decision-log entry PER
    frequency (by design -- see its own docstring), all sharing one
    ``evidence`` dict stamped with the single refit's before/after counts, so
    the coalescing signal is not the entry COUNT but that every entry from
    this row group carries IDENTICAL evidence. Resolving window ids AFTER
    coalescing would leave same-window, omitted-window rows ungrouped (no
    shared window id to coalesce on yet), so each row would cost its OWN
    refit and the two entries' ``n_peaks_before``/``n_peaks_after`` would
    disagree (2->1, then 1->0) instead of matching (2->0 on both)."""
    path = stage5_multi_file
    wid = _window_with_n_peaks(path, 2)
    peaks = sorted(_fitted_wf(path, wid).fitted_peaks, key=lambda p: p.frequency_mhz)
    f1, f2 = float(peaks[0].frequency_mhz), float(peaks[1].frequency_mhz)
    n_start = len(peaks)

    cur = _write_curation(tmp_path, "coalesce.csv", f"remove,,{f1},\nremove,,{f2},\n")
    n_log_before = len(review_log_impl(path))

    apply_curation_impl(path, cur)

    log = review_log_impl(path)
    new_entries = log[n_log_before:]
    assert len(new_entries) == 2  # one decision per removed frequency
    assert all(e.window_id == wid and e.kind == "remove" for e in new_entries)
    # Both entries' evidence is the SAME dict from the SAME (single) refit --
    # n_peaks_before is n_start for BOTH, not n_start then n_start - 1.
    assert new_entries[0].evidence == new_entries[1].evidence
    assert new_entries[0].evidence["n_peaks_before"] == n_start
    assert new_entries[0].evidence["n_peaks_after"] == n_start - 2


@pytest.mark.integration
def test_curation_file_omitted_window_rows_do_not_coalesce_across_windows(
    stage5_multi_file, tmp_path
):
    """The other half of the resolve-before-coalesce guard, and the dangerous
    half.

    ``test_curation_file_omitted_window_rows_coalesce`` pins that same-window
    rows still group. This pins that DIFFERENT-window rows do not. It is the
    more important direction: every omitted-window row parses to the single
    shared ``_DERIVE_WINDOW_SENTINEL``, so an implementation that resolved
    window ids AFTER ``_resolve_curation_plan`` would find every such row
    carrying one identical window id and coalesce rows for DIFFERENT windows
    into ONE action -- then resolve that action to whichever window its first
    target happened to name, and apply the other window's remove against the
    wrong peak set. (Same-window rows would still group there by accident,
    which is why that test alone does not cover this.)

    Two removes in two different live windows must therefore produce two
    separate refits: two decision entries carrying the two window ids, with
    DIFFERENT evidence dicts (identical evidence is the coalescing signal --
    see the same-window test).
    """
    path = stage5_multi_file
    w1, w2 = _two_window_ids(path)
    before = _fitted_by_window(path)
    f1, f2 = before[w1][0], before[w2][0]

    cur = _write_curation(
        tmp_path, "two_windows.csv", f"remove,,{f1},\nremove,,{f2},\n"
    )
    n_log_before = len(review_log_impl(path))

    apply_curation_impl(path, cur)

    log = review_log_impl(path)
    new_entries = log[n_log_before:]
    assert len(new_entries) == 2
    assert {e.window_id for e in new_entries} == {w1, w2}
    # Two windows, two refits: the evidence dicts cannot be the shared one a
    # single coalesced action would stamp on both.
    assert new_entries[0].evidence != new_entries[1].evidence

    after = _fitted_by_window(path)
    assert len(after[w1]) == len(before[w1]) - 1
    assert len(after[w2]) == len(before[w2]) - 1


@pytest.mark.integration
def test_curation_file_uncovered_row_errors_dry_run_and_live(
    stage5_multi_file, tmp_path
):
    """An omitted-window row whose frequency no live window covers cannot
    even be represented in the resolved plan (there is no window id to put
    in it), so it is an error unconditionally -- including on --dry-run,
    unlike an ordinary snap-match ambiguity, which is only a warning there."""
    path = stage5_multi_file
    bad = _uncovered_freq(path)
    cur = _write_curation(tmp_path, "bad.csv", f"remove,,{bad},\n")
    before = _fitted_by_window(path)

    with pytest.raises(ValueError, match=re.escape(f"{bad:.4f}")):
        apply_curation_impl(path, cur, dry_run=True)
    assert _fitted_by_window(path) == before

    with pytest.raises(ValueError, match=re.escape(f"{bad:.4f}")):
        apply_curation_impl(path, cur)
    assert _fitted_by_window(path) == before


@pytest.mark.integration
def test_curation_file_named_window_still_checked_when_wrong(
    stage5_multi_file, tmp_path
):
    path = stage5_multi_file
    wa, wb = _two_window_ids(path)
    target_freq = _fitted_by_window(path)[wa][0]
    cur = _write_curation(tmp_path, "wrong.csv", f"remove,{wb},{target_freq},\n")
    before = _fitted_by_window(path)

    with pytest.raises(ValueError):
        apply_curation_impl(path, cur)

    assert _fitted_by_window(path) == before


# ---------------------------------------------------------------------------
# Undo replay: unaffected by a derived-window edit
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_undo_replay_unaffected_by_derived_window_edit(stage5_multi_file):
    """The decision log records the RESOLVED, concrete window id -- never the
    sentinel -- so undo replay (which reconstructs ops straight from the log
    via _decision_to_op, never touching _resolve_curation_window_ids at all)
    is unaffected: it undoes and restores exactly as it would for a
    named-window edit."""
    path = stage5_multi_file
    baseline = _fitted_by_window(path)
    wid, target = _first_peak(path)

    result = refit_window_impl(str(path), None, remove=[f"uid:{target.peak_uid}"])
    assert result.window_id == wid
    assert _fitted_by_window(path) != baseline

    log = review_log_impl(path)
    assert len(log) == 1
    entry = log[0]
    assert entry.window_id == wid
    assert entry.window_id != _DERIVE_WINDOW_SENTINEL

    undo_result = review_undo_impl(str(path), [entry.order_index])
    assert undo_result.applied == 0
    assert _fitted_by_window(path) == baseline
    assert review_log_impl(path) == []


# ---------------------------------------------------------------------------
# Cross-interface consistency: api / Pipeline / CLI, window omitted
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_review_edit_omitted_window_cross_interface(stage5_multi_file, tmp_path):
    paths = {k: tmp_path / f"deriv_{k}.ftmw" for k in ("api", "pipe", "cli")}
    for p in paths.values():
        shutil.copy(stage5_multi_file, p)

    _wid, target = _first_peak(paths["api"])
    token = f"uid:{target.peak_uid}"

    ftmw.review_edit(paths["api"], remove=[token])
    Pipeline.open(paths["pipe"]).review_edit(remove=[token])
    rc = cmd_review_edit(
        argparse.Namespace(
            file_path=str(paths["cli"]), window=None, add=None, remove=[token]
        )
    )
    assert rc == 0

    ref = _fitted_by_window(paths["api"])
    assert _fitted_by_window(paths["pipe"]) == ref
    assert _fitted_by_window(paths["cli"]) == ref
