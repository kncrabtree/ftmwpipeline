"""W3.1 -- an implied create coalesces into a PRIOR implied create's own
proposed window, within one curation plan.

W3 (``test_implied_create.py``) turned an uncovered ``add`` into an implied
create: a fresh window minted (or an adjacent one widened) for the anchor,
with the add itself carrying the ``created_window`` evidence. That resolution
runs against LIVE windows only, so two omitted-window ``add`` rows in one
window-free gap each read as uncovered and each imply their OWN create -- the
second create's anchor then falls inside the first create's just-minted
window, and ``plan_stage6_window`` refuses the whole plan (dry run and live
apply alike), even though the identical edits applied SEQUENTIALLY succeed
today (the second ``review_edit`` resolves into the window the first one just
built, by ordinary live-window coverage).

This is the fix for that gap: during batch execution, a FRESH implied
create's anchor is checked against the batch's OWN current state
(:func:`~ftmwpipeline._internal.stage6_impl._batch_implied_create_target`,
built from the same inputs and the same coverage predicate
:func:`~ftmwpipeline.preprocessing.window_planning.plan_stage6_window` itself
uses) before minting anything. A hit coalesces the create away entirely (no
second window, no second ``created_facts`` entry) and the paired add applies
as an ORDINARY edit into the real window the first create built -- exactly
the log shape the sequential path already produces. This is NOT a search:
the extent comes from the planner, asked one step earlier; there is no
tolerance, no nearest-peak, no radius.

Only a FRESH implied create coalesces -- a REPLAYED implied create (real,
pinned id from the decision log) and an EXPLICIT ``create`` row both keep
today's refusal unchanged, since both assert a specific window is needed
rather than merely deriving one.

Uses ``stage5_multi_file`` (several live fitted windows), like
``test_implied_create.py``. Writes only to pytest ``tmp_path``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import pytest

from ftmwpipeline._internal.stage4_impl import load_windows_impl
from ftmwpipeline._internal.stage6_impl import (
    apply_curation_impl,
    refit_window_impl,
    review_log_impl,
)
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Helpers
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


def _widest_gap(path: Path) -> Tuple[float, float]:
    """``(lo, hi)`` of the widest gap between adjacent PLANNED windows (live
    or dead) -- the most headroom for placing anchors inside it without
    touching an existing window."""
    plan = load_windows_impl(str(path))["plan"]
    spans = sorted((min(w.freq_range), max(w.freq_range)) for w in plan.windows)
    best: Tuple[float, float] = (0.0, 0.0)
    best_width = -1.0
    for (_lo1, hi1), (lo2, _hi2) in zip(spans, spans[1:]):
        width = lo2 - hi1
        if width > best_width:
            best_width = width
            best = (hi1, lo2)
    return best


def _write_curation(tmp_path: Path, name: str, text: str) -> str:
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def _log_shape(path: Path) -> List[Tuple[int, str, bool]]:
    """``(window_id, kind, carries created_window evidence)`` per decision,
    in log order -- what the coalescing fix must reproduce identically to the
    sequential path."""
    return [
        (e.window_id, e.kind, e.evidence.get("created_window") is not None)
        for e in review_log_impl(path)
    ]


# ---------------------------------------------------------------------------
# The coalescing case itself
# ---------------------------------------------------------------------------


def test_two_close_omitted_window_adds_coalesce_into_one_window(
    stage5_multi_file, tmp_path
):
    path = stage5_multi_file
    lo, hi = _widest_gap(path)
    if hi - lo < 4.0:
        pytest.skip("no gap wide enough to create a window into")
    freq1 = 0.5 * (lo + hi)

    # Learn the extent a lone create at freq1 would take, without mutating
    # anything, so freq2 can be placed reliably INSIDE it (not the window
    # center -- that's freq1's own birth position -- but well clear of it).
    probe_cur = _write_curation(tmp_path, "probe.csv", f"add,,{freq1:.6f},\n")
    probe = apply_curation_impl(path, probe_cur, dry_run=True)
    assert len(probe.created_windows) == 1
    win_lo, win_hi = probe.created_windows[0].freq_range
    freq2 = win_lo + 0.15 * (win_hi - win_lo)
    assert abs(freq2 - freq1) > 0.05  # clear of freq1's own identity

    cur = _write_curation(
        tmp_path, "two_close.csv", f"add,,{freq1:.6f},\nadd,,{freq2:.6f},\n"
    )

    # Dry run: does not raise, and reports exactly ONE created window.
    dry = apply_curation_impl(path, cur, dry_run=True)
    assert len(dry.created_windows) == 1
    assert review_log_impl(path) == []  # a dry run never persists

    # Live apply: one window, two decisions -- first carries created_window,
    # second is an ordinary add into the same (now real) window.
    result = apply_curation_impl(path, cur)
    assert result.applied == len(result.plan) == 4
    assert len(result.created_windows) == 1
    wid = result.created_windows[0].window_id

    log = _log_shape(path)
    assert log == [(wid, "add", True), (wid, "add", False)]
    # Both requested frequencies landed in the ONE minted window (the fitted
    # value moves off the requested seed within the fit, so compare loosely).
    fitted = _fitted_by_window(path)[wid]
    assert len(fitted) == 2
    for requested in (freq1, freq2):
        assert min(abs(f - requested) for f in fitted) < 0.2


def test_batch_matches_sequential_result(stage5_multi_file, tmp_path):
    path = stage5_multi_file
    lo, hi = _widest_gap(path)
    if hi - lo < 4.0:
        pytest.skip("no gap wide enough to create a window into")
    freq1 = 0.5 * (lo + hi)

    probe_cur = _write_curation(tmp_path, "probe.csv", f"add,,{freq1:.6f},\n")
    probe = apply_curation_impl(path, probe_cur, dry_run=True)
    win_lo, win_hi = probe.created_windows[0].freq_range
    freq2 = win_lo + 0.15 * (win_hi - win_lo)

    batch_path = tmp_path / "batch.ftmw"
    seq_path = tmp_path / "sequential.ftmw"
    shutil.copy(path, batch_path)
    shutil.copy(path, seq_path)

    cur = _write_curation(
        tmp_path, "two_close.csv", f"add,,{freq1:.6f},\nadd,,{freq2:.6f},\n"
    )
    apply_curation_impl(batch_path, cur)

    refit_window_impl(str(seq_path), None, add=[freq1])
    refit_window_impl(str(seq_path), None, add=[freq2])

    batch_log = _log_shape(batch_path)
    seq_log = _log_shape(seq_path)
    # Both reduce to the same shape: [created, ordinary] on ONE window id.
    assert [kind for _wid, kind, _cw in batch_log] == [
        kind for _wid, kind, _cw in seq_log
    ]
    assert [cw for _wid, _kind, cw in batch_log] == [cw for _wid, _kind, cw in seq_log]
    batch_wids = {wid for wid, _kind, _cw in batch_log}
    seq_wids = {wid for wid, _kind, _cw in seq_log}
    assert len(batch_wids) == 1
    assert batch_wids == seq_wids  # the SAME real window id, not merely one each
    assert _fitted_by_window(batch_path) == _fitted_by_window(seq_path)


# ---------------------------------------------------------------------------
# What must NOT coalesce
# ---------------------------------------------------------------------------


def test_explicit_create_collision_still_raises(stage5_multi_file, tmp_path):
    path = stage5_multi_file
    lo, hi = _widest_gap(path)
    if hi - lo < 4.0:
        pytest.skip("no gap wide enough to create a window into")
    freq1 = 0.5 * (lo + hi)

    probe_cur = _write_curation(tmp_path, "probe.csv", f"add,,{freq1:.6f},\n")
    probe = apply_curation_impl(path, probe_cur, dry_run=True)
    win_lo, win_hi = probe.created_windows[0].freq_range
    freq2 = win_lo + 0.15 * (win_hi - win_lo)

    before = _fitted_by_window(path)
    cur = _write_curation(
        tmp_path,
        "two_explicit.csv",
        f"create,new,{freq1:.6f},\ncreate,new,{freq2:.6f},\n",
    )

    with pytest.raises(ValueError, match="already falls inside window"):
        apply_curation_impl(path, cur)

    # Refused means refused: nothing installed, nothing persisted.
    assert _fitted_by_window(path) == before
    assert review_log_impl(path) == []


def test_two_far_apart_adds_still_produce_two_creates(stage5_multi_file, tmp_path):
    path = stage5_multi_file
    lo, hi = _widest_gap(path)
    if hi - lo < 20.0:
        pytest.skip("no gap wide enough for two non-colliding created windows")
    freq1 = lo + 0.2 * (hi - lo)
    freq2 = lo + 0.8 * (hi - lo)

    cur = _write_curation(
        tmp_path, "two_far.csv", f"add,,{freq1:.6f},\nadd,,{freq2:.6f},\n"
    )

    dry = apply_curation_impl(path, cur, dry_run=True)
    assert len(dry.created_windows) == 2
    assert dry.created_windows[0].window_id != dry.created_windows[1].window_id

    result = apply_curation_impl(path, cur)
    assert result.applied == len(result.plan) == 4
    assert len(result.created_windows) == 2
    wids = {w.window_id for w in result.created_windows}
    assert len(wids) == 2

    log = _log_shape(path)
    assert [kind for _wid, kind, _cw in log] == ["add", "add"]
    assert [cw for _wid, _kind, cw in log] == [True, True]  # each its OWN create
    assert {wid for wid, _kind, _cw in log} == wids
