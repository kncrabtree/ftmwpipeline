"""W3 -- the implied create, for the one uncovered case.

W2 (``test_derived_window.py``) made a curation edit's window OPTIONAL where
it is a coordinate: an omitted ``add``/``remove`` target derives its window by
live-window coverage, and a frequency no live window covers is an ERROR. W3
turns that error into an implied create, for ``add`` only: an ``add`` whose
frequency no live window covers now mints the window it needs (or widens an
adjacent one, when the gap is too narrow), applies the add into it, and
records the whole thing as ONE decision-log entry -- the add itself, carrying
the structural consequence on its own evidence rather than a separate
``create_window`` entry. ``remove`` is UNCHANGED and permanent: a remove never
implies a create. A NAMED window that does not cover the frequency is still an
error, exactly as before -- the implied create fires only when the window was
OMITTED.

See ``scratch/intent-driven-windowing-plan.md``, W3.

Uses ``stage5_multi_file`` (several live fitted windows), like
``test_derived_window.py``. Writes only to pytest ``tmp_path``.
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
from ftmwpipeline._internal import stage6_impl as s6
from ftmwpipeline._internal.stage4_impl import load_windows_impl
from ftmwpipeline._internal.stage6_impl import (
    apply_curation_impl,
    refit_window_impl,
    review_log_impl,
    review_preview_impl,
    review_undo_impl,
)
from ftmwpipeline.cli.review_commands import cmd_review_edit
from ftmwpipeline.core.data_structures import FittedPeak, FittingResult
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5
from ftmwpipeline.io.stage6_review_serialization import load_stage6_review_from_file
from ftmwpipeline.pipeline import Pipeline

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


def _first_peak_excluding(path: Path, exclude_wid: int) -> Tuple[int, FittedPeak]:
    """Like :func:`_first_peak`, but skips *exclude_wid* -- for a test that
    needs an UNRELATED window's decision alongside an implied create's own."""
    sf = _load_spectrum_fit(path)
    for wf in sf.window_fits:
        if wf.window_id is None or int(wf.window_id) == exclude_wid:
            continue
        for p in wf.fitted_peaks:
            if p.peak_uid is not None:
                return int(wf.window_id), p
    pytest.skip("need a second live window with a fitted peak carrying peak_uid")


def _two_window_ids(path: Path) -> Tuple[int, int]:
    by_w = _fitted_by_window(path)
    wids = sorted(wid for wid, freqs in by_w.items() if freqs)
    if len(wids) < 2:
        pytest.skip("Need at least two fitted windows")
    return wids[0], wids[1]


def _uncovered_freq(path: Path) -> float:
    """A frequency well outside every window in the Stage 4 plan (live or
    dead) -- guaranteed uncovered by any live window a fortiori, and outside
    the analysis band too (2638's trim starts at 26500 MHz)."""
    plan = load_windows_impl(str(path))["plan"]
    lo = min(min(w.freq_range) for w in plan.windows)
    return lo - 1000.0


def _gap_anchor(path: Path) -> float:
    """A molecular frequency inside the analysis band but outside every
    PLANNED window (live or dead) -- reliably triggers ``mode="created"``
    rather than ``mode="widened"`` (the gap is wide enough to hold a new
    window outright). Mirrors ``test_review_preview.py::_anchor_in_a_gap``.
    """
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
    ascending by ``lo``."""
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


def _clear_freq_in_range(path: Path, wid: int, lo: float, hi: float) -> float:
    """A grid point in ``[lo, hi]`` farthest from every fitted peak *wid*
    already carries -- a safe second ``add`` target, never re-birthing an
    existing identity and never accidentally read as a split."""
    wf = _fitted_wf(path, wid)
    peaks = [float(p.frequency_mhz) for p in wf.fitted_peaks]
    if not peaks:
        return 0.5 * (lo + hi)
    grid = np.linspace(lo, hi, 512)[1:-1]
    dist = np.min(np.abs(grid[:, None] - np.asarray(peaks)[None, :]), axis=1)
    return float(grid[int(np.argmax(dist))])


def _write_curation(tmp_path: Path, name: str, text: str) -> str:
    p = tmp_path / name
    p.write_text(text)
    return str(p)


# ---------------------------------------------------------------------------
# The happy path: one call creates the window and adds the peak
# ---------------------------------------------------------------------------


def test_add_uncovered_creates_and_adds_in_one_call(stage5_multi_file):
    path = stage5_multi_file
    freq = _gap_anchor(path)
    before = _fitted_by_window(path)

    result = refit_window_impl(str(path), None, add=[freq])

    assert result.window_id not in before
    assert result.n_peaks_before == 0
    assert result.n_peaks_after == 1
    after = _fitted_by_window(path)
    assert len(after[result.window_id]) == 1


def test_records_exactly_one_add_decision_with_created_window_evidence(
    stage5_multi_file,
):
    path = stage5_multi_file
    freq = _gap_anchor(path)

    result = refit_window_impl(str(path), None, add=[freq])

    log = review_log_impl(path)
    assert len(log) == 1
    entry = log[0]
    assert entry.kind == "add"
    assert entry.window_id == result.window_id
    assert entry.evidence.get("inferred") is True
    created = entry.evidence.get("created_window")
    assert created is not None
    assert set(created) >= {
        "mode",
        "freq_min_mhz",
        "freq_max_mhz",
        "n_points",
        "n_contributors",
        "depends_on",
    }
    assert created["mode"] == "created"
    # No separate create_window entry -- one decision, not two.
    assert not any(e.kind == "create_window" for e in log)


def test_undo_removes_the_peak_and_the_window_leaves_no_stray_window(
    stage5_multi_file,
):
    path = stage5_multi_file
    freq = _gap_anchor(path)
    baseline = _fitted_by_window(path)

    result = refit_window_impl(str(path), None, add=[freq])
    wid = result.window_id
    entry = review_log_impl(path)[0]

    undo_result = review_undo_impl(str(path), [entry.order_index])

    assert undo_result.applied == 0  # nothing survives to replay
    assert review_log_impl(path) == []
    assert _fitted_by_window(path) == baseline
    # No stray empty window left behind: it is gone from the created-windows
    # overlay entirely, not merely emptied of its peak.
    created_ids = {
        int(w.window_id)
        for w in load_stage6_review_from_file(str(path)).created_windows
    }
    assert wid not in created_ids


# ---------------------------------------------------------------------------
# The orphan guard, extended to an implied create
# ---------------------------------------------------------------------------


def test_orphan_guard_refuses_undo_of_implying_add_with_a_dependent(
    stage5_multi_file,
):
    """add at X implies window W; a later add at Y resolves into W by live-
    window coverage; undoing the first alone must be refused, naming both --
    otherwise W vanishes out from under the second and the replay fails
    partway."""
    path = stage5_multi_file
    freq1 = _gap_anchor(path)

    result1 = refit_window_impl(str(path), None, add=[freq1])
    wid = result1.window_id
    entry1 = review_log_impl(path)[0]

    lo = entry1.evidence["created_window"]["freq_min_mhz"]
    hi = entry1.evidence["created_window"]["freq_max_mhz"]
    freq2 = _clear_freq_in_range(path, wid, lo, hi)

    result2 = refit_window_impl(str(path), None, add=[freq2])
    assert result2.window_id == wid  # resolved into the SAME (now live) window

    log = review_log_impl(path)
    assert len(log) == 2
    entry1_id, entry2_id = log[0].order_index, log[1].order_index

    with pytest.raises(ValueError, match="cannot undo") as excinfo:
        review_undo_impl(str(path), [entry1_id])

    msg = str(excinfo.value)
    assert str(wid) in msg
    assert str(entry2_id) in msg
    # Refused up front -- nothing rolled back partway.
    assert review_log_impl(path) == log


# ---------------------------------------------------------------------------
# Replay: an implied window's id is pinned across an unrelated undo
# ---------------------------------------------------------------------------


def test_replay_after_unrelated_undo_keeps_the_implied_windows_id(stage5_multi_file):
    path = stage5_multi_file
    freq1 = _gap_anchor(path)

    result1 = refit_window_impl(str(path), None, add=[freq1])
    wid = result1.window_id

    other_wid, other_target = _first_peak_excluding(path, wid)
    refit_window_impl(str(path), None, remove=[f"uid:{other_target.peak_uid}"])

    log = review_log_impl(path)
    assert len(log) == 2
    unrelated_id = log[1].order_index

    review_undo_impl(str(path), [unrelated_id])

    replayed_log = review_log_impl(path)
    assert len(replayed_log) == 1
    assert replayed_log[0].kind == "add"
    assert replayed_log[0].window_id == wid
    assert replayed_log[0].evidence.get("created_window") is not None
    assert wid in _fitted_by_window(path)
    assert other_wid in _fitted_by_window(path)


# ---------------------------------------------------------------------------
# remove is unaffected: still a permanent error
# ---------------------------------------------------------------------------


def test_remove_uncovered_frequency_still_errors(stage5_multi_file):
    path = stage5_multi_file
    bad = _uncovered_freq(path)
    before = _fitted_by_window(path)

    with pytest.raises(ValueError, match=re.escape(f"{bad:.4f}")):
        refit_window_impl(str(path), None, remove=[bad])

    assert _fitted_by_window(path) == before
    assert review_log_impl(path) == []


# ---------------------------------------------------------------------------
# A named window is still an assertion: the implied create fires only when
# the window column/argument was OMITTED.
# ---------------------------------------------------------------------------


def test_named_but_noncovering_window_on_add_still_errors(stage5_multi_file):
    path = stage5_multi_file
    wa, wb = _two_window_ids(path)
    add_freq = _fitted_by_window(path)[wb][0]  # covered, but by wb, not wa
    before = _fitted_by_window(path)

    with pytest.raises(ValueError, match="outside window"):
        refit_window_impl(str(path), wa, add=[add_freq])

    assert _fitted_by_window(path) == before
    assert review_log_impl(path) == []


# ---------------------------------------------------------------------------
# An anchor outside the analysis band refuses exactly as an explicit create
# ---------------------------------------------------------------------------


def test_anchor_outside_the_analysis_band_refuses(stage5_multi_file):
    path = stage5_multi_file
    before = _fitted_by_window(path)

    with pytest.raises(ValueError, match="outside the analysis band"):
        refit_window_impl(str(path), None, add=[1000.0])

    assert _fitted_by_window(path) == before
    assert review_log_impl(path) == []


# ---------------------------------------------------------------------------
# W1's widened-cascade behavior applies to an implied create too
# ---------------------------------------------------------------------------


def test_implied_create_widened_mode_still_cascades(stage5_multi_file, monkeypatch):
    path = stage5_multi_file
    step = _bin_width_mhz(path)
    live = _live_ranges(path)
    wid, _lo, hi = live[-1]
    dep_wid = live[0][0]
    if dep_wid == wid:
        pytest.skip("need at least two distinct live windows")
    anchor = hi + 2 * step

    orig_succs = s6._cascade_succs

    def fake_succs(window_fits, fit_window_map):
        d = orig_succs(window_fits, fit_window_map)
        d.setdefault(wid, set()).add(dep_wid)
        return d

    monkeypatch.setattr(s6, "_cascade_succs", fake_succs)

    orig_core = s6.refit_window_core
    calls: List[int] = []

    def spy_core(fit_ctx, fit_win, wf, **kwargs):
        calls.append(int(fit_win.window_id))
        return orig_core(fit_ctx, fit_win, wf, **kwargs)

    monkeypatch.setattr(s6, "refit_window_core", spy_core)

    result = refit_window_impl(str(path), None, add=[anchor])

    entry = review_log_impl(path)[0]
    assert entry.evidence["created_window"]["mode"] == "widened"
    assert result.window_id == wid
    assert dep_wid in calls


# ---------------------------------------------------------------------------
# The resolved plan shows the create (review apply --dry-run / preview)
# ---------------------------------------------------------------------------


def test_curation_dry_run_plan_shows_the_create_and_edit_pair(
    stage5_multi_file, tmp_path
):
    path = stage5_multi_file
    freq = _gap_anchor(path)
    cur = _write_curation(tmp_path, "implied.csv", f"add,,{freq},\n")

    result = apply_curation_impl(path, cur, dry_run=True)

    assert len(result.plan) == 2
    create_action, edit_action = result.plan
    assert create_action.kind == "create"
    assert create_action.implied_create is True
    assert edit_action.kind == "edit"
    assert edit_action.implied_create is True
    # A late-bound placeholder id, shared by both halves of the pair -- not
    # yet a real window id (nothing has run).
    assert create_action.window_id == edit_action.window_id
    assert create_action.window_id < 0
    assert edit_action.add == pytest.approx([freq])
    assert edit_action.remove == []
    # Nothing executed on a dry run.
    assert review_log_impl(path) == []


def test_review_preview_implied_create_executes_in_memory(stage5_multi_file, tmp_path):
    path = stage5_multi_file
    freq = _gap_anchor(path)
    cur = _write_curation(tmp_path, "implied.csv", f"add,,{freq},\n")

    preview = review_preview_impl(path, cur)

    assert len(preview.plan) == 2
    # windows is keyed by the REAL window id the (in-memory) create minted.
    assert len(preview.windows) == 1
    wid = next(iter(preview.windows))
    win = preview.windows[wid]
    assert win.n_peaks_before == 0
    assert win.n_peaks_after == 1
    assert win.action_indices == [0, 1]  # both the create and the edit
    # A preview never persists.
    assert review_log_impl(path) == []


# ---------------------------------------------------------------------------
# Cross-interface consistency: api / Pipeline / CLI
# ---------------------------------------------------------------------------


def test_implied_create_cross_interface(stage5_multi_file, tmp_path):
    paths = {k: tmp_path / f"implied_{k}.ftmw" for k in ("api", "pipe", "cli")}
    for p in paths.values():
        shutil.copy(stage5_multi_file, p)

    freq = _gap_anchor(paths["api"])

    ftmw.review_edit(paths["api"], add=[freq])
    Pipeline.open(paths["pipe"]).review_edit(add=[freq])
    rc = cmd_review_edit(
        argparse.Namespace(
            file_path=str(paths["cli"]), window=None, add=[freq], remove=None
        )
    )
    assert rc == 0

    ref = _fitted_by_window(paths["api"])
    assert _fitted_by_window(paths["pipe"]) == ref
    assert _fitted_by_window(paths["cli"]) == ref

    ref_log = review_log_impl(paths["api"])
    assert len(ref_log) == 1
    assert ref_log[0].evidence.get("created_window") is not None
    for k in ("pipe", "cli"):
        log = review_log_impl(paths[k])
        assert len(log) == 1
        assert log[0].kind == ref_log[0].kind == "add"
        assert log[0].evidence.get("created_window") is not None


@pytest.mark.integration
def test_implied_create_reinterpreted_as_an_edit_is_refused(
    stage5_multi_file, monkeypatch
):
    """An implied create ALWAYS records ``created_window``, or it refuses.

    ``_batch_apply_edit_action`` can reinterpret the implying ``add`` against
    the window it landed in (for ``mode="widened"``, a non-empty one) as an
    inferred merge/split. Those appliers record their own decision, which
    carries no ``created_window`` -- and without that evidence
    ``_decision_to_op`` cannot reissue the create before the add, so a replay
    would add into the window at its BASE PLAN width: a hard error partway
    through a replay (leaving the file rolled back with its log emptied, the
    failure ``df3f289`` fixed) or a silently wrong seed.

    So it is refused rather than recorded wrong. The configuration is not
    reachable on real data -- it needs the anchor within snap tolerance of an
    existing peak in the very window a too-narrow gap just widened, and
    ``min_window_half_width_points`` (32) is ~51x the snap tolerance in points
    (0.625), so a peak sits tens of snap-tolerances inside its own window's
    edge; measured on the real 2638 fit, zero configurations, the nearest 40x
    the snap tolerance away. It is forced here through the applier seam,
    because an invariant a replay depends on should hold by construction
    rather than by geometry that a later constant change could quietly move.
    """
    path = stage5_multi_file
    anchor = _gap_anchor(path)
    before = _fitted_by_window(path)
    n_log_before = len(review_log_impl(path))

    orig = s6._batch_apply_edit_action

    def reinterpreting(ctx, window_id, add, remove, **kwargs):
        result = orig(ctx, window_id, add, remove, **kwargs)
        # Stand in for an inferred split/merge applier recording its own
        # entry, which is what the refusal exists to catch.
        ctx.changeset.decisions.append(
            {
                "window_id": window_id,
                "frequency_mhz": float(add[0]),
                "kind": "split",
                "evidence": {"inferred": True},
            }
        )
        return result

    monkeypatch.setattr(s6, "_batch_apply_edit_action", reinterpreting)

    with pytest.raises(ValueError, match="implies creating a window"):
        refit_window_impl(str(path), None, add=[anchor])

    # Refused means refused: no window installed, no peak added, no decision.
    assert _fitted_by_window(path) == before
    assert len(review_log_impl(path)) == n_log_before
