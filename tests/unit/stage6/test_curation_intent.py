"""
Curation-intent inference (an ``edit`` action reinterpreted by the change it
makes to the peak set, not the verb the user typed).

A curated ``add`` beside an existing peak reads as a split of that peak into
two -- the identity model's blend rule then retires the parent's ``peak_uid``
(K goes from 1 to 2, so every member is a new entity). Removing the
components of one blend while adding their replacement reads as a merge.

See ``_infer_curation_intent`` / ``_batch_apply_edit_action`` in
``stage6_impl.py`` for the design; both ``refit_window_impl`` (the ``review
edit`` verb) and ``apply_curation_impl`` (a curation file's ``edit`` row)
reach the same inference through that one function.

Uses the ``stage5_multi_file`` fixture (several live fitted windows) from
``conftest.py``. Writes only to pytest ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import pytest

from ftmwpipeline._internal import stage6_impl as s6mod
from ftmwpipeline._internal.stage6_impl import (
    apply_curation_impl,
    refit_snap_tol_mhz_impl,
    refit_window_impl,
    review_log_impl,
)
from ftmwpipeline.core.data_structures import FittedPeak, FittingResult
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_spectrum_fit(path: Path):
    with h5py.File(str(path), "r") as h5f:
        return load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])


def _fitted_wf(path: Path, wid: int) -> FittingResult:
    sf = _load_spectrum_fit(path)
    return next(wf for wf in sf.window_fits if wf.window_id == wid)


def _first_peak(path: Path) -> Tuple[int, FittedPeak]:
    """(window_id, FittedPeak) of the first fitted peak found."""
    sf = _load_spectrum_fit(path)
    for wf in sf.window_fits:
        if wf.window_id is not None and wf.fitted_peaks:
            return int(wf.window_id), wf.fitted_peaks[0]
    pytest.skip("No fitted peaks in the built subset")


def _window_with_n_peaks(path: Path, n: int) -> FittingResult:
    sf = _load_spectrum_fit(path)
    wf = next(
        (
            w
            for w in sf.window_fits
            if w.window_id is not None and len(w.fitted_peaks) >= n
        ),
        None,
    )
    if wf is None:
        pytest.skip(f"Need a window with at least {n} fitted peaks")
    return wf


def _clear_add_freq(path: Path, wid: int, wf: FittingResult) -> float:
    """An in-window frequency far from every one of *wf*'s fitted peaks.

    Deliberately not the window center: Stage 4 builds a window around the
    detection that seeded it, so the center is the *birth position* of a
    peak already fitted there (mirrors ``test_curation.py``'s
    ``_clear_add_freq``; reimplemented locally to keep this file
    self-contained)."""
    from ftmwpipeline._internal.stage4_impl import load_windows_impl

    plan = load_windows_impl(str(path))["plan"]
    w = next(x for x in plan.windows if int(x.window_id) == wid)
    lo, hi = w.freq_range
    lo, hi = min(lo, hi), max(lo, hi)
    peaks = [float(p.frequency_mhz) for p in wf.fitted_peaks]
    if not peaks:
        return 0.5 * (lo + hi)
    grid = np.linspace(lo, hi, 512)[1:-1]
    dist = np.min(np.abs(grid[:, None] - np.asarray(peaks)[None, :]), axis=1)
    return float(grid[int(np.argmax(dist))])


def _expected_seed_uid(shared, wf: FittingResult, freq_mhz: float) -> int:
    """The ``peak_uid`` a peak seeded at *freq_mhz* in *wf*'s window frame
    would be stamped with -- ground truth computed from the same production
    primitives (``peak_uid_from_offset`` / the window's own frame) the
    applier itself stamps with, independent of ``_infer_curation_intent``."""
    lo, hi = wf.window.freq_range
    lo, hi = min(lo, hi), max(lo, hi)
    center = (lo + hi) / 2.0
    sideband = shared.fit_ctx.sideband
    s = s6mod.sideband_sign(sideband)
    offset = s * (float(freq_mhz) - center)
    return s6mod.peak_uid_from_offset(
        offset,
        center,
        sideband,
        shared.fit_ctx.probe_freq_mhz,
        shared.fit_ctx.active_ft.n_active,
        shared.fit_ctx.sample_dt_us,
    )


def _birth_freq_for_uid(shared, uid: int) -> float:
    """Invert ``peak_uid_from_offset``: the molecular frequency that stamps
    to (approximately -- ``uid`` is itself a rounded value) *uid* for a peak
    fit under *shared*'s frame.

    The inversion needs only the sideband/probe/active-FT frame -- the
    baseband frequency ``peak_uid_from_offset`` derives from ``offset_mhz``
    and ``center_mhz`` collapses, via the sideband-sign algebra, to a
    function of the molecular frequency alone (see ``PointMap`` in
    ``fitting/active_ft.py``), so the window's own ``freq_range`` never
    enters here."""
    sideband = shared.fit_ctx.sideband
    s = s6mod.sideband_sign(sideband)
    n = shared.fit_ctx.active_ft.n_active
    dt = shared.fit_ctx.sample_dt_us
    point = uid / 100.0
    f_bb = point / (n * dt)
    return float(shared.fit_ctx.probe_freq_mhz + s * f_bb)


# ---------------------------------------------------------------------------
# Headline: an add beside a fitted peak is a split, and it retires the
# parent's peak_uid (the defect commit d1de578 states deliberately).
# ---------------------------------------------------------------------------


def test_add_beside_a_peak_is_a_split_and_retires_parent_uid(stage5_multi_file):
    path = stage5_multi_file
    wid, parent = _first_peak(path)
    parent_uid = parent.peak_uid
    assert parent_uid is not None
    parent_freq = float(parent.frequency_mhz)

    wf_before = _fitted_wf(path, wid)
    n_before = len(wf_before.fitted_peaks)
    uids_before = {p.peak_uid for p in wf_before.fitted_peaks}

    snap_tol = refit_snap_tol_mhz_impl(path)
    requested = parent_freq + 0.3 * snap_tol

    refit_window_impl(str(path), wid, add=[requested])

    wf_after = _fitted_wf(path, wid)
    assert len(wf_after.fitted_peaks) == n_before + 1

    uids_after = {p.peak_uid for p in wf_after.fitted_peaks}
    assert parent_uid not in uids_after, "the parent's peak_uid must be retired"

    new_peaks = [p for p in wf_after.fitted_peaks if p.peak_uid not in uids_before]
    assert len(new_peaks) == 2, "a split replaces one peak with two"
    assert new_peaks[0].peak_uid != new_peaks[1].peak_uid
    assert parent_uid not in {p.peak_uid for p in new_peaks}

    log = review_log_impl(path)
    split_entries = [e for e in log if e.window_id == wid and e.kind == "split"]
    assert len(split_entries) == 1
    ev = split_entries[0].evidence
    assert ev.get("inferred") is True
    assert ev["requested_freq_mhz"] == pytest.approx(requested)


def test_curation_file_add_beside_a_peak_is_also_inferred_as_split(
    stage5_multi_file, tmp_path
):
    """The curation-file / batch-dispatcher entry point (``apply_curation_impl``
    -> an ``edit`` row) reaches the same inference as the verb path above."""
    path = stage5_multi_file
    wid, parent = _first_peak(path)
    parent_uid = parent.peak_uid
    parent_freq = float(parent.frequency_mhz)
    n_before = len(_fitted_wf(path, wid).fitted_peaks)

    snap_tol = refit_snap_tol_mhz_impl(path)
    requested = parent_freq + 0.3 * snap_tol

    cur = tmp_path / "curation.csv"
    cur.write_text(f"add,{wid},{requested},\n")
    apply_curation_impl(path, cur)

    wf_after = _fitted_wf(path, wid)
    assert len(wf_after.fitted_peaks) == n_before + 1
    assert parent_uid not in {p.peak_uid for p in wf_after.fitted_peaks}

    log = review_log_impl(path)
    kinds = [e.kind for e in log if e.window_id == wid]
    assert kinds == ["split"]


# ---------------------------------------------------------------------------
# Seeding: (parent position, requested position) -- not the symmetric
# straddle -- unless the requested position collides with the parent's own
# seed, in which case it falls back to the symmetric straddle.
# ---------------------------------------------------------------------------


def test_split_products_seed_at_parent_and_requested_position(stage5_multi_file):
    path = stage5_multi_file
    wid, parent = _first_peak(path)
    parent_freq = float(parent.frequency_mhz)

    wf_before = _fitted_wf(path, wid)
    uids_before = {p.peak_uid for p in wf_before.fitted_peaks}

    snap_tol = refit_snap_tol_mhz_impl(path)
    requested = parent_freq + 0.35 * snap_tol

    shared = s6mod._build_shared_fit_ctx(str(path))
    expected_parent_uid = _expected_seed_uid(shared, wf_before, parent_freq)
    expected_requested_uid = _expected_seed_uid(shared, wf_before, requested)

    acquisition_us = float(shared.fit_ctx.acquisition_us)
    resolution_mhz = 1.0 / acquisition_us
    symmetric_uids = {
        _expected_seed_uid(shared, wf_before, parent_freq - 0.5 * resolution_mhz),
        _expected_seed_uid(shared, wf_before, parent_freq + 0.5 * resolution_mhz),
    }
    assert {
        expected_parent_uid,
        expected_requested_uid,
    } != symmetric_uids, (
        "test setup must actually distinguish the two seeding strategies"
    )

    refit_window_impl(str(path), wid, add=[requested])

    wf_after = _fitted_wf(path, wid)
    new_uids = {
        p.peak_uid for p in wf_after.fitted_peaks if p.peak_uid not in uids_before
    }
    assert new_uids == {expected_parent_uid, expected_requested_uid}


def test_split_falls_back_to_symmetric_straddle_near_parents_own_seed(
    stage5_multi_file,
):
    path = stage5_multi_file
    shared = s6mod._build_shared_fit_ctx(str(path))
    snap_tol = refit_snap_tol_mhz_impl(path)

    sf = _load_spectrum_fit(path)
    chosen = None
    for wf in sf.window_fits:
        if wf.window_id is None or wf.window is None or wf.window.freq_range is None:
            continue
        for p in wf.fitted_peaks:
            if p.peak_uid is None:
                continue
            birth_freq = _birth_freq_for_uid(shared, p.peak_uid)
            if abs(birth_freq - float(p.frequency_mhz)) <= snap_tol:
                chosen = (int(wf.window_id), p, birth_freq)
                break
        if chosen:
            break
    if chosen is None:
        pytest.skip("No fitted peak with a recoverable in-tolerance birth position")

    wid, parent, birth_freq = chosen
    parent_uid = parent.peak_uid
    parent_freq = float(parent.frequency_mhz)

    wf_before = _fitted_wf(path, wid)
    uids_before = {p.peak_uid for p in wf_before.fitted_peaks}

    # Sanity: birth_freq really does round-trip to (within 1 of) the parent's
    # own stamped uid -- the precondition the fallback rule is keyed on.
    recomputed = _expected_seed_uid(shared, wf_before, birth_freq)
    assert abs(recomputed - parent_uid) <= 1

    refit_window_impl(str(path), wid, add=[birth_freq])

    wf_after = _fitted_wf(path, wid)
    new_peaks = [p for p in wf_after.fitted_peaks if p.peak_uid not in uids_before]
    assert len(new_peaks) == 2
    assert new_peaks[0].peak_uid != new_peaks[1].peak_uid
    assert parent_uid not in {p.peak_uid for p in wf_after.fitted_peaks}

    acquisition_us = float(shared.fit_ctx.acquisition_us)
    resolution_mhz = 1.0 / acquisition_us
    expected_symmetric = {
        _expected_seed_uid(shared, wf_before, parent_freq - 0.5 * resolution_mhz),
        _expected_seed_uid(shared, wf_before, parent_freq + 0.5 * resolution_mhz),
    }
    assert {p.peak_uid for p in new_peaks} == expected_symmetric


# ---------------------------------------------------------------------------
# Merge trigger: remove A, remove B (a blend), add C (between them).
# ---------------------------------------------------------------------------


def test_remove_pair_plus_between_add_is_an_inferred_merge(stage5_multi_file):
    path = stage5_multi_file
    wf = _window_with_n_peaks(path, 2)
    wid = int(wf.window_id)
    n_before = len(wf.fitted_peaks)

    fa, fb = sorted(float(p.frequency_mhz) for p in wf.fitted_peaks)[:2]
    uid_a = next(p.peak_uid for p in wf.fitted_peaks if float(p.frequency_mhz) == fa)
    uid_b = next(p.peak_uid for p in wf.fitted_peaks if float(p.frequency_mhz) == fb)
    span = fb - fa
    assert span > 0

    # Real Stage 5 peaks in one window are usually far apart relative to the
    # (tiny) snap tolerance -- widen it for this action only, so the pair
    # reads as "the components of one feature" without depending on the
    # fixture happening to contain a genuine unresolved blend.
    test_tol = span * 1.05 + 1e-9
    between = 0.5 * (fa + fb)

    refit_window_impl(
        str(path), wid, add=[between], remove=[fa, fb], snap_tol_mhz=test_tol
    )

    wf_after = _fitted_wf(path, wid)
    assert len(wf_after.fitted_peaks) == n_before - 1

    uids_after = {p.peak_uid for p in wf_after.fitted_peaks}
    assert uid_a not in uids_after
    assert uid_b not in uids_after

    log = review_log_impl(path)
    merge_entries = [e for e in log if e.window_id == wid and e.kind == "merge"]
    assert len(merge_entries) == 1
    ev = merge_entries[0].evidence
    assert ev.get("inferred") is True
    assert ev["requested_freq_mhz"] == pytest.approx(between)
    assert sorted(ev["merged_from"]) == sorted([fa, fb])


# ---------------------------------------------------------------------------
# Negative cases: what stops this from over-firing.
# ---------------------------------------------------------------------------


def test_clear_add_stays_a_plain_add(stage5_multi_file):
    path = stage5_multi_file
    wid, _parent = _first_peak(path)
    wf_before = _fitted_wf(path, wid)
    n_before = len(wf_before.fitted_peaks)
    uids_before = {p.peak_uid for p in wf_before.fitted_peaks}

    clear_freq = _clear_add_freq(path, wid, wf_before)
    refit_window_impl(str(path), wid, add=[clear_freq])

    wf_after = _fitted_wf(path, wid)
    assert len(wf_after.fitted_peaks) == n_before + 1
    # Every pre-existing identifier survives -- nothing was retired.
    assert uids_before <= {p.peak_uid for p in wf_after.fitted_peaks}

    log = review_log_impl(path)
    kinds = [e.kind for e in log if e.window_id == wid]
    assert kinds == ["add"]


def test_remove_far_apart_pair_stays_a_plain_edit(stage5_multi_file):
    path = stage5_multi_file
    wf = _window_with_n_peaks(path, 2)
    wid = int(wf.window_id)
    n_before = len(wf.fitted_peaks)

    fa, fb = sorted(float(p.frequency_mhz) for p in wf.fitted_peaks)[:2]
    snap_tol = refit_snap_tol_mhz_impl(path)
    if (fb - fa) <= snap_tol:
        pytest.skip("The two peaks are already within snap tolerance")

    clear_freq = _clear_add_freq(path, wid, wf)

    refit_window_impl(str(path), wid, add=[clear_freq], remove=[fa, fb])

    wf_after = _fitted_wf(path, wid)
    assert len(wf_after.fitted_peaks) == n_before - 1

    log = review_log_impl(path)
    kinds = sorted(e.kind for e in log if e.window_id == wid)
    assert kinds == ["add", "remove", "remove"]


# ---------------------------------------------------------------------------
# Mixed action: one add beside a peak (split) and one add well clear (plain
# add), resolved in the same batch.
# ---------------------------------------------------------------------------


def test_mixed_beside_and_clear_adds_does_both_in_one_batch(stage5_multi_file):
    path = stage5_multi_file
    wid, parent = _first_peak(path)
    parent_uid = parent.peak_uid
    parent_freq = float(parent.frequency_mhz)

    wf_before = _fitted_wf(path, wid)
    n_before = len(wf_before.fitted_peaks)

    snap_tol = refit_snap_tol_mhz_impl(path)
    beside = parent_freq + 0.3 * snap_tol
    clear_freq = _clear_add_freq(path, wid, wf_before)

    refit_window_impl(str(path), wid, add=[beside, clear_freq])

    wf_after = _fitted_wf(path, wid)
    assert len(wf_after.fitted_peaks) == n_before + 2
    assert parent_uid not in {p.peak_uid for p in wf_after.fitted_peaks}

    log = review_log_impl(path)
    kinds = sorted(e.kind for e in log if e.window_id == wid)
    assert kinds == ["add", "split"]

    add_entry = next(e for e in log if e.window_id == wid and e.kind == "add")
    assert "inferred" not in add_entry.evidence
