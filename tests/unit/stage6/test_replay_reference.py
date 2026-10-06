"""
The full-replay reference (``replay_full``) and its bitwise state digest.

``replay_full`` recomputes the curated state of a decision log from the
automatic-fit baseline, in memory, reading no curated state. It is the oracle
every Stage 6 write is checked against, so its first property is that it agrees
bit for bit with what the write path persists: an undo replays the surviving
rows from the baseline, and the state it writes must have the digest of
``replay_full`` of those rows -- computed *before* the undo, on the file as it
was, which also shows the reference reads only the baseline and the static
inputs.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal import stage6_impl as s6
from ftmwpipeline._internal.replay_reference import (
    differing_parts,
    persisted_state_digest,
    replay_full,
    state_digest,
)
from ftmwpipeline.core.data_structures import FittingResult
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5

pytestmark = [pytest.mark.integration]


def _fits(path: Path) -> Dict[int, FittingResult]:
    with h5py.File(str(path), "r") as h5f:
        sf = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    return {int(wf.window_id): wf for wf in sf.window_fits if wf.window_id is not None}


def _freqs(path: Path, wid: int) -> List[float]:
    return sorted(float(p.frequency_mhz) for p in _fits(path)[wid].fitted_peaks)


def _clear(path: Path, wid: int) -> float:
    """The in-window position furthest from every fitted peak."""
    wf = _fits(path)[wid]
    assert wf.window is not None
    lo, hi = sorted(float(v) for v in wf.window.freq_range)
    ps = [float(p.frequency_mhz) for p in wf.fitted_peaks]
    return max(
        (lo + (hi - lo) * t / 40 for t in range(4, 37)),
        key=lambda x: min((abs(x - q) for q in ps), default=1e9),
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_undo_matches_reference(
    path: Path, tmp_path: Path, undo: Sequence[int], label: str
) -> None:
    """Undo the serials *undo* on a copy of *path*; the persisted state must
    equal the reference replay of the surviving rows, computed before the
    undo."""
    work = tmp_path / f"undo_{label}.ftmw"
    shutil.copy(path, work)
    log = ftmw.review_log(work)
    survivors = [e for e in log if e.serial not in set(undo)]
    before = _sha(work)
    reference = replay_full(work, survivors)
    assert _sha(work) == before, "replay_full wrote to the file"
    ftmw.review_undo(work, list(undo))
    assert state_digest(reference) == persisted_state_digest(work), differing_parts(
        reference, work
    )
    work.unlink()


def _multi_window_ids(path: Path, n: int, min_peaks: int) -> List[int]:
    fits = _fits(path)
    wids = [w for w in sorted(fits) if len(fits[w].fitted_peaks) >= min_peaks]
    if len(wids) < n:
        pytest.skip(f"fixture has fewer than {n} windows with {min_peaks}+ peaks")
    return wids[:n]


def test_reference_equals_the_undo_replay(stage5_multi_file, tmp_path):
    """Create-free logs of every kind of row: the undo persists exactly the
    reference replay of what survives, bit for bit."""
    fp = stage5_multi_file
    w1, w2, w3 = _multi_window_ids(fp, 3, 2)
    tol = s6.refit_snap_tol_mhz_impl(str(fp))

    ftmw.review_edit(fp, w1, add=[_clear(fp, w1)], frame="raw")
    ftmw.review_edit(fp, w2, remove=[_freqs(fp, w2)[0]], frame="raw")
    ftmw.review_accept(fp, w3)
    ftmw.review_apply(
        fp,
        actions=[
            {"action": "add", "window_id": w3, "freq_mhz": _clear(fp, w3)},
            {"action": "remove", "window_id": w1, "freq_mhz": _freqs(fp, w1)[-1]},
        ],
        frame="raw",
    )
    # An inferred split: an add within snap tolerance of a fitted peak.
    ftmw.review_edit(fp, w2, add=[_freqs(fp, w2)[-1] + 0.5 * tol], frame="raw")
    ftmw.review_accept(fp, w1)
    n = len(ftmw.review_log(fp))

    _assert_undo_matches_reference(fp, tmp_path, [n - 1], "last")
    _assert_undo_matches_reference(fp, tmp_path, [0], "first")
    _assert_undo_matches_reference(fp, tmp_path, [1, 3], "two")
    # Only the bare accepts survive: the undo replays them one by one.
    accepts = {e.serial for e in ftmw.review_log(fp) if e.kind == "accept"}
    _assert_undo_matches_reference(
        fp, tmp_path, [i for i in range(n) if i not in accepts], "accepts"
    )


def test_reference_of_the_empty_log_is_the_reviewed_baseline(
    stage5_multi_file, tmp_path
):
    """Undoing every row persists the reference of the empty log: the
    automatic fit with a fresh ``review run`` review."""
    fp = stage5_multi_file
    (w1,) = _multi_window_ids(fp, 1, 2)
    ftmw.review_edit(fp, w1, add=[_clear(fp, w1)], frame="raw")
    _assert_undo_matches_reference(fp, tmp_path, [0], "all")


def test_the_reference_reads_no_curated_fit(stage5_multi_file, tmp_path):
    """The fitted plan and the spur catalog come from the baseline: with the
    curated ``/stage5_fitting`` deleted the reference is unchanged."""
    fp = stage5_multi_file
    w1, w2 = _multi_window_ids(fp, 2, 2)
    ftmw.review_edit(fp, w1, add=[_clear(fp, w1)], frame="raw")
    ftmw.review_accept(fp, w2)
    log = ftmw.review_log(fp)
    reference = state_digest(replay_full(fp, log))

    stripped = tmp_path / "stripped.ftmw"
    shutil.copy(fp, stripped)
    with h5py.File(str(stripped), "a") as h5f:
        del h5f["stage5_fitting"]
    assert state_digest(replay_full(stripped, log)) == reference


def test_the_digest_sees_one_changed_float(stage5_multi_file):
    """The digest is bitwise: a one-ulp change in one fitted frequency moves
    it, and :func:`differing_parts` names the column."""
    fp = stage5_multi_file
    ftmw.review_run(fp)
    state = replay_full(fp, [])
    assert state_digest(state) == persisted_state_digest(fp)
    import math

    peak = next(p for wf in state.spectrum_fit.window_fits for p in wf.fitted_peaks)
    peak.frequency_mhz = math.nextafter(float(peak.frequency_mhz), math.inf)
    assert state_digest(state) != persisted_state_digest(fp)
    assert "fit.peaks.frequency_mhz" in differing_parts(state, fp)


# ---------------------------------------------------------------------------
# The cascade graph is one definition, fixed by the lineage
# ---------------------------------------------------------------------------


def test_every_write_path_and_the_reference_cascade_over_the_same_graph(
    stage5_multi_file, monkeypatch
):
    """The interactive verbs, ``apply``, the undo's replay and the reference
    all reach the cascade through ``_cascade_batch``, which hands it the one
    plan-derived graph: with no creates, the base graph itself."""
    fp = stage5_multi_file
    w1, w2 = _multi_window_ids(fp, 2, 2)
    expected = dict(s6._build_shared_fit_ctx(str(fp)).base_cascade_sources)
    seen: Dict[str, List[Dict[int, Tuple[int, ...]]]] = {}
    where = ["edit"]
    orig = s6._cascade_refit_dependents

    def spy(**kwargs):
        seen.setdefault(where[0], []).append(dict(kwargs["sources"]))
        return orig(**kwargs)

    monkeypatch.setattr(s6, "_cascade_refit_dependents", spy)
    ftmw.review_edit(fp, w1, add=[_clear(fp, w1)], frame="raw")
    where[0] = "apply"
    ftmw.review_apply(
        fp,
        actions=[{"action": "remove", "window_id": w2, "freq_mhz": _freqs(fp, w2)[0]}],
        frame="raw",
    )
    log = ftmw.review_log(fp)
    where[0] = "reference"
    replay_full(fp, log)
    where[0] = "undo"
    ftmw.review_undo(fp, [0])
    assert set(seen) == {"edit", "apply", "reference", "undo"}
    for name, graphs in seen.items():
        assert graphs and all(g == expected for g in graphs), name


def test_the_base_graph_is_a_lineage_constant(stage5_multi_file):
    """Computed from the fitted plan and the undo baseline: curating, undoing
    and starting a curation (which takes the baseline) leave it as it was,
    and its keys are the windows the automatic fit holds."""
    fp = stage5_multi_file
    w1, w2 = _multi_window_ids(fp, 2, 2)
    before = dict(s6._build_shared_fit_ctx(str(fp)).base_cascade_sources)
    with h5py.File(str(fp), "r") as h5f:
        assert s6.STAGE5_BASELINE_GROUP not in h5f
    assert set(before) == set(_fits(fp))

    ftmw.review_edit(fp, w1, add=[_clear(fp, w1)], frame="raw")
    ftmw.review_edit(fp, w2, remove=[_freqs(fp, w2)[0]], frame="raw")
    with h5py.File(str(fp), "r") as h5f:
        assert s6.STAGE5_BASELINE_GROUP in h5f
    assert dict(s6._build_shared_fit_ctx(str(fp)).base_cascade_sources) == before
    ftmw.review_undo(fp, [0])
    assert dict(s6._build_shared_fit_ctx(str(fp)).base_cascade_sources) == before


# ---------------------------------------------------------------------------
# 655, the dense cascade hub (slow: a full fixture build)
# ---------------------------------------------------------------------------

_SOURCE_655 = Path("examples/blackchirp_data/655")


@pytest.fixture(scope="module")
def built_655(tmp_path_factory) -> Path:
    if not _SOURCE_655.exists():
        pytest.skip("Experiment 655 data not available")
    out = tmp_path_factory.mktemp("replay_reference_655") / "655.ftmw"
    result = ftmw.run_pipeline(
        _SOURCE_655, output=out, trim=(26500.0, 40000.0), detect_start=True
    )
    if result.get("status") != "success":
        pytest.skip(f"655 build stopped: {result.get('error')}")
    return out


@pytest.mark.slow
def test_reference_equals_the_undo_replay_on_655(built_655, tmp_path):
    """On 655, through the 429 hub (32 dependents): edits on the hub and its
    dependents, an inferred split, a merge, a batch, a bare accept. Removes on
    a dependent are made before the hub edits that cascade into it, so the
    log replays (a remove recorded against a cascaded fit can name a peak the
    one-batch replay does not hold -- the defect uid addressing removes)."""
    fp = tmp_path / "655.ftmw"
    shutil.copy(built_655, fp)
    fits = _fits(fp)
    succs = s6._cascade_succs(s6._build_shared_fit_ctx(str(fp)).base_cascade_sources)
    hub = max(succs, key=lambda w: len(succs[w]))
    dependents = sorted(succs[hub])
    touched = {p for p, ds in succs.items() if ds} | {
        d for ds in succs.values() for d in ds
    }
    iso = [
        w for w in sorted(fits) if w not in touched and len(fits[w].fitted_peaks) >= 3
    ]
    if len(dependents) < 2 or len(iso) < 3:
        pytest.skip("655 build has no cascade hub with isolated windows beside it")
    d1, d2 = dependents[:2]
    tol = s6.refit_snap_tol_mhz_impl(str(fp))

    ftmw.review_edit(fp, d1, remove=[_freqs(fp, d1)[0]], frame="raw")
    strongest = max(fits[hub].fitted_peaks, key=lambda p: float(p.snr or 0.0))
    ftmw.review_edit(fp, hub, remove=[float(strongest.frequency_mhz)], frame="raw")
    ftmw.review_edit(fp, d2, add=[_clear(fp, d2)], frame="raw")
    ftmw.review_accept(fp, iso[0])
    ftmw.review_edit(fp, iso[1], add=[_freqs(fp, iso[1])[-1] + 0.5 * tol], frame="raw")
    ftmw.review_apply(
        fp,
        actions=[
            {"action": "add", "window_id": d1, "freq_mhz": _clear(fp, d1)},
            {
                "action": "remove",
                "window_id": iso[2],
                "freq_mhz": _freqs(fp, iso[2])[0],
            },
        ],
        frame="raw",
    )
    p = _freqs(fp, iso[0])
    s6.merge_peaks_impl(str(fp), iso[0], p[:2], frame="raw")
    ftmw.review_edit(fp, hub, add=[_clear(fp, hub)], frame="raw")
    ftmw.review_accept(fp, iso[2])
    n = len(ftmw.review_log(fp))

    _assert_undo_matches_reference(fp, tmp_path, [n - 1], "last")
    _assert_undo_matches_reference(fp, tmp_path, [1], "hub")
    _assert_undo_matches_reference(fp, tmp_path, [0, 4], "two")


# ---------------------------------------------------------------------------
# The plan-derived cascade graph (slow: full fixture builds)
# ---------------------------------------------------------------------------


def _fit_derived_sources(
    window_fits: Sequence[FittingResult],
    fit_window_map: Dict[int, object],
    unavailable: Sequence[int],
) -> Dict[int, Tuple[int, ...]]:
    """The cascade graph the reference used before it was plan-derived: each
    window's non-edge-free frozen primaries in the fits being cascaded, in
    order of first appearance (every frozen primary for a window with no
    usable plan entry)."""
    fitted = {int(wf.window_id) for wf in window_fits if wf.window_id is not None}
    out: Dict[int, Tuple[int, ...]] = {}
    for wf in window_fits:
        if wf.window_id is None:
            continue
        d = int(wf.window_id)
        nonef = s6._non_edge_free_primaries(
            None if d in set(unavailable) else fit_window_map.get(d)
        )
        primaries: Dict[int, None] = {}
        for key, entry in wf.fixed_parameters.items():
            if not key.startswith("frozen_peak_"):
                continue
            p = int(entry["primary_window_id"])
            if p in fitted and p != d and (nonef is None or p in nonef):
                primaries.setdefault(p, None)
        out[d] = tuple(primaries)
    return out


def _edges_in_effect(path: Path) -> Dict[int, set]:
    """``source -> {dependent}`` read off the curated fits' frozen entries."""
    fits = _fits(path)
    plan = {int(w.window_id): w for w in s6.effective_window_plan(str(path)).windows}
    sources = _fit_derived_sources(list(fits.values()), plan, ())
    return s6._cascade_succs(sources)


def _hub(path: Path) -> Tuple[int, List[int]]:
    succs = s6._cascade_succs(s6._build_shared_fit_ctx(str(path)).base_cascade_sources)
    hub = max(succs, key=lambda w: len(succs[w]))
    return hub, sorted(succs[hub])


def _step1_refresh(
    wf: FittingResult,
    sources: Sequence[int],
    fit_window_map: Dict[int, object],
    fit_map: Dict[int, FittingResult],
    min_freeze_snr: float,
) -> None:
    """The step-1 cascade refresh, verbatim but for ignoring *sources*: a
    window's sources are the non-edge-free primaries of its own frozen
    entries, in order of first appearance."""
    d = int(wf.window_id)
    nonef = s6._non_edge_free_primaries(fit_window_map.get(d))
    non_frozen: Dict[str, Dict] = {}
    preserved_edge_free: List[Dict] = []
    src_wids: List[int] = []
    for key, entry in wf.fixed_parameters.items():
        if not key.startswith("frozen_peak_"):
            non_frozen[key] = entry
            continue
        primary = int(entry["primary_window_id"])
        if nonef is None or primary in nonef:
            if primary not in src_wids:
                src_wids.append(primary)
        else:
            preserved_edge_free.append(entry)
    rebuilt: List[Dict] = []
    for primary in src_wids:
        pwf = fit_map.get(primary)
        if pwf is None:
            continue
        for pk in sorted(pwf.fitted_peaks, key=lambda q: float(q.frequency_mhz)):
            if float(pk.snr or 0.0) < min_freeze_snr:
                continue
            rebuilt.append(
                {
                    "peak_index": -1,
                    "primary_window_id": primary,
                    "frequency_mhz": float(pk.frequency_mhz),
                    "amplitude": float(pk.amplitude),
                    "phase": float(pk.phase) if pk.phase is not None else 0.0,
                    "freeze_eligible": True,
                }
            )
    frozen = preserved_edge_free + rebuilt
    rekeyed = {f"frozen_peak_{i}": e for i, e in enumerate(frozen)}
    wf.fixed_parameters = {**non_frozen, **rekeyed}


@pytest.mark.slow
@pytest.mark.parametrize("built", ["built_655", "built_1019"])
def test_the_plan_derived_graph_replays_like_step_1(
    built, request, tmp_path, monkeypatch
):
    """On a create-free log that empties no edge, on a fixture whose automatic
    fit's frozen entries name exactly the plan's live sources, the
    plan-derived reference is bitwise equal to the step-1 one: the graph read
    off the fits being cascaded and the refresh that reads each window's own
    frozen entries (both reproduced here, so a change in either side shows)."""
    fp = tmp_path / "fx.ftmw"
    shutil.copy(request.getfixturevalue(built), fp)
    fits = _fits(fp)
    shared = s6._build_shared_fit_ctx(str(fp))
    plan = {int(w.window_id): w for w in shared.base_plan.windows}
    assert shared.base_cascade_sources == _fit_derived_sources(
        list(fits.values()), plan, shared.unavailable_window_ids
    )
    # A hub that keeps a line above min_freeze_snr after losing its strongest,
    # so the step-1 graph loses no edge to the log.
    succs = s6._cascade_succs(shared.base_cascade_sources)
    min_snr = shared.min_freeze_snr
    strong = {
        w: sorted(
            (p for p in wf.fitted_peaks if float(p.snr or 0.0) >= min_snr),
            key=lambda p: float(p.snr or 0.0),
        )
        for w, wf in fits.items()
    }
    hub = max((w for w in succs if len(strong[w]) >= 2), key=lambda w: len(succs[w]))
    dependents = sorted(succs[hub])
    assert dependents
    d1 = next((d for d in dependents if len(fits[d].fitted_peaks) >= 2), dependents[0])

    ftmw.review_edit(
        fp, hub, remove=[float(strong[hub][-1].frequency_mhz)], frame="raw"
    )
    ftmw.review_edit(fp, d1, add=[_clear(fp, d1)], frame="raw")
    ftmw.review_edit(fp, hub, add=[_clear(fp, hub)], frame="raw")
    log = ftmw.review_log(fp)
    plan_derived = state_digest(replay_full(fp, log))

    orig = s6._cascade_refit_dependents
    calls: List[int] = []

    def step1_graph(**kwargs):
        kwargs["sources"] = _fit_derived_sources(
            kwargs["spectrum_fit"].window_fits,
            kwargs["fit_window_map"],
            kwargs["unavailable_window_ids"],
        )
        cascaded = orig(**kwargs)
        calls.extend(cascaded)
        return cascaded

    monkeypatch.setattr(s6, "_cascade_refit_dependents", step1_graph)
    monkeypatch.setattr(s6, "_refresh_frozen_from_sources", _step1_refresh)
    assert state_digest(replay_full(fp, log)) == plan_derived
    assert set(dependents) <= set(calls)


@pytest.mark.slow
def test_an_edge_an_edit_emptied_carries_the_next_edit_on_655(
    built_655, tmp_path, monkeypatch
):
    """Removing every line of 429 that clears ``min_freeze_snr`` leaves its 32
    dependents no skirt from it; re-adding its strongest line must reach all
    of them again. A graph read from the curated fits had lost those edges
    after the first edit (140 edges in effect fell to 108)."""
    fp = tmp_path / "655.ftmw"
    shutil.copy(built_655, fp)
    fits = _fits(fp)
    hub, dependents = _hub(fp)
    assert hub == 429 and len(dependents) == 32
    shared = s6._build_shared_fit_ctx(str(fp))
    base_graph = dict(shared.base_cascade_sources)
    assert sum(len(v) for v in base_graph.values()) == 140
    min_snr = shared.min_freeze_snr
    before = _edges_in_effect(fp)
    assert before[hub] == set(dependents)
    assert sum(map(len, before.values())) == 140
    strong = [p for p in fits[hub].fitted_peaks if float(p.snr or 0.0) >= min_snr]
    strongest = max(strong, key=lambda p: float(p.snr or 0.0))

    ftmw.review_edit(
        fp, hub, remove=[float(p.frequency_mhz) for p in strong], frame="raw"
    )
    assert _edges_in_effect(fp).get(hub, set()) == set()
    # The plan-derived graph does not move with the curated fits.
    assert dict(s6._build_shared_fit_ctx(str(fp)).base_cascade_sources) == base_graph

    orig_core = s6.refit_window_core
    refit: List[int] = []

    def spy_core(fit_ctx, fit_win, wf, **kwargs):
        refit.append(int(fit_win.window_id))
        return orig_core(fit_ctx, fit_win, wf, **kwargs)

    monkeypatch.setattr(s6, "refit_window_core", spy_core)
    ftmw.review_edit(fp, hub, add=[float(strongest.frequency_mhz)], frame="raw")
    assert set(dependents) <= set(refit)
    after = _edges_in_effect(fp)
    assert after[hub] == set(dependents)
    assert sum(map(len, after.values())) == 140
    assert dict(s6._build_shared_fit_ctx(str(fp)).base_cascade_sources) == base_graph


_SOURCE_1019 = Path("examples/blackchirp_data/1019")


@pytest.fixture(scope="module")
def built_1019(tmp_path_factory) -> Path:
    if not _SOURCE_1019.exists():
        pytest.skip("Experiment 1019 data not available")
    out = tmp_path_factory.mktemp("replay_reference_1019") / "1019.ftmw"
    result = ftmw.run_pipeline(
        _SOURCE_1019, output=out, trim=(26500.0, 40000.0), detect_start=True
    )
    if result.get("status") != "success":
        pytest.skip(f"1019 build stopped: {result.get('error')}")
    return out


@pytest.mark.slow
def test_an_accepted_thaws_primary_cascades_into_its_dependent_on_1019(
    built_1019, tmp_path, monkeypatch
):
    """1019 accepts one Stage 5 thaw, of window 58's line into window 53. 53
    reads 58 in the plan and keeps 58's line as a frozen contributor, so 58 ->
    53 is an edge, and an edit of 58 refits 53 against 58's edited fit."""
    fp = tmp_path / "1019.ftmw"
    shutil.copy(built_1019, fp)
    fits = _fits(fp)
    shared = s6._build_shared_fit_ctx(str(fp))
    thaws = [
        (e.dependent_window_id, e.primary_window_id)
        for wf in fits.values()
        for e in wf.thaw_events
        if e.accepted
    ]
    if (53, 58) not in thaws:
        pytest.skip(f"this 1019 build accepts no 58 -> 53 thaw (accepted: {thaws})")
    assert 58 in shared.base_cascade_sources[53]

    orig_core = s6.refit_window_core
    refit: List[int] = []

    def spy_core(fit_ctx, fit_win, wf, **kwargs):
        refit.append(int(fit_win.window_id))
        return orig_core(fit_ctx, fit_win, wf, **kwargs)

    monkeypatch.setattr(s6, "refit_window_core", spy_core)
    ftmw.review_edit(fp, 58, add=[_clear(fp, 58)], frame="raw")
    assert 53 in refit
    skirt = sorted(
        float(e["frequency_mhz"])
        for k, e in _fits(fp)[53].fixed_parameters.items()
        if k.startswith("frozen_peak_") and int(e["primary_window_id"]) == 58
    )
    min_snr = shared.min_freeze_snr
    assert skirt == sorted(
        float(p.frequency_mhz)
        for p in _fits(fp)[58].fitted_peaks
        if float(p.snr or 0.0) >= min_snr
    )
