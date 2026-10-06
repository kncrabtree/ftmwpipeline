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
from typing import Dict, List, Sequence

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
    plan = {int(w.window_id): w for w in s6.effective_window_plan(str(fp)).windows}
    succs = s6._cascade_succs(list(fits.values()), plan)
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
