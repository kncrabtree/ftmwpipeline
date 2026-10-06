"""
The decision serial: a decision's stable id.

A serial is minted once, when the decision is recorded, from the review's
``next_serial`` high-water mark. It is never reused within the lineage (an undo
never lowers the mark) and never renumbered: ``review undo`` takes serials, an
action's ``action_index`` is the serial of its first row, and a peak's
``derivation`` holds the serial of the decision that created it. Rows are
immutable -- an undo keeps the surviving rows verbatim and recomputes only
their positions -- so undoing one decision changes nothing that names another.
``fit run`` starts a new lineage: the review is dropped and serials restart.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import Pipeline
from ftmwpipeline._internal.stage6_impl import (
    ACTION_INDEX_EVIDENCE_KEY,
    LINEAGE_ID_ATTR,
    STAGE5_BASELINE_GROUP,
    review_undo_impl,
)
from ftmwpipeline.cli.main import main
from ftmwpipeline.core.data_structures import ENGINE_VERSION, FittingResult
from ftmwpipeline.file_manager import NotFoundValueError
from ftmwpipeline.io.fitting_serialization import load_spectrum_fit_from_hdf5

pytestmark = [pytest.mark.integration]


def _fits(path: Path) -> Dict[int, FittingResult]:
    with h5py.File(str(path), "r") as h5f:
        sf = load_spectrum_fit_from_hdf5(h5f["stage5_fitting"])
    return {int(wf.window_id): wf for wf in sf.window_fits if wf.window_id is not None}


def _clear(path: Path, wid: int) -> float:
    wf = _fits(path)[wid]
    assert wf.window is not None
    lo, hi = sorted(float(v) for v in wf.window.freq_range)
    ps = [float(p.frequency_mhz) for p in wf.fitted_peaks]
    return max(
        (lo + (hi - lo) * t / 40 for t in range(4, 37)),
        key=lambda x: min((abs(x - q) for q in ps), default=1e9),
    )


def _windows(path: Path, n: int) -> List[int]:
    fits = _fits(path)
    wids = [w for w in sorted(fits) if fits[w].fitted_peaks]
    if len(wids) < n:
        pytest.skip(f"fixture has fewer than {n} fitted windows")
    return wids[:n]


def _peak_state(path: Path) -> Dict[int, List[Tuple[object, ...]]]:
    """Every window's peaks, bit for bit, with their uid and derivation."""
    return {
        wid: [
            (
                float(p.frequency_mhz),
                float(p.amplitude),
                p.peak_uid,
                p.derivation,
                p.origin,
            )
            for p in wf.fitted_peaks
        ]
        for wid, wf in _fits(path).items()
    }


def _next_serial(path: Path) -> int:
    with h5py.File(str(path), "r") as h5f:
        return int(h5f["stage6_review"].attrs["next_serial"])


def test_serials_follow_recording_and_stamp_the_engine(stage5_multi_file):
    fp = stage5_multi_file
    w1, w2 = _windows(fp, 2)
    ftmw.review_edit(fp, w1, add=[_clear(fp, w1)], frame="raw")
    ftmw.review_accept(fp, w2)
    log = ftmw.review_log(fp)
    assert [(e.order_index, e.serial) for e in log] == [(0, 0), (1, 1)]
    assert [e.evidence[ACTION_INDEX_EVIDENCE_KEY] for e in log] == [0, 1]
    assert _next_serial(fp) == 2
    with h5py.File(str(fp), "r") as h5f:
        assert int(h5f["stage6_review"].attrs["engine_version"]) == ENGINE_VERSION
        assert LINEAGE_ID_ATTR in h5f[STAGE5_BASELINE_GROUP].attrs


def test_undo_takes_serials_and_never_reuses_one(stage5_multi_file):
    fp = stage5_multi_file
    w1, w2 = _windows(fp, 2)
    ftmw.review_edit(fp, w1, add=[_clear(fp, w1)], frame="raw")
    ftmw.review_accept(fp, w2)
    ftmw.review_accept(fp, w1)
    review_undo_impl(fp, [2])  # the newest decision
    assert _next_serial(fp) == 3, "an undo must not lower the high-water mark"

    ftmw.review_accept(fp, w2)
    log = ftmw.review_log(fp)
    assert [(e.order_index, e.serial, e.kind) for e in log] == [
        (0, 0, "add"),
        (1, 1, "accept"),
        (2, 3, "accept"),
    ]
    # Ids are serials, not positions: position 2 is serial 3, serial 2 is gone.
    with pytest.raises(NotFoundValueError):
        review_undo_impl(fp, [2])
    review_undo_impl(fp, [3])
    assert [e.serial for e in ftmw.review_log(fp)] == [0, 1]


def test_undo_of_an_unrelated_accept_changes_nothing_else(stage5_multi_file):
    """The surviving rows are verbatim and no ``derivation`` tag changes:
    nothing a client holds is renumbered. The fits are an undo's one-batch
    replay of the survivors; they equal the sequential ones here only because
    this fixture's edited windows feed no other window (no cascade)."""
    fp = stage5_multi_file
    w1, w2, w3 = _windows(fp, 3)
    ftmw.review_edit(fp, w1, add=[_clear(fp, w1)], frame="raw")
    ftmw.review_accept(fp, w2)
    ftmw.review_edit(fp, w3, add=[_clear(fp, w3)], frame="raw")
    log_before = ftmw.review_log(fp)
    state_before = _peak_state(fp)
    assert {d for ps in state_before.values() for *_, d, _ in ps} >= {0, 2}

    review_undo_impl(fp, [1])

    log_after = ftmw.review_log(fp)
    assert [e.serial for e in log_after] == [0, 2]
    for kept in log_after:
        original = next(e for e in log_before if e.serial == kept.serial)
        # Verbatim but for the position: evidence (chi2r snapshot,
        # action_index) and frequency included.
        assert (kept.window_id, kept.kind, kept.frequency_mhz, kept.evidence) == (
            original.window_id,
            original.kind,
            original.frequency_mhz,
            original.evidence,
        )
    assert _peak_state(fp) == state_before


def test_a_joint_edit_after_an_undo_keys_its_group_by_serial(stage5_multi_file):
    fp = stage5_multi_file
    w1, w2 = _windows(fp, 2)
    ftmw.review_accept(fp, w2)
    ftmw.review_accept(fp, w1)
    review_undo_impl(fp, [0])
    span = _fits(fp)[w1].window
    assert span is not None
    lo, hi = sorted(float(v) for v in span.freq_range)
    ftmw.review_edit(fp, w1, add=[lo + 0.3 * (hi - lo), lo + 0.7 * (hi - lo)])
    rows = [e for e in ftmw.review_log(fp) if e.kind == "add"]
    assert [e.serial for e in rows] == [2, 3]
    assert {e.evidence[ACTION_INDEX_EVIDENCE_KEY] for e in rows} == {2}
    tags = sorted(
        p.derivation for p in _fits(fp)[w1].fitted_peaks if p.derivation is not None
    )
    assert tags == [2, 3]


def test_fit_run_starts_a_new_lineage(stage5_small_source, tmp_path):
    fp = tmp_path / "refit.ftmw"
    shutil.copy(stage5_small_source, fp)
    (w1,) = _windows(fp, 1)
    ftmw.review_accept(fp, w1)
    ftmw.review_edit(fp, w1, add=[_clear(fp, w1)], frame="raw")
    with h5py.File(str(fp), "r") as h5f:
        lineage = h5f[STAGE5_BASELINE_GROUP].attrs[LINEAGE_ID_ATTR]

    ftmw.fit_peaks(str(fp))
    with h5py.File(str(fp), "r") as h5f:
        assert "stage6_review" not in h5f
        assert STAGE5_BASELINE_GROUP not in h5f

    ftmw.review_edit(fp, w1, add=[_clear(fp, w1)], frame="raw")
    assert [e.serial for e in ftmw.review_log(fp)] == [0]
    with h5py.File(str(fp), "r") as h5f:
        assert h5f[STAGE5_BASELINE_GROUP].attrs[LINEAGE_ID_ATTR] != lineage


def test_a_staged_preview_persists_the_serials_an_apply_would_mint(
    stage5_multi_file, tmp_path
):
    """The session persists the preview's own bytes; its rows must carry the
    serials and the high-water mark a plain apply would have written."""
    fp = stage5_multi_file
    w1, w2, w3 = _windows(fp, 3)
    ftmw.review_accept(fp, w1)
    ftmw.review_accept(fp, w2)
    review_undo_impl(fp, [0])  # next_serial stays 2
    csv = tmp_path / "plan.csv"
    csv.write_text(f"accept,{w3},,\n")

    with Pipeline.open(fp).review_session() as session:
        session.review_preview(csv)
        session.review_apply(csv)
    log = ftmw.review_log(fp)
    assert [(e.kind, e.window_id, e.serial) for e in log] == [
        ("accept", w2, 1),
        ("accept", w3, 2),
    ]
    assert log[1].evidence[ACTION_INDEX_EVIDENCE_KEY] == 2
    assert _next_serial(fp) == 3


def test_the_cli_lists_and_undoes_by_serial(stage5_multi_file, capsys):
    fp = stage5_multi_file
    w1, w2, w3 = _windows(fp, 3)
    for w in (w1, w2, w3):
        ftmw.review_accept(fp, w)
    review_undo_impl(fp, [0])  # positions shift; serials do not

    capsys.readouterr()
    assert main(["review", "log", str(fp), "--json"]) == 0
    entries = json.loads(capsys.readouterr().out)["entries"]
    assert [(e["order_index"], e["serial"]) for e in entries] == [(0, 1), (1, 2)]

    assert main(["review", "undo", str(fp), "--id", "2", "--dry-run"]) == 0
    assert f"id 2: accept window {w3}" in capsys.readouterr().out
    assert main(["review", "undo", str(fp), "--id", "2"]) == 0
    assert [e.serial for e in ftmw.review_log(fp)] == [1]
    # Position 0 is serial 1; asking for id 0 is asking for a serial that is gone.
    assert main(["review", "undo", str(fp), "--id", "0", "--json"]) != 0
    assert [e.serial for e in ftmw.review_log(fp)] == [1]
