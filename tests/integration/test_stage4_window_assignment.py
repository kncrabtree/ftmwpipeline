"""
Stage 4 integration tests on real experiment 2638 data.

* cross-interface consistency: CLI == Pipeline == functional API
* real-data sanity: sane window count, strong lines anchor hard windows,
  promoted-only consumption, plan invariants hold
* serialization round-trip + hand-edit
* invalidation on Stage 3 re-detection and on a Stage 1 canonical-settings
  change

Heavy (full 750k FID + FT/noise/detect); marked slow + integration.
"""

import shutil
import subprocess

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import Pipeline
from ftmwpipeline.core.data_structures import WindowDifficulty

pytestmark = [pytest.mark.integration, pytest.mark.slow]

# Strong lines observed in exp 2638 (MHz); each must anchor a hard window.
KNOWN_STRONG = [28817.3, 31328.1, 33839.0, 36350.0, 38861.0]


def _assert_invariants(plan, peaks):
    """Plan invariants: disjoint windows, unique free peaks, acyclic DAG."""
    ws = sorted(plan.windows, key=lambda w: w.freq_range[0])
    for a, b in zip(ws, ws[1:]):
        assert a.freq_range[1] <= b.freq_range[0], "windows overlap"
    free = [li for w in plan.windows for li in w.free_peak_indices]
    assert len(free) == len(set(free)), "a peak is free in >1 window"
    # Only promoted peaks are consumed.
    for li in free:
        assert peaks[li].properties.get("promoted"), "non-promoted peak in plan"
    ids = sorted(w.window_id for w in plan.windows)
    assert sorted(plan.topological_order) == ids
    pos = {w: i for i, w in enumerate(plan.topological_order)}
    for w, dep in plan.dependency_edges:
        assert pos[dep] < pos[w], "dependency edge violates topological order"


def test_cross_interface_consistency(baseline_2638_stage3, temp_ftmw_dir):
    """CLI == Pipeline == functional API for assign_windows."""
    pfile = temp_ftmw_dir / "p.ftmw"
    ffile = temp_ftmw_dir / "f.ftmw"
    cfile = temp_ftmw_dir / "c.ftmw"
    for fp in (pfile, ffile, cfile):
        shutil.copy(baseline_2638_stage3, fp)

    plan_pipe = Pipeline(pfile).assign_windows()
    plan_func = ftmw.assign_windows(ffile)
    res = subprocess.run(
        ["ftmwpipeline", "windows", "run", str(cfile)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert res.returncode == 0, f"CLI failed: {res.stdout}\n{res.stderr}"
    plan_cli = ftmw.load_windows(cfile)

    for other in (plan_func, plan_cli):
        assert plan_pipe.n_windows == other.n_windows
        assert plan_pipe.dependency_edges == other.dependency_edges
        assert plan_pipe.topological_order == other.topological_order
        for a, b in zip(plan_pipe.windows, other.windows):
            assert a.window_id == b.window_id
            assert a.freq_range == pytest.approx(b.freq_range)
            assert a.free_peak_indices == b.free_peak_indices
            assert a.difficulty == b.difficulty
            assert a.batch == b.batch


def test_real_data_sanity(baseline_2638_stage3, temp_ftmw_dir):
    fp = temp_ftmw_dir / "sane.ftmw"
    shutil.copy(baseline_2638_stage3, fp)

    peaks = ftmw.load_peaks(fp)
    promoted = [p for p in peaks if p.properties.get("promoted")]
    plan = ftmw.assign_windows(fp)

    # A sane window count: dense regions collapse into far fewer windows than
    # there are promoted peaks, but the spectrum is not one giant window.
    assert (
        10 < plan.n_windows < len(promoted)
    ), f"implausible window count {plan.n_windows}"
    _assert_invariants(plan, peaks)

    # Every promoted peak is either a free peak or a pruned leakage artifact.
    free = {li for w in plan.windows for li in w.free_peak_indices}
    pruned = sum(
        len(w.diagnostics.get("pruned_leakage_artifacts", [])) for w in plan.windows
    )
    assert len(free) == len(promoted) - pruned

    # Each known strong line anchors a HARD window.
    for line in KNOWN_STRONG:
        hits = [
            w
            for w in plan.windows
            if w.freq_range[0] - 1.0 <= line <= w.freq_range[1] + 1.0
        ]
        assert hits, f"no window near known strong line {line} MHz"
        assert any(
            w.difficulty == WindowDifficulty.HARD for w in hits
        ), f"strong line {line} MHz not in a hard window"

    # Dense strong regions are flagged hard, not exploded into noise windows.
    n_hard = sum(1 for w in plan.windows if w.difficulty == WindowDifficulty.HARD)
    assert n_hard > 0


def test_serialization_round_trip_and_hand_edit(baseline_2638_stage3, temp_ftmw_dir):
    fp = temp_ftmw_dir / "ser.ftmw"
    shutil.copy(baseline_2638_stage3, fp)

    plan = ftmw.assign_windows(fp)
    reloaded = ftmw.load_windows(fp)
    assert reloaded.n_windows == plan.n_windows
    assert reloaded.dependency_edges == plan.dependency_edges
    assert reloaded.topological_order == plan.topological_order
    for a, b in zip(plan.windows, reloaded.windows):
        assert a.freq_range == pytest.approx(b.freq_range)
        assert a.free_peak_indices == b.free_peak_indices
        assert a.difficulty == b.difficulty

    # Hand-edit the persisted plan and confirm it reloads edited.
    from ftmwpipeline.io.window_serialization import (
        load_window_plan_from_hdf5,
        save_window_plan_to_hdf5,
    )

    edited = ftmw.load_windows(fp)
    edited.windows[0].difficulty = (
        WindowDifficulty.EASY
        if edited.windows[0].difficulty == WindowDifficulty.HARD
        else WindowDifficulty.HARD
    )
    with h5py.File(fp, "a") as h5f:
        del h5f["stage4_windows"]
        save_window_plan_to_hdf5(edited, h5f.create_group("stage4_windows"))
    again = ftmw.load_windows(fp)
    assert again.windows[0].difficulty == edited.windows[0].difficulty


def test_redetection_invalidates_stage4(baseline_2638_stage3, temp_ftmw_dir):
    """Re-running Stage 3 detection drops the stale Stage 4 window plan."""
    fp = temp_ftmw_dir / "inv.ftmw"
    shutil.copy(baseline_2638_stage3, fp)

    ftmw.assign_windows(fp)
    with h5py.File(fp, "r") as h5f:
        assert "stage4_windows" in h5f
        import json

        completed = json.loads(h5f["pipeline_stages"].attrs["completed_stages"])
        assert "stage4_windows" in completed

    # Re-detect peaks -> Stage 4 must be invalidated.
    ftmw.detect_peaks(fp)
    with h5py.File(fp, "r") as h5f:
        assert "stage4_windows" not in h5f
        import json

        completed = json.loads(h5f["pipeline_stages"].attrs["completed_stages"])
        assert "stage4_windows" not in completed

    with pytest.raises(Exception):
        ftmw.load_windows(fp)


def test_stage1_change_invalidates_stage4(baseline_2638_stage3, temp_ftmw_dir):
    """Changing the canonical Stage 1 settings invalidates Stage 4."""
    fp = temp_ftmw_dir / "s1.ftmw"
    shutil.copy(baseline_2638_stage3, fp)
    ftmw.assign_windows(fp)

    # A different canonical FT setting cascades the existing invalidation.
    ftmw.compute_ft(fp, start_us=2.0, trim=(26500, 40000))
    with h5py.File(fp, "r") as h5f:
        assert "stage4_windows" not in h5f
        import json

        completed = json.loads(h5f["pipeline_stages"].attrs["completed_stages"])
        assert "stage4_windows" not in completed
        assert "stage3_peaks" not in completed
