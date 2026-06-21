"""
Unit tests for Stage 4 window-plan HDF5 serialization.

Covers the round-trip contract (load -> edit -> save -> load returns the
edited plan) and loud validation of malformed/hand-edited groups.
"""

import h5py
import pytest

from ftmwpipeline.core.data_structures import (
    FitWindow,
    FixedContributor,
    WindowPlan,
)
from ftmwpipeline.io.window_serialization import (
    load_window_plan_from_hdf5,
    save_window_plan_to_hdf5,
)


def _sample_plan():
    return WindowPlan(
        windows=[
            FitWindow(
                window_id=0,
                freq_range=(26500.0, 26520.0),
                free_peak_indices=[0, 1, 2],
                batch=0,
                diagnostics={"edge_coherence_fail": False, "n_strong_in_band": 1},
            ),
            FitWindow(
                window_id=1,
                freq_range=(26530.0, 26536.0),
                free_peak_indices=[5],
                fixed_contributors=[
                    FixedContributor(
                        peak_index=1,
                        primary_window_id=0,
                        frequency_mhz=26510.0,
                        freeze_eligible=False,
                        edge_free=True,
                    )
                ],
                batch=1,
            ),
        ],
        dependency_edges=[(1, 0)],
        topological_order=[0, 1],
        parameters={"edge_m": 64, "edge_threshold": 8.0, "tau_us": None},
        diagnostics={"n_pruned_leakage_artifacts": 4},
    )


def _roundtrip(plan, path):
    with h5py.File(path, "w") as h5f:
        save_window_plan_to_hdf5(plan, h5f.create_group("stage4_windows"))
    with h5py.File(path, "r") as h5f:
        return load_window_plan_from_hdf5(h5f["stage4_windows"])


class TestRoundTrip:
    def test_plan_round_trips(self, tmp_path):
        plan = _sample_plan()
        loaded = _roundtrip(plan, tmp_path / "p.h5")

        assert loaded.n_windows == 2
        assert loaded.dependency_edges == [(1, 0)]
        assert loaded.topological_order == [0, 1]
        assert loaded.parameters["edge_m"] == 64
        assert loaded.diagnostics["n_pruned_leakage_artifacts"] == 4

        w0, w1 = loaded.windows
        assert w0.free_peak_indices == [0, 1, 2]
        assert w0.diagnostics["n_strong_in_band"] == 1
        assert w1.freq_range == (26530.0, 26536.0)
        assert w1.batch == 1
        assert len(w1.fixed_contributors) == 1
        fc = w1.fixed_contributors[0]
        assert fc.peak_index == 1
        assert fc.primary_window_id == 0
        assert fc.freeze_eligible is False
        assert fc.edge_free is True

    def test_edge_free_flag_round_trips(self, tmp_path):
        """A window mixing an edge-free and an edge-bearing contributor keeps
        each flag verbatim through save -> load."""
        plan = _sample_plan()
        plan.windows[1].fixed_contributors.append(
            FixedContributor(
                peak_index=2,
                primary_window_id=0,
                frequency_mhz=26512.0,
                freeze_eligible=True,
                edge_free=False,
            )
        )
        loaded = _roundtrip(plan, tmp_path / "ef.h5")
        flags = {
            fc.peak_index: fc.edge_free for fc in loaded.windows[1].fixed_contributors
        }
        assert flags == {1: True, 2: False}

    def test_legacy_plan_without_edge_free_defaults_false(self, tmp_path):
        """A plan persisted before the edge-free column loads with edge_free
        defaulted to False (back-compat)."""
        path = tmp_path / "legacy.h5"
        plan = _sample_plan()
        with h5py.File(path, "w") as h5f:
            save_window_plan_to_hdf5(plan, h5f.create_group("stage4_windows"))
        # Simulate a legacy file: drop the new column from every window.
        with h5py.File(path, "a") as h5f:
            for name in h5f["stage4_windows/windows"]:
                wg = h5f[f"stage4_windows/windows/{name}"]
                if "fixed_edge_free" in wg:
                    del wg["fixed_edge_free"]
        with h5py.File(path, "r") as h5f:
            loaded = load_window_plan_from_hdf5(h5f["stage4_windows"])
        for w in loaded.windows:
            for fc in w.fixed_contributors:
                assert fc.edge_free is False

    def test_hand_edit_round_trips(self, tmp_path):
        """load -> edit the free set -> save -> load."""
        path = tmp_path / "p.h5"
        plan = _sample_plan()
        with h5py.File(path, "w") as h5f:
            save_window_plan_to_hdf5(plan, h5f.create_group("stage4_windows"))
        loaded = None
        with h5py.File(path, "r") as h5f:
            loaded = load_window_plan_from_hdf5(h5f["stage4_windows"])
        # Curator edits the plan.
        loaded.windows[0].free_peak_indices = [0, 1]
        with h5py.File(path, "w") as h5f:
            save_window_plan_to_hdf5(loaded, h5f.create_group("stage4_windows"))
        with h5py.File(path, "r") as h5f:
            again = load_window_plan_from_hdf5(h5f["stage4_windows"])
        assert again.windows[0].free_peak_indices == [0, 1]

    def test_empty_plan_round_trips(self, tmp_path):
        loaded = _roundtrip(WindowPlan(parameters={"edge_m": 64}), tmp_path / "e.h5")
        assert loaded.n_windows == 0


class TestLoudValidation:
    def test_missing_windows_subgroup_raises(self, tmp_path):
        path = tmp_path / "bad.h5"
        with h5py.File(path, "w") as h5f:
            h5f.create_group("stage4_windows")  # no 'windows' child
        with h5py.File(path, "r") as h5f:
            with pytest.raises(ValueError, match="windows"):
                load_window_plan_from_hdf5(h5f["stage4_windows"])

    def test_missing_required_attr_raises(self, tmp_path):
        path = tmp_path / "bad.h5"
        with h5py.File(path, "w") as h5f:
            save_window_plan_to_hdf5(_sample_plan(), h5f.create_group("stage4_windows"))
        # Drop a required per-window attribute.
        with h5py.File(path, "a") as h5f:
            del h5f["stage4_windows/windows/window_0000"].attrs["batch"]
        with h5py.File(path, "r") as h5f:
            with pytest.raises(ValueError, match="batch"):
                load_window_plan_from_hdf5(h5f["stage4_windows"])

    def test_missing_dataset_raises(self, tmp_path):
        path = tmp_path / "bad.h5"
        with h5py.File(path, "w") as h5f:
            save_window_plan_to_hdf5(_sample_plan(), h5f.create_group("stage4_windows"))
        with h5py.File(path, "a") as h5f:
            del h5f["stage4_windows/windows/window_0000/free_peak_indices"]
        with h5py.File(path, "r") as h5f:
            with pytest.raises(ValueError, match="free_peak_indices"):
                load_window_plan_from_hdf5(h5f["stage4_windows"])
