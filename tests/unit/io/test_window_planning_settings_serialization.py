"""
Unit tests for :mod:`ftmwpipeline.io.window_planning_settings_serialization`.

Verifies the HDF5 round-trip for ``WindowPlanningSettings`` (nested
subgroup layout under ``processing_parameters/stage4_windows``), audit
attrs, overwrite semantics, and sparse-settings handling.
"""

from __future__ import annotations

import h5py
import pytest

from ftmwpipeline.core.window_planning_settings import (
    WindowPlanningSettings,
    resolve,
)
from ftmwpipeline.io.window_planning_settings_serialization import (
    STAGE4_WINDOWS_SETTINGS_PATH,
    load_window_planning_settings_from_h5,
    save_window_planning_settings_to_h5,
    window_planning_settings_present,
)

_SUB_NAMES = ("coherence", "clustering", "contributor", "leakage")


@pytest.fixture
def empty_ftmw(tmp_path):
    p = tmp_path / "exp.ftmw"
    with h5py.File(p, "w") as h5f:
        h5f.create_group("placeholder")
    return str(p)


class TestStage4WindowsSettingsPersistence:
    def test_absent_returns_none(self, empty_ftmw) -> None:
        assert load_window_planning_settings_from_h5(empty_ftmw) is None
        assert window_planning_settings_present(empty_ftmw) is False

    def test_round_trip_resolved_settings(self, empty_ftmw) -> None:
        original = resolve()
        save_window_planning_settings_to_h5(empty_ftmw, original)
        assert window_planning_settings_present(empty_ftmw)
        loaded = load_window_planning_settings_from_h5(empty_ftmw)
        assert loaded is not None
        assert loaded.coherence.edge_m == original.coherence.edge_m
        assert (
            loaded.clustering.max_window_width_mhz
            == original.clustering.max_window_width_mhz
        )
        assert loaded.contributor.min_freeze_snr == original.contributor.min_freeze_snr
        # tau_us is the legitimately-None hard default
        assert loaded.leakage.tau_us is None
        assert original.leakage.tau_us is None

    def test_round_trip_sparse_settings(self, empty_ftmw) -> None:
        s = WindowPlanningSettings()
        save_window_planning_settings_to_h5(empty_ftmw, s)
        loaded = load_window_planning_settings_from_h5(empty_ftmw)
        assert loaded is not None
        assert loaded.is_empty()

    def test_preset_name_audit_attr(self, empty_ftmw) -> None:
        s = WindowPlanningSettings()
        s.coherence.edge_m = 32
        save_window_planning_settings_to_h5(
            empty_ftmw, s, preset_name="instrument_bc_2638"
        )
        with h5py.File(empty_ftmw, "r") as h5f:
            attrs = dict(h5f[STAGE4_WINDOWS_SETTINGS_PATH].attrs)
        assert attrs.get("preset_name") == "instrument_bc_2638"
        assert "creation_time" in attrs

    def test_overwrites_prior_block(self, empty_ftmw) -> None:
        s1 = WindowPlanningSettings()
        s1.coherence.edge_m = 32
        save_window_planning_settings_to_h5(empty_ftmw, s1)
        s2 = WindowPlanningSettings()
        s2.coherence.edge_m = 128
        save_window_planning_settings_to_h5(empty_ftmw, s2)
        loaded = load_window_planning_settings_from_h5(empty_ftmw)
        assert loaded is not None
        assert loaded.coherence.edge_m == 128

    def test_hdf5_subgroup_layout(self, empty_ftmw) -> None:
        s = resolve()
        save_window_planning_settings_to_h5(empty_ftmw, s)
        with h5py.File(empty_ftmw, "r") as h5f:
            grp = h5f[STAGE4_WINDOWS_SETTINGS_PATH]
            for sub_name in _SUB_NAMES:
                assert sub_name in grp
                assert isinstance(grp[sub_name], h5py.Group)

    def test_load_tolerates_missing_subgroup(self, empty_ftmw) -> None:
        s = WindowPlanningSettings()
        s.coherence.edge_m = 32
        save_window_planning_settings_to_h5(empty_ftmw, s)
        with h5py.File(empty_ftmw, "a") as h5f:
            del h5f[STAGE4_WINDOWS_SETTINGS_PATH]["leakage"]
        loaded = load_window_planning_settings_from_h5(empty_ftmw)
        assert loaded is not None
        assert loaded.coherence.edge_m == 32
        assert loaded.leakage.tau_us is None

    def test_distinct_from_stage4_windows_results(self, empty_ftmw) -> None:
        """Settings persist under processing_parameters/stage4_windows — distinct
        from the /stage4_windows results group at the root."""
        s = WindowPlanningSettings()
        s.coherence.edge_m = 32
        save_window_planning_settings_to_h5(empty_ftmw, s)
        with h5py.File(empty_ftmw, "r") as h5f:
            assert STAGE4_WINDOWS_SETTINGS_PATH in h5f
            assert "stage4_windows" not in h5f, (
                "save_window_planning_settings_to_h5 must not touch the "
                "root-level results group"
            )
