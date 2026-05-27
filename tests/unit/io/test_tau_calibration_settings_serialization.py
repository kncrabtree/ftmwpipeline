"""
Unit tests for :mod:`ftmwpipeline.io.tau_calibration_settings_serialization`.

Verifies the HDF5 round-trip for ``TauCalibrationSettings`` (nested
subgroup layout under ``processing_parameters/stage2b_tau``), audit
attrs, overwrite semantics, and sparse-settings handling.
"""

from __future__ import annotations

import h5py
import pytest

from ftmwpipeline.core.tau_calibration_settings import (
    TauCalibrationSettings,
    resolve,
)
from ftmwpipeline.io.tau_calibration_settings_serialization import (
    STAGE2B_TAU_SETTINGS_PATH,
    load_tau_calibration_settings_from_h5,
    save_tau_calibration_settings_to_h5,
    tau_calibration_settings_present,
)


_SUB_NAMES = (
    "stft",
    "polish",
    "aggregation",
    "band",
    "gaussian",
    "recommendation",
)


@pytest.fixture
def empty_ftmw(tmp_path):
    """A bare HDF5 file standing in for an .ftmw at the persistence boundary."""
    p = tmp_path / "exp.ftmw"
    with h5py.File(p, "w") as h5f:
        h5f.create_group("placeholder")
    return str(p)


class TestStage2bTauSettingsPersistence:
    def test_absent_returns_none(self, empty_ftmw) -> None:
        assert load_tau_calibration_settings_from_h5(empty_ftmw) is None
        assert tau_calibration_settings_present(empty_ftmw) is False

    def test_round_trip_resolved_settings(self, empty_ftmw) -> None:
        original = resolve()
        save_tau_calibration_settings_to_h5(empty_ftmw, original)
        assert tau_calibration_settings_present(empty_ftmw)
        loaded = load_tau_calibration_settings_from_h5(empty_ftmw)
        assert loaded is not None
        assert loaded.stft.n_seg == original.stft.n_seg
        assert loaded.polish.polish_snr_cap == original.polish.polish_snr_cap
        assert loaded.aggregation.min_contributors == original.aggregation.min_contributors
        assert loaded.gaussian.tau_G_seeds == original.gaussian.tau_G_seeds
        assert loaded.recommendation.pure_margin_threshold == original.recommendation.pure_margin_threshold

    def test_round_trip_sparse_settings(self, empty_ftmw) -> None:
        s = TauCalibrationSettings()
        save_tau_calibration_settings_to_h5(empty_ftmw, s)
        loaded = load_tau_calibration_settings_from_h5(empty_ftmw)
        assert loaded is not None
        assert loaded.is_empty()

    def test_round_trip_tuple_fields(self, empty_ftmw) -> None:
        s = TauCalibrationSettings()
        s.gaussian.tau_G_seeds = (50.0, 10.0, 2.0)
        s.band.band_labels = ("A", "B", "C")
        s.band.band_edges_mhz = (30000.0, 36000.0)
        save_tau_calibration_settings_to_h5(empty_ftmw, s)
        loaded = load_tau_calibration_settings_from_h5(empty_ftmw)
        assert loaded is not None
        assert loaded.gaussian.tau_G_seeds == (50.0, 10.0, 2.0)
        assert isinstance(loaded.gaussian.tau_G_seeds, tuple)
        assert loaded.band.band_labels == ("A", "B", "C")
        assert isinstance(loaded.band.band_labels, tuple)
        assert loaded.band.band_edges_mhz == (30000.0, 36000.0)

    def test_preset_name_audit_attr(self, empty_ftmw) -> None:
        s = TauCalibrationSettings()
        s.stft.n_seg = 8
        save_tau_calibration_settings_to_h5(
            empty_ftmw, s, preset_name="instrument_bc_2638"
        )
        with h5py.File(empty_ftmw, "r") as h5f:
            attrs = dict(h5f[STAGE2B_TAU_SETTINGS_PATH].attrs)
        assert attrs.get("preset_name") == "instrument_bc_2638"
        assert "creation_time" in attrs

    def test_overwrites_prior_block(self, empty_ftmw) -> None:
        s1 = TauCalibrationSettings()
        s1.stft.n_seg = 8
        save_tau_calibration_settings_to_h5(empty_ftmw, s1)
        s2 = TauCalibrationSettings()
        s2.stft.n_seg = 20
        save_tau_calibration_settings_to_h5(empty_ftmw, s2)
        loaded = load_tau_calibration_settings_from_h5(empty_ftmw)
        assert loaded is not None
        assert loaded.stft.n_seg == 20

    def test_hdf5_subgroup_layout(self, empty_ftmw) -> None:
        s = resolve()
        save_tau_calibration_settings_to_h5(empty_ftmw, s)
        with h5py.File(empty_ftmw, "r") as h5f:
            grp = h5f[STAGE2B_TAU_SETTINGS_PATH]
            for sub_name in _SUB_NAMES:
                assert sub_name in grp
                assert isinstance(grp[sub_name], h5py.Group)

    def test_load_tolerates_missing_subgroup(self, empty_ftmw) -> None:
        """A persisted record missing a sub-block still loads (forward-compat)."""
        s = TauCalibrationSettings()
        s.stft.n_seg = 8
        save_tau_calibration_settings_to_h5(empty_ftmw, s)
        # Manually delete one sub-group and verify the loader survives.
        with h5py.File(empty_ftmw, "a") as h5f:
            del h5f[STAGE2B_TAU_SETTINGS_PATH]["recommendation"]
        loaded = load_tau_calibration_settings_from_h5(empty_ftmw)
        assert loaded is not None
        assert loaded.stft.n_seg == 8
        assert loaded.recommendation.snr_min is None
