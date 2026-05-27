"""
Unit tests for :mod:`ftmwpipeline.io.noise_settings_serialization`.

Verifies the HDF5 round-trip for ``NoiseSettings`` (nested subgroup
layout under ``processing_parameters/stage2_noise``), audit attrs,
overwrite semantics, and sparse-settings handling.
"""

from __future__ import annotations

import h5py
import pytest

from ftmwpipeline.core.noise_settings import (
    NoiseSettings,
    resolve,
)
from ftmwpipeline.io.noise_settings_serialization import (
    STAGE2_NOISE_SETTINGS_PATH,
    load_noise_settings_from_h5,
    noise_settings_present,
    save_noise_settings_to_h5,
)


_SUB_NAMES = ("binning", "skewness", "smoothing", "skirt_exclusion")


@pytest.fixture
def empty_ftmw(tmp_path):
    p = tmp_path / "exp.ftmw"
    with h5py.File(p, "w") as h5f:
        h5f.create_group("placeholder")
    return str(p)


class TestStage2NoiseSettingsPersistence:
    def test_absent_returns_none(self, empty_ftmw) -> None:
        assert load_noise_settings_from_h5(empty_ftmw) is None
        assert noise_settings_present(empty_ftmw) is False

    def test_round_trip_resolved_settings(self, empty_ftmw) -> None:
        original = resolve()
        save_noise_settings_to_h5(empty_ftmw, original)
        assert noise_settings_present(empty_ftmw)
        loaded = load_noise_settings_from_h5(empty_ftmw)
        assert loaded is not None
        assert loaded.binning.subdivision_threshold == original.binning.subdivision_threshold
        assert loaded.skewness.skew_target == original.skewness.skew_target
        assert loaded.smoothing.smoothing_window_mhz == original.smoothing.smoothing_window_mhz
        assert loaded.skirt_exclusion.strong_peak_snr == original.skirt_exclusion.strong_peak_snr

    def test_round_trip_sparse_settings(self, empty_ftmw) -> None:
        s = NoiseSettings()
        save_noise_settings_to_h5(empty_ftmw, s)
        loaded = load_noise_settings_from_h5(empty_ftmw)
        assert loaded is not None
        assert loaded.is_empty()

    def test_preset_name_audit_attr(self, empty_ftmw) -> None:
        s = NoiseSettings()
        s.smoothing.smoothing_window_mhz = 100.0
        save_noise_settings_to_h5(
            empty_ftmw, s, preset_name="instrument_bc_2638"
        )
        with h5py.File(empty_ftmw, "r") as h5f:
            attrs = dict(h5f[STAGE2_NOISE_SETTINGS_PATH].attrs)
        assert attrs.get("preset_name") == "instrument_bc_2638"
        assert "creation_time" in attrs

    def test_overwrites_prior_block(self, empty_ftmw) -> None:
        s1 = NoiseSettings()
        s1.smoothing.smoothing_window_mhz = 100.0
        save_noise_settings_to_h5(empty_ftmw, s1)
        s2 = NoiseSettings()
        s2.smoothing.smoothing_window_mhz = 200.0
        save_noise_settings_to_h5(empty_ftmw, s2)
        loaded = load_noise_settings_from_h5(empty_ftmw)
        assert loaded is not None
        assert loaded.smoothing.smoothing_window_mhz == 200.0

    def test_hdf5_subgroup_layout(self, empty_ftmw) -> None:
        s = resolve()
        save_noise_settings_to_h5(empty_ftmw, s)
        with h5py.File(empty_ftmw, "r") as h5f:
            grp = h5f[STAGE2_NOISE_SETTINGS_PATH]
            for sub_name in _SUB_NAMES:
                assert sub_name in grp
                assert isinstance(grp[sub_name], h5py.Group)

    def test_load_tolerates_missing_subgroup(self, empty_ftmw) -> None:
        s = NoiseSettings()
        s.smoothing.smoothing_window_mhz = 100.0
        save_noise_settings_to_h5(empty_ftmw, s)
        with h5py.File(empty_ftmw, "a") as h5f:
            del h5f[STAGE2_NOISE_SETTINGS_PATH]["skirt_exclusion"]
        loaded = load_noise_settings_from_h5(empty_ftmw)
        assert loaded is not None
        assert loaded.smoothing.smoothing_window_mhz == 100.0
        assert loaded.skirt_exclusion.strong_peak_snr is None

    def test_distinct_from_stage2_noise_result(self, empty_ftmw) -> None:
        """Settings persist under processing_parameters/stage2_noise — distinct
        from the /stage2_noise_result results group."""
        s = NoiseSettings()
        s.smoothing.smoothing_window_mhz = 100.0
        save_noise_settings_to_h5(empty_ftmw, s)
        with h5py.File(empty_ftmw, "r") as h5f:
            assert STAGE2_NOISE_SETTINGS_PATH in h5f
            assert "stage2_noise_result" not in h5f, (
                "save_noise_settings_to_h5 must not touch the results group"
            )
