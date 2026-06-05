"""
Unit tests for :mod:`ftmwpipeline.io.noise_settings_serialization`.

Verifies the HDF5 round-trip for ``NoiseSettings`` (flat attrs on the
``processing_parameters/stage2_noise`` group), audit attrs, overwrite
semantics, and sparse-settings handling.
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
        assert loaded.window_mhz == original.window_mhz
        assert loaded.n_iter == original.n_iter
        assert loaded.region_aware == original.region_aware

    def test_round_trip_bool_and_int(self, empty_ftmw) -> None:
        """The region_aware bool and n_iter int survive the HDF5 round-trip."""
        s = NoiseSettings(window_mhz=60.0, n_iter=5, region_aware=False)
        save_noise_settings_to_h5(empty_ftmw, s)
        loaded = load_noise_settings_from_h5(empty_ftmw)
        assert loaded is not None
        assert loaded.window_mhz == 60.0
        assert loaded.n_iter == 5
        assert bool(loaded.region_aware) is False

    def test_round_trip_sparse_settings(self, empty_ftmw) -> None:
        save_noise_settings_to_h5(empty_ftmw, NoiseSettings())
        loaded = load_noise_settings_from_h5(empty_ftmw)
        assert loaded is not None
        assert loaded.is_empty()

    def test_preset_name_audit_attr(self, empty_ftmw) -> None:
        s = NoiseSettings(window_mhz=60.0)
        save_noise_settings_to_h5(empty_ftmw, s, preset_name="instrument_bc_2638")
        with h5py.File(empty_ftmw, "r") as h5f:
            attrs = dict(h5f[STAGE2_NOISE_SETTINGS_PATH].attrs)
        assert attrs.get("preset_name") == "instrument_bc_2638"
        assert "creation_time" in attrs

    def test_audit_attrs_do_not_leak_into_settings(self, empty_ftmw) -> None:
        """creation_time / preset_name are group bookkeeping, not fields."""
        save_noise_settings_to_h5(
            empty_ftmw, NoiseSettings(window_mhz=60.0), preset_name="p"
        )
        loaded = load_noise_settings_from_h5(empty_ftmw)
        assert loaded is not None
        assert loaded.window_mhz == 60.0
        # Only the eight settings fields are populated; no stray attrs.
        assert loaded.pedestal_mhz is None

    def test_overwrites_prior_block(self, empty_ftmw) -> None:
        save_noise_settings_to_h5(empty_ftmw, NoiseSettings(window_mhz=60.0))
        save_noise_settings_to_h5(empty_ftmw, NoiseSettings(window_mhz=120.0))
        loaded = load_noise_settings_from_h5(empty_ftmw)
        assert loaded is not None
        assert loaded.window_mhz == 120.0

    def test_hdf5_attr_layout(self, empty_ftmw) -> None:
        save_noise_settings_to_h5(empty_ftmw, resolve())
        with h5py.File(empty_ftmw, "r") as h5f:
            grp = h5f[STAGE2_NOISE_SETTINGS_PATH]
            # Flat: fields are attrs on the group, no sub-groups.
            assert len(grp.keys()) == 0
            assert "window_mhz" in grp.attrs
            assert "convolve_mhz" in grp.attrs

    def test_distinct_from_stage2_noise_result(self, empty_ftmw) -> None:
        """Settings persist under processing_parameters/stage2_noise — distinct
        from the /stage2_noise_result results group."""
        save_noise_settings_to_h5(empty_ftmw, NoiseSettings(window_mhz=60.0))
        with h5py.File(empty_ftmw, "r") as h5f:
            assert STAGE2_NOISE_SETTINGS_PATH in h5f
            assert (
                "stage2_noise_result" not in h5f
            ), "save_noise_settings_to_h5 must not touch the results group"
