"""
Unit tests for :mod:`ftmwpipeline.io.peak_detection_settings_serialization`.

Verifies the HDF5 round-trip for ``PeakDetectionSettings`` (nested
subgroup layout under ``processing_parameters/stage3_peaks``), audit
attrs, overwrite semantics, and sparse-settings handling.
"""

from __future__ import annotations

import h5py
import pytest

from ftmwpipeline.core.peak_detection_settings import (
    PeakDetectionSettings,
    resolve,
)
from ftmwpipeline.io.peak_detection_settings_serialization import (
    STAGE3_PEAKS_SETTINGS_PATH,
    load_peak_detection_settings_from_h5,
    peak_detection_settings_present,
    save_peak_detection_settings_to_h5,
)

_SUB_NAMES = ("promotion", "savgol", "primary_pass", "gap_pass")


@pytest.fixture
def empty_ftmw(tmp_path):
    p = tmp_path / "exp.ftmw"
    with h5py.File(p, "w") as h5f:
        h5f.create_group("placeholder")
    return str(p)


class TestStage3PeaksSettingsPersistence:
    def test_absent_returns_none(self, empty_ftmw) -> None:
        assert load_peak_detection_settings_from_h5(empty_ftmw) is None
        assert peak_detection_settings_present(empty_ftmw) is False

    def test_round_trip_resolved_settings(self, empty_ftmw) -> None:
        original = resolve()
        save_peak_detection_settings_to_h5(empty_ftmw, original)
        assert peak_detection_settings_present(empty_ftmw)
        loaded = load_peak_detection_settings_from_h5(empty_ftmw)
        assert loaded is not None
        assert loaded.promotion.min_snr == original.promotion.min_snr
        assert loaded.savgol.sg_window == original.savgol.sg_window
        assert (
            loaded.primary_pass.primary_window == original.primary_pass.primary_window
        )
        assert (
            loaded.gap_pass.gap_leakage_floor_k == original.gap_pass.gap_leakage_floor_k
        )

    def test_round_trip_sparse_settings(self, empty_ftmw) -> None:
        s = PeakDetectionSettings()
        save_peak_detection_settings_to_h5(empty_ftmw, s)
        loaded = load_peak_detection_settings_from_h5(empty_ftmw)
        assert loaded is not None
        assert loaded.is_empty()

    def test_preset_name_audit_attr(self, empty_ftmw) -> None:
        s = PeakDetectionSettings()
        s.promotion.min_snr = 4.0
        save_peak_detection_settings_to_h5(empty_ftmw, s, preset_name="defaults")
        with h5py.File(empty_ftmw, "r") as h5f:
            attrs = dict(h5f[STAGE3_PEAKS_SETTINGS_PATH].attrs)
        assert attrs.get("preset_name") == "defaults"
        assert "creation_time" in attrs

    def test_overwrites_prior_block(self, empty_ftmw) -> None:
        s1 = PeakDetectionSettings()
        s1.promotion.min_snr = 4.0
        save_peak_detection_settings_to_h5(empty_ftmw, s1)
        s2 = PeakDetectionSettings()
        s2.promotion.min_snr = 7.0
        save_peak_detection_settings_to_h5(empty_ftmw, s2)
        loaded = load_peak_detection_settings_from_h5(empty_ftmw)
        assert loaded is not None
        assert loaded.promotion.min_snr == 7.0

    def test_hdf5_subgroup_layout(self, empty_ftmw) -> None:
        s = resolve()
        save_peak_detection_settings_to_h5(empty_ftmw, s)
        with h5py.File(empty_ftmw, "r") as h5f:
            grp = h5f[STAGE3_PEAKS_SETTINGS_PATH]
            for sub_name in _SUB_NAMES:
                assert sub_name in grp
                assert isinstance(grp[sub_name], h5py.Group)

    def test_load_tolerates_missing_subgroup(self, empty_ftmw) -> None:
        s = PeakDetectionSettings()
        s.promotion.min_snr = 4.0
        save_peak_detection_settings_to_h5(empty_ftmw, s)
        with h5py.File(empty_ftmw, "a") as h5f:
            del h5f[STAGE3_PEAKS_SETTINGS_PATH]["gap_pass"]
        loaded = load_peak_detection_settings_from_h5(empty_ftmw)
        assert loaded is not None
        assert loaded.promotion.min_snr == 4.0
        assert loaded.gap_pass.run_gap_pass is None

    def test_distinct_from_stage3_peaks_results(self, empty_ftmw) -> None:
        """Settings persist under processing_parameters/stage3_peaks — distinct
        from the /stage3_peaks results group at the root."""
        s = PeakDetectionSettings()
        s.promotion.min_snr = 4.0
        save_peak_detection_settings_to_h5(empty_ftmw, s)
        with h5py.File(empty_ftmw, "r") as h5f:
            assert STAGE3_PEAKS_SETTINGS_PATH in h5f
            assert "stage3_peaks" not in h5f, (
                "save_peak_detection_settings_to_h5 must not touch the "
                "root-level results group"
            )
