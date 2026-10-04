"""
Unit tests for :mod:`ftmwpipeline.io.provenance`.

Every persisted settings record carries its codec's field-set version, a record
without one reads as pre-provenance (with its values unchanged), and an epoch
stamp that cannot be written fails the stage instead of being swallowed.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

import h5py
import pytest

from ftmwpipeline._internal.atomic import atomic_write
from ftmwpipeline.core.data_structures import FrequencyCalibration
from ftmwpipeline.core.environment import ANALYSIS_EPOCH
from ftmwpipeline.core.noise_settings import NoiseSettings
from ftmwpipeline.core.noise_settings import resolve as resolve_noise
from ftmwpipeline.core.peak_detection_settings import PeakDetectionSettings
from ftmwpipeline.core.stage_fit_settings import StageFitSettings
from ftmwpipeline.core.tau_calibration_settings import TauCalibrationSettings
from ftmwpipeline.core.window_planning_settings import WindowPlanningSettings
from ftmwpipeline.io import provenance
from ftmwpipeline.io.environment_serialization import load_stage_environments
from ftmwpipeline.io.frequency_calibration_serialization import (
    FREQUENCY_CALIBRATION_FIELD_SET_VERSION,
    frequency_calibration_provenance,
    load_frequency_calibration_from_hdf5,
    save_frequency_calibration_to_hdf5,
)
from ftmwpipeline.io.noise_settings_serialization import (
    STAGE2_NOISE_FIELD_SET_VERSION,
    STAGE2_NOISE_SETTINGS_PATH,
    load_noise_settings_from_h5,
    noise_settings_provenance,
    save_noise_settings_to_h5,
)
from ftmwpipeline.io.peak_detection_settings_serialization import (
    STAGE3_PEAKS_FIELD_SET_VERSION,
    peak_detection_settings_provenance,
    save_peak_detection_settings_to_h5,
)
from ftmwpipeline.io.provenance import (
    FIELD_SET_VERSION_ATTR,
    RecordProvenance,
    read_field_set_version,
    record_provenance,
    stamp_stage_epoch,
)
from ftmwpipeline.io.stage_fit_settings_serialization import (
    STAGE5_FIT_FIELD_SET_VERSION,
    save_stage_fit_settings_to_h5,
    stage_fit_settings_provenance,
)
from ftmwpipeline.io.tau_calibration_settings_serialization import (
    STAGE2B_TAU_FIELD_SET_VERSION,
    save_tau_calibration_settings_to_h5,
    tau_calibration_settings_provenance,
)
from ftmwpipeline.io.window_planning_settings_serialization import (
    STAGE4_WINDOWS_FIELD_SET_VERSION,
    save_window_planning_settings_to_h5,
    window_planning_settings_provenance,
)


@pytest.fixture
def empty_ftmw(tmp_path: Any) -> str:
    p = tmp_path / "exp.ftmw"
    with h5py.File(p, "w") as h5f:
        h5f.create_group("placeholder")
    return str(p)


_CODECS = [
    (
        save_noise_settings_to_h5,
        NoiseSettings,
        noise_settings_provenance,
        STAGE2_NOISE_FIELD_SET_VERSION,
    ),
    (
        save_tau_calibration_settings_to_h5,
        TauCalibrationSettings,
        tau_calibration_settings_provenance,
        STAGE2B_TAU_FIELD_SET_VERSION,
    ),
    (
        save_peak_detection_settings_to_h5,
        PeakDetectionSettings,
        peak_detection_settings_provenance,
        STAGE3_PEAKS_FIELD_SET_VERSION,
    ),
    (
        save_window_planning_settings_to_h5,
        WindowPlanningSettings,
        window_planning_settings_provenance,
        STAGE4_WINDOWS_FIELD_SET_VERSION,
    ),
    (
        save_stage_fit_settings_to_h5,
        StageFitSettings,
        stage_fit_settings_provenance,
        STAGE5_FIT_FIELD_SET_VERSION,
    ),
]


class TestRecordProvenance:
    def test_states(self) -> None:
        assert RecordProvenance("x", 2, 2).is_current
        legacy = RecordProvenance("x", None, 2)
        assert legacy.is_pre_provenance and not legacy.is_current
        older = RecordProvenance("x", 1, 2)
        assert older.is_pre_provenance and not older.is_current
        newer = RecordProvenance("x", 3, 2)
        assert newer.is_newer
        assert not newer.is_current and not newer.is_pre_provenance

    def test_read_absent_is_none(self) -> None:
        assert read_field_set_version({}) is None

    def test_read_rejects_a_bool(self) -> None:
        with pytest.raises(ValueError):
            read_field_set_version({FIELD_SET_VERSION_ATTR: True})


@pytest.mark.parametrize("save, cls, read, version", _CODECS)
def test_settings_codec_stamps_its_version(
    empty_ftmw: str,
    save: Callable[..., None],
    cls: Callable[[], Any],
    read: Callable[[str], Optional[RecordProvenance]],
    version: int,
) -> None:
    assert read(empty_ftmw) is None
    save(empty_ftmw, cls())
    prov = read(empty_ftmw)
    assert prov is not None
    assert prov.version == version
    assert prov.is_current


def test_flat_reader_ignores_the_version_attr(empty_ftmw: str) -> None:
    original = resolve_noise()
    with atomic_write(empty_ftmw):
        save_noise_settings_to_h5(empty_ftmw, original)
    assert load_noise_settings_from_h5(empty_ftmw) == original


def test_pre_provenance_record_reads_unchanged(empty_ftmw: str) -> None:
    """Stripping the version (a legacy record) changes only the provenance."""
    original = resolve_noise()
    with atomic_write(empty_ftmw):
        save_noise_settings_to_h5(empty_ftmw, original)
    with h5py.File(empty_ftmw, "a") as h5f:
        del h5f[STAGE2_NOISE_SETTINGS_PATH].attrs[FIELD_SET_VERSION_ATTR]
    assert load_noise_settings_from_h5(empty_ftmw) == original
    prov = record_provenance(
        empty_ftmw, STAGE2_NOISE_SETTINGS_PATH, STAGE2_NOISE_FIELD_SET_VERSION
    )
    assert prov is not None
    assert prov.version is None and prov.is_pre_provenance


def test_frequency_calibration_record(empty_ftmw: str) -> None:
    with h5py.File(empty_ftmw, "a") as h5f:
        assert frequency_calibration_provenance(h5f) is None
        save_frequency_calibration_to_hdf5(FrequencyCalibration(2.5), h5f)
        prov = frequency_calibration_provenance(h5f)
        assert prov is not None
        assert prov.version == FREQUENCY_CALIBRATION_FIELD_SET_VERSION
        assert load_frequency_calibration_from_hdf5(h5f).sigma_floor_khz == 2.5


class TestStampStageEpoch:
    def test_records_the_epoch(self, empty_ftmw: str) -> None:
        with h5py.File(empty_ftmw, "a") as h5f:
            record = stamp_stage_epoch(h5f, "stage3_peaks")
            envs = load_stage_environments(h5f)
        assert record.analysis_epoch == ANALYSIS_EPOCH
        assert envs["stage3_peaks"].analysis_epoch == ANALYSIS_EPOCH

    def test_a_failed_stamp_raises(
        self, empty_ftmw: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _fail(*args: Any, **kwargs: Any) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(
            "ftmwpipeline.io.environment_serialization.save_stage_environment",
            _fail,
        )
        with h5py.File(empty_ftmw, "a") as h5f:
            with pytest.raises(OSError):
                provenance.stamp_stage_epoch(h5f, "stage3_peaks")


class TestAFailedStampLeavesTheStageIncomplete:
    """The stamp is part of completing a stage: when it cannot be written the
    stage is not marked complete."""

    @staticmethod
    def _completed(path: str) -> list:
        with h5py.File(path, "r") as h5f:
            if "pipeline_stages" not in h5f:
                return []
            return json.loads(h5f["pipeline_stages"].attrs["completed_stages"])

    def test_update_stage_completion(
        self, empty_ftmw: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ftmwpipeline._internal.stage2_impl import _update_stage_completion

        def _fail(*args: Any, **kwargs: Any) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(
            "ftmwpipeline.io.environment_serialization.save_stage_environment",
            _fail,
        )
        with pytest.raises(RuntimeError, match="disk full"):
            with atomic_write(empty_ftmw):
                _update_stage_completion(empty_ftmw, "stage3_peaks")
        assert "stage3_peaks" not in self._completed(empty_ftmw)
