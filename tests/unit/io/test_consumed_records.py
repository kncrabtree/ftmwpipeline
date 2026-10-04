"""
Unit tests for the values a stage records as having used.

Stage 3 and Stage 5 record what they took from other stages' results under
``consumed`` in their own settings records; the timebase records the clock
declaration it ran with; the Stage 6 final-products table records the
declaration its calibration state was derived from. Each is read back through
its codec, and a record written before it existed reads as ``None``.
"""

from __future__ import annotations

import json
from typing import Any

import h5py
import pytest

from ftmwpipeline._internal.atomic import atomic_write
from ftmwpipeline.core.data_structures import FinalProducts, Stage6Review
from ftmwpipeline.core.peak_detection_settings import PeakDetectionSettings
from ftmwpipeline.core.stage_fit_settings import ClockSource, StageFitSettings
from ftmwpipeline.core.stage_fit_settings import resolve as resolve_stage_fit
from ftmwpipeline.fitting.tau_calibration import BandMajority
from ftmwpipeline.fitting.timebase_calibration import TimebaseCalibrationResult
from ftmwpipeline.io.peak_detection_settings_serialization import (
    STAGE3_PEAKS_SETTINGS_PATH,
    Stage3Consumed,
    load_peak_detection_consumed_from_h5,
    load_peak_detection_settings_from_h5,
    save_peak_detection_settings_to_h5,
)
from ftmwpipeline.io.stage6_review_serialization import (
    load_stage6_review_from_hdf5,
    read_final_products_calibration_clocks,
    save_stage6_review_to_hdf5,
)
from ftmwpipeline.io.stage_fit_settings_serialization import (
    STAGE_FIT_PATH,
    Stage5Consumed,
    load_stage_fit_consumed_from_h5,
    load_stage_fit_settings_from_h5,
    save_stage_fit_settings_to_h5,
)
from ftmwpipeline.io.timebase_serialization import (
    load_timebase_calibration_from_hdf5,
    save_timebase_calibration_to_hdf5,
)

CLOCKS = (
    ClockSource(5120.0, locked=True, label="DownLO"),
    ClockSource(50000.0, locked=False, label="digitizer"),
)


@pytest.fixture
def empty_ftmw(tmp_path: Any) -> str:
    p = tmp_path / "exp.ftmw"
    with h5py.File(p, "w") as h5f:
        h5f.create_group("placeholder")
    return str(p)


class TestStage3Consumed:
    def test_round_trip(self, empty_ftmw: str) -> None:
        consumed = Stage3Consumed(
            tau_basis_us=9.5, gap_shape="gaussian", tau_basis_source="stage2b_tau_G_maj"
        )
        with atomic_write(empty_ftmw):
            save_peak_detection_settings_to_h5(
                empty_ftmw, PeakDetectionSettings(), consumed=consumed
            )
        assert load_peak_detection_consumed_from_h5(empty_ftmw) == consumed

    def test_the_resolver_never_sees_it(self, empty_ftmw: str) -> None:
        with atomic_write(empty_ftmw):
            save_peak_detection_settings_to_h5(
                empty_ftmw,
                PeakDetectionSettings(),
                consumed=Stage3Consumed(5.0, "lorentzian", "default_5us"),
            )
        assert load_peak_detection_settings_from_h5(empty_ftmw) == (
            PeakDetectionSettings()
        )

    def test_absent_reads_none(self, empty_ftmw: str) -> None:
        assert load_peak_detection_consumed_from_h5(empty_ftmw) is None
        # The sparse user layer (``settings set``) writes no consumed block.
        with atomic_write(empty_ftmw):
            save_peak_detection_settings_to_h5(empty_ftmw, PeakDetectionSettings())
        with h5py.File(empty_ftmw, "r") as h5f:
            assert "consumed" not in h5f[STAGE3_PEAKS_SETTINGS_PATH]
        assert load_peak_detection_consumed_from_h5(empty_ftmw) is None


class TestStage5Consumed:
    BANDS = (
        BandMajority("b0", 26500.0, 27000.0, 12, 9.4, 0.08),
        BandMajority("b1", 27000.0, 27500.0, 7, 9.7, 0.12),
    )

    def test_round_trip_with_every_field_set(self, empty_ftmw: str) -> None:
        consumed = Stage5Consumed(
            tau_calibration_source="persisted",
            tau_maj_us=9.5,
            sigma_tau_us=0.09,
            band_majorities=self.BANDS,
            timebase_epsilon=2.2e-6,
            timebase_sigma_epsilon=7e-8,
            peak_survival_snr_floor=3.3,
            stft_spur_nominees=((27000.0, True), (27100.5, False)),
        )
        with atomic_write(empty_ftmw):
            save_stage_fit_settings_to_h5(
                empty_ftmw, StageFitSettings(), consumed=consumed
            )
        assert load_stage_fit_consumed_from_h5(empty_ftmw) == consumed

    def test_round_trip_with_nothing_consumed(self, empty_ftmw: str) -> None:
        consumed = Stage5Consumed(
            tau_calibration_source="none",
            tau_maj_us=None,
            sigma_tau_us=None,
            band_majorities=None,
            timebase_epsilon=None,
            timebase_sigma_epsilon=None,
            peak_survival_snr_floor=5.0,
        )
        with atomic_write(empty_ftmw):
            save_stage_fit_settings_to_h5(
                empty_ftmw, StageFitSettings(), consumed=consumed
            )
        assert load_stage_fit_consumed_from_h5(empty_ftmw) == consumed

    def test_an_empty_band_table_is_not_none(self, empty_ftmw: str) -> None:
        consumed = Stage5Consumed("persisted", 9.5, 0.09, (), None, None, 3.3)
        with atomic_write(empty_ftmw):
            save_stage_fit_settings_to_h5(
                empty_ftmw, StageFitSettings(), consumed=consumed
            )
        loaded = load_stage_fit_consumed_from_h5(empty_ftmw)
        assert loaded is not None and loaded.band_majorities == ()

    def test_an_empty_nominee_list_is_not_none(self, empty_ftmw: str) -> None:
        consumed = Stage5Consumed(
            "persisted", 9.5, 0.09, None, None, None, 3.3, stft_spur_nominees=()
        )
        with atomic_write(empty_ftmw):
            save_stage_fit_settings_to_h5(
                empty_ftmw, StageFitSettings(), consumed=consumed
            )
        loaded = load_stage_fit_consumed_from_h5(empty_ftmw)
        assert loaded is not None and loaded.stft_spur_nominees == ()

    def test_the_resolver_never_sees_it(self, empty_ftmw: str) -> None:
        resolved = resolve_stage_fit()
        with atomic_write(empty_ftmw):
            save_stage_fit_settings_to_h5(
                empty_ftmw,
                resolved,
                consumed=Stage5Consumed("none", None, None, None, None, None, 5.0),
            )
        assert load_stage_fit_settings_from_h5(empty_ftmw) == resolved

    def test_absent_reads_none(self, empty_ftmw: str) -> None:
        assert load_stage_fit_consumed_from_h5(empty_ftmw) is None
        with atomic_write(empty_ftmw):
            save_stage_fit_settings_to_h5(empty_ftmw, StageFitSettings())
        with h5py.File(empty_ftmw, "r") as h5f:
            assert "consumed" not in h5f[STAGE_FIT_PATH]
        assert load_stage_fit_consumed_from_h5(empty_ftmw) is None


def test_fit_tau_resolves_concrete() -> None:
    """A resolved record says tau was free; it is never left as None."""
    assert resolve_stage_fit().tau.fit_tau is True


def _timebase(**kwargs: Any) -> TimebaseCalibrationResult:
    return TimebaseCalibrationResult(
        epsilon=2.2e-6,
        sigma_epsilon=1e-7,
        n_used=5,
        n_detected=5,
        lattice_g_mhz=640.0,
        tone_reads=(),
        kappa_sys=0.0,
        snr_min=10.0,
        sample_dt_us=0.02,
        start_us=0.0,
        end_us=13.0,
        span_us=13.0,
        preconditions_passed=True,
        **kwargs,
    )


class TestTimebaseClockSources:
    def test_round_trip(self, empty_ftmw: str) -> None:
        with h5py.File(empty_ftmw, "a") as h5f:
            grp = h5f.create_group("timebase_calibration")
            save_timebase_calibration_to_hdf5(_timebase(clock_sources=CLOCKS), grp)
            loaded = load_timebase_calibration_from_hdf5(grp)
        assert loaded.clock_sources == CLOCKS

    def test_a_record_without_it_reads_none(self, empty_ftmw: str) -> None:
        with h5py.File(empty_ftmw, "a") as h5f:
            grp = h5f.create_group("timebase_calibration")
            save_timebase_calibration_to_hdf5(_timebase(clock_sources=CLOCKS), grp)
            del grp.attrs["clock_sources"]
            assert load_timebase_calibration_from_hdf5(grp).clock_sources is None


class TestFinalProductsCalibrationClocks:
    def _save(self, path: str, **kwargs: Any) -> None:
        review = Stage6Review(final_products=FinalProducts())
        with h5py.File(path, "a") as h5f:
            save_stage6_review_to_hdf5(
                review, h5f.create_group("stage6_review"), **kwargs
            )

    def test_round_trip(self, empty_ftmw: str) -> None:
        self._save(empty_ftmw, calibration_clocks=CLOCKS)
        with h5py.File(empty_ftmw, "r") as h5f:
            group = h5f["stage6_review"]
            assert read_final_products_calibration_clocks(group) == CLOCKS
            # The table itself is unchanged by the record beside it.
            loaded = load_stage6_review_from_hdf5(group).final_products
        assert loaded == FinalProducts()

    def test_an_empty_declaration_is_not_none(self, empty_ftmw: str) -> None:
        self._save(empty_ftmw, calibration_clocks=())
        with h5py.File(empty_ftmw, "r") as h5f:
            assert read_final_products_calibration_clocks(h5f["stage6_review"]) == ()

    def test_absent_reads_none(self, empty_ftmw: str) -> None:
        self._save(empty_ftmw)
        with h5py.File(empty_ftmw, "r") as h5f:
            group = h5f["stage6_review"]
            assert "calibration_clocks" not in group["final_products"].attrs
            assert read_final_products_calibration_clocks(group) is None
            assert read_final_products_calibration_clocks(None) is None

    def test_encoded_as_the_clock_source_dicts(self, empty_ftmw: str) -> None:
        self._save(empty_ftmw, calibration_clocks=CLOCKS)
        with h5py.File(empty_ftmw, "r") as h5f:
            raw = h5f["stage6_review/final_products"].attrs["calibration_clocks"]
        assert json.loads(raw) == [c.to_dict() for c in CLOCKS]
