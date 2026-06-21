"""
Unit tests for :mod:`ftmwpipeline.io.stage_fit_settings_serialization`.

Verifies the HDF5 round-trip for ``StageFitSettings`` (nested subgroup
layout under ``processing_parameters/stage5_fit``) and the Stage 2b
``recommended_shape`` stub attribute.
"""

from __future__ import annotations

import h5py
import pytest

from ftmwpipeline.core.peak_shape import PeakShape
from ftmwpipeline.core.stage_fit_settings import (
    ShapeSpec,
    StageFitSettings,
    resolve,
)
from ftmwpipeline.io.stage_fit_settings_serialization import (
    STAGE_FIT_PATH,
    load_stage_fit_settings_from_h5,
    read_stage2b_recommended_shape,
    read_stage2b_vote_rates,
    save_stage_fit_settings_to_h5,
    stage_fit_settings_present,
    write_stage2b_recommended_shape,
)


@pytest.fixture
def empty_ftmw(tmp_path):
    """A bare HDF5 file standing in for an .ftmw at the persistence boundary."""
    p = tmp_path / "exp.ftmw"
    with h5py.File(p, "w") as h5f:
        h5f.create_group("placeholder")  # ensure the file exists with content
    return str(p)


class TestStageFitPersistence:
    def test_absent_returns_none(self, empty_ftmw) -> None:
        assert load_stage_fit_settings_from_h5(empty_ftmw) is None
        assert stage_fit_settings_present(empty_ftmw) is False

    def test_round_trip_resolved_settings(self, empty_ftmw) -> None:
        """A resolved instance round-trips through HDF5 byte-for-byte."""
        original = resolve()  # all hard defaults, shape=Lorentzian
        save_stage_fit_settings_to_h5(empty_ftmw, original)
        assert stage_fit_settings_present(empty_ftmw)
        loaded = load_stage_fit_settings_from_h5(empty_ftmw)
        assert loaded is not None
        assert loaded.shape is not None and loaded.shape.kind is PeakShape.LORENTZIAN
        assert loaded.tau.max_decay_factor == original.tau.max_decay_factor
        assert loaded.rescue.max_rounds == original.rescue.max_rounds
        assert loaded.conservative.max_peaks == original.conservative.max_peaks

    def test_round_trip_sparse_settings(self, empty_ftmw) -> None:
        """An empty StageFitSettings round-trips with all fields back to None."""
        s = StageFitSettings()
        save_stage_fit_settings_to_h5(empty_ftmw, s)
        loaded = load_stage_fit_settings_from_h5(empty_ftmw)
        assert loaded is not None
        assert loaded.shape is None
        assert loaded.tau.max_decay_factor is None
        assert loaded.conservative.max_peaks is None
        assert loaded.rescue.max_rounds is None

    def test_round_trip_gaussian_shape(self, empty_ftmw) -> None:
        s = StageFitSettings(shape=ShapeSpec(kind=PeakShape.GAUSSIAN))
        s.tau.max_decay_factor = 3.0
        save_stage_fit_settings_to_h5(empty_ftmw, s)
        loaded = load_stage_fit_settings_from_h5(empty_ftmw)
        assert loaded is not None
        assert loaded.shape is not None and loaded.shape.kind is PeakShape.GAUSSIAN
        assert loaded.tau.max_decay_factor == 3.0

    def test_preset_name_audit_attr(self, empty_ftmw) -> None:
        s = StageFitSettings(shape=ShapeSpec(kind=PeakShape.GAUSSIAN))
        save_stage_fit_settings_to_h5(empty_ftmw, s, preset_name="instrument_bc_2638")
        with h5py.File(empty_ftmw, "r") as h5f:
            attrs = dict(h5f[STAGE_FIT_PATH].attrs)
        assert attrs.get("preset_name") == "instrument_bc_2638"
        assert "creation_time" in attrs

    def test_overwrites_prior_block(self, empty_ftmw) -> None:
        s1 = StageFitSettings(shape=ShapeSpec(kind=PeakShape.LORENTZIAN))
        s1.tau.max_decay_factor = 5.0
        save_stage_fit_settings_to_h5(empty_ftmw, s1)
        s2 = StageFitSettings(shape=ShapeSpec(kind=PeakShape.GAUSSIAN))
        s2.tau.max_decay_factor = 3.0
        save_stage_fit_settings_to_h5(empty_ftmw, s2)
        loaded = load_stage_fit_settings_from_h5(empty_ftmw)
        assert loaded is not None
        assert loaded.shape is not None and loaded.shape.kind is PeakShape.GAUSSIAN
        assert loaded.tau.max_decay_factor == 3.0

    def test_hdf5_subgroup_layout(self, empty_ftmw) -> None:
        """The on-disk layout exposes one subgroup per sub-dataclass."""
        s = resolve()
        save_stage_fit_settings_to_h5(empty_ftmw, s)
        with h5py.File(empty_ftmw, "r") as h5f:
            grp = h5f[STAGE_FIT_PATH]
            assert isinstance(grp["shape"], h5py.Group)
            for sub_name in (
                "tau",
                "seeder",
                "conservative",
                "penalties",
                "rescue",
                "thaw",
            ):
                assert sub_name in grp
                assert isinstance(grp[sub_name], h5py.Group)


class TestStage2bRecommendedShape:
    def test_absent_returns_none(self, empty_ftmw) -> None:
        assert read_stage2b_recommended_shape(empty_ftmw) is None

    def test_write_noop_when_stage2b_absent(self, empty_ftmw) -> None:
        """Writing the stub with no Stage 2b group present is a silent no-op."""
        write_stage2b_recommended_shape(empty_ftmw, shape=None)
        # Verify nothing was created.
        with h5py.File(empty_ftmw, "r") as h5f:
            assert "stage2b_tau_calibration" not in h5f
        assert read_stage2b_recommended_shape(empty_ftmw) is None

    def test_stamps_sentinel_on_existing_stage2b(self, empty_ftmw) -> None:
        with h5py.File(empty_ftmw, "a") as h5f:
            h5f.create_group("stage2b_tau_calibration")
        write_stage2b_recommended_shape(empty_ftmw, shape=None)
        assert (
            read_stage2b_recommended_shape(empty_ftmw) is None
        )  # sentinel decodes back

    def test_stamps_concrete_shape(self, empty_ftmw) -> None:
        with h5py.File(empty_ftmw, "a") as h5f:
            h5f.create_group("stage2b_tau_calibration")
        write_stage2b_recommended_shape(empty_ftmw, shape="gaussian")
        assert read_stage2b_recommended_shape(empty_ftmw) == "gaussian"

    def test_vote_rates_round_trip(self, empty_ftmw) -> None:
        with h5py.File(empty_ftmw, "a") as h5f:
            h5f.create_group("stage2b_tau_calibration")
        assert read_stage2b_vote_rates(empty_ftmw) == {}  # absent -> empty
        votes = {"exp": 0.21, "gauss": 0.71, "voigt": 0.08}
        write_stage2b_recommended_shape(empty_ftmw, shape="gaussian", vote_rates=votes)
        assert read_stage2b_vote_rates(empty_ftmw) == votes

    def test_vote_rates_cleared_on_reset(self, empty_ftmw) -> None:
        with h5py.File(empty_ftmw, "a") as h5f:
            h5f.create_group("stage2b_tau_calibration")
        write_stage2b_recommended_shape(
            empty_ftmw, shape="gaussian", vote_rates={"gauss": 1.0}
        )
        # A reset verdict (shape=None, no vote_rates) clears the stale breakdown.
        write_stage2b_recommended_shape(empty_ftmw, shape=None)
        assert read_stage2b_vote_rates(empty_ftmw) == {}
