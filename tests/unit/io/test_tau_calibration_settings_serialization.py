"""
Unit tests for :mod:`ftmwpipeline.io.tau_calibration_settings_serialization`.

Verifies the HDF5 round-trip for ``TauCalibrationSettings`` (nested
subgroup layout under ``processing_parameters/stage2b_tau``), audit
attrs, overwrite semantics, and sparse-settings handling.
"""

from __future__ import annotations

import h5py
import pytest

from ftmwpipeline._internal.atomic import atomic_write
from ftmwpipeline.core.tau_calibration_settings import (
    PRODUCER_FIELDS,
    TauCalibrationSettings,
    resolve,
)
from ftmwpipeline.io.provenance import FIELD_SET_VERSION_ATTR
from ftmwpipeline.io.tau_calibration_settings_serialization import (
    SHAPE_RECOMMENDATION_FIELD_SET_VERSION,
    SHAPE_RECOMMENDATION_SETTINGS_PATH,
    STAGE2B_GAUSSIAN_SETTINGS_PATH,
    STAGE2B_LORENTZIAN_FIELD_SET_VERSION,
    STAGE2B_LORENTZIAN_SETTINGS_PATH,
    STAGE2B_TAU_SETTINGS_PATH,
    delete_shape_recommendation_record,
    load_shape_recommendation_record,
    load_tau_calibration_settings_from_h5,
    load_tau_producer_settings_from_h5,
    save_shape_recommendation_record,
    save_tau_calibration_settings_to_h5,
    save_tau_producer_settings_to_h5,
    shape_recommendation_provenance,
    tau_calibration_settings_present,
    tau_producer_settings_provenance,
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
        with atomic_write(empty_ftmw):
            save_tau_calibration_settings_to_h5(empty_ftmw, original)
        assert tau_calibration_settings_present(empty_ftmw)
        loaded = load_tau_calibration_settings_from_h5(empty_ftmw)
        assert loaded is not None
        assert loaded.stft.n_seg == original.stft.n_seg
        assert loaded.polish.polish_snr_cap == original.polish.polish_snr_cap
        assert (
            loaded.aggregation.min_contributors == original.aggregation.min_contributors
        )
        assert loaded.gaussian.tau_G_seeds == original.gaussian.tau_G_seeds
        assert (
            loaded.recommendation.pure_margin_threshold
            == original.recommendation.pure_margin_threshold
        )

    def test_round_trip_sparse_settings(self, empty_ftmw) -> None:
        s = TauCalibrationSettings()
        with atomic_write(empty_ftmw):
            save_tau_calibration_settings_to_h5(empty_ftmw, s)
        loaded = load_tau_calibration_settings_from_h5(empty_ftmw)
        assert loaded is not None
        assert loaded.is_empty()

    def test_round_trip_tuple_fields(self, empty_ftmw) -> None:
        s = TauCalibrationSettings()
        s.gaussian.tau_G_seeds = (50.0, 10.0, 2.0)
        s.band.band_labels = ("A", "B", "C")
        s.band.band_edges_mhz = (30000.0, 36000.0)
        with atomic_write(empty_ftmw):
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
        with atomic_write(empty_ftmw):
            save_tau_calibration_settings_to_h5(empty_ftmw, s, preset_name="defaults")
        with h5py.File(empty_ftmw, "r") as h5f:
            attrs = dict(h5f[STAGE2B_TAU_SETTINGS_PATH].attrs)
        assert attrs.get("preset_name") == "defaults"
        assert "creation_time" in attrs

    def test_overwrites_prior_block(self, empty_ftmw) -> None:
        s1 = TauCalibrationSettings()
        s1.stft.n_seg = 8
        with atomic_write(empty_ftmw):
            save_tau_calibration_settings_to_h5(empty_ftmw, s1)
        s2 = TauCalibrationSettings()
        s2.stft.n_seg = 20
        with atomic_write(empty_ftmw):
            save_tau_calibration_settings_to_h5(empty_ftmw, s2)
        loaded = load_tau_calibration_settings_from_h5(empty_ftmw)
        assert loaded is not None
        assert loaded.stft.n_seg == 20

    def test_hdf5_subgroup_layout(self, empty_ftmw) -> None:
        s = resolve()
        with atomic_write(empty_ftmw):
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
        with atomic_write(empty_ftmw):
            save_tau_calibration_settings_to_h5(empty_ftmw, s)
        # Manually delete one sub-group and verify the loader survives.
        with h5py.File(empty_ftmw, "a") as h5f:
            del h5f[STAGE2B_TAU_SETTINGS_PATH]["recommendation"]
        loaded = load_tau_calibration_settings_from_h5(empty_ftmw)
        assert loaded is not None
        assert loaded.stft.n_seg == 8
        assert loaded.recommendation.snr_min is None


_CONSUMED = {
    "start_us": 2.27,
    "end_us": 15.0,
    "trim_lo_mhz": 26500.0,
    "trim_hi_mhz": 27500.0,
}


def _resolved_with_n_seg(n_seg: int) -> TauCalibrationSettings:
    s = TauCalibrationSettings()
    s.stft.n_seg = n_seg
    return resolve(explicit=s)


class TestStage2bProducerRecords:
    """Each Stage 2b producer records what it used, in its own record."""

    @pytest.mark.parametrize("producer", ["lorentzian", "gaussian", "recommendation"])
    def test_absent_record_reads_none(self, empty_ftmw, producer) -> None:
        assert load_tau_producer_settings_from_h5(empty_ftmw, producer) is None
        assert tau_producer_settings_provenance(empty_ftmw, producer) is None

    @pytest.mark.parametrize("producer", ["lorentzian", "gaussian"])
    def test_twin_record_holds_exactly_its_fields(self, empty_ftmw, producer) -> None:
        resolved = resolve()
        with atomic_write(empty_ftmw):
            save_tau_producer_settings_to_h5(empty_ftmw, producer, resolved)
        loaded = load_tau_producer_settings_from_h5(empty_ftmw, producer)
        assert loaded is not None
        expected = PRODUCER_FIELDS[producer]
        assert set(loaded) == set(expected)
        for block, names in expected.items():
            assert set(loaded[block]) == set(names)
            for name in names:
                assert loaded[block][name] == getattr(getattr(resolved, block), name)
        prov = tau_producer_settings_provenance(empty_ftmw, producer)
        assert prov is not None and prov.is_current

    def test_resolved_unset_round_trips_as_none(self, empty_ftmw) -> None:
        resolved = resolve()
        assert resolved.stft.tau_max_us is None
        with atomic_write(empty_ftmw):
            save_tau_producer_settings_to_h5(empty_ftmw, "lorentzian", resolved)
        loaded = load_tau_producer_settings_from_h5(empty_ftmw, "lorentzian")
        assert loaded is not None
        assert "tau_max_us" in loaded["stft"]
        assert loaded["stft"]["tau_max_us"] is None
        assert loaded["band"]["band_edges_mhz"] is None

    def test_values_read_back_as_python_scalars(self, empty_ftmw) -> None:
        with atomic_write(empty_ftmw):
            save_tau_producer_settings_to_h5(empty_ftmw, "lorentzian", resolve())
        loaded = load_tau_producer_settings_from_h5(empty_ftmw, "lorentzian")
        assert loaded is not None
        assert type(loaded["stft"]["n_seg"]) is int
        assert type(loaded["stft"]["t_sigma"]) is float
        assert type(loaded["polish"]["polish"]) is bool

    def test_records_do_not_overwrite_each_other(self, empty_ftmw) -> None:
        with atomic_write(empty_ftmw):
            save_tau_producer_settings_to_h5(
                empty_ftmw, "lorentzian", _resolved_with_n_seg(12)
            )
        with atomic_write(empty_ftmw):
            save_tau_calibration_settings_to_h5(empty_ftmw, _resolved_with_n_seg(6))
        with atomic_write(empty_ftmw):
            save_tau_producer_settings_to_h5(
                empty_ftmw, "gaussian", _resolved_with_n_seg(8)
            )
        with atomic_write(empty_ftmw):
            save_shape_recommendation_record(
                empty_ftmw,
                _resolved_with_n_seg(14),
                consumed=_CONSUMED,
                tau_max_us=63.65,
                recommended_shape="gaussian",
                vote_rates={"exp": 0.1, "gauss": 0.9, "voigt": 0.0},
            )
        n_seg = {
            p: load_tau_producer_settings_from_h5(empty_ftmw, p)["stft"]["n_seg"]
            for p in ("lorentzian", "gaussian", "recommendation")
        }
        assert n_seg == {"lorentzian": 12, "gaussian": 8, "recommendation": 14}
        recipe = load_tau_calibration_settings_from_h5(empty_ftmw)
        assert recipe is not None and recipe.stft.n_seg == 6

    def test_distinct_paths(self) -> None:
        paths = {
            STAGE2B_TAU_SETTINGS_PATH,
            STAGE2B_LORENTZIAN_SETTINGS_PATH,
            STAGE2B_GAUSSIAN_SETTINGS_PATH,
            SHAPE_RECOMMENDATION_SETTINGS_PATH,
        }
        assert len(paths) == 4

    def test_twin_writer_refuses_the_recommendation(self, empty_ftmw) -> None:
        with pytest.raises(ValueError, match="twin"):
            with atomic_write(empty_ftmw):
                save_tau_producer_settings_to_h5(
                    empty_ftmw, "recommendation", resolve()
                )

    def test_unknown_producer_raises(self, empty_ftmw) -> None:
        with pytest.raises(ValueError, match="producer"):
            load_tau_producer_settings_from_h5(empty_ftmw, "voigt")

    def test_field_set_version_is_stamped(self, empty_ftmw) -> None:
        with atomic_write(empty_ftmw):
            save_tau_producer_settings_to_h5(empty_ftmw, "lorentzian", resolve())
        with h5py.File(empty_ftmw, "r") as h5f:
            attrs = h5f[STAGE2B_LORENTZIAN_SETTINGS_PATH].attrs
            assert int(attrs[FIELD_SET_VERSION_ATTR]) == (
                STAGE2B_LORENTZIAN_FIELD_SET_VERSION
            )


class TestShapeRecommendationRecord:
    def test_round_trip(self, empty_ftmw) -> None:
        resolved = _resolved_with_n_seg(14)
        with atomic_write(empty_ftmw):
            save_shape_recommendation_record(
                empty_ftmw,
                resolved,
                consumed=_CONSUMED,
                tau_max_us=63.65,
                recommended_shape="gaussian",
                vote_rates={"exp": 0.25, "gauss": 0.75, "voigt": 0.0},
            )
        rec = load_shape_recommendation_record(empty_ftmw)
        assert rec is not None
        assert rec.recommended_shape == "gaussian"
        assert rec.vote_rates == {"exp": 0.25, "gauss": 0.75, "voigt": 0.0}
        assert rec.tau_max_us == 63.65
        assert rec.consumed == _CONSUMED
        assert rec.settings["stft"]["n_seg"] == 14
        assert set(rec.settings) == set(PRODUCER_FIELDS["recommendation"])
        prov = shape_recommendation_provenance(empty_ftmw)
        assert prov is not None
        assert prov.version == SHAPE_RECOMMENDATION_FIELD_SET_VERSION

    def test_no_clear_winner_round_trips_as_none(self, empty_ftmw) -> None:
        with atomic_write(empty_ftmw):
            save_shape_recommendation_record(
                empty_ftmw,
                resolve(),
                consumed=_CONSUMED,
                tau_max_us=None,
                recommended_shape=None,
                vote_rates={},
            )
        rec = load_shape_recommendation_record(empty_ftmw)
        assert rec is not None
        assert rec.recommended_shape is None
        assert rec.tau_max_us is None
        assert rec.vote_rates == {}

    def test_missing_consumed_value_raises(self, empty_ftmw) -> None:
        with pytest.raises(ValueError, match="consumed"):
            with atomic_write(empty_ftmw):
                save_shape_recommendation_record(
                    empty_ftmw,
                    resolve(),
                    consumed={"start_us": 0.0},
                    tau_max_us=None,
                    recommended_shape=None,
                    vote_rates={},
                )

    def test_delete(self, empty_ftmw) -> None:
        with atomic_write(empty_ftmw):
            assert delete_shape_recommendation_record(empty_ftmw) is False
        with atomic_write(empty_ftmw):
            save_shape_recommendation_record(
                empty_ftmw,
                resolve(),
                consumed=_CONSUMED,
                tau_max_us=1.0,
                recommended_shape="lorentzian",
                vote_rates={},
            )
        with atomic_write(empty_ftmw):
            assert delete_shape_recommendation_record(empty_ftmw) is True
        assert load_shape_recommendation_record(empty_ftmw) is None
