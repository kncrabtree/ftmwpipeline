"""Field-set provenance of the three Stage 2b producer records.

A record written at the current version reads as current; one without a version
(an old file) or at an older version is pre-provenance; one at a newer version
is flagged as newer. The fingerprint relies on this to raise
``incomplete_provenance`` rather than guess values for an old file.
"""

from __future__ import annotations

import h5py
import pytest

from ftmwpipeline._internal.atomic import atomic_write
from ftmwpipeline.core.tau_calibration_settings import resolve
from ftmwpipeline.io.provenance import FIELD_SET_VERSION_ATTR
from ftmwpipeline.io.tau_calibration_settings_serialization import (
    SHAPE_RECOMMENDATION_FIELD_SET_VERSION,
    SHAPE_RECOMMENDATION_SETTINGS_PATH,
    STAGE2B_GAUSSIAN_FIELD_SET_VERSION,
    STAGE2B_GAUSSIAN_SETTINGS_PATH,
    STAGE2B_LORENTZIAN_FIELD_SET_VERSION,
    STAGE2B_LORENTZIAN_SETTINGS_PATH,
    load_shape_recommendation_record,
    load_tau_producer_settings_from_h5,
    save_shape_recommendation_record,
    save_tau_producer_settings_to_h5,
    shape_recommendation_provenance,
    tau_producer_settings_provenance,
)

_CONSUMED = {
    "start_us": 2.27,
    "end_us": 15.0,
    "trim_lo_mhz": 26500.0,
    "trim_hi_mhz": 27500.0,
}

_PATHS = {
    "lorentzian": STAGE2B_LORENTZIAN_SETTINGS_PATH,
    "gaussian": STAGE2B_GAUSSIAN_SETTINGS_PATH,
    "recommendation": SHAPE_RECOMMENDATION_SETTINGS_PATH,
}
_VERSIONS = {
    "lorentzian": STAGE2B_LORENTZIAN_FIELD_SET_VERSION,
    "gaussian": STAGE2B_GAUSSIAN_FIELD_SET_VERSION,
    "recommendation": SHAPE_RECOMMENDATION_FIELD_SET_VERSION,
}


@pytest.fixture
def ftmw_file(tmp_path):
    p = tmp_path / "exp.ftmw"
    with h5py.File(p, "w") as h5f:
        h5f.create_group("placeholder")
    return str(p)


def _write(path: str, producer: str) -> None:
    resolved = resolve()
    if producer == "recommendation":
        with atomic_write(path):
            save_shape_recommendation_record(
                path,
                resolved,
                consumed=_CONSUMED,
                tau_max_us=63.65,
                recommended_shape="gaussian",
                vote_rates={"exp": 0.1, "gauss": 0.9, "voigt": 0.0},
            )
    else:
        with atomic_write(path):
            save_tau_producer_settings_to_h5(path, producer, resolved)


def _provenance(path: str, producer: str):
    if producer == "recommendation":
        return shape_recommendation_provenance(path)
    return tau_producer_settings_provenance(path, producer)


@pytest.mark.parametrize("producer", list(_PATHS))
class TestProducerRecordProvenance:
    def test_written_record_is_at_the_current_version(
        self, ftmw_file, producer
    ) -> None:
        # Mutation: the writer stops stamping the version attr -> version None.
        _write(ftmw_file, producer)
        prov = _provenance(ftmw_file, producer)
        assert prov is not None
        assert prov.version == _VERSIONS[producer] == prov.current_version
        assert prov.is_current
        assert not prov.is_pre_provenance and not prov.is_newer

    def test_unversioned_record_is_pre_provenance(self, ftmw_file, producer) -> None:
        # Mutation: provenance treats a missing attr as current.
        _write(ftmw_file, producer)
        with h5py.File(ftmw_file, "a") as h5f:
            del h5f[_PATHS[producer]].attrs[FIELD_SET_VERSION_ATTR]
        prov = _provenance(ftmw_file, producer)
        assert prov is not None
        assert prov.version is None
        assert prov.is_pre_provenance and not prov.is_current

    def test_older_version_is_pre_provenance(self, ftmw_file, producer) -> None:
        # Mutation: only "attr missing" counts as pre-provenance.
        _write(ftmw_file, producer)
        with h5py.File(ftmw_file, "a") as h5f:
            h5f[_PATHS[producer]].attrs[FIELD_SET_VERSION_ATTR] = (
                _VERSIONS[producer] - 1
            )
        prov = _provenance(ftmw_file, producer)
        assert prov is not None
        assert prov.is_pre_provenance and not prov.is_current

    def test_newer_version_is_flagged(self, ftmw_file, producer) -> None:
        _write(ftmw_file, producer)
        with h5py.File(ftmw_file, "a") as h5f:
            h5f[_PATHS[producer]].attrs[FIELD_SET_VERSION_ATTR] = (
                _VERSIONS[producer] + 1
            )
        prov = _provenance(ftmw_file, producer)
        assert prov is not None
        assert prov.is_newer and not prov.is_current

    def test_unversioned_record_still_reads(self, ftmw_file, producer) -> None:
        """An old record opens; the version, not a read failure, says it is old."""
        _write(ftmw_file, producer)
        with h5py.File(ftmw_file, "a") as h5f:
            del h5f[_PATHS[producer]].attrs[FIELD_SET_VERSION_ATTR]
        assert load_tau_producer_settings_from_h5(ftmw_file, producer) is not None
        if producer == "recommendation":
            assert load_shape_recommendation_record(ftmw_file) is not None

    def test_absent_record_has_no_provenance(self, ftmw_file, producer) -> None:
        # An absent record is not "pre-provenance": it is absent (None).
        assert _provenance(ftmw_file, producer) is None
