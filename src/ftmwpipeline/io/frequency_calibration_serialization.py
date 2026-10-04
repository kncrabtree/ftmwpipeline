"""HDF5 (de)serialization for the file-level frequency-calibration record.

The ``/frequency_calibration`` group is file-level provenance (a sibling of the
source metadata, not a tracked pipeline stage). It homes the single
user-declared systematic accuracy floor (``sigma_floor_khz``) so any reported
``sigma_f`` is reproducible from the ``.ftmw`` record alone. Files written
before this record existed open fine: the loader returns the default
(``sigma_floor_khz = 0.0``), and
:func:`frequency_calibration_provenance` reports them as having no record.
"""

from __future__ import annotations

from typing import Optional

import h5py

from ..core.data_structures import FrequencyCalibration
from .provenance import RecordProvenance, group_provenance, write_field_set_version

GROUP_NAME = "frequency_calibration"

#: Field-set version of the ``/frequency_calibration`` record (see
#: :mod:`ftmwpipeline.io.provenance`).
FREQUENCY_CALIBRATION_FIELD_SET_VERSION = 1


def save_frequency_calibration_to_hdf5(
    calibration: FrequencyCalibration, h5f: h5py.File
) -> None:
    """Write *calibration* to the file-level ``/frequency_calibration`` group."""
    if GROUP_NAME in h5f:
        del h5f[GROUP_NAME]
    grp = h5f.create_group(GROUP_NAME)
    grp.attrs["sigma_floor_khz"] = float(calibration.sigma_floor_khz)
    write_field_set_version(grp.attrs, FREQUENCY_CALIBRATION_FIELD_SET_VERSION)


def load_frequency_calibration_from_hdf5(h5f: h5py.File) -> FrequencyCalibration:
    """Read the ``/frequency_calibration`` record, or the default when absent."""
    if GROUP_NAME not in h5f:
        return FrequencyCalibration()
    grp = h5f[GROUP_NAME]
    return FrequencyCalibration(
        sigma_floor_khz=float(grp.attrs.get("sigma_floor_khz", 0.0))
    )


def frequency_calibration_provenance(h5f: h5py.File) -> Optional[RecordProvenance]:
    """The record's field-set version against
    :data:`FREQUENCY_CALIBRATION_FIELD_SET_VERSION`, or ``None`` when the file
    holds no record (the floor was never declared and reads as the default)."""
    group = h5f.get(GROUP_NAME)
    if not isinstance(group, h5py.Group):
        return None
    return group_provenance(group, FREQUENCY_CALIBRATION_FIELD_SET_VERSION)
