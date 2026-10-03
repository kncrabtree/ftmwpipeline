"""The report reads the Stage 1 record through Stage 1's own codec.

An unset trim is persisted as the ``__None__`` sentinel and an older record
carries the whole bundle as one JSON ``parameters`` blob. A report that read the
raw attrs would crash on the first and see no band on the second, so both are
exercised here against what the Stage 1 writer actually emits.
"""

from __future__ import annotations

import json

import h5py
import pytest

from ftmwpipeline._internal.report_impl import _assemble_summary
from ftmwpipeline.core.settings import FT_PROCESSING_PATH, FTSettings


def _rewrite_ft_record(path, settings: FTSettings) -> None:
    with h5py.File(path, "a") as h5f:
        group = h5f[FT_PROCESSING_PATH]
        for key in list(group.attrs):
            del group.attrs[key]
        group.attrs.update(settings.to_attrs())


def _persisted(path) -> FTSettings:
    with h5py.File(path, "r") as h5f:
        return FTSettings.from_attrs(dict(h5f[FT_PROCESSING_PATH].attrs))


def test_report_summary_survives_an_unset_trim(stage5_reviewed_file):
    persisted = _persisted(stage5_reviewed_file)
    _rewrite_ft_record(
        stage5_reviewed_file,
        FTSettings(
            start_us=persisted.start_us,
            end_us=persisted.end_us,
            units_power=persisted.units_power,
            trim=None,
        ),
    )
    model = _assemble_summary(stage5_reviewed_file)
    assert model.band_lo_mhz is None
    assert model.band_hi_mhz is None


def test_report_summary_reads_a_blob_only_record(stage5_reviewed_file):
    expected = _assemble_summary(stage5_reviewed_file)
    assert expected.band_lo_mhz is not None  # the fixture is trimmed

    with h5py.File(stage5_reviewed_file, "a") as h5f:
        group = h5f[FT_PROCESSING_PATH]
        bundle = {
            k: (v.item() if hasattr(v, "item") else v) for k, v in group.attrs.items()
        }
        for key in list(group.attrs):
            del group.attrs[key]
        group.attrs["parameters"] = json.dumps(bundle)

    model = _assemble_summary(stage5_reviewed_file)
    assert model.band_lo_mhz == pytest.approx(expected.band_lo_mhz)
    assert model.band_hi_mhz == pytest.approx(expected.band_hi_mhz)
