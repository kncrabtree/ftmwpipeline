"""``ComplexFT`` (not a dataclass) serializes as its declared contract fields."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ftmwpipeline.core.data_structures import ComplexFT
from ftmwpipeline.serialize import ArrayCollector, to_jsonable

pytestmark = [pytest.mark.unit]


def _ft():
    return ComplexFT(
        np.linspace(1.0, 2.0, 4),
        (np.arange(4) + 1j * np.arange(4)).astype(np.complex128),
        {"amplitude_scale": 1e6, "units_label": "uV", "pad_factor": 2},
    )


def test_arrays_go_to_the_sink_and_are_named():
    sink = ArrayCollector()
    out = to_jsonable(_ft(), arrays=sink)
    assert set(sink.arrays) == {"freq_array.npy", "complex_spectrum.npy"}
    assert sink.arrays["complex_spectrum.npy"].dtype == np.complex128
    assert out["freq_array"] == "freq_array.npy"
    assert out["complex_spectrum"] == "complex_spectrum.npy"
    json.dumps(out, allow_nan=False)


def test_arrays_inline_without_a_sink():
    out = to_jsonable(_ft())
    assert out["freq_array"][0] == 1.0 and len(out["freq_array"]) == 4
    assert out["complex_spectrum"][1] == {"real": 1.0, "imag": 1.0}
    json.dumps(out, allow_nan=False)


def test_a_fresh_ft_serializes_an_empty_invalidated_list():
    """``invalidated`` is a declared field of every ``ComplexFT``: ``[]`` when
    nothing was invalidated (a display FT, a loaded one). Mutation: drop the
    field from ``serialize._convert``."""
    out = to_jsonable(_ft())
    assert out["invalidated"] == []
    assert set(out) == {"freq_array", "complex_spectrum", "metadata", "invalidated"}
    assert "invalidated_absent" not in out
    json.dumps(out, allow_nan=False)


def test_invalidated_stage_names_are_serialized_in_order():
    ft = _ft()
    ft.invalidated = ("noise", "peaks", "windows")
    out = to_jsonable(ft, arrays=ArrayCollector())
    assert out["invalidated"] == ["noise", "peaks", "windows"]
    assert type(out["invalidated"]) is list  # a JSON array, not a tuple
    json.dumps(out, allow_nan=False)


def test_the_serialized_fields_are_exactly_the_declared_ones():
    from ftmwpipeline.contract import MANIFEST

    assert set(to_jsonable(_ft())) == set(MANIFEST.fields["ComplexFT"])


def test_a_computed_ft_reports_what_it_invalidated_on_the_wire(tmp_path):
    """``to_jsonable(compute_ft(...))`` carries the run's ``invalidated``."""
    import h5py

    import ftmwpipeline.api as ftmw
    from ftmwpipeline.pipeline import Pipeline

    src = tmp_path / "src.h5"
    t = np.arange(6325) * 0.002
    with h5py.File(src, "w") as f:
        f.attrs["ftmw_input_version"] = 1
        f.attrs["spacing_us"] = 0.002
        f.attrs["probe_freq_mhz"] = 40960.0
        f.create_dataset("fid", data=np.cos(2 * np.pi * 37.0 * t) * np.exp(-t / 3.0))
    path = tmp_path / "exp.ftmw"
    Pipeline.create(str(path), source=str(src), format_name="ftmw-hdf5")

    first = ftmw.compute_ft(path)
    assert to_jsonable(first)["invalidated"] == []
    ftmw.estimate_noise(path)
    changed = ftmw.compute_ft(path, trim=(40970.0, 41100.0))
    assert changed.invalidated == ("noise",)
    assert to_jsonable(changed)["invalidated"] == ["noise"]
    # The Pipeline class serializes the same way for the same change.
    again = Pipeline.open(path).compute_ft(trim=(40970.0, 41100.0))
    assert to_jsonable(again)["invalidated"] == []


def test_a_peak_list_serializes_as_the_plain_list():
    """A ``PeakList`` (a ``list`` that also carries ``invalidated``) keeps its
    plain-list wire form; the attribute is not part of it. Mutation: give
    ``PeakList`` an object form like ``ComplexFT``'s."""
    from ftmwpipeline.core.data_structures import PeakList

    peaks = PeakList([1.5, 2.5], invalidated=("windows",))  # items: any JSON-able
    out = to_jsonable(peaks)
    assert type(out) is list
    assert out == [1.5, 2.5] == to_jsonable([1.5, 2.5])
    assert "windows" not in json.dumps(out)
    assert to_jsonable(PeakList()) == []
