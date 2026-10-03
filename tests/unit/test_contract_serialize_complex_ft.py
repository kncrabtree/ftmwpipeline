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
    assert out["freq_array"][0] == 1.0 and len(out["freq_array"]) == 4
    assert out["complex_spectrum"][1] == {"real": 1.0, "imag": 1.0}
    json.dumps(out, allow_nan=False)
