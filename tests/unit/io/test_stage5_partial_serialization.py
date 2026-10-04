"""The Stage 5 partial-fit codec: faithful, allow-listed, never executes code.

A resumed fit reads each carried window's ``WindowOutcome`` back exactly as the
walk left it (``dev-docs/CONTRACT_STRATEGY.md`` §Stage 5 partial fits); a
``.ftmw`` may come from anyone, so reading one must not run code from it.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from ftmwpipeline.core.peak_shape import PeakShape
from ftmwpipeline.fitting.active_ft import PointMap
from ftmwpipeline.fitting.peak_model import ModelPeak
from ftmwpipeline.fitting.plan_execution import (
    FrozenPeak,
    PartialWalk,
    RescueEvent,
    ThawEvent,
    WalkRecord,
    WindowOutcome,
)
from ftmwpipeline.fitting.residual_screening import ResidualPeakCandidate
from ftmwpipeline.fitting.spur_detection import SpurMaskSpec
from ftmwpipeline.fitting.window_fit import (
    AddStep,
    ConservativeFitResult,
    KnockoutResult,
    ParameterErrors,
    WindowFitResult,
)
from ftmwpipeline.io.stage5_partial_serialization import (
    PartialCodecError,
    decode_graph,
    encode_graph,
    encode_stage5_partial,
    read_stage5_partial_counts,
    read_stage5_partial_provenance,
    read_stage5_partial_windows,
    write_stage5_partial,
)

pytestmark = pytest.mark.unit


def _outcome(wid: int = 7) -> tuple[WindowOutcome, WalkRecord]:
    grid = np.linspace(-1.0, 1.0, 33)
    data = np.exp(1j * grid) * (1.0 + 0.25 * grid)
    inner = WindowFitResult(
        success=True,
        peaks=[ModelPeak(1.5, 0.1, 0.2, peak_uid=12345), ModelPeak(0.5, -0.3, 1.0)],
        peak_errors=[
            ParameterErrors(0.01, 0.002, 0.03),
            ParameterErrors(*[np.nan] * 3),
        ],
        tau_us=2.5,
        tau_error=None,
        fit_tau=True,
        cost=1.25,
        chi_squared=2.5,
        n_data=66,
        n_params=7,
        n_function_evals=42,
        fitted_spectrum=data * 0.9,
        residual=data * 0.1,
        covariance=np.arange(49, dtype=float).reshape(7, 7),
        shape=PeakShape.GAUSSIAN,
        baseline_order=1,
        baseline_coeffs=np.array([1 + 2j, -0.5j]),
        baseline_offset_scale=0.75,
    )
    fit = ConservativeFitResult(
        fit=inner,
        audit_trail=[
            AddStep(1, 0.1, 3.0, 2.0, 4.0, 0.01, 5.0, 4.5, True, "accept", "why")
        ],
        knockouts=[KnockoutResult(0, 0.1, 9.0, 3.0, True)],
    )
    cand = ResidualPeakCandidate(4, 100.0, 2.0, 5.5, 3.0, None, None, None, False)
    rescue = RescueEvent(
        wid, 0, 1, 1, 1, 0, 0, 0, 3.0, 2.0, 2.5, 2.4, True, "ok", [cand]
    )
    thaw = ThawEvent(wid, 3, 9, 101.5, "low", 4.0, 2.0, False, "no")
    frozen = FrozenPeak(9, 3, ModelPeak(0.2, 1.4, 0.0), 101.5, True, False)
    out = WindowOutcome(
        window_id=wid,
        fit=fit,
        fixed_peaks=[frozen],
        offset_grid_mhz=grid,
        complex_spectrum=data,
        rms_noise=np.full(grid.shape, 0.05),
        background=np.zeros(grid.shape, dtype=np.complex128),
        full_fitted_spectrum=data * 0.9,
        full_residual=data * 0.1,
        thaw_events=[thaw],
        rescue_events=[rescue],
        edge_coherence_low=float("nan"),
        edge_coherence_high=-0.0,
        baseline_applied=True,
        baseline_order=1,
        baseline_coeffs=inner.baseline_coeffs,  # aliased, as the walk leaves it
        baseline_offset_scale=0.75,
        baseline_edge_coherence=1.0,
    )
    out._center_mhz = 100.0  # type: ignore[attr-defined]
    out._spur_mask = SpurMaskSpec((0.25,), 0.01)  # type: ignore[attr-defined]
    out._ck_for_window = {  # type: ignore[attr-defined]
        "shape": PeakShape.GAUSSIAN,
        "max_peaks": 6,
        "tau_apodization_us": None,
        "point_map": PointMap(origin_points=12.5, points_per_mhz=3.0),
        "weights": (1, np.float64(2.0), np.int64(3)),
    }
    out._acquisition_us = 10.0  # type: ignore[attr-defined]
    record = WalkRecord(thaws=[thaw], rescues=[rescue], cleanups=[{"window_id": wid}])
    return out, record


def test_round_trip_is_faithful_and_keeps_aliasing():
    out, record = _outcome()
    text, names, buffers = encode_graph({"outcome": out, "record": record})
    back = decode_graph(text, names, buffers)
    o2, r2 = back["outcome"], back["record"]

    assert isinstance(o2, WindowOutcome) and isinstance(o2.fit.fit, WindowFitResult)
    # Values, bit for bit (NaN and -0.0 included), with their dtypes.
    for name in ("offset_grid_mhz", "complex_spectrum", "rms_noise", "background"):
        a, b = getattr(out, name), getattr(o2, name)
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes()
    assert o2.fit.fit.covariance.tobytes() == out.fit.fit.covariance.tobytes()
    assert np.isnan(o2.edge_coherence_low)
    assert str(o2.edge_coherence_high) == "-0.0"
    assert o2.fit.fit.peaks == out.fit.fit.peaks
    assert o2.fit.fit.peak_errors[1].amplitude != o2.fit.fit.peak_errors[1].amplitude
    assert o2.fit.fit.shape is PeakShape.GAUSSIAN
    assert o2.fit.audit_trail == out.fit.audit_trail
    assert o2.fit.knockouts == out.fit.knockouts
    assert o2.fixed_peaks == out.fixed_peaks
    # The walk's private attributes come back too.
    assert o2._center_mhz == 100.0
    assert o2._spur_mask == out._spur_mask
    ck = o2._ck_for_window
    assert ck["shape"] is PeakShape.GAUSSIAN and ck["point_map"] == PointMap(12.5, 3.0)
    assert ck["weights"] == (1, 2.0, 3)
    assert type(ck["weights"][1]) is np.float64 and type(ck["weights"][2]) is np.int64
    # Shared references stay shared (the rescue refresh mutates them in place).
    assert r2.rescues[0] is o2.rescue_events[0]
    assert r2.thaws[0] is o2.thaw_events[0]
    assert o2.baseline_coeffs is o2.fit.fit.baseline_coeffs
    assert r2.cleanups == [{"window_id": 7}]


@pytest.mark.parametrize(
    "value",
    [
        object(),
        {1, 2},
        np.array(["a", "b"]),
        np.array([object()], dtype=object),
        lambda: 0,
    ],
)
def test_values_outside_the_allow_list_are_refused(value):
    with pytest.raises(PartialCodecError):
        encode_graph({"x": value})


@pytest.mark.parametrize(
    "graph",
    [
        {"root": {"$r": 0}, "nodes": [{"$o": "os.system", "f": {}}]},
        {"root": {"$r": 0}, "nodes": [{"$o": "Popen", "f": {}}]},
        {"root": {"$e": "Signals", "v": "SIGKILL"}, "nodes": []},
        {"root": {"$g": "|O", "v": 1}, "nodes": []},
        {"root": {"$r": 5}, "nodes": []},
        {"root": {"$r": 0}, "nodes": [{"$a": ["<f8", 0, [10]]}]},
    ],
)
def test_a_hostile_graph_is_refused_without_running_anything(graph):
    with pytest.raises(PartialCodecError):
        decode_graph(json.dumps(graph), ["<f8"], [np.zeros(3)])


def test_an_object_dtype_buffer_is_refused():
    text, _names, _buffers = encode_graph([1.0])
    with pytest.raises(PartialCodecError):
        decode_graph(text, ["|O"], [np.array([None], dtype=object)])


def test_write_read_round_trip_in_hdf5(tmp_path):
    out, record = _outcome(3)
    partial = PartialWalk(
        phase="initial",
        order=[3, 5],
        outcomes={3: out, 5: None},
        records={3: record, 5: WalkRecord(cleanups=[{"window_id": 5}])},
        accepted_thaw=False,
        walk_mode="dag",
        n_windows=9,
    )
    encoded, refused = encode_stage5_partial(partial)
    assert refused == [] and [e.window_id for e in encoded] == [3, 5]
    path = tmp_path / "p.h5"
    with h5py.File(path, "w") as h5f:
        assert write_stage5_partial(h5f, encoded, {"settings": "{}"}) == [3, 5]
    with h5py.File(path, "r") as h5f:
        prov = read_stage5_partial_provenance(h5f)
        assert prov is not None and prov["settings"] == "{}"
        assert read_stage5_partial_counts(h5f) == {3: 2, 5: 0}
        carried = read_stage5_partial_windows(h5f)
    assert carried.order == [3, 5]
    assert carried.outcomes[5] is None
    assert carried.outcomes[3].fit.fit.peaks == out.fit.fit.peaks
    assert carried.records[5].cleanups == [{"window_id": 5}]


def test_a_window_that_cannot_be_encoded_is_not_kept():
    out, record = _outcome(3)
    out._unexpected = {1, 2}  # type: ignore[attr-defined]
    partial = PartialWalk("initial", [3], {3: out}, {3: record}, False, "dag", 4)
    encoded, refused = encode_stage5_partial(partial)
    assert encoded == [] and refused == [3]
