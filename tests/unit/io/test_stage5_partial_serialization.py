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
    # repr: float repr is exact, and NaN fields compare equal by text
    assert repr(o2.fit.audit_trail) == repr(out.fit.audit_trail)
    assert repr(o2.fit.knockouts) == repr(out.fit.knockouts)
    assert repr(o2.fixed_peaks) == repr(out.fixed_peaks)
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


def _chain_partial() -> PartialWalk:
    """Windows 3 -> 5 -> 8 (each read its predecessor) and an independent 9."""
    outs = {}
    recs = {}
    for wid in (3, 5, 8, 9):
        outs[wid], recs[wid] = _outcome(wid)
    return PartialWalk(
        "initial",
        [3, 9, 5, 8],
        outs,
        recs,
        False,
        "dag",
        12,
        preds={3: set(), 9: set(), 5: {3}, 8: {5}},
    )


def test_a_refused_window_takes_every_window_that_read_it():
    """The kept set stays closed over the windows a kept window read: a resume
    must never carry a dependent whose primary it refits."""
    partial = _chain_partial()
    partial.outcomes[3]._unexpected = {1}  # type: ignore[union-attr]
    encoded, refused = encode_stage5_partial(partial)
    assert [e.window_id for e in encoded] == [9]
    assert sorted(refused) == [3, 5, 8]


def test_a_refused_dependent_keeps_its_primary():
    partial = _chain_partial()
    partial.outcomes[8]._unexpected = {1}  # type: ignore[union-attr]
    encoded, refused = encode_stage5_partial(partial)
    assert [e.window_id for e in encoded] == [3, 9, 5]
    assert refused == [8]


def test_any_encoding_failure_refuses_only_that_window(monkeypatch):
    """Not only the codec's own error: whatever encoding a window raises (short
    of a BaseException) refuses that window and its dependents, so the cancel
    or callback error is what the caller sees."""
    import ftmwpipeline.io.stage5_partial_serialization as ser

    real = ser.encode_graph

    def flaky(root):
        if root["outcome"].window_id == 5:
            raise TypeError("boom")
        return real(root)

    monkeypatch.setattr(ser, "encode_graph", flaky)
    encoded, refused = encode_stage5_partial(_chain_partial())
    assert [e.window_id for e in encoded] == [3, 9]
    assert sorted(refused) == [5, 8]


def test_a_window_whose_write_fails_is_dropped_with_its_dependents(tmp_path):
    import dataclasses as dc

    encoded, _ = encode_stage5_partial(_chain_partial())
    # An object buffer h5py cannot store: the write of window 5 fails.
    encoded = [
        (
            dc.replace(e, buffers=[np.array([object()], dtype=object)])
            if e.window_id == 5
            else e
        )
        for e in encoded
    ]
    path = tmp_path / "p.h5"
    with h5py.File(path, "w") as h5f:
        assert write_stage5_partial(h5f, encoded, {"settings": "{}"}) == [3, 9]
    with h5py.File(path, "r") as h5f:
        assert sorted(h5f["stage5_partial/windows"]) == ["w3", "w9"]
        assert read_stage5_partial_counts(h5f) == {3: 2, 9: 2}
        assert read_stage5_partial_windows(h5f).order == [3, 9]


def test_counts_of_non_integer_window_ids_are_not_read(tmp_path):
    """``window_status`` reads the counts: a malformed dataset there is no
    partial fit to report (the rows stay ``not_run``), never an error."""
    out, record = _outcome(3)
    partial = PartialWalk("initial", [3], {3: out}, {3: record}, False, "dag", 4)
    encoded, _ = encode_stage5_partial(partial)
    path = tmp_path / "p.h5"
    with h5py.File(path, "w") as h5f:
        write_stage5_partial(h5f, encoded, {})
        del h5f["stage5_partial/window_ids"]
        h5f["stage5_partial/window_ids"] = np.array([3.5])
    with h5py.File(path, "r") as h5f:
        assert read_stage5_partial_counts(h5f) is None
        with pytest.raises(PartialCodecError):
            read_stage5_partial_windows(h5f)


# ---- hostile and malformed graphs ---------------------------------------------------------

_BUF = (["<f8"], [np.zeros(3)])


@pytest.mark.parametrize(
    "graph",
    [
        {"root": {"$r": 0}, "nodes": [{"$a": ["<f8", 0, [-1]]}]},  # negative shape
        {"root": {"$r": 0}, "nodes": [{"$a": ["<f8", -1, [2]]}]},  # negative offset
        {"root": {"$r": 0}, "nodes": [{"$a": ["<f8", 0, [10**9, 10**9]]}]},  # huge
        {"root": {"$r": 0}, "nodes": [{"$a": ["<f4", 0, [1]]}]},  # no such buffer
        {"root": {"$r": 0}, "nodes": [{"$a": ["<f8", 0]}]},  # short spec
        {"root": {"$r": 0}, "nodes": [{"$o": "ModelPeak", "f": [1]}]},
        {"root": {"$r": 0}, "nodes": [{"$o": "ModelPeak", "f": {}}]},  # lacks fields
        {"root": {"$r": 0}, "nodes": {"a": 1}},
        {"root": {"$r": -1}, "nodes": [{"$l": []}]},
        {"root": {"$r": True}, "nodes": [{"$l": []}]},
        {"root": {"$t": 5}, "nodes": []},
        {"root": {"$e": "PeakShape", "v": "NOPE"}, "nodes": []},
        {"root": {"$e": "Popen", "v": "x"}, "nodes": []},
        {"root": {"$g": "|S4", "v": "abcd"}, "nodes": []},
        {"root": {"$g": "<M8[s]", "v": 1}, "nodes": []},
        {"root": [1, 2], "nodes": []},  # a bare list is not a node
        {"root": {}, "nodes": []},
        {"nodes": []},
        [],
        "just a string",
    ],
)
def test_a_malformed_graph_is_refused_with_the_codec_error(graph):
    with pytest.raises(PartialCodecError):
        decode_graph(json.dumps(graph), *_BUF)


def test_a_graph_nested_without_bound_is_refused():
    text = '{"root": ' + "[" * 100000 + "1" + "]" * 100000 + ', "nodes": []}'
    with pytest.raises(PartialCodecError):
        decode_graph(text, [], [])


@pytest.mark.parametrize("text", ["", "{", "not json", '{"root": NaN junk'])
def test_an_unreadable_graph_is_refused(text):
    with pytest.raises(PartialCodecError):
        decode_graph(text, [], [])


def test_buffers_must_match_their_dtypes():
    text, names, buffers = encode_graph(np.arange(4.0))
    with pytest.raises(PartialCodecError):
        decode_graph(text, names, [])  # fewer buffers than dtypes
    with pytest.raises(PartialCodecError):
        decode_graph(text, names, [np.arange(4, dtype=np.int64)])  # wrong dtype
    with pytest.raises(PartialCodecError):
        decode_graph(text, names, [np.zeros((2, 2))])  # not one-dimensional


# Every malformed scalar or key is the codec's error, never a raw ValueError /
# IndexError / TypeError / OverflowError.
@pytest.mark.parametrize(
    "graph",
    [
        {"root": {"$g": "<f8", "v": "abc"}, "nodes": []},
        {"root": {"$g": "<f8", "v": [1.0, 2.0]}, "nodes": []},
        {"root": {"$g": "<f8"}, "nodes": []},
        {"root": {"$g": "|u1", "v": -1}, "nodes": []},
        {"root": {"$g": "<i8", "v": 1e300}, "nodes": []},
        {"root": {"$g": "<c16", "v": [1]}, "nodes": []},
        {"root": {"$g": "<c16", "v": ["a", 1]}, "nodes": []},
        {"root": {"$g": "not a dtype", "v": 1}, "nodes": []},
        {"root": {"$c": ["a", "b"]}, "nodes": []},
        {"root": {"$c": [1]}, "nodes": []},
        {"root": {"$e": "PeakShape"}, "nodes": []},
        {"root": {"$e": "PeakShape", "v": 3}, "nodes": []},
        {
            "root": {"$r": 0},
            "nodes": [{"$d": [[{"$r": 1}, 1]]}, {"$l": []}],  # a list as a key
        },
        {"root": {"$r": 0}, "nodes": [{"$a": ["<f8", True, [1]]}]},
    ],
    ids=[
        "scalar_text",
        "scalar_list",
        "scalar_missing",
        "scalar_out_of_range",
        "scalar_overflow",
        "complex_scalar_short",
        "complex_scalar_text",
        "bad_dtype",
        "complex_text",
        "complex_short",
        "enum_member_missing",
        "enum_member_not_text",
        "unhashable_key",
        "bool_offset",
    ],
)
def test_malformed_scalars_and_keys_are_refused_with_the_codec_error(graph):
    with pytest.raises(PartialCodecError):
        decode_graph(json.dumps(graph), *_BUF)


def _peak_node(**fields):
    base = {"amplitude": 1.0, "offset_mhz": 0.5, "phase": 0.0, "peak_uid": None}
    base.update(fields)
    return {"root": {"$r": 0}, "nodes": [{"$o": "ModelPeak", "f": base}]}


def test_a_well_formed_object_node_decodes():
    assert decode_graph(json.dumps(_peak_node()), *_BUF) == ModelPeak(1.0, 0.5, 0.0)


@pytest.mark.parametrize(
    "graph",
    [
        _peak_node(_evil=1),  # not a declared field, not a known stash
        _peak_node(__class__=1),
        _peak_node(amplitude="loud"),
        _peak_node(phase=None),
        _peak_node(peak_uid=1.5),
        _peak_node(offset_mhz={"$c": [1.0, 2.0]}),
    ],
    ids=[
        "undeclared",
        "dunder",
        "text_for_float",
        "none_for_float",
        "float_for_int",
        "complex_for_float",
    ],
)
def test_undeclared_or_wrongly_typed_attributes_are_refused(graph):
    with pytest.raises(PartialCodecError):
        decode_graph(json.dumps(graph), *_BUF)


def test_wrongly_typed_outcome_fields_are_refused():
    out, record = _outcome()
    text, names, buffers = encode_graph({"outcome": out, "record": record})
    doc = json.loads(text)
    node = next(
        n
        for n in doc["nodes"]
        if isinstance(n, dict) and n.get("$o") == "WindowOutcome"
    )
    for name, value in [
        ("fit", 7),
        ("offset_grid_mhz", "abc"),
        ("fixed_peaks", {"$t": [1]}),
        ("_center_mhz", "here"),
    ]:
        bad = json.loads(json.dumps(doc))
        target = next(
            n
            for n in bad["nodes"]
            if isinstance(n, dict) and n.get("$o") == "WindowOutcome"
        )
        target["f"][name] = value
        with pytest.raises(PartialCodecError):
            decode_graph(json.dumps(bad), names, buffers)
    assert node["f"]["_center_mhz"] == 100.0


def test_an_undeclared_attribute_is_not_written():
    out, record = _outcome()
    out.extra = 1.0  # type: ignore[attr-defined]
    with pytest.raises(PartialCodecError):
        encode_graph({"outcome": out, "record": record})


def test_nothing_in_the_graph_is_imported_or_called(tmp_path):
    """A graph naming a callable builds nothing and runs nothing."""
    sentinel = tmp_path / "ran"
    graph = {
        "root": {"$r": 0},
        "nodes": [
            {"$o": "builtins.exec", "f": {"code": f"open({str(sentinel)!r}, 'w')"}}
        ],
    }
    with pytest.raises(PartialCodecError):
        decode_graph(json.dumps(graph), [], [])
    assert not sentinel.exists()


def test_reading_a_partial_fit_with_a_missing_window_group_is_refused(tmp_path):
    out, record = _outcome(3)
    partial = PartialWalk("initial", [3], {3: out}, {3: record}, False, "dag", 4)
    encoded, _ = encode_stage5_partial(partial)
    path = tmp_path / "p.h5"
    with h5py.File(path, "w") as h5f:
        write_stage5_partial(h5f, encoded, {})
        del h5f["stage5_partial/windows/w3"]
    with h5py.File(path, "r") as h5f:
        with pytest.raises(PartialCodecError):
            read_stage5_partial_windows(h5f)


def test_the_provenance_of_another_layout_version_is_not_read(tmp_path):
    path = tmp_path / "p.h5"
    with h5py.File(path, "w") as h5f:
        write_stage5_partial(h5f, [], {"settings": "{}"})
        assert read_stage5_partial_provenance(h5f) is not None
        raw = json.loads(bytes(h5f["stage5_partial/provenance"][()]).decode())
        raw["format_version"] = 0
        del h5f["stage5_partial/provenance"]
        h5f.create_dataset(
            "stage5_partial/provenance",
            data=np.frombuffer(json.dumps(raw).encode(), dtype=np.uint8),
        )
        assert read_stage5_partial_provenance(h5f) is None


def test_the_stored_datasets_are_plain_numeric_arrays(tmp_path):
    out, record = _outcome(3)
    partial = PartialWalk("initial", [3], {3: out}, {3: record}, False, "dag", 4)
    encoded, _ = encode_stage5_partial(partial)
    path = tmp_path / "p.h5"
    with h5py.File(path, "w") as h5f:
        write_stage5_partial(h5f, encoded, {"settings": "{}"})
    with h5py.File(path, "r") as h5f:
        found = []
        h5f["stage5_partial"].visititems(
            lambda n, o: found.append(o) if isinstance(o, h5py.Dataset) else None
        )
        assert found
        for ds in found:
            assert ds.dtype.kind in "buifc" and not ds.dtype.hasobject, ds.name
            assert ds.dtype.names is None and h5py.check_dtype(vlen=ds.dtype) is None
