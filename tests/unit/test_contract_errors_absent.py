"""Absent, the typed error family, and the contract serializer.

Spec: ``dev-docs/CONTRACT_STRATEGY.md`` §Missing values, §Errors, and the
serialization rules (valid JSON always; one missing-value rule).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import enum
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from ftmwpipeline.contract import (
    CONTRACT_VERSION,
    ERROR_SCHEMA,
    STATUS_NOT_RUN,
    STATUS_PRESENT,
    STATUS_UNDEFINED,
    Absent,
    AnalysisEpochMismatchError,
    IncompleteProvenanceError,
    NotFoundError,
    PipelineCompatibilityError,
    PipelineCorruptionError,
    PipelineExistsError,
    PipelineFileError,
    StageDependencyError,
)
from ftmwpipeline.serialize import (
    ArrayCollector,
    absent_column,
    to_jsonable,
    with_status_columns,
)

pytestmark = [pytest.mark.unit]


def _wire(obj, **kw):
    """Convert and prove the result is strict JSON (no NaN/Infinity tokens)."""
    out = to_jsonable(obj, **kw)
    return json.loads(json.dumps(out, allow_nan=False))


# ---- Absent ---------------------------------------------------------------


def test_absent_members_and_wire_spelling():
    assert {a.name for a in Absent} == {"NOT_RUN", "UNDEFINED"}
    assert Absent.NOT_RUN.value == "not_run"
    assert Absent.UNDEFINED.value == "undefined"


def test_absent_status_codes():
    assert STATUS_PRESENT == 0
    assert Absent.NOT_RUN.status == STATUS_NOT_RUN == 1
    assert Absent.UNDEFINED.status == STATUS_UNDEFINED == 2


def test_absent_is_not_none_like():
    assert Absent.NOT_RUN is not None
    assert Absent.NOT_RUN is not Absent.UNDEFINED
    assert Absent.NOT_RUN != Absent.UNDEFINED


def test_contract_version_is_int():
    assert isinstance(CONTRACT_VERSION, int)


# ---- errors ---------------------------------------------------------------


class _Env:
    def __init__(self, epoch):
        self.analysis_epoch = epoch

    def summary(self):
        return f"epoch {self.analysis_epoch}"


def _errors():
    p = Path("x.ftmw")
    return [
        (
            PipelineExistsError(p, "a", "b"),
            "file_exists",
            {},
            (),
        ),
        (
            StageDependencyError("fit", ["stage3"], p, command="ftmwpipeline peaks"),
            "stage_not_run",
            {"missing_dependencies": ["stage3"], "command": "ftmwpipeline peaks"},
            (ValueError,),
        ),
        (PipelineCorruptionError(p, "bad"), "file_corrupt", {}, ()),
        (
            PipelineCompatibilityError(p, "9", "1"),
            "file_incompatible",
            {"file_version": "9", "supported_version": "1"},
            (),
        ),
        (
            AnalysisEpochMismatchError(p, _Env(2), _Env(3)),
            "epoch_mismatch",
            {"file_epoch": 2, "current_epoch": 3},
            (ValueError,),
        ),
        (
            NotFoundError("window", [4, 9]),
            "not_found",
            {"kind": "window", "ids": [4, 9]},
            (KeyError,),
        ),
        (
            IncompleteProvenanceError(["stage2b.shape"]),
            "incomplete_provenance",
            {"missing": ["stage2b.shape"]},
            (ValueError,),
        ),
    ]


@pytest.mark.parametrize("err,code,fields,bases", _errors(), ids=lambda v: str(type(v)))
def test_error_contract(err, code, fields, bases):
    assert isinstance(err, PipelineFileError)
    assert err.code == code
    d = err.to_dict()
    assert d["schema"] == ERROR_SCHEMA == "ftmw/error@1"
    assert d["code"] == code
    assert d["message"] == str(err)
    for k, v in fields.items():
        assert d[k] == v
    # exactly schema, code, message and the declared fields
    assert set(d) == {"schema", "code", "message", *fields}
    assert list(d)[:3] == ["schema", "code", "message"]
    json.dumps(d, allow_nan=False)
    for base in bases:
        assert isinstance(err, base)


def test_legacy_epoch_none_is_valid_json():
    e = AnalysisEpochMismatchError("f", _Env(None), _Env(3))
    d = json.loads(json.dumps(e.to_dict(), allow_nan=False))
    assert d["file_epoch"] is None and d["current_epoch"] == 3


def test_error_codes_unique():
    codes = [c for _, c, _, _ in _errors()]
    assert len(set(codes)) == len(codes)


def test_not_found_carries_every_id():
    e = NotFoundError("window", [3, 7, 11])
    assert e.kind == "window" and e.ids == [3, 7, 11]
    assert "3" in str(e) and "11" in str(e)
    assert e.to_dict()["ids"] == [3, 7, 11]


def test_not_found_str_is_plain_message_and_keyerror_still_catchable():
    e = NotFoundError("peak", ["ab12"], message="no such peak")
    assert str(e) == "no such peak"
    with pytest.raises(KeyError):
        raise e


def test_not_found_rejects_bare_string_ids():
    with pytest.raises(TypeError):
        NotFoundError("window", "12")  # type: ignore[arg-type]


def test_incomplete_provenance_rejects_bare_string():
    with pytest.raises(TypeError):
        IncompleteProvenanceError("stage2b.shape")  # type: ignore[arg-type]


def test_valueerror_clauses_keep_working():
    for exc in (
        StageDependencyError("s", ["x"], Path("f")),
        IncompleteProvenanceError(["a"]),
        AnalysisEpochMismatchError("f", _Env(1), _Env(2)),
    ):
        with pytest.raises(ValueError):
            raise exc


@pytest.mark.parametrize(
    "err",
    [NotFoundError("window", [1, 2]), IncompleteProvenanceError(["a", "b"])],
    ids=["not_found", "incomplete_provenance"],
)
def test_keyword_message_errors_pickle(err):
    clone = pickle.loads(pickle.dumps(err))
    assert type(clone) is type(err)
    assert clone.to_dict() == err.to_dict()


def test_serializer_accepts_error_instance():
    e = NotFoundError("window", [1])
    assert to_jsonable({"error": e})["error"] == e.to_dict()


# ---- Absent on the wire ---------------------------------------------------


def test_absent_field_becomes_null_with_sibling():
    out = _wire({"p": Absent.NOT_RUN, "q": Absent.UNDEFINED, "r": 1.5})
    assert out == {
        "p": None,
        "p_absent": "not_run",
        "q": None,
        "q_absent": "undefined",
        "r": 1.5,
    }


def test_sibling_follows_its_field_and_present_has_none():
    out = to_jsonable({"a": 1, "b": Absent.NOT_RUN, "c": 2})
    assert list(out) == ["a", "b", "b_absent", "c"]
    assert "a_absent" not in out


def test_none_is_not_absent():
    out = _wire({"unset_setting": None})
    assert out == {"unset_setting": None}


def test_absent_in_dataclass_field():
    @dataclasses.dataclass
    class R:
        value: object
        other: int = 1

    out = _wire(R(Absent.UNDEFINED))
    assert out == {"value": None, "value_absent": "undefined", "other": 1}


def test_absent_nested():
    out = _wire({"outer": [{"x": Absent.NOT_RUN}]})
    assert out["outer"][0] == {"x": None, "x_absent": "not_run"}


def test_absent_without_a_name_is_refused():
    with pytest.raises(TypeError):
        to_jsonable(Absent.NOT_RUN)
    with pytest.raises(TypeError):
        to_jsonable([1, Absent.UNDEFINED])


def test_absent_sibling_collision_refused():
    with pytest.raises(ValueError):
        to_jsonable({"a": Absent.NOT_RUN, "a_absent": "x"})


# ---- columnar form --------------------------------------------------------


def test_absent_column_values_and_status():
    vals, status = absent_column([1.0, Absent.NOT_RUN, 3.0, Absent.UNDEFINED])
    assert status.dtype == np.uint8
    assert status.tolist() == [0, 1, 0, 2]
    assert vals.dtype == np.float64
    assert np.isnan(vals[1]) and np.isnan(vals[3])
    assert vals[0] == 1.0 and vals[2] == 3.0


def test_absent_column_custom_fill_and_dtype():
    vals, status = absent_column([4, Absent.NOT_RUN], dtype=np.int64, fill=-1)
    assert vals.tolist() == [4, -1] and vals.dtype == np.int64
    assert status.tolist() == [0, 1]


def test_with_status_columns_always_emits_status():
    cols = with_status_columns(
        {"chi2": [1.0, 2.0], "id": [1, 2]}, absent_capable=["chi2"]
    )
    assert set(cols) == {"chi2", "chi2__status", "id"}
    assert cols["chi2__status"].tolist() == [0, 0]
    assert cols["chi2__status"].dtype == np.uint8


def test_with_status_columns_undeclared_absent_refused():
    with pytest.raises(TypeError):
        with_status_columns({"a": [1.0, Absent.NOT_RUN]}, absent_capable=[])


# ---- non-finite floats and scalars ----------------------------------------


def test_nonfinite_floats_are_valid_json():
    out = _wire(
        {
            "a": float("nan"),
            "b": float("inf"),
            "c": float("-inf"),
            "d": np.float64("nan"),
            "e": np.float32("inf"),
            "l": [float("nan"), 1.0],
        }
    )
    assert out["a"] == "nan" and out["b"] == "inf" and out["c"] == "-inf"
    assert out["d"] == "nan" and out["e"] == "inf"
    assert out["l"] == ["nan", 1.0]


def test_nonfinite_in_inline_array_is_valid_json():
    out = _wire({"x": np.array([1.0, np.nan, np.inf, -np.inf])})
    assert out["x"] == [1.0, "nan", "inf", "-inf"]


def test_numpy_scalars_become_python():
    out = to_jsonable(
        {
            "i": np.int32(5),
            "f": np.float32(0.5),
            "b": np.bool_(True),
            "z": np.array(7.0),
        }
    )
    assert out == {"i": 5, "f": 0.5, "b": True, "z": 7.0}
    assert type(out["i"]) is int
    assert type(out["f"]) is float
    assert type(out["b"]) is bool


def test_bool_stays_bool_not_int():
    out = to_jsonable({"t": True, "one": 1})
    assert out["t"] is True and type(out["one"]) is int


def test_complex_wire_form():
    assert _wire({"c": 1 + 2j}) == {"c": {"real": 1.0, "imag": 2.0}}


def test_misc_types():
    class Color(enum.Enum):
        RED = "red"

    out = _wire(
        {
            "e": Color.RED,
            "p": Path("a/b"),
            "t": (1, 2),
            "s": {3, 1, 2},
            "d": dt.date(2026, 1, 2),
            "n": None,
        }
    )
    assert out == {
        "e": "red",
        "p": "a/b",
        "t": [1, 2],
        "s": [1, 2, 3],
        "d": "2026-01-02",
        "n": None,
    }


def test_unknown_type_is_a_typeerror_not_str():
    with pytest.raises(TypeError):
        to_jsonable({"x": object()})


def test_bad_mapping_key_refused():
    with pytest.raises(TypeError):
        to_jsonable({(1, 2): 3})


# ---- dataclasses and schema stamp -----------------------------------------


def test_dataclass_becomes_dict_of_fields():
    @dataclasses.dataclass
    class P:
        a: int
        b: float

    assert _wire(P(1, 2.5)) == {"a": 1, "b": 2.5}


def test_schema_argument_stamps_first_key():
    out = to_jsonable({"x": 1}, schema="ftmw/thing@1")
    assert list(out)[0] == "schema" and out["schema"] == "ftmw/thing@1"


def test_schema_stamp_idempotent_and_conflict_refused():
    out = to_jsonable({"schema": "ftmw/thing@1", "x": 1}, schema="ftmw/thing@1")
    assert out == {"schema": "ftmw/thing@1", "x": 1}
    with pytest.raises(ValueError):
        to_jsonable({"schema": "ftmw/other@1"}, schema="ftmw/thing@1")


def test_schema_must_be_well_formed_and_target_an_object():
    with pytest.raises(ValueError):
        to_jsonable({"x": 1}, schema="thing")
    with pytest.raises(TypeError):
        to_jsonable([1], schema="ftmw/thing@1")


def test_dataclass_schema_attribute_stamps_wherever_it_appears():
    @dataclasses.dataclass
    class Stamped:
        __ftmw_schema__ = "ftmw/stamped@2"
        v: int

    out = to_jsonable({"inner": [Stamped(1)]})
    assert out["inner"][0] == {"schema": "ftmw/stamped@2", "v": 1}


# ---- arrays ---------------------------------------------------------------


def test_arrays_inline_by_default():
    assert to_jsonable({"a": np.arange(3)}) == {"a": [0, 1, 2]}


def test_array_sink_replaces_arrays_and_zero_d_stays_inline():
    seen = []

    def sink(path, arr):
        seen.append(path)
        return "f.npy"

    out = to_jsonable({"a": np.arange(3), "s": np.array(2.0)}, arrays=sink)
    assert out == {"a": "f.npy", "s": 2.0}
    assert seen == [("a",)]


def test_array_collector_names_and_dedupes():
    c = ArrayCollector()
    out = to_jsonable(
        {"x": np.zeros(2), "y": {"z": np.ones(3)}, "l": [np.ones(1)]}, arrays=c
    )
    assert out == {"x": "x.npy", "y": {"z": "y.z.npy"}, "l": ["l.0.npy"]}
    assert set(c.arrays) == {"x.npy", "y.z.npy", "l.0.npy"}
    c2 = ArrayCollector(prefix="p_")
    assert c2(("k",), np.zeros(1)) == "p_k.npy"
    assert c2(("k",), np.zeros(1)) == "p_k-2.npy"
