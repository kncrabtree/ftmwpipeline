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

from ftmwpipeline._internal.tuning.registry import UnknownKnobError
from ftmwpipeline.contract import (
    CONTRACT_VERSION,
    ERROR_SCHEMA,
    STAGE_KEYS,
    STATUS_NOT_RUN,
    STATUS_PRESENT,
    STATUS_UNDEFINED,
    Absent,
    AlgorithmFailedError,
    AnalysisEpochMismatchError,
    BadSettingError,
    CallbackFailedError,
    CurationConflictError,
    IncompleteProvenanceError,
    NotFoundError,
    NotFoundValueError,
    OperationCancelledError,
    PipelineCompatibilityError,
    PipelineCorruptionError,
    PipelineExistsError,
    PipelineFileError,
    PipelineFileNotFoundError,
    Stage,
    StageDependencyError,
    WriteConflictError,
    key_for_stage,
    stage_for_key,
)
from ftmwpipeline.core.peak_shape import PeakShape
from ftmwpipeline.file_manager import PipelineStageTracker
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


def test_contract_version_is_fourteen():
    # Wave 0 published 1; Wave 1's accessors raised it to 2; the final-products
    # fit fields (Wave 2) to 3; the window_model / spectrum_model accessors to 4;
    # the analysis_fingerprint accessor to 5; CurationAction (curation as data)
    # to 6; Wave 7 to 7; Wave 8 (full capabilities) to 8; Wave 5.1 (events and
    # cancellation) to 9; Wave 5.1b (atomic writes, write_conflict) to 10;
    # Wave 5.2 (Stage 5 partial fits, restart_reason) to 11; the cleanup wave
    # (curation_conflict, ComplexFT.invalidated, degenerate statistics) to 12;
    # windows after a structural merge (merged_from, fit_plan_unavailable) to 13;
    # review attention (AttentionReason, attention_kind with
    # empty_window_residual) to 14.
    assert CONTRACT_VERSION == 15


# ---- stage vocabulary -----------------------------------------------------


def test_stage_values_are_the_cli_object_names():
    assert [s.value for s in Stage] == [
        "data",
        "ft",
        "noise",
        "tau",
        "tau_g",
        "timebase",
        "peaks",
        "windows",
        "fit",
        "review",
    ]
    assert Stage.FT == "ft"  # a str enum: compares with its value


def test_every_internal_stage_key_maps_to_a_stage():
    keys = set(PipelineStageTracker.STAGE_DEPENDENCIES)
    assert set(STAGE_KEYS.values()) == keys
    assert set(STAGE_KEYS) == set(Stage)
    for key in keys:
        assert isinstance(stage_for_key(key), Stage)


def test_stage_mapping_round_trips():
    for stage in Stage:
        assert stage_for_key(key_for_stage(stage)) is stage
        assert key_for_stage(stage.value) == STAGE_KEYS[stage]
    for key in PipelineStageTracker.STAGE_DEPENDENCIES:
        assert key_for_stage(stage_for_key(key)) == key


def test_unmapped_stage_names_raise():
    with pytest.raises(ValueError):
        stage_for_key("stage99_nonexistent")
    with pytest.raises(ValueError):
        stage_for_key("ft")  # a canonical name is not an internal key
    with pytest.raises(ValueError):
        key_for_stage("stage1_complex_ft")  # an internal key is not a stage


def test_stage_mapping_is_read_only():
    with pytest.raises(TypeError):
        STAGE_KEYS[Stage.FT] = "x"  # type: ignore[index]


def test_stage_dependency_error_to_dict_uses_canonical_names():
    e = StageDependencyError(
        "stage5_fitting",
        ["stage1_complex_ft", "stage3_peaks", "stage2b_tau_G_calibration"],
        Path("x.ftmw"),
        command="ftmwpipeline peaks run",
    )
    d = json.loads(json.dumps(e.to_dict(), allow_nan=False))
    assert d["missing_dependencies"] == ["ft", "peaks", "tau_g"]
    # The Python attribute keeps the internal keys.
    assert e.missing_dependencies == [
        "stage1_complex_ft",
        "stage3_peaks",
        "stage2b_tau_G_calibration",
    ]


def test_stage_dependency_error_unmapped_key_fails_loudly_in_to_dict():
    e = StageDependencyError("s", ["stage99_nonexistent"], Path("x.ftmw"))
    with pytest.raises(ValueError):
        e.to_dict()


def test_tracker_raised_dependency_error_publishes_canonical_names():
    tracker = PipelineStageTracker(["stage0_fid_data"])
    with pytest.raises(StageDependencyError) as exc_info:
        tracker.validate_dependencies("stage2_noise_result")
    assert exc_info.value.to_dict()["missing_dependencies"] == ["ft"]


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
            StageDependencyError(
                "fit", ["stage3_peaks"], p, command="ftmwpipeline peaks"
            ),
            "stage_not_run",
            {"missing_dependencies": ["peaks"], "command": "ftmwpipeline peaks"},
            (ValueError,),
        ),
        (PipelineCorruptionError(p, "bad"), "file_corrupt", {}, (RuntimeError,)),
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
            PipelineFileNotFoundError(p),
            "not_found",
            {"kind": "file", "ids": ["x.ftmw"]},
            (NotFoundError, FileNotFoundError, KeyError),
        ),
        (
            IncompleteProvenanceError(["stage2b.shape"]),
            "incomplete_provenance",
            {"missing": ["stage2b.shape"]},
            (ValueError,),
        ),
        (
            NotFoundValueError("peak", [7, 8]),
            "not_found",
            {"kind": "peak", "ids": [7, 8]},
            (KeyError, ValueError),
        ),
        (
            BadSettingError("stage5.tau.tau0_us", "float > 0", -1.0),
            "bad_setting",
            {"path": "stage5.tau.tau0_us", "expected": "float > 0", "value": -1.0},
            (ValueError,),
        ),
        (
            UnknownKnobError("stage9.bogus", "a registered knob path", "stage9.bogus"),
            "bad_setting",
            {
                "path": "stage9.bogus",
                "expected": "a registered knob path",
                "value": "stage9.bogus",
            },
            (ValueError, KeyError),
        ),
        (
            AlgorithmFailedError("fit", "no window could be fitted"),
            "algorithm_failed",
            {"stage": "fit"},
            (RuntimeError,),
        ),
        (
            OperationCancelledError("fit", ["data", "ft"], []),
            "cancelled",
            {"stage": "fit", "completed_stages": ["data", "ft"], "completed_windows": []},
            (),
        ),
        (
            CallbackFailedError("ftmw/window_progress@1"),
            "callback_failed",
            {"event_schema": "ftmw/window_progress@1", "completed_windows": []},
            (),
        ),
        (
            WriteConflictError("/data/exp.ftmw"),
            "write_conflict",
            {"path": "/data/exp.ftmw"},
            (),
        ),
        (
            CurationConflictError("orphans_created_window", [4, 7]),
            "curation_conflict",
            {"reason": "orphans_created_window", "ids": [4, 7]},
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


def test_epoch_mismatch_none_epoch_is_null_with_absent_sibling():
    e = AnalysisEpochMismatchError("f", _Env(None), _Env(3))
    d = json.loads(json.dumps(e.to_dict(), allow_nan=False))
    assert d["file_epoch"] is None and d["file_epoch_absent"] == "not_run"
    assert d["current_epoch"] == 3 and "current_epoch_absent" not in d
    # The Python attributes keep None.
    assert e.file_epoch is None and e.current_epoch == 3


def test_epoch_mismatch_both_none():
    d = AnalysisEpochMismatchError("f", _Env(None), _Env(None)).to_dict()
    assert d["file_epoch_absent"] == d["current_epoch_absent"] == "not_run"
    assert d["file_epoch"] is None and d["current_epoch"] is None


def test_epoch_mismatch_present_epochs_have_no_sibling():
    d = AnalysisEpochMismatchError("f", _Env(2), _Env(3)).to_dict()
    assert not any(k.endswith("_absent") for k in d)


def test_error_codes_unique_per_class():
    # PipelineFileNotFoundError is a NotFoundError, so it shares its code.
    by_class = {type(e): c for e, c, _, _ in _errors()}
    owners = {}
    for cls, code in by_class.items():
        owners.setdefault(code, []).append(cls)
    for code, classes in owners.items():
        if code == "not_found":
            assert set(classes) == {
                NotFoundError,
                NotFoundValueError,
                PipelineFileNotFoundError,
            }
        elif code == "bad_setting":
            # UnknownKnobError is a BadSettingError that is also a KeyError.
            assert set(classes) == {BadSettingError, UnknownKnobError}
        else:
            assert len(classes) == 1, (code, classes)


def _all_subclasses(cls):
    seen = []
    stack = list(cls.__subclasses__())
    while stack:
        c = stack.pop()
        if c not in seen:
            seen.append(c)
            stack.extend(c.__subclasses__())
    return seen


def test_errors_list_covers_every_class():
    covered = {type(e) for e, _, _, _ in _errors()}
    assert set(_all_subclasses(PipelineFileError)) <= covered


def _state(err):
    return {k: v for k, v in vars(err).items() if not isinstance(v, _Env)}


@pytest.mark.parametrize("err,code,fields,bases", _errors(), ids=lambda v: str(type(v)))
def test_every_error_pickles_with_attributes_preserved(err, code, fields, bases):
    clone = pickle.loads(pickle.dumps(err))
    assert type(clone) is type(err)
    assert clone.code == code
    assert clone.args == err.args
    assert str(clone) == str(err)
    assert _state(clone) == _state(err)
    assert clone.to_dict() == err.to_dict()


def test_base_error_pickles():
    e = PipelineFileError("plain")
    clone = pickle.loads(pickle.dumps(e))
    assert type(clone) is PipelineFileError and clone.to_dict() == e.to_dict()


# ---- missing and corrupt files --------------------------------------------


def test_missing_file_error_is_notfound_filenotfound_and_keyerror():
    e = PipelineFileNotFoundError("some/missing.ftmw")
    assert isinstance(e, NotFoundError)
    assert isinstance(e, FileNotFoundError)
    assert isinstance(e, KeyError)
    assert e.code == "not_found"
    assert e.kind == "file" and e.ids == ["some/missing.ftmw"]
    assert e.filepath == Path("some/missing.ftmw")
    d = e.to_dict()
    assert d["code"] == "not_found" and d["kind"] == "file"
    assert d["ids"] == ["some/missing.ftmw"]


def test_missing_file_error_keyword_message():
    e = PipelineFileNotFoundError("a.ftmw", message="custom words")
    assert str(e) == "custom words" and e.to_dict()["message"] == "custom words"


def test_corruption_error_is_a_runtime_error():
    e = PipelineCorruptionError("f.ftmw", "bad", message="Failed to open f.ftmw")
    assert isinstance(e, RuntimeError)
    # The read path let h5py's OSError escape before it was typed.
    assert isinstance(e, OSError)
    assert str(e) == "Failed to open f.ftmw"
    assert e.code == "file_corrupt"


def test_pipeline_open_missing_path_raises_missing_file_error(tmp_path):
    from ftmwpipeline import Pipeline

    missing = tmp_path / "nope.ftmw"
    with pytest.raises(PipelineFileNotFoundError) as exc_info:
        Pipeline.open(missing)
    assert isinstance(exc_info.value, FileNotFoundError)
    assert exc_info.value.ids == [str(missing)]


def test_pipeline_open_junk_file_raises_corruption_error(tmp_path):
    from ftmwpipeline import Pipeline

    junk = tmp_path / "junk.ftmw"
    junk.write_bytes(b"this is not an HDF5 file at all" * 20)
    with pytest.raises(PipelineCorruptionError) as exc_info:
        Pipeline.open(junk)
    assert isinstance(exc_info.value, RuntimeError)
    assert exc_info.value.to_dict()["code"] == "file_corrupt"
    assert exc_info.value.__cause__ is not None


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


# ---- reserved "_absent" keys ----------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        {"a": Absent.NOT_RUN, "a_absent": "x"},  # collides with the sibling
        {"a": 1, "a_absent": "not_run"},  # forged sibling of a present field
        {"a_absent": "not_run"},  # sibling with no base field
        {"_absent": "x"},
        {"a": float("nan"), "a_absent": "x"},
    ],
    ids=["collision", "forged", "orphan", "bare_suffix", "nan_collision"],
)
def test_reserved_absent_key_refused_in_mappings(source):
    with pytest.raises(ValueError, match="reserved"):
        to_jsonable(source)


def test_reserved_absent_key_refused_in_dataclass_and_nested():
    @dataclasses.dataclass
    class R:
        x: int
        x_absent: str = "undefined"

    with pytest.raises(ValueError, match="reserved"):
        to_jsonable(R(1))
    with pytest.raises(ValueError, match="reserved"):
        to_jsonable({"outer": [{"y_absent": "not_run"}]})


def test_keys_merely_containing_absent_are_fine():
    out = to_jsonable({"absent": 1, "absent_count": 2, "x_absent_rate": 3})
    assert out == {"absent": 1, "absent_count": 2, "x_absent_rate": 3}


def test_generated_siblings_are_never_refused():
    out = _wire({"a": Absent.NOT_RUN, "b": float("nan")})
    assert out == {
        "a": None,
        "a_absent": "not_run",
        "b": None,
        "b_absent": "undefined",
    }


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


def test_nonfinite_named_field_is_null_with_undefined_sibling():
    out = _wire(
        {
            "a": float("nan"),
            "b": float("inf"),
            "c": float("-inf"),
            "d": np.float64("nan"),
            "e": np.float32("inf"),
            "f": np.array(np.nan),
            "ok": 1.0,
        }
    )
    for name in "abcdef":
        assert out[name] is None
        assert out[name + "_absent"] == "undefined"
    assert out["ok"] == 1.0 and "ok_absent" not in out


def test_nonfinite_dataclass_field_and_nested_field():
    @dataclasses.dataclass
    class R:
        chi2: float
        n: int = 3

    assert _wire(R(float("nan"))) == {"chi2": None, "chi2_absent": "undefined", "n": 3}
    out = _wire({"rows": [{"x": float("inf")}]})
    assert out["rows"] == [{"x": None, "x_absent": "undefined"}]


def test_nonfinite_in_inline_list_is_null():
    out = _wire({"l": [float("nan"), 1.0, float("inf")], "t": (np.float64("nan"), 2)})
    assert out == {"l": [None, 1.0, None], "t": [None, 2]}


def test_nonfinite_in_inline_array_is_null():
    out = _wire(
        {"x": np.array([1.0, np.nan, np.inf, -np.inf]), "m": np.full((2, 2), np.nan)}
    )
    assert out["x"] == [1.0, None, None, None]
    assert out["m"] == [[None, None], [None, None]]
    assert "x_absent" not in out


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf"), np.float64("nan")]
)
def test_nonfinite_at_top_level_is_a_typeerror(value):
    with pytest.raises(TypeError):
        to_jsonable(value)


def test_nonfinite_array_keeps_nan_through_a_sink():
    c = ArrayCollector()
    out = to_jsonable({"x": np.array([1.0, np.nan])}, arrays=c)
    assert out == {"x": "x.npy"}
    assert np.isnan(c.arrays["x.npy"][1])


def test_complex_with_nonfinite_part():
    out = _wire({"c": complex(float("nan"), 2.0)})
    assert out == {"c": {"real": None, "real_absent": "undefined", "imag": 2.0}}


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


class _Color(enum.Enum):
    RED = "red"
    ONE = 1


class _Level(enum.IntEnum):
    LOW = 1
    HIGH = 2


class _Tag(str, enum.Enum):
    A = "alpha"


def test_str_enum_is_written_as_its_value():
    out = _wire({"shape": PeakShape.LORENTZIAN, "t": _Tag.A})
    assert out == {"shape": PeakShape.LORENTZIAN.value, "t": "alpha"}
    assert type(out["shape"]) is str and "PeakShape" not in json.dumps(out)


def test_int_enum_is_written_as_its_int_value():
    out = _wire({"lvl": _Level.HIGH})
    assert out == {"lvl": 2} and type(out["lvl"]) is int


def test_plain_enum_is_written_as_its_value():
    out = _wire({"c": _Color.RED, "n": _Color.ONE})
    assert out == {"c": "red", "n": 1}


def test_enums_at_top_level_and_in_lists():
    assert to_jsonable(PeakShape.LORENTZIAN) == PeakShape.LORENTZIAN.value
    assert _wire([_Level.LOW, _Color.RED, _Tag.A]) == [1, "red", "alpha"]


def test_enum_mapping_keys_are_written_as_their_value():
    out = _wire(
        {
            PeakShape.LORENTZIAN: 1,
            _Level.HIGH: 2,
            _Color.RED: 3,
            _Tag.A: 4,
        }
    )
    assert out == {PeakShape.LORENTZIAN.value: 1, "2": 2, "red": 3, "alpha": 4}


def test_enum_key_colliding_with_its_value_string_is_refused():
    # A plain Enum and its value string are distinct dict keys that serialize
    # to the same JSON key. (A str-mixin enum hashes equal to its value, so a
    # dict cannot hold both in the first place.)
    assert len({_Color.RED: 1, "red": 2}) == 2
    with pytest.raises(ValueError):
        to_jsonable({_Color.RED: 1, "red": 2})


def test_absent_is_not_a_mapping_key():
    with pytest.raises(TypeError):
        to_jsonable({Absent.NOT_RUN: 1})


def test_enum_valued_field_that_is_absent_still_gets_a_sibling():
    out = _wire({"shape": Absent.NOT_RUN, "other": PeakShape.LORENTZIAN})
    assert out["shape"] is None and out["shape_absent"] == "not_run"


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


@pytest.mark.parametrize(
    "exc",
    [
        PermissionError(13, "Permission denied"),
        OSError("Unable to synchronously open file (unable to lock file, errno = 11)"),
    ],
)
def test_transient_open_errors_are_not_reported_as_corruption(
    exc, monkeypatch, tmp_path
):
    import h5py

    from ftmwpipeline import Pipeline, api

    path = tmp_path / "busy.ftmw"
    path.write_bytes(b"placeholder")

    def refuse(*a, **k):
        raise exc

    monkeypatch.setattr(h5py, "File", refuse)
    for call in (lambda: Pipeline.open(path), lambda: api.read_metadata(path)):
        with pytest.raises(OSError) as info:
            call()
        assert not isinstance(info.value, PipelineCorruptionError)
