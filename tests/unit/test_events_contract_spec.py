"""The promises of CONTRACT_STRATEGY §Events and cancellation that need no data.

Written from the spec, not from the implementation: the event types and their
wire form, the ``CancelToken`` protocol, the two new error codes, the manifest
and capabilities entries, the CLI exit-code mapping, and how ``run_pipeline``
reports a cancel, a failing callback and a failure.
"""

from __future__ import annotations

import dataclasses
import json
import pickle
import threading
from pathlib import Path

import pytest

import ftmwpipeline
import ftmwpipeline.api as ftmw
from ftmwpipeline import (
    CallbackFailedError,
    CancelToken,
    Invalidated,
    OperationCancelledError,
    PipelineWarning,
    ScanProgress,
    StageFinished,
    StageStarted,
    WindowProgress,
    to_jsonable,
)
from ftmwpipeline._internal.events import operation_events
from ftmwpipeline.cli.contract_commands import EXIT_CODES, exit_code_for
from ftmwpipeline.contract import (
    MANIFEST,
    WARNING_FIELDS,
    WINDOW_PHASES,
    Absent,
    Stage,
    capabilities,
)
from ftmwpipeline.file_manager import PipelineFileError, WriteConflictError
from tests._events_support import Recorder, Token

pytestmark = [pytest.mark.unit]

#: The spec's event table: type, schema, fields beyond (schema, operation, stage).
EVENT_TABLE = [
    (StageStarted, "ftmw/stage_started@1", ()),
    (StageFinished, "ftmw/stage_finished@1", ("elapsed_s", "summary")),
    (
        WindowProgress,
        "ftmw/window_progress@1",
        (
            "phase",
            "round",
            "index",
            "total",
            "window_id",
            "n_peaks",
            "chi2r",
            "elapsed_s",
            "dropped",
        ),
    ),
    (ScanProgress, "ftmw/scan_progress@1", ("knob", "value", "index", "total")),
    (Invalidated, "ftmw/invalidated@1", ("stages",)),
    (PipelineWarning, "ftmw/warning@1", ("code", "message")),
]

COMMON = ("schema", "operation", "stage")


def _example(cls: type) -> object:
    if cls is StageStarted:
        return StageStarted("noise run", "noise")
    if cls is StageFinished:
        return StageFinished("noise run", "noise", 1.5, {"n": 3, "nested": {"a": 1}})
    if cls is WindowProgress:
        return WindowProgress(
            "fit run", "fit", "initial", 0, 1, 4, 7, 2, 1.25, 0.5, False
        )
    if cls is ScanProgress:
        return ScanProgress("scan run", None, "stage2.window_mhz", 40.0, 1, 2)
    if cls is Invalidated:
        return Invalidated("ft run", "ft", ("noise", "review"))
    return PipelineWarning(
        "fit run",
        "fit",
        "slow_window",
        "window 3 was slow",
        {"window_id": 3, "elapsed_s": 70.0, "threshold_s": 60.0},
    )


# ---- event types ------------------------------------------------------------


@pytest.mark.parametrize("cls,schema,extra", EVENT_TABLE)
def test_event_carries_its_schema_and_spec_fields(cls, schema, extra):
    ev = _example(cls)
    assert ev.schema == schema
    names = [f.name for f in dataclasses.fields(cls)]
    for name in COMMON + extra:
        assert name in names, f"{cls.__name__} lacks {name}"
    wire = to_jsonable(ev)
    assert wire["schema"] == schema
    assert list(wire)[:3] == list(COMMON)
    for name in extra:
        assert name in wire


@pytest.mark.parametrize("cls,schema,extra", EVENT_TABLE)
def test_event_is_a_frozen_dataclass(cls, schema, extra):
    ev = _example(cls)
    assert dataclasses.is_dataclass(ev)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.operation = "x"  # type: ignore[misc]


@pytest.mark.parametrize("cls,schema,extra", EVENT_TABLE)
def test_event_wire_form_is_strict_json(cls, schema, extra):
    # to_jsonable output survives a strict dump (no NaN, no non-JSON types).
    json.dumps(to_jsonable(_example(cls)), allow_nan=False)


def test_event_stage_is_canonical_or_null_on_the_wire():
    assert to_jsonable(StageStarted("noise run", "noise"))["stage"] == "noise"
    assert to_jsonable(StageStarted("start run", None))["stage"] is None
    assert to_jsonable(StageStarted("run", Stage.PEAKS))["stage"] == "peaks"


def test_event_schemas_are_the_ones_the_spec_names():
    assert {schema for _, schema, _ in EVENT_TABLE} == {
        "ftmw/stage_started@1",
        "ftmw/stage_finished@1",
        "ftmw/window_progress@1",
        "ftmw/scan_progress@1",
        "ftmw/invalidated@1",
        "ftmw/warning@1",
    }


def test_window_progress_wire_form():
    wire = to_jsonable(_example(WindowProgress))
    assert wire == {
        "schema": "ftmw/window_progress@1",
        "operation": "fit run",
        "stage": "fit",
        "phase": "initial",
        "round": 0,
        "index": 1,
        "total": 4,
        "window_id": 7,
        "n_peaks": 2,
        "chi2r": 1.25,
        "elapsed_s": 0.5,
        "dropped": False,
    }


@pytest.mark.parametrize("absent", [Absent.NOT_RUN, Absent.UNDEFINED])
def test_dropped_window_reports_absent_n_peaks_and_chi2r(absent):
    ev = WindowProgress(
        "fit run", "fit", "replan", 1, 2, 3, 9, absent, absent, 0.1, True
    )
    assert ev.dropped is True
    assert isinstance(ev.n_peaks, Absent) and isinstance(ev.chi2r, Absent)
    wire = to_jsonable(ev)
    # The missing-value rule: null plus a sibling marker naming why.
    assert wire["n_peaks"] is None and wire["n_peaks_absent"] == absent.value
    assert wire["chi2r"] is None and wire["chi2r_absent"] == absent.value
    assert wire["dropped"] is True and wire["phase"] == "replan"
    assert wire["round"] == 1
    json.dumps(wire, allow_nan=False)


def test_window_progress_phase_is_one_of_the_four_passes():
    assert WINDOW_PHASES == ("initial", "replan", "fallback", "cascade")
    for phase in WINDOW_PHASES:
        WindowProgress("fit run", "fit", phase, 0, 1, 1, 0, 1, 1.0, 0.1, False)
    with pytest.raises(ValueError):
        WindowProgress("fit run", "fit", "later", 0, 1, 1, 0, 1, 1.0, 0.1, False)


def test_stage_finished_summary_is_carried_through():
    wire = to_jsonable(_example(StageFinished))
    assert wire["summary"] == {"n": 3, "nested": {"a": 1}}
    assert wire["elapsed_s"] == 1.5


def test_invalidated_stages_are_canonical_names():
    ev = _example(Invalidated)
    assert [s.value for s in ev.stages] == ["noise", "review"]
    assert to_jsonable(ev)["stages"] == ["noise", "review"]


def test_scan_progress_knob_is_a_registry_path():
    wire = to_jsonable(_example(ScanProgress))
    assert wire["knob"] == "stage2.window_mhz"
    assert (wire["index"], wire["total"], wire["value"]) == (1, 2, 40.0)


# ---- StageFinished.summary keys -------------------------------------------


def test_summary_keys_are_declared_per_operation_in_capabilities():
    declared = capabilities()["summary_keys"]
    assert list(declared) == list(MANIFEST.summary_keys)
    for verb, parts in declared.items():
        assert set(parts) == {"required", "conditional"}, verb
        keys = parts["required"] + parts["conditional"]
        assert len(set(keys)) == len(keys), verb
    assert {"fit run", "noise run", "review accept", "scan run"} <= set(declared)


def test_summary_keys_match_the_builders_key_constants():
    from ftmwpipeline._internal.stage0_impl import DATA_IMPORT_SUMMARY_KEYS
    from ftmwpipeline._internal.stage1_impl import FT_RUN_SUMMARY_KEYS
    from ftmwpipeline._internal.stage2_impl import NOISE_RUN_SUMMARY_KEYS
    from ftmwpipeline._internal.stage5_impl import FIT_RUN_SUMMARY_KEYS

    for verb, keys in {
        "data import": DATA_IMPORT_SUMMARY_KEYS,
        "ft run": FT_RUN_SUMMARY_KEYS,
        "noise run": NOISE_RUN_SUMMARY_KEYS,
        "fit run": FIT_RUN_SUMMARY_KEYS,
    }.items():
        parts = MANIFEST.summary_keys[verb]
        assert parts["required"] + parts["conditional"] == tuple(keys), verb


# ---- warnings -------------------------------------------------------------


def test_warning_vocabulary_is_the_specs_six_codes():
    assert set(MANIFEST.vocabularies["warning_code"]) == {
        "slow_window",
        "epoch_acknowledged",
        "environment_drift",
        "frame_mismatch",
        "walk_fallback",
        "timebase_skipped",
    }
    assert capabilities()["vocabularies"]["warning_code"] == list(
        MANIFEST.vocabularies["warning_code"]
    )


_WARNING_FIELDS_SPEC = {
    "slow_window": ("window_id", "elapsed_s", "threshold_s"),
    "epoch_acknowledged": ("file_epoch", "current_epoch"),
    "environment_drift": ("fields",),
    "frame_mismatch": ("actions",),
    "walk_fallback": ("reason", "n_windows"),
    "timebase_skipped": (),
}
_WARNING_VALUES = {
    "window_id": 4,
    "elapsed_s": 61.0,
    "threshold_s": 60.0,
    "file_epoch": 3,
    "current_epoch": 4,
    "fields": ["numpy"],
    "actions": [0, 2],
    "reason": "accepted thaw",
    "n_windows": 12,
}


@pytest.mark.parametrize("code,fields", sorted(_WARNING_FIELDS_SPEC.items()))
def test_warning_carries_its_code_specific_fields_on_the_wire(code, fields):
    assert tuple(WARNING_FIELDS[code]) == fields
    ev = PipelineWarning(
        "fit run", "fit", code, "a message", {f: _WARNING_VALUES[f] for f in fields}
    )
    wire = to_jsonable(ev)
    assert wire["schema"] == "ftmw/warning@1"
    assert wire["code"] == code and wire["message"] == "a message"
    for f in fields:
        assert wire[f] == _WARNING_VALUES[f]
    assert set(wire) == {"schema", "operation", "stage", "code", "message", *fields}


def test_warning_rejects_unknown_codes_and_wrong_fields():
    with pytest.raises(ValueError):
        PipelineWarning("fit run", "fit", "not_a_code", "m", {})
    with pytest.raises(ValueError):
        PipelineWarning("fit run", "fit", "walk_fallback", "m", {"reason": "x"})


# ---- CancelToken ------------------------------------------------------------


def test_threading_event_is_a_cancel_token():
    assert isinstance(threading.Event(), CancelToken)


def test_any_object_with_is_set_is_a_cancel_token():
    class Mine:
        def is_set(self) -> bool:
            return False

    assert isinstance(Mine(), CancelToken)
    assert not isinstance(object(), CancelToken)
    assert not isinstance(lambda: False, CancelToken)


def test_cancel_token_is_polled_not_required_to_be_an_event():
    # A token that only offers is_set() stops an operation.
    tok = Token()
    ops = operation_events("fit run", None, tok)
    with ops.stage("fit") as scope:
        scope.check_cancel()  # not set: no-op
        tok.set()
        with pytest.raises(OperationCancelledError):
            scope.check_cancel()


# ---- errors -----------------------------------------------------------------


def test_cancelled_error_code_attributes_and_dict():
    err = OperationCancelledError("fit", ["data", "ft"], [])
    assert err.code == "cancelled"
    assert isinstance(err, PipelineFileError)
    assert (err.stage, err.completed_stages, err.completed_windows) == (
        "fit",
        ["data", "ft"],
        [],
    )
    d = err.to_dict()
    assert d["schema"] == "ftmw/error@1" and d["code"] == "cancelled"
    assert d["message"]
    assert (d["stage"], d["completed_stages"], d["completed_windows"]) == (
        "fit",
        ["data", "ft"],
        [],
    )


def test_cancelled_between_stages_has_a_null_stage_and_wire_form():
    err = OperationCancelledError(None, ["data"], [])
    assert err.stage is None
    d = err.to_dict()
    assert d["stage"] is None and d["completed_stages"] == ["data"]
    json.dumps(d, allow_nan=False)


def test_cancelled_accepts_stage_enums_and_stores_canonical_strings():
    err = OperationCancelledError(Stage.FIT, [Stage.DATA, Stage.FT], [])
    assert err.stage == "fit" and err.completed_stages == ["data", "ft"]
    assert json.dumps(err.to_dict())  # no enum objects left in the dict


def test_callback_failed_error_code_attributes_and_dict():
    err = CallbackFailedError("ftmw/window_progress@1")
    assert err.code == "callback_failed"
    assert isinstance(err, PipelineFileError)
    assert err.event_schema == "ftmw/window_progress@1"
    d = err.to_dict()
    assert d["schema"] == "ftmw/error@1" and d["code"] == "callback_failed"
    assert d["event_schema"] == "ftmw/window_progress@1" and d["message"]
    # [] everywhere except a Stage 5 interruption, which lists the kept windows.
    assert err.completed_windows == [] and d["completed_windows"] == []
    kept = CallbackFailedError("ftmw/window_progress@1", [4, 9])
    assert kept.to_dict()["completed_windows"] == [4, 9]


@pytest.mark.parametrize(
    "err",
    [
        OperationCancelledError("fit", ["data", "ft", "noise"], []),
        OperationCancelledError(None, [], []),
        CallbackFailedError("ftmw/stage_started@1"),
        WriteConflictError("/data/exp.ftmw"),
    ],
)
def test_errors_pickle_round_trip(err):
    # They cross a process boundary when a worker pool raises them.
    back = pickle.loads(pickle.dumps(err))
    assert type(back) is type(err)
    assert back.to_dict() == err.to_dict()
    assert str(back) == str(err)


def test_errors_are_exported_from_the_package_root():
    assert ftmwpipeline.OperationCancelledError is OperationCancelledError
    assert ftmwpipeline.CallbackFailedError is CallbackFailedError
    assert ftmwpipeline.WriteConflictError is WriteConflictError
    for name in (
        "OperationCancelledError",
        "CallbackFailedError",
        "CancelToken",
        "WriteConflictError",
    ):
        assert name in ftmwpipeline.__all__
    for cls in (
        StageStarted,
        StageFinished,
        WindowProgress,
        ScanProgress,
        Invalidated,
        PipelineWarning,
    ):
        assert cls.__name__ in ftmwpipeline.__all__
    assert "Event" in ftmwpipeline.__all__


def test_failing_callback_chains_its_exception_as_cause():
    original = ValueError("listener broke")

    def bad(event):
        raise original

    ops = operation_events("noise run", bad)
    with pytest.raises(CallbackFailedError) as info:
        with ops.stage("noise"):
            pass  # StageStarted is delivered on entry and raises
    assert info.value.__cause__ is original
    assert info.value.event_schema == "ftmw/stage_started@1"


def test_callback_failed_names_the_event_being_delivered():
    def bad_on_finish(event):
        if isinstance(event, StageFinished):
            raise RuntimeError("x")

    ops = operation_events("noise run", bad_on_finish)
    with pytest.raises(CallbackFailedError) as info:
        with ops.stage("noise") as scope:
            scope.finish({})
    assert info.value.event_schema == "ftmw/stage_finished@1"
    assert isinstance(info.value.__cause__, RuntimeError)


# ---- ordering at the scope level ---------------------------------------------


def test_cancelled_or_failed_stage_emits_no_stage_finished():
    for exc in (OperationCancelledError("noise", [], []), RuntimeError("boom")):
        rec = Recorder()
        ops = operation_events("noise run", rec)
        with pytest.raises(type(exc)):
            with ops.stage("noise"):
                raise exc
        assert [type(e) for e in rec.events] == [StageStarted]
        assert ops.completed_stages == []


def test_stage_order_is_started_other_invalidated_finished():
    from ftmwpipeline._internal.events import emit_invalidated

    rec = Recorder()
    ops = operation_events("fit run", rec)
    with ops.stage(Stage.FIT) as scope:
        scope.window_progress(
            phase="initial",
            round=0,
            index=1,
            total=1,
            window_id=0,
            n_peaks=1,
            chi2r=1.0,
            elapsed_s=0.1,
            dropped=False,
            freq_range=(1.0, 2.0),
        )
        emit_invalidated(scope, ["stage6_review"], reason="re-run")
        scope.finish({"n_windows": 1})
    assert [type(e) for e in rec.events] == [
        StageStarted,
        WindowProgress,
        Invalidated,
        StageFinished,
    ]
    assert all(e.operation == "fit run" and e.stage is Stage.FIT for e in rec.events)
    assert ops.completed_stages == ["fit"]


# ---- manifest and capabilities ---------------------------------------------------


def test_contract_version_is_sixteen():
    # 9: events and cancellation (Wave 5.1); 10: write_conflict and atomic
    # writes (Wave 5.1b); 11: Stage 5 partial fits and resume (Wave 5.2);
    # 12: the cleanup wave (curation_conflict, ComplexFT.invalidated,
    # degenerate statistics read UNDEFINED); 13: windows after a structural
    # merge (WindowStatusRow.merged_from, fit_plan_unavailable); 14: review
    # attention (AttentionReason, the attention_kind vocabulary); 15:
    # decision-log action groups, empty-window converged, run_pipeline
    # canonical names, tau_shape; 16: removes logged at the resolved peak;
    # 17: request action indices; 18: calibrated created-window structure.
    assert ftmwpipeline.CONTRACT_VERSION == 18
    assert MANIFEST.contract_version == 18
    assert capabilities()["contract_version"] == 18


def test_event_schemas_are_in_the_manifest_and_capabilities():
    caps = capabilities()
    for _, schema, _ in EVENT_TABLE:
        assert schema in MANIFEST.schemas
        assert schema in caps["schemas"]


def test_error_codes_are_in_the_manifest_and_capabilities():
    caps = capabilities()
    for code in ("cancelled", "callback_failed", "write_conflict", "pipeline_error"):
        assert code in MANIFEST.codes
        assert code in caps["codes"]
    # The manifest and the exception classes agree.
    assert OperationCancelledError.code in caps["codes"]
    assert CallbackFailedError.code in caps["codes"]
    assert WriteConflictError.code in caps["codes"]


@pytest.mark.parametrize("cls,schema,extra", EVENT_TABLE)
def test_event_fields_are_declared_in_the_manifest(cls, schema, extra):
    declared = set(MANIFEST.fields[cls.__name__])
    assert set(COMMON) <= declared
    # Every wire field the spec names is declared (the code-specific fields of
    # PipelineWarning are declared per code, see WARNING_FIELDS).
    assert set(extra) <= declared
    assert capabilities()["fields"][cls.__name__] == list(MANIFEST.fields[cls.__name__])


def test_declared_warning_fields_are_its_wire_keys():
    # PipelineWarning is declared as its wire form: the common fields, plus
    # each code's own under ``PipelineWarning.<code>`` (``details`` is flattened).
    fields = capabilities()["fields"]
    w = _example(PipelineWarning)
    assert list(to_jsonable(w)) == (
        fields["PipelineWarning"] + fields[f"PipelineWarning.{w.code}"]
    )
    for code, names in WARNING_FIELDS.items():
        assert fields[f"PipelineWarning.{code}"] == list(names)


# ---- the CLI's exit codes ----------------------------------------------------------


def test_cancelled_exits_130_and_callback_failed_exits_1():
    assert EXIT_CODES["cancelled"] == 130
    assert exit_code_for(OperationCancelledError("fit", [], [])) == 130
    assert exit_code_for(CallbackFailedError("ftmw/warning@1")) == 1


def test_write_conflict_exits_1():
    # Not in the table: every code outside it exits 1 (§Errors).
    assert "write_conflict" not in EXIT_CODES
    assert exit_code_for(WriteConflictError("/x/y.ftmw")) == 1


# ---- run_pipeline ------------------------------------------------------------------------


def _run(tmp_path: Path, **kwargs):
    return ftmw.run_pipeline(
        tmp_path / "no_such_source",
        tmp_path / "out.ftmw",
        trim=(26500.0, 40000.0),
        progress=False,
        **kwargs,
    )


def test_run_pipeline_failure_error_is_an_error_dict(tmp_path):
    res = _run(tmp_path)
    assert res["status"] == "error"
    err = res["error"]
    assert isinstance(err, dict), "the error is no longer a string"
    assert err["schema"] == "ftmw/error@1"
    assert isinstance(err["code"], str) and err["message"]
    json.dumps(err, allow_nan=False)
    # failed_stage is a canonical stage name.
    assert res["failed_stage"] in {s.value for s in Stage}
    assert res["failed_stage"] == "data"


def test_run_pipeline_cancel_is_raised_not_folded_into_the_result(tmp_path):
    tok = Token()
    tok.set()
    with pytest.raises(OperationCancelledError) as info:
        _run(tmp_path, cancel=tok)
    # Nothing had completed, and the cancel fell before the first stage.
    assert info.value.completed_stages == []
    assert info.value.completed_windows == []
    assert info.value.stage is None
    assert not (tmp_path / "out.ftmw").exists()


def test_run_pipeline_callback_failure_is_raised_not_folded_into_the_result(tmp_path):
    def bad(event):
        raise RuntimeError("listener broke")

    with pytest.raises(CallbackFailedError) as info:
        _run(tmp_path, events=bad)
    assert isinstance(info.value.__cause__, RuntimeError)
    assert info.value.event_schema == "ftmw/stage_started@1"
