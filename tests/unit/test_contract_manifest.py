"""The machine-contract manifest is complete, consistent, and only grows.

Spec: ``dev-docs/CONTRACT_STRATEGY.md`` §Versioning and stability ("a test can
assert that every declared element exists on all three interfaces and that
nothing declared silently disappears").
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
from types import SimpleNamespace

import pytest

import ftmwpipeline
import ftmwpipeline.api as api
from ftmwpipeline.cli.main import create_parser
from ftmwpipeline.contract import (
    MANIFEST,
    SCHEMA_NAME_RE,
    WARNING_FIELDS,
    AccessorSpec,
    ContractManifest,
    PipelineFileError,
)

pytestmark = [pytest.mark.unit]

# --------------------------------------------------------------------------
# Growth guard. Removing or renaming an entry is a breaking change and fails
# the "superset" test. Adding an entry fails the "exact" test until the
# snapshot below is updated in the same commit -- a deliberate act.
# --------------------------------------------------------------------------
SNAPSHOT_ACCESSORS = frozenset(
    {
        "capabilities",
        "frequency_calibration",
        "refit_snap_tol_mhz",
        "read_metadata",
        "read_tables",
        "read_table",
        "settings_defaults",
        "settings_show",
        "get_final_products",
        "review_log",
        "get_pipeline_info",
        "compute_display_ft",
        "fid_samples",
        "display_units",
        "fit_thresholds",
        "window_status",
        "preview_source",
        "window_model",
        "spectrum_model",
        "analysis_fingerprint",
        "status",
    }
)
SNAPSHOT_SCHEMAS = frozenset(
    {
        "ftmw/error@1",
        "ftmw/capabilities@1",
        "ftmw/fid_samples@1",
        "ftmw/display_units@1",
        "ftmw/fit_thresholds@1",
        "ftmw/window_status@1",
        "ftmw/source_preview@1",
        "ftmw/window_model@1",
        "ftmw/spectrum_model@1",
        "ftmw/analysis_fingerprint@1",
        "ftmw/calibration@1",
        "ftmw/snap_tolerance@1",
        "ftmw/metadata@1",
        "ftmw/tables@1",
        "ftmw/table@1",
        "ftmw/settings_defaults@1",
        "ftmw/settings@1",
        "ftmw/final_products@1",
        "ftmw/review_log@1",
        "ftmw/pipeline_info@1",
        "ftmw/display_ft@1",
        "ftmw/curation_action@1",
        "ftmw/status@1",
        "ftmw/run_result@1",
        "ftmw/stage_started@1",
        "ftmw/stage_finished@1",
        "ftmw/window_progress@1",
        "ftmw/scan_progress@1",
        "ftmw/invalidated@1",
        "ftmw/warning@1",
    }
)
SNAPSHOT_CODES = frozenset(
    {
        "stage_not_run",
        "not_found",
        "incomplete_provenance",
        "file_incompatible",
        "file_corrupt",
        "epoch_mismatch",
        "file_exists",
        "bad_setting",
        "algorithm_failed",
        "cancelled",
        "callback_failed",
        "write_conflict",
        "pipeline_error",
    }
)
SNAPSHOT_FILE_BOUND = {
    "capabilities": False,
    "frequency_calibration": True,
    "refit_snap_tol_mhz": True,
    "read_metadata": True,
    "read_tables": True,
    "read_table": True,
    "settings_defaults": False,
    "settings_show": True,
    "get_final_products": True,
    "review_log": True,
    "get_pipeline_info": True,
    "compute_display_ft": True,
    "fid_samples": True,
    "display_units": True,
    "fit_thresholds": True,
    "window_status": True,
    "preview_source": False,
    "window_model": True,
    "spectrum_model": True,
    "analysis_fingerprint": True,
    "status": True,
}
SNAPSHOT_PIPELINE_NAMES = {
    "get_final_products": "final_products",
    "get_pipeline_info": "info",
}
SNAPSHOT_METADATA_KEYS: frozenset = frozenset(
    {
        "file.format_version",
        "file.completed_stages",
        "fid.n_points",
        "fid.duration_us",
        "fid.probe_freq_mhz",
        "fid.sideband",
        "fid.shots",
        "stage3.n_peaks",
        "stage4.n_windows",
        "stage5.n_fitted_peaks",
        "stage5.acquisition_us",
        "stage5.shape",
        "timebase.epsilon",
        "timebase.sigma_epsilon",
        "timebase.kappa_sys",
        "timebase.lattice_g_mhz",
        "timebase.n_detected",
        "timebase.n_used",
        "timebase.preconditions_passed",
    }
    | {
        f"{sec}.{k}"
        for sec in ("ft", "stage1")
        for k in (
            "start_us",
            "end_us",
            "trim_min_mhz",
            "trim_max_mhz",
            "units_power",
            "acquisition_us",
        )
    }
    | {
        f"{sec}.{k}"
        for sec in ("tau", "tau_g")
        for k in (
            "tau_maj_us",
            "sigma_tau_us",
            "n_contributors",
            "n_spur_bins",
            "n_seg",
            "preconditions_passed",
        )
    }
)
SNAPSHOT_TABLES: dict = {
    "fit_peaks": {
        "window_id",
        "frequency_mhz",
        "decay_rate",
        "shape",
        "peak_uid",
        "origin",
        "derivation",
        "clock_lattice",
        "knockout_p_value",
        "knockout_supported",
        "knockout_aicc_delta",
    },
    "windows": {"window_id", "freq_min", "freq_max"},
    "window_status": {
        "window_id",
        "freq_min_mhz",
        "freq_max_mhz",
        "created",
        "n_fitted_peaks",
        "n_fitted_peaks__status",
        "live",
        "live__status",
    },
}
SNAPSHOT_FIELDS: dict = {
    "CalibrationStamp": {
        "state",
        "epsilon",
        "sigma_epsilon",
        "sigma_floor_khz",
        "probe_freq_mhz",
        "sideband",
    },
    "FinalPeak": {
        "peak_uid",
        "window_id",
        "origin",
        "derivation",
        "clock_lattice",
        "knockout_p_value",
        "knockout_supported",
        "knockout_aicc_delta",
        "frequency_mhz",
        "sigma_f_khz",
        "decay_time_us",
        "decay_time_error_us",
        "shape",
        "fwhm_mhz",
        "detection_index",
        "fit_window_mhz",
    },
    "DecisionLogEntry": {
        "order_index",
        "window_id",
        "frequency_mhz",
        "kind",
        "provenance",
        "evidence",
    },
    "RefitWindowResult": {"converged"},
    "PreviewWindowResult": {"converged"},
    "AppliedWindowResult": {"converged"},
    "CurationAction": {"action", "window_id", "peak_uid", "epsilon"},
    "SettingRow": {"path", "value", "type", "nullable", "units", "choices", "bounds"},
    "PipelineInfo": {
        "stage_environments",
        "last_written_with",
        "environment_drift",
        "runtime_environment_drift",
        "current_environment",
        "environment_acknowledged",
        "warnings",
    },
    "ComplexFT": {"freq_array", "complex_spectrum", "metadata"},
    "ComplexFT.metadata": {"amplitude_scale", "units_label", "pad_factor"},
    "StageStarted": {"schema", "operation", "stage"},
    "StageFinished": {"schema", "operation", "stage", "elapsed_s", "summary"},
    "WindowProgress": {
        "schema",
        "operation",
        "stage",
        "phase",
        "round",
        "index",
        "total",
        "window_id",
        "n_peaks",
        "chi2r",
        "elapsed_s",
        "dropped",
    },
    "ScanProgress": {"schema", "operation", "stage", "knob", "value", "index", "total"},
    "Invalidated": {"schema", "operation", "stage", "stages"},
    "PipelineWarning": {"schema", "operation", "stage", "code", "message"},
    "PipelineWarning.slow_window": {"window_id", "elapsed_s", "threshold_s"},
    "PipelineWarning.epoch_acknowledged": {"file_epoch", "current_epoch"},
    "PipelineWarning.environment_drift": {"fields"},
    "PipelineWarning.frame_mismatch": {"actions"},
    "PipelineWarning.walk_fallback": {"reason", "n_windows"},
    "PipelineWarning.timebase_skipped": set(),
}
SNAPSHOT_VOCABULARIES = {
    "decision_kind": {"add", "remove", "merge", "split", "accept", "create_window"},
    "decision_provenance": {"user"},
    "stage_state": {"complete", "partial", "not_run"},
    "warning_code": {
        "slow_window",
        "epoch_acknowledged",
        "environment_drift",
        "frame_mismatch",
        "walk_fallback",
        "timebase_skipped",
    },
    "restart_reason": {
        "restart_requested",
        "settings_changed",
        "incomplete_provenance",
        "thaw_refit",
    },
}


def _all_subclasses(cls):
    seen = []
    stack = list(cls.__subclasses__())
    while stack:
        c = stack.pop()
        if c not in seen:
            seen.append(c)
            stack.extend(c.__subclasses__())
    return seen


def _subparser_choices(parser: argparse.ArgumentParser) -> dict:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def _read_verbs() -> set:
    top = _subparser_choices(create_parser())
    assert "read" in top
    return set(_subparser_choices(top["read"]))


def _verb_parser(name: str) -> argparse.ArgumentParser:
    """The parser of an accessor's CLI verb: always ``read <name>``."""
    parser = create_parser()
    for part in ("read", name):
        choices = _subparser_choices(parser)
        assert part in choices, (name, part)
        parser = choices[part]
    return parser


# ---- every accessor exists on all three interfaces -----------------------


@pytest.mark.parametrize("name", MANIFEST.accessors)
def test_accessor_on_api(name):
    assert callable(getattr(api, name))


@pytest.mark.parametrize("name", MANIFEST.accessors)
def test_accessor_on_pipeline(name):
    assert callable(getattr(ftmwpipeline.Pipeline, MANIFEST.pipeline_names[name]))


@pytest.mark.parametrize("name", MANIFEST.accessors)
def test_accessor_has_a_cli_verb(name):
    # Every accessor's verb is ``read <name>``, spelled as the API name.
    assert isinstance(_verb_parser(name), argparse.ArgumentParser)
    assert name in _read_verbs()


def test_manifest_declares_no_cli_verb_mapping():
    assert not hasattr(MANIFEST, "cli_verbs")
    assert "cli" not in AccessorSpec._fields


# ---- binding kind: file-bound vs file-less -------------------------------

_PATH_NAMES = {"file_path", "path", "filepath"}


def _params(fn):
    return list(inspect.signature(fn).parameters)


def test_file_bound_names_exactly_the_accessors():
    assert set(MANIFEST.file_bound) == set(MANIFEST.accessors)
    assert all(isinstance(v, bool) for v in MANIFEST.file_bound.values())


def test_file_bound_is_read_only():
    with pytest.raises(TypeError):
        MANIFEST.file_bound["x"] = True  # type: ignore[index]


def test_file_bound_must_match_accessors():
    with pytest.raises(ValueError):
        ContractManifest(1, ("a",), (), (), (), {}, {})
    with pytest.raises(ValueError):
        ContractManifest(1, (), (), (), (), {}, {"b": False})
    ok = ContractManifest(1, ("a",), (), (), (), {}, {"a": True})
    assert dict(ok.file_bound) == {"a": True}


def test_accessor_spec_is_name_and_binding():
    spec = AccessorSpec("x", file_bound=True)
    assert tuple(spec) == ("x", True, None)


@pytest.mark.parametrize("name", MANIFEST.accessors)
def test_accessor_binding_matches_declared_kind(name):
    pipeline_name = MANIFEST.pipeline_names[name]
    static = inspect.getattr_static(ftmwpipeline.Pipeline, pipeline_name)
    verb_dests = {a.dest for a in _verb_parser(name)._actions}
    if MANIFEST.file_bound[name]:
        # An instance method of an opened Pipeline that takes no path ...
        assert inspect.isfunction(static)
        assert _params(static)[0] == "self"
        assert not set(_params(static)[1:2]) & _PATH_NAMES
        # ... an api function that takes the path first ...
        assert _params(getattr(api, name))[0] in _PATH_NAMES
        # ... and a read verb with a file argument.
        assert "file_path" in verb_dests
    else:
        # A staticmethod; no path anywhere.
        assert isinstance(static, staticmethod)
        assert (
            not set(_params(getattr(ftmwpipeline.Pipeline, pipeline_name)))
            & _PATH_NAMES
        )
        assert not set(_params(getattr(api, name))) & _PATH_NAMES
        assert "file_path" not in verb_dests


def test_capabilities_is_declared_file_less():
    assert MANIFEST.file_bound["capabilities"] is False


# ---- codes <-> exception classes -----------------------------------------


def test_every_code_maps_to_exactly_one_class():
    # The base class owns the declared fallback code ``pipeline_error``.
    classes = [PipelineFileError, *_all_subclasses(PipelineFileError)]
    for code in MANIFEST.codes:
        # A subclass that refines a code (PipelineFileNotFoundError is a
        # NotFoundError with kind "file") inherits it rather than declaring it.
        owners = [c for c in classes if "code" in vars(c) and c.code == code]
        assert len(owners) == 1, (code, owners)


def test_every_class_code_is_declared():
    for cls in _all_subclasses(PipelineFileError):
        assert cls.code in MANIFEST.codes, cls


def test_base_fallback_code_is_declared():
    # run_pipeline reports a failure that is not a typed error under it.
    assert PipelineFileError.code == "pipeline_error"
    assert PipelineFileError.code in MANIFEST.codes


def test_cli_exit_codes_are_the_documented_table():
    from ftmwpipeline.cli.contract_commands import (
        DEFAULT_ERROR_EXIT,
        EXIT_CODES,
        INTERRUPTED_EXIT,
    )

    # Only the non-default codes are listed; every other code exits 1. The
    # table may name codes a later wave introduces, so it is not compared with
    # MANIFEST.codes.
    assert EXIT_CODES == {"file_corrupt": 2, "algorithm_failed": 2, "cancelled": 130}
    assert DEFAULT_ERROR_EXIT == 1 and INTERRUPTED_EXIT == 130


def test_every_manifest_code_has_an_exit_code():
    from ftmwpipeline.cli.contract_commands import EXIT_CODES, exit_code_for

    for code in MANIFEST.codes:
        stand_in = SimpleNamespace(code=code)  # exit_code_for reads only .code
        got = exit_code_for(stand_in)  # type: ignore[arg-type]
        assert got == EXIT_CODES.get(code, 1)
        assert got in (1, 2, 130)


# ---- schema names ---------------------------------------------------------


@pytest.mark.parametrize("name", MANIFEST.schemas)
def test_schema_names_well_formed(name):
    assert SCHEMA_NAME_RE.match(name)
    payload, _, rev = name[len("ftmw/") :].partition("@")
    assert payload and int(rev) >= 1


@pytest.mark.parametrize(
    "bad",
    ["error@1", "ftmw/Error@1", "ftmw/error", "ftmw/error@0", "ftmw/error@01", "x/e@1"],
)
def test_malformed_schema_rejected(bad):
    assert not SCHEMA_NAME_RE.match(bad)
    with pytest.raises(ValueError):
        ContractManifest(0, (), (bad,), (), (), {})


def test_duplicates_rejected():
    with pytest.raises(ValueError):
        ContractManifest(0, ("a", "a"), (), (), (), {})
    with pytest.raises(ValueError):
        ContractManifest(0, (), (), ("c", "c"), (), {})


def test_no_duplicate_entries_in_live_manifest():
    for group in (MANIFEST.accessors, MANIFEST.schemas, MANIFEST.codes):
        assert len(set(group)) == len(group)


# ---- immutability ---------------------------------------------------------


def test_manifest_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        MANIFEST.accessors = ()  # type: ignore[misc]


def test_manifest_tables_read_only():
    with pytest.raises(TypeError):
        MANIFEST.tables["x"] = ()  # type: ignore[index]


def test_manifest_sequences_are_tuples():
    for group in (MANIFEST.accessors, MANIFEST.schemas, MANIFEST.codes):
        assert isinstance(group, tuple)
    for cols in MANIFEST.tables.values():
        assert isinstance(cols, tuple)


def test_manifest_version_matches_package():
    assert MANIFEST.contract_version == ftmwpipeline.CONTRACT_VERSION == 10
    assert isinstance(ftmwpipeline.CONTRACT_VERSION, int)


# ---- growth guard ---------------------------------------------------------


def test_nothing_declared_disappears():
    assert SNAPSHOT_ACCESSORS <= set(MANIFEST.accessors)
    assert SNAPSHOT_SCHEMAS <= set(MANIFEST.schemas)
    assert SNAPSHOT_CODES <= set(MANIFEST.codes)
    for name, bound in SNAPSHOT_FILE_BOUND.items():
        assert MANIFEST.file_bound[name] is bound
    assert SNAPSHOT_METADATA_KEYS <= set(MANIFEST.metadata_keys)
    for table, cols in SNAPSHOT_TABLES.items():
        assert table in MANIFEST.tables
        assert set(cols) <= set(MANIFEST.tables[table])
    for name, pname in SNAPSHOT_PIPELINE_NAMES.items():
        assert MANIFEST.pipeline_names[name] == pname
    for type_name, names in SNAPSHOT_FIELDS.items():
        assert set(names) <= set(MANIFEST.fields[type_name])
    for vocab, values in SNAPSHOT_VOCABULARIES.items():
        assert values <= set(MANIFEST.vocabularies[vocab])


def test_additions_update_the_snapshot():
    """Adding a contract element must be a deliberate edit of this snapshot."""
    assert set(MANIFEST.accessors) == SNAPSHOT_ACCESSORS
    assert set(MANIFEST.schemas) == SNAPSHOT_SCHEMAS
    assert set(MANIFEST.codes) == SNAPSHOT_CODES
    assert dict(MANIFEST.file_bound) == SNAPSHOT_FILE_BOUND
    assert set(MANIFEST.metadata_keys) == SNAPSHOT_METADATA_KEYS
    assert set(MANIFEST.fields) == set(SNAPSHOT_FIELDS)
    assert {k: set(v) for k, v in MANIFEST.vocabularies.items()} == (
        SNAPSHOT_VOCABULARIES
    )
    assert {n: p for n, p in MANIFEST.pipeline_names.items() if n != p} == (
        SNAPSHOT_PIPELINE_NAMES
    )
    assert set(MANIFEST.tables) == set(SNAPSHOT_TABLES)
    for table, cols in SNAPSHOT_TABLES.items():
        # The snapshot pins the promised core; more columns may be declared
        # only if the code produces them (see the declared-columns tests).
        assert cols <= set(MANIFEST.tables[table])


# ---- declared metadata keys, tables, fields, vocabularies ----------------


def test_metadata_key_snapshot_is_exact():
    assert set(MANIFEST.metadata_keys) == SNAPSHOT_METADATA_KEYS


def test_declared_tables_and_columns_are_in_the_read_registry():
    from ftmwpipeline._internal.read_impl import _TABLE_SPECS

    for table, columns in MANIFEST.tables.items():
        assert table in _TABLE_SPECS, table
        assert set(columns) <= set(_TABLE_SPECS[table].specs), table


def test_declared_blackquill_columns_are_all_declared():
    """Every column read_impl emits for fit_peaks / windows is declared."""
    from ftmwpipeline._internal.read_impl import _TABLE_SPECS

    for table in ("fit_peaks", "windows"):
        assert set(MANIFEST.tables[table]) == set(_TABLE_SPECS[table].specs)


def test_declared_conditional_metadata_scalars_are_written_by_their_stage():
    """tau. / tau_g. / timebase. keys cannot be seen on the stage-5 fixture, so
    they are checked against the attribute lists that write them."""
    from ftmwpipeline._internal.read_impl import _TIMEBASE_ATTRS
    from ftmwpipeline.io.tau_calibration_serialization import _TAU_SCALAR_ATTRS

    for key in MANIFEST.metadata_keys:
        section, _, name = key.partition(".")
        if section in ("tau", "tau_g"):
            assert name in _TAU_SCALAR_ATTRS, key
        elif section == "timebase":
            assert name in _TIMEBASE_ATTRS, key


def _type_registry() -> dict:
    from ftmwpipeline._internal.stage6_impl import (
        AppliedWindowResult,
        PreviewWindowResult,
        RefitWindowResult,
    )
    from ftmwpipeline._internal.tuning.settings_inspection import SettingRow
    from ftmwpipeline.core.calibration import CalibrationStamp
    from ftmwpipeline.core.curation import CurationAction
    from ftmwpipeline.contract import EVENT_TYPES
    from ftmwpipeline.core.data_structures import DecisionLogEntry, FinalPeak

    return {
        **{cls.__name__: cls for cls in EVENT_TYPES},
        "CalibrationStamp": CalibrationStamp,
        "FinalPeak": FinalPeak,
        "DecisionLogEntry": DecisionLogEntry,
        "RefitWindowResult": RefitWindowResult,
        "PreviewWindowResult": PreviewWindowResult,
        "AppliedWindowResult": AppliedWindowResult,
        "CurationAction": CurationAction,
        "SettingRow": SettingRow,
    }


#: Declared types whose result is not a dataclass; their fields are produced
#: keys / attributes, checked against a real file in
#: ``tests/integration/test_contract_declared_elements.py``.
_PRODUCED_TYPES = {"PipelineInfo", "ComplexFT", "ComplexFT.metadata"}

#: ``PipelineWarning.<code>``: a warning code's wire fields, which the
#: dataclass holds in ``details`` (checked against ``WARNING_FIELDS``).
_WARNING_CODE_TYPES = {f"PipelineWarning.{code}" for code in WARNING_FIELDS}


def test_every_declared_type_is_resolvable():
    known = set(_type_registry()) | _PRODUCED_TYPES | _WARNING_CODE_TYPES
    assert set(MANIFEST.fields) <= known, set(MANIFEST.fields) - known


def test_every_declared_field_exists_on_its_dataclass():
    registry = _type_registry()
    for type_name, names in MANIFEST.fields.items():
        if type_name in _PRODUCED_TYPES:
            continue
        if type_name in _WARNING_CODE_TYPES:
            code = type_name.split(".", 1)[1]
            assert tuple(names) == WARNING_FIELDS[code], type_name
            continue
        cls = registry[type_name]
        assert dataclasses.is_dataclass(cls), type_name
        have = {f.name for f in dataclasses.fields(cls)}
        assert set(names) <= have, (type_name, set(names) - have)


def test_complex_ft_declared_attributes_exist():
    import numpy as np

    from ftmwpipeline.core.data_structures import ComplexFT

    ft = ComplexFT(np.zeros(2), np.zeros(2, dtype=complex), {})
    for name in MANIFEST.fields["ComplexFT"]:
        assert hasattr(ft, name), name


def test_decision_vocabularies_match_the_code():
    from ftmwpipeline._internal.stage6_impl import _FIT_EDIT_KINDS
    from ftmwpipeline.core.data_structures import (
        DECISION_KINDS,
        DECISION_PROVENANCES,
    )

    assert set(MANIFEST.vocabularies["decision_kind"]) == set(DECISION_KINDS)
    assert set(MANIFEST.vocabularies["decision_provenance"]) == set(
        DECISION_PROVENANCES
    )
    assert set(_FIT_EDIT_KINDS) <= set(DECISION_KINDS)


def test_decision_kinds_recorded_by_the_code_are_declared():
    """Every literal ``kind=`` a decision is recorded with is in the vocabulary."""
    import re

    from ftmwpipeline._internal import stage6_impl

    source = inspect.getsource(stage6_impl)
    declared = set(MANIFEST.vocabularies["decision_kind"])
    recorded = set(re.findall(r'"kind": "([a-z_]+)"', source))
    # "kind" keys also tag attention evidence; keep only decision-shaped ones.
    decision_like = recorded & {
        "add",
        "remove",
        "merge",
        "split",
        "accept",
        "create_window",
    }
    assert decision_like <= declared
    assert {"add", "remove", "merge", "split", "accept", "create_window"} <= recorded


def test_manifest_rejects_pipeline_names_for_non_accessors():
    with pytest.raises(ValueError):
        ContractManifest(0, ("a",), (), (), (), {}, {"a": True}, {"b": "x"})
    ok = ContractManifest(0, ("a",), (), (), (), {}, {"a": True})
    assert ok.pipeline_names["a"] == "a"


def test_manifest_fields_and_vocabularies_read_only_and_unique():
    with pytest.raises(TypeError):
        MANIFEST.fields["x"] = ()  # type: ignore[index]
    with pytest.raises(TypeError):
        MANIFEST.vocabularies["x"] = ()  # type: ignore[index]
    with pytest.raises(ValueError):
        ContractManifest(0, (), (), (), (), {}, {}, {}, {"T": ("a", "a")})
