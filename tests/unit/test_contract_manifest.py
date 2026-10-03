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
SNAPSHOT_ACCESSORS = frozenset({"capabilities", "fid_samples", "display_units"})
SNAPSHOT_SCHEMAS = frozenset(
    {
        "ftmw/error@1",
        "ftmw/capabilities@1",
        "ftmw/fid_samples@1",
        "ftmw/display_units@1",
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
    }
)
SNAPSHOT_FILE_BOUND = {
    "capabilities": False,
    "fid_samples": True,
    "display_units": True,
}
SNAPSHOT_METADATA_KEYS: frozenset = frozenset()
SNAPSHOT_TABLES: dict = {}


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
    return _subparser_choices(_subparser_choices(create_parser())["read"])[name]


# ---- every accessor exists on all three interfaces -----------------------


@pytest.mark.parametrize("name", MANIFEST.accessors)
def test_accessor_on_api(name):
    assert callable(getattr(api, name))


@pytest.mark.parametrize("name", MANIFEST.accessors)
def test_accessor_on_pipeline(name):
    assert callable(getattr(ftmwpipeline.Pipeline, name))


@pytest.mark.parametrize("name", MANIFEST.accessors)
def test_accessor_is_cli_read_verb(name):
    assert name in _read_verbs()


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
    assert tuple(spec) == ("x", True)


@pytest.mark.parametrize("name", MANIFEST.accessors)
def test_accessor_binding_matches_declared_kind(name):
    static = inspect.getattr_static(ftmwpipeline.Pipeline, name)
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
        assert not set(_params(getattr(ftmwpipeline.Pipeline, name))) & _PATH_NAMES
        assert not set(_params(getattr(api, name))) & _PATH_NAMES
        assert "file_path" not in verb_dests


def test_capabilities_is_declared_file_less():
    assert MANIFEST.file_bound["capabilities"] is False


# ---- codes <-> exception classes -----------------------------------------


def test_every_code_maps_to_exactly_one_class():
    classes = [c for c in _all_subclasses(PipelineFileError)]
    for code in MANIFEST.codes:
        # A subclass that refines a code (PipelineFileNotFoundError is a
        # NotFoundError with kind "file") inherits it rather than declaring it.
        owners = [c for c in classes if "code" in vars(c) and c.code == code]
        assert len(owners) == 1, (code, owners)


def test_every_class_code_is_declared():
    for cls in _all_subclasses(PipelineFileError):
        assert cls.code in MANIFEST.codes, cls


def test_base_fallback_code_is_not_declared():
    assert PipelineFileError.code == "pipeline_error"
    assert PipelineFileError.code not in MANIFEST.codes


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
    assert MANIFEST.contract_version == ftmwpipeline.CONTRACT_VERSION == 1
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


def test_additions_update_the_snapshot():
    """Adding a contract element must be a deliberate edit of this snapshot."""
    assert set(MANIFEST.accessors) == SNAPSHOT_ACCESSORS
    assert set(MANIFEST.schemas) == SNAPSHOT_SCHEMAS
    assert set(MANIFEST.codes) == SNAPSHOT_CODES
    assert dict(MANIFEST.file_bound) == SNAPSHOT_FILE_BOUND
    assert set(MANIFEST.metadata_keys) == SNAPSHOT_METADATA_KEYS
    assert {k: set(v) for k, v in MANIFEST.tables.items()} == {
        k: set(v) for k, v in SNAPSHOT_TABLES.items()
    }
