"""The machine-contract manifest is complete, consistent, and only grows.

Spec: ``dev-docs/CONTRACT_STRATEGY.md`` §Versioning and stability ("a test can
assert that every declared element exists on all three interfaces and that
nothing declared silently disappears").
"""

from __future__ import annotations

import argparse
import dataclasses

import pytest

import ftmwpipeline
import ftmwpipeline.api as api
from ftmwpipeline.cli.main import create_parser
from ftmwpipeline.contract import (
    MANIFEST,
    SCHEMA_NAME_RE,
    ContractManifest,
    PipelineFileError,
)

pytestmark = [pytest.mark.unit]

# --------------------------------------------------------------------------
# Growth guard. Removing or renaming an entry is a breaking change and fails
# the "superset" test. Adding an entry fails the "exact" test until the
# snapshot below is updated in the same commit -- a deliberate act.
# --------------------------------------------------------------------------
SNAPSHOT_ACCESSORS = frozenset({"capabilities"})
SNAPSHOT_SCHEMAS = frozenset({"ftmw/error@1", "ftmw/capabilities@1"})
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


# ---- codes <-> exception classes -----------------------------------------


def test_every_code_maps_to_exactly_one_class():
    classes = [c for c in _all_subclasses(PipelineFileError)]
    for code in MANIFEST.codes:
        owners = [c for c in classes if c.code == code]
        assert len(owners) == 1, (code, owners)


def test_every_class_code_is_declared():
    for cls in _all_subclasses(PipelineFileError):
        assert cls.code in MANIFEST.codes, cls


def test_base_fallback_code_is_not_declared():
    assert PipelineFileError.code == "pipeline_error"
    assert PipelineFileError.code not in MANIFEST.codes


def test_cli_exit_code_covers_every_code():
    from ftmwpipeline.cli.contract_commands import EXIT_CODES

    assert set(MANIFEST.codes) <= set(EXIT_CODES)
    assert set(EXIT_CODES) <= set(MANIFEST.codes)


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
    assert MANIFEST.contract_version == ftmwpipeline.CONTRACT_VERSION
    assert isinstance(ftmwpipeline.CONTRACT_VERSION, int)


# ---- growth guard ---------------------------------------------------------


def test_nothing_declared_disappears():
    assert SNAPSHOT_ACCESSORS <= set(MANIFEST.accessors)
    assert SNAPSHOT_SCHEMAS <= set(MANIFEST.schemas)
    assert SNAPSHOT_CODES <= set(MANIFEST.codes)
    assert SNAPSHOT_METADATA_KEYS <= set(MANIFEST.metadata_keys)
    for table, cols in SNAPSHOT_TABLES.items():
        assert table in MANIFEST.tables
        assert set(cols) <= set(MANIFEST.tables[table])


def test_additions_update_the_snapshot():
    """Adding a contract element must be a deliberate edit of this snapshot."""
    assert set(MANIFEST.accessors) == SNAPSHOT_ACCESSORS
    assert set(MANIFEST.schemas) == SNAPSHOT_SCHEMAS
    assert set(MANIFEST.codes) == SNAPSHOT_CODES
    assert set(MANIFEST.metadata_keys) == SNAPSHOT_METADATA_KEYS
    assert {k: set(v) for k, v in MANIFEST.tables.items()} == {
        k: set(v) for k, v in SNAPSHOT_TABLES.items()
    }
