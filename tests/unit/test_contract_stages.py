"""Stage mappings and the capabilities ``stages`` list agree with the code.

Spec: ``dev-docs/CONTRACT_STRATEGY.md`` §Status and settings. The mappings are
checked against the sources they are derived from (settings inspection and
mutation specs, the tuning registry, the stage tracker), not against a copy.
"""

from __future__ import annotations

import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline._internal.tuning.registry import list_knobs
from ftmwpipeline._internal.tuning.settings_inspection import _STAGE_SPECS
from ftmwpipeline._internal.tuning.settings_mutation import _MUT_SPECS
from ftmwpipeline.contract import (
    MANIFEST,
    STAGE_KEYS,
    STAGE_KNOB_PREFIX,
    STAGE_SETTINGS_PREFIX,
    STAGE_STATES,
    Stage,
    capabilities,
    rerun_order,
    stage_depends_on,
    stage_for_knob_prefix,
)
from ftmwpipeline.file_manager import PipelineStageTracker

pytestmark = [pytest.mark.unit]


def test_mappings_cover_every_stage_and_are_read_only():
    # Mutation: a stage missing from either mapping.
    for mapping in (STAGE_SETTINGS_PREFIX, STAGE_KNOB_PREFIX):
        assert set(mapping) == set(Stage)
        with pytest.raises(TypeError):
            mapping[Stage.FT] = "x"  # type: ignore[index]


def test_settings_prefixes_are_exactly_those_of_the_settings_code():
    # Mutation: a prefix renamed or assigned to the wrong stage.
    mapped = {p for p in STAGE_SETTINGS_PREFIX.values() if p is not None}
    assert mapped == {s.prefix for s in _STAGE_SPECS}
    # stage1 is mutated through its own windowing knobs, not _MUT_SPECS.
    assert set(_MUT_SPECS) == mapped - {"stage1"}
    assert STAGE_SETTINGS_PREFIX[Stage.TAU] == STAGE_SETTINGS_PREFIX[Stage.TAU_G]
    for stage in (Stage.DATA, Stage.TIMEBASE, Stage.REVIEW):
        assert STAGE_SETTINGS_PREFIX[stage] is None


def test_knob_prefixes_own_every_registered_knob_and_none_is_empty():
    # Mutation: a prefix no knob uses, or a registered knob no prefix owns.
    prefixes = [p for p in STAGE_KNOB_PREFIX.values() if p is not None]
    paths = [k.path for k in list_knobs(include_advanced=True)]
    for p in prefixes:
        assert any(x == p or x.startswith(p + ".") for x in paths), p
    for path in paths:
        assert any(path == p or path.startswith(p + ".") for p in prefixes), path
    assert STAGE_KNOB_PREFIX[Stage.TIMEBASE] is None
    assert STAGE_KNOB_PREFIX[Stage.REVIEW] is None


def test_knob_mapping_is_one_to_one_with_an_inverse():
    # Mutation: two stages sharing a knob prefix, or a wrong inverse.
    prefixes = [p for p in STAGE_KNOB_PREFIX.values() if p is not None]
    assert len(prefixes) == len(set(prefixes))
    for stage, prefix in STAGE_KNOB_PREFIX.items():
        if prefix is not None:
            assert stage_for_knob_prefix(prefix) is stage
    with pytest.raises(ValueError):
        stage_for_knob_prefix("stage9")


def test_depends_on_is_the_trackers_graph_in_canonical_names():
    # Mutation: dependencies returned as storage keys, or dropped.
    by_key = {v: k for k, v in STAGE_KEYS.items()}
    for stage in Stage:
        expected = {
            by_key[k]
            for k in PipelineStageTracker.STAGE_DEPENDENCIES[STAGE_KEYS[stage]]
        }
        assert set(stage_depends_on(stage)) == expected
        assert list(stage_depends_on(stage)) == [s for s in Stage if s in expected]


def test_rerun_order_is_a_topological_order_of_every_stage():
    # Mutation: an order that places a stage before a dependency.
    order = rerun_order()
    assert sorted(order, key=list(Stage).index) == list(Stage)
    for i, stage in enumerate(order):
        assert all(order.index(d) < i for d in stage_depends_on(stage))
    assert rerun_order() == order


def test_capabilities_stages_list_matches_the_mappings():
    # Mutation: a field dropped, wrong value, or non-canonical depends_on.
    stages = capabilities()["stages"]
    assert [e["stage"] for e in stages] == [s.value for s in Stage]
    for entry, stage in zip(stages, Stage):
        assert set(entry) == {
            "stage",
            "storage_key",
            "settings_prefix",
            "knob_prefix",
            "depends_on",
        }
        assert entry["storage_key"] == STAGE_KEYS[stage]
        assert entry["settings_prefix"] == STAGE_SETTINGS_PREFIX[stage]
        assert entry["knob_prefix"] == STAGE_KNOB_PREFIX[stage]
        assert entry["depends_on"] == [d.value for d in stage_depends_on(stage)]
    assert ftmw.capabilities() == capabilities()


def test_status_is_declared_with_its_schema_and_vocabulary():
    # Mutation: status dropped from the manifest, or partial left out.
    assert "status" in MANIFEST.accessors
    assert MANIFEST.file_bound["status"] is True
    assert "ftmw/status@1" in MANIFEST.schemas
    assert set(MANIFEST.vocabularies["stage_state"]) == set(STAGE_STATES)
    assert "partial" in STAGE_STATES
