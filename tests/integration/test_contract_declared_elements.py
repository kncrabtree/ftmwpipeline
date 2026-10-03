"""Every declared ``read_metadata`` key and ``read_table`` column is produced.

Binding from the wave that declares the first one onward: the manifest
(``ftmwpipeline.contract.MANIFEST``) lists the metadata keys and table columns
the machine contract promises, and a fresh 2638 build must produce every one of
them (``dev-docs/CONTRACT_STRATEGY.md`` §Versioning and stability). With
nothing declared the checks are trivially satisfied; they gate every later
addition.

The 2638 build is the shared session fixture
``baseline_2638_stage5_small`` (the most complete, still cheap, baseline), so
it is only built when something is actually declared. Marked ``slow`` like
every other test that depends on that fixture.
"""

from __future__ import annotations

import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import MANIFEST, Pipeline

pytestmark = [pytest.mark.integration, pytest.mark.slow]


#: Sections of ``read_metadata`` that exist only once their stage has run. The
#: stage-5 fixture has no tau or timebase calibration; those keys are checked
#: against the attribute lists that write them (tests/unit/test_contract_manifest).
_CONDITIONAL_SECTIONS = {"tau", "tau_g", "timebase"}

#: ``get_pipeline_info`` keys emitted only when non-empty.
_CONDITIONAL_INFO_KEYS = {"warnings"}


@pytest.fixture
def built_2638(request):
    """The prebuilt 2638 file, requested lazily so an empty manifest costs nothing."""
    if not (MANIFEST.metadata_keys or MANIFEST.tables or MANIFEST.fields):
        return None
    return request.getfixturevalue("baseline_2638_stage5_small")


def test_every_declared_metadata_key_is_produced(built_2638):
    missing = []
    if MANIFEST.metadata_keys:
        via_api = ftmw.read_metadata(built_2638)
        via_pipeline = Pipeline.open(built_2638).read_metadata()
        for key in MANIFEST.metadata_keys:
            if key.partition(".")[0] in _CONDITIONAL_SECTIONS:
                continue  # stage not run in this fixture; see the unit test
            if key not in via_api or key not in via_pipeline:
                missing.append(key)
    assert missing == []


def test_every_declared_table_column_is_produced(built_2638):
    missing = {}
    for table, columns in MANIFEST.tables.items():
        produced = ftmw.read_table(built_2638, table)
        absent = [c for c in columns if c not in produced]
        if absent:
            missing[table] = absent
    assert missing == {}


def test_declared_tables_are_readable_tables(built_2638):
    if not MANIFEST.tables:
        return
    readable = ftmw.read_tables(built_2638)
    for table in MANIFEST.tables:
        assert table in readable or table.replace("_", "-") in readable, table


def test_every_declared_pipeline_info_key_is_produced(built_2638):
    info = ftmw.get_pipeline_info(built_2638)
    via_pipeline = Pipeline.open(built_2638).info()
    missing = [
        k
        for k in MANIFEST.fields["PipelineInfo"]
        if k not in _CONDITIONAL_INFO_KEYS and (k not in info or k not in via_pipeline)
    ]
    assert missing == []


def test_every_declared_display_ft_field_is_produced(built_2638):
    ft = ftmw.compute_display_ft(built_2638)
    for name in MANIFEST.fields["ComplexFT"]:
        assert hasattr(ft, name), name
    for key in MANIFEST.fields["ComplexFT.metadata"]:
        assert key in ft.metadata, key


def test_declared_final_peak_and_log_fields_on_real_rows(built_2638):
    import dataclasses

    fp = ftmw.get_final_products(built_2638)
    if fp is not None and fp.peaks:
        have = {f.name for f in dataclasses.fields(fp.peaks[0])}
        assert set(MANIFEST.fields["FinalPeak"]) <= have
    for entry in ftmw.review_log(built_2638):
        have = {f.name for f in dataclasses.fields(entry)}
        assert set(MANIFEST.fields["DecisionLogEntry"]) <= have
        assert entry.kind in MANIFEST.vocabularies["decision_kind"]
        assert entry.provenance in MANIFEST.vocabularies["decision_provenance"]
