"""Every declared ``read_metadata`` key and ``read_table`` column is produced.

Binding from the wave that declares the first one onward: the manifest
(``ftmwpipeline.contract.MANIFEST``) lists the metadata keys, table columns and
record fields the machine contract promises, and real builds must produce every
one of them (``dev-docs/CONTRACT_STRATEGY.md`` §Versioning and stability).

Each check runs on a file in which the thing checked is real:

* ``stage5_reviewed_2638`` -- Stage 6 has run and carries two recorded
  decisions, so ``get_final_products`` and ``review_log`` are non-empty;
* ``tau_timebase_calibrated_2638`` -- Stage 2b (both twins) and the timebase
  calibration have run, so every ``tau.`` / ``tau_g.`` / ``timebase.``
  metadata key is written.

Lists are asserted non-empty before their fields are checked, so a fixture that
stops producing rows fails here rather than passing vacuously. Marked ``slow``
like every test that depends on the 2638 fixtures.
"""

from __future__ import annotations

import dataclasses

import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import MANIFEST, Pipeline
from ftmwpipeline.core.data_structures import DECISION_KINDS, DECISION_PROVENANCES

pytestmark = [pytest.mark.integration, pytest.mark.slow]


#: Sections of ``read_metadata`` written by a stage the reviewed fixture does not
#: run; they are checked on the tau/timebase fixture instead.
_TAU_TIMEBASE_SECTIONS = {"tau", "tau_g", "timebase"}


def _section(key: str) -> str:
    return key.partition(".")[0]


def test_every_declared_metadata_key_is_produced(stage5_reviewed_2638):
    keys = [
        k for k in MANIFEST.metadata_keys if _section(k) not in _TAU_TIMEBASE_SECTIONS
    ]
    assert keys
    via_api = ftmw.read_metadata(stage5_reviewed_2638)
    via_pipeline = Pipeline.open(stage5_reviewed_2638).read_metadata()
    assert [k for k in keys if k not in via_api or k not in via_pipeline] == []


def test_every_declared_tau_and_timebase_metadata_key_is_produced(
    tau_timebase_calibrated_2638,
):
    keys = [k for k in MANIFEST.metadata_keys if _section(k) in _TAU_TIMEBASE_SECTIONS]
    assert {_section(k) for k in keys} == _TAU_TIMEBASE_SECTIONS
    via_api = ftmw.read_metadata(tau_timebase_calibrated_2638)
    via_pipeline = Pipeline.open(tau_timebase_calibrated_2638).read_metadata()
    assert [k for k in keys if k not in via_api or k not in via_pipeline] == []


def test_every_declared_table_column_is_produced(stage5_reviewed_2638):
    assert MANIFEST.tables
    missing = {}
    for table, columns in MANIFEST.tables.items():
        produced = ftmw.read_table(stage5_reviewed_2638, table)
        absent = [c for c in columns if c not in produced]
        if absent:
            missing[table] = absent
    assert missing == {}


def test_declared_tables_are_readable_tables(stage5_reviewed_2638):
    readable = ftmw.read_tables(stage5_reviewed_2638)
    for table in MANIFEST.tables:
        assert table in readable or table.replace("_", "-") in readable, table


def test_every_declared_pipeline_info_key_is_produced(stage5_reviewed_2638):
    info = ftmw.get_pipeline_info(stage5_reviewed_2638)
    via_pipeline = Pipeline.open(stage5_reviewed_2638).info()
    declared = MANIFEST.fields["PipelineInfo"]
    assert declared
    # ``warnings`` is always present (an empty list when there are none).
    assert [k for k in declared if k not in info or k not in via_pipeline] == []


def test_every_declared_display_ft_field_is_produced(stage5_reviewed_2638):
    ft = ftmw.compute_display_ft(stage5_reviewed_2638)
    for name in MANIFEST.fields["ComplexFT"]:
        assert hasattr(ft, name), name
    for key in MANIFEST.fields["ComplexFT.metadata"]:
        assert key in ft.metadata, key


def test_declared_final_peak_fields_on_real_rows(stage5_reviewed_2638):
    fp = ftmw.get_final_products(stage5_reviewed_2638)
    assert fp is not None and fp.peaks
    declared = set(MANIFEST.fields["FinalPeak"])
    assert declared
    for peak in fp.peaks:
        assert declared <= {f.name for f in dataclasses.fields(peak)}


def test_declared_decision_log_fields_on_real_rows(stage5_reviewed_2638):
    log = ftmw.review_log(stage5_reviewed_2638)
    assert len(log) >= 2
    declared = set(MANIFEST.fields["DecisionLogEntry"])
    assert declared
    for entry in log:
        assert declared <= {f.name for f in dataclasses.fields(entry)}


def test_recorded_decision_vocabulary_is_inside_the_declared_set(
    stage5_reviewed_2638,
):
    """Whatever the code records must be declared -- no pre-filtering."""
    log = ftmw.review_log(stage5_reviewed_2638)
    assert log
    kinds = {e.kind for e in log}
    provenances = {e.provenance for e in log}
    assert {"remove", "accept"} <= kinds
    assert kinds <= set(MANIFEST.vocabularies["decision_kind"])
    assert provenances <= set(MANIFEST.vocabularies["decision_provenance"])


def test_declared_vocabularies_are_the_ones_the_code_records_from():
    assert set(MANIFEST.vocabularies["decision_kind"]) == set(DECISION_KINDS)
    assert set(MANIFEST.vocabularies["decision_provenance"]) == set(
        DECISION_PROVENANCES
    )
