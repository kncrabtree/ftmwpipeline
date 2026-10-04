"""``CurationConflictError`` (``curation_conflict``) and the widened ``not_found``.

A valid curation request that conflicts with the file's review state raises
``CurationConflictError``: a ``ValueError`` (what these refusals raised before
they were typed) carrying a stable ``reason`` slug and the ``ids`` involved.
``NotFoundError`` gained the kind ``"decision"`` and may carry frequencies
(floats) as ids.

The refusals themselves, raised from a real fit, are covered by
``tests/unit/stage6/test_curation_typed_refusals.py``; this file pins the class.
"""

from __future__ import annotations

import json
import pickle

import pytest

import ftmwpipeline
from ftmwpipeline import contract
from ftmwpipeline.file_manager import (
    CurationConflictError,
    NotFoundError,
    NotFoundValueError,
    PipelineFileError,
)

pytestmark = [pytest.mark.unit]


def test_it_is_a_value_error_in_the_contract_family():
    err = CurationConflictError("line_already_fitted", [18148190])
    assert isinstance(err, PipelineFileError)
    assert isinstance(err, ValueError)
    assert not isinstance(err, (KeyError, RuntimeError))
    assert CurationConflictError.code == "curation_conflict"


def test_attributes_and_wire_form():
    err = CurationConflictError("targets_span_windows", [3, 7], message="m")
    assert err.reason == "targets_span_windows"
    assert err.ids == [3, 7]
    assert str(err) == "m"
    d = err.to_dict()
    assert d == {
        "schema": "ftmw/error@1",
        "code": "curation_conflict",
        "message": "m",
        "reason": "targets_span_windows",
        "ids": [3, 7],
    }
    assert list(d)[:3] == ["schema", "code", "message"]
    json.dumps(d, allow_nan=False)


def test_ids_default_to_none_involved():
    err = CurationConflictError("baseline_unavailable")
    assert err.ids == []
    assert err.to_dict()["ids"] == []
    assert str(err) == "curation conflict (baseline_unavailable)"


def test_the_default_message_names_the_reason_and_ids():
    err = CurationConflictError("orphans_created_window", [1, 4])
    assert str(err) == "curation conflict (orphans_created_window): 1, 4"


def test_ids_are_copied_and_must_not_be_a_bare_string():
    ids = [1, 2]
    err = CurationConflictError("replay_conflict", ids)
    ids.append(3)
    assert err.ids == [1, 2]
    with pytest.raises(TypeError):
        CurationConflictError("replay_conflict", "12")  # type: ignore[arg-type]


def test_it_pickles_with_its_attributes():
    err = CurationConflictError("replay_conflict", [12, 11], message="replay failed")
    clone = pickle.loads(pickle.dumps(err))
    assert type(clone) is CurationConflictError
    assert (clone.reason, clone.ids, str(clone)) == (
        "replay_conflict",
        [12, 11],
        "replay failed",
    )
    assert clone.to_dict() == err.to_dict()


def test_it_is_declared_in_the_contract():
    assert "curation_conflict" in contract.MANIFEST.codes
    assert "curation_conflict" in contract.capabilities()["codes"]
    assert contract.CurationConflictError is CurationConflictError
    assert "CurationConflictError" in contract.__all__
    assert ftmwpipeline.CurationConflictError is CurationConflictError
    assert contract.CONTRACT_VERSION == ftmwpipeline.CONTRACT_VERSION >= 12


def test_the_cli_exits_one_for_it():
    from ftmwpipeline.cli.contract_commands import EXIT_CODES, exit_code_for

    assert "curation_conflict" not in EXIT_CODES  # the default, a user error
    assert exit_code_for(CurationConflictError("replay_conflict")) == 1


def test_not_found_accepts_the_decision_kind_and_frequency_ids():
    err = NotFoundValueError("decision", [4, 8])
    assert err.kind == "decision" and err.ids == [4, 8]
    assert isinstance(err, (ValueError, KeyError, NotFoundError))
    assert err.to_dict()["kind"] == "decision"

    by_freq = NotFoundValueError("peak", [26611.0556, 26611.0566])
    assert by_freq.ids == [26611.0556, 26611.0566]
    d = by_freq.to_dict()
    assert d["ids"] == [26611.0556, 26611.0566]
    json.dumps(d, allow_nan=False)
    clone = pickle.loads(pickle.dumps(by_freq))
    assert clone.ids == by_freq.ids and clone.kind == "peak"
