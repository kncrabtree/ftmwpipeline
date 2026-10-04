"""``contract_envelope`` and ``emit_contract_result``: the CLI's envelope forms.

Spec: ``dev-docs/CONTRACT_STRATEGY.md`` §Accessors, "Rules every accessor
follows". Every ``read <accessor>`` verb prints one JSON envelope: a dict or
dataclass payload is stamped directly, a list or tuple is wrapped as
``{"schema", "items"}``, a scalar as ``{"schema", "value"}``. Arrays never
inline; they go to ``.npy`` files under ``--output``.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from ftmwpipeline import Absent
from ftmwpipeline.cli.contract_commands import contract_envelope, emit_contract_result

pytestmark = [pytest.mark.unit]

SCHEMA = "ftmw/probe@1"


@dataclasses.dataclass(frozen=True)
class _Row:
    n: int
    live: object = Absent.NOT_RUN


@dataclasses.dataclass(frozen=True)
class _Stamped:
    __ftmw_schema__ = SCHEMA
    x: int


def _emit(result, capsys, *, output=None, schema=SCHEMA):
    emit_contract_result(result, schema=schema, output=output)
    return json.loads(capsys.readouterr().out)


# ---- contract_envelope: the three forms ------------------------------------


def test_list_is_wrapped_as_items():
    rows = [1, 2]
    assert contract_envelope(rows, SCHEMA) == {"schema": SCHEMA, "items": rows}


def test_tuple_is_wrapped_as_items():
    assert contract_envelope((1, 2), SCHEMA) == {"schema": SCHEMA, "items": (1, 2)}


def test_empty_list_is_still_wrapped():
    assert contract_envelope([], SCHEMA) == {"schema": SCHEMA, "items": []}


@pytest.mark.parametrize(
    "scalar", [0.5, 3, True, "text", np.float64(1.25), np.int64(2), np.bool_(True)]
)
def test_scalar_is_wrapped_as_value(scalar):
    env = contract_envelope(scalar, SCHEMA)
    assert env == {"schema": SCHEMA, "value": scalar}
    assert set(env) == {"schema", "value"}


def test_dict_and_dataclass_pass_through_unchanged():
    payload = {"a": 1}
    assert contract_envelope(payload, SCHEMA) is payload
    row = _Row(1)
    assert contract_envelope(row, SCHEMA) is row


# ---- emit_contract_result: the wire forms ----------------------------------


def test_dict_is_stamped_directly(capsys):
    assert _emit({"a": 1}, capsys) == {"schema": SCHEMA, "a": 1}


def test_list_of_records_wire_form_carries_absence_per_row(capsys):
    env = _emit([_Row(1), _Row(2, live=True)], capsys)
    assert env == {
        "schema": SCHEMA,
        "items": [
            {"n": 1, "live": None, "live_absent": "not_run"},
            {"n": 2, "live": True},
        ],
    }


def test_tuple_of_records_matches_the_list_form(capsys):
    rows = [_Row(1), _Row(2, live=False)]
    as_list = _emit(rows, capsys)
    as_tuple = _emit(tuple(rows), capsys)
    assert as_list == as_tuple


def test_scalar_wire_form(capsys):
    assert _emit(np.float64(0.25), capsys) == {"schema": SCHEMA, "value": 0.25}
    assert _emit(7, capsys) == {"schema": SCHEMA, "value": 7}


def test_dataclass_with_declared_schema_is_stamped(capsys):
    assert _emit(_Stamped(3), capsys) == {"schema": SCHEMA, "x": 3}


def test_a_payload_stamped_in_python_must_agree_with_the_verb_schema():
    with pytest.raises(ValueError):
        emit_contract_result(
            {"schema": "ftmw/other@1", "a": 1}, schema=SCHEMA, output=None
        )


def test_matching_python_stamp_is_accepted(capsys):
    assert _emit({"schema": SCHEMA, "a": 1}, capsys) == {"schema": SCHEMA, "a": 1}


def test_absent_in_a_named_field_of_a_wrapped_dict(capsys):
    env = _emit({"schema": SCHEMA, "items": Absent.NOT_RUN}, capsys)
    assert env == {"schema": SCHEMA, "items": None, "items_absent": "not_run"}


def test_none_result_is_value_not_run(capsys):
    """``get_final_products`` before Stage 6 returns ``None`` in Python; its
    envelope is ``value`` null plus ``value_absent`` ``not_run``."""
    assert contract_envelope(None, SCHEMA) == {
        "schema": SCHEMA,
        "value": Absent.NOT_RUN,
    }
    env = _emit(None, capsys)
    assert env == {"schema": SCHEMA, "value": None, "value_absent": "not_run"}


def test_arrays_require_output_and_go_to_npy(capsys, tmp_path):
    result = {"samples": np.arange(3.0), "n": 3}
    with pytest.raises(ValueError, match="--output"):
        emit_contract_result(result, schema=SCHEMA, output=None)
    capsys.readouterr()
    out = tmp_path / "npy"
    env = _emit(result, capsys, output=str(out))
    assert env == {"schema": SCHEMA, "samples": "samples.npy", "n": 3}
    np.testing.assert_array_equal(np.load(out / "samples.npy"), np.arange(3.0))


def test_no_arrays_means_no_output_directory(capsys, tmp_path):
    out = tmp_path / "unused"
    _emit({"a": 1}, capsys, output=str(out))
    assert not out.exists()
