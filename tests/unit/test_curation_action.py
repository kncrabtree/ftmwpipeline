"""Pure tests for ``CurationAction``: the curation-file row as typed data.

No fixture: construction, wire form, row form, and the file-parse equivalence
(``actions_from_curation_file(file) == [from_dict(d) ...]``). Each test names
the mutation it catches.
"""

from __future__ import annotations

import pytest

import ftmwpipeline
from ftmwpipeline import BadSettingError, CurationAction
from ftmwpipeline._internal.stage6_impl import (
    actions_from_curation_file,
    curation_source,
    parse_curation_file,
)
from ftmwpipeline.contract import CURATION_ACTION_SCHEMA, MANIFEST

_KEYS = {
    "schema",
    "action",
    "window_id",
    "freq_mhz",
    "peak_uid",
    "candidate_mhz",
    "frame",
}


# ---- construction: every arity rule names its field ------------------------


@pytest.mark.parametrize(
    "kwargs, field",
    [
        # Mutation: drop the unknown-action check -> no refusal.
        ({"action": "frobnicate", "freq_mhz": 1.0}, "action"),
        # Mutation: allow merge/split through.
        ({"action": "merge", "window_id": 1, "freq_mhz": 1.0}, "action"),
        ({"action": "split", "window_id": 1, "freq_mhz": 1.0}, "action"),
        # Mutation: add without a frequency accepted.
        ({"action": "add"}, "freq_mhz"),
        ({"action": "create"}, "freq_mhz"),
        # Mutation: remove with neither / both of freq and uid accepted.
        ({"action": "remove"}, "freq_mhz"),
        ({"action": "remove", "freq_mhz": 1.0, "peak_uid": 3}, "peak_uid"),
        # Mutation: uid allowed outside remove.
        ({"action": "add", "freq_mhz": 1.0, "peak_uid": 3}, "peak_uid"),
        ({"action": "accept", "window_id": 1, "peak_uid": 3}, "peak_uid"),
        # Mutation: candidate allowed outside accept.
        ({"action": "add", "freq_mhz": 1.0, "candidate_mhz": 2.0}, "candidate_mhz"),
        # Mutation: accept allowed a frequency / no window.
        ({"action": "accept", "window_id": 1, "freq_mhz": 1.0}, "freq_mhz"),
        ({"action": "accept"}, "window_id"),
        # Mutation: negative / bool id accepted.
        ({"action": "add", "window_id": -1, "freq_mhz": 1.0}, "window_id"),
        ({"action": "add", "window_id": True, "freq_mhz": 1.0}, "window_id"),
        ({"action": "remove", "peak_uid": -2}, "peak_uid"),
        # Mutation: non-finite frequency accepted.
        ({"action": "add", "freq_mhz": float("nan")}, "freq_mhz"),
        ({"action": "add", "freq_mhz": float("inf")}, "freq_mhz"),
        # Mutation: bad frame accepted.
        ({"action": "add", "freq_mhz": 1.0, "frame": "cal"}, "frame"),
    ],
)
def test_construction_refusal_names_the_field(kwargs, field):
    with pytest.raises(BadSettingError) as ei:
        CurationAction(**kwargs)
    assert ei.value.path == field


def test_valid_shapes_construct():
    # Mutation: an arity check that over-refuses a legal row.
    CurationAction("add", freq_mhz=1.0)
    CurationAction("add", window_id=3, freq_mhz=1.0, frame="calibrated")
    CurationAction("remove", freq_mhz=1.0)
    CurationAction("remove", peak_uid=4)
    CurationAction("accept", window_id=2)
    CurationAction("accept", window_id=2, candidate_mhz=5.0)
    CurationAction("create", freq_mhz=1.0)
    CurationAction("create", window_id=9, freq_mhz=1.0)


def test_action_is_frozen():
    # Mutation: dataclass(frozen=True) dropped.
    a = CurationAction("add", freq_mhz=1.0)
    with pytest.raises(Exception):
        a.freq_mhz = 2.0  # type: ignore[misc]


# ---- wire form -------------------------------------------------------------


def test_to_dict_always_writes_every_key():
    # Mutation: to_dict omits None fields.
    d = CurationAction("remove", peak_uid=7).to_dict()
    assert set(d) == _KEYS
    assert d["schema"] == "ftmw/curation_action@1" == CURATION_ACTION_SCHEMA
    assert d["freq_mhz"] is None and d["window_id"] is None and d["frame"] is None
    assert d["peak_uid"] == 7


def test_dict_round_trip():
    # Mutation: from_dict loses a field (e.g. frame or candidate).
    acts = [
        CurationAction("add", freq_mhz=1.5, frame="calibrated"),
        CurationAction("add", window_id=2, freq_mhz=1.5),
        CurationAction("remove", peak_uid=11),
        CurationAction("accept", window_id=4, candidate_mhz=2.25, frame="raw"),
        CurationAction("create", window_id=8, freq_mhz=3.0),
    ]
    for a in acts:
        assert CurationAction.from_dict(a.to_dict()) == a


def test_from_dict_leniency_and_refusals():
    # Mutation: schema not checked / unknown keys swallowed / action optional.
    assert CurationAction.from_dict({"action": "add", "freq_mhz": 1.0}) == (
        CurationAction("add", freq_mhz=1.0)
    )
    with pytest.raises(BadSettingError) as ei:
        CurationAction.from_dict({"schema": "ftmw/other@1", "action": "accept"})
    assert ei.value.path == "schema"
    with pytest.raises(BadSettingError) as ei:
        CurationAction.from_dict({"action": "add", "freq_mhz": 1.0, "freq": 2.0})
    assert ei.value.path == "freq"
    with pytest.raises(BadSettingError) as ei:
        CurationAction.from_dict({"freq_mhz": 1.0})
    assert ei.value.path == "action"


def test_from_dict_runs_construction_validation():
    # Mutation: from_dict bypasses __post_init__ checks.
    with pytest.raises(BadSettingError) as ei:
        CurationAction.from_dict({"action": "accept"})
    assert ei.value.path == "window_id"


# ---- row form and file equivalence ----------------------------------------


def test_to_row_forms():
    # Mutation: wrong sentinel (auto vs new), uid/candidate spelling.
    assert CurationAction("add", freq_mhz=1.5).to_row() == "add,auto,1.5,"
    assert CurationAction("create", freq_mhz=1.5).to_row() == "create,new,1.5,"
    assert CurationAction("create", window_id=9, freq_mhz=1.5).to_row() == (
        "create,9,1.5,"
    )
    assert CurationAction("remove", peak_uid=17).to_row() == "remove,auto,uid:17,"
    assert CurationAction("accept", window_id=3).to_row() == "accept,3,,"
    assert CurationAction("accept", window_id=3, candidate_mhz=2.5).to_row() == (
        "accept,3,,candidate=2.5"
    )


_FILE = (
    "remove,auto,uid:17809800,\n"
    "add,auto,26880.3,\n"
    "remove,3,26625.0089,\n"
    "add,4,26626.5,\n"
    "accept,10,,\n"
    "accept,11,,candidate=26700.25\n"
    "create,new,26700.0,\n"
    "create,40,26900.0,\n"
)
_ACTIONS = [
    CurationAction("remove", peak_uid=17809800),
    CurationAction("add", freq_mhz=26880.3),
    CurationAction("remove", window_id=3, freq_mhz=26625.0089),
    CurationAction("add", window_id=4, freq_mhz=26626.5),
    CurationAction("accept", window_id=10),
    CurationAction("accept", window_id=11, candidate_mhz=26700.25),
    CurationAction("create", freq_mhz=26700.0),
    CurationAction("create", window_id=40, freq_mhz=26900.0),
]


def test_file_parses_to_the_same_actions_as_dicts(tmp_path):
    # Mutation: action_from_op mishandles the auto/new window sentinel, the
    # uid token or the candidate param -> the equality breaks.
    p = tmp_path / "c.csv"
    p.write_text(_FILE)
    from_file = actions_from_curation_file(p)
    assert from_file == [CurationAction.from_dict(a.to_dict()) for a in _ACTIONS]
    assert from_file == _ACTIONS
    # sanity: the parser really saw every row
    assert len(parse_curation_file(p)) == len(_ACTIONS)


def test_rows_reparse_to_the_same_actions(tmp_path):
    # Mutation: to_row writes something the parser reads differently.
    p = tmp_path / "rows.csv"
    p.write_text("\n".join(a.to_row() for a in _ACTIONS) + "\n")
    assert actions_from_curation_file(p) == _ACTIONS


def test_file_header_frame_rides_on_frequency_actions_only(tmp_path):
    # Mutation: header frame attached to uid removes / bare accepts, or lost.
    p = tmp_path / "h.csv"
    p.write_text("# frame: raw\nremove,auto,uid:5,\nadd,auto,100.5,\naccept,2,,\n")
    uid_rm, add, acc = actions_from_curation_file(p)
    assert uid_rm.frame is None and acc.frame is None
    assert add.frame == "raw"


# ---- exactly-one-of --------------------------------------------------------


def test_curation_source_exactly_one(tmp_path):
    # Mutation: both / neither silently accepted.
    for path, acts in (
        (None, None),
        ("x.csv", [CurationAction("accept", window_id=1)]),
    ):
        with pytest.raises(BadSettingError) as ei:
            curation_source(path, acts)
        assert ei.value.path == "actions"
    assert curation_source("x.csv", None) is not None
    src = curation_source(None, [{"action": "accept", "window_id": 1}])
    assert src is not None


# ---- manifest / version ----------------------------------------------------


def test_manifest_declares_schema_and_exports():
    # Mutation: schema missing from the manifest, or class not exported.
    assert "ftmw/curation_action@1" in MANIFEST.schemas
    assert ftmwpipeline.CurationAction is CurationAction
    assert ftmwpipeline.CONTRACT_VERSION == 6
