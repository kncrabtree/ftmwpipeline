"""Curation syntax refusals are ``bad_setting`` naming the cell the caller wrote.

CONTRACT_STRATEGY, Errors: ``bad_setting.path`` names what the caller wrote --
an argument by its name, a curation-file cell as ``curation[line <n>].<column>``
and a field of the i-th ``CurationAction`` as ``actions[<i>].<field>``. Every
refusal is still the ``ValueError`` it replaced, with its historical message.

These are pure (no built file): the parser, ``parse_peak_token`` and
``curation_source`` read nothing from a pipeline file.

Mutation: reverting a site to a bare ``ValueError`` fails the type check; a
wrong ``line`` / ``column`` fails the ``path`` check; the offending text is the
``value``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ftmwpipeline import CurationAction
from ftmwpipeline._internal.stage6_impl import curation_source, parse_curation_file
from ftmwpipeline.core.curation import PeakUidToken, parse_peak_token
from ftmwpipeline.file_manager import BadSettingError, PipelineFileError

pytestmark = [pytest.mark.unit]


def _write(tmp_path: Path, text: str) -> str:
    p = tmp_path / "curation.csv"
    p.write_text(text)
    return str(p)


def _check(exc: pytest.ExceptionInfo, path: str, value: object) -> BadSettingError:
    err = exc.value
    assert isinstance(err, BadSettingError)
    assert isinstance(err, (ValueError, PipelineFileError))
    assert err.path == path
    assert err.value == value
    d = err.to_dict()
    assert d["code"] == "bad_setting" and d["path"] == path
    assert err.expected  # says what would have been accepted
    return err


# ---------------------------------------------------------------------------
# Curation-file rows: curation[line <n>].<column>
# ---------------------------------------------------------------------------

_ROW_CASES = [
    # action column
    ("add,5,100.0,\nbogus,6,1.0,\n", "curation[line 2].action", "bogus"),
    ("# note\n\nsplit,24,38449.9,into=3\n", "curation[line 3].action", "split"),
    ("merge,438,38450.10;38450.40,\n", "curation[line 1].action", "merge"),
    # window column
    ("accept,,,\n", "curation[line 1].window", ""),
    ("accept\n", "curation[line 1].window", None),
    ("add,five,100.0,\n", "curation[line 1].window", "five"),
    ("accept,5.5,,\n", "curation[line 1].window", "5.5"),
    # freqs column
    ("add,5,abc,\n", "curation[line 1].freqs", "abc"),
    ("add,5,uid:3,\n", "curation[line 1].freqs", "uid:3"),
    ("add,5,100.0;101.0,\n", "curation[line 1].freqs", "100.0;101.0"),
    ("add,5,,\n", "curation[line 1].freqs", ""),
    ("remove,12,,\n", "curation[line 1].freqs", ""),
    ("remove,12,uid:15;27549.3,\n", "curation[line 1].freqs", "uid:15;27549.3"),
    ("create,new,,\n", "curation[line 1].freqs", ""),
    ("create,new,1.0;2.0,\n", "curation[line 1].freqs", "1.0;2.0"),
    ("accept,5,100.0,\n", "curation[line 1].freqs", "100.0"),
    # a remove row's token goes through parse_peak_token with the cell's path
    ("remove,12,uid:abc,\n", "curation[line 1].freqs", "uid:abc"),
    ("remove,12,uid:,\n", "curation[line 1].freqs", "uid:"),
    ("remove,12,uid:-3,\n", "curation[line 1].freqs", "uid:-3"),
    ("remove,12,abc,\n", "curation[line 1].freqs", "abc"),
    # params column
    ("add,5,100.0,into\n", "curation[line 1].params", "into"),
    ("add,5,100.0,a=1\n", "curation[line 1].params", "a=1"),
    ("remove,5,100.0,a=1\n", "curation[line 1].params", "a=1"),
    ("create,new,100.0,a=1\n", "curation[line 1].params", "a=1"),
    ("accept,5,,candidate=abc\n", "curation[line 1].params", "candidate=abc"),
    # the line number counts comments, blanks and the header row
    (
        "# a comment\naction,window,freqs,params\n\nadd,5,100.0,\nadd,5,abc,\n",
        "curation[line 5].freqs",
        "abc",
    ),
]


@pytest.mark.parametrize("text, path, value", _ROW_CASES)
def test_row_refusal_names_the_cell(tmp_path, text, path, value):
    with pytest.raises(ValueError) as exc:
        parse_curation_file(_write(tmp_path, text))
    _check(exc, path, value)


def test_row_refusal_keeps_the_historical_message(tmp_path):
    """The message is unchanged by the typing (callers match on it)."""
    with pytest.raises(BadSettingError, match="curation line 2: unknown action"):
        parse_curation_file(_write(tmp_path, "add,5,100.0,\nbogus,6,1.0,\n"))
    with pytest.raises(BadSettingError, match="curation line 1: window id 'five'"):
        parse_curation_file(_write(tmp_path, "add,five,100.0,\n"))
    with pytest.raises(BadSettingError, match="curation line 1: malformed peak"):
        parse_curation_file(_write(tmp_path, "remove,12,uid:abc,\n"))


def test_a_valid_file_still_parses(tmp_path):
    ops = parse_curation_file(
        _write(
            tmp_path,
            "# frame: raw\n"
            "action,window,freqs,params\n"
            "add,5,100.0,\n"
            "remove,5,uid:7,\n"
            "accept,5,,candidate=101.0\n",
        )
    )
    assert [o.action for o in ops] == ["add", "remove", "accept"]
    assert ops[1].freqs == [PeakUidToken(7)]


# ---------------------------------------------------------------------------
# Header directives: the cells ``frame`` / ``epsilon`` of their own line
# ---------------------------------------------------------------------------

_HEADER_CASES = [
    # a bad value
    ("# frame: sideways\nadd,5,100.0,\n", "curation[line 1].frame", "sideways"),
    ("# note\n# frame: Sideways\nadd,5,100.0,\n", "curation[line 2].frame", "Sideways"),
    (
        "# frame: calibrated\n# epsilon: abc\nadd,5,100.0,\n",
        "curation[line 2].epsilon",
        "abc",
    ),
    # a conflict with what the file already declared
    (
        "# frame: raw\n# frame: calibrated\nadd,5,100.0,\n",
        "curation[line 2].frame",
        "calibrated",
    ),
    (
        "# frame: calibrated\n# epsilon: 1e-06\n\n# epsilon: 2e-06\nadd,5,100.0,\n",
        "curation[line 4].epsilon",
        2e-06,
    ),
    # end-of-file rules: calibrated without epsilon points at the frame line ...
    (
        "# note\n# frame: calibrated\nadd,5,100.0,\n",
        "curation[line 2].frame",
        "calibrated",
    ),
    # ... and an epsilon without calibrated points at the epsilon line
    ("# epsilon: 2.2e-6\nadd,5,100.0,\n", "curation[line 1].epsilon", 2.2e-6),
    (
        "add,5,100.0,\n# frame: raw\n# epsilon: 2.2e-6\n",
        "curation[line 3].epsilon",
        2.2e-6,
    ),
]


@pytest.mark.parametrize("text, path, value", _HEADER_CASES)
def test_header_refusal_names_the_directive(tmp_path, text, path, value):
    with pytest.raises(ValueError) as exc:
        parse_curation_file(_write(tmp_path, text))
    _check(exc, path, value)


def test_header_refusal_keeps_the_historical_message(tmp_path):
    with pytest.raises(BadSettingError, match="requires an 'epsilon' header"):
        parse_curation_file(_write(tmp_path, "# frame: calibrated\nadd,5,100.0,\n"))
    with pytest.raises(BadSettingError, match="requires a 'frame: calibrated'"):
        parse_curation_file(_write(tmp_path, "# epsilon: 2.2e-6\nadd,5,100.0,\n"))


def test_header_lines_are_recorded_on_the_parsed_header(tmp_path):
    ops = parse_curation_file(
        _write(
            tmp_path,
            "# a note\n# frame: calibrated\n\n# epsilon: 2.2e-6\nadd,5,100.0,\n",
        )
    )
    assert ops.header.frame == "calibrated"
    assert ops.header.frame_line == 2
    assert ops.header.epsilon_line == 4


# ---------------------------------------------------------------------------
# parse_peak_token(path=...)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("token", ["abc", "uid:", "uid:abc", "uid:-3", "uid:1.5"])
def test_parse_peak_token_defaults_to_the_peak_path(token):
    with pytest.raises(ValueError) as exc:
        parse_peak_token(token)
    _check(exc, "peak", token)


@pytest.mark.parametrize("path", ["add", "remove", "curation[line 4].freqs"])
@pytest.mark.parametrize("token", ["abc", "uid:", "uid:abc", "uid:-3", "uid:1.5"])
def test_parse_peak_token_reports_the_callers_path(token, path):
    with pytest.raises(ValueError) as exc:
        parse_peak_token(token, path=path)
    err = _check(exc, path, token)
    assert repr(token) in str(err)  # the message still names the token


def test_parse_peak_token_valid_tokens_ignore_the_path():
    assert parse_peak_token("27549.3259", path="remove") == 27549.3259
    assert parse_peak_token("uid:12", path="remove") == PeakUidToken(12)


# ---------------------------------------------------------------------------
# actions=[...]: actions[<i>].<field>
# ---------------------------------------------------------------------------

_GOOD = CurationAction("add", window_id=3, freq_mhz=100.0, frame="raw")


@pytest.mark.parametrize(
    "bad, field, value",
    [
        # an unknown key is the path's field
        ({"action": "add", "freq_mhz": 1.0, "freq": 2.0}, "freq", None),
        ({"action": "frobnicate", "freq_mhz": 1.0}, "action", "frobnicate"),
        ({"action": "add"}, "freq_mhz", None),
        ({"action": "remove", "freq_mhz": 1.0, "peak_uid": 3}, "peak_uid", None),
        ({"action": "add", "freq_mhz": 1.0, "frame": "cal"}, "frame", "cal"),
        ({"schema": "ftmw/other@1", "action": "accept"}, "schema", None),
    ],
)
@pytest.mark.parametrize("index", [0, 2])
def test_refused_action_dict_is_pathed_by_its_index(bad, field, value, index):
    items = [_GOOD, _GOOD][:index] + [bad]
    with pytest.raises(ValueError) as exc:
        curation_source(None, items)
    err = exc.value
    assert isinstance(err, BadSettingError)
    assert err.path == f"actions[{index}].{field}"
    assert err.to_dict()["path"] == f"actions[{index}].{field}"
    # the action is named in the message
    assert str(err).startswith(f"actions[{index}]: ")
    if value is not None:
        assert err.value == value


@pytest.mark.parametrize("item", [7, "add", None, 3.5, ["add"]])
def test_an_item_that_is_not_an_action_is_pathed_by_its_index(item):
    with pytest.raises(ValueError) as exc:
        curation_source(None, [_GOOD, item])
    err = _check(exc, "actions[1]", item)
    assert "actions[1] is not a CurationAction" in str(err)


def test_good_actions_and_dicts_pass_through():
    out = curation_source(
        None,
        [
            _GOOD,
            {"action": "accept", "window_id": 2},
            CurationAction("add", freq_mhz=1.0),
        ],
    )
    assert len(out) == 3 and out[0] is _GOOD
    assert out[1] == CurationAction("accept", window_id=2)


def test_the_both_or_neither_refusal_keeps_the_actions_path():
    with pytest.raises(BadSettingError) as ei:
        curation_source(None, None)
    assert ei.value.path == "actions"
    with pytest.raises(BadSettingError) as ei:
        curation_source("a.csv", [_GOOD])
    assert ei.value.path == "actions"
