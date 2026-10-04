"""The analysis fingerprint's canonical form (CONTRACT_STRATEGY §Analysis
fingerprint, "Canonical form").

``ftmw/analysis_fingerprint@1`` is frozen once published, so the golden test
below pins the exact bytes and digest of a hand-written input. Its expected
bytes were written out by hand from the spec rules, not produced by calling the
encoder. If it fails, the definition of ``@1`` moved: do not edit the literals,
publish ``@2`` instead.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
import pytest

from ftmwpipeline._internal import fingerprint_impl as fp
from ftmwpipeline.contract import Stage
from ftmwpipeline.core.peak_shape import PeakShape
from ftmwpipeline.core.stage_fit_settings import ClockSource, ShapeSpec

# ---------------------------------------------------------------------------
# Golden freeze of @1
# ---------------------------------------------------------------------------

#: Nested groups; floats 0.1 / 1e-05 / 25.0 / -0.0 / nan / inf / -inf; ints;
#: bools; None; a tuple; a ClockSource (label with a quote, a backslash and a
#: newline); a ShapeSpec; a non-ASCII value and key. Keys are deliberately
#: out of order, and mix upper case, ``_``, lower case and a non-ASCII letter so
#: only code-point order gives the expected sequence.
_GOLDEN_INPUT = {
    "z": "µm",
    "é": 2,
    "B": [True, False, None],
    "_": (ClockSource(5760.0, True, 'up"conv\\x\n'),),
    "a": {
        "y": 0.1,
        "x": 1e-05,
        "w": 25.0,
        "v": -0.0,
        "u": float("nan"),
        "t": float("inf"),
        "s": float("-inf"),
        "r": 7,
        "q": (1, 2.5),
    },
    "shape": ShapeSpec(PeakShape.GAUSSIAN),
    "tau": {"stft": {"n_seg": 8, "tau_max_us": None}},
}

#: Written by hand from the spec: keys by code point (B < _ < a < shape < tau
#: < z < e-acute), ``repr`` floats, non-finite as strings, ``-0.0`` kept, the
#: ClockSource as {"freq_mhz", "label", "locked"} (sorted), only ``"``, ``\`` and
#: control characters escaped (newline as ``\u000a``), UTF-8, no whitespace.
_GOLDEN_BYTES = (
    rb'{"B":[true,false,null],'
    rb'"_":[{"freq_mhz":5760.0,"label":"up\"conv\\x\u000a","locked":true}],'
    rb'"a":{"q":[1,2.5],"r":7,"s":"-inf","t":"inf","u":"nan","v":-0.0,'
    rb'"w":25.0,"x":1e-05,"y":0.1},'
    rb'"shape":{"kind":"gaussian"},'
    rb'"tau":{"stft":{"n_seg":8,"tau_max_us":null}},'
    b'"z":"\xc2\xb5m",'
    b'"\xc3\xa9":2}'
)

#: SHA-256 of ``_GOLDEN_BYTES`` (computed once with hashlib on the literal).
_GOLDEN_DIGEST = "6cfbf66ec73d794d1e88b9aba3a4d69f648e8c675ca2e579be0bb79d5b69fa79"


def test_golden_literal_digest_matches_literal_bytes():
    # Guards the test itself: the digest literal is of the byte literal.
    assert hashlib.sha256(_GOLDEN_BYTES).hexdigest() == _GOLDEN_DIGEST


def test_golden_canonical_bytes_are_frozen():
    # Mutation: any change to key order, float spelling, non-finite spelling,
    # escape form, ClockSource/ShapeSpec form or whitespace changes these bytes.
    assert fp.canonical_json(_GOLDEN_INPUT) == _GOLDEN_BYTES


def test_golden_digest_is_frozen():
    # Mutation: hashing anything but SHA-256 of the canonical bytes (another
    # algorithm, uppercase hex, a salted or prefixed input).
    assert fp.digest_of(_GOLDEN_INPUT) == _GOLDEN_DIGEST


def test_golden_input_insertion_order_is_irrelevant():
    # Mutation: encoding keys in insertion order.
    reordered = dict(reversed(list(_GOLDEN_INPUT.items())))
    assert list(reordered) != list(_GOLDEN_INPUT)
    assert fp.canonical_json(reordered) == _GOLDEN_BYTES


# ---------------------------------------------------------------------------
# Encoder rules
# ---------------------------------------------------------------------------
def _enc(value) -> str:
    return fp.canonical_json(value).decode("utf-8")


def test_keys_sort_by_code_point_at_every_level():
    # Mutation: case-insensitive / locale / insertion-order sorting.
    assert _enc({"b": 1, "B": 2, "a": {"z": 1, "Z": 2, "_": 3}}) == (
        '{"B":2,"a":{"Z":2,"_":3,"z":1},"b":1}'
    )
    assert _enc({"é": 1, "z": 2, "10": 3, "9": 4}) == ('{"10":3,"9":4,"z":2,"é":1}')


def test_non_string_keys_raise():
    # Mutation: coercing keys with str() so {1: ..} and {"1": ..} collide.
    with pytest.raises(TypeError):
        fp.canonical_json({1: "a"})
    with pytest.raises(TypeError):
        fp.canonical_json({"a": {2.5: 1}})


def test_booleans_are_never_numbers():
    # Mutation: testing int before bool (True -> 1), or missing np.bool_.
    assert _enc([True, False, 1, 0]) == "[true,false,1,0]"
    assert _enc([np.bool_(True), np.bool_(False)]) == "[true,false]"
    assert _enc({"a": True}) == '{"a":true}'


def test_integers_have_no_decimal_point():
    # Mutation: writing ints through the float path (``7.0``).
    assert _enc([7, -3, 0, 10**20]) == "[7,-3,0,100000000000000000000]"
    assert _enc([np.int64(7), np.int32(-3), np.uint8(255)]) == "[7,-3,255]"


def test_floats_use_the_shortest_round_trip_decimal():
    # Mutation: a fixed-precision format ("%.6g", "%.17g"), dropping the ``.0``
    # of a whole float, or rewriting the exponent form.
    assert _enc([0.1, 1e-05, 25.0, 1e16, 1.5e-07, 123456789.125]) == (
        "[0.1,1e-05,25.0,1e+16,1.5e-07,123456789.125]"
    )
    assert _enc(2.0) == "2.0"
    assert _enc(2) == "2"


def test_float_and_int_of_equal_value_do_not_collide():
    # Mutation: normalising 25.0 to 25 (or the reverse).
    assert fp.digest_of({"a": 25.0}) != fp.digest_of({"a": 25})


def test_negative_zero_is_kept():
    # Mutation: ``x + 0.0`` / ``== 0`` normalisation of -0.0.
    assert _enc(-0.0) == "-0.0"
    assert _enc(0.0) == "0.0"
    assert _enc(np.float64(-0.0)) == "-0.0"
    assert fp.digest_of({"a": -0.0}) != fp.digest_of({"a": 0.0})


def test_non_finite_values_are_strings():
    # Mutation: emitting bare NaN / Infinity (invalid JSON) or ``null``.
    assert _enc([float("nan"), float("inf"), float("-inf")]) == ('["nan","inf","-inf"]')
    assert _enc([np.float64("nan"), np.float32("inf"), np.float64("-inf")]) == (
        '["nan","inf","-inf"]'
    )
    json.loads(_enc([float("nan"), float("inf")]))  # strict-JSON parseable


def test_nan_float_and_nan_string_encode_alike():
    # The spec's own consequence of spelling non-finite values as strings.
    assert fp.canonical_json(float("nan")) == fp.canonical_json("nan")


def test_numpy_floats_encode_as_the_python_float_of_the_same_value():
    # Mutation: ``repr(np.float64)`` (``np.float64(0.1)`` on numpy 2) or
    # ``str`` of a float32 in its own, shorter precision.
    assert _enc(np.float64(0.1)) == "0.1"
    assert _enc(np.float32(0.5)) == "0.5"
    f32 = np.float32(0.1)
    assert _enc(f32) == repr(float(f32))
    assert _enc(f32) != "0.1"


def test_sequences_are_arrays_and_none_is_null():
    # Mutation: tuples treated as unsupported or encoded as strings.
    assert _enc((1, (2, 3), [4])) == "[1,[2,3],[4]]"
    assert _enc({"a": None, "b": []}) == '{"a":null,"b":[]}'
    assert _enc(()) == "[]"


def test_no_insignificant_whitespace_and_utf8():
    # Mutation: json.dumps default separators; ASCII-escaping non-ASCII text.
    out = fp.canonical_json({"a": [1, 2], "b": {"c": "µ"}})
    assert out == b'{"a":[1,2],"b":{"c":"\xc2\xb5"}}'
    assert b" " not in out and b"\n" not in out


def test_strings_escape_only_quote_backslash_and_controls():
    # Mutation: escaping "/" or non-ASCII, or short escapes (\n) for controls.
    assert _enc('a"b') == r'"a\"b"'
    assert _enc("a\\b") == r'"a\\b"'
    assert _enc("a\nb\tc\x00\x1f") == r'"a\u000ab\u0009c\u0000\u001f"'
    assert _enc("a/b µ \x7f") == '"a/b µ \x7f"'


def test_structured_values_use_their_typed_json_form():
    # Mutation: encoding ShapeSpec as its repr / enum, or ClockSource without
    # its label.
    assert _enc(ShapeSpec(PeakShape.LORENTZIAN)) == '{"kind":"lorentzian"}'
    assert _enc(ClockSource(5120.0, False, "dc")) == (
        '{"freq_mhz":5120.0,"label":"dc","locked":false}'
    )
    assert _enc(ClockSource(5120.0, False, "dc")) != _enc(
        ClockSource(5120.0, False, "")
    )


@pytest.mark.parametrize(
    "bad",
    [{1, 2}, b"bytes", pathlib.Path("x"), object(), complex(1, 2)],
    ids=["set", "bytes", "path", "object", "complex"],
)
def test_unsupported_values_raise_instead_of_stringifying(bad):
    # Mutation: a ``default=str`` style fallback.
    with pytest.raises(TypeError):
        fp.canonical_json({"a": bad})


def test_plain_enum_is_not_silently_encoded():
    # The readers convert enums to their value; the encoder must not guess.
    from enum import Enum

    class E(Enum):
        A = "a"

    with pytest.raises(TypeError):
        fp.canonical_json({"a": E.A})


def test_digest_is_lowercase_hex_sha256_of_canonical_bytes():
    obj = {"b": [1, 2.5], "a": None}
    d = fp.digest_of(obj)
    assert d == hashlib.sha256(fp.canonical_json(obj)).hexdigest()
    assert len(d) == 64 and d == d.lower()
    assert all(c in "0123456789abcdef" for c in d)


def test_canonical_json_is_deterministic():
    obj = {"a": [0.1, {"b": float("nan")}], "c": (1, 2)}
    assert fp.canonical_json(obj) == fp.canonical_json(obj)


# ---------------------------------------------------------------------------
# The coverage map
# ---------------------------------------------------------------------------
def test_every_canonical_stage_is_read_and_ordered_as_the_spec_names_them():
    # Spec: "Every stage key is always present" -- a stage missing from the map
    # or from the stage list would silently drop out of the digest.
    assert [s.value for s in fp.CANONICAL_STAGES] == [
        "data",
        "ft",
        "noise",
        "tau",
        "tau_g",
        "timebase",
        "peaks",
        "windows",
        "fit",
        "review",
    ]
    assert set(fp.CANONICAL_STAGES) == set(Stage)
    assert set(fp._STAGE_RECORDS) == set(Stage)


def test_only_consumers_carry_a_consumed_block():
    # Spec: consumed blocks under ``<stage>.consumed`` for Stage 3 and Stage 5.
    with_consumed = {
        stage
        for stage, records in fp._STAGE_RECORDS.items()
        if any(r.sub_key == "consumed" for r in records)
    }
    assert with_consumed == {Stage.PEAKS, Stage.FIT}


def test_the_two_decay_calibrations_read_distinct_producer_records():
    # Spec: tau and tau_g come from each twin's own producer record.
    tau_keys = set(fp._STAGE_RECORDS[Stage.TAU][0].keys)
    gauss_keys = set(fp._STAGE_RECORDS[Stage.TAU_G][0].keys)
    assert "gaussian" in gauss_keys and "gaussian" not in tau_keys
    assert tau_keys & gauss_keys >= {"stft", "aggregation", "band"}


def test_only_review_may_be_absent_with_a_value():
    # Spec: only ``review.sigma_floor_khz`` has a defined value for an absent
    # record (0.0). Every other absent record is incomplete provenance.
    with_default = {
        stage: r.absent_values
        for stage, records in fp._STAGE_RECORDS.items()
        for r in records
        if r.absent_values is not None
    }
    assert with_default == {Stage.REVIEW: {"sigma_floor_khz": 0.0}}


def test_shape_recommendation_record_is_not_in_the_map():
    # Spec: the recommendation is not hashed itself; its verdict is covered by
    # peaks.consumed and fit.shape.
    names = {
        getattr(r.reader, "__name__", "")
        for records in fp._STAGE_RECORDS.values()
        for r in records
    }
    assert not any("recommend" in n for n in names)
    assert "shape" in fp._STAGE_RECORDS[Stage.FIT][0].keys


# ---------------------------------------------------------------------------
# The manifest declares the accessor
# ---------------------------------------------------------------------------
def test_manifest_declares_the_accessor_and_its_schema():
    # Mutation: an accessor without a manifest entry, a schema name that is not
    # published, or a different Pipeline method name.
    import ftmwpipeline
    import ftmwpipeline.api as api
    from ftmwpipeline import MANIFEST, Pipeline

    assert "analysis_fingerprint" in MANIFEST.accessors
    assert MANIFEST.file_bound["analysis_fingerprint"] is True
    assert MANIFEST.pipeline_names["analysis_fingerprint"] == "analysis_fingerprint"
    assert "ftmw/analysis_fingerprint@1" in MANIFEST.schemas
    assert callable(api.analysis_fingerprint)
    assert callable(Pipeline.analysis_fingerprint)
    assert ftmwpipeline.CONTRACT_VERSION >= 5
