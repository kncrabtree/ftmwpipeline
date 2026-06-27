"""Tests for on-plot attention markers: location capture, serialization, and the
SVG overlay builder."""

from __future__ import annotations

import h5py

from ftmwpipeline._internal.report_html_impl import (
    _ATTENTION_MARKERS,
    _attention_marker_svg,
)
from ftmwpipeline.core.data_structures import (
    AttentionReason,
    Stage6Review,
    WindowReviewStatus,
)
from ftmwpipeline.io.stage6_review_serialization import (
    load_stage6_review_from_hdf5,
    save_stage6_review_to_hdf5,
)

# A simple geometry: data box x in [100, 700] px maps molecular 1000 -> 1006 MHz.
_GEOM = {
    "x0": 100.0,
    "x1": 700.0,
    "y0": 20.0,
    "y1": 480.0,
    "w": 800.0,
    "h": 500.0,
    "flo": 1000.0,
    "fhi": 1006.0,
}


def test_marker_maps_frequency_to_axis_position():
    # Midband frequency lands at the box center; carries the marker letter + title.
    svg = _attention_marker_svg(_GEOM, [(1003.0, "candidate_bearing", "missed line")])
    assert 'class="attn-svg"' in svg
    assert 'viewBox="0 0 800 500"' in svg
    assert "attn-candidate_bearing" in svg
    assert ">C</text>" in svg
    assert "<title>missed line</title>" in svg
    # Caret centered at x = 100 + 0.5*(700-100) = 400.
    assert "L400.0," in svg


def test_marker_skips_out_of_band_and_unmarked_kinds():
    # Frequency outside [flo, fhi] is dropped; a window-wide kind has no glyph.
    assert _attention_marker_svg(_GEOM, [(1100.0, "candidate_bearing", "x")]) == ""
    assert _attention_marker_svg(_GEOM, [(1003.0, "worst_eps", "x")]) == ""


def test_marker_kinds_have_distinct_letters():
    # candidate / spur / merged each render once with their own letter.
    markers = [
        (1001.0, "candidate_bearing", "c"),
        (1003.0, "spur_adjacent", "s"),
        (1005.0, "auto_merged_review", "m"),
    ]
    svg = _attention_marker_svg(_GEOM, markers)
    for kind, letter in (
        ("candidate_bearing", "C"),
        ("spur_adjacent", "S"),
        ("auto_merged_review", "M"),
    ):
        assert _ATTENTION_MARKERS[kind] == letter
        assert f">{letter}</text>" in svg


def test_locations_round_trip(tmp_path):
    rev = Stage6Review(
        window_statuses={
            5: WindowReviewStatus(
                window_id=5,
                attention_reasons=[
                    AttentionReason(
                        kind="candidate_bearing",
                        detail="x",
                        severity=2.0,
                        locations=[100.25, 200.5],
                    ),
                    AttentionReason(kind="worst_eps", detail="y", severity=1.0),
                ],
            )
        }
    )
    fp = tmp_path / "rev.h5"
    with h5py.File(fp, "w") as f:
        save_stage6_review_to_hdf5(rev, f.create_group("stage6_review"))
    with h5py.File(fp, "r") as f:
        rt = load_stage6_review_from_hdf5(f["stage6_review"])

    reasons = rt.window_statuses[5].attention_reasons
    assert reasons[0].locations == [100.25, 200.5]
    assert reasons[1].locations == []  # empty default preserved
