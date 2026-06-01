"""Unit tests for the tuning knob registry."""

import pytest

from ftmwpipeline._internal.tuning import get_knob, list_knobs
from ftmwpipeline._internal.tuning.registry import KnobSpec


def test_registry_non_empty():
    assert len(list_knobs()) > 0


def test_every_knob_is_well_formed():
    for spec in list_knobs():
        assert isinstance(spec, KnobSpec)
        # dotted path
        assert "." in spec.path, spec.path
        # non-empty grid + metric columns
        assert len(spec.default_grid) > 0, spec.path
        assert len(spec.metric_columns) > 0, spec.path
        # callables present
        assert callable(spec.run), spec.path
        assert callable(spec.metric), spec.path
        # rating is one of the three
        assert spec.inst_sensitivity in ("Y", "N", "maybe"), spec.path
        # if a recommender direction is set, it points at a real metric column
        if spec.direction in ("min", "max"):
            assert spec.primary_metric in spec.metric_columns, spec.path


def test_get_knob_unknown_raises_with_hint():
    with pytest.raises(KeyError) as exc:
        get_knob("stageX.nope.not_a_knob")
    # the message lists the registered knobs
    assert "registered knobs" in str(exc.value)


def test_list_knobs_filters_by_stage():
    start = list_knobs("start_detection")
    assert start
    assert all(s.stage == "start_detection" for s in start)
    # a bogus stage yields nothing
    assert list_knobs("no_such_stage") == ()


def test_list_knobs_is_path_sorted():
    paths = [s.path for s in list_knobs()]
    assert paths == sorted(paths)
