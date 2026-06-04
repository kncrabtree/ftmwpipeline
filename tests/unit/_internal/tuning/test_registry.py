"""Unit tests for the tuning knob registry."""

import pytest

from ftmwpipeline._internal.tuning import get_knob, list_knobs
from ftmwpipeline._internal.tuning.registry import KnobSpec


def test_registry_non_empty():
    assert len(list_knobs()) > 0


def test_every_knob_is_well_formed():
    for spec in list_knobs(include_advanced=True):
        assert isinstance(spec, KnobSpec)
        # dotted path
        assert "." in spec.path, spec.path
        # non-empty grid + metric columns
        assert len(spec.default_grid) > 0, spec.path
        assert len(spec.metric_columns) > 0, spec.path
        # callables present
        assert callable(spec.run), spec.path
        assert callable(spec.metric), spec.path
        # rating + tier are from their allowed sets
        assert spec.inst_sensitivity in ("Y", "N", "maybe"), spec.path
        assert spec.tier in ("primary", "advanced"), spec.path
        # if a recommender direction is set, it points at a real metric column
        if spec.direction in ("min", "max"):
            assert spec.primary_metric in spec.metric_columns, spec.path


def test_tier_filtering():
    primary = list_knobs()
    everything = list_knobs(include_advanced=True)
    # the default view is a strict, smaller subset that hides advanced knobs
    assert {s.path for s in primary} <= {s.path for s in everything}
    assert len(primary) < len(everything)
    assert all(s.tier == "primary" for s in primary)
    # at least one advanced knob exists and is hidden by default
    assert any(s.tier == "advanced" for s in everything)


def test_stage1_exposes_band_and_window_not_apodization():
    paths = {s.path for s in list_knobs("stage1", include_advanced=True)}
    assert {"stage1.start_us", "stage1.trim_min_mhz", "stage1.trim_max_mhz",
            "stage1.end_us"} <= paths
    # zpf / expf_us / window_function / units are deliberately not swept
    for excluded in ("zpf", "expf", "window_function", "units"):
        assert not any(excluded in p for p in paths), excluded


def test_selector_matches_path_prefix():
    gauss = list_knobs("stage2b.gaussian", include_advanced=True)
    assert gauss
    assert all(s.path.startswith("stage2b.gaussian.") for s in gauss)
    # a broader prefix is a superset of the narrower one
    all_2b = {s.path for s in list_knobs("stage2b", include_advanced=True)}
    assert {s.path for s in gauss} <= all_2b
    # the shape-recommendation knobs route to the shape metric, not the tau one
    rec = list_knobs("stage2b.recommendation", include_advanced=True)
    assert rec
    assert all("recommended_shape" in s.metric_columns for s in rec)


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


def test_stage3_paths_resolve_to_settings_fields():
    from dataclasses import fields

    from ftmwpipeline.core import peak_detection_settings as pds

    tmpl = pds.PeakDetectionSettings()
    stage3 = list_knobs("stage3", include_advanced=True)
    assert stage3
    for spec in stage3:
        assert spec.stage == "stage3_peaks"
        assert spec.requires == "stage2_noise_result"
        _, sub, field = spec.path.split(".")
        names = {f.name for f in fields(getattr(tmpl, sub))}
        assert field in names, spec.path
        assert spec.metric_columns == (
            "n_total", "n_strong", "n_medium", "n_weak",
            "snr_min", "snr_p10", "snr_p25", "snr_p50", "snr_p90", "snr_max",
        )


def test_stage3_leakage_floor_supersedes_hard_mask():
    # the gap-pass hard S_coh mask knob was replaced by the continuous floor;
    # both leakage-floor knobs are registered and the old knob is gone.
    paths = {s.path for s in list_knobs("stage3", include_advanced=True)}
    assert "stage3.gap_pass.gap_leakage_floor_k" in paths
    assert "stage3.primary_pass.primary_leakage_floor_k" in paths
    assert not any("gap_mask_edge" in p for p in paths)
