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
    assert {
        "stage1.start_us",
        "stage1.trim_min_mhz",
        "stage1.trim_max_mhz",
        "stage1.end_us",
    } <= paths
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
            "n_total",
            "n_strong",
            "n_medium",
            "n_weak",
            "snr_min",
            "snr_p10",
            "snr_p25",
            "snr_p50",
            "snr_p90",
            "snr_max",
        )


def test_stage3_leakage_floor_supersedes_hard_mask():
    # the gap-pass hard S_coh mask knob was replaced by the continuous floor;
    # both leakage-floor knobs are registered and the old knob is gone.
    paths = {s.path for s in list_knobs("stage3", include_advanced=True)}
    assert "stage3.gap_pass.gap_leakage_floor_k" in paths
    assert "stage3.primary_pass.primary_leakage_floor_k" in paths
    assert not any("gap_mask_edge" in p for p in paths)


def test_stage4_paths_resolve_to_settings_fields():
    from dataclasses import fields

    from ftmwpipeline.core import window_planning_settings as wps

    tmpl = wps.WindowPlanningSettings()
    stage4 = list_knobs("stage4", include_advanced=True)
    assert stage4
    for spec in stage4:
        assert spec.stage == "stage4_windows"
        assert spec.requires == "stage3_peaks"
        _, sub, field = spec.path.split(".")
        names = {f.name for f in fields(getattr(tmpl, sub))}
        assert field in names, spec.path
        assert spec.metric_columns == (
            "n_windows",
            "n_hard",
            "n_easy",
            "n_free",
            "n_fixed",
            "n_dep",
            "n_split",
            "width_p50",
            "width_p95",
            "width_max",
        )


def test_stage4_leakage_tau_grid_includes_boxcar():
    # the boxcar (undamped) limit is tau_us=None — it must be a swept grid value,
    # not just reachable via settings=.
    spec = get_knob("stage4.leakage.tau_us")
    assert None in spec.default_grid


def test_stage5_paths_resolve_to_settings_fields():
    from dataclasses import fields

    from ftmwpipeline.core import stage_fit_settings as sfs

    tmpl = sfs.StageFitSettings()
    # the fit-quality family (tau / seeder / conservative / penalties / baseline)
    # shares one metric/plot; rescue / spur / thaw are their own families.
    fit_quality_cols = (
        "eps_p50",
        "eps_p95",
        "n_fail",
        "n_peaks",
        "n_free_tau",
        "sigma_f_khz",
        "chi2r_p50",
        "chi2r_p95",
    )
    family_subs = {"rescue", "spur", "thaw"}
    stage5 = list_knobs("stage5", include_advanced=True)
    assert stage5
    for spec in stage5:
        assert spec.stage == "stage5_fitting"
        assert spec.requires == "stage4_windows"
        # every fit knob carries the window-reduction prepare hook
        assert spec.prepare is not None
        _, sub, field = spec.path.split(".")
        names = {f.name for f in fields(getattr(tmpl, sub))}
        assert field in names, spec.path
        if sub not in family_subs:
            assert spec.metric_columns == fit_quality_cols, spec.path


def test_stage5_rescue_spur_thaw_families_wired():
    from ftmwpipeline._internal.tuning.registry import (
        _RESCUE_COLS,
        _SPUR_COLS,
        _THAW_COLS,
    )

    cases = {
        "stage5.rescue.snr_threshold": ("plot_rescue", _RESCUE_COLS),
        "stage5.spur.integer_tol_mhz": ("plot_spur", _SPUR_COLS),
        "stage5.thaw.residual_edge_threshold": ("plot_thaw", _THAW_COLS),
    }
    for path, (plot_name, cols) in cases.items():
        spec = get_knob(path)
        assert spec.plot is not None and spec.plot.__name__ == plot_name
        assert spec.metric_columns == cols
        # the families reuse the fit runner + window-reduction prepare hook
        assert spec.prepare is not None

    # Y-rated primaries surfaced; N-rated knobs demoted to advanced.
    primary = {s.path for s in list_knobs("stage5")}
    assert "stage5.rescue.snr_threshold" in primary
    assert "stage5.rescue.prominence_threshold" in primary
    assert "stage5.spur.narrowness_ratio" in primary
    assert "stage5.spur.mask_half_width_bins" in primary
    assert "stage5.thaw.residual_edge_threshold" in primary
    assert "stage5.rescue.max_rounds" not in primary
    assert "stage5.spur.enabled" not in primary
    assert "stage5.thaw.max_thaw_rounds" not in primary


def test_stage5_snr_threshold_knobs_hinted():
    # the two window-SNR-gated knobs carry the straddle-sampling hint; others not
    assert get_knob("stage5.tau.fit_tau_min_snr").select_hint == "snr_threshold"
    assert (
        get_knob("stage5.conservative.weak_window_snr_threshold").select_hint
        == "snr_threshold"
    )
    assert get_knob("stage5.baseline.edge_threshold").select_hint is None


def test_stage4_primary_tier_covers_y_rated_knobs():
    primary = {s.path for s in list_knobs("stage4")}  # default = primary only
    # leakage.tau_us is Y-rated but demoted to advanced: the boxcar default only
    # widens windows (split proposals absorb it), so it is a low-leverage control
    # whose fate is deferred to the cross-fixture audit (issue #6).
    assert primary == {
        "stage4.coherence.edge_threshold",
        "stage4.clustering.max_window_width_mhz",
        "stage4.contributor.magnitude_attachment_threshold",
        "stage4.contributor.min_freeze_snr",
    }
    assert get_knob("stage4.leakage.tau_us").tier == "advanced"
