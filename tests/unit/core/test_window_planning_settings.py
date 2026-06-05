"""
Pure unit tests for :mod:`ftmwpipeline.core.window_planning_settings`.

Mirrors :mod:`tests.unit.core.test_peak_detection_settings`,
:mod:`tests.unit.core.test_noise_settings`, and
:mod:`tests.unit.core.test_tau_calibration_settings`: empty dataclass,
resolution-chain precedence per layer, sub-dataclass independence,
attrs round-trip, YAML I/O, unknown-key rejection.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from ftmwpipeline.core.window_planning_settings import (
    _HARD_DEFAULTS,
    ClusteringSubSettings,
    CoherenceSubSettings,
    ContributorSubSettings,
    LeakageSubSettings,
    WindowPlanningSettings,
    from_attrs,
    from_yaml,
    from_yaml_dict,
    resolve,
    to_attrs,
    to_yaml,
    to_yaml_dict,
)

_SUB_NAMES = ("coherence", "clustering", "contributor", "leakage")


# ---------------------------------------------------------------------------
# Field-level defaults
# ---------------------------------------------------------------------------
class TestEmptyDataclass:
    def test_all_subdataclass_fields_default_to_none(self) -> None:
        s = WindowPlanningSettings()
        for sub_name in _SUB_NAMES:
            sub = getattr(s, sub_name)
            for f in fields(sub):
                assert (
                    getattr(sub, f.name) is None
                ), f"{sub_name}.{f.name} should default to None"

    def test_is_empty(self) -> None:
        assert WindowPlanningSettings().is_empty()
        s = WindowPlanningSettings()
        s.coherence.edge_m = 32
        assert not s.is_empty()
        s2 = WindowPlanningSettings()
        s2.leakage.tau_us = 5.0
        assert not s2.is_empty()


# ---------------------------------------------------------------------------
# Resolution chain
# ---------------------------------------------------------------------------
class TestResolve:
    def test_hard_defaults_when_all_layers_empty(self) -> None:
        merged = resolve()
        assert merged.coherence.edge_m == 64
        assert merged.coherence.trim_m == 32
        assert merged.coherence.edge_threshold == 8.0
        assert merged.clustering.max_window_width_mhz == 40.0
        assert merged.clustering.min_window_half_width_mhz == 2.0
        assert merged.contributor.min_freeze_snr == 50.0
        assert merged.contributor.magnitude_attachment_threshold == 0.1
        # leakage.tau_us legitimately stays None (boxcar / undamped limit).
        assert merged.leakage.tau_us is None

    def test_explicit_beats_preset(self) -> None:
        explicit = WindowPlanningSettings()
        explicit.coherence.edge_m = 128
        preset = WindowPlanningSettings()
        preset.coherence.edge_m = 32
        merged = resolve(explicit=explicit, preset=preset)
        assert merged.coherence.edge_m == 128

    def test_preset_beats_persisted(self) -> None:
        preset = WindowPlanningSettings()
        preset.clustering.max_window_width_mhz = 25.0
        persisted = WindowPlanningSettings()
        persisted.clustering.max_window_width_mhz = 60.0
        merged = resolve(preset=preset, persisted=persisted)
        assert merged.clustering.max_window_width_mhz == 25.0

    def test_persisted_beats_recommended(self) -> None:
        persisted = WindowPlanningSettings()
        persisted.contributor.min_freeze_snr = 40.0
        recommended = WindowPlanningSettings()
        recommended.contributor.min_freeze_snr = 80.0
        merged = resolve(persisted=persisted, recommended=recommended)
        assert merged.contributor.min_freeze_snr == 40.0

    def test_recommended_beats_hard_default(self) -> None:
        recommended = WindowPlanningSettings()
        recommended.contributor.min_freeze_snr = 80.0
        merged = resolve(recommended=recommended)
        assert merged.contributor.min_freeze_snr == 80.0

    def test_recommended_layer_can_populate_tau_us(self) -> None:
        """Confirms the resolver slot is wired even though the hard default is None."""
        recommended = WindowPlanningSettings()
        recommended.leakage.tau_us = 5.0
        merged = resolve(recommended=recommended)
        assert merged.leakage.tau_us == 5.0

    def test_full_precedence_chain(self) -> None:
        explicit = WindowPlanningSettings()
        explicit.coherence.edge_m = 128
        preset = WindowPlanningSettings()
        preset.coherence.edge_m = 32
        preset.clustering.max_window_width_mhz = 25.0
        persisted = WindowPlanningSettings()
        persisted.coherence.edge_m = 16
        persisted.clustering.max_window_width_mhz = 60.0
        persisted.contributor.min_freeze_snr = 40.0
        recommended = WindowPlanningSettings()
        recommended.leakage.tau_us = 7.0
        merged = resolve(explicit, preset, persisted, recommended)
        # explicit wins for edge_m
        assert merged.coherence.edge_m == 128
        # preset wins for max_window_width_mhz
        assert merged.clustering.max_window_width_mhz == 25.0
        # persisted wins for min_freeze_snr
        assert merged.contributor.min_freeze_snr == 40.0
        # recommended wins for tau_us
        assert merged.leakage.tau_us == 7.0
        # hard default fills the rest
        assert merged.coherence.trim_m == 32
        assert merged.contributor.magnitude_attachment_threshold == 0.1

    def test_sub_dataclass_independence(self) -> None:
        s = WindowPlanningSettings()
        s.contributor.min_freeze_snr = 25.0
        merged = resolve(explicit=s)
        assert merged.contributor.min_freeze_snr == 25.0
        assert merged.coherence.edge_m == 64
        assert merged.clustering.max_window_width_mhz == 40.0

    def test_resolved_instance_has_no_none_in_hard_default_fields(self) -> None:
        merged = resolve()
        for sub_name, defaults in _HARD_DEFAULTS.items():
            sub = getattr(merged, sub_name)
            for field_name in defaults:
                assert (
                    getattr(sub, field_name) is not None
                ), f"{sub_name}.{field_name} should be non-None after resolve"

    def test_recommended_layer_currently_unused_does_not_break_resolve(self) -> None:
        s = WindowPlanningSettings()
        s.coherence.edge_m = 16
        merged = resolve(explicit=s, recommended=None)
        assert merged.coherence.edge_m == 16


# ---------------------------------------------------------------------------
# attrs round-trip
# ---------------------------------------------------------------------------
class TestAttrsRoundTrip:
    def test_empty_round_trip(self) -> None:
        s = WindowPlanningSettings()
        attrs = to_attrs(s)
        round_tripped = from_attrs(attrs)
        assert round_tripped.is_empty()

    def test_round_trip_preserves_values(self) -> None:
        s = WindowPlanningSettings()
        s.coherence.edge_m = 128
        s.coherence.trim_m = 16
        s.clustering.max_window_width_mhz = 60.0
        s.contributor.min_freeze_snr = 25.0
        s.contributor.magnitude_attachment_threshold = 0.2
        s.leakage.tau_us = 5.0
        rt = from_attrs(to_attrs(s))
        assert rt.coherence.edge_m == 128
        assert rt.coherence.trim_m == 16
        assert rt.clustering.max_window_width_mhz == 60.0
        assert rt.contributor.min_freeze_snr == 25.0
        assert rt.contributor.magnitude_attachment_threshold == 0.2
        assert rt.leakage.tau_us == 5.0
        # Unset fields stay None
        assert rt.coherence.edge_threshold is None
        assert rt.clustering.min_window_half_width_mhz is None

    def test_none_round_trip_per_field(self) -> None:
        s = WindowPlanningSettings()
        attrs = to_attrs(s)
        for sub_name in _SUB_NAMES:
            for field_name, value in attrs[sub_name].items():
                assert value == "__None__", (
                    f"{sub_name}.{field_name} should encode as __None__; "
                    f"got {value!r}"
                )

    def test_round_trip_handles_bytes(self) -> None:
        attrs = {
            "coherence": {"edge_m": 32},
            "clustering": {},
            "contributor": {},
            "leakage": {"tau_us": 5.0},
        }
        rt = from_attrs(attrs)
        assert rt.coherence.edge_m == 32
        assert rt.leakage.tau_us == 5.0


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------
class TestYamlIo:
    def test_yaml_round_trip(self) -> None:
        s = WindowPlanningSettings()
        s.coherence.edge_m = 128
        s.contributor.min_freeze_snr = 25.0
        text = to_yaml(s)
        rt = from_yaml(text)
        assert rt.coherence.edge_m == 128
        assert rt.contributor.min_freeze_snr == 25.0

    def test_yaml_sparse_output_omits_none(self) -> None:
        s = WindowPlanningSettings()
        s.leakage.tau_us = 5.0
        out = to_yaml_dict(s)
        assert out == {"leakage": {"tau_us": 5.0}}

    def test_yaml_from_path(self, tmp_path) -> None:
        p = tmp_path / "preset.yaml"
        p.write_text(
            "coherence:\n  edge_m: 128\n  edge_threshold: 10.0\n"
            "clustering:\n  max_window_width_mhz: 60.0\n"
            "leakage:\n  tau_us: 5.0\n"
        )
        s = from_yaml(p)
        assert s.coherence.edge_m == 128
        assert s.coherence.edge_threshold == 10.0
        assert s.clustering.max_window_width_mhz == 60.0
        assert s.leakage.tau_us == 5.0

    def test_yaml_unknown_subblock_field_raises(self) -> None:
        text = "coherence:\n  bogus_field: 1\n"
        with pytest.raises(ValueError, match=r"unknown 'coherence' fields"):
            from_yaml(text)

    def test_yaml_unknown_top_key_raises(self) -> None:
        text = "bogus_block:\n  x: 1\n"
        with pytest.raises(ValueError, match=r"unknown top-level"):
            from_yaml(text)

    def test_yaml_allows_name_description_metadata(self) -> None:
        text = (
            "name: my_preset\n"
            "description: a docstring\n"
            "coherence:\n  edge_m: 128\n"
        )
        s = from_yaml(text)
        assert s.coherence.edge_m == 128

    def test_yaml_subblock_must_be_mapping(self) -> None:
        text = "coherence: 3\n"
        with pytest.raises(ValueError, match=r"must be a mapping"):
            from_yaml(text)

    def test_yaml_empty_document(self) -> None:
        s = from_yaml_dict(None)
        assert s.is_empty()
