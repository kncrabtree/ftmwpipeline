"""
Pure unit tests for :mod:`ftmwpipeline.core.noise_settings`.

Mirrors :mod:`tests.unit.core.test_stage_fit_settings` and
:mod:`tests.unit.core.test_tau_calibration_settings`: empty dataclass,
resolution-chain precedence per layer, sub-dataclass independence,
attrs round-trip, YAML I/O, unknown-key rejection.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from ftmwpipeline.core.noise_settings import (
    BinningSubSettings,
    NoiseSettings,
    SkewnessSubSettings,
    SkirtExclusionSubSettings,
    SmoothingSubSettings,
    _HARD_DEFAULTS,
    from_attrs,
    from_yaml,
    from_yaml_dict,
    resolve,
    to_attrs,
    to_yaml,
    to_yaml_dict,
)


_SUB_NAMES = ("binning", "skewness", "smoothing", "skirt_exclusion")


# ---------------------------------------------------------------------------
# Field-level defaults
# ---------------------------------------------------------------------------
class TestEmptyDataclass:
    def test_all_subdataclass_fields_default_to_none(self) -> None:
        s = NoiseSettings()
        for sub_name in _SUB_NAMES:
            sub = getattr(s, sub_name)
            for f in fields(sub):
                assert getattr(sub, f.name) is None, (
                    f"{sub_name}.{f.name} should default to None"
                )

    def test_is_empty(self) -> None:
        assert NoiseSettings().is_empty()
        s = NoiseSettings()
        s.binning.abs_min_bin_size = 100
        assert not s.is_empty()
        s2 = NoiseSettings()
        s2.skirt_exclusion.strong_peak_snr = 25.0
        assert not s2.is_empty()


# ---------------------------------------------------------------------------
# Resolution chain
# ---------------------------------------------------------------------------
class TestResolve:
    def test_hard_defaults_when_all_layers_empty(self) -> None:
        merged = resolve()
        assert merged.binning.subdivision_threshold == 0.08
        assert merged.binning.abs_min_bin_size == 300
        assert merged.binning.min_bin_fraction == 1 / 64
        assert merged.binning.min_noise_fraction == 2 / 3
        assert merged.skewness.skew_target == 0.631
        assert merged.skewness.inc == 0.01
        assert merged.smoothing.smoothing_window_mhz == 300.0
        assert merged.skirt_exclusion.strong_peak_snr == 20.0
        assert merged.skirt_exclusion.skirt_exclusion_k == 1.5
        assert merged.skirt_exclusion.max_skirt_exclusion_mhz == 500.0

    def test_explicit_beats_preset(self) -> None:
        explicit = NoiseSettings()
        explicit.smoothing.smoothing_window_mhz = 100.0
        preset = NoiseSettings()
        preset.smoothing.smoothing_window_mhz = 200.0
        merged = resolve(explicit=explicit, preset=preset)
        assert merged.smoothing.smoothing_window_mhz == 100.0

    def test_preset_beats_persisted(self) -> None:
        preset = NoiseSettings()
        preset.smoothing.smoothing_window_mhz = 200.0
        persisted = NoiseSettings()
        persisted.smoothing.smoothing_window_mhz = 400.0
        merged = resolve(preset=preset, persisted=persisted)
        assert merged.smoothing.smoothing_window_mhz == 200.0

    def test_persisted_beats_recommended(self) -> None:
        persisted = NoiseSettings()
        persisted.skirt_exclusion.strong_peak_snr = 15.0
        recommended = NoiseSettings()
        recommended.skirt_exclusion.strong_peak_snr = 30.0
        merged = resolve(persisted=persisted, recommended=recommended)
        assert merged.skirt_exclusion.strong_peak_snr == 15.0

    def test_recommended_beats_hard_default(self) -> None:
        recommended = NoiseSettings()
        recommended.skirt_exclusion.strong_peak_snr = 30.0
        merged = resolve(recommended=recommended)
        assert merged.skirt_exclusion.strong_peak_snr == 30.0

    def test_full_precedence_chain(self) -> None:
        explicit = NoiseSettings()
        explicit.binning.subdivision_threshold = 0.01
        preset = NoiseSettings()
        preset.binning.subdivision_threshold = 0.05
        preset.skewness.skew_target = 0.75
        persisted = NoiseSettings()
        persisted.binning.subdivision_threshold = 0.10
        persisted.skewness.skew_target = 0.60
        persisted.skirt_exclusion.strong_peak_snr = 25.0
        recommended = NoiseSettings()
        recommended.smoothing.smoothing_window_mhz = 100.0
        merged = resolve(explicit, preset, persisted, recommended)
        # explicit wins for subdivision_threshold
        assert merged.binning.subdivision_threshold == 0.01
        # preset wins for skew_target
        assert merged.skewness.skew_target == 0.75
        # persisted wins for strong_peak_snr (no higher layer set it)
        assert merged.skirt_exclusion.strong_peak_snr == 25.0
        # recommended wins for smoothing_window_mhz
        assert merged.smoothing.smoothing_window_mhz == 100.0
        # hard default fills the rest
        assert merged.binning.abs_min_bin_size == 300

    def test_sub_dataclass_independence(self) -> None:
        s = NoiseSettings()
        s.skewness.skew_target = 0.9
        merged = resolve(explicit=s)
        assert merged.skewness.skew_target == 0.9
        assert merged.binning.subdivision_threshold == 0.08
        assert merged.smoothing.smoothing_window_mhz == 300.0

    def test_resolved_instance_has_no_none_in_hard_default_fields(self) -> None:
        merged = resolve()
        for sub_name, defaults in _HARD_DEFAULTS.items():
            sub = getattr(merged, sub_name)
            for field_name in defaults:
                assert getattr(sub, field_name) is not None, (
                    f"{sub_name}.{field_name} should be non-None after resolve"
                )

    def test_recommended_layer_currently_unused_does_not_break_resolve(self) -> None:
        s = NoiseSettings()
        s.binning.abs_min_bin_size = 500
        merged = resolve(explicit=s, recommended=None)
        assert merged.binning.abs_min_bin_size == 500


# ---------------------------------------------------------------------------
# attrs round-trip
# ---------------------------------------------------------------------------
class TestAttrsRoundTrip:
    def test_empty_round_trip(self) -> None:
        s = NoiseSettings()
        attrs = to_attrs(s)
        round_tripped = from_attrs(attrs)
        assert round_tripped.is_empty()

    def test_round_trip_preserves_values(self) -> None:
        s = NoiseSettings()
        s.binning.subdivision_threshold = 0.05
        s.binning.abs_min_bin_size = 500
        s.skewness.skew_target = 0.75
        s.smoothing.smoothing_window_mhz = 100.0
        s.skirt_exclusion.strong_peak_snr = 25.0
        s.skirt_exclusion.skirt_exclusion_k = 2.0
        rt = from_attrs(to_attrs(s))
        assert rt.binning.subdivision_threshold == 0.05
        assert rt.binning.abs_min_bin_size == 500
        assert rt.skewness.skew_target == 0.75
        assert rt.smoothing.smoothing_window_mhz == 100.0
        assert rt.skirt_exclusion.strong_peak_snr == 25.0
        assert rt.skirt_exclusion.skirt_exclusion_k == 2.0
        # Unset fields stay None
        assert rt.skewness.inc is None
        assert rt.skirt_exclusion.max_skirt_exclusion_mhz is None

    def test_none_round_trip_per_field(self) -> None:
        s = NoiseSettings()
        attrs = to_attrs(s)
        for sub_name in _SUB_NAMES:
            for field_name, value in attrs[sub_name].items():
                assert value == "__None__", (
                    f"{sub_name}.{field_name} should encode as __None__; "
                    f"got {value!r}"
                )

    def test_round_trip_handles_bytes(self) -> None:
        attrs = {
            "binning": {"subdivision_threshold": 0.05},
            "skewness": {},
            "smoothing": {},
            "skirt_exclusion": {},
        }
        rt = from_attrs(attrs)
        assert rt.binning.subdivision_threshold == 0.05


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------
class TestYamlIo:
    def test_yaml_round_trip(self) -> None:
        s = NoiseSettings()
        s.binning.subdivision_threshold = 0.05
        s.smoothing.smoothing_window_mhz = 100.0
        text = to_yaml(s)
        rt = from_yaml(text)
        assert rt.binning.subdivision_threshold == 0.05
        assert rt.smoothing.smoothing_window_mhz == 100.0

    def test_yaml_sparse_output_omits_none(self) -> None:
        s = NoiseSettings()
        s.binning.abs_min_bin_size = 200
        out = to_yaml_dict(s)
        assert out == {"binning": {"abs_min_bin_size": 200}}

    def test_yaml_from_path(self, tmp_path) -> None:
        p = tmp_path / "preset.yaml"
        p.write_text(
            "binning:\n  subdivision_threshold: 0.05\n  abs_min_bin_size: 500\n"
            "smoothing:\n  smoothing_window_mhz: 100.0\n"
        )
        s = from_yaml(p)
        assert s.binning.subdivision_threshold == 0.05
        assert s.binning.abs_min_bin_size == 500
        assert s.smoothing.smoothing_window_mhz == 100.0

    def test_yaml_unknown_subblock_field_raises(self) -> None:
        text = "binning:\n  bogus_field: 1\n"
        with pytest.raises(ValueError, match=r"unknown 'binning' fields"):
            from_yaml(text)

    def test_yaml_unknown_top_key_raises(self) -> None:
        text = "bogus_block:\n  x: 1\n"
        with pytest.raises(ValueError, match=r"unknown top-level"):
            from_yaml(text)

    def test_yaml_allows_name_description_metadata(self) -> None:
        text = (
            "name: my_preset\n"
            "description: a docstring\n"
            "binning:\n  subdivision_threshold: 0.05\n"
        )
        s = from_yaml(text)
        assert s.binning.subdivision_threshold == 0.05

    def test_yaml_subblock_must_be_mapping(self) -> None:
        text = "binning: 3\n"
        with pytest.raises(ValueError, match=r"must be a mapping"):
            from_yaml(text)

    def test_yaml_empty_document(self) -> None:
        s = from_yaml_dict(None)
        assert s.is_empty()
