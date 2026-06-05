"""
Pure unit tests for :mod:`ftmwpipeline.core.peak_detection_settings`.

Mirrors :mod:`tests.unit.core.test_noise_settings` and
:mod:`tests.unit.core.test_tau_calibration_settings`: empty dataclass,
resolution-chain precedence per layer, sub-dataclass independence,
attrs round-trip, YAML I/O, unknown-key rejection.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from ftmwpipeline.core.peak_detection_settings import (
    _HARD_DEFAULTS,
    GapPassSubSettings,
    PeakDetectionSettings,
    PrimaryPassSubSettings,
    PromotionSubSettings,
    SavgolSubSettings,
    from_attrs,
    from_yaml,
    from_yaml_dict,
    resolve,
    to_attrs,
    to_yaml,
    to_yaml_dict,
)

_SUB_NAMES = ("promotion", "savgol", "primary_pass", "gap_pass")


# ---------------------------------------------------------------------------
# Field-level defaults
# ---------------------------------------------------------------------------
class TestEmptyDataclass:
    def test_all_subdataclass_fields_default_to_none(self) -> None:
        s = PeakDetectionSettings()
        for sub_name in _SUB_NAMES:
            sub = getattr(s, sub_name)
            for f in fields(sub):
                assert (
                    getattr(sub, f.name) is None
                ), f"{sub_name}.{f.name} should default to None"

    def test_is_empty(self) -> None:
        assert PeakDetectionSettings().is_empty()
        s = PeakDetectionSettings()
        s.promotion.min_snr = 4.0
        assert not s.is_empty()
        s2 = PeakDetectionSettings()
        s2.gap_pass.run_gap_pass = False
        assert not s2.is_empty()


# ---------------------------------------------------------------------------
# Resolution chain
# ---------------------------------------------------------------------------
class TestResolve:
    def test_hard_defaults_when_all_layers_empty(self) -> None:
        merged = resolve()
        assert merged.promotion.min_snr == 3.0
        assert merged.promotion.internal_min_snr == 2.0
        assert merged.promotion.weak_medium_snr == 10.0
        assert merged.promotion.medium_strong_snr == 50.0
        assert merged.savgol.sg_window == 11
        assert merged.savgol.sg_order == 3
        assert merged.savgol.sg_fwhm_coverage == 4.0
        assert merged.savgol.sg_min_window == 5
        assert merged.primary_pass.primary_window == "blackmanharris"
        assert merged.primary_pass.min_exclusion_mhz == 0.0
        assert merged.primary_pass.detection_zpf == 2
        assert merged.gap_pass.run_gap_pass is True
        assert merged.gap_pass.gap_active_zpf == 2
        assert merged.gap_pass.gap_leakage_floor_k == 3.0

    def test_explicit_beats_preset(self) -> None:
        explicit = PeakDetectionSettings()
        explicit.promotion.min_snr = 5.0
        preset = PeakDetectionSettings()
        preset.promotion.min_snr = 10.0
        merged = resolve(explicit=explicit, preset=preset)
        assert merged.promotion.min_snr == 5.0

    def test_preset_beats_persisted(self) -> None:
        preset = PeakDetectionSettings()
        preset.savgol.sg_window = 13
        persisted = PeakDetectionSettings()
        persisted.savgol.sg_window = 21
        merged = resolve(preset=preset, persisted=persisted)
        assert merged.savgol.sg_window == 13

    def test_persisted_beats_recommended(self) -> None:
        persisted = PeakDetectionSettings()
        persisted.gap_pass.gap_leakage_floor_k = 6.0
        recommended = PeakDetectionSettings()
        recommended.gap_pass.gap_leakage_floor_k = 12.0
        merged = resolve(persisted=persisted, recommended=recommended)
        assert merged.gap_pass.gap_leakage_floor_k == 6.0

    def test_recommended_beats_hard_default(self) -> None:
        recommended = PeakDetectionSettings()
        recommended.gap_pass.gap_leakage_floor_k = 12.0
        merged = resolve(recommended=recommended)
        assert merged.gap_pass.gap_leakage_floor_k == 12.0

    def test_full_precedence_chain(self) -> None:
        explicit = PeakDetectionSettings()
        explicit.promotion.min_snr = 7.0
        preset = PeakDetectionSettings()
        preset.promotion.min_snr = 6.0
        preset.savgol.sg_window = 15
        persisted = PeakDetectionSettings()
        persisted.promotion.min_snr = 5.0
        persisted.savgol.sg_window = 9
        persisted.primary_pass.primary_window = "blackman"
        recommended = PeakDetectionSettings()
        recommended.gap_pass.gap_active_zpf = 4
        merged = resolve(explicit, preset, persisted, recommended)
        # explicit wins for min_snr
        assert merged.promotion.min_snr == 7.0
        # preset wins for sg_window
        assert merged.savgol.sg_window == 15
        # persisted wins for primary_window
        assert merged.primary_pass.primary_window == "blackman"
        # recommended wins for gap_active_zpf
        assert merged.gap_pass.gap_active_zpf == 4
        # hard default fills the rest
        assert merged.promotion.weak_medium_snr == 10.0
        assert merged.gap_pass.gap_leakage_floor_k == 3.0

    def test_sub_dataclass_independence(self) -> None:
        s = PeakDetectionSettings()
        s.savgol.sg_window = 99
        merged = resolve(explicit=s)
        assert merged.savgol.sg_window == 99
        # Other sub-dataclasses fall back to hard defaults
        assert merged.promotion.min_snr == 3.0
        assert merged.gap_pass.run_gap_pass is True

    def test_resolved_instance_has_no_none_in_hard_default_fields(self) -> None:
        merged = resolve()
        for sub_name, defaults in _HARD_DEFAULTS.items():
            sub = getattr(merged, sub_name)
            for field_name in defaults:
                assert (
                    getattr(sub, field_name) is not None
                ), f"{sub_name}.{field_name} should be non-None after resolve"

    def test_recommended_layer_currently_unused_does_not_break_resolve(self) -> None:
        s = PeakDetectionSettings()
        s.gap_pass.gap_active_zpf = 8
        merged = resolve(explicit=s, recommended=None)
        assert merged.gap_pass.gap_active_zpf == 8


# ---------------------------------------------------------------------------
# attrs round-trip
# ---------------------------------------------------------------------------
class TestAttrsRoundTrip:
    def test_empty_round_trip(self) -> None:
        s = PeakDetectionSettings()
        attrs = to_attrs(s)
        round_tripped = from_attrs(attrs)
        assert round_tripped.is_empty()

    def test_round_trip_preserves_values(self) -> None:
        s = PeakDetectionSettings()
        s.promotion.min_snr = 4.0
        s.promotion.internal_min_snr = 1.5
        s.savgol.sg_window = 15
        s.savgol.sg_order = 5
        s.primary_pass.primary_window = "hann"
        s.primary_pass.detection_zpf = 0
        s.gap_pass.run_gap_pass = False
        s.gap_pass.gap_leakage_floor_k = 6.0
        rt = from_attrs(to_attrs(s))
        assert rt.promotion.min_snr == 4.0
        assert rt.promotion.internal_min_snr == 1.5
        assert rt.savgol.sg_window == 15
        assert rt.savgol.sg_order == 5
        assert rt.primary_pass.primary_window == "hann"
        assert rt.primary_pass.detection_zpf == 0
        assert rt.gap_pass.run_gap_pass is False
        assert rt.gap_pass.gap_leakage_floor_k == 6.0
        # Unset fields stay None
        assert rt.savgol.sg_fwhm_coverage is None
        assert rt.gap_pass.gap_active_zpf is None

    def test_none_round_trip_per_field(self) -> None:
        s = PeakDetectionSettings()
        attrs = to_attrs(s)
        for sub_name in _SUB_NAMES:
            for field_name, value in attrs[sub_name].items():
                assert value == "__None__", (
                    f"{sub_name}.{field_name} should encode as __None__; "
                    f"got {value!r}"
                )

    def test_round_trip_handles_bytes(self) -> None:
        attrs = {
            "promotion": {"min_snr": 4.0},
            "savgol": {},
            "primary_pass": {"primary_window": b"hann"},
            "gap_pass": {},
        }
        rt = from_attrs(attrs)
        assert rt.promotion.min_snr == 4.0
        assert rt.primary_pass.primary_window == "hann"


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------
class TestYamlIo:
    def test_yaml_round_trip(self) -> None:
        s = PeakDetectionSettings()
        s.promotion.min_snr = 4.0
        s.savgol.sg_window = 15
        text = to_yaml(s)
        rt = from_yaml(text)
        assert rt.promotion.min_snr == 4.0
        assert rt.savgol.sg_window == 15

    def test_yaml_sparse_output_omits_none(self) -> None:
        s = PeakDetectionSettings()
        s.gap_pass.run_gap_pass = False
        out = to_yaml_dict(s)
        assert out == {"gap_pass": {"run_gap_pass": False}}

    def test_yaml_from_path(self, tmp_path) -> None:
        p = tmp_path / "preset.yaml"
        p.write_text(
            "promotion:\n  min_snr: 4.0\n  weak_medium_snr: 12.0\n"
            "savgol:\n  sg_window: 13\n"
            "primary_pass:\n  primary_window: blackman\n"
        )
        s = from_yaml(p)
        assert s.promotion.min_snr == 4.0
        assert s.promotion.weak_medium_snr == 12.0
        assert s.savgol.sg_window == 13
        assert s.primary_pass.primary_window == "blackman"

    def test_yaml_unknown_subblock_field_raises(self) -> None:
        text = "promotion:\n  bogus_field: 1\n"
        with pytest.raises(ValueError, match=r"unknown 'promotion' fields"):
            from_yaml(text)

    def test_yaml_unknown_top_key_raises(self) -> None:
        text = "bogus_block:\n  x: 1\n"
        with pytest.raises(ValueError, match=r"unknown top-level"):
            from_yaml(text)

    def test_yaml_allows_name_description_metadata(self) -> None:
        text = (
            "name: my_preset\n"
            "description: a docstring\n"
            "promotion:\n  min_snr: 4.0\n"
        )
        s = from_yaml(text)
        assert s.promotion.min_snr == 4.0

    def test_yaml_subblock_must_be_mapping(self) -> None:
        text = "promotion: 3\n"
        with pytest.raises(ValueError, match=r"must be a mapping"):
            from_yaml(text)

    def test_yaml_empty_document(self) -> None:
        s = from_yaml_dict(None)
        assert s.is_empty()
