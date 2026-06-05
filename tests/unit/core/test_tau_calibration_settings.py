"""
Pure unit tests for :mod:`ftmwpipeline.core.tau_calibration_settings`.

Mirrors :mod:`tests.unit.core.test_stage_fit_settings`: empty dataclass,
resolution-chain precedence per layer, sub-dataclass independence,
attrs round-trip, YAML I/O, unknown-key rejection. No real data required;
all tests are fast and free of I/O (the YAML tests use ``tmp_path``).
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from ftmwpipeline.core.tau_calibration_settings import (
    AggregationSubSettings,
    BandSubSettings,
    GaussianSubSettings,
    PolishSubSettings,
    RecommendationSubSettings,
    StftSubSettings,
    TauCalibrationSettings,
    _HARD_DEFAULTS,
    from_attrs,
    from_yaml,
    from_yaml_dict,
    resolve,
    to_attrs,
    to_yaml,
    to_yaml_dict,
)

_SUB_NAMES = (
    "stft",
    "polish",
    "aggregation",
    "band",
    "gaussian",
    "recommendation",
)


# ---------------------------------------------------------------------------
# Field-level defaults
# ---------------------------------------------------------------------------
class TestEmptyDataclass:
    def test_all_subdataclass_fields_default_to_none(self) -> None:
        s = TauCalibrationSettings()
        for sub_name in _SUB_NAMES:
            sub = getattr(s, sub_name)
            for f in fields(sub):
                assert (
                    getattr(sub, f.name) is None
                ), f"{sub_name}.{f.name} should default to None"

    def test_is_empty(self) -> None:
        assert TauCalibrationSettings().is_empty()
        s = TauCalibrationSettings()
        s.stft.n_seg = 8
        assert not s.is_empty()
        s2 = TauCalibrationSettings()
        s2.gaussian.snr_min = 30.0
        assert not s2.is_empty()


# ---------------------------------------------------------------------------
# Resolution chain
# ---------------------------------------------------------------------------
class TestResolve:
    def test_hard_defaults_when_all_layers_empty(self) -> None:
        merged = resolve()
        # Spot-check one field per sub-block.
        assert merged.stft.n_seg == 10
        assert merged.stft.t_sigma == 5.0
        assert merged.stft.rss_gate_factor == 5.0
        assert merged.polish.polish is True
        assert merged.polish.polish_snr_cap == 9.0
        assert merged.aggregation.min_contributors == 200
        assert merged.aggregation.sigma_tau_fraction_max == 0.20
        assert merged.band.compute_band_majorities is True
        assert merged.band.min_contributors_per_band == 50
        assert merged.gaussian.snr_min == 20.0
        assert merged.gaussian.min_contributors == 50  # distinct from aggregation
        assert merged.recommendation.snr_min == 20.0
        assert merged.recommendation.pure_margin_threshold == 0.10
        assert merged.recommendation.auto_recommend is True

    def test_min_contributors_collision_keeps_blocks_independent(self) -> None:
        """Aggregation and Gaussian both carry a ``min_contributors`` field
        with different hard defaults. Setting one must not bleed into the other."""
        s = TauCalibrationSettings()
        s.aggregation.min_contributors = 500
        merged = resolve(explicit=s)
        assert merged.aggregation.min_contributors == 500
        assert merged.gaussian.min_contributors == 50  # hard default

        s2 = TauCalibrationSettings()
        s2.gaussian.min_contributors = 25
        merged2 = resolve(explicit=s2)
        assert merged2.gaussian.min_contributors == 25
        assert merged2.aggregation.min_contributors == 200

    def test_explicit_beats_preset(self) -> None:
        explicit = TauCalibrationSettings()
        explicit.stft.n_seg = 16
        preset = TauCalibrationSettings()
        preset.stft.n_seg = 8
        merged = resolve(explicit=explicit, preset=preset)
        assert merged.stft.n_seg == 16

    def test_preset_beats_persisted(self) -> None:
        preset = TauCalibrationSettings()
        preset.stft.n_seg = 8
        persisted = TauCalibrationSettings()
        persisted.stft.n_seg = 4
        merged = resolve(preset=preset, persisted=persisted)
        assert merged.stft.n_seg == 8

    def test_persisted_beats_recommended(self) -> None:
        persisted = TauCalibrationSettings()
        persisted.stft.n_seg = 4
        recommended = TauCalibrationSettings()
        recommended.stft.n_seg = 20
        merged = resolve(persisted=persisted, recommended=recommended)
        assert merged.stft.n_seg == 4

    def test_recommended_beats_hard_default(self) -> None:
        recommended = TauCalibrationSettings()
        recommended.stft.n_seg = 20
        merged = resolve(recommended=recommended)
        assert merged.stft.n_seg == 20

    def test_full_precedence_chain(self) -> None:
        """explicit > preset > persisted > recommended > hard default, per field."""
        explicit = TauCalibrationSettings()
        explicit.stft.n_seg = 1
        preset = TauCalibrationSettings()
        preset.stft.n_seg = 2
        preset.polish.polish_snr_cap = 12.0
        persisted = TauCalibrationSettings()
        persisted.stft.n_seg = 3
        persisted.polish.polish_snr_cap = 6.0
        persisted.gaussian.snr_min = 30.0
        recommended = TauCalibrationSettings()
        recommended.gaussian.snr_min = 40.0
        recommended.recommendation.pure_margin_threshold = 0.25
        merged = resolve(explicit, preset, persisted, recommended)
        # explicit wins for n_seg
        assert merged.stft.n_seg == 1
        # preset wins for polish_snr_cap
        assert merged.polish.polish_snr_cap == 12.0
        # persisted wins for gaussian.snr_min (preset has no value)
        assert merged.gaussian.snr_min == 30.0
        # recommended wins for recommendation.pure_margin_threshold
        assert merged.recommendation.pure_margin_threshold == 0.25
        # hard default fills the rest
        assert merged.aggregation.sigma_tau_fraction_max == 0.20

    def test_sub_dataclass_independence(self) -> None:
        """Overriding one sub-dataclass field does not touch siblings."""
        s = TauCalibrationSettings()
        s.gaussian.snr_min = 33.0
        merged = resolve(explicit=s)
        assert merged.gaussian.snr_min == 33.0
        # Other sub-dataclass values come from hard defaults.
        assert merged.stft.n_seg == 10
        assert merged.aggregation.min_contributors == 200

    def test_resolved_instance_has_no_none_in_hard_default_fields(self) -> None:
        merged = resolve()
        for sub_name, defaults in _HARD_DEFAULTS.items():
            sub = getattr(merged, sub_name)
            for field_name in defaults:
                assert (
                    getattr(sub, field_name) is not None
                ), f"{sub_name}.{field_name} should be non-None after resolve"

    def test_recommended_layer_currently_unused_does_not_break_resolve(self) -> None:
        """Stage 2b call sites pass ``recommended=None``; resolve must accept it."""
        s = TauCalibrationSettings()
        s.stft.n_seg = 12
        merged = resolve(explicit=s, recommended=None)
        assert merged.stft.n_seg == 12


# ---------------------------------------------------------------------------
# attrs round-trip
# ---------------------------------------------------------------------------
class TestAttrsRoundTrip:
    def test_empty_round_trip(self) -> None:
        s = TauCalibrationSettings()
        attrs = to_attrs(s)
        round_tripped = from_attrs(attrs)
        assert round_tripped.is_empty()

    def test_round_trip_preserves_values(self) -> None:
        s = TauCalibrationSettings()
        s.stft.n_seg = 8
        s.stft.tau_max_us = 60.0
        s.polish.polish = False
        s.gaussian.snr_min = 25.0
        s.gaussian.tau_G_seeds = (50.0, 10.0, 2.0)
        s.band.compute_band_majorities = False
        s.band.band_labels = ("low", "high")
        s.band.band_edges_mhz = (30000.0,)
        s.recommendation.pure_margin_threshold = 0.05
        rt = from_attrs(to_attrs(s))
        assert rt.stft.n_seg == 8
        assert rt.stft.tau_max_us == 60.0
        assert rt.polish.polish is False
        assert rt.gaussian.snr_min == 25.0
        assert rt.gaussian.tau_G_seeds == (50.0, 10.0, 2.0)
        assert isinstance(rt.gaussian.tau_G_seeds, tuple)
        assert rt.band.compute_band_majorities is False
        assert rt.band.band_labels == ("low", "high")
        assert rt.band.band_edges_mhz == (30000.0,)
        assert rt.recommendation.pure_margin_threshold == 0.05
        # Unset fields stay None
        assert rt.stft.t_sigma is None
        assert rt.aggregation.min_contributors is None

    def test_none_round_trip_per_field(self) -> None:
        s = TauCalibrationSettings()
        attrs = to_attrs(s)
        for sub_name in _SUB_NAMES:
            for field_name, value in attrs[sub_name].items():
                assert value == "__None__", (
                    f"{sub_name}.{field_name} should encode as __None__; "
                    f"got {value!r}"
                )

    def test_round_trip_handles_bytes(self) -> None:
        attrs = {
            "stft": {"n_seg": 12},
            "polish": {},
            "aggregation": {},
            "band": {"band_labels": [b"low", b"mid", b"high"]},
            "gaussian": {},
            "recommendation": {},
        }
        rt = from_attrs(attrs)
        assert rt.stft.n_seg == 12
        assert rt.band.band_labels == ("low", "mid", "high")


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------
class TestYamlIo:
    def test_yaml_round_trip(self) -> None:
        s = TauCalibrationSettings()
        s.stft.n_seg = 8
        s.gaussian.snr_min = 25.0
        s.gaussian.tau_G_seeds = (50.0, 10.0)
        s.band.band_labels = ("A", "B", "C")
        text = to_yaml(s)
        rt = from_yaml(text)
        assert rt.stft.n_seg == 8
        assert rt.gaussian.snr_min == 25.0
        assert rt.gaussian.tau_G_seeds == (50.0, 10.0)
        assert rt.band.band_labels == ("A", "B", "C")

    def test_yaml_sparse_output_omits_none(self) -> None:
        s = TauCalibrationSettings()
        s.stft.n_seg = 8
        out = to_yaml_dict(s)
        assert out == {"stft": {"n_seg": 8}}

    def test_yaml_from_path(self, tmp_path) -> None:
        p = tmp_path / "preset.yaml"
        p.write_text(
            "stft:\n  n_seg: 12\n"
            "gaussian:\n  snr_min: 30.0\n  tau_G_seeds: [80.0, 20.0, 5.0]\n"
        )
        s = from_yaml(p)
        assert s.stft.n_seg == 12
        assert s.gaussian.snr_min == 30.0
        assert s.gaussian.tau_G_seeds == (80.0, 20.0, 5.0)

    def test_yaml_unknown_subblock_field_raises(self) -> None:
        text = "stft:\n  bogus_field: 1\n"
        with pytest.raises(ValueError, match=r"unknown 'stft' fields"):
            from_yaml(text)

    def test_yaml_unknown_top_key_raises(self) -> None:
        text = "bogus_block:\n  x: 1\n"
        with pytest.raises(ValueError, match=r"unknown top-level"):
            from_yaml(text)

    def test_yaml_allows_name_description_metadata(self) -> None:
        text = "name: my_preset\n" "description: a docstring\n" "stft:\n  n_seg: 8\n"
        s = from_yaml(text)
        assert s.stft.n_seg == 8

    def test_yaml_subblock_must_be_mapping(self) -> None:
        text = "stft: 3\n"
        with pytest.raises(ValueError, match=r"must be a mapping"):
            from_yaml(text)

    def test_yaml_empty_document(self) -> None:
        s = from_yaml_dict(None)
        assert s.is_empty()
