"""
Pure unit tests for :mod:`ftmwpipeline.core.noise_settings`.

Mirrors :mod:`tests.unit.core.test_stage_fit_settings` and
:mod:`tests.unit.core.test_tau_calibration_settings`: empty dataclass,
resolution-chain precedence per layer, attrs round-trip, YAML I/O,
unknown-key rejection. Stage 2 has a single estimator, so ``NoiseSettings``
is flat (the scatter knobs sit directly on it).
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from ftmwpipeline.core.noise_settings import (
    _HARD_DEFAULTS,
    NoiseSettings,
    from_attrs,
    from_yaml,
    from_yaml_dict,
    resolve,
    to_attrs,
    to_yaml,
    to_yaml_dict,
)


# ---------------------------------------------------------------------------
# Field-level defaults
# ---------------------------------------------------------------------------
class TestEmptyDataclass:
    def test_all_fields_default_to_none(self) -> None:
        s = NoiseSettings()
        for f in fields(s):
            assert getattr(s, f.name) is None, f"{f.name} should default to None"

    def test_is_empty(self) -> None:
        assert NoiseSettings().is_empty()
        assert not NoiseSettings(window_mhz=120.0).is_empty()
        assert not NoiseSettings(region_aware=False).is_empty()


# ---------------------------------------------------------------------------
# Resolution chain
# ---------------------------------------------------------------------------
class TestResolve:
    def test_hard_defaults_when_all_layers_empty(self) -> None:
        merged = resolve()
        assert merged.window_mhz == 80.0
        assert merged.pedestal_mhz == 20.0
        assert merged.line_k == 8.0
        assert merged.n_iter == 3
        assert merged.region_aware is True
        assert merged.smoothing_mhz == 800.0
        assert merged.smoothing_percentile == 50.0
        assert merged.convolve_mhz == 200.0

    def test_field_precedence_and_independence(self) -> None:
        explicit = NoiseSettings(window_mhz=40.0)
        preset = NoiseSettings(window_mhz=120.0, line_k=6.0)
        persisted = NoiseSettings(region_aware=False)
        merged = resolve(explicit=explicit, preset=preset, persisted=persisted)
        # explicit wins for window_mhz
        assert merged.window_mhz == 40.0
        # preset wins for line_k (no higher layer set it)
        assert merged.line_k == 6.0
        # persisted wins for region_aware
        assert merged.region_aware is False
        # hard default fills the untouched field
        assert merged.pedestal_mhz == 20.0

    def test_explicit_beats_preset(self) -> None:
        merged = resolve(
            explicit=NoiseSettings(smoothing_mhz=100.0),
            preset=NoiseSettings(smoothing_mhz=200.0),
        )
        assert merged.smoothing_mhz == 100.0

    def test_persisted_beats_preset(self) -> None:
        merged = resolve(
            preset=NoiseSettings(smoothing_mhz=200.0),
            persisted=NoiseSettings(smoothing_mhz=400.0),
        )
        assert merged.smoothing_mhz == 400.0

    def test_persisted_beats_recommended(self) -> None:
        merged = resolve(
            persisted=NoiseSettings(line_k=5.0),
            recommended=NoiseSettings(line_k=12.0),
        )
        assert merged.line_k == 5.0

    def test_recommended_beats_hard_default(self) -> None:
        merged = resolve(recommended=NoiseSettings(line_k=12.0))
        assert merged.line_k == 12.0

    def test_full_precedence_chain(self) -> None:
        explicit = NoiseSettings(window_mhz=40.0)
        preset = NoiseSettings(window_mhz=60.0, line_k=6.0)
        persisted = NoiseSettings(window_mhz=120.0, line_k=4.0, pedestal_mhz=40.0)
        recommended = NoiseSettings(smoothing_mhz=400.0)
        merged = resolve(explicit, preset, persisted, recommended)
        assert merged.window_mhz == 40.0  # explicit
        assert merged.line_k == 4.0  # persisted (outranks preset)
        assert merged.pedestal_mhz == 40.0  # persisted (no higher layer set it)
        assert merged.smoothing_mhz == 400.0  # recommended
        assert merged.n_iter == 3  # hard default

    def test_resolved_instance_has_no_none_in_hard_default_fields(self) -> None:
        merged = resolve()
        for field_name in _HARD_DEFAULTS:
            assert (
                getattr(merged, field_name) is not None
            ), f"{field_name} should be non-None after resolve"

    def test_recommended_layer_currently_unused_does_not_break_resolve(self) -> None:
        merged = resolve(explicit=NoiseSettings(n_iter=5), recommended=None)
        assert merged.n_iter == 5


# ---------------------------------------------------------------------------
# attrs round-trip
# ---------------------------------------------------------------------------
class TestAttrsRoundTrip:
    def test_empty_round_trip(self) -> None:
        round_tripped = from_attrs(to_attrs(NoiseSettings()))
        assert round_tripped.is_empty()

    def test_round_trip_preserves_values(self) -> None:
        s = NoiseSettings(window_mhz=60.0, n_iter=5, region_aware=False)
        rt = from_attrs(to_attrs(s))
        assert rt.window_mhz == 60.0
        assert rt.n_iter == 5
        assert rt.region_aware is False
        # Unset fields stay None
        assert rt.pedestal_mhz is None

    def test_none_round_trip_per_field(self) -> None:
        attrs = to_attrs(NoiseSettings())
        for field_name, value in attrs.items():
            assert (
                value == "__None__"
            ), f"{field_name} should encode as __None__; got {value!r}"

    def test_round_trip_ignores_unknown_keys(self) -> None:
        rt = from_attrs({"window_mhz": 60.0, "bogus": 1})
        assert rt.window_mhz == 60.0


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------
class TestYamlIo:
    def test_yaml_round_trip(self) -> None:
        s = NoiseSettings(window_mhz=120.0, region_aware=False)
        rt = from_yaml(to_yaml(s))
        assert rt.window_mhz == 120.0
        assert rt.region_aware is False

    def test_yaml_sparse_output_omits_none(self) -> None:
        out = to_yaml_dict(NoiseSettings(window_mhz=60.0))
        assert out == {"window_mhz": 60.0}

    def test_yaml_from_path(self, tmp_path) -> None:
        p = tmp_path / "preset.yaml"
        p.write_text("window_mhz: 60.0\nn_iter: 5\n")
        s = from_yaml(p)
        assert s.window_mhz == 60.0
        assert s.n_iter == 5

    def test_yaml_unknown_field_raises(self) -> None:
        with pytest.raises(ValueError, match=r"unknown stage2 fields"):
            from_yaml("bogus_field: 1\n")

    def test_yaml_allows_name_description_metadata(self) -> None:
        s = from_yaml("name: my_preset\ndescription: a docstring\nwindow_mhz: 60.0\n")
        assert s.window_mhz == 60.0

    def test_yaml_root_must_be_mapping(self) -> None:
        with pytest.raises(ValueError, match=r"must be a mapping"):
            from_yaml("- 1\n- 2\n")

    def test_yaml_empty_document(self) -> None:
        assert from_yaml_dict(None).is_empty()
