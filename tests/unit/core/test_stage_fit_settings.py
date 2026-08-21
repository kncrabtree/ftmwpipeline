"""
Pure unit tests for ``StageFitSettings`` + ``ShapeSpec``, the resolution chain,
attrs round-trip, and YAML I/O.

No real data required; all tests are fast and free of I/O (the YAML tests use
``tmp_path``).
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from ftmwpipeline.core.peak_shape import PeakShape
from ftmwpipeline.core.stage_fit_settings import (
    _HARD_DEFAULTS,
    DoubletAlternativeSubSettings,
    ShapeSpec,
    StageFitSettings,
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
    def test_all_subdataclass_fields_default_to_none(self) -> None:
        """Empty StageFitSettings has every field at None across every sub."""
        s = StageFitSettings()
        assert s.shape is None
        for sub_name in (
            "tau",
            "seeder",
            "conservative",
            "penalties",
            "rescue",
            "thaw",
        ):
            sub = getattr(s, sub_name)
            for f in fields(sub):
                assert (
                    getattr(sub, f.name) is None
                ), f"{sub_name}.{f.name} should default to None"

    def test_is_empty(self) -> None:
        assert StageFitSettings().is_empty()
        s = StageFitSettings()
        s.tau.max_decay_factor = 3.0
        assert not s.is_empty()
        s2 = StageFitSettings()
        s2.shape = ShapeSpec(kind=PeakShape.GAUSSIAN)
        assert not s2.is_empty()


# ---------------------------------------------------------------------------
# ShapeSpec coercion
# ---------------------------------------------------------------------------
class TestShapeSpec:
    def test_coerce_string(self) -> None:
        spec = ShapeSpec.coerce("gaussian")
        assert spec is not None and spec.kind is PeakShape.GAUSSIAN

    def test_coerce_string_case_insensitive(self) -> None:
        spec = ShapeSpec.coerce(" Lorentzian ")
        assert spec is not None and spec.kind is PeakShape.LORENTZIAN

    def test_coerce_enum(self) -> None:
        spec = ShapeSpec.coerce(PeakShape.GAUSSIAN)
        assert spec is not None and spec.kind is PeakShape.GAUSSIAN

    def test_coerce_dict(self) -> None:
        spec = ShapeSpec.coerce({"kind": "gaussian"})
        assert spec is not None and spec.kind is PeakShape.GAUSSIAN

    def test_coerce_passthrough(self) -> None:
        original = ShapeSpec(kind=PeakShape.GAUSSIAN)
        assert ShapeSpec.coerce(original) is original

    def test_coerce_none(self) -> None:
        assert ShapeSpec.coerce(None) is None

    def test_coerce_invalid(self) -> None:
        with pytest.raises(ValueError):
            ShapeSpec.coerce("voigt")  # not in PeakShape yet
        with pytest.raises(ValueError):
            ShapeSpec.coerce({"foo": "bar"})


# ---------------------------------------------------------------------------
# Resolution chain
# ---------------------------------------------------------------------------
class TestResolve:
    def test_hard_defaults_when_all_layers_empty(self) -> None:
        merged = resolve()
        assert merged.shape is not None and merged.shape.kind is PeakShape.LORENTZIAN
        assert merged.tau.max_decay_factor == 5.0
        assert merged.tau.tau_penalty_lambda == 50.0
        assert merged.conservative.significance == 0.05
        assert merged.conservative.max_peaks == 0  # 0 = no cap (width-bounded)
        assert merged.penalties.phase_penalty_lambda == 100.0
        assert merged.rescue.max_rounds == 5
        assert merged.thaw.max_thaw_rounds == 2

    def test_explicit_beats_preset(self) -> None:
        explicit = StageFitSettings()
        explicit.tau.max_decay_factor = 2.0
        preset = StageFitSettings()
        preset.tau.max_decay_factor = 3.0
        merged = resolve(explicit=explicit, preset=preset)
        assert merged.tau.max_decay_factor == 2.0

    def test_persisted_beats_preset(self) -> None:
        preset = StageFitSettings()
        preset.tau.max_decay_factor = 3.0
        persisted = StageFitSettings()
        persisted.tau.max_decay_factor = 4.0
        merged = resolve(preset=preset, persisted=persisted)
        assert merged.tau.max_decay_factor == 4.0

    def test_persisted_beats_recommended(self) -> None:
        persisted = StageFitSettings()
        persisted.tau.max_decay_factor = 4.0
        recommended = StageFitSettings()
        recommended.tau.max_decay_factor = 7.0
        merged = resolve(persisted=persisted, recommended=recommended)
        assert merged.tau.max_decay_factor == 4.0

    def test_recommended_beats_hard_default(self) -> None:
        recommended = StageFitSettings()
        recommended.tau.max_decay_factor = 7.0
        merged = resolve(recommended=recommended)
        assert merged.tau.max_decay_factor == 7.0

    def test_full_precedence_chain(self) -> None:
        """explicit > persisted > preset > recommended > hard default, per field."""
        explicit = StageFitSettings()
        explicit.tau.max_decay_factor = 1.0
        preset = StageFitSettings()
        preset.tau.max_decay_factor = 2.0
        preset.conservative.max_peaks = 6
        persisted = StageFitSettings()
        persisted.tau.max_decay_factor = 3.0
        persisted.conservative.max_peaks = 7
        persisted.rescue.max_rounds = 4
        recommended = StageFitSettings()
        recommended.tau.max_decay_factor = 4.0
        recommended.rescue.max_rounds = 6
        recommended.thaw.max_thaw_rounds = 9
        merged = resolve(explicit, preset, persisted, recommended)
        # explicit wins for max_decay_factor
        assert merged.tau.max_decay_factor == 1.0
        # persisted wins for max_peaks (outranks preset)
        assert merged.conservative.max_peaks == 7
        # persisted wins for rescue.max_rounds (preset has no value)
        assert merged.rescue.max_rounds == 4
        # recommended wins for thaw.max_thaw_rounds
        assert merged.thaw.max_thaw_rounds == 9
        # hard default fills the rest
        assert merged.penalties.phase_penalty_lambda == 100.0

    def test_shape_precedence(self) -> None:
        explicit = StageFitSettings(shape=ShapeSpec(kind=PeakShape.GAUSSIAN))
        recommended = StageFitSettings(shape=ShapeSpec(kind=PeakShape.LORENTZIAN))
        merged = resolve(explicit=explicit, recommended=recommended)
        assert merged.shape is not None and merged.shape.kind is PeakShape.GAUSSIAN

    def test_shape_recommended_fallback(self) -> None:
        recommended = StageFitSettings(shape=ShapeSpec(kind=PeakShape.GAUSSIAN))
        merged = resolve(recommended=recommended)
        assert merged.shape is not None and merged.shape.kind is PeakShape.GAUSSIAN

    def test_sub_dataclass_independence(self) -> None:
        """Overriding one sub-dataclass field does not touch siblings."""
        s = StageFitSettings()
        s.tau.max_decay_factor = 3.0
        merged = resolve(explicit=s)
        # tau took the override
        assert merged.tau.max_decay_factor == 3.0
        # other sub-dataclass values came from hard defaults
        assert merged.conservative.significance == 0.05
        assert merged.rescue.snr_threshold == 2.5

    def test_resolved_instance_has_no_none_in_hard_default_fields(self) -> None:
        """Every field with a hard default ends up non-None after resolve."""
        merged = resolve()
        for sub_name, defaults in _HARD_DEFAULTS.items():
            if sub_name == "shape":
                continue
            sub = getattr(merged, sub_name)
            for field_name in defaults:
                assert (
                    getattr(sub, field_name) is not None
                ), f"{sub_name}.{field_name} should be non-None after resolve"


# ---------------------------------------------------------------------------
# attrs round-trip (drives Step 3 HDF5 serialization)
# ---------------------------------------------------------------------------
class TestAttrsRoundTrip:
    def test_empty_round_trip(self) -> None:
        s = StageFitSettings()
        attrs = to_attrs(s)
        assert attrs["shape"] == "__None__"
        round_tripped = from_attrs(attrs)
        assert round_tripped.shape is None
        assert round_tripped.is_empty()

    def test_round_trip_preserves_values(self) -> None:
        s = StageFitSettings(shape=ShapeSpec(kind=PeakShape.GAUSSIAN))
        s.tau.max_decay_factor = 3.0
        s.tau.fit_tau = False
        s.conservative.max_peaks = 4
        s.rescue.max_rounds = 2
        s.thaw.max_thaw_rounds = 1
        rt = from_attrs(to_attrs(s))
        assert rt.shape is not None and rt.shape.kind is PeakShape.GAUSSIAN
        assert rt.tau.max_decay_factor == 3.0
        assert rt.tau.fit_tau is False
        assert rt.conservative.max_peaks == 4
        assert rt.rescue.max_rounds == 2
        assert rt.thaw.max_thaw_rounds == 1
        # Unset fields stay None
        assert rt.tau.tau_penalty_lambda is None
        assert rt.penalties.phase_penalty_lambda is None

    def test_none_round_trip_per_field(self) -> None:
        """Every Optional field encodes as ``__None__`` and decodes back to None."""
        s = StageFitSettings()
        attrs = to_attrs(s)
        for sub_name in (
            "tau",
            "seeder",
            "conservative",
            "penalties",
            "rescue",
            "thaw",
        ):
            for field_name, value in attrs[sub_name].items():
                assert value == "__None__", (
                    f"{sub_name}.{field_name} should encode as __None__; "
                    f"got {value!r}"
                )

    def test_round_trip_handles_bytes(self) -> None:
        """HDF5 sometimes returns bytes for string attrs; decoder tolerates it."""
        attrs = {
            "shape": {"kind": "gaussian"},
            "tau": {"max_decay_factor": 3.0},
            "conservative": {"n_eff_kind": b"perplexity_log1p_snr"},
        }
        rt = from_attrs(attrs)
        assert rt.shape is not None and rt.shape.kind is PeakShape.GAUSSIAN
        assert rt.tau.max_decay_factor == 3.0
        assert rt.conservative.n_eff_kind == "perplexity_log1p_snr"


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------
class TestYamlIo:
    def test_yaml_shape_shorthand(self) -> None:
        text = "shape: gaussian\n"
        s = from_yaml(text)
        assert s.shape is not None and s.shape.kind is PeakShape.GAUSSIAN

    def test_yaml_round_trip(self) -> None:
        s = StageFitSettings(shape=ShapeSpec(kind=PeakShape.GAUSSIAN))
        s.tau.max_decay_factor = 3.0
        s.tau.per_band_tau = True
        s.conservative.max_peaks = 6
        text = to_yaml(s)
        rt = from_yaml(text)
        assert rt.shape is not None and rt.shape.kind is PeakShape.GAUSSIAN
        assert rt.tau.max_decay_factor == 3.0
        assert rt.tau.per_band_tau is True
        assert rt.conservative.max_peaks == 6
        # Unset stays unset
        assert rt.penalties.phase_penalty_lambda is None

    def test_yaml_sparse_output_omits_none(self) -> None:
        s = StageFitSettings()
        s.tau.max_decay_factor = 3.0
        out = to_yaml_dict(s)
        assert out == {"tau": {"max_decay_factor": 3.0}}

    def test_yaml_from_path(self, tmp_path) -> None:
        p = tmp_path / "preset.yaml"
        p.write_text(
            "shape: gaussian\n"
            "tau:\n"
            "  max_decay_factor: 4.0\n"
            "  per_band_tau: true\n"
            "rescue:\n"
            "  max_rounds: 3\n"
            "  snr_threshold: 3.0\n"
        )
        s = from_yaml(p)
        assert s.shape is not None and s.shape.kind is PeakShape.GAUSSIAN
        assert s.tau.max_decay_factor == 4.0
        assert s.tau.per_band_tau is True
        assert s.rescue.max_rounds == 3
        assert s.rescue.snr_threshold == 3.0

    def test_yaml_unknown_subblock_field_raises(self) -> None:
        text = "tau:\n  bogus_field: 1.0\n"
        with pytest.raises(ValueError, match=r"unknown 'tau' fields"):
            from_yaml(text)

    def test_yaml_unknown_top_key_raises(self) -> None:
        text = "bogus_block:\n  x: 1\n"
        with pytest.raises(ValueError, match=r"unknown top-level"):
            from_yaml(text)

    def test_yaml_allows_name_description_metadata(self) -> None:
        text = (
            "name: my_preset\n"
            "description: a docstring\n"
            "tau:\n  max_decay_factor: 4.0\n"
        )
        s = from_yaml(text)
        assert s.tau.max_decay_factor == 4.0

    def test_yaml_subblock_must_be_mapping(self) -> None:
        text = "tau: 3.0\n"
        with pytest.raises(ValueError, match=r"must be a mapping"):
            from_yaml(text)

    def test_yaml_empty_document(self) -> None:
        s = from_yaml_dict(None)
        assert s.is_empty()


# ---------------------------------------------------------------------------
# DoubletAlternativeSubSettings — defaults and resolution chain
# ---------------------------------------------------------------------------
class TestDoubletAlternativeSubSettings:
    def test_hard_defaults_present(self) -> None:
        assert "doublet_alternative" in _HARD_DEFAULTS
        da = _HARD_DEFAULTS["doublet_alternative"]
        assert da["enabled"] is True
        assert da["k_res"] == 1.5
        assert da["r_min"] == 0.05

    def test_resolve_gives_non_none_fields(self) -> None:
        merged = resolve()
        da = merged.doublet_alternative
        assert da.enabled is True
        assert da.k_res == 1.5
        assert da.r_min == 0.05

    def test_persisted_beats_preset(self) -> None:
        preset = StageFitSettings()
        preset.doublet_alternative = DoubletAlternativeSubSettings(k_res=2.0)
        persisted = StageFitSettings()
        persisted.doublet_alternative = DoubletAlternativeSubSettings(k_res=3.0)
        merged = resolve(preset=preset, persisted=persisted)
        assert merged.doublet_alternative.k_res == 3.0

    def test_explicit_beats_persisted(self) -> None:
        explicit = StageFitSettings()
        explicit.doublet_alternative = DoubletAlternativeSubSettings(k_res=1.0)
        persisted = StageFitSettings()
        persisted.doublet_alternative = DoubletAlternativeSubSettings(k_res=3.0)
        merged = resolve(explicit=explicit, persisted=persisted)
        assert merged.doublet_alternative.k_res == 1.0

    def test_attrs_round_trip(self) -> None:
        s = StageFitSettings()
        s.doublet_alternative = DoubletAlternativeSubSettings(
            enabled=False, k_res=2.5, r_min=0.08
        )
        rt = from_attrs(to_attrs(s))
        da = rt.doublet_alternative
        assert da.enabled is False
        assert da.k_res == 2.5
        assert da.r_min == 0.08

    def test_yaml_round_trip(self) -> None:
        s = StageFitSettings()
        s.doublet_alternative = DoubletAlternativeSubSettings(enabled=False, k_res=2.0)
        text = to_yaml(s)
        rt = from_yaml(text)
        assert rt.doublet_alternative.enabled is False
        assert rt.doublet_alternative.k_res == 2.0
        # unset r_min stays None in the preset layer
        assert rt.doublet_alternative.r_min is None
