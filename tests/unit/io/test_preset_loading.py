"""
Unit tests for ``load_preset`` (bare-name + path resolution, ``fit:``
wrapper handling, error surfaces) and for the preset/explicit kwarg
precedence inside ``resolve()``.

These verify the Step 4 surface: ``ftmwpipeline fit-peaks --preset
gaussian_default exp.ftmw`` works end-to-end on the YAML side, and the
override precedence is honoured.
"""

from __future__ import annotations

from importlib.resources import files

import pytest

from ftmwpipeline.core.peak_shape import PeakShape
from ftmwpipeline.core.stage_fit_settings import (
    ShapeSpec,
    StageFitSettings,
    load_preset,
    resolve,
)


class TestPackagedPresetResolution:
    def test_packaged_preset_dir_exists(self) -> None:
        pkg = files("ftmwpipeline.presets")
        names = sorted(
            p.name for p in pkg.iterdir() if p.name.endswith(".yaml")
        )
        assert "gaussian_default.yaml" in names
        assert "lorentzian_legacy.yaml" in names
        assert "instrument_bc_2638.yaml" in names

    def test_load_gaussian_default(self) -> None:
        s = load_preset("gaussian_default")
        assert s.shape is not None and s.shape.kind is PeakShape.GAUSSIAN

    def test_load_lorentzian_legacy(self) -> None:
        s = load_preset("lorentzian_legacy")
        assert s.shape is not None and s.shape.kind is PeakShape.LORENTZIAN

    def test_load_instrument_bc_2638(self) -> None:
        s = load_preset("instrument_bc_2638")
        assert s.shape is not None and s.shape.kind is PeakShape.GAUSSIAN
        assert s.tau.per_band_tau is True
        # Unspecified knobs stay None so the resolver fills hard defaults.
        assert s.tau.max_decay_factor is None
        assert s.conservative.max_peaks is None

    def test_unknown_bare_name_lists_available(self) -> None:
        with pytest.raises(FileNotFoundError, match=r"no packaged preset"):
            load_preset("does_not_exist")


class TestPathPresetResolution:
    def test_load_from_path(self, tmp_path) -> None:
        p = tmp_path / "my_preset.yaml"
        p.write_text(
            "name: my_preset\n"
            "fit:\n"
            "  shape: gaussian\n"
            "  tau:\n"
            "    max_decay_factor: 4.0\n"
        )
        s = load_preset(p)
        assert s.shape is not None and s.shape.kind is PeakShape.GAUSSIAN
        assert s.tau.max_decay_factor == 4.0

    def test_load_from_str_path(self, tmp_path) -> None:
        p = tmp_path / "my_preset.yaml"
        p.write_text("fit:\n  shape: lorentzian\n")
        s = load_preset(str(p))
        assert s.shape is not None and s.shape.kind is PeakShape.LORENTZIAN

    def test_missing_path_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match=r"preset file not found"):
            load_preset(tmp_path / "missing.yaml")

    def test_flat_yaml_no_fit_wrapper(self, tmp_path) -> None:
        """Presets without a ``fit:`` wrapper still parse."""
        p = tmp_path / "flat.yaml"
        p.write_text("shape: gaussian\ntau:\n  max_decay_factor: 7.0\n")
        s = load_preset(p)
        assert s.shape is not None and s.shape.kind is PeakShape.GAUSSIAN
        assert s.tau.max_decay_factor == 7.0


class TestResolutionWithPreset:
    """Preset + explicit kwarg precedence: explicit wins per field."""

    def test_preset_only(self) -> None:
        preset = load_preset("instrument_bc_2638")
        merged = resolve(preset=preset)
        assert merged.shape is not None and merged.shape.kind is PeakShape.GAUSSIAN
        assert merged.tau.per_band_tau is True
        # Hard default for an unspecified field
        assert merged.tau.max_decay_factor == 5.0

    def test_explicit_overrides_preset_shape(self) -> None:
        """``explicit.shape = LORENTZIAN`` beats the preset's GAUSSIAN."""
        preset = load_preset("instrument_bc_2638")
        explicit = StageFitSettings(shape=ShapeSpec(kind=PeakShape.LORENTZIAN))
        merged = resolve(explicit=explicit, preset=preset)
        assert merged.shape is not None and merged.shape.kind is PeakShape.LORENTZIAN
        # The preset still wins for fields the explicit layer didn't set
        assert merged.tau.per_band_tau is True

    def test_explicit_overrides_preset_field(self) -> None:
        preset = StageFitSettings()
        preset.tau.max_decay_factor = 3.0
        preset.conservative.max_peaks = 6
        explicit = StageFitSettings()
        explicit.tau.max_decay_factor = 1.5
        merged = resolve(explicit=explicit, preset=preset)
        # explicit beats preset for the field both set
        assert merged.tau.max_decay_factor == 1.5
        # preset wins where explicit is unset
        assert merged.conservative.max_peaks == 6
