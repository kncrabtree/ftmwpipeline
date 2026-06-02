"""
Unit tests for ``load_preset`` (bare-name + path resolution, wrapper-block
handling for ``fit:`` ↔ ``stage5:``, ``stage2b:`` sibling-block
coexistence, error surfaces) and for the preset/explicit kwarg precedence
inside ``resolve()``.

These verify the Step 4 surface: ``ftmwpipeline fit-peaks --preset
gaussian_default exp.ftmw`` works end-to-end on the YAML side, the
back-compat shim for the legacy ``fit:`` wrapper still parses, and the
Stage 2b ``load_preset`` returns the matching ``stage2b:`` block on the
same packaged preset files.
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
from ftmwpipeline.core.noise_settings import (
    load_preset as load_noise_preset,
)
from ftmwpipeline.core.peak_detection_settings import (
    load_preset as load_peak_preset,
)
from ftmwpipeline.core.tau_calibration_settings import (
    load_preset as load_tau_preset,
)
from ftmwpipeline.core.window_planning_settings import (
    load_preset as load_window_preset,
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
        """The 2638 preset is intentionally empty (every former override
        is now a package-wide default or stamped by Stage 2b)."""
        s = load_preset("instrument_bc_2638")
        assert s.is_empty(), (
            "instrument_bc_2638 should carry no Stage 5 overrides; the "
            "shape is auto-recommended and per_band_tau / tau_penalty_lambda "
            "are package-wide defaults"
        )

    def test_unknown_bare_name_lists_available(self) -> None:
        with pytest.raises(FileNotFoundError, match=r"no packaged preset"):
            load_preset("does_not_exist")


class TestPathPresetResolution:
    def test_load_from_path_stage5_wrapper(self, tmp_path) -> None:
        p = tmp_path / "my_preset.yaml"
        p.write_text(
            "name: my_preset\n"
            "stage5:\n"
            "  shape: gaussian\n"
            "  tau:\n"
            "    max_decay_factor: 4.0\n"
        )
        s = load_preset(p)
        assert s.shape is not None and s.shape.kind is PeakShape.GAUSSIAN
        assert s.tau.max_decay_factor == 4.0

    def test_load_from_path_legacy_fit_wrapper(self, tmp_path) -> None:
        """The ``fit:`` wrapper is a back-compat shim for pre-rename presets.

        Loading a ``fit:``-wrapped preset still parses correctly but emits
        a ``DeprecationWarning`` directing the author to the new
        ``stage5:`` spelling.
        """
        p = tmp_path / "legacy.yaml"
        p.write_text(
            "name: legacy_preset\n"
            "fit:\n"
            "  shape: gaussian\n"
            "  tau:\n"
            "    max_decay_factor: 4.0\n"
        )
        with pytest.warns(DeprecationWarning, match="'fit:' wrapper is deprecated"):
            s = load_preset(p)
        assert s.shape is not None and s.shape.kind is PeakShape.GAUSSIAN
        assert s.tau.max_decay_factor == 4.0

    def test_fit_and_stage5_both_present_raises(self, tmp_path) -> None:
        """A preset must not declare both wrappers."""
        p = tmp_path / "ambiguous.yaml"
        p.write_text(
            "fit:\n  shape: gaussian\n"
            "stage5:\n  shape: lorentzian\n"
        )
        with pytest.raises(ValueError, match=r"both 'fit:' .* and 'stage5:'"):
            load_preset(p)

    def test_load_from_str_path(self, tmp_path) -> None:
        p = tmp_path / "my_preset.yaml"
        p.write_text("stage5:\n  shape: lorentzian\n")
        s = load_preset(str(p))
        assert s.shape is not None and s.shape.kind is PeakShape.LORENTZIAN

    def test_missing_path_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match=r"preset file not found"):
            load_preset(tmp_path / "missing.yaml")

    def test_flat_yaml_no_wrapper(self, tmp_path) -> None:
        """Presets without a wrapper still parse as Stage 5 settings."""
        p = tmp_path / "flat.yaml"
        p.write_text("shape: gaussian\ntau:\n  max_decay_factor: 7.0\n")
        s = load_preset(p)
        assert s.shape is not None and s.shape.kind is PeakShape.GAUSSIAN
        assert s.tau.max_decay_factor == 7.0

    def test_sibling_stage2b_block_ignored_by_stage5_loader(self, tmp_path) -> None:
        """A ``stage2b:`` sibling block must not trip Stage 5's unknown-key gate."""
        p = tmp_path / "two_blocks.yaml"
        p.write_text(
            "stage2b:\n  stft:\n    n_seg: 8\n"
            "stage5:\n  shape: gaussian\n"
        )
        s = load_preset(p)
        assert s.shape is not None and s.shape.kind is PeakShape.GAUSSIAN


class TestStage2bPresetResolution:
    """The Stage 2b ``load_preset`` reads the ``stage2b:`` block from the
    same packaged preset files Stage 5 uses; absence is not an error."""

    def test_packaged_presets_load_empty_when_no_stage2b_block(self) -> None:
        """The three packaged presets carry no ``stage2b:`` block today."""
        for name in ("gaussian_default", "lorentzian_legacy", "instrument_bc_2638"):
            s = load_tau_preset(name)
            assert s.is_empty(), (
                f"packaged preset {name!r} should produce an empty "
                f"TauCalibrationSettings until a stage2b: block lands"
            )

    def test_stage2b_block_populates_dataclass(self, tmp_path) -> None:
        p = tmp_path / "with_stage2b.yaml"
        p.write_text(
            "name: example\n"
            "stage2b:\n"
            "  stft:\n    n_seg: 8\n  polish:\n    polish_snr_cap: 12.0\n"
            "  gaussian:\n    snr_min: 30.0\n"
            "stage5:\n  shape: gaussian\n"
        )
        ts = load_tau_preset(p)
        assert ts.stft.n_seg == 8
        assert ts.polish.polish_snr_cap == 12.0
        assert ts.gaussian.snr_min == 30.0

    def test_stage2b_block_must_be_mapping(self, tmp_path) -> None:
        p = tmp_path / "bad_stage2b.yaml"
        p.write_text("stage2b: 3\n")
        with pytest.raises(ValueError, match=r"'stage2b' block must be a mapping"):
            load_tau_preset(p)

    def test_stage2b_loader_path_resolution(self, tmp_path) -> None:
        p = tmp_path / "stage2b_only.yaml"
        p.write_text("stage2b:\n  stft:\n    n_seg: 4\n")
        ts = load_tau_preset(p)
        assert ts.stft.n_seg == 4

    def test_stage2b_missing_path_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match=r"preset file not found"):
            load_tau_preset(tmp_path / "missing.yaml")


class TestStage2PresetResolution:
    """The Stage 2 ``load_preset`` reads the ``stage2:`` block from the
    same packaged preset files Stages 5 and 2b use; absence is not an error."""

    def test_packaged_presets_load_empty_when_no_stage2_block(self) -> None:
        for name in ("gaussian_default", "lorentzian_legacy", "instrument_bc_2638"):
            s = load_noise_preset(name)
            assert s.is_empty(), (
                f"packaged preset {name!r} should produce an empty "
                f"NoiseSettings until a stage2: block lands"
            )

    def test_stage2_block_populates_dataclass(self, tmp_path) -> None:
        p = tmp_path / "with_stage2.yaml"
        p.write_text(
            "name: example\n"
            "stage2:\n"
            "  window_mhz: 120.0\n  line_k: 6.0\n"
            "stage5:\n  shape: gaussian\n"
        )
        ns = load_noise_preset(p)
        assert ns.window_mhz == 120.0
        assert ns.line_k == 6.0

    def test_stage2_block_must_be_mapping(self, tmp_path) -> None:
        p = tmp_path / "bad_stage2.yaml"
        p.write_text("stage2: 3\n")
        with pytest.raises(ValueError, match=r"'stage2' block must be a mapping"):
            load_noise_preset(p)

    def test_stage2_loader_path_resolution(self, tmp_path) -> None:
        p = tmp_path / "stage2_only.yaml"
        p.write_text("stage2:\n  window_mhz: 50.0\n")
        ns = load_noise_preset(p)
        assert ns.window_mhz == 50.0

    def test_stage2_missing_path_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match=r"preset file not found"):
            load_noise_preset(tmp_path / "missing.yaml")

    def test_sibling_stage2_block_ignored_by_stage5_loader(self, tmp_path) -> None:
        """A ``stage2:`` sibling block must not trip Stage 5's unknown-key gate."""
        p = tmp_path / "two_blocks.yaml"
        p.write_text(
            "stage2:\n  window_mhz: 120.0\n"
            "stage5:\n  shape: gaussian\n"
        )
        s = load_preset(p)
        assert s.shape is not None and s.shape.kind is PeakShape.GAUSSIAN


class TestStage3PresetResolution:
    """The Stage 3 ``load_preset`` reads the ``stage3:`` block from the
    same packaged preset files Stages 2, 2b, and 5 use; absence is not an
    error."""

    def test_packaged_presets_load_empty_when_no_stage3_block(self) -> None:
        for name in ("gaussian_default", "lorentzian_legacy", "instrument_bc_2638"):
            s = load_peak_preset(name)
            assert s.is_empty(), (
                f"packaged preset {name!r} should produce an empty "
                f"PeakDetectionSettings until a stage3: block lands"
            )

    def test_stage3_block_populates_dataclass(self, tmp_path) -> None:
        p = tmp_path / "with_stage3.yaml"
        p.write_text(
            "name: example\n"
            "stage3:\n"
            "  promotion:\n    min_snr: 4.0\n    weak_medium_snr: 12.0\n"
            "  savgol:\n    sg_window: 13\n  primary_pass:\n    primary_window: blackman\n"
            "stage5:\n  shape: gaussian\n"
        )
        ps = load_peak_preset(p)
        assert ps.promotion.min_snr == 4.0
        assert ps.promotion.weak_medium_snr == 12.0
        assert ps.savgol.sg_window == 13
        assert ps.primary_pass.primary_window == "blackman"

    def test_stage3_block_must_be_mapping(self, tmp_path) -> None:
        p = tmp_path / "bad_stage3.yaml"
        p.write_text("stage3: 3\n")
        with pytest.raises(ValueError, match=r"'stage3' block must be a mapping"):
            load_peak_preset(p)

    def test_stage3_loader_path_resolution(self, tmp_path) -> None:
        p = tmp_path / "stage3_only.yaml"
        p.write_text("stage3:\n  savgol:\n    sg_window: 17\n")
        ps = load_peak_preset(p)
        assert ps.savgol.sg_window == 17

    def test_stage3_missing_path_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match=r"preset file not found"):
            load_peak_preset(tmp_path / "missing.yaml")

    def test_sibling_stage3_block_ignored_by_stage5_loader(self, tmp_path) -> None:
        """A ``stage3:`` sibling block must not trip Stage 5's unknown-key gate."""
        p = tmp_path / "two_blocks.yaml"
        p.write_text(
            "stage3:\n  promotion:\n    min_snr: 4.0\n"
            "stage5:\n  shape: gaussian\n"
        )
        s = load_preset(p)
        assert s.shape is not None and s.shape.kind is PeakShape.GAUSSIAN


class TestStage4PresetResolution:
    """The Stage 4 ``load_preset`` reads the ``stage4:`` block from the
    same packaged preset files Stages 2, 2b, 3, and 5 use; absence is not
    an error."""

    def test_packaged_presets_load_empty_when_no_stage4_block(self) -> None:
        for name in ("gaussian_default", "lorentzian_legacy", "instrument_bc_2638"):
            s = load_window_preset(name)
            assert s.is_empty(), (
                f"packaged preset {name!r} should produce an empty "
                f"WindowPlanningSettings until a stage4: block lands"
            )

    def test_stage4_block_populates_dataclass(self, tmp_path) -> None:
        p = tmp_path / "with_stage4.yaml"
        p.write_text(
            "name: example\n"
            "stage4:\n"
            "  coherence:\n    edge_m: 128\n    edge_threshold: 10.0\n"
            "  clustering:\n    max_window_width_mhz: 60.0\n"
            "  leakage:\n    tau_us: 5.0\n"
            "stage5:\n  shape: gaussian\n"
        )
        ws = load_window_preset(p)
        assert ws.coherence.edge_m == 128
        assert ws.coherence.edge_threshold == 10.0
        assert ws.clustering.max_window_width_mhz == 60.0
        assert ws.leakage.tau_us == 5.0

    def test_stage4_block_must_be_mapping(self, tmp_path) -> None:
        p = tmp_path / "bad_stage4.yaml"
        p.write_text("stage4: 3\n")
        with pytest.raises(ValueError, match=r"'stage4' block must be a mapping"):
            load_window_preset(p)

    def test_stage4_loader_path_resolution(self, tmp_path) -> None:
        p = tmp_path / "stage4_only.yaml"
        p.write_text("stage4:\n  contributor:\n    min_freeze_snr: 25.0\n")
        ws = load_window_preset(p)
        assert ws.contributor.min_freeze_snr == 25.0

    def test_stage4_missing_path_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match=r"preset file not found"):
            load_window_preset(tmp_path / "missing.yaml")

    def test_sibling_stage4_block_ignored_by_stage5_loader(self, tmp_path) -> None:
        """A ``stage4:`` sibling block must not trip Stage 5's unknown-key gate."""
        p = tmp_path / "two_blocks.yaml"
        p.write_text(
            "stage4:\n  coherence:\n    edge_m: 32\n"
            "stage5:\n  shape: gaussian\n"
        )
        s = load_preset(p)
        assert s.shape is not None and s.shape.kind is PeakShape.GAUSSIAN


class TestResolutionWithPreset:
    """Preset + explicit kwarg precedence: explicit wins per field."""

    def test_preset_only(self) -> None:
        preset = load_preset("gaussian_default")
        merged = resolve(preset=preset)
        assert merged.shape is not None and merged.shape.kind is PeakShape.GAUSSIAN
        # per_band_tau is the package-wide hard default (True).
        assert merged.tau.per_band_tau is True
        # Hard default for an unspecified field
        assert merged.tau.max_decay_factor == 5.0

    def test_explicit_overrides_preset_shape(self) -> None:
        """``explicit.shape = LORENTZIAN`` beats the preset's GAUSSIAN."""
        preset = load_preset("gaussian_default")
        explicit = StageFitSettings(shape=ShapeSpec(kind=PeakShape.LORENTZIAN))
        merged = resolve(explicit=explicit, preset=preset)
        assert merged.shape is not None and merged.shape.kind is PeakShape.LORENTZIAN
        # The hard default still fills fields neither layer set.
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
