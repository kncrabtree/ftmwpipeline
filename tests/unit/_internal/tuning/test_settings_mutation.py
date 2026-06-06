"""Unit tests for the settings change-grammar core (issue #28, step 4).

``set_setting`` coercion + persistence + stage invalidation, and
``export_settings`` round-tripping through the stages' ``load_preset`` path --
all against bare HDF5 ``.ftmw`` files with stage groups stamped directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import pytest

from ftmwpipeline._internal.tuning import export_settings, set_setting
from ftmwpipeline._internal.tuning.settings_inspection import (
    SOURCE_FTMW,
    resolve_settings_view,
)
from ftmwpipeline.core import noise_settings as noise_mod
from ftmwpipeline.core import peak_shape as ps_mod
from ftmwpipeline.core import stage_fit_settings as fit_mod
from ftmwpipeline.io.noise_settings_serialization import save_noise_settings_to_h5
from ftmwpipeline.io.stage_fit_settings_serialization import (
    save_stage_fit_settings_to_h5,
)


@pytest.fixture
def bare_ftmw(tmp_path: Path) -> Path:
    p = tmp_path / "bare.ftmw"
    with h5py.File(p, "w"):
        pass
    return p


def _row(file: Path, path: str):
    rows = {r.path: r for r in resolve_settings_view(file, include_advanced=True)}
    return rows[path]


def _stamp_stages(file: Path, completed: list[str]) -> None:
    """Stamp pipeline_stages + a placeholder data group per completed stage."""
    from ftmwpipeline.file_manager import PipelineStageTracker

    paths = PipelineStageTracker.STAGE_DATA_PATHS
    with h5py.File(file, "a") as h5f:
        grp = h5f.require_group("pipeline_stages")
        grp.attrs["completed_stages"] = json.dumps(completed)
        for stage in completed:
            data_path = paths.get(stage, stage)
            if data_path not in h5f:
                h5f.require_group(data_path)


# --- set_setting: coercion + persistence -----------------------------------
def test_set_float_persists_and_reads_back(bare_ftmw: Path) -> None:
    result = set_setting(bare_ftmw, "stage2.window_mhz", "123.5")
    assert result.value == 123.5
    row = _row(bare_ftmw, "stage2.window_mhz")
    assert row.source == SOURCE_FTMW
    assert row.value == 123.5


def test_set_int_and_bool_coercion(bare_ftmw: Path) -> None:
    set_setting(bare_ftmw, "stage2.n_iter", "5")
    set_setting(bare_ftmw, "stage2.region_aware", "false")
    assert _row(bare_ftmw, "stage2.n_iter").value == 5
    # HDF5 reads booleans back as numpy bool; compare by value, not identity.
    assert bool(_row(bare_ftmw, "stage2.region_aware").value) is False


def test_set_subblock_field(bare_ftmw: Path) -> None:
    set_setting(bare_ftmw, "stage5.tau.max_decay_factor", "4.0")
    assert _row(bare_ftmw, "stage5.tau.max_decay_factor").value == 4.0


def test_set_shape(bare_ftmw: Path) -> None:
    result = set_setting(bare_ftmw, "stage5.shape", "gaussian")
    assert result.value.kind is ps_mod.PeakShape.GAUSSIAN
    assert _row(bare_ftmw, "stage5.shape").value.kind is ps_mod.PeakShape.GAUSSIAN


def test_set_unknown_knob_raises(bare_ftmw: Path) -> None:
    with pytest.raises(ValueError, match="unknown setting"):
        set_setting(bare_ftmw, "stage2.not_a_field", "1")


def test_set_unknown_stage_raises(bare_ftmw: Path) -> None:
    with pytest.raises(ValueError, match="unknown settings stage"):
        set_setting(bare_ftmw, "stage9.foo", "1")


# --- set_setting: Stage 1 rules --------------------------------------------
@pytest.mark.parametrize("field", ["zpf", "expf_us", "window_function"])
def test_set_stage1_retired_apodization_knobs_rejected(
    bare_ftmw: Path, field: str
) -> None:
    """The retired apodization knobs no longer exist on FTSettings, so setting
    them is rejected as an unknown field."""
    with pytest.raises(ValueError, match="unknown setting"):
        set_setting(bare_ftmw, f"stage1.{field}", "2")


def test_set_stage1_start_us_persists(bare_ftmw: Path) -> None:
    set_setting(bare_ftmw, "stage1.start_us", "3.25")
    row = _row(bare_ftmw, "stage1.start_us")
    assert row.source == SOURCE_FTMW
    assert row.value == 3.25


# --- set_setting: invalidation ---------------------------------------------
def test_set_invalidates_stage_and_downstream(bare_ftmw: Path) -> None:
    _stamp_stages(
        bare_ftmw,
        [
            "stage0_fid_data",
            "stage1_complex_ft",
            "stage2_noise_result",
            "stage3_peaks",
        ],
    )
    result = set_setting(bare_ftmw, "stage2.window_mhz", "90")
    # Stage 2 itself and its dependent Stage 3 are invalidated.
    assert "stage2_noise_result" in result.invalidated
    assert "stage3_peaks" in result.invalidated
    with h5py.File(bare_ftmw, "r") as h5f:
        completed = json.loads(h5f["pipeline_stages"].attrs["completed_stages"])
        assert "stage2_noise_result" not in completed
        assert "stage3_peaks" not in completed
        assert "stage1_complex_ft" in completed  # upstream untouched
        assert "stage2_noise_result" not in h5f  # results dropped
        assert "stage3_peaks" not in h5f


def test_set_on_unrun_stage_invalidates_nothing(bare_ftmw: Path) -> None:
    _stamp_stages(bare_ftmw, ["stage0_fid_data", "stage1_complex_ft"])
    result = set_setting(bare_ftmw, "stage5.tau.max_decay_factor", "4.0")
    assert result.invalidated == ()


# --- export_settings -------------------------------------------------------
def test_export_round_trips_via_load_preset(bare_ftmw: Path, tmp_path: Path) -> None:
    save_noise_settings_to_h5(
        str(bare_ftmw), noise_mod.NoiseSettings(window_mhz=77.0, line_k=9.0)
    )
    fit = fit_mod.StageFitSettings(shape=fit_mod.ShapeSpec(ps_mod.PeakShape.GAUSSIAN))
    fit.tau.max_decay_factor = 4.0
    save_stage_fit_settings_to_h5(str(bare_ftmw), fit)

    out = tmp_path / "preset.yml"
    result = export_settings(bare_ftmw, out)
    assert "stage2.window_mhz" in result.paths
    assert "stage5.shape" in result.paths

    noise = noise_mod.load_preset(out)
    assert noise.window_mhz == 77.0
    assert noise.line_k == 9.0
    loaded_fit = fit_mod.load_preset(out)
    assert loaded_fit.shape.kind is ps_mod.PeakShape.GAUSSIAN
    assert loaded_fit.tau.max_decay_factor == 4.0


def test_export_selector_scopes_blocks(bare_ftmw: Path, tmp_path: Path) -> None:
    save_noise_settings_to_h5(str(bare_ftmw), noise_mod.NoiseSettings(window_mhz=77.0))
    fit = fit_mod.StageFitSettings()
    fit.tau.max_decay_factor = 4.0
    save_stage_fit_settings_to_h5(str(bare_ftmw), fit)

    out = tmp_path / "preset.yml"
    result = export_settings(bare_ftmw, out, "stage5")
    assert all(p.startswith("stage5.") for p in result.paths)
    noise = noise_mod.load_preset(out)
    assert noise.window_mhz is None  # stage2 block excluded by selector


def test_export_nothing_persisted_writes_empty(bare_ftmw: Path, tmp_path: Path) -> None:
    out = tmp_path / "preset.yml"
    result = export_settings(bare_ftmw, out, name="empty")
    assert result.paths == ()
    assert out.exists()
