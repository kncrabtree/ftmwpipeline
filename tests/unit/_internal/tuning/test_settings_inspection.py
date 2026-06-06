"""Unit tests for the resolved-settings inspection core (issue #28, step 2).

Exercises ``resolve_settings_view`` against a bare HDF5 ``.ftmw`` whose stage
groups are stamped directly (no expensive pipeline run): the field-enumeration
walk, the per-layer provenance attribution, and — the D11 invariant — that a
persisted value beats a conflicting preset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import h5py
import pytest

from ftmwpipeline._internal.tuning import SettingRow, resolve_settings_view
from ftmwpipeline._internal.tuning.settings_inspection import (
    SOURCE_DEFAULT,
    SOURCE_FTMW,
    SOURCE_PRESET_PREFIX,
    SOURCE_RECOMMENDED,
)
from ftmwpipeline.core import noise_settings as noise_mod
from ftmwpipeline.core import settings as ft_mod
from ftmwpipeline.core.peak_shape import PeakShape
from ftmwpipeline.io.noise_settings_serialization import save_noise_settings_to_h5


@pytest.fixture
def bare_ftmw(tmp_path: Path) -> Path:
    """An empty but valid HDF5 ``.ftmw`` — every stage group absent."""
    p = tmp_path / "bare.ftmw"
    with h5py.File(p, "w"):
        pass
    return p


def _by_path(rows: tuple[SettingRow, ...]) -> Dict[str, SettingRow]:
    return {r.path: r for r in rows}


def test_enumeration_covers_unswept_and_special_fields(bare_ftmw: Path) -> None:
    rows = resolve_settings_view(bare_ftmw, include_advanced=True)
    paths = {r.path for r in rows}
    # The Stage 1 fields the knob registry deliberately omits, plus the
    # headline start_us, the trim tuple, and the Stage 5 shape discriminator.
    for expected in (
        "stage1.units_power",
        "stage1.trim",
        "stage1.rdc",
        "stage1.start_us",
        "stage1.end_us",
        "stage5.shape",
        "stage2.window_mhz",
        "stage2b.gaussian.snr_min",
        "stage3.savgol.sg_window",
        "stage4.leakage.tau_us",
        "stage5.tau.max_decay_factor",
    ):
        assert expected in paths
    # Nothing persisted / preset / recommended: every row resolves to its
    # hard default.
    assert all(r.source == SOURCE_DEFAULT for r in rows)


def test_default_provenance_matches_resolver(bare_ftmw: Path) -> None:
    rows = _by_path(resolve_settings_view(bare_ftmw, include_advanced=True))
    win = rows["stage2.window_mhz"]
    assert win.source == SOURCE_DEFAULT
    assert win.value == noise_mod.resolve().window_mhz
    assert win.hard_default == noise_mod.resolve().window_mhz
    # A field whose hard default is a concrete non-None value still reads default.
    assert rows["stage1.units_power"].value == 6
    # A field with no hard default stays None at the default layer.
    assert rows["stage1.start_us"].value is None


def test_persisted_provenance(bare_ftmw: Path) -> None:
    save_noise_settings_to_h5(str(bare_ftmw), noise_mod.NoiseSettings(window_mhz=137.0))
    win = _by_path(resolve_settings_view(bare_ftmw, include_advanced=True))[
        "stage2.window_mhz"
    ]
    assert win.source == SOURCE_FTMW
    assert win.value == 137.0


def test_persisted_beats_preset_and_preset_seeds(
    bare_ftmw: Path, tmp_path: Path
) -> None:
    # window_mhz is persisted; pedestal_mhz is only in the preset.
    save_noise_settings_to_h5(str(bare_ftmw), noise_mod.NoiseSettings(window_mhz=137.0))
    preset = tmp_path / "inst.yaml"
    preset.write_text(
        "name: inst\nstage2:\n  window_mhz: 999.0\n  pedestal_mhz: 42.0\n"
    )
    rows = _by_path(
        resolve_settings_view(bare_ftmw, include_advanced=True, preset=preset)
    )
    # D11: the persisted .ftmw value wins over the conflicting preset.
    win = rows["stage2.window_mhz"]
    assert win.source == SOURCE_FTMW
    assert win.value == 137.0
    # The preset seeds a field the file has not persisted.
    ped = rows["stage2.pedestal_mhz"]
    assert ped.source == f"{SOURCE_PRESET_PREFIX}{preset}"
    assert ped.value == 42.0


def test_no_preset_means_no_preset_layer(bare_ftmw: Path, tmp_path: Path) -> None:
    # With no preset argument, a field only a preset would set stays default.
    rows = _by_path(resolve_settings_view(bare_ftmw, include_advanced=True))
    assert rows["stage2.pedestal_mhz"].source == SOURCE_DEFAULT


def test_recommended_shape_provenance(bare_ftmw: Path) -> None:
    # The Stage 5 recommended layer is the Stage 2b auto-recommended shape.
    with h5py.File(bare_ftmw, "a") as h5f:
        grp = h5f.require_group("stage2b_tau_calibration")
        grp.attrs["recommended_shape"] = "gaussian"
    shape = _by_path(resolve_settings_view(bare_ftmw, include_advanced=True))[
        "stage5.shape"
    ]
    assert shape.source == SOURCE_RECOMMENDED
    assert shape.value.kind == PeakShape.GAUSSIAN
    # The hard default remains the Lorentzian reference.
    assert shape.hard_default.kind == PeakShape.LORENTZIAN


def test_start_us_recommended_then_persisted(bare_ftmw: Path) -> None:
    """The issue's worked example: start_us reads recommended (detector output)
    until a value is persisted in the file, then reads .ftmw."""
    with h5py.File(bare_ftmw, "a") as h5f:
        h5f.require_group(ft_mod.RECOMMENDED_PATH).attrs["start_us"] = 2.5
    start = _by_path(resolve_settings_view(bare_ftmw, include_advanced=True))[
        "stage1.start_us"
    ]
    assert start.source == SOURCE_RECOMMENDED
    assert start.value == 2.5

    with h5py.File(bare_ftmw, "a") as h5f:
        h5f.require_group(ft_mod.FT_PROCESSING_PATH).attrs["start_us"] = 3.1
    start2 = _by_path(resolve_settings_view(bare_ftmw, include_advanced=True))[
        "stage1.start_us"
    ]
    assert start2.source == SOURCE_FTMW
    assert start2.value == 3.1


def test_selector_and_tier_filtering(bare_ftmw: Path) -> None:
    all_rows = resolve_settings_view(bare_ftmw, include_advanced=True)
    primary = resolve_settings_view(bare_ftmw, include_advanced=False)
    assert 0 < len(primary) < len(all_rows)
    assert all(r.tier == "primary" for r in primary)

    sel = resolve_settings_view(bare_ftmw, "stage5.rescue", include_advanced=True)
    assert sel
    assert all(r.path.startswith("stage5.rescue.") for r in sel)

    sel_primary = resolve_settings_view(bare_ftmw, "stage5.rescue")
    assert all(r.tier == "primary" for r in sel_primary)


def test_registry_enrichment(bare_ftmw: Path) -> None:
    rows = _by_path(resolve_settings_view(bare_ftmw, include_advanced=True))
    # A registered knob carries its tier + one-line help.
    assert rows["stage2.window_mhz"].tier == "primary"
    assert rows["stage2.window_mhz"].help
    # An unswept field absent from the registry is advanced with empty help.
    assert rows["stage1.units_power"].tier == "advanced"
    assert rows["stage1.units_power"].help == ""
