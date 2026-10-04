"""Typed ``SettingRow`` fields: type, nullable, units, choices, bounds."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import Pipeline
from ftmwpipeline._internal.tuning import resolve_settings_view
from ftmwpipeline._internal.tuning.settings_inspection import (
    SETTING_TYPES,
    _setting_type,
)
from ftmwpipeline._internal.tuning.settings_mutation import set_setting
from ftmwpipeline.cli.main import main
from ftmwpipeline.serialize import to_jsonable


@pytest.fixture
def rows():
    return {r.path: r for r in resolve_settings_view(None, include_advanced=True)}


def test_every_row_has_a_contract_type(rows):
    assert {r.type for r in rows.values()} <= set(SETTING_TYPES)
    assert rows["stage2.window_mhz"].type == "float"
    assert rows["stage2.n_iter"].type == "int"
    assert rows["stage2.region_aware"].type == "bool"
    assert rows["stage1.trim"].type == "float_pair"
    assert rows["stage2b.gaussian.tau_G_seeds"].type == "float_list"
    assert rows["stage2b.band.band_labels"].type == "str_list"
    assert rows["stage3.primary_pass.primary_window"].type == "str"
    assert rows["stage5.shape"].type == "shape_spec"
    assert rows["stage5.spur.clocks"].type == "clock_sources"


def test_choices_units_bounds_only_where_stated(rows):
    n_eff = rows["stage5.conservative.n_eff_kind"]
    assert n_eff.type == "choice" and "kish_mag" in (n_eff.choices or [])
    assert [p for p, r in rows.items() if r.choices] == [n_eff.path]
    assert rows["stage2.window_mhz"].units == "MHz"
    assert rows["stage1.start_us"].units == "us"
    assert rows["stage1.trim"].units == "MHz"
    assert rows["stage2.n_iter"].units is None
    assert all(r.bounds is None for r in rows.values())


def test_nullable_follows_the_hard_default(rows):
    assert rows["stage1.trim"].nullable
    assert not rows["stage2.window_mhz"].nullable
    assert all(r.nullable == (r.hard_default is None) for r in rows.values())


def test_rows_are_typed_json_and_round_trip(tmp_path: Path, rows):
    f = tmp_path / "x.ftmw"
    with h5py.File(f, "w"):
        pass
    payload = json.loads(json.dumps(to_jsonable(list(rows.values()))))
    by = {r["path"]: r for r in payload}
    assert by["stage5.shape"]["value"] == {"kind": "lorentzian"}
    assert by["stage5.spur.clocks"]["hard_default"] == []
    set_setting(str(f), "stage5.shape", by["stage5.shape"]["value"])
    set_setting(
        str(f),
        "stage5.spur.clocks",
        [{"freq_mhz": 5760.0, "locked": True, "label": "a"}],
    )
    got = {r.path: r for r in resolve_settings_view(str(f), "stage5.spur.clocks")}
    assert to_jsonable(got["stage5.spur.clocks"].value)[0]["freq_mhz"] == 5760.0


_TYPING_KEYS = {"type", "nullable", "units", "choices", "bounds"}
_N_EFF = "stage5.conservative.n_eff_kind"


def test_unknown_annotation_fails_loudly():
    # Mutation: fall back to "str" for an unmapped annotation -> no TypeError.
    with pytest.raises(TypeError):
        _setting_type(dict, False, "x.y")


def test_typing_fields_agree_across_api_pipeline_and_cli(capsys):
    # Mutation: drop the typing fields from one interface's rows (or build
    # the CLI items by hand) -> the three interfaces differ.
    via_api = to_jsonable(list(ftmw.settings_defaults("stage2")))
    via_pipe = to_jsonable(list(Pipeline.settings_defaults("stage2")))
    assert via_api == via_pipe
    assert all(_TYPING_KEYS <= set(r) for r in via_api)

    assert main(["read", "settings_defaults", "stage2", "--format", "json"]) == 0
    items = json.loads(capsys.readouterr().out)["items"]
    assert items == via_api
    win = next(r for r in items if r["path"] == "stage2.window_mhz")
    assert (win["type"], win["units"]) == ("float", "MHz")


def test_choice_row_through_api_and_pipeline():
    # Mutation: drop the n_eff_kind choices declaration.
    for fn in (ftmw.settings_defaults, Pipeline.settings_defaults):
        (row,) = fn(_N_EFF, include_advanced=True)
        d = to_jsonable(row)
        assert d["type"] == "choice" and "kish_mag" in d["choices"]


def test_cli_row_reports_null_for_unstated_metadata(capsys):
    # Mutation: report a guessed unit/bounds/choices for an unannotated field.
    argv = ["read", "settings_defaults", "stage2b.stft.n_seg", "--format", "json"]
    assert main(argv) == 0
    (item,) = json.loads(capsys.readouterr().out)["items"]
    assert item["type"] == "int"
    assert item["units"] is None and item["bounds"] is None
    assert item["choices"] is None and item["nullable"] is False
