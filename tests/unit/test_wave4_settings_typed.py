"""Typed ``bad_setting`` refusals, CLI exit map and validate agreement (Wave 4)."""

import json
import pickle

import h5py
import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import Pipeline
from ftmwpipeline.cli import main as _cli_main
from ftmwpipeline.cli.contract_commands import exit_code_for
from ftmwpipeline.core.settings import _parse_trim
from ftmwpipeline.core.stage_fit_settings import ShapeSpec, coerce_clock_sources
from ftmwpipeline.file_manager import (
    AlgorithmFailedError,
    BadSettingError,
    PipelineFileError,
    PipelineFileNotFoundError,
)


@pytest.mark.parametrize(
    "knob, value",
    [
        ("nonsense", 1),  # malformed path
        ("stage9.window_mhz", 1),  # unknown stage
        ("stage2.no_such_field", 1),  # unknown field
        ("stage1.no_such_field", 1),  # unknown stage-1 field
    ],
)
def test_set_setting_unknown_path_is_bad_setting(tmp_path, knob, value):
    from ftmwpipeline._internal.tuning import set_setting

    with pytest.raises(BadSettingError) as ei:
        set_setting(tmp_path / "x.ftmw", knob, value)
    assert ei.value.path == knob
    assert isinstance(ei.value, ValueError)
    assert ei.value.to_dict()["code"] == "bad_setting"


def test_set_setting_wrong_type_is_bad_setting(tmp_path):
    from ftmwpipeline._internal.tuning import set_setting

    with pytest.raises(BadSettingError) as ei:
        set_setting(tmp_path / "x.ftmw", "stage2.window_mhz", "not-a-number")
    assert ei.value.path == "stage2.window_mhz"
    assert ei.value.value == "not-a-number"
    assert "number" in ei.value.expected


def test_set_setting_bad_shape_choice(tmp_path):
    from ftmwpipeline._internal.tuning import set_setting

    with pytest.raises(BadSettingError) as ei:
        set_setting(tmp_path / "x.ftmw", "stage5.shape", "triangle")
    assert ei.value.path == "stage5.shape"
    assert "lorentzian" in ei.value.expected


def test_bad_clocks_and_shape_are_typed():
    with pytest.raises(BadSettingError) as ei:
        coerce_clock_sources([{"freq_mhz": -1}])
    assert ei.value.path == "stage5.spur.clocks"
    with pytest.raises(BadSettingError):
        ShapeSpec.coerce("triangle")


def test_bad_trim_is_typed():
    with pytest.raises(BadSettingError) as ei:
        _parse_trim("40000:26500")
    assert ei.value.path == "stage1.trim"
    with pytest.raises(BadSettingError):
        _parse_trim("abc")


def test_unknown_knob_is_bad_setting_and_key_error():
    from ftmwpipeline._internal.tuning import get_knob

    with pytest.raises(BadSettingError) as ei:
        get_knob("stage2.nope")
    assert isinstance(ei.value, KeyError)
    assert "unknown tuning knob" in str(ei.value)
    assert not str(ei.value).startswith("'")
    again = pickle.loads(pickle.dumps(ei.value))
    assert again.path == "stage2.nope"


def test_preset_content_is_bad_setting(tmp_path):
    from ftmwpipeline.core import noise_settings

    preset = tmp_path / "p.yml"
    preset.write_text("stage2:\n  bogus: 1\n")
    with pytest.raises(BadSettingError) as ei:
        noise_settings.load_preset(str(preset))
    assert ei.value.path == "stage2.bogus"
    preset.write_text("- a\n- b\n")
    with pytest.raises(BadSettingError) as ei:
        noise_settings.load_preset(str(preset))
    assert ei.value.path == "preset"


def test_validate_pipeline_raises_for_unopenable(tmp_path):
    with pytest.raises(PipelineFileNotFoundError):
        ftmw.validate_pipeline(tmp_path / "missing.ftmw")
    bad = tmp_path / "bad.ftmw"
    bad.write_text("not hdf5")
    with pytest.raises(PipelineFileError) as ei:
        ftmw.validate_pipeline(bad)
    assert ei.value.code == "file_corrupt"


def test_exit_code_table():
    assert exit_code_for(AlgorithmFailedError("fit", "x")) == 2
    assert exit_code_for(BadSettingError("a", "b", 1)) == 1


def test_cli_dispatch_maps_typed_errors(tmp_path, capsys):
    missing = str(tmp_path / "missing.ftmw")
    assert _cli_main(["info", missing, "--format", "json"]) == 1
    err = capsys.readouterr().err
    assert json.loads(err.strip().splitlines()[-1])["code"] == "not_found"

    bad = tmp_path / "bad.ftmw"
    bad.write_text("not hdf5")
    assert _cli_main(["info", str(bad)]) == 2
    assert "Error:" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Public-interface reach: api, Pipeline and CLI on a real (bare) file.
# ---------------------------------------------------------------------------


@pytest.fixture
def bare_file(tmp_path):
    """A bare imported ``.ftmw`` (no stage run)."""
    src = tmp_path / "src.h5"
    with h5py.File(src, "w") as f:
        f.attrs["ftmw_input_version"] = 1
        f.attrs["spacing_us"] = 0.002
        f.attrs["probe_freq_mhz"] = 40960.0
        f.create_dataset("fid", data=np.cos(2 * np.pi * np.arange(6325) * 0.01))
    out = tmp_path / "exp.ftmw"
    Pipeline.create(str(out), source=str(src), format_name="ftmw-hdf5")
    return out


_BAD_SETS = [
    ("nonsense", "1"),
    ("stage9.window_mhz", "1"),
    ("stage2.no_such_field", "1"),
    ("stage2.window_mhz", "abc"),
    ("stage5.shape", "triangle"),
    ("stage5.spur.clocks", '[{"freq_mhz": -1}]'),
]


@pytest.mark.parametrize("knob, value", _BAD_SETS)
def test_settings_set_public_calls_raise_bad_setting(bare_file, knob, value):
    # Mutation: reverting settings_mutation to plain ValueError makes the
    # BadSettingError assertion fail while the ValueError one still passes.
    with pytest.raises(BadSettingError) as e_api:
        ftmw.settings_set(bare_file, knob, value)
    with pytest.raises(BadSettingError) as e_pipe:
        Pipeline.open(bare_file).settings_set(knob, value)
    for ei in (e_api, e_pipe):
        assert ei.value.code == "bad_setting"
        assert ei.value.path == knob
        assert ei.value.expected
        assert isinstance(ei.value, ValueError)  # old except clauses still work
    assert e_api.value.to_dict() == e_pipe.value.to_dict()


def test_set_clock_sources_and_scan_run_typed(bare_file):
    # Mutation: coerce_clock_sources / get_knob reverted to built-ins.
    with pytest.raises(BadSettingError) as ei:
        ftmw.set_clock_sources(bare_file, [{"freq_mhz": -5}])
    assert ei.value.path == "stage5.spur.clocks"
    with pytest.raises(BadSettingError) as ei:
        ftmw.scan_run(bare_file, "stage2.nope", grid=[1.0])
    assert ei.value.path == "stage2.nope"
    assert isinstance(ei.value, KeyError)
    with pytest.raises(BadSettingError):
        Pipeline.open(bare_file).scan_run("stage2.nope", grid=[1.0])


def test_preset_content_public_call_is_bad_setting(bare_file, tmp_path):
    # Mutation: preset validation reverted to ValueError.
    preset = tmp_path / "p.yml"
    preset.write_text("- a\n")
    with pytest.raises(BadSettingError) as ei:
        ftmw.settings_show(bare_file, preset=preset)
    assert ei.value.path == "preset"


def test_cli_settings_set_bad_setting_exit_and_json(bare_file, capsys):
    # Mutation: a verb swallowing the typed error into its own message/exit
    # would lose the JSON error object on stderr.
    # ``settings set`` has no --format; the typed error reaches main, which
    # reports it on stderr and exits 1.
    rc = _cli_main(["settings", "set", str(bare_file), "stage2.window_mhz", "abc"])
    assert rc == 1
    assert "Error:" in capsys.readouterr().err
    rc = _cli_main(["settings", "set", str(bare_file), "nonsense", "1"])
    assert rc == 1
    assert "Error:" in capsys.readouterr().err


def test_validate_pipeline_agrees_with_pipeline_on_openable_file(bare_file):
    # Pins the report for a good file (the unopenable case is tested above).
    report = ftmw.validate_pipeline(bare_file)
    assert report == Pipeline.open(bare_file).validate()
    assert report["valid"] is True


def test_cli_exit_map_covers_a_verb_other_than_read(tmp_path, capsys):
    # Mutation: a verb keeping its own 'except Exception: return 1' would
    # exit 1 for a corrupt file here.
    bad = tmp_path / "bad.ftmw"
    bad.write_text("not hdf5")
    # ``info`` opens through the typed opener: file_corrupt exits 2 and
    # not_found exits 1. (Verbs whose stage opens the file with h5py directly
    # do not yet type a corrupt file; see the Wave 7 open-path audit.)
    assert _cli_main(["info", str(bad)]) == 2
    assert _cli_main(["info", str(tmp_path / "no.ftmw")]) == 1
    capsys.readouterr()
