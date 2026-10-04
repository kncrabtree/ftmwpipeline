"""Stage names in status and validation are canonical, on every interface.

``dev-docs/CONTRACT_STRATEGY.md`` §Errors: ``validate_pipeline`` and
``Pipeline.validate`` agree -- a file that cannot be opened raises the typed
open error, an openable file gets a report -- and the report, like
``list_available_stages``, names stages by their canonical names
(``ftmwpipeline.Stage`` values, in re-run order), never by storage key.
"""

from __future__ import annotations

import json
import shutil

import h5py
import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline.cli.main import main
from ftmwpipeline.contract import Stage, key_for_stage
from ftmwpipeline.file_manager import (
    PipelineCorruptionError,
    PipelineFileError,
    PipelineFileNotFoundError,
    rerun_order,
    validate_pipeline_file,
)
from ftmwpipeline.pipeline import Pipeline

pytestmark = [pytest.mark.unit]

_DT_US = 0.002
_N = 6325
_CANONICAL = {stage.value for stage in Stage}
_STORAGE_KEYS = {key_for_stage(stage) for stage in Stage}


@pytest.fixture
def imported(tmp_path):
    """A Stage-0-only file from a small synthetic two-line FID."""
    src = tmp_path / "src.h5"
    t = np.arange(_N) * _DT_US
    rng = np.random.default_rng(0)
    fid = (
        np.cos(2 * np.pi * 37.0 * t) * np.exp(-t / 3.0)
        + np.cos(2 * np.pi * 91.0 * t) * np.exp(-t / 3.0)
        + 0.01 * rng.standard_normal(_N)
    )
    with h5py.File(src, "w") as f:
        f.attrs["ftmw_input_version"] = 1
        f.attrs["spacing_us"] = _DT_US
        f.attrs["probe_freq_mhz"] = 40960.0
        f.create_dataset("fid", data=fid)
    out = tmp_path / "exp.ftmw"
    Pipeline.create(str(out), source=str(src), format_name="ftmw-hdf5")
    return out


@pytest.fixture(scope="module")
def _peaks_template(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("canonical_status")
    src = tmp / "src.h5"
    t = np.arange(_N) * _DT_US
    rng = np.random.default_rng(0)
    fid = (
        np.cos(2 * np.pi * 37.0 * t) * np.exp(-t / 3.0)
        + np.cos(2 * np.pi * 91.0 * t) * np.exp(-t / 3.0)
        + 0.01 * rng.standard_normal(_N)
    )
    with h5py.File(src, "w") as f:
        f.attrs["ftmw_input_version"] = 1
        f.attrs["spacing_us"] = _DT_US
        f.attrs["probe_freq_mhz"] = 40960.0
        f.create_dataset("fid", data=fid)
    out = tmp / "exp.ftmw"
    Pipeline.create(str(out), source=str(src), format_name="ftmw-hdf5")
    ftmw.compute_ft(out)
    ftmw.estimate_noise(out)
    ftmw.detect_peaks(out)
    return out


@pytest.fixture
def peaks_file(_peaks_template, tmp_path):
    out = tmp_path / "peaks.ftmw"
    shutil.copy(_peaks_template, out)
    return out


# ---------------------------------------------------------------------------
# The report names stages canonically
# ---------------------------------------------------------------------------


def test_report_stages_are_canonical_and_in_rerun_order(peaks_file):
    """Mutation: report ``stage_tracker.completed_stages`` / storage keys."""
    stages = validate_pipeline_file(peaks_file)["stages"]
    assert stages["completed_stages"] == ["data", "ft", "noise", "peaks"]
    assert stages["next_available"] == ["tau", "tau_g", "timebase", "windows"]
    for names in stages.values():
        assert set(names) <= _CANONICAL
        assert names == [s for s in rerun_order() if s in names]


def test_a_fresh_file_reports_data_done_and_ft_next(imported):
    stages = validate_pipeline_file(imported)["stages"]
    assert stages == {"completed_stages": ["data"], "next_available": ["ft"]}


def test_report_stage_environments_use_canonical_names(peaks_file):
    report = validate_pipeline_file(peaks_file)
    assert set(report["stage_environments"]) == {"data", "ft", "noise", "peaks"}
    assert not set(report["stage_environments"]) & _STORAGE_KEYS
    # The values are the recorded environments, untouched.
    for env in report["stage_environments"].values():
        assert env["ftmwpipeline"]


def _forge_environment(path, key, **overrides):
    with h5py.File(path, "a") as f:
        group = f["pipeline_stages"]
        blob = json.loads(str(group.attrs["stage_environments"]))
        blob[key] = {**blob[key], **overrides}
        group.attrs["stage_environments"] = json.dumps(blob, sort_keys=True)


def test_environment_drift_names_the_stage_canonically(peaks_file):
    """Mutation: build the drift lines from the storage-keyed map."""
    _forge_environment(peaks_file, "stage3_peaks", ftmwpipeline="9.9.9")
    report = validate_pipeline_file(peaks_file)
    (line,) = report["environment_drift"]
    assert "9.9.9 (peaks)" in line
    assert not any(key in line for key in _STORAGE_KEYS)
    (warning,) = [w for w in report["warnings"] if "different analysis" in w]
    assert "9.9.9 (peaks)" in warning
    assert report["valid"] is True  # drift is a caveat, not corruption


def test_an_environment_key_that_is_no_known_stage_is_kept_as_recorded(peaks_file):
    with h5py.File(peaks_file, "a") as f:
        group = f["pipeline_stages"]
        blob = json.loads(str(group.attrs["stage_environments"]))
        blob["custom_stage"] = dict(blob["stage3_peaks"])
        group.attrs["stage_environments"] = json.dumps(blob, sort_keys=True)
    report = validate_pipeline_file(peaks_file)
    assert set(report["stage_environments"]) == {
        "data",
        "ft",
        "noise",
        "peaks",
        "custom_stage",
    }
    assert report["valid"] is True


def test_missing_data_error_names_the_stage_canonically(peaks_file):
    """Mutation: interpolate the storage key into the error text."""
    with h5py.File(peaks_file, "a") as f:
        del f["stage3_peaks"]
    report = validate_pipeline_file(peaks_file)
    assert report["valid"] is False
    assert report["errors"] == ["Missing data for completed stage: peaks"]
    # The Pipeline class and the functional API return the same report.
    assert Pipeline.open(peaks_file).validate() == report
    assert ftmw.validate_pipeline(peaks_file) == report


def test_an_unknown_completed_key_is_dropped_from_the_canonical_stages(peaks_file):
    """A completed-stage key no stage of this version owns has no canonical
    name: the ``stages`` view omits it, and the integrity check still reports
    its data as missing (under the name the file recorded)."""
    with h5py.File(peaks_file, "a") as f:
        group = f["pipeline_stages"]
        done = json.loads(str(group.attrs["completed_stages"]))
        group.attrs["completed_stages"] = json.dumps(done + ["stage9_future"])
    report = validate_pipeline_file(peaks_file)
    assert report["stages"]["completed_stages"] == ["data", "ft", "noise", "peaks"]
    assert report["errors"] == ["Missing data for completed stage: stage9_future"]


# ---------------------------------------------------------------------------
# Unopenable files raise; they are not reported
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("via", ["file_manager", "api", "pipeline"])
def test_a_missing_file_raises_not_found_on_every_surface(tmp_path, via):
    missing = tmp_path / "gone.ftmw"
    calls = {
        "file_manager": lambda: validate_pipeline_file(missing),
        "api": lambda: ftmw.validate_pipeline(missing),
        "pipeline": lambda: Pipeline.open(missing).validate(),
    }
    with pytest.raises(PipelineFileNotFoundError) as excinfo:
        calls[via]()
    assert excinfo.value.to_dict()["code"] == "not_found"
    assert excinfo.value.to_dict()["kind"] == "file"
    assert not missing.exists()


@pytest.mark.parametrize("via", ["file_manager", "api", "pipeline"])
def test_a_non_hdf5_file_raises_file_corrupt_on_every_surface(tmp_path, via):
    junk = tmp_path / "junk.ftmw"
    junk.write_bytes(b"not hdf5" * 100)
    calls = {
        "file_manager": lambda: validate_pipeline_file(junk),
        "api": lambda: ftmw.validate_pipeline(junk),
        "pipeline": lambda: Pipeline.open(junk).validate(),
    }
    with pytest.raises(PipelineCorruptionError) as excinfo:
        calls[via]()
    assert excinfo.value.to_dict()["code"] == "file_corrupt"
    assert junk.read_bytes() == b"not hdf5" * 100


def test_a_file_that_goes_bad_after_opening_raises_from_pipeline_validate(
    imported,
):
    """``Pipeline.validate`` on an object opened before the file was damaged
    raises the typed error; it does not return ``{"valid": False}``."""
    pipe = Pipeline.open(imported)
    with open(imported, "r+b") as handle:
        handle.truncate(100)
    with pytest.raises(PipelineFileError) as excinfo:
        pipe.validate()
    assert excinfo.value.code == "file_corrupt"
    with pytest.raises(PipelineFileError):
        pipe.info()


# ---------------------------------------------------------------------------
# One answer on every interface
# ---------------------------------------------------------------------------


def test_info_list_stages_and_validate_agree(peaks_file):
    """Mutation: let any one of the three name storage keys again."""
    info = ftmw.get_pipeline_info(peaks_file)
    via_pipeline = Pipeline.open(peaks_file).info()
    report = ftmw.validate_pipeline(peaks_file)
    available = ftmw.list_available_stages(peaks_file)

    completed = ["data", "ft", "noise", "peaks"]
    assert info["completed_stages"] == via_pipeline["completed_stages"] == completed
    assert report["stages"]["completed_stages"] == completed
    assert (
        info["next_available_stages"]
        == via_pipeline["next_available_stages"]
        == report["stages"]["next_available"]
        == available
    )
    assert set(available) <= _CANONICAL
    assert set(info["stage_environments"]) == set(report["stage_environments"])
    assert info["environment_drift"] == report["environment_drift"]


def test_list_available_stages_on_a_fresh_file(imported):
    assert ftmw.list_available_stages(imported) == ["ft"]
    assert Pipeline.open(imported).info()["next_available_stages"] == ["ft"]


def test_workflow_summary_names_and_acts_on_canonical_stages(imported):
    """The summary keyed on the old storage keys, so after the rename it must
    still suggest the next step. Mutation: test for ``stage1_complex_ft``."""
    text = ftmw.workflow_summary(imported)
    assert "Completed: ['data']" in text
    assert "Next available: ['ft']" in text
    assert "stage1_complex_ft" not in text
    assert "ftmw.compute_ft(" in text

    ftmw.compute_ft(imported)
    text = ftmw.workflow_summary(imported)
    assert "Completed: ['data', 'ft']" in text
    assert "ftmw.estimate_noise(" in text


def test_the_cli_info_command_prints_canonical_names(peaks_file, capsys):
    capsys.readouterr()
    assert main(["info", str(peaks_file)]) == 0
    lines = capsys.readouterr().out.splitlines()
    done = next(line for line in lines if line.strip().startswith("completed:"))
    nxt = next(line for line in lines if line.strip().startswith("next available:"))
    assert done.split(":", 1)[1].strip() == "data, ft, noise, peaks"
    assert nxt.split(":", 1)[1].strip() == "tau, tau_g, timebase, windows"

    assert main(["info", str(peaks_file), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["completed_stages"] == ["data", "ft", "noise", "peaks"]
    assert payload["next_available_stages"] == ["tau", "tau_g", "timebase", "windows"]
    assert set(payload["stage_environments"]) == {"data", "ft", "noise", "peaks"}


def test_the_cli_info_command_refuses_an_unopenable_file_with_a_typed_exit(
    tmp_path, capsys
):
    assert main(["info", str(tmp_path / "gone.ftmw")]) == 1
    junk = tmp_path / "junk.ftmw"
    junk.write_bytes(b"not hdf5" * 100)
    assert main(["info", str(junk)]) == 2
    assert not (tmp_path / "gone.ftmw").exists()
