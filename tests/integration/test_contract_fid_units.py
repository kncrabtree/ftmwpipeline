"""``fid_samples`` and ``display_units``: the spec's promises, on all interfaces.

Spec: ``dev-docs/CONTRACT_STRATEGY.md`` §FID samples and §Display units, plus
§Accessors (schemas, read-only) and §Errors. Real 2638 data is used where the
promise is about the stored samples or the Stage 1 chain; hand-edited copies
cover a narrower stored dtype, an unset ``units_power`` and the refusals.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import MANIFEST, Pipeline
from ftmwpipeline._internal.read_impl import fid_samples_impl
from ftmwpipeline._internal.stage5_impl import _load_display_style
from ftmwpipeline.cli.main import main
from ftmwpipeline.contract import DISPLAY_UNITS_SCHEMA, FID_SAMPLES_SCHEMA
from ftmwpipeline.core.settings import _HARD_DEFAULTS
from ftmwpipeline.file_manager import (
    PipelineCorruptionError,
    PipelineFileNotFoundError,
    StageDependencyError,
)

pytestmark = [pytest.mark.integration]

FID_PATH = "stage0_fid_data/time_series_data"
FT_GROUP = "processing_parameters/ft_processing"
REC_GROUP = "stage0_fid_data/recommended_processing"


def _md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _cli_json(argv, capsys):
    rc = main(argv)
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


@pytest.fixture(scope="module")
def stage0_file(exp_2638_data_path, tmp_path_factory):
    """2638 imported (Stage 0 only). Read-only reference; copy to mutate."""
    fp = tmp_path_factory.mktemp("fid_units_stage0") / "stage0.ftmw"
    ftmw.import_data(fp, source=exp_2638_data_path)
    return fp


@pytest.fixture
def stage0_copy(stage0_file, tmp_path):
    fp = tmp_path / "copy.ftmw"
    shutil.copy(stage0_file, fp)
    return fp


# ---------------------------------------------------------------- fid_samples


def test_fid_samples_equals_stored_values_in_order(stage0_file):
    res = ftmw.fid_samples(stage0_file)
    with h5py.File(stage0_file, "r") as f:
        stored = f[FID_PATH][...]
        stored_dtype = str(f[FID_PATH].dtype)
    assert set(res) == {"samples", "stored_dtype"}
    assert isinstance(res["samples"], np.ndarray)
    assert res["samples"].dtype == np.float64 and res["samples"].ndim == 1
    np.testing.assert_array_equal(res["samples"], stored)
    assert res["stored_dtype"] == stored_dtype == "float64"


def test_fid_samples_unchanged_by_later_stages(stage0_file, baseline_2638_stage1):
    # Write-once: nothing after import rewrites the samples. The Stage 1
    # baseline is a separate import of the same experiment.
    np.testing.assert_array_equal(
        ftmw.fid_samples(stage0_file)["samples"],
        ftmw.fid_samples(baseline_2638_stage1)["samples"],
    )


def test_fid_samples_is_not_windowed_scaled_or_mean_removed(stage0_file):
    samples = ftmw.fid_samples(stage0_file)["samples"]
    with h5py.File(stage0_file, "r") as f:
        n_points = int(f["stage0_fid_data/acquisition"].attrs["n_points"])
    assert samples.size == n_points
    # A mean-removed array would have a (numerically) zero mean; the stored one
    # is compared sample-for-sample above, this guards the whole-record length.
    assert np.isfinite(samples).all()


def test_fid_samples_promotes_narrower_dtype_losslessly(stage0_copy):
    with h5py.File(stage0_copy, "a") as f:
        original = f[FID_PATH][...]
        narrow = original.astype(np.float32)
        attrs = dict(f[FID_PATH].attrs)
        del f[FID_PATH]
        ds = f.create_dataset(FID_PATH, data=narrow)
        for k, v in attrs.items():
            ds.attrs[k] = v
    res = fid_samples_impl(stage0_copy)
    assert res["stored_dtype"] == "float32"
    assert res["samples"].dtype == np.float64
    np.testing.assert_array_equal(res["samples"], narrow.astype(np.float64))
    # Through the public interface as well.
    assert ftmw.fid_samples(stage0_copy)["stored_dtype"] == "float32"


def test_fid_samples_result_does_not_alias_the_file(stage0_copy):
    res = ftmw.fid_samples(stage0_copy)
    res["samples"][:] = 0.0  # mutating the result must be harmless
    again = ftmw.fid_samples(stage0_copy)["samples"]
    assert np.any(again != 0.0)


def test_fid_samples_read_leaves_file_byte_identical(stage0_copy):
    before = _md5(stage0_copy)
    ftmw.fid_samples(stage0_copy)
    Pipeline.open(stage0_copy).fid_samples()
    assert _md5(stage0_copy) == before


def test_fid_samples_missing_dataset_raises_stage_dependency(stage0_copy):
    with h5py.File(stage0_copy, "a") as f:
        del f[FID_PATH]
    with pytest.raises(StageDependencyError) as info:
        ftmw.fid_samples(stage0_copy)
    err = info.value
    assert err.code == "stage_not_run"
    assert err.missing_dependencies == ["stage0_fid_data"]
    assert err.to_dict()["missing_dependencies"] == ["data"]


def test_fid_samples_missing_stage0_group_raises_stage_dependency(stage0_copy):
    with h5py.File(stage0_copy, "a") as f:
        del f["stage0_fid_data"]
    with pytest.raises(StageDependencyError):
        ftmw.fid_samples(stage0_copy)


def test_fid_samples_bare_stamped_file_raises_stage_dependency(tmp_path):
    bare = tmp_path / "bare.ftmw"
    with h5py.File(bare, "w") as f:
        f.attrs["ftmw_format_version"] = "1.0"
    with pytest.raises(StageDependencyError) as info:
        fid_samples_impl(bare)
    assert info.value.to_dict()["missing_dependencies"] == ["data"]


def test_fid_samples_missing_file_is_typed_not_found(tmp_path):
    missing = tmp_path / "nope.ftmw"
    with pytest.raises(PipelineFileNotFoundError) as info:
        ftmw.fid_samples(missing)
    assert info.value.code == "not_found"
    assert not missing.exists()  # a read never creates the file


def test_fid_samples_corrupt_file_is_typed(tmp_path):
    junk = tmp_path / "junk.ftmw"
    junk.write_bytes(b"not hdf5" * 100)
    with pytest.raises(PipelineCorruptionError):
        ftmw.fid_samples(junk)


# -------------------------------------------------------------- display_units


def test_display_units_before_stage1_matches_the_import_recommendation(stage0_file):
    res = ftmw.display_units(stage0_file)
    assert set(res) == {"amplitude_scale", "units_label", "units_power"}
    with h5py.File(stage0_file, "r") as f:
        assert FT_GROUP not in f  # genuinely before Stage 1
        power = int(f[REC_GROUP].attrs["units_power"])
    assert res["units_power"] == power
    assert res["amplitude_scale"] == 10.0**power
    assert isinstance(res["amplitude_scale"], float)
    assert isinstance(res["units_label"], str) and res["units_label"]


def test_display_units_matches_compute_display_ft_after_stage1(baseline_2638_stage1):
    units = ftmw.display_units(baseline_2638_stage1)
    meta = ftmw.compute_display_ft(baseline_2638_stage1).metadata
    assert units["amplitude_scale"] == meta["amplitude_scale"]
    assert units["units_label"] == meta["units_label"]


def test_display_units_before_stage1_equals_the_value_stage1_adopts(
    stage0_file, baseline_2638_stage1
):
    # The pre-Stage-1 guarantee: the Stage 1 chain already names the pair that
    # Stage 1 later persists (compute_display_ft itself needs Stage 1).
    before = ftmw.display_units(stage0_file)
    after = ftmw.display_units(baseline_2638_stage1)
    assert before == after


def test_display_units_before_stage1_equals_the_display_style_chain(stage0_file):
    scale, label, _trim = _load_display_style(str(stage0_file))
    res = ftmw.display_units(stage0_file)
    assert (res["amplitude_scale"], res["units_label"]) == (scale, label)


@pytest.mark.parametrize(
    "power,label", [(0, "V"), (3, "mV"), (6, "µV"), (9, "nV"), (12, "pV")]
)
def test_display_units_follows_persisted_power_and_matches_display_ft(
    stage0_file, tmp_path, power, label
):
    fp = tmp_path / f"p{power}.ftmw"
    shutil.copy(stage0_file, fp)
    ftmw.compute_ft(fp, trim=(26500, 40000), units_power=power)
    units = ftmw.display_units(fp)
    assert units == {
        "amplitude_scale": 10.0**power,
        "units_label": label,
        "units_power": power,
    }
    meta = ftmw.compute_display_ft(fp).metadata
    assert units["amplitude_scale"] == meta["amplitude_scale"]
    assert units["units_label"] == meta["units_label"]


def test_display_units_stays_equal_through_later_stages(baseline_2638_stage3):
    units = ftmw.display_units(baseline_2638_stage3)
    meta = ftmw.compute_display_ft(baseline_2638_stage3).metadata
    assert units["amplitude_scale"] == meta["amplitude_scale"]
    assert units["units_label"] == meta["units_label"]


def test_display_units_unset_everywhere_falls_to_hard_default(stage0_copy, capsys):
    # The chain is persisted > recommended > hard default, so with no layer
    # naming a power the hard default answers: units_power is a plain int, never
    # an Absent marker, and the pair still equals the Stage 1 chain's.
    with h5py.File(stage0_copy, "a") as f:
        for group in (FT_GROUP, REC_GROUP):
            if group in f and "units_power" in f[group].attrs:
                del f[group].attrs["units_power"]
    res = ftmw.display_units(stage0_copy)
    scale, label, _trim = _load_display_style(str(stage0_copy))
    assert (res["amplitude_scale"], res["units_label"]) == (scale, label)
    assert res["units_power"] == _HARD_DEFAULTS["units_power"]
    assert res["amplitude_scale"] == 10.0 ** res["units_power"]
    rc, out, _ = _cli_json(["read", "display_units", str(stage0_copy)], capsys)
    wire = json.loads(out)
    assert rc == 0 and wire["units_power"] == res["units_power"]
    assert "units_power_absent" not in wire


def test_display_units_read_leaves_file_byte_identical(baseline_2638_stage1, tmp_path):
    fp = tmp_path / "ro.ftmw"
    shutil.copy(baseline_2638_stage1, fp)
    before = _md5(fp)
    ftmw.display_units(fp)
    Pipeline.open(fp).display_units()
    assert _md5(fp) == before


def test_display_units_before_stage1_read_leaves_file_byte_identical(stage0_copy):
    before = _md5(stage0_copy)
    ftmw.display_units(stage0_copy)
    assert _md5(stage0_copy) == before


def test_display_units_missing_file_is_typed_not_found(tmp_path):
    with pytest.raises(PipelineFileNotFoundError):
        ftmw.display_units(tmp_path / "nope.ftmw")


def test_display_units_corrupt_file_is_typed(tmp_path):
    junk = tmp_path / "junk.ftmw"
    junk.write_bytes(b"not hdf5" * 100)
    with pytest.raises(PipelineCorruptionError):
        ftmw.display_units(junk)


# ------------------------------------------------------------ cross-interface


def test_manifest_declares_both_accessors_and_schemas():
    for name in ("fid_samples", "display_units"):
        assert name in MANIFEST.accessors
        assert MANIFEST.file_bound[name] is True
    assert FID_SAMPLES_SCHEMA == "ftmw/fid_samples@1"
    assert DISPLAY_UNITS_SCHEMA == "ftmw/display_units@1"
    assert FID_SAMPLES_SCHEMA in MANIFEST.schemas
    assert DISPLAY_UNITS_SCHEMA in MANIFEST.schemas


def test_fid_samples_agrees_across_interfaces(baseline_2638_stage1, tmp_path, capsys):
    via_api = ftmw.fid_samples(baseline_2638_stage1)
    via_pipe = Pipeline.open(baseline_2638_stage1).fid_samples()
    out_dir = tmp_path / "npy"
    rc, out, err = _cli_json(
        ["read", "fid_samples", str(baseline_2638_stage1), "-o", str(out_dir)],
        capsys,
    )
    assert rc == 0
    env = json.loads(out)
    assert env["schema"] == FID_SAMPLES_SCHEMA
    assert env["samples"] == "samples.npy"
    assert env["stored_dtype"] == via_api["stored_dtype"] == via_pipe["stored_dtype"]
    via_cli = np.load(out_dir / "samples.npy")
    assert via_cli.dtype == np.float64
    np.testing.assert_array_equal(via_api["samples"], via_pipe["samples"])
    np.testing.assert_array_equal(via_api["samples"], via_cli)
    assert [p.name for p in out_dir.iterdir()] == ["samples.npy"]


def test_fid_samples_cli_without_output_is_a_user_error(stage0_file, capsys):
    rc, out, err = _cli_json(["read", "fid_samples", str(stage0_file)], capsys)
    assert rc == 1 and out == "" and err.strip()


def test_display_units_agrees_across_interfaces(baseline_2638_stage1, capsys):
    via_api = ftmw.display_units(baseline_2638_stage1)
    via_pipe = Pipeline.open(baseline_2638_stage1).display_units()
    rc, out, _ = _cli_json(
        ["read", "display_units", str(baseline_2638_stage1), "--format", "json"],
        capsys,
    )
    assert rc == 0
    via_cli = json.loads(out)
    assert via_cli.pop("schema") == DISPLAY_UNITS_SCHEMA
    assert via_api == via_pipe == via_cli


def test_display_units_agrees_across_interfaces_before_stage1(stage0_file, capsys):
    via_api = ftmw.display_units(stage0_file)
    rc, out, _ = _cli_json(["read", "display_units", str(stage0_file)], capsys)
    via_cli = json.loads(out)
    via_cli.pop("schema")
    assert rc == 0 and via_api == via_cli == Pipeline.open(stage0_file).display_units()


def test_cli_refusals_are_typed_error_json_with_exit_codes(
    stage0_copy, tmp_path, capsys
):
    # Missing file: not_found / kind file, exit 1.
    rc, out, err = _cli_json(
        ["read", "display_units", str(tmp_path / "gone.ftmw")], capsys
    )
    payload = json.loads(err)
    assert rc == 1 and out == ""
    assert payload["schema"] == "ftmw/error@1" and payload["code"] == "not_found"
    assert payload["kind"] == "file"

    # Corrupt file: file_corrupt, exit 2.
    junk = tmp_path / "junk.ftmw"
    junk.write_bytes(b"not hdf5" * 100)
    rc, out, err = _cli_json(["read", "display_units", str(junk)], capsys)
    assert rc == 2 and json.loads(err)["code"] == "file_corrupt"

    # No Stage 0 FID: stage_not_run naming the canonical stage, exit 1.
    with h5py.File(stage0_copy, "a") as f:
        del f[FID_PATH]
    rc, out, err = _cli_json(
        ["read", "fid_samples", str(stage0_copy), "-o", str(tmp_path / "o")], capsys
    )
    payload = json.loads(err)
    assert rc == 1 and out == ""
    assert payload["code"] == "stage_not_run"
    assert payload["missing_dependencies"] == ["data"]
    assert not (tmp_path / "o").exists()
