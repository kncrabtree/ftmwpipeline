"""The declared existing surface (CONTRACT_STRATEGY.md §Accessors, "Already
present - declared as contract"): cross-interface parity, read purity, typed
refusals.

Every accessor in ``MANIFEST`` that predates the contract is exercised through
its declared interface triple: ``api.<name>``, ``Pipeline.<pipeline_name>`` and
the one CLI verb ``read <name>``. Nothing here writes a file: the fixture is
the session-scoped post-Stage-6 2638 build (``stage5_reviewed_2638``), read
only, and outputs go to pytest ``tmp_path``. It is reviewed so that
``get_final_products`` and ``review_log`` carry real rows; every check on them
asserts non-empty first.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import MANIFEST, Pipeline, to_jsonable
from ftmwpipeline.cli.main import main
from ftmwpipeline.contract import (
    CALIBRATION_SCHEMA,
    FINAL_PRODUCTS_SCHEMA,
    METADATA_SCHEMA,
    PIPELINE_INFO_SCHEMA,
    REVIEW_LOG_SCHEMA,
)
from ftmwpipeline.file_manager import (
    NotFoundError,
    PipelineCorruptionError,
    PipelineFileNotFoundError,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _cli_json(argv, capsys):
    rc = main(argv)
    cap = capsys.readouterr()
    assert rc == 0, cap.err
    return json.loads(cap.out)


def _jsonable(obj):
    return json.loads(json.dumps(to_jsonable(obj)))


@pytest.fixture
def stage5(stage5_reviewed_2638):
    return str(stage5_reviewed_2638)


# ---- every declared accessor, through all three interfaces ----------------

#: Declared file-bound accessors that take no extra argument.
_PLAIN = [
    "frequency_calibration",
    "refit_snap_tol_mhz",
    "read_metadata",
    "read_tables",
    "get_final_products",
    "review_log",
    "get_pipeline_info",
]


def test_every_declared_existing_accessor_is_covered_here():
    # Subset: later accessors are covered by their own test files.
    covered = set(_PLAIN) | {
        "read_table",
        "settings_show",
        "settings_defaults",
        "compute_display_ft",
        "capabilities",
    }
    assert covered <= set(MANIFEST.accessors)


@pytest.mark.parametrize("name", _PLAIN)
def test_api_and_pipeline_agree(name, stage5):
    via_api = getattr(ftmw, name)(stage5)
    pname = MANIFEST.pipeline_names[name]
    via_pipeline = getattr(Pipeline.open(stage5), pname)()
    assert _jsonable(via_api) == _jsonable(via_pipeline)


def test_the_fixture_carries_real_rows(stage5):
    fp = ftmw.get_final_products(stage5)
    assert fp is not None and fp.peaks
    assert len(ftmw.review_log(stage5)) >= 2


def test_read_table_api_and_pipeline_agree(stage5):
    for table, columns in MANIFEST.tables.items():
        a = ftmw.read_table(stage5, table)
        p = Pipeline.open(stage5).read_table(table)
        assert set(columns) <= set(a)
        assert set(a) == set(p)
        for col in a:
            np.testing.assert_array_equal(a[col], p[col])


def test_settings_api_and_pipeline_agree(stage5):
    assert _jsonable(ftmw.settings_show(stage5)) == _jsonable(
        Pipeline.open(stage5).settings_show()
    )
    assert _jsonable(ftmw.settings_defaults()) == _jsonable(
        Pipeline.settings_defaults()
    )


def test_cli_read_metadata_equals_api(stage5, capsys):
    cli = _cli_json(["read", "read_metadata", stage5], capsys)
    assert cli.pop("schema") == METADATA_SCHEMA
    assert cli == _jsonable(ftmw.read_metadata(stage5))


def test_cli_read_table_equals_api(stage5, tmp_path, capsys):
    for table in MANIFEST.tables:
        out = tmp_path / table
        env = _cli_json(
            ["read", "read_table", stage5, table, "--output", str(out)], capsys
        )
        cols = ftmw.read_table(stage5, table)
        for col in MANIFEST.tables[table]:
            assert env[col] == f"{col}.npy"
            np.testing.assert_array_equal(np.load(out / env[col]), cols[col])


def test_cli_snap_tolerance_equals_api(stage5, capsys):
    cli = _cli_json(["read", "refit_snap_tol_mhz", stage5], capsys)
    assert cli["value"] == ftmw.refit_snap_tol_mhz(stage5)
    assert cli["schema"].startswith("ftmw/snap_tolerance@")


def test_cli_frequency_calibration_equals_api(stage5, capsys):
    cli = _cli_json(["read", "frequency_calibration", stage5], capsys)
    stamp = ftmw.frequency_calibration(stage5)
    assert cli.pop("schema") == CALIBRATION_SCHEMA
    assert cli == _jsonable(dataclasses.asdict(stamp))
    assert set(MANIFEST.fields["CalibrationStamp"]) == set(cli)


def test_cli_pipeline_info_equals_api(stage5, capsys):
    cli = _cli_json(["read", "get_pipeline_info", stage5], capsys)
    info = ftmw.get_pipeline_info(stage5)
    assert cli.pop("schema") == PIPELINE_INFO_SCHEMA
    assert set(cli) == set(info)
    assert cli["completed_stages"] == list(info["completed_stages"])
    for key in MANIFEST.fields["PipelineInfo"]:
        assert key in cli and key in info


def test_cli_final_products_equals_api(stage5, capsys):
    cli = _cli_json(["read", "get_final_products", stage5], capsys)
    fp = ftmw.get_final_products(stage5)
    assert fp is not None and fp.peaks
    assert cli["schema"] == FINAL_PRODUCTS_SCHEMA
    assert len(cli["peaks"]) == len(fp.peaks)
    declared = set(MANIFEST.fields["FinalPeak"])
    for row, peak in zip(cli["peaks"], fp.peaks):
        assert declared <= set(row)
        assert row["frequency_mhz"] == peak.frequency_mhz
        assert row["peak_uid"] == peak.peak_uid


def test_cli_review_log_equals_api(stage5, capsys):
    cli = _cli_json(["read", "review_log", stage5], capsys)
    log = ftmw.review_log(stage5)
    assert log
    assert cli["schema"] == REVIEW_LOG_SCHEMA
    assert [r["kind"] for r in cli["items"]] == [e.kind for e in log]
    declared = set(MANIFEST.fields["DecisionLogEntry"])
    for row in cli["items"]:
        assert declared <= set(row)


def test_display_ft_agrees_across_interfaces(stage5, tmp_path, capsys):
    via_api = ftmw.compute_display_ft(stage5)
    via_pipeline = Pipeline.open(stage5).compute_display_ft()
    out = tmp_path / "ft"
    env = _cli_json(
        [
            "read",
            "compute_display_ft",
            stage5,
            "--format",
            "json",
            "--output",
            str(out),
        ],
        capsys,
    )
    assert env["freq_array"] == "freq_array.npy"
    assert env["complex_spectrum"] == "complex_spectrum.npy"
    for ft in (via_pipeline,):
        np.testing.assert_array_equal(ft.freq_array, via_api.freq_array)
        np.testing.assert_array_equal(ft.complex_spectrum, via_api.complex_spectrum)
    freq = np.load(out / "freq_array.npy")
    spec = np.load(out / "complex_spectrum.npy")
    assert spec.dtype == np.complex128
    np.testing.assert_array_equal(freq, via_api.freq_array)
    np.testing.assert_array_equal(spec, via_api.complex_spectrum)
    assert env["metadata"] == _jsonable(via_api.metadata)
    assert set(env["metadata"]) == set(MANIFEST.fields["ComplexFT.metadata"])
    assert set(env) == set(MANIFEST.fields["ComplexFT"])


def test_display_ft_pad_factor_flag_matches_api(stage5, tmp_path, capsys):
    out = tmp_path / "ft3"
    env = _cli_json(
        [
            "read",
            "compute_display_ft",
            stage5,
            "--pad-factor",
            "3",
            "--format",
            "json",
            "--output",
            str(out),
        ],
        capsys,
    )
    ref = ftmw.compute_display_ft(stage5, pad_factor=3)
    assert env["metadata"]["pad_factor"] == 3
    np.testing.assert_array_equal(np.load(out / "freq_array.npy"), ref.freq_array)


def test_display_ft_without_output_is_a_user_error(stage5, capsys):
    # Arrays are never inlined (spec §Accessors): no --output, no arrays.
    rc = main(["read", "compute_display_ft", stage5, "--format", "json"])
    cap = capsys.readouterr()
    assert rc == 1 and cap.out == ""


def test_display_ft_envelope_has_no_absent_markers(stage5, tmp_path, capsys):
    env = _cli_json(
        [
            "read",
            "compute_display_ft",
            stage5,
            "--format",
            "json",
            "--output",
            str(tmp_path / "a"),
        ],
        capsys,
    )
    assert not any(k.endswith("_absent") for k in env)
    assert not any(k.endswith("_absent") for k in env["metadata"])


# ---- reads never write the file -------------------------------------------


def test_every_declared_read_leaves_the_file_byte_identical(stage5, tmp_path, capsys):
    before = _md5(stage5)
    for name in _PLAIN:
        getattr(ftmw, name)(stage5)
        getattr(Pipeline.open(stage5), MANIFEST.pipeline_names[name])()
    ftmw.read_table(stage5, "fit_peaks")
    ftmw.settings_show(stage5)
    ftmw.settings_defaults()
    ftmw.compute_display_ft(stage5)
    Pipeline.open(stage5).compute_display_ft(pad_factor=3)
    for argv in (
        ["read", "read_metadata", stage5],
        ["read", "read_tables", stage5],
        ["read", "refit_snap_tol_mhz", stage5],
        ["read", "review_log", stage5],
        ["read", "frequency_calibration", stage5],
        ["read", "get_pipeline_info", stage5],
        ["read", "get_final_products", stage5],
        ["read", "settings_show", stage5],
        ["read", "read_table", stage5, "windows", "--output", str(tmp_path / "t")],
        [
            "read",
            "compute_display_ft",
            stage5,
            "--format",
            "json",
            "--output",
            str(tmp_path / "o"),
        ],
    ):
        main(argv)
        capsys.readouterr()
    assert _md5(stage5) == before


# ---- typed refusals -------------------------------------------------------

_FILE_READS = [n for n in _PLAIN if n not in ("refit_snap_tol_mhz",)] + [
    "refit_snap_tol_mhz",
    "compute_display_ft",
]


@pytest.mark.parametrize("name", _FILE_READS)
def test_missing_file_is_a_typed_not_found(name, tmp_path):
    missing = tmp_path / "absent.ftmw"
    with pytest.raises(NotFoundError) as exc:
        getattr(ftmw, name)(missing)
    assert isinstance(exc.value, PipelineFileNotFoundError)
    assert exc.value.to_dict()["code"] == "not_found"
    assert not missing.exists()  # a failed read does not create the file


@pytest.mark.parametrize(
    "name", ["read_metadata", "get_pipeline_info", "compute_display_ft"]
)
def test_non_hdf5_file_is_a_typed_corruption_error(name, tmp_path):
    junk = tmp_path / "junk.ftmw"
    junk.write_bytes(b"not hdf5" * 100)
    before = _md5(junk)
    with pytest.raises(PipelineCorruptionError) as exc:
        getattr(ftmw, name)(junk)
    assert exc.value.to_dict()["code"] == "file_corrupt"
    assert _md5(junk) == before


def test_missing_file_cli_reports_error_json_on_stderr(tmp_path, capsys):
    rc = main(
        [
            "read",
            "compute_display_ft",
            str(tmp_path / "gone.ftmw"),
            "--format",
            "json",
            "--output",
            str(tmp_path / "o"),
        ]
    )
    cap = capsys.readouterr()
    assert rc == 1 and cap.out == ""
    payload = json.loads(cap.err)
    assert payload["schema"] == "ftmw/error@1" and payload["code"] == "not_found"


def test_corrupt_file_cli_exits_2(tmp_path, capsys):
    junk = tmp_path / "junk.ftmw"
    junk.write_bytes(b"not hdf5" * 100)
    rc = main(["read", "compute_display_ft", str(junk), "--format", "json"])
    cap = capsys.readouterr()
    assert rc == 2 and json.loads(cap.err)["code"] == "file_corrupt"


def test_unknown_table_is_refused_without_a_result(stage5):
    with pytest.raises(ValueError):
        ftmw.read_table(stage5, "no_such_table")


def test_invalid_pad_factor_is_refused(stage5):
    with pytest.raises(ValueError):
        ftmw.compute_display_ft(stage5, pad_factor=0)
