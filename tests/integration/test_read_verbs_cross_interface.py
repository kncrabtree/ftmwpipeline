"""Every accessor is the same on API, Pipeline and ``read <name> --format json``.

Mandatory cross-interface category (``dev-docs/TESTING_STRATEGY.md``) for the
rules every accessor follows (``dev-docs/CONTRACT_STRATEGY.md`` §Accessors):
one CLI verb per accessor spelled as its API name, the schema stamped in the
Python payload where the payload can carry it, and one envelope form per kind
of result (a dict or dataclass stamped directly, a list wrapped as ``items``, a
scalar wrapped as ``value``). Runs on the session-scoped reviewed 2638 build,
read only; arrays go to pytest ``tmp_path``.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import MANIFEST, Absent, Pipeline
from ftmwpipeline.cli.contract_commands import contract_envelope
from ftmwpipeline.cli.main import main
from ftmwpipeline.contract import (
    CALIBRATION_SCHEMA,
    FINAL_PRODUCTS_SCHEMA,
    REVIEW_LOG_SCHEMA,
    SNAP_TOLERANCE_SCHEMA,
)
from ftmwpipeline.core.data_structures import ComplexFT
from ftmwpipeline.serialize import ArrayCollector, to_jsonable

pytestmark = [pytest.mark.integration, pytest.mark.slow]

#: name -> (envelope kind, schema stamped in the Python payload)
#: ``direct``: dict or dataclass stamped directly; ``items``: a list or tuple
#: wrapped; ``value``: a scalar wrapped. ``py_stamp`` is ``"dict"`` when the
#: Python dict carries ``"schema"``, ``"class"`` when its dataclass declares
#: ``__ftmw_schema__``, ``None`` when only the CLI envelope carries it.
KINDS = {
    "capabilities": ("direct", "dict"),
    "frequency_calibration": ("direct", "class"),
    "refit_snap_tol_mhz": ("value", None),
    "read_metadata": ("direct", None),
    "read_tables": ("direct", None),
    "read_table": ("direct", None),
    "settings_defaults": ("items", None),
    "settings_show": ("items", None),
    "get_final_products": ("direct", "class"),
    "review_log": ("items", None),
    "get_pipeline_info": ("direct", None),
    "compute_display_ft": ("direct", None),
    "fid_samples": ("direct", "dict"),
    "display_units": ("direct", "dict"),
    "fit_thresholds": ("direct", "dict"),
    "window_status": ("direct", "dict"),
    "preview_source": ("direct", "dict"),
    "window_model": ("direct", "dict"),
    "spectrum_model": ("direct", "dict"),
    "analysis_fingerprint": ("direct", "dict"),
}


def test_every_accessor_has_a_declared_kind_here():
    assert set(KINDS) == set(MANIFEST.accessors) == set(EXPECTED_SCHEMA)


#: The schema each accessor's envelope must carry, pinned independently of the
#: registration code so a mislabeled payload fails.
EXPECTED_SCHEMA = {
    "capabilities": "ftmw/capabilities@1",
    "frequency_calibration": "ftmw/calibration@1",
    "refit_snap_tol_mhz": "ftmw/snap_tolerance@1",
    "read_metadata": "ftmw/metadata@1",
    "read_tables": "ftmw/tables@1",
    "read_table": "ftmw/table@1",
    "settings_defaults": "ftmw/settings_defaults@1",
    "settings_show": "ftmw/settings@1",
    "get_final_products": "ftmw/final_products@1",
    "review_log": "ftmw/review_log@1",
    "get_pipeline_info": "ftmw/pipeline_info@1",
    "compute_display_ft": "ftmw/display_ft@1",
    "fid_samples": "ftmw/fid_samples@1",
    "display_units": "ftmw/display_units@1",
    "fit_thresholds": "ftmw/fit_thresholds@1",
    "window_status": "ftmw/window_status@1",
    "preview_source": "ftmw/source_preview@1",
    "window_model": "ftmw/window_model@1",
    "spectrum_model": "ftmw/spectrum_model@1",
    "analysis_fingerprint": "ftmw/analysis_fingerprint@1",
}


def _first_window_id(path):
    return int(ftmw.load_fit(path).window_fits[0].window_id)


def _py_args(name, path, source):
    """Positional and keyword arguments of the Python call, after the path."""
    if name == "read_table":
        return (), {"table": "fit_peaks", "columns": ["frequency_mhz", "snr"]}
    if name == "preview_source":
        return (source,), {}
    if name == "window_model":
        return (_first_window_id(path),), {"components": True}
    return (), {}


def _cli_argv(name, path, source):
    if name == "read_table":
        return ["fit_peaks", "--columns", "frequency_mhz,snr"]
    if name == "preview_source":
        return [source]
    if name == "window_model":
        return [str(_first_window_id(path)), "--components"]
    return []


def _call(name, path, source, via):
    args, kwargs = _py_args(name, path, source)
    if MANIFEST.file_bound[name]:
        if via == "api":
            return getattr(ftmw, name)(path, *args, **kwargs)
        pname = MANIFEST.pipeline_names[name]
        return getattr(Pipeline.open(path), pname)(*args, **kwargs)
    if via == "api":
        return getattr(ftmw, name)(*args, **kwargs)
    return getattr(Pipeline, MANIFEST.pipeline_names[name])(*args, **kwargs)


def _run_cli(name, path, source, out, capsys):
    argv = ["read", name]
    if MANIFEST.file_bound[name]:
        argv.append(path)
    argv += _cli_argv(name, path, source)
    argv += ["--format", "json", "--output", str(out)]
    rc = main(argv)
    cap = capsys.readouterr()
    assert rc == 0, cap.err
    return json.loads(cap.out)


def _wire(result, schema, collector):
    return json.loads(
        json.dumps(
            to_jsonable(
                contract_envelope(result, schema), schema=schema, arrays=collector
            )
        )
    )


@pytest.fixture
def stage5(stage5_reviewed_2638):
    return str(stage5_reviewed_2638)


@pytest.mark.parametrize("name", MANIFEST.accessors)
def test_api_pipeline_and_cli_agree(name, stage5, exp_2638_data_path, tmp_path, capsys):
    source = str(exp_2638_data_path)
    via_api = _call(name, stage5, source, "api")
    via_pipeline = _call(name, stage5, source, "pipeline")
    out = tmp_path / "out"
    env = _run_cli(name, stage5, source, out, capsys)

    schema = env["schema"]
    assert schema == EXPECTED_SCHEMA[name]
    api_arrays, pipe_arrays = ArrayCollector(), ArrayCollector()
    assert _wire(via_api, schema, api_arrays) == env
    assert _wire(via_pipeline, schema, pipe_arrays) == env
    assert set(api_arrays.arrays) == set(pipe_arrays.arrays)
    for fname, array in api_arrays.arrays.items():
        np.testing.assert_array_equal(pipe_arrays.arrays[fname], array)
        np.testing.assert_array_equal(np.load(out / fname), array)


@pytest.mark.parametrize("name", MANIFEST.accessors)
def test_python_payload_carries_its_schema_where_it_can(
    name, stage5, exp_2638_data_path
):
    _, py_stamp = KINDS[name]
    source = str(exp_2638_data_path)
    for via in ("api", "pipeline"):
        result = _call(name, stage5, source, via)
        if py_stamp == "dict":
            assert result["schema"] in MANIFEST.schemas
        elif py_stamp == "class":
            assert result is not None
            assert type(result).__ftmw_schema__ in MANIFEST.schemas


@pytest.mark.parametrize("name", MANIFEST.accessors)
def test_envelope_kind_matches_the_python_result(
    name, stage5, exp_2638_data_path, tmp_path, capsys
):
    kind, _ = KINDS[name]
    source = str(exp_2638_data_path)
    result = _call(name, stage5, source, "api")
    env = _run_cli(name, stage5, source, tmp_path / "o", capsys)
    if kind == "items":
        assert isinstance(result, (list, tuple))
        assert set(env) == {"schema", "items"}
        assert len(env["items"]) == len(result)
    elif kind == "value":
        assert not isinstance(result, (dict, list, tuple))
        assert set(env) == {"schema", "value"}
    else:
        assert isinstance(result, (dict, ComplexFT)) or dataclasses.is_dataclass(result)
        assert "items" not in env and "value" not in env


def test_items_envelopes_are_non_empty_on_real_rows(stage5, capsys, tmp_path):
    for name in ("review_log", "settings_show", "settings_defaults"):
        env = _run_cli(name, stage5, "", tmp_path / "o", capsys)
        assert env["items"], name


def test_list_accessor_wraps_as_items_with_its_schema(stage5, tmp_path, capsys):
    env = _run_cli("review_log", stage5, "", tmp_path / "o", capsys)
    assert env["schema"] == REVIEW_LOG_SCHEMA
    assert [row["kind"] for row in env["items"]] == [
        e.kind for e in ftmw.review_log(stage5)
    ]


def test_scalar_accessor_wraps_as_value(stage5, tmp_path, capsys):
    env = _run_cli("refit_snap_tol_mhz", stage5, "", tmp_path / "o", capsys)
    assert env == {
        "schema": SNAP_TOLERANCE_SCHEMA,
        "value": ftmw.refit_snap_tol_mhz(stage5),
    }


def test_dataclass_accessor_is_stamped_directly(stage5, tmp_path, capsys):
    env = _run_cli("frequency_calibration", stage5, "", tmp_path / "o", capsys)
    stamp = ftmw.frequency_calibration(stage5)
    assert env["schema"] == CALIBRATION_SCHEMA == type(stamp).__ftmw_schema__
    assert set(MANIFEST.fields["CalibrationStamp"]) == set(env) - {"schema"}


def test_text_columns_are_saved_without_pickling(stage5, tmp_path, capsys):
    out = tmp_path / "o"
    rc = main(
        [
            "read",
            "read_table",
            stage5,
            "fit_peaks",
            "--columns",
            "shape,origin,clock_lattice",
            "--output",
            str(out),
        ]
    )
    assert rc == 0, capsys.readouterr().err
    capsys.readouterr()
    table = ftmw.read_table(stage5, "fit_peaks", ["shape", "origin", "clock_lattice"])
    for col in ("shape", "origin", "clock_lattice"):
        saved = np.load(out / f"{col}.npy")  # allow_pickle defaults to False
        assert saved.dtype.kind == "U"
        assert saved.tolist() == [str(v) for v in table[col]]


def test_dict_accessor_is_stamped_directly(stage5, tmp_path, capsys):
    env = _run_cli("read_metadata", stage5, "", tmp_path / "o", capsys)
    assert set(env) - {"schema"} == set(ftmw.read_metadata(stage5))


# ---- get_final_products before Stage 6: absent, not an empty list ----------


def test_final_products_before_stage6_is_value_null_not_run(
    baseline_2638_stage5_small, tmp_path, capsys
):
    path = str(baseline_2638_stage5_small)
    assert ftmw.get_final_products(path) is None
    assert Pipeline.open(path).final_products() is None
    env = _run_cli("get_final_products", path, "", tmp_path / "o", capsys)
    assert env == {
        "schema": FINAL_PRODUCTS_SCHEMA,
        "value": None,
        "value_absent": "not_run",
    }


def test_final_products_after_stage6_is_stamped_directly(stage5, tmp_path, capsys):
    env = _run_cli("get_final_products", stage5, "", tmp_path / "o", capsys)
    assert env["schema"] == FINAL_PRODUCTS_SCHEMA
    assert env["peaks"] and "value_absent" not in env
    # No bare null: every null field of a line carries its reason.
    for line in env["peaks"]:
        for key, value in line.items():
            if value is None:
                assert line[key + "_absent"] in ("not_run", "undefined"), key


# ---- window_status before Stage 5, and the table form ----------------------


def test_window_status_before_stage5_rows_are_null_plus_absent(
    baseline_2638_stage4_small, tmp_path, capsys
):
    path = str(baseline_2638_stage4_small)
    env = _run_cli("window_status", path, "", tmp_path / "o", capsys)
    assert env["windows"]
    for row in env["windows"]:
        assert row["n_fitted_peaks"] is None
        assert row["n_fitted_peaks_absent"] == "not_run"
        assert row["live"] is None
        assert row["live_absent"] == "not_run"
    assert all(
        w.n_fitted_peaks is Absent.NOT_RUN for w in ftmw.window_status(path)["windows"]
    )


def test_window_status_table_columns_match_the_rows(stage5):
    rows = ftmw.window_status(stage5)["windows"]
    table = ftmw.read_table(stage5, "window_status")
    np.testing.assert_array_equal(table["window_id"], [w.window_id for w in rows])
    np.testing.assert_array_equal(table["live"], [w.live for w in rows])
    assert (table["live__status"] == 0).all()


# ---- get_pipeline_info always carries warnings -----------------------------


@pytest.mark.parametrize("fixture", ["stage5_reviewed_2638", "baseline_2638_stage1"])
def test_pipeline_info_always_has_warnings(fixture, request, tmp_path, capsys):
    path = str(request.getfixturevalue(fixture))
    for info in (
        ftmw.get_pipeline_info(path),
        Pipeline.open(path).info(),
    ):
        assert isinstance(info["warnings"], list)
    env = _run_cli("get_pipeline_info", path, "", tmp_path / "o", capsys)
    assert isinstance(env["warnings"], list)
    assert "warnings_absent" not in env
