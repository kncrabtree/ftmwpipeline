"""``window_status`` is the same on API, Pipeline, CLI and ``read_table``.

Mandatory cross-interface category (``dev-docs/TESTING_STRATEGY.md``) for the
``window_status`` accessor of the machine contract
(``dev-docs/CONTRACT_STRATEGY.md`` §Window status), on real 2638 data: the
session-scoped Stage 4 and Stage 5 baselines are reused, never rebuilt.
"""

from __future__ import annotations

import hashlib
import json
import shutil

import h5py
import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import Pipeline
from ftmwpipeline.cli.main import main
from ftmwpipeline.file_manager import StageDependencyError

pytestmark = [pytest.mark.integration]

COLUMNS = (
    "window_id",
    "freq_min_mhz",
    "freq_max_mhz",
    "created",
    "n_fitted_peaks",
    "n_fitted_peaks__status",
    "live",
    "live__status",
)


def _md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def _via_cli(path, tmp_path, capsys):
    """Run ``read window_status`` and load the envelope plus its .npy columns."""
    out = tmp_path / "cli_out"
    rc = main(["read", "window_status", str(path), "--format", "json", "-o", str(out)])
    cap = capsys.readouterr()
    assert rc == 0, cap.err
    envelope = json.loads(cap.out)
    arrays = {c: np.load(out / envelope[c]) for c in COLUMNS}
    return envelope, arrays


def _assert_same(a, b):
    assert set(a) == set(b)
    for col in COLUMNS:
        assert a[col].dtype == b[col].dtype, col
        np.testing.assert_array_equal(a[col], b[col], err_msg=col)


@pytest.fixture(scope="module")
def stage5_with_created(baseline_2638_stage5_small, tmp_path_factory):
    """A copy of the Stage 5 baseline that also records one created window."""
    path = tmp_path_factory.mktemp("ws_created") / "created.ftmw"
    shutil.copy(baseline_2638_stage5_small, path)
    used = {int(i) for i in ftmw.read_table(str(path), "windows")["window_id"]}
    new_id = max(used) + 100
    with h5py.File(path, "a") as h5f:
        grp = h5f.require_group("stage6_review/created_windows")
        grp.attrs["data"] = json.dumps(
            [{"window_id": new_id, "freq_range": [27000.0, 27010.0]}]
        )
    return path, new_id


@pytest.mark.parametrize("stage", ["stage4", "stage5"])
def test_api_pipeline_cli_and_table_agree(
    stage,
    baseline_2638_stage4_small,
    baseline_2638_stage5_small,
    tmp_path,
    capsys,
):
    path = {"stage4": baseline_2638_stage4_small, "stage5": baseline_2638_stage5_small}[
        stage
    ]
    via_api = ftmw.window_status(str(path))
    via_pipeline = Pipeline.open(str(path)).window_status()
    envelope, via_cli = _via_cli(path, tmp_path, capsys)
    table = ftmw.read_table(str(path), "window_status")

    assert via_api["schema"] == via_pipeline["schema"] == "ftmw/window_status@1"
    assert envelope["schema"] == "ftmw/window_status@1"
    _assert_same(via_api, via_pipeline)
    _assert_same(via_api, {"schema": 0, **via_cli})
    assert list(table) == list(COLUMNS)
    for col in COLUMNS:
        np.testing.assert_array_equal(table[col], via_api[col], err_msg=col)


def test_with_created_windows_all_interfaces_agree(
    stage5_with_created, tmp_path, capsys
):
    path, new_id = stage5_with_created
    via_api = ftmw.window_status(str(path))
    _, via_cli = _via_cli(path, tmp_path, capsys)
    _assert_same(via_api, {"schema": 0, **via_cli})
    i = via_api["window_id"].tolist().index(new_id)
    assert via_api["created"][i]
    assert via_api["freq_min_mhz"][i] == 27000.0
    assert via_api["freq_max_mhz"][i] == 27010.0


def test_rows_match_the_stage4_plan(baseline_2638_stage4_small):
    path = str(baseline_2638_stage4_small)
    plan = ftmw.load_windows(path).windows
    payload = ftmw.window_status(path)
    got = {
        w: (lo, hi)
        for w, lo, hi in zip(
            payload["window_id"].tolist(),
            payload["freq_min_mhz"].tolist(),
            payload["freq_max_mhz"].tolist(),
        )
    }
    assert got == {w.window_id: tuple(w.freq_range) for w in plan}
    assert not payload["created"].any()


def test_stage4_only_file_reports_fit_columns_not_run(
    baseline_2638_stage4_small, tmp_path, capsys
):
    path = baseline_2638_stage4_small
    payload = ftmw.window_status(str(path))
    assert (payload["n_fitted_peaks__status"] == 1).all()
    assert (payload["live__status"] == 1).all()
    _, via_cli = _via_cli(path, tmp_path, capsys)
    assert (via_cli["n_fitted_peaks__status"] == 1).all()
    assert (via_cli["live__status"] == 1).all()


def test_stage5_counts_match_the_fit(baseline_2638_stage5_small):
    path = str(baseline_2638_stage5_small)
    payload = ftmw.window_status(path)
    fit = ftmw.load_fit(path)
    expected = {wf.window_id: len(wf.fitted_peaks) for wf in fit.window_fits}
    got = dict(zip(payload["window_id"].tolist(), payload["n_fitted_peaks"].tolist()))
    assert got == expected
    assert (payload["n_fitted_peaks__status"] == 0).all()
    assert (payload["live__status"] == 0).all()
    np.testing.assert_array_equal(payload["live"], payload["n_fitted_peaks"] > 0)


def test_created_window_not_yet_refit_is_zero_and_not_live(stage5_with_created):
    path, new_id = stage5_with_created
    payload = ftmw.window_status(str(path))
    i = payload["window_id"].tolist().index(new_id)
    assert payload["n_fitted_peaks"][i] == 0 and not payload["live"][i]
    assert payload["live__status"][i] == 0


def test_every_interface_leaves_the_file_byte_identical(
    stage5_with_created, tmp_path, capsys
):
    path, _ = stage5_with_created
    before = _md5(path)
    ftmw.window_status(str(path))
    Pipeline.open(str(path)).window_status()
    ftmw.read_table(str(path), "window_status")
    _via_cli(path, tmp_path, capsys)
    assert _md5(path) == before


def test_cli_before_stage4_reports_error_json_and_names_the_command(
    baseline_2638_stage3, tmp_path, capsys
):
    rc = main(
        [
            "read",
            "window_status",
            str(baseline_2638_stage3),
            "--format",
            "json",
            "-o",
            str(tmp_path / "o"),
        ]
    )
    cap = capsys.readouterr()
    assert rc == 1 and cap.out == ""
    payload = json.loads(cap.err)
    assert payload["schema"] == "ftmw/error@1"
    assert payload["code"] == "stage_not_run"
    assert payload["command"] == "windows run"
    assert payload["missing_dependencies"] == ["windows"]


def test_api_and_pipeline_before_stage4_raise_the_same_typed_error(
    baseline_2638_stage3,
):
    errors = []
    for call in (
        lambda: ftmw.window_status(str(baseline_2638_stage3)),
        lambda: Pipeline.open(str(baseline_2638_stage3)).window_status(),
    ):
        with pytest.raises(StageDependencyError) as info:
            call()
        errors.append(info.value.to_dict())
    assert errors[0] == errors[1]
    assert errors[0]["command"] == "windows run"
