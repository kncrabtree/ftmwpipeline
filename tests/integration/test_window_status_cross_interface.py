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
from ftmwpipeline import Absent, Pipeline
from ftmwpipeline.cli.main import main
from ftmwpipeline.file_manager import StageDependencyError
from ftmwpipeline.serialize import to_jsonable

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
    "merged_from",
)


def _md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def _via_cli(path, capsys):
    """Run ``read window_status``: the envelope (rows are records, no arrays)."""
    rc = main(["read", "window_status", str(path), "--format", "json"])
    cap = capsys.readouterr()
    assert rc == 0, cap.err
    return json.loads(cap.out)


def _rows(payload):
    return {w.window_id: w for w in payload["windows"]}


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
    capsys,
):
    path = {"stage4": baseline_2638_stage4_small, "stage5": baseline_2638_stage5_small}[
        stage
    ]
    via_api = ftmw.window_status(str(path))
    via_pipeline = Pipeline.open(str(path)).window_status()
    envelope = _via_cli(path, capsys)
    table = ftmw.read_table(str(path), "window_status")

    assert via_api["schema"] == via_pipeline["schema"] == "ftmw/window_status@1"
    assert via_api == via_pipeline
    assert envelope == to_jsonable(via_api)
    assert envelope["schema"] == "ftmw/window_status@1"

    rows = via_api["windows"]
    assert list(table) == list(COLUMNS)
    np.testing.assert_array_equal(table["window_id"], [w.window_id for w in rows])
    np.testing.assert_array_equal(table["freq_min_mhz"], [w.freq_min_mhz for w in rows])
    np.testing.assert_array_equal(table["created"], [w.created for w in rows])
    for name in ("n_fitted_peaks", "live"):
        absent = [isinstance(getattr(w, name), Absent) for w in rows]
        np.testing.assert_array_equal(table[f"{name}__status"] == 1, absent)


def test_with_created_windows_all_interfaces_agree(stage5_with_created, capsys):
    path, new_id = stage5_with_created
    via_api = ftmw.window_status(str(path))
    assert _via_cli(path, capsys) == to_jsonable(via_api)
    row = _rows(via_api)[new_id]
    assert row.created
    assert row.freq_min_mhz == 27000.0
    assert row.freq_max_mhz == 27010.0


def test_rows_match_the_stage4_plan(baseline_2638_stage4_small):
    path = str(baseline_2638_stage4_small)
    plan = ftmw.load_windows(path).windows
    got = {
        w.window_id: (w.freq_min_mhz, w.freq_max_mhz)
        for w in ftmw.window_status(path)["windows"]
    }
    assert got == {w.window_id: tuple(w.freq_range) for w in plan}
    assert not any(w.created for w in ftmw.window_status(path)["windows"])


def test_stage4_only_file_reports_fit_fields_not_run_on_every_interface(
    baseline_2638_stage4_small, capsys
):
    path = baseline_2638_stage4_small
    payload = ftmw.window_status(str(path))
    assert payload["windows"]
    assert all(w.n_fitted_peaks is Absent.NOT_RUN for w in payload["windows"])
    assert all(w.live is Absent.NOT_RUN for w in payload["windows"])
    for row in _via_cli(path, capsys)["windows"]:
        assert row["n_fitted_peaks"] is None
        assert row["n_fitted_peaks_absent"] == "not_run"
        assert row["live"] is None
        assert row["live_absent"] == "not_run"
        assert "window_id_absent" not in row


def test_stage5_counts_match_the_fit(baseline_2638_stage5_small):
    path = str(baseline_2638_stage5_small)
    rows = ftmw.window_status(path)["windows"]
    fit = ftmw.load_fit(path)
    expected = {wf.window_id: len(wf.fitted_peaks) for wf in fit.window_fits}
    # A complete fit that kept no line in a window has no entry for it: 0 lines.
    assert {w.window_id: w.n_fitted_peaks for w in rows} == {
        w.window_id: expected.get(w.window_id, 0) for w in rows
    }
    assert set(expected) <= {w.window_id for w in rows}
    assert all(not isinstance(w.live, Absent) for w in rows)
    assert all(w.live is (w.n_fitted_peaks > 0) for w in rows)


def test_created_window_not_yet_refit_is_zero_and_not_live(stage5_with_created):
    path, new_id = stage5_with_created
    row = _rows(ftmw.window_status(str(path)))[new_id]
    assert row.n_fitted_peaks == 0 and row.live is False


def test_every_interface_leaves_the_file_byte_identical(stage5_with_created, capsys):
    path, _ = stage5_with_created
    before = _md5(path)
    ftmw.window_status(str(path))
    Pipeline.open(str(path)).window_status()
    ftmw.read_table(str(path), "window_status")
    _via_cli(path, capsys)
    assert _md5(path) == before


def test_cli_before_stage4_reports_error_json_and_names_the_command(
    baseline_2638_stage3, capsys
):
    rc = main(["read", "window_status", str(baseline_2638_stage3), "--format", "json"])
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


def test_read_table_before_stage4_is_typed_on_api_and_pipeline(baseline_2638_stage3):
    errors = []
    for call in (
        lambda: ftmw.read_table(str(baseline_2638_stage3), "window_status"),
        lambda: Pipeline.open(str(baseline_2638_stage3)).read_table("window_status"),
    ):
        with pytest.raises(StageDependencyError) as info:
            call()
        errors.append(info.value.to_dict())
    assert errors[0] == errors[1]
    assert errors[0]["command"] == "windows run"
