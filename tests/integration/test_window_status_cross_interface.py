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
from dataclasses import replace

import h5py
import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import Absent, BadSettingError, Pipeline
from ftmwpipeline._internal.atomic import atomic_write
from ftmwpipeline.cli.main import main
from ftmwpipeline.core.stage_fit_settings import ClockSource
from ftmwpipeline.file_manager import StageDependencyError
from ftmwpipeline.fitting.timebase_calibration import TimebaseCalibrationResult
from ftmwpipeline.io.stage_fit_settings_serialization import (
    load_stage_fit_settings_from_h5,
    save_stage_fit_settings_to_h5,
)
from ftmwpipeline.io.timebase_serialization import (
    GROUP_PATH,
    save_timebase_calibration_to_hdf5,
)
from ftmwpipeline.io.window_serialization import (
    load_window_plan_from_hdf5,
    save_fitted_plan_to_hdf5,
)
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


def _via_cli(path, capsys, *extra):
    """Run ``read window_status``: the envelope (rows are records, no arrays)."""
    rc = main(["read", "window_status", str(path), "--format", "json", *extra])
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


def _merge_two_lowest_windows(path, *, store_plan):
    """Pretend Stage 5 merged the two lowest-frequency windows of *path*.

    The survivor takes the lower id and the union range. With *store_plan* the
    fit carries the plan it was made on; without, only the accepted replan
    record names the merge (a fit from before the plan was stored).
    """
    with h5py.File(path, "a") as h5f:
        plan = load_window_plan_from_hdf5(h5f["stage4_windows"])
        low, high = sorted(plan.windows, key=lambda w: w.freq_range)[:2]
        survivor, absorbed = sorted((low, high), key=lambda w: w.window_id)
        fit = h5f["stage5_fitting"]
        fit.attrs["final_plan_revision"] = 1
        fit.attrs["replan_history"] = json.dumps(
            [
                {
                    "triggering_window_id": absorbed.window_id,
                    "partner_window_id": survivor.window_id,
                    "surviving_window_id": survivor.window_id,
                    "edge_side": "low",
                    "edge_coherence_before": 9.0,
                    "revision_before": 0,
                    "revision_after": 1,
                    "accepted": True,
                    "reason": "",
                }
            ]
        )
        merged_range = (
            min(low.freq_range[0], high.freq_range[0]),
            max(low.freq_range[1], high.freq_range[1]),
        )
        if store_plan:
            survivor.freq_range = merged_range
            survivor.diagnostics = {
                **dict(survivor.diagnostics or {}),
                "merged_from": [survivor.window_id, absorbed.window_id],
            }
            plan.windows = [w for w in plan.windows if w is not absorbed]
            plan.topological_order = [
                i for i in plan.topological_order if i != absorbed.window_id
            ]
            plan.plan_revision = 1
            save_fitted_plan_to_hdf5(plan, fit)
    return survivor.window_id, absorbed.window_id, merged_range


@pytest.fixture(
    scope="module",
    params=["stored_plan", "replan_record_only"],
)
def stage5_merged(request, baseline_2638_stage5_small, tmp_path_factory):
    """A copy of the Stage 5 baseline made to look like it applied one merge."""
    path = tmp_path_factory.mktemp("ws_merged") / "merged.ftmw"
    shutil.copy(baseline_2638_stage5_small, path)
    survivor, absorbed, bounds = _merge_two_lowest_windows(
        path, store_plan=request.param == "stored_plan"
    )
    return path, survivor, absorbed, bounds


def test_merged_windows_agree_on_every_interface(stage5_merged, capsys):
    path, survivor, absorbed, bounds = stage5_merged
    via_api = ftmw.window_status(str(path))
    via_pipeline = Pipeline.open(str(path)).window_status()
    envelope = _via_cli(path, capsys)
    table = ftmw.read_table(str(path), "window_status")

    assert via_api == via_pipeline
    assert envelope == to_jsonable(via_api)
    rows = _rows(via_api)
    assert absorbed not in rows
    assert (rows[survivor].freq_min_mhz, rows[survivor].freq_max_mhz) == bounds
    assert rows[survivor].merged_from == (absorbed,)
    assert all(w.merged_from == () for i, w in rows.items() if i != survivor)

    by_id = {w["window_id"]: w for w in envelope["windows"]}
    assert by_id[survivor]["merged_from"] == [absorbed]
    assert all(w["merged_from"] == [] for i, w in by_id.items() if i != survivor)
    assert list(table["merged_from"]) == [
        json.dumps(w.merged_from) for w in via_api["windows"]
    ]
    assert json.dumps([absorbed]) in list(table["merged_from"])


def test_merged_rows_leave_the_file_byte_identical(stage5_merged, capsys):
    path = stage5_merged[0]
    before = _md5(path)
    ftmw.window_status(str(path))
    ftmw.read_table(str(path), "window_status")
    _via_cli(path, capsys)
    assert _md5(path) == before


def test_an_unmerged_fit_has_empty_merged_from_everywhere(
    baseline_2638_stage5_small, capsys
):
    path = baseline_2638_stage5_small
    via_api = ftmw.window_status(str(path))
    assert all(w.merged_from == () for w in via_api["windows"])
    assert all(w["merged_from"] == [] for w in _via_cli(path, capsys)["windows"])
    table = ftmw.read_table(str(path), "window_status")
    assert set(table["merged_from"]) == {"[]"}


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


# ---- frame="raw" | "calibrated" ---------------------------------------------


def _make_self_calibrated(path, epsilon=2.2e-6):
    """Declare an unlocked digitizer and stamp a passing timebase result, so
    the raw and calibrated frames differ (``test_frame_parameter.py``'s
    recipe)."""
    persisted = load_stage_fit_settings_from_h5(str(path))
    clocks = (ClockSource(5120.0, locked=True), ClockSource(6250.0, locked=False))
    settings = replace(persisted, spur=replace(persisted.spur, clocks=clocks))
    with atomic_write(str(path)):
        save_stage_fit_settings_to_h5(str(path), settings)
    result = TimebaseCalibrationResult(
        epsilon=epsilon,
        sigma_epsilon=0.1e-6,
        n_used=5,
        n_detected=5,
        lattice_g_mhz=320.0,
        tone_reads=(),
        kappa_sys=0.0,
        snr_min=10.0,
        sample_dt_us=0.02,
        start_us=0.0,
        end_us=13.0,
        span_us=13.0,
        preconditions_passed=True,
    )
    with h5py.File(str(path), "a") as h5f:
        if GROUP_PATH in h5f:
            del h5f[GROUP_PATH]
        save_timebase_calibration_to_hdf5(result, h5f.create_group(GROUP_PATH))


@pytest.fixture(scope="module")
def stage5_self_calibrated(baseline_2638_stage5_small, tmp_path_factory):
    path = tmp_path_factory.mktemp("ws_frame") / "sc.ftmw"
    shutil.copy(baseline_2638_stage5_small, path)
    _make_self_calibrated(path)
    return path


@pytest.mark.parametrize("frame", ["raw", "calibrated"])
def test_each_frame_agrees_on_every_interface(stage5_self_calibrated, frame, capsys):
    path = stage5_self_calibrated
    via_api = ftmw.window_status(str(path), frame=frame)
    via_pipeline = Pipeline.open(str(path)).window_status(frame=frame)
    envelope = _via_cli(path, capsys, "--frame", frame)
    assert via_api == via_pipeline
    assert envelope == to_jsonable(via_api)
    assert envelope["frame"] == via_api["frame"] == frame


def test_cli_default_frame_is_raw(stage5_self_calibrated, capsys):
    path = stage5_self_calibrated
    envelope = _via_cli(path, capsys)
    assert envelope == to_jsonable(ftmw.window_status(str(path), frame="raw"))
    calibrated = _via_cli(path, capsys, "--frame", "calibrated")
    assert [w["freq_min_mhz"] for w in envelope["windows"]] != [
        w["freq_min_mhz"] for w in calibrated["windows"]
    ]


def test_an_unknown_frame_is_refused_alike(stage5_self_calibrated, capsys):
    path = stage5_self_calibrated
    errors = []
    for call in (
        lambda: ftmw.window_status(str(path), frame="molecular"),
        lambda: Pipeline.open(str(path)).window_status(frame="molecular"),
    ):
        with pytest.raises(BadSettingError) as info:
            call()
        errors.append(info.value.to_dict())
    assert errors[0] == errors[1]
    assert errors[0]["code"] == "bad_setting" and errors[0]["path"] == "frame"
    # The CLI's parser refuses it as the curation verbs' --frame does.
    with pytest.raises(SystemExit) as exit_info:
        main(["read", "window_status", str(path), "--frame", "molecular"])
    assert exit_info.value.code == 2
    capsys.readouterr()


def test_an_unknown_frame_is_refused_before_the_stage_dependency(
    baseline_2638_stage3,
):
    with pytest.raises(BadSettingError):
        ftmw.window_status(str(baseline_2638_stage3), frame="molecular")
    with pytest.raises(StageDependencyError):
        ftmw.window_status(str(baseline_2638_stage3), frame="calibrated")


def test_every_frame_leaves_the_file_byte_identical(stage5_self_calibrated, capsys):
    path = stage5_self_calibrated
    before = _md5(path)
    ftmw.window_status(str(path), frame="calibrated")
    _via_cli(path, capsys, "--frame", "calibrated")
    assert _md5(path) == before
