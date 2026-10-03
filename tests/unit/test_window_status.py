"""``window_status`` -- the per-window status table of the machine contract.

Written against ``dev-docs/CONTRACT_STRATEGY.md`` §Window status: one row per
Stage 4 plan window and per created window; columns ``window_id``,
``freq_min_mhz``, ``freq_max_mhz``, ``created``, ``n_fitted_peaks``, ``live``;
the two fit-derived columns are ``Absent.NOT_RUN`` (``__status`` ``1``) before
Stage 5; refusal before Stage 4 is a typed ``StageDependencyError``; a read
never writes the file. Files are built with h5py in ``tmp_path`` only.
"""

from __future__ import annotations

import hashlib
import json

import h5py
import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import MANIFEST, Absent, Pipeline
from ftmwpipeline._internal.read_impl import (
    READ_TABLES,
    WINDOW_STATUS_COLUMN_SPECS,
    read_table_impl,
    read_tables_impl,
    window_status_impl,
)
from ftmwpipeline.core.data_structures import (
    FittedPeak,
    FittingResult,
    FitWindow,
    SpectrumFit,
    WindowPlan,
)
from ftmwpipeline.file_manager import PipelineFileNotFoundError, StageDependencyError
from ftmwpipeline.io.fitting_serialization import save_spectrum_fit_to_hdf5
from ftmwpipeline.io.window_serialization import save_window_plan_to_hdf5

NOT_RUN = Absent.NOT_RUN.status

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

# Plan windows deliberately NOT in id order or frequency order of id.
PLAN = [(0, 100.0, 110.0), (1, 120.0, 130.0), (2, 140.0, 150.0)]


def _peak(wid: int, k: int) -> FittedPeak:
    return FittedPeak(
        detection_index=100 * wid + k,
        frequency_mhz=100.0 + wid * 20.0 + k,
        amplitude=0.5,
        decay_rate=0.2,
        phase=0.3,
        frequency_error=1e-4,
        amplitude_error=0.01,
        decay_rate_error=2e-3,
        phase_error=0.02,
        snr=10.0,
        chi_squared=1.0,
        window_id=wid,
    )


def _fit(counts):
    """A Stage 5 fit holding ``counts[window_id]`` lines per window."""
    windows, peaks = [], []
    for wid, n in counts.items():
        result = FittingResult(
            success=True,
            fitted_spectrum=None,
            cost=1.0,
            iterations=2,
            aic=10.0,
            reduced_chi2=1.1,
            window=None,
            window_id=wid,
            shape="lorentzian",
        )
        result.fitted_peaks = [_peak(wid, k) for k in range(n)]
        peaks.extend(result.fitted_peaks)
        windows.append(result)
    return SpectrumFit(
        window_fits=windows,
        fitted_peaks=peaks,
        parameters={"acquisition_us": 12.73},
    )


def _build(path, *, plan=PLAN, fit=None, created=None):
    """Write a minimal file: optional plan, Stage 5 fit and created windows."""
    with h5py.File(path, "w") as h5f:
        h5f.attrs["ftmw_format_version"] = "1.0"
        # What Pipeline.open needs to accept the file as a pipeline file.
        source = h5f.create_group("source_metadata")
        source.attrs["source_path"] = "/data/synthetic"
        source.attrs["format_name"] = "blackchirp"
        source.attrs["import_timestamp"] = "2026-07-21T16:21:31"
        source.attrs["source_hash"] = "0000000000000000"
        source.attrs["source_mtime"] = 0.0
        source.attrs["loader_parameters"] = "{}"
        h5f.create_group("pipeline_stages").attrs["completed_stages"] = json.dumps(
            ["stage0_fid_data"]
        )
        if plan is not None:
            save_window_plan_to_hdf5(
                WindowPlan(
                    windows=[
                        FitWindow(window_id=w, freq_range=(lo, hi), batch=0)
                        for w, lo, hi in plan
                    ],
                    dependency_edges=[],
                    topological_order=[w for w, _, _ in plan],
                ),
                h5f.create_group("stage4_windows"),
            )
        if fit is not None:
            save_spectrum_fit_to_hdf5(_fit(fit), h5f.create_group("stage5_fitting"))
        if created is not None:
            grp = h5f.create_group("stage6_review/created_windows")
            grp.attrs["data"] = json.dumps(
                [{"window_id": w, "freq_range": [lo, hi]} for w, lo, hi in created]
            )
    return path


def _md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def _rows(payload):
    return list(
        zip(
            payload["window_id"].tolist(),
            payload["freq_min_mhz"].tolist(),
            payload["freq_max_mhz"].tolist(),
            payload["created"].tolist(),
        )
    )


@pytest.fixture
def plan_only(tmp_path):
    return _build(tmp_path / "plan.ftmw")


@pytest.fixture
def fitted(tmp_path):
    return _build(tmp_path / "fit.ftmw", fit={0: 2, 1: 0, 2: 1})


# ---- schema and registration ----------------------------------------------


def test_payload_schema_and_columns(plan_only):
    payload = window_status_impl(plan_only)
    assert payload["schema"] == "ftmw/window_status@1"
    assert set(payload) == {"schema", *COLUMNS}


def test_dtypes(fitted):
    payload = window_status_impl(fitted)
    assert payload["window_id"].dtype == np.int64
    assert payload["freq_min_mhz"].dtype == np.float64
    assert payload["freq_max_mhz"].dtype == np.float64
    assert payload["created"].dtype == np.bool_
    assert payload["n_fitted_peaks"].dtype == np.int64
    assert payload["live"].dtype == np.bool_
    assert payload["n_fitted_peaks__status"].dtype == np.uint8
    assert payload["live__status"].dtype == np.uint8


def test_columns_are_equal_length(plan_only):
    payload = window_status_impl(plan_only)
    assert len({len(payload[c]) for c in COLUMNS}) == 1


def test_manifest_declares_accessor_schema_and_table():
    assert "window_status" in MANIFEST.accessors
    assert MANIFEST.file_bound["window_status"] is True
    assert "ftmw/window_status@1" in MANIFEST.schemas
    assert tuple(MANIFEST.tables["window_status"]) == COLUMNS
    assert "window_status" in READ_TABLES
    assert tuple(WINDOW_STATUS_COLUMN_SPECS) == COLUMNS


# ---- rows: plan windows and created windows --------------------------------


def test_one_row_per_plan_window(plan_only):
    payload = window_status_impl(plan_only)
    assert sorted(_rows(payload)) == [(w, lo, hi, False) for w, lo, hi in PLAN]
    assert not payload["created"].any()


def test_one_row_per_plan_window_and_per_created_window(tmp_path):
    path = _build(tmp_path / "c.ftmw", created=[(7, 112.0, 118.0), (8, 160.0, 170.0)])
    payload = window_status_impl(path)
    assert len(payload["window_id"]) == len(PLAN) + 2
    by_id = {r[0]: r for r in _rows(payload)}
    assert by_id[7] == (7, 112.0, 118.0, True)
    assert by_id[8] == (8, 160.0, 170.0, True)
    for w, lo, hi in PLAN:
        assert by_id[w] == (w, lo, hi, False)


def test_window_ids_are_unique_rows(tmp_path):
    path = _build(tmp_path / "c.ftmw", created=[(7, 112.0, 118.0)])
    ids = window_status_impl(path)["window_id"].tolist()
    assert len(ids) == len(set(ids))


def test_created_bounds_are_reported_min_then_max(tmp_path):
    path = _build(tmp_path / "c.ftmw", created=[(7, 118.0, 112.0)])
    row = {r[0]: r for r in _rows(window_status_impl(path))}[7]
    assert row[1] <= row[2] and row[1:3] == (112.0, 118.0)


def test_stage6_group_without_created_record_means_no_created(tmp_path):
    path = _build(tmp_path / "s6.ftmw")
    with h5py.File(path, "a") as h5f:
        h5f.create_group("stage6_review")
    payload = window_status_impl(path)
    assert len(payload["window_id"]) == len(PLAN)
    assert not payload["created"].any()


def test_empty_created_list_adds_no_rows(tmp_path):
    path = _build(tmp_path / "c.ftmw", created=[])
    assert len(window_status_impl(path)["window_id"]) == len(PLAN)


# ---- Absent.NOT_RUN before Stage 5 ----------------------------------------


def test_before_stage5_fit_columns_are_not_run(plan_only):
    payload = window_status_impl(plan_only)
    n = len(PLAN)
    np.testing.assert_array_equal(payload["n_fitted_peaks__status"], [NOT_RUN] * n)
    np.testing.assert_array_equal(payload["live__status"], [NOT_RUN] * n)
    assert NOT_RUN == 1


def test_before_stage5_created_windows_are_also_not_run(tmp_path):
    path = _build(tmp_path / "c.ftmw", created=[(7, 112.0, 118.0)])
    payload = window_status_impl(path)
    assert (payload["n_fitted_peaks__status"] == NOT_RUN).all()
    assert (payload["live__status"] == NOT_RUN).all()


def test_not_run_is_never_reported_as_zero_with_ok_status(plan_only):
    """Absence is carried by status; a not-run count must never read as data."""
    payload = window_status_impl(plan_only)
    assert not (payload["n_fitted_peaks__status"] == 0).any()
    assert not (payload["live__status"] == 0).any()


def test_identifying_columns_are_never_absent(plan_only):
    payload = window_status_impl(plan_only)
    for col in ("window_id", "freq_min_mhz", "freq_max_mhz", "created"):
        assert f"{col}__status" not in payload
    assert np.isfinite(payload["freq_min_mhz"]).all()


# ---- Stage 5 values: live and n_fitted_peaks -------------------------------


def test_after_stage5_counts_and_live_follow_the_fit(fitted):
    payload = window_status_impl(fitted)
    by_id = dict(
        zip(
            payload["window_id"].tolist(),
            zip(payload["n_fitted_peaks"].tolist(), payload["live"].tolist()),
        )
    )
    assert by_id == {0: (2, True), 1: (0, False), 2: (1, True)}
    assert (payload["n_fitted_peaks__status"] == 0).all()
    assert (payload["live__status"] == 0).all()


def test_live_means_at_least_one_fitted_line(fitted):
    payload = window_status_impl(fitted)
    np.testing.assert_array_equal(payload["live"], payload["n_fitted_peaks"] >= 1)


def test_zero_peak_window_after_stage5_is_data_not_absent(fitted):
    payload = window_status_impl(fitted)
    i = payload["window_id"].tolist().index(1)
    assert payload["n_fitted_peaks"][i] == 0
    assert payload["n_fitted_peaks__status"][i] == 0
    assert payload["live__status"][i] == 0


def test_counts_agree_with_the_fit_window_table(fitted):
    fit_cols = read_table_impl(fitted, "fit_windows")
    payload = window_status_impl(fitted)
    counts = dict(zip(fit_cols["window_id"].tolist(), fit_cols["n_peaks"].tolist()))
    for wid, n in zip(
        payload["window_id"].tolist(), payload["n_fitted_peaks"].tolist()
    ):
        assert n == counts[wid]


def test_created_window_with_fitted_lines_is_live(tmp_path):
    path = _build(
        tmp_path / "c.ftmw",
        fit={0: 1, 1: 1, 2: 1, 7: 3},
        created=[(7, 112.0, 118.0)],
    )
    payload = window_status_impl(path)
    i = payload["window_id"].tolist().index(7)
    assert payload["created"][i] and payload["live"][i]
    assert payload["n_fitted_peaks"][i] == 3


# ---- read_table form --------------------------------------------------------


def test_read_table_is_the_same_columns_without_schema(fitted):
    payload = window_status_impl(fitted)
    table = read_table_impl(fitted, "window_status")
    assert list(table) == list(COLUMNS)
    for col in COLUMNS:
        np.testing.assert_array_equal(table[col], payload[col])


def test_read_table_hyphenated_name(plan_only):
    assert list(read_table_impl(plan_only, "window-status")) == list(COLUMNS)


def test_read_table_column_selection(fitted):
    table = read_table_impl(
        fitted, "window_status", columns=["window_id", "n_fitted_peaks"]
    )
    assert list(table) == ["window_id", "n_fitted_peaks"]


def test_read_table_unknown_column_is_refused(plan_only):
    with pytest.raises(ValueError, match="bogus"):
        read_table_impl(plan_only, "window_status", columns=["bogus"])


def test_read_tables_listing_reports_window_status(plan_only):
    entry = read_tables_impl(plan_only)["window_status"]
    assert entry["available"] is True
    assert list(entry["columns"]) == list(COLUMNS)


def test_read_tables_listing_marks_unavailable_before_stage4(tmp_path):
    path = _build(tmp_path / "none.ftmw", plan=None)
    assert read_tables_impl(path)["window_status"]["available"] is False


# ---- typed refusals ---------------------------------------------------------


def test_before_stage4_raises_stage_dependency_error_naming_the_command(tmp_path):
    path = _build(tmp_path / "none.ftmw", plan=None)
    with pytest.raises(StageDependencyError) as info:
        window_status_impl(path)
    err = info.value
    assert err.command == "windows run"
    assert err.to_dict()["missing_dependencies"] == ["windows"]
    assert err.to_dict()["code"] == "stage_not_run"
    assert "windows run" in str(err)


def test_before_stage4_refusal_holds_on_every_python_surface(tmp_path):
    path = _build(tmp_path / "none.ftmw", plan=None)
    for call in (
        lambda: ftmw.window_status(str(path)),
        lambda: Pipeline.open(str(path)).window_status(),
    ):
        with pytest.raises(StageDependencyError) as info:
            call()
        assert info.value.command == "windows run"


def test_read_table_before_stage4_is_a_value_error_naming_the_command(tmp_path):
    """The table form follows the other tables' missing-stage convention."""
    path = _build(tmp_path / "none.ftmw", plan=None)
    with pytest.raises(ValueError, match=r"windows run"):
        read_table_impl(path, "window_status")


def test_stage5_without_stage4_is_still_refused(tmp_path):
    path = _build(tmp_path / "odd.ftmw", plan=None, fit={0: 1})
    with pytest.raises(StageDependencyError):
        window_status_impl(path)


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(PipelineFileNotFoundError):
        ftmw.window_status(str(tmp_path / "gone.ftmw"))


# ---- a read never writes ----------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"fit": {0: 2, 1: 0, 2: 1}},
        {"fit": {0: 1, 7: 1}, "created": [(7, 112.0, 118.0)]},
    ],
    ids=["plan", "fit", "fit+created"],
)
def test_read_leaves_the_file_byte_identical(tmp_path, kwargs):
    path = _build(tmp_path / "ro.ftmw", **kwargs)
    before = _md5(path)
    ftmw.window_status(str(path))
    Pipeline.open(str(path)).window_status()
    read_table_impl(path, "window_status")
    read_tables_impl(path)
    assert _md5(path) == before


def test_refused_read_leaves_the_file_byte_identical(tmp_path):
    path = _build(tmp_path / "ro.ftmw", plan=None)
    before = _md5(path)
    with pytest.raises(StageDependencyError):
        ftmw.window_status(str(path))
    assert _md5(path) == before


# ---- ordering is deterministic ----------------------------------------------


def test_repeated_reads_are_identical(fitted):
    a, b = window_status_impl(fitted), window_status_impl(fitted)
    for col in COLUMNS:
        np.testing.assert_array_equal(a[col], b[col])
