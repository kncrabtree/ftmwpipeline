"""``fit_thresholds``: the thresholds the persisted Stage 5 fit applied.

Spec: ``dev-docs/CONTRACT_STRATEGY.md`` §Fit thresholds. The promises checked:

* the two named thresholds are read from the persisted fit diagnostics;
* every field is ``Absent.NOT_RUN`` with no Stage 5 fit, and a threshold the
  fit never recorded is ``NOT_RUN`` -- never a guessed default;
* a read never writes the file;
* the old-fit display readers share the one diagnostics mapping and do not
  invent the stale ``4.0`` VIF threshold.

Fixtures are tiny synthetic Stage-0 files with a hand-written ``diagnostics``
attribute, so no fit is run.
"""

from __future__ import annotations

import hashlib
import json
import math

import h5py
import numpy as np
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import MANIFEST, Absent, Pipeline, to_jsonable
from ftmwpipeline._internal.stage5_impl import (
    DEFAULT_PROMOTION_MIN_SNR,
    derive_survival_floor,
    fit_thresholds_from_diagnostics,
    grading_thresholds_from_diagnostics,
    read_fit_thresholds_impl,
)
from ftmwpipeline.contract import FIT_THRESHOLDS_SCHEMA
from ftmwpipeline.core.stage_fit_settings import _HARD_DEFAULTS
from ftmwpipeline.file_manager import PipelineFileNotFoundError

pytestmark = [pytest.mark.unit]

_FIELDS = ("peak_survival_snr_floor", "vif_collapse_threshold")

#: The hard defaults of the two settings the grading fallback resolves.
_DEFAULT_FACTOR = _HARD_DEFAULTS["peak_survival"]["snr_survival_factor"]
_DEFAULT_VIF = float(_HARD_DEFAULTS["peak_survival"]["vif_collapse_threshold"])


@pytest.fixture
def base_file(tmp_path):
    """A Stage-0-only file (no Stage 5 fit)."""
    src = tmp_path / "src.h5"
    n, dt = 2048, 0.002
    t = np.arange(n) * dt
    fid = np.cos(2 * np.pi * 37.0 * t) * np.exp(-t / 3.0)
    with h5py.File(src, "w") as f:
        f.attrs["ftmw_input_version"] = 1
        f.attrs["spacing_us"] = dt
        f.attrs["probe_freq_mhz"] = 40960.0
        f.create_dataset("fid", data=fid)
    out = tmp_path / "exp.ftmw"
    Pipeline.create(str(out), source=str(src), format_name="ftmw-hdf5")
    return out


def _with_diagnostics(path, diagnostics):
    with h5py.File(path, "a") as f:
        f.require_group("stage5_fitting").attrs["diagnostics"] = json.dumps(diagnostics)
    return path


def _md5(path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_manifest_declares_the_accessor_and_schema():
    assert "fit_thresholds" in MANIFEST.accessors
    assert MANIFEST.file_bound["fit_thresholds"] is True
    assert FIT_THRESHOLDS_SCHEMA == "ftmw/fit_thresholds@1"
    assert FIT_THRESHOLDS_SCHEMA in MANIFEST.schemas


def test_recorded_thresholds_are_reported_by_name(base_file):
    _with_diagnostics(
        base_file,
        {
            "peak_survival": {"snr_floor": 3.52},
            "vif_collapse": {"vif_threshold": 25.0},
        },
    )
    res = ftmw.fit_thresholds(base_file)
    assert res["schema"] == "ftmw/fit_thresholds@1"
    assert res["peak_survival_snr_floor"] == pytest.approx(3.52)
    assert res["vif_collapse_threshold"] == pytest.approx(25.0)


def test_values_are_the_recorded_ones_not_defaults(base_file):
    _with_diagnostics(
        base_file,
        {
            "peak_survival": {"snr_floor": 7.25},
            "vif_collapse": {"vif_threshold": 4.0},
        },
    )
    res = ftmw.fit_thresholds(base_file)
    assert res["peak_survival_snr_floor"] == 7.25
    assert res["vif_collapse_threshold"] == 4.0


def test_no_stage5_fit_every_field_not_run(base_file):
    res = ftmw.fit_thresholds(base_file)
    assert res["schema"] == "ftmw/fit_thresholds@1"
    for name in _FIELDS:
        assert res[name] is Absent.NOT_RUN


def test_no_stage5_fit_wire_form_has_null_and_absent_sibling(base_file):
    wire = to_jsonable(ftmw.fit_thresholds(base_file))
    json.dumps(wire, allow_nan=False)
    for name in _FIELDS:
        assert wire[name] is None
        assert wire[f"{name}_absent"] == "not_run"


@pytest.mark.parametrize(
    "diag, present, missing",
    [
        (
            {"peak_survival": {"snr_floor": 3.5}},
            "peak_survival_snr_floor",
            "vif_collapse_threshold",
        ),
        (
            {"vif_collapse": {"vif_threshold": 25.0}},
            "vif_collapse_threshold",
            "peak_survival_snr_floor",
        ),
    ],
)
def test_threshold_never_recorded_is_not_run(base_file, diag, present, missing):
    _with_diagnostics(base_file, diag)
    res = ftmw.fit_thresholds(base_file)
    assert isinstance(res[present], float)
    assert res[missing] is Absent.NOT_RUN
    wire = to_jsonable(res)
    assert wire[f"{missing}_absent"] == "not_run"
    assert f"{present}_absent" not in wire


def test_fit_with_empty_diagnostics_is_all_not_run(base_file):
    _with_diagnostics(base_file, {})
    res = ftmw.fit_thresholds(base_file)
    assert all(res[name] is Absent.NOT_RUN for name in _FIELDS)


@pytest.mark.parametrize(
    "diag",
    [
        None,
        {},
        {"peak_survival": None, "vif_collapse": "x"},
        {"peak_survival": {"snr_floor": None}},
        {"peak_survival": {"snr_floor": "3.5"}},
        {"peak_survival": {"snr_floor": True}},
        {"peak_survival": {"snr_floor": float("nan")}},
        {"vif_collapse": {"vif_threshold": float("inf")}},
        {"vif_collapse": {}},
    ],
)
def test_unusable_recorded_value_is_not_run_not_a_default(diag):
    out = fit_thresholds_from_diagnostics(diag)
    assert set(out) == set(_FIELDS)
    for name in _FIELDS:
        assert out[name] is Absent.NOT_RUN


def test_integer_value_is_reported_as_float():
    out = fit_thresholds_from_diagnostics({"vif_collapse": {"vif_threshold": 25}})
    assert out["vif_collapse_threshold"] == 25.0
    assert isinstance(out["vif_collapse_threshold"], float)
    assert math.isfinite(out["vif_collapse_threshold"])


def test_pipeline_method_matches_api(base_file):
    _with_diagnostics(
        base_file,
        {"peak_survival": {"snr_floor": 3.3}, "vif_collapse": {"vif_threshold": 9.0}},
    )
    assert Pipeline.open(base_file).fit_thresholds() == ftmw.fit_thresholds(base_file)
    assert read_fit_thresholds_impl(base_file) == ftmw.fit_thresholds(base_file)


def test_read_leaves_file_byte_identical(base_file):
    _with_diagnostics(
        base_file,
        {"peak_survival": {"snr_floor": 3.5}, "vif_collapse": {"vif_threshold": 25.0}},
    )
    before = _md5(base_file)
    ftmw.fit_thresholds(base_file)
    Pipeline.open(base_file).fit_thresholds()
    assert _md5(base_file) == before


def test_read_with_no_fit_leaves_file_byte_identical(base_file):
    before = _md5(base_file)
    ftmw.fit_thresholds(base_file)
    assert _md5(base_file) == before


def test_missing_file_is_a_typed_not_found_error(tmp_path):
    with pytest.raises(PipelineFileNotFoundError) as ei:
        read_fit_thresholds_impl(tmp_path / "gone.ftmw")
    assert ei.value.to_dict()["code"] == "not_found"
    with pytest.raises(FileNotFoundError):
        ftmw.fit_thresholds(tmp_path / "gone.ftmw")


# ---- the old-fit display readers share the mapping and invent nothing ------


def _persist_promotion_min_snr(path, value):
    with h5py.File(path, "a") as f:
        f.require_group("stage3_peaks").attrs["promotion_min_snr"] = value
    return path


def test_grading_thresholds_use_recorded_values(base_file):
    floor, vif = grading_thresholds_from_diagnostics(
        str(base_file),
        {"peak_survival": {"snr_floor": 5.0}, "vif_collapse": {"vif_threshold": 9.0}},
    )
    assert (floor, vif) == (5.0, 9.0)


def test_grading_fallback_with_no_persisted_state_is_the_defaults(base_file):
    """Nothing persisted: persisted > recommended > default ends at the default."""
    floor, vif = grading_thresholds_from_diagnostics(str(base_file), {})
    assert floor == pytest.approx(DEFAULT_PROMOTION_MIN_SNR * _DEFAULT_FACTOR)
    assert vif == _DEFAULT_VIF
    assert vif != 4.0


def test_grading_fallback_uses_the_files_persisted_promotion_cutoff(base_file):
    _persist_promotion_min_snr(base_file, 5.0)
    floor, vif = grading_thresholds_from_diagnostics(str(base_file), {})
    assert floor == pytest.approx(5.0 * _DEFAULT_FACTOR)
    assert floor != pytest.approx(DEFAULT_PROMOTION_MIN_SNR * _DEFAULT_FACTOR)
    assert vif == _DEFAULT_VIF


def test_grading_fallback_uses_the_files_resolved_stage5_settings(base_file):
    _persist_promotion_min_snr(base_file, 5.0)
    ftmw.settings_set(base_file, "stage5.peak_survival.vif_collapse_threshold", 12.0)
    ftmw.settings_set(base_file, "stage5.peak_survival.snr_survival_factor", 1.5)
    floor, vif = grading_thresholds_from_diagnostics(str(base_file), {})
    assert vif == 12.0
    assert floor == pytest.approx(5.0 * 1.5)


def test_grading_fallback_explicit_floor_beats_the_derived_one(base_file):
    _persist_promotion_min_snr(base_file, 5.0)
    ftmw.settings_set(base_file, "stage5.peak_survival.snr_survival_floor", 7.25)
    floor, _vif = grading_thresholds_from_diagnostics(str(base_file), {})
    assert floor == 7.25


def test_grading_fallback_is_the_derivation_stage5_runs(base_file):
    _persist_promotion_min_snr(base_file, 4.4)
    floor, _vif = grading_thresholds_from_diagnostics(str(base_file), {})
    assert floor == derive_survival_floor(4.4, _DEFAULT_FACTOR)


def test_grading_thresholds_fall_back_per_field(base_file):
    _persist_promotion_min_snr(base_file, 5.0)
    floor, vif = grading_thresholds_from_diagnostics(
        str(base_file), {"peak_survival": {"snr_floor": 6.0}}
    )
    assert floor == 6.0 and vif == _DEFAULT_VIF
    floor, vif = grading_thresholds_from_diagnostics(
        str(base_file), {"vif_collapse": {"vif_threshold": 8.0}}
    )
    assert floor == pytest.approx(5.0 * _DEFAULT_FACTOR) and vif == 8.0


def test_fallback_does_not_leak_into_the_accessor(base_file):
    """The grader defaults are display-only; the contract still says NOT_RUN."""
    _persist_promotion_min_snr(base_file, 5.0)
    _with_diagnostics(base_file, {})
    assert grading_thresholds_from_diagnostics(str(base_file), {})[0] == (
        pytest.approx(5.0 * _DEFAULT_FACTOR)
    )
    res = ftmw.fit_thresholds(base_file)
    assert res["vif_collapse_threshold"] is Absent.NOT_RUN
