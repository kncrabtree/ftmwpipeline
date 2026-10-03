"""``fit_thresholds`` agrees across API, Pipeline and CLI, on a real fit.

Mandatory cross-interface category (``dev-docs/TESTING_STRATEGY.md``). Uses
the session-scoped 2638 small Stage 5 build; nothing here writes to it.
"""

from __future__ import annotations

import hashlib
import json
import shutil

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import Absent, Pipeline, to_jsonable
from ftmwpipeline.cli.main import main

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _cli(argv, capsys):
    rc = main(argv)
    cap = capsys.readouterr()
    return rc, cap.out, cap.err


def _md5(path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_real_fit_thresholds_agree_across_interfaces(
    baseline_2638_stage5_small, capsys
):
    path = baseline_2638_stage5_small
    via_api = ftmw.fit_thresholds(path)
    via_pipeline = Pipeline(path).fit_thresholds()
    rc, out, err = _cli(
        ["read", "fit_thresholds", str(path), "--format", "json"], capsys
    )
    assert rc == 0 and err == ""
    via_cli = json.loads(out)
    assert via_api == via_pipeline
    assert to_jsonable(via_api) == via_cli
    assert via_cli["schema"] == "ftmw/fit_thresholds@1"


def test_real_fit_reports_what_the_fit_recorded(baseline_2638_stage5_small):
    path = baseline_2638_stage5_small
    with h5py.File(path, "r") as f:
        diag = json.loads(f["stage5_fitting"].attrs["diagnostics"])
    res = ftmw.fit_thresholds(path)
    survival = diag.get("peak_survival") or {}
    vif = diag.get("vif_collapse") or {}
    if "snr_floor" in survival:
        assert res["peak_survival_snr_floor"] == pytest.approx(survival["snr_floor"])
    else:
        assert res["peak_survival_snr_floor"] is Absent.NOT_RUN
    if "vif_threshold" in vif:
        assert res["vif_collapse_threshold"] == pytest.approx(vif["vif_threshold"])
    else:
        assert res["vif_collapse_threshold"] is Absent.NOT_RUN


def test_real_fit_read_is_byte_identical(baseline_2638_stage5_small, capsys):
    path = baseline_2638_stage5_small
    before = _md5(path)
    ftmw.fit_thresholds(path)
    Pipeline(path).fit_thresholds()
    _cli(["read", "fit_thresholds", str(path), "--format", "json"], capsys)
    assert _md5(path) == before


def test_no_fit_agrees_across_interfaces_all_not_run(
    baseline_2638_stage4_small, tmp_path, capsys
):
    path = tmp_path / "nofit.ftmw"
    shutil.copy(baseline_2638_stage4_small, path)
    via_api = ftmw.fit_thresholds(path)
    via_pipeline = Pipeline(path).fit_thresholds()
    rc, out, _ = _cli(["read", "fit_thresholds", str(path), "--format", "json"], capsys)
    assert rc == 0
    via_cli = json.loads(out)
    assert via_api == via_pipeline
    assert to_jsonable(via_api) == via_cli
    for name in ("peak_survival_snr_floor", "vif_collapse_threshold"):
        assert via_api[name] is Absent.NOT_RUN
        assert via_cli[name] is None
        assert via_cli[f"{name}_absent"] == "not_run"


def test_cli_missing_file_is_not_found_error_on_stderr(tmp_path, capsys):
    rc, out, err = _cli(
        ["read", "fit_thresholds", str(tmp_path / "gone.ftmw"), "--format", "json"],
        capsys,
    )
    assert rc == 1 and out == ""
    payload = json.loads(err)
    assert payload["schema"] == "ftmw/error@1" and payload["code"] == "not_found"
