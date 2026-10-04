"""Every stage-running call reports the stages it invalidated.

``dev-docs/CONTRACT_STRATEGY.md`` §Status and settings: a stage-running call's
result carries ``invalidated``, canonical stage names in ``rerun_order``, empty
when nothing was invalidated; the CLI's human output names them. The same
change, made through each interface, must report the same stages and leave the
same stages complete.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List

import h5py
import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline.cli.main import main
from ftmwpipeline.file_manager import canonical_invalidated, rerun_order
from ftmwpipeline.pipeline import Pipeline

pytestmark = pytest.mark.integration

_NEW_TRIM = (27000.0, 39000.0)


def _completed(path: Path) -> List[str]:
    with h5py.File(path, "r") as h5f:
        return sorted(json.loads(h5f["pipeline_stages"].attrs["completed_stages"]))


def test_rerun_order_is_the_canonical_topological_order() -> None:
    assert rerun_order() == (
        "data",
        "ft",
        "noise",
        "tau",
        "tau_g",
        "timebase",
        "peaks",
        "windows",
        "fit",
        "review",
    )
    keys = ["stage6_review", "stage2_noise_result", "stage3_peaks", "stage3_peaks"]
    assert canonical_invalidated(keys) == ("noise", "peaks", "review")


@pytest.mark.parametrize("via", ["api", "pipeline", "cli"])
def test_stage1_change_reports_the_same_invalidation(
    baseline_2638_stage2: Path, tmp_path: Path, via: str, capsys
) -> None:
    fp = tmp_path / "w.ftmw"
    shutil.copy(baseline_2638_stage2, fp)
    if via == "api":
        reported = ftmw.compute_ft(fp, trim=_NEW_TRIM).invalidated
    elif via == "pipeline":
        reported = Pipeline.open(fp).compute_ft(trim=_NEW_TRIM).invalidated
    else:
        capsys.readouterr()
        assert main(["ft", "run", str(fp), "--trim", "27000:39000"]) == 0
        out = capsys.readouterr().out
        assert "Invalidated (re-run to refresh): noise" in out
        reported = ("noise",)
    assert reported == ("noise",)
    assert "stage2_noise_result" not in _completed(fp)


def test_identical_rerun_reports_nothing(
    baseline_2638_stage2: Path, tmp_path: Path, capsys
) -> None:
    fp = tmp_path / "w.ftmw"
    shutil.copy(baseline_2638_stage2, fp)
    assert ftmw.compute_ft(fp, trim=(26500, 40000)).invalidated == ()
    assert ftmw.estimate_noise(fp).invalidated == ()
    capsys.readouterr()
    assert main(["noise", "run", str(fp)]) == 0
    assert "Invalidated" not in capsys.readouterr().out
    assert "stage2_noise_result" in _completed(fp)


def test_force_reimport_reports_what_it_discarded(
    baseline_2638_stage2: Path, exp_2638_data_path: str, tmp_path: Path
) -> None:
    fp = tmp_path / "w.ftmw"
    shutil.copy(baseline_2638_stage2, fp)
    result = ftmw.import_data(fp, source=exp_2638_data_path, force=True)
    assert result["invalidated"] == ["ft", "noise"]
    reused = ftmw.import_data(fp, source=exp_2638_data_path)
    assert reused["invalidated"] == []
