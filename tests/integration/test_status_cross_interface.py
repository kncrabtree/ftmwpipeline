"""``status`` is identical on API, Pipeline and ``read status``; it never writes.

Spec: ``dev-docs/CONTRACT_STRATEGY.md`` §Status and settings.
"""

from __future__ import annotations

import json
import shutil

import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline import Pipeline
from ftmwpipeline.cli.main import main
from ftmwpipeline.contract import (
    PipelineFileNotFoundError,
    Stage,
    rerun_order,
    stage_depends_on,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _cli(path, capsys):
    rc = main(["read", "status", str(path), "--format", "json"])
    cap = capsys.readouterr()
    assert rc == 0, cap.err
    return json.loads(cap.out)


def test_status_agrees_across_interfaces(baseline_2638_stage1, capsys):
    # Mutation: any interface computing or shaping the payload differently.
    path = str(baseline_2638_stage1)
    via_api = ftmw.status(path)
    assert via_api == Pipeline.open(path).status() == _cli(path, capsys)
    assert via_api["schema"] == "ftmw/status@1"
    json.dumps(via_api, allow_nan=False)


def test_partially_run_file_reports_states_and_runnable(baseline_2638_stage1):
    # Mutation: state read from the wrong stage record, or runnable not
    # requiring every dependency complete (timebase needs only ft here).
    s = ftmw.status(str(baseline_2638_stage1))
    assert [e["stage"] for e in s["stages"]] == [x.value for x in Stage]
    state = {e["stage"]: e["state"] for e in s["stages"]}
    assert state["data"] == state["ft"] == "complete"
    assert all(v == "not_run" for k, v in state.items() if k not in ("data", "ft"))
    assert s["runnable"] == ["noise", "timebase"]
    for e in s["stages"]:
        assert e["depends_on"] == [d.value for d in stage_depends_on(e["stage"])]
    assert s["rerun_order"] == [x.value for x in rerun_order()]


def test_complete_file_has_nothing_runnable(stage5_reviewed_2638):
    # Mutation: completed stages still listed runnable.
    s = ftmw.status(str(stage5_reviewed_2638))
    assert all(e["state"] == "complete" for e in s["stages"])
    assert s["runnable"] == []


def test_status_never_writes(baseline_2638_stage1, tmp_path):
    # Mutation: status opening the file for write or touching its record.
    copy = tmp_path / "s.ftmw"
    shutil.copy(baseline_2638_stage1, copy)
    before = copy.read_bytes()
    ftmw.status(str(copy))
    Pipeline.open(str(copy)).status()
    assert copy.read_bytes() == before


def test_status_on_a_missing_file_is_a_typed_error(tmp_path, capsys):
    # Mutation: a bare exception or exit 0 for an unopenable file.
    missing = tmp_path / "nope.ftmw"
    with pytest.raises(PipelineFileNotFoundError):
        ftmw.status(str(missing))
    rc = main(["read", "status", str(missing), "--format", "json"])
    err = json.loads(capsys.readouterr().err)
    assert rc != 0 and err["code"] == "not_found"
