"""``status(path)``: per-stage state, runnable set and refresh order.

Normative spec: ``dev-docs/CONTRACT_STRATEGY.md`` §Status and settings. The
stage graph is :data:`ftmwpipeline.contract.STAGE_KEYS` over
``PipelineStageTracker.STAGE_DEPENDENCIES``; this module only reads which
stages the file records as complete. Never writes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..contract import (
    STATUS_SCHEMA,
    Stage,
    key_for_stage,
    rerun_order,
    stage_depends_on,
)
from ..file_manager import open_pipeline_file


def status_impl(file_path: str) -> Dict[str, Any]:
    """The ``ftmw/status@1`` payload for ``file_path``.

    A stage is ``complete`` when the file's stage record lists it and
    ``not_run`` otherwise (``partial`` is part of the vocabulary but nothing
    produces it yet). ``runnable`` lists the stages that are not complete but
    whose dependencies all are, in enum order.
    """
    _, _, tracker = open_pipeline_file(Path(file_path))
    complete = {s for s in Stage if tracker.is_completed(key_for_stage(s))}
    stages: List[Dict[str, Any]] = []
    runnable: List[str] = []
    for stage in Stage:
        deps = stage_depends_on(stage)
        stages.append(
            {
                "stage": stage.value,
                "state": "complete" if stage in complete else "not_run",
                "depends_on": [d.value for d in deps],
            }
        )
        if stage not in complete and all(d in complete for d in deps):
            runnable.append(stage.value)
    return {
        "schema": STATUS_SCHEMA,
        "stages": stages,
        "runnable": runnable,
        "rerun_order": [s.value for s in rerun_order()],
    }
