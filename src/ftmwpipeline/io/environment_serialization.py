"""
Per-stage analysis-environment provenance in the ``.ftmw`` container.

Layout (all attrs, no datasets -- the record is a handful of short strings)::

    /.attrs:
        last_written_with        (JSON)  -- the environment of the most recent
                                            stage write, for cheap display
    /pipeline_stages/.attrs:
        stage_environments       (JSON)  -- {stage_name: environment dict}
        environment_ack          (JSON)  -- the user's acknowledgement of an
                                            epoch mismatch, when one was given

The per-stage map lives beside ``completed_stages`` on ``/pipeline_stages``
rather than on each stage's own data group, for two reasons: the stage-to-group
mapping is not one-to-one (Stage 1 persists no group at all, Stage 2b writes
two), and keeping it in one place means a reader answers "was this file written
by one environment or several?" with a single attribute read instead of a walk.

Everything here is legacy-safe. A file written before environment recording
carries none of these attrs; it reads back as an empty map, which every
consumer treats as "unknown", never as "incompatible".
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import h5py

from ..core.environment import EnvironmentRecord

logger = logging.getLogger(__name__)

__all__ = [
    "save_stage_environment",
    "load_stage_environments",
    "load_last_written_with",
    "save_environment_ack",
    "load_environment_ack",
]

_STAGES_GROUP = "pipeline_stages"
_STAGE_ENVS_ATTR = "stage_environments"
_LAST_WRITTEN_ATTR = "last_written_with"
_ACK_ATTR = "environment_ack"


def _read_json_attr(holder: Any, name: str) -> Optional[Any]:
    raw = holder.attrs.get(name) if holder is not None else None
    if raw is None:
        return None
    try:
        return json.loads(str(raw))
    except (ValueError, TypeError):
        logger.warning("Ignoring malformed %r attribute", name)
        return None


def save_stage_environment(
    h5f: h5py.File, stage_name: str, record: EnvironmentRecord
) -> None:
    """Record *record* as the environment that produced *stage_name*.

    Also refreshes the root ``last_written_with`` stamp. Both writes are
    additive: an existing entry for the same stage is replaced (the stage was
    re-run, so the newer environment is the truthful one) and every other
    stage's entry is left alone -- which is exactly what makes a mixed-version
    file representable.
    """
    stages = h5f.require_group(_STAGES_GROUP)
    existing = _read_json_attr(stages, _STAGE_ENVS_ATTR) or {}
    if not isinstance(existing, dict):
        existing = {}
    existing[str(stage_name)] = record.to_dict()
    stages.attrs[_STAGE_ENVS_ATTR] = json.dumps(existing, sort_keys=True)

    h5f.attrs[_LAST_WRITTEN_ATTR] = json.dumps(
        {**record.to_dict(), "written_at": datetime.now().isoformat()},
        sort_keys=True,
    )


def load_stage_environments(h5f: h5py.File) -> Dict[str, EnvironmentRecord]:
    """The per-stage environment map, or ``{}`` on a file predating the stamp."""
    stages = h5f.get(_STAGES_GROUP)
    blob = _read_json_attr(stages, _STAGE_ENVS_ATTR)
    if not isinstance(blob, dict):
        return {}
    out: Dict[str, EnvironmentRecord] = {}
    for stage, payload in blob.items():
        if isinstance(payload, dict):
            out[str(stage)] = EnvironmentRecord.from_dict(payload)
    return out


def load_last_written_with(h5f: h5py.File) -> Optional[EnvironmentRecord]:
    """The environment of the most recent stage write, or ``None`` if unstamped."""
    blob = _read_json_attr(h5f, _LAST_WRITTEN_ATTR)
    if not isinstance(blob, dict):
        return None
    return EnvironmentRecord.from_dict(blob)


def save_environment_ack(
    h5f: h5py.File, record: EnvironmentRecord, *, reason: str = ""
) -> None:
    """Persist the user's acknowledgement of an analysis-epoch mismatch.

    Recorded in the file rather than taken as a transient command-line flag, on
    the same principle the frequency-accuracy floor follows: a fact that
    changes how a result should be read has to be reproducible from the record
    alone. A file whose curation was applied across an epoch boundary should say
    so in its report forever, not only in the terminal session where it happened.
    """
    stages = h5f.require_group(_STAGES_GROUP)
    stages.attrs[_ACK_ATTR] = json.dumps(
        {
            "acknowledged_environment": record.to_dict(),
            "reason": str(reason),
            "acknowledged_at": datetime.now().isoformat(),
        },
        sort_keys=True,
    )


def load_environment_ack(h5f: h5py.File) -> Optional[Dict[str, Any]]:
    """The persisted epoch-mismatch acknowledgement, or ``None``."""
    stages = h5f.get(_STAGES_GROUP)
    blob = _read_json_attr(stages, _ACK_ATTR)
    return blob if isinstance(blob, dict) else None
