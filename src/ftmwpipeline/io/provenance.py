"""How a stage records what it used.

The analysis fingerprint (``dev-docs/CONTRACT_STRATEGY.md`` §Analysis
fingerprint) is computed from the file alone, through each stage's codec. That
only works if every stage leaves behind a complete account of what it ran with.
This module holds the two pieces of that account every stage shares; the rest
is each stage's own codec.

A stage that completes does three things, in this order:

1. **Persist the values it ran with, resolved, through its codec.** The codec
   writes every field of the record and stamps the record's *field-set
   version* (:func:`write_field_set_version`, with the codec's own
   ``*_FIELD_SET_VERSION`` constant). A record at the current version has every
   field present, so a ``None`` in it means "resolved to unset", never "not
   recorded". Values a stage took from another stage's result (a decay time
   from Stage 2b, an epsilon from the timebase) are recorded *by the consumer*,
   in its own record, under ``consumed``; re-running the upstream stage does
   not invalidate the consumer, whose record still says what it used.
2. **Stamp its analysis epoch** with :func:`stamp_stage_epoch` (or
   :func:`stamp_stage_epoch_in_file`). This is part of completing the stage, not
   advice about it: a stamp that cannot be written raises, and the caller
   must not mark the stage complete. Only the drift *report* that follows a
   stamp is advisory.
3. **Mark itself complete** (``_update_stage_completion`` does 2 and 3 for
   every tracked stage after Stage 1).

A producer that is not a tracked stage but whose output feeds later stages (the
Stage 2b shape recommendation) does 1 and 2 under its own record name. A path
that rewrites a stage's persisted products after the fact (the Stage 6
final-products refresh) restamps that stage.

Reading: :func:`record_provenance` (or the per-codec ``*_provenance`` wrapper)
reports a record's version against the codec's current one. A record with no
version, or an older one, is *pre-provenance*: it was written before the
stage persisted everything, so its ``None`` fields cannot be told apart from
fields that were never recorded. Readers do not change what they return for
such a record; the fingerprint raises ``incomplete_provenance`` for it instead
of inventing values.

Bumping a version: a change that adds, removes or re-means a field of a record
bumps that codec's ``*_FIELD_SET_VERSION`` by one. Nothing else does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import h5py
import numpy as np

from .._internal.atomic import h5open
from ..core.environment import EnvironmentRecord

logger = logging.getLogger(__name__)

#: The attribute every persisted settings record carries its field-set version
#: under, on the record's own group.
FIELD_SET_VERSION_ATTR = "field_set_version"


@dataclass(frozen=True)
class RecordProvenance:
    """A persisted record's field-set version against its codec's current one.

    Attributes
    ----------
    path : str
        The record's HDF5 group path.
    version : int or None
        The stored version; ``None`` for a record written before versions were
        recorded.
    current_version : int
        The version the running codec writes.
    """

    path: str
    version: Optional[int]
    current_version: int

    @property
    def is_current(self) -> bool:
        """Every field is present; ``None`` means resolved to unset."""
        return self.version == self.current_version

    @property
    def is_pre_provenance(self) -> bool:
        """Written before this record held every field it holds today."""
        return self.version is None or self.version < self.current_version

    @property
    def is_newer(self) -> bool:
        """Written by a newer codec than the running one."""
        return self.version is not None and self.version > self.current_version


def write_field_set_version(attrs: h5py.AttributeManager, version: int) -> None:
    """Stamp *version* on a record's attrs. Called by the record's codec."""
    attrs[FIELD_SET_VERSION_ATTR] = np.int64(version)


def read_field_set_version(attrs: Mapping[str, Any]) -> Optional[int]:
    """The stored field-set version, or ``None`` when the record has none."""
    raw = attrs.get(FIELD_SET_VERSION_ATTR)
    if raw is None:
        return None
    if isinstance(raw, (bool, np.bool_)):
        raise ValueError(f"malformed {FIELD_SET_VERSION_ATTR!r} attr: {raw!r}")
    return int(raw)


def group_provenance(group: h5py.Group, current_version: int) -> RecordProvenance:
    """The :class:`RecordProvenance` of an open record group."""
    return RecordProvenance(
        path=str(group.name).lstrip("/"),
        version=read_field_set_version(group.attrs),
        current_version=int(current_version),
    )


def record_provenance(
    file_path: str, path: str, current_version: int
) -> Optional[RecordProvenance]:
    """The :class:`RecordProvenance` of the record at *path*, or ``None`` if
    the file holds no such record."""
    with h5open(file_path, "r") as h5f:
        group = h5f.get(path)
        if not isinstance(group, h5py.Group):
            return None
        return group_provenance(group, current_version)


# ---------------------------------------------------------------------------
# Analysis-epoch stamps
# ---------------------------------------------------------------------------
def stamp_stage_epoch(
    h5f: h5py.File, stage_name: str, *, rerun: bool = False
) -> EnvironmentRecord:
    """Record the running environment (and so the analysis epoch) for
    *stage_name*, then report any drift. Returns the record written.

    The record is mandatory: a failure to capture or write it raises. Callers
    stamp *before* marking the stage complete, so a stage whose stamp failed is
    never reported complete.

    The report that follows is advisory and never raises. ``rerun`` says the
    stage was already complete before this call; with no prior stamp for it,
    that is a re-run over a result produced before environment recording, which
    no epoch gate can speak to, and it is warned about.
    """
    from ..core.environment import capture_environment
    from .environment_serialization import (
        load_stage_environments,
        save_stage_environment,
    )

    current = capture_environment()
    previous = load_stage_environments(h5f)
    save_stage_environment(h5f, stage_name, current)
    _report_epoch_drift(stage_name, current, previous, rerun=rerun)
    return current


def stamp_stage_epoch_in_file(
    file_path: str, stage_name: str, *, rerun: bool = False
) -> EnvironmentRecord:
    """:func:`stamp_stage_epoch` on a file path (opens it for append)."""
    with h5open(file_path, "a") as h5f:
        return stamp_stage_epoch(h5f, stage_name, rerun=rerun)


def _report_epoch_drift(
    stage_name: str,
    current: EnvironmentRecord,
    previous: Mapping[str, EnvironmentRecord],
    *,
    rerun: bool,
) -> None:
    """Warn about a legacy re-run and about a file that now mixes environments.

    Advisory: describing drift must never fail the stage it describes.
    """
    try:
        from ..core.environment import describe_environment_drift

        if rerun and stage_name not in previous:
            logger.warning(
                "%s was re-run over a result that carries no environment "
                "stamp (this file predates environment recording), so "
                "reproducibility against the original run cannot be verified: "
                "any numerical change between the version that produced it and "
                "%s applies silently. Compare the stage's outputs before and "
                "after if the original values matter.",
                stage_name,
                current.ftmwpipeline or "the running version",
            )

        # Compare against what was already on the file, excluding this stage's
        # own prior entry (re-running a stage legitimately replaces it).
        others = {k: v for k, v in previous.items() if k != stage_name}
        drift = describe_environment_drift(others, current)
        if drift:
            logger.warning(
                "%s was written by a different environment than this file's "
                "other stages; the file now mixes analysis environments. "
                "Differences: %s. Run 'ftmwpipeline info' for the full "
                "per-stage record.",
                stage_name,
                "; ".join(drift),
            )
    except Exception as exc:  # pragma: no cover - the report is never fatal
        logger.debug("Could not report environment drift for %s: %s", stage_name, exc)


__all__ = [
    "FIELD_SET_VERSION_ATTR",
    "RecordProvenance",
    "write_field_set_version",
    "read_field_set_version",
    "group_provenance",
    "record_provenance",
    "stamp_stage_epoch",
    "stamp_stage_epoch_in_file",
]
