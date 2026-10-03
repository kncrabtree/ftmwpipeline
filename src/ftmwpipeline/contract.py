"""The machine contract: version, missing-value vocabulary, and manifest.

This module is the single import point for the published, versioned surface a
program may rely on (normative spec: ``dev-docs/CONTRACT_STRATEGY.md``):

- :data:`CONTRACT_VERSION` -- the integer a client gates on (never
  ``__version__``).
- :class:`Absent` -- the two meanings of "no value" (``NOT_RUN`` and
  ``UNDEFINED``) every contract field uses instead of ``None`` / ``nan`` /
  ``-1``. Its wire and columnar forms are applied by
  :func:`ftmwpipeline.serialize.to_jsonable`.
- :data:`MANIFEST` -- the enumeration of every contract element (accessors,
  schema names, error codes, declared ``read_metadata`` keys and declared
  ``read_table`` tables/columns). Tests assert that everything declared here
  exists on all three interfaces and that nothing declared disappears.
- :func:`capabilities` -- the manifest as a payload, for clients.
- The typed error family (re-exported from :mod:`ftmwpipeline.file_manager`).

Adding to the contract
----------------------
Edit the ``_ACCESSORS`` / ``_SCHEMAS`` / ``_CODES`` / ``_METADATA_KEYS`` /
``_TABLES`` literals below -- that is the only place entries are declared --
and raise :data:`CONTRACT_VERSION` per the spec's versioning rules. Entries are
only ever appended; removing or renaming one is a breaking change.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Mapping, Tuple

from .file_manager import (
    ERROR_SCHEMA,
    AnalysisEpochMismatchError,
    IncompleteProvenanceError,
    NotFoundError,
    PipelineCompatibilityError,
    PipelineCorruptionError,
    PipelineExistsError,
    PipelineFileError,
    StageDependencyError,
)

#: The machine-contract version. ``0`` means "not yet announced": the contract
#: may still change incompatibly (pre-1.0.0 rules). The announcement sets it to
#: ``1``; every additive change after that raises it.
CONTRACT_VERSION: int = 0

#: Schema name of the :func:`capabilities` payload.
CAPABILITIES_SCHEMA = "ftmw/capabilities@1"

#: ``ftmw/<payload>@<n>``: lowercase payload name, positive integer revision.
SCHEMA_NAME_RE = re.compile(r"^ftmw/[a-z][a-z0-9_]*@[1-9][0-9]*$")


class Absent(enum.Enum):
    """Why a contract field has no value.

    ``NOT_RUN``
        The stage or quantity does not exist in this file (never computed,
        never tested).
    ``UNDEFINED``
        Computed, but the quantity has no value (e.g. chi2_r with zero degrees
        of freedom, a failed K-1 refit).

    The member value is the wire spelling used in the ``"<field>_absent"``
    sibling key; :attr:`status` is the code in a ``<column>__status`` column.
    Compare by identity (``x is Absent.NOT_RUN``). Members are deliberately not
    falsy-special and not ``None``-like: test for them explicitly.
    """

    NOT_RUN = "not_run"
    UNDEFINED = "undefined"

    @property
    def status(self) -> int:
        """The ``uint8`` columnar status code (``1`` not run, ``2`` undefined)."""
        return STATUS_NOT_RUN if self is Absent.NOT_RUN else STATUS_UNDEFINED


#: Columnar status codes (``<column>__status``, dtype ``uint8``).
STATUS_PRESENT: int = 0
STATUS_NOT_RUN: int = 1
STATUS_UNDEFINED: int = 2


@dataclass(frozen=True)
class ContractManifest:
    """Immutable enumeration of every declared contract element.

    Attributes
    ----------
    contract_version : int
        Equal to :data:`CONTRACT_VERSION`.
    accessors : tuple of str
        Public accessor names, each present on ``ftmwpipeline.api``, on
        :class:`~ftmwpipeline.Pipeline`, and through the CLI ``read`` object.
    schemas : tuple of str
        Payload schema names (``ftmw/<payload>@<n>``).
    codes : tuple of str
        Error codes a :class:`PipelineFileError` may carry.
    metadata_keys : tuple of str
        Declared ``read_metadata`` keys.
    tables : Mapping[str, tuple of str]
        Declared ``read_table`` tables and, per table, their declared columns.
        Read-only.
    """

    contract_version: int
    accessors: Tuple[str, ...]
    schemas: Tuple[str, ...]
    codes: Tuple[str, ...]
    metadata_keys: Tuple[str, ...]
    tables: Mapping[str, Tuple[str, ...]]

    def __post_init__(self) -> None:
        for group in ("accessors", "schemas", "codes", "metadata_keys"):
            values = getattr(self, group)
            if len(set(values)) != len(values):
                raise ValueError(f"duplicate entry in manifest {group}")
        for name in self.schemas:
            if not SCHEMA_NAME_RE.match(name):
                raise ValueError(f"malformed schema name: {name!r}")
        frozen = MappingProxyType({k: tuple(v) for k, v in self.tables.items()})
        object.__setattr__(self, "tables", frozen)


# --------------------------------------------------------------------------
# The declarations. This is the ONE place contract elements are added.
# --------------------------------------------------------------------------

_ACCESSORS: Tuple[str, ...] = ("capabilities",)

_SCHEMAS: Tuple[str, ...] = (
    ERROR_SCHEMA,
    CAPABILITIES_SCHEMA,
)

_CODES: Tuple[str, ...] = (
    StageDependencyError.code,  # "stage_not_run"
    NotFoundError.code,  # "not_found"
    IncompleteProvenanceError.code,  # "incomplete_provenance"
    PipelineCompatibilityError.code,  # "file_incompatible"
    PipelineCorruptionError.code,  # "file_corrupt"
    AnalysisEpochMismatchError.code,  # "epoch_mismatch"
    PipelineExistsError.code,  # "file_exists"
)

_METADATA_KEYS: Tuple[str, ...] = ()

_TABLES: Dict[str, Tuple[str, ...]] = {}

MANIFEST = ContractManifest(
    contract_version=CONTRACT_VERSION,
    accessors=_ACCESSORS,
    schemas=_SCHEMAS,
    codes=_CODES,
    metadata_keys=_METADATA_KEYS,
    tables=_TABLES,
)


def capabilities() -> Dict[str, Any]:
    """What this installation's machine contract offers.

    Returns
    -------
    dict
        ``{"schema": "ftmw/capabilities@1", "contract_version": int,
        "schemas": [...], "accessors": [...], "codes": [...]}``, read from
        :data:`MANIFEST`. Already JSON-able; file-independent.
    """
    return {
        "schema": CAPABILITIES_SCHEMA,
        "contract_version": MANIFEST.contract_version,
        "schemas": list(MANIFEST.schemas),
        "accessors": list(MANIFEST.accessors),
        "codes": list(MANIFEST.codes),
    }


__all__ = [
    "CONTRACT_VERSION",
    "CAPABILITIES_SCHEMA",
    "ERROR_SCHEMA",
    "SCHEMA_NAME_RE",
    "Absent",
    "STATUS_PRESENT",
    "STATUS_NOT_RUN",
    "STATUS_UNDEFINED",
    "ContractManifest",
    "MANIFEST",
    "capabilities",
    # Typed error family
    "PipelineFileError",
    "PipelineExistsError",
    "StageDependencyError",
    "PipelineCorruptionError",
    "PipelineCompatibilityError",
    "AnalysisEpochMismatchError",
    "NotFoundError",
    "IncompleteProvenanceError",
]
