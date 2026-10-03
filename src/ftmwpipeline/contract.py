"""The machine contract: version, missing-value vocabulary, and manifest.

This module is the single import point for the published, versioned surface a
program may rely on (normative spec: ``dev-docs/CONTRACT_STRATEGY.md``):

- :data:`CONTRACT_VERSION` -- the integer a client gates on (never
  ``__version__``).
- :class:`Stage` -- the canonical stage vocabulary every contract payload
  uses to name a stage, with :func:`stage_for_key` / :func:`key_for_stage`
  mapping to and from the internal storage keys.
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
Edit the ``_ACCESSORS`` (name and binding) / ``_SCHEMAS`` / ``_CODES`` / ``_METADATA_KEYS`` /
``_TABLES`` literals below -- that is the only place entries are declared --
and raise :data:`CONTRACT_VERSION` per the spec's versioning rules. Entries are
only ever appended; removing or renaming one is a breaking change.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping, NamedTuple, Tuple, Union

from .file_manager import (
    ERROR_SCHEMA,
    AnalysisEpochMismatchError,
    IncompleteProvenanceError,
    NotFoundError,
    PipelineCompatibilityError,
    PipelineCorruptionError,
    PipelineExistsError,
    PipelineFileError,
    PipelineFileNotFoundError,
    StageDependencyError,
)

#: The machine-contract version. The first published contract is ``1``; each
#: release that adds (or, before 1.0.0, changes) contract elements raises it by
#: one, so a client can gate on it as well as on :func:`capabilities`.
CONTRACT_VERSION: int = 1

#: Schema name of the :func:`capabilities` payload.
CAPABILITIES_SCHEMA = "ftmw/capabilities@1"

#: Schema name of the :func:`fit_thresholds` payload.
FIT_THRESHOLDS_SCHEMA = "ftmw/fit_thresholds@1"

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


class Stage(str, enum.Enum):
    """The canonical stage vocabulary.

    Values are the CLI object names. Every contract payload that names a stage
    (e.g. ``missing_dependencies`` of a ``stage_not_run`` error) uses these
    values. :func:`stage_for_key` / :func:`key_for_stage` map to and from the
    internal storage keys (``PipelineStageTracker.STAGE_DEPENDENCIES``).
    """

    DATA = "data"
    FT = "ft"
    NOISE = "noise"
    TAU = "tau"
    TAU_G = "tau_g"
    TIMEBASE = "timebase"
    PEAKS = "peaks"
    WINDOWS = "windows"
    FIT = "fit"
    REVIEW = "review"


#: Internal storage key of each canonical stage (read-only). Covers every key
#: of ``PipelineStageTracker.STAGE_DEPENDENCIES``.
STAGE_KEYS: Mapping[Stage, str] = MappingProxyType(
    {
        Stage.DATA: "stage0_fid_data",
        Stage.FT: "stage1_complex_ft",
        Stage.NOISE: "stage2_noise_result",
        Stage.TAU: "stage2b_tau_calibration",
        Stage.TAU_G: "stage2b_tau_G_calibration",
        Stage.TIMEBASE: "timebase_calibration",
        Stage.PEAKS: "stage3_peaks",
        Stage.WINDOWS: "stage4_windows",
        Stage.FIT: "stage5_fitting",
        Stage.REVIEW: "stage6_review",
    }
)

_STAGE_BY_KEY: Mapping[str, Stage] = MappingProxyType(
    {key: stage for stage, key in STAGE_KEYS.items()}
)


def stage_for_key(key: str) -> Stage:
    """The canonical :class:`Stage` for an internal storage key.

    Raises
    ------
    ValueError
        If ``key`` has no canonical stage. Never passes an internal spelling
        through.
    """
    try:
        return _STAGE_BY_KEY[key]
    except KeyError:
        raise ValueError(f"no canonical stage for internal key {key!r}") from None


def key_for_stage(stage: Union[Stage, str]) -> str:
    """The internal storage key for a canonical stage (or its value string).

    Raises
    ------
    ValueError
        If ``stage`` is not a canonical stage.
    """
    return STAGE_KEYS[Stage(stage)]


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
    file_bound : Mapping[str, bool]
        Per accessor, whether it reads a file. A file-bound accessor is a
        :class:`~ftmwpipeline.Pipeline` instance method taking no path, an
        ``api`` function whose first parameter is the path, and a ``read`` verb
        with a file argument. A file-less one is a ``Pipeline`` staticmethod
        and an ``api`` function, both without a path, and a ``read`` verb
        without a file argument. Keys equal :attr:`accessors`. Read-only.
    """

    contract_version: int
    accessors: Tuple[str, ...]
    schemas: Tuple[str, ...]
    codes: Tuple[str, ...]
    metadata_keys: Tuple[str, ...]
    tables: Mapping[str, Tuple[str, ...]]
    file_bound: Mapping[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for group in ("accessors", "schemas", "codes", "metadata_keys"):
            values = getattr(self, group)
            if len(set(values)) != len(values):
                raise ValueError(f"duplicate entry in manifest {group}")
        if set(self.file_bound) != set(self.accessors):
            raise ValueError("file_bound must name exactly the manifest accessors")
        object.__setattr__(
            self,
            "file_bound",
            MappingProxyType(
                {name: bool(self.file_bound[name]) for name in self.accessors}
            ),
        )
        for name in self.schemas:
            if not SCHEMA_NAME_RE.match(name):
                raise ValueError(f"malformed schema name: {name!r}")
        frozen = MappingProxyType({k: tuple(v) for k, v in self.tables.items()})
        object.__setattr__(self, "tables", frozen)


# --------------------------------------------------------------------------
# The declarations. This is the ONE place contract elements are added.
# --------------------------------------------------------------------------


class AccessorSpec(NamedTuple):
    """One accessor declaration: its name and whether it reads a file."""

    name: str
    file_bound: bool


_ACCESSORS: Tuple[AccessorSpec, ...] = (
    AccessorSpec("capabilities", file_bound=False),
    AccessorSpec("fit_thresholds", file_bound=True),
)

_SCHEMAS: Tuple[str, ...] = (
    ERROR_SCHEMA,
    CAPABILITIES_SCHEMA,
    FIT_THRESHOLDS_SCHEMA,
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
    accessors=tuple(spec.name for spec in _ACCESSORS),
    schemas=_SCHEMAS,
    codes=_CODES,
    metadata_keys=_METADATA_KEYS,
    tables=_TABLES,
    file_bound={spec.name: spec.file_bound for spec in _ACCESSORS},
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
    "FIT_THRESHOLDS_SCHEMA",
    "ERROR_SCHEMA",
    "SCHEMA_NAME_RE",
    "Absent",
    "Stage",
    "STAGE_KEYS",
    "stage_for_key",
    "key_for_stage",
    "AccessorSpec",
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
    "PipelineFileNotFoundError",
    "IncompleteProvenanceError",
]
