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
  schema names, error codes, declared ``read_metadata`` keys, declared
  ``read_table`` tables/columns, declared result-type fields and frozen
  vocabularies). Tests assert that everything declared here
  exists on all three interfaces and that nothing declared disappears.
- :func:`capabilities` -- the manifest as a payload, for clients.
- The typed error family (re-exported from :mod:`ftmwpipeline.file_manager`).

Adding to the contract
----------------------
Edit the ``_ACCESSORS`` (name, binding, and any non-default Pipeline/CLI
spelling) / ``_SCHEMAS`` / ``_CODES`` / ``_METADATA_KEYS`` / ``_TABLES`` /
``_FIELDS`` / ``_VOCABULARIES`` literals below -- that is the only place entries are declared --
and raise :data:`CONTRACT_VERSION` per the spec's versioning rules. Entries are
only ever appended; removing or renaming one is a breaking change.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping, NamedTuple, Optional, Tuple, Union

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

#: Schema names of the FID-samples and display-units payloads.
FID_SAMPLES_SCHEMA = "ftmw/fid_samples@1"
DISPLAY_UNITS_SCHEMA = "ftmw/display_units@1"

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
    pipeline_names : Mapping[str, str]
        Per accessor, the :class:`~ftmwpipeline.Pipeline` method that serves
        it (the accessor's own name unless it is declared otherwise, e.g.
        ``get_pipeline_info`` is :meth:`Pipeline.info`). Keys equal
        :attr:`accessors`. Read-only.
    cli_verbs : Mapping[str, tuple of str]
        Per accessor, the CLI verb path that exposes it (``("read", name)``
        unless the existing exposure lives under another verb, e.g.
        ``("review", "log")``). Keys equal :attr:`accessors`. Read-only.
    fields : Mapping[str, tuple of str]
        Declared fields of each contract result type: a dataclass field, or,
        for a dict result, a produced key. Read-only.
    vocabularies : Mapping[str, tuple of str]
        Frozen closed vocabularies (e.g. the decision-log ``kind`` values),
        each checked against the code that produces its values. Read-only.
    """

    contract_version: int
    accessors: Tuple[str, ...]
    schemas: Tuple[str, ...]
    codes: Tuple[str, ...]
    metadata_keys: Tuple[str, ...]
    tables: Mapping[str, Tuple[str, ...]]
    file_bound: Mapping[str, bool] = field(default_factory=dict)
    pipeline_names: Mapping[str, str] = field(default_factory=dict)
    cli_verbs: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    fields: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    vocabularies: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)

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
        for attr in ("pipeline_names", "cli_verbs"):
            if set(getattr(self, attr)) - set(self.accessors):
                raise ValueError(f"{attr} names a non-accessor")
        object.__setattr__(
            self,
            "pipeline_names",
            MappingProxyType(
                {n: self.pipeline_names.get(n, n) for n in self.accessors}
            ),
        )
        object.__setattr__(
            self,
            "cli_verbs",
            MappingProxyType(
                {n: tuple(self.cli_verbs.get(n, ("read", n))) for n in self.accessors}
            ),
        )
        for attr in ("fields", "vocabularies"):
            members = getattr(self, attr)
            for key, values in members.items():
                if len(set(values)) != len(values):
                    raise ValueError(f"duplicate entry in manifest {attr}[{key!r}]")
            object.__setattr__(
                self,
                attr,
                MappingProxyType({k: tuple(v) for k, v in members.items()}),
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
    """One accessor declaration.

    ``file_bound`` says whether it reads a file. ``pipeline_name`` is the
    :class:`~ftmwpipeline.Pipeline` method when it differs from ``name``.
    ``cli`` is the existing CLI verb path when the accessor is not served by
    ``read <name>`` (an accessor is never given a second verb).
    """

    name: str
    file_bound: bool
    pipeline_name: Optional[str] = None
    cli: Optional[Tuple[str, ...]] = None


_ACCESSORS: Tuple[AccessorSpec, ...] = (
    AccessorSpec("capabilities", file_bound=False),
    # Already present; declared as contract (Wave 1, task 1.1).
    AccessorSpec("frequency_calibration", True, cli=("timebase", "state")),
    AccessorSpec("refit_snap_tol_mhz", True, cli=("review", "snap-tolerance")),
    AccessorSpec("read_metadata", True, cli=("read", "meta")),
    AccessorSpec("read_tables", True, cli=("read", "list")),
    AccessorSpec("read_table", True, cli=("read", "table")),
    AccessorSpec("settings_defaults", False, cli=("settings", "defaults")),
    AccessorSpec("settings_show", True, cli=("settings", "show")),
    AccessorSpec(
        "get_final_products",
        True,
        pipeline_name="final_products",
        cli=("report", "table"),
    ),
    AccessorSpec("review_log", True, cli=("review", "log")),
    AccessorSpec("get_pipeline_info", True, pipeline_name="info", cli=("info",)),
    AccessorSpec("compute_display_ft", True),
    AccessorSpec("fid_samples", file_bound=True),
    AccessorSpec("display_units", file_bound=True),
)

_SCHEMAS: Tuple[str, ...] = (
    ERROR_SCHEMA,
    CAPABILITIES_SCHEMA,
    FID_SAMPLES_SCHEMA,
    DISPLAY_UNITS_SCHEMA,
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

_FT_WINDOW_KEYS: Tuple[str, ...] = (
    "start_us",
    "end_us",
    "trim_min_mhz",
    "trim_max_mhz",
    "units_power",
    "acquisition_us",
)
_TAU_KEYS: Tuple[str, ...] = (
    "tau_maj_us",
    "sigma_tau_us",
    "n_contributors",
    "n_spur_bins",
    "n_seg",
    "preconditions_passed",
)

#: Declared ``read_metadata`` keys. The ``ft.`` section is emitted under both
#: ``ft.`` and its ``stage1.`` alias; the ``tau.`` / ``tau_g.`` / ``timebase.``
#: scalars exist only once their stage has run.
_METADATA_KEYS: Tuple[str, ...] = (
    "file.format_version",
    "file.completed_stages",
    "fid.n_points",
    "fid.duration_us",
    "fid.probe_freq_mhz",
    "fid.sideband",
    "fid.shots",
    *(f"ft.{k}" for k in _FT_WINDOW_KEYS),
    *(f"stage1.{k}" for k in _FT_WINDOW_KEYS),
    "stage3.n_peaks",
    "stage4.n_windows",
    "stage5.n_fitted_peaks",
    "stage5.acquisition_us",
    "stage5.shape",
    *(f"tau.{k}" for k in _TAU_KEYS),
    *(f"tau_g.{k}" for k in _TAU_KEYS),
    "timebase.epsilon",
    "timebase.sigma_epsilon",
    "timebase.kappa_sys",
    "timebase.lattice_g_mhz",
    "timebase.n_detected",
    "timebase.n_used",
    "timebase.preconditions_passed",
)

#: Declared ``read_table`` tables and the columns promised for each.
_TABLES: Dict[str, Tuple[str, ...]] = {
    "fit_peaks": (
        "detection_index",
        "window_id",
        "shape",
        "frequency_mhz",
        "frequency_error",
        "amplitude",
        "amplitude_error",
        "phase",
        "phase_error",
        "decay_rate",
        "decay_rate_error",
        "snr",
        "chi_squared",
        "origin",
        "clock_lattice",
        "flat_decay",
        "derivation",
        "peak_uid",
        "knockout_delta_chi2",
        "knockout_expected_delta_chi2",
        "knockout_supported",
        "knockout_p_value",
        "knockout_n_eff",
        "knockout_aicc_delta",
        "unresolved_spread_mhz",
    ),
    "windows": (
        "window_id",
        "freq_min",
        "freq_max",
        "batch",
        "n_free_peaks",
        "n_fixed_contributors",
    ),
}

#: Declared fields of the contract result types: a dataclass field, or a
#: produced key for a dict result (``PipelineInfo``, ``ComplexFT.metadata``).
_FIELDS: Dict[str, Tuple[str, ...]] = {
    "CalibrationStamp": (
        "state",
        "epsilon",
        "sigma_epsilon",
        "sigma_floor_khz",
        "probe_freq_mhz",
        "sideband",
    ),
    "FinalPeak": (
        "peak_uid",
        "window_id",
        "origin",
        "derivation",
        "clock_lattice",
        "knockout_p_value",
        "knockout_supported",
        "knockout_aicc_delta",
        "frequency_mhz",
        "frequency_raw_mhz",
        "f_baseband_mhz",
        "sigma_f_khz",
        "sigma_stat_khz",
        "sigma_eps_khz",
        "sigma_floor_khz",
    ),
    "DecisionLogEntry": (
        "order_index",
        "window_id",
        "frequency_mhz",
        "kind",
        "provenance",
        "evidence",
    ),
    "RefitWindowResult": ("converged",),
    "PreviewWindowResult": ("converged",),
    "AppliedWindowResult": ("converged",),
    "PipelineInfo": (
        "stage_environments",
        "last_written_with",
        "environment_drift",
        "runtime_environment_drift",
        "current_environment",
        "environment_acknowledged",
        "warnings",
    ),
    "ComplexFT": ("freq_array", "complex_spectrum", "metadata"),
    "ComplexFT.metadata": ("amplitude_scale", "units_label", "pad_factor"),
}

#: Frozen closed vocabularies. ``decision_kind`` / ``decision_provenance`` are
#: checked against ``core.data_structures.DECISION_KINDS`` /
#: ``DECISION_PROVENANCES``, which the Stage 6 code records from.
_VOCABULARIES: Dict[str, Tuple[str, ...]] = {
    "decision_kind": ("add", "remove", "merge", "split", "accept", "create_window"),
    "decision_provenance": ("user",),
}

MANIFEST = ContractManifest(
    contract_version=CONTRACT_VERSION,
    accessors=tuple(spec.name for spec in _ACCESSORS),
    schemas=_SCHEMAS,
    codes=_CODES,
    metadata_keys=_METADATA_KEYS,
    tables=_TABLES,
    file_bound={spec.name: spec.file_bound for spec in _ACCESSORS},
    pipeline_names={s.name: s.pipeline_name for s in _ACCESSORS if s.pipeline_name},
    cli_verbs={s.name: s.cli for s in _ACCESSORS if s.cli},
    fields=_FIELDS,
    vocabularies=_VOCABULARIES,
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
