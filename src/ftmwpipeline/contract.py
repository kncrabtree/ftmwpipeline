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
- The schema-name constants (``*_SCHEMA``) and the entity-table records
  (:class:`WindowStatusRow`, :class:`FidPreviewRow`).
- The typed error family (re-exported from :mod:`ftmwpipeline.file_manager`).
- The event types a long operation delivers to its ``events`` callback
  (:class:`StageStarted`, :class:`StageFinished`, :class:`WindowProgress`,
  :class:`ScanProgress`, :class:`Invalidated`, :class:`PipelineWarning`; their
  union :data:`Event`) and the :class:`CancelToken` protocol its ``cancel``
  argument satisfies.

Adding to the contract
----------------------
Edit the ``_ACCESSORS`` (name, binding, and any non-default Pipeline
spelling) / ``_SCHEMAS`` / ``_CODES`` / ``_METADATA_KEYS`` / ``_TABLES`` /
``_FIELDS`` / ``_VOCABULARIES`` literals below -- that is the only place entries are declared --
and raise :data:`CONTRACT_VERSION` per the spec's versioning rules. Entries are
only ever appended; removing or renaming one is a breaking change.
"""

from __future__ import annotations

import enum
import math
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

from .core.absent import STATUS_NOT_RUN, STATUS_PRESENT, STATUS_UNDEFINED, Absent
from .core.calibration import CalibrationStamp
from .core.curation import CURATION_ACTION_SCHEMA, CurationAction
from .core.data_structures import ATTENTION_KINDS, FinalProducts
from .file_manager import (
    ERROR_SCHEMA,
    AlgorithmFailedError,
    AnalysisEpochMismatchError,
    BadSettingError,
    CallbackFailedError,
    CurationConflictError,
    IncompleteProvenanceError,
    NotFoundError,
    NotFoundValueError,
    OperationCancelledError,
    PipelineCompatibilityError,
    PipelineCorruptionError,
    PipelineExistsError,
    PipelineFileError,
    PipelineFileNotFoundError,
    PipelineStageTracker,
    StageDependencyError,
    WriteConflictError,
)

#: The machine-contract version. The first published contract is ``1``; each
#: release that adds (or, before 1.0.0, changes) contract elements raises it by
#: one, so a client can gate on it as well as on :func:`capabilities`.
CONTRACT_VERSION: int = 14

#: Schema name of the :func:`capabilities` payload.
CAPABILITIES_SCHEMA = "ftmw/capabilities@1"

#: Schema names of the accessor payloads. The API, ``Pipeline`` and the CLI
#: return the same stamped object: a dict payload carries the name under its
#: ``"schema"`` key, a dataclass payload declares it as ``__ftmw_schema__``.
FID_SAMPLES_SCHEMA = "ftmw/fid_samples@1"
DISPLAY_UNITS_SCHEMA = "ftmw/display_units@1"
FIT_THRESHOLDS_SCHEMA = "ftmw/fit_thresholds@1"
WINDOW_STATUS_SCHEMA = "ftmw/window_status@1"
SOURCE_PREVIEW_SCHEMA = "ftmw/source_preview@1"
WINDOW_MODEL_SCHEMA = "ftmw/window_model@1"
SPECTRUM_MODEL_SCHEMA = "ftmw/spectrum_model@1"
#: Frozen once published: a different definition is published as ``@2``.
ANALYSIS_FINGERPRINT_SCHEMA = "ftmw/analysis_fingerprint@1"
STATUS_SCHEMA = "ftmw/status@1"

#: Schema names of the declared existing accessors. Their Python results are
#: unchanged (a dataclass, list, scalar or plain dict); the CLI envelope stamps
#: them: a dict directly, a list as ``{"schema", "items": [...]}``, a scalar as
#: ``{"schema", "value": x}``, an object as its fields plus ``"schema"``.
#: (``CalibrationStamp`` and ``FinalProducts`` declare their own
#: ``__ftmw_schema__``; the constants here are read from them.)
CALIBRATION_SCHEMA: str = CalibrationStamp.__ftmw_schema__
SNAP_TOLERANCE_SCHEMA = "ftmw/snap_tolerance@1"
METADATA_SCHEMA = "ftmw/metadata@1"
TABLES_SCHEMA = "ftmw/tables@1"
TABLE_SCHEMA = "ftmw/table@1"
SETTINGS_DEFAULTS_SCHEMA = "ftmw/settings_defaults@1"
SETTINGS_SCHEMA = "ftmw/settings@1"
FINAL_PRODUCTS_SCHEMA: str = FinalProducts.__ftmw_schema__
REVIEW_LOG_SCHEMA = "ftmw/review_log@1"
PIPELINE_INFO_SCHEMA = "ftmw/pipeline_info@1"
DISPLAY_FT_SCHEMA = "ftmw/display_ft@1"
RUN_RESULT_SCHEMA = "ftmw/run_result@1"
#: Schema names of the events a long operation delivers (Wave 5.1).
STAGE_STARTED_SCHEMA = "ftmw/stage_started@1"
STAGE_FINISHED_SCHEMA = "ftmw/stage_finished@1"
WINDOW_PROGRESS_SCHEMA = "ftmw/window_progress@1"
SCAN_PROGRESS_SCHEMA = "ftmw/scan_progress@1"
INVALIDATED_SCHEMA = "ftmw/invalidated@1"
WARNING_SCHEMA = "ftmw/warning@1"
# CURATION_ACTION_SCHEMA (imported above) names a CurationAction's wire form:
# a request type, not a result -- review_apply / review_preview take a
# sequence of these in place of a curation file.

#: ``ftmw/<payload>@<n>``: lowercase payload name, positive integer revision.
SCHEMA_NAME_RE = re.compile(r"^ftmw/[a-z][a-z0-9_]*@[1-9][0-9]*$")


# --------------------------------------------------------------------------
# Entity-table records. An accessor that returns one row per window or FID
# returns a list of these; an absent field is an Absent per row.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowStatusRow:
    """One row of :func:`~ftmwpipeline.api.window_status`.

    A window of the plan the fit was made on, or a window Stage 6 created
    (``created``). Before a complete Stage 5 fit, and for a fit no structural
    merge revised, the plan is the Stage 4 plan; after a merge the survivor's
    row carries the merged range and ``merged_from`` names the ids it absorbed,
    ascending (empty for every other row; a JSON list), and no row carries an
    absorbed id. A window
    is ``live`` when the Stage 5 fit holds at least one fitted line in it.
    Before Stage 5, ``n_fitted_peaks`` and ``live`` are :attr:`Absent.NOT_RUN`.
    """

    window_id: int
    freq_min_mhz: float
    freq_max_mhz: float
    created: bool
    n_fitted_peaks: Union[int, Absent]
    live: Union[bool, Absent]
    merged_from: Tuple[int, ...] = ()


@dataclass(frozen=True)
class FidPreviewRow:
    """One row of a source's FID table (:func:`~ftmwpipeline.api.preview_source`).

    Every field except ``index`` is a value or an :class:`Absent`:
    ``NOT_RUN`` when the source does not declare it (an import default is never
    reported in its place), ``UNDEFINED`` when it cannot be stated without load
    parameters (or is not finite). ``sideband`` is ``"upper"`` or ``"lower"``.
    ``channel`` is the source's channel identifier, spelled exactly as import's
    ``--channel`` / ``channel=`` takes it (a Keysight record's ``"Channel_3"``),
    and ``NOT_RUN`` for a source without channels.
    """

    index: int
    n_points: Union[int, Absent] = Absent.NOT_RUN
    spacing_us: Union[float, Absent] = Absent.NOT_RUN
    probe_freq_mhz: Union[float, Absent] = Absent.NOT_RUN
    sideband: Union[str, Absent] = Absent.NOT_RUN
    shots: Union[int, Absent] = Absent.NOT_RUN
    channel: Union[str, Absent] = Absent.NOT_RUN


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


#: Provenance entries that are not a stage's own storage key but are stamped in
#: the per-stage environment map: the published name of each (read-only). The
#: HDF5 key stays as written; only the name a report publishes differs.
PROVENANCE_NAMES: Mapping[str, str] = MappingProxyType(
    {"stage2b_shape_recommendation": "tau_shape"}
)


def canonical_provenance_name(key: str) -> str:
    """The published name for a per-stage provenance key.

    A stage's storage key gives its canonical stage name
    (:func:`stage_for_key`); the shape recommendation's environment entry
    (``stage2b_shape_recommendation``) is published as ``tau_shape``. A key that
    is neither is returned unchanged (a record may carry keys from a later
    release).
    """
    named = PROVENANCE_NAMES.get(key)
    if named is not None:
        return named
    try:
        return stage_for_key(key).value
    except ValueError:
        return key


def key_for_stage(stage: Union[Stage, str]) -> str:
    """The internal storage key for a canonical stage (or its value string).

    Raises
    ------
    ValueError
        If ``stage`` is not a canonical stage.
    """
    return STAGE_KEYS[Stage(stage)]


#: Settings / preset prefix of each canonical stage (read-only): the
#: ``_StageSpec.prefix`` of ``settings_inspection`` and the key of
#: ``settings_mutation._MUT_SPECS``. ``tau`` and ``tau_g`` share ``stage2b``.
#: ``None``: the stage has no settings record (Stage 0 has none, the timebase
#: has no preset-driven settings, Stage 6 takes curation actions).
STAGE_SETTINGS_PREFIX: Mapping[Stage, Optional[str]] = MappingProxyType(
    {
        Stage.DATA: None,
        Stage.FT: "stage1",
        Stage.NOISE: "stage2",
        Stage.TAU: "stage2b",
        Stage.TAU_G: "stage2b",
        Stage.TIMEBASE: None,
        Stage.PEAKS: "stage3",
        Stage.WINDOWS: "stage4",
        Stage.FIT: "stage5",
        Stage.REVIEW: None,
    }
)

#: Tuning-registry knob prefix of each canonical stage (read-only): the leading
#: dotted segment of the ``KnobSpec.path`` values that feed that stage. Stage 0's
#: knobs are the start-detection sweep (``stage0.*``). ``tau`` and ``tau_g``
#: share ``stage2b``: the twins (and the shape recommendation) run one STFT
#: classifier recipe, so its knobs feed both, and the registry labels every
#: ``stage2b`` knob alike. Not one-to-one, so there is no inverse. ``None``:
#: the stage has no registered knob.
STAGE_KNOB_PREFIX: Mapping[Stage, Optional[str]] = MappingProxyType(
    {
        Stage.DATA: "stage0",
        Stage.FT: "stage1",
        Stage.NOISE: "stage2",
        Stage.TAU: "stage2b",
        Stage.TAU_G: "stage2b",
        Stage.TIMEBASE: None,
        Stage.PEAKS: "stage3",
        Stage.WINDOWS: "stage4",
        Stage.FIT: "stage5",
        Stage.REVIEW: None,
    }
)


def stage_depends_on(stage: Union[Stage, str]) -> Tuple[Stage, ...]:
    """The stages ``stage`` requires, canonical and in enum order.

    Read from ``PipelineStageTracker.STAGE_DEPENDENCIES``.
    """
    key = key_for_stage(stage)
    deps = {_STAGE_BY_KEY[k] for k in PipelineStageTracker.STAGE_DEPENDENCIES[key]}
    return tuple(s for s in Stage if s in deps)


def rerun_order() -> Tuple[Stage, ...]:
    """Every stage in the dependency-respecting order a full refresh follows.

    A fixed topological order: of the stages whose dependencies are all placed,
    the one earliest in the :class:`Stage` enum comes next.
    """
    order: list = []
    remaining = list(Stage)
    while remaining:
        nxt = next(s for s in remaining if all(d in order for d in stage_depends_on(s)))
        order.append(nxt)
        remaining.remove(nxt)
    return tuple(order)


# --------------------------------------------------------------------------
# Events and cancellation (CONTRACT_STRATEGY §Events and cancellation).
# --------------------------------------------------------------------------


@runtime_checkable
class CancelToken(Protocol):
    """What a long operation's ``cancel`` argument must offer.

    One method, :meth:`is_set`; a :class:`threading.Event` qualifies. The
    operation polls it at its check points (before and between stages, between
    windows, between scan values) and raises
    :class:`~ftmwpipeline.file_manager.OperationCancelledError` once it is set.
    """

    def is_set(self) -> bool:
        """True once the caller wants the operation to stop."""
        ...


#: Codes of :class:`PipelineWarning`, each with the code-specific fields its
#: wire form carries (beside ``code`` and ``message``). The ``warning_code``
#: vocabulary is the keys; additions are additive.
WARNING_FIELDS: Mapping[str, Tuple[str, ...]] = MappingProxyType(
    {
        "slow_window": ("window_id", "elapsed_s", "threshold_s"),
        "epoch_acknowledged": ("file_epoch", "current_epoch"),
        "environment_drift": ("fields",),
        "frame_mismatch": ("actions",),
        "walk_fallback": ("reason", "n_windows"),
        "timebase_skipped": (),
    }
)

#: The ``phase`` of a :class:`WindowProgress` pass: the fit's first walk (or a
#: Stage 6 call's own windows), a structural replan round, a sequential re-walk
#: after a parallel walk fell back, and Stage 6's re-fit of dependent windows.
WINDOW_PHASES: Tuple[str, ...] = ("initial", "replan", "fallback", "cascade")


def _stage_or_none(stage: Union[Stage, str, None]) -> Optional[Stage]:
    return None if stage is None else Stage(stage)


@dataclass(frozen=True)
class _EventBase:
    """Fields every event carries.

    ``schema`` is set from the class (never passed); ``operation`` is the CLI
    verb of the call that emits it (``"fit run"``, ``"run"``); ``stage`` the
    canonical stage it concerns, or ``None`` (``null`` on the wire) where none
    applies. Construction normalizes a stage string to :class:`Stage`.
    """

    __ftmw_schema__: ClassVar[str] = ""

    schema: str = field(init=False)
    operation: str
    stage: Optional[Stage]

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema", type(self).__ftmw_schema__)
        object.__setattr__(self, "stage", _stage_or_none(self.stage))


@dataclass(frozen=True)
class StageStarted(_EventBase):
    """A stage began. Emitted first for every stage, after the cancel check."""

    __ftmw_schema__: ClassVar[str] = STAGE_STARTED_SCHEMA


@dataclass(frozen=True)
class StageFinished(_EventBase):
    """A stage finished and its results are written.

    ``summary`` has exactly the keys of the same verb's ``ftmw/run_result@1``
    summary (one builder makes both). Never emitted for a cancelled or failed
    stage.
    """

    __ftmw_schema__: ClassVar[str] = STAGE_FINISHED_SCHEMA

    elapsed_s: float
    summary: Mapping[str, Any] = field(hash=False)


@dataclass(frozen=True)
class WindowProgress(_EventBase):
    """One window finished fitting.

    Events come in passes, each identified by its ``(phase, round)`` pair.
    ``phase`` is one of :data:`WINDOW_PHASES`: ``"initial"`` (the fit's first
    walk, or a Stage 6 call's own windows), ``"replan"`` (a structural replan
    round), ``"fallback"`` (a sequential re-walk after the parallel walk of the
    same round fell back; it reports windows that round's earlier pass already
    reported) or ``"cascade"`` (Stage 6's re-fit of dependent windows).
    ``round`` is ``0`` for the initial walk, its fallback and every Stage 6
    pass, and the replan round's number (from 1) for a replan round and its
    fallback. ``index`` counts finished windows within the pass from 1 (windows
    can finish out of id order); ``total`` is the number of windows in the
    pass, fixed when it begins; ``elapsed_s`` the window's own fitting time. A
    dropped window has ``dropped=True`` and its ``n_peaks`` / ``chi2r`` are
    :class:`Absent`; a ``chi2r`` without a finite value is
    :attr:`Absent.UNDEFINED`.
    """

    __ftmw_schema__: ClassVar[str] = WINDOW_PROGRESS_SCHEMA

    phase: str
    round: int
    index: int
    total: int
    window_id: int
    n_peaks: Union[int, Absent]
    chi2r: Union[float, Absent]
    elapsed_s: float
    dropped: bool

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.phase not in WINDOW_PHASES:
            raise ValueError(
                f"phase must be one of {WINDOW_PHASES}, not {self.phase!r}"
            )
        if isinstance(self.chi2r, float) and not math.isfinite(self.chi2r):
            object.__setattr__(self, "chi2r", Absent.UNDEFINED)


@dataclass(frozen=True)
class ScanProgress(_EventBase):
    """One scanned value finished; ``knob`` is a registry path."""

    __ftmw_schema__: ClassVar[str] = SCAN_PROGRESS_SCHEMA

    knob: str
    value: Any = field(hash=False)
    index: int
    total: int


@dataclass(frozen=True)
class Invalidated(_EventBase):
    """The call invalidated ``stages`` (canonical, in ``rerun_order``).

    Emitted once per call that invalidates something, at the moment it does;
    equal to the result's ``invalidated``.
    """

    __ftmw_schema__: ClassVar[str] = INVALIDATED_SCHEMA

    stages: Tuple[Stage, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "stages", tuple(Stage(s) for s in self.stages))


@dataclass(frozen=True)
class PipelineWarning(_EventBase):
    """A typed warning: ``code`` from the ``warning_code`` vocabulary.

    The code-specific fields (:data:`WARNING_FIELDS`) are held in ``details``
    in Python and are flattened beside ``code`` and ``message`` on the wire;
    ``details`` must carry exactly the fields its code declares.
    """

    __ftmw_schema__: ClassVar[str] = WARNING_SCHEMA

    code: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        try:
            declared = WARNING_FIELDS[self.code]
        except KeyError:
            raise ValueError(f"unknown warning code {self.code!r}") from None
        if set(self.details) != set(declared):
            raise ValueError(
                f"warning {self.code!r} carries fields {sorted(declared)}, "
                f"not {sorted(self.details)}"
            )
        object.__setattr__(self, "details", dict(self.details))

    def __ftmw_items__(self) -> Tuple[Tuple[str, Any], ...]:
        """The wire form's fields: the common ones, then ``details`` flattened."""
        head = tuple(
            (name, getattr(self, name))
            for name in ("schema", "operation", "stage", "code", "message")
        )
        return head + tuple((k, self.details[k]) for k in WARNING_FIELDS[self.code])


#: Any event a long operation delivers to its ``events`` callback.
Event = Union[
    StageStarted,
    StageFinished,
    WindowProgress,
    ScanProgress,
    Invalidated,
    PipelineWarning,
]

#: The ``events`` argument of a long operation.
EventCallback = Callable[[Event], None]

#: Every event type, in declaration order.
EVENT_TYPES: Tuple[type, ...] = (
    StageStarted,
    StageFinished,
    WindowProgress,
    ScanProgress,
    Invalidated,
    PipelineWarning,
)


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
        :attr:`accessors`. Read-only. Every accessor's CLI verb is
        ``read <name>`` (spelled as the accessor name), so no verb mapping is
        declared.
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
        if set(self.pipeline_names) - set(self.accessors):
            raise ValueError("pipeline_names names a non-accessor")
        object.__setattr__(
            self,
            "pipeline_names",
            MappingProxyType(
                {n: self.pipeline_names.get(n, n) for n in self.accessors}
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
    :class:`~ftmwpipeline.Pipeline` method when it differs from ``name``. The
    CLI verb is always ``read <name>``.
    """

    name: str
    file_bound: bool
    pipeline_name: Optional[str] = None


_ACCESSORS: Tuple[AccessorSpec, ...] = (
    AccessorSpec("capabilities", file_bound=False),
    # Already present; declared as contract (Wave 1, task 1.1).
    AccessorSpec("frequency_calibration", True),
    AccessorSpec("refit_snap_tol_mhz", True),
    AccessorSpec("read_metadata", True),
    AccessorSpec("read_tables", True),
    AccessorSpec("read_table", True),
    AccessorSpec("settings_defaults", False),
    AccessorSpec("settings_show", True),
    AccessorSpec("get_final_products", True, pipeline_name="final_products"),
    AccessorSpec("review_log", True),
    AccessorSpec("get_pipeline_info", True, pipeline_name="info"),
    AccessorSpec("compute_display_ft", True),
    AccessorSpec("fid_samples", file_bound=True),
    AccessorSpec("display_units", file_bound=True),
    AccessorSpec("fit_thresholds", file_bound=True),
    AccessorSpec("window_status", file_bound=True),
    AccessorSpec("preview_source", file_bound=False),
    # The fitted model, evaluated (Wave 2).
    AccessorSpec("window_model", file_bound=True),
    AccessorSpec("spectrum_model", file_bound=True),
    # The analysis fingerprint (Wave 2c).
    AccessorSpec("analysis_fingerprint", file_bound=True),
    # Per-stage state, runnable set and refresh order (Wave 7).
    AccessorSpec("status", file_bound=True),
)

_SCHEMAS: Tuple[str, ...] = (
    ERROR_SCHEMA,
    CAPABILITIES_SCHEMA,
    FID_SAMPLES_SCHEMA,
    DISPLAY_UNITS_SCHEMA,
    FIT_THRESHOLDS_SCHEMA,
    WINDOW_STATUS_SCHEMA,
    SOURCE_PREVIEW_SCHEMA,
    WINDOW_MODEL_SCHEMA,
    SPECTRUM_MODEL_SCHEMA,
    ANALYSIS_FINGERPRINT_SCHEMA,
    STATUS_SCHEMA,
    CALIBRATION_SCHEMA,
    SNAP_TOLERANCE_SCHEMA,
    METADATA_SCHEMA,
    TABLES_SCHEMA,
    TABLE_SCHEMA,
    SETTINGS_DEFAULTS_SCHEMA,
    SETTINGS_SCHEMA,
    FINAL_PRODUCTS_SCHEMA,
    REVIEW_LOG_SCHEMA,
    PIPELINE_INFO_SCHEMA,
    DISPLAY_FT_SCHEMA,
    CURATION_ACTION_SCHEMA,
    RUN_RESULT_SCHEMA,
    STAGE_STARTED_SCHEMA,
    STAGE_FINISHED_SCHEMA,
    WINDOW_PROGRESS_SCHEMA,
    SCAN_PROGRESS_SCHEMA,
    INVALIDATED_SCHEMA,
    WARNING_SCHEMA,
)

_CODES: Tuple[str, ...] = (
    StageDependencyError.code,  # "stage_not_run"
    NotFoundError.code,  # "not_found"
    IncompleteProvenanceError.code,  # "incomplete_provenance"
    PipelineCompatibilityError.code,  # "file_incompatible"
    PipelineCorruptionError.code,  # "file_corrupt"
    AnalysisEpochMismatchError.code,  # "epoch_mismatch"
    PipelineExistsError.code,  # "file_exists"
    BadSettingError.code,  # "bad_setting"
    AlgorithmFailedError.code,  # "algorithm_failed"
    OperationCancelledError.code,  # "cancelled"
    CallbackFailedError.code,  # "callback_failed"
    WriteConflictError.code,  # "write_conflict"
    CurationConflictError.code,  # "curation_conflict"
    # The base class's declared fallback: run_pipeline reports a failure that
    # is not a typed error under it (§Events and cancellation).
    PipelineFileError.code,  # "pipeline_error"
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
#: ``ft.`` and its ``stage1.`` alias. A declared key of a stage that has not
#: run is omitted (read with ``.get()``); a key that is present without a value
#: is an :class:`Absent`, never ``None``. The ``ft.`` settings echoes of an
#: unset bound or trim stay ``None``.
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
        "detection_index__status",
        "window_id",
        "shape",
        "frequency_mhz",
        "frequency_error",
        "frequency_error__status",
        "amplitude",
        "amplitude_error",
        "amplitude_error__status",
        "phase",
        "phase__status",
        "phase_error",
        "phase_error__status",
        "decay_rate",
        "decay_rate__status",
        "decay_rate_error",
        "decay_rate_error__status",
        "snr",
        "snr__status",
        "chi_squared",
        "chi_squared__status",
        "origin",
        "clock_lattice",
        "clock_lattice__status",
        "flat_decay",
        "derivation",
        "derivation__status",
        "peak_uid",
        "peak_uid__status",
        "knockout_delta_chi2",
        "knockout_delta_chi2__status",
        "knockout_expected_delta_chi2",
        "knockout_expected_delta_chi2__status",
        "knockout_supported",
        "knockout_supported__status",
        "knockout_p_value",
        "knockout_p_value__status",
        "knockout_n_eff",
        "knockout_n_eff__status",
        "knockout_aicc_delta",
        "knockout_aicc_delta__status",
        "unresolved_spread_mhz",
        "unresolved_spread_mhz__status",
    ),
    "windows": (
        "window_id",
        "freq_min",
        "freq_max",
        "batch",
        "n_free_peaks",
        "n_fixed_contributors",
    ),
    "window_status": (
        "window_id",
        "freq_min_mhz",
        "freq_max_mhz",
        "created",
        "n_fitted_peaks",
        "n_fitted_peaks__status",
        "live",
        "live__status",
        "merged_from",
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
        # Per-line fit fields (Wave 2), joined from the Stage 5 fit of the
        # line's window; Absent from the start.
        "decay_time_us",
        "decay_time_error_us",
        "shape",
        "fwhm_mhz",
        "detection_index",
        "fit_window_mhz",
    ),
    "WindowStatusRow": (
        "window_id",
        "freq_min_mhz",
        "freq_max_mhz",
        "created",
        "n_fitted_peaks",
        "live",
        "merged_from",
    ),
    "DecisionLogEntry": (
        "order_index",
        "window_id",
        "frequency_mhz",
        "kind",
        "provenance",
        "evidence",
    ),
    # Review attention (contract 14): a WindowReviewStatus's reasons, through
    # get_review_status and ``review show --json``. ``detail`` is a human
    # sentence whose text is not contract.
    "AttentionReason": ("kind", "detail", "severity", "locations", "evidence"),
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
    # Wave 8: types earlier waves added without declaring them.
    "CurationAction": (
        "action",
        "window_id",
        "freq_mhz",
        "peak_uid",
        "candidate_mhz",
        "frame",
        "epsilon",
    ),
    "SettingRow": (
        "path",
        "value",
        "source",
        "hard_default",
        "tier",
        "help",
        "type",
        "nullable",
        "units",
        "choices",
        "bounds",
    ),
    "ComplexFT": ("freq_array", "complex_spectrum", "metadata", "invalidated"),
    "ComplexFT.metadata": ("amplitude_scale", "units_label", "pad_factor"),
    # Events (Wave 5.1). Declared as their wire form. PipelineWarning's
    # code-specific fields live in ``details`` in Python and are flattened on
    # the wire: its entry is the common fields, and each code's own fields are
    # declared under ``PipelineWarning.<code>`` (from WARNING_FIELDS, below).
    "StageStarted": ("schema", "operation", "stage"),
    "StageFinished": ("schema", "operation", "stage", "elapsed_s", "summary"),
    "WindowProgress": (
        "schema",
        "operation",
        "stage",
        "phase",
        "round",
        "index",
        "total",
        "window_id",
        "n_peaks",
        "chi2r",
        "elapsed_s",
        "dropped",
    ),
    "ScanProgress": ("schema", "operation", "stage", "knob", "value", "index", "total"),
    "Invalidated": ("schema", "operation", "stage", "stages"),
    "PipelineWarning": ("schema", "operation", "stage", "code", "message"),
}
# A warning's further wire fields, per code: ``PipelineWarning.<code>``.
_FIELDS.update(
    {f"PipelineWarning.{code}": names for code, names in WARNING_FIELDS.items()}
)

#: The ``state`` values of a :func:`~ftmwpipeline.api.status` stage entry.
#: ``partial``: Stage 5 after a cancelled (or callback-failed) run kept the
#: windows that finished.
STAGE_STATES: Tuple[str, ...] = ("complete", "partial", "not_run")

#: The non-null ``restart_reason`` values of a ``fit run`` summary: why a fit
#: with a partial fit present started over instead of resuming it.
FIT_RESTART_REASONS: Tuple[str, ...] = (
    "restart_requested",
    "settings_changed",
    "incomplete_provenance",
    "thaw_refit",
)

#: Frozen closed vocabularies. ``decision_kind`` / ``decision_provenance`` are
#: checked against ``core.data_structures.DECISION_KINDS`` /
#: ``DECISION_PROVENANCES``, which the Stage 6 code records from.
_VOCABULARIES: Dict[str, Tuple[str, ...]] = {
    "decision_kind": ("add", "remove", "merge", "split", "accept", "create_window"),
    "decision_provenance": ("user",),
    "stage_state": STAGE_STATES,
    "warning_code": tuple(WARNING_FIELDS),
    "restart_reason": FIT_RESTART_REASONS,
    # Checked against ``core.data_structures.ATTENTION_KINDS``, which
    # ``review run`` records from.
    "attention_kind": ATTENTION_KINDS,
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
    fields=_FIELDS,
    vocabularies=_VOCABULARIES,
)


def capabilities() -> Dict[str, Any]:
    """What this installation's machine contract offers.

    Returns
    -------
    dict
        ``{"schema": "ftmw/capabilities@1", "contract_version": int,
        "schemas": [...], "accessors": [...], "codes": [...], "stages": [{"stage",
        "storage_key", "settings_prefix", "knob_prefix", "depends_on"}],
        "metadata_keys": [...], "tables": {name: [columns]},
        "fields": {type: [fields]}, "vocabularies": {name: [values]},
        "file_bound": {accessor: bool}, "pipeline_names": {accessor: name}}``,
        read from :data:`MANIFEST`. Already JSON-able and deterministic
        (manifest order); file-independent. ``pipeline_names`` names every
        accessor's ``Pipeline`` method (its own name unless declared otherwise).
    """
    return {
        "schema": CAPABILITIES_SCHEMA,
        "contract_version": MANIFEST.contract_version,
        "schemas": list(MANIFEST.schemas),
        "accessors": list(MANIFEST.accessors),
        "codes": list(MANIFEST.codes),
        "stages": [
            {
                "stage": stage.value,
                "storage_key": STAGE_KEYS[stage],
                "settings_prefix": STAGE_SETTINGS_PREFIX[stage],
                "knob_prefix": STAGE_KNOB_PREFIX[stage],
                "depends_on": [d.value for d in stage_depends_on(stage)],
            }
            for stage in Stage
        ],
        "metadata_keys": list(MANIFEST.metadata_keys),
        "tables": {k: list(v) for k, v in MANIFEST.tables.items()},
        "fields": {k: list(v) for k, v in MANIFEST.fields.items()},
        "vocabularies": {k: list(v) for k, v in MANIFEST.vocabularies.items()},
        "file_bound": dict(MANIFEST.file_bound),
        "pipeline_names": {
            a: MANIFEST.pipeline_names.get(a, a) for a in MANIFEST.accessors
        },
    }


__all__ = [
    "CONTRACT_VERSION",
    "FIT_RESTART_REASONS",
    "CAPABILITIES_SCHEMA",
    "FID_SAMPLES_SCHEMA",
    "DISPLAY_UNITS_SCHEMA",
    "FIT_THRESHOLDS_SCHEMA",
    "WINDOW_STATUS_SCHEMA",
    "SOURCE_PREVIEW_SCHEMA",
    "WINDOW_MODEL_SCHEMA",
    "SPECTRUM_MODEL_SCHEMA",
    "ANALYSIS_FINGERPRINT_SCHEMA",
    "STATUS_SCHEMA",
    "CALIBRATION_SCHEMA",
    "SNAP_TOLERANCE_SCHEMA",
    "METADATA_SCHEMA",
    "TABLES_SCHEMA",
    "TABLE_SCHEMA",
    "SETTINGS_DEFAULTS_SCHEMA",
    "SETTINGS_SCHEMA",
    "FINAL_PRODUCTS_SCHEMA",
    "REVIEW_LOG_SCHEMA",
    "PIPELINE_INFO_SCHEMA",
    "DISPLAY_FT_SCHEMA",
    "CURATION_ACTION_SCHEMA",
    "RUN_RESULT_SCHEMA",
    "STAGE_STARTED_SCHEMA",
    "STAGE_FINISHED_SCHEMA",
    "WINDOW_PROGRESS_SCHEMA",
    "SCAN_PROGRESS_SCHEMA",
    "INVALIDATED_SCHEMA",
    "WARNING_SCHEMA",
    "CurationAction",
    "WindowStatusRow",
    "FidPreviewRow",
    "ERROR_SCHEMA",
    "SCHEMA_NAME_RE",
    "Absent",
    "Stage",
    "STAGE_KEYS",
    "stage_for_key",
    "key_for_stage",
    "PROVENANCE_NAMES",
    "canonical_provenance_name",
    "STAGE_SETTINGS_PREFIX",
    "STAGE_KNOB_PREFIX",
    "stage_depends_on",
    "rerun_order",
    "STAGE_STATES",
    "AccessorSpec",
    "STATUS_PRESENT",
    "STATUS_NOT_RUN",
    "STATUS_UNDEFINED",
    "ContractManifest",
    "MANIFEST",
    "capabilities",
    # Events and cancellation
    "CancelToken",
    "Event",
    "EventCallback",
    "EVENT_TYPES",
    "StageStarted",
    "StageFinished",
    "WindowProgress",
    "ScanProgress",
    "Invalidated",
    "PipelineWarning",
    "WARNING_FIELDS",
    "WINDOW_PHASES",
    # Typed error family
    "PipelineFileError",
    "PipelineExistsError",
    "StageDependencyError",
    "PipelineCorruptionError",
    "PipelineCompatibilityError",
    "AnalysisEpochMismatchError",
    "NotFoundError",
    "NotFoundValueError",
    "PipelineFileNotFoundError",
    "IncompleteProvenanceError",
    "BadSettingError",
    "AlgorithmFailedError",
    "OperationCancelledError",
    "CallbackFailedError",
    "WriteConflictError",
    "CurationConflictError",
]
