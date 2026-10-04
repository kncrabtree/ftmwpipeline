"""Events, cancellation, and the log lines rendered from events -- the one place.

Normative spec: ``dev-docs/CONTRACT_STRATEGY.md`` §Events and cancellation.

Every long operation takes ``events`` (a callback) and ``cancel`` (a
:class:`~ftmwpipeline.contract.CancelToken`). Inside the pipeline both travel as
one :class:`OperationEvents`, built once at the top of the operation by
:func:`operation_events`:

- :meth:`OperationEvents.emit` renders the event's log line(s) (at today's
  logger name, level and text) and then delivers the event to the callback. A
  callback that raises becomes :class:`CallbackFailedError`, with the callback's
  exception as ``__cause__``.
- :meth:`OperationEvents.check_cancel` raises :class:`OperationCancelledError`
  once the token is set, carrying the interrupted stage and the stages this
  operation completed.
- :meth:`OperationEvents.stage` scopes one stage: it checks for a cancel (a
  check *between* stages), emits ``StageStarted``, and yields a
  :class:`StageScope`; the stage calls :meth:`StageScope.finish` after its
  results are written, which emits ``StageFinished``. A stage that raises emits
  no ``StageFinished``.

With no callback and no token the same object still renders every log line, so
call sites never branch on whether a caller is listening, and a call site that
used to log a line directly now emits the event instead (never both).

A composite operation (``run_pipeline``) passes its own :class:`OperationEvents`
as the ``events`` argument of the stage calls it makes;
:func:`operation_events` then hands that same object back, so the nested stage
reports under the composite's ``operation`` and records its completion there.

The renderer (:func:`render`) and its line tables are the single source of the
log lines the spec lists (stage start/end, the ``window %d/%d`` progress line,
the per-window detail / dropped / slow lines, the invalidation warning, the
walk-fallback warning). Add a verb's start/end line to
:data:`STAGE_START_LINES` / :data:`STAGE_END_LINES`, a warning code's line to
:data:`WARNING_LINES`.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from ..contract import (
    WARNING_FIELDS,
    Absent,
    CancelToken,
    Event,
    EventCallback,
    Invalidated,
    PipelineWarning,
    Stage,
    StageFinished,
    StageStarted,
    WindowProgress,
    key_for_stage,
    rerun_order,
)
from ..file_manager import (
    BadSettingError,
    CallbackFailedError,
    OperationCancelledError,
)

#: The parent's cancel poll interval while pool workers fit (seconds). Bounds the
#: cancel latency of the parallel Stage 5 walk.
CANCEL_POLL_S = 0.2

StageLike = Union[Stage, str]

# ---------------------------------------------------------------------------
# Log-line tables (the single renderer's data)
# ---------------------------------------------------------------------------

_PLAN_EXECUTION_LOGGER = "ftmwpipeline.fitting.plan_execution"
_STAGE5_LOGGER = "ftmwpipeline._internal.stage5_impl"
_FILE_MANAGER_LOGGER = "ftmwpipeline.file_manager"
_STAGE6_LOGGER = "ftmwpipeline._internal.stage6_impl"
_STAGE2_LOGGER = "ftmwpipeline._internal.stage2_impl"
_STAGE3_LOGGER = "ftmwpipeline._internal.stage3_impl"
_STAGE4_LOGGER = "ftmwpipeline._internal.stage4_impl"

#: The sub-step progress line (``_internal.progress.WINDOW_LOG_PREFIX``): the
#: run display reads ``n`` / ``total`` from ``record.args``.
WINDOW_PROGRESS_LOG_TEMPLATE = "window %d/%d"
#: The per-window detail record, one per fitted window, always at INFO.
WINDOW_DETAIL_LOG_TEMPLATE = "w%d [%.1f-%.1f MHz]: %d peaks, chi2r=%.3g, %.1fs"
#: A window the cleanup cascaded to empty.
WINDOW_DROPPED_LOG_TEMPLATE = "w%d [%.1f-%.1f MHz]: dropped (cascaded to empty)"
#: The slow-window record, emitted IN ADDITION to the detail line -- at WARNING,
#: under its own template -- when a window exceeds the threshold. Level and
#: template identify it together: a consumer filtering by level and one matching
#: on ``record.msg`` agree on which records mean "slow".
SLOW_WINDOW_LOG_TEMPLATE = "slow window w%d [%.1f-%.1f MHz]: %.1fs (over %.0fs)"
WALK_FALLBACK_LOG_TEMPLATE = (
    "dag walk falling back to the sequential walk (%s); re-fitting %d "
    "windows for cross-window correctness"
)
INVALIDATION_LOG_TEMPLATE = "%s; invalidated stage(s) %s -- re-run them to refresh."


@dataclass(frozen=True)
class LogLine:
    """One rendered log record: where it goes and how its args are built.

    ``args`` receives the event and the render-only ``detail`` mapping (values a
    line prints that the event does not carry, such as a window's frequency
    range) and returns the ``%``-args for ``template``.
    """

    logger: str
    level: int
    template: str
    args: Callable[[Any, Mapping[str, Any]], Tuple[Any, ...]]

    def emit(self, event: Any, detail: Mapping[str, Any]) -> None:
        logging.getLogger(self.logger).log(
            self.level, self.template, *self.args(event, detail)
        )

    def text(self, event: Any, detail: Mapping[str, Any]) -> str:
        return self.template % self.args(event, detail)


def _fit_end_args(event: Any, detail: Mapping[str, Any]) -> Tuple[Any, ...]:
    s = event.summary
    return (
        s["n_windows"],
        s["n_fitted_peaks"],
        s["n_thaw_accepted"],
        s["n_thaw_events"],
        s["n_rescue_accepted"],
        s["n_rescue_events"],
        s["n_rescue_added"],
        s["n_rescue_origin_pruned"],
        s["n_replan_accepted"],
        s["final_plan_revision"],
    )


#: The line a stage logs as it starts, by the stage's own verb. A verb without
#: an entry logs no start line.
STAGE_START_LINES: Dict[str, LogLine] = {
    "review run": LogLine(
        _STAGE6_LOGGER,
        logging.INFO,
        "Stage 6 review: routing attention for %d windows in %s",
        lambda e, d: (d["n_windows"], d["path"]),
    ),
}

#: The line a stage logs once its results are written, by the stage's own verb.
#: Rendered from ``StageFinished.summary``.
STAGE_END_LINES: Dict[str, LogLine] = {
    "noise run": LogLine(
        _STAGE2_LOGGER,
        logging.INFO,
        "Stage 2: Noise estimation results saved and marked complete",
        lambda e, d: (),
    ),
    "peaks run": LogLine(
        _STAGE3_LOGGER,
        logging.INFO,
        "Stage 3: detected %d peaks (active grid); %d promoted at SNR>=%.3g",
        lambda e, d: (
            e.summary["n_peaks"],
            e.summary["n_promoted"],
            e.summary["promotion_min_snr"],
        ),
    ),
    "review run": LogLine(
        _STAGE6_LOGGER,
        logging.INFO,
        "Saved Stage 6 review to %s: %d windows, %d need attention",
        lambda e, d: (d["path"], e.summary["n_windows"], e.summary["n_attention"]),
    ),
    "windows run": LogLine(
        _STAGE4_LOGGER,
        logging.INFO,
        "Stage 4: %d windows, %d batches, %d free peaks, "
        "%d fixed contributors, %d dependencies",
        lambda e, d: (
            e.summary["n_windows"],
            e.summary["n_batches"],
            e.summary["n_free_peaks"],
            e.summary["n_fixed_contributors"],
            e.summary["n_dependencies"],
        ),
    ),
    "fit run": LogLine(
        _STAGE5_LOGGER,
        logging.INFO,
        "Stage 5: %d windows, %d fitted peaks; thaw %d/%d accepted, "
        "rescue %d/%d rounds accepted (added %d peaks, %d rescue-origin pruned), "
        "%d structural replans accepted (revision %d)",
        _fit_end_args,
    ),
}


def _float(value: Union[float, Absent]) -> float:
    """A value for a ``%g`` slot: an absent one prints as ``nan``, as before."""
    return float("nan") if isinstance(value, Absent) else float(value)


def _freq(detail: Mapping[str, Any]) -> Tuple[float, float]:
    lo, hi = detail["freq_range"]
    return float(lo), float(hi)


#: The line each warning code logs. A code without an entry logs nothing (its
#: caller reports it some other way, e.g. ``run``'s progress note).
WARNING_LINES: Dict[str, LogLine] = {
    "slow_window": LogLine(
        _PLAN_EXECUTION_LOGGER,
        logging.WARNING,
        SLOW_WINDOW_LOG_TEMPLATE,
        lambda e, d: (
            e.details["window_id"],
            *_freq(d),
            e.details["elapsed_s"],
            e.details["threshold_s"],
        ),
    ),
    "walk_fallback": LogLine(
        _PLAN_EXECUTION_LOGGER,
        logging.WARNING,
        WALK_FALLBACK_LOG_TEMPLATE,
        lambda e, d: (e.details["reason"], e.details["n_windows"]),
    ),
    "epoch_acknowledged": LogLine(
        _STAGE6_LOGGER,
        logging.WARNING,
        "Editing a Stage 5 fit produced under analysis epoch %s with "
        "epoch %s; proceeding on the acknowledgement recorded in the "
        "file. The curated fit mixes two analysis environments.",
        lambda e, d: (e.details["file_epoch"], e.details["current_epoch"]),
    ),
}

_WINDOW_DETAIL_LINE = LogLine(
    _PLAN_EXECUTION_LOGGER,
    logging.INFO,
    WINDOW_DETAIL_LOG_TEMPLATE,
    lambda e, d: (e.window_id, *_freq(d), e.n_peaks, _float(e.chi2r), e.elapsed_s),
)
_WINDOW_DROPPED_LINE = LogLine(
    _PLAN_EXECUTION_LOGGER,
    logging.INFO,
    WINDOW_DROPPED_LOG_TEMPLATE,
    lambda e, d: (e.window_id, *_freq(d)),
)
_WINDOW_PROGRESS_LINE = LogLine(
    _PLAN_EXECUTION_LOGGER,
    logging.INFO,
    WINDOW_PROGRESS_LOG_TEMPLATE,
    lambda e, d: (e.index, e.total),
)
_INVALIDATION_LINE = LogLine(
    _FILE_MANAGER_LOGGER,
    logging.WARNING,
    INVALIDATION_LOG_TEMPLATE,
    lambda e, d: (
        d["reason"],
        sorted(key_for_stage(s) for s in e.stages),
    ),
)

#: The per-window lines of each stage that emits ``WindowProgress``. Stage 6
#: (``review``) has none yet.
WINDOW_LINES: Dict[Stage, Tuple[LogLine, LogLine, LogLine]] = {
    # (detail, dropped, progress)
    Stage.FIT: (_WINDOW_DETAIL_LINE, _WINDOW_DROPPED_LINE, _WINDOW_PROGRESS_LINE),
}


def render(
    event: Event,
    *,
    verb: Optional[str] = None,
    detail: Optional[Mapping[str, Any]] = None,
) -> None:
    """Log the line(s) *event* stands for. The one renderer.

    ``verb`` is the emitting stage's own CLI verb (it selects the stage start /
    end line); ``detail`` carries render-only values the line prints but the
    event does not (``freq_range`` for a window, ``reason`` for an
    invalidation). An event with no line renders nothing.
    """
    d: Mapping[str, Any] = detail or {}
    if isinstance(event, StageStarted):
        line = STAGE_START_LINES.get(verb or "")
        if line is not None:
            line.emit(event, d)
    elif isinstance(event, StageFinished):
        line = STAGE_END_LINES.get(verb or "")
        if line is not None:
            line.emit(event, d)
    elif isinstance(event, WindowProgress):
        lines = WINDOW_LINES.get(event.stage) if event.stage is not None else None
        if lines is not None:
            detail_line, dropped_line, progress_line = lines
            (dropped_line if event.dropped else detail_line).emit(event, d)
            progress_line.emit(event, d)
    elif isinstance(event, Invalidated):
        _INVALIDATION_LINE.emit(event, d)
    elif isinstance(event, PipelineWarning):
        line = WARNING_LINES.get(event.code)
        if line is not None:
            line.emit(event, d)
    # ScanProgress: no line.


def warning_text(
    code: str, details: Mapping[str, Any], detail: Mapping[str, Any]
) -> str:
    """The rendered text of *code*'s log line (its ``PipelineWarning.message``)."""
    line = WARNING_LINES[code]
    probe = PipelineWarning(
        operation="", stage=None, code=code, message="", details=dict(details)
    )
    return line.text(probe, detail)


# ---------------------------------------------------------------------------
# Per-operation state
# ---------------------------------------------------------------------------


class OperationEvents:
    """The events callback, cancel token and progress of one long operation.

    Build it with :func:`operation_events`. It is itself an events callback
    (``ops(event)`` is ``ops.emit(event)``), so a composite operation passes it
    as the ``events`` argument of the calls it makes and they report into it.

    Attributes
    ----------
    operation : str
        The CLI verb of the operation (``"fit run"``, ``"run"``); stamped on
        every event.
    completed_stages : list of str
        Canonical stages this operation finished and wrote, in order.
    completed_windows : list of int
        Always empty until Stage 5 partial persistence (Wave 5.2).
    current_stage : Stage or None
        The stage running now (``None`` between stages).
    """

    def __init__(
        self,
        operation: str,
        callback: Optional[EventCallback] = None,
        cancel: Optional[CancelToken] = None,
    ) -> None:
        self.operation = operation
        self.callback = callback
        self.cancel = cancel
        self.completed_stages: List[str] = []
        self.completed_windows: List[int] = []
        self.current_stage: Optional[Stage] = None
        self._environment_checked = False
        # committing(): depth, and a callback failure held until it ends.
        self._committing = 0
        self._held_failure: Optional[CallbackFailedError] = None

    def __call__(self, event: Event) -> None:
        self.emit(event)

    # -- delivery ----------------------------------------------------------

    def emit(
        self,
        event: Event,
        *,
        verb: Optional[str] = None,
        detail: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Render *event*'s log line(s), then deliver it to the callback.

        Raises
        ------
        CallbackFailedError
            When the callback raises (chained as ``__cause__``).
        """
        render(event, verb=verb, detail=detail)
        if self.callback is None or self._held_failure is not None:
            return
        try:
            self.callback(event)
        except Exception as exc:
            failure = CallbackFailedError(event.schema)
            failure.__cause__ = exc
            if self._committing:
                # Inside a write that must complete: hold the failure (and
                # stop delivering) until the write is done.
                self._held_failure = failure
                return
            raise failure from exc

    @contextmanager
    def committing(self) -> Iterator[None]:
        """A block that, once begun, completes: the final write of a stage, or
        a restore-then-replay that must not stop half way.

        Inside it a cancel is not honoured (check points pass and
        :meth:`cancel_requested` is ``False``; the next check point after the
        block honours it), and a callback that raises is not allowed to abort
        the block: its :class:`CallbackFailedError` is raised when the block
        ends, and nothing more is delivered meanwhile. Log lines still render.
        """
        self._committing += 1
        try:
            yield
        finally:
            self._committing -= 1
        if not self._committing and self._held_failure is not None:
            failure, self._held_failure = self._held_failure, None
            raise failure

    # -- cancellation ------------------------------------------------------

    def cancel_requested(self) -> bool:
        """True once the cancel token is set (``False`` inside
        :meth:`committing`)."""
        if self._committing:
            return False
        return self.cancel is not None and bool(self.cancel.is_set())

    def check_cancel(self) -> None:
        """A cancel check point.

        Raises
        ------
        OperationCancelledError
            When the token is set; ``stage`` is :attr:`current_stage`.
        """
        if self.cancel_requested():
            raise self.cancelled_error()

    def cancelled_error(self) -> OperationCancelledError:
        """The ``cancelled`` error for a cancel honoured now."""
        return OperationCancelledError(
            None if self.current_stage is None else self.current_stage.value,
            list(self.completed_stages),
            list(self.completed_windows),
        )

    # -- stages ------------------------------------------------------------

    def mark_completed(self, stage: StageLike) -> None:
        """Record *stage* as finished and written (idempotent)."""
        name = Stage(stage).value
        if name not in self.completed_stages:
            self.completed_stages.append(name)

    @contextmanager
    def stage(
        self,
        stage: Optional[StageLike],
        *,
        verb: Optional[str] = None,
        detail: Optional[Mapping[str, Any]] = None,
        file_path: Optional[Union[str, "os.PathLike[str]"]] = None,
    ) -> Iterator["StageScope"]:
        """Run one stage: cancel check, ``StageStarted``, then the body.

        The body calls :meth:`StageScope.finish` once its results are written.
        The check before the stage reports ``stage=None`` (between stages);
        checks inside report this stage. ``stage=None`` scopes a step that is
        not a canonical stage (``start run``, ``report run``): its events carry
        ``stage: null`` and finishing it records no completed stage.
        ``detail`` carries render-only values for the verb's start line.
        ``file_path`` names the file the stage works on: the operation's one
        ``environment_drift`` check (:meth:`check_environment`) runs against it
        right after ``StageStarted``.
        """
        self.check_cancel()
        st = None if stage is None else Stage(stage)
        scope = StageScope(self, st, verb)
        previous = self.current_stage
        self.current_stage = scope.stage
        try:
            self.emit(
                StageStarted(self.operation, scope.stage), verb=verb, detail=detail
            )
            if file_path is not None:
                self.check_environment(file_path, stage=st)
            yield scope
        finally:
            self.current_stage = previous

    def detached_scope(
        self, stage: Optional[StageLike], *, verb: Optional[str] = None
    ) -> "StageScope":
        """A :class:`StageScope` for *stage* that emits no ``StageStarted``.

        For code that reports inside a stage some other function began (or a
        direct call into a stage's internals, such as a test driving the fit
        walk): window, warning and invalidation events still flow.
        """
        return StageScope(self, None if stage is None else Stage(stage), verb)

    # -- environment -------------------------------------------------------

    def check_environment(
        self,
        file_path: Union[str, "os.PathLike[str]"],
        *,
        stage: Optional[StageLike] = None,
    ) -> None:
        """Emit ``environment_drift`` once per operation when the file's recorded
        analysis environment differs from the running one.

        The comparison is the one ``info`` reports as
        ``runtime_environment_drift``
        (:func:`~ftmwpipeline.core.environment.describe_runtime_drift`);
        ``fields`` are the differing ``EnvironmentRecord`` field names. The
        warning has no log line (``info`` and ``validate`` already report the
        drift), so with no callback nothing is read. Only the first call of an
        operation checks; later ones, and a file that cannot be read, do
        nothing.
        """
        if self._environment_checked:
            return
        self._environment_checked = True
        if self.callback is None:
            return
        try:
            import h5py

            from ..core.environment import capture_environment, describe_runtime_drift
            from ..io.environment_serialization import load_stage_environments

            with h5py.File(os.fspath(file_path), "r") as h5f:
                envs = load_stage_environments(h5f)
            lines = describe_runtime_drift(envs, capture_environment())
        except (OSError, KeyError, ValueError):
            return
        if not lines:
            return
        self.warn(
            "environment_drift",
            "The file's recorded analysis environment differs from the running "
            "one: " + "; ".join(lines),
            stage=stage,
            fields=[line.split(":", 1)[0] for line in lines],
        )

    def warn(
        self,
        code: str,
        message: Optional[str] = None,
        *,
        stage: Optional[StageLike] = None,
        detail: Optional[Mapping[str, Any]] = None,
        **fields: Any,
    ) -> None:
        """Emit a :class:`PipelineWarning` (``stage`` defaults to the current one).

        ``message`` defaults to the code's rendered log text.
        """
        st = Stage(stage) if stage is not None else self.current_stage
        d = detail or {}
        if message is None:
            message = warning_text(code, fields, d)
        self.emit(
            PipelineWarning(self.operation, st, code, message, dict(fields)),
            detail=d,
        )


class StageScope:
    """The reporting handle of one running stage (from :meth:`OperationEvents.stage`)."""

    def __init__(
        self, ops: OperationEvents, stage: Optional[Stage], verb: Optional[str]
    ) -> None:
        self.ops = ops
        self.stage = stage
        self.verb = verb
        self._t0 = time.monotonic()
        self.finished = False
        # (stages, reasons) while collect_invalidations() is active.
        self._collected: Optional[Tuple[Set[Stage], List[str]]] = None

    @property
    def operation(self) -> str:
        return self.ops.operation

    def emit(self, event: Event, *, detail: Optional[Mapping[str, Any]] = None) -> None:
        self.ops.emit(event, verb=self.verb, detail=detail)

    def check_cancel(self) -> None:
        """A cancel check point inside this stage (``stage`` is this stage)."""
        if self.ops.cancel_requested():
            previous = self.ops.current_stage
            self.ops.current_stage = self.stage
            try:
                raise self.ops.cancelled_error()
            finally:
                self.ops.current_stage = previous

    def cancel_requested(self) -> bool:
        return self.ops.cancel_requested()

    def committing(self) -> "ContextManager[None]":
        """:meth:`OperationEvents.committing` of this stage's operation."""
        return self.ops.committing()

    def warn(
        self,
        code: str,
        message: Optional[str] = None,
        *,
        detail: Optional[Mapping[str, Any]] = None,
        **fields: Any,
    ) -> None:
        """Emit a :class:`PipelineWarning` for this stage."""
        self.ops.warn(code, message, stage=self.stage, detail=detail, **fields)

    def window_progress(
        self,
        *,
        phase: str,
        index: int,
        total: int,
        window_id: int,
        n_peaks: Union[int, Absent],
        chi2r: Union[float, Absent],
        elapsed_s: float,
        dropped: bool,
        freq_range: Tuple[float, float],
    ) -> None:
        """Emit one :class:`WindowProgress` (renders the detail/dropped and
        ``window n/total`` lines)."""
        self.emit(
            WindowProgress(
                self.operation,
                self.stage,
                phase,
                int(index),
                int(total),
                int(window_id),
                n_peaks,
                chi2r,
                float(elapsed_s),
                bool(dropped),
            ),
            detail={"freq_range": freq_range},
        )

    def invalidated(self, stages: Sequence[StageLike], *, reason: str) -> None:
        """Emit :class:`Invalidated` for *stages* (nothing when empty).

        ``reason`` is the log line's lead (why the stages were dropped).
        """
        if not stages:
            return
        if self._collected is not None:
            self._collected[0].update(Stage(s) for s in stages)
            if reason not in self._collected[1]:
                self._collected[1].append(reason)
            return
        names = {Stage(s) for s in stages}
        ordered = tuple(s for s in rerun_order() if s in names)
        self.emit(
            Invalidated(self.operation, self.stage, ordered),
            detail={"reason": reason},
        )

    @contextmanager
    def collect_invalidations(self) -> Iterator[None]:
        """Combine every invalidation inside into ONE :class:`Invalidated`.

        For a call that invalidates in several steps (an import that both
        overwrites an analysed file and moves its start hint): the event is
        emitted once, when the block ends, naming the union of the stages (the
        result's ``invalidated``); its log line joins the steps' reasons. If the
        block raises, what was already dropped is still logged but nothing is
        delivered. Nested use joins the outer collection.
        """
        if self._collected is not None:
            yield
            return
        self._collected = (set(), [])
        try:
            yield
        except BaseException:
            stages, reasons = self._collected
            self._collected = None
            if stages:
                ordered = tuple(s for s in rerun_order() if s in stages)
                render(
                    Invalidated(self.operation, self.stage, ordered),
                    detail={"reason": "; ".join(reasons)},
                )
            raise
        stages, reasons = self._collected
        self._collected = None
        self.invalidated(list(stages), reason="; ".join(reasons))

    def finish(
        self,
        summary: Mapping[str, Any],
        *,
        detail: Optional[Mapping[str, Any]] = None,
        wrote: bool = True,
    ) -> None:
        """The stage's results are written: record it and emit ``StageFinished``.

        Call once, after the final write (and after any ``Invalidated``).
        ``detail`` carries render-only values for the verb's end line.
        ``wrote=False`` (a dry run, a preview) finishes without recording the
        stage in ``completed_stages`` -- it wrote nothing.
        """
        if self.finished:
            name = None if self.stage is None else self.stage.value
            raise RuntimeError(f"stage {name!r} already finished")
        self.finished = True
        if wrote and self.stage is not None:
            self.ops.mark_completed(self.stage)
        self.emit(
            StageFinished(
                self.operation,
                self.stage,
                time.monotonic() - self._t0,
                dict(summary),
            ),
            detail=detail,
        )


def operation_events(
    operation: str,
    events: Optional[EventCallback] = None,
    cancel: Optional[CancelToken] = None,
) -> OperationEvents:
    """The :class:`OperationEvents` of a long operation's public arguments.

    An :class:`OperationEvents` passed as *events* (a composite operation
    calling a stage) is returned as is, and then *cancel* must be ``None``: the
    composite's token governs.

    Raises
    ------
    BadSettingError
        If *events* is not callable, *cancel* has no ``is_set()``, or both a
        composite's events and a separate token are given.
    """
    if isinstance(events, OperationEvents):
        if cancel is not None and cancel is not events.cancel:
            raise BadSettingError(
                "cancel",
                "None when events is an enclosing operation's events",
                cancel,
            )
        return events
    if events is not None and not callable(events):
        raise BadSettingError("events", "a callable taking one event", events)
    if cancel is not None and not isinstance(cancel, CancelToken):
        raise BadSettingError("cancel", "an object with is_set() -> bool", cancel)
    return OperationEvents(operation, events, cancel)


def detached_scope(
    stage: Optional[StageLike], *, verb: Optional[str] = None
) -> StageScope:
    """A scope with no callback and no token: renders log lines only.

    The default for an internal function reachable without an operation (a
    direct call of the Stage 5 walk, ``invalidate_stages_in_file`` from a path
    not yet threaded).
    """
    return OperationEvents("").detached_scope(stage, verb=verb)


def emit_invalidated(
    scope: Optional[StageScope],
    keys: Sequence[str],
    *,
    reason: str,
) -> None:
    """Report that the storage *keys* were invalidated.

    Through *scope* when given; otherwise the warning line is rendered with no
    delivery. Nothing happens for an empty *keys*.
    """
    if not keys:
        return
    from ..contract import stage_for_key

    stages = [stage_for_key(k) for k in keys]
    if scope is None:
        names = set(stages)
        ordered = tuple(s for s in rerun_order() if s in names)
        render(Invalidated("", None, ordered), detail={"reason": reason})
        return
    scope.invalidated(stages, reason=reason)


__all__ = [
    "CANCEL_POLL_S",
    "LogLine",
    "OperationEvents",
    "StageScope",
    "STAGE_START_LINES",
    "STAGE_END_LINES",
    "WARNING_LINES",
    "WINDOW_LINES",
    "WARNING_FIELDS",
    "detached_scope",
    "emit_invalidated",
    "operation_events",
    "render",
    "warning_text",
]
