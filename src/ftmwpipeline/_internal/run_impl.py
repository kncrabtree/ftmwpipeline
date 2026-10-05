"""End-to-end pipeline orchestration for the ``run`` command.

Drives a raw source through every stage in sequence (import -> start detection
-> FT -> timebase -> noise -> tau -> peaks -> windows -> fit -> review, then
optionally the report) by calling the existing :class:`~ftmwpipeline.pipeline.Pipeline` stage
methods. It adds no analysis -- each stage's logic stays in its own
``_internal/stage*_impl``. The orchestrator owns the stage ordering, the
non-fatal timebase handling, the structured result, and the live per-stage
progress display (with a Stage-5 percentage bridged from the existing
``plan_execution`` per-window logs).
"""

from __future__ import annotations

import inspect
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO, Tuple, Union

from ..contract import CancelToken, EventCallback, Stage
from ..file_manager import (
    BadSettingError,
    CallbackFailedError,
    OperationCancelledError,
    PipelineFileError,
)
from .events import OperationEvents, operation_events
from .progress import StageProgress

#: The run's progress labels as canonical stages. "start detection" (a stamp
#: on the data stage) and "report" (artifacts) are steps, not stages: they map
#: to ``None`` and contribute nothing to ``completed_stages``.
RUN_STAGES: Dict[str, Optional[Stage]] = {
    "import": Stage.DATA,
    "start detection": None,
    "FT": Stage.FT,
    "timebase": Stage.TIMEBASE,
    "noise": Stage.NOISE,
    "calibrate tau": Stage.TAU,
    "peaks": Stage.PEAKS,
    "windows": Stage.WINDOWS,
    "fit": Stage.FIT,
    "review": Stage.REVIEW,
    "report": None,
}


def canonical_run_step(label: str) -> Optional[str]:
    """A run step's canonical stage name, or ``None`` for a step that is not a
    stage (``"start detection"``, ``"report"``) or an unknown label."""
    stage = RUN_STAGES.get(label)
    return None if stage is None else stage.value


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _default_output(source: Union[str, Path]) -> Path:
    """Derive a ``<stem>.ftmw`` path (cwd) from a raw source name."""
    src = Path(source)
    stem = src.stem or src.name
    return Path.cwd() / f"{stem}.ftmw"


def _call(
    method: Callable[..., Any],
    params: Optional[Dict[str, Any]],
    preset: Optional[str],
    ops: Optional[OperationEvents] = None,
) -> Any:
    """Call a stage method with *params*, adding ``preset`` and ``events`` only
    if it accepts them.

    The per-stage override dict plus an optional preset name; the preset is
    forwarded solely to stages whose signature carries a ``preset`` parameter
    (noise / tau / peaks / windows / fit), so passing one is always safe. The
    run's :class:`OperationEvents` goes in as ``events`` to every stage that
    reports events, so the stage reports into the run (its cancel token rides
    along).
    """
    kwargs = dict(params or {})
    parameters = inspect.signature(method).parameters
    if preset is not None and "preset" in parameters:
        kwargs.setdefault("preset", preset)
    if ops is not None and "events" in parameters:
        kwargs["events"] = ops
    return method(**kwargs)


def _error_dict(exc: BaseException) -> Dict[str, Any]:
    """The ``ftmw/error@1`` dict of a failure that stopped the run.

    A typed error gives its own; anything else is carried under the base code
    ``pipeline_error`` with its message.
    """
    if isinstance(exc, PipelineFileError):
        return exc.to_dict()
    return PipelineFileError(str(exc) or type(exc).__name__).to_dict()


def run_pipeline_impl(
    source: Union[str, Path],
    output: Optional[Union[str, Path]] = None,
    *,
    trim: Optional[Tuple[float, float]] = None,
    sigma_floor_khz: Optional[float] = None,
    force: bool = True,
    format_name: Optional[str] = None,
    fid_index: Optional[int] = None,
    detect_start: bool = True,
    calibrate: bool = True,
    clocks: Optional[Any] = None,
    report: bool = False,
    report_output_dir: Optional[Union[str, Path]] = None,
    start_detection_params: Optional[Dict[str, Any]] = None,
    ft_params: Optional[Dict[str, Any]] = None,
    noise_params: Optional[Dict[str, Any]] = None,
    tau_params: Optional[Dict[str, Any]] = None,
    peak_params: Optional[Dict[str, Any]] = None,
    window_params: Optional[Dict[str, Any]] = None,
    fit_params: Optional[Dict[str, Any]] = None,
    timebase_params: Optional[Dict[str, Any]] = None,
    review_params: Optional[Dict[str, Any]] = None,
    report_params: Optional[Dict[str, Any]] = None,
    preset: Optional[str] = None,
    progress: bool = True,
    progress_stream: Optional[TextIO] = None,
    events: Optional[EventCallback] = None,
    cancel: Optional[CancelToken] = None,
) -> Dict[str, Any]:
    """Drive *source* through every pipeline stage and return a structured result.

    Runs import -> start detection -> FT -> timebase -> noise -> tau -> peaks ->
    windows -> fit -> review (and, with ``report=True``, the report) in order, by calling the
    existing :class:`~ftmwpipeline.pipeline.Pipeline` stage methods. *trim* (the
    active-band FT range, MHz) is required -- there is no active-band
    auto-detector. Each ``*_params`` dict is forwarded to the matching stage; an
    optional *preset* name is forwarded to the stages that accept one.
    *start_detection_params* is forwarded to ``detect_start_time`` (e.g.
    ``{"settings": StartDetectionSettings(guard_margin_us=1.0)}``) when
    *detect_start* is True.

    Timebase calibration is non-fatal: it auto-resolves the instrument clock
    declaration (explicit *clocks* > persisted ``spur.clocks`` > the loader's
    auto-extracted ``recommended_clock_sources``) and, when none is resolvable,
    is **skipped with a warning** rather than failing the run. ``calibrate=False``
    skips it deliberately.

    Returns a dict with ``pipeline_file``, ``status`` (``"success"`` /
    ``"error"``), ``completed_stages`` (the canonical stages actually written, in order, each
    once -- what a cancel's ``completed_stages`` holds; start detection and the
    report are not stages and add nothing), ``failed_stage`` (the canonical
    stage of the failing step; ``None`` on success and when the failing step is
    start detection or the report), ``failed_step`` (the progress label of the
    failing step -- ``"import"``, ``"start detection"``, ``"FT"``, ...,
    ``"report"`` -- else ``None``),
    ``error`` (that failure's ``ftmw/error@1`` dict, else ``None``),
    ``timebase`` (``"calibrated"`` / ``"skipped"`` / ``"not_requested"``),
    ``report`` (the ``report_run`` paths, or ``None``), and ``elapsed_s``.
    Stops at the first failing stage.

    ``events`` / ``cancel`` follow the long-operation contract: events are
    reported with ``operation="run"``; the token is checked before every step
    and inside the stages that check it. A cancel raises
    :class:`OperationCancelledError` (``completed_stages`` canonical) and a
    failing callback :class:`CallbackFailedError` -- neither is folded into
    the result. A skipped timebase calibration emits a ``timebase_skipped``
    warning.
    """
    from ..pipeline import Pipeline

    if trim is None:
        raise BadSettingError(
            "stage1.trim",
            "an active-band (min, max) FT range in MHz",
            None,
            message=(
                "run_pipeline requires `trim` (the active-band FT range in MHz); "
                "there is no active-band auto-detector."
            ),
        )
    out_path = Path(output) if output is not None else _default_output(source)

    # Stage banner plan, in call order (drives the [i/N] counter).
    stages = ["import"]
    if detect_start:
        stages.append("start detection")
    stages.append("FT")
    if calibrate:
        stages.append("timebase")
    stages += ["noise", "calibrate tau", "peaks", "windows", "fit"]
    stages.append("review")
    if report:
        stages.append("report")

    ops = operation_events("run", events, cancel)
    reporter = StageProgress(len(stages), stream=progress_stream, enabled=progress)
    # The result's completed_stages is the same canonical, ordered, de-duplicated
    # list a cancel reports (a step that is not a stage counts toward its owner).
    completed: List[str] = ops.completed_stages

    def _done(label: str, shape: Optional[str] = None) -> None:
        # A stage that reports through the shared events has already recorded
        # itself (and a tau twin it built); this only backstops the stage the
        # step is. A step that is not a stage records nothing.
        stage = RUN_STAGES[label]
        if label == "calibrate tau" and shape == "gaussian":
            stage = Stage.TAU_G  # tau_g is written; tau only if a twin was built
        if stage is not None:
            ops.mark_completed(stage)

    result: Dict[str, Any] = {
        "source": str(source),
        "pipeline_file": str(out_path),
        "status": "success",
        "completed_stages": completed,
        "failed_stage": None,
        "failed_step": None,
        "error": None,
        "timebase": "calibrated" if calibrate else "not_requested",
        "report": None,
        "elapsed_s": 0.0,
    }

    t0 = time.monotonic()
    # No transaction around the run: each stage call is its own atomic write
    # (§Crash safety), so a kill keeps every stage that finished before it.
    with reporter.capture_logs():
        try:
            ops.check_cancel()
            with reporter.stage("import"):
                pipe = _call(
                    Pipeline.create,
                    dict(
                        filepath=out_path,
                        source=source,
                        format_name=format_name,
                        fid_index=fid_index,
                        force=force,
                    ),
                    None,
                    ops,
                )
            _done("import")

            if detect_start:
                ops.check_cancel()
                with reporter.stage("start detection"):
                    _call(pipe.detect_start_time, start_detection_params, None, ops)
                _done("start detection")

            ops.check_cancel()
            with reporter.stage("FT"):
                ftp = dict(ft_params or {})
                ftp.setdefault("trim", trim)
                _call(pipe.compute_ft, ftp, None, ops)
            _done("FT")

            if calibrate:
                ops.check_cancel()
                with reporter.stage("timebase"):
                    try:
                        _call(
                            pipe.calibrate_timebase,
                            dict(clocks=clocks, **(timebase_params or {})),
                            None,
                            ops,
                        )
                        result["timebase"] = "calibrated"
                        _done("timebase")
                    except (OperationCancelledError, CallbackFailedError):
                        raise
                    except Exception as exc:  # non-fatal: warn + skip
                        result["timebase"] = "skipped"
                        note = (
                            f"timebase calibration skipped — {exc} "
                            "Frequencies are reported as precision-only "
                            "(uncalibrated state)."
                        )
                        reporter.note(f"  warning: {note}")
                        ops.warn("timebase_skipped", note, stage=Stage.TIMEBASE)

            ops.check_cancel()
            with reporter.stage("noise"):
                _call(pipe.estimate_noise, noise_params, preset, ops)
            _done("noise")

            ops.check_cancel()
            with reporter.stage("calibrate tau"):
                _call(pipe.calibrate_tau, tau_params, preset, ops)
            _done("calibrate tau", (tau_params or {}).get("shape"))

            ops.check_cancel()
            with reporter.stage("peaks"):
                _call(pipe.detect_peaks, peak_params, preset, ops)
            _done("peaks")

            ops.check_cancel()
            with reporter.stage("windows"):
                _call(pipe.assign_windows, window_params, preset, ops)
            _done("windows")

            ops.check_cancel()
            with reporter.stage("fit"):
                _call(pipe.fit_peaks, fit_params, preset, ops)
            _done("fit")

            ops.check_cancel()
            with reporter.stage("review"):
                rp = dict(review_params or {})
                if sigma_floor_khz is not None:
                    rp.setdefault("sigma_floor_khz", sigma_floor_khz)
                _call(pipe.review_run, rp, None, ops)
            _done("review")

            if report:
                ops.check_cancel()
                with reporter.stage("report"):
                    rdir = (
                        report_output_dir
                        if report_output_dir is not None
                        else out_path.parent / f"{out_path.stem}_report"
                    )
                    result["report"] = _call(
                        pipe.report_run,
                        dict(output_dir=rdir, **(report_params or {})),
                        None,
                        ops,
                    )
                _done("report")
        except (OperationCancelledError, CallbackFailedError):
            # Not folded into the result: a cancel (or a failing callback)
            # aborts the run. Every completed stage stays as written.
            raise
        except Exception as exc:
            result["status"] = "error"
            result["failed_step"] = reporter._label or None
            result["failed_stage"] = canonical_run_step(reporter._label)
            result["error"] = _error_dict(exc)
            result["completed_stages"] = list(completed)
            result["elapsed_s"] = time.monotonic() - t0
            return result

    result["completed_stages"] = list(completed)
    result["elapsed_s"] = time.monotonic() - t0
    return result
