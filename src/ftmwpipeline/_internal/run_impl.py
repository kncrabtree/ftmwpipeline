"""End-to-end pipeline orchestration for the ``run`` command.

Drives a raw source through every stage in sequence (import -> FT -> timebase
-> noise -> tau -> peaks -> windows -> fit -> review, then optionally the
report) by calling the existing :class:`~ftmwpipeline.pipeline.Pipeline` stage
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

from .progress import StageProgress

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _default_output(source: Union[str, Path]) -> Path:
    """Derive a ``<stem>.ftmw`` path (cwd) from a raw source name."""
    src = Path(source)
    stem = src.stem or src.name
    return Path.cwd() / f"{stem}.ftmw"


def _call(
    method: Callable[..., Any], params: Optional[Dict[str, Any]], preset: Optional[str]
) -> Any:
    """Call a stage method with *params*, adding ``preset`` only if it accepts it.

    The per-stage override dict plus an optional preset name; the preset is
    forwarded solely to stages whose signature carries a ``preset`` parameter
    (noise / tau / peaks / windows / fit), so passing one is always safe.
    """
    kwargs = dict(params or {})
    if preset is not None and "preset" in inspect.signature(method).parameters:
        kwargs.setdefault("preset", preset)
    return method(**kwargs)


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
) -> Dict[str, Any]:
    """Drive *source* through every pipeline stage and return a structured result.

    Runs import -> FT -> timebase -> noise -> tau -> peaks -> windows -> fit ->
    review (and, with ``report=True``, the report) in order, by calling the
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
    ``"error"``), ``completed_stages``, ``failed_stage``, ``error``, ``timebase``
    (``"calibrated"`` / ``"skipped"`` / ``"not_requested"``), ``report`` (the
    ``report_run`` paths, or ``None``), and ``elapsed_s``. Stops at the first
    failing stage.
    """
    from ..pipeline import Pipeline

    if trim is None:
        raise ValueError(
            "run_pipeline requires `trim` (the active-band FT range in MHz); "
            "there is no active-band auto-detector."
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

    reporter = StageProgress(len(stages), stream=progress_stream, enabled=progress)
    completed: List[str] = []
    result: Dict[str, Any] = {
        "source": str(source),
        "pipeline_file": str(out_path),
        "status": "success",
        "completed_stages": completed,
        "failed_stage": None,
        "error": None,
        "timebase": "calibrated" if calibrate else "not_requested",
        "report": None,
        "elapsed_s": 0.0,
    }

    t0 = time.monotonic()
    with reporter.capture_logs():
        try:
            with reporter.stage("import"):
                pipe = Pipeline.create(
                    out_path,
                    source=source,
                    format_name=format_name,
                    fid_index=fid_index,
                    force=force,
                )
            completed.append("import")

            if detect_start:
                with reporter.stage("start detection"):
                    pipe.detect_start_time(**(start_detection_params or {}))
                completed.append("start detection")

            with reporter.stage("FT"):
                ftp = dict(ft_params or {})
                ftp.setdefault("trim", trim)
                pipe.compute_ft(**ftp)
            completed.append("FT")

            if calibrate:
                with reporter.stage("timebase"):
                    try:
                        pipe.calibrate_timebase(
                            clocks=clocks, **(timebase_params or {})
                        )
                        result["timebase"] = "calibrated"
                        completed.append("timebase")
                    except Exception as exc:  # non-fatal: warn + skip
                        result["timebase"] = "skipped"
                        reporter.note(
                            f"  warning: timebase calibration skipped — {exc} "
                            "Frequencies are reported as precision-only "
                            "(uncalibrated state)."
                        )

            with reporter.stage("noise"):
                _call(pipe.estimate_noise, noise_params, preset)
            completed.append("noise")

            with reporter.stage("calibrate tau"):
                _call(pipe.calibrate_tau, tau_params, preset)
            completed.append("calibrate tau")

            with reporter.stage("peaks"):
                _call(pipe.detect_peaks, peak_params, preset)
            completed.append("peaks")

            with reporter.stage("windows"):
                _call(pipe.assign_windows, window_params, preset)
            completed.append("windows")

            with reporter.stage("fit"):
                _call(pipe.fit_peaks, fit_params, preset)
            completed.append("fit")

            with reporter.stage("review"):
                rp = dict(review_params or {})
                if sigma_floor_khz is not None:
                    rp.setdefault("sigma_floor_khz", sigma_floor_khz)
                pipe.review_run(**rp)
            completed.append("review")

            if report:
                with reporter.stage("report"):
                    rdir = (
                        report_output_dir
                        if report_output_dir is not None
                        else out_path.parent / f"{out_path.stem}_report"
                    )
                    result["report"] = pipe.report_run(
                        output_dir=rdir, **(report_params or {})
                    )
                completed.append("report")
        except Exception as exc:
            result["status"] = "error"
            result["failed_stage"] = reporter._label
            result["error"] = str(exc)
            result["elapsed_s"] = time.monotonic() - t0
            return result

    result["elapsed_s"] = time.monotonic() - t0
    return result
