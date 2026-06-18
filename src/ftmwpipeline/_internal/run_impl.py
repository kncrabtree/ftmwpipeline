"""End-to-end pipeline orchestration for the ``run`` command.

Drives a raw source through every stage in sequence (import -> FT -> noise ->
tau -> peaks -> windows -> fit -> timebase -> review, then optionally the
report) by calling the existing :class:`~ftmwpipeline.pipeline.Pipeline` stage
methods. It adds no analysis -- each stage's logic stays in its own
``_internal/stage*_impl``. The orchestrator owns the stage ordering, the
non-fatal timebase handling, the structured result, and the live per-stage
progress display (with a Stage-5 percentage bridged from the existing
``plan_execution`` per-window logs). See ``dev-docs/planning/pipeline-run.md``.
"""

from __future__ import annotations

import inspect
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    TextIO,
    Tuple,
    Union,
    cast,
)

# The Stage-5 per-window progress log (``plan_execution`` logs
# ``"window %d/%d w%d [...]: ..."`` at INFO). Matching the message *template*
# (not the rendered string) lets us read n/total straight from ``record.args``.
_WINDOW_LOG_PREFIX = "window %d/%d"


# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------


class _ProgressHandler(logging.Handler):
    """Bridge stage logs into the run's progress display.

    Renders the Stage-5 per-window records as a percentage on the active stage's
    line, surfaces WARNING+ records on their own line (above the bar), and drops
    everything else -- so the user sees stage progress and real Stage-5
    completion without the full INFO firehose.
    """

    def __init__(self, reporter: "_StageProgress") -> None:
        super().__init__(level=logging.INFO)
        self._reporter = reporter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if record.levelno >= logging.WARNING:
                self._reporter.message(self.format(record))
                return
            msg, args = record.msg, record.args
            if (
                isinstance(msg, str)
                and msg.startswith(_WINDOW_LOG_PREFIX)
                and isinstance(args, tuple)
                and len(args) >= 2
            ):
                self._reporter.substep(int(cast(int, args[0])), int(cast(int, args[1])))
        except Exception:  # logging must never crash the run
            pass


class _StageProgress:
    """Per-stage banner + sub-step bar writer (to stderr by default).

    ``enabled=False`` makes every method a no-op (for quiet/library callers).
    Rendering is TTY-aware: a ``\\r``-updated bar on a terminal, periodic plain
    lines otherwise (so it stays readable under ``conda run`` / pipes).
    """

    def __init__(
        self,
        total_stages: int,
        *,
        stream: Optional[TextIO] = None,
        enabled: bool = True,
    ) -> None:
        self._stream: TextIO = stream if stream is not None else sys.stderr
        self._enabled = enabled
        self._total = total_stages
        self._index = 0
        self._label = ""
        self._isatty = bool(getattr(self._stream, "isatty", lambda: False)())
        self._line_open = False
        self._last_decile = -1

    def _w(self, text: str) -> None:
        if self._enabled:
            self._stream.write(text)
            self._stream.flush()

    def _clear_line(self) -> None:
        if self._line_open and self._isatty:
            self._w("\r\033[K")
        self._line_open = False

    @contextmanager
    def capture_logs(self) -> Iterator[None]:
        """Install the bridge handler for the run; restore logging afterward."""
        if not self._enabled:
            yield
            return
        root = logging.getLogger()
        prev_level = root.level
        handler = _ProgressHandler(self)
        handler.setFormatter(logging.Formatter("  %(levelname)s: %(message)s"))
        root.addHandler(handler)
        # Let INFO records reach the handler (so per-window progress shows); the
        # handler itself drops non-progress INFO, so the log stays quiet.
        if prev_level == logging.NOTSET or prev_level > logging.INFO:
            root.setLevel(logging.INFO)
        try:
            yield
        finally:
            root.removeHandler(handler)
            root.setLevel(prev_level)

    @contextmanager
    def stage(self, label: str) -> Iterator[None]:
        """Run a stage: print its banner, time it, mark ✓ / ✗ on exit."""
        self._index += 1
        self._label = label
        self._last_decile = -1
        start = time.monotonic()
        self._w(f"[{self._index}/{self._total}] {label} …")
        self._line_open = True
        if not self._isatty:
            self._w("\n")
            self._line_open = False
        try:
            yield
        except Exception:
            self._clear_line()
            self._w(f"[{self._index}/{self._total}] {label} ✗ failed\n")
            raise
        else:
            self._clear_line()
            elapsed = time.monotonic() - start
            self._w(f"[{self._index}/{self._total}] {label} ✓ ({elapsed:.1f}s)\n")

    def substep(self, n: int, total: int) -> None:
        """Render a within-stage percentage from an (n, total) sub-step count."""
        if not self._enabled or total <= 0:
            return
        pct = max(0, min(100, int(100 * n / total)))
        if self._isatty:
            self._w(
                f"\r\033[K[{self._index}/{self._total}] {self._label} … "
                f"{pct:3d}% ({n}/{total})"
            )
            self._line_open = True
        elif pct // 10 != self._last_decile:
            self._w(f"    {self._label}: {pct}% ({n}/{total})\n")
        self._last_decile = pct // 10

    def message(self, text: str) -> None:
        """Surface a warning/error line above the current progress line."""
        self._clear_line()
        self._w(text + "\n")

    def note(self, text: str) -> None:
        """An orchestrator-level note (e.g. a skipped optional stage)."""
        self._clear_line()
        self._w(text + "\n")


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

    Runs import -> FT -> noise -> tau -> peaks -> windows -> fit -> timebase ->
    review (and, with ``report=True``, the report) in order, by calling the
    existing :class:`~ftmwpipeline.pipeline.Pipeline` stage methods. *trim* (the
    active-band FT range, MHz) is required -- there is no active-band
    auto-detector. Each ``*_params`` dict is forwarded to the matching stage; an
    optional *preset* name is forwarded to the stages that accept one.

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
    stages += ["FT", "noise", "calibrate tau", "peaks", "windows", "fit"]
    if calibrate:
        stages.append("timebase")
    stages.append("review")
    if report:
        stages.append("report")

    reporter = _StageProgress(len(stages), stream=progress_stream, enabled=progress)
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
                    pipe.detect_start_time()
                completed.append("start detection")

            with reporter.stage("FT"):
                ftp = dict(ft_params or {})
                ftp.setdefault("trim", trim)
                pipe.compute_ft(**ftp)
            completed.append("FT")

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
