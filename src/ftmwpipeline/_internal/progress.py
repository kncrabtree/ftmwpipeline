"""Shared per-stage progress display for long-running CLI operations.

A small, dependency-free reporter used by the end-to-end ``run`` orchestration
and the report renderer. It prints a per-stage banner and bridges a sub-step
count -- emitted by the worker as a ``"window %d/%d ..."`` INFO log -- into a
percentage on the active stage's line, so a caller gets live progress without
threading a callback through every layer. Surfaces WARNING+ records and drops
the rest, so the stage logs stay quiet. TTY-aware: a ``\\r``-updated bar on a
terminal, periodic plain lines otherwise (readable under ``conda run`` / pipes).
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Iterator, Optional, TextIO, cast

# The sub-step progress log template workers emit (e.g. the Stage-5
# ``plan_execution`` per-window log, and the report's per-window render /
# per-figure embed logs). Matching the message *template* (not the rendered
# string) lets the handler read n/total straight from ``record.args``.
WINDOW_LOG_PREFIX = "window %d/%d"


class ProgressHandler(logging.Handler):
    """Bridge worker logs into a :class:`StageProgress` display.

    Renders the ``"window %d/%d ..."`` sub-step records as a percentage on the
    active stage's line, surfaces WARNING+ records on their own line (above the
    bar), and drops everything else -- so the user sees stage progress without
    the full INFO firehose.
    """

    def __init__(self, reporter: "StageProgress") -> None:
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
                and msg.startswith(WINDOW_LOG_PREFIX)
                and isinstance(args, tuple)
                and len(args) >= 2
            ):
                self._reporter.substep(int(cast(int, args[0])), int(cast(int, args[1])))
        except Exception:  # logging must never crash the operation
            pass


class StageProgress:
    """Per-stage banner + sub-step bar writer (to stderr by default).

    ``enabled=False`` makes every method a no-op (for quiet/library callers).
    With a single stage the ``[i/N]`` counter is dropped (a one-shot operation
    reads better without it).
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

    def _prefix(self) -> str:
        return "" if self._total <= 1 else f"[{self._index}/{self._total}] "

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
        """Install the bridge handler for the operation; restore logging after."""
        if not self._enabled:
            yield
            return
        root = logging.getLogger()
        prev_level = root.level
        handler = ProgressHandler(self)
        handler.setFormatter(logging.Formatter("  %(levelname)s: %(message)s"))
        root.addHandler(handler)
        # Let INFO records reach the handler (so per-step progress shows); the
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
        self._w(f"{self._prefix()}{label} …")
        self._line_open = True
        if not self._isatty:
            self._w("\n")
            self._line_open = False
        try:
            yield
        except Exception:
            self._clear_line()
            self._w(f"{self._prefix()}{label} ✗ failed\n")
            raise
        else:
            self._clear_line()
            elapsed = time.monotonic() - start
            self._w(f"{self._prefix()}{label} ✓ ({elapsed:.1f}s)\n")

    def substep(self, n: int, total: int) -> None:
        """Render a within-stage percentage from an (n, total) sub-step count."""
        if not self._enabled or total <= 0:
            return
        pct = max(0, min(100, int(100 * n / total)))
        if self._isatty:
            self._w(f"\r\033[K{self._prefix()}{self._label} … {pct:3d}% ({n}/{total})")
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
