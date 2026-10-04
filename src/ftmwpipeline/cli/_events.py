"""CLI plumbing for long verbs: ``--events`` and Ctrl-C cancellation.

``dev-docs/CONTRACT_STRATEGY.md`` (§Events and cancellation, CLI):

- every long verb accepts ``--events``, which writes each event to stderr as
  one JSON line (``to_jsonable(event)``);
- the first Ctrl-C sets the operation's cancel token, so the verb stops at its
  next check point and exits 130 with the ``cancelled`` error (the existing
  code -> exit mapping in ``contract_commands``; the ``ftmw/error@1`` dict on
  stderr under ``--json``); a second Ctrl-C raises :class:`KeyboardInterrupt`
  at once.

A verb wires both in two steps::

    add_events_argument(parser)                    # at registration
    ...
    with operation_controls(args) as (events, cancel):
        result = some_impl(..., events=events, cancel=cancel)

and lets :class:`~ftmwpipeline.file_manager.OperationCancelledError` propagate
to ``cli.main.main`` like any contract error.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
from contextlib import contextmanager
from types import FrameType
from typing import Iterator, Optional, Tuple

from ..contract import CancelToken, Event, EventCallback
from ..serialize import to_jsonable
from ._json_output import json_mode

#: Namespace attribute set by ``--events``.
EVENTS_ATTR = "events"


def add_events_argument(parser: argparse.ArgumentParser) -> None:
    """Add ``--events`` to a long verb's parser."""
    parser.add_argument(
        "--events",
        dest=EVENTS_ATTR,
        action="store_true",
        default=False,
        help="Write each progress event to stderr as one JSON line "
        "(ftmw/stage_started@1, ftmw/window_progress@1, ...).",
    )


def write_event(event: Event) -> None:
    """Write *event* to stderr as one JSON line (the ``--events`` callback)."""
    sys.stderr.write(json.dumps(to_jsonable(event), allow_nan=False) + "\n")
    sys.stderr.flush()


class SignalCancelToken:
    """A cancel token a signal handler can set safely.

    :class:`threading.Event` takes a lock in ``set()``; a second SIGINT landing
    while a first handler is inside that ``set()`` re-enters it and deadlocks on
    its own lock. Setting a plain flag never blocks.
    """

    def __init__(self) -> None:
        self._flag = False

    def set(self) -> None:
        self._flag = True

    def is_set(self) -> bool:
        return self._flag


@contextmanager
def first_interrupt_cancels(
    token: SignalCancelToken, *, announce: bool = True
) -> Iterator[None]:
    """While active, the first SIGINT sets *token* instead of interrupting.

    That handler first restores Python's default one, so a second Ctrl-C
    raises :class:`KeyboardInterrupt` immediately (even one arriving while the
    first handler still runs). The previous handler is restored on exit.
    Outside the main thread (where signals cannot be handled) this does
    nothing.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGINT)

    def _handler(signum: int, frame: Optional[FrameType]) -> None:
        # Restore first: from here a further SIGINT is a KeyboardInterrupt.
        signal.signal(signal.SIGINT, signal.default_int_handler)
        token.set()
        if announce:
            # os.write, not sys.stderr: the handler may run while the main
            # thread is inside a buffered stderr write (a log line), and a
            # re-entrant buffered write raises.
            try:
                os.write(
                    2,
                    b"\nCancelling at the next check point "
                    b"(press Ctrl-C again to abort immediately)...\n",
                )
            except OSError:
                pass

    signal.signal(signal.SIGINT, _handler)
    try:
        yield
    finally:
        signal.signal(
            signal.SIGINT,
            previous if previous is not None else signal.default_int_handler,
        )


@contextmanager
def operation_controls(
    args: argparse.Namespace,
) -> Iterator[Tuple[Optional[EventCallback], CancelToken]]:
    """The ``(events, cancel)`` pair a long verb passes to its operation.

    ``events`` is :func:`write_event` under ``--events``, else ``None``;
    ``cancel`` is a :class:`SignalCancelToken` the first Ctrl-C sets.
    """
    token = SignalCancelToken()
    events: Optional[EventCallback] = (
        write_event if getattr(args, EVENTS_ATTR, False) else None
    )
    # A human gets a one-line notice on the first Ctrl-C; machine-readable
    # stderr (--events / --json) is left to the event and error lines.
    announce = events is None and not json_mode(args)
    with first_interrupt_cancels(token, announce=announce):
        yield events, token


__all__ = [
    "EVENTS_ATTR",
    "SignalCancelToken",
    "add_events_argument",
    "first_interrupt_cancels",
    "operation_controls",
    "write_event",
]
