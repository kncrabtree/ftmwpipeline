"""Uniform ``--json`` machine-readable output for every CLI verb.

``dev-docs/CONTRACT_STRATEGY.md`` (section Serialization) makes ``--json`` one
switch on every verb. :func:`install_json` adds it to each leaf verb parser and
wraps the verb's handler so that, under ``--json``:

- human prints on stdout are discarded (logs stay on stderr), and stdout
  carries exactly one JSON document;
- a stage-running or curation verb calls :func:`record_run_result` and gets the
  ``ftmw/run_result@1`` envelope;
- any other verb calls :func:`record_payload` for its natural payload; a plot
  verb records nothing and gets ``{"paths": [...]}``, the image files written;
- a failing verb (nonzero exit, nothing recorded) has its captured stdout
  replayed on stderr so the message is not lost.

Everything is serialized through :func:`ftmwpipeline.serialize.to_jsonable`
with ``allow_nan=False``. A contract error still propagates to
``cli.main.main``, which prints the ``ftmw/error@1`` dict on stderr.

The human (non ``--json``) output of a verb is never touched: the recorders
are no-ops for the verb's own printing.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence

from ..contract import RUN_RESULT_SCHEMA
from ..serialize import to_jsonable

#: Namespace attribute: ``--json`` was given (or a ``--format json`` synonym).
_MODE_ATTR = "ftmw_json_mode"
#: Namespace attribute holding what the verb recorded.
_SLOT_ATTR = "_ftmw_json_slot"
#: Namespace attribute holding the verb's ``"<object> <verb>"`` name.
_VERB_ATTR = "_ftmw_json_verb"
#: Namespace attribute: a ``read`` accessor already prints its envelope.
PASSTHROUGH_ATTR = "json_passthrough"
#: Verbs whose existing ``--format json`` is a synonym for ``--json``.
_FORMAT_SYNONYM_VERBS = ("info", "timebase state", "review snap-tolerance")


def json_mode(args: argparse.Namespace) -> bool:
    """True when the verb runs under ``--json`` (or its ``--format json``)."""
    return bool(getattr(args, _MODE_ATTR, False))


def invalidated_of(result: Any) -> List[str]:
    """The stages *result* invalidated, as a list of canonical names.

    Reads the ``"invalidated"`` key of a result dict, or the ``invalidated``
    attribute of a dataclass / ``ComplexFT`` / ``PeakList``. Anything else (or a
    non-sequence such as a ``bool``) is "nothing invalidated".
    """
    if result is None:
        return []
    if isinstance(result, Mapping):
        value = result.get("invalidated", ())
    else:
        value = getattr(result, "invalidated", ())
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [str(getattr(s, "value", s)) for s in value]


def record_run_result(
    args: argparse.Namespace,
    *,
    stage: Optional[str],
    summary: Mapping[str, Any],
    result: Any = None,
    invalidated: Optional[Sequence[str]] = None,
) -> None:
    """Record the ``ftmw/run_result@1`` for the running verb.

    ``stage`` is the canonical stage name (a ``contract.Stage`` value) or
    ``None``; ``summary`` the scalars the human output reports (no arrays);
    ``invalidated`` is taken from ``result`` unless given.
    """
    names = list(invalidated) if invalidated is not None else invalidated_of(result)
    setattr(
        args,
        _SLOT_ATTR,
        {
            "kind": "run",
            "stage": getattr(stage, "value", stage),
            "invalidated": names,
            "summary": dict(summary),
        },
    )


def record_payload(args: argparse.Namespace, payload: Any) -> None:
    """Record the natural payload of a non-stage verb."""
    setattr(args, _SLOT_ATTR, {"kind": "payload", "payload": payload})


def _dumps(obj: Any, **kwargs: Any) -> str:
    return json.dumps(obj, allow_nan=False, indent=2, **kwargs)


def _render(args: argparse.Namespace, slot: Dict[str, Any]) -> str:
    if slot["kind"] == "run":
        body = {
            "verb": getattr(args, _VERB_ATTR, None),
            "stage": slot["stage"],
            "invalidated": slot["invalidated"],
            "summary": slot["summary"],
        }
        return _dumps(to_jsonable(body, schema=RUN_RESULT_SCHEMA))
    return _dumps(to_jsonable(slot["payload"]))


@contextlib.contextmanager
def _track_saved_figures(paths: List[str]) -> Iterator[None]:
    """Record every path ``Figure.savefig`` writes while the verb runs."""
    try:
        from matplotlib.figure import Figure
    except ImportError:  # pragma: no cover - matplotlib is a hard dependency
        yield
        return
    original = Figure.savefig

    def savefig(self: Any, fname: Any, *a: Any, **kw: Any) -> Any:
        out = original(self, fname, *a, **kw)
        if isinstance(fname, (str, os.PathLike)):
            paths.append(os.fspath(fname))
        return out

    Figure.savefig = savefig  # type: ignore[method-assign]
    try:
        yield
    finally:
        Figure.savefig = original  # type: ignore[method-assign]


def _wrap(
    func: Callable[[argparse.Namespace], int],
    verb: str,
    has_json_format: bool,
) -> Callable[[argparse.Namespace], int]:
    """Wrap one verb handler with the ``--json`` behaviour."""

    def wrapped(args: argparse.Namespace) -> int:
        wants = bool(getattr(args, "json_output", False))
        synonym = (
            verb in _FORMAT_SYNONYM_VERBS and getattr(args, "format", None) == "json"
        )
        if getattr(args, PASSTHROUGH_ATTR, False) or not (wants or synonym):
            return func(args)
        if has_json_format:
            args.format = "json"
        setattr(args, _MODE_ATTR, True)
        setattr(args, _VERB_ATTR, verb)
        real_stdout = sys.stdout
        captured = io.StringIO()
        saved: List[str] = []
        with contextlib.redirect_stdout(captured), _track_saved_figures(saved):
            rc = func(args)
        slot = getattr(args, _SLOT_ATTR, None)
        if slot is None and rc == 0:
            slot = {"kind": "payload", "payload": {"paths": saved}}
        if slot is None:
            sys.stderr.write(captured.getvalue())
            return rc
        print(_render(args, slot), file=real_stdout)
        return rc

    wrapped.__name__ = getattr(func, "__name__", "verb")
    wrapped.__doc__ = func.__doc__
    return wrapped


def install_json(parser: argparse.ArgumentParser) -> None:
    """Add ``--json`` to every leaf verb of *parser* and wrap its handler."""

    def walk(p: argparse.ArgumentParser, path: List[str]) -> None:
        for action in p._actions:
            if isinstance(action, argparse._SubParsersAction):
                seen = set()
                for name, sub in action.choices.items():
                    if id(sub) in seen:  # an alias of a parser already visited
                        continue
                    seen.add(id(sub))
                    walk(sub, path + [name])
                return
        func = p.get_default("func")
        if func is None or "--json" in p._option_string_actions:
            return
        fmt = p._option_string_actions.get("--format")
        has_json_format = bool(fmt is not None and "json" in (fmt.choices or ()))
        p.add_argument(
            "--json",
            dest="json_output",
            action="store_true",
            default=False,
            help="Print machine-readable JSON on stdout (nothing else); errors "
            "go to stderr as JSON",
        )
        p.set_defaults(func=_wrap(func, " ".join(path), has_json_format))

    walk(parser, [])
