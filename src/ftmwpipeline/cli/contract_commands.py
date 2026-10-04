"""CLI plumbing for machine-contract accessors under the ``read`` object.

A contract accessor is a read-only function that returns a result the machine
contract has frozen. This module is the one place that turns such a function
into a ``read <name>`` verb (every manifest accessor has exactly one, spelled as
its API name): it runs the accessor, wraps the result in its schema-stamped
envelope (:func:`contract_envelope`), passes it through
:func:`ftmwpipeline.serialize.to_jsonable`, prints the JSON to stdout, writes
array fields as ``.npy`` files into ``--output`` (the envelope names the file
in place of the array), and reports a contract error as its ``to_dict()`` JSON
on stderr with a mapped exit code.

It also holds the CLI's one contract-code -> exit-code table
(:data:`EXIT_CODES`, :func:`exit_code_for`); every verb that reports a
:class:`~ftmwpipeline.file_manager.PipelineFileError` exits through it.

Registering a new accessor is one call from ``read_commands``::

    register_accessor(
        read_sub,
        "frequency_calibration",
        lambda path, **kw: Pipeline.open(path).frequency_calibration(**kw),
        help="Frequency calibration stamp",
        schema=CALIBRATION_SCHEMA,
    )

Whether the verb takes a file argument is read from ``MANIFEST.file_bound``.

Extra verb-specific options go through ``add_args`` (a callable receiving the
verb's parser) and reach the accessor via ``call_kwargs`` (``args -> dict``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from ..contract import MANIFEST, Absent, PipelineFileError
from ..serialize import ArrayCollector, to_jsonable
from .utils import setup_logging

#: Exit code for each contract error code that does not exit
#: :data:`DEFAULT_ERROR_EXIT` (CONTRACT_STRATEGY §Errors). The single place this
#: mapping lives; every other code exits 1 (``callback_failed`` among them).
#: ``cancelled`` (a set cancel token, e.g. the first Ctrl-C of a long verb)
#: exits 130 like a bare interrupt.
EXIT_CODES: Dict[str, int] = {
    "file_corrupt": 2,
    "algorithm_failed": 2,
    "cancelled": 130,
}
DEFAULT_ERROR_EXIT = 1

#: Exit code for an interrupted (Ctrl-C) command, as the stage verbs use.
INTERRUPTED_EXIT = 130


def exit_code_for(exc: PipelineFileError) -> int:
    """The CLI exit code for a contract error."""
    return EXIT_CODES.get(exc.code, DEFAULT_ERROR_EXIT)


def report_contract_error(exc: PipelineFileError, fmt: str) -> int:
    """Report *exc* (JSON dict on stderr under json, ``Error:`` text otherwise)."""
    if fmt == "json":
        # to_dict() is already wire-form; serializing it again would trip the
        # reserved ``_absent`` key rule.
        print(json.dumps(exc.to_dict(), allow_nan=False), file=sys.stderr)
    else:
        print(f"Error: {exc}", file=sys.stderr)
    return exit_code_for(exc)


def contract_envelope(result: Any, schema: str) -> Any:
    """The object to serialize for *result* under *schema*.

    A list or tuple becomes ``{"schema", "items": [...]}``; a scalar becomes
    ``{"schema", "value": x}``. ``None`` -- a pre-contract accessor's "not
    produced yet", such as ``get_final_products`` before Stage 6 -- becomes
    ``{"schema", "value": null, "value_absent": "not_run"}``. Anything else (a
    dict, a dataclass, a ``ComplexFT``) is stamped directly by
    :func:`to_jsonable`, which also checks that a payload already stamped in
    Python carries the same name.
    """
    if result is None:
        return {"schema": schema, "value": Absent.NOT_RUN}
    if isinstance(result, (list, tuple)):
        return {"schema": schema, "items": result}
    if isinstance(result, (bool, int, float, str, np.generic)):
        return {"schema": schema, "value": result}
    return result


def emit_contract_result(result: Any, *, schema: str, output: Optional[str]) -> None:
    """Serialize *result*, write its arrays to *output*, print the envelope."""
    collector = ArrayCollector()
    payload = to_jsonable(
        contract_envelope(result, schema), schema=schema, arrays=collector
    )
    if collector.arrays:
        if output is None:
            raise ValueError(
                "this result contains array data; pass --output DIR to write "
                "it as .npy files"
            )
        out_dir = Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, array in collector.arrays.items():
            np.save(out_dir / name, array)
    print(json.dumps(payload, allow_nan=False, indent=2))


def run_accessor_command(
    args: argparse.Namespace,
    accessor: Callable[..., Any],
    *,
    schema: str,
    takes_file: bool = True,
    call_kwargs: Optional[Callable[[argparse.Namespace], Dict[str, Any]]] = None,
) -> int:
    """Run one accessor for the CLI and return the process exit code."""
    setup_logging(getattr(args, "verbose", False))
    fmt = getattr(args, "format", "json")
    try:
        kwargs = call_kwargs(args) if call_kwargs else {}
        if takes_file:
            path = args.file_path
            if not path.endswith(".ftmw"):
                path += ".ftmw"
            result = accessor(path, **kwargs)
        else:
            result = accessor(**kwargs)
        emit_contract_result(
            result, schema=schema, output=getattr(args, "output", None)
        )
    except PipelineFileError as exc:
        return report_contract_error(exc, fmt)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return DEFAULT_ERROR_EXIT
    except KeyboardInterrupt:
        print("Error: Interrupted by user", file=sys.stderr)
        return INTERRUPTED_EXIT
    return 0


def register_accessor(
    read_sub: Any,
    name: str,
    accessor: Callable[..., Any],
    *,
    help: str,
    schema: str,
    add_args: Optional[Callable[[argparse.ArgumentParser], None]] = None,
    call_kwargs: Optional[Callable[[argparse.Namespace], Dict[str, Any]]] = None,
) -> argparse.ArgumentParser:
    """Add ``read <name>`` for a manifest accessor to the ``read`` subparsers.

    ``accessor`` is called as ``accessor(path, **call_kwargs(args))`` when
    ``MANIFEST.file_bound[name]``, else as ``accessor(**call_kwargs(args))``.
    ``schema`` is the accessor's schema name: it stamps the envelope (see
    :func:`contract_envelope`) and must equal any stamp the result already
    carries.
    """
    takes_file = MANIFEST.file_bound[name]
    parser: argparse.ArgumentParser = read_sub.add_parser(
        name, help=help, description=help
    )
    if takes_file:
        parser.add_argument("file_path", help="Path to the .ftmw experiment")
    if add_args:
        add_args(parser)
    parser.add_argument(
        "--format",
        choices=("json",),
        default="json",
        help="Output format (json only; default: json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        metavar="DIR",
        help="Directory for .npy files of any array fields",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.set_defaults(
        # Already machine-readable: --json is accepted and changes nothing.
        json_passthrough=True,
        func=lambda a: run_accessor_command(
            a,
            accessor,
            schema=schema,
            takes_file=takes_file,
            call_kwargs=call_kwargs,
        ),
    )
    return parser
