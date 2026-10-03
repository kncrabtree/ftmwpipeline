"""CLI plumbing for machine-contract accessors under the ``read`` object.

A contract accessor is a read-only function that returns a result the machine
contract has frozen. This module is the one place that turns such a function
into a ``read <name>`` verb: it runs the accessor, passes the result through
:func:`ftmwpipeline.serialize.to_jsonable`, prints the JSON envelope to stdout,
writes array fields as ``.npy`` files into ``--output`` (the envelope names the
file in place of the array), and, under ``--format json``, reports a contract
error as its ``to_dict()`` JSON on stderr with a mapped exit code.

Registering a new accessor is one call from ``read_commands``::

    register_accessor(
        read_sub,
        "frequency_calibration",
        lambda path, **kw: Pipeline.open(path).frequency_calibration(),
        help="Frequency calibration stamp",
        schema="ftmw/frequency_calibration@1",
    )

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

from ..contract import PipelineFileError
from ..serialize import ArrayCollector, to_jsonable
from .utils import setup_logging

#: Exit code for each contract error code (CLI_STRATEGY: 1 user error,
#: 2 processing error). The single place this mapping lives; a code absent
#: here exits 1.
EXIT_CODES: Dict[str, int] = {
    "stage_not_run": 1,
    "not_found": 1,
    "incomplete_provenance": 1,
    "file_exists": 1,
    "file_incompatible": 1,
    "epoch_mismatch": 1,
    "file_corrupt": 2,
}
DEFAULT_ERROR_EXIT = 1


def exit_code_for(exc: PipelineFileError) -> int:
    """The CLI exit code for a contract error."""
    return EXIT_CODES.get(exc.code, DEFAULT_ERROR_EXIT)


def report_contract_error(exc: PipelineFileError, fmt: str) -> int:
    """Report *exc* (JSON dict on stderr under json, ``Error:`` text otherwise)."""
    if fmt == "json":
        print(json.dumps(to_jsonable(exc.to_dict()), allow_nan=False), file=sys.stderr)
    else:
        print(f"Error: {exc}", file=sys.stderr)
    return exit_code_for(exc)


def emit_contract_result(
    result: Any, *, schema: Optional[str], output: Optional[str]
) -> None:
    """Serialize *result*, write its arrays to *output*, print the envelope."""
    collector = ArrayCollector()
    payload = to_jsonable(result, schema=schema, arrays=collector)
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
    schema: Optional[str] = None,
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
    return 0


def register_accessor(
    read_sub: Any,
    name: str,
    accessor: Callable[..., Any],
    *,
    help: str,
    schema: Optional[str] = None,
    takes_file: bool = True,
    add_args: Optional[Callable[[argparse.ArgumentParser], None]] = None,
    call_kwargs: Optional[Callable[[argparse.Namespace], Dict[str, Any]]] = None,
) -> argparse.ArgumentParser:
    """Add ``read <name>`` for a contract accessor to the ``read`` subparsers.

    ``accessor`` is called as ``accessor(path, **call_kwargs(args))`` (or
    ``accessor(**...)`` when ``takes_file`` is false). ``schema`` stamps the
    envelope when the result is a plain dict that carries no stamp already.
    """
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
        func=lambda a: run_accessor_command(
            a,
            accessor,
            schema=schema,
            takes_file=takes_file,
            call_kwargs=call_kwargs,
        )
    )
    return parser
