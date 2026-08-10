"""CLI subcommands for the cross-cutting ``read`` meta-object.

``read`` is the file's read-only data tap: it dumps what a stage persisted, as
delimited text, without recomputing anything and without reconstructing the
full record. ``read table <file> <name>`` writes a CSV (or TSV / JSON) of one
table's columns; ``read meta <file>`` writes the cheap top-level scalars;
``read list <file>`` shows which tables the file carries and what columns each
one has.

This is the raw persisted data, deliberately: floats round-trip exactly and the
on-disk sentinels are preserved. The *presentation* view -- the calibrated,
uncertainty-budgeted final line list -- is ``report table``, not this.

All the work lives in ``_internal.read_impl``, shared with
``Pipeline.read_table`` / ``api.read_table``; this module only parses arguments
and formats output.
"""

from __future__ import annotations

import argparse
from typing import Any, List, Optional

from .._internal.read_impl import (
    READ_TABLES,
    VALID_READ_FORMATS,
    format_metadata_impl,
    format_table_impl,
    read_metadata_impl,
    read_table_impl,
    read_tables_impl,
    write_text_impl,
)
from ..file_manager import PipelineFileError
from .utils import setup_logging

#: What a bad request looks like here: a missing or unreadable file, an unknown
#: table/column/format, or a stage that has not been run. All of them are user
#: errors (exit 1), and all carry a message that already says what to do.
_USER_ERRORS = (FileNotFoundError, PipelineFileError, ValueError)


def _ensure_ftmw(path: str) -> str:
    return path if path.endswith(".ftmw") else path + ".ftmw"


def _split_columns(raw: Optional[str]) -> Optional[List[str]]:
    """Parse ``--columns a,b,c`` into a column list (``None`` when unset)."""
    if raw is None:
        return None
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if not names:
        raise ValueError("--columns was given but lists no column names")
    return names


def _emit(text: str, output: Optional[str], what: str) -> int:
    """Write *text* to *output* or stdout, and report which happened."""
    if output is None:
        print(text, end="")
    else:
        path = write_text_impl(text, output)
        print(f"read {what}: wrote {path}")
    return 0


def cmd_read_table(args: argparse.Namespace) -> int:
    """Dump one persisted table as delimited text."""
    setup_logging(getattr(args, "verbose", False))
    file_path = _ensure_ftmw(args.file_path)
    try:
        columns = _split_columns(getattr(args, "columns", None))
        table = read_table_impl(file_path, args.table, columns)
        text = format_table_impl(table, getattr(args, "format", "csv"))
    except _USER_ERRORS as exc:
        print(f"Error: {exc}")
        return 1
    return _emit(text, getattr(args, "output", None), args.table)


def cmd_read_meta(args: argparse.Namespace) -> int:
    """Dump the file's cheap top-level scalars."""
    setup_logging(getattr(args, "verbose", False))
    file_path = _ensure_ftmw(args.file_path)
    try:
        metadata = read_metadata_impl(file_path)
        text = format_metadata_impl(metadata, getattr(args, "format", "csv"))
    except _USER_ERRORS as exc:
        print(f"Error: {exc}")
        return 1
    return _emit(text, getattr(args, "output", None), "meta")


def cmd_read_list(args: argparse.Namespace) -> int:
    """Show the readable tables, their availability, and their columns."""
    setup_logging(getattr(args, "verbose", False))
    file_path = _ensure_ftmw(args.file_path)
    try:
        tables = read_tables_impl(file_path)
    except _USER_ERRORS as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Readable tables in {file_path}:")
    width = max((len(name) for name in tables), default=0)
    for name, entry in tables.items():
        if entry["available"]:
            rows = entry["n_rows"]
            status = f"{rows} row(s)" if rows is not None else "available"
        else:
            status = "not available (stage not run)"
        print(f"  {name:<{width}}  {status}")
        print(f"    group:   {entry['group']}")
        print(f"    columns: {', '.join(entry['columns'])}")
    print()
    print(
        "Dump one with 'read table <file> <name> [--columns a,b] "
        "[--output out.csv]'."
    )
    print(
        "These are the persisted values as stored. For the calibrated final "
        "line list use 'report table'."
    )
    return 0


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--format`` / ``--output`` / ``--verbose`` options."""
    parser.add_argument(
        "--format",
        choices=VALID_READ_FORMATS,
        default="csv",
        help="Output format (default: csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        metavar="PATH",
        help="Write to this file instead of stdout",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )


def register_read_commands(subparsers: Any) -> None:
    """Register the ``read`` object group and its verbs."""
    read = subparsers.add_parser(
        "read",
        help="Dump persisted data (CSV/TSV/JSON) without recomputing anything",
        description=(
            "Read-only tap on a .ftmw file. Dumps what a stage persisted as "
            "delimited text -- no recomputation, no full-record "
            "reconstruction, and only the columns asked for. Values are the "
            "persisted ones (floats round-trip exactly, on-disk sentinels "
            "preserved); for the calibrated final line list use 'report table'."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ftmwpipeline read list  exp_2638.ftmw
  ftmwpipeline read table exp_2638.ftmw fit_peaks
  ftmwpipeline read table exp_2638.ftmw fit_peaks \\
      --columns frequency_mhz,decay_rate,shape --output lines.csv
  ftmwpipeline read table exp_2638.ftmw windows --columns window_id,freq_min,freq_max
  ftmwpipeline read meta  exp_2638.ftmw --format json
        """,
    )
    read_sub = read.add_subparsers(dest="read_command", help="read subcommands")

    p_table = read_sub.add_parser(
        "table",
        help="Dump one persisted table's columns",
        description=(
            "Dump one persisted table. Pass --columns to read only what you "
            "need -- that is what keeps the read cheap on a large file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_table.add_argument("file_path", help="Path to the .ftmw experiment")
    p_table.add_argument(
        "table",
        metavar="TABLE",
        help="Table to dump (see 'read list'): " + ", ".join(READ_TABLES),
    )
    p_table.add_argument(
        "--columns",
        default=None,
        metavar="A,B,C",
        help="Comma-separated columns to read, in order (default: all). "
        "See 'read list' for each table's columns.",
    )
    _add_output_args(p_table)
    p_table.set_defaults(func=cmd_read_table)

    p_meta = read_sub.add_parser(
        "meta",
        help="Dump the file's cheap top-level scalars",
        description=(
            "Dump the file's top-level scalars as key/value rows: provenance, "
            "FID acquisition, per-stage counts, the fit's acquisition_us, and "
            "the timebase calibration. Group attributes only -- no stage "
            "artifact is deserialized."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_meta.add_argument("file_path", help="Path to the .ftmw experiment")
    _add_output_args(p_meta)
    p_meta.set_defaults(func=cmd_read_meta)

    p_list = read_sub.add_parser(
        "list",
        help="Show which tables the file carries and their columns",
    )
    p_list.add_argument("file_path", help="Path to the .ftmw experiment")
    p_list.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    p_list.set_defaults(func=cmd_read_list)

    # 'read' with no subcommand prints its help.
    def _read_help(args: argparse.Namespace) -> int:
        read.print_help()
        return 1

    read.set_defaults(func=_read_help)
