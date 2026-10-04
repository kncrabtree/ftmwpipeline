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

``read <accessor>`` (one per machine-contract accessor, spelled as its API
name: ``read capabilities``, ``read read_table``, ``read window_status``, ...)
prints the accessor's schema-stamped JSON envelope; see
:func:`register_contract_accessors`. ``table`` / ``meta`` / ``list`` are the
human-facing views and are not contract.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict, List, Optional

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
from ..contract import (
    CALIBRATION_SCHEMA,
    CAPABILITIES_SCHEMA,
    DISPLAY_FT_SCHEMA,
    DISPLAY_UNITS_SCHEMA,
    FID_SAMPLES_SCHEMA,
    FINAL_PRODUCTS_SCHEMA,
    FIT_THRESHOLDS_SCHEMA,
    MANIFEST,
    METADATA_SCHEMA,
    PIPELINE_INFO_SCHEMA,
    REVIEW_LOG_SCHEMA,
    SETTINGS_DEFAULTS_SCHEMA,
    SETTINGS_SCHEMA,
    SNAP_TOLERANCE_SCHEMA,
    SOURCE_PREVIEW_SCHEMA,
    SPECTRUM_MODEL_SCHEMA,
    TABLE_SCHEMA,
    TABLES_SCHEMA,
    WINDOW_MODEL_SCHEMA,
    WINDOW_STATUS_SCHEMA,
    Absent,
)
from ..file_manager import PipelineFileError
from .contract_commands import exit_code_for, register_accessor, report_contract_error
from .utils import setup_logging

#: What a bad request looks like here: a missing or unreadable file, an unknown
#: table/column/format, or a stage that has not been run. All carry a message
#: that already says what to do. A contract error exits with the code from the
#: CLI's one table (:func:`exit_code_for`); anything else here exits 1.
_USER_ERRORS = (FileNotFoundError, PipelineFileError, ValueError)


def _report(exc: BaseException, fmt: str) -> int:
    """Report *exc* and return its exit code.

    Under ``--format json`` a contract error goes to stderr as its error dict;
    otherwise ``Error: ...`` is printed as before.
    """
    if isinstance(exc, PipelineFileError):
        if fmt == "json":
            return report_contract_error(exc, fmt)
        print(f"Error: {exc}")
        return exit_code_for(exc)
    print(f"Error: {exc}")
    return 1


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
        return _report(exc, getattr(args, "format", "csv"))
    return _emit(text, getattr(args, "output", None), args.table)


def cmd_read_meta(args: argparse.Namespace) -> int:
    """Dump the file's cheap top-level scalars."""
    setup_logging(getattr(args, "verbose", False))
    file_path = _ensure_ftmw(args.file_path)
    try:
        metadata = read_metadata_impl(file_path)
        text = format_metadata_impl(metadata, getattr(args, "format", "csv"))
    except _USER_ERRORS as exc:
        return _report(exc, getattr(args, "format", "csv"))
    return _emit(text, getattr(args, "output", None), "meta")


def cmd_read_list(args: argparse.Namespace) -> int:
    """Show the readable tables, their availability, and their columns."""
    setup_logging(getattr(args, "verbose", False))
    file_path = _ensure_ftmw(args.file_path)
    try:
        tables = read_tables_impl(file_path)
    except _USER_ERRORS as exc:
        return _report(exc, getattr(args, "format", "csv"))

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


def _settings_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "selector",
        nargs="?",
        default=None,
        help="Filter by dotted-path prefix, e.g. stage2b or stage2b.gaussian",
    )
    parser.add_argument(
        "--include-advanced",
        dest="include_advanced",
        action="store_true",
        help="Include advanced-tier settings",
    )
    parser.add_argument(
        "--preset",
        default=None,
        help="Preset for the .yml provenance layer (bare name or YAML path)",
    )


def _settings_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "selector": args.selector,
        "include_advanced": args.include_advanced,
        "preset": args.preset,
    }


def _table_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "table", metavar="TABLE", help="Table to read: " + ", ".join(READ_TABLES)
    )
    parser.add_argument(
        "--columns",
        default=None,
        metavar="A,B,C",
        help="Comma-separated columns to read, in order (default: all)",
    )


def _display_ft_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pad-factor",
        type=int,
        default=None,
        help="Zero-fill factor (default: the display default)",
    )


def _source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "source", help="Data source (e.g. a Blackchirp directory or a CSV)"
    )
    parser.add_argument(
        "--source-format",
        dest="source_format",
        default=None,
        metavar="NAME",
        help="Source format to use instead of auto-detection",
    )


def _grid_arg(parser: argparse.ArgumentParser) -> None:
    from .._internal.model_impl import MODEL_GRIDS

    parser.add_argument(
        "--grid",
        choices=MODEL_GRIDS,
        default="active",
        help="Grid to evaluate on: the native active FT the fit used "
        "(default) or the zero-filled display FT",
    )


def _window_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("window_id", type=int, help="Window to evaluate")
    _grid_arg(parser)
    parser.add_argument(
        "--components",
        action="store_true",
        help="Also return one array per fitted line, keyed by peak_uid",
    )


def register_contract_accessors(read_sub: Any) -> None:
    """Register ``read <name>`` for every manifest accessor (JSON envelopes).

    The verb is spelled exactly as the functional-API name, so three of them
    (``read_metadata``, ``read_tables``, ``read_table``) sit beside the
    human-facing ``meta`` / ``list`` / ``table`` verbs above, which are not
    contract.
    """
    from ..pipeline import Pipeline

    def opened(method: str) -> Callable[..., Any]:
        """A file-bound accessor: the named method of the opened Pipeline."""
        return lambda path, **kw: getattr(Pipeline.open(path), method)(**kw)

    def final_products(path: str, **kw: Any) -> Any:
        # None before Stage 6 in Python (Wave 3 migrates it); absent on the wire.
        result = Pipeline.open(path).final_products(**kw)
        if result is None:
            return {"schema": FINAL_PRODUCTS_SCHEMA, "items": Absent.NOT_RUN}
        return result

    def register(
        name: str,
        accessor: Callable[..., Any],
        schema: str,
        help: str,
        **extra: Any,
    ) -> None:
        register_accessor(read_sub, name, accessor, help=help, schema=schema, **extra)

    register(
        "capabilities",
        Pipeline.capabilities,
        CAPABILITIES_SCHEMA,
        "Machine-contract version, schemas, accessors and error codes",
    )
    register(
        "frequency_calibration",
        opened("frequency_calibration"),
        CALIBRATION_SCHEMA,
        "The frequency calibration the file is under now",
    )
    register(
        "refit_snap_tol_mhz",
        opened("refit_snap_tol_mhz"),
        SNAP_TOLERANCE_SCHEMA,
        "The Stage 6 curation snap tolerance (MHz) of this file",
    )
    register(
        "read_metadata",
        opened("read_metadata"),
        METADATA_SCHEMA,
        "The file's cheap top-level scalars",
    )
    register(
        "read_tables",
        opened("read_tables"),
        TABLES_SCHEMA,
        "The readable tables, their availability and columns",
    )
    register(
        "read_table",
        opened("read_table"),
        TABLE_SCHEMA,
        "One persisted table as columns (arrays via --output)",
        add_args=_table_args,
        call_kwargs=lambda a: {
            "table": a.table,
            "columns": _split_columns(a.columns),
        },
    )
    register(
        "settings_defaults",
        Pipeline.settings_defaults,
        SETTINGS_DEFAULTS_SCHEMA,
        "Every setting at its hard default (no file needed)",
        add_args=_settings_args,
        call_kwargs=_settings_kwargs,
    )
    register(
        "settings_show",
        opened("settings_show"),
        SETTINGS_SCHEMA,
        "Resolved value and provenance of each setting",
        add_args=_settings_args,
        call_kwargs=_settings_kwargs,
    )
    register(
        "get_final_products",
        final_products,
        FINAL_PRODUCTS_SCHEMA,
        "The persisted Stage 6 calibrated final-products table",
    )
    register(
        "review_log",
        opened("review_log"),
        REVIEW_LOG_SCHEMA,
        "The persisted Stage 6 decision log, in execution order",
    )
    register(
        "get_pipeline_info",
        opened(MANIFEST.pipeline_names["get_pipeline_info"]),
        PIPELINE_INFO_SCHEMA,
        "File status, stage record and analysis environment",
    )
    register(
        "compute_display_ft",
        opened("compute_display_ft"),
        DISPLAY_FT_SCHEMA,
        "Zero-padded display FT over the analysis band (arrays via --output)",
        add_args=_display_ft_args,
        call_kwargs=lambda a: (
            {} if a.pad_factor is None else {"pad_factor": a.pad_factor}
        ),
    )
    register(
        "fid_samples",
        opened("fid_samples"),
        FID_SAMPLES_SCHEMA,
        "Stored Stage 0 FID samples (samples go to .npy under --output)",
    )
    register(
        "display_units",
        opened("display_units"),
        DISPLAY_UNITS_SCHEMA,
        "Display amplitude scale and units label",
    )
    register(
        "fit_thresholds",
        opened("fit_thresholds"),
        FIT_THRESHOLDS_SCHEMA,
        "Thresholds the persisted Stage 5 fit applied",
    )
    register(
        "window_status",
        opened("window_status"),
        WINDOW_STATUS_SCHEMA,
        "Per-window status: plan, created windows, Stage 5 coverage",
    )
    register(
        "preview_source",
        Pipeline.preview_source,
        SOURCE_PREVIEW_SCHEMA,
        "Format and FID table of a data source, without importing it",
        add_args=_source_args,
        call_kwargs=lambda a: {"source": a.source, "format_name": a.source_format},
    )
    register(
        "window_model",
        opened("window_model"),
        WINDOW_MODEL_SCHEMA,
        "One window's fitted model, data, noise and spur mask (arrays: --output)",
        add_args=_window_model_args,
        call_kwargs=lambda a: {
            "window_id": a.window_id,
            "grid": a.grid,
            "components": a.components,
        },
    )
    register(
        "spectrum_model",
        opened("spectrum_model"),
        SPECTRUM_MODEL_SCHEMA,
        "The whole fitted spectrum model, data and residual (arrays via --output)",
        add_args=_grid_arg,
        call_kwargs=lambda a: {"grid": a.grid},
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
  ftmwpipeline read capabilities
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

    register_contract_accessors(read_sub)

    # 'read' with no subcommand prints its help.
    def _read_help(args: argparse.Namespace) -> int:
        read.print_help()
        return 1

    read.set_defaults(func=_read_help)
