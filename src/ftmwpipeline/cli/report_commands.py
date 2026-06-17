"""CLI commands for the Stage 6 ``report`` object (object-verb grammar).

Level 1 (``report table``): render the persisted calibrated final-products
table to CSV / JSON / LaTeX. Level 2 (``report summary``): a Markdown methods +
results document. Level 3 (``report full``): a local, linked HTML site with a
per-window detail page. Reports render the persisted record; they never
recompute the fit.
"""

from __future__ import annotations

import argparse
from typing import Any, Optional

from .._internal.report_html_impl import VALID_WINDOW_FILTERS, report_full_impl
from .._internal.report_impl import (
    VALID_FORMATS,
    report_summary_impl,
    report_table_impl,
)
from .utils import add_stage_object, setup_logging


def _ensure_ftmw(path: str) -> str:
    return path if path.endswith(".ftmw") else path + ".ftmw"


def cmd_report_table(args: argparse.Namespace) -> int:
    """Render the calibrated final-products table (report Level 1)."""
    setup_logging(getattr(args, "verbose", False))
    file_path = _ensure_ftmw(args.file_path)
    fmt: str = getattr(args, "format", "csv")
    output: Optional[str] = getattr(args, "output", None)

    try:
        text = report_table_impl(file_path, fmt=fmt, output=output)
    except (ValueError, KeyError) as exc:
        print(f"Error: {exc}")
        return 1

    if output is not None:
        print(f"report table: wrote {fmt} to {output}")
    else:
        print(text, end="")
    return 0


def cmd_report_summary(args: argparse.Namespace) -> int:
    """Render the methods + results summary document (report Level 2)."""
    setup_logging(getattr(args, "verbose", False))
    file_path = _ensure_ftmw(args.file_path)
    output: Optional[str] = getattr(args, "output", None)
    include_table: bool = getattr(args, "include_table", False)

    try:
        text = report_summary_impl(
            file_path, output=output, include_table=include_table
        )
    except (ValueError, KeyError) as exc:
        print(f"Error: {exc}")
        return 1

    if output is not None:
        print(f"report summary: wrote markdown to {output}")
    else:
        print(text, end="")
    return 0


def cmd_report_full(args: argparse.Namespace) -> int:
    """Assemble the linked-HTML per-window report site (report Level 3)."""
    setup_logging(getattr(args, "verbose", False))
    file_path = _ensure_ftmw(args.file_path)
    output_dir: str = args.output_dir
    windows: str = getattr(args, "windows", "all")

    try:
        index_path = report_full_impl(file_path, output_dir=output_dir, windows=windows)
    except (ValueError, KeyError) as exc:
        print(f"Error: {exc}")
        return 1

    print(f"report full: wrote HTML site to {index_path}")
    return 0


def register_report_commands(subparsers: Any) -> None:
    """Register the ``report`` object-verb subcommands."""
    verbs = add_stage_object(
        subparsers,
        "report",
        help="Generate reports from the finalized record",
        description=(
            "Report generation over the persisted Stage 6 record.\n\n"
            "Renders the finalized analysis; never recomputes the fit. Requires\n"
            "'review run' to have consolidated the calibrated final products.\n\n"
            "Verbs: table (Level 1 data export: CSV / JSON / LaTeX),\n"
            "       summary (Level 2 methods + results document: Markdown),\n"
            "       full (Level 3 linked-HTML per-window site)"
        ),
    )

    p_table = verbs.add_parser(
        "table",
        help="Export the calibrated final-products table (CSV / JSON / LaTeX)",
        description=(
            "Serialize the persisted calibrated final-products table for\n"
            "downstream use: a CSV with a commented provenance header, a JSON\n"
            "twin, or a LaTeX booktabs table for paper supporting information."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_table.add_argument(
        "file_path", help="Path to .ftmw pipeline file (.ftmw auto-added)"
    )
    p_table.add_argument(
        "--format",
        dest="format",
        choices=VALID_FORMATS,
        default="csv",
        help="Output format (default csv).",
    )
    p_table.add_argument(
        "--output",
        dest="output",
        default=None,
        metavar="PATH",
        help="Write to PATH instead of stdout.",
    )
    p_table.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging.",
    )
    p_table.set_defaults(func=cmd_report_table)

    p_summary = verbs.add_parser(
        "summary",
        help="Methods + results document (Markdown)",
        description=(
            "Render a Markdown methods + results document: static,\n"
            "code-versioned per-stage algorithm prose interleaved with the\n"
            "per-experiment numbers from each persisted stage. The full line\n"
            "list is the companion 'report table' export unless --include-table\n"
            "inlines it."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_summary.add_argument(
        "file_path", help="Path to .ftmw pipeline file (.ftmw auto-added)"
    )
    p_summary.add_argument(
        "--output",
        dest="output",
        default=None,
        metavar="PATH",
        help="Write to PATH instead of stdout.",
    )
    p_summary.add_argument(
        "--include-table",
        dest="include_table",
        action="store_true",
        default=False,
        help="Inline the full calibrated line table instead of pointing to it.",
    )
    p_summary.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging.",
    )
    p_summary.set_defaults(func=cmd_report_summary)

    p_full = verbs.add_parser(
        "full",
        help="Linked-HTML per-window report site",
        description=(
            "Assemble a local, linked HTML site over the persisted record: an\n"
            "index (summary + final table + window links) and one detail page\n"
            "per fit window (the fit figure, fitted lines, parameter\n"
            "covariance, ledger candidates, and the fit log). Reuses the\n"
            "existing 'fit show' figure renderer; never recomputes the fit."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_full.add_argument(
        "file_path", help="Path to .ftmw pipeline file (.ftmw auto-added)"
    )
    p_full.add_argument(
        "--output-dir",
        dest="output_dir",
        required=True,
        metavar="DIR",
        help="Directory to write the HTML site into (created if absent).",
    )
    p_full.add_argument(
        "--windows",
        dest="windows",
        choices=VALID_WINDOW_FILTERS,
        default="all",
        help=(
            "Which windows get a detail page: 'all' (default) or 'attention' "
            "(only windows the review flagged). The index lists every window."
        ),
    )
    p_full.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging.",
    )
    p_full.set_defaults(func=cmd_report_full)
