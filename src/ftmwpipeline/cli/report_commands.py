"""CLI commands for the Stage 6 ``report`` object (object-verb grammar).

Level 1 (``report table``): render the persisted calibrated final-products
table to CSV / JSON / LaTeX. Reports render the persisted record; they never
recompute the fit. Levels 2 (``summary``) and 3 (``full``) are added later.
"""

from __future__ import annotations

import argparse
from typing import Any, Optional

from .._internal.report_impl import VALID_FORMATS, report_table_impl
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
            "Verbs: table (Level 1 data export: CSV / JSON / LaTeX)"
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
