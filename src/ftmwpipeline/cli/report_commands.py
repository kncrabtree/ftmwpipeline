"""CLI commands for the Stage 6 ``report`` object (object-verb grammar).

``report run`` is the default Stage 6 deliverable: it writes the Level-1
calibrated final-products table (CSV) and the self-contained Level-3 HTML report
(every window folded in) in one call, with flags to trim the output (table only,
HTML only, attention windows only, a summary without per-window detail, …).
``report table``
exports just the Level-1 table to CSV / JSON / LaTeX. Reports render the
persisted record; they never recompute the fit.
"""

from __future__ import annotations

import argparse
from typing import Any, Optional

from .._internal.report_html_impl import (
    VALID_WINDOW_FILTERS,
    report_run_impl,
)
from .._internal.report_impl import (
    VALID_FORMATS,
    report_table_impl,
)
from .utils import add_stage_object, setup_logging


def _ensure_ftmw(path: str) -> str:
    return path if path.endswith(".ftmw") else path + ".ftmw"


def _add_catalog_args(parser: argparse.ArgumentParser) -> None:
    """Add the shared ``--catalog`` / ``--catalog-nsigma`` cross-ref options."""
    parser.add_argument(
        "--catalog",
        dest="catalog",
        default=None,
        metavar="PATH",
        help=(
            "Proximity-flag each reported line against a frequency catalog "
            "(CSV: frequency_mhz[, uncertainty_khz[, label]]). Echoes the "
            "nearest catalog label; never an assignment and never alters the fit."
        ),
    )
    parser.add_argument(
        "--catalog-nsigma",
        dest="catalog_n_sigma",
        type=float,
        default=3.0,
        metavar="N",
        help="Catalog match tolerance in combined sigmas (default 3).",
    )


def cmd_report_table(args: argparse.Namespace) -> int:
    """Render the calibrated final-products table (report Level 1)."""
    setup_logging(getattr(args, "verbose", False))
    file_path = _ensure_ftmw(args.file_path)
    fmt: str = getattr(args, "format", "csv")
    output: Optional[str] = getattr(args, "output", None)

    try:
        text = report_table_impl(
            file_path,
            fmt=fmt,
            output=output,
            catalog=getattr(args, "catalog", None),
            catalog_n_sigma=getattr(args, "catalog_n_sigma", 3.0),
        )
    except (ValueError, KeyError) as exc:
        print(f"Error: {exc}")
        return 1

    if output is not None:
        print(f"report table: wrote {fmt} to {output}")
    else:
        print(text, end="")
    return 0


def cmd_report_run(args: argparse.Namespace) -> int:
    """Write the default Stage 6 deliverables: the L1 table + the L3 report."""
    file_path = _ensure_ftmw(args.file_path)

    emit_table = not getattr(args, "no_table", False)
    emit_html = not getattr(args, "level1_only", False)
    scope = "summary" if getattr(args, "summary", False) else "full"

    # Live progress: the HTML render logs per-window, which the StageProgress
    # capture renders as a percentage (rendering all windows can take minutes).
    # Verbose wants the full log instead, so the bar yields to it; otherwise we
    # skip setup_logging so its INFO handler does not flood the progress display.
    from .._internal.progress import StageProgress

    verbose = getattr(args, "verbose", False)
    show_progress = emit_html and not getattr(args, "quiet", False) and not verbose
    if not show_progress:
        setup_logging(verbose)
    reporter = StageProgress(1, enabled=show_progress)
    try:
        with reporter.capture_logs():
            with reporter.stage("rendering report"):
                out = report_run_impl(
                    file_path,
                    output_dir=args.output_dir,
                    windows=getattr(args, "windows", "all"),
                    emit_table=emit_table,
                    emit_html=emit_html,
                    table_format=getattr(args, "format", "csv"),
                    scope=scope,
                    catalog=getattr(args, "catalog", None),
                    catalog_n_sigma=getattr(args, "catalog_n_sigma", 3.0),
                )
    except (ValueError, KeyError) as exc:
        print(f"Error: {exc}")
        return 1

    if out.get("table") is not None:
        print(f"report run: wrote table to {out['table']}")
    if out.get("html") is not None:
        print(
            f"report run: wrote self-contained {scope} HTML report to {out['html']}"
        )
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
            "Verbs: run   (default deliverable: Level-1 table + Level-3 HTML report),\n"
            "       table (Level-1 data export only: CSV / JSON / LaTeX)"
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
    _add_catalog_args(p_table)
    p_table.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging.",
    )
    p_table.set_defaults(func=cmd_report_table)

    p_run = verbs.add_parser(
        "run",
        help="Default deliverable: Level-1 table + Level-3 HTML report",
        description=(
            "Write the default Stage 6 deliverables into --output-dir: the\n"
            "Level-1 calibrated line table (<stem>_lines.csv) and the\n"
            "self-contained Level-3 HTML report with every window folded in\n"
            "(<stem>_report.html). Reuses the existing renderers; never\n"
            "recomputes the fit.\n\n"
            "Trim the output with --level1-only (table only), --no-table (HTML\n"
            "only), --windows attention (only flagged windows get a detail\n"
            "page), or --summary (HTML index + methods, no per-window detail)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_run.add_argument(
        "file_path", help="Path to .ftmw pipeline file (.ftmw auto-added)"
    )
    p_run.add_argument(
        "--output-dir",
        dest="output_dir",
        required=True,
        metavar="DIR",
        help="Directory to write the report artifacts into (created if absent).",
    )
    p_run.add_argument(
        "--windows",
        dest="windows",
        choices=VALID_WINDOW_FILTERS,
        default="all",
        help=(
            "Which windows get a detail page: 'all' (default) or 'attention' "
            "(only windows the review flagged). The index lists every window."
        ),
    )
    p_run.add_argument(
        "--format",
        dest="format",
        choices=VALID_FORMATS,
        default="csv",
        help="Format for the Level-1 table artifact (default csv).",
    )
    artifact = p_run.add_mutually_exclusive_group()
    artifact.add_argument(
        "--level1-only",
        dest="level1_only",
        action="store_true",
        default=False,
        help="Write only the Level-1 table (skip the HTML report).",
    )
    artifact.add_argument(
        "--no-table",
        dest="no_table",
        action="store_true",
        default=False,
        help="Write only the HTML report (skip the Level-1 table).",
    )
    p_run.add_argument(
        "--summary",
        dest="summary",
        action="store_true",
        default=False,
        help=(
            "Emit the self-contained HTML index + methods only, without the "
            "per-window detail (a lighter, portable report)."
        ),
    )
    _add_catalog_args(p_run)
    p_run.add_argument(
        "--quiet",
        dest="quiet",
        action="store_true",
        default=False,
        help="Suppress the live per-window render-progress display.",
    )
    p_run.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging.",
    )
    p_run.set_defaults(func=cmd_report_run)
