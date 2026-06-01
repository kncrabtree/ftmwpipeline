"""CLI subcommands for the companion parameter-tuning surface (``tune``).

Nested subcommands under ``tune``:

- ``tune list``  — enumerate the registered tunable knobs.
- ``tune scan``  — sweep one knob across a grid on a copy of a ``.ftmw``,
  print the metric table, write a CSV (and a plot if the knob has one), and
  show how to apply a chosen value.

Per the repo's pattern (cf. ``start_commands``) the CLI reaches the shared
``_internal.tuning`` engine directly. ``--interactive`` is the one CLI-only
affordance — it steers a plot to an interactive backend instead of a file; the
``Pipeline`` / functional-API ``tune_scan`` always write a file so script
callers own figure handling.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, List, Optional

from .utils import print_error, setup_logging

logger = logging.getLogger(__name__)


def cmd_tune_list(args: argparse.Namespace) -> int:
    """Print the registered tunable knobs as a table."""
    from .._internal.tuning import list_knobs

    specs = list_knobs(getattr(args, "stage", None))
    if not specs:
        print("No tunable knobs registered" + (
            f" for stage {args.stage!r}." if getattr(args, "stage", None) else "."
        ))
        return 0

    rows = [
        (s.path, s.stage, s.inst_sensitivity,
         ",".join(_fmt(v) for v in s.default_grid))
        for s in specs
    ]
    headers = ("knob", "stage", "inst", "default grid")
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows))
        for i in range(len(headers))
    ]
    sep = "  "
    print(sep.join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print(sep.join("-" * widths[i] for i in range(len(headers))))
    for r in rows:
        print(sep.join(r[i].ljust(widths[i]) for i in range(len(r))))
    print()
    print("Run 'ftmwpipeline tune scan <file> --knob <knob>' to sweep one.")
    return 0


def cmd_tune_scan(args: argparse.Namespace) -> int:
    """Sweep a knob across a grid and report the metric table."""
    setup_logging(getattr(args, "verbose", False))
    from .._internal.tuning import get_knob, run_scan

    file_path = args.file_path
    if not file_path.endswith(".ftmw"):
        file_path = file_path + ".ftmw"

    try:
        spec = get_knob(args.knob)
    except KeyError as e:
        print_error(str(e))
        return 1

    grid: Optional[List[float]] = None
    if args.grid is not None:
        try:
            grid = [float(x) for x in args.grid.split(",") if x.strip()]
        except ValueError:
            print_error(f"--grid must be comma-separated numbers, got {args.grid!r}")
            return 1

    output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()

    # Quiet the per-value stage logging so the progress indicator stays clean
    # (a sweep re-runs the stage once per value, repeating any INFO/WARNING).
    # Verbose keeps it all. Saved/restored so in-process callers are unaffected.
    pkg_logger = logging.getLogger("ftmwpipeline")
    prev_level = pkg_logger.level
    if not getattr(args, "verbose", False):
        pkg_logger.setLevel(logging.ERROR)
    try:
        result = run_scan(
            spec,
            Path(file_path),
            grid=grid,
            output_dir=output_dir,
            reuse=args.reuse,
            make_plot=not args.no_plot,
            interactive=args.interactive,
            quiet=args.quiet,
        )
    except FileNotFoundError as e:
        print_error(f"Pipeline file not found: {e}")
        return 1
    except Exception as e:
        print_error(f"Knob scan failed: {e}")
        if getattr(args, "verbose", False):
            import traceback

            traceback.print_exc()
        return 1
    finally:
        pkg_logger.setLevel(prev_level)

    print()
    print(result.as_table())
    print()
    if result.csv_path is not None:
        print(f"CSV: {result.csv_path}")
    if result.plot_path is not None:
        print(f"Plot: {result.plot_path}")
    print()
    print(result.apply_instructions)
    return 0


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def register_tune_commands(subparsers: Any) -> None:
    """Register the ``tune`` namespace and its subcommands."""
    tune = subparsers.add_parser(
        "tune",
        help="Scan/visualize tunable pipeline parameters for your instrument",
        description=(
            "Companion tuning surface. 'tune list' enumerates the tunable "
            "knobs; 'tune scan' sweeps one across a grid on a copy of a .ftmw "
            "and reports a metric table (plus a CSV and, where available, a "
            "plot), with instructions for applying a chosen value."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    tune_sub = tune.add_subparsers(dest="tune_command", help="tune subcommands")

    p_list = tune_sub.add_parser(
        "list",
        help="List registered tunable knobs (optionally filtered by stage)",
    )
    p_list.add_argument(
        "--stage",
        type=str,
        default=None,
        help="Restrict to one stage label, e.g. stage2_noise / start_detection",
    )
    p_list.set_defaults(func=cmd_tune_list)

    p_scan = tune_sub.add_parser(
        "scan",
        help="Sweep one knob across a grid and report the metric table",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_scan.add_argument(
        "file_path",
        help="Path to .ftmw built through the knob's upstream stage",
    )
    p_scan.add_argument(
        "--knob",
        required=True,
        help="Dotted knob path (see 'tune list'), e.g. stage2.scatter.window_mhz",
    )
    p_scan.add_argument(
        "--grid",
        type=str,
        default=None,
        help="Comma-separated values to sweep (default: the knob's grid)",
    )
    p_scan.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for the CSV/plot/working copy (default: current dir)",
    )
    p_scan.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse an existing working copy instead of re-copying the input",
    )
    p_scan.add_argument(
        "--interactive",
        action="store_true",
        help="Show the plot interactively instead of writing a file (CLI-only)",
    )
    p_scan.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting even when the knob has a plot adapter",
    )
    p_scan.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress the per-value progress indicator",
    )
    p_scan.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    p_scan.set_defaults(func=cmd_tune_scan)

    # 'tune' with no subcommand prints its help.
    def _tune_help(args: argparse.Namespace) -> int:
        tune.print_help()
        return 1

    tune.set_defaults(func=_tune_help)
