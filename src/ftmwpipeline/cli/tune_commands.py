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
from typing import Any, List, Optional, Tuple

from .utils import print_error, setup_logging

logger = logging.getLogger(__name__)


def _elide_path(path: str, prev: Optional[str]) -> str:
    """Render ``path`` with leading dotted segments shared with ``prev`` blanked
    to equal-width padding, so a column of paths reads as a prefix tree:

        stage2.group1.setting1
                     .setting2
              .group2.setting1

    Segments carry their leading dot (``"stage2"``, ``".group1"``, ``".s1"``);
    once a segment differs from the previous row, it and all that follow print
    literally. The result keeps ``len(path)`` so downstream columns stay aligned.
    """
    def _segs(p: str) -> List[str]:
        parts = p.split(".")
        return [parts[0]] + ["." + part for part in parts[1:]]

    segs = _segs(path)
    if prev is None:
        return path
    prev_segs = _segs(prev)
    out: List[str] = []
    matching = True
    for i, seg in enumerate(segs):
        if matching and i < len(prev_segs) and prev_segs[i] == seg:
            out.append(" " * len(seg))
        else:
            matching = False
            out.append(seg)
    return "".join(out)


def cmd_tune_list(args: argparse.Namespace) -> int:
    """Print the registered tunable knobs as a single prefix-elided table.

    Shows primary-tier knobs by default; ``--all`` reveals advanced ones. A
    positional ``selector`` filters by dotted-path prefix (e.g. ``stage2b`` /
    ``stage2b.gaussian``). Rows repeat dotted prefixes only when they change; a
    blank line separates stages.
    """
    from .._internal.tuning import list_knobs

    selector = getattr(args, "selector", None)
    show_all = bool(getattr(args, "all", False))
    specs = list_knobs(selector, include_advanced=show_all)
    if not specs:
        suffix = f" matching {selector!r}." if selector else "."
        print("No tunable knobs registered" + suffix
              + ("" if show_all else " (try --all for advanced knobs)."))
        return 0

    headers = ("knob", "tier", "inst", "default grid")
    rows = [
        (s.path, s.tier, s.inst_sensitivity,
         ",".join(_fmt(v) for v in s.default_grid))
        for s in specs
    ]
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows))
        for i in range(len(headers))
    ]
    sep = "  "

    def _line(cells: Any) -> str:
        return sep.join(cells[i].ljust(widths[i]) for i in range(len(cells)))

    print(_line(headers))
    print(_line(tuple("-" * widths[i] for i in range(len(headers)))))
    prev_path: Optional[str] = None
    for s, r in zip(specs, rows):
        stage = s.path.split(".")[0]
        if prev_path is not None and stage != prev_path.split(".")[0]:
            print()  # blank line between stages
        knob_cell = _elide_path(s.path, prev_path)
        print(_line((knob_cell,) + r[1:]))
        prev_path = s.path

    print()
    if not show_all:
        hidden = [
            s for s in list_knobs(selector, include_advanced=True)
            if s.tier == "advanced"
        ]
        if hidden:
            print(f"{len(hidden)} advanced knob(s) hidden; use --all to show them.")
    print("Run 'ftmwpipeline tune scan <file> --knob <knob>' to sweep one.")
    return 0


def _parse_zoom(text: str) -> List[Tuple[float, float]]:
    """Parse ``--zoom`` (``lo-hi,lo-hi,...``) into ``(lo, hi)`` MHz windows.

    Raises ``ValueError`` on a malformed entry so the caller can report it.
    """
    regions: List[Tuple[float, float]] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        lo_s, _, hi_s = part.partition("-")
        if not _ or not hi_s.strip():
            raise ValueError(part)
        lo, hi = float(lo_s), float(hi_s)
        if hi <= lo:
            raise ValueError(part)
        regions.append((lo, hi))
    return regions


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

    zoom_regions: Optional[List[Tuple[float, float]]] = None
    if getattr(args, "zoom", None):
        try:
            zoom_regions = _parse_zoom(args.zoom)
        except ValueError as e:
            print_error(
                f"--zoom must be comma-separated lo-hi MHz ranges (lo<hi), "
                f"bad entry: {e}"
            )
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
            zoom_regions=zoom_regions,
            n_zoom=getattr(args, "n_zoom", None),
            zoom_width_mhz=getattr(args, "zoom_width", None),
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


def cmd_tune_scan_all(args: argparse.Namespace) -> int:
    """Sweep every knob matched by a selector, each on its default grid."""
    setup_logging(getattr(args, "verbose", False))
    from .._internal.tuning import list_knobs, run_scan_batch

    file_path = args.file_path
    if not file_path.endswith(".ftmw"):
        file_path = file_path + ".ftmw"
    if not Path(file_path).exists():
        print_error(f"Pipeline file not found: {file_path}")
        return 1

    selector = getattr(args, "selector", None)
    specs = list_knobs(selector, include_advanced=bool(args.all))
    if not specs:
        suffix = f" matching {selector!r}" if selector else ""
        print_error(f"No tunable knobs{suffix}"
                    + ("" if args.all else " (try --all for advanced knobs)."))
        return 1

    zoom_regions: Optional[List[Tuple[float, float]]] = None
    if getattr(args, "zoom", None):
        try:
            zoom_regions = _parse_zoom(args.zoom)
        except ValueError as e:
            print_error(
                f"--zoom must be comma-separated lo-hi MHz ranges (lo<hi), "
                f"bad entry: {e}"
            )
            return 1

    output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
    print(f"Batch-scanning {len(specs)} knob(s) into {output_dir} ...")

    pkg_logger = logging.getLogger("ftmwpipeline")
    prev_level = pkg_logger.level
    if not getattr(args, "verbose", False):
        pkg_logger.setLevel(logging.ERROR)
    try:
        items = run_scan_batch(
            specs,
            Path(file_path),
            output_dir=output_dir,
            reuse=args.reuse,
            make_plot=not args.no_plot,
            quiet=args.quiet,
            zoom_regions=zoom_regions,
            n_zoom=getattr(args, "n_zoom", None),
            zoom_width_mhz=getattr(args, "zoom_width", None),
        )
    finally:
        pkg_logger.setLevel(prev_level)

    n_ok = 0
    for item in items:
        print()
        print(f"===== {item.knob} =====")
        if item.result is not None:
            n_ok += 1
            print(item.result.as_table())
            if item.result.csv_path is not None:
                print(f"CSV: {item.result.csv_path}")
            if item.result.plot_path is not None:
                print(f"Plot: {item.result.plot_path}")
        else:
            print_error(f"FAILED: {item.error}")

    n_failed = len(items) - n_ok
    print()
    print(f"Batch complete: {n_ok} ok, {n_failed} failed. Outputs in {output_dir}.")
    return 0 if n_failed == 0 else 2


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _add_zoom_args(parser: argparse.ArgumentParser) -> None:
    """Attach the per-region zoom controls shared by 'scan' and 'scan-all'.

    These steer the zoom panels of the region-based plots (Stage 3 peak
    detection, Stage 4 window planning); knobs with other plots ignore them.
    """
    parser.add_argument(
        "--zoom",
        type=str,
        default=None,
        metavar="LO-HI,LO-HI",
        help="Explicit MHz zoom windows for the plot's region panels, e.g. "
             "35000-35800,38400-38500 (overrides the auto-selected regions)",
    )
    parser.add_argument(
        "--n-zoom",
        type=int,
        default=None,
        help="How many regions to auto-select when --zoom is not given",
    )
    parser.add_argument(
        "--zoom-width",
        type=float,
        default=None,
        help="Width (MHz) of each auto-selected region when --zoom is not given",
    )


def register_tune_commands(subparsers: Any) -> None:
    """Register the ``tune`` namespace and its subcommands."""
    tune = subparsers.add_parser(
        "tune",
        help="Scan/visualize tunable pipeline parameters for your instrument",
        description=(
            "Companion tuning surface. 'tune list' enumerates the tunable "
            "knobs; 'tune scan' sweeps one across a grid on a copy of a .ftmw "
            "and reports a metric table (plus a CSV and, where available, a "
            "plot), with instructions for applying a chosen value; 'tune "
            "scan-all' batches that sweep over a whole stage / sub-block."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    tune_sub = tune.add_subparsers(dest="tune_command", help="tune subcommands")

    p_list = tune_sub.add_parser(
        "list",
        help="List registered tunable knobs, grouped by stage -> sub-block",
    )
    p_list.add_argument(
        "selector",
        nargs="?",
        default=None,
        help="Filter by dotted-path prefix, e.g. stage2b or stage2b.gaussian "
             "(a stage label like stage2_noise also matches)",
    )
    p_list.add_argument(
        "--all",
        action="store_true",
        help="Include advanced-tier knobs (hidden by default)",
    )
    p_list.add_argument(
        "--stage",
        dest="selector",
        type=str,
        default=None,
        help=argparse.SUPPRESS,  # back-compat alias for the positional selector
    )
    p_list.set_defaults(func=cmd_tune_list)

    p_scan = tune_sub.add_parser(
        "scan",
        help="Sweep one knob across a grid and report the metric table",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Progress streams to stderr as each value completes. If you launch "
            "through a wrapper that captures subprocess output (e.g. `conda "
            "run`), pass its passthrough flag (`conda run --no-capture-output`) "
            "or run the entry point in an activated env to see it live."
        ),
    )
    p_scan.add_argument(
        "file_path",
        help="Path to .ftmw built through the knob's upstream stage",
    )
    p_scan.add_argument(
        "--knob",
        required=True,
        help="Dotted knob path (see 'tune list'), e.g. stage2.window_mhz",
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
    _add_zoom_args(p_scan)
    p_scan.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    p_scan.set_defaults(func=cmd_tune_scan)

    p_scan_all = tune_sub.add_parser(
        "scan-all",
        help="Sweep every knob in a stage/sub-block on its default grid",
        description=(
            "Batch convenience over 'tune scan': sweep every knob matched by the "
            "selector, each on its default grid, writing each knob's table/CSV/"
            "plot. A knob whose required stage is absent is reported as failed "
            "and the batch continues."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_scan_all.add_argument(
        "file_path",
        help="Path to .ftmw built through the matched knobs' upstream stages",
    )
    p_scan_all.add_argument(
        "selector",
        nargs="?",
        default=None,
        help="Dotted-path prefix to scan, e.g. stage2b or stage2b.gaussian "
             "(omit to scan every knob)",
    )
    p_scan_all.add_argument(
        "--all",
        action="store_true",
        help="Include advanced-tier knobs (hidden by default)",
    )
    p_scan_all.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for the CSVs/plots/working copies (default: current dir)",
    )
    p_scan_all.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse existing working copies instead of re-copying the input",
    )
    p_scan_all.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting even when a knob has a plot adapter",
    )
    p_scan_all.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress the per-value progress indicator",
    )
    _add_zoom_args(p_scan_all)
    p_scan_all.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    p_scan_all.set_defaults(func=cmd_tune_scan_all)

    # 'tune' with no subcommand prints its help.
    def _tune_help(args: argparse.Namespace) -> int:
        tune.print_help()
        return 1

    tune.set_defaults(func=_tune_help)
