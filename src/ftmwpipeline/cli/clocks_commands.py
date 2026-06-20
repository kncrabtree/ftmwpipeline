"""CLI for the ``clocks`` instrument clock-declaration object.

Declares the instrument clock fundamentals the Stage 5 spur gate builds its
lattice prior from, on an already-imported ``.ftmw``.  This is the no-code path
for data that did not embed clocks (CSV, a native-HDF5 file without a
``/clock_sources`` group) or for revising a declaration after import.

Grammar mirrors the ``settings`` / ``scan`` meta-objects: ``clocks <verb>``.
Each clock token is ``freq[:locked|free[:label]]`` -- a fundamental in MHz, an
optional lock state (default ``locked``), and an optional label.
"""

import argparse
from typing import Any, List, Optional

from .._internal.clocks_impl import (
    clear_clock_sources_impl,
    get_clock_sources_impl,
    remove_clock_sources_impl,
    set_clock_sources_impl,
    stage5_fit_present,
)
from ..core.stage_fit_settings import ClockSource
from .utils import setup_logging


def _parse_clock_token(token: str) -> ClockSource:
    """Parse a ``freq[:locked|free[:label]]`` token into a :class:`ClockSource`."""
    parts = token.split(":")
    try:
        freq = float(parts[0])
    except ValueError as exc:
        raise ValueError(
            f"Invalid clock frequency in {token!r}: expected a number in MHz"
        ) from exc
    locked = True
    label = ""
    if len(parts) >= 2 and parts[1] != "":
        state = parts[1].lower()
        if state in ("locked", "lock", "l", "true", "1"):
            locked = True
        elif state in ("free", "unlocked", "u", "false", "0"):
            locked = False
        else:
            raise ValueError(
                f"Invalid lock state {parts[1]!r} in {token!r}: "
                "use 'locked' or 'free'"
            )
    if len(parts) >= 3:
        label = ":".join(parts[2:])
    return ClockSource(freq_mhz=freq, locked=locked, label=label)


def _print_clocks(clocks: Optional[tuple]) -> None:
    if not clocks:
        print("   (no clock sources declared)")
        return
    for c in clocks:
        state = "locked" if c.locked else "free"
        label = f"  [{c.label}]" if c.label else ""
        print(f"   {c.freq_mhz:.6g} MHz  {state}{label}")


def _warn_if_stale(file_path: str) -> None:
    if stage5_fit_present(file_path):
        print(
            "\nNote: a Stage 5 fit already exists on this file and predates this "
            "declaration.\n      Re-run 'fit run' for the declaration to affect "
            "spur gating."
        )


def cmd_clocks_show(args: argparse.Namespace) -> int:
    setup_logging(args.verbose)
    try:
        clocks = get_clock_sources_impl(args.file_path)
        print(f"Declared clock sources for {args.file_path}:")
        _print_clocks(clocks)
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


def cmd_clocks_set(args: argparse.Namespace) -> int:
    setup_logging(args.verbose)
    try:
        sources: List[ClockSource] = [_parse_clock_token(t) for t in args.clocks]
        result = set_clock_sources_impl(args.file_path, sources, replace=True)
        print(f"Declared {len(result)} clock source(s):")
        _print_clocks(result)
        _warn_if_stale(args.file_path)
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


def cmd_clocks_add(args: argparse.Namespace) -> int:
    setup_logging(args.verbose)
    try:
        sources: List[ClockSource] = [_parse_clock_token(t) for t in args.clocks]
        result = set_clock_sources_impl(args.file_path, sources, replace=False)
        print(f"Clock sources now ({len(result)}):")
        _print_clocks(result)
        _warn_if_stale(args.file_path)
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


def cmd_clocks_remove(args: argparse.Namespace) -> int:
    setup_logging(args.verbose)
    try:
        freqs = [float(f) for f in args.freqs]
        result = remove_clock_sources_impl(args.file_path, freqs)
        print(f"Clock sources now ({len(result)}):")
        _print_clocks(result)
        _warn_if_stale(args.file_path)
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


def cmd_clocks_clear(args: argparse.Namespace) -> int:
    setup_logging(args.verbose)
    try:
        clear_clock_sources_impl(args.file_path)
        print("Cleared clock-source declaration.")
        _warn_if_stale(args.file_path)
        return 0
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


def register_clocks_commands(subparsers: Any) -> None:
    """Register the ``clocks`` object group and its verbs."""
    clocks = subparsers.add_parser(
        "clocks",
        help="Declare instrument clock sources for the Stage 5 spur gate",
        description=(
            "Declare the instrument clock fundamentals (e.g. 5760, not the "
            "11520 product) the Stage 5 spur gate builds its lattice prior "
            "from. Writes the recommended declaration layer, so an explicit "
            "fit-time --clocks and persisted Stage 5 settings still outrank it."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show the current declaration
  ftmwpipeline clocks show exp.ftmw

  # Replace the declaration (token: freq[:locked|free[:label]])
  ftmwpipeline clocks set exp.ftmw 5760:locked:synth 6250:free:digitizer

  # Append one source
  ftmwpipeline clocks add exp.ftmw 16000:locked:awg

  # Remove by frequency, or clear all
  ftmwpipeline clocks remove exp.ftmw 6250
  ftmwpipeline clocks clear exp.ftmw
        """,
    )
    clocks_sub = clocks.add_subparsers(dest="clocks_command", help="clocks subcommands")

    p_show = clocks_sub.add_parser("show", help="Show the declared clock sources")
    p_show.add_argument("file_path", help="Path to the .ftmw experiment")
    p_show.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    p_show.set_defaults(func=cmd_clocks_show)

    p_set = clocks_sub.add_parser(
        "set", help="Replace the declaration with the given clock sources"
    )
    p_set.add_argument("file_path", help="Path to the .ftmw experiment")
    p_set.add_argument(
        "clocks",
        nargs="+",
        metavar="freq[:locked|free[:label]]",
        help="Clock fundamental(s) in MHz, e.g. 5760:locked:synth",
    )
    p_set.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    p_set.set_defaults(func=cmd_clocks_set)

    p_add = clocks_sub.add_parser(
        "add", help="Append clock sources to the current declaration"
    )
    p_add.add_argument("file_path", help="Path to the .ftmw experiment")
    p_add.add_argument(
        "clocks",
        nargs="+",
        metavar="freq[:locked|free[:label]]",
        help="Clock fundamental(s) in MHz to append",
    )
    p_add.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    p_add.set_defaults(func=cmd_clocks_add)

    p_remove = clocks_sub.add_parser(
        "remove", help="Remove declared clock sources by frequency (MHz)"
    )
    p_remove.add_argument("file_path", help="Path to the .ftmw experiment")
    p_remove.add_argument(
        "freqs", nargs="+", metavar="FREQ_MHZ", help="Frequency(ies) to remove"
    )
    p_remove.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    p_remove.set_defaults(func=cmd_clocks_remove)

    p_clear = clocks_sub.add_parser("clear", help="Clear the clock-source declaration")
    p_clear.add_argument("file_path", help="Path to the .ftmw experiment")
    p_clear.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    p_clear.set_defaults(func=cmd_clocks_clear)

    def _clocks_help(args: argparse.Namespace) -> int:
        clocks.print_help()
        return 1

    clocks.set_defaults(func=_clocks_help)
