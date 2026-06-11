"""
Main entry point for ftmwpipeline CLI.

This module provides the unified command-line interface for FTMW processing tasks,
integrating validation/version commands with FT processing commands.
"""

import argparse
import sys
from typing import List, Optional, cast

from .. import __version__
from ..workflows import validate_installation
from .data_commands import add_data_subcommands
from .fitting_commands import register_fitting_commands
from .ft_commands import add_ft_subcommands
from .info_commands import add_info_subcommand
from .noise_commands import register_noise_commands
from .peak_commands import register_peak_commands
from .scan_commands import register_scan_commands
from .settings_commands import register_settings_commands
from .start_commands import register_start_commands
from .tau_commands import register_tau_commands
from .timebase_commands import register_timebase_commands
from .utils import setup_logging
from .window_commands import register_window_commands


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate installation."""
    results = validate_installation()

    print("ftmwpipeline installation validation:")
    print("-" * 40)

    all_good = True
    for component, status in results.items():
        status_str = "OK" if status else "FAILED"
        print(f"{component:20s}: {status_str}")
        if not status:
            all_good = False

    print("-" * 40)
    if all_good:
        print("All components working correctly!")
        return 0
    else:
        print("Some components have issues")
        return 1


def cmd_version(args: argparse.Namespace) -> int:
    """Show version information."""
    from .. import PACKAGE_INFO

    print(f"ftmwpipeline {__version__}")
    print(f"Description: {PACKAGE_INFO['description']}")
    print(f"Optional dependencies:")
    print(
        f"  matplotlib: {'available' if PACKAGE_INFO['has_matplotlib'] else 'missing'}"
    )
    print(f"  plotly: {'available' if PACKAGE_INFO['has_plotly'] else 'missing'}")

    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="ftmwpipeline",
        description="FTMW spectroscopy signal processing and peak fitting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Object-verb grammar: ftmwpipeline <object> <verb> <file.ftmw> [options]
Every stage object accepts 'run' and 'show'; the stageN synonym is
interchangeable with the name (ft run == stage1 run).

Stage objects (synonym):
  data (stage0)     import <src> | show   Import experimental data / view FID
  start             run | show            Detect/stamp start_us / sweep diagnostic
  ft (stage1)       run | show            Compute / visualize the Fourier transform
  noise (stage2)    run | show            Estimate / visualize frequency-dependent noise
  tau (stage2b)     run [--gaussian]      STFT tau calibration (--gaussian: tau_G twin)
                    show --kind heatmap|distribution
  timebase          run | show            Scope-clock scale-error (eps) self-calibration
  peaks (stage3)    run | show            Detect/classify peaks / overlay them
  windows (stage4)  run | show            Plan fit windows / overlay the plan
  fit (stage5)      run | show | check    Fit lines / overlay / SNR-aware assessment

Meta objects (cross-cutting, optional dotted selector):
  scan      list | run | all              Knob registry; sweep one / all knobs
  settings  show | set | export           Resolved value + provenance; persist; preset

Utility (bare commands):
  formats           List available data formats
  info              Show provenance and stage status for a .ftmw file
  validate          Check installation and dependencies
  version           Show version and package information

Examples:
  ftmwpipeline data import exp_2638.ftmw examples/blackchirp_data/2638/
  ftmwpipeline ft run exp_2638.ftmw --trim 26500:40000
  ftmwpipeline noise run exp_2638.ftmw
  ftmwpipeline info exp_2638.ftmw --format json
        """,
    )

    parser.add_argument(
        "--version", action="version", version=f"ftmwpipeline {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add data loading commands (Stage 0)
    add_data_subcommands(subparsers)

    # Add start-time detection commands (pre-Stage 1)
    register_start_commands(subparsers)

    # Add FT processing commands (Stage 1)
    add_ft_subcommands(subparsers)

    # Add noise estimation commands (Stage 2)
    register_noise_commands(subparsers)

    # Add tau calibration commands (Stage 2b)
    register_tau_commands(subparsers)

    # Add scope-timebase self-calibration commands
    register_timebase_commands(subparsers)

    # Add peak detection commands (Stage 3)
    register_peak_commands(subparsers)

    # Add window assignment commands (Stage 4)
    register_window_commands(subparsers)

    # Add fitting commands (Stage 5)
    register_fitting_commands(subparsers)

    # Cross-cutting parameter-scan surface
    register_scan_commands(subparsers)

    # Cross-cutting resolved-settings inspection
    register_settings_commands(subparsers)

    # Pipeline-file info command
    add_info_subcommand(subparsers)

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate installation and dependencies"
    )
    validate_parser.set_defaults(func=cmd_validate)

    # Version command
    version_parser = subparsers.add_parser(
        "version", help="Show version information and optional dependencies"
    )
    version_parser.set_defaults(func=cmd_version)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    return cast(int, args.func(args))


if __name__ == "__main__":
    sys.exit(main())
