"""
Main entry point for ftmwpipeline CLI.

This module provides the unified command-line interface for FTMW processing tasks,
integrating existing validation/version commands with new FT processing commands.
"""

import argparse
import sys
from typing import List, Optional

from .. import __version__
from ..workflows import validate_installation

from .utils import setup_logging
from .ft_commands import add_ft_subcommands
from .data_commands import add_data_subcommands
from .noise_commands import register_noise_commands
from .peak_commands import register_peak_commands
from .window_commands import register_window_commands
from .info_commands import add_info_subcommand



def cmd_validate(args) -> int:
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


def cmd_version(args) -> int:
    """Show version information."""
    from .. import PACKAGE_INFO
    
    print(f"ftmwpipeline {__version__}")
    print(f"Description: {PACKAGE_INFO['description']}")
    print(f"Optional dependencies:")
    print(f"  matplotlib: {'available' if PACKAGE_INFO['has_matplotlib'] else 'missing'}")
    print(f"  plotly: {'available' if PACKAGE_INFO['has_plotly'] else 'missing'}")
    
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog='ftmwpipeline',
        description='FTMW spectroscopy signal processing and peak fitting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Commands:
  Stage 0 (Data import):
    import-data     Import experimental data, creating a .ftmw file
    visualize-data  Visualize imported FID data
    formats         List available data formats

  Stage 1 (FT processing):
    compute-ft      Compute the Fourier transform
    visualize-ft    Visualize the FT spectrum

  Stage 2 (Noise estimation):
    estimate-noise  Estimate frequency-dependent noise
    visualize-noise Visualize noise estimation

  Stage 3 (Peak detection):
    detect-peaks    Detect and classify peaks (two-pass)
    visualize-peaks Overlay classified peaks on the spectrum

  Stage 4 (Window assignment):
    assign-windows    Turn promoted peaks into a fit-window plan
    visualize-windows Overlay the window plan on the spectrum

  Utility:
    info            Show provenance and stage status for a .ftmw file
    validate        Check installation and dependencies
    version         Show version and package information

Examples:
  ftmwpipeline import-data exp_2638.ftmw --source examples/blackchirp_data/2638/
  ftmwpipeline compute-ft exp_2638.ftmw --zpf 2 --expf_us 5.0 --trim 26500:40000
  ftmwpipeline estimate-noise exp_2638.ftmw
  ftmwpipeline info exp_2638.ftmw --format json
        """
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version=f'ftmwpipeline {__version__}'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add data loading commands (Stage 0)
    add_data_subcommands(subparsers)
    
    # Add FT processing commands (Stage 1)
    add_ft_subcommands(subparsers)
    
    # Add noise estimation commands (Stage 2)
    register_noise_commands(subparsers)

    # Add peak detection commands (Stage 3)
    register_peak_commands(subparsers)

    # Add window assignment commands (Stage 4)
    register_window_commands(subparsers)

    # Pipeline-file info command
    add_info_subcommand(subparsers)

    # Validate command
    validate_parser = subparsers.add_parser(
        'validate',
        help='Validate installation and dependencies'
    )
    validate_parser.set_defaults(func=cmd_validate)
    
    # Version command
    version_parser = subparsers.add_parser(
        'version',
        help='Show version information and optional dependencies'
    )
    version_parser.set_defaults(func=cmd_version)
    
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    if not hasattr(args, 'func'):
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())