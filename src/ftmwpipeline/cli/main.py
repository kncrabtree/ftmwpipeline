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



def cmd_validate(args) -> int:
    """Validate installation."""
    results = validate_installation()
    
    print("ftmwpipeline installation validation:")
    print("-" * 40)
    
    all_good = True
    for component, status in results.items():
        status_str = "✅ OK" if status else "❌ FAILED"
        print(f"{component:20s}: {status_str}")
        if not status:
            all_good = False
    
    print("-" * 40)
    if all_good:
        print("✅ All components working correctly!")
        return 0
    else:
        print("❌ Some components have issues")
        return 1


def cmd_version(args) -> int:
    """Show version information."""
    from .. import PACKAGE_INFO
    
    print(f"ftmwpipeline {__version__}")
    print(f"Description: {PACKAGE_INFO['description']}")
    print(f"Optional dependencies:")
    print(f"  matplotlib: {'✅' if PACKAGE_INFO['has_matplotlib'] else '❌'}")
    print(f"  plotly: {'✅' if PACKAGE_INFO['has_plotly'] else '❌'}")
    
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog='ftmwpipeline',
        description='FTMW spectroscopy signal processing and peak fitting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Commands:
  Data Loading (Stage 0):
    data-load       Load experimental data from various formats
    data-visualize  Visualize loaded FID data for validation
    data-info       Show information about data formats
    
  FT Processing (Stage 1):
    ft-process      Process FID data and compute Fourier Transform
    ft-visualize    Visualize cached FT spectrum results
    
  Utility Commands:
    validate        Check installation and dependencies
    version         Show version and package information

Examples:
  # Stage-based pipeline workflow
  ftmwpipeline data-load exp_2638 --source examples/blackchirp_data/2638/
  ftmwpipeline data-visualize exp_2638
  ftmwpipeline ft-process exp_2638 --from-cache --zpf 2 --expf_us 3.0
  ftmwpipeline ft-visualize exp_2638
  
  # Utility commands
  ftmwpipeline validate
  ftmwpipeline version

Planned Commands:
  noise-estimate  Estimate baseline noise from cached FT data
  noise-visualize Visualize noise estimation results
  cache           Manage pipeline cache files and storage
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