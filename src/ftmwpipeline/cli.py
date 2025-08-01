#!/usr/bin/env python3
"""
Command-line interface for ftmwpipeline.

This module provides a command-line interface for common FTMW processing tasks.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .pipeline import Pipeline
from .workflows import process_experiment, batch_process_experiments, validate_installation


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def cmd_process(args) -> int:
    """Process a single experiment."""
    setup_logging(args.verbose)
    
    try:
        results = process_experiment(
            args.input, 
            output_dir=args.output_dir
        )
        
        print(f"Processing completed: {results['status']}")
        if results['status'] == 'not_implemented':
            print("Note: Actual processing will be implemented in later phases")
            return 0
        
        return 0 if results['status'] == 'success' else 1
        
    except Exception as e:
        print(f"Error processing {args.input}: {e}", file=sys.stderr)
        return 1


def cmd_batch(args) -> int:
    """Process multiple experiments."""
    setup_logging(args.verbose)
    
    # Read input files
    if args.file_list:
        with open(args.file_list, 'r') as f:
            input_files = [line.strip() for line in f if line.strip()]
    else:
        input_files = args.inputs
    
    if not input_files:
        print("No input files specified", file=sys.stderr)
        return 1
    
    try:
        results = batch_process_experiments(
            input_files,
            output_dir=args.output_dir,
            parallel=args.parallel
        )
        
        success_count = sum(1 for r in results if r.get('status') == 'success')
        print(f"Processed {len(results)} experiments, {success_count} successful")
        
        return 0 if success_count == len(results) else 1
        
    except Exception as e:
        print(f"Error in batch processing: {e}", file=sys.stderr)
        return 1


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
    from . import PACKAGE_INFO
    
    print(f"ftmwpipeline {__version__}")
    print(f"Description: {PACKAGE_INFO['description']}")
    print(f"Optional dependencies:")
    print(f"  matplotlib: {'✅' if PACKAGE_INFO['has_matplotlib'] else '❌'}")
    print(f"  plotly: {'✅' if PACKAGE_INFO['has_plotly'] else '❌'}")
    
    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog='ftmwpipeline',
        description='FTMW spectroscopy signal processing and peak fitting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ftmwpipeline process data.h5                    # Process single file
  ftmwpipeline batch *.h5 -o results/             # Batch process
  ftmwpipeline validate                           # Check installation
  ftmwpipeline version                            # Show version info
        """
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version=f'ftmwpipeline {__version__}'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser(
        'process', 
        help='Process a single FTMW experiment'
    )
    process_parser.add_argument(
        'input',
        help='Input data file path'
    )
    process_parser.add_argument(
        '-o', '--output-dir',
        help='Output directory for results'
    )
    process_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    process_parser.set_defaults(func=cmd_process)
    
    # Batch command
    batch_parser = subparsers.add_parser(
        'batch',
        help='Process multiple FTMW experiments'
    )
    batch_parser.add_argument(
        'inputs',
        nargs='*',
        help='Input data files'
    )
    batch_parser.add_argument(
        '-f', '--file-list',
        help='File containing list of input files'
    )
    batch_parser.add_argument(
        '-o', '--output-dir',
        help='Output directory for results'
    )
    batch_parser.add_argument(
        '-p', '--parallel',
        action='store_true',
        help='Process files in parallel'
    )
    batch_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    batch_parser.set_defaults(func=cmd_batch)
    
    # Validate command
    validate_parser = subparsers.add_parser(
        'validate',
        help='Validate installation'
    )
    validate_parser.set_defaults(func=cmd_validate)
    
    # Version command
    version_parser = subparsers.add_parser(
        'version',
        help='Show version information'
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