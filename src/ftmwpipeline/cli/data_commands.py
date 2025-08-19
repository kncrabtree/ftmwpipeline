"""
CLI commands for Stage 0 data loading operations.

This module provides CLI commands for loading experimental data from various
formats and visualizing FID data for validation.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from ..io.data_loaders import list_formats, get_format_info
from .._internal.stage0_impl import import_data_impl, load_fid_from_pipeline_impl, visualize_fid_impl, get_pipeline_info_impl
from .utils import setup_logging


def cmd_data_load(args) -> int:
    """
    Import experimental data into a .ftmw pipeline file.
    
    This command handles data loading from various experimental formats
    (BlackChirp, CSV, HDF5, etc.) and creates a .ftmw pipeline file
    for use in subsequent pipeline stages.
    """
    setup_logging(args.verbose)
    
    try:
        # Validate inputs
        if not args.file_path:
            print("❌ Error: file_path is required")
            return 1
        
        if not args.source:
            print("❌ Error: --source path is required")
            return 1
        
        file_path = args.file_path
        if not file_path.endswith('.ftmw'):
            file_path = file_path + '.ftmw'
        
        print(f"🔍 Importing data into pipeline file '{file_path}'")
        print(f"📁 Source: {args.source}")
        
        # Prepare loading parameters
        format_params = {}
        
        # Handle format-specific parameters
        if args.format == 'blackchirp' and args.fid_index is not None:
            format_params['fid_index'] = args.fid_index
        elif args.format == 'csv':
            # CSV format requires explicit parameters
            if args.spacing_us is None:
                print("❌ Error: CSV format requires --spacing_us parameter")
                return 1
            if args.probe_freq_mhz is None:
                print("❌ Error: CSV format requires --probe_freq_mhz parameter")
                return 1
            format_params.update({
                'spacing_us': args.spacing_us,
                'probe_freq_mhz': args.probe_freq_mhz,
                'sideband': args.sideband or 'upper',
                'shots': args.shots or 1
            })
        
        # Use shared implementation for data import
        result = import_data_impl(
            file_path=file_path,
            source=args.source,
            format_name=args.format,
            **format_params
        )
        
        # Display results
        print(f"✅ Data import completed successfully!")
        print(f"📁 Pipeline file: {result['pipeline_file']}")
        print(f"📊 Source format: {result['format_name']}")
        
        # Show FID metadata
        fid_info = result['fid_metadata']
        print(f"📊 FID Information:")
        print(f"   Data points: {fid_info['n_points']:,}")
        print(f"   Duration: {fid_info['duration_us']:.1f} μs")
        print(f"   Probe freq: {fid_info['probe_freq_mhz']:.3f} MHz")
        print(f"   Sideband: {fid_info['sideband']}")
        print(f"   Shots: {fid_info['shots']:,}")
        
        # Show file size
        pipeline_file = Path(result['pipeline_file'])
        file_size_mb = pipeline_file.stat().st_size / (1024 * 1024)
        print(f"💿 File size: {file_size_mb:.2f} MB")
        
        print(f"\n💡 Next steps:")
        print(f"   • Visualize FID: ftmwpipeline data-visualize {file_path}")
        print(f"   • Process FT: ftmwpipeline ft-process {file_path}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


def cmd_data_visualize(args) -> int:
    """
    Visualize FID data from pipeline file.
    
    This command loads FID data from a .ftmw pipeline file and creates plots for
    data validation and quality assessment.
    """
    setup_logging(args.verbose)
    
    try:
        file_path = args.file_path
        if not file_path.endswith('.ftmw'):
            file_path = file_path + '.ftmw'
        
        print(f"📊 Visualizing FID data from '{file_path}'...")
        
        # Use shared implementation for FID visualization
        try:
            fig = visualize_fid_impl(
                file_path=file_path,
                show_metadata=args.show_metadata,
                title=f"Pipeline {Path(file_path).stem} - FID Data"
            )
            
            if args.save:
                output_file = Path(f"{Path(file_path).stem}_fid.png")
                fig.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"💾 Plot saved: {output_file}")
            
            if not args.no_show:
                import matplotlib.pyplot as plt
                plt.show()
                
            print(f"✅ FID visualization completed")
            return 0
            
        except Exception as e:
            print(f"❌ Error creating visualization: {e}")
            # Fall back to basic info display using pipeline info
            try:
                file_path_obj, source_metadata, stage_tracker, fid = get_pipeline_info_impl(file_path)
                
                print(f"\n📊 FID Data Summary:")
                print(f"   Data points: {fid.n_points:,}")
                print(f"   Duration: {fid.duration_us:.1f} μs")
                print(f"   Spacing: {fid.spacing:.4e} s")
                print(f"   Probe frequency: {fid.probe_freq_mhz:.3f} MHz")
                print(f"   Sideband: {fid.sideband.value}")
                print(f"   Shots: {fid.shots}")
                
                if args.show_metadata:
                    print(f"\n📋 Source Metadata:")
                    print(f"   Source path: {source_metadata.source_path}")
                    print(f"   Format: {source_metadata.format_name}")
                    print(f"   Import time: {source_metadata.import_timestamp}")
                    if source_metadata.loader_parameters:
                        print(f"   Loader parameters: {source_metadata.loader_parameters}")
                
                return 0
            except Exception as info_error:
                print(f"❌ Error getting pipeline info: {info_error}")
                return 1
        
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


def cmd_data_info(args) -> int:
    """
    Show information about available data formats and loaders.
    """
    setup_logging(args.verbose)
    
    try:
        if args.format:
            # Show info about specific format
            try:
                info = get_format_info(args.format)
                print(f"📊 Format Information: {args.format}")
                print(f"   Loader class: {info['loader_class']}")
                
                if info['file_extensions']:
                    print(f"   File extensions: {', '.join(info['file_extensions'])}")
                
                if info['directory_indicators']:
                    print(f"   Directory indicators: {', '.join(info['directory_indicators'])}")
                
                if info['required_parameters']:
                    print(f"   Required parameters: {', '.join(info['required_parameters'])}")
                else:
                    print(f"   Required parameters: None")
                
                if info['optional_parameters']:
                    print(f"   Optional parameters:")
                    for param, default in info['optional_parameters'].items():
                        print(f"     • {param} (default: {default})")
                
            except ValueError as e:
                print(f"❌ Error: {e}")
                return 1
        else:
            # Show all available formats
            formats = list_formats()
            print(f"📊 Available Data Formats ({len(formats)}):")
            print()
            
            for fmt in formats:
                try:
                    info = get_format_info(fmt)
                    print(f"🔹 {fmt}")
                    print(f"   Loader: {info['loader_class']}")
                    
                    if info['file_extensions']:
                        print(f"   Extensions: {', '.join(info['file_extensions'])}")
                    elif info['directory_indicators']:
                        print(f"   Directory format (indicators: {', '.join(info['directory_indicators'])})")
                    
                    if info['required_parameters']:
                        print(f"   Requires: {', '.join(info['required_parameters'])}")
                    
                    print()
                except Exception:
                    print(f"🔹 {fmt} (info unavailable)")
                    print()
        
        return 0
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


def add_data_subcommands(subparsers: argparse._SubParsersAction) -> None:
    """
    Add data loading subcommands to the argument parser.
    
    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Subparser object to add commands to
    """
    
    # data-load command
    load_parser = subparsers.add_parser(
        'data-load',
        help='Load experimental data from various formats',
        description='Load FTMW experimental data and cache as FID object for pipeline processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import BlackChirp experiment (auto-detect format)
  ftmwpipeline data-load exp_2638.ftmw --source examples/blackchirp_data/2638/
  
  # Import BlackChirp with specific FID index
  ftmwpipeline data-load exp_2638.ftmw --source examples/blackchirp_data/2638/ --fid-index 1
  
  # Import CSV file (requires explicit parameters)  
  ftmwpipeline data-load exp_csv.ftmw --source data.csv --format csv --spacing_us 0.02 --probe_freq_mhz 40960
  
  # Force specific format
  ftmwpipeline data-load exp_2638.ftmw --source examples/blackchirp_data/2638/ --format blackchirp
        """
    )
    
    load_parser.add_argument('file_path', help='Path to .ftmw pipeline file to create')
    load_parser.add_argument('--source', required=True, help='Path to data source (file or directory)')
    load_parser.add_argument('--format', choices=['blackchirp', 'csv', 'hdf5'], 
                           help='Data format (auto-detected if not specified)')
    load_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # BlackChirp-specific options
    load_parser.add_argument('--fid-index', type=int, help='FID index for BlackChirp format (default: 0)')
    
    # CSV-specific options
    load_parser.add_argument('--spacing_us', type=float, help='Time spacing in μs (required for CSV)')
    load_parser.add_argument('--probe_freq_mhz', type=float, help='Probe frequency in MHz (required for CSV)')
    load_parser.add_argument('--sideband', choices=['upper', 'lower'], help='Sideband for CSV (default: upper)')
    load_parser.add_argument('--shots', type=int, help='Number of shots for CSV (default: 1)')
    
    load_parser.set_defaults(func=cmd_data_load)
    
    # data-visualize command  
    viz_parser = subparsers.add_parser(
        'data-visualize',
        help='Visualize cached FID data',
        description='Load FID data from cache and create validation plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic FID visualization
  ftmwpipeline data-visualize exp_2638.ftmw
  
  # Show metadata and save plot
  ftmwpipeline data-visualize exp_2638.ftmw --show-metadata --save
  
  # Non-interactive mode
  ftmwpipeline data-visualize exp_2638.ftmw --no-show --save
        """
    )
    
    viz_parser.add_argument('file_path', help='Path to .ftmw pipeline file')
    viz_parser.add_argument('--show-metadata', action='store_true', help='Display metadata information')
    viz_parser.add_argument('--save', action='store_true', help='Save plot to file')
    viz_parser.add_argument('--no-show', action='store_true', help='Do not display plot interactively')
    viz_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    viz_parser.set_defaults(func=cmd_data_visualize)
    
    # data-info command
    info_parser = subparsers.add_parser(
        'data-info',
        help='Show information about data formats',
        description='Display information about available data format loaders',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available formats
  ftmwpipeline data-info
  
  # Show details about specific format
  ftmwpipeline data-info --format blackchirp
        """
    )
    
    info_parser.add_argument('--format', help='Show details about specific format')
    info_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    info_parser.set_defaults(func=cmd_data_info)