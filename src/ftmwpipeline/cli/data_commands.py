"""
CLI commands for Stage 0 data loading operations.

This module provides CLI commands for loading experimental data from various
formats and visualizing FID data for validation.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from ..io.data_loaders import detect_format, validate_source, load_fid, list_formats, get_format_info
from ..io.fid_serialization import save_fid_cache, load_fid_cache
from ..io.result_serialization import get_cache_info
from .utils import setup_logging


def cmd_data_load(args) -> int:
    """
    Load experimental data and cache as FID object.
    
    This command handles data loading from various experimental formats
    (BlackChirp, CSV, HDF5, etc.) and caches the resulting FID object
    for use in subsequent pipeline stages.
    """
    setup_logging(args.verbose)
    
    try:
        # Validate inputs
        if not args.experiment_id:
            print("❌ Error: experiment_id is required")
            return 1
        
        if not args.source:
            print("❌ Error: --source path is required")
            return 1
        
        source_path = Path(args.source)
        if not source_path.exists():
            print(f"❌ Error: Source path does not exist: {source_path}")
            return 1
        
        print(f"🔍 Loading data for experiment '{args.experiment_id}'")
        print(f"📁 Source: {source_path}")
        
        # Format detection or validation
        format_name = args.format
        if format_name is None:
            print("🔍 Auto-detecting data format...")
            format_name = detect_format(source_path)
            
            if format_name is None:
                print(f"❌ Error: Could not detect data format for: {source_path}")
                print(f"💡 Try specifying format explicitly with --format")
                print(f"   Available formats: {', '.join(list_formats())}")
                return 1
            else:
                print(f"✅ Detected format: {format_name}")
        else:
            print(f"🎯 Using specified format: {format_name}")
        
        # Validate source with detected/specified format
        print(f"🔍 Validating source with {format_name} loader...")
        validation = validate_source(source_path, format_name)
        
        if not validation['valid']:
            print(f"❌ Source validation failed:")
            for error in validation['errors']:
                print(f"   • {error}")
            return 1
        
        print("✅ Source validation passed")
        
        # Show detected metadata
        if validation.get('metadata'):
            print("\n📊 Detected metadata:")
            metadata = validation['metadata']
            if 'n_fids' in metadata:
                print(f"   Available FIDs: {metadata['n_fids']}")
            if 'probe_freq_mhz' in metadata:
                print(f"   Probe frequency: {metadata['probe_freq_mhz']:.3f} MHz")
            if 'spacing_us' in metadata:
                print(f"   Time spacing: {metadata['spacing_us']:.4f} μs")
            if 'duration_us' in metadata:
                print(f"   FID duration: {metadata['duration_us']:.1f} μs")
            if 'n_points' in metadata:
                print(f"   Data points: {metadata['n_points']:,}")
        
        # Prepare loading parameters
        load_params = {}
        
        # Handle format-specific parameters
        if format_name == 'blackchirp' and args.fid_index is not None:
            load_params['fid_index'] = args.fid_index
        elif format_name == 'csv':
            # CSV format requires explicit parameters
            if args.spacing_us is None:
                print("❌ Error: CSV format requires --spacing_us parameter")
                return 1
            if args.probe_freq_mhz is None:
                print("❌ Error: CSV format requires --probe_freq_mhz parameter")
                return 1
            load_params.update({
                'spacing_us': args.spacing_us,
                'probe_freq_mhz': args.probe_freq_mhz,
                'sideband': args.sideband or 'upper',
                'shots': args.shots or 1
            })
        
        # Load FID data
        print(f"\n⚡ Loading FID data...")
        try:
            fid = load_fid(source_path, format_name, **load_params)
            print(f"✅ FID data loaded successfully")
            print(f"   Data points: {fid.n_points:,}")
            print(f"   Duration: {fid.duration_us:.1f} μs")
            print(f"   Probe freq: {fid.probe_freq_mhz:.3f} MHz")
            print(f"   Sideband: {fid.sideband.value}")
            print(f"   Shots: {fid.shots}")
            
        except Exception as e:
            print(f"❌ Error loading FID data: {e}")
            return 1
        
        # Cache FID data
        print(f"\n💾 Caching FID data...")
        try:
            cache_file = save_fid_cache(args.experiment_id, fid, args.cache_dir)
            print(f"✅ FID cached successfully")
            print(f"📁 Cache file: {cache_file}")
            
            # Show cache information
            cache_size_mb = cache_file.stat().st_size / (1024 * 1024)
            print(f"💿 Cache size: {cache_size_mb:.2f} MB")
            
        except Exception as e:
            print(f"❌ Error caching FID data: {e}")
            return 1
        
        print(f"\n🎉 Data loading completed successfully!")
        print(f"💡 Next steps:")
        print(f"   • Visualize FID: ftmwpipeline data-visualize {args.experiment_id}")
        print(f"   • Process FT: ftmwpipeline ft-process {args.experiment_id} --from-cache")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


def cmd_data_visualize(args) -> int:
    """
    Visualize cached FID data for validation.
    
    This command loads FID data from cache and creates plots for
    data validation and quality assessment.
    """
    setup_logging(args.verbose)
    
    try:
        # Check if cache exists
        cache_info = get_cache_info(args.experiment_id, args.cache_dir)
        
        # Check for FID cache file
        fid_cache_file = Path(args.cache_dir) / f"{args.experiment_id}_fid.h5"
        if not fid_cache_file.exists():
            print(f"❌ Error: FID cache not found for experiment '{args.experiment_id}'")
            print(f"💡 Run data loading first: ftmwpipeline data-load {args.experiment_id} --source <path>")
            return 1
        
        # Load FID from cache
        print(f"📊 Loading FID data for visualization...")
        try:
            fid = load_fid_cache(args.experiment_id, args.cache_dir)
            print(f"✅ FID data loaded from cache")
        except Exception as e:
            print(f"❌ Error loading FID cache: {e}")
            return 1
        
        # Import visualization function
        try:
            from ..visualization.fid_visualization import plot_fid
        except ImportError:
            print("❌ Error: FID visualization not available")
            print("💡 This feature will be implemented in the visualization module")
            return 1
        
        # Create FID plot
        print(f"📊 Creating FID visualization...")
        try:
            fig = plot_fid(fid, 
                          show_metadata=args.show_metadata,
                          title=f"Experiment {args.experiment_id} - FID Data")
            
            if args.save:
                output_file = Path(f"{args.experiment_id}_fid.png")
                fig.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"💾 Plot saved: {output_file}")
            
            if not args.no_show:
                import matplotlib.pyplot as plt
                plt.show()
                
            print(f"✅ FID visualization completed")
            return 0
            
        except Exception as e:
            print(f"❌ Error creating visualization: {e}")
            # Fall back to basic info display
            print(f"\n📊 FID Data Summary:")
            print(f"   Data points: {fid.n_points:,}")
            print(f"   Duration: {fid.duration_us:.1f} μs")
            print(f"   Spacing: {fid.spacing * 1e6:.4f} μs")
            print(f"   Probe frequency: {fid.probe_freq_mhz:.3f} MHz")
            print(f"   Sideband: {fid.sideband.value}")
            print(f"   Shots: {fid.shots}")
            
            if args.show_metadata and fid.metadata:
                print(f"\n📋 Source Metadata:")
                for key, value in fid.metadata.items():
                    if isinstance(value, dict):
                        print(f"   {key}: {type(value).__name__} with {len(value)} items")
                    else:
                        print(f"   {key}: {value}")
            
            return 0
        
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
  # Load BlackChirp experiment (auto-detect format)
  ftmwpipeline data-load exp_2638 --source examples/blackchirp_data/2638/
  
  # Load BlackChirp with specific FID index
  ftmwpipeline data-load exp_2638 --source examples/blackchirp_data/2638/ --fid-index 1
  
  # Load CSV file (requires explicit parameters)  
  ftmwpipeline data-load exp_csv --source data.csv --format csv --spacing_us 0.02 --probe_freq_mhz 40960
  
  # Force specific format
  ftmwpipeline data-load exp_2638 --source examples/blackchirp_data/2638/ --format blackchirp
        """
    )
    
    load_parser.add_argument('experiment_id', help='Experiment identifier for caching')
    load_parser.add_argument('--source', required=True, help='Path to data source (file or directory)')
    load_parser.add_argument('--format', choices=['blackchirp', 'csv', 'hdf5'], 
                           help='Data format (auto-detected if not specified)')
    load_parser.add_argument('--cache-dir', default='cache', help='Cache directory (default: cache)')
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
  ftmwpipeline data-visualize exp_2638
  
  # Show metadata and save plot
  ftmwpipeline data-visualize exp_2638 --show-metadata --save
  
  # Non-interactive mode
  ftmwpipeline data-visualize exp_2638 --no-show --save
        """
    )
    
    viz_parser.add_argument('experiment_id', help='Experiment identifier')
    viz_parser.add_argument('--cache-dir', default='cache', help='Cache directory (default: cache)')
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