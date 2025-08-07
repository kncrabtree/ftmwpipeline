"""
FT processing and visualization commands.

This module implements the ft-process and ft-visualize subcommands
for basic FTMW data processing and visualization.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

from .utils import (
    setup_logging, 
    parse_frequency_range, 
    validate_cache_dir,
    print_error,
    print_processing_params
)

# Import existing APIs
from ..io.fid_serialization import load_fid_cache
from ..io.result_serialization import save_pipeline_cache, load_pipeline_cache
from ..visualization.spectrum_visualization import plot_complex_ft_from_cache


def cmd_ft_process(args) -> int:
    """
    Process cached FID data and compute Fourier Transform.
    
    This command loads FID data from Stage 0 cache, applies FT processing
    with specified parameters, and caches the ComplexFT results for later use.
    
    Stage-based workflow:
    1. Load FID data from Stage 0 cache (data-load command output)
    2. Apply zero padding and exponential filtering
    3. Compute complex Fourier Transform
    4. Optionally trim to specified frequency range
    5. Cache ComplexFT results for visualization and analysis
    
    Future integration points:
    - Batch processing mode with parameter files
    - Interactive parameter selection
    - Configuration file support for default parameters
    - Progress tracking for large datasets
    """
    setup_logging(args.verbose)
    
    try:
        # Validate inputs
        cache_dir = validate_cache_dir(args.cache_dir)
        
        # Parse optional trim range
        trim_range = None
        if args.trim:
            try:
                trim_range = parse_frequency_range(args.trim)
            except ValueError as e:
                print_error(f"Invalid trim range: {e}")
                return 1
        
        print(f"Processing experiment {args.experiment_id} from Stage 0 cache")
        print_processing_params(args.zpf, args.expf_us, trim_range)
        print()
        
        # Load FID data from Stage 0 cache
        print("Loading FID data from cache...")
        try:
            fid = load_fid_cache(args.experiment_id, str(cache_dir))
        except FileNotFoundError:
            print_error(f"No FID data cached for experiment '{args.experiment_id}'")
            print("")
            print("Stage 0 (Data Loading) must be completed before FT processing.")
            print(f"Run: ftmwpipeline data-load {args.experiment_id} --source <path>")
            print("")
            print("For example:")
            print(f"  ftmwpipeline data-load {args.experiment_id} --source examples/blackchirp_data/2638/")
            print(f"  ftmwpipeline ft-process {args.experiment_id}")
            return 1
        except Exception as e:
            print_error(f"Failed to load FID from cache: {e}")
            print("The FID cache file may be corrupted or in an incompatible format.")
            print(f"Try rerunning: ftmwpipeline data-load {args.experiment_id} --source <path>")
            return 1
        
        print(f"Loaded FID with {len(fid.data)} points from cache")
        
        # Apply FT processing
        print("Computing Fourier Transform...")
        try:
            complex_ft = fid.ft(zpf=args.zpf, expf_us=args.expf_us)
        except Exception as e:
            print_error(f"Failed to compute FT: {e}")
            return 1
        
        print(f"FT computed: {len(complex_ft.complex_spectrum)} frequency points")
        print(f"Frequency range: {complex_ft.freq_array[0]:.1f}-{complex_ft.freq_array[-1]:.1f} MHz")
        
        # Apply frequency trimming if requested
        if trim_range:
            print(f"Trimming to {trim_range[0]:.1f}-{trim_range[1]:.1f} MHz...")
            try:
                complex_ft = complex_ft.trim_to_range(trim_range[0], trim_range[1])
                print(f"Trimmed spectrum: {len(complex_ft.complex_spectrum)} points")
            except Exception as e:
                print_error(f"Failed to trim spectrum: {e}")
                return 1
        
        # Cache results
        cache_file = cache_dir / f"{args.experiment_id}_cache.h5"
        print(f"Caching results to {cache_file}...")
        try:
            save_pipeline_cache(
                experiment_id=args.experiment_id,
                complex_ft=complex_ft,
                noise_result=None,  # Not computed yet
                cache_dir=str(cache_dir)
            )
        except Exception as e:
            print_error(f"Failed to save cache: {e}")
            return 1
        
        print("✅ FT processing completed successfully!")
        print(f"Results cached in: {cache_file}")
        print(f"Use 'ftmwpipeline ft-visualize {args.experiment_id}' to view the spectrum")
        
        return 0
        
    except KeyboardInterrupt:
        print_error("Processing interrupted by user", 130)
        return 130
    except Exception as e:
        print_error(f"Unexpected error during processing: {e}")
        return 1


def cmd_ft_visualize(args) -> int:
    """
    Visualize cached FT results.
    
    This command loads cached ComplexFT data and creates an interactive
    plot showing magnitude and real/imaginary components.
    
    Future integration points:
    - Multiple plot backends (matplotlib, plotly, bokeh)
    - Customizable plot layouts and styling
    - Export to various image formats
    - Batch plot generation from config files
    - Overlay multiple experiments for comparison
    """
    setup_logging(args.verbose)
    
    try:
        # Validate cache directory
        cache_dir = validate_cache_dir(args.cache_dir)
        cache_file = cache_dir / f"{args.experiment_id}_cache.h5"
        
        if not cache_file.exists():
            print_error(f"Cache file not found: {cache_file}")
            print(f"Run 'ftmwpipeline ft-process {args.experiment_id}' first")
            return 1
        
        print(f"Loading cached data for {args.experiment_id}...")
        
        # Load cached results
        try:
            pipeline_cache = load_pipeline_cache(args.experiment_id, str(cache_dir))
        except Exception as e:
            print_error(f"Failed to load cache: {e}")
            return 1
        
        if pipeline_cache.get('complex_ft') is None:
            print_error("No ComplexFT data found in cache")
            return 1
        
        complex_ft = pipeline_cache['complex_ft']
        print(f"Loaded spectrum: {len(complex_ft.complex_spectrum)} points")
        print(f"Frequency range: {complex_ft.freq_array[0]:.1f}-{complex_ft.freq_array[-1]:.1f} MHz")
        
        # Generate plot
        print("Creating spectrum plot...")
        try:
            if args.no_interactive:
                # Note: Current visualization API doesn't support direct save_path
                # This is a placeholder for future enhancement
                output_file = args.output or f"{args.experiment_id}_spectrum.png"
                
                print("Note: Non-interactive plot saving not yet implemented in visualization API")
                print("      For now, showing plot with matplotlib backend...")
                
                # Use the cache-based plotting function with matplotlib backend
                plot_complex_ft_from_cache(
                    experiment_id=args.experiment_id,
                    cache_dir=str(cache_dir),
                    title=f"Experiment {args.experiment_id} - FT Spectrum",
                    backend='matplotlib',
                    interactive=True  # Keep interactive for now
                )
                
                print("✅ Plot displayed (save functionality will be added in future iterations)")
            else:
                # Show interactive plot
                plot_complex_ft_from_cache(
                    experiment_id=args.experiment_id,
                    cache_dir=str(cache_dir),
                    title=f"Experiment {args.experiment_id} - FT Spectrum",
                    backend='plotly',  # Default to plotly for interactivity
                    interactive=True
                )
                
                print("✅ Interactive plot displayed")
                
        except Exception as e:
            print_error(f"Failed to generate plot: {e}")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        print_error("Visualization interrupted by user", 130)
        return 130
    except Exception as e:
        print_error(f"Unexpected error during visualization: {e}")
        return 1


def add_ft_subcommands(subparsers) -> None:
    """
    Add FT processing subcommands to the argument parser.
    
    Future integration points:
    - Add interactive mode flags for parameter prompts
    - Add batch processing options with config file support
    - Add advanced parameter validation
    - Add pipeline stage selection options
    """
    
    # ft-process command
    ft_process_parser = subparsers.add_parser(
        'ft-process',
        help='Process cached FID data and compute Fourier Transform',
        description='Load FID data from Stage 0 cache, apply FT processing, and cache ComplexFT results.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stage-based workflow (recommended)
  ftmwpipeline data-load exp_2638 --source examples/blackchirp_data/2638/
  ftmwpipeline ft-process exp_2638 --zpf 1 --expf_us 5.0
  ftmwpipeline ft-process exp_2638 --zpf 2 --expf_us 10.0 --trim 26500:40000
        """
    )
    
    ft_process_parser.add_argument(
        'experiment_id',
        help='Identifier for the cached experiment (from data-load command)'
    )
    ft_process_parser.add_argument(
        '--zpf',
        type=int,
        default=1,
        help='Zero padding factor for improved frequency resolution (default: 1)'
    )
    ft_process_parser.add_argument(
        '--expf_us',
        type=float,
        default=5.0,
        help='Exponential filter in microseconds for sensitivity enhancement (default: 5.0)'
    )
    ft_process_parser.add_argument(
        '--trim',
        help='Frequency range to keep as "min:max" in MHz (e.g., "26500:40000")'
    )
    ft_process_parser.add_argument(
        '--cache-dir',
        default='cache/',
        help='Cache directory for results (default: cache/)'
    )
    ft_process_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    ft_process_parser.set_defaults(func=cmd_ft_process)
    
    # ft-visualize command  
    ft_visualize_parser = subparsers.add_parser(
        'ft-visualize',
        help='Visualize cached FT spectrum results',
        description='Load cached ComplexFT data and create interactive spectrum plots.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ftmwpipeline ft-visualize exp_2638                    # Interactive plot
  ftmwpipeline ft-visualize exp_2638 --no-interactive   # Save to PNG
  ftmwpipeline ft-visualize exp_2638 --output plot.png  # Save with custom name
        """
    )
    
    ft_visualize_parser.add_argument(
        'experiment_id',
        help='Identifier for the cached experiment data'
    )
    ft_visualize_parser.add_argument(
        '--cache-dir',
        default='cache/',
        help='Cache directory containing results (default: cache/)'
    )
    ft_visualize_parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='Save plot to image file instead of showing interactive plot'
    )
    ft_visualize_parser.add_argument(
        '--output',
        help='Output image file path (used with --no-interactive)'
    )
    ft_visualize_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    ft_visualize_parser.set_defaults(func=cmd_ft_visualize)