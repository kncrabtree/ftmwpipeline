"""
FT processing and visualization commands.

This module implements the ft-process and ft-visualize subcommands
for basic FTMW data processing and visualization.
"""

import argparse
import sys
import numpy as np
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
from ..io.fid_serialization import load_fid_cache, update_fid_processing_defaults
from ..io.result_serialization import save_pipeline_cache, load_pipeline_cache
from ..visualization.spectrum_visualization import plot_complex_ft


def cmd_ft_process(args) -> int:
    """
    Validate and store user-provided FT processing settings for analysis.
    
    This command loads FID data from Stage 0 cache, validates FT processing
    parameters, and provides detailed feedback about the processing steps.
    Designed for power users and automated pipeline processes who want to
    validate specific parameter combinations.
    
    Use this command to:
    - Validate processing parameters with immediate feedback
    - Test parameter combinations for optimal results
    - Get detailed processing statistics and information
    - Programmatically validate parameters in automated workflows
    
    For interactive parameter exploration with visualization, use ft-visualize.
    
    Stage-based workflow:
    1. Load FID data from Stage 0 cache (data-load command output)
    2. Test preprocessing (windowing, filtering, zero-padding)
    3. Test FFT computation and frequency range
    4. Provide detailed feedback without permanent storage
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
        
        # Test FT processing to validate parameters and provide feedback
        print("Testing FT processing parameters...")
        try:
            # Stage 1: FID Preprocessing  
            preprocessed_fid = fid.preprocess(
                start_us=getattr(args, 'start_us', None),
                end_us=getattr(args, 'end_us', None), 
                zpf=args.zpf, 
                expf_us=args.expf_us,
                window_function=getattr(args, 'window_function', None),
                units_power=getattr(args, 'units_power', 6)
            )
            print(f"✓ Preprocessing complete: {len(preprocessed_fid.data):,} points (zero-padded)")
            
            # Stage 2: FFT Calculation
            complex_spectrum, freq_array = preprocessed_fid.compute_fft()
            print(f"✓ FFT computation complete: {len(complex_spectrum):,} frequency points")
            print(f"  Frequency range: {freq_array[0]:.1f} - {freq_array[-1]:.1f} MHz")
            
            # Stage 3: Test post-processing (trimming) if requested
            final_points = len(complex_spectrum)
            if trim_range:
                mask = (freq_array >= trim_range[0]) & (freq_array <= trim_range[1])
                if not np.any(mask):
                    print_error(f"No data points in trim range [{trim_range[0]:.1f}, {trim_range[1]:.1f}] MHz")
                    return 1
                final_points = np.sum(mask)
                print(f"✓ Trimming validated: {final_points:,} points in range [{trim_range[0]:.1f}, {trim_range[1]:.1f}] MHz")
            
        except Exception as e:
            print_error(f"Failed to process FT: {e}")
            return 1
        
        print()
        print("✅ FT processing completed successfully!")
        print(f"   Processing parameters validated with {len(fid.data):,} FID points")
        print(f"   Final spectrum: {final_points:,} frequency points")
        print()
        print("📌 ComplexFT will be calculated on-demand during visualization")
        print(f"   Use: ftmwpipeline ft-visualize {args.experiment_id}")
        if trim_range:
            print(f"        ftmwpipeline ft-visualize {args.experiment_id} --trim {trim_range[0]:.0f}:{trim_range[1]:.0f}")
        print()
        print("💾 FID data remains cached for reuse with different processing parameters")
        
        return 0
        
    except KeyboardInterrupt:
        print_error("Processing interrupted by user", 130)
        return 130
    except Exception as e:
        print_error(f"Unexpected error during processing: {e}")
        return 1


def cmd_ft_visualize(args) -> int:
    """
    Interactive parameter exploration powered by visualization.
    
    This is the companion tool to ft-process, designed for exploratory usage
    where users want to experiment with different processing parameters and
    see immediate visual feedback. Applies FT processing, creates interactive
    spectrum plots, and offers to save good parameter combinations as defaults.
    
    Use this command to:
    - Interactively explore different processing parameters with visual feedback
    - Find optimal parameters for your data through experimentation
    - Save complete parameter sets (preprocessing + postprocessing) as defaults
    - Create high-quality plots for publications/presentations
    
    Key features:
    - On-demand ComplexFT calculation (no permanent storage unless requested)
    - Interactive parameter persistence (save complete parameter sets as defaults)
    - Matplotlib backend for reliable CLI visualization
    - Support for static image export
    
    Workflow:
    1. Load FID data from Stage 0 cache
    2. Apply custom processing parameters (preprocessing + postprocessing)
    3. Display interactive spectrum plot
    4. Optionally save complete parameter set as defaults for this experiment
    """
    setup_logging(args.verbose)
    
    try:
        # Validate cache directory
        cache_dir = validate_cache_dir(args.cache_dir)
        
        # Parse optional trim range
        trim_range = None
        if args.trim:
            try:
                trim_range = parse_frequency_range(args.trim)
            except ValueError as e:
                print_error(f"Invalid trim range: {e}")
                return 1
        
        print(f"Visualizing experiment {args.experiment_id} with on-demand FT calculation")
        
        # Load FID data from Stage 0 cache
        print("Loading FID data from cache...")
        try:
            fid = load_fid_cache(args.experiment_id, str(cache_dir))
        except FileNotFoundError:
            print_error(f"No FID data cached for experiment '{args.experiment_id}'")
            print("")
            print("Stage 0 (Data Loading) must be completed before visualization.")
            print(f"Run: ftmwpipeline data-load {args.experiment_id} --source <path>")
            print("")
            return 1
        except Exception as e:
            print_error(f"Failed to load FID from cache: {e}")
            return 1
        
        print(f"Loaded FID with {len(fid.data):,} points from cache")
        
        # Get processing parameters (merge user input with defaults)
        processing_params = {
            'start_us': getattr(args, 'start_us', None),
            'end_us': getattr(args, 'end_us', None),
            'zpf': getattr(args, 'zpf', 1),
            'expf_us': getattr(args, 'expf_us', 5.0),
            'window_function': getattr(args, 'window_function', None),
            'units_power': getattr(args, 'units_power', 6)
        }
        
        # Filter out None values for cleaner display
        display_params = {k: v for k, v in processing_params.items() if v is not None}
        if display_params:
            print("Processing parameters:")
            for param, value in display_params.items():
                print(f"  {param}: {value}")
        
        # Calculate ComplexFT on-demand using separated stages
        print("Computing ComplexFT on-demand...")
        try:
            # Stage 1: FID Preprocessing
            preprocessed_fid = fid.preprocess(**processing_params)
            
            # Stage 2: FFT Calculation  
            complex_spectrum, freq_array = preprocessed_fid.compute_fft()
            
            # Stage 3: Post-processing (ComplexFT creation)
            from ..core.data_structures import ComplexFT
            complex_ft = ComplexFT.from_spectrum(
                complex_spectrum=complex_spectrum,
                freq_array=freq_array,
                metadata={'processing_params': preprocessed_fid.processing_params}
            )
            
            print(f"✓ ComplexFT calculated: {len(complex_ft.complex_spectrum):,} frequency points")
            print(f"  Frequency range: {complex_ft.freq_array[0]:.1f} - {complex_ft.freq_array[-1]:.1f} MHz")
            
        except Exception as e:
            print_error(f"Failed to compute ComplexFT: {e}")
            return 1
        
        # Apply frequency trimming if requested
        if trim_range:
            print(f"Trimming to {trim_range[0]:.1f} - {trim_range[1]:.1f} MHz...")
            try:
                complex_ft = complex_ft.trim_to_range(trim_range[0], trim_range[1])
                print(f"✓ Trimmed spectrum: {len(complex_ft.complex_spectrum):,} points")
            except Exception as e:
                print_error(f"Failed to trim spectrum: {e}")
                return 1
        
        # Generate plot
        print("Creating spectrum plot...")
        try:
            plot_title = f"Experiment {args.experiment_id} - FT Spectrum"
            if trim_range:
                plot_title += f" ({trim_range[0]:.0f}-{trim_range[1]:.0f} MHz)"
            
            # Always use matplotlib backend for CLI - it's more reliable than plotly for CLI usage
            if args.no_interactive:
                # Create static plot and save to file
                fig = plot_complex_ft(
                    complex_ft=complex_ft,
                    title=plot_title,
                    backend='matplotlib',
                    interactive=False
                )
                if args.output:
                    fig.savefig(args.output, dpi=150, bbox_inches='tight')
                    print(f"✅ Plot saved to: {args.output}")
                else:
                    # Save with default name
                    output_file = f"{args.experiment_id}_spectrum.png"
                    fig.savefig(output_file, dpi=150, bbox_inches='tight')
                    print(f"✅ Plot saved to: {output_file}")
                
                # Close the figure to free memory
                import matplotlib.pyplot as plt
                plt.close(fig)
            else:
                # Create interactive matplotlib plot
                fig = plot_complex_ft(
                    complex_ft=complex_ft,
                    title=plot_title,
                    backend='matplotlib',
                    interactive=True
                )
                # Show the interactive plot
                import matplotlib.pyplot as plt
                plt.show()
                print("✅ Interactive plot displayed")
                print("   Close the plot window to continue...")
                
        except Exception as e:
            print_error(f"Failed to generate plot: {e}")
            return 1
        
        # Implement interactive parameter persistence
        if not args.no_interactive:
            # Check if user provided custom parameters different from cache defaults
            cached_fid = load_fid_cache(args.experiment_id, str(cache_dir))
            default_params = {
                'start_us': cached_fid.processing.start_us,
                'end_us': cached_fid.processing.end_us,
                'zpf': cached_fid.processing.zpf,
                'expf_us': cached_fid.processing.expf_us,
                'window_function': cached_fid.processing.winf,
                'rdc': cached_fid.processing.rdc,
                'units_power': cached_fid.processing.units_power
            }
            
            # Determine which parameters were customized by user
            user_params = {
                'start_us': getattr(args, 'start_us', None),
                'end_us': getattr(args, 'end_us', None),
                'zpf': getattr(args, 'zpf', 1),
                'expf_us': getattr(args, 'expf_us', 5.0),
                'window_function': getattr(args, 'window_function', None),
                'rdc': True,  # Always true in current implementation
                'units_power': getattr(args, 'units_power', 6)
            }
            
            # Find parameters that differ from cached defaults
            custom_params = {}
            for param, user_value in user_params.items():
                default_value = default_params.get(param)
                if user_value != default_value and user_value is not None:
                    # Special handling for default values that might indicate customization
                    if param == 'zpf' and user_value != 1:  # zpf=1 is default
                        custom_params[param] = user_value
                    elif param == 'expf_us' and user_value != 5.0:  # expf_us=5.0 is default
                        custom_params[param] = user_value
                    elif param == 'units_power' and user_value != 6:  # units_power=6 is default
                        custom_params[param] = user_value
                    elif param in ['start_us', 'end_us', 'window_function'] and user_value is not None:
                        custom_params[param] = user_value
            
            # Also check trim parameters (include as post-processing setting)
            if trim_range:
                custom_params['trim_range'] = f"{trim_range[0]:.0f}:{trim_range[1]:.0f}"
            
            # Offer to save parameters if user provided custom values
            if custom_params:
                print()
                print("You used custom processing parameters:")
                for param, value in custom_params.items():
                    print(f"  {param}: {value}")
                print()
                
                try:
                    response = input("Save complete parameter set as defaults for future processing? (y/N): ").strip()
                    if response.lower().startswith('y'):
                        # Save all parameters (preprocessing + postprocessing)
                        # Handle trim_range specially for storage format
                        params_to_save = {k: v for k, v in custom_params.items()}
                        if 'trim_range' in params_to_save:
                            # Store trim settings in a format that can be used by future stages
                            trim_value = params_to_save.pop('trim_range')
                            params_to_save['default_trim_range'] = trim_value
                        
                        if params_to_save:
                            update_fid_processing_defaults(args.experiment_id, params_to_save, str(cache_dir))
                            print("✅ Complete parameter set saved as defaults for this experiment")
                            print("   Future pipeline stages will use these preprocessing and postprocessing parameters by default")
                        else:
                            print("No parameters to save")
                    else:
                        print("Parameters not saved - using for visualization only")
                except (EOFError, KeyboardInterrupt):
                    print("\nParameters not saved - using for visualization only")
        
        print()
        print("💡 ComplexFT calculated on-demand from cached FID data")
        print("   Try different parameters without permanent storage:")
        print(f"   ftmwpipeline ft-visualize {args.experiment_id} --zpf 2 --expf_us 3.0")
        if not trim_range:
            print(f"   ftmwpipeline ft-visualize {args.experiment_id} --trim 26500:40000")
        
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
        help='Validate and store user-provided FT processing settings for analysis',
        description='Power user tool to validate specific FT processing parameters and provide detailed feedback.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Purpose: Validate and store user-provided processing settings
Intended for: Power users and automated pipeline processes

Examples:
  # Validate basic processing parameters
  ftmwpipeline ft-process exp_2638 --zpf 1 --expf_us 5.0
  
  # Test parameter combinations with trimming
  ftmwpipeline ft-process exp_2638 --zpf 2 --expf_us 10.0 --trim 26500:40000
  
  # Test windowing and scaling parameters  
  ftmwpipeline ft-process exp_2638 --start-us 1.0 --end-us 10.0 --units-power 3

Workflow:
  1. ftmwpipeline data-load exp_2638 --source examples/blackchirp_data/2638/
  2. ftmwpipeline ft-process exp_2638 [--parameters]  # Power user validation
     OR
     ftmwpipeline ft-visualize exp_2638 [--parameters] # Interactive exploration
        """
    )
    
    ft_process_parser.add_argument(
        'experiment_id',
        help='Identifier for the cached experiment (from data-load command)'
    )
    ft_process_parser.add_argument(
        '--start-us',
        type=float,
        help='FID window start time in microseconds (earlier points set to 0)'
    )
    ft_process_parser.add_argument(
        '--end-us', 
        type=float,
        help='FID window end time in microseconds (later points set to 0)'
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
        '--window-function',
        help='Windowing function (hann, blackman, hamming, etc.)'
    )
    ft_process_parser.add_argument(
        '--units-power',
        type=int,
        default=6,
        help='Scaling factor for spectrum units as power of 10 (default: 6 for μV)'
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
        help='Interactive parameter exploration powered by visualization',
        description='Companion tool to ft-process for exploratory parameter discovery through visualization.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Purpose: Interactive parameter exploration powered by visualization
Intended for: Exploratory usage and parameter discovery

Examples:
  # Interactive exploration with default parameters
  ftmwpipeline ft-visualize exp_2638
  
  # Explore custom parameters (will offer to save complete set)
  ftmwpipeline ft-visualize exp_2638 --zpf 2 --expf_us 3.0 --trim 26500:40000
  
  # Static image export
  ftmwpipeline ft-visualize exp_2638 --no-interactive --output spectrum.png
  
  # Advanced windowing and scaling exploration
  ftmwpipeline ft-visualize exp_2638 --start-us 2.0 --end-us 12.0 --window-function hann --units-power 3

Key Features:
  - Interactive parameter exploration with immediate visual feedback
  - Save complete parameter sets (preprocessing + postprocessing) as defaults
  - Matplotlib-based reliable visualization for CLI environments
  - Support for both interactive display and static image export

Workflow:
  1. Experiment with different parameters until you find good ones
  2. Save complete parameter set when prompted (y/N)
  3. Future pipeline stages will use saved parameters as defaults
        """
    )
    
    ft_visualize_parser.add_argument(
        'experiment_id',
        help='Identifier for the cached experiment data'
    )
    ft_visualize_parser.add_argument(
        '--start-us',
        type=float,
        help='FID window start time in microseconds (earlier points set to 0)'
    )
    ft_visualize_parser.add_argument(
        '--end-us',
        type=float,
        help='FID window end time in microseconds (later points set to 0)'
    )
    ft_visualize_parser.add_argument(
        '--zpf',
        type=int,
        default=1,
        help='Zero padding factor for improved frequency resolution (default: 1)'
    )
    ft_visualize_parser.add_argument(
        '--expf_us',
        type=float,
        default=5.0,
        help='Exponential filter in microseconds for sensitivity enhancement (default: 5.0)'
    )
    ft_visualize_parser.add_argument(
        '--window-function',
        help='Windowing function (hann, blackman, hamming, etc.)'
    )
    ft_visualize_parser.add_argument(
        '--units-power',
        type=int,
        default=6,
        help='Scaling factor for spectrum units as power of 10 (default: 6 for μV)'
    )
    ft_visualize_parser.add_argument(
        '--trim',
        help='Frequency range to keep as "min:max" in MHz (e.g., "26500:40000")'
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