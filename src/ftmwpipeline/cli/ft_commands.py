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
        
        # Merge user parameters with cached recommended settings
        cached_defaults = fid.processing
        
        # Test FT processing to validate parameters and provide feedback
        print("Testing FT processing parameters...")
        try:
            # Stage 1: FID Preprocessing  
            preprocessed_fid = fid.preprocess(
                start_us=args.start_us if args.start_us is not None else cached_defaults.start_us,
                end_us=args.end_us if args.end_us is not None else cached_defaults.end_us, 
                zpf=args.zpf if args.zpf is not None else (cached_defaults.zpf if cached_defaults.zpf is not None else 1), 
                expf_us=args.expf_us if args.expf_us is not None else (cached_defaults.expf_us if cached_defaults.expf_us is not None else 5.0),
                window_function=args.window_function if args.window_function is not None else cached_defaults.winf,
                units_power=args.units_power if args.units_power is not None else (cached_defaults.units_power if cached_defaults.units_power is not None else 6)
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
    Enhanced interactive parameter exploration with FID visualization panels.
    
    This is the companion tool to ft-process, designed for exploratory usage
    where users want to experiment with different processing parameters and
    see immediate visual feedback. Creates enhanced multi-panel plots showing
    the complete processing workflow from raw FID to final spectrum.
    
    Enhanced visualization features:
    - Raw FID panel with windowing bounds (start_us/end_us vertical lines)
    - Preprocessed FID panel showing effects of filtering, windowing, zero-padding
    - Traditional spectrum panels (magnitude and real/imaginary components)
    - Interactive parameter exploration with immediate visual feedback
    - Save complete parameter sets (preprocessing + postprocessing) as defaults
    
    Use this command to:
    - Visualize the complete FID-to-spectrum processing workflow
    - Understand the effects of windowing, filtering, and preprocessing parameters
    - Optimize parameters by seeing their impact on both time and frequency domains
    - Create comprehensive diagnostic plots for publications/presentations
    - Save optimal parameter combinations for automated processing
    
    Key features:
    - On-demand ComplexFT calculation (no permanent storage unless requested)
    - Enhanced 3-panel visualization showing processing stages
    - Interactive parameter persistence with complete parameter sets
    - Matplotlib backend for reliable CLI visualization
    - Support for both interactive display and static image export
    
    Workflow:
    1. Load FID data from Stage 0 cache
    2. Apply custom processing parameters (preprocessing + postprocessing)
    3. Display enhanced multi-panel plot showing complete processing workflow
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
        
        # Get processing parameters (merge user input with cached recommended settings)
        cached_defaults = fid.processing  # Recommended settings from cached FID
        processing_params = {
            'start_us': args.start_us if args.start_us is not None else cached_defaults.start_us,
            'end_us': args.end_us if args.end_us is not None else cached_defaults.end_us,
            'zpf': args.zpf if args.zpf is not None else (cached_defaults.zpf if cached_defaults.zpf is not None else 1),
            'expf_us': args.expf_us if args.expf_us is not None else (cached_defaults.expf_us if cached_defaults.expf_us is not None else 5.0),
            'window_function': args.window_function if args.window_function is not None else cached_defaults.winf,
            'units_power': args.units_power if args.units_power is not None else (cached_defaults.units_power if cached_defaults.units_power is not None else 6)
        }
        
        # Update FID processing parameters for visualization consistency
        # (The original cached parameters are preserved)
        from ..core.data_structures import FIDProcessingParameters
        current_processing = FIDProcessingParameters(
            start_us=processing_params['start_us'],
            end_us=processing_params['end_us'],
            winf=processing_params['window_function'],
            zpf=processing_params['zpf'],
            rdc=True,
            expf_us=processing_params['expf_us'],
            units_power=processing_params['units_power']
        )
        # Temporarily update for visualization (doesn't affect cache)
        fid.processing = current_processing
        
        # Display all processing parameters (show complete parameter set)
        print("Processing parameters:")
        print(f"  start_us: {processing_params['start_us'] or 'None (full FID start)'}")
        print(f"  end_us: {processing_params['end_us'] or 'None (full FID end)'}")
        print(f"  zpf: {processing_params['zpf']}")
        print(f"  expf_us: {processing_params['expf_us'] or 'None (no exponential filter)'}")
        print(f"  window_function: {processing_params['window_function'] or 'None (no windowing)'}")
        print(f"  units_power: {processing_params['units_power']}")
        
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
        
        # Generate enhanced plot with FID panels
        print("Creating enhanced spectrum plot with FID panels...")
        try:
            plot_title = f"Experiment {args.experiment_id} - Enhanced FT Visualization"
            if trim_range:
                plot_title += f" ({trim_range[0]:.0f}-{trim_range[1]:.0f} MHz)"
            
            # Always use matplotlib backend for CLI - it's more reliable than plotly for CLI usage
            if args.no_interactive:
                # Create static plot and save to file
                fig = plot_complex_ft(
                    complex_ft=complex_ft,
                    title=plot_title,
                    backend='matplotlib',
                    interactive=False,
                    fid=fid,
                    preprocessed_fid=preprocessed_fid,
                    show_fid_panels=True
                )
                if args.output:
                    fig.savefig(args.output, dpi=150, bbox_inches='tight')
                    print(f"✅ Enhanced plot saved to: {args.output}")
                else:
                    # Save with default name
                    output_file = f"{args.experiment_id}_enhanced_spectrum.png"
                    fig.savefig(output_file, dpi=150, bbox_inches='tight')
                    print(f"✅ Enhanced plot saved to: {output_file}")
                
                # Close the figure to free memory
                import matplotlib.pyplot as plt
                plt.close(fig)
            else:
                # Create interactive matplotlib plot
                fig = plot_complex_ft(
                    complex_ft=complex_ft,
                    title=plot_title,
                    backend='matplotlib',
                    interactive=True,
                    fid=fid,
                    preprocessed_fid=preprocessed_fid,
                    show_fid_panels=True
                )
                # Show the interactive plot
                import matplotlib.pyplot as plt
                plt.show()
                print("✅ Enhanced interactive plot displayed")
                print("   Close the plot window to continue...")
                
        except Exception as e:
            print_error(f"Failed to generate enhanced plot: {e}")
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
        help='Zero padding factor for improved frequency resolution (default: from cache or 1)'
    )
    ft_process_parser.add_argument(
        '--expf_us',
        type=float,
        help='Exponential filter in microseconds for sensitivity enhancement (default: from cache or 5.0)'
    )
    ft_process_parser.add_argument(
        '--window-function',
        help='Windowing function (hann, blackman, hamming, etc.)'
    )
    ft_process_parser.add_argument(
        '--units-power',
        type=int,
        help='Scaling factor for spectrum units as power of 10 (default: from cache or 6)'
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
        help='Enhanced interactive parameter exploration with FID visualization panels',
        description='Enhanced companion tool to ft-process showing complete FID-to-spectrum processing workflow.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Purpose: Enhanced interactive parameter exploration with FID visualization panels
Intended for: Understanding processing workflow and parameter optimization

Enhanced Visualization:
  - Raw FID panel shows original time-domain data with windowing bounds
  - Preprocessed FID panel shows effects of filtering, windowing, zero-padding
  - Spectrum panels show magnitude and real/imaginary frequency components
  - Interactive parameter exploration with complete processing workflow visibility

Examples:
  # Enhanced visualization with windowing bounds displayed
  ftmwpipeline ft-visualize exp_2638 --start-us 2.0 --end-us 12.0 --expf_us 5.0
  
  # Explore custom parameters with trimmed frequency range
  ftmwpipeline ft-visualize exp_2638 --zpf 2 --expf_us 3.0 --trim 26500:40000
  
  # Static enhanced image export for presentations
  ftmwpipeline ft-visualize exp_2638 --start-us 2.0 --end-us 12.0 --no-interactive --output enhanced_spectrum.png
  
  # Compare preprocessing effects with different window functions
  ftmwpipeline ft-visualize exp_2638 --window-function hann --expf_us 10.0

Key Features:
  - Enhanced 3-panel visualization showing complete FID-to-spectrum workflow  
  - Interactive parameter exploration with immediate visual feedback
  - Raw FID panel with windowing bounds (start_us/end_us vertical lines)
  - Preprocessed FID panel showing effects of filtering and preprocessing
  - Save complete parameter sets (preprocessing + postprocessing) as defaults
  - Matplotlib-based reliable visualization for CLI environments
  - Support for both interactive display and static image export

Workflow:
  1. Visualize complete processing workflow from FID to spectrum
  2. Experiment with parameters and see effects in all processing stages
  3. Save complete parameter set when prompted (y/N)
  4. Future pipeline stages will use saved parameters as defaults
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
        help='Zero padding factor for improved frequency resolution (default: from cache or 1)'
    )
    ft_visualize_parser.add_argument(
        '--expf_us',
        type=float,
        help='Exponential filter in microseconds for sensitivity enhancement (default: from cache or 5.0)'
    )
    ft_visualize_parser.add_argument(
        '--window-function',
        help='Windowing function (hann, blackman, hamming, etc.)'
    )
    ft_visualize_parser.add_argument(
        '--units-power',
        type=int,
        help='Scaling factor for spectrum units as power of 10 (default: from cache or 6)'
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