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
    print_error,
    print_processing_params
)

# Import shared implementations
from .._internal.stage1_impl import compute_ft_impl, visualize_ft_impl, save_ft_parameters_impl, compare_ft_parameters_impl
from .._internal.shared_utils import parse_frequency_range


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
        file_path = args.file_path
        if not file_path.endswith('.ftmw'):
            file_path = file_path + '.ftmw'
        
        # Parse optional trim range
        trim_range = None
        if args.trim:
            try:
                trim_range = parse_frequency_range(args.trim)
            except ValueError as e:
                print_error(f"Invalid trim range: {e}")
                return 1
        
        print(f"Validating FT processing parameters for '{file_path}'")
        print_processing_params(args.zpf, args.expf_us, trim_range)
        print()
        
        # Use shared implementation for FT validation
        try:
            result = compute_ft_impl(
                file_path=file_path,
                start_us=args.start_us,
                end_us=args.end_us,
                zpf=args.zpf,
                expf_us=args.expf_us,
                window_function=args.window_function,
                units_power=args.units_power,
                trim_range=trim_range,
                validate_only=False  # Validate and store parameters (aligns with help text)
            )
            
            print("✅ FT processing validation and parameter storage completed successfully!")
            print(f"   Processing parameters validated and stored with {result['fid_points']:,} FID points")
            print(f"   Preprocessed to {result['preprocessed_points']:,} points (zero-padded)")
            print(f"   Final spectrum: {result['frequency_points']:,} frequency points")
            if 'trimmed_points' in result:
                print(f"   After trimming: {result['trimmed_points']:,} points")
            
            print()
            print("📌 Parameters stored for subsequent pipeline stages")
            print("   ComplexFT will be calculated on-demand when needed")
            print(f"   Next steps: ftmwpipeline estimate-noise {file_path}")
            print(f"              ftmwpipeline ft-visualize {file_path}")
            if trim_range:
                print(f"              ftmwpipeline ft-visualize {file_path} --trim {trim_range[0]:.0f}:{trim_range[1]:.0f}")
            
            return 0
            
        except FileNotFoundError:
            print_error(f"Pipeline file not found: {file_path}")
            print("")
            print("Stage 0 (Data Import) must be completed before FT processing.")
            print(f"Run: ftmwpipeline data-load {file_path} --source <path>")
            print("")
            print("For example:")
            print(f"  ftmwpipeline data-load {file_path} --source examples/blackchirp_data/2638/")
            print(f"  ftmwpipeline ft-process {file_path}")
            return 1
        except Exception as e:
            print_error(f"Failed to process FT: {e}")
            return 1
        
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
    1. Load FID data from .ftmw pipeline file
    2. Apply custom processing parameters (preprocessing + postprocessing)
    3. Display enhanced multi-panel plot showing complete processing workflow
    4. Optionally save complete parameter set as defaults for this experiment
    """
    setup_logging(args.verbose)
    
    try:
        # Ensure file path has .ftmw extension
        file_path = args.file_path
        if not file_path.endswith('.ftmw'):
            file_path = file_path + '.ftmw'
        
        # Parse optional trim range
        trim_range = None
        if args.trim:
            try:
                trim_range = parse_frequency_range(args.trim)
            except ValueError as e:
                print_error(f"Invalid trim range: {e}")
                return 1
        
        print(f"Visualizing FT from '{file_path}' with on-demand calculation")
        print_processing_params(args.zpf, args.expf_us, trim_range)
        print()
        
        # Use shared implementation for FT visualization
        try:
            # Generate title for plot
            pipeline_name = Path(file_path).stem
            plot_title = f"Pipeline {pipeline_name} - Enhanced FT Visualization"
            if trim_range:
                plot_title += f" ({trim_range[0]:.0f}-{trim_range[1]:.0f} MHz)"
            
            fig = visualize_ft_impl(
                file_path=file_path,
                start_us=args.start_us,
                end_us=args.end_us,
                zpf=args.zpf,
                expf_us=args.expf_us,
                window_function=args.window_function,
                units_power=args.units_power,
                trim_range=trim_range,
                title=plot_title,
                show_fid_panels=True,
                backend='matplotlib',
                interactive=not args.no_interactive
            )
            
            # Handle output based on mode
            if args.no_interactive:
                # Save static plot
                if args.output:
                    fig.savefig(args.output, dpi=150, bbox_inches='tight')
                    print(f"✅ Enhanced plot saved to: {args.output}")
                else:
                    # Save with default name
                    output_file = f"{pipeline_name}_enhanced_spectrum.png"
                    fig.savefig(output_file, dpi=150, bbox_inches='tight')
                    print(f"✅ Enhanced plot saved to: {output_file}")
                
                # Close the figure to free memory
                import matplotlib.pyplot as plt
                plt.close(fig)
            else:
                # Show interactive plot
                import matplotlib.pyplot as plt
                plt.show()
                print("✅ Enhanced interactive plot displayed")
                print("   Close the plot window to continue...")
            
            print()
            print("💡 ComplexFT calculated on-demand from pipeline file")
            print("   Try different parameters without permanent storage:")
            print(f"   ftmwpipeline ft-visualize {file_path} --zpf 2 --expf_us 3.0")
            if not trim_range:
                print(f"   ftmwpipeline ft-visualize {file_path} --trim 26500:40000")
            
            return 0
            
        except FileNotFoundError:
            print_error(f"Pipeline file not found: {file_path}")
            print("")
            print("Stage 0 (Data Import) must be completed before FT visualization.")
            print(f"Run: ftmwpipeline data-load {file_path} --source <path>")
            print("")
            print("For example:")
            print(f"  ftmwpipeline data-load {file_path} --source examples/blackchirp_data/2638/")
            print(f"  ftmwpipeline ft-visualize {file_path}")
            return 1
        except Exception as e:
            print_error(f"Failed to visualize FT: {e}")
            return 1
        
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
  ftmwpipeline ft-process exp_2638.ftmw --zpf 1 --expf_us 5.0
  
  # Test parameter combinations with trimming
  ftmwpipeline ft-process exp_2638.ftmw --zpf 2 --expf_us 10.0 --trim 26500:40000
  
  # Test windowing and scaling parameters  
  ftmwpipeline ft-process exp_2638.ftmw --start-us 1.0 --end-us 10.0 --units-power 3

Workflow:
  1. ftmwpipeline data-load exp_2638.ftmw --source examples/blackchirp_data/2638/
  2. ftmwpipeline ft-process exp_2638.ftmw [--parameters]  # Power user validation
     OR
     ftmwpipeline ft-visualize exp_2638.ftmw [--parameters] # Interactive exploration
        """
    )
    
    ft_process_parser.add_argument(
        'file_path',
        help='Path to .ftmw pipeline file'
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
  ftmwpipeline ft-visualize exp_2638.ftmw --start-us 2.0 --end-us 12.0 --expf_us 5.0
  
  # Explore custom parameters with trimmed frequency range
  ftmwpipeline ft-visualize exp_2638.ftmw --zpf 2 --expf_us 3.0 --trim 26500:40000
  
  # Static enhanced image export for presentations
  ftmwpipeline ft-visualize exp_2638.ftmw --start-us 2.0 --end-us 12.0 --no-interactive --output enhanced_spectrum.png
  
  # Compare preprocessing effects with different window functions
  ftmwpipeline ft-visualize exp_2638.ftmw --window-function hann --expf_us 10.0

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
        'file_path',
        help='Path to .ftmw pipeline file'
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