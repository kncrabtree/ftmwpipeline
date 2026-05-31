"""
Noise estimation and visualization commands.

This module implements the estimate-noise and visualize-noise subcommands
for Stage 2 noise estimation and diagnostic visualization.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .utils import (
    setup_logging, 
    print_error,
    print_processing_params
)

# Import shared implementations
from .._internal.stage2_impl import (
    compute_noise_estimation_impl, 
    visualize_noise_impl, 
    save_noise_parameters_impl
)


def cmd_estimate_noise(args) -> int:
    """
    Estimate frequency-dependent noise using adaptive binning.
    
    This command performs Stage 2 noise estimation on ComplexFT data stored
    in a .ftmw pipeline file. The algorithm uses recursive binary subdivision
    to create adaptive frequency bins, then identifies noise points within each
    bin by iteratively removing high-magnitude points until target skewness
    is reached (typically ~0.631 for Rayleigh-distributed noise).
    
    Key algorithm features:
    - Adaptive binning based on local variance to handle varying spectral features
    - Skewness-based noise identification for robust separation of signal/noise
    - RMS smoothing for stable noise estimates across frequency
    - Configurable parameters for different experiment types
    
    Use this command to:
    - Generate noise estimates for SNR calculations and peak detection thresholds
    - Create baseline noise models for spectral fitting algorithms
    - Validate noise characteristics across different frequency regions
    - Prepare data for subsequent peak detection and analysis stages
    
    Stage dependencies:
    - Requires Stage 1 (FT computation) to be completed first
    - Results are saved to Stage 2 cache for visualization and analysis
    
    Workflow:
    1. Load ComplexFT data from .ftmw pipeline file (Stage 1 requirement)
    2. Apply adaptive binning strategy based on local spectral variance
    3. Identify noise points within each bin using skewness optimization
    4. Compute smoothed RMS noise estimate across the frequency range
    5. Save NoiseResult with diagnostics to Stage 2 cache
    """
    setup_logging(args.verbose)
    
    try:
        # Ensure file path has .ftmw extension
        file_path = args.file_path
        if not file_path.endswith('.ftmw'):
            file_path = file_path + '.ftmw'
        
        # Prepare parameters, filtering out None values
        params = {}
        if args.skew_target is not None:
            params['skew_target'] = args.skew_target
        if args.min_bin_fraction is not None:
            params['min_bin_fraction'] = args.min_bin_fraction
        if args.smoothing_window_mhz is not None:
            params['smoothing_window_mhz'] = args.smoothing_window_mhz
        if args.min_noise_fraction is not None:
            params['min_noise_fraction'] = args.min_noise_fraction

        # Add from_saved_params flag
        params['from_saved_params'] = args.from_saved_params
        if args.preset is not None:
            params['preset'] = args.preset

        # Scatter (high-pass) estimator selection + knobs.
        params['method'] = args.method
        if args.region_aware is not None:
            params['region_aware'] = args.region_aware
        if args.window_mhz is not None:
            params['window_mhz'] = args.window_mhz
        if args.pedestal_mhz is not None:
            params['pedestal_mhz'] = args.pedestal_mhz
        if args.line_k is not None:
            params['line_k'] = args.line_k
        if args.n_iter is not None:
            params['n_iter'] = args.n_iter
        if args.smoothing_mhz is not None:
            params['smoothing_mhz'] = args.smoothing_mhz
        if args.smoothing_percentile is not None:
            params['smoothing_percentile'] = args.smoothing_percentile
        if args.convolve_mhz is not None:
            params['convolve_mhz'] = args.convolve_mhz

        print(f"Estimating noise for: {file_path}")
        
        # Print parameters being used
        if params:
            print("\nNoise estimation parameters:")
            for param, value in params.items():
                if param != 'from_saved_params':
                    print(f"  {param}: {value}")
            if args.from_saved_params:
                print("  Using saved parameters (ignoring command-line values)")
        else:
            print("Using default parameters for all settings")
        
        # Perform noise estimation using shared implementation
        result = compute_noise_estimation_impl(
            file_path=file_path,
            **params
        )
        
        noise_result = result['noise_result']
        complex_ft = result['complex_ft']
        parameters_used = result['parameters_used']
        
        # Save results to pipeline file
        from .._internal.stage2_impl import save_noise_result_impl
        save_noise_result_impl(
            file_path=file_path,
            noise_result=noise_result,
            complex_ft=complex_ft,
            parameters_used=parameters_used
        )
        
        # Display summary results
        print("\nNoise estimation completed successfully!")
        print("\nResults summary:")
        print(f"  Total frequency points: {result['total_points']:,}")
        print(f"  Noise points identified: {result['noise_points']:,}")
        print(f"  Noise fraction: {result['noise_points']/result['total_points']:.3f}")
        print(f"  Frequency range: {result['frequency_range'][0]:.1f} - {result['frequency_range'][1]:.1f} MHz")
        
        # Noise statistics
        rms_mean = noise_result.rms_noise.mean()
        rms_std = noise_result.rms_noise.std()
        rms_min = noise_result.rms_noise.min()
        rms_max = noise_result.rms_noise.max()
        
        print(f"\nRMS noise statistics:")
        print(f"  Mean RMS: {rms_mean:.2e}")
        print(f"  Std deviation: {rms_std:.2e}")
        print(f"  Range: {rms_min:.2e} - {rms_max:.2e} ({rms_max/rms_min:.1f}x)")
        
        # Algorithm diagnostics
        bin_info = noise_result.bin_info
        print(f"\nAlgorithm diagnostics:")
        print(f"  Strategy: {bin_info.get('algorithm', 'unknown')}")
        print(f"  Number of bins: {bin_info.get('n_bins', 'unknown')}")
        print(f"  Smoothing window: {bin_info.get('smoothing_window_mhz', 'unknown')} MHz")
        
        print(f"\nResults saved to: {file_path}")
        print("Use 'visualize-noise' command to create diagnostic plots")
        
        return 0
        
    except FileNotFoundError as e:
        print_error(f"Pipeline file not found: {e}")
        return 1
    except ValueError as e:
        print_error(f"Invalid parameters or missing dependencies: {e}")
        print("Hint: Run 'compute-ft' command first to generate ComplexFT data")
        return 1
    except Exception as e:
        print_error(f"Noise estimation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_visualize_noise(args) -> int:
    """
    Create noise estimation diagnostic visualization.
    
    This command creates comprehensive diagnostic plots for noise estimation
    results, showing the complete spectral analysis including noise points,
    adaptive bin boundaries, and RMS noise estimates. Essential for validating
    noise estimation quality and understanding algorithm behavior.
    
    Visualization features:
    - Full magnitude spectrum with noise points highlighted
    - Adaptive bin boundaries showing algorithm subdivision strategy
    - RMS noise estimate with confidence intervals (1x, 3x, 5x RMS levels)
    - Algorithm statistics and parameter summary
    - Configurable plot appearance and output options
    
    Use this command to:
    - Validate noise estimation quality and algorithm convergence
    - Understand adaptive binning behavior across different spectral regions
    - Create diagnostic plots for publications and presentations
    - Debug noise estimation issues and parameter optimization
    - Generate standardized noise analysis reports
    
    Stage dependencies:
    - Requires Stage 2 (noise estimation) to be completed first
    - Loads both NoiseResult and ComplexFT data from .ftmw pipeline file
    
    Output options:
    - Interactive matplotlib plots for exploration
    - Static image export for reports and documentation
    - Plotly support for web-based interactive visualization
    - Customizable appearance and layout options
    
    Workflow:
    1. Load noise estimation results from .ftmw pipeline file
    2. Create comprehensive diagnostic visualization
    3. Display interactive plot or save to file
    4. Optionally save visualization parameters for consistency
    """
    setup_logging(args.verbose)
    
    try:
        # Ensure file path has .ftmw extension
        file_path = args.file_path
        if not file_path.endswith('.ftmw'):
            file_path = file_path + '.ftmw'
        
        # Prepare visualization parameters, filtering out None values
        viz_params = {}
        if args.y_max_factor is not None:
            viz_params['y_max_factor'] = args.y_max_factor
        if args.figsize is not None:
            # Parse figsize as "width,height"
            try:
                width, height = map(float, args.figsize.split(','))
                viz_params['figsize'] = (width, height)
            except ValueError:
                print_error(f"Invalid figsize format: {args.figsize}. Use 'width,height' format.")
                return 1
        if args.title is not None:
            viz_params['title'] = args.title
        if args.show_bin_boundaries is not None:
            viz_params['show_bin_boundaries'] = args.show_bin_boundaries
        if args.show_noise_points is not None:
            viz_params['show_noise_points'] = args.show_noise_points
        if args.backend is not None:
            viz_params['backend'] = args.backend
        if args.interactive is not None:
            viz_params['interactive'] = args.interactive
        
        print(f"Creating noise visualization for: {file_path}")
        
        # Print visualization parameters being used
        if viz_params:
            print("\nVisualization parameters:")
            for param, value in viz_params.items():
                print(f"  {param}: {value}")
        
        # Create visualization using shared implementation
        fig = visualize_noise_impl(
            file_path=file_path,
            **viz_params
        )
        
        # Handle output file
        if args.output:
            try:
                output_path = Path(args.output)
                if viz_params.get('backend') == 'plotly':
                    fig.write_html(str(output_path))
                else:
                    fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
                print(f"Visualization saved to: {output_path}")
            except Exception as e:
                print_error(f"Failed to save visualization: {e}")
                return 1
        
        # Save parameters if requested
        if args.save_params:
            try:
                # Filter parameters to save (exclude None and default values)
                params_to_save = {}
                if viz_params.get('y_max_factor') is not None and viz_params['y_max_factor'] != 20.0:
                    params_to_save['y_max_factor'] = viz_params['y_max_factor']
                if viz_params.get('show_bin_boundaries') is not None and viz_params['show_bin_boundaries'] != True:
                    params_to_save['show_bin_boundaries'] = viz_params['show_bin_boundaries']
                if viz_params.get('show_noise_points') is not None and viz_params['show_noise_points'] != True:
                    params_to_save['show_noise_points'] = viz_params['show_noise_points']
                
                if params_to_save:
                    save_noise_parameters_impl(file_path, {'visualization': params_to_save})
                    print(f"Saved {len(params_to_save)} visualization parameters")
                else:
                    print("No custom parameters to save")
                    
            except Exception as e:
                print_error(f"Warning: Failed to save parameters: {e}")
        
        print("Noise visualization completed successfully!")
        
        # Show the plot if not saving to file (and interactive mode)
        if not args.output and viz_params.get('backend', 'matplotlib') == 'matplotlib':
            try:
                import matplotlib.pyplot as plt
                plt.show()
            except Exception as e:
                print_error(f"Warning: Could not display plot: {e}")
        
        return 0
        
    except FileNotFoundError as e:
        print_error(f"Pipeline file not found: {e}")
        return 1
    except ValueError as e:
        print_error(f"Invalid parameters or missing dependencies: {e}")
        print("Hint: Run 'estimate-noise' command first to generate noise results")
        return 1
    except Exception as e:
        print_error(f"Noise visualization failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def register_noise_commands(subparsers):
    """Register noise estimation commands with the main CLI parser."""
    
    # estimate-noise command
    parser_estimate = subparsers.add_parser(
        'estimate-noise',
        help='Estimate frequency-dependent noise using adaptive binning',
        description='Perform Stage 2 noise estimation with configurable parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # File argument
    parser_estimate.add_argument(
        'file_path',
        help='Path to .ftmw pipeline file (.ftmw extension added if missing)'
    )
    
    # Noise estimation parameters
    parser_estimate.add_argument(
        '--skew-target',
        type=float,
        help='Target skewness for noise identification (default: 0.631 for Rayleigh distribution)'
    )
    
    parser_estimate.add_argument(
        '--min-bin-fraction',
        type=float,
        help='Minimum bin size as fraction of total data (default: 1/64 = 0.0156)'
    )
    
    parser_estimate.add_argument(
        '--smoothing-window-mhz',
        type=float,
        help='RMS smoothing window size in MHz (default: auto-calculated from bin width)'
    )
    
    parser_estimate.add_argument(
        '--min-noise-fraction',
        type=float,
        help='Minimum fraction of points that must be noise per bin (default: 2/3 = 0.667)'
    )
    
    # Parameter management
    parser_estimate.add_argument(
        '--from-saved-params',
        action='store_true',
        help='Use saved parameters and ignore command-line values (legacy)'
    )

    parser_estimate.add_argument(
        '--preset',
        type=str,
        default=None,
        help=(
            'Stage 2 preset to apply (bare packaged name or path to a '
            'YAML file). Mutually exclusive with --from-saved-params and '
            'with per-knob flags that explicitly set the same field.'
        ),
    )

    # Estimator selection + scatter (high-pass) knobs.
    parser_estimate.add_argument(
        '--method',
        choices=['adaptive', 'scatter'],
        default='scatter',
        help=(
            'Noise estimator: "scatter" (high-pass, region-aware; default; '
            'immune to the leakage pedestal on high-SNR, line-dense spectra) '
            'or "adaptive" (legacy level-based binning)'
        ),
    )
    parser_estimate.add_argument(
        '--no-region-aware',
        dest='region_aware',
        action='store_false',
        default=None,
        help='Scatter estimator: use the fixed mid-regime factor instead of the '
             'Rician C(R) lookup',
    )
    parser_estimate.add_argument(
        '--window-mhz',
        type=float,
        help='Scatter estimator: per-region scatter-MAD window width in MHz '
             '(default: 80)',
    )
    parser_estimate.add_argument(
        '--pedestal-mhz',
        type=float,
        help='Scatter estimator: leakage-pedestal running-median width in MHz '
             '(default: 20)',
    )
    parser_estimate.add_argument(
        '--line-k',
        type=float,
        help='Scatter estimator: robust-sigma multiple flagging a bin as a line '
             '(default: 8)',
    )
    parser_estimate.add_argument(
        '--n-iter',
        type=int,
        help='Scatter estimator: self-mask refinement iterations (default: 3)',
    )
    parser_estimate.add_argument(
        '--smoothing-mhz',
        type=float,
        help='Scatter estimator: broad lower-envelope sigma smoothing width in '
             'MHz (default: 800; 0 disables)',
    )
    parser_estimate.add_argument(
        '--smoothing-percentile',
        type=float,
        help='Scatter estimator: smoothing percentile (default: 50 = median; '
             'lower = more aggressive floor de-inflation)',
    )
    parser_estimate.add_argument(
        '--convolve-mhz',
        type=float,
        help='Scatter estimator: Gaussian sigma in MHz of the 2nd '
             'step-removing smoothing pass (default: 200; 0 disables)',
    )

    # General options
    parser_estimate.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output and detailed diagnostics'
    )
    
    parser_estimate.set_defaults(func=cmd_estimate_noise)
    
    # visualize-noise command
    parser_visualize = subparsers.add_parser(
        'visualize-noise',
        help='Create noise estimation diagnostic visualization',
        description='Generate comprehensive diagnostic plots for noise estimation results',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # File argument
    parser_visualize.add_argument(
        'file_path',
        help='Path to .ftmw pipeline file with noise estimation results'
    )
    
    # Visualization parameters
    parser_visualize.add_argument(
        '--y-max-factor',
        type=float,
        help='Y-axis maximum as multiple of median RMS noise (default: 20.0)'
    )
    
    parser_visualize.add_argument(
        '--figsize',
        type=str,
        help='Figure size as "width,height" in inches (default: "16,6" for wide aspect)'
    )
    
    parser_visualize.add_argument(
        '--title',
        type=str,
        help='Custom title for the plot (default: auto-generated from filename)'
    )
    
    parser_visualize.add_argument(
        '--show-bin-boundaries',
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        help='Show adaptive bin boundaries as vertical lines (default: true)'
    )
    
    parser_visualize.add_argument(
        '--show-noise-points',
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        help='Highlight identified noise points (default: true)'
    )
    
    parser_visualize.add_argument(
        '--backend',
        choices=['matplotlib', 'plotly'],
        help='Plotting backend (default: matplotlib)'
    )
    
    parser_visualize.add_argument(
        '--interactive',
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        help='Create interactive plots (default: true)'
    )
    
    # Output options
    parser_visualize.add_argument(
        '-o', '--output',
        type=str,
        help='Save plot to file (format determined by extension: .png, .pdf, .svg, .html)'
    )
    
    parser_visualize.add_argument(
        '--save-params',
        action='store_true',
        help='Save custom visualization parameters for future use'
    )
    
    # General options
    parser_visualize.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output and detailed diagnostics'
    )
    
    parser_visualize.set_defaults(func=cmd_visualize_noise)