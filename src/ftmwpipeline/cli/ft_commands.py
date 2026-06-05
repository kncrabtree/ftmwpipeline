"""
FT processing and visualization commands.

This module implements the compute-ft and visualize-ft subcommands
for basic FTMW data processing and visualization.
"""

import argparse
import sys
from pathlib import Path

from .utils import setup_logging, print_error, print_processing_params
from ._argspec import add_settings_args, settings_from_namespace

# Import shared implementations
from .._internal.stage1_impl import compute_ft_impl, visualize_ft_impl


def cmd_ft_process(args: argparse.Namespace) -> int:
    """Validate and store user-provided FT processing settings for analysis.

    This command loads FID data from Stage 0 cache, validates FT processing
    parameters, and provides detailed feedback about the processing steps.
    Designed for power users and automated pipeline processes who want to
    validate specific parameter combinations.

    Use this command to:
    - Validate processing parameters with immediate feedback
    - Test parameter combinations for optimal results
    - Get detailed processing statistics and information
    - Programmatically validate parameters in automated workflows

    For interactive parameter exploration with visualization, use visualize-ft.

    Stage-based workflow:
    1. Load FID data from Stage 0 cache (import-data command output)
    2. Test preprocessing (windowing, filtering, zero-padding)
    3. Test FFT computation and frequency range
    4. Provide detailed feedback without permanent storage
    """
    setup_logging(args.verbose)

    try:
        file_path: str = args.file_path
        if not file_path.endswith(".ftmw"):
            file_path = file_path + ".ftmw"

        settings = settings_from_namespace(args)
        trim_range = args.trim  # already a (float, float) tuple or None

        print(f"Validating FT processing parameters for '{file_path}'")
        print_processing_params(args.zpf, args.expf_us, trim_range)
        print()

        try:
            result = compute_ft_impl(
                file_path=file_path,
                settings=settings,
                validate_only=False,
                persist=True,
            )

            print(
                "FT processing validation and parameter storage completed successfully!"
            )
            print(
                f"   Processing parameters validated and stored with "
                f"{result['fid_points']:,} FID points"
            )
            print(
                f"   Preprocessed to {result['preprocessed_points']:,} points "
                f"(zero-padded)"
            )
            print(f"   Final spectrum: {result['frequency_points']:,} frequency points")
            if "trimmed_points" in result:
                print(f"   After trimming: {result['trimmed_points']:,} points")

            print()
            print("Parameters stored for subsequent pipeline stages")
            print("   ComplexFT will be calculated on-demand when needed")
            print(f"   Next steps: ftmwpipeline estimate-noise {file_path}")
            print(f"              ftmwpipeline visualize-ft {file_path}")
            if trim_range:
                print(
                    f"              ftmwpipeline visualize-ft {file_path} "
                    f"--trim {trim_range[0]:.0f}:{trim_range[1]:.0f}"
                )

            return 0

        except FileNotFoundError:
            print_error(f"Pipeline file not found: {file_path}")
            print("")
            print("Stage 0 (Data Import) must be completed before FT processing.")
            print(f"Run: ftmwpipeline import-data {file_path} --source <path>")
            print("")
            print("For example:")
            print(
                f"  ftmwpipeline import-data {file_path} "
                f"--source examples/blackchirp_data/2638/"
            )
            print(f"  ftmwpipeline compute-ft {file_path}")
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


def cmd_ft_visualize(args: argparse.Namespace) -> int:
    """Enhanced interactive parameter exploration with FID visualization panels.

    This is the companion tool to compute-ft, designed for exploratory usage
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
        file_path: str = args.file_path
        if not file_path.endswith(".ftmw"):
            file_path = file_path + ".ftmw"

        settings = settings_from_namespace(args)
        trim_range = args.trim  # already a (float, float) tuple or None

        print(f"Visualizing FT from '{file_path}' with on-demand calculation")
        print_processing_params(args.zpf, args.expf_us, trim_range)
        print()

        try:
            pipeline_name = Path(file_path).stem
            plot_title = f"Pipeline {pipeline_name} - Enhanced FT Visualization"
            if trim_range:
                plot_title += f" ({trim_range[0]:.0f}-{trim_range[1]:.0f} MHz)"

            fig = visualize_ft_impl(
                file_path=file_path,
                settings=settings,
                title=plot_title,
                show_fid_panels=True,
                backend="matplotlib",
                interactive=not args.no_interactive,
            )

            if args.no_interactive:
                if args.output:
                    fig.savefig(args.output, dpi=150, bbox_inches="tight")
                    print(f"Enhanced plot saved to: {args.output}")
                else:
                    output_file = f"{pipeline_name}_enhanced_spectrum.png"
                    fig.savefig(output_file, dpi=150, bbox_inches="tight")
                    print(f"Enhanced plot saved to: {output_file}")

                import matplotlib.pyplot as plt

                plt.close(fig)
            else:
                import matplotlib.pyplot as plt

                plt.show()
                print("Enhanced interactive plot displayed")
                print("   Close the plot window to continue...")

            print()
            print("ComplexFT calculated on-demand from pipeline file")
            print("   Try different parameters without permanent storage:")
            print(f"   ftmwpipeline visualize-ft {file_path} --zpf 2 --expf_us 3.0")
            if not trim_range:
                print(f"   ftmwpipeline visualize-ft {file_path} --trim 26500:40000")

            return 0

        except FileNotFoundError:
            print_error(f"Pipeline file not found: {file_path}")
            print("")
            print("Stage 0 (Data Import) must be completed before FT visualization.")
            print(f"Run: ftmwpipeline import-data {file_path} --source <path>")
            print("")
            print("For example:")
            print(
                f"  ftmwpipeline import-data {file_path} "
                f"--source examples/blackchirp_data/2638/"
            )
            print(f"  ftmwpipeline visualize-ft {file_path}")
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


def add_ft_subcommands(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Add FT processing subcommands to the argument parser."""

    # compute-ft command
    ft_process_parser = subparsers.add_parser(
        "compute-ft",
        help="Validate and store user-provided FT processing settings for analysis",
        description=(
            "Power user tool to validate specific FT processing parameters "
            "and provide detailed feedback."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Purpose: Validate and store user-provided processing settings
Intended for: Power users and automated pipeline processes

Examples:
  # Validate basic processing parameters
  ftmwpipeline compute-ft exp_2638.ftmw --zpf 1 --expf_us 5.0

  # Test parameter combinations with trimming
  ftmwpipeline compute-ft exp_2638.ftmw --zpf 2 --expf_us 10.0 --trim 26500:40000

  # Test windowing and scaling parameters
  ftmwpipeline compute-ft exp_2638.ftmw --start-us 1.0 --end-us 10.0 --units-power 3

Workflow:
  1. ftmwpipeline import-data exp_2638.ftmw --source examples/blackchirp_data/2638/
  2. ftmwpipeline compute-ft exp_2638.ftmw [--parameters]  # Power user validation
     OR
     ftmwpipeline visualize-ft exp_2638.ftmw [--parameters] # Interactive exploration
        """,
    )
    ft_process_parser.add_argument("file_path", help="Path to .ftmw pipeline file")
    add_settings_args(ft_process_parser)
    ft_process_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    ft_process_parser.set_defaults(func=cmd_ft_process)

    # visualize-ft command
    ft_visualize_parser = subparsers.add_parser(
        "visualize-ft",
        help="Enhanced interactive parameter exploration with FID visualization panels",
        description=(
            "Enhanced companion tool to compute-ft showing complete "
            "FID-to-spectrum processing workflow."
        ),
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
  ftmwpipeline visualize-ft exp_2638.ftmw --start-us 2.0 --end-us 12.0 --expf_us 5.0

  # Explore custom parameters with trimmed frequency range
  ftmwpipeline visualize-ft exp_2638.ftmw --zpf 2 --expf_us 3.0 --trim 26500:40000

  # Static enhanced image export for presentations
  ftmwpipeline visualize-ft exp_2638.ftmw --start-us 2.0 --end-us 12.0 \\
      --no-interactive --output enhanced_spectrum.png

  # Compare preprocessing effects with different window functions
  ftmwpipeline visualize-ft exp_2638.ftmw --window-function hann --expf_us 10.0

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
        """,
    )
    ft_visualize_parser.add_argument("file_path", help="Path to .ftmw pipeline file")
    add_settings_args(ft_visualize_parser)
    ft_visualize_parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Save plot to image file instead of showing interactive plot",
    )
    ft_visualize_parser.add_argument(
        "--output",
        help="Output image file path (used with --no-interactive)",
    )
    ft_visualize_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    ft_visualize_parser.set_defaults(func=cmd_ft_visualize)
