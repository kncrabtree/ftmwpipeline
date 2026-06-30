"""
FT processing and visualization commands.

This module implements the ``ft run`` and ``ft show`` subcommands
for basic FTMW data processing and visualization.
"""

import argparse
import sys
from pathlib import Path

# Import shared implementations
from .._internal.stage1_impl import compute_ft_impl, visualize_ft_impl
from ._argspec import add_settings_args, settings_from_namespace
from .utils import (
    add_stage_object,
    print_error,
    print_processing_params,
    setup_logging,
)


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

    For interactive parameter exploration with visualization, use 'ft show'.

    Stage-based workflow:
    1. Load FID data from Stage 0 cache ('data import' command output)
    2. Test preprocessing (active-region selection, DC removal)
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
        print_processing_params(trim_range)
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
            print(f"   Preprocessed to {result['preprocessed_points']:,} points")
            print(f"   Final spectrum: {result['frequency_points']:,} frequency points")
            if "trimmed_points" in result:
                print(f"   After trimming: {result['trimmed_points']:,} points")

            print()
            print("Parameters stored for subsequent pipeline stages")
            print("   ComplexFT will be calculated on-demand when needed")
            print(f"   Next steps: ftmwpipeline noise run {file_path}")
            print(f"              ftmwpipeline ft show {file_path}")
            if trim_range:
                print(
                    f"              ftmwpipeline ft show {file_path} "
                    f"--trim {trim_range[0]:.0f}:{trim_range[1]:.0f}"
                )

            return 0

        except FileNotFoundError:
            print_error(f"Pipeline file not found: {file_path}")
            print("")
            print("Stage 0 (Data Import) must be completed before FT processing.")
            print(f"Run: ftmwpipeline data import {file_path} <path>")
            print("")
            print("For example:")
            print(
                f"  ftmwpipeline data import {file_path} "
                f"examples/blackchirp_data/2638/"
            )
            print(f"  ftmwpipeline ft run {file_path}")
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
    """Interactive parameter exploration with FID visualization panels.

    This is the companion tool to 'ft run', designed for exploratory usage
    where users want to try different processing parameters and see immediate
    visual feedback. Creates multi-panel plots showing the complete processing
    workflow from raw FID to final spectrum. Visualization never persists; to
    store settings as canonical, use 'ft run'.

    Visualization panels:
    - Raw FID panel with windowing bounds (start_us/end_us vertical lines)
    - Preprocessed FID panel showing the active-region selection and DC removal
    - Spectrum panels (magnitude and real/imaginary components)

    Use this command to:
    - Visualize the complete FID-to-spectrum processing workflow
    - Understand the effects of the active-region and trim parameters
    - See the impact of parameters on both time and frequency domains
    - Create diagnostic plots for publications/presentations

    Key features:
    - On-demand ComplexFT calculation (never persisted)
    - Matplotlib backend for reliable CLI visualization
    - Support for both interactive display and static image export

    Workflow:
    1. Load FID data from .ftmw pipeline file
    2. Apply the requested processing parameters (active region, trim, scale)
    3. Display the multi-panel plot showing the complete processing workflow
    """
    setup_logging(args.verbose)

    try:
        file_path: str = args.file_path
        if not file_path.endswith(".ftmw"):
            file_path = file_path + ".ftmw"

        settings = settings_from_namespace(args)
        trim_range = args.trim  # already a (float, float) tuple or None

        print(f"Visualizing FT from '{file_path}' with on-demand calculation")
        print_processing_params(trim_range)
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
                interactive=not args.no_interactive,
            )

            if args.output:
                fig.savefig(args.output, dpi=150, bbox_inches="tight")
                print(f"Enhanced plot saved to: {args.output}")

                import matplotlib.pyplot as plt

                plt.close(fig)
            elif not args.no_interactive:
                import matplotlib.pyplot as plt

                plt.show()
                print("Enhanced interactive plot displayed")
                print("   Close the plot window to continue...")
            else:
                import matplotlib.pyplot as plt

                plt.close(fig)
                print(
                    "(no --output and --no-interactive: figure rendered but not "
                    "saved; pass --output to write it)"
                )

            print()
            print("ComplexFT calculated on-demand from pipeline file")
            print("   Try different parameters without permanent storage:")
            print(f"   ftmwpipeline ft show {file_path} --start-us 2.0 --end-us 12.0")
            if not trim_range:
                print(f"   ftmwpipeline ft show {file_path} --trim 26500:40000")

            return 0

        except FileNotFoundError:
            print_error(f"Pipeline file not found: {file_path}")
            print("")
            print("Stage 0 (Data Import) must be completed before FT visualization.")
            print(f"Run: ftmwpipeline data import {file_path} <path>")
            print("")
            print("For example:")
            print(
                f"  ftmwpipeline data import {file_path} "
                f"examples/blackchirp_data/2638/"
            )
            print(f"  ftmwpipeline ft show {file_path}")
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


def add_ft_subcommands(subparsers: argparse._SubParsersAction) -> None:
    """Add FT processing (Stage 1) object-verb subcommands."""
    verbs = add_stage_object(
        subparsers,
        "ft",
        synonym="stage1",
        help="Stage 1: FT processing (run / show)",
        description="Compute and visualize the Fourier transform (Stage 1).",
    )

    # ft run
    ft_process_parser = verbs.add_parser(
        "run",
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
  # Compute the canonical (unapodized) FT
  ftmwpipeline ft run exp_2638.ftmw

  # Restrict to the analysis band
  ftmwpipeline ft run exp_2638.ftmw --trim 26500:40000

  # Set the active window and scaling
  ftmwpipeline ft run exp_2638.ftmw --start-us 1.0 --end-us 10.0 --units-power 3

Workflow:
  1. ftmwpipeline data import exp_2638.ftmw examples/blackchirp_data/2638/
  2. ftmwpipeline ft run exp_2638.ftmw [--parameters]   # Power user validation
     OR
     ftmwpipeline ft show exp_2638.ftmw [--parameters]  # Interactive exploration
        """,
    )
    ft_process_parser.add_argument("file_path", help="Path to .ftmw pipeline file")
    add_settings_args(ft_process_parser)
    ft_process_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    ft_process_parser.set_defaults(func=cmd_ft_process)

    # ft show
    ft_visualize_parser = verbs.add_parser(
        "show",
        help="Interactive parameter exploration with FID visualization panels",
        description=(
            "Companion to 'ft run' showing the complete "
            "FID-to-spectrum processing workflow."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Purpose: Interactive parameter exploration with FID visualization panels
Intended for: Understanding the processing workflow and choosing parameters

Visualization panels:
  - Raw FID panel shows original time-domain data with windowing bounds
  - Preprocessed FID panel shows the active-region selection and DC removal
  - Spectrum panels show magnitude and real/imaginary frequency components

Examples:
  # Visualization with windowing bounds displayed
  ftmwpipeline ft show exp_2638.ftmw --start-us 2.0 --end-us 12.0

  # Explore the active window with a trimmed frequency range
  ftmwpipeline ft show exp_2638.ftmw --start-us 2.0 --trim 26500:40000

  # Static image export for presentations
  ftmwpipeline ft show exp_2638.ftmw --start-us 2.0 --end-us 12.0 \\
      --no-interactive --output enhanced_spectrum.png

Note: 'ft show' never persists settings. To store FT settings as canonical,
use 'ft run'.

Workflow:
  1. Visualize the complete processing workflow from FID to spectrum
  2. Try different parameters and see their effects in all processing stages
  3. Re-run 'ft run' with the chosen parameters to store them
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
