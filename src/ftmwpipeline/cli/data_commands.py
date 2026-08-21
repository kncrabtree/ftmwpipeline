"""
CLI commands for Stage 0 data loading operations.

This module provides CLI commands for loading experimental data from various
formats and visualizing FID data for validation.
"""

import argparse
from pathlib import Path

import h5py

from .._internal.stage0_impl import (
    get_pipeline_info_impl,
    import_data_impl,
    visualize_fid_impl,
)
from ..io.data_loaders import get_format_info, list_formats
from ..io.fid_serialization import load_acquisition_segments_from_hdf5
from .utils import add_stage_object, setup_logging


def cmd_data_load(args: argparse.Namespace) -> int:
    """
    Import experimental data into a .ftmw pipeline file.

    This command handles data loading from various experimental formats
    (Blackchirp, native ftmw-hdf5, CSV, Keysight-MAT) and creates a .ftmw
    pipeline file for use in subsequent pipeline stages.
    """
    setup_logging(args.verbose)

    try:
        # Validate inputs
        if not args.file_path:
            print("Error: file_path is required")
            return 1

        if not args.source:
            print("Error: --source path is required")
            return 1

        file_path = args.file_path
        if not file_path.endswith(".ftmw"):
            file_path = file_path + ".ftmw"

        print(f"Importing data into pipeline file '{file_path}'")
        print(f"Source: {args.source}")

        # Prepare loading parameters
        format_params = {}

        # Handle format-specific parameters
        if args.format == "blackchirp" and args.fid_index is not None:
            format_params["fid_index"] = args.fid_index

        # Generic-loader acquisition metadata (CSV and native ftmw-hdf5). These
        # may also arrive via a --metadata sidecar or, for ftmw-hdf5, embedded
        # in the file, so missing required values are reported by the loader's
        # resolver (with the precedence rule) rather than pre-checked here.
        if args.format in (None, "csv", "ftmw-hdf5"):
            for attr in ("spacing_us", "probe_freq_mhz", "sideband", "shots"):
                value = getattr(args, attr, None)
                if value is not None:
                    format_params[attr] = value
            if getattr(args, "metadata", None) is not None:
                format_params["metadata"] = args.metadata
            if getattr(args, "column", None) is not None:
                format_params["column"] = args.column

        # Keysight-MAT / segmented scope-record parameters
        if args.pre_record_us is not None:
            format_params["pre_record_us"] = args.pre_record_us
        if args.frame_period_us is not None:
            format_params["frame_period_us"] = args.frame_period_us
        if args.n_frames is not None:
            format_params["n_frames"] = args.n_frames
        if args.frame is not None:
            format_params["frame"] = args.frame
        if args.keep_frames:
            format_params["keep_frames"] = True
        if args.channel is not None:
            format_params["channel"] = args.channel
        if args.interleave_factors is not None:
            format_params["interleave_factors"] = args.interleave_factors
        if getattr(args, "chirp_start_us", None) is not None:
            format_params["chirp_start_us"] = args.chirp_start_us
        if getattr(args, "chirp_end_us", None) is not None:
            format_params["chirp_end_us"] = args.chirp_end_us
        if getattr(args, "start_margin_us", None) is not None:
            format_params["start_margin_us"] = args.start_margin_us

        # Use shared implementation for data import
        result = import_data_impl(
            file_path=file_path,
            source=args.source,
            format_name=args.format,
            force=getattr(args, "force", False),
            **format_params,
        )

        # Display results
        print("Data import completed successfully!")
        print(f"Pipeline file: {result['pipeline_file']}")
        print(f"Source format: {result['format_name']}")

        # Show FID metadata
        fid_info = result["fid_metadata"]
        print("FID Information:")
        print(f"   Data points: {fid_info['n_points']:,}")
        print(f"   Duration: {fid_info['duration_us']:.1f} μs")
        print(f"   Probe freq: {fid_info['probe_freq_mhz']:.3f} MHz")
        print(f"   Sideband: {fid_info['sideband']}")
        print(f"   Shots: {fid_info['shots']:,}")

        # Show file size
        pipeline_file = Path(result["pipeline_file"])
        file_size_mb = pipeline_file.stat().st_size / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")

        print("\nNext steps:")
        print(f"   • Visualize FID: ftmwpipeline data show {file_path}")
        print(f"   • Process FT: ftmwpipeline ft run {file_path}")

        return 0

    except KeyboardInterrupt:
        print("\nOperation canceled by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


def cmd_data_visualize(args: argparse.Namespace) -> int:
    """
    Visualize FID data from pipeline file.

    This command loads FID data from a .ftmw pipeline file and creates plots for
    data validation and quality assessment.
    """
    setup_logging(args.verbose)

    try:
        file_path = args.file_path
        if not file_path.endswith(".ftmw"):
            file_path = file_path + ".ftmw"

        print(f"Visualizing FID data from '{file_path}'...")

        # Use shared implementation for FID visualization
        try:
            fig = visualize_fid_impl(
                file_path=file_path,
                show_metadata=args.show_metadata,
                title=f"Pipeline {Path(file_path).stem} - FID Data",
            )

            if args.save:
                output_file = Path(f"{Path(file_path).stem}_fid.png")
                fig.savefig(output_file, dpi=300, bbox_inches="tight")
                print(f"Plot saved: {output_file}")

            if not args.no_show:
                import matplotlib.pyplot as plt

                plt.show()

            print("FID visualization completed")
            return 0

        except Exception as e:
            print(f"Error creating visualization: {e}")
            # Fall back to basic info display using pipeline info
            try:
                file_path_obj, source_metadata, stage_tracker, fid = (
                    get_pipeline_info_impl(file_path)
                )

                print("\nFID Data Summary:")
                print(f"   Data points: {fid.n_points:,}")
                print(f"   Duration: {fid.duration_us:.1f} μs")
                print(f"   Spacing: {fid.spacing:.4e} s")
                print(f"   Probe frequency: {fid.probe_freq_mhz:.3f} MHz")
                print(f"   Sideband: {fid.sideband.value}")
                print(f"   Shots: {fid.shots}")

                # Show acquisition segment map when present
                try:
                    with h5py.File(file_path, "r") as h5f:
                        if "stage0_fid_data" in h5f:
                            segs = load_acquisition_segments_from_hdf5(
                                h5f["stage0_fid_data"]
                            )
                            if segs is not None:
                                frame_sel_str = (
                                    str(segs.frame_selection)
                                    if segs.frame_selection is not None
                                    else "avg"
                                )
                                has_frames = segs.frames is not None
                                print("\nAcquisition Segments:")
                                print(
                                    f"   Pre-record: {segs.pre_record_us:.2f} µs"
                                    f"  ({len(segs.pre_record):,} samples)"
                                )
                                print(f"   Frame period: {segs.frame_period_us:.2f} µs")
                                print(f"   Frames: {segs.n_frames}")
                                print(f"   Frame selection: {frame_sel_str}")
                                print(
                                    f"   Tail: {len(segs.tail):,} samples"
                                    f"  ({len(segs.tail) * segs.sample_dt * 1e6:.2f} µs)"
                                )
                                print(f"   Per-frame data stored: {has_frames}")
                except Exception:
                    pass  # Segment display failure is non-fatal

                if args.show_metadata:
                    print("\nSource Metadata:")
                    print(f"   Source path: {source_metadata.source_path}")
                    print(f"   Format: {source_metadata.format_name}")
                    print(f"   Import time: {source_metadata.import_timestamp}")
                    if source_metadata.loader_parameters:
                        print(
                            f"   Loader parameters: {source_metadata.loader_parameters}"
                        )

                return 0
            except Exception as info_error:
                print(f"Error getting pipeline info: {info_error}")
                return 1

    except KeyboardInterrupt:
        print("\nOperation canceled by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


def cmd_data_info(args: argparse.Namespace) -> int:
    """
    Show information about available data formats and loaders.
    """
    setup_logging(args.verbose)

    try:
        if args.format:
            # Show info about specific format
            try:
                info = get_format_info(args.format)
                print(f"Format Information: {args.format}")
                print(f"   Loader class: {info['loader_class']}")

                if info["file_extensions"]:
                    print(f"   File extensions: {', '.join(info['file_extensions'])}")

                if info["directory_indicators"]:
                    print(
                        f"   Directory indicators: {', '.join(info['directory_indicators'])}"
                    )

                if info["required_parameters"]:
                    print(
                        f"   Required parameters: {', '.join(info['required_parameters'])}"
                    )
                else:
                    print("   Required parameters: None")

                if info["optional_parameters"]:
                    print("   Optional parameters:")
                    for param, default in info["optional_parameters"].items():
                        print(f"     • {param} (default: {default})")

            except ValueError as e:
                print(f"Error: {e}")
                return 1
        else:
            # Show all available formats
            formats = list_formats()
            print(f"Available Data Formats ({len(formats)}):")
            print()

            for fmt in formats:
                try:
                    info = get_format_info(fmt)
                    print(f"{fmt}")
                    print(f"   Loader: {info['loader_class']}")

                    if info["file_extensions"]:
                        print(f"   Extensions: {', '.join(info['file_extensions'])}")
                    elif info["directory_indicators"]:
                        print(
                            f"   Directory format (indicators: {', '.join(info['directory_indicators'])})"
                        )

                    if info["required_parameters"]:
                        print(f"   Requires: {', '.join(info['required_parameters'])}")

                    print()
                except Exception:
                    print(f"{fmt} (info unavailable)")
                    print()

        return 0

    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


def add_data_subcommands(subparsers: argparse._SubParsersAction) -> None:
    """Add data import (Stage 0) object-verb subcommands.

    The ``data`` object carries ``import`` (create the .ftmw from a source) and
    ``show`` (visualize the imported FID). ``formats`` stays a bare utility
    command (it is file-global, not a stage action), registered separately.
    """
    verbs = add_stage_object(
        subparsers,
        "data",
        synonym="stage0",
        help="Stage 0: data import (import / show)",
        description="Import experimental data into a .ftmw and visualize the FID.",
    )

    # data import
    load_parser = verbs.add_parser(
        "import",
        help="Load experimental data from various formats, creating a .ftmw",
        description="Load FTMW experimental data and cache as FID object for pipeline processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import Blackchirp experiment (auto-detect format)
  ftmwpipeline data import exp_2638.ftmw examples/blackchirp_data/2638/

  # Import Blackchirp with specific FID index
  ftmwpipeline data import exp_2638.ftmw examples/blackchirp_data/2638/ --fid-index 1

  # Import a CSV column of voltage samples (metadata via flags)
  ftmwpipeline data import exp_csv.ftmw data.csv --format csv --spacing_us 0.02 --probe_freq_mhz 40960

  # Import a CSV, selecting a named column and reading metadata from a sidecar
  ftmwpipeline data import exp_csv.ftmw data.csv --format csv --column volts --metadata data.meta.json

  # Import a native ftmwpipeline HDF5 file (self-describing; clocks embedded)
  ftmwpipeline data import exp.ftmw mydata.h5 --format ftmw-hdf5

  # Force a specific format
  ftmwpipeline data import exp_2638.ftmw examples/blackchirp_data/2638/ --format blackchirp
        """,
    )

    load_parser.add_argument("file_path", help="Path to .ftmw pipeline file to create")
    load_parser.add_argument("source", help="Path to data source (file or directory)")
    load_parser.add_argument(
        "--format",
        choices=["blackchirp", "csv", "ftmw-hdf5", "keysight-mat"],
        help="Data format (auto-detected if not specified)",
    )
    load_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )
    load_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing file even with different source or layout",
    )

    # Blackchirp-specific options
    load_parser.add_argument(
        "--fid-index", type=int, help="FID index for Blackchirp format (default: 0)"
    )

    # Generic-loader acquisition metadata (CSV and native ftmw-hdf5).
    # For ftmw-hdf5 these override the file's embedded values; a --metadata
    # sidecar sits between the two (explicit flag > sidecar > embedded > default).
    load_parser.add_argument(
        "--spacing_us",
        type=float,
        help="Sample period in μs (required for CSV unless set in a sidecar)",
    )
    load_parser.add_argument(
        "--probe_freq_mhz",
        type=float,
        help="Probe/LO frequency in MHz (default: 0, a direct sampler whose "
        "baseband is the molecular frequency)",
    )
    load_parser.add_argument(
        "--sideband",
        choices=["upper", "lower"],
        help="Sideband (default: upper)",
    )
    load_parser.add_argument(
        "--shots", type=int, help="Number of averaged shots (default: 1)"
    )
    load_parser.add_argument(
        "--column",
        default=None,
        help="CSV voltage column: an integer index (0-based) or a column name "
        "(default: first column)",
    )
    load_parser.add_argument(
        "--metadata",
        default=None,
        help="Path to a JSON/YAML sidecar of acquisition metadata and clock "
        "declarations (auto-discovered as <source>.ftmwmeta.json/.yaml if omitted)",
    )

    # Keysight-MAT / segmented scope-record options
    load_parser.add_argument(
        "--pre-record-us",
        dest="pre_record_us",
        type=float,
        help="Pre-record duration in µs (required for keysight-mat)",
    )
    load_parser.add_argument(
        "--frame-period-us",
        dest="frame_period_us",
        type=float,
        help="Frame repetition period in µs (required for keysight-mat)",
    )
    load_parser.add_argument(
        "--n-frames",
        dest="n_frames",
        type=int,
        help="Number of frames in the record (required for keysight-mat)",
    )
    load_parser.add_argument(
        "--frame",
        dest="frame",
        type=int,
        default=None,
        help="Single frame index (0-based) to import instead of coherent average",
    )
    load_parser.add_argument(
        "--keep-frames",
        dest="keep_frames",
        action="store_true",
        help="Retain per-frame data in the pipeline file",
    )
    load_parser.add_argument(
        "--channel",
        dest="channel",
        default=None,
        help="Channel group name for keysight-mat (e.g. Channel_3); "
        "defaults to the sole channel present",
    )
    load_parser.add_argument(
        "--interleave-factors",
        dest="interleave_factors",
        default=None,
        type=lambda s: [int(x) for x in s.split(",")],
        metavar="M[,M,...]",
        help="Comma-separated ADC interleave factors for offset cleanup "
        "(e.g. 16,512).  Applied sequentially on the pre-record before "
        "slicing/averaging.",
    )

    # Declared chirp-window timing (keysight-mat and any format without
    # embedded chirp metadata).
    load_parser.add_argument(
        "--chirp-start-us",
        dest="chirp_start_us",
        type=float,
        default=None,
        help="Pre-chirp hardware delay within the frame (µs from frame t=0). "
        "Optional; used together with --chirp-end-us for provenance only.",
    )
    load_parser.add_argument(
        "--chirp-end-us",
        dest="chirp_end_us",
        type=float,
        default=None,
        help="End of the chirp sweep within the frame (µs from frame t=0). "
        "When given, a recommended start_us is derived at import time as "
        "chirp_end_us + start_margin_us (or the default guard margin).",
    )
    load_parser.add_argument(
        "--start-margin-us",
        dest="start_margin_us",
        type=float,
        default=None,
        help="Instrument-specific ringdown guard margin (µs) added past the "
        "declared chirp end. Overrides the start-detector default (0.67 µs) "
        "when --chirp-end-us is set.",
    )

    load_parser.set_defaults(func=cmd_data_load)

    # data show
    viz_parser = verbs.add_parser(
        "show",
        help="Visualize cached FID data",
        description="Load FID data from cache and create validation plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic FID visualization
  ftmwpipeline data show exp_2638.ftmw

  # Show metadata and save plot
  ftmwpipeline data show exp_2638.ftmw --show-metadata --save

  # Non-interactive mode
  ftmwpipeline data show exp_2638.ftmw --no-show --save
        """,
    )

    viz_parser.add_argument("file_path", help="Path to .ftmw pipeline file")
    viz_parser.add_argument(
        "--show-metadata", action="store_true", help="Display metadata information"
    )
    viz_parser.add_argument("--save", action="store_true", help="Save plot to file")
    viz_parser.add_argument(
        "--no-show", action="store_true", help="Do not display plot interactively"
    )
    viz_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )

    viz_parser.set_defaults(func=cmd_data_visualize)

    # formats command
    info_parser = subparsers.add_parser(
        "formats",
        help="Show information about data formats",
        description="Display information about available data format loaders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available formats
  ftmwpipeline formats

  # Show details about specific format
  ftmwpipeline formats --format blackchirp
        """,
    )

    info_parser.add_argument("--format", help="Show details about specific format")
    info_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )

    info_parser.set_defaults(func=cmd_data_info)
