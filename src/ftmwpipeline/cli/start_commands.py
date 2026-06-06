"""CLI subcommands for data-driven FID start-time detection (pre-Stage 1).

Two verbs on the ``start`` object:

- ``start run``: sweep the FID window start, find the chirp-end collapse, and
  stamp the recommended ``start_us`` into the Stage 0 recommended layer.
- ``start show``: plot the Σ|FT|-vs-start sweep diagnostic.

Both delegate to the shared :mod:`_internal.start_detection_impl` orchestration
per the dual-interface rule.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Dict, Optional

from .._internal.start_detection_impl import detect_start_time_impl
from ..core.start_detection_settings import StartDetectionSettings
from .utils import add_stage_object, print_error, setup_logging

logger = logging.getLogger(__name__)


def _settings_from_args(args: argparse.Namespace) -> StartDetectionSettings:
    """Build a settings bundle from the per-knob CLI flags."""
    overrides: Dict[str, Any] = {}
    if args.sweep_max_us is not None:
        overrides["sweep_max_us"] = float(args.sweep_max_us)
    if args.step_us is not None:
        overrides["step_us"] = float(args.step_us)
    if args.guard_margin_us is not None:
        overrides["guard_margin_us"] = float(args.guard_margin_us)
    if args.floor_factor is not None:
        overrides["floor_factor"] = float(args.floor_factor)
    if args.band is not None:
        overrides["band_min_mhz"] = float(args.band[0])
        overrides["band_max_mhz"] = float(args.band[1])
    return StartDetectionSettings(**overrides)


def cmd_detect_start(args: argparse.Namespace) -> int:
    """Run start-time detection and stamp the recommended start_us."""
    setup_logging(args.verbose)
    file_path = args.file_path
    if not file_path.endswith(".ftmw"):
        file_path = file_path + ".ftmw"

    settings = _settings_from_args(args)
    print(f"Detecting FID start time for: {file_path}")
    try:
        out = detect_start_time_impl(
            file_path, settings=settings, stamp=not args.no_stamp
        )
    except FileNotFoundError as e:
        print_error(f"Pipeline file not found: {e}")
        return 1
    except Exception as e:
        print_error(f"Start detection failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    r = out["start_detection"]
    band = (
        f"{r.band_mhz[0]:.0f}-{r.band_mhz[1]:.0f} MHz"
        if r.band_mhz is not None
        else "full spectrum"
    )
    drop = r.plateau / r.floor if r.floor else float("inf")
    print("\nStart detection completed!")
    print("\nResults summary:")
    print(f"  integration band   : {band}")
    print(f"  chirp detected     : {r.chirp_detected} (plateau/floor = {drop:.0f})")
    print(f"  chirp-end          : {r.chirp_end_us:.3f} us")
    print(f"  recommended start  : {r.start_us:.3f} us")
    if out["stamped"]:
        print(f"\nStamped recommended start_us = {r.start_us:.3f} us to {file_path}.")
        print("A later compute_ft with no explicit start_us will inherit it.")
    elif not r.chirp_detected:
        print(
            "\nNo chirp collapse found; nothing stamped. Provide start_us "
            "manually or check the FID."
        )
    else:
        print("\n(--no-stamp) Nothing written.")
    return 0


def cmd_visualize_start_detection(args: argparse.Namespace) -> int:
    """Render the Σ|FT|-vs-start sweep diagnostic."""
    setup_logging(args.verbose)
    file_path = args.file_path
    if not file_path.endswith(".ftmw"):
        file_path = file_path + ".ftmw"

    from ..visualization.start_detection_visualization import (
        plot_start_detection_from_file,
    )

    try:
        fig = plot_start_detection_from_file(
            file_path, settings=_settings_from_args(args)
        )
    except Exception as e:
        print_error(f"Failed to create start-detection visualization: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    return _handle_figure_output(fig, args)


def _handle_figure_output(fig: Any, args: argparse.Namespace) -> int:
    """Save or display a matplotlib figure based on CLI flags."""
    output: Optional[str] = getattr(args, "output", None)
    if output:
        try:
            fig.savefig(output, dpi=150, bbox_inches="tight")
            print(f"Plot saved to: {output}")
        except Exception as e:
            print_error(f"Failed to save plot: {e}")
            return 1
    else:
        try:
            import matplotlib.pyplot as plt

            plt.show()
        except Exception as e:
            print_error(f"Could not display plot: {e}")
            return 1
    return 0


def _add_detection_knobs(parser: argparse.ArgumentParser) -> None:
    """Shared per-knob flags for both subcommands."""
    parser.add_argument(
        "--sweep-max-us",
        type=float,
        help="Upper bound of the start-time sweep (default 7.5; capped to FID)",
    )
    parser.add_argument(
        "--step-us",
        type=float,
        help="Sweep step in us (default 0.02)",
    )
    parser.add_argument(
        "--guard-margin-us",
        type=float,
        help=(
            "Margin added past the chirp end for the switch-bounce ringdown "
            "(default 0.67; instrument-specific)"
        ),
    )
    parser.add_argument(
        "--floor-factor",
        type=float,
        help="Chirp-end = first start where Σ|FT| < factor*floor (default 3.0)",
    )
    parser.add_argument(
        "--band",
        type=float,
        nargs=2,
        metavar=("MIN_MHZ", "MAX_MHZ"),
        help="Explicit integration band override (default: Stage 1 trim or full)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )


def register_start_commands(subparsers: Any) -> None:
    """Register start-detection (pre-Stage-1) object-verb subcommands."""
    verbs = add_stage_object(
        subparsers,
        "start",
        synonym=None,
        help="Start detection: detect/stamp start_us (run / show)",
        description="Detect a good FID start_us and plot the sweep diagnostic.",
    )

    parser_det = verbs.add_parser(
        "run",
        help="Detect a good FID start_us from the data and stamp it",
        description=(
            "Sweep the FID window start time, integrate the FT magnitude over "
            "the active band, and find the chirp-end collapse. The recommended "
            "start is chirp_end + guard margin (switch-bounce settling); it is "
            "stamped to the Stage 0 recommended_processing layer so a later "
            "compute_ft with no explicit start_us inherits it."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_det.add_argument(
        "file_path", help="Path to .ftmw pipeline file (extension added if missing)"
    )
    _add_detection_knobs(parser_det)
    parser_det.add_argument(
        "--no-stamp",
        action="store_true",
        help="Report the recommendation without writing it to the file",
    )
    parser_det.set_defaults(func=cmd_detect_start)

    parser_viz = verbs.add_parser(
        "show",
        help="Plot the Σ|FT|-vs-start_us sweep diagnostic",
        description=(
            "Two-panel diagnostic: the full Σ|FT| sweep (log y) with the "
            "chirp-end and recommended start marked, plus a linear zoom on "
            "the post-chirp floor."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_viz.add_argument(
        "file_path", help="Path to .ftmw pipeline file with the FID imported"
    )
    _add_detection_knobs(parser_viz)
    parser_viz.add_argument(
        "-o",
        "--output",
        type=str,
        help="Save plot to file instead of displaying interactively",
    )
    parser_viz.set_defaults(func=cmd_visualize_start_detection)
