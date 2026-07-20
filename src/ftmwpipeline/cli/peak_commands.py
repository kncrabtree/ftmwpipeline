"""
Peak detection and visualization commands (Stage 3).

Implements the ``peaks run`` and ``peaks show`` subcommands. Thin
wrappers over the shared ``_internal.stage3_impl`` implementation -- identical
behavior to the Pipeline class and functional API.
"""

import argparse
from pathlib import Path
from typing import Any

from .._internal.stage3_impl import detect_peaks_impl, visualize_peaks_impl
from ..core.peak_detection_settings import PeakDetectionSettings
from ._argspec import add_settings_args, settings_from_namespace
from .utils import add_stage_object, print_error, setup_logging


def _ensure_ftmw(path: str) -> str:
    return path if path.endswith(".ftmw") else path + ".ftmw"


def cmd_detect_peaks(args: argparse.Namespace) -> int:
    """Run Stage 3 two-pass peak detection on a .ftmw pipeline file.

    An apodized (Blackman-Harris) primary pass builds the robust coarse peak
    list; a shape-aware matched-filter gap pass then recovers weak lines the
    apodization suppressed. Both passes raise their detection floor continuously
    by the local coherent-leakage amplitude (no hard mask), and every peak is
    scored on the active FT. Peaks are classified by SNR
    (weak/medium/strong) and persisted to the file for hand-curation before
    Stage 4.

    Requires Stage 1 ('ft run') and Stage 2 ('noise run') first.
    """
    setup_logging(args.verbose)
    try:
        file_path = _ensure_ftmw(args.file_path)
        # The per-knob flags are generated from PeakDetectionSettings field
        # metadata; reconstruct a sparse settings bundle (unset fields fall
        # through the resolver). A preset and per-knob flags compose: the flags
        # are the explicit layer, the preset the preset layer beneath persisted.
        settings = settings_from_namespace(args, PeakDetectionSettings)
        preset = args.preset
        print(f"Detecting peaks for: {file_path}")
        result = detect_peaks_impl(
            file_path=file_path,
            settings=None if settings.is_empty() else settings,
            preset=preset,
        )
        peaks = result["peaks"]
        promoted = [p for p in peaks if p.properties.get("promoted")]
        n_strong = sum(
            1
            for p in promoted
            if p.classification and p.classification.value == "strong"
        )
        n_medium = sum(
            1
            for p in promoted
            if p.classification and p.classification.value == "medium"
        )
        n_weak = sum(
            1 for p in promoted if p.classification and p.classification.value == "weak"
        )
        print("\nPeak detection completed successfully!")
        print(f"  Active acquisition T: {result['acquisition_us']:.2f} us")
        print(f"  Total detected: {result['n_peaks']:,}")
        print(
            f"  Promoted (SNR >= {result['promotion_min_snr']:.1f}): "
            f"{result['n_promoted']:,}"
        )
        print(
            f"    primary pass: {result['n_primary']:,}   "
            f"gap pass: {result['n_gap']:,}"
        )
        print(
            f"    strong: {n_strong:,}   medium: {n_medium:,}   weak: {n_weak:,}"
            " (promoted only)"
        )
        print(f"\nResults saved to: {file_path}")
        print("Use 'peaks show' to inspect detected peaks")
        return 0
    except FileNotFoundError as e:
        print_error(f"Pipeline file not found: {e}")
        return 1
    except ValueError as e:
        print_error(f"Invalid parameters or missing dependencies: {e}")
        print("Hint: run 'ft run' then 'noise run' first")
        return 1
    except Exception as e:
        print_error(f"Peak detection failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def cmd_visualize_peaks(args: argparse.Namespace) -> int:
    """Overlay classified Stage 3 peaks on the full-resolution spectrum.

    Default is an interactive matplotlib window for manual inspection. Pass
    ``--no-interactive`` together with ``--output`` to save a static image
    instead (direct it into scratch/ to keep the working tree clean).
    """
    setup_logging(args.verbose)
    try:
        file_path = _ensure_ftmw(args.file_path)
        figsize = None
        if args.figsize is not None:
            try:
                w, h = map(float, args.figsize.split(","))
                figsize = (w, h)
            except ValueError:
                print_error(f"Invalid figsize {args.figsize!r}; use 'width,height'")
                return 1

        print(f"Creating peak visualization for: {file_path}")
        fig = visualize_peaks_impl(
            file_path=file_path,
            figsize=figsize,
            title=args.title,
            y_max_factor=args.y_max_factor,
            interactive=not args.no_interactive,
            show_snr_histogram=args.snr_histogram,
        )

        if args.output:
            fig.savefig(str(Path(args.output)), dpi=300, bbox_inches="tight")
            print(f"Visualization saved to: {args.output}")
        elif not args.no_interactive:
            import matplotlib.pyplot as plt

            plt.show()
        print("Peak visualization completed successfully!")
        return 0
    except FileNotFoundError as e:
        print_error(f"Pipeline file not found: {e}")
        return 1
    except ValueError as e:
        print_error(f"Invalid parameters or missing dependencies: {e}")
        print("Hint: run 'peaks run' first")
        return 1
    except Exception as e:
        print_error(f"Peak visualization failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def register_peak_commands(subparsers: Any) -> None:
    """Register peak detection (Stage 3) object-verb subcommands."""
    verbs = add_stage_object(
        subparsers,
        "peaks",
        synonym="stage3",
        help="Stage 3: peak detection (run / show)",
        description="Detect/classify peaks and overlay them (Stage 3).",
    )

    p_detect = verbs.add_parser(
        "run",
        help="Detect and classify peaks (Stage 3, two-pass)",
        description=(
            "Stage 3 two-pass peak detection with SNR classification.\n\n"
            "Detection scores on the active FT, built from the persisted\n"
            "Stage 1 settings (including its frequency trim range).  Run\n"
            "'ft run' with the desired --trim to set the analysis band first."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_detect.add_argument(
        "file_path", help="Path to .ftmw pipeline file (.ftmw auto-added)"
    )
    # Per-knob flags, generated from PeakDetectionSettings field metadata (the
    # single declaration site shared with `settings` / `scan`). The gap-pass
    # toggle is a BooleanOptionalAction, so it spells both --gap-pass and the
    # historical --no-gap-pass.
    add_settings_args(p_detect, PeakDetectionSettings)
    p_detect.add_argument(
        "--preset",
        dest="preset",
        type=str,
        default=None,
        help=(
            "Stage 3 preset (bare name resolves against packaged presets, or "
            "a path to a YAML file carrying a 'stage3:' block). Composes with "
            "per-knob flags: the flags are the explicit layer, the preset the "
            "layer beneath the persisted record. Knobs the per-flag CLI does "
            "not "
            "expose -- detection_zpf, gap_active_zpf, primary_leakage_floor_k, "
            "gap_leakage_floor_k, internal_min_snr, sg_fwhm_coverage, "
            "sg_min_window -- flow through this flag only."
        ),
    )
    p_detect.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose diagnostics"
    )
    p_detect.set_defaults(func=cmd_detect_peaks)

    p_vis = verbs.add_parser(
        "show",
        help="Overlay classified Stage 3 peaks on the spectrum",
        description="Diagnostic plot of detected/classified peaks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_vis.add_argument("file_path", help="Path to .ftmw file with Stage 3 results")
    p_vis.add_argument("--figsize", type=str, help="'width,height' in inches")
    p_vis.add_argument("--title", type=str, help="Custom plot title")
    p_vis.add_argument(
        "--y-max-factor",
        dest="y_max_factor",
        type=float,
        help="Y-axis max as multiple of median rms (default: 25.0)",
    )
    p_vis.add_argument(
        "--no-interactive",
        dest="no_interactive",
        action="store_true",
        help="Do not open a window (use with --output to save)",
    )
    p_vis.add_argument(
        "-o",
        "--output",
        type=str,
        help="Save plot to this path (e.g. scratch/peaks.png)",
    )
    p_vis.add_argument(
        "--snr-histogram",
        dest="snr_histogram",
        action="store_true",
        help=(
            "Add a second panel: user-grid SNR distribution with the "
            "promotion cutoff marked (curation view)"
        ),
    )
    p_vis.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose diagnostics"
    )
    p_vis.set_defaults(func=cmd_visualize_peaks)
