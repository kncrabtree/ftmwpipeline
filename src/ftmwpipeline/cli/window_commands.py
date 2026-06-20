"""
Window-assignment commands (Stage 4).

Implements the ``windows run`` and ``windows show`` subcommands. Thin
wrappers over the shared ``_internal.stage4_impl`` implementation -- identical
behaviour to the Pipeline class and functional API.
"""

import argparse
from pathlib import Path
from typing import Any

from .._internal.stage4_impl import assign_windows_impl, visualize_windows_impl
from ..core.window_planning_settings import WindowPlanningSettings
from ._argspec import add_settings_args, settings_from_namespace
from .utils import add_stage_object, print_error, setup_logging


def _ensure_ftmw(path: str) -> str:
    return path if path.endswith(".ftmw") else path + ".ftmw"


def cmd_assign_windows(args: argparse.Namespace) -> int:
    """Run Stage 4 window assignment on a .ftmw pipeline file.

    Turns the promoted Stage 3 peaks into a fit plan: disjoint analysis
    windows, each carrying the peaks to fit freely, the strong out-of-band
    lines whose leakage is carried frozen, a fit dependency order, and a
    difficulty class. The plan is persisted for hand-curation before Stage 5.

    Requires Stage 3 ('peaks run') first.
    """
    setup_logging(args.verbose)
    try:
        file_path = _ensure_ftmw(args.file_path)
        # The per-knob flags are generated from WindowPlanningSettings field
        # metadata; reconstruct a sparse settings bundle (unset fields fall
        # through the resolver). --preset is mutually exclusive with knobs.
        settings = settings_from_namespace(args, WindowPlanningSettings)
        if args.preset is not None and not settings.is_empty():
            print_error(
                "--preset and per-knob flags are mutually exclusive; pass one"
            )
            return 1
        print(f"Assigning windows for: {file_path}")
        result = assign_windows_impl(
            file_path=file_path,
            settings=None if settings.is_empty() else settings,
            preset=args.preset,
        )
        plan = result["plan"]
        print("\nWindow assignment completed successfully!")
        print(f"  Promoted peaks consumed: {result['n_promoted']:,}")
        print(
            f"  Windows: {result['n_windows']:,} "
            f"({result['n_hard']:,} hard, {result['n_easy']:,} easy)"
        )
        print(f"  Free peaks: {result['n_free_peaks']:,}")
        print(f"  Fixed contributors: {result['n_fixed_contributors']:,}")
        print(
            f"  Fit dependencies: {result['n_dependencies']:,}   "
            f"parallel batches: {result['n_batches']:,}"
        )
        n_split = sum(1 for w in plan.windows if w.split_proposal is not None)
        n_joint = sum(1 for w in plan.windows if w.needs_joint_treatment)
        if n_split or n_joint:
            print(
                f"  Hard-window annotations: {n_split:,} split proposals, "
                f"{n_joint:,} need joint treatment"
            )
        unexplained = plan.diagnostics.get("unexplained_coherent_regions_mhz")
        if unexplained:
            print(
                f"  WARNING: {len(unexplained)} coherent region(s) with no "
                "promoted peak -- possible undetected line(s)"
            )
        print(f"\nResults saved to: {file_path}")
        print("Use 'windows show' to inspect the window plan")
        return 0
    except FileNotFoundError as e:
        print_error(f"Pipeline file not found: {e}")
        return 1
    except ValueError as e:
        print_error(f"Invalid parameters or missing dependencies: {e}")
        print("Hint: run 'peaks run' first")
        return 1
    except Exception as e:
        print_error(f"Window assignment failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def cmd_visualize_windows(args: argparse.Namespace) -> int:
    """Overlay the Stage 4 window plan on the spectrum.

    Default is an interactive matplotlib window. Pass ``--no-interactive``
    together with ``--output`` to save a static image instead (direct it into
    scratch/ to keep the working tree clean).
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

        print(f"Creating window visualization for: {file_path}")
        fig = visualize_windows_impl(
            file_path=file_path,
            figsize=figsize,
            title=args.title,
            y_max_factor=args.y_max_factor,
            backend="matplotlib",
            interactive=not args.no_interactive,
        )

        if args.output:
            fig.savefig(str(Path(args.output)), dpi=300, bbox_inches="tight")
            print(f"Visualization saved to: {args.output}")
        elif not args.no_interactive:
            import matplotlib.pyplot as plt

            plt.show()
        print("Window visualization completed successfully!")
        return 0
    except FileNotFoundError as e:
        print_error(f"Pipeline file not found: {e}")
        return 1
    except ValueError as e:
        print_error(f"Invalid parameters or missing dependencies: {e}")
        print("Hint: run 'windows run' first")
        return 1
    except Exception as e:
        print_error(f"Window visualization failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def register_window_commands(subparsers: Any) -> None:
    """Register window assignment (Stage 4) object-verb subcommands."""
    verbs = add_stage_object(
        subparsers,
        "windows",
        synonym="stage4",
        help="Stage 4: window assignment (run / show)",
        description="Plan and overlay fit windows (Stage 4).",
    )

    p_assign = verbs.add_parser(
        "run",
        help="Turn promoted peaks into a fit-window plan (Stage 4)",
        description=(
            "Stage 4 window assignment.\n\n"
            "Turns the promoted Stage 3 peaks into a fit plan: disjoint\n"
            "analysis windows, each carrying free peaks, fixed (frozen-leakage)\n"
            "contributors, a fit dependency order, and a difficulty class.\n"
            "Run 'peaks run' first."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_assign.add_argument(
        "file_path", help="Path to .ftmw pipeline file (.ftmw auto-added)"
    )
    # Per-knob flags, generated from WindowPlanningSettings field metadata (the
    # single declaration site shared with `settings` / `scan`).
    add_settings_args(p_assign, WindowPlanningSettings)
    p_assign.add_argument(
        "--preset",
        dest="preset",
        type=str,
        default=None,
        help=(
            "Stage 4 preset (bare name resolves against packaged presets, or "
            "a path to a YAML file carrying a 'stage4:' block). Mutually "
            "exclusive with per-knob flags."
        ),
    )
    p_assign.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose diagnostics"
    )
    p_assign.set_defaults(func=cmd_assign_windows)

    p_vis = verbs.add_parser(
        "show",
        help="Overlay the Stage 4 window plan on the spectrum",
        description="Diagnostic plot of the window plan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_vis.add_argument("file_path", help="Path to .ftmw file with Stage 4 results")
    p_vis.add_argument("--figsize", type=str, help="'width,height' in inches")
    p_vis.add_argument("--title", type=str, help="Custom plot title")
    p_vis.add_argument(
        "--y-max-factor",
        dest="y_max_factor",
        type=float,
        help="Spectrum-panel y-axis headroom (default: 25.0)",
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
        help="Save plot to this path (e.g. scratch/windows.png)",
    )
    p_vis.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose diagnostics"
    )
    p_vis.set_defaults(func=cmd_visualize_windows)
