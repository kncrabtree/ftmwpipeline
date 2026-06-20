"""CLI subcommands for Stage 2b: data-driven tau calibration.

Two verbs on the ``tau`` object:

- ``tau run`` (``--gaussian`` for the τ_G twin): runs the STFT calibration and
  persists the result.
- ``tau show --kind heatmap``: 2D STFT magnitude across (frame x molecular freq).
- ``tau show --kind distribution``: tau histogram + tau-vs-SNR + tau-vs-freq +
  GMM overlay.

All delegate to the shared :mod:`_internal.stage2b_impl` orchestration
layer per the dual-interface rule.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Optional

from .._internal.stage2b_g_impl import calibrate_tau_G_impl
from .._internal.stage2b_impl import calibrate_tau_impl, load_tau_calibration_impl
from ..core.tau_calibration_settings import TauCalibrationSettings
from ._argspec import add_settings_args, settings_from_namespace
from .utils import add_stage_object, print_error, setup_logging

logger = logging.getLogger(__name__)


def cmd_calibrate_tau(args: argparse.Namespace) -> int:
    """Run the STFT tau calibration and persist the result."""
    setup_logging(args.verbose)
    file_path = args.file_path
    if not file_path.endswith(".ftmw"):
        file_path = file_path + ".ftmw"

    # The per-knob flags are generated from TauCalibrationSettings field
    # metadata; reconstruct a sparse settings bundle (unset fields fall through
    # the resolver). A preset and per-knob flags compose: the flags are the
    # explicit layer, the preset the preset layer beneath the persisted one.
    settings = settings_from_namespace(args, TauCalibrationSettings)
    preset = args.preset

    print(f"Running STFT tau calibration for: {file_path}")
    try:
        result = calibrate_tau_impl(
            file_path,
            settings=None if settings.is_empty() else settings,
            preset=preset,
        )
    except FileNotFoundError as e:
        print_error(f"Pipeline file not found: {e}")
        return 1
    except ValueError as e:
        print_error(f"Tau calibration failed: {e}")
        return 1
    except Exception as e:
        print_error(f"Tau calibration failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    tc = result["tau_calibration"]
    print("\nTau calibration completed successfully!")
    print("\nResults summary:")
    print(f"  tau_maj            : {tc.tau_maj_us:.3f} us")
    print(
        f"  sigma_tau          : {tc.sigma_tau_us:.3f} us  "
        f"(spread/tau_maj = {tc.sigma_tau_us / tc.tau_maj_us:.3f})"
    )
    print(f"  contributors       : {tc.n_contributors}")
    print(f"  spur bins (raw)    : {tc.n_spur_bins}")
    print(f"  spur clusters      : {len(tc.spur_clusters)}")
    print(
        f"  bimodal (GMM)      : {tc.bimodality.two_component_preferred} "
        f"(delta_aic = {tc.bimodality.delta_aic:.1f})"
    )
    print(f"  preconditions pass : {tc.preconditions_passed}")
    if not tc.preconditions_passed:
        for note in tc.preconditions_notes:
            if note != "ok":
                print(f"    - {note}")
    if result.get("invalidated_stages"):
        print(
            "\nInvalidated downstream stages: "
            + ", ".join(result["invalidated_stages"])
        )
    print(f"\nResults saved to: {file_path}")
    print("Use 'tau show --kind heatmap|distribution' for diagnostics.")
    return 0


def cmd_calibrate_tau_G(args: argparse.Namespace) -> int:
    """Run the STFT Gaussian-shape τ_G calibration and persist the result."""
    setup_logging(args.verbose)
    file_path = args.file_path
    if not file_path.endswith(".ftmw"):
        file_path = file_path + ".ftmw"

    # The per-knob flags are generated from TauCalibrationSettings field
    # metadata; reconstruct a sparse settings bundle. The shared
    # --min-contributors flag lands on aggregation.min_contributors, but the
    # τ_G twin's precondition reads gaussian.min_contributors -- route it
    # across before the empty-check so an only-min_contributors call still
    # passes settings through.
    settings = settings_from_namespace(args, TauCalibrationSettings)
    if settings.aggregation.min_contributors is not None:
        settings.gaussian.min_contributors = settings.aggregation.min_contributors
        settings.aggregation.min_contributors = None
    preset = args.preset

    print(f"Running STFT τ_G (Gaussian-shape) calibration for: {file_path}")
    try:
        result = calibrate_tau_G_impl(
            file_path,
            settings=None if settings.is_empty() else settings,
            preset=preset,
        )
    except FileNotFoundError as e:
        print_error(f"Pipeline file not found: {e}")
        return 1
    except ValueError as e:
        print_error(f"τ_G calibration failed: {e}")
        return 1
    except Exception as e:
        print_error(f"τ_G calibration failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    tc = result["tau_G_calibration"]
    spread = tc.sigma_tau_us / tc.tau_maj_us if tc.tau_maj_us > 0 else float("nan")
    print("\nτ_G calibration completed successfully!")
    print("\nResults summary:")
    print(f"  tau_G_maj          : {tc.tau_maj_us:.3f} us")
    print(
        f"  sigma_tau_G        : {tc.sigma_tau_us:.3f} us  "
        f"(spread/tau_G_maj = {spread:.3f})"
    )
    print(f"  eligible bins      : {tc.n_contributors}")
    print(
        f"  bimodal (GMM)      : {tc.bimodality.two_component_preferred} "
        f"(delta_aic = {tc.bimodality.delta_aic:.1f})"
    )
    print(f"  preconditions pass : {tc.preconditions_passed}")
    if not tc.preconditions_passed:
        for note in tc.preconditions_notes:
            if note != "ok":
                print(f"    - {note}")
    if result.get("invalidated_stages"):
        print(
            "\nInvalidated downstream stages: "
            + ", ".join(result["invalidated_stages"])
        )
    print(f"\nResults saved to: {file_path}")
    return 0


def cmd_visualize_tau_heatmap(args: argparse.Namespace) -> int:
    """Render the 2D STFT magnitude heatmap (figure 08 in the research dir)."""
    setup_logging(args.verbose)
    file_path = args.file_path
    if not file_path.endswith(".ftmw"):
        file_path = file_path + ".ftmw"

    from ..visualization.tau_calibration_visualization import (
        plot_tau_heatmap_from_file,
    )

    try:
        fig = plot_tau_heatmap_from_file(file_path)
    except Exception as e:
        print_error(f"Failed to create tau-heatmap visualization: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    return _handle_figure_output(fig, args)


def cmd_visualize_tau_distribution(args: argparse.Namespace) -> int:
    """Render the tau-distribution analysis (figure 09 in the research dir)."""
    setup_logging(args.verbose)
    file_path = args.file_path
    if not file_path.endswith(".ftmw"):
        file_path = file_path + ".ftmw"

    from ..visualization.tau_calibration_visualization import (
        plot_tau_distribution_from_file,
    )

    try:
        fig = plot_tau_distribution_from_file(file_path)
    except Exception as e:
        print_error(f"Failed to create tau-distribution visualization: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    return _handle_figure_output(fig, args)


def _handle_figure_output(fig: Any, args: argparse.Namespace) -> int:
    """Save or display a matplotlib figure based on CLI flags."""
    output: Optional[str] = getattr(args, "output", None)
    interactive = not output
    if output:
        try:
            fig.savefig(output, dpi=150, bbox_inches="tight")
            print(f"Plot saved to: {output}")
        except Exception as e:
            print_error(f"Failed to save plot: {e}")
            return 1
    elif interactive:
        try:
            import matplotlib.pyplot as plt

            plt.show()
        except Exception as e:
            print_error(f"Could not display plot: {e}")
            return 1
    return 0


def _cmd_tau_run(args: argparse.Namespace) -> int:
    """Dispatch ``tau run``: ``--gaussian`` selects the τ_G twin."""
    if getattr(args, "gaussian", False):
        return cmd_calibrate_tau_G(args)
    return cmd_calibrate_tau(args)


def _cmd_tau_show(args: argparse.Namespace) -> int:
    """Dispatch ``tau show``: ``--kind`` selects heatmap (default) or distribution."""
    if getattr(args, "kind", "heatmap") == "distribution":
        return cmd_visualize_tau_distribution(args)
    return cmd_visualize_tau_heatmap(args)


def register_tau_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register tau calibration (Stage 2b) object-verb subcommands.

    The pure-exp and Gaussian-shape calibrations are unified under ``tau run``
    (``--gaussian`` selects the τ_G twin), and the heatmap / distribution
    diagnostics under ``tau show --kind``.
    """
    verbs = add_stage_object(
        subparsers,
        "tau",
        synonym="stage2b",
        help="Stage 2b: tau calibration (run / show)",
        description="Calibrate the molecular decay constant and view diagnostics.",
    )

    # --- tau run -----------------------------------------------------------
    parser_cal = verbs.add_parser(
        "run",
        help="Run the STFT tau calibration (Stage 2b; --gaussian for the τ_G twin)",
        description=(
            "Extract a data-driven majority-vote molecular decay constant "
            "(tau_maj) and its robust spread (sigma_tau) from the raw FID "
            "via the sliding-active-window STFT. Persists the result to "
            "/stage2b_tau_calibration in the .ftmw file.\n\n"
            "With --gaussian, runs the Gaussian-shape twin instead: per-bin "
            "Voigt fits on the STFT contributor pool yield a per-band tau_G "
            "majority, persisted to /stage2b_tau_G_calibration. The two are "
            "independent and can coexist on one .ftmw file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_cal.add_argument(
        "file_path", help="Path to .ftmw pipeline file (extension added if missing)"
    )
    parser_cal.add_argument(
        "--gaussian",
        action="store_true",
        help="Run the Gaussian-shape τ_G calibration twin (Voigt-fit per-band "
        "tau_G) instead of the pure-exp calibration",
    )
    # Per-knob flags, generated from TauCalibrationSettings field metadata (the
    # single declaration site shared with `settings` / `scan`). The Gaussian-only
    # knobs (--snr-min / --tau-g-bound-* / --delta-chi2r-min /
    # --tau-g-upper-fraction) are consumed only with --gaussian.
    add_settings_args(parser_cal, TauCalibrationSettings)
    parser_cal.add_argument(
        "--preset",
        type=str,
        default=None,
        help=(
            "Stage 2b preset to apply (bare packaged name or path to a "
            "YAML file). Composes with per-knob flags: the flags are the "
            "explicit layer, the preset the layer beneath the persisted "
            "record."
        ),
    )
    parser_cal.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser_cal.set_defaults(func=_cmd_tau_run)

    # --- tau show ----------------------------------------------------------
    parser_show = verbs.add_parser(
        "show",
        help="Stage 2b diagnostics (--kind heatmap | distribution)",
        description=(
            "--kind heatmap (default): 2D STFT magnitude heatmap (frame x "
            "molecular frequency). Streaks at constant magnitude vs frame "
            "index are clock spurs; exponential-decay streaks are real "
            "molecular lines.\n\n"
            "--kind distribution: the contributor tau histogram with the "
            "majority-vote tau_maj overlay, plus per-bin tau vs SNR and tau "
            "vs molecular frequency scatters and the 1- vs 2-component GMM "
            "fit."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_show.add_argument(
        "file_path", help="Path to .ftmw pipeline file with Stage 2b completed"
    )
    parser_show.add_argument(
        "--kind",
        choices=["heatmap", "distribution"],
        default="heatmap",
        help="Which diagnostic to plot (default: heatmap)",
    )
    parser_show.add_argument(
        "-o",
        "--output",
        type=str,
        help="Save plot to file instead of displaying interactively",
    )
    parser_show.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser_show.set_defaults(func=_cmd_tau_show)
