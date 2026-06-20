"""CLI subcommands for Stage 2b: data-driven tau calibration.

Three verbs on the ``tau`` object:

- ``tau run`` (``--gaussian`` for the Gaussian τ_G variant): runs the STFT
  calibration and persists the result.
- ``tau recommend``: runs the 3-way Lorentzian/Gaussian/Voigt shape vote and
  stamps the recommended shape onto the file.
- ``tau show --kind heatmap`` (``--gaussian`` for the τ_G group): 2D STFT
  magnitude across (frame x molecular freq).
- ``tau show --kind distribution``: tau histogram + tau-vs-SNR + tau-vs-freq +
  GMM overlay.

All delegate to the shared :mod:`_internal.stage2b_impl` orchestration layer per
the dual-interface rule.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Optional

from .._internal.shape_recommendation_impl import recommend_shape_impl
from .._internal.stage2b_impl import calibrate_tau_impl
from ..core.tau_calibration_settings import TauCalibrationSettings
from ._argspec import add_settings_args, settings_from_namespace
from .utils import add_stage_object, print_error, setup_logging

logger = logging.getLogger(__name__)


def cmd_calibrate_tau(args: argparse.Namespace) -> int:
    """Run the STFT tau calibration and persist the result.

    ``--gaussian`` selects the pure-Gaussian τ_G variant; the default is the
    pure-exponential (Lorentzian) calibration.
    """
    setup_logging(args.verbose)
    file_path = args.file_path
    if not file_path.endswith(".ftmw"):
        file_path = file_path + ".ftmw"

    gaussian = getattr(args, "gaussian", False)
    shape = "gaussian" if gaussian else "lorentzian"

    # The per-knob flags are generated from TauCalibrationSettings field
    # metadata; reconstruct a sparse settings bundle (unset fields fall through
    # the resolver). A preset and per-knob flags compose: the flags are the
    # explicit layer, the preset the preset layer beneath the persisted one. The
    # shared --min-contributors flag is routed onto the Gaussian block inside the
    # impl, so every interface treats it identically.
    settings = settings_from_namespace(args, TauCalibrationSettings)
    preset = args.preset

    label = "τ_G (Gaussian-shape)" if gaussian else "tau"
    print(f"Running STFT {label} calibration for: {file_path}")
    try:
        result = calibrate_tau_impl(
            file_path,
            shape=shape,
            settings=None if settings.is_empty() else settings,
            preset=preset,
        )
    except FileNotFoundError as e:
        print_error(f"Pipeline file not found: {e}")
        return 1
    except Exception as e:
        print_error(f"Tau calibration failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    tc = result["tau_calibration"]
    spread = tc.sigma_tau_us / tc.tau_maj_us if tc.tau_maj_us > 0 else float("nan")
    tau_name = "tau_G_maj" if gaussian else "tau_maj"
    sigma_name = "sigma_tau_G" if gaussian else "sigma_tau"
    count_name = "eligible bins" if gaussian else "contributors"
    print(f"\n{label} calibration completed successfully!")
    print("\nResults summary:")
    print(f"  {tau_name:<18}: {tc.tau_maj_us:.3f} us")
    print(
        f"  {sigma_name:<18}: {tc.sigma_tau_us:.3f} us  "
        f"(spread/{tau_name} = {spread:.3f})"
    )
    print(f"  {count_name:<18}: {tc.n_contributors}")
    if not gaussian:
        print(f"  {'spur bins (raw)':<18}: {tc.n_spur_bins}")
        print(f"  {'spur clusters':<18}: {len(tc.spur_clusters)}")
    print(
        f"  {'bimodal (GMM)':<18}: {tc.bimodality.two_component_preferred} "
        f"(delta_aic = {tc.bimodality.delta_aic:.1f})"
    )
    print(f"  {'preconditions pass':<18}: {tc.preconditions_passed}")
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
    if not gaussian:
        print("Use 'tau show --kind heatmap|distribution' for diagnostics.")
    return 0


def cmd_recommend_shape(args: argparse.Namespace) -> int:
    """Run the 3-way shape vote and stamp the recommended shape onto the file."""
    setup_logging(args.verbose)
    file_path = args.file_path
    if not file_path.endswith(".ftmw"):
        file_path = file_path + ".ftmw"

    settings = settings_from_namespace(args, TauCalibrationSettings)
    preset = args.preset

    print(f"Running 3-way shape recommendation for: {file_path}")
    try:
        result = recommend_shape_impl(
            file_path,
            settings=None if settings.is_empty() else settings,
            preset=preset,
        )
    except FileNotFoundError as e:
        print_error(f"Pipeline file not found: {e}")
        return 1
    except Exception as e:
        print_error(f"Shape recommendation failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    rec = result["shape_recommendation"]
    rates = rec.vote_rates
    print("\nShape recommendation completed.")
    print("\nResults summary:")
    print(f"  recommended shape  : {rec.recommended_shape}")
    print(
        "  vote rates         : "
        f"exp={rates['exp'] * 100:.1f}%  "
        f"gauss={rates['gauss'] * 100:.1f}%  "
        f"voigt={rates['voigt'] * 100:.1f}%"
    )
    print(f"  contributors       : {rec.n_contributors}")
    if result.get("groups_written"):
        print("  stamped onto       : " + ", ".join(result["groups_written"]))
    print(f"\nResults saved to: {file_path}")
    return 0


def cmd_visualize_tau_heatmap(args: argparse.Namespace) -> int:
    """Render the 2D STFT magnitude heatmap (frame x molecular frequency)."""
    setup_logging(args.verbose)
    file_path = args.file_path
    if not file_path.endswith(".ftmw"):
        file_path = file_path + ".ftmw"

    shape = "gaussian" if getattr(args, "gaussian", False) else "lorentzian"

    from ..visualization.tau_calibration_visualization import (
        plot_tau_heatmap_from_file,
    )

    try:
        fig = plot_tau_heatmap_from_file(file_path, shape=shape)
    except Exception as e:
        print_error(f"Failed to create tau-heatmap visualization: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    return _handle_figure_output(fig, args)


def cmd_visualize_tau_distribution(args: argparse.Namespace) -> int:
    """Render the tau-distribution analysis (histogram + scatters + GMM)."""
    setup_logging(args.verbose)
    file_path = args.file_path
    if not file_path.endswith(".ftmw"):
        file_path = file_path + ".ftmw"

    shape = "gaussian" if getattr(args, "gaussian", False) else "lorentzian"

    from ..visualization.tau_calibration_visualization import (
        plot_tau_distribution_from_file,
    )

    try:
        fig = plot_tau_distribution_from_file(file_path, shape=shape)
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


def _cmd_tau_show(args: argparse.Namespace) -> int:
    """Dispatch ``tau show``: ``--kind`` selects heatmap (default) or distribution."""
    if getattr(args, "kind", "heatmap") == "distribution":
        return cmd_visualize_tau_distribution(args)
    return cmd_visualize_tau_heatmap(args)


def register_tau_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register tau calibration (Stage 2b) object-verb subcommands.

    The pure-exp and Gaussian-shape calibrations are unified under ``tau run``
    (``--gaussian`` selects the τ_G variant), the shape vote under
    ``tau recommend``, and the heatmap / distribution diagnostics under
    ``tau show --kind``.
    """
    verbs = add_stage_object(
        subparsers,
        "tau",
        synonym="stage2b",
        help="Stage 2b: tau calibration (run / recommend / show)",
        description="Calibrate the molecular decay constant and view diagnostics.",
    )

    # --- tau run -----------------------------------------------------------
    parser_cal = verbs.add_parser(
        "run",
        help="Run the STFT tau calibration (Stage 2b; --gaussian for the τ_G variant)",
        description=(
            "Extract a data-driven majority-vote molecular decay constant "
            "(tau_maj) and its robust spread (sigma_tau) from the raw FID "
            "via the sliding-active-window STFT. Persists the result to "
            "/stage2b_tau_calibration in the .ftmw file.\n\n"
            "With --gaussian, runs the Gaussian-shape variant instead: a "
            "per-bin pure-Gaussian fit on the STFT contributor pool yields a "
            "per-band tau_G majority, persisted to /stage2b_tau_G_calibration. "
            "The two are independent and can coexist on one .ftmw file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_cal.add_argument(
        "file_path", help="Path to .ftmw pipeline file (extension added if missing)"
    )
    parser_cal.add_argument(
        "--gaussian",
        action="store_true",
        help="Run the Gaussian-shape τ_G calibration (per-band tau_G) instead of "
        "the pure-exp calibration",
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
    parser_cal.set_defaults(func=cmd_calibrate_tau)

    # --- tau recommend -----------------------------------------------------
    parser_rec = verbs.add_parser(
        "recommend",
        help="Run the 3-way L/G/V shape vote and stamp the recommended shape",
        description=(
            "Run the per-bin 3-way (Lorentzian / Gaussian / Voigt) AICc shape "
            "vote on the STFT contributor pool and stamp the SNR-weighted "
            "winner onto every present Stage 2b group. Stage 3 and Stage 5 read "
            "this recommendation to choose the line shape and the matching tau "
            "calibration. Run automatically as part of 'tau run' when "
            "auto_recommend is on."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_rec.add_argument(
        "file_path", help="Path to .ftmw pipeline file (extension added if missing)"
    )
    add_settings_args(parser_rec, TauCalibrationSettings)
    parser_rec.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Stage 2b preset to apply (bare packaged name or path to a YAML file).",
    )
    parser_rec.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser_rec.set_defaults(func=cmd_recommend_shape)

    # --- tau show ----------------------------------------------------------
    parser_show = verbs.add_parser(
        "show",
        help="Stage 2b diagnostics (--kind heatmap | distribution; --gaussian)",
        description=(
            "--kind heatmap (default): 2D STFT magnitude heatmap (frame x "
            "molecular frequency). Streaks at constant magnitude vs frame "
            "index are clock spurs; exponential-decay streaks are real "
            "molecular lines.\n\n"
            "--kind distribution: the contributor tau histogram with the "
            "majority-vote tau_maj overlay, plus per-bin tau vs SNR and tau "
            "vs molecular frequency scatters and the 1- vs 2-component GMM "
            "fit.\n\n"
            "--gaussian views the Gaussian τ_G group instead of the pure-exp "
            "group."
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
        "--gaussian",
        action="store_true",
        help="View the Gaussian τ_G calibration group instead of the pure-exp group",
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
