"""CLI subcommands for Stage 2b: data-driven tau calibration.

Three subcommands:

- ``calibrate-tau``: runs the STFT calibration and persists the result.
- ``visualize-tau-heatmap``: 2D STFT magnitude across (frame x molecular freq).
- ``visualize-tau-distribution``: tau histogram + tau-vs-SNR + tau-vs-freq +
  GMM overlay.

All three delegate to the shared :mod:`_internal.stage2b_impl` orchestration
layer per the dual-interface rule.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .._internal.stage2b_g_impl import calibrate_tau_G_impl
from .._internal.stage2b_impl import calibrate_tau_impl, load_tau_calibration_impl
from .utils import print_error, setup_logging

logger = logging.getLogger(__name__)


def cmd_calibrate_tau(args) -> int:
    """Run the STFT tau calibration and persist the result."""
    setup_logging(args.verbose)
    file_path = args.file_path
    if not file_path.endswith(".ftmw"):
        file_path = file_path + ".ftmw"

    kwargs: Dict[str, Any] = {}
    if args.n_seg is not None:
        kwargs["n_seg"] = int(args.n_seg)
    if args.t_sigma is not None:
        kwargs["t_sigma"] = float(args.t_sigma)
    if args.tau_max_us is not None:
        kwargs["tau_max_us"] = float(args.tau_max_us)
    if args.rss_gate_factor is not None:
        kwargs["rss_gate_factor"] = float(args.rss_gate_factor)
    if args.sigma_time is not None:
        kwargs["sigma_time"] = float(args.sigma_time)
    if args.min_contributors is not None:
        kwargs["min_contributors"] = int(args.min_contributors)
    if args.sigma_tau_fraction_max is not None:
        kwargs["sigma_tau_fraction_max"] = float(args.sigma_tau_fraction_max)
    if args.bimodality_dominant_fraction is not None:
        kwargs["bimodality_dominant_fraction"] = float(
            args.bimodality_dominant_fraction
        )
    if args.preset is not None:
        kwargs["preset"] = args.preset

    print(f"Running STFT tau calibration for: {file_path}")
    try:
        result = calibrate_tau_impl(file_path, **kwargs)
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
    print("Use 'visualize-tau-heatmap' / 'visualize-tau-distribution' for diagnostics.")
    return 0


def cmd_calibrate_tau_G(args) -> int:
    """Run the STFT Gaussian-shape τ_G calibration and persist the result."""
    setup_logging(args.verbose)
    file_path = args.file_path
    if not file_path.endswith(".ftmw"):
        file_path = file_path + ".ftmw"

    kwargs: Dict[str, Any] = {}
    if args.n_seg is not None:
        kwargs["n_seg"] = int(args.n_seg)
    if args.t_sigma is not None:
        kwargs["t_sigma"] = float(args.t_sigma)
    if args.tau_max_us is not None:
        kwargs["tau_max_us"] = float(args.tau_max_us)
    if args.rss_gate_factor is not None:
        kwargs["rss_gate_factor"] = float(args.rss_gate_factor)
    if args.sigma_time is not None:
        kwargs["sigma_time"] = float(args.sigma_time)
    if args.snr_min is not None:
        kwargs["snr_min"] = float(args.snr_min)
    if args.tau_g_bound_lo is not None:
        kwargs["tau_G_bound_lo"] = float(args.tau_g_bound_lo)
    if args.tau_g_bound_hi is not None:
        kwargs["tau_G_bound_hi"] = float(args.tau_g_bound_hi)
    if args.delta_chi2r_min is not None:
        kwargs["delta_chi2r_min"] = float(args.delta_chi2r_min)
    if args.tau_g_upper_fraction is not None:
        kwargs["tau_G_upper_fraction"] = float(args.tau_g_upper_fraction)
    if args.min_contributors is not None:
        kwargs["min_contributors"] = int(args.min_contributors)
    if args.sigma_tau_fraction_max is not None:
        kwargs["sigma_tau_fraction_max"] = float(args.sigma_tau_fraction_max)
    if args.bimodality_dominant_fraction is not None:
        kwargs["bimodality_dominant_fraction"] = float(
            args.bimodality_dominant_fraction
        )
    if args.preset is not None:
        kwargs["preset"] = args.preset

    print(f"Running STFT τ_G (Gaussian-shape) calibration for: {file_path}")
    try:
        result = calibrate_tau_G_impl(file_path, **kwargs)
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


def cmd_visualize_tau_heatmap(args) -> int:
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


def cmd_visualize_tau_distribution(args) -> int:
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


def _handle_figure_output(fig, args) -> int:
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


def register_tau_commands(subparsers) -> None:
    """Register Stage 2b CLI subcommands on the parent subparsers."""
    # --- calibrate-tau -----------------------------------------------------
    parser_cal = subparsers.add_parser(
        "calibrate-tau",
        help="Run the STFT tau calibration (Stage 2b)",
        description=(
            "Extract a data-driven majority-vote molecular decay constant "
            "(tau_maj) and its robust spread (sigma_tau) from the raw FID "
            "via the sliding-active-window STFT. Persists the result to "
            "/stage2b_tau_calibration in the .ftmw file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_cal.add_argument(
        "file_path", help="Path to .ftmw pipeline file (extension added if missing)"
    )
    parser_cal.add_argument(
        "--n-seg",
        type=int,
        help="Number of non-overlapping STFT frames (default 10)",
    )
    parser_cal.add_argument(
        "--t-sigma",
        type=float,
        help="Above-threshold gate factor on per-frame SNR (default 5.0)",
    )
    parser_cal.add_argument(
        "--tau-max-us",
        type=float,
        help="Saturation cap on recovered tau (default 5 * T_full)",
    )
    parser_cal.add_argument(
        "--rss-gate-factor",
        type=float,
        help="Bad-fit gate strength (default 5.0)",
    )
    parser_cal.add_argument(
        "--sigma-time",
        type=float,
        help=(
            "Time-domain sigma_t override; default measures from the FID "
            "active-region tail."
        ),
    )
    parser_cal.add_argument(
        "--min-contributors",
        type=int,
        help="Pre-condition minimum contributor count (default 200)",
    )
    parser_cal.add_argument(
        "--sigma-tau-fraction-max",
        type=float,
        help="Pre-condition sigma_tau/tau_maj upper bound (default 0.20)",
    )
    parser_cal.add_argument(
        "--bimodality-dominant-fraction",
        type=float,
        help=(
            "Pre-condition floor on dominant-cluster weight when the GMM "
            "prefers two components (default 0.70)"
        ),
    )
    parser_cal.add_argument(
        "--preset",
        type=str,
        default=None,
        help=(
            "Stage 2b preset to apply (bare packaged name or path to a "
            "YAML file). Mutually exclusive with per-knob flags that "
            "explicitly set the same field."
        ),
    )
    parser_cal.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser_cal.set_defaults(func=cmd_calibrate_tau)

    # --- calibrate-tau-G ----------------------------------------------------
    parser_cg = subparsers.add_parser(
        "calibrate-tau-G",
        help="Run the STFT Gaussian-shape τ_G calibration (Stage 2b twin)",
        description=(
            "Per-bin Voigt fits on the STFT contributor pool yield a "
            "per-band τ_G majority that the Stage 5 Gaussian-shape fit "
            "consumes. Persists to /stage2b_tau_G_calibration. Independent "
            "of the pure-exp calibrate-tau; both can coexist on one .ftmw "
            "file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_cg.add_argument(
        "file_path",
        help="Path to .ftmw pipeline file (extension added if missing)",
    )
    parser_cg.add_argument(
        "--n-seg",
        type=int,
        help="Number of non-overlapping STFT frames (default 10)",
    )
    parser_cg.add_argument(
        "--t-sigma",
        type=float,
        help="Above-threshold gate factor on per-frame SNR (default 5.0)",
    )
    parser_cg.add_argument(
        "--tau-max-us",
        type=float,
        help="Saturation cap on the STFT classifier (default 5 * T_full)",
    )
    parser_cg.add_argument(
        "--rss-gate-factor",
        type=float,
        help="Bad-fit gate strength (default 5.0)",
    )
    parser_cg.add_argument(
        "--sigma-time",
        type=float,
        help=(
            "Time-domain sigma_t override; default measures from the FID "
            "active-region tail."
        ),
    )
    parser_cg.add_argument(
        "--snr-min",
        type=float,
        help="Per-bin SNR floor on the Voigt-fit contributor pool (default 20.0)",
    )
    parser_cg.add_argument(
        "--tau-g-bound-lo",
        type=float,
        help="Lower bound on the Voigt τ_G parameter (default 0.5 us)",
    )
    parser_cg.add_argument(
        "--tau-g-bound-hi",
        type=float,
        help="Upper bound on the Voigt τ_G parameter (default 100.0 us)",
    )
    parser_cg.add_argument(
        "--delta-chi2r-min",
        type=float,
        help=(
            "Minimum χ²ᵣ improvement (pure-exp − Voigt) for a bin to enter "
            "the calibration (default 1.0)"
        ),
    )
    parser_cg.add_argument(
        "--tau-g-upper-fraction",
        type=float,
        help=("Bins whose τ_G ≥ fraction*tau_g_bound_hi are dropped (default 0.7)"),
    )
    parser_cg.add_argument(
        "--min-contributors",
        type=int,
        help="Pre-condition minimum eligible bin count (default 50)",
    )
    parser_cg.add_argument(
        "--sigma-tau-fraction-max",
        type=float,
        help="Pre-condition sigma_tau_G/tau_G_maj upper bound (default 0.20)",
    )
    parser_cg.add_argument(
        "--bimodality-dominant-fraction",
        type=float,
        help=(
            "Pre-condition floor on dominant-cluster weight when the GMM "
            "prefers two components (default 0.70)"
        ),
    )
    parser_cg.add_argument(
        "--preset",
        type=str,
        default=None,
        help=(
            "Stage 2b preset to apply (bare packaged name or path to a "
            "YAML file). The same preset block drives both τ twins."
        ),
    )
    parser_cg.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser_cg.set_defaults(func=cmd_calibrate_tau_G)

    # --- visualize-tau-heatmap ---------------------------------------------
    parser_heat = subparsers.add_parser(
        "visualize-tau-heatmap",
        help="2D STFT magnitude heatmap (frame x molecular frequency)",
        description=(
            "Plot log10 |S_n(f)| across the n_seg STFT frames and the trim "
            "frequency range. Streaks at constant magnitude vs frame index "
            "are clock spurs; exponential-decay streaks are real molecular "
            "lines."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_heat.add_argument(
        "file_path", help="Path to .ftmw pipeline file with Stage 2b completed"
    )
    parser_heat.add_argument(
        "-o",
        "--output",
        type=str,
        help="Save plot to file instead of displaying interactively",
    )
    parser_heat.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser_heat.set_defaults(func=cmd_visualize_tau_heatmap)

    # --- visualize-tau-distribution ----------------------------------------
    parser_dist = subparsers.add_parser(
        "visualize-tau-distribution",
        help="tau histogram + tau-vs-SNR + tau-vs-freq + GMM overlay",
        description=(
            "Plot the contributor tau histogram with the majority-vote "
            "tau_maj overlay, plus the per-bin tau vs SNR and tau vs "
            "molecular frequency scatters, and the 1- vs 2-component GMM "
            "fit. The two-third partition of the frequency range is "
            "annotated when present."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_dist.add_argument(
        "file_path", help="Path to .ftmw pipeline file with Stage 2b completed"
    )
    parser_dist.add_argument(
        "-o",
        "--output",
        type=str,
        help="Save plot to file instead of displaying interactively",
    )
    parser_dist.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser_dist.set_defaults(func=cmd_visualize_tau_distribution)
