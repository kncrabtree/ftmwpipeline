"""
Stage 5 fitting commands.

Implements the ``fit-peaks`` and ``visualize-fit`` subcommands. Thin
wrappers over the shared ``_internal.stage5_impl`` implementation --
identical behaviour to the Pipeline class and functional API.
"""

import argparse
from pathlib import Path
from typing import Any

from .._internal.stage5_impl import fit_peaks_impl, visualize_fit_impl
from .utils import print_error, setup_logging


def _ensure_ftmw(path: str) -> str:
    return path if path.endswith(".ftmw") else path + ".ftmw"


def cmd_fit_peaks(args: argparse.Namespace) -> int:
    """Run Stage 5 fitting on a .ftmw pipeline file.

    Fits each Stage 4 window's lines via the conservative add-one-peak loop
    on the active-portion FT, with the frozen-contributor model carrying
    out-of-band lines, then runs the residual edge-coherence handshake
    (local thaw + structural replan). Persists the result to
    ``/stage5_fitting``.

    Requires Stage 4 (assign-windows) first.
    """
    setup_logging(args.verbose)
    try:
        file_path = _ensure_ftmw(args.file_path)
        print(f"Fitting peaks for: {file_path}")
        result = fit_peaks_impl(
            file_path=file_path,
            tau0_us=args.tau0_us,
            fit_tau=args.fit_tau,
            max_decay_factor=args.max_decay_factor,
            residual_edge_threshold=args.residual_edge_threshold,
            residual_edge_m=args.residual_edge_m,
            max_thaw_rounds=args.max_thaw_rounds,
            max_replan_rounds=args.max_replan_rounds,
            max_residual_rescue_rounds=args.max_residual_rescue_rounds,
            rescue_snr_threshold=args.rescue_snr_threshold,
            tau_maj_override_us=args.tau_maj_override_us,
            sigma_tau_override_us=args.sigma_tau_override_us,
            per_band_tau=args.per_band_tau,
            shape=args.shape,
            preset=args.preset,
        )
        print("\nFitting completed successfully!")
        print(f"  Windows fitted: {result['n_windows']:,}")
        print(f"  Fitted peaks:   {result['n_fitted_peaks']:,}")
        print(
            f"  Thaw events:    {result['n_thaw_accepted']:,} accepted "
            f"of {result['n_thaw_events']:,}"
        )
        if result["n_rescue_events"]:
            print(
                f"  Rescue rounds:  {result['n_rescue_accepted']:,} accepted "
                f"of {result['n_rescue_events']:,} "
                f"(added {result['n_rescue_added']:,} peaks, "
                f"{result['n_rescue_origin_pruned']:,} rescue-origin pruned)"
            )
        print(
            f"  Structural replans: {result['n_replan_accepted']:,} accepted "
            f"of {result['n_replan_events']:,} "
            f"(final plan revision {result['final_plan_revision']})"
        )
        print(f"\nResults saved to: {file_path}")
        print("Use 'visualize-fit' to inspect the fit")
        return 0
    except FileNotFoundError as e:
        print_error(f"Pipeline file not found: {e}")
        return 1
    except ValueError as e:
        print_error(f"Invalid parameters or missing dependencies: {e}")
        print("Hint: run 'assign-windows' first")
        return 1
    except Exception as e:
        print_error(f"Fitting failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def cmd_visualize_fit(args: argparse.Namespace) -> int:
    """Overlay the Stage 5 fit on the spectrum.

    Default is an interactive matplotlib window showing the overview
    (fitted-model overlay on the persisted spectrum). With ``--window-id``,
    draws a per-window detail figure (re/im, magnitude+residual, time
    envelope, audit-trail rendering). Pass ``--no-interactive`` together
    with ``--output`` to save a static image instead.
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

        print(f"Creating fit visualization for: {file_path}")
        fig = visualize_fit_impl(
            file_path=file_path,
            figsize=figsize,
            title=args.title,
            window_id=args.window_id,
            backend="matplotlib",
            interactive=not args.no_interactive,
        )

        if args.output:
            fig.savefig(str(Path(args.output)), dpi=300, bbox_inches="tight")
            print(f"Visualization saved to: {args.output}")
        elif not args.no_interactive:
            import matplotlib.pyplot as plt

            plt.show()
        print("Fit visualization completed successfully!")
        return 0
    except FileNotFoundError as e:
        print_error(f"Pipeline file not found: {e}")
        return 1
    except ValueError as e:
        print_error(f"Invalid parameters or missing dependencies: {e}")
        print("Hint: run 'fit-peaks' first")
        return 1
    except Exception as e:
        print_error(f"Fit visualization failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def register_fitting_commands(subparsers: Any) -> None:
    """Register the fit-peaks and visualize-fit subcommands."""
    p_fit = subparsers.add_parser(
        "fit-peaks",
        help="Fit each Stage 4 window's lines (Stage 5)",
        description=(
            "Stage 5 per-window fitting.\n\n"
            "Fits each Stage 4 window's lines via the conservative\n"
            "add-one-peak loop on the active-portion FT, with the\n"
            "frozen-contributor model carrying out-of-band lines, then\n"
            "runs the residual edge-coherence handshake (local thaw +\n"
            "structural replan). Persists per-peak parameters, the audit\n"
            "trail, the thaw / replan histories, and the parameters used\n"
            "to /stage5_fitting. Run 'assign-windows' first."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_fit.add_argument(
        "file_path", help="Path to .ftmw pipeline file (.ftmw auto-added)"
    )
    p_fit.add_argument(
        "--tau0-us",
        dest="tau0_us",
        type=float,
        help="Starting / default shared decay constant per window (us). "
        "Defaults to the Stage 1 expf_us when set, otherwise to T_active/3.",
    )
    p_fit.add_argument(
        "--no-fit-tau",
        dest="fit_tau",
        action="store_false",
        default=None,
        help="Hold the per-window tau fixed at tau0 (default: free).",
    )
    p_fit.add_argument(
        "--max-decay-factor",
        dest="max_decay_factor",
        type=float,
        help="Tau bound factor k: tau in [tau0/k, tau0*k] (default 5).",
    )
    p_fit.add_argument(
        "--residual-edge-threshold",
        dest="residual_edge_threshold",
        type=float,
        help="S_coh threshold above which a residual edge triggers a thaw.",
    )
    p_fit.add_argument(
        "--residual-edge-m",
        dest="residual_edge_m",
        type=int,
        help="Band width (in active-FT bins) of the residual-edge test.",
    )
    p_fit.add_argument(
        "--max-thaw-rounds",
        dest="max_thaw_rounds",
        type=int,
        help="Maximum local-thaw rounds per window per call.",
    )
    p_fit.add_argument(
        "--max-replan-rounds",
        dest="max_replan_rounds",
        type=int,
        help="Maximum structural-replan rounds per call (0 disables).",
    )
    p_fit.add_argument(
        "--max-residual-rescue-rounds",
        dest="max_residual_rescue_rounds",
        type=int,
        help="Cap on per-window residual-rescue + joint-refit cycles. "
        "Omit to use the calibrated default (currently 5); pass 0 to "
        "disable the rescue pass entirely (escape hatch for diagnostic "
        "re-fits). The rescue is a structural part of the fit and runs "
        "on every window's post-thaw fit by default.",
    )
    p_fit.add_argument(
        "--rescue-snr-threshold",
        dest="rescue_snr_threshold",
        type=float,
        help="Detector SNR threshold (in sigma_c) for rescue candidates "
        "(default 2.5; ignored when --max-residual-rescue-rounds is 0).",
    )
    p_fit.add_argument(
        "--tau-maj-override",
        dest="tau_maj_override_us",
        type=float,
        help="Manual override for Stage 2b tau_maj (us). Must be paired "
        "with --sigma-tau-override; beats any persisted Stage 2b "
        "calibration for this fit. Useful for A/B-ing a hand-tuned tau "
        "anchor or forcing a calibrated tau when Stage 2b has not been "
        "run.",
    )
    p_fit.add_argument(
        "--sigma-tau-override",
        dest="sigma_tau_override_us",
        type=float,
        help="Manual override for Stage 2b sigma_tau (us). Required when "
        "--tau-maj-override is set (atomic pair).",
    )
    p_fit.add_argument(
        "--no-per-band-tau",
        dest="per_band_tau",
        action="store_false",
        default=None,
        help="Skip per-band tau routing and use the Stage 2b band-wide "
        "(tau_maj, sigma_tau) anchor for every window. The default "
        "(per-band routing on) maps each window to its band-local tau "
        "majority -- crucial for wide bands with monotonic horn-coupling "
        "tau (~ 1/f). Omit both this flag and any preset that sets "
        "per_band_tau and the resolver picks True from the hard defaults.",
    )
    p_fit.add_argument(
        "--shape",
        dest="shape",
        choices=("lorentzian", "gaussian"),
        default=None,
        help=(
            "Per-line envelope shape. 'lorentzian' (default) uses "
            "exp(-t/tau); 'gaussian' uses exp(-(t/tau_G)**2). Pass "
            "'gaussian' to consume the Stage 2b tau_G calibration "
            "(calibrate-tau-G) in place of the pure-exp Stage 2b. "
            "Omit to fall through to the resolved StageFitSettings "
            "(preset / persisted / hard default)."
        ),
    )
    p_fit.add_argument(
        "--preset",
        dest="preset",
        default=None,
        metavar="NAME_OR_PATH",
        help=(
            "Load a Stage 5 fit preset by bare name (one of the "
            "packaged presets under ftmwpipeline/presets/, e.g. "
            "'gaussian_default', 'lorentzian_legacy', "
            "'instrument_bc_2638') or by path to a YAML file. The "
            "preset enters the resolution chain at the preset layer; "
            "the per-knob CLI flags above still win per-field. "
            "Mutually exclusive with the api/Pipeline 'settings' "
            "kwarg."
        ),
    )
    p_fit.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose diagnostics"
    )
    p_fit.set_defaults(func=cmd_fit_peaks)

    p_vis = subparsers.add_parser(
        "visualize-fit",
        help="Overlay the Stage 5 fit on the spectrum",
        description=(
            "Diagnostic plot of the Stage 5 fit. Default is the spectrum-\n"
            "wide overview (fitted model overlay + residual magnitude);\n"
            "with --window-id, shows a per-window detail figure (re/im,\n"
            "magnitude+residual, time envelope, audit-trail rendering)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_vis.add_argument("file_path", help="Path to .ftmw file with Stage 5 results")
    p_vis.add_argument("--figsize", type=str, help="'width,height' in inches")
    p_vis.add_argument("--title", type=str, help="Custom plot title")
    p_vis.add_argument(
        "--window-id",
        dest="window_id",
        type=int,
        help="Show a per-window detail figure for this window id.",
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
        help="Save plot to this path (e.g. scratch/fit.png)",
    )
    p_vis.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose diagnostics"
    )
    p_vis.set_defaults(func=cmd_visualize_fit)
