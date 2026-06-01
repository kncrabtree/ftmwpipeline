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
from .._internal.stage5_validation_impl import validate_stage5_shape_error_impl
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


def cmd_validate_stage5_shape_error(args: argparse.Namespace) -> int:
    """Assess a persisted Stage 5 fit against the SNR-aware framework.

    Read-only. Prints Tier 1 (SNR-aware per-window acceptance, binned by the
    brightest in-window peak SNR), Tier 2 (rescue/merge/thaw gate firing), and,
    when ``--ground-truth`` is given, Tier 3 (known-line recall/precision,
    frequency accuracy, and the instrument accuracy floor). Requires a completed
    Stage 5 fit; does not modify the file.
    """
    setup_logging(args.verbose)
    try:
        file_path = _ensure_ftmw(args.file_path)
        report = validate_stage5_shape_error_impl(
            file_path=file_path,
            kappa=args.kappa,
            noise_floor=args.noise_floor,
            ground_truth=args.ground_truth,
            match_tol_fwhm=args.match_tol_fwhm,
        )
        _print_validation_report(report)
        return 0
    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        return 1
    except ValueError as e:
        print_error(f"Invalid parameters or missing Stage 5 fit: {e}")
        print("Hint: run 'fit-peaks' first")
        return 1
    except Exception as e:
        print_error(f"Stage 5 validation failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def _print_validation_report(report: dict) -> None:
    """Render the validate-stage5-shape-error report as text."""
    params = report["parameters"]
    t1 = report["tier1"]
    print(
        f"\nStage 5 shape-error validation (shape={params['shape']}, "
        f"kappa={params['kappa']:.3g}, F={params['noise_floor']:.3g})"
    )
    print(
        f"  Tier 1 (SNR-aware, chi2r <= F + (kappa*SNR_max)^2): "
        f"{t1.get('n_pass', 0)}/{t1.get('n_windows', 0)} windows pass "
        f"(rate {t1.get('pass_rate', 0.0):.3f})"
    )
    print(
        f"    {'SNR_max bin':>11} {'n':>5} {'chi2r_med':>10} "
        f"{'eps_med':>9} {'pass_rate':>10}"
    )
    for b in t1.get("snr_bins", []):
        med = b["chi2r_median"]
        print(
            f"    {b['snr_bin']:>11} {b['n']:>5} "
            f"{(med if med is not None else float('nan')):>10.3f} "
            f"{b['epsilon_median']*100:>8.2f}% {b['pass_rate']:>10.3f}"
        )

    t2 = report["tier2"]
    print(
        f"  Tier 2 (gate firing): merge {t2['merge_fire_rate']:.3f}, "
        f"rescue-origin pruned {t2['n_pruned_rescue_origin']}, "
        f"limit-cycle {t2['limit_cycle_rounds']}, "
        f"rescue {t2['n_rescue_accepted']}/{t2['n_rescue_rounds']}, "
        f"thaw {t2['n_thaw_accepted']}/{t2['n_thaw']}, "
        f"replan {t2['n_replan']}"
    )

    t3 = report.get("tier3")
    if t3:
        print(
            f"  Tier 3 (ground truth {Path(t3['ground_truth']).name}): "
            f"recall {t3['recall']:.3f} ({t3['n_matched']}/{t3['n_catalog']}), "
            f"precision {t3.get('precision', 0.0):.3f} "
            f"(of {t3['n_fitted']} fitted; loose)"
        )
        fr = t3.get("freq_residual_khz")
        if fr:
            print(
                f"    freq residual: median {fr['median']:+.3f} kHz, "
                f"rms {fr['rms']:.3f} kHz, max |.| {fr['max_abs']:.3f} kHz"
            )
        af = t3.get("accuracy_floor")
        if af:
            print(
                f"    accuracy floor (detrended): {af['detrended_rms_khz']:.3f} kHz "
                f"(raw {af['raw_rms_khz']:.3f}, drift "
                f"{af['drift_slope_khz_per_ghz']:+.3f} kHz/GHz) "
                f"-- instrument, not a defect"
            )
        sh = t3.get("sigma_f_honesty")
        if sh:
            print(
                f"    sigma_f honesty: reported median "
                f"{sh['median_reported_sigma_khz']:.3f} kHz, "
                f"residual/sigma median {sh['median_residual_over_sigma']:.2f}"
            )


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

    p_val = subparsers.add_parser(
        "validate-stage5-shape-error",
        help="Assess a Stage 5 fit against the SNR-aware acceptance framework",
        description=(
            "Read-only cross-fixture validation of a completed Stage 5 fit.\n\n"
            "Tier 1 is the SNR-aware acceptance gate: a window passes iff\n"
            "chi2r <= F + (kappa*SNR_max)^2 (F the noise-regime allowance), with\n"
            "the fractional model deficit eps = sqrt(max(chi2r-F,0))/SNR_max\n"
            "reported and binned by the brightest in-window peak SNR. At extreme\n"
            "SNR the per-window\n"
            "reduced chi-squared is a model-fidelity floor, not a noise\n"
            "statistic, so the raw chi2r gate is meaningless. Tier 2 reports\n"
            "rescue/merge/thaw gate firing. With --ground-truth, Tier 3 matches\n"
            "fitted lines to a catalog CSV and reports recall, frequency\n"
            "accuracy, and the instrument accuracy floor. Does not modify the\n"
            "file. Run 'fit-peaks' first."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_val.add_argument(
        "file_path", help="Path to .ftmw file with Stage 5 results"
    )
    p_val.add_argument(
        "--kappa",
        dest="kappa",
        type=float,
        default=None,
        help="Tolerated fractional model deficit for the SNR-aware gate "
        "(default 0.05, just above the measured ~1-3%% vinyl-cyanide deficit).",
    )
    p_val.add_argument(
        "--noise-floor",
        dest="noise_floor",
        type=float,
        default=None,
        help="Noise-regime allowance F in chi2r <= F + (kappa*SNR_max)^2 "
        "(default 3.0; budgets for the reduced-chi2 sampling scatter of a good "
        "fit at low SNR, where the deficit term is negligible).",
    )
    p_val.add_argument(
        "--ground-truth",
        dest="ground_truth",
        type=str,
        default=None,
        metavar="CSV",
        help="Catalog CSV (a 'freq_mhz' column) to run Tier 3 against.",
    )
    p_val.add_argument(
        "--match-tol-fwhm",
        dest="match_tol_fwhm",
        type=float,
        default=0.5,
        help="Tier-3 match tolerance in units of the per-window line FWHM "
        "(default 0.5).",
    )
    p_val.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose diagnostics"
    )
    p_val.set_defaults(func=cmd_validate_stage5_shape_error)
