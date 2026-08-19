"""CLI subcommands for scope-timebase self-calibration.

Three verbs on the ``timebase`` object:

- ``timebase run``: demodulates the active FID at the Rb-locked clock spur
  lattice, fits the shared fractional scale error ``eps``, and persists it.
- ``timebase show``: prints the persisted summary plus the per-tone table
  (kept, rejected, and drift-control sections).
- ``timebase state``: prints the *derived* frequency calibration the file is
  under -- which frame its frequencies are in, and by how much they are
  corrected. Has no ``run``/``show`` form: it is derived, never persisted, and
  ``show`` is the persisted measurement's per-tone table.

All delegate to the shared ``_internal`` orchestration layer per the
dual-interface rule. No plotting in this surface.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict

from .._internal.stage6_impl import frequency_calibration_impl
from .._internal.timebase_impl import (
    calibrate_timebase_impl,
    load_timebase_calibration_impl,
)
from .utils import add_stage_object, print_error, setup_logging

logger = logging.getLogger(__name__)


def cmd_calibrate_timebase(args: argparse.Namespace) -> int:
    """Run the timebase self-calibration and persist the result."""
    setup_logging(args.verbose)
    file_path = args.file_path
    if not file_path.endswith(".ftmw"):
        file_path = file_path + ".ftmw"

    kwargs: Dict[str, Any] = {}
    if args.kappa_sys is not None:
        kwargs["kappa_sys"] = float(args.kappa_sys)
    if args.snr_min is not None:
        kwargs["snr_min"] = float(args.snr_min)

    print(f"Running scope-timebase self-calibration for: {file_path}")
    try:
        result = calibrate_timebase_impl(file_path, **kwargs)
    except FileNotFoundError as e:
        print_error(f"Pipeline file not found: {e}")
        return 1
    except ValueError as e:
        print_error(f"Timebase calibration failed: {e}")
        return 1
    except Exception as e:
        print_error(f"Timebase calibration failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    tc = result["timebase_calibration"]
    print("\nTimebase calibration completed successfully!")
    print("\nResults summary:")
    print(
        f"  epsilon            : {tc.epsilon * 1e6:+.3f} +- "
        f"{tc.sigma_epsilon * 1e6:.3f} ppm"
    )
    print(f"  lattice g          : {tc.lattice_g_mhz:.1f} MHz")
    print(f"  tones used         : {tc.n_used} / {tc.n_detected} detected")
    print(f"  preconditions pass : {tc.preconditions_passed}")
    if not tc.preconditions_passed:
        for note in tc.preconditions_notes:
            if note != "ok":
                print(f"    - {note}")
    print(f"\nResults saved to: {file_path}")
    print("Use 'timebase show' for the per-tone table.")
    return 0


def cmd_show_timebase(args: argparse.Namespace) -> int:
    """Print the persisted timebase summary and per-tone table."""
    setup_logging(args.verbose)
    file_path = args.file_path
    if not file_path.endswith(".ftmw"):
        file_path = file_path + ".ftmw"

    try:
        loaded = load_timebase_calibration_impl(file_path)
    except FileNotFoundError as e:
        print_error(f"Pipeline file not found: {e}")
        return 1
    except ValueError as e:
        print_error(str(e))
        return 1
    except Exception as e:
        print_error(f"Failed to load timebase calibration: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    tc = loaded["timebase_calibration"]
    print(f"Timebase calibration for: {file_path}")
    print(f"  created            : {loaded.get('creation_time', 'unknown')}")
    print(
        f"  epsilon            : {tc.epsilon * 1e6:+.3f} +- "
        f"{tc.sigma_epsilon * 1e6:.3f} ppm"
    )
    print(f"  lattice g          : {tc.lattice_g_mhz:.1f} MHz")
    print(f"  tones used         : {tc.n_used} / {tc.n_detected} detected")
    print(f"  kappa_sys          : {tc.kappa_sys * 1e6:.3f} ppm")
    print(f"  snr_min            : {tc.snr_min:.1f}")
    print(f"  active region      : [{tc.start_us:.3f}, {tc.end_us:.3f}) us")
    print(f"  preconditions pass : {tc.preconditions_passed}")
    if not tc.preconditions_passed:
        for note in tc.preconditions_notes:
            if note != "ok":
                print(f"    - {note}")

    by_snr = sorted(tc.tone_reads, key=lambda t: -t.snr)
    kept = [t for t in by_snr if t.used]
    rejected = [t for t in by_snr if not t.used and not t.drift_control]
    controls = [t for t in by_snr if t.drift_control]

    def _print_section(title: str, rows: Any) -> None:
        print(f"\n{title} ({len(rows)}):")
        if not rows:
            print("  (none)")
            return
        print(
            f"  {'bb (MHz)':>10}  {'k':>4}  {'df (kHz)':>9}  "
            f"{'sig (kHz)':>9}  {'p/n':>8}  {'eps (ppm)':>10}"
        )
        for t in rows:
            eps_tone = (t.df_mhz / t.f_bb_mhz) * 1e6 if t.f_bb_mhz else float("nan")
            print(
                f"  {t.f_bb_mhz:>10.1f}  {t.k:>4d}  {t.df_mhz * 1e3:>+9.2f}  "
                f"{t.sigma_mhz * 1e3:>9.2f}  {t.snr:>8.1f}  {eps_tone:>+10.3f}"
            )

    _print_section("Kept lattice tones", kept)
    _print_section("Rejected lattice tones", rejected)
    _print_section("Drift controls (unlocked clocks; not in eps fit)", controls)
    return 0


#: What each derived state means for the frequencies the file reports, printed
#: alongside the state so the reader does not have to look the vocabulary up.
_STATE_NOTES = {
    "rb_locked": (
        "no unlocked clock declared; axis absolutely calibrated as acquired "
        "(eps is a null op)"
    ),
    "self_calibrated": (
        "unlocked digitizer with a measured timebase calibration applied; "
        "raw and calibrated frames differ"
    ),
    "uncalibrated": (
        "unlocked digitizer with no usable timebase calibration; frequencies "
        "reported as acquired and caveated -- run 'timebase run'"
    ),
}


def cmd_timebase_state(args: argparse.Namespace) -> int:
    """Print the derived frequency-calibration state (never mutates)."""
    setup_logging(args.verbose)
    file_path = args.file_path
    if not file_path.endswith(".ftmw"):
        file_path = file_path + ".ftmw"

    try:
        stamp = frequency_calibration_impl(file_path)
    except FileNotFoundError as e:
        print_error(str(e))
        return 1
    except Exception as e:
        print_error(f"Failed to read the frequency calibration: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1

    if args.format == "json":
        print(
            json.dumps(
                {
                    "state": stamp.state,
                    "epsilon": stamp.epsilon,
                    "sigma_epsilon": stamp.sigma_epsilon,
                    "sigma_floor_khz": stamp.sigma_floor_khz,
                    "probe_freq_mhz": stamp.probe_freq_mhz,
                    "sideband": stamp.sideband,
                },
                indent=2,
            )
        )
        return 0

    print(f"Frequency calibration for: {file_path}")
    print(f"  state              : {stamp.state}")
    note = _STATE_NOTES.get(stamp.state)
    if note:
        print(f"                       ({note})")
    print(
        f"  epsilon            : {stamp.epsilon * 1e6:+.3f} +- "
        f"{stamp.sigma_epsilon * 1e6:.3f} ppm"
    )
    print(f"  sigma floor        : {stamp.sigma_floor_khz:.3f} kHz")
    if stamp.probe_freq_mhz is None:
        print("  probe / sideband   : (no FID header; no frame conversion possible)")
    else:
        print(f"  probe frequency    : {stamp.probe_freq_mhz:.6f} MHz")
        print(f"  sideband           : {stamp.sideband}")
    return 0


def register_timebase_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register scope-timebase self-calibration object-verb subcommands."""
    verbs = add_stage_object(
        subparsers,
        "timebase",
        help="Scope-timebase self-calibration (run / show / state)",
        description=(
            "Measure the digitizer-clock fractional scale error eps from the "
            "Rb-locked spur lattice and view the per-tone diagnostics."
        ),
    )

    # --- timebase run ------------------------------------------------------
    parser_run = verbs.add_parser(
        "run",
        help="Measure eps from the Rb-locked spur lattice and persist it",
        description=(
            "Demodulate the active FID at each Rb-locked clock-lattice "
            "frequency (multiples of the locked fundamentals' GCD), read each "
            "tone's residual offset by an ML fine-frequency scan, and fit the "
            "shared fractional scale error eps (a baseband tone reads "
            "f_bb_true * (1 + eps)) with consistency outlier rejection. The "
            "clock declaration is taken from the persisted Stage 5 spur.clocks; "
            "declare it first via 'settings set' or a fit preset. Persists eps "
            "to /timebase_calibration. Measures eps only -- applying it to the "
            "frequency axis is out of scope.\n\n"
            "eps is a digitizer-clock error, so it scales the BASEBAND "
            "frequency: the molecular-frame correction is "
            "f_corr = probe + (f_raw - probe)/(1 + eps), not f_raw/(1 + eps). "
            "The two differ by probe_freq * eps/(1 + eps)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_run.add_argument(
        "file_path", help="Path to .ftmw pipeline file (extension added if missing)"
    )
    parser_run.add_argument(
        "--kappa-sys",
        type=float,
        help="Systematic per-tone fractional floor folded into sigma_tot "
        "(default 0.2e-6)",
    )
    parser_run.add_argument(
        "--snr-min",
        type=float,
        help="Peak/noise gate for a tone to count as detected (default 8.0)",
    )
    parser_run.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser_run.set_defaults(func=cmd_calibrate_timebase)

    # --- timebase show -----------------------------------------------------
    parser_show = verbs.add_parser(
        "show",
        help="Print the persisted timebase summary and per-tone table",
        description=(
            "Print eps +- sigma (ppm), the lattice spacing, the kept/detected "
            "counts, and the per-tone table sorted by SNR with kept, rejected, "
            "and drift-control sections."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_show.add_argument(
        "file_path",
        help="Path to .ftmw pipeline file with a timebase calibration",
    )
    parser_show.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser_show.set_defaults(func=cmd_show_timebase)

    # --- timebase state ----------------------------------------------------
    parser_state = verbs.add_parser(
        "state",
        help="Print the derived frequency calibration the file is under",
        description=(
            "Answer, for any file at any stage, what frame its frequencies are "
            "in and by how much they are corrected: the derived calibration "
            "state (rb_locked / self_calibrated / uncalibrated), the eps +- "
            "sigma that will be applied, the declared systematic accuracy "
            "floor, and the probe/sideband the calibrated frame is defined "
            "against.\n\n"
            "The state is DERIVED, never stored -- it follows from the clock "
            "declaration ('clocks show') plus whether a usable timebase "
            "calibration is present -- so it cannot disagree with what the "
            "pipeline will actually apply, and it is readable long before "
            "Stage 6 builds a final-products table. Read-only: this verb "
            "never writes to the file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser_state.add_argument(
        "file_path", help="Path to .ftmw pipeline file (extension added if missing)"
    )
    parser_state.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text)",
    )
    parser_state.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser_state.set_defaults(func=cmd_timebase_state)
