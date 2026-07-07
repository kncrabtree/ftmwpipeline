"""CLI ``run`` command: drive a raw source through the whole pipeline.

A single bare verb that imports a raw source and runs every stage in sequence
(FT -> noise -> tau -> peaks -> windows -> fit -> timebase -> review, then
optionally the report), with live per-stage progress. Orchestration only; the
real logic lives in :func:`run_pipeline_impl`.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

from .._internal.run_impl import run_pipeline_impl
from ..core.noise_settings import NoiseSettings
from ..core.peak_detection_settings import PeakDetectionSettings
from ..core.settings import FTSettings, _parse_trim
from ..core.stage_fit_settings import StageFitSettings
from ..core.start_detection_settings import StartDetectionSettings
from ..core.tau_calibration_settings import TauCalibrationSettings
from ..core.window_planning_settings import WindowPlanningSettings
from ._argspec import (
    add_settings_args,
    add_start_detection_args,
    settings_from_namespace,
    start_settings_from_namespace,
)


def _parse_clocks(spec: Optional[str]) -> Optional[List[dict]]:
    """Parse a ``--clocks`` spec into clock-source dicts.

    Comma-separated fundamentals in MHz; append ``:u`` (or ``:unlocked``) to mark
    one free-running (the digitizer). ``"5120,5760"`` -> two locked sources;
    ``"5120,5760,6250:u"`` -> the 6250 source unlocked. Returns ``None`` for an
    empty spec (auto-resolution then applies).
    """
    if not spec:
        return None
    out: List[dict] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        freq_s, _, flag = tok.partition(":")
        locked = flag.strip().lower() not in {"u", "unlocked", "free", "0", "false"}
        out.append({"freq_mhz": float(freq_s), "locked": locked})
    return out or None


def _start_detection_params_from_namespace(
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    """Build ``start_detection_params`` from the parsed ``--start.*`` flags.

    ``StartDetectionSettings`` is a frozen dataclass with concrete hard
    defaults (not the ``Optional``-everywhere resolution-chain pattern the
    other stages use), so only the user-specified fields are passed through;
    the dataclass defaults fill the rest. ``None`` when no ``--start.*`` flag
    was given, so behavior is unchanged from a bare ``run``.
    """
    overrides = start_settings_from_namespace(args, prefix="start")
    if not overrides:
        return None
    return {"settings": StartDetectionSettings(**overrides)}


def _stage_settings_params(
    args: argparse.Namespace, cls: Any, prefix: str
) -> Optional[Dict[str, Any]]:
    """Reconstruct a stage's sparse settings from its ``--{prefix}.*`` flags.

    Returns ``{"settings": <instance>}`` only when at least one namespaced
    flag was given (``instance.is_empty()`` is False); ``None`` otherwise, so
    a bare ``run`` with no namespaced knobs is unaffected.
    """
    inst = settings_from_namespace(args, cls, prefix=prefix)
    return None if inst.is_empty() else {"settings": inst}


def cmd_run(args: argparse.Namespace) -> int:
    """Run the full pipeline on a raw source."""
    trim = args.trim  # already a (min, max) tuple from _parse_trim, or None

    # FT is special-cased: compute_ft takes the FTSettings fields as flat
    # kwargs (start_us / end_us / units_power), not a `settings=` bundle, so
    # ft_params stays the flat-kwargs-dict shape `run_pipeline_impl` already
    # expects (it setdefaults `trim` onto it).
    ft_overrides = settings_from_namespace(args, FTSettings, prefix="ft").overrides()
    ft_params = ft_overrides or None

    result = run_pipeline_impl(
        args.source,
        output=args.output,
        trim=trim,
        sigma_floor_khz=args.sigma_floor,
        force=args.force,
        format_name=args.format_name,
        fid_index=args.fid_index,
        detect_start=args.detect_start,
        calibrate=args.calibrate,
        clocks=_parse_clocks(args.clocks),
        report=args.report,
        report_output_dir=args.report_dir,
        preset=args.preset,
        progress=not args.quiet,
        start_detection_params=_start_detection_params_from_namespace(args),
        ft_params=ft_params,
        noise_params=_stage_settings_params(args, NoiseSettings, "noise"),
        tau_params=_stage_settings_params(args, TauCalibrationSettings, "tau"),
        peak_params=_stage_settings_params(args, PeakDetectionSettings, "peaks"),
        window_params=_stage_settings_params(args, WindowPlanningSettings, "windows"),
        fit_params=_stage_settings_params(args, StageFitSettings, "fit"),
    )

    if result["status"] == "error":
        print(
            f"run: failed at stage '{result['failed_stage']}': {result['error']}\n"
            f"     completed: {', '.join(result['completed_stages']) or '(none)'}"
        )
        return 1

    summary = (
        f"run: {result['pipeline_file']} — "
        f"{len(result['completed_stages'])} stages in {result['elapsed_s']:.1f}s "
        f"(timebase {result['timebase']})"
    )
    if result.get("report"):
        rep = result["report"]
        parts = [p for p in (rep.get("table"), rep.get("html")) if p]
        summary += "; report: " + ", ".join(parts)
    print(summary)
    return 0


def register_run_command(subparsers: Any) -> None:
    """Register the bare ``run`` end-to-end command."""
    p = subparsers.add_parser(
        "run",
        help="Run the full pipeline on a raw source (import through review)",
        description=(
            "Drive a raw data source through every pipeline stage in sequence:\n"
            "import -> FT -> noise -> tau -> peaks -> windows -> fit -> timebase\n"
            "-> review, with live per-stage progress. With --report it also emits\n"
            "the Level-1 table + Level-3 HTML report.\n\n"
            "--trim (the active-band FT range) is required. Tau calibration and\n"
            "start detection run by default; timebase calibration runs by default\n"
            "but is non-fatal — it auto-resolves the instrument clocks (e.g. from\n"
            "Blackchirp clocks.csv) and warns + skips when none is declared\n"
            "(--no-cal skips it deliberately). A fresh build by default."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("source", help="Path to the raw data source (file or directory)")
    p.add_argument(
        "--output",
        dest="output",
        default=None,
        metavar="PATH",
        help="Destination .ftmw file (derived from the source name if omitted).",
    )
    p.add_argument(
        "--trim",
        "--ft.trim",
        dest="trim",
        type=_parse_trim,
        required=True,
        metavar="MIN:MAX",
        help=(
            "Active-band FT range in MHz, as MIN:MAX (required). "
            "--ft.trim is an alias for this same flag."
        ),
    )
    p.add_argument(
        "--sigma-floor",
        dest="sigma_floor",
        type=float,
        default=None,
        metavar="KHZ",
        help="Accuracy floor (kHz) folded into the σ_f budget at review.",
    )
    p.add_argument(
        "--preset",
        dest="preset",
        default=None,
        metavar="NAME",
        help="Settings preset forwarded to the stages that accept one.",
    )
    p.add_argument(
        "--report",
        dest="report",
        action="store_true",
        default=False,
        help="Also emit the Level-1 table + Level-3 HTML report at the end.",
    )
    p.add_argument(
        "--report-dir",
        dest="report_dir",
        default=None,
        metavar="DIR",
        help="Directory for the --report artifacts (default <stem>_report/).",
    )
    p.add_argument(
        "--no-start-detect",
        dest="detect_start",
        action="store_false",
        default=True,
        help="Skip start-time detection before the FT.",
    )
    p.add_argument(
        "--no-cal",
        dest="calibrate",
        action="store_false",
        default=True,
        help="Skip timebase calibration (no warning); frequencies stay precision-only.",
    )
    p.add_argument(
        "--clocks",
        dest="clocks",
        default=None,
        metavar="SPEC",
        help=(
            "Explicit clock declaration, MHz fundamentals comma-separated; append "
            "':u' to mark one unlocked (e.g. '5120,5760'). Overrides auto-detection."
        ),
    )
    p.add_argument(
        "--no-force",
        dest="force",
        action="store_false",
        default=True,
        help="Do not overwrite an existing file built from a different source.",
    )
    p.add_argument(
        "--format",
        dest="format_name",
        default=None,
        metavar="NAME",
        help="Input format name (auto-detected if omitted).",
    )
    p.add_argument(
        "--fid-index",
        dest="fid_index",
        type=int,
        default=None,
        metavar="N",
        help="FID index for multi-FID formats (e.g. Blackchirp).",
    )
    p.add_argument(
        "--quiet",
        dest="quiet",
        action="store_true",
        default=False,
        help="Suppress the live per-stage progress display.",
    )

    # Namespaced per-knob passthrough: `--<stage>.<flag>`, generated from each
    # stage's own settings dataclass (the same class its `*_commands.py`
    # subcommand uses), so `run` never drifts from the per-stage CLI surface.
    # These compose with the flags above (explicit --trim etc. still win their
    # own lane) and route into the `*_params` override dicts `run_pipeline_impl`
    # already accepts. Only start/ft/noise/tau/peaks/windows/fit have a knob
    # surface today -- timebase/review/report have no settings-dataclass/CLI
    # knob surface to mirror (see dev-docs/planning/pipeline-run.md).
    grp_start = p.add_argument_group("stage knobs: start")
    add_start_detection_args(grp_start, prefix="start")

    # `trim` stays excluded from the generated `--ft.*` flags because
    # `--ft.trim` is registered above as an explicit second option string on
    # the canonical `--trim` flag (a true alias, same dest/value) rather than
    # a separate namespaced flag; keeping the exclusion here is what avoids
    # argparse raising a duplicate-option error over `--ft.trim`.
    grp_ft = p.add_argument_group("stage knobs: ft")
    add_settings_args(grp_ft, FTSettings, prefix="ft", exclude={"trim"})

    grp_noise = p.add_argument_group("stage knobs: noise")
    add_settings_args(grp_noise, NoiseSettings, prefix="noise")

    grp_tau = p.add_argument_group("stage knobs: tau")
    add_settings_args(grp_tau, TauCalibrationSettings, prefix="tau")

    grp_peaks = p.add_argument_group("stage knobs: peaks")
    add_settings_args(grp_peaks, PeakDetectionSettings, prefix="peaks")

    grp_windows = p.add_argument_group("stage knobs: windows")
    add_settings_args(grp_windows, WindowPlanningSettings, prefix="windows")

    grp_fit = p.add_argument_group("stage knobs: fit")
    add_settings_args(grp_fit, StageFitSettings, prefix="fit")

    p.set_defaults(func=cmd_run)
