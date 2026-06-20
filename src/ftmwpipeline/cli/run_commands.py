"""CLI ``run`` command: drive a raw source through the whole pipeline.

A single bare verb that imports a raw source and runs every stage in sequence
(FT -> noise -> tau -> peaks -> windows -> fit -> timebase -> review, then
optionally the report), with live per-stage progress. Orchestration only; the
real logic lives in :func:`run_pipeline_impl`.
"""

from __future__ import annotations

import argparse
from typing import Any, List, Optional

from .._internal.run_impl import run_pipeline_impl
from ..core.settings import _parse_trim


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


def cmd_run(args: argparse.Namespace) -> int:
    """Run the full pipeline on a raw source."""
    trim = args.trim  # already a (min, max) tuple from _parse_trim, or None

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
        dest="trim",
        type=_parse_trim,
        required=True,
        metavar="MIN:MAX",
        help="Active-band FT range in MHz, as MIN:MAX (required).",
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
    p.set_defaults(func=cmd_run)
