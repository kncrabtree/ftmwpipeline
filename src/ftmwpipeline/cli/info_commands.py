"""
Pipeline-file information command.

`info <file.ftmw>` reports provenance, validity, and stage status for a
pipeline file. `--format json` emits a machine-readable object (and nothing
else on stdout) for scripting and integration.
"""

import argparse
import json
from typing import Any

from ..api import get_pipeline_info
from .utils import print_error, setup_logging


def cmd_info(args: argparse.Namespace) -> int:
    """Show information about a .ftmw pipeline file."""
    setup_logging(getattr(args, "verbose", False))

    file_path = args.file_path
    if not file_path.endswith(".ftmw"):
        file_path = file_path + ".ftmw"

    info = get_pipeline_info(file_path)

    if args.format == "json":
        print(json.dumps(info, indent=2, default=str))
        return 0 if info.get("valid") else 1

    if not info.get("valid", False):
        print_error(
            f"Pipeline file invalid or unreadable: {file_path}\n"
            f"  {info.get('error') or info.get('errors')}"
        )
        return 1

    print(f"Pipeline: {file_path}")
    print(f"  source:          {info.get('source_path')}")
    print(f"  format:          {info.get('format')}")
    print(f"  file format:     {info.get('format_version') or '(legacy, unstamped)'}")
    print(f"  created with:    {info.get('created_with') or '(unknown)'}")
    print(f"  imported:        {info.get('import_time')}")
    print(
        f"  completed:       {', '.join(info.get('completed_stages', [])) or '(none)'}"
    )
    print(
        f"  next available:  {', '.join(info.get('next_available_stages', [])) or '(none)'}"
    )
    _print_environment(info)
    if info.get("warnings"):
        print(f"  warnings:        {info['warnings']}")
    return 0


def _print_environment(info: dict) -> None:
    """Print the analysis-environment record: what produced each stage.

    Printed per stage rather than once for the file, because the fact worth
    surfacing is whether the artifacts agree -- a file whose Stage 5 fit and
    Stage 6 curation came from different code is the case a single stamp
    cannot express.
    """
    from ..core.environment import EnvironmentRecord

    last = info.get("last_written_with")
    if last:
        print(f"  last written by: {EnvironmentRecord.from_dict(last).summary()}")

    envs = info.get("stage_environments") or {}
    if not envs:
        print("  environment:     (not recorded; file predates the stamp)")
        current = info.get("current_environment")
        if current:
            print(
                "      running now: "
                f"{EnvironmentRecord.from_dict(current).summary()} -- "
                "reproducibility against the original run cannot be verified."
            )
        return

    epochs = {d.get("analysis_epoch") for d in envs.values() if d.get("analysis_epoch")}
    versions = {d.get("ftmwpipeline") for d in envs.values() if d.get("ftmwpipeline")}
    if len(epochs) <= 1 and len(versions) <= 1:
        blas = next(
            (d.get("blas") for d in envs.values() if d.get("blas")), "(unknown)"
        )
        print(f"  environment:     uniform across {len(envs)} stage(s); BLAS {blas}")
    else:
        print("  environment:     MIXED -- stages came from different versions:")
        for stage in sorted(envs):
            rec = EnvironmentRecord.from_dict(envs[stage])
            print(f"      {stage:<28} {rec.summary()}")

    drift = info.get("environment_drift") or []
    for line in drift:
        print(f"      drift: {line}")
    for line in info.get("runtime_environment_drift") or []:
        print(f"      vs running environment: {line}")
    if info.get("environment_acknowledged"):
        print(
            "      note: an analysis-epoch mismatch was acknowledged; curation "
            "crossed an epoch boundary."
        )


def add_info_subcommand(subparsers: Any) -> None:
    """Register the `info` command."""
    parser = subparsers.add_parser(
        "info",
        help="Show provenance and stage status for a .ftmw pipeline file",
        description="Report information about a pipeline file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ftmwpipeline info exp_2638.ftmw
  ftmwpipeline info exp_2638.ftmw --format json
        """,
    )
    parser.add_argument("file_path", help="Path to .ftmw pipeline file")
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.set_defaults(func=cmd_info)
