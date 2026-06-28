"""Build a fixture end-to-end from checked-in example data.

A development utility: drives Stages 0-6 (optionally through timebase calibration
and the Stage 6 review pass) on one of the `examples/blackchirp_data/<name>`
experiments, producing a fresh `.ftmw` for validation and benchmarking. Always
builds fresh from the raw data -- never patches an existing file -- because a
persisted setting in an old `.ftmw` outranks the code default and would silently
exercise stale behavior.

Usage:
    python scripts/development/build_fixture.py 655
    python scripts/development/build_fixture.py 2638 --output scratch/2638.ftmw
    python scripts/development/build_fixture.py 1512 --no-review

Pass no explicit line shape: the Stage 2b vote stamps the recommended shape.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import ftmwpipeline.api as ftmw

# The home-instrument analysis band shared by the Blackchirp example fixtures.
DEFAULT_TRIM = (26500.0, 40000.0)


def build_fixture(
    name: str,
    output: str | None = None,
    trim: tuple[float, float] = DEFAULT_TRIM,
    calibrate: bool = True,
    review: bool = True,
) -> str:
    """Build ``examples/blackchirp_data/<name>`` end-to-end; return the path."""
    src = f"examples/blackchirp_data/{name}"
    out = output or f"scratch/{name}.ftmw"
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    print(f"[{time.strftime('%H:%M:%S')}] build {name} -> {out}", flush=True)
    res = ftmw.run_pipeline(
        src, output=out, trim=trim, force=True, calibrate=calibrate, progress=False
    )
    if res["status"] != "success":
        raise SystemExit(
            f"build failed at {res.get('failed_stage')}: {res.get('error')}"
        )
    if review:
        ftmw.review_run(out)
    dt = time.monotonic() - t0
    fp = ftmw.get_final_products(out)
    n = len(fp.peaks) if fp is not None else 0
    print(
        f"[{time.strftime('%H:%M:%S')}] done {name}: {n} final lines, "
        f"{len(res['completed_stages'])} stages, {dt:.0f}s -> {out}",
        flush=True,
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name", help="fixture name under examples/blackchirp_data/")
    ap.add_argument("--output", help="output .ftmw path (default scratch/<name>.ftmw)")
    ap.add_argument(
        "--trim",
        nargs=2,
        type=float,
        metavar=("LO", "HI"),
        default=DEFAULT_TRIM,
        help="analysis band in MHz (default 26500 40000)",
    )
    ap.add_argument(
        "--no-calibrate",
        action="store_true",
        help="skip timebase calibration (the final-products eps correction)",
    )
    ap.add_argument(
        "--no-review", action="store_true", help="skip the Stage 6 review pass"
    )
    args = ap.parse_args()
    build_fixture(
        args.name,
        output=args.output,
        trim=(args.trim[0], args.trim[1]),
        calibrate=not args.no_calibrate,
        review=not args.no_review,
    )


if __name__ == "__main__":
    main()
