#!/usr/bin/env python3
"""Process one FTMW experiment end to end with the functional API.

This mirrors the Quickstart walkthrough in code: it drives the bundled
Blackchirp experiment ``2638`` from its raw free-induction decay through every
stage — import, Fourier transform, noise, decay-time calibration, peak
detection, window assignment, and fitting — then prints a short summary of the
result.

Run it from the repository root::

    python examples/basic_usage.py            # writes to a temporary directory
    python examples/basic_usage.py out.ftmw   # writes a named .ftmw you can keep

The same sequence is available through the ``Pipeline`` class and the
``ftmwpipeline`` command line; see the documentation for those interfaces.
"""

import sys
import tempfile
from pathlib import Path

import ftmwpipeline.api as ftmw
from ftmwpipeline.workflows import validate_installation

# The example experiment lives beside this script. It is a real 15 µs,
# 750k-point Blackchirp FID (40.96 GHz probe, lower sideband) whose active
# spectral band is 26500–40000 MHz — the trim the canonical transform needs.
EXAMPLE_DATA = Path(__file__).parent / "blackchirp_data" / "2638"
ACTIVE_BAND = (26500, 40000)


def main(output: Path) -> None:
    print("=== ftmwpipeline basic usage ===\n")

    # 1. Confirm the installation is functional before doing any real work.
    print("Installation check:")
    for component, ok in validate_installation().items():
        print(f"  {component:18s}: {'OK' if ok else 'FAILED'}")
    print()

    if not EXAMPLE_DATA.exists():
        sys.exit(
            f"Example data not found at {EXAMPLE_DATA}. Run this script from a "
            "source checkout — the example experiment ships with the repository, "
            "not the installed package."
        )

    # 2. Drive the experiment through every stage. Each call persists its result
    #    into the single self-contained .ftmw file and depends on the one before
    #    it; re-running a stage with new parameters is safe.
    print(f"Processing {EXAMPLE_DATA}\n -> {output}\n")

    ftmw.import_data(output, source=str(EXAMPLE_DATA))
    ftmw.compute_ft(output, trim=ACTIVE_BAND)
    ftmw.estimate_noise(output)
    ftmw.calibrate_tau(output)
    peaks = ftmw.detect_peaks(output)
    plan = ftmw.assign_windows(output)
    fit = ftmw.fit_peaks(output)

    # 3. Report what came out.
    print("Result:")
    print(f"  peaks detected : {len(peaks)}")
    print(f"  fit windows    : {plan.n_windows}")
    print(f"  lines fitted   : {fit.n_fitted_peaks}")
    print()
    print(
        "Inspect it with:\n"
        f"  ftmwpipeline info {output}\n"
        f"  ftmwpipeline fit show {output}\n"
    )
    print("=== done ===")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(Path(sys.argv[1]))
    else:
        with tempfile.TemporaryDirectory(prefix="ftmw_example_") as tmp:
            main(Path(tmp) / "exp_2638.ftmw")
