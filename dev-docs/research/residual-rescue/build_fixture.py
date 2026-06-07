"""Rebuild the 2638 fixture the residual-rescue study runs against, through fit.

The rescue chain operates on a fully fitted experiment: import ->
detect_start_time -> compute_ft (unapodized, native-length) -> estimate_noise
(scatter) -> calibrate_tau -> detect_peaks -> assign_windows -> fit_peaks. The
per-window rollups the report references (the scratch/stage5-validation tree)
are produced by the validation harness reading this fixture; this script
regenerates only the fixture, into the gitignored scratch tree.

    conda run -n ftmwpipeline-dev python dev-docs/research/residual-rescue/build_fixture.py
"""

from __future__ import annotations

import ftmwpipeline.api as ftmw

FIXTURE = "scratch/stage5-validation/exp_2638.ftmw"
SOURCE = "examples/blackchirp_data/2638"
TRIM = (26500.0, 40000.0)


def build() -> str:
    ftmw.import_data(FIXTURE, source=SOURCE, force=True)
    ftmw.detect_start_time(FIXTURE, band=TRIM, stamp=True)
    ftmw.compute_ft(FIXTURE, trim=TRIM)
    ftmw.estimate_noise(FIXTURE)
    ftmw.calibrate_tau(FIXTURE)
    ftmw.detect_peaks(FIXTURE)
    ftmw.assign_windows(FIXTURE)
    ftmw.fit_peaks(FIXTURE)
    return FIXTURE


if __name__ == "__main__":
    import pathlib

    pathlib.Path(FIXTURE).parent.mkdir(parents=True, exist_ok=True)
    print("built", build())
