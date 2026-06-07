"""Rebuild the 2638 fixture this audit runs against, through window assignment.

The audit examines Stage 4 window planning on the canonical unapodized FT after
the Stage 2/3 rework. It needs a fixture carried through window assignment:
import -> detect_start_time -> compute_ft (unapodized, native-length) ->
estimate_noise (scatter) -> calibrate_tau -> detect_peaks -> assign_windows.

The diagnostic drivers the report cites (compare_noise, check_edge_threshold,
inspect_outlier_windows, check_doublet_recoupling) were one-off scratch probes;
their findings are recorded in the report. This script regenerates only the
fixture they read, into the gitignored scratch tree.

    conda run -n ftmwpipeline-dev python dev-docs/research/stage4-poststage23-audit/build_fixture.py
"""

from __future__ import annotations

import ftmwpipeline.api as ftmw

FIXTURE = "scratch/stage4-audit/exp_2638.ftmw"
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
    return FIXTURE


if __name__ == "__main__":
    import pathlib

    pathlib.Path(FIXTURE).parent.mkdir(parents=True, exist_ok=True)
    print("built", build())
