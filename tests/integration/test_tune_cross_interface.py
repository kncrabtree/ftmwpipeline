"""Cross-interface consistency for the companion `tune` surface.

The CLI (`tune list`/`tune scan`), the `Pipeline` methods, and the functional
`api` functions are thin wrappers over one engine; they must agree. Uses the
session-scoped raw Stage 0+1 baseline and a fast, stable noise knob so no extra
build cost is incurred.
"""

from pathlib import Path

import pytest

import ftmwpipeline.api as ftmw
from ftmwpipeline.cli.main import main as cli_main
from ftmwpipeline.pipeline import Pipeline

pytestmark = pytest.mark.integration

# A cheap, stable knob: re-runs only Stage 2 noise on the Stage 1 baseline.
KNOB = "stage2.scatter.window_mhz"
LEAF = "window_mhz"
GRID = [40.0, 80.0]


def _rows(result):
    return [(r.value, r.metrics) for r in result.rows]


def test_tune_list_parity():
    a = ftmw.tune_list()
    p = Pipeline.tune_list()
    assert [k.path for k in a] == [k.path for k in p]
    assert len(a) > 0


def test_api_pipeline_scan_parity(baseline_2638_stage1_raw, tmp_path):
    ra = ftmw.tune_scan(
        baseline_2638_stage1_raw, KNOB, grid=GRID,
        output_dir=tmp_path / "api", make_plot=False, quiet=True,
    )
    rp = Pipeline.open(baseline_2638_stage1_raw).tune_scan(
        KNOB, grid=GRID, output_dir=tmp_path / "pipe", make_plot=False, quiet=True,
    )
    assert _rows(ra) == _rows(rp)
    assert ra.metric_columns == rp.metric_columns


def test_cli_scan_matches_api(baseline_2638_stage1_raw, tmp_path, capsys):
    ra = ftmw.tune_scan(
        baseline_2638_stage1_raw, KNOB, grid=GRID,
        output_dir=tmp_path / "api", make_plot=False, quiet=True,
    )
    capsys.readouterr()  # clear

    rc = cli_main([
        "tune", "scan", str(baseline_2638_stage1_raw),
        "--knob", KNOB,
        "--grid", ",".join(str(g) for g in GRID),
        "--output-dir", str(tmp_path / "cli"),
        "--no-plot",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    assert LEAF in out
    for col in ra.metric_columns:
        assert col in out
    # every swept metric value from the api result appears in the CLI table
    for row in ra.rows:
        assert f"{row.metrics['median_sigma']:.6g}" in out


def test_cli_list_runs(capsys):
    rc = cli_main(["tune", "list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "stage2.scatter.window_mhz" in out


def test_input_file_not_mutated_by_scan(baseline_2638_stage1_raw, tmp_path):
    import shutil

    work = tmp_path / "copy.ftmw"
    shutil.copy(baseline_2638_stage1_raw, work)
    before = work.stat().st_size
    ftmw.tune_scan(work, KNOB, grid=GRID, output_dir=tmp_path / "o",
                   make_plot=False, quiet=True)
    assert work.stat().st_size == before
