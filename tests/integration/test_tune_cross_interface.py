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


def test_tune_list_include_advanced_parity():
    # include_advanced and the selector behave identically across the two surfaces
    a_all = ftmw.tune_list(include_advanced=True)
    p_all = Pipeline.tune_list(include_advanced=True)
    assert [k.path for k in a_all] == [k.path for k in p_all]
    # advanced reveals strictly more than the default view
    assert len(a_all) > len(ftmw.tune_list())
    a_sel = ftmw.tune_list("stage2b", include_advanced=True)
    p_sel = Pipeline.tune_list("stage2b", include_advanced=True)
    assert [k.path for k in a_sel] == [k.path for k in p_sel]
    assert a_sel and all(k.path.startswith("stage2b.") for k in a_sel)


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


def test_api_pipeline_scan_batch_parity(baseline_2638_stage1_raw, tmp_path):
    # the stage2.scatter sub-block (all primary, Stage-1-only deps) batch-scans
    a = ftmw.tune_scan_batch(
        baseline_2638_stage1_raw, "stage2.scatter",
        output_dir=tmp_path / "api", make_plot=False, quiet=True,
    )
    p = Pipeline.open(baseline_2638_stage1_raw).tune_scan_batch(
        "stage2.scatter", output_dir=tmp_path / "pipe", make_plot=False, quiet=True,
    )
    assert [it.knob for it in a] == [it.knob for it in p]
    assert len(a) == 3 and all(it.ok for it in a)
    assert [_rows(it.result) for it in a] == [_rows(it.result) for it in p]


def test_scan_batch_isolates_failures(baseline_2638_stage1_raw, tmp_path):
    # stage2b.stft knobs drive calibrate_tau, which needs Stage 2 (absent on the
    # Stage-0/1 baseline) -> every knob fails, but the batch never aborts: it
    # returns one BatchItem per matched knob, each carrying its captured error.
    items = ftmw.tune_scan_batch(
        baseline_2638_stage1_raw, "stage2b.stft", include_advanced=True,
        output_dir=tmp_path / "b", make_plot=False, quiet=True,
    )
    assert items and all(not it.ok and it.error for it in items)


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
    # single header row, full stage-leading path, and an elided continuation
    assert "knob" in out and "tier" in out
    assert "stage1.start_us" in out  # first row prints its full path
    assert ".window_mhz" in out      # a later sibling renders elided


def test_elide_path_blanks_shared_prefix():
    from ftmwpipeline.cli.tune_commands import _elide_path

    assert _elide_path("stage2.group1.setting1", None) == "stage2.group1.setting1"
    # shared "stage2.group1" blanked to equal-width padding, ".setting2" aligned
    assert (
        _elide_path("stage2.group1.setting2", "stage2.group1.setting1")
        == " " * len("stage2.group1") + ".setting2"
    )
    # only "stage2" shared -> ".group2.setting1" prints from the first difference
    assert (
        _elide_path("stage2.group2.setting1", "stage2.group1.setting2")
        == " " * len("stage2") + ".group2.setting1"
    )
    # elided cell keeps the original length so downstream columns stay aligned
    assert len(_elide_path("stage2.group1.setting2", "stage2.group1.setting1")) == len(
        "stage2.group1.setting2"
    )


def test_input_file_not_mutated_by_scan(baseline_2638_stage1_raw, tmp_path):
    import shutil

    work = tmp_path / "copy.ftmw"
    shutil.copy(baseline_2638_stage1_raw, work)
    before = work.stat().st_size
    ftmw.tune_scan(work, KNOB, grid=GRID, output_dir=tmp_path / "o",
                   make_plot=False, quiet=True)
    assert work.stat().st_size == before
