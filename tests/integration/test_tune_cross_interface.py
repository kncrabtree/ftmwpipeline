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
KNOB = "stage2.window_mhz"
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
    # the stage2 knobs (all primary, Stage-1-only deps) batch-scans
    a = ftmw.tune_scan_batch(
        baseline_2638_stage1_raw, "stage2",
        output_dir=tmp_path / "api", make_plot=False, quiet=True,
    )
    p = Pipeline.open(baseline_2638_stage1_raw).tune_scan_batch(
        "stage2", output_dir=tmp_path / "pipe", make_plot=False, quiet=True,
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
    assert "stage0.guard_margin_us" in out  # first row prints its full path
    assert ".window_mhz" in out             # a later sibling renders elided


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


def test_stage3_scan_runs_and_plots(baseline_2638_stage2, tmp_path):
    # A real Stage 3 sweep on the production Stage-2 baseline: a row per grid
    # value with the by-SNR-band passed-peak columns, plus the spectrum plot.
    r = ftmw.tune_scan(
        baseline_2638_stage2, "stage3.promotion.min_snr", grid=[3.0, 5.0],
        output_dir=tmp_path, quiet=True,
    )
    assert [row.value for row in r.rows] == [3.0, 5.0]
    assert r.metric_columns == (
        "n_total", "n_strong", "n_medium", "n_weak",
        "snr_min", "snr_p10", "snr_p25", "snr_p50", "snr_p90", "snr_max",
    )
    # raising the promotion floor cannot pass more peaks to Stage 4
    totals = [row.metrics["n_total"] for row in r.rows]
    assert totals[0] >= totals[1]
    # the lowest passed SNR tracks the promotion cutoff
    for row in r.rows:
        assert row.metrics["snr_min"] >= row.value - 1e-6
    assert r.plot_path is not None and r.plot_path.exists()


def test_stage4_scan_runs_and_plots(baseline_2638_stage3, tmp_path):
    # A real Stage 4 sweep on the Stage-3 baseline: a row per grid value with the
    # plan-shape columns, plus the boundary-overlay plot. Raising the coherence
    # cutoff relaxes leakage flagging, so the HARD-window count cannot rise.
    r = ftmw.tune_scan(
        baseline_2638_stage3, "stage4.coherence.edge_threshold",
        grid=[6.0, 8.0, 10.0], output_dir=tmp_path, quiet=True,
    )
    assert [row.value for row in r.rows] == [6.0, 8.0, 10.0]
    assert r.metric_columns == (
        "n_windows", "n_hard", "n_easy", "n_free", "n_fixed", "n_dep",
        "n_split", "width_p50", "width_p95", "width_max",
    )
    hard = [row.metrics["n_hard"] for row in r.rows]
    assert hard == sorted(hard, reverse=True)
    assert all(row.metrics["n_windows"] > 0 for row in r.rows)
    assert r.plot_path is not None and r.plot_path.exists()


def test_parse_zoom_valid_and_invalid():
    from ftmwpipeline.cli.tune_commands import _parse_zoom

    assert _parse_zoom("35000-35800,38400-38500") == [
        (35000.0, 35800.0), (38400.0, 38500.0),
    ]
    assert _parse_zoom(" 100-200 ") == [(100.0, 200.0)]
    assert _parse_zoom("100-200,,") == [(100.0, 200.0)]  # blanks skipped
    for bad in ("100", "200-100", "abc-200", "100-"):
        with pytest.raises(ValueError):
            _parse_zoom(bad)


def test_cli_scan_explicit_zoom_renders(baseline_2638_stage2, tmp_path):
    # --zoom flows CLI -> engine -> adapter and pins the requested windows; the
    # plot still renders (a real Stage 3 sweep on the production baseline).
    rc = cli_main([
        "tune", "scan", str(baseline_2638_stage2),
        "--knob", "stage3.promotion.min_snr", "--grid", "3,5",
        "--zoom", "35000-35800,38400-38500",
        "--output-dir", str(tmp_path),
    ])
    assert rc == 0
    assert list(tmp_path.glob("tune_*.png")), "expected a rendered plot"


def test_api_scan_accepts_zoom_count_width(baseline_2638_stage2, tmp_path):
    # the auto-selector count/width override is accepted on the functional surface
    # and produces a plot without error.
    r = ftmw.tune_scan(
        baseline_2638_stage2, "stage3.promotion.min_snr", grid=[3.0, 5.0],
        output_dir=tmp_path, quiet=True, n_zoom=4, zoom_width_mhz=200.0,
    )
    assert r.plot_path is not None and r.plot_path.exists()


def test_reduce_plan_for_fit_subsets_windows(baseline_2638_stage4, tmp_path):
    # the window-selection prepare hook trims the plan to the requested budget
    # (top-SNR + sample, dependency-closed) without any fitting.
    import shutil

    from ftmwpipeline._internal.tuning.fit_support import (
        FitWindowSelection, reduce_plan_for_fit,
    )

    fp = tmp_path / "reduce.ftmw"
    shutil.copy(baseline_2638_stage4, fp)
    before = ftmw.load_windows(fp).n_windows
    reduce_plan_for_fit(fp, FitWindowSelection(top_snr=3, sample=10))
    after = ftmw.load_windows(fp).n_windows
    assert 3 <= after < before  # reduced, but kept at least the top-SNR windows


def test_stage5_fit_scan_runs_and_plots(baseline_2638_stage4_small, tmp_path):
    # A real Stage 5 fit sweep on the small (dependency-free) windows fixture:
    # the floor-aware eps/pass columns plus the fit-quality plot.
    r = ftmw.tune_scan(
        baseline_2638_stage4_small, "stage5.baseline.edge_threshold",
        grid=[2.5, 8.0], output_dir=tmp_path, quiet=True, fit_all=True,
    )
    assert [row.value for row in r.rows] == [2.5, 8.0]
    assert r.metric_columns == (
        "eps_p50", "eps_p95", "n_fail", "n_peaks", "n_free_tau",
        "sigma_f_khz", "chi2r_p50", "chi2r_p95",
    )
    assert all(row.metrics["n_peaks"] > 0 for row in r.rows)
    assert r.plot_path is not None and r.plot_path.exists()


def test_input_file_not_mutated_by_scan(baseline_2638_stage1_raw, tmp_path):
    import shutil

    work = tmp_path / "copy.ftmw"
    shutil.copy(baseline_2638_stage1_raw, work)
    before = work.stat().st_size
    ftmw.tune_scan(work, KNOB, grid=GRID, output_dir=tmp_path / "o",
                   make_plot=False, quiet=True)
    assert work.stat().st_size == before
